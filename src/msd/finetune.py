"""Fine-tune A0 into A1 with AlignTune: LoRA on a harmless closed-form task, then merge to fp16 (plan Task 14).

    python scripts/finetune.py --config configs/ft_lora.yaml --dry-run
    python scripts/finetune.py --config configs/ft_lora.yaml --max-steps 20 --merge     # smoke run, then
    python scripts/finetune.py --config configs/ft_lora.yaml                             # the real run
    python scripts/finetune.py --config configs/ft_lora_strong.yaml                     # the strong retry

Every training value (model, data paths, LoRA, quantisation, learning rate, epochs, batches, sequence length,
precision, seed, loss masking, saving) is in the YAML config; this program holds no default for any of them and
refuses unknown or out-of-range keys. `--data` replaces the training file, `--out` the output folder (default
`results/finetune/<config stem>`, and `<config stem>-smoke` with `--max-steps`), `--seed` the seed.

Before anything is trained the training data goes through the in-process validator of scripts/validate_ft_data.py
(`msd.ftdata`): schema, bare answers, no refusals, language coverage, the harmless-subject allowlist, and no
overlap with heldout, taskdev or the safety test set (each checked when its file exists, and said to be NOT
checked otherwise). The sha256 of the training file goes into the summary. Every example is also measured with
the model's chat template and the run is refused if any is longer than `train.max_seq_length`.

`--dry-run` does all of that, prints the plan (rows per language, steps = rows x epochs / effective batch,
token lengths and the chat-template check when a tokenizer can be loaded) and writes nothing; it loads no model
and needs no GPU and no AlignTune. It does import transformers (the default tokenizer loader) and, for the
template check, trl, each only if installed: without them the plan says what was not checked. A real run
builds the trainer through AlignTune's `create_sft_trainer`, trains, checks that the adapter on disk is the
configured one, that the trainer saw every row and that the loss is finite and fell, merges the adapter into a
fresh fp16 copy of the base model, and writes `finetune_summary.json` (config and data hashes, library versions,
steps, sampled loss history, wall-clock seconds, seed, `smoke`, device, `loss_ok`; never a token or an absolute
path). AlignTune trains into a persistent work folder `work/finetune/<config stem>[-smoke]/`, kept (and named
in the error) on any failure; after training the adapter is copied into `<out>/adapter`, the merged weights go
into `<out>/merged`, each under a hidden partial name that is renamed only when whole, and the summary is
written last. `--max-steps N` is a smoke run: N steps, every step logged, merge only with `--merge`, never
pushed. `--merge-only --adapter PATH` trains nothing: it checks that adapter against the config, merges it into
`<out>/merged` and adds a `merge` section to the summary (the recovery after a failed merge).

`--push-to ORG/NAME` uploads the merged folder after a successful full run, but only to a repo that already
exists and for which `HfApi.repo_info(...).private is True` (checked before training starts). The token is read
from HF_TOKEN only (there is no token flag), is never printed, and no repo is ever created. Without the flag
the Hub is not contacted. A gated base model needs the same token in the environment to download.

The call into AlignTune and the merge were written from the source of aligntune 0.1.31 (docs/toolkit-notes.md,
"AlignTune SFT (read from source)") and have not been run against it. UNVERIFIED until the first GPU-runtime
smoke run (docs/runtime.md, section 5): the keyword names, the CuratorKIT data path, TRL's assistant-only mask
for Tiny Aya's template, fp16 stability on a T4 and the memory use.

Exit codes:
  0  done: a full or smoke run finished and passed its checks, or a dry run found nothing wrong
  1  a data or config problem, a failed quality check (validation, overlap, a loss that is NaN or did not fall,
     rows missing, an adapter that is not the configured one), a failure while training, merging or uploading,
     or a Hub repo that is missing or not private
  2  usage error: bad or missing arguments, --push-to without HF_TOKEN or with --max-steps or --no-merge,
     a non-empty --out, or an --out that is, or is inside, a data folder
"""

from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from msd import ftdata
from msd.atomicdir import atomic_output_dir, check_out_dir, note_stale_partials
from msd.config import LANGS
from msd.restarun import scrub

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2

SUMMARY_FILE = "finetune_summary.json"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
SMOKE_SUFFIX = "-smoke"
TEMPLATE_FILE = "default_chat_template.jinja"  # not chat_template.jinja: the trainer saves its own under that name
SMOKE_MARKER = "SMOKE.txt"
NOT_PUBLISHED = ("lexsi_provenance.json",)  # AlignTune's provenance file records absolute paths
DEFAULT_TEMPLATE = "default"
IGNORE_INDEX = -100  # PyTorch's cross-entropy ignore index: a label with this value takes no part in the loss
ALIGNTUNE_VERSION = "0.1.31"
LIBRARIES = (
    "aligntune", "trl", "transformers", "peft", "torch", "bitsandbytes", "datasets", "accelerate", "curatorkit", "huggingface_hub",
)
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_FLOAT = ".8g"
_ONE_DECIMAL = ".1f"
NOTE = (
    "UNVERIFIED: the call into AlignTune and the merge were written from the source of aligntune "
    f"{ALIGNTUNE_VERSION} and had not been run when this was written; they count as checked only after the first "
    "GPU-runtime smoke run (docs/runtime.md, section 5)."
)


class FinetuneError(RuntimeError):
    """A refusal or failure with a message meant for the person running the script."""


class ConfigError(FinetuneError):
    """The config file is missing, is not valid YAML, or holds a wrong, missing or unknown key."""


# --- the schema -------------------------------------------------------------------------------------------------
# A rule takes a value and returns None when it is fine, or the rest of an error sentence when it is not.

Rule = Callable[[Any], "str | None"]

_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def integer(minimum: int) -> Rule:
    def check(value: Any) -> str | None:
        if not _is_int(value):
            return f"must be an integer, got {value!r}"
        if value < minimum:
            return f"must be at least {minimum}, got {value}"
        return None

    return check


def number(*, above=None, at_least=None, below=None, at_most=None) -> Rule:
    def check(value: Any) -> str | None:
        if isinstance(value, str):
            return f"must be a number, got the text {value!r} (PyYAML reads 1e-4 as text: write 1.0e-4)"
        if not _is_number(value):
            return f"must be a number, got {value!r}"
        if not math.isfinite(value):
            return f"must be a finite number, got {value!r}"
        if above is not None and not value > above:
            return f"must be above {above}, got {value}"
        if at_least is not None and not value >= at_least:
            return f"must be at least {at_least}, got {value}"
        if below is not None and not value < below:
            return f"must be below {below}, got {value}"
        if at_most is not None and not value <= at_most:
            return f"must be at most {at_most}, got {value}"
        return None

    return check


def flag(value: Any) -> str | None:
    return None if isinstance(value, bool) else f"must be true or false, got {value!r}"


def text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"must be a non-empty string, got {value!r}"
    return None


def choice(*options: str) -> Rule:
    def check(value: Any) -> str | None:
        if isinstance(value, str) and value in options:
            return None
        hint = ' (YAML reads an unquoted no as false: write "no" in quotes)' if value is False else ""
        return f"must be one of {', '.join(options)}, got {value!r}{hint}"

    return check


def text_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return f"must be a non-empty list of names, got {value!r}"
    if not all(isinstance(v, str) and v.strip() for v in value):
        return f"must hold only non-empty strings, got {value!r}"
    if len(set(value)) != len(value):
        return f"must not repeat a name, got {value!r}"
    return None


def language_list(value: Any) -> str | None:
    problem = text_list(value)
    if problem:
        return problem
    unknown = [v for v in value if v not in LANGS]
    return f"may only name study languages ({', '.join(LANGS)}), got {unknown!r}" if unknown else None


def folder_name(value: Any) -> str | None:
    if not isinstance(value, str) or not _FOLDER_NAME.fullmatch(value) or value.endswith("."):
        return f"must be a plain folder name (letters, digits, dot, dash, underscore), got {value!r}"
    return None


def device_map(value: Any) -> str | None:
    if isinstance(value, Mapping) and value:
        return None
    if isinstance(value, str) and value.strip():
        return None
    return f"must be a device name such as auto, or a mapping such as {{\"\": 0}}, got {value!r}"


SCHEDULERS = ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup")

SCHEMA: dict[str, Any] = {
    "base_model": text,
    "data": {
        "train": text,
        "heldout": text,
        "taskdev": text,
        "pipeline": text,
        "testset": text,
        "langs": language_list,
    },
    "lora": {
        "rank": integer(1),
        "alpha": number(above=0),
        "dropout": number(at_least=0, below=1),
        "target_modules": text_list,
        "bias": choice("none", "all", "lora_only"),
    },
    "quantization": {
        "mode": choice("none", "4bit", "8bit"),
        "quant_type": choice("nf4", "fp4"),
        "double_quant": flag,
    },
    "train": {
        "learning_rate": number(above=0, at_most=1),
        "epochs": integer(1),
        "per_device_batch_size": integer(1),
        "gradient_accumulation_steps": integer(1),
        "max_seq_length": integer(1),
        "warmup_ratio": number(at_least=0, below=1),
        "lr_scheduler": choice(*SCHEDULERS),
        "weight_decay": number(at_least=0),
        "max_grad_norm": number(above=0),
        "gradient_checkpointing": flag,
        "fp16": flag,
        "bf16": flag,
        "seed": integer(0),
        "assistant_only_loss": flag,
        "logging_steps": integer(1),
        "save_steps": integer(1),
        "save_total_limit": integer(1),
    },
    "aligntune": {
        "backend": choice("trl"),
        "device_map": device_map,
        "load_best_model_at_end": flag,
        "eval_strategy": choice("no", "steps", "epoch"),
        "loggers": text_list,
    },
    "output": {
        "adapter_dir": folder_name,
        "merged_dir": folder_name,
        "merge_dtype": choice("float16", "bfloat16", "float32"),
        "keep_checkpoints": flag,
    },
    "checks": {
        "history_points": integer(2),
        "label_check_rows": integer(1),
        "label_tokens_min": integer(1),
        "label_tokens_max": integer(1),
    },
}


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _walk(schema: Mapping[str, Any], data: Any, path: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(data, Mapping):
        errors.append(f"{path or 'the config'} must be a mapping (indented key: value lines), got {data!r}")
        return None
    missing = [k for k in schema if k not in data]
    for key in data:
        if key not in schema:
            near = difflib.get_close_matches(str(key), missing, n=1)
            hint = f" (did you mean '{_join(path, near[0])}'?)" if near else ""
            errors.append(f"unknown key '{_join(path, str(key))}'{hint}")
    errors.extend(f"missing key '{_join(path, key)}'" for key in missing)
    checked: dict[str, Any] = {}
    for key, rule in schema.items():
        if key not in data:
            continue
        if isinstance(rule, Mapping):
            checked[key] = _walk(rule, data[key], _join(path, key), errors)
        else:
            problem = rule(data[key])
            if problem:
                errors.append(f"{_join(path, key)} {problem}")
            else:
                checked[key] = data[key]
    return checked


def _to_namespace(schema: Mapping[str, Any], data: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{k: (_to_namespace(schema[k], v) if isinstance(schema[k], Mapping) else v) for k, v in data.items()})


def parse_config(data: Any, source: str = "config") -> SimpleNamespace:
    """Validate a loaded config (a dict) and return it with attribute access: `cfg.train.learning_rate`.

    Every problem is collected and reported in one `ConfigError`."""
    errors: list[str] = []
    checked = _walk(SCHEMA, data, "", errors)
    if checked is not None and not errors:
        if checked["train"]["fp16"] and checked["train"]["bf16"]:
            errors.append("train.fp16 and train.bf16 cannot both be true")
        if checked["output"]["adapter_dir"] == checked["output"]["merged_dir"]:
            errors.append("output.adapter_dir and output.merged_dir must differ")
        if checked["checks"]["label_tokens_min"] > checked["checks"]["label_tokens_max"]:
            errors.append("checks.label_tokens_min must not be above checks.label_tokens_max")
    if errors:
        raise ConfigError(f"{source}: {len(errors)} problem(s) in the config:\n" + "\n".join(f"  - {e}" for e in errors))
    return _to_namespace(SCHEMA, checked)


def load_config(path: str | Path) -> SimpleNamespace:
    """Read a YAML config file and validate it (`ConfigError` when it is missing, malformed or wrong)."""
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file {path} not found") from None
    except OSError as exc:
        raise ConfigError(f"config file {path} cannot be read: {exc.strerror or exc}") from None
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: not valid YAML ({exc})") from None
    return parse_config(data, source=path.name)


# --- comparing two configs --------------------------------------------------------------------------------------


def _is_section(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(isinstance(k, str) and k for k in value)


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted leaf keys of a nested config; a list, a scalar or an odd mapping such as {"": 0} is a leaf."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = _join(prefix, str(key))
        if _is_section(value):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


STRONG_MAY_DIFFER = ("lora.rank", "lora.alpha", "train.learning_rate", "train.epochs")


def check_strong_relation(base: Mapping[str, Any], strong: Mapping[str, Any]) -> list[str]:
    """What is wrong with `strong` as the strong retry of `base` (both loaded YAML dicts); empty when it is right.

    The strong config must raise the learning rate, the epochs and the LoRA rank, keep the LoRA scale
    alpha / rank, and differ in nothing else."""
    flat_base, flat_strong = flatten(base), flatten(strong)
    problems = []
    for key, word in (("train.learning_rate", "higher"), ("train.epochs", "more"), ("lora.rank", "higher")):
        if not flat_strong[key] > flat_base[key]:
            problems.append(f"{key} must be {word} than the base config's {flat_base[key]!r}, got {flat_strong[key]!r}")
    scale_base = flat_base["lora.alpha"] / flat_base["lora.rank"]
    scale_strong = flat_strong["lora.alpha"] / flat_strong["lora.rank"]
    if not math.isclose(scale_base, scale_strong):
        problems.append(
            f"lora.alpha / lora.rank must stay {scale_base} (only the capacity grows), got {scale_strong}"
        )
    for key in sorted(set(flat_base) | set(flat_strong)):
        if key in STRONG_MAY_DIFFER:
            continue
        if flat_base.get(key) != flat_strong.get(key):
            problems.append(f"{key} differs from the base config ({flat_base.get(key)!r} vs {flat_strong.get(key)!r}): not a documented difference")
    return problems


# --- plan arithmetic and the quality checks ---------------------------------------------------------------------


def plan_steps(rows: int, epochs: int, per_device_batch: int, accumulation: int) -> int:
    """Optimizer steps for `rows` rows: rows x epochs over the effective batch, rounded up."""
    return math.ceil(rows * epochs / (per_device_batch * accumulation))


def loss_check(history: Sequence[Sequence[float]]) -> tuple[bool, str]:
    """(ok, why) for a logged loss history of `[step, loss]` pairs.

    Fails when nothing was logged, when a loss is NaN or infinite, when only one loss was logged (a decrease
    cannot be seen), or when the mean of the second half of the logged losses is not below the mean of the
    first half."""
    values = [float(loss) for _step, loss in history]
    if not values:
        return False, "no loss was logged, so the run cannot be called successful"
    if not all(math.isfinite(v) for v in values):
        return False, "a logged loss is not finite (NaN or inf): fp16 overflow or a diverged run"
    if len(values) < 2:
        return False, "need at least two logged losses to see a decrease, got one (use --max-steps 2 or more)"
    half = len(values) // 2
    first, last = sum(values[:half]) / half, sum(values[half:]) / (len(values) - half)
    if not last < first:
        return False, f"the loss did not decrease (mean of the first half {first:{_FLOAT}}, of the second half {last:{_FLOAT}})"
    return True, f"the loss fell from {first:{_FLOAT}} to {last:{_FLOAT}} (mean of the first and second half of {len(values)} logged points)"


def sample_history(history: Sequence[Sequence[float]], points: int) -> list[list[float]]:
    """At most `points` entries of the history, evenly spaced, the first and the last always among them."""
    entries = [list(h) for h in history]
    n = len(entries)
    if n <= points:
        return entries
    return [entries[i * (n - 1) // (points - 1)] for i in range(points)]


def _tidy(history: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[step, float(format(loss, _FLOAT))] for step, loss in history]


def _show_history(history: Sequence[Sequence[float]], points: int) -> str:
    return " ".join(f"{step}:{loss:{_FLOAT}}" for step, loss in sample_history(history, points))


# --- the call into AlignTune -------------------------------------------------------------------------------------


def _compute_dtype(train: SimpleNamespace) -> str:
    if train.fp16:
        return "float16"
    return "bfloat16" if train.bf16 else "float32"


def _precision(train: SimpleNamespace) -> str:
    if train.fp16:
        return "fp16"
    return "bf16" if train.bf16 else "fp32"


def _quantization(cfg: SimpleNamespace) -> dict[str, Any] | None:
    q = cfg.quantization
    if q.mode == "none":
        return None
    if q.mode == "8bit":
        return {"load_in_8bit": True}
    return {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": _compute_dtype(cfg.train),  # AlignTune's own default is bfloat16, absent on a T4
        "bnb_4bit_quant_type": q.quant_type,
        "bnb_4bit_use_double_quant": q.double_quant,
    }


def chat_template_text(template: Any) -> str:
    """The text of a tokenizer's default chat template, whatever shape the tokenizer keeps it in.

    Tiny Aya Global stores `chat_template` in tokenizer_config.json as a named list
    `[{"name": "default", "template": ...}, ...]`; transformers 5 turns that into a dict, and TRL's
    `has_generation_markers` and its template swap need a string. A string is taken as it is, a dict or a list
    gives its "default" entry, and anything else is an error that says what was found."""
    where = "the tokenizer's chat_template"
    if template is None:
        raise FinetuneError("the tokenizer has no chat_template (it is None)")
    if isinstance(template, str):
        entry, label = template, where
    elif isinstance(template, (Mapping, list)):
        kind = "mapping" if isinstance(template, Mapping) else "list"
        if isinstance(template, Mapping):
            named = dict(template)
        else:
            named = {item.get("name"): item.get("template") for item in template if isinstance(item, Mapping)}
        if not named:
            raise FinetuneError(f"{where} is an empty {kind}")
        if DEFAULT_TEMPLATE not in named:
            raise FinetuneError(
                f"{where} holds the named templates {sorted(map(str, named))} but none is called '{DEFAULT_TEMPLATE}'"
            )
        entry, label = named[DEFAULT_TEMPLATE], f"the '{DEFAULT_TEMPLATE}' template in {where}"
    else:
        raise FinetuneError(f"{where} is {type(template).__name__}, expected text or a named list of templates")
    if not isinstance(entry, str):
        raise FinetuneError(f"{label} is a {type(entry).__name__}, not text")
    if not entry.strip():
        raise FinetuneError(f"{label} is empty")
    return entry


def _same_value(have: Any, want: Any) -> bool:
    """A flag is compared as a flag, a number as a number (epochs are a float in transformers), an enum by its value."""
    if isinstance(want, bool):
        return have is want
    if isinstance(want, (int, float)):
        if isinstance(have, bool):
            return False
        try:
            return float(have) == float(want)
        except (TypeError, ValueError):
            return False
    return str(getattr(have, "value", have)) == str(want)


def expected_trainer_args(cfg: SimpleNamespace, seed: int) -> dict[str, tuple[str, Any]]:
    """The Hugging Face TrainingArguments attributes that must come out as the config says, with the config key of each."""
    t = cfg.train
    return {
        "fp16": ("train.fp16", t.fp16),
        "bf16": ("train.bf16", t.bf16),
        "learning_rate": ("train.learning_rate", t.learning_rate),
        "num_train_epochs": ("train.epochs", t.epochs),
        "per_device_train_batch_size": ("train.per_device_batch_size", t.per_device_batch_size),
        "gradient_accumulation_steps": ("train.gradient_accumulation_steps", t.gradient_accumulation_steps),
        "max_length": ("train.max_seq_length", t.max_seq_length),
        "assistant_only_loss": ("train.assistant_only_loss", t.assistant_only_loss),
        "seed": ("train.seed", seed),
        "gradient_checkpointing": ("train.gradient_checkpointing", t.gradient_checkpointing),
        "save_total_limit": ("train.save_total_limit", t.save_total_limit),
        "warmup_steps": ("train.warmup_ratio", t.warmup_ratio),  # a float below 1 is a ratio in transformers >= 5.2
        "lr_scheduler_type": ("train.lr_scheduler", t.lr_scheduler),
    }


@dataclass
class StartCheck:
    """What to verify once the Hugging Face trainer has been built and before its first step.

    Two checks. The trainer-argument check compares thirteen attributes of the trainer's own `TrainingArguments`
    (see `expected_trainer_args`) with the config, because AlignTune drops or rewrites keywords it does not
    know; `args_compared` stays None when none could be compared. The labelled-token check: TRL turns the assistant part of each example into `labels` (everything else is
    `IGNORE_INDEX`). The task answer is one letter, so a row should have a handful of labelled tokens: none means
    the assistant mask matched nothing, a great many means the loss covers the prompt too. The real trainer calls
    `observe` from an `on_train_begin` callback; `label_counts` stays None, with `label_note` saying why, when
    the rows could not be read."""

    rows: int
    minimum: int
    maximum: int
    expected_args: Mapping[str, tuple[str, Any]] = field(default_factory=dict)  # TrainingArguments attribute -> (config key, value)
    label_counts: list[int] | None = None
    label_note: str | None = None
    args_compared: list[str] | None = None
    args_missing: list[str] = field(default_factory=list)
    args_note: str | None = None

    def check_labels(self, counts: Sequence[int]) -> None:
        self.label_counts = [int(n) for n in counts]
        self.label_note = None
        for index, n in enumerate(self.label_counts, start=1):
            if not self.minimum <= n <= self.maximum:
                raise FinetuneError(
                    f"row {index} of the tokenised training data has {n} labelled tokens, expected {self.minimum} to "
                    f"{self.maximum} (checks.label_tokens_min / label_tokens_max): "
                    + (
                        "the assistant mask matched nothing, so check that the chat template has generation markers "
                        "and that train.assistant_only_loss is on"
                        if n < self.minimum
                        else "the loss seems to cover more than the answer, so check train.assistant_only_loss and the chat template"
                    )
                )

    def observe(self, *, train_dataloader: Any, args: Any) -> None:
        """Called with what the Hugging Face trainer passes to `on_train_begin`; raises `FinetuneError` on a bad row."""
        counts = self._read_label_counts(train_dataloader)
        if counts is not None:
            self.check_labels(counts)
        self.check_args(args)

    def check_args(self, args: Any) -> None:
        """Compare the Hugging Face trainer's own arguments with the config: AlignTune and TRL may change or drop a keyword.

        An attribute the installed version does not have is listed as not found, not as a difference."""
        if args is None:
            self.args_note = "the callback received no TrainingArguments"
            return
        compared, missing, wrong = [], [], []
        for name, (key, want) in self.expected_args.items():
            if not hasattr(args, name):
                missing.append(name)
                continue
            have = getattr(args, name)
            compared.append(name)
            if not _same_value(have, want):
                wrong.append(f"{name} is {have!r} but the config asks {want!r} ({key})")
        self.args_compared = sorted(compared) or None
        self.args_missing = sorted(missing)
        if wrong:
            raise FinetuneError(
                "the Hugging Face trainer was built with other arguments than the config: " + "; ".join(wrong)
                + ". AlignTune or TRL changed or ignored a setting; see docs/toolkit-notes.md."
            )

    def _read_label_counts(self, loader: Any) -> list[int] | None:
        if loader is None:
            self.label_note = "no dataloader was passed to the callback"
            return None
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            self.label_note = "the dataloader has no dataset"
            return None
        try:
            rows = [dataset[i] for i in range(min(self.rows, len(dataset)))]
        except Exception as exc:
            self.label_note = f"the tokenised rows cannot be read ({type(exc).__name__}: {exc})"
            return None
        if not rows:
            self.label_note = "the dataset has no rows"
            return None
        try:
            values = [row["labels"] for row in rows]
        except Exception:
            self.label_note = "the tokenised rows have no labels column"
            return None
        return [sum(1 for v in (x.tolist() if hasattr(x, "tolist") else x) if v != IGNORE_INDEX) for x in values]


def build_trainer_params(
    cfg: SimpleNamespace, *, data_path: Path, adapter_dir: Path, chat_template_path: Path, seed: int, max_steps: int | None
) -> dict[str, Any]:
    """The keyword arguments of AlignTune's `create_sft_trainer`, all taken from the config.

    The names are the ones read from the source of aligntune 0.1.31 (docs/toolkit-notes.md, "AlignTune SFT (read
    from source)"). A misspelled keyword is NOT an error in AlignTune: it is dropped silently, which is why every
    name here is pinned by tests/test_finetune.py. A smoke run (`max_steps` set) logs every step, so that a
    decrease of the loss can be seen after a few steps."""
    t, lora, tune = cfg.train, cfg.lora, cfg.aligntune
    params: dict[str, Any] = {
        "model_name": cfg.base_model,
        "dataset_name": str(data_path),
        "backend": tune.backend,
        "output_dir": str(adapter_dir),
        "num_epochs": t.epochs,
        "batch_size": t.per_device_batch_size,
        "learning_rate": t.learning_rate,
        "max_seq_length": t.max_seq_length,
        "use_peft": True,
        "lora_r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "lora_target_modules": list(lora.target_modules),
        "lora_bias": lora.bias,
        "precision": _precision(t),  # explicit: AlignTune's AUTO can pick bf16 on a T4
        "fp16": t.fp16,
        "bf16": t.bf16,
        "gradient_checkpointing": t.gradient_checkpointing,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "warmup_ratio": t.warmup_ratio,
        "lr_scheduler": t.lr_scheduler,
        "weight_decay": t.weight_decay,
        "max_grad_norm": t.max_grad_norm,
        "assistant_only_loss": t.assistant_only_loss,
        "chat_template_path": str(chat_template_path),  # a SFTConfig field AlignTune copies on; see chat_template_text
        "seed": seed,
        "data_seed": seed,
        "logging_steps": 1 if max_steps is not None else t.logging_steps,
        "save_steps": t.save_steps,
        "save_total_limit": t.save_total_limit,
        "device_map": tune.device_map,
        "load_best_model_at_end": tune.load_best_model_at_end,
        "eval_strategy": tune.eval_strategy,
        "loggers": list(tune.loggers),
    }
    quantization = _quantization(cfg)
    if quantization is not None:
        params["quantization"] = quantization
    if max_steps is not None:
        params["max_steps"] = max_steps
    return params


class _AlignTuneRun:
    """The real AlignTune trainer behind the small interface `run` uses: `train()` returns plain numbers.

    `train()` returns `{mean_train_loss, steps, loss_history: [[step, loss]...], rows_seen}`. The numbers come from
    the dict `TRLSFTTrainer.train()` returns (`final_loss`, which is `TrainOutput.training_loss`, the mean over
    the whole run, and `steps`) and from the Hugging Face trainer inside it
    (`state.log_history`, `train_dataset`); each is None or empty when the installed version does not offer it,
    and `run` then refuses or notes it. Read from the source of aligntune 0.1.31, never run."""

    def __init__(self, inner: Any):
        self.inner = inner

    def train(self) -> dict[str, Any]:
        result = self.inner.train() or {}
        hf = getattr(self.inner, "trainer", None)
        state = getattr(hf, "state", None)
        log = getattr(state, "log_history", None) or []
        logged = [entry for entry in log if isinstance(entry, Mapping) and "loss" in entry]
        history = [[entry.get("step", index + 1), float(entry["loss"])] for index, entry in enumerate(logged)]
        steps = result.get("steps")
        if steps is None:
            steps = getattr(state, "global_step", None)
        dataset = getattr(hf, "train_dataset", None)
        rows_seen = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else None
        return {"mean_train_loss": result.get("final_loss"), "steps": steps, "loss_history": history, "rows_seen": rows_seen}


def _hook_start_check(inner: Any, start_check: StartCheck) -> None:
    """Make the Hugging Face trainer AlignTune builds inside `train()` call `start_check.observe` before its first step.

    AlignTune's `train()` creates the TRL trainer and runs it in one method, so there is no moment in between to
    look at the data; its SFT trainer passes `callbacks=self.get_hf_callbacks()` to TRL, which is wrapped here."""
    hook = getattr(inner, "get_hf_callbacks", None)
    if hook is None:
        start_check.label_note = "this AlignTune has no get_hf_callbacks hook"
        return
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        start_check.label_note = f"transformers could not be imported ({exc})"
        return

    class _StartCheckCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            start_check.observe(train_dataloader=kwargs.get("train_dataloader"), args=args)

    callback = _StartCheckCallback()
    inner.get_hf_callbacks = lambda: [*hook(), callback]


def aligntune_factory(params: Mapping[str, Any], start_check: StartCheck | None = None) -> _AlignTuneRun:
    """Build AlignTune's SFT trainer from `params` (nothing is loaded until `train()`); the only AlignTune import."""
    try:
        from aligntune.core.backend_factory import create_sft_trainer
    except ImportError as exc:
        raise FinetuneError(
            f"AlignTune could not be imported ({exc}). Install it with `pip install aligntune=={ALIGNTUNE_VERSION}` on "
            "the runtime that trains (it needs Python 3.11 or newer; docs/runtime.md, section 5)."
        ) from exc
    inner = create_sft_trainer(**params)
    if start_check is not None:
        _hook_start_check(inner, start_check)
    return _AlignTuneRun(inner)


def default_merge(base_model: str, adapter_dir: Path, out_dir: Path, *, dtype: str) -> None:
    """Merge the LoRA adapter into a fresh, unquantised copy of the base model and save it with the tokenizer.

    The same PEFT calls as AlignTune's merger (`PeftModel.from_pretrained`, `merge_and_unload`, `save_pretrained`),
    without its provenance file (absolute paths) and its transformers-4 legacy fields. Loaded on the CPU."""
    try:
        import torch
        import transformers
        from peft import PeftModel
    except ImportError as exc:
        raise FinetuneError(
            f"merging needs torch, transformers and peft ({exc}); they come with `pip install aligntune=={ALIGNTUNE_VERSION}`"
        ) from exc
    model = transformers.AutoModelForCausalLM.from_pretrained(base_model, dtype=getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)  # transformers 5 writes safetensors by default
    transformers.AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)


def default_tokenizer_loader(name: str) -> Any:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(name)


def default_hf_api_factory(token: str) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise FinetuneError(f"--push-to needs huggingface_hub ({exc}); pip install huggingface_hub") from exc
    return HfApi(token=token)


def default_device_info() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing, no driver, anything: the name is a note, never a reason to fail
        return "unknown"


def library_versions() -> dict[str, str | None]:
    """Installed versions of the libraries that matter, read from package metadata (nothing is imported)."""
    found: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


# --- the data ---------------------------------------------------------------------------------------------------


def shown(path: str | Path, root: Path) -> str:
    """`path` relative to the repository root when it is inside it, else only its name: never an absolute path."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.name


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


@dataclass
class DataCheck:
    path: Path
    sha256: str
    records: list[dict[str, Any]]
    per_language: dict[str, int]
    heldout_checked: bool
    taskdev_checked: bool
    safety_checked: bool
    notes: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.records)


def _read(path: Path, where: str, root: Path) -> tuple[list[Any], list[ftdata.Problem]]:
    try:
        return ftdata.read_jsonl_records(path, where)
    except (OSError, ValueError) as exc:
        raise FinetuneError(f"cannot read the {where} data {shown(path, root)}: {getattr(exc, 'strerror', None) or exc}") from None


def check_data(cfg: SimpleNamespace, train_path: Path, root: Path) -> DataCheck:
    """Run the `msd.ftdata` validator in-process; raise `FinetuneError` (with its report) when any rule fails.

    Same rules as scripts/validate_ft_data.py: schema, bare answers, no refusals, script per language, language
    coverage and balance, the harmless-subject allowlist (always checked, from `data.pipeline`), no overlap
    between train and heldout or taskdev, none with the safety dev/report splits. Heldout, taskdev and the safety
    test set are checked when their files exist; a missing one is said to be NOT checked and recorded."""
    data = cfg.data
    pipeline = _resolve(data.pipeline, root)
    try:
        subjects = ftdata.load_allowed_subjects(pipeline)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise FinetuneError(f"the subject allowlist cannot be read from {shown(pipeline, root)}: {getattr(exc, 'strerror', None) or exc}") from None
    train, problems = _read(train_path, "train", root)
    notes: list[str] = []
    heldout_path, taskdev_path, testset_dir = (_resolve(data.heldout, root), _resolve(data.taskdev, root), _resolve(data.testset, root))
    heldout: list[Any] | None = None
    taskdev: list[Any] | None = None
    if heldout_path.is_file():
        heldout, more = _read(heldout_path, "heldout", root)
        problems += more
    else:
        notes.append(f"note: {shown(heldout_path, root)} not found, so overlap with the held-out items is NOT checked")
    if taskdev_path.is_file():
        taskdev, more = _read(taskdev_path, "taskdev", root)
        problems += more
    else:
        notes.append(f"note: {shown(taskdev_path, root)} not found, so overlap with the dev task items is NOT checked")
    dev_texts: list[str] = []
    report_texts: list[str] = []
    safety_counts = None
    if testset_dir.is_dir():
        try:
            dev_texts, report_texts, safety_counts = ftdata.load_safety_texts(testset_dir)
        except (OSError, ValueError) as exc:
            raise FinetuneError(f"the safety test set {shown(testset_dir, root)} cannot be read: {getattr(exc, 'strerror', None) or exc}") from None
    else:
        notes.append(f"note: {shown(testset_dir, root)} not found, so overlap with the safety dev/report splits is NOT checked")
    report = ftdata.validate(
        train,
        heldout if heldout is not None else [],
        expected_langs=data.langs,
        safety_dev=dev_texts,
        safety_report=report_texts,
        allowed_subjects=subjects,
        safety_counts=safety_counts,
        taskdev=taskdev,
        read_problems=problems,
    )
    if heldout is None:  # an absent heldout file leaves every language "too few rows" there; that is not the train data's fault
        in_train = {lang for lang, n in report.counts["train"].items() if n}
        report.problems = [
            p for p in report.problems
            if not (p.rule == "missing_lang" and p.where == "set" and p.lang in in_train)
        ]
    if not report.ok:
        raise FinetuneError(
            f"refusing to train: the data failed validation (rules: {', '.join(report.failed_rules())})\n{ftdata.render(report)}"
        )
    counts = report.counts["train"]
    return DataCheck(
        path=train_path,
        sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        records=train,
        per_language={lang: counts.get(lang, 0) for lang in data.langs},
        heldout_checked=heldout is not None,
        taskdev_checked=taskdev is not None,
        safety_checked=safety_counts is not None,
        notes=notes,
    )


def token_lengths(tokenizer: Any, records: Sequence[Mapping[str, Any]]) -> list[int]:
    """Tokens per example with the model's chat template, as TRL will count them (the template adds the special tokens)."""
    lengths = []
    for record in records:
        rendered = tokenizer.apply_chat_template(record["messages"], tokenize=False)
        lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
    return lengths


def _length_line(lengths: Sequence[int], limit: int) -> tuple[str, int]:
    ordered = sorted(lengths)
    over = sum(1 for n in ordered if n > limit)
    mean = sum(ordered) / len(ordered)
    line = (
        f"token lengths: min {ordered[0]}, median {ordered[len(ordered) // 2]}, mean {mean:{_ONE_DECIMAL}}, max {ordered[-1]}; "
        f"{over} over max_seq_length {limit}"
    )
    return line, over


# --- push guard -------------------------------------------------------------------------------------------------


def check_push_target(api: Any, repo_id: str) -> None:
    """Refuse unless the Hub repo exists and `repo_info(...).private is True`. Nothing is ever created."""
    try:
        info = api.repo_info(repo_id, repo_type="model")
    except Exception as exc:
        raise FinetuneError(
            f"cannot read the Hub repo {repo_id} ({type(exc).__name__}: {exc}); create it by hand as a PRIVATE "
            "model repo first: this script never creates one"
        ) from None
    private = getattr(info, "private", None)
    if private is not True:
        raise FinetuneError(f"the Hub repo {repo_id} is not private (private={private!r}); refusing to upload the model there")


# --- arguments --------------------------------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse repeats unrecognised arguments, and one may be a token
        super().error(scrub(message))


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive whole number, got {text!r}")
    return value


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"the seed must be a non-negative whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"the seed must be a non-negative whole number, got {text!r}")
    return value


def _repo_id(text: str) -> str:
    if not REPO_ID.fullmatch(text):
        raise argparse.ArgumentTypeError(f"expected ORG/NAME (letters, digits, dot, dash, underscore), got {text!r}")
    return text


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="finetune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="YAML config, e.g. configs/ft_lora.yaml (every training value is in it)")
    ap.add_argument("--data", type=Path, help="training jsonl (default: data.train of the config)")
    ap.add_argument("--out", type=Path, help="output folder; must not exist or be empty (default: results/finetune/<config stem>)")
    ap.add_argument("--dry-run", action="store_true", help="check the config and the data and print the plan; writes nothing and loads no model (the tokenizer is read with transformers and the chat template checked with trl, each only if installed)")
    ap.add_argument("--max-steps", type=_positive_int, help="smoke run: train this many steps only; output goes to <config stem>-smoke, no merge unless --merge")
    ap.add_argument("--merge", action=argparse.BooleanOptionalAction, default=None,
                    help="merge the adapter into fp16 weights (default: yes, except in a smoke run)")
    ap.add_argument("--push-to", type=_repo_id, metavar="ORG/NAME",
                    help="after a successful full run upload the merged folder to this PRIVATE Hub repo (HF_TOKEN from the environment)")
    ap.add_argument("--seed", type=_seed, help="seed (default: train.seed of the config)")
    ap.add_argument("--merge-only", action="store_true",
                    help="train nothing: merge the adapter given by --adapter into <out>/merged and add a merge section to the summary")
    ap.add_argument("--adapter", type=Path, help="with --merge-only: the adapter folder to merge (e.g. <out>/adapter left by a failed merge)")
    return ap


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse and cross-check; any usage error prints the usage and exits with code 2 (SystemExit)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    args.smoke = args.max_steps is not None
    if args.merge_only:
        if args.adapter is None:
            ap.error("--merge-only needs --adapter PATH, the adapter folder to merge")
        given = (
            ("--max-steps", args.smoke), ("--no-merge", args.merge is False), ("--seed", args.seed is not None),
            ("--dry-run", args.dry_run), ("--push-to", bool(args.push_to)), ("--data", args.data is not None),
        )
        clash = [name for name, on in given if on]
        if clash:
            ap.error(f"--merge-only cannot be combined with {', '.join(clash)}: it merges an existing adapter and trains nothing")
    elif args.adapter is not None:
        ap.error("--adapter is only for --merge-only")
    args.merge_effective = args.merge if args.merge is not None else not args.smoke
    if args.push_to:
        if args.smoke:
            ap.error("--push-to cannot be combined with --max-steps: a smoke run is never pushed")
        if not args.merge_effective:
            ap.error("--push-to needs the merged model: drop --no-merge")
        if not args.dry_run and not os.environ.get("HF_TOKEN"):
            ap.error("--push-to needs the token in the HF_TOKEN environment variable (there is no token flag)")
    return args


# --- the run ----------------------------------------------------------------------------------------------------


class UsageError(FinetuneError):
    """A bad output folder: the same exit code as a bad argument."""


@dataclass
class Hooks:
    trainer_factory: Callable[[Mapping[str, Any]], Any]
    merge_fn: Callable[..., None]
    tokenizer_loader: Callable[[str], Any]
    hf_api_factory: Callable[[str], Any]
    clock: Callable[[], float]
    device_info: Callable[[], str]
    versions: Callable[[], Mapping[str, str | None]]


def _release_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:  # only when a trainer already imported it; never imported here
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _check_adapter(adapter_dir: Path, cfg: SimpleNamespace, name: str) -> None:
    """The adapter on disk must be the one the config asked for: AlignTune drops a misspelled keyword silently."""
    path = adapter_dir / ADAPTER_CONFIG
    if not path.is_file():
        raise FinetuneError(f"the trainer finished but left no {ADAPTER_CONFIG} in the {name} folder, so there is no adapter to merge")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinetuneError(f"{ADAPTER_CONFIG} cannot be read: {exc}") from None
    modules = saved.get("target_modules")
    modules = {modules} if isinstance(modules, str) else set(modules or ())
    wanted = (
        ("r", "rank", saved.get("r"), cfg.lora.rank),
        ("lora_alpha", "alpha", saved.get("lora_alpha"), cfg.lora.alpha),
        ("lora_dropout", "dropout", saved.get("lora_dropout"), cfg.lora.dropout),
        ("bias", "bias", saved.get("bias"), cfg.lora.bias),
        ("base_model_name_or_path", "base model", saved.get("base_model_name_or_path"), cfg.base_model),
        ("target_modules", "target modules", modules, set(cfg.lora.target_modules)),
    )
    problems = [
        f"{key} is {have!r} in the adapter but the config asks for {want!r} ({label})"
        for key, label, have, want in wanted
        if have != want
    ]
    if problems:
        raise FinetuneError(
            "the adapter the trainer saved does not match the config: " + "; ".join(problems)
            + ". AlignTune ignores keywords it does not know, so check the names in docs/toolkit-notes.md."
        )
    if not any((adapter_dir / weights).is_file() for weights in ADAPTER_WEIGHTS):
        raise FinetuneError(f"the {name} folder has {ADAPTER_CONFIG} but none of the weights files ({' or '.join(ADAPTER_WEIGHTS)})")


def _train_and_check(
    hooks: Hooks, cfg: SimpleNamespace, args: argparse.Namespace, check: DataCheck, seed: int, work: Path
) -> dict[str, Any]:
    params = build_trainer_params(
        cfg, data_path=check.path, adapter_dir=work, chat_template_path=work / TEMPLATE_FILE, seed=seed, max_steps=args.max_steps
    )
    start_check = StartCheck(
        rows=cfg.checks.label_check_rows, minimum=cfg.checks.label_tokens_min, maximum=cfg.checks.label_tokens_max,
        expected_args=expected_trainer_args(cfg, seed),
    )
    trainer = hooks.trainer_factory(params, start_check)
    result = trainer.train()
    del trainer
    _release_memory()
    if start_check.label_counts is None:
        print(f"note: the labelled-token check could not run ({start_check.label_note or 'the trainer never reported its rows'})", file=sys.stderr)
    if start_check.args_compared is None:
        print(f"note: the trainer-argument check could not run ({start_check.args_note or 'no argument was found to compare'})", file=sys.stderr)

    _check_adapter(work, cfg, "work")
    rows_seen = result.get("rows_seen")
    if rows_seen is not None and rows_seen != check.rows:
        raise FinetuneError(
            f"the trainer reports training on {rows_seen} rows but {check.rows} are in the file: AlignTune or CuratorKIT "
            "dropped rows; look at the smoke run's data step before a full run"
        )
    t = cfg.train
    planned = plan_steps(check.rows, t.epochs, t.per_device_batch_size, t.gradient_accumulation_steps)
    if args.max_steps is not None:
        planned = min(planned, args.max_steps)
    steps = result.get("steps")
    if steps is None:
        raise FinetuneError("the trainer did not report how many steps it ran")
    if abs(steps - planned) > t.epochs:  # the step count may differ by one per epoch with the last partial batch
        raise FinetuneError(
            f"the trainer ran {steps} steps but the plan is {planned} ({check.rows} rows x {t.epochs} epochs / effective batch "
            f"{t.per_device_batch_size * t.gradient_accumulation_steps}); it probably saw other data than the file"
        )
    history = [[step, float(loss)] for step, loss in result.get("loss_history") or []]
    ok, reason = loss_check(history)
    mean_loss = result.get("mean_train_loss")
    if ok and mean_loss is not None and not math.isfinite(float(mean_loss)):
        ok, reason = False, "the mean training loss is not finite (NaN or inf)"
    if not ok:
        raise FinetuneError(f"{reason}; logged losses (step:loss) {_show_history(history, cfg.checks.history_points)}")
    return {
        "rows_seen": rows_seen,
        "planned": planned,
        "steps": steps,
        "mean_train_loss": None if mean_loss is None else float(format(float(mean_loss), _FLOAT)),
        "last_logged_loss": float(format(history[-1][1], _FLOAT)),
        "history": history,
        "loss_reason": reason,
        "args_check": (
            "unavailable"
            if start_check.args_compared is None
            else {"compared": start_check.args_compared, "not_found": start_check.args_missing}
        ),
        "label_check": (
            "unavailable"
            if start_check.label_counts is None
            else {"counts": start_check.label_counts, "allowed_min": start_check.minimum, "allowed_max": start_check.maximum}
        ),
    }


def _summary(
    cfg: SimpleNamespace, args: argparse.Namespace, config_path: Path, check: DataCheck, seed: int, trained: dict[str, Any],
    merged: bool, seconds: float, hooks: Hooks, root: Path, template_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "smoke": args.smoke,
        "max_steps": args.max_steps,
        "config": {"path": shown(config_path, root), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "overrides": {
            "data": shown(args.data, root) if args.data else None,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "base_model": cfg.base_model,
        "data": {
            "train": {
                "path": shown(check.path, root),
                "sha256": check.sha256,
                "rows": check.rows,
                "rows_per_language": check.per_language,
            },
            "heldout_checked": check.heldout_checked,
            "taskdev_checked": check.taskdev_checked,
            "safety_checked": check.safety_checked,
            "validation": "ok",
        },
        "seed": seed,
        "planned_steps": trained["planned"],
        "steps": trained["steps"],
        "rows_seen": trained["rows_seen"],
        "mean_train_loss": trained["mean_train_loss"],
        "last_logged_loss": trained["last_logged_loss"],
        "loss_history": _tidy(sample_history(trained["history"], cfg.checks.history_points)),
        "loss_ok": True,
        "loss_reason": trained["loss_reason"],
        "args_check": trained["args_check"],
        "label_check": trained["label_check"],
        "chat_template_sha256": template_sha256,
        "wall_clock_seconds": float(format(seconds, _FLOAT)),
        "device": hooks.device_info(),
        "libraries": dict(hooks.versions()),
        "merged": merged,
        "merge_dtype": cfg.output.merge_dtype,
        "push_to": args.push_to,
        "note": NOTE,
    }


def _plan_lines(
    cfg: SimpleNamespace, args: argparse.Namespace, config_path: Path, check: DataCheck, seed: int, out: Path, root: Path,
    length_line: str, template_line: str, work: Path,
) -> list[str]:
    t = cfg.train
    effective = t.per_device_batch_size * t.gradient_accumulation_steps
    steps = plan_steps(check.rows, t.epochs, t.per_device_batch_size, t.gradient_accumulation_steps)
    where = f"{shown(out, root)} (folders {cfg.output.adapter_dir}/ and {cfg.output.merged_dir}/, {SUMMARY_FILE})"
    lines = [
        "fine-tune plan" + (" (dry run: nothing is written, nothing is trained)" if args.dry_run else ""),
        f"config: {shown(config_path, root)} (sha256 {hashlib.sha256(config_path.read_bytes()).hexdigest()})",
        f"base model: {cfg.base_model}",
        f"training data: {shown(check.path, root)}, {check.rows} rows, sha256 {check.sha256}",
        "rows per language: " + ", ".join(f"{lang} {n}" for lang, n in check.per_language.items()),
        "validation: ok ("
        + f"heldout {'checked' if check.heldout_checked else 'NOT checked'}, "
        + f"taskdev {'checked' if check.taskdev_checked else 'NOT checked'}, "
        + f"safety splits {'checked' if check.safety_checked else 'NOT checked'})",
        f"lora: rank {cfg.lora.rank}, alpha {cfg.lora.alpha}, dropout {cfg.lora.dropout}, targets {', '.join(cfg.lora.target_modules)}; "
        f"quantization {cfg.quantization.mode}; precision {_precision(t)}",
        f"effective batch: {t.per_device_batch_size} x {t.gradient_accumulation_steps} = {effective}",
        f"planned optimizer steps: {steps} ({check.rows} rows x {t.epochs} epochs / effective batch {effective})",
    ]
    if args.smoke:
        lines.append(f"smoke run: stops after {args.max_steps} steps, logs every step, output {where}")
    else:
        lines.append(f"output: {where}")
    lines.append(f"work folder: {shown(work, root)} (AlignTune writes its checkpoints here; kept if the run fails)")
    lines.append(length_line)
    lines.append(template_line)
    lines.append(f"seed: {seed}; merge: {'yes' if args.merge_effective else 'no'} ({cfg.output.merge_dtype})")
    if args.push_to:
        lines.append(
            f"push: {args.push_to} after a successful run, only if it exists and is private"
            + (" (not checked in a dry run)" if args.dry_run else "")
        )
    return lines


def _load_tokenizer(hooks: Hooks, cfg: SimpleNamespace, *, required: bool) -> tuple[Any, str | None]:
    """The base model's tokenizer and None, or None and why it could not be loaded (a dry run does without one)."""
    try:
        return hooks.tokenizer_loader(cfg.base_model), None
    except Exception as exc:
        if required:
            raise FinetuneError(f"cannot load the tokenizer of {cfg.base_model} ({type(exc).__name__}: {exc})") from None
        return None, f"{type(exc).__name__}: {scrub(str(exc))}"


def _measure(tokenizer: Any, why: str | None, cfg: SimpleNamespace, check: DataCheck) -> str:
    """The token-length line; refuses when an example is longer than `train.max_seq_length`."""
    limit = cfg.train.max_seq_length
    if tokenizer is None:
        return f"token lengths: not estimated (no tokenizer: {why})"
    lengths = token_lengths(tokenizer, check.records)
    line, over = _length_line(lengths, limit)
    if over:
        raise FinetuneError(
            f"{over} of {check.rows} training examples are longer than train.max_seq_length {limit} tokens (longest "
            f"{max(lengths)}); TRL would cut the end of each, so the answer would be missing and the example would train on "
            "nothing. Raise train.max_seq_length or drop those rows."
        )
    return line


def _template_verdict(text: str) -> str:
    """Whether TRL can mask the assistant part with this template; refuses when it cannot (checked when trl imports)."""
    try:
        from trl.chat_template_utils import cohere2_chat_template, has_generation_markers
    except Exception as exc:  # not installed off the GPU runtime, or a broken install: a note, never a reason to stop
        return f"TRL check skipped (trl could not be imported here: {type(exc).__name__}); the GPU runtime makes it"
    if has_generation_markers(text):
        return "has generation markers, so TRL can mask everything but the answer"
    if text == cohere2_chat_template:
        return "is TRL's cohere2 template, which TRL swaps for its training variant with generation markers"
    raise FinetuneError(
        f"the tokenizer's default chat template ({len(text)} characters) has no generation markers and is not TRL's "
        "cohere2 template, so train.assistant_only_loss could not mask the prompt and TRL would stop (or train on the "
        "whole sequence). Use a model whose template TRL knows, or supply a template with {% generation %} markers."
    )


def _prepare_template(tokenizer: Any) -> tuple[str | None, str | None, str]:
    """The normalised template text, its sha256 and the plan line; refuses an unusable template."""
    if tokenizer is None:
        return None, None, "chat template: not checked (no tokenizer)"
    text = chat_template_text(getattr(tokenizer, "chat_template", None))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest, f"chat template: {len(text)} characters, sha256 {digest}; {_template_verdict(text)}"


def _default_out(root: Path, stem: str, smoke: bool) -> Path:
    return Path(root) / "results" / "finetune" / (stem + (SMOKE_SUFFIX if smoke else ""))


def _work_dir(root: Path, stem: str, smoke: bool) -> Path:
    return Path(root) / "work" / "finetune" / (stem + (SMOKE_SUFFIX if smoke else ""))


def _work_contents(work: Path) -> list[str]:
    return sorted(p.name for p in work.iterdir() if p.name != TEMPLATE_FILE) if work.is_dir() else []


def _check_work_dir(work: Path, root: Path) -> None:
    """A work folder that holds an earlier run's files (checkpoints, an adapter) is never overwritten."""
    if work.is_symlink() or (work.exists() and not work.is_dir()):
        raise UsageError(f"the work folder {shown(work, root)} is not a plain folder; move it away")
    left = _work_contents(work)
    if left:
        raise UsageError(
            f"the work folder {shown(work, root)} is not empty ({', '.join(left)}): it is what an earlier run left for "
            "recovery. Move or delete it, then run again"
        )


def _kept_note(work: Path, root: Path, adapter: Path | None) -> str:
    left = _work_contents(work)
    text = (
        f"the work folder {shown(work, root)} was kept, holding {', '.join(left)} (AlignTune cannot resume from it by "
        "itself: docs/runtime.md, 'After a failure')"
        if left
        else f"the work folder {shown(work, root)} was kept, empty apart from the chat template file"
    )
    if adapter is not None:
        text += f"; the finished adapter is in {shown(adapter, root)}: merge it with --merge-only --adapter {shown(adapter, root)}"
    return text


def _copy_adapter(work: Path, folder: Path) -> None:
    """The adapter's own files, not the checkpoints (optimizer states), the tracker logs or the provenance file."""
    for item in sorted(work.iterdir()):
        if item.is_file() and item.name != TEMPLATE_FILE and item.name not in NOT_PUBLISHED:
            shutil.copyfile(item, folder / item.name)


def _write_atomically(path: Path, text: str) -> None:
    """Write `path` whole or not at all: a file in the same folder, flushed to disk, then renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _merge(hooks: Hooks, cfg: SimpleNamespace, adapter_dir: Path, merged_out: Path, marker: str | None = None) -> None:
    """Merge into a hidden partial folder that becomes `merged_out` only when the merge finished.

    `marker` (a smoke run) is written beside the weights, so that a smoke model cannot pass for A1."""
    with atomic_output_dir(merged_out) as folder:
        hooks.merge_fn(cfg.base_model, adapter_dir, folder, dtype=cfg.output.merge_dtype)
        if not any(folder.iterdir()):
            raise FinetuneError(f"the merge wrote nothing into {cfg.output.merged_dir}")
        if marker is not None:
            (folder / SMOKE_MARKER).write_text(marker + "\n", encoding="utf-8")


def _loss_text(value: float | None) -> str:
    return "not reported" if value is None else format(value, _FLOAT)


def _json_text(data: Mapping[str, Any]) -> str:
    return scrub(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _data_folders(cfg: SimpleNamespace, train_path: Path, root: Path) -> list[tuple[str, Path]]:
    folders = {_resolve(p, root).parent for p in (cfg.data.train, cfg.data.heldout, cfg.data.taskdev)} | {train_path.parent}
    return [("data", d) for d in sorted(folders)]


def execute_merge_only(args: argparse.Namespace, root: Path, hooks: Hooks) -> int:
    """`--merge-only --adapter PATH`: merge an adapter that is already on disk; nothing is trained, no data is read."""
    cfg = load_config(args.config)
    out = args.out if args.out else Path(root) / "results" / "finetune" / args.config.stem
    adapter = _resolve(args.adapter, root)
    merged_out = out / cfg.output.merged_dir
    try:
        check_out_dir(merged_out, sources=_data_folders(cfg, _resolve(cfg.data.train, root), root))
    except (FileExistsError, ValueError) as exc:
        raise UsageError(str(exc)) from None
    note_stale_partials(merged_out)
    _check_adapter(adapter, cfg, "adapter")  # the adapter must be the configured one, base model included

    started = hooks.clock()
    _merge(hooks, cfg, adapter, merged_out)
    seconds = hooks.clock() - started
    section = {
        "adapter": shown(adapter, root),
        "dtype": cfg.output.merge_dtype,
        "wall_clock_seconds": float(format(seconds, _FLOAT)),
        "libraries": dict(hooks.versions()),
    }
    summary_path = out / SUMMARY_FILE
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FinetuneError(f"the merged model is in {shown(merged_out, root)} but {SUMMARY_FILE} cannot be read: {exc}") from None
        if not isinstance(summary, dict):
            raise FinetuneError(f"the merged model is in {shown(merged_out, root)} but {SUMMARY_FILE} is not a JSON object")
    else:
        summary = {
            "schema": 1,
            "config": {"path": shown(args.config, root), "sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()},
            "base_model": cfg.base_model,
            "merge_dtype": cfg.output.merge_dtype,
            "note": NOTE,
        }
    summary["merged"] = True
    summary["merge"] = section
    _write_atomically(summary_path, _json_text(summary))
    print(f"merged {shown(adapter, root)} into {shown(merged_out, root)} ({cfg.output.merge_dtype}); {SUMMARY_FILE} has the merge section")
    return EXIT_OK


def execute(args: argparse.Namespace, root: Path, hooks: Hooks) -> int:
    if args.merge_only:
        return execute_merge_only(args, root, hooks)
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.train.seed
    stem = args.config.stem
    out = args.out if args.out else _default_out(root, stem, args.smoke)
    if args.smoke and args.out and Path(args.out).resolve() == _default_out(root, stem, False).resolve():
        raise UsageError(
            f"a smoke run cannot write into {shown(args.out, root)}, the folder of the real run; leave --out out (the "
            f"default is {shown(_default_out(root, stem, True), root)}) or choose another folder"
        )
    train_path = args.data if args.data else _resolve(cfg.data.train, root)
    try:
        check_out_dir(out, sources=_data_folders(cfg, train_path, root))
    except (FileExistsError, ValueError) as exc:
        hint = f" (an adapter a failed run left there can be merged: --merge-only --adapter {shown(out / cfg.output.adapter_dir, root)})"
        raise UsageError(str(exc) + (hint if (out / cfg.output.adapter_dir).is_dir() else "")) from None
    note_stale_partials(out)
    work = _work_dir(root, stem, args.smoke)
    _check_work_dir(work, root)

    check = check_data(cfg, train_path, root)
    for note in check.notes:
        print(note, file=sys.stderr)
    tokenizer, why = _load_tokenizer(hooks, cfg, required=not args.dry_run)
    length_line = _measure(tokenizer, why, cfg, check)
    template_text, template_sha256, template_line = _prepare_template(tokenizer)
    for line in _plan_lines(cfg, args, args.config, check, seed, out, root, length_line, template_line, work):
        print(line)
    if args.dry_run:
        return EXIT_OK

    api = None
    if args.push_to:
        api = hooks.hf_api_factory(os.environ["HF_TOKEN"])
        check_push_target(api, args.push_to)  # before the hours of training, not after

    started = hooks.clock()
    work.mkdir(parents=True, exist_ok=True)
    (work / TEMPLATE_FILE).write_text(template_text, encoding="utf-8")
    adapter_out = out / cfg.output.adapter_dir
    published = False
    try:
        trained = _train_and_check(hooks, cfg, args, check, seed, work)
        with atomic_output_dir(adapter_out) as folder:
            _copy_adapter(work, folder)
        published = True
        merged = False
        if args.merge_effective:
            marker = (
                f"smoke run of {args.max_steps} steps: this model is not A1 and must not be used for the study"
                if args.smoke else None
            )
            _merge(hooks, cfg, adapter_out, out / cfg.output.merged_dir, marker)
            merged = True
        seconds = hooks.clock() - started
        summary = _summary(cfg, args, args.config, check, seed, trained, merged, seconds, hooks, root, template_sha256)
        _write_atomically(out / SUMMARY_FILE, _json_text(summary))  # last: its presence means the run finished
    except FinetuneError as exc:
        raise FinetuneError(f"{exc}; {_kept_note(work, root, adapter_out if published else None)}") from None
    except BaseException:  # a CUDA error, Ctrl-C: the traceback stays, and the recovery folder is named
        print(f"note: {_kept_note(work, root, adapter_out if published else None)}", file=sys.stderr)
        raise
    if cfg.output.keep_checkpoints:
        print(f"kept the work folder {shown(work, root)} (output.keep_checkpoints): AlignTune's checkpoints are in it")
    else:
        shutil.rmtree(work, ignore_errors=True)

    print(f"wrote {shown(out, root)} ({'smoke run, ' if args.smoke else ''}{trained['steps']} steps, mean train loss {_loss_text(trained['mean_train_loss'])}, last logged loss {trained['last_logged_loss']:{_FLOAT}}; {trained['loss_reason']})")
    if args.smoke:
        print("smoke run only: this model is not A1; run without --max-steps for the real fine-tune", file=sys.stderr)
    if api is not None:
        try:
            check_push_target(api, args.push_to)  # again: the repo may have been made public during the hours of training
        except FinetuneError as exc:
            raise FinetuneError(f"{exc}; the training result is kept in {shown(out, root)}") from None
        try:
            api.upload_folder(
                folder_path=str(out / cfg.output.merged_dir), repo_id=args.push_to, repo_type="model",
                commit_message=f"A1 fine-tune ({stem}); training data sha256 {check.sha256}",
            )
        except Exception as exc:
            raise FinetuneError(
                f"the upload to {args.push_to} failed ({type(exc).__name__}: {exc}); the training result is kept in {shown(out, root)}"
            ) from None
        print(f"uploaded {shown(out / cfg.output.merged_dir, root)} to the private repo {args.push_to}")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    trainer_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    merge_fn: Callable[..., None] | None = None,
    tokenizer_loader: Callable[[str], Any] | None = None,
    hf_api_factory: Callable[[str], Any] | None = None,
    clock: Callable[[], float] | None = None,
    device_info: Callable[[], str] | None = None,
    versions: Callable[[], Mapping[str, str | None]] | None = None,
    root: Path | None = None,
) -> int:
    """Run the fine-tune; every keyword is an injection point tests use (fakes for AlignTune, PEFT, the Hub, the clock).

    `root` is the repository root that the paths inside the config and the default `--out` are relative to; the
    script passes its own, and the default is the current folder."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    hooks = Hooks(
        trainer_factory=trainer_factory or aligntune_factory,
        merge_fn=merge_fn or default_merge,
        tokenizer_loader=tokenizer_loader or default_tokenizer_loader,
        hf_api_factory=hf_api_factory or default_hf_api_factory,
        clock=clock or time.monotonic,
        device_info=device_info or default_device_info,
        versions=versions or library_versions,
    )
    try:
        return execute(args, Path(root) if root is not None else Path.cwd(), hooks)
    except UsageError as exc:
        print(f"error: {scrub(str(exc))}", file=sys.stderr)
        return EXIT_USAGE
    except (FinetuneError, OSError, ValueError, ImportError) as exc:
        print(f"error: {scrub(str(exc))}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        print(scrub(traceback.format_exc()), file=sys.stderr)
        return EXIT_ERROR
