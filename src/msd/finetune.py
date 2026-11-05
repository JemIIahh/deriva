"""Fine-tune A0 into A1 with AlignTune (plan Task 14): the config schema comes first, the run is added below.

Every training value lives in a YAML file (configs/ft_lora.yaml, configs/ft_lora_strong.yaml). This module holds
no default for any of them: it checks that each key is present, has the right type and is in range, and that no
unknown key is there, and it reports every problem at once.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from msd.config import LANGS


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
        "group_by_length": flag,
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
    "checks": {"history_points": integer(2)},
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
