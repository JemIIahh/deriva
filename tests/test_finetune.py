"""scripts/finetune.py and msd.finetune: the A1 fine-tune, with AlignTune, PEFT, a tokenizer and the Hub replaced by fakes.

AlignTune is not installed in the test environment and no model is ever loaded. The fake trainer receives exactly
the keyword arguments the real `create_sft_trainer` would, writes the files the real one writes (an adapter
folder, checkpoint folders) and returns the numbers the real wrapper extracts (docs/toolkit-notes.md, "AlignTune
SFT (read from source)"). What these tests cannot show is that the REAL AlignTune behaves like that: that is the
first GPU-runtime run (docs/runtime.md, section 5).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from test_validate_ft import FOUR, rec, user_turn

from msd import finetune
from msd.schema import Prompt, write_jsonl

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "finetune.py"
MODULE = REPO / "src" / "msd" / "finetune.py"
REAL_CONFIG = REPO / "configs" / "ft_lora.yaml"
TOKEN = "hf_SENTINELTOKENVALUE0123456789abcdef"  # stands in for a Hugging Face token: it must never be printed
HEAVY = ("torch", "transformers", "aligntune", "peft", "trl", "huggingface_hub", "bitsandbytes")


# --- fakes ------------------------------------------------------------------------------------------------------


def losses_down(n=6):
    return [[i + 1, 2.0 / (i + 1)] for i in range(n)]


class FakeTrainer:
    """What the AlignTune wrapper hands back: train() writes the adapter files and returns the numbers."""

    def __init__(self, params, owner):
        self.params, self.owner = params, owner

    def train(self):
        owner, p = self.owner, self.params
        if owner.fail_with is not None:
            raise owner.fail_with
        out = Path(p["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        rows = sum(1 for line in Path(p["dataset_name"]).read_text(encoding="utf-8").splitlines() if line.strip())
        eff = p["batch_size"] * p["gradient_accumulation_steps"]
        steps = p.get("max_steps") or math.ceil(rows * p["num_epochs"] / eff)
        adapter = {
            "peft_type": "LORA",
            "base_model_name_or_path": p["model_name"],
            "r": p["lora_r"],
            "lora_alpha": p["lora_alpha"],
            "target_modules": sorted(p["lora_target_modules"]),
        }
        adapter.update(owner.adapter_overrides)
        if owner.write_adapter:
            (out / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
            (out / "adapter_model.safetensors").write_bytes(b"fake adapter weights")
        (out / "checkpoint-250").mkdir()
        (out / "checkpoint-250" / "optimizer.pt").write_bytes(b"fake optimizer")
        history = owner.losses if owner.losses is not None else losses_down(min(steps, 6) if steps > 1 else 1)
        last = history[-1][1] if history else float("nan")
        return {
            "final_loss": owner.final_loss if owner.final_loss is not None else last,
            "steps": owner.steps if owner.steps is not None else steps,
            "loss_history": history,
            "rows_seen": owner.rows_seen if owner.rows_seen is not None else rows,
        }


class FakeFactory:
    def __init__(self, **options):
        self.calls: list[dict] = []
        self.fail_with = None
        self.losses = None
        self.final_loss = None
        self.steps = None
        self.rows_seen = None
        self.write_adapter = True
        self.adapter_overrides: dict = {}
        for key, value in options.items():
            setattr(self, key, value)

    def __call__(self, params):
        self.calls.append(dict(params))
        return FakeTrainer(params, self)


class FakeMerge:
    def __init__(self, error=None):
        self.calls: list[dict] = []
        self.error = error

    def __call__(self, base_model, adapter_dir, out_dir, *, dtype):
        adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
        self.calls.append(
            {"base_model": base_model, "adapter_dir": adapter_dir, "out_dir": out_dir, "dtype": dtype,
             "adapter_existed": (adapter_dir / "adapter_config.json").is_file()}
        )
        if self.error is not None:
            raise self.error
        out_dir.mkdir(parents=True)
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        (out_dir / "model.safetensors").write_bytes(b"fake merged weights")
        (out_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeTokenizer:
    """Word counts stand in for token counts: the user turns in the fixtures are about fifteen words."""

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return "\n".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False  # the template text already holds the special tokens
        return {"input_ids": list(range(len(text.split())))}


class FakeApi:
    def __init__(self, *, private=True, info_error=None, upload_error=None):
        self.private, self.info_error, self.upload_error = private, info_error, upload_error
        self.calls: list[tuple] = []

    def repo_info(self, repo_id, repo_type="model", **kwargs):
        self.calls.append(("repo_info", repo_id, repo_type))
        if self.info_error is not None:
            raise self.info_error
        return types.SimpleNamespace(private=self.private)

    def upload_folder(self, *, folder_path, repo_id, repo_type="model", **kwargs):
        self.calls.append(("upload_folder", str(folder_path), repo_id, repo_type))
        if self.upload_error is not None:
            raise self.upload_error

    def create_repo(self, *args, **kwargs):
        raise AssertionError("finetune.py must never create a repo")


class ApiFactory:
    def __init__(self, api):
        self.api, self.tokens = api, []

    def __call__(self, token):
        self.tokens.append(token)
        return self.api


def forbidden(*args, **kwargs):
    raise AssertionError("this must not be called")


# --- the world a run happens in -----------------------------------------------------------------------------------


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


class World:
    """A repository root with valid fixture data, the real base config, and fakes for everything heavy."""

    def __init__(self, tmp_path, *, name="repo"):
        self.root = tmp_path / name
        self.root.mkdir()
        self.train = [rec(lang, i) for lang in FOUR for i in range(10)]
        self.heldout = [rec(lang, 1000 + i) for lang in FOUR for i in range(3)]
        self.taskdev = [rec(lang, 2000 + i) for lang in FOUR for i in range(3)]
        ft = self.root / "data" / "ft"
        write_lines(ft / "train.jsonl", self.train)
        write_lines(ft / "heldout.jsonl", self.heldout)
        write_lines(ft / "taskdev.jsonl", self.taskdev)
        (self.root / "pipelines").mkdir()
        (self.root / "pipelines" / "ft_data.yaml").write_text(
            yaml.safe_dump({"extra": {"harmless_subjects": ["history"]}}), encoding="utf-8"
        )
        self.config = self.root / "configs" / "ft_lora.yaml"
        self.config.parent.mkdir()
        self.write_config()
        self.factory = FakeFactory()
        self.merge = FakeMerge()
        self.out = self.root / "results" / "finetune" / "ft_lora"

    def raw(self):
        return yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))

    def write_config(self, mutate=None):
        data = self.raw()
        if mutate:
            mutate(data)
        self.config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def rewrite(self, name, rows):
        write_lines(self.root / "data" / "ft" / name, rows)

    def run(self, *args, factory=None, merge=None, api_factory=None, tokenizer_loader=None, clock=None,
            versions=None, device_info=None, config=True):
        argv = (["--config", str(self.config)] if config else []) + list(args)
        return finetune.main(
            argv,
            trainer_factory=factory if factory is not None else self.factory,
            merge_fn=merge if merge is not None else self.merge,
            tokenizer_loader=tokenizer_loader if tokenizer_loader is not None else (lambda name: FakeTokenizer()),
            hf_api_factory=api_factory if api_factory is not None else forbidden,
            clock=clock if clock is not None else iter([100.0, 112.5]).__next__,
            versions=versions if versions is not None else (lambda: {"aligntune": "0.0-fake", "trl": None}),
            device_info=device_info if device_info is not None else (lambda: "FakeGPU"),
            root=self.root,
        )

    def summary(self, out=None):
        return json.loads((Path(out or self.out) / "finetune_summary.json").read_text(encoding="utf-8"))

    def files(self):
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*"))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture(autouse=True)
def _no_token_in_the_environment(monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def siblings(path):
    return sorted(p.name for p in Path(path).parent.iterdir()) if Path(path).parent.exists() else []


# --- the plan arithmetic and the loss check ----------------------------------------------------------------------


def test_plan_steps_is_rows_times_epochs_over_the_effective_batch_rounded_up():
    assert finetune.plan_steps(6000, 2, 1, 16) == 750
    assert finetune.plan_steps(6000, 3, 1, 16) == 1125
    assert finetune.plan_steps(40, 2, 1, 16) == 5
    assert finetune.plan_steps(41, 2, 1, 16) == 6
    assert finetune.plan_steps(100, 1, 4, 2) == 13


@pytest.mark.parametrize(
    "history,ok",
    [
        ([[1, 2.0], [2, 1.0]], True),
        ([[1, 2.0], [2, 1.5], [3, 1.4], [4, 0.9]], True),
        ([[1, 2.0], [2, 2.0]], False),  # flat
        ([[1, 1.0], [2, 2.0]], False),  # up
        ([[1, 0.5], [2, 0.6], [3, 1.0], [4, 1.5]], False),  # rising: the second half is higher
        ([[1, 2.0], [2, 0.1], [3, 0.5], [4, 0.9]], True),  # a dip and a partial recovery still ends below the start
        ([[1, 2.0]], False),  # one point cannot show a decrease
        ([], False),
        ([[1, 2.0], [2, float("nan")]], False),
        ([[1, 2.0], [2, float("inf")]], False),
        ([[1, float("-inf")], [2, 1.0]], False),
    ],
)
def test_loss_check(history, ok):
    good, reason = finetune.loss_check(history)
    assert good is ok
    assert reason  # always says why, also when it is fine


def test_loss_check_reasons_name_the_failure():
    assert "finite" in finetune.loss_check([[1, 1.0], [2, float("nan")]])[1]
    assert "decrease" in finetune.loss_check([[1, 1.0], [2, 1.0]])[1]
    assert "two" in finetune.loss_check([[1, 1.0]])[1]


def test_sample_history_keeps_first_and_last_and_at_most_the_limit():
    history = [[i, 100.0 - i] for i in range(1, 101)]
    sampled = finetune.sample_history(history, 5)
    assert len(sampled) == 5 and sampled[0] == history[0] and sampled[-1] == history[-1]
    assert [s[0] for s in sampled] == sorted(s[0] for s in sampled)
    assert finetune.sample_history(history[:3], 5) == history[:3]


# --- the call into AlignTune: every config key reaches the factory ------------------------------------------------


def set_in(data, dotted, value):
    node = data
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value


# config key -> (the keyword of create_sft_trainer it must change, a new value, the value that keyword must then hold,
# a mutation to apply first when the key is only read in some modes)
def _quantised(data):
    data["quantization"]["mode"] = "4bit"


REACHES = {
    "base_model": ("model_name", "FakeOrg/other-model", "FakeOrg/other-model", None),
    "lora.rank": ("lora_r", 8, 8, None),
    "lora.alpha": ("lora_alpha", 99, 99, None),
    "lora.dropout": ("lora_dropout", 0.25, 0.25, None),
    "lora.target_modules": ("lora_target_modules", ["q_proj", "v_proj"], ["q_proj", "v_proj"], None),
    "lora.bias": ("lora_bias", "all", "all", None),
    "quantization.mode": ("quantization", "4bit", {"load_in_4bit": True}, None),
    "quantization.quant_type": ("quantization", "fp4", {"bnb_4bit_quant_type": "fp4"}, _quantised),
    "quantization.double_quant": ("quantization", False, {"bnb_4bit_use_double_quant": False}, _quantised),
    "train.learning_rate": ("learning_rate", 3.0e-5, 3.0e-5, None),
    "train.epochs": ("num_epochs", 5, 5, None),
    "train.per_device_batch_size": ("batch_size", 2, 2, None),
    "train.gradient_accumulation_steps": ("gradient_accumulation_steps", 3, 3, None),
    "train.max_seq_length": ("max_seq_length", 4096, 4096, None),
    "train.warmup_ratio": ("warmup_ratio", 0.3, 0.3, None),
    "train.lr_scheduler": ("lr_scheduler", "linear", "linear", None),
    "train.weight_decay": ("weight_decay", 0.2, 0.2, None),
    "train.max_grad_norm": ("max_grad_norm", 0.5, 0.5, None),
    "train.gradient_checkpointing": ("gradient_checkpointing", False, False, None),
    "train.fp16": ("fp16", False, False, None),
    "train.bf16": ("bf16", True, True, lambda d: d["train"].update(fp16=False)),
    "train.seed": ("seed", 777, 777, None),
    "train.assistant_only_loss": ("assistant_only_loss", False, False, None),
    "train.logging_steps": ("logging_steps", 7, 7, None),
    "train.save_steps": ("save_steps", 33, 33, None),
    "train.save_total_limit": ("save_total_limit", 5, 5, None),
    "aligntune.device_map": ("device_map", "auto", "auto", None),
    "aligntune.group_by_length": ("group_by_length", True, True, None),
    "aligntune.load_best_model_at_end": ("load_best_model_at_end", True, True, None),
    "aligntune.eval_strategy": ("eval_strategy", "steps", "steps", None),
    "aligntune.loggers": ("loggers", ["tensorboard"], ["tensorboard"], None),
}


@pytest.mark.parametrize("key", sorted(REACHES))
def test_changing_a_config_value_changes_what_the_trainer_receives(tmp_path, key):
    keyword, new, expected, prepare = REACHES[key]

    base = World(tmp_path, name="base")
    assert base.run() == 0
    before = base.factory.calls[0]

    changed = World(tmp_path, name="changed")

    def mutate(data):
        if prepare:
            prepare(data)
        set_in(data, key, new)

    changed.write_config(mutate)
    assert changed.run() == 0, key
    after = changed.factory.calls[0]
    value = after[keyword]
    if isinstance(expected, dict):
        assert all(value[k] == v for k, v in expected.items()), (key, value)
    else:
        assert value == expected, key
    assert value != before.get(keyword), f"{key} did not change {keyword}"


def test_the_fixed_keywords_and_the_derived_ones(world):
    assert world.run() == 0
    params = world.factory.calls[0]
    raw = world.raw()
    assert params["backend"] == raw["aligntune"]["backend"] == "trl"
    assert params["use_peft"] is True
    assert params["precision"] == "fp16"  # derived from fp16 true / bf16 false; AlignTune's AUTO could pick bf16 on a T4
    assert params["data_seed"] == params["seed"] == raw["train"]["seed"]
    assert params["dataset_name"] == str(world.root / "data" / "ft" / "train.jsonl")
    assert Path(params["output_dir"]).name == raw["output"]["adapter_dir"]
    assert "quantization" not in params  # mode none: nothing quantised, nothing sent
    assert "max_steps" not in params  # not a smoke run


def test_the_precision_follows_the_flags(tmp_path):
    for fp16, bf16, expected in ((True, False, "fp16"), (False, True, "bf16"), (False, False, "fp32")):
        w = World(tmp_path, name=f"p{fp16}{bf16}")
        w.write_config(lambda d, f=fp16, b=bf16: d["train"].update(fp16=f, bf16=b))
        assert w.run() == 0
        assert w.factory.calls[0]["precision"] == expected


def test_four_bit_compute_dtype_follows_the_precision_flags(tmp_path):
    w = World(tmp_path)
    w.write_config(lambda d: d["quantization"].update(mode="4bit"))
    assert w.run() == 0
    q = w.factory.calls[0]["quantization"]
    assert q["load_in_4bit"] is True and q["bnb_4bit_compute_dtype"] == "float16"  # AlignTune's default is bfloat16
    w2 = World(tmp_path, name="eight")
    w2.write_config(lambda d: d["quantization"].update(mode="8bit"))
    assert w2.run() == 0
    assert w2.factory.calls[0]["quantization"] == {"load_in_8bit": True}


def test_build_trainer_params_is_a_plain_function_of_the_config():
    cfg = finetune.load_config(REAL_CONFIG)
    params = finetune.build_trainer_params(cfg, data_path=Path("d.jsonl"), adapter_dir=Path("a"), seed=cfg.train.seed, max_steps=None)
    again = finetune.build_trainer_params(cfg, data_path=Path("d.jsonl"), adapter_dir=Path("a"), seed=cfg.train.seed, max_steps=None)
    assert params == again
    smoke = finetune.build_trainer_params(cfg, data_path=Path("d.jsonl"), adapter_dir=Path("a"), seed=cfg.train.seed, max_steps=3)
    assert smoke["max_steps"] == 3 and smoke["logging_steps"] == 1  # every step is logged so a decrease can be seen


COVERED_BY_OTHER_TESTS = {
    # keys whose effect is not a trainer keyword; each names the test that proves it reaches the code
    "aligntune.backend",  # test_the_fixed_keywords_and_the_derived_ones
    "data.train", "data.heldout", "data.taskdev", "data.pipeline", "data.testset", "data.langs",  # the data tests below
    "output.adapter_dir", "output.merged_dir", "output.merge_dtype", "output.keep_checkpoints",  # the layout tests
    "checks.history_points",  # test_summary_history_is_sampled_to_the_configured_number_of_points
}


def test_every_key_of_the_real_config_has_a_test_that_it_reaches_the_code():
    leaves = set(finetune.flatten(yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))))
    leaves.discard("base_model")  # in REACHES
    missing = {k for k in leaves if k not in REACHES and k not in COVERED_BY_OTHER_TESTS}
    assert not missing, f"config keys no test follows into the code: {sorted(missing)}"
    stale = {k for k in set(REACHES) | COVERED_BY_OTHER_TESTS if k not in leaves and k != "base_model"}
    assert not stale, f"tests name keys the real config no longer has: {sorted(stale)}"


def test_no_hyperparameter_literal_in_the_module():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    allowed = {0, 1, 2}  # exit codes, indices, halves; every training value lives in the YAML
    found = sorted(
        {
            (node.lineno, node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and node.value not in allowed
        }
    )
    assert not found, f"numeric literals in src/msd/finetune.py (put them in the config): {found}"


def test_the_module_names_none_of_the_configs_values():
    text = MODULE.read_text(encoding="utf-8")
    for value in ("20261105", "2048", "tiny-aya-global", "q_proj"):
        assert value not in text, f"{value!r} is a config value and must not be written in the module"


# --- dry run -----------------------------------------------------------------------------------------------------


def test_dry_run_prints_the_plan_and_writes_nothing(world, capsys):
    before = world.files()
    assert world.run("--dry-run", factory=forbidden, merge=forbidden) == 0
    out = capsys.readouterr().out
    assert world.files() == before  # nothing written anywhere
    assert not world.out.exists()
    assert "dry run" in out
    assert "rows per language: en 10, hi 10, ar 10, sw 10" in out
    assert "planned optimizer steps: 5 (40 rows x 2 epochs / effective batch 16)" in out
    assert "sha256" in out
    train_sha = hashlib.sha256((world.root / "data" / "ft" / "train.jsonl").read_bytes()).hexdigest()
    assert train_sha in out
    assert "token lengths: min" in out and "0 over max_seq_length 2048" in out


def test_dry_run_needs_neither_torch_nor_aligntune(world, monkeypatch, capsys):
    for name in HEAVY:
        monkeypatch.setitem(sys.modules, name, None)  # any import of these raises ImportError
    code = finetune.main(["--config", str(world.config), "--dry-run"], root=world.root)
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "token lengths: not estimated" in out.out
    assert "planned optimizer steps: 5" in out.out
    assert not world.out.exists()


def test_dry_run_with_a_failing_tokenizer_still_succeeds_and_says_why(world, capsys):
    def broken(name):
        raise OSError(f"cannot reach the Hub with {TOKEN}")

    assert world.run("--dry-run", tokenizer_loader=broken) == 0
    out = capsys.readouterr()
    assert "token lengths: not estimated" in out.out
    assert TOKEN not in out.out + out.err


def test_dry_run_refuses_a_max_seq_length_below_the_longest_example(world, capsys):
    world.write_config(lambda d: d["train"].update(max_seq_length=5))
    assert world.run("--dry-run") == 1
    err = capsys.readouterr().err
    assert "max_seq_length" in err and "5" in err


def test_dry_run_still_validates_the_data(world, capsys):
    world.rewrite("train.jsonl", world.train + [rec("en", 500, item="security_studies/test/500")])
    assert world.run("--dry-run") == 1
    assert "security_studies" in capsys.readouterr().err


def test_dry_run_does_not_touch_the_hub_even_with_push_to(world, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert world.run("--dry-run", "--push-to", "org/name", api_factory=forbidden) == 0
    out = capsys.readouterr()
    assert "org/name" in out.out and "not checked" in out.out
    assert TOKEN not in out.out + out.err


def test_dry_run_checks_the_out_dir_like_a_real_run(world, capsys):
    world.out.mkdir(parents=True)
    (world.out / "something").write_text("x")
    assert world.run("--dry-run") == 2
    assert "not empty" in capsys.readouterr().err


def test_dry_run_with_max_steps_shows_the_smoke_plan(world, capsys):
    assert world.run("--dry-run", "--max-steps", "3") == 0
    out = capsys.readouterr().out
    assert "smoke" in out and "3" in out and "ft_lora-smoke" in out


# --- the data is refused before any training ---------------------------------------------------------------------


def refused(world, capsys, *args):
    assert world.run(*args) == 1
    err = capsys.readouterr().err
    assert world.factory.calls == [] and world.merge.calls == []  # nothing trained
    assert not world.out.exists()
    assert not any(n.endswith(".partial") for n in siblings(world.out))
    return err


def test_a_subject_off_the_allowlist_is_refused(world, capsys):
    world.rewrite("train.jsonl", world.train + [rec("en", 500, item="security_studies/test/500")])
    err = refused(world, capsys)
    assert "security_studies" in err and "allowlist" in err


def test_overlap_with_heldout_is_refused(world, capsys):
    world.rewrite("train.jsonl", world.train + [rec("en", 1000)])  # the heldout en item, same prompt
    err = refused(world, capsys)
    assert "overlap" in err


def test_overlap_with_taskdev_is_refused(world, capsys):
    world.rewrite("train.jsonl", world.train + [rec("en", 2000)])
    err = refused(world, capsys)
    assert "overlap" in err and "taskdev" in err


def test_a_missing_language_is_refused(world, capsys):
    world.rewrite("train.jsonl", [r for r in world.train if r["lang"] != "sw"])
    err = refused(world, capsys)
    assert "sw" in err and "missing_lang" in err


def test_a_refusal_in_the_training_answers_is_refused(world, capsys):
    bad = rec("en", 600, assistant="I'm sorry, but I can't help with that.")
    world.rewrite("train.jsonl", world.train + [bad])
    err = refused(world, capsys)
    assert "refusal" in err


def test_overlap_with_the_safety_test_set_is_refused(world, capsys):
    prompt = lambda i, split, text: Prompt(id=f"en-{split}-{i}", lang="en", kind="harmful", text=text, split=split, source="s", license="mit")  # noqa: E731
    leaked = world.train[3]["messages"][0]["content"]
    write_jsonl(world.root / "data" / "testset" / "en" / "dev.jsonl", [prompt(0, "dev", leaked)])
    write_jsonl(world.root / "data" / "testset" / "en" / "report.jsonl", [prompt(0, "report", "an unrelated harmful report prompt")])
    err = refused(world, capsys)
    assert "safety" in err and "overlap" in err


def test_a_clean_safety_test_set_is_checked_and_passes(world, capsys):
    prompt = lambda i, split, text: Prompt(id=f"en-{split}-{i}", lang="en", kind="harmful", text=text, split=split, source="s", license="mit")  # noqa: E731
    write_jsonl(world.root / "data" / "testset" / "en" / "dev.jsonl", [prompt(0, "dev", "one dev prompt")])
    write_jsonl(world.root / "data" / "testset" / "en" / "report.jsonl", [prompt(0, "report", "one report prompt")])
    assert world.run() == 0
    assert world.summary()["data"]["safety_checked"] is True


def test_missing_pipeline_file_is_refused_not_skipped(world, capsys):
    (world.root / "pipelines" / "ft_data.yaml").unlink()
    err = refused(world, capsys)
    assert "ft_data.yaml" in err


def test_a_missing_train_file_is_refused(world, capsys):
    (world.root / "data" / "ft" / "train.jsonl").unlink()
    err = refused(world, capsys)
    assert "train.jsonl" in err


def test_a_malformed_json_line_is_refused(world, capsys):
    path = world.root / "data" / "ft" / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    refused(world, capsys)


def test_absent_heldout_and_taskdev_files_are_noted_and_the_run_goes_on(world, capsys):
    (world.root / "data" / "ft" / "heldout.jsonl").unlink()
    (world.root / "data" / "ft" / "taskdev.jsonl").unlink()
    assert world.run() == 0
    out = capsys.readouterr()
    assert "heldout" in out.err and "NOT checked" in out.err
    data = world.summary()["data"]
    assert data["heldout_checked"] is False and data["taskdev_checked"] is False


def test_present_heldout_and_taskdev_files_are_checked_and_recorded(world):
    assert world.run() == 0
    data = world.summary()["data"]
    assert data["heldout_checked"] is True and data["taskdev_checked"] is True and data["safety_checked"] is False


def test_data_override_replaces_the_config_path_and_is_hashed(world):
    other = world.root / "elsewhere" / "mine.jsonl"
    write_lines(other, [r for r in world.train if int(r["id"].rsplit("/", 1)[1]) < 9])  # 9 per language
    assert world.run("--data", str(other)) == 0
    assert world.factory.calls[0]["dataset_name"] == str(other)
    summary = world.summary()
    assert summary["data"]["train"]["sha256"] == hashlib.sha256(other.read_bytes()).hexdigest()
    assert summary["data"]["train"]["rows"] == 36


def test_the_training_data_hash_and_row_counts_are_in_the_summary(world):
    assert world.run() == 0
    train = world.summary()["data"]["train"]
    assert train["sha256"] == hashlib.sha256((world.root / "data" / "ft" / "train.jsonl").read_bytes()).hexdigest()
    assert train["rows"] == 40 and train["rows_per_language"] == {"en": 10, "hi": 10, "ar": 10, "sw": 10}
    assert train["path"] == "data/ft/train.jsonl"


# --- the run, smoke naming, merging ------------------------------------------------------------------------------


def test_a_full_run_writes_the_adapter_the_merged_model_and_the_summary(world):
    assert world.run() == 0
    assert (world.out / "adapter" / "adapter_config.json").is_file()
    assert (world.out / "merged" / "model.safetensors").is_file() and (world.out / "merged" / "tokenizer.json").is_file()
    assert (world.out / "finetune_summary.json").is_file()
    assert siblings(world.out) == ["ft_lora"]  # no partial folder left next to it
    (call,) = world.merge.calls
    assert call["base_model"] == "CohereLabs/tiny-aya-global"
    assert call["dtype"] == "float16"
    assert call["adapter_existed"] is True  # the adapter was on disk when the merge was called
    assert call["adapter_dir"].name == "adapter" and call["out_dir"].name == "merged"
    assert call["adapter_dir"].parent == call["out_dir"].parent  # both inside the same (partial) output folder
    assert world.summary()["smoke"] is False and world.summary()["merged"] is True


def test_the_merge_receives_the_adapter_folder_the_trainer_wrote(world):
    assert world.run() == 0
    trainer_output = Path(world.factory.calls[0]["output_dir"])
    assert world.merge.calls[0]["adapter_dir"] == trainer_output


def test_checkpoint_folders_are_removed_unless_kept(tmp_path):
    a = World(tmp_path, name="a")
    assert a.run() == 0
    assert not (a.out / "adapter" / "checkpoint-250").exists()
    b = World(tmp_path, name="b")
    b.write_config(lambda d: d["output"].update(keep_checkpoints=True))
    assert b.run() == 0
    assert (b.out / "adapter" / "checkpoint-250" / "optimizer.pt").is_file()


def test_folder_names_inside_out_come_from_the_config(tmp_path):
    w = World(tmp_path)
    w.write_config(lambda d: d["output"].update(adapter_dir="lora", merged_dir="a1_fp16"))
    assert w.run() == 0
    assert (w.out / "lora" / "adapter_config.json").is_file() and (w.out / "a1_fp16" / "config.json").is_file()
    assert w.merge.calls[0]["dtype"] == "float16"


def test_merge_dtype_comes_from_the_config(tmp_path):
    w = World(tmp_path)
    w.write_config(lambda d: d["output"].update(merge_dtype="bfloat16"))
    assert w.run() == 0
    assert w.merge.calls[0]["dtype"] == "bfloat16"
    assert w.summary()["merge_dtype"] == "bfloat16"


def test_smoke_run_is_named_and_flagged_and_does_not_merge_by_default(world):
    assert world.run("--max-steps", "3") == 0
    smoke_out = world.root / "results" / "finetune" / "ft_lora-smoke"
    assert smoke_out.is_dir() and not world.out.exists()
    params = world.factory.calls[0]
    assert params["max_steps"] == 3 and params["logging_steps"] == 1
    assert world.merge.calls == []
    summary = world.summary(smoke_out)
    assert summary["smoke"] is True and summary["merged"] is False and summary["steps"] == 3
    assert summary["max_steps"] == 3
    assert not (smoke_out / "merged").exists()


def test_smoke_run_can_merge_once_with_the_merge_flag(world):
    assert world.run("--max-steps", "3", "--merge") == 0
    assert len(world.merge.calls) == 1
    assert (world.root / "results" / "finetune" / "ft_lora-smoke" / "merged" / "config.json").is_file()


def test_no_merge_leaves_only_the_adapter(world):
    assert world.run("--no-merge") == 0
    assert world.merge.calls == [] and not (world.out / "merged").exists()
    assert world.summary()["merged"] is False


def test_an_explicit_out_wins_over_both_defaults(world, tmp_path):
    out = tmp_path / "somewhere" / "a1"
    assert world.run("--out", str(out), "--max-steps", "3") == 0
    assert (out / "finetune_summary.json").is_file()
    assert not (world.root / "results").exists()


@pytest.mark.parametrize("value", ["0", "-3", "x", "1.5"])
def test_max_steps_must_be_a_positive_integer(world, value, capsys):
    assert world.run("--max-steps", value) == 2
    assert world.factory.calls == []


def test_seed_override_reaches_the_trainer_and_the_summary(world):
    assert world.run("--seed", "123") == 0
    params = world.factory.calls[0]
    assert params["seed"] == 123 and params["data_seed"] == 123
    assert world.summary()["seed"] == 123
    assert world.summary()["overrides"]["seed"] == 123


def test_seed_default_comes_from_the_config(world):
    assert world.run() == 0
    assert world.summary()["seed"] == world.raw()["train"]["seed"]
    assert world.summary()["overrides"] == {"data": None, "max_steps": None, "seed": None}


@pytest.mark.parametrize("value", ["-1", "x", "1.5"])
def test_seed_must_be_a_non_negative_integer(world, value):
    assert world.run("--seed", value) == 2


# --- checks after training, before the merge -----------------------------------------------------------------------


def failed_run(world, capsys, **factory_options):
    world.factory = FakeFactory(**factory_options)
    assert world.run() == 1
    err = capsys.readouterr().err
    assert not world.out.exists()
    assert not any(name.endswith(".partial") for name in siblings(world.out))
    return err


def test_nan_loss_fails_the_run_and_nothing_is_merged_or_kept(world, capsys):
    err = failed_run(world, capsys, losses=[[1, 2.0], [2, float("nan")], [3, 1.0], [4, 0.9]])
    assert "finite" in err
    assert world.merge.calls == []


def test_infinite_final_loss_fails_the_run(world, capsys):
    err = failed_run(world, capsys, final_loss=float("inf"))
    assert "finite" in err and world.merge.calls == []


def test_a_loss_that_never_decreased_fails_the_run(world, capsys):
    err = failed_run(world, capsys, losses=[[1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0]])
    assert "decrease" in err and world.merge.calls == []


def test_the_loss_history_is_printed_when_the_check_fails_so_the_run_is_not_a_blind_loss(world, capsys):
    err = failed_run(world, capsys, losses=[[1, 1.0], [2, 1.25], [3, 1.5], [4, 2.0]])
    assert "1.25" in err and "2" in err


def test_one_logged_loss_cannot_show_a_decrease(world, capsys):
    world.factory = FakeFactory()
    assert world.run("--max-steps", "1") == 1
    assert "two" in capsys.readouterr().err


def test_no_logged_loss_fails_the_run(world, capsys):
    failed_run(world, capsys, losses=[], final_loss=0.5)


def test_rows_the_trainer_did_not_see_fail_the_run(world, capsys):
    err = failed_run(world, capsys, rows_seen=39)
    assert "39" in err and "40" in err and world.merge.calls == []


def test_a_trainer_that_does_not_report_rows_is_noted_not_refused(world):
    class Silent(FakeFactory):
        def __call__(self, params):
            trainer = super().__call__(params)
            original = trainer.train

            def train():
                result = original()
                result.pop("rows_seen")
                return result

            trainer.train = train
            return trainer

    world.factory = Silent()
    assert world.run() == 0
    assert world.summary()["rows_seen"] is None


def test_a_step_count_far_from_the_plan_fails_the_run(world, capsys):
    err = failed_run(world, capsys, steps=2)  # planned 5
    assert "steps" in err and "5" in err


def test_a_step_count_one_off_per_epoch_is_tolerated(world):
    world.factory = FakeFactory(steps=6)  # planned 5, two epochs
    assert world.run() == 0


def test_an_adapter_with_the_wrong_rank_fails_the_run(world, capsys):
    err = failed_run(world, capsys, adapter_overrides={"r": 8})
    assert "rank" in err or "'r'" in err
    assert world.merge.calls == []


def test_an_adapter_with_other_target_modules_fails_the_run(world, capsys):
    err = failed_run(world, capsys, adapter_overrides={"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    assert "target_modules" in err


def test_an_adapter_for_another_base_model_fails_the_run(world, capsys):
    err = failed_run(world, capsys, adapter_overrides={"base_model_name_or_path": "Someone/else"})
    assert "base_model" in err


def test_a_missing_adapter_fails_the_run(world, capsys):
    err = failed_run(world, capsys, write_adapter=False)
    assert "adapter_config.json" in err


def test_a_longer_example_than_max_seq_length_is_refused_before_training(world, capsys):
    world.write_config(lambda d: d["train"].update(max_seq_length=5))
    assert world.run() == 1
    assert world.factory.calls == []
    assert "max_seq_length" in capsys.readouterr().err


def test_the_token_count_uses_the_chat_template_text_without_adding_special_tokens(world):
    seen = []

    class Spy(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False):
            seen.append([m["role"] for m in messages])
            return super().apply_chat_template(messages, tokenize=tokenize)

    assert world.run(tokenizer_loader=lambda name: Spy()) == 0
    assert seen and all(roles == ["user", "assistant"] for roles in seen)


def test_the_tokenizer_is_loaded_for_the_base_model(world):
    names = []

    def loader(name):
        names.append(name)
        return FakeTokenizer()

    assert world.run(tokenizer_loader=loader) == 0
    assert names == ["CohereLabs/tiny-aya-global"]


# --- atomic output ------------------------------------------------------------------------------------------------


def test_a_trainer_that_raises_leaves_nothing_behind(world, capsys):
    world.factory = FakeFactory(fail_with=RuntimeError("CUDA out of memory"))
    assert world.run() == 1
    assert "CUDA out of memory" in capsys.readouterr().err
    assert not world.out.exists()
    assert not any(n.endswith(".partial") for n in siblings(world.out))
    assert world.merge.calls == []


def test_a_merge_that_raises_leaves_nothing_behind(world, capsys):
    world.merge = FakeMerge(error=MemoryError("not enough RAM"))
    assert world.run() == 1
    err = capsys.readouterr().err
    assert "not enough RAM" in err
    assert not world.out.exists()
    assert not any(n.endswith(".partial") for n in siblings(world.out))


def test_a_trainer_factory_that_cannot_be_built_leaves_nothing_behind(world, capsys):
    def broken(params):
        raise ImportError("No module named 'aligntune'")

    assert world.run(factory=broken) == 1
    assert not world.out.exists()


def test_a_non_empty_out_is_a_usage_error_and_nothing_runs(world, capsys):
    world.out.mkdir(parents=True)
    (world.out / "keep.txt").write_text("keep")
    assert world.run() == 2
    assert "not empty" in capsys.readouterr().err
    assert world.factory.calls == [] and (world.out / "keep.txt").read_text() == "keep"


def test_an_out_inside_the_data_folder_is_refused(world, capsys):
    assert world.run("--out", str(world.root / "data" / "ft")) == 2
    assert world.factory.calls == []


def test_an_empty_existing_out_is_fine(world):
    world.out.mkdir(parents=True)
    assert world.run() == 0
    assert (world.out / "finetune_summary.json").is_file()


def test_a_stale_partial_folder_is_named_not_deleted(world, capsys):
    stale = world.out.parent / ".ft_lora.deadbeef.partial"
    stale.mkdir(parents=True)
    (stale / "x").write_text("leftover")
    assert world.run() == 0
    assert ".ft_lora.deadbeef.partial" in capsys.readouterr().err
    assert (stale / "x").read_text() == "leftover"


# --- the summary --------------------------------------------------------------------------------------------------


def test_summary_records_the_run(world):
    assert world.run() == 0
    s = world.summary()
    assert s["smoke"] is False and s["loss_ok"] is True and s["merged"] is True
    assert s["base_model"] == "CohereLabs/tiny-aya-global"
    assert s["config"] == {
        "path": "configs/ft_lora.yaml",
        "sha256": hashlib.sha256(world.config.read_bytes()).hexdigest(),
    }
    assert s["steps"] == 5 and s["planned_steps"] == 5
    assert s["wall_clock_seconds"] == 12.5
    assert s["device"] == "FakeGPU"
    assert s["libraries"] == {"aligntune": "0.0-fake", "trl": None}
    assert s["seed"] == world.raw()["train"]["seed"]
    assert s["rows_seen"] == 40
    assert s["final_loss"] == pytest.approx(s["loss_history"][-1][1])
    assert s["loss_history"][0][1] > s["loss_history"][-1][1]
    assert s["merge_dtype"] == "float16"
    assert s["push_to"] is None


def test_summary_history_is_sampled_to_the_configured_number_of_points(world):
    world.factory = FakeFactory(losses=[[i, 50.0 - i / 4] for i in range(1, 101)])
    world.write_config(lambda d: d["checks"].update(history_points=5))
    assert world.run("--no-merge") == 0
    history = world.summary()["loss_history"]
    assert len(history) == 5 and history[0] == [1, 49.75] and history[-1] == [100, 25.0]


def test_summary_has_no_token_and_no_absolute_path(world, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HOME", str(world.root.parent))
    assert world.run() == 0
    text = (world.out / "finetune_summary.json").read_text(encoding="utf-8")
    assert TOKEN not in text
    assert str(world.root) not in text and str(world.root.parent) not in text
    assert "/Users/" not in text and "/home/" not in text and "/private/" not in text and "/tmp/" not in text


def test_summary_shows_a_config_outside_the_root_by_name_only(world, tmp_path):
    elsewhere = tmp_path / "outside" / "mine.yaml"
    elsewhere.parent.mkdir()
    elsewhere.write_text(world.config.read_text(encoding="utf-8"), encoding="utf-8")
    world.config = elsewhere
    out = world.root / "o"
    assert world.run("--out", str(out)) == 0
    text = (out / "finetune_summary.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and world.summary(out)["config"]["path"] == "mine.yaml"


def test_identical_fake_runs_give_byte_identical_summaries(tmp_path):
    a, b = World(tmp_path, name="one"), World(tmp_path, name="two")
    assert a.run() == 0 and b.run() == 0
    assert (a.out / "finetune_summary.json").read_bytes() == (b.out / "finetune_summary.json").read_bytes()


def test_summary_is_sorted_json_with_a_trailing_newline(world):
    assert world.run() == 0
    text = (world.out / "finetune_summary.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    data = json.loads(text)
    assert list(data) == sorted(data)


def test_the_summary_says_what_is_unverified(world):
    assert world.run() == 0
    assert "UNVERIFIED" in world.summary()["note"]


def test_the_real_device_and_versions_helpers_never_raise_without_the_libraries(monkeypatch):
    for name in HEAVY:
        monkeypatch.setitem(sys.modules, name, None)
    assert isinstance(finetune.default_device_info(), str)
    versions = finetune.library_versions()
    assert set(versions) >= {"aligntune", "trl", "transformers", "peft", "torch"}
    assert all(v is None or isinstance(v, str) for v in versions.values())


# --- pushing to the Hub -------------------------------------------------------------------------------------------


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)


def test_without_push_to_the_hub_is_never_touched(world, token):
    assert world.run(api_factory=forbidden) == 0


def test_push_to_a_private_repo_uploads_the_merged_folder_once(world, token, capsys):
    api = FakeApi(private=True)
    factory = ApiFactory(api)
    assert world.run("--push-to", "my-org/a1-fp16", api_factory=factory) == 0
    assert factory.tokens == [TOKEN]
    assert api.calls[0] == ("repo_info", "my-org/a1-fp16", "model")
    uploads = [c for c in api.calls if c[0] == "upload_folder"]
    assert uploads == [("upload_folder", str(world.out / "merged"), "my-org/a1-fp16", "model")]
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert world.summary()["push_to"] == "my-org/a1-fp16"


def test_the_repo_is_checked_before_training_starts(world, token):
    api = FakeApi(private=False)
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(api)) == 1
    assert world.factory.calls == []  # a public repo is found in the first second, not after the training


def test_a_public_repo_is_refused_and_never_written_or_created(world, token, capsys):
    api = FakeApi(private=False)
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(api)) == 1
    assert [c[0] for c in api.calls] == ["repo_info"]
    err = capsys.readouterr().err
    assert "private" in err and "my-org/a1" in err
    assert not world.out.exists()


@pytest.mark.parametrize("private", [None, "yes", 1, "True"])
def test_only_a_boolean_true_counts_as_private(world, token, private):
    api = FakeApi(private=private)
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(api)) == 1
    assert not any(c[0] == "upload_folder" for c in api.calls)


def test_a_repo_that_cannot_be_read_is_refused_and_the_token_is_scrubbed(world, token, capsys):
    api = FakeApi(info_error=RuntimeError(f"401 for token {TOKEN}: repository not found"))
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(api)) == 1
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert "my-org/a1" in out.err and "repository not found" in out.err
    assert world.factory.calls == []
    assert [c[0] for c in api.calls] == ["repo_info"]  # no creation attempt, no upload


def test_an_upload_failure_keeps_the_local_result_and_scrubs_the_token(world, token, capsys):
    api = FakeApi(upload_error=RuntimeError(f"network down, token {TOKEN}"))
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(api)) == 1
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert "upload" in out.err and "kept" in out.err
    assert (world.out / "merged" / "config.json").is_file() and (world.out / "finetune_summary.json").is_file()


def test_push_to_without_hf_token_is_a_usage_error(world, capsys):
    assert world.run("--push-to", "my-org/a1", api_factory=forbidden) == 2
    assert "HF_TOKEN" in capsys.readouterr().err
    assert world.factory.calls == []


def test_only_hf_token_is_read_not_its_aliases(world, monkeypatch, capsys):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", TOKEN)
    assert world.run("--push-to", "my-org/a1", api_factory=forbidden) == 2
    assert TOKEN not in capsys.readouterr().err


@pytest.mark.parametrize("repo", ["no-slash", "a/b/c", "/name", "org/", "org/na me", "org/../x", "", "https://huggingface.co/o/n"])
def test_a_malformed_repo_id_is_a_usage_error(world, token, repo):
    assert world.run("--push-to", repo, api_factory=forbidden) == 2
    assert world.factory.calls == []


def test_push_to_needs_the_merged_model(world, token, capsys):
    assert world.run("--push-to", "my-org/a1", "--no-merge", api_factory=forbidden) == 2
    assert "merge" in capsys.readouterr().err


def test_a_smoke_run_is_never_pushed(world, token, capsys):
    assert world.run("--push-to", "my-org/a1", "--max-steps", "3", api_factory=forbidden) == 2
    assert "smoke" in capsys.readouterr().err
    assert world.factory.calls == []


def test_a_failed_training_is_never_pushed(world, token):
    api = FakeApi()
    world.factory = FakeFactory(losses=[[1, 1.0], [2, 1.0]])
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(api)) == 1
    assert [c[0] for c in api.calls] == ["repo_info"]


def test_the_token_is_never_printed_on_a_normal_run(world, token, capsys):
    assert world.run("--push-to", "my-org/a1", api_factory=ApiFactory(FakeApi())) == 0
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err


def test_scrub_replaces_tokens_from_the_environment_and_by_shape(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "plain-secret-without-prefix")
    text = finetune.scrub(f"a {TOKEN} b plain-secret-without-prefix c")
    assert TOKEN not in text and "plain-secret-without-prefix" not in text and text.count("<token>") == 2


# --- the command line ---------------------------------------------------------------------------------------------


def test_help_documents_the_exit_codes_and_every_option(capsys):
    assert finetune.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "Exit codes" in out
    for line in ("0 ", "1 ", "2 "):
        assert any(row.strip().startswith(line) for row in out.splitlines()), line
    for option in ("--config", "--data", "--out", "--dry-run", "--max-steps", "--merge", "--no-merge", "--push-to", "--seed"):
        assert option in out
    assert "UNVERIFIED" in out


def test_help_through_the_script():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "Exit codes" in result.stdout


def test_a_missing_config_option_is_a_usage_error(capsys):
    assert finetune.main([]) == 2
    assert "--config" in capsys.readouterr().err


def test_an_unknown_option_is_a_usage_error_and_a_token_in_it_is_scrubbed(world, capsys):
    assert world.run("--bogus", TOKEN) == 2
    assert TOKEN not in capsys.readouterr().err


def test_a_config_file_that_does_not_exist_is_a_config_problem(world, capsys):
    world.config = world.root / "configs" / "nope.yaml"
    assert world.run() == 1
    assert "nope.yaml" in capsys.readouterr().err
    assert world.factory.calls == []


def test_a_bad_config_value_is_a_config_problem_with_every_error_listed(world, capsys):
    world.write_config(lambda d: (d["train"].update(epochs=0), d["train"].update(learning_rate=-1.0)))
    assert world.run() == 1
    err = capsys.readouterr().err
    assert "train.epochs" in err and "train.learning_rate" in err
    assert world.factory.calls == []


def test_importing_the_module_loads_no_heavy_library():
    code = (
        "import sys; sys.path.insert(0, %r); import msd.finetune; "
        "bad = sorted(set(%r) & set(sys.modules)); assert not bad, bad"
    ) % (str(REPO / "src"), HEAVY)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_the_script_runs_a_dry_run_end_to_end_without_any_heavy_library(world):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(world.config), "--dry-run"],
        capture_output=True, text=True, timeout=120, cwd=world.root,
        env={"PATH": "/usr/bin:/bin", "HOME": str(world.root), "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "planned optimizer steps" in result.stdout


# --- the real AlignTune wrapper, against a fake aligntune module ---------------------------------------------------


class _FakeAlignTuneTrainer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        hf = types.SimpleNamespace(
            state=types.SimpleNamespace(
                global_step=7,
                log_history=[
                    {"loss": 2.0, "step": 1, "learning_rate": 1e-4},
                    {"eval_loss": 9.9, "step": 2},  # an entry without "loss" is skipped
                    {"loss": float("nan"), "step": 3},
                    {"train_runtime": 3.0, "step": 7},
                ],
            ),
            train_dataset=[0] * 38,
        )
        self.trainer = hf

    def train(self):
        return {"final_loss": 1.25, "steps": 7, "epochs": 2, "metrics": {}, "model_path": "x", "training_time": 3.0}


def install_fake_aligntune(monkeypatch):
    calls = []

    def create_sft_trainer(**kwargs):
        calls.append(kwargs)
        return _FakeAlignTuneTrainer(kwargs)

    module = types.ModuleType("aligntune.core.backend_factory")
    module.create_sft_trainer = create_sft_trainer
    for name in ("aligntune", "aligntune.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aligntune.core.backend_factory", module)
    return calls


def test_the_aligntune_wrapper_passes_the_keywords_through_and_reads_the_numbers_back(monkeypatch):
    calls = install_fake_aligntune(monkeypatch)
    trainer = finetune.aligntune_factory({"model_name": "m", "lora_r": 4})
    assert calls == [{"model_name": "m", "lora_r": 4}]  # exactly the keywords, nothing added
    result = trainer.train()
    assert result["final_loss"] == 1.25 and result["steps"] == 7 and result["rows_seen"] == 38
    history = result["loss_history"]
    assert history[0] == [1, 2.0] and len(history) == 2 and math.isnan(history[1][1])


def test_the_aligntune_wrapper_falls_back_on_the_trainer_state_for_the_step_count(monkeypatch):
    install_fake_aligntune(monkeypatch)
    trainer = finetune.aligntune_factory({})
    trainer.inner.train = lambda: {"final_loss": 1.0}
    assert trainer.train()["steps"] == 7  # state.global_step


def test_a_missing_aligntune_gives_an_install_hint(world, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "aligntune", None)
    monkeypatch.setitem(sys.modules, "aligntune.core", None)
    monkeypatch.setitem(sys.modules, "aligntune.core.backend_factory", None)
    code = finetune.main(
        ["--config", str(world.config)], tokenizer_loader=lambda n: FakeTokenizer(), merge_fn=world.merge, root=world.root,
        clock=iter([1.0, 2.0]).__next__, versions=lambda: {}, device_info=lambda: "x",
    )
    assert code == 1
    assert "pip install aligntune==" in capsys.readouterr().err
    assert not world.out.exists()


def test_the_default_merge_gives_an_install_hint_without_peft(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(finetune.FinetuneError, match="pip install"):
        finetune.default_merge("base", tmp_path / "adapter", tmp_path / "merged", dtype="float16")


def test_user_turn_helper_is_the_one_the_fixtures_use():
    assert user_turn("en", 1).startswith("What is the capital of country number 1?")
