"""The fine-tune config schema (src/msd/finetune.py) and the two real configs in configs/.

Pure Python on dicts and the two YAML files: no trainer, no data, no network.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from msd import finetune
from msd.config import MODEL_IDS

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
BASE_PATH = CONFIGS / "ft_lora.yaml"
STRONG_PATH = CONFIGS / "ft_lora_strong.yaml"


def raw(path=BASE_PATH):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def mutated(path_in_config, value, *, base=BASE_PATH):
    data = raw(base)
    node = data
    *parents, leaf = path_in_config.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value
    return data


def removed(path_in_config, *, base=BASE_PATH):
    data = raw(base)
    node = data
    *parents, leaf = path_in_config.split(".")
    for key in parents:
        node = node[key]
    del node[leaf]
    return data


# --- the two real files -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [BASE_PATH, STRONG_PATH], ids=lambda p: p.name)
def test_the_real_configs_load_and_validate(path):
    cfg = finetune.load_config(path)
    assert cfg.base_model == MODEL_IDS["A0"]
    assert cfg.quantization.mode == "none"  # plain fp16 LoRA: the choice is argued in the file
    assert cfg.train.fp16 is True and cfg.train.bf16 is False  # a T4 has no bf16
    assert cfg.train.assistant_only_loss is True
    assert cfg.train.gradient_checkpointing is True
    assert cfg.output.merge_dtype == "float16"
    assert cfg.aligntune.backend == "trl"
    assert cfg.data.langs == ["en", "hi", "ar", "sw"]


@pytest.mark.parametrize("path,comments", [(BASE_PATH, 20), (STRONG_PATH, 10)], ids=["base", "strong"])
def test_the_real_configs_explain_themselves_in_comments(path, comments):
    # the base file argues every value; the strong file argues only the four it changes and points at the base
    text = path.read_text(encoding="utf-8")
    assert text.count("#") >= comments, "the configs justify their values in comments"
    for word in ("QLoRA", "UNVERIFIED"):
        assert word in text


def test_strong_differs_from_base_only_where_documented():
    base, strong = raw(BASE_PATH), raw(STRONG_PATH)
    assert finetune.check_strong_relation(base, strong) == []
    differing = {k for k in finetune.flatten(base) if finetune.flatten(base)[k] != finetune.flatten(strong)[k]}
    assert differing == {"lora.rank", "lora.alpha", "train.learning_rate", "train.epochs"}


def test_strong_relation_names_every_way_it_is_broken():
    base = raw(BASE_PATH)
    weaker = copy.deepcopy(raw(STRONG_PATH))
    weaker["train"]["learning_rate"] = base["train"]["learning_rate"] / 2
    weaker["train"]["epochs"] = base["train"]["epochs"]
    weaker["lora"]["rank"] = base["lora"]["rank"]
    weaker["lora"]["alpha"] = base["lora"]["alpha"] + 1  # also breaks alpha = 2 x rank
    weaker["train"]["seed"] = base["train"]["seed"] + 1  # a difference nobody documented
    problems = " | ".join(finetune.check_strong_relation(base, weaker))
    for needle in ("learning_rate", "epochs", "rank", "alpha", "train.seed"):
        assert needle in problems


def test_strong_relation_wants_the_lora_scale_unchanged():
    base = raw(BASE_PATH)
    strong = raw(STRONG_PATH)
    strong["lora"]["alpha"] = strong["lora"]["rank"]  # scale 1 instead of 2
    assert any("alpha" in p for p in finetune.check_strong_relation(base, strong))


# --- valid values -----------------------------------------------------------------------------------------------


def test_parse_config_gives_attribute_access_to_every_value():
    cfg = finetune.parse_config(raw())
    assert cfg.lora.rank == 16 and cfg.lora.target_modules[0] == "q_proj"
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert cfg.aligntune.device_map == {"": 0}
    assert cfg.aligntune.eval_strategy == "no"  # the string, not YAML's False
    assert cfg.checks.history_points == 20


def test_an_integer_is_a_fine_learning_rate_bound_and_a_float_a_fine_weight_decay():
    data = mutated("train.weight_decay", 0)
    assert finetune.parse_config(data).train.weight_decay == 0
    data = mutated("train.max_grad_norm", 1)
    assert finetune.parse_config(data).train.max_grad_norm == 1


def test_four_bit_quantisation_is_accepted():
    cfg = finetune.parse_config(mutated("quantization.mode", "4bit"))
    assert cfg.quantization.mode == "4bit" and cfg.quantization.quant_type == "nf4"


# --- invalid values ---------------------------------------------------------------------------------------------

BAD = [
    # (dotted key, bad value, text the error must contain)
    ("train.epochs", "2", "train.epochs"),
    ("train.epochs", 2.5, "train.epochs"),
    ("train.epochs", True, "train.epochs"),
    ("train.epochs", 0, "train.epochs"),
    ("train.per_device_batch_size", 0, "per_device_batch_size"),
    ("train.gradient_accumulation_steps", -1, "gradient_accumulation_steps"),
    ("train.max_seq_length", 0, "max_seq_length"),
    ("train.learning_rate", "1e-4", "train.learning_rate"),  # PyYAML reads 1e-4 as text
    ("train.learning_rate", 0, "train.learning_rate"),
    ("train.learning_rate", -1.0e-4, "train.learning_rate"),
    ("train.learning_rate", 2.0, "train.learning_rate"),
    ("train.learning_rate", math.nan, "train.learning_rate"),
    ("train.learning_rate", math.inf, "train.learning_rate"),
    ("train.learning_rate", True, "train.learning_rate"),
    ("train.warmup_ratio", 1.0, "warmup_ratio"),
    ("train.warmup_ratio", -0.1, "warmup_ratio"),
    ("train.weight_decay", -0.01, "weight_decay"),
    ("train.max_grad_norm", 0, "max_grad_norm"),
    ("train.lr_scheduler", "bogus", "lr_scheduler"),
    ("train.fp16", "yes", "train.fp16"),
    ("train.gradient_checkpointing", 1, "gradient_checkpointing"),
    ("train.assistant_only_loss", "true", "assistant_only_loss"),
    ("train.seed", -1, "train.seed"),
    ("train.seed", 1.5, "train.seed"),
    ("train.seed", True, "train.seed"),
    ("train.logging_steps", 0, "logging_steps"),
    ("train.save_steps", 0, "save_steps"),
    ("train.save_total_limit", 0, "save_total_limit"),
    ("lora.rank", 0, "lora.rank"),
    ("lora.alpha", 0, "lora.alpha"),
    ("lora.dropout", 1.0, "lora.dropout"),
    ("lora.dropout", -0.1, "lora.dropout"),
    ("lora.bias", "weird", "lora.bias"),
    ("lora.target_modules", [], "target_modules"),
    ("lora.target_modules", "all-linear", "target_modules"),
    ("lora.target_modules", ["q_proj", 5], "target_modules"),
    ("lora.target_modules", ["q_proj", "q_proj"], "target_modules"),
    ("quantization.mode", "int4", "quantization.mode"),
    ("quantization.quant_type", "fp3", "quant_type"),
    ("quantization.double_quant", "yes", "double_quant"),
    ("base_model", "", "base_model"),
    ("base_model", 7, "base_model"),
    ("data.langs", [], "data.langs"),
    ("data.langs", ["en", "xx"], "data.langs"),
    ("data.langs", ["en", "en"], "data.langs"),
    ("data.train", "", "data.train"),
    ("data.heldout", 3, "data.heldout"),
    ("aligntune.backend", "vllm", "aligntune.backend"),
    ("aligntune.group_by_length", "no", "group_by_length"),
    ("aligntune.eval_strategy", False, "eval_strategy"),  # an unquoted no in YAML is False: say so
    ("aligntune.loggers", "none", "loggers"),
    ("aligntune.device_map", 3.5, "device_map"),
    ("output.merge_dtype", "float8", "merge_dtype"),
    ("output.adapter_dir", "a/b", "adapter_dir"),
    ("output.adapter_dir", "..", "adapter_dir"),
    ("output.merged_dir", "", "merged_dir"),
    ("output.keep_checkpoints", "no", "keep_checkpoints"),
    ("checks.history_points", 1, "history_points"),
    ("checks.label_check_rows", 0, "label_check_rows"),
    ("checks.label_check_rows", "8", "label_check_rows"),
    ("checks.label_tokens_min", 0, "label_tokens_min"),
    ("checks.label_tokens_min", 1.5, "label_tokens_min"),
    ("checks.label_tokens_max", 0, "label_tokens_max"),
    ("checks.label_tokens_max", True, "label_tokens_max"),
]


@pytest.mark.parametrize("key,value,needle", BAD, ids=[f"{k}={v!r}" for k, v, _ in BAD])
def test_a_bad_value_is_refused_and_named(key, value, needle):
    with pytest.raises(finetune.ConfigError) as exc:
        finetune.parse_config(mutated(key, value))
    assert needle in str(exc.value)


def test_fp16_and_bf16_together_are_refused():
    data = mutated("train.bf16", True)
    with pytest.raises(finetune.ConfigError, match="fp16.*bf16|bf16.*fp16"):
        finetune.parse_config(data)


def test_the_label_bounds_must_be_ordered():
    data = mutated("checks.label_tokens_min", 20)  # the base config's maximum is 16
    with pytest.raises(finetune.ConfigError, match="label_tokens_min.*label_tokens_max"):
        finetune.parse_config(data)


def test_the_real_configs_carry_the_label_check_values():
    for path in (BASE_PATH, STRONG_PATH):
        checks = finetune.load_config(path).checks
        assert (checks.label_check_rows, checks.label_tokens_min, checks.label_tokens_max) == (8, 1, 16)


def test_adapter_and_merged_folders_must_differ():
    data = mutated("output.merged_dir", raw()["output"]["adapter_dir"])
    with pytest.raises(finetune.ConfigError, match="adapter_dir.*merged_dir|differ"):
        finetune.parse_config(data)


def test_an_unknown_top_level_key_is_an_error():
    data = raw()
    data["learning_rate"] = 1.0e-4  # a natural slip: it belongs under train
    with pytest.raises(finetune.ConfigError, match="unknown key 'learning_rate'"):
        finetune.parse_config(data)


def test_an_unknown_nested_key_is_an_error_with_a_suggestion():
    data = raw()
    data["train"]["learnin_rate"] = data["train"].pop("learning_rate")
    with pytest.raises(finetune.ConfigError) as exc:
        finetune.parse_config(data)
    assert "unknown key 'train.learnin_rate'" in str(exc.value)
    assert "train.learning_rate" in str(exc.value)  # the missing key is reported too


@pytest.mark.parametrize(
    "key",
    ["base_model", "data", "lora.rank", "lora.target_modules", "train.learning_rate", "train.seed", "train.fp16",
     "aligntune.backend", "output.merge_dtype", "checks.history_points", "quantization.mode",
     "checks.label_check_rows", "checks.label_tokens_min", "checks.label_tokens_max"],
)
def test_a_missing_required_key_is_an_error(key):
    with pytest.raises(finetune.ConfigError, match=key.split(".")[-1]):
        finetune.parse_config(removed(key))


def test_every_error_is_reported_at_once():
    data = mutated("train.epochs", 0)
    data["train"]["learning_rate"] = -1.0
    data["bogus"] = 1
    with pytest.raises(finetune.ConfigError) as exc:
        finetune.parse_config(data)
    text = str(exc.value)
    assert "train.epochs" in text and "train.learning_rate" in text and "bogus" in text


def test_a_section_must_be_a_mapping():
    data = raw()
    data["train"] = 5
    with pytest.raises(finetune.ConfigError, match="train"):
        finetune.parse_config(data)


def test_the_top_level_must_be_a_mapping():
    with pytest.raises(finetune.ConfigError):
        finetune.parse_config([1, 2, 3])


def test_load_config_names_a_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(finetune.ConfigError, match="not found|No such file"):
        finetune.load_config(tmp_path / "nope.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("train: [unclosed\n", encoding="utf-8")
    with pytest.raises(finetune.ConfigError, match="not valid YAML"):
        finetune.load_config(broken)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(finetune.ConfigError):
        finetune.load_config(empty)


def test_flatten_gives_dotted_leaf_keys():
    flat = finetune.flatten({"a": {"b": 1, "c": {"d": [1, 2]}}, "e": 3})
    assert flat == {"a.b": 1, "a.c.d": [1, 2], "e": 3}
