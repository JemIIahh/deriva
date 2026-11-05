"""Tests for the streaming weight blend (msd.blend) on tiny safetensors files.

Nothing here touches the network or a real model. Needs torch (CPU): the optional `blend` extra.
"""

from __future__ import annotations

import gc
import json
import re
import struct
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from msd import blend  # noqa: E402
from msd.blend import blend_state_dicts  # noqa: E402

NAMES = [
    "embed.weight",
    "layers.0.attn.weight",
    "layers.0.norm.weight",
    "layers.1.attn.weight",
    "layers.1.norm.weight",
    "lm_head.weight",
]
SHAPES = {
    "embed.weight": (4, 3),
    "layers.0.attn.weight": (3, 3),
    "layers.0.norm.weight": (3,),
    "layers.1.attn.weight": (3, 3),
    "layers.1.norm.weight": (3,),
    "lm_head.weight": (4, 3),
}
INDEX = "model.safetensors.index.json"


# --- builders --------------------------------------------------------------------------------------------------


def make_tensors(dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    return {n: (torch.randn(SHAPES[n], generator=gen) * 3).to(dtype) for n in NAMES}


def two_shards(tensors):
    names = sorted(tensors)
    return {
        "model-00001-of-00002.safetensors": {n: tensors[n] for n in names[:3]},
        "model-00002-of-00002.safetensors": {n: tensors[n] for n in names[3:]},
    }


def three_shards(tensors):
    """A different split and different file names: A0's layout need not match A1's."""
    names = sorted(tensors)
    groups = (names[0::3], names[1::3], names[2::3])
    return {f"model-0000{i + 1}-of-00003.safetensors": {n: tensors[n] for n in g} for i, g in enumerate(groups)}


def one_file(tensors):
    return {"model.safetensors": dict(tensors)}


def write_model(root: Path, shards, *, metadata=None, label="a1", extra_files=True):
    """shards maps file name -> {tensor name: tensor}. A lone model.safetensors gets no index file."""
    root.mkdir(parents=True, exist_ok=True)
    meta = {"format": "pt"} if metadata is None else metadata
    for filename, tensors in shards.items():
        save_file(dict(tensors), str(root / filename), metadata=meta)
    if list(shards) != ["model.safetensors"]:
        weight_map = {n: f for f, ts in shards.items() for n in ts}
        total = sum(t.numel() * t.element_size() for ts in shards.values() for t in ts.values())
        (root / INDEX).write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2, sort_keys=True)
        )
    if extra_files:
        (root / "config.json").write_text(json.dumps({"who": label, "model_type": "tiny"}))
        (root / "generation_config.json").write_text(json.dumps({"who": label, "do_sample": False}))
        (root / "tokenizer.json").write_text(json.dumps({"who": label, "vocab": {"a": 0}}))
        (root / "tokenizer_config.json").write_text(json.dumps({"who": label}))
        (root / "chat_template.jinja").write_text("{{ messages }} " + label)


def read_model(root: Path):
    """Every tensor of a model directory (single file or shards), keyed by name."""
    out = {}
    for path in sorted(root.glob("*.safetensors")):
        out.update(load_file(str(path)))
    return out


def same_bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)
    )


def expected_blend(a1, a0, t):
    """The documented arithmetic: float32 (float64 for float64), cast back to the tensor's dtype."""
    cd = torch.float64 if a1.dtype == torch.float64 else torch.float32
    return (a1.to(cd) * (1.0 - t) + a0.to(cd) * t).to(a1.dtype)


@pytest.fixture
def setup(tmp_path):
    """bf16 A1 in two shards, A0 in three shards of another split; returns (a1_dir, a0_dir, out_dir, a1, a0)."""

    def build(dtype=torch.bfloat16, a1_layout=two_shards, a0_layout=three_shards, **kw):
        a1 = make_tensors(dtype, 1)
        a0 = make_tensors(dtype, 2)
        write_model(tmp_path / "a1", a1_layout(a1), label="a1", **kw)
        write_model(tmp_path / "a0", a0_layout(a0), label="a0", **kw)
        return tmp_path / "a1", tmp_path / "a0", tmp_path / "out", a1, a0

    return build


# --- the brief's four tests ------------------------------------------------------------------------------------


def test_blend_t0_equals_a1(setup):
    a1_dir, a0_dir, out, a1, _ = setup()
    blend_state_dicts(a1_dir, a0_dir, 0.0, out)
    got = read_model(out)
    assert set(got) == set(a1)
    assert all(same_bits(got[n], a1[n]) for n in a1)


def test_blend_t1_equals_a0(setup):
    a1_dir, a0_dir, out, _, a0 = setup()
    blend_state_dicts(a1_dir, a0_dir, 1.0, out)
    got = read_model(out)
    assert set(got) == set(a0)
    assert all(same_bits(got[n], a0[n]) for n in a0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_blend_midpoint_is_mean_and_dtype_preserved(setup, dtype):
    a1_dir, a0_dir, out, a1, a0 = setup(dtype=dtype)
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    got = read_model(out)
    for n in NAMES:
        assert got[n].dtype == dtype
        assert same_bits(got[n], ((a1[n].float() + a0[n].float()) / 2).to(dtype))
        # and within one rounding step of the plain mean
        assert torch.allclose(got[n].float(), (a1[n].float() + a0[n].float()) / 2, rtol=1e-2, atol=1e-2)


def test_shape_mismatch_raises(setup):
    a1_dir, a0_dir, out, a1, a0 = setup()
    a0 = dict(a0, **{"layers.0.attn.weight": torch.zeros(3, 4, dtype=torch.bfloat16)})
    for f in a0_dir.glob("*"):
        f.unlink()
    write_model(a0_dir, three_shards(a0), label="a0")
    with pytest.raises(ValueError, match=re.escape("layers.0.attn.weight")) as err:
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert "shape" in str(err.value)
    assert not out.exists()


# --- the arithmetic --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0.1, 0.25, 0.75, 0.9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_blend_is_float32_arithmetic_cast_back(setup, dtype, t):
    a1_dir, a0_dir, out, a1, a0 = setup(dtype=dtype)
    blend_state_dicts(a1_dir, a0_dir, t, out)
    got = read_model(out)
    for n in NAMES:
        assert same_bits(got[n], expected_blend(a1[n], a0[n], t))


def test_float64_is_computed_in_float64(setup):
    a1_dir, a0_dir, out, a1, a0 = setup(dtype=torch.float64)
    blend_state_dicts(a1_dir, a0_dir, 0.3, out)
    got = read_model(out)
    for n in NAMES:
        assert got[n].dtype == torch.float64
        assert same_bits(got[n], expected_blend(a1[n], a0[n], 0.3))


@pytest.mark.parametrize("t", [0, 1, 0.5])
def test_integer_t_is_accepted(setup, t):
    a1_dir, a0_dir, out, _, _ = setup()
    blend_state_dicts(a1_dir, a0_dir, t, out)
    assert (out / INDEX).exists()


def test_odd_shapes_scalar_and_empty(tmp_path):
    a1 = {"scalar": torch.tensor(2.0), "empty": torch.zeros(0, 3), "vec": torch.arange(5, dtype=torch.float32)}
    a0 = {"scalar": torch.tensor(4.0), "empty": torch.zeros(0, 3), "vec": torch.arange(5, dtype=torch.float32) * 2}
    write_model(tmp_path / "a1", one_file(a1))
    write_model(tmp_path / "a0", one_file(a0), label="a0")
    blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 0.5, tmp_path / "out")
    got = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert got["scalar"].shape == () and got["scalar"].item() == 3.0
    assert got["empty"].shape == (0, 3)
    assert same_bits(got["vec"], torch.arange(5, dtype=torch.float32) * 1.5)


# --- layouts, index, metadata, other files ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1_layout, a0_layout",
    [
        (two_shards, three_shards),
        (three_shards, two_shards),
        (one_file, two_shards),
        (two_shards, one_file),
        (one_file, one_file),
        (two_shards, two_shards),
    ],
)
def test_layouts_may_differ_and_output_keeps_a1_filenames(setup, a1_layout, a0_layout):
    a1_dir, a0_dir, out, a1, a0 = setup(a1_layout=a1_layout, a0_layout=a0_layout)
    blend_state_dicts(a1_dir, a0_dir, 0.25, out)
    out_shards = sorted(p.name for p in out.glob("*.safetensors"))
    assert out_shards == sorted(p.name for p in a1_dir.glob("*.safetensors"))
    assert (out / INDEX).exists() == (a1_dir / INDEX).exists()
    got = read_model(out)
    assert set(got) == set(NAMES)
    for n in NAMES:
        assert same_bits(got[n], expected_blend(a1[n], a0[n], 0.25))
    # each output shard holds the same tensor names as the A1 shard of the same name
    for p in a1_dir.glob("*.safetensors"):
        assert set(load_file(str(out / p.name))) == set(load_file(str(p)))


def test_index_json_is_correct(setup):
    a1_dir, a0_dir, out, a1, _ = setup(dtype=torch.float16)
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    index = json.loads((out / INDEX).read_text())
    assert set(index) == {"metadata", "weight_map"}
    assert index["weight_map"] == json.loads((a1_dir / INDEX).read_text())["weight_map"]
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in a1.values())
    # every weight_map entry points at a shard that really holds that tensor, and nothing else is missing
    for name, filename in index["weight_map"].items():
        with safe_open(str(out / filename), framework="pt") as f:
            assert name in f.keys()
    assert sum(len(load_file(str(p))) for p in out.glob("*.safetensors")) == len(NAMES)


def test_index_keeps_other_a1_index_metadata(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    index = json.loads((a1_dir / INDEX).read_text())
    index["metadata"]["total_parameters"] = 123
    index["metadata"]["total_size"] = 1  # a stale value must be replaced, not copied
    (a1_dir / INDEX).write_text(json.dumps(index))
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    got = json.loads((out / INDEX).read_text())["metadata"]
    assert got["total_parameters"] == 123
    assert got["total_size"] != 1


def test_single_file_output_has_no_index(setup):
    a1_dir, a0_dir, out, _, _ = setup(a1_layout=one_file)
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert (out / "model.safetensors").exists()
    assert not (out / INDEX).exists()


def test_metadata_is_preserved_and_format_is_added_when_missing(setup, tmp_path):
    a1_dir, a0_dir, out, _, _ = setup(metadata={"format": "pt", "note": "x", "run": "7"})
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    for p in out.glob("*.safetensors"):
        with safe_open(str(p), framework="pt") as f:
            assert f.metadata() == {"format": "pt", "note": "x", "run": "7"}
    # an A1 file that carries no metadata at all gets the format key a loader asks for
    for name, seed in (("plain", 1), ("plain0", 2)):
        (tmp_path / name).mkdir()
        save_file(make_tensors(torch.float32, seed), str(tmp_path / name / "model.safetensors"))  # no metadata=
    blend_state_dicts(tmp_path / "plain", tmp_path / "plain0", 0.5, tmp_path / "plain_out")
    with safe_open(str(tmp_path / "plain_out" / "model.safetensors"), framework="pt") as f:
        assert f.metadata() == {"format": "pt"}


def test_non_weight_files_are_copied_from_a1(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    (a1_dir / "pytorch_model.bin").write_bytes(b"old weights")
    (a1_dir / ".cache").mkdir()
    (a1_dir / ".cache" / "state").write_text("hf download cache")
    (a1_dir / "onnx").mkdir()
    (a1_dir / "onnx" / "model.onnx").write_bytes(b"x")
    (a1_dir / "unused.safetensors.index.json").write_text("{}")
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    copied = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    for name in copied:
        assert (out / name).read_bytes() == (a1_dir / name).read_bytes(), name
        assert json.loads((out / name).read_text()).get("who", "a1") == "a1" if name.endswith(".json") else True
    names = {p.name for p in out.iterdir()}
    assert names == copied | {INDEX} | {p.name for p in a1_dir.glob("model-*.safetensors")}


# --- inputs that must be refused -------------------------------------------------------------------------------


def test_name_set_mismatch_raises_naming_both_sides(setup, tmp_path):
    a1_dir, a0_dir, out, a1, a0 = setup()
    a0 = {n: t for n, t in a0.items() if n != "lm_head.weight"}
    a0["extra.weight"] = torch.zeros(2, dtype=torch.bfloat16)
    for f in a0_dir.glob("*"):
        f.unlink()
    write_model(a0_dir, two_shards(a0), label="a0")
    with pytest.raises(ValueError) as err:
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    msg = str(err.value)
    assert "lm_head.weight" in msg and "extra.weight" in msg
    assert not out.exists()


def test_name_mismatch_message_lists_only_a_few_names(setup, tmp_path):
    a1 = {f"w{i:02d}": torch.zeros(1) for i in range(20)}
    write_model(tmp_path / "a1", one_file(a1))
    write_model(tmp_path / "a0", one_file({"other": torch.zeros(1)}), label="a0")
    with pytest.raises(ValueError) as err:
        blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 0.5, tmp_path / "out")
    msg = str(err.value)
    assert "w00" in msg and "w19" not in msg and "more" in msg


def test_floating_dtype_mismatch_raises_not_cast(setup):
    a1_dir, a0_dir, out, a1, a0 = setup()
    a0 = dict(a0, **{"embed.weight": a0["embed.weight"].to(torch.float32)})
    for f in a0_dir.glob("*"):
        f.unlink()
    write_model(a0_dir, three_shards(a0), label="a0")
    with pytest.raises(ValueError, match=re.escape("embed.weight")) as err:
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert "dtype" in str(err.value)
    assert not out.exists()


def _int_models(tmp_path, a0_ids):
    a1 = {"w": torch.ones(3, dtype=torch.float32), "ids": torch.tensor([1, 2, 3], dtype=torch.int64), "mask": torch.tensor([True, False])}
    a0 = {"w": torch.ones(3, dtype=torch.float32) * 3, "ids": torch.tensor(a0_ids, dtype=torch.int64), "mask": torch.tensor([True, False])}
    write_model(tmp_path / "a1", one_file(a1))
    write_model(tmp_path / "a0", one_file(a0), label="a0")
    return a1, a0


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_identical_non_floating_tensors_are_copied_from_a1(tmp_path, t):
    a1, _ = _int_models(tmp_path, [1, 2, 3])
    blend_state_dicts(tmp_path / "a1", tmp_path / "a0", t, tmp_path / "out")
    got = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert same_bits(got["ids"], a1["ids"]) and got["ids"].dtype == torch.int64
    assert same_bits(got["mask"], a1["mask"]) and got["mask"].dtype == torch.bool
    assert got["w"].dtype == torch.float32


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_differing_non_floating_tensors_raise(tmp_path, t):
    _int_models(tmp_path, [1, 2, 4])
    with pytest.raises(ValueError, match="ids"):
        blend_state_dicts(tmp_path / "a1", tmp_path / "a0", t, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("t", [-0.1, 1.1, float("nan"), float("inf"), float("-inf"), 2, -1])
def test_t_out_of_range_or_not_finite_raises(setup, t):
    a1_dir, a0_dir, out, _, _ = setup()
    with pytest.raises(ValueError, match="t"):
        blend_state_dicts(a1_dir, a0_dir, t, out)
    assert not out.exists()


@pytest.mark.parametrize("t", ["0.5", None, True])
def test_t_of_the_wrong_type_raises(setup, t):
    a1_dir, a0_dir, out, _, _ = setup()
    with pytest.raises(TypeError):
        blend_state_dicts(a1_dir, a0_dir, t, out)
    assert not out.exists()


def test_missing_weights_raise(tmp_path):
    (tmp_path / "a1").mkdir()
    (tmp_path / "a1" / "config.json").write_text("{}")
    write_model(tmp_path / "a0", one_file(make_tensors(torch.float32, 2)))
    with pytest.raises(FileNotFoundError, match="A1"):
        blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 0.5, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="A0"):
        blend_state_dicts(tmp_path / "a0", tmp_path / "nope", 0.5, tmp_path / "out")


def test_single_file_next_to_an_index_is_ambiguous(setup):
    a1_dir, a0_dir, out, a1, _ = setup()
    save_file(dict(a1), str(a1_dir / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="both"):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)


def test_index_that_disagrees_with_its_shard_raises(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    index = json.loads((a1_dir / INDEX).read_text())
    victim = sorted(index["weight_map"])[0]
    other = next(f for f in set(index["weight_map"].values()) if f != index["weight_map"][victim])
    index["weight_map"][victim] = other
    (a1_dir / INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index"):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)


def test_index_pointing_outside_the_directory_raises(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    index = json.loads((a1_dir / INDEX).read_text())
    index["weight_map"]["embed.weight"] = "../elsewhere.safetensors"
    (a1_dir / INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError, match="file name"):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)


def test_index_naming_a_missing_shard_raises(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    next(a1_dir.glob("model-00002-*.safetensors")).unlink()
    with pytest.raises(FileNotFoundError, match="model-00002"):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)


# --- non-finite values -----------------------------------------------------------------------------------------


def test_inf_and_nan_inputs_are_copied_exactly_at_the_endpoints(tmp_path):
    a1 = {"w": torch.tensor([float("inf"), 1.0, float("nan"), -float("inf")], dtype=torch.bfloat16)}
    a0 = {"w": torch.tensor([float("-inf"), float("inf"), 2.0, 5.0], dtype=torch.bfloat16)}
    write_model(tmp_path / "a1", one_file(a1))
    write_model(tmp_path / "a0", one_file(a0), label="a0")
    blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 0.0, tmp_path / "t0")
    blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 1.0, tmp_path / "t1")
    assert same_bits(load_file(str(tmp_path / "t0" / "model.safetensors"))["w"], a1["w"])  # 0 * inf is not NaN here
    assert same_bits(load_file(str(tmp_path / "t1" / "model.safetensors"))["w"], a0["w"])


def test_non_finite_output_from_finite_inputs_raises(setup, monkeypatch):
    a1_dir, a0_dir, out, _, _ = setup()
    monkeypatch.setattr(blend, "_mix", lambda a1, a0, t: torch.full_like(a1, float("nan")))
    with pytest.raises(ValueError, match="non-finite") as err:
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert any(n in str(err.value) for n in NAMES)
    assert not out.exists()


def test_non_finite_output_is_not_an_error_when_an_input_was_already_non_finite(tmp_path):
    a1 = {"w": torch.tensor([1.0, float("nan")], dtype=torch.float32)}
    a0 = {"w": torch.tensor([3.0, 4.0], dtype=torch.float32)}
    write_model(tmp_path / "a1", one_file(a1))
    write_model(tmp_path / "a0", one_file(a0), label="a0")
    blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 0.5, tmp_path / "out")
    got = load_file(str(tmp_path / "out" / "model.safetensors"))["w"]
    assert got[0].item() == 2.0 and got[1].isnan().item()


# --- output directory handling ---------------------------------------------------------------------------------


def test_non_empty_out_dir_is_refused_and_left_alone(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_out_dir_that_is_a_file_is_refused(setup):
    a1_dir, a0_dir, out, _, _ = setup()
    out.write_text("a file")
    with pytest.raises(FileExistsError):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)


def test_empty_existing_out_dir_is_accepted_and_missing_parents_are_created(setup, tmp_path):
    a1_dir, a0_dir, out, _, _ = setup()
    out.mkdir()
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert (out / INDEX).exists()
    deep = tmp_path / "x" / "y" / "z"
    blend_state_dicts(a1_dir, a0_dir, 0.5, deep)
    assert (deep / INDEX).exists()


def test_nothing_but_the_output_is_left_next_to_it(setup, tmp_path):
    a1_dir, a0_dir, out, _, _ = setup()
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a0", "a1", "out"]


class Boom(RuntimeError):
    pass


@pytest.mark.parametrize("exc", [Boom, KeyboardInterrupt])
@pytest.mark.parametrize("pre_existing_empty", [False, True])
def test_interrupted_write_leaves_no_final_out_dir(setup, tmp_path, monkeypatch, exc, pre_existing_empty):
    a1_dir, a0_dir, out, _, _ = setup()
    if pre_existing_empty:
        out.mkdir()
    real_mix = blend._mix
    calls = []

    def failing(a1, a0, t):
        calls.append(1)
        if len(calls) == 4:  # fourth tensor: the first shard is complete, the second is half written
            raise exc("disk full")
        return real_mix(a1, a0, t)

    monkeypatch.setattr(blend, "_mix", failing)
    with pytest.raises(exc):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert len(calls) == 4
    if pre_existing_empty:
        assert out.is_dir() and not any(out.iterdir())
    else:
        assert not out.exists()
    # no temporary folder survives either
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["a0", "a1"] + (["out"] if pre_existing_empty else []))


def test_a_failed_run_can_be_repeated_into_the_same_out_dir(setup, monkeypatch):
    a1_dir, a0_dir, out, a1, a0 = setup()
    real_mix = blend._mix
    monkeypatch.setattr(blend, "_mix", lambda *a: (_ for _ in ()).throw(Boom("once")))
    with pytest.raises(Boom):
        blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    monkeypatch.setattr(blend, "_mix", real_mix)
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    assert set(read_model(out)) == set(NAMES)


# --- determinism -----------------------------------------------------------------------------------------------


def _tree_bytes(root: Path):
    return {p.name: p.read_bytes() for p in sorted(root.iterdir())}


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0])
def test_rerun_is_byte_identical(setup, tmp_path, t):
    a1_dir, a0_dir, out, _, _ = setup()
    blend_state_dicts(a1_dir, a0_dir, t, out)
    blend_state_dicts(a1_dir, a0_dir, t, tmp_path / "out2")
    assert _tree_bytes(out) == _tree_bytes(tmp_path / "out2")


def test_output_does_not_depend_on_the_hash_order_of_shard_metadata(setup):
    """safetensors.torch.save_file writes several metadata keys in a per-process random order; ours must not."""
    a1_dir, a0_dir, out, _, _ = setup(metadata={"format": "pt", "z": "1", "a": "2", "q": "3", "m": "4"})
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    for p in out.glob("*.safetensors"):
        raw = p.read_bytes()
        (n,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8 : 8 + n])
        assert list(header["__metadata__"]) == ["a", "format", "m", "q", "z"]


# --- the file writer -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool],
)
def test_written_files_are_valid_for_every_dtype(tmp_path, dtype):
    a1 = {"w": (torch.arange(12).reshape(3, 4) % 2 == 0) if dtype == torch.bool else torch.arange(12).reshape(3, 4).to(dtype), "z": torch.ones(2, dtype=torch.float32)}
    write_model(tmp_path / "a1", one_file(a1))
    write_model(tmp_path / "a0", one_file(a1), label="a0")
    blend_state_dicts(tmp_path / "a1", tmp_path / "a0", 0.5, tmp_path / "out")
    path = tmp_path / "out" / "model.safetensors"
    got = load_file(str(path))
    assert got["w"].dtype == dtype and same_bits(got["w"], a1["w"])
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert n % 8 == 0  # the data section starts 8-byte aligned, as the library writes it
    header = json.loads(raw[8 : 8 + n])
    spans = sorted(v["data_offsets"] for k, v in header.items() if k != "__metadata__")
    assert spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # contiguous, no holes
    assert spans[-1][1] == len(raw) - 8 - n  # and covering the whole data section


def test_written_header_matches_what_the_library_reads(setup):
    a1_dir, a0_dir, out, a1, _ = setup()
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    for p in out.glob("*.safetensors"):
        with safe_open(str(p), framework="pt") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                assert tuple(sl.get_shape()) == tuple(a1[name].shape)
                assert sl.get_dtype() == "BF16"


# --- memory: shard by shard, tensor by tensor -------------------------------------------------------------------


class _Spy:
    """Wraps msd.blend.safe_open: records which files are open at once and how many loaded tensors are alive."""

    def __init__(self, real, a1_dir, a0_dir):
        self.real, self.a1_dir, self.a0_dir = real, str(a1_dir), str(a0_dir)
        self.open_now = set()
        self.max_open = {"a1": 0, "a0": 0}
        self.live = 0
        self.max_live = 0
        self.loaded = []  # (role, file, name)

    def role(self, path):
        return "a1" if str(path).startswith(self.a1_dir) else "a0"

    def __call__(self, path, *args, **kwargs):
        return _SpyHandle(self, str(path), self.real(path, *args, **kwargs))


class _SpyHandle:
    def __init__(self, spy, path, handle):
        self.spy, self.path, self.handle = spy, path, handle

    def __enter__(self):
        spy = self.spy
        spy.open_now.add(self.path)
        for role in ("a1", "a0"):
            spy.max_open[role] = max(spy.max_open[role], sum(1 for p in spy.open_now if spy.role(p) == role))
        self.handle.__enter__()
        return self

    def __exit__(self, *exc):
        self.spy.open_now.discard(self.path)
        return self.handle.__exit__(*exc)

    def keys(self):
        return self.handle.keys()

    def metadata(self):
        return self.handle.metadata()

    def get_slice(self, name):
        return self.handle.get_slice(name)

    def get_tensor(self, name):
        spy = self.spy
        tensor = self.handle.get_tensor(name)
        spy.loaded.append((spy.role(self.path), Path(self.path).name, name))
        spy.live += 1
        spy.max_live = max(spy.max_live, spy.live)

        def freed():
            spy.live -= 1

        weakref.finalize(tensor, freed)
        return tensor


def test_blend_holds_one_a1_shard_one_a0_shard_and_two_tensors_at_a_time(setup, monkeypatch):
    a1_dir, a0_dir, out, a1, a0 = setup()
    spy = _Spy(blend.safe_open, a1_dir, a0_dir)
    monkeypatch.setattr(blend, "safe_open", spy)
    blend_state_dicts(a1_dir, a0_dir, 0.5, out)
    gc.collect()
    assert spy.max_open == {"a1": 1, "a0": 1}
    assert spy.open_now == set()  # everything closed again
    # six tensors in the model, three shards for A1's two: at most the pair being mixed is alive at once
    assert spy.max_live <= 2 < len(NAMES)
    assert spy.live == 0
    # every tensor was read once from each model
    assert sorted(n for r, _, n in spy.loaded if r == "a1") == sorted(NAMES)
    assert sorted(n for r, _, n in spy.loaded if r == "a0") == sorted(NAMES)
    assert set(read_model(out)) == set(NAMES)


def test_endpoints_do_not_read_the_model_that_is_not_used(setup, monkeypatch):
    a1_dir, a0_dir, out, _, _ = setup()
    spy = _Spy(blend.safe_open, a1_dir, a0_dir)
    monkeypatch.setattr(blend, "safe_open", spy)
    blend_state_dicts(a1_dir, a0_dir, 0.0, out)
    assert {r for r, _, _ in spy.loaded} == {"a1"}
    assert spy.max_live <= 1
    spy.loaded.clear()
    blend_state_dicts(a1_dir, a0_dir, 1.0, out.with_name("out1"))
    assert {r for r, _, _ in spy.loaded} == {"a0"}
    assert spy.max_live <= 1
