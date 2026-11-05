"""scripts/run_resta.py and msd.restarun: the ReSta repair arm, with SafeTune replaced by a fake trainer.

SafeTune is not installed in the test environment and no model is ever loaded. The fake trainer has the
constructor and `apply()` that the source of safetune 0.1.12 has (docs/toolkit-notes.md, "SafeTune ReSta
(read from source)") and does the documented arithmetic with numpy. What these tests cannot show is that the
REAL SafeTune behaves like that: that is the first GPU-runtime sanity run (plan Task 16, Step 6).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from msd import restarun

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_resta.py"
TOKEN = "hf_SENTINELTOKENVALUE0123456789abcdef"  # stands in for a Hugging Face token: it must never be printed
BASE_ID = "FakeOrg/tiny-base"
ALIGNED_ID = "FakeOrg/tiny-global"


# --- fakes -----------------------------------------------------------------------------------------------------


class FakeModel:
    def __init__(self, **weights):
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    def state_dict(self):
        return self.weights  # live arrays, as nn.Module.state_dict() hands out live tensors

    def save_pretrained(self, folder, safe_serialization=True):
        assert safe_serialization is True
        Path(folder, "config.json").write_text(json.dumps({"fake": True}))
        np.savez(Path(folder) / "weights.npz", **self.weights)


class FakeTokenizer:
    def save_pretrained(self, folder):
        Path(folder, "tokenizer.json").write_text("{}")


class FakeTrainer:
    """Same constructor and apply() as safetune.runner.recover.ReStaTrainer 0.1.12; arithmetic without DARE."""

    instances: list = []

    def __init__(self, model=None, *, base_model=None, aligned_model=None, alpha=1.0, dare=True,
                 dare_drop_rate=None, dare_seed=0, device=None, **kwargs):
        self.model, self.base_model, self.aligned_model = model, base_model, aligned_model
        self.kwargs = dict(alpha=alpha, dare=dare, dare_drop_rate=dare_drop_rate, dare_seed=dare_seed,
                           device=device, **kwargs)
        self.applied = 0
        FakeTrainer.instances.append(self)

    def apply(self):
        self.applied += 1
        for name, value in self.model.weights.items():
            delta = self.aligned_model.weights[name] - self.base_model.weights[name]
            value += np.float32(self.kwargs["alpha"]) * delta  # in place, like the real one
        return self.model


@pytest.fixture(autouse=True)
def _reset_fake_trainer():
    FakeTrainer.instances = []


def make_models():
    drifted = FakeModel(embed=[[1, 2], [3, 4]], norm=[1, 1])
    base = FakeModel(embed=[[0, 0], [0, 0]], norm=[0, 0])
    aligned = FakeModel(embed=[[4, 8], [12, 16]], norm=[2, 4])
    return drifted, base, aligned


class World:
    """Three fake models behind a loader that records what it was asked to load."""

    def __init__(self, tmp_path, models=None):
        self.drifted_dir = tmp_path / "A1"
        self.drifted_dir.mkdir()
        (self.drifted_dir / "config.json").write_text("{}")
        self.out = tmp_path / "A2" / "alpha_0.25"
        self.models = dict(zip((str(self.drifted_dir), BASE_ID, ALIGNED_ID), models or make_models()))
        self.loads: list[str] = []
        self.tokenizer_loads: list[str] = []
        self.original_drifted = {k: v.copy() for k, v in self.models[str(self.drifted_dir)].weights.items()}

    def load_model(self, name):
        self.loads.append(str(name))
        return self.models[str(name)]

    def load_tokenizer(self, name):
        self.tokenizer_loads.append(str(name))
        return FakeTokenizer()

    def argv(self, *extra, alpha="0.25"):
        return ["--drifted", str(self.drifted_dir), "--base", BASE_ID, "--aligned", ALIGNED_ID, "--alpha", alpha,
                "--out", str(self.out), *extra]

    def run(self, *extra, alpha="0.25", trainer_cls=FakeTrainer):
        return restarun.main(self.argv(*extra, alpha=alpha), trainer_cls=trainer_cls, load_model=self.load_model,
                             load_tokenizer=self.load_tokenizer)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def siblings_of_out(world):
    return sorted(p.name for p in world.out.parent.iterdir()) if world.out.parent.exists() else []


# --- the call into SafeTune ---------------------------------------------------------------------------------------


def test_run_builds_the_trainer_as_safetune_documents_and_saves_the_patched_model(world, capsys):
    assert world.run() == 0
    (trainer,) = FakeTrainer.instances
    drifted, base, aligned = (world.models[k] for k in (str(world.drifted_dir), BASE_ID, ALIGNED_ID))
    assert trainer.model is drifted and trainer.base_model is base and trainer.aligned_model is aligned
    assert trainer.kwargs == {"alpha": 0.25, "dare": True, "dare_drop_rate": 0.3, "dare_seed": 0, "device": "cpu"}
    assert trainer.applied == 1
    assert world.loads == [str(world.drifted_dir), BASE_ID, ALIGNED_ID]
    assert world.tokenizer_loads == [str(world.drifted_dir)]  # the patched model keeps the drifted model's tokenizer
    saved = np.load(world.out / "weights.npz")
    expected = {k: world.original_drifted[k] + np.float32(0.25) * (aligned.weights[k] - base.weights[k]) for k in saved}
    assert set(saved) == {"embed", "norm"}
    assert all(np.array_equal(saved[k], expected[k]) for k in saved)
    assert (world.out / "config.json").is_file() and (world.out / "tokenizer.json").is_file()
    out = capsys.readouterr()
    assert str(world.out) in out.out
    assert siblings_of_out(world) == ["alpha_0.25"]


def test_alpha_zero_still_calls_safetune_and_writes_the_drifted_weights(world):
    assert world.run(alpha="0") == 0
    (trainer,) = FakeTrainer.instances
    assert trainer.kwargs["alpha"] == 0.0 and trainer.applied == 1
    saved = np.load(world.out / "weights.npz")
    assert all(np.array_equal(saved[k], world.original_drifted[k]) for k in saved)


@pytest.mark.parametrize("alpha, expected", [("0", 0.0), ("1", 1.0), ("0.5", 0.5), ("1e-3", 0.001)])
def test_alpha_is_passed_as_a_float(world, alpha, expected):
    assert world.run(alpha=alpha) == 0
    assert FakeTrainer.instances[0].kwargs["alpha"] == expected
    assert isinstance(FakeTrainer.instances[0].kwargs["alpha"], float)


def test_dare_options_are_passed_explicitly(world):
    assert world.run("--no-dare", "--dare-drop-rate", "0.5", "--dare-seed", "7") == 0
    kwargs = FakeTrainer.instances[0].kwargs
    assert kwargs["dare"] is False and kwargs["dare_drop_rate"] == 0.5 and kwargs["dare_seed"] == 7


def test_the_patched_object_apply_returns_is_the_one_saved(world):
    class ReturnsNew(FakeTrainer):
        def apply(self):
            super().apply()
            return FakeModel(embed=[[9, 9], [9, 9]], norm=[9, 9])

    assert world.run(trainer_cls=ReturnsNew) == 0
    saved = np.load(world.out / "weights.npz")
    assert np.array_equal(saved["norm"], [9, 9])


def test_an_apply_that_returns_nothing_saves_the_drifted_model(world):
    class ReturnsNone(FakeTrainer):
        def apply(self):
            super().apply()

    assert world.run(trainer_cls=ReturnsNone) == 0
    assert (world.out / "weights.npz").is_file()


def test_sidecar_records_what_was_done_and_nothing_secret(world, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert world.run("--dare-seed", "3") == 0
    raw = (world.out / "resta_patch.json").read_text()
    info = json.loads(raw)
    assert info["alpha"] == 0.25 and info["dare"] is True and info["dare_drop_rate"] == 0.3 and info["dare_seed"] == 3
    assert info["drifted"] == str(world.drifted_dir) and info["base"] == BASE_ID and info["aligned"] == ALIGNED_ID
    assert info["tensors"] == 2 and info["dtypes"] == ["float32"] and info["device"] == "cpu"
    assert "alpha * dare(" in info["formula"]
    assert "UNVERIFIED" in info["note"] or "unverified" in info["note"].lower()
    assert TOKEN not in raw and "hf_" not in raw


def test_sidecar_formula_without_dare_is_the_plain_arithmetic(world):
    assert world.run("--no-dare") == 0
    info = json.loads((world.out / "resta_patch.json").read_text())
    assert info["dare"] is False
    assert info["formula"] == "theta_safe = theta_drifted + alpha * (theta_aligned - theta_base)"


# --- argument validation: exit 2, before anything is loaded or written ---------------------------------------------


def assert_nothing_happened(world):
    assert world.loads == [] and FakeTrainer.instances == []
    assert not world.out.exists() and siblings_of_out(world) == []


@pytest.mark.parametrize("alpha", ["nan", "inf", "-inf", "-0.1", "1.01", "2", "abc", ""])
def test_a_bad_alpha_is_a_usage_error(world, alpha, capsys):
    assert world.run(alpha=alpha) == 2
    assert "alpha" in capsys.readouterr().err
    assert_nothing_happened(world)


@pytest.mark.parametrize("flag", ["--drifted", "--base", "--aligned", "--alpha", "--out"])
def test_every_main_flag_is_required(world, flag):
    argv = world.argv()
    i = argv.index(flag)
    del argv[i : i + 2]
    assert restarun.main(argv, trainer_cls=FakeTrainer, load_model=world.load_model) == 2
    assert_nothing_happened(world)


def test_a_non_empty_out_is_refused_and_left_alone(world):
    world.out.mkdir(parents=True)
    (world.out / "keep.txt").write_text("mine")
    assert world.run() == 2
    assert [p.name for p in world.out.iterdir()] == ["keep.txt"]
    assert world.loads == []


def test_drifted_must_be_an_existing_directory(world, tmp_path):
    argv = world.argv()
    argv[argv.index("--drifted") + 1] = str(tmp_path / "no-such-dir")
    assert restarun.main(argv, trainer_cls=FakeTrainer, load_model=world.load_model) == 2
    assert_nothing_happened(world)


def test_identical_base_and_aligned_are_refused_because_the_safety_vector_would_be_zero(world, capsys):
    argv = world.argv()
    argv[argv.index("--aligned") + 1] = BASE_ID
    assert restarun.main(argv, trainer_cls=FakeTrainer, load_model=world.load_model) == 2
    assert "same" in capsys.readouterr().err
    assert_nothing_happened(world)


@pytest.mark.parametrize("extra", [("--dare-drop-rate", "1.0"), ("--dare-drop-rate", "-0.1"), ("--dare-drop-rate", "nan"),
                                    ("--dare-drop-rate", "x"), ("--dare-seed", "1.5"), ("--dare-seed", "-1")])
def test_bad_dare_options_are_usage_errors(world, extra):
    assert world.run(*extra) == 2
    assert_nothing_happened(world)


def test_help_works_without_safetune_and_says_what_is_unverified(capsys):
    assert restarun.main(["--help"]) == 0
    text = capsys.readouterr().out
    assert "--drifted" in text and "--alpha" in text and "HF_TOKEN" in text
    assert "UNVERIFIED" in text
    assert "--token" not in text  # the token comes from the environment only


# --- failures: exit 1, no output folder, no leftovers ----------------------------------------------------------------


def test_missing_safetune_is_a_clear_error_before_any_model_is_loaded(world, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "safetune", None)  # makes `import safetune` raise ImportError
    code = restarun.main(world.argv(), load_model=world.load_model, load_tokenizer=world.load_tokenizer)
    assert code == 1
    err = capsys.readouterr().err
    assert "safetune" in err.lower() and "pip install" in err
    assert_nothing_happened(world)


def test_the_default_trainer_is_recover_ReStaTrainer_imported_lazily(world, monkeypatch):
    runner = types.ModuleType("safetune.runner")
    recover = types.ModuleType("safetune.runner.recover")
    recover.ReStaTrainer = FakeTrainer
    runner.recover = recover
    package = types.ModuleType("safetune")
    package.runner = runner
    for name, module in (("safetune", package), ("safetune.runner", runner), ("safetune.runner.recover", recover)):
        monkeypatch.setitem(sys.modules, name, module)
    code = restarun.main(world.argv(), load_model=world.load_model, load_tokenizer=world.load_tokenizer)
    assert code == 0 and len(FakeTrainer.instances) == 1


def test_a_safetune_that_fails_to_import_for_another_reason_is_reported_too(world, monkeypatch, capsys):
    class Broken:
        def __getattr__(self, name):
            raise ImportError(f"cannot import name {name!r} (a dependency of safetune is missing)")

    monkeypatch.setitem(sys.modules, "safetune", types.ModuleType("safetune"))
    monkeypatch.setitem(sys.modules, "safetune.runner", Broken())
    code = restarun.main(world.argv(), load_model=world.load_model, load_tokenizer=world.load_tokenizer)
    assert code == 1
    assert "dependency of safetune" in capsys.readouterr().err
    assert_nothing_happened(world)


def test_a_tensor_name_mismatch_is_refused_naming_it(tmp_path, capsys):
    drifted, base, aligned = make_models()
    base = FakeModel(embed=[[0, 0], [0, 0]], other=[0, 0])  # no "norm", has an extra "other"
    world = World(tmp_path, (drifted, base, aligned))
    assert world.run() == 1
    err = capsys.readouterr().err
    assert "norm" in err and "other" in err and "base" in err
    assert FakeTrainer.instances == [] and not world.out.exists() and siblings_of_out(world) == []


def test_a_shape_mismatch_is_refused_naming_it(tmp_path, capsys):
    drifted, base, aligned = make_models()
    aligned = FakeModel(embed=[[1, 2, 3]], norm=[2, 4])
    world = World(tmp_path, (drifted, base, aligned))
    assert world.run() == 1
    err = capsys.readouterr().err
    assert "embed" in err and "shape" in err and "aligned" in err
    assert FakeTrainer.instances == [] and not world.out.exists()


def test_a_failing_apply_leaves_no_output(world, capsys):
    class Fails(FakeTrainer):
        def apply(self):
            raise RuntimeError("out of memory")

    assert world.run(trainer_cls=Fails) == 1
    assert "out of memory" in capsys.readouterr().err
    assert not world.out.exists() and siblings_of_out(world) == []


def test_a_failing_save_leaves_no_output(world):
    drifted = world.models[str(world.drifted_dir)]

    def broken_save(folder, safe_serialization=True):
        Path(folder, "half.bin").write_bytes(b"x")
        raise OSError("disk full")

    drifted.save_pretrained = broken_save
    assert world.run() == 1
    assert not world.out.exists() and siblings_of_out(world) == []


def test_an_interrupt_leaves_no_output_and_propagates(world):
    class Interrupted(FakeTrainer):
        def apply(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        world.run(trainer_cls=Interrupted)
    assert not world.out.exists() and siblings_of_out(world) == []


def test_a_model_that_cannot_be_loaded_is_a_plain_error(world, capsys):
    def load_model(name):
        raise OSError(f"{name} is a gated repo; accept the licence")

    code = restarun.main(world.argv(), trainer_cls=FakeTrainer, load_model=load_model)
    assert code == 1
    assert "gated" in capsys.readouterr().err
    assert not world.out.exists() and siblings_of_out(world) == []


# --- the token ------------------------------------------------------------------------------------------------------


def test_a_token_in_an_error_message_is_never_printed(world, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "a-token-without-the-usual-prefix")

    def load_model(name):
        raise OSError(f"401 for {name} with Bearer {TOKEN} and a-token-without-the-usual-prefix, also hf_AnotherTokenShape12345")

    assert restarun.main(world.argv(), trainer_cls=FakeTrainer, load_model=load_model) == 1
    err = capsys.readouterr()
    shown = err.out + err.err
    assert TOKEN not in shown and "a-token-without-the-usual-prefix" not in shown and "hf_AnotherTokenShape" not in shown
    assert "<token>" in shown


def test_an_unexpected_exception_traceback_is_scrubbed_too(world, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    class Surprise(FakeTrainer):
        def apply(self):
            raise ZeroDivisionError(f"surprise {TOKEN}")

    assert world.run(trainer_cls=Surprise) == 1
    shown = "".join(capsys.readouterr())
    assert TOKEN not in shown and "ZeroDivisionError" in shown


def test_a_successful_run_never_prints_the_token(world, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert world.run() == 0
    shown = "".join(capsys.readouterr())
    assert TOKEN not in shown


def test_there_is_no_token_flag_and_a_mistaken_one_is_not_echoed_back(world, capsys):
    # argparse repeats unrecognised arguments in its error message; the value after --token is a secret
    assert world.run("--token", TOKEN) == 2
    shown = capsys.readouterr()
    assert "unrecognized arguments" in shown.err
    assert TOKEN not in shown.out + shown.err and "<token>" in shown.err
    assert_nothing_happened(world)


# --- the default loaders --------------------------------------------------------------------------------------------


def test_default_loaders_use_transformers_with_the_stored_dtype_and_nothing_else(monkeypatch):
    calls = []

    class Loaded:
        def eval(self):  # stands in for torch's nn.Module.eval(); not Python's builtin eval
            calls.append("eval")
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda name, **kw: calls.append(("model", name, kw)) or Loaded()
    )
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name, **kw: calls.append(("tokenizer", name, kw)) or "tok"
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert isinstance(restarun.load_causal_lm("some/model"), Loaded)
    assert restarun.load_tokenizer("some/dir") == "tok"
    # dtype="auto" keeps each model in the dtype it was saved in; no token, no remote code
    assert calls == [("model", "some/model", {"dtype": "auto"}), "eval", ("tokenizer", "some/dir", {})]


# --- the script -----------------------------------------------------------------------------------------------------


def run_script(*argv, **kw):
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, **kw)


def test_script_help_and_usage_errors(tmp_path):
    assert run_script("--help").returncode == 0
    assert run_script().returncode == 2
    bad = run_script("--drifted", str(tmp_path), "--base", "a", "--aligned", "b", "--alpha", "2", "--out", str(tmp_path / "out"))
    assert bad.returncode == 2 and not (tmp_path / "out").exists()


@pytest.mark.skipif(importlib.util.find_spec("safetune") is not None, reason="safetune is installed: the run would try to load real models")
def test_script_without_safetune_stops_early_with_one_clear_error(tmp_path):
    drifted = tmp_path / "A1"
    drifted.mkdir()
    done = run_script("--drifted", str(drifted), "--base", "a/b", "--aligned", "c/d", "--alpha", "0.25", "--out", str(tmp_path / "out"))
    assert done.returncode == 1
    assert "safetune" in done.stderr.lower() and "pip install" in done.stderr
    assert "Traceback" not in done.stderr
    assert not (tmp_path / "out").exists()


def test_importing_the_module_pulls_in_none_of_the_heavy_packages():
    code = (
        "import sys; import msd.restarun; "
        "bad = [m for m in ('safetune', 'torch', 'transformers') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_the_script_is_a_thin_wrapper():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "restarun" in text and len(text.splitlines()) < 40
    assert "import json" not in text  # no logic of its own
    assert "import safetune" not in text and "from safetune" not in text  # SafeTune is imported in msd.restarun only
