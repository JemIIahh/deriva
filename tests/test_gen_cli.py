"""scripts/generate.py and msd.genrun: arguments, prompt loading, metadata, exit codes.

Everything runs against a throwaway repo layout under tmp_path (data/testset/<lang>/<split>.jsonl),
a fake or injected backend, and never a model.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from msd import gen, genrun
from msd.gen import GenOut
from msd.schema import Generation, Prompt, read_jsonl, write_jsonl

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate.py"
MARK = "SENTINEL-PROMPT-TEXT"  # stands in for prompt text; it must never be printed


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "testset").mkdir(parents=True)
    return root


def build_testset(root, langs=("en",), split="dev", n=6, testset_dir=None):
    base = Path(testset_dir) if testset_dir else root / "data" / "testset"
    for lang in langs:
        write_jsonl(
            base / lang / f"{split}.jsonl",
            [Prompt(f"{lang}-{i}", lang, "harmful" if i % 2 == 0 else "benign_edgy", f"{MARK} {lang} {i}", split,
                    "fixture", "fixture") for i in range(n)],
        )


@pytest.fixture
def scratch(tmp_path):
    return tmp_path / "scratch_out"


class Hf:
    """Stands in for gen.hf_backend: counts how often a model would have been loaded."""

    def __init__(self, fail_on_call=None):
        self.built = []
        self.calls = []
        self.fail_on_call = fail_on_call

    def __call__(self, model_path, *, max_new_tokens=256, dtype="float16"):
        self.built.append({"model_path": model_path, "max_new_tokens": max_new_tokens, "dtype": dtype})

        def backend(texts):
            self.calls.append(list(texts))
            if self.fail_on_call and len(self.calls) == self.fail_on_call:
                raise RuntimeError("CUDA out of memory (pretend)")
            return [GenOut(f"hf-out for {t[-3:]}", "stop") for t in texts]

        return backend


@pytest.fixture
def hf(monkeypatch):
    fake = Hf()
    monkeypatch.setattr(gen, "hf_backend", fake)
    return fake


def run_cli(repo, *argv):
    return genrun.main(list(argv), root=repo)


def fake_args(scratch, *extra, arm="A0", split="dev"):
    return ["--arm", arm, "--split", split, "--backend", "fake", "--out-root", str(scratch), *extra]


def hf_args(scratch, *extra, arm="A0", split="dev"):
    return ["--arm", arm, "--split", split, "--backend", "hf", "--model-path", "org/model",
            "--out-root", str(scratch), *extra]


# --- usage errors: exit code 2 -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],  # nothing at all
        ["--arm", "A0"],  # no split
        ["--split", "dev"],  # no arm
        ["--arm", "A9", "--split", "dev", "--backend", "fake"],  # unknown arm
        ["--arm", "a0", "--split", "dev", "--backend", "fake"],
        ["--arm", "A0", "--split", "train", "--backend", "fake"],  # unknown split
        ["--arm", "A0", "--split", "dev", "--backend", "gpu"],  # unknown backend
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--alpha", "0.5"],  # alpha on an arm without one
        ["--arm", "BASE", "--split", "dev", "--backend", "fake", "--alpha", "0"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--alpha", "1.0"],
        ["--arm", "A3", "--split", "dev", "--backend", "fake", "--alpha", "0.5"],  # Harden has no alpha either
        ["--arm", "A2", "--split", "dev", "--backend", "fake"],  # alpha is required
        ["--arm", "A2n", "--split", "dev", "--backend", "fake"],
        ["--arm", "A2", "--split", "dev", "--backend", "fake", "--alpha", "none"],
        ["--arm", "A2", "--split", "dev", "--backend", "fake", "--alpha", "abc"],
        ["--arm", "A2", "--split", "dev", "--backend", "fake", "--alpha", "nan"],
        ["--arm", "A2", "--split", "dev", "--backend", "fake", "--alpha", "-0.5"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--langs", "en,fr"],  # unknown language
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--langs", "en,en"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--langs", ""],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--limit", "0"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--limit", "-3"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--limit", "x"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--batch-size", "0"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--max-new-tokens", "0"],
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--dtype", "fp16"],
        ["--arm", "A0", "--split", "dev", "--backend", "hf"],  # the real backend needs --model-path
        ["--arm", "A0", "--split", "dev", "--backend", "fake", "--surprise"],
    ],
)
def test_bad_arguments_exit_2_and_write_nothing(repo, scratch, capsys, argv):
    build_testset(repo)
    if "--out-root" not in argv:
        argv = [*argv, "--out-root", str(scratch)]
    assert run_cli(repo, *argv) == 2
    assert not scratch.exists()
    assert "usage" in capsys.readouterr().err.lower()


def test_help_exits_0(repo, capsys):
    assert run_cli(repo, "--help") == 0
    assert "--alpha" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arm, extra",
    [("BASE", []), ("A0", []), ("A1", []), ("A2", ["--alpha", "0.25"]), ("A2n", ["--alpha", "0.5"]),
     ("A3", []), ("A3", ["--alpha", "none"]), ("A0", ["--alpha", "none"]), ("A1", ["--alpha", "None"])],
)
def test_valid_arm_alpha_combinations_run(repo, scratch, arm, extra):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch, *extra, arm=arm)) == 0


def test_defaults(repo):
    args = genrun.parse_args(["--arm", "A0", "--split", "dev", "--backend", "fake", "--out-root", str(repo / "x")],
                             root=repo)
    assert args.langs == ["en", "hi", "hinglish", "ar", "sw"]
    assert args.testset_dir == repo / "data" / "testset"
    assert args.limit is None and args.batch_size == 32 and args.max_new_tokens == 256 and args.dtype == "float16"
    args = genrun.parse_args(["--arm", "A0", "--split", "dev", "--model-path", "m"], root=repo)
    assert args.out_root == repo / "results" and args.backend == "hf"
    assert args.alpha is None and args.force_new_meta is False


# --- the fake-backend guard ----------------------------------------------------------------------------------


@pytest.mark.parametrize("sub", ["", "generate", "dev/A0", "../results", "../data/../results"])
def test_fake_backend_is_refused_in_the_real_results_root(repo, capsys, sub):
    build_testset(repo)
    real = repo / "results"
    target = real / sub if sub else real
    assert genrun.main(fake_args(target), root=repo) == 2
    assert not real.exists()  # nothing was created
    assert "fake" in capsys.readouterr().err


def test_fake_backend_is_refused_by_default_out_root(repo):
    build_testset(repo)
    assert genrun.main(["--arm", "A0", "--split", "dev", "--backend", "fake"], root=repo) == 2
    assert not (repo / "results").exists()


def test_fake_backend_is_refused_through_a_symlink_into_results(repo, tmp_path):
    build_testset(repo)
    (repo / "results").mkdir()
    link = tmp_path / "innocent"
    link.symlink_to(repo / "results", target_is_directory=True)
    assert genrun.main(fake_args(link), root=repo) == 2
    assert list((repo / "results").iterdir()) == []


def test_fake_backend_is_refused_in_a_results_dir_relative_to_the_cwd(repo, tmp_path, monkeypatch):
    build_testset(repo)
    monkeypatch.chdir(tmp_path)
    assert genrun.main(fake_args(Path("results")), root=repo) == 2
    assert not (tmp_path / "results").exists()


def test_fake_backend_is_fine_elsewhere_and_the_real_backend_may_use_results(repo, hf, scratch):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch)) == 0
    assert run_cli(repo, "--arm", "A0", "--split", "dev", "--backend", "hf", "--model-path", "m") == 0  # default out root
    assert (repo / "results" / "dev" / "A0" / "none" / "en.gens.jsonl").is_file()


BLOCKED_SPELLINGS = [
    "results", "./results", "results/", "results/../results", "{abs}", "{abs}/.", "{abs}/../results",
    "{link}", "{link}/sub", "Results", "RESULTS/x", "results/generate", "{abs_upper}", "../{reponame}/results",
]
ALLOWED_SPELLINGS = ["~/results", "results_x", "results.bak", "{root}", "{elsewhere}/results"]


@pytest.fixture
def layout(repo, tmp_path, monkeypatch):
    (repo / "results").mkdir()
    link = tmp_path / "lnk"
    link.symlink_to(repo / "results", target_is_directory=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(repo)
    return {"root": str(repo), "abs": str(repo / "results"), "link": str(link), "elsewhere": str(elsewhere),
            "abs_upper": str(repo / "RESULTS"), "reponame": repo.name}


@pytest.mark.parametrize("spelling", BLOCKED_SPELLINGS)
def test_every_spelling_of_the_real_results_dir_is_recognised(repo, layout, spelling):
    assert genrun.is_in_results(Path(spelling.format(**layout)), repo)


@pytest.mark.parametrize("spelling", ALLOWED_SPELLINGS)
def test_other_directories_are_not_taken_for_results(repo, layout, spelling):
    assert not genrun.is_in_results(Path(spelling.format(**layout)), repo)


def test_a_case_variant_of_results_is_recognised_from_another_cwd(repo, layout, monkeypatch):
    monkeypatch.chdir(layout["elsewhere"])
    assert genrun.is_in_results(repo / "Results", repo)  # works on a case-sensitive file system too
    assert genrun.is_in_results(repo / "RESULTS" / "generate" / "deep", repo)
    assert not genrun.is_in_results(repo / "Resultsx", repo)


def test_fake_backend_with_a_case_variant_out_root_writes_nothing(repo, layout, capsys):
    build_testset(repo, n=2)
    for variant in ("Results", "RESULTS/x", str(repo / "Results")):
        assert genrun.main(fake_args(variant), root=repo) == 2
    assert list((repo / "results").iterdir()) == []  # on a case-insensitive file system Results IS results
    assert not (repo / "Results").exists() or (repo / "Results").samefile(repo / "results")
    assert "fake" in capsys.readouterr().err


def test_a_path_that_is_the_same_directory_by_the_file_system_is_refused(repo, layout, tmp_path, monkeypatch):
    mount = tmp_path / "mount"
    mount.mkdir()
    assert not genrun.is_in_results(mount / "sub", repo)
    # the file system says `mount` is results/ (a bind mount, a case-insensitive spelling the name test misses ...)
    monkeypatch.setattr(os.path, "samefile", lambda a, b: Path(a).name == "mount")
    assert genrun.is_in_results(mount / "sub", repo)
    build_testset(repo, n=2)
    assert genrun.main(fake_args(mount / "sub"), root=repo) == 2


# --- a normal run -------------------------------------------------------------------------------------------------


def test_fake_run_writes_gens_and_meta_at_the_documented_paths(repo, scratch, capsys):
    build_testset(repo, n=6)
    assert run_cli(repo, *fake_args(scratch, "--batch-size", "4")) == 0
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    assert gens_file == scratch / "dev" / "A0" / "none" / "en.gens.jsonl"
    rows = read_jsonl(gens_file, Generation)
    assert [r.prompt_id for r in rows] == [f"en-{i}" for i in range(6)]
    assert all(r.arm == "A0" and r.alpha is None and r.finish_reason == "stop" and gen.FAKE_MARKER in r.text
               for r in rows)
    meta = json.loads(gen.meta_path(gens_file).read_text(encoding="utf-8"))
    assert meta["model_path"] == "fake" and meta["arm"] == "A0" and meta["alpha"] is None
    assert meta["split"] == "dev" and meta["lang"] == "en" and meta["backend"] == "fake"
    assert meta["batch_size"] == 4 and meta["max_new_tokens"] == 256 and meta["dtype"] == "float16"
    assert meta["n_rows"] == 6 and meta["msd_version"] and meta["timestamp_utc"].endswith("Z")
    assert "git_commit" in meta
    shown = capsys.readouterr()
    assert MARK not in shown.out + shown.err  # counts and paths only, never prompt text
    assert "en" in shown.out and "6" in shown.out


def test_alpha_goes_into_the_path_the_rows_and_the_meta(repo, scratch):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch, "--alpha", "0.25", arm="A2")) == 0
    gens_file = scratch / "dev" / "A2" / "0.25" / "en.gens.jsonl"
    assert [r.alpha for r in read_jsonl(gens_file, Generation)] == [0.25, 0.25]
    assert json.loads(gen.meta_path(gens_file).read_text(encoding="utf-8"))["alpha"] == 0.25


def test_extra_prompts_are_generated_into_the_same_gens_file_after_the_test_set(repo, scratch):
    build_testset(repo, n=2)
    args = genrun.parse_args(fake_args(scratch), root=repo)
    extra = [Prompt("task-0", "en", "task", f"{MARK} question", "dev", "fixture", "fixture", "B")]
    available = [("en", repo / "data" / "testset" / "en" / "dev.jsonl")]
    assert genrun.run(args, available, repo, extra_prompts={"en": extra, "hi": extra}) == 0
    rows = read_jsonl(gen.gens_path(scratch, "dev", "A0", None, "en"), Generation)
    assert [r.prompt_id for r in rows] == ["en-0", "en-1", "task-0"]
    assert json.loads(gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en")).read_text())["n_rows"] == 3
    assert genrun.run(args, available, repo, extra_prompts={"en": extra}) == 0  # a rerun has nothing to do


def test_the_argument_helpers_and_error_tuple_are_public_and_leave_no_private_alias():
    for public in ("alpha_arg", "langs_arg", "positive_int", "PLAIN_ERRORS"):
        assert hasattr(genrun, public), public
    for private in ("_alpha_arg", "_langs_arg", "_positive_int", "_PLAIN_ERRORS"):
        assert not hasattr(genrun, private), private
    assert genrun.positive_int("3") == 3 and genrun.alpha_arg("none") is None and genrun.alpha_arg("0.5") == 0.5
    assert genrun.langs_arg("en,hi") == ["en", "hi"]
    assert gen.ResumeError in genrun.PLAIN_ERRORS and genrun.MetaMismatchError in genrun.PLAIN_ERRORS


def test_report_split_reads_report_files(repo, scratch):
    build_testset(repo, n=3, split="report")
    assert run_cli(repo, *fake_args(scratch, split="report")) == 0
    assert (scratch / "report" / "A0" / "none" / "en.gens.jsonl").is_file()


def test_limit_takes_the_first_n_prompts_in_file_order(repo, scratch):
    build_testset(repo, n=9)
    assert run_cli(repo, *fake_args(scratch, "--limit", "4")) == 0
    rows = read_jsonl(gen.gens_path(scratch, "dev", "A0", None, "en"), Generation)
    assert [r.prompt_id for r in rows] == ["en-0", "en-1", "en-2", "en-3"]


def test_limit_larger_than_the_file_just_takes_the_file(repo, scratch):
    build_testset(repo, n=3)
    assert run_cli(repo, *fake_args(scratch, "--limit", "50")) == 0
    assert len(read_jsonl(gen.gens_path(scratch, "dev", "A0", None, "en"), Generation)) == 3


def test_a_bigger_limit_on_a_later_run_continues_the_same_file(repo, scratch):
    build_testset(repo, n=8)
    assert run_cli(repo, *fake_args(scratch, "--limit", "3")) == 0
    assert run_cli(repo, *fake_args(scratch, "--limit", "8")) == 0
    rows = read_jsonl(gen.gens_path(scratch, "dev", "A0", None, "en"), Generation)
    assert [r.prompt_id for r in rows] == [f"en-{i}" for i in range(8)]


def test_several_languages_each_get_their_own_file(repo, scratch):
    build_testset(repo, langs=("en", "hi", "ar"), n=2)
    assert run_cli(repo, *fake_args(scratch, "--langs", "ar,en")) == 0
    produced = sorted(p.name for p in (scratch / "dev" / "A0" / "none").glob("*.gens.jsonl"))
    assert produced == ["ar.gens.jsonl", "en.gens.jsonl"]  # hi was not asked for


def test_testset_dir_option(repo, scratch, tmp_path):
    other = tmp_path / "elsewhere"
    build_testset(repo, n=2, testset_dir=other)
    assert run_cli(repo, *fake_args(scratch, "--testset-dir", str(other))) == 0


# --- the real backend path -------------------------------------------------------------------------------------------


def test_hf_backend_gets_its_options_and_the_prompt_texts(repo, hf, scratch):
    build_testset(repo, n=3)
    argv = hf_args(scratch, "--max-new-tokens", "64", "--dtype", "bfloat16", "--batch-size", "2")
    assert run_cli(repo, *argv) == 0
    assert hf.built == [{"model_path": "org/model", "max_new_tokens": 64, "dtype": "bfloat16"}]
    assert [len(c) for c in hf.calls] == [2, 1]
    assert hf.calls[0] == [f"{MARK} en 0", f"{MARK} en 1"]  # raw prompt text; the backend applies the chat template
    meta = json.loads(gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en")).read_text(encoding="utf-8"))
    assert meta["model_path"] == "org/model" and meta["backend"] == "hf" and meta["dtype"] == "bfloat16"
    assert meta["max_new_tokens"] == 64 and meta["batch_size"] == 2


def test_the_model_is_built_once_for_all_languages_and_not_at_all_when_nothing_is_left(repo, hf, scratch):
    build_testset(repo, langs=("en", "hi"), n=2)
    assert run_cli(repo, *hf_args(scratch, "--langs", "en,hi")) == 0
    assert len(hf.built) == 1
    assert run_cli(repo, *hf_args(scratch, "--langs", "en,hi")) == 0  # everything finished: no model load
    assert len(hf.built) == 1 and len(hf.calls) == 2


def test_a_rerun_after_completion_leaves_every_file_byte_identical(repo, scratch, capsys):
    build_testset(repo, langs=("en", "hi"), n=4)
    assert run_cli(repo, *fake_args(scratch, "--langs", "en,hi")) == 0
    files = sorted(scratch.rglob("*.gens.jsonl"))
    before = [p.read_bytes() for p in files]
    capsys.readouterr()
    assert run_cli(repo, *fake_args(scratch, "--langs", "en,hi")) == 0
    assert [p.read_bytes() for p in files] == before
    assert "0 new" in capsys.readouterr().out


# --- missing languages ---------------------------------------------------------------------------------------------------


def test_languages_that_are_not_built_yet_are_skipped_with_a_message(repo, scratch, capsys):
    build_testset(repo, langs=("en",), n=2)
    assert run_cli(repo, *fake_args(scratch, "--langs", "en,hi,ar")) == 0
    shown = capsys.readouterr()
    assert "hi" in shown.err and "ar" in shown.err and "not built" in shown.err
    assert (scratch / "dev" / "A0" / "none" / "en.gens.jsonl").is_file()
    assert not (scratch / "dev" / "A0" / "none" / "hi.gens.jsonl").exists()


def test_default_langs_on_a_laptop_with_only_english(repo, scratch, capsys):
    build_testset(repo, langs=("en",), n=2)
    assert run_cli(repo, *fake_args(scratch)) == 0
    assert capsys.readouterr().err.count("not built") == 4


def test_no_requested_language_available_is_a_failure_and_no_model_is_loaded(repo, hf, scratch, capsys):
    build_testset(repo, langs=("en",), n=2)
    rc = run_cli(repo, *hf_args(scratch, "--langs", "hi,sw"))
    assert rc not in (0, 2)
    assert hf.built == []  # failing fast: no GPU time spent
    assert not scratch.exists()
    assert "no requested language" in capsys.readouterr().err


def test_a_testset_that_does_not_exist_at_all_fails(repo, scratch):
    assert run_cli(repo, *fake_args(scratch, "--testset-dir", str(repo / "nope"))) not in (0, 2)


def test_an_empty_prompt_file_is_an_error_not_a_skip(repo, scratch, capsys):
    build_testset(repo, n=0)
    assert run_cli(repo, *fake_args(scratch)) not in (0, 2)
    assert "no prompts" in capsys.readouterr().err


def test_a_prompt_file_with_rows_of_another_split_or_language_is_refused(repo, scratch):
    write_jsonl(repo / "data" / "testset" / "en" / "dev.jsonl",
                [Prompt("x", "en", "harmful", "t", "report", "s", "l")])
    assert run_cli(repo, *fake_args(scratch)) not in (0, 2)
    write_jsonl(repo / "data" / "testset" / "en" / "dev.jsonl",
                [Prompt("x", "hi", "harmful", "t", "dev", "s", "l")])
    assert run_cli(repo, *fake_args(scratch)) not in (0, 2)


# --- runtime errors never exit 0 ------------------------------------------------------------------------------------------


def test_a_backend_crash_exits_nonzero_and_keeps_finished_batches(repo, monkeypatch, scratch, capsys):
    build_testset(repo, n=6)
    monkeypatch.setattr(gen, "hf_backend", Hf(fail_on_call=2))
    rc = run_cli(repo, *hf_args(scratch, "--batch-size", "2"))
    assert rc not in (0, 2)
    assert "out of memory" in capsys.readouterr().err
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    assert [r.prompt_id for r in read_jsonl(gens_file, Generation)] == ["en-0", "en-1"]
    assert gen.meta_path(gens_file).is_file()  # the partial file is documented too

    monkeypatch.setattr(gen, "hf_backend", Hf())
    assert run_cli(repo, *hf_args(scratch, "--batch-size", "2")) == 0  # a rerun resumes
    assert len(read_jsonl(gens_file, Generation)) == 6


def test_a_model_load_failure_exits_nonzero(repo, monkeypatch, scratch, capsys):
    build_testset(repo, n=2)

    def boom(*args, **kwargs):
        raise ImportError("hf_backend needs the 'torch' and 'transformers' packages")

    monkeypatch.setattr(gen, "hf_backend", boom)
    assert run_cli(repo, *hf_args(scratch)) not in (0, 2)
    assert "torch" in capsys.readouterr().err


def test_a_corrupt_existing_gens_file_exits_nonzero(repo, scratch, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, *fake_args(scratch)) == 0
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    lines = gens_file.read_bytes().split(b"\n")
    lines[1] = b"{not json"
    gens_file.write_bytes(b"\n".join(lines))
    assert run_cli(repo, *fake_args(scratch)) not in (0, 2)
    assert "line 2" in capsys.readouterr().err


def test_a_wrong_arm_file_exits_nonzero(repo, scratch):
    build_testset(repo, n=2)
    gens_file = gen.gens_path(scratch, "dev", "A1", None, "en")
    write_jsonl(gens_file, [Generation("en-0", "A0", None, "x", "stop")])  # an A0 row filed under A1
    assert run_cli(repo, *fake_args(scratch, arm="A1")) not in (0, 2)


# --- run metadata ----------------------------------------------------------------------------------------------------------------


def test_changing_the_setup_of_an_existing_file_is_refused_unless_forced(repo, scratch, capsys):
    build_testset(repo, n=6)
    assert run_cli(repo, *fake_args(scratch, "--limit", "3", "--batch-size", "4")) == 0
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    before = gens_file.read_bytes()
    capsys.readouterr()

    rc = run_cli(repo, *fake_args(scratch, "--limit", "6", "--batch-size", "8"))
    assert rc not in (0, 2)
    err = capsys.readouterr().err
    assert "--force-new-meta" in err and "batch_size" in err
    assert gens_file.read_bytes() == before  # refused before any generation

    assert run_cli(repo, *fake_args(scratch, "--limit", "6", "--batch-size", "8", "--force-new-meta")) == 0
    meta = json.loads(gen.meta_path(gens_file).read_text(encoding="utf-8"))
    assert meta["batch_size"] == 8 and meta["n_rows"] == 6
    assert [e["batch_size"] for e in meta["earlier"]] == [4]  # the replaced setup stays on record
    assert len(read_jsonl(gens_file, Generation)) == 6


@pytest.mark.parametrize(
    "flag, value",
    [("--max-new-tokens", "64"), ("--dtype", "bfloat16"), ("--batch-size", "7")],
)
def test_each_generation_setting_is_part_of_the_refusal_rule(repo, scratch, flag, value):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch)) == 0
    assert run_cli(repo, *fake_args(scratch, flag, value)) not in (0, 2)


def test_a_different_model_path_is_refused(repo, hf, scratch):
    build_testset(repo, n=2)
    assert run_cli(repo, *hf_args(scratch)) == 0
    argv = hf_args(scratch)
    argv[argv.index("org/model")] = "org/other-model"
    assert run_cli(repo, *argv) not in (0, 2)
    assert run_cli(repo, *argv, "--force-new-meta") == 0


def test_switching_between_fake_and_real_backend_in_one_directory_is_refused(repo, hf, scratch):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch)) == 0
    assert run_cli(repo, *hf_args(scratch)) not in (0, 2)  # fake rows must not become a real run


def test_the_same_setup_again_is_accepted_and_a_no_op_rerun_leaves_the_meta_untouched(repo, scratch):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch)) == 0
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en"))
    first = meta_file.read_bytes()
    assert run_cli(repo, *fake_args(scratch)) == 0
    assert meta_file.read_bytes() == first  # nothing new was generated and the setup is the same: no rewrite
    assert "earlier" not in json.loads(first)
    assert run_cli(repo, *fake_args(scratch, "--force-new-meta")) == 0  # forcing the same setup changes nothing either
    assert meta_file.read_bytes() == first


def test_a_forced_setup_change_with_nothing_to_generate_still_updates_the_meta(repo, scratch):
    build_testset(repo, n=3)
    assert run_cli(repo, *fake_args(scratch, "--batch-size", "4")) == 0
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    before = gens_file.read_bytes()
    assert run_cli(repo, *fake_args(scratch, "--batch-size", "8", "--force-new-meta")) == 0
    meta = json.loads(gen.meta_path(gens_file).read_text(encoding="utf-8"))
    assert meta["batch_size"] == 8 and [e["batch_size"] for e in meta["earlier"]] == [4]
    assert gens_file.read_bytes() == before


def test_a_run_that_adds_rows_records_the_previous_meta_once(repo, scratch):
    build_testset(repo, n=6)
    assert run_cli(repo, *fake_args(scratch, "--limit", "2")) == 0
    assert run_cli(repo, *fake_args(scratch, "--limit", "6")) == 0
    meta = json.loads(gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en")).read_text(encoding="utf-8"))
    assert meta["n_rows"] == 6 and [e["n_rows"] for e in meta["earlier"]] == [2]


def test_an_unreadable_meta_file_is_refused_unless_forced(repo, scratch):
    build_testset(repo, n=2)
    assert run_cli(repo, *fake_args(scratch)) == 0
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en"))
    meta_file.write_text("{ torn", encoding="utf-8")
    assert run_cli(repo, *fake_args(scratch)) not in (0, 2)
    assert run_cli(repo, *fake_args(scratch, "--force-new-meta")) == 0
    assert json.loads(meta_file.read_text(encoding="utf-8"))["n_rows"] == 2


def test_check_meta_compatible_compares_exactly_the_setup_fields():
    base = {"model_path": "m", "arm": "A2", "alpha": 0.25, "split": "dev", "lang": "en", "batch_size": 4,
            "max_new_tokens": 256, "dtype": "float16", "backend": "hf", "n_rows": 3,
            "git_commit": "a" * 40, "msd_version": "0.1.0", "timestamp_utc": "2026-11-05T10:00:00Z"}
    new_run = {**base, "n_rows": 9, "git_commit": "b" * 40, "msd_version": "0.2.0", "timestamp_utc": "2026-11-06T10:00:00Z"}
    assert genrun.meta_differences(base, new_run) == []  # a new commit or time is not a different setup
    for key, other in [("model_path", "m2"), ("arm", "A3"), ("alpha", 0.5), ("alpha", None), ("batch_size", 8),
                       ("max_new_tokens", 128), ("dtype", "bfloat16"), ("backend", "fake")]:
        assert [d[0] for d in genrun.meta_differences(base, {**new_run, key: other})] == [key]


def test_the_meta_file_never_holds_a_token(repo, scratch, hf, monkeypatch):
    token = "hf_" + "k3Y9z8X7w6" * 4
    monkeypatch.setenv("HF_TOKEN", token)
    build_testset(repo, n=2)
    argv = hf_args(scratch)
    argv[argv.index("org/model")] = f"org/{token}"
    assert run_cli(repo, *argv) == 0
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en"))
    assert token not in meta_file.read_text(encoding="utf-8")
    assert token not in gen.gens_path(scratch, "dev", "A0", None, "en").read_text(encoding="utf-8")


# --- hard kills: the meta must already be on disk -------------------------------------------------------------------------

KILLER = r"""
import os, sys
from pathlib import Path
from msd import gen, genrun
calls = {"n": 0}
real_fake = gen.fake_backend
def killing_fake():
    inner = real_fake()
    def run(prompts):
        calls["n"] += 1
        if calls["n"] == int(os.environ["KILL_ON_CALL"]):
            os._exit(137)  # what SIGKILL, the OOM killer or a Colab disconnect does: no handler runs
        return inner(prompts)
    return run
gen.fake_backend = killing_fake
sys.exit(genrun.main(sys.argv[2:], root=Path(sys.argv[1])))
"""


def run_killed(repo, kill_on_call, *argv):
    env = {**os.environ, "KILL_ON_CALL": str(kill_on_call)}
    return subprocess.run([sys.executable, "-c", KILLER, str(repo), *argv], capture_output=True, text=True, env=env)


def test_a_hard_kill_leaves_a_meta_that_guards_the_rerun(repo, scratch):
    build_testset(repo, n=8)
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    meta_file = gen.meta_path(gens_file)

    killed = run_killed(repo, 2, *fake_args(scratch, "--batch-size", "3"))
    assert killed.returncode == 137, killed.stderr
    assert gens_file.read_bytes().count(b"\n") == 3  # the first batch is on disk
    assert json.loads(meta_file.read_text(encoding="utf-8"))["batch_size"] == 3  # and so is its meta
    assert not meta_file.with_name(meta_file.name + ".tmp").exists()
    before = gens_file.read_bytes()

    assert run_cli(repo, *fake_args(scratch, "--batch-size", "5")) not in (0, 2)  # another setup must not join the file
    assert gens_file.read_bytes() == before
    assert json.loads(meta_file.read_text(encoding="utf-8"))["batch_size"] == 3

    assert run_cli(repo, *fake_args(scratch, "--batch-size", "3")) == 0  # the same setup resumes
    assert [r.prompt_id for r in read_jsonl(gens_file, Generation)] == [f"en-{i}" for i in range(8)]
    assert json.loads(meta_file.read_text(encoding="utf-8"))["n_rows"] == 8


def test_a_kill_in_the_middle_of_a_later_language_keeps_the_first_one_clean(repo, scratch):
    build_testset(repo, langs=("en", "hi"), n=4)
    killed = run_killed(repo, 4, *fake_args(scratch, "--langs", "en,hi", "--batch-size", "2"))  # hi, 2nd batch
    assert killed.returncode == 137
    for lang, rows in (("en", 4), ("hi", 2)):
        gens_file = gen.gens_path(scratch, "dev", "A0", None, lang)
        assert gens_file.read_bytes().count(b"\n") == rows and gen.meta_path(gens_file).is_file()
    assert run_cli(repo, *fake_args(scratch, "--langs", "en,hi", "--batch-size", "2")) == 0


def test_a_gens_file_with_rows_but_no_meta_is_refused_unless_forced(repo, scratch, capsys):
    build_testset(repo, n=6)
    assert run_cli(repo, *fake_args(scratch, "--limit", "3")) == 0
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    meta_file = gen.meta_path(gens_file)
    meta_file.unlink()  # a kill before the meta existed, or a deleted file
    before = gens_file.read_bytes()
    capsys.readouterr()

    assert run_cli(repo, *fake_args(scratch, "--limit", "6")) not in (0, 2)
    err = capsys.readouterr().err
    assert str(gens_file) in err and "--force-new-meta" in err and "no meta" in err
    assert gens_file.read_bytes() == before and not meta_file.exists()

    assert run_cli(repo, *fake_args(scratch, "--limit", "6", "--force-new-meta")) == 0
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["n_rows"] == 6 and meta["unrecorded_rows"] == 3  # the rows made without a record are marked as such
    assert run_cli(repo, *fake_args(scratch, "--limit", "6")) == 0  # and now the setup is on record


def test_the_meta_is_on_disk_before_the_first_batch_is_generated(repo, scratch, monkeypatch):
    build_testset(repo, n=5)
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A0", None, "en"))
    seen = []

    def spy_backend():
        def run(prompts):
            seen.append(json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else None)
            return [GenOut("x", "stop") for _ in prompts]

        return run

    monkeypatch.setattr(gen, "fake_backend", spy_backend)
    assert run_cli(repo, *fake_args(scratch, "--batch-size", "2")) == 0
    assert len(seen) == 3 and all(m is not None for m in seen)
    assert seen[0]["n_rows"] == 0 and seen[0]["batch_size"] == 2 and seen[0]["backend"] == "fake"
    assert json.loads(meta_file.read_text(encoding="utf-8"))["n_rows"] == 5  # updated at the end
    assert not meta_file.with_name(meta_file.name + ".tmp").exists()


def test_a_failure_before_any_row_does_not_lock_the_setup(repo, monkeypatch, scratch, capsys):
    build_testset(repo, n=4)
    monkeypatch.setattr(gen, "hf_backend", Hf(fail_on_call=1))  # for example an out-of-memory in the first batch
    assert run_cli(repo, *hf_args(scratch, "--batch-size", "32")) not in (0, 2)
    gens_file = gen.gens_path(scratch, "dev", "A0", None, "en")
    assert not gens_file.exists()  # no rows, so the leftover meta describes nothing
    monkeypatch.setattr(gen, "hf_backend", Hf())
    assert run_cli(repo, *hf_args(scratch, "--batch-size", "2")) == 0  # a smaller batch is not refused
    meta = json.loads(gen.meta_path(gens_file).read_text(encoding="utf-8"))
    assert meta["batch_size"] == 2 and "earlier" not in meta


# --- everything refusable is refused before the first generation -----------------------------------------------------------------


def test_a_duplicate_prompt_id_in_the_test_set_is_refused_before_any_language_runs(repo, hf, scratch, capsys):
    build_testset(repo, langs=("en", "hi"), n=3)
    rows = read_jsonl(repo / "data" / "testset" / "hi" / "dev.jsonl", Prompt)
    write_jsonl(repo / "data" / "testset" / "hi" / "dev.jsonl", [*rows, rows[0]])
    assert run_cli(repo, *hf_args(scratch, "--langs", "en,hi")) not in (0, 2)
    err = capsys.readouterr().err
    assert "duplicate" in err and "hi-0" in err
    assert hf.built == [] and not scratch.exists()  # not even English was generated


def test_a_corrupt_gens_file_of_a_later_language_is_found_before_an_earlier_one_runs(repo, hf, scratch, capsys):
    build_testset(repo, langs=("en", "hi"), n=4)
    hi_file = gen.gens_path(scratch, "dev", "A0", None, "hi")
    assert run_cli(repo, *hf_args(scratch, "--langs", "hi")) == 0
    lines = hi_file.read_bytes().split(b"\n")
    lines[1] = b"{not json"
    hi_file.write_bytes(b"\n".join(lines))
    built_before = len(hf.built)
    capsys.readouterr()

    assert run_cli(repo, *hf_args(scratch, "--langs", "en,hi")) not in (0, 2)
    assert "line 2" in capsys.readouterr().err
    assert len(hf.built) == built_before  # the model was not loaded
    assert not gen.gens_path(scratch, "dev", "A0", None, "en").exists()  # and English was not generated


def test_a_wrong_arm_row_of_a_later_language_is_found_before_an_earlier_one_runs(repo, scratch):
    build_testset(repo, langs=("en", "hi"), n=2)
    write_jsonl(gen.gens_path(scratch, "dev", "A1", None, "hi"), [Generation("hi-0", "A0", None, "x", "stop")])
    assert run_cli(repo, *fake_args(scratch, "--langs", "en,hi", arm="A1")) not in (0, 2)
    assert not gen.gens_path(scratch, "dev", "A1", None, "en").exists()


def test_load_prompts_refuses_duplicate_ids(tmp_path):
    path = tmp_path / "en" / "dev.jsonl"
    prompts = [Prompt("a", "en", "harmful", "t", "dev", "s", "l"), Prompt("a", "en", "harmful", "u", "dev", "s", "l")]
    write_jsonl(path, prompts)
    with pytest.raises(ValueError, match="duplicate.*'a'|'a'.*duplicate"):
        genrun.load_prompts(path, "en", "dev", None)


# --- the script itself ----------------------------------------------------------------------------------------------------------------


def run_script(*argv, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, cwd=cwd)


def test_script_help_and_usage_errors(tmp_path):
    assert run_script("--help").returncode == 0
    assert run_script().returncode == 2
    assert run_script("--arm", "A0", "--split", "dev", "--alpha", "0.5", "--backend", "fake",
                      "--out-root", str(tmp_path / "never-created")).returncode == 2
    assert not (tmp_path / "never-created").exists()


def test_script_fake_run_end_to_end_and_guard(tmp_path):
    testset_dir = tmp_path / "ts"
    build_testset(None, n=5, testset_dir=testset_dir)
    out = tmp_path / "out"
    done = run_script("--arm", "A2", "--alpha", "0.5", "--split", "dev", "--backend", "fake", "--langs", "en,hi",
                      "--limit", "3", "--testset-dir", str(testset_dir), "--out-root", str(out))
    assert done.returncode == 0, done.stderr
    assert MARK not in done.stdout + done.stderr
    assert len(read_jsonl(out / "dev" / "A2" / "0.5" / "en.gens.jsonl", Generation)) == 3
    assert "hi" in done.stderr and "not built" in done.stderr

    # the guard, with the cwd in the temp dir so that a regression could only ever write under tmp_path
    guard = run_script("--arm", "A0", "--split", "dev", "--backend", "fake", "--testset-dir", str(testset_dir),
                       "--out-root", "results", cwd=tmp_path)
    assert guard.returncode == 2 and not (tmp_path / "results").exists()
    guard = run_script("--arm", "A0", "--split", "dev", "--backend", "fake", "--testset-dir", str(testset_dir),
                       "--out-root", "Results/x", cwd=tmp_path)
    assert guard.returncode == 2 and not (tmp_path / "Results").exists()


def test_script_exit_code_is_nonzero_when_nothing_could_be_generated(tmp_path):
    done = run_script("--arm", "A0", "--split", "dev", "--backend", "fake", "--langs", "sw",
                      "--testset-dir", str(tmp_path / "empty"), "--out-root", str(tmp_path / "out"))
    assert done.returncode not in (0, 2)
