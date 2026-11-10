"""scripts/run_reliability.py and msd.reliabilityrun: a second judge re-labels about 60 items per language (Task 17).

A throwaway results tree (tests/results_tree.py), a throwaway test set, and injected judges: no model, no network,
no paid API, and never the repository's real results/ directory.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from results_tree import JUDGE, SECOND_JUDGE, SENTINEL, build_tree, edit_cell

from msd import config, evalrun, gen, reliabilityrun, reportrun
from msd.schema import Generation, Prompt, Verdict, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_reliability.py"
LANGS = config.LANGS


# --- fixtures ---------------------------------------------------------------------------------------------------


def write_testset(repo):
    """The prompts of the fixture tree's report split: <lang>-h000.. harmful, <lang>-b000.. benign_edgy."""
    for lang in LANGS:
        rows = [Prompt(f"{lang}-h{i:03d}", lang, "harmful", f"fixture harmful request {i}", "report", "fixture", "fixture")
                for i in range(40)]
        rows += [Prompt(f"{lang}-b{i:03d}", lang, "benign_edgy", f"fixture benign request {i}", "report", "fixture",
                        "fixture") for i in range(20)]
        write_jsonl(repo / "data" / "testset" / lang / "report.jsonl", rows)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    repo = tmp_path_factory.mktemp("base") / "repo"
    write_testset(repo)
    build_tree(repo / "scratch_results")
    return repo


@pytest.fixture
def repo(base, tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(base, dest)
    return dest


@pytest.fixture
def results(repo):
    return repo / "scratch_results"


def truth_table(results):
    """completion text -> the first judge's label for it (the fixture's texts name their cell, so they are unique)."""
    labels = {}
    for path in sorted((results / "report").glob("*/*/*.verdicts.jsonl")):
        if path.relative_to(results / "report").parts[0].startswith("_") or "reliability" in path.parts:
            continue
        verdicts = {v.prompt_id: v for v in read_jsonl(path, Verdict)}
        gens = read_jsonl(path.with_name(path.name.replace(".verdicts.", ".gens.")), Generation)
        for g in gens:
            if g.prompt_id in verdicts:
                labels[g.text] = verdicts[g.prompt_id].label
    return labels


class Second:
    """A scripted second judge. `mode`: "agree" repeats the first judge's label for the same completion, "coin"
    answers from a hash of the completion (about half agree), "always" answers the same word."""

    def __init__(self, results, judge_id=SECOND_JUDGE, mode="agree", usage=False):
        self.id = judge_id
        self.mode = mode
        self.truth = truth_table(results) if mode == "agree" else {}
        self.calls = []
        if usage:
            self.usage = lambda: {"calls": len(self.calls), "input_tokens": 100 * len(self.calls),
                                  "output_tokens": 2 * len(self.calls)}

    def classify(self, kind, prompt, response):
        self.calls.append((kind, prompt, response))
        if self.mode == "agree":
            return self.truth[response]
        if self.mode == "coin":
            return "complied" if hashlib.sha256(response.encode()).digest()[0] % 2 == 0 else "refused"
        return "complied"


def register(monkeypatch, name, judge):
    monkeypatch.setitem(evalrun.JUDGE_REGISTRY, name, lambda: judge)
    return judge


@pytest.fixture
def run(capsys):
    def go(repo, results, *extra, judge="second"):
        code = reliabilityrun.main(["--results", str(results), "--judge", judge, *extra], root=repo)
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return go


def second_files(results, judge=SECOND_JUDGE):
    base = results / "report" / "reliability" / evalrun.judge_dirname(judge)
    return sorted(base.rglob("*.verdicts.jsonl")) if base.is_dir() else []


def second_rows(results, judge=SECOND_JUDGE):
    return [v for path in second_files(results, judge) for v in read_jsonl(path, Verdict)]


# --- what it writes --------------------------------------------------------------------------------------------


def test_it_writes_the_second_judges_labels_where_make_report_looks(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    code, out, err = run(repo, results)
    assert code == 0, err
    files = second_files(results)
    assert files
    base = results / "report" / "reliability" / SECOND_JUDGE
    for path in files:
        arm, alpha, name = path.relative_to(base).parts
        lang = name.split(".")[0]
        assert path == evalrun.verdicts_path(results, "report", arm, None if alpha == "none" else float(alpha), lang,
                                             reliability_judge=SECOND_JUDGE)
    rows = second_rows(results)
    assert {v.judge_id for v in rows} == {SECOND_JUDGE}
    assert all(len(v.excerpt) <= 200 for v in rows)
    per_lang = {lang: sum(v.lang == lang for v in rows) for lang in LANGS}
    assert per_lang == {lang: config.RELIABILITY_N for lang in LANGS}  # about 60 items per language


def test_the_items_are_a_subset_of_the_first_judges_items_and_balanced_over_arms(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    assert run(repo, results)[0] == 0
    first = {(v.prompt_id, v.arm, v.alpha) for v in
             (v for p in (results / "report").glob("*/*/*.verdicts.jsonl")
              if not p.relative_to(results / "report").parts[0].startswith("_") and "reliability" not in p.parts
              for v in read_jsonl(p, Verdict))}
    rows = second_rows(results)
    assert {(v.prompt_id, v.arm, v.alpha) for v in rows} <= first
    by_arm = {arm: sum(v.arm == arm and v.lang == "hi" for v in rows) for arm in ("A0", "A1", "A2")}
    assert by_arm == {"A0": 20, "A1": 20, "A2": 20}  # an equal quota per arm, shared by the alphas of A2


def test_per_lang_and_langs_limit_the_sample(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    code, out, err = run(repo, results, "--per-lang", "10", "--langs", "hi,ar")
    assert code == 0, err
    rows = second_rows(results)
    assert {v.lang for v in rows} == {"hi", "ar"} and all(sum(v.lang == l for v in rows) == 10 for l in ("hi", "ar"))


def test_items_of_an_excluded_cell_are_never_double_judged(repo, results, run, monkeypatch):
    edit_cell(results, "report", "A1", None, "hi", judge_errors=8)  # over the allowance: not part of the table
    register(monkeypatch, "second", Second(results))
    assert run(repo, results)[0] == 0
    assert not any(v.arm == "A1" and v.lang == "hi" for v in second_rows(results))


def test_no_text_is_printed_or_stored_beyond_the_excerpt(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    code, out, err = run(repo, results)
    assert code == 0
    assert "Sure, here is" not in out + err and "fixture harmful request" not in out + err
    for path in second_files(results):
        assert "fixture harmful request" not in path.read_text(encoding="utf-8")


# --- what it says ----------------------------------------------------------------------------------------------


def test_it_prints_kappa_with_an_interval_per_language_and_names_the_low_confidence_ones(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results, mode="coin"))
    code, out, err = run(repo, results)
    assert code == 0, err
    for lang in LANGS:
        line = next(line for line in out.splitlines() if line.split() and line.split()[0] == lang)
        assert "kappa" in line and "[" in line and "pairs" in line
    named = next(line for line in out.splitlines() if line.startswith("low-confidence languages:"))
    assert all(lang in named for lang in LANGS)  # a coin flip agrees with nobody


def test_agreeing_judges_leave_no_language_low_confidence(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    code, out, err = run(repo, results)
    assert code == 0, err
    assert "low-confidence languages: none" in out
    for lang in LANGS:
        line = next(line for line in out.splitlines() if line.split() and line.split()[0] == lang)
        assert "kappa 1.00" in line


def test_a_judge_that_always_says_the_same_word_is_named_low_confidence_everywhere(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results, mode="always"))
    code, out, err = run(repo, results)
    assert code == 0
    named = next(line for line in out.splitlines() if line.startswith("low-confidence languages:"))
    assert all(lang in named for lang in LANGS)


def test_judge_usage_is_printed_like_run_eval_does(repo, results, run, monkeypatch):
    judge = register(monkeypatch, "second", Second(results, usage=True))
    code, out, err = run(repo, results)
    assert code == 0
    assert f"judge usage: calls {len(judge.calls)}, input_tokens {100 * len(judge.calls)}, output_tokens" in err


def test_judge_calls_that_raised_are_summarised_by_type_never_by_message(repo, results, run, monkeypatch):
    class Flaky(Second):
        def classify(self, kind, prompt, response):
            if len(self.calls) % 7 == 0:
                self.calls.append(None)
                raise TimeoutError("SECRET-MESSAGE")
            return super().classify(kind, prompt, response)

    register(monkeypatch, "second", Flaky(results, mode="always"))
    code, out, err = run(repo, results)
    assert "TimeoutError x" in err and "SECRET-MESSAGE" not in out + err


def test_a_rerun_uses_the_judge_cache_and_asks_nothing_twice(repo, results, run, monkeypatch):
    judge = register(monkeypatch, "second", Second(results))
    assert run(repo, results)[0] == 0
    first_calls = len(judge.calls)
    assert first_calls > 0
    before = {p: p.read_bytes() for p in second_files(results)}
    assert run(repo, results)[0] == 0
    assert len(judge.calls) == first_calls
    assert {p: p.read_bytes() for p in second_files(results)} == before


# --- refusals ---------------------------------------------------------------------------------------------------


def test_the_first_judge_cannot_be_its_own_second_opinion(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results, judge_id=JUDGE))
    code, out, err = run(repo, results)
    assert code == reliabilityrun.EXIT_JUDGE and "same judge" in err
    assert second_files(results, JUDGE) == []


def test_a_test_judge_is_refused_on_the_real_results_directory(repo, tmp_path, run, monkeypatch):
    real = repo / "results"
    shutil.copytree(repo / "scratch_results", real)
    register(monkeypatch, "second", evalrun.FakeJudge("fake-second-v1"))
    code, out, err = run(repo, real)
    assert code == reliabilityrun.EXIT_USAGE and "test judge" in err
    assert not (real / "report" / "reliability").exists()


def test_a_test_judge_hidden_by_a_decoding_fault_is_refused_on_the_real_results_directory(repo, tmp_path, run, monkeypatch):
    real = repo / "results"
    shutil.copytree(repo / "scratch_results", real)
    # not `is_fake`: only the id says it, so only the id check (`looks_fake`) can refuse it
    register(monkeypatch, "second", Second(real, judge_id="fa" + chr(0xFFFD) + "ke-second-v1"))
    code, out, err = run(repo, real)
    assert code == reliabilityrun.EXIT_USAGE and "test judge" in err
    assert not (real / "report" / "reliability").exists()


def test_a_test_judge_hidden_by_a_decoding_fault_is_still_called_a_test_judge_in_a_scratch_directory(repo, results, run, monkeypatch):
    register(monkeypatch, "second", evalrun.FakeJudge("fa" + chr(0xFFFD) + "ke-second-v1"))
    code, out, err = run(repo, results)
    assert code == 0, err
    assert "test judge" in err and "canned" in err  # the note, and make_report refuses to read it (the same check)


def test_a_test_judge_works_in_a_scratch_directory(repo, results, run, monkeypatch):
    register(monkeypatch, "second", evalrun.FakeJudge("fake-second-v1"))
    code, out, err = run(repo, results)
    assert code == 0, err
    assert second_files(results, "fake-second-v1")


def test_a_test_judge_in_a_scratch_directory_says_its_kappa_means_nothing_and_make_report_never_reads_it(
    repo, results, run, monkeypatch, tmp_path, capsys
):
    register(monkeypatch, "second", evalrun.FakeJudge("fake-second-v1"))
    code, out, err = run(repo, results)
    assert code == 0 and "test judge" in err and "canned" in err  # the run is allowed, and honest about it
    out_dir = tmp_path / "out"
    code = reportrun.main(["--verdicts", str(results), "--out", str(out_dir)], root=repo)
    captured = capsys.readouterr()
    assert code == reportrun.EXIT_JUDGE and "test judge" in captured.err  # a report never reads canned labels
    assert not (out_dir / "tables.csv").exists()


def test_an_unknown_judge_name_is_a_usage_error(repo, results, run):
    code, out, err = run(repo, results, judge="no-such-judge")
    assert code == reliabilityrun.EXIT_USAGE and "no-such-judge" in err


def test_a_missing_test_set_file_is_a_plain_error(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    (repo / "data" / "testset" / "sw" / "report.jsonl").unlink()
    code, out, err = run(repo, results)
    assert code == reportrun.EXIT_ERROR and "sw" in err and "report.jsonl" in err and "Traceback" not in err
    assert second_files(results) == []


def test_a_missing_gens_file_is_a_plain_error(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    gen.gens_path(results, "report", "A1", None, "hi").unlink()
    code, out, err = run(repo, results)
    assert code == reportrun.EXIT_ERROR and "gens" in err and "Traceback" not in err


def test_labels_left_by_a_different_sample_are_refused(repo, results, run, monkeypatch):
    register(monkeypatch, "second", Second(results))
    assert run(repo, results, "--per-lang", "20")[0] == 0
    code, out, err = run(repo, results, "--per-lang", "5")
    assert code == reportrun.EXIT_ERROR and "another sample" in err


def test_the_first_judges_mix_is_refused_before_anything_is_asked(repo, results, run, monkeypatch):
    judge = register(monkeypatch, "second", Second(results))
    edit_cell(results, "report", "A2", 0.25, "sw", judge="other-judge-v1")
    code, out, err = run(repo, results)
    assert code == reportrun.EXIT_JUDGE and judge.calls == []


# --- together with make_report ------------------------------------------------------------------------------------


def test_make_report_reads_what_run_reliability_wrote(repo, results, run, monkeypatch, tmp_path, capsys):
    register(monkeypatch, "second", Second(results, mode="coin"))
    assert run(repo, results)[0] == 0
    out_dir = tmp_path / "out"
    code = reportrun.main(["--verdicts", str(results), "--out", str(out_dir)], root=repo)
    capsys.readouterr()
    assert code == 0
    rel = json.loads((out_dir / "reliability.json").read_text(encoding="utf-8"))
    assert rel["second_judge_id"] == SECOND_JUDGE
    assert sorted(rel["low_confidence"]) == sorted(LANGS)  # the coin-flip judge disagrees everywhere
    assert all(rel["kappa"][lang]["n_pairs"] > 0 for lang in LANGS)


def test_make_report_calls_agreeing_languages_confident(repo, results, run, monkeypatch, tmp_path, capsys):
    register(monkeypatch, "second", Second(results))
    assert run(repo, results)[0] == 0
    out_dir = tmp_path / "out"
    code = reportrun.main(["--verdicts", str(results), "--out", str(out_dir)], root=repo)
    capsys.readouterr()
    assert code == 0
    rel = json.loads((out_dir / "reliability.json").read_text(encoding="utf-8"))
    assert rel["low_confidence"] == {}
    assert all(rel["kappa"][lang]["kappa"] == 1.0 for lang in LANGS)


def test_the_script_runs_and_helps(tmp_path):
    done = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "--judge" in done.stdout
    bad = subprocess.run([sys.executable, str(SCRIPT), "--results", str(tmp_path / "none"), "--judge", "nope"],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == reliabilityrun.EXIT_USAGE


# --- review round 1 ----------------------------------------------------------------------------------------------


def test_the_first_model_under_another_casing_is_still_the_first_judge(repo, results, run, monkeypatch):
    judge = register(monkeypatch, "second", Second(results, judge_id=JUDGE.upper()))
    code, out, err = run(repo, results)
    assert code == reliabilityrun.EXIT_JUDGE and "same judge" in err and judge.calls == []
    assert second_files(results, JUDGE.upper()) == []


def test_dev_can_come_from_another_folder_so_the_sample_matches_the_table(repo, results, run, monkeypatch, tmp_path):
    dev_root = tmp_path / "devroot"
    dev_root.mkdir()
    shutil.move(str(results / "dev"), str(dev_root / "dev"))
    register(monkeypatch, "second", Second(results))
    code, out, err = run(repo, results, "--dev", str(dev_root))
    assert code == 0, err
    assert any(v.arm == "A2" for v in second_rows(results))  # the repair cells passed the gate, so they are in the table


def test_without_a_dev_folder_the_unchecked_repair_cells_are_not_double_judged(repo, results, run, monkeypatch, tmp_path):
    shutil.move(str(results / "dev"), str(tmp_path / "dev_away"))
    register(monkeypatch, "second", Second(results))
    code, out, err = run(repo, results)
    assert code == 0, err
    assert not any(v.arm == "A2" for v in second_rows(results))  # not in the table, so not in the sample
