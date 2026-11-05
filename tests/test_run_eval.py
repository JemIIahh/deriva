"""scripts/run_eval.py and msd.evalrun: the evaluation orchestrator.

Everything runs against a throwaway repo layout under tmp_path (data/testset/<lang>/<split>.jsonl and
data/ft/{heldout,taskdev}.jsonl), a scripted backend standing in for the model and stub judges. No model,
no network, and never the repository's real results/ directory.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from msd import evalrun, gen, genrun
from msd.aggregate import build_table
from msd.evalrun import JudgeMixError
from msd.gen import GenOut
from msd.schema import Generation, Prompt, TaskScore, Verdict, read_jsonl, write_jsonl

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py"
MARK = "SENTINEL-PROMPT-TEXT"  # stands in for safety prompt text; it must never be printed or stored in eval.json
TASK_MARK = "SENTINEL-TASK-TEXT"  # same, for task questions
ANSWER = "SENTINEL-COMPLETION-TEXT"  # same, for model completions


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "testset").mkdir(parents=True)
    (root / "data" / "ft").mkdir(parents=True)
    return root


@pytest.fixture
def scratch(tmp_path):
    return tmp_path / "scratch_out"


# --- fixture data ------------------------------------------------------------------------------------------


def build_testset(root, langs=("en",), split="dev", n=6):
    base = Path(root) / "data" / "testset"
    for lang in langs:
        write_jsonl(
            base / lang / f"{split}.jsonl",
            [Prompt(f"{lang}-{i}", lang, "harmful" if i % 2 == 0 else "benign_edgy", f"{MARK} {lang} {i}", split,
                    "fixture", "fixture") for i in range(n)],
        )


def task_gold(i):
    return "B" if i % 2 == 0 else "C"


def task_record(lang, i):
    return {
        "id": f"{lang}:subject/test/{i}",
        "lang": lang,
        "messages": [
            {"role": "user", "content": f"{TASK_MARK} {lang} question {i}\nA. one\nB. two\nC. three\n"
                                        "Answer with only the letter of the correct option."},
            {"role": "assistant", "content": task_gold(i)},
        ],
        "gold": task_gold(i),
    }


def build_tasks(root, name="heldout.jsonl", langs=("en",), n=4):
    path = Path(root) / "data" / "ft" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(task_record(lang, i)) + "\n" for lang in langs for i in range(n)),
                    encoding="utf-8")
    return path


# --- scripted model and stub judges --------------------------------------------------------------------------


def default_text(prompt):
    return "B" if TASK_MARK in prompt else f"Sure, here you go. {ANSWER}"


class Scripted:
    """Stands in for gen.hf_backend: counts how many times a model would have been loaded."""

    def __init__(self, text_for=default_text, finish_for=lambda prompt: "stop"):
        self.text_for, self.finish_for = text_for, finish_for
        self.built = 0
        self.prompts_seen = 0

    def __call__(self, model_path, *, max_new_tokens=256, dtype="float16"):
        self.built += 1

        def backend(texts):
            self.prompts_seen += len(texts)
            return [GenOut(self.text_for(t), self.finish_for(t)) for t in texts]

        return backend


@pytest.fixture
def model(monkeypatch):
    fake = Scripted()
    monkeypatch.setattr(gen, "hf_backend", fake)
    return fake


def use_model(monkeypatch, **kwargs):
    fake = Scripted(**kwargs)
    monkeypatch.setattr(gen, "hf_backend", fake)
    return fake


class StubJudge:
    """complied for harmful, refused for benign_edgy; `fail_if` makes a call answer unparseably (judge_error)."""

    def __init__(self, judge_id="J1", fail_if=None):
        self.id = judge_id
        self.fail_if = fail_if or (lambda kind, prompt, response: False)
        self.calls = []

    def classify(self, kind, prompt, response):
        self.calls.append((kind, prompt, response))
        if self.fail_if(kind, prompt, response):
            return "I am unable to decide"
        return "complied" if kind == "harmful" else "refused"


def register(monkeypatch, name, judge):
    monkeypatch.setitem(evalrun.JUDGE_REGISTRY, name, lambda: judge)
    return judge


@pytest.fixture
def judge(monkeypatch):
    return register(monkeypatch, "j1", StubJudge("J1"))


# --- running -------------------------------------------------------------------------------------------------


def args_for(scratch, *extra, arm="A1", split="dev", judge_name="j1", model_args=True):
    argv = ["--arm", arm, "--split", split, "--out", str(scratch), "--judge", judge_name]
    if model_args:
        argv += ["--backend", "hf", "--model", "org/model"]
    return [*argv, *extra]


def run_cli(repo, *argv):
    return evalrun.main(list(argv), root=repo)


def evaluate(repo, *argv):
    return evalrun.evaluate(evalrun.parse_args(list(argv), root=repo), root=repo)


def files_under(path):
    return sorted(str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file())


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in Path(path).rglob("*") if p.is_file()}


def eval_json(scratch, split="dev", arm="A1", alpha=None, lang="en"):
    return json.loads(evalrun.eval_path(scratch, split, arm, alpha, lang).read_text(encoding="utf-8"))


# --- path helpers ----------------------------------------------------------------------------------------------


def test_path_helpers_build_the_split_qualified_layout(tmp_path):
    out = tmp_path / "res"
    base = out / "dev" / "A2" / "0.25"
    assert gen.gens_path(out, "dev", "A2", 0.25, "hi") == base / "hi.gens.jsonl"
    assert evalrun.verdicts_path(out, "dev", "A2", 0.25, "hi") == base / "hi.verdicts.jsonl"
    assert evalrun.tasks_path(out, "dev", "A2", 0.25, "hi") == base / "hi.tasks.jsonl"
    assert evalrun.eval_path(out, "dev", "A2", 0.25, "hi") == base / "hi.eval.json"
    assert evalrun.tasks_path(out, "report", "A1", None, "en") == out / "report" / "A1" / "none" / "en.tasks.jsonl"
    assert evalrun.cache_path(out, "report", "judge-1") == out / "report" / "_cache" / "judge-1.jsonl"
    assert evalrun.verdicts_path(out, "dev", "A2", 0.25, "hi", reliability_judge="judge-2") == (
        out / "dev" / "reliability" / "judge-2" / "A2" / "0.25" / "hi.verdicts.jsonl")
    assert evalrun.replaced_verdicts_path(out, "dev", "judge-1", "A1", None, "en") == (
        out / "dev" / "_replaced" / "judge-1" / "A1" / "none" / "en.verdicts.jsonl")


def test_alpha_is_canonical_in_every_path(tmp_path):
    assert evalrun.verdicts_path(tmp_path, "dev", "A2", 0.3000004, "en") == (
        tmp_path / "dev" / "A2" / "0.3" / "en.verdicts.jsonl")
    assert evalrun.verdicts_path(tmp_path, "dev", "A2", 1, "en").parent.name == "1.0"


@pytest.mark.parametrize(
    "call",
    [
        lambda f, p: f(p, "train", "A1", None, "en"),  # unknown split
        lambda f, p: f(p, "dev", "A1", None, "fr"),  # unknown language
        lambda f, p: f(p, "dev", "A9", None, "en"),  # unknown arm
        lambda f, p: f(p, "dev", "A1", 0.5, "en"),  # alpha on an arm without one
        lambda f, p: f(p, "dev", "A2", None, "en"),  # an arm that needs alpha
    ],
)
@pytest.mark.parametrize("helper", ["verdicts_path", "tasks_path", "eval_path"])
def test_path_helpers_validate_like_gens_path(tmp_path, helper, call):
    with pytest.raises(ValueError):
        call(getattr(evalrun, helper), tmp_path)


def test_cache_and_replaced_paths_validate_the_split(tmp_path):
    with pytest.raises(ValueError):
        evalrun.cache_path(tmp_path, "train", "j")
    with pytest.raises(ValueError):
        evalrun.replaced_verdicts_path(tmp_path, "train", "j", "A1", None, "en")


def test_judge_ids_become_safe_collision_free_file_names():
    assert evalrun.judge_dirname("judge-1") == "judge-1"
    assert evalrun.judge_dirname("fake-judge-v1") == "fake-judge-v1"
    ids = ["a/b", "a_b", "A", "a", "..", ".", "a b", "a\\b", "x" * 300, "x" * 301, "claude-haiku@2026-11", "Ünï",
           "judge:one", "judge.one", "a/b ", "CON"]
    names = [evalrun.judge_dirname(i) for i in ids]
    assert len(set(names)) == len(ids)  # distinct ids never share a name
    for name in names:
        assert re.fullmatch(r"[a-z0-9@._-]+", name), name
        assert name not in {".", ".."} and not name.startswith(".") and len(name) <= 100
    assert evalrun.judge_dirname("a/b") == evalrun.judge_dirname("a/b")  # stable
    with pytest.raises(ValueError):
        evalrun.judge_dirname("")


# --- the file set per (arm, alpha, lang) ---------------------------------------------------------------------


def test_writes_one_file_set_per_arm_alpha_lang(repo, scratch, model, judge, capsys):
    langs = ("en", "hi")
    build_testset(repo, langs, split="report", n=6)
    build_tasks(repo, "heldout.jsonl", langs, n=4)
    assert run_cli(repo, *args_for(scratch, arm="A1", split="report")) == 0
    assert run_cli(repo, *args_for(scratch, "--alpha", "0.25", arm="A2", split="report")) == 0
    for arm, alpha, alpha_dir in (("A1", None, "none"), ("A2", 0.25, "0.25")):
        for lang in langs:
            folder = scratch / "report" / arm / alpha_dir
            assert sorted(p.name for p in folder.glob(f"{lang}.*")) == [
                f"{lang}.eval.json", f"{lang}.gens.jsonl", f"{lang}.gens.meta.json", f"{lang}.tasks.jsonl",
                f"{lang}.verdicts.jsonl",
            ]
            gens = read_jsonl(gen.gens_path(scratch, "report", arm, alpha, lang), Generation)
            verdicts = read_jsonl(evalrun.verdicts_path(scratch, "report", arm, alpha, lang), Verdict)
            scores = read_jsonl(evalrun.tasks_path(scratch, "report", arm, alpha, lang), TaskScore)
            assert len(gens) == 6 + 4  # safety and task completions share one gens file
            assert [v.prompt_id for v in verdicts] == [f"{lang}-{i}" for i in range(6)]
            assert {v.kind for v in verdicts} == {"harmful", "benign_edgy"}
            assert {v.judge_id for v in verdicts} == {"J1"} and {v.arm for v in verdicts} == {arm}
            assert [s.prompt_id for s in scores] == [f"{lang}:subject/test/{i}" for i in range(4)]
            assert [s.correct for s in scores] == [True, False, True, False]  # the scripted model always says B
            assert all(s.arm == arm and s.alpha == alpha and s.lang == lang for s in scores)
    assert files_under(scratch / "report") == sorted(
        [f"{arm}/{a}/{lang}.{ext}" for arm, a in (("A1", "none"), ("A2", "0.25")) for lang in langs
         for ext in ("eval.json", "gens.jsonl", "gens.meta.json", "tasks.jsonl", "verdicts.jsonl")]
        + [f"_cache/{evalrun.judge_dirname('J1')}.jsonl"])
    shown = capsys.readouterr()
    assert MARK not in shown.out + shown.err and TASK_MARK not in shown.out + shown.err
    assert ANSWER not in shown.out + shown.err


def test_the_judge_cache_is_shared_across_arms_and_languages(repo, scratch, monkeypatch, judge):
    build_testset(repo, ("en",), n=6)
    use_model(monkeypatch)
    assert run_cli(repo, *args_for(scratch, arm="A0")) == 0
    first_calls = len(judge.calls)
    assert first_calls == 6
    assert run_cli(repo, *args_for(scratch, arm="A1")) == 0  # same prompts and completions: all cache hits
    assert len(judge.calls) == first_calls
    assert files_under(scratch / "dev" / "_cache") == [evalrun.judge_dirname("J1") + ".jsonl"]


def test_alpha_none_and_no_alpha_are_the_same_path(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--alpha", "none")) == 0
    before = snapshot(scratch)
    assert run_cli(repo, *args_for(scratch)) == 0
    assert snapshot(scratch) == before
    assert (scratch / "dev" / "A1" / "none" / "en.verdicts.jsonl").is_file()


# --- stages ----------------------------------------------------------------------------------------------------


def test_default_stages_are_generate_judge_and_task():
    assert evalrun.STAGES == ("generate", "judge", "task")


def test_generate_stage_alone_needs_no_judge_and_writes_gens_meta_and_eval_json(repo, scratch, model):
    build_testset(repo, n=4)
    build_tasks(repo, "taskdev.jsonl", n=3)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "hf", "--model", "m",
                   "--stages", "generate") == 0
    assert files_under(scratch) == ["dev/A1/none/en.eval.json", "dev/A1/none/en.gens.jsonl",
                                    "dev/A1/none/en.gens.meta.json"]
    assert len(read_jsonl(gen.gens_path(scratch, "dev", "A1", None, "en"), Generation)) == 4 + 3
    report = eval_json(scratch)
    assert report["verdicts"] is None and report["tasks"] is None and report["judge_id"] is None
    assert report["n_generations"] == 7


def test_judge_and_task_stages_work_from_existing_gens_without_a_model(repo, scratch, monkeypatch, judge):
    build_testset(repo, n=4)
    build_tasks(repo, "taskdev.jsonl", n=3)
    fake = use_model(monkeypatch)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    assert fake.built == 1
    gens_before = gen.gens_path(scratch, "dev", "A1", None, "en").read_bytes()
    monkeypatch.setattr(gen, "hf_backend", lambda *a, **k: pytest.fail("the model must not be loaded"))
    # no --model, no --backend: judging happens elsewhere
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 0
    assert evalrun.verdicts_path(scratch, "dev", "A1", None, "en").is_file()
    assert not evalrun.tasks_path(scratch, "dev", "A1", None, "en").exists()
    assert len(judge.calls) == 4
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--stages", "task") == 0
    assert len(read_jsonl(evalrun.tasks_path(scratch, "dev", "A1", None, "en"), TaskScore)) == 3
    assert len(judge.calls) == 4  # scoring tasks never touches the judge
    assert gen.gens_path(scratch, "dev", "A1", None, "en").read_bytes() == gens_before
    report = eval_json(scratch)
    assert report["verdicts"]["n"] == 4 and report["tasks"]["n"] == 3 and report["judge_id"] == "J1"


def test_judge_stage_with_missing_gens_is_a_clear_error_and_writes_nothing(repo, scratch, judge, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1
    err = capsys.readouterr().err
    assert "en.gens.jsonl" in err and "generate" in err
    assert not scratch.exists() or files_under(scratch) == []
    assert judge.calls == []


def test_task_stage_with_missing_gens_is_a_clear_error(repo, scratch, capsys):
    build_testset(repo, n=4)
    build_tasks(repo, "taskdev.jsonl", n=2)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--stages", "task") == 1
    assert "generate" in capsys.readouterr().err


def test_missing_gens_of_a_later_language_is_found_before_an_earlier_one_is_judged(repo, scratch, monkeypatch, judge):
    build_testset(repo, ("en", "hi"), n=4)
    use_model(monkeypatch)
    assert run_cli(repo, *args_for(scratch, "--langs", "en", "--stages", "generate")) == 0
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1", "--langs", "en,hi",
                   "--stages", "judge") == 1
    assert judge.calls == [] and not evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists()


def test_stage_order_on_the_command_line_does_not_matter(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "task,judge,generate")) == 0
    assert evalrun.verdicts_path(scratch, "dev", "A1", None, "en").is_file()


# --- tasks ----------------------------------------------------------------------------------------------------


def test_missing_task_file_is_skipped_loudly_not_an_error(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)  # the dev split wants data/ft/taskdev.jsonl, which does not exist
    assert run_cli(repo, *args_for(scratch)) == 0
    out = capsys.readouterr().out
    assert "task: skipped (no " in out and "taskdev.jsonl" in out
    assert not evalrun.tasks_path(scratch, "dev", "A1", None, "en").exists()
    assert len(read_jsonl(gen.gens_path(scratch, "dev", "A1", None, "en"), Generation)) == 4
    report = eval_json(scratch)
    assert "taskdev.jsonl" in report["task_skipped"] and report["tasks"] is None


def test_the_run_summary_carries_task_skipped(repo, scratch, model, judge):
    build_testset(repo, n=4)
    summary = evaluate(repo, *args_for(scratch))
    assert "taskdev.jsonl" in summary.task_skipped
    build_tasks(repo, "taskdev.jsonl", n=2)
    summary = evaluate(repo, *args_for(scratch))
    assert summary.task_skipped is None


def test_report_split_uses_heldout_and_dev_split_uses_taskdev(repo, scratch, model, judge):
    build_testset(repo, split="dev", n=2)
    build_testset(repo, split="report", n=2)
    build_tasks(repo, "taskdev.jsonl", n=3)
    heldout = build_tasks(repo, "heldout.jsonl", n=5)
    assert run_cli(repo, *args_for(scratch, split="dev")) == 0
    assert run_cli(repo, *args_for(scratch, split="report")) == 0
    assert len(read_jsonl(evalrun.tasks_path(scratch, "dev", "A1", None, "en"), TaskScore)) == 3
    assert len(read_jsonl(evalrun.tasks_path(scratch, "report", "A1", None, "en"), TaskScore)) == 5
    assert heldout.name == "heldout.jsonl"


def test_ft_dir_option(repo, scratch, model, judge, tmp_path):
    build_testset(repo, n=2)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "taskdev.jsonl").write_text(json.dumps(task_record("en", 0)) + "\n", encoding="utf-8")
    assert run_cli(repo, *args_for(scratch, "--ft-dir", str(other))) == 0
    assert len(read_jsonl(evalrun.tasks_path(scratch, "dev", "A1", None, "en"), TaskScore)) == 1


def test_a_language_with_no_task_rows_is_skipped_for_that_language_only(repo, scratch, model, judge, capsys):
    build_testset(repo, ("en", "hi"), n=4)
    build_tasks(repo, "taskdev.jsonl", ("en",), n=3)
    assert run_cli(repo, *args_for(scratch)) == 0
    out = capsys.readouterr().out
    assert "task: skipped" in out and "hi" in out
    assert evalrun.tasks_path(scratch, "dev", "A1", None, "en").is_file()
    assert not evalrun.tasks_path(scratch, "dev", "A1", None, "hi").exists()
    assert eval_json(scratch, lang="hi")["task_skipped"] and eval_json(scratch, lang="en")["task_skipped"] is None


def test_task_items_are_never_judged_and_safety_items_never_task_scored(repo, scratch, model, judge, monkeypatch):
    build_testset(repo, n=6)
    build_tasks(repo, "taskdev.jsonl", n=4)
    scored = []
    real = evalrun.score_task
    monkeypatch.setattr(evalrun, "score_task", lambda pred, gold: scored.append((pred, gold)) or real(pred, gold))
    assert run_cli(repo, *args_for(scratch)) == 0
    assert len(judge.calls) == 6
    assert {kind for kind, _, _ in judge.calls} == {"harmful", "benign_edgy"}
    assert all(TASK_MARK not in prompt and "B" != response for _, prompt, response in judge.calls)
    assert len(scored) == 4
    assert all(gold in {"B", "C"} and pred == "B" for pred, gold in scored)  # never a safety completion
    assert not any(ANSWER in pred for pred, _ in scored)


def test_limit_applies_to_safety_prompts_and_task_items(repo, scratch, model, judge):
    build_testset(repo, n=8)
    build_tasks(repo, "taskdev.jsonl", n=8)
    assert run_cli(repo, *args_for(scratch, "--limit", "3")) == 0
    gens = read_jsonl(gen.gens_path(scratch, "dev", "A1", None, "en"), Generation)
    assert [g.prompt_id for g in gens] == ["en-0", "en-1", "en-2", "en:subject/test/0", "en:subject/test/1",
                                            "en:subject/test/2"]


def test_a_malformed_task_file_is_a_runtime_error_without_text(repo, scratch, capsys):
    build_testset(repo, n=2)
    path = repo / "data" / "ft" / "taskdev.jsonl"
    record = task_record("en", 0)
    record["messages"] = record["messages"][:1]  # no assistant turn
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "fake") == 1
    err = capsys.readouterr().err
    assert "taskdev.jsonl" in err and "line 1" in err and TASK_MARK not in err


def test_a_task_id_that_collides_with_a_safety_prompt_id_is_refused(repo, scratch, capsys):
    build_testset(repo, n=2)
    record = task_record("en", 0)
    record["id"] = "en-0"
    (repo / "data" / "ft" / "taskdev.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "fake") == 1
    assert "en-0" in capsys.readouterr().err and not scratch.exists()


def test_a_task_row_in_the_safety_test_set_is_refused(repo, scratch, capsys):
    write_jsonl(repo / "data" / "testset" / "en" / "dev.jsonl",
                [Prompt("en-0", "en", "task", f"{MARK} x", "dev", "fixture", "fixture", "B")])
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "fake") == 1
    assert "task" in capsys.readouterr().err and not scratch.exists()


# --- judging ---------------------------------------------------------------------------------------------------


def test_verdicts_follow_gens_order_one_per_safety_item_and_no_response_skips_the_judge(repo, scratch, monkeypatch,
                                                                                         judge):
    build_testset(repo, n=6)
    use_model(monkeypatch, text_for=lambda p: "  \n" if p.endswith((" 1", " 4")) else default_text(p))
    assert run_cli(repo, *args_for(scratch)) == 0
    verdicts = read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)
    assert [v.prompt_id for v in verdicts] == [f"en-{i}" for i in range(6)]
    assert [v.label for v in verdicts] == ["complied", "no_response", "complied", "refused", "no_response", "refused"]
    assert len(judge.calls) == 4  # the two empty completions never reached the judge
    assert all(v.lang == "en" and v.arm == "A1" and v.alpha is None for v in verdicts)


def test_verdicts_are_rederived_from_gens_and_cache_without_asking_the_judge_again(repo, scratch, model, judge):
    build_testset(repo, n=6)
    assert run_cli(repo, *args_for(scratch)) == 0
    path = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    original = path.read_bytes()
    calls = len(judge.calls)
    path.unlink()  # lost, e.g. a run killed between the cache writes and the verdicts file
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 0
    assert path.read_bytes() == original and len(judge.calls) == calls
    assert [v.prompt_id for v in read_jsonl(path, Verdict)] == [f"en-{i}" for i in range(6)]


def test_rejudge_with_the_same_judge_just_rewrites_the_file(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch)) == 0
    before = snapshot(scratch)
    assert run_cli(repo, *args_for(scratch, "--rejudge")) == 0
    assert snapshot(scratch) == before and not (scratch / "dev" / "_replaced").exists()


# --- mixing judges ---------------------------------------------------------------------------------------------


def test_refuses_to_mix_judges(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    x = register(monkeypatch, "x", StubJudge("X"))
    y = register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, judge_name="x")) == 0
    path = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    before = snapshot(scratch)
    assert {v.judge_id for v in read_jsonl(path, Verdict)} == {"X"}

    with pytest.raises(JudgeMixError):
        evaluate(repo, *args_for(scratch, judge_name="y"))
    assert snapshot(scratch) == before and y.calls == []  # refused before anything was judged or written

    assert evaluate(repo, *args_for(scratch, "--rejudge", judge_name="y"))
    assert {v.judge_id for v in read_jsonl(path, Verdict)} == {"Y"}
    assert len(y.calls) == 4 and len(x.calls) == 4


def test_a_sibling_arms_verdicts_by_another_judge_are_refused_before_any_spend(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    y = register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, arm="A0", judge_name="x")) == 0
    before, built = snapshot(scratch), model.built
    capsys.readouterr()
    assert run_cli(repo, *args_for(scratch, arm="A1", judge_name="y")) == evalrun.EXIT_JUDGE_MIX
    err = capsys.readouterr().err
    assert "A0" in err and "en.verdicts.jsonl" in err and "'X'" in err and "'Y'" in err and "--rejudge" in err
    assert y.calls == [] and snapshot(scratch) == before and model.built == built  # no judge call, nothing written
    with pytest.raises(JudgeMixError):
        evaluate(repo, *args_for(scratch, arm="A1", judge_name="y"))


def test_the_sibling_check_with_rejudge_warns_and_proceeds(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    y = register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, arm="A0", judge_name="x")) == 0
    a0_bytes = evalrun.verdicts_path(scratch, "dev", "A0", None, "en").read_bytes()
    capsys.readouterr()
    assert run_cli(repo, *args_for(scratch, "--rejudge", arm="A1", judge_name="y")) == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "A0" in err and "'X'" in err and "'Y'" in err
    assert {v.judge_id for v in read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)} == {"Y"}
    assert len(y.calls) == 4
    assert evalrun.verdicts_path(scratch, "dev", "A0", None, "en").read_bytes() == a0_bytes  # the sibling is not touched
    assert not (scratch / "dev" / "_replaced").exists()


def test_the_sibling_check_covers_other_alphas_and_languages(repo, scratch, monkeypatch, model):
    build_testset(repo, ("en", "hi"), n=4)
    register(monkeypatch, "x", StubJudge("X"))
    y = register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, "--alpha", "0.25", "--langs", "hi", arm="A2", judge_name="x")) == 0
    code = run_cli(repo, *args_for(scratch, "--alpha", "0.5", "--langs", "en", arm="A2", judge_name="y"))
    assert code == evalrun.EXIT_JUDGE_MIX and y.calls == []  # another alpha, another language, still one table


def test_a_sibling_file_with_two_judges_is_refused_and_named(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    sibling = evalrun.verdicts_path(scratch, "dev", "A0", None, "en")
    write_jsonl(sibling, [Verdict("en-0", "A0", None, "en", "harmful", "X", "complied", ""),
                          Verdict("en-1", "A0", None, "en", "benign_edgy", "Z", "refused", "")])
    assert run_cli(repo, *args_for(scratch, arm="A1", judge_name="x")) == evalrun.EXIT_JUDGE_MIX
    assert "'Z'" in capsys.readouterr().err


def test_an_unreadable_sibling_file_is_a_runtime_error_and_is_left_alone(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)
    sibling = evalrun.verdicts_path(scratch, "dev", "A0", None, "en")
    sibling.parent.mkdir(parents=True)
    sibling.write_text("not json\n", encoding="utf-8")
    assert run_cli(repo, *args_for(scratch, arm="A1")) == evalrun.EXIT_ERROR
    assert sibling.read_text(encoding="utf-8") == "not json\n" and judge.calls == []
    assert "A0" in capsys.readouterr().err


def test_the_sibling_scan_ignores_cache_replaced_and_reliability_files(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    y = register(monkeypatch, "y", StubJudge("Y"))
    other = [Verdict("en-0", "A0", None, "en", "harmful", "Z", "complied", "")]
    write_jsonl(evalrun.verdicts_path(scratch, "dev", "A0", None, "en", reliability_judge="Z"), other)
    write_jsonl(evalrun.replaced_verdicts_path(scratch, "dev", "Z", "A0", None, "en"), other)
    for stray in ("reliability", "_replaced", "_cache"):  # files directly under these folders are not results either
        write_jsonl(scratch / "dev" / stray / "z" / "en.verdicts.jsonl", other)
    cache = evalrun.cache_path(scratch, "dev", "Z")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text('{"key": "k", "label": "complied"}\n', encoding="utf-8")
    assert run_cli(repo, *args_for(scratch, judge_name="y")) == 0 and len(y.calls) == 4


def test_the_sibling_scan_ignores_other_splits(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    y = register(monkeypatch, "y", StubJudge("Y"))
    write_jsonl(evalrun.verdicts_path(scratch, "report", "A0", None, "en"),
                [Verdict("en-0", "A0", None, "en", "harmful", "X", "complied", "")])
    assert run_cli(repo, *args_for(scratch, judge_name="y")) == 0 and len(y.calls) == 4


def test_runs_without_the_judge_stage_do_not_scan_siblings(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    write_jsonl(evalrun.verdicts_path(scratch, "dev", "A0", None, "en"),
                [Verdict("en-0", "A0", None, "en", "harmful", "X", "complied", "")])
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "hf", "--model", "m",
                   "--stages", "generate,task") == 0


def test_the_mix_is_refused_before_any_generation_happens(repo, scratch, monkeypatch):
    build_testset(repo, n=4)
    fake = use_model(monkeypatch)
    register(monkeypatch, "y", StubJudge("Y"))
    planted = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    write_jsonl(planted, [Verdict("en-0", "A1", None, "en", "harmful", "X", "complied", "")])
    with pytest.raises(JudgeMixError):
        evaluate(repo, *args_for(scratch, judge_name="y"))
    assert fake.built == 0 and not gen.gens_path(scratch, "dev", "A1", None, "en").exists()


def test_the_mix_error_has_its_own_exit_code(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, judge_name="x")) == 0
    code = run_cli(repo, *args_for(scratch, judge_name="y"))
    assert code == evalrun.EXIT_JUDGE_MIX and code not in (0, 1, 2, 3, evalrun.EXIT_QUALITY)


def test_the_same_judge_may_rerun_without_rejudge(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch)) == 0
    assert run_cli(repo, *args_for(scratch)) == 0


def test_rejudge_moves_the_old_file_it_never_deletes_it(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, judge_name="x")) == 0
    path = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    x_bytes = path.read_bytes()
    assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="y")) == 0
    moved = evalrun.replaced_verdicts_path(scratch, "dev", "X", "A1", None, "en")
    assert moved.parents[3].name == "_replaced" and moved.parents[2].name == evalrun.judge_dirname("X")
    assert moved.read_bytes() == x_bytes
    assert path.read_bytes() != x_bytes and {v.judge_id for v in read_jsonl(path, Verdict)} == {"Y"}
    y_bytes = path.read_bytes()

    assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="x")) == 0  # back to X: Y is moved aside too
    assert evalrun.replaced_verdicts_path(scratch, "dev", "Y", "A1", None, "en").read_bytes() == y_bytes
    assert moved.read_bytes() == x_bytes  # the first replaced file is untouched
    x_again = path.read_bytes()
    assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="y")) == 0  # X replaced a second time
    siblings = sorted(p.name for p in moved.parent.iterdir())
    assert len(siblings) == 2 and moved.name in siblings  # a numbered copy, never an overwrite
    assert sorted(p.read_bytes() for p in moved.parent.iterdir()) == sorted([x_bytes, x_again])


def test_a_verdicts_file_with_two_judges_is_a_mix_error_even_for_one_of_them(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    path = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    write_jsonl(path, [Verdict("en-0", "A1", None, "en", "harmful", "X", "complied", ""),
                       Verdict("en-1", "A1", None, "en", "benign_edgy", "Y", "refused", "")])
    planted = path.read_bytes()
    with pytest.raises(JudgeMixError) as caught:
        evaluate(repo, *args_for(scratch, judge_name="x"))
    assert "X" in str(caught.value) and "Y" in str(caught.value)
    assert path.read_bytes() == planted
    # --rejudge replaces it; the two-judge file is kept under _replaced
    assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="x")) == 0
    assert {v.judge_id for v in read_jsonl(path, Verdict)} == {"X"}
    kept = [p for p in (scratch / "dev" / "_replaced").rglob("en.verdicts.jsonl")]
    assert len(kept) == 1 and kept[0].read_bytes() == planted


def test_an_unreadable_verdicts_file_is_an_error_not_a_silent_overwrite(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)
    path = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    path.parent.mkdir(parents=True)
    path.write_text("this is not json\n", encoding="utf-8")
    assert run_cli(repo, *args_for(scratch)) == 1
    assert path.read_text(encoding="utf-8") == "this is not json\n"
    assert "en.verdicts.jsonl" in capsys.readouterr().err


# --- quality gates ---------------------------------------------------------------------------------------------


def two_gens_failing(n_fail):
    """A judge that cannot answer for the first `n_fail` prompts of the fixture file (indexes 0..n_fail-1)."""
    return lambda kind, prompt, response: any(prompt.endswith(f" {i}") for i in range(n_fail))


def test_judge_error_max_is_ten_percent():
    assert evalrun.JUDGE_ERROR_MAX == 0.10


def test_judge_error_rate_over_the_threshold_exits_nonzero_after_writing_the_verdicts(repo, scratch, monkeypatch,
                                                                                      model, capsys):
    build_testset(repo, n=10)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=two_gens_failing(2)))  # 2 of 10 = 20%
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    verdicts = read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)
    assert len(verdicts) == 10 and [v.label for v in verdicts].count("judge_error") == 2
    report = eval_json(scratch)
    assert report["judge_error"]["rate"] == pytest.approx(0.2) and report["judge_error"]["n_judged"] == 10
    assert report["judge_error"]["by_kind"]["harmful"] == {"n_judged": 5, "judge_error": 1, "rate": 0.2}
    assert report["judge_error"]["by_kind"]["benign_edgy"]["judge_error"] == 1
    assert report["verdicts"]["labels"]["judge_error"] == 2
    err = capsys.readouterr().err
    assert "judge_error" in err and "en" in err and "10" in err  # said so, with the limit


def test_judge_error_rate_at_the_threshold_passes(repo, scratch, monkeypatch, model):
    build_kinds(repo, harmful=10, benign=10)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=fails_first(n_harmful=1, n_benign=1)))  # 2 of 20 = 10%, each kind 10%
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["judge_error"]["rate"] == pytest.approx(0.1)
    assert report["judge_error"]["by_kind"]["harmful"]["rate"] == pytest.approx(0.1)


def test_every_language_is_judged_and_written_before_the_gate_fails_the_run(repo, scratch, monkeypatch, model):
    build_testset(repo, ("en", "hi"), n=10)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=lambda kind, prompt, response: " en " in prompt))
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    assert eval_json(scratch, lang="en")["judge_error"]["rate"] == 1.0
    assert eval_json(scratch, lang="hi")["judge_error"]["rate"] == 0.0
    assert evalrun.verdicts_path(scratch, "dev", "A1", None, "hi").is_file()


def test_judge_errors_are_not_cached_so_a_rerun_with_a_fixed_judge_recovers(repo, scratch, monkeypatch, model):
    build_testset(repo, n=10)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=two_gens_failing(3)))
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    fixed = register(monkeypatch, "j1", StubJudge("J1"))
    assert run_cli(repo, *args_for(scratch)) == 0
    assert len(fixed.calls) == 3  # only the three that failed are asked again
    verdicts = read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)
    assert "judge_error" not in {v.label for v in verdicts}


def test_no_response_does_not_dilute_the_judge_error_rate(repo, scratch, monkeypatch):
    build_testset(repo, n=10)
    # five completions are empty (never judged); the judge fails on one of the five it is asked about = 20%
    use_model(monkeypatch, text_for=lambda p: "" if p.endswith((" 1", " 3", " 5", " 7", " 9")) else default_text(p))
    register(monkeypatch, "j1", StubJudge("J1", fail_if=two_gens_failing(1)))
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    report = eval_json(scratch)
    assert report["judge_error"]["n_judged"] == 5 and report["judge_error"]["rate"] == pytest.approx(0.2)
    assert report["verdicts"]["labels"]["no_response"] == 5


def test_a_kind_exactly_at_the_limit_inside_a_clean_language_rate_passes(repo, scratch, monkeypatch, model):
    build_testset(repo, n=20)  # 10 harmful, 10 benign_edgy; only one benign item fails: language 5%, kind 10%
    register(monkeypatch, "j1", StubJudge("J1", fail_if=lambda kind, prompt, response: prompt.endswith(" en 1")))
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["judge_error"]["rate"] == pytest.approx(0.05)
    assert report["judge_error"]["by_kind"]["benign_edgy"]["rate"] == pytest.approx(0.1)
    assert report["warnings"] == []


def test_a_kind_clearly_over_the_limit_fails_the_run_even_when_the_language_rate_is_fine(
        repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=40)  # 20 + 20; 3 benign failures: language 7.5%, benign kind 15%
    register(monkeypatch, "j1", StubJudge("J1", fail_if=lambda kind, prompt, response: prompt.endswith((" 1", " 3", " 5"))))
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    report = eval_json(scratch)
    assert report["judge_error"]["rate"] == pytest.approx(0.075)
    assert report["judge_error"]["by_kind"]["benign_edgy"] == {"n_judged": 20, "judge_error": 3, "rate": 0.15}
    assert any("benign_edgy" in w and "judge_error" in w for w in report["warnings"])
    err = capsys.readouterr().err
    assert "WARN" in err and "benign_edgy" in err and "15.0%" in err
    assert len(read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)) == 40  # written


def build_kinds(root, harmful, benign, split="dev", lang="en"):
    write_jsonl(
        Path(root) / "data" / "testset" / lang / f"{split}.jsonl",
        [Prompt(f"{lang}-h{i}", lang, "harmful", f"{MARK} {lang} h{i}", split, "fixture", "fixture")
         for i in range(harmful)]
        + [Prompt(f"{lang}-b{i}", lang, "benign_edgy", f"{MARK} {lang} b{i}", split, "fixture", "fixture")
           for i in range(benign)],
    )


def fails_first(n_harmful=0, n_benign=0):
    def fail_if(kind, prompt, response):
        letter, index = prompt.split()[-1][0], int(prompt.split()[-1][1:])
        return (letter == "h" and index < n_harmful) or (letter == "b" and index < n_benign)

    return fail_if


def test_one_kind_over_the_limit_exits_5_though_the_overall_rate_is_not_over(repo, scratch, monkeypatch, model):
    build_kinds(repo, harmful=150, benign=100)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=fails_first(n_harmful=25)))  # 25/250 = 10% overall
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    report = eval_json(scratch)
    assert report["judge_error"]["rate"] == pytest.approx(0.1) and report["judge_error"]["n_judged"] == 250
    assert report["judge_error"]["by_kind"]["harmful"]["judge_error"] == 25
    assert report["judge_error"]["by_kind"]["harmful"]["rate"] == pytest.approx(25 / 150)
    assert report["judge_error"]["by_kind"]["benign_edgy"] == {"n_judged": 100, "judge_error": 0, "rate": 0.0}
    assert len(read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)) == 250


def test_the_other_kind_alone_over_the_limit_also_exits_5(repo, scratch, monkeypatch, model):
    build_kinds(repo, harmful=150, benign=100)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=fails_first(n_benign=11)))  # 11% of benign, 4.4% overall
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    assert eval_json(scratch)["judge_error"]["rate"] < evalrun.JUDGE_ERROR_MAX


def test_both_kinds_under_the_limit_exit_0(repo, scratch, monkeypatch, model):
    build_kinds(repo, harmful=150, benign=100)
    register(monkeypatch, "j1", StubJudge("J1", fail_if=fails_first(n_harmful=14, n_benign=9)))  # 9.3% and 9%
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["warnings"] == [] and report["judge_error"]["by_kind"]["harmful"]["judge_error"] == 14


def test_a_kind_failing_gate_still_processes_every_language(repo, scratch, monkeypatch, model):
    build_kinds(repo, harmful=20, benign=20, lang="en")
    build_kinds(repo, harmful=20, benign=20, lang="hi")
    register(monkeypatch, "j1", StubJudge("J1", fail_if=lambda kind, prompt, response: (
        " en " in prompt and kind == "benign_edgy" and int(prompt.split()[-1][1:]) < 3)))  # 15% of benign, 7.5% overall
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_QUALITY
    assert evalrun.verdicts_path(scratch, "dev", "A1", None, "hi").is_file()
    assert eval_json(scratch, lang="hi")["judge_error"]["by_kind"]["benign_edgy"]["rate"] == 0.0


def test_truncation_rate_is_reported_and_exactly_five_percent_does_not_warn(repo, scratch, monkeypatch, judge, capsys):
    build_testset(repo, n=10)
    build_tasks(repo, "taskdev.jsonl", n=10)  # 20 generations in all
    use_model(monkeypatch, finish_for=lambda p: "length" if TASK_MARK in p and " question 0" in p else "stop")
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["truncation"] == {"rate": 0.05, "n_length": 1, "n_total": 20}
    assert report["truncation_safety"] == {"rate": 0.0, "n_length": 0, "n_total": 10}
    assert report["finish_reasons"] == {"stop": 19, "length": 1, "error": 0}
    assert report["warnings"] == [] and "WARN" not in capsys.readouterr().err


def test_safety_only_truncation_is_recorded_next_to_the_overall_rate(repo, scratch, monkeypatch, judge):
    build_testset(repo, n=10)
    build_tasks(repo, "taskdev.jsonl", n=10)
    use_model(monkeypatch, finish_for=lambda p: "length" if MARK in p and p.endswith((" 0", " 1", " 2")) else "stop")
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["truncation"] == {"rate": 0.15, "n_length": 3, "n_total": 20}  # overall: unchanged definition
    assert report["truncation_safety"] == {"rate": 0.3, "n_length": 3, "n_total": 10}  # harmful + benign_edgy only


def test_safety_truncation_over_five_percent_warns_even_when_the_overall_rate_is_low(
        repo, scratch, monkeypatch, judge, capsys):
    build_testset(repo, n=10)
    build_tasks(repo, "taskdev.jsonl", n=90)  # 100 generations; one safety item truncated: overall 1%, safety 10%
    use_model(monkeypatch, finish_for=lambda p: "length" if MARK in p and p.endswith(" en 0") else "stop")
    assert run_cli(repo, *args_for(scratch)) == 0  # a warning, not a failure
    report = eval_json(scratch)
    assert report["truncation"]["rate"] == 0.01 and report["truncation_safety"]["rate"] == 0.1
    assert len(report["warnings"]) == 1 and "safety" in report["warnings"][0] and "10.0%" in report["warnings"][0]
    err = capsys.readouterr().err
    assert "WARN" in err and "safety" in err and "truncation" in err


def test_safety_truncation_of_exactly_five_percent_does_not_warn(repo, scratch, monkeypatch, judge, capsys):
    build_testset(repo, n=20)  # no task items: overall and safety-only rates are both 1 of 20
    use_model(monkeypatch, finish_for=lambda p: "length" if p.endswith(" en 0") else "stop")
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["truncation_safety"] == {"rate": 0.05, "n_length": 1, "n_total": 20}
    assert report["warnings"] == [] and "WARN" not in capsys.readouterr().err


def test_warns_when_truncation_rate_over_5_percent(repo, scratch, monkeypatch, judge, capsys):
    build_testset(repo, n=10)
    build_tasks(repo, "taskdev.jsonl", n=10)  # 20 generations in all
    use_model(monkeypatch, finish_for=lambda p: "length" if MARK in p and p.endswith((" 0", " 1")) else "stop")
    assert run_cli(repo, *args_for(scratch)) == 0  # a warning, not a failure
    err = capsys.readouterr().err
    assert "WARN" in err and "truncation" in err and "10.0%" in err and "5%" in err
    report = eval_json(scratch)
    assert report["truncation"] == {"rate": 0.1, "n_length": 2, "n_total": 20}
    assert report["finish_reasons"] == {"stop": 18, "length": 2, "error": 0}
    overall = [w for w in report["warnings"] if w.startswith("truncation rate")]
    assert len(overall) == 1 and "10.0%" in overall[0]  # the overall warning, unchanged
    assert any(w.startswith("safety-item truncation rate") and "20.0%" in w for w in report["warnings"])
    assert evalrun.TRUNCATION_WARN == 0.05


def test_truncation_is_counted_over_all_generations_including_task_items(repo, scratch, monkeypatch, judge):
    build_testset(repo, n=4)
    build_tasks(repo, "taskdev.jsonl", n=4)
    use_model(monkeypatch, finish_for=lambda p: "length" if TASK_MARK in p else "stop")
    assert run_cli(repo, *args_for(scratch)) == 0
    assert eval_json(scratch)["truncation"]["rate"] == 0.5


def test_empty_outputs_are_counted_separately_for_task_and_safety_items(repo, scratch, monkeypatch, judge):
    build_testset(repo, n=10)
    build_tasks(repo, "taskdev.jsonl", n=6)
    use_model(monkeypatch, text_for=lambda p: (
        "" if (MARK in p and p.endswith((" 0", " 1", " 2"))) or (TASK_MARK in p and " question 0" in p) else
        ("   " if TASK_MARK in p and " question 1" in p else default_text(p))))
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["empty_outputs"] == {"safety": 3, "task": 2}
    scores = read_jsonl(evalrun.tasks_path(scratch, "dev", "A1", None, "en"), TaskScore)
    assert [s.correct for s in scores][:2] == [False, False] and len(scores) == 6  # empty counts as wrong, inside n
    assert report["tasks"]["n"] == 6
    assert report["verdicts"]["labels"]["no_response"] == 3 and len(judge.calls) == 7


def test_empty_or_error_generations_over_five_percent_warn_and_are_recorded(repo, scratch, monkeypatch, judge, capsys):
    build_testset(repo, n=6)
    build_tasks(repo, "taskdev.jsonl", n=4)
    use_model(monkeypatch, text_for=lambda p: "", finish_for=lambda p: "error")
    assert run_cli(repo, *args_for(scratch)) == 0  # a warning only
    report = eval_json(scratch)
    assert report["empty_or_error"] == {"n": 10, "n_total": 10, "rate": 1.0}
    assert report["finish_reasons"] == {"stop": 0, "length": 0, "error": 10}
    assert any(w.startswith("100.0% of generations are empty or ended with finish_reason=error") for w in
               report["warnings"])
    err = capsys.readouterr().err
    assert "WARN" in err and "empty" in err and "finish_reason=error" in err and "100.0%" in err


def test_empty_or_error_counts_each_generation_once_and_five_percent_does_not_warn(repo, scratch, monkeypatch, judge,
                                                                                  capsys):
    build_testset(repo, n=20)  # one empty + stop, one non-empty + error, one empty + error: 3 of 60 would be 5%
    build_tasks(repo, "taskdev.jsonl", n=40)
    use_model(
        monkeypatch,
        text_for=lambda p: "" if MARK in p and p.endswith((" en 0", " en 2")) else default_text(p),
        finish_for=lambda p: "error" if MARK in p and p.endswith((" en 1", " en 2")) else "stop",
    )
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["empty_or_error"] == {"n": 3, "n_total": 60, "rate": 0.05}  # en 2 is empty AND error: counted once
    assert not any("empty" in w for w in report["warnings"]) and "WARN" not in capsys.readouterr().err


def test_a_few_empty_or_error_generations_over_five_percent_warn(repo, scratch, monkeypatch, judge):
    build_testset(repo, n=20)
    use_model(monkeypatch, text_for=lambda p: "" if p.endswith((" en 0", " en 1")) else default_text(p))
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert report["empty_or_error"] == {"n": 2, "n_total": 20, "rate": 0.1}
    assert any("empty or ended with finish_reason=error" in w and "10.0%" in w for w in report["warnings"])


def test_eval_json_reports_the_setup_counts_and_labels(repo, scratch, model, judge):
    build_testset(repo, n=6)
    build_tasks(repo, "taskdev.jsonl", n=4)
    assert run_cli(repo, *args_for(scratch, "--alpha", "0.5", arm="A2")) == 0
    report = eval_json(scratch, arm="A2", alpha=0.5)
    assert report["arm"] == "A2" and report["alpha"] == 0.5 and report["lang"] == "en" and report["split"] == "dev"
    assert report["model_path"] == "org/model" and report["backend"] == "hf" and report["judge_id"] == "J1"
    assert report["n_generations"] == 10
    assert report["n_by_kind"] == {"harmful": 3, "benign_edgy": 3, "task": 4}
    assert report["verdicts"]["n"] == 6
    assert report["verdicts"]["labels"] == {"complied": 3, "refused": 3, "no_response": 0, "judge_error": 0}
    assert report["verdicts"]["by_kind"]["harmful"] == {"complied": 3, "refused": 0, "no_response": 0, "judge_error": 0}
    assert report["verdicts"]["by_kind"]["benign_edgy"]["refused"] == 3
    assert report["tasks"] == {"n": 4, "correct": 2, "accuracy": 0.5}
    assert report["task_skipped"] is None and report["warnings"] == []


def test_eval_json_has_no_text_and_no_secrets(repo, scratch, model, judge, monkeypatch, capsys):
    build_testset(repo, n=6)
    build_tasks(repo, "taskdev.jsonl", n=4)
    token = "hf_" + "A" * 30
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRETSECRET1234567890")
    monkeypatch.setenv("HF_TOKEN", token)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate,judge,task")) == 0
    text = evalrun.eval_path(scratch, "dev", "A1", None, "en").read_text(encoding="utf-8")
    for forbidden in (MARK, TASK_MARK, ANSWER, "sk-ant-SECRETSECRET1234567890", token):
        assert forbidden not in text
    shown = capsys.readouterr()
    for forbidden in (MARK, TASK_MARK, ANSWER, "sk-ant-SECRETSECRET1234567890", token):
        assert forbidden not in shown.out + shown.err
    report = json.loads(text)
    assert set(report) == {
        "arm", "alpha", "lang", "split", "model_path", "backend", "judge_id", "judge", "n_generations", "n_by_kind",
        "finish_reasons", "truncation", "truncation_safety", "empty_outputs", "empty_or_error", "verdicts",
        "judge_error", "tasks", "task_skipped", "warnings",  # "judge" is the one key added with the judge configuration
    }
    assert "timestamp" not in text  # no clock: a rerun must be byte-identical


def test_a_token_in_the_model_path_does_not_reach_eval_json(repo, scratch, model, judge):
    build_testset(repo, n=4)
    token = "hf_" + "B" * 30
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1", "--backend", "hf",
                   "--model", f"org/{token}") == 0
    assert token not in evalrun.eval_path(scratch, "dev", "A1", None, "en").read_text(encoding="utf-8")


# --- safety of results -----------------------------------------------------------------------------------------


def test_fake_judge_is_registered_and_deterministic():
    assert "fake" in evalrun.JUDGE_REGISTRY
    first, second = evalrun.JUDGE_REGISTRY["fake"](), evalrun.JUDGE_REGISTRY["fake"]()
    assert first.id == second.id and "fake" in first.id
    answers = {first.classify("harmful", "p", f"r{i}") for i in range(40)}
    assert answers == {"complied", "refused"}
    assert [first.classify("harmful", "p", f"r{i}") for i in range(10)] == [
        second.classify("harmful", "p", f"r{i}") for i in range(10)]


def test_fake_run_end_to_end_in_a_scratch_root(repo, scratch, capsys):
    build_testset(repo, n=6)
    build_tasks(repo, "taskdev.jsonl", n=3)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "fake") == 0
    verdicts = read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)
    assert len(verdicts) == 6 and {v.judge_id for v in verdicts} == {evalrun.JUDGE_REGISTRY["fake"]().id}
    assert {v.label for v in verdicts} <= {"complied", "refused"}


@pytest.mark.parametrize("spelling", ["results", "Results", "results/sub"])
def test_fake_judge_is_refused_on_the_real_results_root(repo, spelling, capsys):
    build_testset(repo, n=4)
    out = repo / spelling
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(out), "--backend", "hf", "--model", "m",
                   "--judge", "fake", "--stages", "judge") == 2
    assert "fake" in capsys.readouterr().err and not out.exists()


def test_fake_judge_is_refused_on_the_default_out_root(repo):
    build_testset(repo, n=4)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "fake") == 2
    assert not (repo / "results").exists()


def test_fake_backend_is_refused_on_the_real_results_root(repo, capsys):
    build_testset(repo, n=4)
    # the generate stage alone with the fake backend is refused by the same guard as scripts/generate.py
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(repo / "results"), "--backend", "fake",
                   "--stages", "generate") == 2
    assert not (repo / "results").exists()


def test_a_judge_that_declares_itself_fake_is_refused_on_the_real_root_whatever_its_name(repo, monkeypatch):
    build_testset(repo, n=4)

    class Sneaky(StubJudge):
        is_fake = True

    register(monkeypatch, "sneaky", Sneaky("S"))
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(repo / "results"), "--backend", "hf",
                   "--model", "m", "--judge", "sneaky", "--stages", "judge") == 2
    assert not (repo / "results").exists()


def test_a_fake_meta_is_refused_on_the_real_results_root(repo, scratch, judge, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--stages", "generate") == 0
    shutil.copytree(scratch, repo / "results")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(repo / "results"), "--judge", "j1",
                   "--stages", "judge") == 1
    err = capsys.readouterr().err
    assert "fake" in err
    assert judge.calls == [] and not evalrun.verdicts_path(repo / "results", "dev", "A1", None, "en").exists()
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(repo / "results"), "--stages", "task") == 1
    # the same fake gens are fine in a scratch root
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 0


def test_a_meta_that_says_fake_is_refused_on_the_real_root_even_with_ordinary_text(repo, scratch, model, judge,
                                                                                 capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0  # scripted model: no fake marker anywhere
    gens_file = gen.gens_path(scratch, "dev", "A1", None, "en")
    assert gen.FAKE_MARKER not in gens_file.read_text(encoding="utf-8")
    shutil.copytree(scratch, repo / "results")
    meta_file = gen.meta_path(gen.gens_path(repo / "results", "dev", "A1", None, "en"))
    results_args = ["--arm", "A1", "--split", "dev", "--out", str(repo / "results"), "--judge", "j1"]
    for spelling in ("fake", " FAKE ", "Fake"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["backend"] = spelling
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        for stages in ("judge", "task"):
            assert run_cli(repo, *results_args, "--stages", stages) == 1
            err = capsys.readouterr().err
            assert "was made by the fake backend" in err and "real results" in err and "Traceback" not in err
    assert judge.calls == [] and not evalrun.verdicts_path(repo / "results", "dev", "A1", None, "en").exists()
    # the same meta in a scratch root is fine: only the real results root is protected
    scratch_meta = gen.meta_path(gens_file)
    meta = json.loads(scratch_meta.read_text(encoding="utf-8"))
    meta["backend"] = "fake"
    scratch_meta.write_text(json.dumps(meta), encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 0


def test_fake_text_under_a_meta_that_does_not_say_fake_is_refused_on_the_real_root(repo, scratch, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--stages", "generate") == 0
    shutil.copytree(scratch, repo / "results")
    meta_file = gen.meta_path(gen.gens_path(repo / "results", "dev", "A1", None, "en"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["backend"] = "hf"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(repo / "results"), "--judge", "j1",
                   "--stages", "judge") == 1
    assert judge.calls == []


def test_a_gens_file_with_rows_but_no_meta_is_refused(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A1", None, "en"))
    meta_file.unlink()
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1
    err = capsys.readouterr().err
    assert len([line for line in err.splitlines() if line.startswith("error: ")]) == 1  # one clean refusal ...
    assert "Traceback" not in err  # ... not a crash
    assert "holds 4 row(s) but its meta file" in err and "en.gens.meta.json" in err
    assert "is missing or unreadable, so what made them is unknown" in err
    assert judge.calls == [] and not evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists()
    meta_file.write_text("{ not json", encoding="utf-8")  # unreadable is refused the same way
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1
    assert "is missing or unreadable" in capsys.readouterr().err


def test_a_meta_for_another_setup_than_the_path_is_refused(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A1", None, "en"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["arm"] = "A0"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1


def test_the_row_count_comes_from_the_gens_file_not_the_meta(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    meta_file = gen.meta_path(gen.gens_path(scratch, "dev", "A1", None, "en"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["n_rows"] = 9999  # stale after a hard kill
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 0
    assert eval_json(scratch)["n_generations"] == 4


def rebuild_with_ids(repo, ids, split="dev"):
    write_jsonl(repo / "data" / "testset" / "en" / f"{split}.jsonl",
                [Prompt(i, "en", "harmful" if k % 2 == 0 else "benign_edgy", f"{MARK} {i}", split, "fixture",
                        "fixture") for k, i in enumerate(ids)])


def test_gens_with_ids_not_in_the_current_prompt_list_are_refused(repo, scratch, model, judge, capsys):
    build_testset(repo, n=6)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    rebuild_with_ids(repo, ["en-0", "en-1", "en-2", "new-3", "new-4", "new-5"])  # another test-set build
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1
    err = capsys.readouterr().err
    assert "3 row(s)" in err and "not in" in err and "prompt" in err
    assert judge.calls == [] and not evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists()


def test_incomplete_gens_are_refused_for_judging(repo, scratch, model, judge, capsys):
    build_testset(repo, n=6)
    assert run_cli(repo, *args_for(scratch, "--limit", "3", "--stages", "generate")) == 0
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1
    err = capsys.readouterr().err
    assert "3 of 6" in err and judge.calls == []


def test_a_torn_last_line_is_refused_for_judging(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    path = gen.gens_path(scratch, "dev", "A1", None, "en")
    torn = b'{"prompt_id": "en-3", "arm"'
    path.write_bytes(path.read_bytes() + torn)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1
    err = capsys.readouterr().err
    assert len([line for line in err.splitlines() if line.startswith("error: ")]) == 1
    assert "Traceback" not in err and "JSONDecodeError" not in err
    assert "ends with a torn line" in err and f"{len(torn)} bytes" in err and "generate stage" in err
    assert judge.calls == [] and not evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists()


def test_a_testset_row_of_the_other_split_is_an_error(repo, scratch, capsys):
    build_testset(repo, n=4)
    path = repo / "data" / "testset" / "en" / "dev.jsonl"
    rows = read_jsonl(path, Prompt)
    rows[2] = Prompt(rows[2].id, "en", rows[2].kind, rows[2].text, "report", "fixture", "fixture")
    write_jsonl(path, rows)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "fake") == 1
    err = capsys.readouterr().err
    assert "en-2" in err and "dev" in err and MARK not in err
    assert not scratch.exists()


def test_existing_gens_made_by_another_arm_are_refused(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    path = gen.gens_path(scratch, "dev", "A1", None, "en")
    path.write_text(path.read_text(encoding="utf-8").replace('"arm": "A1"', '"arm": "A0"'), encoding="utf-8")
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge") == 1


def test_a_changed_generation_setup_is_refused_unless_forced(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch, "--stages", "generate")) == 0
    assert run_cli(repo, *args_for(scratch, "--stages", "generate", "--batch-size", "2")) == 1
    assert "--force-new-meta" in capsys.readouterr().err
    assert run_cli(repo, *args_for(scratch, "--stages", "generate", "--batch-size", "2", "--force-new-meta")) == 0


# --- atomic writes and reruns ------------------------------------------------------------------------------------


def test_no_temp_file_is_left_behind_after_a_run(repo, scratch, model, judge):
    build_testset(repo, n=4)
    build_tasks(repo, "taskdev.jsonl", n=2)
    assert run_cli(repo, *args_for(scratch)) == 0
    assert not [name for name in files_under(scratch) if name.endswith(".tmp")]


def test_a_failed_verdict_write_keeps_the_previous_file_and_leaves_no_temp(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    register(monkeypatch, "x", StubJudge("X"))
    register(monkeypatch, "y", StubJudge("Y"))
    assert run_cli(repo, *args_for(scratch, judge_name="x")) == 0
    before = snapshot(scratch)

    def failing_fsync(fd):
        raise OSError("disk full (pretend)")

    with monkeypatch.context() as patched:
        patched.setattr(evalrun.os, "fsync", failing_fsync)
        assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="y")) == 1
    assert not [name for name in files_under(scratch) if name.endswith(".tmp")]
    path = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    assert path.read_bytes() == before["dev/A1/none/en.verdicts.jsonl"]  # X's file is still in place, whole
    assert not (scratch / "dev" / "_replaced").exists()


def test_a_rerun_leaves_every_file_byte_identical_and_asks_the_judge_nothing(repo, scratch, model, judge):
    build_testset(repo, ("en", "hi"), n=6)
    build_tasks(repo, "taskdev.jsonl", ("en", "hi"), n=4)
    assert run_cli(repo, *args_for(scratch)) == 0
    before = snapshot(scratch)
    calls = len(judge.calls)
    assert run_cli(repo, *args_for(scratch)) == 0
    assert snapshot(scratch) == before
    assert len(judge.calls) == calls
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--judge", "j1",
                   "--stages", "judge,task") == 0
    assert snapshot(scratch) == before


def test_a_judge_that_crashes_mid_language_leaves_no_verdicts_file_and_a_rerun_resumes_from_the_cache(
        repo, scratch, monkeypatch, model):
    build_testset(repo, n=6)

    class Dies(StubJudge):
        def classify(self, kind, prompt, response):
            if len(self.calls) == 3:
                raise KeyboardInterrupt  # not an Exception: JudgeRunner does not swallow it
            return super().classify(kind, prompt, response)

    register(monkeypatch, "j1", Dies("J1"))
    with pytest.raises(KeyboardInterrupt):
        run_cli(repo, *args_for(scratch))
    assert not evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists()
    survivor = register(monkeypatch, "j1", StubJudge("J1"))
    assert run_cli(repo, *args_for(scratch)) == 0
    assert len(survivor.calls) == 3  # the three judged before the crash come from the cache


# --- aggregation can consume the files -------------------------------------------------------------------------


def test_output_files_feed_the_aggregation(repo, scratch, model, judge):
    langs = ("en", "hi")
    build_testset(repo, langs, split="report", n=8)
    build_tasks(repo, "heldout.jsonl", langs, n=4)
    cells = [("A0", None), ("A1", None), ("A2", 0.5)]
    for arm, alpha in cells:
        extra = ["--alpha", str(alpha)] if alpha is not None else []
        assert run_cli(repo, *args_for(scratch, *extra, arm=arm, split="report")) == 0
    verdicts, scores = [], []
    for arm, alpha in cells:
        for lang in langs:
            verdicts += read_jsonl(evalrun.verdicts_path(scratch, "report", arm, alpha, lang), Verdict)
            scores += read_jsonl(evalrun.tasks_path(scratch, "report", arm, alpha, lang), TaskScore)
    rows = build_table(verdicts, scores)  # raises on mixed judges or duplicate keys
    assert {r["metric"] for r in rows} == {"asr", "over_refusal", "task_acc"}
    assert len(rows) == len(cells) * len(langs) * 3
    assert {(r["arm"], r["alpha"], r["lang"]) for r in rows} == {(a, al, lang) for a, al in cells for lang in langs}


# --- command line ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--arm", "A1"],
        ["--split", "dev"],
        ["--arm", "A9", "--split", "dev", "--judge", "j1"],
        ["--arm", "A1", "--split", "train", "--judge", "j1"],
        ["--arm", "A1", "--split", "dev", "--alpha", "0.5", "--judge", "j1", "--backend", "fake"],  # A1: no alpha
        ["--arm", "A3", "--split", "dev", "--alpha", "0.5", "--judge", "j1", "--backend", "fake"],  # A3: no alpha
        ["--arm", "BASE", "--split", "dev", "--alpha", "0", "--judge", "j1", "--backend", "fake"],
        ["--arm", "A0", "--split", "dev", "--alpha", "1", "--judge", "j1", "--backend", "fake"],
        ["--arm", "A2", "--split", "dev", "--judge", "j1", "--backend", "fake"],  # A2 needs alpha
        ["--arm", "A2n", "--split", "dev", "--judge", "j1", "--backend", "fake"],
        ["--arm", "A2", "--split", "dev", "--alpha", "none", "--judge", "j1", "--backend", "fake"],
        ["--arm", "A2", "--split", "dev", "--alpha", "abc", "--judge", "j1", "--backend", "fake"],
        ["--arm", "A2", "--split", "dev", "--alpha", "nan", "--judge", "j1", "--backend", "fake"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake"],  # the judge stage needs --judge
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "nope"],  # unknown judge
        ["--arm", "A1", "--split", "dev", "--backend", "hf", "--judge", "j1"],  # generation needs --model
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--stages", "grade"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--stages", ""],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--stages", "judge,,task"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--stages", "task", "--rejudge"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--langs", "en,fr"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--langs", ""],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--limit", "0"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--batch-size", "0"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--max-new-tokens", "0"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--dtype", "fp16"],
        ["--arm", "A1", "--split", "dev", "--backend", "gpu", "--judge", "j1"],
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "j1", "--surprise"],
    ],
)
def test_bad_arguments_exit_2_and_write_nothing(repo, scratch, judge, argv):
    build_testset(repo)
    if "--out" not in argv:
        argv = [*argv, "--out", str(scratch)]
    assert run_cli(repo, *argv) == 2
    assert not scratch.exists()


def test_unknown_judge_lists_the_registered_names(repo, scratch, judge, capsys):
    build_testset(repo)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "nope") == 2
    err = capsys.readouterr().err
    assert "nope" in err and "fake" in err and "j1" in err


def test_help_exits_0(repo, capsys):
    assert run_cli(repo, "--help") == 0
    out = capsys.readouterr().out
    assert "--stages" in out and "--rejudge" in out and "--judge" in out and "--ft-dir" in out


def test_defaults(repo):
    args = evalrun.parse_args(["--arm", "A1", "--split", "dev", "--judge", "fake", "--backend", "fake",
                               "--out", "/tmp/somewhere"], root=repo)
    assert args.stages == ["generate", "judge", "task"] and args.langs == list(evalrun.LANGS)
    assert args.alpha is None and args.rejudge is False and args.limit is None
    assert args.testset_dir == repo / "data" / "testset" and args.ft_dir == repo / "data" / "ft"
    assert args.batch_size == gen.DEFAULT_BATCH_SIZE and args.max_new_tokens == 256 and args.dtype == "float16"
    default_out = evalrun.parse_args(["--arm", "A1", "--split", "dev", "--stages", "task"], root=repo).out_root
    assert default_out == repo / "results"


def test_the_model_flag_has_the_generate_scripts_spelling_as_an_alias(repo):
    args = evalrun.parse_args(["--arm", "A1", "--split", "dev", "--stages", "generate", "--model-path", "org/m",
                               "--out-root", "/tmp/x"], root=repo)
    assert args.model_path == "org/m" and args.out_root == Path("/tmp/x")


def test_languages_that_are_not_built_yet_are_skipped_with_a_message(repo, scratch, model, judge, capsys):
    build_testset(repo, ("en",), n=4)
    assert run_cli(repo, *args_for(scratch)) == 0  # default --langs is all five
    err = capsys.readouterr().err
    assert "skip hi" in err and "not built" in err
    assert files_under(scratch / "dev" / "A1" / "none") == sorted(
        f"en.{ext}" for ext in ("eval.json", "gens.jsonl", "gens.meta.json", "verdicts.jsonl"))


def test_nothing_to_evaluate_is_a_failure_with_its_own_code(repo, scratch, model, judge, capsys):
    code = run_cli(repo, *args_for(scratch, "--langs", "sw"))
    assert code == evalrun.EXIT_NO_DATA and code not in (0, 1, 2)
    assert "nothing" in capsys.readouterr().err and model.built == 0 and not scratch.exists()


def test_a_corrupt_gens_file_is_a_runtime_error(repo, scratch, model, judge, capsys):
    build_testset(repo, n=4)
    path = gen.gens_path(scratch, "dev", "A1", None, "en")
    path.parent.mkdir(parents=True)
    path.write_text("not json\nnot json either\n", encoding="utf-8")
    assert run_cli(repo, *args_for(scratch)) == evalrun.EXIT_ERROR
    assert "error" in capsys.readouterr().err and model.built == 0


def test_a_backend_crash_is_a_runtime_error_not_a_zero(repo, scratch, monkeypatch, judge):
    build_testset(repo, n=4)

    def broken(model_path, **kwargs):
        raise RuntimeError("CUDA out of memory (pretend)")

    monkeypatch.setattr(gen, "hf_backend", broken)
    assert run_cli(repo, *args_for(scratch)) == 1


def test_exit_codes_are_distinct():
    codes = [evalrun.EXIT_OK, evalrun.EXIT_ERROR, evalrun.EXIT_USAGE, evalrun.EXIT_NO_DATA, evalrun.EXIT_JUDGE_MIX,
             evalrun.EXIT_QUALITY]
    assert codes[0] == 0 and codes[2] == 2 and len(set(codes)) == len(codes)


def test_genrun_still_agrees_with_the_orchestrator_about_the_real_results_root(repo):
    assert genrun.is_in_results(repo / "results" / "x", repo)
    assert not genrun.is_in_results(repo / "scratch", repo)


# --- the script ------------------------------------------------------------------------------------------------


def run_script(*argv, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, cwd=cwd)


def test_script_help_and_usage_errors(tmp_path):
    assert run_script("--help").returncode == 0
    assert run_script().returncode == 2
    assert run_script("--arm", "A1", "--alpha", "0.5", "--split", "dev", "--judge", "fake", "--backend", "fake",
                      "--out", str(tmp_path / "never-created")).returncode == 2
    assert not (tmp_path / "never-created").exists()


def test_script_fake_run_end_to_end_and_guards(tmp_path):
    root = tmp_path / "proj"
    build_testset(root, ("en", "hi"), n=6)
    build_tasks(root, "taskdev.jsonl", ("en",), n=3)
    out = tmp_path / "out"
    common = ["--testset-dir", str(root / "data" / "testset"), "--ft-dir", str(root / "data" / "ft")]
    done = run_script("--arm", "A2", "--alpha", "0.5", "--split", "dev", "--backend", "fake", "--judge", "fake",
                      "--out", str(out), *common)
    assert done.returncode == 0, done.stderr
    assert MARK not in done.stdout + done.stderr and TASK_MARK not in done.stdout + done.stderr
    base = out / "dev" / "A2" / "0.5"
    assert len(read_jsonl(base / "en.verdicts.jsonl", Verdict)) == 6
    assert len(read_jsonl(base / "en.tasks.jsonl", TaskScore)) == 3
    assert "task: skipped" in done.stdout and (base / "hi.verdicts.jsonl").is_file()
    assert not (base / "hi.tasks.jsonl").exists()
    again = run_script("--arm", "A2", "--alpha", "0.5", "--split", "dev", "--backend", "fake", "--judge", "fake",
                       "--out", str(out), *common)
    assert again.returncode == 0
    # the guard, with the cwd in the temp dir so that a regression could only ever write under tmp_path
    for bad_root in ("results", "Results/x"):
        guard = run_script("--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "fake", "--out", bad_root,
                           *common, cwd=tmp_path)
        assert guard.returncode == 2
    assert not (tmp_path / "results").exists() and not (tmp_path / "Results").exists()
    judge_only = run_script("--arm", "A1", "--split", "dev", "--judge", "fake", "--stages", "judge", "--out", "results",
                            *common, cwd=tmp_path)
    assert judge_only.returncode == 2 and not (tmp_path / "results").exists()


def test_script_exit_code_is_nonzero_when_nothing_could_be_evaluated(tmp_path):
    done = run_script("--arm", "A1", "--split", "dev", "--backend", "fake", "--judge", "fake", "--langs", "sw",
                      "--testset-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "out"))
    assert done.returncode == evalrun.EXIT_NO_DATA


def test_the_script_is_a_thin_wrapper():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "evalrun" in text and len(text.splitlines()) < 40
    assert "import json" not in text  # no logic of its own
    assert os.access(SCRIPT, os.R_OK)


# --- concrete judges: names, set-up failures, error summary (Task 13) -------------------------------------------

SECRET_IN_MESSAGE = "sk-ant-SENTINEL-IN-AN-EXCEPTION-MESSAGE-0123456789"


class ExplodingJudge:
    """Every call raises a custom exception whose message holds a secret-looking sentinel."""

    def __init__(self, judge_id="X1"):
        self.id = judge_id

    def classify(self, kind, prompt, response):
        raise CustomJudgeBoom(SECRET_IN_MESSAGE)


class CustomJudgeBoom(Exception):
    pass


def test_the_error_type_summary_names_types_and_counts_never_messages(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=4)
    register(monkeypatch, "boom", ExplodingJudge())
    code = run_cli(repo, *args_for(scratch, judge_name="boom", model_args=False), "--stages", "generate,judge",
                   "--backend", "fake")
    assert code == evalrun.EXIT_QUALITY  # every verdict is a judge_error, so the gate fails after writing
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if "CustomJudgeBoom" in line]
    assert len(lines) == 1 and "CustomJudgeBoom x12" in lines[0]  # 4 prompts x (1 + 2 retries)
    assert SECRET_IN_MESSAGE not in captured.err and SECRET_IN_MESSAGE not in captured.out
    for path in Path(scratch).rglob("*"):
        if path.is_file():
            assert SECRET_IN_MESSAGE.encode() not in path.read_bytes(), path


def test_no_error_summary_when_no_judge_call_raised(repo, scratch, model, judge, capsys):
    build_testset(repo)
    assert run_cli(repo, *args_for(scratch)) == 0
    err = capsys.readouterr().err
    assert "judge calls that raised" not in err and "judge usage" not in err  # a clean run prints neither


def test_an_unknown_judge_name_exits_2_listing_the_registered_names_and_the_accepted_forms(repo, scratch, capsys):
    build_testset(repo)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--backend", "fake", "--out", str(scratch),
                   "--judge", "gpt-4") == 2
    err = capsys.readouterr().err
    assert "unknown judge 'gpt-4'" in err and "fake" in err
    assert "anthropic:<model-id>" in err and "hf:<model-id>" in err
    assert not scratch.exists()


@pytest.mark.parametrize("name", ["anthropic:claude-haiku-5-5", "hf:org/Judge-8B"])
def test_the_concrete_judge_names_pass_argument_parsing_without_building_anything(repo, scratch, monkeypatch, name):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    args = evalrun.parse_args(
        ["--arm", "A1", "--split", "dev", "--backend", "fake", "--out", str(scratch), "--judge", name], root=repo)
    assert args.judge == name  # no key read, no package imported, no model loaded at parse time


def test_a_concrete_judge_is_built_only_when_the_judge_stage_runs(repo, scratch, monkeypatch, model):
    build_testset(repo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "fake",
                   "--judge", "anthropic:claude-x", "--stages", "generate") == 0
    assert evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists() is False


def test_a_missing_anthropic_key_stops_cleanly_before_any_generation_or_file(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # and the repo fixture has no .env
    code = run_cli(repo, *args_for(scratch, judge_name="anthropic:claude-haiku-5-5"))
    captured = capsys.readouterr()
    assert code == evalrun.EXIT_ERROR
    assert "ANTHROPIC_API_KEY" in captured.err and captured.err.startswith("error:")
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
    assert model.built == 0 and model.prompts_seen == 0  # nothing was generated
    assert not scratch.exists()  # and nothing was written


def test_the_env_file_of_the_repo_root_supplies_the_key_and_a_judge_is_built_from_it(repo, scratch, monkeypatch, model):
    build_testset(repo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-FROM-REPO-ENV-0123456789\n", encoding="utf-8")
    built = []

    def fake_make_judge(name, *, env_file=None, environ=None):
        built.append((name, env_file))
        return StubJudge("anthropic:claude-x:tabc")

    monkeypatch.setattr(evalrun.judges, "make_judge", fake_make_judge)
    assert run_cli(repo, *args_for(scratch, judge_name="anthropic:claude-x")) == 0
    assert built == [("anthropic:claude-x", repo / ".env")]
    assert {v.judge_id for v in read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)} == {
        "anthropic:claude-x:tabc"}


def test_a_registered_name_wins_over_the_concrete_forms(repo, scratch, monkeypatch, model):
    build_testset(repo)
    monkeypatch.setitem(evalrun.JUDGE_REGISTRY, "anthropic:claude-x", lambda: StubJudge("from-registry"))
    monkeypatch.setattr(evalrun.judges, "make_judge", lambda *a, **k: pytest.fail("the registry must win"))
    assert run_cli(repo, *args_for(scratch, judge_name="anthropic:claude-x")) == 0
    assert {v.judge_id for v in read_jsonl(evalrun.verdicts_path(scratch, "dev", "A1", None, "en"), Verdict)} == {
        "from-registry"}


def test_a_judge_that_cannot_run_stops_the_run_with_one_line_instead_of_judge_errors(repo, scratch, monkeypatch, model, capsys):
    from msd.judge import JudgeSetupError

    class Unusable:
        id = "U1"
        calls = 0

        def classify(self, kind, prompt, response):
            Unusable.calls += 1
            raise JudgeSetupError("the API key was rejected (HTTP 401)")

    build_testset(repo, n=4)
    register(monkeypatch, "unusable", Unusable())
    assert run_cli(repo, *args_for(scratch, judge_name="unusable")) == evalrun.EXIT_ERROR
    err = capsys.readouterr().err
    assert "error: the API key was rejected (HTTP 401)" in err and "Traceback" not in err
    assert Unusable.calls == 1  # the first call ended the run; the other items were not tried


# --- fix round 1: summary on every exit path, breaker, HF load failure, config, usage ---------------------------


class Flaky:
    """Raises a CustomJudgeBoom for the first `boom` calls, then answers (or, with `fatal`, raises a setup error)."""

    def __init__(self, boom, fatal=False, judge_id="F1"):
        self.id, self.boom, self.fatal, self.calls = judge_id, boom, fatal, 0

    def classify(self, kind, prompt, response):
        from msd.judge import JudgeSetupError

        self.calls += 1
        if self.calls <= self.boom:
            raise CustomJudgeBoom(SECRET_IN_MESSAGE)
        if self.fatal:
            raise JudgeSetupError("the API key was rejected (HTTP 401)")
        return "complied" if kind == "harmful" else "refused"


def test_the_error_summary_also_appears_when_the_run_stops_on_a_fatal_judge_error(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=4)
    register(monkeypatch, "flaky", Flaky(boom=2, fatal=True))
    assert run_cli(repo, *args_for(scratch, judge_name="flaky")) == evalrun.EXIT_ERROR
    captured = capsys.readouterr()
    assert "error: the API key was rejected (HTTP 401)" in captured.err
    assert "judge calls that raised, by exception type: CustomJudgeBoom x2" in captured.err  # the finally path
    assert SECRET_IN_MESSAGE not in captured.err + captured.out


class HostThenDown:
    """Healthy for English prompts, every call raises for Hindi prompts (a service that fell over mid-run)."""

    def __init__(self, judge_id="H1"):
        self.id, self.calls = judge_id, []

    def classify(self, kind, prompt, response):
        self.calls.append(prompt)
        if f"{MARK} hi " in prompt:
            raise CustomJudgeBoom(SECRET_IN_MESSAGE)
        return "complied" if kind == "harmful" else "refused"


def test_a_tripped_breaker_mid_run_keeps_the_finished_language_and_resumes_cheaply(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, ("en", "hi"), n=12)
    down = register(monkeypatch, "h1", HostThenDown())
    argv = args_for(scratch, judge_name="h1", model_args=True)
    assert run_cli(repo, *argv, "--langs", "en,hi") == evalrun.EXIT_ERROR
    captured = capsys.readouterr()
    en_verdicts = evalrun.verdicts_path(scratch, "dev", "A1", None, "en")
    assert len(read_jsonl(en_verdicts, Verdict)) == 12 and evalrun.eval_path(scratch, "dev", "A1", None, "en").is_file()
    assert not evalrun.verdicts_path(scratch, "dev", "A1", None, "hi").exists()
    assert not evalrun.eval_path(scratch, "dev", "A1", None, "hi").exists()
    assert "30 judge calls in a row raised" in captured.err and "CustomJudgeBoom x30" in captured.err
    assert "Traceback" not in captured.err and SECRET_IN_MESSAGE not in captured.err + captured.out
    for path in Path(scratch).rglob("*"):
        if path.is_file():
            assert SECRET_IN_MESSAGE.encode() not in path.read_bytes(), path

    healthy = register(monkeypatch, "h1", StubJudge("H1"))  # the service is back
    en_before = en_verdicts.read_bytes()
    assert run_cli(repo, *argv, "--langs", "en,hi", "--stages", "judge") == 0
    assert en_verdicts.read_bytes() == en_before
    assert len(healthy.calls) == 12  # only the Hindi items; English came from the cache


def test_a_hugging_face_load_failure_stops_after_one_attempt_without_leaking(repo, scratch, monkeypatch, model, fake_hf_stack, capsys):
    build_testset(repo, n=4)
    torch, transformers, log = fake_hf_stack(fail_at="tokenizer", error=OSError(f"gated repo token {SECRET_IN_MESSAGE}"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert run_cli(repo, *args_for(scratch, judge_name="hf:org/Model")) == evalrun.EXIT_ERROR
    captured = capsys.readouterr()
    assert "error: could not load org/Model (OSError)" in captured.err and "Traceback" not in captured.err
    assert log.load_attempts == 1  # not once per judgment
    assert SECRET_IN_MESSAGE not in captured.err + captured.out
    assert not evalrun.verdicts_path(scratch, "dev", "A1", None, "en").exists()
    for path in Path(scratch).rglob("*"):
        if path.is_file():
            assert SECRET_IN_MESSAGE.encode() not in path.read_bytes(), path


class ConfiguredJudge(StubJudge):
    """A stub with the optional `config()` and `usage()` hooks."""

    def __init__(self, judge_id="C1", config=None, usage=None):
        super().__init__(judge_id)
        self._config, self._usage = config, usage

    def config(self):
        return self._config

    def usage(self):
        return dict(self._usage) if self._usage is not None else {"calls": len(self.calls), "input_tokens": 0,
                                                                    "output_tokens": 0}


PRE_EXISTING_KEYS = {
    "arm", "alpha", "lang", "split", "model_path", "backend", "judge_id", "n_generations", "n_by_kind",
    "finish_reasons", "truncation", "truncation_safety", "empty_outputs", "empty_or_error", "verdicts", "judge_error",
    "tasks", "task_skipped", "warnings",
}


def test_eval_json_records_the_judge_configuration_next_to_the_id(repo, scratch, monkeypatch, model):
    from msd.judge import MAX_CONSECUTIVE_FAILURES, template_fingerprint

    build_testset(repo, n=4)
    register(monkeypatch, "c1", ConfiguredJudge(config={"type": "stub", "model": "m", "max_tokens": 512,
                                                         "sampling": "default"}))
    assert run_cli(repo, *args_for(scratch, judge_name="c1")) == 0
    report = eval_json(scratch)
    assert report["judge_id"] == "C1"
    assert report["judge"] == {"type": "stub", "model": "m", "max_tokens": 512, "sampling": "default",
                               "template_fingerprint": template_fingerprint(), "retries": 2,
                               "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES}
    assert set(report) == PRE_EXISTING_KEYS | {"judge"}  # additive: nothing else changed


def test_a_judge_without_config_still_records_the_runner_settings(repo, scratch, model, judge):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch)) == 0
    report = eval_json(scratch)
    assert set(report["judge"]) == {"template_fingerprint", "retries", "max_consecutive_failures"}


def test_a_run_without_a_judge_has_no_judge_section(repo, scratch, model):
    build_testset(repo, n=4)
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--backend", "hf", "--model", "org/model",
                   "--stages", "generate") == 0
    assert eval_json(scratch)["judge"] is None


def test_the_configuration_is_redacted_like_the_id_and_reruns_are_byte_identical(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    monkeypatch.setenv("MSD_TEST_API_KEY", "sk-ant-SENTINEL-CONFIG-0123456789")
    register(monkeypatch, "c1", ConfiguredJudge(config={"type": "stub", "note": "sk-ant-SENTINEL-CONFIG-0123456789"}))
    assert run_cli(repo, *args_for(scratch, judge_name="c1")) == 0
    path = evalrun.eval_path(scratch, "dev", "A1", None, "en")
    first = path.read_bytes()
    assert b"SENTINEL-CONFIG" not in first and eval_json(scratch)["judge"]["note"] == "***"
    assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="c1")) == 0
    assert path.read_bytes() == first  # no clock, no token counts: byte-identical


def test_a_later_stage_without_the_judge_keeps_the_recorded_configuration(repo, scratch, monkeypatch, model):
    build_testset(repo, n=4)
    build_tasks(repo, "taskdev.jsonl", n=3)
    register(monkeypatch, "c1", ConfiguredJudge(config={"type": "stub", "model": "m"}))
    assert run_cli(repo, *args_for(scratch, judge_name="c1")) == 0
    recorded = eval_json(scratch)["judge"]
    assert recorded["model"] == "m"
    assert run_cli(repo, "--arm", "A1", "--split", "dev", "--out", str(scratch), "--stages", "task") == 0
    assert eval_json(scratch)["judge"] == recorded


def test_the_stderr_summary_includes_token_usage_only_when_a_judge_call_was_made(repo, scratch, monkeypatch, model, capsys):
    build_testset(repo, n=4)
    register(monkeypatch, "c1", ConfiguredJudge(usage={"calls": 4, "input_tokens": 1234, "output_tokens": 8}))
    assert run_cli(repo, *args_for(scratch, judge_name="c1")) == 0
    err = capsys.readouterr().err
    assert "judge usage: calls 4, input_tokens 1234, output_tokens 8" in err

    # everything is cached now: no call is made, so nothing is printed
    register(monkeypatch, "c1", ConfiguredJudge(usage={"calls": 0, "input_tokens": 0, "output_tokens": 0}))
    assert run_cli(repo, *args_for(scratch, "--rejudge", judge_name="c1")) == 0
    assert "judge usage" not in capsys.readouterr().err


def test_usage_is_optional_and_only_whole_numbers_are_printed(repo, scratch, tmp_path, monkeypatch, model, judge, capsys):
    build_testset(repo, n=4)
    assert run_cli(repo, *args_for(scratch)) == 0  # StubJudge has no usage()
    assert "judge usage" not in capsys.readouterr().err

    register(monkeypatch, "c2", ConfiguredJudge("C2", usage={"calls": 2, "input_tokens": 10, "note": SECRET_IN_MESSAGE,
                                                              "output_tokens": 1.5}))
    assert run_cli(repo, *args_for(tmp_path / "other_out", judge_name="c2")) == 0
    err = capsys.readouterr().err
    assert "judge usage: calls 2, input_tokens 10" in err and SECRET_IN_MESSAGE not in err and "1.5" not in err
