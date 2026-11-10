"""scripts/export_public.py and msd.export: the public export, as a dry run into a folder (Task 19, Steps 1-5).

Everything runs on fixture trees under tmp_path: a frozen fixture test set (tests/conftest.py), a fixture results
tree with its real report outputs (tests/results_tree.py + msd.reportrun), a fixture sources manifest. No network, no
GPU, no paid API, no .env. The export writes only under --out, and these tests prove that nothing in it could carry a
full completion, a home path or a secret, and that the code has no way to publish anything.
"""

import ast
import contextlib
import dataclasses
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
import yaml
from conftest import BENIGN, HARM_A, HARM_B, entry
from results_tree import ALPHAS, SENTINEL, build_tree, edit_cell, write_cell, write_second_judge, default_spec

from msd import config, evalrun, export, redact, reportrun, testset
from msd.schema import Generation, Prompt, TaskScore, Verdict, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "export_public.py"
TEMPLATES = ROOT / "templates"

USERS = "/" + "Users"  # never written out whole: tests/ is scanned for home paths (tests/test_hygiene.py)
HOME = "/" + "home"
REPLACEMENT_BYTES = bytes([0xEF, 0xBF, 0xBD])

VERDICT_KEYS = ["prompt_id", "lang", "arm", "alpha", "label", "judge_id", "excerpt"]
TASK_KEYS = ["prompt_id", "lang", "arm", "alpha", "correct"]
PROMPT_KEYS = ["id", "lang", "kind", "split", "source", "license", "text"]
TRANSLATED_KEYS = ["id", "lang", "kind", "split", "source", "source_license", "license", "derived_from", "text"]
TRANSLATORS = {"hi": "nllb-600m", "ar": "nllb-600m", "sw": "nllb-600m", "hinglish": "llm-chat"}
EVAL_ONLY, EXCLUDED, FT_ONLY = "fix/eval-only", "fix/excluded", "fix/finetune"


# --- fixtures ------------------------------------------------------------------------------------------------------


def write_sources(path, **overrides):
    """A fixture manifest: the three test-set sources, an evaluation-only one, an excluded one, a fine-tune one."""
    entries = [
        entry(HARM_A, "harmful_prompts", "mit"),
        entry(HARM_B, "harmful_prompts", "cc-by-4.0"),
        entry(BENIGN, "benign_edgy_prompts", "CC BY 4.0 (fixture)"),
        entry(EVAL_ONLY, "task_cross_check", "cc-by-sa-4.0", redistribute=False),
        entry(EXCLUDED, "excluded", "cc-by-nc-4.0", notes="non-commercial licence"),
        entry(FT_ONLY, "fine_tune_data", "apache-2.0"),
    ]
    for name, changes in overrides.items():
        for e in entries:
            if e["name"] == name:
                e.update(changes)
    path.write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def frozen_testset(tmp_path, build, monkeypatch):
    """A real frozen fixture test set, its build record beside it, translated by names a real build would record."""
    translate = lambda texts, src, tgt: [f"[{tgt}] {t}" for t in texts]  # noqa: E731
    monkeypatch.setitem(testset.TRANSLATORS, "nllb-600m", lambda: translate)
    monkeypatch.setitem(testset.TRANSLATORS, "llm-chat", lambda: translate)
    build(freeze=True, summary_path=tmp_path / "testset_build_summary.json", translator_names=TRANSLATORS)
    return tmp_path / "testset"


@pytest.fixture(scope="module")
def made_results(tmp_path_factory):
    """A results tree with its second judge and the four report outputs make_report writes beside it."""
    root = build_tree(tmp_path_factory.mktemp("base") / "results")
    write_second_judge(root)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert reportrun.main(["--verdicts", str(root), "--out", str(root)], root=root.parent / "repo") == 0
    return root


@pytest.fixture
def results(made_results, tmp_path):
    dest = tmp_path / "results"
    shutil.copytree(made_results, dest)
    return dest


@pytest.fixture
def sources(tmp_path):
    return write_sources(tmp_path / "sources.yaml")


@pytest.fixture
def run(capsys, tmp_path, results, frozen_testset, sources):
    """run(*extra, out=None, **paths) -> (exit code, stdout, stderr, out dir); every path defaults to the fixtures."""

    def go(*extra, out=None, **paths):
        out = Path(out) if out is not None else tmp_path / "export"
        args = {"results": results, "testset": frozen_testset, "sources": sources, "templates": TEMPLATES, **paths}
        argv = [f"--{name.replace('_', '-')}={value}" for name, value in args.items() if value is not None]
        code = export.main([*argv, "--out", str(out), *extra], root=tmp_path / "repo")
        captured = capsys.readouterr()
        return code, captured.out, captured.err, out

    return go


def files_under(out):
    return sorted(p.relative_to(out).as_posix() for p in Path(out).rglob("*") if p.is_file())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def assert_nothing_written(tmp_path, out):
    assert not Path(out).exists(), "a refused export must leave no output folder"
    assert not list(tmp_path.glob(".export*")), "a refused export must leave no partial folder"


# --- the layout ----------------------------------------------------------------------------------------------------


def test_export_writes_the_documented_layout_and_nothing_else(run):
    code, out_text, err_text, out = run()
    assert code == export.EXIT_OK, err_text
    files = files_under(out)
    fixed = {
        "MANIFEST.json", "dataset/README.md", "dataset/prompts.jsonl", "dataset/prompts_translated.jsonl",
        "dataset/aggregates/tables.csv", "dataset/aggregates/reliability.json", "dataset/aggregates/heatmap.png",
        "dataset/aggregates/alpha_curve.png", "models/A1/README.md", "models/A2/README.md",
    }
    assert fixed <= set(files)
    rest = sorted(set(files) - fixed)
    assert rest and all(re.fullmatch(r"dataset/results_public/(dev|report)/A[0-9a-z]+/(none|[0-9.]+)/"
                                     r"(en|hi|hinglish|ar|sw)\.(verdicts|tasks)\.jsonl", f) for f in rest), rest
    # no completions file, no eval.json, no second-judge folder, no cache: only what the public rows come from
    assert not [f for f in files if f.endswith((".gens.jsonl", ".eval.json")) or "/reliability/" in f or "_cache" in f]
    assert "A0" in "".join(rest) and not (out / "models" / "A0").exists()  # a model card only for A1, A2, A3
    assert (out / "dataset" / "results_public" / "report" / "A1" / "none" / "hi.verdicts.jsonl").is_file()
    assert (out / "dataset" / "results_public" / "dev" / "A2" / "0.25" / "sw.tasks.jsonl").is_file()


def test_stdout_says_dry_run_and_lists_the_files_but_never_row_text(run):
    code, out_text, err_text, out = run()
    assert code == export.EXIT_OK
    assert "dry run" in out_text.lower()
    for name in ("MANIFEST.json", "dataset/prompts.jsonl", "dataset/prompts_translated.jsonl", "models/A2/README.md"):
        assert name in out_text
    for text in (out_text, err_text):
        assert SENTINEL not in text
    assert files_under(out)


def test_only_a1_a2_and_a3_get_a_model_card_and_a3_only_when_it_ran(run, results):
    code, *_rest, out = run()
    assert code == 0
    assert sorted(p.name for p in (out / "models").iterdir()) == ["A1", "A2"]
    for split in ("dev", "report"):
        for lang in config.LANGS:
            write_cell(results, default_spec(split, "A3", None, lang))
    code, _o, err, out3 = run(out=out.parent / "export3")
    assert code == 0, err
    assert sorted(p.name for p in (out3 / "models").iterdir()) == ["A1", "A2", "A3"]
    assert (out3 / "dataset" / "results_public" / "report" / "A3" / "none" / "ar.verdicts.jsonl").is_file()


# --- results_public: only the public row -----------------------------------------------------------------------------


def test_every_exported_verdict_row_has_only_the_spec_fields(run, made_results):
    code, *_rest, out = run()
    assert code == 0
    exported = sorted((out / "dataset" / "results_public").rglob("*.verdicts.jsonl"))
    assert exported
    total = 0
    for path in exported:
        for row in read_rows(path):
            assert list(row) == VERDICT_KEYS
            assert len(row["excerpt"]) <= 200
            assert row["judge_id"] == "stub-judge-v1"
            total += 1
    source = sum(len(read_rows(p)) for split in ("dev", "report")
                 for p in (made_results / split).glob("*/*/*.verdicts.jsonl"))  # a second judge's labels sit deeper
    assert total == source  # every verdict of every cell is there: none lost, none added


def test_task_rows_carry_correctness_and_no_text(run, made_results):
    code, *_rest, out = run()
    assert code == 0
    rows = read_rows(out / "dataset" / "results_public" / "report" / "A1" / "none" / "en.tasks.jsonl")
    assert rows and all(list(r) == TASK_KEYS and isinstance(r["correct"], bool) for r in rows)
    source = read_jsonl(made_results / "report" / "A1" / "none" / "en.tasks.jsonl", TaskScore)
    assert [r["prompt_id"] for r in rows] == [s.prompt_id for s in source]
    assert [r["correct"] for r in rows] == [s.correct for s in source]


LONG_TAIL = "UNIQUETAIL"


def plant_long_completions(results, *cells):
    """Make the chosen cells' verdict excerpts, and their gens files, hold 600-character completions."""
    for split, arm, alpha, lang in cells:
        spec = default_spec(split, arm, alpha, lang)
        write_cell(results, spec)
        path = evalrun.verdicts_path(results, split, arm, alpha, lang)
        rows = read_jsonl(path, Verdict)
        long = [f"[{i}] " + " ".join(["x" * 9] * 45) + f" {LONG_TAIL}{i:03d} " + "y" * 100 for i, _ in enumerate(rows)]
        write_jsonl(path, [Verdict(v.prompt_id, v.arm, v.alpha, v.lang, v.kind, v.judge_id, v.label, text)
                           for v, text in zip(rows, long)])  # an old or foreign file whose excerpts are whole completions
        gens = path.with_name(f"{lang}.gens.jsonl")
        write_jsonl(gens, [Generation(v.prompt_id, v.arm, v.alpha, text, "stop") for v, text in zip(rows, long)])
    return len(long)


def test_export_contains_no_full_completions(run, results):
    cells = [("report", "A1", None, "hi"), ("report", "A2", 0.25, "en"), ("dev", "A1", None, "sw")]
    n = plant_long_completions(results, *cells)
    code, out_text, err_text, out = run()
    assert code == 0, err_text
    assert LONG_TAIL not in out_text + err_text
    every = {rel: (out / rel).read_bytes() for rel in files_under(out)}
    for rel, data in every.items():  # cards and manifest included
        assert LONG_TAIL.encode() not in data, rel
        assert " ".join(["x" * 9] * 30).encode() not in data, rel  # an excerpt holds fewer than 200 characters
        assert b"y" * 20 not in data, rel
    rows = read_rows(out / "dataset" / "results_public" / "report" / "A1" / "none" / "hi.verdicts.jsonl")
    assert len(rows) == 60 and n == 60
    for i, row in enumerate(rows):  # not vacuous: the head of each completion is what was kept, minus its last word
        head = (f"[{i}] " + " ".join(["x" * 9] * 45))[:199]
        assert row["excerpt"] == head[:head.rindex(" ")] + chr(0x2026)  # the word the cut fell in is dropped, a marker added
        assert list(row) == VERDICT_KEYS


def test_the_export_never_opens_a_completions_file(run, results):
    for gens in results.rglob("*.gens.jsonl"):
        gens.unlink()
    code, _o, err, out = run()
    assert code == 0, err  # the same export without a single completion on disk


def test_a_secret_shaped_excerpt_is_redacted_in_the_public_row(run, results):
    path = results / "report" / "A1" / "none" / "hi.verdicts.jsonl"
    rows = read_jsonl(path, Verdict)
    token = "hf_" + "Q1w2E3r4T5" * 3
    write_jsonl(path, [Verdict(v.prompt_id, v.arm, v.alpha, v.lang, v.kind, v.judge_id, v.label, f"see {token} ok")
                       for v in rows])
    code, _o, err, out = run()
    assert code == 0, err
    blob = (out / "dataset" / "results_public" / "report" / "A1" / "none" / "hi.verdicts.jsonl").read_text("utf-8")
    assert token not in blob and "see *** ok" in blob


def test_a_test_judge_is_refused_with_the_judge_exit_code(run, results, tmp_path):
    edit_cell(results, "report", "A1", None, "hi", judge="fake-judge-1")
    code, _o, err, out = run()
    assert code == export.EXIT_JUDGE
    assert "judge" in err.lower()
    assert_nothing_written(tmp_path, out)


def test_a_missing_report_split_is_exit_3(run, results, tmp_path):
    shutil.rmtree(results / "report")
    code, _o, err, out = run()
    assert code == export.EXIT_NO_DATA
    assert "report" in err
    assert_nothing_written(tmp_path, out)


def test_a_report_split_without_judged_verdicts_is_exit_3(run, results, tmp_path):
    for arm, alphas in (("A0", (None,)), ("A1", (None,)), ("A2", (0.1, 0.25, 0.5))):
        for alpha in alphas:
            for lang in config.LANGS:
                edit_cell(results, "report", arm, alpha, lang, with_verdicts=False)  # the judge stage never ran
    code, _o, err, out = run()
    assert code == export.EXIT_NO_DATA
    assert "judge" in err
    assert_nothing_written(tmp_path, out)


def test_a_missing_dev_split_is_noted_and_the_export_goes_on(run, results):
    shutil.rmtree(results / "dev")
    code, _o, err, out = run()
    assert code == 0
    assert "dev" in err.lower()
    assert not (out / "dataset" / "results_public" / "dev").exists()


def test_an_inconsistent_results_tree_is_refused_by_the_walker(run, results, tmp_path):
    (results / "report" / "NOT-AN-ARM").mkdir()
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR
    assert "NOT-AN-ARM" in err
    assert_nothing_written(tmp_path, out)


# --- prompts: licence rules and the translated file ------------------------------------------------------------------


def test_prompts_are_english_originals_and_translations_go_to_their_own_marked_file(run, frozen_testset):
    code, *_rest, out = run()
    assert code == 0
    english = read_rows(out / "dataset" / "prompts.jsonl")
    translated = read_rows(out / "dataset" / "prompts_translated.jsonl")
    assert len(english) == 21 and len(translated) == 84  # 12 harmful + 9 benign-edgy per language, four translations
    assert all(list(r) == PROMPT_KEYS and r["lang"] == "en" for r in english)
    assert all(list(r) == TRANSLATED_KEYS and r["lang"] != "en" for r in translated)
    assert {r["license"] for r in translated} == {"CC-BY-NC-4.0"}
    for r in translated:
        assert r["derived_from"] == ("llm-chat" if r["lang"] == "hinglish" else "NLLB"), r["lang"]
    assert {r["source"] for r in english} <= {HARM_A, HARM_B, BENIGN}
    assert {(r["source"], r["license"]) for r in english} == {(HARM_A, "mit"), (HARM_B, "cc-by-4.0"),
                                                               (BENIGN, "CC BY 4.0 (fixture)")}
    assert {(r["source"], r["source_license"]) for r in translated} == {(HARM_A, "mit"), (HARM_B, "cc-by-4.0"),
                                                                        (BENIGN, "CC BY 4.0 (fixture)")}
    # the same rows as the frozen test set: ids, kinds, splits and texts, nothing invented, nothing dropped
    frozen = [p for lang in config.LANGS for split in ("dev", "report")
              for p in read_jsonl(frozen_testset / lang / f"{split}.jsonl", Prompt)]
    published = {r["id"]: r for r in english + translated}
    assert sorted(published) == sorted(p.id for p in frozen)
    for p in frozen:
        r = published[p.id]
        assert (r["lang"], r["kind"], r["split"], r["source"], r["text"]) == (p.lang, p.kind, p.split, p.source, p.text)
    # deleting the one file removes every translated prompt, and the manifest says that file is optional
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"]["dataset/prompts_translated.jsonl"]["optional"] is True
    assert manifest["files"]["dataset/prompts.jsonl"]["optional"] is False


def test_a_source_that_is_evaluation_only_is_refused_by_name(run, sources, tmp_path):
    write_sources(sources, **{HARM_B: {"redistribute": False}})
    code, _o, err, out = run()
    assert code == export.EXIT_LICENSE
    assert HARM_B in err and "redistribute" in err
    assert_nothing_written(tmp_path, out)


def test_an_excluded_source_and_an_unknown_source_are_refused_by_name(run, sources, tmp_path):
    write_sources(sources, **{HARM_A: {"used_for": "excluded", "notes": "now excluded"}})
    code, _o, err, out = run()
    assert code == export.EXIT_LICENSE and HARM_A in err
    assert_nothing_written(tmp_path, out)

    manifest = yaml.safe_load(sources.read_text(encoding="utf-8"))
    sources.write_text(yaml.safe_dump([e for e in manifest if e["name"] != BENIGN], sort_keys=False), encoding="utf-8")
    code, _o, err, out = run()
    assert code == export.EXIT_LICENSE and BENIGN in err and "manifest" in err
    assert_nothing_written(tmp_path, out)


def test_an_unfrozen_or_changed_test_set_is_refused(run, frozen_testset, tmp_path):
    english = frozen_testset / "en" / "dev.jsonl"
    original = english.read_bytes()
    english.write_bytes(original + original.split(b"\n")[0] + b"\n")  # one row added after the freeze
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "dev.jsonl" in err
    assert_nothing_written(tmp_path, out)

    english.write_bytes(original)
    (frozen_testset / "MANIFEST.json").unlink()
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "MANIFEST.json" in err
    assert_nothing_written(tmp_path, out)


def test_a_missing_build_record_or_an_identity_translator_is_refused(run, frozen_testset, tmp_path):
    record = frozen_testset.parent / "testset_build_summary.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    data["languages"]["hi"]["translator"] = "identity"  # the texts would be copies, not translations
    record.write_text(json.dumps(data), encoding="utf-8")
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "hi" in err and "identity" in err
    assert_nothing_written(tmp_path, out)

    record.unlink()
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "build record" in err
    assert_nothing_written(tmp_path, out)


# --- aggregates -------------------------------------------------------------------------------------------------------


def test_the_four_aggregates_are_copied_byte_for_byte_and_hashed(run, made_results):
    code, *_rest, out = run()
    assert code == 0
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in ("tables.csv", "reliability.json", "heatmap.png", "alpha_curve.png"):
        source = (made_results / name).read_bytes()
        assert (out / "dataset" / "aggregates" / name).read_bytes() == source
        assert manifest["files"][f"dataset/aggregates/{name}"]["sha256"] == sha(source)


def test_a_missing_table_is_refused_and_a_missing_figure_is_a_warning(run, results, tmp_path):
    (results / "heatmap.png").unlink()
    code, _o, err, out = run()
    assert code == 0
    assert "heatmap.png" in err
    assert "dataset/aggregates/heatmap.png" not in json.loads((out / "MANIFEST.json").read_text("utf-8"))["files"]
    shutil.rmtree(out)
    (results / "tables.csv").unlink()
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "tables.csv" in err and "make_report" in err
    assert_nothing_written(tmp_path, out)


def test_aggregates_may_come_from_another_folder(run, results, tmp_path, made_results):
    agg = tmp_path / "agg"
    agg.mkdir()
    for name in ("tables.csv", "reliability.json", "heatmap.png", "alpha_curve.png"):
        (results / name).rename(agg / name)
    code, _o, err, out = run("--aggregates", str(agg))
    assert code == 0, err
    assert (out / "dataset" / "aggregates" / "tables.csv").read_bytes() == (made_results / "tables.csv").read_bytes()


# --- the manifest ---------------------------------------------------------------------------------------------------


def test_the_manifest_is_deterministic_complete_and_free_of_clocks_and_paths(run, tmp_path):
    code, _o, _e, first = run(out=tmp_path / "one")
    code2, _o2, _e2, second = run(out=tmp_path / "two")
    assert code == code2 == 0
    assert (first / "MANIFEST.json").read_bytes() == (second / "MANIFEST.json").read_bytes()
    assert {rel: (first / rel).read_bytes() for rel in files_under(first)} == \
           {rel: (second / rel).read_bytes() for rel in files_under(second)}

    text = (first / "MANIFEST.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert text == json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert set(manifest) == {"checks", "format", "files", "testset_freeze_sha256"}
    assert manifest["testset_freeze_sha256"] == sha((tmp_path / "testset" / "MANIFEST.json").read_bytes())
    assert sorted(manifest["files"]) == [f for f in files_under(first) if f != "MANIFEST.json"]
    for rel, info in manifest["files"].items():
        data = (first / rel).read_bytes()
        assert set(info) == {"bytes", "n", "optional", "role", "sha256"}
        assert info["sha256"] == sha(data) and info["bytes"] == len(data)
        if rel.endswith(".jsonl"):
            assert info["n"] == len([line for line in data.split(b"\n") if line.strip()])
        elif rel.endswith(".csv"):
            assert info["n"] == len([line for line in data.split(b"\n") if line.strip()]) - 1  # without the header
        else:
            assert info["n"] is None
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text)  # no timestamp
    assert str(tmp_path) not in text and USERS not in text and HOME not in text


def test_the_manifest_changes_when_the_inputs_change(run, results, tmp_path):
    _c, _o, _e, first = run(out=tmp_path / "one")
    edit_cell(results, "report", "A1", None, "hi", asr=0.9)  # another verdict mix
    _c, _o, _e, second = run(out=tmp_path / "two")
    one = json.loads((first / "MANIFEST.json").read_text("utf-8"))["files"]
    two = json.loads((second / "MANIFEST.json").read_text("utf-8"))["files"]
    key = "dataset/results_public/report/A1/none/hi.verdicts.jsonl"
    assert one[key]["sha256"] != two[key]["sha256"]
    assert one["dataset/prompts.jsonl"] == two["dataset/prompts.jsonl"]


# --- the output folder ----------------------------------------------------------------------------------------------


def test_a_non_empty_out_folder_is_refused_and_left_alone(run, tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    code, _o, err, _out = run(out=out)
    assert code == export.EXIT_USAGE
    assert "not empty" in err
    assert files_under(out) == ["keep.txt"] and (out / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_an_empty_out_folder_is_fine_and_an_out_folder_inside_an_input_is_not(run, results, tmp_path):
    (tmp_path / "export").mkdir()
    code, _o, err, out = run()
    assert code == 0, err
    code, _o, err, _out = run(out=results / "export")
    assert code == export.EXIT_USAGE
    assert "results" in err
    assert not (results / "export").exists()


def test_the_script_takes_no_force_flag_and_writes_only_under_out(run, tmp_path, results, frozen_testset):
    parser = export.build_parser()
    assert not any("force" in opt for action in parser._actions for opt in action.option_strings)
    before = {p for p in tmp_path.rglob("*")}
    code, *_rest, out = run()
    assert code == 0
    new = {p for p in tmp_path.rglob("*")} - before
    assert new and all(p == out or out in p.parents for p in new)  # nothing outside --out changed


# --- templates -------------------------------------------------------------------------------------------------------


def test_render_fills_every_placeholder_and_a_missing_value_is_an_error():
    assert export.render_template("a {{x}} b {{ y }} c {{x}}", {"x": "1", "y": "2"}) == "a 1 b 2 c 1"
    assert export.render_template("no placeholders", {"unused": "v"}) == "no placeholders"
    with pytest.raises(export.TemplateError, match="y"):
        export.render_template("a {{x}} {{y}}", {"x": "1"})
    for blank in ("", "  \n", None):
        with pytest.raises(export.TemplateError, match="x"):
            export.render_template("{{x}}", {"x": blank})
    with pytest.raises(export.TemplateError, match="malformed"):
        export.render_template("{{ not a name }}", {})
    with pytest.raises(export.TemplateError, match="malformed"):
        export.render_template("stray }} braces", {})
    # a value that itself holds braces is inserted as text, never expanded again
    assert export.render_template("{{x}}", {"x": "{{y}}", "y": "no"}) == "{{y}}"


def test_a_placeholder_without_a_value_stops_the_export(run, tmp_path):
    templates = tmp_path / "templates"
    shutil.copytree(TEMPLATES, templates)
    card = templates / "dataset_card.md"
    card.write_text(card.read_text(encoding="utf-8") + "\n{{value_nobody_supplies}}\n", encoding="utf-8")
    code, _o, err, out = run(templates=templates)
    assert code == export.EXIT_ERROR
    assert "value_nobody_supplies" in err and "dataset_card.md" in err
    assert_nothing_written(tmp_path, out)


def test_the_shipped_templates_use_only_known_placeholders_and_are_clean():
    for name in ("model_card.md", "dataset_card.md"):
        data = (TEMPLATES / name).read_bytes()
        assert REPLACEMENT_BYTES not in data
        assert not re.search(rb"(?<![\w.~-])/(?:Users|home)/[A-Za-z0-9._-]+", data)
        text = data.decode("utf-8")
        found = set(re.findall(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}", text))
        assert found, f"{name} has no placeholders"
        assert found <= (export.MODEL_CARD_FIELDS if name == "model_card.md" else export.DATASET_CARD_FIELDS), name


# --- the cards -------------------------------------------------------------------------------------------------------


def test_model_card_names_license_and_aup(run):
    code, *_rest, out = run()
    assert code == 0
    for arm in ("A1", "A2"):
        card = (out / "models" / arm / "README.md").read_text(encoding="utf-8")
        assert "CC-BY-NC" in card
        assert "Acceptable Use Policy" in card
        assert "CohereLabs/tiny-aya-global" in card and "gated" in card
        assert "inherit" in card  # the fine-tuned and the repaired models carry the base model's terms
        assert "LSAL" in card and "AlignTune" in card and "CuratorKIT" in card and "SafeTune" in card
        assert "unsloth" in card and "AGPL" in card and "mergekit" in card and "LGPL" in card
        assert "not a safety guarantee" in card.lower()
        assert "research" in card.lower() and "article" in card.lower()
        assert ", ".join(config.LANGS) in card
        for limit in ("Hinglish", "task-accuracy", "judge", "native-speaker", "small"):
            assert limit in card, limit
        assert "harmless" in card and "closed-form" in card
        assert f"{arm}" in card


def test_the_a2_card_says_which_alpha_was_chosen_and_the_a1_card_names_no_repair(run, made_results):
    code, *_rest, out = run()
    assert code == 0
    chosen = json.loads((made_results / "reliability.json").read_text("utf-8"))["alpha_choice"]["chosen"]
    a2 = (out / "models" / "A2" / "README.md").read_text(encoding="utf-8")
    a1 = (out / "models" / "A1" / "README.md").read_text(encoding="utf-8")
    assert chosen is not None and f"alpha {chosen:g}" in a2
    assert "ReSta" in a2 and "tiny-aya-base" in a2
    assert "No repair step" in a1 and "alpha" not in a1  # A1 is the fine-tuned model: there is no alpha to name


def test_dataset_card_lists_sources_licences_and_the_rules(run, frozen_testset, tmp_path):
    code, *_rest, out = run()
    assert code == 0
    card = (out / "dataset" / "README.md").read_text(encoding="utf-8")
    manifest = yaml.safe_load((tmp_path / "sources.yaml").read_text(encoding="utf-8"))
    for e in manifest:
        if e["used_for"] == "excluded":
            assert e["name"] not in card
            continue
        row = next(line for line in card.splitlines() if e["name"] in line and line.startswith("|"))
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert cells[1] == e["license"] and cells[2] == e["used_for"], row
        assert cells[3].startswith("no" if e.get("redistribute") is False else "yes"), row
    eval_row = next(line for line in card.splitlines() if EVAL_ONLY in line and line.startswith("|"))
    assert "evaluation only" in eval_row
    freeze = sha((frozen_testset / "MANIFEST.json").read_bytes())
    assert freeze in card
    assert sha((out / "dataset" / "aggregates" / "tables.csv").read_bytes()) in card
    for text in ("200 characters", "dev", "report", "NLLB", "CC-BY-NC-4.0", "MultiJail", "ethic", "prompts_translated.jsonl",
                 "research and evaluation", "full completion"):
        assert text in card, text
    assert "CC-BY-NC" in card and "Acceptable Use Policy" in card  # the model outputs in the excerpts carry the terms


def test_cards_never_hold_a_home_path_a_replacement_character_or_a_secret_shape(run):
    code, *_rest, out = run()
    assert code == 0
    for rel in files_under(out):
        data = (out / rel).read_bytes()
        assert REPLACEMENT_BYTES not in data, rel
        assert not re.search(rb"(?<![\w.~-])/(?:Users|home)/[A-Za-z0-9._-]+", data), rel
        assert not re.search(rb"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,})", data), rel


# --- the safety scan: paths and secrets never leave -------------------------------------------------------------------


def plant_in_reliability(results, text):
    path = results / "reliability.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["warnings"] = [*data["warnings"], text]
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("planted", [
    f"see {USERS}/someone/code/results/tables.csv",
    f"run on {HOME}/runner/work/msd",
    "C:" + chr(92) + "Users" + chr(92) + "someone" + chr(92) + "msd",
], ids=["macos-home", "linux-home", "windows-home"])
def test_a_home_path_in_any_output_stops_the_export(run, results, tmp_path, planted):
    plant_in_reliability(results, planted)
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "reliability.json" in err_text and "path" in err_text
    assert planted not in out_text + err_text and "someone" not in out_text + err_text  # the message never echoes it
    assert_nothing_written(tmp_path, out)


def test_the_path_of_an_input_or_output_folder_in_any_output_stops_the_export(run, results, tmp_path):
    plant_in_reliability(results, f"computed from {results}")
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR and "path" in err_text
    assert_nothing_written(tmp_path, out)


@pytest.mark.parametrize("planted", [
    "hf_" + "AbCdEf0123" * 3,
    "sk-ant-api03-" + "Zx9Yw8Vu7T" * 3,
    "sk-" + "proj" + "0123456789abcdefghij",
    "ghp_" + "A1B2C3D4E5" * 3,
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "-----BEGIN " + "PRIVATE KEY-----",
], ids=["hf", "anthropic", "openai", "github", "aws", "private-key"])
def test_a_secret_shaped_string_in_any_output_stops_the_export(run, results, tmp_path, planted):
    tables = results / "tables.csv"
    tables.write_text(tables.read_text(encoding="utf-8") + f"# {planted}\n", encoding="utf-8")
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "tables.csv" in err_text and "secret" in err_text
    assert planted not in out_text + err_text
    assert_nothing_written(tmp_path, out)


def test_the_value_of_a_secret_environment_variable_in_any_output_stops_the_export(run, results, tmp_path, monkeypatch):
    value = "plain-words-but-secret-0451"  # not token-shaped: only the environment tells it is a secret
    monkeypatch.setenv("FIXTURE_API_KEY", value)
    plant_in_reliability(results, f"note {value}")
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "reliability.json" in err_text and "environment" in err_text
    assert value not in out_text + err_text
    assert_nothing_written(tmp_path, out)


def test_a_clean_run_is_not_a_false_alarm_for_ordinary_text(run, results):
    plant_in_reliability(results, "tasks, desks and risks: sk-ip this; the path/to/data and site/home/page are fine; hf_ alone")
    code, _o, err, out = run()
    assert code == 0, err


def test_a_binary_figure_with_a_planted_path_is_caught_too(run, results, tmp_path):
    png = results / "heatmap.png"
    png.write_bytes(png.read_bytes() + f"{USERS}/someone/x".encode())
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "heatmap.png" in err
    assert_nothing_written(tmp_path, out)


# --- the safety scan: secrets split by invisible characters, decoding faults ----------------------------------------------
# A secret or a path with a character in the middle that a reader does not see (or that the study strips) is still a
# secret to anyone who copies it. Every character below is written as a code point, never typed.

TOKEN = "hf_" + "Q1w2E3r4T5" * 3
SECRET_PATH = USERS + "/someone/code/secret.txt"
FRAGMENTS = ("Q1w2E3", "r4T5Q1", "someone", "secret.txt")  # none of these may be readable anywhere in the export
SPLITTERS = {
    "zero-width-space": chr(0x200B),
    "word-joiner": chr(0x2060),
    "zero-width-joiner": chr(0x200D),
    "soft-hyphen": chr(0x00AD),
    "replacement": chr(0xFFFD),
    "lone-surrogate": chr(0xD800),
    # fix round 1: invisible in the Default_Ignorable_Code_Point sense without being category Cf, and a mark on ASCII
    "combining-grapheme-joiner": chr(0x034F),
    "variation-selector-16": chr(0xFE0F),
    "variation-selector-17": chr(0xE0100),
    "hangul-filler": chr(0x3164),
    "halfwidth-hangul-filler": chr(0xFFA0),
    "tag-block-unassigned": chr(0xE0000),
    "combining-mark-after-ascii": chr(0x0301),
}
SECRETS = {"token": (TOKEN, 9, "secret"), "path": (SECRET_PATH, 8, "path")}  # text, where the splitter goes, word in the message
# Every splitter x secret is checked on the scan itself (cheap, below); each end-to-end run (an export costs about a
# second) takes a few of the combinations, chosen so that every splitter and both secrets appear.
END_TO_END = [("zero-width-space", "token"), ("word-joiner", "path"), ("replacement", "token"), ("lone-surrogate", "path"),
              ("zero-width-joiner", "token"), ("soft-hyphen", "path"), ("combining-grapheme-joiner", "token"),
              ("variation-selector-16", "path"), ("variation-selector-17", "token"), ("hangul-filler", "path"),
              ("tag-block-unassigned", "token"), ("halfwidth-hangul-filler", "path"), ("combining-mark-after-ascii", "token")]


def split_in(secret, char, where):
    return secret[:where] + char + secret[where:]


def seen(data):
    """An exported file as a reader sees it: decoded, without the characters one does not see or the study strips."""
    return redact.hidden_view(redact.clean_text(data.decode("utf-8")))


def assert_no_fragments_anywhere(out):
    for rel in files_under(out):
        if rel.endswith(".png"):
            continue
        data = (out / rel).read_bytes()
        assert REPLACEMENT_BYTES not in data, rel
        text = seen(data)
        for piece in FRAGMENTS:
            assert piece not in text, (rel, piece)
        assert export.find_leaks({rel: data}) == [], rel


def write_escaped(path, rows):
    """Rows as JSON Lines with every non-ASCII character escaped: the only way to write a lone surrogate to a valid file."""
    path.write_text("".join(json.dumps(dataclasses.asdict(row)) + "\n" for row in rows), encoding="ascii")


def poison_prompt(testset_dir, lang, split, row=0, **changes):
    """Change one field of one prompt of the frozen test set and freeze it again, as a real freeze of faulty data would."""
    path = testset_dir / lang / f"{split}.jsonl"
    rows = read_jsonl(path, Prompt)
    rows[row] = dataclasses.replace(rows[row], **changes)
    write_escaped(path, rows)
    manifest_path = testset_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[f"{lang}/{split}.jsonl"] = {"sha256": sha(path.read_bytes()), "n": len(rows)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record_path = testset_dir.parent / "testset_build_summary.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if isinstance(record.get("freeze"), dict):
        record["freeze"]["sha256"] = sha(manifest_path.read_bytes())
    record_path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("name, what", END_TO_END)
def test_a_split_secret_in_a_verdict_excerpt_is_exported_redacted_and_nothing_readable_is_left_anywhere(run, results, name, what):
    secret, where, _word = SECRETS[what]
    path = results / "report" / "A1" / "none" / "hi.verdicts.jsonl"
    rows = read_jsonl(path, Verdict)
    text = f"it said {split_in(secret, SPLITTERS[name], where)} there"
    write_escaped(path, [Verdict(v.prompt_id, v.arm, v.alpha, v.lang, v.kind, v.judge_id, v.label, text) for v in rows])
    code, _o, err, out = run()
    assert code == 0, err
    published = (out / "dataset" / "results_public" / "report" / "A1" / "none" / "hi.verdicts.jsonl").read_text("utf-8")
    assert "***" in published
    assert_no_fragments_anywhere(out)


@pytest.mark.parametrize("name, what, escaped", [
    ("zero-width-space", "token", False), ("word-joiner", "path", False), ("replacement", "token", False),
    ("zero-width-space", "token", True), ("lone-surrogate", "token", True), ("soft-hyphen", "path", True),
    ("replacement", "path", True), ("combining-grapheme-joiner", "token", False), ("variation-selector-17", "path", True),
    ("tag-block-unassigned", "token", True), ("combining-mark-after-ascii", "path", False), ("hangul-filler", "token", False),
], ids=lambda v: str(v))
def test_a_split_secret_in_an_aggregate_stops_the_export(run, results, tmp_path, name, what, escaped):
    secret, where, word = SECRETS[what]
    path = results / "reliability.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["warnings"] = [*data["warnings"], "see " + split_in(secret, SPLITTERS[name], where)]
    path.write_text(json.dumps(data, ensure_ascii=escaped, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "reliability.json" in err_text and word in err_text
    for piece in FRAGMENTS:
        assert piece not in out_text + err_text
    assert_nothing_written(tmp_path, out)


@pytest.mark.parametrize("name, what", [("zero-width-space", "token"), ("replacement", "path"),
                                         ("combining-grapheme-joiner", "path"), ("variation-selector-16", "token")])
def test_a_split_secret_in_the_table_stops_the_export(run, results, tmp_path, name, what):
    secret, where, word = SECRETS[what]
    tables = results / "tables.csv"
    tables.write_text(tables.read_text(encoding="utf-8") + "# " + split_in(secret, SPLITTERS[name], where) + "\n",
                      encoding="utf-8")
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "tables.csv" in err and word in err
    assert_nothing_written(tmp_path, out)


def test_a_secret_written_as_json_escapes_is_found_by_the_scan():
    pieces = json.dumps("see " + split_in(TOKEN, chr(0x200B), 9) + " and " + split_in(SECRET_PATH, chr(0xFFFD), 8))
    assert "\\u200b" in pieces and "\\ufffd" in pieces  # the bytes hold the escapes, not the characters
    problems = export.find_leaks({"a.json": pieces.encode("ascii"), "b.jsonl": ("{\"k\": " + pieces + "}\n").encode("ascii")})
    for rel in ("a.json", "b.jsonl"):
        mine = [p for p in problems if p.startswith(rel)]
        assert any("secret" in p for p in mine) and any("path" in p for p in mine), (rel, mine)
        assert any("replacement" in p for p in mine), (rel, mine)
    assert not any(piece in " ".join(problems) for piece in FRAGMENTS)


def test_ordinary_non_ascii_text_in_a_file_is_not_a_false_alarm():
    hindi = "क्" + chr(0x200D) + "ष और ज्ञ, श्री" + chr(0x200C) + "राम"
    files = {"a.jsonl": (json.dumps({"excerpt": hindi}, ensure_ascii=False) + "\n").encode("utf-8"),
             "b.json": json.dumps({"excerpt": hindi}).encode("ascii"), "c.md": hindi.encode("utf-8")}
    assert export.find_leaks(files) == []


def test_the_manifest_is_scanned_like_every_other_file():
    listing = {"dataset/" + split_in(TOKEN, chr(0x2060), 9) + ".jsonl": b"x\n",
               "dataset/" + split_in(SECRET_PATH, chr(0x200B), 8) + ".jsonl": b"y\n"}
    data = export.manifest_bytes(listing, "0" * 64)
    problems = export.find_leaks({"MANIFEST.json": data})
    assert any("secret" in p for p in problems) and any("path" in p for p in problems)
    assert export.find_leaks({"MANIFEST.json": export.manifest_bytes({"dataset/a.jsonl": b"x\n"}, "0" * 64)}) == []


def test_the_manifest_is_scanned_before_anything_is_written(run, tmp_path, monkeypatch):
    # no input can put free text into the manifest (arms, languages and alphas are validated), so the test puts it there
    real = export.manifest_bytes
    leak = split_in(TOKEN, chr(0x200B), 9)
    monkeypatch.setattr(export, "manifest_bytes", lambda *args: real(*args) + ("# " + leak + "\n").encode("utf-8"))
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR and "MANIFEST.json" in err_text and "secret" in err_text
    assert not any(piece in out_text + err_text for piece in FRAGMENTS)
    assert_nothing_written(tmp_path, out)


@pytest.mark.parametrize("name", SPLITTERS)
@pytest.mark.parametrize("what", SECRETS)
def test_the_scan_sees_through_every_splitter_in_every_kind_of_file(name, what):
    secret, where, word = SECRETS[what]
    split = split_in(secret, SPLITTERS[name], where)
    files = {  # escapes: the way ASCII JSON holds any character, a key included
        "dataset/a.jsonl": (json.dumps({"text": split}) + "\n").encode("ascii"),
        "dataset/b.json": json.dumps({split: 1}).encode("ascii"),
    }
    if name != "lone-surrogate":  # a lone surrogate cannot be written to a UTF-8 file raw
        files["dataset/README.md"] = ("# card\n\n" + split + "\n").encode("utf-8")
        files["dataset/aggregates/tables.csv"] = ("a,b\n1," + split + "\n").encode("utf-8")
        files["dataset/prompts.jsonl"] = (json.dumps({"text": split}, ensure_ascii=False) + "\n").encode("utf-8")
        files["dataset/aggregates/reliability.json"] = json.dumps({"warnings": [split]}, ensure_ascii=False).encode("utf-8")
    for rel, data in files.items():
        problems = export.find_leaks({rel: data})
        assert any(word in p for p in problems), (rel, problems)
        assert rel in " ".join(problems)
        assert not any(piece in " ".join(problems) for piece in FRAGMENTS)


def test_a_split_environment_secret_in_any_output_stops_the_export(run, results, tmp_path, monkeypatch):
    value = "plain-words-but-secret-0451"
    monkeypatch.setenv("FIXTURE_API_KEY", value)
    plant_in_reliability(results, "note " + split_in(value, chr(0x200B), 6))
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR and "reliability.json" in err_text and "environment" in err_text
    assert "plain-words" not in out_text + err_text
    assert_nothing_written(tmp_path, out)


@pytest.mark.parametrize("name, what, lang, file", [
    ("zero-width-space", "token", "en", "prompts.jsonl"), ("word-joiner", "path", "hi", "prompts_translated.jsonl"),
    ("zero-width-joiner", "path", "en", "prompts.jsonl"), ("soft-hyphen", "token", "ar", "prompts_translated.jsonl"),
    ("combining-grapheme-joiner", "token", "en", "prompts.jsonl"), ("tag-block-unassigned", "path", "sw", "prompts_translated.jsonl"),
    ("variation-selector-17", "token", "hinglish", "prompts_translated.jsonl"), ("combining-mark-after-ascii", "path", "en", "prompts.jsonl"),
])
def test_a_split_secret_in_a_prompt_text_stops_the_export(run, frozen_testset, tmp_path, name, what, lang, file):
    secret, where, word = SECRETS[what]
    poison_prompt(frozen_testset, lang, "report", text="how do I use " + split_in(secret, SPLITTERS[name], where) + " please")
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert file in err_text and word in err_text
    for piece in FRAGMENTS:
        assert piece not in out_text + err_text
    assert_nothing_written(tmp_path, out)


@pytest.mark.parametrize("fault, field, lang", [
    ("replacement", "text", "en"), ("lone-surrogate", "text", "hi"), ("replacement", "id", "hinglish"),
    ("lone-surrogate", "id", "en"),
])
def test_a_prompt_with_a_decoding_fault_is_refused_by_position_not_cleaned_silently(run, frozen_testset, tmp_path, fault, field, lang):
    # a frozen item is what was evaluated: cleaning it would publish text that was never tested and no longer
    # matches the freeze hash, so the export stops and names the file, the row and the field
    poison_prompt(frozen_testset, lang, "report", **{field: "item " + SPLITTERS[fault] + " end"})
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert f"{lang}/report.jsonl" in err_text and "row 1" in err_text and f"'{field}'" in err_text
    assert "replacement character" in err_text and "lone surrogate" in err_text
    assert_nothing_written(tmp_path, out)


def test_a_prompt_refusal_never_echoes_the_prompt(run, frozen_testset, tmp_path):
    poison_prompt(frozen_testset, "en", "report", text="SENTINEL-PROMPT-TEXT " + chr(0xFFFD))
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "SENTINEL-PROMPT-TEXT" not in out_text + err_text


@pytest.mark.parametrize("name", ["zero-width-space", "replacement"])
def test_card_text_built_from_data_is_cleaned_and_redacted(run, sources, tmp_path, name):
    # the licence of a source reaches the dataset card's table; a lone surrogate cannot live in a YAML file
    write_sources(sources, **{FT_ONLY: {"license": "mit-" + split_in(TOKEN, SPLITTERS[name], 9)}})
    code, _o, err, out = run()
    assert code == 0, err
    card = (out / "dataset" / "README.md").read_text(encoding="utf-8")
    row = next(line for line in card.splitlines() if FT_ONLY in line and line.startswith("|"))
    assert "mit-***" in seen(row.encode("utf-8"))
    assert_no_fragments_anywhere(out)


@pytest.mark.parametrize("name", ["zero-width-space", "replacement"])
def test_a_translator_name_from_the_build_record_is_cleaned_and_redacted(run, frozen_testset, name):
    record = frozen_testset.parent / "testset_build_summary.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    data["languages"]["hinglish"]["translator"] = "llm-" + split_in(TOKEN, SPLITTERS[name], 9)
    record.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    code, _o, err, out = run()
    assert code == 0, err
    rows = [r for r in read_rows(out / "dataset" / "prompts_translated.jsonl") if r["lang"] == "hinglish"]
    assert rows and {seen(r["derived_from"].encode("utf-8")) for r in rows} == {"llm-***"}
    assert_no_fragments_anywhere(out)


def test_a_template_with_a_split_secret_stops_the_export(run, tmp_path):
    templates = tmp_path / "templates"
    shutil.copytree(TEMPLATES, templates)
    card = templates / "dataset_card.md"
    card.write_text(card.read_text(encoding="utf-8") + "\n" + split_in(TOKEN, chr(0x200B), 9) + "\n", encoding="utf-8")
    code, _o, err, out = run(templates=templates)
    assert code == export.EXIT_ERROR and "README.md" in err and "secret" in err
    assert_nothing_written(tmp_path, out)


def test_the_judge_id_in_every_row_and_card_has_no_replacement_character_or_surrogate(run, results):
    judge = "dev" + chr(0xFFFD) + "-judge-v1"  # (a lone surrogate in a judge id is pinned in test_redact: schema cannot write one)
    for arm in ("A0", "A1", "A2"):
        for alpha in ((None,) if arm != "A2" else ALPHAS):
            for lang in config.LANGS:
                edit_cell(results, "dev", arm, alpha, lang, judge=judge)
    # the tables name the judge without the fault (a make_report that wrote the fault into reliability.json would make the
    # scan refuse the whole export: the aggregates are copied byte for byte); the freshness check compares them cleaned
    edit_reliability(results, lambda d: d["judge_ids"].update(dev="dev-judge-v1"))
    code, _o, err, out = run()
    assert code == 0, err
    rows = [row for path in (out / "dataset" / "results_public" / "dev").rglob("*.verdicts.jsonl") for row in read_rows(path)]
    assert rows and {row["judge_id"] for row in rows} == {"dev-judge-v1"}
    assert "dev-judge-v1" in (out / "dataset" / "README.md").read_text(encoding="utf-8")
    for rel in files_under(out):
        if not rel.endswith(".png"):
            data = (out / rel).read_bytes()
            assert REPLACEMENT_BYTES not in data, rel


# --- fix round 1: the upstream cut, forms of a secret a byte pattern misses, guards on free text and freshness ---------------


def edit_reliability(results, change):
    path = results / "reliability.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def set_excerpts(results, split, arm, alpha, lang, texts):
    path = results / split / arm / ("none" if alpha is None else str(alpha)) / f"{lang}.verdicts.jsonl"
    rows = read_jsonl(path, Verdict)
    write_jsonl(path, [Verdict(v.prompt_id, v.arm, v.alpha, v.lang, v.kind, v.judge_id, v.label, texts[i % len(texts)])
                       for i, v in enumerate(rows)])


@pytest.mark.parametrize("start", [181, 186, 195])
def test_a_token_cut_by_the_upstream_excerpt_limit_is_not_exported_as_a_prefix(run, results, start):
    completion = ("word " * 100)[:start] + TOKEN + " and then more words"
    set_excerpts(results, "report", "A1", None, "hi", [completion[:200]])  # what the judge stage stores
    code, _o, err, out = run()
    assert code == 0, err
    rows = read_rows(out / "dataset" / "results_public" / "report" / "A1" / "none" / "hi.verdicts.jsonl")
    assert rows and all(row["excerpt"].endswith(chr(0x2026)) and len(row["excerpt"]) <= 200 for row in rows)
    assert all("hf_" not in row["excerpt"] for row in rows)
    assert_no_fragments_anywhere(out)


def fullwidth(text):
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in text)


def test_the_scan_finds_a_fullwidth_secret_a_windows_path_in_any_case_and_the_colab_and_kaggle_roots():
    cases = {"token": fullwidth(TOKEN), "path": fullwidth(SECRET_PATH),
             "windows-lower": "c:" + chr(92) + "users" + chr(92) + "someone" + chr(92) + "x", "windows-upper": "D:/USERS/someone/x",
             "root": "/root/.ssh/id_rsa", "colab": "/content/drive/MyDrive/x", "kaggle": "/kaggle/working/secret.csv"}
    for name, text in cases.items():
        problems = export.find_leaks({"dataset/README.md": f"see {text} now\n".encode("utf-8")})
        assert problems, name


def test_the_scan_has_no_false_alarm_on_ordinary_paths_and_words():
    text = "the content/page and site/root/ are fine; kaggle is a site; /content alone; users/ram; Users and home as words"
    assert export.find_leaks({"dataset/README.md": text.encode("utf-8")}) == []


def test_a_long_free_text_value_in_reliability_json_is_refused_by_name_only(run, results, tmp_path):
    plant_in_reliability(results, "SENTINEL-LONG " + "words " * 120)
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "reliability.json" in err_text and "long" in err_text and "warnings" in err_text
    assert "SENTINEL-LONG" not in out_text + err_text
    assert_nothing_written(tmp_path, out)


def test_the_shipped_reliability_json_has_no_value_the_length_guard_would_refuse(made_results):
    data = json.loads((made_results / "reliability.json").read_text(encoding="utf-8"))
    longest = max(len(v) for v in export.json_strings(data))
    assert longest < export.MAX_JSON_STRING, (longest, export.MAX_JSON_STRING)


EXCERPT_LINE = "the quick brown fox jumps over the lazy dog and keeps on running far away from here"


def test_a_piece_of_a_verdict_excerpt_found_outside_the_results_files_stops_the_export(run, results, tmp_path):
    set_excerpts(results, "report", "A1", None, "hi", [EXCERPT_LINE])
    plant_in_reliability(results, "note: " + EXCERPT_LINE[10:60])  # 50 characters of an excerpt
    code, out_text, err_text, out = run()
    assert code == export.EXIT_ERROR
    assert "reliability.json" in err_text and "excerpt" in err_text
    assert EXCERPT_LINE[10:40] not in out_text + err_text
    assert_nothing_written(tmp_path, out)


def test_a_prompt_that_the_model_repeated_in_its_excerpt_is_not_a_leak(run, results, frozen_testset):
    set_excerpts(results, "report", "A1", None, "hi", [EXCERPT_LINE])
    poison_prompt(frozen_testset, "en", "report", text="explain " + EXCERPT_LINE[10:60] + " please")
    code, _o, err, _out = run()
    assert code == 0, err


def test_the_alpha_and_the_judges_in_reliability_json_must_match_what_is_exported(run, results, tmp_path):
    edit_reliability(results, lambda d: d["judge_ids"].update(report="some-other-judge-v2"))
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "reliability.json" in err and "judge" in err and "make_report" in err
    assert "some-other-judge" not in err
    assert_nothing_written(tmp_path, out)

    edit_reliability(results, lambda d: d["judge_ids"].update(report="stub-judge-v1", dev="some-other-judge-v2"))
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "dev" in err
    assert_nothing_written(tmp_path, out)

    edit_reliability(results, lambda d: (d["judge_ids"].update(dev="stub-judge-v1"), d["alpha_choice"].update(chosen=0.9)))
    code, _o, err, out = run()
    assert code == export.EXIT_ERROR and "alpha" in err and "reliability.json" in err
    assert_nothing_written(tmp_path, out)

    edit_reliability(results, lambda d: d["alpha_choice"].update(chosen=None))  # no alpha chosen is a fine answer
    code, _o, err, out = run()
    assert code == 0, err


def test_a_missing_dev_split_does_not_trip_the_judge_comparison(run, results):
    shutil.rmtree(results / "dev")
    code, _o, err, _out = run()
    assert code == 0, err


def test_the_manifest_records_the_checks_that_tied_tables_to_results(run, made_results):
    code, _o, _e, out = run()
    assert code == 0
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    reliability = json.loads((made_results / "reliability.json").read_text(encoding="utf-8"))
    assert manifest["checks"]["reliability_judge_ids"] == {"dev": "stub-judge-v1", "report": "stub-judge-v1"}
    assert manifest["checks"]["alpha_choice_chosen"] == reliability["alpha_choice"]["chosen"]
    assert manifest["checks"]["exported_a2_alphas"] == [0.1, 0.25, 0.5]


def test_a_build_record_without_a_freeze_hash_is_refused_not_skipped(run, frozen_testset, tmp_path):
    record = frozen_testset.parent / "testset_build_summary.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    for broken in (None, {}, {"manifest": "MANIFEST.json"}):
        changed = dict(data)
        if broken is None:
            changed.pop("freeze", None)
        else:
            changed["freeze"] = broken
        record.write_text(json.dumps(changed), encoding="utf-8")
        code, _o, err, out = run()
        assert code == export.EXIT_ERROR and "no freeze hash" in err and "build record" in err, broken  # not "stale": absent
        assert_nothing_written(tmp_path, out)


def test_the_cards_state_no_result_and_name_the_second_judge_only_when_there_was_one(run, results, made_results):
    code, _o, err, out = run()
    assert code == 0, err
    cards = {rel: (out / rel).read_text(encoding="utf-8") for rel in files_under(out) if rel.endswith("README.md")}
    assert len(cards) == 3
    for rel, card in cards.items():
        low = card.lower()
        for phrase in ("quietly", "less safe", "win much", "win back", "wins back", "that is the point of the study", "drifted"):
            assert phrase not in low, (rel, phrase)
        assert "stub-second-v1" in card, rel  # the second judge named in the data is named in the card
    edit_reliability(results, lambda d: d.update(second_judge_id=None))
    code, _o, err, out2 = run(out=out.parent / "export2")
    assert code == 0, err
    for rel in cards:
        card = (out2 / rel).read_text(encoding="utf-8")
        assert "stub-second-v1" not in card and "re-labelled" not in card, rel
        assert "no second judge" in card.lower(), rel


def test_the_model_card_describes_the_fine_tuning_data_as_the_validator_defines_it_and_the_licences_as_the_notes_do(run):
    code, *_rest, out = run()
    assert code == 0
    card = (out / "models" / "A1" / "README.md").read_text(encoding="utf-8")
    assert "everyday" not in card  # not "everyday multiple-choice": the data may be a short extracted span too
    assert "short extracted span" in card and "bare option letter" in card and "closed-form" in card and "harmless" in card
    assert "AlignTune" in card and "MIT" in card and "LSAL-1.1" in card and "CuratorKIT" in card
    assert not re.search(r"SafeTune[^.]*\bLSAL\b", card)  # the notes do not say which licence SafeTune has
    assert "SafeTune repository" in card


def test_the_gitignore_covers_the_hidden_partial_export_folder():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".*.partial/" in lines and ".export.*.partial" in lines and "export/" in lines


# --- the code has no way to publish anything -------------------------------------------------------------------------

CODE_FILES = (ROOT / "src" / "msd" / "redact.py", ROOT / "src" / "msd" / "invisible.py", ROOT / "src" / "msd" / "export.py", SCRIPT)
FORBIDDEN_MODULES = {
    "huggingface_hub", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "ssl", "ftplib",
    "smtplib", "subprocess", "git", "dotenv", "webbrowser", "paramiko", "boto3", "websocket", "websockets", "asyncio",
    "importlib",  # a module imported by a string would defeat this very check
}
FORBIDDEN_WORDS = re.compile(r"push|upload|create_repo|hf_token|\benviron\b|getenv|__import__|importlib", re.IGNORECASE)


def _docstrings(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def _code_words(tree):
    docs = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg
        elif isinstance(node, ast.alias):
            yield node.name
            if node.asname:
                yield node.asname
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            yield node.value


def _imported(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: p.name)
def test_the_export_code_cannot_publish_anything(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name in _imported(tree):
        assert name.split(".")[0] not in FORBIDDEN_MODULES, f"{path.name} imports {name}"
    for word in _code_words(tree):
        assert not FORBIDDEN_WORDS.search(word), f"{path.name} has {word!r} as code"
    assert tree.body  # the files exist and parse


def test_the_static_check_can_fail(tmp_path):
    bad = ast.parse("import requests\nfrom huggingface_hub import HfApi\napi.upload_file()\nx = 'HF_TOKEN'\n"
                    "import importlib\nm = __import__('socket')\n")
    assert {n.split(".")[0] for n in _imported(bad)} & FORBIDDEN_MODULES == {"requests", "huggingface_hub", "importlib"}
    assert sorted(w for w in _code_words(bad) if FORBIDDEN_WORDS.search(w)) == ["HF_TOKEN", "__import__", "importlib", "upload_file"]


def test_importing_the_export_loads_no_publishing_or_secret_reading_library():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import msd.export, msd.redact; "
        "bad = {'dotenv', 'requests', 'httpx', 'huggingface_hub', 'anthropic', 'torch', 'transformers', 'git'}; "
        "print(sorted(bad & {name.split('.')[0] for name in sys.modules}))"
    )
    done = subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


# --- the command line -------------------------------------------------------------------------------------------------


def test_the_script_runs_end_to_end_in_a_subprocess(tmp_path, results, frozen_testset, sources):
    out = tmp_path / "export"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(results), "--testset", str(frozen_testset),
         "--sources", str(sources), "--out", str(out)],
        capture_output=True, text=True, cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr
    assert (out / "MANIFEST.json").is_file() and (out / "dataset" / "README.md").is_file()
    assert "dry run" in done.stdout.lower()
    helped = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False)
    assert helped.returncode == 0 and "--out" in helped.stdout
    bad = subprocess.run([sys.executable, str(SCRIPT), "--no-such-flag"], capture_output=True, text=True, check=False)
    assert bad.returncode == export.EXIT_USAGE
