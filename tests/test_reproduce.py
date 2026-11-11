"""msd.reproduce: the headline table recomputed from the PUBLIC export (Task 20).

A synthetic export is built end to end by the real code (tests/export_tree.py: frozen test set, results tree,
make_report, export_public), then the table is rebuilt from the public rows alone and compared with what the original
run wrote. Nothing here touches a model, a judge, the network or the repository's real folders.

What may be recomputed and what has to be taken as recorded is the point of the module: the rates, intervals, paired
changes and counts come from the public rows with the same code the report uses; which cells were left out, which
languages are low-confidence and which alpha was chosen depend on data the export does not hold (eval.json files,
completions, the second judge) and come from the exported reliability.json.
"""

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from msd import config, reportrun, reproduce

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ["default", "exclusions", "no_alpha", "no_second_judge", "noisy_second_judge"]
REPORT_ROWS = "dataset/results_public/report"


@pytest.fixture
def export_copy(tmp_path, public_exports):
    """`export_copy(scenario)` -> a private copy of the shared export that a test may change."""

    def make(scenario="default"):
        dest = tmp_path / f"export-{scenario}"
        shutil.copytree(public_exports(scenario).export, dest)
        return dest

    return make


def rows_of(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def reliability_path(export):
    return export / "dataset" / "aggregates" / "reliability.json"


def change_reliability(export, change):
    path = reliability_path(export)
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- the numbers come from the public rows, and equal the original run's ---------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_table_recomputed_from_public_rows_equals_the_original_runs_table(scenario, public_exports):
    built = public_exports(scenario)
    original = reportrun.analyze(built.results)
    public = reproduce.table_from_public(built.export)
    assert original.table, "the fixture must give the comparison something to compare"
    assert public.rows == original.table  # every rate, interval, n, excluded count, status and low_confidence flag
    assert public.deltas == original.deltas
    assert public.heat == original.heat
    assert public.chosen_alpha == original.choice.chosen
    assert public.heat_excluded == original.heat_excluded
    assert public.curve_excluded == original.curve_excluded
    assert public.not_chosen == original.not_chosen
    assert public.task_not_applicable == original.task_not_applicable
    assert set(public.low_confidence) == set(original.low_confidence)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_tables_csv_is_byte_identical_to_the_one_the_report_wrote(scenario, public_exports, tmp_path):
    built = public_exports(scenario)
    written = reproduce.table_from_public(built.export).write_tables(tmp_path / "recomputed" / "tables.csv")
    exported = built.export / "dataset" / "aggregates" / "tables.csv"
    assert written.read_bytes() == exported.read_bytes() == (built.results / "tables.csv").read_bytes()
    assert reproduce.sha256_file(written) == reproduce.manifest_tables_sha256(built.export)


def test_the_scenarios_really_differ_so_the_equality_above_is_not_vacuous(public_exports):
    tables = {name: reproduce.table_from_public(public_exports(name).export) for name in SCENARIOS}
    assert tables["default"].chosen_alpha == 0.5 and not tables["default"].curve_excluded
    assert tables["exclusions"].curve_excluded and tables["exclusions"].heat_excluded == {("hi", "A2"): "judge errors"}
    assert tables["no_alpha"].chosen_alpha is None and tables["no_alpha"].heat["before"]["arm"] == "A0"
    assert tables["no_second_judge"].low_confidence and tables["noisy_second_judge"].low_confidence
    assert not tables["default"].low_confidence
    assert any(r["status"] == "not_applicable" for r in tables["default"].rows)  # hinglish has no task items
    assert tables["default"].deltas and set(tables["default"].deltas) <= set(config.LANGS)


def test_a_changed_public_row_changes_the_recomputed_table_so_numbers_are_not_copied(export_copy, tmp_path):
    export = export_copy()
    path = export / REPORT_ROWS / "A1" / "none" / "hi.verdicts.jsonl"
    rows = rows_of(path)
    rows[0]["label"] = "complied" if rows[0]["label"] == "refused" else "refused"
    write_rows(path, rows)
    recomputed = reproduce.table_from_public(export).write_tables(tmp_path / "out" / "tables.csv")
    exported = export / "dataset" / "aggregates" / "tables.csv"
    assert recomputed.read_bytes() != exported.read_bytes()
    lines = reproduce.explain_difference(recomputed, exported)
    assert lines and all("arm=A1" in line and "lang=hi" in line for line in lines)  # only the changed cell differs


def test_decisions_that_need_non_public_data_are_taken_from_reliability_json(export_copy, tmp_path):
    export = export_copy("exclusions")
    original = reproduce.table_from_public(export)
    assert not [r for r in original.rows if (r["arm"], r["alpha"], r["lang"], r["metric"]) == ("A2", 0.5, "hi", "asr")]

    def forget_the_exclusion(data):
        data["excluded_cells"] = [e for e in data["excluded_cells"] if e["reason"] != "judge_error"]

    change_reliability(export, forget_the_exclusion)
    changed = reproduce.table_from_public(export)
    assert [r for r in changed.rows if (r["arm"], r["alpha"], r["lang"], r["metric"]) == ("A2", 0.5, "hi", "asr")]

    change_reliability(export, lambda data: data["low_confidence"].update({"en": ["not_assessed"]}))
    flagged = reproduce.table_from_public(export)
    assert {r["low_confidence"] for r in flagged.rows if r["lang"] == "en"} == {True}
    assert {r["low_confidence"] for r in original.rows if r["lang"] == "en"} == {False}


# --- plain refusals --------------------------------------------------------------------------------------------------


def test_a_missing_reliability_json_is_refused_and_says_what_it_is_for(export_copy):
    export = export_copy()
    reliability_path(export).unlink()
    with pytest.raises(reproduce.ReproduceError, match=r"reliability\.json.*dataset/aggregates"):
        reproduce.table_from_public(export)


def test_a_reliability_json_that_is_not_json_is_refused(export_copy):
    export = export_copy()
    reliability_path(export).write_text("{ not json", encoding="utf-8")
    with pytest.raises(reproduce.ReproduceError, match=r"reliability\.json.*not valid JSON"):
        reproduce.table_from_public(export)


@pytest.mark.parametrize("key", ["excluded_cells", "alpha_choice", "low_confidence", "judge_ids", "heatmap"])
def test_a_reliability_json_without_a_recorded_decision_is_refused_naming_it(export_copy, key):
    export = export_copy()
    change_reliability(export, lambda data: data.pop(key))
    with pytest.raises(reproduce.ReproduceError, match=key):
        reproduce.table_from_public(export)


def test_a_reliability_json_of_other_results_is_refused_when_the_judge_differs(export_copy):
    export = export_copy()
    change_reliability(export, lambda data: data["judge_ids"].update({"report": "some-other-judge-v9"}))
    with pytest.raises(reproduce.ReproduceError, match=r"judge.*some-other-judge-v9|some-other-judge-v9.*judge"):
        reproduce.table_from_public(export)


def test_a_chosen_alpha_that_is_not_in_the_public_rows_is_refused(export_copy):
    export = export_copy()
    change_reliability(export, lambda data: data["alpha_choice"].update({"chosen": 0.35}))
    with pytest.raises(reproduce.ReproduceError, match=r"alpha 0\.35"):
        reproduce.table_from_public(export)


def test_a_heatmap_that_compares_arms_with_no_public_rows_is_refused(export_copy):
    export = export_copy()
    change_reliability(export, lambda data: data["heatmap"]["after"].update({"arm": "A3", "alpha": None}))
    with pytest.raises(reproduce.ReproduceError, match=r"heatmap.*A3"):
        reproduce.table_from_public(export)


def test_prompts_without_a_kind_for_the_verdicts_are_refused_and_the_file_is_named(export_copy):
    export = export_copy()
    (export / "dataset" / "prompts_translated.jsonl").unlink()  # the optional file: its prompts carry the kinds of hi, ar, sw
    with pytest.raises(reproduce.ReproduceError, match=r"prompts_translated\.jsonl") as caught:
        reproduce.table_from_public(export)
    assert "kind" in str(caught.value) and "harmful" in str(caught.value)


def test_a_prompt_whose_kind_disagrees_across_files_is_refused(export_copy):
    export = export_copy()
    path = export / "dataset" / "prompts.jsonl"
    rows = rows_of(path)
    clone = {**rows[0], "kind": "benign_edgy" if rows[0]["kind"] == "harmful" else "harmful"}
    write_rows(path, [*rows, clone])
    with pytest.raises(reproduce.ReproduceError, match=r"two kinds|more than one kind"):
        reproduce.table_from_public(export)


def test_a_missing_report_split_is_refused(export_copy):
    export = export_copy()
    shutil.rmtree(export / REPORT_ROWS)
    with pytest.raises(reproduce.ReproduceError, match=r"report"):
        reproduce.table_from_public(export)


def test_a_verdict_row_that_is_not_a_public_row_is_refused_with_file_and_line(export_copy):
    export = export_copy()
    path = export / REPORT_ROWS / "A0" / "none" / "en.verdicts.jsonl"
    rows = rows_of(path)
    del rows[2]["label"]
    write_rows(path, rows)
    with pytest.raises(reproduce.ReproduceError, match=r"en\.verdicts\.jsonl.*line 3") as caught:
        reproduce.table_from_public(export)
    assert "label" in str(caught.value)


def test_a_label_the_study_does_not_use_is_refused_with_file_and_line(export_copy):
    export = export_copy()
    path = export / REPORT_ROWS / "A2" / "0.5" / "sw.verdicts.jsonl"
    rows = rows_of(path)
    rows[4]["label"] = "maybe"
    write_rows(path, rows)
    with pytest.raises(reproduce.ReproduceError, match=r"sw\.verdicts\.jsonl line 5.*label"):
        reproduce.table_from_public(export)


def test_rows_in_the_wrong_folder_are_refused(export_copy):
    export = export_copy()
    src = export / REPORT_ROWS / "A0" / "none" / "en.verdicts.jsonl"
    shutil.copy(src, export / REPORT_ROWS / "A0" / "none" / "hi.verdicts.jsonl")  # a copied file would change the table
    with pytest.raises(reproduce.ReproduceError, match=r"hi\.verdicts\.jsonl"):
        reproduce.table_from_public(export)


def test_a_repeated_verdict_is_refused_rather_than_counted_twice(export_copy):
    export = export_copy()
    path = export / REPORT_ROWS / "A1" / "none" / "ar.verdicts.jsonl"
    rows = rows_of(path)
    write_rows(path, [*rows, rows[0]])
    with pytest.raises(reproduce.ReproduceError, match=r"twice|duplicate"):
        reproduce.table_from_public(export)


def test_an_export_folder_that_is_not_one_is_refused_plainly(tmp_path):
    with pytest.raises(reproduce.ReproduceError, match=r"export"):
        reproduce.table_from_public(tmp_path / "nowhere")


# --- the manifest ----------------------------------------------------------------------------------------------------


def test_an_intact_export_has_no_manifest_problems(public_exports):
    assert reproduce.verify_manifest(public_exports().export) == []


def test_a_changed_file_is_named_by_the_manifest_check(export_copy):
    export = export_copy()
    tables = export / "dataset" / "aggregates" / "tables.csv"
    tables.write_bytes(tables.read_bytes().replace(b"0.050000", b"0.060000", 1))
    problems = reproduce.verify_manifest(export)
    assert len(problems) == 1 and "dataset/aggregates/tables.csv" in problems[0] and "changed" in problems[0]


def test_a_missing_file_is_named_by_the_manifest_check(export_copy):
    export = export_copy()
    (export / REPORT_ROWS / "A1" / "none" / "hi.tasks.jsonl").unlink()
    problems = reproduce.verify_manifest(export)
    assert len(problems) == 1 and "hi.tasks.jsonl" in problems[0] and "missing" in problems[0]


def test_a_manifest_that_points_outside_the_export_is_refused(export_copy):
    export = export_copy()
    manifest = json.loads((export / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files"]["../outside.txt"] = {"sha256": "0" * 64}
    (export / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(reproduce.ReproduceError, match=r"outside"):
        reproduce.verify_manifest(export)


def test_no_manifest_is_refused_with_a_plain_message(export_copy):
    export = export_copy()
    (export / "MANIFEST.json").unlink()
    with pytest.raises(reproduce.ReproduceError, match=r"MANIFEST\.json"):
        reproduce.verify_manifest(export)
    with pytest.raises(reproduce.ReproduceError, match=r"MANIFEST\.json"):
        reproduce.manifest_tables_sha256(export)


# --- explaining a difference -----------------------------------------------------------------------------------------


def test_explain_difference_is_empty_for_identical_files_and_names_the_cell_and_column_otherwise(public_exports, tmp_path):
    exported = public_exports().export / "dataset" / "aggregates" / "tables.csv"
    assert reproduce.explain_difference(exported, exported) == []
    text = exported.read_text(encoding="utf-8").replace("A0,,en,asr,0.050000", "A0,,en,asr,0.075000", 1)
    other = tmp_path / "other.csv"
    other.write_text(text, encoding="utf-8")
    lines = reproduce.explain_difference(other, exported)
    assert len(lines) == 1
    for word in ("arm=A0", "lang=en", "metric=asr", "rate", "0.075000", "0.050000"):
        assert word in lines[0], word


def test_explain_difference_names_rows_only_one_side_has_and_pure_formatting_changes(public_exports, tmp_path):
    exported = public_exports().export / "dataset" / "aggregates" / "tables.csv"
    lines = exported.read_text(encoding="utf-8").splitlines(keepends=True)
    shorter = tmp_path / "shorter.csv"
    shorter.write_text("".join(lines[:-1]), encoding="utf-8")
    missing = reproduce.explain_difference(shorter, exported)
    assert len(missing) == 1 and "only in the export" in missing[0]
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(exported.read_bytes().replace(b"\n", b"\r\n"))
    formatting = reproduce.explain_difference(crlf, exported)
    assert len(formatting) == 1 and "formatting" in formatting[0]


def test_explain_difference_stops_after_a_few_lines_and_says_how_many_more(public_exports, tmp_path):
    exported = public_exports().export / "dataset" / "aggregates" / "tables.csv"
    other = tmp_path / "all-wrong.csv"
    other.write_text(exported.read_text(encoding="utf-8").replace(",0.", ",1."), encoding="utf-8")
    lines = reproduce.explain_difference(other, exported, limit=5)
    assert len(lines) == 6 and "more" in lines[-1]


# --- the figures: what is plotted --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_plotted_numbers_of_both_figures_match_the_exported_table(scenario, public_exports, tmp_path):
    built = public_exports(scenario)
    public = reproduce.table_from_public(built.export)
    recomputed = reproduce.read_table_csv(public.write_tables(tmp_path / "tables.csv"))
    exported = reproduce.read_table_csv(built.export / "dataset" / "aggregates" / "tables.csv")
    ours = reproduce.figure_numbers(recomputed, public.heat, public.chosen_alpha)
    theirs = reproduce.figure_numbers(exported, public.heat, public.chosen_alpha)
    assert ours == theirs
    assert ours["heatmap"] and ours["alpha_curve"]


def test_figure_numbers_name_exactly_what_each_figure_draws(public_exports):
    public = reproduce.table_from_public(public_exports().export)
    numbers = reproduce.figure_numbers(public.rows, public.heat, public.chosen_alpha)
    heat = numbers["heatmap"]
    assert {(arm, alpha) for _lang, arm, alpha, *_rest in heat} == {("A1", None), ("A2", 0.5)}
    assert {lang for lang, *_rest in heat} == set(config.LANGS)
    curve = numbers["alpha_curve"]
    assert {alpha for _lang, arm, alpha, *_rest in curve if arm == "A2"} == {0.1, 0.25, 0.5}
    assert "hinglish" not in {lang for lang, *_rest in curve}  # no task items: nothing to put on the x axis
    row = next(r for r in public.rows if (r["arm"], r["alpha"], r["lang"], r["metric"]) == ("A2", 0.5, "hi", "asr"))
    assert ("hi", "A2", 0.5, row["rate"], row["lo"], row["hi"], row["n"]) in heat


def test_the_figures_are_drawn_from_the_recomputed_table(public_exports, tmp_path):
    public = reproduce.table_from_public(public_exports().export)
    written = public.write_figures(tmp_path / "figs")
    assert [p.name for p in written] == ["heatmap.png", "alpha_curve.png"]
    for path in written:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_no_heatmap_is_drawn_when_the_report_compared_nothing(public_exports, tmp_path):
    public = reproduce.table_from_public(public_exports().export)
    written = dataclasses.replace(public, heat=None, deltas={}).write_figures(tmp_path / "figs")
    assert [p.name for p in written] == ["alpha_curve.png"]


def test_the_layout_it_reads_is_the_layout_the_exporter_writes():
    """reproduce keeps its own copy of the paths (importing msd.export would need pyyaml); this is the pin."""
    from msd import export

    assert reproduce.RESULTS_ROOT == export.RESULTS_ROOT and reproduce.AGGREGATES_ROOT == export.AGGREGATES_ROOT
    assert reproduce.PROMPT_FILES == (export.PROMPTS_FILE, export.TRANSLATED_FILE)
    assert reproduce.MANIFEST_NAME == export.MANIFEST_NAME
    assert reproduce.TABLES_FILE == f"{export.AGGREGATES_ROOT}/tables.csv"
    assert reproduce.RELIABILITY_FILE == f"{export.AGGREGATES_ROOT}/reliability.json"


# --- the optional GPU section's two helpers ---------------------------------------------------------------------------------


def test_the_published_prompts_of_one_language_are_rebuilt_in_the_layout_the_scripts_read(public_exports, tmp_path):
    from msd.schema import Prompt, read_jsonl

    export = public_exports().export
    folder = reproduce.rebuild_testset(export, tmp_path / "testset", lang="hi", limit=20)
    assert folder == tmp_path / "testset" and sorted(p.name for p in folder.iterdir()) == ["hi"]
    rebuilt = read_jsonl(folder / "hi" / "report.jsonl", Prompt)
    published = [r for r in rows_of(export / "dataset" / "prompts_translated.jsonl") if r["lang"] == "hi" and r["split"] == "report"]
    assert len(rebuilt) == 20 and [p.id for p in rebuilt] == [r["id"] for r in published[:20]]
    assert {p.license for p in rebuilt} <= {"mit", "cc-by-4.0"}  # the source's licence, as in the frozen test set
    assert {(p.split, p.lang) for p in rebuilt} == {("report", "hi")} and all(p.text and p.source for p in rebuilt)
    whole = reproduce.rebuild_testset(export, tmp_path / "whole", lang="en")
    assert len(read_jsonl(whole / "en" / "report.jsonl", Prompt)) == 52


def test_rebuilding_for_a_language_the_export_has_no_prompts_for_is_refused(public_exports, tmp_path):
    with pytest.raises(reproduce.ReproduceError, match="no report prompts for language 'xx'"):
        reproduce.rebuild_testset(public_exports().export, tmp_path / "t", lang="xx")


def test_a_fresh_run_is_counted_against_the_published_labels(public_exports, tmp_path):
    from msd.schema import Verdict, write_jsonl

    export = public_exports().export
    published = rows_of(export / REPORT_ROWS / "A1" / "none" / "ar.verdicts.jsonl")
    flip = {"complied": "refused", "refused": "complied"}
    fresh = [Verdict(r["prompt_id"], "A1", None, "ar", "harmful", "fresh-judge", flip.get(r["label"], r["label"]) if i < 5 else r["label"], "x")
             for i, r in enumerate(published[:12])]
    fresh.append(Verdict("not-a-published-id", "A1", None, "ar", "harmful", "fresh-judge", "refused", "x"))
    write_jsonl(tmp_path / "fresh.jsonl", fresh)
    counted = reproduce.compare_with_published(export, tmp_path / "fresh.jsonl", arm="A1", alpha=None, lang="ar")
    assert counted == {"compared": 12, "same": 7, "only_fresh": 1, "only_published": len(published) - 12}


def test_comparing_with_a_cell_that_was_not_published_is_refused(public_exports, tmp_path):
    with pytest.raises(reproduce.ReproduceError, match=r"no published labels for A3"):
        reproduce.compare_with_published(public_exports().export, tmp_path / "x.jsonl", arm="A3", alpha=None, lang="hi")


# --- it runs where only the study's light dependencies are installed -----------------------------------------------


def test_the_module_needs_no_model_stack_or_api_client():
    """A stranger installs numpy and matplotlib (pyproject's `dependencies` minus the ones this does not use); the model
    stack, API clients, pandas, yaml and dotenv must not be needed."""
    code = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'anthropic', 'safetensors', 'huggingface_hub', 'peft', 'datasets', 'pandas',\n"
        "             'yaml', 'dotenv'):\n"
        "    sys.modules[name] = None  # an import of any of these raises ImportError\n"
        "import msd.reproduce\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""}, cwd=ROOT)
    assert done.returncode == 0, done.stderr
