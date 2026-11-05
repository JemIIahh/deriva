"""Loaders, source rules, dedupe, split assignment, registries and the build summary."""

import json
import os
import random
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from msd import testset
from msd.config import NON_ENGLISH
from msd.schema import Prompt, read_jsonl
from msd.sources import load_sources
from msd.splits import norm_text
from msd.testset import (
    ForbiddenSourceError,
    PoolItem,
    RawItem,
    SourceSpec,
    TranslatorConfigError,
    UnknownNameError,
)
from msd.translate import translated_id

from conftest import BENIGN, FIXTURE_NAMES, HARM_A, HARM_B, LICENSES, TEST_SIZES, write_csv_file, write_jsonl_file

REAL_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "sources.yaml"


_write_csv = write_csv_file
_write_jsonl = write_jsonl_file


def _file_for(name):
    """The (only) local file name a real source's loader reads."""
    return next(iter(testset.SOURCES[name].files))


# --- loaders, on tiny local fixture files -------------------------------------------------


def test_load_multijail_reads_english_column_and_survives_a_bom(tmp_path):
    name = "DAMO-NLP-SG/MultiJail"
    _write_csv(
        tmp_path / _file_for(name),
        ["id", "source", "tags", "en", "zh", "ar", "sw"],
        [["0", "s", "[]", "english one", "x", "y", "z"], ["1", "s", "[]", "  ", "x", "y", "z"], ["2", "s", "[]", "english two", "", "", ""]],
        bom=True,
    )
    items = testset.SOURCES[name].load(tmp_path)
    assert items == [RawItem("0", "english one"), RawItem("1", "  "), RawItem("2", "english two")]


def test_load_harmbench_standard_only_by_default_contextual_on_request(tmp_path):
    _write_csv(
        tmp_path / "harmbench_behaviors_text_all.csv",
        ["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"],
        [
            ["std behavior", "standard", "c", "", "", "std_1"],
            ["ctx behavior", "contextual", "c", "context", "some context", "ctx_1"],
            ["copy behavior", "copyright", "c", "hash_check", "", "copy_1"],
        ],
    )
    assert testset.load_harmbench(tmp_path) == [RawItem("std_1", "std behavior")]
    both = testset.load_harmbench(tmp_path, categories=("standard", "contextual"))
    assert both == [RawItem("std_1", "std behavior"), RawItem("ctx_1", "some context\n\n---\n\nctx behavior")]
    # copyright rows are never loadable: they ask for copyrighted text, not harm
    with pytest.raises(ValueError, match="copyright"):
        testset.load_harmbench(tmp_path, categories=("copyright",))


def test_load_advbench_uses_goal_column_with_row_number_keys(tmp_path):
    name = "llm-attacks/llm-attacks (AdvBench)"
    _write_csv(tmp_path / _file_for(name), ["goal", "target"], [["goal one", "t"], ["goal two", "t"]])
    assert testset.SOURCES[name].load(tmp_path) == [RawItem("0000", "goal one"), RawItem("0001", "goal two")]


def test_load_aya_redteaming_keeps_english_rows_only(tmp_path):
    name = "CohereLabs/aya_redteaming"
    _write_jsonl(
        tmp_path / _file_for(name),
        [
            {"prompt": "english a", "language": "English", "harm_category": "x"},
            {"prompt": "hindi b", "language": "Hindi", "harm_category": "x"},
            {"prompt": "english c", "language": "English", "harm_category": "x"},
        ],
    )
    # the key is the row number in the file, so skipped rows leave a gap
    assert testset.SOURCES[name].load(tmp_path) == [RawItem("0000", "english a"), RawItem("0002", "english c")]


def test_load_nemotron_keeps_unsafe_generic_unredacted_prompts(tmp_path):
    name = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
    base = {"response": "r", "violated_categories": "x", "language": "en"}
    _write_jsonl(
        tmp_path / _file_for(name),
        [
            {**base, "id": "u1", "prompt": "unsafe generic", "prompt_label": "unsafe", "tag": "generic"},
            {**base, "id": "s1", "prompt": "safe generic", "prompt_label": "safe", "tag": "generic"},
            {**base, "id": "j1", "prompt": "unsafe jailbreak", "prompt_label": "unsafe", "tag": "jailbreaking"},
            {**base, "id": "r1", "prompt": "text [REDACTED] here", "prompt_label": "unsafe", "tag": "generic"},
            {**base, "id": "u2", "prompt": "unsafe two", "prompt_label": "unsafe", "tag": "generic"},
        ],
    )
    assert testset.SOURCES[name].load(tmp_path) == [RawItem("u1", "unsafe generic"), RawItem("u2", "unsafe two")]


def test_load_xstest_keeps_safe_prompts_only(tmp_path):
    name = "Paul/XSTest"
    _write_csv(
        tmp_path / _file_for(name),
        ["id", "prompt", "type", "label", "focus", "note"],
        [["1", "safe prompt", "homonyms", "safe", "f", ""], ["2", "unsafe prompt", "contrast_homonyms", "unsafe", "f", ""]],
    )
    assert testset.SOURCES[name].load(tmp_path) == [RawItem("1", "safe prompt")]


def test_load_orbench_hard_reads_prompt_column(tmp_path):
    name = "bench-llm/or-bench"
    _write_csv(tmp_path / _file_for(name), ["prompt", "category"], [["p one", "illegal"], ["p two", "privacy"]])
    assert testset.SOURCES[name].load(tmp_path) == [RawItem("0000", "p one"), RawItem("0001", "p two")]


def test_loader_names_the_missing_column(tmp_path):
    name = "llm-attacks/llm-attacks (AdvBench)"
    _write_csv(tmp_path / _file_for(name), ["wrong", "target"], [["x", "y"]])
    with pytest.raises(ValueError, match="goal"):
        testset.SOURCES[name].load(tmp_path)


def test_missing_raw_file_error_gives_the_download_url(tmp_path):
    name = "Paul/XSTest"
    with pytest.raises(FileNotFoundError) as exc:
        testset.SOURCES[name].load(tmp_path)
    message = str(exc.value)
    assert "xstest_prompts.csv" in message and "https://huggingface.co/datasets/Paul/XSTest" in message
    assert "curl" in message


def test_entry_dirname_is_a_stable_slug_of_the_manifest_name():
    assert testset.entry_dirname("DAMO-NLP-SG/MultiJail") == "damo-nlp-sg-multijail"
    assert testset.entry_dirname("llm-attacks/llm-attacks (AdvBench)") == "llm-attacks-llm-attacks-advbench"
    assert testset.entry_dirname("CohereLabs/aya_redteaming") == "coherelabs-aya-redteaming"


def test_fetch_commands_cover_every_default_source_file(tmp_path):
    commands = testset.fetch_commands(tmp_path)
    for name in testset.DEFAULT_SOURCES:
        for filename, url in testset.SOURCES[name].files.items():
            expected_path = tmp_path / testset.entry_dirname(name) / filename
            assert any(url in c and str(expected_path) in c for c in commands)


# --- which sources may feed the test set --------------------------------------------------


@pytest.fixture(scope="module")
def real_manifest():
    return load_sources(REAL_MANIFEST)


def test_default_sources_are_usable_in_the_real_manifest(real_manifest):
    by_name = {e["name"]: e for e in real_manifest}
    kinds = {}
    for name in testset.DEFAULT_SOURCES:
        assert name in testset.SOURCES, name
        kinds[name] = testset.check_usable(by_name[name])
    assert set(kinds.values()) == {"harmful", "benign_edgy"}
    assert kinds["Paul/XSTest"] == "benign_edgy" and kinds["CohereLabs/aya_redteaming"] == "harmful"


@pytest.mark.parametrize(
    "name",
    [
        "facebook/belebele",  # evaluation only: redistribute false
        "walledai/MultiJail",  # excluded
        "ToxicityPrompts/DAMO-MultiJail",  # excluded
        "CohereLabs/Global-MMLU",  # usable, wrong role (task cross-check)
        "CohereLabs/aya_dataset",  # usable, wrong role (fine-tune data)
        "not/in-the-manifest",
    ],
)
def test_builder_raises_for_a_forbidden_source_before_loading_anything(real_manifest, tmp_path, name):
    calls = []
    spies = {
        n: SourceSpec(n, "x", {}, lambda raw_dir, n=n: calls.append(n) or [RawItem("0", "text")])
        for n in [*testset.DEFAULT_SOURCES, name]
    }
    with pytest.raises(ForbiddenSourceError):
        testset.load_pool(real_manifest, tmp_path, names=["Paul/XSTest", name], sources=spies)
    assert calls == []  # nothing was read, not even the permitted source named first


def test_a_usable_manifest_entry_without_a_loader_is_refused(tmp_path, fixture_manifest):
    with pytest.raises(ForbiddenSourceError, match="no loader"):
        testset.load_pool(fixture_manifest, tmp_path, names=[HARM_A], sources={})


def test_check_usable_follows_used_for_and_redistribute():
    base = {"name": "n", "license": "mit"}
    assert testset.check_usable({**base, "used_for": "harmful_prompts"}) == "harmful"
    assert testset.check_usable({**base, "used_for": "benign_edgy_prompts", "redistribute": True}) == "benign_edgy"
    with pytest.raises(ForbiddenSourceError, match="redistribute"):
        testset.check_usable({**base, "used_for": "harmful_prompts", "redistribute": False})
    with pytest.raises(ForbiddenSourceError, match="excluded"):
        testset.check_usable({**base, "used_for": "excluded"})
    with pytest.raises(ForbiddenSourceError, match="task_qa"):
        testset.check_usable({**base, "used_for": "task_qa"})


def test_prompt_license_and_source_come_from_the_manifest(tmp_path, fixture_manifest, fixture_sources):
    items, _ = testset.load_pool(fixture_manifest, tmp_path, names=FIXTURE_NAMES, sources=fixture_sources)
    assert {i.source for i in items} == set(FIXTURE_NAMES)
    for item in items:
        assert item.license == LICENSES[item.source]
    assert {i.kind for i in items if i.source == BENIGN} == {"benign_edgy"}
    assert {i.kind for i in items if i.source in (HARM_A, HARM_B)} == {"harmful"}
    assert len({i.id for i in items}) == len(items)


# --- dedupe, blanks and long prompts ----------------------------------------------------------


def test_duplicates_are_removed_counted_and_the_first_source_wins(tmp_path, fixture_manifest, fixture_sources):
    items, stats = testset.load_pool(fixture_manifest, tmp_path, names=[HARM_A, HARM_B, BENIGN], sources=fixture_sources)
    assert stats.duplicates_removed == {HARM_A: 0, HARM_B: 1, BENIGN: 0}
    assert stats.loaded == {HARM_A: 14, HARM_B: 12, BENIGN: 9}
    assert stats.kept == {HARM_A: 14, HARM_B: 11, BENIGN: 9}
    survivors = [i for i in items if norm_text(i.text) == "harm a question number 3 please"]
    assert [i.source for i in survivors] == [HARM_A]

    # reversing the source order flips which copy survives (the order is the tie-break)
    items2, stats2 = testset.load_pool(fixture_manifest, tmp_path, names=[HARM_B, HARM_A, BENIGN], sources=fixture_sources)
    assert stats2.duplicates_removed == {HARM_B: 0, HARM_A: 1, BENIGN: 0}
    survivors2 = [i for i in items2 if norm_text(i.text) == "harm a question number 3 please"]
    assert [i.source for i in survivors2] == [HARM_B]


def test_blank_and_overlong_texts_are_removed_and_counted(tmp_path, fixture_manifest):
    rows = [RawItem("0", "fine one"), RawItem("1", "   "), RawItem("2", "x" * 51), RawItem("3", "fine two")]
    sources = {HARM_A: SourceSpec(HARM_A, "ha", {}, lambda raw_dir: list(rows))}
    items, stats = testset.load_pool(fixture_manifest, tmp_path, names=[HARM_A], sources=sources, max_chars=50)
    assert [i.text for i in items] == ["fine one", "fine two"]
    assert stats.blank_removed == {HARM_A: 1}
    assert stats.too_long_removed == {HARM_A: 1}
    assert stats.kept == {HARM_A: 2} and stats.loaded == {HARM_A: 4}


def test_texts_are_stripped_but_otherwise_untouched(tmp_path, fixture_manifest):
    sources = {HARM_A: SourceSpec(HARM_A, "ha", {}, lambda raw_dir: [RawItem("0", "  Keep   My  Casing?\n")])}
    items, _ = testset.load_pool(fixture_manifest, tmp_path, names=[HARM_A], sources=sources)
    assert items[0].text == "Keep   My  Casing?"


def test_duplicate_ids_inside_a_source_are_an_error(tmp_path, fixture_manifest):
    sources = {HARM_A: SourceSpec(HARM_A, "ha", {}, lambda raw_dir: [RawItem("0", "one"), RawItem("0", "two")])}
    with pytest.raises(ValueError, match="duplicate"):
        testset.load_pool(fixture_manifest, tmp_path, names=[HARM_A], sources=sources)


# --- split assignment ---------------------------------------------------------------------


def _pool(n_a=14, n_b=11, n_benign=9):
    items = []
    for i in range(n_a):
        items.append(PoolItem(f"ha-{i:03d}", "harmful", f"harm a {i}", HARM_A, "mit"))
    for i in range(n_b):
        items.append(PoolItem(f"hb-{i:03d}", "harmful", f"harm b {i}", HARM_B, "cc-by-4.0"))
    for i in range(n_benign):
        items.append(PoolItem(f"be-{i:03d}", "benign_edgy", f"benign {i}", BENIGN, "cc-by-4.0"))
    return items


def _assignment(prompts):
    return {p.id: p.split for p in prompts}


def test_split_assignment_is_independent_of_input_order():
    pool = _pool()
    base, base_short = testset.assign_splits(pool, seed=11, sizes=TEST_SIZES)
    for shuffle_seed in range(5):
        shuffled = pool[:]
        random.Random(shuffle_seed).shuffle(shuffled)
        again, short = testset.assign_splits(shuffled, seed=11, sizes=TEST_SIZES)
        assert _assignment(again) == _assignment(base)
        assert short == base_short
    # and the order of the returned prompts is stable too
    assert [p.id for p in again] == [p.id for p in base]


def test_split_assignment_depends_on_the_seed():
    pool = _pool()
    a, _ = testset.assign_splits(pool, seed=1, sizes=TEST_SIZES)
    b, _ = testset.assign_splits(pool, seed=2, sizes=TEST_SIZES)
    assert _assignment(a) != _assignment(b)


def test_split_assignment_is_independent_of_pythonhashseed():
    code = (
        "import json, sys\n"
        "from msd import testset\n"
        "pool = [testset.PoolItem(f'id-{i:03d}', 'harmful', f'text {i}', 'src', 'mit') for i in range(30)]\n"
        "sizes = {'harmful': {'report': 10, 'dev': 5}, 'benign_edgy': {'report': 0, 'dev': 0}}\n"
        "out, _ = testset.assign_splits(pool, seed=3, sizes=sizes)\n"
        "print(json.dumps([[p.id, p.split] for p in out]))\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outputs.add(res.stdout.strip())
    assert len(outputs) == 1


def test_split_rank_is_sha256_of_the_json_encoded_seed_and_id():
    import hashlib

    expected = hashlib.sha256(json.dumps([5, "abc"], ensure_ascii=False).encode("utf-8")).hexdigest()
    assert testset.split_rank(5, "abc") == expected
    # selection uses a different, equally stable hash so selection and split are independent
    assert testset.select_rank(5, "abc") != testset.split_rank(5, "abc")
    assert testset.select_rank(5, "abc") == testset.select_rank(5, "abc")


def test_dev_and_report_have_the_same_source_mix():
    # a big and a small source: picking the lowest hashes and then splitting by the same hash
    # would send every item of the big source to the report split
    pool = [PoolItem(f"big-{i:04d}", "harmful", f"big {i}", HARM_A, "mit") for i in range(500)]
    pool += [PoolItem(f"small-{i:04d}", "harmful", f"small {i}", HARM_B, "mit") for i in range(60)]
    sizes = {"harmful": {"report": 40, "dev": 20}, "benign_edgy": {"report": 0, "dev": 0}}
    for seed in (1, 2, 3):
        prompts, shortfalls = testset.assign_splits(pool, seed=seed, sizes=sizes)
        assert shortfalls == []
        counts = Counter((p.split, p.source) for p in prompts)
        # 30 per source, split 20 report / 10 dev, give or take one
        for source in (HARM_A, HARM_B):
            assert abs(counts[("report", source)] - 20) <= 1
            assert abs(counts[("dev", source)] - 10) <= 1


def test_assigned_sizes_and_disjoint_dev_report():
    prompts, shortfalls = testset.assign_splits(_pool(), seed=3, sizes=TEST_SIZES)
    assert shortfalls == []
    counts = Counter((p.kind, p.split) for p in prompts)
    assert counts == {
        ("harmful", "report"): 8, ("harmful", "dev"): 4,
        ("benign_edgy", "report"): 6, ("benign_edgy", "dev"): 3,
    }
    assert len({p.id for p in prompts}) == len(prompts)  # no id in both splits, none twice
    assert all(p.lang == "en" and p.gold is None for p in prompts)


def test_selection_is_balanced_across_sources_not_proportional_to_pool_size():
    pool = _pool(n_a=40, n_b=6, n_benign=9)  # harm-a is much larger than harm-b
    prompts, _ = testset.assign_splits(pool, seed=3, sizes=TEST_SIZES)
    harmful = Counter(p.source for p in prompts if p.kind == "harmful")
    assert harmful == {HARM_A: 6, HARM_B: 6}  # 12 wanted, equal share each


def test_balanced_selection_hands_leftover_to_the_larger_source():
    pool = _pool(n_a=40, n_b=3, n_benign=9)  # harm-b has only 3 to give
    prompts, shortfalls = testset.assign_splits(pool, seed=3, sizes=TEST_SIZES)
    harmful = Counter(p.source for p in prompts if p.kind == "harmful")
    assert harmful == {HARM_A: 9, HARM_B: 3}
    assert shortfalls == []


def test_shortfall_is_recorded_not_padded():
    pool = _pool(n_a=3, n_b=2, n_benign=4)  # 5 harmful (want 12), 4 benign (want 9)
    prompts, shortfalls = testset.assign_splits(pool, seed=3, sizes=TEST_SIZES)
    counts = Counter((p.kind, p.split) for p in prompts)
    # the report split is filled first, then dev; nothing is duplicated
    assert counts == {("harmful", "report"): 5, ("benign_edgy", "report"): 4}
    assert len({p.id for p in prompts}) == len(prompts) == 9
    assert {(s["kind"], s["split"]): (s["wanted"], s["got"]) for s in shortfalls} == {
        ("harmful", "report"): (8, 5),
        ("harmful", "dev"): (4, 0),
        ("benign_edgy", "report"): (6, 4),
        ("benign_edgy", "dev"): (3, 0),
    }


def test_shortfall_in_the_dev_split_only():
    pool = _pool(n_a=6, n_b=4, n_benign=9)  # 10 harmful: report 8 full, dev gets 2 of 4
    prompts, shortfalls = testset.assign_splits(pool, seed=3, sizes=TEST_SIZES)
    counts = Counter((p.kind, p.split) for p in prompts)
    assert counts[("harmful", "report")] == 8 and counts[("harmful", "dev")] == 2
    assert shortfalls == [{"kind": "harmful", "split": "dev", "wanted": 4, "got": 2}]


# --- the pipeline: same prompt, same split in every language ---------------------------------


def test_same_english_item_lands_in_the_same_split_in_every_language(tmp_path, build):
    build()
    out = tmp_path / "testset"
    english = {p.id: p for s in ("dev", "report") for p in read_jsonl(out / "en" / f"{s}.jsonl", Prompt)}
    assert len(english) == 21  # 12 harmful + 9 benign-edgy
    for lang in NON_ENGLISH:
        for split in ("dev", "report"):
            for p in read_jsonl(out / lang / f"{split}.jsonl", Prompt):
                src = next(e for e in english.values() if translated_id(e.id, lang) == p.id)
                assert src.split == split == p.split
                assert (src.kind, src.source, src.license) == (p.kind, p.source, p.license)


def test_dev_and_report_are_disjoint_at_the_english_level(tmp_path, build):
    build()
    out = tmp_path / "testset"
    dev = {p.id for p in read_jsonl(out / "en" / "dev.jsonl", Prompt)}
    report = {p.id for p in read_jsonl(out / "en" / "report.jsonl", Prompt)}
    assert dev and report and not dev & report


def test_qc_drops_are_counted_per_language_split_and_kind(tmp_path, build, monkeypatch):
    # a back-translator that corrupts every text containing "number 1" fails the meaning check
    def bad_back(texts, src, tgt):
        return [("garbled" if "number 1" in t else t.removeprefix(f"[{src}] ")) for t in texts]

    monkeypatch.setitem(testset.TRANSLATORS, "bad-back", lambda: bad_back)
    summary = build(back_translator_names={lang: "bad-back" for lang in NON_ENGLISH})
    out = tmp_path / "testset"
    for lang in NON_ENGLISH:
        info = summary["languages"][lang]
        written = Counter()
        for split in ("dev", "report"):
            for p in read_jsonl(out / lang / f"{split}.jsonl", Prompt):
                written[(split, p.kind)] += 1
        kept = {(s, k): n for s, kinds in info["kept"].items() for k, n in kinds.items() if n}
        assert kept == dict(written)
        dropped_total = sum(n for kinds in info["dropped"].values() for n in kinds.values())
        assert info["dropped_total"] == dropped_total > 0
        assert info["kept_total"] + info["dropped_total"] == 21
    # English is never QC-dropped
    assert summary["languages"]["en"]["dropped_total"] == 0


def test_langs_en_builds_english_only_and_needs_no_translator(tmp_path, build, monkeypatch):
    for var in testset.TRANSLATE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    summary = build(langs=("en",), translator_names={}, back_translator_names={})
    out = tmp_path / "testset"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.jsonl")) == ["en/dev.jsonl", "en/report.jsonl"]
    assert list(summary["languages"]) == ["en"]


def test_a_requested_language_without_a_translator_is_an_error(build):
    with pytest.raises(ValueError, match="hi"):
        build(langs=("en", "hi"), translator_names={}, back_translator_names={})


def test_translators_are_resolved_before_any_work_and_unknown_names_fail(tmp_path, build):
    with pytest.raises(UnknownNameError):
        build(translator_names={lang: "nope" for lang in NON_ENGLISH})
    assert not (tmp_path / "testset").exists()


def test_each_translator_name_is_built_once(build, monkeypatch):
    built = []

    def factory():
        built.append(1)
        return lambda texts, src, tgt: [f"[{tgt}] {t}" for t in texts]

    monkeypatch.setitem(testset.TRANSLATORS, "counting", factory)
    build(translator_names={lang: "counting" for lang in NON_ENGLISH})
    assert len(built) == 1  # one model load shared by hi, hinglish, ar and sw


def test_summary_records_pool_selection_and_shortfalls(build):
    # 25 harmful survive dedupe but 30 are asked for: the dev split is short by 5
    summary = build(sizes={"harmful": {"report": 20, "dev": 10}, "benign_edgy": {"report": 6, "dev": 3}})
    assert summary["seed"] == 7
    assert summary["sources_used"] == list(FIXTURE_NAMES)
    assert summary["licenses"] == LICENSES
    assert summary["pool"]["duplicates_removed_total"] == 1
    assert summary["pool"]["duplicates_removed"] == {HARM_A: 0, HARM_B: 1, BENIGN: 0}
    assert summary["pool"]["kept"] == {HARM_A: 14, HARM_B: 11, BENIGN: 9}
    assert summary["english"]["selected"] == {
        "report": {"harmful": 20, "benign_edgy": 6},
        "dev": {"harmful": 5, "benign_edgy": 3},
    }
    assert summary["english"]["shortfalls"] == [{"kind": "harmful", "split": "dev", "wanted": 10, "got": 5}]
    assert summary["english"]["by_source"] == {
        HARM_A: {"harmful": 14},
        HARM_B: {"harmful": 11},
        BENIGN: {"benign_edgy": 9},
    }


def test_summary_file_merges_languages_across_runs(tmp_path, build):
    path = tmp_path / "summary.json"
    build(langs=("en", "hi"), summary_path=path)
    build(langs=("en", "sw"), summary_path=path)
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert set(merged["languages"]) == {"en", "hi", "sw"}


# --- registries ---------------------------------------------------------------------------


def test_identity_translator_returns_texts_unchanged():
    tr = testset.get_translator("identity")
    assert tr(["a", "b c"], "en", "hi") == ["a", "b c"]


def test_exact_same_meaning():
    exact = testset.get_same_meaning("exact")
    assert exact("a b", "a b") is True
    assert exact("a b", "a  b") is False


def test_unknown_registry_names_raise_and_list_the_known_ones():
    with pytest.raises(UnknownNameError) as exc:
        testset.get_translator("claude-opus")
    assert "identity" in str(exc.value) and "nllb-600m" in str(exc.value)
    with pytest.raises(UnknownNameError, match="chrf"):
        testset.get_same_meaning("vibes")
    assert issubclass(UnknownNameError, ValueError)


def test_default_translators_are_non_claude_and_cover_every_non_english_language():
    assert set(testset.DEFAULT_TRANSLATORS) == set(NON_ENGLISH) == set(testset.DEFAULT_BACK_TRANSLATORS)
    for name in [*testset.DEFAULT_TRANSLATORS.values(), *testset.DEFAULT_BACK_TRANSLATORS.values()]:
        assert name in testset.TRANSLATORS and "claude" not in name.lower()
    assert testset.DEFAULT_SAME_MEANING in testset.SAME_MEANING
    assert "claude" not in " ".join(testset.TRANSLATORS).lower()


def test_importing_testset_loads_no_model_library():
    code = (
        "import sys\n"
        "import msd.testset\n"
        "print(any(m in sys.modules for m in ('transformers', 'torch')))\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "False"


def test_nllb_without_transformers_raises_a_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # makes `import transformers` fail
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as exc:
        testset.get_translator("nllb-600m")
    message = str(exc.value)
    assert "transformers" in message and "torch" in message and "nllb-600m" in message


def _fake_model_modules(monkeypatch, record):
    """Install fake `transformers` and `torch` modules that record how they are driven."""

    class Batch(dict):
        def to(self, device):
            record["device"] = device
            return self

    class Tok:
        src_lang = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            record["tokenizer_id"] = model_id
            return cls()

        def convert_tokens_to_ids(self, token):
            return {"eng_Latn": 11, "hin_Deva": 22, "arb_Arab": 33, "swh_Latn": 44}[token]

        def __call__(self, texts, **kwargs):
            record.setdefault("src_langs", []).append(self.src_lang)
            return Batch(texts=list(texts))

        def batch_decode(self, outputs, skip_special_tokens=True):
            return [f"<{bos}>{text}" for bos, text in outputs]

    class Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            record["model_id"] = model_id
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, texts, forced_bos_token_id, **kwargs):
            record.setdefault("batches", []).append(list(texts))
            return [(forced_bos_token_id, t) for t in texts]

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = Tok
    transformers.AutoModelForSeq2SeqLM = Model
    torch = types.ModuleType("torch")

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    torch.no_grad = NoGrad
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_nllb_translator_drives_the_model_with_the_right_language_codes(monkeypatch):
    record = {}
    _fake_model_modules(monkeypatch, record)
    tr = testset.get_translator("nllb-600m")
    assert record["model_id"] == record["tokenizer_id"] == testset.NLLB_MODEL_ID == "facebook/nllb-200-distilled-600M"

    texts = ["a much longer text than the others here", "short", "mid length text"]
    out = tr(texts, "en", "hi")
    assert out == [f"<22>{t}" for t in texts]  # order restored, forced BOS = hin_Deva
    assert set(record["src_langs"]) == {"eng_Latn"}
    assert tr(["x"], "hi", "en") == ["<11>x"]
    assert tr(["x"], "en", "sw") == ["<44>x"]
    assert tr(["x"], "ar", "en") == ["<11>x"]
    assert tr([], "en", "ar") == []


def test_nllb_translator_rejects_hinglish_and_unknown_languages(monkeypatch):
    _fake_model_modules(monkeypatch, {})
    tr = testset.get_translator("nllb-600m")
    with pytest.raises(ValueError, match="hinglish"):
        tr(["x"], "en", "hinglish")
    with pytest.raises(ValueError, match="fr"):
        tr(["x"], "fr", "en")


def test_llm_translator_refuses_to_run_without_its_env_vars(monkeypatch):
    for var in testset.TRANSLATE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(TranslatorConfigError) as exc:
        testset.get_translator("llm-chat")
    for var in ("MSD_TRANSLATE_BASE_URL", "MSD_TRANSLATE_MODEL", "MSD_TRANSLATE_API_KEY"):
        assert var in str(exc.value)
    # one var missing is still a refusal, and the message names only what is missing
    monkeypatch.setenv("MSD_TRANSLATE_BASE_URL", "https://example.org/v1")
    monkeypatch.setenv("MSD_TRANSLATE_MODEL", "some-model")
    with pytest.raises(TranslatorConfigError) as exc2:
        testset.get_translator("llm-chat")
    assert "MSD_TRANSLATE_API_KEY" in str(exc2.value) and "MSD_TRANSLATE_MODEL" not in str(exc2.value)


def test_llm_translator_posts_chat_requests_and_returns_the_replies(monkeypatch):
    monkeypatch.setenv("MSD_TRANSLATE_BASE_URL", "https://example.org/v1/")
    monkeypatch.setenv("MSD_TRANSLATE_MODEL", "some-model")
    monkeypatch.setenv("MSD_TRANSLATE_API_KEY", "sekret")
    sent = []

    def fake_post(url, headers, payload, timeout):
        sent.append((url, headers, payload))
        text = payload["messages"][-1]["content"]
        return {"choices": [{"message": {"content": f"  hinglish::{text}  "}}]}

    monkeypatch.setattr(testset, "_http_post", fake_post)
    tr = testset.get_translator("llm-chat")
    assert tr(["one", "two"], "en", "hinglish") == ["hinglish::one", "hinglish::two"]
    url, headers, payload = sent[0]
    assert url == "https://example.org/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sekret"
    assert payload["model"] == "some-model" and payload["temperature"] == 0
    system = payload["messages"][0]
    assert system["role"] == "system" and "Hinglish" in system["content"] and "English" in system["content"]
    assert "do not answer" in system["content"].lower()
    assert payload["messages"][-1] == {"role": "user", "content": "one"}


def test_llm_translator_does_not_swallow_a_malformed_reply(monkeypatch):
    monkeypatch.setenv("MSD_TRANSLATE_BASE_URL", "https://example.org/v1")
    monkeypatch.setenv("MSD_TRANSLATE_MODEL", "m")
    monkeypatch.setenv("MSD_TRANSLATE_API_KEY", "k")
    monkeypatch.setattr(testset, "_http_post", lambda *a, **kw: {"error": "overloaded"})
    tr = testset.get_translator("llm-chat")
    with pytest.raises(RuntimeError, match="reply"):
        tr(["one"], "en", "hi")


# --- the chrF stand-in -----------------------------------------------------------------------


def test_chrf_identical_texts_score_one_and_empty_scores_zero():
    assert testset.chrf_score("How do I renew my passport?", "How do I renew my passport?") == 1.0
    assert testset.chrf_score("abc", "") == 0.0
    assert testset.chrf_score("", "abc") == 0.0
    assert testset.chrf_score("", "") == 0.0


def test_chrf_ignores_case_and_spacing_via_norm_text():
    assert testset.chrf_score("How  do I RENEW?", "how do i renew?") == 1.0


def test_chrf_orders_a_paraphrase_above_an_unrelated_text():
    original = "What is the best way to learn a new language quickly?"
    paraphrase = "What is the best way to quickly learn a new language?"
    unrelated = "The train to the coast leaves at noon on Tuesdays."
    assert testset.chrf_score(original, paraphrase) > 0.8
    assert testset.chrf_score(original, unrelated) < 0.4
    assert testset.chrf_score(original, paraphrase) > testset.chrf_score(original, unrelated)


def test_chrf_is_deterministic_and_bounded():
    a, b = "Tell me about the history of Lagos", "Tell me about Lagos history"
    scores = {testset.chrf_score(a, b) for _ in range(5)}
    assert len(scores) == 1
    assert 0.0 < scores.pop() < 1.0


def test_chrf_same_meaning_uses_the_named_threshold():
    same = testset.get_same_meaning("chrf")
    original = "What is the best way to learn a new language quickly?"
    assert same(original, original) is True
    assert same(original, "What is the best way to quickly learn a new language?") is True
    assert same(original, "The train to the coast leaves at noon on Tuesdays.") is False
    assert isinstance(testset.CHRF_THRESHOLD, float) and 0.0 < testset.CHRF_THRESHOLD < 1.0
    assert same(original, original[: int(len(original) * 0.25)]) is False  # a truncated back-translation
    # the verdict is exactly "score >= threshold"
    probe = "What is the best way to learn a language?"
    assert same(original, probe) == (testset.chrf_score(original, probe) >= testset.CHRF_THRESHOLD)


# --- the HTTP helper, against a local server (no outside network) -----------------------------


@pytest.fixture
def local_server():
    import http.server
    import threading

    seen = []
    script = []  # (status, body) replies, consumed in order

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(length))))
            status, body = script.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode("utf-8"))

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", seen, script
    server.shutdown()
    thread.join()


def test_http_post_sends_json_and_retries_a_server_error(local_server, monkeypatch):
    base, seen, script = local_server
    monkeypatch.setattr(testset, "_RETRY_SLEEP_S", 0.0)
    script += [(503, {"error": "busy"}), (200, {"choices": [{"message": {"content": "ok"}}]})]
    reply = testset._http_post(f"{base}/v1/chat/completions", {"Authorization": "Bearer k"}, {"a": 1}, 5)
    assert reply["choices"][0]["message"]["content"] == "ok"
    assert [s[0] for s in seen] == ["/v1/chat/completions"] * 2 and seen[0][1] == "Bearer k" and seen[0][2] == {"a": 1}


def test_http_post_raises_on_a_client_error_without_retrying(local_server, monkeypatch):
    base, seen, script = local_server
    monkeypatch.setattr(testset, "_RETRY_SLEEP_S", 0.0)
    script += [(401, {"error": "bad key"})]
    with pytest.raises(RuntimeError, match="HTTP 401"):
        testset._http_post(f"{base}/x", {}, {}, 5)
    assert len(seen) == 1


def test_nemotron_has_a_loader_but_is_not_a_default_source():
    assert "nvidia/Nemotron-Safety-Guard-Dataset-v3" in testset.SOURCES
    assert "nvidia/Nemotron-Safety-Guard-Dataset-v3" not in testset.DEFAULT_SOURCES


# --- fix round 1: M5, redistribute must be a real bool ----------------------------------------------


@pytest.mark.parametrize("bad", ["false", "False", "no", 0, 1, None, []])
def test_check_usable_rejects_a_non_bool_redistribute(bad):
    entry = {"name": "n", "license": "mit", "used_for": "harmful_prompts", "redistribute": bad}
    with pytest.raises(ForbiddenSourceError, match="redistribute"):
        testset.check_usable(entry)


def test_check_usable_still_accepts_an_absent_or_true_redistribute():
    entry = {"name": "n", "license": "mit", "used_for": "harmful_prompts"}
    assert testset.check_usable(entry) == "harmful"
    assert testset.check_usable({**entry, "redistribute": True}) == "harmful"


def test_fetch_commands_for_a_source_without_a_loader_is_a_clean_error(tmp_path):
    with pytest.raises(ForbiddenSourceError, match="no loader"):
        testset.fetch_commands(tmp_path, ["facebook/belebele"])


# --- fix round 1: I4, the chrF stand-in must reject what it claims to catch --------------------------

REVIEW_ORIGINAL = "How do I pick the lock on my own front door?"
REVIEW_BAD_BACK_TRANSLATIONS = {
    "answer instead of a translation": (
        "To pick the lock on your own front door, insert a tension wrench into the "
        "bottom of the lock, then rake the pins with a pick until the lock turns."
    ),
    "question with an appended answer": REVIEW_ORIGINAL + " First, insert a tension wrench, then rake the pins until it turns.",
    "translation wrapped in a preface": "Here is the translation: " + REVIEW_ORIGINAL,
    "truncated to half": REVIEW_ORIGINAL[: len(REVIEW_ORIGINAL) // 2],
    "a refusal": "I'm sorry, but I can't help with picking locks.",
    "a short refusal": "I cannot help with that.",
    "unrelated text": "The train to the coast leaves at noon on Tuesdays.",
}


@pytest.mark.parametrize("name", sorted(REVIEW_BAD_BACK_TRANSLATIONS))
def test_chrf_same_meaning_rejects_the_failure_modes_it_claims_to_catch(name):
    same = testset.get_same_meaning("chrf")
    assert same(REVIEW_ORIGINAL, REVIEW_BAD_BACK_TRANSLATIONS[name]) is False


def test_chrf_same_meaning_accepts_identical_and_faithful_back_translations():
    same = testset.get_same_meaning("chrf")
    assert same(REVIEW_ORIGINAL, REVIEW_ORIGINAL) is True
    assert same(REVIEW_ORIGINAL, "How can I pick the lock of my own front door?") is True
    assert same("What is the best way to learn a new language quickly?",
                "What is the best way to quickly learn a new language?") is True


def test_chrf_same_meaning_in_devanagari_still_passes_when_faithful():
    same = testset.get_same_meaning("chrf")
    hindi = "मैं अपने घर का ताला कैसे खोलूं?"
    assert same(hindi, hindi) is True
    assert same(hindi, "मैं अपने घर का ताला कैसे खोल सकता हूं?") is True
    assert same(hindi, hindi + " " + hindi) is False  # repeated: length guard
    assert same(hindi, "आज मौसम बहुत अच्छा है") is False


def test_length_ratio_guard_is_what_rejects_a_same_content_but_too_long_or_short_text():
    same = testset.get_same_meaning("chrf")
    doubled = REVIEW_ORIGINAL + " " + REVIEW_ORIGINAL  # chrF alone would accept it (F1 is about 0.67)
    assert testset.chrf_score(REVIEW_ORIGINAL, doubled) >= testset.CHRF_THRESHOLD
    assert same(REVIEW_ORIGINAL, doubled) is False
    half = REVIEW_ORIGINAL[: len(REVIEW_ORIGINAL) // 2]  # chrF alone would accept it too
    assert testset.chrf_score(REVIEW_ORIGINAL, half) >= testset.CHRF_THRESHOLD
    assert same(REVIEW_ORIGINAL, half) is False
    assert 0.0 < testset.LENGTH_RATIO_MIN < 1.0 < testset.LENGTH_RATIO_MAX


def test_chrf_is_an_f1_so_it_is_symmetric():
    a, b = "Tell me about the history of Lagos", "Tell me about Lagos history in the city"
    assert testset.chrf_score(a, b) == testset.chrf_score(b, a)


def test_chrf_same_meaning_always_returns_a_real_bool():
    same = testset.get_same_meaning("chrf")
    for original, back in [("a b", "a b"), ("a b", "c d"), ("", ""), ("x", ""), ("", "x")]:
        assert type(same(original, back)) is bool
