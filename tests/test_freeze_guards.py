"""Fix round 1: a frozen set is not rebuilt silently (I2), and the languages of one test set are
consistent with each other and with their recorded provenance (I3)."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from msd import testset
from msd.config import LANGS, NON_ENGLISH
from msd.schema import Prompt, read_jsonl, write_jsonl
from msd.testset import RawItem, SourceSpec
from msd.translate import translated_id

from conftest import BENIGN, FIXTURE_NAMES, HARM_A, TEST_SIZES

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}


# --- I2: no silent rebuild or re-freeze of a frozen set ----------------------------------------------


def test_build_refuses_to_run_over_a_frozen_set_and_touches_nothing(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    before = snapshot(out)
    for kwargs in ({}, {"langs": ("en",)}, {"freeze": True}):
        with pytest.raises(testset.AlreadyFrozenError, match="refreeze"):
            build(**kwargs)
    assert snapshot(out) == before


def test_refreeze_allows_a_rebuild_and_a_second_freeze(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    first = sha(out / "MANIFEST.json")
    summary = build(freeze=True, refreeze=True)
    assert sha(out / "MANIFEST.json") == first  # same inputs, same bytes
    assert summary["freeze"]["sha256"] == first


def test_freeze_refuses_when_a_manifest_exists_unless_refreeze(tmp_path, build, freeze_kwargs):
    summary = build()
    out = tmp_path / "testset"
    result = testset.freeze(out, summary=summary, **freeze_kwargs)
    assert result.sha256 == sha(out / "MANIFEST.json") and result.path == out / "MANIFEST.json"
    with pytest.raises(testset.AlreadyFrozenError, match="refreeze"):
        testset.freeze(out, summary=summary, **freeze_kwargs)
    again = testset.freeze(out, summary=summary, refreeze=True, **freeze_kwargs)
    assert again.sha256 == result.sha256


def test_a_dry_manifest_never_blocks_a_build_or_the_real_freeze(tmp_path, build):
    build(langs=("en",))
    testset.freeze(tmp_path / "testset", dry=True)
    build()  # no AlreadyFrozenError


def test_cli_frozen_set_exit_codes_and_sha256_line(tmp_path):
    raw = tmp_path / "raw"
    from conftest import write_csv_file

    write_csv_file(raw / "paul-xstest" / "xstest_prompts.csv", ["id", "prompt", "type", "label"],
                   [[str(i), f"xs safe {i}", "t", "safe"] for i in range(210)])  # 200 report + 10 dev
    # use the identity pipeline over the real file layout through the library entry point of the CLI
    out = tmp_path / "out"
    base = [sys.executable, str(SCRIPTS / "build_testset.py"), "--raw", str(raw), "--out", str(out),
            "--source", "Paul/XSTest", "--translator", "identity", "--back-translator", "identity",
            "--same-meaning", "exact"]
    assert subprocess.run(base, capture_output=True, text=True).returncode == 0
    freeze_only = [sys.executable, str(SCRIPTS / "build_testset.py"), "--raw", str(raw), "--out", str(out),
                   "--source", "Paul/XSTest", "--freeze-only"]
    first = subprocess.run(freeze_only, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert sha(out / "MANIFEST.json") in first.stdout  # printed so it can go into the freeze log
    second = subprocess.run(freeze_only, capture_output=True, text=True)
    assert second.returncode == 3 and "refreeze" in second.stderr
    rebuild = subprocess.run(base, capture_output=True, text=True)
    assert rebuild.returncode == 3 and "refreeze" in rebuild.stderr
    ok = subprocess.run([*freeze_only, "--refreeze"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert subprocess.run([*base, "--refreeze"], capture_output=True, text=True).returncode == 0


# --- I3: consistency across languages ----------------------------------------------------------------------


def _english(out):
    return {p.id: p for s in ("dev", "report") for p in read_jsonl(out / "en" / f"{s}.jsonl", Prompt)}


def _rewrite_summary_hashes(summary, out, langs):
    outputs = dict(summary["outputs"])
    for lang in langs:
        for split in ("dev", "report"):
            outputs[f"{lang}/{split}.jsonl"] = sha(out / lang / f"{split}.jsonl")
    return {**summary, "outputs": outputs}


def test_check_consistency_passes_on_a_clean_set(tmp_path, build):
    summary = build()
    testset.check_consistency(tmp_path / "testset", summary)


def test_an_item_in_different_splits_across_languages_is_rejected(tmp_path, build):
    summary = build()
    out = tmp_path / "testset"
    dev = read_jsonl(out / "hi" / "dev.jsonl", Prompt)
    report = read_jsonl(out / "hi" / "report.jsonl", Prompt)
    moved = replace(dev[0], split="report")  # the row agrees with its new file, but not with English
    write_jsonl(out / "hi" / "dev.jsonl", dev[1:])
    write_jsonl(out / "hi" / "report.jsonl", [*report, moved])
    with pytest.raises(testset.ConsistencyError, match="split"):
        testset.check_consistency(out, summary)


@pytest.mark.parametrize("field,value", [("kind", "benign_edgy"), ("source", "somewhere/else"), ("license", "cc0")])
def test_a_row_whose_kind_source_or_license_differs_from_english_is_rejected(tmp_path, build, field, value):
    summary = build()
    out = tmp_path / "testset"
    rows = read_jsonl(out / "ar" / "report.jsonl", Prompt)
    target = next(p for p in rows if getattr(p, field) != value)
    write_jsonl(out / "ar" / "report.jsonl", [replace(p, **{field: value}) if p is target else p for p in rows])
    with pytest.raises(testset.ConsistencyError, match=field):
        testset.check_consistency(out, summary)


def test_a_non_english_row_without_an_english_partner_is_rejected(tmp_path, build):
    summary = build()
    out = tmp_path / "testset"
    rows = read_jsonl(out / "sw" / "dev.jsonl", Prompt)
    write_jsonl(out / "sw" / "dev.jsonl", [*rows, replace(rows[0], id=translated_id("not-an-english-id", "sw"))])
    with pytest.raises(testset.ConsistencyError, match="no English partner"):
        testset.check_consistency(out, summary)


def test_a_silent_gap_in_a_language_is_rejected(tmp_path, build):
    summary = build()
    out = tmp_path / "testset"
    rows = read_jsonl(out / "hinglish" / "dev.jsonl", Prompt)
    write_jsonl(out / "hinglish" / "dev.jsonl", rows[1:])  # one row vanished, no QC drop recorded
    with pytest.raises(testset.ConsistencyError, match="no QC drop"):
        testset.check_consistency(out, summary)


def test_a_gap_accounted_for_by_a_recorded_qc_drop_passes(tmp_path, build, monkeypatch):
    def bad_back(texts, src, tgt):
        return [("garbled" if "number 1" in t else t.removeprefix(f"[{src}] ")) for t in texts]

    monkeypatch.setitem(testset.TRANSLATORS, "bad-back", lambda: bad_back)
    summary = build(back_translator_names={lang: "bad-back" for lang in NON_ENGLISH})
    out = tmp_path / "testset"
    english = _english(out)
    for lang in NON_ENGLISH:
        dropped = summary["languages"][lang]["dropped_ids"]
        assert dropped and set(dropped) <= set(english)
        present = {p.id for s in ("dev", "report") for p in read_jsonl(out / lang / f"{s}.jsonl", Prompt)}
        assert {translated_id(i, lang) for i in dropped}.isdisjoint(present)
        assert len(present) + len(dropped) == len(english)
    testset.check_consistency(out, summary)


def test_a_recorded_drop_that_is_actually_present_is_rejected(tmp_path, build):
    summary = build()
    out = tmp_path / "testset"
    english_id = next(iter(_english(out)))
    bad = json.loads(json.dumps(summary))
    bad["languages"]["hi"]["dropped_ids"] = [english_id]
    with pytest.raises(testset.ConsistencyError, match="recorded as dropped"):
        testset.check_consistency(out, bad)


def test_a_language_without_a_build_record_is_rejected(tmp_path, build):
    summary = build()
    bad = json.loads(json.dumps(summary))
    del bad["languages"]["sw"]
    with pytest.raises(testset.ConsistencyError, match="build record"):
        testset.check_consistency(tmp_path / "testset", bad)


def test_freeze_refuses_an_inconsistent_set_even_when_its_hashes_were_re_recorded(tmp_path, build, freeze_kwargs):
    summary = build()
    out = tmp_path / "testset"
    dev = read_jsonl(out / "hi" / "dev.jsonl", Prompt)
    report = read_jsonl(out / "hi" / "report.jsonl", Prompt)
    write_jsonl(out / "hi" / "dev.jsonl", dev[1:])
    write_jsonl(out / "hi" / "report.jsonl", [*report, replace(dev[0], split="report")])
    forged = _rewrite_summary_hashes(summary, out, ["hi"])  # the record agrees with the files now
    with pytest.raises(testset.ConsistencyError):
        testset.freeze(out, summary=forged, **freeze_kwargs)
    assert not (out / "MANIFEST.json").exists()


def test_freeze_refuses_when_an_output_changed_after_it_was_recorded(tmp_path, build, freeze_kwargs):
    summary = build()
    out = tmp_path / "testset"
    target = out / "ar" / "dev.jsonl"
    target.write_bytes(target.read_bytes().replace(b"harm", b"hurm", 1))
    with pytest.raises(testset.ConsistencyError, match="recorded"):
        testset.freeze(out, summary=summary, **freeze_kwargs)


@pytest.mark.parametrize("fault, ascii_only", [(0xFFFD, False), (0xFFFD, True), (0xD800, True)],
                         ids=["replacement-raw", "replacement-escaped", "lone-surrogate-escaped"])
def test_freeze_refuses_a_prompt_with_a_decoding_fault(tmp_path, build, freeze_kwargs, fault, ascii_only):
    # found at freeze time, not at export time after every GPU run: the export would refuse such a prompt anyway
    summary = build()
    out = tmp_path / "testset"
    target = out / "ar" / "dev.jsonl"
    rows = read_jsonl(target, Prompt)
    rows[0] = replace(rows[0], text=rows[0].text + " " + chr(fault))
    target.write_text("".join(json.dumps(asdict(r), ensure_ascii=ascii_only) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="decoding fault") as exc:
        testset.freeze(out, summary=summary, **freeze_kwargs)
    assert "ar/dev.jsonl" in str(exc.value)
    assert not (out / "MANIFEST.json").exists()


def test_freeze_needs_a_build_record_and_every_language_in_it(tmp_path, build, freeze_kwargs):
    summary = build()
    out = tmp_path / "testset"
    with pytest.raises(ValueError, match="build record"):
        testset.freeze(out, **freeze_kwargs)
    short = json.loads(json.dumps(summary))
    del short["languages"]["sw"]
    with pytest.raises(testset.ConsistencyError, match="sw"):
        testset.freeze(out, summary=short, **freeze_kwargs)


@pytest.mark.parametrize(
    "override,word",
    [
        ({"seed": 8}, "seed"),
        ({"source_names": (HARM_A, BENIGN)}, "sources"),
        ({"sizes": {"harmful": {"report": 9, "dev": 4}, "benign_edgy": {"report": 6, "dev": 3}}}, "sizes"),
    ],
)
def test_freeze_refuses_when_seed_sources_or_sizes_differ_from_the_record(tmp_path, build, freeze_kwargs, override, word):
    summary = build()
    out = tmp_path / "testset"
    with pytest.raises(testset.ConsistencyError, match=word):
        testset.freeze(out, summary=summary, **{**freeze_kwargs, **override})
    assert not (out / "MANIFEST.json").exists()


# --- I3: the build record binds runs together --------------------------------------------------------------------


@pytest.fixture
def file_backed(tmp_path, fixture_sources):
    """harm-a reads a real raw file, so its sha256 is part of the build record."""
    raw_file = tmp_path / "raw" / testset.entry_dirname(HARM_A) / "a.txt"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("".join(f"harm a question number {i} please\n" for i in range(14)), encoding="utf-8")

    def load(raw_dir):
        lines = (Path(raw_dir) / "a.txt").read_text(encoding="utf-8").splitlines()
        return [RawItem(f"{i:03d}", line) for i, line in enumerate(lines)]

    sources = {**fixture_sources, HARM_A: SourceSpec(HARM_A, "ha", {"a.txt": "https://example.org/a.txt"}, load)}
    return sources, raw_file


def test_the_build_record_holds_raw_hashes_output_hashes_and_dropped_ids(tmp_path, build, file_backed):
    sources, raw_file = file_backed
    summary = build(sources=sources)
    out = tmp_path / "testset"
    assert summary["raw_files"][HARM_A] == {"a.txt": sha(raw_file)}
    assert summary["raw_files"][BENIGN] == {}  # a source with no raw file records no hash
    assert sorted(summary["outputs"]) == sorted(f"{lang}/{split}.jsonl" for lang in LANGS for split in ("dev", "report"))
    for rel, digest in summary["outputs"].items():
        assert digest == sha(out / rel)
    assert all(summary["languages"][lang]["dropped_ids"] == [] for lang in LANGS)


def test_per_language_runs_of_one_build_merge_and_stay_consistent(tmp_path, build):
    path = tmp_path / "summary.json"
    build(langs=("en", "hi"), summary_path=path)
    build(langs=("en", "sw"), summary_path=path)
    summary = build(langs=("en", "hinglish", "ar"), summary_path=path)
    assert set(summary["languages"]) == set(LANGS)
    assert json.loads(path.read_text(encoding="utf-8"))["outputs"] == summary["outputs"]
    testset.check_consistency(tmp_path / "testset", summary)


def test_a_later_run_with_a_different_raw_pool_is_refused_before_anything_is_written(tmp_path, build, fixture_sources):
    # the reviewer's r2 scenario: en+hi, then en+hinglish+ar+sw from a re-downloaded, bigger raw file
    path = tmp_path / "summary.json"
    build(langs=("en", "hi"), summary_path=path)
    out = tmp_path / "testset"
    before, record_before = snapshot(out), path.read_bytes()
    bigger = {**fixture_sources, HARM_A: SourceSpec(
        HARM_A, "ha", {}, lambda d: [RawItem(f"{i:03d}", f"harm a question number {i} please") for i in range(54)])}
    with pytest.raises(testset.ConsistencyError, match="differs"):
        build(langs=("en", "hinglish", "ar", "sw"), sources=bigger, summary_path=path)
    assert snapshot(out) == before and path.read_bytes() == record_before


def test_a_later_run_with_a_different_seed_is_refused(tmp_path, build):
    path = tmp_path / "summary.json"
    build(langs=("en", "hi"), summary_path=path)
    before = snapshot(tmp_path / "testset")
    with pytest.raises(testset.ConsistencyError, match="seed"):
        build(langs=("en", "sw"), seed=8, summary_path=path)
    assert snapshot(tmp_path / "testset") == before


def test_a_later_run_with_a_changed_raw_file_is_refused(tmp_path, build, file_backed):
    sources, raw_file = file_backed
    path = tmp_path / "summary.json"
    build(langs=("en", "hi"), sources=sources, summary_path=path)
    raw_file.write_text(raw_file.read_text(encoding="utf-8") + "an extra row appended later\n", encoding="utf-8")
    with pytest.raises(testset.ConsistencyError, match="raw file"):
        build(langs=("en", "sw"), sources=sources, summary_path=path)


def test_a_later_run_with_other_sources_or_sizes_is_refused(tmp_path, build):
    path = tmp_path / "summary.json"
    build(langs=("en", "hi"), summary_path=path)
    with pytest.raises(testset.ConsistencyError, match="sources"):
        build(langs=("en", "sw"), source_names=(HARM_A, BENIGN), summary_path=path)
    with pytest.raises(testset.ConsistencyError, match="sizes"):
        build(langs=("en", "sw"), sizes={"harmful": {"report": 8, "dev": 4}, "benign_edgy": {"report": 5, "dev": 3}},
              summary_path=path)


def test_an_unreadable_existing_record_is_refused_not_overwritten(tmp_path, build):
    path = tmp_path / "summary.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(testset.ConsistencyError, match="not valid JSON"):
        build(langs=("en",), summary_path=path)
    assert path.read_text(encoding="utf-8") == "{not json"


def test_stale_language_files_from_another_build_fail_the_end_of_run_check(tmp_path, build):
    build(langs=("en", "hi"))  # no summary file kept
    with pytest.raises(testset.ConsistencyError, match="hi"):
        build(langs=("en", "sw"), seed=8)  # English changed underneath the old hi files


def test_freeze_refuses_when_a_raw_file_changed_after_the_build(tmp_path, build, file_backed, freeze_kwargs):
    sources, raw_file = file_backed
    summary = build(sources=sources)
    raw_file.write_text(raw_file.read_text(encoding="utf-8") + "more\n", encoding="utf-8")
    with pytest.raises(testset.ConsistencyError, match="raw file"):
        testset.freeze(tmp_path / "testset", summary=summary, **{**freeze_kwargs, "sources": sources})
    # unchanged raw file: the same call freezes
    raw_file.write_text(raw_file.read_text(encoding="utf-8").removesuffix("more\n"), encoding="utf-8")
    result = testset.freeze(tmp_path / "testset", summary=summary, **{**freeze_kwargs, "sources": sources})
    assert result.sha256 == sha(tmp_path / "testset" / "MANIFEST.json")


def test_the_english_digest_taken_before_writing_equals_the_written_files_sha256(tmp_path, build, fixture_manifest, fixture_sources):
    pool, _ = testset.load_pool(fixture_manifest, tmp_path, FIXTURE_NAMES, fixture_sources)
    english, _ = testset.assign_splits(pool, 7, TEST_SIZES)
    summary = build(langs=("en",))
    expected = testset.english_digests(english)
    assert expected == {rel: summary["outputs"][rel] for rel in ("en/dev.jsonl", "en/report.jsonl")}
    assert all(expected[rel] == sha(tmp_path / "testset" / rel) for rel in expected)


def test_cli_a_second_language_run_with_a_different_seed_exits_4(tmp_path):
    raw = tmp_path / "raw"
    from conftest import write_csv_file

    write_csv_file(raw / "paul-xstest" / "xstest_prompts.csv", ["id", "prompt", "type", "label"],
                   [[str(i), f"xs safe {i}", "t", "safe"] for i in range(12)])
    base = [sys.executable, str(SCRIPTS / "build_testset.py"), "--raw", str(raw), "--out", str(tmp_path / "out"),
            "--source", "Paul/XSTest", "--translator", "identity", "--back-translator", "identity",
            "--same-meaning", "exact"]
    assert subprocess.run([*base, "--langs", "hi"], capture_output=True, text=True).returncode == 0
    res = subprocess.run([*base, "--langs", "sw", "--seed", "8"], capture_output=True, text=True)
    assert res.returncode == 4 and "seed" in res.stderr and "Traceback" not in res.stderr
