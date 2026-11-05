"""Build and freeze the test set: the four brief tests, plus verify_freeze behaviour."""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from msd import testset
from msd.config import LANGS
from msd.schema import Prompt, read_jsonl
from msd.splits import LeakageError

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
EXPECTED_FILES = sorted(f"{lang}/{split}.jsonl" for lang in LANGS for split in ("dev", "report"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _line_count(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


# --- the four brief tests ---------------------------------------------------------------


def test_build_writes_split_files_and_manifest(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"

    on_disk = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.jsonl"))
    assert on_disk == EXPECTED_FILES

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert sorted(manifest) == EXPECTED_FILES
    for rel, info in manifest.items():
        assert set(info) == {"sha256", "n"}
        assert info["sha256"] == _sha(out / rel)
        assert info["n"] == _line_count(out / rel) > 0
        rows = read_jsonl(out / rel, Prompt)
        assert len(rows) == info["n"]
        assert {p.split for p in rows} == {Path(rel).stem}
        assert {p.lang for p in rows} == {Path(rel).parts[0]}


def test_counts_per_language_and_kind(tmp_path, build, fixture_manifest):
    licenses = {e["name"]: e["license"] for e in fixture_manifest}
    summary = build()
    out = tmp_path / "testset"
    for lang in LANGS:
        for split in ("dev", "report"):
            rows = read_jsonl(out / lang / f"{split}.jsonl", Prompt)
            counts = Counter(p.kind for p in rows)
            assert counts == {
                "harmful": ({"dev": 4, "report": 8}[split]),
                "benign_edgy": ({"dev": 3, "report": 6}[split]),
            }
            # provenance: source is a manifest entry name, license is that entry's license
            for p in rows:
                assert p.license == licenses[p.source]
    for lang in LANGS:
        assert summary["languages"][lang]["kept"]["report"] == {"harmful": 8, "benign_edgy": 6}
        assert summary["languages"][lang]["kept"]["dev"] == {"harmful": 4, "benign_edgy": 3}


def test_build_fails_on_dev_report_overlap(tmp_path, build, monkeypatch):
    # a translator that collapses every text to one string makes a dev and a report prompt equal
    monkeypatch.setitem(testset.TRANSLATORS, "collapse", lambda: (lambda texts, src, tgt: ["same text"] * len(texts)))
    monkeypatch.setitem(testset.TRANSLATORS, "echo", lambda: (lambda texts, src, tgt: list(texts)))
    monkeypatch.setitem(testset.SAME_MEANING, "always", lambda: (lambda a, b: True))
    with pytest.raises(LeakageError):
        build(
            langs=("en", "sw"),
            translator_names={"sw": "collapse"},
            back_translator_names={"sw": "echo"},
            same_meaning_name="always",
        )
    # a failed build writes no test-set files (and certainly no manifest)
    out = tmp_path / "testset"
    assert not list(out.rglob("*.jsonl"))
    assert not (out / "MANIFEST.json").exists()


def test_verify_freeze_detects_edit(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    script = [sys.executable, str(SCRIPTS / "verify_freeze.py"), "--dir", str(out)]

    ok = subprocess.run(script, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr

    target = out / "hi" / "report.jsonl"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01  # change one byte
    target.write_bytes(bytes(data))

    bad = subprocess.run(script, capture_output=True, text=True)
    assert bad.returncode != 0
    assert "hi/report.jsonl" in bad.stdout + bad.stderr


# --- verify_freeze: every failure mode ------------------------------------------------


def test_verify_freeze_passes_on_untouched_set(tmp_path, build):
    build(freeze=True)
    assert testset.verify_freeze(tmp_path / "testset") == []


def test_verify_freeze_flags_a_changed_byte(tmp_path, build):
    build(freeze=True)
    target = tmp_path / "testset" / "ar" / "dev.jsonl"
    target.write_bytes(target.read_bytes().replace(b"harm", b"hurm", 1))
    problems = testset.verify_freeze(tmp_path / "testset")
    assert len(problems) == 1 and "ar/dev.jsonl" in problems[0] and "sha256" in problems[0]


def test_verify_freeze_flags_a_missing_file(tmp_path, build):
    build(freeze=True)
    (tmp_path / "testset" / "sw" / "report.jsonl").unlink()
    problems = testset.verify_freeze(tmp_path / "testset")
    assert any("sw/report.jsonl" in p and "missing" in p for p in problems)


def test_verify_freeze_flags_an_extra_file(tmp_path, build):
    build(freeze=True)
    (tmp_path / "testset" / "en" / "extra.jsonl").write_text("{}\n", encoding="utf-8")
    problems = testset.verify_freeze(tmp_path / "testset")
    assert any("en/extra.jsonl" in p and "not in MANIFEST" in p for p in problems)


def test_verify_freeze_flags_an_n_mismatch(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["en/dev.jsonl"]["n"] += 1
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    problems = testset.verify_freeze(out)
    assert len(problems) == 1 and "en/dev.jsonl" in problems[0] and "n" in problems[0]


def test_verify_freeze_needs_a_manifest(tmp_path):
    (tmp_path / "t").mkdir()
    assert any("MANIFEST.json" in p for p in testset.verify_freeze(tmp_path / "t"))
    script = [sys.executable, str(SCRIPTS / "verify_freeze.py"), "--dir", str(tmp_path / "t")]
    assert subprocess.run(script, capture_output=True, text=True).returncode != 0


def test_verify_freeze_script_exit_codes_for_missing_and_extra(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    script = [sys.executable, str(SCRIPTS / "verify_freeze.py"), "--dir", str(out)]
    (out / "stray.txt").write_text("x", encoding="utf-8")
    assert subprocess.run(script, capture_output=True, text=True).returncode != 0
    (out / "stray.txt").unlink()
    assert subprocess.run(script, capture_output=True, text=True).returncode == 0
    (out / "en" / "dev.jsonl").unlink()
    assert subprocess.run(script, capture_output=True, text=True).returncode != 0


# --- freezing ----------------------------------------------------------------------------


def test_real_freeze_refuses_a_partial_set(tmp_path, build, freeze_kwargs):
    summary = build(langs=("en",))  # English only, no freeze
    out = tmp_path / "testset"
    with pytest.raises(ValueError, match="hinglish"):
        testset.freeze(out, summary=summary, **freeze_kwargs)
    assert not (out / "MANIFEST.json").exists()
    assert not (out / "MANIFEST.dry.json").exists()


def test_freeze_is_deterministic_and_rebuild_is_byte_identical(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    first = (out / "MANIFEST.json").read_bytes()
    build(freeze=True, refreeze=True)  # an explicit re-freeze of the same inputs
    assert (out / "MANIFEST.json").read_bytes() == first


def test_build_freeze_flag_with_partial_langs_is_refused(tmp_path, build):
    with pytest.raises(ValueError, match="partial"):
        build(langs=("en",), freeze=True)


# --- fix round 1, I1: the real freeze is complete or it is not a freeze ---------------------------------


def _write_manifest(out, name, files):
    manifest = {
        rel: {"sha256": hashlib.sha256((out / rel).read_bytes()).hexdigest(), "n": _line_count(out / rel)}
        for rel in files
    }
    (out / name).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def test_verify_freeze_fails_for_a_partial_manifest_under_the_real_name(tmp_path, build):
    build(langs=("en",))
    out = tmp_path / "testset"
    _write_manifest(out, "MANIFEST.json", ["en/dev.jsonl", "en/report.jsonl"])  # self-consistent but partial
    problems = testset.verify_freeze(out)
    assert any("hi/dev.jsonl" in p and "not listed" in p for p in problems)
    assert any("sw/report.jsonl" in p and "not listed" in p for p in problems)
    script = [sys.executable, str(SCRIPTS / "verify_freeze.py"), "--dir", str(out)]
    assert subprocess.run(script, capture_output=True, text=True).returncode != 0


def test_verify_freeze_fails_for_an_empty_manifest(tmp_path, build):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "MANIFEST.json").write_text("{}", encoding="utf-8")
    assert testset.verify_freeze(empty_dir)
    build()  # a complete set on disk, but an empty manifest
    full = tmp_path / "testset"
    (full / "MANIFEST.json").write_text("{}", encoding="utf-8")
    problems = testset.verify_freeze(full)
    assert any("not listed" in p for p in problems) and any("not in MANIFEST" in p for p in problems)


def test_verify_freeze_fails_when_a_language_file_and_its_entry_are_both_dropped(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    del manifest["sw/report.jsonl"]
    (out / "sw" / "report.jsonl").unlink()
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    problems = testset.verify_freeze(out)
    assert any("sw/report.jsonl" in p and "not listed" in p for p in problems)


def test_verify_freeze_fails_for_a_manifest_entry_that_is_not_a_test_set_file(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    (out / "en" / "extra.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["en/extra.jsonl"] = {"sha256": hashlib.sha256((out / "en" / "extra.jsonl").read_bytes()).hexdigest(), "n": 1}
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("en/extra.jsonl" in p and "not a test-set file" in p for p in testset.verify_freeze(out))


def test_verify_freeze_fails_for_empty_test_set_files(tmp_path):
    out = tmp_path / "t"
    for lang in LANGS:
        (out / lang).mkdir(parents=True)
        for split in ("dev", "report"):
            (out / lang / f"{split}.jsonl").write_bytes(b"")
    _write_manifest(out, "MANIFEST.json", EXPECTED_FILES)
    problems = testset.verify_freeze(out)
    assert any("n is 0" in p for p in problems)


def test_dry_freeze_writes_the_dry_manifest_and_never_the_real_name(tmp_path, build):
    build(langs=("en",))
    out = tmp_path / "testset"
    result = testset.freeze(out, dry=True)
    assert (out / "MANIFEST.dry.json").is_file() and not (out / "MANIFEST.json").exists()
    assert sorted(result.manifest) == ["en/dev.jsonl", "en/report.jsonl"]
    assert testset.verify_freeze(out, dry=True) == []
    assert any("MANIFEST.json" in p for p in testset.verify_freeze(out))  # not a real freeze


def test_dry_verify_still_catches_changed_missing_and_extra_files(tmp_path, build):
    build(langs=("en",))
    out = tmp_path / "testset"
    testset.freeze(out, dry=True)
    script = [sys.executable, str(SCRIPTS / "verify_freeze.py"), "--dry", "--dir", str(out)]
    assert subprocess.run(script, capture_output=True, text=True).returncode == 0

    target = out / "en" / "dev.jsonl"
    original = target.read_bytes()
    target.write_bytes(original.replace(b"harm", b"hurm", 1))
    changed = subprocess.run(script, capture_output=True, text=True)
    assert changed.returncode != 0 and "en/dev.jsonl" in changed.stdout
    target.write_bytes(original)

    (out / "hi").mkdir()
    (out / "hi" / "dev.jsonl").write_text("{}\n", encoding="utf-8")  # built after the dry freeze
    assert any("hi/dev.jsonl" in p and "not in" in p for p in testset.verify_freeze(out, dry=True))
    (out / "hi" / "dev.jsonl").unlink()
    target.unlink()
    assert any("en/dev.jsonl" in p and "missing" in p for p in testset.verify_freeze(out, dry=True))


def test_a_dry_manifest_next_to_a_real_freeze_is_not_an_extra_file(tmp_path, build):
    build(langs=("en",))
    testset.freeze(tmp_path / "testset", dry=True)
    build(freeze=True)  # a dry manifest never blocks the real build and freeze
    assert testset.verify_freeze(tmp_path / "testset") == []
