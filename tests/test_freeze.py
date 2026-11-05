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


def test_freeze_refuses_a_partial_set_unless_allowed(tmp_path, build):
    build(langs=("en",))  # English only, no freeze
    out = tmp_path / "testset"
    with pytest.raises(ValueError, match="hinglish"):
        testset.freeze(out)
    assert not (out / "MANIFEST.json").exists()
    manifest = testset.freeze(out, require_langs=("en",))  # explicit partial freeze for a dry check
    assert sorted(manifest) == ["en/dev.jsonl", "en/report.jsonl"]


def test_freeze_is_deterministic_and_rebuild_is_byte_identical(tmp_path, build):
    build(freeze=True)
    out = tmp_path / "testset"
    first = (out / "MANIFEST.json").read_bytes()
    build(freeze=True)
    assert (out / "MANIFEST.json").read_bytes() == first


def test_build_freeze_flag_with_partial_langs_is_refused(tmp_path, build):
    with pytest.raises(ValueError, match="partial"):
        build(langs=("en",), freeze=True)
