"""scripts/build_testset.py end to end on tiny local raw files (no network, no model)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from msd import testset
from msd.schema import Prompt, read_jsonl

from conftest import write_csv_file, write_jsonl_file

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
BUILD = [sys.executable, str(SCRIPTS / "build_testset.py")]
VERIFY = [sys.executable, str(SCRIPTS / "verify_freeze.py")]


def _raw_dir(raw, name):
    return raw / testset.entry_dirname(name)


@pytest.fixture
def raw(tmp_path):
    """A few bland rows for every source (Nemotron too, which is not a default), in the real file formats."""
    root = tmp_path / "raw"
    d = lambda name, file: _raw_dir(root, name) / file  # noqa: E731
    write_csv_file(
        d("centerforaisafety/HarmBench", "harmbench_behaviors_text_all.csv"),
        ["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"],
        [[f"hb behavior {i}", "standard", "c", "", "", f"hb_{i}"] for i in range(6)],
    )
    write_csv_file(
        d("llm-attacks/llm-attacks (AdvBench)", "harmful_behaviors.csv"),
        ["goal", "target"],
        [[f"adv goal {i}", "t"] for i in range(6)],
    )
    write_csv_file(
        d("DAMO-NLP-SG/MultiJail", "MultiJail.csv"),
        ["id", "source", "tags", "en"],
        [[str(i), "s", "[]", f"mj english {i}"] for i in range(6)],
        bom=True,
    )
    write_jsonl_file(
        d("CohereLabs/aya_redteaming", "aya_eng.jsonl"),
        [{"prompt": f"aya english {i}", "language": "English"} for i in range(6)],
    )
    write_jsonl_file(
        d("nvidia/Nemotron-Safety-Guard-Dataset-v3", "en_test.jsonl"),
        [{"id": f"n{i}", "prompt": f"nemo english {i}", "prompt_label": "unsafe", "tag": "generic"} for i in range(6)],
    )
    write_csv_file(
        d("Paul/XSTest", "xstest_prompts.csv"),
        ["id", "prompt", "type", "label"],
        [[str(i), f"xs safe {i}", "t", "safe"] for i in range(6)],
    )
    write_csv_file(
        d("bench-llm/or-bench", "or-bench-hard-1k.csv"),
        ["prompt", "category"],
        [[f"or hard {i}", "x"] for i in range(6)],
    )
    return root


def run(args, raw, tmp_path):
    cmd = [*BUILD, "--raw", str(raw), "--out", str(tmp_path / "out"), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_english_only_build_needs_no_translator_and_records_shortfalls(raw, tmp_path):
    res = run(["--langs", "en"], raw, tmp_path)
    assert res.returncode == 0, res.stderr
    out = tmp_path / "out"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.jsonl")) == ["en/dev.jsonl", "en/report.jsonl"]
    assert "SHORTFALL harmful/report" in res.stdout  # 24 harmful rows exist, 450 are wanted
    summary = json.loads((tmp_path / "out_build_summary.json").read_text(encoding="utf-8"))
    assert summary["english"]["shortfalls"] and summary["pool"]["duplicates_removed_total"] == 0
    rows = read_jsonl(out / "en" / "report.jsonl", Prompt) + read_jsonl(out / "en" / "dev.jsonl", Prompt)
    assert len(rows) == 24 + 12  # everything that exists, nothing padded
    assert {p.license for p in rows} == {e["license"] for e in _manifest_entries() if e["name"] in testset.DEFAULT_SOURCES}


def _manifest_entries():
    from msd.sources import load_sources

    return load_sources(Path(__file__).resolve().parent.parent / "data" / "sources.yaml")


def test_full_identity_dry_run_freezes_and_verifies(raw, tmp_path):
    res = run(["--translator", "identity", "--back-translator", "identity", "--same-meaning", "exact", "--freeze"], raw, tmp_path)
    assert res.returncode == 0, res.stderr
    out = tmp_path / "out"
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest) == 10
    assert subprocess.run([*VERIFY, "--dir", str(out)], capture_output=True, text=True).returncode == 0
    (out / "sw" / "dev.jsonl").write_bytes((out / "sw" / "dev.jsonl").read_bytes() + b"\n{}\n")
    assert subprocess.run([*VERIFY, "--dir", str(out)], capture_output=True, text=True).returncode != 0


def test_per_language_runs_merge_into_one_summary(raw, tmp_path):
    base = ["--translator", "identity", "--back-translator", "identity", "--same-meaning", "exact"]
    assert run([*base, "--langs", "hi"], raw, tmp_path).returncode == 0
    assert run([*base, "--langs", "sw"], raw, tmp_path).returncode == 0
    summary = json.loads((tmp_path / "out_build_summary.json").read_text(encoding="utf-8"))
    assert set(summary["languages"]) == {"en", "hi", "sw"}
    assert summary["languages"]["hi"]["translator"] == "identity"


def test_a_forbidden_source_is_refused_with_exit_2(raw, tmp_path):
    res = run(["--langs", "en", "--source", "Paul/XSTest", "--source", "facebook/belebele"], raw, tmp_path)
    assert res.returncode == 2
    assert "belebele" in res.stderr and not (tmp_path / "out").exists()


def test_freeze_of_a_partial_build_is_refused_but_a_dry_freeze_is_labelled(raw, tmp_path):
    res = run(["--langs", "en", "--freeze"], raw, tmp_path)
    assert res.returncode == 2 and "partial" in res.stderr
    assert not (tmp_path / "out" / "MANIFEST.json").exists()
    dry = run(["--langs", "en", "--dry-freeze"], raw, tmp_path)
    assert dry.returncode == 0 and "NOT the real freeze" in dry.stdout
    assert subprocess.run([*VERIFY, "--dir", str(tmp_path / "out")], capture_output=True, text=True).returncode == 0
    only = subprocess.run([*BUILD, "--out", str(tmp_path / "out"), "--freeze-only"], capture_output=True, text=True)
    assert only.returncode == 2 and "hinglish" in only.stderr  # the real freeze needs all five


def test_unknown_translator_and_missing_raw_files_exit_2(raw, tmp_path):
    bad = run(["--translator", "hi=claude-opus"], raw, tmp_path)
    assert bad.returncode == 2 and "claude-opus" in bad.stderr
    missing = subprocess.run([*BUILD, "--raw", str(tmp_path / "no-raw"), "--out", str(tmp_path / "o2"), "--langs", "en"],
                             capture_output=True, text=True)
    assert missing.returncode == 2 and "curl" in missing.stderr


def test_print_fetch_lists_the_curl_commands(tmp_path):
    res = subprocess.run([*BUILD, "--raw", str(tmp_path / "raw"), "--print-fetch"], capture_output=True, text=True)
    assert res.returncode == 0
    lines = res.stdout.strip().splitlines()
    assert len(lines) == len(testset.DEFAULT_SOURCES) and all(line.startswith("curl ") for line in lines)
