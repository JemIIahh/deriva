"""Shared fixtures: a tiny fixture source set, fake translators and a fake registry.

Nothing here touches the network or a model. Fixture prompts are bland placeholders, not
real red-teaming text.
"""

from __future__ import annotations

import pytest

from msd import testset
from msd.testset import RawItem, SourceSpec

HARM_A = "fix/harm-a"
HARM_B = "fix/harm-b"
BENIGN = "fix/benign"
LICENSES = {HARM_A: "mit", HARM_B: "cc-by-4.0", BENIGN: "CC BY 4.0 (fixture)"}

# Small sizes so the fixture pool is enough: 12 harmful and 9 benign-edgy per language.
TEST_SIZES = {
    "harmful": {"report": 8, "dev": 4},
    "benign_edgy": {"report": 6, "dev": 3},
}


def entry(name, used_for, license_, **extra):
    return {
        "name": name,
        "url": f"https://example.org/{name}",
        "license": license_,
        "languages": ["en"],
        "used_for": used_for,
        "verified_on": "2026-11-05",
        **extra,
    }


@pytest.fixture
def fixture_manifest():
    return [
        entry(HARM_A, "harmful_prompts", LICENSES[HARM_A]),
        entry(HARM_B, "harmful_prompts", LICENSES[HARM_B]),
        entry(BENIGN, "benign_edgy_prompts", LICENSES[BENIGN]),
    ]


def _rows(prefix, n):
    return [RawItem(key=f"{i:03d}", text=f"{prefix} number {i} please") for i in range(n)]


@pytest.fixture
def fixture_sources():
    """Three fake sources: harm-a (14 rows), harm-b (12 rows, one repeats a harm-a text with
    different case and spacing), benign (9 rows)."""
    harm_a = _rows("harm a question", 14)
    harm_b = _rows("harm b question", 11) + [RawItem(key="099", text="  HARM A Question  Number 3 PLEASE ")]
    benign = _rows("benign edgy question", 9)
    return {
        HARM_A: SourceSpec(HARM_A, "ha", {}, lambda raw_dir, rows=harm_a: list(rows)),
        HARM_B: SourceSpec(HARM_B, "hb", {}, lambda raw_dir, rows=harm_b: list(rows)),
        BENIGN: SourceSpec(BENIGN, "be", {}, lambda raw_dir, rows=benign: list(rows)),
    }


FIXTURE_NAMES = (HARM_A, HARM_B, BENIGN)


def fake_forward(texts, src, tgt):
    return [f"[{tgt}] {t}" for t in texts]


def fake_back(texts, src, tgt):
    return [t.removeprefix(f"[{src}] ") for t in texts]


@pytest.fixture
def fake_registry(monkeypatch):
    """Register 'fake-fwd' / 'fake-back' translators and an 'always' same_meaning by name."""
    monkeypatch.setitem(testset.TRANSLATORS, "fake-fwd", lambda: fake_forward)
    monkeypatch.setitem(testset.TRANSLATORS, "fake-back", lambda: fake_back)
    monkeypatch.setitem(testset.SAME_MEANING, "always", lambda: (lambda original, back: True))
    return None


@pytest.fixture
def build(tmp_path, fixture_manifest, fixture_sources, fake_registry):
    """Callable running `testset.run_build` on the fixture sources into tmp_path/testset.

    Keyword overrides replace any argument. The result carries `.out_dir` for convenience via
    the returned summary dict's own paths; tests use `tmp_path / "testset"` directly.
    """
    from msd.config import LANGS, NON_ENGLISH

    def run(**overrides):
        args = dict(
            out_dir=tmp_path / "testset",
            raw_root=tmp_path / "raw",
            manifest=fixture_manifest,
            langs=LANGS,
            source_names=FIXTURE_NAMES,
            seed=7,
            sizes=TEST_SIZES,
            sources=fixture_sources,
            translator_names={lang: "fake-fwd" for lang in NON_ENGLISH},
            back_translator_names={lang: "fake-back" for lang in NON_ENGLISH},
            same_meaning_name="exact",
        )
        args.update(overrides)
        return testset.run_build(**args)

    return run


# --- raw-file writers for loader and CLI tests -----------------------------------------------------


def write_csv_file(path, header, rows, bom=False):
    import csv
    import io

    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("﻿" if bom else "") + buf.getvalue(), encoding="utf-8", newline="")


def write_jsonl_file(path, rows):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def freeze_kwargs(tmp_path, fixture_sources):
    """Keyword arguments that make `testset.freeze` (real) check the fixture build's record."""
    return dict(
        raw_root=tmp_path / "raw",
        seed=7,
        source_names=FIXTURE_NAMES,
        sizes=TEST_SIZES,
        sources=fixture_sources,
    )


# --- live tests ------------------------------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Tests marked `live` spend money on a real API, so they run only when asked for: `pytest -m live`.

    A key in the shell (or an exported ANTHROPIC_API_KEY in a CI job) must never make a plain `pytest` pay.
    """
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="live test: spends money on a real API; run it with `pytest -m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
