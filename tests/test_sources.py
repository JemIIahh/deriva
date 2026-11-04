import copy
from datetime import date
from pathlib import Path

import pytest

from msd.sources import (
    LANG_CODES,
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    USED_FOR,
    load_sources,
    validate_entry,
    validate_sources,
)

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "sources.yaml"

GOOD = {
    "name": "Example/dataset",
    "url": "https://huggingface.co/datasets/Example/dataset",
    "license": "cc-by-4.0",
    "languages": ["en", "sw"],
    "used_for": "harmful_prompts",
    "verified_on": "2026-11-05",
}


def good(**overrides):
    entry = copy.deepcopy(GOOD)
    entry.update(overrides)
    return entry


# --- the real manifest ---------------------------------------------------------------------


def test_sources_manifest_complete():
    entries = load_sources(MANIFEST)  # raises ValueError on the first malformed entry
    assert entries, "data/sources.yaml has no entries"
    for entry in entries:
        for key in REQUIRED_KEYS:
            assert key in entry, f"{entry.get('name')!r} is missing {key!r}"
            assert entry[key] not in ("", None, []) or key == "languages", (entry["name"], key)
        assert set(entry) <= set(REQUIRED_KEYS) | set(OPTIONAL_KEYS)
        assert entry["used_for"] in USED_FOR
        assert set(entry["languages"]) <= set(LANG_CODES)
        assert entry["verified_on"] == "2026-11-05"


def test_manifest_records_every_required_candidate():
    names = " ".join(e["name"].casefold() for e in load_sources(MANIFEST))
    for needle in ("multijail", "xstest", "harmbench", "advbench", "belebele", "aya"):
        assert needle in names, f"no manifest entry mentions {needle!r}"


def test_manifest_names_are_unique():
    names = [e["name"] for e in load_sources(MANIFEST)]
    assert len(names) == len(set(names))


def test_excluded_entries_say_why():
    for entry in load_sources(MANIFEST):
        if entry["used_for"] == "excluded":
            assert entry.get("notes", "").strip(), f"{entry['name']!r} is excluded without a reason"


# --- the validator, on in-memory dicts ------------------------------------------------------


def test_good_entry_has_no_problems():
    assert validate_entry(GOOD) == []
    assert validate_entry(good(notes="gated; request access")) == []


def test_constants_match_the_spec():
    assert set(LANG_CODES) == {"en", "hi", "hinglish", "ar", "sw"}
    assert set(USED_FOR) == {"harmful_prompts", "benign_edgy_prompts", "task_qa",
                             "task_cross_check", "fine_tune_data", "excluded"}
    assert set(REQUIRED_KEYS) == {"name", "url", "license", "languages", "used_for", "verified_on"}
    assert set(OPTIONAL_KEYS) == {"notes"}


@pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
def test_missing_required_key_is_a_problem(key):
    entry = copy.deepcopy(GOOD)
    del entry[key]
    problems = validate_entry(entry)
    assert problems and any(key in p for p in problems)


@pytest.mark.parametrize("key", ["name", "url", "license", "used_for", "verified_on"])
@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_string_for_a_required_key_is_a_problem(key, empty):
    problems = validate_entry(good(**{key: empty}))
    assert problems and any(key in p for p in problems)


@pytest.mark.parametrize("key", ["name", "url", "license"])
def test_non_string_for_a_text_key_is_a_problem(key):
    assert validate_entry(good(**{key: None}))
    assert validate_entry(good(**{key: 7}))


def test_unknown_used_for_is_a_problem():
    problems = validate_entry(good(used_for="training"))
    assert problems and any("used_for" in p for p in problems)


@pytest.mark.parametrize("bad", [["en", "fr"], ["zh"], ["EN"], ["hi", ""], ["sw", None]])
def test_language_outside_the_five_codes_is_a_problem(bad):
    problems = validate_entry(good(languages=bad))
    assert problems and any("languages" in p for p in problems)


@pytest.mark.parametrize("bad", ["en", None, ("en",), {"en": 1}])
def test_languages_must_be_a_list(bad):
    assert validate_entry(good(languages=bad))


def test_duplicate_language_codes_are_a_problem():
    assert validate_entry(good(languages=["en", "en"]))


def test_empty_language_list_only_ok_for_excluded():
    assert validate_entry(good(languages=[]))
    assert validate_entry(good(languages=[], used_for="excluded", notes="no study language")) == []


@pytest.mark.parametrize("bad", ["2026-11-5", "11/05/2026", "2026-13-01", "2026-02-30",
                                 "20261105", "2026-11-05T00:00", "yesterday", date(2026, 11, 5), 20261105])
def test_verified_on_must_be_a_real_yyyy_mm_dd_string(bad):
    problems = validate_entry(good(verified_on=bad))
    assert problems and any("verified_on" in p for p in problems)


def test_unknown_keys_are_a_problem():
    problems = validate_entry(good(licence="mit"))
    assert problems and any("licence" in p for p in problems)


@pytest.mark.parametrize("notes", ["", "  ", 3])
def test_notes_when_present_must_be_a_non_empty_string(notes):
    assert validate_entry(good(notes=notes))


def test_excluded_entry_needs_a_reason_in_notes():
    assert validate_entry(good(used_for="excluded"))
    assert validate_entry(good(used_for="excluded", notes=""))
    assert validate_entry(good(used_for="excluded", notes="license unclear")) == []


def test_non_mapping_entry_is_a_problem():
    assert validate_entry("Example/dataset")
    assert validate_entry(None)


def test_url_must_be_http_or_https():
    assert validate_entry(good(url="huggingface.co/datasets/x"))
    assert validate_entry(good(url="ftp://example.com/x"))
    assert validate_entry(good(url="http://example.com/x")) == []


# --- list-level validation and loading -------------------------------------------------------


def test_validate_sources_reports_duplicate_names_and_entry_problems():
    problems = validate_sources([GOOD, good(), good(name="Other", used_for="nope")])
    assert any("duplicate" in p.lower() for p in problems)
    assert any("Other" in p and "used_for" in p for p in problems)
    assert validate_sources([GOOD, good(name="Other")]) == []


def test_load_sources_raises_on_a_bad_file(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text(
        "- name: X\n  url: https://example.com\n  license: ''\n  languages: [en]\n"
        "  used_for: harmful_prompts\n  verified_on: '2026-11-05'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="license"):
        load_sources(bad)


def test_load_sources_rejects_an_unquoted_yaml_date(tmp_path):
    # YAML turns an unquoted 2026-11-05 into a date object; the manifest must quote it.
    bad = tmp_path / "sources.yaml"
    bad.write_text(
        "- name: X\n  url: https://example.com\n  license: mit\n  languages: [en]\n"
        "  used_for: harmful_prompts\n  verified_on: 2026-11-05\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verified_on"):
        load_sources(bad)


def test_load_sources_rejects_a_non_list_document(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text("name: X\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sources(bad)
