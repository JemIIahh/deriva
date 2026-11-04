import copy
import re
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
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["verified_on"])
        assert isinstance(entry.get("redistribute", True), bool)


REQUIRED_CANDIDATES = (
    "DAMO-NLP-SG/MultiJail",  # the MultiJail-style multilingual jailbreak set
    "Paul/XSTest",
    "centerforaisafety/HarmBench",
    "llm-attacks/llm-attacks (AdvBench)",
    "facebook/belebele",
    "CohereLabs/aya_dataset",
)


def test_manifest_records_every_required_candidate():
    by_name = {e["name"]: e for e in load_sources(MANIFEST)}
    for name in REQUIRED_CANDIDATES:
        assert name in by_name, f"no manifest entry named {name!r}"
    required = [by_name[n] for n in REQUIRED_CANDIDATES]
    assert not all(e["used_for"] == "excluded" for e in required)
    # the manifest has to leave something usable for every role the study needs
    roles = {e["used_for"] for e in by_name.values()}
    assert {"harmful_prompts", "benign_edgy_prompts", "task_cross_check", "fine_tune_data"} <= roles
    # each spec-named source is actually usable, not parked in `excluded`
    for entry in required:
        assert entry["used_for"] != "excluded", entry["name"]


def test_belebele_is_evaluation_only():
    belebele = next(e for e in load_sources(MANIFEST) if e["name"] == "facebook/belebele")
    assert belebele["license"] == "cc-by-sa-4.0"
    assert belebele["used_for"] == "task_cross_check"
    assert belebele["redistribute"] is False


def test_no_usable_manifest_entry_breaks_the_license_rule():
    # the same rule validate_entry enforces, asserted on the real file
    for entry in load_sources(MANIFEST):
        if entry["used_for"] == "excluded":
            continue
        problems = [p for p in validate_entry(entry) if "license" in p]
        assert problems == [], entry["name"]


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
    assert set(OPTIONAL_KEYS) == {"notes", "redistribute"}


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


# --- the license rule: only clean licenses may be used; share-alike only if evaluation-only -------

NOTES = "reason recorded here"


@pytest.mark.parametrize(
    "bad",
    [
        "cc-by-nc-4.0",
        "CC BY-NC 4.0",
        "cc-by-nc-sa-4.0",
        "CC-BY-NC",
        "Creative Commons Attribution-NonCommercial 4.0 International",
        "non-commercial research use only",
        "noncommercial",
        "conflicting: front matter apache-2.0; card body CC BY-NC-SA 4.0",
        "none stated on the card",
        "none",
        "No license",
        "not stated",
        "other",
        "other (license_name: see-upstream)",
        "unknown",
        "unclear",
        "missing",
        "unspecified",
    ],
)
def test_unusable_license_on_a_usable_entry_is_a_problem(bad):
    problems = validate_entry(good(license=bad))
    assert problems and any("license" in p for p in problems), bad
    # the same string is fine on an excluded entry, which is where such sources belong
    assert validate_entry(good(license=bad, used_for="excluded", notes=NOTES)) == [], bad


@pytest.mark.parametrize("bad", ["cc-by-nc-4.0", "cc-by-nc-sa-4.0", "CC BY-NC 4.0"])
def test_non_commercial_stays_an_error_even_when_evaluation_only(bad):
    assert validate_entry(good(license=bad, redistribute=False))


@pytest.mark.parametrize(
    "bad", ["conflicting: apache-2.0 vs cc-by-nc", "none stated", "other", "unknown"]
)
def test_conflicting_or_unknown_stays_an_error_even_when_evaluation_only(bad):
    assert validate_entry(good(license=bad, redistribute=False))


@pytest.mark.parametrize(
    "sa",
    [
        "cc-by-sa-4.0",
        "CC BY-SA 4.0",
        "CC-BY-SA-3.0",
        "cc by sa 4.0",
        "Creative Commons Attribution-ShareAlike 4.0 International",
    ],
)
def test_share_alike_needs_redistribute_false(sa):
    assert any("license" in p for p in validate_entry(good(license=sa)))
    assert any("license" in p for p in validate_entry(good(license=sa, redistribute=True)))
    assert validate_entry(good(license=sa, redistribute=False)) == []


def test_belebele_shaped_entry_passes_only_with_redistribute_false():
    belebele = good(
        name="facebook/belebele",
        license="cc-by-sa-4.0",
        languages=["en", "hi", "ar", "sw"],
        used_for="task_cross_check",
    )
    assert validate_entry({**belebele, "redistribute": False}) == []
    assert validate_entry(belebele)
    assert validate_entry({**belebele, "redistribute": True})


@pytest.mark.parametrize("bad", ["false", "no", 0, 1, None, [], "False"])
def test_redistribute_must_be_a_bool(bad):
    problems = validate_entry(good(redistribute=bad))
    assert problems and any("redistribute" in p for p in problems)


@pytest.mark.parametrize("value", [True, False])
def test_redistribute_bool_is_accepted(value):
    assert validate_entry(good(redistribute=value)) == []


@pytest.mark.parametrize(
    "ok",
    [
        "mit",
        "MIT",
        "MIT License",
        "apache-2.0",
        "Apache License 2.0",
        "cc-by-4.0",
        "CC BY 4.0",
        "cc0-1.0",
        "odc-by",
        "Open Data Commons Attribution License (ODC-By) v1.0",
    ],
)
def test_clean_licenses_pass(ok):
    assert validate_entry(good(license=ok)) == []


@pytest.mark.parametrize("weird", ["made-up-license-1.0", "proprietary", "see website"])
def test_unrecognised_license_is_a_problem_on_a_usable_entry(weird):
    problems = validate_entry(good(license=weird))
    assert problems and any("license" in p for p in problems)


def test_license_matcher_ignores_case_and_spacing():
    assert validate_entry(good(license="  CC   BY-NC   4.0 "))
    assert validate_entry(good(license="CC_BY_SA_4.0"))
    assert validate_entry(good(license="CC_BY_SA_4.0", redistribute=False)) == []


def test_duplicate_names_are_matched_ignoring_case_and_surrounding_space():
    problems = validate_sources([good(name="Org/Data"), good(name="  org/data ")])
    assert any("duplicate" in p.lower() for p in problems)
    assert validate_sources([good(name="Org/Data"), good(name="Org/Other")]) == []


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
