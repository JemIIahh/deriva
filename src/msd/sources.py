"""Validation and loading for the public-data source manifest (data/sources.yaml).

Optional key `redistribute` (bool, default true): false means the source is evaluation only,
the model is scored on it and only scores are published; its items are not copied into our
released datasets. Required for any usable entry whose license is share-alike.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from msd.config import LANGS

LANG_CODES: tuple[str, ...] = LANGS
REQUIRED_KEYS: tuple[str, ...] = ("name", "url", "license", "languages", "used_for", "verified_on")
OPTIONAL_KEYS: tuple[str, ...] = ("notes", "redistribute")
USED_FOR: frozenset[str] = frozenset(
    {
        "harmful_prompts",
        "benign_edgy_prompts",
        "task_qa",
        "task_cross_check",
        "fine_tune_data",
        "excluded",
    }
)

_TEXT_KEYS = ("name", "url", "license", "used_for", "verified_on")
_DATE_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}")

# --- the license rule -------------------------------------------------------------------------
#
# A usable (non-`excluded`) entry must carry a clean, recognised license. The matcher works on
# the license string lower-cased, trimmed, with runs of whitespace/underscores turned into "-",
# so "CC BY-SA 4.0" and "cc_by_sa_4.0" both become "cc-by-sa-4.0". Checks, in order:
#   1. DENY non-commercial: the token "nc" or "non-commercial"/"noncommercial" (cc-by-nc-4.0).
#   2. DENY unclear: contains "conflict", or the tokens none/other/unknown/unclear/missing/
#      unspecified/unstated, or "not-stated", "no-license", "see-upstream", "n/a".
#   3. SHARE-ALIKE: contains "-sa" (not followed by a letter) or "share". An error unless the
#      entry says `redistribute: false` (evaluation only: scores are published, items are not
#      copied into our released datasets, so the share-alike condition is never triggered).
#   4. ALLOW: whatever is left must start with a recognised family (mit, apache, bsd, isc,
#      unlicense, cc0, any cc-by version, odc-by, "creative commons attribution", public domain).
# Anything that fails 1 or 2, or matches nothing in 4, is an error for a usable entry; such
# sources belong under `used_for: excluded` with the reason in notes. This is a convention check,
# not legal advice: it cannot prove that a recognised license is the right one for the data.
_NON_COMMERCIAL = re.compile(r"(?<![a-z0-9])nc(?![a-z0-9])|non-?commercial")
_UNCLEAR = re.compile(
    r"conflict|(?<![a-z])(?:none|other|unknown|unclear|missing|unspecified|unstated)(?![a-z])"
    r"|not-stated|no-licen[cs]e|see-upstream|n/a"
)
_SHARE_ALIKE = re.compile(r"-sa(?![a-z])|share")
_PERMISSIVE = re.compile(
    r"^(?:(?:mit|apache|bsd|isc)(?![a-z])|unlicense|cc0|cc-by(?![a-z])|odc-by"
    r"|open-data-commons-attribution|creative-commons-attribution|public-domain)"
)


def _normalise_license(text: str) -> str:
    return re.sub(r"[\s_]+", "-", text.strip().casefold())


def _license_problems(text: str, redistribute: object) -> list[str]:
    """Problems with the license of a usable (non-excluded) entry."""
    norm = _normalise_license(text)
    if _NON_COMMERCIAL.search(norm):
        return [f"license {text!r} is non-commercial; only an excluded entry may use it"]
    if _UNCLEAR.search(norm):
        return [f"license {text!r} is conflicting, missing, other or unknown; exclude the entry"]
    if _SHARE_ALIKE.search(norm):
        if redistribute is False:
            return []
        return [
            f"license {text!r} is share-alike; set redistribute: false (evaluation only) "
            "or exclude the entry"
        ]
    if not _PERMISSIVE.search(norm):
        return [
            f"license {text!r} is not a recognised permissive license; read the card, then "
            "extend _PERMISSIVE in src/msd/sources.py or exclude the entry"
        ]
    return []


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_date(value: object) -> bool:
    # Must be a string: YAML turns an unquoted 2026-11-05 into a date object, which is
    # rejected so the file always holds the literal text.
    if not isinstance(value, str) or not _DATE_SHAPE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate_entry(entry: Any) -> list[str]:
    """Return a list of human-readable problems with one manifest entry (empty if fine)."""
    if not isinstance(entry, Mapping):
        return [f"entry must be a mapping, got {type(entry).__name__}"]

    problems: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in entry:
            problems.append(f"missing required key {key!r}")
    for key in entry:
        if key not in REQUIRED_KEYS and key not in OPTIONAL_KEYS:
            problems.append(f"unknown key {key!r}")

    for key in _TEXT_KEYS:
        if key in entry and not _is_text(entry[key]):
            problems.append(f"{key} must be a non-empty string, got {entry[key]!r}")

    used_for = entry.get("used_for")
    if _is_text(used_for) and used_for not in USED_FOR:
        problems.append(f"used_for {used_for!r} not one of {sorted(USED_FOR)}")

    url = entry.get("url")
    if _is_text(url) and not url.startswith(("http://", "https://")):
        problems.append(f"url must start with http:// or https://, got {url!r}")

    verified_on = entry.get("verified_on")
    if _is_text(verified_on) and not _valid_date(verified_on):
        problems.append(f"verified_on must be a real YYYY-MM-DD date, got {verified_on!r}")

    if "languages" in entry:
        languages = entry["languages"]
        if not isinstance(languages, list):
            problems.append(f"languages must be a list, got {languages!r}")
        else:
            bad = [code for code in languages if code not in LANG_CODES]
            if bad:
                problems.append(f"languages has codes outside {list(LANG_CODES)}: {bad!r}")
            if len(set(map(repr, languages))) != len(languages):
                problems.append(f"languages has duplicates: {languages!r}")
            if not languages and used_for != "excluded":
                problems.append("languages may only be empty for an excluded entry")

    if "redistribute" in entry and not isinstance(entry["redistribute"], bool):
        problems.append(f"redistribute must be true or false, got {entry['redistribute']!r}")

    if "notes" in entry and not _is_text(entry["notes"]):
        problems.append(f"notes, when present, must be a non-empty string, got {entry['notes']!r}")
    if used_for == "excluded" and not _is_text(entry.get("notes")):
        problems.append("an excluded entry needs its reason in notes")

    if used_for != "excluded" and _is_text(entry.get("license")):
        problems.extend(_license_problems(entry["license"], entry.get("redistribute", True)))

    return problems


def validate_sources(entries: Sequence[Any]) -> list[str]:
    """Validate every entry and the manifest as a whole; problems are prefixed with the entry name."""
    problems: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        name = entry.get("name") if isinstance(entry, Mapping) else None
        label = name if _is_text(name) else f"#{index}"
        problems.extend(f"{label}: {p}" for p in validate_entry(entry))
        if _is_text(name):
            key = name.strip().casefold()
            if key in seen:
                problems.append(f"{label}: duplicate name")
            seen.add(key)
    return problems


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the manifest; raise ValueError listing every problem if it is malformed."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list of entries, got {type(data).__name__}")
    problems = validate_sources(data)
    if problems:
        raise ValueError(f"{path} is malformed:\n  " + "\n  ".join(problems))
    return data
