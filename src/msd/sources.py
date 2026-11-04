"""Validation and loading for the public-data source manifest (data/sources.yaml)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from msd.config import LANGS

LANG_CODES: tuple[str, ...] = LANGS
REQUIRED_KEYS: tuple[str, ...] = ("name", "url", "license", "languages", "used_for", "verified_on")
OPTIONAL_KEYS: tuple[str, ...] = ("notes",)
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

    if "notes" in entry and not _is_text(entry["notes"]):
        problems.append(f"notes, when present, must be a non-empty string, got {entry['notes']!r}")
    if used_for == "excluded" and not _is_text(entry.get("notes")):
        problems.append("an excluded entry needs its reason in notes")

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
            if name in seen:
                problems.append(f"{label}: duplicate name")
            seen.add(name)
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
