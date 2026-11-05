"""Text normalisation and train/dev/report leakage checks."""

from __future__ import annotations

import unicodedata
from typing import Iterable


class LeakageError(ValueError):
    """Raised when the same (normalised) text appears in two sets that must be disjoint."""


# Marks that change the bytes of a text but not what a reader sees as the same sentence:
# Arabic vowel marks (tashkeel U+064B-U+065F, superscript alef U+0670) and the stretching
# tatweel U+0640 are dropped; the Arabic question mark and the Devanagari danda / double danda
# are mapped to their ASCII look-alikes.
_ARABIC_MARKS = {cp: None for cp in (*range(0x064B, 0x0660), 0x0670, 0x0640)}
_PUNCT_MAP = {0x061F: "?", 0x0964: ".", 0x0965: "."}
_DROP_OR_MAP = {**_ARABIC_MARKS, **_PUNCT_MAP}


def norm_text(s: str) -> str:
    """Normalise a text for leakage checks.

    NFKC, then drop Unicode format characters (category Cf: ZWJ, ZWNJ, ZWSP, BOM, soft hyphen,
    Arabic letter mark), Arabic tashkeel and tatweel, map the Arabic question mark to "?" and
    the Devanagari danda (and double danda) to ".", casefold and collapse runs of whitespace.
    """
    folded = unicodedata.normalize("NFKC", s)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    return " ".join(folded.translate(_DROP_OR_MAP).casefold().split())


def _normed(texts: Iterable[str]) -> set[str]:
    return {n for n in (norm_text(t) for t in texts) if n}


def _check(name_a: str, a: set[str], name_b: str, b: set[str]) -> None:
    overlap = sorted(a & b)
    if overlap:
        shown = ", ".join(repr(t[:60]) for t in overlap[:3])
        raise LeakageError(f"{len(overlap)} text(s) shared between {name_a} and {name_b}, e.g. {shown}")


def assert_no_leakage(dev: Iterable[str], report: Iterable[str], train: Iterable[str] = ()) -> None:
    """Raise LeakageError if dev and report overlap, or either overlaps train."""
    dev_n, report_n, train_n = _normed(dev), _normed(report), _normed(train)
    _check("dev", dev_n, "report", report_n)
    _check("dev", dev_n, "train", train_n)
    _check("report", report_n, "train", train_n)
