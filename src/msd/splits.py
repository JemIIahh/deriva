"""Text normalisation and train/dev/report leakage checks."""

from __future__ import annotations

import unicodedata
from typing import Iterable


class LeakageError(ValueError):
    """Raised when the same (normalised) text appears in two sets that must be disjoint."""


def norm_text(s: str) -> str:
    """NFKC-normalise, casefold and collapse runs of whitespace."""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


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
