"""Frozen record types and JSONL IO for the multilingual safety drift study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TypeVar

KINDS: frozenset[str] = frozenset({"harmful", "benign_edgy", "task"})
SPLITS: frozenset[str] = frozenset({"dev", "report"})
LABELS: frozenset[str] = frozenset({"complied", "refused", "no_response", "judge_error"})

T = TypeVar("T")


def _require(field_name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"unknown {field_name} {value!r}; expected one of {sorted(allowed)}")


@dataclass(frozen=True)
class Prompt:
    id: str
    lang: str
    kind: str
    text: str
    split: str
    source: str
    license: str
    gold: str | None = None

    def __post_init__(self) -> None:
        _require("kind", self.kind, KINDS)
        _require("split", self.split, SPLITS)


@dataclass(frozen=True)
class Generation:
    prompt_id: str
    arm: str
    alpha: float | None
    text: str
    finish_reason: str


@dataclass(frozen=True)
class Verdict:
    prompt_id: str
    arm: str
    alpha: float | None
    lang: str
    kind: str
    judge_id: str
    label: str
    excerpt: str

    def __post_init__(self) -> None:
        _require("kind", self.kind, KINDS)
        _require("label", self.label, LABELS)


@dataclass(frozen=True)
class TaskScore:
    prompt_id: str
    arm: str
    alpha: float | None
    lang: str
    correct: bool


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    """Write dataclass rows, one JSON object per line, as raw UTF-8 (no \\u escapes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, cls: type[T]) -> list[T]:
    """Read a JSONL file back into instances of the frozen dataclass `cls`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return [cls(**json.loads(line)) for line in fh if line.strip()]
