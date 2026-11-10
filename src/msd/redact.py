"""What of a verdict may be public: `public_row`, and the one way rows are written to a public file (Task 19).

A verdict carries a prompt id, language, arm, alpha, kind, judge id, label and an excerpt of the completion. The
public row keeps exactly seven of those, in this order:

    prompt_id, lang, arm, alpha, label, judge_id, excerpt

and nothing else: no kind, no prompt text, no completion beyond the excerpt, which is capped at `PUBLIC_MAX_CHARS` (200)
characters. The cap is a promise made in the dataset card, so a larger `max_chars` is an error, not a choice.

The excerpt is cut on characters, never on bytes, and never inside one written character: a base letter keeps its
vowel signs and marks (Devanagari "कि", Arabic letters with harakat), a conjunct keeps its virama, an emoji keeps its
joiners. When the cut would fall inside such a cluster it moves back to the start of it, so an excerpt may be a few
characters short of the cap but is never mojibake and never ends on a half letter. The replacement character
(U+FFFD) and lone surrogates are removed from it: they mean a decoding fault upstream and must not be published.

Anything token-shaped in the excerpt or the judge id goes through the same redaction every other record of the study
uses (`msd.gen._redact`: `hf_...` tokens and the value of any secret-looking environment variable become `***`), before
the cut, so a cut can never leave half a token behind. A verdict from a test judge is refused: its canned labels must
never be published.

This module reads nothing but the objects it is given and has no way to send anything anywhere.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Mapping

from msd import gen
from msd.gaterun import looks_fake
from msd.schema import TaskScore, Verdict

PUBLIC_KEYS: tuple[str, ...] = ("prompt_id", "lang", "arm", "alpha", "label", "judge_id", "excerpt")
TASK_KEYS: tuple[str, ...] = ("prompt_id", "lang", "arm", "alpha", "correct")
PUBLIC_MAX_CHARS = 200

# Characters that never reach a public file: lone surrogates and the replacement character (written as code points,
# not typed, so no editing tool can turn the escape into the raw character).
_UNWANTED = re.compile("[" + chr(0xD800) + "-" + chr(0xDFFF) + chr(0xFFFD) + "]")
_ZWJ, _ZWNJ = chr(0x200D), chr(0x200C)
_VIRAMA_CLASS = 9  # canonical combining class of the sign that joins two consonants (Devanagari, Bengali, ...)
_LINE_BREAKS = {chr(code): "\\u%04x" % code for code in (0x85, 0x2028, 0x2029)}  # str.splitlines() splits at these


class FakeJudgeError(ValueError):
    """A verdict whose judge id says "fake": canned labels of a test judge, never to be published."""


def redact_text(text: str) -> str:
    """The study's existing redaction (`msd.gen._redact`) under a public name."""
    return gen._redact(text)


def _glues(before: str, after: str) -> bool:
    """Does `after` belong to the same written character as `before`: a combining mark, a joiner, an emoji modifier,
    or whatever follows a virama or a joiner?"""
    if unicodedata.category(after).startswith("M") or after in (_ZWJ, _ZWNJ) or 0x1F3FB <= ord(after) <= 0x1F3FF:
        return True
    return before == _ZWJ or unicodedata.combining(before) == _VIRAMA_CLASS


def cut_excerpt(text: str, max_chars: int) -> str:
    """The first `max_chars` characters of `text`, moved back to the start of the cluster the cut would split."""
    text = _UNWANTED.sub("", text)
    if len(text) <= max_chars:
        return text
    cut = max_chars
    while cut > 0 and _glues(text[cut - 1], text[cut]):
        cut -= 1
    return text[:cut]


def _check_max_chars(max_chars: Any) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 0 <= max_chars <= PUBLIC_MAX_CHARS:
        raise ValueError(
            f"max_chars must be a whole number from 0 to {PUBLIC_MAX_CHARS} (the cap the dataset card promises), "
            f"got {max_chars!r}"
        )


def public_row(v: Verdict, max_chars: int = PUBLIC_MAX_CHARS) -> dict:
    """The public form of a verdict: exactly `PUBLIC_KEYS`, the excerpt cut to `max_chars` characters."""
    _check_max_chars(max_chars)
    if looks_fake(v.judge_id):
        raise FakeJudgeError(
            f"verdict {v.prompt_id!r} was made by the test judge {redact_text(v.judge_id)!r}; its canned labels must "
            "never be published. Rejudge with a real judge"
        )
    return {
        "prompt_id": v.prompt_id,
        "lang": v.lang,
        "arm": v.arm,
        "alpha": v.alpha,
        "label": v.label,
        "judge_id": redact_text(v.judge_id),
        "excerpt": cut_excerpt(redact_text(v.excerpt), max_chars),
    }


def public_task_row(s: TaskScore) -> dict:
    """The public form of a task score: which item, which cell, right or wrong. No text of any kind."""
    return {"prompt_id": s.prompt_id, "lang": s.lang, "arm": s.arm, "alpha": s.alpha, "correct": s.correct}


def public_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """`rows` as JSON Lines in UTF-8: one object per physical line, raw characters (no `\\u` escapes for letters).

    U+0085, U+2028 and U+2029 are written as escapes: a raw one inside a string would cut the line in two for any
    reader that uses `str.splitlines()`. The escapes decode to the very same string."""
    out = []
    for row in rows:
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        for char, escape in _LINE_BREAKS.items():
            line = line.replace(char, escape)
        out.append(line + "\n")
    return "".join(out).encode("utf-8")
