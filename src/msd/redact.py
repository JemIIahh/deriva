"""What of a verdict may be public: `public_row`, and the one way rows are written to a public file (Task 19).

A verdict carries a prompt id, language, arm, alpha, kind, judge id, label and an excerpt of the completion. The
public row keeps exactly seven of those, in this order:

    prompt_id, lang, arm, alpha, label, judge_id, excerpt

and nothing else: no kind, no prompt text, no completion beyond the excerpt, which is capped at `PUBLIC_MAX_CHARS` (200)
characters. The cap is a promise made in the dataset card, so a larger `max_chars` is an error, not a choice.

Every string that goes into a public row (the excerpt, the judge id, the ids, language and arm) passes through the
same three steps, in this order, and the order is the point:

1. `clean_text`: the replacement character (U+FFFD) and lone surrogates are removed. They mean a decoding fault
   upstream and must not be published; they are also the cheapest way to hide a secret from a pattern, so they go
   first.
2. `public_text`: redaction. Token- and key-shaped strings, home paths and the value of any secret-looking environment
   variable become `***` (`msd.gen.redact_with`, the study's own redaction, plus the shapes below). The redaction runs on the text as it is AND on a
   copy without the characters a reader does not see (Unicode format characters such as U+200B, U+2060, U+00AD, U+FEFF,
   the bidi marks and the joiners, and control characters). A secret split by an invisible character stays copy-pasteable
   but defeats a pattern, so when the invisible-free copy changes under redaction, THAT redaction is published (the
   excerpt then loses its joiners, the price of the check); otherwise the redaction of the text as it is, so ordinary
   Devanagari, Arabic and Persian text keeps the joiners that are meaningful there.
3. `cut_excerpt`: the cut, and nothing else. It is on characters, never bytes, and never inside one written character: a
   base letter keeps its vowel signs and marks (Devanagari "कि", Arabic letters with harakat), a conjunct keeps its
   virama, an emoji keeps its joiners. When the cut would fall inside such a cluster it moves back to the start of it,
   so an excerpt may be a few characters short of the cap but is never mojibake and never ends on a half letter. The
   redaction comes before the cut so a cut cannot leave half a token behind.

A verdict from a test judge is refused, also when removable characters hide the word "fake".

This module reads nothing but the objects it is given and has no way to send anything anywhere.
"""

from __future__ import annotations

import functools
import json
import re
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

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
_KEPT_CONTROLS = range(0x09, 0x0E)  # tab, line feed, vertical tab, form feed, carriage return stay in text

# The shapes of what must not be published, as text patterns (ASCII rules for \w). `msd.export` compiles the same
# strings for bytes to scan every file it writes, so redaction and scan can never disagree about what a path or a key is.
_TAIL = r"""[^\s"'<>`)\]]*"""
HOME_PATH = r"(?<![\w.~-])/(?:Users|home)/[A-Za-z0-9._-]+(?:/" + _TAIL + ")?"
WINDOWS_PATH = r"(?<![A-Za-z])[A-Za-z]:[\\/]+(?:Users|Documents and Settings)(?:[\\/]+" + _TAIL + ")?"
SECRET_SHAPES: tuple[tuple[str, str], ...] = (
    ("a Hugging Face token", r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{16,}"),
    ("an API key", r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    ("a GitHub token", r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("an AWS key id", r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
    ("a private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|[\s\S]*)"),
)
_TEXT_PATTERNS = tuple(re.compile(p, re.ASCII) for p in (HOME_PATH, WINDOWS_PATH, *(p for _name, p in SECRET_SHAPES)))


class FakeJudgeError(ValueError):
    """A verdict whose judge id says "fake": canned labels of a test judge, never to be published."""


def clean_text(text: str) -> str:
    """`text` without the replacement character and lone surrogates: the first step for every public string."""
    return _UNWANTED.sub("", text)


def _is_invisible(code: int) -> bool:
    category = unicodedata.category(chr(code))
    return category == "Cf" or (category == "Cc" and code not in _KEPT_CONTROLS)


@functools.lru_cache(maxsize=1)
def _invisible() -> re.Pattern[str]:
    """Every format character (category Cf) and every control character except tab, line feed and carriage return."""
    points = [c for c in range(sys.maxunicode + 1) if _is_invisible(c)]
    ranges, start = [], points[0]
    for before, after in zip(points, points[1:] + [None]):
        if after != before + 1:
            ranges.append((start, before))
            start = after
    body = "".join(re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in ranges)
    return re.compile("[" + body + "]")


def strip_invisible(text: str) -> str:
    """`text` without the characters a reader does not see: format characters (zero-width space and joiners, word joiner,
    soft hyphen, byte-order mark, bidi marks, ...) and control characters other than tab, line feed and carriage return."""
    return text if text.isascii() and text.isprintable() else _invisible().sub("", text)


def _scrub(text: str, secrets: Sequence[str]) -> str:
    text = gen.redact_with(text, secrets)
    for pattern in _TEXT_PATTERNS:
        text = pattern.sub("***", text)
    return text


def public_text(text: str, secrets: Sequence[str] | None = None) -> str:
    """Any string for a public row: cleaned (`clean_text`), then redacted as described in the module docstring.
    `secrets` are the values of the secret environment variables (`gen.secret_env_values()`, read here when not given);
    a caller with many strings reads them once."""
    if secrets is None:
        secrets = gen.secret_env_values()
    text = clean_text(text)
    hidden = strip_invisible(text)
    scrubbed = _scrub(hidden, secrets)
    if scrubbed != hidden or hidden == text:
        return scrubbed
    return _scrub(text, secrets)  # nothing found without the invisible characters: the text as it is, joiners kept


def _glues(before: str, after: str) -> bool:
    """Does `after` belong to the same written character as `before`: a combining mark, a joiner, an emoji modifier,
    or whatever follows a virama or a joiner?"""
    if unicodedata.category(after).startswith("M") or after in (_ZWJ, _ZWNJ) or 0x1F3FB <= ord(after) <= 0x1F3FF:
        return True
    return before == _ZWJ or unicodedata.combining(before) == _VIRAMA_CLASS


def cut_excerpt(text: str, max_chars: int) -> str:
    """The first `max_chars` characters of `text`, moved back to the start of the cluster the cut would split. Only
    cuts: clean and redact first (`public_text`)."""
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
    """The public form of a verdict: exactly `PUBLIC_KEYS`, every text field cleaned and redacted, the excerpt cut to
    `max_chars` characters."""
    _check_max_chars(max_chars)
    secrets = gen.secret_env_values()
    judge = public_text(v.judge_id, secrets)
    if looks_fake(v.judge_id) or looks_fake(clean_text(v.judge_id)) or looks_fake(judge):
        raise FakeJudgeError(
            f"verdict {public_text(v.prompt_id, secrets)!r} was made by the test judge {judge!r}; its canned labels must "
            "never be published. Rejudge with a real judge"
        )
    return {
        "prompt_id": public_text(v.prompt_id, secrets),
        "lang": public_text(v.lang, secrets),
        "arm": public_text(v.arm, secrets),
        "alpha": v.alpha,
        "label": v.label,
        "judge_id": judge,
        "excerpt": cut_excerpt(public_text(v.excerpt, secrets), max_chars),
    }


def public_task_row(s: TaskScore) -> dict:
    """The public form of a task score: which item, which cell, right or wrong. No text of any kind."""
    secrets = gen.secret_env_values()
    return {"prompt_id": public_text(s.prompt_id, secrets), "lang": public_text(s.lang, secrets),
            "arm": public_text(s.arm, secrets), "alpha": s.alpha, "correct": s.correct}


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
