"""What of a verdict may be public: `public_row`, and the one way rows are written to a public file (Task 19).

A verdict carries a prompt id, language, arm, alpha, kind, judge id, label and an excerpt of the completion. The
public row keeps exactly seven of those, in this order:

    prompt_id, lang, arm, alpha, label, judge_id, excerpt

and nothing else: no kind, no prompt text, no completion beyond the excerpt, which is capped at `PUBLIC_MAX_CHARS` (200)
characters. The cap is a promise made in the dataset card, so a larger `max_chars` is an error, not a choice.

Every string that goes into a public row (the excerpt, the judge id, the ids, language and arm) passes through the
same steps, in this order, and the order is the point:

1. `clean_text`: the replacement character (U+FFFD) and lone surrogates are removed. They mean a decoding fault
   upstream and must not be published; they are also the cheapest way to hide a secret from a pattern, so they go
   first.
2. `public_text`: redaction. Token- and key-shaped strings, home paths and the value of any secret-looking environment
   variable become `***` (`msd.gen.redact_with`, the study's own redaction, plus the shapes below). The redaction runs on
   the text as it is AND on copies without what a reader does not see (`msd.invisible.hidden_view`: format characters,
   the rest of Unicode's Default_Ignorable_Code_Point set such as U+034F, variation selectors, Hangul fillers and tag
   characters, and combining marks on ASCII letters) and under NFKC and NFKD (fullwidth letters, also with a combining
   mark on each). A secret split by an invisible character stays copy-pasteable but defeats a pattern, so when a copy
   changes under redaction, THAT redaction is published (the excerpt then loses its joiners and marks, the price of the
   check); otherwise the redaction of the text as it is, so ordinary Devanagari, Arabic and Persian text keeps the joiners
   and marks that are meaningful there. For an excerpt that is, or may be, cut (see below) the whole excerpt is redacted
   first, so a value that is all there (also one of several words, such as a pass phrase) is matched whole.
   Known limits: text that is merely ENCODED (percent-encoded, a literal backslash-u written out in a string, base64) is
   not decoded and so not recognised; and a secret that the upstream cut split is not a whole secret any more, so for an
   environment value of SEVERAL WORDS cut at the boundary the words before the cut can stay (only the word the cut split
   is dropped); a value without spaces, a token and a path leave nothing behind.
3. For an excerpt that is, or may be, cut: the last whitespace-delimited chunk of the redacted text is dropped. What the
   cut split lives in that chunk (a fragment of a word, a token, a path), half of a secret is not recognised by a pattern
   that needs the whole of it, and so the fragment has to go. The result is redacted once more (it must be a fixed point
   of the redaction) and ends with an ellipsis.
4. `cut_excerpt`: the cut, and nothing else. It is on characters, never bytes, and never inside one written character: a
   base letter keeps its vowel signs and marks (Devanagari "कि", Arabic letters with harakat), a conjunct keeps its
   virama, an emoji keeps its joiners. When the cut would fall inside such a cluster it moves back to the start of it
   (by at most `_MAX_BACKOFF` characters, so a long chain of marks cannot empty an excerpt), so an excerpt may be a few
   characters short of the cap but is never mojibake and never ends on a half letter.

The excerpt arrives already cut: the judge stage stores `completion[:EXCERPT_CHARS]` (`msd.judge`), so on real data the
cut happens BEFORE this module sees the text, in the middle of whatever stood at character 200, and `cut_excerpt` has
nothing left to cut. An excerpt of `EXCERPT_CHARS` characters or more, or one this module has to shorten, is therefore
treated as truncated: its last whitespace-delimited chunk (a fragment of a word, a token, a path or a cluster) is dropped
whole and an ellipsis (U+2026) ends it so a reader can see it was cut, all within the cap. All five study languages are
written with spaces; an excerpt with no whitespace at all is published as the ellipsis alone, nothing being known about
where a word ended. A shorter excerpt is the whole of the completion and is published as it is.

A verdict from a test judge is refused, also when the word "fake" is hidden by a decoding fault, invisible
characters or fullwidth letters (`msd.gaterun.looks_fake`, the one check every judge id goes through).

This module reads nothing but the objects it is given and has no way to send anything anywhere.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from msd import gen
from msd.gaterun import looks_fake
from msd.invisible import hidden_view, strip_decoding_faults, strip_invisible  # noqa: F401  (re-exported: `msd.redact` is where callers look)
from msd.judge import EXCERPT_CHARS
from msd.schema import TaskScore, Verdict

PUBLIC_KEYS: tuple[str, ...] = ("prompt_id", "lang", "arm", "alpha", "label", "judge_id", "excerpt")
TASK_KEYS: tuple[str, ...] = ("prompt_id", "lang", "arm", "alpha", "correct")
PUBLIC_MAX_CHARS = 200

_ZWJ, _ZWNJ = chr(0x200D), chr(0x200C)
_VIRAMA_CLASS = 9  # canonical combining class of the sign that joins two consonants (Devanagari, Bengali, ...)
_LINE_BREAKS = {chr(code): "\\u%04x" % code for code in (0x85, 0x2028, 0x2029)}  # str.splitlines() splits at these
_ELLIPSIS = chr(0x2026)  # ends an excerpt that was cut
_LAST_CHUNK = re.compile(r"\S+\s*\Z")  # the last whitespace-delimited chunk and the whitespace after it
_MAX_BACKOFF = 12  # how far a cut may move back to keep a cluster whole

# The shapes of what must not be published, as text patterns (ASCII rules for \w). `msd.export` compiles the same
# strings for bytes to scan every file it writes, so redaction and scan can never disagree about what a path or a key is.
_TAIL = r"""[^\s"'<>`)\]]*"""
HOME_PATH = r"(?<![\w.~-])/(?:Users|home|root|content|kaggle)/[A-Za-z0-9._-]+(?:/" + _TAIL + ")?"  # incl. Colab and Kaggle
WINDOWS_PATH = r"(?<![A-Za-z])[A-Za-z]:[\\/]+(?i:Users|Documents and Settings)(?:[\\/]+" + _TAIL + ")?"  # any letter case
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
    return strip_decoding_faults(text)


def _scrub(text: str, secrets: Sequence[str]) -> str:
    text = gen.redact_with(text, secrets)
    for pattern in _TEXT_PATTERNS:
        text = pattern.sub("***", text)
    return text


def _views(text: str) -> list[str]:
    """The copies of `text` to redact next to the text itself: without what a reader does not see, and the same under
    NFKC (a fullwidth token is a token) and under NFKD (a fullwidth letter with a combining mark on it composes under NFKC
    into one non-ASCII letter that no pattern matches; decomposed, the mark is a mark on an ASCII letter and goes)."""
    hidden = hidden_view(text)
    views = [hidden]
    if not hidden.isascii():
        for form in ("NFKC", "NFKD"):
            folded = hidden_view(unicodedata.normalize(form, hidden))
            if folded not in views:
                views.append(folded)
    return views


def public_text(text: str, secrets: Sequence[str] | None = None) -> str:
    """Any string for a public row: cleaned (`clean_text`), then redacted as described in the module docstring.
    `secrets` are the values of the secret environment variables (`gen.secret_env_values()`, read here when not given);
    a caller with many strings reads them once."""
    if secrets is None:
        secrets = gen.secret_env_values()
    text = clean_text(text)
    for view in _views(text):
        scrubbed = _scrub(view, secrets)
        if scrubbed != view:
            return scrubbed  # a copy gave something away: publish that redaction (the joiners and marks are the price)
    return _scrub(text, secrets)  # nothing found in any copy: the text as it is, joiners and marks kept


def _glues(before: str, after: str) -> bool:
    """Does `after` belong to the same written character as `before`: a combining mark, a joiner, an emoji modifier,
    or whatever follows a virama or a joiner?"""
    if unicodedata.category(after).startswith("M") or after in (_ZWJ, _ZWNJ) or 0x1F3FB <= ord(after) <= 0x1F3FF:
        return True
    return before == _ZWJ or unicodedata.combining(before) == _VIRAMA_CLASS


def cut_excerpt(text: str, max_chars: int) -> str:
    """The first `max_chars` characters of `text`, moved back to the start of the cluster the cut would split (by at most
    `_MAX_BACKOFF` characters: a longer chain is cut where it stands, so it cannot empty the excerpt). Only cuts: clean
    and redact around it (`public_row`)."""
    if len(text) <= max_chars:
        return text
    cut = max_chars
    while cut > 0 and _glues(text[cut - 1], text[cut]):
        cut -= 1
        if max_chars - cut > _MAX_BACKOFF:
            return text[:max_chars]
    return text[:cut]


def _check_max_chars(max_chars: Any) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 0 <= max_chars <= PUBLIC_MAX_CHARS:
        raise ValueError(
            f"max_chars must be a whole number from 0 to {PUBLIC_MAX_CHARS} (the cap the dataset card promises), "
            f"got {max_chars!r}"
        )


def _public_excerpt(excerpt: str, max_chars: int, secrets: Sequence[str]) -> str:
    """The published excerpt: clean and redact the whole of it; when it was cut, drop the last chunk and fit it under the
    cap with the ellipsis. Redacting first lets a value of several words (a pass phrase) match while all of it is there;
    redacting again after the cut keeps the result a fixed point of the redaction."""
    text = clean_text(excerpt)
    if len(excerpt) < EXCERPT_CHARS and len(text) <= max_chars:
        return cut_excerpt(public_text(text, secrets), max_chars)  # the whole completion: nothing was cut
    if max_chars == 0:
        return ""
    room = max_chars - len(_ELLIPSIS)
    kept = _LAST_CHUNK.sub("", cut_excerpt(public_text(text, secrets), room)).rstrip()
    return cut_excerpt(public_text(kept, secrets), room) + _ELLIPSIS


def public_row(v: Verdict, max_chars: int = PUBLIC_MAX_CHARS) -> dict:
    """The public form of a verdict: exactly `PUBLIC_KEYS`, every text field cleaned and redacted, the excerpt cut to
    `max_chars` characters."""
    _check_max_chars(max_chars)
    secrets = gen.secret_env_values()
    judge = public_text(v.judge_id, secrets)
    if looks_fake(v.judge_id) or looks_fake(judge):
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
        "excerpt": _public_excerpt(v.excerpt, max_chars, secrets),
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
