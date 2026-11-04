"""Task scoring: script-aware answer normalisation and multiple-choice extraction."""

import re
import string
import unicodedata

# Multiple-choice gold labels are a single letter A-E (either case).
_MC_GOLD = re.compile(r"[A-Ea-e]")

# A letter the reply opens with: "B", "B.", "B)", "B:", "(B)", "B) Paris".
# A word that merely starts with the letter ("Because") or an article ("A man")
# is not followed by a delimiter or the end of the reply, so it does not match.
_LEADING_LETTER = re.compile(r"\s*(?:\(([A-E])\)|([A-E])(?:[.):]|\s*$))", re.IGNORECASE)

# "Answer: B", "The answer is B", "answer is **B**", "Answer: (B)". A capital
# letter may be followed by more text; a lowercase one only by punctuation or the
# end of the line, so "answer a question" / "the answer is a number" do not read as A.
_ANSWER_CUE = re.compile(
    r"\banswer\b(?:\s+is)?[\s:=*(\[]*"
    r"(?:(?-i:([A-E]))(?!\w)|([a-e])(?=[.,;:)\]*!]|[ \t]*(?:\n|$)))",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    """Canonical form for comparing free-form answers across scripts.

    NFKC, casefold, every Unicode decimal digit (Arabic-Indic, Devanagari, ...)
    mapped to its ASCII digit, punctuation of any script (including the danda
    and the Arabic question mark and comma) turned into a gap, and whitespace
    collapsed. Punctuation becomes a space rather than vanishing so that "3.5"
    cannot collide with "35".
    """
    s = unicodedata.normalize("NFKC", s).casefold()
    out = []
    for ch in s:
        if unicodedata.category(ch) == "Nd":
            out.append(str(unicodedata.decimal(ch)))
        elif ch in string.punctuation or unicodedata.category(ch).startswith("P"):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def _extract_letter(pred: str) -> str | None:
    """The multiple-choice letter a reply commits to, upper-cased, or None."""
    leading = _LEADING_LETTER.match(pred)
    if leading:
        return (leading.group(1) or leading.group(2)).upper()
    cued = [m.group(1) or m.group(2) for m in _ANSWER_CUE.finditer(pred)]
    if cued:
        return cued[-1].upper()  # the last stated answer wins
    return None


def score_task(pred: str, gold: str) -> bool:
    """True when `pred` answers `gold`.

    A gold of a single letter A-E is a multiple-choice item: the reply must
    commit to that letter. Anything else is compared as normalised text, exactly
    (not as a substring).
    """
    gold = gold.strip()
    if _MC_GOLD.fullmatch(gold):
        return _extract_letter(pred) == gold.upper()
    return normalize_answer(pred) == normalize_answer(gold)
