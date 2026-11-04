"""Task scoring: script-aware answer normalisation and multiple-choice extraction."""

import re
import string
import unicodedata

# Multiple-choice gold labels are a single letter A-E (either case).
_MC_GOLD = re.compile(r"[A-Ea-e]")

# Punctuation a bare letter may be followed by: ASCII, danda, double danda,
# Arabic question mark / comma / semicolon, Urdu full stop, and markdown stars.
_FOLLOW = r"[.):;,\]*`!?।॥؟،؛۔]"
# The rest of the line is empty (an explanation may follow on the next line).
_LINE_END = r"[ \t]*(?:\n|$)"

# A letter right after a cue. A capital may be followed by more text; a lowercase
# one only by punctuation or the end of the line, so "answer a question" /
# "the answer is a number" do not read as option A.
_LETTER_AFTER_CUE = (
    r"(?:(?-i:([A-E]))(?!\w)|([a-e])(?=" + _FOLLOW + "|" + _LINE_END + "))"
)

# What may sit between a cue word and its letter: "ni" / "is" / "है" / "هي",
# a qualifier ("sahihi", "الصحيحة"), and separators or markdown (": ", " - ", "**").
_CUE_TAIL = (
    r"(?:\s+(?:sahihi|الصحيحة|الصحيح))?"
    r"(?:\s+(?:is|ni|है|هي|هو)(?!\w))?"
    r"[\s:=\-–—*`(\[]*"
)

# Strong cues: the reply states its answer. Includes "the correct option is B"
# and its hi / ar / sw equivalents, so echoed option lists do not hide the answer.
_ANSWER_WORDS = (
    r"answer|उत्तर|ال[إا]جابة|الجواب|jibu"
    r"|(?:correct|right|final)\s+(?:option|choice)"
    r"|सही\s+विकल्प|الخيار\s+الصحيح|chaguo\s+sahihi"
)
# Weak cues: a bare "Option B". A reply often states its letter first and then
# explains why another option is wrong ("B. Paris, not option A"), so a leading
# letter beats these. With no leading letter they count only when every mention
# names the same letter ("Option A is wrong. Option B is correct." commits to none).
_OPTION_WORDS = r"option|choice|विकल्प|الخيار|chaguo"


def _cue(words: str) -> re.Pattern[str]:
    return re.compile(
        r"(?<!\w)(?:" + words + r")(?!\w)" + _CUE_TAIL + _LETTER_AFTER_CUE, re.IGNORECASE
    )


_ANSWER_CUE = _cue(_ANSWER_WORDS)
_OPTION_CUE = _cue(_OPTION_WORDS)

# A letter the reply opens with, after any markdown: "B", "B.", "B)", "B।", "(B)",
# "**B**", "B - Paris", "B is correct", "B" then a newline and an explanation.
# A word that merely starts with the letter ("Because") or an article ("A lot of
# people say C") is followed by none of these, so it does not match.
_LEADING = re.compile(
    r"[\s*_`]*(?:[(\[]([A-E])[)\]]"
    r"|([A-E])(?=" + _FOLLOW + "|" + _LINE_END + r"|\s+[-–—](?:\s|$)"
    r"|\s+(?:is|are)\s+(?:the\s+)?(?:correct|right)(?!\w)))",
    re.IGNORECASE,
)

# The letter is followed by another capital letter joined by and/or/comma:
# "B and C", "C or B", "A, B". A hedged reply commits to no single answer.
_HEDGE = re.compile(
    r"\s*(?:[,;/&،؛]|(?<!\w)(?:and|or|na|au|ama|और|या|أو|و)(?!\w))"
    r"\s*[(\[]?(?-i:[A-E])(?!\w)",
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


def _commit(m: re.Match[str], text: str) -> str | None:
    """The matched letter, upper-cased, or None when the reply hedges ("B and C")."""
    if _HEDGE.match(text, m.end()):
        return None
    return next(g for g in m.groups() if g).upper()


def _extract_letter(pred: str) -> str | None:
    """The multiple-choice letter a reply commits to, upper-cased, or None.

    Precedence: a strong cue ("Answer: B", "The correct option is B", "उत्तर: B",
    "الإجابة: B", "Jibu: B"), then a letter the reply opens with ("B. Paris, not
    option A"), then weak option cues ("Option B"). A strong cue beats a leading
    letter, so an echoed option list ("A. Paris  B. London  The answer is B")
    scores as B.
    """
    text = unicodedata.normalize("NFKC", pred)
    answered = list(_ANSWER_CUE.finditer(text))
    if answered:
        return _commit(answered[-1], text)  # the last stated answer wins
    leading = _LEADING.match(text)
    if leading:
        return _commit(leading, text)
    optioned = {_commit(m, text) for m in _OPTION_CUE.finditer(text)}
    return optioned.pop() if len(optioned) == 1 else None


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
