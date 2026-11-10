"""Characters a reader does not see, and the view of a text without them.

A secret, a path or the word "fake" with an invisible character in the middle is still readable to anyone who copies it
and invisible to a pattern. Two things hide text this way:

* Characters Unicode marks as ignorable when displayed: the Default_Ignorable_Code_Point property, which is more than
  category Cf (the combining grapheme joiner U+034F, the variation selectors U+FE00-FE0F and U+E0100-E01EF, the Hangul
  fillers U+115F, U+1160, U+3164 and U+FFA0, the Mongolian free variation selectors, the whole tag block U+E0000-E0FFF
  including its unassigned code points). `unicodedata` has no such property, so the ranges are listed below, copied from
  DerivedCoreProperties.txt of the Unicode Character Database (Default_Ignorable_Code_Point, Unicode 15.1 and 16.0, which
  agree). Format characters (category Cf) and control characters (Cc, except tab, line feed, vertical tab, form feed and
  carriage return) are removed as well, read from `unicodedata` so a newer Python adds its own.
* A combining mark on an ASCII letter: "h" + U+0301 + "f" is displayed as an accented letter, but with the mark removed
  it is "hf" again.

`hidden_view(text)` is the text without both. It is a VIEW for looking through disguises: callers check it next to the
text as it is, and publish it only when it exposed something (see `msd.redact.public_text`), so the joiners and marks of
ordinary Devanagari, Arabic and Persian text survive. Marks are removed only after an ASCII character, never from a
non-ASCII word.

`strip_decoding_faults(text)` removes the two things a decoding fault leaves, U+FFFD and lone surrogates. They are not
invisible (a reader sees the replacement character), but they split a secret or the word "fake" just as well, so every
check that looks through disguises removes them first.

Nothing here reads a file, the environment or the network.
"""

from __future__ import annotations

import functools
import re
import sys
import unicodedata

# Default_Ignorable_Code_Point, as inclusive ranges (DerivedCoreProperties.txt).
DEFAULT_IGNORABLE: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),    # soft hyphen
    (0x034F, 0x034F),    # combining grapheme joiner
    (0x061C, 0x061C),    # Arabic letter mark
    (0x115F, 0x1160),    # Hangul choseong and jungseong fillers
    (0x17B4, 0x17B5),    # Khmer inherent vowels
    (0x180B, 0x180F),    # Mongolian free variation selectors and vowel separator
    (0x200B, 0x200F),    # zero-width space, joiners, direction marks
    (0x202A, 0x202E),    # bidirectional embeddings and overrides
    (0x2060, 0x206F),    # word joiner, invisible operators, deprecated format characters (incl. unassigned 2065)
    (0x3164, 0x3164),    # Hangul filler
    (0xFE00, 0xFE0F),    # variation selectors 1-16
    (0xFEFF, 0xFEFF),    # byte-order mark, zero-width no-break space
    (0xFFA0, 0xFFA0),    # halfwidth Hangul filler
    (0xFFF0, 0xFFF8),    # unassigned, reserved as ignorable
    (0x1BCA0, 0x1BCA3),  # shorthand format controls
    (0x1D173, 0x1D17A),  # musical symbol beam and slur controls
    (0xE0000, 0xE0FFF),  # tag characters and variation selectors 17-256, and the unassigned code points between
)
_FAULTS = re.compile("[" + chr(0xD800) + "-" + chr(0xDFFF) + chr(0xFFFD) + "]")  # written as code points, never typed
_KEPT_CONTROLS = range(0x09, 0x0E)  # tab, line feed, vertical tab, form feed, carriage return stay in text


def _class(points: list[int]) -> str:
    """A regular-expression character class for these (sorted) code points."""
    ranges, start = [], points[0]
    for before, after in zip(points, points[1:] + [None]):
        if after != before + 1:
            ranges.append((start, before))
            start = after
    return "[" + "".join(re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in ranges) + "]"


@functools.lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """(the invisible characters, an ASCII character followed by combining marks), built in one pass over Unicode."""
    invisible = {code for lo, hi in DEFAULT_IGNORABLE for code in range(lo, hi + 1)}
    marks = []
    for code in range(sys.maxunicode + 1):
        category = unicodedata.category(chr(code))
        if category == "Cf" or (category == "Cc" and code not in _KEPT_CONTROLS):
            invisible.add(code)
        elif code >= 0x300 and category in ("Mn", "Me"):
            marks.append(code)
    return re.compile(_class(sorted(invisible)) + "+"), re.compile(r"([\x00-\x7f])" + _class(marks) + "+")


def strip_decoding_faults(text: str) -> str:
    """`text` without the replacement character (U+FFFD) and lone surrogates."""
    return _FAULTS.sub("", text)


def strip_invisible(text: str) -> str:
    """`text` without the characters a reader does not see (see the module docstring), tab, line feed and carriage
    return kept."""
    return text if text.isascii() and text.isprintable() else _patterns()[0].sub("", text)


def hidden_view(text: str) -> str:
    """`text` without invisible characters and without the combining marks that sit on an ASCII character."""
    text = strip_invisible(text)
    return text if text.isascii() else _patterns()[1].sub(r"\1", text)
