"""msd.redact.public_row: the only way a verdict reaches a public file (Task 19).

A public row keeps the prompt id, language, arm, alpha, label, judge id and a short excerpt, and nothing else. The
excerpt is cut on characters (never bytes), never inside a letter-plus-vowel-sign cluster, and never holds a
replacement character or a lone surrogate. Everything runs on strings built here: no model, no judge, no network.
"""

import json
import unicodedata

import pytest

from msd.redact import PUBLIC_KEYS, PUBLIC_MAX_CHARS, public_jsonl, public_row, public_task_row
from msd.schema import TaskScore, Verdict

REPLACEMENT = chr(0xFFFD)  # not typed as an escape: the editing tools turn a typed escape into the raw character
SPEC_KEYS = ["prompt_id", "lang", "arm", "alpha", "label", "judge_id", "excerpt"]

DEVANAGARI = "हिन्दी भाषा में सुरक्षित उत्तर देना क्षमता का प्रश्न है। " * 12  # vowel signs, viramas and conjuncts
ARABIC = "الْحَمْدُ لِلّٰهِ رَبِّ الْعَالَمِينَ، وَهٰذَا جَوَابٌ مُفَصَّلٌ. " * 12  # letters with harakat (combining marks)


def verdict(excerpt="a short excerpt", **changes):
    fields = dict(prompt_id="en-h001", arm="A1", alpha=None, lang="en", kind="harmful", judge_id="stub-judge-v1",
                  label="refused", excerpt=excerpt)
    fields.update(changes)
    return Verdict(**fields)


def is_mark(ch):
    return unicodedata.category(ch).startswith("M")


# --- the four fields of the brief -----------------------------------------------------------------------------------


def test_public_row_has_only_spec_fields():
    row = public_row(verdict("an excerpt", alpha=0.25, arm="A2", lang="hi", label="complied"))
    assert list(row) == SPEC_KEYS
    assert list(PUBLIC_KEYS) == SPEC_KEYS
    assert row == {"prompt_id": "en-h001", "lang": "hi", "arm": "A2", "alpha": 0.25, "label": "complied",
                   "judge_id": "stub-judge-v1", "excerpt": "an excerpt"}
    assert "kind" not in row  # a verdict also carries the kind; the public row does not
    json.dumps(row)  # plain JSON types only


def test_public_row_alpha_stays_none_for_an_arm_without_alpha():
    assert public_row(verdict(alpha=None))["alpha"] is None


def test_excerpt_truncated_to_max_chars():
    text = "abcdefghij" * 60  # 600 characters
    assert public_row(verdict(text))["excerpt"] == text[:200]  # the default is 200
    assert public_row(verdict(text), max_chars=10)["excerpt"] == "abcdefghij"
    assert public_row(verdict(text), max_chars=0)["excerpt"] == ""
    assert public_row(verdict("short"), max_chars=10)["excerpt"] == "short"  # shorter than the cap: unchanged
    assert public_row(verdict("x" * 200))["excerpt"] == "x" * 200
    assert PUBLIC_MAX_CHARS == 200


@pytest.mark.parametrize("bad", [201, 5000, -1, True, 2.5, None, "200"])
def test_max_chars_beyond_the_published_cap_or_not_a_count_is_refused(bad):
    with pytest.raises(ValueError, match="max_chars"):
        public_row(verdict("text"), max_chars=bad)


# --- the cut is on characters, and never inside a cluster ------------------------------------------------------------


@pytest.mark.parametrize("text", [DEVANAGARI, ARABIC], ids=["devanagari", "arabic"])
def test_cut_is_on_characters_not_bytes_and_never_splits_a_cluster(text):
    assert len(text.encode("utf-8")) > 1.5 * len(text)  # a letter is two or three bytes: a byte cut would be mojibake
    for cap in range(1, 120):
        excerpt = public_row(verdict(text), max_chars=cap)["excerpt"]
        assert len(excerpt) <= cap
        assert text.startswith(excerpt)
        assert len(excerpt) >= cap - 5, (cap, len(excerpt))  # backs off over a cluster, not over the whole text
        assert excerpt.encode("utf-8").decode("utf-8") == excerpt  # no mojibake, whole code points only
        assert REPLACEMENT not in excerpt
        rest = text[len(excerpt):]
        assert rest
        assert not is_mark(rest[0]), f"cap {cap}: the dropped part starts with a combining mark"
        assert not excerpt.endswith(("्", "‍")), f"cap {cap}: the excerpt ends on a virama or a joiner"


def test_default_cap_on_a_devanagari_excerpt_is_at_most_200_characters_and_whole():
    excerpt = public_row(verdict(DEVANAGARI))["excerpt"]
    assert 195 <= len(excerpt) <= 200
    assert DEVANAGARI.startswith(excerpt)
    assert not is_mark(DEVANAGARI[len(excerpt)])


def test_a_text_that_is_all_marks_cuts_to_nothing_rather_than_a_lone_mark():
    marks = "́" * 50  # combining acute accents, no base letter
    assert public_row(verdict(marks), max_chars=10)["excerpt"] == ""


def test_the_cut_keeps_an_emoji_joiner_sequence_whole():
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    excerpt = public_row(verdict("ab" + family), max_chars=4)["excerpt"]
    assert excerpt == "ab"  # the man (position 2) and the joiner (3) are both gone: no dangling joiner


# --- characters that must never reach a public file ------------------------------------------------------------------


def test_replacement_character_and_lone_surrogates_never_survive():
    dirty = "ab" + REPLACEMENT + "cd" + chr(0xD800) + "ef" + chr(0xDC00) + "gh"
    assert public_row(verdict(dirty))["excerpt"] == "abcdefgh"
    json.dumps(public_row(verdict(dirty)), ensure_ascii=False).encode("utf-8")  # encodes: no surrogate is left


def test_a_replacement_character_at_the_cut_does_not_count_against_the_cap():
    assert public_row(verdict("abc" + REPLACEMENT * 5 + "defg"), max_chars=5)["excerpt"] == "abcde"


def test_token_shaped_text_in_the_excerpt_and_the_judge_id_is_redacted():
    token = "hf_" + "A1b2C3d4E5" * 3
    row = public_row(verdict(f"leaked {token} here", judge_id=f"hf:{token}"))
    assert token not in json.dumps(row)
    assert "***" in row["excerpt"] and "***" in row["judge_id"]


def test_the_cap_applies_after_the_redaction_so_the_row_never_grows():
    token = "hf_" + "Z9y8X7w6V5" * 3
    row = public_row(verdict("t" * 190 + token + "tail"), max_chars=200)
    assert len(row["excerpt"]) <= 200
    assert "hf_" not in row["excerpt"]


# --- the judge: a test judge is refused ------------------------------------------------------------------------------


@pytest.mark.parametrize("judge", ["fake-judge", "FAKE", "my-Fake-v2", "ｆａｋｅ-judge", "fa​ke-judge"])
def test_a_verdict_from_a_test_judge_is_refused(judge):
    with pytest.raises(ValueError, match="test judge"):
        public_row(verdict(judge_id=judge))


def test_a_real_judge_id_passes_through_unchanged():
    assert public_row(verdict(judge_id="anthropic:claude-x:t1a2b3c4d5"))["judge_id"] == "anthropic:claude-x:t1a2b3c4d5"


# --- task rows and the file text -------------------------------------------------------------------------------------


def test_public_task_row_keeps_the_item_id_and_correctness_only():
    row = public_task_row(TaskScore("en-t007", "A2", 0.1, "sw", True))
    assert row == {"prompt_id": "en-t007", "lang": "sw", "arm": "A2", "alpha": 0.1, "correct": True}


def test_public_jsonl_is_utf8_one_row_per_physical_line_whatever_the_text():
    rows = [public_row(verdict("a b c\x85d\ne\r\nf")), public_row(verdict("हिन्दी"))]
    data = public_jsonl(rows)
    assert data.endswith(b"\n")
    text = data.decode("utf-8")
    assert len(text.splitlines()) == 2  # str.splitlines would otherwise split at U+2028, U+2029 and U+0085
    assert [json.loads(line) for line in text.splitlines()] == rows  # the escapes decode to the same strings
    assert "हिन्दी" in text  # raw UTF-8, no \u escapes for ordinary letters
    assert public_jsonl([]) == b""
