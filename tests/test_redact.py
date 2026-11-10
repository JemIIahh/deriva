"""msd.redact.public_row: the only way a verdict reaches a public file (Task 19).

A public row keeps the prompt id, language, arm, alpha, label, judge id and a short excerpt, and nothing else. The
excerpt is cut on characters (never bytes), never inside a letter-plus-vowel-sign cluster, and never holds a
replacement character or a lone surrogate. Everything runs on strings built here: no model, no judge, no network.
"""

import json
import unicodedata

import pytest

from msd.redact import (
    PUBLIC_KEYS,
    PUBLIC_MAX_CHARS,
    clean_text,
    cut_excerpt,
    public_jsonl,
    public_row,
    public_task_row,
    public_text,
    strip_invisible,
)
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


# --- security fix: clean first, then redact, then cut; and nothing hides a secret from the redaction -----------------

USERS = "/" + "Users"  # never written whole: tests/ is scanned for home paths (tests/test_hygiene.py)
HOME = "/" + "home"
TOKEN = "hf_" + "A1b2C3d4E5F6g7H8i9J0k1L2m3N4o5P6q7"  # three characters of prefix and 34 of body
HOME_PATH = USERS + "/ram/secret/file.txt"
LINUX_PATH = HOME + "/runner/work/secret/file.txt"
# What an attacker (or a decoding fault) can put inside a secret: characters the reader does not see, or that the
# study strips. Every one is written as a code point.
SPLITTERS = {
    "replacement": chr(0xFFFD),
    "lone-surrogate-high": chr(0xD800),
    "lone-surrogate-low": chr(0xDC00),
    "zero-width-space": chr(0x200B),
    "word-joiner": chr(0x2060),
    "soft-hyphen": chr(0x00AD),
    "byte-order-mark": chr(0xFEFF),
    "left-to-right-mark": chr(0x200E),
    "bidi-isolate": chr(0x2066),
    "zero-width-joiner": chr(0x200D),
    "zero-width-non-joiner": chr(0x200C),
    "arabic-number-sign": chr(0x0600),
    "nul": chr(0),
    "backspace": chr(8),
}
SPLIT_IDS = list(SPLITTERS)


def split_at(text, where, char):
    return text[:where] + char + text[where:]


def visible(text):
    """`text` as a reader sees it: without any character `strip_invisible` or `clean_text` would remove."""
    return strip_invisible(clean_text(text))


@pytest.mark.parametrize("name", SPLIT_IDS)
@pytest.mark.parametrize("where", [1, 3, 12, 20, len(TOKEN) - 1])
def test_a_token_split_by_an_invisible_character_is_published_redacted(name, where):
    text = f"see {split_at(TOKEN, where, SPLITTERS[name])} here"
    for field in ("excerpt", "judge_id"):
        row = public_row(verdict(text, judge_id=text) if field == "judge_id" else verdict(text))
        shown = visible(row[field])
        assert "***" in shown, (name, where, field)
        assert not any(TOKEN[i:i + 6] in shown for i in range(3, len(TOKEN) - 5)), (name, where, field, shown)
        assert "hf_" not in shown
        assert chr(0xFFFD) not in row[field] and not any(0xD800 <= ord(c) <= 0xDFFF for c in row[field])


@pytest.mark.parametrize("name", SPLIT_IDS)
@pytest.mark.parametrize("path", [HOME_PATH, LINUX_PATH], ids=["macos", "linux"])
@pytest.mark.parametrize("where", [1, 3, 6, 9, 14, 24])
def test_a_home_path_split_by_an_invisible_character_is_published_redacted(name, path, where):
    row = public_row(verdict(f"wrote it to {split_at(path, where, SPLITTERS[name])} ok"))
    shown = visible(row["excerpt"])
    assert "***" in shown, (name, where)
    for part in ("Users", "home", "ram", "runner", "secret", "file.txt"):
        assert part not in shown, (name, where, part, shown)


@pytest.mark.parametrize("text, gone", [
    (f"see {HOME_PATH} now", ("ram", "secret")),
    (f"see {LINUX_PATH}, now", ("runner", "secret")),
    ("C:" + chr(92) + "Users" + chr(92) + "ram" + chr(92) + "secret.txt later", ("ram", "secret")),
    ("key sk-ant-api03-" + "Zx9Yw8Vu7T" * 3 + " end", ("Zx9Yw8",)),
    ("key ghp_" + "A1B2C3D4E5" * 3 + " end", ("A1B2C3",)),
    ("id AKIA" + "ABCDEFGHIJKLMNOP" + " end", ("ABCDEFG",)),
    ("-----BEGIN " + "PRIVATE KEY-----\nMIIBVQIBADANBgkq\n-----END PRIVATE KEY----- after", ("MIIBVQ",)),
], ids=["macos", "linux", "windows", "anthropic", "github", "aws", "private-key"])
def test_paths_and_other_secret_shapes_are_redacted_in_the_public_row_not_only_refused_later(text, gone):
    shown = public_row(verdict(text))["excerpt"]
    assert "***" in shown
    for part in gone:
        assert part not in shown


def test_ordinary_text_that_only_looks_a_little_like_a_path_or_a_key_is_untouched():
    text = "tasks and desks; site/home/page and path/to/data; sk-ip it; hf_ alone; a/Users/b is not a home path"
    assert public_row(verdict(text))["excerpt"] == text


def test_joiners_in_ordinary_devanagari_arabic_and_persian_text_come_through_unchanged():
    zwj, zwnj = chr(0x200D), chr(0x200C)
    hindi = "क्" + zwj + "ष और ज्ञ का उच्चारण, श्री" + zwnj + "राम। " * 3  # a joiner and a non-joiner inside real words
    persian = "می" + zwnj + "خواهم که بروم و " + "کتاب" + zwnj + "ها را بخوانم."
    for text in (hindi, persian, "ab" + zwj + "cd"):
        assert public_row(verdict(text))["excerpt"] == text
        assert text.encode("utf-8").decode("utf-8") == text


def test_a_token_among_joiners_loses_the_joiners_of_that_excerpt_and_nothing_else_leaks():
    zwj = chr(0x200D)
    text = "क्" + zwj + "ष " + TOKEN + " ही"
    excerpt = public_row(verdict(text))["excerpt"]
    assert "***" in excerpt and TOKEN[3:15] not in excerpt
    assert zwj not in excerpt  # the price of redacting the copy without invisible characters


def test_the_code_point_safe_cut_still_works_on_text_with_joiners():
    zwj = chr(0x200D)
    text = ("क्" + zwj + "ष कि न्दी ") * 40
    for cap in range(1, 80):
        excerpt = public_row(verdict(text), max_chars=cap)["excerpt"]
        assert len(excerpt) <= cap and text.startswith(excerpt)
        rest = text[len(excerpt):]
        assert rest and not is_mark(rest[0]) and rest[0] != zwj
        assert not excerpt.endswith(("्", zwj))


def test_clean_text_removes_only_the_replacement_character_and_lone_surrogates():
    zwj = chr(0x200D)
    assert clean_text("a" + chr(0xFFFD) + "b" + chr(0xD800) + "c" + chr(0xDFFF) + zwj + "d") == "abc" + zwj + "d"
    assert clean_text("plain हिन्दी text") == "plain हिन्दी text"


def test_cut_excerpt_only_cuts():
    dirty = "ab" + chr(0xFFFD) + "cd"
    assert cut_excerpt(dirty, 10) == dirty  # cleaning is a separate, earlier step
    assert cut_excerpt("abcdef", 3) == "abc"


def test_strip_invisible_removes_format_characters_and_controls_but_keeps_text_and_whitespace():
    assert strip_invisible("a" + chr(0x200B) + "b" + chr(0x200D) + chr(0x2060) + "c" + chr(0) + chr(8) + "d") == "abcd"
    assert strip_invisible("two words\tand\nlines\r\n") == "two words\tand\nlines\r\n"
    assert strip_invisible("हिन्दी العربية") == "हिन्दी العربية"


def test_the_judge_id_is_cleaned_like_every_other_text_field():
    row = public_row(verdict(judge_id="stub" + chr(0xFFFD) + "-judge" + chr(0xD800) + "-v1"))
    assert row["judge_id"] == "stub-judge-v1"
    row = public_row(verdict(judge_id="anthropic:" + chr(0xFFFD) * 3))
    assert row["judge_id"] == "anthropic:"


def test_a_test_judge_hidden_by_removable_characters_is_still_refused():
    for judge in ("fa" + chr(0xFFFD) + "ke-judge", "fa" + chr(0xD800) + "ke", "f" + chr(0xFFFD) + "a" + chr(0xFFFD) + "ke"):
        with pytest.raises(ValueError, match="test judge"):
            public_row(verdict(judge_id=judge))


def test_the_other_text_fields_of_a_row_are_cleaned_and_redacted_too():
    row = public_row(verdict(prompt_id="en-h" + chr(0xFFFD) + "001", lang="e" + chr(0xD800) + "n", arm="A" + chr(0x200B) + "1"))
    assert (row["prompt_id"], row["lang"]) == ("en-h001", "en")
    assert row["arm"] == "A" + chr(0x200B) + "1"  # an invisible character is not removed from plain text, only looked through
    leaked = public_row(verdict(prompt_id="id-" + split_at(TOKEN, 5, chr(0x200B))))
    assert "***" in leaked["prompt_id"] and TOKEN[3:15] not in visible(leaked["prompt_id"])
    task = public_task_row(TaskScore("it" + chr(0xFFFD) + "em-1", "A1", None, "en", True))
    assert task["prompt_id"] == "item-1"
    task = public_task_row(TaskScore("item-" + split_at(TOKEN, 8, chr(0x2060)), "A1", None, "en", True))
    assert "***" in task["prompt_id"] and TOKEN[3:15] not in visible(task["prompt_id"])


def test_public_text_cleans_then_redacts_the_text_without_invisible_characters_else_the_text_as_it_is():
    assert public_text("a" + chr(0xFFFD) + "b") == "ab"
    assert public_text("x " + split_at(TOKEN, 9, chr(0x200B)) + " y") == "x *** y"
    assert public_text("ab" + chr(0x200D) + "cd") == "ab" + chr(0x200D) + "cd"


def test_the_value_of_a_secret_environment_variable_is_redacted_even_when_split(monkeypatch):
    secret = "plain-secret-value-0123456789"
    monkeypatch.setenv("MSD_TEST_API_KEY", secret)
    for splitter in ("", chr(0x200B), chr(0x2060), chr(0xFFFD), chr(0xD800)):
        excerpt = public_row(verdict("it was " + split_at(secret, 10, splitter) + " here"))["excerpt"]
        assert "***" in excerpt and "secret-value" not in visible(excerpt), repr(splitter)
    assert public_row(verdict("nothing here"))["excerpt"] == "nothing here"


def test_a_row_reads_the_environment_once_not_once_per_string(monkeypatch):
    # public_text runs several times per row; reading the whole environment each time made the export tests three times slower
    from msd import gen
    calls = []
    real = gen.secret_env_values
    monkeypatch.setattr(gen, "secret_env_values", lambda: calls.append(1) or real())
    public_row(verdict("an excerpt " + split_at(TOKEN, 6, chr(0x200B))))
    assert len(calls) == 1
    public_task_row(TaskScore("item-1", "A1", None, "en", True))
    assert len(calls) == 2
