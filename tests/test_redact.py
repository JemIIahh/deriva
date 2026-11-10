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
    hidden_view,
    public_jsonl,
    public_row,
    public_task_row,
    public_text,
    strip_invisible,
)
from msd.gaterun import looks_fake
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
    text = "word " * 120  # 600 characters, a word at every fifth one
    shown = public_row(verdict(text))["excerpt"]
    assert len(shown) <= 200 and shown.endswith(chr(0x2026)) and text.startswith(shown[:-1])  # the default is 200
    assert len(shown) >= 190  # one word and the marker are given up, not more
    assert public_row(verdict(text), max_chars=12)["excerpt"] == "word word" + chr(0x2026)
    assert public_row(verdict(text), max_chars=0)["excerpt"] == ""
    assert public_row(verdict("short"), max_chars=10)["excerpt"] == "short"  # shorter than the cap and the limit: unchanged
    assert public_row(verdict("x" * 199))["excerpt"] == "x" * 199  # shorter than the upstream limit: the whole completion
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
        excerpt = cut_excerpt(text, cap)
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
    excerpt = cut_excerpt(DEVANAGARI, 200)
    assert 195 <= len(excerpt) <= 200
    assert DEVANAGARI.startswith(excerpt)
    assert not is_mark(DEVANAGARI[len(excerpt)])
    shown = public_row(verdict(DEVANAGARI))["excerpt"]  # as a row: the last word and a marker are given up
    assert 180 <= len(shown) <= 200 and shown.endswith(chr(0x2026)) and DEVANAGARI.startswith(shown[:-1])
    assert DEVANAGARI[len(shown) - 1].isspace()


def test_a_text_that_is_all_marks_cuts_to_nothing_rather_than_a_lone_mark():
    marks = "́" * 50  # combining acute accents, no base letter
    assert cut_excerpt(marks, 10) == ""
    assert public_row(verdict(marks), max_chars=10)["excerpt"] == chr(0x2026)


def test_the_cut_keeps_an_emoji_joiner_sequence_whole():
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    excerpt = cut_excerpt("ab" + family, 4)
    assert excerpt == "ab"  # the man (position 2) and the joiner (3) are both gone: no dangling joiner


# --- characters that must never reach a public file ------------------------------------------------------------------


def test_replacement_character_and_lone_surrogates_never_survive():
    dirty = "ab" + REPLACEMENT + "cd" + chr(0xD800) + "ef" + chr(0xDC00) + "gh"
    assert public_row(verdict(dirty))["excerpt"] == "abcdefgh"
    json.dumps(public_row(verdict(dirty)), ensure_ascii=False).encode("utf-8")  # encodes: no surrogate is left


def test_a_replacement_character_before_the_cut_does_not_count_against_the_cap():
    shown = public_row(verdict("ab" + REPLACEMENT * 5 + " cd ef"), max_chars=6)["excerpt"]
    assert shown == "ab" + chr(0x2026)  # the cleaned text is "ab cd ef": room for five characters, the last word dropped


def test_token_shaped_text_in_the_excerpt_and_the_judge_id_is_redacted():
    token = "hf_" + "A1b2C3d4E5" * 3
    row = public_row(verdict(f"leaked {token} here", judge_id=f"hf:{token}"))
    assert token not in json.dumps(row)
    assert "***" in row["excerpt"] and "***" in row["judge_id"]


def test_the_cap_applies_after_the_redaction_so_the_row_never_grows():
    token = "hf_" + "Z9y8X7w6V5" * 3
    row = public_row(verdict("t " * 95 + token + " tail"), max_chars=200)
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
    """`text` as a reader sees it: without any character `hidden_view` or `clean_text` would remove."""
    return hidden_view(clean_text(text))


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
        excerpt = cut_excerpt(text, cap)
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


# --- fix round 1: more characters that hide a secret, and the upstream cut -----------------------------------------------
# Unicode's Default_Ignorable_Code_Point set is more than category Cf: the combining grapheme joiner, variation
# selectors, Hangul fillers and the unassigned tag block are invisible too. Every character is a code point.

MORE_SPLITTERS = {
    "combining-grapheme-joiner": chr(0x034F),
    "variation-selector-16": chr(0xFE0F),
    "variation-selector-17": chr(0xE0100),
    "mongolian-selector": chr(0x180B),
    "hangul-filler": chr(0x3164),
    "halfwidth-hangul-filler": chr(0xFFA0),
    "tag-block-unassigned": chr(0xE0000),
    "combining-mark-after-ascii": chr(0x0301),
}


@pytest.mark.parametrize("name", list(MORE_SPLITTERS))
@pytest.mark.parametrize("where", [1, 3, 12, 20, len(TOKEN) - 1])
def test_a_token_split_by_a_default_ignorable_or_a_mark_is_published_redacted(name, where):
    text = f"see {split_at(TOKEN, where, MORE_SPLITTERS[name])} here"
    row = public_row(verdict(text, judge_id=text))
    for field in ("excerpt", "judge_id"):
        shown = visible(row[field])
        assert "***" in shown and "hf_" not in shown, (name, where, field, shown)
        assert not any(TOKEN[i:i + 6] in shown for i in range(3, len(TOKEN) - 5)), (name, where, field, shown)


@pytest.mark.parametrize("name", list(MORE_SPLITTERS))
@pytest.mark.parametrize("path", [HOME_PATH, LINUX_PATH], ids=["macos", "linux"])
@pytest.mark.parametrize("where", [1, 3, 6, 9, 14, 24])
def test_a_home_path_split_by_a_default_ignorable_or_a_mark_is_published_redacted(name, path, where):
    shown = visible(public_row(verdict(f"wrote it to {split_at(path, where, MORE_SPLITTERS[name])} ok"))["excerpt"])
    assert "***" in shown, (name, where)
    for part in ("Users", "home", "ram", "runner", "secret", "file.txt"):
        assert part not in shown, (name, where, part, shown)


@pytest.mark.parametrize("secret", [TOKEN, HOME_PATH], ids=["token", "path"])
def test_a_combining_mark_over_every_letter_of_a_secret_does_not_hide_it(secret):
    marked = "".join(ch + chr(0x0301) for ch in secret)
    shown = visible(public_row(verdict(f"see {marked} here"))["excerpt"])
    assert "***" in shown and not any(secret[i:i + 5] in shown for i in range(3, len(secret) - 5))


def test_ordinary_text_with_marks_and_default_ignorables_in_it_keeps_them_when_nothing_is_secret():
    hindi = "कि और क्षेत्र, हिन्दी" + chr(0x200D) + " भाषा"
    arabic = "الْحَمْدُ لِلّٰهِ" + chr(0x200C) + " رَبِّ"
    cafe = "cafe" + chr(0x0301) + " au lait, " + "a" + chr(0xFE0F) + "b"  # a mark over an ASCII letter, a variation selector
    for text in (hindi, arabic, cafe):
        assert public_row(verdict(text))["excerpt"] == text


@pytest.mark.parametrize("name", list(MORE_SPLITTERS))
def test_a_test_judge_hidden_by_a_default_ignorable_is_still_refused(name):
    if name == "combining-mark-after-ascii":
        pytest.skip("a mark changes the letter it sits on: that is not the word 'fake'")
    for judge in ("fa" + MORE_SPLITTERS[name] + "ke-judge", "FAKE" + MORE_SPLITTERS[name], "f" + MORE_SPLITTERS[name] + "a" + MORE_SPLITTERS[name] + "ke"):
        with pytest.raises(ValueError, match="test judge"):
            public_row(verdict(judge_id=judge))
        assert looks_fake(judge)


def test_a_fullwidth_token_and_path_are_redacted_too():
    wide = lambda s: "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in s)  # noqa: E731
    for secret in (TOKEN, HOME_PATH):
        shown = public_row(verdict(f"see {wide(secret)} here"))["excerpt"]
        assert "***" in shown, secret
        assert not any(part in unicodedata.normalize("NFKC", shown) for part in ("A1b2C3", "ram", "secret"))


def test_a_windows_path_in_any_case_and_the_colab_and_kaggle_roots_are_redacted():
    for path in ("c:" + chr(92) + "users" + chr(92) + "ram" + chr(92) + "x.txt", "D:/USERS/ram/x.txt",
                 "C:" + chr(92) + "DOCUMENTS AND SETTINGS" + chr(92) + "ram", "/root/.ssh/id_rsa", "/content/drive/MyDrive/x",
                 "/kaggle/working/secret.csv", "/kaggle/input/data"):
        shown = public_row(verdict(f"see {path} now"))["excerpt"]
        assert shown == "see *** now", (path, shown)


# --- the upstream cut: a real excerpt arrives already cut at 200 characters (msd.judge.EXCERPT_CHARS) ---------------

MARK = chr(0x2026)
VIRAMA = chr(0x094D)
WORDS = "word " * 100  # whitespace-delimited filler: index 185, 190, ... fall on the start of a word


def upstream(completion):
    """What the judge stage stores: the first 200 characters of the completion, nothing else."""
    return completion[:200]


@pytest.mark.parametrize("start", [181, 185, 186, 190, 195, 198])
def test_a_token_cut_by_the_upstream_limit_publishes_no_prefix_of_it(start):
    shown = public_row(verdict(upstream(WORDS[:start] + TOKEN + " and then more words")))["excerpt"]
    assert "hf_" not in shown and TOKEN[3:7] not in shown, (start, shown)
    assert shown.endswith(MARK) and len(shown) <= 200


@pytest.mark.parametrize("start", [181, 185, 190, 195])
def test_an_environment_secret_cut_by_the_upstream_limit_publishes_no_prefix_of_it(monkeypatch, start):
    value = "s3cr3t-" + "k" * 25  # 32 characters, nothing like a token shape
    monkeypatch.setenv("MSD_FIX_API_KEY", value)
    shown = public_row(verdict(upstream(WORDS[:start] + value + " and then more words")))["excerpt"]
    assert "s3cr3t" not in shown and "kkkk" not in shown, (start, shown)
    assert shown.endswith(MARK) and len(shown) <= 200


@pytest.mark.parametrize("start", [180, 185, 190, 195, 198])
def test_a_path_cut_by_the_upstream_limit_publishes_no_part_of_it(start):
    path = USERS + "/ram/Documents/project/results.csv"
    shown = public_row(verdict(upstream(WORDS[:start] + path + " and then more words")))["excerpt"]
    assert "ram" not in shown and "Docum" not in shown and USERS not in shown, (start, shown)
    assert shown.endswith(MARK) and len(shown) <= 200


def test_an_upstream_cut_inside_a_conjunct_does_not_end_on_a_virama():
    completion = "a " * 98 + "कक" + "क" + VIRAMA + "षत्रिय और बहुत कुछ"  # index 199 is the virama
    assert upstream(completion)[-1] == VIRAMA
    shown = public_row(verdict(upstream(completion)))["excerpt"]
    body = shown[:-1]
    assert shown.endswith(MARK) and not body.endswith((VIRAMA, chr(0x200D))) and not is_mark(body[-1])
    assert len(shown) <= 200 and completion.startswith(body.rstrip())


def test_an_excerpt_shorter_than_the_upstream_limit_is_complete_and_has_no_marker():
    for text in ("a short reply", "x " * 99 + "y", "हिन्दी भाषा"):  # 199 characters at most
        assert len(text) < 200
        assert public_row(verdict(text))["excerpt"] == text


def test_an_excerpt_of_exactly_the_upstream_limit_counts_as_cut():
    text = ("word " * 40)[:200]
    assert len(text) == 200
    shown = public_row(verdict(text))["excerpt"]
    assert shown.endswith(MARK) and len(shown) <= 200 and text.startswith(shown[:-1])


def test_the_marker_and_the_dropped_word_always_fit_under_the_cap():
    text = "lorem ipsum dolor sit amet " * 30
    for cap in range(1, 201):
        shown = public_row(verdict(text), max_chars=cap)["excerpt"]
        assert len(shown) <= cap, cap
        assert shown.endswith(MARK) or cap == 0
        assert text.startswith(shown[:-1].rstrip())


def test_an_excerpt_without_any_whitespace_is_published_as_the_marker_alone():
    # nothing tells where a word ended, so nothing of it is kept (all five study languages are written with spaces)
    assert public_row(verdict("abcdefghij" * 60))["excerpt"] == MARK


def test_the_cut_never_removes_more_than_a_cluster_from_a_long_chain():
    chain = "a" + chr(0x0301) * 60
    assert len(cut_excerpt(chain, 40)) >= 28  # the back-off is capped: a long chain cannot empty the excerpt
    assert cut_excerpt("ab" + "क" + VIRAMA + "ष", 3) == "ab"  # a short cluster still moves back whole


# --- fix round 2: a fullwidth secret with a mark on every letter; an environment value with spaces; a decoding fault in a judge id ----


def fullwidth_marked(text):
    """Every character made fullwidth where Unicode has a fullwidth form, each followed by a combining acute accent. NFKC
    composes some of these into one non-ASCII letter (a fullwidth o and the accent become one o-acute), which hides the
    letter from a pattern; NFKD does not."""
    wide = lambda ch: chr(ord(ch) + 0xFEE0) if 0x21 <= ord(ch) <= 0x7E else ch  # noqa: E731
    return "".join(wide(ch) + chr(0x0301) for ch in text)


def folded(text):
    """`text` as a reader who removes marks and folds compatibility forms (NFKD) sees it."""
    return hidden_view(unicodedata.normalize("NFKD", text))


SECRET_KINDS = {
    "hugging-face-token": TOKEN,
    "api-key": "sk-" + "A1b2C3d4E5F6g7H8i9J0k1",
    "github-token": "ghp_" + "A1b2C3d4E5F6g7H8i9J0k1L2m3",
    "aws-key-id": "AKIA" + "ABCDEFGHIJKLMNOP",
    "macos-path": HOME_PATH,
    "linux-path": LINUX_PATH,
    "windows-path": "C:" + chr(92) + "Users" + chr(92) + "ram" + chr(92) + "secret" + chr(92) + "file.txt",
}


def fragments(secret):
    return [secret[i:i + 5] for i in range(3, len(secret) - 5)]


@pytest.mark.parametrize("kind", list(SECRET_KINDS))
def test_a_fullwidth_secret_with_a_mark_on_every_letter_is_redacted(kind):
    secret = SECRET_KINDS[kind]
    for field in ("excerpt", "judge_id"):
        text = f"see {fullwidth_marked(secret)} here"
        row = public_row(verdict(text, judge_id=text))
        shown = folded(row[field])
        assert "***" in shown, (kind, field, shown)
        assert not any(part in shown for part in fragments(secret)), (kind, field, shown)


def test_a_fullwidth_environment_value_with_a_mark_on_every_letter_is_redacted(monkeypatch):
    value = "plain-secret-value-0123456789"
    monkeypatch.setenv("MSD_FIX_API_KEY", value)
    shown = folded(public_row(verdict(f"it was {fullwidth_marked(value)} here"))["excerpt"])
    assert "***" in shown and "secret-value" not in shown


def test_ordinary_fullwidth_text_with_marks_is_not_changed_when_nothing_is_secret():
    text = "Ｈｅｌｌｏ" + chr(0x0301) + " wörld, कि और क्षेत्र"
    assert public_row(verdict(text))["excerpt"] == text


MULTI_WORD = "correct horse battery staple 99"  # an environment value with spaces: a pass phrase, not a key


@pytest.mark.parametrize("start", [60, 120, 165, 167, 168, 169])
def test_a_multi_word_environment_value_inside_the_excerpt_is_redacted_whole(monkeypatch, start):
    monkeypatch.setenv("MSD_FIX_API_KEY", MULTI_WORD)
    text = upstream(WORDS[:start] + MULTI_WORD + "." + " and then more words" * 10)
    assert MULTI_WORD in text and len(text) == 200  # the whole value is inside what the judge stored, and the text is cut
    shown = public_row(verdict(text))["excerpt"]
    assert not any(word in shown for word in ("correct", "horse", "battery", "staple")), (start, shown)
    assert shown.endswith(MARK) and len(shown) <= 200
    assert ("***" in shown) == (start < 167)  # from 167 on the value ends in the last chunk, dropped with its replacement


@pytest.mark.parametrize("start", [175, 180, 190, 195])
def test_a_multi_word_environment_value_cut_by_the_upstream_limit_loses_the_word_the_cut_split(monkeypatch, start):
    # known limit (module docstring): the words before the cut are no longer the whole value, so they can stay
    monkeypatch.setenv("MSD_FIX_API_KEY", MULTI_WORD)
    text = upstream(WORDS[:start] + MULTI_WORD + " and then more words")
    assert not text.endswith(MULTI_WORD)
    shown = public_row(verdict(text))["excerpt"]
    assert shown.endswith(MARK) and len(shown) <= 200
    assert "99" not in shown and "stap" not in shown and "sta" not in shown.replace("start", "")


@pytest.mark.parametrize("fault", [chr(0xFFFD), chr(0xD800), chr(0xDC00)], ids=["replacement", "high-surrogate", "low-surrogate"])
def test_a_test_judge_hidden_by_a_decoding_fault_is_refused_by_the_check_itself(fault):
    for judge in ("fa" + fault + "ke-judge", "f" + fault + "a" + fault + "k" + fault + "e", "FAKE" + fault, fault + "Fake-judge"):
        assert looks_fake(judge), judge
        with pytest.raises(ValueError, match="test judge"):
            public_row(verdict(judge_id=judge))
    assert not looks_fake("real" + fault + "-judge-v1") and not looks_fake(fault)


def test_every_judge_check_is_the_one_function_that_sees_through_a_decoding_fault():
    from msd import gaterun, redact, reliabilityrun, reportrun
    assert reportrun.looks_fake is gaterun.looks_fake and reliabilityrun.looks_fake is gaterun.looks_fake
    assert redact.looks_fake is gaterun.looks_fake
