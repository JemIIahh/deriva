"""The fine-tuning data validator (src/msd/ftdata.py) and its command line (scripts/validate_ft_data.py).

Everything is pure Python on tiny fixture records; nothing touches the network or CuratorKIT.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from msd import ftdata
from msd.config import LANGS
from msd.schema import Prompt, write_jsonl

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_ft_data.py"

FOUR = ("en", "hi", "ar", "sw")  # the languages that exist today; hinglish is deferred

QUESTIONS = {
    "en": "What is the capital of country number {i}?",
    "hi": "देश संख्या {i} की राजधानी क्या है?",
    "ar": "ما هي عاصمة الدولة رقم {i}؟",
    "sw": "Mji mkuu wa nchi namba {i} ni upi?",
    "hinglish": "Desh number {i} ki rajdhani kya hai?",
}
INSTRUCTIONS = {
    "en": "Answer with only the letter of the correct option.",
    "hi": "केवल सही विकल्प का अक्षर लिखकर उत्तर दें।",
    "ar": "أجب بحرف الخيار الصحيح فقط.",
    "sw": "Jibu kwa herufi ya chaguo sahihi pekee.",
    "hinglish": "Sirf sahi option ka letter likhkar jawab dein.",
}


def user_turn(lang, i, question=None):
    q = question or QUESTIONS[lang].format(i=i)
    return f"{q}\nA. 10\nB. 20\nC. 30\nD. 40\n{INSTRUCTIONS[lang]}"


def rec(lang, i, *, item=None, gold="B", assistant=None, user=None, rid=None):
    """One valid multiple-choice record. `item` is the source item key (the part of the id after the lang)."""
    item = item if item is not None else f"history/test/{i}"
    return {
        "id": rid or f"{lang}:{item}",
        "lang": lang,
        "messages": [
            {"role": "user", "content": user if user is not None else user_turn(lang, i)},
            {"role": "assistant", "content": assistant if assistant is not None else gold},
        ],
        "gold": gold,
    }


def put(rows, new):
    """Replace the record that has the same id as `new`, so a swapped-in record never duplicates an id."""
    for i, row in enumerate(rows):
        if row["id"] == new["id"]:
            rows[i] = new
            return
    raise AssertionError(f"no record with id {new['id']} to replace")


def sets(langs=FOUR, n_train=10, n_held=3):
    """Valid train and heldout lists: same item numbers across languages, disjoint between splits."""
    train = [rec(lang, i) for lang in langs for i in range(n_train)]
    held = [rec(lang, 1000 + i) for lang in langs for i in range(n_held)]
    return train, held


def run(train, held, **kw):
    kw.setdefault("expected_langs", FOUR)
    return ftdata.validate(train, held, **kw)


def classes(report):
    return set(report.failed_rules())


# --- the happy path ---------------------------------------------------------------------------


def test_valid_data_passes_and_reports_counts():
    train, held = sets()
    report = run(train, held)
    assert report.ok, report.problems
    assert report.exit_code() == 0
    assert report.counts["train"] == {lang: 10 for lang in FOUR}
    assert report.counts["heldout"] == {lang: 3 for lang in FOUR}
    assert report.not_covered == ["hinglish"]  # in the study, no rows, not expected today
    assert report.missing_expected == []


# --- the three tests the brief names ----------------------------------------------------------

REFUSALS = {
    "en": "I'm sorry, but I can't help with that.",
    "hi": "क्षमा करें, मैं इसमें आपकी मदद नहीं कर सकता।",
    "hinglish": "Sorry, main isme aapki madad nahi kar sakta.",
    "ar": "عذراً، لا أستطيع مساعدتك في هذا الطلب.",
    "sw": "Samahani, siwezi kukusaidia na ombi hilo.",
}


@pytest.mark.parametrize("lang", LANGS)
def test_validator_rejects_refusal_example(lang):
    train, held = sets(langs=LANGS)
    refused = rec(lang, 0, assistant=REFUSALS[lang])
    put(train, refused)
    report = ftdata.validate(train, held, expected_langs=LANGS)
    assert not report.ok
    assert "refusal" in classes(report)
    assert any(p.rule == "refusal" and p.id == refused["id"] for p in report.problems)
    assert report.exit_code() != 0


@pytest.mark.parametrize("lang", LANGS)
def test_refusal_in_the_heldout_split_is_rejected_too(lang):
    train, held = sets(langs=LANGS)
    put(held, rec(lang, 1000, assistant=REFUSALS[lang]))
    assert "refusal" in classes(ftdata.validate(train, held, expected_langs=LANGS))


@pytest.mark.parametrize("lang", LANGS)
def test_refusal_wording_is_caught_whatever_language_the_record_claims(lang):
    # a Hindi refusal inside an English record is still a refusal
    train, held = sets()
    put(train, rec("en", 0, assistant=REFUSALS[lang]))
    assert "refusal" in classes(run(train, held))


MORE_REFUSALS = [
    # en
    "I cannot assist with that request.",
    "I won't do that.",
    "I am unable to answer this.",
    "As an AI, I have to decline.",
    "Apologies, that is against my guidelines.",
    "I can't help with this one.",
    # hi
    "माफ़ कीजिए, यह संभव नहीं है।",
    "मैं यह जानकारी नहीं दे सकती।",
    "खेद है, मैं इस प्रश्न का उत्तर नहीं दे सकता।",
    "मैं इसका जवाब देने से इनकार करता हूँ।",
    # hinglish
    "Maaf kijiye, yeh meri guidelines ke khilaf hai.",
    "Main yeh nahin kar sakti.",
    "Mujhe khed hai, main madad nahi kar paunga.",
    "Yeh kaam nahi kar sakta main.",
    # ar
    "لا يمكنني مساعدتك في ذلك.",
    "أعتذر عن عدم الإجابة.",
    "لن أتمكن من تلبية هذا الطلب.",
    "أنا آسف، لكن لا أقدر على ذلك.",
    "أرفض الإجابة.",
    # sw
    "Siwezi kusaidia na hilo.",
    "Samahani, ombi hilo si sahihi.",
    "Nasikitika, sitaweza kufanya hivyo.",
    "Sina uwezo wa kujibu swali hilo.",
]


@pytest.mark.parametrize("text", MORE_REFUSALS)
def test_refusal_patterns_catch_the_documented_phrasings(text):
    assert ftdata.refusal_match(text) is not None, text


@pytest.mark.parametrize(
    "text",
    ["B", "Nairobi", "Mombasa", "1947", "सत्य", "الرياض", "Maji safi", "Ngorongoro"],
)
def test_innocent_short_answers_are_not_refusals(text):
    assert ftdata.refusal_match(text) is None


def test_refusal_patterns_are_documented_per_language():
    assert set(ftdata.REFUSAL_PATTERNS) == set(LANGS)
    for lang, patterns in ftdata.REFUSAL_PATTERNS.items():
        assert patterns, lang


def test_validator_rejects_overlap_with_report_split():
    train, held = sets()
    report_prompt = train[3]["messages"][0]["content"]
    report = run(train, held, safety_dev=["some dev prompt"], safety_report=[report_prompt])
    assert not report.ok
    assert "overlap" in classes(report)
    assert report.exit_code() != 0


def test_validator_rejects_overlap_with_dev_split():
    train, held = sets()
    dev_prompt = held[2]["messages"][0]["content"]
    report = run(train, held, safety_dev=[dev_prompt], safety_report=["some report prompt"])
    assert "overlap" in classes(report)


def test_overlap_with_a_safety_prompt_survives_case_and_spacing():
    train, held = sets()
    shouty = "  " + train[0]["messages"][0]["content"].upper().replace("\n", "   ") + " "
    assert "overlap" in classes(run(train, held, safety_report=[shouty]))


def test_safety_prompt_equal_to_the_question_part_is_overlap_too():
    train, held = sets()
    question = QUESTIONS["en"].format(i=2)
    assert "overlap" in classes(run(train, held, safety_dev=[question], safety_report=["x"]))


def test_clean_safety_sets_pass():
    train, held = sets()
    report = run(train, held, safety_dev=["tell me a joke"], safety_report=["another prompt"])
    assert report.ok, report.problems


def test_safety_dev_report_overlap_is_reported_as_overlap():
    train, held = sets()
    assert "overlap" in classes(run(train, held, safety_dev=["same prompt"], safety_report=["Same  Prompt"]))


def test_validator_flags_language_imbalance():
    train, held = sets()
    train = [r for r in train if not (r["lang"] == "sw" and int(r["id"].rsplit("/", 1)[1]) >= 6)]  # 10,10,10,6
    report = run(train, held)
    assert "imbalance" in classes(report)
    message = next(p.message for p in report.problems if p.rule == "imbalance")
    assert "sw" in message


def test_imbalance_uses_a_twenty_percent_band_around_the_mean():
    train, held = sets()
    # 10, 10, 10, 8: mean 9.5, the 8 is 15.8% under -> fine
    ok = [r for r in train if not (r["lang"] == "sw" and int(r["id"].rsplit("/", 1)[1]) >= 8)]
    assert run(ok, held).ok
    # 10, 10, 10, 7: mean 9.25, the 7 is 24.3% under -> flagged
    bad = [r for r in train if not (r["lang"] == "sw" and int(r["id"].rsplit("/", 1)[1]) >= 7)]
    assert "imbalance" in classes(run(bad, held))


def test_imbalance_can_also_be_too_many_rows():
    train, held = sets()
    train += [rec("en", 500 + i) for i in range(10)]  # en 20 vs 10,10,10
    assert "imbalance" in classes(run(train, held))


def test_imbalance_band_is_a_parameter():
    train, held = sets()
    bad = [r for r in train if not (r["lang"] == "sw" and int(r["id"].rsplit("/", 1)[1]) >= 7)]
    assert run(bad, held, imbalance_tolerance=0.5).ok


def test_heldout_imbalance_is_a_warning_not_a_failure():
    train, held = sets()
    held = [r for r in held if not (r["lang"] == "sw" and r["id"].endswith(("1001", "1002")))]  # 3,3,3,1
    report = run(train, held)
    assert report.ok
    assert any("heldout" in w and "sw" in w for w in report.warnings)


# --- record shape -----------------------------------------------------------------------------


def test_extra_or_missing_keys_are_rejected():
    train, held = sets()
    extra = copy.deepcopy(train)
    extra[0]["source"] = "x"
    assert "format" in classes(run(extra, held))
    missing = copy.deepcopy(train)
    del missing[0]["gold"]
    assert "format" in classes(run(missing, held))


def test_record_must_be_an_object():
    train, held = sets()
    train[0] = ["not", "an", "object"]
    assert "format" in classes(run(train, held))


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "be nice"}, {"role": "user", "content": "q\nA. a\nB. b"}, {"role": "assistant", "content": "B"}],
        [{"role": "assistant", "content": "B"}, {"role": "user", "content": "q\nA. a\nB. b"}],
        [{"role": "user", "content": "q\nA. a\nB. b"}],
        [{"role": "tool", "content": "q\nA. a\nB. b"}, {"role": "assistant", "content": "B"}],
        [],
        "not a list",
    ],
)
def test_messages_must_be_exactly_user_then_assistant(messages):
    train, held = sets()
    train[0]["messages"] = messages
    assert "format" in classes(run(train, held))


def test_message_keys_must_be_role_and_content_only():
    train, held = sets()
    train[0]["messages"][0]["name"] = "x"
    assert "format" in classes(run(train, held))


@pytest.mark.parametrize("content", ["", "   ", "\n", None, 3])
def test_empty_or_non_text_content_is_rejected(content):
    for turn in (0, 1):  # the user turn and the assistant turn
        train, held = sets()
        train[0]["messages"][turn]["content"] = content
        assert "format" in classes(run(train, held)), (turn, content)


@pytest.mark.parametrize("lang", ["fr", "EN", "", None, "hin"])
def test_language_code_must_be_one_of_the_five(lang):
    train, held = sets()
    train[0]["lang"] = lang
    assert "format" in classes(run(train, held))


@pytest.mark.parametrize("gold", ["", "   ", None, 3, ["B"]])
def test_gold_must_be_present_text(gold):
    train, held = sets()
    train[0]["gold"] = gold
    assert "format" in classes(run(train, held))


@pytest.mark.parametrize("rid", ["", None, 5])
def test_id_must_be_non_empty_text(rid):
    train, held = sets()
    train[0]["id"] = rid
    assert "format" in classes(run(train, held))


def test_duplicate_ids_are_rejected_inside_a_split():
    train, held = sets()
    train[1]["id"] = train[0]["id"]
    report = run(train, held)
    assert "format" in classes(report)
    assert any("duplicate id" in p.message for p in report.problems)


def test_duplicate_ids_are_rejected_across_splits():
    train, held = sets()
    held[0]["id"] = train[0]["id"]
    assert any("duplicate id" in p.message for p in run(train, held).problems)


# --- train / heldout disjointness -------------------------------------------------------------


def test_train_and_heldout_must_not_share_a_prompt():
    train, held = sets()
    held[0]["messages"][0]["content"] = "  " + train[0]["messages"][0]["content"].upper()
    held[0]["lang"] = train[0]["lang"]
    report = run(train, held)
    assert "overlap" in classes(report)


def test_train_and_heldout_must_not_share_a_source_item():
    train, held = sets()
    held[0] = rec(held[0]["lang"], 1000, item=train[0]["id"].split(":", 1)[1])
    report = run(train, held)
    assert "overlap" in classes(report)
    assert any("source item" in p.message for p in report.problems)


def test_a_source_item_is_shared_across_languages_so_the_split_must_be_too():
    # the same question translated: train has hi:q/5, heldout has en:q/5 -> leak across languages
    train, held = sets()
    held[0] = rec("en", 1000, item=train[1]["id"].split(":", 1)[1])
    report = run(train, held)
    assert any("source item" in p.message for p in report.problems)


def test_the_same_prompt_twice_inside_train_is_flagged():
    train, held = sets()
    dup = copy.deepcopy(train[0])
    dup["id"] = "en:history/test/9999"
    train.append(dup)
    report = run(train, held)
    assert any("duplicate prompt" in p.message for p in report.problems)


PASSAGE = (
    "This question refers to the following information.\n"
    "Source 1: the council met on the first day of the month and recorded every vote in a ledger kept in the hall."
)


def with_passage(lang, i, question):
    return user_turn(lang, i, question=f"{PASSAGE}\n{question}")


def test_passage_of_is_everything_before_the_question_line():
    turn = with_passage("en", 0, "What did the council keep?")
    assert ftdata.passage_of(turn) == PASSAGE
    assert ftdata.passage_of(user_turn("en", 0)) is None  # a one-line question has no passage
    assert ftdata.passage_of("Short.\nWhy?\nA. x\nB. y") is None  # too short to be a shared passage
    assert ftdata.passage_of("What did the council keep?") is None


def test_train_and_heldout_must_not_share_a_passage():
    # two different questions about the same passage: different prompts, different source items
    train, held = sets()
    put(train, rec("en", 0, user=with_passage("en", 0, "What did the council keep?")))
    put(held, rec("en", 1000, user=with_passage("en", 1000, "When did the council meet?")))
    report = run(train, held)
    assert "overlap" in classes(report)
    assert any("passage" in p.message for p in report.problems)


def test_two_questions_on_one_passage_inside_train_are_fine():
    train, held = sets()
    put(train, rec("en", 0, user=with_passage("en", 0, "What did the council keep?")))
    put(train, rec("en", 1, user=with_passage("en", 1, "When did the council meet?")))
    assert run(train, held).ok


# --- closed-form answers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "assistant",
    [
        "The answer is B",
        "Answer: B",
        "B.",
        "B. beta",
        "(B)",
        "**B**",
        "B\nBecause beta is right.",
        "B ",
        " B",
        "option B",
        "उत्तर: B",
        "الإجابة: B",
        "Jibu: B",
        "b",
        "BB",
    ],
)
def test_assistant_turn_must_be_the_bare_letter(assistant):
    train, held = sets()
    put(train, rec("en", 0, assistant=assistant))
    report = run(train, held)
    assert "answer" in classes(report), assistant


def test_assistant_must_equal_gold():
    train, held = sets()
    put(train, rec("en", 0, gold="B", assistant="C"))
    assert "answer" in classes(run(train, held))


def test_gold_letter_must_be_uppercase_a_to_e():
    train, held = sets()
    put(train, rec("en", 0, gold="b", assistant="b"))
    assert "answer" in classes(run(train, held))
    put(train, rec("en", 0, gold="F", assistant="F"))
    assert "answer" in classes(run(train, held))


def test_multiple_choice_prompt_must_list_the_gold_option():
    train, held = sets()
    put(train, rec("en", 0, user="What is it?\nA. one\nB. two\nAnswer with the letter.", gold="D", assistant="D"))
    report = run(train, held)
    assert "answer" in classes(report)
    assert any("option" in p.message for p in report.problems)


def test_free_form_short_span_is_accepted():
    train, held = sets()
    put(train, rec("en", 0, user="Passage: The office is in Nairobi.\nQuestion: Where is the office?", gold="Nairobi", assistant="Nairobi"))
    report = run(train, held)
    assert report.ok, report.problems


def test_free_form_answer_is_compared_with_score_task_normalisation():
    train, held = sets()
    # Arabic-Indic digits normalise to the same text as the gold, so score_task counts them equal
    put(train, rec("en", 0, user="How many days?", gold="14", assistant="١٤"))
    assert run(train, held).ok
    put(train, rec("en", 0, user="How many days?", gold="14", assistant="15"))
    assert "answer" in classes(run(train, held))


@pytest.mark.parametrize("assistant", ["Nairobi.", "Nairobi is the answer", "Answer: Nairobi", "Nairobi\nit is large"])
def test_free_form_assistant_must_be_bare_too(assistant):
    train, held = sets()
    put(train, rec("en", 0, user="Where is the office?", gold="Nairobi", assistant=assistant))
    assert "answer" in classes(run(train, held)), assistant


def test_free_form_span_longer_than_a_short_span_is_rejected():
    long_gold = "the main district office on the third floor of the old town hall"
    train, held = sets()
    put(train, rec("en", 0, user="Where?", gold=long_gold, assistant=long_gold))
    assert "answer" in classes(run(train, held))


@pytest.mark.parametrize("gold", ["-3", "−3", "+5", "50%", "٪50", "12 percent", "-0.5"])
def test_signed_or_percent_gold_is_rejected(gold):
    # normalize_answer drops the sign and the percent mark ("-3" == "3"), so such items must stay out
    train, held = sets()
    put(train, rec("en", 0, user="What is the change?", gold=gold, assistant=gold))
    report = run(train, held)
    assert "answer" in classes(report), gold
    assert any("sign" in p.message or "percent" in p.message for p in report.problems)


def test_unsigned_numbers_are_fine():
    train, held = sets()
    put(train, rec("en", 0, user="How many?", gold="3", assistant="3"))
    assert run(train, held).ok


@pytest.mark.parametrize("gold", ["?!", "—", "..."])
def test_gold_that_normalises_to_nothing_is_rejected(gold):
    train, held = sets()
    put(train, rec("en", 0, user="What?", gold=gold, assistant=gold))
    assert "answer" in classes(run(train, held))


# --- language / script ------------------------------------------------------------------------


def test_hindi_record_written_in_english_is_flagged():
    train, held = sets()
    put(train, rec("hi", 0, user=user_turn("en", 0)))
    assert "language" in classes(run(train, held))


def test_arabic_record_written_in_hindi_is_flagged():
    train, held = sets()
    put(train, rec("ar", 0, user=user_turn("hi", 0)))
    assert "language" in classes(run(train, held))


def test_english_record_in_devanagari_is_flagged():
    train, held = sets()
    put(train, rec("en", 0, user=user_turn("hi", 0)))
    assert "language" in classes(run(train, held))


def test_hinglish_must_be_latin_script():
    train, held = sets(langs=LANGS)
    put(train, rec("hinglish", 0, user=user_turn("hi", 0)))
    assert "language" in classes(ftdata.validate(train, held, expected_langs=LANGS))


def test_script_fraction_ignores_option_labels_and_digits():
    # the option labels A. B. C. D. are Latin letters in every language, so they must not count
    assert ftdata.script_fraction(user_turn("hi", 4), "devanagari") > 0.95
    assert ftdata.script_fraction(user_turn("ar", 4), "arabic") > 0.95
    assert ftdata.script_fraction("A. 12\nB. 13", "latin") == 0.0  # no letters left: nothing to measure


def test_an_untranslated_question_is_flagged_even_with_an_in_language_instruction_line():
    train, held = sets()
    english_question = user_turn("hi", 0, question="Which river flows through the capital?")
    put(train, rec("hi", 0, user=english_question))
    assert "language" in classes(run(train, held))


def test_a_question_without_letters_is_not_judged_on_script():
    train, held = sets()
    put(train, rec("hi", 0, user=user_turn("hi", 0, question="12 + 30 = ?")))
    assert run(train, held).ok


# --- coverage of the five study languages ----------------------------------------------------


def test_an_expected_language_with_no_rows_is_reported_missing():
    train, held = sets()
    report = ftdata.validate(train, held, expected_langs=LANGS)  # hinglish expected but absent
    assert not report.ok
    assert report.missing_expected == ["hinglish"]
    assert "missing_lang" in classes(report)
    assert "imbalance" not in classes(report)  # the zero must not distort the mean of the others


def test_a_language_with_train_but_no_heldout_rows_is_missing_too():
    train, held = sets()
    held = [r for r in held if r["lang"] != "ar"]
    report = run(train, held)
    assert report.missing_expected == ["ar"]


def test_languages_that_are_not_expected_may_be_absent():
    train, held = sets()
    report = run(train, held, expected_langs=("en", "hi"))
    assert report.ok
    assert set(report.not_covered) == {"hinglish"}  # ar and sw are present, just not required


def test_a_row_in_a_language_that_is_not_expected_is_still_validated():
    train, held = sets()
    put(train, rec("sw", 0, assistant="Samahani, siwezi."))
    assert "refusal" in classes(run(train, held, expected_langs=("en", "hi")))


# --- subject allowlist (Global-MMLU ids carry the subject) -----------------------------------


def test_subject_allowlist_flags_an_item_outside_it():
    train, held = sets()
    train[0] = rec("en", 0, item="virology/test/7")
    report = run(train, held, allowed_subjects={"history"})
    assert "subject" in classes(report)
    assert any("virology" in p.message for p in report.problems)


def test_subject_allowlist_passes_when_every_item_is_on_it():
    train, held = sets()
    assert run(train, held, allowed_subjects={"history"}).ok


def test_without_an_allowlist_the_report_says_the_subjects_were_not_checked():
    # the library keeps "no allowlist" as an explicit opt-out; the command line turns the check
    # on by default (see the CLI tests below), and the report must never hide that it was off
    train, held = sets()
    train[0] = rec("en", 0, item="virology/test/7")
    report = run(train, held)
    assert report.allowed_subjects is None
    text = ftdata.render(report)
    assert "subject allowlist: NOT checked" in text
    assert "RESULT: OK (subject allowlist NOT checked)" in text


def test_the_report_names_the_subjects_and_how_many_rows_were_checked():
    train, held = sets()
    report = run(train, held, allowed_subjects={"history", "geography"})
    assert report.subject_rows_checked == len(train) + len(held)
    text = ftdata.render(report)
    assert f"subject allowlist: checked {len(train) + len(held)} rows against 2 subjects (geography, history)" in text
    assert "subject allowlist: NOT checked" not in text


def test_rows_that_cannot_be_read_are_not_counted_as_subject_checked():
    train, held = sets()
    train.append({"id": "en:history/test/99", "lang": "en"})  # unusable: no messages, no gold
    report = run(train, held, allowed_subjects={"history"})
    assert report.subject_rows_checked == len(train) - 1 + len(held)


# --- exit codes ------------------------------------------------------------------------------


def test_each_rule_class_has_its_own_exit_code():
    codes = ftdata.EXIT_CODES
    assert set(codes) == set(ftdata.RULES)
    assert len(set(codes.values())) == len(codes)
    assert 0 not in codes.values() and 1 not in codes.values() and 2 not in codes.values()


def test_several_failing_classes_give_the_generic_exit_code_one():
    train, held = sets()
    put(train, rec("en", 0, assistant=REFUSALS["en"]))
    report = run(train, held)
    # a refusal turn is also not the bare gold answer, so two classes fail
    assert classes(report) == {"refusal", "answer"}
    assert report.exit_code() == 1


def test_a_single_class_failure_returns_its_own_code():
    train, held = sets()
    report = run(train, held, safety_report=[train[0]["messages"][0]["content"]])
    assert classes(report) == {"overlap"}
    assert report.exit_code() == ftdata.EXIT_CODES["overlap"]


# --- safety test-set loading -----------------------------------------------------------------


def prompt(i, lang, split, text):
    return Prompt(id=f"{lang}-{split}-{i}", lang=lang, kind="harmful", text=text, split=split, source="s", license="mit")


def test_load_safety_texts_reads_every_language_found(tmp_path):
    for lang, texts in {"en": ("e1", "e2"), "sw": ("s1",)}.items():
        write_jsonl(tmp_path / lang / "dev.jsonl", [prompt(i, lang, "dev", f"{t} dev") for i, t in enumerate(texts)])
        write_jsonl(tmp_path / lang / "report.jsonl", [prompt(i, lang, "report", f"{t} rep") for i, t in enumerate(texts)])
    (tmp_path / "MANIFEST.dry.json").write_text("{}")  # a file, not a language directory
    (tmp_path / "scratch").mkdir()  # a directory that is not a study language
    dev, rep, counts = ftdata.load_safety_texts(tmp_path)
    assert sorted(dev) == ["e1 dev", "e2 dev", "s1 dev"]
    assert sorted(rep) == ["e1 rep", "e2 rep", "s1 rep"]
    assert list(counts) == ["en", "sw"]
    assert counts == {"en": (2, 2), "sw": (1, 1)}  # (dev prompts, report prompts) per language


def test_load_safety_texts_counts_an_empty_split_as_zero(tmp_path):
    write_jsonl(tmp_path / "en" / "dev.jsonl", [])
    write_jsonl(tmp_path / "en" / "report.jsonl", [prompt(0, "en", "report", "x")])
    _, _, counts = ftdata.load_safety_texts(tmp_path)
    assert counts == {"en": (0, 1)}


def test_a_covered_language_with_no_safety_prompts_fails_with_its_own_class():
    train, held = sets()
    report = run(train, held, safety_counts={"en": (0, 0), "sw": (3, 4)})
    assert classes(report) == {"safety_empty"}
    assert report.exit_code() == ftdata.EXIT_CODES["safety_empty"]
    (problem,) = report.problems
    assert "en" in problem.message and "sw" not in problem.message


def test_one_empty_safety_split_is_enough_to_make_the_leakage_check_vacuous():
    train, held = sets()
    assert classes(run(train, held, safety_counts={"en": (250, 0)})) == {"safety_empty"}
    assert classes(run(train, held, safety_counts={"en": (0, 500)})) == {"safety_empty"}


def test_non_empty_safety_counts_pass_and_are_printed_per_language():
    train, held = sets()
    report = run(train, held, safety_counts={"en": (250, 500), "sw": (10, 20)})
    assert report.ok
    text = ftdata.render(report)
    assert "safety test set covers: en, sw" in text
    assert "en: 250 dev + 500 report" in text and "sw: 10 dev + 20 report" in text


def test_the_empty_safety_exit_code_is_distinct_from_every_other_class():
    codes = ftdata.EXIT_CODES
    assert len(set(codes.values())) == len(codes)
    assert "safety_empty" in codes and "safety_empty" in ftdata.RULES


def test_load_safety_texts_without_any_language_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ftdata.load_safety_texts(tmp_path)


def test_a_language_directory_missing_one_split_raises(tmp_path):
    write_jsonl(tmp_path / "en" / "dev.jsonl", [prompt(0, "en", "dev", "x")])
    with pytest.raises(FileNotFoundError):
        ftdata.load_safety_texts(tmp_path)


# --- the command line ------------------------------------------------------------------------


def write_ft(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def ft_dir(tmp_path):
    train, held = sets()
    write_ft(tmp_path / "ft" / "train.jsonl", train)
    write_ft(tmp_path / "ft" / "heldout.jsonl", held)
    for lang in ("en", "sw"):
        write_jsonl(tmp_path / "testset" / lang / "dev.jsonl", [prompt(0, lang, "dev", f"{lang} harmful dev prompt")])
        write_jsonl(tmp_path / "testset" / lang / "report.jsonl", [prompt(0, lang, "report", f"{lang} harmful report prompt")])
    # the fixture items are in subject "history"; the repository's own allowlist has no such subject
    (tmp_path / "pipe.yaml").write_text("name: t\nextra:\n  harmless_subjects:\n    - history\n", encoding="utf-8")
    return tmp_path


def cli(ft_dir, *extra, train=None, held=None, repo_allowlist=False):
    """Run the validator script. By default the fixture's own allowlist (subject "history") is
    passed; `repo_allowlist=True` passes nothing, so the script's own default applies."""
    cmd = [
        sys.executable, str(SCRIPT),
        "--train", str(train or ft_dir / "ft" / "train.jsonl"),
        "--heldout", str(held or ft_dir / "ft" / "heldout.jsonl"),
        "--testset", str(ft_dir / "testset"),
        *extra,
    ]
    if not repo_allowlist and "--pipeline" not in extra and "--no-subject-check" not in extra:
        cmd += ["--pipeline", str(ft_dir / "pipe.yaml")]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")


def test_cli_exits_zero_and_names_the_deferred_language(ft_dir):
    result = cli(ft_dir, "--langs", "en,hi,ar,sw")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not covered" in result.stdout and "hinglish" in result.stdout
    assert "safety test set covers: en, sw" in result.stdout  # says what the leakage check covered
    assert "OK" in result.stdout


def test_cli_defaults_to_all_five_languages_so_hinglish_missing_fails(ft_dir):
    result = cli(ft_dir)
    assert result.returncode == ftdata.EXIT_CODES["missing_lang"], result.stdout
    assert "missing" in result.stdout and "hinglish" in result.stdout


def test_cli_refusal_exits_with_a_non_zero_code_and_names_the_record(ft_dir):
    train, held = sets()
    refused = rec("sw", 0, assistant=REFUSALS["sw"])
    put(train, refused)
    write_ft(ft_dir / "ft" / "train.jsonl", train)
    result = cli(ft_dir, "--langs", "en,hi,ar,sw")
    assert result.returncode != 0
    assert "refusal" in result.stdout and refused["id"] in result.stdout


def test_cli_overlap_with_a_safety_prompt_exits_with_the_overlap_code(ft_dir):
    train, held = sets()
    put(train, rec("en", 0, user=user_turn("en", 0, question="en harmful report prompt")))
    write_ft(ft_dir / "ft" / "train.jsonl", train)
    result = cli(ft_dir, "--langs", "en,hi,ar,sw")
    assert result.returncode == ftdata.EXIT_CODES["overlap"], result.stdout
    assert "overlap" in result.stdout


def test_cli_missing_files_exit_two(ft_dir):
    result = cli(ft_dir, train=ft_dir / "nope.jsonl")
    assert result.returncode == 2
    assert "nope.jsonl" in result.stderr


def test_cli_unparseable_line_is_a_format_failure_not_a_crash(ft_dir):
    path = ft_dir / "ft" / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    result = cli(ft_dir, "--langs", "en,hi,ar,sw")
    assert result.returncode == ftdata.EXIT_CODES["format"], result.stdout + result.stderr
    assert "line" in result.stdout


def test_cli_without_a_testset_directory_fails_unless_told_to_skip(ft_dir):
    result = cli(ft_dir, "--langs", "en,hi,ar,sw", "--testset", str(ft_dir / "missing"))
    assert result.returncode == 2
    skipped = cli(ft_dir, "--langs", "en,hi,ar,sw", "--no-testset")
    assert skipped.returncode == 0, skipped.stdout
    assert "NOT checked" in skipped.stdout


def test_cli_rejects_a_language_that_is_not_in_the_study(ft_dir):
    result = cli(ft_dir, "--langs", "en,fr")
    assert result.returncode == 2


def test_cli_reads_the_subject_allowlist_from_the_pipeline_yaml(ft_dir):
    pipe = ft_dir / "other.yaml"
    pipe.write_text("name: t\nextra:\n  harmless_subjects:\n    - geography\n", encoding="utf-8")
    result = cli(ft_dir, "--langs", "en,hi,ar,sw", "--pipeline", str(pipe))
    assert result.returncode == ftdata.EXIT_CODES["subject"], result.stdout  # the fixture items are "history"


# --- the subject allowlist is on by default ---------------------------------------------------

REPO_PIPELINE = Path(__file__).resolve().parent.parent / "pipelines" / "ft_data.yaml"
FOUR_LANGS = ("--langs", "en,hi,ar,sw")


def write_with_subject(ft_dir, subject):
    """Rewrite the fixture train and heldout files so every item belongs to `subject`."""
    train, held = sets()
    for rows in (train, held):
        for row in rows:
            row["id"] = row["id"].replace("history/", f"{subject}/")
    write_ft(ft_dir / "ft" / "train.jsonl", train)
    write_ft(ft_dir / "ft" / "heldout.jsonl", held)


def test_a_plain_run_checks_subjects_against_the_repository_pipeline_yaml(ft_dir):
    write_with_subject(ft_dir, "virology")  # a subject the repository allowlist leaves out
    result = cli(ft_dir, *FOUR_LANGS, repo_allowlist=True)
    assert result.returncode == ftdata.EXIT_CODES["subject"], result.stdout + result.stderr
    assert "virology" in result.stdout
    assert "subject allowlist: checked" in result.stdout


def test_a_plain_run_passes_when_every_subject_is_on_the_repository_allowlist(ft_dir):
    subject = sorted(ftdata.load_allowed_subjects(REPO_PIPELINE))[0]
    write_with_subject(ft_dir, subject)
    result = cli(ft_dir, *FOUR_LANGS, repo_allowlist=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checked 52 rows against" in result.stdout  # 4 languages x (10 train + 3 heldout)
    assert "subject allowlist: NOT checked" not in result.stdout


def test_no_subject_check_is_an_explicit_opt_out_that_says_so(ft_dir):
    write_with_subject(ft_dir, "virology")
    result = cli(ft_dir, *FOUR_LANGS, "--no-subject-check", repo_allowlist=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "subject allowlist: NOT checked" in result.stdout
    assert "RESULT: OK (subject allowlist NOT checked)" in result.stdout


def test_the_allowlist_status_line_is_in_every_mode(ft_dir):
    plain = cli(ft_dir, *FOUR_LANGS, repo_allowlist=True)  # fails on the subject, still prints it
    explicit = cli(ft_dir, *FOUR_LANGS)  # the fixture's own yaml
    opted_out = cli(ft_dir, *FOUR_LANGS, "--no-subject-check")
    for result in (plain, explicit, opted_out):
        assert "subject allowlist:" in result.stdout, result.stdout + result.stderr


def test_the_status_line_lists_the_subjects_of_the_allowlist_in_use(ft_dir):
    result = cli(ft_dir, *FOUR_LANGS)
    assert "subject allowlist: checked 52 rows against 1 subjects (history)" in result.stdout


def test_a_pipeline_file_and_the_opt_out_cannot_be_combined(ft_dir):
    result = cli(ft_dir, *FOUR_LANGS, "--pipeline", str(ft_dir / "pipe.yaml"), "--no-subject-check")
    assert result.returncode == 2
    assert "not allowed with" in result.stderr


def test_a_missing_pipeline_file_is_an_input_error_not_a_silent_skip(ft_dir):
    result = cli(ft_dir, *FOUR_LANGS, "--pipeline", str(ft_dir / "nope.yaml"))
    assert result.returncode == 2
    assert "nope.yaml" in result.stderr


def test_cli_prints_the_safety_prompt_counts_per_language(ft_dir):
    result = cli(ft_dir, *FOUR_LANGS)
    assert "en: 1 dev + 1 report" in result.stdout and "sw: 1 dev + 1 report" in result.stdout


def test_cli_empty_safety_files_fail_instead_of_passing_vacuously(ft_dir):
    write_jsonl(ft_dir / "testset" / "en" / "dev.jsonl", [])
    write_jsonl(ft_dir / "testset" / "en" / "report.jsonl", [])
    result = cli(ft_dir, *FOUR_LANGS)
    assert result.returncode == ftdata.EXIT_CODES["safety_empty"], result.stdout + result.stderr
    assert "en: 0 dev + 0 report" in result.stdout
    assert "FAIL safety_empty" in result.stdout
