import pytest

from msd.scoring import normalize_answer, score_task


# --- normalize_answer --------------------------------------------------


def test_normalize_arabic_indic_digits():
    assert normalize_answer("٣") == normalize_answer("3") == "3"


@pytest.mark.parametrize(
    "digits, ascii_digits",
    [
        ("०१२३४५६७८९", "0123456789"),  # Devanagari
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),  # Arabic-Indic
        ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),  # Extended Arabic-Indic (Persian/Urdu)
        ("０１２３", "0123"),  # fullwidth, via NFKC
    ],
)
def test_normalize_maps_every_decimal_digit_script(digits, ascii_digits):
    assert normalize_answer(digits) == ascii_digits


def test_normalize_multidigit_number_across_scripts():
    assert normalize_answer("१२") == normalize_answer("١٢") == normalize_answer("12") == "12"


@pytest.mark.parametrize("mark", ["।", "؟", "،", "۔", ".", "?", ",", "!", ";", ":", '"', "(", ")"])
def test_normalize_strips_punctuation_including_non_ascii(mark):
    assert normalize_answer(f"yes{mark}") == "yes"
    assert normalize_answer(f"{mark}yes{mark}") == "yes"


def test_normalize_collapses_whitespace_and_casefolds():
    assert normalize_answer("  The   CAPITAL \n is\tParis ") == "the capital is paris"
    assert normalize_answer("Straße") == normalize_answer("STRASSE") == "strasse"


def test_normalize_does_not_glue_numbers_across_punctuation():
    # Punctuation becomes a gap, so "3.5" cannot collide with "35".
    assert normalize_answer("3.5") != normalize_answer("35")


def test_normalize_empty_and_punctuation_only():
    assert normalize_answer("") == ""
    assert normalize_answer(" ।؟ ") == ""


# --- score_task: multiple-choice letters --------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        "B",
        "b",
        "(B)",
        "B.",
        "B)",
        " B \n",
        "Answer: B",
        "Answer: B.",
        "answer: b",
        "Answer: (B)",
        "The answer is B",
        "The answer is B.",
        "The answer is **B**",
        "B) Paris",
        "B. Paris is the capital.",
        "(B) Paris",
        "Answer: B\nBecause Paris is the capital.",
        "answer: b\nbecause Paris is the capital.",
        "Answer: B, since Paris is the capital.",
    ],
)
def test_mc_letter_extraction(pred):
    assert score_task(pred, "B") is True


@pytest.mark.parametrize(
    "pred",
    [
        "Because it is the capital",  # a word that merely starts with the letter
        "Banana",
        "A man walked in",  # article, not option A
        "C",
        "Answer: C",
        "The answer is C.",
        "",
        "   ",
        "I am not sure.",
    ],
)
def test_mc_letter_rejects_wrong_or_missing_letter(pred):
    assert score_task(pred, "B") is False


def test_mc_ordinary_words_after_answer_are_not_a_letter():
    # "answer a question" / "the answer is a number" use the article "a", not option A.
    assert score_task("To answer a question about Paris, think.", "A") is False
    assert score_task("The answer is a number between one and five.", "A") is False
    assert score_task("The answer is a", "A") is True


def test_mc_gold_letter_is_case_insensitive_and_covers_a_to_e():
    assert score_task("Answer: b", "B")
    assert score_task("B", "b")
    assert score_task("The answer is E", "e")
    assert score_task("(A)", " A ")
    assert not score_task("D", "E")


def test_mc_takes_the_last_stated_answer():
    assert score_task("Answer: A. On reflection, the answer is B", "B") is True
    assert score_task("The answer is B. Actually, answer: C", "B") is False


# --- score_task: free-form answers --------------------------------------


def test_hindi_match_ignores_danda_and_punctuation():
    assert score_task("दिल्ली।", "दिल्ली") is True
    assert score_task("  दिल्ली  ।", "दिल्ली") is True
    assert score_task("मुंबई।", "दिल्ली") is False


def test_arabic_question_mark_and_comma_are_ignored():
    assert score_task("القاهرة؟", "القاهرة") is True
    assert score_task("نعم،", "نعم") is True


def test_numeric_answer_matches_across_digit_scripts():
    assert score_task("٣", "3") is True
    assert score_task("3।", "३") is True
    assert score_task("The answer: ١٢.", "the answer 12") is True
    assert score_task("4", "3") is False


def test_free_form_is_exact_after_normalisation_not_substring():
    assert score_task("Paris.", "paris") is True
    assert score_task("The capital is Paris", "Paris") is False


def test_gold_letter_outside_a_to_e_is_free_form():
    # "F" is not a multiple-choice letter here, so it is compared as text.
    assert score_task("f.", "F") is True
    assert score_task("Answer: F", "F") is False


# --- score_task: cue beats a leading letter ------------------------------


def test_answer_cue_beats_an_echoed_option_list():
    echoed = "A. Paris\nB. London\nC. Rome\nThe answer is C"
    assert score_task(echoed, "C") is True
    assert score_task(echoed, "A") is False  # the leading "A." must not earn false credit
    inline = "A) Paris B) London. Answer: B"
    assert score_task(inline, "B") is True
    assert score_task(inline, "A") is False


def test_option_cue_beats_a_leading_letter_but_a_correct_option_cue_beats_option_chatter():
    assert score_task("A. Paris\nB. London\nOption B", "B") is True
    assert score_task("A. Paris\nB. London\nOption B", "A") is False
    # "Option A is wrong" must not override the stated correct option.
    assert score_task("Option A is wrong. The correct option is B", "B") is True
    assert score_task("Option A is wrong. The correct option is B", "A") is False
    # An explicit answer beats later chatter about another option.
    assert score_task("Answer: B. Option A is wrong.", "B") is True
    assert score_task("Answer: B. Option A is wrong.", "A") is False


def test_conflicting_bare_option_mentions_are_ambiguous():
    reply = "Option A is wrong. Option B is correct."
    assert score_task(reply, "A") is False
    assert score_task(reply, "B") is False


# --- score_task: format robustness (English) -----------------------------


@pytest.mark.parametrize(
    "pred",
    [
        "B\nBecause Paris is the capital.",  # explanation on the next line
        "B\n\nParis is the capital.",
        "\n\nB",
        "Ｂ",  # fullwidth, via NFKC
        "**B**",
        "**B.** Paris",
        "`B`",
        "Option B",
        "option b",
        "The correct option is B",
        "The correct answer is B",
        "The right choice: B",
        "B is correct",
        "B is the correct answer",
        "B - Paris",
        "Answer - B",
        "Answer is: B",
    ],
)
def test_mc_format_variants_commit_to_b(pred):
    assert score_task(pred, "B") is True
    assert score_task(pred, "C") is False


def test_mc_guards_still_hold_for_every_gold_letter():
    for gold in "ABCDE":
        for pred in (
            "Because",
            "Banana",
            "A lot of people say C",
            "To answer a question about Paris, think.",
            "The answer is a number.",
        ):
            assert score_task(pred, gold) is False, (pred, gold)


@pytest.mark.parametrize(
    "pred",
    [
        "Answer: B and C",
        "Answer: C or B",
        "Answer: B, C",
        "The answer is B or C",
        "A and B",
        "A, B",
        "B and C",
        "Option B or option C",
    ],
)
def test_hedged_multi_letter_replies_score_false_for_every_named_letter(pred):
    for gold in "ABCDE":
        assert score_task(pred, gold) is False, (pred, gold)


def test_a_comma_that_continues_into_prose_is_not_a_hedge():
    assert score_task("Answer: B, since Paris is the capital", "B") is True
    assert score_task("The answer is B and the reason is geography.", "B") is True


# --- score_task: Hindi, Arabic, Swahili ----------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        "B।",  # bare letter + danda
        "B॥",
        "उत्तर: B",
        "उत्तर है B",
        "सही उत्तर B है",
        "विकल्प B",
        "सही विकल्प B है",
        "उत्तर: B\nक्योंकि पेरिस राजधानी है।",
    ],
)
def test_hindi_mc_replies(pred):
    assert score_task(pred, "B") is True
    assert score_task(pred, "C") is False


@pytest.mark.parametrize(
    "pred",
    [
        "B؟",  # bare letter + Arabic question mark
        "B،",  # Arabic comma
        "الإجابة: B",
        "الاجابة: B",  # without the hamza
        "الإجابة الصحيحة هي B",
        "الجواب: B",
        "الخيار B",
        "الخيار الصحيح هو B",
    ],
)
def test_arabic_mc_replies(pred):
    assert score_task(pred, "B") is True
    assert score_task(pred, "C") is False


@pytest.mark.parametrize(
    "pred",
    [
        "B.",  # bare letter, Latin punctuation
        "Jibu: B",
        "jibu: b",
        "Jibu ni B",
        "Jibu sahihi ni B",
        "Chaguo B",
        "Chaguo sahihi ni B",
    ],
)
def test_swahili_mc_replies(pred):
    assert score_task(pred, "B") is True
    assert score_task(pred, "C") is False


@pytest.mark.parametrize(
    "pred",
    [
        "उत्तर: B और C",
        "उत्तर: B या C",
        "الإجابة: B أو C",
        "الإجابة: B و C",
        "Jibu: B au C",
        "Jibu: B na C",
    ],
)
def test_hedged_native_replies_score_false(pred):
    assert score_task(pred, "B") is False
    assert score_task(pred, "C") is False
