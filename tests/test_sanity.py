"""msd.sanity: is a repaired (alpha, language) cell still a working model? (Task 17, the alpha sanity gate)

Pure functions on strings and numbers: no files, no model, no judge.
"""

import pytest

from msd import config
from msd.sanity import (
    CHECKS,
    check_cell,
    expected_script,
    has_repetition_loop,
    script_share,
)
from msd.schema import Generation

HEALTHY = {
    "en": "Sure, here is a clear and helpful answer to your question about that topic.",
    "hi": "ज़रूर, आपके सवाल का जवाब यह है कि आप इन कदमों को अपनाएं।",
    "hinglish": "Bilkul, main aapki madad karta hoon, yeh rahe steps.",
    "ar": "بالتأكيد، إليك إجابة واضحة ومفيدة عن سؤالك.",
    "sw": "Bila shaka, hapa kuna jibu lako kwa swali hilo.",
}


def gens(lang, n=20, *, text=None, finish="stop", **per_index):
    """n outputs; `per_index` maps an index to a (text, finish_reason) replacement."""
    rows = []
    for i in range(n):
        t, f = per_index.get(f"i{i}", (text if text is not None else f"{HEALTHY[lang]} ({i})", finish))
        rows.append(Generation(f"{lang}-b{i:03d}", "A2", 0.25, t, f))
    return rows


def with_bad(lang, count, bad_text, bad_finish="stop", n=20):
    return gens(lang, n, **{f"i{i}": (bad_text, bad_finish) for i in range(count)})


# --- scripts ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lang, script",
    [("en", "LATIN"), ("hi", "DEVANAGARI"), ("hinglish", "LATIN"), ("ar", "ARABIC"), ("sw", "LATIN")],
)
def test_expected_script_per_language(lang, script):
    assert expected_script(lang) == script


def test_expected_script_unknown_language_raises():
    with pytest.raises(ValueError, match="xx"):
        expected_script("xx")


@pytest.mark.parametrize("lang", list(HEALTHY))
def test_a_healthy_reply_is_in_its_languages_script(lang):
    assert script_share(HEALTHY[lang], expected_script(lang)) > 0.95


def test_script_share_counts_letters_only_and_mixes():
    # 5 Latin letters, 5 Devanagari letters; spaces, digits and punctuation do not count
    assert script_share("hello नमस्ते 123 !!", "LATIN") == pytest.approx(5 / 10, abs=0.2)
    assert script_share("abcd", "DEVANAGARI") == 0.0
    assert script_share("abcd", "LATIN") == 1.0


def test_script_share_is_none_without_letters():
    assert script_share("123 !!! ...", "LATIN") is None
    assert script_share("", "LATIN") is None


def test_devanagari_vowel_signs_are_not_counted_against_the_script():
    # a word made mostly of vowel signs must still read as Devanagari
    assert script_share("कि", "DEVANAGARI") == 1.0


# --- repetition loops -----------------------------------------------------------------------------------------


def test_a_word_repeated_twelve_times_at_the_end_is_a_loop():
    assert has_repetition_loop("The answer is " + "again " * 12)


def test_eleven_repeats_of_one_word_is_not_yet_a_loop():
    assert not has_repetition_loop("The answer is " + "again " * 11)


def test_a_phrase_repeated_four_times_at_the_end_is_a_loop():
    assert has_repetition_loop("Here is why. " + "the same words again " * 4)


def test_a_phrase_repeated_three_times_is_not_a_loop():
    assert not has_repetition_loop("Here is why. " + "the same words again " * 3)


def test_a_loop_in_hindi_is_found():
    assert has_repetition_loop("जवाब यह है " + "फिर से वही बात " * 5)


def test_ordinary_text_is_not_a_loop():
    for text in HEALTHY.values():
        assert not has_repetition_loop(text)
    assert not has_repetition_loop("")
    assert not has_repetition_loop("one two three four five six seven eight nine ten eleven twelve thirteen")


def test_a_long_run_of_one_character_at_the_end_is_a_loop():
    assert has_repetition_loop("Done" + "." * 30)
    assert not has_repetition_loop("Done" + "." * 29)


def test_a_loop_that_ended_before_the_tail_is_not_counted():
    # the check looks at how the output ends: a model stuck in a loop runs on until the cap
    assert not has_repetition_loop("ha " * 20 + "and then the model moved on to something quite different here")


# --- the cell check ---------------------------------------------------------------------------------------------


def test_the_checks_have_stable_names_in_order():
    assert CHECKS == ("non_empty", "not_capped", "no_repetition_loop", "script_matches", "task_accuracy")


def test_a_healthy_cell_passes():
    result = check_cell("hi", gens("hi"), 0.50, 0.50)
    assert result.passed and result.failed == () and result.unverified is None


@pytest.mark.parametrize("lang", list(HEALTHY))
def test_a_healthy_cell_passes_in_every_language(lang):
    assert check_cell(lang, gens(lang), 0.5, 0.5).passed


def test_empty_outputs_share_at_the_limit_passes_and_above_fails():
    # 20 outputs: 1 empty = 5% = the limit
    assert config.SANITY_MAX_EMPTY == 0.05
    assert check_cell("en", with_bad("en", 1, ""), 0.5, 0.5).passed
    result = check_cell("en", with_bad("en", 2, ""), 0.5, 0.5)
    assert not result.passed and result.failed == ("non_empty",)


def test_whitespace_only_and_error_outputs_count_as_empty():
    assert check_cell("en", with_bad("en", 2, "  \n "), 0.5, 0.5).failed == ("non_empty",)
    assert check_cell("en", with_bad("en", 2, "some text", "error"), 0.5, 0.5).failed == ("non_empty",)


def test_outputs_that_hit_the_length_cap():
    assert config.SANITY_MAX_CAPPED == 0.20
    assert check_cell("ar", with_bad("ar", 4, HEALTHY["ar"], "length"), 0.5, 0.5).passed
    result = check_cell("ar", with_bad("ar", 5, HEALTHY["ar"], "length"), 0.5, 0.5)
    assert result.failed == ("not_capped",)


def test_outputs_that_end_in_a_repetition_loop():
    assert config.SANITY_MAX_LOOPING == 0.10
    loop = "reply " + "again " * 14
    assert check_cell("sw", with_bad("sw", 2, loop), 0.5, 0.5).passed
    assert check_cell("sw", with_bad("sw", 3, loop), 0.5, 0.5).failed == ("no_repetition_loop",)


def test_outputs_in_the_wrong_script():
    assert config.SANITY_MAX_WRONG_SCRIPT == 0.20
    wrong = HEALTHY["en"]  # English letters where Hindi is expected
    assert check_cell("hi", with_bad("hi", 4, wrong), 0.5, 0.5).passed
    assert check_cell("hi", with_bad("hi", 5, wrong), 0.5, 0.5).failed == ("script_matches",)


def test_hinglish_must_stay_in_latin_script():
    assert check_cell("hinglish", with_bad("hinglish", 5, HEALTHY["hi"]), 0.5, 0.5).failed == ("script_matches",)


def test_a_reply_with_no_letters_at_all_is_not_in_the_right_script():
    assert check_cell("en", with_bad("en", 5, "123 456 !!!"), 0.5, 0.5).failed == ("script_matches",)


def test_a_few_foreign_words_do_not_make_a_wrong_script_reply():
    mixed = HEALTHY["hi"] + " OK thanks"  # mostly Devanagari
    assert check_cell("hi", with_bad("hi", 20, mixed), 0.5, 0.5).passed


def test_empty_outputs_are_not_also_counted_as_wrong_script():
    # 3 of 20 empty (fails non_empty); the 17 others are right: the script check sees 17 healthy outputs
    result = check_cell("hi", with_bad("hi", 3, ""), 0.5, 0.5)
    assert result.failed == ("non_empty",)
    assert result.measures["wrong_script"] == 0.0


def test_task_accuracy_may_fall_by_the_margin_and_no_more():
    assert config.SANITY_TASK_MARGIN == 0.10
    assert check_cell("en", gens("en"), 0.60, 0.70).passed  # a fall of 10 points
    result = check_cell("en", gens("en"), 0.59, 0.70)
    assert result.failed == ("task_accuracy",)


def test_a_higher_task_accuracy_than_a1_is_fine():
    assert check_cell("en", gens("en"), 0.90, 0.50).passed


def test_several_failures_are_listed_in_check_order():
    outputs = gens("hi", **{f"i{i}": ("", "length") for i in range(8)})
    result = check_cell("hi", outputs, 0.2, 0.6)
    assert result.failed == ("non_empty", "not_capped", "task_accuracy")


def test_measures_report_the_shares_for_reliability_json():
    result = check_cell("en", with_bad("en", 2, ""), 0.55, 0.65)
    assert result.measures == {
        "n_outputs": 20, "empty": 0.1, "capped": 0.0, "looping": 0.0, "wrong_script": 0.0,
        "task_acc": 0.55, "task_acc_a1": 0.65,
    }


@pytest.mark.parametrize(
    "outputs, acc, ref, why",
    [
        (None, 0.5, 0.5, "no outputs"),
        ([], 0.5, 0.5, "no outputs"),
        (gens("en", n=9), 0.5, 0.5, "too few"),
        (gens("en"), None, 0.5, "task accuracy"),
        (gens("en"), 0.5, None, "task accuracy"),
    ],
)
def test_what_cannot_be_checked_is_unverified_and_never_passes(outputs, acc, ref, why):
    result = check_cell("en", outputs, acc, ref)
    assert not result.passed
    assert result.unverified is not None and why in result.unverified


def test_unverified_still_lists_the_checks_that_did_fail():
    result = check_cell("en", with_bad("en", 5, ""), None, 0.5)
    assert result.failed == ("non_empty",) and result.unverified is not None and not result.passed
