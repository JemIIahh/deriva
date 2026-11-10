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


def test_a_four_word_phrase_repeated_three_times_is_a_loop_because_the_tail_is_twelve_words():
    # fewer repeats than the first version asked for: a 30-word block repeated 4 times does not fit in 256 tokens
    assert has_repetition_loop("Here is why. " + "the same words again " * 3)


def test_a_phrase_repeated_twice_is_not_a_loop():
    # three repeats are enough once the tail is long enough (see SANITY_LOOP_REPEATS); two never are
    assert not has_repetition_loop("Here is why. " + "the same words again " * 2)


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


def test_a_reply_with_no_letters_at_all_fails_the_cell():
    # it used to be reported as the wrong script; a reply with no letters is near-empty (SANITY_MIN_LETTERS), which
    # is checked first and takes it out of the script check
    assert check_cell("en", with_bad("en", 5, "123 456 !!!"), 0.5, 0.5).failed == ("non_empty",)


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


# --- loops cut off by the length cap (review finding: the commonest real loop was missed) ---------------------------

CUT_LOOPS = [
    pytest.param(("जवाब यह है " + "फिर से वही बात " * 30)[:-3], id="hindi-cut-mid-word"),
    pytest.param(("Here is why. " + "I cannot help with that. " * 30)[:-5], id="english-cut-mid-word"),
    pytest.param("हा" * 60, id="hindi-no-spaces"),
    pytest.param("ab" * 60, id="latin-no-spaces"),
    pytest.param(("مرة أخرى نفس الكلام " * 30)[:-4], id="arabic-cut-mid-word"),
    pytest.param(("Bilkul, yeh raha jawab. " + "main madad nahi kar sakta " * 20)[:-2], id="hinglish-cut"),
    pytest.param("Here: " + "the quick brown fox jumps over the lazy dog and then goes home again today " * 3,
                 id="a-long-block-repeated-three-times"),
]


@pytest.mark.parametrize("text", CUT_LOOPS)
def test_a_loop_cut_off_mid_word_by_the_cap_is_found(text):
    assert has_repetition_loop(text)


def test_the_loop_style_the_fixtures_use_still_trips_it():
    assert has_repetition_loop("Sure, here is a clear answer. " + "फिर से वही बात " * 10)
    assert has_repetition_loop("Sure, here is a clear answer. " + "the same words again " * 10)


FALSE_ALARMS = [
    pytest.param("Is it ready? Yes, yes, yes!", id="short-yes-list"),
    pytest.param("Verse one is long and winding. Chorus: la la la la. Verse two is different. Chorus: la la la la.",
                 id="refrain-used-twice"),
    pytest.param("Here is the table:\n0 0 0 0\n", id="a-row-of-zeros"),
    pytest.param("Counts per bin: 0 0 0 0 0 0 1 0 0 0 0 0", id="mostly-zeros-with-a-one"),
    pytest.param("The refrain goes: Hey now, hey now, we go. Hey now, hey now, we go. That is all.", id="refrain-twice"),
    pytest.param("hahaha that is funny", id="laughter"),
    pytest.param("Step 1: mix. Step 2: stir. Step 3: bake. Step 4: cool. Step 5: serve. Step 6: enjoy the result.",
                 id="numbered-steps"),
]


@pytest.mark.parametrize("text", FALSE_ALARMS)
def test_ordinary_text_with_a_repeated_short_phrase_is_not_a_loop(text):
    assert not has_repetition_loop(text)


@pytest.mark.parametrize("lang, loop", [("hi", "फिर से वही बात "), ("ar", "مرة أخرى نفس الكلام ")])
def test_a_cell_of_mid_word_loops_that_ran_to_the_cap_fails_the_loop_check(lang, loop):
    cut = (HEALTHY[lang] + " " + loop * 40)[:-3]
    outputs = with_bad(lang, 18, cut, bad_finish="length", n=100)  # 18% of 100 outputs; capped 18% is under 20%
    result = check_cell(lang, outputs, 0.5, 0.5)
    assert result.measures["looping"] == pytest.approx(0.18)
    assert result.failed == ("no_repetition_loop",)


# --- near-empty replies (review finding) ------------------------------------------------------------------------


@pytest.mark.parametrize("lang, terse", [("hi", "नहीं।"), ("en", "No."), ("ar", "لا."), ("sw", "La.")])
def test_a_cell_of_one_word_replies_fails_the_non_empty_check(lang, terse):
    outputs = gens(lang, 100, text=terse)
    result = check_cell(lang, outputs, 0.5, 0.5)
    assert result.measures["empty"] == 1.0
    assert result.failed == ("non_empty",)  # not also a wrong-script failure: a near-empty reply has no script to judge


def test_a_few_terse_replies_among_real_ones_do_not_fail_the_cell():
    outputs = with_bad("hi", 4, "नहीं।", n=100)  # 4% is under the 5% limit
    assert check_cell("hi", outputs, 0.5, 0.5).passed


def test_a_short_but_real_sentence_is_not_near_empty():
    assert config.SANITY_MIN_LETTERS == 3
    assert check_cell("en", gens("en", 100, text="Yes, sure thing."), 0.5, 0.5).passed
    assert check_cell("en", gens("en", 100, text="Yes."), 0.5, 0.5).passed  # exactly three letters


# --- task accuracy needs enough items too (review finding) --------------------------------------------------------


def test_one_of_one_correct_task_item_is_not_evidence():
    result = check_cell("en", gens("en"), 1.0, 1.0, dev_task_n=1, a1_task_n=200)
    assert not result.passed and result.unverified and "task" in result.unverified and "too few" in result.unverified
    result = check_cell("en", gens("en"), 1.0, 1.0, dev_task_n=200, a1_task_n=3)
    assert not result.passed and result.unverified and "A1" in result.unverified and "too few" in result.unverified


def test_enough_task_items_leave_the_check_to_the_accuracies():
    from msd.stats import MIN_N

    assert check_cell("en", gens("en"), 0.5, 0.5, dev_task_n=MIN_N, a1_task_n=MIN_N).passed
    assert check_cell("en", gens("en"), 0.1, 0.5, dev_task_n=MIN_N, a1_task_n=MIN_N).failed == ("task_accuracy",)


# --- round 2: the task-accuracy check applies only where there are task items (config, never inferred) -------------


def test_the_languages_with_task_items_are_named_in_config_with_a_reason():
    assert config.TASK_ACCURACY_LANGS == ("en", "hi", "ar", "sw")
    assert set(config.TASK_ACCURACY_LANGS) < set(config.LANGS) and "hinglish" not in config.TASK_ACCURACY_LANGS
    assert config.TASK_ACCURACY_REASON.strip()  # the one-line why, shown next to the exemption in reliability.json


def test_a_language_without_task_items_has_no_task_check_and_can_pass():
    result = check_cell("hinglish", gens("hinglish"), None, None)
    assert result.passed and result.failed == () and result.unverified is None
    assert result.task_accuracy == "not_applicable"


def test_the_exemption_comes_from_config_not_from_missing_numbers():
    # numbers given for an exempt language are not used: a "fall" in them is not a failure, and none is shown
    result = check_cell("hinglish", gens("hinglish"), 0.05, 0.95)
    assert result.passed and result.task_accuracy == "not_applicable"
    assert result.measures["task_acc"] is None and result.measures["task_acc_a1"] is None


@pytest.mark.parametrize("lang", ["en", "hi", "hinglish", "ar", "sw"])
def test_every_other_check_still_applies_to_a_language_without_task_items(lang):
    loop = "फिर से वही बात " if lang == "hi" else "مرة أخرى نفس الكلام " if lang == "ar" else "the same words again "
    cases = {
        "no_repetition_loop": gens(lang, text=f"{HEALTHY[lang]} " + loop * 10),
        "non_empty": with_bad(lang, 5, ""),
        "not_capped": gens(lang, finish="length"),
        "script_matches": with_bad(lang, 10, HEALTHY["en"] if lang in ("hi", "ar") else HEALTHY["hi"]),
    }
    for check, outputs in cases.items():
        result = check_cell(lang, outputs, None, None)
        assert check in result.failed, (lang, check, result)
        assert not result.passed
        if lang not in config.TASK_ACCURACY_LANGS:
            assert result.task_accuracy == "not_applicable"
            assert result.failed == (check,)  # and nothing else is reported for the task check


@pytest.mark.parametrize("lang", list(config.TASK_ACCURACY_LANGS))
def test_missing_task_data_for_a_language_that_has_task_items_stays_unverified(lang):
    for acc, ref in ((None, None), (None, 0.5), (0.5, None)):
        result = check_cell(lang, gens(lang), acc, ref)
        assert not result.passed and result.unverified and "task accuracy" in result.unverified
        assert result.task_accuracy == "unverified"


def test_the_task_accuracy_status_says_what_happened_for_a_language_with_task_items():
    assert check_cell("en", gens("en"), 0.6, 0.6).task_accuracy == "passed"
    assert check_cell("en", gens("en"), 0.2, 0.6).task_accuracy == "failed"
    assert check_cell("en", gens("en"), 1.0, 1.0, dev_task_n=1, a1_task_n=200).task_accuracy == "unverified"
