import pytest

from msd.splits import LeakageError, assert_no_leakage, norm_text


def test_norm_text_nfkc_casefold_and_whitespace():
    assert norm_text("  How   do I\tRENEW?\n") == "how do i renew?"
    # NFKC folds the full-width form; casefold maps sharp s to ss.
    assert norm_text("ＡＢＣ") == "abc"
    assert norm_text("Straße") == "strasse"


def test_leak_dev_report_after_normalisation():
    with pytest.raises(LeakageError):
        assert_no_leakage(["How do I renew?"], [" how do I  RENEW? "])


def test_train_overlap_raises():
    with pytest.raises(LeakageError):
        assert_no_leakage(["alpha one"], ["beta two"], train=["x", " BETA   two"])
    with pytest.raises(LeakageError):
        assert_no_leakage(["alpha one"], ["beta two"], train=["ALPHA ONE"])


def test_clean_sets_pass():
    assert assert_no_leakage(["alpha one"], ["beta two"], train=["gamma three"]) is None
    assert assert_no_leakage(["alpha one"], ["beta two"]) is None


def test_leakage_error_is_a_value_error():
    assert issubclass(LeakageError, ValueError)


# --- norm_text hardening for Arabic and Hindi leakage detection -------------------------------


def test_norm_text_strips_format_characters():
    # ZWJ, ZWNJ, ZWSP, BOM, soft hyphen and the Arabic letter mark are all category Cf
    assert norm_text("a‍b‌c​d﻿e­f؜g") == "abcdefg"


def test_norm_text_strips_arabic_tashkeel_and_tatweel():
    vocalised = "كَيْفَ أَكْتُبُ رِسَالَةً"  # fatha, sukun, damma, kasra, tanwin
    assert norm_text(vocalised) == norm_text("كيف أكتب رسالة")
    assert norm_text("كــيــف") == norm_text("كيف")  # tatweel U+0640
    assert norm_text("إِلَىٰ") == norm_text("إلى")  # kasra and superscript alef U+0670
    # a hamza-on-alef is a letter, not a mark, so it is kept
    assert norm_text("أكتب") != norm_text("اكتب")


def test_norm_text_maps_arabic_question_mark_and_devanagari_danda():
    assert norm_text("ما هذا؟") == norm_text("ما هذا?") == "ما هذا?"
    assert norm_text("यह क्या है।") == norm_text("यह क्या है.") == "यह क्या है."
    assert norm_text("हाँ॥") == norm_text("हाँ.")  # double danda U+0965


def test_norm_text_keeps_nfkc_casefold_whitespace():
    assert norm_text("  ＡＢＣ \t Straße\n") == "abc strasse"


def test_leak_detected_for_vocalised_arabic_variant():
    dev = ["كيف أصنع كعكة؟"]
    report = ["كَيْفَ أَصْنَعُ كَعْكَةً؟"]  # vocalised (fatha, sukun, damma, tanwin)
    with pytest.raises(LeakageError):
        assert_no_leakage(dev, report)


def test_leak_detected_for_zwj_zwnj_variant():
    dev = ["मुझे यह क्यों करना चाहिए"]
    report = ["मुझे यह क्यों\u200d करना\u200c चाहिए"]
    with pytest.raises(LeakageError):
        assert_no_leakage(dev, report)
    # the same invisible-character variant is caught against train, too
    with pytest.raises(LeakageError):
        assert_no_leakage(["एक दो"], ["तीन चार"], train=["तीन\u200c चार"])


def test_leak_detected_for_danda_vs_period_variant():
    with pytest.raises(LeakageError):
        assert_no_leakage(["यह एक परीक्षण है।"], ["यह एक परीक्षण है."])
    with pytest.raises(LeakageError):
        assert_no_leakage(["क्या यह सही है॥"], ["क्या यह सही है."])
