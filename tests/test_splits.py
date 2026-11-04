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
