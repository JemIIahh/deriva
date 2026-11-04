import numpy as np
import pytest

from msd.stats import (
    InsufficientData,
    bootstrap_ci,
    cohen_kappa,
    kappa_ci,
    paired_diff_ci,
    pooled_paired_diff_ci,
)


def _flags(n: int, ones: int, seed: int) -> list[int]:
    """Exactly `ones` ones among n items, in a seeded random order."""
    rng = np.random.default_rng(seed)
    return rng.permutation([1] * ones + [0] * (n - ones)).tolist()


# --- kappa -------------------------------------------------------------


def test_kappa_perfect_is_one():
    labels = ["safe", "unsafe", "refuse", "safe", "unsafe", "refuse", "safe"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_known_value():
    # po = 3/4; pe = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = 0.25 / 0.5 = 0.5
    a = ["y", "y", "n", "n"]
    b = ["y", "n", "n", "n"]
    assert cohen_kappa(a, b) == pytest.approx(0.5)


def test_kappa_independent_near_zero():
    rng = np.random.default_rng(0)
    a = rng.choice(["safe", "unsafe"], size=2000).tolist()
    b = rng.choice(["safe", "unsafe"], size=2000).tolist()
    assert abs(cohen_kappa(a, b)) < 0.1


def test_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa(["a", "b"], ["a"])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_kappa_ci_brackets_point():
    # 48 of 60 agree, two labels.
    a = ["y"] * 30 + ["n"] * 30
    b = ["y"] * 24 + ["n"] * 6 + ["n"] * 24 + ["y"] * 6
    point, lo, hi = kappa_ci(a, b, n_boot=500, seed=3)
    assert point == pytest.approx(cohen_kappa(a, b))
    assert lo < point < hi
    assert -1.0 <= lo and hi <= 1.0


# --- bootstrap_ci --------------------------------------------------------


def test_bootstrap_ci_contains_point_and_narrows_with_n():
    small = _flags(50, 15, seed=1)
    large = _flags(500, 150, seed=2)
    p_small, lo_small, hi_small = bootstrap_ci(small)
    p_large, lo_large, hi_large = bootstrap_ci(large)
    assert p_small == pytest.approx(0.3)
    assert p_large == pytest.approx(0.3)
    assert lo_small <= p_small <= hi_small
    assert lo_large <= p_large <= hi_large
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_bootstrap_ci_is_reproducible_for_a_seed():
    flags = _flags(80, 20, seed=4)
    assert bootstrap_ci(flags, seed=7) == bootstrap_ci(flags, seed=7)


def test_small_n_raises_insufficient_data():
    with pytest.raises(InsufficientData):
        bootstrap_ci([0, 1, 0, 1, 0])
    assert issubclass(InsufficientData, ValueError)
    # An explicit lower floor is honoured.
    point, lo, hi = bootstrap_ci([0, 1, 0, 1, 0], min_n=5)
    assert lo <= point <= hi


def test_bootstrap_ci_rejects_non_binary_input():
    with pytest.raises(ValueError):
        bootstrap_ci([0.5] * 20)


# --- paired differences ---------------------------------------------------


def test_paired_diff_detects_known_shift():
    before = _flags(300, 60, seed=10)  # 20% unsafe
    after = _flags(300, 120, seed=11)  # 40% unsafe
    point, lo, hi = paired_diff_ci(before, after)
    assert point == pytest.approx(0.2)
    assert lo > 0
    assert lo < point < hi


def test_paired_diff_no_shift_straddles_zero():
    before = _flags(300, 90, seed=12)
    after = _flags(300, 90, seed=13)
    point, lo, hi = paired_diff_ci(before, after)
    assert point == pytest.approx(0.0)
    assert lo < 0 < hi


def test_paired_diff_rejects_misaligned_input():
    with pytest.raises(ValueError):
        paired_diff_ci([0, 1] * 10, [0, 1] * 9)


def test_pooled_paired_diff_weights_by_items():
    # hi: 20 -> 40 of 100 (+0.2); ar: flat at 30 of 300. Pooled: +20 / 400.
    pairs = {
        "hi": (_flags(100, 20, seed=20), _flags(100, 40, seed=21)),
        "ar": (_flags(300, 90, seed=22), _flags(300, 90, seed=23)),
    }
    point, lo, hi = pooled_paired_diff_ci(pairs)
    assert point == pytest.approx(0.05)
    assert lo < point < hi


def test_pooled_paired_diff_detects_shift_across_languages():
    pairs = {
        "hi": (_flags(150, 30, seed=30), _flags(150, 48, seed=31)),
        "ar": (_flags(150, 30, seed=32), _flags(150, 48, seed=33)),
    }
    point, lo, _ = pooled_paired_diff_ci(pairs)
    assert point == pytest.approx(0.12)
    assert lo > 0


def test_pooled_paired_diff_rejects_bad_input():
    with pytest.raises(ValueError):
        pooled_paired_diff_ci({})
    with pytest.raises(ValueError):
        pooled_paired_diff_ci({"hi": ([0, 1] * 10, [0, 1] * 9)})
    with pytest.raises(InsufficientData):
        pooled_paired_diff_ci({"hi": ([0, 1, 0], [1, 1, 0])})
