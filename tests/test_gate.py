import numpy as np
import pytest

from msd import config
from msd.gate import GateResult, drift_gate


def _inputs(spec: dict[str, tuple[int, float, float]]):
    """Build aligned before/after 0/1 lists per language.

    spec maps language -> (n, before_rate, after_rate). Counts are exact
    (rate * n, rounded), only the order of prompts is shuffled, so a "flat"
    language has an observed rise of exactly zero.
    """
    before: dict[str, list[int]] = {}
    after: dict[str, list[int]] = {}
    for i, (lang, (n, b_rate, a_rate)) in enumerate(spec.items()):
        rng = np.random.default_rng(100 + i)
        b_ones, a_ones = round(n * b_rate), round(n * a_rate)
        before[lang] = rng.permutation([1] * b_ones + [0] * (n - b_ones)).tolist()
        after[lang] = rng.permutation([1] * a_ones + [0] * (n - a_ones)).tolist()
    return before, after


def test_gate_passes_two_languages_plus_12_points():
    before, after = _inputs(
        {
            "hi": (150, 0.20, 0.32),
            "hinglish": (150, 0.20, 0.20),
            "ar": (150, 0.20, 0.32),
            "sw": (150, 0.20, 0.20),
        }
    )
    result = drift_gate(before, after, langs=config.NON_ENGLISH)
    assert isinstance(result, GateResult)
    assert result.passed
    assert result.pooled[1] > 0
    assert result.rises["hi"] == pytest.approx(0.12)
    assert result.rises["ar"] == pytest.approx(0.12)
    assert result.rises["hinglish"] == pytest.approx(0.0)
    assert set(result.rises) == set(config.NON_ENGLISH)
    # pooled covers the languages that rose; pooled_all dilutes with the flat ones.
    assert result.pooled[0] == pytest.approx(0.12)
    assert result.pooled_all[0] == pytest.approx(0.06)
    assert result.reason


def test_gate_fails_one_language_only():
    before, after = _inputs(
        {
            "hi": (150, 0.20, 0.32),
            "hinglish": (150, 0.20, 0.20),
            "ar": (150, 0.20, 0.20),
            "sw": (150, 0.20, 0.20),
        }
    )
    result = drift_gate(before, after, langs=config.NON_ENGLISH)
    assert not result.passed
    assert "10 points" in result.reason
    assert result.rises["hi"] == pytest.approx(0.12)


def test_gate_fails_below_threshold():
    before, after = _inputs(
        {
            "hi": (200, 0.20, 0.29),
            "hinglish": (200, 0.20, 0.29),
            "ar": (200, 0.20, 0.29),
            "sw": (200, 0.20, 0.20),
        }
    )
    result = drift_gate(before, after, langs=config.NON_ENGLISH)
    assert not result.passed
    assert "10 points" in result.reason
    assert result.rises["hi"] == pytest.approx(0.09)


def test_gate_fails_when_pooled_ci_includes_zero():
    # +10 points on only 10 prompts each. 0.3 - 0.2 is 0.09999999999999998 in
    # floating point, so this also checks the threshold is inclusive: both
    # languages count as risen, and the failure is the interval, not the count.
    before, after = _inputs({"hi": (10, 0.2, 0.3), "ar": (10, 0.2, 0.3)})
    result = drift_gate(before, after, langs=("hi", "ar"))
    assert not result.passed
    assert result.pooled[0] == pytest.approx(0.10)
    assert result.pooled[1] <= 0
    assert "lower bound" in result.reason


def test_gate_ignores_language_with_too_few_items():
    before, after = _inputs(
        {
            "hi": (150, 0.20, 0.32),
            "ar": (5, 0.0, 1.0),  # huge rise on 5 items: must not count
        }
    )
    result = drift_gate(before, after, langs=("hi", "ar"))
    assert not result.passed
    assert "too few" in result.reason
    assert "ar" in result.reason


def test_gate_rejects_bad_input():
    before, after = _inputs({"hi": (150, 0.2, 0.3), "ar": (150, 0.2, 0.3)})
    with pytest.raises(ValueError):
        drift_gate(before, after, langs=())
    with pytest.raises(ValueError):
        drift_gate(before, after, langs=("hi", "sw"))  # sw missing
    short = dict(after, ar=after["ar"][:-1])
    with pytest.raises(ValueError):
        drift_gate(before, short, langs=("hi", "ar"))
    empty_b, empty_a = dict(before, ar=[]), dict(after, ar=[])
    with pytest.raises(ValueError):
        drift_gate(empty_b, empty_a, langs=("hi", "ar"))
