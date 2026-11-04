"""Bootstrap confidence intervals and Cohen's kappa, numpy only.

Every interval is a percentile bootstrap drawn from `numpy.random.default_rng(seed)`,
so a given (data, seed) always returns the same numbers.
"""

from collections.abc import Mapping, Sequence

import numpy as np

# Below this many items a bootstrap interval says almost nothing.
MIN_N = 10


class InsufficientData(ValueError):
    """Too few items to put a meaningful interval on a rate."""


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")


def _percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _as_flags(values: Sequence[int]) -> np.ndarray:
    """0/1 sequence -> int array; rejects empty input and non-binary values."""
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("expected a non-empty one-dimensional 0/1 sequence")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("flags must be 0 or 1")
    return arr.astype(np.int64)


def _as_paired(before: Sequence[int], after: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    b, a = _as_flags(before), _as_flags(after)
    if b.shape != a.shape:
        raise ValueError(
            f"before and after must be aligned by prompt: {b.size} vs {a.size} items"
        )
    return b, a


def bootstrap_ci(
    flags: Sequence[int],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    min_n: int = MIN_N,
) -> tuple[float, float, float]:
    """Rate of 1s with a percentile bootstrap interval: (point, lo, hi)."""
    _check_level(level)
    x = _as_flags(flags)
    if x.size < min_n:
        raise InsufficientData(f"need at least {min_n} items, got {x.size}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    lo, hi = _percentile_ci(x[idx].mean(axis=1), level)
    return float(x.mean()), lo, hi


def _encode(a: Sequence[str], b: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("cannot compute kappa on empty input")
    labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    return (
        np.array([index[x] for x in a]),
        np.array([index[x] for x in b]),
        len(labels),
    )


def _kappa(ca: np.ndarray, cb: np.ndarray, k: int) -> float:
    n = ca.size
    p_observed = float(np.mean(ca == cb))
    p_chance = float(
        (np.bincount(ca, minlength=k) / n) @ (np.bincount(cb, minlength=k) / n)
    )
    if p_chance >= 1.0:
        # Both raters used one and the same label throughout: perfect agreement.
        return 1.0
    return (p_observed - p_chance) / (1.0 - p_chance)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two raters' labels."""
    return _kappa(*_encode(a, b))


def kappa_ci(
    a: Sequence[str],
    b: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Cohen's kappa with a percentile bootstrap over items: (point, lo, hi)."""
    _check_level(level)
    ca, cb, k = _encode(a, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ca.size, size=(n_boot, ca.size))
    samples = np.array([_kappa(ca[i], cb[i], k) for i in idx])
    lo, hi = _percentile_ci(samples, level)
    return _kappa(ca, cb, k), lo, hi


def paired_diff_ci(
    before: Sequence[int],
    after: Sequence[int],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    min_n: int = MIN_N,
) -> tuple[float, float, float]:
    """Rise in rate, mean(after) - mean(before), for the same prompts.

    The two 0/1 lists are aligned by prompt. Prompt indices are resampled
    jointly, so the pairing is preserved inside every bootstrap draw.
    """
    _check_level(level)
    b, a = _as_paired(before, after)
    if b.size < min_n:
        raise InsufficientData(f"need at least {min_n} items, got {b.size}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.size, size=(n_boot, b.size))
    lo, hi = _percentile_ci((a[idx] - b[idx]).mean(axis=1), level)
    return float(a.mean() - b.mean()), lo, hi


def pooled_paired_diff_ci(
    pairs: Mapping[str, tuple[Sequence[int], Sequence[int]]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    min_n: int = MIN_N,
) -> tuple[float, float, float]:
    """Pooled rise across languages: stratified paired bootstrap.

    Each draw resamples prompts within every language (keeping the before/after
    pairing), pools all resampled items, and takes the difference of the pooled
    rates. Languages are therefore weighted by their number of items.
    """
    _check_level(level)
    if not pairs:
        raise ValueError("need at least one language to pool")
    rng = np.random.default_rng(seed)
    total = 0
    point_sum = 0
    boot_sum = np.zeros(n_boot)
    for lang in sorted(pairs):  # sorted: the result must not depend on dict order
        b, a = _as_paired(*pairs[lang])
        if b.size < min_n:
            raise InsufficientData(f"{lang}: need at least {min_n} items, got {b.size}")
        idx = rng.integers(0, b.size, size=(n_boot, b.size))
        boot_sum += (a[idx] - b[idx]).sum(axis=1)
        point_sum += int(a.sum() - b.sum())
        total += b.size
    lo, hi = _percentile_ci(boot_sum / total, level)
    return point_sum / total, lo, hi
