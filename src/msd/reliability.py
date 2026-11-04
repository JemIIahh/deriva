"""Judge reliability: which items a second judge re-labels, and how well the two agree.

`sample_items` picks the items; the caller has them re-judged by a different judge
and feeds both label sets to `agreement`. Languages whose agreement is too low (or
could not be measured) come back from `low_confidence_langs` and are flagged in the
figures rather than hidden.
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from msd.config import KAPPA_LOW, RELIABILITY_N
from msd.schema import Verdict
from msd.stats import MIN_N, kappa_ci

# Only these labels say what the model did; the others carry no judgement to compare.
_JUDGED = frozenset({"complied", "refused"})

Item = tuple[str, str, float | None]  # (prompt id, arm, alpha)

_NOT_ASSESSED = (math.nan, math.nan, math.nan)


def _item_key(item: Item) -> tuple:
    prompt_id, arm, alpha = item
    return (prompt_id, arm, alpha is not None, alpha or 0.0)


def _allocate(capacity: Mapping[str, int], total: int) -> dict[str, int]:
    """Split `total` as evenly as possible over arms, none above its capacity.

    Arms take turns in alphabetical order, so the remainder always goes to the
    same arms. An arm that runs out hands its unused share to the others.
    """
    taken = {arm: 0 for arm in capacity}
    remaining = min(total, sum(capacity.values()))
    while remaining > 0:
        open_arms = sorted(arm for arm in capacity if taken[arm] < capacity[arm])
        share, extra = divmod(remaining, len(open_arms))
        for i, arm in enumerate(open_arms):
            give = min(share + (1 if i < extra else 0), capacity[arm] - taken[arm])
            taken[arm] += give
            remaining -= give
    return taken


def sample_items(
    verdicts: Sequence[Verdict],
    n_per_lang: int = RELIABILITY_N,
    seed: int = 0,
) -> list[Item]:
    """Pick up to `n_per_lang` (prompt id, arm, alpha) items per language to re-judge.

    Only `complied` and `refused` verdicts are eligible. Within a language the items
    are spread evenly across arms (all alphas of an arm share its quota), chosen by a
    seeded shuffle; an arm with too few items hands its share to the others, and a
    language with fewer eligible items than asked gives them all. The result is sorted
    by (prompt id, arm, alpha), so it depends only on the data and the seed.
    """
    by_lang_arm: dict[str, dict[str, set[Item]]] = defaultdict(lambda: defaultdict(set))
    for v in verdicts:
        if v.label in _JUDGED:
            by_lang_arm[v.lang][v.arm].add((v.prompt_id, v.arm, v.alpha))

    chosen: list[Item] = []
    for lang, arms in by_lang_arm.items():
        quota = _allocate({arm: len(items) for arm, items in arms.items()}, n_per_lang)
        for arm, items in arms.items():
            pool = sorted(items, key=_item_key)
            random.Random(f"{seed}|{lang}|{arm}").shuffle(pool)
            chosen.extend(pool[: quota[arm]])
    return sorted(chosen, key=_item_key)


def _single_judge(verdicts: Sequence[Verdict], which: str) -> str | None:
    judges = sorted({v.judge_id for v in verdicts})
    if len(judges) > 1:
        raise ValueError(f"list {which} comes from more than one judge: {judges}")
    return judges[0] if judges else None


def _by_key(verdicts: Sequence[Verdict], which: str) -> dict[Item, Verdict]:
    keyed: dict[Item, Verdict] = {}
    for v in verdicts:
        key = (v.prompt_id, v.arm, v.alpha)
        if key in keyed:
            raise ValueError(f"duplicate verdict in list {which} would be counted twice: {key}")
        keyed[key] = v
    return keyed


def agreement(a: Sequence[Verdict], b: Sequence[Verdict]) -> dict[str, tuple[float, float, float]]:
    """Cohen's kappa between two judges per language: (kappa, lo, hi).

    The lists are joined on (prompt id, arm, alpha); a pair where either judge
    gave something other than `complied` or `refused` is dropped. A language with
    fewer than `MIN_N` usable pairs gets (nan, nan, nan): too few to say anything.
    Every language that appears in either list is present in the result, so one
    that was never double-judged shows up as not assessed instead of vanishing.

    Raises `ValueError` if both lists come from the same judge (a judge agrees
    with itself by construction), if one list mixes judges, or if a list repeats
    a key.
    """
    judge_a, judge_b = _single_judge(a, "a"), _single_judge(b, "b")
    if judge_a is not None and judge_a == judge_b:
        raise ValueError(f"both lists come from the same judge {judge_a!r}; agreement would be meaningless")
    keyed_a, keyed_b = _by_key(a, "a"), _by_key(b, "b")

    labels: dict[str, tuple[list[str], list[str]]] = {
        v.lang: ([], []) for v in [*a, *b]
    }
    for key in sorted(keyed_a.keys() & keyed_b.keys(), key=_item_key):
        va, vb = keyed_a[key], keyed_b[key]
        if va.label in _JUDGED and vb.label in _JUDGED:
            labels[va.lang][0].append(va.label)
            labels[va.lang][1].append(vb.label)

    return {
        lang: kappa_ci(la, lb) if len(la) >= MIN_N else _NOT_ASSESSED
        for lang, (la, lb) in sorted(labels.items())
    }


def low_confidence_langs(
    agr: Mapping[str, tuple[float, float, float]],
    threshold: float = KAPPA_LOW,
) -> set[str]:
    """Languages whose judge agreement is below `threshold`, or was not assessed (nan)."""
    return {lang for lang, (kappa, _, _) in agr.items() if math.isnan(kappa) or kappa < threshold}
