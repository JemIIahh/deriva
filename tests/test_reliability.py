import dataclasses
import math
import random
from collections import Counter

import pytest

from msd.reliability import agreement, low_confidence_langs, sample_items
from msd.schema import Verdict


def _v(
    i: int,
    label: str = "complied",
    *,
    lang: str = "en",
    arm: str = "A0",
    alpha: float | None = None,
    judge: str = "j1",
    kind: str = "harmful",
) -> Verdict:
    return Verdict(
        prompt_id=f"{kind}-{lang}-{i}",
        arm=arm,
        alpha=alpha,
        lang=lang,
        kind=kind,
        judge_id=judge,
        label=label,
        excerpt="",
    )


def _pool(langs=("en", "hi"), arms=("A0", "A1", "A2"), per_arm=40) -> list[Verdict]:
    return [
        _v(i, "complied" if i % 2 else "refused", lang=lang, arm=arm)
        for lang in langs
        for arm in arms
        for i in range(per_arm)
    ]


def _per_lang_arm(items, verdicts) -> Counter:
    lang_of = {v.prompt_id: v.lang for v in verdicts}
    return Counter((lang_of[pid], arm) for pid, arm, _ in items)


# ---------------------------------------------------------------- sample_items


def test_sample_items_is_n_per_language_and_arm_balanced():
    verdicts = _pool()
    items = sample_items(verdicts, n_per_lang=30, seed=0)
    counts = _per_lang_arm(items, verdicts)
    assert len(items) == 60
    assert dict(counts) == {
        (lang, arm): 10 for lang in ("en", "hi") for arm in ("A0", "A1", "A2")
    }


def test_sample_items_only_takes_complied_or_refused():
    verdicts = _pool(langs=("en",), arms=("A0", "A1"), per_arm=20)
    junk = [
        _v(100 + i, label, arm=arm)
        for i, (label, arm) in enumerate(
            [("no_response", "A0"), ("judge_error", "A0"), ("no_response", "A1")] * 5
        )
    ]
    items = sample_items(verdicts + junk, n_per_lang=40, seed=0)
    chosen = set(items)
    eligible = {(v.prompt_id, v.arm, v.alpha) for v in verdicts}
    assert chosen == eligible  # asked for more than exist: everything eligible, nothing else


def test_sample_items_remainder_is_deterministic_and_total_is_exact():
    verdicts = _pool(langs=("en",), per_arm=40)
    first = sample_items(verdicts, n_per_lang=10, seed=3)
    again = sample_items(list(reversed(verdicts)), n_per_lang=10, seed=3)
    assert first == again  # input order does not matter
    assert len(first) == 10
    sizes = sorted(Counter(arm for _, arm, _ in first).values())
    assert sizes == [3, 3, 4]


def test_sample_items_short_arm_gives_its_quota_to_the_others():
    pool = [_v(i, arm="A0") for i in range(2)]
    pool += [_v(i, arm="A1") for i in range(50)]
    pool += [_v(i, arm="A2") for i in range(50)]
    items = sample_items(pool, n_per_lang=9, seed=0)
    by_arm = Counter(arm for _, arm, _ in items)
    assert by_arm["A0"] == 2
    assert sum(by_arm.values()) == 9


def test_sample_items_keeps_alphas_of_one_arm_distinct():
    pool = [_v(i, arm="A2", alpha=a) for a in (0.25, 0.5, 1.0) for i in range(10)]
    items = sample_items(pool, n_per_lang=30, seed=0)
    assert len(items) == 30
    assert {alpha for _, _, alpha in items} == {0.25, 0.5, 1.0}


def test_sample_items_is_seeded_and_sorted():
    verdicts = _pool()
    a = sample_items(verdicts, n_per_lang=12, seed=1)
    b = sample_items(verdicts, n_per_lang=12, seed=1)
    c = sample_items(verdicts, n_per_lang=12, seed=2)
    assert a == b
    assert a != c
    assert a == sorted(a, key=lambda t: (t[0], t[1], t[2] is not None, t[2] or 0.0))


# ------------------------------------------------------------------- agreement


def _judged(judge: str, labels: dict[tuple, str], *, lang: str = "en") -> list[Verdict]:
    return [
        _v(i, label, lang=lang, arm=arm, alpha=alpha, judge=judge)
        for (i, arm, alpha), label in labels.items()
    ]


def test_agreement_joins_on_prompt_arm_alpha():
    # The same prompt appears under two alphas with opposite labels. A join on prompt
    # id alone (or on prompt id and arm) would pair the wrong rows and lower kappa.
    keys = [(i, "A2", a) for i in range(10) for a in (0.5, 1.0)]
    labels = {k: ("complied" if k[2] == 0.5 else "refused") for k in keys}
    a = _judged("j1", labels)
    b = _judged("j2", labels)
    random.Random(0).shuffle(b)
    agr = agreement(a, b)
    assert set(agr) == {"en"}
    kappa, lo, hi = agr["en"]
    assert kappa == pytest.approx(1.0)
    assert lo <= kappa <= hi


def test_agreement_is_per_language_and_signed():
    same = {(i, "A0", None): ("complied" if i % 2 else "refused") for i in range(20)}
    flip = {k: ("refused" if v == "complied" else "complied") for k, v in same.items()}
    a = _judged("j1", same, lang="en") + _judged("j1", same, lang="hi")
    b = _judged("j2", same, lang="en") + _judged("j2", flip, lang="hi")
    agr = agreement(a, b)
    assert agr["en"][0] == pytest.approx(1.0)
    assert agr["hi"][0] == pytest.approx(-1.0)


def test_agreement_drops_unlabelled_and_unmatched_pairs():
    base = {(i, "A0", None): ("complied" if i % 2 else "refused") for i in range(12)}
    a = _judged("j1", base)
    b = _judged("j2", base)
    # Extra rows that must not count: a no_response on one side, a row only one judge saw.
    a.append(_v(50, "complied", judge="j1"))
    b.append(_v(50, "no_response", judge="j2"))
    a.append(_v(51, "complied", judge="j1"))
    agr = agreement(a, b)
    assert agr["en"][0] == pytest.approx(1.0)


def test_agreement_with_under_ten_pairs_is_not_assessed():
    labels = {(i, "A0", None): ("complied" if i % 2 else "refused") for i in range(9)}
    agr = agreement(_judged("j1", labels), _judged("j2", labels))
    assert all(math.isnan(x) for x in agr["en"])


def test_agreement_language_with_no_usable_pair_is_present_and_not_assessed():
    # Only one judge labelled "sw": the language must show up as nan, not vanish,
    # otherwise low_confidence_langs would never flag it.
    labels = {(i, "A0", None): ("complied" if i % 2 else "refused") for i in range(12)}
    a = _judged("j1", labels, lang="en") + _judged("j1", labels, lang="sw")
    b = _judged("j2", labels, lang="en")
    agr = agreement(a, b)
    assert all(math.isnan(x) for x in agr["sw"])
    assert low_confidence_langs(agr) == {"sw"}


def test_agreement_rejects_a_judge_compared_with_itself():
    labels = {(i, "A0", None): "complied" for i in range(12)}
    with pytest.raises(ValueError, match="same judge"):
        agreement(_judged("j1", labels), _judged("j1", labels))


def test_agreement_rejects_a_list_with_two_judges():
    labels = {(i, "A0", None): "complied" for i in range(12)}
    mixed = _judged("j1", labels)[:6] + _judged("j3", labels)[6:]
    with pytest.raises(ValueError, match="more than one judge"):
        agreement(mixed, _judged("j2", labels))
    with pytest.raises(ValueError, match="more than one judge"):
        agreement(_judged("j2", labels), mixed)


def test_agreement_rejects_duplicate_keys():
    labels = {(i, "A0", None): "complied" for i in range(12)}
    a = _judged("j1", labels)
    with pytest.raises(ValueError, match="duplicate"):
        agreement(a + a[:1], _judged("j2", labels))


def _sixty(label_of) -> dict[tuple, str]:
    return {(i, "A0", None): label_of(i) for i in range(60)}


def test_agreement_both_judges_all_refused_is_not_assessed():
    # One label throughout is not agreement, it is an untested judge: kappa is undefined.
    labels = _sixty(lambda i: "refused")
    agr = agreement(_judged("j1", labels), _judged("j2", labels))
    assert all(math.isnan(x) for x in agr["en"])
    assert low_confidence_langs(agr) == {"en"}


def test_agreement_both_judges_all_complied_is_not_assessed():
    labels = _sixty(lambda i: "complied")
    agr = agreement(_judged("j1", labels), _judged("j2", labels))
    assert all(math.isnan(x) for x in agr["en"])
    assert low_confidence_langs(agr) == {"en"}


def test_agreement_one_dissent_in_sixty_still_gives_a_number_and_is_flagged():
    a = _sixty(lambda i: "refused")
    b = _sixty(lambda i: "complied" if i == 0 else "refused")
    agr = agreement(_judged("j1", a), _judged("j2", b))
    assert agr["en"][0] == pytest.approx(0.0)
    assert not any(math.isnan(x) for x in agr["en"])
    assert low_confidence_langs(agr) == {"en"}  # kappa 0.0 is below the threshold


def test_agreement_degenerate_after_dropping_unusable_pairs_is_not_assessed():
    # Both judges say "refused" on every usable pair; the only "complied" sits in a pair
    # that is dropped (the other judge gave no_response), so it must not count as variety.
    a = _judged("j1", _sixty(lambda i: "refused"))
    b = _judged("j2", _sixty(lambda i: "refused"))
    a[0] = _v(0, "complied", judge="j1")
    b[0] = _v(0, "no_response", judge="j2")
    agr = agreement(a, b)
    assert all(math.isnan(x) for x in agr["en"])


def test_agreement_normal_mixed_sample_is_unchanged():
    labels = _sixty(lambda i: "complied" if i % 2 else "refused")
    agr = agreement(_judged("j1", labels), _judged("j2", labels))
    kappa, lo, hi = agr["en"]
    assert kappa == pytest.approx(1.0)
    assert lo <= kappa <= hi
    assert low_confidence_langs(agr) == set()


def test_agreement_rejects_a_key_whose_two_verdicts_disagree_on_language():
    labels = {(i, "A0", None): ("complied" if i % 2 else "refused") for i in range(12)}
    a = _judged("j1", labels, lang="en")
    b = _judged("j2", labels, lang="en")
    # Same (prompt id, arm, alpha), but the second judge's row says another language.
    b[3] = dataclasses.replace(b[3], lang="hi")
    with pytest.raises(ValueError, match="lang"):
        agreement(a, b)


# --------------------------------------------------------- low_confidence_langs


def test_low_confidence_below_0_6():
    agr = {
        "en": (0.9, 0.8, 0.95),
        "hi": (0.59, 0.4, 0.75),
        "ar": (0.6, 0.45, 0.8),  # exactly at the threshold is not below it
        "sw": (float("nan"),) * 3,  # not assessed counts as low confidence
    }
    assert low_confidence_langs(agr) == {"hi", "sw"}
    assert low_confidence_langs(agr, threshold=0.95) == {"en", "hi", "ar", "sw"}
    assert low_confidence_langs({}) == set()


# ------------------------------------------------------------ usable_pair_counts (Task 17)


def test_usable_pair_counts_counts_what_agreement_would_use():
    from msd.reliability import usable_pair_counts

    base = {(i, "A0", None): ("complied" if i % 2 else "refused") for i in range(12)}
    a = _judged("j1", base) + _judged("j1", base, lang="hi")
    b = _judged("j2", base) + _judged("j2", base, lang="hi")
    a.append(_v(50, "complied", judge="j1"))
    b.append(_v(50, "no_response", judge="j2"))  # unlabelled on one side: not a usable pair
    a.append(_v(51, "complied", judge="j1"))  # seen by one judge only
    a.append(_v(60, "complied", lang="sw", judge="j1"))  # a language the second judge never saw
    assert usable_pair_counts(a, b) == {"en": 12, "hi": 12, "sw": 0}


def test_usable_pair_counts_rejects_what_agreement_rejects():
    from msd.reliability import usable_pair_counts

    one = _judged("j1", {(i, "A0", None): "complied" for i in range(12)})
    with pytest.raises(ValueError, match="same judge"):
        usable_pair_counts(one, one)
