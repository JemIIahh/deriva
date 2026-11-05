"""The fine-tuning data preparation library (src/msd/ftprep.py): closed-form prompts from
Global-MMLU rows, a global item-level train/heldout split, and the join back from CuratorKIT's
export. Pure Python on tiny fixtures: no network, no CuratorKIT.
"""

from __future__ import annotations

import json

import pytest

from msd import ftdata, ftprep
from msd.config import LANGS

ALLOWED = {"high_school_geography", "nutrition"}

QUESTIONS = {
    "en": "Which river flows through the capital of country number {i}?",
    "hi": "देश संख्या {i} की राजधानी से कौन सी नदी बहती है?",
    "ar": "أي نهر يمر عبر عاصمة الدولة رقم {i}؟",
    "sw": "Mto upi unapita katika mji mkuu wa nchi namba {i}?",
}


def raw_row(lang, i, subject="high_school_geography", answer="B", **over):
    row = {
        "sample_id": f"{subject}/test/{i}",
        "subject": subject,
        "question": QUESTIONS[lang].format(i=i),
        "option_a": "10",
        "option_b": "20",
        "option_c": "30",
        "option_d": "40",
        "answer": answer,
    }
    row.update(over)
    return row


def write_raw(root, lang, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{lang}_test.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def make_raw(root, langs=("en", "hi", "ar", "sw"), n=40, other=0):
    for lang in langs:
        rows = [raw_row(lang, i) for i in range(n)]
        rows += [raw_row(lang, 500 + i, subject="virology") for i in range(other)]  # not on the allowlist
        write_raw(root, lang, rows)


# --- the closed-form prompt ---------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "hi", "ar", "sw"])
def test_user_turn_is_question_options_then_the_one_line_instruction(lang):
    turn = ftprep.build_user_turn(lang, "Q?", ["one", "two", "three", "four"])
    lines = turn.split("\n")
    assert lines[0] == "Q?"
    assert lines[1:5] == ["A. one", "B. two", "C. three", "D. four"]
    assert lines[5] == ftprep.INSTRUCTION_LINES[lang]
    assert len(lines) == 6
    assert ftdata.option_letters(turn) == ["A", "B", "C", "D"]
    assert ftdata.question_part(turn) == "Q?"


def test_instruction_lines_exist_for_the_four_source_languages_only():
    assert set(ftprep.INSTRUCTION_LINES) == {"en", "hi", "ar", "sw"}
    assert "hinglish" not in ftprep.INSTRUCTION_LINES  # deferred: comes from translating the hi rows


# --- one raw row -> one candidate ---------------------------------------------------------------


def test_convert_row_builds_a_candidate_with_the_bare_letter():
    cand, why = ftprep.convert_row("hi", raw_row("hi", 3), ALLOWED)
    assert why is None
    assert cand["id"] == "hi:high_school_geography/test/3"
    assert cand["lang"] == "hi"
    assert cand["output"] == cand["gold"] == "B"
    assert cand["item"] == "high_school_geography/test/3"
    assert cand["subject"] == "high_school_geography"
    assert cand["instruction"].startswith(QUESTIONS["hi"].format(i=3))
    assert cand["instruction"].endswith(ftprep.INSTRUCTION_LINES["hi"])
    assert "input" not in cand


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"subject": "virology", "sample_id": "virology/test/1"}, "subject_not_allowed"),
        ({"answer": "E"}, "bad_answer"),
        ({"answer": ""}, "bad_answer"),
        ({"answer": "b"}, "bad_answer"),
        ({"option_c": ""}, "empty_field"),
        ({"option_d": "   "}, "empty_field"),
        ({"question": " "}, "empty_field"),
        ({"option_a": "20"}, "duplicate_options"),
        ({"question": "Pick one:\nA. this\nB. that"}, "option_line_in_question"),
    ],
)
def test_convert_row_drops_unusable_rows_with_a_reason(over, reason):
    cand, why = ftprep.convert_row("en", raw_row("en", 1, **over), ALLOWED)
    assert cand is None
    assert why == reason


def test_convert_row_drops_a_row_written_in_the_wrong_script():
    # an untranslated English question in the hi file
    cand, why = ftprep.convert_row("hi", raw_row("hi", 1, question="Which river is it?"), ALLOWED)
    assert cand is None and why == "wrong_script"


def test_convert_row_keeps_options_on_one_line():
    cand, why = ftprep.convert_row("en", raw_row("en", 1, option_a="two\nlines  here"), ALLOWED)
    assert why is None
    assert "A. two lines here" in cand["instruction"].split("\n")


# --- splitting by source item --------------------------------------------------------------------


def test_assign_splits_is_deterministic_disjoint_and_sized():
    ids = [f"s/test/{i}" for i in range(100)]
    a = ftprep.assign_splits(ids, seed=1, heldout_pool=10, train_pool=60)
    b = ftprep.assign_splits(list(reversed(ids)), seed=1, heldout_pool=10, train_pool=60)
    assert a == b  # the order the ids arrive in does not matter
    assert sum(v == "heldout" for v in a.values()) == 10
    assert sum(v == "train" for v in a.values()) == 60
    assert len(a) == 70  # the rest are unused
    c = ftprep.assign_splits(ids, seed=2, heldout_pool=10, train_pool=60)
    assert a != c  # the seed matters


def test_assign_splits_small_universe_gives_heldout_first():
    ids = [f"s/test/{i}" for i in range(12)]
    a = ftprep.assign_splits(ids, seed=1, heldout_pool=10, train_pool=60)
    assert sum(v == "heldout" for v in a.values()) == 10
    assert sum(v == "train" for v in a.values()) == 2  # train gets what is left; nothing is padded


def test_item_groups_join_questions_about_the_same_passage():
    passage = "This question refers to the following information.\nSource 1: " + "the council met and voted. " * 5
    rows = [
        raw_row("en", 1, question=f"{passage}\nWhat did it do?"),
        raw_row("en", 2, question=f"{passage}\nWhen did it meet?"),
        raw_row("en", 3, question="A single line question?"),
        raw_row("en", 4, question="Another passage that is long enough to count as one.\n" * 3 + "Why?"),
    ]
    groups = ftprep.item_groups(rows)
    assert groups[rows[0]["sample_id"]] == groups[rows[1]["sample_id"]] == rows[0]["sample_id"]  # named by the smallest id
    assert rows[2]["sample_id"] not in groups and rows[3]["sample_id"] not in groups  # singletons need no group


def test_assign_splits_never_splits_a_group_and_never_overfills_a_pool():
    ids = [f"s/test/{i:03d}" for i in range(60)]
    group_of = {i: "g1" for i in ids[:5]} | {i: "g2" for i in ids[5:9]}
    split = ftprep.assign_splits(ids, seed=4, heldout_pool=10, train_pool=30, group_of=group_of)
    for members in (ids[:5], ids[5:9]):
        assert len({split.get(m) for m in members}) == 1  # all in one split (or all unused)
    assert sum(v == "heldout" for v in split.values()) <= 10
    assert sum(v == "train" for v in split.values()) <= 30
    assert sum(v == "heldout" for v in split.values()) >= 8  # greedy fill gets close to the pool size


def test_prepare_keeps_every_passage_inside_one_split_in_every_language(tmp_path):
    passage = "This question refers to the following information.\nSource 1: " + "the council met and voted. " * 5
    hi_passage = "यह प्रश्न निम्नलिखित जानकारी से संबंधित है।\nस्रोत 1: " + "परिषद की बैठक हुई और मतदान हुआ। " * 5
    for lang, base in (("en", passage), ("hi", hi_passage)):
        rows = [raw_row(lang, i) for i in range(30)]
        for k in range(6):  # six items share one passage, six more share another
            share = base if k < 3 else base + " Two."
            rows[k * 2]["question"] = f"{share}\n" + QUESTIONS[lang].format(i=k * 2)
            rows[k * 2 + 12]["question"] = f"{share}\n" + QUESTIONS[lang].format(i=k * 2 + 12)
        write_raw(tmp_path, lang, rows)
    cands, _ = ftprep.prepare(tmp_path, ("en", "hi"), ALLOWED, seed=9, heldout_pool=8, train_pool=14)
    for lang in ("en", "hi"):
        by_passage = {}
        for c in (c for c in cands if c["lang"] == lang):
            p = ftdata.passage_of(c["instruction"])
            if p:
                by_passage.setdefault(p, set()).add(c["split"])
        assert by_passage and all(len(splits) == 1 for splits in by_passage.values()), lang
    report = ftdata.validate(
        [ftprep.to_record(c) for c in cands if c["split"] == "train"],
        [ftprep.to_record(c) for c in cands if c["split"] == "heldout"],
        expected_langs=("en", "hi"),
    )
    assert not any("passage" in p.message for p in report.problems)


# --- prepare --------------------------------------------------------------------------------------


def test_prepare_builds_a_disjoint_parallel_candidate_set(tmp_path):
    make_raw(tmp_path, n=40, other=5)
    cands, summary = ftprep.prepare(tmp_path, ("en", "hi", "ar", "sw"), ALLOWED, seed=3, heldout_pool=6, train_pool=20)
    by_lang = {lang: [c for c in cands if c["lang"] == lang] for lang in ("en", "hi", "ar", "sw")}
    for lang, rows in by_lang.items():
        assert sum(c["split"] == "heldout" for c in rows) == 6, lang
        assert sum(c["split"] == "train" for c in rows) == 20, lang
    # the same items in every language, and no item in both splits
    items = {split: {c["item"] for c in cands if c["split"] == split} for split in ("train", "heldout")}
    assert not items["train"] & items["heldout"]
    for lang, rows in by_lang.items():
        assert {c["item"] for c in rows if c["split"] == "heldout"} == items["heldout"]
    assert all(c["subject"] in ALLOWED for c in cands)  # virology never reaches the candidates
    assert all(c["stratum"] == f"{c['split']}:{c['lang']}" for c in cands)  # the sampler's category
    assert summary["dropped"]["en"].get("subject_not_allowed") == 5
    assert summary["seed"] == 3


def test_prepare_lists_train_rows_before_heldout_rows_so_dedup_keeps_train(tmp_path):
    make_raw(tmp_path, n=40)
    cands, _ = ftprep.prepare(tmp_path, ("en", "hi"), ALLOWED, seed=3, heldout_pool=6, train_pool=20)
    splits = [c["split"] for c in cands]
    assert splits == sorted(splits, key=lambda s: s != "train")  # all train, then all heldout


def test_prepare_counts_a_language_specific_drop_and_keeps_the_others(tmp_path):
    make_raw(tmp_path, n=40)
    rows = [raw_row("hi", i) for i in range(40)]
    rows[0]["question"] = "An untranslated English question"
    write_raw(tmp_path, "hi", rows)
    cands, summary = ftprep.prepare(tmp_path, ("en", "hi"), ALLOWED, seed=3, heldout_pool=6, train_pool=20)
    assert summary["dropped"]["hi"] == {"wrong_script": 1}
    assert summary["dropped"]["en"] == {}
    en = [c for c in cands if c["lang"] == "en"]
    hi = [c for c in cands if c["lang"] == "hi"]
    assert len(en) == 26 and len(hi) in (25, 26)


def test_prepare_missing_raw_file_names_the_fetch_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_ft_data.py fetch"):
        ftprep.prepare(tmp_path, ("en",), ALLOWED, seed=1, heldout_pool=2, train_pool=2)


def test_prepare_output_validates(tmp_path):
    """Candidates -> records -> the validator: the prep output is what the validator accepts."""
    make_raw(tmp_path, n=60)
    cands, _ = ftprep.prepare(tmp_path, ("en", "hi", "ar", "sw"), ALLOWED, seed=3, heldout_pool=6, train_pool=30)
    train = [ftprep.to_record(c) for c in cands if c["split"] == "train"]
    held = [ftprep.to_record(c) for c in cands if c["split"] == "heldout"]
    report = ftdata.validate(train, held, expected_langs=("en", "hi", "ar", "sw"), allowed_subjects=ALLOWED)
    assert report.ok, report.problems


def test_to_record_has_exactly_the_four_keys():
    cand, _ = ftprep.convert_row("sw", raw_row("sw", 2), ALLOWED)
    rec = ftprep.to_record(cand)
    assert list(rec) == ["id", "lang", "messages", "gold"]
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert rec["messages"][1]["content"] == rec["gold"] == "B"


# --- finalize: the join back from CuratorKIT -----------------------------------------------------


def cand(lang, n, split="train", rank=None):
    c, why = ftprep.convert_row(lang, raw_row(lang, n), ALLOWED)
    assert why is None
    c.update(split=split, rank=n if rank is None else rank)
    return c


def exported(c):
    return {"messages": [{"role": "user", "content": c["instruction"]}, {"role": "assistant", "content": c["output"]}]}


def test_join_exported_returns_the_candidate_behind_every_exported_line():
    cs = [cand("en", i) for i in range(5)]
    survivors = ftprep.join_exported(cs, [exported(cs[3]), exported(cs[1])])
    assert [c["id"] for c in survivors] == [cs[3]["id"], cs[1]["id"]]


def test_join_exported_rejects_a_line_no_candidate_produced():
    cs = [cand("en", i) for i in range(3)]
    alien = exported(cs[0])
    alien["messages"][0]["content"] += " tampered"
    with pytest.raises(ValueError, match="no candidate"):
        ftprep.join_exported(cs, [alien])


def test_join_exported_rejects_the_same_line_twice():
    cs = [cand("en", i) for i in range(3)]
    with pytest.raises(ValueError, match="twice"):
        ftprep.join_exported(cs, [exported(cs[0]), exported(cs[0])])


def test_join_exported_picks_the_first_candidate_when_two_share_text():
    first, second = cand("en", 1), cand("en", 1)
    second["id"] = "en:high_school_geography/test/dup"
    survivors = ftprep.join_exported([first, second], [exported(first)])
    assert survivors == [first]


def test_cap_keeps_the_best_ranked_rows_per_language_and_split():
    survivors = [cand("en", i, rank=i) for i in range(10)] + [cand("en", 50 + i, split="heldout", rank=50 + i) for i in range(6)]
    survivors += [cand("hi", i, rank=i) for i in range(4)]
    kept = ftprep.cap_by_rank(survivors, train_cap=5, heldout_cap=3)
    en_train = [c for c in kept if c["lang"] == "en" and c["split"] == "train"]
    assert [c["rank"] for c in en_train] == [0, 1, 2, 3, 4]
    assert len([c for c in kept if c["split"] == "heldout"]) == 3
    assert len([c for c in kept if c["lang"] == "hi"]) == 4  # fewer than the cap: nothing padded


def test_finalize_writes_the_two_files_and_a_summary(tmp_path):
    cands = [cand("en", i, rank=i) for i in range(8)] + [cand("hi", i, rank=i) for i in range(8)]
    cands += [cand("en", 100 + i, split="heldout", rank=100 + i) for i in range(3)]
    cands += [cand("hi", 100 + i, split="heldout", rank=100 + i) for i in range(3)]
    lost = cands[2]  # CuratorKIT dropped this one (dedup or a gate)
    out = tmp_path / "out"
    summary = ftprep.finalize(
        cands,
        [exported(c) for c in cands if c is not lost],
        out,
        train_cap=5,
        heldout_cap=2,
        rejected=[{"rejecting_step": "SchemaGate", "rejection_reason": "above_max_tokens:900", "id": "x", "metadata": {"lang": "en", "split": "train"}}],
    )
    train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    held = [json.loads(line) for line in (out / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 10 and len(held) == 4
    assert all(list(r) == ["id", "lang", "messages", "gold"] for r in train + held)
    assert lost["id"] not in {r["id"] for r in train}
    assert summary["candidates"]["train"] == {"en": 8, "hi": 8}
    assert summary["curatorkit_survivors"]["train"] == {"en": 7, "hi": 8}
    assert summary["removed_by_curatorkit"]["train"] == {"en": 1}
    assert summary["final"]["train"] == {"en": 5, "hi": 5}
    assert summary["final"]["heldout"] == {"en": 2, "hi": 2}
    assert summary["gate_rejections"] == {"SchemaGate: above_max_tokens": 1}
    report = ftdata.validate(train, held, expected_langs=("en", "hi"))
    assert report.ok, report.problems


# --- fetching (no network: the page getter is injected) -----------------------------------------


def page(rows, total, start=0):
    return {"num_rows_total": total, "rows": [{"row_idx": start + i, "row": r} for i, r in enumerate(rows)]}


def test_fetch_language_reads_every_page_and_keeps_only_allowlisted_subjects_in_order():
    data = [raw_row("en", i, subject="nutrition" if i % 2 else "virology") for i in range(25)]
    calls = []

    def get_page(lang, offset, length):
        calls.append((lang, offset, length))
        return page(data[offset : offset + length], len(data), offset)

    rows = ftprep.fetch_language("en", {"nutrition"}, get_page=get_page, page_size=10, workers=1)
    assert [r["sample_id"] for r in rows] == [d["sample_id"] for d in data if d["subject"] == "nutrition"]
    assert sorted(o for _, o, _ in calls) == [0, 10, 20]
    assert set(rows[0]) == set(ftprep.RAW_FIELDS)


def test_fetch_language_retries_a_failed_page_then_succeeds():
    data = [raw_row("en", i, subject="nutrition") for i in range(5)]
    attempts = {"n": 0}

    def get_page(lang, offset, length):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("timeout")
        return page(data[offset : offset + length], len(data), offset)

    rows = ftprep.fetch_language("en", {"nutrition"}, get_page=get_page, page_size=10, workers=1, sleep=lambda s: None)
    assert len(rows) == 5 and attempts["n"] == 2


def test_fetch_language_gives_up_after_the_attempts_and_says_which_page():
    def get_page(lang, offset, length):
        raise OSError("down")

    with pytest.raises(RuntimeError, match="offset 0"):
        ftprep.fetch_language("en", {"nutrition"}, get_page=get_page, page_size=10, workers=1, attempts=2, sleep=lambda s: None)


def test_fetch_language_waits_as_long_as_a_rate_limit_asks():
    data = [raw_row("en", i, subject="nutrition") for i in range(5)]
    waits = []
    state = {"n": 0}

    def get_page(lang, offset, length):
        state["n"] += 1
        if state["n"] == 1:
            raise ftprep.RateLimited(17)
        return page(data[offset : offset + length], len(data), offset)

    rows = ftprep.fetch_language("en", {"nutrition"}, get_page=get_page, page_size=10, workers=1, sleep=waits.append)
    assert len(rows) == 5
    assert waits == [17]


def test_fetch_language_refuses_a_short_read():
    data = [raw_row("en", i, subject="nutrition") for i in range(5)]

    def get_page(lang, offset, length):
        return page(data[offset : offset + length][:2], 5, offset)  # server returns fewer rows than asked

    with pytest.raises(RuntimeError, match="rows"):
        ftprep.fetch_language("en", {"nutrition"}, get_page=get_page, page_size=10, workers=1)


def test_fetch_languages_are_the_study_languages_the_source_covers():
    assert set(ftprep.SOURCE_LANGS) == {"en", "hi", "ar", "sw"}
    assert set(ftprep.SOURCE_LANGS) <= set(LANGS)
