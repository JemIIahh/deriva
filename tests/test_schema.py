import pytest

from msd.schema import Generation, Prompt, TaskScore, Verdict, read_jsonl, write_jsonl

ARABIC = "كيف أجدد بطاقتي؟"
HINDI = "मैं अपना आईडी कैसे नवीनीकृत करूँ?"


def test_roundtrip_preserves_rtl_and_devanagari(tmp_path):
    rows = [
        Prompt(id="p1", lang="ar", kind="benign_edgy", text=ARABIC, split="dev",
               source="seed", license="CC0"),
        Prompt(id="p2", lang="hi", kind="task", text=HINDI, split="report",
               source="seed", license="CC0", gold="B"),
    ]
    path = tmp_path / "prompts.jsonl"
    write_jsonl(path, rows)

    back = read_jsonl(path, Prompt)
    assert back == rows
    assert back[0].text == ARABIC
    assert back[1].text == HINDI
    assert back[0].gold is None
    assert back[1].gold == "B"

    raw = path.read_bytes()
    assert ARABIC.encode("utf-8") in raw
    assert HINDI.encode("utf-8") in raw
    assert b"\\u" not in raw


def test_roundtrip_generation_verdict_and_taskscore_with_none_alpha(tmp_path):
    gens = [
        Generation(prompt_id="p1", arm="BASE", alpha=None, text=ARABIC, finish_reason="stop"),
        Generation(prompt_id="p1", arm="A1", alpha=0.25, text="ok", finish_reason="length"),
    ]
    verdicts = [
        Verdict(prompt_id="p1", arm="BASE", alpha=None, lang="ar", kind="harmful",
                judge_id="j1", label="refused", excerpt=ARABIC),
    ]
    scores = [TaskScore(prompt_id="p2", arm="A2", alpha=0.5, lang="hi", correct=True)]

    for name, rows, cls in (
        ("gens.jsonl", gens, Generation),
        ("verdicts.jsonl", verdicts, Verdict),
        ("scores.jsonl", scores, TaskScore),
    ):
        path = tmp_path / name
        write_jsonl(path, rows)
        assert read_jsonl(path, cls) == rows

    back = read_jsonl(tmp_path / "gens.jsonl", Generation)
    assert back[0].alpha is None
    assert back[1].alpha == 0.25


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        Verdict(prompt_id="p1", arm="BASE", alpha=None, lang="en", kind="harmful",
                judge_id="j1", label="safe", excerpt="x")


def test_unknown_kind_and_split_rejected():
    with pytest.raises(ValueError):
        Prompt(id="p1", lang="en", kind="spicy", text="x", split="dev",
               source="s", license="l")
    with pytest.raises(ValueError):
        Prompt(id="p1", lang="en", kind="harmful", text="x", split="train",
               source="s", license="l")


def test_dataclasses_are_frozen():
    p = Prompt(id="p1", lang="en", kind="harmful", text="x", split="dev",
               source="s", license="l")
    with pytest.raises(AttributeError):
        p.text = "y"
