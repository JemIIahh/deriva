"""msd.gen: the generation runner with fake backends (no model, no GPU, no torch)."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from msd import gen
from msd.gen import BackendError, GenOut, ResumeError, generate
from msd.schema import Generation, Prompt, read_jsonl, write_jsonl


def make_prompts(n, prefix="p"):
    return [
        Prompt(id=f"{prefix}{i}", lang="en", kind="harmful", text=f"text {prefix}{i}", split="dev",
               source="fixture", license="fixture")
        for i in range(n)
    ]


class Recorder:
    """A fake backend: remembers each call's input and answers `out:<prompt text>`."""

    def __init__(self, finish_reason="stop"):
        self.calls = []
        self.finish_reason = finish_reason

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [GenOut(f"out:{t}", self.finish_reason) for t in texts]

    @property
    def sizes(self):
        return [len(c) for c in self.calls]


def ids_in(path):
    return [g.prompt_id for g in read_jsonl(path, Generation)]


def seed_file(path, rows):
    write_jsonl(path, rows)


def gen_row(pid, arm="A1", alpha=None, text="seeded", finish="stop"):
    return Generation(pid, arm, alpha, text, finish)


# --- the four brief tests -------------------------------------------------------------------------


def test_resume_skips_finished_ids(tmp_path):
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(5)
    seed_file(out, [gen_row("p0"), gen_row("p1")])
    backend = Recorder()

    n = generate(prompts, backend, out, arm="A1", alpha=None, batch_size=2)

    assert n == 3
    assert backend.calls == [["text p2", "text p3"], ["text p4"]]  # finished prompts never reach the backend
    rows = read_jsonl(out, Generation)
    assert [r.prompt_id for r in rows] == ["p0", "p1", "p2", "p3", "p4"]
    assert rows[0].text == "seeded" and rows[1].text == "seeded"  # existing rows untouched
    assert rows[2].text == "out:text p2"


def test_no_duplicate_rows_after_rerun(tmp_path):
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(7)
    first = Recorder()
    assert generate(prompts, first, out, arm="A1", alpha=None, batch_size=3) == 7
    before = out.read_bytes()

    second = Recorder()
    assert generate(prompts, second, out, arm="A1", alpha=None, batch_size=3) == 0

    assert second.calls == []  # nothing left to do: the backend is not even called
    assert out.read_bytes() == before  # byte-identical
    assert len(set(ids_in(out))) == len(ids_in(out)) == 7


@pytest.mark.parametrize(
    "n_prompts, batch_size, expected",
    [(7, 3, [3, 3, 1]), (6, 3, [3, 3]), (2, 8, [2]), (3, 1, [1, 1, 1]), (0, 4, [])],
)
def test_batches_respect_batch_size(tmp_path, n_prompts, batch_size, expected):
    backend = Recorder()
    n = generate(make_prompts(n_prompts), backend, tmp_path / "g.jsonl", arm="A0", alpha=None, batch_size=batch_size)
    assert backend.sizes == expected
    assert n == n_prompts
    assert all(size <= batch_size for size in backend.sizes)


def test_batches_cover_only_the_unfinished_prompts_in_input_order(tmp_path):
    out = tmp_path / "g.jsonl"
    seed_file(out, [gen_row("p0", "A0"), gen_row("p1", "A0"), gen_row("p2", "A0")])
    backend = Recorder()
    generate(make_prompts(8), backend, out, arm="A0", alpha=None, batch_size=2)
    assert backend.calls == [["text p3", "text p4"], ["text p5", "text p6"], ["text p7"]]


def test_finish_reason_is_recorded(tmp_path):
    reasons = ["stop", "length", "error", "stop"]

    def backend(texts):
        return [GenOut(f"t{i}", reasons.pop(0)) for i, _ in enumerate(texts)]

    out = tmp_path / "g.jsonl"
    generate(make_prompts(4), backend, out, arm="A0", alpha=None, batch_size=2)
    assert [r.finish_reason for r in read_jsonl(out, Generation)] == ["stop", "length", "error", "stop"]


# --- backend contract ------------------------------------------------------------------------------


def test_unknown_finish_reason_from_backend_is_refused_and_nothing_is_written(tmp_path):
    out = tmp_path / "g.jsonl"
    with pytest.raises(BackendError, match="finish_reason"):
        generate(make_prompts(2), lambda texts: [GenOut("x", "eos") for _ in texts], out, arm="A0", alpha=None)
    assert not out.exists()


@pytest.mark.parametrize("returned", [1, 3])
def test_backend_returning_the_wrong_count_is_a_loud_error(tmp_path, returned):
    out = tmp_path / "g.jsonl"

    def backend(texts):
        return [GenOut("x", "stop")] * returned

    with pytest.raises(BackendError, match="2"):
        generate(make_prompts(2), backend, out, arm="A0", alpha=None, batch_size=2)
    assert not out.exists()  # never silently misaligned


def test_backend_returning_non_genout_items_is_refused(tmp_path):
    with pytest.raises(BackendError, match="GenOut"):
        generate(make_prompts(1), lambda texts: ["plain string"], tmp_path / "g.jsonl", arm="A0", alpha=None)
    with pytest.raises(BackendError, match="list"):
        generate(make_prompts(1), lambda texts: None, tmp_path / "g2.jsonl", arm="A0", alpha=None)


def test_bad_batch_keeps_earlier_batches_and_a_rerun_finishes_the_rest(tmp_path):
    out = tmp_path / "g.jsonl"
    calls = []

    def flaky(texts):
        calls.append(list(texts))
        if len(calls) == 3:
            raise RuntimeError("CUDA out of memory (pretend)")
        return [GenOut(f"out:{t}", "stop") for t in texts]

    prompts = make_prompts(7)
    with pytest.raises(RuntimeError, match="out of memory"):
        generate(prompts, flaky, out, arm="A1", alpha=None, batch_size=3)
    assert ids_in(out) == ["p0", "p1", "p2", "p3", "p4", "p5"]  # two full batches persisted; nothing of the failing one

    healthy = Recorder()
    assert generate(prompts, healthy, out, arm="A1", alpha=None, batch_size=3) == 1
    assert healthy.calls == [["text p6"]]
    assert ids_in(out) == [p.id for p in prompts]


def test_wrong_count_on_a_later_batch_keeps_the_earlier_ones(tmp_path):
    out = tmp_path / "g.jsonl"
    state = {"n": 0}

    def backend(texts):
        state["n"] += 1
        return [GenOut("x", "stop") for _ in texts][: len(texts) - (1 if state["n"] == 2 else 0)]

    with pytest.raises(BackendError):
        generate(make_prompts(4), backend, out, arm="A0", alpha=None, batch_size=2)
    assert ids_in(out) == ["p0", "p1"]


def test_empty_text_is_preserved_and_not_regenerated(tmp_path):
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(3)
    texts = {"text p0": "", "text p1": "fine", "text p2": ""}
    backend = lambda batch: [GenOut(texts[t], "stop") for t in batch]  # noqa: E731
    generate(prompts, backend, out, arm="A0", alpha=None)
    rows = read_jsonl(out, Generation)
    assert [r.text for r in rows] == ["", "fine", ""]  # empty means no_response downstream; never dropped

    again = Recorder()
    assert generate(prompts, again, out, arm="A0", alpha=None) == 0
    assert again.calls == []


def test_backend_receives_the_raw_prompt_text_unchanged(tmp_path):
    prompt = Prompt(id="x1", lang="hi", kind="harmful", text="  नमस्ते\nसंसार  ", split="dev", source="s", license="l")
    backend = Recorder()
    generate([prompt], backend, tmp_path / "g.jsonl", arm="A0", alpha=None)
    assert backend.calls == [["  नमस्ते\nसंसार  "]]


# --- rows: format, order, alpha ---------------------------------------------------------------------


def test_rows_are_generation_records_with_arm_and_alpha(tmp_path):
    out = tmp_path / "g.jsonl"
    generate(make_prompts(2), Recorder("length"), out, arm="A2", alpha=0.25)
    assert read_jsonl(out, Generation) == [
        Generation("p0", "A2", 0.25, "out:text p0", "length"),
        Generation("p1", "A2", 0.25, "out:text p1", "length"),
    ]


def test_alpha_is_stored_in_its_canonical_form(tmp_path):
    out = tmp_path / "g.jsonl"
    generate(make_prompts(1), Recorder(), out, arm="A2n", alpha=0.1 + 0.2)  # 0.30000000000000004
    assert read_jsonl(out, Generation)[0].alpha == 0.3
    # and the same blend asked for as 0.3 resumes the file instead of tripping the wrong-alpha guard
    assert generate(make_prompts(2), Recorder(), out, arm="A2n", alpha=0.3) == 1


def test_written_bytes_equal_write_jsonl_bytes_for_awkward_text(tmp_path):
    awkward = "line\nbreak \r\n tab\t quote\" back\\slash \u2028 sep \u0085 nel \x0b vt \x0c ff é 😀 مرحبا नमस्ते"
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(2)
    generate(prompts, lambda texts: [GenOut(awkward, "stop") for _ in texts], out, arm="A0", alpha=None)

    reference = tmp_path / "ref.jsonl"
    write_jsonl(reference, [Generation(p.id, "A0", None, awkward, "stop") for p in prompts])
    assert out.read_bytes() == reference.read_bytes()
    back = read_jsonl(out, Generation)
    assert [r.text for r in back] == [awkward, awkward]
    assert out.read_bytes().count(b"\n") == 2  # exotic separators never add a physical line
    assert "é 😀 مرحبا नमस्ते".encode("utf-8") in out.read_bytes()  # raw UTF-8 for non-ASCII, not \u escapes


def test_written_file_survives_a_resume_with_awkward_text(tmp_path):
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(4)
    awkward = lambda texts: [GenOut("a\u2028b\u0085c\x0bd\x0ce", "stop") for _ in texts]  # noqa: E731
    generate(prompts[:2], awkward, out, arm="A0", alpha=None)
    assert generate(prompts, awkward, out, arm="A0", alpha=None) == 2  # exotic separators are not read as torn lines
    assert ids_in(out) == [p.id for p in prompts]


def test_parent_directories_are_created(tmp_path):
    out = tmp_path / "a" / "b" / "c" / "g.jsonl"
    generate(make_prompts(1), Recorder(), out, arm="A0", alpha=None)
    assert out.is_file()


def test_progress_callback_is_called_after_each_persisted_batch(tmp_path):
    seen = []
    generate(make_prompts(5), Recorder(), tmp_path / "g.jsonl", arm="A0", alpha=None, batch_size=2,
             progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 5), (4, 5), (5, 5)]


# --- crash recovery ----------------------------------------------------------------------------------


def _good_lines(n, arm="A1"):
    """The exact bytes of n finished rows (p0..p{n-1}) as the runner writes them."""
    return b"".join(
        (json.dumps(asdict(gen_row(f"p{i}", arm)), ensure_ascii=False) + "\n").encode("utf-8") for i in range(n)
    )


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(b'{"prompt_id": "p2", "arm": "A1", "alpha": nu', id="cut-mid-json-no-newline"),
        pytest.param(b'{"prompt_id": "p2", "arm": "A1", "alpha": null, "text": "complete", "finish_reason": "stop"}',
                     id="complete-json-but-no-newline"),
        pytest.param(b'{"prompt_id": "p2", "arm": "A1", "alp\n', id="garbage-line-with-newline"),
        pytest.param(b"\x00\x00\x00\x00\x00\x00", id="zero-filled-tail"),
        pytest.param(b'{"prompt_id": "p2", "text": "\xff\xfe broken utf8', id="invalid-utf8"),
        pytest.param(b'{"prompt_id": "p2", "arm": "A1"}\n', id="valid-json-missing-fields"),
        pytest.param(b"\n", id="blank-last-line"),
    ],
)
def test_a_torn_last_line_is_dropped_and_its_prompt_regenerated(tmp_path, tail):
    out = tmp_path / "g.jsonl"
    good = _good_lines(2)
    out.write_bytes(good + tail)
    prompts = make_prompts(4)
    backend = Recorder()

    n = generate(prompts, backend, out, arm="A1", alpha=None, batch_size=8)

    assert n == 2
    assert backend.calls == [["text p2", "text p3"]]  # p2 was regenerated, not trusted
    assert out.read_bytes().startswith(good)  # the two good rows are untouched
    rows = read_jsonl(out, Generation)  # and the file parses cleanly end to end
    assert [r.prompt_id for r in rows] == ["p0", "p1", "p2", "p3"]
    assert rows[2].text == "out:text p2"
    assert out.read_bytes().endswith(b"\n")


def test_a_torn_tail_alone_leaves_the_file_truncated_to_the_last_good_line(tmp_path):
    out = tmp_path / "g.jsonl"
    good = _good_lines(3)
    out.write_bytes(good + b'{"prompt_id": "p3", "ar')
    backend = Recorder()
    assert generate(make_prompts(3), backend, out, arm="A1", alpha=None) == 0  # all three prompts are finished
    assert out.read_bytes() == good  # the torn bytes are gone
    assert backend.calls == []


def test_a_partial_batch_prefix_survives_a_torn_write(tmp_path):
    # the runner writes a whole batch in one write; a crash can keep the first rows and tear the next one
    out = tmp_path / "g.jsonl"
    out.write_bytes(_good_lines(3) + b'{"prompt_id": "p3", "arm": "A1", "alpha": null, "te')
    backend = Recorder()
    assert generate(make_prompts(5), backend, out, arm="A1", alpha=None, batch_size=2) == 2
    assert backend.calls == [["text p3", "text p4"]]
    assert ids_in(out) == ["p0", "p1", "p2", "p3", "p4"]


def test_a_corrupt_middle_line_is_a_loud_error_and_the_file_is_not_touched(tmp_path):
    out = tmp_path / "g.jsonl"
    lines = _good_lines(3).split(b"\n")
    damaged = b"\n".join([lines[0], b'{"prompt_id": "p1", "arm"', lines[2], b""])
    out.write_bytes(damaged)
    backend = Recorder()
    with pytest.raises(ResumeError, match="line 2"):
        generate(make_prompts(5), backend, out, arm="A1", alpha=None)
    assert out.read_bytes() == damaged
    assert backend.calls == []


def test_a_blank_line_in_the_middle_is_a_loud_error(tmp_path):
    out = tmp_path / "g.jsonl"
    lines = _good_lines(2).split(b"\n")
    out.write_bytes(b"\n".join([lines[0], b"", lines[1], b""]))
    with pytest.raises(ResumeError, match="line 2"):
        generate(make_prompts(3), Recorder(), out, arm="A1", alpha=None)


def test_a_torn_tail_after_a_corrupt_middle_line_is_still_an_error(tmp_path):
    out = tmp_path / "g.jsonl"
    good = _good_lines(2).split(b"\n")
    out.write_bytes(b"\n".join([good[0], b"not json at all", b'{"prompt_id": "p2", "ar']))
    with pytest.raises(ResumeError, match="line 2"):
        generate(make_prompts(3), Recorder(), out, arm="A1", alpha=None)


def test_duplicate_ids_in_an_existing_file_are_a_loud_error(tmp_path):
    out = tmp_path / "g.jsonl"
    seed_file(out, [gen_row("p0"), gen_row("p1"), gen_row("p0")])
    backend = Recorder()
    with pytest.raises(ResumeError, match="p0"):
        generate(make_prompts(4), backend, out, arm="A1", alpha=None)
    assert backend.calls == []


@pytest.mark.parametrize(
    "existing_arm, existing_alpha, run_arm, run_alpha",
    [
        ("A0", None, "A1", None),  # another arm
        ("A2", 0.25, "A2", 0.5),  # another alpha
        ("A2", 0.25, "A3", None),
        ("A3", None, "A3", 0.5),  # None vs a number
        ("A3", 0.5, "A3", None),
    ],
)
def test_rows_of_another_arm_or_alpha_in_an_existing_file_are_a_loud_error(
    tmp_path, existing_arm, existing_alpha, run_arm, run_alpha
):
    out = tmp_path / "g.jsonl"
    seed_file(out, [gen_row("p0", existing_arm, existing_alpha)])
    before = out.read_bytes()
    backend = Recorder()
    with pytest.raises(ResumeError, match="arm"):
        generate(make_prompts(3), backend, out, arm=run_arm, alpha=run_alpha)
    assert out.read_bytes() == before and backend.calls == []


def test_existing_rows_for_ids_not_in_the_prompt_list_are_a_loud_error(tmp_path):
    out = tmp_path / "g.jsonl"
    seed_file(out, [gen_row("p0"), gen_row("zzz")])
    with pytest.raises(ResumeError, match="zzz"):
        generate(make_prompts(3), Recorder(), out, arm="A1", alpha=None)


def test_duplicate_prompt_ids_in_the_input_are_a_loud_error(tmp_path):
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(3) + [make_prompts(1)[0]]
    backend = Recorder()
    with pytest.raises(ValueError, match="p0"):
        generate(prompts, backend, out, arm="A1", alpha=None)
    assert backend.calls == [] and not out.exists()


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, None])
def test_bad_batch_sizes_are_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="batch_size"):
        generate(make_prompts(2), Recorder(), tmp_path / "g.jsonl", arm="A0", alpha=None, batch_size=bad)


def test_a_failed_write_is_rolled_back(tmp_path, monkeypatch):
    out = tmp_path / "g.jsonl"
    prompts = make_prompts(4)
    generate(prompts[:2], Recorder(), out, arm="A0", alpha=None)
    before = out.read_bytes()

    real_fsync = os.fsync
    state = {"calls": 0}

    def flaky_fsync(fd):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("disk full (pretend)")
        return real_fsync(fd)

    monkeypatch.setattr(gen.os, "fsync", flaky_fsync)
    with pytest.raises(OSError, match="disk full"):
        generate(prompts, Recorder(), out, arm="A0", alpha=None, batch_size=2)
    monkeypatch.undo()
    assert out.read_bytes() == before  # nothing partial from the failing batch stays behind

    assert generate(prompts, Recorder(), out, arm="A0", alpha=None, batch_size=2) == 2


# --- arm / alpha rules and paths ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm, alpha, ok",
    [
        ("BASE", None, True), ("A0", None, True), ("A1", None, True),
        ("BASE", 0.5, False), ("A0", 0.0, False), ("A1", 1.0, False),
        ("A2", 0.5, True), ("A2", 0.0, True), ("A2", 1.0, True), ("A2", None, False),
        ("A2n", 0.25, True), ("A2n", None, False),
        ("A3", None, True), ("A3", 0.5, True),  # A3 is a fine-tune: alpha is allowed but not required
        ("A9", None, False), ("a0", None, False),
        ("A2", float("nan"), False), ("A2", float("inf"), False), ("A2", -0.1, False),
    ],
)
def test_check_arm_alpha(arm, alpha, ok):
    if ok:
        gen.check_arm_alpha(arm, alpha)
    else:
        with pytest.raises(ValueError):
            gen.check_arm_alpha(arm, alpha)


def test_generate_applies_the_arm_alpha_rules(tmp_path):
    with pytest.raises(ValueError, match="alpha"):
        generate(make_prompts(1), Recorder(), tmp_path / "g.jsonl", arm="A2", alpha=None)
    with pytest.raises(ValueError, match="alpha"):
        generate(make_prompts(1), Recorder(), tmp_path / "g.jsonl", arm="A0", alpha=0.5)


@pytest.mark.parametrize(
    "alpha, text",
    [(None, "none"), (0.25, "0.25"), (0.5, "0.5"), (1.0, "1.0"), (0.0, "0.0"), (0.75, "0.75"),
     (0.1 + 0.2, "0.3"), (0.125, "0.125"), (-0.0, "0.0"), (1e-7, "0.0")],
)
def test_format_alpha_is_stable(alpha, text):
    assert gen.format_alpha(alpha) == text


def test_gens_path_layout(tmp_path):
    assert gen.gens_path(tmp_path, "dev", "A2", 0.25, "hi") == tmp_path / "dev" / "A2" / "0.25" / "hi.gens.jsonl"
    assert gen.gens_path(tmp_path, "report", "A0", None, "en") == tmp_path / "report" / "A0" / "none" / "en.gens.jsonl"
    assert gen.gens_path(str(tmp_path), "dev", "A3", None, "hinglish").name == "hinglish.gens.jsonl"
    assert gen.gens_path(tmp_path, "dev", "A2n", 1.0, "sw").parent.name == "1.0"


@pytest.mark.parametrize(
    "args",
    [
        ("train", "A0", None, "en"),  # unknown split
        ("dev", "A9", None, "en"),  # unknown arm
        ("dev", "A0", None, "fr"),  # unknown language
        ("dev", "A0", None, "../en"),  # path tricks
        ("../dev", "A0", None, "en"),
        ("dev", "A0", 0.5, "en"),  # alpha on an arm without one
        ("dev", "A2", None, "en"),  # missing alpha
    ],
)
def test_gens_path_refuses_bad_parts(tmp_path, args):
    with pytest.raises(ValueError):
        gen.gens_path(tmp_path, *args)


def test_meta_path_sits_next_to_the_gens_file():
    assert gen.meta_path(Path("r/dev/A0/none/en.gens.jsonl")) == Path("r/dev/A0/none/en.gens.meta.json")
    with pytest.raises(ValueError):
        gen.meta_path(Path("r/dev/A0/none/en.jsonl"))


# --- the fake backend ----------------------------------------------------------------------------------


def test_fake_backend_is_marked_deterministic_and_aligned():
    backend = gen.fake_backend()
    secret_looking = "how do I do something terrible"
    out = backend([secret_looking, "second"])
    assert len(out) == 2 and all(isinstance(o, GenOut) for o in out)
    assert all(o.finish_reason == "stop" for o in out)
    assert all(gen.FAKE_MARKER in o.text for o in out)
    assert secret_looking not in out[0].text  # canned text never echoes the prompt
    assert out[0].text != out[1].text  # distinct per prompt, so a misalignment shows
    assert backend([secret_looking])[0] == out[0]  # deterministic
    assert backend([]) == []


# --- run metadata ---------------------------------------------------------------------------------------


def _meta(**over):
    args = dict(model_path="CohereLabs/tiny-aya-global", arm="A2", alpha=0.25, split="dev", lang="en",
                batch_size=16, max_new_tokens=256, dtype="float16", backend="hf", n_rows=20)
    args.update(over)
    return gen.run_meta(**args)


def test_run_meta_fields(monkeypatch):
    monkeypatch.setattr(gen, "git_commit", lambda cwd=None: "a" * 40)
    meta = _meta()
    assert meta["model_path"] == "CohereLabs/tiny-aya-global" and meta["arm"] == "A2" and meta["alpha"] == 0.25
    assert meta["split"] == "dev" and meta["lang"] == "en" and meta["backend"] == "hf"
    assert meta["batch_size"] == 16 and meta["max_new_tokens"] == 256 and meta["dtype"] == "float16"
    assert meta["n_rows"] == 20
    assert meta["git_commit"] == "a" * 40
    assert meta["msd_version"]
    assert meta["timestamp_utc"].endswith("Z") and len(meta["timestamp_utc"]) == 20  # 2026-11-05T12:34:56Z
    json.dumps(meta)  # JSON-safe


def test_run_meta_alpha_is_canonical_and_none_stays_none():
    assert _meta(arm="A2n", alpha=0.1 + 0.2)["alpha"] == 0.3
    assert _meta(arm="A0", alpha=None)["alpha"] is None


def test_run_meta_git_commit_is_best_effort(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(gen.subprocess, "run", boom)
    assert gen.git_commit() is None
    assert _meta()["git_commit"] is None


def test_git_commit_in_this_repository_is_a_full_hash_or_none():
    commit = gen.git_commit(Path(__file__).resolve().parent)
    assert commit is None or (len(commit) == 40 and set(commit) <= set("0123456789abcdef"))


def test_run_meta_never_contains_the_token(monkeypatch):
    token = "hf_" + "A1b2C3d4E5" * 4
    monkeypatch.setenv("HF_TOKEN", token)
    meta = _meta(model_path=f"/mnt/{token}/merged")  # someone pasted a token into a path
    text = json.dumps(meta)
    assert token not in text and "hf_A1b2" not in text
    monkeypatch.setenv("SOME_API_KEY", "super-secret-value-123")
    assert "super-secret-value-123" not in json.dumps(_meta(model_path="x/super-secret-value-123/y"))


def test_run_meta_has_no_environment_dump(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "Z" * 30)
    assert set(_meta()) <= {
        "model_path", "arm", "alpha", "split", "lang", "batch_size", "max_new_tokens", "dtype", "backend",
        "n_rows", "msd_version", "git_commit", "timestamp_utc", "earlier",
    }


# --- laziness -------------------------------------------------------------------------------------------


def test_importing_gen_does_not_import_torch_or_transformers():
    code = (
        "import sys, msd.gen, msd.genrun\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers')]\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_hf_backend_without_torch_fails_clearly_and_only_when_called():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None  # `import torch` raises ImportError\n"
        "sys.modules['transformers'] = None\n"
        "import msd.gen as gen  # importing never needs them\n"
        "try:\n"
        "    gen.hf_backend('some/model')\n"
        "except ImportError as exc:\n"
        "    assert 'torch' in str(exc) and 'transformers' in str(exc), exc\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
