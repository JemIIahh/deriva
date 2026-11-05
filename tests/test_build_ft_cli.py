"""scripts/build_ft_data.py prepare and finalize end to end on tiny local raw files.

CuratorKIT itself is not run here (it lives in its own venv); the test stands in for it by
exporting the candidates the way its messages exporter does, minus one dropped sample.
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

from msd import ftdata

from test_ftprep import ALLOWED, make_raw

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
BUILD = [sys.executable, str(SCRIPTS / "build_ft_data.py")]
VALIDATE = [sys.executable, str(SCRIPTS / "validate_ft_data.py")]
LANGS4 = ("en", "hi", "ar", "sw")


def write_pipeline(path, **extra):
    settings = {
        "harmless_subjects": sorted(ALLOWED),
        "split_seed": 5,
        "heldout_pool": 6,
        "train_pool": 24,
        "heldout_cap": 4,
        "train_cap": 20,
    }
    settings.update(extra)
    path.write_text(yaml.safe_dump({"name": "t", "extra": settings}), encoding="utf-8")


def run(cmd, *args):
    return subprocess.run([*cmd, *args], capture_output=True, text=True, encoding="utf-8")


def test_prepare_then_finalize_then_validate(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw, n=50, other=4)
    pipeline = tmp_path / "pipe.yaml"
    write_pipeline(pipeline)
    cands_path = tmp_path / "ft" / "prep" / "candidates.jsonl"

    prep = run(BUILD, "prepare", "--pipeline", str(pipeline), "--raw", str(raw), "--out", str(cands_path))
    assert prep.returncode == 0, prep.stderr
    cands = [json.loads(line) for line in cands_path.read_text(encoding="utf-8").splitlines()]
    assert len(cands) == 4 * (6 + 24)
    summary = json.loads((cands_path.parent / "prep_summary.json").read_text(encoding="utf-8"))
    assert summary["dropped"]["en"] == {"subject_not_allowed": 4}

    # stand in for CuratorKIT: export every candidate's messages, minus one it "deduplicated"
    curated = tmp_path / "ft" / "curated"
    curated.mkdir()
    lost = cands[3]
    lines = [
        {"messages": [{"role": "user", "content": c["instruction"]}, {"role": "assistant", "content": c["output"]}]}
        for c in cands
        if c is not lost
    ]
    (curated / "sft_messages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")
    (curated / "manifest.json").write_text(json.dumps({"stage_counts": {"ExactDeduplicator": {"input_count": 120, "output_count": 119}}}), encoding="utf-8")

    fin = run(BUILD, "finalize", "--pipeline", str(pipeline), "--curated", str(curated), "--candidates", str(cands_path), "--out", str(tmp_path / "ft"))
    assert fin.returncode == 0, fin.stderr
    train = [json.loads(line) for line in (tmp_path / "ft" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    held = [json.loads(line) for line in (tmp_path / "ft" / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    # 24 train and 6 heldout candidates per language; the lost one is an English train row, but 23 are
    # still more than the cap of 20, so the caps (20 train, 4 heldout per language) decide the counts
    assert lost["split"] == "train" and lost["lang"] == "en"
    assert len(train) == 4 * 20 and len(held) == 4 * 4
    assert lost["id"] not in {r["id"] for r in train}
    assert all(r["gold"] == r["messages"][1]["content"] for r in train + held)
    build = json.loads((tmp_path / "ft" / "build_summary.json").read_text(encoding="utf-8"))
    assert build["curatorkit_stage_counts"]["ExactDeduplicator"]["output_count"] == 119
    assert build["removed_by_curatorkit"][lost["split"]] == {lost["lang"]: 1}

    # the last step of the recipe: the validator accepts what finalize wrote
    testset = tmp_path / "testset"
    (testset / "en").mkdir(parents=True)
    for split in ("dev", "report"):
        (testset / "en" / f"{split}.jsonl").write_text(
            json.dumps({"id": f"x-{split}", "lang": "en", "kind": "harmful", "text": f"harmful {split} prompt", "split": split, "source": "s", "license": "mit", "gold": None}) + "\n",
            encoding="utf-8",
        )
    val = run(
        VALIDATE,
        "--train", str(tmp_path / "ft" / "train.jsonl"),
        "--heldout", str(tmp_path / "ft" / "heldout.jsonl"),
        "--testset", str(testset),
        "--pipeline", str(pipeline),
        "--langs", ",".join(LANGS4),
    )
    assert val.returncode == 0, val.stdout + val.stderr
    assert "hinglish" in val.stdout


def test_finalize_refuses_an_export_line_it_cannot_trace(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw, n=30)
    pipeline = tmp_path / "pipe.yaml"
    write_pipeline(pipeline)
    cands_path = tmp_path / "ft" / "prep" / "candidates.jsonl"
    assert run(BUILD, "prepare", "--pipeline", str(pipeline), "--raw", str(raw), "--out", str(cands_path)).returncode == 0
    curated = tmp_path / "ft" / "curated"
    curated.mkdir()
    (curated / "sft_messages.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "made up"}, {"role": "assistant", "content": "B"}]}) + "\n",
        encoding="utf-8",
    )
    fin = run(BUILD, "finalize", "--pipeline", str(pipeline), "--curated", str(curated), "--candidates", str(cands_path), "--out", str(tmp_path / "ft"))
    assert fin.returncode == 2
    assert "no candidate" in fin.stderr


def test_prepare_without_the_raw_files_says_how_to_fetch_them(tmp_path):
    pipeline = tmp_path / "pipe.yaml"
    write_pipeline(pipeline)
    result = run(BUILD, "prepare", "--pipeline", str(pipeline), "--raw", str(tmp_path / "nothing"), "--out", str(tmp_path / "c.jsonl"))
    assert result.returncode == 2
    assert "build_ft_data.py fetch" in result.stderr


def test_a_language_without_a_source_is_refused(tmp_path):
    result = run(BUILD, "fetch", "--langs", "hinglish", "--pipeline", str(tmp_path / "x.yaml"))
    assert result.returncode == 2
    assert "hinglish" in result.stderr


def test_the_settings_in_the_real_pipeline_yaml_are_complete():
    from msd import ftprep

    settings = ftprep.load_settings(Path(__file__).resolve().parent.parent / "pipelines" / "ft_data.yaml")
    assert settings["train_cap"] == 1500 and settings["heldout_cap"] == 200  # the brief's targets
    assert settings["train_pool"] >= settings["train_cap"] and settings["heldout_pool"] >= settings["heldout_cap"]
    assert "virology" not in settings["harmless_subjects"]
    assert not {"security_studies", "computer_security", "moral_scenarios"} & settings["harmless_subjects"]
    # borderline subjects dropped in fix round 1: sociology is contested social science and
    # clinical_knowledge is clinical medicine, neither is everyday civics or basic science
    assert not {"sociology", "clinical_knowledge"} & settings["harmless_subjects"]
    assert len(settings["harmless_subjects"]) == 14
    assert ftdata.load_allowed_subjects(Path(__file__).resolve().parent.parent / "pipelines" / "ft_data.yaml") == settings["harmless_subjects"]


def test_the_pipeline_yaml_masks_exactly_the_instruction_lines_ftprep_writes():
    import re

    from msd import ftprep

    config = yaml.safe_load((Path(__file__).resolve().parent.parent / "pipelines" / "ft_data.yaml").read_text(encoding="utf-8"))
    for normalizer in (n for n in config["normalizers"] if n["type"] in ("exact_dedup", "minhash_dedup")):
        patterns = [re.compile(p) for p in normalizer["ignore_for_dedup"]]
        for lang, line in ftprep.INSTRUCTION_LINES.items():
            assert any(p.fullmatch(line) for p in patterns), (normalizer["type"], lang)
        assert any(p.search("A. some option") for p in patterns)  # the option labels are masked too
    samplers = [n for n in config["normalizers"] if n["type"] == "stratified_sampler"]
    strata = {k for n in samplers for k in n["target_distribution"]}
    assert strata == {f"{split}:{lang}" for split in ("train", "heldout") for lang in ftprep.SOURCE_LANGS}
    assert [e["type"] for e in config["exporters"]] == ["messages"]
    assert "generators" not in config  # no LLM key: the generator step is only a commented stub


# --- taskdev: built by the same commands ---------------------------------------------------------


def build_all(tmp_path, name, **extra):
    """prepare, a stand-in for CuratorKIT (export everything), finalize; returns the ft directory."""
    raw = tmp_path / "raw"
    if not raw.exists():
        make_raw(raw, n=80, other=3)
    pipeline = tmp_path / f"{name}.yaml"
    write_pipeline(pipeline, taskdev_pool=10, taskdev_cap=8, **extra)
    ft = tmp_path / name
    cands_path = ft / "prep" / "candidates.jsonl"
    prep = run(BUILD, "prepare", "--pipeline", str(pipeline), "--raw", str(raw), "--out", str(cands_path))
    assert prep.returncode == 0, prep.stderr
    cands = [json.loads(line) for line in cands_path.read_text(encoding="utf-8").splitlines()]
    curated = ft / "curated"
    curated.mkdir()
    lines = [{"messages": [{"role": "user", "content": c["instruction"]}, {"role": "assistant", "content": c["output"]}]} for c in cands]
    (curated / "sft_messages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")
    fin = run(BUILD, "finalize", "--pipeline", str(pipeline), "--curated", str(curated), "--candidates", str(cands_path), "--out", str(ft))
    assert fin.returncode == 0, fin.stderr
    return ft, pipeline, cands


def test_taskdev_is_built_by_prepare_and_finalize_and_the_validator_accepts_it(tmp_path):
    ft, pipeline, cands = build_all(tmp_path, "run")
    assert sum(c["split"] == "taskdev" for c in cands) == 4 * 10
    dev = [json.loads(line) for line in (ft / "taskdev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dev) == 4 * 8 and all(list(r) == ["id", "lang", "messages", "gold"] for r in dev)
    for lang in LANGS4:  # the same eight items in every language
        assert {r["id"].split(":", 1)[1] for r in dev if r["lang"] == lang} == {r["id"].split(":", 1)[1] for r in dev if r["lang"] == "en"}
    summary = json.loads((ft / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["final"]["taskdev"] == {lang: 8 for lang in LANGS4}
    val = run(
        VALIDATE,
        "--train", str(ft / "train.jsonl"),
        "--heldout", str(ft / "heldout.jsonl"),
        "--no-testset",
        "--pipeline", str(pipeline),
        "--langs", ",".join(LANGS4),
        "--require-taskdev",
    )
    assert val.returncode == 0, val.stdout + val.stderr
    assert "taskdev:  en 8, hi 8, ar 8, sw 8 (total 32)" in val.stdout


def test_a_rebuild_gives_byte_identical_train_heldout_and_taskdev(tmp_path):
    first, _, _ = build_all(tmp_path, "one")
    second, _, _ = build_all(tmp_path, "two")
    for name in ("train.jsonl", "heldout.jsonl", "taskdev.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_adding_taskdev_to_the_settings_does_not_change_train_or_heldout_bytes(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw, n=80, other=3)
    outputs = {}
    for name, extra in (("plain", {}), ("dev", {"taskdev_pool": 10, "taskdev_cap": 8})):
        pipeline = tmp_path / f"{name}.yaml"
        write_pipeline(pipeline, **extra)
        ft = tmp_path / name
        cands_path = ft / "prep" / "candidates.jsonl"
        assert run(BUILD, "prepare", "--pipeline", str(pipeline), "--raw", str(raw), "--out", str(cands_path)).returncode == 0
        cands = [json.loads(line) for line in cands_path.read_text(encoding="utf-8").splitlines()]
        curated = ft / "curated"
        curated.mkdir()
        lines = [{"messages": [{"role": "user", "content": c["instruction"]}, {"role": "assistant", "content": c["output"]}]} for c in cands]
        (curated / "sft_messages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")
        fin = run(BUILD, "finalize", "--pipeline", str(pipeline), "--curated", str(curated), "--candidates", str(cands_path), "--out", str(ft))
        assert fin.returncode == 0, fin.stderr
        outputs[name] = ft
    assert not (outputs["plain"] / "taskdev.jsonl").exists()
    for name in ("train.jsonl", "heldout.jsonl"):
        assert (outputs["plain"] / name).read_bytes() == (outputs["dev"] / name).read_bytes(), name


def test_the_built_taskdev_file_loads_through_the_orchestrators_own_readers_for_split_dev(tmp_path):
    from msd import evalrun

    ft, _, _ = build_all(tmp_path, "run")
    path = ft / evalrun.TASK_FILES["dev"]
    records = evalrun.read_task_records(path)
    assert len(records) == 4 * 8
    for lang in LANGS4:
        prompts = evalrun.task_prompts(records, lang, "dev", None, "taskdev.jsonl")
        assert len(prompts) == 8
        assert {p.kind for p in prompts} == {"task"} and {p.split for p in prompts} == {"dev"}
        assert all(p.gold in ("A", "B", "C", "D") and p.text.strip() for p in prompts)
        assert all(p.lang == lang for p in prompts)
    limited = evalrun.task_prompts(records, "en", "dev", 3, "taskdev.jsonl")
    assert [p.id for p in limited] == [r["id"] for r in records if r["lang"] == "en"][:3]


def test_the_real_pipeline_yaml_asks_for_two_hundred_taskdev_items_per_language():
    from msd import ftprep

    settings = ftprep.load_settings(Path(__file__).resolve().parent.parent / "pipelines" / "ft_data.yaml")
    assert settings["taskdev_cap"] == 200
    assert settings["taskdev_pool"] >= settings["taskdev_cap"]
