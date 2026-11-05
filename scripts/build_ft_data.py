#!/usr/bin/env python
"""Build the fine-tuning data around CuratorKIT: fetch, prepare, (CuratorKIT), finalize.

A thin command-line wrapper over `msd.ftprep`. The settings (harmless-subject allowlist, split
seed, pool sizes, caps) live in the pipeline YAML under `extra:`, the single place they are
recorded. The whole run, in order (from the repository root):

    python scripts/build_ft_data.py fetch                  # raw rows -> data/raw/<source>/
    python scripts/build_ft_data.py prepare                # -> data/ft/prep/candidates.jsonl
    envs/curatorkit/bin/curatorkit run pipelines/ft_data.yaml --output-dir data/ft/curated
    python scripts/build_ft_data.py finalize               # -> data/ft/train.jsonl, heldout.jsonl, taskdev.jsonl
    python scripts/validate_ft_data.py --langs en,hi,ar,sw

fetch needs the network (the public Hugging Face datasets-server, no token) and only reads the
allowlisted subjects of the test split. prepare and finalize run in the core venv; only the
CuratorKIT step uses envs/curatorkit. Hinglish has no source: its rows are produced later by
translating the Hindi rows, so it is not built here.

taskdev.jsonl (the dev split's own task items, drawn from the items train and heldout left
unused) is built by the same commands when the YAML settings hold taskdev_pool and taskdev_cap;
without them the outputs are exactly the train and heldout files, byte for byte as before.

Exit status: 0 ok; 2 bad arguments, missing data or a failed fetch.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import ftprep  # noqa: E402
from msd.testset import entry_dirname  # noqa: E402

PIPELINE = ROOT / "pipelines" / "ft_data.yaml"
RAW = ROOT / "data" / "raw" / entry_dirname(ftprep.SOURCE_NAME)
FT = ROOT / "data" / "ft"


def langs_arg(value: str) -> list[str]:
    langs = [v.strip() for v in value.split(",") if v.strip()]
    bad = [lang for lang in langs if lang not in ftprep.SOURCE_LANGS]
    if not langs or bad:
        raise argparse.ArgumentTypeError(
            f"languages must be a subset of {', '.join(ftprep.SOURCE_LANGS)} (hinglish has no source; got {value!r})"
        )
    return langs


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_fetch(args: argparse.Namespace) -> int:
    settings = ftprep.load_settings(args.pipeline)
    info: dict[str, object] = {}
    for lang in args.langs:
        rows = ftprep.fetch_language(lang, settings["harmless_subjects"], workers=args.workers)
        path = ftprep.raw_path(args.raw, lang)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{lang}: {len(rows)} allowlisted rows -> {path}")
    for lang in ftprep.SOURCE_LANGS:  # every raw file on disk, not only the ones fetched in this call
        path = ftprep.raw_path(args.raw, lang)
        if path.is_file():
            info[lang] = {"rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()), "sha256": ftprep.file_sha256(path)}
    revision = ftprep.dataset_revision()
    if revision is None:
        print("warning: could not read the dataset revision from the Hub API; FETCH.json records null")
    write_json(
        Path(args.raw) / "FETCH.json",
        {
            "source": ftprep.SOURCE_NAME,
            "dataset": ftprep.DATASET,
            "split": ftprep.SPLIT,
            "via": ftprep.HF_ROWS_URL,
            "fetched_on": datetime.date.today().isoformat(),
            "revision": revision,  # head commit seen at fetch time: {"revision", "last_modified"} or null
            "subjects": sorted(settings["harmless_subjects"]),
            "files": info,
        },
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    settings = ftprep.load_settings(args.pipeline)
    cands, summary = ftprep.prepare(
        args.raw,
        args.langs,
        settings["harmless_subjects"],
        seed=settings["split_seed"],
        heldout_pool=settings["heldout_pool"],
        train_pool=settings["train_pool"],
        taskdev_pool=settings.get("taskdev_pool", 0),
    )
    ftprep.write_candidates(args.out, cands)
    write_json(Path(args.out).with_name("prep_summary.json"), summary)
    print(f"{len(cands)} candidates -> {args.out}")
    for split, counts in summary["candidates"].items():
        print(f"  {split}: {counts}")
    for lang, reasons in summary["dropped"].items():
        print(f"  dropped {lang}: {reasons}")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    settings = ftprep.load_settings(args.pipeline)
    curated = Path(args.curated)
    candidates = ftprep.read_jsonl_dicts(args.candidates)
    exported = ftprep.read_jsonl_dicts(curated / "sft_messages.jsonl")
    rejected = ftprep.read_jsonl_dicts(curated / "rejected.jsonl") if (curated / "rejected.jsonl").is_file() else []
    summary = ftprep.finalize(
        candidates,
        exported,
        args.out,
        train_cap=settings["train_cap"],
        heldout_cap=settings["heldout_cap"],
        taskdev_cap=settings.get("taskdev_cap"),
        rejected=rejected,
    )
    manifest = curated / "manifest.json"
    if manifest.is_file():
        stage_counts = json.loads(manifest.read_text(encoding="utf-8")).get("stage_counts")
        summary["curatorkit_stage_counts"] = stage_counts
    prep_summary = Path(args.candidates).with_name("prep_summary.json")
    if prep_summary.is_file():
        summary["prepare"] = json.loads(prep_summary.read_text(encoding="utf-8"))
    write_json(Path(args.out) / "build_summary.json", summary)
    for key in ("candidates", "curatorkit_survivors", "removed_by_curatorkit", "capped_off", "final"):
        print(f"{key}: {summary[key]}")
    print(f"gate rejections: {summary['gate_rejections']}")
    written = ["train.jsonl", "heldout.jsonl", *([ftprep.TASKDEV_FILE] if "taskdev" in summary["final"] else [])]
    print("wrote " + ", ".join(str(Path(args.out) / name) for name in written))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--pipeline", type=Path, default=PIPELINE, help="pipeline YAML holding the settings (extra:)")

    p = sub.add_parser("fetch", help="download the allowlisted raw rows")
    common(p)
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--langs", type=langs_arg, default=list(ftprep.SOURCE_LANGS))
    p.add_argument("--workers", type=int, default=3, help="parallel page requests (the server rate-limits above a few)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("prepare", help="build closed-form candidates for CuratorKIT")
    common(p)
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--langs", type=langs_arg, default=list(ftprep.SOURCE_LANGS))
    p.add_argument("--out", type=Path, default=FT / "prep" / "candidates.jsonl")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("finalize", help="join CuratorKIT's export back to ids and write train/heldout")
    common(p)
    p.add_argument("--curated", type=Path, default=FT / "curated")
    p.add_argument("--candidates", type=Path, default=FT / "prep" / "candidates.jsonl")
    p.add_argument("--out", type=Path, default=FT)
    p.set_defaults(func=cmd_finalize)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
