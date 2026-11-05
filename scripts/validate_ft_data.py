#!/usr/bin/env python
"""Validate the fine-tuning data (train.jsonl and heldout.jsonl).

A thin command-line wrapper over `msd.ftdata`; see that module for every rule.

    python scripts/validate_ft_data.py --langs en,hi,ar,sw

--langs names the languages the data must cover (default: all five study languages). A study
language that is not listed and has no rows (today: hinglish, whose rows need a translation
endpoint) is reported as "not covered" and does not fail the run; a listed language with too few
rows does. The safety dev and report splits are read from every language directory found under
--testset, and the output says which languages that was and how many dev and report prompts each
has; a covered language whose files are empty fails (a check against nothing proves nothing).

The harmless-subject allowlist (extra.harmless_subjects) is checked by default, read from the
repository's pipelines/ft_data.yaml. --pipeline points at another file; --no-subject-check turns
the rule off and the output then says "subject allowlist: NOT checked". The output always has an
allowlist status line (which subjects and how many rows were checked, or NOT checked).

Exit status: 0 ok; 1 several rule classes failed; 2 bad arguments or unreadable input;
10 format, 11 refusal, 12 answer not bare / not gold, 13 overlap (train, heldout or safety
splits), 14 language imbalance, 15 wrong script for the language code, 16 expected language
missing, 17 subject not on the allowlist, 18 a covered safety language has no dev or no report
prompts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import ftdata  # noqa: E402
from msd.config import LANGS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "ft" / "train.jsonl")
    ap.add_argument("--heldout", type=Path, default=ROOT / "data" / "ft" / "heldout.jsonl")
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset", help="safety test-set directory")
    ap.add_argument("--no-testset", action="store_true", help="skip the safety-split overlap check (says so in the output)")
    ap.add_argument("--langs", default=",".join(LANGS), help="comma-separated languages the data must cover")
    pipeline = ap.add_mutually_exclusive_group()
    pipeline.add_argument(
        "--pipeline",
        type=Path,
        default=ROOT / "pipelines" / "ft_data.yaml",
        help="pipeline YAML holding extra.harmless_subjects (default: %(default)s)",
    )
    pipeline.add_argument(
        "--no-subject-check",
        action="store_true",
        help='skip the harmless-subject allowlist (the output then says "subject allowlist: NOT checked")',
    )
    ap.add_argument("--tolerance", type=float, default=ftdata.IMBALANCE_TOLERANCE, help="allowed share off the mean count")
    args = ap.parse_args(argv)

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    bad = [lang for lang in langs if lang not in LANGS]
    if not langs or bad:
        ap.error(f"--langs must be a comma-separated subset of {', '.join(LANGS)} (got {args.langs!r})")

    try:
        train, train_problems = ftdata.read_jsonl_records(args.train, "train")
        heldout, held_problems = ftdata.read_jsonl_records(args.heldout, "heldout")
        subjects = None if args.no_subject_check else ftdata.load_allowed_subjects(args.pipeline)
        if args.no_testset:
            dev, report_texts, safety_counts = [], [], None
        else:
            dev, report_texts, safety_counts = ftdata.load_safety_texts(args.testset)
    except (OSError, ValueError) as exc:  # missing file, unreadable file, bad allowlist
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = ftdata.validate(
        train,
        heldout,
        expected_langs=langs,
        safety_dev=dev,
        safety_report=report_texts,
        allowed_subjects=subjects,
        safety_counts=safety_counts,
        imbalance_tolerance=args.tolerance,
        read_problems=[*train_problems, *held_problems],
    )
    print(ftdata.render(report))
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
