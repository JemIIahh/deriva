#!/usr/bin/env python
"""Build (and optionally freeze) the multilingual safety test set.

A thin command-line wrapper over `msd.testset`; see that module for what each step does.

    # English only (no translator, no model, no credentials needed)
    python scripts/build_testset.py --langs en

    # all five languages with the default translators (needs a model environment and an LLM endpoint)
    python scripts/build_testset.py --langs all --freeze

    # a dry run of the whole pipeline with no models at all
    python scripts/build_testset.py --translator identity --back-translator identity \\
        --same-meaning exact --out /tmp/scratch/testset

English is always built (splits are assigned on the English items); `--langs hi,sw` builds en, hi
and sw. Raw files are read from --raw/<entry>/ (`--print-fetch` lists the curl commands).

The build record (--summary, default <out>_build_summary.json) ties per-language runs together:
a run made from other inputs (seed, sources, sizes, raw files, English output) is refused, and the
real freeze checks the files against it. A directory that already holds MANIFEST.json is frozen:
building or freezing over it needs --refreeze. Dry or partial freezes (--dry-freeze) write
MANIFEST.dry.json and are checked with `verify_freeze.py --dry`.

Exit status: 0 ok; 1 dev/report leakage; 2 bad arguments, missing data, unavailable translator or
an incomplete set; 3 refused because MANIFEST.json already exists (use --refreeze); 4 refused or
failed the build-record / cross-language consistency check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import testset  # noqa: E402
from msd.config import LANGS, NON_ENGLISH  # noqa: E402
from msd.sources import load_sources  # noqa: E402
from msd.splits import LeakageError  # noqa: E402


def parse_assignments(values: list[str], defaults: dict[str, str], flag: str) -> dict[str, str]:
    """`NAME` applies to every non-English language, `LANG=NAME` to one; explicit beats bare.
    Raises ValueError (exit 2) for a language that is not a non-English study language."""
    result = dict(defaults)
    explicit = [v for v in values if "=" in v]
    for value in (v for v in values if "=" not in v):
        result.update({lang: value for lang in NON_ENGLISH})
    for value in explicit:
        lang, name = value.split("=", 1)
        if lang not in NON_ENGLISH:
            raise ValueError(f"{flag} {value!r}: language must be one of {list(NON_ENGLISH)}")
        result[lang] = name
    return result


def parse_langs(text: str) -> tuple[str, ...]:
    if text.strip() == "all":
        return LANGS
    langs = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [lang for lang in langs if lang not in LANGS]
    if unknown or not langs:
        raise ValueError(f"--langs must be 'all' or a comma list from {list(LANGS)}, got {text!r}")
    return langs


def print_summary(summary: dict) -> None:
    pool = summary["pool"]
    print(f"seed {summary['seed']}; sources: {', '.join(summary['sources_used'])}")
    print(f"pool: loaded {sum(pool['loaded'].values())}, blank {sum(pool['blank_removed'].values())}, "
          f"too long (> {pool['max_chars']} chars) {sum(pool['too_long_removed'].values())}, "
          f"duplicates removed {pool['duplicates_removed_total']}, kept {sum(pool['kept'].values())}")
    for source, kinds in summary["english"]["by_source"].items():
        print(f"  selected from {source}: {kinds}")
    for shortfall in summary["english"]["shortfalls"]:
        print(f"SHORTFALL {shortfall['kind']}/{shortfall['split']}: wanted {shortfall['wanted']}, got {shortfall['got']}")
    for lang, info in summary["languages"].items():
        print(f"{lang}: kept {info['kept_total']}, QC-dropped {info['dropped_total']}  kept={info['kept']}")


def print_frozen(result: "testset.FreezeResult") -> None:
    print(f"FROZE the test set: {result.path}")
    print(f"{result.path.name} sha256: {result.sha256}")
    print("append today's date and this sha256 to docs/FREEZE_LOG.md (this script does not write tracked files)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="all", help="'all' or a comma list (English is always built)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "sources.yaml")
    ap.add_argument("--summary", type=Path, default=None,
                    help="build record JSON (default: <out>_build_summary.json next to --out); merged across runs")
    ap.add_argument("--seed", type=int, default=testset.DEFAULT_SEED)
    ap.add_argument("--source", action="append", dest="sources", metavar="ENTRY",
                    help="manifest entry name to use (repeatable; default: the standard set)")
    ap.add_argument("--translator", action="append", default=[], metavar="[LANG=]NAME",
                    help=f"forward translator; known: {sorted(testset.TRANSLATORS)}")
    ap.add_argument("--back-translator", action="append", default=[], metavar="[LANG=]NAME",
                    help="back-translator (a different model family from the forward one)")
    ap.add_argument("--same-meaning", default=testset.DEFAULT_SAME_MEANING,
                    help=f"same-meaning check; known: {sorted(testset.SAME_MEANING)}")
    ap.add_argument("--freeze", action="store_true", help="after building, write MANIFEST.json (needs all 5 languages)")
    ap.add_argument("--freeze-only", action="store_true",
                    help="only write MANIFEST.json for what is in --out, after checking it against the build record")
    ap.add_argument("--refreeze", action="store_true",
                    help="allow building or freezing over a directory that already holds MANIFEST.json")
    ap.add_argument("--dry-freeze", action="store_true",
                    help="after building a PARTIAL set, write MANIFEST.dry.json for a dry check; not the real freeze")
    ap.add_argument("--print-fetch", action="store_true", help="print the curl commands for the raw files and exit")
    args = ap.parse_args(argv)

    source_names = tuple(args.sources) if args.sources else testset.DEFAULT_SOURCES
    summary_path = args.summary or args.out.with_name(args.out.name + "_build_summary.json")
    try:
        if args.print_fetch:
            print("\n".join(testset.fetch_commands(args.raw, source_names)))
            return 0
        if args.freeze_only:
            record_path = summary_path
            if not record_path.is_file():
                raise FileNotFoundError(f"build record {record_path} not found; build every language first")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            result = testset.freeze(
                args.out, summary=record, raw_root=args.raw, seed=args.seed, source_names=source_names,
                refreeze=args.refreeze,
            )
            print_frozen(result)
            return 0

        langs = parse_langs(args.langs)
        translators = parse_assignments(args.translator, testset.DEFAULT_TRANSLATORS, "--translator")
        back_translators = parse_assignments(args.back_translator, testset.DEFAULT_BACK_TRANSLATORS, "--back-translator")
        summary = testset.run_build(
            out_dir=args.out,
            raw_root=args.raw,
            manifest=load_sources(args.manifest),
            translator_names=translators,
            back_translator_names=back_translators,
            langs=langs,
            source_names=source_names,
            seed=args.seed,
            same_meaning_name=args.same_meaning,
            freeze=args.freeze,
            refreeze=args.refreeze,
            summary_path=summary_path,
        )
        print_summary(summary)
        if args.dry_freeze:
            result = testset.freeze(args.out, dry=True)
            print(f"DRY FREEZE of {len(result.manifest)} files into {result.path.name} "
                  f"(sha256 {result.sha256}): NOT the real freeze")
        elif args.freeze:
            print_frozen(testset.FreezeResult(
                manifest={}, path=args.out / summary["freeze"]["manifest"], sha256=summary["freeze"]["sha256"]))
        print(f"build record: {summary_path}")
        return 0
    except LeakageError as exc:
        print(f"error: leakage: {exc}", file=sys.stderr)
        return 1
    except testset.AlreadyFrozenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except testset.ConsistencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except (testset.ForbiddenSourceError, testset.UnknownNameError, testset.TranslatorConfigError,
            FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
