#!/usr/bin/env python
"""Check data/testset against its MANIFEST.json; exit 0 if intact, 1 on any mismatch.

Fails unless all ten `<lang>/{dev,report}.jsonl` files (five languages) are listed in MANIFEST.json,
exist, match their sha256 and row count `n`, and no other file is present. A missing or unreadable
manifest also fails.

    python scripts/verify_freeze.py [--dir data/testset]
    python scripts/verify_freeze.py --dry [--dir data/testset]   # check MANIFEST.dry.json instead

`--dry` checks the manifest written by a dry or partial freeze: it still catches a changed, missing
or extra file, but makes no completeness claim and is never a substitute for the real check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd.testset import DRY_MANIFEST_NAME, MANIFEST_NAME, verify_freeze  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--dry", action="store_true", help=f"check {DRY_MANIFEST_NAME} (a dry or partial freeze)")
    args = ap.parse_args(argv)
    manifest_name = DRY_MANIFEST_NAME if args.dry else MANIFEST_NAME
    problems = verify_freeze(args.dir, dry=args.dry)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        print(f"{len(problems)} problem(s): {args.dir} does not match {manifest_name}", file=sys.stderr)
        return 1
    print(f"OK {args.dir} matches {manifest_name}" + (" (dry check only)" if args.dry else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
