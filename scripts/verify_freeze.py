#!/usr/bin/env python
"""Check data/testset against its MANIFEST.json; exit 0 if intact, 1 on any mismatch.

Fails on a changed byte, a missing file, a file under the directory that the manifest does not
list, or a row count (`n`) that no longer matches. A missing or unreadable manifest also fails.

    python scripts/verify_freeze.py [--dir data/testset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd.testset import MANIFEST_NAME, verify_freeze  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "testset")
    args = ap.parse_args(argv)
    problems = verify_freeze(args.dir)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        print(f"{len(problems)} problem(s): {args.dir} does not match {MANIFEST_NAME}", file=sys.stderr)
        return 1
    print(f"OK {args.dir} matches {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
