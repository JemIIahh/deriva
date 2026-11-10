#!/usr/bin/env python
"""Build the report tables and figures from a results folder (reads only; deterministic).

    python scripts/make_report.py --verdicts results/ --out results/
    python scripts/make_report.py --verdicts results_paid/ --dev results/ --out results_paid/   # dev split elsewhere

Writes tables.csv, heatmap.png, alpha_curve.png and reliability.json (judge ids, kappa per language, the
low-confidence languages, the excluded cells with the sanity check each failed, the alpha chosen on dev). Run
`python scripts/make_report.py --help` for the rules and the exit codes. The logic lives in `src/msd/reportrun.py`;
this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import reportrun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return reportrun.main(argv, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
