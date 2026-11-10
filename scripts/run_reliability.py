#!/usr/bin/env python
"""Run a second judge on a sample of the report verdicts and measure how well the two judges agree.

    python scripts/run_reliability.py --results results/ --judge NAME

Writes <results>/report/reliability/<judge>/<arm>/<alpha|none>/<lang>.verdicts.jsonl, prints Cohen's kappa with an
interval per language, and names the languages whose agreement is too low or too thinly measured to trust. Run
`python scripts/run_reliability.py --help` for every option and the exit codes. The logic lives in
`src/msd/reliabilityrun.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import reliabilityrun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return reliabilityrun.main(argv, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
