#!/usr/bin/env python
"""Run the drift gate on the dev verdicts of A0 and A1 and write results/gate.json.

    python scripts/run_gate.py
    python scripts/run_gate.py --results results/ --langs hi,ar,sw --strong-tried

Reads <results>/dev/{A0,A1}/none/<lang>.verdicts.jsonl (dev only, there is no way to point it at the report split)
and writes <results>/gate.json with the decision and its branch. Run `python scripts/run_gate.py --help` for every
option and the exit codes. The logic lives in `src/msd/gaterun.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import gaterun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return gaterun.main(argv, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
