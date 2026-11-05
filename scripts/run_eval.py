#!/usr/bin/env python
"""Generate, judge and score one (arm, alpha) across the study languages (resumable, stage by stage).

    python scripts/run_eval.py --arm A1 --split dev --model PATH --judge NAME --out results/
    python scripts/run_eval.py --arm A2 --alpha 0.25 --split report --model PATH --judge NAME
    python scripts/run_eval.py --arm A1 --split report --judge NAME --stages judge,task   # no model needed

Writes <out>/<split>/<arm>/<alpha|none>/<lang>.{gens,verdicts,tasks}.jsonl and <lang>.eval.json. Run
`python scripts/run_eval.py --help` for every option and the exit codes. The logic lives in `src/msd/evalrun.py`;
this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import evalrun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return evalrun.main(argv, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
