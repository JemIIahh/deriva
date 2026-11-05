#!/usr/bin/env python
"""Repair the drifted model A1 with SafeTune's ReSta patch: theta_safe = A1 + alpha * (A0 - BASE).

    python scripts/run_resta.py --drifted A1_DIR --base CohereLabs/tiny-aya-base \\
        --aligned CohereLabs/tiny-aya-global --alpha 0.25 --out A2_alpha_0.25

Run `python scripts/run_resta.py --help` for every option, the exit codes and what is still UNVERIFIED. The logic
lives in `src/msd/restarun.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import restarun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return restarun.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
