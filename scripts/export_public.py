#!/usr/bin/env python
"""Build the public export as a dry run into a folder (nothing is sent anywhere).

    python scripts/export_public.py --out export/
    python scripts/export_public.py --results results/ --testset data/testset/ --sources data/sources.yaml --out export/

Writes the prompts, the public result rows, the aggregates, the model and dataset cards and a manifest, all under
--out, which must be new or empty. Run `python scripts/export_public.py --help` for the layout, the rules and the exit
codes. The logic lives in `src/msd/export.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import export  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return export.main(argv, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
