#!/usr/bin/env python
"""Run a model over the test-set prompts and store the completions (resumable).

    python scripts/generate.py --arm A0 --split dev --model-path CohereLabs/tiny-aya-global --limit 20
    python scripts/generate.py --arm A2 --alpha 0.25 --split report --model-path /path/to/merged

Writes <out-root>/<split>/<arm>/<alpha|none>/<lang>.gens.jsonl and <lang>.gens.meta.json. Run
`python scripts/generate.py --help` for every option and the exit codes. The logic lives in
`src/msd/gen.py` and `src/msd/genrun.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import genrun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return genrun.main(argv, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
