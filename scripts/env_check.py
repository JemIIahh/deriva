#!/usr/bin/env python
"""Print a JSON report of the runtime: Python, GPU, torch/CUDA, toolkit versions, Tiny Aya access.

    python scripts/env_check.py            # includes a HEAD check of both Tiny Aya repos on Hugging Face
    python scripts/env_check.py --offline  # no network call at all

Run it on the laptop, on Kaggle and on Colab and compare. A missing package shows as null.
HF_TOKEN is read from the environment or from .env and is used only for the HEAD checks; the report
says whether one was found (`hf_token_present`) and never contains it. The logic lives in
`src/msd/envcheck.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import envcheck  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    return envcheck.main(argv, env_file=ROOT / ".env")


if __name__ == "__main__":
    main()
