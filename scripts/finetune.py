#!/usr/bin/env python
"""Fine-tune A0 into A1 with AlignTune (LoRA on a harmless closed-form task, merged to fp16).

    python scripts/finetune.py --config configs/ft_lora.yaml --dry-run
    python scripts/finetune.py --config configs/ft_lora.yaml --max-steps 20 --merge
    python scripts/finetune.py --config configs/ft_lora.yaml

Run `python scripts/finetune.py --help` for every option, the exit codes and what is still UNVERIFIED. The logic
lives in `src/msd/finetune.py`; this file is the thin command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msd import finetune  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return finetune.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
