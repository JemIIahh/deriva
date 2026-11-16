# Deriva: multilingual safety drift

This project (public name Deriva; the code package is `msd`) is built to measure, language by language (English, Hindi, Hinglish, Arabic and Swahili), whether fine-tuning the
small multilingual model Tiny Aya Global on a harmless task changes how often it complies with harmful requests and
refuses harmless ones. It then tries to repair any change without retraining, with SafeTune's ReSta weight patch, and
reports every number with its interval.

No result is stated here: nothing has been measured yet. The results will be in the article (`article/ARTICLE.md`)
and in the public export once the runs are done.

## Install

The code needs Python 3.12 or newer. Each toolkit gets its own environment, because their version pins differ:

- `envs/core`: the project itself (`src/msd`, `scripts/`, the tests): `uv venv envs/core --python 3.12`, then
  `uv pip install --python envs/core/bin/python -e ".[dev,blend]"` (`blend` adds the CPU torch that the blend
  tests use). Run the tests with `envs/core/bin/python -m pytest`. The dependencies are not pinned yet, so a fresh
  install may get newer versions than the ones tested (UNVERIFIED that they all still pass).
- AlignTune (the fine-tune) and SafeTune (the repair) are installed only on the GPU runtime, one session each.
  CuratorKIT and AuditKIT have their own environments under `envs/`. Details: `docs/runtime.md` and
  `docs/toolkit-notes.md`.

To only reproduce the published table and figures from the export, see `notebooks/reproduce.ipynb` (CPU only, no
model, no key).

## Where to start

- `docs/RUNBOOK.md`: the whole study in order, from the environment check to the article, with every command.
- `docs/runtime.md`: the accounts, the private workspace repo and the free GPU runtimes (Kaggle or Colab).
- `docs/superpowers/specs/` and `docs/superpowers/plans/`: the design and the build plan.

## Licences

- **Models.** Tiny Aya Global and Tiny Aya Base are released by Cohere Labs under CC-BY-NC-4.0 together with the
  Cohere Labs Acceptable Use Policy, and they are gated on Hugging Face. A model made from them here (the fine-tune,
  the repaired model) inherits those terms: non-commercial use, with attribution.
- **Toolkits.** AlignTune's own code is MIT, with a hard dependency on CuratorKIT, which is under the Lexsi Labs
  Source Available License 1.1 (LSAL-1.1): free for research, evaluation, education and audit, non-commercial only.
  For SafeTune's licence, see the SafeTune repository.
- **Data.** Every source of the test set is listed with its licence in `data/sources.yaml`. Hindi, Arabic and Swahili
  are machine-translated with NLLB-200 (weights CC-BY-NC-4.0) and Hinglish with a chat model; in the public export
  every translated prompt carries the licence CC-BY-NC-4.0. Whether the translations are released is the owner's
  decision.
- **This repository's own code:** <<OWNER: choose a licence for the code before publishing it>>.

This is our reading of the licences, not legal advice.

## AI assistance

This work was built by the author together with Claude Code, an AI coding assistant from Anthropic, which helped
write and review the code and the documentation. <<OWNER: confirm this sentence and how much Claude Code did>>
