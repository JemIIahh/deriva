# Deriva: multilingual safety drift

Does harmless fine-tuning quietly make a small multilingual model less safe, and can the damage be repaired without retraining?

Deriva ("drift" in Spanish; the code package is called `msd`) is a study built to answer that for **Tiny Aya Global** in English, Hindi, Hinglish, Arabic and Swahili. It fine-tunes the model on a harmless task, measures how often it goes along with harmful requests and refuses harmless ones, then tries to repair any change with **SafeTune's ReSta** weight patch. Every number is reported per language, with its interval.

> **Status (2026-11-16): no result yet.** The design, the code and the protocol are done. No model has been fine-tuned or measured, and nothing is stated here as a finding. See [Status](#status).

## Links

| What | Where |
|---|---|
| Write-up (Hackathon submission) | [`article/ARTICLE.md`](article/ARTICLE.md), also as the card of [Jemiiah/deriva](https://huggingface.co/Jemiiah/deriva) on Hugging Face |
| Model collection | https://huggingface.co/collections/Jemiiah/deriva-models (no weights yet) |
| Dataset collection | https://huggingface.co/collections/Jemiiah/deriva-datasets (card only, no data yet) |
| Run the whole study | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |

## The three questions

1. **Drift.** Does a harmless LoRA fine-tune change how often Tiny Aya Global goes along with harmful requests?
2. **Unequal drift.** Is the change the same in every language, or larger in some?
3. **Repair.** Can ReSta bring safety back without retraining and without breaking the model?

## How it works

```
harmless exam questions
        |  CuratorKIT: filter, de-duplicate, sample
        v
 fine-tuning data
        |
Tiny Aya Global (A0, "original")
        |  AlignTune: LoRA fine-tune
        v
 drifted model (A1)
        |  SafeTune ReSta, at a strength chosen on a dev split only
        v
 repaired model (A2)        compared with a plain weight blend (A2n)

Every model is measured with the same code, per language, with intervals:
attack success rate  |  over-refusal  |  task accuracy
```

What keeps the numbers honest:

- The report split of the test set is **frozen with a hash before any model runs on it** (`docs/FREEZE_LOG.md`).
- Every choice, including the repair strength, is made on the dev split only.
- A **drift gate** decides whether there is anything to repair.
- A second judge from another model family measures how far the automatic judge can be trusted.

## Status

| Done | Not done |
|---|---|
| Design and pre-registered rules | Translation of the test set into Hindi, Hinglish, Arabic and Swahili |
| All code, with a test suite (4121 tests pass, 2026-11-16) | Freezing the test set |
| Source and licence list for every dataset, model and toolkit | Any GPU run: fine-tune, repair, judge |
| Fine-tuning data pipeline (CuratorKIT has run) | Results, figures, released models and data |
| English part of the test-set build | |

The two figures in `article/` are labelled placeholders, not results.

## Repository layout

| Path | What is in it |
|---|---|
| `src/msd/` | the code of the study, as one Python package |
| `scripts/` | the command-line stages, from `build_testset.py` to `export_public.py` and `check_article.py` |
| `configs/`, `pipelines/` | the fine-tune settings and the CuratorKIT pipeline |
| `kaggle/` | the environment check and the stage notebook for a Kaggle or Colab GPU |
| `notebooks/reproduce.ipynb` | rebuilds the published tables and figures on a CPU (no model, no key) |
| `templates/` | the dataset card and model card templates |
| `data/sources.yaml` | the source manifest, with every licence (the rest of `data/` is built, not stored in git) |
| `docs/` | the runbook, runtime notes, toolkit notes, freeze log and design |
| `tests/` | the unit tests |
| `article/` | the write-up and the list of what is filled in after the run |

## Quick start

The code needs Python 3.12 or newer. Each toolkit gets its own environment, because their version pins differ.

```bash
uv venv envs/core --python 3.12
uv pip install --python envs/core/bin/python -e ".[dev,blend]"   # "blend" adds the CPU torch that the blend tests use
envs/core/bin/python -m pytest                                   # the test suite
envs/core/bin/python scripts/check_article.py article/ARTICLE.md data/sources.yaml
```

The dependencies are not pinned yet, so a fresh install may get newer versions than the ones tested (UNVERIFIED that they all still pass).

AlignTune (the fine-tune) and SafeTune (the repair) are installed only on the GPU runtime, one session each. CuratorKIT has its own environment under `envs/`. Details are in `docs/runtime.md` and `docs/toolkit-notes.md`.

## Running the study

Follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md) from step 0 to step 13. Each step says what to run, where (laptop or GPU runtime), what it writes and what the next step reads. [`docs/runtime.md`](docs/runtime.md) covers the accounts and the free GPU runtimes (Kaggle or Colab). The design and build plan are in `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Licences

This is our reading of the licences, not legal advice.

- **This repository's own code:** no licence has been chosen yet, so by default all rights are reserved by the author. The licences of the tools and data it depends on are below.
- **Models.** Tiny Aya Global and Tiny Aya Base are released by Cohere Labs under CC-BY-NC-4.0 together with the Cohere Labs Acceptable Use Policy, and they are gated on Hugging Face. A model made from them here (the fine-tune, the repaired model) inherits those terms: non-commercial use, with attribution.
- **Toolkits.** AlignTune's own code is MIT, with a hard dependency on CuratorKIT, which is under the Lexsi Labs Source Available License 1.1 (LSAL-1.1): free for research, evaluation, education and audit, non-commercial only. SafeTune is under the Lexsi Labs Source Available License 1.2 (LSAL-1.2): noncommercial use, with the licence and its responsible-use conditions kept in every copy.
- **Data.** Every source of the test set is listed with its licence in `data/sources.yaml`. Hindi, Arabic and Swahili are machine-translated with NLLB-200 (weights CC-BY-NC-4.0) and Hinglish with a chat model; in the public export every translated prompt carries the licence CC-BY-NC-4.0.

## AI assistance

This work was built by the author together with Claude Code, an AI coding assistant from Anthropic, which wrote most of the code and the documentation.
