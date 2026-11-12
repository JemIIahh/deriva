# Runbook: the first real run, step by step

This page is the order of the whole study, from an empty GPU runtime to a submitted article. Each step says what to
run, where, what it writes, what the next step reads, and what to do when it fails. Nothing on this page has run on a
GPU yet (written 2026-11-12). The code behind every command is tested on the Mac with fakes only.

"The design doc" is `docs/superpowers/specs/2026-11-03-multilingual-safety-drift-design.md`.

Labels: **UNVERIFIED** = not confirmed from a page or a run (the same use as `docs/runtime.md`).
**`<<OWNER: ...>>`** = a value or a decision only the owner has. **Owner's yes** = an install or an action outside
this folder that needs the owner to agree first.

## How to read this page

- **Two places run things.** "Mac" means a terminal in the repository folder, with `envs/core/bin/python`.
  "Runtime" means a Kaggle or Colab GPU notebook running `kaggle/10_run_stage.ipynb`. The same notebook works on both.
  Kaggle or Colab is the owner's choice (not answered yet): the steps are the same.
- **One results root: `results/`.** Every script reads and writes `results/` by default. Do not pass `--out` to
  `run_eval.py` or `--results` to the other scripts in this runbook (only the two smoke runs use a scratch folder,
  on purpose). The runtime pulls `results/` from the private workspace repo at its start and uploads it after each
  step. The Mac downloads it once, before step 9.
- **One session at a time.** Two sessions that write the same files at once can damage them (the code assumes one
  writer per file). Finish or stop one runtime session before you start the next.
- **Nothing is published by any step before step 13.** Results go only to the PRIVATE workspace repo. A1 goes only to
  a PRIVATE model repo. The A2 models are not uploaded at all.

## The stage notebook in one minute (Kaggle or Colab)

1. Open `kaggle/10_run_stage.ipynb` on the runtime. Kaggle: create or import a notebook from the file, then in it:
   Add-ons > Secrets, add `HF_TOKEN`; switch Internet on; pick a GPU (`docs/runtime.md`, section 1, step 5; menu
   wording UNVERIFIED). Colab: File > Upload notebook; Runtime > Change runtime type > GPU; the key icon (Secrets) >
   add `HF_TOKEN` with Notebook access on (menu wording UNVERIFIED).
2. Add the secret `ANTHROPIC_API_KEY` only when a step uses an `anthropic:` judge. It is optional. The notebook says
   "ANTHROPIC_API_KEY not set" when it is absent, and that is fine for every other step.
3. In the settings cell set `WORKSPACE_REPO` once. For each step of this runbook set either `STAGE` and `STAGE_ARGS`
   (one script), or `STEPS` (several scripts in order, in one session). Set `EXTRA_PIP` and `A1_REPO` only when the
   step says so. Then Run all.
4. What the notebook does: it stops unless the workspace repo is private; installs the wheel; pulls `scripts/`,
   `pipelines/`, `configs/`, `data/` and `results/`; downloads A1 into the folder `A1/` when `A1_REPO` is set; runs
   the steps; after each step that changed `results/`, it uploads the whole `results/` folder (never weights,
   `work/`, crash leftovers or `.env`).
5. When a step fails, the notebook stops with "step N (script) stopped with exit code X". The steps after it do not
   run. Nothing the failed step wrote is uploaded. Read the script's own message above that line. To keep what it
   wrote anyway (for example half-finished completions to resume from), run `upload_results({})` in a new cell.
6. A long step can be lost with the session. Split it by language (`--langs hi`, then `--langs ar`, ...) as
   separate entries of `STEPS`: each one is uploaded when it ends. Running the same command again resumes: the
   generation skips finished prompts (with the same settings), and the judge reuses its saved labels.
7. The weekly GPU quota, the session length, the GPU memory (T4), the RAM and the disk of the free runtimes are
   UNVERIFIED (`docs/runtime.md`, section 3). Colab's FAQ says at most 12 hours per session and nothing guaranteed.

The values below use these placeholders. Replace them everywhere:

| Placeholder | What it is |
|---|---|
| `<hf-user>/msd-workspace` | the PRIVATE Hugging Face dataset repo (step 1) |
| `<hf-user>/<a1-repo>` | the PRIVATE Hugging Face model repo for A1 (step 1) |
| `<judge>` | the stand-in judge: `hf:Qwen/Qwen3-4B` (`docs/toolkit-notes.md`, "Judge candidates"). Use `hf:google/gemma-4-E4B-it` instead if the test-set translator of step 2 is a Qwen model (the judge should not share a family with the model that wrote the prompts). Whether Gemma fits one T4 is UNVERIFIED. |
| `<paid-judge>` | `anthropic:<<OWNER: Claude model id>>`, only once the key exists (it comes LAST) |

## Step 0. Environment check

- **Mac:** `envs/core/bin/python scripts/env_check.py --offline` prints the laptop's environment as JSON and writes
  nothing. Without `--offline` it also checks access to both Tiny Aya repos (needs `HF_TOKEN` in the shell).
- **Runtime:** run `kaggle/00_env_check.ipynb` (it saves `env_check.json` in the working folder), or
  `kaggle/10_run_stage.ipynb` with its defaults (`STAGE = "env_check"`, which uploads nothing).
- **Next:** copy the JSON into the table of `docs/runtime.md`, section 5, and pick the runtime by the rule there.
- **If it fails:** "HF_TOKEN not found" means the secret is missing. `401 unauthorized` with a token means a wrong
  token or a licence not accepted yet (`docs/runtime.md`, section 5).

## Step 1. The private workspace repo, the A1 model repo and the wheel (owner, once)

The account steps are in `docs/runtime.md`, section 1 (licence, token, `.env`, Kaggle or Colab, notebook secrets).
Then, on the Mac:

1. **Owner's yes:** install the Hugging Face client into the project environment (it is not there now):
   `uv pip install --python envs/core/bin/python huggingface_hub`
2. Build the wheel: `uv build --wheel --out-dir dist .` It holds `src/msd` only (version 0.1.0, so the file is
   always `dist/msd-0.1.0-py3-none-any.whl`).
3. **Owner's yes:** create the private dataset repo and upload the wheel and the code. This is the command of
   `docs/runtime.md`, section 5, with `configs/` added and a privacy check before the uploads (call signatures
   UNVERIFIED, as there):

   ```
   set -a; . ./.env; set +a
   envs/core/bin/python -c "from huggingface_hub import HfApi; a=HfApi(); r='<hf-user>/msd-workspace'; \
   a.create_repo(r, repo_type='dataset', private=True, exist_ok=True); \
   assert a.repo_info(r, repo_type='dataset').private is True, 'the workspace repo must be private'; \
   a.upload_file(path_or_fileobj='dist/msd-0.1.0-py3-none-any.whl', path_in_repo='wheels/msd-0.1.0-py3-none-any.whl', repo_id=r, repo_type='dataset'); \
   [a.upload_folder(folder_path=d, path_in_repo=d, repo_id=r, repo_type='dataset', ignore_patterns=['__pycache__', '*.pyc']) for d in ('scripts', 'pipelines', 'configs')]"
   ```

4. **Owner, by hand on the Hugging Face website:** create a PRIVATE model repo for A1, `<hf-user>/<a1-repo>`. The
   fine-tune never creates a repo; it only uploads to one that exists and is private.
5. The data goes up after step 2 (the test set must be frozen first).

- **Writes:** the two private repos. **Next reads:** the stage notebook pulls the wheel, `scripts/`, `pipelines/`,
  `configs/` and `data/` from the workspace repo.
- **Whenever `src/msd`, `scripts/`, `pipelines/` or `configs/` change:** rebuild the wheel and run the upload again.
  The wheel keeps its name, so the new one replaces the old one (keep exactly one wheel there). A stale copy means
  the runtime runs old code.

## Step 2. Build the five-language test set and freeze it (Mac)

Today `data/testset/` holds English only (`en/dev.jsonl`, `en/report.jsonl` and a dry manifest). This step builds
Hindi, Hinglish, Arabic and Swahili from it and freezes all five. **This is the one step that needs a chat model.**

What the code needs (read from `src/msd/testset.py`):

- Forward translation of `hi`, `ar` and `sw`: the open model `facebook/nllb-200-distilled-600M` through `torch`,
  `transformers` and `sentencepiece` (weights CC-BY-NC-4.0). `envs/core` has torch but not the other two.
- Forward translation of `hinglish`, AND the back-translation check of all four non-English languages: the
  translator `llm-chat`, an OpenAI-compatible chat-completions endpoint set by three variables in `.env`:
  `MSD_TRANSLATE_BASE_URL`, `MSD_TRANSLATE_MODEL`, `MSD_TRANSLATE_API_KEY` (see `.env.example`). Which model and
  service is the owner's choice: `<<OWNER: chat model and endpoint for the translation>>` (UNVERIFIED which). The
  design asks for a non-Claude translator (design doc, section 6, "Freeze order and the key"), which also avoids a
  model that refuses to translate harmful prompts. The article names it (`article/FILL_IN.md`).

Commands:

1. **Owner's yes:** a separate environment for the translation, so the test environment stays as it is (package
   versions UNVERIFIED; how fast NLLB runs on the Mac is UNVERIFIED):

   ```
   uv venv envs/translate --python 3.12
   uv pip install --python envs/translate/bin/python pyyaml torch transformers sentencepiece
   ```

2. Fill the three `MSD_TRANSLATE_*` lines in `.env`, then load `.env` into the shell (it prints nothing):
   `set -a; . ./.env; set +a`
3. Build and freeze: `envs/translate/bin/python scripts/build_testset.py --langs all --freeze`
   It prints, per language, how many prompts were kept and how many the translation check dropped, any `SHORTFALL`
   line, then `FROZE the test set: ...` and `MANIFEST.json sha256: <hash>`.
4. Check the freeze: `envs/core/bin/python scripts/verify_freeze.py` must print `OK ... matches MANIFEST.json`.
5. Record it: add a row to `docs/FREEZE_LOG.md` (the procedure is there) and commit that file.
6. Check the fine-tune data against the full test set (the overlap check now covers all five languages):
   `envs/core/bin/python scripts/finetune.py --config configs/ft_lora.yaml --dry-run`
7. Upload the frozen test set and the fine-tune data to the workspace repo (call signatures UNVERIFIED):

   ```
   envs/core/bin/python -c "from huggingface_hub import HfApi; a=HfApi(); r='<hf-user>/msd-workspace'; \
   assert a.repo_info(r, repo_type='dataset').private is True, 'the workspace repo must be private'; \
   a.upload_folder(folder_path='data/testset', path_in_repo='data/testset', repo_id=r, repo_type='dataset'); \
   a.upload_folder(folder_path='data/ft', path_in_repo='data/ft', repo_id=r, repo_type='dataset', allow_patterns=['train.jsonl', 'heldout.jsonl', 'taskdev.jsonl'])"
   ```

- **Writes:** `data/testset/<lang>/{dev,report}.jsonl` for en, hi, hinglish, ar and sw, `data/testset/MANIFEST.json`,
  and the build record `data/testset_build_summary.json` (all git-ignored; step 10 needs the build record).
- **Next reads:** every `run_eval.py` step reads `data/testset/`. The report split is generated ONLY when
  `data/testset/MANIFEST.json` exists and verifies; otherwise `run_eval.py` stops before generating and says to run
  this step. Each report-split result records the manifest's sha256 (`testset_manifest_sha256`).
- **If it fails** (exit codes of `build_testset.py`):
  - 2 with "translator 'llm-chat' is not configured": a `MSD_TRANSLATE_*` variable is empty or `.env` was not
    loaded into the shell. 2 with "needs the 'transformers' and 'torch' packages": you used the wrong environment.
    2 with "translation endpoint returned HTTP ...": the endpoint, model name or key is wrong.
  - To build one language at a time: `--langs hi` (builds en and hi), then `--langs hinglish`, and so on, without
    `--freeze`. The build record ties the runs together. Then freeze what is built:
    `envs/core/bin/python scripts/build_testset.py --freeze-only`
  - 3: the set is already frozen. Do not refreeze after any report-split result exists. If you must, use
    `--refreeze`, add a row to the freeze log that says why, and run every report-split cell again.
  - 4: the inputs changed since an earlier run (seed, sources, raw files or the English output); the message says
    which. 1: a dev prompt and a report prompt are the same text (leakage).

## Step 3. Fine-tune A1 (runtime)

Everything about this step, with the reasons, is in `docs/runtime.md`, "Fine-tune A1 with AlignTune". In short, as
separate runs of the stage notebook with `STAGE = "finetune"` and `EXTRA_PIP = "aligntune==0.1.31"`:

| Run | `STAGE_ARGS` | Writes |
|---|---|---|
| plan | `--config configs/ft_lora.yaml --dry-run` | nothing |
| smoke | `--config configs/ft_lora.yaml --max-steps 20 --merge` | `results/finetune/ft_lora-smoke/` |
| real | `--config configs/ft_lora.yaml --push-to <hf-user>/<a1-repo>` | `results/finetune/ft_lora/`, and the merged A1 in `<hf-user>/<a1-repo>` |

Read `label_check` in the smoke run's `finetune_summary.json` before the real run (`docs/runtime.md`, "Reading
label_check"). The strong recipe (`configs/ft_lora_strong.yaml`) runs only if the gate asks for it (step 5).

- **Next reads:** `<hf-user>/<a1-repo>` is A1 from now on: the `--model` of every A1 evaluation and the `A1_REPO` of
  the ReSta sessions. The summary in `results/finetune/ft_lora/finetune_summary.json` is uploaded (the weights are not).
- **Option:** the real run and the A1 dev generation of step 4 can share one session, which saves downloading A1:
  `STEPS = [("finetune", "--config configs/ft_lora.yaml --push-to <hf-user>/<a1-repo>"), ("run_eval", "--arm A1 --split dev --model results/finetune/ft_lora/merged --stages generate,task")]`
  The merged folder exists only in that session. If that generation has to be finished in a later session, use
  `--model <hf-user>/<a1-repo> --force-new-meta` there: the same weights under another name, and the meta keeps both.
- **If it fails:** `docs/runtime.md`, "After a failure". One more case: a later session pulls `results/finetune/`, so
  a second smoke run with the same `--out` is refused as "not empty". Give it a new folder, for example
  `--out results/finetune/ft_lora-smoke-2`.

## Step 4. Dev split: A0 and A1 (runtime)

The gate (step 5) and the alpha choice (step 9) both need A0 and A1 on the dev split, generated AND judged, in all
five languages. Generate and judge as separate steps, so the model and the judge never share the GPU memory.

**4a. Smoke check of the judge, 10 items, in a scratch folder** (the folder `smoke` is not under `results/`, so it
is never uploaded and never mixes with real results):

```
STEPS = [("run_eval", "--arm A0 --split dev --model CohereLabs/tiny-aya-global --langs hi --limit 10 --stages generate,task --out smoke"),
         ("run_eval", "--arm A0 --split dev --judge <judge> --langs hi --limit 10 --stages judge --out smoke")]
```

Read the printed counts: `judge_error` should be 0 or close to it. Many judge errors mean the judge does not work
in this setup: fix that before step 4b.

**4b. The real dev cells:**

```
STEPS = [("run_eval", "--arm A0 --split dev --model CohereLabs/tiny-aya-global --stages generate,task"),
         ("run_eval", "--arm A0 --split dev --judge <judge> --stages judge"),
         ("run_eval", "--arm A1 --split dev --model <hf-user>/<a1-repo> --stages generate,task"),
         ("run_eval", "--arm A1 --split dev --judge <judge> --stages judge")]
```

- **Writes:** `results/dev/A0/none/` and `results/dev/A1/none/`, per language: `<lang>.gens.jsonl` (full
  completions), `<lang>.gens.meta.json`, `<lang>.verdicts.jsonl`, `<lang>.tasks.jsonl` (en, hi, ar, sw; Hinglish has
  no task items and says `task_skipped`), `<lang>.eval.json`; and the judge's label cache `results/dev/_cache/`.
- **Read:** the `tasks` block of each `<lang>.eval.json`. A1 should answer the bare letter more often than A0
  (`docs/runtime.md`, "Held-out accuracy check").
- **Next reads:** step 5 reads the verdicts and eval.json files of both arms; step 9 reads all of it.
- **If it fails** (exit codes of `run_eval.py`): 5 = the judge failed on more than 10% of a language or kind; the
  verdicts were written for inspection but must not be used: fix the judge, then run the judge step again.
  4 = verdicts of another judge are already there (one judge per split): `--rejudge` replaces them and moves the old
  ones to `_replaced/`, never deletes them. 1 with "out of memory" on the first batch: run again with a smaller
  `--batch-size` (default 32). If rows were already written, a different `--batch-size` also needs
  `--force-new-meta`, and the meta then records both settings. 3 = no test-set file for any language (step 2 or the
  data upload is missing).

## Step 5. The drift gate (runtime or Mac)

`STAGE = "run_gate"`, `STAGE_ARGS = ""` (on the Mac: `envs/core/bin/python scripts/run_gate.py`, after downloading
`results/` as in step 9). It reads `results/dev/A0/none/` and `results/dev/A1/none/` for hi, hinglish, ar and sw and
writes `results/gate.json`. Exit 0 means the gate was evaluated, whether it passed or not. Exit 1 is a data problem
(the message names the file); then `gate.json` is not there.

Read `branch` in `results/gate.json`:

- `continue`: go to step 6.
- `stronger_recipe`: the fine-tune did not move safety enough. Run the strong recipe (step 3 with
  `--config configs/ft_lora_strong.yaml --push-to <hf-user>/<a1-strong-repo>`, a second private model repo made by
  hand). The old A1 dev cells must go first, or `run_eval.py` refuses to mix two models in one file: remove
  `results/dev/A1` from the workspace repo (call UNVERIFIED:
  `envs/core/bin/python -c "from huggingface_hub import HfApi; HfApi().delete_folder('results/dev/A1', repo_id='<hf-user>/msd-workspace', repo_type='dataset')"`).
  Then run the A1 lines of step 4b with the strong repo as `--model`, and the gate again with
  `STAGE_ARGS = "--strong-tried"`. From then on the strong repo is A1.
- `pivot_prevention` (the strong recipe was tried and the gate still failed): stop and tell Claude. The design's
  answer is the prevention arm A3 (SafeTune Harden), which was skipped by a ruling and is not built (design doc,
  section 8, risk 1).

The gate.json in the workspace repo is the last one that succeeded: a failed run is not uploaded.

## Step 6. ReSta per alpha, dev and report (runtime)

ReSta repairs A1 with the safety vector of Tiny Aya: `A1 + alpha * (Global - Base)`. One notebook run per alpha:
the A2 model it makes lives only on that runtime's disk, so its dev and report evaluations run in the same session
(`STEPS`). Settings for every ReSta session:

- `EXTRA_PIP = "safetune==0.1.12"` (the version the code was written against, `docs/toolkit-notes.md`; whether it
  installs on the runtime is UNVERIFIED). A fresh session: never in the same session as the fine-tune's AlignTune.
- `A1_REPO = "<hf-user>/<a1-repo>"`: `run_resta.py --drifted` needs A1 as a local folder; the notebook downloads it
  into `A1/`.
- ReSta loads three models on the CPU: about 20 GB of RAM in bf16 (`docs/toolkit-notes.md`, "Memory and device").
  Whether a free runtime has that much is UNVERIFIED. The disk must hold A1, the two Tiny Aya models and one A2
  (each about 6.7 GB): UNVERIFIED.

Order of the alphas (`docs/toolkit-notes.md`, "The alpha grid"): 0 first (the no-op check), then 0.1, 0.25 and 0.5;
then 0.05, 0.15 and 0.35 only if the curve has a gap; 1.0 last, as the break control. The `--out` folder and the
`--alpha` folder name use the same spelling: 0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0.

The session for alpha 0.1 (change all four 0.1 for another alpha):

```
STEPS = [("run_resta", "--drifted A1 --base CohereLabs/tiny-aya-base --aligned CohereLabs/tiny-aya-global --alpha 0.1 --out results/resta/A2_alpha_0.1"),
         ("run_eval", "--arm A2 --alpha 0.1 --split dev --model results/resta/A2_alpha_0.1 --stages generate,task"),
         ("run_eval", "--arm A2 --alpha 0.1 --split report --model results/resta/A2_alpha_0.1 --stages generate,task"),
         ("run_eval", "--arm A2 --alpha 0.1 --split dev --judge <judge> --stages judge"),
         ("run_eval", "--arm A2 --alpha 0.1 --split report --judge <judge> --stages judge")]
```

- **Writes:** `results/resta/A2_alpha_0.1/` (the patched model; its `resta_patch.json` records alpha and the DARE
  settings and is uploaded, the weights are not), `results/dev/A2/0.1/` and `results/report/A2/0.1/` (the same files
  as in step 4).
- **The alpha 0 check.** After the alpha 0.0 session, the dev completions of A2 at 0.0 should equal A1's, because
  alpha 0 adds nothing (UNVERIFIED until it runs). Count the equal ones for a language, for example Hindi, on the Mac
  after step 9's download (in a notebook cell, write `!python` instead of `envs/core/bin/python`):
  `envs/core/bin/python -c "import json; f=lambda p: {r['prompt_id']: r['text'] for r in map(json.loads, open(p))}; a=f('results/dev/A1/none/hi.gens.jsonl'); b=f('results/dev/A2/0.0/hi.gens.jsonl'); print(len(a), 'A1 rows,', sum(a[k] == b.get(k) for k in a), 'equal')"`
  If many differ, stop and tell Claude before any other alpha (`docs/toolkit-notes.md`, "Things that could make
  alpha = 0 differ").
- **Read each alpha's answers:** SafeTune reports that a too-large alpha breaks the model while it still refuses.
  Read some benign answers in `results/dev/A2/<alpha>/<lang>.gens.jsonl` (step 9's sanity gate also checks this).
- **Next reads:** step 9 needs the dev cells of every alpha that was run, in every language, and the report cells
  for the table and the alpha curve. An alpha that was not run is simply not a candidate.
- **If it fails:** `run_resta.py` exit 1 with "SafeTune" in the message: the package did not install or import
  (fallback: `docs/runtime.md`, section 4). Killed with no message, or "out of memory": not enough RAM (UNVERIFIED
  which runtime has enough). Exit 2: `--out` is not empty (a pulled `results/resta/A2_alpha_<a>/` from an earlier
  session counts): use a new `--out`, for example `results/resta/A2_alpha_0.1-b`, and the same folder in the
  `--model` of that session. A failed `run_eval.py` step: as in step 4.
- **Keeping A2 at the chosen alpha.** No step uploads A2. If the owner wants to keep or later release A2 at the alpha
  chosen in step 9, re-run ReSta at that alpha in a new session (the DARE seed is fixed, so it should give the same
  weights; bit for bit is UNVERIFIED) with `STAGE = "run_resta"` and `--out work/A2_release`, then, in a new cell,
  upload `work/A2_release` to a PRIVATE model repo made by hand. Making it public is step 13, the owner's decision.

**Optional: the naive blends (arm A2n).** They need A1 and A0 as local folders, no GPU and no SafeTune
(`msd.blend`; blend values t = 0.25, 0.5, 0.75). They are optional and go early if time is short (cut list
below). In a notebook cell, after `A1/` is there: download A0 with
`snapshot_download("CohereLabs/tiny-aya-global", local_dir="A0", token=token)`, then
`from pathlib import Path; from msd.blend import blend_state_dicts; blend_state_dicts(Path("A1"), Path("A0"), 0.5, Path("results/resta/A2n_0.5"), a0_dtype="cast_to_a1")`.
The default `a0_dtype="strict"` refuses A0 and A1 with different float types (A1 is merged in fp16; Tiny Aya's
stored type is UNVERIFIED). The blend writes no record of its settings: note t and `a0_dtype` in the run log. Then
evaluate it like an alpha: `--arm A2n --alpha 0.5 --model results/resta/A2n_0.5` on dev and report.

## Step 7. Report split: BASE, A0 and A1 (runtime)

```
STEPS = [("run_eval", "--arm BASE --split report --model CohereLabs/tiny-aya-base --stages generate,task"),
         ("run_eval", "--arm A0 --split report --model CohereLabs/tiny-aya-global --stages generate,task"),
         ("run_eval", "--arm A1 --split report --model <hf-user>/<a1-repo> --stages generate,task"),
         ("run_eval", "--arm BASE --split report --judge <judge> --stages judge"),
         ("run_eval", "--arm A0 --split report --judge <judge> --stages judge"),
         ("run_eval", "--arm A1 --split report --judge <judge> --stages judge")]
```

- **Writes:** `results/report/{BASE,A0,A1}/none/` (the same files as in step 4). The A2 report cells came from step 6.
- **The freeze gate.** If a step stops with "the report split is generated only from the frozen test set", the
  runtime has no verified `data/testset/MANIFEST.json`: finish step 2, including its upload.
- **BASE** is the pre-alignment model, a reference only (design doc, section 4: "Reference only, cheap"). The gate,
  the alpha choice and the heatmap do not need it. Two things to know: whether it has a chat template (the generation
  applies one) is UNVERIFIED, and if its step fails on that, tell Claude (there is no fallback in the code); and the
  article draft says Base is "used only inside the repair arithmetic", so if BASE is evaluated, that sentence must
  change (`<<OWNER: run BASE as a reference arm, or not>>`).
- **The paid judge (optional, the key comes LAST).** The design judges only the report split with the paid model and
  never mixes judges inside one split. When the key and a spend cap exist:
  1. Cost check first: the step 4a smoke with `--judge <paid-judge>` in place of `<judge>` (and `EXTRA_PIP =
     "anthropic"`, the `ANTHROPIC_API_KEY` secret set). The judge step prints `judge usage: calls ..., input_tokens
     ..., output_tokens ...` for 10 items. The full report split is about 2,500 safety prompts per arm and alpha.
  2. Then rejudge EVERY report cell (BASE, A0, A1, and A2 at each alpha that was run), one entry per cell, for
     example `("run_eval", "--arm A1 --split report --judge <paid-judge> --stages judge --rejudge")` and
     `("run_eval", "--arm A2 --alpha 0.1 --split report --judge <paid-judge> --stages judge --rejudge")`. The old
     verdicts move to `results/report/_replaced/`. Until every cell is rejudged the split holds two judges and
     step 9 refuses it. The dev split keeps the stand-in judge; step 9 records and warns about that, as designed.
  3. This can also run on the Mac after step 9's download (**owner's yes** to
     `uv pip install --python envs/core/bin/python anthropic`; the key is read from `.env`). Then send `results/`
     back to the workspace repo, with the same exclusions as the notebook (call UNVERIFIED):
     `envs/core/bin/python -c "from huggingface_hub import HfApi; a=HfApi(); r='<hf-user>/msd-workspace'; assert a.repo_info(r, repo_type='dataset').private is True; a.upload_folder(folder_path='results', path_in_repo='results', repo_id=r, repo_type='dataset', ignore_patterns=['*.safetensors', '*.bin', '*.pt', '*.pth', 'work/**', '**/merged/**', 'merged/**', '**/.*.partial/**', '.*.partial/**', '.env', '**/.env'])"`

## Step 8. Second-judge reliability (runtime or Mac)

Run it once, after the report judge is final (after the paid rejudge, if there is one).

`STAGE = "run_reliability"`, `STAGE_ARGS = "--judge <second judge>"`. The second judge must differ from the judge
of the report verdicts: with the stand-in `<judge>` on the report split, use `<paid-judge>` or the other open model
(`hf:google/gemma-4-E4B-it`, fit on a T4 UNVERIFIED); with `<paid-judge>` on the report split, use `<judge>`.
An `anthropic:` second judge needs `EXTRA_PIP = "anthropic"` and the key secret.

- **Writes:** `results/report/reliability/<second judge>/...` (60 items per language by default) and prints Cohen's
  kappa per language and the low-confidence languages.
- **If it fails:** exit 4 = a judge problem (two judges in the report split, or the second judge is the first).
  Exit 1 with "labels of another sample": an earlier run with other settings left its labels; move that folder away
  in the workspace repo to start over. Only one second-judge folder may exist for step 9.

## Step 9. The report: tables, figures and the alpha choice (Mac)

1. Download the results from the workspace repo (needs `huggingface_hub` from step 1 and `.env` loaded into the shell
   as in step 1; call UNVERIFIED):
   `envs/core/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('<hf-user>/msd-workspace', repo_type='dataset', local_dir='.', allow_patterns=['results/**'])"`
   This writes `results/` (git-ignored), with the full completions in it: never share or commit that folder.
2. `envs/core/bin/python scripts/make_report.py`

- **Needs:** in every language, the dev cells of A0, A1 and every A2 alpha that was run (judged, judge errors under
  the limit in their eval.json); the report cells; one judge per split. Without the second judge of step 8 it still
  runs, but every language is marked low-confidence.
- **Writes:** `results/tables.csv`, `results/heatmap.png`, `results/alpha_curve.png`, `results/reliability.json`.
  Either all four are new or none exists.
- **Read:** `alpha_choice` in `reliability.json`: the chosen alpha, or `null` with the reason of every candidate. The
  alpha is chosen on dev only; the numbers in the tables are the report split's.
- **If it fails:** exit 1 = a data problem, the message names the file; 3 = no report split; 4 = a judge problem
  (mixed judges, a test judge, a second judge that is the first).
- It can also run on the runtime (`STAGE = "make_report"`); the four files are then uploaded with `results/`.

## Step 10. The public export, as a dry run (Mac)

`envs/core/bin/python scripts/export_public.py --out export/`

- **Reads:** `results/` (through the same checked reader as the report), the frozen `data/testset/` with its build
  record `data/testset_build_summary.json`, `data/sources.yaml`, `templates/`. It never opens a completions file.
- **Writes:** `export/` only (git-ignored): the dataset folder with its card, the public per-item rows, the four
  report files, a model card per arm, and `MANIFEST.json`. Nothing is sent anywhere.
- **Then:** the owner looks through `export/`. To run it again, remove the folder first (`rm -r export`): a
  non-empty `--out` is refused.
- **If it fails:** 1 = a data problem or the safety scan found a secret or a home path (the message names the
  file and the kind, never the value); 3 = no report split; 4 = a judge problem; 5 = a source that may not be
  redistributed.

## Step 11. Run the reproduction notebook once for real (Mac)

The notebook `notebooks/reproduce.ipynb` has never run as a notebook. Do this before publishing.

1. **Owner's yes:** `uv pip install --python envs/core/bin/python nbclient ipykernel`
2. `envs/core/bin/python -m pytest -q tests/test_reproduce_notebook.py`: with those two packages its two kernel tests
   run instead of being skipped.
3. Run it on the real export, from the repository folder (the kernel name `python3` resolving to `envs/core` is
   UNVERIFIED):
   `envs/core/bin/python -c "import nbformat, nbclient; nb = nbformat.read('notebooks/reproduce.ipynb', as_version=4); nbclient.NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute(); nbformat.write(nb, 'reproduced/reproduce.executed.ipynb')"`
   It must finish without an error and print `MATCH` (and no `MISMATCH`). It writes into `reproduced/`
   (git-ignored). Leave `RUN_GPU = False`: the GPU section is optional and not part of this check.

## Step 12. Fill in the article (Mac)

Follow `article/FILL_IN.md`: one line per placeholder, with the file and key each value comes from. Copy
`export/dataset/aggregates/heatmap.png` and `alpha_curve.png` next to the article. Then:

`envs/core/bin/python scripts/check_article.py article/ARTICLE.md data/sources.yaml`

with no flag. It must exit 0: that is the state to submit. (While placeholders are left, the same command with
`--draft` is the check that passes.)

## Step 13. Owner-only steps

None of the scripts publishes anything. Each of these is the owner's own action, after reading what goes out:

1. Decide which account publishes: `<<OWNER: publishing account>>`.
2. Upload `export/dataset/` as a public dataset (its card is `export/dataset/README.md`). Decide first whether the
   machine-translated prompts (`prompts_translated.jsonl`, CC-BY-NC-4.0) are released; the export marks that file
   optional.
3. Upload the model cards of `export/models/`, and weights only for the arms the owner decides to release (A1 from
   its private repo; A2 from step 6, "Keeping A2"). Make them public only then. Put them in a Hugging Face collection.
4. Publish the code repository (it is local only now).
5. Publish the article on Hugging Face, then fill in the submission form (it comes by email a couple of days before
   the deadline).
6. Keep the workspace repo PRIVATE for good: it holds the full completions, harmful ones included.

## If the schedule slips: the cut list

Cut in this order (design doc, section 8, risk 7, and the rulings in the build log):

1. **A3, the prevention arm** (SafeTune Harden). Already cut: skipped by a ruling, not built.
2. **The English-only variant** of the fine-tune data. Not built; nothing to do.
3. **Extra alpha values.** Keep 0.0 (the no-op check) and 0.1, 0.25, 0.5; drop 0.05, 0.15, 0.35 and the 1.0 break
   control first. The blends (A2n) are optional and go with them (they are not in the design's list; placing them
   here is a suggestion).
4. **The demo Space.** Not built; nothing to do.

If more must go, the report-split BASE cells (step 7) are a reference only and nothing else depends on them (a
suggestion, not in the design's list).

Fewer languages only if the gate and the measurement still work: the freeze needs all five languages (it cannot be
smaller), and the gate needs at least two non-English languages and changes its pass chance when one is left out.
Decide that before the gate runs, never after seeing it.

Never cut: the freeze and its log (step 2), the split between dev (choices) and report (numbers), the one-judge
rule for each set of numbers, the second-judge reliability check (step 8), and the run of the reproduction notebook
(step 11).

## Which file comes from where

| File | Written by (step) | Read by (step) |
|---|---|---|
| `data/testset/<lang>/{dev,report}.jsonl`, `MANIFEST.json` | `build_testset.py` (2) | `run_eval.py` (4, 6, 7), `run_reliability.py` (8), `export_public.py` (10) |
| `data/testset_build_summary.json` | `build_testset.py` (2) | `build_testset.py --freeze-only` (2), `export_public.py` (10) |
| `docs/FREEZE_LOG.md` | the owner, by hand (2) | the article (12) |
| `results/finetune/<config stem>/finetune_summary.json` | `finetune.py` (3) | the article (12) |
| `<hf-user>/<a1-repo>` (A1 weights) | `finetune.py --push-to` (3) | `run_eval.py --model` (4, 7), the notebook's `A1_REPO` (6) |
| `results/<split>/<arm>/<alpha or none>/<lang>.*` | `run_eval.py` (4, 6, 7) | `run_gate.py` (5), `run_reliability.py` (8), `make_report.py` (9), `export_public.py` (10) |
| `results/gate.json` | `run_gate.py` (5) | the article (12) |
| `results/resta/A2_alpha_<a>/resta_patch.json` | `run_resta.py` (6) | the article (12) |
| `results/report/reliability/<judge>/...` | `run_reliability.py` (8) | `make_report.py` (9) |
| `results/{tables.csv,heatmap.png,alpha_curve.png,reliability.json}` | `make_report.py` (9) | `export_public.py` (10), the article (12) |
| `export/` | `export_public.py` (10) | the reproduction notebook (11), the article (12), the owner (13) |
