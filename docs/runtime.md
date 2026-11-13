# Runtime: where the GPU work runs

The Mac cannot train or run Tiny Aya (no CUDA). The organizers said free Kaggle or Colab GPUs are
the compute. This page is the plan for using them. Status on 2026-11-05: everything that needs your
accounts is waiting for you (section 1). The code side is built and tested on the Mac only.

Labels used below: **UNVERIFIED** = I could not confirm it from a page I read. **DEFERRED-RUNTIME**
= cannot be done until the accounts exist, so it is written down but not run.

## 1. Owner checklist (do these in order)

1. **Accept the Tiny Aya license on Hugging Face.** Log in to Hugging Face, open both pages, fill
   the short form (name, affiliation, country, tick "non-commercial use ONLY") and click agree:
   - https://huggingface.co/CohereLabs/tiny-aya-global
   - https://huggingface.co/CohereLabs/tiny-aya-base

   Access is automatic (both repos are "gated: auto" in the public API, checked 2026-11-05), and
   Hugging Face's gated-model docs say access is immediate with automatic approval. Nothing else of
   the Tiny Aya family is needed (the others are regional or thinking variants; names in `src/msd/config.py`).
2. **Create a Hugging Face token** at https://huggingface.co/settings/tokens. Hugging Face's docs
   (https://huggingface.co/docs/hub/security-tokens, read 2026-11-05) define the roles:
   - `read`: read access only. Enough to download the two Tiny Aya models after step 1.
   - `write`: also lets the token push to repos you can write to. **Needed for saving results to
     our private repo**, which every Kaggle/Colab run does. So make a `write` token (name it
     `msd-runtime`).
   - `fine-grained`: limits a token to named repos. Safer, but I have not checked its screens in
     detail; `write` is fine for a short hackathon.
3. **Put the token in `.env`** in `~/Desktop/lexsi-hackathon/` as the line `HF_TOKEN=...` (copy
   `.env.example` to `.env`; `.env` is git-ignored). Never paste the token into chat.
4. **Pick the GPU route** (one line of difference: Kaggle can be started and watched from here once
   set up; with Colab you press "Run all" yourself every time).
   - **Kaggle:** create an account and verify the phone number (the design doc says GPU and
     internet need it; UNVERIFIED, no Kaggle page was readable). Then make an API token at
     https://www.kaggle.com/settings/api. The current Kaggle command-line docs name two ways: a
     token in `KAGGLE_API_TOKEN`, or a legacy `kaggle.json` key (read 2026-11-05 at
     https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md). They do not mention
     `KAGGLE_USERNAME` / `KAGGLE_KEY`. Put into `.env` whichever the settings page gives you
     (`.env.example` has all three names); I check which one works the first time we use it.
   - **Colab:** nothing to create beyond your Google login.
5. **Add `HF_TOKEN` as a secret inside the notebook site** (Kaggle cannot do this from the
   command line, see section 2). Do it after the first notebook exists, because the secret is
   attached to a notebook:
   - Kaggle: create a notebook (New Notebook), then in it open Add-ons > Secrets and add a secret
     labelled `HF_TOKEN`. Also switch Internet on and pick the GPU in that notebook's settings. A
     notebook that I later push from the command line is a different notebook, so it probably needs
     the secret attached too (UNVERIFIED). Until it is, its first run stops at the first cell with
     the message "HF_TOKEN not found". That is expected and harmless.
   - Colab: the key icon in the left bar (Secrets) > add `HF_TOKEN` and switch "Notebook access"
     on (exact Colab menu wording from memory, UNVERIFIED).
6. **Later, when I ask:** say yes to creating the private Hugging Face dataset repo
   `<your-hf-username>/msd-workspace` (section 2). Nothing is created on Hugging Face, Kaggle or
   Colab until then.

## 2. Workspace design (not built yet)

One **private Hugging Face dataset repo**, `<owner>/msd-workspace`, holds everything the runtimes
need, and both Kaggle and Colab pull from it and push to it with `HF_TOKEN`:

```
wheels/msd-<version>-py3-none-any.whl   the code in src/msd (keep exactly one wheel here)
scripts/                                copy of this repo's scripts/ (the stage scripts)
pipelines/                              copy of this repo's pipelines/ (CuratorKIT YAML the scripts read)
configs/                                copy of this repo's configs/ (the fine-tune YAML files)
data/                                   the frozen test set and the fine-tune data (data/testset/, data/ft/*.jsonl)
results/                                the ONE results root: every stage reads and writes it (docs/RUNBOOK.md)
```

**Known gap, stated plainly:** the wheel contains only the `msd` package from `src/`. It does NOT
contain `scripts/` or `pipelines/`. So the workspace repo must also carry `scripts/` and
`pipelines/` (and `data/` when a stage needs it), and they must be uploaded again whenever they
change; a stale copy there means the runtime runs old code. There is no sync helper: `docs/RUNBOOK.md`, step 1,
has the upload command.

Two thin notebooks live in `kaggle/` (they work on Colab too, nothing is Kaggle-only):
`00_env_check.ipynb` (install the wheel, print the environment JSON and save it as `env_check.json`) and `10_run_stage.ipynb`
(check that the workspace repo is private, pull `scripts/`, `pipelines/`, `configs/`, `data/` and `results/`,
optionally download A1 into `A1/`, run `python scripts/<STAGE>.py` or an ordered list of `STEPS` in one session, and
upload the whole `results/` folder after each step when it differs from what was pulled or last uploaded, so a
failed upload is sent again by the next run of the last cell; never weights). Every stage script exists; the
order of the real run is in `docs/RUNBOOK.md`. They take the token from `HF_TOKEN` in the environment, a Kaggle
secret, or a Colab secret, and never print it; an `ANTHROPIC_API_KEY` secret is loaded the same way when present
(optional, for an `anthropic:` judge only).

**Kaggle launch from here** (needs the Kaggle account, a token, and your yes):
`kaggle kernels push -p <folder>` uploads a notebook and runs it ("Pushes new code/notebook and
metadata to a kernel, then runs the kernel"; `--no-run` only saves). The folder holds the notebook
and a `kernel-metadata.json`. Template (fill in the username):

```json
{"id": "<kaggle-username>/msd-run-stage", "title": "msd run stage", "code_file": "10_run_stage.ipynb",
 "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
 "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
 "competition_sources": [], "kernel_sources": [], "model_sources": []}
```

Then `kaggle kernels status <id>` to watch it. Sources: Kaggle's metadata docs
(https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md) and kernel commands
(https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md), both read 2026-11-05.

**Can the command line attach the secret? No.** The same Kaggle docs page says, under "Using
Secrets in Kernels": "Define Secrets on Kaggle.com (no CLI support)", and the metadata file has no
secrets field. So the `HF_TOKEN` secret is added in the Kaggle web page, inside a notebook that
already exists (checklist step 5). UNVERIFIED: whether a notebook created by `kaggle kernels push`
gets the secret too, or keeps it across later pushes (one third-party search snippet claimed it may
not; I could not open that page). Expect to attach it by hand once to the pushed notebook. The first
cell of both notebooks stops with the message "HF_TOKEN not found" if it is missing, so a missing
secret shows up in seconds.

## 3. Free GPU limits (read 2026-11-05)

Only facts from pages I actually read. Kaggle's own documentation pages
(www.kaggle.com/docs/notebooks, /docs/efficient-gpu-usage, /page/GPU-tips-and-tricks) are built by
JavaScript and returned only their titles, so most Kaggle limits are UNVERIFIED. The numbers will be
shown on the Kaggle account page after sign-up.

| | Kaggle | Colab (free) |
|---|---|---|
| GPU type | Default GPU is `NvidiaTeslaT4` = "GPU T4 x2" (two cards; memory per card UNVERIFIED, env_check will report it). The list also names A100, L4, H100 and TPUs, and says "some of these are only available to participants of specific competitions, and some are only available to Kaggle admins" without saying which. Source: kaggle-api `docs/kernels.md`, "Accelerators available as of Sep 2026". | Not named. "The types of GPUs and TPUs that are available in Colab vary over time." Not guaranteed: "Colab resources are not guaranteed and not unlimited." Source: https://research.google.com/colaboratory/faq.html |
| Weekly GPU quota | UNVERIFIED | Not published: "Colab does not publish these limits, in part because they can vary over time." Same FAQ. |
| Session length | UNVERIFIED (`kaggle kernels push --timeout <seconds>` exists, so a run has a settable maximum) | "notebooks can run for at most 12 hours, depending on availability and your usage patterns." Idle: "Runtimes will time out if you are idle" (no number given). Same FAQ. |
| Internet | Off unless set: `enable_internet` "If not specified, will be `false`" (metadata docs). Phone verification requirement UNVERIFIED. | FAQ is silent on runtime internet access: UNVERIFIED. |
| Memory / disk | UNVERIFIED | "standard system memory profile" on free, no numbers; extra Drive space does not add VM disk. Same FAQ. |

Practical reading: Colab can drop the GPU or stop at any time, so every stage must save to the
workspace repo and be resumable. Kaggle's numbers need to be read off the account after sign-up.

## 4. Python versions and SafeTune (needs Python 3.12 or newer)

Required Python per package (PyPI JSON, read 2026-11-05): `safetune` 0.1.12 needs >=3.12,
`aligntune` 0.1.30 >=3.11 (0.1.31 was published later the same day, also >=3.11), `curatorkit` 0.1.22 >=3.11, `auditkit` 1.1.3 >=3.10. Our own `msd` needs >=3.12.

- **Colab:** the newest runtime snapshot is Python 3.13. Evidence:
  https://github.com/googlecolab/backend-info README lists runtime 2026.10 = Python 3.13.18
  (2026.07 and 2026.04 = 3.12.13, 2026.01 and 2025.10 = 3.12.12), and its latest `pip-freeze.txt`
  (commit 066189cf29, 2026-11-04) has `cp313` wheels. Colab can pin an older runtime. Which one a
  new notebook gets today is not stated on any page I read: UNVERIFIED. The same file lists
  transformers 5.22.1, peft 0.21.2, accelerate 1.15.0, datasets 4.8.5, safetensors 0.8.0 and no
  vllm, bitsandbytes or trl (its torch line is a CPU build, so it looks like the CPU image; the GPU image may differ).
- **Kaggle:** the image recipe (https://github.com/Kaggle/docker-python, `Dockerfile.tmpl`, commit
  6a20e87c19 of 2026-10-30) builds from a Colab runtime image dated 2026-10-27 and uses
  `/usr/local/lib/python3.13`, so Python 3.13, and `kaggle_requirements.txt` asks for
  `transformers>=5.0.0`. This is the recipe, not a live notebook: UNVERIFIED until env_check runs there.

So the Python rule for SafeTune should pass on both. What is still open is whether SafeTune,
AlignTune and AuditKIT agree with the preinstalled torch and transformers 5.x; the first
`env_check` plus an install attempt answers it (section 5). If SafeTune will not install: spec
section 8 ("Risks and the decision rule"), item 3 says run ReSta on the Mac or do the same weight
arithmetic by hand with safetensors and say so. That is plan Task 16 (`src/msd/blend.py` streams
tensor by tensor; a hand-written ReSta uses the same pattern).

## 5. DEFERRED-RUNTIME commands (plan Task 10, Step 7)

Runs on the Mac now (already done once, 2026-11-05: `msd-0.1.0-py3-none-any.whl` built; the wheel
installed into a throwaway venv and `python -m msd.envcheck --offline` ran from it):

```
uv build --wheel --out-dir dist .        # or: python -m pip wheel . --no-deps -w dist
python scripts/env_check.py --offline    # the laptop baseline, no network
python scripts/env_check.py              # adds the Hugging Face HEAD check of both Tiny Aya repos
```

Without a token the HEAD check reports `401 unauthorized` for both repos (seen 2026-11-05). With a
token it should report `ok` once the license is accepted (Hugging Face answers either 200 or a 307
to its own cache for the same repo; both count as `ok`; any other redirect shows as
`unexpected_redirect`, for example a renamed repo). A `403 forbidden` is the likely answer for
"token fine, license not accepted", but I have not seen it with a real token: UNVERIFIED.

If the report shows `hf_token_present: true` and still `401 unauthorized`, there are two possible
causes: the token is wrong or expired, or the license for that repo is not accepted yet. Check both:
make a fresh token (step 2) and open these two pages again to see that you have access:
https://huggingface.co/CohereLabs/tiny-aya-global and https://huggingface.co/CohereLabs/tiny-aya-base

Waiting for the owner (all DEFERRED-RUNTIME):

```
# 1. Create the private workspace repo and upload the wheel + scripts + pipelines (needs the owner's yes and HF_TOKEN).
#    HfApi() reads HF_TOKEN from the shell environment, not from .env, so load .env into this shell first
#    (prints nothing; values with spaces need quotes in .env):
set -a; . ./.env; set +a
python -c "from huggingface_hub import HfApi; a=HfApi(); r='<hf-user>/msd-workspace'; \
a.create_repo(r, repo_type='dataset', private=True); \
a.upload_file(path_or_fileobj='dist/msd-0.1.0-py3-none-any.whl', path_in_repo='wheels/msd-0.1.0-py3-none-any.whl', repo_id=r, repo_type='dataset'); \
[a.upload_folder(folder_path=d, path_in_repo=d, repo_id=r, repo_type='dataset', ignore_patterns=['__pycache__', '*.pyc']) for d in ('scripts', 'pipelines')]"
# Repeat the upload_folder part whenever scripts/ or pipelines/ change (see the known gap in section 2).
# docs/RUNBOOK.md, step 1, has the same command with configs/ and a privacy check added; step 2 uploads data/.
# (huggingface_hub is not installed in envs/core; the exact call signatures are from memory: UNVERIFIED.)

# 2. Kaggle: put the WORKSPACE_REPO name into a copy of kaggle/00_env_check.ipynb, add kernel-metadata.json
#    (template in section 2, code_file 00_env_check.ipynb), then
kaggle kernels push -p <folder-with-notebook-and-metadata>
kaggle kernels status <kaggle-username>/<slug>
kaggle kernels output <kaggle-username>/<slug> -p <outdir>      # downloads the files the notebook wrote, i.e. env_check.json
#    (the notebook saves the report to env_check.json in its working folder; that this folder is what
#    `kernels output` downloads is UNVERIFIED until the first run; the JSON is also printed in the notebook log)

# 3. Colab: https://colab.research.google.com > File > Upload notebook > kaggle/00_env_check.ipynb;
#    Runtime > Change runtime type > GPU; add the HF_TOKEN secret; Run all.

# 4. On each runtime, after env_check, try the toolkits and run env_check again:
pip install aligntune safetune auditkit
python -m msd.envcheck
```

Record each JSON here, then choose the primary runtime. Rule: the runtime where `aligntune`,
`safetune` and `auditkit` install without breaking torch, and the GPU has about 15 GB of memory;
if both qualify, prefer Kaggle (it can be launched from here); if neither installs SafeTune, use the
fallback in section 4.

| | Laptop (envs/core, 2026-11-05) | Kaggle | Colab |
|---|---|---|---|
| python / platform | 3.12.12, macOS arm64 | DEFERRED-RUNTIME | DEFERRED-RUNTIME |
| GPU, VRAM | none, none | DEFERRED-RUNTIME | DEFERRED-RUNTIME |
| torch / CUDA | not installed | DEFERRED-RUNTIME | DEFERRED-RUNTIME |
| toolkit versions | aligntune, safetune: not installed; auditkit is in its own venv `envs/auditkit`, so null here | DEFERRED-RUNTIME | DEFERRED-RUNTIME |
| Tiny Aya HEAD (no token) | 401 on both | DEFERRED-RUNTIME | DEFERRED-RUNTIME |
| Decision | | | |

### Fine-tune A1 with AlignTune (plan Task 14, DEFERRED-RUNTIME)

`scripts/finetune.py` trains A1 = A0 + a LoRA adapter on the harmless Global-MMLU letter task and merges it to
fp16 weights. Every training value is in `configs/ft_lora.yaml` (and `configs/ft_lora_strong.yaml`, the retry);
what each AlignTune keyword does is in `docs/toolkit-notes.md`, "AlignTune SFT (read from source)". The code
is tested with fakes only. Nothing below has run on a GPU, and every time or memory figure is an UNVERIFIED
estimate until the first run replaces it.

**What works today, on the Mac, with no GPU and no AlignTune** (done 2026-11-05 on the real data: 6,000 rows,
1,500 per language, train sha256 `348c164f...01ce80` as recorded in the notes, validation ok with heldout,
taskdev and the safety splits all checked, 750 steps for the base config and 1,125 for the strong one; the
token lengths line said "not estimated" because there is no tokenizer here):

```
python scripts/finetune.py --config configs/ft_lora.yaml --dry-run
python scripts/finetune.py --config configs/ft_lora_strong.yaml --dry-run
```

**Before the first run on Kaggle or Colab** (three steps, stated plainly):

1. The wheel in the workspace repo must be rebuilt and replaced (`uv build --wheel --out-dir dist .`, upload, keep
   exactly one): `msd.finetune` is new. `scripts/finetune.py`, `pipelines/` and `data/` (with `data/ft/` and
   `data/testset/`) must be current there as well.
2. The stage notebook now pulls `configs/*` as well as `scripts/*`, `pipelines/*` and `data/**`. Upload the two
   YAML files into the workspace repo at `configs/ft_lora.yaml` and `configs/ft_lora_strong.yaml` (no workaround
   under `data/` is needed any more) and pass `--config configs/ft_lora.yaml`. The summary records that path and the
   file's sha256. Use the updated `kaggle/10_run_stage.ipynb` (Kaggle and Colab keep their own copy of a notebook).
3. The notebook installs a toolkit only when you ask: set `EXTRA_PIP = "aligntune==0.1.31"` in its settings cell
   (default empty; installed with `pip install -q` before the stage runs; needs Python 3.11 or newer; 0.1.31 was
   published 2026-11-05). It pulls torch/transformers/trl/peft pins (`transformers>=5.15,<6`, `trl==1.7.1`,
   `torch>=2.4,<2.14`) and may replace the preinstalled torch: run `env_check` again afterwards (UNVERIFIED either
   way). Use this runtime only for the fine-tune, not for SafeTune (one environment per toolkit); leave `EXTRA_PIP`
   empty for the other stages.

The token: the notebook already puts `HF_TOKEN` into the environment; the gated base model
(`CohereLabs/tiny-aya-global`, licence accepted on its page) is downloaded with it. `finetune.py` never prints it.
On a two-T4 Kaggle runtime the config uses the first card only (`aligntune.device_map: {"": 0}`).

Run these in order, each as a separate run of the stage notebook (`STAGE`, `STAGE_ARGS` in its second cell):

```
# 1. Plan on the runtime (a tokenizer can be loaded here, so the token lengths and the longest example are printed;
#    check that max_seq_length 2048 covers it). Writes nothing.
STAGE = "finetune"
EXTRA_PIP = "aligntune==0.1.31"  # settings cell; it brings transformers and trl, which give the dry run its token lengths and template check
STAGE_ARGS = "--config configs/ft_lora.yaml --dry-run"

# 2. Smoke run: 20 steps, every step logged, merges once so the merge path (CPU RAM, tokenizer save) is proven
#    BEFORE the long run. Output goes to results/finetune/ft_lora-smoke, its merged/ holds a SMOKE.txt marker, it is
#    not A1 and is never pushed; a smoke run refuses --out results/finetune/ft_lora (the real run's folder).
STAGE_ARGS = "--config configs/ft_lora.yaml --max-steps 20 --merge"
#    Read the printed lines and results/finetune/ft_lora-smoke/finetune_summary.json: rows_seen must be 6000 (a lower
#    number means AlignTune/CuratorKIT dropped rows), loss_ok true, steps 20, label_check with about 3 labelled
#    tokens per row (0 or a large number means the assistant mask is wrong), args_check listing the 13 trainer
#    arguments it compared, chat_template_sha256, peak memory in the log (UNVERIFIED how it is shown; watch
#    nvidia-smi). If the adapter check, the keyword names or the assistant-only mask are wrong this is where it shows.
#    READ label_check BEFORE THE FULL RUN: see "Reading label_check" below the commands.

#    Do not read too much into the smoke run's loss. The task answer is one letter, so each example has only about 3
#    labelled tokens (the letter and the two end tokens), so a logged step averages only about 48 labelled tokens
#    (an effective batch of 16 examples): with 20 steps the loss is noisy, and a chance failure of the "loss fell" rule is NOT proof of a bug. Look at the printed
#    per-step losses, label_check and args_check, and run the smoke once more before deciding.

# 3. The real run: 2 epochs = 750 optimizer steps. Writes results/finetune/ft_lora/{adapter,merged,finetune_summary.json}.
STAGE_ARGS = "--config configs/ft_lora.yaml --push-to <hf-user>/<private-model-repo>"
#    --push-to is how the merged weights leave the runtime (the notebook's own upload never sends weights, below): the
#    repo must already exist and be a PRIVATE MODEL repo you made by hand, HF_TOKEN must be allowed to write to it;
#    nothing is created, nothing is uploaded without the flag, and privacy is checked before training and again right
#    before the upload. Without it the merged model stays on the runtime's disk and is gone with the session.

# 4. Only if step 3's held-out accuracy (below) shows the model barely moved: the strong retry, 3 epochs = 1,125 steps.
STAGE_ARGS = "--config configs/ft_lora_strong.yaml"
```

**Reading `label_check`** (in `finetune_summary.json` of the smoke run, before the full run). The label mask is what
makes the experiment valid: without it the loss also covers the chat template's long safety preamble and the question,
a confound. The check runs at step 0 of training (a callback on `on_train_begin`) and has three outcomes:

1. **Counts, about 3 each** (`label_check: {counts: [...], allowed_min, allowed_max}`; `label_check_note` null). The
   mask works: the letter and the two end tokens are the only labelled tokens. Go on to the full run.
2. **`label_check: "unavailable"`** (with `label_check_note` saying why, and a WARNING on stdout and stderr). The
   callback could not read the labelled rows: the mask was NOT verified. Do not start the full run. A real run
   without the check stops at step 0 (exit 1) and tells you so. Inspect the first batch by hand in the notebook
   (build the same `SFTTrainer` or tokenise a few rows with the template saved in the work folder, and look at
   `labels` against `input_ids`) and find out why the callback saw nothing, for example a `labels` column that the
   dataloader removed. Only after that, if you decide to accept an unverified mask, run the full run with
   `--allow-unchecked-labels`: it prints a WARNING and records `label_check_waived: true` in the summary.
3. **Out of bounds** (the run exits 1 at step 0, naming the row and the count, in a smoke run as in a real one; no summary is written).
   0 labelled tokens: the assistant mask matched nothing (the template has no generation markers, or
   `assistant_only_loss` did not reach TRL). A large count: the loss covers more than the answer. Fix the cause;
   `--allow-unchecked-labels` never waives this.

The notebook then uploads the whole `results/` folder (with `results/finetune/` in it) to the private workspace repo,
**without weights**: its upload skips `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `work/**`, any `merged/` folder and
hidden `.partial` folders, so the summary and logs travel but the adapter and the 6.7 GB merged model do not (they go
only through `--push-to`, above). A failed run exits non-zero and stops the notebook before the upload.

**Where things are written.** AlignTune trains into a persistent work folder, `work/finetune/<config stem>[-smoke]/`
(checkpoints every `save_steps`, the adapter files at the end). It is kept on ANY failure, including Ctrl-C, and the
error names it; a later run refuses to start while it holds anything but the chat-template file, so a failed run
cannot silently overwrite its own recovery files. After training the adapter is copied (without checkpoints) into its
own folder `<out>/adapter`, then the merged weights go into `<out>/merged`, each written under a hidden `.partial`
name and renamed only when whole. The adapter folder also holds a small `finetune_marker.json` (`smoke`,
`max_steps`, the config stem): it says which kind of run saved the adapter, so that `--merge-only` never has to guess.
`finetune_summary.json` is written twice: right after training, before the merge starts, as a provisional file
(`merged: false`, `merge_pending: true`, with the loss history and every check), and again after the merge succeeds
as the final one (`merged: true`, no `merge_pending`). So `merged: true` is what means the run finished, and a merge
that dies leaves the training record behind. A successful run deletes the work folder (`output.keep_checkpoints: true` keeps it). A killed session
(SIGKILL, a dropped runtime) cannot clean up: a hidden `.<name>.<8 hex>.partial` folder stays and is named at the
start of the next run, never deleted.

**After a failure.** AlignTune itself has no resume: it has no switch to continue from a checkpoint, so a dropped run
starts again. What is kept is for recovery by hand, and **only within the SAME live session**: everything below sits
on the runtime's disk, the notebook uploads neither adapter weights nor `work/`, and a failed stage stops before its
upload step, so a session that ends takes the adapter, the work folder and the provisional summary with it. Two
habits make that survivable. Run the 20-step smoke with `--merge` first (step 2 above), so that a merge problem
(CPU RAM, the tokenizer save) shows up in minutes, not after the long run. And if a merge fails and the session has
to end anyway, copy `<out>/adapter` somewhere durable by hand first (it is only the adapter, not the 6.7 GB model;
keep `finetune_marker.json` in it, which says whether it came from a smoke run).

- The merge failed (out of RAM, say): the error (for an out-of-memory error the `note:` line above the traceback)
  names `<out>/adapter` and prints the whole command, `--out` of the failed run included. Copy it as it is, for a smoke
  run as for the real one:
  `STAGE_ARGS = "--config configs/ft_lora.yaml --merge-only --adapter results/finetune/ft_lora/adapter --out results/finetune/ft_lora"`
  (a failed smoke run prints `results/finetune/ft_lora-smoke/adapter` and `--out results/finetune/ft_lora-smoke`). It
  checks that adapter against the config (base model, rank, alpha, dropout, bias, target modules, and that the weights
  file is there), merges into `<out>/merged` and completes the provisional `finetune_summary.json` the failed run left:
  the training record stays, `merged` becomes true, `merge_pending` goes, and a `merge` section is added (a small
  summary is written when there was none). It trains nothing and reads no data; it takes no `--data`, `--seed`,
  `--max-steps`, `--no-merge`, `--dry-run` or `--push-to`.
  It reads `finetune_marker.json` in the adapter folder to tell a smoke adapter from a real one. A smoke adapter is
  merged with a `SMOKE.txt` beside the weights and `smoke: true` in the summary, and is refused (exit 2) with the real
  run's folder `results/finetune/ft_lora` as `--out`, given or by default; a real adapter is refused with a folder
  named `*-smoke` as `--out`. An adapter without a marker (saved before it existed) counts as a smoke adapter only if
  its own folder or its parent is named `*-smoke`; a damaged marker is an error, never a guess.
- Training failed or the session was interrupted: look in the named work folder for `checkpoint-<step>/`. Continuing
  one needs your own script (`Trainer.train(resume_from_checkpoint=<folder>)`; UNVERIFIED that it works through
  AlignTune's objects). Otherwise move or delete the work folder and run again.
- A smoke run that fails the "loss fell" rule by chance (the 20-step loss is noisy, see step 2) keeps its work
  folder `work/finetune/ft_lora-smoke`, and a rerun of the smoke is refused (exit 2) until that folder is gone. The
  error prints the command: `rm -r work/finetune/ft_lora-smoke` (in a notebook cell: `!rm -r work/finetune/ft_lora-smoke`
  from the repository folder). Delete it only after you have read what you need from it; the real run has its own
  folder, `work/finetune/ft_lora`, and the same rule.

**Held-out accuracy check, A0 against A1, and the dev references of the alpha choice.** `run_eval.py` reads
`data/ft/taskdev.jsonl` for `--split dev` and `data/ft/heldout.jsonl` for `--split report` and scores the bare answer
letter in its task stage (no judge needed for that part). The generate stage also writes the safety completions of that
split (the study needs them for A0 and A1 anyway); `--limit N` caps both for a quick look. Dev A0 and dev A1 must be
run with the judge stage as well (`--stages generate,judge,task` with `--judge`, a key or a GPU judge): `make_report.py`
lets an alpha win only if its dev over-refusal is no higher than the larger of A0's and A1's plus the margin, so a dev
A0 or A1 without judged verdicts gives no over-refusal reference and no alpha can ever be chosen. Leave `--langs` out
(all five languages): the task items exist for en, hi, ar and sw only, and the task stage skips Hinglish loudly
(`task_skipped` in its eval.json) while its safety completions are still judged. Run the dev split first, A0 then A1,
then the report split once:

```
STAGE = "run_eval"
STAGE_ARGS = "--arm A0 --split dev --model CohereLabs/tiny-aya-global --judge <judge> --stages generate,judge,task"
STAGE_ARGS = "--arm A1 --split dev --model results/finetune/ft_lora/merged --judge <judge> --stages generate,judge,task"
# then the same two with --split report
```

(`--model results/finetune/ft_lora/merged` works only in the same live session that made it: merged weights are never
uploaded to the workspace repo and a later notebook run starts from a fresh working folder, so in a later run pass the
`--push-to` repo id (`<hf-user>/<private-model-repo>`) as `--model`. `run_eval.py` writes into the one results root
`results/` by default: do not pass `--out`, so the gate, the report and the export find the files. `docs/RUNBOOK.md`,
step 4, splits generation and judging into separate steps.) Read `results/dev/<arm>/none/<lang>.eval.json`: its `tasks` block has `n`, `correct` and
`accuracy` (one file per language). Compare with the always-D floor in `docs/toolkit-notes.md` (31% on dev, 27 to 28% on the
heldout split). What A1 must show is that it answers the bare letter more often than A0 does (the point of the
task); how much is a result, not a gate here.

**Which dev cells must exist before `make_report.py` can choose an alpha.** Every one of these, in every language
(en, hi, hinglish, ar, sw), generated and judged by the one judge, with judge errors under the allowance in its
eval.json:

- dev A0 and dev A1 (no alpha): their dev over-refusal is the reference an alpha is measured against, and A1's dev
  task accuracy is what each repaired cell's must stay near;
- dev A2 at every alpha of the grid (`--arm A2 --alpha <a> --split dev`), with its generations kept: the sanity gate
  reads the dev benign completions, and an A2 cell never run on dev is excluded;
- dev A3 (no alpha) and dev A2n at each blend value, if those arms are used: they go through the same gate, and a
  report cell without its dev cell is excluded.

The task-accuracy part of the gate needs task items, so it runs for en, hi, ar and sw (their dev task cells must exist
too: a missing one is "unverified" and keeps the alpha out); for Hinglish, which has no task items, it is "not
applicable" and recorded as such, while the other four checks and the over-refusal rule still apply. The report split
needs the same arms and alphas, judged by the same judge; when it is rejudged into a fresh folder, give the dev folder
with `--dev`.

**Expected cost, all UNVERIFIED estimates** (nothing measured): with batch 1 and gradient checkpointing a step
processes 16 sequences of mostly 500 to 1,500 tokens; AlignTune's own heuristic puts the base config at 1.7 h
(1,024-token sequences) to 3.4 h (2,048) on a T4 (formulas and table in the notes, with its +/-30%), so plan 2 to
4 hours for the base config and 3 to 6 hours for the strong one, the 20-step smoke run at a few minutes plus model
download (6.7 GB) and a merge of a few minutes. Memory: about 13 to 14 GB at the longest examples against 16 GB
(arithmetic in the notes). If step 2 runs out of memory, lower `train.max_seq_length` only to what the dry run's
longest example needs, never below it (the run refuses a shorter limit); `per_device_batch_size` is already 1. A
Kaggle or Colab session limit (section 3 lists only what was read) may cut a 4 to 6 hour run: the strong retry is the
one at risk, and there is no resume.
