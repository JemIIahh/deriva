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
data/                                   test set, fine-tune data (only what a stage needs)
results/<stage>/                        per-item JSONL, logs, written by the notebooks
```

**Known gap, stated plainly:** the wheel contains only the `msd` package from `src/`. It does NOT
contain `scripts/` or `pipelines/`. So the workspace repo must also carry `scripts/` and
`pipelines/` (and `data/` when a stage needs it), and they must be uploaded again whenever they
change; a stale copy there means the runtime runs old code. A sync helper does not exist yet (a
later task adds it).

Two thin notebooks live in `kaggle/` (they work on Colab too, nothing is Kaggle-only):
`00_env_check.ipynb` (install the wheel, print the environment JSON and save it as `env_check.json`) and `10_run_stage.ipynb` (set
`STAGE`, pull `scripts/`, `pipelines/` and `data/`, run `python scripts/<STAGE>.py`, push
`results/<STAGE>/`). Later tasks add the stage
scripts; the notebooks do not change. They take the token from `HF_TOKEN` in the environment, a
Kaggle secret, or a Colab secret, and never print it.

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

**Before the first run on Kaggle or Colab** (three gaps, stated plainly; `kaggle/10_run_stage.ipynb` is not changed):

1. The wheel in the workspace repo must be rebuilt and replaced (`uv build --wheel --out-dir dist .`, upload, keep
   exactly one): `msd.finetune` is new. `scripts/finetune.py`, `pipelines/` and `data/` (with `data/ft/` and
   `data/testset/`) must be current there as well.
2. The stage notebook pulls only `scripts/*`, `pipelines/*` and `data/**`, **not `configs/`**. Upload the two
   YAML files into the workspace repo as `data/configs/ft_lora.yaml` and `data/configs/ft_lora_strong.yaml`
   (the `data/**` pattern pulls them) and pass `--config data/configs/ft_lora.yaml`. The summary records that
   path and the file's sha256, which is the same as the one in `configs/`.
3. The notebook installs no toolkit. In the pushed copy of the notebook (not in the repo file) add a code cell
   above the run cell: `!pip install -q aligntune==0.1.31` (needs Python 3.11 or newer; 0.1.31 was published
   2026-11-05). It pulls torch/transformers/trl/peft pins (`transformers>=5.15,<6`, `trl==1.7.1`,
   `torch>=2.4,<2.14`) and may replace the preinstalled torch: run `python scripts/env_check.py` again afterwards
   (UNVERIFIED either way). Use this runtime only for the fine-tune, not for SafeTune (one environment per toolkit).

The token: the notebook already puts `HF_TOKEN` into the environment; the gated base model
(`CohereLabs/tiny-aya-global`, licence accepted on its page) is downloaded with it. `finetune.py` never prints it.
On a two-T4 Kaggle runtime the config uses the first card only (`aligntune.device_map: {"": 0}`).

Run these in order, each as a separate run of the stage notebook (`STAGE`, `STAGE_ARGS` in its second cell):

```
# 1. Plan on the runtime (a tokenizer can be loaded here, so the token lengths and the longest example are printed;
#    check that max_seq_length 2048 covers it). Writes nothing.
STAGE = "finetune"
STAGE_ARGS = "--config data/configs/ft_lora.yaml --dry-run"

# 2. Smoke run: 20 steps, every step logged, merges once so the merge path (CPU RAM, tokenizer save) is proven
#    BEFORE the long run. Output goes to results/finetune/ft_lora-smoke; it is not A1 and is never pushed.
STAGE_ARGS = "--config data/configs/ft_lora.yaml --max-steps 20 --merge"
#    Read the printed lines and results/finetune/ft_lora-smoke/finetune_summary.json: rows_seen must be 6000 (a lower
#    number means AlignTune/CuratorKIT dropped rows), loss_ok true, steps 20, peak memory in the log (UNVERIFIED how it
#    is shown; watch nvidia-smi). If the adapter check, the keyword names or the assistant-only mask are wrong this is
#    where it shows.

# 3. The real run: 2 epochs = 750 optimizer steps. Writes results/finetune/ft_lora/{adapter,merged,finetune_summary.json}.
STAGE_ARGS = "--config data/configs/ft_lora.yaml"
#    Optional, after a successful run only: add  --push-to <hf-user>/<private-repo>  (the repo must already exist and be
#    PRIVATE; HF_TOKEN from the environment; nothing is created, nothing is uploaded without the flag).

# 4. Only if step 3's held-out accuracy (below) shows the model barely moved: the strong retry, 3 epochs = 1,125 steps.
STAGE_ARGS = "--config data/configs/ft_lora_strong.yaml"
```

The notebook then uploads `results/finetune/` to the private workspace repo, merged weights included
(about 6.7 GB in fp16 plus the adapter); a failed run exits non-zero and uploads nothing it did not finish.

**Where things are written.** AlignTune trains into a persistent work folder, `work/finetune/<config stem>[-smoke]/`
(checkpoints every `save_steps`, the adapter files at the end). It is kept on ANY failure, including Ctrl-C, and the
error names it; a later run refuses to start while it holds anything but the chat-template file, so a failed run
cannot silently overwrite its own recovery files. After training the adapter is copied (without checkpoints) into its
own folder `<out>/adapter`, then the merged weights go into `<out>/merged`, each written under a hidden `.partial`
name and renamed only when whole, and `finetune_summary.json` is written last, so its presence means the run
finished. A successful run deletes the work folder (`output.keep_checkpoints: true` keeps it). A killed session
(SIGKILL, a dropped runtime) cannot clean up: a hidden `.<name>.<8 hex>.partial` folder stays and is named at the
start of the next run, never deleted.

**After a failure.** AlignTune itself has no resume: it has no switch to continue from a checkpoint, so a dropped run
starts again. What is kept is for recovery by hand.

- The merge failed (out of RAM, say): the error names `<out>/adapter`. Finish without training again:
  `STAGE_ARGS = "--config data/configs/ft_lora.yaml --merge-only --adapter results/finetune/ft_lora/adapter"`. It
  checks that adapter against the config (base model, rank, alpha, target modules), merges into
  `<out>/merged` and adds a `merge` section to `finetune_summary.json` (it writes a small one when the failed run
  left none). It trains nothing and reads no data; it takes no `--data`, `--seed`, `--max-steps`, `--no-merge`,
  `--dry-run` or `--push-to`.
- Training failed or the session was interrupted: look in the named work folder for `checkpoint-<step>/`. Continuing
  one needs your own script (`Trainer.train(resume_from_checkpoint=<folder>)`; UNVERIFIED that it works through
  AlignTune's objects). Otherwise move or delete the work folder and run again.

**Held-out accuracy check, A0 against A1** (the task items only; no judge and no API key). `run_eval.py` reads
`data/ft/taskdev.jsonl` for `--split dev` and `data/ft/heldout.jsonl` for `--split report` and scores the bare answer
letter with `--stages generate,task`. The generate stage also writes the safety completions of that split (the
study needs them for A0 and A1 anyway); `--limit N` caps both for a quick look. Use `--langs en,hi,ar,sw` (hinglish
has no task rows). Run the dev split first, A0 then A1, then the report split once:

```
STAGE = "run_eval"
STAGE_ARGS = "--arm A0 --split dev --model CohereLabs/tiny-aya-global --out results/run_eval --stages generate,task --langs en,hi,ar,sw"
STAGE_ARGS = "--arm A1 --split dev --model results/finetune/ft_lora/merged --out results/run_eval --stages generate,task --langs en,hi,ar,sw"
# then the same two with --split report
```

(A later notebook run starts from a fresh working folder and the notebook pulls only scripts, pipelines and data,
so fetch the merged model first in an extra cell, `snapshot_download(WORKSPACE_REPO, repo_type="dataset",
local_dir=".", token=token, allow_patterns=["results/finetune/ft_lora/merged/*"])`, or pass the `--push-to` repo id
as `--model`.) Read `results/run_eval/dev/<arm>/none/<lang>.eval.json`: its `task` block has `n`, `correct` and
`accuracy` per language. Compare with the always-D floor in `docs/toolkit-notes.md` (31% on dev, 27 to 28% on the
heldout split). What A1 must show is that it answers the bare letter more often than A0 does (the point of the
task); how much is a result, not a gate here.

**Expected cost, all UNVERIFIED estimates** (nothing measured): with batch 1 and gradient checkpointing a step
processes 16 sequences of mostly 500 to 1,500 tokens; AlignTune's own heuristic puts the base config at 1.7 h
(1,024-token sequences) to 3.4 h (2,048) on a T4 (formulas and table in the notes, with its +/-30%), so plan 2 to
4 hours for the base config and 3 to 6 hours for the strong one, the 20-step smoke run at a few minutes plus model
download (6.7 GB) and a merge of a few minutes. Memory: about 13 to 14 GB at the longest examples against 16 GB
(arithmetic in the notes). If step 2 runs out of memory, lower `train.max_seq_length` only to what the dry run's
longest example needs, never below it (the run refuses a shorter limit); `per_device_batch_size` is already 1. A
Kaggle or Colab session limit (section 3 lists only what was read) may cut a 4 to 6 hour run: the strong retry is the
one at risk, and there is no resume.
