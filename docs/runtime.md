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
5. **Add `HF_TOKEN` as a secret inside the notebook site, once** (Kaggle cannot do this from the
   command line, see section 2):
   - Kaggle: open a notebook > Add-ons > Secrets > add a secret labelled `HF_TOKEN`. Also switch
     Internet on and pick the GPU in the notebook settings.
   - Colab: the key icon in the left bar (Secrets) > add `HF_TOKEN` and switch "Notebook access"
     on (exact Colab menu wording from memory, UNVERIFIED).
6. **Later, when I ask:** say yes to creating the private Hugging Face dataset repo
   `<your-hf-username>/msd-workspace` (section 2). Nothing is created on Hugging Face, Kaggle or
   Colab until then.

## 2. Workspace design (not built yet)

One **private Hugging Face dataset repo**, `<owner>/msd-workspace`, holds everything the runtimes
need, and both Kaggle and Colab pull from it and push to it with `HF_TOKEN`:

```
wheels/msd-<version>-py3-none-any.whl   the code (keep exactly one wheel here)
scripts/                                copy of this repo's scripts/ (the stage scripts)
data/                                   test set, fine-tune data
results/<stage>/                        per-item JSONL, logs, written by the notebooks
```

Two thin notebooks live in `kaggle/` (they work on Colab too, nothing is Kaggle-only):
`00_env_check.ipynb` (install the wheel, print the environment JSON) and `10_run_stage.ipynb` (set
`STAGE`, run `python scripts/<STAGE>.py`, push `results/<STAGE>/`). Later tasks add the stage
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
secrets field. So the `HF_TOKEN` secret is added once in the Kaggle web page (checklist step 5).
UNVERIFIED: whether the secret stays attached to the notebook after a later `kaggle kernels push`
(one third-party search snippet claimed it may not; I could not open that page). The first cell of
both notebooks stops with the message "HF_TOKEN not found" if it is missing, so a failed push shows
up in seconds.

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
`aligntune` 0.1.30 >=3.11, `curatorkit` 0.1.22 >=3.11, `auditkit` 1.1.3 >=3.10. Our own `msd` needs >=3.12.

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
token it reports `200 ok` once the license is accepted, or `403 forbidden` if it is not (403 is the
usual Hugging Face answer for "token fine, license not accepted"; not seen yet: UNVERIFIED).

Waiting for the owner (all DEFERRED-RUNTIME):

```
# 1. Create the private workspace repo and upload the wheel + scripts (needs the owner's yes and HF_TOKEN):
python -c "from huggingface_hub import HfApi; a=HfApi(); r='<hf-user>/msd-workspace'; \
a.create_repo(r, repo_type='dataset', private=True); \
a.upload_file(path_or_fileobj='dist/msd-0.1.0-py3-none-any.whl', path_in_repo='wheels/msd-0.1.0-py3-none-any.whl', repo_id=r, repo_type='dataset'); \
a.upload_folder(folder_path='scripts', path_in_repo='scripts', repo_id=r, repo_type='dataset')"
# (huggingface_hub is not installed in envs/core; the exact call signatures are from memory: UNVERIFIED.)

# 2. Kaggle: put the WORKSPACE_REPO name into a copy of kaggle/00_env_check.ipynb, add kernel-metadata.json
#    (template in section 2, code_file 00_env_check.ipynb), then
kaggle kernels push -p <folder-with-notebook-and-metadata>
kaggle kernels status <kaggle-username>/<slug>
kaggle kernels output <kaggle-username>/<slug> -p <outdir>      # the JSON is in the notebook log

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
