# Toolkit notes

Facts found by running the Lexsi toolkits, for the article and for whoever reruns this. Short on
purpose; each section says when it was checked.

## CuratorKIT 0.1.22 (checked 2026-11-05)

Installed in its own venv, `envs/curatorkit`, with no extras (no `[pdf]`: it pulls MinerU, AGPL).
Everything below uses only that base install: pydantic, typer, tqdm, pyyaml.

### Commands (repository root)

```
python scripts/build_ft_data.py fetch      # public datasets-server, no token; 10-15 min for 4 languages
python scripts/build_ft_data.py prepare    # data/raw/... -> data/ft/prep/candidates.jsonl
envs/curatorkit/bin/curatorkit run pipelines/ft_data.yaml --output-dir data/ft/curated --dry-run   # plan only
envs/curatorkit/bin/curatorkit run pipelines/ft_data.yaml --output-dir data/ft/curated             # ~70 s, no LLM call
python scripts/build_ft_data.py finalize   # joins the export back to ids -> data/ft/train.jsonl, heldout.jsonl, taskdev.jsonl
python scripts/validate_ft_data.py --langs en,hi,ar,sw   # checks the subject allowlist of pipelines/ft_data.yaml by default
```

`--dry-run` only parses the YAML and prints the steps; it does not open the input file.

### What the output folder offers

`--output-dir` receives: `sft_messages.jsonl` (one `{"messages": [...]}` per kept sample),
`rejected.jsonl` (every gate rejection with `rejection_reason` and `rejecting_step`, and the sample's
extra columns under `metadata`), `manifest.json` (per-step counts, dedup and sampler stats),
`dataset_card.md`, `README.md`, `lexsi_provenance.json`, `checksums.txt`.

The `README.md` declares the Hugging Face dataset config: it is named after the exporter file, so
`sft_messages` (default), with a single split called `train`. Checked with `datasets` 5.1.0 in a
throwaway venv (`datasets` is not installed in any project venv):

```
datasets.get_dataset_config_names("data/ft/curated")        -> ['sft_messages']
datasets.load_dataset("data/ft/curated", "sft_messages")    -> DatasetDict({train: 8986 rows, features: ['messages']})
datasets.load_dataset("data/ft/curated")                    -> the same
datasets.load_dataset("data/ft/curated", "sft_alpaca")      -> ValueError (only exporters in the YAML get a config)
```

So the config name the docs leave out is the exporter's file stem, and it works. The folder is a
curation result, not our release format: the split is always `train` and the only feature is
`messages`.

### Gotchas that shaped the pipeline

- Exporters write only `messages`. The sample id, language and gold answer are not exported, so
  `finalize` matches each exported line back to its candidate by the exact (user, assistant) text.
  That works because dedup keeps one copy of any text and the YAML adds no text cleaning.
- CuratorKIT cannot build the closed-form prompt and has no language-id gate and no single-field
  gate (`schema` measures `instruction + output` together, in whitespace tokens). So
  `scripts/build_ft_data.py prepare` builds the prompt and applies the script check and the
  harmless-subject allowlist before CuratorKIT, and `validate_ft_data.py` re-checks both after
  (the allowlist check is on by default there; only `--no-subject-check` turns it off, and the
  output then says `subject allowlist: NOT checked`).
  CuratorKIT does the reading, the schema/length gate, exact and MinHash dedup, language
  equalising and the export.
- `stage_counts` in `manifest.json` is keyed by step name, so two `stratified_sampler` entries
  overwrite each other (only the last is listed). We use two on purpose (train strata, heldout
  strata); per-stratum numbers come from our own join instead.
- `stratified_sampler` cuts every listed category down to the smallest one (it never pads). A
  category with no samples is skipped with a warning, not an error.
- `minhash_dedup` at 0.85 judges the whole `instruction + output` text, so it is not an
  exact-duplicate filter. It removed 146 of 9,185 samples here (train, heldout and taskdev rows
  together). The two dedup steps and the samplers removed 207 samples in all (8 exact copies, 146
  near-duplicates, 53 by the samplers); of the 199 whose text no longer survives, 142 shared a
  source passage with a kept sample
  (different questions about the same text) and some others were mirror-image questions
  (kinetic energy doubles / momentum doubles). `ignore_for_dedup` masks the fixed option labels
  and instruction lines so they do not inflate similarity. Train rows are read first, so a
  heldout sample that resembles a train sample is the one dropped.
- `output_split` shuffles randomly, so it cannot make a held-out set that is disjoint by source
  item; our split is made before CuratorKIT, by item and passage.
- The generation, hallucination and reward steps need an LLM endpoint and key (LiteLLM); none exists,
  so `pipelines/ft_data.yaml` has only a commented stub where a generator would go.

## Fetching Global-MMLU rows (checked 2026-11-05)

No `pyarrow` or `datasets` in the core venv, so rows come from the Hugging Face datasets-server
`/rows` endpoint (public, 100 rows a page, about 141 pages per language for the 14,042-row test
split). It answers HTTP 429 after a few hundred requests in a few minutes; `fetch` honours
`Retry-After` and uses 3 workers. The `/filter` endpoint returned HTTP 500 and timeouts, so subjects
are filtered locally.

## Provenance of the fine-tuning data (checked 2026-11-05)

What the rebuilt `data/ft/` came from, so the same files can be rebuilt and compared. `data/` is
not committed, so this is the record.

- Dataset: `CohereLabs/Global-MMLU`, test split, configs en / hi / ar / sw, Apache-2.0 (raw card).
  Hub revision (head commit read from `https://huggingface.co/api/datasets/CohereLabs/Global-MMLU`,
  no token): `0e619dbeb34206cd48705a1a0ea7fb21cae09993`, last modified 2025-08-14T20:12:53Z. The
  datasets-server `/rows` endpoint does not say which commit it served. The head commit predates
  the fetch by more than a year, so it is the revision the rows came from. `fetch` now records
  it in `FETCH.json` under `revision` (the existing `FETCH.json` was patched with the same value).
- Raw files, `data/raw/coherelabs-global-mmlu-fine-tune-subset/`, 3,793 rows each, fetched with the
  first 16-subject allowlist. `prepare` re-filters them to the 14 subjects in
  `pipelines/ft_data.yaml` (it dropped 466 rows per language, the sociology and clinical_knowledge
  rows), so the raw files still hold those two subjects and `FETCH.json` still lists 16.

  | file | sha256 |
  | --- | --- |
  | `en_test.jsonl` | `163e2eaf35e29f1621502ed2cd9a2e35e52cff195940e1326decdad4ebe946e3` |
  | `hi_test.jsonl` | `f1a779028156461146ae2f372f9701fb5724bdd4a367dc34510b7b02bd5ad476` |
  | `ar_test.jsonl` | `3dd93c3118af6fcace5b1cf4a0610e01d179e7fc6021593b39d3ba877dbf73af` |
  | `sw_test.jsonl` | `84f5a96c73c893afe00a0ab97927b0483379d2f168788f579cd7106a7a124167` |

- Expected output of `prepare` -> CuratorKIT -> `finalize` on those files with the committed
  `pipelines/ft_data.yaml` (CuratorKIT 0.1.22, seed 20261105; a second full run gave identical
  bytes, only the manifest timestamps and the provenance chain differ):

  | file | rows | sha256 |
  | --- | --- | --- |
  | `data/ft/train.jsonl` | 6,000 (1,500 per language) | `348c164f35d092d4a237872556c0a59d866ba6eea3367126ea33807b5401ce80` |
  | `data/ft/heldout.jsonl` | 800 (200 per language) | `c90d3ee401d2740911c1895885c15fdcc0edf2516baec2266c47e10f660899bc` |
  | `data/ft/taskdev.jsonl` | 800 (200 per language, the same 200 items in all four) | `cb3cc328d186131d4d2720771c6acf8e7268f3c724403c54f37bdb42081d424b` |

  CuratorKIT's own export, `data/ft/curated/sft_messages.jsonl`, has 8,986 rows (6,964 train, 996
  heldout and 1,026 taskdev before the per-language caps) and 47 gate rejections (all
  `above_max_tokens`, one of them a taskdev row). The `train.jsonl` and `heldout.jsonl` hashes are
  the ones from before `taskdev.jsonl` existed: adding it changed neither file (compared with
  `cmp` on the real data, and by `test_adding_taskdev_to_the_settings_does_not_change_train_or_heldout_bytes`).

### The dev task file, `data/ft/taskdev.jsonl` (built 2026-11-05)

The task items alpha is chosen on (the dev split), so that the report split can keep all 200
`heldout.jsonl` items per language. `src/msd/evalrun.py` reads it as `TASK_FILES["dev"]` and the
records have exactly the heldout shape, `{id, lang, messages: [user, assistant], gold}`; the real
file loads through `evalrun.read_task_records` and `task_prompts(..., split="dev")` (checked).

- Drawn from the allowlisted items that neither train nor heldout took: `prepare` ranks the items
  once with the split seed and fills heldout, then train, then (a third pass) taskdev up to
  `taskdev_pool: 260` from what is left. Whole passage groups only, so a passage never straddles
  two splits. Pool and cap are `taskdev_pool` and `taskdev_cap: 200` in `pipelines/ft_data.yaml`.
- Read by CuratorKIT after train and heldout, so a taskdev item that near-duplicates either is the
  one dropped (13 taskdev candidates were removed this way or by the length gate). `finalize`
  then keeps, per language, the best-ranked items that survived in every language, so all four
  languages hold the same 200 items (they did; if too few were shared a language would fill up
  from its own survivors and the validator would warn, never fail, about differing item sets).
- Pool and result per language (en / hi / ar / sw): 260 / 260 / 260 / 259 candidates, 256 / 257 /
  258 / 255 CuratorKIT survivors, **200 / 200 / 200 / 200** final. No shortfall.
- Disjointness, checked on the real files with code independent of the validator: 0 items shared
  with `train.jsonl` or `heldout.jsonl`, and also 0 with the whole train and heldout pools
  (including the rows capped off); 0 shared normalised prompts; the 30 distinct (language,
  passage) pairs in taskdev (12% of rows carry a passage) appear in neither file.
- What that does and does not rule out: every overlap check (the validator's and the independent
  one) is an exact match after normalisation (`norm_text`: NFKC, case, whitespace, Arabic marks),
  on the source item id, the whole prompt, and the passage. "0 shared normalised prompts" therefore
  does not mean "no repeated questions": the whole prompt includes the options, so the same
  question asked with a different option set is a different prompt. There is one such source-dataset
  twin, found by comparing the question text above the options: `astronomy/test/9` (taskdev) and
  `astronomy/test/116` (train) have the same question stem and the same correct answer (the comet
  nuclei option, letter D in both) in en, ar and sw (the Hindi stems are translated differently
  and do not match), because Global-MMLU itself repeats the question with a slightly different set
  of options. It is 1 of the 200 items; it is a different item id, so no rule can call it a leak,
  and CuratorKIT's MinHash (0.85) did not drop it, most likely because the option sets differ (not
  checked further). An exact-stem scan of
  all four languages against train and heldout found no other.
- Composition: 14 subjects, psychology 30, biology 22, human aging 19, nutrition 18, prehistory
  16, physics 15, astronomy 14, geography 13, government, world history, anatomy and US history
  11 each, global facts 7, European history 2 (the pool is a random draw, so the mix is whatever
  was left). Gold letters A / B / C / D = 44 / 45 / 49 / 62, the same in every language because
  the items are the same: a model that always answers D scores 31% on dev, a little above the
  27-28% floor of the heldout split. Report the dev accuracy against that floor.
- The validator checks it (`scripts/validate_ft_data.py`, exit code 19 when it shares a prompt,
  item or passage with train or heldout); see the module docstring of `src/msd/ftdata.py`.

## Judge candidates (checked 2026-11-05)

What the judge does: read one request and one reply (any of en, hi, hinglish, ar, sw) and answer
`complied` or `refused` for a harmful or a benign-but-edgy request. That is general instruction
following in five languages, so the candidates are multilingual instruct models, not guard
models that output safe/unsafe. Claude through the API is one family (`anthropic:<model-id>`); the
open judge (`hf:<model-id>`, `msd.judges.HFLocalJudge`) must come from another family for the
reliability check (design doc, section 6). Judges may be any model (organizer ruling, 2026-11-03).

How this was read: each Hugging Face model card (and the Qwen3 blog) was fetched on 2026-11-05 with a
page-to-text tool that summarises; the quoted sentences are as it returned them. A fact I could not
read on a card is marked UNVERIFIED. No accuracy is claimed for any model: nothing was run (no GPU,
no download); measuring that is the reliability check's job.

| | `Qwen/Qwen3-4B` | `google/gemma-4-E4B-it` | `meta-llama/Llama-3.1-8B-Instruct` | `CohereLabs/aya-expanse-8b` |
| --- | --- | --- | --- | --- |
| Role | **primary open judge** | **alternative** | not recommended | not recommended |
| Family / developer | Qwen3, Qwen team | Gemma 4, Google DeepMind | Llama 3.1, Meta | Aya Expanse, Cohere Labs |
| Licence (card) | apache-2.0 | Apache 2.0 | Llama 3.1 Community License | cc-by-nc-4.0 plus Cohere's Acceptable Use Policy |
| Access | open | no gate shown on the page (UNVERIFIED) | gated, accept on Hugging Face | gated, accept on Hugging Face |
| Size | 4.0B | 8B with embeddings, 4.5B effective | 8B | 8B |
| Languages the card names | "Support of 100+ languages and dialects"; no list on the card | "multilingual support in over 140 languages" and "Out-of-the-box support for 35+ languages, pre-trained on 140+ languages"; no list on the card | "English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai" | 23 named: Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, Vietnamese |
| en | not named on the card; named in the Qwen3 blog list | not named on the card (UNVERIFIED) | yes | yes |
| hi | not named on the card; named in the blog list | not named on the card (UNVERIFIED) | yes | yes |
| hinglish | not stated anywhere I read | not stated | not stated | not stated |
| ar | not named on the card; named in the blog list ("Arabic (Standard, Najdi, Levantine, Egyptian, ...)") | not named on the card (UNVERIFIED) | no | yes |
| sw | not named on the card; named in the blog list ("Swahili", under "Other") | not named on the card (UNVERIFIED) | no | no |
| Thinking mode | hybrid, on by default; `enable_thinking=False` switches it off (documented on the card) | on only with `<|think|>` in the system prompt; the card's example passes `enable_thinking=False` | none | none |
| Weights in 16-bit | about 8 GB (4.0B x 2 bytes) | about 16 GB as stored (8B BF16 incl. embeddings); whether the embeddings must sit on the GPU is UNVERIFIED | about 16 GB | about 16 GB |

Sources: `huggingface.co/Qwen/Qwen3-4B` and `/Qwen/Qwen3-8B` (same languages sentence; "119 languages
and dialects" is from the blog, `qwenlm.github.io/blog/qwen3/`, which lists the languages by family),
`/google/gemma-4-E4B-it` and `/google/gemma-4-12B-it`, `/meta-llama/Llama-3.1-8B-Instruct`,
`/CohereLabs/aya-expanse-8b`. The Llama card also lists "use in languages beyond those explicitly
referenced as supported" as out of scope, so Arabic and Swahili there are outside what Meta supports.

### Recommendation

- **Primary: `Qwen/Qwen3-4B`** (Apache-2.0). It is the only candidate whose official material I could
  read that names Hindi, Arabic and Swahili as well as English. That is the blog list, not the card:
  the card says only "100+ languages and dialects", so cite the blog if the article says this.
  The `-Instruct-2507` variant has no language statement on its card beyond "Substantial gains in
  long-tail knowledge coverage across multiple languages", so its coverage is UNVERIFIED; use the
  hybrid `Qwen3-4B`. At 4B, 16-bit weights are about 8 GB, so it should fit one Kaggle or Colab T4
  (about 15 GB expected; the memory per card is UNVERIFIED until `env_check` reports it,
  `docs/runtime.md`). `HFLocalJudge` passes `enable_thinking=False`, which this card documents,
  so the 8 new tokens are the answer.
- **Alternative: `google/gemma-4-E4B-it`** (Apache-2.0, a third family). Use it if the stand-in
  translator of the test set turns out to be a Qwen model (the judge should not share a family with
  the model that wrote the prompts), or if Qwen3-4B fails the reliability check in one language. Its
  card names no individual language, so which of our five it covers is UNVERIFIED; its 16 GB of stored
  weights may not fit one T4 in 16-bit (UNVERIFIED).
- **Gaps, stated plainly.** Hinglish (romanized Hindi) is not named for any candidate, and no card
  gives a measured score in our five languages. Expect the weakest judging in Swahili and Hinglish;
  that is what the per-language kappa in the reliability check measures, and a language under kappa
  0.6 is marked low-confidence, not hidden.
- **Not recommended.** `Llama-3.1-8B-Instruct`: lacks Arabic and Swahili in its supported list, gated,
  custom licence. `aya-expanse-8b`: same lab and family as the model under test (Tiny Aya), so the judge
  and the subject would share blind spots; also non-commercial and without Swahili. Bigger models
  (`Qwen3-8B`, `gemma-4-12B-it`, `Qwen3.8-27B`) need more than one T4 or quantisation; `HFLocalJudge`
  loads one device in 16-bit and does neither. `Qwen3.8-27B`'s card has no languages statement.

Not done (DEFERRED-RUNTIME): the brief's "ten items per language on the GPU runtime" check, because no
GPU runtime exists yet. It is the first thing to run once one does: the judge stage of
`scripts/run_eval.py` with `--judge hf:Qwen/Qwen3-4B --limit 10` on a runtime with `torch` and
`transformers`, reading the verdicts for plain `judge_error` counts before anything else.

## SafeTune ReSta (read from source)

Checked 2026-11-05. Nothing was run: SafeTune is not installed in any project venv, there is no GPU
here and no weights were downloaded. Everything below is read from source text. The source files
are the `safetune` 0.1.12 wheel from PyPI (sha256 `91de846c2e0e386f52009a93371a065185e261bce4c39fef7bbedb3ee12534d7`,
unpacked, not installed) and the same paths at the head of `github.com/Lexsi-Labs/SafeTune` (commit
`3901a9f48a`, 2026-11-03). The ReSta files are byte-identical in both. Files read:
`src/safetune/runner/recover/_layer.py` (`ReStaTrainer`), `.../_base.py` (`_RecoverBase`),
`src/safetune/interventions/recover/resta.py` (`apply_resta`), `src/safetune/core/optim/resta.py`
(`RESTAWrapper`), `.../recover/_invariant.py`, `src/safetune/runner/utils/model_utils.py`,
`src/safetune/config.py`, and the pages `docs/user-guide/recover/layer/resta.md`,
`docs/reference/api/recover.md`. Anything I could not read directly is marked UNVERIFIED.

Package facts (PyPI JSON): needs Python >=3.12, `torch>=2.7.0`, `transformers>=5.15,<6`, and drags in
bitsandbytes, trl, peft, wandb, lm-eval and others. `from safetune.runner import recover` also imports
the harden, steer and unlearn pillars (`runner/__init__.py`), so the import is heavy; whether it
imports cleanly on Kaggle or Colab is UNVERIFIED.

### Constructor and call

```python
from safetune.runner import recover
recover.ReStaTrainer(model=None, *, base_model=None, aligned_model=None, alpha=1.0, dare=True,
                     dare_drop_rate=None, dare_seed=0, device=None, **kwargs)   # kwargs: model_id, results_dir, drift_task
patched = trainer.apply()        # apply(*, alpha=None, dare=None, dare_drop_rate=None, dare_seed=None, device=None)
```

- **It takes loaded `nn.Module` objects, not paths or ids.** If `model` is omitted, the base class
  loads `model_id` itself with SafeTune's `load_model`, which is not what we want (next bullets).
- **`apply()` mutates the drifted model in place and returns that same object. It never writes to
  disk** (`_RecoverBase.apply` docstring: "Never write to disk here; the caller calls
  `save_checkpoint` separately"). A sweep over alphas therefore needs a fresh drifted model per
  alpha (reload or deepcopy); applying twice stacks the patch.
- **Saving** is a separate call, `trainer.save_checkpoint(patched, tokenizer, name)`. It writes to
  `<results_dir>/checkpoints/<name>` (the default `results_dir` is inside the installed package), calls
  `save_pretrained(..., safe_serialization=True)`, the tokenizer's `save_pretrained`, and adds a
  `lexsi_provenance.json`. Our wrapper does not use it: it saves with `save_pretrained` itself, into a
  temporary folder it renames into place, so the output path and the tokenizer source are ours.
- The loaders SafeTune ships (`load_model`, `load_model_cpu`) default their dtype to the runtime
  config: fp32 on CPU, bf16 on a bf16-capable GPU. Used as is, they would load A1 as fp32 and save
  an fp32 A2 next to a bf16 A1. Our wrapper loads the three models itself with `dtype="auto"` (each
  in the dtype its own config says), so a patched model is saved in the dtype of the drifted one. That
  `dtype="auto"` keeps the stored dtype in the transformers version we get is UNVERIFIED.

### What `alpha` multiplies

```
safety vector    v = theta_aligned - theta_base                      (fp32, one tensor at a time)
no DARE          theta_safe = theta_drifted + alpha * v
DARE (default)   theta_safe = theta_drifted + alpha * dare(v)        dare keeps each entry with prob 1-p, scaled by 1/(1-p)
```

The code, per tensor: `val.copy_((val.float() + alpha * delta).to(val.dtype))`. So:

- `alpha = 0` adds exactly zero: the result is the drifted model, entry by entry, bit for bit, because
  the fp32 round trip of a bf16 or fp16 value is exact. It is NOT a no-op in code (it still computes
  every delta, so it costs the same time), and `assert_mutates` logs a WARNING that "model weights
  appear unchanged" (a log line, not an exception; expected at alpha 0).
- `alpha = 1` adds the whole `aligned - base` vector. It is not a "restore A0" setting; see the next section.
- `v` is the whole post-training step that turned Tiny Aya Base into Tiny Aya Global, not a
  safety-only vector. Nothing in the code separates safety from the rest of post-training.
- **DARE is on by default in `ReStaTrainer`** (`dare=True`, `dare_drop_rate=None` meaning p = 0.3, the
  paper's value, `dare_seed=0`), and off by default in the lower-level `apply_resta` (`dare=False`). The
  docs page states both. With DARE the mask is drawn per tensor in the order of `aligned.state_dict()`
  from a CPU generator seeded with `dare_seed`, so a fixed seed repeats on the same library and
  transformers version, and the draw order is UNVERIFIED across versions. With `dare_seed=None` the
  global RNG is used (not repeatable). The mask applies to every tensor in the vector, embeddings and
  norms included; `ReStaTrainer` does not expose `param_filter`.
- The docs report that at p = 0.9 (the old default, "legacy") a 0.5B Qwen model broke (garbled answers)
  while p = 0.3 restored refusal; the default is now 0.3. Not reproduced here (UNVERIFIED).

### Reading it against our two arms (arithmetic, not a measurement)

Write A1 = A0 + L for the merged fine-tune step L (zero outside the LoRA target modules). Then
ReSta gives `A1 + alpha*(A0 - BASE)`, which at alpha = 1 is `2*A0 - BASE + L`: it adds a second copy of
the Base-to-Global step on top of a model that already contains one, and leaves L in place. The naive blend
`(1-t)*A1 + t*A0` is `A0 + (1-t)*L`: it removes L, and t = 1 returns A0 exactly. So ReSta at alpha = 1 is
not A0 and not near it, and tensors the fine-tune did not touch (embeddings, norms) are moved past A0
by ReSta and left alone by the blend. SafeTune's own pages say the same in words, as a measurement
on Tiny Aya (the `ReStaTrainer` docstring and the ReSta guide, both quoted as read): "On Tiny Aya,
alpha=1 breaks the model; use alpha~0.25 (sweep: 0.1/0.25 restore refusal with normal answers, >=0.5
breaks it)". We have not reproduced that; which Tiny Aya model, drift and test it comes from is not
stated (UNVERIFIED). It matters for us because the first grid, (0, 0.25, 0.5, 0.75, 1.0), had three of five
points in the range SafeTune reports as breaking; the grid was changed (next section). Read the benign
answers and the task accuracy of every ReSta arm before reading its safety numbers.

### The alpha grid

`config.ALPHAS` is the ReSta sweep (arm A2) only: `(0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0)`. The naive
blend (arm A2n) keeps its own settings, t in (0.25, 0.5, 0.75), which are not this grid.

- **Evidence.** SafeTune's ReSta guide (`docs/user-guide/recover/layer/resta.md`, read 2026-11-05) says: "On Tiny
  Aya, alpha=1 breaks the model; use alpha~0.25 (sweep: 0.1/0.25 restore refusal with normal answers, >=0.5
  breaks it)". It is SafeTune's report, not our measurement, and which Tiny Aya model and drift it used is not
  stated.
- **Reasoning.** Our A1 already contains the Base-to-Global step once (A1 = A0 + L), and ReSta adds
  `alpha * (A0 - BASE)` on top of it, so every alpha above zero over-applies that step; a model-breaking
  alpha of 0.5 or more is what that arithmetic would lead one to expect. The useful range is therefore the
  low end, and the old grid had one point (0.25) there and three above the reported break.
- **The grid.** Dense below 0.5 (0.05, 0.1, 0.15, 0.25, 0.35), 0.5 as the first reported-broken point, 0 as the
  no-op check, and 1.0 kept as a deliberate break control, so the article can show the break instead of
  asserting it. The folder names (`gen.format_alpha`) read back exactly: 0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5,
  1.0 (tested).
- **Staging at the runtime (plan Task 16, Step 7).** Eight ReSta runs plus three blends, each generated and judged
  in five languages, is a lot for free GPU quotas. Run alpha 0 (the Step 6 sanity check), then 0.1, 0.25 and
  0.5 first, with a benign-answer sanity check on each (read some non-harmful answers: the failure SafeTune
  describes is a model that still refuses but talks garbage). Add 0.05, 0.15 and 0.35 only if the first
  three leave a gap in the safety-against-task-quality curve; run 1.0 last, as the control. Pick the featured
  alpha on the dev split only.

### Memory and device

- The three models must be loaded in full. In bf16 that is about 3 x 6.7 GB = 20 GB for a 3.35B model
  (the plan's figure, Task 16, Step 6; `docs/runtime.md` has no RAM number for either free runtime); in
  fp32 it would be 3 x 13.4 GB = 40 GB. The safety vector itself is streamed: the extra memory is a few
  fp32 copies of the largest tensor (probably the embedding matrix, size UNVERIFIED), which with DARE
  includes the random mask and the masked product. SafeTune's docs say earlier versions instead held about
  three fp32 copies of the whole model for the vector, "about 40 GB for a 3.35B model", on top of the loaded
  models; the streaming change removed that.
- `device=None` computes each delta on the drifted weight's device, so with the drifted model on a
  16 GB GPU and the two references on CPU it moves tensors across per step; `device="cpu"` with all
  three on CPU needs the 20 GB of RAM and no GPU. Our wrapper loads all three on CPU and passes
  `device="cpu"`. Whether a free Kaggle or Colab runtime has that much RAM is UNVERIFIED (`docs/runtime.md`).

### Things that could make `alpha = 0` differ from the drifted model

The source offers no reason when the three state dicts line up. These are the ways it could still differ,
and why the wrapper's sanity step (greedy outputs at alpha 0 equal A1 on 5 prompts) stays mandatory:

1. **Key mismatch is silent.** The vector covers `[k for k in aligned if k in base]`, and a tensor whose
   name is missing from either reference is simply skipped, with only an INFO log line (the count). If
   Tiny Aya Base and Global name their tensors differently (UNVERIFIED), ReSta would patch fewer
   tensors and say nothing. Our wrapper refuses unless all three models have the same tensor names and shapes
   (when it checks, and one false alarm it can raise, is under "How `scripts/run_resta.py` calls it").
2. **Load and save round trip.** `apply()` is exact, but the model is loaded and saved by transformers:
   a dtype cast on load, a renamed or fused key, tied weights written once. The bits can change before
   ReSta touches them. Compare the tensors of the alpha-0 output with those of `--drifted`, name by name
   and bit by bit (a few lines with `safetensors`; no such helper exists in the repo yet); the 5-prompt check alone would not catch a small dtype change.
3. **Non-finite entries.** `0 * inf` and `0 * nan` are NaN. If `aligned - base` held an inf or nan, alpha 0
   would corrupt that entry. Not expected in released bf16 weights; not checked.
4. **Tied weights.** If the embedding and the output head are tied, the name that wins is the last
   one in the drifted state dict (the code says so); at alpha 0 that changes nothing.
5. **DARE with a fresh seed** does not matter at alpha 0 (the delta is multiplied by zero).

### How `scripts/run_resta.py` calls it (written from the source above, never run against it)

`src/msd/restarun.py` does, in this order: check the arguments (exit 2: `--alpha` finite in [0, 1], `--out`
missing or empty, not a symbolic link and not equal to or inside the drifted folder or a local `--base` /
`--aligned` folder, `--drifted` a folder, `--base` and `--aligned` not the same model); name any stale
`.<out>.<8 hex>.partial` folder left next to `--out` by a killed earlier run (one line, nothing deleted);
compare tensor names and shapes of the LOCAL folders from their safetensors headers, before any model is
loaded (the drifted folder against each of `--base` / `--aligned` that is a folder; a folder with no
`model.safetensors` is skipped); import SafeTune (exit 1 with one line if it is missing, before any weight is
read); load the drifted model, then the base, and compare them as loaded (names and shapes of the state dicts),
then load the aligned model and compare that one too, so a wrong Hub reference stops the run after the second
load and not the third (each load is a download of the model's full size); the comparison after loading is kept
even when the headers agreed, because tied weights and renamed keys can make a loaded state dict differ from its
file. The header comparison is strict and can refuse a pair that would have loaded to the same state dict, for
example a tied `lm_head.weight` stored in one folder and not in the other; if that happens for Tiny Aya Base
against Global, the fix is to compare after loading only for that pair, and the refusal text says which check
fired ("compared from the safetensors headers" means this one). Then build `recover.ReStaTrainer(drifted, base_model=..., aligned_model=..., alpha=...,
dare=..., dare_drop_rate=..., dare_seed=..., device="cpu")` and call `apply()`; save with `save_pretrained`
plus the drifted model's tokenizer into a temporary folder renamed to `--out`, with `resta_patch.json`
recording alpha, the DARE settings, the three sources, the SafeTune version and the dtypes.

One decision the brief did not make: **DARE is on by default** because `ReStaTrainer` has it on, with
`dare_drop_rate=0.3` and `dare_seed=0` passed explicitly (`--no-dare`, `--dare-drop-rate`, `--dare-seed` change
them). The brief's call `ReStaTrainer(..., alpha=alpha).apply()` leaves all of this to the library's defaults, so an
article that says "ReSta" should say whether DARE was on. The `ReStaTrainer` page in the SafeTune docs shows
`dare=False` in its example while the class default is `True`.

### Running it (DEFERRED-RUNTIME: plan Task 16, Steps 5 to 7)

On a runtime that has the three models, enough RAM (about 20 GB for bf16) and `pip install safetune`:

```
export HF_TOKEN=...      # in the shell; there is no token flag
python scripts/run_resta.py --drifted A1 --base CohereLabs/tiny-aya-base --aligned CohereLabs/tiny-aya-global --alpha 0 --out A2_alpha_0.00
# Step 6: the tensors of A2_alpha_0.00 must equal A1's bit for bit, and greedy outputs must match on 5 prompts.
# Only then: --alpha 0.1, 0.25, 0.5 with a benign-answer check, then the rest of config.ALPHAS as needed
# ("The alpha grid" above; each run reloads the three models).
python -c "from pathlib import Path; from msd.blend import blend_state_dicts; blend_state_dicts(Path('A1'), Path('A0'), 0.5, Path('A2n_0.50'))"
```

The naive blend needs no SafeTune and no GPU: `msd.blend` reads shard headers first (names, shapes, dtypes),
then streams one tensor at a time, so it fits a laptop.

### Fallback, if SafeTune will not install

Without DARE the method is plain arithmetic, `drifted + alpha * (aligned - base)`, tensor by tensor in fp32
and cast back, which is what `src/msd/blend.py` already does for a different formula. Our own streaming
implementation of it would be a valid substitute (plan Task 16, Step 6 allows it) and would need no
torch model objects: three safetensors directories in, one out. It would not reproduce SafeTune's
default, which has DARE on (p = 0.3, seed 0), because the mask depends on SafeTune's draw order. If the
study falls back, say "ReSta without DARE" in the article. That fallback is NOT built; the formula above is
what it would implement.

### What is still UNVERIFIED

The real call (`scripts/run_resta.py` is written against this source and tested with a fake trainer
only), the key names of Tiny Aya Base against Global, the stored dtype and what `dtype="auto"` gives,
peak memory, SafeTune's Tiny Aya alpha claim, and a clean `import safetune` on Kaggle or Colab. All of
them are settled by the first GPU-runtime sanity run (plan Task 16, Step 6).

## AlignTune SFT (read from source)

Checked 2026-11-05. Nothing was run: AlignTune is not installed in any project venv, there is no GPU here
and no model weights were downloaded. Everything below is read from source text: the `aligntune` 0.1.31
package metadata (PyPI, published 2026-11-05) and the same paths in a shallow clone of
`github.com/Lexsi-Labs/aligntune` at commit `906da95363b3238955aacb28486ea5ecc5cdbd9a` (2026-11-05); TRL
at the `v1.7.1` tag; PEFT `main` (merge path); the public Hugging Face API record of
`CohereLabs/tiny-aya-global` (no token) and the public `modeling_cohere2.py` of transformers `main`.
AlignTune files read: `core/backend_factory.py` (`create_sft_trainer`), `core/sft/config.py`,
`core/sft/config_loader.py`, `core/sft/training_args_builder.py`, `core/model_loader.py`,
`core/precision_handler.py`, `core/peft/`, `core/merge/peft_merger.py`, `core/trainer_base.py`,
`core/advisor.py`, `data/manager.py`, `data/curator_adapter.py`, `data/loaders/`,
`backends/trl/sft/sft_generation.py`, `utils/config_extractor.py`, `utils/config_utils.py`,
`utils/environment.py`, `cli/{validate,advise,merge}.py`, `examples/sft_trl_1.py` and
`notebooks/test_algos_trl_aya.ipynb`. Anything I could not read directly is marked UNVERIFIED.

### Package, licence and pins

- `pip install aligntune==0.1.31`, Python >=3.11. Its own code is MIT. The wheel's licence expression is
  `MIT AND Apache-2.0 AND AGPL-3.0-only AND LGPL-3.0-only` because it vendors unsloth and unsloth_zoo
  (Apache-2.0 with AGPL-3.0 parts) and mergekit (LGPL-3.0). `backend="trl"` switches the unsloth backend
  off (`_disable_unsloth_backend`), so the AGPL parts are not used by this study; that is my reading of the
  code, not legal advice.
- **Hard dependency `curatorkit[connectors]>=0.1.15`**, and CuratorKIT is under the Lexsi Labs Source
  Available License 1.1 (`License-Expression: LicenseRef-LSAL-1.1` in the installed 0.1.22 metadata,
  `LICENSE.md` read): free for research, evaluation, education and audit, noncommercial only. A study that
  publishes an article is inside that grant; a paid product built on the same install would not be.
- Pins that matter: `transformers>=5.15,<6`, **`trl==1.7.1` (exact)**, `peft>=0.18.0`, `torch>=2.4,<2.14`,
  `bitsandbytes>=0.45.5` (not 0.46.0 or 0.48.0), `torchao>=0.16`, `accelerate>=1.6`, `datasets>=2.14`
  (TRL 1.7.1 itself needs `datasets>=4.7`), `huggingface-hub>=0.34`.
- Clash check against the other toolkits: SafeTune 0.1.12 asks `trl>=0.12,<1.15`, `transformers>=5.15,<6`,
  `torch>=2.7` and Python >=3.12; CuratorKIT 0.1.22 asks Python >=3.11. The ranges overlap, so there is no
  hard conflict on paper, but the plan's one-venv-per-toolkit rule stays: AlignTune pulls a large tree
  (bitsandbytes, torchao, the vendored unsloth and mergekit) and an install on a runtime that ships its own
  torch may replace that torch. Whether it does on Kaggle or Colab is UNVERIFIED. `kaggle/10_run_stage.ipynb`
  installs no toolkit, so `docs/runtime.md` carries the `pip install` cell.

### `create_sft_trainer`

```python
from aligntune.core.backend_factory import create_sft_trainer      # also exported as aligntune.create_sft_trainer
create_sft_trainer(model_name=None, dataset_name=None, backend="auto", output_dir="./output", num_epochs=3,
                   batch_size=4, learning_rate=2e-4, max_seq_length=512, max_samples=None, system_prompt=None,
                   config=None, rope_type=None, rope_target_max_seq_length=None, rope_factor=None,
                   attn_implementation="auto", sliding_window=None, **kwargs)
```

- It only builds config objects and returns a trainer; **nothing is loaded** until `trainer.train()`, which runs
  `setup_model` and `setup_data` itself. It calls `set_seed(kwargs["seed"], default 42)` on entry.
- Everything that is not a named parameter is a keyword argument read with `kwargs.get(name, default)`. A
  misspelled keyword is NOT an error: it lands in `extra_params` and is dropped unless it happens to be a valid
  `trl.SFTConfig` field (next bullet). That is why `src/msd/finetune.py` keeps the exact keyword list in one
  tested function and a test pins every name.
- After the TRL `SFTConfig` is built (`core/sft/training_args_builder.py`), every keyword that is also a valid
  `SFTConfig` field is copied on with `setattr` (`extract_extra_and_missing_params`; it always wins and skips
  `__post_init__` validation). Keywords that are not `SFTConfig` fields (`lora_r`, `quantization`, ...) are
  read earlier, by the factory, as below.
- `backend="trl"` is the one to name (`"auto"` may pick unsloth when it is importable).
- The factory also has `config=` (a YAML path or dict, run through `parse_config_to_unified`). We do not use it:
  that path has other defaults (`use_peft` True, `lora_dropout` 0.05, `data_seed` 47) and `validate config`
  below shows the SFT YAML loader is broken at this commit. Everything is passed as keywords.

### Turning LoRA on, and every LoRA key

LoRA is **off by default**: `use_peft` (aliases `peft_enabled`, `use_lora`) defaults to False.

| keyword | default | meaning |
| --- | --- | --- |
| `use_peft` | False | switch LoRA on |
| `lora_r` | 16 | rank |
| `lora_alpha` | 32 | scaling numerator (scale = alpha / r) |
| `lora_dropout` | 0.1 here (0.05 on the YAML path) | adapter dropout |
| `lora_target_modules` | None | None falls back to q/k/v/o_proj; the string `"all-linear"` is resolved by AlignTune to the linear layers without the output embedding; a list names modules |
| `lora_bias` | `"none"` | |
| `lora_variant` | `"standard"` | |
| `use_rslora`, `use_dora`, `init_lora_weights` | off | only via `extra_params`; `use_dora` warns "not available" and falls back to LoRA |

Precedence when a key appears twice (`get_peft_value`): keywords (extra_params), then `config.model.peft.*`,
then `config.model.*`. The adapter is applied by AlignTune itself (`core/peft/lora.py`, `peft.get_peft_model`
with a `LoraConfig`) after the model is built; TRL's `SFTTrainer` is then called with `peft_config=None`.

### Quantisation and precision, with the T4 in mind

- Quantisation is the keyword `quantization={...}` with keys `load_in_4bit`, `load_in_8bit`,
  `bnb_4bit_compute_dtype` (**default `"bfloat16"`**, must be `"float16"` on a T4), `bnb_4bit_quant_type`
  (`"nf4"`) and `bnb_4bit_use_double_quant`. When it is non-empty the model goes through
  `prepare_model_for_kbit_training`. Our configs use **no quantisation** (the choice is argued in the
  configs): the adapter is then trained on the same fp16 weights it is merged into.
- Precision is the keyword `precision` (`"fp16"`, `"bf16"`, `"fp32"`, `"auto"`; `ModelConfig.precision`
  defaults to AUTO). **AUTO asks `torch.cuda.is_bf16_supported()`, which can answer True on a T4 (bf16 is
  emulated there), so `precision="fp16"` must be passed explicitly**; `PrecisionHandler` then sets the
  fp16/bf16 flags of the training arguments, and the keywords `fp16` / `bf16` are backfilled onto the
  `SFTConfig` as well. TRL's own `SFTConfig` turns `bf16` on when `fp16` is unset, which is the other reason
  to send both flags. The model is loaded with `torch_dtype=` (a deprecated alias of `dtype` in
  transformers 5; still accepted at 5.15, UNVERIFIED on the version the runtime resolves).
- `device_map` defaults to `"auto"`; on one GPU `{"": 0}` keeps the whole model on it.

### Data: a local jsonl of `messages`, the chat template, and the loss

- `dataset_name` may be a local file path. `LoaderResolver` picks `JSONLoader`, which calls
  `datasets.load_dataset("json", data_files=path)` (a DatasetDict with one `train` split).
- **No conversion is needed for our file.** `data/ft/train.jsonl` is `{id, lang, messages:[user, assistant], gold}`
  per line. `DataManager` then runs CuratorKIT over the split unconditionally (`curate_split`, a temporary
  parquet file, `format="auto"`, schema gate on, `min_tokens=0`, `max_tokens=1_000_000`, no clean, no dedup),
  recognises the `messages` column (`task_type` conversational), restores the accepted conversations
  (`restore_accepted_rows`, `_normalize_sft_messages`) and ends with a `messages` column plus
  `chat_template_kwargs={"enable_thinking": False}`; only those two columns go to TRL. `system_prompt=None`
  injects nothing. I ran the same CuratorKIT 0.1.22 reader with AlignTune's gate settings over our
  `train.jsonl` (the JSONL reader, not the parquet path): all 6,000 rows pass, 1,500 per language, with
  `lang` and `gold` carried as metadata. **The parquet round trip itself is UNVERIFIED**, and so is
  whether any row can be dropped there. `finetune.py` therefore compares the number of rows the trainer
  reports it trained on with the number in the file and refuses a run that dropped any (the 20-step smoke run
  shows it before the full run).
- The chat template is applied by TRL, not by AlignTune: `SFTTrainer` tokenises `messages` with
  `tokenizer.apply_chat_template`.
- **Loss on the assistant turn only.** TRL's `completion_only_loss` applies to prompt-completion datasets
  only (checked in `sft_trainer.py` at v1.7.1); for `messages` the switch is `assistant_only_loss=True`. It needs the
  template to mark the assistant span with `{% generation %}`, and TRL's `get_training_chat_template` swaps in
  a patched template when the tokenizer's own one equals a known one. **Tiny Aya Global's template (3,239
  characters, from the public record) is byte-identical to TRL's `cohere2.jinja`**, so TRL swaps in
  `cohere2_training.jinja`, whose mask covers `{content}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` of the assistant
  turn and nothing else. For our rows that is the answer letter plus the two end tokens. AlignTune passes
  both `assistant_only_loss` (default False) and `completion_only_loss` (default None) through to the
  `SFTConfig`. Without it the loss would also cover the ~2,000-character safety system preamble the
  template always injects and the question. The byte-identity and the mask are read from text; that the
  swap happens in the installed TRL is UNVERIFIED until the smoke run (its first batch can be inspected).
- **The template is stored as a named list, and that crashes TRL (review finding C1, fixed in `finetune.py`).**
  Tiny Aya Global's `tokenizer_config.json` keeps `chat_template` as `[{"name": "default", "template": ...}, ...]`.
  transformers 5.15 (`tokenization_utils_base.py`, in `__init__`) turns such a list into a dict
  `{name: template}`, so `tokenizer.chat_template` is a dict. TRL v1.7.1 then calls
  `has_generation_markers(chat_template)` with it, and that is `re.search(...)` on a dict: a `TypeError`, after
  the 6.7 GB download. What the sources show: `SFTConfig` has a field `chat_template_path` (`sft_config.py`),
  read in `sft_trainer.py` before the marker check; when it names an existing `.jinja` or `.j2` file TRL reads
  the text into `processing_class.chat_template` (a string), and only then calls `has_generation_markers` and
  the swap to `cohere2_training_chat_template`. AlignTune does not list `chat_template_path` as a parameter,
  but its backfill copies every extra keyword that is a valid `SFTConfig` field onto the config with `setattr`
  before the `SFTTrainer` is built, so it reaches TRL. `finetune.py` therefore writes the tokenizer's own
  **default** template (a str as it is, a dict or a list by the entry named `default`, anything else an error)
  to `work/finetune/<config stem>[-smoke]/default_chat_template.jinja` (not `chat_template.jinja`: the trainer saves its own
  template under that name next to the adapter) and passes that path as `chat_template_path`.
  Before training (and in `--dry-run`) it checks that the text has generation markers or equals
  `trl.chat_template_utils.cohere2_chat_template` (when trl imports; otherwise the plan says the check was
  skipped). Not run against the installed libraries: that TRL then swaps the template as read.
- **The labelled-token check.** TRL v1.7.1 builds a `labels` column (`build_labels`; `IGNORE_INDEX` -100 off the
  assistant turn) and raises if an example has no assistant token, so the useful addition is the upper bound that
  catches a loss over the whole sequence. `finetune.py` counts, for the first `checks.label_check_rows` tokenised
  rows, the labels that are not -100 and refuses a row outside `checks.label_tokens_min` to
  `checks.label_tokens_max` (a one-letter answer plus the two end tokens is about 3). **AlignTune 0.1.31 gives no
  moment to look at the data first:** `TRLSFTTrainer.train()` (`sft_generation.py`) builds the TRL `SFTTrainer`
  (which tokenises) and calls `trainer.train()` in one method, so `inner.trainer` does not exist before it. The
  check is therefore a `transformers.TrainerCallback` whose `on_train_begin` receives `args` and
  `train_dataloader` (transformers v5.15 `trainer.py`: `_prepare_for_training` sets the dataloader on the callback
  handler before `callback_handler.on_train_begin`); it is added by wrapping the instance method
  `get_hf_callbacks`, which `sft_generation.py` passes as `callbacks=`. If the hook, the dataloader or the
  `labels` column is not there the check is `unavailable`. **A smoke run records that**
  (`label_check: "unavailable"`, `label_check_note`) and prints a WARNING on stdout and stderr; **a real run
  stops at step 0** (the callback raises; with no hook or no transformers, before anything is loaded), exit 1,
  unless `--allow-unchecked-labels` is given, which goes on with a WARNING and `label_check_waived: true`. A
  trainer that finishes without ever calling the callback is refused after the fact. An out-of-bounds count is
  always refused. Not run: that the rows in `train_dataloader.dataset` carry `labels` as in TRL's source.
- **The trainer-argument check.** The same `on_train_begin` callback receives `args`, the trainer's own
  `TrainingArguments` (an `SFTConfig`), and `finetune.py` compares thirteen attributes with the config:
  `fp16`, `bf16`, `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`,
  `gradient_accumulation_steps`, `max_length`, `assistant_only_loss`, `seed`, `gradient_checkpointing`,
  `save_total_limit`, `warmup_steps` (the config's `warmup_ratio`; a float below 1 is a ratio in transformers
  >= 5.2) and `lr_scheduler_type` (an enum, compared by its value). A difference stops the run and names the
  config key; an attribute the installed version lacks is listed as `not_found`; if nothing could be compared the
  summary says `args_check: "unavailable"`. Not run: that these attribute names and types are what the installed
  transformers 5.15 and TRL 1.7.1 give.
- `loss_type` is `"nll"` (AlignTune accepts only `nll` and `dft`, so TRL's newer `chunked_nll` is not
  reachable). Full logits over the 262,144-token vocabulary are therefore materialised.
- No packing (`packing=False`). TRL's `max_length` is AlignTune's `max_seq_length`, truncating from the start
  of the sequence (`keep_start`): **a truncated example loses its end, so its assistant span is cut or gone and
  it trains on nothing.** `finetune.py` counts tokens with the chat template and refuses a config whose
  `max_seq_length` is below the longest example.

### Batch, schedule, checkpointing and the other defaults to override

| keyword | AlignTune default | what we send and why |
| --- | --- | --- |
| `gradient_accumulation_steps` | 1 | from the config |
| `warmup_ratio` | 0.1 | sent as `warmup_steps=<ratio>` (a float below 1 is a ratio in transformers >=5.2); `warmup_steps` is ignored unless `warmup_ratio` is None |
| `lr_scheduler` | `"cosine"` | from the config |
| `weight_decay`, `max_grad_norm` | 0.01, 1.0 | from the config |
| `gradient_checkpointing` | **False**; `gradient_checkpointing_kwargs` `{"use_reentrant": False}` | from the config. The model-level `use_gradient_checkpointing` (default True) is NOT read on the SFT/TRL path; only this keyword reaches the `SFTConfig` |
| `group_by_length` | True (`backend_factory`) | **not sent, and not a config key**: transformers 5 replaced it by `train_sampling_strategy` (`trainer.py` tests `args.train_sampling_strategy == "group_by_length"`), so the old keyword is not an `SFTConfig` field and the backfill would drop it. Whether a length-sorted sampler is on by default is not read here; the random default is assumed, UNVERIFIED |
| `load_best_model_at_end` | True (`metric_for_best_model="eval_loss"`) | False: there is no eval set |
| `eval_strategy` | `"steps"` | `"no"` (it becomes `"no"` anyway when there is no eval dataset) |
| `loggers` / `report_to` | `["tensorboard"]` / `"none"` | `report_to="none"` falls back to `loggers`, so `loggers=["none"]` |
| `device_map` | `"auto"` | `{"": 0}`: the whole model on the one GPU; `"auto"` may offload layers to the CPU |
| `save_steps` (`save_interval`), `logging_steps` | 500, 10 | from the config; `save_total_limit` reaches the `SFTConfig` by the backfill only |
| `seed`, `data_seed` | 42, None | both from the config's seed |
| `max_steps` | None | the smoke switch (`--max-steps`) |

### Where files go, and merging

- `TRLSFTTrainer.train()` (in `backends/trl/sft/sft_generation.py`) builds the model, trains, then calls
  `trainer.trainer.save_model()`: **the adapter (`adapter_model.safetensors`, `adapter_config.json`) lands in
  `output_dir` itself**, next to `checkpoint-<step>/` folders every `save_steps`, plus an AlignTune
  `lexsi_provenance.json` (it records absolute paths, so it never goes into anything we publish).
  `finetune.py` therefore gives AlignTune a persistent, non-atomic work folder
  `work/finetune/<config stem>[-smoke]/` as `output_dir` (kept on any failure and named in the error), then
  copies only the adapter's files (no checkpoints, no provenance file, not its own template file) into
  `<out>/adapter`, merges into `<out>/merged`, and writes `finetune_summary.json` last, each part atomically.
  `--merge-only --adapter PATH` repeats just the merge from a kept adapter.
- `train()` returns a dict with `final_loss`, `steps`, `epochs`, `metrics`, `model_path` and `training_time`.
  **`final_loss` is `train_result.training_loss`, the mean of the loss over the whole run, not the last loss**; the
  summary calls it `mean_train_loss` and adds `last_logged_loss` from `log_history`.
  The per-step losses are in `trainer.trainer.state.log_history` (entries with a `"loss"` key).
- **There is no resume API**: a dropped Kaggle or Colab session restarts the run (the checkpoint folders are
  there for a manual recovery only; UNVERIFIED that `Trainer.train(resume_from_checkpoint=...)` could be
  reached from outside).
- Merge, three ways: `trainer.merge_adapters(output_path, adapter_path)`;
  `PEFTMerger().merge_lora(base_model, output_path, adapter_path, tokenizer, torch_dtype="float16", ...)`
  (`core/merge/peft_merger.py`: load the base, `PeftModel.from_pretrained`, `merge_and_unload`,
  `save_pretrained`, save the tokenizer, add transformers-4 legacy fields to `config.json`, write
  provenance); and the CLI `aligntune merge run --method lora-merge --base B --adapter A --output O --dtype float16`.
  `finetune.py` does the same three PEFT calls itself, so no provenance file with absolute paths and no
  legacy fields end up in the model folder that may be pushed. **The base must be reloaded unquantised in
  fp16 before the adapter is merged**; merging into a 4-bit base is not possible. On the CPU, PEFT upcasts
  the fp16 delta matmul to fp32 (`cast_to_fp32`, read on PEFT `main`); RAM is about 2x the fp16 model
  (about 14 GB, UNVERIFIED), which is why the merge runs after the trainer and its GPU memory are freed.

### Seed and determinism

`set_seed` (`utils/environment.py`) seeds `random`, `numpy`, `torch`, `torch.cuda` and transformers' own seed;
the keyword `seed` is also written to the `SFTConfig`, and we send `data_seed` as well. Two runs with the same
seed, data, library versions and hardware should therefore draw the same data order and the same LoRA
initialisation. Bit-identical weights are NOT promised: fp16 training on a GPU uses non-deterministic
kernels, and `torch.use_deterministic_algorithms` is not switched on (UNVERIFIED either way). The summary
records the seed and every library version.

### `aligntune validate config` and `aligntune advise estimate`

- `aligntune validate config <file>`: its SFT branch (`SFTConfigLoader._dict_to_config`) builds `ModelConfig`
  with `lora_alpha`, `lora_dropout` and `target_modules`, none of which `ModelConfig` has (checked by walking
  its AST), so it raises `TypeError` for an SFT YAML and falls through to the RL loader, which then
  complains about RL fields. **It is not usable for an SFT config at this commit.** Not run (not installed),
  so this is a reading, not an observed failure. `scripts/finetune.py --dry-run` does the validation we need.
- `aligntune advise estimate --model M --dataset-size N --algorithm lora|qlora|sft --gpu t4 --batch-size B
  --seq-len L --epochs E --grad-accum A` is a pure heuristic (`core/advisor.py`), nothing is measured.
  **It infers the model size from the name** (`infer_model_size`: the first of `0.5b 1b 2b 3b 7b 13b 34b 70b 110b`
  whose digits occur anywhere in the lower-cased name), so `CohereLabs/tiny-aya-global` has no size and
  is assumed to be 7B; a name that contains `3b`, for example `tiny-aya-3b`, gives 3,000M. It has a T4 profile
  (16 GB, 65 fp16 TFLOPS). The formulas, re-implemented by hand here without importing AlignTune
  (VRAM = (M/250 + bs*seq*M/250000 + 5% adapter) x {lora 0.3, qlora 0.15, sft 1.0}; tokens/s = 65*0.5/(2M/1000) x
  {lora 1.1, others 1.0} x 1e12 / (M x 1e6); hours = rows x seq x epochs / tokens/s), for M = 3,000, batch 1,
  6,000 rows, 2 epochs, whole sequences of `seq` tokens, the advisor's own uncertainty +/-30%:

  | algorithm | seq | VRAM | tokens/s | hours |
  | --- | --- | --- | --- | --- |
  | lora | 1024 | 7.5 GB | 1,986 | 1.7 |
  | lora | 2048 | 11.2 GB | 1,986 | 3.4 |
  | qlora | 2048 | 5.6 GB | 1,806 | 3.8 |
  | sft (full) | 2048 | 36.6 GB | 1,806 | 3.8 |

  The advisor charges every sequence at full length; our rows are mostly shorter, so the hours are an upper
  bound on that formula. It does not model the 262k-vocabulary logits (below), so its VRAM is optimistic.
  Command to reproduce:
  `aligntune advise estimate --model tiny-aya-3b --dataset-size 6000 --algorithm lora --gpu t4 --batch-size 1 --seq-len 2048 --epochs 2 --grad-accum 16`.
  UNVERIFIED: that this is what the command prints.

### Tiny Aya Global: architecture, template, generation, training caveats

From the public record `https://huggingface.co/api/models/CohereLabs/tiny-aya-global` (sha
`af89d219b53ed9b13b8a4645f9c8028973510324`, last modified 2026-09-30; gated "auto"; licence CC-BY-NC-4.0) and
`modeling_cohere2.py` (transformers `main`). The gated `config.json` could not be read without a token; the
numbers below come from the config of a third-party derivative with the same parameter count
(`EMD123/tiny-aya-kosher-3.3B`), so treat each as UNVERIFIED for the original.

- Class `Cohere2ForCausalLM`, `model_type` `cohere2`, 3,349,227,520 parameters, stored in BF16 (6.72 GB).
  Hidden size 2048, MLP 11008, 36 layers, 16 attention heads and 4 KV heads of dimension 128, vocabulary
  262,144 with tied embeddings, sliding window 4096 on 3 of every 4 layers, RoPE theta 50,000, `layer_norm_eps`
  1e-5, parallel attention/MLP blocks.
- The embedding alone is 262,144 x 2048 = 0.54B parameters (1.07 GB in fp16). **The logits are what make
  training memory large:** one 2,048-token example is 2,048 x 262,144 = 537M logits, 1.07 GB in fp16 and 2.15 GB
  once the loss upcasts to fp32, with the same again for the gradient.
- Linear layers (read from `modeling_cohere2.py`): attention `q_proj`, `k_proj`, `v_proj`, `o_proj`; MLP
  `gate_proj`, `up_proj`, `down_proj`; and `lm_head` (tied to the embedding, never a LoRA target). Rank 16 on
  those seven names is 839,680 parameters per layer, **30.2M in all** (121 MB in fp32).
- Tokenizer: `bos` `<BOS_TOKEN>`, `eos` `<|END_OF_TURN_TOKEN|>`, `pad` `<PAD>` (exists), `unk` `<UNK>`. The padding
  side is UNVERIFIED.
- Chat template: roles must alternate, and **it always injects a long "contextual safety mode" system preamble
  (about 2,000 characters, perhaps 450 tokens, the token count UNVERIFIED) before every conversation.** The
  assistant turn is `<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{content}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`;
  the generation prompt is the first three tokens of that. Our one-letter answers sit in a template whose
  preamble is about the safety behaviour the study measures.
- Generation settings on the model card: `max_new_tokens=4096, do_sample=True, temperature=0.1, top_p=0.95`
  (the card says only `pip install -U transformers accelerate`). Greedy decoding is what `run_eval.py` uses.
- Others have fine-tuned it with TRL 0.28 under transformers 5.0 to 5.2 (public model repos
  `qgallouedec/tiny-aya-global-SFT` and `sergiopaniego/tiny-aya-global-SFT`), so Cohere2 trains under
  transformers >=5.0. AlignTune ships its own Tiny Aya notebook (`notebooks/test_algos_trl_aya.ipynb`:
  `create_sft_trainer(..., use_peft=True, lora_r=16, lora_alpha=32, max_steps=2)`), and `model_loader` handles
  Cohere2 as a plain causal LM.
- Memory on a T4 (16 GB), arithmetic only, UNVERIFIED: fp16 weights 6.7 GB + LoRA (rank 16: 30M fp32
  parameters, gradients and two Adam moments, about 0.5 GB) + logits and their gradient for the longest
  example (about 5 GB at 2,048 tokens, about 2 GB at 800) + checkpointed activations (about 0.3 GB) puts the
  longest batches near 13 to 14 GB and typical ones near 10 GB. Batch size 1, gradient checkpointing and the
  longest example's token count (printed by `--dry-run`) are what keep it inside 16 GB.
- fp16 stability: the model is bf16-native and a T4 has no bf16, so overflow in fp16 is a real risk
  (UNVERIFIED). The loss finite-check in `finetune.py` stops a NaN run and the merge is written in fp16 only
  after that check.
- The base checkpoint `tiny-aya-base` has no chat template; AlignTune's `setup_tokenizer` has a fallback
  for that case. Not relevant here (A1 starts from Global).

### How `scripts/finetune.py` uses it (written from the source above, never run against it)

`src/msd/finetune.py` reads a YAML config (`configs/ft_lora.yaml`, `configs/ft_lora_strong.yaml`; unknown keys and
out-of-range values are errors), validates the training data in-process with `msd.ftdata` (the same rules as
`scripts/validate_ft_data.py`, plus: no overlap with heldout or taskdev when those files exist, and the
subject allowlist), and only then builds the trainer. The call is the one the table above lists:

```python
create_sft_trainer(model_name=<base_model>, dataset_name=<train.jsonl>, backend="trl", output_dir=<adapter dir>,
                   num_epochs=..., batch_size=..., learning_rate=..., max_seq_length=...,
                   use_peft=True, lora_r=..., lora_alpha=..., lora_dropout=..., lora_target_modules=[...],
                   precision="fp16", fp16=True, bf16=False, gradient_checkpointing=True, gradient_accumulation_steps=...,
                   warmup_ratio=..., lr_scheduler=..., weight_decay=..., max_grad_norm=..., assistant_only_loss=True,
                   chat_template_path="work/finetune/<stem>/default_chat_template.jinja",
                   seed=..., data_seed=..., logging_steps=..., save_steps=..., save_total_limit=..., loggers=["none"],
                   load_best_model_at_end=False, eval_strategy="no", device_map={"": 0})
```

then `trainer.train()`, the loss and row-count checks, the merge, and `finetune_summary.json`. A conversion of
the data would be needed only if the CuratorKIT parquet round trip turned out to change the rows; the
smoke run decides that.

### What is still UNVERIFIED

The real call (`scripts/finetune.py` is written against this source and tested with a fake trainer only);
that `import aligntune` and its install work on Kaggle or Colab without replacing torch; the parquet round
trip and any dropped row; that TRL swaps in `cohere2_training.jinja` and masks only the answer (the labelled-token check reports about 3 per row if it does); that `chat_template_path` reaches the `SFTTrainer` through the backfill; that `get_hf_callbacks` is honoured and the dataloader rows carry `labels`; fp16 stability
of Cohere2 on a T4; peak memory and the real token lengths (the 450-token preamble is a guess); the T4 hours;
the `aligntune advise` output; whether `save_total_limit` and `loggers=["none"]` reach the `SFTConfig`
through the backfill; the architecture numbers of the gated original; the CPU merge RAM. All of them are
settled by the first GPU-runtime run (`docs/runtime.md`, section 5, finetune steps 1 to 3).
