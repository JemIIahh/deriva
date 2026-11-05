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
python scripts/build_ft_data.py finalize   # joins the export back to ids -> data/ft/train.jsonl, heldout.jsonl
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
datasets.load_dataset("data/ft/curated", "sft_messages")    -> DatasetDict({train: 7960 rows, features: ['messages']})
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
  exact-duplicate filter. It removed 134 of 8,147 samples here. The two dedup steps and the
  samplers removed 195 samples in all (8 exact copies, 134 near-duplicates, 53 by the samplers);
  of the 187 whose text no longer survives, 131 shared a source passage with a kept sample
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

  CuratorKIT's own export, `data/ft/curated/sft_messages.jsonl`, has 7,960 rows (6,964 train and
  996 heldout before the per-language caps) and 46 gate rejections (all `above_max_tokens`).

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
