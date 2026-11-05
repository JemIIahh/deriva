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
