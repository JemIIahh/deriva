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
python scripts/validate_ft_data.py --langs en,hi,ar,sw --pipeline pipelines/ft_data.yaml
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
datasets.load_dataset("data/ft/curated", "sft_messages")    -> DatasetDict({train: 7976 rows, features: ['messages']})
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
  harmless-subject allowlist before CuratorKIT, and `validate_ft_data.py` re-checks both after.
  CuratorKIT does the reading, the schema/length gate, exact and MinHash dedup, language
  equalising and the export.
- `stage_counts` in `manifest.json` is keyed by step name, so two `stratified_sampler` entries
  overwrite each other (only the last is listed). We use two on purpose (train strata, heldout
  strata); per-stratum numbers come from our own join instead.
- `stratified_sampler` cuts every listed category down to the smallest one (it never pads). A
  category with no samples is skipped with a warning, not an error.
- `minhash_dedup` at 0.85 judges the whole `instruction + output` text, so it is not an
  exact-duplicate filter. It removed 123 of 8,146 samples here. Of the 178 samples removed by the
  two dedup steps and the samplers together, 124 shared a source passage with a kept sample
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
