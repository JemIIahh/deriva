---
license: other
license_name: mixed-see-card
language:
- en
- hi
- ar
- sw
tags:
- safety
- multilingual
- red-teaming
- evaluation
---

# Multilingual safety drift: test set and per-item results

The test set, the per-item results and the tables of a study of whether and how fine-tuning a small multilingual model
(Tiny Aya) changes its safety, language by language: {{languages}}. It holds harmful-request prompts and harmless-but-edgy
prompts, the label a judge gave each reply from each model arm, the task-accuracy score of each item, and the tables
computed from them. The article explains the study; this card explains the files and what you may do with them.

## Files

| File | What it holds |
|---|---|
| `prompts.jsonl` | the {{n_prompts_en}} English prompts, with their source and licence |
| `prompts_translated.jsonl` | the {{n_prompts_translated}} machine-translated prompts, marked CC-BY-NC-4.0 (see below) |
| `results_public/<split>/<arm>/<alpha or none>/<language>.verdicts.jsonl` | one row per judged reply: prompt id, language, arm, alpha, label, judge id, excerpt |
| `results_public/<split>/<arm>/<alpha or none>/<language>.tasks.jsonl` | one row per task item: prompt id, language, arm, alpha, correct |
| `aggregates/` | `tables.csv`, `reliability.json`, `heatmap.png`, `alpha_curve.png` |

Arms in the results: {{arms}}.

## Prompts and splits

{{prompt_counts_table}}

The **dev** split was used to choose settings (the repair strength, the judge prompts) and to check that a model after
the repair step still works. The **report** split was used for the numbers in the article and for nothing else. No prompt
is in both. The test set is frozen: the sha256 of its freeze manifest is `{{freeze_sha256}}`.

## Sources and licences

{{sources_table}}

A source marked "no (evaluation only)" was used to score models and only the scores are published: none of its items
are copied here. {{n_excluded_sources}} further sources were checked and not used (non-commercial, unclear or
conflicting licences); the reasons are in the study's source manifest, `data/sources.yaml` in the code repository.

Several sources require attribution (CC-BY): the `source` field of every prompt row names its source; cite it when you
use the row. MultiJail-style sources (DAMO-NLP-SG/MultiJail is MIT-licensed) come with their own ethics statements,
which apply here too.

## Translated prompts

Every non-English prompt is a machine translation of an English prompt, checked by translating it back. The
translators were:

{{translators_table}}

The NLLB-200 model is released under CC-BY-NC-4.0, and whether its outputs inherit that is not settled, so **every row
of `prompts_translated.jsonl` is marked `license: CC-BY-NC-4.0`**, with `derived_from` naming the translator, whichever
translator made it. The `source_license` field keeps the licence of the English source. All translations are in this
one file, so a reader who needs only permissively licensed data can drop the whole file.

## Per-item results

{{n_verdict_rows}} verdict rows and {{n_task_rows}} task rows. The report split was judged by `{{report_judge}}` and the
dev split by `{{dev_judge}}`. A verdict row keeps an **excerpt of at most 200 characters** of the model's reply, cut at a
character boundary, and nothing more: no full completion is published, and no prompt text is repeated in a result row.

The replies come from models derived from Tiny Aya, which is released under CC-BY-NC-4.0 and the Cohere Labs Acceptable
Use Policy, so the excerpts, labels and tables made by us are released under **CC-BY-NC-4.0** as well, and the
Acceptable Use Policy applies to any use of them. The sha256 of `aggregates/tables.csv` is `{{tables_csv_sha256}}`; the
reproduction notebook recomputes the tables from these files and checks it.

## Ethics statement

The prompts include requests for harmful content, taken from public red-teaming benchmarks, because measuring safety
needs them. They are here for research and evaluation of model safety only. Do not use them to produce harmful content
or to attack a deployed system. We publish no full harmful completion: result rows hold labels and excerpts of at most
200 characters.

## Limits

- The report split has a few hundred prompts per language; read the confidence intervals in `tables.csv`.
- Labels come from AI judges. {{second_judge_sentence}} `reliability.json` records the agreement per language, where
  there is one, and which languages are low-confidence.
- Non-English prompts are machine translations; no native-speaker review was done.
- Hinglish has no closed-form task data, so it has no task-accuracy rows.

## Licence by file

| File | Licence |
|---|---|
| `prompts.jsonl` | the licence of each row's source, in its `license` field |
| `prompts_translated.jsonl` | CC-BY-NC-4.0 |
| `results_public/` | CC-BY-NC-4.0, with the Cohere Labs Acceptable Use Policy |
| `aggregates/` | CC-BY-NC-4.0, with the Cohere Labs Acceptable Use Policy |
