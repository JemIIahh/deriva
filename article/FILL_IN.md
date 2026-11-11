# Filling in the article draft

The draft in article/ARTICLE.md has no measured result in it. Every place that needs a real number, link or owner fact is a marker of the form `<<RESULT: name>>`, `<<LINK: name>>` or `<<OWNER: name>>`. This file has one line for each distinct marker, in the order of first appearance, saying which exported file and which column or key the value comes from. A marker that appears in several places (for example the attack-success cells of the language table) is one line here: fill each place from the matching language.

Steps after the GPU runs:

1. Run the study and the export: `envs/core/bin/python scripts/make_report.py ...` then `envs/core/bin/python scripts/export_public.py --out export/`.
2. Copy `export/dataset/aggregates/heatmap.png` and `export/dataset/aggregates/alpha_curve.png` next to the article (into article/), see the figures list below.
3. Replace each marker, using the lines below. Delete the marker brackets as well as the name. Write the value in plain words, with its interval.
4. Run `envs/core/bin/python scripts/check_article.py article/ARTICLE.md data/sources.yaml` with no flag. It must exit 0. That is the state to submit.

Value paths below are relative to the repository, after the report and the export have run. The aggregate files are the ones in `export/dataset/aggregates/`: `tables.csv`, `reliability.json`, `heatmap.png`, `alpha_curve.png`. The manifest is `export/MANIFEST.json`.

## Placeholders, one line each

- `<<OWNER: author name as it should appear>>`: the owner. Appears in the byline and in Teammates; use the same name in both.
- `<<LINK: code repository>>`: the URL of the public code repository, after the owner publishes it.
- `<<LINK: dataset on Hugging Face>>`: the URL of the dataset the owner uploads from `export/dataset/` (its card is `export/dataset/README.md`).
- `<<LINK: model collection on Hugging Face>>`: the URL of the model collection the owner creates for the arms in `export/models/`.
- `<<RESULT: headline answer in one sentence>>`: write last. One sentence that gives the answer to each of the three questions using only numbers that appear in Section 8. Source: the three short answers.
- `<<RESULT: whether arm A3 was built>>`: say yes or no. Source: `results/gate.json` key `branch` (A3 is built only when it is `pivot_prevention`) and whether `export/MANIFEST.json` lists `models/A3/README.md`.
- `<<LINK: CuratorKIT repository>>`: the URL of the CuratorKIT repository or package page, read from the package that was installed (the version used is in `docs/toolkit-notes.md`).
- `<<OWNER: SafeTune licence as read from its repository>>`: the owner or a reader opens the LICENSE file in https://github.com/Lexsi-Labs/SafeTune and names the licence in the Section 2 table cell. The study never re-read it. If its terms limit use, add a sentence to the Section 2 licence paragraphs.
- `<<RESULT: judge model ids for dev and report>>`: `reliability.json` key `judge_ids` (`dev` and `report`), and `second_judge_id` for the second judge.
- `<<OWNER: whether a paid judge was used for the report split, and which one>>`: the owner. Must agree with `reliability.json` key `judge_ids.report`; a stand-in judge and a paid judge are never mixed inside one set of numbers.
- `<<OWNER: name of the model behind that endpoint and the terms it was used under>>`: the owner. The translator names per language are recorded in `data/testset_build_summary.json` (key `languages`, translator names) and in the dataset card's translators table.
- `<<OWNER: which model arms were uploaded with weights>>`: the owner. Compare with the model cards listed in `export/MANIFEST.json` under `models/` (the export writes cards and never weights).
- `<<RESULT: fine-tune training summary, from finetune_summary.json>>`: `results/finetune/<config stem>/finetune_summary.json`, keys `mean_train_loss`, `last_logged_loss`, `rows_seen`, `merged`, `label_check`. One or two plain sentences.
- `<<RESULT: alpha chosen on dev, with the reason>>`: `reliability.json` key `alpha_choice`: `chosen`, and for the other candidates `candidates[].reasons`, plus `rule` and `over_refusal_margin`. If `chosen` is null, say that no strength qualified and give the reasons.
- `<<RESULT: freeze hash of the test set>>`: sha256 of `data/testset/MANIFEST.json`, the hash appended to `docs/FREEZE_LOG.md` at the real freeze.
- `<<RESULT: freeze date>>`: the date on the same line of `docs/FREEZE_LOG.md`.
- `<<RESULT: prompts kept and dropped per language after the translation check>>`: `data/testset_build_summary.json` key `languages`, kept and dropped per split and kind; the same counts are in the table at the top of `export/dataset/README.md`.
- `<<OWNER: GPU service and GPU type used for the final run>>`: the owner (Kaggle or Colab, and the GPU named in the notebook output).
- `<<RESULT: gate outcome, with the languages that rose by at least ten points>>`: `results/gate.json` keys `passed`, `rises` (per language, in points), `pooled`, `reason`, `n`.
- `<<RESULT: gate branch taken, and whether the stronger recipe was run>>`: `results/gate.json` keys `branch` and `strong_tried`.
- `<<RESULT: inter-judge agreement per language, with intervals>>`: `reliability.json` key `kappa` (per language: `kappa`, `lo`, `hi`, `n_pairs`), and `kappa_threshold`.
- `<<RESULT: low-confidence languages>>`: `reliability.json` key `low_confidence` (with the reasons), and the `low_confidence` column of `tables.csv`. Say "none" if the key is empty.
- `<<RESULT: partial or null results in plain words>>`: write from `gate.json` (a gate that did not pass), from `tables.csv` rows whose interval (`lo` to `hi`) leaves the question open, and from `reliability.json` keys `excluded_cells` and `warnings`. If nothing qualifies, write that plainly.
- `<<OWNER: confirm that no native speaker reviewed the translations or the judge labels>>`: the owner. If someone did, say who by role and for which languages, and remove the matching limit.
- `<<RESULT: answer to question 1, drift>>`: `tables.csv` rows `metric=asr`, `arm=A0` against `arm=A1`, per language (`rate`, `lo`, `hi`, `n`), and `gate.json` `passed`. One or two sentences.
- `<<RESULT: answer to question 2, unequal across languages>>`: the same rows, compared across languages; say which languages differ and whether the intervals overlap.
- `<<RESULT: answer to question 3, repair>>`: `tables.csv` rows `arm=A1` against `arm=A2` (alpha = `alpha_choice.chosen`) for `asr`, `over_refusal` and `task_acc`, and against the `A2n` rows.
- `<<RESULT: attack success rate of the original model per language>>`: `tables.csv`, `arm=A0`, `metric=asr`, one cell per language: `rate` with `lo` to `hi`, as percentages.
- `<<RESULT: attack success rate of the drifted model per language>>`: the same for `arm=A1`.
- `<<RESULT: attack success rate of the repaired model per language>>`: the same for `arm=A2` at the alpha named by `alpha_choice.chosen`. If no alpha was chosen, write "none chosen" in every cell.
- `<<RESULT: attack success rate of the naive blend per language>>`: the same for `arm=A2n` at the blend value named by the next marker (the blend value sits in the `alpha` column).
- `<<RESULT: change from original to drifted per language>>`: the difference of the two `tables.csv` rates (A1 minus A0), in points, with the paired interval from `msd.reportrun.paired_deltas` run on the public verdict rows (the function the heatmap uses). If no paired interval is computed, give the difference only and edit the "How to read the numbers" paragraph to say that no paired interval is given.
- `<<RESULT: change from drifted to repaired per language>>`: the same for A2 minus A1.
- `<<RESULT: blend value shown for the naive blend>>`: choose one of 0.25, 0.5, 0.75 (the `A2n` rows of `tables.csv`) and say which and why. Pick it before looking at the safety rates, for example the one whose task accuracy is closest to the repaired model's.
- `<<RESULT: whether the repair does better than the naive blend>>`: compare the `A2` and `A2n` rows of `tables.csv` on `asr`, `over_refusal` and `task_acc`; state it as a comparison with intervals, not as a ranking.
- `<<RESULT: over-refusal per arm and language>>`: `tables.csv`, `metric=over_refusal`, arms A0, A1 and A2, per language: `rate`, `lo`, `hi`.
- `<<RESULT: task accuracy per arm and language>>`: `tables.csv`, `metric=task_acc`, arms A0, A1 and A2; rows with `status=not_applicable` (Hinglish) stay "not applicable".
- `<<RESULT: reading of the heatmap in plain words>>`: describe `heatmap.png` using `tables.csv`; name the hatched, dotted and crossed rows and what `reliability.json` key `heatmap` says is compared.
- `<<RESULT: reading of the alpha curve in plain words>>`: describe `alpha_curve.png` using the A2 rows of `tables.csv` along `alpha`.
- `<<RESULT: cells left out by the sanity gate or the judge-error limit>>`: `reliability.json` key `excluded_cells` (arm, alpha, lang, reason, `failed_checks`) and `sanity_gate`. Say "none" if empty.
- `<<LINK: reproduction notebook>>`: the URL of notebooks/reproduce.ipynb as published in the code repository (or its Kaggle or Colab copy).
- `<<RESULT: reproduction check outcome>>`: the notebook's own pass or fail line: the sha256 of the rebuilt `tables.csv` against the `export/MANIFEST.json` entry. Write it as the notebook prints it.
- `<<OWNER: confirm what you personally checked, for example the sources, the design decisions and the results>>`: the owner, in one plain sentence about what they checked. Do not claim more than they did.
- `<<OWNER: decision to keep this disclosure paragraph>>`: the owner decides whether to keep the Claude Code disclosure paragraph in Teammates. The checker requires a paragraph that names Claude Code, so keep it unless the check rules change.

## Figures still to be added

The draft shows two pictures that do not exist until the report has run. Copy both from `export/dataset/aggregates/` into article/ before the final check, because the checker in final mode needs the files to exist:

- `heatmap.png`, the per-language heatmap of attack success rate (the picture the article is built around).
- `alpha_curve.png`, safety against task accuracy along the repair-strength grid.

## Re-check before publishing

These statements in the draft were written from the repository as it stood when the draft was written. Each can change before the final run. Check them against the sources named here.

- Which source supplied how many English prompts (113, 112, 112, 113, 150, 150): `data/testset_build_summary.json`, key `sources_used` and the selection counts, after the real freeze.
- No Hinglish fine-tuning rows exist: `data/ft/` has rows in en, hi, ar and sw only. If Hinglish rows are added, change Section 6 and the Hinglish line in the limits.
- Sources cleared but not used (Nemotron-Safety-Guard-Dataset-v3, belebele, MMMLU, aya_dataset): the code uses none of them (only the default source set in `src/msd/testset.py`). If a run adds one, move it to the used table.
- 25 sources checked and left out: count the `excluded` entries of `data/sources.yaml` again if it changes.
- AuditKIT is not used. If the owner adds it, change the last Section 2 paragraph.
- SafeTune and its licence: see the SafeTune marker above. The DARE default (rate 0.3, seed 0) and the alpha grid are from `src/msd/config.py` and `docs/toolkit-notes.md`; recheck them if the code changes.
- The gate's pass rates (0%, 0.3%, 25%, 51%, 84%, about 43%) are a reviewer's simulation at 150 harmful prompts per language, copied as the review recorded them. The simulation settings are not saved in the repository.
- Limits that come from the working log: held-out interval about 7 points, dev floor 31% against 27 to 28%, 18 near-duplicate template pairs (4 at 0.8 or more), one source twin (astronomy test items 9 and 116). Recompute them if the data is rebuilt.
- The checklist's own wording of the nine section titles is longer than the project brief's (the slide at `reference/kickoff-slides-text.txt`, slide 29). The article and `scripts/check_article.py` use the brief's titles. If the organizers want theirs, change `REQUIRED_HEADINGS` in the script and the article's headings together.
