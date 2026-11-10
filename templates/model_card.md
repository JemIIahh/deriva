---
license: cc-by-nc-4.0
base_model: {{base_model}}
language:
- en
- hi
- ar
- sw
tags:
- safety
- multilingual
- research
---

# Tiny Aya Global, {{arm_title}} (arm {{arm}})

This is one model from a study of **multilingual safety drift**: fine-tuning a small multilingual model (Tiny Aya
Global) on a harmless task quietly makes it less safe, and not equally in every language; a repair that needs no
retraining can win much of it back. The study covers {{languages}}. This card describes arm **{{arm}}**.

## What this model is

{{how_made}}

{{repair_line}}

The test set, the per-item results and the tables are in the study's dataset. The per-language numbers live in the
article, not here, so that a card and the article can never disagree.

## What it was fine-tuned on

The fine-tuning task is harmless and closed-form: everyday multiple-choice questions answered with the bare option
letter. There is no harmful content and no refusal example in the training data. That is the point of the study:
ordinary, harmless fine-tuning is enough to move safety behaviour.

## Intended use

Research on how safety behaviour changes per language, and on repairing it. Nothing else.

This model is **not a safety guarantee**. A low attack-success rate in our tables means "low on our test set, with our
judge, in that language", not "safe". It can still produce harmful content, and it can still refuse harmless requests.
Do not use it where harm to a person is possible without your own evaluation in your own languages.

## Licence and terms: read before you use it

- **Base model.** Tiny Aya Global (`{{base_model}}`) is released under **CC-BY-NC-4.0** together with the Cohere Labs
  **Acceptable Use Policy**, and it is gated on Hugging Face: you must accept its terms on the base model's page
  before you can use it, or this model.
- **This model inherits those terms.** The drifted and the repaired models are derivatives of Tiny Aya Global, so
  CC-BY-NC (non-commercial, with attribution) and the Acceptable Use Policy apply to this model too. Releasing it does
  not widen what the base model allows.
- **Toolkits.** It was made with Lexsi Labs toolkits (AlignTune, CuratorKIT, SafeTune), which come under the Lexsi Labs
  Source Available License (LSAL): free for research, evaluation, education and audit, non-commercial only.
  AlignTune's own code is MIT but it requires CuratorKIT, and its package bundles unsloth (Apache-2.0 with AGPL-3.0
  parts) and mergekit (LGPL-3.0). This study used the TRL backend, which switches the unsloth backend off; that is our
  reading of the code, not legal advice.

## Known limits

- **Small samples.** The report split has a few hundred prompts per language, so one language's rate has an interval of
  several points. The tables show the intervals; read them.
- **Judge limits.** Whether a reply complied or refused was decided by an AI judge (`{{report_judge}}`). A second judge
  from another model family re-labelled a sample. Judges make mistakes, and the agreement differs by language; a
  language where the second judge was not run, or agrees weakly, is marked low-confidence in the tables.
- **No native-speaker review.** The non-English prompts are machine translations that passed a back-translation check.
  No native-speaker review was done.
- **Hinglish has no task-accuracy floor.** No closed-form task data exists for Hinglish, so a repaired Hinglish model is
  checked on its outputs only, not on task accuracy.
- **Five languages only**, in the settings we tested; nothing here says how any other language behaves.
