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

This is one model from a study of **multilingual safety drift**: whether, and how, fine-tuning a small multilingual
model (Tiny Aya Global) on a harmless task changes how safe it is, whether that differs from language to language, and
whether a repair that needs no retraining helps. The study covers {{languages}}. What it found is in the article and in
the tables of the study's dataset, not on this card. This card describes arm **{{arm}}**.

## What this model is

{{how_made}}

{{repair_line}}

The test set, the per-item results and the tables are in the study's dataset. The per-language numbers live in the
article, not here, so that a card and the article can never disagree.

## What it was fine-tuned on

The fine-tuning task is harmless and closed-form: questions with a short answer that can be checked exactly, either a
bare option letter for a multiple-choice question or a short extracted span for the rest, on harmless subjects. The
training data is validated to hold no refusal-style answer, no harmful content and nothing from the safety test set. The
study asks whether ordinary, harmless fine-tuning moves safety behaviour; it does not assume that it does.

## Intended use

Research on whether and how safety behaviour changes per language, and on whether it can be repaired. Nothing else.

This model is **not a safety guarantee**. A low attack-success rate in our tables means "low on our test set, with our
judge, in that language", not "safe". It can still produce harmful content, and it can still refuse harmless requests.
Do not use it where harm to a person is possible without your own evaluation in your own languages.

## Licence and terms: read before you use it

- **Base model.** Tiny Aya Global (`{{base_model}}`) is released under **CC-BY-NC-4.0** together with the Cohere Labs
  **Acceptable Use Policy**, and it is gated on Hugging Face: you must accept its terms on the base model's page
  before you can use it, or this model.
- **This model inherits those terms.** The fine-tuned model and the model after the repair step are derivatives of Tiny Aya Global, so
  CC-BY-NC (non-commercial, with attribution) and the Acceptable Use Policy apply to this model too. Releasing it does
  not widen what the base model allows.
- **Toolkits.** It was made with Lexsi Labs toolkits (AlignTune, CuratorKIT, SafeTune). AlignTune's own code is MIT, with
  a hard dependency on CuratorKIT, which is under the Lexsi Labs Source Available License 1.1 (LSAL-1.1): free for
  research, evaluation, education and audit, non-commercial only. AlignTune's package also bundles unsloth (Apache-2.0 with
  AGPL-3.0 parts) and mergekit (LGPL-3.0). This study used the TRL backend, which switches the unsloth backend off; that is
  our reading of the code, not legal advice. For SafeTune's own licence, see the SafeTune repository.

## Known limits

- **Small samples.** The number of report-split prompts per language is in the dataset card. With samples that size a
  language's rate has a wide interval. The tables show the intervals; read them.
- **Judge limits.** Whether a reply complied or refused was decided by an AI judge (`{{report_judge}}`). {{second_judge_sentence}}
  Judges make mistakes, and the agreement can differ by language; a language where the second judge was not run, or
  agrees weakly, is marked low-confidence in the tables.
- **No native-speaker review.** The non-English prompts are machine translations that passed a back-translation check.
  No native-speaker review was done.
- **Hinglish has no task-accuracy floor.** No closed-form task data exists for Hinglish, so a Hinglish model after the repair step is
  checked on its outputs only, not on task accuracy.
- **Five languages only**, in the settings we tested; nothing here says how any other language behaves.
