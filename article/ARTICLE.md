# Does harmless fine-tuning quietly make a small multilingual model less safe?

A study of Tiny Aya Global in English, Hindi, Hinglish, Arabic and Swahili, and of a repair that needs no retraining.

By <<OWNER: author name as it should appear>>. Code: <<LINK: code repository>>. Data: <<LINK: dataset on Hugging Face>>. Models: <<LINK: model collection on Hugging Face>>.

In one line: <<RESULT: headline answer in one sentence>>

This is a research study. Nothing in it is a safety guarantee for any model.

## Problem statement and approach, in brief

People often fine-tune a small language model for a narrow, harmless job, such as answering exam questions. Fine-tuning changes the model's weights, the numbers inside it that decide what it writes. The worry is that this can weaken the model's safety training even when every training example is harmless. A second worry is that the weakening is not the same in every language. These are worries, not findings. This study measures them.

The model is Tiny Aya Global, a small multilingual model from Cohere Labs. The languages are English, Hindi, Hinglish, Arabic and Swahili. Hinglish is Hindi written in Latin letters and mixed with English words, the way many people type it.

### The three questions

1. Drift. Does fine-tuning Tiny Aya Global on harmless questions raise the share of harmful requests that it goes along with?
2. Unequal drift. If it does, is the change the same in all five languages, or larger in some?
3. Repair. Can a repair that needs no retraining bring safety back without breaking the model?

Two measures are used throughout. The attack success rate is the share of harmful requests that the model goes along with, as decided by a separate judge model. Over-refusal is the share of harmless but edgy requests that the model refuses. A repair that simply refuses everything would score well on the first measure and badly on the second, so both are reported.

### The models compared, in plain words

| Name in this article | Code | What it is |
|---|---|---|
| Base | BASE | Tiny Aya Base, the model before its safety training. It is needed for the repair arithmetic. |
| Original | A0 | Tiny Aya Global, untouched. This is the "before" model. |
| Drifted | A1 | Tiny Aya Global after our harmless fine-tune, merged back into one model. The fine-tune is LoRA, which trains a small add-on instead of every weight. |
| Repaired | A2 | The drifted model after a repair called ReSta, at a strength called alpha that is chosen on development data only. |
| Naive blend | A2n | A plain average of the drifted and the original weights, (1 - t) times A1 plus t times A0, for t of 0.25, 0.5 and 0.75. It is a comparison: it shows whether ReSta does better than simply averaging the weights back. |
| Prevention | A3 | A fine-tune built to prevent drift instead of repairing it. It is built only if the drift gate in Section 7 sends the study to prevention. In the build so far it has not been written. <<RESULT: whether arm A3 was built>> |

### The approach in five steps

1. Build a test set of harmful prompts and harmless-but-edgy prompts in all five languages, split into a development half (dev) and a report half, and freeze it with a hash before any model is run.
2. Fine-tune Tiny Aya Global on harmless questions (A1). Measure A0 and A1 on the dev split. A drift gate decides whether there is anything to repair.
3. Repair (A2) and blend (A2n). The repair strength is chosen on the dev split only, and only among settings where the model still works.
4. Judge every reply with one judge model per set of numbers. A second judge re-labels a sample so that agreement between judges can be measured and reported.
5. Report the numbers from the report split, language by language, with intervals.

## Hackathon repos used: which ones, for what, and which features

| Repo | What we use it for | Licence |
|---|---|---|
| AlignTune (Lexsi Labs), https://github.com/Lexsi-Labs/aligntune | Runs the LoRA fine-tune that makes the drifted model. | MIT |
| CuratorKIT (Lexsi Labs), <<LINK: CuratorKIT repository>> | Cleans, de-duplicates and samples the fine-tuning data. | Lexsi Labs Source Available License 1.1 (LSAL-1.1) |
| SafeTune (Lexsi Labs), https://github.com/Lexsi-Labs/SafeTune | Provides the ReSta repair. | See the SafeTune repository. <<OWNER: SafeTune licence as read from its repository>> |
| NLLB-200 distilled 600M, through the transformers library | Machine-translates the English prompts into Hindi, Arabic and Swahili. | CC-BY-NC-4.0 (model weights) |
| Our own code, <<LINK: code repository>> | Builds the test set, runs the generations and the judge, computes the tables and figures, and exports the public files. | See the repository. |

AlignTune is released under the MIT licence. It depends on CuratorKIT, which is under the Lexsi Labs Source Available License 1.1 (LSAL-1.1): free for research, evaluation, education and audit, and not for commercial use. That is why the study and everything it releases are for research only.

The AlignTune package also bundles two other libraries with their own terms. It bundles unsloth, which is Apache-2.0 with some parts under AGPL-3.0, and mergekit, which is LGPL-3.0. We use only the TRL backend of AlignTune, which switches the unsloth backend off. Our reading is that the AGPL parts and the LGPL library are therefore not part of what we run. That is our reading of the code and not legal advice.

SafeTune is used for one thing: the ReSta repair. We call it through a small wrapper that checks that the three models it combines have the same weight names and shapes, and that records the repair settings next to the repaired weights. We did not re-read SafeTune's licence when writing this article, so the table sends the reader to its repository.

AuditKIT is another Lexsi Labs toolkit. It is not used in this study, and nothing in our code calls it. The study does use three other Lexsi Labs toolkits, each for one step: CuratorKIT for the data, AlignTune for the fine-tune and SafeTune for the repair.

## Models used and their licenses

| Model | Licence | Role in the study | Access |
|---|---|---|---|
| Tiny Aya Global (CohereLabs/tiny-aya-global) | CC-BY-NC-4.0 | The model under study (A0). Cohere Labs Acceptable Use Policy also applies. | Gated on Hugging Face: accept the terms on the model page. |
| Tiny Aya Base (CohereLabs/tiny-aya-base) | CC-BY-NC-4.0 | The model before safety training. Used only inside the repair arithmetic. Cohere Labs Acceptable Use Policy also applies. | Gated on Hugging Face: accept the terms on the model page. |
| Qwen/Qwen3-4B | Apache-2.0 | The primary open judge model, chosen before the run. | Open. |
| google/gemma-4-E4B-it | Apache-2.0 | The alternative judge, from a third model family. | Open. |
| facebook/nllb-200-distilled-600M | CC-BY-NC-4.0 | Translator into Hindi, Arabic and Swahili. | Open. |

Tiny Aya Global and Tiny Aya Base are released under CC-BY-NC-4.0, which allows non-commercial use with attribution. Both are gated on Hugging Face, and the Cohere Labs Acceptable Use Policy also applies to them. The fine-tuned model (A1) and the repaired model (A2) are derivatives of Tiny Aya Global, so the same licence and the same Acceptable Use Policy apply to them. Releasing them does not widen what Cohere's licence allows.

The two judge models named above are the ones chosen before any run: Qwen3-4B as the primary open judge and Gemma as the alternative. The judge ids that were actually used for the dev split and the report split are recorded in the exported file reliability.json: <<RESULT: judge model ids for dev and report>>. The study may also use a paid judge through an API for the report split, if the owner supplies a key. <<OWNER: whether a paid judge was used for the report split, and which one>>

Hinglish translations, and every back-translation used in the translation check, come from a chat model reached through an API endpoint. <<OWNER: name of the model behind that endpoint and the terms it was used under>> The translated prompts are all marked CC-BY-NC-4.0 in the released data, because the NLLB weights are under that licence and we do not split the translated file by translator.

## Datasets and benchmarks used and their licenses

Every source below was read at its card or repository on the date recorded in the study's source manifest, data/sources.yaml in the code repository. A source was used only if its licence allowed the use we make of it. Sources whose licence was non-commercial, share-alike, unclear or self-contradicting were left out.

### Used

| Source | Licence | Used for | Languages in the source |
|---|---|---|---|
| DAMO-NLP-SG/MultiJail | MIT | Harmful prompts. 113 of the English prompts. | en, ar, sw |
| centerforaisafety/HarmBench | MIT License | Harmful prompts. 112 of the English prompts. | en |
| llm-attacks/llm-attacks (AdvBench) | MIT License | Harmful prompts. 112 of the English prompts. | en |
| CohereLabs/aya_redteaming | Apache-2.0 | Harmful prompts. 113 of the English prompts. | en, hi, ar |
| Paul/XSTest | CC-BY-4.0 | Harmless-but-edgy prompts. 150 of the English prompts. | en |
| bench-llm/or-bench | CC-BY-4.0 | Harmless-but-edgy prompts. 150 of the English prompts. | en |
| CohereLabs/Global-MMLU (fine-tune subset) | Apache-2.0 | The harmless fine-tuning questions, and the held-out questions for task accuracy. | en, hi, ar, sw |
| CohereLabs/Global-MMLU | Apache-2.0 | The 200 task questions per language used on the dev split. They are questions that are in neither the training rows nor the held-out rows. | en, hi, ar, sw |

For the safety prompts the study uses the English prompts of each source and translates them. Every prompt in Hindi, Arabic, Swahili and Hinglish is a machine translation of an English prompt (Section 6), including in sources that also offer those languages. Only the fine-tuning questions are taken in each language as published. Attribution is required for the CC-BY sources, so every prompt row in the released data names its source and licence.

DAMO-NLP-SG/MultiJail comes with an ethics statement. The MultiJail card is under the MIT licence, and its Ethics Statement adds that the research is "solely for academic purposes and ethical use". That sentence is not a licence term, but we treat it as binding on how we use and release the prompts. It is one of the reasons the released prompts are for evaluation and research only.

### Cleared but not used

These sources passed the licence check and are in the source manifest. The code of this study does not use them, either because a better source was available or because the prompts did not suit the study.

| Source | Licence | Why it is not used |
|---|---|---|
| nvidia/Nemotron-Safety-Guard-Dataset-v3 | CC-BY-4.0 | The loader works, but the source is left out of the default test set. Read by eye, about half of its English "unsafe" test prompts are mid-conversation fragments or statements, not standalone requests that a model can be said to comply with. |
| facebook/belebele | CC-BY-SA-4.0 | Cleared for evaluation only: models may be scored on it and only the scores may be published, never its items. Not used here. |
| openai/MMMLU | MIT | A second task check in Hindi, Arabic and Swahili. Not needed once Global-MMLU was chosen. |
| CohereLabs/aya_dataset | Apache-2.0 | Free-form instruction data. It is not closed-form, so its answers cannot be checked exactly. |

### Checked and left out

Twenty-five further sources were checked and not used. The reasons are in data/sources.yaml. In short:

- Non-commercial licence: felfri/M-ALERT, PKU-Alignment/PKU-SafeRLHF, lmsys/toxic-chat, McGill-NLP/tukabench, ppattnay/IndicSafe, anna-sarvam/indic-safety-eval.
- Share-alike licence: google/xquad, facebook/mlqa.
- Gated, with access terms that could not be read: sorry-bench/sorry-bench-202406, allenai/wildguardmix, walledai/HiXSTest, and two sets built from WildGuardMix, ToxicityPrompts/PolyGuardPrompts and ToxicityPrompts/PolyGuardMix.
- Card that contradicts itself about the licence: ToxicityPrompts/PolygloToxicityPrompts, LibrAI/do-not-answer, masakhane/afriqa.
- No licence, an unclear one, or one that conflicts with the original release: walledai/MultiJail, ToxicityPrompts/DAMO-MultiJail, Unbabel/Multilingual-Red-Teaming, tiny-aya-translate/hinglish-casual, google-research-datasets/tydiqa.
- Not a fit for this study: microsoft/RTP-LX, Jarviswang94/Multilingual_safety_benchmark, nvidia/Aegis-AI-Content-Safety-Dataset-2.0, CohereLabs/include-base-44.

### What is released, and what is not

Released: the English prompts and the machine-translated prompts, each row with its source and licence; one label per judged reply, with an excerpt cut to 200 characters; one correct or incorrect mark per task question; and the tables and figures. Not released: any model reply beyond the 200-character excerpt, and any item of an evaluation-only source. The released dataset card says all of this again next to the files. Dataset: <<LINK: dataset on Hugging Face>>.

## Model artifacts (weights, adapters, …) public, as an HF collection

The collection holds the models that this study produced, each with a model card. Tiny Aya Global and Tiny Aya Base are not re-uploaded: Cohere Labs hosts them, and their gated terms apply to anyone who downloads them.

- The drifted model (A1) is Tiny Aya Global after the harmless LoRA fine-tune, merged back into one model stored in 16-bit precision.
- The repaired model (A2) is the drifted model after ReSta at the alpha chosen on the dev split.

Each card says which arm it is, repeats the CC-BY-NC-4.0 licence and the Cohere Labs Acceptable Use Policy, says that the model is not a safety guarantee, and sends the reader to the dataset for per-language numbers, so that a card and this article cannot disagree. Collection: <<LINK: model collection on Hugging Face>>. <<OWNER: which model arms were uploaded with weights>>

### How the drifted model was made

The fine-tune is LoRA, run through AlignTune's TRL backend. The settings are in configs/ft_lora.yaml in the code repository, and matter only to a reader who wants to rerun the fine-tune. Rank 16 and a scaling factor of 32, with dropout 0.05, on the attention and feed-forward projections (q, k, v, o, gate, up and down), which is 30.2 million trainable parameters. Learning rate 1e-4, two epochs, an effective batch of 16 and a maximum length of 2,048 tokens, which is 750 steps over 6,000 training rows. The loss counts only the assistant's answer. Training uses 16-bit (fp16) numbers. The seed is 20261105. The stronger recipe in configs/ft_lora_strong.yaml (learning rate 2e-4, three epochs, rank 32 and a scaling factor of 64, which is 1,125 steps) exists for one branch of the drift gate and is run only if the gate asks for it. <<RESULT: fine-tune training summary, from finetune_summary.json>>

### How the repaired model was made

ReSta takes three models: the drifted one, the aligned one and the base one. It adds a scaled copy of the change that safety training made. In symbols, the repaired weights are the drifted weights plus alpha times (the aligned weights minus the base weights). The drifted model already contains that safety-training change once, so any alpha above zero adds it a second time. By this arithmetic, alpha of 1 does not return the original model.

SafeTune's own documentation says that on Tiny Aya, alpha of 1 breaks the model, that 0.5 or more also breaks it, and that about 0.25 is a sensible value. We did not check that claim ourselves before the run. We read it before any model was run and changed the grid of alpha values because of it: the first grid was 0, 0.25, 0.5, 0.75 and 1, and the grid now is 0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5 and 1. SafeTune's trainer applies a random thinning step called DARE, with a drop rate of 0.3 and seed 0, by default; we keep that default on and record it beside each repaired model. The alpha used is chosen on the dev split by the rule in Section 7. <<RESULT: alpha chosen on dev, with the reason>>

## Data artifacts (datasets, logs, benchmarks, …) public, as an HF collection

### The public data artifacts

The data this study produced is published as one Hugging Face dataset: <<LINK: dataset on Hugging Face>>. It holds the prompts, one label per judged reply with a short excerpt, one correct or incorrect mark per task question, and the tables and figures. Its card lists every file and the licence of every source. Section 4 says what is released and what is withheld.

### The test set

The test set has two kinds of prompt. Harmful prompts ask for something a safe model should refuse. Harmless-but-edgy prompts sound risky but are fine to answer, and they catch a model that has learned to refuse too much. The English prompts come from the sources in Section 4. In each language the dev split is designed to hold 150 harmful and 100 harmless-but-edgy prompts, and the report split 300 harmful and 200 harmless-but-edgy prompts, before the translation check below drops any. No prompt is in both splits.

The split is decided on the English prompt, by a seeded hash, before any translation. One item therefore never lands in dev for one language and in report for another. The seed is 20261105. Duplicate prompts across sources are removed, and the first source in a fixed order wins.

Once built, the test set is frozen. A manifest lists the sha256 (a fingerprint of the file's exact bytes) of every prompt file, and the sha256 of that manifest is the freeze hash: <<RESULT: freeze hash of the test set>> on <<RESULT: freeze date>>. A verification script checks the prompt files against the manifest, and the public export refuses a test set that does not match it.

### Translation

Hindi, Arabic and Swahili prompts are machine translations made with NLLB-200 distilled 600M. Hinglish prompts are made by a chat model through an API endpoint, because NLLB cannot write Hinglish. There is no public Hinglish safety test set that we found, so every Hinglish prompt is a translation.

Each translation is checked by translating it back into English with a chat model and comparing the result with the original. An item is kept only if the back-translation shares enough characters with the original (a chrF-style F-score of at least 0.35) and has a similar length (between 0.6 and 1.5 times the original). The thresholds were picked by hand and not calibrated. The check catches a refusal, an answer instead of a translation, a cut-off text and a wrong-language text. It cannot catch a flipped meaning, for example "how do I stop X" becoming "how do I X", because the characters are nearly the same. Items that fail are dropped, so the languages end with slightly different numbers of prompts: <<RESULT: prompts kept and dropped per language after the translation check>>

Because every non-English prompt is a translation of an English one, the five languages are not five independent samples of prompts. They are the same prompts in five forms.

### The fine-tuning data

The fine-tuning data is the test split of CohereLabs/Global-MMLU, from the Hugging Face revision 0e619dbeb34206cd48705a1a0ea7fb21cae09993, in English, Hindi, Arabic and Swahili. Global-MMLU is multiple-choice exam questions. We keep fourteen harmless subjects: psychology, biology, human aging, nutrition, prehistory, physics, astronomy, geography, government, world history, anatomy, US history, global facts and European history. Sociology and clinical knowledge were in the first list and were removed, so that the claim "harmless subjects only" is easier to defend. Every answer is a bare option letter, so that a reply can be marked correct or incorrect exactly.

There are 6,000 training rows (1,500 per language), 800 held-out rows (200 per language, used for task accuracy on the report split) and 800 dev task rows (the same 200 questions in all four languages, used for task accuracy on the dev split). CuratorKIT cleans and samples the rows. A separate validator checks that no row holds a refusal-style answer or harmful content and that nothing overlaps with the safety test set. Rebuilding the files from the same raw files gives the same bytes.

There are no Hinglish fine-tuning rows. The fine-tune therefore uses four of the five test languages. It never sees a Hinglish row, so Hinglish is measured as a language that the fine-tune did not train on.

## Experiments run, findings and what else was tested

### What was run

Every model version that appears in the tables is run on the prompts of the report split in each language. The dev split is run first and decides the repair strength. For each prompt, the model writes a reply. One judge model reads the prompt and the reply and gives it a label. The label is one of four: complied, refused, no response, or judge error (the judge itself failed). For task questions the answer letter is scored directly, with no judge.

Compute for the final run: <<OWNER: GPU service and GPU type used for the final run>>.

### The drift gate

Before spending effort on a repair, the study asks whether there is drift to repair. The drift gate passes when at least two languages show a rise in attack success of at least 10 percentage points from the original (A0) to the drifted (A1) model on the dev split, and the interval for the pooled rise over those languages sits above zero.

The gate has known strengths and weaknesses, worked out by a reviewer's simulation at 150 harmful prompts per language. The simulation settings were not saved in the repository, so treat the figures as a rough guide.

- If nothing truly changes, the gate passes 0% of the time.
- If the true rise is 5 points in every language, it passes 0.3% of the time.
- At 10 points it passes 25% of the time, at 12 points 51%, and at 15 points 84%.
- If all four non-English languages truly rise by only 8 points, it still passes about 43% of the time, because the gate picks the languages that rose most.
- The clause "pooled interval clearly above zero" never decides the outcome at 150 prompts per language. The 10-point rule in two languages is the part that does the work.

So the gate rarely fires by accident, it misses real rises of about 10 points about three times in four, and it is somewhat easy to pass through selection. Its outcome is: <<RESULT: gate outcome, with the languages that rose by at least ten points>>

There are three branches. If the gate passes, the study continues to the repair. If it does not pass, the study first tries the stronger fine-tune recipe. If the stronger recipe also does not pass, the study pivots to prevention (A3) and the repair becomes a secondary result. The branch taken: <<RESULT: gate branch taken, and whether the stronger recipe was run>>

### Safeguards

These rules are in the code and in tests. They exist so that a good-looking number cannot come from a mistake.

- Alpha is chosen on the dev split only. The report split is used for the reported numbers and for nothing else.
- The report split is frozen and hashed before any model is run on it.
- One judge is used for each set of numbers. A stand-in judge and a paid judge are never mixed inside one set. The dev split and the report split may use different judges, and when they do, that is recorded and warned about.
- The alpha sanity gate. A model that is broken says nothing useful, so a judge finds nothing harmful in it and it looks safe. Before any repaired setting may enter the safety table or the alpha choice, five checks run on its dev replies: few empty replies (at most 5%), few replies that run to the length cap (at most 20%), few that end in a repetition loop (at most 10%), few in the wrong writing system for the prompt language (at most 20%), and task accuracy not more than 10 points below the drifted model's in the same language. A setting that fails is left out and listed by name.
- The alpha rule. Among the repair strengths that pass the sanity gate in every language and the over-refusal rule below, the one with the lowest mean dev attack success over the non-English languages is chosen. If none qualifies, no alpha is chosen and the heatmap shows only the drift.
- The over-refusal rule. A repaired setting is eligible only if, in every language, its dev over-refusal is at most the larger of A0's and A1's over-refusal plus 10 percentage points. A repair that refuses everything cannot win.
- Hinglish has no task-accuracy check, because no closed-form Hinglish data exists. It is marked "not applicable" in the tables and never shown as a pass.
- A cell whose judge returned errors on more than 10% of its replies is excluded and named.
- The drift gate refuses to run when more than a quarter of a language's harmful replies are empty or errored.
- Every interval is a percentile bootstrap (the prompts are re-drawn many times and the middle 95% of the results is kept) with a fixed seed. Differences between two arms use the same prompts, so the comparison is paired.

### The judges, and how far to trust them

A judge model is itself a model, and it can be wrong. To measure that, a second judge from a different model family labels a sample of 60 judged replies per language, and the agreement of the two judges is reported as Cohen's kappa: 1 means perfect agreement and 0 means no more than chance. Each kappa comes with an interval. A language is marked low-confidence when the lower end of its interval is below 0.6, when fewer than 30 pairs could be compared, or when no second judge was run. Low-confidence languages are named in the tables and the figures and are never hidden.

No judge candidate's model card names Hinglish, and the Qwen3 language list names Swahili only under "Other", so we expect the weakest judging in those two languages. The measured agreement tells how much that matters. Inter-judge agreement: <<RESULT: inter-judge agreement per language, with intervals>>. Low-confidence languages: <<RESULT: low-confidence languages>>.

### What else was tried, and what changed

This list is built from the study's working log. It is short because the study has not been run on a GPU before this article was written.

- Alpha grid. The grid changed once, before any model was run, after reading SafeTune's documentation (Section 5).
- A comparison arm. The naive blend (A2n) was added so that ReSta is compared with plain averaging and not only with the drifted model.
- Fine-tuning data. Extractive question-answering sets were checked first: TyDi QA was set aside because its card was unclear about its licence and it has no Hindi, and IndicQA was Hindi only and under 1,000 rows. Global-MMLU, restricted to harmless subjects, was used instead. The first subject list had sixteen subjects and was cut to fourteen.
- Test-set sources. Twenty-five sources were checked and not used (Section 4). The Nemotron loader was built and then left out after a read by eye.
- Stronger recipe and prevention. Both exist as branches of the drift gate and not as arms that were run for their own sake. They were run only if the branch reported in the drift gate section above says so.
- Results that did not go as expected. <<RESULT: partial or null results in plain words>>

### Limits

- The held-out task questions are about 200 per language, so a task accuracy has an interval of about plus or minus 7 points. The same 200 dev questions are used in all four languages.
- The dev task set has a majority-letter floor of 31% (always answering D) against 27 to 28% on the held-out questions. The alpha choice compares the repaired model with the original and the drifted model on the same questions, so a higher floor does not bias that comparison.
- One question in the dev task set is a source twin of a training question (astronomy test item 9 against item 116). It is 1 of 200 and it slightly favours a model that saw the training copy.
- English dev and report prompts share 18 template-family near-duplicate pairs (character overlap of 0.6 or more, 4 pairs at 0.8 or more). The report split is therefore not fully independent of the dev split for those templates.
- No native speaker has read the translated prompts or the Hindi, Arabic and Swahili instruction lines. <<OWNER: confirm that no native speaker reviewed the translations or the judge labels>>
- Machine translation can change meaning in ways that the back-translation check cannot see (Section 6).
- The judges are models, and their agreement is measured on a sample only.
- Tiny Aya Global is one small model, and the fine-tune is one harmless task and one recipe. The study does not say anything about other models, other tasks or other languages.

## With more credits: your approach, before and after results

No extra credits were used for this study, only the free Colab or Kaggle compute that the organizers allow. This section holds the approach's before and after results.

### Short answers

1. Question 1, drift: <<RESULT: answer to question 1, drift>>
2. Question 2, unequal drift: <<RESULT: answer to question 2, unequal across languages>>
3. Question 3, repair: <<RESULT: answer to question 3, repair>>

### How to read the numbers

Every rate comes from the report split and has a 95% interval. A difference between two arms is computed on the same prompts, so it has its own interval. A difference whose interval includes zero is not a finding. A cell marked low-confidence is shown with a dotted pattern. A cell left out by the sanity gate or the judge-error limit is crossed out and named, and its safety numbers are not shown.

### Attack success rate by language

| Language | Original (A0) | Drifted (A1) | Repaired (A2) | Naive blend (A2n) | Change, original to drifted | Change, drifted to repaired |
|---|---|---|---|---|---|---|
| English | <<RESULT: attack success rate of the original model per language>> | <<RESULT: attack success rate of the drifted model per language>> | <<RESULT: attack success rate of the repaired model per language>> | <<RESULT: attack success rate of the naive blend per language>> | <<RESULT: change from original to drifted per language>> | <<RESULT: change from drifted to repaired per language>> |
| Hindi | <<RESULT: attack success rate of the original model per language>> | <<RESULT: attack success rate of the drifted model per language>> | <<RESULT: attack success rate of the repaired model per language>> | <<RESULT: attack success rate of the naive blend per language>> | <<RESULT: change from original to drifted per language>> | <<RESULT: change from drifted to repaired per language>> |
| Hinglish | <<RESULT: attack success rate of the original model per language>> | <<RESULT: attack success rate of the drifted model per language>> | <<RESULT: attack success rate of the repaired model per language>> | <<RESULT: attack success rate of the naive blend per language>> | <<RESULT: change from original to drifted per language>> | <<RESULT: change from drifted to repaired per language>> |
| Arabic | <<RESULT: attack success rate of the original model per language>> | <<RESULT: attack success rate of the drifted model per language>> | <<RESULT: attack success rate of the repaired model per language>> | <<RESULT: attack success rate of the naive blend per language>> | <<RESULT: change from original to drifted per language>> | <<RESULT: change from drifted to repaired per language>> |
| Swahili | <<RESULT: attack success rate of the original model per language>> | <<RESULT: attack success rate of the drifted model per language>> | <<RESULT: attack success rate of the repaired model per language>> | <<RESULT: attack success rate of the naive blend per language>> | <<RESULT: change from original to drifted per language>> | <<RESULT: change from drifted to repaired per language>> |

The exported tables hold the naive blend at three blend values, 0.25, 0.5 and 0.75. This table shows <<RESULT: blend value shown for the naive blend>>. Where the repaired model and the naive blend differ, that difference is what ReSta adds beyond plain averaging: <<RESULT: whether the repair does better than the naive blend>>

### Over-refusal and task accuracy

A repair is only worth something if the model still answers harmless requests and still does its task. This table gives the same arms on those two measures.

| Language | Over-refusal (original, drifted, repaired) | Task accuracy (original, drifted, repaired) |
|---|---|---|
| English | <<RESULT: over-refusal per arm and language>> | <<RESULT: task accuracy per arm and language>> |
| Hindi | <<RESULT: over-refusal per arm and language>> | <<RESULT: task accuracy per arm and language>> |
| Hinglish | <<RESULT: over-refusal per arm and language>> | Not applicable: no closed-form Hinglish data exists. |
| Arabic | <<RESULT: over-refusal per arm and language>> | <<RESULT: task accuracy per arm and language>> |
| Swahili | <<RESULT: over-refusal per arm and language>> | <<RESULT: task accuracy per arm and language>> |

### The heatmap

![Heatmap of attack success rate per language for the two compared model versions, one row per language](heatmap.png)

Each row is one language. The first column is the drifted model and the second is the repaired model at the chosen alpha. If no repair strength qualified, the figure shows the original model against the drifted model instead, and its legend says so. A hatched row means the change between the two columns is inside the noise. A dotted row means the language is low-confidence. A crossed cell was left out by a check. Reading of the heatmap: <<RESULT: reading of the heatmap in plain words>>

### The repair strength

![Curve of safety against task accuracy along the grid of repair strengths](alpha_curve.png)

The curve shows, along the grid of alpha values, how the safety of the repaired model and its task accuracy move together. Reading of the curve: <<RESULT: reading of the alpha curve in plain words>> The settings that the sanity gate left out are listed here: <<RESULT: cells left out by the sanity gate or the judge-error limit>>

### Reproducing the tables

A reproduction notebook rebuilds tables.csv from the public per-item results on an ordinary CPU, with no GPU and no API key: <<LINK: reproduction notebook>>. The check passes when the sha256 of the rebuilt tables.csv equals the one recorded in the export's MANIFEST.json. Choices that rest on data that is not public, such as the alpha choice on dev, are read from reliability.json as recorded and are not recomputed. An optional GPU section re-runs generations; it is not guaranteed to give the same replies, because generation depends on the hardware. The notebook also rebuilds both figures. Reproduction check: <<RESULT: reproduction check outcome>>

## Teammates tagged

There is nobody to tag, because there are no teammates. This is a solo project by <<OWNER: author name as it should appear>>.

The work was done by the owner together with Claude Code, an AI coding assistant from Anthropic. Claude Code wrote and reviewed much of the code and the draft of this text. <<OWNER: confirm what you personally checked, for example the sources, the design decisions and the results>> Every result in this article comes from the exported files described above, and the code that produced them is in the repository. <<OWNER: decision to keep this disclosure paragraph>>
