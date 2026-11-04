# Multilingual safety drift: measure it, then repair it without retraining

Design document. Written 2026-11-03 for the Lexsi AI Safety & Alignment Hackathon (Track 2).
Status: DRAFT, not yet approved by the owner. Nothing has been built.

Legend: **[Owner]** = the owner decided or said it. **[My call]** = a default I picked; the owner has not confirmed it and can change it. **[Verify]** = I believe it but have not checked it against the source.

---

## 1. Intent and why

Small multilingual models like Tiny Aya (3.35B parameters, 70+ languages) are released with safety training. Teams then fine-tune them for a local job (a help assistant, a classifier). Fine-tuning on harmless data is known to wear that safety training down. The open question for a multilingual model is whether it wears down evenly. Safety training is usually strongest in English, so we suspect it erodes fastest in lower-resource languages and in code-mixed text (Hindi written with English words, for example). Nobody fine-tuning for Swahili or Hinglish would find out unless they tested in that language.

The project makes that visible and fixes it:

> **Claim:** Fine-tuning a small multilingual model quietly makes it less safe, and not equally in every language. We measure the gap per language and repair it without retraining.

Why this project, for this hackathon: it uses the organizers' own tools as one connected system (data -> fine-tune -> audit -> repair -> audit again), it fits Track 2 (it sits between "Multilingual and code-mixed safety" in 2a and "Repairing safety drift without retraining" in 2b), and the multilingual test set is something SafeTune does not have today, so it is our own contribution.

How we are judged (slide 30, double-blind): write-up clarity /10, practical usefulness /10, code/notebook quality /10, overall review /100. The combination is not stated. A small project that is clean, reproducible and honestly written is the target; an ambitious messy one loses.

## 2. What the owner decided

- **[Owner]** Track 2 (AI Red Teaming, Safety & Alignment), with a multilingual angle.
- **[Owner]** Solo: owner + Claude. Registered as a team of 1. No human teammates.
- **[Owner]** OK to use an Anthropic API key as teacher / judge model. The owner supplies the key LAST, so everything is built on free stand-ins first and the key is a config change at the end.
- **[Owner]** Registered and joined the Discord. Organizer ruling (Discord ticket, 2026-11-03): teacher, validation, comparison models and AuditKIT judges can be any model; only the core submission model must be one of the 7 Cohere models.
- **[Owner]** Reads English only (answered 2026-11-04) and leaves the judge-check method to me (section 6).
- **[Organizers]** Compute: an organizer replied in Discord (screenshot seen 2026-11-04): "for compute you can use colab/kaggle free GPUs". No organizer-provided GPU. Same reply: Cohere models primarily; other models only to further the submission (teacher / validation etc.).
- **Not yet answered by the owner:** whether they have a phone-verified Kaggle account and will share its API token (section 9), whether to disclose Claude Code's help in the article, which account publishes the public artifacts.

## 3. Goals, non-goals, anti-goals

**Goals**
1. A per-language, before/after picture of safety for one fine-tuned Tiny Aya: how often it answers harmful requests it should refuse (attack success), how often it refuses harmless requests (over-refusal), and how good it is at the task. At least three metrics, as the website asks for more than one.
2. A repair (SafeTune ReSta) that restores safety while keeping most of the task quality, shown with a safety-vs-task-quality curve over its `alpha` setting, and compared with plain fine-tuning.
3. An answer to a specific question: ReSta uses no language data (it works on model weights), so does a language-blind patch restore safety evenly across languages, or does it leave some languages behind? **[Verify]** that ReSta really is data-free; the earlier handoff phrased this as "repair built from English", which was probably wrong.
4. A public multilingual safety test set (harmful prompts, over-refusal prompts, code-mixed variants) with clear labels and scoring. This is our original contribution.
5. One clean notebook that a stranger can run to recompute the headline table and the heatmap from our published per-item labels (guaranteed; CPU only, a few minutes), with a documented optional path to regenerate and re-judge a small subset on a free GPU. **[My call]** A full rerun of the whole pipeline is out of reach on free GPU tiers and a paid judge; the article says so plainly.
6. The Hugging Face article with the 9 required sections and a per-language before/after heatmap.

**Non-goals** (out unless there is spare time after everything above is done)
- CircuitKIT, circuit-level fixes, compression/quantization, tamper resistance, activation steering, unlearning, AgentTune.
- More than one fine-tuning task, more than one repair method, or more than one base model family.
- Multi-turn audits, agentic safety, prompt-injection testing.
- A polished web demo. A small HF Space is optional, and only if the core is finished early.
- Beating anyone's benchmark number. We are building a measured, honest recipe.

**Anti-goals** (would make this a failure even if it technically runs)
- Claiming drift or repair we did not measure. A null or partial result (little drift in some languages, repair that costs too much task quality) is acceptable if reported plainly. Overselling is not.
- Reading judge noise as model differences. Judges are weaker in some languages; we measure that and say it.
- Test leakage: fine-tuning on, or tuning `alpha` on, the prompts we report results on. Use separate dev and report splits.
- Publishing harmful model completions. Publish prompts, labels and scores; do not publish full harmful outputs (see section 6).
- A project only the author can rerun, or an article written like a lab diary.
- Cloning the organizers' demo notebooks with our logo on them.

## 4. The pipeline (4 of the 6 repos, one connected system)

| Step | Repo | What it does here |
|---|---|---|
| 1 | **CuratorKIT** | Builds the harmless multilingual fine-tuning data. With a teacher model, also helps build / translate the multilingual evaluation prompts. |
| 2 | **AlignTune** | LoRA fine-tunes **Tiny Aya Global** on that harmless task. |
| 3 | **AuditKIT** (plus SafeTune's evaluation) | Measures attack success, over-refusal and task quality per language, before vs after, including code-mixed prompts. |
| 4 | **SafeTune ReSta** ("Recover") | Weight-space patch, no retraining. Needs three models: base = Tiny Aya **Base**, aligned = Tiny Aya **Global**, drifted = our fine-tune. Sweep `alpha`. |
| 5 | **AuditKIT** | Re-measures the repaired models; compares with plain fine-tuning. |
| 6 | (publish) | Models, data, logs as public HF collections; the notebook; the article. |

Hand-offs between toolkits go through output folders and `lexsi_provenance.json`.

**Arms to evaluate** (every arm on the same report split, every language):
- A0: Tiny Aya Global, untouched ("before").
- A1: Global + LoRA fine-tune ("after", the drifted model).
- A2: A1 + ReSta at several `alpha` values ("repaired").
- A2n: naive baseline, a straight blend of the weights of A1 and A0 at a few blend values, no ReSta. **[My call]** It costs no GPU time and tells us whether ReSta earns its place. It is also the fallback repair if the ReSta triple below misbehaves.
- A3 *(optional, if time)*: Global + safety-preserving fine-tune (SafeTune Harden) on the same task data, for "prevent vs cure".
- Reference only, cheap: Tiny Aya Base, to show how much safety comes from alignment at all.

**About the ReSta triple:** the patch is built from Global minus Base, so it carries all of Global's post-training (instruction following, multilingual tuning), not only safety. **[My call]** Describe results as "restoring Global's alignment delta", never "restoring safety only", and use the capability set and over-refusal rate to catch side effects on non-safety behaviour. **[Verify]** once SafeTune is installed that this is how ReSta defines its vector, and from the model cards that Global is a clean post-training continuation of Base (the regional variants are merges, Global should not be).

## 5. Constraints

- **Deadline:** submit by end of day **Sat Nov 14 (AoE)**. The submission form arrives by email a couple of days before. Aim to have everything uploaded by Nov 13. Demo Day (Nov 22) is only for shortlisted teams.
- **Owner's other deadlines:** Flip the Switch Nov 7, Builder.io and Railwalker Nov 9, NEAR Nov 14. The owner's attention is scarce; the plan must need few owner actions and no long reviews.
- **Solo, no human teammates.** The owner's own steps are listed in section 9.
- **No local GPU.** The Mac is an M1 Pro, 16 GB RAM, no CUDA, about 56 GB free disk. Training and model evaluation run on Kaggle / Colab (free tier; an organizer confirmed in Discord on 2026-11-04 that free Colab / Kaggle GPUs are the compute route; see section 9). Local machine is for data prep, orchestration and small CPU jobs only.
- **Environments:** one virtual environment per toolkit, made with `uv`. Python needs differ (AlignTune 3.11+, AuditKIT 3.10+, CuratorKIT 3.11+, SafeTune 3.12+), and transformers versions differ. Check early that SafeTune and AlignTune install on the free notebook runtimes. Existing smoke-tested envs: `envs/auditkit`, `envs/curatorkit`.
- **Licenses.** Tiny Aya is CC-BY-NC plus Cohere's Acceptable Use Policy, and gated on Hugging Face (owner accepts the license and supplies an HF token). The Lexsi toolkits are LSAL (non-commercial). Anything we publish that derives from Tiny Aya (adapters, the fine-tuned and repaired models) must carry the same terms in its model card. Avoid CuratorKIT's `[pdf]` extra (pulls MinerU, AGPL) and the Unsloth backend (AGPL parts) unless there is no alternative.
- **Everything public on Hugging Face:** models, adapters, logs, datasets, benchmarks, as HF collections. Dataset licenses must allow it; each dataset used gets its license named in article section 4.
- **Secrets:** keys and tokens only in `.env`, never in chat, notebooks or repos. The Anthropic key needs an owner-set spend cap.
- **Process:** nothing is built (code, scaffolding, installs for the project) until the owner approves this document. Then the next step is the implementation plan.
- **Git:** the project folder is not a git repo. Before any `git init`, commit or push: ask which identity, and check `git config --global user.name` and `gh auth status`. No AI-attribution trailers in commits or PRs (owner's standing rule).

## 6. Evaluation design

**Languages [My call]:** English, Hindi, Hinglish (Hindi in Latin script, mixed with English), Arabic, Swahili. Reasons: spread of resource levels, three scripts plus a code-mixed case, all covered by Tiny Aya's 70+ languages, and public multilingual jailbreak data exists for at least Arabic and Swahili **[Verify]**. Hindi and Hinglish prompts would probably need teacher-model translation, which makes translation quality a known risk. The owner reads English only **[Owner]**, so no human on our side can check Hindi, Hinglish, Arabic or Swahili translations. **[My call]** Translate the report split with two translators from different model families and keep only items whose back-translation to English preserves the original meaning (checked by a third pass); report how many items were dropped per language. The set is swappable as long as it keeps (a) English, (b) at least one low-resource language, (c) at least one code-mixed variety.

**Metrics, per language and per arm**
1. **Attack success rate:** share of harmful prompts the model complies with. Report direct requests, and code-mixed or lightly obfuscated variants as a second row. Plain wording first; fancy jailbreak wrappers only if time allows.
2. **Over-refusal rate:** share of harmless-but-edgy prompts the model wrongly refuses (XSTest-style, which is English-only today; other languages need translation) **[Verify]**. This is what stops "repair" from being "refuse everything".
3. **Task quality:** **[My call]** the x-axis of the headline safety-vs-quality curve is accuracy on a held-out set of the fine-tuning task itself. So the task must have closed-form answers (multiple choice, a short extracted answer, or a label) that score without a judge. A multiple-choice reading-comprehension set that exists in these languages (Belebele-style, covers romanized Hindi **[Verify]**) is the second, cross-check axis and a capability-retention measure.
4. **Judge reliability** (the owner reads English only and delegated the method to me, 2026-11-04): on a sample of about 60 items per language, compare two judges from different model families (the final judge, and a second one the builder picks; check which of our languages each officially supports **[Verify]**) and report agreement per language (Cohen's kappa with a confidence interval). It is inter-judge agreement and the article calls it that; it is not human agreement. **[My call]** Claude spot-checking alone does not count, since it is circular when Claude is also the final judge and a translator. English is the one language with a human check: the owner may optionally look over about 30 English verdicts (about 10 minutes) near the end. **[My call]** Any language with kappa below 0.6 is marked low-confidence in the heatmap and the article, not hidden. Without this check the heatmap cannot be trusted in Swahili and Hinglish, where no judge is likely to be formally validated.

**Sizes [My call]:** aim for roughly 300 harmful and 200 harmless-edgy prompts per language, so that a rate near 50% has a confidence interval of about plus or minus 5 points. Report bootstrap confidence intervals everywhere; mark cells whose before/after difference is inside the noise. A dev split (for choosing `alpha` and judge prompts) and a separate report split (for the numbers in the article).

**Public data to start from [Verify each license before use]:** MultiJail-style multilingual jailbreak sets, XSTest (over-refusal), HarmBench / AdvBench (English), Belebele (capability). Cite each with its license in article section 4. Our released test set is built from permitted sources plus our own translations / additions, with provenance recorded.

**Safe publishing [My call]:** release prompts, labels and scores. Do not release complete harmful completions; logs keep the judge label plus a short truncated or redacted excerpt. Public per-item logs keep: prompt id, language, arm, `alpha`, judge label, judge id, and the excerpt. That is enough to recompute every table and the heatmap. Keep harm categories to a small fixed list. Ask the organizers in the Discord ticket channel if the "everything public" rule conflicts with this.

**Judges:** AuditKIT's judge can be any model. Stand-in until the key arrives: a free or open judge model (builder picks, for example a small open safety classifier, or a free-tier hosted model). Final judge: Claude via the Anthropic key. Record which judge produced every number; if the judge changes mid-project, rerun the affected numbers rather than mixing them.

**Freeze order and the key [My call]:** the report split is built and frozen by about Nov 6, with a non-Claude stand-in translator (this also avoids a Claude model refusing to translate harmful prompts). Baselines run against it before the Anthropic key arrives. If the key arrives by about Nov 11, the final numbers are re-judged with it; if not, the stand-in judge defines the headline and the article says so. Only the report split is judged by the paid model, using the cheapest tier that passes the reliability check. Rough size: about 2,500 prompts times 8 or more arm and `alpha` settings, about 20,000 judgments. The owner sets the spend cap when handing over the key; I give a cost estimate before that.

## 7. Fine-tuning task and data

**[My call]** A public-services question-answering assistant: given a short notice about an everyday service (how to renew an ID, register a business, apply for a school place) and a question, it answers in the user's language. Answers are closed-form (multiple choice or a short extracted answer) so the task scores without a judge. Harmless data only, no refusal examples. Two reasons for "harmless only": it is the realistic way teams fine-tune, and refusal-free data is the usual cause of safety drift. The builder may change the task if another one is easier to build and score; it must stay harmless, multilingual and automatically scorable.

**Data source:** CuratorKIT builds or filters the set. Without the Anthropic key, use a free stand-in generator or an existing public multilingual instruction set with a permissive license (the builder should look at Cohere's Aya dataset first, since it is human-written, multilingual and Cohere-owned **[Verify license and language coverage]**). Do not let data quality block the pipeline: the first run can use a small, imperfect set to produce numbers; the key-quality set replaces it for the final run if time allows.

**Fine-tune:** a single LoRA run on Tiny Aya Global through AlignTune's default backend (TRL). Size and hyperparameters are the builder's, constrained by free-GPU memory (a 3.35B model with QLoRA / LoRA should fit a 16 GB T4; check with AlignTune's estimator before launching). The recipe must be strong enough to produce measurable drift; choosing it is part of the drift probe (section 8). Any such choice (learning rate, epochs, rank) is reported in the article, not hidden.

**One extra arm worth considering [My call]:** an English-only variant of the fine-tuning data vs the multilingual one, to see whether drift is worse in languages the fine-tune never saw. Only if A0-A2 are done.

## 8. Risks and the decision rule

1. **No measurable drift.** Tiny Aya Global reportedly scores very high on a multilingual jailbreak test (a press-reported 91.1% safe rate **[Verify]**), so there may be little room to fall, or the fine-tune may not move it. **Gate, around Nov 5 [My call]:** probe on about 150 dev-split harmful prompts per language (never the report split), Global vs a small LoRA fine-tune on harmless data. One language at that size has noise of roughly plus or minus 7 points, so the gate pools. Pass = unsafe rate up by 10 points or more in at least two non-English languages, with the pooled difference across them clearly above zero. If it does not pass: (a) try one stronger recipe, documented (higher learning rate, more epochs or higher LoRA rank; deliberately inducing drift is standard practice and is reported as such); if still nothing, (b) the headline becomes prevention (A3, Harden vs plain fine-tune) and the drift is reported as a measured null. The multilingual audit and the public test set (goal 4) stand in every branch. The pipeline for A0 to A2 is built whichever way the gate goes. I flip the branch and tell the owner straight away; the owner can veto. Either outcome is a valid article, as long as it is measured.
2. **Repair costs too much task quality, or only helps some languages.** That is a finding, not a failure, and the `alpha` curve shows it.
3. **Toolkit install / version clashes on Kaggle or Colab.** Check early; if SafeTune will not install on a free runtime, run ReSta on the Mac (its patch step is advertised as light) or fall back to doing the same weight arithmetic by hand and say so.
4. **Memory.** ReSta handles three 3.35B models (about 20 GB in BF16). Free Colab CPU RAM is small; Kaggle has more. Apply the patch tensor by tensor if needed.
5. **Free GPU quotas.** Kaggle's weekly GPU hours and Colab session limits cap how many evaluation runs fit. Budget the number of (arm x language x prompt) generations early; keep the evaluation prompt count and the number of `alpha` values small enough to fit.
6. **Judge quality in low-resource languages.** Mitigated by the reliability check in section 6.
7. **Time.** Four overlapping owner deadlines. If something has to be cut, cut in this order: A3, the English-only variant, extra `alpha` values, the demo Space. Never cut the reliability check or the clean notebook.

## 9. What the owner must do (and when)

1. **GPU and who presses run:** free Kaggle / Colab GPUs are the official route (an organizer confirmed in Discord on 2026-11-04: "for compute you can use colab/kaggle free GPUs"). The slides mention no organizer-provided GPU and no compute credits beyond the $20 HF coupon (first 800 redeemers) and $200 HF credits for Demo Day teams after Nov 16. A paid GPU is optional and would only make runs faster and the eval bigger. Kaggle needs an account with phone verification (for GPU and internet) and has a command-line API, so with an API token in `.env` the agent can launch and monitor runs itself. Colab has no API, so every Colab run needs the owner to click. **[My call]** Plan for few large, pre-tested, resumable notebooks that write results to a private HF repo (made public at the end); that works either way. **Answered 2026-11-04:** the owner has no Kaggle account they can remember. Options: create one (phone verification, a few minutes) and put the API token in `.env`, or use Colab and click "run" (Colab has no API). The plan works with either; the owner picks.
2. **Hugging Face:** accept the Tiny Aya license (gated), create an HF token and put it in `.env`; say which HF account (and GitHub account, if we publish code there) should own the public artifacts. Redeem the $20 coupon from the Nov 1 email (check spam; needs a payment method on HF; first 800 redeemers).
3. **Approve this document** (gate before any build). Languages are answered (English only; the judge-check method is in section 6). Optional, near the end: look over about 30 English judge verdicts.
4. **Last:** Anthropic API key plus a spend cap (I give a cost estimate first; the owner sets the cap).
5. Confirm the plan to say in the article that Claude Code helped build the project. **[My call, proposed]** Disclose it: the rules are silent on AI help and candor is the safest course. Can also be confirmed with the organizers in the Discord ticket channel.
6. Watch for the submission-form email (due a couple of days before Nov 14) and submit.

## 10. Deliverables

- Public HF collection of **models**: the LoRA adapter / fine-tuned model, repaired model(s) at the chosen `alpha`, (optionally) the Harden model; model cards with license terms.
- Public HF collection of **data**: the multilingual safety test set, the fine-tuning data, evaluation logs (with the redaction policy above).
- **One notebook** (Kaggle / Colab, free tier) that reproduces the headline table and heatmap from the public artifacts.
- **HF article** with the 9 required sections: (1) problem and approach, (2) repos used, for what, and which features, (3) models and licenses, (4) datasets / benchmarks and licenses, (5) model collection, (6) data collection, (7) experiments, findings, what else was tested, (8) approach and before/after results with the per-language heatmap, (9) teammates tagged (none).
- Optional: ROI comparison (small self-hosted model vs a closed API) if cheap to add; optional small HF Space.

## 11. Left to the builder

Dataset choice and exact sizes; hyperparameters and the fine-tune recipe that produces drift; judge model for the stand-in phase; the `alpha` grid; prompt formats and the harm category list; how to split dev and report data; notebook structure; how to run ReSta if SafeTune does not install on a free runtime; whether to add the Harden arm; article structure within the 9 required sections.

## 12. Rough timeline [My call]

- Nov 3-5: this design, approval, implementation plan. Environment check on Kaggle / Colab. Drift probe and the go / no-go decision.
- Nov 6-7: evaluation harness and baseline numbers for A0 (and Base).
- Nov 8-10: fine-tune A1, run ReSta and the `alpha` sweep, re-evaluate.
- Nov 11-12: judge reliability check, final-judge rerun if the key has arrived, heatmap, notebook cleanup, optional extras.
- Nov 13: upload all artifacts, write the article, buffer. Nov 14: submit.
