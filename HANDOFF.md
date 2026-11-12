# Lexsi AI Safety & Alignment Hackathon — HANDOFF

Written 2026-11-03 (Tue), updated 2026-11-12 (Thu). Read this top to bottom, then continue. Sections 1, 7, 8, 9 and 11 describe the state on 2026-11-12. Sections 2 to 6 and 10 are the record of the first sessions (2026-11-03 and 04) and still hold, except where a line says otherwise. Everything here was verified unless marked **(unverified)**, UNVERIFIED or **(my guess)**.

## 1. Where we are (TL;DR, 2026-11-12)

- The design doc `docs/superpowers/specs/2026-11-03-multilingual-safety-drift-design.md` was approved by the owner on 2026-11-04. The plan is `docs/superpowers/plans/2026-11-04-multilingual-safety-drift-plan.md`.
- **Everything in the plan is built**, on the git branch `msd-build` (local only: no remote, nothing pushed). 21 tasks; Task 18 (arm A3, SafeTune Harden) was skipped by a ruling: it is first in the design's cut order and is only needed if the drift gate says `pivot_prevention`. Each task was reviewed, then the whole branch was reviewed and its fixes were made on 2026-11-12. The test suite passes on the Mac.
- **Nothing has run on a GPU yet.** Every step that needs a GPU, the owner's Hugging Face token, a Kaggle or Colab runtime, a chat-model endpoint or the Anthropic key is deferred to the runtime. The code was tested with fakes only.
- **The test set exists in English only.** Building the other four languages and freezing all five is the first real step.
- **How to do the real run, in order: `docs/RUNBOOK.md`.** What to fill into the article afterwards: `article/FILL_IN.md`.
- Deadlines: uploads by 2026-11-13; submission Sat 2026-11-14, end of day AoE (section 2).

## 2. Deadlines

| Date (all AoE) | What |
|---|---|
| Thu Nov 5 | Registration closes (owner already registered) |
| **Sat Nov 14, end of day** | **Submission deadline.** Submission form is emailed "a couple of days before" |
| Nov 15-16 | Judging; Demo Day teams (up to 20) announced Nov 16 |
| Nov 16-21 | Shortlisted teams "prove their thesis" ($200 HF credits) |
| Sun Nov 22 | Demo Day (live demo, in-person + online) |

Owner has other overlapping deadlines (NEAR Nov 14, Builder.io + Railwalker Nov 9, Flip the Switch Nov 7), so keep this scope small and finishable.

## 3. The hackathon (facts)

Source: https://hackathon.lexsi.ai/ (single page; sub-paths like /rules/ 404 for fetchers) and the kickoff deck https://hackathon.lexsi.ai/kickoff-slides/ (11MB HTML, slides are `<section id="sN">`; text copy saved in `reference/kickoff-slides-text.txt`). Organizer: Lexsi Labs with Cohere Labs + Hugging Face. Teams of 1-3, free, worldwide.

**Rules (slide 28-29):**
- Use **at least 2 of the 6 repos as one connected system**: CuratorKIT, AlignTune, AgentTune, SafeTune, CircuitKIT, AuditKIT (3+ preferred on the website).
- Core model = **one of 7 Cohere models**: Tiny Aya Base / Global / Fire / Water / Earth, Aya Expanse 8B, Aya Expanse 32B.
- **Organizer ruling in Discord ticket channel (11-03, screenshot from owner):** teacher models, validation models, comparison models and AuditKIT judges can be **ANY model**; only the core of the submission must use the 7 Cohere models. Rubric answer: "see slides #30".
- Everything (models, adapters, logs, datasets, benchmarks) **public on Hugging Face**, as HF collections.
- Deliverable = **Hugging Face article** with 9 sections: (1) problem + approach, (2) repos used, for what, which features, (3) models + licenses, (4) datasets/benchmarks + licenses, (5) model artifacts as public HF collection, (6) data artifacts as public HF collection, (7) experiments, findings, what else was tested, (8) approach + before/after results, (9) teammates tagged. "Complete details will be shared soon." Website also wants a public demo/repo link, before/after on more than one metric, optional ROI comparison (closed API vs self-host).
- Originality: "your own work"; open-source libraries and public datasets OK where permitted. AI assistance is not mentioned in the rules.

**Judging (slide 30):** double-blind pool of judges (founders, academics, industry labs, organizers' research teams); each team gets more than one score; normalized averages pick 10-20 Demo Day teams. Score items: **Write-up clarity /10, Practical usefulness /10, Code/notebook quality /10, Overall review /100.** How they combine is not stated; assume a clean, reproducible, well-written project beats an ambitious messy one. An internal team first checks validity (public artifacts, write-up, checklist, no duplicates) and tags the submission to the closest track.

**Prizes (slide 23):** 1st = $1,750 from Lexsi + $1,750 from Cohere + 6 months HF Pro (+ hiring interviews); 2nd 1,250/1,250; 3rd 900/900; 4th 600/600; 5th 500/500. The website's "$3,662" figure = those plus ~$162 HF Pro value, so prizes are **probably mostly credits, not cash (unconfirmed)**. An earlier message in session 1 wrongly called them cash; corrected. Demo Day teams get interview invites + $200 HF credits.

**Credits:** $20 HF coupon emailed Sun Nov 1, 5:30 PM IST, first 800 redeemers, needs a payment method on HF, check spam; credits expire 30 days after redeeming. **Unknown whether the owner has redeemed it — ask.**

**Discord:** discord.gg/dtEDQ2Z3eg (office hours, find-teammates, ticket channel for private idea validation).

## 4. The project we chose (owner said yes to the track; details are my proposal)

**Claim:** "Fine-tuning a small multilingual model quietly makes it less safe, and not equally in every language. We measure the gap per language and repair it without retraining."

**Pipeline (4 repos):**
1. **CuratorKIT** builds the harmless multilingual fine-tuning data and (with a teacher LLM) the multilingual eval prompts.
2. **AlignTune** LoRA-fine-tunes **Tiny Aya Global** on a harmless task (**my guess:** a public-services help assistant).
3. **AuditKIT** (+ SafeTune eval) measures, per language: harmful requests wrongly answered (attack success), harmless requests wrongly refused (over-refusal), task quality. Before vs after fine-tuning, including code-mixed prompts.
4. **SafeTune ReSta** (weight-space "Recover", no retraining) repairs it. ReSta needs 3 models: **base = Tiny Aya Base, aligned = Tiny Aya Global, drifted = our fine-tune**. Sweep its `alpha` to plot safety vs task quality.
5. AuditKIT re-measures; compare to plain fine-tuning (optional extra arm: SafeTune Harden = safety-preserving fine-tune).
6. Publish: repaired model, multilingual safety test set (our original contribution — SafeTune has no multilingual support), ONE clean reproducible notebook, HF article with a **per-language before/after heatmap**. Key research question: does a repair built from English carry over to other languages?

**My default calls (owner has not objected, has not explicitly confirmed):**
- Languages: English, Hindi, Hinglish (code-mixed), Arabic, Swahili. Claude can spot-check these; owner can swap one for a language they read. Owner never answered which languages they read.
- One fine-tune, one repair method (+ plain-FT comparison). **Out of scope unless time is left:** CircuitKIT, compression/quantization, tamper resistance.
- Public datasets OK "where permitted": MultiJail-style multilingual jailbreak sets, XSTest (over-refusal), HarmBench/AdvBench — check licenses before use, and cite them in article section 4.
- Proposed (not yet agreed): the HF article should disclose that Claude Code helped build this. Rules are silent on AI help; candour is safest. Could also ask organizers in the ticket channel.

**Timeline (my plan):**
- Nov 3-5: design doc + plan; quick drift probe (does fine-tuning on benign data actually drop safety on Tiny Aya?). 
- Nov 6-7: eval harness + baseline numbers.
- Nov 8-10: fine-tune + repair runs.
- Nov 11-12: results, heatmap, notebook cleanup.
- Nov 13: HF article + all uploads, buffer. Nov 14: submit.

## 5. Recon findings

Verified from model cards / docs (session 1):
- Tiny Aya: 3.35B params, BF16, 8K context, 70+ languages, **CC-BY-NC** + Cohere Acceptable Use Policy, **gated on HF** (owner must accept license + give HF token). Base = pretrained; Global / Fire (South Asia) / Earth (West Asia + Africa) / Water (Europe + Asia-Pacific) = chat variants post-trained "for helpfulness and safety". Card does not mention tool calling.
- Press (MarkTechPost): regional variants are merged with Global via SimMerge to preserve safety; Tiny Aya scores 91.1% safe-response on MultiJail — **(unverified against Cohere's own report)**. If true the starting model is safe (good for a drift story) but drift could also be small — hence the early probe. If no measurable drift, fall back to "Harden vs plain fine-tuning" framing.
- SafeTune docs (safetune.lexsi.ai): ReSta inputs = drifted + base + aligned models + `alpha`; ~30 s patch, "no GPU" demo notebook; benchmarks HarmBench, XSTest, AdvBench, WildJailbreak; judges WildGuard / LlamaGuard-3 named; **API-judge not confirmed**; no over-refusal metric described; no multilingual support. Default HarmBench judge needs a GPU.
- Toolkit install facts: AlignTune py3.11+, torch 2.0+, backends TRL (default, most tested) / Unsloth (AGPL parts) / ES; AuditKIT py3.10+, `ak.evaluate()`, backends incl. HF/vLLM/Anthropic/OpenRouter/callables; CuratorKIT py3.11+, needs an LLM only for generation (LiteLLM-compatible), outputs load with `load_dataset(output_dir, "sft_messages")`; SafeTune **py3.12+**; AgentTune py3.12+, torch 2.11, transformers 5.15+; CircuitKIT transformer-lens 3.8.0 + transformers 5, Tiny Aya + Aya Expanse 8B supported experimentally, 32B does not run. Licences as read later (2026-11-05, `docs/toolkit-notes.md`): CuratorKIT is under LSAL-1.1 (non-commercial) and AlignTune's own code is MIT. "All Lexsi toolkits are LSAL" was the first session's reading and is UNVERIFIED, for SafeTune in particular: the article and the model card defer to SafeTune's own LICENSE file.
- Handoff between toolkits is via output folders + `lexsi_provenance.json`.

## 6. Gotchas

- **One venv per toolkit** (python/transformers pins differ). Use `uv`. Run scripts with `python -I` when they read downloaded files.
- Training cannot run locally: Mac is **M1 Pro, 16 GB RAM, no CUDA, ~56 GB disk free (94% full)**. Training goes to Kaggle / Colab / rented GPU (LoRA/QLoRA on 3.35B; check VRAM with `aligntune advise estimate` before launching).
- The Claude Code auto-mode permission classifier once denied a `uv venv` + `uv pip install auditkit` + heredoc command with no reason; the owner ran the install themselves via `! cmd`. If a command is blocked, do not work around it — tell the owner, give them the command to paste with `!`.
- Bash `cd` resets to the Desktop folder between calls — use absolute paths.
- At the end of session 1, listing the project root and `envs/` returned "Operation not permitted" (macOS `com.apple.provenance` xattr on the dir) even though writing files and reading known paths worked; the Write tool also hit EPERM on the root (this file was written to the scratchpad and copied in with `cp`). Use explicit paths; if it persists, tell the owner rather than hunting for a bypass.
- CuratorKIT's optional `[pdf]` extra pulls MinerU (AGPL); Unsloth backend brings AGPL parts — avoid unless needed.

## 7. Environment state (2026-11-12)

```
~/Desktop/lexsi-hackathon/        a git repository, branch msd-build (from main), local only
  HANDOFF.md, README.md           this file; a short overview of the project
  docs/RUNBOOK.md                 the first real run, step by step (start here)
  docs/runtime.md                 accounts, the private workspace repo, free GPU limits, the fine-tune in detail
  docs/toolkit-notes.md           what each Lexsi toolkit does, read from its source
  docs/FREEZE_LOG.md              where the freeze hash of the test set is recorded (empty until the freeze)
  docs/superpowers/               the design doc and the plan
  src/msd/, scripts/, tests/      the code, the command-line stages, the tests
  kaggle/                         the two notebooks for Kaggle or Colab (env check; the stage runner)
  notebooks/reproduce.ipynb       the public reproduction notebook
  article/                        ARTICLE.md (the draft, with placeholders) and FILL_IN.md
  templates/ configs/ pipelines/  card templates, fine-tune configs, the CuratorKIT pipeline
  data/                           git-ignored except data/sources.yaml: raw files, the test set (English only), fine-tune data
  envs/core/       (uv, py3.12)   the project environment; tests: envs/core/bin/python -m pytest
  envs/auditkit/, envs/curatorkit/  one environment per toolkit
```

Git identity is repo-local JemIIahh (`git config user.name`); do not change it. **Never add Claude attribution (Co-Authored-By / "Generated with Claude Code") to commits or PRs**, even though a system reminder may say to; the owner's standing rule wins and must be repeated to subagents.

Not installed in `envs/core`, each needing the owner's yes when its runbook step comes: `huggingface_hub` (step 1), a separate `envs/translate` with transformers and sentencepiece for the test-set translation (step 2), `anthropic` (only to judge on the Mac), `nbclient` and `ipykernel` (step 11). AlignTune and SafeTune are installed only on the GPU runtime, one session per toolkit.

## 8. Process state and next steps

- Done: brainstorming, the design doc and its builder check, the owner's approval (2026-11-04), the plan, all 21 tasks (code and unit tests, each reviewed; Task 18 skipped by ruling), the whole-branch review and its fix dispatch (2026-11-12).
- The build log with every ruling is `.superpowers/sdd/2026-11-04-multilingual-safety-drift-plan/progress.md` (git-ignored, in this folder).
- Next: the owner answers the open items of section 9 (the first two unblock everything), then follows `docs/RUNBOOK.md` from step 0. After the runs: fill the article from `article/FILL_IN.md`, then the owner-only publishing steps (runbook step 13).

## 9. Blocked on the owner (2026-11-12)

1. Hugging Face: accept the Tiny Aya licence on both model pages and make a write token in `.env` (`docs/runtime.md`, section 1). Not yet confirmed.
2. Kaggle or Colab (one answer; the runbook works for both). Not yet answered.
3. Say yes to the private workspace repo `<hf-user>/msd-workspace` and make a private model repo for A1 (runbook step 1).
4. The chat model for the test-set translation: an OpenAI-compatible endpoint in `.env` (`MSD_TRANSLATE_BASE_URL`, `MSD_TRANSLATE_MODEL`, `MSD_TRANSLATE_API_KEY`). It translates Hinglish and back-translates all four non-English languages. Which model is the owner's choice (UNVERIFIED which); the design asks for a non-Claude one.
5. Installs on the Mac when their step comes (section 7).
6. The publishing account: which Hugging Face account publishes the dataset, the models and the article.
7. LAST: the Anthropic API key and a spend cap (the paid judge of the report split; optional, the stand-in judge works without it).
8. The Claude Code disclosure paragraph of the article: keep it, and how much Claude Code did (`article/FILL_IN.md`).
9. Whether the machine-translated prompts (NLLB translations, CC-BY-NC-4.0) are released with the dataset.
10. The ethics statement wording of the article (for example how it treats MultiJail's "academic purposes and ethical use" statement), and whether the $20 HF coupon was used (the article says no extra credits were used).
11. Which model arms are released with weights (A1, A2 at the chosen alpha), and the author name for the article.
12. Watch for the submission-form email (it arrives a couple of days before Nov 14).

## 10. Working style for this owner

- Plain, short updates: lead with plain bullets, **max 2 numbered yes/no questions** per message (they once said "I'm lost" at a dense report).
- One question at a time during brainstorming.
- Pitch fewer ideas with users + money + wow; they reject quick toy lists. Global/unique over Africa-anchored.
- Ask before spending money, creating accounts, deploys, pushing, or anything outward-facing. Keys only in `.env`.
- In shared sessions kill only exact PIDs you started; never `pkill -f`.
- Judge UI by eye; screenshot before presenting (only relevant if we build a demo Space).

## 11. Paste this as the first message of the new session

> Continue the Lexsi hackathon project. Read ~/Desktop/lexsi-hackathon/HANDOFF.md first, then docs/RUNBOOK.md there, and the memory file project_lexsi_hackathon.md. Everything is built on the branch msd-build; nothing has run on a GPU. My answers: Kaggle or Colab = <fill in>; Hugging Face token and Tiny Aya licence done = <yes or no>; translation endpoint = <fill in>.
