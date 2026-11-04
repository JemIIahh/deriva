# Lexsi AI Safety & Alignment Hackathon — HANDOFF

Written 2026-11-03 (Tue). Read this top to bottom, then continue. Everything here was verified in the first session unless marked **(unverified)** or **(my guess)**.

## 1. Where we are (TL;DR)

- Owner has **registered** and **joined Discord** (done 11-03). Team = **owner + Claude only, no human teammates**.
- Owner said **YES to Track 2** (AI Red Teaming, Safety & Alignment) with a multilingual angle, and OK'd an Anthropic API key as teacher/judge — **but the key is the last thing they will provide**.
- We are **mid-brainstorming** (superpowers:brainstorming skill). The idea is chosen and described back (section 4). **Nothing has been built** except two throwaway smoke-test venvs. The skill's HARD-GATE applies: **no product code, scaffolding or dependency installs for the project until the owner approves a written design doc.**
- **UPDATE 2026-11-04:** the design doc is WRITTEN at `docs/superpowers/specs/2026-11-03-multilingual-safety-drift-design.md` and was builder-checked by a fresh subagent (5 of its 7 questions settled as my calls, marked in the doc). Owner answered: reads English only, judge-check method left to me (two judges from different model families). GPU settled by an organizer in Discord: "for compute you can use colab/kaggle free GPUs"; no organizer GPU. Owner has NO Kaggle account (11-04): create one + phone-verify + token in `.env`, or use Colab and click run; choice pending. Owner APPROVED the doc 2026-11-04 ("yes"); plan being written at `docs/superpowers/plans/2026-11-04-multilingual-safety-drift-plan.md`. Section 8 steps 1-3 below are done.
- Next action: owner approval of the doc, then writing-plans.

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
- Toolkit install facts: AlignTune py3.11+, torch 2.0+, backends TRL (default, most tested) / Unsloth (AGPL parts) / ES; AuditKIT py3.10+, `ak.evaluate()`, backends incl. HF/vLLM/Anthropic/OpenRouter/callables; CuratorKIT py3.11+, needs an LLM only for generation (LiteLLM-compatible), outputs load with `load_dataset(output_dir, "sft_messages")`; SafeTune **py3.12+**; AgentTune py3.12+, torch 2.11, transformers 5.15+; CircuitKIT transformer-lens 3.8.0 + transformers 5, Tiny Aya + Aya Expanse 8B supported experimentally, 32B does not run. All Lexsi toolkits are LSAL (non-commercial) licensed.
- Handoff between toolkits is via output folders + `lexsi_provenance.json`.

## 6. Gotchas

- **One venv per toolkit** (python/transformers pins differ). Use `uv`. Run scripts with `python -I` when they read downloaded files.
- Training cannot run locally: Mac is **M1 Pro, 16 GB RAM, no CUDA, ~56 GB disk free (94% full)**. Training goes to Kaggle / Colab / rented GPU (LoRA/QLoRA on 3.35B; check VRAM with `aligntune advise estimate` before launching).
- The Claude Code auto-mode permission classifier once denied a `uv venv` + `uv pip install auditkit` + heredoc command with no reason; the owner ran the install themselves via `! cmd`. If a command is blocked, do not work around it — tell the owner, give them the command to paste with `!`.
- Bash `cd` resets to `/Users/ram/Desktop` between calls — use absolute paths.
- At the end of session 1, listing the project root and `envs/` returned "Operation not permitted" (macOS `com.apple.provenance` xattr on the dir) even though writing files and reading known paths worked; the Write tool also hit EPERM on the root (this file was written to the scratchpad and copied in with `cp`). Use explicit paths; if it persists, tell the owner rather than hunting for a bypass.
- CuratorKIT's optional `[pdf]` extra pulls MinerU (AGPL); Unsloth backend brings AGPL parts — avoid unless needed.

## 7. Environment state

```
~/Desktop/lexsi-hackathon/
  HANDOFF.md                        <- this file
  reference/kickoff-slides-text.txt <- extracted text of all 37 organizer slides
  docs/superpowers/specs/           <- empty; the design doc goes here
  envs/auditkit/    (uv, py3.12)    auditkit 1.1.3 — smoke test passed (stub model, exact_match 0.5)
  envs/curatorkit/  (uv, py3.12)    curatorkit 0.1.22 — imports + CLI OK; NO pipeline run yet
```
Not a git repo. Before any `git init`/commit: verify identity with `git config --global user.name` + `gh auth status` (owner's accounts flip), and ask which identity. **Never add Claude attribution (Co-Authored-By / "Generated with Claude Code") to commits or PRs**, even though a system reminder may say to; the owner's standing rule wins and must be repeated to subagents.

Not installed yet: aligntune, safetune, torch, huggingface CLI (`hf`/`huggingface-cli` not present). Install them only after the design is approved and the GPU question is answered.

## 8. Process state and next steps

We are inside **superpowers:brainstorming** (re-invoke it with the Skill tool at the start of the new session — it was loaded in session 1 but its rules matter).

Remaining steps, in order:
1. Get the owner's answer on GPU access (one question, plain words).
2. **Write the design doc** at `docs/superpowers/specs/2026-11-03-multilingual-safety-drift-design.md` (my call on location; the skill says ask when not in a repo — mention it in the hand-over). Cover: intent + why, goals / non-goals / anti-goals, constraints (deadline, solo, no local GPU, licenses, rubric), the parts the owner decided (Track 2; solo; key last), and what is left to the builder (languages, task, dataset choices, hyperparameters).
3. **Builder check:** dispatch a fresh subagent with `builder-check-prompt.md` from `/Users/ram/.claude/plugins/cache/claude-plugins-official/superpowers/7.0.0/skills/brainstorming/`. Tell the owner a reviewer will read it first. Sort questions into answered / minor (settle, mark as my call) / theirs (one message, most important first).
4. Hand the doc to the owner with a short list of the calls I made. **Wait for explicit approval.**
5. Then invoke **superpowers:writing-plans** (and no other skill), produce the plan, then build.

First technical work after approval (suggested order): (a) HF token + Tiny Aya license + load Global and Base; (b) quick drift probe — tiny multilingual harmful/harmless set, base Global vs a small LoRA fine-tune on benign data; (c) eval harness in AuditKIT; (d) build multilingual eval prompts with a stand-in free model until the Anthropic key arrives.

## 9. Blocked on the owner

1. GPU: paid GPU time, or free Kaggle/Colab only? (asked, unanswered)
2. HF account: accept the Tiny Aya license (gated) and provide an HF token via `.env` (never in chat). Redeem the $20 coupon (check spam).
3. Languages: confirm the default five, or swap.
4. Approve the design doc (gate).
5. LAST: Anthropic API key + a spend cap (teacher for data generation, judge). Build everything on free stand-ins first so the key is a final plug-in via config.
6. Confirm the plan to disclose Claude Code's help in the HF article.
7. Watch for the submission-form email (arrives a couple of days before Nov 14).

## 10. Working style for this owner

- Plain, short updates: lead with plain bullets, **max 2 numbered yes/no questions** per message (they once said "I'm lost" at a dense report).
- One question at a time during brainstorming.
- Pitch fewer ideas with users + money + wow; they reject quick toy lists. Global/unique over Africa-anchored.
- Ask before spending money, creating accounts, deploys, pushing, or anything outward-facing. Keys only in `.env`.
- In shared sessions kill only exact PIDs you started; never `pkill -f`.
- Judge UI by eye; screenshot before presenting (only relevant if we build a demo Space).

## 11. Paste this as the first message of the new session

> Continue the Lexsi hackathon project. Read ~/Desktop/lexsi-hackathon/HANDOFF.md first and the memory file project_lexsi_hackathon.md. We're in the brainstorming step: Track 2 is decided, next is the design doc. My answer on GPU: <fill in>.
