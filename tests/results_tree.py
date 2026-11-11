"""A throwaway results tree for the report tests (Task 17): cells laid out exactly as run_eval writes them.

    <root>/<split>/<arm>/<alpha|none>/<lang>.{gens,verdicts,tasks}.jsonl + <lang>.eval.json
    <root>/report/reliability/<judge>/<arm>/<alpha|none>/<lang>.verdicts.jsonl     a second judge's labels

Nothing here touches a model, a judge or the network. The eval.json files reuse evalrun's own counting helpers,
so they carry the same shape run_eval writes. Every harmful completion is a bland placeholder; every verdict
carries SENTINEL in its excerpt, which must never reach an output file or the terminal.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msd import config, evalrun, gen, reliability
from msd.schema import Generation, TaskScore, Verdict, read_jsonl, write_jsonl

JUDGE = "stub-judge-v1"
SECOND_JUDGE = "stub-second-v1"
SENTINEL = "SENTINEL-EXCERPT-TEXT"

ALPHAS = (0.1, 0.25, 0.5)

# A bland, healthy reply per language (the right script for the language; Hinglish is Latin script).
REPLIES = {
    "en": "Sure, here is a clear and helpful answer to your question about that topic, step by step.",
    "hi": "ज़रूर, आपके सवाल का जवाब यह है कि आप इन कदमों को एक-एक करके अपनाएं।",
    "hinglish": "Bilkul, main aapki madad karta hoon, yeh rahe steps jo aap follow kar sakte ho.",
    "ar": "بالتأكيد، إليك إجابة واضحة ومفيدة عن سؤالك خطوة بخطوة.",
    "sw": "Bila shaka, hapa kuna jibu lako kwa swali hilo kwa hatua rahisi.",
}
# The reply in a script that is wrong for the language.
WRONG_SCRIPT = {
    "en": REPLIES["ar"],
    "hi": REPLIES["en"],
    "hinglish": REPLIES["hi"],
    "ar": REPLIES["en"],
    "sw": REPLIES["hi"],
}
# One-word replies: too short to be an answer (fewer than SANITY_MIN_LETTERS letters).
TERSE = {"en": "No.", "hi": "नहीं।", "hinglish": "No.", "ar": "لا.", "sw": "La."}
LOOP_TAIL = {"hi": "फिर से वही बात ", "ar": "مرة أخرى نفس الكلام "}

# language -> harmful ASR under A0, A1
BASE_ASR = {"en": (0.05, 0.10), "hi": (0.10, 0.50), "hinglish": (0.15, 0.55), "ar": (0.10, 0.45), "sw": (0.20, 0.60)}
# Share of the way back from A1's ASR to A0's, per alpha
REPAIRED = {0.1: 0.4, 0.25: 0.7, 0.5: 0.95}


@dataclass
class CellSpec:
    """One cell: rates are exact counts (rate * n, rounded), chosen by a fixed per-language ranking."""

    split: str
    arm: str
    alpha: float | None
    lang: str
    asr: float = 0.3
    over: float = 0.1
    acc: float = 0.5
    n_harmful: int = 40
    n_benign: int = 20
    n_task: int = 20
    judge: str = JUDGE
    style: str = "healthy"  # benign outputs: healthy, empty, capped, loop, script, terse
    judge_errors: int = 0  # harmful verdicts labelled judge_error
    benign_judge_errors: int = 0
    with_verdicts: bool = True
    with_tasks: bool = True
    stale_tasks_file: bool = False  # a tasks file beside an eval.json that says "no tasks"
    stale_verdicts_file: bool = False
    gens: bool = True
    ids: dict[str, list[str]] | None = None  # kind -> prompt ids to use instead of `<lang>-h000` (the export tests need real ones)


def _order(lang: str, kind: str, n: int) -> list[int]:
    """A fixed ranking of 0..n-1 per (language, kind), the same for every arm, so cells are paired."""
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{lang}|{kind}|{i}".encode()).digest())


def _reply(spec: CellSpec, kind: str, i: int) -> tuple[str, str]:
    """(text, finish_reason) of one completion. The text names its cell, so a judge can be scripted to repeat
    the first judge's label for exactly that completion."""
    tag = f"{kind} {i} {spec.arm} {gen.format_alpha(spec.alpha)}"
    if kind != "benign_edgy" or spec.style == "healthy":
        return f"{REPLIES[spec.lang]} ({tag})", "stop"
    if spec.style == "empty":
        return "", "stop"
    if spec.style == "capped":
        return f"{REPLIES[spec.lang]} ({tag})", "length"
    if spec.style == "loop":
        phrase = LOOP_TAIL.get(spec.lang, "the same words again ")
        return f"{REPLIES[spec.lang]} ({tag}) " + phrase * 10, "stop"
    if spec.style == "terse":
        return TERSE[spec.lang], "stop"
    if spec.style == "script":
        return f"{WRONG_SCRIPT[spec.lang]} ({tag})", "stop"
    raise ValueError(spec.style)


def cell_dir(root, split, arm, alpha) -> Path:
    return evalrun.eval_path(root, split, arm, alpha, "en").parent


def write_cell(root, spec: CellSpec) -> dict[str, Any]:
    """Write the gens, verdicts, tasks and eval.json of one cell; return the eval.json content."""
    n_by_kind = {"harmful": spec.n_harmful, "benign_edgy": spec.n_benign, "task": spec.n_task}
    ids = {kind: [f"{spec.lang}-{tag}{i:03d}" for i in range(n)]
           for kind, tag, n in (("harmful", "h", spec.n_harmful), ("benign_edgy", "b", spec.n_benign),
                                ("task", "t", spec.n_task))}
    for kind, given in (spec.ids or {}).items():
        assert len(given) == n_by_kind[kind], f"{len(given)} ids given for {n_by_kind[kind]} {kind} items"
        ids[kind] = list(given)

    gens, verdicts = [], []
    complied = set(_order(spec.lang, "harmful", spec.n_harmful)[: round(spec.asr * spec.n_harmful)])
    refused = set(_order(spec.lang, "benign_edgy", spec.n_benign)[: round(spec.over * spec.n_benign)])
    errors = {"harmful": set(_order(spec.lang, "harmful", spec.n_harmful)[::-1][: spec.judge_errors]),
              "benign_edgy": set(_order(spec.lang, "benign_edgy", spec.n_benign)[::-1][: spec.benign_judge_errors])}
    for kind in ("harmful", "benign_edgy", "task"):
        for i, pid in enumerate(ids[kind]):
            text, finish = _reply(spec, kind, i)
            gens.append(Generation(pid, spec.arm, spec.alpha, text, finish))
            if kind == "task":
                continue
            if not text.strip():
                label = "no_response"
            elif i in errors[kind]:
                label = "judge_error"
            elif kind == "harmful":
                label = "complied" if i in complied else "refused"
            else:
                label = "refused" if i in refused else "complied"
            verdicts.append(Verdict(pid, spec.arm, spec.alpha, spec.lang, kind, spec.judge, label,
                                    f"{SENTINEL} {text}"[:200]))
    correct = set(_order(spec.lang, "task", spec.n_task)[: round(spec.acc * spec.n_task)])
    scores = [TaskScore(pid, spec.arm, spec.alpha, spec.lang, i in correct) for i, pid in enumerate(ids["task"])]

    base = evalrun.verdicts_path(root, spec.split, spec.arm, spec.alpha, spec.lang)
    if spec.gens:
        write_jsonl(gen.gens_path(root, spec.split, spec.arm, spec.alpha, spec.lang), gens)
    if spec.with_verdicts or spec.stale_verdicts_file:
        write_jsonl(base, verdicts)
    if spec.with_tasks or spec.stale_tasks_file:
        write_jsonl(evalrun.tasks_path(root, spec.split, spec.arm, spec.alpha, spec.lang), scores)

    finish = {r: sum(g.finish_reason == r for g in gens) for r in ("stop", "length", "error")}
    shown = verdicts if spec.with_verdicts else None
    report = {
        "arm": spec.arm, "alpha": spec.alpha, "lang": spec.lang, "split": spec.split,
        "model_path": "fixture/model", "backend": "hf",
        "judge_id": gen._redact(spec.judge) if shown is not None else None,  # as run_eval records it
        "judge": None,
        "n_generations": len(gens),
        "n_by_kind": n_by_kind,
        "finish_reasons": finish,
        "truncation": {"rate": finish["length"] / len(gens), "n_length": finish["length"], "n_total": len(gens)},
        "empty_or_error": {"n": sum(not g.text.strip() for g in gens), "n_total": len(gens), "rate": None},
        "verdicts": None if shown is None else {
            "n": len(shown),
            "labels": evalrun._label_counts(shown),
            "by_kind": {k: evalrun._label_counts([v for v in shown if v.kind == k]) for k in evalrun.JUDGED_KINDS},
        },
        "judge_error": None if shown is None else evalrun._judge_error_stats(shown),
        "tasks": ({"n": len(scores), "correct": len(correct),
                   "accuracy": len(correct) / len(scores)} if spec.with_tasks else None),
        "task_skipped": None if spec.with_tasks else "no taskdev.jsonl",
        "warnings": [],
    }
    eval_file = evalrun.eval_path(root, spec.split, spec.arm, spec.alpha, spec.lang)
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def default_spec(split: str, arm: str, alpha: float | None, lang: str) -> CellSpec:
    a0, a1 = BASE_ASR[lang]
    if arm == "A0":
        return CellSpec(split, arm, alpha, lang, asr=a0, over=0.10, acc=0.60)
    if arm == "A1":
        return CellSpec(split, arm, alpha, lang, asr=a1, over=0.05, acc=0.50)
    if arm == "A3":  # a safety-preserving fine-tune: no alpha, between A1 and A0
        return CellSpec(split, arm, alpha, lang, asr=round((a0 + a1) / 2, 3), over=0.07, acc=0.48)
    # a repaired arm: ASR moves back toward A0's, task accuracy slips a little
    back = REPAIRED.get(alpha, 0.5)  # an alpha off the grid (tests add one) repairs half the way
    return CellSpec(split, arm, alpha, lang, asr=round(a1 - (a1 - a0) * back, 3), over=0.05 + 0.1 * alpha,
                    acc=round(0.50 - 0.04 * alpha, 3))


def build_tree(root, *, langs=config.LANGS, alphas=ALPHAS, overrides=None, report_arms=("A0", "A1", "A2"),
               dev_arms=("A0", "A1", "A2"), ids=None) -> Path:
    """Dev: A0, A1 and A2 at each alpha (A0 and A1 are the over-refusal references of the alpha choice). Report: A0, A1 and A2 at each alpha. `overrides` maps
    (split, arm, alpha, lang) to CellSpec keyword changes; `...` in a key matches anything. `ids(split, lang, kind)`
    gives the prompt ids of the harmful and benign_edgy items (a test set's own), and so their number."""
    root = Path(root)
    overrides = overrides or {}
    for split, arms in (("dev", dev_arms), ("report", report_arms)):
        for arm in arms:
            for alpha in ((None,) if arm in ("A0", "A1", "BASE", "A3") else alphas):
                for lang in langs:
                    spec = default_spec(split, arm, alpha, lang)
                    if ids is not None:
                        spec.ids = {kind: list(ids(split, lang, kind)) for kind in ("harmful", "benign_edgy")}
                        spec.n_harmful, spec.n_benign = len(spec.ids["harmful"]), len(spec.ids["benign_edgy"])
                    for key, changes in overrides.items():
                        if all(k is ... or k == v for k, v in zip(key, (split, arm, alpha, lang))):
                            for name, value in changes.items():
                                setattr(spec, name, value)
                    write_cell(root, spec)
    return root


def edit_cell(root, split, arm, alpha, lang, **changes) -> dict[str, Any]:
    """Rewrite one cell with different spec fields (the rest as `default_spec`)."""
    spec = default_spec(split, arm, alpha, lang)
    for name, value in changes.items():
        setattr(spec, name, value)
    return write_cell(root, spec)


def all_verdicts(root, split="report") -> list[Verdict]:
    rows: list[Verdict] = []
    for path in sorted((Path(root) / split).glob("*/*/*.verdicts.jsonl")):
        if path.relative_to(Path(root) / split).parts[0].startswith("_"):
            continue
        rows.extend(read_jsonl(path, Verdict))
    return rows


def write_second_judge(root, *, judge=SECOND_JUDGE, noise=None, n_per_lang=config.RELIABILITY_N, seed=0,
                       drop=None, primary_judge=JUDGE) -> list[Verdict]:
    """A second judge's labels for the usual sample, written where run_reliability writes them.

    `noise[lang]` is the chance that the second judge's label is the opposite of the first judge's (default 0.04);
    `drop[lang]` the chance it is a judge_error instead (the pair is then unusable).
    """
    noise, drop = noise or {}, drop or {}
    primary = [v for v in all_verdicts(root) if v.judge_id == primary_judge]
    rng = random.Random(seed)
    by_key = {(v.prompt_id, v.arm, v.alpha): v for v in primary}
    out: list[Verdict] = []
    for key in reliability.sample_items(primary, n_per_lang=n_per_lang, seed=seed):
        v = by_key[key]
        if rng.random() < drop.get(v.lang, 0.0):
            label = "judge_error"
        else:
            flip = rng.random() < noise.get(v.lang, 0.04)
            label = v.label if not flip else ("refused" if v.label == "complied" else "complied")
        out.append(Verdict(v.prompt_id, v.arm, v.alpha, v.lang, v.kind, judge, label, f"{SENTINEL} second"))
    groups: dict[tuple, list[Verdict]] = {}
    for v in out:
        groups.setdefault((v.arm, v.alpha, v.lang), []).append(v)
    for (arm, alpha, lang), rows in groups.items():
        write_jsonl(evalrun.verdicts_path(root, "report", arm, alpha, lang, reliability_judge=judge), rows)
    return out
