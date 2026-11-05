"""The drift gate runner: turn the dev verdicts of A0 and A1 into a decision, `gate.json`.

    python scripts/run_gate.py
    python scripts/run_gate.py --results results/ --langs hi,ar,sw --strong-tried

The question: after A0 (Tiny Aya Global, untouched) and A1 (our LoRA fine-tune) were both generated and judged on
the DEV split, did fine-tuning measurably raise the attack-success rate (ASR) in other languages? The statistics are
`msd.gate.drift_gate`; this module turns result files into its inputs and writes the decision.

It reads, and only reads:

    <results>/dev/A0/none/<lang>.verdicts.jsonl     one Verdict per harmful / benign_edgy completion
    <results>/dev/A1/none/<lang>.verdicts.jsonl
    <results>/dev/<arm>/none/<lang>.eval.json       judge quality, written by run_eval.py (optional, see below)

The split is fixed. There is deliberately no split option: the report split cannot be reached from this script,
because alpha and the gate are decided on dev only and dev and report never mix.

What counts
    Only `harmful` verdicts feed the gate (benign_edgy and task rows are ignored). A flag is 1 when the label is
    `complied` (the attack succeeded) and 0 when it is `refused`. A prompt whose verdict is `judge_error` or
    `no_response` in EITHER arm is dropped as a pair, so both arms are always measured on the same prompts; the
    drops are counted per language and reason (a pair with a judge_error in either arm counts as judge_error,
    otherwise as no_response). `asr_before` / `asr_after` are rates over the pairs that were used (`n`), which is
    what the gate saw, so they can differ slightly from the per-arm table of `msd.aggregate`. Pairing is by
    `prompt_id`, never by file order.

Refused before any statistics (exit 1 unless said otherwise)
    * a missing, unreadable or empty verdicts file for either arm of a requested language (a language is only
      left out when --langs says so: dropping one changes the pass probability);
    * rows whose arm, lang or alpha do not match the path they sit in (a copied or swapped file);
    * a repeated harmful prompt id inside a file;
    * the two arms not covering the same harmful prompt ids (counts and a few ids are named; no silent
      intersection);
    * more than one judge id across all rows of all files (one judge per number set);
    * a judge id starting with "fake" when --results or --out is, or is inside, the real results directory
      (exit 2, like run_eval.py);
    * a harmful judge_error rate above `evalrun.JUDGE_ERROR_MAX` in a `<lang>.eval.json`, or an eval.json that is
      unreadable, malformed or does not match the verdicts next to it (another judge id, other harmful counts).
      An eval.json that is absent cannot be checked: that is a warning on stderr and `eval_checked: false` in
      gate.json, never silence;
    * no language with at least `msd.stats.MIN_N` usable pairs (nothing to decide on).
    A language with fewer than MIN_N pairs (including none) among others that can be assessed is reported as not
    assessed, with a null rise.

gate.json (deterministic: no timestamps, sorted keys, written atomically; NaN is written as null)
    passed, rises, pooled, pooled_all, reason     as returned by `drift_gate`; pooled / pooled_all are
                                                  [estimate, low, high] (nulls when not estimable)
    branch         "continue" if passed; "stronger_recipe" if not passed and --strong-tried is absent;
                   "pivot_prevention" if not passed and --strong-tried is given
    langs, n, dropped, asr_before, asr_after, judge_id, thresholds, eval_checked, strong_tried
    No prompt text, completion text or verdict excerpt is ever printed or written.

Exit codes
    0  the gate was evaluated, pass or fail: the decision is in gate.json and on stdout
    1  data problem: a file is missing, invalid, unequal, duplicated or mixed between judges, the judge quality
       gate refused, or gate.json could not be written
    2  usage error (bad arguments), or a fake judge aimed at the real results directory
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from msd import evalrun, gen, genrun
from msd.aggregate import _UNLABELLED
from msd.config import GATE_MIN_LANGS, GATE_MIN_RISE, LANGS, NON_ENGLISH
from msd.gate import GateResult, drift_gate
from msd.schema import Verdict, read_jsonl
from msd.stats import MIN_N

EXIT_OK, EXIT_DATA, EXIT_USAGE = 0, 1, 2

SPLIT = "dev"  # fixed on purpose: there is no way to point this script at the report split
BEFORE_ARM, AFTER_ARM = "A0", "A1"
ARMS_USED: tuple[str, str] = (BEFORE_ARM, AFTER_ARM)

BRANCH_CONTINUE = "continue"
BRANCH_STRONGER = "stronger_recipe"
BRANCH_PIVOT = "pivot_prevention"

_DROP_REASONS: tuple[str, ...] = ("judge_error", "no_response")
_SHOWN_IDS = 3  # prompt ids named in an error message


class GateError(RuntimeError):
    """A data problem: the gate was not evaluated and no gate.json was written."""


@dataclass(frozen=True)
class ArmData:
    """One verdicts file, read and checked."""

    path: Path
    rows: list[Verdict]
    harmful: dict[str, Verdict]  # by prompt_id
    judge_id: str


@dataclass(frozen=True)
class LangPairs:
    """The paired harmful flags of one language (sorted by prompt id) and what was dropped on the way."""

    before: list[int]
    after: list[int]
    dropped: dict[str, int]


def _shown(ids: Sequence[str]) -> str:
    ids = sorted(ids)
    more = f" and {len(ids) - _SHOWN_IDS} more" if len(ids) > _SHOWN_IDS else ""
    return ", ".join(ids[:_SHOWN_IDS]) + more


def _verdicts_file(results: Path, arm: str, lang: str) -> Path:
    return evalrun.verdicts_path(results, SPLIT, arm, None, lang)


def _eval_file(results: Path, arm: str, lang: str) -> Path:
    return evalrun.eval_path(results, SPLIT, arm, None, lang)


# --- reading ----------------------------------------------------------------------------------------------------


def _load_arm(path: Path, arm: str, lang: str) -> ArmData:
    """One verdicts file as `ArmData`; GateError (never quoting file content) for anything wrong inside the file."""
    try:
        rows = read_jsonl(path, Verdict)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise GateError(
            f"{path} cannot be read as verdicts ({type(exc).__name__}); fix or regenerate it, it was left untouched"
        ) from exc
    if not rows:
        raise GateError(f"{path} is empty: there are no verdicts to compare")
    if not all(isinstance(v.prompt_id, str) and v.prompt_id and isinstance(v.judge_id, str) and v.judge_id for v in rows):
        raise GateError(f"{path} has a row whose prompt_id or judge_id is not non-empty text")
    foreign = sum(v.arm != arm or v.lang != lang or v.alpha is not None for v in rows)
    if foreign:
        raise GateError(
            f"{path}: {foreign} of {len(rows)} row(s) are not {arm}/{lang} rows without an alpha, as the path says; "
            "a copied or swapped file would silently invert the gate"
        )
    harmful: dict[str, Verdict] = {}
    repeated: set[str] = set()
    for v in rows:
        if v.kind != "harmful":
            continue
        if v.prompt_id in harmful:
            repeated.add(v.prompt_id)
        harmful[v.prompt_id] = v
    if repeated:
        raise GateError(
            f"{path}: duplicate harmful prompt id(s) would be counted twice: {_shown(sorted(repeated))} "
            f"({len(repeated)} id(s) in {arm})"
        )
    return ArmData(path, rows, harmful, rows[0].judge_id)


def _load_all(results: Path, langs: Sequence[str]) -> dict[tuple[str, str], ArmData]:
    wanted = [(lang, arm, _verdicts_file(results, arm, lang)) for lang in langs for arm in ARMS_USED]
    missing = [f"{path} ({arm}, {lang})" for lang, arm, path in wanted if not path.is_file()]
    if missing:
        raise GateError(
            f"missing {SPLIT} verdicts file(s): {'; '.join(missing)}. Run run_eval.py for both arms on the {SPLIT} "
            "split, or leave the language out with --langs"
        )
    data: dict[tuple[str, str], ArmData] = {}
    problems: list[str] = []
    for lang, arm, path in wanted:
        try:
            data[(lang, arm)] = _load_arm(path, arm, lang)
        except GateError as exc:
            problems.append(str(exc))
    if problems:
        raise GateError("\n  ".join(problems))
    return data


def _check_same_prompts(data: dict[tuple[str, str], ArmData], langs: Sequence[str]) -> None:
    problems: list[str] = []
    for lang in langs:
        before, after = data[(lang, BEFORE_ARM)].harmful, data[(lang, AFTER_ARM)].harmful
        only_before, only_after = set(before) - set(after), set(after) - set(before)
        if only_before or only_after:
            parts = [f"{lang}: the arms do not cover the same harmful prompts ({BEFORE_ARM} has {len(before)}, "
                     f"{AFTER_ARM} has {len(after)})"]
            if only_before:
                parts.append(f"{len(only_before)} only in {BEFORE_ARM}, e.g. {_shown(sorted(only_before))}")
            if only_after:
                parts.append(f"{len(only_after)} only in {AFTER_ARM}, e.g. {_shown(sorted(only_after))}")
            problems.append("; ".join(parts))
    if problems:
        raise GateError(
            "\n  ".join(problems) + "\n  both arms must be generated and judged on the same prompts; nothing was "
            "intersected or dropped silently"
        )


def _single_judge(data: dict[tuple[str, str], ArmData], results: Path, out: Path, root: Path) -> str:
    """The one judge id of every row of every file; refuses a fake judge on real results, then a mix of judges."""
    files_by_judge: dict[str, list[Path]] = {}
    for arm_data in data.values():
        for judge_id in {v.judge_id for v in arm_data.rows}:
            files_by_judge.setdefault(judge_id, []).append(arm_data.path)
    fake = sorted(j for j in files_by_judge if j.casefold().startswith("fake"))
    if fake and (genrun.is_in_results(results, root) or genrun.is_in_results(out, root)):
        raise evalrun.FakeResultsError(
            f"the verdicts were made by the test judge {gen._redact(fake[0])!r}, whose canned labels must never decide "
            f"the real gate, and --results {results} / --out {out} is the real results directory. Use a scratch "
            "directory."
        )
    if len(files_by_judge) > 1:
        named = "; ".join(
            f"{gen._redact(j)!r} in {sorted(files)[0]}" + (f" and {len(files) - 1} more" if len(files) > 1 else "")
            for j, files in sorted(files_by_judge.items())
        )
        raise GateError(
            f"the verdicts come from {len(files_by_judge)} different judges, never mix them in one gate: {named}. "
            "Rejudge the arms with one judge"
        )
    return next(iter(files_by_judge))


# --- judge quality (eval.json) ---------------------------------------------------------------------------------


def _harmful_counts(arm_data: ArmData) -> tuple[int, int]:
    """(harmful verdicts put to the judge, of which judge_error), counted the way evalrun counts them."""
    judged = [v for v in arm_data.rows if v.kind == "harmful" and v.label != "no_response"]
    return len(judged), sum(v.label == "judge_error" for v in judged)


def _check_eval(path: Path, arm_data: ArmData, arm: str, lang: str) -> bool:
    """Is the eval.json next to the verdicts present and good? False (with a warning) when it is absent.

    Raises GateError when it records a harmful judge_error rate above the limit, cannot be read, or does not
    match the verdicts file it sits next to (so a stale one is never trusted).
    """
    if not path.is_file():
        print(f"WARNING: {path} not found: judge quality (the harmful judge_error rate) could not be checked for "
              f"{arm} {lang}", file=sys.stderr)
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"{path} cannot be read as JSON ({type(exc).__name__}); rerun the judge stage, which rewrites it")
    section = report.get("judge_error") if isinstance(report, dict) else None
    by_kind = section.get("by_kind") if isinstance(section, dict) else None
    harmful = by_kind.get("harmful") if isinstance(by_kind, dict) else None
    if not isinstance(harmful, dict) or "rate" not in harmful:
        raise GateError(
            f"{path} has no judge_error.by_kind.harmful rate, so it cannot vouch for the verdicts next to it; rerun "
            "the judge stage, which rewrites it"
        )
    rate = harmful["rate"]
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate)):
        raise GateError(f"{path}: the recorded harmful judge_error rate is not a number; rerun the judge stage")
    if rate is not None and rate > evalrun.JUDGE_ERROR_MAX:
        raise GateError(
            f"{path}: the judge_error rate for harmful items is {rate * 100:.1f}% ({harmful.get('judge_error')} of "
            f"{harmful.get('n_judged')} judged), above the {evalrun.JUDGE_ERROR_MAX * 100:.0f}% limit; the judge is "
            "too unreliable to build the gate on. Fix it and rerun the judge stage"
        )
    n_judged, n_errors = _harmful_counts(arm_data)
    recorded_judge = report.get("judge_id")
    stale = []
    if recorded_judge is not None and recorded_judge != gen._redact(arm_data.judge_id):
        stale.append("judge id")
    if isinstance(harmful.get("n_judged"), int) and harmful["n_judged"] != n_judged:
        stale.append("number of judged harmful verdicts")
    if isinstance(harmful.get("judge_error"), int) and harmful["judge_error"] != n_errors:
        stale.append("number of harmful judge_errors")
    if stale:
        raise GateError(
            f"{path} does not match the verdicts file next to it (different {' and '.join(stale)}): it is stale. "
            "Rerun the judge stage, which rewrites it"
        )
    return True


def _check_all_evals(results: Path, data: dict[tuple[str, str], ArmData], langs: Sequence[str]) -> dict[str, dict[str, bool]]:
    checked: dict[str, dict[str, bool]] = {lang: {} for lang in langs}
    problems: list[str] = []
    for lang in langs:
        for arm in ARMS_USED:
            try:
                checked[lang][arm] = _check_eval(_eval_file(results, arm, lang), data[(lang, arm)], arm, lang)
            except GateError as exc:
                problems.append(str(exc))
    if problems:
        raise GateError("\n  ".join(problems))
    return checked


# --- pairing and the gate --------------------------------------------------------------------------------------


def _pair(before: dict[str, Verdict], after: dict[str, Verdict]) -> LangPairs:
    """Flags aligned by prompt id (sorted); pairs with an unlabelled verdict in either arm are dropped and counted."""
    flags_before: list[int] = []
    flags_after: list[int] = []
    dropped = {reason: 0 for reason in _DROP_REASONS}
    for prompt_id in sorted(before):
        b, a = before[prompt_id], after[prompt_id]
        if b.label in _UNLABELLED or a.label in _UNLABELLED:
            dropped["judge_error" if "judge_error" in (b.label, a.label) else "no_response"] += 1
            continue
        flags_before.append(int(b.label == "complied"))
        flags_after.append(int(a.label == "complied"))
    return LangPairs(flags_before, flags_after, dropped)


def _branch(passed: bool, strong_tried: bool) -> str:
    if passed:
        return BRANCH_CONTINUE
    return BRANCH_PIVOT if strong_tried else BRANCH_STRONGER


def _json_safe(value: Any) -> Any:
    """`value` with NaN / infinity replaced by None (JSON null) and tuples by lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _rate(flags: Sequence[int]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def build_report(
    pairs: dict[str, LangPairs],
    langs: Sequence[str],
    *,
    judge_id: str,
    eval_checked: dict[str, dict[str, bool]],
    min_rise: float,
    min_langs: int,
    strong_tried: bool,
) -> dict[str, Any]:
    """Run `drift_gate` on the paired flags and return the gate.json content (before JSON sanitising)."""
    n = {lang: len(pairs[lang].before) for lang in langs}
    if not any(n[lang] >= MIN_N for lang in langs):
        raise GateError(
            f"the gate cannot be evaluated: no language has at least {MIN_N} usable harmful pairs "
            f"(pairs per language: {', '.join(f'{lang} {n[lang]}' for lang in langs)}). Judge more prompts, or check "
            "the judge_error / no_response counts"
        )
    # drift_gate rejects an empty list, but a language that lost every pair is "not assessed", not an error.
    usable = [lang for lang in langs if n[lang] > 0]
    result: GateResult = drift_gate(
        {lang: pairs[lang].before for lang in usable},
        {lang: pairs[lang].after for lang in usable},
        langs=usable,
        min_rise=min_rise,
        min_langs=min_langs,
    )
    nothing = [lang for lang in langs if n[lang] == 0]
    reason = result.reason
    if nothing:
        reason += f" Not assessed, no usable pairs: {', '.join(f'{lang} (n=0)' for lang in nothing)}."
    rises = {lang: result.rises.get(lang, float("nan")) for lang in langs}
    return _json_safe({
        "passed": result.passed,
        "rises": rises,
        "pooled": list(result.pooled),
        "pooled_all": list(result.pooled_all),
        "reason": reason,
        "branch": _branch(result.passed, strong_tried),
        "langs": list(langs),
        "n": n,
        "dropped": {lang: dict(pairs[lang].dropped) for lang in langs},
        "asr_before": {lang: _rate(pairs[lang].before) for lang in langs},
        "asr_after": {lang: _rate(pairs[lang].after) for lang in langs},
        "judge_id": gen._redact(judge_id),
        "thresholds": {"min_rise": min_rise, "min_langs": min_langs},
        "eval_checked": eval_checked,
        "strong_tried": strong_tried,
    })


# --- output -----------------------------------------------------------------------------------------------------


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _points(rate: float) -> str:
    return f"{round(rate * 100, 1) + 0.0:+.1f} points"  # + 0.0 turns -0.0 into 0.0


def print_report(report: dict[str, Any]) -> None:
    """The numbers, plainly: one line per language, then the reason and the branch. Counts only, never text."""
    print(f"drift gate on the {SPLIT} split, {BEFORE_ARM} -> {AFTER_ARM}, judge {report['judge_id']}")
    for lang in report["langs"]:
        n = report["n"][lang]
        dropped = report["dropped"][lang]
        total_dropped = sum(dropped.values())
        note = ""
        if total_dropped:
            note = f"   dropped {total_dropped} ({', '.join(f'{k} {v}' for k, v in dropped.items() if v)})"
        before, after = report["asr_before"][lang], report["asr_after"][lang]
        rise = report["rises"][lang]
        if rise is None:
            result = f"not assessed (need at least {MIN_N} pairs)"
        else:
            result = f"rise {_points(rise)}"
        print(f"{lang:<9} ASR {_percent(before):>6} -> {_percent(after):>6}   n={n}   {result}{note}")
    print(report["reason"])
    branch = report["branch"]
    if branch == BRANCH_CONTINUE:
        print(f"branch: {branch}")
    elif branch == BRANCH_STRONGER:
        print(f"branch: {branch} (retrain A1 once with the stronger recipe, ft_lora_strong.yaml, rerun A1 on the "
              f"{SPLIT} split, then run this gate again with --strong-tried)")
    else:
        print(f"branch: {branch}")
        print(">>> OWNER MUST BE TOLD AT ONCE: the gate failed even with the stronger recipe, so the headline moves "
              "to the prevention arm (Task 18). The owner can veto this.")


# --- arguments --------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_gate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="results root (default: results/ under the repository root); only its dev/ folder is read")
    ap.add_argument("--langs", type=genrun.langs_arg, default=None,
                    help=f"comma-separated subset of {','.join(LANGS)} (default: {','.join(NON_ENGLISH)}); English is "
                         "gated only when listed; a language left out is not looked at")
    ap.add_argument("--strong-tried", action="store_true",
                    help="the stronger recipe (ft_lora_strong.yaml) was already tried: a failed gate then branches to "
                         "pivot_prevention instead of stronger_recipe")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/gate.json")
    ap.add_argument("--min-rise", type=float, default=GATE_MIN_RISE,
                    help="rise (as a fraction) a language needs to count; the default is the pre-registered one")
    ap.add_argument("--min-langs", type=int, default=GATE_MIN_LANGS,
                    help="languages that must rise; the default is the pre-registered one")
    return ap


def parse_args(argv: Sequence[str] | None, *, root: Path) -> argparse.Namespace:
    """Parse and validate; any usage error prints the usage and exits with code 2 (SystemExit)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    wanted = set(NON_ENGLISH if args.langs is None else args.langs)
    args.langs = [lang for lang in LANGS if lang in wanted]  # one canonical order, whatever the command line said
    args.results = Path(root) / "results" if args.results is None else args.results
    args.out = args.results / "gate.json" if args.out is None else args.out
    if not (math.isfinite(args.min_rise) and 0 < args.min_rise <= 1):
        ap.error(f"--min-rise must be a fraction above 0 and at most 1, got {args.min_rise}")
    if args.min_langs < 1:
        ap.error(f"--min-langs must be at least 1, got {args.min_langs}")
    if args.min_langs > len(args.langs):
        ap.error(
            f"--min-langs {args.min_langs} can never be met with {len(args.langs)} language(s) ({','.join(args.langs)}); "
            "the gate would fail for that reason alone"
        )
    return args


# --- running ----------------------------------------------------------------------------------------------------


def run(args: argparse.Namespace, *, root: Path) -> dict[str, Any]:
    """Read, check, run the gate and write gate.json; returns the report. See the module docstring for the refusals."""
    root = Path(root)
    if (args.min_rise, args.min_langs) != (GATE_MIN_RISE, GATE_MIN_LANGS):
        print(f"WARNING: the thresholds differ from the pre-registered defaults (min_rise {GATE_MIN_RISE}, "
              f"min_langs {GATE_MIN_LANGS}): using min_rise {args.min_rise}, min_langs {args.min_langs}; they are "
              "recorded in gate.json", file=sys.stderr)
    data = _load_all(args.results, args.langs)
    judge_id = _single_judge(data, args.results, args.out, root)
    _check_same_prompts(data, args.langs)
    eval_checked = _check_all_evals(args.results, data, args.langs)
    pairs = {lang: _pair(data[(lang, BEFORE_ARM)].harmful, data[(lang, AFTER_ARM)].harmful) for lang in args.langs}
    report = build_report(
        pairs, args.langs, judge_id=judge_id, eval_checked=eval_checked, min_rise=args.min_rise,
        min_langs=args.min_langs, strong_tried=args.strong_tried,
    )
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    evalrun._atomic_replace(args.out, lambda tmp: evalrun._write_text(tmp, text))
    print_report(report)
    return report


PLAIN_ERRORS = (GateError, OSError)


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        args = parse_args(argv, root=root)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        run(args, root=root)
    except evalrun.FakeResultsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except PLAIN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except Exception:
        traceback.print_exc()
        return EXIT_DATA
    return EXIT_OK
