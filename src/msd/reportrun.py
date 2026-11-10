"""The report builder: tables.csv, heatmap.png, alpha_curve.png and reliability.json from a results folder.

    python scripts/make_report.py --verdicts results/ --out results/
    python scripts/make_report.py --verdicts results_paid/ --dev results/ --out results_paid/   # dev lives elsewhere

It reads, and only reads, the result layout run_eval.py writes (`msd.evalrun`):

    <verdicts>/<split>/<arm>/<alpha|none>/<lang>.eval.json              what a cell holds, and its judge quality
                                          <lang>.verdicts.jsonl         one Verdict per harmful / benign_edgy completion
                                          <lang>.tasks.jsonl            one TaskScore per task item
                                          <lang>.gens.jsonl             completions (the dev ones feed the sanity gate)
    <verdicts>/report/reliability/<judge>/<arm>/<alpha|none>/<lang>.verdicts.jsonl     a second judge's labels

and writes four files into --out (default: the --verdicts folder): `tables.csv`, `heatmap.png`, `alpha_curve.png`
and `reliability.json`. Run `python scripts/make_report.py --help` for the exit codes. The tables are a pure
function of the files read: no clock, no randomness, fixed float formatting, so two runs give the same bytes.

What counts
    Walker.   Only `<split>/<arm>/<alpha>/<lang>.eval.json` files make a cell. The folders `_cache`, `_replaced`
              (with its numbered `*.verdicts.jsonl.N`) and `reliability` are never walked as cells; any other
              folder that is not an arm is refused. A verdicts or tasks file without an eval.json is not a cell.
    eval.json decides what a cell holds, never file existence: an eval.json that says "no verdicts" (or "no tasks")
              makes any such file beside it stale and it is ignored; one that promises a file which is missing, or
              whose counts do not match the file, is refused (stale or wrong). The eval.json must agree with the
              path it sits in (arm, alpha, language, split).
    Splits.   The tables come from the `report` split alone. The `dev` split feeds only the sanity gate and the
              alpha choice. They are never combined into one number. A split may be judged by another judge than the
              other (the alpha is chosen on dev; the numbers are the report's): that is recorded and warned about.
    Judge.    One judge id per split. A verdict with no judge id, two ids in one split, or an id that says "fake"
              (after NFKC, ignoring case and invisible characters) is refused (exit 4), always, wherever the files
              are. The second judge (below) may not be the first judge and may not be a test judge either.
    Judge errors.  A cell whose `judge_error` rate (overall, or for either kind) is above the allowance recorded in
              its eval.json (and never above `evalrun.JUDGE_ERROR_MAX`) is excluded and named: its safety rates
              (asr, over_refusal) are left out of the table. Its task accuracy needs no judge and stays.
    Sanity gate (`msd.sanity`).  Every (arm, alpha, language) of a gated arm (`GATED_ARMS`: A2, A2n with its blend
              value in the alpha slot, and A3) is checked on the dev benign-edgy outputs and dev task accuracy
              against A1's. A cell that fails, or cannot be checked (never run on dev, outputs missing, too few task
              items), is excluded like a judge-error cell and listed with the failed check. BASE, A0 and A1 are what
              the others are compared with and are not gated.
    Alpha.    Chosen on dev only (`choose_alpha`): the A2 alpha with the lowest mean dev attack-success rate over
              the non-English languages, among alphas that in EVERY language passed the gate and whose dev
              over-refusal is at most the larger of A0's and A1's plus `config.OVER_REFUSAL_MARGIN` (so a "repair"
              that refuses everything cannot win; when A0's or A1's dev over-refusal is missing, no alpha is
              eligible). None if there is none; the heatmap then shows the drift (A0 to A1) and says so, and the
              reasons are in the warning and in reliability.json.
    Dev.      `--dev PATH` names the folder that holds the dev split when it is not the --verdicts folder (a report
              rejudged into a fresh folder has none). A dev split judged by another judge than the report is
              warned about and recorded in reliability.json, not refused: the stand-in judge may cover dev and the
              paid judge the report split.
    Low confidence.  A language is low-confidence when no second judge was run, when its inter-judge kappa could not
              be assessed, when the lower bound of its kappa interval is below `config.KAPPA_LOW`, or when fewer than
              `config.RELIABILITY_MIN_PAIRS` of its double-judged items could be compared. It is marked in
              tables.csv (`low_confidence`), reliability.json and the heatmap, never hidden.

Files written
    tables.csv        `aggregate.build_table` rows of the report split plus `low_confidence`
    heatmap.png       attack success rate, A1 against A2 at the chosen alpha, one row per language
    alpha_curve.png   safety against task accuracy along the A2 alpha grid
    reliability.json  judge ids, kappa per language, low-confidence languages, excluded cells, the sanity gate,
                      the alpha choice (every candidate with its reasons), what the heatmap compares, warnings. Counts and numbers only: no prompt,
                      completion or excerpt text. Sorted keys, NaN as null, no timestamps.

Previous outputs are deleted before anything is read, and again if writing fails half way (a figure error), so a
failed run leaves no tables that look current and no partial set: either all four files are new or none exists.

Exit codes: 0 done; 1 a data problem (an unreadable, stale or inconsistent file, an unexpected folder, more than
one second-judge folder, a write error); 2 usage error; 3 no report split to read; 4 a judge problem (a verdict
without a judge id, mixed judges, a test judge, a second judge that is the first).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import traceback
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from msd import aggregate, evalrun, gen, reliability, report, sanity
from msd.config import (
    ARMS,
    KAPPA_LOW,
    LANGS,
    NON_ENGLISH,
    OVER_REFUSAL_MARGIN,
    RELIABILITY_MIN_PAIRS,
    SANITY_MAX_CAPPED,
    SANITY_MAX_EMPTY,
    SANITY_MAX_LOOPING,
    SANITY_MAX_WRONG_SCRIPT,
    SANITY_MIN_SCRIPT_SHARE,
    SANITY_TASK_MARGIN,
)
from msd.gaterun import json_safe, looks_fake
from msd.schema import Generation, TaskScore, Verdict, read_jsonl
from msd.stats import MIN_N, InsufficientData, paired_diff_ci

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_DATA, EXIT_JUDGE = 0, 1, 2, 3, 4

OUTPUT_FILES: tuple[str, ...] = ("tables.csv", "heatmap.png", "alpha_curve.png", "reliability.json")
TABLE_COLUMNS: tuple[str, ...] = (*aggregate.COLUMNS, "low_confidence")
FLOAT_DIGITS = 6  # decimals of rate / lo / hi in tables.csv

DEV, REPORT = "dev", "report"
# Every safety arm that is not a reference: the repair arms that take an alpha (A2, and A2n with its blend value in
# the alpha slot) and A3 (Harden, no alpha). BASE, A0 and A1 are what the others are compared with.
GATED_ARMS: frozenset[str] = gen.ALPHA_REQUIRED_ARMS | {"A3"}
REFERENCE_ARMS: tuple[str, ...] = ("A0", "A1")  # their dev over-refusal bounds the over-refusal of a chosen alpha
CHOICE_ARM = "A2"  # the alpha of this arm is the one chosen

CHOICE_RULE = (
    "lowest mean dev attack-success rate over the non-English languages among the A2 alphas that, in every "
    "language, passed the sanity gate, had a measurable attack-success rate, and had a dev over-refusal rate no "
    "higher than the larger of A0's and A1's plus the over-refusal margin (an alpha is not eligible when A0's or "
    "A1's over-refusal, or its own, is unavailable); ties go to the higher mean dev task accuracy, then to the "
    "smaller alpha"
)

_SKIPPED_FOLDERS = ("reliability",)  # besides every name that starts with "_" (_cache, _replaced) or "."
_SHOWN = 3  # file names or ids quoted in one message


def _rid(judge_id: str | None) -> str | None:
    """A judge id as it may be printed or stored: token-shaped parts hidden, as every other record does."""
    return None if judge_id is None else gen._redact(judge_id)


_FINGERPRINT = re.compile(r":t[0-9a-f]{6,}$")  # the template fingerprint a judge id ends with: hf:<model>:t<hex>


def judge_family(judge_id: str) -> str:
    """The judge without its template fingerprint, NFKC and case-folded: provider and model only."""
    return _FINGERPRINT.sub("", unicodedata.normalize("NFKC", judge_id).strip().casefold())


def same_judge(a: str, b: str) -> bool:
    """Are two judge ids the same provider and model? A rerun of the same model under a changed template, or
    with its name in another case, is still that judge: agreement with oneself says nothing."""
    return judge_family(a) == judge_family(b)


class ReportError(RuntimeError):
    """A data problem: nothing was written."""


class JudgeError(ReportError):
    """A verdict set whose judge is missing, mixed or a test judge."""


class NoDataError(ReportError):
    """There is no report split to read."""


# --- cells ------------------------------------------------------------------------------------------------------


def _order(arm: str, alpha: float | None, lang: str) -> tuple:
    """arms in config order, alpha ascending with none first, languages in config order."""
    return (
        ARMS.index(arm) if arm in ARMS else len(ARMS), alpha is not None, alpha or 0.0,
        LANGS.index(lang) if lang in LANGS else len(LANGS),
    )


def lang_order(langs) -> list[str]:
    known = [lang for lang in LANGS if lang in langs]
    return known + sorted(set(langs) - set(known))


def cell_name(arm: str, alpha: float | None, lang: str) -> str:
    """`A2/0.25/hi`, the folder path of the cell without the split."""
    return f"{arm}/{gen.format_alpha(alpha)}/{lang}"


@dataclass
class Cell:
    """One (arm, alpha, language) of one split, read and checked."""

    split: str
    arm: str
    alpha: float | None
    lang: str
    eval_file: Path
    eval: dict[str, Any]
    verdicts: list[Verdict] | None  # None: its eval.json says the judge stage did not run
    scores: list[TaskScore] | None  # None: its eval.json says there are no task scores
    judge_id: str | None

    @property
    def key(self) -> tuple[str, float | None, str]:
        return (self.arm, self.alpha, self.lang)

    @property
    def name(self) -> str:
        return cell_name(self.arm, self.alpha, self.lang)


@dataclass
class Split:
    name: str
    cells: dict[tuple[str, float | None, str], Cell] = field(default_factory=dict)
    judge_id: str | None = None
    notes: list[str] = field(default_factory=list)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
        raise ReportError(
            f"{path} cannot be read as JSON ({type(exc).__name__}); rerun run_eval.py, which rewrites it"
        ) from None


def read_verdict_rows(path: Path) -> list[Verdict]:
    """The rows of a verdicts file. A row with no judge id is a JudgeError; anything unreadable is a ReportError.
    Neither message quotes file content."""
    try:
        lines = path.read_bytes().decode("utf-8").split("\n")  # "\n" only: U+2028 may sit raw inside a string
    except (OSError, ValueError) as exc:
        raise ReportError(f"{path} cannot be read as verdicts ({type(exc).__name__})") from None
    rows: list[Verdict] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise ReportError(f"{path} line {lineno} is not valid JSON; fix or regenerate the file") from None
        judge = obj.get("judge_id") if isinstance(obj, dict) else None
        if not (isinstance(judge, str) and judge.strip()):
            raise JudgeError(
                f"{path} line {lineno}: the verdict has no judge id (missing or empty); a table needs to know "
                "which judge made every number"
            )
        try:
            rows.append(Verdict(**obj))
        except (TypeError, ValueError):
            raise ReportError(f"{path} line {lineno} is not a verdict record; fix or regenerate the file") from None
    return rows


def _check_rows_match_path(path: Path, rows: Sequence, arm: str, alpha: float | None, lang: str) -> None:
    foreign = sum(r.arm != arm or r.lang != lang or r.alpha != alpha for r in rows)
    if foreign:
        raise ReportError(
            f"{path}: {foreign} of {len(rows)} row(s) are not {cell_name(arm, alpha, lang)} rows, as the path says; "
            "a copied or swapped file would silently change the table"
        )


def _read_scores(path: Path) -> list[TaskScore]:
    try:
        return read_jsonl(path, TaskScore)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ReportError(f"{path} cannot be read as task scores ({type(exc).__name__})") from None


def _load_verdicts(root: Path, split: str, arm: str, alpha: float | None, lang: str, data: dict, eval_file: Path):
    """(verdicts or None, judge id or None) of a cell, checked against its eval.json."""
    section = data.get("verdicts")
    if section is None:
        return None, None  # the eval.json says the judge stage did not run: a verdicts file beside it is stale
    path = evalrun.verdicts_path(root, split, arm, alpha, lang)
    if not (isinstance(section, dict) and _is_count(section.get("n")) and isinstance(section.get("labels"), dict)):
        raise ReportError(f"{eval_file} has a verdicts section without n and labels; rerun the judge stage")
    if not path.is_file():
        raise ReportError(
            f"{eval_file} says it holds {section['n']} verdicts but {path} is missing; rerun the judge stage"
        )
    rows = read_verdict_rows(path)
    ids = sorted({v.judge_id for v in rows})
    if len(ids) > 1:
        raise JudgeError(
            f"{path} holds verdicts from {len(ids)} judges {[_rid(i) for i in ids]}; never mix judges in one table"
        )
    judge = ids[0] if ids else None
    recorded = data.get("judge_id")
    if judge is not None and recorded != gen._redact(judge):
        raise JudgeError(
            f"{eval_file} names judge {recorded!r} but the verdicts next to it come from {_rid(judge)!r}; one of them was "
            "replaced by another judge's run. Never mix judges in one table"
        )
    _check_rows_match_path(path, rows, arm, alpha, lang)
    counts = {label: sum(v.label == label for v in rows) for label in ("complied", "refused", "no_response", "judge_error")}
    recorded_counts = {label: section["labels"].get(label, 0) for label in counts}
    if len(rows) != section["n"] or counts != recorded_counts:
        raise ReportError(
            f"{path} does not match the {eval_file.name} next to it ({len(rows)} rows against {section['n']} "
            "recorded, or other label counts): it is stale or wrong. Rerun the judge stage, which rewrites both"
        )
    if judge is None:
        raise JudgeError(f"{path} holds no verdicts, so the judge of {eval_file} cannot be told")
    return rows, judge


def _check_judge_error_section(eval_file: Path, data: dict, rows: Sequence[Verdict]) -> None:
    """The judge_error section an eval.json always carries must be there and agree with the verdicts."""
    section = data.get("judge_error")
    needed = ("n_judged", "judge_error", "rate", "max_allowed", "by_kind")
    missing = [key for key in needed if not isinstance(section, dict) or key not in section]
    if missing:
        raise ReportError(
            f"{eval_file} lacks judge_error.{missing[0]} that run_eval.py always writes, so the allowance of the "
            "judge cannot be read; rerun the judge stage, which rewrites it"
        )
    allowance = section["max_allowed"]
    if isinstance(allowance, bool) or not isinstance(allowance, (int, float)) or not math.isfinite(allowance):
        raise ReportError(f"{eval_file}: judge_error.max_allowed is not a number; rerun the judge stage")
    expected = evalrun.judge_error_stats(rows)
    by_kind = section["by_kind"]
    same = (
        section["n_judged"] == expected["n_judged"] and section["judge_error"] == expected["judge_error"]
        and isinstance(by_kind, dict)
        and all(
            isinstance(by_kind.get(kind), dict)
            and by_kind[kind].get("n_judged") == stat["n_judged"] and by_kind[kind].get("judge_error") == stat["judge_error"]
            for kind, stat in expected["by_kind"].items()
        )
    )
    if not same:
        raise ReportError(
            f"{eval_file} does not match the verdicts next to it (other judge_error counts): it is stale or wrong. "
            "Rerun the judge stage, which rewrites it"
        )


def judge_error_breaches(cell: Cell) -> list[str]:
    """What is above the cell's judge_error allowance: the whole cell and/or one kind, each with its rate.

    The rates are computed from the verdicts (whose counts were checked against eval.json when it was read), never
    taken from the derived `rate` fields of eval.json. The allowance is the one recorded in the cell's eval.json,
    but never looser than `evalrun.JUDGE_ERROR_MAX`.
    """
    if cell.verdicts is None:
        return []
    stats = evalrun.judge_error_stats(cell.verdicts)
    allowance = min(cell.eval["judge_error"]["max_allowed"], evalrun.JUDGE_ERROR_MAX)
    breaches = []
    for name, stat in (("overall", stats), *stats["by_kind"].items()):
        rate = stat.get("rate")
        if rate is not None and rate > allowance:
            breaches.append(
                f"{name} judge_error {rate * 100:.1f}% ({stat['judge_error']} of {stat['n_judged']} judged), above "
                f"the {allowance * 100:.1f}% allowance"
            )
    return breaches


def _load_cell(root: Path, split: str, arm: str, alpha_dir: str, eval_file: Path) -> Cell:
    lang = eval_file.name[: -len(".eval.json")]
    if lang not in LANGS:
        raise ReportError(f"{eval_file}: {lang!r} is not a study language {list(LANGS)}")
    alpha = None if alpha_dir == "none" else float(alpha_dir)
    data = _json_file(eval_file)
    if not isinstance(data, dict):
        raise ReportError(f"{eval_file} is not a JSON object; rerun run_eval.py, which rewrites it")
    for key, want in (("arm", arm), ("lang", lang), ("split", split)):
        if data.get(key) != want:
            raise ReportError(
                f"{eval_file} says {key} {data.get(key)!r} but sits in the folder of {key} {want!r}: a copied or "
                "misplaced file"
            )
    recorded_alpha = data.get("alpha")
    try:
        same_alpha = gen.canonical_alpha(recorded_alpha) == alpha
    except ValueError:
        same_alpha = False
    if not same_alpha:
        raise ReportError(f"{eval_file} says alpha {recorded_alpha!r} but sits in the folder {alpha_dir!r}")

    verdicts, judge = _load_verdicts(root, split, arm, alpha, lang, data, eval_file)
    if verdicts is not None:
        _check_judge_error_section(eval_file, data, verdicts)

    scores: list[TaskScore] | None = None
    section = data.get("tasks")
    if section is not None:
        path = evalrun.tasks_path(root, split, arm, alpha, lang)
        if not (isinstance(section, dict) and _is_count(section.get("n")) and _is_count(section.get("correct"))):
            raise ReportError(f"{eval_file} has a tasks section without n and correct; rerun the task stage")
        if not path.is_file():
            raise ReportError(
                f"{eval_file} says {section['n']} task items were scored but {path} is missing; rerun the task stage"
            )
        scores = _read_scores(path)
        _check_rows_match_path(path, scores, arm, alpha, lang)
        if len(scores) != section["n"] or sum(s.correct for s in scores) != section["correct"]:
            raise ReportError(
                f"{path} does not match the {eval_file.name} next to it ({len(scores)} scores against "
                f"{section['n']} recorded, or another number correct): it is stale or wrong. Rerun the task stage"
            )
    return Cell(split, arm, alpha, lang, eval_file, data, verdicts, scores, judge)


def _alpha_dirs(arm_dir: Path) -> list[Path]:
    found = []
    for path in sorted(arm_dir.iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        try:
            ok = path.name == "none" or gen.format_alpha(float(path.name)) == path.name
        except ValueError:
            ok = False
        if not ok:
            raise ReportError(f"{path}: {path.name!r} is not an alpha folder ('none' or a number such as '0.25')")
        found.append(path)
    return found


def load_split(root: Path, split: str) -> Split:
    """Every cell of one split. Only `*.eval.json` files make a cell; see the module docstring."""
    result = Split(split)
    base = Path(root) / split
    if not base.is_dir():
        return result
    for arm_dir in sorted(base.iterdir()):
        name = arm_dir.name
        if name.startswith(("_", ".")) or name in _SKIPPED_FOLDERS or not arm_dir.is_dir():
            continue
        if name not in ARMS:
            raise ReportError(
                f"{arm_dir}: {name!r} is not an arm folder (expected one of {list(ARMS)}, or _cache, _replaced, "
                "reliability); move it out of the results folder"
            )
        for alpha_dir in _alpha_dirs(arm_dir):
            evals = {p.name[: -len(".eval.json")] for p in alpha_dir.glob("*.eval.json")}
            for loose in sorted(alpha_dir.glob("*.verdicts.jsonl")):
                lang = loose.name[: -len(".verdicts.jsonl")]
                if lang not in evals:
                    alpha = None if alpha_dir.name == "none" else float(alpha_dir.name)
                    result.notes.append(
                        f"ignored {split} {cell_name(name, alpha, lang)}: {loose.name} has no {lang}.eval.json beside "
                        "it, so it is not a cell (rerun run_eval.py for it)"
                    )
            for eval_file in sorted(alpha_dir.glob("*.eval.json")):
                cell = _load_cell(Path(root), split, name, alpha_dir.name, eval_file)
                result.cells[cell.key] = cell

    judges: dict[str, list[Path]] = defaultdict(list)
    for cell in result.cells.values():
        if cell.judge_id is not None:
            judges[cell.judge_id].append(cell.eval_file)
    fake = sorted(j for j in judges if looks_fake(j))
    if fake:
        raise JudgeError(
            f"the {split} verdicts were made by the test judge {gen._redact(fake[0])!r} (in {judges[fake[0]][0]}), "
            "whose canned labels must never go into a report. Rejudge them with a real judge"
        )
    if len(judges) > 1:
        named = "; ".join(
            f"{gen._redact(j)!r} in {sorted(files)[0]}" + (f" and {len(files) - 1} more" if len(files) > 1 else "")
            for j, files in sorted(judges.items())
        )
        raise JudgeError(
            f"the {split} verdicts come from {len(judges)} different judges, never mix them in one table: {named}. "
            "Rejudge the arms with one judge (run_eval.py --rejudge)"
        )
    result.judge_id = next(iter(judges), None)
    return result


def read_second_judge_folder(
    folder: Path, primary: str | None, *, allow_test_judge: bool = False
) -> tuple[str | None, list[Verdict]]:
    """The second judge's id and verdicts from one `<root>/report/reliability/<judge>/` folder, or (None, []).

    Raises JudgeError for a folder that mixes judges, holds a test judge's labels, or holds the first judge's own.
    `allow_test_judge` is for exactly one caller, scripts/run_reliability.py reading back the labels it has just
    written with an injected test judge in a scratch directory (it has already refused a test judge on the real
    results directory); make_report never passes it, so a report never reads canned labels.
    """
    rows: list[Verdict] = []
    for path in sorted(folder.glob("*/*/*.verdicts.jsonl")):
        arm, alpha_dir, name = path.relative_to(folder).parts
        try:
            alpha = None if alpha_dir == "none" else float(alpha_dir)
        except ValueError:
            raise ReportError(f"{path}: {alpha_dir!r} is not an alpha folder ('none' or a number such as '0.25')") from None
        found = read_verdict_rows(path)
        _check_rows_match_path(path, found, arm, alpha, name[: -len(".verdicts.jsonl")])
        rows.extend(found)
    ids = sorted({v.judge_id for v in rows})
    if not ids:
        return None, []
    if len(ids) > 1:
        raise JudgeError(
            f"the second judge's labels in {folder} come from {len(ids)} judges {[_rid(i) for i in ids]}; never mix them"
        )
    if looks_fake(ids[0]) and not allow_test_judge:
        raise JudgeError(f"the second judge {_rid(ids[0])!r} is a test judge; its canned labels say nothing about agreement")
    if primary is not None and same_judge(ids[0], primary):
        raise JudgeError(
            f"the second judge {_rid(ids[0])!r} is the same judge as the first; agreement with oneself is meaningless"
        )
    return ids[0], rows


def load_second_judge(root: Path, primary: str | None) -> tuple[str | None, list[Verdict]]:
    """The second judge's id and verdicts from `<root>/report/reliability/<judge>/...`, or (None, [])."""
    base = Path(root) / REPORT / "reliability"
    folders = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not folders:
        return None, []
    if len(folders) > 1:
        names = ", ".join(p.name for p in folders)
        raise ReportError(
            f"more than one second-judge folder under {base}: {names}. Keep one (move the others away); the "
            "report compares the first judge with a single second judge"
        )
    return read_second_judge_folder(folders[0], primary)


# --- the sanity gate ----------------------------------------------------------------------------------------------


class _Tagged:
    """A Generation seen with its language (a gens file has none), so the path check can use it."""

    def __init__(self, generation: Generation, lang: str) -> None:
        self.arm, self.alpha, self.lang = generation.arm, generation.alpha, lang


def _dev_benign_outputs(root: Path, cell: Cell) -> list[Generation] | None:
    """The generations of a dev cell for its benign-edgy prompts (replies to safety prompts, never task items),
    or None when they cannot be found. `root` is the folder that holds the dev split."""
    if cell.verdicts is None:
        return None  # the benign prompts are told apart by their verdicts
    ids = {v.prompt_id for v in cell.verdicts if v.kind == "benign_edgy"}
    path = gen.gens_path(root, DEV, cell.arm, cell.alpha, cell.lang)
    if not ids or not path.is_file():
        return None
    try:
        gens = read_jsonl(path, Generation)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ReportError(f"{path} cannot be read as generations ({type(exc).__name__})") from None
    _check_rows_match_path(path, [_Tagged(g, cell.lang) for g in gens], cell.arm, cell.alpha, cell.lang)
    outputs = [g for g in gens if g.prompt_id in ids]
    return outputs if len(outputs) == len(ids) else None


def _task_counts(cell: Cell | None) -> tuple[float | None, int | None]:
    """(accuracy, number of items) of a cell from its scored items, which were checked against eval.json when it
    was read; never from the derived `accuracy` field of eval.json. (None, None) without any task items."""
    if cell is None or not cell.scores:
        return None, None
    return sum(score.correct for score in cell.scores) / len(cell.scores), len(cell.scores)


def run_sanity_gate(
    dev_root: Path, dev: Split, report_split: Split
) -> dict[tuple[str, float | None, str], sanity.SanityResult]:
    """The five checks for every (arm, alpha, language) of a gated arm (`GATED_ARMS`) in either split. They run on
    the dev split, which lives under `dev_root`."""
    keys = {key for split in (dev, report_split) for key in split.cells if key[0] in GATED_ARMS}
    a1_accuracy = {lang: _task_counts(dev.cells.get(("A1", None, lang))) for lang in LANGS}
    results = {}
    for key in sorted(keys, key=lambda k: _order(*k)):
        arm, alpha, lang = key
        cell = dev.cells.get(key)
        if cell is None:
            results[key] = sanity.SanityResult(
                (), "this cell was not run on the dev split, so it was never checked", {}
            )
            continue
        accuracy, n_items = _task_counts(cell)
        a1_acc, a1_n = a1_accuracy[lang]
        results[key] = sanity.check_cell(
            lang, _dev_benign_outputs(dev_root, cell), accuracy, a1_acc, dev_task_n=n_items, a1_task_n=a1_n
        )
    return results


# --- the alpha choice ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    alpha: float
    eligible: bool
    reasons: tuple[str, ...]
    mean_asr: float | None
    mean_over_refusal: float | None
    mean_task_acc: float | None
    over_refusal_failures: tuple[str, ...] = ()  # "hi 50% vs 20% allowed": the languages that refused too much


@dataclass(frozen=True)
class AlphaChoice:
    chosen: float | None
    candidates: tuple[Candidate, ...]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pct(rate: float) -> str:
    """"20%", or "12.5%" when the rate is not a whole percent."""
    value = rate * 100
    return f"{value:.0f}%" if abs(value - round(value)) < 0.05 else f"{value:.1f}%"


def choose_alpha(
    rows: Sequence[Mapping],
    blocked: Mapping[tuple[float, str], Sequence[str]],
    *,
    langs: Sequence[str],
    arm: str = CHOICE_ARM,
) -> AlphaChoice:
    """Choose the alpha of `arm` from DEV table rows (`aggregate.build_table`), see `CHOICE_RULE`.

    `blocked` maps (alpha, language) to the reasons a cell must not count (failed or unverified sanity gate,
    judge errors). An alpha is eligible only if every language in `langs` has a cell that is not blocked, an
    attack-success rate from at least `MIN_N` items, and a dev over-refusal rate (from at least `MIN_N` items) no
    higher than the larger of A0's and A1's in that language plus `OVER_REFUSAL_MARGIN`: otherwise the lowest
    attack-success rate would go to an alpha that refuses everything. Fail closed: when A0's or A1's over-refusal
    (rows of arm A0 / A1, no alpha) or the alpha's own is missing or rests on too few items, the alpha is not
    eligible and the reason says which. The mean is over the non-English languages of `langs`, or over all of
    them when there is none. The caller must pass dev rows only: nothing here can tell.
    """
    cells: dict[tuple[float, str], dict[str, Mapping]] = defaultdict(dict)
    refs: dict[tuple[str, str], Mapping] = {}
    for r in rows:
        if r["arm"] == arm and r["alpha"] is not None:
            cells[(r["alpha"], r["lang"])][r["metric"]] = r
        elif r["arm"] in REFERENCE_ARMS and r["alpha"] is None and r["metric"] == "over_refusal":
            refs[(r["arm"], r["lang"])] = r

    def usable(row: Mapping | None) -> bool:
        return row is not None and row["rate"] is not None and row["n"] >= MIN_N

    allowed: dict[str, tuple[float, str] | list[str]] = {}  # lang -> (limit, how it came about) or the missing arms
    for lang in langs:
        missing = [ref for ref in REFERENCE_ARMS if not usable(refs.get((ref, lang)))]
        if missing:
            allowed[lang] = missing
        else:
            rates = {ref: refs[(ref, lang)]["rate"] for ref in REFERENCE_ARMS}
            limit = max(rates.values()) + OVER_REFUSAL_MARGIN
            allowed[lang] = (limit, f"the larger of A0 {_pct(rates['A0'])} and A1 {_pct(rates['A1'])}, plus "
                                    f"{_pct(OVER_REFUSAL_MARGIN)}")
    alphas = sorted({alpha for alpha, _ in cells} | {alpha for alpha, _ in blocked})
    scored = [lang for lang in langs if lang in NON_ENGLISH] or list(langs)

    candidates = []
    for alpha in alphas:
        reasons: list[str] = []
        failures: list[str] = []
        for lang in langs:
            reasons.extend(f"{lang}: {why}" for why in blocked.get((alpha, lang), ()))
            metrics = cells.get((alpha, lang))
            if metrics is None:
                if (alpha, lang) not in blocked:
                    reasons.append(f"{lang}: no dev cell")
                continue
            asr = metrics.get("asr")
            if asr is None or asr["rate"] is None or asr["n"] < MIN_N:
                reasons.append(f"{lang}: attack success rate not measured on enough items")
            limit = allowed[lang]
            over = metrics.get("over_refusal")
            if isinstance(limit, list):
                reasons.append(f"{lang}: over-refusal reference not available (no usable {' or '.join(limit)} dev "
                               "over-refusal rate)")
            elif not usable(over):
                reasons.append(f"{lang}: over-refusal not measured on enough items")
            elif over["rate"] > limit[0] + 1e-9:
                reasons.append(f"{lang}: over-refusal {_pct(over['rate'])} is above the {_pct(limit[0])} allowed "
                               f"({limit[1]})")
                failures.append(f"{lang} {_pct(over['rate'])} vs {_pct(limit[0])} allowed")

        def mean(metric: str, over) -> float | None:
            return _mean([cells[(alpha, lang)][metric]["rate"] for lang in over
                          if metric in cells.get((alpha, lang), {}) and cells[(alpha, lang)][metric]["rate"] is not None])

        candidates.append(Candidate(
            alpha=alpha, eligible=not reasons, reasons=tuple(reasons), mean_asr=mean("asr", scored),
            mean_over_refusal=mean("over_refusal", scored), mean_task_acc=mean("task_acc", scored),
            over_refusal_failures=tuple(failures),
        ))
    eligible = [c for c in candidates if c.eligible and c.mean_asr is not None]
    best = min(
        eligible,
        key=lambda c: (c.mean_asr, -(c.mean_task_acc if c.mean_task_acc is not None else -1.0), c.alpha),
        default=None,
    )
    return AlphaChoice(best.alpha if best else None, tuple(candidates))


# --- low confidence -----------------------------------------------------------------------------------------------


def low_confidence_reasons(
    agr: Mapping[str, tuple[float, float, float]],
    pairs: Mapping[str, int],
    langs: Sequence[str],
    *,
    has_second_judge: bool = True,
) -> dict[str, list[str]]:
    """Why each language in `langs` is low-confidence; a confident language is absent from the result.

    `agr` is `reliability.agreement` (kappa, lo, hi per language; NaN when it could not be assessed), `pairs`
    `reliability.usable_pair_counts`. Reasons, in this order: `no_second_judge` (alone), `not_assessed`
    (alone), `kappa_lower_bound_below_threshold` (the lower bound of the kappa interval is below
    `config.KAPPA_LOW`: the point estimate alone is not trusted), `too_few_pairs` (fewer than
    `config.RELIABILITY_MIN_PAIRS` comparable pairs).
    """
    out: dict[str, list[str]] = {}
    for lang in langs:
        if not has_second_judge:
            out[lang] = ["no_second_judge"]
            continue
        kappa, lo, _ = agr.get(lang, (math.nan, math.nan, math.nan))
        if math.isnan(kappa) or math.isnan(lo):
            out[lang] = ["not_assessed"]
            continue
        why = []
        if lo < KAPPA_LOW:
            why.append("kappa_lower_bound_below_threshold")
        if pairs.get(lang, 0) < RELIABILITY_MIN_PAIRS:
            why.append("too_few_pairs")
        if why:
            out[lang] = why
    return out


# --- the analysis -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Exclusion:
    split: str
    arm: str
    alpha: float | None
    lang: str
    reason: str  # judge_error | sanity_check
    failed_checks: tuple[str, ...]
    detail: str

    @property
    def name(self) -> str:
        return f"{self.split} {cell_name(self.arm, self.alpha, self.lang)}"


@dataclass
class Analysis:
    root: Path
    dev: Split
    report: Split
    second_judge_id: str | None
    exclusions: list[Exclusion]
    sanity: dict[tuple[str, float | None, str], sanity.SanityResult]
    choice: AlphaChoice
    table: list[dict]  # report-split rows plus low_confidence
    report_verdicts: list[Verdict]  # the report verdicts that entered the table
    agreement: dict[str, tuple[float, float, float]]
    pairs: dict[str, int]
    low_confidence: dict[str, list[str]]
    heat: dict[str, Any] | None
    deltas: dict[str, tuple[float, float, float]]
    heat_excluded: dict[tuple[str, str], str]
    curve_excluded: dict[str, list[float | str]]
    not_chosen: dict[float, list[str]]  # alphas not eligible because they refuse too much -> "hi 50% vs 20% allowed"
    notes: list[str]
    warnings: list[str]


def _sanity_text(result: sanity.SanityResult) -> str:
    parts = []
    if result.failed:
        parts.append("failed " + ", ".join(result.failed))
    if result.unverified:
        parts.append("could not be checked: " + result.unverified)
    return "; ".join(parts)


def _paired_deltas(
    verdicts: Sequence[Verdict], before: tuple[str, float | None], after: tuple[str, float | None], table: Sequence[Mapping]
) -> tuple[dict[str, tuple[float, float, float]], list[str]]:
    """Per language, the paired rise in attack success (after - before) over the harmful prompts both arms have a
    label for. A language without enough changed pairs has none. A delta that is more than
    `report.DELTA_TOLERANCE` away from the difference of the two table cells (so many pairs were dropped that it
    describes other prompts) is not drawn, and a warning says so."""
    flags: dict[tuple[str, tuple[str, float | None]], dict[str, int]] = defaultdict(dict)
    for v in verdicts:
        if v.kind == "harmful" and v.label in ("complied", "refused") and (v.arm, v.alpha) in (before, after):
            flags[(v.lang, (v.arm, v.alpha))][v.prompt_id] = int(v.label == "complied")
    rate = {(r["lang"], r["arm"], r["alpha"]): r for r in table if r["metric"] == "asr"}
    deltas: dict[str, tuple[float, float, float]] = {}
    warnings: list[str] = []
    for lang in lang_order({lang for lang, _ in flags}):
        b, a = flags.get((lang, before), {}), flags.get((lang, after), {})
        ids = sorted(set(b) & set(a))
        if not ids:
            continue  # a side was left out (excluded cell) or never labelled: nothing to pair
        try:
            delta = paired_diff_ci([b[i] for i in ids], [a[i] for i in ids])
        except InsufficientData:
            continue
        cell_b, cell_a = rate.get((lang, *before)), rate.get((lang, *after))
        if cell_b and cell_a and cell_b["rate"] is not None and cell_a["rate"] is not None:
            difference = cell_a["rate"] - cell_b["rate"]
            if abs(delta[0] - difference) > report.DELTA_TOLERANCE:
                warnings.append(
                    f"the paired change for {lang} ({delta[0] * 100:+.1f} points over {len(ids)} pairs) is more than "
                    f"{report.DELTA_TOLERANCE * 100:.0f} points from the change between the two cells shown "
                    f"({difference * 100:+.1f}); it is not drawn, so many pairs must have been dropped"
                )
                continue
        deltas[lang] = delta
    return deltas, warnings


def analyze(root: Path, *, with_second_judge: bool = True, dev_root: Path | None = None) -> Analysis:
    """Read the results folder and decide everything; writes nothing. Raises ReportError, JudgeError, NoDataError.

    `with_second_judge=False` leaves the second judge's folder unread (scripts/run_reliability.py, which is about
    to write it): every language then counts as having no second judge. `dev_root` is the folder that holds the
    `dev` split when it is not `root` (a report rejudged into a fresh folder has no dev split of its own).
    """
    root = Path(root)
    if dev_root is not None and not Path(dev_root).is_dir():
        raise ReportError(f"--dev {dev_root} is not a folder: expected a results folder with a dev/ split in it")
    dev_root = Path(dev_root) if dev_root is not None else root
    if not root.is_dir():
        raise NoDataError(f"{root} is not a folder: there are no results to report")
    report_split = load_split(root, REPORT)
    if not report_split.cells:
        raise NoDataError(
            f"no report split under {root}: expected {root / REPORT}/<arm>/<alpha>/<lang>.eval.json files "
            f"(run run_eval.py with --split {REPORT})"
        )
    dev = load_split(dev_root, DEV)
    notes = [*dev.notes, *report_split.notes]
    warnings: list[str] = []
    if not dev.cells:
        where = "" if dev_root == root else f" under --dev {dev_root}"
        warnings.append(
            f"there is no dev split{where}: the sanity gate cannot be run, so every repair cell is excluded and no "
            "alpha can be chosen"
        )
    if dev.judge_id and report_split.judge_id and dev.judge_id != report_split.judge_id:
        warnings.append(
            f"the dev split was judged by {_rid(dev.judge_id)!r} and the report split by {_rid(report_split.judge_id)!r}: the "
            "alpha was chosen with another judge than the one behind the numbers"
        )

    # judge errors: the safety verdicts of an over-allowance cell are left out
    exclusions: list[Exclusion] = []
    judge_blocked: dict[tuple[str, tuple], list[str]] = {}
    for split in (dev, report_split):
        for cell in split.cells.values():
            breaches = judge_error_breaches(cell)
            if breaches:
                judge_blocked[(split.name, cell.key)] = breaches
                exclusions.append(Exclusion(split.name, cell.arm, cell.alpha, cell.lang, "judge_error", (),
                                            "; ".join(breaches)))

    # the sanity gate
    gate = run_sanity_gate(dev_root, dev, report_split)
    for key, result in gate.items():
        cell = report_split.cells.get(key)
        if cell is not None and not result.passed:
            exclusions.append(Exclusion(REPORT, *key, "sanity_check", result.failed, _sanity_text(result)))

    # the alpha, from dev
    blocked: dict[tuple[float, str], list[str]] = defaultdict(list)
    for (arm, alpha, lang), result in gate.items():
        if arm == CHOICE_ARM and not result.passed:
            blocked[(alpha, lang)].append(f"sanity gate {_sanity_text(result)}")
    for (split_name, (arm, alpha, lang)), breaches in judge_blocked.items():
        if split_name == DEV and arm == CHOICE_ARM:
            blocked[(alpha, lang)].append("judge errors over the allowance: " + "; ".join(breaches))
    dev_cells = [c for c in dev.cells.values() if c.arm in (CHOICE_ARM, *REFERENCE_ARMS)]  # A0 / A1: over-refusal references
    dev_rows = aggregate.build_table(
        [v for c in dev_cells if (DEV, c.key) not in judge_blocked for v in c.verdicts or []],
        [s for c in dev_cells for s in c.scores or []],
    ) if dev_cells else []
    dev_langs = lang_order({c.lang for c in dev.cells.values() if c.arm in ("A1", CHOICE_ARM)})
    choice = choose_alpha(dev_rows, blocked, langs=dev_langs)

    # the table: the report split only, minus the excluded safety verdicts
    left_out = {e.name: e for e in exclusions if e.split == REPORT}
    verdicts: list[Verdict] = []
    scores: list[TaskScore] = []
    for cell in report_split.cells.values():
        if f"{REPORT} {cell.name}" not in left_out:
            verdicts.extend(cell.verdicts or [])
        scores.extend(cell.scores or [])
    try:
        rows = aggregate.build_table(verdicts, scores)
    except ValueError as exc:
        raise ReportError(f"the report split cannot be tabulated: {exc}") from None

    # the second judge
    second_id, second = load_second_judge(root, report_split.judge_id) if with_second_judge else (None, [])
    try:
        agr = reliability.agreement(verdicts, second) if second else {}
        pairs = reliability.usable_pair_counts(verdicts, second) if second else {}
    except ValueError as exc:
        raise ReportError(f"the two judges cannot be compared: {exc}") from None
    table_langs = lang_order({r["lang"] for r in rows})
    low = low_confidence_reasons(agr, pairs, table_langs, has_second_judge=second_id is not None)
    if second_id is None:
        warnings.append("no second judge was run, so every language is low-confidence (run scripts/run_reliability.py)")
    for row in rows:
        row["low_confidence"] = row["lang"] in low

    # what the figures compare
    heat = None
    if choice.chosen is not None:
        before, after = ("A1", None), (CHOICE_ARM, choice.chosen)
    else:
        before, after = ("A0", None), ("A1", None)
        why = "; ".join(f"alpha {c.alpha:g}: {c.reasons[0]}" for c in choice.candidates if c.reasons)
        warnings.append(
            "no alpha is eligible, so none was chosen"
            + (f" ({why})" if why else " (there is no dev run of A2)")
            + "; the heatmap shows the drift from A0 to A1 instead of the repair"
        )
    shown = {(r["arm"], r["alpha"]) for r in rows if r["metric"] == "asr"}
    deltas: dict[str, tuple[float, float, float]] = {}
    if before in shown and after in shown:
        heat = {"metric": "asr", "before": {"arm": before[0], "alpha": before[1]},
                "after": {"arm": after[0], "alpha": after[1]}}
        deltas, delta_warnings = _paired_deltas(verdicts, before, after, rows)
        warnings.extend(delta_warnings)
    else:
        warnings.append(
            f"heatmap.png was not written: there are no attack-success rows for both {before[0]} and {after[0]}"
        )
    heat_excluded = {
        (e.lang, e.arm): "judge errors" if e.reason == "judge_error" else "sanity check"
        for e in exclusions
        if e.split == REPORT and heat is not None and (e.arm, e.alpha) in (before, after)
    }
    # what the alpha curve cannot draw: A2 points and the single-model markers (A0, A1) that were left out
    left_alphas: dict[str, set[float]] = defaultdict(set)
    left_arms: dict[str, set[str]] = defaultdict(set)
    for e in exclusions:
        if e.split != REPORT:
            continue
        if e.arm == CHOICE_ARM and e.alpha is not None:
            left_alphas[e.lang].add(e.alpha)
        elif e.arm in REFERENCE_ARMS:
            left_arms[e.lang].add(e.arm)
    curve_excluded = {lang: [*sorted(left_arms[lang]), *sorted(left_alphas[lang])]
                      for lang in sorted(set(left_alphas) | set(left_arms))}
    not_chosen = {c.alpha: list(c.over_refusal_failures) for c in choice.candidates if c.over_refusal_failures}

    return Analysis(
        root, dev, report_split, second_id, exclusions, gate, choice, rows, verdicts, agr, pairs, low, heat, deltas,
        heat_excluded, curve_excluded, not_chosen, notes, warnings,
    )


# --- writing ------------------------------------------------------------------------------------------------------


def reliability_json(a: Analysis) -> dict[str, Any]:
    """The content of reliability.json: numbers, names and reasons only, never text."""
    langs = lang_order({r["lang"] for r in a.table})
    kappa = {}
    for lang in langs:
        k, lo, hi = a.agreement.get(lang, (math.nan, math.nan, math.nan))
        kappa[lang] = {"kappa": k, "lo": lo, "hi": hi, "n_pairs": a.pairs.get(lang, 0)}
    excluded = sorted(a.exclusions, key=lambda e: (e.reason, e.split != REPORT, _order(e.arm, e.alpha, e.lang)))
    return json_safe({
        "alpha_choice": {
            "arm": CHOICE_ARM,
            "candidates": [
                {"alpha": c.alpha, "eligible": c.eligible, "reasons": list(c.reasons), "mean_asr": c.mean_asr,
                 "mean_over_refusal": c.mean_over_refusal, "mean_task_acc": c.mean_task_acc,
                 "over_refusal_failures": list(c.over_refusal_failures)}
                for c in a.choice.candidates
            ],
            "chosen": a.choice.chosen,
            "over_refusal_margin": OVER_REFUSAL_MARGIN,
            "rule": CHOICE_RULE,
            "split": DEV,
        },
        "excluded_cells": [
            {"split": e.split, "arm": e.arm, "alpha": e.alpha, "lang": e.lang, "reason": e.reason,
             "failed_checks": list(e.failed_checks), "detail": e.detail}
            for e in excluded
        ],
        "heatmap": a.heat,
        "judge_ids": {DEV: _rid(a.dev.judge_id), REPORT: _rid(a.report.judge_id)},
        "kappa": kappa,
        "kappa_threshold": KAPPA_LOW,
        "low_confidence": {lang: a.low_confidence[lang] for lang in langs if lang in a.low_confidence},
        "min_pairs": RELIABILITY_MIN_PAIRS,
        "sanity_gate": {
            "cells": [
                {"arm": arm, "alpha": alpha, "lang": lang, "passed": r.passed, "failed_checks": list(r.failed),
                 "unverified": r.unverified, "measures": r.measures}
                for (arm, alpha, lang), r in a.sanity.items()
            ],
            "limits": {
                "max_empty": SANITY_MAX_EMPTY, "max_capped": SANITY_MAX_CAPPED, "max_looping": SANITY_MAX_LOOPING,
                "min_script_share": SANITY_MIN_SCRIPT_SHARE, "max_wrong_script": SANITY_MAX_WRONG_SCRIPT,
                "task_margin": SANITY_TASK_MARGIN,
            },
        },
        "second_judge_id": _rid(a.second_judge_id),
        "warnings": [*a.notes, *a.warnings],
    })


def clear_outputs(out_dir: Path) -> None:
    """Delete the previous outputs so a failed run cannot leave tables that look current."""
    for name in OUTPUT_FILES:
        path = out_dir / name
        if path.is_dir() and not path.is_symlink():
            raise ReportError(f"{path} is a folder; this run writes a file there. Choose another --out")
        if path.exists() or path.is_symlink():
            path.unlink()


def discard_outputs(out_dir: Path) -> None:
    """Best-effort removal of the four outputs after a failed write: no partial set may stay behind."""
    for name in OUTPUT_FILES:
        try:
            (out_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


def write_outputs(a: Analysis, out_dir: Path) -> list[Path]:
    """Write the four files (each through a temp file and a rename); figures that have nothing to draw are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    path = out_dir / "tables.csv"
    evalrun.atomic_replace(path, lambda tmp: aggregate.write_table_csv(
        a.table, tmp, columns=TABLE_COLUMNS, float_digits=FLOAT_DIGITS))
    written.append(path)

    langs = lang_order({r["lang"] for r in a.table})
    if a.heat is not None:
        path = out_dir / "heatmap.png"
        compared = {(a.heat[side]["arm"], a.heat[side]["alpha"]) for side in ("before", "after")}
        shown_rows = [r for r in a.table if (r["arm"], r["alpha"]) in compared]  # one alpha per arm
        evalrun.atomic_replace(path, lambda tmp: report.heatmap(
            shown_rows, metric="asr", arm_before=a.heat["before"]["arm"], arm_after=a.heat["after"]["arm"],
            low_conf=set(a.low_confidence), out_path=tmp, deltas=a.deltas, excluded=a.heat_excluded,
        ))
        written.append(path)
    try:
        path = out_dir / "alpha_curve.png"
        evalrun.atomic_replace(path, lambda tmp: report.alpha_curve(
            a.table, langs=langs, out_path=tmp, chosen_alpha=a.choice.chosen, excluded=a.curve_excluded,
            not_chosen=a.not_chosen))
        written.append(path)
    except ValueError as exc:
        a.warnings.append(f"alpha_curve.png was not written: {exc}")

    path = out_dir / "reliability.json"
    text = json.dumps(reliability_json(a), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    evalrun.atomic_replace(path, lambda tmp: evalrun.write_text(tmp, text))
    written.append(path)
    return written


# --- arguments and running ----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="make_report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, default=None,
                    help="results root with dev/ and report/ folders (default: results/ under the repository root)")
    ap.add_argument("--out", type=Path, default=None, help="where tables.csv, heatmap.png, alpha_curve.png and "
                                                           "reliability.json go (default: the --verdicts folder)")
    ap.add_argument("--dev", type=Path, default=None,
                    help="results folder that holds the dev split, when it is not the --verdicts folder (a report "
                         "rejudged into a fresh folder has none); default: the --verdicts folder. A dev split judged "
                         "by another judge than the report is warned about and recorded, not refused")
    return ap


def parse_args(argv: Sequence[str] | None, *, root: Path) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.verdicts = Path(root) / "results" if args.verdicts is None else args.verdicts
    args.out = args.verdicts if args.out is None else args.out
    return args


def _say(a: Analysis, written: Sequence[Path]) -> None:
    print(f"report judge: {_rid(a.report.judge_id)}; dev judge: {_rid(a.dev.judge_id)}; "
          f"second judge: {_rid(a.second_judge_id)}")
    print(f"read {len(a.report.cells)} report cell(s) and {len(a.dev.cells)} dev cell(s) from {a.root}")
    for note in a.notes:
        print(f"NOTE: {note}", file=sys.stderr)
    for e in sorted(a.exclusions, key=lambda e: (e.split != REPORT, e.reason, _order(e.arm, e.alpha, e.lang))):
        print(f"WARNING: excluded {e.name} ({e.reason}): {e.detail}", file=sys.stderr)
    for warning in a.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    low = [lang for lang in lang_order(set(a.low_confidence))]
    print(f"low-confidence languages: {', '.join(low) if low else 'none'}")
    if a.choice.chosen is not None:
        print(f"alpha chosen on dev ({CHOICE_ARM}): {a.choice.chosen:g}")
    else:
        print("alpha chosen on dev: none")
    for path in written:
        print(f"wrote {path}")


def run(args: argparse.Namespace) -> Analysis:
    clear_outputs(Path(args.out))
    analysis = analyze(args.verdicts, dev_root=args.dev)
    try:
        written = write_outputs(analysis, Path(args.out))
    except BaseException:
        discard_outputs(Path(args.out))  # a failed figure must not leave a fresh tables.csv without its reliability.json
        raise
    _say(analysis, written)
    return analysis


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        args = parse_args(argv, root=root)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        run(args)
    except JudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_JUDGE
    except NoDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_DATA
    except (ReportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    return EXIT_OK
