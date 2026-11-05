"""Evaluation orchestrator: generate, judge and score one (arm, alpha) across languages.

    python scripts/run_eval.py --arm A1 --split dev --model PATH --judge NAME --out results/
    python scripts/run_eval.py --arm A2 --alpha 0.25 --split report --model PATH --judge NAME --out results/
    python scripts/run_eval.py --arm A1 --split report --judge NAME --stages judge,task   # elsewhere, no model

For each language it works on `<testset-dir>/<lang>/<split>.jsonl` (harmful and benign_edgy prompts) plus
the closed-form task items of that language and split, and produces, next to each other:

    <out>/<split>/<arm>/<alpha|none>/<lang>.gens.jsonl       completions: safety prompts AND task items
                                     <lang>.gens.meta.json   how they were made (msd.genrun)
                                     <lang>.verdicts.jsonl   one Verdict per harmful / benign_edgy completion
                                     <lang>.tasks.jsonl      one TaskScore per task item (no judge involved)
                                     <lang>.eval.json        counts and quality gates, no text, no secrets
    <out>/<split>/_cache/<judge>.jsonl                       the judge's label cache, shared by arms and languages
    <out>/<split>/_replaced/<old judge>/<arm>/<alpha>/...    verdicts moved aside by --rejudge (never deleted)
    <out>/<split>/reliability/<judge>/<arm>/<alpha>/<lang>.verdicts.jsonl   a second judge's labels (Task 17)

Every path is built by a helper in this module or by `msd.gen.gens_path`. Tooling that walks the tree for
results must skip `_cache`, `_replaced` and `reliability`.

Stages (`--stages generate,judge,task`, default all three; they always run in that order)
    generate  needs --model (or --backend fake). Runs `msd.genrun.run`: meta first, resume, the fake-backend
              guard, all as for scripts/generate.py. The task items are generated here too, into the same
              gens file, because this is the stage that needs the GPU.
    judge     works from the existing gens file only (no model), calls the judge on harmful / benign_edgy
              completions and writes the verdicts file.
    task      works from the existing gens file only, scores task items with `msd.scoring.score_task`.
    A judge or task run with no gens file is an error that says to run the generate stage first. A task item
    is never sent to the judge and a safety item is never scored as a task.

Task items (model-evaluated, closed form): the report split uses `<ft-dir>/heldout.jsonl`, the dev split uses
`<ft-dir>/taskdev.jsonl`. A record is `{id, lang, messages: [user, assistant], gold}`; the user turn becomes a
`Prompt(kind="task", gold=...)`. A missing task file (taskdev.jsonl is built by a later task) or a language with
no rows in it is skipped loudly ("task: skipped (no <file>)", `task_skipped` in the summary and in eval.json),
never silently and never as an error. `--limit` applies to task items as well.

One judge per table: a verdicts file that holds another judge's labels (or two judges') raises `JudgeMixError`
before anything is generated or judged, unless `--rejudge` is given; then the old file is MOVED to
`_replaced/` and the new one written fresh. The verdicts file is derived from the gens file plus the judge
cache and replaced atomically (temp file, fsync, rename) at the end of each language, rows in gens order; a
killed run loses at most the judge calls since the last cached label.

Refusals (all before the first judge call): a gens file whose meta says backend "fake", or whose text carries
the fake marker, on the real results root; a gens file with rows but no meta; ids that are not in the current
prompt list of this split (a stale file from another test-set build); a torn or incomplete gens file. Row
counts come from the gens file, never from the meta's n_rows. A `fake` judge, like the fake backend, is refused
when --out is, or is inside, the real results/ directory.

Quality gates, per language, in `<lang>.eval.json` (and printed): generations by finish_reason; truncation
rate (share of finish_reason "length" among ALL generations) with a WARNING above `TRUNCATION_WARN` (5%);
empty completions, counted separately for task and safety items; verdict label counts; judge_error rate and
counts per kind. A judge_error rate above `JUDGE_ERROR_MAX` (10%) fails the run (exit 5) AFTER the verdicts
file is written, so the errors can be inspected. The rate is judge_error / (verdicts that were put to the judge),
so empty completions (no_response, never judged) cannot hide a broken judge. A single kind above the limit is
a warning.

Exit codes: 0 done; 1 any other failure (a refused, corrupt or incomplete file, a backend or model error, a
failed write); 2 usage error (bad arguments, an unknown judge, a fake backend or judge aimed at results/);
3 none of the requested languages has a test-set file; 4 JudgeMixError; 5 judge_error rate above the limit.
Never 0 on an error. Prompt texts and completions are never printed or written to eval.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from msd import gen, genrun
from msd.config import ARMS, LANGS
from msd.gen import gens_path
from msd.judge import JUDGE_TEMPLATES, Judge, JudgeRunner
from msd.schema import SPLITS, Generation, Prompt, TaskScore, Verdict, read_jsonl, write_jsonl
from msd.scoring import score_task

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_DATA, EXIT_JUDGE_MIX, EXIT_QUALITY = 0, 1, 2, 3, 4, 5

STAGES: tuple[str, ...] = ("generate", "judge", "task")

JUDGE_ERROR_MAX = 0.10  # a language whose judge_error rate is above this fails the run
TRUNCATION_WARN = 0.05  # a truncation rate above this is warned about

# The fine-tune held-out file that holds the task items of each split. The dev one is built by a later task.
TASK_FILES: dict[str, str] = {"report": "heldout.jsonl", "dev": "taskdev.jsonl"}

JUDGED_KINDS: tuple[str, ...] = tuple(JUDGE_TEMPLATES)  # harmful, benign_edgy
_LABEL_ORDER: tuple[str, ...] = ("complied", "refused", "no_response", "judge_error")
_FINISH_ORDER: tuple[str, ...] = ("stop", "length", "error")
_PROGRESS_EVERY = 50


class EvalError(RuntimeError):
    """A file or setup that the orchestrator refuses to evaluate; nothing was judged or written for it."""


class JudgeMixError(RuntimeError):
    """A verdicts file holds labels from another judge (or from two); one table must come from one judge."""


class QualityGateError(RuntimeError):
    """A language's judge_error rate is above JUDGE_ERROR_MAX. The verdicts were written; `summary` has the rest."""

    def __init__(self, message: str, summary: "RunSummary") -> None:
        super().__init__(message)
        self.summary = summary


class NoDataError(RuntimeError):
    """None of the requested languages has a test-set file for the split."""


class FakeResultsError(RuntimeError):
    """A fake judge was aimed at the real results directory."""


# --- paths ----------------------------------------------------------------------------------------------------

_PLAIN_JUDGE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_MAX_PREFIX = 60


def judge_dirname(judge_id: str) -> str:
    """A file-name-safe, collision-free name for a judge id (cache file, `_replaced/` and `reliability/` folders).

    A lowercase id made of letters, digits, `.`, `_` and `-` (up to 64 characters, not ending in a dot) is used
    as it is. Anything else becomes a sanitised lowercase prefix plus `@` plus 12 hex digits of the sha256 of the
    exact id. `@` never occurs in a plain name, so a hashed name cannot equal a plain one, and the hash keeps
    "A", "a/b" and "a_b" (and ids that differ only by case, which a case-insensitive file system would merge)
    apart.
    """
    if not isinstance(judge_id, str) or not judge_id:
        raise ValueError(f"a judge id must be a non-empty string, got {judge_id!r}")
    if _PLAIN_JUDGE_NAME.fullmatch(judge_id) and not judge_id.endswith("."):
        return judge_id
    prefix = re.sub(r"[^a-z0-9._-]+", "_", judge_id.casefold()).strip("._-")[:_MAX_PREFIX].strip("._-") or "judge"
    return f"{prefix}@{hashlib.sha256(judge_id.encode('utf-8')).hexdigest()[:12]}"


def _relative(out_root: str | Path, split: str, arm: str, alpha: float | None, lang: str, suffix: str) -> Path:
    """`<arm>/<alpha|none>/<lang><suffix>`; validated by `gens_path` (split, language, arm and alpha rules)."""
    gens_file = gens_path(out_root, split, arm, alpha, lang)
    return gens_file.relative_to(Path(out_root) / split).with_name(f"{lang}{suffix}")


def verdicts_path(
    out_root: str | Path, split: str, arm: str, alpha: float | None, lang: str, *, reliability_judge: str | None = None
) -> Path:
    """`<out>/<split>/<arm>/<alpha|none>/<lang>.verdicts.jsonl`, or, for a second judge's reliability labels,
    `<out>/<split>/reliability/<judge>/<arm>/<alpha|none>/<lang>.verdicts.jsonl`."""
    relative = _relative(out_root, split, arm, alpha, lang, ".verdicts.jsonl")
    if reliability_judge is None:
        return Path(out_root) / split / relative
    return Path(out_root) / split / "reliability" / judge_dirname(reliability_judge) / relative


def tasks_path(out_root: str | Path, split: str, arm: str, alpha: float | None, lang: str) -> Path:
    """`<out>/<split>/<arm>/<alpha|none>/<lang>.tasks.jsonl`."""
    return Path(out_root) / split / _relative(out_root, split, arm, alpha, lang, ".tasks.jsonl")


def eval_path(out_root: str | Path, split: str, arm: str, alpha: float | None, lang: str) -> Path:
    """`<out>/<split>/<arm>/<alpha|none>/<lang>.eval.json`."""
    return Path(out_root) / split / _relative(out_root, split, arm, alpha, lang, ".eval.json")


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")


def cache_path(out_root: str | Path, split: str, judge_id: str) -> Path:
    """`<out>/<split>/_cache/<judge>.jsonl`: one label cache per judge, shared by all arms and languages."""
    _check_split(split)
    return Path(out_root) / split / "_cache" / f"{judge_dirname(judge_id)}.jsonl"


def replaced_verdicts_path(
    out_root: str | Path, split: str, old_judge_id: str, arm: str, alpha: float | None, lang: str
) -> Path:
    """Where `--rejudge` moves a verdicts file made by `old_judge_id`: `<out>/<split>/_replaced/<judge>/<arm>/...`."""
    relative = _relative(out_root, split, arm, alpha, lang, ".verdicts.jsonl")
    return Path(out_root) / split / "_replaced" / judge_dirname(old_judge_id) / relative


# --- judges ---------------------------------------------------------------------------------------------------


class FakeJudge:
    """A deterministic stand-in for plumbing tests: canned labels from a hash of the reply, never a real judgement.

    `is_fake` makes the orchestrator refuse it when --out is the real results directory, so its labels can
    never be mistaken for measurements.
    """

    is_fake = True

    def __init__(self, judge_id: str = "fake-judge-v1") -> None:
        self.id = judge_id

    def classify(self, kind: str, prompt: str, response: str) -> str:
        digest = hashlib.sha256(f"{kind}|{response}".encode("utf-8")).digest()
        return "complied" if digest[0] % 2 == 0 else "refused"


# `--judge NAME` is looked up here. A factory is called once per run, only when the judge stage runs. Concrete
# judges are registered by the module that defines them (Task 13); only the test judge is registered here.
JUDGE_REGISTRY: dict[str, Callable[[], Judge]] = {"fake": FakeJudge}


def _make_judge(name: str, out_root: Path, root: Path) -> Judge:
    judge = JUDGE_REGISTRY[name]()
    if not isinstance(getattr(judge, "id", None), str) or not judge.id:
        raise EvalError(f"judge {name!r} has no usable id")
    if getattr(judge, "is_fake", False) and genrun.is_in_results(out_root, root):
        raise FakeResultsError(
            f"judge {name!r} is a test judge whose canned labels must never end up among real results, and --out "
            f"{out_root} is the real results directory. Use a scratch directory."
        )
    return judge


# --- arguments ------------------------------------------------------------------------------------------------


def _stages_arg(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(",")]
    if not text.strip() or any(not part for part in parts):
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of stages from {list(STAGES)}")
    unknown = [part for part in parts if part not in STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages {unknown}; expected a subset of {list(STAGES)}")
    return [stage for stage in STAGES if stage in parts]  # canonical order, duplicates dropped


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--alpha", type=genrun._alpha_arg, default=None,
                    help="blend / repair setting ('none' or omitted: no alpha); required for A2 and A2n, "
                         "forbidden for BASE, A0, A1, A3")
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--model", "--model-path", dest="model_path",
                    help="Hugging Face repo id or local (merged) model directory; needed by the generate stage "
                         "with --backend hf")
    ap.add_argument("--judge", help="judge name from the registry; needed by the judge stage")
    ap.add_argument("--out", "--out-root", dest="out_root", type=Path, default=None,
                    help="default: results/ under the repository root")
    ap.add_argument("--stages", type=_stages_arg, default=list(STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)} (default: all three)")
    ap.add_argument("--langs", type=genrun._langs_arg, default=None,
                    help=f"comma-separated subset of {','.join(LANGS)} (default: all)")
    ap.add_argument("--limit", type=genrun._positive_int, default=None,
                    help="only the first N prompts of each language file and the first N task items (smoke runs)")
    ap.add_argument("--rejudge", action="store_true",
                    help="replace verdicts made by another judge: the old file is moved to _replaced/, not deleted")
    ap.add_argument("--testset-dir", type=Path, default=None, help="default: data/testset under the repository root")
    ap.add_argument("--ft-dir", type=Path, default=None,
                    help="folder with heldout.jsonl (report) and taskdev.jsonl (dev); default: data/ft")
    ap.add_argument("--batch-size", type=genrun._positive_int, default=gen.DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=genrun._positive_int, default=256)
    ap.add_argument("--dtype", choices=genrun.DTYPES, default="float16")
    ap.add_argument("--backend", choices=("hf", "fake"), default="hf")
    ap.add_argument("--force-new-meta", action="store_true",
                    help="generate stage: accept a different setup than the one recorded in an existing meta file")
    return ap


def parse_args(argv: Sequence[str] | None, *, root: Path) -> argparse.Namespace:
    """Parse and validate; any usage error prints the usage and exits with code 2 (SystemExit)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        args.alpha = gen.check_arm_alpha(args.arm, args.alpha)
    except ValueError as exc:
        ap.error(str(exc))
    args.langs = list(LANGS) if args.langs is None else args.langs
    args.out_root = Path(root) / "results" if args.out_root is None else args.out_root
    args.testset_dir = Path(root) / "data" / "testset" if args.testset_dir is None else args.testset_dir
    args.ft_dir = Path(root) / "data" / "ft" if args.ft_dir is None else args.ft_dir
    if "judge" in args.stages:
        if not args.judge:
            ap.error("--judge is required for the judge stage (use --stages to leave it out)")
        if args.judge not in JUDGE_REGISTRY:
            ap.error(f"unknown judge {args.judge!r}; registered judges: {', '.join(sorted(JUDGE_REGISTRY))}")
    elif args.rejudge:
        ap.error("--rejudge only makes sense with the judge stage")
    if "generate" in args.stages:
        if args.backend == "fake" and genrun.is_in_results(args.out_root, root):
            ap.error(
                f"--backend fake is refused for --out {args.out_root}: that is the real results directory, and "
                "canned fake text must never end up among real results. Use a scratch directory."
            )
        if args.backend == "hf" and not args.model_path:
            ap.error("--model is required for the generate stage with --backend hf")
    if args.model_path is None and args.backend == "fake":
        args.model_path = "fake"
    return args


# --- prompts: safety test set and task items ------------------------------------------------------------------


def read_task_records(path: Path) -> list[dict[str, Any]]:
    """The records of a task file (`{id, lang, messages: [user, assistant], gold}` per line), validated.

    Errors name the file and line and never quote text.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path.name} line {lineno}"
            try:
                rec = json.loads(line)
            except ValueError:
                raise ValueError(f"{where}: not valid JSON") from None
            if not isinstance(rec, dict):
                raise ValueError(f"{where}: not a JSON object")
            rid, lang, gold, messages = rec.get("id"), rec.get("lang"), rec.get("gold"), rec.get("messages")
            if not (isinstance(rid, str) and rid):
                raise ValueError(f"{where}: id must be non-empty text")
            if lang not in LANGS:
                raise ValueError(f"{where}: lang must be one of {list(LANGS)}")
            if not (isinstance(gold, str) and gold.strip()):
                raise ValueError(f"{where}: gold must be non-empty text")
            ok = (
                isinstance(messages, list) and len(messages) == 2
                and all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages)
                and [m.get("role") for m in messages] == ["user", "assistant"]
                and messages[0]["content"].strip()
            )
            if not ok:
                raise ValueError(f"{where}: messages must be a user turn then an assistant turn, the user turn not empty")
            if rid in seen:
                raise ValueError(f"{where}: duplicate id {rid!r}")
            seen.add(rid)
            records.append(rec)
    return records


def task_prompts(records: Sequence[dict[str, Any]], lang: str, split: str, limit: int | None, source: str) -> list[Prompt]:
    """The task items of `lang` as `Prompt(kind="task")` (text = the user turn), the first `limit` of them."""
    rows = [rec for rec in records if rec["lang"] == lang]
    if limit is not None:
        rows = rows[:limit]
    return [
        Prompt(rec["id"], lang, "task", rec["messages"][0]["content"], split, source, "n/a (fine-tune held-out item)",
               rec["gold"])
        for rec in rows
    ]


@dataclass
class LangPlan:
    """Everything one language needs: its prompts (safety first, then task items) and its files."""

    lang: str
    testset_file: Path
    safety: list[Prompt]
    tasks: list[Prompt]
    task_skipped: str | None  # why this language has no task items, or None
    gens_file: Path
    meta_file: Path
    verdicts_file: Path
    tasks_file: Path
    eval_file: Path
    task_file_missing: bool = False

    @property
    def prompts(self) -> list[Prompt]:
        return [*self.safety, *self.tasks]


def _discover(args: argparse.Namespace) -> list[tuple[str, Path]]:
    available: list[tuple[str, Path]] = []
    for lang in args.langs:
        path = args.testset_dir / lang / f"{args.split}.jsonl"
        if path.is_file():
            available.append((lang, path))
        else:
            print(f"skip {lang}: {path} not found (language not built yet)", file=sys.stderr)
    if not available:
        raise NoDataError(
            f"no requested language has a {args.split} test-set file under {args.testset_dir}; nothing to evaluate"
        )
    return available


def _make_plans(args: argparse.Namespace, available: list[tuple[str, Path]]) -> tuple[list[LangPlan], str | None]:
    task_file = args.ft_dir / TASK_FILES[args.split]
    records: list[dict[str, Any]] | None = None
    run_skipped: str | None = None
    if task_file.is_file():
        records = read_task_records(task_file)
    else:
        run_skipped = f"no {task_file.name}"
        print(f"task: skipped (no {task_file})")

    plans: list[LangPlan] = []
    for lang, path in available:
        safety = genrun.load_prompts(path, lang, args.split, args.limit)
        not_judged = sorted({p.kind for p in safety} - set(JUDGED_KINDS))
        if not_judged:
            raise EvalError(f"{path}: the safety test set may only hold {list(JUDGED_KINDS)} prompts, found {not_judged}")
        tasks: list[Prompt] = []
        skipped = run_skipped
        if records is not None:
            tasks = task_prompts(records, lang, args.split, args.limit, task_file.name)
            if not tasks:
                skipped = f"no {lang} rows in {task_file.name}"
                print(f"task: skipped for {lang} (no {lang} rows in {task_file})")
        clash = sorted({p.id for p in safety} & {p.id for p in tasks})
        if clash:
            raise EvalError(f"{task_file.name}: {len(clash)} task id(s) collide with safety prompt ids, e.g. {clash[:3]}")
        gens_file = gen.gens_path(args.out_root, args.split, args.arm, args.alpha, lang)
        plans.append(LangPlan(
            lang, path, safety, tasks, skipped, gens_file, gen.meta_path(gens_file),
            verdicts_path(args.out_root, args.split, args.arm, args.alpha, lang),
            tasks_path(args.out_root, args.split, args.arm, args.alpha, lang),
            eval_path(args.out_root, args.split, args.arm, args.alpha, lang),
            task_file_missing=records is None,
        ))
    return plans, run_skipped


# --- reading what exists --------------------------------------------------------------------------------------


def _stale_ids(gens_file: Path, valid_ids: set[str]) -> list[str]:
    """Prompt ids in the gens file that are not among `valid_ids` (lines that are not records are left to the scan)."""
    stale: list[str] = []
    with gens_file.open("rb") as fh:
        for raw in fh:
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            pid = obj.get("prompt_id") if isinstance(obj, dict) else None
            if isinstance(pid, str) and pid not in valid_ids:
                stale.append(pid)
    return stale


def _load_gens(plan: LangPlan, args: argparse.Namespace, root: Path) -> tuple[list[Generation], dict[str, Any]]:
    """The finished, checked completions of one language and the meta that describes them.

    Raises EvalError for a missing, stale, torn, incomplete, meta-less or fake gens file. Rows are counted from
    the gens file; the meta's n_rows is never consulted.
    """
    gens_file = plan.gens_file
    if not gens_file.is_file():
        raise EvalError(
            f"{gens_file} not found: there are no completions to judge or score for {plan.lang}. Run the generate "
            "stage first (on a GPU runtime), then copy the results here"
        )
    valid = {p.id for p in plan.prompts}
    stale = _stale_ids(gens_file, valid)
    if stale:
        hint = ""
        if plan.task_file_missing or plan.task_skipped:
            hint = f" (the task items are not available here: {plan.task_skipped}; the file may hold task completions)"
        raise EvalError(
            f"{gens_file}: {len(stale)} row(s) are for prompt ids that are not in the current {plan.lang}/{args.split} "
            f"prompt list, e.g. {stale[:3]}; this file was made for another test set, --limit or task file{hint}"
        )
    existing = gen.scan_existing(plan.prompts, gens_file, arm=args.arm, alpha=args.alpha)  # ResumeError if corrupt
    if existing.torn_bytes:
        raise EvalError(
            f"{gens_file} ends with a torn line ({existing.torn_bytes} bytes, a crash tail); run the generate stage "
            "again to repair it"
        )
    gens = read_jsonl(gens_file, Generation)

    meta: dict[str, Any] = {}
    if gens:
        loaded = genrun.read_meta(plan.meta_file, strict=False)
        if loaded is None:
            raise EvalError(
                f"{gens_file} holds {len(gens)} row(s) but its meta file {plan.meta_file} is missing or unreadable, so "
                "what made them is unknown; regenerate it, or restore the meta file"
            )
        meta = loaded
        for key, want in (("arm", args.arm), ("alpha", args.alpha), ("split", args.split), ("lang", plan.lang)):
            if meta.get(key) != want:
                raise EvalError(f"{plan.meta_file} describes {key} {meta.get(key)!r}, but its path says {want!r}")
        if genrun.is_in_results(args.out_root, root):
            fake_meta = str(meta.get("backend", "")).strip().casefold() == "fake"
            if fake_meta or any(g.text.startswith(gen.FAKE_MARKER) for g in gens):
                raise EvalError(
                    f"{gens_file} was made by the fake backend and --out {args.out_root} is the real results "
                    "directory; canned fake text must never be judged or scored among real results"
                )

    done = {g.prompt_id for g in gens}
    missing = [p.id for p in plan.prompts if p.id not in done]
    if missing:
        raise EvalError(
            f"{gens_file}: {len(missing)} of {len(plan.prompts)} prompts have no completion yet, e.g. {missing[:3]}; "
            "finish the generate stage first"
        )
    return gens, meta


def _read_verdicts(path: Path) -> list[Verdict]:
    try:
        return read_jsonl(path, Verdict)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise EvalError(
            f"{path} cannot be read as verdicts ({type(exc).__name__}); move it aside or fix it, it was left untouched"
        ) from exc


def _check_mix(plan: LangPlan, judge_id: str, rejudge: bool) -> str | None:
    """Refuse (JudgeMixError) a verdicts file of another judge; return the `_replaced/` label of a file to move aside."""
    if not plan.verdicts_file.is_file():
        return None
    ids = sorted({v.judge_id for v in _read_verdicts(plan.verdicts_file)})
    if not ids or ids == [judge_id]:
        return None
    if not rejudge:
        what = f"judge {ids[0]!r}" if len(ids) == 1 else f"{len(ids)} judges {ids}"
        raise JudgeMixError(
            f"{plan.verdicts_file} holds verdicts from {what}, but this run uses judge {judge_id!r}; one table must "
            "come from one judge. Pass --rejudge to replace them (the old file is moved to _replaced/, not deleted)"
        )
    if len(ids) == 1:
        return ids[0]
    return "mixed-" + hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:8]


# --- writing --------------------------------------------------------------------------------------------------


def _fsync_path(path: Path) -> None:
    with path.open("ab") as fh:
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_replace(path: Path, write_tmp: Callable[[Path], None], before_replace: Callable[[], None] | None = None) -> None:
    """Write `path` through a temp file next to it (written, fsynced, renamed); a failure leaves the old file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_tmp(tmp)
        _fsync_path(tmp)
        if before_replace is not None:
            before_replace()
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass  # best effort; the original error is what matters
        raise


def _write_text(tmp: Path, text: str) -> None:
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    n = 1
    while path.with_name(f"{path.name}.{n}").exists():
        n += 1
    return path.with_name(f"{path.name}.{n}")


def _install_verdicts(plan: LangPlan, args: argparse.Namespace, verdicts: list[Verdict], old_label: str | None) -> None:
    def move_old() -> None:
        if old_label is None:
            return
        dest = _unique(replaced_verdicts_path(args.out_root, args.split, old_label, args.arm, args.alpha, plan.lang))
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(plan.verdicts_file, dest)  # moved, never deleted

    _atomic_replace(plan.verdicts_file, lambda tmp: write_jsonl(tmp, verdicts), before_replace=move_old)


# --- statistics -----------------------------------------------------------------------------------------------


def _percent(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _label_counts(verdicts: Sequence[Verdict]) -> dict[str, int]:
    return {label: sum(v.label == label for v in verdicts) for label in _LABEL_ORDER}


def _judge_error_stats(verdicts: Sequence[Verdict]) -> dict[str, Any]:
    """judge_error over the verdicts that were put to the judge (everything except no_response)."""

    def stats(rows: Sequence[Verdict]) -> dict[str, Any]:
        judged = [v for v in rows if v.label != "no_response"]
        errors = sum(v.label == "judge_error" for v in judged)
        return {"n_judged": len(judged), "judge_error": errors, "rate": _rate(errors, len(judged))}

    overall = stats(verdicts)
    return {**overall, "max_allowed": JUDGE_ERROR_MAX,
            "by_kind": {kind: stats([v for v in verdicts if v.kind == kind]) for kind in JUDGED_KINDS}}


def _eval_report(
    plan: LangPlan,
    args: argparse.Namespace,
    gens: Sequence[Generation],
    meta: dict[str, Any],
    verdicts: Sequence[Verdict] | None,
    scores: Sequence[TaskScore] | None,
    judge_id: str | None,
) -> dict[str, Any]:
    """The `<lang>.eval.json` content: counts and rates only, no prompt or completion text, no clock."""
    kind_of = {p.id: p.kind for p in plan.prompts}
    n_total = len(gens)
    finish = {reason: sum(g.finish_reason == reason for g in gens) for reason in _FINISH_ORDER}
    n_length = finish["length"]
    truncation = {"rate": _rate(n_length, n_total), "n_length": n_length, "n_total": n_total}
    empty = {
        "safety": sum(not g.text.strip() for g in gens if kind_of[g.prompt_id] in JUDGED_KINDS),
        "task": sum(not g.text.strip() for g in gens if kind_of[g.prompt_id] == "task"),
    }
    warnings: list[str] = []
    if truncation["rate"] is not None and truncation["rate"] > TRUNCATION_WARN:
        warnings.append(
            f"truncation rate {_percent(truncation['rate'])} ({n_length} of {n_total} generations stopped at "
            f"max_new_tokens) exceeds {TRUNCATION_WARN * 100:.0f}%; answers may be cut off, consider a larger "
            "--max-new-tokens"
        )

    verdict_section = error_section = None
    if verdicts is not None:
        verdict_section = {
            "n": len(verdicts),
            "labels": _label_counts(verdicts),
            "by_kind": {kind: _label_counts([v for v in verdicts if v.kind == kind]) for kind in JUDGED_KINDS},
        }
        error_section = _judge_error_stats(verdicts)
        if error_section["rate"] is not None and error_section["rate"] > JUDGE_ERROR_MAX:
            warnings.append(
                f"judge_error rate {_percent(error_section['rate'])} ({error_section['judge_error']} of "
                f"{error_section['n_judged']} judged) exceeds the {JUDGE_ERROR_MAX * 100:.0f}% limit"
            )
        else:
            for kind, stat in error_section["by_kind"].items():
                if stat["rate"] is not None and stat["rate"] > JUDGE_ERROR_MAX:
                    warnings.append(
                        f"judge_error rate for {kind} items is {_percent(stat['rate'])} ({stat['judge_error']} of "
                        f"{stat['n_judged']} judged), above {JUDGE_ERROR_MAX * 100:.0f}%"
                    )
    task_section = None
    if scores is not None:
        correct = sum(s.correct for s in scores)
        task_section = {"n": len(scores), "correct": correct, "accuracy": _rate(correct, len(scores))}

    return {
        "arm": args.arm,
        "alpha": args.alpha,
        "lang": plan.lang,
        "split": args.split,
        "model_path": meta.get("model_path"),
        "backend": meta.get("backend"),
        "judge_id": gen._redact(judge_id) if judge_id else None,
        "n_generations": n_total,
        "n_by_kind": {kind: sum(kind_of[g.prompt_id] == kind for g in gens) for kind in (*JUDGED_KINDS, "task")},
        "finish_reasons": finish,
        "truncation": truncation,
        "empty_outputs": empty,
        "verdicts": verdict_section,
        "judge_error": error_section,
        "tasks": task_section,
        "task_skipped": plan.task_skipped,
        "warnings": warnings,
    }


# --- running --------------------------------------------------------------------------------------------------


@dataclass
class RunSummary:
    """What a run did: one eval report per language, and what was skipped or warned about."""

    languages: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_languages: list[str] = field(default_factory=list)
    task_skipped: str | None = None  # why the whole task stage was skipped (no task file), or None
    warnings: list[str] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)  # languages whose judge_error rate is over the limit


def _judge_language(plan: LangPlan, gens: Sequence[Generation], runner: JudgeRunner) -> list[Verdict]:
    by_id = {p.id: p for p in plan.prompts}
    todo = [(by_id[g.prompt_id], g) for g in gens if by_id[g.prompt_id].kind in JUDGED_KINDS]  # never a task item
    verdicts: list[Verdict] = []
    for done, (prompt, generation) in enumerate(todo, start=1):
        verdicts.append(runner.verdict(prompt, generation))
        if done % _PROGRESS_EVERY == 0:
            print(f"  {plan.lang}: judged {done}/{len(todo)}", file=sys.stderr, flush=True)
    return verdicts


def _score_language(plan: LangPlan, gens: Sequence[Generation]) -> list[TaskScore]:
    by_id = {p.id: p for p in plan.tasks}  # only task items; a safety completion is never scored
    return [
        TaskScore(p.id, g.arm, g.alpha, p.lang, bool(score_task(g.text, p.gold or "")))
        for g in gens
        if (p := by_id.get(g.prompt_id)) is not None
    ]


def evaluate(args: argparse.Namespace, *, root: Path) -> RunSummary:
    """Run the selected stages for every requested language; see the module docstring.

    Raises NoDataError (no language has a test set), FakeResultsError, JudgeMixError, EvalError and the
    generation errors of `msd.gen`; QualityGateError after everything was written when a language's judge_error
    rate is above JUDGE_ERROR_MAX.
    """
    root = Path(root)
    stages = args.stages
    judge = _make_judge(args.judge, args.out_root, root) if "judge" in stages else None
    available = _discover(args)
    plans, task_skipped = _make_plans(args, available)
    summary = RunSummary(
        skipped_languages=[lang for lang in args.langs if lang not in {p.lang for p in plans}],
        task_skipped=task_skipped,
    )

    # Refuse everything that can be refused before the first generation and before the first judge call.
    old_labels: dict[str, str | None] = {}
    if judge is not None:
        old_labels = {plan.lang: _check_mix(plan, judge.id, args.rejudge) for plan in plans}
    if "generate" not in stages and ("judge" in stages or "task" in stages):
        for plan in plans:
            _load_gens(plan, args, root)

    if "generate" in stages:
        genrun.run(args, available, root, extra_prompts={plan.lang: plan.tasks for plan in plans})

    runner = JudgeRunner(judge, cache_path(args.out_root, args.split, judge.id)) if judge is not None else None
    for plan in plans:
        gens, meta = _load_gens(plan, args, root)
        verdicts: list[Verdict] | None = None
        scores: list[TaskScore] | None = None
        ran_judge = ran_task = False

        if runner is not None:
            verdicts = _judge_language(plan, gens, runner)
            _install_verdicts(plan, args, verdicts, old_labels[plan.lang])
            ran_judge = True
        elif plan.verdicts_file.is_file():
            verdicts = _read_verdicts(plan.verdicts_file)
        if "task" in stages and plan.tasks:
            scores = _score_language(plan, gens)
            _atomic_replace(plan.tasks_file, lambda tmp, rows=scores: write_jsonl(tmp, rows))
            ran_task = True
        elif plan.tasks and plan.tasks_file.is_file():
            scores = read_jsonl(plan.tasks_file, TaskScore)

        ids = {v.judge_id for v in verdicts or []}
        if len(ids) > 1:
            raise JudgeMixError(f"{plan.verdicts_file} holds verdicts from {len(ids)} judges {sorted(ids)}")
        judge_id = runner.judge.id if runner is not None else (next(iter(ids)) if ids else None)

        report = _eval_report(plan, args, gens, meta, verdicts, scores, judge_id)
        _atomic_replace(plan.eval_file, lambda tmp, text=json.dumps(report, ensure_ascii=False, indent=2) + "\n":
                        _write_text(tmp, text))
        summary.languages[plan.lang] = report
        gate_failed = ran_judge and _gate_failed(report)
        if gate_failed:
            summary.gate_failures.append(plan.lang)
        summary.warnings.extend(f"{plan.lang}: {warning}" for warning in report["warnings"])
        _announce(plan, args, report, ran_judge=ran_judge, ran_task=ran_task, gate_failed=gate_failed)

    if summary.gate_failures:
        raise QualityGateError(
            f"judge_error rate is above {JUDGE_ERROR_MAX * 100:.0f}% for: {', '.join(summary.gate_failures)}; the "
            "verdicts were written for inspection, but these results must not be used",
            summary,
        )
    print(f"evaluation complete: {len(plans)} language(s), stages {','.join(stages)}")
    return summary


def _gate_failed(report: dict[str, Any]) -> bool:
    """True when the language's judge_error rate is above JUDGE_ERROR_MAX (strictly)."""
    rate = (report["judge_error"] or {}).get("rate")
    return rate is not None and rate > JUDGE_ERROR_MAX


def _announce(
    plan: LangPlan, args: argparse.Namespace, report: dict[str, Any], *, ran_judge: bool, ran_task: bool,
    gate_failed: bool,
) -> None:
    """Print counts and paths (never text) and the warnings; a failed gate is said on stderr."""
    lang = plan.lang
    finish = report["finish_reasons"]
    empty = report["empty_outputs"]
    truncation = report["truncation"]
    print(
        f"{lang}: {report['n_generations']} generations (stop {finish['stop']}, length {finish['length']}, "
        f"error {finish['error']}), truncation {_percent(truncation['rate'] or 0.0)}, empty outputs: "
        f"safety {empty['safety']}, task {empty['task']} -> {plan.gens_file}"
    )
    if ran_judge:
        labels = report["verdicts"]["labels"]
        print(
            f"{lang}: judged {report['verdicts']['n']} with {report['judge_id']}: " +
            ", ".join(f"{label} {labels[label]}" for label in _LABEL_ORDER) + f" -> {plan.verdicts_file}"
        )
        errors = report["judge_error"]
        if gate_failed:
            print(
                f"error: {lang}: judge_error rate {_percent(errors['rate'])} ({errors['judge_error']} of "
                f"{errors['n_judged']} judged) is above the {JUDGE_ERROR_MAX * 100:.0f}% limit; verdicts written to "
                f"{plan.verdicts_file} for inspection",
                file=sys.stderr,
            )
    if ran_task:
        tasks = report["tasks"]
        print(f"{lang}: {tasks['n']} task items scored, {tasks['correct']} correct -> {plan.tasks_file}")
    for warning in report["warnings"]:
        print(f"WARNING [{lang} {args.arm}/{gen.format_alpha(args.alpha)} {args.split}]: {warning}", file=sys.stderr)


_PLAIN_ERRORS = (*genrun._PLAIN_ERRORS, EvalError)


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        args = parse_args(argv, root=root)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

    try:
        evaluate(args, root=root)
    except JudgeMixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_JUDGE_MIX
    except QualityGateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_QUALITY
    except NoDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_DATA
    except FakeResultsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except _PLAIN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    return EXIT_OK
