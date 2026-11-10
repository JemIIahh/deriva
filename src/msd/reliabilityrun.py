"""The judge-reliability run: a second judge re-labels a sample of the report completions, and the two are compared.

    python scripts/run_reliability.py --results results/ --judge NAME
    python scripts/run_reliability.py --results results/ --judge NAME --per-lang 30 --langs hi,ar

It takes the verdicts that go into the report tables (exactly those `scripts/make_report.py` would use: cells left
out for judge errors or a failed sanity check are not sampled), picks about `--per-lang` of them per language
(`msd.reliability.sample_items`: an equal share per arm, a seeded shuffle, so the same data gives the same sample),
shows the second judge the same prompts and completions and writes its labels where make_report looks for them:

    <results>/report/reliability/<judge>/<arm>/<alpha|none>/<lang>.verdicts.jsonl

then prints Cohen's kappa with a bootstrap interval for each language and names the low-confidence languages
(`reportrun.low_confidence_reasons`: the lower bound of kappa below `config.KAPPA_LOW`, fewer than
`config.RELIABILITY_MIN_PAIRS` comparable pairs, or no kappa at all). The second judge's labels share the on-disk
label cache of `msd.judge.JudgeRunner` (`<results>/report/_cache/<judge>.jsonl`), so a rerun asks nothing twice.

Rules
    The second judge must differ from the judge behind the report verdicts, and the report verdicts must come from
    one real judge: a fake or missing judge id, or two judges, is refused (exit 4) before the second judge is asked
    anything. A test judge (`fake`) is refused when --results is the real results directory (exit 2).
    The labels already in the judge's folder must belong to the sample of this run (or a part of it). Labels left
    by another sample (another --per-lang, --seed or --langs for a language, or changed first-judge results)
    would change the kappa without anyone seeing it, so they are refused (exit 1); move the folder away to start
    over. Languages not in this run are left alone.
    Only counts, kappa and language names are printed; prompt and completion text never is. Each label row keeps
    the same 200-character excerpt as every verdicts file. If a judge call raised, the exception TYPES are
    summarised on stderr and, when the judge reports it, its token usage ("judge usage: ...") as run_eval.py does.

Exit codes: 0 done (low-confidence languages are reported, not an error); 1 any other failure (a missing test set
or completions file, labels of another sample, a judge that cannot run, a failed write); 2 usage error (bad
arguments, an unknown judge name, a test judge aimed at results/); 3 no report split to read; 4 a judge problem
(a verdict without a judge id, mixed judges, a test judge among the report verdicts, a second judge that is the
first).
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from msd import evalrun, gen, genrun, judges, reliability, reportrun
from msd.config import LANGS, RELIABILITY_N
from msd.gaterun import looks_fake
from msd.judge import JudgeRunner
from msd.schema import Generation, Prompt, Verdict, read_jsonl, write_jsonl

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_DATA, EXIT_JUDGE = 0, 1, 2, 3, 4

SPLIT = reportrun.REPORT  # the reliability sample is drawn from the report split only
_PROGRESS_EVERY = 50

Key = tuple[str, str, float | None]  # (prompt id, arm, alpha)
CellKey = tuple[str, float | None, str]  # (arm, alpha, language)


# --- arguments ----------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_reliability", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=Path, default=None, help="results root with a report/ folder (default: results/ "
                                                               "under the repository root)")
    ap.add_argument("--judge", required=True, help="the SECOND judge: 'anthropic:<model-id>' (needs "
                                                   "ANTHROPIC_API_KEY), 'hf:<model-id>' (GPU runtime) or a registered "
                                                   "name; it must not be the judge of the report verdicts")
    ap.add_argument("--per-lang", type=genrun.positive_int, default=RELIABILITY_N,
                    help=f"items to double-judge per language (default {RELIABILITY_N})")
    ap.add_argument("--langs", type=genrun.langs_arg, default=None,
                    help=f"comma-separated subset of {','.join(LANGS)} (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sample (default 0); the same data and seed "
                                                        "give the same items")
    ap.add_argument("--testset-dir", type=Path, default=None, help="default: data/testset under the repository root")
    ap.add_argument("--dev", type=Path, default=None,
                    help="results folder that holds the dev split, when it is not --results (as for make_report.py: "
                         "the sample must be the one the tables are made from); default: --results")
    return ap


def parse_args(argv: Sequence[str] | None, *, root: Path) -> argparse.Namespace:
    """Parse and validate; any usage error prints the usage and exits with code 2 (SystemExit)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    args.results = Path(root) / "results" if args.results is None else args.results
    args.testset_dir = Path(root) / "data" / "testset" if args.testset_dir is None else args.testset_dir
    args.langs = list(LANGS) if args.langs is None else args.langs
    if args.judge not in evalrun.JUDGE_REGISTRY and not judges.is_known_judge_name(args.judge):
        ap.error(
            f"unknown judge {args.judge!r}; registered judges: {', '.join(sorted(evalrun.JUDGE_REGISTRY))}; "
            f"other accepted forms: {judges.ACCEPTED_FORMS}"
        )
    return args


# --- the sample and what it needs -----------------------------------------------------------------------------------


def _folder(results: Path, judge_id: str) -> Path:
    return Path(results) / SPLIT / "reliability" / evalrun.judge_dirname(judge_id)


def _by_cell(items: Sequence[Key], lang_of: dict[Key, str]) -> dict[CellKey, list[Key]]:
    cells: dict[CellKey, list[Key]] = defaultdict(list)
    for item in items:
        _, arm, alpha = item
        cells[(arm, alpha, lang_of[item])].append(item)
    return dict(sorted(cells.items(), key=lambda kv: reportrun._order(*kv[0])))


def _load_prompts(testset_dir: Path, langs: Sequence[str]) -> dict[str, dict[str, Prompt]]:
    prompts: dict[str, dict[str, Prompt]] = {}
    for lang in reportrun.lang_order(langs):
        path = Path(testset_dir) / lang / f"{SPLIT}.jsonl"
        if not path.is_file():
            raise reportrun.ReportError(
                f"{path} not found: the {SPLIT} test set of {lang} is needed to show the second judge the prompts"
            )
        prompts[lang] = {p.id: p for p in genrun.load_prompts(path, lang, SPLIT, None)}
    return prompts


def _load_generations(results: Path, cells: dict[CellKey, list[Key]]) -> dict[CellKey, dict[str, Generation]]:
    gens: dict[CellKey, dict[str, Generation]] = {}
    for (arm, alpha, lang), items in cells.items():
        path = gen.gens_path(results, SPLIT, arm, alpha, lang)
        if not path.is_file():
            raise reportrun.ReportError(
                f"{path} not found: the completions (gens file) of {reportrun.cell_name(arm, alpha, lang)} are needed "
                "to show them to the second judge; copy the gens files next to the verdicts"
            )
        try:
            rows = read_jsonl(path, Generation)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise reportrun.ReportError(f"{path} cannot be read as generations ({type(exc).__name__})") from None
        reportrun._check_rows_match_path(path, [reportrun._Tagged(g, lang) for g in rows], arm, alpha, lang)
        by_id = {g.prompt_id: g for g in rows}
        missing = sorted({item[0] for item in items} - set(by_id))
        if missing:
            raise reportrun.ReportError(
                f"{path} has no completion for {len(missing)} sampled item(s), e.g. {missing[:3]}: it is stale or "
                "incomplete"
            )
        gens[(arm, alpha, lang)] = by_id
    return gens


def _check_existing_labels(folder: Path, judge_id: str, wanted: dict[str, set[Key]]) -> None:
    """Refuse labels in `folder` for the languages of this run that are not in this run's sample."""
    for lang, keys in wanted.items():
        for path in sorted(folder.glob(f"*/*/{lang}.verdicts.jsonl")):
            rows = reportrun.read_verdict_rows(path)
            other = sorted({v.judge_id for v in rows} - {judge_id})
            if other:
                raise reportrun.JudgeError(
                    f"{path} holds labels from {[reportrun._rid(i) for i in other]}, not from {reportrun._rid(judge_id)!r}; "
                    "never mix judges in one comparison"
                )
            extra = {(v.prompt_id, v.arm, v.alpha) for v in rows} - keys
            if extra:
                raise reportrun.ReportError(
                    f"{path} holds {len(extra)} label(s) for items that are not in this run's sample: they were left "
                    "by another sample (another --per-lang, --seed or --langs, or changed first-judge results). "
                    f"Mixing samples would change the kappa unseen; move {folder} away to start over"
                )


# --- the run ------------------------------------------------------------------------------------------------------


def _agreement_lines(
    first: Sequence[Verdict], second: Sequence[Verdict], shown: Sequence[str]
) -> tuple[list[str], dict[str, list[str]]]:
    agr = reliability.agreement(first, second)
    pairs = reliability.usable_pair_counts(first, second)
    low = reportrun.low_confidence_reasons(agr, pairs, shown)
    lines = []
    for lang in shown:
        kappa, lo, hi = agr.get(lang, (math.nan, math.nan, math.nan))
        text = (f"kappa {kappa:.2f} [{lo:.2f}, {hi:.2f}]" if not (math.isnan(kappa) or math.isnan(lo))
                else "kappa not assessed")
        flag = f"  LOW CONFIDENCE ({', '.join(low[lang])})" if lang in low else ""
        lines.append(f"{lang:<9} {text}  pairs {pairs.get(lang, 0)}{flag}")
    return lines, low


def run(args: argparse.Namespace, *, root: Path) -> dict[str, list[str]]:
    """Do the run; returns the low-confidence languages with their reasons. Raises the errors listed in `main`."""
    results = Path(args.results)
    # the first judge's checks run here, first; the dev split decides which repair cells are in the tables
    analysis = reportrun.analyze(results, with_second_judge=False, dev_root=args.dev)
    first_id = analysis.report.judge_id
    if first_id is None or not analysis.report_verdicts:
        raise reportrun.ReportError(
            f"{results / SPLIT} holds no judged verdicts that enter the tables, so there is nothing to double-judge"
        )
    for note in [*analysis.notes, *analysis.warnings, *analysis.terminal_notes]:
        print(f"NOTE: {note}", file=sys.stderr)

    judge = evalrun.build_judge(args.judge, results, Path(root))
    if looks_fake(judge.id) and genrun.is_in_results(results, Path(root)):
        raise evalrun.FakeResultsError(
            f"judge {args.judge!r} is a test judge (its id says fake) whose canned labels must never end up among "
            f"real results, and --results {results} is the real results directory. Use a scratch directory."
        )
    if reportrun.same_judge(judge.id, first_id):
        raise reportrun.JudgeError(
            f"judge {reportrun._rid(judge.id)!r} is the same judge (the same provider and model, whatever its case or "
            "template fingerprint) that made the report verdicts; a judge agrees with itself by construction, so it "
            "cannot be its own second opinion. Use another judge"
        )

    in_scope = [v for v in analysis.report_verdicts if v.lang in args.langs]
    items = reliability.sample_items(in_scope, args.per_lang, seed=args.seed)
    if not items:
        raise reportrun.ReportError(
            f"no judged complied/refused verdicts for {', '.join(args.langs)} in {results / SPLIT}: nothing to sample"
        )
    first = {(v.prompt_id, v.arm, v.alpha): v for v in in_scope}
    lang_of = {key: first[key].lang for key in items}
    cells = _by_cell(items, lang_of)
    langs_run = reportrun.lang_order({lang for _, _, lang in cells})

    prompts = _load_prompts(args.testset_dir, langs_run)
    generations = _load_generations(results, cells)
    for (arm, alpha, lang), cell_items in cells.items():
        for key in cell_items:
            prompt = prompts[lang].get(key[0])
            if prompt is None or prompt.kind != first[key].kind:
                raise reportrun.ReportError(
                    f"the {lang} test set does not match the verdicts of {reportrun.cell_name(arm, alpha, lang)} "
                    f"(prompt {key[0]!r} is missing or of another kind): it is a different build of the test set"
                )

    folder = _folder(results, judge.id)
    _check_existing_labels(folder, judge.id, {lang: {k for k in items if lang_of[k] == lang} for lang in langs_run})

    print(f"first judge: {reportrun._rid(first_id)}; second judge: {reportrun._rid(judge.id)}")
    print(f"double-judging {len(items)} item(s) in {len(langs_run)} language(s) ({args.per_lang} per language at most)")
    runner = JudgeRunner(judge, evalrun.cache_path(results, SPLIT, judge.id))
    written = 0
    try:
        done = 0
        for (arm, alpha, lang), cell_items in cells.items():
            rows = []
            for key in cell_items:
                rows.append(runner.verdict(prompts[lang][key[0]], generations[(arm, alpha, lang)][key[0]]))
                done += 1
                if done % _PROGRESS_EVERY == 0:
                    print(f"  judged {done}/{len(items)}", file=sys.stderr, flush=True)
            path = evalrun.verdicts_path(results, SPLIT, arm, alpha, lang, reliability_judge=judge.id)
            evalrun.atomic_replace(path, lambda tmp, rows=rows: write_jsonl(tmp, rows))
            written += 1
    finally:
        evalrun.report_judge_run(runner)
    print(f"wrote {written} file(s) under {folder}")

    test_judge = looks_fake(judge.id)  # only reachable in a scratch directory: the real one was refused above
    if test_judge:
        print("NOTE: the second judge is a test judge, so its labels are canned and the kappa below says nothing; "
              "make_report.py refuses to read them", file=sys.stderr)
    _, second = reportrun.read_second_judge_folder(folder, first_id, allow_test_judge=test_judge)
    shown = reportrun.lang_order(set(langs_run) | {v.lang for v in second})
    lines, low = _agreement_lines(analysis.report_verdicts, second, shown)
    for line in lines:
        print(line)
    for lang in shown:
        stat = evalrun.judge_error_stats([v for v in second if v.lang == lang])
        if stat["rate"] is not None and stat["rate"] > evalrun.JUDGE_ERROR_MAX:
            print(f"WARNING: the second judge failed on {stat['rate'] * 100:.1f}% of the {lang} items "
                  f"({stat['judge_error']} of {stat['n_judged']}); those pairs are not compared", file=sys.stderr)
    print(f"low-confidence languages: {', '.join(l for l in shown if l in low) if low else 'none'}")
    return low


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        args = parse_args(argv, root=root)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        run(args, root=root)
    except reportrun.JudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_JUDGE
    except reportrun.NoDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_DATA
    except evalrun.FakeResultsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (reportrun.ReportError, *evalrun.PLAIN_ERRORS) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    return EXIT_OK
