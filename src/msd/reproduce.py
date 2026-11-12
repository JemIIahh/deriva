"""The headline table, rebuilt from the PUBLIC export (Task 20): what `notebooks/reproduce.ipynb` runs.

    export/MANIFEST.json                              the files of the export with their sha256
    export/dataset/prompts.jsonl, prompts_translated.jsonl      the prompts: id, lang, kind, split, ...
    export/dataset/results_public/report/<arm>/<alpha|none>/<lang>.{verdicts,tasks}.jsonl     the public rows
    export/dataset/aggregates/{tables.csv, reliability.json, heatmap.png, alpha_curve.png}     what make_report wrote

`table_from_public(export_dir)` rebuilds `tables.csv` from the public rows with the code the report itself uses
(`aggregate.build_table`, the `status` column, the paired changes of the heatmap, the one CSV writer), so the same rows
give the same bytes. `verify_manifest` says whether the files are the ones that were exported, and `explain_difference`
says in plain words where two tables part.

What is recomputed, and what is recorded
    Recomputed from the public rows, never copied: every rate, its bootstrap interval, n, the excluded count, the
    `status` column ("not_applicable" for a language with no task items), and the paired change per language that the
    heatmap prints. The public verdict rows carry no `kind` (harmful or benign_edgy); it is read back from the prompt
    files by prompt id (a row whose prompt is not there is refused, never guessed).
    Recorded, taken from the exported reliability.json: a decision that rests on data the export does not hold. The
    sanity gate looks at completions, the judge-error allowance sits in each cell's eval.json, the kappa of a language
    needs the second judge's labels, and the alpha was chosen on the dev split. So `excluded_cells` (which report cells
    were left out of the table), `low_confidence` (which languages are marked), `alpha_choice` (the chosen alpha and
    which candidates refused too much) and `heatmap` (which two arms it compares) are read as given. They are checked
    where the public rows can check them (the judge, the alpha, the arms), and a missing or inconsistent one is
    refused with a plain message; they are not recomputed, and the notebook says so.
    Not reproducible from public data: whether a recorded decision was right. The claim is "the published table
    follows from the published rows and these recorded decisions", which is what the hash shows.

Only the report split is read: the table comes from it alone (the dev split feeds the recorded decisions).
This module reads nothing but the export folder it is given, writes only where it is told to, and needs no model
stack, API client or network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from msd import aggregate, gen, redact, reportrun
from msd.config import ARMS, LANGS
from msd.reportrun import DEV, REPORT, TABLE_COLUMNS, Exclusion
from msd.schema import Prompt, TaskScore, Verdict, write_jsonl

# Where the exporter (msd.export) puts things; tests/test_reproduce.py pins these against it.
MANIFEST_NAME = "MANIFEST.json"
RESULTS_ROOT = "dataset/results_public"
AGGREGATES_ROOT = "dataset/aggregates"
PROMPT_FILES = ("dataset/prompts.jsonl", "dataset/prompts_translated.jsonl")
TABLES_FILE = f"{AGGREGATES_ROOT}/tables.csv"
RELIABILITY_FILE = f"{AGGREGATES_ROOT}/reliability.json"

# The decisions the exported reliability.json must record (see the module docstring).
RECORDED_KEYS = ("alpha_choice", "excluded_cells", "heatmap", "judge_ids", "low_confidence")
DIFFERENCE_LINES = 10  # `explain_difference` names this many differences, then says how many more


class ReproduceError(RuntimeError):
    """The export cannot be used as it is (a file is missing, changed or inconsistent); the message says which."""


# --- hashes and the manifest ---------------------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(export_dir: Path) -> dict[str, Any]:
    path = export_dir / MANIFEST_NAME
    if not path.is_file():
        raise ReproduceError(
            f"{MANIFEST_NAME} is not in {export_dir}: it lists every file of the export with its sha256. "
            "Point EXPORT_DIR at the export folder (the one export_public.py wrote, or a complete download of it)"
        )
    try:
        files = json.loads(path.read_text(encoding="utf-8")).get("files")
    except (OSError, ValueError, AttributeError):
        raise ReproduceError(f"{MANIFEST_NAME} cannot be read as the manifest of an export (valid JSON with a 'files' list)") from None
    if not isinstance(files, dict):
        raise ReproduceError(f"{MANIFEST_NAME} has no 'files' list, so it is not the manifest of an export")
    return files


def verify_manifest(export_dir: str | Path) -> list[str]:
    """One message per file that is missing or whose sha256 is not the one `MANIFEST.json` records; [] when every
    listed file is as exported. A manifest entry that points outside the export folder is refused."""
    root = Path(export_dir)
    problems = []
    for rel, entry in sorted(_manifest_files(root).items()):
        parts = Path(rel).parts
        if Path(rel).is_absolute() or ".." in parts:
            raise ReproduceError(f"{MANIFEST_NAME} lists {rel!r}, which is outside the export folder; it is not a manifest this export wrote")
        target = root / rel
        if not target.is_file():
            problems.append(f"{rel}: missing (the manifest lists it)")
            continue
        recorded = entry.get("sha256") if isinstance(entry, dict) else None
        found = sha256_file(target)
        if found != recorded:
            problems.append(f"{rel}: changed since the export was built (its sha256 is {found[:12]}..., the manifest says {str(recorded)[:12]}...)")
    return problems


def manifest_tables_sha256(export_dir: str | Path) -> str:
    """The sha256 that `MANIFEST.json` records for the exported tables.csv: what the recomputed table must hash to."""
    entry = _manifest_files(Path(export_dir)).get(TABLES_FILE)
    recorded = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(recorded, str) or not recorded:
        raise ReproduceError(f"{MANIFEST_NAME} records no sha256 for {TABLES_FILE}; it is not the manifest of this export")
    return recorded


# --- reading ---------------------------------------------------------------------------------------------------------


def _jsonl(path: Path, rel: str) -> list[tuple[int, Any]]:
    """(line number, decoded value) of every non-blank line; the split is on "\\n" alone, as the exporter writes it."""
    try:
        lines = path.read_bytes().decode("utf-8").split("\n")
    except (OSError, ValueError):
        raise ReproduceError(f"{rel} cannot be read as UTF-8 text") from None
    found = []
    for number, line in enumerate(lines, start=1):
        if line.strip():
            try:
                found.append((number, json.loads(line)))
            except ValueError:
                raise ReproduceError(f"{rel} line {number} is not valid JSON") from None
    return found


@dataclass(frozen=True)
class Recorded:
    """What reliability.json records (see the module docstring), checked for shape."""

    exclusions: tuple[Exclusion, ...]
    low_confidence: dict[str, list[str]]
    chosen_alpha: float | None
    heat: dict[str, Any] | None
    not_chosen: dict[float, list[str]]
    report_judge: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_recorded(export_dir: str | Path) -> Recorded:
    """The recorded decisions of the exported reliability.json, or a ReproduceError that says what is wrong with it."""
    root = Path(export_dir)
    path = root / RELIABILITY_FILE
    if not path.is_file():
        raise ReproduceError(
            f"reliability.json is missing from {AGGREGATES_ROOT}/ of {root}: it records the decisions that cannot be "
            "recomputed from the public rows (which cells were left out of the table, which languages are low-confidence, "
            "which alpha was chosen). Point EXPORT_DIR at the complete export folder"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ReproduceError("reliability.json is not valid JSON, so the recorded decisions cannot be read") from None
    if not isinstance(data, dict):
        raise ReproduceError("reliability.json is not a JSON object, so the recorded decisions cannot be read")
    for key in RECORDED_KEYS:
        if key not in data:
            raise ReproduceError(f"reliability.json has no '{key}': it is not the file make_report.py wrote, or it was cut, so a recorded decision is missing")
    try:
        choice, cells, low, judges, heat = (data[k] for k in ("alpha_choice", "excluded_cells", "low_confidence", "judge_ids", "heatmap"))
        chosen = choice["chosen"]
        if chosen is not None and not _is_number(chosen):
            raise TypeError
        exclusions = tuple(
            Exclusion(c["split"], c["arm"], c["alpha"], c["lang"], c["reason"], tuple(c["failed_checks"]), c["detail"]) for c in cells
        )
        if not all(e.split in ("dev", REPORT) and isinstance(e.arm, str) and isinstance(e.lang, str) and
                   (e.alpha is None or _is_number(e.alpha)) for e in exclusions):
            raise TypeError
        if not (isinstance(low, dict) and all(isinstance(v, list) for v in low.values())):
            raise TypeError
        report_judge = judges["report"]
        if not isinstance(report_judge, str):
            raise TypeError
        not_chosen = {c["alpha"]: list(c["over_refusal_failures"]) for c in choice["candidates"] if c["over_refusal_failures"]}
        if heat is not None:
            heat = {"metric": heat["metric"], **{side: {"arm": heat[side]["arm"], "alpha": heat[side]["alpha"]} for side in ("before", "after")}}
    except (KeyError, TypeError, AttributeError):
        raise ReproduceError(
            "reliability.json has a recorded decision in a shape this reader does not know (alpha_choice, excluded_cells, "
            "low_confidence, judge_ids or heatmap): it is not the file make_report.py wrote"
        ) from None
    return Recorded(exclusions, {k: list(v) for k, v in low.items()}, chosen, heat, not_chosen, report_judge)


def _prompt_kinds(root: Path) -> dict[tuple[str, str, str], str]:
    """(split, language, prompt id) -> kind, from the exported prompt files."""
    kinds: dict[tuple[str, str, str], str] = {}
    if not (root / PROMPT_FILES[0]).is_file():
        raise ReproduceError(f"{PROMPT_FILES[0]} is missing: the public verdict rows carry no kind, and it is read from the prompt files")
    for rel in PROMPT_FILES:
        if not (root / rel).is_file():
            continue  # the translated prompts are an optional file; a verdict that needs it is refused below, by name
        for number, row in _jsonl(root / rel, rel):
            if not (isinstance(row, dict) and all(isinstance(row.get(k), str) for k in ("id", "lang", "kind", "split"))):
                raise ReproduceError(f"{rel} line {number} is not a prompt row (id, lang, kind, split)")
            key = (row["split"], row["lang"], row["id"])
            if kinds.setdefault(key, row["kind"]) != row["kind"]:
                raise ReproduceError(f"{rel} line {number}: prompt {row['id'][:16]} ({row['lang']}, {row['split']}) has two kinds in the prompt files")
    return kinds


def _cell_files(base: Path, suffix: str) -> list[tuple[Path, str, float | None, str]]:
    """(path, arm, alpha, language) of every public `*.<suffix>.jsonl` file of a split, in path order."""
    found = []
    for path in sorted(base.glob(f"*/*/*.{suffix}.jsonl")):
        arm, alpha_dir, name = path.relative_to(base).parts
        lang = name[: -len(f".{suffix}.jsonl")]
        try:
            alpha = None if alpha_dir == "none" else float(alpha_dir)
        except ValueError:
            alpha = math.nan
        if arm not in ARMS or lang not in LANGS or alpha != alpha:  # NaN: the folder is not an alpha
            raise ReproduceError(
                f"{RESULTS_ROOT}/{REPORT}/{path.relative_to(base).as_posix()} is not where a public {suffix} file of this study "
                "sits (<arm>/<alpha or none>/<language>)"
            )
        found.append((path, arm, alpha, lang))
    return found


def _check_path(rel: str, rows: Sequence, arm: str, alpha: float | None, lang: str) -> None:
    foreign = sum(r.arm != arm or r.lang != lang or r.alpha != alpha for r in rows)
    if foreign:
        raise ReproduceError(f"{rel}: {foreign} of {len(rows)} row(s) are not {arm}/{alpha if alpha is not None else 'none'}/{lang} rows, as the path says; a copied or swapped file would silently change the table")


def _read_report_rows(root: Path, kinds: Mapping[tuple[str, str, str], str]) -> tuple[list[Verdict], list[TaskScore]]:
    base = root / RESULTS_ROOT / REPORT
    if not base.is_dir():
        raise ReproduceError(f"there is no report split in the export: expected {RESULTS_ROOT}/{REPORT}/<arm>/<alpha>/<language>.verdicts.jsonl under {root}")
    verdicts: list[Verdict] = []
    unknown: dict[str, int] = {}  # language -> verdict rows whose prompt is in no prompt file
    for path, arm, alpha, lang in _cell_files(base, "verdicts"):
        rel = path.relative_to(root).as_posix()
        rows = []
        for number, row in _jsonl(path, rel):
            missing = [k for k in redact.PUBLIC_KEYS if not (isinstance(row, dict) and k in row)]
            if missing:
                raise ReproduceError(f"{rel} line {number} is not a public verdict row: it lacks {', '.join(missing)}")
            try:
                kind = kinds.get((REPORT, row["lang"], row["prompt_id"]))
                if kind is None:
                    unknown[row["lang"]] = unknown.get(row["lang"], 0) + 1
                    continue
                rows.append(Verdict(row["prompt_id"], row["arm"], row["alpha"], row["lang"], kind, row["judge_id"], row["label"], row["excerpt"]))
            except (TypeError, ValueError) as exc:  # a value of the wrong type (a list as a language), or an unknown label
                raise ReproduceError(f"{rel} line {number} is not a public verdict row ({exc})") from None
        _check_path(rel, rows, arm, alpha, lang)
        verdicts.extend(rows)
    if unknown:
        optional = PROMPT_FILES[1]
        hint = f" ({optional} is not in the export, and it carries the kinds of every language but English)" if not (root / optional).is_file() else ""
        raise ReproduceError(
            f"{sum(unknown.values())} verdict row(s) ({', '.join(sorted(unknown))}) have a prompt id that is in none of the prompt files{hint}. "
            "The public rows carry no kind (harmful or benign_edgy); it is read from the prompts by id, so without it "
            "attack success and over-refusal cannot be told apart"
        )
    scores: list[TaskScore] = []
    for path, arm, alpha, lang in _cell_files(base, "tasks"):
        rel = path.relative_to(root).as_posix()
        rows = []
        for number, row in _jsonl(path, rel):
            missing = [k for k in redact.TASK_KEYS if not (isinstance(row, dict) and k in row)]
            if missing or not isinstance(row["correct"], bool):
                raise ReproduceError(f"{rel} line {number} is not a public task row" + (f": it lacks {', '.join(missing)}" if missing else ": 'correct' is not true or false"))
            rows.append(TaskScore(row["prompt_id"], row["arm"], row["alpha"], row["lang"], row["correct"]))
        _check_path(rel, rows, arm, alpha, lang)
        scores.extend(rows)
    return verdicts, scores


# --- the table -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicTable:
    """The table rebuilt from the public rows, and what the two figures need to be drawn from it."""

    rows: list[dict]  # `tables.csv` rows: aggregate.build_table + low_confidence + status
    deltas: dict[str, tuple[float, float, float]]  # paired change per language, as printed beside the heatmap
    heat: dict[str, Any] | None  # RECORDED: the two arms the heatmap compares
    chosen_alpha: float | None  # RECORDED: chosen on the dev split
    low_confidence: dict[str, list[str]]  # RECORDED: language -> why (the second judge's labels are not public)
    exclusions: tuple[Exclusion, ...]  # RECORDED: the cells left out (sanity gate, judge-error allowance)
    heat_excluded: dict[tuple[str, str], str]
    curve_excluded: dict[str, list[float | str]]
    not_chosen: dict[float, list[str]]  # RECORDED: alphas that refused too much
    task_not_applicable: list[str]
    judge_id: str
    n_verdict_rows: int
    n_task_rows: int
    notes: list[str] = field(default_factory=list)

    def write_tables(self, path: str | Path) -> Path:
        """tables.csv through the report's own writer: the same rows are the same bytes."""
        return reportrun.write_tables_csv(self.rows, Path(path))

    def write_figures(self, out_dir: str | Path) -> list[Path]:
        """heatmap.png (when the report compared two arms) and alpha_curve.png, drawn by the report's own code."""
        written, problem = reportrun.write_figures(
            Path(out_dir), self.rows, heat=self.heat, deltas=self.deltas, low_confidence=set(self.low_confidence),
            heat_excluded=self.heat_excluded, chosen_alpha=self.chosen_alpha, curve_excluded=self.curve_excluded,
            not_chosen=self.not_chosen, task_not_applicable=self.task_not_applicable,
        )
        note = f"alpha_curve.png was not drawn: {problem}"
        if problem is not None and note not in self.notes:  # drawing twice must not say it twice
            self.notes.append(note)
        return written


def table_from_public(export_dir: str | Path) -> PublicTable:
    """Rebuild the report's table from the public rows of `export_dir`; raises `ReproduceError` with a plain message
    when the export cannot be used. See the module docstring for what is recomputed and what is recorded."""
    root = Path(export_dir)
    if not root.is_dir():
        raise ReproduceError(f"{root} is not a folder: EXPORT_DIR must be the export folder (the one export_public.py wrote, or a complete download of it)")
    recorded = read_recorded(root)
    verdicts, scores = _read_report_rows(root, _prompt_kinds(root))

    judges = sorted({v.judge_id for v in verdicts})
    if len(judges) > 1:
        raise ReproduceError(f"the public report rows come from {len(judges)} different judges {judges}; a table never mixes judges")
    if judges and redact.public_text(recorded.report_judge) != judges[0]:
        raise ReproduceError(
            f"reliability.json says the report verdicts were made by judge {recorded.report_judge!r}, but the public rows come from "
            f"{judges[0]!r}: the recorded decisions belong to other results, or a file was swapped"
        )
    a2_alphas = sorted({r.alpha for r in (*verdicts, *scores) if r.arm == reportrun.CHOICE_ARM and r.alpha is not None})
    if recorded.chosen_alpha is not None and recorded.chosen_alpha not in a2_alphas:
        raise ReproduceError(
            f"reliability.json chose alpha {recorded.chosen_alpha:g}, which is not among the A2 alphas of the public rows "
            f"({', '.join(f'{a:g}' for a in a2_alphas) or 'none'}): the recorded decisions belong to other results"
        )

    left_out = {(e.arm, e.alpha, e.lang) for e in recorded.exclusions if e.split == REPORT}
    entering = [v for v in verdicts if (v.arm, v.alpha, v.lang) not in left_out]  # the cells the report left out
    try:
        rows = reportrun.with_status(aggregate.build_table(entering, scores))
    except ValueError as exc:
        raise ReproduceError(f"the public rows cannot be tabulated: {exc}") from None
    for row in rows:
        row["low_confidence"] = row["lang"] in recorded.low_confidence

    deltas: dict[str, tuple[float, float, float]] = {}
    notes: list[str] = []
    compared = None
    if recorded.heat is not None:
        compared = tuple((recorded.heat[side]["arm"], recorded.heat[side]["alpha"]) for side in ("before", "after"))
        shown = {(r["arm"], r["alpha"]) for r in rows if r["metric"] == "asr"}
        for arm, alpha in compared:
            if (arm, alpha) not in shown:
                raise ReproduceError(
                    f"reliability.json says the heatmap compares {arm}" + (f" at alpha {alpha:g}" if alpha is not None else "")
                    + ", but the public rows have no attack-success rate for it"
                )
        deltas, notes = reportrun.paired_deltas(entering, compared[0], compared[1], rows)
    heat_excluded, curve_excluded = reportrun.figure_exclusions(recorded.exclusions, compared)
    return PublicTable(
        rows=rows, deltas=deltas, heat=recorded.heat, chosen_alpha=recorded.chosen_alpha,
        low_confidence=recorded.low_confidence, exclusions=recorded.exclusions, heat_excluded=heat_excluded,
        curve_excluded=curve_excluded, not_chosen=recorded.not_chosen,
        task_not_applicable=reportrun.lang_order({r["lang"] for r in rows if r["status"] == "not_applicable"}),
        judge_id=judges[0] if judges else recorded.report_judge, n_verdict_rows=len(verdicts), n_task_rows=len(scores),
        notes=notes,
    )


# --- comparing with the exported table -------------------------------------------------------------------------------


def _raw_rows(path: str | Path) -> tuple[list[str], dict[tuple[str, str, str, str], dict[str, str]]]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = {(r["arm"], r["alpha"], r["lang"], r["metric"]): r for r in reader}
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        raise ReproduceError(f"{Path(path).name} cannot be read as a results table ({type(exc).__name__})") from None
    return header, rows


def read_table_csv(path: str | Path) -> list[dict]:
    """A `tables.csv` back into rows as `aggregate.build_table` and the report make them (numbers as numbers, an empty
    cell as None, true / false as a bool)."""
    header, raw = _raw_rows(path)
    if header != list(TABLE_COLUMNS):
        raise ReproduceError(f"{Path(path).name} does not have the columns of a results table ({', '.join(TABLE_COLUMNS)})")
    rows = []
    for r in raw.values():
        try:
            rows.append({
                "arm": r["arm"], "alpha": float(r["alpha"]) if r["alpha"] else None, "lang": r["lang"], "metric": r["metric"],
                **{k: float(r[k]) if r[k] else None for k in ("rate", "lo", "hi")},
                "n": int(r["n"]), "excluded": int(r["excluded"]), "low_confidence": r["low_confidence"] == "true",
                "status": r["status"],
            })
        except ValueError:
            raise ReproduceError(f"{Path(path).name} has a number it cannot read in the row for {r['arm']} {r['lang']} {r['metric']}") from None
    return rows


def _label(key: tuple[str, str, str, str]) -> str:
    arm, alpha, lang, metric = key
    return f"arm={arm} alpha={alpha or 'none'} lang={lang} metric={metric}"


def explain_difference(recomputed: str | Path, exported: str | Path, limit: int = DIFFERENCE_LINES) -> list[str]:
    """[] when the two tables are the same bytes; otherwise plain lines saying where they part: a row only one side has, or
    a column with the two values side by side (as written in the files). At most `limit` lines, then how many more."""
    if Path(recomputed).read_bytes() == Path(exported).read_bytes():
        return []
    head_a, ours = _raw_rows(recomputed)
    head_b, theirs = _raw_rows(exported)
    if head_a != head_b:
        return [f"the columns differ: {', '.join(head_a)} here, {', '.join(head_b)} in the export"]
    lines = []
    for key in [*ours, *(k for k in theirs if k not in ours)]:
        if key not in theirs:
            lines.append(f"row {_label(key)} is only in the recomputed table")
        elif key not in ours:
            lines.append(f"row {_label(key)} is only in the export")
        else:
            lines += [f"row {_label(key)}: column {column} is {ours[key][column]!r} here but {theirs[key][column]!r} in the export"
                      for column in head_a if ours[key][column] != theirs[key][column]]
    if not lines:
        return ["the two files hold the same values but differ in formatting (line endings, quoting or number format), so their hashes differ"]
    if len(lines) > limit:
        lines = [*lines[:limit], f"... and {len(lines) - limit} more differences"]
    return lines


def figure_numbers(rows: Sequence[Mapping], heat: Mapping[str, Any] | None, chosen_alpha: float | None = None) -> dict[str, Any]:
    """The numbers the two figures draw, taken from table rows (recomputed, or read back from the exported tables.csv),
    in table order. PNG bytes cannot be compared (matplotlib draws slightly differently across versions and platforms),
    the plotted numbers can.

    heatmap      one (language, arm, alpha, rate, lo, hi, n) for each cell of the two compared arms
    alpha_curve  one (language, arm, alpha, task_acc, asr) for each point: the A2 line points and the A0 / A1 markers that
                 have both numbers (a language with no task items has no x to plot)
    The paired changes printed beside the heatmap are not here: they come from the verdict pairs, and no exported file
    records them (`PublicTable.deltas` has them)."""
    compared = {(heat[side]["arm"], heat[side]["alpha"]) for side in ("before", "after")} if heat is not None else set()
    heatmap = [(r["lang"], r["arm"], r["alpha"], r["rate"], r["lo"], r["hi"], r["n"])
               for r in rows if r["metric"] == "asr" and (r["arm"], r["alpha"]) in compared]
    rates: dict[tuple, dict[str, float]] = {}
    for r in rows:
        if r["metric"] in ("asr", "task_acc") and r["rate"] is not None:
            drawn = (r["arm"] == reportrun.CHOICE_ARM and r["alpha"] is not None) or (r["arm"] in reportrun.REFERENCE_ARMS and r["alpha"] is None)
            if drawn:
                rates.setdefault((r["lang"], r["arm"], r["alpha"]), {})[r["metric"]] = r["rate"]
    curve = [(lang, arm, alpha, m["task_acc"], m["asr"]) for (lang, arm, alpha), m in rates.items() if "asr" in m and "task_acc" in m]
    return {"heatmap": heatmap, "alpha_curve": curve, "chosen_alpha": chosen_alpha}


# --- the optional GPU section: a small fresh run, compared with the published labels ---------------------------------


def rebuild_testset(export_dir: str | Path, out_dir: str | Path, *, lang: str, split: str = REPORT, limit: int | None = None,
                    as_split: str | None = None) -> Path:
    """The exported prompts of one language and split, written in the layout the study's scripts read
    (`<out_dir>/<lang>/<split>.jsonl`, one `Prompt` per line), the first `limit` of them; returns `out_dir`, which is what
    `--testset-dir` takes. A translated prompt keeps its source's licence, as in the test set it was exported from.

    `as_split` writes the same prompts under another split label (`<out_dir>/<lang>/<as_split>.jsonl`, each row's split
    set to it). The GPU spot check uses `as_split="dev"`: run_eval generates the report split only from the frozen test
    set (a MANIFEST.json that verifies, `msd.genrun`), which a copy rebuilt from the export is not, and the comparison
    with the published labels goes by prompt id, so the label of the scratch run does not change what is compared."""
    if as_split is not None and as_split not in (DEV, REPORT):
        raise ValueError(f"as_split must be {DEV!r} or {REPORT!r}, got {as_split!r}")
    root = Path(export_dir)
    found = []
    for rel in PROMPT_FILES:
        if (root / rel).is_file():
            found += [row for _n, row in _jsonl(root / rel, rel) if isinstance(row, dict) and row.get("lang") == lang and row.get("split") == split]
    if not found:
        raise ReproduceError(f"the export has no {split} prompts for language {lang!r} (look at {PROMPT_FILES[0]} and {PROMPT_FILES[1]})")
    label = split if as_split is None else as_split
    try:
        prompts = [Prompt(r["id"], r["lang"], r["kind"], r["text"], label if as_split else r["split"], r["source"],
                          r.get("source_license", r["license"]))
                   for r in found[:limit]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReproduceError(f"a prompt row of {lang}/{split} is not a prompt ({type(exc).__name__})") from None
    write_jsonl(Path(out_dir) / lang / f"{label}.jsonl", prompts)
    return Path(out_dir)


def compare_with_published(export_dir: str | Path, fresh_verdicts: str | Path, *, arm: str, alpha: float | None, lang: str) -> dict[str, int]:
    """How a fresh judged run lines up with the published labels of the same cell: for the prompts both have, how many got
    the same label. The two will differ somewhat (sampling in the generation, the judge's own variation); this counts, it
    does not decide. Keys: compared, same, only_fresh, only_published."""
    root = Path(export_dir)
    published_file = root / RESULTS_ROOT / REPORT / arm / gen.format_alpha(alpha) / f"{lang}.verdicts.jsonl"
    if not published_file.is_file():
        raise ReproduceError(f"the export has no published labels for {arm} {lang} at {published_file.relative_to(root).as_posix()}")
    published = {row["prompt_id"]: row["label"] for _n, row in _jsonl(published_file, published_file.name)}
    try:
        fresh = {v.prompt_id: v.label for v in reportrun.read_verdict_rows(Path(fresh_verdicts))}
    except reportrun.ReportError as exc:
        raise ReproduceError(str(exc)) from None
    both = sorted(set(published) & set(fresh))
    return {"compared": len(both), "same": sum(published[i] == fresh[i] for i in both),
            "only_fresh": len(set(fresh) - set(published)), "only_published": len(set(published) - set(fresh))}
