"""The public export, as a dry run into a folder (Task 19, steps 1-5).

    python scripts/export_public.py --out export/
    python scripts/export_public.py --results results/ --testset data/testset/ --sources data/sources.yaml --out export/

It builds, under --out and nowhere else, everything that is allowed to be published, and nothing is sent anywhere:
this module has no way to publish. What is published, where, and under which account is a separate decision that the
owner takes after looking at this folder. The folder is built whole in a hidden folder next to --out and renamed into
place, so a refused or failed run leaves no --out at all. A non-empty --out is refused (there is no --force).

Layout (the manifest lists every file with its role)

    MANIFEST.json                                  path -> sha256, rows, bytes, role, optional; no clock, no absolute path
    dataset/README.md                              the dataset card (templates/dataset_card.md)
    dataset/prompts.jsonl                          the English prompts: id, lang, kind, split, source, license, text
    dataset/prompts_translated.jsonl               every machine-translated prompt, each row `license: CC-BY-NC-4.0` and
                                                   `derived_from` the translator; the whole file is optional: delete it
                                                   to drop every translated prompt
    dataset/results_public/<split>/<arm>/<alpha|none>/<lang>.verdicts.jsonl   `msd.redact.public_row` rows, nothing else
    dataset/results_public/<split>/<arm>/<alpha|none>/<lang>.tasks.jsonl      item id, lang, arm, alpha, correct
    dataset/aggregates/{tables.csv,reliability.json,heatmap.png,alpha_curve.png}   copied byte for byte from make_report
    models/<arm>/README.md                         a model card for A1, A2 and A3 (A3 when it ran); no weights

What is read, and what never is
    Results come through `msd.reportrun.load_split`, the same checked reader the report uses (`_cache`, `_replaced`
    and `reliability` are never read as cells). Completions files (`*.gens.jsonl`) are never opened: a completion
    cannot reach the export because nothing here reads one. A verdict's excerpt is cut to 200 characters (see
    `msd.redact`). The test set must be intact and frozen (`msd.testset.verify_freeze`) and come with its build record,
    which names the translator of each language. A prompt whose source is evaluation-only (`redistribute: false`),
    excluded, or missing from the manifest is refused by name (`msd.testset.check_usable`).

The safety scan
    Before anything is written, every file about to be exported (the manifest too) is scanned: no home path (macOS,
    Linux or Windows), none of the input or output folders' own paths, no string shaped like a token or key, no value of
    a secret-looking environment variable, and no U+FFFD or lone surrogate. A file is scanned as it is, again without the
    characters a reader does not see (zero-width space, joiners, soft hyphen, ...) and without U+FFFD and lone surrogates,
    and a JSON file once more with its escapes decoded: a secret split by any of them is still a secret. A hit refuses the
    whole export and names the file and the kind of thing, never the thing itself.

Free text from data
    Verdict rows, ids, judge ids and every string a card prints from data (source names and licences, translator names)
    pass through `msd.redact.public_text`: cleaned, then redacted. Prompts are the exception: a frozen item is what was
    evaluated, so it is neither cleaned nor redacted. A prompt with U+FFFD or a lone surrogate is refused by file, row and
    field, and a prompt with a secret or a path in it is refused by the scan.

Cards
    Templates are plain markdown with `{{name}}` placeholders. A placeholder without a value, a blank value, or a
    malformed brace is an error, never an empty string.

Exit codes: 0 done; 1 a data problem (a stale or unreadable file, a failed safety scan, a bad template); 2 usage error
(including an --out that is not empty or sits inside an input); 3 no report split to export; 4 a judge problem (a test
judge, mixed judges); 5 a source that may not be redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from msd import evalrun, gen, redact, reportrun
from msd.atomicdir import atomic_output_dir, check_out_dir, note_stale_partials
from msd.config import ARMS, LANGS, MODEL_IDS
from msd.schema import Prompt, read_jsonl
from msd.sources import load_sources
from msd.testset import MANIFEST_NAME, ForbiddenSourceError, check_usable, file_sha256, verify_freeze

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_DATA, EXIT_JUDGE, EXIT_LICENSE = 0, 1, 2, 3, 4, 5

FORMAT = 1
DATASET, MODELS = "dataset", "models"
SPLITS_EXPORTED = ("dev", "report")
CARD_ARMS = ("A1", "A2", "A3")  # a model card for each of these that has results
TRANSLATED_LICENSE = "CC-BY-NC-4.0"  # the licence of the NLLB model; marked on every translated row, whoever translated it
NLLB = "NLLB"
AGGREGATES = reportrun.OUTPUT_FILES  # tables.csv, heatmap.png, alpha_curve.png, reliability.json
REQUIRED_AGGREGATES = ("tables.csv", "reliability.json")  # a figure with nothing to draw may be missing: a warning
# Free text in reliability.json could carry completion text, so a string longer than this is refused. The longest value
# make_report writes is the alpha-choice rule (about 440 characters, constant text); a verdict excerpt is at most 200.
MAX_JSON_STRING = 600
PIECE, PIECE_STEP = 32, 8  # a piece of a verdict excerpt of 39 characters or more (PIECE + PIECE_STEP - 1) is found outside results_public/

PROMPTS_FILE, TRANSLATED_FILE = f"{DATASET}/prompts.jsonl", f"{DATASET}/prompts_translated.jsonl"
RESULTS_ROOT, AGGREGATES_ROOT = f"{DATASET}/results_public", f"{DATASET}/aggregates"

MODEL_CARD_FIELDS = frozenset({"arm", "arm_title", "base_model", "how_made", "languages", "repair_line", "report_judge",
                               "second_judge_sentence"})
DATASET_CARD_FIELDS = frozenset({
    "arms", "dev_judge", "freeze_sha256", "languages", "n_excluded_sources", "n_prompts_en", "n_prompts_translated",
    "n_task_rows", "n_verdict_rows", "prompt_counts_table", "report_judge", "second_judge_sentence", "sources_table",
    "tables_csv_sha256", "translators_table",
})


class ExportError(RuntimeError):
    """A data problem: nothing was written."""


class UsageError(ExportError):
    """A problem with the command line or --out."""


class NoDataError(ExportError):
    """There is no report split to export."""


class JudgeExportError(ExportError):
    """A verdict set whose judge is a test judge, missing or mixed."""


class LicenseError(ExportError):
    """A prompt whose source may not be redistributed."""


class TemplateError(ExportError):
    """A card template with a missing value or a malformed brace."""


# --- templates ---------------------------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")


def render_template(template: str, values: Mapping[str, str | None]) -> str:
    """`template` with every `{{name}}` replaced by `values[name]`, in one pass (a value is never expanded again).

    A placeholder with no value, or a blank one, is a TemplateError naming it; so is a brace pair that is not a
    placeholder. Values nobody asks for are ignored, so a template may drop a line."""
    leftover = _PLACEHOLDER.sub("", template)
    if "{{" in leftover or "}}" in leftover:
        raise TemplateError("malformed placeholder: a '{{' or '}}' that is not a {{name}} of lower-case letters, digits and _")
    missing = sorted({name for name in _PLACEHOLDER.findall(template)
                      if not isinstance(values.get(name), str) or not values[name].strip()})
    if missing:
        raise TemplateError(f"no value for placeholder(s) {', '.join(missing)}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _render_file(path: Path, values: Mapping[str, str | None]) -> str:
    try:
        template = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ExportError(f"template {path.name} cannot be read ({type(exc).__name__}); expected it at {path}") from None
    try:
        return render_template(template, values)
    except TemplateError as exc:
        raise TemplateError(f"{path.name}: {exc}") from None


# --- the test set and its prompts -------------------------------------------------------------------------------------


def _json_file(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportError(f"{what} {path.name} cannot be read as JSON ({type(exc).__name__})") from None


def _translators(summary_path: Path, freeze_sha: str) -> dict[str, str]:
    """Language -> the translator the build record names; refuses a record that is missing, stale or has no translator."""
    if not summary_path.is_file():
        raise ExportError(
            f"the build record {summary_path.name} is missing (expected at {summary_path}): it names the translator of "
            "each language, and every translated prompt is marked with it. scripts/build_testset.py writes it; or "
            "name it with --build-summary"
        )
    record = _json_file(summary_path, "the build record")
    freeze = record.get("freeze") if isinstance(record, dict) else None
    recorded = freeze.get("sha256") if isinstance(freeze, dict) else None
    if not isinstance(recorded, str) or not recorded:
        raise ExportError(
            f"the build record {summary_path.name} holds no freeze hash, so it cannot be checked against the test set being "
            "exported (its translators would be taken on trust); rebuild and freeze the test set with scripts/build_testset.py"
        )
    if recorded != freeze_sha:
        raise ExportError(f"the build record {summary_path.name} was written for another freeze of the test set; it is stale")
    languages = record.get("languages") if isinstance(record, dict) else None
    found: dict[str, str] = {}
    for lang in LANGS:
        if lang == "en":
            continue
        recorded = languages.get(lang) if isinstance(languages, dict) else None
        name = recorded.get("translator") if isinstance(recorded, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ExportError(f"the build record {summary_path.name} names no translator for {lang}; rebuild that language")
        if name.strip() == "identity":
            raise ExportError(
                f"the build record says {lang} was 'translated' with the identity translator: its texts are copies of "
                "the English ones, not translations, and must not be published"
            )
        found[lang] = redact.public_text(name.strip())  # it is printed in a card and on every translated row
    return found


def _derived_from(translator: str) -> str:
    return NLLB if translator.lower().startswith("nllb") else translator


def _check_sources(rows: Sequence[Prompt], manifest: Sequence[Mapping[str, Any]], manifest_name: str) -> None:
    """Every source that fed a prompt must be in the manifest, usable, and redistributable; a mismatching licence is stale."""
    by_name = {e["name"]: e for e in manifest}
    for name in sorted({r.source for r in rows}):
        if name not in by_name:
            raise LicenseError(
                f"prompt source {name!r} is not in the source manifest {manifest_name}; its licence cannot be checked, "
                "so its prompts are not exported"
            )
        try:
            check_usable(by_name[name])
        except ForbiddenSourceError as exc:
            raise LicenseError(f"{exc}; refusing to export its prompts") from None
    for row in rows:
        if row.license != by_name[row.source]["license"]:
            raise ExportError(
                f"prompt {row.id[:12]} says license {row.license!r} for source {row.source!r}, the manifest now says "
                f"{by_name[row.source]['license']!r}: the test set and {manifest_name} disagree; fix the manifest"
            )


_PROMPT_FIELDS = ("id", "lang", "kind", "split", "source", "license", "text")  # the fields of a Prompt that are published


def _read_prompts(testset_dir: Path) -> list[Prompt]:
    rows: list[Prompt] = []
    for lang in LANGS:
        for split in SPLITS_EXPORTED:
            path = testset_dir / lang / f"{split}.jsonl"
            try:
                part = read_jsonl(path, Prompt)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ExportError(f"{lang}/{split}.jsonl cannot be read as prompts ({type(exc).__name__})") from None
            for number, row in enumerate(part, 1):
                for field in _PROMPT_FIELDS:
                    value = getattr(row, field)
                    if redact.clean_text(value) != value:
                        raise ExportError(
                            f"{lang}/{split}.jsonl row {number}: the '{field}' holds a replacement character (U+FFFD) or a "
                            "lone surrogate, a decoding fault upstream. A frozen item is what was evaluated, so it is not "
                            "cleaned here: fix it at its source, then rebuild and freeze the test set"
                        )
                if row.lang != lang or row.split != split:
                    raise ExportError(f"{lang}/{split}.jsonl holds a row for {row.lang}/{row.split}; the test set is inconsistent")
            rows.extend(part)
    return rows


def _prompt_rows(rows: Sequence[Prompt], translators: Mapping[str, str]) -> tuple[list[dict], list[dict]]:
    english, translated = [], []
    for p in rows:
        if p.lang == "en":
            english.append({"id": p.id, "lang": p.lang, "kind": p.kind, "split": p.split, "source": p.source,
                            "license": p.license, "text": p.text})
        else:
            translated.append({"id": p.id, "lang": p.lang, "kind": p.kind, "split": p.split, "source": p.source,
                               "source_license": p.license, "license": TRANSLATED_LICENSE,
                               "derived_from": _derived_from(translators[p.lang]), "text": p.text})
    return english, translated


# --- results ---------------------------------------------------------------------------------------------------------


@dataclass
class Results:
    files: dict[str, bytes]
    n_verdict_rows: int
    n_task_rows: int
    arms: set[str]
    judges: dict[str, str | None]  # split -> redacted judge id, None when the split was not exported
    a2_alphas: set[float]  # the alphas of arm A2 in the report split


def _read_results(results: Path, notes: list[str]) -> Results:
    found = Results({}, 0, 0, set(), {}, set())
    for split in SPLITS_EXPORTED:
        try:
            loaded = reportrun.load_split(results, split)
        except reportrun.JudgeError as exc:
            raise JudgeExportError(str(exc)) from None
        except reportrun.ReportError as exc:
            raise ExportError(str(exc)) from None
        if not loaded.cells:
            if split == "report":
                raise NoDataError(f"no report split to export: no cell under {results / 'report'}. Run run_eval.py first")
            notes.append(f"no dev split under {results}: the export holds the report split only")
            found.judges[split] = None
            continue
        found.judges[split] = None if loaded.judge_id is None else redact.public_text(loaded.judge_id)
        for cell in loaded.cells.values():
            found.arms.add(cell.arm)
            if split == "report" and cell.arm == "A2" and cell.alpha is not None:
                found.a2_alphas.add(cell.alpha)
            if cell.verdicts:
                try:
                    rows = [redact.public_row(v) for v in cell.verdicts]
                except redact.FakeJudgeError as exc:
                    raise JudgeExportError(str(exc)) from None
                rel = evalrun.verdicts_path(RESULTS_ROOT, split, cell.arm, cell.alpha, cell.lang).as_posix()
                found.files[rel] = redact.public_jsonl(rows)
                found.n_verdict_rows += len(rows)
            if cell.scores:
                rows = [redact.public_task_row(s) for s in cell.scores]
                rel = evalrun.tasks_path(RESULTS_ROOT, split, cell.arm, cell.alpha, cell.lang).as_posix()
                found.files[rel] = redact.public_jsonl(rows)
                found.n_task_rows += len(rows)
    if found.judges["report"] is None:
        raise NoDataError(f"the report split under {results} holds no judged verdicts, so there is nothing to export. Run the judge stage")
    return found


def _read_aggregates(folder: Path, notes: list[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for name in AGGREGATES:
        path = folder / name
        if path.is_file():
            found[f"{AGGREGATES_ROOT}/{name}"] = path.read_bytes()
        elif name in REQUIRED_AGGREGATES:
            raise ExportError(f"{name} is missing from {folder}: run scripts/make_report.py first (or name its folder with --aggregates)")
        else:
            notes.append(f"{name} is not in {folder}, so it is not in the export (make_report skips a figure that has nothing to draw)")
    return found


# --- cards -----------------------------------------------------------------------------------------------------------

_ARM_CARDS = {
    "A1": (
        "after fine-tuning",
        "Tiny Aya Global fine-tuned with LoRA (AlignTune, TRL backend) on the harmless task below. This is the "
        "\"after\" model of the study: the one whose safety is compared with the original's.",
    ),
    "A2": (
        "after the ReSta repair step",
        "The fine-tuned model (arm A1) with a weight-space step applied and no retraining: SafeTune's ReSta moves its "
        "weights toward the aligned model (Tiny Aya Global) and away from the pre-alignment base (Tiny Aya Base), scaled "
        "by a strength called alpha. No gradient step is taken. Whether it helps is what the study measures.",
    ),
    "A3": (
        "fine-tuned with Harden",
        "Tiny Aya Global fine-tuned on the same harmless task data as arm A1, but with SafeTune's Harden, which is meant "
        "to keep safety during the fine-tune rather than restore it afterwards.",
    ),
}


def json_strings(value: Any) -> list[str]:
    """Every string and every key in a decoded JSON value."""
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found += json_strings(item)
    elif isinstance(value, list):
        for item in value:
            found += json_strings(item)
    return found


def _reliability(raw: bytes) -> dict[str, Any]:
    """reliability.json decoded and checked for free text: a long string could hold completion text, so it is refused
    (the message names the top-level key, never the text)."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ExportError("reliability.json cannot be read as JSON; it was not written by make_report.py") from None
    if not isinstance(data, dict):
        raise ExportError("reliability.json is not a JSON object; it was not written by make_report.py")
    for key, value in data.items():
        longest = max((len(text) for text in json_strings(value)), default=0)
        if longest > MAX_JSON_STRING:
            raise ExportError(
                f"reliability.json: the value under '{key}' holds a string of {longest} characters, longer than the "
                f"{MAX_JSON_STRING} this file ever needs; free text that long could hold a completion, so nothing is exported"
            )
    return data


def _chosen_alpha(reliability: Mapping[str, Any]) -> float | None:
    try:
        chosen = reliability["alpha_choice"]["chosen"]
    except (KeyError, TypeError):
        raise ExportError("reliability.json has no alpha_choice.chosen; it was not written by make_report.py") from None
    if chosen is not None and (isinstance(chosen, bool) or not isinstance(chosen, (int, float))):
        raise ExportError("reliability.json: alpha_choice.chosen is not a number or null")
    return chosen


def _check_tables_match_results(reliability: Mapping[str, Any], results: Results, chosen: float | None) -> dict[str, Any]:
    """The tables were made from these results: the judges reliability.json names are the judges of the verdicts being
    exported, and the alpha it chose is one of the A2 alphas being exported. Returns what was checked, for the manifest."""
    recorded = reliability.get("judge_ids")
    if not isinstance(recorded, dict):
        raise ExportError("reliability.json has no judge_ids; it was not written by make_report.py")
    for split in SPLITS_EXPORTED:
        exported = results.judges.get(split)
        if exported is None:
            continue  # this split is not exported, so there is nothing to compare
        named = recorded.get(split)
        if not isinstance(named, str) or redact.public_text(named) != exported:
            raise ExportError(
                f"reliability.json names a different judge for the {split} split than the {split} verdicts being exported: "
                "the tables are stale or belong to other results. Run scripts/make_report.py again, or name the folder that "
                "holds the right tables with --aggregates"
            )
    if chosen is not None and chosen not in results.a2_alphas:
        raise ExportError(
            f"reliability.json chose alpha {chosen:g}, which is not among the A2 alphas of the report split being exported "
            f"({', '.join(f'{a:g}' for a in sorted(results.a2_alphas)) or 'none'}): the tables are stale or belong to other "
            "results. Run scripts/make_report.py again"
        )
    return {
        "reliability_judge_ids": {split: results.judges.get(split) for split in SPLITS_EXPORTED},
        "alpha_choice_chosen": chosen,
        "exported_a2_alphas": sorted(results.a2_alphas),
    }


def _second_judge_sentence(reliability: Mapping[str, Any]) -> str:
    second = reliability.get("second_judge_id")
    if isinstance(second, str) and second.strip():
        return f"A second judge (`{_cell(second)}`) re-labelled a sample."
    return "No second judge was run on this data, so how far the labels can be trusted was not assessed."


def _repair_line(arm: str, chosen: float | None) -> str:
    sides = f"ReSta's base side is Tiny Aya Base (`{MODEL_IDS['BASE']}`) and its aligned side is Tiny Aya Global (`{MODEL_IDS['A0']}`)."
    if arm == "A2" and chosen is not None:
        return (f"Repair: ReSta at alpha {chosen:g}, chosen on the dev split (never the report split) by the rule recorded "
                f"in `reliability.json` in the dataset. {sides}")
    if arm == "A2":
        return ("Repair: ReSta. No alpha passed the dev checks (the sanity gate and the over-refusal bound), so no alpha is "
                f"named; `reliability.json` in the dataset lists every candidate and why it failed. {sides}")
    if arm == "A3":
        return "No repair step was applied: safety was protected during the fine-tune instead."
    return "No repair step was applied to this model."


def _cell(text: Any) -> str:
    """A table cell: cleaned and redacted like every string from data (`redact.public_text`), one line, the pipe escaped."""
    return " ".join(redact.public_text(str(text)).split()).replace("|", "\\|")


def _sources_table(manifest: Sequence[Mapping[str, Any]], english: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(r["source"] for r in english)
    lines = ["| Source | Licence | Used for | Redistribute | English prompts here |", "|---|---|---|---|---|"]
    for e in manifest:
        if e["used_for"] == "excluded":
            continue
        shared = "no (evaluation only)" if e.get("redistribute") is False else "yes"
        lines.append(f"| [{_cell(e['name'])}]({_cell(e['url'])}) | {_cell(e['license'])} | {_cell(e['used_for'])} | {shared} | "
                     f"{counts[e['name']]} |")
    return "\n".join(lines)


def _counts_table(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter((r["lang"], r["split"], r["kind"]) for r in rows)
    lines = ["| Language | dev harmful | dev benign_edgy | report harmful | report benign_edgy |", "|---|---|---|---|---|"]
    for lang in LANGS:
        cells = [counts[(lang, split, kind)] for split in SPLITS_EXPORTED for kind in ("harmful", "benign_edgy")]
        lines.append(f"| {lang} | " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _translators_table(translators: Mapping[str, str]) -> str:
    lines = ["| Language | Translated with | `derived_from` |", "|---|---|---|"]
    lines += [f"| {lang} | {_cell(translators[lang])} | {_derived_from(translators[lang])} |" for lang in LANGS if lang in translators]
    return "\n".join(lines)


# --- the safety scan ------------------------------------------------------------------------------------------------

_HOME_PATTERNS = tuple(re.compile(p.encode("ascii")) for p in (redact.HOME_PATH, redact.WINDOWS_PATH))
_SECRET_PATTERNS = tuple((kind, re.compile(p.encode("ascii"))) for kind, p in redact.SECRET_SHAPES)
_REPLACEMENT_BYTES = bytes([0xEF, 0xBF, 0xBD])  # U+FFFD in UTF-8
_BINARY = (".png",)
_HIDDEN = " (hidden by characters a reader does not see)"
_FOLDED = " (under Unicode compatibility folding)"
_ESCAPED = " (written as JSON escapes)"


def _folder_strings(folders: Sequence[Path]) -> list[bytes]:
    found: set[bytes] = set()
    for folder in folders:
        for form in (Path(folder).absolute(), Path(folder).resolve()):
            if len(form.parts) >= 3:  # never a bare root such as "/" or "/tmp"
                found.add(str(form).encode("utf-8"))
    return sorted(found)


def _matches(data: bytes, own: Sequence[bytes]) -> list[tuple[int, str]]:
    """(offset, what) of every path, folder path and secret shape in `data`."""
    hits = [(m.start(), "an absolute path (a home folder)") for pattern in _HOME_PATTERNS for m in pattern.finditer(data)]
    for folder in own:
        start = data.find(folder)
        if start >= 0:
            hits.append((start, "an absolute path (the path of an input or output folder of this run)"))
    hits += [(m.start(), f"a secret-shaped string ({kind})") for kind, pattern in _SECRET_PATTERNS for m in pattern.finditer(data)]
    return hits


def _line(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def _json_text(rel: str, text: str) -> str | None:
    """Every string and key in a .json file (or in each line of a .jsonl file), one per line, as the JSON reader decodes
    them (so `\\u200b` is the zero-width space it stands for); None for any other file or text that is not JSON."""
    if rel.endswith(".json"):
        chunks = [text]
    elif rel.endswith(".jsonl"):
        chunks = [line for line in text.split("\n") if line.strip()]
    else:
        return None
    found: list[str] = []
    for chunk in chunks:
        try:
            found += json_strings(json.loads(chunk))
        except (ValueError, RecursionError):
            return None
    return "\n".join(found)


def _looked_through(text: str) -> list[tuple[str, str]]:
    """`text` without what a reader does not see (and U+FFFD and lone surrogates), and that under NFKC."""
    hidden = redact.hidden_view(redact.clean_text(text))
    views = [(_HIDDEN, hidden)] if hidden != text else []
    folded = redact.hidden_view(unicodedata.normalize("NFKC", hidden)) if not hidden.isascii() else hidden
    if folded != hidden:
        views.append((_FOLDED, folded))
    return views


def _views(rel: str, text: str) -> list[tuple[str, str]]:
    """Every way a reader can come to see `text`: as it is, without the characters a reader does not see or the study
    removes, under NFKC (fullwidth letters), and, for JSON, with the escapes decoded."""
    views = [("", text), *_looked_through(text)]
    decoded = _json_text(rel, text)
    if decoded is not None:
        views.append((_ESCAPED, decoded))
        views += [(_ESCAPED + note, view) for note, view in _looked_through(decoded)]
    return views


def find_leaks(files: Mapping[str, bytes], folders: Sequence[Path] = ()) -> list[str]:
    """One message per problem in `files` (path -> bytes): a home path, the path of one of `folders`, a secret-shaped
    string, the value of a secret environment variable, U+FFFD or a lone surrogate. A text file is scanned as it is, again
    without its invisible characters (`msd.invisible`: zero-width space, joiners, soft hyphen, combining grapheme joiner,
    variation selectors, ..., and combining marks on ASCII letters) and without U+FFFD and lone surrogates, again under
    NFKC (a fullwidth token), and a JSON file once more with its escapes decoded, so a secret split by any of them is
    found too. Not decoded: percent-encoding and backslash-u written out in a string (a known limit). A message names
    the file (and line) and the kind of thing; it never repeats the thing."""
    own = _folder_strings(folders)
    secrets = gen.secret_env_values()
    problems: list[str] = []
    for rel in sorted(files):
        data = files[rel]
        views = [("", data)]
        texts: list[str] = []
        if not rel.endswith(_BINARY):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                problems.append(f"{rel}: is not valid UTF-8")
                continue
            if _REPLACEMENT_BYTES in data:
                problems.append(f"{rel}:{_line(data, data.index(_REPLACEMENT_BYTES))}: contains the replacement "
                                "character U+FFFD (a decoding fault upstream)")
            seen = _views(rel, text)
            texts = [view for _note, view in seen]
            views = [(note, view.encode("utf-8", "surrogatepass")) for note, view in seen]
            decoded = [view for note, view in seen if note == _ESCAPED]
            if decoded and redact.clean_text(decoded[0]) != decoded[0]:
                problems.append(f"{rel}: contains a replacement character (U+FFFD) or a lone surrogate{_ESCAPED}")
        found: set[str] = set()
        for note, blob in views:
            for start, what in sorted(_matches(blob, own)):
                if note and what in found:
                    continue  # already reported for the file as it is
                found.add(what)
                where = f":{_line(blob, start)}" if _ESCAPED not in note else ""
                problems.append(f"{rel}{where}: contains {what}{note}")
        if texts and not any("secret-shaped" in what for what in found):
            if any(gen.redact_with(view, secrets) != view for view in texts):
                problems.append(f"{rel}: contains the value of a secret environment variable")
    return problems


def _excerpts_outside_results(files: Mapping[str, bytes]) -> list[str]:
    """A piece of a published verdict excerpt (39 characters or more) in a file that is not a results file: free text in an
    aggregate or a card that repeats a model's reply. The prompt files are left out: a model often repeats the question it
    was asked, and prompts are published whole. Names the file, never the text."""
    excerpts = {row["excerpt"] for rel, data in files.items()
                if rel.startswith(f"{RESULTS_ROOT}/") and rel.endswith(".verdicts.jsonl")
                for row in (json.loads(line) for line in data.decode("utf-8").split("\n") if line)}
    pieces = {e[i:i + PIECE] for e in excerpts for i in range(0, len(e) - PIECE + 1, PIECE_STEP)}
    problems: list[str] = []
    if not pieces:
        return problems
    for rel in sorted(files):
        if rel.startswith(f"{RESULTS_ROOT}/") or rel in (PROMPTS_FILE, TRANSLATED_FILE) or rel.endswith(_BINARY):
            continue
        text = files[rel].decode("utf-8", "replace")
        decoded = _json_text(rel, text)
        for body in (text, decoded) if decoded is not None else (text,):
            if any(body[i:i + PIECE] in pieces for i in range(len(body) - PIECE + 1)):
                problems.append(f"{rel}: contains {PIECE + PIECE_STEP - 1} or more characters of a verdict excerpt (a model's "
                                "reply); only the results files may hold those")
                break
    return problems


# --- the manifest ---------------------------------------------------------------------------------------------------


def _rows(rel: str, data: bytes) -> int | None:
    lines = sum(1 for line in data.split(b"\n") if line.strip())
    if rel.endswith(".jsonl"):
        return lines
    if rel.endswith(".csv"):
        return max(0, lines - 1)  # without the header
    return None


def _role(rel: str) -> str:
    if rel == PROMPTS_FILE:
        return "prompts"
    if rel == TRANSLATED_FILE:
        return "prompts_translated"
    if rel.startswith(f"{RESULTS_ROOT}/"):
        return "results_verdicts" if rel.endswith(".verdicts.jsonl") else "results_tasks"
    if rel.startswith(f"{AGGREGATES_ROOT}/"):
        return "aggregate"
    return "dataset_card" if rel.startswith(f"{DATASET}/") else "model_card"


def manifest_bytes(files: Mapping[str, bytes], freeze_sha: str, checks: Mapping[str, Any] | None = None) -> bytes:
    """MANIFEST.json for `files`: sorted, indented, no clock and no absolute path, so two runs on one input match.
    `checks` records what tied the tables to the results (judges, chosen alpha, the A2 alphas exported)."""
    listing = {
        rel: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "n": _rows(rel, data),
              "role": _role(rel), "optional": rel == TRANSLATED_FILE}
        for rel, data in sorted(files.items())
    }
    body = {"format": FORMAT, "testset_freeze_sha256": freeze_sha, "checks": dict(checks or {}), "files": listing}
    return (json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# --- the whole export -----------------------------------------------------------------------------------------------


@dataclass
class Inputs:
    results: Path
    testset: Path
    sources: Path
    aggregates: Path
    templates: Path
    build_summary: Path
    out: Path
    root: Path


@dataclass
class Built:
    files: dict[str, bytes]  # every file but MANIFEST.json
    manifest: bytes
    notes: list[str]


def build(inputs: Inputs) -> Built:
    """Everything the export holds, in memory, checked and scanned; nothing is written."""
    notes: list[str] = []
    try:
        manifest = load_sources(inputs.sources)
    except (OSError, ValueError) as exc:
        raise ExportError(f"the source manifest {inputs.sources.name} cannot be used: {exc}") from None

    problems = verify_freeze(inputs.testset)
    if problems:
        raise ExportError(
            f"the test set under {inputs.testset} is not an intact frozen set, so it is not exported: "
            + "; ".join(problems[:5]) + ("" if len(problems) <= 5 else f"; and {len(problems) - 5} more")
        )
    freeze_sha = file_sha256(inputs.testset / MANIFEST_NAME)
    translators = _translators(inputs.build_summary, freeze_sha)

    prompts = _read_prompts(inputs.testset)
    _check_sources(prompts, manifest, inputs.sources.name)
    english, translated = _prompt_rows(prompts, translators)

    results = _read_results(inputs.results, notes)
    aggregates = _read_aggregates(inputs.aggregates, notes)
    tables_sha = hashlib.sha256(aggregates[f"{AGGREGATES_ROOT}/tables.csv"]).hexdigest()
    reliability = _reliability(aggregates[f"{AGGREGATES_ROOT}/reliability.json"])
    chosen = _chosen_alpha(reliability)
    checks = _check_tables_match_results(reliability, results, chosen)
    second_sentence = _second_judge_sentence(reliability)

    languages = ", ".join(LANGS)
    arms = [arm for arm in ARMS if arm in results.arms]
    report_judge = results.judges["report"]
    files: dict[str, bytes] = {PROMPTS_FILE: redact.public_jsonl(english), TRANSLATED_FILE: redact.public_jsonl(translated)}
    files.update(results.files)
    files.update(aggregates)
    files[f"{DATASET}/README.md"] = _render_file(inputs.templates / "dataset_card.md", {
        "arms": ", ".join(arms),
        "dev_judge": results.judges["dev"] or "(no dev split in this export)",
        "freeze_sha256": freeze_sha,
        "languages": languages,
        "n_excluded_sources": str(sum(e["used_for"] == "excluded" for e in manifest)),
        "n_prompts_en": str(len(english)),
        "n_prompts_translated": str(len(translated)),
        "n_task_rows": str(results.n_task_rows),
        "n_verdict_rows": str(results.n_verdict_rows),
        "prompt_counts_table": _counts_table([*english, *translated]),
        "report_judge": report_judge,
        "second_judge_sentence": second_sentence,
        "sources_table": _sources_table(manifest, english),
        "tables_csv_sha256": tables_sha,
        "translators_table": _translators_table(translators),
    }).encode("utf-8")
    for arm in CARD_ARMS:
        if arm in results.arms:
            title, how_made = _ARM_CARDS[arm]
            files[f"{MODELS}/{arm}/README.md"] = _render_file(inputs.templates / "model_card.md", {
                "arm": arm, "arm_title": title, "base_model": MODEL_IDS["A0"], "how_made": how_made,
                "languages": languages, "repair_line": _repair_line(arm, chosen), "report_judge": report_judge,
                "second_judge_sentence": second_sentence,
            }).encode("utf-8")

    leaks = _excerpts_outside_results(files)
    manifest_data = manifest_bytes(files, freeze_sha, checks)
    leaks += find_leaks({**files, MANIFEST_NAME: manifest_data}, [inputs.results, inputs.testset, inputs.sources,
                                                                  inputs.aggregates, inputs.templates, inputs.out, inputs.root])
    if leaks:
        shown = "\n  ".join(leaks[:8])
        more = f"\n  ... and {len(leaks) - 8} more" if len(leaks) > 8 else ""
        raise ExportError(f"refusing to export, the safety scan found {len(leaks)} problem(s):\n  {shown}{more}")
    return Built(files, manifest_data, notes)


def write(built: Built, out: Path) -> None:
    """Write the export whole, or not at all: files into a hidden folder beside `out`, renamed into place at the end."""
    with atomic_output_dir(out) as folder:
        for rel, data in {**built.files, MANIFEST_NAME: built.manifest}.items():
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


# --- arguments and running ------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="export_public", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="the folder to build the export in; must not exist or be empty "
                                                           "(default: export/ under the repository root)")
    ap.add_argument("--results", type=Path, default=None, help="results root with dev/ and report/ (default: results/)")
    ap.add_argument("--testset", type=Path, default=None, help="the frozen test set (default: data/testset/)")
    ap.add_argument("--sources", type=Path, default=None, help="the source manifest (default: data/sources.yaml)")
    ap.add_argument("--aggregates", type=Path, default=None,
                    help="the folder with tables.csv, reliability.json, heatmap.png, alpha_curve.png (default: --results)")
    ap.add_argument("--templates", type=Path, default=None, help="the card templates (default: templates/)")
    ap.add_argument("--build-summary", type=Path, default=None,
                    help="the test set's build record (default: <testset>_build_summary.json beside it)")
    return ap


def parse_inputs(argv: Sequence[str] | None, *, root: Path) -> Inputs:
    args = build_parser().parse_args(argv)
    root = Path(root)
    testset = args.testset if args.testset is not None else root / "data" / "testset"
    results = args.results if args.results is not None else root / "results"
    return Inputs(
        results=results,
        testset=testset,
        sources=args.sources if args.sources is not None else root / "data" / "sources.yaml",
        aggregates=args.aggregates if args.aggregates is not None else results,
        templates=args.templates if args.templates is not None else root / "templates",
        build_summary=(args.build_summary if args.build_summary is not None
                       else testset.with_name(testset.name + "_build_summary.json")),
        out=args.out if args.out is not None else root / "export",
        root=root,
    )


def _check_out(inputs: Inputs) -> None:
    try:
        check_out_dir(inputs.out, sources=[("results", inputs.results), ("test set", inputs.testset),
                                           ("aggregates", inputs.aggregates), ("templates", inputs.templates)])
    except (FileExistsError, ValueError) as exc:
        raise UsageError(str(exc)) from None


def _say(built: Built, out: Path) -> None:
    files = {**built.files, MANIFEST_NAME: built.manifest}
    print(f"dry run: {len(files)} files written under {out}; nothing was sent anywhere")
    for rel in sorted(files):
        n = _rows(rel, files[rel])
        print(f"  {rel}  ({'' if n is None else f'{n} rows, '}{len(files[rel])} bytes)")
    for note in built.notes:
        print(f"NOTE: {note}", file=sys.stderr)


def run(inputs: Inputs) -> Built:
    _check_out(inputs)
    note_stale_partials(inputs.out)
    built = build(inputs)
    write(built, inputs.out)
    _say(built, inputs.out)
    return built


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        inputs = parse_inputs(argv, root=root)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        run(inputs)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except NoDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_DATA
    except JudgeExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_JUDGE
    except LicenseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_LICENSE
    except (ExportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    return EXIT_OK
