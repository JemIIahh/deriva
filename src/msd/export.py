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
    Before anything is written, every file about to be exported is scanned: no home path (macOS, Linux or Windows),
    none of the input or output folders' own paths, no string shaped like a token or key, no value of a secret-looking
    environment variable, and no U+FFFD. A hit refuses the whole export and names the file and the kind of thing, never
    the thing itself.

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

PROMPTS_FILE, TRANSLATED_FILE = f"{DATASET}/prompts.jsonl", f"{DATASET}/prompts_translated.jsonl"
RESULTS_ROOT, AGGREGATES_ROOT = f"{DATASET}/results_public", f"{DATASET}/aggregates"

MODEL_CARD_FIELDS = frozenset({"arm", "arm_title", "base_model", "how_made", "languages", "repair_line", "report_judge"})
DATASET_CARD_FIELDS = frozenset({
    "arms", "dev_judge", "freeze_sha256", "languages", "n_excluded_sources", "n_prompts_en", "n_prompts_translated",
    "n_task_rows", "n_verdict_rows", "prompt_counts_table", "report_judge", "sources_table", "tables_csv_sha256",
    "translators_table",
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
    if isinstance(freeze, dict) and freeze.get("sha256") not in (None, freeze_sha):
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
        found[lang] = name.strip()
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


def _read_prompts(testset_dir: Path) -> list[Prompt]:
    rows: list[Prompt] = []
    for lang in LANGS:
        for split in SPLITS_EXPORTED:
            path = testset_dir / lang / f"{split}.jsonl"
            try:
                part = read_jsonl(path, Prompt)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ExportError(f"{lang}/{split}.jsonl cannot be read as prompts ({type(exc).__name__})") from None
            for row in part:
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


def _read_results(results: Path, notes: list[str]) -> Results:
    found = Results({}, 0, 0, set(), {})
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
        found.judges[split] = None if loaded.judge_id is None else redact.redact_text(loaded.judge_id)
        for cell in loaded.cells.values():
            found.arms.add(cell.arm)
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
        "drifted, after fine-tuning",
        "Tiny Aya Global fine-tuned with LoRA (AlignTune, TRL backend) on the harmless task below. This is the "
        "\"after\" model of the study: the one whose safety may have drifted.",
    ),
    "A2": (
        "repaired with ReSta",
        "The drifted model (arm A1) repaired without retraining by SafeTune's ReSta: a weight-space step that moves the "
        "drifted weights back toward the aligned model (Tiny Aya Global) and away from the pre-alignment base (Tiny Aya "
        "Base), scaled by a strength called alpha. No gradient step is taken.",
    ),
    "A3": (
        "safety-preserving fine-tune",
        "Tiny Aya Global fine-tuned on the same harmless task data as arm A1, but with SafeTune's Harden, which tries to "
        "keep safety during the fine-tune instead of repairing it afterwards.",
    ),
}


def _chosen_alpha(reliability: bytes) -> float | None:
    try:
        chosen = json.loads(reliability.decode("utf-8"))["alpha_choice"]["chosen"]
    except (ValueError, KeyError, TypeError):
        raise ExportError("reliability.json has no alpha_choice.chosen; it was not written by make_report.py") from None
    if chosen is not None and (isinstance(chosen, bool) or not isinstance(chosen, (int, float))):
        raise ExportError("reliability.json: alpha_choice.chosen is not a number or null")
    return chosen


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


def _cell(text: str) -> str:
    """A table cell: one line, the pipe escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _sources_table(manifest: Sequence[Mapping[str, Any]], english: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(r["source"] for r in english)
    lines = ["| Source | Licence | Used for | Redistribute | English prompts here |", "|---|---|---|---|---|"]
    for e in manifest:
        if e["used_for"] == "excluded":
            continue
        shared = "no (evaluation only)" if e.get("redistribute") is False else "yes"
        lines.append(f"| [{_cell(e['name'])}]({e['url']}) | {_cell(e['license'])} | {e['used_for']} | {shared} | {counts[e['name']]} |")
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

_HOME_PATH = re.compile(rb"(?<![\w.~-])/(?:Users|home)/[A-Za-z0-9._-]+")
_WINDOWS_PATH = re.compile(rb"(?<![A-Za-z])[A-Za-z]:[\\/]+(?:Users|Documents and Settings)")
_SECRET_SHAPES = (
    ("a Hugging Face token", re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{16,}")),
    ("an API key", re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")),
    ("a GitHub token", re.compile(rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("an AWS key id", re.compile(rb"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}")),
    ("a private key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
_REPLACEMENT_BYTES = bytes([0xEF, 0xBF, 0xBD])  # U+FFFD in UTF-8
_BINARY = (".png",)


def _folder_strings(folders: Sequence[Path]) -> list[bytes]:
    found: set[bytes] = set()
    for folder in folders:
        for form in (Path(folder).absolute(), Path(folder).resolve()):
            if len(form.parts) >= 3:  # never a bare root such as "/" or "/tmp"
                found.add(str(form).encode("utf-8"))
    return sorted(found)


def find_leaks(files: Mapping[str, bytes], folders: Sequence[Path] = ()) -> list[str]:
    """One message per problem in `files` (path -> bytes): a home path, the path of one of `folders`, a secret-shaped
    string, the value of a secret environment variable, U+FFFD. A message names the file (and line) and the kind of
    thing; it never repeats the thing."""
    own = _folder_strings(folders)
    problems: list[str] = []

    def at(rel: str, data: bytes, start: int, what: str) -> None:
        line = data.count(b"\n", 0, start) + 1
        problems.append(f"{rel}:{line}: contains {what}")

    for rel in sorted(files):
        data = files[rel]
        binary = rel.endswith(_BINARY)
        for pattern in (_HOME_PATH, _WINDOWS_PATH):
            for m in pattern.finditer(data):
                at(rel, data, m.start(), "an absolute path (a home folder)")
        for folder in own:
            start = data.find(folder)
            if start >= 0:
                at(rel, data, start, "an absolute path (the path of an input or output folder of this run)")
        for kind, pattern in _SECRET_SHAPES:
            for m in pattern.finditer(data):
                at(rel, data, m.start(), f"a secret-shaped string ({kind})")
        if binary:
            continue
        if _REPLACEMENT_BYTES in data:
            at(rel, data, data.index(_REPLACEMENT_BYTES), "the replacement character U+FFFD (a decoding fault upstream)")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: is not valid UTF-8")
            continue
        if gen._redact(text) != text and not any(p.startswith(f"{rel}:") and "secret-shaped" in p for p in problems):
            problems.append(f"{rel}: contains the value of a secret environment variable")
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


def manifest_bytes(files: Mapping[str, bytes], freeze_sha: str) -> bytes:
    """MANIFEST.json for `files`: sorted, indented, no clock and no absolute path, so two runs on one input match."""
    listing = {
        rel: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "n": _rows(rel, data),
              "role": _role(rel), "optional": rel == TRANSLATED_FILE}
        for rel, data in sorted(files.items())
    }
    body = {"format": FORMAT, "testset_freeze_sha256": freeze_sha, "files": listing}
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
    chosen = _chosen_alpha(aggregates[f"{AGGREGATES_ROOT}/reliability.json"])

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
            }).encode("utf-8")

    manifest_data = manifest_bytes(files, freeze_sha)
    leaks = find_leaks({**files, MANIFEST_NAME: manifest_data}, [inputs.results, inputs.testset, inputs.sources,
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
