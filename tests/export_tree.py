"""A throwaway PUBLIC export for the reproduction tests (Task 20), built by the real code end to end.

    frozen fixture test set  ->  results tree (its prompt ids)  ->  make_report  ->  export_public

so the folder it returns is what `scripts/export_public.py` writes: `MANIFEST.json`, `dataset/prompts*.jsonl`,
`dataset/results_public/...` and `dataset/aggregates/...`. Unlike the Task 19 fixtures, whose verdict ids are made up
(`en-h000`) and so match no prompt, the verdicts here carry the ids of the test set's own prompts, as a real run's do:
that is how the public rows get their `kind` back (the export drops it). Nothing touches a model, a judge, the network
or the real repository folders.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from results_tree import ALPHAS, build_tree, write_second_judge

from msd import config, export, reportrun, testset
from msd.schema import Prompt, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
TRANSLATORS = {"hi": "nllb-600m", "ar": "nllb-600m", "sw": "nllb-600m", "hinglish": "llm-chat"}

# Large enough that every cell has at least `stats.MIN_N` (10) items of each judged kind, so the alpha can be chosen, and that
# a repair moves at least `stats.MIN_DISCORDANT` (10) of the 40 report prompts, so a paired change has an interval to print.
SIZES = {"harmful": {"report": 40, "dev": 20}, "benign_edgy": {"report": 12, "dev": 12}}

# As in the real study (`config.TASK_ACCURACY_LANGS`): Hinglish has no task items, so no cell of it has a tasks file and its
# task-accuracy rows are "not_applicable" in the table.
ALWAYS = {(..., ..., ..., "hinglish"): {"with_tasks": False}}

# name -> what differs from the plain tree. Each is a situation the table must reproduce exactly.
SCENARIOS: dict[str, dict] = {
    "default": {},
    # a judge-error exclusion in the report split, and a dev sanity failure that blocks one alpha and (through
    # the gate, which covers both splits) leaves that report cell out of the table
    "exclusions": {
        "overrides": {
            ("report", "A2", 0.5, "hi"): {"judge_errors": 5},
            ("dev", "A2", 0.25, "sw"): {"style": "empty"},
        },
    },
    # no alpha passes the dev checks: the heatmap shows the drift (A0 to A1) instead of the repair
    "no_alpha": {"overrides": {("dev", "A2", ..., "hi"): {"style": "empty"}}},
    "no_second_judge": {"second_judge": False},
    # a second judge that often disagrees: some languages come out low-confidence, which is a recorded decision
    "noisy_second_judge": {"noise": {"hi": 0.5, "ar": 0.5}},
}


@dataclass
class PublicExport:
    root: Path
    results: Path  # the private results tree (with tables.csv and the other report outputs)
    export: Path  # the public export, as export_public.py wrote it


def _write_sources(path: Path) -> Path:
    from conftest import BENIGN, HARM_A, HARM_B, entry  # at call time: conftest imports this module

    entries = [entry(HARM_A, "harmful_prompts", "mit"), entry(HARM_B, "harmful_prompts", "cc-by-4.0"),
               entry(BENIGN, "benign_edgy_prompts", "CC BY 4.0 (fixture)")]
    path.write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _freeze_testset(base: Path, sources_file: Path) -> Path:
    """A frozen fixture test set of `SIZES` per language, with its build record beside it."""
    from conftest import BENIGN, HARM_A, HARM_B

    def rows(prefix, n):
        return [testset.RawItem(key=f"{i:03d}", text=f"{prefix} number {i} please") for i in range(n)]

    pools = {HARM_A: ("ha", rows("harm a question", 40)), HARM_B: ("hb", rows("harm b question", 40)),
             BENIGN: ("be", rows("benign edgy question", 30))}
    specs = {name: testset.SourceSpec(name, prefix, {}, lambda raw_dir, items=items: list(items))
             for name, (prefix, items) in pools.items()}
    manifest = yaml.safe_load(sources_file.read_text(encoding="utf-8"))
    forward = lambda texts, src, tgt: [f"[{tgt}] {t}" for t in texts]  # noqa: E731
    back = lambda texts, src, tgt: [t.removeprefix(f"[{src}] ") for t in texts]  # noqa: E731
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(testset.TRANSLATORS, "nllb-600m", lambda: forward)
        patch.setitem(testset.TRANSLATORS, "llm-chat", lambda: forward)
        patch.setitem(testset.TRANSLATORS, "fake-back", lambda: back)
        testset.run_build(
            out_dir=base / "testset", raw_root=base / "raw", manifest=manifest, langs=config.LANGS,
            source_names=tuple(pools), seed=7, sizes=SIZES, sources=specs, translator_names=TRANSLATORS,
            back_translator_names={lang: "fake-back" for lang in config.NON_ENGLISH}, same_meaning_name="exact",
            freeze=True, summary_path=base / "testset_build_summary.json",
        )
    return base / "testset"


def build_public_export(base: Path, scenario: str = "default") -> PublicExport:
    """Build the whole chain under `base` (which must not exist or be empty) and return where things are."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    options = SCENARIOS[scenario]
    sources_file = _write_sources(base / "sources.yaml")
    frozen = _freeze_testset(base, sources_file)

    prompt_ids: dict[tuple[str, str, str], list[str]] = {}
    for lang in config.LANGS:
        for split in ("dev", "report"):
            for prompt in read_jsonl(frozen / lang / f"{split}.jsonl", Prompt):
                prompt_ids.setdefault((split, lang, prompt.kind), []).append(prompt.id)
    results = build_tree(base / "results", alphas=ALPHAS, overrides={**ALWAYS, **options.get("overrides", {})},
                         ids=lambda split, lang, kind: prompt_ids[(split, lang, kind)])
    if options.get("second_judge", True):
        write_second_judge(results, noise=options.get("noise"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        code = reportrun.main(["--verdicts", str(results), "--out", str(results)], root=base / "repo")
        assert code == reportrun.EXIT_OK
        out = base / "export"
        code = export.main(["--results", str(results), "--testset", str(frozen), "--sources", str(sources_file),
                            "--templates", str(ROOT / "templates"), "--out", str(out)], root=base / "repo")
        assert code == export.EXIT_OK
    return PublicExport(base, results, out)
