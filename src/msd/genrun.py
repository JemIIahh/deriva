"""Command line for the generation runner (scripts/generate.py is a thin entry point to `main`).

    python scripts/generate.py --arm A0 --split dev --model-path CohereLabs/tiny-aya-global
    python scripts/generate.py --arm A2 --alpha 0.25 --split report --model-path /kaggle/working/merged
    python scripts/generate.py --arm A0 --split dev --backend fake --out-root /tmp/scratch --limit 20

For each requested language it reads `<testset-dir>/<lang>/<split>.jsonl`, runs `msd.gen.generate`
and writes `msd.gen.gens_path(out_root, split, arm, alpha, lang)` plus a `<lang>.gens.meta.json`
next to it (model, arm, alpha, split, batch size, max_new_tokens, dtype, backend, msd version,
git commit, UTC time, row count). Languages whose test-set file is not there yet are skipped with a
message; it is a failure when no requested language could be generated.

The `fake` backend writes clearly marked canned text and is refused when `--out-root` is, or is
inside, the real `results/` directory, so fake output can never end up among real results.

A meta file is the record of how a gens file was made, and it is written BEFORE the first batch
(atomically: temp file, fsync, rename), so even a SIGKILL, an out-of-memory kill or a lost Colab
session leaves a gens file with its meta next to it. Rows from two setups must not share a file:

  * a gens file that has rows and a meta with a different model path, alpha, batch size,
    max_new_tokens, dtype or backend is refused, and so is a gens file that has rows but NO meta
    (what made them is unknown), before any generation or model load, unless `--force-new-meta` is
    given. A forced run keeps the replaced setup in the new meta under "earlier", and marks rows
    of unknown origin as "unrecorded_rows".
  * a meta with no rows behind it (a first batch that failed, for example out of memory) describes
    nothing and is simply replaced, so a retry with a smaller --batch-size is fine.
  * the meta is rewritten only when it is new, when the setup changed, or when new rows were
    generated; a rerun that has nothing to do leaves it byte-identical.

Everything that can be refused is refused before the first generation, for every requested
language: duplicate prompt ids, a corrupt or foreign existing gens file, a meta conflict.
Resuming after a crash is therefore: run the same command again (same settings).
There is one writer per output file; do not start two runs on the same --out-root at once.

Exit codes: 0 done; 2 usage error (bad arguments, fake backend aimed at results/); 3 none of the
requested languages has a test-set file; 1 any other failure (a refused or corrupt file, a backend
or model error). Never 0 on an error. Prompt texts and completions are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from msd import gen
from msd.config import ARMS, LANGS
from msd.schema import SPLITS, Prompt, read_jsonl

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_DATA = 0, 1, 2, 3

# The setup fields: a gens file must be made with one value of each (commit, version and time may change).
SETUP_KEYS: tuple[str, ...] = (
    "model_path", "arm", "alpha", "batch_size", "max_new_tokens", "dtype", "backend",
)
DTYPES: tuple[str, ...] = ("float16", "bfloat16", "float32")


class MetaMismatchError(RuntimeError):
    """An existing meta file records a different setup (or cannot be read)."""


# --- arguments --------------------------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _alpha_arg(text: str) -> float | None:
    if text.strip().lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'none', got {text!r}") from None


def _langs_arg(text: str) -> list[str]:
    langs = [part.strip() for part in text.split(",")]
    if not text.strip() or any(not lang for lang in langs):
        raise argparse.ArgumentTypeError("expected a comma-separated list of languages")
    unknown = [lang for lang in langs if lang not in LANGS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown languages {unknown}; expected a subset of {list(LANGS)}")
    if len(set(langs)) != len(langs):
        raise argparse.ArgumentTypeError(f"duplicate languages in {text!r}")
    return langs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="generate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", help="Hugging Face repo id or local (merged) model directory; required for --backend hf")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--alpha", type=_alpha_arg, default=None,
                    help="blend / repair setting; required for A2 and A2n, forbidden for BASE, A0, A1, A3")
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--langs", type=_langs_arg, default=None, help=f"comma-separated subset of {','.join(LANGS)} (default: all)")
    ap.add_argument("--limit", type=_positive_int, default=None, help="only the first N prompts of each language file (smoke runs)")
    ap.add_argument("--out-root", type=Path, default=None, help="default: results/ under the repository root")
    ap.add_argument("--testset-dir", type=Path, default=None, help="default: data/testset under the repository root")
    ap.add_argument("--batch-size", type=_positive_int, default=gen.DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=_positive_int, default=256)
    ap.add_argument("--dtype", choices=DTYPES, default="float16")
    ap.add_argument("--backend", choices=("hf", "fake"), default="hf")
    ap.add_argument("--force-new-meta", action="store_true",
                    help="accept a different setup than the one recorded in an existing meta file")
    return ap


def _folded(path: Path) -> list[str]:
    return [part.casefold() for part in path.parts]


def is_in_results(out_root: Path, root: Path) -> bool:
    """True if `out_root` is, or is inside, the real results directory (of the repo or of the cwd).

    Compared on resolved paths (so `..` and symlinks are followed), case-folded (a case-insensitive
    file system such as the macOS default treats `Results` as `results`), and, where the paths
    exist, by `os.path.samefile` on the target and each of its parents (the file system itself says
    they are one directory).
    """
    target = Path(out_root).resolve()
    target_parts = _folded(target)
    for base in {(Path(root) / "results").resolve(), (Path.cwd() / "results").resolve()}:
        base_parts = _folded(base)
        if target_parts[: len(base_parts)] == base_parts:
            return True
        if base.exists():
            for candidate in (target, *target.parents):
                try:
                    if candidate.exists() and os.path.samefile(candidate, base):
                        return True
                except OSError:
                    continue
    return False


def parse_args(argv: Sequence[str] | None, *, root: Path) -> argparse.Namespace:
    """Parse and validate; any usage error prints the usage and exits with code 2 (SystemExit)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        gen.check_arm_alpha(args.arm, args.alpha)
    except ValueError as exc:
        ap.error(str(exc))
    args.langs = list(LANGS) if args.langs is None else args.langs
    args.out_root = Path(root) / "results" if args.out_root is None else args.out_root
    args.testset_dir = Path(root) / "data" / "testset" if args.testset_dir is None else args.testset_dir
    if args.backend == "fake" and is_in_results(args.out_root, root):
        ap.error(
            f"--backend fake is refused for --out-root {args.out_root}: that is the real results directory, and "
            "canned fake text must never end up among real results. Use a scratch directory."
        )
    if args.backend == "hf" and not args.model_path:
        ap.error("--model-path is required for --backend hf")
    if args.model_path is None:
        args.model_path = "fake"
    return args


# --- prompts and metadata ----------------------------------------------------------------------------------


def load_prompts(path: Path, lang: str, split: str, limit: int | None) -> list[Prompt]:
    """The first `limit` prompts (all if None) of a test-set file; rows must belong to this lang and split."""
    prompts = read_jsonl(path, Prompt)
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"{path}: no prompts to generate for")
    repeated = sorted(pid for pid, n in Counter(p.id for p in prompts).items() if n > 1)
    if repeated:
        raise ValueError(f"{path}: duplicate prompt ids {repeated[:5]}")
    wrong = [p.id for p in prompts if p.lang != lang or p.split != split]
    if wrong:
        raise ValueError(f"{path}: {len(wrong)} row(s) are not {lang}/{split} rows, e.g. {wrong[:3]}")
    return prompts


def meta_differences(old: dict[str, Any], new: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(key, old value, new value) for each setup field that differs between two meta dicts."""
    return [(key, old.get(key), new.get(key)) for key in SETUP_KEYS if old.get(key) != new.get(key)]


def read_meta(meta_file: Path, *, strict: bool) -> dict[str, Any] | None:
    """The meta dict in `meta_file`, None if absent; an unreadable file raises unless not `strict`."""
    if not meta_file.is_file():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        if strict:
            raise MetaMismatchError(
                f"{meta_file} cannot be read ({exc}); pass --force-new-meta to replace it"
            ) from exc
        return None
    return meta


def check_meta(
    meta_file: Path, gens_file: Path, new: dict[str, Any], *, n_rows: int, force: bool
) -> tuple[dict[str, Any] | None, bool, int]:
    """Decide what to do with the meta of a gens file that already holds `n_rows` finished rows.

    Returns (the previous meta to keep as history, whether the meta must be written before
    generating, the number of rows of unknown origin to mark). Raises MetaMismatchError for a file
    with rows whose meta is missing, unreadable or records a different setup, unless `force`. A
    file without rows has nothing to protect: any old meta is dropped and a new one is written.
    """
    if n_rows == 0:
        return None, True, 0
    old = read_meta(meta_file, strict=False)
    if not force:
        if not meta_file.exists():
            raise MetaMismatchError(
                f"{gens_file} already holds {n_rows} row(s) but there is no meta file {meta_file}, so what made "
                "them is unknown (a run that was killed before the meta existed, or a deleted file). Pass "
                "--force-new-meta to continue anyway; the rows are then marked as unrecorded in the new meta"
            )
        if old is None:
            read_meta(meta_file, strict=True)  # raises with the reason and the flag
        diffs = meta_differences(old, new)
        if diffs:
            shown = "; ".join(f"{key}: {was!r} -> {now!r}" for key, was, now in diffs)
            raise MetaMismatchError(
                f"{meta_file} records a different setup than this run ({shown}). Rows made with two setups must not "
                "share a file: use another --out-root, or pass --force-new-meta to accept it (the old setup is kept "
                "in the new meta file under 'earlier')"
            )
        return old, False, 0
    if old is None:
        return None, True, n_rows
    return old, bool(meta_differences(old, new)), 0


def write_meta(meta_file: Path, meta: dict[str, Any], previous: dict[str, Any] | None) -> None:
    """Write `meta` atomically (temp file, fsync, rename); `previous` (minus its own history) goes under "earlier"."""
    if previous is not None:
        meta = {**meta, "earlier": [*previous.get("earlier", []), {k: v for k, v in previous.items() if k != "earlier"}]}
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_file.with_name(meta_file.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, meta_file)


def count_rows(path: Path) -> int:
    return path.read_bytes().count(b"\n") if path.is_file() else 0


# --- running ------------------------------------------------------------------------------------------------


class _LazyBackend:
    """Builds the real backend (loads the model) on the first call, so a run with nothing to do loads nothing."""

    def __init__(self, factory: Callable[[], gen.Backend]) -> None:
        self._factory = factory
        self._backend: gen.Backend | None = None

    def __call__(self, prompts: list[str]) -> list[gen.GenOut]:
        if self._backend is None:
            self._backend = self._factory()
        return self._backend(prompts)


def _make_backend(args: argparse.Namespace) -> gen.Backend:
    if args.backend == "fake":
        return gen.fake_backend()
    return gen.hf_backend(args.model_path, max_new_tokens=args.max_new_tokens, dtype=args.dtype)


def _meta_for(args: argparse.Namespace, lang: str, n_rows: int, root: Path, unrecorded: int = 0) -> dict[str, Any]:
    meta = gen.run_meta(
        model_path=args.model_path, arm=args.arm, alpha=args.alpha, split=args.split, lang=lang,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, dtype=args.dtype, backend=args.backend,
        n_rows=n_rows, repo_root=root,
    )
    if unrecorded:
        meta["unrecorded_rows"] = unrecorded  # rows that were already in the file with no record of their setup
    return meta


@dataclass(frozen=True)
class _Plan:
    lang: str
    prompts: list[Prompt]
    gens_file: Path
    meta_file: Path
    previous: dict[str, Any] | None  # the meta this run replaces (kept as history)
    record_first: bool  # the meta must be written before the first batch
    unrecorded: int
    n_done: int


def run(
    args: argparse.Namespace,
    available: list[tuple[str, Path]],
    root: Path,
    *,
    extra_prompts: Mapping[str, Sequence[Prompt]] | None = None,
) -> int:
    """Generate for every (language, test-set file) in `available`; return the exit code (0).

    `extra_prompts` maps a language to prompts that are generated into the same gens file after the
    test-set prompts (`msd.evalrun` adds the closed-form task items this way, so there is one gens file,
    one meta and one resume rule per arm / alpha / language / split).
    """
    # Everything that can be refused is refused here, for every language, before the first generation
    # and before any model load: duplicate prompt ids, a corrupt or foreign gens file (a dry scan with
    # the code `generate` uses), a missing or conflicting meta.
    plans: list[_Plan] = []
    for lang, path in available:
        prompts = load_prompts(path, lang, args.split, args.limit) + list((extra_prompts or {}).get(lang, ()))
        gens_file = gen.gens_path(args.out_root, args.split, args.arm, args.alpha, lang)
        meta_file = gen.meta_path(gens_file)
        n_done = len(gen.scan_existing(prompts, gens_file, arm=args.arm, alpha=args.alpha).done)
        previous, record_first, unrecorded = check_meta(
            meta_file, gens_file, _meta_for(args, lang, n_done, root), n_rows=n_done, force=args.force_new_meta
        )
        plans.append(_Plan(lang, prompts, gens_file, meta_file, previous, record_first, unrecorded, n_done))

    backend = _LazyBackend(lambda: _make_backend(args))
    total_new = 0
    for plan in plans:
        if plan.record_first:  # so that even a hard kill leaves the setup on record next to its rows
            write_meta(plan.meta_file, _meta_for(args, plan.lang, plan.n_done, root, plan.unrecorded), plan.previous)

        def progress(done: int, total: int, lang: str = plan.lang) -> None:
            print(f"  {lang}: {done}/{total}", file=sys.stderr, flush=True)

        try:
            new = gen.generate(
                plan.prompts, backend, plan.gens_file, arm=args.arm, alpha=args.alpha, batch_size=args.batch_size,
                progress=progress,
            )
        except BaseException:
            rows = count_rows(plan.gens_file)
            if rows > plan.n_done:  # a crash after some batches: record how many rows exist now
                write_meta(plan.meta_file, _meta_for(args, plan.lang, rows, root, plan.unrecorded), plan.previous)
            raise
        if new:
            write_meta(plan.meta_file, _meta_for(args, plan.lang, count_rows(plan.gens_file), root, plan.unrecorded),
                       plan.previous)
        total_new += new
        print(f"{plan.lang}: {len(plan.prompts)} prompts, {new} new, {len(plan.prompts) - new} already done "
              f"-> {plan.gens_file}")
    print(f"done: {total_new} new completions in {len(plans)} language(s)")
    return EXIT_OK


_PLAIN_ERRORS = (gen.ResumeError, gen.BackendError, MetaMismatchError, ValueError, ImportError, OSError)


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        args = parse_args(argv, root=root)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

    available: list[tuple[str, Path]] = []
    for lang in args.langs:
        path = args.testset_dir / lang / f"{args.split}.jsonl"
        if path.is_file():
            available.append((lang, path))
        else:
            print(f"skip {lang}: {path} not found (language not built yet)", file=sys.stderr)
    if not available:
        print(
            f"error: no requested language has a {args.split} test-set file under {args.testset_dir}; nothing to generate",
            file=sys.stderr,
        )
        return EXIT_NO_DATA

    try:
        return run(args, available, root)
    except _PLAIN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
