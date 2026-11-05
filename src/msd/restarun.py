"""Repair the drifted model A1 with SafeTune's ReSta patch (plan Task 16): theta_safe = A1 + alpha * (A0 - BASE).

    python scripts/run_resta.py --drifted A1_DIR --base CohereLabs/tiny-aya-base \\
        --aligned CohereLabs/tiny-aya-global --alpha 0.25 --out A2_alpha_0.25

`--drifted` is the local folder of the drifted model (A1). `--base` and `--aligned` are Hugging Face repo ids or
local folders (Tiny Aya Base and Tiny Aya Global; both are gated: accept the licence on the model pages and
put the token in the HF_TOKEN environment variable, there is no token flag and no token is ever printed or
written). `--out` must not exist or must be empty; it is written to a temporary folder and renamed into place,
so a failed run leaves nothing behind. It receives the patched model (`save_pretrained`), the tokenizer of
`--drifted`, and `resta_patch.json`, which records every setting of the run (no secrets).

What it calls (read from the source of safetune 0.1.12, see docs/toolkit-notes.md, "SafeTune ReSta (read from
source)"; nothing below has been run against the real package):

    from safetune.runner import recover
    recover.ReStaTrainer(drifted_model, base_model=base_model, aligned_model=aligned_model, alpha=alpha,
                         dare=..., dare_drop_rate=..., dare_seed=..., device="cpu").apply()

The trainer takes loaded models (this module loads the three, each in the dtype it was saved in, on the CPU:
about three times the model size in RAM), patches the drifted one in place and returns it, and writes nothing;
this module then saves it. The arithmetic is, per tensor and in float32, `drifted + alpha * dare(aligned - base)`;
`alpha = 0` adds exactly zero. `--dare` is on by default because `ReStaTrainer` has DARE on by default (drop
probability 0.3, seed 0, the RESTA paper's values); every DARE setting is passed explicitly so the call does not
depend on the library's defaults, and `--no-dare` gives the plain `drifted + alpha * (aligned - base)`.
SafeTune's own docs report that on Tiny Aya alpha = 1 breaks the model and alpha of 0.5 or more does too.

UNVERIFIED until the first GPU-runtime sanity run (plan Task 16, Step 6): that this call matches the real
package, that the three models have the same tensor names, and that `--alpha 0` returns the drifted model
bit for bit. Run alpha 0 first, compare its tensors and greedy outputs with the drifted model on 5 prompts,
and do not sweep alpha on a guess.

SafeTune is imported inside `import_trainer` only, after the arguments are checked and before any model is
loaded, so `--help`, the argument checks and this module's import work without it (it needs Python 3.12,
torch 2.7 and transformers 5.15; docs/runtime.md, section 4). The three models are compared tensor by tensor
(names and shapes) before the trainer is built, because ReSta skips a tensor that is missing from either
reference without a word; a mismatch is refused. The comparison is made as early as it can be:

  1. Local folders first, from the safetensors headers, before any model is loaded (the drifted folder against
     each of --base / --aligned that is a folder; a folder with no model.safetensors, or an index, is skipped).
  2. Hub ids pairwise, right after each model loads: drifted against base after the second load, drifted
     against aligned after the third, so a wrong reference costs one download instead of two.
  3. That check after loading is kept even when the headers agreed: tied weights and renamed keys can make a
     loaded state dict differ from the file it came from.

A header comparison can also refuse a pair that would have loaded to the same state dict (a tied weight such as
lm_head stored in one folder and not in the other); the message says the check came from the headers.

A stale `.<out>.<8 hex>.partial` folder left by a killed earlier run is named in one line at the start and never
deleted.

Exit codes: 0 done; 1 any failure (SafeTune missing or not importable, a model that cannot be loaded, mismatched
tensors, a failed patch or save); 2 usage error (bad or missing arguments, a non-empty --out, --drifted not a
folder, --base and --aligned the same model, an --out that is a symbolic link or is, or is inside, one of the
local model folders).
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from msd.atomicdir import atomic_output_dir, check_out_dir, note_stale_partials
from msd.blend import tensor_headers

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2

DEFAULT_DARE_DROP_RATE = 0.3  # the RESTA paper's value, and ReStaTrainer's default when dare_drop_rate is None
DEFAULT_DARE_SEED = 0  # ReStaTrainer's default
SIDECAR = "resta_patch.json"
NOTE = (
    "UNVERIFIED: the call into safetune.runner.recover.ReStaTrainer was written from the source of safetune 0.1.12 "
    "and had not been run when this wrapper was written; it counts as checked only after the alpha=0 sanity run "
    "(greedy outputs equal the drifted model on 5 prompts, plan Task 16, Step 6)."
)

_TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_TOKEN_SHAPE = re.compile(r"hf_[A-Za-z0-9]{8,}")
_LISTED_NAMES = 5


class ReStaError(RuntimeError):
    """A refusal or failure with a message meant for the person running the script."""


def scrub(text: str) -> str:
    """Replace Hugging Face tokens (the ones in the environment and anything shaped like `hf_...`) in `text`."""
    for variable in _TOKEN_VARIABLES:
        value = os.environ.get(variable)
        if value and len(value) >= 8:
            text = text.replace(value, "<token>")
    return _TOKEN_SHAPE.sub("<token>", text)


# --- arguments -------------------------------------------------------------------------------------------------


def _alpha(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"alpha must be a finite number from 0 to 1, got {text!r}")
    return value


def _drop_rate(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"the DARE drop rate must be a number from 0 up to (not including) 1, got {text!r}")
    return value


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"the DARE seed must be a non-negative integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"the DARE seed must be a non-negative integer, got {text!r}")
    return value


def _model_ref(text: str) -> str:
    if not text.strip():
        raise argparse.ArgumentTypeError("expected a Hugging Face repo id or a folder")
    return text.strip()


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse repeats unrecognised arguments, and one may be a token
        super().error(scrub(message))


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="run_resta", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drifted", required=True, type=Path, help="folder of the drifted model (A1)")
    ap.add_argument("--base", required=True, type=_model_ref, help="pre-alignment model: repo id or folder (Tiny Aya Base)")
    ap.add_argument("--aligned", required=True, type=_model_ref, help="aligned model: repo id or folder (Tiny Aya Global, A0)")
    ap.add_argument("--alpha", required=True, type=_alpha, help="safety vector scale, from 0 to 1 (0 changes nothing)")
    ap.add_argument("--out", required=True, type=Path, help="folder for the patched model; must not exist or be empty")
    ap.add_argument("--dare", action=argparse.BooleanOptionalAction, default=True,
                    help="DARE drop-and-rescale on the safety vector (default on, as in ReStaTrainer)")
    ap.add_argument("--dare-drop-rate", type=_drop_rate, default=DEFAULT_DARE_DROP_RATE,
                    help=f"DARE drop probability p (default {DEFAULT_DARE_DROP_RATE}, the paper's value)")
    ap.add_argument("--dare-seed", type=_seed, default=DEFAULT_DARE_SEED, help=f"DARE mask seed (default {DEFAULT_DARE_SEED})")
    return ap


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse and validate; any usage error prints the usage and exits with code 2 (SystemExit)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.drifted.is_dir():
        ap.error(f"--drifted must be an existing folder, got {str(args.drifted)!r}")
    if _same_model(args.base, args.aligned):
        ap.error("--base and --aligned are the same model; the safety vector (aligned - base) would be zero")
    sources = [("drifted", args.drifted)] + [
        (label, Path(ref)) for label, ref in (("base", args.base), ("aligned", args.aligned)) if Path(ref).is_dir()
    ]
    try:
        check_out_dir(args.out, sources=sources)
    except (FileExistsError, ValueError) as exc:  # not empty / not a folder; a symbolic link or inside a model folder
        ap.error(str(exc))
    return args


def _same_model(a: str, b: str) -> bool:
    if a == b:
        return True
    pa, pb = Path(a), Path(b)
    return pa.is_dir() and pb.is_dir() and pa.resolve() == pb.resolve()


# --- the pieces that need heavy packages, each imported only when used -------------------------------------------


def import_trainer() -> Callable[..., Any]:
    """`safetune.runner.recover.ReStaTrainer`, imported now (the only place SafeTune is imported)."""
    try:
        from safetune.runner import recover

        return recover.ReStaTrainer
    except ImportError as exc:
        raise ReStaError(
            f"SafeTune could not be imported ({exc}). Install it with `pip install safetune` on the runtime that runs "
            "this (it needs Python 3.12 or newer, torch 2.7 and transformers 5.15; docs/runtime.md, section 4). "
            "If it will not install, the plain-arithmetic fallback is described in docs/toolkit-notes.md."
        ) from exc
    except AttributeError as exc:
        raise ReStaError(
            f"the installed safetune has no recover.ReStaTrainer ({exc}); this wrapper was written against 0.1.12"
        ) from exc


def load_causal_lm(name: str):
    """A causal LM from a repo id or folder on the CPU, in the dtype it was saved in (`dtype="auto"`)."""
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype="auto")
    model.eval()
    return model


def load_tokenizer(name: str):
    import transformers

    return transformers.AutoTokenizer.from_pretrained(name)


_default_load_tokenizer = load_tokenizer  # `main` has a parameter of the same name


# --- the run ---------------------------------------------------------------------------------------------------


def _few(names) -> str:
    names = sorted(names)
    shown = ", ".join(repr(n) for n in names[:_LISTED_NAMES])
    return shown + (f" and {len(names) - _LISTED_NAMES} more" if len(names) > _LISTED_NAMES else "")


def _shapes_of(model) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in model.state_dict().items()}


def _problems(label: str, reference: dict[str, tuple[int, ...]], other: dict[str, tuple[int, ...]]) -> list[str]:
    """What is odd about `other` (named `label`) next to `reference`: missing, extra and differently shaped tensors."""
    missing, extra = set(reference) - set(other), set(other) - set(reference)
    odd_shape = [n for n in set(reference) & set(other) if tuple(reference[n]) != tuple(other[n])]
    problems = []
    if missing:
        problems.append(f"{label} lacks {_few(missing)}")
    if extra:
        problems.append(f"{label} has extra {_few(extra)}")
    if odd_shape:
        problems.append(f"{label} has another shape for {_few(odd_shape)}")
    return problems


def _refuse(problems: list[str], *, source: str = "") -> None:
    """ReSta skips a tensor missing from either reference without a word; refuse instead."""
    if problems:
        raise ReStaError(
            f"the drifted, base and aligned models do not hold the same tensors{source}, and ReSta would skip the odd "
            "ones silently: " + "; ".join(problems)
        )


def _check_local_headers(args: argparse.Namespace) -> None:
    """Compare tensor names and shapes of the local folders from their safetensors headers, loading nothing.

    The drifted folder is the reference. A folder with no safetensors file is skipped (its models are checked
    after loading); so is everything when the drifted folder has none."""
    try:
        reference = {name: shape for name, (shape, _dtype) in tensor_headers(args.drifted).items()}
    except FileNotFoundError:
        return
    problems = []
    for label, ref in (("base", args.base), ("aligned", args.aligned)):
        if not Path(ref).is_dir():
            continue
        try:
            other = {name: shape for name, (shape, _dtype) in tensor_headers(ref).items()}
        except FileNotFoundError:
            continue
        problems += _problems(label, reference, other)
    _refuse(problems, source=" (compared from the safetensors headers, before any model was loaded)")


def _check_loaded(label: str, drifted, other) -> None:
    """The same comparison on a model that has just been loaded, so a wrong one stops the run before the next load."""
    _refuse(_problems(label, _shapes_of(drifted), _shapes_of(other)))


def _dtypes(model) -> list[str]:
    return sorted({str(getattr(v, "dtype", "unknown")).removeprefix("torch.") for v in model.state_dict().values()})


def _safetune_version() -> str | None:
    try:
        return importlib.metadata.version("safetune")
    except importlib.metadata.PackageNotFoundError:
        return None


def _sidecar(args: argparse.Namespace, patched) -> dict:
    inner = "dare(theta_aligned - theta_base)" if args.dare else "(theta_aligned - theta_base)"
    return {
        "method": "SafeTune ReStaTrainer (ReSta)",
        "formula": f"theta_safe = theta_drifted + alpha * {inner}",
        "alpha": args.alpha,
        "dare": args.dare,
        "dare_drop_rate": args.dare_drop_rate,
        "dare_seed": args.dare_seed,
        "device": "cpu",
        "drifted": str(args.drifted),
        "base": args.base,
        "aligned": args.aligned,
        "safetune_version": _safetune_version(),
        "tensors": len(patched.state_dict()),
        "dtypes": _dtypes(patched),
        "note": NOTE,
    }


def run(
    args: argparse.Namespace,
    *,
    trainer_cls: Callable[..., Any] | None = None,
    load_model: Callable[[str], Any] | None = None,
    load_tokenizer: Callable[[str], Any] | None = None,
) -> None:
    note_stale_partials(args.out)  # leftovers of a killed earlier run: named, never deleted
    _check_local_headers(args)  # local folders: names and shapes from the headers, before anything is loaded
    trainer_cls = trainer_cls if trainer_cls is not None else import_trainer()  # before any model is loaded
    load_model = load_model if load_model is not None else load_causal_lm
    load_tokenizer = load_tokenizer if load_tokenizer is not None else _default_load_tokenizer

    print("loading the drifted, base and aligned models in full on the CPU (about 3x one model in RAM)", file=sys.stderr)
    drifted = load_model(str(args.drifted))
    base = load_model(args.base)
    _check_loaded("base", drifted, base)  # a wrong base is found before the aligned model is fetched
    aligned = load_model(args.aligned)
    _check_loaded("aligned", drifted, aligned)
    print(f"dtypes: drifted {_dtypes(drifted)}, base {_dtypes(base)}, aligned {_dtypes(aligned)}", file=sys.stderr)

    trainer = trainer_cls(
        drifted,
        base_model=base,
        aligned_model=aligned,
        alpha=args.alpha,
        dare=args.dare,
        dare_drop_rate=args.dare_drop_rate,
        dare_seed=args.dare_seed,
        device="cpu",
    )
    patched = trainer.apply()  # patches the drifted model in place and returns it
    if patched is None:
        patched = drifted
    del trainer, base, aligned
    gc.collect()

    tokenizer = load_tokenizer(str(args.drifted))
    with atomic_output_dir(args.out) as folder:
        patched.save_pretrained(folder, safe_serialization=True)
        tokenizer.save_pretrained(folder)
        (folder / SIDECAR).write_text(json.dumps(_sidecar(args, patched), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    trainer_cls: Callable[..., Any] | None = None,
    load_model: Callable[[str], Any] | None = None,
    load_tokenizer: Callable[[str], Any] | None = None,
) -> int:
    """Run the patch; `trainer_cls`, `load_model` and `load_tokenizer` are the injection points tests use."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse: 2 for a usage error, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        run(args, trainer_cls=trainer_cls, load_model=load_model, load_tokenizer=load_tokenizer)
    except (ReStaError, OSError, ValueError, ImportError) as exc:
        print(f"error: {scrub(str(exc))}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        print(scrub(traceback.format_exc()), file=sys.stderr)
        return EXIT_ERROR
    print(f"wrote {args.out} (ReSta, alpha={args.alpha}, dare={args.dare})")
    print(
        "next: until the first sanity run, check that --alpha 0 gives the drifted model's tensors and greedy outputs "
        "before using any other alpha (plan Task 16, Step 6)",
        file=sys.stderr,
    )
    return EXIT_OK
