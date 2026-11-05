"""Naive weight blend of two checkpoints, streamed tensor by tensor: A2n = (1 - t) * A1 + t * A0.

    from msd.blend import blend_state_dicts
    blend_state_dicts(Path("A1"), Path("A0"), 0.5, Path("A2n_0.50"))
    blend_state_dicts(Path("A1_fp16"), Path("A0_bf16"), 0.5, Path("A2n_0.50"), a0_dtype="cast_to_a1")

`t = 0` gives A1 (the drifted model) and `t = 1` gives A0 (the aligned one). It is the baseline the ReSta
repair (`scripts/run_resta.py`) is compared with, and it needs no GPU: only safetensors files and CPU torch
(the optional `blend` extra; torch is imported when a blend runs, not when this module is imported).

Inputs: two directories, each holding either one `model.safetensors` or shards named in a
`model.safetensors.index.json` (a directory with both is refused as ambiguous). The two models may be split
differently: A0's tensors are found through A0's own index. The output keeps A1's layout and shard file
names. It is written to a temporary sibling folder and renamed into place, so a crash or a failed check
leaves no `out_dir` behind (or leaves a pre-existing empty one empty). `out_dir` must not exist or must be
empty, must not be a symbolic link, and must not be, or be inside, the A1 or A0 folder; all of that is checked
before a file is read (`msd.atomicdir`). A partial folder left by a killed run is named in a one-line note on
stderr at the start of the next run and never deleted.

What the output holds
    weights       every A1 shard, same file name, same tensor names, same dtypes, same shapes. A model that
                  was one file stays one file with no index; a sharded one gets a new
                  `model.safetensors.index.json` (`weight_map` as in A1, `total_size` recomputed, any other
                  `metadata` keys of A1's index kept).
    metadata      each shard's safetensors metadata is A1's, written with its keys sorted; `format: pt` is
                  added when A1's shard has none (a loader asks for it).
    other files   copied from the top level of the A1 folder, byte for byte, by an ALLOWLIST: config.json,
                  generation_config.json, tokenizer* (every tokenizer file), special_tokens_map.json,
                  added_tokens.json, chat_template*, *.jinja, *.model, vocab*, merges.txt and
                  preprocessor_config.json; but never a name ending in .bak, .tmp, .orig, .swp or ~. Anything
                  else is left behind on purpose: weights and weight indexes, adapter_config.json, trainer
                  state, optimizer and RNG files, READMEs, sub-folders and dot-files. Nothing is taken from the
                  A0 folder. The output keeps no record of how it was made (no sidecar); keep the arguments,
                  `t` and `a0_dtype` included, in your own manifest.

Arithmetic
    A floating tensor is computed in float32 (float64 for a float64 tensor) as `a1 * (1 - t) + a0 * t` and
    cast back to A1's dtype (float16, bfloat16, float32 and so on).
    `t == 0` and `t == 1` do not compute: the A1 (or A0) tensor is copied bit for bit, so an inf or NaN
    survives and `0 * inf` never yields NaN. At `t == 0` the A0 floating tensors are not even read, at
    `t == 1` the A1 ones are not.
    A tensor that is not floating (integer, bool, complex buffers) must be identical in A1 and A0, for every
    `t`, or the blend raises; it is copied from A1.
    A defensive check, not a case that arises: a convex combination of finite numbers is finite, so with finite
    inputs and `t` in [0, 1] the result cannot hold a NaN or inf (a test blends the largest finite values of every
    dtype at many `t`). If one ever appears, from a bug in the arithmetic or a corrupted read, the blend raises
    naming the tensor. Inputs that were already non-finite pass through as they would.

Dtypes: `a0_dtype="strict"` (the default) refuses a floating dtype mismatch between A1 and A0 for the same
tensor, because the arms of this study are normally saved in one dtype and a mismatch means one of the two was
saved wrongly. `a0_dtype="cast_to_a1"` is the explicit opt-in for the case where they really differ (A1 is
the fp16 merge made on a T4, A0 comes from the Hub as bf16): each A0 floating tensor whose dtype differs from
A1's is converted to float32 and then to A1's dtype with round-to-nearest-even, and blended as usual, so the
output is always in A1's dtype. Only float16, bfloat16, float32 and float64 are supported for the cast. A cast
that would turn a finite value into inf (bf16 reaches 3e38, fp16 stops at 65504) raises, naming the tensor;
values too small for the target flush to subnormals or zero without notice, which is what that conversion does.
Tensors whose dtypes already match are not touched. At `t == 0` no A0 tensor is read, so nothing is cast or
checked beyond the header. Non-floating tensors are never cast: a mismatch there is always an error.

Refused before anything is written (ValueError, all reading only the shard headers): tensor names that are
not exactly the same set in A1 and A0 (a few missing or extra names are listed), a shape mismatch, and a
dtype mismatch under `a0_dtype="strict"` (or any mismatch that is not float against float). `t` that is not a
real number is a TypeError; one that is not finite or outside [0, 1] is a ValueError, and so is an unknown
`a0_dtype`.

Memory: one A1 shard file and one A0 shard file are open at a time, and the output is written straight to
disk, so a model's tensors are never held together: at most the tensor being mixed from each side plus its
float32 temporaries are alive (about 10 bytes per element of the largest tensor for bf16 inputs). Two full
models are never loaded.

Determinism: the same inputs give byte-identical output files. The shard files are written by this module,
not by `safetensors.torch.save_file`, because that function writes a shard's metadata keys in a per-process
random order (checked with safetensors 0.8.0: six runs, six different files, once there are two or more
keys). Here the header is compact JSON with metadata keys sorted, padded with spaces to a multiple of 8
bytes, followed by the raw little-endian tensor bytes. Tensors are laid out widest dtype first, then by
name, so that every payload starts at a multiple of its item size, as in a library-written file (the library
hands out zero-copy tensors at those addresses). This is the documented safetensors layout and the library
reads the files back; the tensors, not the byte layout, equal A1's at `t == 0` (a library-written A1 file may
order equal-width dtypes differently).
"""

from __future__ import annotations

import fnmatch
import json
import math
import numbers
import shutil
import sys
from pathlib import Path
from typing import Literal

from safetensors import safe_open

from msd.atomicdir import atomic_output_dir, check_out_dir, note_stale_partials

SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"

# Bytes per element for the dtype names a safetensors header uses. A name not listed here is refused.
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8, "C64": 8,
}  # fmt: skip

# Top-level files of the A1 folder that are copied into the output (fnmatch patterns, case sensitive) ...
_COPY_ALLOWLIST = (
    "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json",
    "chat_template*", "*.jinja", "*.model", "vocab*", "merges.txt", "preprocessor_config.json",
)  # fmt: skip
# ... unless they are editor or backup leftovers.
_NEVER_COPY = ("*.bak", "*.tmp", "*.orig", "*.swp", "*~")

A0DtypeMode = Literal["strict", "cast_to_a1"]
_A0_DTYPE_MODES = ("strict", "cast_to_a1")
# Header dtype names that `cast_to_a1` can convert between, and their torch names.
_CASTABLE = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
_LISTED_NAMES = 5  # how many tensor names an error message shows before "and N more"


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("msd.blend needs torch (CPU is enough): pip install 'msd[blend]'") from exc
    return torch


# --- reading a model directory (headers only) ------------------------------------------------------------------


class _Model:
    """What a checkpoint directory holds, read from its headers: no tensor data is loaded."""

    def __init__(self, label: str, root: Path):
        self.label = label
        self.root = root
        self.sharded = False
        self.shard_of: dict[str, str] = {}  # tensor name -> shard file name
        self.tensors: dict[str, tuple[tuple[int, ...], str]] = {}  # name -> (shape, dtype name)
        self.shard_meta: dict[str, dict[str, str]] = {}  # shard file -> safetensors metadata
        self.index_meta: dict = {}  # "metadata" of the index file (sharded only)

    @property
    def files(self) -> list[str]:
        return sorted(set(self.shard_of.values()))


def _read_model(label: str, root) -> _Model:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{label}: {root} is not a directory")
    single, index = root / SINGLE_FILE, root / INDEX_FILE
    model = _Model(label, root)
    if single.exists() and index.exists():
        raise ValueError(f"{label}: {root} holds both {SINGLE_FILE} and {INDEX_FILE}; remove one, it is ambiguous")
    if index.exists():
        model.sharded = True
        try:
            doc = json.loads(index.read_text(encoding="utf-8"))
            weight_map = doc["weight_map"]
            model.index_meta = dict(doc.get("metadata") or {})
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{label}: {index} is not a usable {INDEX_FILE} ({exc!r})") from exc
        if not isinstance(weight_map, dict) or not all(isinstance(v, str) for v in weight_map.values()):
            raise ValueError(f"{label}: weight_map in {index} must map tensor names to shard file names")
        for filename in set(weight_map.values()):
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"{label}: {filename!r} in {index} is not a plain shard file name")
        wanted = {f: {n for n, g in weight_map.items() if g == f} for f in set(weight_map.values())}
    elif single.exists():
        wanted = {SINGLE_FILE: None}
    else:
        raise FileNotFoundError(f"{label}: no {SINGLE_FILE} and no {INDEX_FILE} in {root}")

    for filename in sorted(wanted):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"{label}: shard {filename} named by {INDEX_FILE} is missing from {root}")
        with safe_open(str(path), framework="numpy") as handle:  # headers only: no tensor is read, torch is not needed
            names = list(handle.keys())
            model.shard_meta[filename] = dict(handle.metadata() or {})
            for name in names:
                piece = handle.get_slice(name)
                shape, dtype = tuple(piece.get_shape()), piece.get_dtype()
                if dtype not in _ITEMSIZE:
                    raise ValueError(f"{label}: tensor {name!r} has dtype {dtype}, which this blend does not know")
                model.tensors[name] = (shape, dtype)
                model.shard_of[name] = filename
        if wanted[filename] is not None and set(names) != wanted[filename]:
            odd = sorted(set(names) ^ wanted[filename])
            raise ValueError(
                f"{label}: {INDEX_FILE} and shard {filename} disagree about {_few(odd)}"
            )
    return model


def tensor_headers(root) -> dict[str, tuple[tuple[int, ...], str]]:
    """`{tensor name: (shape, safetensors dtype name)}` of a checkpoint folder, from its headers alone.

    Reads a single `model.safetensors` or the shards of `model.safetensors.index.json`; loads no tensor and
    needs no torch. FileNotFoundError when the folder holds neither; ValueError for an unusable index."""
    return dict(_read_model("model", root).tensors)


def _few(names) -> str:
    names = sorted(names)
    shown = ", ".join(repr(n) for n in names[:_LISTED_NAMES])
    return shown + (f" and {len(names) - _LISTED_NAMES} more" if len(names) > _LISTED_NAMES else "")


def _nbytes(shape, dtype: str) -> int:
    return math.prod(shape) * _ITEMSIZE[dtype]


def _is_float(dtype: str) -> bool:
    return dtype.startswith(("F", "BF"))


def _check_compatible(a1: _Model, a0: _Model, a0_dtype: str = "strict") -> None:
    only1, only0 = set(a1.tensors) - set(a0.tensors), set(a0.tensors) - set(a1.tensors)
    if only1 or only0:
        parts = []
        if only1:
            parts.append(f"only in A1: {_few(only1)}")
        if only0:
            parts.append(f"only in A0: {_few(only0)}")
        raise ValueError("A1 and A0 do not hold the same tensors (" + "; ".join(parts) + ")")
    shapes, dtypes, uncastable, floats_differ = [], [], [], False
    for name in sorted(a1.tensors):
        (s1, d1), (s0, d0) = a1.tensors[name], a0.tensors[name]
        if s1 != s0:
            shapes.append(f"{name!r}: A1 {list(s1)} vs A0 {list(s0)}")
        if d1 == d0:
            continue
        if _is_float(d1) and _is_float(d0):
            floats_differ = True
            if a0_dtype == "cast_to_a1":
                if d1 not in _CASTABLE or d0 not in _CASTABLE:
                    uncastable.append(f"{name!r}: A0 {d0} to A1 {d1}")
                continue
        dtypes.append(f"{name!r}: A1 {d1} vs A0 {d0}")
    if shapes:
        raise ValueError("shape mismatch between A1 and A0 for " + "; ".join(shapes[:_LISTED_NAMES]))
    if uncastable:
        raise ValueError(
            "a0_dtype='cast_to_a1' converts only between float16, bfloat16, float32 and float64, not for "
            + "; ".join(uncastable[:_LISTED_NAMES])
        )
    if dtypes:
        hint = (
            "; if A0 is stored in another floating dtype on purpose, pass a0_dtype='cast_to_a1' to convert it to A1's"
            if floats_differ and a0_dtype == "strict"
            else ""
        )
        raise ValueError("dtype mismatch between A1 and A0 (no silent cast) for " + "; ".join(dtypes[:_LISTED_NAMES]) + hint)


# --- the arithmetic --------------------------------------------------------------------------------------------


def _mix(a1, a0, t: float):
    """`a1 * (1 - t) + a0 * t` in float32 (float64 for float64), cast back to a1's dtype. Inputs are not modified."""
    torch = _torch()
    compute = torch.float64 if a1.dtype == torch.float64 else torch.float32
    out = a1.to(compute, copy=True)
    out.mul_(1.0 - t)
    other = a0.to(compute, copy=True)
    other.mul_(t)
    out.add_(other)
    del other
    return out.to(a1.dtype)


def _a0_tensor(other, a1: _Model, a0: _Model, name: str):
    """A0's tensor `name`, converted to A1's dtype through float32 when the two differ (`a0_dtype="cast_to_a1"`)."""
    tensor = other.get(name)
    d1, d0 = a1.tensors[name][1], a0.tensors[name][1]
    if d1 == d0:
        return tensor
    torch = _torch()
    cast = tensor.to(torch.float32).to(getattr(torch, _CASTABLE[d1]))  # round to nearest even
    if bool((torch.isinf(cast) & torch.isfinite(tensor)).any()):
        raise ValueError(
            f"casting {name!r} from A0's {d0} to A1's {d1} overflows to inf: it holds finite values outside the "
            f"{d1} range (a0_dtype='cast_to_a1')"
        )
    return cast


def _blend_tensor(name: str, a1, a0, t: float):
    """The mix of two tensors; the finite-output check is defensive (see the module docstring), it cannot fire for finite inputs."""
    torch = _torch()
    out = _mix(a1, a0, t)
    if not bool(torch.isfinite(out).all()) and bool(torch.isfinite(a1).all()) and bool(torch.isfinite(a0).all()):
        raise ValueError(f"blending {name!r} at t={t} produced non-finite values from finite inputs")
    return out


# --- writing ---------------------------------------------------------------------------------------------------


class _OneOpen:
    """Reads tensors of the second model while keeping at most one of its shard files open."""

    def __init__(self, model: _Model):
        self.model = model
        self._file = None
        self._handle = None

    def get(self, name: str):
        filename = self.model.shard_of[name]
        if filename != self._file:
            self.close()
            handle = safe_open(str(self.model.root / filename), framework="pt")
            self._handle = handle.__enter__()
            self._file = filename
        return self._handle.get_tensor(name)

    def close(self) -> None:
        if self._handle is not None:
            handle, self._handle, self._file = self._handle, None, None
            handle.__exit__(None, None, None)


def _tensor_bytes(tensor, expected: int, name: str):
    arr = tensor.contiguous().reshape(-1).view(_torch().uint8).numpy()
    if arr.nbytes != expected:
        raise ValueError(f"tensor {name!r} came out as {arr.nbytes} bytes, its header says {expected}")
    return arr


def _shard_header(a1: _Model, filename: str, names: list[str]) -> bytes:
    meta = dict(a1.shard_meta[filename])
    meta.setdefault("format", "pt")
    header: dict = {"__metadata__": dict(sorted(meta.items()))}
    offset = 0
    for name in names:
        shape, dtype = a1.tensors[name]
        size = _nbytes(shape, dtype)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return len(raw).to_bytes(8, "little") + raw


def _write_shard(a1: _Model, a0: _Model, t: float, filename: str, folder: Path) -> None:
    torch = _torch()
    # widest dtype first, then by name: every payload then starts at a multiple of its item size
    names = sorted((n for n, f in a1.shard_of.items() if f == filename), key=lambda n: (-_ITEMSIZE[a1.tensors[n][1]], n))
    other = _OneOpen(a0)
    try:
        with open(folder / filename, "wb") as out, safe_open(str(a1.root / filename), framework="pt") as mine:
            out.write(_shard_header(a1, filename, names))
            for name in names:
                shape, dtype = a1.tensors[name]
                if _is_float(dtype) and t == 0.0:
                    tensor = mine.get_tensor(name)
                elif _is_float(dtype) and t == 1.0:
                    tensor = _a0_tensor(other, a1, a0, name)
                elif _is_float(dtype):
                    tensor = _blend_tensor(name, mine.get_tensor(name), _a0_tensor(other, a1, a0, name), t)
                else:
                    tensor = mine.get_tensor(name)
                    if not torch.equal(tensor, other.get(name)):
                        raise ValueError(f"tensor {name!r} is not floating ({dtype}) and differs between A1 and A0")
                out.write(memoryview(_tensor_bytes(tensor, _nbytes(shape, dtype), name)))
                del tensor
    finally:
        other.close()


def _write_index(a1: _Model, folder: Path) -> None:
    total = sum(_nbytes(*a1.tensors[n]) for n in a1.tensors)
    meta = {k: v for k, v in a1.index_meta.items() if k != "total_size"}
    meta["total_size"] = total
    doc = {"metadata": meta, "weight_map": dict(sorted(a1.shard_of.items()))}
    (folder / INDEX_FILE).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_copied(name: str) -> bool:
    if name.startswith(".") or any(fnmatch.fnmatchcase(name, pattern) for pattern in _NEVER_COPY):
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _COPY_ALLOWLIST)


def _copy_other_files(root: Path, folder: Path) -> None:
    for path in sorted(root.iterdir()):
        if path.is_file() and _is_copied(path.name):
            shutil.copyfile(path, folder / path.name)


def _check_t(t) -> float:
    if isinstance(t, bool) or not isinstance(t, numbers.Real):
        raise TypeError(f"t must be a real number, got {type(t).__name__}")
    t = float(t)
    if not math.isfinite(t) or not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be a finite number in [0, 1], got {t}")
    return t


def _check_a0_dtype(value) -> str:
    if not isinstance(value, str) or value not in _A0_DTYPE_MODES:
        raise ValueError(f"a0_dtype must be one of {list(_A0_DTYPE_MODES)}, got {value!r}")
    return value


def blend_state_dicts(a1_dir: Path, a0_dir: Path, t: float, out_dir: Path, *, a0_dtype: A0DtypeMode = "strict") -> None:
    """Write `(1 - t) * A1 + t * A0` to `out_dir`, shard by shard. See the module docstring for the rules.

    `a0_dtype="strict"` (default) refuses a floating dtype mismatch; `"cast_to_a1"` converts A0's floating
    tensors to A1's dtype first (float32, round to nearest even; an overflow to inf raises)."""
    if sys.byteorder != "little":  # pragma: no cover - the writer emits little-endian bytes from memory
        raise RuntimeError("msd.blend writes tensor bytes as they are in memory and supports little-endian hosts only")
    t = _check_t(t)
    a0_dtype = _check_a0_dtype(a0_dtype)
    out_dir = Path(out_dir)
    check_out_dir(out_dir, sources=(("A1", a1_dir), ("A0", a0_dir)))  # before any file is read
    note_stale_partials(out_dir)
    _torch()
    a1, a0 = _read_model("A1", a1_dir), _read_model("A0", a0_dir)
    _check_compatible(a1, a0, a0_dtype)

    with atomic_output_dir(out_dir) as folder:
        for filename in a1.files:
            _write_shard(a1, a0, t, filename, folder)
        if a1.sharded:
            _write_index(a1, folder)
        _copy_other_files(a1.root, folder)
