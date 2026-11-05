"""Naive weight blend of two checkpoints, streamed tensor by tensor: A2n = (1 - t) * A1 + t * A0.

    from msd.blend import blend_state_dicts
    blend_state_dicts(Path("A1"), Path("A0"), 0.5, Path("A2n_0.50"))

`t = 0` gives A1 (the drifted model) and `t = 1` gives A0 (the aligned one). It is the baseline the ReSta
repair (`scripts/run_resta.py`) is compared with, and it needs no GPU: only safetensors files and CPU torch
(the optional `blend` extra; torch is imported when a blend runs, not when this module is imported).

Inputs: two directories, each holding either one `model.safetensors` or shards named in a
`model.safetensors.index.json` (a directory with both is refused as ambiguous). The two models may be split
differently: A0's tensors are found through A0's own index. The output keeps A1's layout and shard file
names. It is written to a temporary sibling folder and renamed into place, so a crash or a failed check
leaves no `out_dir` behind (or leaves a pre-existing empty one empty). `out_dir` must not exist or must be
empty.

What the output holds
    weights       every A1 shard, same file name, same tensor names, same dtypes, same shapes. A model that
                  was one file stays one file with no index; a sharded one gets a new
                  `model.safetensors.index.json` (`weight_map` as in A1, `total_size` recomputed, any other
                  `metadata` keys of A1's index kept).
    metadata      each shard's safetensors metadata is A1's, written with its keys sorted; `format: pt` is
                  added when A1's shard has none (a loader asks for it).
    other files   every regular, non-hidden file in the top level of the A1 directory that is not a weight
                  file or a weight index (config.json, generation_config.json, tokenizer files, the chat
                  template, a README), copied byte for byte. Sub-folders, dot-entries (an HF download cache),
                  `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.gguf`, `*.h5`, `*.msgpack`,
                  `*.onnx` and `*.index.json` are not copied, and nothing is taken from the A0 directory.

Arithmetic
    A floating tensor is computed in float32 (float64 for a float64 tensor) as `a1 * (1 - t) + a0 * t` and
    cast back to its own dtype (float16, bfloat16, float32 and so on).
    `t == 0` and `t == 1` do not compute: the A1 (or A0) tensor is copied bit for bit, so an inf or NaN
    survives and `0 * inf` never yields NaN. At `t == 0` the A0 floating tensors are not even read, at
    `t == 1` the A1 ones are not.
    A tensor that is not floating (integer, bool, complex buffers) must be identical in A1 and A0, for every
    `t`, or the blend raises; it is copied from A1.
    If a blended floating tensor holds a NaN or inf although both of its inputs are finite, the blend
    raises, naming the tensor. (Inputs that were already non-finite pass through as they would.)

Refused before anything is written (ValueError, all reading only the shard headers): tensor names that are
not exactly the same set in A1 and A0 (a few missing or extra names are listed), a shape mismatch, and a
dtype mismatch. A floating dtype mismatch is refused rather than cast: the arms of this study are saved in
one dtype, so a mismatch means one of the two was saved wrongly, and a silent cast would hide it. `t` that is
not a real number is a TypeError; one that is not finite or outside [0, 1] is a ValueError.

Memory: one A1 shard file and one A0 shard file are open at a time, and the output is written straight to
disk, so a model's tensors are never held together: at most the tensor being mixed from each side plus its
float32 temporaries are alive (about 10 bytes per element of the largest tensor for bf16 inputs). Two full
models are never loaded.

Determinism: the same inputs give byte-identical output files. The shard files are written by this module,
not by `safetensors.torch.save_file`, because that function writes a shard's metadata keys in a per-process
random order (checked with safetensors 0.8.0: six runs, six different files, once there are two or more
keys). Here the header is compact JSON with metadata keys sorted and tensors in name order, padded with
spaces to a multiple of 8 bytes, followed by the raw little-endian tensor bytes. This is the documented
safetensors layout and the library reads the files back; the tensors, not the byte layout, equal A1's at
`t == 0` (a library-written A1 file has its tensors in another order).
"""

from __future__ import annotations

import json
import math
import numbers
import os
import shutil
import sys
import uuid
from pathlib import Path

from safetensors import safe_open

SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"

# Bytes per element for the dtype names a safetensors header uses. A name not listed here is refused.
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8, "C64": 8,
}  # fmt: skip

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack", ".onnx", ".index.json")
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
        with safe_open(str(path), framework="pt") as handle:
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


def _few(names) -> str:
    names = sorted(names)
    shown = ", ".join(repr(n) for n in names[:_LISTED_NAMES])
    return shown + (f" and {len(names) - _LISTED_NAMES} more" if len(names) > _LISTED_NAMES else "")


def _nbytes(shape, dtype: str) -> int:
    return math.prod(shape) * _ITEMSIZE[dtype]


def _is_float(dtype: str) -> bool:
    return dtype.startswith(("F", "BF"))


def _check_compatible(a1: _Model, a0: _Model) -> None:
    only1, only0 = set(a1.tensors) - set(a0.tensors), set(a0.tensors) - set(a1.tensors)
    if only1 or only0:
        parts = []
        if only1:
            parts.append(f"only in A1: {_few(only1)}")
        if only0:
            parts.append(f"only in A0: {_few(only0)}")
        raise ValueError("A1 and A0 do not hold the same tensors (" + "; ".join(parts) + ")")
    shapes, dtypes = [], []
    for name in sorted(a1.tensors):
        (s1, d1), (s0, d0) = a1.tensors[name], a0.tensors[name]
        if s1 != s0:
            shapes.append(f"{name!r}: A1 {list(s1)} vs A0 {list(s0)}")
        if d1 != d0:
            dtypes.append(f"{name!r}: A1 {d1} vs A0 {d0}")
    if shapes:
        raise ValueError("shape mismatch between A1 and A0 for " + "; ".join(shapes[:_LISTED_NAMES]))
    if dtypes:
        raise ValueError("dtype mismatch between A1 and A0 (no silent cast) for " + "; ".join(dtypes[:_LISTED_NAMES]))


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


def _blend_tensor(name: str, a1, a0, t: float):
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
    names = sorted(n for n, f in a1.shard_of.items() if f == filename)
    other = _OneOpen(a0)
    try:
        with open(folder / filename, "wb") as out, safe_open(str(a1.root / filename), framework="pt") as mine:
            out.write(_shard_header(a1, filename, names))
            for name in names:
                shape, dtype = a1.tensors[name]
                if _is_float(dtype) and t == 0.0:
                    tensor = mine.get_tensor(name)
                elif _is_float(dtype) and t == 1.0:
                    tensor = other.get(name)
                elif _is_float(dtype):
                    tensor = _blend_tensor(name, mine.get_tensor(name), other.get(name), t)
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


def _copy_other_files(root: Path, folder: Path) -> None:
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or not path.is_file() or path.name.endswith(_WEIGHT_SUFFIXES):
            continue
        shutil.copyfile(path, folder / path.name)


def _fsync_all(folder: Path) -> None:
    for path in folder.iterdir():
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _check_t(t) -> float:
    if isinstance(t, bool) or not isinstance(t, numbers.Real):
        raise TypeError(f"t must be a real number, got {type(t).__name__}")
    t = float(t)
    if not math.isfinite(t) or not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be a finite number in [0, 1], got {t}")
    return t


def _check_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"out_dir {out_dir} exists and is not a directory")
        if any(out_dir.iterdir()):
            raise FileExistsError(f"out_dir {out_dir} is not empty; refusing to write into it")


def blend_state_dicts(a1_dir: Path, a0_dir: Path, t: float, out_dir: Path) -> None:
    """Write `(1 - t) * A1 + t * A0` to `out_dir`, shard by shard. See the module docstring for the rules."""
    if sys.byteorder != "little":  # pragma: no cover - the writer emits little-endian bytes from memory
        raise RuntimeError("msd.blend writes tensor bytes as they are in memory and supports little-endian hosts only")
    t = _check_t(t)
    out_dir = Path(out_dir)
    _check_out_dir(out_dir)
    _torch()
    a1, a0 = _read_model("A1", a1_dir), _read_model("A0", a0_dir)
    _check_compatible(a1, a0)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    folder = out_dir.parent / f".{out_dir.name}.{uuid.uuid4().hex[:8]}.partial"
    folder.mkdir()
    try:
        for filename in a1.files:
            _write_shard(a1, a0, t, filename, folder)
        if a1.sharded:
            _write_index(a1, folder)
        _copy_other_files(a1.root, folder)
        _fsync_all(folder)
        if out_dir.exists():  # an empty directory: a rename onto it works on POSIX, a removal first works everywhere
            out_dir.rmdir()
        os.replace(folder, out_dir)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
