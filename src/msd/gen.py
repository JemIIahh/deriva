"""Generation runner: run a model over prompts and store the completions, resumably.

`generate(prompts, backend, out_path, arm=..., alpha=...)` takes raw prompts, asks a `Backend` for
completions in batches and appends one `msd.schema.Generation` row per prompt to `out_path`. A later
task judges the rows; nothing here judges or scores.

Backend contract
    `Backend = Callable[[list[str]], list[GenOut]]`. It is called with at most `batch_size` RAW user
    prompt strings (not chat-templated: `hf_backend` applies the model's chat template itself, with a
    single user turn) and must return exactly one `GenOut` per prompt, in the same order. A wrong
    count, a non-`GenOut` item or a `finish_reason` outside `FINISH_REASONS` ("stop", "length",
    "error") is a `BackendError`; nothing from that batch is written. An empty `text` is allowed and
    kept (downstream treats it as no_response).

Files, resuming and crash safety
    * Rows are written in input order, a whole batch per write, then flushed and fsynced; a batch
      that fails (the backend raises, or the write does) leaves the file as it was before the batch,
      and the exception propagates. Earlier batches stay.
    * On start the existing file is read. Finished ids are skipped. The LAST line is allowed to be
      torn (a crash mid-write): a last line that is unterminated, not valid UTF-8 / JSON, or not a
      complete record is dropped, the file is truncated to the last good line, and its prompt is
      generated again. Anything else wrong is a loud `ResumeError` and the file is not touched: a
      bad or blank line in the middle, a duplicate id, a row of another arm or alpha (arms never
      share a file), or a row whose id is not among the prompts (a different test set).
    * A rerun after completion generates nothing and leaves the file byte-identical.
    * Result paths are built only by `gens_path`: `<out_root>/<split>/<arm>/<alpha|none>/<lang>.gens.jsonl`.
    * Single writer per output file. There is no lock: two runs appending to the same `out_path` at
      the same time duplicate rows, and the rollback of a failed write (a truncate to the size
      before the batch) could cut off rows the other run appended. Run one stage at a time.

Arm and alpha
    BASE, A0, A1 and A3 (Harden, a fine-tune with no blend setting) take no alpha; A2 (ReSta) and
    A2n (naive blend) require one. Alpha is stored rounded to 6 decimals ("canonical") so one
    setting always means one path and one value in the rows.

Decoding (`hf_backend`)
    Greedy: `do_sample=False`, `num_beams=1`. "Temperature 0" in the plan means exactly this; no
    temperature argument is passed because greedy ignores it. Batch composition (the padding of
    the batch) can change fp16 greedy outputs slightly, and output order is input order (no
    sort-by-length), so the batch size is part of the run metadata (`run_meta`) and must stay the
    same for a file that is resumed.

`import msd.gen` never imports torch or transformers; `hf_backend` imports them when called.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from msd.config import ARMS, LANGS
from msd.schema import SPLITS, Generation, Prompt

FINISH_REASONS: frozenset[str] = frozenset({"stop", "length", "error"})

# BASE, A0, A1 and A3 (Harden, a safety-preserving fine-tune) are single models with no blend / repair setting.
ALPHA_FORBIDDEN_ARMS: frozenset[str] = frozenset({"BASE", "A0", "A1", "A3"})
ALPHA_REQUIRED_ARMS: frozenset[str] = frozenset({"A2", "A2n"})  # ReSta alpha, naive blend value

DEFAULT_BATCH_SIZE = 32
ALPHA_DECIMALS = 6
FAKE_MARKER = "[FAKE BACKEND - canned text, not model output]"


class ResumeError(ValueError):
    """The existing output file cannot be resumed safely; it was left untouched."""


class BackendError(RuntimeError):
    """The backend broke its contract (wrong number of outputs, a bad item or finish_reason)."""


@dataclass(frozen=True)
class GenOut:
    """One completion: the decoded text (may be empty) and why generation ended."""

    text: str
    finish_reason: str


Backend = Callable[[list[str]], list[GenOut]]


# --- arm, alpha and paths -----------------------------------------------------------------------------


def canonical_alpha(alpha: float | None) -> float | None:
    """`alpha` rounded to 6 decimals (None stays None); NaN, infinity, negative values and non-numbers raise."""
    if alpha is None:
        return None
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a number or None, got {alpha!r}")
    value = float(alpha)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"alpha must be finite and not negative, got {alpha!r}")
    return round(value, ALPHA_DECIMALS) + 0.0  # + 0.0: no negative zero


def format_alpha(alpha: float | None) -> str:
    """The stable path component for an alpha: 'none', '0.25', '1.0', '0.3' (never exponent notation)."""
    value = canonical_alpha(alpha)
    if value is None:
        return "none"
    text = f"{value:.{ALPHA_DECIMALS}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def check_arm_alpha(arm: str, alpha: float | None) -> float | None:
    """Validate the arm and its alpha rule; return the canonical alpha. Raises ValueError otherwise."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    value = canonical_alpha(alpha)
    if arm in ALPHA_FORBIDDEN_ARMS and value is not None:
        raise ValueError(f"arm {arm} takes no alpha (got {alpha!r})")
    if arm in ALPHA_REQUIRED_ARMS and value is None:
        raise ValueError(f"arm {arm} needs an alpha (got none)")
    return value


def gens_path(out_root: str | Path, split: str, arm: str, alpha: float | None, lang: str) -> Path:
    """`<out_root>/<split>/<arm>/<alpha|none>/<lang>.gens.jsonl`; the only place that builds this path."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}; expected one of {list(LANGS)}")
    value = check_arm_alpha(arm, alpha)
    return Path(out_root) / split / arm / format_alpha(value) / f"{lang}.gens.jsonl"


_GENS_SUFFIX = ".gens.jsonl"
_META_SUFFIX = ".gens.meta.json"


def meta_path(gens_file: str | Path) -> Path:
    """The run-metadata file that sits next to a gens file: `<lang>.gens.jsonl` -> `<lang>.gens.meta.json`."""
    gens_file = Path(gens_file)
    if not gens_file.name.endswith(_GENS_SUFFIX):
        raise ValueError(f"{gens_file} is not a *{_GENS_SUFFIX} file")
    return gens_file.with_name(gens_file.name[: -len(_GENS_SUFFIX)] + _META_SUFFIX)


# --- the runner -----------------------------------------------------------------------------------------


def _check_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")


def _encode_rows(rows: Sequence[Generation]) -> bytes:
    """The bytes `msd.schema.write_jsonl` writes for these rows (equal by test): raw UTF-8, one line each."""
    return b"".join((json.dumps(asdict(row), ensure_ascii=False) + "\n").encode("utf-8") for row in rows)


_ROW_FIELDS = frozenset(f.name for f in fields(Generation))


def _parse_row(raw: bytes) -> Generation | None:
    """A complete Generation record from one line of bytes, or None if the line is not one."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != _ROW_FIELDS:
        return None
    alpha = obj["alpha"]
    if alpha is not None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
            return None
        alpha = float(alpha)
    if not (isinstance(obj["prompt_id"], str) and isinstance(obj["arm"], str) and isinstance(obj["text"], str)):
        return None
    if obj["finish_reason"] not in FINISH_REASONS:
        return None
    return Generation(obj["prompt_id"], obj["arm"], alpha, obj["text"], obj["finish_reason"])


@dataclass(frozen=True)
class ExistingGens:
    """What an existing gens file holds: finished ids, the byte length of its good part, its size."""

    done: frozenset[str]
    good_bytes: int
    size: int

    @property
    def torn_bytes(self) -> int:
        """Bytes after the last good line: a crash tail that a run would cut off."""
        return self.size - self.good_bytes


def scan_existing(prompts: Sequence[Prompt], path: str | Path, *, arm: str, alpha: float | None) -> ExistingGens:
    """Read an existing gens file without touching it; the checks `generate` makes before it writes.

    Only a torn LAST line is forgiven (the good part then ends before it). Everything else that is
    wrong raises ResumeError: a bad or blank middle line, a duplicate id, a row of another arm or
    alpha, a row whose id is not among `prompts`. Duplicate prompt ids raise ValueError, as does an
    arm / alpha pair that `check_arm_alpha` refuses. A missing file is an empty one. The CLI calls
    this for every language before the first generation, so a bad file is found before GPU time is spent.
    """
    value = check_arm_alpha(arm, alpha)
    repeated = sorted(pid for pid, n in Counter(p.id for p in prompts).items() if n > 1)
    if repeated:
        raise ValueError(f"duplicate prompt ids in the input: {repeated[:5]}")
    valid_ids = {p.id for p in prompts}
    path = Path(path)
    if not path.exists():
        return ExistingGens(frozenset(), 0, 0)
    data = path.read_bytes()
    done: set[str] = set()
    pos = good = lineno = 0
    while pos < len(data):
        lineno += 1
        newline = data.find(b"\n", pos)
        row = _parse_row(data[pos:newline]) if newline != -1 else None  # an unterminated line is always torn
        if row is None:
            if newline == -1 or newline + 1 == len(data):
                break  # the last line, torn by a crash: dropped by `generate`
            raise ResumeError(
                f"{path}: line {lineno} is not a valid generation record (and it is not the last line, so this "
                "is not a crash tail). The file was left untouched; fix or move it"
            )
        if row.arm != arm or canonical_alpha(row.alpha) != value:
            raise ResumeError(
                f"{path}: line {lineno} holds a row for arm {row.arm!r} alpha {row.alpha!r}, but this run is arm "
                f"{arm!r} alpha {value!r}; arms and alphas never share a file"
            )
        if row.prompt_id in done:
            raise ResumeError(f"{path}: line {lineno}: duplicate prompt id {row.prompt_id!r}")
        if row.prompt_id not in valid_ids:
            raise ResumeError(
                f"{path}: line {lineno}: prompt id {row.prompt_id!r} is not among this run's prompts "
                "(a different test set, or a smaller --limit than the file was made with)"
            )
        done.add(row.prompt_id)
        pos = good = newline + 1
    return ExistingGens(frozenset(done), good, len(data))


def _truncate(path: Path, size: int) -> None:
    with path.open("r+b") as fh:
        fh.truncate(size)
        fh.flush()
        os.fsync(fh.fileno())


def _append_durable(path: Path, blob: bytes) -> None:
    """Append `blob` (whole lines) and fsync it; on any failure put the file back as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    size_before = path.stat().st_size if existed else 0
    try:
        with path.open("ab") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            if existed:
                _truncate(path, size_before)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass  # best effort; the original error is what matters
        raise


def _call_backend(backend: Backend, texts: list[str]) -> list[GenOut]:
    result = backend(list(texts))
    if not isinstance(result, (list, tuple)):
        raise BackendError(f"backend must return a list of GenOut, got {type(result).__name__}")
    if len(result) != len(texts):
        raise BackendError(f"backend returned {len(result)} outputs for {len(texts)} prompts; need exactly one each")
    for i, out in enumerate(result):
        if not isinstance(out, GenOut):
            raise BackendError(f"backend output {i} is a {type(out).__name__}, not a GenOut")
        if not isinstance(out.text, str):
            raise BackendError(f"backend output {i} has text of type {type(out.text).__name__}, not str")
        if out.finish_reason not in FINISH_REASONS:
            raise BackendError(
                f"backend output {i} has finish_reason {out.finish_reason!r}; expected one of {sorted(FINISH_REASONS)}"
            )
    return list(result)


def generate(
    prompts: Sequence[Prompt],
    backend: Backend,
    out_path: str | Path,
    *,
    arm: str,
    alpha: float | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Generate completions for the unfinished `prompts` and append them to `out_path`; return how many.

    See the module docstring for the full contract. `progress(done, total)` is called after each
    persisted batch with the counts of this run. Raises ValueError for bad arguments or duplicate
    prompt ids, ResumeError for an existing file that cannot be resumed, BackendError for a backend
    that breaks its contract; anything the backend raises propagates after earlier batches are saved.
    """
    value = check_arm_alpha(arm, alpha)
    _check_batch_size(batch_size)
    prompts = list(prompts)
    out_path = Path(out_path)

    existing = scan_existing(prompts, out_path, arm=arm, alpha=value)
    done = existing.done
    if existing.torn_bytes:
        _truncate(out_path, existing.good_bytes)  # drop the torn last line (the scan found nothing else wrong)
        print(
            f"msd.gen: dropped {existing.torn_bytes} torn byte(s) from the end of {out_path.name} (a crash tail); "
            "its prompt is generated again",
            file=sys.stderr,
        )

    todo = [p for p in prompts if p.id not in done]
    written = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        outs = _call_backend(backend, [p.text for p in batch])
        rows = [Generation(p.id, arm, value, o.text, o.finish_reason) for p, o in zip(batch, outs)]
        _append_durable(out_path, _encode_rows(rows))
        written += len(rows)
        if progress is not None:
            progress(written, len(todo))
    return written


# --- backends ---------------------------------------------------------------------------------------------


def fake_backend() -> Backend:
    """A backend for plumbing tests: clearly marked canned text, distinct per prompt, never echoing it."""

    def run(prompts: list[str]) -> list[GenOut]:
        return [
            GenOut(f"{FAKE_MARKER} {hashlib.sha256(p.encode('utf-8')).hexdigest()[:8]}", "stop") for p in prompts
        ]

    return run


_TORCH_DTYPES: dict[str, str] = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


def _dtype_keyword(transformers: Any) -> str:
    """`dtype` for transformers >= 4.56 (and for an unreadable version); `torch_dtype` for older ones."""
    match = re.match(r"(\d+)\.(\d+)", str(getattr(transformers, "__version__", "")))
    return "torch_dtype" if match and (int(match[1]), int(match[2])) < (4, 56) else "dtype"


def _eos_ids(tokenizer: Any, model: Any) -> list[int]:
    ids: set[int] = set()
    for value in (getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                  getattr(tokenizer, "eos_token_id", None)):
        if value is None:
            continue
        ids.update(int(i) for i in (value if isinstance(value, (list, tuple, set)) else [value]))
    return sorted(ids)


def hf_backend(model_path: str, *, max_new_tokens: int = 256, dtype: str = "float16") -> Backend:
    """A greedy Hugging Face `transformers` backend; loads the model now, imports torch now (lazily).

    `model_path` is a Hugging Face repo id or a local directory (for example a merged safetensors
    directory). The backend takes RAW user prompts and applies the tokenizer's chat template itself
    (one user turn, `add_generation_prompt=True`), then tokenises the batch left-padded without
    adding special tokens again (the template already carries the BOS). Decoding is greedy
    (`do_sample=False`, `num_beams=1`; no temperature is passed: "temperature 0" means this).
    `dtype` is "float16" (default: a T4 has no native bf16), "bfloat16" or "float32".

    Per prompt, `finish_reason` is "length" when no end-of-sequence token appears among the new
    tokens and the new tokens fill `max_new_tokens`, else "stop". The prompt tokens are cut off, the
    rest is decoded with `skip_special_tokens=True`. Rows come back in input order (no sorting by
    length); because padding depends on batch composition, fp16 greedy outputs can differ slightly
    between batch sizes, so record the batch size (see `run_meta`). Runs on CUDA when available,
    else on the CPU (float16 on the CPU prints one warning on stderr). A gated download takes its
    token from the HF_TOKEN environment variable, read by transformers itself; nothing here reads,
    prints or stores it.
    """
    if dtype not in _TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(_TORCH_DTYPES)}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens!r}")
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "hf_backend needs the 'torch' and 'transformers' packages, which are not installed here; run the "
            "generation on the GPU runtime (Kaggle or Colab, see docs/runtime.md)"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype == "float16":
        print(
            "msd.gen: warning: fp16 on CPU is slow or unsupported; use --dtype float32 for CPU smoke runs",
            file=sys.stderr,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"  # generation continues from the right edge, so pad on the left
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"the tokenizer of {model_path} has neither a pad token nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, **{_dtype_keyword(transformers): getattr(torch, _TORCH_DTYPES[dtype])}
    )
    model = model.to(device)
    model.eval()
    eos_ids = _eos_ids(tokenizer, model)

    def run(prompts: list[str]) -> list[GenOut]:
        if not prompts:
            return []
        templated = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
        batch = tokenizer(templated, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        input_ids = batch["input_ids"]
        options: dict[str, Any] = {"eos_token_id": eos_ids} if eos_ids else {}
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **options,
            )
        new_tokens = sequences[:, input_ids.shape[1] :]  # left padding: every prompt ends at the same column
        rows = new_tokens.tolist()
        texts = tokenizer.batch_decode(rows, skip_special_tokens=True)
        if len(rows) != len(prompts) or len(texts) != len(prompts):
            raise BackendError(f"model returned {len(rows)} rows for {len(prompts)} prompts")
        full = len(rows[0]) >= max_new_tokens if rows else False
        stops = set(eos_ids)
        return [
            GenOut(text, "length" if full and not stops.intersection(row) else "stop")
            for text, row in zip(texts, rows)
        ]

    return run


# --- run metadata -----------------------------------------------------------------------------------------

_TOKEN_SHAPED = re.compile(r"hf_[A-Za-z0-9]{20,}")
_SECRET_ENV_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|API_?KEY|KEY", re.IGNORECASE)
_MIN_SECRET_LEN = 8


def _redact(text: str) -> str:
    """Replace anything token-shaped, and the value of any secret-looking environment variable, with ***."""
    text = _TOKEN_SHAPED.sub("***", text)
    for name, value in os.environ.items():
        if len(value) >= _MIN_SECRET_LEN and _SECRET_ENV_NAME.search(name):
            text = text.replace(value, "***")
    return text


def git_commit(cwd: str | Path | None = None) -> str | None:
    """The full hash of HEAD in `cwd` (default: the current directory), or None for any failure."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except Exception:
        return None
    text = done.stdout.strip()
    return text if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", text) else None


def msd_version() -> str:
    try:
        return str(importlib.metadata.version("msd"))
    except Exception:
        return "unknown"


def run_meta(
    *,
    model_path: str,
    arm: str,
    alpha: float | None,
    split: str,
    lang: str,
    batch_size: int,
    max_new_tokens: int,
    dtype: str,
    backend: str,
    n_rows: int,
    repo_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """What a gens file was made with, as a JSON-safe dict, for the `<lang>.gens.meta.json` file.

    Holds only the explicit fields below plus the msd version, the git commit of `repo_root` (best
    effort, None if unavailable) and a UTC timestamp. No environment variable is ever stored. The
    environment is read for one purpose only: to find secret-looking values (a token-shaped string,
    or the value of a variable whose name contains TOKEN, SECRET, PASSWORD or KEY) so that they can
    be replaced by *** in the strings that are stored.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "model_path": _redact(str(model_path)),
        "arm": arm,
        "alpha": canonical_alpha(alpha),
        "split": split,
        "lang": lang,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "dtype": _redact(str(dtype)),
        "backend": _redact(str(backend)),
        "n_rows": n_rows,
        "msd_version": msd_version(),
        "git_commit": git_commit(repo_root),
        "timestamp_utc": stamp,
    }
