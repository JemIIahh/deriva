"""Environment check for the machine we run on (laptop, Kaggle, Colab).

`collect()` builds one flat, JSON-safe dict; `main()` prints it as JSON and returns it. Nothing in
here may crash: every probe that can fail (a missing package, a broken torch, no nvidia-smi, no
network) turns into `None` or an error string in the report.

What the report holds:
  python, platform                 interpreter version and OS string
  gpu_name, gpu_vram_mib, gpu_count  first GPU (torch if it sees one, else nvidia-smi), else None
  torch, torch_cuda                torch version and the CUDA version it was built for, else None
  packages                         version string or None for each toolkit we may need
  hf_token_present                 a boolean only. The token itself is never stored or printed.
  offline                          True when --offline skipped every network call
  hf_access                        HEAD check per Tiny Aya repo: {"status", "result", "error"}

HEAD check. We send `HEAD https://huggingface.co/<repo>/resolve/main/config.json`, with the token
as a bearer header when one exists. Redirects are not followed (a redirect to the download CDN is
"ok", and the token must not travel to another host). `result` is one of:
  ok            2xx/3xx: the file is reachable with this token (licence accepted)
  unauthorized  401: no token, a bad token, or a repo Hugging Face hides. Seen without a token
                on 2026-11-05 for both Tiny Aya repos, which are gated.
  forbidden     403: the token works but access is not granted (licence not accepted, or a
                fine-grained token without access to the repo)
  not_found     404
  http_error    any other status
  timeout       no answer within the short timeout
  error         DNS, connection or other failure; `error` holds the text
  skipped_offline  --offline
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from msd.config import MODEL_IDS

PACKAGES: tuple[str, ...] = (
    "aligntune",
    "safetune",
    "auditkit",
    "transformers",
    "bitsandbytes",
    "vllm",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "safetensors",
)

HF_HOST = "https://huggingface.co"
PROBE_FILE = "config.json"  # small, not stored in LFS, present in every model repo
HEAD_TIMEOUT = 8.0  # seconds
ERROR_TEXT_LIMIT = 200
NVIDIA_SMI_TIMEOUT = 10

_SMI_LINE = re.compile(r"^(?P<name>[^,]+),\s*(?P<mib>\d+)\s*MiB$")


# --- packages -----------------------------------------------------------------------------------


def package_version(name: str) -> str | None:
    """Installed version of a distribution, or None for ANY failure (missing, broken metadata)."""
    try:
        return str(importlib.metadata.version(name))
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    return {name: package_version(name) for name in PACKAGES}


# --- torch and GPU ------------------------------------------------------------------------------


def _attempt(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _torch_gpus(torch: Any) -> list[tuple[str, int]]:
    if not torch.cuda.is_available():
        return []
    gpus = []
    for index in range(int(torch.cuda.device_count())):
        name = str(torch.cuda.get_device_name(index))
        total_bytes = int(torch.cuda.get_device_properties(index).total_memory)
        gpus.append((name, total_bytes // (1024 * 1024)))
    return gpus


def torch_report() -> tuple[str | None, str | None, list[tuple[str, int]]]:
    """(torch version, CUDA version torch was built for, [(gpu name, vram MiB)]); None/[] on any failure."""
    try:
        import torch
    except Exception:  # not installed, or its import blows up
        return None, None, []
    version = _attempt(lambda: str(torch.__version__))
    cuda_version = _attempt(lambda: None if torch.version.cuda is None else str(torch.version.cuda))
    gpus = _attempt(lambda: _torch_gpus(torch), default=[])
    return version, cuda_version, gpus


def query_nvidia_smi() -> str | None:
    """Raw `nvidia-smi` CSV text, or None if the tool is absent, fails or hangs."""
    try:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else None
    except Exception:
        return None


def parse_nvidia_smi(text: Any) -> list[tuple[str, int]]:
    """Parse `name, 15360 MiB` lines. Lines that do not match are dropped; never raises."""
    if not isinstance(text, str):
        return []
    gpus = []
    for line in text.splitlines():
        match = _SMI_LINE.match(line.strip())
        if match and match["name"].strip():
            gpus.append((match["name"].strip(), int(match["mib"])))
    return gpus


# --- Hugging Face access ------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a redirect as an HTTPError instead of following it (and sending the token along)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _default_open(request: urllib.request.Request, timeout: float | None = None) -> Any:
    return _OPENER.open(request, timeout=timeout)


def find_hf_token(environ: Mapping[str, str] | None = None, env_file: Path | str | None = None) -> str | None:
    """HF_TOKEN from the environment, else from `env_file` (python-dotenv). None if absent or blank."""
    env = os.environ if environ is None else environ
    token = (env.get("HF_TOKEN") or "").strip()
    if token:
        return token
    if env_file is not None and Path(env_file).is_file():
        try:
            from dotenv import dotenv_values

            token = (dotenv_values(env_file).get("HF_TOKEN") or "").strip()
        except Exception:
            token = ""
    return token or None


def _scrub(text: str, token: str | None) -> str:
    if token:
        text = text.replace(token, "***")
    text = re.sub(r"hf_[A-Za-z0-9]{20,}", "***", text)  # any token-shaped string, ours or not
    return text[:ERROR_TEXT_LIMIT]


def _status_result(status: int) -> str:
    if 200 <= status < 400:
        return "ok"
    return {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(status, "http_error")


def hf_head(
    repo_id: str,
    token: str | None = None,
    *,
    opener: Callable[..., Any] | None = None,
    timeout: float = HEAD_TIMEOUT,
) -> dict[str, Any]:
    """HEAD the probe file of a repo. Returns {"status", "result", "error"}; never raises."""
    open_url = opener or _default_open
    try:
        request = urllib.request.Request(
            f"{HF_HOST}/{repo_id}/resolve/main/{PROBE_FILE}", method="HEAD"
        )
        request.add_header("User-Agent", "msd-env-check")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        response = open_url(request, timeout=timeout)
        status = int(response.status)
        _attempt(response.close)
        return {"status": status, "result": _status_result(status), "error": None}
    except urllib.error.HTTPError as exc:
        _attempt(exc.close)
        return {"status": int(exc.code), "result": _status_result(int(exc.code)), "error": None}
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return {"status": None, "result": "timeout", "error": f"timed out after {timeout:g}s"}
        return {"status": None, "result": "error", "error": _scrub(f"URLError: {exc.reason}", token)}
    except TimeoutError:
        return {"status": None, "result": "timeout", "error": f"timed out after {timeout:g}s"}
    except Exception as exc:
        return {"status": None, "result": "error", "error": _scrub(f"{type(exc).__name__}: {exc}", token)}


def tiny_aya_repos() -> list[str]:
    """Repo ids of the two Tiny Aya models we need (aligned Global, and Base), from config."""
    return [MODEL_IDS[arm] for arm in ("A0", "BASE") if MODEL_IDS.get(arm)]


# --- the report ---------------------------------------------------------------------------------


def collect(
    *,
    offline: bool = False,
    environ: Mapping[str, str] | None = None,
    env_file: Path | str | None = None,
    opener: Callable[..., Any] | None = None,
    timeout: float = HEAD_TIMEOUT,
) -> dict[str, Any]:
    token = find_hf_token(environ, env_file)
    torch_version, torch_cuda, gpus = torch_report()
    if not gpus:
        gpus = parse_nvidia_smi(query_nvidia_smi())
    hf_access = {}
    for repo in tiny_aya_repos():
        if offline:
            hf_access[repo] = {"status": None, "result": "skipped_offline", "error": None}
        else:
            hf_access[repo] = hf_head(repo, token, opener=opener, timeout=timeout)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_name": gpus[0][0] if gpus else None,
        "gpu_vram_mib": gpus[0][1] if gpus else None,
        "gpu_count": len(gpus) if gpus else None,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "packages": package_versions(),
        "hf_token_present": token is not None,
        "offline": bool(offline),
        "hf_access": hf_access,
    }


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | str | None = None,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="env_check", description="Report the runtime environment as JSON.")
    ap.add_argument("--offline", action="store_true", help="make no network call (skip the Hugging Face HEAD checks)")
    ap.add_argument("--timeout", type=float, default=HEAD_TIMEOUT, help="seconds per Hugging Face HEAD check")
    args = ap.parse_args(argv)
    report = collect(
        offline=args.offline, environ=environ, env_file=env_file, opener=opener, timeout=args.timeout
    )
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main(env_file=Path.cwd() / ".env")
