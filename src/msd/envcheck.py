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
as a bearer header when one exists. Redirects are not followed, so the token never travels to
another host; the Location header of a redirect is only read. `result` is one of:
  ok            2xx; or a 3xx whose Location is HF's resolve-cache of the same repo
                (`/api/resolve-cache/models/<repo>/...` on huggingface.co), which is HF's normal
                answer for a reachable file. Seen without a token on 2026-11-05 for public repos
                (openai-community/gpt2 -> 307 to its resolve-cache, gpt2 -> 200).
  unexpected_redirect  any other 3xx (a renamed repo, another host, no Location): never "ok", so it
                cannot be mistaken for "licence accepted"
  unauthorized  401: no token, a bad or expired token, or a repo Hugging Face hides. Seen without
                a token on 2026-11-05 for both Tiny Aya repos, which are gated. With a token
                present it can also mean the licence is not accepted yet: check both.
  forbidden     403: the token works but access is not granted. Expected when the licence is not
                accepted or a fine-grained token lacks the repo; NOT yet seen with a real token, so
                treat that reading as UNVERIFIED (as docs/runtime.md does)
  not_found     404
  http_error    any other status
  timeout       no answer within the short timeout
  error         DNS, connection or other failure; `error` holds the text
  skipped_offline  --offline

A token that is not plain printable ASCII without spaces (for example an internal newline from a
multi-line quoted `.env` value) is treated as absent: nothing is sent and `hf_token_present` is false.
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
import urllib.parse
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
_TOKEN_OK = re.compile(r"[\x21-\x7e]+")  # printable ASCII, no whitespace: safe as an HTTP header value
_TOKEN_JUNK = re.compile(r"[^\x21-\x7e]+")
MIN_SCRUB_FRAGMENT = 6


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


def _clean_token(value: Any) -> str | None:
    """The stripped token if it is plain printable ASCII without spaces, else None."""
    token = value.strip() if isinstance(value, str) else ""
    return token if token and _TOKEN_OK.fullmatch(token) else None


def find_hf_token(environ: Mapping[str, str] | None = None, env_file: Path | str | None = None) -> str | None:
    """HF_TOKEN from the environment, else from `env_file` (python-dotenv). None if absent, blank,
    unreadable or malformed (whitespace or control characters inside); never raises."""
    env = os.environ if environ is None else environ
    token = _clean_token(env.get("HF_TOKEN"))
    if token:
        return token
    if env_file is not None:
        try:
            if Path(env_file).is_file():
                from dotenv import dotenv_values

                return _clean_token(dotenv_values(env_file).get("HF_TOKEN"))
        except Exception:
            pass
    return None


def _scrub(text: str, token: str | None) -> str:
    """Remove the token, every printable fragment of it, and anything token-shaped, from error text."""
    if token:
        text = text.replace(token, "***")
        for fragment in _TOKEN_JUNK.split(token):
            if len(fragment) >= MIN_SCRUB_FRAGMENT:
                text = text.replace(fragment, "***")
    text = re.sub(r"hf_[A-Za-z0-9]{20,}", "***", text)  # any token-shaped string, ours or not
    return text[:ERROR_TEXT_LIMIT]


def _is_resolve_cache_of(location: Any, repo_id: str) -> bool:
    """True if a redirect Location is HF's resolve-cache of this same repo, on huggingface.co."""
    if not location:
        return False
    parts = urllib.parse.urlsplit(str(location))
    if parts.scheme not in ("", "https") or parts.netloc.lower() not in ("", "huggingface.co"):
        return False
    return parts.path.lower().startswith(f"/api/resolve-cache/models/{repo_id}/".lower())


def _status_result(status: int, location: Any = None, repo_id: str = "") -> str:
    if 200 <= status < 300:
        return "ok"
    if 300 <= status < 400:
        return "ok" if _is_resolve_cache_of(location, repo_id) else "unexpected_redirect"
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
    secret = token  # kept for scrubbing even if it turns out unusable
    if token and not _TOKEN_OK.fullmatch(token):
        token = None  # never send a malformed token
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
        status = int(exc.code)
        location = _attempt(lambda: exc.headers.get("Location"))
        _attempt(exc.close)
        return {"status": status, "result": _status_result(status, location, repo_id), "error": None}
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return {"status": None, "result": "timeout", "error": f"timed out after {timeout:g}s"}
        return {"status": None, "result": "error", "error": _scrub(f"URLError: {exc.reason}", secret)}
    except TimeoutError:
        return {"status": None, "result": "timeout", "error": f"timed out after {timeout:g}s"}
    except Exception as exc:
        return {"status": None, "result": "error", "error": _scrub(f"{type(exc).__name__}: {exc}", secret)}


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
