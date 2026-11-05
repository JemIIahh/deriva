"""Environment check (msd.envcheck + scripts/env_check.py): no real network, no real GPU."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import itertools
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from msd import envcheck
from msd.config import MODEL_IDS

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "env_check.py"

SECRET = "hf_TESTSECRETTOKEN0123456789abcdefghij"
GLOBAL_ID = "CohereLabs/tiny-aya-global"
BASE_ID = "CohereLabs/tiny-aya-base"

REQUIRED_KEYS = {
    "python",
    "platform",
    "gpu_name",
    "gpu_vram_mib",
    "gpu_count",
    "torch",
    "torch_cuda",
    "packages",
    "hf_token_present",
    "offline",
    "hf_access",
}
PACKAGES = (
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


# --- fakes --------------------------------------------------------------------------------------


def fake_torch(*, cuda=True, names=("Tesla T4",), total_mib=15095, version="2.9.0+cu126", cuda_version="12.6"):
    """A stand-in for the torch module with only the attributes envcheck reads."""
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda_version if cuda else None),
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: len(names) if cuda else 0,
            get_device_name=lambda i: names[i],
            get_device_properties=lambda i: SimpleNamespace(total_memory=total_mib * 1024 * 1024),
        ),
    )


class FakeResponse:
    def __init__(self, status):
        self.status = status
        self.closed = False

    def close(self):
        self.closed = True


class FakeOpener:
    """Records every request; answers with `plan` (an int status, or an exception to raise)."""

    def __init__(self, plan=200):
        self.plan = plan
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append((request, timeout))
        plan = self.plan[request.full_url] if isinstance(self.plan, dict) else self.plan
        if isinstance(plan, BaseException):
            raise plan
        if plan >= 300:  # urllib raises HTTPError for every non-2xx answer (redirects are not followed)
            raise urllib.error.HTTPError(request.full_url, plan, "msg", {}, None)
        return FakeResponse(plan)


def forbidden_opener(request, timeout=None):
    raise AssertionError("a network call was made")


@pytest.fixture
def hermetic(monkeypatch):
    """No torch and no nvidia-smi, whatever machine the tests run on."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: None)


# --- the brief's required test ------------------------------------------------------------------


def test_env_check_reports_missing_package_as_null(monkeypatch, hermetic):
    real_version = importlib.metadata.version

    def flaky_version(name):
        if name == "safetune":
            raise ImportError("simulated import failure for safetune")
        return real_version("pytest")  # any installed distribution: a version string

    monkeypatch.setattr(importlib.metadata, "version", flaky_version)
    report = envcheck.collect(offline=True, environ={})
    assert report["packages"]["safetune"] is None
    assert isinstance(report["packages"]["auditkit"], str)  # others still reported


def test_every_package_check_failure_yields_null_not_a_crash(monkeypatch, hermetic):
    def broken(name):
        raise RuntimeError("anything at all")

    monkeypatch.setattr(importlib.metadata, "version", broken)
    report = envcheck.collect(offline=True, environ={})
    assert set(report["packages"]) == set(PACKAGES)
    assert all(v is None for v in report["packages"].values())


def test_package_not_installed_is_null(monkeypatch, hermetic):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert envcheck.package_versions()["vllm"] is None


def test_package_list_is_exactly_the_documented_one():
    assert tuple(envcheck.PACKAGES) == PACKAGES


# --- GPU and torch ------------------------------------------------------------------------------


def test_gpu_absent_reports_nulls(hermetic):
    report = envcheck.collect(offline=True, environ={})
    assert report["gpu_name"] is None
    assert report["gpu_vram_mib"] is None
    assert report["gpu_count"] is None
    assert report["torch"] is None
    assert report["torch_cuda"] is None


def test_gpu_from_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(names=("Tesla T4", "Tesla T4")))
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: None)
    report = envcheck.collect(offline=True, environ={})
    assert report["gpu_name"] == "Tesla T4"
    assert report["gpu_vram_mib"] == 15095
    assert report["gpu_count"] == 2
    assert report["torch"] == "2.9.0+cu126"
    assert report["torch_cuda"] == "12.6"


def test_cpu_only_torch_with_no_nvidia_smi(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=False, version="2.9.0"))
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: None)
    report = envcheck.collect(offline=True, environ={})
    assert report["torch"] == "2.9.0"
    assert report["torch_cuda"] is None
    assert report["gpu_name"] is None and report["gpu_vram_mib"] is None


def test_cpu_torch_but_nvidia_smi_sees_a_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=False, version="2.9.0"))
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: "NVIDIA L4, 23034 MiB\n")
    report = envcheck.collect(offline=True, environ={})
    assert report["gpu_name"] == "NVIDIA L4"
    assert report["gpu_vram_mib"] == 23034
    assert report["gpu_count"] == 1
    assert report["torch"] == "2.9.0" and report["torch_cuda"] is None


def test_no_torch_but_nvidia_smi(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: "Tesla T4, 15360 MiB\nTesla T4, 15360 MiB\n")
    report = envcheck.collect(offline=True, environ={})
    assert (report["gpu_name"], report["gpu_vram_mib"], report["gpu_count"]) == ("Tesla T4", 15360, 2)
    assert report["torch"] is None


def test_torch_that_blows_up_does_not_crash(monkeypatch):
    class Angry:
        __version__ = "9.9"

        def __getattr__(self, name):
            raise RuntimeError("torch is broken")

    monkeypatch.setitem(sys.modules, "torch", Angry())
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: None)
    report = envcheck.collect(offline=True, environ={})
    assert report["gpu_name"] is None
    json.dumps(report)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tesla T4, 15360 MiB", [("Tesla T4", 15360)]),
        ("Tesla T4, 15360 MiB\n", [("Tesla T4", 15360)]),
        ("NVIDIA A100-SXM4-40GB, 40960 MiB\r\n", [("NVIDIA A100-SXM4-40GB", 40960)]),
        ("Tesla T4, 15360 MiB\nTesla T4, 15360 MiB\n", [("Tesla T4", 15360), ("Tesla T4", 15360)]),
        ("\n  NVIDIA L4 ,  23034 MiB  \n\n", [("NVIDIA L4", 23034)]),
        ("", []),
        ("   \n", []),
        ("garbage", []),
        ("Tesla T4", []),
        ("Tesla T4, lots", []),
        ("Tesla T4, 15360", []),
        ("[N/A], [N/A]", []),
        ("N/A, N/A", []),
        (", 15360 MiB", []),
        ("Tesla T4, -5 MiB", []),
        ("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", []),
        ("bad line\nTesla T4, 15360 MiB", [("Tesla T4", 15360)]),
    ],
)
def test_parse_nvidia_smi(text, expected):
    assert envcheck.parse_nvidia_smi(text) == expected


def test_parse_nvidia_smi_tolerates_non_string():
    assert envcheck.parse_nvidia_smi(None) == []


def test_query_nvidia_smi_returns_none_when_the_tool_is_missing(monkeypatch):
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: None)
    monkeypatch.setattr(envcheck.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert envcheck.query_nvidia_smi() is None


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "oserror"])
def test_query_nvidia_smi_swallows_failures(monkeypatch, failure):
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def run(cmd, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        if failure == "oserror":
            raise OSError("cannot exec")
        return SimpleNamespace(returncode=9, stdout="Tesla T4, 15360 MiB")

    monkeypatch.setattr(envcheck.subprocess, "run", run)
    assert envcheck.query_nvidia_smi() is None


def test_query_nvidia_smi_returns_stdout_and_uses_the_documented_query(monkeypatch):
    seen = {}
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="Tesla T4, 15360 MiB\n")

    monkeypatch.setattr(envcheck.subprocess, "run", run)
    assert envcheck.query_nvidia_smi() == "Tesla T4, 15360 MiB\n"
    assert "--query-gpu=name,memory.total" in seen["cmd"]
    assert "--format=csv,noheader" in seen["cmd"]


# --- HEAD access check --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, result",
    [
        (200, "ok"),
        (302, "ok"),
        (401, "unauthorized"),
        (403, "forbidden"),
        (404, "not_found"),
        (500, "http_error"),
        (429, "http_error"),
    ],
)
def test_hf_head_maps_status_codes(status, result):
    out = envcheck.hf_head(GLOBAL_ID, "tok", opener=FakeOpener(status))
    assert out["status"] == status
    assert out["result"] == result
    assert out["error"] is None


@pytest.mark.parametrize(
    "exc, result",
    [
        (TimeoutError("timed out"), "timeout"),
        (urllib.error.URLError(TimeoutError("timed out")), "timeout"),
        (urllib.error.URLError("[Errno 8] nodename nor servname provided"), "error"),
        (ConnectionResetError("reset"), "error"),
        (ValueError("anything"), "error"),
    ],
)
def test_hf_head_never_raises_and_records_an_error_string(exc, result):
    out = envcheck.hf_head(GLOBAL_ID, None, opener=FakeOpener(exc))
    assert out["status"] is None
    assert out["result"] == result
    assert isinstance(out["error"], str) and out["error"]


def test_hf_head_sends_a_head_request_for_the_gated_file_with_a_short_timeout():
    opener = FakeOpener(200)
    envcheck.hf_head(BASE_ID, SECRET, opener=opener)
    ((request, timeout),) = opener.calls
    assert request.get_method() == "HEAD"
    assert request.full_url == f"https://huggingface.co/{BASE_ID}/resolve/main/config.json"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert timeout is not None and timeout <= 30


def test_hf_head_without_a_token_sends_no_authorization_header():
    opener = FakeOpener(401)
    envcheck.hf_head(BASE_ID, None, opener=opener)
    assert opener.calls[0][0].get_header("Authorization") is None


def test_hf_head_scrubs_the_token_out_of_error_strings():
    out = envcheck.hf_head(GLOBAL_ID, SECRET, opener=FakeOpener(ValueError(f"bad header Bearer {SECRET}")))
    assert SECRET not in json.dumps(out)
    assert out["result"] == "error"


def test_default_opener_does_not_follow_redirects():
    handler = envcheck._NoRedirect()
    request = urllib.request.Request("https://huggingface.co/x", method="HEAD")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/x") is None


# --- token lookup -------------------------------------------------------------------------------


def test_token_from_environment():
    assert envcheck.find_hf_token(environ={"HF_TOKEN": f" {SECRET} "}) == SECRET


def test_token_falls_back_to_env_file_and_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=\nHF_TOKEN=hf_FROMFILE0123456789abcdefghijklmn\n", encoding="utf-8")
    assert envcheck.find_hf_token(environ={}, env_file=env_file) == "hf_FROMFILE0123456789abcdefghijklmn"
    assert envcheck.find_hf_token(environ={"HF_TOKEN": SECRET}, env_file=env_file) == SECRET


def test_token_absent_or_blank_is_none(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=\n", encoding="utf-8")
    assert envcheck.find_hf_token(environ={}, env_file=env_file) is None
    assert envcheck.find_hf_token(environ={"HF_TOKEN": "  "}, env_file=env_file) is None
    assert envcheck.find_hf_token(environ={}, env_file=tmp_path / "missing.env") is None
    assert envcheck.find_hf_token(environ={}) is None


# --- the report ---------------------------------------------------------------------------------


def test_report_has_the_documented_keys_and_checks_both_tiny_aya_repos(hermetic):
    opener = FakeOpener(200)
    report = envcheck.collect(environ={}, opener=opener)
    assert set(report) == REQUIRED_KEYS
    assert set(report["hf_access"]) == {GLOBAL_ID, BASE_ID}
    assert report["python"] == platform.python_version()
    assert report["offline"] is False
    assert len(opener.calls) == 2


def test_tiny_aya_repos_come_from_config():
    assert envcheck.tiny_aya_repos() == [MODEL_IDS["A0"], MODEL_IDS["BASE"]]


def test_offline_makes_no_network_call_at_all(monkeypatch, hermetic):
    def trip(*args, **kwargs):
        raise AssertionError("a network call was made")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", trip)
    monkeypatch.setattr(urllib.request, "urlopen", trip)
    report = envcheck.collect(offline=True, environ={"HF_TOKEN": SECRET}, opener=forbidden_opener)
    assert report["offline"] is True
    for repo in (GLOBAL_ID, BASE_ID):
        assert report["hf_access"][repo] == {"status": None, "result": "skipped_offline", "error": None}
    # and the default opener path is off too
    report = envcheck.collect(offline=True, environ={}, opener=None)
    assert report["hf_access"][GLOBAL_ID]["result"] == "skipped_offline"


def test_main_offline_flag_skips_the_network(monkeypatch, capsys, hermetic):
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden_opener)
    result = envcheck.main(["--offline"], environ={})
    assert result["offline"] is True
    assert json.loads(capsys.readouterr().out) == result


def test_main_prints_json_equal_to_the_returned_dict(capsys, hermetic):
    result = envcheck.main([], environ={}, opener=FakeOpener(401))
    out = capsys.readouterr().out
    assert json.loads(out) == result
    assert result["hf_access"][GLOBAL_ID]["result"] == "unauthorized"


def test_token_is_used_but_never_appears_in_the_dict_or_the_output(capsys, hermetic):
    opener = FakeOpener(200)
    result = envcheck.main([], environ={"HF_TOKEN": SECRET}, opener=opener)
    out = capsys.readouterr()
    assert all(call[0].get_header("Authorization") == f"Bearer {SECRET}" for call in opener.calls)
    assert result["hf_token_present"] is True
    assert SECRET not in json.dumps(result)
    assert SECRET not in out.out and SECRET not in out.err
    assert "hf_TESTSECRET" not in out.out


def test_token_from_env_file_is_used_but_never_printed(capsys, tmp_path, hermetic):
    env_file = tmp_path / ".env"
    env_file.write_text(f"HF_TOKEN={SECRET}\n", encoding="utf-8")
    opener = FakeOpener(200)
    result = envcheck.main([], environ={}, env_file=env_file, opener=opener)
    out = capsys.readouterr()
    assert result["hf_token_present"] is True
    assert opener.calls[0][0].get_header("Authorization") == f"Bearer {SECRET}"
    assert SECRET not in out.out + out.err + json.dumps(result)


def test_no_token_means_false_and_no_authorization_header(hermetic):
    opener = FakeOpener(401)
    result = envcheck.collect(environ={}, opener=opener)
    assert result["hf_token_present"] is False
    assert all(call[0].get_header("Authorization") is None for call in opener.calls)


def test_token_inside_an_exception_message_is_scrubbed_from_the_report(hermetic):
    opener = FakeOpener(ValueError(f"oops {SECRET}"))
    result = envcheck.collect(environ={"HF_TOKEN": SECRET}, opener=opener)
    assert SECRET not in json.dumps(result)


GPU_CASES = {
    "none": (None, None),
    "torch": (fake_torch(), None),
    "smi_good": (None, "Tesla T4, 15360 MiB\n"),
    "smi_bad": (None, "garbage in, garbage out"),
}
NET_CASES = {
    "ok": FakeOpener(200),
    "denied": FakeOpener(403),
    "timeout": FakeOpener(TimeoutError("t")),
    "boom": FakeOpener(ValueError("x")),
}


@pytest.mark.parametrize(
    "offline, token, gpu, net",
    list(itertools.product([True, False], [None, SECRET], GPU_CASES, ["ok", "denied", "timeout", "boom"])),
)
def test_json_is_valid_for_every_combination(monkeypatch, capsys, offline, token, gpu, net):
    torch_mod, smi = GPU_CASES[gpu]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: smi)
    environ = {"HF_TOKEN": token} if token else {}
    result = envcheck.main(["--offline"] if offline else [], environ=environ, opener=NET_CASES[net])
    out = capsys.readouterr().out
    assert json.loads(out) == result
    assert set(result) == REQUIRED_KEYS
    assert result["hf_token_present"] is bool(token)
    assert token is None or token not in out
    assert set(result["packages"]) == set(PACKAGES)
    for entry in result["hf_access"].values():
        assert set(entry) == {"status", "result", "error"}


# --- the script ---------------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("env_check_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_main_returns_a_dict(capsys, hermetic):
    module = _load_script()
    result = module.main(["--offline"])
    assert isinstance(result, dict)
    assert json.loads(capsys.readouterr().out) == result


def test_script_runs_offline_as_a_subprocess_and_prints_valid_json():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--offline"], capture_output=True, text=True, timeout=120, cwd=ROOT
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert set(data) == REQUIRED_KEYS
    assert data["offline"] is True
    assert isinstance(data["hf_token_present"], bool)
    assert data["hf_access"][GLOBAL_ID]["result"] == "skipped_offline"


def test_module_entry_point_runs_offline():
    proc = subprocess.run(
        [sys.executable, "-m", "msd.envcheck", "--offline"], capture_output=True, text=True, timeout=120, cwd=ROOT
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["offline"] is True
