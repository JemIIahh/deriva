"""The thin Kaggle/Colab notebooks under kaggle/ are valid, empty, secret-free notebooks."""

import json
import os
import re
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parent.parent / "kaggle"
NOTEBOOKS = ("00_env_check.ipynb", "10_run_stage.ipynb")

TOKEN_LIKE = re.compile(r"hf_[A-Za-z0-9]{20,}")
SECRET_ASSIGNMENT = re.compile(r"(?i)(api_key|secret|token|password|kaggle_key)\s*=\s*[\"'][^\"']{8,}[\"']")


def load(name):
    return json.loads((KAGGLE / name).read_text(encoding="utf-8"))


def source_of(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_nbformat_4_json(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    assert isinstance(nb["nbformat_minor"], int)
    assert isinstance(nb["metadata"], dict)
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], (list, str))
        if cell["cell_type"] == "code":
            assert "outputs" in cell and "execution_count" in cell


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_outputs(name):
    for cell in load(name)["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_secrets_or_personal_paths(name):
    text = (KAGGLE / name).read_text(encoding="utf-8")
    assert not TOKEN_LIKE.search(text)
    assert not SECRET_ASSIGNMENT.search(text)
    assert "/Users/" not in text and "/home/" not in text and "C:\\" not in text
    assert "ram" not in re.findall(r"/[A-Za-z]+/([A-Za-z]+)/", text)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_cells_stay_thin(name):
    for cell in load(name)["cells"]:
        if cell["cell_type"] == "code":
            assert len(source_of(cell).strip().splitlines()) <= 25


def test_env_check_notebook_installs_the_wheel_from_hf_and_runs_the_check():
    text = "\n".join(source_of(c) for c in load("00_env_check.ipynb")["cells"])
    assert "HF_TOKEN" in text
    assert "wheels/" in text and "pip" in text
    assert "envcheck" in text


def test_run_stage_notebook_takes_a_stage_and_calls_the_script():
    nb = load("10_run_stage.ipynb")
    text = "\n".join(source_of(c) for c in nb["cells"])
    assert re.search(r"^STAGE\s*=", text, re.M) and re.search(r"^STEPS\s*=", text, re.M)
    assert "scripts/" in text
    assert "HF_TOKEN" in text
    # every stage script exists now (final review I8): no cell may still say that later tasks add them
    assert "later task" not in text.lower()
    assert "docs/RUNBOOK.md" in source_of(nb["cells"][0])  # the order of the real run lives in the runbook


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_code_cells_are_valid_python(name):
    for cell in load(name)["cells"]:
        if cell["cell_type"] == "code":
            compile(source_of(cell), name, "exec")


def _token_cell(name):
    cells = [source_of(c) for c in load(name)["cells"] if c["cell_type"] == "code"]
    return next(src for src in cells if "def find_token" in src)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_token_cell_reads_the_environment_and_fails_clearly_without_one(name, monkeypatch, capsys):
    src = _token_cell(name)
    monkeypatch.setenv("HF_TOKEN", "dummy-value-for-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exec(compile(src, name, "exec"), {})
    assert capsys.readouterr().out.splitlines()[0] == "HF_TOKEN found"  # says it found one, never shows it
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(AssertionError, match="HF_TOKEN not found"):
        exec(compile(src, name, "exec"), {})


# --- final review I6: the judge key reaches the runtime the same way, optionally -----------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_judge_key_is_optional_and_its_absence_is_said_plainly(name, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "dummy-value-for-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exec(compile(_token_cell(name), name, "exec"), {})  # no error: the stand-in judge needs no key
    assert capsys.readouterr().out.splitlines() == [
        "HF_TOKEN found", "ANTHROPIC_API_KEY not set (only an anthropic: judge needs it)"]
    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_judge_key_from_a_kaggle_secret_goes_into_the_environment_in_memory_and_is_never_printed(
        name, monkeypatch, capsys):
    import sys
    import types

    secrets = {"HF_TOKEN": "kaggle-secret-value-0001", "ANTHROPIC_API_KEY": "kaggle-secret-value-0002"}
    module = types.ModuleType("kaggle_secrets")
    module.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=lambda label: secrets[label])
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    for variable in secrets:
        monkeypatch.setenv(variable, "placeholder")  # so that the undo removes what the cell sets
        monkeypatch.delenv(variable)
    exec(compile(_token_cell(name), name, "exec"), {})
    assert {variable: os.environ[variable] for variable in secrets} == secrets  # for the scripts this notebook starts
    out = capsys.readouterr().out
    assert out.splitlines() == ["HF_TOKEN found", "ANTHROPIC_API_KEY found"]
    assert not any(value in out for value in secrets.values())


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_judge_key_is_never_written_to_a_file_by_the_token_cell(name):
    src = _token_cell(name)
    assert "ANTHROPIC_API_KEY" in src and "open(" not in src and "write" not in src


# --- cells run inside a Jupyter/Colab/Kaggle kernel, whose sys.argv is not ours -----------------------

KERNEL_ARGV = ["ipykernel_launcher.py", "-f", "/x/kernel.json"]


def _cells_calling_envcheck(name):
    """Code cells that call main() or use msd.envcheck (the ones that could parse sys.argv)."""
    sources = [source_of(c) for c in load(name)["cells"] if c["cell_type"] == "code"]
    return [src for src in sources if "main(" in src or "envcheck" in src]


@pytest.fixture
def kernel_runtime(monkeypatch, tmp_path):
    """A kernel-like process: kernel argv, no torch/GPU, a fake 401 from Hugging Face, scratch cwd."""
    import sys
    import urllib.error

    from msd import envcheck

    def fake_open(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(sys, "argv", list(KERNEL_ARGV))
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(envcheck, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(envcheck, "_default_open", fake_open)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_cell(src, name):
    namespace = {}
    try:
        exec(compile(src, name, "exec"), namespace)
    except SystemExit as exc:  # argparse exits instead of raising
        pytest.fail(f"{name}: cell called sys.exit({exc.code}) under a kernel argv")
    return namespace


def test_at_least_one_cell_calls_envcheck():
    assert _cells_calling_envcheck("00_env_check.ipynb")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_cells_calling_envcheck_survive_a_kernel_argv(name, kernel_runtime, capsys):
    for src in _cells_calling_envcheck(name):
        namespace = _run_cell(src, name)
        if "main(" in src:
            report = namespace["report"]
            assert isinstance(report, dict)
            assert report["hf_token_present"] is False
            assert report["hf_access"]["CohereLabs/tiny-aya-global"]["result"] == "unauthorized"


def test_env_check_notebook_also_writes_the_report_to_a_file(kernel_runtime, capsys):
    for src in _cells_calling_envcheck("00_env_check.ipynb"):
        namespace = _run_cell(src, "00_env_check.ipynb")
    written = json.loads((kernel_runtime / "env_check.json").read_text(encoding="utf-8"))
    assert written == namespace["report"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_token_cell_survives_a_kernel_argv(name, kernel_runtime, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "dummy-value-for-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _run_cell(_token_cell(name), name)
    assert capsys.readouterr().out.splitlines()[0] == "HF_TOKEN found"


def test_run_stage_notebook_pulls_pipelines_as_well_as_scripts():
    text = "\n".join(source_of(c) for c in load("10_run_stage.ipynb")["cells"])
    assert '"pipelines/*"' in text and '"scripts/*"' in text


def test_run_stage_notebook_pulls_the_configs_too():
    text = "\n".join(source_of(c) for c in load("10_run_stage.ipynb")["cells"])
    assert '"configs/*"' in text and '"data/**"' in text  # the fine-tune configs live in configs/, not under data/


# --- final review C1/C2: one results root carried between sessions, steps chained in one session -------------------


def _code_cells():
    return [source_of(c) for c in load("10_run_stage.ipynb")["cells"] if c["cell_type"] == "code"]


def _cell_with(needle):
    (src,) = [src for src in _code_cells() if needle in src]
    return src


def _settings():
    namespace = {}
    exec(compile(_cell_with("WORKSPACE_REPO = "), "10_run_stage.ipynb", "exec"), namespace)
    return namespace


def test_the_settings_default_to_one_env_check_with_no_steps_no_install_and_no_model():
    settings = _settings()
    assert settings["STAGE"] == "env_check" and settings["STAGE_ARGS"] == "" and settings["STEPS"] == []
    assert settings["EXTRA_PIP"] == "" and settings["A1_REPO"] == ""


class _Info:
    def __init__(self, private):
        self.private = private


class FakeHub:
    """Stands in for huggingface_hub: records every call, answers repo_info with a privacy flag."""

    def __init__(self, private=True):
        self.private, self.log = private, []
        hub = self

        class Api:
            def repo_info(self, repo_id, **kwargs):
                hub.log.append(("repo_info", repo_id, kwargs))
                return _Info(hub.private)

            def list_repo_files(self, repo_id, **kwargs):
                hub.log.append(("list_repo_files", repo_id, kwargs))
                return ["wheels/msd-0.1.0-py3-none-any.whl", "scripts/run_eval.py"]

            def upload_folder(self, **kwargs):
                hub.log.append(("upload_folder", kwargs))

            def create_repo(self, *args, **kwargs):
                raise AssertionError("the notebook must never create a repo")

        self.HfApi = Api

    def hf_hub_download(self, repo_id, filename, **kwargs):
        self.log.append(("hf_hub_download", repo_id, filename, kwargs))
        return f"downloaded/{filename}"

    def snapshot_download(self, repo_id, **kwargs):
        self.log.append(("snapshot_download", repo_id, kwargs))
        return kwargs.get("local_dir", ".")

    def module(self):
        import types

        module = types.ModuleType("huggingface_hub")
        for name in ("HfApi", "hf_hub_download", "snapshot_download"):
            setattr(module, name, getattr(self, name))
        return module

    def calls(self, kind):
        return [entry for entry in self.log if entry[0] == kind]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    import subprocess
    import sys

    fake = FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake.module())
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: fake.log.append(("subprocess", list(cmd), kwargs)))
    monkeypatch.chdir(tmp_path)
    return fake


def _exec(src, namespace):
    exec(compile(src, "10_run_stage.ipynb", "exec"), namespace)
    return namespace


def _base_namespace(hub, **settings):
    namespace = {**_settings(), "WORKSPACE_REPO": "someone/msd-workspace", "token": "dummy", "HfApi": hub.HfApi}
    namespace.update(settings)
    return namespace


# the pull: private first, then the wheel, then code, data and every earlier result


def test_the_pull_checks_the_repo_is_private_first_and_pulls_the_results_too(hub, capsys):
    _exec(_cell_with("snapshot_download(WORKSPACE_REPO"), _base_namespace(hub))
    kinds = [entry[0] for entry in hub.log]
    assert kinds[0] == "repo_info" and hub.log[0][2]["repo_type"] == "dataset"
    assert kinds.index("repo_info") < kinds.index("snapshot_download")
    (pull,) = hub.calls("snapshot_download")
    assert pull[1] == "someone/msd-workspace" and pull[2]["repo_type"] == "dataset" and pull[2]["local_dir"] == "."
    patterns = pull[2]["allow_patterns"]
    for needed in ("scripts/*", "pipelines/*", "configs/*", "data/**", "results/**"):
        assert needed in patterns, needed
    assert not any(pattern.startswith(("work", ".env", "A1")) for pattern in patterns)


@pytest.mark.parametrize("private", [False, None, "yes"])
def test_a_workspace_repo_that_is_not_private_stops_the_pull_before_anything_is_downloaded(hub, private):
    hub.private = private
    with pytest.raises(RuntimeError, match="is not a private dataset repo"):
        _exec(_cell_with("snapshot_download(WORKSPACE_REPO"), _base_namespace(hub))
    assert [entry[0] for entry in hub.log] == ["repo_info"]  # no wheel, no pull, no install


def test_run_stage_notebook_pulls_pipelines_as_well_as_scripts():
    text = "\n".join(_code_cells())
    assert '"pipelines/*"' in text and '"scripts/*"' in text


def test_run_stage_notebook_pulls_the_configs_too():
    text = "\n".join(_code_cells())
    assert '"configs/*"' in text and '"data/**"' in text  # the fine-tune configs live in configs/, not under data/


# A1 for run_resta.py --drifted, which needs a local folder


def test_a1_is_downloaded_into_a_local_folder_only_when_a1_repo_is_set(hub, capsys):
    cell = _cell_with('local_dir="A1"')
    _exec(cell, _base_namespace(hub))
    assert hub.calls("snapshot_download") == []
    assert "A1_REPO is empty" in capsys.readouterr().out
    _exec(cell, _base_namespace(hub, A1_REPO=" someone/a1-private "))
    (download,) = hub.calls("snapshot_download")
    assert download[1] == "someone/a1-private" and download[2]["local_dir"] == "A1" and download[2]["token"] == "dummy"
    assert "repo_type" not in download[2] or download[2]["repo_type"] == "model"
    assert "dummy" not in capsys.readouterr().out  # the token is never printed


# the upload: the whole results/ folder, private repo only, never weights, work/ or .env, and only when it changed


def _upload_namespace(hub, tmp_path):
    return _exec(_cell_with("upload_folder"), _base_namespace(hub))


def test_an_upload_sends_the_whole_results_folder_after_checking_the_repo_is_private(hub, tmp_path, capsys):
    ns = _upload_namespace(hub, tmp_path)
    before = ns["results_state"]()
    (tmp_path / "results" / "dev" / "A0" / "none").mkdir(parents=True)
    (tmp_path / "results" / "dev" / "A0" / "none" / "hi.eval.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results" / "gate.json").write_text("{}", encoding="utf-8")
    ns["upload_results"](before)
    assert [entry[0] for entry in hub.log] == ["repo_info", "upload_folder"]
    assert hub.log[0][1] == "someone/msd-workspace" and hub.log[0][2]["repo_type"] == "dataset"
    upload = hub.log[1][1]
    assert upload["folder_path"] == "results" and upload["path_in_repo"] == "results"
    assert upload["repo_id"] == "someone/msd-workspace" and upload["repo_type"] == "dataset"
    assert "uploaded results/" in capsys.readouterr().out


@pytest.mark.parametrize("private", [False, None, "yes"])
def test_an_upload_to_a_repo_that_is_not_private_is_refused_and_sends_nothing(hub, tmp_path, private):
    ns = _upload_namespace(hub, tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x.json").write_text("{}", encoding="utf-8")
    hub.private = private
    with pytest.raises(RuntimeError, match="private"):
        ns["upload_results"]({})
    assert [entry[0] for entry in hub.log] == ["repo_info"]  # asked, then refused: upload_folder was never called


def test_an_unchanged_results_folder_is_not_uploaded_and_the_hub_is_not_asked(hub, tmp_path, capsys):
    ns = _upload_namespace(hub, tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "pulled.json").write_text("{}", encoding="utf-8")  # pulled at the start, not changed
    ns["upload_results"](ns["results_state"]())
    assert hub.log == [] and "nothing to upload" in capsys.readouterr().out
    ns["upload_results"]({})  # no results folder at all is the same: nothing to send
    (tmp_path / "results" / "pulled.json").unlink()
    (tmp_path / "results").rmdir()
    hub.log.clear()
    ns["upload_results"](ns["results_state"]())
    assert hub.log == []


def test_the_upload_never_sends_weights_the_work_folder_partial_folders_or_an_env_file(hub, tmp_path):
    import fnmatch

    ns = _upload_namespace(hub, tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x.json").write_text("{}", encoding="utf-8")
    ns["upload_results"]({})
    patterns = hub.calls("upload_folder")[0][1]["ignore_patterns"]
    for needed in ("*.safetensors", "*.bin", "*.pt", "*.pth", "work/**", "**/merged/**"):
        assert needed in patterns, needed
    # huggingface_hub matches the path relative to folder_path with fnmatch, where * also matches a slash
    never = ["finetune/ft_lora/merged/model.safetensors", "finetune/ft_lora/merged/config.json",
             "finetune/ft_lora/adapter/adapter_model.safetensors", "finetune/ft_lora/adapter/adapter_model.bin",
             "finetune/ft_lora/checkpoint-250/optimizer.pt", "finetune/ft_lora/checkpoint-250/rng.pth",
             "work/finetune/ft_lora/checkpoint-250/scheduler.pt", "finetune/ft_lora-smoke/merged/SMOKE.txt",
             "finetune/.ft_lora.0a1b2c3d.partial/adapter_config.json", ".env", "finetune/.env"]
    for path in never:
        assert any(fnmatch.fnmatch(path, pattern) for pattern in patterns), path
    keep = ["finetune/ft_lora/finetune_summary.json", "finetune/ft_lora/adapter/adapter_config.json", "gate.json",
            "tables.csv", "heatmap.png", "reliability.json", "dev/A0/none/hi.gens.jsonl", "dev/A0/none/hi.eval.json",
            "dev/_cache/judge.jsonl", "report/reliability/j2/A1/none/hi.verdicts.jsonl"]
    for path in keep:
        assert not any(fnmatch.fnmatch(path, pattern) for pattern in patterns), path


def test_the_upload_keeps_its_privacy_guard_before_the_upload_and_says_why_in_plain_words():
    text = _cell_with("upload_folder")
    assert text.index(".private") < text.index("upload_folder(")
    assert "is not a private dataset repo" in text and "nothing was uploaded" in text
    assert "completions" in text and "repo_info" in text and "create_repo" not in text


def test_the_markdown_says_weights_travel_only_by_push_to():
    intro = source_of(load("10_run_stage.ipynb")["cells"][0])
    assert "--push-to" in intro and "weights" in intro and "EXTRA_PIP" in intro
    assert "STEPS" in intro and "results/" in intro and "A1_REPO" in intro


# the run: one STAGE, or STEPS in order, stopping at the first failure, uploading after each step


def _run_steps(monkeypatch, tmp_path, *, steps=(), stage="finetune", stage_args="--config configs/ft_lora.yaml --dry-run",
               extra_pip="", fail_on=None, writes=True):
    """Exec the run cell with a recorded subprocess.run: each step writes a file under results/ (unless `writes` is
    False), and the step whose script is `fail_on` exits 3. Returns (calls, uploads)."""
    import subprocess

    calls, uploads = [], []

    def run(cmd, **kwargs):
        cmd = list(cmd)
        calls.append((cmd, kwargs))
        script = Path(cmd[1]).stem if len(cmd) > 1 and cmd[1].startswith("scripts/") else None
        if script is not None and writes:
            (tmp_path / "results").mkdir(exist_ok=True)
            (tmp_path / "results" / f"{len(calls)}-{script}.json").write_text("{}", encoding="utf-8")
        if script is not None and script == fail_on:
            raise subprocess.CalledProcessError(3, cmd)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    namespace = {"STAGE": stage, "STAGE_ARGS": stage_args, "STEPS": list(steps), "EXTRA_PIP": extra_pip}
    upload_cell = _cell_with("upload_folder")
    exec(compile(upload_cell, "10_run_stage.ipynb", "exec"), namespace)
    namespace["upload_results"] = lambda before: uploads.append(namespace["results_state"]() != before)
    exec(compile(_cell_with("scripts/{script}.py"), "10_run_stage.ipynb", "exec"), namespace)
    return calls, uploads


def test_one_stage_runs_its_script_with_its_arguments(monkeypatch, tmp_path):
    calls, uploads = _run_steps(monkeypatch, tmp_path)
    (stage,) = [c[0] for c in calls]
    assert stage[1] == "scripts/finetune.py" and stage[2:] == ["--config", "configs/ft_lora.yaml", "--dry-run"]
    assert calls[0][1].get("check") is True and uploads == [True]


def test_steps_run_in_order_in_one_session_and_each_is_followed_by_an_upload(monkeypatch, tmp_path, capsys):
    steps = [("run_resta", "--drifted A1 --alpha 0.1 --out work/resta/A2_alpha_0.1"),
             ("run_eval", "--arm A2 --alpha 0.1 --split dev --model work/resta/A2_alpha_0.1"),
             ["run_eval", "--arm A2 --alpha 0.1 --split report --model work/resta/A2_alpha_0.1"]]
    calls, uploads = _run_steps(monkeypatch, tmp_path, steps=steps, stage="ignored when STEPS is set")
    assert [c[0][1:] for c in calls] == [
        ["scripts/run_resta.py", "--drifted", "A1", "--alpha", "0.1", "--out", "work/resta/A2_alpha_0.1"],
        ["scripts/run_eval.py", "--arm", "A2", "--alpha", "0.1", "--split", "dev", "--model", "work/resta/A2_alpha_0.1"],
        ["scripts/run_eval.py", "--arm", "A2", "--alpha", "0.1", "--split", "report", "--model", "work/resta/A2_alpha_0.1"]]
    assert uploads == [True, True, True]  # after every step, so a lost session keeps what the earlier steps made
    out = capsys.readouterr().out
    assert "step 1 of 3" in out and "step 3 of 3" in out


def test_the_first_failing_step_stops_the_chain_with_a_plain_message_and_is_not_uploaded(monkeypatch, tmp_path):
    steps = [("run_resta", "--alpha 0.1"), ("run_eval", "--split dev"), ("make_report", "")]
    with pytest.raises(RuntimeError) as caught:
        _run_steps(monkeypatch, tmp_path, steps=steps, fail_on="run_eval")
    message = str(caught.value)
    assert "step 2 (run_eval)" in message and "exit code 3" in message and "did not run" in message
    assert "upload_results({})" in message  # how to keep what the failed step wrote, if wanted
    assert caught.value.__cause__ is None and caught.value.__suppress_context__  # one plain line, no chained traceback


def test_a_failing_step_runs_nothing_after_it(monkeypatch, tmp_path):
    import subprocess

    seen = []

    def run(cmd, **kwargs):
        seen.append(list(cmd)[1])
        if list(cmd)[1] == "scripts/run_eval.py":
            raise subprocess.CalledProcessError(5, cmd)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    namespace = {"STAGE": "x", "STAGE_ARGS": "", "STEPS": [("run_resta", ""), ("run_eval", ""), ("make_report", "")],
                 "EXTRA_PIP": "", "results_state": dict, "upload_results": lambda before: seen.append("upload")}
    with pytest.raises(RuntimeError, match="exit code 5"):
        exec(compile(_cell_with("scripts/{script}.py"), "10_run_stage.ipynb", "exec"), namespace)
    assert seen == ["scripts/run_resta.py", "upload", "scripts/run_eval.py"]  # make_report never started


@pytest.mark.parametrize("bad", [[("run_eval",)], [("../evil", "")], [("scripts/run_eval.py", "")], ["run_eval --arm A0"],
                                 [("run_eval", ["--arm", "A0"])], [("run eval", "")]])
def test_a_malformed_step_is_refused_before_anything_runs(monkeypatch, tmp_path, bad):
    calls = []
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    namespace = {"STAGE": "x", "STAGE_ARGS": "", "STEPS": [("env_check", ""), *bad], "EXTRA_PIP": "aligntune==0.1.31",
                 "results_state": dict, "upload_results": lambda before: None}
    with pytest.raises(ValueError, match="pair of strings"):
        exec(compile(_cell_with("scripts/{script}.py"), "10_run_stage.ipynb", "exec"), namespace)
    assert calls == []  # not even the install


def test_the_default_env_check_runs_once_and_uploads_nothing_even_with_pulled_results(hub, tmp_path, monkeypatch, capsys):
    import subprocess

    (tmp_path / "results" / "dev").mkdir(parents=True)
    (tmp_path / "results" / "dev" / "pulled.json").write_text("{}", encoding="utf-8")  # an earlier session's result
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: calls.append(list(cmd)))  # env_check writes nothing
    namespace = _base_namespace(hub)
    _exec(_cell_with("upload_folder"), namespace)
    _exec(_cell_with("scripts/{script}.py"), namespace)
    assert [c[1] for c in calls] == ["scripts/env_check.py"]
    assert hub.log == []  # no upload and not even a repo_info
    assert "nothing to upload" in capsys.readouterr().out


# EXTRA_PIP: a toolkit installed on the runtime, once, before the first step


def test_extra_pip_is_installed_once_before_the_first_step(monkeypatch, tmp_path):
    calls, _ = _run_steps(monkeypatch, tmp_path, steps=[("finetune", "--dry-run"), ("finetune", "--max-steps 20 --merge")],
                          extra_pip="aligntune==0.1.31")
    install, first, second = (c[0] for c in calls)
    assert install[1:4] == ["-m", "pip", "install"] and "aligntune==0.1.31" in install
    assert first[1] == "scripts/finetune.py" and second[1] == "scripts/finetune.py"
    assert all(kwargs.get("check") is True for _, kwargs in calls)  # a failed install stops the notebook


def test_an_empty_extra_pip_installs_nothing(monkeypatch, tmp_path):
    calls, _ = _run_steps(monkeypatch, tmp_path, extra_pip="   ")
    assert [c[0][1] for c in calls] == ["scripts/finetune.py"]
