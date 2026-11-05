"""The thin Kaggle/Colab notebooks under kaggle/ are valid, empty, secret-free notebooks."""

import json
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
    assert re.search(r"^STAGE\s*=", text, re.M)
    assert "scripts/" in text
    assert "HF_TOKEN" in text
    assert "later task" in text.lower()  # documents that later tasks add the stage scripts


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
    exec(compile(src, name, "exec"), {})
    assert capsys.readouterr().out.strip() == "HF_TOKEN found"  # says it found one, never shows it
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(AssertionError, match="HF_TOKEN not found"):
        exec(compile(src, name, "exec"), {})


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
    _run_cell(_token_cell(name), name)
    assert capsys.readouterr().out.strip() == "HF_TOKEN found"


def test_run_stage_notebook_pulls_pipelines_as_well_as_scripts():
    text = "\n".join(source_of(c) for c in load("10_run_stage.ipynb")["cells"])
    assert '"pipelines/*"' in text and '"scripts/*"' in text


# --- the upload cell only ever sends completions to a private repo ---------------------------------------


class _Info:
    def __init__(self, private):
        self.private = private


def _upload_cell():
    cells = [source_of(c) for c in load("10_run_stage.ipynb")["cells"] if c["cell_type"] == "code"]
    return next(src for src in cells if "upload_folder" in src)


def _run_upload_cell(private, tmp_path, monkeypatch, log):
    class FakeApi:
        def repo_info(self, repo_id, **kwargs):
            log.append(("repo_info", repo_id, kwargs))
            return _Info(private)

        def upload_folder(self, **kwargs):
            log.append(("upload_folder", kwargs))

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "run_eval").mkdir(parents=True)
    namespace = {"STAGE": "run_eval", "WORKSPACE_REPO": "someone/msd-workspace", "token": "dummy", "HfApi": FakeApi}
    exec(compile(_upload_cell(), "10_run_stage.ipynb", "exec"), namespace)


def test_upload_cell_checks_the_repo_is_private_before_uploading(tmp_path, monkeypatch):
    log = []
    _run_upload_cell(True, tmp_path, monkeypatch, log)
    assert [entry[0] for entry in log] == ["repo_info", "upload_folder"]
    assert log[0][2]["repo_type"] == "dataset" and log[0][1] == "someone/msd-workspace"


@pytest.mark.parametrize("private", [False, None, "yes"])
def test_upload_cell_refuses_a_repo_that_is_not_private_and_uploads_nothing(private, tmp_path, monkeypatch):
    log = []
    with pytest.raises(RuntimeError, match="private"):
        _run_upload_cell(private, tmp_path, monkeypatch, log)
    assert [entry[0] for entry in log] == ["repo_info"]  # asked, then refused: upload_folder was never called


def test_upload_cell_says_why_in_plain_words():
    text = _upload_cell()
    assert ".private" in text and "repo_info" in text
    assert "completions" in text and "private" in text.lower()


def test_upload_cell_with_no_results_folder_uploads_and_checks_nothing(tmp_path, monkeypatch):
    class FakeApi:
        def __init__(self):
            raise AssertionError("no Hugging Face call is needed when there is nothing to upload")

    monkeypatch.chdir(tmp_path)
    namespace = {"STAGE": "run_eval", "WORKSPACE_REPO": "someone/msd-workspace", "token": "dummy", "HfApi": FakeApi}
    exec(compile(_upload_cell(), "10_run_stage.ipynb", "exec"), namespace)
