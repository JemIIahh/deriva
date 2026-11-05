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
