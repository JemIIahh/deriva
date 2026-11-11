"""notebooks/reproduce.ipynb (Task 20): the reproduction notebook is clean, honest and actually reproduces.

Three kinds of test:

* structure, always: valid nbformat 4, no stored outputs, no secrets or personal paths, text between the cells, the
  recomputed-versus-recorded explanation, and the optional GPU section behind `RUN_GPU = False`;
* the code cells run for real, always, in plain Python (no kernel needed), against a synthetic export built by the
  real export code: the hash matches, and an export that was changed makes the notebook fail with a plain
  explanation;
* the same two runs through a real notebook runner (`nbclient` and `ipykernel`), only when one is importable; when
  none is, they are skipped with the reason and the plain-Python runs above are what proves the cells work.

Nothing here touches a model, a judge, a key or the network.
"""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "reproduce.ipynb"

TOKEN_LIKE = re.compile(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}")
SECRET_ASSIGNMENT = re.compile(r"(?i)(api_key|secret|token|password|kaggle_key)\s*=\s*[\"'][^\"']{8,}[\"']")


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_of(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def code_cells():
    return [source_of(c) for c in load()["cells"] if c["cell_type"] == "code"]


def text_of_markdown():
    return "\n".join(source_of(c) for c in load()["cells"] if c["cell_type"] == "markdown")


# --- structure ---------------------------------------------------------------------------------------------------------


def test_the_notebook_is_valid_nbformat_4_json():
    nb = load()
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int)
    assert isinstance(nb["metadata"], dict) and "widgets" not in nb["metadata"]  # no widget state
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["metadata"], dict) and isinstance(cell["source"], (list, str))
        if cell["cell_type"] == "code":
            assert "outputs" in cell and "execution_count" in cell


def test_the_notebook_is_stored_clean_with_no_outputs_and_no_execution_counts():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
        assert "attachments" not in cell


def test_the_notebook_has_no_secrets_personal_paths_or_replacement_characters():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not TOKEN_LIKE.search(text) and not SECRET_ASSIGNMENT.search(text)
    assert "/Users/" not in text and "/home/" not in text and "C:\\" not in text
    assert "\N{REPLACEMENT CHARACTER}" not in text and "\\ufffd" not in text.lower()


def test_every_code_cell_compiles_and_none_is_empty_or_only_comments():
    for number, src in enumerate(code_cells(), start=1):
        tree = ast.parse(src, f"cell {number}")
        assert tree.body, f"code cell {number} does nothing (empty or only comments)"
        compile(src, f"cell {number}", "exec")


def test_cells_stay_short_enough_to_read():
    for cell in load()["cells"]:
        lines = source_of(cell).strip().splitlines()
        assert lines, "an empty cell"
        assert len(lines) <= (45 if cell["cell_type"] == "code" else 30)


def test_every_code_cell_has_a_text_cell_right_before_it():
    cells = load()["cells"]
    assert cells[0]["cell_type"] == "markdown"
    for before, cell in zip(cells, cells[1:]):
        if cell["cell_type"] == "code":
            assert before["cell_type"] == "markdown", "two code cells in a row: say in words what the second one does"


def test_the_first_code_cell_sets_export_dir_to_the_export_folder():
    first = code_cells()[0]
    namespace = {}
    exec(compile(first, "first cell", "exec"), namespace)
    assert re.search(r"^EXPORT_DIR\s*=", first, re.M)
    assert Path(namespace["EXPORT_DIR"]) == Path("export")


def test_the_text_says_what_is_recomputed_and_what_is_a_recorded_decision_and_why():
    text = text_of_markdown().lower()
    for needed in ("recomputed", "recorded", "reliability.json", "sanity", "second judge", "low-confidence",
                   "which alpha"):
        assert needed in text, needed
    assert "completions" in text and "not published" in text  # why the other decisions cannot be redone


def test_the_text_says_png_bytes_are_not_compared_and_what_is_instead():
    text = text_of_markdown().lower()
    assert "png" in text and "bytes" in text and "matplotlib" in text and "plotted numbers" in text


def test_the_text_documents_how_to_install_from_a_checkout_and_from_a_downloaded_copy():
    text = text_of_markdown()
    assert "pip install -e ." in text and "PYTHONPATH=src" in text and "matplotlib" in text
    assert "no key" in text.lower() or "no api key" in text.lower()


def test_the_hash_check_has_a_non_zero_outcome_not_a_shrug():
    text = "\n".join(code_cells())
    assert "AssertionError" in text and "sha256" in text and "explain_difference" in text


def _gpu_cell():
    cells = [src for src in code_cells() if re.search(r"^RUN_GPU\s*=", src, re.M)]
    assert len(cells) == 1, "exactly one cell sets RUN_GPU"
    return cells[0]


def test_the_gpu_section_is_behind_run_gpu_false_and_everything_in_it_is_guarded():
    src = _gpu_cell()
    tree = ast.parse(src)
    assert re.search(r"^RUN_GPU\s*=\s*False\s*(#.*)?$", src, re.M)
    for node in tree.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["RUN_GPU"]:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue  # importing is harmless; calling is not
        assert isinstance(node, ast.If) and ast.unparse(node.test) == "RUN_GPU", ast.unparse(node)[:80]
    assert "subprocess" in src and "run_eval" in src  # the section does something when it is switched on
    # the whole notebook never switches it on by itself
    assert not any(re.search(r"^RUN_GPU\s*=\s*True", c, re.M) for c in code_cells())


def test_the_gpu_section_says_it_is_not_guaranteed_and_what_it_needs_and_holds_no_key():
    cells = load()["cells"]
    index = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "RUN_GPU" in source_of(c))
    text = (source_of(cells[index - 1]) + source_of(cells[index])).lower()
    for needed in ("not guaranteed", "gated", "judge", "tiny aya", "differ"):
        assert needed in text, needed
    code = source_of(cells[index])
    assert "os.environ" in code and "HF_TOKEN" in code and "ANTHROPIC_API_KEY" in code  # read from the environment
    assert not SECRET_ASSIGNMENT.search(code) and not TOKEN_LIKE.search(code)
    assert text_of_markdown().lower().count("regenerat") >= 1


# --- the cells run, in plain Python -----------------------------------------------------------------------------------


@pytest.fixture
def notebook_dir(tmp_path, public_exports, monkeypatch):
    """A scratch working folder with a copy of the synthetic export at `export/`, as a downloaded copy would sit."""
    shutil.copytree(public_exports().export, tmp_path / "export")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def ipython_stub(monkeypatch):
    """The notebook shows pictures through IPython (always present in a notebook kernel, not in this environment): a
    stand-in lets the cells run in plain Python."""
    shown = []
    display = types.ModuleType("IPython.display")
    display.Image = lambda *a, **k: ("image", k.get("filename") or (a[0] if a else None))
    display.display = lambda *items: shown.extend(items)
    package = types.ModuleType("IPython")
    package.display = display
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    return shown


@pytest.fixture
def no_subprocess_and_no_ipython(monkeypatch, ipython_stub):
    """As `ipython_stub`, and no process may be started: the GPU section is off, so the notebook must not start one."""
    started = []

    def refuse(*args, **kwargs):
        started.append(args)
        raise AssertionError(f"the notebook started a process: {args[:1]}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    return types.SimpleNamespace(started=started, shown=ipython_stub)


def run_cells(namespace=None, replace=()):
    """Run every code cell in one namespace, as a kernel would; `replace` is (old, new) text swaps made in the cells first."""
    namespace = {} if namespace is None else namespace
    for number, src in enumerate(code_cells(), start=1):
        for old, new in replace:
            src = src.replace(old, new)
        exec(compile(src, f"reproduce.ipynb cell {number}", "exec"), namespace)
    return namespace


def test_the_cells_run_and_the_recomputed_table_matches_the_exported_hash(notebook_dir, no_subprocess_and_no_ipython, capsys):
    namespace = run_cells()
    out = capsys.readouterr().out
    exported = (notebook_dir / "export" / "dataset" / "aggregates" / "tables.csv").read_bytes()
    recomputed = next(Path(namespace["OUT_DIR"]).glob("tables.csv")).read_bytes()
    assert recomputed == exported
    assert hashlib.sha256(recomputed).hexdigest() in out  # the hash is printed, so a reader can see what matched
    assert "MATCH" in out and "MISMATCH" not in out
    pictures = {p.name for p in Path(namespace["OUT_DIR"]).glob("*.png")}
    assert {"heatmap.png", "alpha_curve.png"} <= pictures
    assert not no_subprocess_and_no_ipython.started  # the GPU section stayed off
    assert namespace["RUN_GPU"] is False


def test_the_notebook_leaves_the_export_untouched(notebook_dir, no_subprocess_and_no_ipython):
    before = {p.relative_to(notebook_dir / "export"): p.read_bytes() for p in (notebook_dir / "export").rglob("*") if p.is_file()}
    run_cells()
    after = {p.relative_to(notebook_dir / "export"): p.read_bytes() for p in (notebook_dir / "export").rglob("*") if p.is_file()}
    assert before == after


def test_an_edited_tables_csv_makes_the_notebook_fail_with_a_plain_message(notebook_dir, no_subprocess_and_no_ipython):
    tables = notebook_dir / "export" / "dataset" / "aggregates" / "tables.csv"
    tables.write_bytes(tables.read_bytes().replace(b"A1,,hi,asr,0.500000", b"A1,,hi,asr,0.400000", 1))
    with pytest.raises(AssertionError, match=r"tables\.csv") as caught:
        run_cells()
    assert "changed" in str(caught.value).lower()  # the export no longer matches its MANIFEST.json


def test_an_edited_tables_csv_with_a_fixed_manifest_still_fails_on_the_recomputation(notebook_dir, no_subprocess_and_no_ipython, capsys):
    """The manifest check passes (someone updated the hash too), so only the recomputation can catch it."""
    tables = notebook_dir / "export" / "dataset" / "aggregates" / "tables.csv"
    tables.write_bytes(tables.read_bytes().replace(b"A1,,hi,asr,0.500000", b"A1,,hi,asr,0.400000", 1))
    manifest = notebook_dir / "export" / "MANIFEST.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"]["dataset/aggregates/tables.csv"]["sha256"] = hashlib.sha256(tables.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"MISMATCH|do not match|does not match"):
        run_cells()
    out = capsys.readouterr().out
    assert "arm=A1" in out and "lang=hi" in out and "metric=asr" in out and "0.400000" in out  # which number, not a shrug


def test_an_edited_verdict_row_with_a_fixed_manifest_fails_because_the_numbers_are_recomputed(notebook_dir, no_subprocess_and_no_ipython, capsys):
    path = notebook_dir / "export" / "dataset" / "results_public" / "report" / "A1" / "none" / "hi.verdicts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows[0]["label"] = "complied" if rows[0]["label"] == "refused" else "refused"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = notebook_dir / "export" / "MANIFEST.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"]["dataset/results_public/report/A1/none/hi.verdicts.jsonl"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AssertionError):
        run_cells()
    assert "lang=hi" in capsys.readouterr().out


def test_started_in_the_notebooks_folder_it_finds_the_export_one_level_up(tmp_path, public_exports, no_subprocess_and_no_ipython, monkeypatch, capsys):
    """Jupyter (and nbconvert) run a notebook with its own folder as the working directory: `notebooks/`, not the repo root."""
    shutil.copytree(public_exports().export, tmp_path / "export")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")
    namespace = run_cells()
    assert Path(namespace["EXPORT_DIR"]) == Path("..") / "export"
    assert "MATCH" in capsys.readouterr().out and (tmp_path / "notebooks" / "reproduced" / "tables.csv").is_file()
    assert not (tmp_path / "export" / "reproduced").exists()


def test_without_msd_installed_and_without_a_src_folder_the_import_cell_says_what_to_do(tmp_path, monkeypatch, no_subprocess_and_no_ipython):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "msd", None)  # `from msd import ...` now fails as if it were not installed
    with pytest.raises(AssertionError, match=r"msd is not installed.*install lines"):
        run_cells()


def test_a_missing_export_folder_fails_with_a_message_that_says_where_to_look(tmp_path, monkeypatch, no_subprocess_and_no_ipython):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match=r"export"):
        run_cells()


# --- the optional GPU section: never run by default, but its commands are checked --------------------------------------


GPU_ON = (("RUN_GPU = False", "RUN_GPU = True"), ('"YOUR-HF-USERNAME/published-A1-model"', '"someone/published-A1"'))


@pytest.fixture
def fake_gpu_run(monkeypatch, ipython_stub):
    """A stand-in for `subprocess.run`: it checks that the command is one `scripts/run_eval.py` accepts (its own parser),
    that the rebuilt test set is where it says, and writes the fresh verdicts a real run would (half of them flipped)."""
    from msd import evalrun
    from msd.schema import Prompt, Verdict, read_jsonl, write_jsonl

    calls = []
    Path("scripts").mkdir()
    Path("scripts", "run_eval.py").touch()  # only its existence is looked at; nothing is run

    def run(argv, check=False, **kwargs):
        assert check is True and Path(argv[1]).name == "run_eval.py" and argv[1].endswith("scripts/run_eval.py")
        args = evalrun.parse_args(argv[2:], root=ROOT)
        calls.append(args)
        prompts = read_jsonl(Path(args.testset_dir) / args.langs[0] / "report.jsonl", Prompt)[: args.limit]
        published = {}
        for line in (Path("export") / "dataset/results_public/report/A1/none/hi.verdicts.jsonl").read_text("utf-8").split("\n"):
            if line:
                row = json.loads(line)
                published[row["prompt_id"]] = row["label"]
        flip = {"complied": "refused", "refused": "complied"}
        rows = [Verdict(p.id, args.arm, None, p.lang, p.kind, "fresh-judge", published[p.id] if i % 2 else flip.get(published[p.id], "refused"), "x")
                for i, p in enumerate(prompts)]
        write_jsonl(evalrun.verdicts_path(args.out_root, args.split, args.arm, None, args.langs[0]), rows)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_the_gpu_section_when_switched_on_builds_a_command_run_eval_accepts_and_compares_labels(
        notebook_dir, fake_gpu_run, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "dummy-value-for-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-value-for-test")
    run_cells(replace=GPU_ON)
    (args,) = fake_gpu_run
    assert (args.arm, args.split, list(args.langs), args.limit, args.model_path) == ("A1", "report", ["hi"], 20, "someone/published-A1")
    assert args.judge == "anthropic:claude-sonnet-5-5" and list(args.stages) == ["generate", "judge"]
    assert not (Path(args.ft_dir) / "heldout.jsonl").exists()  # no task items are generated for a 20-item check
    assert "10 of 20 labels equal the published ones" in capsys.readouterr().out


@pytest.mark.parametrize("missing, message", [("HF_TOKEN", "HF_TOKEN is not set"), ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY is not set")])
def test_the_gpu_section_says_which_key_is_missing_and_starts_nothing(notebook_dir, fake_gpu_run, monkeypatch, missing, message):
    for name in ("HF_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "dummy-value-for-test")
    monkeypatch.delenv(missing)
    with pytest.raises(AssertionError, match=message):
        run_cells(replace=GPU_ON)
    assert fake_gpu_run == []


def test_the_gpu_section_refuses_the_placeholder_model_id(notebook_dir, fake_gpu_run, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-value-for-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-value-for-test")
    with pytest.raises(AssertionError, match="edit MODEL"):
        run_cells(replace=GPU_ON[:1])
    assert fake_gpu_run == []


# --- the cells run, in a real notebook kernel (only when a runner is installed) ---------------------------------------


def _runner():
    pytest.importorskip("nbformat", reason="no notebook runner: pip install nbclient ipykernel in envs/core to run the notebook itself")
    pytest.importorskip("nbclient", reason="no notebook runner: pip install nbclient ipykernel in envs/core to run the notebook itself")
    pytest.importorskip("ipykernel", reason="no notebook kernel: pip install ipykernel in envs/core to run the notebook itself")
    import nbclient
    import nbformat

    return nbformat, nbclient


def _execute(workdir, monkeypatch):
    nbformat, nbclient = _runner()
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(workdir)}})
    client.execute()
    return nb


def test_the_notebook_executes_in_a_kernel_and_prints_a_match(notebook_dir, monkeypatch):
    nb = _execute(notebook_dir, monkeypatch)
    printed = "".join(o.get("text", "") for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs)
    assert "MATCH" in printed and "MISMATCH" not in printed


def test_the_notebook_fails_in_a_kernel_when_tables_csv_was_edited(notebook_dir, monkeypatch):
    _runner()
    from nbclient.exceptions import CellExecutionError

    tables = notebook_dir / "export" / "dataset" / "aggregates" / "tables.csv"
    tables.write_bytes(tables.read_bytes().replace(b"A1,,hi,asr,0.500000", b"A1,,hi,asr,0.400000", 1))
    with pytest.raises(CellExecutionError):
        _execute(notebook_dir, monkeypatch)
