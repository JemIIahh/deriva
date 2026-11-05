"""msd.atomicdir: write a folder to a temporary sibling and rename it into place only when it is complete."""

from __future__ import annotations

import pytest

from msd.atomicdir import atomic_output_dir, check_out_dir


class Boom(RuntimeError):
    pass


def test_success_renames_the_folder_into_place_and_leaves_nothing_else(tmp_path):
    out = tmp_path / "out"
    with atomic_output_dir(out) as folder:
        assert folder.parent == tmp_path and folder != out and not out.exists()
        (folder / "a.txt").write_text("a")
    assert (out / "a.txt").read_text() == "a"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_missing_parents_are_created(tmp_path):
    out = tmp_path / "x" / "y" / "out"
    with atomic_output_dir(out) as folder:
        (folder / "f").write_text("1")
    assert (out / "f").is_file()


def test_an_empty_existing_out_dir_is_replaced(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with atomic_output_dir(out) as folder:
        (folder / "f").write_text("1")
    assert (out / "f").is_file()


@pytest.mark.parametrize("exc", [Boom, KeyboardInterrupt])
def test_a_failure_removes_the_partial_folder_and_creates_no_out_dir(tmp_path, exc):
    out = tmp_path / "out"
    with pytest.raises(exc):
        with atomic_output_dir(out) as folder:
            (folder / "half.bin").write_bytes(b"x")
            raise exc("stop")
    assert list(tmp_path.iterdir()) == []


def test_a_failure_leaves_a_pre_existing_empty_out_dir_empty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(Boom):
        with atomic_output_dir(out) as folder:
            (folder / "half.bin").write_bytes(b"x")
            raise Boom("stop")
    assert out.is_dir() and list(out.iterdir()) == []
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_nested_files_are_synced_and_kept(tmp_path):
    out = tmp_path / "out"
    with atomic_output_dir(out) as folder:
        (folder / "sub").mkdir()
        (folder / "sub" / "deep.txt").write_text("d")
        (folder / "top.txt").write_text("t")
    assert (out / "sub" / "deep.txt").read_text() == "d" and (out / "top.txt").read_text() == "t"


def test_check_out_dir_refuses_a_non_empty_directory_and_a_file(tmp_path):
    check_out_dir(tmp_path / "missing")  # fine
    empty = tmp_path / "empty"
    empty.mkdir()
    check_out_dir(empty)  # fine
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep").write_text("k")
    with pytest.raises(FileExistsError, match="not empty"):
        check_out_dir(full)
    afile = tmp_path / "file"
    afile.write_text("f")
    with pytest.raises(FileExistsError, match="not a directory"):
        check_out_dir(afile)


def test_a_non_empty_out_dir_is_refused_on_entry_too(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("k")
    with pytest.raises(FileExistsError):
        with atomic_output_dir(out):
            pytest.fail("the body must not run")
    assert [p.name for p in out.iterdir()] == ["keep"]
    assert [p.name for p in tmp_path.iterdir()] == ["out"]
