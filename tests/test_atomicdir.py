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


# === fix round 1: where out_dir may be, and leftovers of an interrupted run ======================================


def test_check_out_dir_refuses_a_symlink_and_a_dangling_one(tmp_path):
    from msd.atomicdir import check_out_dir

    target = tmp_path / "t"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    for name in ("link", "dangling"):
        with pytest.raises(ValueError, match="symbolic link"):
            check_out_dir(tmp_path / name)
        with pytest.raises(ValueError, match="symbolic link"):
            with atomic_output_dir(tmp_path / name):
                pytest.fail("the body must not run")
    assert list(target.iterdir()) == [] and sorted(p.name for p in tmp_path.iterdir()) == ["dangling", "link", "t"]


def test_check_out_dir_refuses_the_inside_of_a_source_directory(tmp_path):
    from msd.atomicdir import check_out_dir

    src = tmp_path / "src"
    src.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    sources = (("A1", src), ("A0", other))
    for out, label in ((src, "A1"), (src / "x", "A1"), (src / "x" / "y", "A1"), (other / "z", "A0")):
        with pytest.raises(ValueError, match=label):
            check_out_dir(out, sources=sources)
    check_out_dir(tmp_path / "src-out", sources=sources)  # a sibling with a similar name is not inside
    check_out_dir(tmp_path / "elsewhere", sources=sources)
    (tmp_path / "alias").symlink_to(src, target_is_directory=True)
    with pytest.raises(ValueError, match="A1"):
        check_out_dir(tmp_path / "alias" / "x", sources=sources)  # inside through a symlinked parent


def test_stale_partials_lists_only_this_outputs_leftover_folders(tmp_path):
    from msd.atomicdir import stale_partials

    out = tmp_path / "out"
    mine = tmp_path / ".out.abc12345.partial"
    mine.mkdir()
    (mine / "a").write_bytes(b"1234")
    (mine / "sub").mkdir()
    (mine / "sub" / "b").write_bytes(b"123456")
    (tmp_path / ".other.abc12345.partial").mkdir()  # another output
    (tmp_path / ".out.abc12345.tmp").mkdir()  # not a partial
    (tmp_path / ".out.file.partial").write_text("a file, not a folder")
    (tmp_path / "out.abc.partial").mkdir()  # not hidden
    assert stale_partials(out) == [(mine, 10)]
    assert stale_partials(tmp_path / "no" / "such" / "out") == []  # a missing parent is fine


def test_stale_partials_matches_the_exact_name_and_nothing_like_it(tmp_path):
    from msd.atomicdir import stale_partials

    (tmp_path / ".o[1].abcdef12.partial").mkdir()  # glob characters in the output's name are just characters
    (tmp_path / ".o1.abcdef12.partial").mkdir()
    assert [p.name for p, _ in stale_partials(tmp_path / "o[1]")] == [".o[1].abcdef12.partial"]
    # another output whose name merely starts with ours ("out.v2" vs "out") is not ours
    (tmp_path / ".out.v2.abcdef12.partial").mkdir()
    (tmp_path / ".out.abcdef12.partial").mkdir()
    (tmp_path / ".out.not-hex!.partial").mkdir()
    assert [p.name for p, _ in stale_partials(tmp_path / "out")] == [".out.abcdef12.partial"]
    assert [p.name for p, _ in stale_partials(tmp_path / "out.v2")] == [".out.v2.abcdef12.partial"]


def test_a_leftover_made_by_atomic_output_dir_itself_is_what_stale_partials_finds(tmp_path):
    from msd.atomicdir import stale_partials

    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        with atomic_output_dir(out) as folder:
            (folder / "half").write_bytes(b"x" * 7)
            raise KeyboardInterrupt
    assert stale_partials(out) == []  # a clean exit removes its folder
    # what a SIGKILL would leave: the same kind of folder, never cleaned up
    killed = tmp_path / ".out.0123abcd.partial"
    killed.mkdir()
    (killed / "half").write_bytes(b"x" * 7)
    assert stale_partials(out) == [(killed, 7)]


@pytest.mark.parametrize("size, text", [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KiB"), (3000, "2.9 KiB"),
                                         (5 * 1024**2, "5.0 MiB"), (int(2.5 * 1024**3), "2.5 GiB")])
def test_note_stale_partials_says_one_line_with_sizes_and_deletes_nothing(tmp_path, capsys, size, text):
    from msd.atomicdir import note_stale_partials

    leftover = tmp_path / ".out.deadbeef.partial"
    leftover.mkdir()
    with open(leftover / "f", "wb") as fh:
        fh.truncate(size)
    note_stale_partials(tmp_path / "out")
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1 and ".out.deadbeef.partial" in lines[0] and text in lines[0] and "not deleted" in lines[0]
    assert (leftover / "f").exists()


def test_note_stale_partials_names_every_leftover_on_the_one_line(tmp_path, capsys):
    from msd.atomicdir import note_stale_partials

    for suffix in ("aaaa1111", "bbbb2222"):
        (tmp_path / f".out.{suffix}.partial").mkdir()
    note_stale_partials(tmp_path / "out")
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and "aaaa1111" in lines[0] and "bbbb2222" in lines[0]


def test_note_stale_partials_is_silent_when_there_are_none(tmp_path, capsys):
    from msd.atomicdir import note_stale_partials

    note_stale_partials(tmp_path / "out")
    assert capsys.readouterr().err == ""
