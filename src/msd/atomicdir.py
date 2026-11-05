"""Write a model folder so that a crash never leaves a half-written one that looks complete.

    with atomic_output_dir(out_dir) as folder:     # a fresh hidden folder next to out_dir
        ...write every file into `folder`...
    # here out_dir exists and holds exactly what was written

The body writes into `.<name>.<8 hex digits>.partial` in the same parent directory (same file system, so the
final rename is atomic). When the body finishes, every file is flushed to disk and the folder is renamed to
`out_dir`. On any exception (a `KeyboardInterrupt` included) the partial folder is deleted and `out_dir` is
not created; an `out_dir` that already existed, empty, is left empty. A process killed outright (SIGKILL, a
power cut) cannot clean up, so its partial folder stays; `note_stale_partials` names such leftovers at the
start of the next run and deletes nothing.

`out_dir` must not exist or must be an empty directory (`check_out_dir`): a non-empty one or a file is refused
before the body runs (FileExistsError), and so are a symbolic link or dangling link (ValueError: renaming a
folder onto a link would replace or follow it) and, when the caller names the input folders, an `out_dir` that is
one of them or inside one (ValueError, after resolving links, so a symlinked parent does not hide it).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

_SUFFIXES = ("KiB", "MiB", "GiB")


def check_out_dir(out_dir: Path, *, sources: Iterable[tuple[str, Path]] = ()) -> None:
    """Refuse an `out_dir` that must not be written; `sources` is `(label, folder)` pairs it must stay out of."""
    out_dir = Path(out_dir)
    if out_dir.is_symlink():  # true for a dangling link too
        raise ValueError(f"out_dir {out_dir} is a symbolic link; give the real path of the folder to create")
    resolved = out_dir.resolve()
    for label, source in sources:
        root = Path(source).resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"out_dir {out_dir} is, or is inside, the {label} folder {source}; choose another folder")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"out_dir {out_dir} exists and is not a directory")
        if any(out_dir.iterdir()):
            raise FileExistsError(f"out_dir {out_dir} is not empty; refusing to write into it")


def _folder_bytes(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:  # vanished while we looked
                pass
    return total


def stale_partials(out_dir: Path) -> list[tuple[Path, int]]:
    """`(folder, bytes)` of the leftover `.<name>.<8 hex>.partial` folders next to `out_dir`, sorted by name."""
    out_dir = Path(out_dir)
    parent = out_dir.parent
    if not parent.is_dir():
        return []
    pattern = re.compile(rf"\.{re.escape(out_dir.name)}\.[0-9a-f]{{8}}\.partial")
    found = [p for p in parent.iterdir() if pattern.fullmatch(p.name) and p.is_dir() and not p.is_symlink()]
    return [(p, _folder_bytes(p)) for p in sorted(found)]


def _size_text(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for suffix in _SUFFIXES:
        value /= 1024
        if value < 1024 or suffix == _SUFFIXES[-1]:
            return f"{value:.1f} {suffix}"
    raise AssertionError("unreachable")  # pragma: no cover


def note_stale_partials(out_dir: Path) -> None:
    """Say on stderr, in one line, which leftover partial folders of an interrupted run sit next to `out_dir`."""
    leftovers = stale_partials(out_dir)
    if leftovers:
        listed = ", ".join(f"{p.name} ({_size_text(size)})" for p, size in leftovers)
        print(
            f"note: leftover partial folder(s) from an interrupted run next to {out_dir}, not deleted "
            f"(remove them by hand if you do not need them): {listed}",
            file=sys.stderr,
        )


def _fsync_tree(folder: Path) -> None:
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


@contextmanager
def atomic_output_dir(out_dir: Path) -> Iterator[Path]:
    out_dir = Path(out_dir)
    check_out_dir(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    folder = out_dir.parent / f".{out_dir.name}.{uuid.uuid4().hex[:8]}.partial"
    folder.mkdir()
    try:
        yield folder
        _fsync_tree(folder)
        if out_dir.exists():  # an empty directory: removing it first works on every platform
            out_dir.rmdir()
        os.replace(folder, out_dir)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
