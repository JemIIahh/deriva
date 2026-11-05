"""Write a model folder so that a crash never leaves a half-written one that looks complete.

    with atomic_output_dir(out_dir) as folder:     # a fresh hidden folder next to out_dir
        ...write every file into `folder`...
    # here out_dir exists and holds exactly what was written

The body writes into `.<name>.<random>.partial` in the same parent directory (same file system, so the
final rename is atomic). When the body finishes, every file is flushed to disk and the folder is renamed to
`out_dir`. On any exception (a `KeyboardInterrupt` included) the partial folder is deleted and `out_dir` is
not created; an `out_dir` that already existed, empty, is left empty. `out_dir` must not exist or must be an
empty directory; a non-empty one, or a file, is refused before the body runs.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def check_out_dir(out_dir: Path) -> None:
    """Raise FileExistsError unless `out_dir` is missing or an empty directory."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"out_dir {out_dir} exists and is not a directory")
        if any(out_dir.iterdir()):
            raise FileExistsError(f"out_dir {out_dir} is not empty; refusing to write into it")


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
