"""Repository hygiene guards: two mistakes that already happened once and are easy to repeat.

1. The Unicode replacement character (U+FFFD, bytes EF BF BD in UTF-8) in a source, test, script, notebook,
   doc or pipeline file. It appears when a tool writes a mangled escape as the raw character, and it is
   invisible in most editors. The shell check `grep -P` with that byte pattern is a silent no-op on the
   development machine (it matches nothing and says nothing), so this test replaces it.
2. An absolute path into someone's home folder on macOS (the users directory, then a user name) in code or
   tests: it works on one laptop and nowhere else, and it leaks the user name.

Both scans read bytes, skip files that contain a NUL byte (binary), and run over the files git knows about:
tracked files plus new files that are not ignored, so a file is checked before its first commit too.
The needles are built from pieces so that this file does not contain them and does not flag itself.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# U+FFFD as UTF-8, split in two so that this file does not hold the three bytes in a row.
REPLACEMENT_CHARACTER = b"\xef\xbf" + b"\xbd"
REPLACEMENT_PATTERN = re.compile(re.escape(REPLACEMENT_CHARACTER))
# The macOS users directory followed by a user name; built from pieces so this file does not match itself.
HOME_PATH_PATTERN = re.compile(rb"/" + b"Users" + rb"/[A-Za-z0-9._-]+")

TEXT_DIRS = ("src", "tests", "scripts", "kaggle", "docs", "pipelines")
HOME_DIRS = ("src", "scripts", "kaggle", "tests")


def scan(paths, pattern: re.Pattern[bytes], root: Path) -> list[str]:
    """`path:line` of every match of `pattern` in the files, skipping binary files (any NUL byte)."""
    hits = []
    for path in paths:
        if not path.is_file():  # in the index but deleted from the working tree
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for match in pattern.finditer(data):
            hits.append(f"{path.relative_to(root).as_posix()}:{data.count(b'\n', 0, match.start()) + 1}")
    return hits


def git_files(dirs) -> list[Path]:
    """Tracked files plus untracked, not-ignored files under `dirs`; the test is skipped without git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *dirs],
            capture_output=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"hygiene guard needs `git ls-files` in a git checkout of {ROOT}: {type(exc).__name__}")
    return [ROOT / os.fsdecode(name) for name in out.split(b"\0") if name]


# --- the repository ----------------------------------------------------------------------------------------------


def test_no_replacement_character_in_tracked_text_files():
    files = git_files(TEXT_DIRS)
    assert ROOT / "tests" / "test_hygiene.py" in files, "the file listing is broken: this very file is not in it"
    assert ROOT / "src" / "msd" / "config.py" in files
    hits = scan(files, REPLACEMENT_PATTERN, ROOT)
    assert not hits, "U+FFFD (EF BF BD) found, replace it with the intended character: " + ", ".join(hits)


def test_no_absolute_home_path_in_code_and_tests():
    files = git_files(HOME_DIRS)
    assert ROOT / "tests" / "test_hygiene.py" in files, "the file listing is broken: this very file is not in it"
    hits = scan(files, HOME_PATH_PATTERN, ROOT)
    assert not hits, "an absolute home path (a macOS users folder) found, use a path relative to the repo: " + ", ".join(hits)


# --- the guards themselves: control files prove each scan can fail ---------------------------------------------


def test_scan_finds_the_replacement_character_in_a_control_file(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_bytes(b"first line\nprint('a" + REPLACEMENT_CHARACTER + b"b')\n")
    clean = tmp_path / "clean.py"
    clean.write_bytes("café — مرحبا हिन्दी\n".encode("utf-8"))
    escaped = tmp_path / "escaped.py"
    escaped.write_text("x = '\\ufffd'  # the six-character escape is not the raw character\n")
    assert scan([bad, clean, escaped], REPLACEMENT_PATTERN, tmp_path) == ["bad.py:2"]


def test_scan_finds_the_character_whatever_its_neighbours(tmp_path):
    # as the first and the last bytes of a file, and several times in one line
    path = tmp_path / "edge.md"
    path.write_bytes(REPLACEMENT_CHARACTER + b"x" + REPLACEMENT_CHARACTER + REPLACEMENT_CHARACTER)
    assert scan([path], REPLACEMENT_PATTERN, tmp_path) == ["edge.md:1"] * 3


def test_scan_skips_binary_files_and_missing_ones(tmp_path):
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\x00\x01" + REPLACEMENT_CHARACTER)
    assert scan([binary, tmp_path / "gone.py"], REPLACEMENT_PATTERN, tmp_path) == []


def test_needle_is_the_utf8_encoding_of_u_fffd():
    assert REPLACEMENT_CHARACTER == "\N{REPLACEMENT CHARACTER}".encode("utf-8")
    assert REPLACEMENT_CHARACTER == bytes([0xEF, 0xBF, 0xBD])


def test_home_path_scan_finds_a_control_file(tmp_path):
    users = "/" + "Users"  # not written out, so this file is clean
    bad = tmp_path / "bad.py"
    bad.write_text(f"ROOT = '{users}/someone/code/project'\nOK = 1\nOTHER = '{users}/another.name'\n")
    fine = tmp_path / "fine.py"
    fine.write_text(f"DOC = '{users}/<name>/code'  # a placeholder is not a user name\nT = '{users}/'\nU = '/tmp/x'\nV = 'Users/x'\n")
    assert scan([bad, fine], HOME_PATH_PATTERN, tmp_path) == ["bad.py:1", "bad.py:3"]
