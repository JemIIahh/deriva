"""Repository hygiene guards: two mistakes that already happened once and are easy to repeat.

1. The Unicode replacement character (U+FFFD, bytes EF BF BD in UTF-8) in ANY file git knows about: root
   files, reference/, docs, notebooks and the data manifest included, not a list of folders. It appears when
   a tool writes a mangled escape as the raw character, and it is invisible in most editors. The shell check
   `grep -P` with that byte pattern is a silent no-op on the development machine (it matches nothing and says
   nothing), so this test replaces it.
2. An absolute path into someone's home folder (the macOS users folder or the Linux home folder, then a user
   name) in code or tests: it works on one laptop and nowhere else, and it leaks the user name. This scan stays
   in src, scripts, kaggle and tests: notes such as HANDOFF.md at the repo root may name where a thing lives.

Both scans read bytes, skip files that contain a NUL byte (binary), and run over the files git knows about:
tracked files plus new files that are not ignored (so a file is checked before its first commit, and the
ignored data/ files and the virtual environments are not). The needles are built from pieces so that this
file does not contain them and does not flag itself.
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
# An absolute path into a home folder: the macOS users directory or the Linux home directory, then a user name.
# It must start a path (not follow a letter, digit, dot or tilde, so "site/<dir>/page" is not one), and it is built
# from pieces so this file does not match itself.
HOME_PATH_PATTERN = re.compile(rb"(?<![\w.~-])/(?:" + b"Users" + rb"|" + b"home" + rb")/[A-Za-z0-9._-]+")

HOME_DIRS = ("src", "scripts", "kaggle", "tests")  # the FFFD scan has no such list: it covers every file git knows


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


def git_files(dirs=(), root: Path = ROOT) -> list[Path]:
    """Tracked files plus untracked, not-ignored files of the repository at `root`, narrowed to `dirs` when given.

    With no `dirs` that is every file git knows about, the root files included. The test is skipped without git."""
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    if dirs:
        command += ["--", *dirs]
    try:
        out = subprocess.run(command, capture_output=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"hygiene guard needs `git ls-files` in a git checkout of {root}: {type(exc).__name__}")
    return [root / os.fsdecode(name) for name in out.split(b"\0") if name]


# --- the repository ----------------------------------------------------------------------------------------------


def test_no_replacement_character_in_any_file_git_knows():
    files = git_files()
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


# --- fix round 1: every git-known file for the replacement character, /home paths too ---------------------------------


def _git_repo(tmp_path: Path) -> Path:
    """A throw-away repository with files at the root, in a nested folder, ignored, and binary."""
    try:
        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"needs git: {type(exc).__name__}")
    bad = b"x" + REPLACEMENT_CHARACTER + b"y\n"
    (tmp_path / ".gitignore").write_text("ignored.txt\ndata/*\n!data/sources.yaml\n")
    (tmp_path / "notes.txt").write_bytes(bad)  # at the root
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "slides.txt").write_bytes(bad)  # in a folder no list of directories would name
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sources.yaml").write_bytes(bad)  # the one data file that is committed
    (tmp_path / "data" / "big.jsonl").write_bytes(bad)  # ignored: data/ stays local
    (tmp_path / "ignored.txt").write_bytes(bad)
    (tmp_path / "blob.bin").write_bytes(b"\x00" + bad)  # binary: skipped
    (tmp_path / "clean.txt").write_text("fine\n")
    return tmp_path


def test_the_file_listing_covers_the_whole_repository_not_a_list_of_folders(tmp_path):
    repo = _git_repo(tmp_path)
    listed = {p.relative_to(repo).as_posix() for p in git_files(root=repo)}
    assert listed == {".gitignore", "notes.txt", "reference/slides.txt", "data/sources.yaml", "blob.bin", "clean.txt"}
    hits = scan(git_files(root=repo), REPLACEMENT_PATTERN, repo)
    assert sorted(hits) == ["data/sources.yaml:1", "notes.txt:1", "reference/slides.txt:1"]  # not the ignored files, not the binary


def test_the_file_listing_can_still_be_narrowed_to_folders(tmp_path):
    repo = _git_repo(tmp_path)
    assert {p.relative_to(repo).as_posix() for p in git_files(("reference",), root=repo)} == {"reference/slides.txt"}


def test_the_real_replacement_scan_reaches_root_files_reference_and_the_data_manifest():
    listed = set(git_files())
    for name in ("pyproject.toml", "HANDOFF.md", ".gitignore", "data/sources.yaml"):
        assert ROOT / name in listed, name
    assert any(p.parent == ROOT / "reference" for p in listed)
    assert not any(p.relative_to(ROOT).parts[0] == "envs" for p in listed)  # the virtual environments are ignored


def test_the_home_path_scan_stays_in_code_and_tests_so_the_handoff_note_is_not_flagged():
    files = git_files(HOME_DIRS)
    assert files and all(p.relative_to(ROOT).parts[0] in HOME_DIRS for p in files)
    assert ROOT / "HANDOFF.md" not in files and ROOT / "docs" / "runtime.md" not in files


def test_home_path_scan_finds_the_linux_home_folder_too(tmp_path):
    home = "/" + "home"  # not written out, so this file is clean
    bad = tmp_path / "bad.py"
    bad.write_text(f"A = '{home}/someone/code'\nB = \"{home}/runner\"\nC = f'{{x}} {home}/dev.name/y'\n")
    fine = tmp_path / "fine.py"
    fine.write_text(f"D = '{home}/<name>/code'\nE = '{home}/'\nF = 'site{home}/page'\nG = 'my_home/x'\nH = '~/code'\n")
    assert scan([bad, fine], HOME_PATH_PATTERN, tmp_path) == ["bad.py:1", "bad.py:2", "bad.py:3"]


def test_home_path_scan_still_finds_users_paths_after_a_colon_or_slashes(tmp_path):
    users = "/" + "Users"
    path = tmp_path / "paths.txt"
    path.write_text(f"C:{users}/name/x\nfile://{users}/name/x\nsrc{users}/not-absolute\n")
    assert scan([path], HOME_PATH_PATTERN, tmp_path) == ["paths.txt:1", "paths.txt:2"]
