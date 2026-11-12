"""A frozen fixture test set for tests that generate the report split (only a frozen set may: msd.genrun).

`freeze_fixture(testset_dir)` fills every `<lang>/{dev,report}.jsonl` the test did not build with one placeholder row,
then writes MANIFEST.json the way a freeze writes it: `msd.testset.freeze(dry=True)` hashes the files, and its
MANIFEST.dry.json is renamed to the real name. A real freeze also needs a build record; a fixture has none, so this is
the shortcut. `msd.testset.verify_freeze` accepts the result. Returns the sha256 of MANIFEST.json.
"""

from __future__ import annotations

from pathlib import Path

from msd import testset
from msd.config import LANGS
from msd.schema import Prompt, write_jsonl


def freeze_fixture(testset_dir: str | Path) -> str:
    base = Path(testset_dir)
    for lang in LANGS:
        for split in ("dev", "report"):
            path = base / lang / f"{split}.jsonl"
            if not path.is_file():
                write_jsonl(path, [Prompt(f"{lang}-{split}-filler-0", lang, "harmful", f"filler {lang} {split}", split,
                                          "fixture", "fixture")])
    dry = testset.freeze(base, dry=True)
    dry.path.replace(base / testset.MANIFEST_NAME)
    assert testset.verify_freeze(base) == [], "the fixture freeze must verify"
    return testset.file_sha256(base / testset.MANIFEST_NAME)
