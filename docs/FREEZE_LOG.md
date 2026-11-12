# Freeze log of the test set

This file records every real freeze of the safety test set (`data/testset/`). The test set itself is not in git
(`data/` is git-ignored). Its `MANIFEST.json` lists the sha256 and the row count of each of the ten files
`<lang>/{dev,report}.jsonl`, and the sha256 of that `MANIFEST.json` is the one number that names the frozen set.
Writing it here, in a tracked file, before any report-split run, is the proof that the set did not change after the
results were seen.

The article quotes the hash and the date from this table (`article/FILL_IN.md`: "freeze hash of the test set" and
"freeze date"). Every report-split generation also records the same hash in its `<lang>.gens.meta.json` and
`<lang>.eval.json` (key `testset_manifest_sha256`), so a result can be matched to its freeze.

## How to record a freeze

1. Freeze. Either build and freeze in one go, or freeze what is already built:

   ```
   envs/translate/bin/python scripts/build_testset.py --langs all --freeze
   envs/core/bin/python scripts/build_testset.py --freeze-only
   ```

   (`docs/RUNBOOK.md`, step 2, says which environment builds the translations.) On success the script prints two
   lines like these:

   ```
   FROZE the test set: data/testset/MANIFEST.json
   MANIFEST.json sha256: <64 hex characters>
   ```

   and then "append today's date and this sha256 to docs/FREEZE_LOG.md". The script never writes a tracked file
   itself.
2. Check it: `envs/core/bin/python scripts/verify_freeze.py`. It must print `OK ... matches MANIFEST.json` and exit 0.
   A real check needs all ten files. `--dry` checks the dry manifest of a partial build and does NOT count.
3. Check the hash a second way (optional): `shasum -a 256 data/testset/MANIFEST.json` on the Mac. It hashes the
   bytes of the file, which is what the script printed (`msd.testset.file_sha256`), so the two must be equal.
4. Add one row to the table below: the date (UTC), the full 64-character hash, the command, the reason, and the
   result of step 2. Commit this file.

A set that is already frozen cannot be built or frozen again by accident: the script refuses (exit 3) until you
pass `--refreeze`. Refreeze only before any report-split result exists. If you must refreeze later, add a new row
that says why, and run every report-split cell again: the generation refuses to continue a gens file made under
another freeze (`testset_manifest_sha256` is part of its setup).

## Freezes

| Date (UTC) | MANIFEST.json sha256 | Command | Reason | verify_freeze.py |
|---|---|---|---|---|
