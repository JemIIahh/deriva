"""Fine-tuning data preparation: Global-MMLU rows -> closed-form candidates -> final records.

The pipeline is: fetch (raw rows from the Hugging Face datasets-server) -> prepare (this module
builds the closed-form prompts and splits by source item) -> CuratorKIT (dedup, gates, balance,
export; pipelines/ft_data.yaml) -> finalize (this module joins CuratorKIT's export back to the
ids, caps each language and writes train.jsonl / heldout.jsonl, and taskdev.jsonl when the
settings ask for it).

`taskdev.jsonl` holds the task items the dev split's alpha is chosen on. It is drawn from the items
that neither train nor heldout took: a third pass over the same seeded ranking, so the train and
heldout assignment (and their bytes) cannot move when it is switched on. In the code that holds
only up to CuratorKIT: after that it relies on CuratorKIT 0.1.22 keeping the first sample of a
MinHash duplicate group and passing unlisted strata through its samplers, which the unit tests do
not exercise (they stand in for CuratorKIT). The guard is the sha256 of the real train.jsonl and
heldout.jsonl, to be re-checked after any CuratorKIT upgrade.

CuratorKIT's exporters write only the `messages` of a sample, never its id, language or gold, and
its gates cannot build a prompt, so the two thin steps around it live here, in pure Python with
no CuratorKIT import (they must run in the core venv).

A candidate is `{id, lang, split, stratum, instruction, output, gold, item, subject, rank}`:
  * `instruction` is the whole user turn (question, lettered options, one-line instruction),
    `output` the bare answer letter; CuratorKIT reads these as an Alpaca row and the extra fields
    ride along as metadata,
  * `item` is the source item (`high_school_geography/test/12`), identical in every language, so a
    question and its translations always land in the same split,
  * `rank` is the position of the item's group (items that share a passage form one group) in a
    seeded global ranking; finalize keeps the best-ranked rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

import yaml

from msd import ftdata
from msd.config import LANGS

SOURCE_NAME = "CohereLabs/Global-MMLU (fine-tune subset)"
SOURCE_LANGS: tuple[str, ...] = ("en", "hi", "ar", "sw")  # study languages Global-MMLU covers
DATASET = "CohereLabs/Global-MMLU"
SPLIT = "test"
OPTION_FIELDS = ("option_a", "option_b", "option_c", "option_d")
RAW_FIELDS = ("sample_id", "subject", "question", *OPTION_FIELDS, "answer")
ANSWERS = ("A", "B", "C", "D")
TASKDEV_FILE = ftdata.TASKDEV_FILE  # the file the evaluation orchestrator reads for split dev (evalrun.TASK_FILES)
SPLIT_ORDER: dict[str, int] = {"train": 0, "heldout": 1, "taskdev": 2}  # candidate order: dedup keeps the earlier row
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_API_URL = "https://huggingface.co/api/datasets"

# The fixed one-line instruction that ends every prompt. These are template strings written for
# this task (not dataset content); a native speaker should read them before release. The reply
# the instruction asks for is the Latin letter, because the option labels are Latin letters in
# every language and score_task reads a Latin A-E.
INSTRUCTION_LINES: dict[str, str] = {
    "en": "Answer with only the letter of the correct option.",
    "hi": "केवल सही विकल्प का अक्षर लिखकर उत्तर दें।",
    "ar": "أجب بحرف الخيار الصحيح فقط.",
    "sw": "Jibu kwa herufi ya chaguo sahihi pekee.",
}

_SETTINGS_KEYS = ("split_seed", "heldout_pool", "train_pool", "heldout_cap", "train_cap")
_TASKDEV_KEYS = ("taskdev_pool", "taskdev_cap")  # optional: the dev split's own task items


# --- settings (kept in the pipeline YAML, `extra:`) -------------------------------------------


def load_settings(pipeline_yaml: str | Path) -> dict[str, Any]:
    """`extra.harmless_subjects` plus the seed, pool sizes and caps, validated.

    `taskdev_pool` and `taskdev_cap` are optional; when present they are returned too.
    """
    with Path(pipeline_yaml).open("r", encoding="utf-8") as fh:
        extra = (yaml.safe_load(fh) or {}).get("extra") or {}
    subjects = ftdata.load_allowed_subjects(pipeline_yaml)
    missing = [k for k in _SETTINGS_KEYS if not isinstance(extra.get(k), int) or isinstance(extra.get(k), bool)]
    if missing:
        raise ValueError(f"{pipeline_yaml}: extra must hold integers for {', '.join(missing)}")
    settings = {k: extra[k] for k in _SETTINGS_KEYS}
    settings["harmless_subjects"] = subjects
    if any(k in extra for k in _TASKDEV_KEYS):  # optional, but both or neither
        bad = [k for k in _TASKDEV_KEYS if not isinstance(extra.get(k), int) or isinstance(extra.get(k), bool) or extra[k] <= 0]
        if bad:
            raise ValueError(f"{pipeline_yaml}: extra must hold positive integers for both taskdev_pool and taskdev_cap, or neither (check {', '.join(bad)})")
        if extra["taskdev_pool"] < extra["taskdev_cap"]:
            raise ValueError(f"{pipeline_yaml}: taskdev_pool ({extra['taskdev_pool']}) must be at least taskdev_cap ({extra['taskdev_cap']})")
        settings.update({k: extra[k] for k in _TASKDEV_KEYS})
    return settings


# --- prompts ------------------------------------------------------------------------------------


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def build_user_turn(lang: str, question: str, options: Sequence[str]) -> str:
    """Question, then `A. ...` to `D. ...` lines, then the one-line instruction in the language."""
    lines = [question, *(f"{letter}. {text}" for letter, text in zip(ANSWERS, options)), INSTRUCTION_LINES[lang]]
    return "\n".join(lines)


def convert_row(lang: str, row: Mapping[str, Any], allowed_subjects: Collection[str]) -> tuple[dict | None, str | None]:
    """(candidate without split/rank, None) for a usable row, else (None, reason it was dropped)."""
    if row.get("subject") not in allowed_subjects:
        return None, "subject_not_allowed"
    answer = row.get("answer")
    if answer not in ANSWERS:
        return None, "bad_answer"
    question = "\n".join(line for line in (_one_line(l) for l in str(row.get("question", "")).splitlines()) if line)
    options = [_one_line(row.get(f, "")) for f in OPTION_FIELDS]
    if not question or not all(options):
        return None, "empty_field"
    if ftdata.option_letters(question):
        return None, "option_line_in_question"
    if len({o.casefold() for o in options}) < len(options):
        return None, "duplicate_options"
    user = build_user_turn(lang, question, options)
    if ftdata.script_problem(lang, user):
        return None, "wrong_script"
    sample_id = str(row["sample_id"])
    return (
        {
            "id": f"{lang}:{sample_id}",
            "lang": lang,
            "instruction": user,
            "output": answer,
            "gold": answer,
            "item": sample_id,
            "subject": row["subject"],
        },
        None,
    )


def to_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """The public record shape: exactly id, lang, messages, gold."""
    return {
        "id": candidate["id"],
        "lang": candidate["lang"],
        "messages": [
            {"role": "user", "content": candidate["instruction"]},
            {"role": "assistant", "content": candidate["output"]},
        ],
        "gold": candidate["gold"],
    }


# --- splitting by source item ------------------------------------------------------------------


def item_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """sample_id -> group id for questions that share a passage; the group is named by its smallest id.

    Only items that really share a passage with another item get an entry. Several Global-MMLU
    history questions refer to one source text, so they must stay together in one split.
    """
    members: dict[str, list[str]] = {}
    for row in rows:
        passage = ftdata.passage_of(str(row.get("question", "")))
        if passage:
            members.setdefault(passage, []).append(str(row["sample_id"]))
    return {i: min(ids) for ids in members.values() if len(ids) > 1 for i in ids}


def global_ranking(ids: Iterable[str], seed: int) -> list[str]:
    """The ids in a seeded pseudo-random order that does not depend on the order they arrive in."""
    return sorted(set(ids), key=lambda i: hashlib.sha256(f"{seed}|{i}".encode()).hexdigest())


def assign_splits(
    item_ids: Iterable[str],
    seed: int,
    heldout_pool: int,
    train_pool: int,
    group_of: Mapping[str, str] | None = None,
    taskdev_pool: int = 0,
) -> dict[str, str]:
    """item -> "heldout", "train" or "taskdev", one assignment for every language.

    Items are taken a group at a time (a group is the items of `group_of` sharing a value, every
    other item its own group) in a seeded global ranking: heldout fills up to `heldout_pool`
    items, then train up to `train_pool`, then taskdev up to `taskdev_pool` from what both left.
    A group that does not fit the room left is skipped for that split, never cut, so a passage
    never straddles the split and a pool is never overfilled. A small universe gives heldout
    first, then whatever is left to train, then to taskdev (nothing is padded); items beyond all
    pools are unused. The third pass only reads what the first two left, so `taskdev_pool` never
    changes the heldout or train assignment (0, the default, switches it off).
    """
    group_of = group_of or {}
    members: dict[str, list[str]] = {}
    for item in set(item_ids):
        members.setdefault(group_of.get(item, item), []).append(item)
    split: dict[str, str] = {}
    taken = {"heldout": 0, "train": 0, "taskdev": 0}
    room = {"heldout": heldout_pool, "train": train_pool, "taskdev": taskdev_pool}
    pending = global_ranking(members, seed)
    for name in ("heldout", "train", "taskdev"):
        left = []
        for group in pending:
            if taken[name] + len(members[group]) <= room[name]:
                split.update({item: name for item in members[group]})
                taken[name] += len(members[group])
            else:
                left.append(group)
        pending = left
    return split


# --- prepare -----------------------------------------------------------------------------------


def raw_path(raw_dir: str | Path, lang: str) -> Path:
    return Path(raw_dir) / f"{lang}_{SPLIT}.jsonl"


def _read_raw(raw_dir: str | Path, lang: str) -> list[dict[str, Any]]:
    path = raw_path(raw_dir, lang)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run: python scripts/build_ft_data.py fetch --langs {lang}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(
    raw_dir: str | Path,
    langs: Sequence[str],
    allowed_subjects: Collection[str],
    *,
    seed: int,
    heldout_pool: int,
    train_pool: int,
    taskdev_pool: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Candidates for every language (train rows first, then heldout, then taskdev) and a summary of the drops."""
    raw = {lang: _read_raw(raw_dir, lang) for lang in langs}
    allowed_rows = {lang: [r for r in rows if r.get("subject") in allowed_subjects] for lang, rows in raw.items()}
    universe = {r["sample_id"] for rows in allowed_rows.values() for r in rows}
    reference = "en" if "en" in allowed_rows else langs[0]  # questions are parallel across languages
    group_of = item_groups(allowed_rows[reference])
    split_of = assign_splits(universe, seed, heldout_pool, train_pool, group_of, taskdev_pool)
    group_rank = {g: n for n, g in enumerate(global_ranking({group_of.get(i, i) for i in universe}, seed))}
    rank_of = {item: group_rank[group_of.get(item, item)] for item in universe}

    candidates: list[dict[str, Any]] = []
    dropped: dict[str, dict[str, int]] = {}
    outside_pool: dict[str, int] = {}
    for lang in langs:
        reasons: Counter[str] = Counter()
        unused = 0
        for row in raw[lang]:
            cand, why = convert_row(lang, row, allowed_subjects)
            if cand is None:
                reasons[why] += 1
            elif cand["item"] not in split_of:
                unused += 1
            else:
                cand["split"] = split_of[cand["item"]]
                cand["stratum"] = f"{cand['split']}:{lang}"  # what CuratorKIT's stratified sampler balances on
                cand["rank"] = rank_of[cand["item"]]
                candidates.append(cand)
        dropped[lang] = dict(reasons)
        outside_pool[lang] = unused

    order = {lang: n for n, lang in enumerate(langs)}
    candidates.sort(key=lambda c: (SPLIT_ORDER[c["split"]], order[c["lang"]], c["rank"]))
    summary = {
        "source": SOURCE_NAME,
        "seed": seed,
        "heldout_pool": heldout_pool,
        "train_pool": train_pool,
        **({"taskdev_pool": taskdev_pool} if taskdev_pool else {}),
        "allowed_subjects": sorted(allowed_subjects),
        "raw_rows": {lang: len(rows) for lang, rows in raw.items()},
        "universe_items": len(universe),
        "passage_groups": len(set(group_of.values())),
        "items_in_passage_groups": len(group_of),
        "dropped": dropped,
        "outside_pool": outside_pool,
        "candidates": _by_split_lang(candidates, bool(taskdev_pool)),
    }
    return candidates, summary


def _by_split_lang(rows: Iterable[Mapping[str, Any]], with_taskdev: bool = False) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"train": {}, "heldout": {}, **({"taskdev": {}} if with_taskdev else {})}
    for row in rows:
        out[row["split"]][row["lang"]] = out[row["split"]].get(row["lang"], 0) + 1
    return {split: {lang: out[split][lang] for lang in LANGS if lang in out[split]} for split in out}


# --- finalize: join CuratorKIT's export back to ids -------------------------------------------


def join_exported(candidates: Sequence[dict[str, Any]], exported: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The candidate behind each exported `{messages: [...]}` line, matched on (user, assistant) text.

    CuratorKIT keeps the first of any duplicates, so when two candidates share text the first
    one is the survivor. A line that matches nothing, or the same candidate twice, is an error.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in candidates:
        index.setdefault((cand["instruction"], cand["output"]), cand)
    used: set[str] = set()
    survivors: list[dict[str, Any]] = []
    for line in exported:
        user, assistant = line["messages"][0]["content"], line["messages"][1]["content"]
        cand = index.get((user, assistant))
        if cand is None:
            raise ValueError(f"exported line has no candidate: {ftdata.clip(user)!r}")
        if cand["id"] in used:
            raise ValueError(f"candidate {cand['id']} was exported twice")
        used.add(cand["id"])
        survivors.append(cand)
    return survivors


def cap_by_rank(survivors: Iterable[dict[str, Any]], *, train_cap: int, heldout_cap: int) -> list[dict[str, Any]]:
    """At most `train_cap` / `heldout_cap` rows per language, best rank first; nothing is padded."""
    caps = {"train": train_cap, "heldout": heldout_cap}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cand in survivors:
        groups.setdefault((cand["split"], cand["lang"]), []).append(cand)
    kept: list[dict[str, Any]] = []
    for split in ("train", "heldout"):
        for lang in LANGS:
            rows = sorted(groups.get((split, lang), []), key=lambda c: (c["rank"], c["id"]))
            kept.extend(rows[: caps[split]])
    return kept


def cap_taskdev(survivors: Iterable[dict[str, Any]], *, cap: int) -> list[dict[str, Any]]:
    """At most `cap` taskdev rows per language, the same items in every language where possible.

    Items that survived in every language come first, best rank first (the rank belongs to the
    item, so every language picks the same ones). A language that has fewer than `cap` of those
    adds its own best-ranked remaining rows; nothing is padded, so a language can end up short.
    """
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for cand in survivors:
        if cand["split"] == "taskdev":
            by_lang.setdefault(cand["lang"], []).append(cand)
    ordered = {
        lang: sorted(by_lang[lang], key=lambda c: (c["rank"], c["id"])) for lang in LANGS if lang in by_lang
    }
    if not ordered:
        return []
    shared = set.intersection(*({c["item"] for c in rows} for rows in ordered.values()))
    kept: list[dict[str, Any]] = []
    for rows in ordered.values():
        chosen = [c for c in rows if c["item"] in shared][:cap]
        taken = {c["id"] for c in chosen}
        chosen += [c for c in rows if c["id"] not in taken][: max(0, cap - len(chosen))]
        kept.extend(sorted(chosen, key=lambda c: (c["rank"], c["id"])))
    return kept


def _removed(before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {
        split: {lang: n - after[split].get(lang, 0) for lang, n in before[split].items() if n - after[split].get(lang, 0)}
        for split in before
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize(
    candidates: Sequence[dict[str, Any]],
    exported: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    train_cap: int,
    heldout_cap: int,
    taskdev_cap: int | None = None,
    rejected: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write `train.jsonl` and `heldout.jsonl` (and `taskdev.jsonl`) under `out_dir`; return the counts.

    `taskdev_cap` is given exactly when `prepare` was run with a taskdev pool: taskdev candidates
    without a cap, or a cap without candidates, mean the settings and the candidates file do not
    belong together and are refused rather than guessed at.
    """
    has_dev = any(c["split"] == "taskdev" for c in candidates)
    if has_dev and taskdev_cap is None:
        raise ValueError("the candidates hold taskdev rows but taskdev_cap is not set (add taskdev_pool and taskdev_cap to extra:)")
    if taskdev_cap is not None and not has_dev:
        raise ValueError("taskdev_cap is set but the candidates hold no taskdev rows (prepare was run without taskdev_pool, or no unused items were left); run prepare again with the same settings")
    survivors = join_exported(candidates, exported)
    kept = cap_by_rank(survivors, train_cap=train_cap, heldout_cap=heldout_cap)
    if has_dev:
        kept = kept + cap_taskdev(survivors, cap=taskdev_cap)
    out = Path(out_dir)
    for split in ("train", "heldout", *(("taskdev",) if has_dev else ())):
        _write_jsonl(out / f"{split}.jsonl", (to_record(c) for c in kept if c["split"] == split))

    gates = Counter(f"{r.get('rejecting_step')}: {str(r.get('rejection_reason', '')).split(':', 1)[0]}" for r in rejected)
    gate_langs = Counter((r.get("metadata") or {}).get("lang", "unknown") for r in rejected)
    cands_n, surv_n, final_n = (_by_split_lang(rows, has_dev) for rows in (candidates, survivors, kept))
    return {
        "candidates": cands_n,
        "curatorkit_survivors": surv_n,
        "removed_by_curatorkit": _removed(cands_n, surv_n),
        "gate_rejections": dict(gates),
        "gate_rejections_by_lang": dict(gate_langs),
        "capped_off": _removed(surv_n, final_n),
        "final": final_n,
        "caps": {"train": train_cap, "heldout": heldout_cap, **({"taskdev": taskdev_cap} if has_dev else {})},
    }


def read_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_candidates(path: str | Path, candidates: Iterable[Mapping[str, Any]]) -> None:
    _write_jsonl(Path(path), candidates)


# --- fetch -------------------------------------------------------------------------------------

GetPage = Callable[[str, int, int], Mapping[str, Any]]


class RateLimited(OSError):
    """HTTP 429 from the datasets-server; `retry_after` is how long it asked us to wait (seconds)."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after:g}s")
        self.retry_after = retry_after


def http_get_page(lang: str, offset: int, length: int, timeout: float = 60.0) -> Mapping[str, Any]:
    """One page of rows from the datasets-server (public, no token)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": lang, "split": SPLIT, "offset": offset, "length": length}
    )
    try:
        with urllib.request.urlopen(f"{HF_ROWS_URL}?{query}", timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimited(float(exc.headers.get("Retry-After") or 30)) from exc
        raise


def http_get_json(url: str, timeout: float = 30.0) -> Mapping[str, Any]:
    """A public JSON document (no token)."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        return json.load(resp)


def dataset_revision(*, get_json: Callable[[str], Mapping[str, Any]] = http_get_json) -> dict[str, str] | None:
    """The dataset's current commit on the Hub: {"revision": sha, "last_modified": ISO time}.

    The rows endpoint does not say which commit it served, so provenance is the head commit seen
    at fetch time (its last-modified date tells whether it predates the fetch). None when the
    Hub API cannot be reached or gives no sha: the revision is recorded when cheap, never required.
    """
    try:
        payload = get_json(f"{HF_API_URL}/{DATASET}")
    except Exception:  # noqa: BLE001 (offline, HTTP error or bad JSON: provenance is best effort)
        return None
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        return None
    return {"revision": sha, "last_modified": str(payload.get("lastModified") or "")}


def fetch_language(
    lang: str,
    allowed_subjects: Collection[str],
    *,
    get_page: GetPage = http_get_page,
    page_size: int = 100,
    workers: int = 3,
    attempts: int = 8,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Every row of the language's test split whose subject is allowed, in dataset order.

    Pages are fetched in parallel and retried with a growing pause; a page that returns fewer
    rows than the dataset holds there is an error, never a silent gap.
    """

    def one(offset: int) -> tuple[int, list[Mapping[str, Any]], int]:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                payload = get_page(lang, offset, page_size)
                return offset, [r["row"] for r in payload["rows"]], int(payload["num_rows_total"])
            except Exception as exc:  # noqa: BLE001 (network and JSON errors alike are retried)
                last = exc
                sleep(getattr(exc, "retry_after", None) or min(2.0 * 2**attempt, 30.0))
        raise RuntimeError(f"{lang}: page at offset {offset} failed after {attempts} attempts: {last!r}")

    first_offset, first_rows, total = one(0)
    pages = {first_offset: first_rows}
    offsets = list(range(page_size, total, page_size))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for offset, rows, _ in pool.map(one, offsets):
            pages[offset] = rows
    kept: list[dict[str, Any]] = []
    seen = 0
    for offset in sorted(pages):
        rows = pages[offset]
        if len(rows) != min(page_size, total - offset):
            raise RuntimeError(f"{lang}: page at offset {offset} returned {len(rows)} rows, expected {min(page_size, total - offset)}")
        seen += len(rows)
        kept.extend({k: r[k] for k in RAW_FIELDS} for r in rows if r.get("subject") in allowed_subjects)
    if seen != total:
        raise RuntimeError(f"{lang}: read {seen} rows, the dataset reports {total}")
    return kept


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
