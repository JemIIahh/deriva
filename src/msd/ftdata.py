"""Validation of the fine-tuning data: closed-form, harmless, refusal-free, balanced and leak-free.

A record is `{id, lang, messages: [{role: user, content}, {role: assistant, content}], gold}`.
The user turn holds the question (and, for multiple choice, lettered options `A. ...` plus a
one-line instruction). The assistant turn is EXACTLY the bare answer: one letter A-E, or a short
extracted span. `msd.scoring.score_task` scores the study by exact match with no cue stripping for
free-form gold, so a training target with an "Answer:" prefix or an explanation would teach the
model a format the scorer cannot read.

`validate` returns a `Report` of `Problem`s, each tagged with a rule class (`RULES`). The command
line (`scripts/validate_ft_data.py`) maps a single failing class to its own exit code
(`EXIT_CODES`), several failing classes to 1, and unreadable input to 2.

Rules, by class:
  format        exact keys, roles user then assistant only, non-empty text, valid language code,
                gold present, the id starts with `<lang>:` (its own language code), no duplicate
                ids (inside a split or across the two)
  refusal       no refusal-style assistant turn (REFUSAL_PATTERNS, every list applied to every turn)
  answer        the assistant turn is the bare gold answer and equals gold under score_task;
                multiple-choice gold is an upper-case letter that the prompt lists; free-form gold
                is a short span with no sign, percent mark or trailing punctuation to be dropped
                by normalize_answer ("-3" and "3" are the same answer to the scorer, so signed and
                percent gold stay out of the data)
  overlap       train and heldout share no prompt (norm_text of the full user turn), no source
                item (the id without its language prefix, so a question and its translation count
                as one), no passage (the text above the question line, several MMLU questions
                share one), and neither shares a prompt with the safety dev or report splits.
                All of this compares text of ONE language at a time: a Hindi passage in train and
                its English original in heldout are not compared here. That cross-lingual case is
                closed at prepare time (msd.ftprep groups every language's version of an item and
                of a passage through the English item and passage, then assigns whole groups to
                one split), so the validator only has to see same-language text.
  imbalance     a language's train count is more than 20% off the mean of the expected languages
  language      the question is written in the script of its language code (Devanagari for hi,
                Arabic for ar, Latin for en / sw / hinglish); en versus sw is NOT told apart
  missing_lang  an expected language has no train rows or no heldout rows
  subject       with an allowlist, every item belongs to a harmless-subject (Global-MMLU ids are
                `<lang>:<subject>/<split>/<n>`, so the subject is in the id). The command line
                turns this on by default (the repository's pipelines/ft_data.yaml) and only
                `--no-subject-check` skips it; the report always says which it was.
  safety_empty  a language the safety test set lists as covered has no dev prompts or no report
                prompts, which would make the leakage check pass without comparing anything
  taskdev_overlap
                the optional dev task file (`taskdev.jsonl`, exit code 19: the 10-18 range was
                full) shares a prompt (norm_text of the full user turn), a source item (any
                language: `hi:x/test/3` in taskdev and `en:x/test/3` in train are one item) or a
                passage with train or heldout. Its other problems use the ordinary classes with
                where="taskdev": format, refusal, answer, language and subject as everywhere,
                overlap for a repeated prompt inside it or a prompt equal to a safety prompt,
                imbalance for an uneven count per language and missing_lang for an expected
                language with no rows.

taskdev is optional. `validate(taskdev=None)` (a missing file) leaves the report's `taskdev_built`
False and render prints "taskdev: not built"; the command line turns that into an error only with
`--require-taskdev`. The dev items of different languages should be the same items; when they are
not the report carries a warning, not a failure (a language can lose an item to a language-specific
gate).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

import yaml

from msd.config import LANGS
from msd.schema import Prompt, read_jsonl
from msd.scoring import normalize_answer, score_task
from msd.splits import LeakageError, assert_no_leakage, norm_text

REQUIRED_KEYS: tuple[str, ...] = ("id", "lang", "messages", "gold")
RULES: tuple[str, ...] = (
    "format",
    "refusal",
    "answer",
    "overlap",
    "imbalance",
    "language",
    "missing_lang",
    "subject",
    "safety_empty",
    "taskdev_overlap",
)
# 0 ok, 1 several classes failed, 2 bad arguments or unreadable input (as in build_testset.py).
EXIT_CODES: dict[str, int] = {
    "format": 10,
    "refusal": 11,
    "answer": 12,
    "overlap": 13,
    "imbalance": 14,
    "language": 15,
    "missing_lang": 16,
    "subject": 17,
    "safety_empty": 18,
    "taskdev_overlap": 19,
}
TASKDEV_FILE = "taskdev.jsonl"  # next to heldout.jsonl; evalrun.TASK_FILES["dev"] reads the same name
IMBALANCE_TOLERANCE = 0.20
MAX_ANSWER_WORDS = 8
MAX_ANSWER_CHARS = 80
CLIP = 100  # a quoted item is cut to this many characters in every message

# --- refusal-style wording ---------------------------------------------------------------------
#
# Patterns are matched against norm_text(assistant turn): NFKC, casefolded, Arabic tashkeel and
# tatweel dropped, danda mapped to ".". Every list is applied to every assistant turn whatever
# the record's language code, because a refusal in a Hindi record is often English. Patterns are
# deliberately about refusal phrasing, not topics: a harmless answer never needs them.

REFUSAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "en": (
        r"\bi\s+(?:can(?:'|’)?t|cannot|can\s+not|won(?:'|’)?t|will\s+not|am\s+unable\s+to"
        r"|am\s+not\s+able\s+to|must\s+decline|have\s+to\s+decline|refuse\s+to|decline\s+to)\b",
        r"\bi(?:'|’)m\s+(?:sorry|unable|not\s+able|afraid|not\s+going\s+to)\b",
        r"\b(?:sorry|apologi[sz]e|apologies)\b",
        r"\bas\s+an\s+ai\b",
        r"\b(?:unable|not\s+able)\s+to\s+(?:help|assist|comply|provide|answer)\b",
        r"\b(?:can(?:'|’)?t|cannot)\s+(?:help|assist|comply|provide|fulfi?l+|do\s+that|answer)\b",
        r"\bagainst\s+(?:my|our)\s+(?:guidelines|policy|policies|programming)\b",
    ),
    "hi": (
        r"मैं[^.!?\n]{0,80}?नहीं[^.!?\n]{0,25}?(?:सकता|सकती|सकते|पाऊं|पाऊँ|चाहता|चाहती)",
        r"नहीं\s+कर\s+(?:सकता|सकती|सकते)",
        r"(?:क्षमा|माफ[़]?|माफी)\s*(?:कीजिए|कीजिये|करें|करिए|कर\s*दें)",
        r"खेद\s+(?:है|के\s+साथ)",
        r"(?:इनकार|मना)\s+(?:करता|करती)",
    ),
    "hinglish": (
        r"\b(?:nahi|nahin|nhi)\s+kar\s+sak(?:ta|ti|te)\b",
        r"\b(?:main|mai|mein|mujhe)\b[^.!?\n]{0,60}?\b(?:nahi|nahin|nhi)\b[^.!?\n]{0,25}?\bsak(?:ta|ti|te)\b",
        r"\b(?:maaf|maf|kshama)\s+(?:kijiye|kijiyega|kijiyo|karein|kare|karna|karo|kariye)\b",
        r"\b(?:khed|afsos)\s+hai\b",
        r"\bmain\s+(?:madad|help)\s+(?:nahi|nahin|nhi)\b",
    ),
    "ar": (
        r"لا\s+[أإا]ستطيع",
        r"لا\s+يمكنني",
        r"لا\s+[أإا]قدر",
        r"لن\s+[أإا]تمكن",
        r"لست\s+قادر",
        r"لا\s+يسعني",
        r"عذر[اً]",
        r"[آأإا]سف\b",
        r"[أإا]عتذر",
        r"[أإا]رفض\b",
    ),
    "sw": (
        r"\bsamahani\b",
        r"\bsiwezi\b",
        r"\bsitaweza\b",
        r"\bsiruhusiwi\b",
        r"\bnasikitika\b",
        r"\bsina\s+(?:uwezo|ruhusa)\b",
    ),
}
_REFUSAL_RE: dict[str, tuple[re.Pattern[str], ...]] = {
    lang: tuple(re.compile(p) for p in patterns) for lang, patterns in REFUSAL_PATTERNS.items()
}


def refusal_match(text: str) -> tuple[str, str] | None:
    """(language list, pattern) of the first refusal-style pattern found in `text`, else None."""
    normed = norm_text(text).replace("’", "'")
    for lang, patterns in _REFUSAL_RE.items():
        for pattern in patterns:
            if pattern.search(normed):
                return lang, pattern.pattern
    return None


# --- answers ------------------------------------------------------------------------------------

_OPTION_LINE = re.compile(r"(?m)^[ \t]*([A-E])[.)][ \t]+\S")
_OPTION_A = re.compile(r"(?m)^[ \t]*A[.)][ \t]")
_OPTION_LABEL = re.compile(r"(?m)^[ \t]*[A-E][.)][ \t]*")
_LETTER = re.compile(r"[A-E]")
_ANY_LETTER_GOLD = re.compile(r"[A-Ea-e]")  # what score_task treats as multiple choice
_SIGNED_OR_PERCENT = re.compile(
    r"(?<!\w)[-+−–—]\s*\d|%|٪|\bper\s?cent\b|\bpercent\b|प्रतिशत|بالمئة|بالمائة|في\s+الم[ئا]ة|\basilimia\b",
    re.IGNORECASE,
)
_CUE_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|option|choice|reply|response)\b"
    r"|^\s*(?:उत्तर|जवाब|विकल्प)"
    r"|^\s*(?:الإجابة|الاجابة|الجواب|الخيار)"
    r"|^\s*(?:jibu|chaguo|jawabu|jawab)\b",
    re.IGNORECASE,
)
_TERMINAL = ".!?।॥؟。"


def clip(text: str, n: int = CLIP) -> str:
    """At most `n` characters of `text`, for messages that quote a record."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def option_letters(user_turn: str) -> list[str]:
    """The option letters listed in a prompt, one per `A. ...` line, in order."""
    return _OPTION_LINE.findall(user_turn)


def question_part(user_turn: str) -> str:
    """The prompt up to its first `A. ...` option line (the whole prompt when it lists none)."""
    match = _OPTION_A.search(user_turn)
    return user_turn[: match.start()].strip() if match else user_turn.strip()


MIN_PASSAGE_CHARS = 80


def passage_of(user_turn: str) -> str | None:
    """The shared text a question refers to: its question part minus the last (question) line.

    Global-MMLU history items put a long source text above the question, and several questions
    share one source. None when the question part is a single line or what precedes its last
    line is shorter than MIN_PASSAGE_CHARS (too short to be a passage worth tracking).
    """
    lines = [ln.strip() for ln in question_part(user_turn).split("\n") if ln.strip()]
    if len(lines) < 2:
        return None
    passage = "\n".join(lines[:-1])
    return passage if len(passage) >= MIN_PASSAGE_CHARS else None


def item_key(record_id: str) -> str:
    """The source item an id points at: the id without its `<lang>:` prefix."""
    return record_id.split(":", 1)[1] if ":" in record_id else record_id


def item_subject(record_id: str) -> str:
    """The subject of a Global-MMLU style item key (`high_school_geography/test/12`)."""
    return item_key(record_id).split("/", 1)[0]


def gold_unsafe_for_scoring(gold: str) -> str | None:
    """Why `gold` cannot be used as free-form gold, or None.

    normalize_answer turns punctuation into a gap, so "-3" and "3" (and "50%" and "50") are the
    same answer to the scorer. Such items stay out of the data rather than change the scorer.
    """
    if _SIGNED_OR_PERCENT.search(gold):
        return f"gold {clip(gold)!r} carries a sign or percent mark, which normalize_answer drops"
    if not normalize_answer(gold):
        return f"gold {clip(gold)!r} normalises to empty text"
    return None


def answer_problems(user_turn: str, assistant: str, gold: str) -> list[str]:
    """Why the assistant turn is not the bare gold answer (empty list when it is)."""
    problems: list[str] = []
    letters = option_letters(user_turn)
    if _ANY_LETTER_GOLD.fullmatch(gold.strip()) or len(letters) >= 2:
        # multiple choice: score_task wants the committed letter, and a bare letter is the target
        if not _LETTER.fullmatch(gold):
            problems.append(f"gold {clip(gold)!r} is not a single upper-case letter A-E")
        elif gold not in letters:
            problems.append(
                f"gold option {gold} is not among the options the prompt lists ({', '.join(letters) or 'none'})"
            )
        if assistant != gold:
            problems.append(f"assistant turn {clip(assistant)!r} is not the bare gold letter {gold!r}")
        return problems

    unsafe = gold_unsafe_for_scoring(gold)
    if unsafe:
        problems.append(unsafe)
    if gold != gold.strip():
        problems.append(f"gold {clip(gold)!r} has leading or trailing whitespace")
    if not score_task(assistant, gold):
        problems.append(
            f"assistant turn {clip(assistant)!r} does not equal gold {clip(gold)!r} under score_task normalisation"
        )
    if assistant != assistant.strip() or "\n" in assistant:
        problems.append(f"assistant turn {clip(assistant)!r} has surrounding whitespace or a line break")
    if len(assistant.split()) > MAX_ANSWER_WORDS or len(assistant) > MAX_ANSWER_CHARS:
        problems.append(
            f"assistant turn is not a short span (over {MAX_ANSWER_WORDS} words or {MAX_ANSWER_CHARS} characters)"
        )
    if _CUE_PREFIX.search(assistant):
        problems.append(f"assistant turn {clip(assistant)!r} starts with an answer cue word")
    if assistant and assistant[-1] in _TERMINAL and not gold.endswith(assistant[-1]):
        problems.append(f"assistant turn {clip(assistant)!r} ends with punctuation that gold does not have")
    return problems


# --- language / script --------------------------------------------------------------------------

_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "devanagari": ((0x0900, 0x097F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
}
# (script, minimum share of the prompt's letters). Hindi and Arabic prompts may carry Latin
# terms and names; en / sw / hinglish are Latin-script throughout.
LANG_SCRIPT: dict[str, tuple[str, float]] = {
    "en": ("latin", 0.9),
    "sw": ("latin", 0.9),
    "hinglish": ("latin", 0.9),
    "hi": ("devanagari", 0.5),
    "ar": ("arabic", 0.5),
}


def _letters(text: str) -> list[str]:
    return [ch for ch in _OPTION_LABEL.sub("", text) if unicodedata.category(ch).startswith("L")]


def script_fraction(text: str, script: str) -> float:
    """Share of the letters in `text` that belong to `script` (0.0 when there are no letters).

    Option labels (`A.`) are ignored: they are Latin letters in every language.
    """
    ranges = _SCRIPT_RANGES[script]
    letters = _letters(text)
    if not letters:
        return 0.0
    hits = sum(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in letters)
    return hits / len(letters)


def script_problem(lang: str, user_turn: str) -> str | None:
    """Why the question of `user_turn` is not written in the script of `lang`, or None.

    Only the question part is measured: the one-line instruction is in the right language by
    construction and would hide an untranslated question. A question without letters is skipped.
    """
    question = question_part(user_turn)
    if not _letters(question):
        return None
    script, floor = LANG_SCRIPT[lang]
    share = script_fraction(question, script)
    if share < floor:
        return f"{lang} question is only {share:.0%} {script} letters (needs {floor:.0%})"
    return None


# --- the report ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    rule: str  # one of RULES
    where: str  # "train", "heldout", "safety" or "set"
    id: str | None
    message: str


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    expected_langs: tuple[str, ...] = LANGS
    missing_expected: list[str] = field(default_factory=list)
    not_covered: list[str] = field(default_factory=list)
    allowed_subjects: tuple[str, ...] | None = None  # None: the subject allowlist was NOT checked
    subject_rows_checked: int = 0
    safety_counts: dict[str, tuple[int, int]] | None = None  # lang -> (dev, report) prompts; None: not checked
    taskdev_built: bool = False  # False: no taskdev file was given (counts has no "taskdev" entry)
    missing_taskdev: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def failed_rules(self) -> list[str]:
        failed = {p.rule for p in self.problems}
        return [rule for rule in RULES if rule in failed]

    def exit_code(self) -> int:
        failed = self.failed_rules()
        if not failed:
            return 0
        return EXIT_CODES[failed[0]] if len(failed) == 1 else 1


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _structure(rec: Any) -> tuple[list[str], bool]:
    """(format problems, whether the fields the other rules read are usable)."""
    if not isinstance(rec, dict):
        return [f"record is not a JSON object ({type(rec).__name__})"], False
    problems: list[str] = []
    keys = set(rec)
    if keys != set(REQUIRED_KEYS):
        missing, extra = sorted(set(REQUIRED_KEYS) - keys), sorted(keys - set(REQUIRED_KEYS))
        problems.append(f"keys must be exactly {list(REQUIRED_KEYS)} (missing {missing}, unexpected {extra})")
    usable = True
    if "id" in rec and not _text(rec["id"]):
        problems.append("id must be non-empty text")
        usable = False
    if "lang" in rec and rec["lang"] not in LANGS:
        problems.append(f"lang {rec['lang']!r} is not one of {list(LANGS)}")
        usable = False
    if "gold" in rec and not _text(rec["gold"]):
        problems.append("gold must be non-empty text")
        usable = False
    messages = rec.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        problems.append("messages must be a list of exactly two turns: user, then assistant")
        usable = False
    else:
        for turn, role in zip(messages, ("user", "assistant")):
            if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
                problems.append(f"each message must be an object with exactly the keys role and content ({role} turn)")
                usable = False
            elif turn["role"] != role:
                problems.append(f"message roles must be user then assistant, found {turn['role']!r} where {role} belongs")
                usable = False
            elif not _text(turn["content"]):
                problems.append(f"{role} turn content must be non-empty text")
                usable = False
    return problems, usable and all(k in rec for k in REQUIRED_KEYS)


def read_jsonl_records(path: str | Path, where: str) -> tuple[list[Any], list[Problem]]:
    """Parsed lines of a JSONL file; a line that is not JSON becomes a format Problem."""
    records: list[Any] = []
    problems: list[Problem] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                problems.append(Problem("format", where, None, f"line {n}: not valid JSON ({exc.msg})"))
    return records, problems


def load_safety_texts(testset_dir: str | Path) -> tuple[list[str], list[str], dict[str, tuple[int, int]]]:
    """(dev texts, report texts, {language: (dev prompts, report prompts)}) from the test set.

    Reads `<testset_dir>/<lang>/{dev,report}.jsonl`. Every study-language directory found is read,
    so a partly built test set is checked as far as it exists; the per-language counts say what
    the check covered (and `validate` fails a language whose files are empty).
    """
    root = Path(testset_dir)
    dev: list[str] = []
    report: list[str] = []
    found: dict[str, tuple[int, int]] = {}
    for lang in LANGS:
        folder = root / lang
        if not folder.is_dir():
            continue
        sizes: list[int] = []
        for split, sink in (("dev", dev), ("report", report)):
            path = folder / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found (the {lang} test-set directory is incomplete)")
            texts = [p.text for p in read_jsonl(path, Prompt)]
            sink.extend(texts)
            sizes.append(len(texts))
        found[lang] = (sizes[0], sizes[1])
    if not found:
        raise FileNotFoundError(f"no language directory ({', '.join(LANGS)}) under {root}")
    return dev, report, found


def load_allowed_subjects(pipeline_yaml: str | Path) -> set[str]:
    """The harmless-subject allowlist recorded in the pipeline config (`extra.harmless_subjects`)."""
    with Path(pipeline_yaml).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    subjects = ((data or {}).get("extra") or {}).get("harmless_subjects")
    if not isinstance(subjects, list) or not subjects or not all(_text(s) for s in subjects):
        raise ValueError(f"{pipeline_yaml}: extra.harmless_subjects must be a non-empty list of subject names")
    return set(subjects)


def _user(rec: dict[str, Any]) -> str:
    return rec["messages"][0]["content"]


def _assistant(rec: dict[str, Any]) -> str:
    return rec["messages"][1]["content"]


def _leak_texts(records: Iterable[dict[str, Any]]) -> list[str]:
    """What a safety prompt could collide with: the whole user turn and its question part."""
    texts: list[str] = []
    for rec in records:
        user = _user(rec)
        texts.append(user)
        texts.append(question_part(user))
    return texts


def _counts(records: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        lang = rec.get("lang") if isinstance(rec, dict) else None
        if lang in LANGS:
            out[lang] = out.get(lang, 0) + 1
    return {lang: out[lang] for lang in LANGS if lang in out}


def _imbalanced(counts: dict[str, int], expected: Sequence[str], tolerance: float) -> list[str]:
    """Messages for expected languages whose count is more than `tolerance` off the mean count."""
    present = {lang: counts[lang] for lang in expected if counts.get(lang, 0) > 0}
    if len(present) < 2:
        return []
    mean = sum(present.values()) / len(present)
    return [
        f"{lang} has {n} rows, {abs(n - mean) / mean:.0%} {'above' if n > mean else 'below'} the mean of "
        f"{mean:.1f} across {', '.join(present)} (allowed {tolerance:.0%})"
        for lang, n in present.items()
        if abs(n - mean) / mean > tolerance
    ]


def _check_split(
    records: Sequence[Any],
    where: str,
    allowed_subjects: Collection[str] | None,
    problems: list[Problem],
) -> list[dict[str, Any]]:
    """Per-record rules; returns the structurally usable records for the cross-record rules.

    A usable record is one whose subject was read, so `len(result)` is the number of rows the
    subject allowlist was applied to when there is one.
    """
    usable: list[dict[str, Any]] = []
    for rec in records:
        bad, ok = _structure(rec)
        rid = rec.get("id") if isinstance(rec, dict) and isinstance(rec.get("id"), str) else None
        problems.extend(Problem("format", where, rid, m) for m in bad)
        if not ok:
            continue
        usable.append(rec)
        rid = rec["id"]
        if rid.split(":", 1)[0] != rec["lang"] or ":" not in rid:  # ids are `<lang>:<item>`; the lang is what item_key strips
            problems.append(Problem("format", where, rid, f"{where} id {rid!r} must start with '{rec['lang']}:' (the record's lang)"))
        hit = refusal_match(_assistant(rec))
        if hit:
            problems.append(Problem("refusal", where, rid, f"assistant turn {clip(_assistant(rec))!r} looks like a refusal ({hit[0]} pattern)"))
        problems.extend(
            Problem("answer", where, rid, m) for m in answer_problems(_user(rec), _assistant(rec), rec["gold"])
        )
        wrong = script_problem(rec["lang"], _user(rec))
        if wrong:
            problems.append(Problem("language", where, rid, wrong))
        if allowed_subjects is not None and item_subject(rid) not in allowed_subjects:
            problems.append(Problem("subject", where, rid, f"subject {item_subject(rid)!r} is not on the harmless-subject allowlist"))
    return usable


def _first_by(records: Iterable[dict[str, Any]], key: Any) -> dict[str, str]:
    """key(record) -> id of the first record that has it."""
    out: dict[str, str] = {}
    for rec in records:
        out.setdefault(key(rec), rec["id"])
    return out


def _check_taskdev(
    dev: Sequence[dict[str, Any]],
    train: Sequence[dict[str, Any]],
    heldout: Sequence[dict[str, Any]],
    problems: list[Problem],
) -> None:
    """The cross-record rules for the dev task file: unique inside, and disjoint from train and heldout.

    Everything is compared with the normalisation of the other leakage checks (`norm_text`), and
    the source item without its language prefix, so an item in one language of taskdev and another
    language of train is one item. Like the train/heldout rules this sees same-language text only;
    the prepare step keeps each item and passage on one side across languages.
    """
    seen: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for rec in dev:
        if rec["id"] in seen:
            problems.append(Problem("format", "taskdev", rec["id"], "duplicate id (also in taskdev)"))
        seen.setdefault(rec["id"], rec["id"])
        key = norm_text(_user(rec))
        if key in prompts:
            problems.append(Problem("overlap", "taskdev", rec["id"], f"duplicate prompt (same as {prompts[key]})"))
        else:
            prompts[key] = rec["id"]
    passages = {}
    for rec in dev:
        passage = passage_of(_user(rec))
        if passage:
            passages.setdefault(norm_text(passage), rec["id"])
    for name, recs in (("train", train), ("heldout", heldout)):
        other_prompts = _first_by(recs, lambda r: norm_text(_user(r)))
        other_passages = {}
        for rec in recs:
            passage = passage_of(_user(rec))
            if passage:
                other_passages.setdefault(norm_text(passage), rec["id"])
        other_items = _first_by(recs, lambda r: item_key(r["id"]))
        for rec in dev:
            hit = other_prompts.get(norm_text(_user(rec)))
            if hit:
                problems.append(Problem("taskdev_overlap", "taskdev", rec["id"], f"prompt also in {name} ({hit})"))
            hit = other_items.get(item_key(rec["id"]))
            if hit:
                problems.append(
                    Problem("taskdev_overlap", "taskdev", rec["id"], f"source item {clip(item_key(rec['id']))!r} is also in {name} ({hit})")
                )
        for key, rid in passages.items():
            if key in other_passages:
                problems.append(Problem("taskdev_overlap", "taskdev", rid, f"passage also in {name} ({other_passages[key]})"))


def _taskdev_item_warnings(dev: Sequence[dict[str, Any]], expected: Sequence[str]) -> list[str]:
    """A warning when the expected languages do not all have the same dev items (empty list when they do)."""
    items = {lang: {item_key(r["id"]) for r in dev if r["lang"] == lang} for lang in expected}
    items = {lang: found for lang, found in items.items() if found}
    if len(items) < 2:
        return []
    shared = set.intersection(*items.values())
    union = set.union(*items.values())
    if shared == union:
        return []
    return [
        f"taskdev: the languages do not hold the same items ({len(shared)} items in every language, "
        f"{len(union) - len(shared)} in only some of them)"
    ]


def validate(
    train: Sequence[Any],
    heldout: Sequence[Any],
    *,
    expected_langs: Sequence[str] = LANGS,
    safety_dev: Iterable[str] = (),
    safety_report: Iterable[str] = (),
    allowed_subjects: Collection[str] | None = None,
    safety_counts: dict[str, tuple[int, int]] | None = None,
    taskdev: Sequence[Any] | None = None,
    imbalance_tolerance: float = IMBALANCE_TOLERANCE,
    read_problems: Sequence[Problem] = (),
) -> Report:
    """Run every rule on the train and heldout records (and taskdev when given) and return the Report.

    `taskdev=None` means there is no dev task file: nothing is checked for it and the Report says
    "not built". A list (even an empty one) is checked as the dev split and must share no prompt,
    item or passage with train or heldout.

    `allowed_subjects=None` means the subject rule is off, and the Report says so. `safety_counts`
    is what `load_safety_texts` returned for the languages the safety texts came from (None when
    no safety texts were loaded): a listed language with no dev or no report prompt fails.
    """
    expected = tuple(lang for lang in LANGS if lang in set(expected_langs))
    safety_dev, safety_report = list(safety_dev), list(safety_report)
    problems: list[Problem] = list(read_problems)

    train_ok = _check_split(train, "train", allowed_subjects, problems)
    held_ok = _check_split(heldout, "heldout", allowed_subjects, problems)
    dev_ok = _check_split(taskdev, "taskdev", allowed_subjects, problems) if taskdev is not None else []

    # ids: unique inside each split and across the two
    seen: dict[str, str] = {}
    for where, recs in (("train", train_ok), ("heldout", held_ok)):
        for rec in recs:
            if rec["id"] in seen:
                problems.append(Problem("format", where, rec["id"], f"duplicate id (also in {seen[rec['id']]})"))
            else:
                seen[rec["id"]] = where

    # prompts: no repeats inside a split, none shared by train and heldout
    prompts: dict[str, dict[str, str]] = {"train": {}, "heldout": {}}
    for where, recs in (("train", train_ok), ("heldout", held_ok)):
        for rec in recs:
            key = norm_text(_user(rec))
            if key in prompts[where]:
                problems.append(Problem("overlap", where, rec["id"], f"duplicate prompt (same as {prompts[where][key]})"))
            else:
                prompts[where][key] = rec["id"]
    for key, rid in prompts["heldout"].items():
        if key in prompts["train"]:
            problems.append(Problem("overlap", "heldout", rid, f"prompt also in train ({prompts['train'][key]})"))
    # Same-language passages only. The cross-lingual case (a Hindi passage in train, its English
    # original in heldout) cannot be seen from the text; prepare keeps every language's version
    # of a passage on one side by grouping through the English passage (msd.ftprep.item_groups).
    passages: dict[str, dict[str, str]] = {"train": {}, "heldout": {}}
    for where, recs in (("train", train_ok), ("heldout", held_ok)):
        for rec in recs:
            passage = passage_of(_user(rec))
            if passage:
                passages[where].setdefault(norm_text(passage), rec["id"])
    for key, rid in passages["heldout"].items():
        if key in passages["train"]:
            problems.append(Problem("overlap", "heldout", rid, f"passage also in train ({passages['train'][key]})"))
    items = {item_key(rec["id"]): rec["id"] for rec in train_ok}
    for rec in held_ok:
        other = items.get(item_key(rec["id"]))
        if other:
            problems.append(Problem("overlap", "heldout", rec["id"], f"source item {clip(item_key(rec['id']))!r} is also in train ({other})"))

    if taskdev is not None:
        _check_taskdev(dev_ok, train_ok, held_ok, problems)

    # the safety splits: dev vs report once, then each fine-tuning split against each of them
    if safety_dev or safety_report:
        try:
            assert_no_leakage(safety_dev, safety_report)
        except LeakageError as exc:
            problems.append(Problem("overlap", "safety", None, f"safety dev and report splits overlap: {exc}"))
        for where, recs in (("train", train_ok), ("heldout", held_ok), ("taskdev", dev_ok)):
            if not recs:
                continue
            texts = _leak_texts(recs)
            for name, dev, rep in (("dev", safety_dev, ()), ("report", (), safety_report)):
                try:
                    assert_no_leakage(dev, rep, train=texts)
                except LeakageError as exc:
                    normed = {norm_text(t) for t in (dev or rep)}
                    ids = [r["id"] for r in recs if norm_text(_user(r)) in normed or norm_text(question_part(_user(r))) in normed]
                    problems.append(
                        Problem("overlap", where, ids[0] if ids else None, f"{where} overlaps the safety {name} split: {exc}")
                    )

    for lang, (n_dev, n_report) in (safety_counts or {}).items():
        if n_dev == 0 or n_report == 0:
            problems.append(
                Problem(
                    "safety_empty",
                    "safety",
                    None,
                    f"safety test set lists {lang} as covered but has {n_dev} dev and {n_report} report prompts, "
                    "so the leakage check compares nothing for it",
                )
            )

    counts = {"train": _counts(train_ok), "heldout": _counts(held_ok)}
    for message in _imbalanced(counts["train"], expected, imbalance_tolerance):
        problems.append(Problem("imbalance", "train", None, message))
    warnings = [f"heldout: {m}" for m in _imbalanced(counts["heldout"], expected, imbalance_tolerance)]
    missing_taskdev: list[str] = []
    if taskdev is not None:
        counts["taskdev"] = _counts(dev_ok)
        for message in _imbalanced(counts["taskdev"], expected, imbalance_tolerance):
            problems.append(Problem("imbalance", "taskdev", None, message))
        missing_taskdev = [lang for lang in expected if counts["taskdev"].get(lang, 0) == 0]
        for lang in missing_taskdev:
            problems.append(Problem("missing_lang", "taskdev", None, f"expected language {lang} has no taskdev rows"))
        warnings += _taskdev_item_warnings(dev_ok, expected)

    missing = [
        lang for lang in expected if counts["train"].get(lang, 0) == 0 or counts["heldout"].get(lang, 0) == 0
    ]
    for lang in missing:
        have = f"train {counts['train'].get(lang, 0)}, heldout {counts['heldout'].get(lang, 0)}"
        problems.append(Problem("missing_lang", "set", None, f"expected language {lang} has too few rows ({have})"))
    present = set(counts["train"]) | set(counts["heldout"])
    not_covered = [lang for lang in LANGS if lang not in expected and lang not in present]

    return Report(
        problems,
        warnings,
        counts,
        expected,
        missing,
        not_covered,
        allowed_subjects=None if allowed_subjects is None else tuple(sorted(allowed_subjects)),
        subject_rows_checked=len(train_ok) + len(held_ok) + len(dev_ok) if allowed_subjects is not None else 0,
        safety_counts=None if safety_counts is None else dict(safety_counts),
        taskdev_built=taskdev is not None,
        missing_taskdev=missing_taskdev,
    )


def render(report: Report, max_per_rule: int = 5) -> str:
    """The report as text for the command line.

    Two status lines are always printed, whether the check ran or not: the subject allowlist (which
    subjects, how many rows, or NOT checked) and the safety test set (which languages and how many
    prompts, or NOT checked), so a skipped or vacuous check can never read as a pass.
    """
    lines: list[str] = []
    for split in ("train", "heldout"):
        counts = report.counts[split]
        shown = ", ".join(f"{lang} {n}" for lang, n in counts.items()) or "none"
        lines.append(f"{split + ':':<9} {shown} (total {sum(counts.values())})")
    if report.taskdev_built:
        counts = report.counts["taskdev"]
        shown = ", ".join(f"{lang} {n}" for lang, n in counts.items()) or "none"
        lines.append(f"{'taskdev:':<9} {shown} (total {sum(counts.values())})")
    else:
        lines.append("taskdev: not built (optional; --require-taskdev turns a missing file into an error)")
    lines.append(f"expected languages: {', '.join(report.expected_langs)}")
    if report.missing_expected:
        lines.append(f"missing expected languages: {', '.join(report.missing_expected)}")
    if report.missing_taskdev:
        lines.append(f"missing expected languages in taskdev: {', '.join(report.missing_taskdev)}")
    if report.not_covered:
        lines.append(f"not covered (in the study, no rows, not expected here): {', '.join(report.not_covered)}")
    if report.allowed_subjects is None:
        lines.append("subject allowlist: NOT checked (--no-subject-check was given; no subject is enforced)")
    else:
        names = ", ".join(report.allowed_subjects)
        lines.append(
            f"subject allowlist: checked {report.subject_rows_checked} rows "
            f"against {len(report.allowed_subjects)} subjects ({names})"
        )
    if report.safety_counts is None:
        lines.append("safety test set: NOT checked (skipped)")
    else:
        covered = list(report.safety_counts)
        lacking = [lang for lang in LANGS if lang not in covered]
        tail = f"; not built yet, so not checked: {', '.join(lacking)}" if lacking else ""
        lines.append(f"safety test set covers: {', '.join(covered)}{tail}")
        sizes = "; ".join(f"{lang}: {d} dev + {r} report" for lang, (d, r) in report.safety_counts.items())
        lines.append(f"safety prompts compared against: {sizes}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    for rule in report.failed_rules():
        found = [p for p in report.problems if p.rule == rule]
        lines.append(f"FAIL {rule}: {len(found)} problem(s), exit code {EXIT_CODES[rule]}")
        for p in found[:max_per_rule]:
            label = f"{p.where} {p.id}" if p.id else p.where
            lines.append(f"  [{label}] {p.message}")
        if len(found) > max_per_rule:
            lines.append(f"  ... and {len(found) - max_per_rule} more")
    if report.ok:
        skipped = " (subject allowlist NOT checked)" if report.allowed_subjects is None else ""
        lines.append(f"RESULT: OK{skipped}")
    else:
        lines.append(f"RESULT: FAILED ({', '.join(report.failed_rules())})")
    return "\n".join(lines)
