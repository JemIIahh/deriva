#!/usr/bin/env python
"""Check the Hugging Face article against the submission rules. Exit 0 only when every rule passes.

    python scripts/check_article.py article/ARTICLE.md data/sources.yaml            # the check before submitting
    python scripts/check_article.py article/ARTICLE.md data/sources.yaml --draft    # the check while it is a draft

The article has nine required sections. This script checks what a script can check; it does not judge whether the
article is good, and it does not read the results. The rules, in the order they are reported:

  headings           exactly these nine level-2 headings, in this order, none missing, none twice, none extra
                     (use level 3 for subsections). Headings inside code blocks are ignored.
  heading-levels     no heading level is skipped (a level 4 under a level 2 is an error).
  source-licenses    every entry of the source list that is not `excluded` is named, with its licence, on one line of
                     the "Models used and their licenses" or the "Datasets and benchmarks used and their licenses" section. An entry with
                     `redistribute: false` must also say "evaluation only" on that line. A row for an entry whose name
                     starts with another entry's name ("X (fine-tune subset)" and "X") counts only for the longer one.
  licenses-and-tools the licences and tools the study depends on are written out (Tiny Aya's licence, gated access
                     and use policy; AlignTune MIT with its CuratorKIT dependency under LSAL-1.1; unsloth AGPL and
                     mergekit LGPL; the TRL backend only; NLLB CC-BY-NC; the MultiJail ethics statement; the
                     toolkits named). See REQUIRED_MENTIONS.
  disclosure         one paragraph says that Claude Code, an AI coding assistant, helped build the work.
  empty-or-links-only  every section holds prose, not only links and not nothing.
  figures            every image has alt text (at least three words); the per-language heatmap is shown; a local
                     image file exists (in --draft mode it may be missing if FILL_IN.md lists it as still to be added).
  hype               a short list of hype words is not used.
  claims             DRAFT MODE ONLY. No sentence or heading states the study's result as fact outside a
                     <<RESULT: ...>> placeholder, because nothing has been measured yet (the final article may state
                     results). Everywhere: sentences of the shapes in CLAIM_PATTERNS. In the front matter and in the
                     experiments and results sections, also any sentence that puts a language or a model version next to
                     a word of comparison or a verb of result (SCOPED_CLAIM_PATTERNS), any percentage, number of points,
                     decimal or number word with a unit, and any cell of a results table that is not a placeholder or
                     "Not applicable"; method settings belong in Sections 1 to 6. Alt text, link text and the words
                     inside bold, italic, code and block quotes are read as plain text.
  placeholders       <<RESULT: name>>, <<LINK: name>> and <<OWNER: name>> are the only markers. With --draft they pass
                     and are listed; without it each one fails. A malformed marker always fails.
  unfinished-markers TODO, TBD, XXX and {{...}} fail in both modes.
  fill-in            DRAFT MODE ONLY. FILL_IN.md beside the article (or --fill-in) has a line for every placeholder.

The claims rule is a text-level backstop. It protects the draft before any result exists, by catching the sentences that
most often state one, and it cannot be complete: a result can always be worded in a way no pattern knows, and the script
does not read meaning. It has no list of approved sentences and does not try to cover every idiom. A person must read the
draft before it is submitted.

HTML comments (<!-- ... -->) do not render, so the rules about content do not read them: a section, a licence sentence or
the disclosure inside a comment does not count. Placeholders and TODO-style markers are still found inside a comment.

Exit codes: 0 every rule passed; 1 at least one rule failed (each is printed as "FAIL [rule] line N: what is wrong");
2 a usage problem (a file cannot be read, the source list is malformed or not valid YAML).

The headings are the nine sections of the organizers' submission checklist, word for word as slide 29 of
reference/kickoff-slides-text.txt gives them (a test compares them with that file). If the organizers change a title,
change REQUIRED_HEADINGS and the article's headings together.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from msd.sources import load_sources  # noqa: E402

# The nine section titles of the organizers' submission checklist (reference/kickoff-slides-text.txt, slide 29), word for word.
REQUIRED_HEADINGS: tuple[str, ...] = (
    "Problem statement and approach, in brief",
    "Hackathon repos used: which ones, for what, and which features",
    "Models used and their licenses",
    "Datasets and benchmarks used and their licenses",
    "Model artifacts (weights, adapters, …) public, as an HF collection",
    "Data artifacts (datasets, logs, benchmarks, …) public, as an HF collection",
    "Experiments run, findings and what else was tested",
    "With more credits: your approach, before and after results",
    "Teammates tagged",
)
REPOS, MODELS, DATASETS = REQUIRED_HEADINGS[1], REQUIRED_HEADINGS[2], REQUIRED_HEADINGS[3]
MODEL_ARTIFACTS, DATA_ARTIFACTS, TEAMMATES = REQUIRED_HEADINGS[4], REQUIRED_HEADINGS[5], REQUIRED_HEADINGS[8]

MARKER_KINDS: tuple[str, ...] = ("RESULT", "LINK", "OWNER")
MIN_PROSE_WORDS = 12  # a section with fewer words of its own (links and placeholders do not count) is "links only"
MIN_ALT_WORDS = 3  # alt text of a figure: a short description, not one word
EVALUATION_ONLY = re.compile(r"evaluation[\s-]*only", re.I)
HEATMAP_NAME = "heatmap"  # the per-language heatmap is the image whose file name contains this
FILL_IN_NAME = "FILL_IN.md"

# --- what a line of the article can look like --------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]*)[^)]*\)")
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_CANDIDATE_MARKER = re.compile(r"<<[^<>\n]*>>")
_VALID_MARKER = re.compile(r"<<(RESULT|LINK|OWNER):[ \t]*([^<>\n]*?)[ \t]*>>")
_UNFINISHED = re.compile(r"\bTODO\b|\bTBD\b|\bXXX+\b|\{\{.*?\}\}", re.I)
_WORD = re.compile(r"[^\W\d_]{2,}")

HYPE = re.compile(
    r"\b(?:state[- ]of[- ]the[- ]art|groundbreaking|ground-breaking|revolutionary|game[- ]chang\w*|breakthrough|"
    r"unprecedented|dramatic(?:ally)?|stunning(?:ly)?|astonishing|incredible|remarkabl[ey]|cutting[- ]edge|"
    r"world[- ]class|best[- ]in[- ]class)\b",
    re.I,
)

# --- the claims guard ----------------------------------------------------------------------------------------------
# Draft mode only. Each pattern is the shape of a sentence that reports what the study found. The study has not been
# run yet, so such a sentence may only exist inside a <<RESULT: ...>> placeholder. Present tense ("falls", "adds") and
# questions are how the draft describes the method and the question, so they pass. Headings are checked too. This is a
# text-level net, not a proof: a result can always be worded in a way no pattern knows, so the figures rule below is a
# second net, and a human reads the draft before it is submitted.

_GAP = r"[^.?!]{0,50}?"
_TARGET = (
    r"(?:safety|refusals?|refusal rate|over[- ]refusal|attack success|asr|compliance|alignment|guardrails|"
    r"task accuracy|accuracy)"
)
_CHANGE_PAST = (
    r"eroded|weakened|degraded|undermined|damaged|compromised|lowered|reduced|cut|raised|boosted|increased|"
    r"decreased|worsened|improved"
)
_CHANGE_PRESENT = r"erodes|weakens|degrades|undermines|damages|compromises"
_DRIFT_WORDS = r"(?:drift|gap|rise|erosion|increase|decrease|decline|change|differences?)"
_SUPERLATIVES = r"(?:largest|biggest|greatest|strongest|widest|weakest|worst|highest|lowest|smallest)"
CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "it says fine-tuning made the model less safe",
        re.compile(rf"\b(?:made|makes|left|leaves)\b{_GAP}\b(?:less safe|unsafe|more harmful|more vulnerable|less aligned)\b", re.I),
    ),
    (
        "it says a model is less safe, or safer",
        re.compile(
            r"\b(?:is|are|was|were|became|become|becomes|remains|remained|stayed|stays)\b[^.?!]{0,40}?\b(?:less safe|safer|more harmful|"
            r"more vulnerable|less aligned|more unsafe|less robust|more compliant|less compliant|more likely to (?:comply|refuse)|"
            r"less likely to (?:comply|refuse))\b",
            re.I,
        ),
    ),
    (
        "it says safety was restored or repaired",
        re.compile(
            # "repaired" followed by a noun names an arm ("the repaired model"); it does not say that anything worked
            rf"\b(?:restored|restores|recovered|recovers|repaired(?!\s+(?:model|models|setting|settings|arm|arms|cell|cells|"
            rf"weights|version|versions|checkpoint|checkpoints|strength|strengths)\b)|repairs|fixed|fixes|undid|undoes|reversed)\b{_GAP}"
            r"\b(?:safety|refusals?|alignment|drift|safe)\b",
            re.I,
        ),
    ),
    (
        "it says something was brought back",
        re.compile(r"\b(?:brought|put|got)\b[^.?!]{0,40}?\bback\b", re.I),
    ),
    (
        "it says the repair worked or failed",
        re.compile(
            r"\b(?:repair|repairs|patch|resta|blend)\b[^.?!]{0,40}?\b(?:worked|succeeded|failed|helped|recovered|"
            r"brought back|did not help|did help)\b",
            re.I,
        ),
    ),
    (
        "it says a rate went up or down",
        re.compile(
            rf"\b(?:attack success|over[- ]refusal|refusal rate|unsafe rate|asr|task accuracy|accuracy|safety)\b[^.?!]{{0,40}}?"
            r"\b(?:rose|fell|increased|decreased|dropped|went up|went down|went from|grew|shrank|doubled|halved|climbed|improved|"
            r"declined|worsened)\b",
            re.I,
        ),
    ),
    (
        "it says something was weakened, cut or raised",
        re.compile(rf"\b(?:{_CHANGE_PAST}|{_CHANGE_PRESENT})\b{_GAP}\b{_TARGET}\b", re.I),
    ),
    (
        "it says something was weakened, cut or raised",
        re.compile(
            rf"\b{_TARGET}\b[^.?!]{{0,40}}?\b(?:was|were|is|are|got|gets|has been|have been)\s+(?:\w+ly\s+)?(?:{_CHANGE_PAST})\b",
            re.I,
        ),
    ),
    (
        "it says a number went from one value to another",
        re.compile(r"\b(?:went|goes|moved|climbed|jumped)\s+(?:from|up|down|to)\b", re.I),
    ),
    (
        "it says how large the drift or gap was",
        re.compile(
            r"\b(?:drift|gap|erosion)\b[^.?!]{0,30}?\b(?:is|was|were|grew|widened|narrowed|shrank)\b[^.?!]{0,20}?"
            r"\b(?:larger|bigger|smaller|wider|narrower|stronger|weaker|greater|worse|largest|biggest|widest|strongest|weakest)\b",
            re.I,
        ),
    ),
    (
        "it names the largest or smallest drift, gap or rise",
        re.compile(rf"^(?=.*?\b{_DRIFT_WORDS}\b)(?=.*?\b{_SUPERLATIVES}\b)", re.I | re.S),
    ),
    (
        "it says what the model complied with or refused",
        re.compile(r"\b(?:complied|refused)\b[^.?!]{0,25}?\b(?:more|fewer|less|most|many|all|almost|nearly|half)\b", re.I),
    ),
    (
        "it says the gate passed or failed",
        re.compile(
            r"\bgate\b[^.?!]{0,30}?\b(?:passed|failed|fired|triggered|did not pass)\b|"
            r"\bthe study (?:continued|pivoted|moved on|went on|switched|carried on)\b",
            re.I,
        ),
    ),
    (
        "it reports a finding",
        re.compile(
            r"\b(?:we found that|we observed|we observe that|our results show|the results show|results show that|"
            r"this shows that|this confirms|we confirm|we measured that|turned out)\b|"
            r"\bwe\s+(?:also\s+)?(?:found|saw|observed|measured|detected|noticed)\b|"
            r"\b(?:shows?|showed|reveals?|revealed|demonstrates?|demonstrated|confirms?|confirmed|indicates?|indicated|suggests?|suggested)\s+that\b|"
            r"\b(?:was|were|has been|have been)\s+(?:also\s+)?(?:found|observed|seen|detected)\b|"
            r"\b(?:found|finds?)\s+that\b",
            re.I,
        ),
    ),
)

# In the front matter (under the title, before the first section) and in the two sections that hold results, a sentence
# that names a language or a model version has no business using a word of comparison or a verb of result: those places
# hold questions, pointers and placeholders, and the method is in Sections 1 to 6. The patterns below are stricter than
# CLAIM_PATTERNS for that reason, and they run only there. They still cannot know every way to say a result.
_NAME = (
    r"(?:english|hindi|hinglish|arabic|swahili|original|drifted|repaired|naive blend|resta|the repair"
    r"|(?-i:BASE|A0|A1|A2n|A2|A3))"
)
_NOT_SUPERLATIVES = (
    "test|request|suggest|interest|rest|west|invest|protest|contest|digest|arrest|attest|manifest|harvest|honest|modest|"
    "quest|guest|chest|nest|pest|vest|forest|crest"
)
_COMPARISON = (
    r"(?:more|less|fewer|higher|lower|larger|smaller|bigger|wider|narrower|stronger|weaker|greater|worse|better|most|least|"
    r"safer|permissive|affected|unaffected|stable|flat|unchanged|hardest|weakest|"
    rf"(?!(?:{_NOT_SUPERLATIVES})\b)\w+est)"
)
_RESULT_VERB = (
    r"(?:dropped|drops|rose|rises|fell|falls|moved|held(?!-out)|stayed|remained|lost|gained|improved|improves|helped|helps|"
    r"hurt|hurts|beats|works|failed|fails|passed|cleared|crossed|stands? out|saw|showed|became|degraded|degrades)"
)
_MEASURE = (
    r"(?:attack success(?: rates?)?|asr|refusal rates?|refusals?|over-?refusal(?: rates?)?|caution|scores?|safety|"
    r"task accuracy|accuracy|compliance|alignment|guardrails?|drift|the gap|rates?)"
)
SCOPED_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "it puts a language or a model next to a comparison",
        re.compile(rf"^(?=.*?\b{_NAME}\b)(?=.*?\b{_COMPARISON}\b)", re.I | re.S),
    ),
    (
        "it says what a language or a model did",
        re.compile(rf"\b{_NAME}\b[^.?!]{{0,80}}?\b{_RESULT_VERB}\b", re.I),
    ),
    (
        "it says what happened to a measure",
        re.compile(rf"\b{_MEASURE}\b[^.?!]{{0,60}}?\b(?:{_RESULT_VERB}|{_COMPARISON})\b", re.I),
    ),
)

# A percentage or a number of points in the experiments and the results sections has to be a result, and a result may
# only stand inside a placeholder. Method settings (limits, thresholds, margins) belong in the earlier sections.
_FIGURE = re.compile(r"\d+(?:[.,]\d+)?(?:\s*%|\s+percent\b|[\s-]*(?:percentage[\s-]*)?(?:points?|pts?|pct)\b|\s*pp\b)", re.I)
_NUMBER_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
)
_FIGURE_SHAPES: tuple[re.Pattern[str], ...] = (
    _FIGURE,
    # a decimal ("0.25", ".5"); not a version inside a name ("Qwen3.5") and not a licence version ("CC-BY-4.0")
    re.compile(r"(?<![\w.])(?<![A-Za-z]-)\d*\.\d+\b"),
    # a number in words followed by a unit ("twelve points", "twenty-five percent", "ten per cent", "fifteen pct")
    re.compile(
        rf"\b{_NUMBER_WORD}(?:[\s-]+{_NUMBER_WORD})*[\s-]*(?:percentage[\s-]*)?(?:points?|percent|per[\s-]*cent|pct|pts?|pp|x)\b",
        re.I,
    ),
    # "a quarter", "half", "3x", "1 in 4", "one in four"
    re.compile(r"\b(?:quarters?|half|halves)\b", re.I),
    re.compile(r"\b\d+(?:[.,]\d+)?\s*[x\u00d7](?![\w])", re.I),
    re.compile(rf"\b(?:\d+|{_NUMBER_WORD})\s+in\s+(?:\d+|{_NUMBER_WORD})\b", re.I),
)
FIGURE_SECTIONS: tuple[str, ...] = (REQUIRED_HEADINGS[6], REQUIRED_HEADINGS[7])
RESULT_TABLES = REQUIRED_HEADINGS[7]  # every data cell of a table here is a placeholder or "Not applicable"


def _first_figure(text: str) -> str | None:
    for shape in _FIGURE_SHAPES:
        found = shape.search(text)
        if found:
            return found.group(0).strip()
    return None


# --- licences and tools the article must write out ---------------------------------------------------------------


def _lic(text: str) -> str:
    """A regex for a licence name that ignores case and the difference between -, space and _ ("CC BY-SA 4.0")."""
    parts = [re.escape(p) for p in re.split(r"[-\s_]+", text.strip().lower()) if p]
    return r"(?<![a-z0-9])" + r"[-\s_]*".join(parts) + r"(?![a-z0-9])"


@dataclass(frozen=True)
class Mention:
    """One thing the article must say: a paragraph (or list item or table row) in one of `sections` matches every
    pattern in `all_of`. `label` finishes the sentence "The article does not mention ..." """

    label: str
    sections: tuple[str, ...]
    all_of: tuple[str, ...]


REQUIRED_MENTIONS: tuple[Mention, ...] = (
    Mention("Tiny Aya's licence (CC-BY-NC-4.0)", (MODELS,), (r"tiny aya", _lic("CC-BY-NC-4.0"))),
    Mention("that Tiny Aya is gated on Hugging Face", (MODELS,), (r"tiny aya", r"\bgated\b")),
    Mention("the Cohere Labs Acceptable Use Policy that also applies to Tiny Aya", (MODELS,), (r"cohere", r"acceptable use policy")),
    Mention("that AlignTune is released under the MIT licence", (REPOS, MODELS), (r"aligntune", _lic("MIT"))),
    Mention("that CuratorKIT is under the Lexsi Labs Source Available License 1.1 (LSAL-1.1)", (REPOS, MODELS), (r"curatorkit", _lic("LSAL-1.1"))),
    Mention("that AlignTune depends on CuratorKIT", (REPOS, MODELS), (r"aligntune", r"(?:depends|requires?)", r"curatorkit")),
    Mention("that the AlignTune package bundles unsloth, which has AGPL parts", (REPOS, MODELS), (r"unsloth", r"agpl")),
    Mention("that the AlignTune package bundles mergekit, which is LGPL", (REPOS, MODELS), (r"mergekit", r"lgpl")),
    Mention("that only the TRL backend of AlignTune is used", (REPOS, MODELS), (r"\btrl\b", r"backend")),
    Mention("NLLB and its CC-BY-NC licence, for the machine translations", (REPOS, MODELS, DATASETS, DATA_ARTIFACTS), (r"nllb", _lic("CC-BY-NC"))),
    Mention("the MultiJail ethics statement", (DATASETS,), (r"multijail", r"ethics statement")),
    Mention("SafeTune, the repair toolkit, and what it is used for", (REPOS,), (r"safetune",)),
    Mention("AuditKIT, and whether it was used", (REPOS,), (r"auditkit",)),
)


# --- reading the article -----------------------------------------------------------------------------------------


class UsageError(Exception):
    """A file cannot be read or the source list is malformed: exit code 2, never a traceback."""


@dataclass(frozen=True)
class Finding:
    rule: str
    message: str
    line: int | None = None

    def render(self) -> str:
        where = f" line {self.line}" if self.line else ""
        return f"FAIL [{self.rule}]{where}: {self.message}"


@dataclass(frozen=True)
class Placeholder:
    kind: str
    name: str
    line: int


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    placeholders: list[Placeholder] = field(default_factory=list)
    draft: bool = False

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, rule: str, message: str, line: int | None = None) -> None:
        self.findings.append(Finding(rule, message, line))


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    line: int


@dataclass(frozen=True)
class Block:
    """A paragraph, a list item or a table row: the unit a mention must sit in. `line` is its first line."""

    line: int
    text: str


class Doc:
    """The article split into lines, code blocks, headings, sections and blocks."""

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.in_code = self._mark_code(self.lines)
        self.headings = [
            Heading(len(m.group(1)), m.group(2).strip(), i + 1)
            for i, line in enumerate(self.lines)
            if not self.in_code[i] and (m := _HEADING.match(line))
        ]

    @staticmethod
    def _mark_code(lines: list[str]) -> list[bool]:
        marks, fence = [], None
        for line in lines:
            m = _FENCE_OPEN.match(line)
            if fence is None:
                marks.append(bool(m))  # the opening line is code too
                if m:
                    fence = (m.group(1)[0], len(m.group(1)))
            else:
                marks.append(True)
                if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not line.strip().strip(fence[0]):
                    fence = None
        return marks

    def text_lines(self, first: int = 1, last: int | None = None) -> list[tuple[int, str]]:
        """(line number, text) of the lines outside code blocks, from `first` to `last` (1-based, inclusive)."""
        end = len(self.lines) if last is None else last
        return [(i + 1, self.lines[i]) for i in range(first - 1, end) if not self.in_code[i]]

    def sections(self, title: str) -> list[tuple[int, int]]:
        """(first body line, last body line) of every level-2 section called `title`; the body runs to the next
        level-1 or level-2 heading."""
        spans = []
        for idx, h in enumerate(self.headings):
            if h.level == 2 and h.text == title:
                end = len(self.lines)
                for later in self.headings[idx + 1 :]:
                    if later.level <= 2:
                        end = later.line - 1
                        break
                spans.append((h.line + 1, end))
        return spans

    def section_lines(self, title: str) -> list[tuple[int, str]]:
        return [pair for first, last in self.sections(title) for pair in self.text_lines(first, last)]

    def section_blocks(self, titles: tuple[str, ...]) -> list[Block]:
        return [b for title in titles for b in blocks(self.section_lines(title))]


def blocks(lines: list[tuple[int, str]]) -> list[Block]:
    """Group lines into paragraphs. A blank line, a list item, a table row, a heading or a gap (a code block removed
    from the lines) ends a paragraph; a table row and a list item are each their own block."""
    out: list[Block] = []
    current: list[tuple[int, str]] = []

    def flush() -> None:
        if current:
            out.append(Block(current[0][0], " ".join(t.strip() for _, t in current)))
            current.clear()

    previous = None
    for number, text in lines:
        stripped = text.strip()
        if previous is not None and number != previous + 1:
            flush()
        previous = number
        if not stripped or _HEADING.match(text):
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            out.append(Block(number, stripped))
            continue
        if _LIST_ITEM.match(text):
            flush()
        current.append((number, text))
    flush()
    return out


_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)


def strip_comments(text: str) -> str:
    """The article as a reader sees it: HTML comments (which do not render) become blanks, line breaks kept so that line
    numbers stay right. An unterminated comment hides the rest of the page, as it does in a browser."""
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _strip_markers(text: str) -> str:
    return _VALID_MARKER.sub(" ", text)


_ODD_SPACES = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_QUOTE_MARK = re.compile(r"^[ \t]*(?:>[ \t]?)+")
_EMPHASIS = re.compile(r"[*`]+|(?<!\w)_+|_+(?!\w)")  # * and ` anywhere; _ only at the edge of a word (resta_patch.json keeps its _)


def _alt_of(tag: str) -> str:
    alt = re.search(r"""alt\s*=\s*(?:"([^"]*)"|'([^']*)')""", tag, re.I)
    return ((alt.group(1) or alt.group(2)) if alt else "").strip()


def _quote_free(line: str) -> str:
    """One line without its leading block-quote marks ("> ", "> > ") and with odd spaces (no-break space) made plain."""
    return _QUOTE_MARK.sub("", _ODD_SPACES.sub(" ", line))


def _plain(text: str) -> str:
    """The words a reader reads, for matching a claim or a figure: an image becomes its alt text, a link its text, the marks
    * _ ` that make bold, italic and code disappear, and runs of white space become one space. Call it after the markers
    are stripped."""
    text = _IMAGE.sub(lambda m: " " + m.group(1) + " ", _ODD_SPACES.sub(" ", text))
    text = _HTML_IMAGE.sub(lambda m: " " + _alt_of(m.group(0)) + " ", text)
    text = _LINK.sub(r"\1", text)
    return re.sub(r"\s+", " ", _EMPHASIS.sub("", text)).strip()


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# --- the rules ---------------------------------------------------------------------------------------------------


def check_headings(doc: Doc, report: Report) -> None:
    level_two = [h for h in doc.headings if h.level == 2]
    texts = [h.text for h in level_two]
    for number, required in enumerate(REQUIRED_HEADINGS, 1):
        lines = [h.line for h in level_two if h.text == required]
        if not lines:
            report.add(
                "headings",
                f"Section {number} '{required}' is missing. It must be a level-2 heading (##) with exactly these words.",
            )
        elif len(lines) > 1:
            where = " and ".join(str(n) for n in lines) if len(lines) == 2 else ", ".join(str(n) for n in lines)
            times = "twice" if len(lines) == 2 else f"{len(lines)} times"
            report.add("headings", f"The heading '{required}' appears {times} (lines {where}). Each section must appear once.", lines[1])
    for h in level_two:
        if h.text not in REQUIRED_HEADINGS:
            report.add(
                "headings",
                f"Extra level-2 heading '{h.text}'. The article has exactly the nine required level-2 headings; "
                "use level 3 (###) for anything else.",
                h.line,
            )
    once = [r for r in REQUIRED_HEADINGS if texts.count(r) == 1]
    actual = [t for t in texts if t in once]
    if actual != once:
        k = next(i for i, (a, b) in enumerate(zip(actual, once)) if a != b)
        report.add(
            "headings",
            f"The headings are in the wrong order: '{actual[k]}' comes where '{once[k]}' must come. "
            "The order is: " + "; ".join(f"{i}. {r}" for i, r in enumerate(REQUIRED_HEADINGS, 1)) + ".",
            next(h.line for h in level_two if h.text == actual[k]),
        )


def check_heading_levels(doc: Doc, report: Report) -> None:
    previous = 0
    for h in doc.headings:
        if previous == 0 and h.level > 2:
            report.add("heading-levels", f"The first heading, '{h.text}', is level {h.level}. Start with level 1 (the title) or level 2.", h.line)
        elif previous and h.level > previous + 1:
            report.add(
                "heading-levels",
                f"Heading '{h.text}' is level {h.level} but follows a level {previous} heading. Do not skip a level "
                f"(use level {previous + 1}).",
                h.line,
            )
        previous = h.level


def check_sources(doc: Doc, sources: list[dict], report: Report) -> None:
    lines = doc.section_lines(DATASETS) + doc.section_lines(MODELS)
    where = f"'{DATASETS}' or '{MODELS}'"
    all_names = [e["name"] for e in sources]
    for entry in sources:
        if entry["used_for"] == "excluded":
            continue
        name, licence = entry["name"], entry["license"]
        name_re = re.compile(r"(?<![A-Za-z0-9_/])" + re.escape(name) + r"(?![A-Za-z0-9_])", re.I)
        licence_re = re.compile(_lic(licence), re.I)
        # another entry's longer name may start with this name ("X" and "X (fine-tune subset)"); a row for the longer
        # one must not count as a row for this one
        longer = [re.compile(re.escape(n), re.I) for n in all_names if len(n) > len(name) and n.lower().startswith(name.lower())]

        def without_longer(text: str) -> str:
            for pattern in longer:
                text = pattern.sub(lambda m: " " * len(m.group(0)), text)
            return text

        named = [(n, t) for n, t in lines if name_re.search(without_longer(t))]
        if not named:
            report.add("source-licenses", f"The source '{name}' (licence {licence}) is not named in {where}.")
            continue
        with_licence = [(n, t) for n, t in named if licence_re.search(t)]
        if not with_licence:
            report.add(
                "source-licenses",
                f"The source '{name}' is named but its licence '{licence}' is not on the same line. "
                "Put the name and the licence on one line (one table row).",
                named[0][0],
            )
            continue
        if entry.get("redistribute") is False and not any(EVALUATION_ONLY.search(t) for _, t in with_licence):
            report.add(
                "source-licenses",
                f"The source '{name}' has redistribute: false, so its line must say it is used for evaluation only "
                "(scores are published, its items are not).",
                with_licence[0][0],
            )


def check_mentions(doc: Doc, report: Report) -> None:
    for mention in REQUIRED_MENTIONS:
        compiled = [re.compile(p, re.I) for p in mention.all_of]
        pool = doc.section_blocks(mention.sections)
        if not any(all(c.search(b.text) for c in compiled) for b in pool):
            sections = " or ".join(f"'{s}'" for s in mention.sections)
            report.add("licenses-and-tools", f"The article does not mention {mention.label}. Say it in one paragraph of {sections}.")


def check_disclosure(doc: Doc, report: Report) -> None:
    for block in blocks(doc.text_lines()):
        low = block.text.lower()
        if "claude code" in low and (re.search(r"\bAI\b", block.text) or "assistant" in low):
            return
    report.add(
        "disclosure",
        "No paragraph says that Claude Code, an AI coding assistant, helped build the work. Add one plain paragraph "
        "that names Claude Code and says it is an AI assistant (the 'Teammates tagged' section is the natural place).",
    )


def _prose_words(lines: list[tuple[int, str]]) -> int:
    count = 0
    for _, text in lines:
        if _HEADING.match(text):
            continue
        text = _IMAGE.sub(" ", text)
        text = _HTML_IMAGE.sub(" ", text)
        text = _VALID_MARKER.sub(" ", text)
        text = _LINK.sub(r"\1", text)
        text = _URL.sub(" ", text)
        count += len(_WORD.findall(text))
    return count


def check_sections_have_prose(doc: Doc, report: Report) -> None:
    for title in REQUIRED_HEADINGS:
        for first, last in doc.sections(title):
            words = _prose_words(doc.text_lines(first, last))
            if words == 0:
                report.add("empty-or-links-only", f"The section '{title}' is empty. Write what it has to say in plain sentences.", first - 1)
            elif words < MIN_PROSE_WORDS:
                report.add(
                    "empty-or-links-only",
                    f"The section '{title}' holds only links or a few words ({words}). Say in a sentence or two what "
                    "the links are and what the reader finds there.",
                    first - 1,
                )


def _figures(doc: Doc) -> list[tuple[int, str, str]]:
    """(line, alt text, path) of every figure outside code blocks, Markdown or HTML."""
    found = []
    for number, text in doc.text_lines():
        for m in _IMAGE.finditer(text):
            found.append((number, m.group(1).strip(), m.group(2).strip()))
        for m in _HTML_IMAGE.finditer(text):
            alt = re.search(r"""alt\s*=\s*(?:"([^"]*)"|'([^']*)')""", m.group(0), re.I)
            src = re.search(r"""src\s*=\s*(?:"([^"]*)"|'([^']*)')""", m.group(0), re.I)
            found.append((number, ((alt.group(1) or alt.group(2)) if alt else "").strip(), ((src.group(1) or src.group(2)) if src else "")))
    return found


def _is_local(path: str) -> bool:
    return bool(path) and not re.match(r"(?:[a-z][a-z0-9+.-]*:|//|<<)", path, re.I)


def check_figures(doc: Doc, article: Path, fill_in: str | None, report: Report) -> None:
    figures = _figures(doc)
    for number, alt, path in figures:
        if len(re.findall(r"\w+", alt)) < MIN_ALT_WORDS:
            report.add(
                "figures",
                f"The figure '{path}' has no usable alt text. Describe the picture in at least {MIN_ALT_WORDS} words "
                "inside the brackets, for a reader who cannot see it.",
                number,
            )
    if not any(HEATMAP_NAME in Path(path.split("?")[0]).name.lower() for _, _, path in figures):
        report.add(
            "figures",
            "The per-language heatmap is not shown. Add an image whose file name contains 'heatmap' "
            "(for example ![Heatmap of ...](heatmap.png)) to the before and after results section.",
        )
    for number, _, path in figures:
        if not _is_local(path):
            continue
        target = article.parent / path.split("#")[0].split("?")[0]
        if target.is_file():
            continue
        name = Path(path).name
        if report.draft:
            if fill_in is not None and name in fill_in:
                continue
            if fill_in is not None:  # the fill-in file exists but does not list the picture
                report.add(
                    "fill-in",
                    f"The figure file '{name}' does not exist yet and {FILL_IN_NAME} does not list it as still to be added.",
                    number,
                )
        else:
            report.add(
                "figures",
                f"The figure file '{path}' does not exist next to the article. Copy it from the export "
                f"(dataset/aggregates/{name}) before submitting.",
                number,
            )


def check_hype(doc: Doc, report: Report) -> None:
    for number, text in doc.text_lines():
        m = HYPE.search(text)
        if m:
            report.add("hype", f"The word '{m.group(0)}' reads as hype. State what was measured, in plain words.", number)


def _strict_spans(doc: Doc) -> list[tuple[int, int]]:
    """Line ranges where SCOPED_CLAIM_PATTERNS also run: the front matter and the two sections that hold results."""
    first_section = next((h.line for h in doc.headings if h.level == 2), len(doc.lines) + 1)
    spans = [(1, first_section - 1)]
    for title in FIGURE_SECTIONS:
        spans.extend(doc.sections(title))
    return spans


def check_claims(doc: Doc, report: Report) -> None:
    lines = [(number, _quote_free(text)) for number, text in doc.text_lines()]
    units = [(b.line, b.text) for b in blocks(lines)] + [(h.line, h.text) for h in doc.headings]
    strict = _strict_spans(doc)
    for line, raw in sorted(units):
        text = _plain(_strip_markers(raw))
        patterns = CLAIM_PATTERNS + (SCOPED_CLAIM_PATTERNS if any(a <= line <= b for a, b in strict) else ())
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if not sentence or sentence.endswith("?"):
                continue
            for why, pattern in patterns:
                if pattern.search(sentence):
                    shown = sentence if len(sentence) <= 90 else sentence[:87] + "..."
                    report.add(
                        "claims",
                        f"This sentence states a result as fact ({why}), but the result has not been measured yet: "
                        f'"{shown}" Put the result inside a <<RESULT: ...>> placeholder and write the question here.',
                        line,
                    )
                    break
    for title in FIGURE_SECTIONS:
        for first, last in doc.sections(title):
            section = [(number, _quote_free(text)) for number, text in doc.text_lines(first, last)]
            found = [(block.line, _URL.sub(" ", _plain(_strip_markers(block.text)))) for block in blocks(section)]
            # code blocks are not prose, but a result can sit in one as well
            found += [(number, _strip_markers(doc.lines[number - 1])) for number in range(first, last + 1) if doc.in_code[number - 1]]
            for number, visible in sorted(found):
                figure = _first_figure(visible)
                if figure:
                    report.add(
                        "claims",
                        f"The figure '{figure}' stands outside a <<RESULT: ...>> placeholder in the section '{title}'. "
                        "In a draft, every percentage, every number of points and every decimal in that section must be a "
                        "result inside a placeholder; put method settings (limits, thresholds, margins) in Sections 1 to 6.",
                        number,
                    )
    check_result_tables(doc, report)


_SEPARATOR_ROW = re.compile(r"^\|?[\s:|-]*-[\s:|-]*\|?$")


def check_result_tables(doc: Doc, report: Report) -> None:
    """In the before and after section every data cell of a table (not its header, not its row label) is a placeholder
    or 'Not applicable', so that no number or word that reads as a result stands in the table by hand."""
    for first, last in doc.sections(RESULT_TABLES):
        rows: list[tuple[int, str]] = []
        for number, text in doc.text_lines(first, last) + [(last + 2, "")]:  # the sentinel closes the last table
            stripped = _quote_free(text).strip()
            if stripped.startswith("|") and (not rows or number == rows[-1][0] + 1):
                rows.append((number, stripped))
                continue
            data = rows[2:] if len(rows) > 1 and _SEPARATOR_ROW.match(rows[1][1]) else rows[1:]
            for row_number, row in data:
                cells = [c.strip() for c in _VALID_MARKER.sub("\x00", row).strip("|").split("|")][1:]
                for cell in cells:
                    if re.fullmatch(r"[\x00\s,;/()\-]*", cell) and "\x00" in cell:
                        continue
                    if re.match(r"not applicable\b", cell, re.I):
                        continue
                    shown = _plain(cell) or "(empty)"
                    report.add(
                        "claims",
                        f"The table cell '{shown}' is not a placeholder. In a draft, every data cell of a results table in the "
                        f"section '{RESULT_TABLES}' must be a <<RESULT: ...>> placeholder or 'Not applicable'.",
                        row_number,
                    )
            rows = [(number, stripped)] if stripped.startswith("|") else []


def collect_markers(doc: Doc, report: Report) -> None:
    """Valid markers into report.placeholders (everywhere, code blocks included); malformed ones are findings."""
    for i, line in enumerate(doc.lines):
        number = i + 1
        for m in _VALID_MARKER.finditer(line):
            name = re.sub(r"\s+", " ", m.group(2)).strip()
            if name:
                report.placeholders.append(Placeholder(m.group(1), name, number))
            else:
                report.add("placeholders", f"Malformed placeholder {m.group(0)!r}: it has no name. Write <<{m.group(1)}: short name>>.", number)
        if doc.in_code[i]:
            continue
        rest = _VALID_MARKER.sub(" ", line)
        stray = [m.group(0) for m in _CANDIDATE_MARKER.finditer(rest)]
        leftovers = _CANDIDATE_MARKER.sub(" ", rest)
        if stray or "<<" in leftovers or ">>" in leftovers:
            shown = stray[0] if stray else ("<<" if "<<" in leftovers else ">>")
            report.add(
                "placeholders",
                f"Malformed placeholder {shown!r}. The only markers are <<RESULT: name>>, <<LINK: name>> and <<OWNER: name>>.",
                number,
            )


def check_placeholders_final(report: Report, limit: int = 15) -> None:
    placeholders = report.placeholders
    if not placeholders:
        return
    kinds = ", ".join(f"{k} {sum(p.kind == k for p in placeholders)}" for k in MARKER_KINDS if any(p.kind == k for p in placeholders))
    report.add(
        "placeholders",
        f"The article still has {_plural(len(placeholders), 'placeholder')} ({kinds}). Fill each one in from the exported results "
        f"(see {FILL_IN_NAME}) before submitting, or run with --draft to check a draft.",
    )
    for p in placeholders[:limit]:
        report.add("placeholders", f"<<{p.kind}: {p.name}>> is still in the text.", p.line)
    if len(placeholders) > limit:
        report.add("placeholders", f"... and {len(placeholders) - limit} more (run with --draft to list them all).")


def check_unfinished(doc: Doc, report: Report) -> None:
    for i, line in enumerate(doc.lines):
        m = _UNFINISHED.search(line)
        if m:
            report.add("unfinished-markers", f"'{m.group(0)}' marks unfinished text. Finish the sentence or remove the marker.", i + 1)


def check_fill_in(report: Report, fill_in: str | None, fill_in_path: Path) -> None:
    if fill_in is None:
        report.add(
            "fill-in",
            f"{FILL_IN_NAME} was not found at {fill_in_path}. In draft mode it must list every placeholder and every "
            "figure that is still to be added.",
        )
        return
    flat = re.sub(r"\s+", " ", fill_in)
    seen = set()
    for p in report.placeholders:
        if p.name in seen:
            continue
        seen.add(p.name)
        if p.name not in flat:
            report.add("fill-in", f"The placeholder '{p.name}' has no line in {FILL_IN_NAME}, so filling it in would not be mechanical.", p.line)


# --- the whole check ---------------------------------------------------------------------------------------------


def _read(path: Path, what: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"cannot read {what} {path}: {type(exc).__name__}: {exc}") from None


def check(article_path: Path | str, sources_path: Path | str, *, draft: bool = False, fill_in_path: Path | str | None = None) -> Report:
    """Run every rule; the result holds the findings and the placeholders. Raises UsageError for an unreadable file."""
    article_path, sources_path = Path(article_path), Path(sources_path)
    text = _read(article_path, "the article")
    try:
        sources = load_sources(sources_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise UsageError(f"cannot use the source list {sources_path}: {' '.join(str(exc).split())}") from None
    fill_in_path = Path(fill_in_path) if fill_in_path else article_path.parent / FILL_IN_NAME
    fill_in = _read(fill_in_path, "the fill-in list") if draft and fill_in_path.is_file() else None

    doc = Doc(strip_comments(text))  # what a reader sees: the rules about content read this
    raw = Doc(text)  # what is in the file: markers and unfinished text are found even inside a comment
    report = Report(draft=draft)
    check_headings(doc, report)
    check_heading_levels(doc, report)
    check_sources(doc, sources, report)
    check_mentions(doc, report)
    check_disclosure(doc, report)
    check_sections_have_prose(doc, report)
    check_figures(doc, article_path, fill_in, report)
    check_hype(doc, report)
    if draft:
        check_claims(doc, report)
    collect_markers(raw, report)
    if not draft:
        check_placeholders_final(report)
    check_unfinished(raw, report)
    if draft:
        check_fill_in(report, fill_in, fill_in_path)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("article", type=Path, help="the article, for example article/ARTICLE.md")
    ap.add_argument("sources", type=Path, help="the source list, data/sources.yaml")
    ap.add_argument("--draft", action="store_true", help="allow <<RESULT/LINK/OWNER: ...>> placeholders and list them; also check FILL_IN.md and the claims guard")
    ap.add_argument("--fill-in", type=Path, default=None, help=f"the fill-in list (default: {FILL_IN_NAME} beside the article); draft mode only")
    args = ap.parse_args(argv)
    try:
        report = check(args.article, args.sources, draft=args.draft, fill_in_path=args.fill_in)
    except UsageError as exc:
        print(f"check_article: {exc}", file=sys.stderr)
        return 2
    for finding in report.findings:
        print(finding.render())
    if report.findings:
        print(f"{_plural(len(report.findings), 'problem')} found in {args.article}. The article does not pass.")
        return 1
    mode = " (draft mode)" if args.draft else ""
    print(f"OK{mode}: {args.article} passes every rule.")
    if args.draft:
        kinds = ", ".join(f"{k} {sum(p.kind == k for p in report.placeholders)}" for k in MARKER_KINDS)
        print(f"{_plural(len(report.placeholders), 'placeholder')} remain ({kinds}). Fill each one in before the final run:")
        for p in report.placeholders:
            print(f"  line {p.line:>4}  {p.kind:<6}  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
