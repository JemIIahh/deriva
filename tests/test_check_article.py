"""scripts/check_article.py: the rules a Hugging Face article must pass before the owner submits it (Task 21).

Every test builds a tiny article and a tiny source list under tmp_path. The checker reads the article, the source
manifest (`data/sources.yaml` shape) and, in draft mode, the FILL_IN.md beside the article. Nothing here touches the
network. The last tests run the checker on the real article of the repository.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_article.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_article", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_article"] = module
    spec.loader.exec_module(module)
    return module


check_article = _load_script()

# The nine titles of the organizers' submission checklist (reference/kickoff-slides-text.txt, slide 29), word for word.
HEADINGS = [
    "Problem statement and approach, in brief",
    "Hackathon repos used: which ones, for what, and which features",
    "Models used and their licenses",
    "Datasets and benchmarks used and their licenses",
    "Model artifacts (weights, adapters, …) public, as an HF collection",
    "Data artifacts (datasets, logs, benchmarks, …) public, as an HF collection",
    "Experiments run, findings and what else was tested",
    "With more credits: your approach, before and after results",
    "Teammates tagged",
]
PROBLEM, REPOS, MODELS, DATASETS, MODEL_ARTIFACTS, DATA_ARTIFACTS, EXPERIMENTS, APPROACH, TEAMMATES = HEADINGS

# --- a small, valid article ------------------------------------------------------------------------------------------
# Each named piece below is one sentence or paragraph a rule looks for, so a test can drop exactly one.

PIECES = {
    "question": (
        "Does fine-tuning a small multilingual model on a harmless task make it less safe? Is the change the same in "
        "every language? This study measures it in five languages and tests one repair that needs no retraining."
    ),
    "repos": (
        "CuratorKIT builds the fine-tuning data. AlignTune runs the fine-tuning. SafeTune ReSta is the repair step. "
        "AuditKIT was not used for the numbers."
    ),
    "aligntune_mit": "AlignTune is released under the MIT licence.",
    "curatorkit_lsal": (
        "CuratorKIT is under the Lexsi Labs Source Available License 1.1 (LSAL-1.1), which allows research use only."
    ),
    "aligntune_depends": "AlignTune depends on CuratorKIT, so that licence applies to anyone who installs AlignTune.",
    "unsloth": "The AlignTune package bundles unsloth, which has AGPL parts.",
    "mergekit": "It also bundles mergekit, which is LGPL.",
    "trl": "This study uses the TRL backend only, which switches the unsloth backend off.",
    "tiny_licence": "Tiny Aya Global is released under the CC-BY-NC-4.0 licence.",
    "gated": "Tiny Aya is gated on Hugging Face, so you must accept its terms before you can download it.",
    "aup": "Cohere Labs also applies its Acceptable Use Policy to the model.",
    "nllb": "The machine translations were made with NLLB-200, whose weights are CC-BY-NC-4.0.",
    "multijail": (
        "DAMO-NLP-SG/MultiJail is MIT licensed, and its ethics statement says the prompts are for academic and "
        "ethical use only."
    ),
    "disclosure": (
        "The work was done by one person together with Claude Code, an AI coding assistant that wrote and reviewed "
        "much of the code and the first draft of this text."
    ),
}

SOURCE_ROWS = {
    "harm": "| fix/harm | mit | harmful prompts |",
    "benign": "| fix/benign | cc-by-4.0 | harmless but edgy prompts |",
    "eval": "| fix/eval | cc-by-sa-4.0 | evaluation only: scores are published, items are not |",
}

PLAIN_COLLECTION = (
    "All models and adapters of the study are published in one public Hugging Face collection, with a model card for "
    "each one. The link is below.\n\n[Model collection](https://huggingface.co/collections/example/models)"
)
PLAIN_DATA = (
    "The test set, the per-item labels and the tables are published in one public dataset collection, with a card "
    "that says what each file holds. The link is below.\n\n[Data collection](https://huggingface.co/collections/example/data)"
)
PLAIN_EXPERIMENTS = (
    "One stronger fine-tuning recipe exists in case the first one does not move the model. Alpha was chosen on the "
    "dev split only, and the report split was kept apart. Every limit of the judges is written down here."
)
PLAIN_APPROACH = (
    "The figure below shows the attack success rate per language for the original model and for the repaired model."
    "\n\n![Heatmap of attack success rate per language, before and after the repair](heatmap.png)"
)


def make_article(*, drop=(), extra="", sections=None, heading_overrides=None) -> str:
    """The nine sections with every named piece in it, minus the pieces named in `drop`."""
    keep = {name: text for name, text in PIECES.items() if name not in drop}
    body = {
        PROBLEM: keep.get("question", ""),
        REPOS: "\n\n".join(
            keep[name] for name in ("repos", "aligntune_mit", "curatorkit_lsal", "aligntune_depends", "unsloth", "mergekit", "trl")
            if name in keep
        ),
        MODELS: "\n\n".join(keep[name] for name in ("tiny_licence", "gated", "aup", "nllb") if name in keep),
        DATASETS: (
            "These are the public sources the study uses, with the licence of each one.\n\n"
            "| Source | Licence | Use |\n| --- | --- | --- |\n"
            + "\n".join(row for name, row in SOURCE_ROWS.items() if name not in drop)
            + ("\n\n" + keep["multijail"] if "multijail" in keep else "")
        ),
        MODEL_ARTIFACTS: PLAIN_COLLECTION,
        DATA_ARTIFACTS: PLAIN_DATA,
        EXPERIMENTS: PLAIN_EXPERIMENTS,
        APPROACH: PLAIN_APPROACH,
        TEAMMATES: "None. " + keep.get("disclosure", "Nobody else worked on it, and nobody else helped in any way at all."),
    }
    body.update(sections or {})
    names = [(heading_overrides or {}).get(h, h) for h in HEADINGS]
    parts = ["# A test article\n"]
    for name, title in zip(HEADINGS, names):
        parts.append(f"## {title}\n\n{body[name]}\n")
    return "\n".join(parts) + extra


def source_entry(name, license_, used_for, **extra):
    return {
        "name": name,
        "url": f"https://example.org/{name}",
        "license": license_,
        "languages": ["en"],
        "used_for": used_for,
        "verified_on": "2026-11-05",
        **extra,
    }


SOURCES = [
    source_entry("fix/harm", "mit", "harmful_prompts"),
    source_entry("fix/benign", "cc-by-4.0", "benign_edgy_prompts"),
    source_entry("fix/eval", "cc-by-sa-4.0", "task_cross_check", redistribute=False),
    source_entry("fix/dropped", "cc-by-nc-4.0", "excluded", notes="non-commercial licence"),
]


def write_case(tmp_path: Path, article: str, *, sources=SOURCES, fill_in=None, image=True) -> tuple[Path, Path]:
    """Write ARTICLE.md, the source manifest, and optionally FILL_IN.md and heatmap.png into tmp_path/article."""
    folder = tmp_path / "article"
    folder.mkdir(exist_ok=True)
    article_path = folder / "ARTICLE.md"
    article_path.write_text(article, encoding="utf-8")
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(yaml.safe_dump(sources, allow_unicode=True), encoding="utf-8")
    if fill_in is not None:
        (folder / "FILL_IN.md").write_text(fill_in, encoding="utf-8")
    if image:
        (folder / "heatmap.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return article_path, sources_path


def run(tmp_path, article, *, draft=False, fill_in=None, sources=SOURCES, image=True):
    article_path, sources_path = write_case(tmp_path, article, sources=sources, fill_in=fill_in, image=image)
    return check_article.check(article_path, sources_path, draft=draft)


def rules(report) -> set[str]:
    return {f.rule for f in report.findings}


def messages(report) -> str:
    return "\n".join(f.render() for f in report.findings)


def cli(tmp_path, article, *flags, fill_in=None, sources=SOURCES, image=True):
    article_path, sources_path = write_case(tmp_path, article, sources=sources, fill_in=fill_in, image=image)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(article_path), str(sources_path), *flags], capture_output=True, text=True
    )


# --- the starting point is clean ---------------------------------------------------------------------------------------


def test_the_small_valid_article_passes_every_rule(tmp_path):
    report = run(tmp_path, make_article())
    assert report.findings == [], messages(report)
    assert report.ok


# --- 1. the nine headings ----------------------------------------------------------------------------------------------


def test_nine_required_headings(tmp_path):
    assert list(check_article.REQUIRED_HEADINGS) == HEADINGS
    assert run(tmp_path, make_article()).ok


SLIDES = ROOT / "reference" / "kickoff-slides-text.txt"


@pytest.mark.skipif(not SLIDES.is_file(), reason="the kickoff slide text is not in this checkout")
def test_required_headings_are_the_organizers_checklist_word_for_word():
    # slide 29 lists the nine sections; the titles are read from that file, never retyped
    import re

    lines = SLIDES.read_text(encoding="utf-8").splitlines()
    start = next(n for n, line in enumerate(lines) if line.strip() == "===== SECTION 29 =====")
    titles = []
    for line in lines[start : start + 16]:
        m = re.match(r"^\s*([1-9])\s+(\S.*?)\s*$", line)
        if m and int(m.group(1)) == len(titles) + 1:
            titles.append(m.group(2))
    assert len(titles) == 9
    assert list(check_article.REQUIRED_HEADINGS) == titles
    assert HEADINGS == titles


@pytest.mark.parametrize("missing", HEADINGS)
def test_a_missing_required_heading_is_named(tmp_path, missing):
    article = make_article().replace(f"## {missing}\n", f"### {missing}\n")
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert f"'{missing}'" in messages(report)
    assert "missing" in messages(report)


def test_headings_in_the_wrong_order_fail(tmp_path):
    article = make_article()
    article = article.replace(f"## {MODEL_ARTIFACTS}", "## @@").replace(f"## {DATA_ARTIFACTS}", f"## {MODEL_ARTIFACTS}").replace("## @@", f"## {DATA_ARTIFACTS}")
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert "order" in messages(report)
    assert DATA_ARTIFACTS in messages(report)


def test_a_duplicated_heading_fails_and_names_both_lines(tmp_path):
    article = make_article(extra=f"\n## {TEAMMATES}\n\nNone again, in plain words, so that the section has some prose to read.\n")
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert f"'{TEAMMATES}'" in messages(report)
    assert "twice" in messages(report)


def test_an_extra_level_two_heading_fails(tmp_path):
    report = run(tmp_path, make_article(extra="\n## Acknowledgements\n\nThanks to everyone who read the draft and told us what was unclear.\n"))
    assert "headings" in rules(report)
    assert "'Acknowledgements'" in messages(report)
    assert "extra" in messages(report).lower()


def test_level_three_subsections_are_allowed(tmp_path):
    article = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n### What failed\n\nNothing has been run yet, so this part stays short and plain.\n"})
    assert run(tmp_path, article).ok


def test_a_heading_inside_a_code_block_is_not_a_heading(tmp_path):
    article = make_article(extra="\n```\n## Not a real section\n```\n")
    assert run(tmp_path, article).ok


def test_heading_text_must_match_exactly(tmp_path):
    article = make_article(heading_overrides={TEAMMATES: "Teammates"})
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert f"'{TEAMMATES}'" in messages(report)


# --- 2. heading levels ---------------------------------------------------------------------------------------------------


def test_a_skipped_heading_level_fails(tmp_path):
    article = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n#### Too deep\n\nThis heading skips level three, which breaks the outline of the page.\n"})
    report = run(tmp_path, article)
    assert "heading-levels" in rules(report)
    assert "Too deep" in messages(report)


def test_a_document_that_starts_at_level_three_fails(tmp_path):
    report = run(tmp_path, "### Lost\n\n" + make_article())
    assert "heading-levels" in rules(report)


# --- 3. every used source, with its licence ---------------------------------------------------------------------------------


def test_every_used_source_license_mentioned(tmp_path):
    assert run(tmp_path, make_article()).ok  # the excluded fix/dropped entry is not required
    for name in ("harm", "benign", "eval"):
        report = run(tmp_path, make_article(drop=(name,)))
        assert "source-licenses" in rules(report), name
        assert f"fix/{name}" in messages(report)


def test_a_source_with_the_wrong_licence_on_its_line_fails(tmp_path):
    article = make_article().replace("| fix/harm | mit |", "| fix/harm | apache-2.0 |")
    report = run(tmp_path, article)
    assert "source-licenses" in rules(report)
    assert "fix/harm" in messages(report)
    assert "mit" in messages(report)


def test_the_licence_must_sit_on_the_line_of_its_source(tmp_path):
    # the licence appears in the table, but on another source's line
    article = make_article().replace("| fix/harm | mit |", "| fix/harm | n/a |")
    report = run(tmp_path, article)
    assert "source-licenses" in rules(report)


def test_licence_spelling_may_differ_in_case_and_separators(tmp_path):
    article = make_article().replace("cc-by-sa-4.0", "CC BY-SA 4.0").replace("| cc-by-4.0 |", "| CC-BY-4.0 |")
    assert run(tmp_path, article).ok


def test_a_source_named_in_the_models_section_counts(tmp_path):
    article = make_article(drop=("harm",))
    article = article.replace(
        "Cohere Labs also applies", "fix/harm is also listed here under mit. Cohere Labs also applies"
    )
    assert run(tmp_path, article).ok


def test_a_source_named_only_in_another_section_does_not_count(tmp_path):
    article = make_article(drop=("harm",)).replace(f"## {TEAMMATES}\n\n", f"## {TEAMMATES}\n\nfix/harm mit is mentioned here only.\n\n")
    report = run(tmp_path, article)
    assert "source-licenses" in rules(report)


def test_a_redistribute_false_source_must_be_marked_evaluation_only(tmp_path):
    article = make_article().replace("evaluation only: scores are published, items are not", "used for scoring")
    report = run(tmp_path, article)
    assert "source-licenses" in rules(report)
    assert "fix/eval" in messages(report)
    assert "evaluation only" in messages(report)


def test_evaluation_dash_only_spelling_is_accepted(tmp_path):
    assert run(tmp_path, make_article().replace("evaluation only:", "evaluation-only:")).ok


# --- 4. disclosure ------------------------------------------------------------------------------------------------------------


def test_disclosure_paragraph_present(tmp_path):
    assert run(tmp_path, make_article()).ok
    report = run(tmp_path, make_article(drop=("disclosure",)))
    assert "disclosure" in rules(report)
    assert "Claude Code" in messages(report)


def test_disclosure_must_say_it_is_an_ai_assistant(tmp_path):
    article = make_article(sections={TEAMMATES: "None. The work was done with Claude Code and a lot of coffee, mostly at night."})
    report = run(tmp_path, article)
    assert "disclosure" in rules(report)


# --- 5. licences and tools --------------------------------------------------------------------------------------------------


MENTION_PIECES = [
    ("tiny_licence", "Tiny Aya"),
    ("gated", "gated"),
    ("aup", "Acceptable Use Policy"),
    ("aligntune_mit", "AlignTune"),
    ("curatorkit_lsal", "LSAL-1.1"),
    ("aligntune_depends", "depends"),
    ("unsloth", "AGPL"),
    ("mergekit", "LGPL"),
    ("trl", "TRL"),
    ("nllb", "NLLB"),
    ("multijail", "ethics statement"),
]


@pytest.mark.parametrize("piece,word", MENTION_PIECES)
def test_each_required_licence_and_tool_is_mentioned(tmp_path, piece, word):
    report = run(tmp_path, make_article(drop=(piece,)))
    assert "licenses-and-tools" in rules(report), piece
    assert word.lower() in messages(report).lower(), messages(report)


@pytest.mark.parametrize("toolkit", ["SafeTune", "AuditKIT"])
def test_the_toolkits_are_named_in_the_repos_section(tmp_path, toolkit):
    # CuratorKIT and AlignTune are named by their licence sentences; SafeTune and AuditKIT only by this paragraph
    article = make_article().replace(PIECES["repos"], PIECES["repos"].replace(toolkit, "a toolkit"))
    assert article != make_article()
    report = run(tmp_path, article)
    assert "licenses-and-tools" in rules(report), toolkit
    assert toolkit in messages(report)


def test_a_mention_in_the_wrong_section_does_not_count(tmp_path):
    # the Tiny Aya licence belongs with the models, not in the teammates section
    article = make_article(drop=("tiny_licence",)).replace(f"## {TEAMMATES}\n\n", f"## {TEAMMATES}\n\n" + PIECES["tiny_licence"] + "\n\n")
    report = run(tmp_path, article)
    assert "licenses-and-tools" in rules(report)


# --- 6. plain language ----------------------------------------------------------------------------------------------------------


def test_a_section_of_only_links_fails(tmp_path):
    article = make_article(sections={MODEL_ARTIFACTS: "https://huggingface.co/collections/example/models\n\n[the data](https://huggingface.co/collections/example/data)"})
    report = run(tmp_path, article)
    assert "empty-or-links-only" in rules(report)
    assert f"'{MODEL_ARTIFACTS}'" in messages(report)
    assert "links" in messages(report)


def test_an_empty_section_fails(tmp_path):
    report = run(tmp_path, make_article(sections={DATA_ARTIFACTS: ""}))
    assert "empty-or-links-only" in rules(report)
    assert f"'{DATA_ARTIFACTS}'" in messages(report)


def test_a_figure_without_alt_text_fails(tmp_path):
    article = make_article(sections={APPROACH: PLAIN_APPROACH.replace("![Heatmap of attack success rate per language, before and after the repair]", "![]")})
    report = run(tmp_path, article)
    assert "figures" in rules(report)
    assert "alt text" in messages(report)


def test_a_figure_with_one_word_of_alt_text_fails(tmp_path):
    article = make_article(sections={APPROACH: PLAIN_APPROACH.replace("![Heatmap of attack success rate per language, before and after the repair]", "![heatmap]")})
    assert "figures" in rules(run(tmp_path, article))


def test_the_per_language_heatmap_must_be_referenced(tmp_path):
    article = make_article(sections={APPROACH: PLAIN_APPROACH.split("\n\n")[0]})
    report = run(tmp_path, article)
    assert "figures" in rules(report)
    assert "heatmap" in messages(report)


def test_the_heatmap_may_sit_in_a_subfolder_or_at_a_url(tmp_path):
    for path in ("figures/heatmap.png", "https://example.org/files/heatmap.png"):
        article = make_article(sections={APPROACH: PLAIN_APPROACH.replace("(heatmap.png)", f"({path})")})
        folder = tmp_path / path.replace("/", "_").replace(":", "_")
        folder.mkdir()
        article_path, sources_path = write_case(folder, article)
        if not path.startswith("http"):
            (article_path.parent / "figures").mkdir(exist_ok=True)
            (article_path.parent / path).write_bytes(b"\x89PNG")
        assert check_article.check(article_path, sources_path).ok, path


@pytest.mark.parametrize("word", ["groundbreaking", "state-of-the-art", "revolutionary", "dramatically", "breakthrough"])
def test_hype_words_fail(tmp_path, word):
    report = run(tmp_path, make_article(extra=f"\nThis is a {word} study that readers should love.\n"))
    assert "hype" in rules(report)
    assert word in messages(report)


# --- 7. placeholders and unfinished text ------------------------------------------------------------------------------------------

FILL_IN_OK = "# Fill in\n\nheatmap.png is still to be added.\n"


def draft_article(extra_sentence):
    return make_article(sections={APPROACH: PLAIN_APPROACH + "\n\n" + extra_sentence})


def in_problem_section(sentence):
    # section 1 has no figure rule, so only the claims patterns can catch the sentence here
    return make_article(sections={PROBLEM: PIECES["question"] + "\n\n" + sentence})


def test_draft_mode_lists_placeholders_and_passes(tmp_path):
    article = draft_article(
        "Result: <<RESULT: pooled attack success per language>>. Models: <<LINK: model collection url>>. "
        "Date: <<OWNER: submission date>>."
    )
    fill_in = FILL_IN_OK + "pooled attack success per language\nmodel collection url\nsubmission date\n"
    report = run(tmp_path, article, draft=True, fill_in=fill_in)
    assert report.ok, messages(report)
    assert [(p.kind, p.name) for p in report.placeholders] == [
        ("RESULT", "pooled attack success per language"),
        ("LINK", "model collection url"),
        ("OWNER", "submission date"),
    ]


def test_without_draft_any_placeholder_fails(tmp_path):
    for kind in ("RESULT", "LINK", "OWNER"):
        article = draft_article(f"Value: <<{kind}: some value>>.")
        report = run(tmp_path, article, draft=False)
        assert "placeholders" in rules(report), kind
        text = messages(report)
        assert "placeholder" in text and "some value" in text, text


def test_the_final_run_message_says_how_many_placeholders_remain(tmp_path):
    article = draft_article("A <<RESULT: one>> and a <<RESULT: two>> and a <<LINK: three>>.")
    report = run(tmp_path, article, draft=False)
    assert "3 placeholders" in messages(report)


@pytest.mark.parametrize("bad", ["<<RESULT>>", "<<RESULT: >>", "<<NUMBER: 12>>", "<<RESULT: no end", "RESULT: no start>>"])
def test_a_malformed_marker_fails_even_in_draft_mode(tmp_path, bad):
    report = run(tmp_path, draft_article(f"Value: {bad} here."), draft=True, fill_in=FILL_IN_OK)
    assert "placeholders" in rules(report), bad
    assert "malformed" in messages(report).lower()


@pytest.mark.parametrize("marker", ["TODO", "TBD", "XXX", "{{value}}", "todo"])
@pytest.mark.parametrize("draft", [False, True])
def test_unfinished_text_fails_in_both_modes(tmp_path, marker, draft):
    report = run(tmp_path, draft_article(f"The next number is {marker} for now."), draft=draft, fill_in=FILL_IN_OK)
    assert "unfinished-markers" in rules(report)
    assert marker in messages(report)


def test_a_placeholder_inside_a_code_block_still_counts(tmp_path):
    article = make_article(extra="\n```\nvalue = <<RESULT: hidden in code>>\n```\n")
    report = run(tmp_path, article, draft=False)
    assert "placeholders" in rules(report)


def test_a_word_that_merely_contains_todo_is_fine(tmp_path):
    assert run(tmp_path, draft_article("The mastodon and the xxxl shirt are not markers.")).ok


# --- 8. the claims guard (draft only) ------------------------------------------------------------------------------------------------

CLAIMS = [
    "Fine-tuning made the model less safe in Swahili.",
    "The fine-tune makes Tiny Aya less safe in Hindi.",
    "The repair restored safety in every language.",
    "ReSta recovered the refusals that the fine-tune had removed.",
    "The repair worked for Arabic and failed for Hinglish.",
    "Attack success rose by twelve points in Swahili.",
    "Over-refusal dropped after the repair.",
    "The drift was larger in Hinglish than in English.",
    "We found that the gap widened.",
    "The results show that the patch helps.",
]


@pytest.mark.parametrize("sentence", CLAIMS)
def test_a_draft_sentence_that_asserts_a_result_fails(tmp_path, sentence):
    report = run(tmp_path, draft_article(sentence), draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report), sentence
    assert "has not been measured" in messages(report)


@pytest.mark.parametrize("sentence", CLAIMS)
def test_the_same_claim_inside_a_result_placeholder_passes(tmp_path, sentence):
    article = draft_article(f"<<RESULT: {sentence.rstrip('.')}>>.")
    report = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK + sentence.rstrip(".") + "\n")
    assert "claims" not in rules(report), messages(report)


def test_questions_and_plain_method_text_are_not_claims(tmp_path):
    safe = [
        "Does fine-tuning make the model less safe in some languages?",
        "The repair adds a scaled copy of the post-training step to the fine-tuned weights.",
        "A model that is broken says nothing useful, so it can look safe to a judge.",
        "The sanity gate fails a cell whose task accuracy falls more than ten points below the fine-tuned model.",
    ]
    for sentence in safe:
        report = run(tmp_path, in_problem_section(sentence), draft=True, fill_in=FILL_IN_OK)
        assert "claims" not in rules(report), (sentence, messages(report))


def test_repaired_as_a_describing_word_is_not_a_claim(tmp_path):
    # "the repaired model", "a repaired setting" name an arm; they do not say the repair worked
    safe = [
        "Before a repaired setting may enter the safety table, five checks run on its dev replies.",
        "The repaired model keeps its safety numbers in the table with an interval.",
        "Each repaired arm is compared with the original on the same prompts for refusals.",
    ]
    for sentence in safe:
        report = run(tmp_path, draft_article(sentence), draft=True, fill_in=FILL_IN_OK)
        assert "claims" not in rules(report), (sentence, messages(report))
    # the verb use still fails
    for sentence in ("The team repaired the safety of the model.", "The patch repaired refusals in Swahili."):
        report = run(tmp_path, draft_article(sentence), draft=True, fill_in=FILL_IN_OK)
        assert "claims" in rules(report), sentence


def test_the_claims_guard_does_not_run_on_the_final_article(tmp_path):
    # once the results are measured, the article may say what they are
    assert run(tmp_path, draft_article("Fine-tuning made the model less safe in Swahili."), draft=False).ok


# --- 8b. fix round 1: the guard must also catch plain rephrasings ------------------------------------------------------------


REPHRASED_CLAIMS = [
    "After fine-tuning, the model complied with more harmful Swahili requests.",
    "Safety in Hindi was eroded by the fine-tune.",
    "The drifted model is less safe than the original in Arabic.",
    "ReSta brought refusals back in every language.",
    "Hinglish showed the largest drift of the five languages.",
    "The repair cut attack success by half in Swahili.",
    "Attack success in Arabic went from 12% to 31%.",
    "Fine-tuning weakened the model's refusals in Hindi.",
    "Our measurements show that drift is uneven across languages.",
    "Drift was found in all four non-English languages.",
    "The repaired model is safer than the drifted one in every language.",
    "The gate passed, so the study continued to the repair.",
]


@pytest.mark.parametrize("sentence", REPHRASED_CLAIMS)
def test_a_rephrased_result_sentence_fails_in_draft_mode(tmp_path, sentence):
    report = run(tmp_path, in_problem_section(sentence), draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report), sentence
    assert "has not been measured" in messages(report)


@pytest.mark.parametrize("sentence", REPHRASED_CLAIMS)
def test_a_rephrased_result_inside_a_result_placeholder_passes(tmp_path, sentence):
    article = in_problem_section(f"<<RESULT: {sentence.rstrip('.')}>>.")
    report = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK + sentence.rstrip(".") + "\n")
    assert "claims" not in rules(report), messages(report)


@pytest.mark.parametrize("sentence", REPHRASED_CLAIMS)
def test_a_rephrased_result_sentence_is_also_caught_in_the_experiments_section(tmp_path, sentence):
    article = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n" + sentence})
    assert "claims" in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)), sentence


def test_a_heading_that_states_a_result_fails(tmp_path):
    heading = "### Fine-tuning made the model less safe in Swahili"
    article = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n" + heading + "\n\nThe text under the heading is plain and says nothing about a result."})
    report = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report)
    assert "Fine-tuning made the model less safe in Swahili" in messages(report)


def test_a_heading_that_is_a_question_or_plain_method_is_fine(tmp_path):
    article = make_article(
        sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n### Does fine-tuning make the model less safe?\n\nWe ask it of five languages in turn, with the same prompts.\n\n### The safeguards\n\nThey are listed here in plain words."}
    )
    assert "claims" not in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK))


@pytest.mark.parametrize("text", ["The rate is 31% in Arabic.", "It is 12 points higher in Hindi.", "The gap is 5 percentage points.", "A 10-point rule applies here.", "It is 3 percent."])
@pytest.mark.parametrize("section", ["EXPERIMENTS", "APPROACH"])
def test_a_figure_outside_a_placeholder_in_sections_7_and_8_fails_in_draft_mode(tmp_path, text, section):
    name = {"EXPERIMENTS": EXPERIMENTS, "APPROACH": APPROACH}[section]
    base = {"EXPERIMENTS": PLAIN_EXPERIMENTS, "APPROACH": PLAIN_APPROACH}[section]
    report = run(tmp_path, make_article(sections={name: base + "\n\n" + text}), draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report), text
    assert "placeholder" in messages(report)
    assert name in messages(report)


def test_a_figure_inside_a_placeholder_in_those_sections_passes(tmp_path):
    article = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\nThe rate is <<RESULT: rate in Arabic, 31%>> in Arabic."})
    assert "claims" not in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK + "rate in Arabic, 31%\n"))


def test_method_figures_are_fine_in_the_earlier_sections_and_in_final_mode(tmp_path):
    text = "The sanity gate allows at most 20% empty replies and a fall of at most 10 points in task accuracy."
    early = make_article(sections={MODELS: PIECES["tiny_licence"] + "\n\n" + PIECES["gated"] + "\n\n" + PIECES["aup"] + "\n\n" + PIECES["nllb"] + "\n\n" + text})
    assert "claims" not in rules(run(tmp_path, early, draft=True, fill_in=FILL_IN_OK))
    late = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n" + text})
    assert run(tmp_path, late, draft=False).ok  # the final article may state figures


def test_numbers_that_are_not_percentages_or_points_are_not_figures(tmp_path):
    text = "A second judge labels 60 replies per language, and at least 30 pairs must be comparable."
    article = make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n" + text})
    assert "claims" not in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK))


# --- 8c. fix round 2: alt text and Markdown marks must not hide a result ------------------------------------------------------


@pytest.mark.parametrize("section", ["EXPERIMENTS", "APPROACH"])
def test_a_result_in_the_alt_text_of_a_figure_fails_in_draft_mode(tmp_path, section):
    name = {"EXPERIMENTS": EXPERIMENTS, "APPROACH": APPROACH}[section]
    base = {"EXPERIMENTS": PLAIN_EXPERIMENTS, "APPROACH": PLAIN_APPROACH}[section]
    image = "![Heatmap: Swahili at 66% attack success after the repair, the highest of the five languages](heatmap.png)"
    report = run(tmp_path, make_article(sections={name: base + "\n\n" + image}), draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report), section
    assert "66%" in messages(report)


def test_alt_text_is_still_required_and_a_plain_description_still_passes(tmp_path):
    article = make_article(sections={APPROACH: PLAIN_APPROACH.replace("![Heatmap of attack success rate per language, before and after the repair]", "![]")})
    assert "figures" in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK))
    assert run(tmp_path, make_article(), draft=True, fill_in=FILL_IN_OK).ok


def test_a_claim_in_the_alt_text_of_a_figure_is_a_claim_in_any_section(tmp_path):
    image = "![Heatmap showing that fine-tuning made the model less safe in Swahili](heatmap.png)"
    article = make_article(sections={PROBLEM: PIECES["question"] + "\n\n" + image})
    assert "claims" in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK))


MARKED_UP_CLAIMS = {
    "bold": "Fine-tuning made the model less **safe** in Swahili.",
    "bold-both": "Fine-tuning **made the model less safe** in Swahili.",
    "italic": "Fine-tuning made the model _less safe_ in Swahili.",
    "star-italic": "Fine-tuning made the model *less safe* in Swahili.",
    "code": "Fine-tuning made the model `less safe` in Swahili.",
    "link": "Fine-tuning made the [model](x.md) less safe in Swahili.",
    "link-text": "Fine-tuning [made the model less safe](https://example.org/a.b) in Swahili.",
    "blockquote": "> Fine-tuning made the model\n> less safe in Swahili.",
    "blockquote-wrapped-inside-the-phrase": "> Fine-tuning made the model less\n> safe in Swahili.",
    "nested-blockquote": "> > Fine-tuning made the model less\n> > safe in Swahili.",
    "nbsp": "Fine-tuning made the model less safe in Swahili.",
    "narrow-nbsp": "Fine-tuning made the model less safe in Swahili.",
}


@pytest.mark.parametrize("sentence", list(MARKED_UP_CLAIMS.values()), ids=list(MARKED_UP_CLAIMS))
def test_markdown_marks_and_odd_spaces_do_not_hide_a_claim(tmp_path, sentence):
    for article in (in_problem_section(sentence), draft_article(sentence)):
        report = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)
        assert "claims" in rules(report), sentence


def test_a_marked_up_claim_inside_a_result_placeholder_still_passes(tmp_path):
    article = in_problem_section("<<RESULT: fine-tuning made the model _less **safe**_ in Swahili>>.")
    assert "claims" not in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK + "fine-tuning made the model _less **safe**_ in Swahili\n"))


def test_plain_text_with_links_and_marks_is_still_not_a_claim(tmp_path):
    safe = [
        "The **repair** is a [ReSta](https://example.org/resta) step that needs no retraining.",
        "> The report split is _frozen_ before any model runs on it.",
        "The file `resta_patch.json` records the settings of the repair.",
    ]
    for sentence in safe:
        report = run(tmp_path, in_problem_section(sentence), draft=True, fill_in=FILL_IN_OK)
        assert "claims" not in rules(report), (sentence, messages(report))


# --- 8d. fix round 2: Sections 7 and 8 and the front matter may not say how a language or a model did ---------------------------


def in_experiments(sentence):
    return make_article(sections={EXPERIMENTS: PLAIN_EXPERIMENTS + "\n\n" + sentence})


def in_front_matter(sentence):
    """The lines under the title, before the first level-2 heading."""
    return make_article().replace("# A test article\n", "# A test article\n\n" + sentence + "\n", 1)


def in_data_limits(sentence):
    """Section 6 holds method text, where a language name next to a comparison is allowed."""
    return make_article(sections={DATA_ARTIFACTS: PLAIN_DATA + "\n\n" + sentence})


SCOPED = {"front matter": in_front_matter, "section 7": in_experiments, "section 8": draft_article}

REVIEW_SENTENCES = [
    "Swahili is the most affected language.",
    "Hindi was hit hardest.",
    "The repair improved Hindi and Arabic.",
    "ReSta beats the naive blend.",
    "Refusal rates dropped sharply after training.",
    "Safety degraded in Swahili.",
    "Attack success was higher after fine-tuning.",
]

MORE_SCOPED_RESULTS = [
    # a language or a model next to a comparison
    "The drifted model is worse than the original.",
    "Hindi is the safest language of the five.",
    "Hinglish has fewer refusals than English.",
    "A2 looks better than A1 in Arabic.",
    "The naive blend is more permissive than the repair.",
    "Swahili is the weakest language.",
    "English is unchanged.",
    "Arabic is the hardest case.",
    "The original model is stable and Hindi is flat.",
    # a language or a model next to a verb that says what happened
    "English stayed flat.",
    "Arabic held up.",
    "A1 lost refusals in Hinglish.",
    "The naive blend failed.",
    "A2 passed every check.",
    "Hindi gained from the repair.",
    "Swahili showed the biggest change.",
    "The repaired model became safe.",
    "BASE remained stable.",
    "Hinglish crossed the line.",
    "The repair works.",
    "A2n helped in Arabic.",
    "A1 hurt Hindi.",
    "The repair moved Arabic.",
    "English fell.",
    # a measure (usually without a model name) next to a verb or a comparison
    "Scores fell in every language.",
    "Caution rose after the repair.",
    "The refusal rate was lower after the repair.",
    "Task accuracy stayed the same.",
    "Attack success improved.",
    "Over-refusal held steady.",
    "Safety was worse.",
]


@pytest.mark.parametrize("where", list(SCOPED))
@pytest.mark.parametrize("sentence", REVIEW_SENTENCES + MORE_SCOPED_RESULTS)
def test_a_result_about_a_language_or_a_model_fails_in_sections_7_and_8_and_the_front_matter(tmp_path, where, sentence):
    report = run(tmp_path, SCOPED[where](sentence), draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report), (where, sentence)
    assert "has not been measured" in messages(report)


@pytest.mark.parametrize("where", list(SCOPED))
@pytest.mark.parametrize("sentence", REVIEW_SENTENCES)
def test_the_same_result_inside_a_placeholder_passes_there(tmp_path, where, sentence):
    name = sentence.rstrip(".")
    report = run(tmp_path, SCOPED[where](f"<<RESULT: {name}>>."), draft=True, fill_in=FILL_IN_OK + name + "\n")
    assert "claims" not in rules(report), (where, messages(report))


def test_the_new_patterns_are_scoped_to_the_results_sections_and_the_front_matter(tmp_path):
    # method text in Section 6 may put a language next to a comparison
    for sentence in ("Hindi has more fine-tuning rows than Swahili does.", "Hinglish has no fine-tuning rows and Arabic has the fewest checks."):
        report = run(tmp_path, in_data_limits(sentence), draft=True, fill_in=FILL_IN_OK)
        assert "claims" not in rules(report), (sentence, messages(report))


PLAIN_SCOPED_TEXT = [
    "Every model version is run on the prompts of the report split in each language.",
    "Each row is one language, and each column is one model version.",
    "The repaired model is compared with the original model on the same prompts.",
    "The first column is the fine-tuned model and the second is the repaired model at the chosen alpha.",
    "Hindi and Arabic are read from the same table as English.",
    "Does the repair work in Swahili?",
    "A hatched row means that the change between the two columns is inside the noise.",
    "The held-out questions are kept apart from the training rows.",
    "Where the repaired model and the naive blend differ, that difference is what ReSta adds beyond plain averaging.",
    "The test set holds requests in every language and the judges read each reply.",
]


@pytest.mark.parametrize("where", list(SCOPED))
@pytest.mark.parametrize("sentence", PLAIN_SCOPED_TEXT)
def test_plain_text_that_names_a_language_or_a_model_is_not_a_claim(tmp_path, where, sentence):
    report = run(tmp_path, SCOPED[where](sentence), draft=True, fill_in=FILL_IN_OK)
    assert "claims" not in rules(report), (where, sentence, messages(report))


def test_a_scoped_result_stays_a_claim_through_markdown_marks(tmp_path):
    for sentence in ("Swahili is the **most** affected language.", "The repair _improved_ [Hindi](h.md) and Arabic.", "> Safety\n> degraded in Swahili."):
        assert "claims" in rules(run(tmp_path, in_experiments(sentence), draft=True, fill_in=FILL_IN_OK)), sentence



def test_the_script_says_plainly_that_the_claims_guard_is_a_backstop_and_cannot_be_complete():
    doc = " ".join(check_article.__doc__.split())
    assert "text-level backstop" in doc
    assert "cannot be complete" in doc
    assert "A person must read the draft" in doc


# --- 8e. fix round 2: decimals, number words, table cells and code in Sections 7 and 8 -------------------------------------------


def in_approach(text):
    return make_article(sections={APPROACH: PLAIN_APPROACH + "\n\n" + text})


FIGURE_SHAPES = [
    "The weight is 0.25 for the second setting.",
    "A cut-off of .5 applies here.",
    "The change was twelve points.",
    "The change was twenty-five percent.",
    "The change was ten per cent.",
    "The change was 12 pts.",
    "The change was fifteen pct.",
    "The change was two percentage points.",
    "The change was a quarter.",
    "Only a quarter of the replies changed.",
    "The change was half.",
    "The change was 1 in 4.",
    "The change was one in four.",
    "The change was 3x.",
    "The change was 3 x.",
    "The change was 2.5 times.",
]


@pytest.mark.parametrize("sentence", FIGURE_SHAPES)
@pytest.mark.parametrize("where", ["section 7", "section 8"])
def test_a_decimal_or_a_number_word_with_a_unit_fails_in_sections_7_and_8(tmp_path, where, sentence):
    article = {"section 7": in_experiments, "section 8": in_approach}[where](sentence)
    report = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)
    assert "claims" in rules(report), (where, sentence)
    assert "placeholder" in messages(report)


@pytest.mark.parametrize("sentence", FIGURE_SHAPES)
def test_the_same_figure_shapes_pass_in_the_method_sections_and_inside_a_placeholder(tmp_path, sentence):
    assert "claims" not in rules(run(tmp_path, in_problem_section(sentence), draft=True, fill_in=FILL_IN_OK)), sentence
    name = sentence.rstrip(".")
    article = in_approach(f"<<RESULT: {name}>>.")
    assert "claims" not in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK + name + "\n")), sentence


def test_plain_numbers_and_names_with_digits_are_not_figures(tmp_path):
    safe = [
        "One judge reads each reply, and five languages are compared.",
        "A second judge labels 60 replies per language, and at least 30 pairs must be comparable.",
        "The study has 150 harmful prompts per language, in two splits.",
        "The judge is Qwen3-4B and the licence is CC-BY-NC-4.0.",
        "The files are tables.csv, reliability.json and alpha_curve.png.",
        "Cohen's kappa runs from minus one to one.",
    ]
    for sentence in safe:
        for article in (in_experiments(sentence), in_approach(sentence)):
            assert "claims" not in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)), sentence


RESULT_TABLE_HEAD = "| Language | Original | Drifted |\n|---|---|---|\n"


def with_table(rows):
    return in_approach(RESULT_TABLE_HEAD + rows)


@pytest.mark.parametrize(
    "row",
    [
        "| English | yes | no |",
        "| English | <<RESULT: rate>> | high |",
        "| English | n/a | <<RESULT: rate>> |",
        "| English | <<RESULT: rate>> | <<RESULT: rate>> and more text |",
        "| English | <<RESULT: rate>> | |",
        "| English | see the figure | <<RESULT: rate>> |",
    ],
)
def test_a_data_cell_in_a_section_8_table_must_be_a_placeholder_or_not_applicable(tmp_path, row):
    report = run(tmp_path, with_table(row), draft=True, fill_in=FILL_IN_OK + "rate\n")
    assert "claims" in rules(report), row
    assert "not a placeholder" in messages(report)


def test_placeholders_and_not_applicable_cells_pass_and_headers_and_labels_are_not_data(tmp_path):
    rows = (
        "| English | <<RESULT: rate>> | <<RESULT: rate>> |\n"
        "| Hinglish | <<RESULT: rate>> | Not applicable: no closed-form Hinglish data exists. |\n"
        "| Swahili | <<RESULT: rate>> / <<RESULT: interval>> | not applicable |"
    )
    report = run(tmp_path, with_table(rows), draft=True, fill_in=FILL_IN_OK + "rate\ninterval\n")
    assert "claims" not in rules(report), messages(report)


def test_the_table_cell_rule_covers_every_table_in_section_8_and_only_there(tmp_path):
    two_tables = RESULT_TABLE_HEAD + "| English | <<RESULT: rate>> | <<RESULT: rate>> |\n\nThe second table follows.\n\n" + RESULT_TABLE_HEAD + "| Hindi | yes | <<RESULT: rate>> |"
    assert "claims" in rules(run(tmp_path, in_approach(two_tables), draft=True, fill_in=FILL_IN_OK + "rate\n"))
    in_section_7 = in_experiments(RESULT_TABLE_HEAD + "| English | yes | no |")
    assert "claims" not in rules(run(tmp_path, in_section_7, draft=True, fill_in=FILL_IN_OK))
    in_section_6 = in_data_limits(RESULT_TABLE_HEAD + "| English | yes | no |")
    assert "claims" not in rules(run(tmp_path, in_section_6, draft=True, fill_in=FILL_IN_OK))


@pytest.mark.parametrize("code", ["Swahili: 66%", "alpha = 0.5", "drift = twelve points"])
@pytest.mark.parametrize("where", ["section 7", "section 8"])
def test_a_figure_inside_a_code_block_in_sections_7_and_8_fails_too(tmp_path, where, code):
    article = {"section 7": in_experiments, "section 8": in_approach}[where](f"```\n{code}\n```")
    assert "claims" in rules(run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)), (where, code)


def test_a_code_block_in_the_method_sections_may_hold_settings(tmp_path):
    assert "claims" not in rules(run(tmp_path, in_problem_section("```\nalpha = 0.5\nlearning_rate = 1e-4\n```"), draft=True, fill_in=FILL_IN_OK))


# --- 9. FILL_IN.md and the figure file -----------------------------------------------------------------------------------------------


def test_draft_mode_needs_a_fill_in_file_beside_the_article(tmp_path):
    report = run(tmp_path, make_article(), draft=True, fill_in=None)
    assert "fill-in" in rules(report)
    assert "FILL_IN.md" in messages(report)


def test_every_placeholder_must_have_a_line_in_fill_in(tmp_path):
    article = draft_article("Value: <<RESULT: the unlisted number>>.")
    report = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)
    assert "fill-in" in rules(report)
    assert "the unlisted number" in messages(report)


def test_a_figure_that_does_not_exist_yet_must_be_listed_in_fill_in(tmp_path):
    report = run(tmp_path, make_article(), draft=True, fill_in="# Fill in\n\nNothing about the picture here.\n", image=False)
    assert "fill-in" in rules(report)
    assert "heatmap.png" in messages(report)
    assert run(tmp_path, make_article(), draft=True, fill_in=FILL_IN_OK, image=False).ok


def test_in_the_final_run_the_figure_file_must_exist(tmp_path):
    report = run(tmp_path, make_article(), draft=False, image=False)
    assert "figures" in rules(report)
    assert "heatmap.png" in messages(report)
    assert "does not exist" in messages(report)


# --- 10. the command line ---------------------------------------------------------------------------------------------------------------------


def test_exit_code_zero_and_a_plain_line_when_the_article_passes(tmp_path):
    result = cli(tmp_path, make_article())
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK")


def test_exit_code_one_and_plain_messages_when_a_rule_fails(tmp_path):
    result = cli(tmp_path, make_article(drop=("disclosure",)))
    assert result.returncode == 1
    assert "FAIL" in result.stdout and "[disclosure]" in result.stdout
    assert "Traceback" not in result.stderr


def test_draft_flag_changes_the_exit_code_for_a_draft(tmp_path):
    article = draft_article("The number is <<RESULT: pooled drift>>.")
    fill_in = FILL_IN_OK + "pooled drift\n"
    draft = cli(tmp_path, article, "--draft", fill_in=fill_in)
    assert draft.returncode == 0, draft.stdout
    assert "1 placeholder" in draft.stdout and "pooled drift" in draft.stdout
    final = cli(tmp_path, article, fill_in=fill_in)
    assert final.returncode == 1
    assert "placeholder" in final.stdout


def test_a_missing_article_is_a_usage_error_not_a_traceback(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "none.md"), str(tmp_path / "none.yaml")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "none.md" in result.stderr


def test_a_malformed_source_list_is_a_usage_error(tmp_path):
    article_path, sources_path = write_case(tmp_path, make_article())
    sources_path.write_text("- name: broken\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), str(article_path), str(sources_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_the_article_must_be_utf8(tmp_path):
    article_path, sources_path = write_case(tmp_path, make_article())
    article_path.write_bytes(b"# Bad\n\xff\xfe\n")
    result = subprocess.run([sys.executable, str(SCRIPT), str(article_path), str(sources_path)], capture_output=True, text=True)
    assert result.returncode == 2


# --- 10b. fix round 1: a broken source list, comments that do not render, names that start alike ----------------------------


@pytest.mark.parametrize("raw", ["- name: [unclosed\n  license: mit\n", "- name: fix/harm\n\turl: https://example.org\n"], ids=["unclosed-bracket", "tab-indent"])
def test_a_source_list_that_is_not_valid_yaml_is_a_usage_error_not_a_traceback(tmp_path, raw):
    article_path, sources_path = write_case(tmp_path, make_article())
    sources_path.write_text(raw, encoding="utf-8")
    with pytest.raises(check_article.UsageError) as excinfo:
        check_article.check(article_path, sources_path)
    assert "source list" in str(excinfo.value)
    result = subprocess.run([sys.executable, str(SCRIPT), str(article_path), str(sources_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr + result.stdout
    assert "source list" in result.stderr


def comment(text):
    return "<!-- " + text + " -->"


def test_a_whole_section_hidden_in_a_comment_is_missing(tmp_path):
    article = make_article().replace(f"## {TEAMMATES}\n", f"<!--\n## {TEAMMATES}\n") + "\n-->\n"
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert f"'{TEAMMATES}'" in messages(report)


def test_a_disclosure_that_sits_in_a_comment_does_not_count(tmp_path):
    article = make_article(sections={TEAMMATES: "None. " + comment(PIECES["disclosure"])})
    assert "disclosure" in rules(run(tmp_path, article))


def test_a_licence_sentence_that_sits_in_a_comment_does_not_count(tmp_path):
    models = "\n\n".join([comment(PIECES["tiny_licence"]), PIECES["gated"], PIECES["aup"], PIECES["nllb"]])
    report = run(tmp_path, make_article(sections={MODELS: models}))
    assert "licenses-and-tools" in rules(report)
    assert "licence (CC-BY-NC-4.0)" in messages(report)


def test_a_source_row_that_sits_in_a_comment_does_not_count(tmp_path):
    rows = "| fix/benign | cc-by-4.0 | harmless but edgy prompts |\n| fix/eval | cc-by-sa-4.0 | evaluation only: scores are published |"
    datasets = "The public sources are listed here with the licence of each one.\n\n" + rows + "\n\n" + comment("| fix/harm | mit | harmful prompts |") + "\n\n" + PIECES["multijail"]
    report = run(tmp_path, make_article(drop=("harm",), sections={DATASETS: datasets}))
    assert "source-licenses" in rules(report)
    assert "fix/harm" in messages(report)


def test_an_unterminated_comment_hides_the_rest_of_the_article(tmp_path):
    article = make_article().replace(f"## {TEAMMATES}\n", f"<!-- never closed\n## {TEAMMATES}\n")
    assert "headings" in rules(run(tmp_path, article))


@pytest.mark.parametrize("draft", [False, True])
def test_an_unterminated_comment_is_named_with_the_line_it_opens_on(tmp_path, draft):
    article = make_article().replace(f"## {TEAMMATES}\n", f"<!-- never closed\n## {TEAMMATES}\n")
    line = article.splitlines().index("<!-- never closed") + 1
    report = run(tmp_path, article, draft=draft, fill_in=FILL_IN_OK)
    assert f"comment opened at line {line} is never closed" in messages(report)
    assert "headings" in rules(report)  # the hidden heading is still reported
    result = cli(tmp_path, article, *(["--draft"] if draft else []), fill_in=FILL_IN_OK)
    assert result.returncode == 1
    assert f"comment opened at line {line} is never closed" in result.stdout


def test_an_unterminated_comment_at_the_end_of_the_file_is_named(tmp_path):
    article = make_article() + "\n<!-- a note that was never finished\n"
    line = article.splitlines().index("<!-- a note that was never finished") + 1
    report = run(tmp_path, article)
    assert f"comment opened at line {line} is never closed" in messages(report)
    assert not report.ok


def test_closed_comments_are_not_reported(tmp_path):
    article = make_article(extra="\n<!-- one -->\n\n<!--\ntwo\nlines\n-->\n")
    report = run(tmp_path, article)
    assert "never closed" not in messages(report)
    assert report.ok, messages(report)


def test_placeholders_and_unfinished_markers_in_a_comment_are_still_seen(tmp_path):
    article = make_article(extra="\n" + comment("<<RESULT: hidden result>> and TODO later") + "\n")
    draft = run(tmp_path, article, draft=True, fill_in=FILL_IN_OK)
    assert [p.name for p in draft.placeholders] == ["hidden result"]
    assert "fill-in" in rules(draft)  # FILL_IN_OK does not list it
    assert "unfinished-markers" in rules(draft)
    final = run(tmp_path, article)
    assert "placeholders" in rules(final)


def test_line_numbers_stay_right_after_a_multi_line_comment(tmp_path):
    article = make_article(extra="\n<!--\none\ntwo\n-->\n## An extra section\n\nWords that make the section long enough to pass the prose rule easily.\n")
    report = run(tmp_path, article)
    line = article.splitlines().index("## An extra section") + 1
    assert any(f.rule == "headings" and f.line == line for f in report.findings), messages(report)


def test_a_source_name_that_starts_another_entrys_name_needs_its_own_row(tmp_path):
    sources = [
        source_entry("fix/mmlu", "apache-2.0", "task_cross_check"),
        source_entry("fix/mmlu (fine-tune subset)", "apache-2.0", "fine_tune_data"),
    ]
    both = "| fix/mmlu | apache-2.0 | task questions |\n| fix/mmlu (fine-tune subset) | apache-2.0 | training questions |"
    only_subset = "| fix/mmlu (fine-tune subset) | apache-2.0 | training questions |"

    def article(rows):
        return make_article(drop=("harm", "benign", "eval"), sections={DATASETS: "The public sources are listed with the licence of each one.\n\n" + rows + "\n\n" + PIECES["multijail"]})

    assert "source-licenses" not in rules(run(tmp_path, article(both), sources=sources))
    report = run(tmp_path, article(only_subset), sources=sources)
    assert "source-licenses" in rules(report)
    assert "'fix/mmlu' " in messages(report) or "'fix/mmlu'" in messages(report)


# --- 11. the real article ------------------------------------------------------------------------------------------------------------------------


REAL_ARTICLE = ROOT / "article" / "ARTICLE.md"
REAL_SOURCES = ROOT / "data" / "sources.yaml"


def test_the_real_article_passes_the_checker():
    """Draft mode while placeholders remain; the plain run once they are filled in (the state the owner submits)."""
    assert REAL_ARTICLE.is_file(), "article/ARTICLE.md has not been written"
    draft = "<<" in REAL_ARTICLE.read_text(encoding="utf-8")
    report = check_article.check(REAL_ARTICLE, REAL_SOURCES, draft=draft)
    assert report.ok, messages(report)


def test_the_real_draft_is_a_draft_and_lists_its_placeholders_in_fill_in():
    text = REAL_ARTICLE.read_text(encoding="utf-8")
    if "<<" not in text:
        pytest.skip("no placeholders left: the article has been filled in")
    fill_in = (ROOT / "article" / "FILL_IN.md").read_text(encoding="utf-8")
    report = check_article.check(REAL_ARTICLE, REAL_SOURCES, draft=True)
    assert report.placeholders, "a draft without placeholders is not a draft"
    for p in report.placeholders:
        assert p.name in fill_in, p.name
    assert "heatmap.png" in fill_in


def test_the_real_article_names_the_exported_file_names_the_fill_in_relies_on():
    fill_in = (ROOT / "article" / "FILL_IN.md").read_text(encoding="utf-8")
    for name in ("tables.csv", "reliability.json", "heatmap.png", "alpha_curve.png", "MANIFEST.json"):
        assert name in fill_in, name
    assert textwrap.dedent(fill_in).strip()


def real_text():
    return " ".join(REAL_ARTICLE.read_text(encoding="utf-8").split())


def test_the_real_article_asks_the_owner_about_the_gate_simulation_and_says_where_the_figures_come_from():
    report = check_article.check(REAL_ARTICLE, REAL_SOURCES, draft=True)
    names = [p.name for p in report.placeholders if p.kind == "OWNER"]
    assert any("gate simulation was re-run" in n for n in names), names
    text = real_text()
    assert "As noted in the project log (not public)" in text
    assert "comes from the project notes only" in text
    for gone in ("As recorded", "never decides the outcome"):
        assert gone not in text, gone


def test_the_real_article_names_the_models_it_releases_only_through_the_upload_marker():
    text = real_text()
    assert "The released models" not in text
    assert "Which arms those are: <<OWNER: which model arms were uploaded with weights>>" in text
    assert "The repaired model (A2), only if a repair strength qualified" in text
    assert "DARE" in text and "thinned" in text and "on by default" in text


@pytest.mark.parametrize(
    "phrase",
    [
        "A setting fails if any one of these is true:",
        "always answering D would score 31%",
        "always answering the most common letter would score 27 to 28%",
        "The alpha choice compares models on the same questions, so the higher dev figure does not tilt it.",
        "18 pairs of English prompts are in both dev and report.",
        "shares at least 60% of its wording (4 pairs share at least 80%)",
        "AlignTune can train in two ways, called backends: unsloth and TRL (Hugging Face's training library).",
        "This is our reading, not legal advice.",
        "a character-overlap score called chrF of at least 0.35",
    ],
)
def test_the_real_article_uses_the_plain_wording_the_review_asked_for(phrase):
    assert phrase in real_text()


def test_the_real_article_lists_the_five_sanity_checks_as_bullets():
    lines = REAL_ARTICLE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if "A setting fails if any one of these is true:" in line)
    bullets = []
    for line in lines[start + 1 :]:
        if not line.startswith("  - "):
            break
        bullets.append(line.strip())
    assert len(bullets) == 5, bullets
