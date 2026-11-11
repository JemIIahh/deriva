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

HEADINGS = [
    "Problem and approach",
    "Repos used and for what",
    "Models and licenses",
    "Datasets and licenses",
    "Model collection",
    "Data collection",
    "Experiments and what else was tried",
    "Approach with before/after results",
    "Teammates",
]

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
        "Problem and approach": keep.get("question", ""),
        "Repos used and for what": "\n\n".join(
            keep[name] for name in ("repos", "aligntune_mit", "curatorkit_lsal", "aligntune_depends", "unsloth", "mergekit", "trl")
            if name in keep
        ),
        "Models and licenses": "\n\n".join(keep[name] for name in ("tiny_licence", "gated", "aup", "nllb") if name in keep),
        "Datasets and licenses": (
            "These are the public sources the study uses, with the licence of each one.\n\n"
            "| Source | Licence | Use |\n| --- | --- | --- |\n"
            + "\n".join(row for name, row in SOURCE_ROWS.items() if name not in drop)
            + ("\n\n" + keep["multijail"] if "multijail" in keep else "")
        ),
        "Model collection": PLAIN_COLLECTION,
        "Data collection": PLAIN_DATA,
        "Experiments and what else was tried": PLAIN_EXPERIMENTS,
        "Approach with before/after results": PLAIN_APPROACH,
        "Teammates": "None. " + keep.get("disclosure", "Nobody else worked on it, and nobody else helped in any way at all."),
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


@pytest.mark.parametrize("missing", HEADINGS)
def test_a_missing_required_heading_is_named(tmp_path, missing):
    article = make_article().replace(f"## {missing}\n", f"### {missing}\n")
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert f"'{missing}'" in messages(report)
    assert "missing" in messages(report)


def test_headings_in_the_wrong_order_fail(tmp_path):
    article = make_article()
    article = article.replace("## Model collection", "## @@").replace("## Data collection", "## Model collection").replace("## @@", "## Data collection")
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert "order" in messages(report)
    assert "Data collection" in messages(report)


def test_a_duplicated_heading_fails_and_names_both_lines(tmp_path):
    article = make_article(extra="\n## Teammates\n\nNone again, in plain words, so that the section has some prose to read.\n")
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert "'Teammates'" in messages(report)
    assert "twice" in messages(report)


def test_an_extra_level_two_heading_fails(tmp_path):
    report = run(tmp_path, make_article(extra="\n## Acknowledgements\n\nThanks to everyone who read the draft and told us what was unclear.\n"))
    assert "headings" in rules(report)
    assert "'Acknowledgements'" in messages(report)
    assert "extra" in messages(report).lower()


def test_level_three_subsections_are_allowed(tmp_path):
    article = make_article(sections={"Experiments and what else was tried": PLAIN_EXPERIMENTS + "\n\n### What failed\n\nNothing has been run yet, so this part stays short and plain.\n"})
    assert run(tmp_path, article).ok


def test_a_heading_inside_a_code_block_is_not_a_heading(tmp_path):
    article = make_article(extra="\n```\n## Not a real section\n```\n")
    assert run(tmp_path, article).ok


def test_heading_text_must_match_exactly(tmp_path):
    article = make_article(heading_overrides={"Teammates": "Teammates tagged"})
    report = run(tmp_path, article)
    assert "headings" in rules(report)
    assert "'Teammates'" in messages(report)


# --- 2. heading levels ---------------------------------------------------------------------------------------------------


def test_a_skipped_heading_level_fails(tmp_path):
    article = make_article(sections={"Experiments and what else was tried": PLAIN_EXPERIMENTS + "\n\n#### Too deep\n\nThis heading skips level three, which breaks the outline of the page.\n"})
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
    article = make_article(drop=("harm",)).replace("## Datasets and licenses", "## Datasets and licenses", 1)
    article = article.replace(
        "Cohere Labs also applies", "fix/harm is also listed here under mit. Cohere Labs also applies"
    )
    assert run(tmp_path, article).ok


def test_a_source_named_only_in_another_section_does_not_count(tmp_path):
    article = make_article(drop=("harm",)).replace("## Teammates\n\n", "## Teammates\n\nfix/harm mit is mentioned here only.\n\n")
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
    article = make_article(sections={"Teammates": "None. The work was done with Claude Code and a lot of coffee, mostly at night."})
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
    article = make_article(drop=("tiny_licence",)).replace("## Teammates\n\n", "## Teammates\n\n" + PIECES["tiny_licence"] + "\n\n")
    report = run(tmp_path, article)
    assert "licenses-and-tools" in rules(report)


# --- 6. plain language ----------------------------------------------------------------------------------------------------------


def test_a_section_of_only_links_fails(tmp_path):
    article = make_article(sections={"Model collection": "https://huggingface.co/collections/example/models\n\n[the data](https://huggingface.co/collections/example/data)"})
    report = run(tmp_path, article)
    assert "empty-or-links-only" in rules(report)
    assert "'Model collection'" in messages(report)
    assert "links" in messages(report)


def test_an_empty_section_fails(tmp_path):
    report = run(tmp_path, make_article(sections={"Data collection": ""}))
    assert "empty-or-links-only" in rules(report)
    assert "'Data collection'" in messages(report)


def test_a_figure_without_alt_text_fails(tmp_path):
    article = make_article(sections={"Approach with before/after results": PLAIN_APPROACH.replace("![Heatmap of attack success rate per language, before and after the repair]", "![]")})
    report = run(tmp_path, article)
    assert "figures" in rules(report)
    assert "alt text" in messages(report)


def test_a_figure_with_one_word_of_alt_text_fails(tmp_path):
    article = make_article(sections={"Approach with before/after results": PLAIN_APPROACH.replace("![Heatmap of attack success rate per language, before and after the repair]", "![heatmap]")})
    assert "figures" in rules(run(tmp_path, article))


def test_the_per_language_heatmap_must_be_referenced(tmp_path):
    article = make_article(sections={"Approach with before/after results": PLAIN_APPROACH.split("\n\n")[0]})
    report = run(tmp_path, article)
    assert "figures" in rules(report)
    assert "heatmap" in messages(report)


def test_the_heatmap_may_sit_in_a_subfolder_or_at_a_url(tmp_path):
    for path in ("figures/heatmap.png", "https://example.org/files/heatmap.png"):
        article = make_article(sections={"Approach with before/after results": PLAIN_APPROACH.replace("(heatmap.png)", f"({path})")})
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
    return make_article(sections={"Approach with before/after results": PLAIN_APPROACH + "\n\n" + extra_sentence})


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
        report = run(tmp_path, draft_article(sentence), draft=True, fill_in=FILL_IN_OK)
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
