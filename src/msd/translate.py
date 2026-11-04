"""Translate English prompts into a study language with a back-translation quality check."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from msd.config import NON_ENGLISH
from msd.schema import Prompt

# (texts, source language, target language) -> one translation per text, in order.
Translator = Callable[[list[str], str, str], list[str]]


@dataclass(frozen=True)
class QCReport:
    """Per-language counts of items kept and dropped by the back-translation check."""

    kept: dict[str, int]
    dropped: dict[str, int]


def translated_id(source_id: str, lang: str) -> str:
    """Stable id for the `lang` translation of the item `source_id`.

    The pair is JSON-encoded before hashing (the project's cache-key convention) so no
    choice of id or language can shift the boundary between them; a plain "id|lang" join
    would make ("x|hi", "sw") and ("x", "hi|sw") collide. Deterministic across runs and
    processes: sha1 over UTF-8 bytes, no hash(), no randomness.
    """
    encoded = json.dumps([source_id, lang], ensure_ascii=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _translate_checked(
    translator: Translator,
    texts: list[str],
    src: str,
    tgt: str,
    ids: list[str],
    step: str,
) -> list[str]:
    """Call `translator` and refuse any output that is misaligned, non-text or blank.

    A non-list result or a wrong count raises (there is no way to tell which input an output
    belongs to), and so does a blank or non-string translation (a translator failure must not
    be quietly folded into the meaning-drop count). The error names the step, and the item
    where one is at fault.
    """
    out = translator(list(texts), src, tgt)
    if not isinstance(out, list):
        # a bare string (or tuple, or None) must not be indexed as if it were one text per input
        raise ValueError(
            f"{step} translator ({src}->{tgt}) must return a list of texts, got {type(out).__name__}"
        )
    if len(out) != len(texts):
        raise ValueError(
            f"{step} translator ({src}->{tgt}) returned {len(out)} texts for {len(texts)} inputs"
        )
    for item_id, text in zip(ids, out):
        if not isinstance(text, str):
            raise ValueError(
                f"{step} translator ({src}->{tgt}) returned {type(text).__name__} for item {item_id!r}"
            )
        if not text.strip():
            raise ValueError(
                f"{step} translator ({src}->{tgt}) returned an empty translation for item {item_id!r}"
            )
    return list(out)


def translate_with_qc(
    items_en: Sequence[Prompt],
    lang: str,
    tr_a: Translator,
    tr_b: Translator,
    same_meaning: Callable[[str, str], bool],
) -> tuple[list[Prompt], QCReport]:
    """Translate English items into `lang` with `tr_a`, back-translate with `tr_b`, and keep
    only the items whose back-translation `same_meaning(original, back)` accepts.

    Each kept item is the English item with only `id` (see `translated_id`), `lang` and
    `text` replaced; `kind`, `split`, `source`, `license` and `gold` carry through, so a
    translation inherits the split its English item was given before translation.

    Raises ValueError for a target that is not one of the non-English study languages
    (English included), for an input item that is not English, repeats an id or has blank text
    (the message names the item), for a translator that returns a non-list, the wrong number of
    texts or a blank/non-string translation, and for a `same_meaning` that does not return a
    real bool. Items are dropped only when `same_meaning` says False. Output order follows the
    input.
    """
    if lang not in NON_ENGLISH:
        raise ValueError(f"target language must be one of {list(NON_ENGLISH)}, got {lang!r}")
    ids = [item.id for item in items_en]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate item ids in input: {dupes[:5]}")
    not_english = [item.id for item in items_en if item.lang != "en"]
    if not_english:
        raise ValueError(f"translate_with_qc takes English items only; not 'en': {not_english[:5]}")
    for item in items_en:
        if not isinstance(item.text, str) or not item.text.strip():
            # the fault is in the input, not in a translator
            raise ValueError(f"input item {item.id!r} has blank text; fix or drop it before translating")

    if not items_en:
        return [], QCReport(kept={lang: 0}, dropped={lang: 0})

    texts = [item.text for item in items_en]
    forward = _translate_checked(tr_a, texts, "en", lang, ids, "forward")
    back = _translate_checked(tr_b, forward, lang, "en", ids, "back")

    kept: list[Prompt] = []
    for item, translated, back_text in zip(items_en, forward, back):
        verdict = same_meaning(item.text, back_text)
        if not isinstance(verdict, bool):
            raise ValueError(
                f"same_meaning must return a bool, got {type(verdict).__name__} "
                f"for item {item.id!r}"
            )
        if verdict:
            kept.append(replace(item, id=translated_id(item.id, lang), lang=lang, text=translated))
    report = QCReport(kept={lang: len(kept)}, dropped={lang: len(items_en) - len(kept)})
    return kept, report
