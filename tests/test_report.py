import itertools
import math

import pytest
from matplotlib.text import Text

from msd.config import ALPHAS, LANGS
from msd.report import (
    DELTA_TOLERANCE,
    _build_alpha_curve_figure,
    _build_heatmap_figure,
    _cell_text,
    alpha_curve,
    cell_style,
    heatmap,
)

MIN_PNG_BYTES = 5 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _row(arm, lang, metric, rate, n, *, alpha=None, lo=None, hi=None, excluded=0) -> dict:
    return {
        "arm": arm,
        "alpha": alpha,
        "lang": lang,
        "metric": metric,
        "rate": rate,
        "lo": lo,
        "hi": hi,
        "n": n,
        "excluded": excluded,
    }


def _heatmap_rows() -> list[dict]:
    """Five languages, A0 vs A1 on asr; `sw` has too few items for an interval."""
    rates = {"en": (0.04, 0.05), "hi": (0.10, 0.31), "hinglish": (0.12, 0.35),
             "ar": (0.08, 0.30), "sw": (0.20, 0.45)}
    rows = []
    for lang, (before, after) in rates.items():
        n = 6 if lang == "sw" else 120
        for arm, rate in (("A0", before), ("A1", after)):
            lo, hi = (None, None) if n < 10 else (max(0.0, rate - 0.05), rate + 0.05)
            rows.append(_row(arm, lang, "asr", rate, n, lo=lo, hi=hi))
    return rows


def _assert_png(path) -> None:
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC)
    assert len(data) > MIN_PNG_BYTES


# ------------------------------------------------------------------ cell_style


def test_cell_style_solid_when_ci_excludes_zero():
    assert cell_style((0.2, 0.1, 0.3), "hi", set()) == "solid"
    assert cell_style((-0.2, -0.3, -0.1), "hi", set()) == "solid"


def test_cell_style_hatched_when_ci_spans_zero():
    assert cell_style((0.02, -0.05, 0.09), "hi", set()) == "hatched"
    assert cell_style((0.05, 0.0, 0.1), "hi", set()) == "hatched"  # touching zero is inside
    assert cell_style(None, "hi", set()) == "hatched"  # no interval: cannot claim a difference


def test_cell_style_low_confidence_wins():
    assert cell_style((0.2, 0.1, 0.3), "ar", {"ar"}) == "low_confidence"
    assert cell_style((0.02, -0.05, 0.09), "ar", {"ar"}) == "low_confidence"
    assert cell_style(None, "ar", {"ar"}) == "low_confidence"
    assert cell_style((0.2, 0.1, 0.3), "hi", {"ar"}) == "solid"


# ------------------------------------------------------------------- cell text


def test_cell_text_shows_percentage_interval_and_n():
    text = _cell_text(_row("A0", "en", "asr", 0.375, 120, lo=0.3, hi=0.45))
    assert "37.5%" in text
    assert "30.0" in text and "45.0" in text
    assert "n=120" in text


def test_cell_text_small_n_is_flagged_and_has_no_interval():
    text = _cell_text(_row("A0", "sw", "asr", 0.5, 6))
    assert "50.0%" in text
    assert "n<10" in text
    assert "[" not in text
    # even if an interval were somehow supplied, a cell this small must not print it
    assert "[" not in _cell_text(_row("A0", "sw", "asr", 0.5, 6, lo=0.1, hi=0.9))


def test_cell_text_no_data():
    assert _cell_text(_row("A0", "sw", "asr", None, 0)) == "no data"
    assert _cell_text(None) == "no data"


# --------------------------------------------------------------------- heatmap


def test_heatmap_writes_png_over_5kb(tmp_path):
    out = tmp_path / "figs" / "heat.png"
    deltas = {
        "en": (0.01, -0.03, 0.05),  # inside the noise: hatched
        "hi": (0.21, 0.12, 0.30),  # solid
        "hinglish": (0.23, 0.14, 0.32),
        "ar": (0.22, 0.13, 0.31),  # low confidence wins over the solid interval
        # "sw" has no paired interval: hatched
    }
    heatmap(
        _heatmap_rows(),
        metric="asr",
        arm_before="A0",
        arm_after="A1",
        low_conf={"ar"},
        out_path=out,
        deltas=deltas,
    )
    _assert_png(out)


def _build(rows=None, *, low_conf=(), deltas=None):
    return _build_heatmap_figure(
        _heatmap_rows() if rows is None else rows, metric="asr", arm_before="A0", arm_after="A1",
        low_conf=set(low_conf), deltas=deltas,
    )


def _cells(fig) -> dict:
    """Heatmap cell rectangles by (lang, arm), read off the figure rather than the PNG."""
    cells = {}
    for patch in fig.axes[0].patches:
        gid = patch.get_gid() or ""
        if gid.startswith("cell:"):
            _, lang, arm = gid.split(":")
            cells[(lang, arm)] = patch
    return cells


def _notes(fig) -> dict[str, str]:
    return {t.get_gid()[5:]: t.get_text() for t in fig.axes[0].texts if (t.get_gid() or "").startswith("note:")}


def _legend(fig) -> dict[str, str]:
    """Legend text -> hatch of its swatch."""
    (legend,) = fig.legends
    return {t.get_text(): h.get_hatch() or "" for t, h in zip(legend.get_texts(), legend.legend_handles)}


# Deltas that match the displayed cells of _heatmap_rows() (after - before).
_CLEAR = {"hi": (0.21, 0.12, 0.30), "hinglish": (0.23, 0.14, 0.32), "ar": (0.22, 0.13, 0.31)}


def test_heatmap_draws_solid_hatched_and_low_confidence_rows_on_the_right_rows():
    deltas = {**_CLEAR, "en": (0.01, -0.03, 0.05)}  # en is inside the noise; sw has no interval
    fig = _build(low_conf={"ar"}, deltas=deltas)
    cells = _cells(fig)
    hatch = {lang: {cells[(lang, arm)].get_hatch() or "" for arm in ("A0", "A1")} for lang in LANGS}

    assert hatch["hi"] == {""} and hatch["hinglish"] == {""}  # solid: no pattern on either cell
    (hatched,) = hatch["en"]  # hatched: one and the same pattern on both cells
    assert hatch["en"] == hatch["sw"] == {hatched} and hatched != ""
    (dotted,) = hatch["ar"]  # low confidence wins over the clear interval
    assert dotted != "" and dotted != hatched
    # Rows keep the language order and both cells of a row share its height.
    for row, lang in enumerate(LANGS):
        assert [cells[(lang, arm)].get_y() for arm in ("A0", "A1")] == pytest.approx([row + 0.04] * 2)


def test_heatmap_legend_swatches_carry_the_same_patterns_as_the_cells():
    fig = _build(low_conf={"ar"}, deltas={**_CLEAR, "en": (0.01, -0.03, 0.05)})
    cells = _cells(fig)
    legend = _legend(fig)
    by_prefix = {text.split(":")[0]: pattern for text, pattern in legend.items()}
    assert set(by_prefix) == {"solid row", "hatched row", "dotted row"}
    assert by_prefix["solid row"] == (cells[("hi", "A0")].get_hatch() or "") == ""
    assert by_prefix["hatched row"] == cells[("en", "A0")].get_hatch() != ""
    assert by_prefix["dotted row"] == cells[("ar", "A0")].get_hatch() != ""
    assert by_prefix["hatched row"] != by_prefix["dotted row"]


def test_heatmap_legend_says_the_marking_applies_to_the_whole_row():
    assert list(_legend(_build())) == [
        "solid row: change is clear (CI excludes 0)",
        "hatched row: change between the two columns is inside the noise (CI includes 0) "
        "or could not be measured",
        "dotted row: the two judges disagree too much, or were not double-checked; read with care",
    ]


def test_heatmap_nan_or_swapped_interval_is_never_solid():
    nan = math.nan
    assert cell_style((0.2, nan, nan), "hi", set()) == "hatched"
    assert cell_style((0.2, 0.1, nan), "hi", set()) == "hatched"
    assert cell_style((0.2, nan, 0.3), "hi", set()) == "hatched"
    assert cell_style((0.2, 0.3, 0.1), "hi", set()) == "hatched"  # lo above hi
    assert cell_style((-0.2, -0.1, -0.3), "hi", set()) == "hatched"
    assert cell_style((0.2, 0.1, math.inf), "hi", set()) == "hatched"
    assert cell_style((0.2, 0.1, 0.3), "hi", set()) == "solid"  # and the ordinary case stands
    assert cell_style((0.2, nan, nan), "ar", {"ar"}) == "low_confidence"  # low confidence still wins


def test_heatmap_unmeasurable_row_is_hatched_and_shows_no_interval_even_with_a_clear_delta():
    # sw has n=6 in both cells; a caller supplies a clear delta for it anyway.
    deltas = {**_CLEAR, "sw": (0.25, 0.10, 0.40)}
    fig = _build(deltas=deltas)
    cells, notes = _cells(fig), _notes(fig)
    assert cells[("sw", "A0")].get_hatch() and cells[("sw", "A1")].get_hatch()
    assert "no paired" in notes["sw"] and "[" not in notes["sw"] and "+25.0" not in notes["sw"]
    assert not cells[("hi", "A0")].get_hatch()  # the measurable rows are untouched
    assert "+21.0 pp" in notes["hi"]


def test_heatmap_row_with_one_empty_cell_is_hatched_and_shows_no_interval():
    rows = [r for r in _heatmap_rows() if r["lang"] == "hi" and r["arm"] == "A0"]
    rows.append(_row("A1", "hi", "asr", None, 0))  # no data after
    fig = _build(rows, deltas={"hi": (0.21, 0.12, 0.30)})
    cells, notes = _cells(fig), _notes(fig)
    assert cells[("hi", "A0")].get_hatch()  # the measured cell is hatched, the empty one stays plain grey
    assert "no paired" in notes["hi"]


def test_heatmap_low_confidence_wins_over_forced_hatching():
    fig = _build(low_conf={"sw"})
    cells = _cells(fig)
    assert cells[("sw", "A0")].get_hatch() == cells[("sw", "A1")].get_hatch()
    assert cells[("sw", "A0")].get_hatch() != _cells(_build())[("sw", "A0")].get_hatch()


def test_heatmap_rejects_a_delta_that_does_not_match_the_displayed_cells(tmp_path):
    out = tmp_path / "heat.png"
    # hi is shown as 0.10 -> 0.31 (+0.21); a delta of -0.30 belongs to some other pair of arms.
    with pytest.raises(ValueError) as err:
        heatmap(_heatmap_rows(), metric="asr", arm_before="A0", arm_after="A1", low_conf=set(),
                out_path=out, deltas={"hi": (-0.30, -0.40, -0.20)})
    message = str(err.value)
    assert "hi" in message and "-0.30" in message and "0.21" in message
    assert not out.exists()  # refused before anything was written


def test_heatmap_accepts_a_delta_within_the_tolerance():
    assert DELTA_TOLERANCE == pytest.approx(0.05)
    fig = _build(deltas={"hi": (0.21 + 0.04, 0.15, 0.35)})  # 4 pp off the displayed +21 pp
    assert not _cells(fig)[("hi", "A0")].get_hatch()
    with pytest.raises(ValueError, match="hi"):
        _build(deltas={"hi": (0.21 + 0.06, 0.15, 0.35)})  # 6 pp off


def test_heatmap_rejects_a_duplicate_row():
    rows = _heatmap_rows()
    rows.append(dict(rows[0]))  # a second A0 / en / asr row
    with pytest.raises(ValueError, match="duplicate"):
        _build(rows)


def test_heatmap_without_deltas_still_renders(tmp_path):
    out = tmp_path / "heat.png"
    heatmap(_heatmap_rows(), metric="asr", arm_before="A0", arm_after="A1",
            low_conf=set(), out_path=out)
    _assert_png(out)


def test_heatmap_tolerates_missing_languages_and_empty_cells(tmp_path):
    rows = [r for r in _heatmap_rows() if r["lang"] in ("en", "ar")]
    rows.append(_row("A1", "hi", "asr", None, 0, excluded=40))  # only the "after" cell
    out = tmp_path / "heat.png"
    heatmap(rows, metric="asr", arm_before="A0", arm_after="A1", low_conf=set(), out_path=out)
    _assert_png(out)


def test_heatmap_rejects_several_alphas_for_one_arm(tmp_path):
    rows = [
        _row("A0", lang, "asr", 0.1, 100, lo=0.05, hi=0.15) for lang in ("en", "hi")
    ] + [
        _row("A2", lang, "asr", 0.2, 100, alpha=alpha, lo=0.1, hi=0.3)
        for lang in ("en", "hi")
        for alpha in (0.5, 1.0)
    ]
    with pytest.raises(ValueError, match="alpha"):
        heatmap(rows, metric="asr", arm_before="A0", arm_after="A2",
                low_conf=set(), out_path=tmp_path / "heat.png")
    assert not (tmp_path / "heat.png").exists()
    # Filtered to one alpha per arm, the same rows are fine.
    one = [r for r in rows if r["alpha"] in (None, 0.5)]
    heatmap(one, metric="asr", arm_before="A0", arm_after="A2",
            low_conf=set(), out_path=tmp_path / "heat.png")
    _assert_png(tmp_path / "heat.png")


def test_heatmap_with_no_matching_rows_raises(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        heatmap(_heatmap_rows(), metric="task_acc", arm_before="A0", arm_after="A1",
                low_conf=set(), out_path=tmp_path / "heat.png")


# ----------------------------------------------------------------- alpha_curve


def _curve_rows(langs=LANGS) -> list[dict]:
    rows = []
    for k, lang in enumerate(langs):
        base_asr = 0.10 + 0.03 * k
        for arm, asr, acc in (("A0", base_asr, 0.60), ("A1", base_asr + 0.20, 0.62)):
            rows.append(_row(arm, lang, "asr", asr, 100, lo=asr - 0.05, hi=asr + 0.05))
            rows.append(_row(arm, lang, "task_acc", acc, 100, lo=acc - 0.04, hi=acc + 0.04))
        for alpha in ALPHAS:
            asr = base_asr + 0.20 * (1 - alpha)
            acc = 0.60 + 0.02 * alpha
            rows.append(_row("A2", lang, "asr", asr, 100, alpha=alpha, lo=asr - 0.05, hi=asr + 0.05))
            rows.append(_row("A2", lang, "task_acc", acc, 100, alpha=alpha, lo=acc - 0.04, hi=acc + 0.04))
    return rows


def test_alpha_curve_writes_png_over_5kb(tmp_path):
    out = tmp_path / "figs" / "curve.png"
    alpha_curve(_curve_rows(), langs=list(LANGS), out_path=out)
    _assert_png(out)


def test_alpha_curve_skips_a_language_without_a2_rows(tmp_path):
    rows = _curve_rows(langs=("en", "hi"))
    out = tmp_path / "curve.png"
    alpha_curve(rows, langs=["en", "hi", "sw"], out_path=out)  # "sw" has no rows at all
    _assert_png(out)


def test_alpha_curve_copes_with_missing_intervals_and_partial_points(tmp_path):
    rows = _curve_rows(langs=("en",))
    for r in rows:
        if r["alpha"] == 0.25:
            r["lo"] = r["hi"] = None
    before = len(rows)
    rows = [r for r in rows if not (r["alpha"] == 0.35 and r["metric"] == "task_acc")]
    assert len(rows) == before - 1, "the filter must really drop one task_acc point (the alpha must be on the grid)"
    out = tmp_path / "curve.png"
    alpha_curve(rows, langs=["en"], out_path=out)
    _assert_png(out)


def test_alpha_curve_with_nothing_to_plot_raises(tmp_path):
    with pytest.raises(ValueError, match="A2"):
        alpha_curve(_curve_rows(langs=("en",)), langs=["sw"], out_path=tmp_path / "c.png")


def test_alpha_curve_rejects_a_duplicate_row(tmp_path):
    rows = _curve_rows(langs=("en",))
    dup = next(r for r in rows if r["arm"] == "A2" and r["alpha"] == 0.5 and r["metric"] == "asr")
    rows.append({**dup, "rate": 0.9})  # a second row for the same (arm, alpha, lang, metric)
    with pytest.raises(ValueError, match="duplicate"):
        alpha_curve(rows, langs=["en"], out_path=tmp_path / "c.png")
    assert not (tmp_path / "c.png").exists()


def _line_colours(fig) -> dict[str, str]:
    return {line.get_gid()[5:]: line.get_color() for line in fig.axes[0].lines
            if (line.get_gid() or "").startswith("line:")}


def test_alpha_curve_colours_follow_the_language_not_its_position_among_those_drawn():
    full = _line_colours(_build_alpha_curve_figure(_curve_rows(), langs=list(LANGS)))
    assert len(set(full.values())) == len(LANGS)
    without_hi = [r for r in _curve_rows() if r["lang"] != "hi"]
    partial = _line_colours(_build_alpha_curve_figure(without_hi, langs=list(LANGS)))
    assert set(partial) == set(LANGS) - {"hi"}
    assert partial == {lang: colour for lang, colour in full.items() if lang != "hi"}


_ATTACHED_PX = 12  # a label further than this from its point needs a leader line


def _labels(fig) -> dict[str, object]:
    return {t.get_gid(): t for t in fig.axes[0].texts if (t.get_gid() or "").startswith("label:")}


def _text_box(fig, label):
    """The label's own text box, not the larger one Annotation reports with its leader line."""
    renderer = fig.canvas.get_renderer()
    label.update_positions(renderer)
    return Text.get_window_extent(label, renderer)


def _gap_px(fig, label) -> float:
    """Pixels from the label's point to the nearest edge of the label's box (0 if inside)."""
    box = _text_box(fig, label)
    x, y = fig.axes[0].transData.transform(label.xy)
    dx = max(box.x0 - x, 0.0, x - box.x1)
    dy = max(box.y0 - y, 0.0, y - box.y1)
    return math.hypot(dx, dy)


def test_alpha_curve_standard_labels_do_not_overlap_and_stay_attached_to_their_points():
    fig = _build_alpha_curve_figure(_curve_rows(), langs=list(LANGS))
    labels = _labels(fig)
    # Every language's two ends are always named; any other alpha that is labelled must obey too.
    assert {f"label:{lang}:{a}" for lang in LANGS for a in ("0", "1")} <= set(labels)
    boxes = {gid: _text_box(fig, t) for gid, t in labels.items()}
    for (g1, b1), (g2, b2) in itertools.combinations(boxes.items(), 2):
        assert not b1.overlaps(b2), f"labels {g1} and {g2} overlap"
    for gid, label in labels.items():
        if _gap_px(fig, label) > _ATTACHED_PX:
            assert label.arrow_patch is not None, f"{gid} floats {_gap_px(fig, label):.0f}px from its point"
    # Leaving some alphas unlabelled must be said in the figure, in words.
    if len(labels) < len(LANGS) * len(ALPHAS):
        said = " ".join(t.get_text() for t in fig.texts)
        assert "alphas in between" in said
    # No label sits on top of a data point (its own or a neighbour's).
    to_px = fig.axes[0].transData.transform
    for line in (l for l in fig.axes[0].lines if (l.get_gid() or "").startswith("line:")):
        for point in to_px(line.get_xydata()):
            for gid, box in boxes.items():
                assert not box.contains(*point), f"label {gid} covers a data point"


# ----------------------------------------------- excluded cells and the chosen alpha (Task 17)


def _patch_of(fig, lang, arm):
    return next(p for p in fig.axes[0].patches if p.get_gid() == f"cell:{lang}:{arm}")


def _heatmap_without(lang, arm):
    return [r for r in _heatmap_rows() if not (r["lang"] == lang and r["arm"] == arm)]


def test_heatmap_draws_an_excluded_cell_crossed_with_its_reason():
    fig = _build_heatmap_figure(
        _heatmap_without("hi", "A1"), metric="asr", arm_before="A0", arm_after="A1", low_conf=set(),
        excluded={("hi", "A1"): "judge errors"},
    )
    assert _patch_of(fig, "hi", "A1").get_hatch() == "xxxx"
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "excluded" in texts and "judge errors" in texts
    assert "no data" not in texts  # it is not simply missing
    said = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "crossed" in said


def test_heatmap_without_excluded_cells_has_no_crossed_key():
    fig = _build()
    said = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "crossed" not in said


def test_heatmap_refuses_an_excluded_cell_that_also_has_a_row():
    with pytest.raises(ValueError, match="excluded"):
        _build_heatmap_figure(
            _heatmap_rows(), metric="asr", arm_before="A0", arm_after="A1", low_conf=set(),
            excluded={("hi", "A1"): "judge errors"},
        )


def test_heatmap_ignores_a_delta_for_a_row_with_an_excluded_cell(tmp_path):
    out = tmp_path / "heat.png"
    heatmap(
        _heatmap_without("hi", "A1"), metric="asr", arm_before="A0", arm_after="A1", low_conf=set(),
        out_path=out, deltas={"hi": (0.2, 0.1, 0.3)}, excluded={("hi", "A1"): "sanity check"},
    )
    _assert_png(out)


def test_alpha_curve_rings_the_chosen_alpha_and_lists_the_left_out_cells():
    fig = _build_alpha_curve_figure(
        _curve_rows(), langs=list(LANGS), chosen_alpha=0.25, excluded={"hi": [0.5], "sw": [0.5, 1.0]},
    )
    rings = [line.get_gid() for line in fig.axes[0].lines if (line.get_gid() or "").startswith("chosen:")]
    assert sorted(rings) == sorted(f"chosen:{lang}" for lang in LANGS)
    said = " ".join(t.get_text() for t in fig.texts)
    assert "Left out" in said and "hi: alpha 0.5" in said and "sw: alpha 0.5, 1" in said
    assert "chosen on dev" in " ".join(t.get_text() for t in fig.legends[0].get_texts())


def test_alpha_curve_without_extras_is_as_before():
    fig = _build_alpha_curve_figure(_curve_rows(), langs=list(LANGS))
    assert not [line for line in fig.axes[0].lines if (line.get_gid() or "").startswith("chosen:")]
    assert "Left out" not in " ".join(t.get_text() for t in fig.texts)
    assert "chosen on dev" not in " ".join(t.get_text() for t in fig.legends[0].get_texts())


def test_alpha_curve_chosen_alpha_missing_for_a_language_rings_only_the_others():
    rows = [r for r in _curve_rows() if not (r["lang"] == "hi" and r["arm"] == "A2" and r["alpha"] == 0.25)]
    fig = _build_alpha_curve_figure(rows, langs=list(LANGS), chosen_alpha=0.25)
    rings = {line.get_gid() for line in fig.axes[0].lines if (line.get_gid() or "").startswith("chosen:")}
    assert rings == {f"chosen:{lang}" for lang in LANGS if lang != "hi"}


def test_alpha_curve_legend_stays_inside_the_figure_with_the_chosen_alpha_named():
    # the "ring" entry once ran off the right edge, cutting off the alpha it names
    fig = _build_alpha_curve_figure(_curve_rows(), langs=list(LANGS), chosen_alpha=0.25,
                                    excluded={"hi": [0.5]})
    fig.canvas.draw()
    legend = fig.legends[0]
    box, page = legend.get_window_extent(), fig.bbox
    assert box.x0 >= page.x0 and box.x1 <= page.x1 and box.y0 >= page.y0 and box.y1 <= page.y1
    assert "0.25" in " ".join(t.get_text() for t in legend.get_texts())
