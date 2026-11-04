import pytest

from msd.config import ALPHAS, LANGS
from msd.report import _cell_text, alpha_curve, cell_style, heatmap

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


def test_heatmap_draws_solid_hatched_and_low_confidence_differently(tmp_path):
    rows = _heatmap_rows()
    clear = {lang: (0.2, 0.1, 0.3) for lang in LANGS}  # every interval excludes 0
    noisy = {lang: (0.0, -0.1, 0.1) for lang in LANGS}  # every interval spans 0

    def render(name, deltas, low_conf):
        out = tmp_path / name
        heatmap(rows, metric="asr", arm_before="A0", arm_after="A1", low_conf=low_conf,
                out_path=out, deltas=deltas)
        return out.read_bytes()

    solid = render("solid.png", clear, set())
    hatched = render("hatched.png", noisy, set())
    low = render("low.png", clear, set(LANGS))
    assert len({solid, hatched, low}) == 3


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
    rows = [r for r in rows if not (r["alpha"] == 0.75 and r["metric"] == "task_acc")]
    out = tmp_path / "curve.png"
    alpha_curve(rows, langs=["en"], out_path=out)
    _assert_png(out)


def test_alpha_curve_with_nothing_to_plot_raises(tmp_path):
    with pytest.raises(ValueError, match="A2"):
        alpha_curve(_curve_rows(langs=("en",)), langs=["sw"], out_path=tmp_path / "c.png")
