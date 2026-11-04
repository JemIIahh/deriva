"""Report figures: the before/after heatmap and the alpha trade-off curve.

Both take `aggregate.build_table` rows. Figures are drawn straight onto an Agg
canvas (no pyplot, so no global state), use a colour-blind-safe palette, and carry
only ASCII text: language codes, never Arabic or Devanagari glyphs, so no font is
needed beyond matplotlib's own.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.patheffects import withStroke

from msd.config import LANGS
from msd.stats import MIN_N

DPI = 150

# Heatmap cell colour: cividis is perceptually uniform and readable with every common
# colour vision deficiency. Rates sit on a fixed 0-100% scale so figures compare.
_CMAP = colormaps["cividis"]
_NORM = Normalize(vmin=0.0, vmax=1.0)
_NO_DATA_FILL = "#e3e3e3"
_GREY_TEXT = "#333333"

# Cell pattern per style. Diagonal lines and dots look nothing alike, even in greyscale.
_HATCH = {"solid": "", "hatched": "////", "low_confidence": "...."}

# Okabe-Ito: the standard colour-blind-safe qualitative palette (black and yellow left out).
_LANG_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")

_METRIC_TITLES = {
    "asr": "Attack success rate (lower is safer)",
    "over_refusal": "Over-refusal rate (lower is better)",
    "task_acc": "Task accuracy (higher is better)",
}

Triple = tuple[float, float | None, float | None]  # rate, lo, hi


def cell_style(delta_ci: tuple | None, lang: str, low_conf: set[str]) -> str:
    """How a language's before/after comparison is drawn.

    "low_confidence" if the judge is unreliable for `lang` (wins over everything);
    "hatched" if the change is inside the noise, i.e. its interval (point, lo, hi)
    contains 0, or there is no interval to judge it by; otherwise "solid".
    """
    if lang in low_conf:
        return "low_confidence"
    if delta_ci is None:
        return "hatched"
    _, lo, hi = delta_ci
    return "hatched" if lo <= 0.0 <= hi else "solid"


def _cell_text(row: Mapping | None) -> str:
    """Rate as a percentage, its interval, and n; "n<10" instead of an interval when tiny."""
    if row is None or row["rate"] is None or row["n"] == 0:
        return "no data"
    text = f"{row['rate'] * 100:.1f}%"
    if row["n"] < MIN_N:
        return f"{text}\nn<{MIN_N}"
    if row["lo"] is not None and row["hi"] is not None:
        text += f"\n[{row['lo'] * 100:.1f}, {row['hi'] * 100:.1f}]"
    return f"{text}\nn={row['n']}"


def _label(arm: str, alpha: float | None) -> str:
    return arm if alpha is None else f"{arm} (alpha={alpha:g})"


def _is_light(rgba: Sequence[float]) -> bool:
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b > 0.55


def _lang_order(langs: set[str]) -> list[str]:
    known = [lang for lang in LANGS if lang in langs]
    return known + sorted(langs - set(known))


def _new_figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)  # Agg draws it; also lets text be measured before saving
    return fig


def _save(fig: Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, format="png")


# ------------------------------------------------------------------------ heatmap

_ROW_IN = 1.05  # height of one heatmap row, inches
_HEAD_IN = 0.95  # room above the grid for title and column headers
_FOOT_IN = 1.25  # room below the grid for legend and colour bar
_LEFT = 0.11  # grid's left edge, as a fraction of the figure width


def _single_alphas(
    used: Sequence[Mapping], arms: Sequence[str], metric: str
) -> dict[str, float | None]:
    alpha_of: dict[str, float | None] = {}
    for arm in arms:
        alphas = {r["alpha"] for r in used if r["arm"] == arm}
        if len(alphas) > 1:
            shown = sorted(alphas, key=lambda a: (a is not None, a or 0.0))
            raise ValueError(
                f"arm {arm!r} has several alpha values {shown} for metric {metric!r}; "
                "filter the rows to one alpha per arm first"
            )
        alpha_of[arm] = next(iter(alphas), None)
    return alpha_of


def heatmap(
    rows: Sequence[Mapping],
    *,
    metric: str,
    arm_before: str,
    arm_after: str,
    low_conf: set[str],
    out_path: str | Path,
    deltas: Mapping[str, tuple[float, float, float]] | None = None,
) -> None:
    """One row per language, two columns (`arm_before`, `arm_after`) for one metric.

    Each cell shows the rate, its interval and n, coloured on a fixed 0-100% scale.
    A language's row is hatched when its change is inside the noise and dotted when
    its judge is unreliable (`cell_style`); the change itself is printed to the right.
    `deltas` maps a language to its paired difference (point, lo, hi), computed by the
    caller with `stats.paired_diff_ci`; a language missing from it has no interval and
    so counts as inside the noise.

    `rows` must hold one alpha per arm: filter to it first. Raises `ValueError`
    otherwise, or if no row matches `metric` and the two arms.
    """
    arms = (arm_before, arm_after)
    used = [r for r in rows if r["metric"] == metric and r["arm"] in arms]
    if not used:
        raise ValueError(f"no rows for metric {metric!r} and arms {list(arms)}; nothing to draw")
    alpha_of = _single_alphas(used, arms, metric)
    cell_of = {(r["lang"], r["arm"]): r for r in used}
    langs = _lang_order({r["lang"] for r in used})
    deltas = deltas or {}

    n = len(langs)
    height = _HEAD_IN + _ROW_IN * n + _FOOT_IN
    fig = _new_figure(9.0, height)
    ax = fig.add_axes((_LEFT, _FOOT_IN / height, 0.63, _ROW_IN * n / height))
    ax.set_xlim(0, 2)
    ax.set_ylim(n, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_xticks([0.5, 1.5], [_label(a, alpha_of[a]) for a in arms], fontsize=12, fontweight="bold")
    ax.set_yticks([r + 0.5 for r in range(n)], langs, fontsize=12, fontweight="bold")

    for r, lang in enumerate(langs):
        style = cell_style(deltas.get(lang), lang, low_conf)
        for c, arm in enumerate(arms):
            _draw_cell(ax, c, r, cell_of.get((lang, arm)), style)
        delta = deltas.get(lang)
        note = (
            f"{delta[0] * 100:+.1f} pp\n[{delta[1] * 100:+.1f}, {delta[2] * 100:+.1f}]"
            if delta is not None
            else "no paired\ninterval"
        )
        ax.text(2.06, r + 0.5, note, ha="left", va="center", fontsize=9.5, color=_GREY_TEXT)
    ax.annotate("change, after - before", (2.06, 0), xytext=(0, 4), textcoords="offset points",
                annotation_clip=False, ha="left", va="bottom", fontsize=9.5, fontweight="bold",
                color=_GREY_TEXT)

    fig.text(_LEFT, 1 - 0.3 / height, _METRIC_TITLES.get(metric, metric),
             fontsize=14, fontweight="bold", ha="left", va="center")
    _draw_heatmap_key(fig, height)
    _save(fig, out_path)


def _draw_cell(ax, col: int, row_idx: int, row: Mapping | None, style: str) -> None:
    head, _, detail = _cell_text(row).partition("\n")
    has_data = head != "no data"
    fill = _CMAP(_NORM(row["rate"])) if has_data else _NO_DATA_FILL
    ink = "black" if not has_data or _is_light(fill) else "white"
    patterned = has_data and style != "solid"
    rect = Rectangle(
        (col + 0.03, row_idx + 0.04), 0.94, 0.92, facecolor=fill, linewidth=0,
        edgecolor=ink if patterned else "none", hatch=_HATCH[style] if patterned else "",
    )
    rect.set_hatch_linewidth(0.8)
    ax.add_patch(rect)
    # Text over a pattern gets a white plate so it stays readable.
    plate = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}
    kw = {"ha": "center", "va": "center", "color": "black" if patterned else ink,
          "bbox": plate if patterned else None}
    ax.text(col + 0.5, row_idx + (0.36 if detail else 0.5), head,
            fontsize=13 if detail else 10, fontweight="bold", **kw)
    if detail:
        ax.text(col + 0.5, row_idx + 0.7, detail, fontsize=8.5, **kw)


def _draw_heatmap_key(fig: Figure, height: float) -> None:
    grey = {"facecolor": "#b8b8b8", "edgecolor": "black", "linewidth": 0.8}
    fig.legend(
        handles=[
            Patch(label="solid: change is clear (95% CI excludes 0)", **grey),
            Patch(hatch=_HATCH["hatched"],
                  label="hatched: change inside the noise (CI includes 0, or none)", **grey),
            Patch(hatch=_HATCH["low_confidence"],
                  label="dotted: low judge agreement, read with care", **grey),
        ],
        loc="lower left", bbox_to_anchor=(_LEFT - 0.02, 0.012), frameon=False, fontsize=9.5,
        handlelength=2.2, handleheight=1.4, labelspacing=0.7,
    )
    cax = fig.add_axes((0.66, 0.55 / height, 0.24, 0.17 / height))
    bar = fig.colorbar(ScalarMappable(norm=_NORM, cmap=_CMAP), cax=cax, orientation="horizontal")
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    bar.outline.set_visible(False)
    cax.tick_params(labelsize=9, length=2)
    fig.text(0.66, 0.95 / height, "colour = rate; text = rate, [95% CI], n", fontsize=9.5,
             color=_GREY_TEXT, ha="left", va="center")


# ------------------------------------------------------------------- alpha curve


def _percent(triple: Triple) -> Triple:
    return tuple(None if v is None else v * 100 for v in triple)


def _whisker_size(t: Triple):
    rate, lo, hi = t
    if lo is None or hi is None:
        return None
    return [[max(0.0, rate - lo)], [max(0.0, hi - rate)]]


def _draw_whiskers(ax, acc: Triple, asr: Triple, colour: str) -> None:
    xerr, yerr = _whisker_size(acc), _whisker_size(asr)
    if xerr is not None or yerr is not None:
        ax.errorbar(acc[0], asr[0], xerr=xerr, yerr=yerr, fmt="none", ecolor=colour,
                    elinewidth=0.8, capsize=0, alpha=0.3, clip_on=False, zorder=1)


_Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
_LABEL_PT = 7.5
# Where a label may sit relative to its point: nearest first, each ring of directions in turn.
_LABEL_SPOTS = tuple(
    (sx * dist, sy * dist)
    for dist in (7, 11, 16)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
)


def _hit(a: _Box, b: _Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_labels(ax, labels, markers, lines) -> None:
    """Write each (x, y, text, colour) label beside its point, clear of everything else.

    Takes the nearest spot that overlaps no earlier label, no marker (x, y, radius in
    points), no stretch of a line (list of (x, y) vertices) and stays inside the axes;
    when every spot collides, the one with the fewest collisions wins.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    to_px = ax.transData.transform

    def square(x: float, y: float, r: float) -> _Box:
        return (x - r, y - r, x + r, y + r)

    taken: list[_Box] = [square(*to_px((x, y)), r * px) for x, y, r in markers]
    for line in lines:
        pts = [to_px(p) for p in line]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            steps = max(1, int(((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5 / (3 * px)))
            taken += [
                square(xa + (xb - xa) * i / steps, ya + (yb - ya) * i / steps, 1.5 * px)
                for i in range(steps + 1)
            ]
    axes_box = ax.get_window_extent()
    inside: _Box = (axes_box.x0, axes_box.y0, axes_box.x1, axes_box.y1)

    for x, y, text, colour in labels:
        probe = ax.text(0, 0, text, fontsize=_LABEL_PT)
        extent = probe.get_window_extent(renderer)
        probe.remove()
        w, h = extent.width + 2 * px, extent.height + 2 * px  # the halo adds a little
        cx, cy = to_px((x, y))
        best: tuple[int, tuple, _Box] | None = None
        for dx, dy in _LABEL_SPOTS:
            left = cx + dx * px - (w if dx < 0 else 0 if dx > 0 else w / 2)
            bottom = cy + dy * px - (h if dy < 0 else 0 if dy > 0 else h / 2)
            box = (left, bottom, left + w, bottom + h)
            clashes = sum(_hit(box, t) for t in taken)
            clashes += not (inside[0] <= box[0] and box[2] <= inside[2]
                            and inside[1] <= box[1] and box[3] <= inside[3])
            if best is None or clashes < best[0]:
                best = (clashes, (dx, dy), box)
            if clashes == 0:
                break
        _, (dx, dy), box = best
        taken.append(box)
        ax.annotate(
            text, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=_LABEL_PT,
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            color=colour, path_effects=[withStroke(linewidth=2.5, foreground="white")], zorder=4,
        )


def alpha_curve(
    rows: Sequence[Mapping],
    *,
    langs: Sequence[str],
    out_path: str | Path,
) -> None:
    """Safety against task accuracy, one line per language along the A2 alpha grid.

    x is task accuracy (`task_acc`), y is attack success rate (`asr`; lower is safer).
    A language's line runs through its A2 points in alpha order, each labelled with
    its alpha; A0 and A1 are single markers (square, diamond). Faint whiskers show the
    intervals that exist. A language with no A2 point is skipped; `ValueError` if that
    leaves nothing to draw.
    """
    rates: dict[tuple, dict[str, Triple]] = {}
    for r in rows:
        if r["metric"] in ("asr", "task_acc") and r["rate"] is not None:
            key = (r["arm"], r["alpha"], r["lang"])
            rates.setdefault(key, {})[r["metric"]] = _percent((r["rate"], r["lo"], r["hi"]))

    def points(arm: str, lang: str) -> list[tuple[float | None, Triple, Triple]]:
        """(alpha, task_acc, asr), in percent, for every (arm, lang) point that has both."""
        found = [
            (alpha, m["task_acc"], m["asr"])
            for (a, alpha, lg), m in rates.items()
            if a == arm and lg == lang and "asr" in m and "task_acc" in m
        ]
        return sorted(found, key=lambda t: (t[0] is not None, t[0] or 0.0))

    curves = [(lang, [p for p in points("A2", lang) if p[0] is not None]) for lang in langs]
    curves = [(lang, pts) for lang, pts in curves if pts]
    if not curves:
        raise ValueError(f"no A2 rows with both asr and task_acc for any of {list(langs)}")

    fig = _new_figure(9.0, 5.6)
    ax = fig.add_axes((0.08, 0.12, 0.64, 0.72))
    labels, markers, lines = [], [], []
    for k, (lang, pts) in enumerate(curves):
        colour = _LANG_COLOURS[k % len(_LANG_COLOURS)]
        ax.plot([p[1][0] for p in pts], [p[2][0] for p in pts], "-o", color=colour,
                linewidth=1.6, markersize=5.5, clip_on=False, zorder=3)
        lines.append([(acc[0], asr[0]) for _, acc, asr in pts])
        for alpha, acc, asr in pts:
            _draw_whiskers(ax, acc, asr, colour)
            labels.append((acc[0], asr[0], f"{alpha:g}", colour))
            markers.append((acc[0], asr[0], 3.5))
        for arm, marker in (("A0", "s"), ("A1", "D")):
            for _, acc, asr in points(arm, lang):
                _draw_whiskers(ax, acc, asr, colour)
                ax.plot(acc[0], asr[0], marker, color=colour, markersize=9, linestyle="none",
                        markeredgecolor="black", markeredgewidth=1.0, clip_on=False, zorder=5)
                markers.append((acc[0], asr[0], 6.0))

    ax.set_xlabel("Task accuracy (%, higher is better)", fontsize=11)
    ax.set_ylabel("Attack success rate (%, lower is safer)", fontsize=11)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()  # rates cannot leave 0-100
    ax.set_xlim(max(0.0, x0), min(100.0, x1))
    ax.set_ylim(max(0.0, y0), min(100.0, y1))
    _place_labels(ax, labels, markers, lines)  # after the limits: it works in pixels

    fig.text(0.08, 0.955, "Safety against task accuracy along the A2 alpha grid", fontsize=14,
             fontweight="bold", ha="left", va="center")
    fig.text(0.08, 0.905, "Down and to the right is better. Numbers on the line are alpha.",
             fontsize=10.5, color=_GREY_TEXT, ha="left", va="center")
    handles = [
        Line2D([], [], color=_LANG_COLOURS[k % len(_LANG_COLOURS)], linewidth=1.6,
               marker="o", markersize=5.5, label=f"{lang} (A2 line)")
        for k, (lang, _) in enumerate(curves)
    ] + [
        Line2D([], [], color="#888888", marker=marker, markersize=8.5, markeredgecolor="black",
               linestyle="none", label=f"{arm} (one point)")
        for arm, marker in (("A0", "s"), ("A1", "D"))
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.745, 0.5), frameon=False,
               fontsize=10, labelspacing=0.9)
    _save(fig, out_path)
