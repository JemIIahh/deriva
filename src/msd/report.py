"""Report figures: the before/after heatmap and the alpha trade-off curve.

Both take `aggregate.build_table` rows. Figures are drawn straight onto an Agg
canvas (no pyplot, so no global state), use a colour-blind-safe palette, and carry
only ASCII text: language codes, never Arabic or Devanagari glyphs, so no font is
needed beyond matplotlib's own.
"""

import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
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

# Largest gap, as a fraction, between a supplied paired difference and (after rate - before
# rate) of the two cells shown; the paired difference can differ slightly when items were excluded.
DELTA_TOLERANCE = 0.05

# Heatmap cell colour: cividis is perceptually uniform and readable with every common
# colour vision deficiency. Rates sit on a fixed 0-100% scale so figures compare.
_CMAP = colormaps["cividis"]
_NORM = Normalize(vmin=0.0, vmax=1.0)
_NO_DATA_FILL = "#e3e3e3"
_GREY_TEXT = "#333333"

# Cell pattern per style. Diagonal lines and dots look nothing alike, even in greyscale.
_HATCH = {"solid": "", "hatched": "////", "low_confidence": "....", "excluded": "xxxx"}

# Okabe-Ito: the standard colour-blind-safe qualitative palette (black and yellow left out).
_LANG_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")

_METRIC_TITLES = {
    "asr": "Attack success rate (lower is safer)",
    "over_refusal": "Over-refusal rate (lower is better)",
    "task_acc": "Task accuracy (higher is better)",
}

Triple = tuple[float, float | None, float | None]  # rate, lo, hi


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _valid_interval(delta_ci: tuple | None) -> bool:
    """A paired difference (point, lo, hi) that can be drawn: all finite and lo <= hi."""
    return delta_ci is not None and all(_finite(v) for v in delta_ci) and delta_ci[1] <= delta_ci[2]


def cell_style(delta_ci: tuple | None, lang: str, low_conf: set[str]) -> str:
    """How a language's before/after comparison is drawn.

    "low_confidence" if the judge is unreliable for `lang` (wins over everything);
    "solid" only if the change is clear, i.e. its interval (point, lo, hi) is finite,
    ordered (lo <= hi) and excludes 0; otherwise "hatched": the change is inside the
    noise, or there is no usable interval to judge it by (missing, nan, swapped).
    """
    if lang in low_conf:
        return "low_confidence"
    if not _valid_interval(delta_ci):
        return "hatched"
    _, lo, hi = delta_ci
    return "solid" if lo > 0.0 or hi < 0.0 else "hatched"


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
    fig = Figure(figsize=(width, height), dpi=DPI)
    FigureCanvasAgg(fig)  # Agg draws it; also lets text be measured before saving
    return fig


def _save(fig: Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, format="png")


# ------------------------------------------------------------------------ heatmap

_ROW_IN = 1.05  # height of one heatmap row, inches
_HEAD_IN = 0.95  # room above the grid for title and column headers
_FOOT_IN = 1.75  # room below the grid for colour bar and legend
_FOOT_EXTRA_IN = 0.35  # one more legend line, when some cell is left out
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


def _measured(cell: Mapping | None) -> bool:
    """True if the cell has a rate from at least `MIN_N` items."""
    return cell is not None and cell["rate"] is not None and cell["n"] >= MIN_N


def _shown_delta(
    lang: str, before: Mapping | None, after: Mapping | None, delta: tuple | None
) -> tuple | None:
    """The paired difference to print for `lang`, or None if the row has no usable interval.

    A supplied delta is ignored when either displayed cell is unmeasured (no data, or
    fewer than `MIN_N` items) or the interval is not a finite, ordered one. Otherwise its
    point estimate must sit within `DELTA_TOLERANCE` of (after rate - before rate), so a
    delta computed for other arms or another slice cannot be drawn next to these cells.
    """
    if delta is None or not (_measured(before) and _measured(after)) or not _valid_interval(delta):
        return None
    difference = after["rate"] - before["rate"]
    if abs(delta[0] - difference) > DELTA_TOLERANCE:
        raise ValueError(
            f"paired difference for language {lang!r} is {delta[0]:+.2f} but its displayed cells "
            f"differ by {difference:+.2f} ({before['rate']:.2f} -> {after['rate']:.2f}); "
            f"more than the tolerance of {DELTA_TOLERANCE}, so it was not computed for these cells"
        )
    return delta


def heatmap(
    rows: Sequence[Mapping],
    *,
    metric: str,
    arm_before: str,
    arm_after: str,
    low_conf: set[str],
    out_path: str | Path,
    deltas: Mapping[str, tuple[float, float, float]] | None = None,
    excluded: Mapping[tuple[str, str], str] | None = None,
) -> None:
    """One row per language, two columns (`arm_before`, `arm_after`) for one metric.

    Each cell shows the rate, its interval and n, coloured on a fixed 0-100% scale.
    A language's row is hatched when its change is inside the noise and dotted when
    its judge is unreliable (`cell_style`); the change itself is printed to the right.
    `deltas` maps a language to its paired difference (point, lo, hi), computed by the
    caller with `stats.paired_diff_ci`; a language missing from it has no interval and
    so counts as inside the noise. So does a language with an unmeasured cell (no data,
    or fewer than `MIN_N` items): its delta is ignored and "no paired interval" shown.
    `excluded` maps (language, arm) to a short reason for a cell that was left out of the table (too many
    judge errors, failed sanity check): it is drawn crossed and says "excluded" and the reason, not "no data".
    A cell named there must have no row in `rows` (`ValueError` otherwise).

    `rows` must hold one alpha per arm: filter to it first. Raises `ValueError`
    otherwise, if no row matches `metric` and the two arms, if two rows describe the same
    cell, or if a delta's point estimate is more than `DELTA_TOLERANCE` away from the
    difference of the two cells shown (it was computed for something else). Nothing is
    written unless the whole figure could be built.
    """
    fig = _build_heatmap_figure(
        rows, metric=metric, arm_before=arm_before, arm_after=arm_after,
        low_conf=low_conf, deltas=deltas, excluded=excluded,
    )
    _save(fig, out_path)


def _build_heatmap_figure(
    rows: Sequence[Mapping],
    *,
    metric: str,
    arm_before: str,
    arm_after: str,
    low_conf: set[str],
    deltas: Mapping[str, tuple[float, float, float]] | None = None,
    excluded: Mapping[tuple[str, str], str] | None = None,
) -> Figure:
    """The figure `heatmap` saves; split out so tests can inspect what was drawn."""
    arms = (arm_before, arm_after)
    excluded = {key: reason for key, reason in (excluded or {}).items() if key[1] in arms}
    used = [r for r in rows if r["metric"] == metric and r["arm"] in arms]
    if not used:
        raise ValueError(f"no rows for metric {metric!r} and arms {list(arms)}; nothing to draw")
    alpha_of = _single_alphas(used, arms, metric)
    cell_of: dict[tuple[str, str], Mapping] = {}
    for r in used:
        key = (r["lang"], r["arm"])
        if key in cell_of:
            raise ValueError(
                f"duplicate row for arm {r['arm']!r}, alpha {r['alpha']!r}, lang {r['lang']!r}, "
                f"metric {metric!r}; one of them would silently win"
            )
        cell_of[key] = r
    for lang, arm in excluded:
        if (lang, arm) in cell_of:
            raise ValueError(
                f"cell {arm!r} / language {lang!r} is listed as excluded ({excluded[(lang, arm)]}) but also has a "
                f"row for metric {metric!r}; a cell is either in the table or left out of it"
            )
    langs = _lang_order({r["lang"] for r in used} | {lang for lang, _ in excluded})
    deltas = deltas or {}
    shown = {lang: _shown_delta(lang, cell_of.get((lang, arm_before)),
                                cell_of.get((lang, arm_after)), deltas.get(lang))
             for lang in langs}  # checks every supplied delta before anything is drawn

    n = len(langs)
    foot = _FOOT_IN + (_FOOT_EXTRA_IN if excluded else 0.0)
    height = _HEAD_IN + _ROW_IN * n + foot
    fig = _new_figure(9.0, height)
    ax = fig.add_axes((_LEFT, foot / height, 0.63, _ROW_IN * n / height))
    ax.set_xlim(0, 2)
    ax.set_ylim(n, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_xticks([0.5, 1.5], [_label(a, alpha_of[a]) for a in arms], fontsize=12, fontweight="bold")
    ax.set_yticks([r + 0.5 for r in range(n)], langs, fontsize=12, fontweight="bold")

    for r, lang in enumerate(langs):
        delta = shown[lang]
        style = cell_style(delta, lang, low_conf)
        for c, arm in enumerate(arms):
            _draw_cell(ax, c, r, cell_of.get((lang, arm)), style, gid=f"cell:{lang}:{arm}",
                       left_out=excluded.get((lang, arm)))
        note = (
            f"{delta[0] * 100:+.1f} pp\n[{delta[1] * 100:+.1f}, {delta[2] * 100:+.1f}]"
            if delta is not None
            else "no paired\ninterval"
        )
        ax.text(2.06, r + 0.5, note, ha="left", va="center", fontsize=9.5, color=_GREY_TEXT,
                gid=f"note:{lang}")
    ax.annotate("change, after - before", (2.06, 0), xytext=(0, 4), textcoords="offset points",
                annotation_clip=False, ha="left", va="bottom", fontsize=9.5, fontweight="bold",
                color=_GREY_TEXT)

    fig.text(_LEFT, 1 - 0.3 / height, _METRIC_TITLES.get(metric, metric),
             fontsize=14, fontweight="bold", ha="left", va="center")
    _draw_heatmap_key(fig, height, foot, has_excluded=bool(excluded))
    return fig


def _draw_cell(
    ax, col: int, row_idx: int, row: Mapping | None, style: str, *, gid: str, left_out: str | None = None
) -> None:
    if left_out is not None:
        _draw_left_out_cell(ax, col, row_idx, left_out, gid)
        return
    head, _, detail = _cell_text(row).partition("\n")
    has_data = head != "no data"
    fill = _CMAP(_NORM(row["rate"])) if has_data else _NO_DATA_FILL
    ink = "black" if not has_data or _is_light(fill) else "white"
    patterned = has_data and style != "solid"
    rect = Rectangle(
        (col + 0.03, row_idx + 0.04), 0.94, 0.92, facecolor=fill, linewidth=0,
        edgecolor=ink if patterned else "none", hatch=_HATCH[style] if patterned else "",
        gid=gid,
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


def _draw_left_out_cell(ax, col: int, row_idx: int, reason: str, gid: str) -> None:
    """A cell that was left out of the table: grey, crossed, and it says why (never a bare "no data")."""
    rect = Rectangle(
        (col + 0.03, row_idx + 0.04), 0.94, 0.92, facecolor=_NO_DATA_FILL, linewidth=0,
        edgecolor="black", hatch=_HATCH["excluded"], gid=gid,
    )
    rect.set_hatch_linewidth(0.8)
    ax.add_patch(rect)
    plate = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}
    ax.text(col + 0.5, row_idx + 0.40, "excluded", fontsize=12, fontweight="bold", ha="center", va="center",
            color="black", bbox=plate)
    ax.text(col + 0.5, row_idx + 0.68, reason, fontsize=8.5, ha="center", va="center", color="black", bbox=plate)


def _draw_heatmap_key(fig: Figure, height: float, foot: float, *, has_excluded: bool) -> None:
    grey = {"facecolor": "#b8b8b8", "edgecolor": "black", "linewidth": 0.8}
    handles = [
        Patch(label="solid row: change is clear (CI excludes 0)", **grey),
        Patch(hatch=_HATCH["hatched"],
              label="hatched row: change between the two columns is inside the noise "
                    "(CI includes 0) or could not be measured", **grey),
        Patch(hatch=_HATCH["low_confidence"],
              label="dotted row: the two judges disagree too much, or were not "
                    "double-checked; read with care", **grey),
    ]
    if has_excluded:
        handles.append(Patch(hatch=_HATCH["excluded"],
                             label="crossed cell: left out of the table (too many judge errors, or the "
                                   "model failed the sanity check)", **grey))
    fig.legend(
        handles=handles,
        loc="lower left", bbox_to_anchor=(_LEFT - 0.02, 0.1 / height), frameon=False, fontsize=9.5,
        handlelength=2.2, handleheight=1.4, labelspacing=0.7,
    )
    cax = fig.add_axes((_LEFT, (foot - 0.47) / height, 0.26, 0.17 / height))
    bar = fig.colorbar(ScalarMappable(norm=_NORM, cmap=_CMAP), cax=cax, orientation="horizontal")
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    bar.outline.set_visible(False)
    cax.tick_params(labelsize=9, length=2)
    fig.text(_LEFT + 0.28, (foot - 0.385) / height, "colour = rate; text = rate, [95% CI], n", fontsize=9.5,
             color=_GREY_TEXT, ha="left", va="center")


# ------------------------------------------------------------------- alpha curve


def _lang_colours(langs: Sequence[str]) -> dict[str, str]:
    """Line colour per language: its place in `config.LANGS`, so it never shifts when another
    language is skipped. Languages outside `LANGS` continue after them, in sorted order."""
    index = {lang: i for i, lang in enumerate(LANGS)}
    index.update({lang: len(LANGS) + i for i, lang in enumerate(sorted(set(langs) - set(LANGS)))})
    return {lang: _LANG_COLOURS[index[lang] % len(_LANG_COLOURS)] for lang in langs}


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
_MARKER_R_PT = 2.75  # radius of an A2 line marker
_LEADER_PX = 12  # a label whose box is further than this from its point gets a leader line
# Where a label may sit relative to its point: distance in points (nearest ring first), then
# direction in degrees (diagonals before axes before in-betweens).
_RINGS_PT = (5.5, 9, 13, 18, 24, 31, 40)
_ANGLES = (45, -45, 135, -135, 0, 180, 90, -90, 22.5, -22.5, 157.5, -157.5, 67.5, -67.5, 112.5, -112.5)


def _overlaps(boxes: Sequence[_Box], others: np.ndarray) -> int:
    """How many (box, other) pairs overlap; `others` is an (n, 4) array of boxes."""
    if not len(boxes) or not len(others):
        return 0
    a = np.asarray(boxes, dtype=float)[:, None, :]
    b = others[None, :, :]
    return int(((a[..., 0] < b[..., 2]) & (b[..., 0] < a[..., 2])
                & (a[..., 1] < b[..., 3]) & (b[..., 1] < a[..., 3])).sum())


def _plan_labels(ax, labels, markers, lines) -> tuple[list[tuple], int]:
    """Choose a spot for each (x, y, text, colour, gid) label: [(label, spot, tail)], clashes.

    A spot is the nearest one whose box overlaps no earlier label or leader line, no marker
    (x, y, radius in points) and no stretch of a line (list of (x, y) vertices), and stays
    inside the axes; when every spot collides, the one with the fewest collisions wins and
    its collisions are added to the count that is returned. A label that ends up further
    than `_LEADER_PX` from its point is tied to it by a thin leader line (the `tail`),
    which must not cross other labels, markers or lines either. The most crowded points
    choose first, since they have the fewest options; drawing order is the input order.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    to_px = ax.transData.transform

    def square(x: float, y: float, r: float) -> _Box:
        return (x - r, y - r, x + r, y + r)

    seed: list[_Box] = [square(*to_px((x, y)), r * px) for x, y, r in markers]
    for line in lines:
        pts = [to_px(p) for p in line]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            steps = max(1, int(((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5 / (3 * px)))
            seed += [
                square(xa + (xb - xa) * i / steps, ya + (yb - ya) * i / steps, 1.5 * px)
                for i in range(steps + 1)
            ]
    taken = np.asarray(seed, dtype=float).reshape(-1, 4)  # markers, lines, then placed labels
    axes_box = ax.get_window_extent()
    inside: _Box = (axes_box.x0, axes_box.y0, axes_box.x1, axes_box.y1)

    def leader(cx: float, cy: float, box: _Box) -> list[_Box]:
        """Small squares along the straight leader from the point to the nearest spot of `box`."""
        nx, ny = min(max(cx, box[0]), box[2]), min(max(cy, box[1]), box[3])
        length = math.hypot(nx - cx, ny - cy)
        if length <= _LEADER_PX:
            return []
        steps = range(int(2 * px + _MARKER_R_PT * px), int(length - 2), 3)  # leave the marker alone
        return [square(cx + (nx - cx) * d / length, cy + (ny - cy) * d / length, 1.0) for d in steps]

    where = [to_px((lab[0], lab[1])) for lab in labels]
    crowd = [sum(math.hypot(a[0] - b[0], a[1] - b[1]) < 60 * px for b in where) for a in where]
    chosen: dict[int, tuple] = {}
    total = 0
    for i in sorted(range(len(labels)), key=lambda i: -crowd[i]):
        x, y, text, colour, gid = labels[i]
        probe = ax.text(0, 0, text, fontsize=_LABEL_PT)
        extent = probe.get_window_extent(renderer)
        probe.remove()
        w, h = extent.width + 2 * px, extent.height + 2 * px  # the halo adds a little
        cx, cy = where[i]
        best: tuple[int, tuple, _Box, list[_Box]] | None = None
        for ring in _RINGS_PT:
            for angle in _ANGLES:
                ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                ha = "left" if ux > 0.38 else "right" if ux < -0.38 else "center"
                va = "bottom" if uy > 0.38 else "top" if uy < -0.38 else "center"
                left = cx + ring * ux * px - {"left": 0, "right": w, "center": w / 2}[ha]
                bottom = cy + ring * uy * px - {"bottom": 0, "top": h, "center": h / 2}[va]
                box = (left, bottom, left + w, bottom + h)
                tail = leader(cx, cy, box)
                clashes = _overlaps([box], taken) + _overlaps(tail, taken)
                clashes += not (inside[0] <= box[0] and box[2] <= inside[2]
                                and inside[1] <= box[1] and box[3] <= inside[3])
                if best is None or clashes < best[0]:
                    best = (clashes, (ring * ux, ring * uy, ha, va), box, tail)
                if clashes == 0:
                    break
            else:
                continue
            break
        clashes, spot, box, tail = best
        total += clashes
        taken = np.vstack([taken, np.asarray([box, *tail], dtype=float)])
        chosen[i] = (labels[i], spot, bool(tail))
    return [chosen[i] for i in range(len(labels))], total


def _draw_labels(ax, plans: Sequence[tuple]) -> None:
    for (x, y, text, colour, gid), (dx, dy, ha, va), has_leader in plans:
        ax.annotate(
            text, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=_LABEL_PT,
            ha=ha, va=va, color=colour, zorder=4, gid=gid,
            path_effects=[withStroke(linewidth=2.5, foreground="white")],
            arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.7, "shrinkA": 0,
                        "shrinkB": _MARKER_R_PT + 1} if has_leader else None,
        )


def alpha_curve(
    rows: Sequence[Mapping],
    *,
    langs: Sequence[str],
    out_path: str | Path,
    chosen_alpha: float | None = None,
    excluded: Mapping[str, Sequence[float | str]] | None = None,
    not_chosen: Mapping[float, Sequence[str]] | None = None,
) -> None:
    """Safety against task accuracy, one line per language along the A2 alpha grid.

    x is task accuracy (`task_acc`), y is attack success rate (`asr`; lower is safer).
    A language's line runs through its A2 points in alpha order, each labelled with
    its alpha; A0 and A1 are single markers (square, diamond). Faint whiskers show the
    intervals that exist. A language with no A2 point is skipped; `ValueError` if that
    leaves nothing to draw.

    `chosen_alpha` (the alpha picked on dev) gets a ring around each language's point at that alpha, named in
    the legend. `excluded` maps a language to what was left out of the table for it (so it is not in `rows`):
    A2 alphas as numbers, a single-model marker as its arm name ("A0", "A1"); a line under the plot lists them,
    because a gap in a line, or a missing marker, must not go unexplained. `not_chosen` maps an alpha that was
    not eligible to be chosen (its cells are in the table) to the reasons, one short text each, and is listed
    on a line of its own.
    """
    _save(_build_alpha_curve_figure(rows, langs=langs, chosen_alpha=chosen_alpha, excluded=excluded,
                                    not_chosen=not_chosen), out_path)


_ALPHA_TOLERANCE = 1e-9


def _left_out_note(excluded: Mapping[str, Sequence[float | str]] | None) -> str:
    """"hi: A1, alpha 0.5; sw: alpha 0.5, 1" for the cells left out, languages in config order; "" if none.

    Arm names (A0, A1) come first, then the alphas in ascending order.
    """
    parts = []
    for lang in _lang_order(set(excluded or {})):
        items = list(excluded[lang])
        arms = sorted({str(x) for x in items if isinstance(x, str)})
        alphas = sorted({float(x) for x in items if not isinstance(x, str)})
        text = [*arms, *([f"alpha {', '.join(f'{a:g}' for a in alphas)}"] if alphas else [])]
        if text:
            parts.append(f"{lang}: {', '.join(text)}")
    return "; ".join(parts)


def _not_chosen_note(not_chosen: Mapping[float, Sequence[str]] | None) -> str:
    """"alpha 0.5 (hi 50% vs 20% allowed; ar 60% vs 20% allowed)" for the alphas not eligible to be chosen."""
    parts = [f"alpha {alpha:g} ({'; '.join(not_chosen[alpha])})" for alpha in sorted(not_chosen or {}) if not_chosen[alpha]]
    return "; ".join(parts)


def _build_alpha_curve_figure(
    rows: Sequence[Mapping],
    *,
    langs: Sequence[str],
    chosen_alpha: float | None = None,
    excluded: Mapping[str, Sequence[float | str]] | None = None,
    not_chosen: Mapping[float, Sequence[str]] | None = None,
) -> Figure:
    """The figure `alpha_curve` saves; split out so tests can inspect what was drawn."""
    rates: dict[tuple, dict[str, Triple]] = {}
    seen: set[tuple] = set()
    for r in rows:
        if r["metric"] not in ("asr", "task_acc"):
            continue
        row_key = (r["arm"], r["alpha"], r["lang"], r["metric"])
        if row_key in seen:
            raise ValueError(
                f"duplicate row for arm {r['arm']!r}, alpha {r['alpha']!r}, lang {r['lang']!r}, "
                f"metric {r['metric']!r}; one of them would silently win"
            )
        seen.add(row_key)
        if r["rate"] is not None:
            rates.setdefault(row_key[:3], {})[r["metric"]] = _percent((r["rate"], r["lo"], r["hi"]))

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

    colour_of = _lang_colours(langs)
    note = _left_out_note(excluded)
    refused = _not_chosen_note(not_chosen)
    note_lines = (textwrap.wrap(f"Left out (cell excluded, see reliability.json): {note}", width=110) if note else [])
    note_lines += (textwrap.wrap(f"Not chosen (refuses too many benign prompts, still in the table): {refused}",
                                 width=110) if refused else [])
    extra = 0.18 * len(note_lines)  # room under the x label for the note, inches
    height = 5.6 + extra
    fig = _new_figure(9.0, height)
    ax = fig.add_axes((0.08, (0.12 * 5.6 + extra) / height, 0.64, 0.72 * 5.6 / height))
    rung = []  # (lang, x, y) of every point at the chosen alpha
    labels, ends, markers, lines = [], [], [], []
    for lang, pts in curves:
        colour = colour_of[lang]
        ax.plot([p[1][0] for p in pts], [p[2][0] for p in pts], "-o", color=colour,
                linewidth=1.6, markersize=5.5, clip_on=False, zorder=3, gid=f"line:{lang}")
        lines.append([(acc[0], asr[0]) for _, acc, asr in pts])
        for alpha, acc, asr in pts:
            _draw_whiskers(ax, acc, asr, colour)
            label = (acc[0], asr[0], f"{alpha:g}", colour, f"label:{lang}:{alpha:g}")
            labels.append(label)
            if alpha in (pts[0][0], pts[-1][0]):
                ends.append(label)
            markers.append((acc[0], asr[0], 3.5))
            if chosen_alpha is not None and abs(alpha - chosen_alpha) < _ALPHA_TOLERANCE:
                ax.plot(acc[0], asr[0], "o", markersize=13, markerfacecolor="none", markeredgecolor="black",
                        markeredgewidth=1.6, linestyle="none", clip_on=False, zorder=6, gid=f"chosen:{lang}")
                rung.append(lang)
                markers.append((acc[0], asr[0], 7.0))
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
    # After the limits: it works in pixels. If some label cannot be put clear of the rest,
    # only each line's two ends are named and the subtitle says so.
    plans, clashes = _plan_labels(ax, labels, markers, lines)
    ends_only = clashes > 0
    if ends_only:
        plans, _ = _plan_labels(ax, ends, markers, lines)
    _draw_labels(ax, plans)

    def from_top(inches: float) -> float:
        return 1 - inches / height

    fig.text(0.08, from_top(0.25), "Safety against task accuracy along the A2 alpha grid", fontsize=14,
             fontweight="bold", ha="left", va="center")
    fig.text(0.08, from_top(0.48 if ends_only else 0.53), "Down and to the right is better."
             if ends_only else "Down and to the right is better. Numbers on the line are alpha.",
             fontsize=10.5, color=_GREY_TEXT, ha="left", va="center")
    if ends_only:
        fig.text(0.08, from_top(0.69), "Numbers mark alpha at each line's two ends; the points between are "
                 "the alphas in between, in order.", fontsize=10.5, color=_GREY_TEXT,
                 ha="left", va="center")
    for i, line in enumerate(note_lines):
        fig.text(0.08, (0.20 + 0.18 * (len(note_lines) - 1 - i)) / height, line,
                 fontsize=9, color=_GREY_TEXT, ha="left", va="center")
    handles = [
        Line2D([], [], color=colour_of[lang], linewidth=1.6,
               marker="o", markersize=5.5, label=f"{lang} (A2 line)")
        for lang, _ in curves
    ] + [
        Line2D([], [], color="#888888", marker=marker, markersize=8.5, markeredgecolor="black",
               linestyle="none", label=f"{arm} (one point)")
        for arm, marker in (("A0", "s"), ("A1", "D"))
    ]
    if rung:
        handles.append(Line2D([], [], color="black", marker="o", markersize=11, markerfacecolor="none",
                              markeredgewidth=1.6, linestyle="none",
                              label=f"ring: chosen on dev\n(alpha {chosen_alpha:g})"))  # two lines: one ran off the page
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.745, 0.5), frameon=False,
               fontsize=10, labelspacing=0.9)
    return fig
