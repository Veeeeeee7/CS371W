#!/usr/bin/env python3
"""The paper's result figures, drawn by make_paper.py from a run folder (no torch):

  fig_maps     Figure fig:maps: the selected points at S = 500, (a) OODSelect and (b) GTV + smoothing, over the
               Zone A test points and the 50 km test tiles, in EPSG:5070 (CONUS Albers, metres). Violet #6250d6 =
               selected and in a component (>= c = 20 points), red #e34948 = selected, not in one.
  fig_budgets  Figure fig:budgets: corr_te (top) and purity (bottom) of OODSelect, the Laplacian (mu 3) and GTV
               (mu 3) against the budget S; left each search alone, right followed by smoothing. Dotted line S = 500.
  fig_mu       Figure fig:mu: corr_te (top) and purity (bottom) of the Laplacian and GTV at S = 500 against mu (log
               axis), OODSelect (mu = 0) as a dashed reference line, dotted line at mu = 3.
Adapted from results/2026-11-11/paper_figures/fig_maps.py and fig_budgets.py (same fonts, colours and sizes);
fig_mu follows fig_budgets' style. Colours: the dataviz categorical slots 1-3 for the searches, in fixed order.
PDFs are written without a creation date, so the same data gives the same bytes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SERIES = [("OODSelect", "#2a78d6", "o"), ("Laplacian", "#eb6834", "s"), ("GTV", "#1baf7a", "^")]
IN_COMP, OUT_COMP = "#6250d6", "#e34948"   # violet, red
REPORTED_S = 500
REPORTED_MU = 3
PDF_META = {"CreationDate": None}

SMALL = {  # fig_budgets / fig_mu
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.titlesize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
    "ytick.major.size": 2, "xtick.minor.size": 1.2, "xtick.minor.width": 0.4, "pdf.fonttype": 42}
MAPS = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42,  # embed TrueType fonts; paper checkers reject Type 3
    "savefig.dpi": 300,
}


def _save(fig, out: Path, preview: Path | None, dpi: int = 250) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata=PDF_META)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------ maps
TILE_M = 50000      # test tiles are 50 km squares named '<i>_<j>'
SIMPLIFY_M = 300    # outline simplification: keeps the PDF small, invisible at print size
PAD_M = 10000       # map margin around the state
COUNTY_FILL = "#f0efec"
STATE_EDGE = "#b9b8b2"
POOL_GRAY = "#c9c8c2"


def fig_maps(counties_path: Path, pool_lonlat: tuple, tiles: list[str], panels: list[tuple],
             out: Path, preview: Path | None = None) -> list[str]:
    """panels: (title, DataFrame of the selected points with lon, lat, in_component). Returns one line per panel."""
    import geopandas as gpd
    from pyproj import Transformer
    to_albers = Transformer.from_crs(4269, 5070, always_xy=True)       # NAD83 lon/lat -> EPSG:5070
    lines = []
    with plt.rc_context(MAPS):
        counties = gpd.read_parquet(counties_path).to_crs(5070)
        state = counties.dissolve()
        counties["geometry"] = counties.geometry.simplify(SIMPLIFY_M)
        state["geometry"] = state.geometry.simplify(SIMPLIFY_M)
        pool_xy = to_albers.transform(*pool_lonlat)
        minx, miny, maxx, maxy = state.total_bounds
        fig = plt.figure(figsize=(6.3, 2.35))
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.42], wspace=0.03,
                                left=0.0, right=1.0, bottom=0.0, top=0.93)
        for k, (title, points) in enumerate(panels):
            ax = fig.add_subplot(grid[0, k])
            counties.plot(ax=ax, color=COUNTY_FILL, edgecolor="white", linewidth=0.25, zorder=0)
            state.boundary.plot(ax=ax, color=STATE_EDGE, linewidth=0.5, zorder=1)
            ax.scatter(pool_xy[0], pool_xy[1], s=0.12, color=POOL_GRAY, linewidths=0, zorder=2, rasterized=True)
            for tile in tiles:
                i, j = tile.split("_")
                ax.add_patch(Rectangle((int(i) * TILE_M, int(j) * TILE_M), TILE_M, TILE_M, fill=False,
                                       ec=MUTED, lw=0.4, zorder=3))
            ax.set_xlim(minx - PAD_M, maxx + PAD_M)
            ax.set_ylim(miny - PAD_M, maxy + PAD_M)
            ax.set_aspect("equal")
            ax.axis("off")
            inside = points.in_component.to_numpy(bool)
            x, y = to_albers.transform(points.lon.to_numpy(), points.lat.to_numpy())
            x, y = np.asarray(x), np.asarray(y)
            ax.scatter(x[~inside], y[~inside], s=1.6, color=OUT_COMP, linewidths=0, zorder=4)
            ax.scatter(x[inside], y[inside], s=1.6, color=IN_COMP, linewidths=0, zorder=5)
            n_comp = points.loc[inside, "component"].nunique()
            lines.append(f"{title}: {len(points)} points, {n_comp} components holding {int(inside.sum())}")
            ax.set_title(f"{title}: {n_comp} components", fontsize=7.2, loc="left", pad=1)
            if k == 1:  # 100 km scale bar in the lower left of panel (b)
                bar_x, bar_y = minx + 40000, miny + 80000
                ax.plot([bar_x, bar_x + 100000], [bar_y, bar_y], color=INK, lw=1.1, solid_capstyle="butt")
                ax.text(bar_x + 50000, bar_y + 8000, "100 km", ha="center", va="bottom", fontsize=6.2)
        ax_legend = fig.add_subplot(grid[0, 2])
        ax_legend.axis("off")

        def dot(color, label):
            return Line2D([0], [0], marker="o", ls="", color=color, ms=2.8, label=label)
        handles = [dot(IN_COMP, "Selected, in a\ncomponent"), dot(OUT_COMP, "Selected, not in\na component"),
                   dot(POOL_GRAY, "Other Zone A\ntest point"),
                   Line2D([0], [0], marker="s", ls="", mfc="none", mec=MUTED, ms=4.5, label="Test tile")]
        ax_legend.legend(handles=handles, loc="center left", frameon=False, fontsize=6.4,
                         handletextpad=0.3, labelspacing=1.1)
        _save(fig, out, preview)
    return lines


# ------------------------------------------------------------------ budgets
def fig_budgets(d: pd.DataFrame, out: Path, preview: Path | None = None) -> None:
    """d: the run's subsets table (search, smoothing, S, r_te, purity) restricted to the budget grid."""
    with plt.rc_context(SMALL):
        fig, axes = plt.subplots(2, 2, figsize=(3.03, 2.9), sharex=True, sharey="row",
                                 gridspec_kw=dict(hspace=0.12, wspace=0.08, left=0.15, right=0.98, top=0.86,
                                                  bottom=0.12))
        for j, sm in enumerate((False, True)):
            ax_r, ax_p = axes[0, j], axes[1, j]
            for name, color, marker in SERIES:
                s = d[(d.search == name) & (d.smoothing == sm)].sort_values("S")
                for ax, col in [(ax_r, "r_te"), (ax_p, "purity")]:
                    ax.plot(s.S, s[col], color=color, lw=1.1, marker=marker, ms=2.6, mew=0, label=name, zorder=3)
            ax_r.set_title("Search alone" if not sm else "+ smoothing", pad=2)
            for ax in (ax_r, ax_p):
                ax.axvline(REPORTED_S, color=MUTED, lw=0.6, ls=":", zorder=1)
                ax.yaxis.grid(True, color=GRID, lw=0.5)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            ax_r.axhline(0, color=INK2, lw=0.5, zorder=1)
            ax_p.set_xscale("log")
            ax_p.set_xticks([10, 100, 1000, 10000])
            ax_p.set_xticklabels(["10", "100", "1k", "10k"])
        axes[0, 0].set_ylim(-1.05, 0.7)
        axes[0, 0].set_yticks([-1, -0.5, 0, 0.5])
        axes[0, 0].set_ylabel(r"corr$_\mathrm{te}$")
        axes[1, 0].set_ylim(0, 1.0)
        axes[1, 0].set_yticks([0, 0.5, 1])
        axes[1, 0].set_ylabel("Purity")
        fig.supxlabel("Budget $S$ (points)", fontsize=7, y=0.01)
        h, l = axes[1, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.0,
                   bbox_to_anchor=(0.56, 1.0))
        _save(fig, out, preview)


# ------------------------------------------------------------------ mu
def fig_mu(d: pd.DataFrame, reference: dict, out: Path, preview: Path | None = None) -> None:
    """d: the plain (unsmoothed) Laplacian and GTV searches at S = 500 (search, mu, r_te, purity); reference:
    OODSelect's r_te and purity at S = 500 (mu = 0), drawn as a dashed line."""
    with plt.rc_context(SMALL):
        fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(3.03, 2.6), sharex=True,
                                         gridspec_kw=dict(hspace=0.12, left=0.15, right=0.98, top=0.88,
                                                          bottom=0.13))
        name0, color0, _ = SERIES[0]
        for ax, col in [(ax_r, "r_te"), (ax_p, "purity")]:
            ax.axhline(reference[col], color=color0, lw=0.9, ls="--", zorder=2, label=f"{name0} ($\\mu=0$)")
            for name, color, marker in SERIES[1:]:
                s = d[d.search == name].sort_values("mu")
                ax.plot(s.mu, s[col], color=color, lw=1.1, marker=marker, ms=2.6, mew=0, label=name, zorder=3)
            ax.axvline(REPORTED_MU, color=MUTED, lw=0.6, ls=":", zorder=1)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # corr_te axis fitted to the values (all negative here), ticks every 0.1, the zero line only when in range.
        vals = np.r_[d.r_te.to_numpy(float), reference["r_te"]]
        lo, hi = np.floor(vals.min() * 10) / 10, np.ceil(vals.max() * 10) / 10
        ax_r.set_ylim(lo - 0.02, hi + 0.02)
        ax_r.set_yticks(np.round(np.arange(lo, hi + 1e-9, 0.1), 1))
        if lo < 0 < hi:
            ax_r.axhline(0, color=INK2, lw=0.5, zorder=1)
        ax_r.set_ylabel(r"corr$_\mathrm{te}$")
        ax_p.set_ylim(0, 1.0)
        ax_p.set_yticks([0, 0.5, 1])
        ax_p.set_ylabel("Purity")
        ax_p.set_xscale("log")
        ticks = sorted(d.mu.unique())
        ax_p.set_xticks(ticks)
        ax_p.set_xticklabels([f"{t:g}" for t in ticks])
        ax_p.minorticks_off()
        ax_p.set_xlabel(r"Regularization weight $\mu$")
        h, l = ax_r.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.0,
                   bbox_to_anchor=(0.56, 1.0))
        _save(fig, out, preview)
