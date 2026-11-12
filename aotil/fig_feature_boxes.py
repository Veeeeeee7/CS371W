#!/usr/bin/env python3
"""
The paper's data figure (Figure fig:data): each model feature by set (Train and ID test are Zone AE,
OOD is Zone A) of the seed-0 split. Box plots for the six numeric features (the box spans the 25th
to 75th percentile, the line is the median, the whiskers reach the 5th and 95th percentiles, no
outliers drawn) and one 100%-stacked bar per set for the three categorical ones (land cover: its
seven most common classes plus "other"; soil flood frequency; fluvial class, over the points with a
soil class).

Adapted from Victor's approved design, Claude outputs/fig-feature-boxes-2026-11-07/fig_feature_boxes.py
(docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md Part F): the sets come from zoo.split_tiles and
zoo.subset, terrain variability is elev_sd_r300, the water-unit recoding is gone (pull_soil_points.py
already leaves water units missing), and land cover's classes are the seven most common of the data.
Layout, colours and whiskers are the design's.

run_experiments.py --stage data writes the sets (figure_sets) and the land-cover classes (top_classes) to the run
folder; make_paper.py draws the figure from them (draw).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zoo  # noqa: E402

INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#898781"
SETS = [("Train", "#86b6ef"), ("ID test", "#2a78d6"), ("OOD", "#eb6834")]   # AE light/dark blue, A orange
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # slots 1-7
CAT_DARK_TEXT = {1, 2, 3, 4}       # slots whose fill is under 3:1 against white text
ORD = ["#86b6ef", "#599be8", "#307edc", "#2164b7", "#164c91", "#0d366b"]  # blue ordinal ramp, None (lightest)
# to Very frequent (darkest), evenly spaced in OKLab lightness (2026-11-11; the old ramp's Very rare and Rare
# were too close to tell apart). The caption explains the ramp, so the panel has no legend.
N_LANDCOVER = 7
NUMERIC = [("slope", "Slope\n(degrees)"), ("elev_sd_r300", "Terrain variability\n(m)"),
           ("hand", "Height above\ndrainage (m)"), ("wetness", "Topographic\nwetness (index)"),
           ("log10_dist_river", "Distance to river\n(log$_{10}$ m)"),
           ("log10_dist_coast", "Distance to coast\n(log$_{10}$ m)")]
FF_NAMES = {0: "None", 1: "Very rare", 2: "Rare", 3: "Occasional", 4: "Frequent",
            5: "Very frequent"}
NLCD_NAMES = {11: "open water", 21: "developed open space", 22: "low-intensity developed",
              23: "medium-intensity developed", 24: "high-intensity developed",
              31: "barren land", 41: "deciduous forest", 42: "evergreen forest",
              43: "mixed forest", 52: "shrub and scrub", 71: "grassland", 81: "pasture and hay",
              82: "crops", 90: "woody wetland", 95: "herbaceous wetland"}


def figure_sets(df: pd.DataFrame, seed: int = 0, protocol: str = "band") -> dict:
    """{"Train", "ID test", "OOD"}: the seed's split, as zoo.cell_data builds it (Zone AE points of
    the training tiles; Zone AE and Zone A points of the test tiles)."""
    tr_tiles, te_tiles = zoo.split_tiles(df, seed)
    ae = zoo.subset(df, "AE", protocol)
    a = zoo.subset(df, "A", protocol)
    return {"Train": ae[ae[zoo.TILE].isin(tr_tiles)], "ID test": ae[ae[zoo.TILE].isin(te_tiles)],
            "OOD": a[a[zoo.TILE].isin(te_tiles)]}


def top_classes(values: pd.Series, k: int = N_LANDCOVER) -> list[int]:
    """The k most common classes of values (ties by the smaller code), most common first."""
    counts = values.dropna().astype(int).value_counts()
    order = sorted(counts.index, key=lambda c: (-counts[c], c))
    return [int(c) for c in order[:k]]


def class_shares(values: pd.Series, classes: list) -> list[float]:
    """The share of the non-missing values in each class; "other" takes every class not listed."""
    v = values.dropna()
    listed = [c for c in classes if c != "other"]
    out = []
    for c in classes:
        out.append(float((~v.isin(listed)).mean()) if c == "other" else float((v == c).mean()))
    return out


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def stacked(ax, groups: dict, col: str, classes: list, colors: list, label_of, min_label=0.07,
            dark=()):
    """One 100%-stacked bar per set; segments wider than min_label carry their label, dark text
    on the segments whose index is in dark."""
    names = list(groups)
    for row, name in enumerate(names):
        left = 0.0
        for i, (code, color, share) in enumerate(zip(classes, colors,
                                                     class_shares(groups[name][col], classes))):
            if share <= 0:
                continue
            ax.barh(row, share, left=left, height=0.62, color=color, edgecolor="white",
                    linewidth=0.6)
            if share >= min_label:
                ax.text(left + share / 2, row, label_of(code), ha="center", va="center",
                        fontsize=6, color=INK if i in dark else "white")
            left += share
    ax.set_yticks(range(len(names)), ["Train", "ID test", "OOD"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.tick_params(axis="y", length=0)
    tidy(ax)
    ax.spines["left"].set_visible(False)


def draw(groups: dict, landcover: list[int], out: Path, preview: Path | None = None) -> None:
    plt.rcParams.update({
        # The design's Liberation Serif, or a font with the same metrics (Times New Roman, the
        # paper's Times), so the labels fit as designed on any machine; DejaVu Serif is wider.
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.size": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.3, 2.75))
    gs = fig.add_gridspec(2, 20, height_ratios=[1.25, 1], hspace=0.75, wspace=2.2,
                          left=0.05, right=0.985, top=0.90, bottom=0.09)
    for i, (col, title) in enumerate(NUMERIC):
        ax = fig.add_subplot(gs[0, i * 20 // 6:(i + 1) * 20 // 6])
        data = [g[col].dropna().to_numpy() for g in groups.values()]
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.62, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.0),
                        whiskerprops=dict(color=INK2, linewidth=0.6),
                        capprops=dict(color=INK2, linewidth=0.6),
                        boxprops=dict(linewidth=0.5, edgecolor=INK2))
        for patch, (_, color) in zip(bp["boxes"], SETS):
            patch.set_facecolor(color)
        ax.set_xticks([1, 2, 3], ["Train", "ID", "OOD"])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_title(title, color=INK, pad=3)
        tidy(ax)

    lc = landcover + ["other"]
    ax = fig.add_subplot(gs[1, 0:9])
    stacked(ax, groups, "landcover", lc, CAT[:len(landcover)] + [MUTED],
            lambda c: "other" if c == "other" else str(c), dark=CAT_DARK_TEXT)
    ax.set_title("Land cover (NLCD class code)", color=INK, pad=3)

    ff = [0, 1, 2, 3, 4, 5]
    ax = fig.add_subplot(gs[1, 9:16])
    stacked(ax, groups, "soil_floodfreq", ff, ORD, lambda c: FF_NAMES[c], min_label=0.3,
            dark={0})
    ax.set_title("Soil flood frequency", color=INK, pad=3)
    ax.set_yticklabels([])

    ax = fig.add_subplot(gs[1, 16:20])
    stacked(ax, groups, "soil_fluvclass", [1, 0], ["#1c5cab", "#d9d8d3"], lambda c: "",
            min_label=2)
    for row, name in enumerate(groups):
        share = class_shares(groups[name].soil_fluvclass, [1])[0]
        ax.text(share + 0.03, row, f"fluvial {100 * share:.1f}%", va="center", fontsize=6,
                color=INK)
    ax.set_title("Fluvial class", color=INK, pad=3)
    ax.set_yticklabels([])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})   # same bytes for the same data
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=220)
    plt.close(fig)
