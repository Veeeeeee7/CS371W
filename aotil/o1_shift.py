#!/usr/bin/env python3
"""
O1 (the shift) and O2 (aggregate accuracy-on-the-line) per cell, as defined in
docs/PROJECT-PLAN.md section 3.

O1, per scope (all points; land; paired land): each model's AUC on the held-out Zone AE half
(ID) and on the Zone A half (OOD) of the same test tiles. gap = ID AUC - Zone A AUC averaged
over the models (positive = worse on Zone A), with a 95% bootstrap interval over the cell's
50 km test tiles. With several tags, "pooled" rows average the cells and bootstrap the union
of their test tiles. O2: the paper's probit ID-vs-OOD correlation and its model-count rule.

Reads   <cells_dir>/<tag>.npz and <tag>_models.csv, and the dataset (tile, block, land cover)
Writes  <out_csv>, one row per (cell, scope)

    python aotil/o1_shift.py data/interim/aotil/cells_w7 \\
        data/processed_w7/dataset_FL.parquet FL_band_AE_s0,FL_band_AE_s1 out.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op      # noqa: E402
from report_paper import model_convergence  # noqa: E402

N_BOOT = 2000  # percentile intervals need about 1,000-2,000 draws (Carpenter and Bithell 2000)
# Bootstrap every k-th model (1 = all). The zoo fits its 8 families round-robin (model i is
# family i mod 8), so a stride sharing a factor with 8 skips families: 10, the first look's
# stride (docs/RUN-W7-REPORT-2026-10-16.md, B2), keeps only dt, hgb, mlp and rf.
MODEL_STRIDE = 1
NLCD_OPEN_WATER = 11  # NLCD land cover class code
SCOPE_NAMES = ("all", "land", "paired land")


def is_land(rows: pd.DataFrame) -> np.ndarray:
    """A point with an NLCD class other than open water; no class (e.g. offshore) is not land."""
    return (rows.landcover.notna() & (rows.landcover != NLCD_OPEN_WATER)).to_numpy()


def blocks_with_both_zones(points: pd.DataFrame) -> set:
    """10 km blocks (block_id) that hold positives of both zones."""
    positives = points[points.label == 1]
    n_zones = positives.groupby("block_id").domain.nunique()
    return set(n_zones[n_zones == 2].index)


def scopes(o: pd.DataFrame, i: pd.DataFrame, paired_blocks: set):
    """(name, Zone A mask, Zone AE mask) per scope, over the rows of the Zone A half o and
    the Zone AE half i. Paired land keeps land points in blocks with positives of both
    zones, so the two halves are compared on the same ground."""
    land_o = is_land(o)
    land_i = is_land(i)
    yield "all", np.ones(len(o), bool), np.ones(len(i), bool)
    yield "land", land_o, land_i
    paired_o = land_o & o.block_id.isin(paired_blocks).to_numpy()
    paired_i = land_i & i.block_id.isin(paired_blocks).to_numpy()
    yield "paired land", paired_o, paired_i


def o2(cells_dir: str, tag: str) -> dict:
    """The O2 columns, the same on every scope row of a cell.

    r_full_test       the paper's aggregate r (oodselect_paper.hard_corr, an eps-regularised
                      Pearson of probit ID vs probit OOD accuracy) on the test models of the
                      60/20/20 model split, with a Fisher 95% CI; report_paper's full-set row
    r_all_models      plain Pearson of the clamped probits over all models
    n_models_needed   report_paper.model_convergence: models needed before r moves < 1%
    Accuracy is binary correctness at the argmax (oodselect_paper.correctness), as the
    selection sees it.
    """
    _, Z_ood, acc_id, info = op.load_paper_cell(cells_dir, tag)
    _, _, test_models = op.model_split(info["n_models"])
    all_points = np.ones(info["d"], np.float32)
    r = op.hard_corr(Z_ood[test_models], acc_id[test_models], all_points, info["d"],
                     is_mask=True)
    lo, hi = op.fisher_ci(r, len(test_models))
    convergence = model_convergence(acc_id, Z_ood)
    probit_id = norm.ppf(np.clip(acc_id, op.EPS, 1 - op.EPS))
    probit_ood = norm.ppf(np.clip(Z_ood.mean(1), op.EPS, 1 - op.EPS))
    r_all = float(np.corrcoef(probit_id, probit_ood)[0, 1])
    return dict(r_all_models=r_all, r_full_test=r, r_full_test_ci_lo=lo, r_full_test_ci_hi=hi,
                n_test_models=len(test_models), n_models_needed=convergence.n_needed.iloc[0])


def rebuild_p(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p(flood) per model and point from the stored true-class scores (S = p if y = 1, else
    1 - p). Positives also go through 1 - (1 - S), so a positive and a negative that share a
    p still tie: trees predict few distinct p, and breaking those ties moves their AUCs by up
    to 0.1. The round trip keeps p only to about 1e-16 (float32 for xgb), so nb, xgb and some
    hgb AUCs ranked on it are off by up to about 0.002; per_model avoids this where it can."""
    return np.where(y[None] == 1, 1 - (1 - S), 1 - S)


def cell_arrays(cells_dir: str, tag: str, d: pd.DataFrame, stride: int = MODEL_STRIDE) -> dict:
    """What O1 needs from one cell, with "o" = Zone A (OOD) half and "i" = Zone AE (ID) half:
    labels (yo, yi), rebuilt p per model and point (Po, Pi), the points' rows of the dataset d
    (o, i) and their tiles (to, ti), the model count N, the zoo's stored AUCs, and the score
    groups of every stride-th model for the bootstrap (go, gi)."""
    z = np.load(f"{cells_dir}/{tag}.npz", allow_pickle=True)
    models = pd.read_csv(f"{cells_dir}/{tag}_models.csv")
    yo, yi = z["y_ood"], z["y_id"]
    Po = rebuild_p(z["S_ood"], yo)
    Pi = rebuild_p(z["S_id"], yi)
    o = d.loc[z["point_id"]]
    i = d.loc[z["point_id_id"]]
    return dict(yo=yo, yi=yi, Po=Po, Pi=Pi, o=o, i=i, to=o.tile_50km.to_numpy(),
                ti=i.tile_50km.to_numpy(), N=len(models),
                auc_id=models.auc_id.to_numpy(float), auc_ood=models.auc_ood.to_numpy(float),
                go=score_groups(Po[::stride], yo), gi=score_groups(Pi[::stride], yi))


def score_groups(P: np.ndarray, y: np.ndarray) -> dict:
    """The part of weighted_aucs that does not change between bootstrap draws. Per model (row
    of P), each point gets the rank of its score among the model's distinct scores, so tied
    points share a group id; model k's ids are offset by k * n so that all models share one
    id space of N * n slots. The ids are split into positives (gp) and negatives (gn)."""
    N, n = P.shape
    groups = np.empty((N, n), np.intp)
    for k in range(N):
        groups[k] = np.unique(P[k], return_inverse=True)[1] + k * n
    pos = y == 1
    return dict(N=N, n=n, pos=pos, gp=groups[:, pos].ravel(), gn=groups[:, ~pos].ravel())


def weighted_aucs(sg: dict, w: np.ndarray) -> np.ndarray:
    """ROC AUC of every model in sg under point weights w, equal to
    roc_auc_score(y, P[k], sample_weight=w) per model (ties count half, as in sklearn).

    With the points grouped by identical score in ascending order, and Pos_g / Neg_g the
    summed weights of group g's positives / negatives,
        AUC = sum_g Pos_g * (Neg_below_g + Neg_g / 2) / (W_pos * W_neg).
    Two bincounts and a cumulative sum do this for all models at once, instead of a sort per
    model and draw. NaN for every model when either class has no weight."""
    N, n, pos = sg["N"], sg["n"], sg["pos"]
    w_pos, w_neg = w[pos], w[~pos]
    pair_weight = w_pos.sum() * w_neg.sum()
    if pair_weight == 0:
        return np.full(N, np.nan)
    pos_per_group = np.bincount(sg["gp"], np.tile(w_pos, N), minlength=N * n).reshape(N, n)
    neg_per_group = np.bincount(sg["gn"], np.tile(w_neg, N), minlength=N * n).reshape(N, n)
    neg_below = np.cumsum(neg_per_group, axis=1) - neg_per_group
    return (pos_per_group * (neg_below + 0.5 * neg_per_group)).sum(axis=1) / pair_weight


def tile_weights(draw: np.ndarray, to: np.ndarray, ti: np.ndarray):
    """Integer point weights from one resample of tiles: each point counts as often as its
    tile was drawn, so points that share a tile stay together."""
    wo, wi = np.zeros(len(to)), np.zeros(len(ti))
    tiles, counts = np.unique(draw, return_counts=True)
    for tile, count in zip(tiles, counts):
        wo[to == tile] += count
        wi[ti == tile] += count
    return wo, wi


def boot_gap(c: dict, wo, wi) -> float:
    """The mean ID - Zone A AUC gap over the bootstrapped models (cell_arrays' stride) under
    point weights; NaN when a half has no weight on a class."""
    return float(np.mean(weighted_aucs(c["gi"], wi) - weighted_aucs(c["go"], wo)))


def per_model(c: dict, mo, mi, scope: str = ""):
    """Per-model ID and Zone A AUC within the masks, ranked on the rebuilt p. On the "all"
    scope, the zoo's stored AUCs, which were ranked on p itself (a hand-built cell without
    them is ranked on the rebuilt p)."""
    if scope == "all" and "auc_id" in c:
        return c["auc_id"], c["auc_ood"]
    y_id = c["yi"][mi]
    a_id = np.array([roc_auc_score(y_id, p[mi]) for p in c["Pi"]])
    y_ood = c["yo"][mo]
    a_od = np.array([roc_auc_score(y_ood, p[mo]) for p in c["Po"]])
    return a_id, a_od


def summary(a_id, a_od) -> dict:
    """Mean ID and Zone A AUC, mean gap, share of models with gap > 0 (worse), the same two
    over the quarter of models with the highest ID AUC (Q4_*), and the least-squares slope
    of Zone A AUC on ID AUC across models."""
    gap = a_id - a_od
    q4 = a_id >= np.quantile(a_id, .75)
    return dict(ID_AUC=a_id.mean(), A_AUC=a_od.mean(), gap=gap.mean(), worse=(gap > 0).mean(),
                Q4_gap=gap[q4].mean(), Q4_worse=(gap[q4] > 0).mean(),
                slope_A_on_ID=np.polyfit(a_id, a_od, 1)[0])


def o1(c: dict, tag: str, paired_blocks: set, n_boot: int, seed: int) -> list[dict]:
    """One row per scope for one cell. A bootstrap draw that leaves either half with no
    weight on a class (AUC undefined) is skipped; n_boot_used counts the rest."""
    test_tiles = np.unique(c["to"])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(test_tiles, len(test_tiles)) for _ in range(n_boot)]
    rows = []
    for scope, mo, mi in scopes(c["o"], c["i"], paired_blocks):
        s = summary(*per_model(c, mo, mi, scope))
        gaps = []
        for draw in draws:
            wo, wi = tile_weights(draw, c["to"], c["ti"])
            gap = boot_gap(c, wo * mo, wi * mi)
            if not np.isnan(gap):
                gaps.append(gap)
        rows.append(dict(cell=tag, scope=scope, n_A=int(mo.sum()), n_AE=int(mi.sum()),
                         ID_AUC=s["ID_AUC"], A_AUC=s["A_AUC"], gap=s["gap"],
                         ci_lo=np.percentile(gaps, 2.5), ci_hi=np.percentile(gaps, 97.5),
                         n_boot_used=len(gaps), worse=s["worse"], Q4_gap=s["Q4_gap"],
                         Q4_worse=s["Q4_worse"], slope_A_on_ID=s["slope_A_on_ID"]))
    return rows


def pooled_draw_gap(cells: list, scope_masks: list, draw: np.ndarray) -> float:
    """Mean over the cells of boot_gap under one tile draw, every cell weighted by the same
    draw; NaN as soon as one cell's gap is undefined."""
    gaps = []
    for c, (mo, mi) in zip(cells, scope_masks):
        wo, wi = tile_weights(draw, c["to"], c["ti"])
        gap = boot_gap(c, wo * mo, wi * mi)
        if np.isnan(gap):
            return np.nan
        gaps.append(gap)
    return np.mean(gaps)


def n_distinct_points(frames: list, masks: list) -> int:
    """Distinct point_ids among the masked rows of the frames."""
    ids = []
    for frame, mask in zip(frames, masks):
        ids.append(frame.index.to_numpy()[mask])
    return len(np.unique(np.concatenate(ids)))


def o1_pooled(cs: dict, paired_blocks: set, n_boot: int, seed: int) -> list[dict]:
    """One row per scope over several cells (cell = "pooled"); O2 columns stay empty.

    Point estimates are means over the cells (equal weights: the cells have equal model
    counts). The bootstrap resamples the union of the cells' test tiles and weights every
    cell by the same draw, because a tile can be a test tile in several seeds, so the seeds
    are not independent samples of ground. n_A and n_AE sum the cells' counts (a point in two
    cells counts twice: point-evaluations, not a sample size); n_A_distinct and n_AE_distinct
    count each point once; n_tiles_shared counts tiles that are test tiles in several cells."""
    tags = list(cs)
    cells = list(cs.values())
    cell_tiles = [np.unique(c["to"]) for c in cells]
    union, n_cells_per_tile = np.unique(np.concatenate(cell_tiles), return_counts=True)
    rng = np.random.default_rng(seed)
    draws = [rng.choice(union, len(union)) for _ in range(n_boot)]
    cell_scopes = [list(scopes(c["o"], c["i"], paired_blocks)) for c in cells]

    rows = []
    for j, scope in enumerate(SCOPE_NAMES):
        scope_masks = []                      # (Zone A mask, Zone AE mask) per cell
        for masks in cell_scopes:
            _, mo, mi = masks[j]
            scope_masks.append((mo, mi))
        per_cell = []
        for c, (mo, mi) in zip(cells, scope_masks):
            per_cell.append(summary(*per_model(c, mo, mi, scope)))
        gaps = []
        for draw in draws:
            gap = pooled_draw_gap(cells, scope_masks, draw)
            if not np.isnan(gap):
                gaps.append(gap)

        masks_o = [mo for mo, _ in scope_masks]
        masks_i = [mi for _, mi in scope_masks]
        row = dict(cell="pooled", scope=scope,
                   n_A=int(sum(mask.sum() for mask in masks_o)),
                   n_AE=int(sum(mask.sum() for mask in masks_i)))
        for key in per_cell[0]:
            row[key] = float(np.mean([stats[key] for stats in per_cell]))
        row.update(ci_lo=np.percentile(gaps, 2.5), ci_hi=np.percentile(gaps, 97.5),
                   n_boot_used=len(gaps), pooled_cells=";".join(tags), n_tiles_union=len(union),
                   n_A_distinct=n_distinct_points([c["o"] for c in cells], masks_o),
                   n_AE_distinct=n_distinct_points([c["i"] for c in cells], masks_i),
                   n_tiles_shared=int((n_cells_per_tile > 1).sum()))
        rows.append(row)
    return rows


def print_row(row: dict) -> None:
    """The row as a dict, floats rounded to 3 decimals."""
    shown = {}
    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 3)
        elif isinstance(value, np.integer):
            value = int(value)
        shown[key] = value
    print(shown, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells_dir")
    parser.add_argument("dataset")
    parser.add_argument("tags", help="comma-separated cell tags")
    parser.add_argument("out_csv")
    parser.add_argument("--boot", type=int, default=N_BOOT, help="tile-bootstrap draws")
    parser.add_argument("--model-stride", type=int, default=MODEL_STRIDE,
                        help="bootstrap over every k-th model (default 1, all models; "
                             "10 keeps only 4 of the 8 families)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args()

    points = pd.read_parquet(args.dataset).drop(columns=["geometry"], errors="ignore")
    points = points.set_index("point_id")
    paired_blocks = blocks_with_both_zones(points)

    rows = []
    cells = {}
    for tag in args.tags.split(","):
        cells[tag] = cell_arrays(args.cells_dir, tag, points, args.model_stride)
        o2_columns = o2(args.cells_dir, tag)
        for o1_row in o1(cells[tag], tag, paired_blocks, args.boot, args.seed):
            row = dict(o1_row, **o2_columns, cells_dir=args.cells_dir)
            rows.append(row)
            print_row(row)
    if len(cells) > 1:
        for pooled_row in o1_pooled(cells, paired_blocks, args.boot, args.seed):
            row = dict(pooled_row, cells_dir=args.cells_dir)
            rows.append(row)
            print_row(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
