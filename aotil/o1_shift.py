#!/usr/bin/env python3
"""
O1 (the shift) and O2 (aggregate accuracy-on-the-line) per cell, as defined in
docs/PROJECT-PLAN.md section 3.

O1, per scope (all points; land; paired land), in three metrics: accuracy (the primary one),
AUROC and F1 of the flood class. Each model is scored on the held-out Zone AE half (ID) and on
the Zone A half (OOD) of the same test tiles; gap = ID - Zone A averaged over the models
(positive = worse on Zone A), with a 95% bootstrap interval over the cell's 50 km test tiles.
All three metrics use the same tile draws. A model predicts flood iff p > 0.5, so a tie at 0.5
counts as dry: accuracy is the mean of oodselect_paper.correctness over a half's points, from
the stored scores, the rule the subset search uses. Two more outputs: the gaps of the 15
strongest test models (highest stored ID AUROC, as the twins' g picks them), and a prevalence
check, Zone A accuracy reweighted to the Zone AE flood share. With several tags, "pooled" rows
average the cells and bootstrap the union of their test tiles. O2: the paper's probit
ID-vs-OOD correlation and its model-count rule.

The per-tile gap (Victor's decision of 2026-10-24, docs/TODO-CLAUDE-CODE.md item 3). The two
halves cover the same test tiles in very different proportions, so the pooled gap mixes the shift
with where the points lie. Per tile, each gap is computed within the tile, with the tile's ID
points weighted to its Zone A share of positives (accuracy, F1; AUROC unweighted), and averaged
over the tiles, each tile counting once; the bootstrap draws the same tiles as above and averages
the drawn tiles' gaps. Also the unweighted ("raw") per-tile accuracy and F1. Two side tables per
run: one row per test tile, and one per model family.

Columns: the AUROC ones keep their names (ID_AUC, A_AUC, gap, ci_lo, ...); accuracy, F1, the
flood shares, the prevalence check, the strongest models and the per-tile gap follow them,
prefixed acc_, f1_, flood_share_, adjacc_, strong15_ and tile_. Pooled rows leave the tile_
columns empty.

Reads   <cells_dir>/<tag>.npz and <tag>_models.csv, and the dataset (tile, block, land cover,
        county); county names from data/interim/gis/counties_FL.parquet when it exists
Writes  <out_csv>, one row per (cell, scope); beside it <stem>_tiles.csv, one row per (cell,
        scope, test tile), and <stem>_families.csv, one row per (cell, scope, model family)

    python aotil/o1_shift.py data/interim/aotil/cells_w7 \\
        data/processed_w7/dataset_FL.parquet FL_band_AE_s0,FL_band_AE_s1 out.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
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
STRONGEST = 15        # the strongest test models, whose accuracy gap the twins' g measures
# Prefixes of the accuracy, F1, prevalence and per-tile columns; the CSV writes them after every
# AUROC one.
NEW_PREFIXES = ("acc_", "f1_", "flood_share_", "adjacc_", "strong15_", "tile_")
TP, FP, FN, TN = range(4)   # outcome codes of a (model, point) pair
# The per-tile metrics: accuracy and F1 with each tile's ID points weighted to the tile's Zone A
# share of positives, AUROC (which does not depend on that share), and unweighted accuracy and F1.
TILE_METRICS = ("acc", "f1", "auc", "acc_raw", "f1_raw")
COUNTIES = Path(__file__).resolve().parent.parent / "data/interim/gis/counties_FL.parquet"


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
    (o, i) and their tiles (to, ti), the model count N, the zoo's stored AUCs, the score
    groups of every stride-th model for the bootstrap (go, gi), and whether each model is right
    on each point (Zo, Zi: oodselect_paper.correctness of the stored scores, all models)."""
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
                go=score_groups(Po[::stride], yo), gi=score_groups(Pi[::stride], yi),
                Zo=op.correctness(z["S_ood"], yo)[0].astype(bool),
                Zi=op.correctness(z["S_id"], yi)[0].astype(bool))


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


def boot_aucs(c: dict, wo, wi) -> tuple[np.ndarray, np.ndarray]:
    """ID and Zone A AUC of each bootstrapped model (cell_arrays' stride) under point weights;
    NaN for every model when a half has no weight on a class."""
    return weighted_aucs(c["gi"], wi), weighted_aucs(c["go"], wo)


def boot_gap(c: dict, wo, wi) -> float:
    """The mean ID - Zone A AUC gap over the bootstrapped models (cell_arrays' stride) under
    point weights; NaN when a half has no weight on a class."""
    a_id, a_od = boot_aucs(c, wo, wi)
    return float(np.mean(a_id - a_od))


# ----------------------------------------------------------------------------- accuracy and F1
def outcome_counts(Z: np.ndarray, y: np.ndarray, tile_idx: np.ndarray, n_tiles: int,
                   mask: np.ndarray) -> np.ndarray:
    """(models, tiles, 4) counts of TP, FP, FN and TN among the points in mask.

    Z says whether each model is right on each point (oodselect_paper.correctness). A model
    predicts flood iff p > 0.5, so a flood point is a TP when the model is right and an FN when
    it is wrong, a dry point a TN when right and an FP when wrong; tile_idx is each point's tile.
    One bootstrap draw is then a weighted sum over the tiles, with no loop over points."""
    n_models = Z.shape[0]
    flood = np.asarray(y) == 1
    outcome = np.where(Z, np.where(flood, TP, TN), np.where(flood, FN, FP))
    keep = np.flatnonzero(mask)
    slot = (np.arange(n_models)[:, None] * n_tiles + tile_idx[keep]) * 4 + outcome[:, keep]
    counts = np.bincount(slot.ravel(), minlength=n_models * n_tiles * 4)
    return counts.reshape(n_models, n_tiles, 4).astype(float)


def rates(totals: np.ndarray) -> dict:
    """Per-model accuracy, F1 of the flood class (2TP / (2TP + FP + FN)), true positive and
    true negative rates, and the flood share, from (models, 4) counts of TP, FP, FN, TN."""
    tp, fp, fn, tn = totals[:, TP], totals[:, FP], totals[:, FN], totals[:, TN]
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(acc=(tp + tn) / n, f1=2 * tp / (2 * tp + fp + fn), tpr=tp / (tp + fn),
                    tnr=tn / (tn + fp), flood=(tp + fn) / n)


def scope_counts(c: dict, mo, mi) -> dict:
    """The outcome counts of both halves within one scope, per tile of the cell."""
    tiles = np.unique(np.concatenate([c["to"], c["ti"]]))
    return dict(tiles=tiles,
                o=outcome_counts(c["Zo"], c["yo"], np.searchsorted(tiles, c["to"]), len(tiles), mo),
                i=outcome_counts(c["Zi"], c["yi"], np.searchsorted(tiles, c["ti"]), len(tiles), mi))


def tile_vector(draw: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """How often each of tiles was drawn; drawn tiles not in tiles are ignored."""
    at = np.searchsorted(tiles, draw)
    hit = at < len(tiles)
    hit[hit] = tiles[at[hit]] == draw[hit]
    return np.bincount(at[hit], minlength=len(tiles)).astype(float)


def count_metrics(counts: dict, w: np.ndarray) -> dict:
    """Per-model (ID, Zone A) accuracy, F1 and accuracy reweighted to the Zone AE flood share,
    pi_AE * TPR_A + (1 - pi_AE) * TNR_A (whose ID side is plain accuracy), and the two halves'
    flood shares, with every tile counted w[t] times."""
    o = rates(np.einsum("mtk,t->mk", counts["o"], w))
    i = rates(np.einsum("mtk,t->mk", counts["i"], w))
    share_ae = i["flood"]
    return dict(acc=(i["acc"], o["acc"]), f1=(i["f1"], o["f1"]),
                adjacc=(i["acc"], share_ae * o["tpr"] + (1 - share_ae) * o["tnr"]),
                flood_share_AE=float(i["flood"][0]), flood_share_A=float(o["flood"][0]))


def strongest_models(c: dict) -> np.ndarray | None:
    """The STRONGEST test models of the 60/20/20 model split with the highest stored ID AUROC,
    picked as results/2026-10-19/paper_figures/region_twins_ref.py picks them; None for a cell
    without the stored AUCs or with too few models (under 3) for the split."""
    if "auc_id" not in c or c["N"] < 3:
        return None
    _, _, test = op.model_split(c["N"])
    return test[np.argsort(-c["auc_id"][test])[:STRONGEST]]


def draw_gaps(values: dict, strong, aucs) -> dict:
    """One bootstrap draw's mean gaps: accuracy, F1 and reweighted accuracy over all models, and
    for the strongest models (when strong is given) accuracy, F1 and, when aucs (per-model ID
    and Zone A AUC of every model) is given, AUROC."""
    out = {}
    for metric in ("acc", "f1", "adjacc"):
        on_id, on_a = values[metric]
        out[metric] = float(np.mean(on_id - on_a))
    if strong is not None:
        for metric in ("acc", "f1"):
            on_id, on_a = values[metric]
            out[f"strong15_{metric}"] = float(np.mean(on_id[strong] - on_a[strong]))
        if aucs is not None:
            out["strong15_auc"] = float(np.mean(aucs[0][strong] - aucs[1][strong]))
    return out


def interval(name: str, boot: list[dict]) -> dict:
    """<name>_ci_lo and <name>_ci_hi: the 95% percentile interval of the draws' <name> gap."""
    values = [b[name] for b in boot if name in b]
    if not values:
        return {f"{name}_ci_lo": np.nan, f"{name}_ci_hi": np.nan}
    return {f"{name}_ci_lo": np.percentile(values, 2.5),
            f"{name}_ci_hi": np.percentile(values, 97.5)}


def metric_point_columns(values: dict, strong, a_id, a_od) -> dict:
    """The point estimates of the accuracy, F1, flood-share, prevalence-check and strongest-model
    columns; a_id and a_od are the per-model AUROCs of the row (the strongest-model AUROC)."""
    cols = {}
    for metric in ("acc", "f1"):
        s = summary(*values[metric])
        cols.update({f"{metric}_ID": s["ID_AUC"], f"{metric}_A": s["A_AUC"],
                     f"{metric}_gap": s["gap"], f"{metric}_worse": s["worse"],
                     f"{metric}_Q4_gap": s["Q4_gap"], f"{metric}_Q4_worse": s["Q4_worse"],
                     f"{metric}_slope_A_on_ID": s["slope_A_on_ID"]})
    acc_id, adjacc_a = values["adjacc"]
    cols.update(flood_share_A=values["flood_share_A"], flood_share_AE=values["flood_share_AE"],
                adjacc_A=float(adjacc_a.mean()), adjacc_gap=float(np.mean(acc_id - adjacc_a)))
    pairs = {"acc": values["acc"], "f1": values["f1"], "auc": (a_id, a_od)}
    for metric, (on_id, on_a) in pairs.items():
        if strong is None:
            point = (np.nan, np.nan, np.nan)
        else:
            point = (on_id[strong].mean(), on_a[strong].mean(),
                     np.mean(on_id[strong] - on_a[strong]))
        cols.update({f"strong15_{metric}_ID": point[0], f"strong15_{metric}_A": point[1],
                     f"strong15_{metric}_gap": point[2]})
    return cols


def metric_columns(point: dict, boot: list[dict]) -> dict:
    """The new columns of one row, in CSV order: point estimates with their intervals."""
    cols = {}
    for metric in ("acc", "f1"):
        for key in ("ID", "A", "gap"):
            cols[f"{metric}_{key}"] = point[f"{metric}_{key}"]
        cols.update(interval(metric, boot))
        for key in ("worse", "Q4_gap", "Q4_worse", "slope_A_on_ID"):
            cols[f"{metric}_{key}"] = point[f"{metric}_{key}"]
    for key in ("flood_share_A", "flood_share_AE", "adjacc_A", "adjacc_gap"):
        cols[key] = point[key]
    cols.update(interval("adjacc", boot))
    for metric in ("acc", "f1", "auc"):
        for key in ("ID", "A", "gap"):
            cols[f"strong15_{metric}_{key}"] = point[f"strong15_{metric}_{key}"]
        cols.update(interval(f"strong15_{metric}", boot))
    return cols


# ----------------------------------------------------------------------------- per-tile gap
def tile_draws(to: np.ndarray, n_boot: int, seed: int) -> list:
    """n_boot resamples, with replacement, of the cell's test tiles (those of the Zone A half,
    to): the draws of every bootstrap in o1, pooled and per tile alike."""
    test_tiles = np.unique(to)
    rng = np.random.default_rng(seed)
    return [rng.choice(test_tiles, len(test_tiles)) for _ in range(n_boot)]


def draw_matrix(draws: list, tiles: np.ndarray) -> np.ndarray:
    """(draws, tiles): how often each draw holds each of tiles (tile_vector per draw)."""
    return np.array([tile_vector(draw, tiles) for draw in draws]).reshape(len(draws), len(tiles))


def tile_aucs(P: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ROC AUC of every model (row of P) on the points in mask, ranked on P as per_model ranks
    the rebuilt p (weighted_aucs with unit weights, sklearn's roc_auc_score to 1e-12); NaN for
    every model when the points lack a class."""
    keep = np.flatnonzero(mask)
    return weighted_aucs(score_groups(P[:, keep], y[keep]), np.ones(len(keep)))


def tile_acc_f1(k: np.ndarray, w_pos=1.0, w_neg=1.0) -> tuple[np.ndarray, np.ndarray]:
    """(models, tiles) accuracy and F1 of the flood class from (models, tiles, 4) outcome counts,
    each positive weighted by w_pos and each negative by w_neg (per tile)."""
    tp, fp, fn, tn = k[..., TP], k[..., FP], k[..., FN], k[..., TN]
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = (w_pos * tp + w_neg * tn) / (w_pos * (tp + fn) + w_neg * (fp + tn))
        f1 = 2 * w_pos * tp / (2 * w_pos * tp + w_neg * fp + w_pos * fn)
    return acc, f1


def tile_gaps(c: dict, mo, mi) -> dict:
    """The per-tile metrics of one scope. Per test tile (one with a point of either half in the
    masks): the halves' point counts (n_ID, n_A) and flood shares, the models' mean accuracy on
    both halves together (acc_both), and per metric of TILE_METRICS the [tile x model] matrices
    of the ID value, the Zone A value and their gap, NaN on the tiles where ok[metric] is False.
    acc_pooled holds each model's (ID, Zone A) accuracy over all the scope's points.

    Within tile t, with rho a half's share of positives in t, the weighted metrics give each ID
    positive the weight rho_A / rho_ID and each ID negative (1 - rho_A) / (1 - rho_ID), so the
    ID half has the Zone A half's share; Zone A points weigh 1. They are undefined when Zone A
    holds a class the ID half lacks. Every metric needs points in both halves, F1 positives in
    both, and AUROC (unweighted) both classes in both."""
    counts = scope_counts(c, mo, mi)
    o, i = counts["o"], counts["i"]
    present = o[0].sum(1) + i[0].sum(1) > 0
    tiles, o, i = counts["tiles"][present], o[:, present], i[:, present]
    pos_o, neg_o = o[0, :, TP] + o[0, :, FN], o[0, :, FP] + o[0, :, TN]
    pos_i, neg_i = i[0, :, TP] + i[0, :, FN], i[0, :, FP] + i[0, :, TN]
    n_o, n_i = pos_o + neg_o, pos_i + neg_i
    both = (n_o > 0) & (n_i > 0)
    weighted = both & ((pos_o == 0) | (pos_i > 0)) & ((neg_o == 0) | (neg_i > 0))
    ok = dict(acc=weighted, f1=weighted & (pos_o > 0),
              auc=(pos_o > 0) & (neg_o > 0) & (pos_i > 0) & (neg_i > 0),
              acc_raw=both, f1_raw=both & (pos_o > 0) & (pos_i > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_o, rho_i = pos_o / n_o, pos_i / n_i
        w_pos = np.where(pos_i > 0, rho_o / rho_i, 0.0)
        w_neg = np.where(neg_i > 0, (neg_o / n_o) / (neg_i / n_i), 0.0)
        acc_both = ((o[..., TP] + o[..., TN] + i[..., TP] + i[..., TN]) / (n_o + n_i)).mean(0)
    acc_i, f1_i = tile_acc_f1(i, w_pos, w_neg)
    acc_i_raw, f1_i_raw = tile_acc_f1(i)
    acc_o, f1_o = tile_acc_f1(o)
    auc_i, auc_o = np.full(acc_o.shape, np.nan), np.full(acc_o.shape, np.nan)
    for k in np.flatnonzero(ok["auc"]):
        auc_i[:, k] = tile_aucs(c["Pi"], c["yi"], mi & (c["ti"] == tiles[k]))
        auc_o[:, k] = tile_aucs(c["Po"], c["yo"], mo & (c["to"] == tiles[k]))
    on_id = dict(acc=acc_i, f1=f1_i, auc=auc_i, acc_raw=acc_i_raw, f1_raw=f1_i_raw)
    on_a = dict(acc=acc_o, f1=f1_o, auc=auc_o, acc_raw=acc_o, f1_raw=f1_o)
    t = dict(tiles=tiles, n_ID=n_i, n_A=n_o, flood_share_ID=rho_i, flood_share_A=rho_o,
             acc_both=acc_both, ok=ok, ID={}, A={}, gap={},
             acc_pooled=(rates(i.sum(1))["acc"], rates(o.sum(1))["acc"]))
    for metric in TILE_METRICS:
        undefined = ~ok[metric][:, None]
        t["ID"][metric] = np.where(undefined, np.nan, on_id[metric].T)
        t["A"][metric] = np.where(undefined, np.nan, on_a[metric].T)
        t["gap"][metric] = t["ID"][metric] - t["A"][metric]
    return t


def signed_rank_p(values: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p of values against 0, scipy's defaults; NaN without
    values or when scipy refuses them."""
    if len(values) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(values).pvalue)
        except ValueError:
            return np.nan


def tile_summary(t: dict, metric: str, K: np.ndarray, models: np.ndarray) -> dict:
    """One metric's per-tile columns over the given models (indices) of tile_gaps' t, K being
    the draw_matrix of the bootstrap draws on t's tiles:
        ID, A        the mean over tiles and models of each half's metric
        gap          the mean over models of each model's mean gap over the tiles
        ci_lo, ci_hi 95% percentile interval of the draws; a draw averages each model's gaps on
                     the drawn tiles, a tile drawn twice counting twice (NaN draws, which hold
                     only tiles where the metric is undefined, are left out)
        worse        share of models whose mean gap is > 0
        tiles_pos    tiles whose mean gap over the models is > 0
        median       median of those tile gaps
        wilcoxon_p   signed-rank p of those tile gaps against 0
        n_tiles      tiles used; n_dropped tiles where the metric is undefined"""
    ok = t["ok"][metric]
    out = dict(ID=np.nan, A=np.nan, gap=np.nan, ci_lo=np.nan, ci_hi=np.nan, worse=np.nan,
               tiles_pos=0, median=np.nan, wilcoxon_p=np.nan, n_tiles=int(ok.sum()),
               n_dropped=int((~ok).sum()))
    if not ok.any():
        return out
    G = t["gap"][metric][ok][:, models]               # tiles used x models
    per_tile, per_model = G.mean(1), G.mean(0)
    k = K[:, ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = ((k @ G) / k.sum(1, keepdims=True)).mean(1)
    boot = boot[np.isfinite(boot)]
    if len(boot):
        out.update(ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5))
    out.update(ID=float(t["ID"][metric][ok][:, models].mean()),
               A=float(t["A"][metric][ok][:, models].mean()), gap=float(per_model.mean()),
               worse=float((per_model > 0).mean()), tiles_pos=int((per_tile > 0).sum()),
               median=float(np.median(per_tile)), wilcoxon_p=signed_rank_p(per_tile))
    return out


def tile_columns(t: dict, K: np.ndarray, strong) -> dict:
    """The tile_ columns of one row: tile_n_tiles, tile_<metric>_<key> (tile_summary) over all
    models, then tile_strong15_<metric>_<key> over the strongest models (NaN when strong is
    None; their n_tiles and n_dropped are those of all models, so not repeated)."""
    everyone = np.arange(t["gap"]["acc"].shape[1])
    cols = {"tile_n_tiles": len(t["tiles"])}
    for metric in TILE_METRICS:
        for key, value in tile_summary(t, metric, K, everyone).items():
            cols[f"tile_{metric}_{key}"] = value
    for metric in TILE_METRICS:
        s = tile_summary(t, metric, K, everyone if strong is None else strong)
        for key, value in s.items():
            if key not in ("n_tiles", "n_dropped"):
                cols[f"tile_strong15_{metric}_{key}"] = np.nan if strong is None else value
    return cols


def county_names(path: Path = COUNTIES) -> dict:
    """County FIPS code -> name, from the counties layer; empty when it is missing."""
    if not Path(path).exists():
        return {}
    counties = pd.read_parquet(path, columns=["GEOID", "NAME"])
    return dict(zip(counties.GEOID.astype(str), counties.NAME.astype(str)))


def dominant_counties(c: dict, mo, mi, tiles: np.ndarray) -> list:
    """The most common county (FIPS code) among each tile's points of both halves in the masks,
    ties to the lower code; "" without county columns. A point's county is the one it lies in
    (meta_point_county), else its DFIRM's (county_fips), as regions.point_county has it."""
    rows = pd.concat([c["o"][mo], c["i"][mi]])
    county = np.full(len(rows), None, dtype=object)
    for column in ("county_fips", "meta_point_county"):      # the later one wins where known
        if column in rows:
            known = rows[column].notna().to_numpy()
            county[known] = rows[column].astype(str).to_numpy()[known]
    tile_of = np.concatenate([c["to"][mo], c["ti"][mi]])
    out = []
    for tile in tiles:
        n = pd.Series(county[tile_of == tile]).dropna().value_counts()
        out.append(min(n.index, key=lambda code: (-n[code], code)) if len(n) else "")
    return out


def tile_table(c: dict, t: dict, mo, mi, tag: str, scope: str, names: dict) -> pd.DataFrame:
    """One row per test tile of tile_gaps' t: point counts and flood shares of the two halves,
    the models' mean accuracy on both halves together, the dominant county, and per metric the
    mean over models of the ID value, the Zone A value and the gap (NaN where undefined)."""
    fips = dominant_counties(c, mo, mi, t["tiles"])
    table = pd.DataFrame(dict(cell=tag, scope=scope, tile=t["tiles"], county_fips=fips,
                              county=[names.get(code, "") for code in fips],
                              n_ID=t["n_ID"].astype(int), n_A=t["n_A"].astype(int),
                              flood_share_ID=t["flood_share_ID"],
                              flood_share_A=t["flood_share_A"], acc_both=t["acc_both"]))
    for metric in TILE_METRICS:
        for key in ("ID", "A", "gap"):
            table[f"{metric}_{key}"] = t[key][metric].mean(1)
    return table


def family_table(t: dict, K: np.ndarray, kinds: np.ndarray, tag: str, scope: str) -> pd.DataFrame:
    """One row per model family (kinds: each model's family): its model count, its mean pooled
    accuracy on the two halves and their gap, its tile_acc columns (tile_summary over its
    models) and its raw per-tile accuracy gap."""
    acc_id, acc_a = t["acc_pooled"]
    rows = []
    for family in sorted(set(kinds)):
        models = np.flatnonzero(kinds == family)
        s = tile_summary(t, "acc", K, models)
        row = dict(cell=tag, scope=scope, family=family, n_models=len(models),
                   acc_ID=acc_id[models].mean(), acc_A=acc_a[models].mean(),
                   acc_gap=np.mean(acc_id[models] - acc_a[models]))
        for key in ("ID", "A", "gap", "ci_lo", "ci_hi", "worse", "tiles_pos"):
            row[f"tile_acc_{key}"] = s[key]
        row["tile_acc_raw_gap"] = tile_summary(t, "acc_raw", K, models)["gap"]
        rows.append(row)
    return pd.DataFrame(rows)


def tile_tables(c: dict, tag: str, paired_blocks: set, n_boot: int, seed: int,
                kinds: np.ndarray, names: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-tile and per-family tables of one cell, a block per scope, with o1's draws."""
    draws = tile_draws(c["to"], n_boot, seed)
    tiles, families = [], []
    for scope, mo, mi in scopes(c["o"], c["i"], paired_blocks):
        t = tile_gaps(c, mo, mi)
        tiles.append(tile_table(c, t, mo, mi, tag, scope, names))
        families.append(family_table(t, draw_matrix(draws, t["tiles"]), kinds, tag, scope))
    return pd.concat(tiles, ignore_index=True), pd.concat(families, ignore_index=True)


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
    weight on a class (AUC undefined) is skipped for every metric; n_boot_used counts the rest.
    The strongest-model columns are filled on the "all" scope only. The per-tile columns
    (tile_columns) use every draw. A cell without stored scores (no Zo, Zi: a hand-built one)
    gets the AUROC columns only."""
    draws = tile_draws(c["to"], n_boot, seed)
    counted = "Zo" in c
    every_model_bootstrapped = c["gi"]["N"] == c["N"]
    rows = []
    for scope, mo, mi in scopes(c["o"], c["i"], paired_blocks):
        a_id, a_od = per_model(c, mo, mi, scope)
        s = summary(a_id, a_od)
        if counted:
            counts = scope_counts(c, mo, mi)
            strong = strongest_models(c) if scope == "all" else None
        gaps, boot = [], []
        for draw in draws:
            wo, wi = tile_weights(draw, c["to"], c["ti"])
            aucs = boot_aucs(c, wo * mo, wi * mi)
            gap = float(np.mean(aucs[0] - aucs[1]))      # boot_gap
            if np.isnan(gap):
                continue
            gaps.append(gap)
            if counted:
                values = count_metrics(counts, tile_vector(draw, counts["tiles"]))
                boot.append(draw_gaps(values, strong, aucs if every_model_bootstrapped else None))
        row = dict(cell=tag, scope=scope, n_A=int(mo.sum()), n_AE=int(mi.sum()),
                   ID_AUC=s["ID_AUC"], A_AUC=s["A_AUC"], gap=s["gap"],
                   ci_lo=np.percentile(gaps, 2.5), ci_hi=np.percentile(gaps, 97.5),
                   n_boot_used=len(gaps), worse=s["worse"], Q4_gap=s["Q4_gap"],
                   Q4_worse=s["Q4_worse"], slope_A_on_ID=s["slope_A_on_ID"])
        if counted:
            values = count_metrics(counts, np.ones(len(counts["tiles"])))
            row.update(metric_columns(metric_point_columns(values, strong, a_id, a_od), boot))
            t = tile_gaps(c, mo, mi)
            row.update(tile_columns(t, draw_matrix(draws, t["tiles"]), strong))
        rows.append(row)
    return rows


def pooled_draw(cells: list, scope_masks: list, draw: np.ndarray):
    """(mean over the cells of boot_gap, each cell's boot_aucs) under one tile draw, every cell
    weighted by the same draw; (NaN, None) as soon as one cell's gap is undefined."""
    gaps, aucs = [], []
    for c, (mo, mi) in zip(cells, scope_masks):
        wo, wi = tile_weights(draw, c["to"], c["ti"])
        cell_aucs = boot_aucs(c, wo * mo, wi * mi)
        gap = float(np.mean(cell_aucs[0] - cell_aucs[1]))      # boot_gap
        if np.isnan(gap):
            return np.nan, None
        gaps.append(gap)
        aucs.append(cell_aucs)
    return np.mean(gaps), aucs


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
    count each point once; n_tiles_shared counts tiles that are test tiles in several cells.
    Accuracy, F1, the prevalence check and the strongest models are pooled the same way, each
    draw's gap being the mean of the cells' gaps; they need every cell's stored scores."""
    tags = list(cs)
    cells = list(cs.values())
    cell_tiles = [np.unique(c["to"]) for c in cells]
    union, n_cells_per_tile = np.unique(np.concatenate(cell_tiles), return_counts=True)
    rng = np.random.default_rng(seed)
    draws = [rng.choice(union, len(union)) for _ in range(n_boot)]
    cell_scopes = [list(scopes(c["o"], c["i"], paired_blocks)) for c in cells]
    counted = all("Zo" in c for c in cells)
    every_model_bootstrapped = all(c["gi"]["N"] == c["N"] for c in cells)

    rows = []
    for j, scope in enumerate(SCOPE_NAMES):
        scope_masks = []                      # (Zone A mask, Zone AE mask) per cell
        for masks in cell_scopes:
            _, mo, mi = masks[j]
            scope_masks.append((mo, mi))
        per_cell, per_cell_aucs = [], []
        for c, (mo, mi) in zip(cells, scope_masks):
            a_id, a_od = per_model(c, mo, mi, scope)
            per_cell.append(summary(a_id, a_od))
            per_cell_aucs.append((a_id, a_od))
        if counted:
            counts = [scope_counts(c, mo, mi) for c, (mo, mi) in zip(cells, scope_masks)]
            strong = [strongest_models(c) if scope == "all" else None for c in cells]
        gaps, boot = [], []
        for draw in draws:
            gap, cell_aucs = pooled_draw(cells, scope_masks, draw)
            if np.isnan(gap):
                continue
            gaps.append(gap)
            if counted:
                per_draw = []
                for k, s, aucs in zip(counts, strong, cell_aucs):
                    values = count_metrics(k, tile_vector(draw, k["tiles"]))
                    per_draw.append(draw_gaps(values, s, aucs if every_model_bootstrapped
                                              else None))
                boot.append({key: float(np.mean([d[key] for d in per_draw]))
                             for key in per_draw[0]})

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
        if counted:
            points = []
            for k, s, (a_id, a_od) in zip(counts, strong, per_cell_aucs):
                values = count_metrics(k, np.ones(len(k["tiles"])))
                points.append(metric_point_columns(values, s, a_id, a_od))
            point = {key: float(np.mean([p[key] for p in points])) for key in points[0]}
            row.update(metric_columns(point, boot))
        rows.append(row)
    return rows


def accuracy_first(row: dict) -> dict:
    """The row with the accuracy columns (acc_, adjacc_) right after cell and scope."""
    lead = {key: row[key] for key in ("cell", "scope") if key in row}
    accuracy = {key: value for key, value in row.items() if key.startswith(("acc_", "adjacc_"))}
    rest = {key: value for key, value in row.items() if key not in lead and key not in accuracy}
    return {**lead, **accuracy, **rest}


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
    names = county_names()
    tile_frames, family_frames = [], []
    for tag in args.tags.split(","):
        cells[tag] = cell_arrays(args.cells_dir, tag, points, args.model_stride)
        o2_columns = o2(args.cells_dir, tag)
        for o1_row in o1(cells[tag], tag, paired_blocks, args.boot, args.seed):
            row = dict(o1_row, **o2_columns, cells_dir=args.cells_dir)
            rows.append(row)
            print_row(accuracy_first(row))
        kinds = pd.read_csv(f"{args.cells_dir}/{tag}_models.csv").kind.to_numpy()
        tiles, families = tile_tables(cells[tag], tag, paired_blocks, args.boot, args.seed,
                                      kinds, names)
        tile_frames.append(tiles)
        family_frames.append(families)
    if len(cells) > 1:
        for pooled_row in o1_pooled(cells, paired_blocks, args.boot, args.seed):
            row = dict(pooled_row, cells_dir=args.cells_dir)
            rows.append(row)
            print_row(accuracy_first(row))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    added = [column for column in table.columns if column.startswith(NEW_PREFIXES)]
    table = table[[column for column in table.columns if column not in added] + added]
    table.round(4).to_csv(out, index=False)
    print(f"wrote {out}")
    for suffix, frames in (("tiles", tile_frames), ("families", family_frames)):
        side = out.with_name(f"{out.stem}_{suffix}.csv")
        pd.concat(frames, ignore_index=True).round(4).to_csv(side, index=False)
        print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
