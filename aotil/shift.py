#!/usr/bin/env python3
"""The shift from Zone AE (ID) to Zone A (OOD): Table tab:shift and the paragraph "Aggregation hides the shift".

Every model of the zoo (N = 290). For accuracy, AUROC and F1 (flood class), Delta = ID minus OOD:
  pooled    each model's metric over all ID test points and over all OOD (Zone A test) points, averaged over the
            models (pooled AUROC is the zoo's own roc_auc_score, auc_id / auc_ood of the model table)
  per tile  each model's metric within each test tile, averaged over the models, then over the tiles where the
            metric is defined (accuracy: both sets present; F1: floods in both sets; AUROC: both classes in both
            sets), with 95% t-intervals over the tiles (each tile's value averaged over the models)
  test      two-sided Wilcoxon signed-rank test of the tiles' Delta against 0 (the tile is the unit: points in one
            tile are spatially correlated), Holm's correction across the three metrics
Also: the tiles where the models do worse on Zone A, the Palm Beach tile (its share of each test set and the
models' accuracy on its points), the per-tile flood shares, and corr_te on all Zone A test points (the 58 test
models, probit Pearson as subsets.probit_r, Fisher 95% interval).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subsets  # noqa: E402

PALM_BEACH_TILE = "30_10"          # the tile around Palm Beach (checked against its counties in shift())
PALM_BEACH_FIPS = "12099"
METRICS = ("acc", "auc", "f1")
NAMES = dict(acc="Accuracy", auc="AUROC", f1="F1")


def auc_rows(P, y):
    """ROC AUC of every model (row of P) on the given points, from average ranks (sklearn's
    roc_auc_score); NaN for every model when a class is missing."""
    pos = np.asarray(y) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.full(P.shape[0], np.nan)
    R = rankdata(P, axis=1)
    return (R[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def acc_f1_rows(Z, y):
    """Accuracy and flood-class F1 (2TP / (2TP + FP + FN)) of every model on the given points."""
    flood = np.asarray(y) == 1
    tp = (Z[:, flood]).sum(1)
    fn = (~Z[:, flood]).sum(1)
    fp = (~Z[:, ~flood]).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Z.mean(1), 2 * tp / (2 * tp + fp + fn)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p values."""
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for rank, k in enumerate(keys):
        running = max(running, min(1.0, (m - rank) * pvals[k]))
        out[k] = running
    return out


def t_interval(x, level=0.95):
    """mean +- t_{n-1} * sd / sqrt(n) over the tile values (NaN tiles left out)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, m = len(x), float(x.mean())
    h = float(t_dist.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m - h, m + h


def shift(z, models: pd.DataFrame, dataset: pd.DataFrame, county_names: dict | None = None) -> dict:
    """Table tab:shift and the paragraph's numbers.

    z: the cell's npz arrays (S_ood, S_id, y_ood, y_id, tile_50km, point_id_id); models: its model table;
    dataset: the point table indexed by point_id with tile_50km and meta_point_county. Returns
    dict(summary=..., per_tile=DataFrame, per_model=DataFrame)."""
    yo, yi = np.asarray(z["y_ood"]), np.asarray(z["y_id"])
    Zo = (z["S_ood"] > 0.5) | ((z["S_ood"] == 0.5) & (yo[None] == 0))
    Zi = (z["S_id"] > 0.5) | ((z["S_id"] == 0.5) & (yi[None] == 0))
    # p(flood) from the stored true-class scores (S = p if y = 1, else 1 - p); positives go through
    # 1 - (1 - S) like negatives, so two points with the same p rank as ties.
    Po = np.where(yo[None] == 1, 1 - (1 - z["S_ood"]), 1 - z["S_ood"])
    Pi = np.where(yi[None] == 1, 1 - (1 - z["S_id"]), 1 - z["S_id"])
    to = np.asarray(z["tile_50km"]).astype(str)
    ti = dataset.loc[np.asarray(z["point_id_id"]).astype(str), "tile_50km"].astype(str).values
    county_names = county_names or {}

    acc_i, f1_i = acc_f1_rows(Zi, yi)
    acc_o, f1_o = acc_f1_rows(Zo, yo)
    pooled = dict(acc=(float(acc_i.mean()), float(acc_o.mean())), f1=(float(f1_i.mean()), float(f1_o.mean())),
                  auc=(float(models.auc_id.mean()), float(models.auc_ood.mean())))

    tiles = sorted(set(to) | set(ti))
    per = {m: dict(ID=[], A=[]) for m in METRICS}
    rows = []
    for tile in tiles:
        mo, mi = to == tile, ti == tile
        both = mo.any() and mi.any()
        ao, fo = acc_f1_rows(Zo[:, mo], yo[mo]) if mo.any() else (None, None)
        ai, fi = acc_f1_rows(Zi[:, mi], yi[mi]) if mi.any() else (None, None)
        floods = both and (yo[mo] == 1).any() and (yi[mi] == 1).any()
        classes = both and len(set(yo[mo])) == 2 and len(set(yi[mi])) == 2
        values = {"acc": (both, ai, ao), "f1": (floods, fi, fo),
                  "auc": (classes, auc_rows(Pi[:, mi], yi[mi]) if classes else None,
                          auc_rows(Po[:, mo], yo[mo]) if classes else None)}
        row = dict(tile=tile, n_id=int(mi.sum()), n_ood=int(mo.sum()),
                   flood_id=float(yi[mi].mean()) if mi.any() else np.nan,
                   flood_ood=float(yo[mo].mean()) if mo.any() else np.nan)
        for m, (ok, vi, vo) in values.items():
            per[m]["ID"].append(vi if ok else np.full(Zo.shape[0], np.nan))
            per[m]["A"].append(vo if ok else np.full(Zo.shape[0], np.nan))
            row[f"{m}_id"] = float(np.mean(vi)) if ok else np.nan
            row[f"{m}_ood"] = float(np.mean(vo)) if ok else np.nan
            row[f"{m}_delta"] = float(np.mean(vi - vo)) if ok else np.nan
        pts = dataset.loc[np.concatenate([np.asarray(z["point_id"]).astype(str)[mo],
                                          np.asarray(z["point_id_id"]).astype(str)[mi]])]
        top = pts.meta_point_county.dropna().astype(str).value_counts()
        row["county_fips"] = top.index[0] if len(top) else ""
        row["county"] = county_names.get(row["county_fips"], row["county_fips"])
        rows.append(row)

    res = {}
    for m in METRICS:
        I = np.array(per[m]["ID"]).T                      # models x tiles
        A = np.array(per[m]["A"]).T
        keep = ~np.isnan(I).all(0)
        d_tile = np.nanmean(I[:, keep] - A[:, keep], axis=0)   # each tile's Delta, mean over models
        res[m] = dict(pooled_id=pooled[m][0], pooled_ood=pooled[m][1],
                      pooled_delta=pooled[m][0] - pooled[m][1],
                      tile_id=float(np.nanmean(I[:, keep])), tile_ood=float(np.nanmean(A[:, keep])),
                      tile_delta=float(d_tile.mean()), n_tiles=int(keep.sum()),
                      tiles_worse=int((d_tile > 0).sum()),
                      ci_id=t_interval(np.nanmean(I[:, keep], axis=0)),
                      ci_ood=t_interval(np.nanmean(A[:, keep], axis=0)),
                      ci_delta=t_interval(d_tile), wilcoxon_p=float(wilcoxon(d_tile).pvalue))
    p_holm = holm({m: res[m]["wilcoxon_p"] for m in METRICS})
    for m in METRICS:
        res[m]["holm_p"] = p_holm[m]

    per_tile = pd.DataFrame(rows)
    pb_id, pb_ood = ti == PALM_BEACH_TILE, to == PALM_BEACH_TILE
    pb_row = per_tile[per_tile.tile == PALM_BEACH_TILE]
    palm = dict(tile=PALM_BEACH_TILE, county_fips=pb_row.county_fips.iloc[0] if len(pb_row) else "",
                share_id=float(pb_id.mean()), share_ood=float(pb_ood.mean()),
                n_id=int(pb_id.sum()), n_ood=int(pb_ood.sum()),
                acc_id=float(Zi[:, pb_id].mean()), acc_ood=float(Zo[:, pb_ood].mean()))
    worst = per_tile.sort_values("acc_delta").head(3)
    floods = pd.concat([per_tile.flood_id, per_tile.flood_ood])

    # corr_te on all Zone A test points: the test models of the 60/20/20 split.
    Zf = Zo.astype(np.float64)
    acc_id_all = Zi.astype(np.float64).mean(1)
    _, _, test = subsets.model_split(Zo.shape[0])
    r_all = subsets.probit_r(Zf[test], acc_id_all[test], np.ones(Zo.shape[1]))
    summary = dict(metrics=res, palm_beach=palm, n_models=int(Zo.shape[0]), n_tiles=len(tiles),
                   corr_te_all=r_all, corr_te_all_ci=subsets.fisher_ci(r_all, len(test)), n_test_models=len(test),
                   tile_flood_share_range=(float(floods.min()), float(floods.max())),
                   most_negative_acc_tiles=worst[["tile", "county", "acc_delta", "flood_id", "flood_ood"]]
                   .to_dict("records"))
    per_model = pd.DataFrame(dict(idx=np.arange(Zo.shape[0]), kind=models.kind.values,
                                  acc_id=acc_i, acc_ood=acc_o, f1_id=f1_i, f1_ood=f1_o,
                                  auc_id=models.auc_id.values, auc_ood=models.auc_ood.values))
    return dict(summary=summary, per_tile=per_tile, per_model=per_model)
