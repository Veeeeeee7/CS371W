#!/usr/bin/env python3
"""Checks for shift.py (Table tab:shift). Run from the repo root: .venv/bin/python aotil/test_shift.py

1. acc_f1_rows on a hand count; auc_rows against sklearn's roc_auc_score (ties included), NaN without a class.
2. holm against a hand computation; t_interval against scipy's t.interval.
3. shift() on hand-made cells:
   a. two tiles whose ID and OOD halves are identical inside each tile, in different proportions: per-tile Delta = 0
      for every metric, pooled Delta != 0 (the paper's "Aggregation hides the shift" in miniature);
   b. a tile where one set has a single class: left out of AUROC (and counted), kept for accuracy;
   c. the Palm Beach numbers, the tile table and corr_te on all OOD points.
4. The real cell (skipped when data/interim/aotil/cells_v4_2026-11-07 or the dataset is missing): the paper's Table 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shift  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def check_rows():
    print("1. per-model metrics")
    Z = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], bool)
    y = np.array([1, 1, 0, 0])
    acc, f1 = shift.acc_f1_rows(Z, y)
    # model 0: TP 2, FN 0, FP 1 (the wrong negative) -> F1 = 4 / 5; model 1: TP 1, FN 1, FP 0 -> 2 / 3
    check(np.allclose(acc, [0.75, 0.75]) and np.allclose(f1, [0.8, 2 / 3]), "accuracy and F1 by hand")
    rng = np.random.default_rng(0)
    P = np.round(rng.random((5, 60)), 1)                 # many ties
    yy = rng.integers(0, 2, 60)
    ours = shift.auc_rows(P, yy)
    ref = np.array([roc_auc_score(yy, p) for p in P])
    check(np.allclose(ours, ref), "AUROC = sklearn's, ties included")
    check(np.isnan(shift.auc_rows(P, np.ones(60, int))).all(), "AUROC NaN when a class is missing")


def check_stats():
    print("2. Holm and t-intervals")
    h = shift.holm(dict(a=0.01, b=0.04, c=0.03))
    check(np.isclose(h["a"], 0.03) and np.isclose(h["c"], 0.06) and np.isclose(h["b"], 0.06),
          f"Holm: 3 x 0.01, max(2 x 0.03, .), max(1 x 0.04, 0.06) -> {h}")
    x = np.array([0.1, -0.05, 0.2, 0.03, np.nan, 0.08])
    lo, hi = shift.t_interval(x)
    v = x[~np.isnan(x)]
    ref = t_dist.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / np.sqrt(len(v)))
    check(np.allclose((lo, hi), ref), "t-interval over the non-NaN tiles = scipy's")


def cell_from(tiles_o, y_o, ok_o, tiles_i, y_i, ok_i, n_models=4):
    """A cell dict from per-point tiles, labels and correctness (same for every model unless 2-D)."""
    ok_o = np.broadcast_to(ok_o, (n_models, len(y_o)))
    ok_i = np.broadcast_to(ok_i, (n_models, len(y_i)))
    S_o = np.where(ok_o, 0.8, 0.2)
    S_i = np.where(ok_i, 0.8, 0.2)
    z = dict(S_ood=S_o, S_id=S_i, y_ood=np.asarray(y_o), y_id=np.asarray(y_i), tile_50km=np.asarray(tiles_o),
             point_id=np.array([f"O{i}" for i in range(len(y_o))]),
             point_id_id=np.array([f"I{i}" for i in range(len(y_i))]))
    ds = pd.DataFrame(dict(point_id=list(z["point_id"]) + list(z["point_id_id"]),
                           tile_50km=list(tiles_o) + list(tiles_i),
                           meta_point_county=["12099" if t == shift.PALM_BEACH_TILE else "12001"
                                              for t in list(tiles_o) + list(tiles_i)])).set_index("point_id")
    models = pd.DataFrame(dict(kind=["rf"] * n_models, auc_id=np.linspace(0.6, 0.9, n_models),
                               auc_ood=np.linspace(0.6, 0.9, n_models)))
    return z, models, ds


def check_shift():
    print("3. shift() on hand-made cells")
    # a. tile X: every model right on everything; tile Y: right on half. ID has 10 X + 30 Y points,
    # OOD 30 X + 10 Y: within a tile ID = OOD, pooled they differ.
    def tile(name, blocks, frac_ok):
        """`blocks` copies of one 10-point block: labels 1, 0, ...; the first frac_ok of each block right."""
        y = np.tile([1, 0] * 5, blocks)
        ok = np.tile(np.arange(10) < frac_ok * 10, blocks)
        return [name] * (10 * blocks), y, ok
    tx_i, yx_i, okx_i = tile("A", 1, 1.0)
    ty_i, yy_i, oky_i = tile(shift.PALM_BEACH_TILE, 3, 0.5)
    tx_o, yx_o, okx_o = tile("A", 3, 1.0)
    ty_o, yy_o, oky_o = tile(shift.PALM_BEACH_TILE, 1, 0.5)
    z, models, ds = cell_from(tx_o + ty_o, np.r_[yx_o, yy_o], np.r_[okx_o, oky_o],
                              tx_i + ty_i, np.r_[yx_i, yy_i], np.r_[okx_i, oky_i], n_models=20)
    out = shift.shift(z, models, ds, {"12099": "Palm Beach"})
    m = out["summary"]["metrics"]
    check(all(abs(m[k]["tile_delta"]) < 1e-12 for k in ("acc", "f1", "auc")),
          f"identical halves inside each tile: per-tile Delta 0 (acc {m['acc']['tile_delta']:.3g}, "
          f"F1 {m['f1']['tile_delta']:.3g}, AUROC {m['auc']['tile_delta']:.3g})")
    check(abs(m["acc"]["pooled_delta"]) > 0.1, f"pooled Delta != 0 ({m['acc']['pooled_delta']:+.3f})")
    pb = out["summary"]["palm_beach"]
    check(pb["n_id"] == 30 and pb["n_ood"] == 10 and np.isclose(pb["share_id"], 0.75) and pb["county_fips"] == "12099"
          and np.isclose(pb["acc_id"], 0.5), "Palm Beach: counts, shares, county, accuracy")
    pt = out["per_tile"]
    check(set(pt.tile) == {"A", shift.PALM_BEACH_TILE} and pt.set_index("tile").county[shift.PALM_BEACH_TILE] == "Palm Beach",
          "per-tile table: one row per tile with its county")
    check(-1 <= out["summary"]["corr_te_all"] <= 1 and len(out["summary"]["corr_te_all_ci"]) == 2,
          "corr_te on all OOD points with an interval")
    # b. one-class tile in the OOD set
    tiles_o = ["A"] * 20 + ["B"] * 20
    y_o = np.r_[np.array([1, 0] * 10), np.ones(20, int)]
    rng = np.random.default_rng(1)
    ok_o = rng.random((4, 40)) < 0.7
    tiles_i = ["A"] * 20 + ["B"] * 20
    y_i = np.array([1, 0] * 20)
    ok_i = rng.random((4, 40)) < 0.7
    z, models, ds = cell_from(tiles_o, y_o, ok_o, tiles_i, y_i, ok_i, n_models=4)
    m = shift.shift(z, models, ds)["summary"]["metrics"]
    check(m["acc"]["n_tiles"] == 2 and m["auc"]["n_tiles"] == 1 and m["f1"]["n_tiles"] == 2,
          "a tile whose OOD set has one class: out of AUROC (1 of 2 tiles), in accuracy and F1")


def check_real():
    print("4. the real cell")
    cells = Path("data/interim/aotil/cells_v4_2026-11-07")
    data = Path("data/processed_v4/dataset_FL.parquet")
    if not (cells / "FL_band_AE_s0.npz").exists() or not data.exists():
        print("  skip  (no cell or dataset)")
        return
    z = dict(np.load(cells / "FL_band_AE_s0.npz", allow_pickle=True))
    models = pd.read_csv(cells / "FL_band_AE_s0_models.csv")
    ds = pd.read_parquet(data, columns=["point_id", "tile_50km", "meta_point_county"]).set_index("point_id")
    s = shift.shift(z, models, ds)["summary"]
    m = s["metrics"]
    got = [round(m[k][f], 3) for k in ("acc", "auc", "f1") for f in ("pooled_delta", "tile_delta")]
    check(got == [-0.034, 0.017, -0.038, 0.029, -0.016, 0.068], f"Table 1 Deltas {got}")
    check([m[k]["tiles_worse"] for k in ("acc", "auc", "f1")] == [19, 19, 22]
          and [m[k]["n_tiles"] for k in ("acc", "auc", "f1")] == [27, 25, 27], "19/27, 19/25, 22/27 tiles")
    check([round(m[k]["holm_p"], 3) for k in ("auc", "f1", "acc")] == [0.044, 0.016, 0.091], "Holm p 0.044, 0.016, 0.091")
    check(round(s["corr_te_all"], 2) == 0.84, f"corr_te on all Zone A {s['corr_te_all']:+.3f}")


def main() -> int:
    check_rows()
    check_stats()
    check_shift()
    check_real()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
