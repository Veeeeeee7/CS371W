#!/usr/bin/env python3
"""
Checks for o1_shift.py. Run from the repo root:

    python aotil/test_o1_shift.py [--cells data/interim/aotil/cells_w7 --tag FL_band_AE_s0
                                   --dataset data/processed_w7/dataset_FL.parquet --draws 3]

1. weighted_aucs (the vectorised AUC behind the tile bootstrap) equals sklearn's
   roc_auc_score(y, p, sample_weight=w) to 1e-12, model by model, on synthetic data built to be
   hard for it: many tied scores (tree-like models with few distinct p, a constant model, ties
   between a positive and a negative) and weights with many zeros. NaN when a class has no
   weight.
2. The same on a real cell, all 300 models and both halves, under the first few tile-bootstrap
   draws that o1() makes, on the "all" and "land" scopes (a scope enters as zero weights).
   Skipped when the cell or the dataset is missing.
3. The default --model-stride covers all 8 model families; stride 10 covers only 4.
4. Pooled rows on a synthetic two-cell case: n_A / n_AE sum the cells, n_A_distinct /
   n_AE_distinct count a point shared by two cells once, n_tiles_shared counts shared tiles.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o1_shift as o1s  # noqa: E402

warnings.filterwarnings("ignore")   # sklearn warns on single-class bootstrap draws

FAILS = []


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


def sk_aucs(P, y, w):
    """roc_auc_score of every model (row of P) under weights w: the reference."""
    return np.array([roc_auc_score(y, p, sample_weight=w) for p in P])


def max_err(P, y, w):
    """Largest |weighted_aucs - roc_auc_score| over the models (rows of P)."""
    fast = o1s.weighted_aucs(o1s.score_groups(P, y), w)
    return float(np.max(np.abs(fast - sk_aucs(P, y, w))))


def synthetic_scores(rng, N, n):
    """N models' scores on n points: continuous, rounded to a few levels (trees),
    a handful of distinct values, and one constant model."""
    P = rng.random((N, n))
    P[1::4] = np.round(P[1::4], 1)                      # 11 levels
    P[2::4] = rng.integers(0, 4, (len(P[2::4]), n)) / 3  # 4 levels
    P[3] = 0.5                                          # all tied
    return np.round(P, 9)


def test_synthetic():
    print("1. weighted_aucs vs roc_auc_score, synthetic, heavy ties and zero weights")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(6):
        N, n = 80, int(rng.integers(200, 3000))
        y = (rng.random(n) < rng.uniform(0.1, 0.6)).astype(int)
        P = synthetic_scores(rng, N, n)
        weightings = (np.ones(n),
                      rng.integers(0, 4, n).astype(float),     # tile-like counts, many 0
                      rng.random(n) * (rng.random(n) < 0.6))   # fractional, 40% zero
        for w in weightings:
            worst = max(worst, max_err(P, y, w))
    check(worst < 1e-12, f"6 data sets x 3 weightings x 80 models: max |diff| = {worst:.2e}")

    y = np.r_[np.ones(50, int), np.zeros(50, int)]
    P = synthetic_scores(rng, 5, 100)
    w = np.r_[np.zeros(50), np.ones(50)]
    check(np.isnan(o1s.weighted_aucs(o1s.score_groups(P, y), w)).all(),
          "no weight on the positives -> NaN for every model")

    # A positive and a negative that tie count half: AUC 0.5 with one of each.
    groups = o1s.score_groups(np.array([[0.3, 0.3]]), np.array([1, 0]))
    a = o1s.weighted_aucs(groups, np.ones(2))
    check(np.allclose(a, 0.5), f"a tied (positive, negative) pair counts half: {a[0]}")


def test_real(cells, tag, dataset, n_draws):
    print(f"2. weighted_aucs vs roc_auc_score on {tag}, all models, {n_draws} tile-bootstrap draws")
    if not (Path(cells) / f"{tag}.npz").exists() or not Path(dataset).exists():
        print(f"  skip  {cells}/{tag}.npz or {dataset} missing")
        return
    columns = ["point_id", "tile_50km", "landcover", "block_id"]
    points = pd.read_parquet(dataset, columns=columns).set_index("point_id")
    c = o1s.cell_arrays(cells, tag, points, stride=1)
    check(c["go"]["N"] == c["N"] and c["gi"]["N"] == c["N"],
          f"stride 1 keeps all {c['N']} models in the bootstrap")

    test_tiles = np.unique(c["to"])
    rng = np.random.default_rng(0)                          # the draws o1() makes
    draws = [rng.choice(test_tiles, len(test_tiles)) for _ in range(n_draws)]
    land_o = o1s.is_land(c["o"])                            # the "land" scope's rule
    land_i = o1s.is_land(c["i"])
    worst, worst_gap = 0.0, 0.0
    for draw in draws:
        wo, wi = o1s.tile_weights(draw, c["to"], c["ti"])
        for mo, mi in ((1.0, 1.0), (land_o, land_i)):      # all points; land
            ref_o = sk_aucs(c["Po"], c["yo"], wo * mo)
            ref_i = sk_aucs(c["Pi"], c["yi"], wi * mi)
            err_o = np.abs(o1s.weighted_aucs(c["go"], wo * mo) - ref_o).max()
            err_i = np.abs(o1s.weighted_aucs(c["gi"], wi * mi) - ref_i).max()
            worst = max(worst, err_o, err_i)
            gap_err = abs(o1s.boot_gap(c, wo * mo, wi * mi) - np.mean(ref_i - ref_o))
            worst_gap = max(worst_gap, gap_err)
    check(worst < 1e-12, f"{n_draws} draws x 2 scopes x 2 halves x {c['N']} models: "
                         f"max |diff| = {worst:.2e}")
    check(worst_gap < 1e-12,
          f"boot_gap = the per-model roc_auc_score loop: max |diff| = {worst_gap:.2e}")

    all_o = np.ones(len(c["yo"]), bool)
    all_i = np.ones(len(c["yi"]), bool)
    a_id, a_od = o1s.per_model(c, all_o, all_i)
    err_id = np.abs(o1s.weighted_aucs(c["gi"], np.ones(len(c["yi"]))) - a_id).max()
    err_od = np.abs(o1s.weighted_aucs(c["go"], np.ones(len(c["yo"]))) - a_od).max()
    err = max(err_id, err_od)
    check(err < 1e-12,
          f"unit weights = the point-estimate AUCs (per_model): max |diff| = {err:.2e}")

    print("3. families in the bootstrapped models")
    models = pd.read_csv(Path(cells) / f"{tag}_models.csv")
    n_families = models.kind.iloc[::o1s.MODEL_STRIDE].nunique()
    families_10 = sorted(models.kind.iloc[::10].unique())
    check(n_families == models.kind.nunique() == 8,
          f"default stride {o1s.MODEL_STRIDE}: {n_families} of 8 families")
    check(families_10 == ["dt", "hgb", "mlp", "rf"], f"stride 10 (first look): only {families_10}")


def test_pooled_counts():
    print("4. pooled counts, synthetic two-cell case")
    rng = np.random.default_rng(2)
    # 6 tiles of 40 points per half; cell a tests tiles 0-3, cell b tiles 2-5,
    # so tiles 2 and 3 (and their points) are in both cells.
    rows = []
    for t in range(6):
        for zone in ("A", "AE"):
            for j in range(40):
                rows.append(dict(point_id=f"{zone}{t}_{j}", tile_50km=f"T{t}", domain=zone,
                                 landcover=11 if j < 5 else 41, block_id=f"B{t}_{j % 2}",
                                 label=int(rng.random() < 0.4)))
    points = pd.DataFrame(rows).set_index("point_id")
    paired_blocks = o1s.blocks_with_both_zones(points)

    cells = {}
    for tag, tiles in (("a", ["T0", "T1", "T2", "T3"]), ("b", ["T2", "T3", "T4", "T5"])):
        in_cell = points.tile_50km.isin(tiles)
        o = points[(points.domain == "A") & in_cell]
        i = points[(points.domain == "AE") & in_cell]
        yo, yi = o.label.to_numpy(), i.label.to_numpy()
        Po = np.round(rng.random((16, len(o))) + 0.3 * yo, 9)
        Pi = np.round(rng.random((16, len(i))) + 0.3 * yi, 9)
        cells[tag] = dict(yo=yo, yi=yi, Po=Po, Pi=Pi, o=o, i=i, to=o.tile_50km.to_numpy(),
                          ti=i.tile_50km.to_numpy(), N=16,
                          go=o1s.score_groups(Po, yo), gi=o1s.score_groups(Pi, yi))
    by_scope = {}
    for row in o1s.o1_pooled(cells, paired_blocks, 5, 0):
        by_scope[row["scope"]] = row

    r = by_scope["all"]
    check(r["n_A"] == 320 and r["n_A_distinct"] == 240
          and r["n_AE"] == 320 and r["n_AE_distinct"] == 240,
          f"all: n_A {r['n_A']} (sum), n_A_distinct {r['n_A_distinct']}, "
          f"n_AE {r['n_AE']}, n_AE_distinct {r['n_AE_distinct']} (expect 320/240/320/240)")
    r = by_scope["land"]
    check(r["n_A"] == 280 and r["n_A_distinct"] == 210,
          f"land: n_A {r['n_A']}, n_A_distinct {r['n_A_distinct']} (expect 280/210)")
    tiles_ok = True
    for row in by_scope.values():
        tiles_ok = tiles_ok and row["n_tiles_union"] == 6 and row["n_tiles_shared"] == 2
    check(tiles_ok, f"n_tiles_union {r['n_tiles_union']}, n_tiles_shared {r['n_tiles_shared']} "
                    "(expect 6, 2)")
    r = by_scope["all"]
    check(np.isfinite(r["ci_lo"]) and r["n_boot_used"] == 5,
          f"pooled interval finite, {r['n_boot_used']} of 5 draws used")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default="data/interim/aotil/cells_w7")
    parser.add_argument("--tag", default="FL_band_AE_s0")
    parser.add_argument("--dataset", default="data/processed_w7/dataset_FL.parquet")
    parser.add_argument("--draws", type=int, default=3, help="tile-bootstrap draws in check 2")
    args = parser.parse_args()
    test_synthetic()
    test_real(args.cells, args.tag, args.dataset, args.draws)
    test_pooled_counts()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
