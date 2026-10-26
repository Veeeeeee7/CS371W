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
5. Accuracy and F1 from outcome counts, synthetic: a hand-built case with known TP, FP, FN and
   TN gives exact accuracy and F1; a positive and a negative at p = 0.5 both count as predicted
   dry; integer tile weights give the same accuracy and F1 as duplicating the points; with
   equal flood shares the reweighted Zone A accuracy equals plain accuracy.
6. On a real cell, accuracy equals oodselect_paper.correctness(S, y)[0].mean(1) on both halves,
   and the strongest 15 are test models with the highest stored ID AUROC. Skipped when the cell
   or the dataset is missing.
Per-tile gap (docs/TODO-CLAUDE-CODE.md item 3), synthetic cases with known answers:
7. One tile only: every per-tile gap equals the pooled gap with the same label weights,
   computed point by point.
8. Two tiles whose ID and Zone A halves are identical inside each tile, in different
   proportions: every per-tile gap is exactly 0, the pooled accuracy gap is not.
9. One tile, ID flood share 0.8 and Zone A 0.5, the same TPR and TNR: weighted accuracy and F1
   gaps 0, raw accuracy gap 0.09.
10. A tile whose Zone A half lacks negatives, and one with no negatives in either half: both are
    dropped from AUROC and counted; accuracy keeps them.
11. Every tile drawn once: the bootstrap draw equals the point estimate; a tile drawn twice
    counts twice.
12. A class present in Zone A and absent from the ID half: the tile's weighted accuracy and F1
    are NaN, dropped and counted; raw accuracy keeps it. The reverse (ID holds a class Zone A
    lacks) is defined, the class getting weight 0.
13. o1_tile_null.shuffle_zones keeps each (tile, label) group's count in each half, moves each
    point with its label, tile, scores and correctness, and depends on its seed.
14. On a real cell: per-tile raw accuracy equals correctness per tile, weighted accuracy equals
    rho_A * TPR_ID + (1 - rho_A) * TNR_ID, per-tile AUROC equals roc_auc_score on the rebuilt p,
    and AUROC uses exactly the tiles where both halves hold both classes. Skipped when the cell
    or the dataset is missing.
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
import o1_tile_null  # noqa: E402
import oodselect_paper as op  # noqa: E402

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


def totals_of(S, y, tiles=None, w=None):
    """(models, 4) TP, FP, FN, TN of stored scores S on labels y, each tile weighted by w."""
    y = np.asarray(y)
    tiles = np.zeros(len(y), int) if tiles is None else np.asarray(tiles)
    n_tiles = tiles.max() + 1
    Z = op.correctness(S, y)[0].astype(bool)
    counts = o1s.outcome_counts(Z, y, tiles, n_tiles, np.ones(len(y), bool))
    w = np.ones(n_tiles) if w is None else np.asarray(w, float)
    return np.einsum("mtk,t->mk", counts, w)


def stored_scores(p, y):
    """S as zoo stores it: p for a flood point, 1 - p for a dry one."""
    return np.where(np.asarray(y)[None] == 1, p, 1 - p)


def test_accuracy_f1():
    print("5. accuracy and F1 from outcome counts, synthetic")
    # 3 TP, 1 FP, 2 FN, 4 TN
    y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    p = np.array([[.9, .8, .7, .6, .2, .1, .3, .4, .2, .1]])
    totals = totals_of(stored_scores(p, y), y)
    rates = o1s.rates(totals)
    check(totals.tolist() == [[3, 1, 2, 4]] and rates["acc"][0] == 7 / 10
          and rates["f1"][0] == 6 / 9,
          f"TP, FP, FN, TN = {totals[0].astype(int).tolist()}: accuracy {rates['acc'][0]} "
          f"(7/10), F1 {rates['f1'][0]:.6f} (6/9)")

    y = np.array([1, 0])
    totals = totals_of(stored_scores(np.array([[0.5, 0.5]]), y), y)
    check(totals.tolist() == [[0, 0, 1, 1]],
          "p = 0.5 is predicted dry for a positive (FN) and a negative (TN)")

    rng = np.random.default_rng(5)
    n, tiles = 60, rng.integers(0, 5, 60)
    y = (rng.random(n) < 0.4).astype(int)
    p = np.round(rng.random((7, n)), 1)                   # some p exactly 0.5
    w = np.array([2, 0, 1, 3, 1])
    weighted = o1s.rates(totals_of(stored_scores(p, y), y, tiles, w))
    copies = np.repeat(np.arange(n), w[tiles])
    duplicated = o1s.rates(totals_of(stored_scores(p[:, copies], y[copies]), y[copies]))
    check(all(np.array_equal(weighted[key], duplicated[key]) for key in ("acc", "f1")),
          "integer tile weights = duplicating the points (accuracy and F1, 7 models)")

    def half(n_flood, n_dry, seed):
        rng = np.random.default_rng(seed)
        y = np.r_[np.ones(n_flood, int), np.zeros(n_dry, int)]
        tiles = rng.integers(0, 3, len(y))
        Z = op.correctness(stored_scores(rng.random((9, len(y))), y), y)[0].astype(bool)
        return o1s.outcome_counts(Z, y, tiles, 3, np.ones(len(y), bool))

    values = o1s.count_metrics({"o": half(30, 70, 1), "i": half(60, 140, 2)}, np.ones(3))
    acc_a = values["acc"][1]
    adjacc_a = values["adjacc"][1]
    check(values["flood_share_A"] == values["flood_share_AE"] == 0.3
          and np.allclose(adjacc_a, acc_a, rtol=0, atol=1e-12),
          f"equal flood shares (0.3): reweighted = plain Zone A accuracy, max |diff| "
          f"{np.abs(adjacc_a - acc_a).max():.1e}")
    values = o1s.count_metrics({"o": half(30, 70, 1), "i": half(100, 100, 2)}, np.ones(3))
    check(not np.allclose(values["adjacc"][1], values["acc"][1]),
          "different flood shares (0.3, 0.5): the reweighting moves Zone A accuracy")


def test_real_accuracy(cells, tag, dataset):
    print(f"6. accuracy on {tag} from correctness, and the strongest models")
    if not (Path(cells) / f"{tag}.npz").exists() or not Path(dataset).exists():
        print(f"  skip  {cells}/{tag}.npz or {dataset} missing")
        return
    columns = ["point_id", "tile_50km", "landcover", "block_id"]
    points = pd.read_parquet(dataset, columns=columns).set_index("point_id")
    c = o1s.cell_arrays(cells, tag, points)
    every = (np.ones(len(c["yo"]), bool), np.ones(len(c["yi"]), bool))
    counts = o1s.scope_counts(c, *every)
    values = o1s.count_metrics(counts, np.ones(len(counts["tiles"])))
    z = np.load(Path(cells) / f"{tag}.npz", allow_pickle=True)
    want_id = op.correctness(z["S_id"], z["y_id"])[0].mean(1)
    want_a = op.correctness(z["S_ood"], z["y_ood"])[0].mean(1)
    err = max(np.abs(values["acc"][0] - want_id).max(), np.abs(values["acc"][1] - want_a).max())
    check(err < 1e-6, f"accuracy = correctness(S, y)[0].mean(1) on both halves, "
                      f"{c['N']} models: max |diff| {err:.1e} (float32 means)")

    strong = o1s.strongest_models(c)
    _, _, test = op.model_split(c["N"])
    rest = np.setdiff1d(test, strong)
    highest = len(rest) == 0 or c["auc_id"][strong].min() >= c["auc_id"][rest].max()
    check(len(strong) == min(o1s.STRONGEST, len(test)) and np.isin(strong, test).all()
          and highest,
          f"the strongest {len(strong)} are test models with the highest stored ID AUROC")


# ----------------------------------------------------------------------------- per-tile gap
def hand_cell(to, yo, Po, ti, yi, Pi) -> dict:
    """A hand-built cell with the keys tile_gaps reads, from each half's tiles, labels and
    p(flood) per model and point; correctness as zoo's stored scores give it."""
    yo, yi = np.asarray(yo), np.asarray(yi)
    return dict(to=np.asarray(to), ti=np.asarray(ti), yo=yo, yi=yi, Po=Po, Pi=Pi,
                Zo=op.correctness(stored_scores(Po, yo), yo)[0].astype(bool),
                Zi=op.correctness(stored_scores(Pi, yi), yi)[0].astype(bool))


def tile_run(c, draws=()):
    """(tile_gaps, tile_columns) of a hand-built cell on all its points."""
    t = o1s.tile_gaps(c, np.ones(len(c["yo"]), bool), np.ones(len(c["yi"]), bool))
    return t, o1s.tile_columns(t, o1s.draw_matrix(list(draws), t["tiles"]), None)


def noisy_p(rng, y, n_models, signal):
    """p(flood) on two decimals (ties, some exactly 0.5), higher for positives by signal."""
    return np.round(np.clip(rng.random((n_models, len(y))) + signal * (2 * y - 1), 0, 1), 2)


def weighted_acc_f1(P, y, w):
    """Per-model accuracy and F1 with point weights w, point by point: the reference."""
    flood = P > 0.5
    pos = y == 1
    tp, fp, fn = (flood & pos) @ w, (flood & ~pos) @ w, (~flood & pos) @ w
    return (flood == pos) @ w / w.sum(), 2 * tp / (2 * tp + fp + fn)


def counted(right: int, wrong: int, y: int) -> np.ndarray:
    """p(flood) of `right` points the model gets right and `wrong` it gets wrong, of label y."""
    hit, miss = (0.8, 0.2) if y == 1 else (0.2, 0.8)
    return np.r_[np.full(right, hit), np.full(wrong, miss)]


def test_tile_one_tile():
    print("7. per tile, one tile: equals the pooled gap with the same label weights")
    rng = np.random.default_rng(7)
    yo = (rng.random(90) < 0.4).astype(int)
    yi = (rng.random(120) < 0.65).astype(int)
    Po, Pi = noisy_p(rng, yo, 6, 0.25), noisy_p(rng, yi, 6, 0.3)
    c = hand_cell(["T0"] * 90, yo, Po, ["T0"] * 120, yi, Pi)
    _, cols = tile_run(c)
    rho_o, rho_i = yo.mean(), yi.mean()
    w = np.where(yi == 1, rho_o / rho_i, (1 - rho_o) / (1 - rho_i))
    acc_w, f1_w = weighted_acc_f1(Pi, yi, w)
    acc_i, f1_i = weighted_acc_f1(Pi, yi, np.ones(len(yi)))
    acc_o, f1_o = weighted_acc_f1(Po, yo, np.ones(len(yo)))
    auc = sk_aucs(Pi, yi, None) - sk_aucs(Po, yo, None)
    want = dict(acc=np.mean(acc_w - acc_o), f1=np.mean(f1_w - f1_o), auc=np.mean(auc),
                acc_raw=np.mean(acc_i - acc_o), f1_raw=np.mean(f1_i - f1_o))
    err = max(abs(cols[f"tile_{m}_gap"] - want[m]) for m in o1s.TILE_METRICS)
    check(err < 1e-12 and cols["tile_n_tiles"] == 1,
          f"5 metrics, 6 models: max |per tile - pooled with label weights| = {err:.1e}")


def test_tile_identical_halves():
    print("8. per tile, identical halves inside each tile, different proportions")
    rng = np.random.default_rng(8)
    y0, y1 = np.r_[np.ones(8, int), np.zeros(12, int)], np.r_[np.ones(11, int), np.zeros(9, int)]
    P0, P1 = noisy_p(rng, y0, 5, 0.05), noisy_p(rng, y1, 5, 0.6)  # tile T0 hard, T1 easy
    c = hand_cell(["T0"] * 20 + ["T1"] * 60, np.r_[y0, np.tile(y1, 3)],
                  np.hstack([P0, np.tile(P1, 3)]),
                  ["T0"] * 60 + ["T1"] * 20, np.r_[np.tile(y0, 3), y1],
                  np.hstack([np.tile(P0, 3), P1]))
    t, cols = tile_run(c)
    exact = all(np.all(t["gap"][m][t["ok"][m]] == 0) and cols[f"tile_{m}_gap"] == 0
                and cols[f"tile_{m}_n_tiles"] == 2 for m in o1s.TILE_METRICS)
    every = np.ones(80, bool)
    pooled = o1s.count_metrics(o1s.scope_counts(c, every, every), np.ones(2))
    pooled_gap = float(np.mean(pooled["acc"][0] - pooled["acc"][1]))
    check(exact and abs(pooled_gap) > 0.05,
          f"every per-tile gap exactly 0 on both tiles; pooled accuracy gap {pooled_gap:+.3f}")


def test_tile_label_weights():
    print("9. per tile, ID flood share 0.8 and Zone A 0.5 with the same TPR (0.9) and TNR (0.6)")
    Pi = np.tile(np.r_[counted(72, 8, 1), counted(12, 8, 0)], (3, 1))
    Po = np.tile(np.r_[counted(45, 5, 1), counted(30, 20, 0)], (3, 1))
    yi = np.r_[np.ones(80, int), np.zeros(20, int)]
    yo = np.r_[np.ones(50, int), np.zeros(50, int)]
    _, cols = tile_run(hand_cell(["T0"] * 100, yo, Po, ["T0"] * 100, yi, Pi))
    check(abs(cols["tile_acc_gap"]) < 1e-12 and abs(cols["tile_f1_gap"]) < 1e-12
          and abs(cols["tile_acc_raw_gap"] - 0.09) < 1e-12,
          f"weighted accuracy gap {cols['tile_acc_gap']:.1e}, weighted F1 gap "
          f"{cols['tile_f1_gap']:.1e}, raw accuracy gap {cols['tile_acc_raw_gap']:.3f} "
          "(0.84 - 0.75)")


def four_tile_cell(rng):
    """T0, T1: both classes in both halves; T2: the Zone A half holds only positives; T3: only
    positives in both halves (as tile 30_8 of the v3 cell)."""
    to, yo, ti, yi = [], [], [], []
    for tile, a_pos, a_neg, i_pos, i_neg in (("T0", 10, 14, 12, 9), ("T1", 7, 9, 15, 15),
                                             ("T2", 12, 0, 8, 10), ("T3", 9, 0, 13, 0)):
        to += [tile] * (a_pos + a_neg)
        yo += [1] * a_pos + [0] * a_neg
        ti += [tile] * (i_pos + i_neg)
        yi += [1] * i_pos + [0] * i_neg
    yo, yi = np.array(yo), np.array(yi)
    return hand_cell(to, yo, noisy_p(rng, yo, 4, 0.2), ti, yi, noisy_p(rng, yi, 4, 0.3))


def test_tile_auc_drop():
    print("10. per tile, tiles missing a class in one or both halves are dropped from AUROC")
    c = four_tile_cell(np.random.default_rng(10))
    t, cols = tile_run(c)
    want = []
    for tile in ("T0", "T1"):
        mo, mi = c["to"] == tile, c["ti"] == tile
        on_id = sk_aucs(c["Pi"][:, mi], c["yi"][mi], None)
        on_a = sk_aucs(c["Po"][:, mo], c["yo"][mo], None)
        want.append(np.mean(on_id - on_a))
    check(cols["tile_auc_n_tiles"] == 2 and cols["tile_auc_n_dropped"] == 2
          and np.isnan(t["gap"]["auc"][2:]).all() and np.isfinite(t["gap"]["auc"][:2]).all()
          and abs(cols["tile_auc_gap"] - np.mean(want)) < 1e-12,
          f"AUROC on {cols['tile_auc_n_tiles']} of 4 tiles ({cols['tile_auc_n_dropped']} dropped), "
          f"gap = the mean of the 2 tiles' roc_auc_score gaps")
    check(cols["tile_acc_n_tiles"] == 4 and cols["tile_acc_n_dropped"] == 0,
          "accuracy keeps all 4 tiles")


def test_tile_bootstrap():
    print("11. per tile, the bootstrap mean of a draw")
    c = four_tile_cell(np.random.default_rng(11))
    t, cols = tile_run(c, [np.array(["T2", "T0", "T3", "T1"])])
    err = max(abs(cols[f"tile_{m}_{end}"] - cols[f"tile_{m}_gap"])
              for m in o1s.TILE_METRICS for end in ("ci_lo", "ci_hi"))
    check(err < 1e-12, f"every tile drawn once = the point estimate, 5 metrics: "
                       f"max |diff| {err:.1e}")
    _, twice = tile_run(c, [np.array(["T0", "T0", "T1", "T9"])])   # T9 is no tile of the cell
    g = t["gap"]["acc"].mean(1)
    want = (2 * g[0] + g[1]) / 3
    check(abs(twice["tile_acc_ci_lo"] - want) < 1e-12,
          f"draw (T0, T0, T1): {twice['tile_acc_ci_lo']:.6f} = (2 g_T0 + g_T1) / 3")


def test_tile_missing_class():
    print("12. per tile, a class in Zone A that the ID half lacks")
    rng = np.random.default_rng(12)
    yo = np.r_[np.ones(10, int), np.zeros(10, int), np.ones(6, int), np.zeros(9, int),
               np.zeros(12, int)]
    yi = np.r_[np.ones(9, int), np.zeros(11, int), np.zeros(14, int), np.ones(5, int),
               np.zeros(7, int)]
    to = ["T0"] * 20 + ["T1"] * 15 + ["T2"] * 12   # T2: Zone A holds only negatives
    ti = ["T0"] * 20 + ["T1"] * 14 + ["T2"] * 12   # T1: the ID half holds only negatives
    c = hand_cell(to, yo, noisy_p(rng, yo, 5, 0.2), ti, yi, noisy_p(rng, yi, 5, 0.3))
    t, cols = tile_run(c)
    check(not t["ok"]["acc"][1] and np.isnan(t["gap"]["acc"][1]).all()
          and np.isnan(t["gap"]["f1"][1]).all() and cols["tile_acc_n_dropped"] == 1
          and cols["tile_f1_n_dropped"] >= 1 and cols["tile_acc_raw_n_dropped"] == 0,
          f"T1 weighted accuracy and F1 NaN, dropped ({cols['tile_acc_n_dropped']} for "
          f"accuracy); raw accuracy keeps all {cols['tile_acc_raw_n_tiles']} tiles")
    mi = c["ti"] == "T2"
    tnr = c["Zi"][:, mi & (c["yi"] == 0)].mean(1)
    check(t["ok"]["acc"][2] and np.allclose(t["ID"]["acc"][2], tnr, rtol=0, atol=1e-12),
          "T2 (Zone A all dry): defined, the ID positives weigh 0, weighted ID accuracy = ID TNR")


def test_shuffle_zones():
    print("13. o1_tile_null.shuffle_zones")
    rng = np.random.default_rng(13)
    to = np.repeat(["T0", "T1", "T2"], [30, 25, 40])
    ti = np.repeat(["T0", "T1", "T2"], [20, 35, 15])
    yo, yi = (rng.random(95) < 0.5).astype(int), (rng.random(70) < 0.6).astype(int)
    c = hand_cell(to, yo, noisy_p(rng, yo, 4, 0.2), ti, yi, noisy_p(rng, yi, 4, 0.3))
    c["o"] = pd.DataFrame(dict(k=np.arange(95)), index=[f"a{k}" for k in range(95)])
    c["i"] = pd.DataFrame(dict(k=np.arange(70)), index=[f"e{k}" for k in range(70)])
    c.update(N=4, auc_id=np.zeros(4), auc_ood=np.zeros(4))
    s = o1_tile_null.shuffle_zones(c, 1)

    def group_counts(tiles, y):
        return pd.Series(1, index=pd.MultiIndex.from_arrays([tiles, y])).groupby(level=[0, 1]).sum()

    same_counts = (group_counts(s["to"], s["yo"]).equals(group_counts(c["to"], c["yo"]))
                   and group_counts(s["ti"], s["yi"]).equals(group_counts(c["ti"], c["yi"])))
    by_id = {}
    for half, key in (("o", "o"), ("i", "i")):
        y, tiles, P, Z = c[f"y{key}"], c[f"t{key}"], c[f"P{key}"], c[f"Z{key}"]
        for k, point in enumerate(c[half].index):
            by_id[point] = (y[k], tiles[k], P[:, k], Z[:, k])
    travels = True
    for half in ("o", "i"):
        for k, point in enumerate(s[half].index):
            y, tile, p, z = by_id[point]
            travels &= (s[f"y{half}"][k] == y and s[f"t{half}"][k] == tile
                        and np.array_equal(s[f"P{half}"][:, k], p)
                        and np.array_equal(s[f"Z{half}"][:, k], z))
    moved = int(s["o"].index.str.startswith("e").sum())
    again = o1_tile_null.shuffle_zones(c, 1)["o"].index.equals(s["o"].index)
    other = not o1_tile_null.shuffle_zones(c, 2)["o"].index.equals(s["o"].index)
    check(same_counts and travels and moved > 0 and again and other
          and sorted(s["o"].index.tolist() + s["i"].index.tolist()) == sorted(by_id),
          f"(tile, label) counts kept in each half; every point keeps its label, tile, p and "
          f"correctness; {moved} former ID points now in Zone A; seeded")


def test_real_tiles(cells, tag, dataset):
    print(f"14. per tile on {tag}, all points")
    if not (Path(cells) / f"{tag}.npz").exists() or not Path(dataset).exists():
        print(f"  skip  {cells}/{tag}.npz or {dataset} missing")
        return
    columns = ["point_id", "tile_50km", "landcover", "block_id"]
    points = pd.read_parquet(dataset, columns=columns).set_index("point_id")
    c = o1s.cell_arrays(cells, tag, points)
    t = o1s.tile_gaps(c, np.ones(len(c["yo"]), bool), np.ones(len(c["yi"]), bool))
    tiles = np.unique(np.r_[c["to"], c["ti"]])
    err_raw = err_weighted = 0.0
    n_auc = 0
    for k, tile in enumerate(tiles):
        mo, mi = c["to"] == tile, c["ti"] == tile
        both = all(len(np.unique(y)) == 2 for y in (c["yo"][mo], c["yi"][mi]))
        n_auc += both
        err_raw = max(err_raw, np.abs(t["ID"]["acc_raw"][k] - c["Zi"][:, mi].mean(1)).max(),
                      np.abs(t["A"]["acc_raw"][k] - c["Zo"][:, mo].mean(1)).max())
        if t["ok"]["acc"][k]:
            rho_a = c["yo"][mo].mean()
            tpr = c["Zi"][:, mi & (c["yi"] == 1)].mean(1) if rho_a > 0 else 0.0
            tnr = c["Zi"][:, mi & (c["yi"] == 0)].mean(1) if rho_a < 1 else 0.0
            err_weighted = max(err_weighted,
                               np.abs(t["ID"]["acc"][k] - (rho_a * tpr + (1 - rho_a) * tnr)).max())
    check(len(t["tiles"]) == len(tiles) and err_raw < 1e-12 and err_weighted < 1e-12,
          f"{len(tiles)} tiles: raw accuracy = correctness per tile (max |diff| {err_raw:.1e}); "
          f"weighted ID accuracy = rho_A TPR + (1 - rho_A) TNR ({err_weighted:.1e})")
    used = np.flatnonzero(t["ok"]["auc"])
    err = 0.0
    for k in used[:3]:
        mo, mi = c["to"] == t["tiles"][k], c["ti"] == t["tiles"][k]
        err = max(err, np.abs(t["ID"]["auc"][k] - sk_aucs(c["Pi"][:, mi], c["yi"][mi], None)).max(),
                  np.abs(t["A"]["auc"][k] - sk_aucs(c["Po"][:, mo], c["yo"][mo], None)).max())
    check(len(used) == n_auc and err < 1e-12,
          f"AUROC on the {len(used)} tiles where both halves hold both classes; first 3 tiles x "
          f"{c['N']} models = roc_auc_score on the rebuilt p (max |diff| {err:.1e})")


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
    test_accuracy_f1()
    test_real_accuracy(args.cells, args.tag, args.dataset)
    test_tile_one_tile()
    test_tile_identical_halves()
    test_tile_label_weights()
    test_tile_auc_drop()
    test_tile_bootstrap()
    test_tile_missing_class()
    test_shuffle_zones()
    test_real_tiles(args.cells, args.tag, args.dataset)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
