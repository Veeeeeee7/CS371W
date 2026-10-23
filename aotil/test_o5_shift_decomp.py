#!/usr/bin/env python3
"""
Checks for o5_shift_decomp.py (docs/TASK-O5-REGIONS.md Part C). Run from the repo root:

    .venv/bin/python aotil/test_o5_shift_decomp.py [--out-dir results/2026-10-17/o5]

Sections 1-7 check the module's helpers on synthetic data against direct computations: the
feature space, kNN conflict and distance, the mid-rank percentile, the bootstrap multiplicities,
the pooled paired difference, terrain-adjusted OLS and pair-weighted null, the C3 weights and
the cross-fitted propensity.
Sections 8-11 use the real data and the written outputs, and are skipped when those are missing:
8 seed 0 through seed_points / c3_seed against brute-force kNN in an independently built space;
9 internal consistency of C_conflict, C_support and C_decomposition; 10 C2 rebuilt from
S1_controls.csv, S2_null.parquet and sklearn's kNN without the module's pooling code; 11 C3's
counts and accuracies rebuilt from the cells. Exit code 0 when every check passes.
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

import match_controls as mc  # noqa: E402
import o5_shift_decomp as sd  # noqa: E402
import oodselect_paper as op  # noqa: E402
import zoo  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


FEATS = ["landcover", "slope", "tri_300", "wetindexcat", "log10_dist_river", "log10_dist_coast", "hand"]
CELLS = Path("data/interim/aotil/cells_w7")
PREFIX = "FL_band_AE"
SEEDS = (0, 1, 2)
WATER = 11
TOL_CSV = 2e-6          # the outputs are written with %.6g
LABELS = ("all", 1, 0)
REAL = {}               # real-data inputs, loaded once and shared by sections 8, 10 and 11


# ----------------------------------------------- real-data helpers (no o5_shift_decomp code)
def real_inputs(o5_dir: Path):
    """The dataset and S1 / S2 / regions_land, loaded once; None when an input is missing."""
    if "df" in REAL:
        return REAL
    need = [Path("data/processed_w7/FEATURES.txt"), Path("data/processed_w7/dataset_FL.parquet"),
            *(CELLS / f"{PREFIX}_s{s}.npz" for s in SEEDS), *(CELLS / f"{PREFIX}_s{s}_models.csv" for s in SEEDS),
            o5_dir / "S1_controls.csv", o5_dir / "S2_null.parquet", o5_dir / "regions_land.csv"]
    if not all(p.exists() for p in need):
        return None
    df, feats = zoo.load("FL", Path("data/processed_w7"))
    REAL.update(df=df, feats=feats, rows={}, conf={},
                K=pd.read_csv(o5_dir / "S1_controls.csv", dtype={"point_id": str, "control_point_id": str}),
                rl=pd.read_csv(o5_dir / "regions_land.csv"), N=pd.read_parquet(o5_dir / "S2_null.parquet"))
    return REAL


def cell_rows(s: int) -> dict:
    """zoo.cell_data's halves for seed s: the rows the 300 models were fitted and scored on."""
    if s not in REAL["rows"]:
        REAL["rows"][s] = zoo.cell_data(REAL["df"], REAL["feats"], "band", "AE", s)
    return REAL["rows"][s]


def own_space(tr: pd.DataFrame, feats):
    """The 9-D space rebuilt without FeatureSpace: the 7 features with the training
    median filled in, 0/1 indicators for missing hand and wetindexcat, every column
    standardized by the training mean and population SD. Returns raw rows -> 9-D."""
    raw = tr[feats].to_numpy(float)
    med = np.nanmedian(raw, axis=0)
    miss = [feats.index("hand"), feats.index("wetindexcat")]

    def fill(r):
        r = np.asarray(r, float)
        return np.hstack([np.where(np.isnan(r), med, r), np.isnan(r[:, miss]).astype(float)])

    filled = fill(raw)
    mu, sdv = filled.mean(axis=0), filled.std(axis=0)
    if not np.all(sdv > 0):
        raise ValueError("an SD-0 column in the training space")

    def to_space(r):
        return (fill(r) - mu) / sdv
    return to_space


def own_conflict(s: int):
    """Per test point of seed s (Zone A test = half A, Zone AE test = half AE):
    conflict_k10/25/50 and knn_dist_k25 from sklearn's ball-tree kNN in own_space,
    and the number of (point, k) with a tie at the k-th neighbour distance."""
    if s in REAL["conf"]:
        return REAL["conf"][s]
    d, feats = cell_rows(s), REAL["feats"]
    space = own_space(d["tr"], feats)
    ytr = d["tr"].label.to_numpy()
    nn = NearestNeighbors(algorithm="ball_tree").fit(space(d["tr"][feats]))
    parts, n_tie = [], 0
    for half, t in (("A", d["ood_te"]), ("AE", d["id_te"])):
        dist, idx = nn.kneighbors(space(t[feats]), n_neighbors=max(sd.KS) + 1)
        y = t.label.to_numpy()
        differ = ytr[idx] != y[:, None]
        for k in sd.KS:
            n_tie += int((np.abs(dist[:, k - 1] - dist[:, k]) <= 1e-9 * np.maximum(1.0, dist[:, k])).sum())
        parts.append(pd.DataFrame({"point_id": t.point_id.astype(str).to_numpy(), "half": half, "label": y,
                                   "land": t.landcover.to_numpy() != WATER,
                                   **{f"conflict_k{k}": differ[:, :k].mean(axis=1) for k in sd.KS},
                                   "knn_dist_k25": dist[:, :25].mean(axis=1)}))
    T = pd.concat(parts, ignore_index=True).set_index("point_id", drop=False)
    if not T.index.is_unique:
        raise ValueError(f"seed {s}: a point id is in both test halves")
    REAL["conf"][s] = (T, n_tie)
    return REAL["conf"][s]


def own_test_models(n_models: int) -> np.ndarray:
    """The paper's 60 test models: the 20% split of two sklearn splits at random_state 42."""
    _, te = train_test_split(np.arange(n_models), test_size=0.2, random_state=42)
    return np.sort(te)


def own_top15(models: pd.DataFrame):
    """(the 15 test models with the highest auc_id, ties by index; the 60 test models)."""
    te = own_test_models(len(models))
    auc = models.auc_id.to_numpy(float)
    by_auc = sorted(te, key=lambda i: (-auc[i], i))
    return np.sort(np.array(by_auc[:15])), te


def own_correct(S, y) -> np.ndarray:
    """1 where the stored true-class probability S gives the right class: S > 0.5, or
    S = 0.5 on a y = 0 point (argmax over [1 - p, p] takes class 0 on a tie)."""
    S, y = np.asarray(S, float), np.asarray(y)
    return ((S > 0.5) | ((S == 0.5) & (y[None, :] == 0))).astype(float)


def own_midrank(obs: float, null) -> float:
    """100 * (#{null < obs} + 0.5 #{|null - obs| <= 1e-12}) / #null over the finite null values."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan
    tie = np.abs(null - obs) <= 1e-12
    return float(100.0 * (((null < obs) & ~tie).sum() + 0.5 * tie.sum()) / len(null))


def null_summary(nv) -> dict:
    """Mean and 2.5 / 97.5 % of the finite null values, and their number."""
    nv = nv[np.isfinite(nv)]
    if not len(nv):
        return dict(null_mean=np.nan, null_p2_5=np.nan, null_p97_5=np.nan, n_null=0)
    lo, hi = np.percentile(nv, [2.5, 97.5])
    return dict(null_mean=float(nv.mean()), null_p2_5=float(lo), null_p97_5=float(hi), n_null=len(nv))


def test_feature_space():
    print("1. FeatureSpace")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 7))
    X[:, 0] = rng.choice([11, 21, 41, 90], 500)
    X[rng.random(500) < 0.2, 6] = np.nan
    X[rng.random(500) < 0.1, 3] = np.nan
    fs = sd.FeatureSpace(FEATS).fit(X)
    T = fs.transform(X)
    check(T.shape == (500, 9) and fs.columns[-2:] == ["miss_hand", "miss_wetindexcat"], f"9 columns: {fs.columns}")
    check(np.allclose(fs.median, np.nanmedian(X, axis=0)), "medians are the training nanmedians")
    check(np.allclose(T.mean(0), 0, atol=1e-12) and np.allclose(T.std(0), 1), "training columns: mean 0, SD 1")
    back = T * fs.sd + fs.mean
    check(np.array_equal(back[:, 7] > 0.5, np.isnan(X[:, 6])) and np.array_equal(back[:, 8] > 0.5, np.isnan(X[:, 3])),
          "indicators = 1 exactly where hand / wetindexcat are missing")
    filled = np.where(np.isnan(X), fs.median, X)
    check(np.allclose(back[:, :7], filled), "missing values filled with the training median")
    Y = X.copy()
    Y[:, 2] = 3.0
    fs2 = sd.FeatureSpace(FEATS).fit(Y)
    check(fs2.zero_sd == ["tri_300"] and np.allclose(fs2.transform(Y)[:, 2], 0), "SD-0 column recorded and only centred")
    Z = rng.normal(size=(5, 7))
    Z[0, 6] = np.nan
    check(np.isfinite(fs.transform(Z)).all(), "new rows with gaps transform to finite values")


def test_knn():
    print("2. knn_stats vs brute force")
    rng = np.random.default_rng(1)
    Xtr, ytr = rng.normal(size=(400, 4)), rng.integers(0, 2, 400)
    Xq, yq = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    got = sd.knn_stats(cKDTree(Xtr), ytr, Xq, yq, ks=(3, 10, 25), workers=1)
    D = np.sqrt(((Xq[:, None, :] - Xtr[None]) ** 2).sum(-1))
    o = np.argsort(D, axis=1)
    ok = all(np.allclose(got[f"conflict_k{k}"], (ytr[o[:, :k]] != yq[:, None]).mean(1)) for k in (3, 10, 25))
    check(ok, "conflict_k for k = 3, 10, 25")
    check(np.allclose(got["knn_dist_k25"], np.take_along_axis(D, o[:, :25], 1).mean(1)), "mean distance to the 25 nearest")


def test_midrank():
    print("3. midrank_percentile")
    null = np.array([0.0, 1.0, 2.0, 3.0])
    check(sd.midrank_percentile(5, null) == 100 and sd.midrank_percentile(-1, null) == 0, "above all 100, below all 0")
    check(sd.midrank_percentile(2.0, null) == 100 * 2.5 / 4, "a tie counts half: 62.5")
    check(sd.midrank_percentile(2.0 + 5e-13, null) == 62.5, "|diff| <= 1e-12 is a tie")
    check(sd.midrank_percentile(1.5, np.r_[null, np.nan]) == 50, "NaN dropped")
    check(np.isnan(sd.midrank_percentile(np.nan, null)), "NaN observed -> NaN")


def test_boot():
    print("4. boot_multiplicities")
    M = sd.boot_multiplicities(7, 300, "x")
    check(M.shape == (300, 7) and np.all(M.sum(1) == 7), "each draw resamples 7 units")
    check(np.array_equal(M, sd.boot_multiplicities(7, 300, "x")) and not np.array_equal(M, sd.boot_multiplicities(7, 300, "y")),
          "same key -> same draws; other key -> other draws")
    check(abs(M.mean() - 1) < 0.05, f"mean multiplicity ~1: {M.mean():.3f}")


def test_pairs():
    print("5. pair_acc + pooled_diff / pooled_adj")
    rng = np.random.default_rng(2)
    n_pts = 600
    ids = np.array([f"p{i}" for i in range(n_pts)])
    st = pd.DataFrame({"conflict_k25": rng.random(n_pts), "knn_dist_k25": rng.random(n_pts),
                       **{z: rng.normal(size=n_pts) for z in sd.TERRAIN_Z}}, index=ids)
    perm = rng.permutation(n_pts)
    units = [(ids[perm[a:b]], ids[perm[300 + a:300 + b]]) for a, b in ((0, 40), (40, 110), (110, 130), (130, 300))]
    accs = [sd.pair_acc(st, q, c, measures=("conflict_k25", "knn_dist_k25")) for q, c in units]
    m = np.array([2.0, 0.0, 1.0, 1.0])
    Qs = np.concatenate([np.tile(units[i][0], int(m[i])) for i in range(4)])
    Cs = np.concatenate([np.tile(units[i][1], int(m[i])) for i in range(4)])
    want = float((st.loc[Qs, "conflict_k25"].to_numpy() - st.loc[Cs, "conflict_k25"].to_numpy()).mean())
    check(abs(sd.pooled_diff(accs, m, "conflict_k25") - want) < 1e-12, "pooled_diff = mean over the repeated pairs")
    X = np.vstack([np.column_stack([np.ones(len(Qs)), np.ones(len(Qs)), st.loc[Qs, list(sd.TERRAIN_Z)]]),
                   np.column_stack([np.ones(len(Cs)), np.zeros(len(Cs)), st.loc[Cs, list(sd.TERRAIN_Z)]])])
    y = np.r_[st.loc[Qs, "conflict_k25"], st.loc[Cs, "conflict_k25"]]
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    check(abs(sd.pooled_adj(accs, m, "conflict_k25") - b) < 1e-10, f"pooled_adj = lstsq on the stacked rows ({b:+.5f})")
    st2 = st.copy()
    for q, c in units:                       # controls with their query's terrain
        st2.loc[c, list(sd.TERRAIN_Z)] = st2.loc[q, list(sd.TERRAIN_Z)].to_numpy()
    accs2 = [sd.pair_acc(st2, q, c, measures=("conflict_k25",)) for q, c in units]
    one = np.ones(4)
    check(abs(sd.pooled_adj(accs2, one, "conflict_k25") - sd.pooled_diff(accs2, one, "conflict_k25")) < 1e-10,
          "pair-identical terrain: adjusted = raw paired difference")
    empty = sd.pair_acc(st, [], [], measures=("conflict_k25",))
    check(np.isnan(sd.pooled_diff([empty], [1.0], "conflict_k25")) and np.isnan(sd.pooled_adj([empty], [1.0], "conflict_k25")),
          "no pairs -> NaN")
    # null_values: draw d pools the d-th patch of EVERY unit, pair-weighted (not a mean of unit means)
    sizes = {("u", 0): (5, 60), ("u", 1): (2, 9), ("v", 0): (40, 3), ("v", 1): (31, 33)}   # (unit, draw) -> pair slice
    nacc = {k: sd.pair_acc(st, ids[perm[a:a + b]], ids[perm[300 + a:300 + a + b]], measures=("conflict_k25",))
            for k, (a, b) in sizes.items()}
    nv = sd.null_values(nacc, ["u", "v"], "conflict_k25", adjusted=False, n_draws=3)
    want = []
    for d in (0, 1):
        q = np.concatenate([ids[perm[a:a + b]] for (u, dd), (a, b) in sizes.items() if dd == d])
        c = np.concatenate([ids[perm[300 + a:300 + a + b]] for (u, dd), (a, b) in sizes.items() if dd == d])
        want.append(float((st.loc[q, "conflict_k25"].to_numpy() - st.loc[c, "conflict_k25"].to_numpy()).mean()))
    check(np.allclose(nv[:2], want, atol=1e-12) and np.isnan(nv[2]),
          f"null_values: draw d = pair-weighted mean over every unit's d-th patch pairs {np.round(want, 4).tolist()}; "
          f"a draw missing a unit -> NaN (got {np.round(nv, 4).tolist()})")


def test_c3():
    print("6. c3_weights")
    rng = np.random.default_rng(3)
    odds = rng.lognormal(size=1000)
    G = rng.dirichlet(np.ones(3), size=1000)
    oc, cap = sd.clip_odds(odds)
    check(cap == np.quantile(odds, 0.99) and np.array_equal(oc, np.minimum(odds, cap)) and (odds > cap).sum() == 10,
          "clip_odds caps at the 99th percentile (10 of 1000 clipped)")
    w, ess = sd.c3_weights(oc, G, np.ones(3))
    check(np.allclose(w, oc / oc.sum()) and abs(w.sum() - 1) < 1e-12, "all m = 1: the clipped binary odds, normalised")
    check(abs(ess - 1 / np.sum(w ** 2)) < 1e-9, "ESS = 1 / sum w^2")
    w2, _ = sd.c3_weights(oc, G, np.array([2.0, 0.0, 1.0]))
    r2 = oc * (2 * G[:, 0] + G[:, 2])
    check(np.allclose(w2, r2 / r2.sum()), "multiplicities (2, 0, 1): oc * (2 g_1 + g_3), normalised; no re-clipping")
    # exact discrete example: x in {0, 1, 2}; AE density p, region densities q1, q2
    x = np.repeat([0, 1, 2], [500, 300, 200])           # AE sample: p = (.5, .3, .2)
    p = np.array([0.5, 0.3, 0.2])
    q1, q2 = np.array([0.1, 0.3, 0.6]), np.array([0.6, 0.3, 0.1])
    n1, n2 = 100, 300                                    # region sizes -> pooled q
    qp = (n1 * q1 + n2 * q2) / (n1 + n2)
    acc = np.array([0.9, 0.7, 0.4])[x] + 0.0
    odds = (qp / p)[x]                                   # exact density ratio (constant drops out)
    Gx = np.column_stack([(n1 * q1 / (n1 * q1 + n2 * q2))[x], (n2 * q2 / (n1 * q1 + n2 * q2))[x]])
    w, _ = sd.c3_weights(odds, Gx, np.ones(2))
    check(abs(w @ acc - qp @ np.array([0.9, 0.7, 0.4])) < 1e-12, "pooled target: reweighted mean = E_q[acc]")
    w1, _ = sd.c3_weights(odds, Gx, np.array([1.0, 0.0]))
    check(abs(w1 @ acc - q1 @ np.array([0.9, 0.7, 0.4])) < 1e-12, "one region via composition: = E_q1[acc]")
    w3, _ = sd.c3_weights(odds, Gx, np.array([2.0, 1.0]))
    qm = (2 * n1 * q1 + n2 * q2) / (2 * n1 + n2)
    check(abs(w3 @ acc - qm @ np.array([0.9, 0.7, 0.4])) < 1e-12, "multiplicities (2, 1): = E over the resampled mixture")


def test_crossfit():
    print("7. crossfit_propensity")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1200, 5))
    y = rng.integers(0, 2, 1200)
    p0, p1, auc, fold = sd.crossfit_propensity(X[y == 0], X[y == 1], seed=1)
    yy = np.r_[np.zeros((y == 0).sum()), np.ones((y == 1).sum())]
    check(set(fold) == set(range(5)) and all(abs((fold == f).sum() - 240) <= 1 for f in range(5)), "5 folds of 240")
    check(all(abs(yy[fold == f].mean() - yy.mean()) < 0.01 for f in range(5)), "folds are stratified")
    m = HistGradientBoostingClassifier(random_state=1, **sd.HGB_PARAMS).fit(np.vstack([X[y == 0], X[y == 1]]), yy)
    auc_in = roc_auc_score(yy, m.predict_proba(np.vstack([X[y == 0], X[y == 1]]))[:, 1])
    check(abs(auc - 0.5) < 0.06 and auc_in > 0.8, f"random labels: cross-fitted AUC {auc:.3f} ~ 0.5, in-sample {auc_in:.3f}")
    A, B = rng.normal(0, 1, (500, 3)), rng.normal(6, 1, (100, 3))
    _, pb, auc2, _ = sd.crossfit_propensity(A, B, seed=2)
    check(auc2 > 0.99 and np.all((pb >= 0) & (pb <= 1)), f"separated Gaussians: AUC {auc2:.4f}")


def test_real(o5_dir: Path):
    print("8. real data, seed 0 (the module's own seed_points / c3_seed)")
    if real_inputs(o5_dir) is None:
        print("  skip  inputs missing")
        return
    df, feats, K, rl = REAL["df"], REAL["feats"], REAL["K"], REAL["rl"]
    check(feats == FEATS, f"the 7 model features in order: {feats}")
    ctx = mc.load_contexts([0], str(CELLS), PREFIX, "data/processed/dataset_FL.parquet",
                           "results/2026-10-16/oodselect_paper", "results/2026-10-16/regions", 500, 20,
                           verbose=False)[0]
    P, info, c3 = sd.seed_points(0, df, feats, ctx, CELLS, PREFIX, K, 2, log=lambda *a: None)
    REAL["P0"] = P
    check(info["zero_sd"] == "", "no SD-0 column in the training space")
    d = cell_rows(0)
    space = own_space(d["tr"], feats)
    Xtr = space(d["tr"][feats])
    ytr = d["tr"].label.to_numpy()
    raw_all = pd.concat([d["ood_te"], d["id_te"]]).set_index("point_id")
    rng = np.random.default_rng(5)
    R = P[(P.component >= 0) & P.land]
    AE = P[(P.half == "AE") & P.land]
    k0 = K[(K.seed == 0) & (K.status == "matched")]
    CT = P.loc[k0.control_point_id.to_numpy()]
    for name, grp in (("region land points", R), ("Zone AE test land points", AE), ("S1 controls", CT)):
        pick = grp.iloc[rng.choice(len(grp), 40, replace=False)]
        Xq = space(raw_all.loc[pick.point_id, feats])
        D = np.sqrt(((Xq[:, None, :] - Xtr[None]) ** 2).sum(-1))
        o = np.argsort(D, axis=1)
        lab = raw_all.loc[pick.point_id, "label"].to_numpy()
        ok = all(np.allclose(pick[f"conflict_k{k}"], (ytr[o[:, :k]] != lab[:, None]).mean(1)) for k in sd.KS)
        check(ok, f"conflict_k10/25/50 on 40 {name} = brute force (own feature space)")
        check(np.allclose(pick.knn_dist_k25, np.take_along_axis(D, o[:, :25], 1).mean(1)),
              f"knn_dist_k25 on the 40 {name} = brute force")
    pts = ctx.pts.set_index("point_id")
    c = pts.loc[k0.control_point_id]
    q = pts.loc[k0.point_id]
    check(bool(c.land.all() and (~c.selected).all() and (c.component < 0).all()),
          f"{len(k0)} S1 controls: land, unselected, outside the regions")
    check(np.array_equal(c.label.to_numpy(), q.label.to_numpy()) and np.array_equal(c.tile_50km.to_numpy(), q.tile_50km.to_numpy()),
          "control has the region point's label and tile")
    ide = d["id_te"]
    land_ae_ids = set(ide.point_id[ide.landcover.to_numpy() != WATER].astype(str))
    reg_ids = set(K[K.seed == 0].point_id)                    # S1 has a row for every region land point
    want = reg_ids | set(k0.control_point_id) | land_ae_ids
    got = set(P.index[P.propensity.notna()])
    check(got == want and info["n_eval"] == len(want),
          f"C1 propensity on exactly {len(reg_ids)} region land + {len(k0)} controls + {len(land_ae_ids)} land Zone AE "
          f"test points (scored {len(got)}; {len(got - want)} outside that set, e.g. water)")
    rl0 = rl[(rl.seed == 0) & (rl.n_land > 0)].set_index("component").n_land
    check(R.groupby("component").size().reindex(rl0.index).fillna(0).astype(int).equals(rl0.astype(int)),
          "region land counts = regions_land.csv")
    models = pd.read_csv(CELLS / f"{PREFIX}_s0_models.csv")
    top, te = own_top15(models)
    check(np.array_equal(te, np.sort(op.model_split(len(models))[2])), "own 60 test models = op.model_split's")
    check(models.auc_id.iloc[top].min() > models.auc_id.iloc[np.setdiff1d(te, top)].max(),
          "the 15 highest auc_id among the 60 test models are unambiguous (no tie at the 15th)")
    msgs = []
    rows, _ = sd.c3_seed(0, c3, P, K, rl, models, 2, log=lambda *a: msgs.append(" ".join(map(str, a))))
    m = re.search(r"top-15 models \[([0-9, ]*)\]", " ".join(msgs))
    used = np.sort([int(x) for x in m.group(1).split(",")]) if m and m.group(1).strip() else np.zeros(0, int)
    check(np.array_equal(used, top), f"c3_seed's top-15 model ids = the 15 highest auc_id test models {top.tolist()} "
          f"(c3_seed used {used.tolist()})")
    rows = pd.DataFrame(rows)
    check(bool((rows.n_ae_test_land == len(land_ae_ids)).all()),
          f"c3_seed: n_ae_test_land = {len(land_ae_ids)} land Zone AE test points of {len(ide)} "
          f"(c3_seed: {sorted(set(rows.n_ae_test_land))})")
    z = np.load(CELLS / f"{PREFIX}_s0.npz", allow_pickle=True)
    check(np.array_equal(z["point_id_id"].astype(str), ide.point_id.astype(str).to_numpy())
          and np.array_equal(z["y_id"], ide.label.to_numpy()), "cell's Zone AE test columns = zoo.cell_data's id_te rows")
    Zi = own_correct(z["S_id"], z["y_id"])
    land_ae = ide.landcover.to_numpy() != WATER
    acc = float(Zi[top][:, land_ae].mean())
    got_acc = float(rows[(rows.target == "all") & (rows.model_set == "top15")].acc_ae.iloc[0])
    check(abs(got_acc - acc) < 1e-6,           # the module averages float32 correctness
          f"c3_seed: top-15 acc_ae {got_acc:.6f} = own correctness on land Zone AE test {acc:.6f} "
          f"(|diff| {abs(got_acc - acc):.1e})")


def test_outputs(out_dir: Path, o5_dir: Path):
    print("9. written outputs")
    files = [out_dir / f for f in ("C_conflict.csv", "C_support.csv", "C_decomposition.csv")]
    if not all(f.exists() for f in files):
        print("  skip  outputs missing")
        return
    C2, C1, C3 = (pd.read_csv(f) for f in files)
    K = pd.read_csv(o5_dir / "S1_controls.csv")
    rl = pd.read_csv(o5_dir / "regions_land.csv")
    pr = C2[C2.primary]
    check(len(pr) == 1, "exactly one primary row")
    p = pr.iloc[0]
    check(p.analysis == "raw" and p.scope == "all" and p.measure == "conflict_k25", "primary = raw / all / conflict_k25")
    check(int(p.n_pairs) == int((K.status == "matched").sum()) and int(p.n_regions) == 16,
          f"primary pairs = every matched region land point ({int(p.n_pairs)}), 16 regions")
    want_met = bool(p.ci_lo > 0 and p.null_percentile > 95)
    check(str(p.met) == str(want_met), f"met follows the rule ({p.met})")
    worst = 0.0
    for T in (C2, C1):
        reg = T[(T.scope_kind == "region") & (T.analysis == "raw")]
        for _, r in T[(T.scope_kind == "pooled") & (T.analysis == "raw") & (T.label_subset == "all")].iterrows():
            rr = reg[reg.measure == r.measure]
            if r.scope in sd.TYPES:
                rr = rr[rr.type == r.scope]
            elif r.scope.startswith("seed="):
                rr = rr[rr.seed == int(r.scope.split("=")[1])]
            w = (rr.n_pairs * rr["diff"]).sum() / rr.n_pairs.sum()
            worst = max(worst, abs(w - r["diff"]))
    check(worst < 1e-5, f"pooled raw diff = pair-weighted mean of region diffs (max |diff| {worst:.1e})")
    check(np.allclose(C3.total, C3.acc_ae - C3.acc_region, atol=1e-6)
          and np.allclose(C3.total, C3.covariate + C3.concept, atol=1e-6), "C3: total = acc_ae - acc_region = covariate + concept")
    per = C3[C3.seed != "pooled"]
    worst = 0.0
    for _, r in C3[C3.seed == "pooled"].iterrows():
        g = per[(per.target == r.target) & (per.model_set == r.model_set) & per.seed.isin(str(r.seeds_pooled).split(";"))]
        worst = max(worst, abs(g.acc_ae_w.mean() - r.acc_ae_w), abs(g.acc_region.mean() - r.acc_region))
    check(worst < 1e-5, "pooled C3 = mean over seeds")
    worst, worst_m = 0.0, 0.0
    for _, r in per[per.target.isin(["all", *sd.TYPES])].iterrows():
        g = per[(per.seed == r.seed) & (per.model_set == r.model_set) & per.target.str.startswith("region_c")]
        if r.target != "all":
            cs = rl[(rl.seed == int(r.seed)) & (rl.type == r.target)].component
            g = g[g.target.isin([f"region_c{c}" for c in cs])]
        worst = max(worst, abs((g.n_region_land * g.acc_ae_w).sum() / g.n_region_land.sum() - r.acc_ae_w_mix))
        worst_m = max(worst_m, abs(g.ae_weight_mass.sum() - r.ae_weight_mass))
    check(worst < 1e-5 and worst_m < 1e-5, "acc_ae_w_mix = point-weighted mean of the per-region acc_ae_w; masses add up")
    a = per[per.target == "all"]
    check(np.allclose(a.ae_weight_mass, 1) and np.allclose(a.region_point_share, 1), "target 'all': mass 1, share 1")
    ok = C3[C3.covariate_ci_lo.notna()]
    check(bool(((ok.covariate_ci_lo <= ok.covariate + 1e-9) & (ok.covariate <= ok.covariate_ci_hi + 1e-9)).mean() > 0.8),
          "C3 intervals mostly contain their point estimates")


def test_rebuild_c2(out_dir: Path, o5_dir: Path):
    print("10. C2 rebuilt from S1_controls.csv, S2_null.parquet and an independent kNN (all seeds)")
    f = out_dir / "C_conflict.csv"
    if real_inputs(o5_dir) is None or not f.exists():
        print("  skip  inputs or C_conflict.csv missing")
        return
    C2 = pd.read_csv(f)
    K, rl, N = REAL["K"], REAL["rl"], REAL["N"]
    meas = [f"conflict_k{k}" for k in sd.KS]
    T, n_tie = {}, 0
    for s in SEEDS:
        T[s], nt = own_conflict(s)
        n_tie += nt
    check(n_tie == 0, f"no tie at the 10th / 25th / 50th neighbour distance on any test point ({n_tie})")
    if "P0" in REAL:
        P0 = REAL["P0"]
        t0 = T[0].loc[P0.index]
        worst = max(float(np.abs(t0[c].to_numpy() - P0[c].to_numpy(float)).max()) for c in meas + ["knn_dist_k25"])
        check(len(t0) == len(T[0]) and worst < 1e-9,
              f"independent kNN = seed_points on all {len(P0)} seed-0 test points (max |diff| {worst:.1e})")
    pos = {s: pd.Series(np.arange(len(T[s])), index=T[s].index) for s in SEEDS}
    V = {s: T[s][meas].to_numpy(float) for s in SEEDS}
    lab_of = {s: T[s].label.to_numpy() for s in SEEDS}

    def pair_sum(s, q, c):
        """(sum over pairs of conflict(q) - conflict(c) per measure, number of pairs)."""
        iq, ic = pos[s].loc[list(q)].to_numpy(), pos[s].loc[list(c)].to_numpy()
        return (V[s][iq] - V[s][ic]).sum(axis=0), len(iq)

    # ---- inputs: units, S1 pairs, S2 patches
    land = rl[(rl.n_land > 0) & rl.seed.isin(SEEDS)]
    units = [(int(r.seed), int(r.component)) for r in land.itertuples()]
    types = {u: t for u, t in zip(units, land.type)}
    KA = K[K.seed.isin(SEEDS)]
    cnt = KA.groupby(["seed", "component"]).size()
    check(len(KA) == int(land.n_land.sum()) and all(int(cnt.get(u, 0)) == n for u, n in zip(units, land.n_land)),
          f"S1 rows = the {int(land.n_land.sum())} region land points of regions_land.csv, by region")
    km = KA[KA.status == "matched"]
    ok = True
    for s in SEEDS:
        A = T[s][(T[s].half == "A") & T[s].land]
        ks, ka = km[km.seed == s], KA[KA.seed == s]
        ok &= bool(ka.point_id.isin(A.index).all() and ks.control_point_id.isin(A.index).all()
                   and ks.control_point_id.is_unique and not ks.control_point_id.isin(ka.point_id).any()
                   and np.array_equal(A.loc[ks.point_id, "label"].to_numpy(), A.loc[ks.control_point_id, "label"].to_numpy()))
    check(ok, f"{len(km)} matched S1 pairs: land Zone A test points, controls used once, outside the regions, same label")
    N = N[N.seed.isin(SEEDS)]
    n_draws = int(N.draw.max()) + 1
    check(not N.duplicated(["seed", "component", "draw"]).any(), f"S2: one patch per (region, draw), {n_draws} draws")
    obs, nul = {}, {}
    for u in units:
        k = km[(km.seed == u[0]) & (km.component == u[1])]
        for lab in LABELS:
            kk = k if lab == "all" else k[k.label == lab]
            obs[(u, lab)] = pair_sum(u[0], kk.point_id, kk.control_point_id)
            # per draw: summed differences, number of pairs, whether the unit has that draw's patch
            nul[(u, lab)] = dict(sums=np.zeros((n_draws, len(meas))), n=np.zeros(n_draws),
                                 have=np.zeros(n_draws, bool))
    bad_patch = 0
    for row in N.itertuples(index=False):
        u = (int(row.seed), int(row.component))
        if u not in types:
            continue
        q = np.asarray(row.land_point_ids, dtype=object)
        c = np.asarray(row.control_point_ids, dtype=object)
        m = np.array([isinstance(x, str) for x in c], bool)
        q, c = q[m].astype(str), c[m].astype(str)
        lq = lab_of[u[0]][pos[u[0]].loc[q].to_numpy()] if len(q) else np.zeros(0, int)
        lc = lab_of[u[0]][pos[u[0]].loc[c].to_numpy()] if len(c) else np.zeros(0, int)
        bad_patch += int(not np.array_equal(lq, lc))
        for lab in LABELS:
            sel = np.ones(len(q), bool) if lab == "all" else lq == lab
            S, n = pair_sum(u[0], q[sel], c[sel])
            patch = nul[(u, lab)]
            patch["sums"][row.draw] = S
            patch["n"][row.draw] = n
            patch["have"][row.draw] = True
    check(bad_patch == 0, f"S2: every patch point's control has its label ({bad_patch} patches with a mismatch)")

    def pooled(us, lab):
        """Per measure: the pair-weighted mean difference over the units' matched pairs."""
        n = sum(obs[(u, lab)][1] for u in us)
        if not n > 0:
            return np.full(len(meas), np.nan)
        return sum(obs[(u, lab)][0] for u in us) / n

    def null_pooled(us, lab):
        """(n_draws, measures): draw d = sum over the units' d-th patch pairs / their number (pair-weighted)."""
        S = sum(nul[(u, lab)]["sums"] for u in us)
        n = sum(nul[(u, lab)]["n"] for u in us)
        have = np.all([nul[(u, lab)]["have"] for u in us], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = S / n[:, None]
        out[~have | (n == 0)] = np.nan
        return out

    scope_list = [("all", units, "all")]
    for t in sd.TYPES:
        scope_list.append((t, [u for u in units if types[u] == t], "all"))
    for s in SEEDS:
        scope_list.append((f"seed={s}", [u for u in units if u[0] == s], "all"))
    for lb in (1, 0):
        # a region enters a label scope only if it has a matched pair of that label
        with_pair = [u for u in units if obs[(u, lb)][1] > 0]
        scope_list.append((f"label={lb}", with_pair, lb))
    scope_list = [x for x in scope_list if x[1]]
    raw_pooled = C2[(C2.analysis == "raw") & (C2.scope_kind == "pooled")]
    # rows checked, rows that differ and the largest value difference, kept apart for the
    # all / type / seed scopes and the label scopes
    tally = {"scope": dict(rows=0, bad=0, worst=0.0), "label": dict(rows=0, bad=0, worst=0.0)}
    prim = None
    for name, us, lab in scope_list:
        t = tally["scope" if lab == "all" else "label"]
        o = pooled(us, lab)
        NV = null_pooled(us, lab)
        n_pairs = int(sum(obs[(u, lab)][1] for u in us))
        rows = raw_pooled[raw_pooled.scope == name]
        n_boot = int(rows.n_boot.iloc[0]) if len(rows) else sd.N_BOOT
        M = sd.boot_multiplicities(len(us), n_boot, name)       # the module's seeded draws; the pooling is ours
        Su = np.array([obs[(u, lab)][0] for u in us])
        nu = np.array([obs[(u, lab)][1] for u in us], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = (M @ Su) / (M @ nu)[:, None]
        for j, v in enumerate(meas):
            r = rows[rows.measure == v]
            t["rows"] += 1
            if len(r) != 1:
                t["bad"] += 1
                continue
            r = r.iloc[0]
            ns = null_summary(NV[:, j])
            ci = np.nanpercentile(B[:, j], [2.5, 97.5])
            mine = dict(diff=o[j], ci_lo=ci[0], ci_hi=ci[1], null_mean=ns["null_mean"], null_p2_5=ns["null_p2_5"],
                        null_p97_5=ns["null_p97_5"])
            pct = own_midrank(o[j], NV[:, j])
            dev = max(abs(float(r[k]) - float(x)) for k, x in mine.items())
            t["worst"] = max(t["worst"], dev)
            exact = (int(r.n_pairs) == n_pairs and int(r.n_regions) == len(us) and int(r.n_null) == ns["n_null"]
                     and abs(float(r.null_percentile) - pct) < 1e-9
                     and str(r.met) == str(bool(ci[0] > 0 and pct > 95)))
            t["bad"] += int(dev > TOL_CSV or not exact)
            if name == "all" and v == f"conflict_k{sd.K_PRIMARY}":
                prim = (r, mine, pct, n_pairs, len(us), ns["n_null"], us)
    # the primary, line by line
    if prim is None:
        check(False, "primary row (raw, pooled, all, conflict_k25) found in C_conflict.csv")
    else:
        r, mine, pct, n_pairs, n_units, n_null, us = prim
        check(bool(r.primary), "the rebuilt primary's row is the one flagged primary")
        check(abs(r["diff"] - mine["diff"]) < TOL_CSV and int(r.n_pairs) == n_pairs == int((km.status == "matched").sum())
              and int(r.n_regions) == n_units == 16,
              f"PRIMARY rebuilt: {mine['diff']:+.6f} over {n_pairs} pairs in {n_units} regions "
              f"(C_conflict {r['diff']:+.6f}, {int(r.n_pairs)} pairs, {int(r.n_regions)} regions)")
        check(abs(r.null_mean - mine["null_mean"]) < TOL_CSV and abs(r.null_p2_5 - mine["null_p2_5"]) < TOL_CSV
              and abs(r.null_p97_5 - mine["null_p97_5"]) < TOL_CSV and int(r.n_null) == n_null == n_draws,
              f"primary's pair-weighted pooled null over {n_null} draws: mean {mine['null_mean']:+.6f}, "
              f"2.5/97.5% {mine['null_p2_5']:+.6f}/{mine['null_p97_5']:+.6f} (C_conflict {r.null_mean:+.6f}, "
              f"{r.null_p2_5:+.6f}/{r.null_p97_5:+.6f})")
        check(abs(r.null_percentile - pct) < 1e-9, f"primary's null percentile {pct:.2f} (C_conflict {r.null_percentile:.2f})")
        check(abs(r.ci_lo - mine["ci_lo"]) < TOL_CSV and abs(r.ci_hi - mine["ci_hi"]) < TOL_CSV,
              f"primary's region-bootstrap interval [{mine['ci_lo']:+.6f}, {mine['ci_hi']:+.6f}] "
              f"(C_conflict [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}])")
        want_met = bool(mine["ci_lo"] > 0 and pct > 95)
        check(str(r.met) == str(want_met), f"primary met = {want_met} (C_conflict {r.met})")
        # group means of the primary's row (the levels the report quotes)
        j = meas.index(f"conflict_k{sd.K_PRIMARY}")
        reg, mq, mc_, ae = [], [], [], []
        for s in SEEDS:
            reg.append(V[s][pos[s].loc[KA[KA.seed == s].point_id].to_numpy(), j])
            mq.append(V[s][pos[s].loc[km[km.seed == s].point_id].to_numpy(), j])
            mc_.append(V[s][pos[s].loc[km[km.seed == s].control_point_id].to_numpy(), j])
            ae.append(V[s][((T[s].half == "AE") & T[s].land).to_numpy(), j])
        reg, mq, mc_, ae = np.concatenate(reg), np.concatenate(mq), np.concatenate(mc_), np.concatenate(ae)
        lv = dict(region_all_mean=reg.mean(), region_matched_mean=mq.mean(), controls_mean=mc_.mean(), ae_test_mean=ae.mean())
        dv = max(abs(float(r[k]) - x) for k, x in lv.items())
        check(dv < TOL_CSV and int(r.n_region_land) == len(reg) and int(r.n_ae_test) == len(ae),
              "primary's group means: " + ", ".join(f"{k} {x:.4f}" for k, x in lv.items())
              + f" over {len(reg)} region land / {len(ae)} land Zone AE test points (max |diff| {dv:.1e})")
    n_label_scopes = sum(1 for x in scope_list if x[2] != "all")
    n_other_scopes = len(scope_list) - n_label_scopes
    t = tally["scope"]
    check(t["bad"] == 0 and t["rows"] == n_other_scopes * len(meas),
          f"all {t['rows']} pooled raw rows of the all / type / seed scopes (k = 10/25/50 x {n_other_scopes} scopes): "
          f"diff, interval, null mean/2.5/97.5, percentile, pairs, regions, met = rebuild "
          f"({t['bad']} differ; max |value diff| {t['worst']:.1e})")
    t = tally["label"]
    label_ok = t["bad"] == 0 and t["rows"] == n_label_scopes * len(meas)
    msg = (f"all {t['rows']} pooled raw rows of the label scopes (k = 10/25/50 x {n_label_scopes} scopes, regions "
           f"with a pair of that label): the same columns = rebuild ({t['bad']} differ; max |value diff| {t['worst']:.1e})")
    if not label_ok:
        msg += "; stored C outputs predate the label-scope fix; rerun docs/TASK-O5-REGIONS.md Part C"
    check(label_ok, msg)
    reg_rows = C2[(C2.analysis == "raw") & (C2.scope_kind == "region")]
    n_bad, worst_v = 0, 0.0
    for u in units:
        NV = null_pooled([u], "all")
        o = pooled([u], "all")
        for j, v in enumerate(meas):
            r = reg_rows[(reg_rows.seed == u[0]) & (reg_rows.component == u[1]) & (reg_rows.measure == v)]
            if len(r) != 1:
                n_bad += 1
                continue
            r = r.iloc[0]
            ns = null_summary(NV[:, j])
            dev = max(abs(float(r["diff"]) - o[j]), abs(float(r.null_mean) - ns["null_mean"]))
            worst_v = max(worst_v, dev)
            n_bad += int(dev > TOL_CSV or abs(float(r.null_percentile) - own_midrank(o[j], NV[:, j])) > 1e-9
                         or int(r.n_null) != ns["n_null"])
    check(n_bad == 0, f"all {len(units) * len(meas)} per-region raw rows: diff, null mean, null percentile = rebuild "
          f"({n_bad} differ; max |value diff| {worst_v:.1e})")


def test_c3_cells(out_dir: Path, o5_dir: Path):
    print("11. C3 counts and accuracies rebuilt from the cells (all seeds)")
    f = out_dir / "C_decomposition.csv"
    if real_inputs(o5_dir) is None or not f.exists():
        print("  skip  inputs or C_decomposition.csv missing")
        return
    C3 = pd.read_csv(f, dtype={"seed": str})
    K, rl = REAL["K"], REAL["rl"]
    acc_ae_seed, n_land_seed = {}, {}
    for s in SEEDS:
        d = cell_rows(s)
        ide, od = d["id_te"], d["ood_te"]
        z = np.load(CELLS / f"{PREFIX}_s{s}.npz", allow_pickle=True)
        check(np.array_equal(z["point_id_id"].astype(str), ide.point_id.astype(str).to_numpy())
              and np.array_equal(z["point_id"].astype(str), od.point_id.astype(str).to_numpy())
              and np.array_equal(z["y_id"], ide.label.to_numpy()) and np.array_equal(z["y_ood"], od.label.to_numpy()),
              f"seed {s}: the cell's columns = zoo.cell_data's test rows")
        land_ae = ide.landcover.to_numpy() != WATER
        n_land = int(land_ae.sum())
        n_land_seed[s] = n_land
        rows = C3[C3.seed == str(s)]
        check(len(rows) > 0 and bool((rows.n_ae_test_land == n_land).all()),
              f"seed {s}: n_ae_test_land = {n_land} land Zone AE test points (of {len(ide)}) in all {len(rows)} rows "
              f"(C_decomposition: {sorted(set(rows.n_ae_test_land.astype(int)))})")
        models = pd.read_csv(CELLS / f"{PREFIX}_s{s}_models.csv")
        top, te = own_top15(models)
        check(models.auc_id.iloc[top].min() > models.auc_id.iloc[np.setdiff1d(te, top)].max(),
              f"seed {s}: top 15 by auc_id {top.tolist()} (no tie at the 15th)")
        Zi, Zo = own_correct(z["S_id"], z["y_id"]), own_correct(z["S_ood"], z["y_ood"])
        po = pd.Series(np.arange(len(od)), index=od.point_id.astype(str).to_numpy())
        Ks = K[K.seed == s]
        kms = Ks[Ks.status == "matched"]
        rl_s = rl[(rl.seed == s) & (rl.n_land > 0)]
        targets = {"all": rl_s.component.tolist()}
        for t in sd.TYPES:
            targets[t] = rl_s[rl_s.type == t].component.tolist()
        for c in rl_s.component:
            targets[f"region_c{c}"] = [c]
        worst, n_bad = 0.0, 0
        for ms, idx in (("top15", top), ("test60", te)):
            acc_ae = float(Zi[idx][:, land_ae].mean())
            acc_ae_seed[(s, ms)] = acc_ae
            for tname, comps in targets.items():
                if not comps:
                    continue
                r = rows[(rows.target == tname) & (rows.model_set == ms)]
                if len(r) != 1:
                    n_bad += 1
                    continue
                r = r.iloc[0]
                reg = po.loc[Ks[Ks.component.isin(comps)].point_id].to_numpy()
                ctl = po.loc[kms[kms.component.isin(comps)].control_point_id].to_numpy()
                mine = dict(acc_ae=acc_ae, acc_region=float(Zo[idx][:, reg].mean()),
                            acc_controls=float(Zo[idx][:, ctl].mean()) if len(ctl) else np.nan)
                dev = max(abs(float(r[k]) - x) for k, x in mine.items() if np.isfinite(x))
                worst = max(worst, dev)
                n_bad += int(dev > TOL_CSV or int(r.n_region_land) != len(reg) or int(r.n_controls) != len(ctl)
                             or int(r.n_models) != len(idx))
        check(n_bad == 0, f"seed {s}: acc_ae, acc_region, acc_controls, n_region_land, n_controls of every "
              f"(target, model set) row = own correctness (top15: acc_ae {acc_ae_seed[(s, 'top15')]:.6f}; "
              f"{n_bad} differ; max |diff| {worst:.1e})")
    pooled = C3[C3.seed == "pooled"]
    n_bad = 0
    for _, r in pooled.iterrows():
        ss = [int(x) for x in str(r.seeds_pooled).split(";")]
        n_bad += int(int(r.n_ae_test_land) != sum(n_land_seed[s] for s in ss)
                     or abs(r.acc_ae - np.mean([acc_ae_seed[(s, r.model_set)] for s in ss])) > TOL_CSV)
    check(len(pooled) > 0 and n_bad == 0,
          f"pooled rows: n_ae_test_land = {sum(n_land_seed.values())} (summed over seeds), acc_ae = mean over seeds "
          f"({n_bad} of {len(pooled)} differ)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results/2026-10-17/o5"))
    ap.add_argument("--o5-dir", type=Path, default=Path("results/2026-10-17/o5"))
    args = ap.parse_args()
    test_feature_space()
    test_knn()
    test_midrank()
    test_boot()
    test_pairs()
    test_c3()
    test_crossfit()
    test_real(args.o5_dir)
    test_outputs(args.out_dir, args.o5_dir)
    test_rebuild_c2(args.out_dir, args.o5_dir)
    test_c3_cells(args.out_dir, args.o5_dir)
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
