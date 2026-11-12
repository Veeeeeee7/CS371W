#!/usr/bin/env python3
"""
Checks for zoo.py (the grid population of docs/TASK-ZOO-GRIDS-2026-11-06.md, with the one-hot
inputs and the duplicate drop of docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D5 and D6) and the
parts of audit_cell.py that check it. Run from the repo root:

    .venv/bin/python aotil/test_zoo.py [--cells data/interim/aotil/cells_v4_2026-11-07 \
        --data data/processed_v4]

1. The grids: every family's size and order (families in sorted code-name order, settings in
   itertools.product order, first axis slowest), the brief's values, 369 settings in all; without
   xgboost the other families keep their settings and seeds.
2. The specs: grid_idx 0..368, every (family, setting) unique, the seeds drawn in grid order from
   default_rng(cell seed), the same specs on every call.
3. build: lr no longer fixes max_iter (the grid sets it), nb takes its grid setting, rf/et/hgb/
   xgb keep their fixed settings.
4. The drop rule on synthetic predictions: predicts_flood (a tie at p = 0.5 is dry, at either
   label), is_degenerate with ties exactly at the 5% and 95% cuts kept, one point inside dropped.
5. split_population: unusable and degenerate settings are dropped, then every model whose 0/1
   predictions on the ID test points and the OOD points equal those of a model earlier in grid
   order ("duplicate", the first in grid order kept, duplicate_of naming it); a model that agrees
   on one half only is kept; degenerate models never count as the first; the order is the grid's.
5b. model_inputs (D5): one 0/1 column per class of each categorical feature, from the full
   dataset (fixed and complete: a class absent from a subset keeps its column, no reference
   column dropped), NaN in all of a feature's columns where it is missing, numeric features as
   they are, in FEATURES.txt order; a class that is not a whole number or a clashing column raises.
5c. hgb and xgb receive the NaN of the one-hot columns as is; every other family has the median
   imputer first.
6. fit_one: all training rows and every varying feature (a constant column is dropped); the
   flood count equals predicts_flood on its ID scores.
7. run_cell and main on a synthetic dataset with categorical features: the cell holds the kept
   models in grid order (idx 0..N-1, grid_idx increasing), kept + dropped = the grid, S rows = the
   table's means, every model on the one-hot input columns, no two kept models with the same
   predictions; --n-models is refused.
8. The model split over populations that are not 300: model_split(360) gives 216/72/72 and
   every size is a disjoint cover; the family split covers every model.
9. audit_cell.py on the synthetic cell: PASS (every dropped setting, duplicates included, refits
   as stored); FAIL for a kept degenerate model, a kept duplicate, a missing or incomplete dropped
   table, a changed seed, and a table without grid_idx.
10. The real cell (skipped when missing): 369 settings = kept + dropped, no kept model degenerate
    or a duplicate, every model on all the input columns of model_inputs and all training rows.
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op  # noqa: E402
import zoo  # noqa: E402

FAILS = []
PY = sys.executable
REPO = Path(__file__).resolve().parent.parent
# The brief's grids, written out again here so that a change to zoo.GRIDS shows up.
BRIEF = {
    "dt": [("max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 12, None]),
           ("min_samples_leaf", [1, 20, 200, 2000])],
    "rf": [("n_estimators", [10, 30, 100]), ("max_depth", [1, 2, 3, 5, 8, None]),
           ("max_features", [1, "sqrt", None])],
    "et": [("n_estimators", [10, 30, 100]), ("max_depth", [1, 2, 3, 5, 8, None]),
           ("max_features", [1, "sqrt", None])],
    "hgb": [("learning_rate", [0.01, 0.03, 0.1, 0.5]), ("max_iter", [3, 10, 50, 300]),
            ("max_depth", [1, 3, None])],
    "xgb": [("learning_rate", [0.01, 0.03, 0.1, 0.5]), ("n_estimators", [3, 10, 50, 300]),
            ("max_depth", [1, 3, 8])],
    "lr": [("C", [0.001, 0.01, 0.1, 1, 100]), ("max_iter", [1, 2, 3, 5, 100]),
           ("class_weight", [None, "balanced"])],
    "mlp": [("hidden_layer_sizes", [(2,), (8,), (16,), (32,), (64, 32)]),
            ("max_iter", [3, 20, 200]), ("learning_rate_init", [1e-4, 1e-3, 1e-2])],
    "nb": [("var_smoothing", [1e-9, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5, 1,
                              2, 3, 10]),
           ("priors", [None, (0.5, 0.5)])],
}
COUNTS = {"dt": 40, "et": 54, "hgb": 48, "lr": 50, "mlp": 45, "nb": 30, "rf": 54, "xgb": 48}
FEATURES = ["f_a", "f_b", "f_c", "f_d", "f_const"]


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


# ----------------------------------------------------------------- 1-3: grids, specs, build
def check_grids():
    print("1. grids")
    settings = zoo.grid_settings()
    check(list(settings) == sorted(COUNTS), f"families in sorted code-name order: {list(settings)}")
    check({k: len(v) for k, v in settings.items()} == COUNTS,
          f"grid sizes {({k: len(v) for k, v in settings.items()})}")
    check(sum(len(v) for v in settings.values()) == 369, "369 settings in all")
    for kind, axes in BRIEF.items():
        names = [name for name, _ in axes]
        want = [dict(zip(names, combo)) for combo in itertools.product(*[v for _, v in axes])]
        check(settings[kind] == want,
              f"{kind}: the brief's axes, product order, first axis slowest")
    check(settings["dt"][:5] == [dict(max_depth=1, min_samples_leaf=m) for m in (1, 20, 200, 2000)]
          + [dict(max_depth=2, min_samples_leaf=1)], "dt: the leaf size cycles fastest")
    with mock.patch.object(zoo, "_HAS_XGB", False):
        no_xgb = zoo.zoo_specs(0)
    with_xgb = zoo.zoo_specs(0)
    check(len(no_xgb) == 321 and "xgb" not in {sp["kind"] for sp in no_xgb},
          "without xgboost: 321 settings, no xgb")
    check(no_xgb == with_xgb[:321], "without xgboost the other families keep settings and seeds")


def check_specs():
    print("2. specs")
    specs = zoo.zoo_specs(0)
    check([sp["grid_idx"] for sp in specs] == list(range(369)), "grid_idx 0..368 in order")
    keys = {(sp["kind"], repr(sorted(sp["kw"].items()))) for sp in specs}
    check(len(keys) == 369, "every (family, setting) unique")
    rng = np.random.default_rng(0)
    seeds = [int(rng.integers(1 << 30)) for _ in range(369)]
    check([sp["seed"] for sp in specs] == seeds, "seeds drawn in grid order from default_rng(0)")
    check(zoo.zoo_specs(0) == specs, "the same specs on every call")
    check([sp["seed"] for sp in zoo.zoo_specs(1)] != seeds, "another cell seed, other seeds")
    check(set(specs[0]) == {"grid_idx", "kind", "kw", "seed"},
          f"a spec holds grid_idx, kind, kw and seed only: {sorted(specs[0])}")


def check_build():
    print("3. build")
    lr = zoo.build("lr", dict(C=0.1, max_iter=3, class_weight="balanced"), 7)[-1]
    check(lr.max_iter == 3 and lr.C == 0.1 and lr.class_weight == "balanced"
          and lr.random_state == 7, "lr takes the grid's max_iter, C and class_weight")
    check(zoo.build("lr", dict(C=1), 7)[-1].max_iter == 100,
          "lr without max_iter keeps the library default (100), not 400")
    nb = zoo.build("nb", dict(var_smoothing=0.3, priors=(0.5, 0.5)), 7)[-1]
    check(nb.var_smoothing == 0.3 and tuple(nb.priors) == (0.5, 0.5), "nb takes its setting")
    rf = zoo.build("rf", dict(n_estimators=10, max_depth=None, max_features=1), 7)[-1]
    check(rf.n_jobs == 1 and rf.max_features == 1 and rf.bootstrap, "rf: n_jobs 1, bootstrap")
    et = zoo.build("et", dict(n_estimators=10, max_depth=1, max_features="sqrt"), 7)[-1]
    check(et.n_jobs == 1 and not et.bootstrap, "et: n_jobs 1, its default bootstrap (off)")
    hgb = zoo.build("hgb", dict(learning_rate=0.5, max_iter=3, max_depth=None), 7)
    check(hgb.early_stopping is False, "hgb: early stopping off")
    if zoo._HAS_XGB:
        xgb = zoo.build("xgb", dict(learning_rate=0.1, n_estimators=3, max_depth=8), 7)
        check(xgb.tree_method == "hist" and xgb.n_jobs == 1 and xgb.device == "cpu",
              "xgb: hist, n_jobs 1, cpu")


# ----------------------------------------------------------------- 4-6: drop rule, fits
def check_drop_rule():
    print("4. drop rule")
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.5, 0.5, 0.2, 0.51, 0.7])
    S = np.where(y == 1, p, 1.0 - p)
    got = zoo.predicts_flood(S, y)
    check(got.tolist() == (p > 0.5).tolist(), f"predicts_flood = p > 0.5, ties dry: {got.tolist()}")
    for n_flood, n, want in [(1, 20, False), (0, 20, True), (19, 20, False), (20, 20, True),
                             (2, 40, False), (1, 40, True), (38, 40, False), (39, 40, True),
                             (542, 10835, False), (541, 10835, True),
                             (10293, 10835, False), (10294, 10835, True)]:
        check(zoo.is_degenerate(n_flood, n) == want,
              f"is_degenerate({n_flood}, {n}) = {want} (share {n_flood / n:.4f})")
    try:
        zoo.is_degenerate(0, 0)
        check(False, "no ID test points raises")
    except ValueError:
        check(True, "no ID test points raises")


def fake_fit(kind, grid_idx, s_id, y):
    """A fit_one result with given ID scores (only what split_population reads)."""
    sp = dict(grid_idx=grid_idx, kind=kind, kw=dict(max_depth=grid_idx), seed=100 + grid_idx)
    return dict(sp=sp, s_id=np.asarray(s_id, float), s_od=np.asarray(s_id, float),
                n_flood_id=int(zoo.predicts_flood(s_id, y).sum()),
                id=(0.7, 0.8), ood=(0.6, 0.7))


def check_split_population():
    print("5. split_population")
    y = np.array([1] * 10 + [0] * 10)
    good = np.r_[np.full(10, 0.8), np.full(10, 0.7)]       # 10 of 20 flood
    other = np.r_[np.full(10, 0.8), np.full(5, 0.7), np.full(5, 0.3)]   # 15 of 20 flood
    all_flood = np.r_[np.full(10, 0.8), np.full(10, 0.2)]  # 20 of 20 flood: degenerate
    edge = np.r_[np.full(1, 0.9), np.full(9, 0.1), np.full(10, 0.9)]   # 1 of 20 = 5%: kept
    fits = [fake_fit("dt", 0, good, y), fake_fit("dt", 1, good * 0.99 + 0.005, y), None,
            fake_fit("dt", 3, all_flood, y), fake_fit("dt", 4, edge, y),
            fake_fit("dt", 5, all_flood, y), fake_fit("dt", 6, other, y),
            fake_fit("dt", 7, good, y)]
    # grid 8: the ID predictions of grid 0, other OOD predictions -> not a duplicate.
    split_ood = fake_fit("dt", 8, good, y)
    split_ood["s_od"] = other
    fits.append(split_ood)
    specs = [dict(grid_idx=i, kind="dt", kw=dict(max_depth=i), seed=100 + i) for i in range(9)]
    kept, dropped = zoo.split_population(specs, fits, y, y)
    check([got["sp"]["grid_idx"] for got in kept] == [0, 4, 6, 8],
          f"kept in grid order: the first of the repeats, the one at exactly 5%, a model that "
          f"agrees on the ID half only: {[got['sp']['grid_idx'] for got in kept]}")
    reasons = [(row["grid_idx"], row["reason"]) for row in dropped]
    check(reasons == [(1, "duplicate"), (2, "unusable"), (3, "degenerate"), (5, "degenerate"),
                      (7, "duplicate")], f"dropped: {reasons}")
    check(dropped[0]["duplicate_of"] == 0 and dropped[4]["duplicate_of"] == 0,
          "a duplicate names the first model in grid order with its predictions (other scores, "
          "the same 0/1 predictions, still repeat it)")
    check(np.isnan(dropped[2]["duplicate_of"]) and dropped[3]["reason"] == "degenerate",
          "a degenerate model is dropped as degenerate, never as a duplicate of another")
    check(dropped[2]["flood_share_id"] == 1.0 and np.isnan(dropped[1]["flood_share_id"]),
          "a degenerate row carries its flood share; an unusable one NaN")
    check(dropped[0]["flood_share_id"] == 0.5 and dropped[2]["hp_max_depth"] == 3,
          "a duplicate carries its flood share; dropped rows carry the setting as hp_<name>")
    key = zoo.prediction_key(fits[0], y, y)
    check(key == zoo.prediction_key(fits[1], y, y) and key != zoo.prediction_key(split_ood, y, y),
          "prediction_key: equal for the same 0/1 predictions on both halves only")
    table = zoo.family_counts(specs, kept, dropped)
    check(table.loc["dt"].to_dict() == dict(settings=9, degenerate=2, duplicate=2, unusable=1,
                                           kept=4), f"family_counts: {table.loc['dt'].to_dict()}")


def check_onehot():
    print("5b. model_inputs (one-hot)")
    df = pd.DataFrame({"f_a": [0.5, 1.5, np.nan, 2.0], "landcover": [90.0, 42.0, 90.0, 21.0],
                       "soil_floodfreq": [0.0, np.nan, 5.0, 0.0],
                       "soil_fluvclass": [1.0, 0.0, np.nan, 0.0], "other": [1, 2, 3, 4]})
    feats = ["landcover", "f_a", "soil_floodfreq", "soil_fluvclass"]
    out, inputs = zoo.model_inputs(df, feats)
    check(inputs == ["landcover_21", "landcover_42", "landcover_90", "f_a", "soil_floodfreq_0",
                     "soil_floodfreq_5", "soil_fluvclass_0", "soil_fluvclass_1"],
          f"one column per class present, sorted, in FEATURES order, none dropped: {inputs}")
    lc = out[["landcover_21", "landcover_42", "landcover_90"]].to_numpy()
    check((lc.sum(1) == 1).all() and lc[0].tolist() == [0, 0, 1], "each row one 1 per feature")
    soil = out[["soil_floodfreq_0", "soil_floodfreq_5"]].to_numpy()
    check(np.isnan(soil[1]).all() and np.isnan(out[["soil_fluvclass_0", "soil_fluvclass_1"]]
                                               .to_numpy()[2]).all()
          and not np.isnan(soil[[0, 2, 3]]).any(),
          "a missing feature is NaN in all of its columns, and only there")
    check(out.f_a.equals(df.f_a) and np.isnan(out.f_a[2]) and out.other.tolist() == [1, 2, 3, 4],
          "numeric features and every other column as they were")
    sub, sub_inputs = zoo.model_inputs(df.iloc[:2], feats)
    full_levels = zoo.onehot_levels(df, feats)
    check(full_levels["landcover"] == [21.0, 42.0, 90.0] and "landcover_21" not in sub_inputs,
          "the classes come from the frame given (zoo.load passes the full dataset)")
    X = out.loc[[3], inputs].to_numpy(float)
    check(X[0, inputs.index("landcover_21")] == 1 and X[0, inputs.index("soil_floodfreq_0")] == 1,
          "a class seen in one row only still has its column")
    try:
        zoo.model_inputs(df.assign(landcover=[90.5, 42, 90, 21]), feats)
        check(False, "a class that is not a whole number raises")
    except ValueError:
        check(True, "a class that is not a whole number raises")
    try:
        zoo.model_inputs(df.assign(landcover_90=1.0), feats)
        check(False, "a clash with an existing column raises")
    except ValueError:
        check(True, "a clash with an existing column raises")
    no_cat, plain = zoo.model_inputs(df, ["f_a"])
    check(plain == ["f_a"] and list(no_cat.columns) == list(df.columns),
          "no categorical feature: the inputs are the features, nothing added")


def check_nan_inputs():
    print("5c. NaN reaches hgb and xgb; the other families impute")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"landcover": rng.choice([21.0, 42.0, 90.0], 300),
                       "soil_floodfreq": np.where(rng.random(300) < 0.2, np.nan,
                                                  rng.integers(0, 3, 300).astype(float)),
                       "f_a": rng.normal(0, 1, 300)})
    df, inputs = zoo.model_inputs(df, ["landcover", "soil_floodfreq", "f_a"])
    X = df[inputs].to_numpy(float)
    y = (rng.random(300) < 0.5).astype(int)
    A = dict(Xtr=X, ytr=y, Xid=X[:100], yid=y[:100], Xod=X[100:200], yod=y[100:200])
    seen = {}
    real_build = zoo.build

    def spy_build(kind, kw, seed, device="cpu"):
        model = real_build(kind, kw, seed, device)
        fit = model.fit

        def recording_fit(Xs, ys, *args, **kwargs):
            seen[kind] = np.isnan(Xs).any()
            return fit(Xs, ys, *args, **kwargs)
        model.fit = recording_fit
        return model

    kinds = ["hgb", "dt", "lr"] + (["xgb"] if zoo._HAS_XGB else [])
    kws = dict(hgb=dict(learning_rate=0.1, max_iter=5, max_depth=3), dt=dict(max_depth=3),
               lr=dict(C=1, max_iter=100), xgb=dict(learning_rate=0.1, n_estimators=5,
                                                     max_depth=3))
    with mock.patch.object(zoo, "build", spy_build):
        for kind in kinds:
            got = zoo.fit_one(dict(grid_idx=0, kind=kind, kw=kws[kind], seed=1), A)
            check(got is not None, f"{kind} fits on the one-hot inputs with NaN")
    check(seen.get("hgb") and (seen.get("xgb") or not zoo._HAS_XGB),
          "hgb and xgb receive the NaN of the missing soil class as is")
    for kind in ("dt", "rf", "et", "lr", "mlp", "nb"):
        first = zoo.build(kind, {}, 1).steps[0][1]
        check(type(first).__name__ == "SimpleImputer" and first.strategy == "median",
              f"{kind}: the training-median imputer comes first")


def synthetic_arrays(seed=5, n=400):
    """cell_data-like arrays with a constant column and a few NaNs."""
    rng = np.random.default_rng(seed)
    out = {}
    for half, size in (("tr", n), ("id", n // 2), ("od", n // 2)):
        y = rng.integers(0, 2, size)
        X = np.column_stack([rng.normal(y, 1.0), rng.normal(0, 1, size), rng.normal(-y, 2.0),
                             rng.integers(0, 3, size).astype(float), np.ones(size)])
        X[rng.choice(size, 5, replace=False), 2] = np.nan
        out[f"X{half}"], out[f"y{half}"] = X, y
    return out


def check_fit_one():
    print("6. fit_one")
    A = synthetic_arrays()
    for sp in (dict(grid_idx=0, kind="lr", kw=dict(C=1, max_iter=100, class_weight=None), seed=3),
               dict(grid_idx=1, kind="hgb", kw=dict(learning_rate=0.1, max_iter=10, max_depth=3),
                    seed=4)):
        got = zoo.fit_one(sp, A)
        check(got is not None and got["cols"].tolist() == [0, 1, 2, 3],
              f"{sp['kind']}: every varying column, the constant one dropped")
        check(got["n_train"] == len(A["Xtr"]), f"{sp['kind']}: n_train = all {len(A['Xtr'])} rows")
        check(got["n_flood_id"] == int(zoo.predicts_flood(got["s_id"], A["yid"]).sum()),
              f"{sp['kind']}: n_flood_id = predicts_flood on its ID scores")
        row = zoo.model_row(got, FEATURES, idx=7)
        check(row["idx"] == 7 and row["grid_idx"] == sp["grid_idx"]
              and row["features"] == "f_a;f_b;f_c;f_d"
              and row["flood_share_id"] == got["n_flood_id"] / len(A["yid"])
              and "feat_frac" not in row and "row_frac" not in row,
              f"{sp['kind']}: model_row idx, grid_idx, features, flood share, no feat/row draws")
        check(row["acc_train_sub"] == row["acc_train_all"],
              f"{sp['kind']}: without extra rows the two training accuracies are the same rows")
    one_label = dict(A, ytr=np.zeros_like(A["ytr"]))
    check(zoo.fit_one(dict(grid_idx=0, kind="lr", kw={}, seed=1), one_label) is None,
          "one training label: None")
    two_const = dict(A, Xtr=np.column_stack([A["Xtr"][:, 0], np.ones((len(A["ytr"]), 4))]))
    check(zoo.fit_one(dict(grid_idx=0, kind="lr", kw={}, seed=1), two_const) is None,
          "fewer than two varying columns: None")


# ----------------------------------------------------------------- 7: run_cell, main
def synthetic_dataset(seed=11) -> pd.DataFrame:
    """20 tiles of both zones, both protocols; features with signal, NaNs and a constant."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for t in range(20):
        for dom, lab, proto, n in (("AE", 1, None, 25), ("AE", 0, "band", 25),
                                   ("A", 1, None, 15), ("A", 0, "band", 15)):
            for _ in range(n):
                rows.append(dict(point_id=f"P{pid:05d}", lon=-82 + t * 0.3 + rng.uniform(0, .2),
                                 lat=27 + (t % 4) * 0.3 + rng.uniform(0, .2),
                                 tile_50km=f"{t // 4}_{t % 4}", domain=dom, label=lab,
                                 neg_protocol=proto, block_id=f"B{pid % 50}",
                                 county_fips=f"12{t:03d}", meta_comid=float(pid),
                                 meta_source_cit="FEMA"))
                pid += 1
    df = pd.DataFrame(rows)
    y, is_a = df.label.to_numpy(), (df.domain == "A").to_numpy()
    n = len(df)
    df["f_a"] = rng.normal(0, 1, n) + 1.2 * y
    df["f_b"] = rng.normal(0, 1, n) - 0.8 * y * (1 - 0.7 * is_a)
    df["f_c"] = rng.gamma(2.0, 1.0, n) + 0.3 * y
    df["f_d"] = rng.integers(0, 4, n).astype(float)
    df["f_const"] = 1.0
    df.loc[rng.choice(n, 30, replace=False), "f_b"] = np.nan
    df["landcover"] = np.where(y == 1, rng.choice([90.0, 95.0, 42.0], n),
                               rng.choice([42.0, 81.0, 21.0], n))
    df["soil_floodfreq"] = np.where(rng.random(n) < 0.1, np.nan,
                                    np.where(y == 1, rng.choice([0.0, 4.0, 5.0], n), 0.0))
    return df


def write_data(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "FEATURES.txt").write_text(
        "# synthetic\nlandcover\nf_a f_b\nf_c\nf_d f_const\nsoil_floodfreq\n")
    synthetic_dataset().to_parquet(folder / "dataset_FL.parquet")
    return folder


def thread_env() -> dict:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    return env


def run(args: list) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-u"] + args, cwd=REPO, env=thread_env(), capture_output=True,
                          text=True)


def check_run_cell(tmp: Path) -> Path:
    print("7. run_cell and main")
    data = write_data(tmp / "data")
    df, feats = zoo.load("FL", data)
    cell = zoo.run_cell(df, feats, "band", "AE", 0, n_jobs=2)
    models, dropped = cell["models"], cell["dropped"]
    check(models.idx.tolist() == list(range(len(models))), "idx = 0..N-1")
    check((np.diff(models.grid_idx) > 0).all(), "kept models in grid order")
    check(sorted(models.grid_idx.tolist() + dropped.grid_idx.tolist()) == list(range(369)),
          f"kept {len(models)} + dropped {len(dropped)} = the 369 settings, once each")
    check(cell["S_id"].shape == (len(models), len(cell["y_id"]))
          and cell["S_ood"].shape == (len(models), len(cell["y_ood"])), "S rows = kept models")
    check(np.allclose(cell["S_id"].mean(1), models.score_id, atol=1e-6)
          and np.allclose(cell["S_ood"].mean(1), models.score_ood, atol=1e-6),
          "each S row is its table row's model (score means)")
    shares = [zoo.predicts_flood(s, cell["y_id"]).sum() for s in cell["S_id"]]
    check(not any(zoo.is_degenerate(int(k), len(cell["y_id"])) for k in shares),
          "no kept model is degenerate")
    check(set(dropped.reason) <= {"degenerate", "duplicate"} and len(dropped) > 0
          and (dropped.reason == "duplicate").any(),
          f"the weak settings drop degenerate and duplicate models "
          f"({dropped.reason.value_counts().to_dict()})")
    keys = [zoo.prediction_key(dict(s_id=a, s_od=b), cell["y_id"], cell["y_ood"])
            for a, b in zip(cell["S_id"], cell["S_ood"])]
    check(len(set(keys)) == len(keys), "no two kept models with the same predictions")
    dup = dropped[dropped.reason == "duplicate"]
    check(dup.duplicate_of.isin(models.grid_idx).all()
          and (dup.duplicate_of < dup.grid_idx).all(),
          "each duplicate names a kept model earlier in grid order")
    specs = zoo.zoo_specs(0)
    check(all(int(r.seed) == specs[int(r.grid_idx)]["seed"] for r in models.itertuples()),
          "each row carries its setting's seed")
    tr = cell_rows(df, feats)
    varying = [feats[j] for j in zoo.varying_columns(tr[feats].to_numpy(float))]
    check(set(models.n_train) == {len(tr)} and set(models.n_feat) == {len(varying)}
          and set(models.features) == {";".join(varying)}
          and any(c.startswith("landcover_") for c in varying)
          and "f_const" not in varying and "landcover" not in varying,
          f"every model: all training rows, the {len(varying)} varying input columns "
          f"(one-hot, the constant one dropped)")
    serial = zoo.run_cell(df, feats, "band", "AE", 0, n_jobs=1)
    check(np.array_equal(serial["S_id"], cell["S_id"]) and serial["models"].equals(models),
          "n_jobs 1 gives the same cell as n_jobs 2")

    out = tmp / "cells"
    got = run(["aotil/zoo.py", "--state", "FL", "--protocol", "band", "--train-dom", "AE",
               "--seed", "0", "--data", str(data), "--out", str(out), "--n-jobs", "2"])
    tag = "FL_band_AE_s0"
    check(got.returncode == 0 and (out / f"{tag}_dropped.csv").exists(),
          "main writes the cell, the model table and the dropped table")
    written = pd.read_csv(out / f"{tag}_models.csv")
    check(len(written) == len(models) and written.grid_idx.tolist() == models.grid_idx.tolist(),
          "the written table is run_cell's")
    refused = run(["aotil/zoo.py", "--data", str(data), "--out", str(tmp / "x"),
                   "--n-models", "300"])
    check(refused.returncode == 2 and "--n-models" in refused.stderr,
          "--n-models is refused (argparse error)")
    return out


def cell_rows(df, feats):
    return zoo.cell_data(df, feats, "band", "AE", 0)["tr"]


# ----------------------------------------------------------------- 8: model split
def check_model_split():
    print("8. model split over other population sizes")
    sel, val, te = op.model_split(360)
    check((len(sel), len(val), len(te)) == (216, 72, 72),
          f"360 models: {len(sel)}/{len(val)}/{len(te)}")
    for n in (300, 311, 360, 369):
        parts = op.model_split(n)
        joined = np.concatenate(parts)
        check(len(np.unique(joined)) == n and sorted(joined.tolist()) == list(range(n)),
              f"{n} models: selection, validation and test cover every model once")
    kinds = np.array([sp["kind"] for sp in zoo.zoo_specs(0)])[::3]
    for rule in ("trees->others", "others->trees"):
        parts = op.make_split(len(kinds), rule, kinds, 0)
        joined = np.concatenate(parts)
        check(sorted(joined.tolist()) == list(range(len(kinds))),
              f"{rule} on {len(kinds)} grid models covers every model once")


# ----------------------------------------------------------------- 9: audit
def audit(cells: Path, data: Path, out: Path) -> tuple[int, dict]:
    got = run(["aotil/audit_cell.py", "--cells-dir", str(cells), "--tag", "FL_band_AE_s0",
               "--data", str(data), "--features", str(data / "FEATURES.txt"),
               "--out", str(out), "--overwrite"])
    report = json.loads(out.read_text()) if out.exists() else {}
    return got.returncode, report


def failed_checks(report: dict) -> list[str]:
    return [row["check"] for row in report.get("checks", []) if row["status"] == "FAIL"]


def damaged_copy(tmp: Path, cells: Path, name: str) -> Path:
    folder = tmp / name
    shutil.copytree(cells, folder)
    return folder


def check_audit_helpers():
    """stored_flood_count and grid_partition_problems on hand-made inputs."""
    import audit_cell
    n = 10835
    share = float(pd.read_csv(pd.io.common.StringIO(
        pd.DataFrame(dict(s=[542 / n])).to_csv(index=False))).s.iloc[0])
    check(audit_cell.stored_flood_count(share, n) == 542,
          f"stored_flood_count reads a CSV round-tripped share back as 542 ({share!r})")
    check(audit_cell.stored_flood_count(0.5 / n, n) is None
          and audit_cell.stored_flood_count(float("nan"), n) is None,
          "stored_flood_count: a share that is no count over n, or NaN, gives None")
    kept = pd.DataFrame(dict(grid_idx=[0, 2, 3]))
    check(audit_cell.grid_partition_problems(kept, pd.DataFrame(dict(grid_idx=[1])), 4) == [],
          "grid_partition_problems: kept 0, 2, 3 and dropped 1 cover a 4-setting grid")
    bad = audit_cell.grid_partition_problems(kept, pd.DataFrame(dict(grid_idx=[2])), 4)
    check(any("both kept and dropped" in b for b in bad)
          and any("neither kept nor dropped" in b for b in bad),
          f"grid_partition_problems: an overlap and a gap are both named {bad}")
    bad = audit_cell.grid_partition_problems(pd.DataFrame(dict(grid_idx=[3, 0, 2])),
                                             pd.DataFrame(dict(grid_idx=[1])), 4)
    check(bad == ["model table not in grid order"], f"grid_partition_problems: order {bad}")


def check_audit(tmp: Path, cells: Path):
    print("9. audit_cell.py")
    check_audit_helpers()
    data = tmp / "data"
    tag = "FL_band_AE_s0"
    code, report = audit(cells, data, tmp / "audit_clean.json")
    check(code == 0 and report.get("verdict") == "PASS",
          f"clean cell: PASS {failed_checks(report)}")
    refits = report.get("dropped_refits", [])
    check(len(refits) > 0 and all(r["ok"] for r in refits)
          and any(r["reason"] == "duplicate" for r in refits),
          f"every dropped setting refits as stored, duplicates included "
          f"({sum(r['reason'] == 'duplicate' for r in refits)} duplicates)")

    # A degenerate model kept: copy a dropped setting's row into the table without its scores
    # would break the shapes, so instead make a kept model's ID scores predict flood everywhere.
    folder = damaged_copy(tmp, cells, "kept_degenerate")
    npz = dict(np.load(folder / f"{tag}.npz"))
    y = npz["y_id"]
    npz["S_id"] = npz["S_id"].copy()
    npz["S_id"][0] = np.where(y == 1, 0.9, 0.1)
    np.savez_compressed(folder / f"{tag}.npz", **npz)
    code, report = audit(folder, data, tmp / "audit_degenerate.json")
    check(code == 1 and "drop rule" in failed_checks(report),
          f"a kept degenerate model fails the drop rule: {failed_checks(report)}")

    folder = damaged_copy(tmp, cells, "kept_duplicate")
    npz = dict(np.load(folder / f"{tag}.npz"))
    for half in ("S_id", "S_ood"):
        npz[half] = npz[half].copy()
        npz[half][1] = npz[half][0]
    np.savez_compressed(folder / f"{tag}.npz", **npz)
    models = pd.read_csv(folder / f"{tag}_models.csv")
    for column in ("score_id", "score_ood", "acc_id", "acc_ood", "flood_share_id"):
        models.loc[1, column] = models.loc[0, column]
    models.to_csv(folder / f"{tag}_models.csv", index=False)
    code, report = audit(folder, data, tmp / "audit_duplicate.json")
    check(code == 1 and "drop rule" in failed_checks(report),
          f"a kept duplicate fails the drop rule: {failed_checks(report)}")

    folder = damaged_copy(tmp, cells, "no_dropped")
    (folder / f"{tag}_dropped.csv").unlink()
    code, report = audit(folder, data, tmp / "audit_no_dropped.json")
    check(code == 1 and "dropped settings" in failed_checks(report),
          f"a missing dropped table fails: {failed_checks(report)}")

    folder = damaged_copy(tmp, cells, "dropped_incomplete")
    dropped = pd.read_csv(folder / f"{tag}_dropped.csv")
    dropped.iloc[1:].to_csv(folder / f"{tag}_dropped.csv", index=False)
    code, report = audit(folder, data, tmp / "audit_incomplete.json")
    check(code == 1 and "kept + dropped = grid" in failed_checks(report),
          f"a setting neither kept nor dropped fails: {failed_checks(report)}")

    folder = damaged_copy(tmp, cells, "seed_changed")
    models = pd.read_csv(folder / f"{tag}_models.csv")
    models.loc[3, "seed"] += 1
    models.to_csv(folder / f"{tag}_models.csv", index=False)
    code, report = audit(folder, data, tmp / "audit_seed.json")
    check(code == 1 and "table == zoo_specs" in failed_checks(report),
          f"a changed seed fails: {failed_checks(report)}")

    folder = damaged_copy(tmp, cells, "pre_grid")
    models = pd.read_csv(folder / f"{tag}_models.csv").drop(columns=["grid_idx"])
    models.to_csv(folder / f"{tag}_models.csv", index=False)
    code, report = audit(folder, data, tmp / "audit_pre_grid.json")
    check(code == 1 and "grid table" in failed_checks(report) and not report.get("refits"),
          f"a table without grid_idx fails and is not refitted: {failed_checks(report)}")


# ----------------------------------------------------------------- 10: the real cell
def check_real(cells: Path, data: Path):
    print(f"10. the real cell ({cells})")
    tag = "FL_band_AE_s0"
    if not (cells / f"{tag}.npz").exists() or not (data / "FEATURES.txt").exists():
        print("  skip  (no cell or no dataset)")
        return
    models = pd.read_csv(cells / f"{tag}_models.csv")
    dropped = pd.read_csv(cells / f"{tag}_dropped.csv")
    npz = np.load(cells / f"{tag}.npz")
    check(sorted(models.grid_idx.tolist() + dropped.grid_idx.tolist()) == list(range(369)),
          f"{len(models)} kept + {len(dropped)} dropped = 369 settings")
    counts = [int(zoo.predicts_flood(s, npz["y_id"]).sum()) for s in npz["S_id"]]
    check(not any(zoo.is_degenerate(k, len(npz["y_id"])) for k in counts),
          "no kept model is degenerate (recounted from S_id)")
    keys = [zoo.prediction_key(dict(s_id=a, s_od=b), npz["y_id"], npz["y_ood"])
            for a, b in zip(npz["S_id"], npz["S_ood"])]
    check(len(set(keys)) == len(keys), "no two kept models with the same predictions")
    df, feats = zoo.load("FL", data)
    tr = zoo.cell_data(df, feats, "band", "AE", 0)["tr"]
    used = [feats[j] for j in zoo.varying_columns(tr[feats].to_numpy(float))]
    check(set(models.n_feat) == {len(used)} and set(models.features) == {";".join(used)}
          and set(models.n_train) == {len(tr)},
          f"every model: the {len(used)} input columns and {len(tr):,} training rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=Path("data/interim/aotil/cells_v4_2026-11-07"))
    ap.add_argument("--data", type=Path, default=Path("data/processed_v4"))
    a = ap.parse_args()
    check_grids()
    check_specs()
    check_build()
    check_drop_rule()
    check_split_population()
    check_onehot()
    check_nan_inputs()
    check_fit_one()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cells = check_run_cell(tmp)
        check_model_split()
        check_audit(tmp, cells)
    check_real(a.cells, a.data)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
