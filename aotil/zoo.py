#!/usr/bin/env python3
"""
Fit the model population of one cell and score it on the cell's ID and OOD test halves.

A cell is (state, negative protocol, training zone, seed). The 50 km tiles are split once, 70/30,
and every model sees that split. A test tile is kept only if it holds positives of both zones, so
the ID half (training zone) and the OOD half (other zone) cover the same ground.

The population is one model per setting of a fixed hyperparameter grid per family (GRIDS, the
paper's Table tab:models), so models differ only by family and setting: every model trains on all
features and all training rows of the cell. Some settings are deliberately weak, to spread ID
accuracy. Two drop rules run after fitting, in this order (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md
D6). A model that predicts flood for fewer than 5% or more than 95% of the ID test points is
degenerate (is_degenerate): its accuracy on any subset only reflects the subset's share of
positives, which the subset search could exploit. A model whose 0/1 predictions on every ID test
point and every OOD point equal those of a model earlier in grid order is a duplicate: the first
in grid order is kept. Each setting draws its estimator seed from the cell seed in grid order, so
any one model can be refitted alone (audit_cell.py does this through cell_data and fit_one).

The model inputs (model_inputs, D5): each numeric feature as is, and each categorical feature of
CATEGORICAL one-hot encoded, one 0/1 column per class present in the full dataset (a fixed list,
no dropped reference column). A point missing a categorical feature has NaN in all of that
feature's columns; families with an imputer fill each column with its training median, hgb and
xgb take the NaN as is.

Inputs:  <data>/FEATURES.txt and <data>/dataset_<state>.parquet
Outputs: <out>/<tag>.npz (scores of the kept models and per-point keys, see main),
         <out>/<tag>_models.csv (one row per kept model, in grid order, see model_row) and
         <out>/<tag>_dropped.csv (one row per dropped setting, with why: its share of ID test
         points predicted flood, and for a duplicate the grid_idx of the model it repeats), with
         tag = <state>_<protocol>_<train-dom>_s<seed>

Run with one BLAS/OpenMP thread, as below, so a --n-jobs 1 run matches a parallel one (run_cell):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    .venv/bin/python -u aotil/zoo.py --state FL --protocol band --train-dom AE --seed 0 \\
        --data data/processed_v4 --out data/interim/aotil/cells_v4_<date> --n-jobs -1
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# xgboost is optional (it is the one family that can use a GPU); without it the population
# has seven families. Any import failure counts, e.g. a missing OpenMP runtime on macOS.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# The settings with few iterations are deliberate: an undertrained model is a legitimate
# low-accuracy member of the population. sklearn warns on every one of them, which buries the real
# output.
warnings.filterwarnings("ignore")

TILE = "tile_50km"
MIN_TRAIN = 500        # rows: a cell's fewest training rows

# The features the models see one-hot encoded (the paper's Data statistics): land cover (NLCD
# class codes), soil flood frequency (ordinal, but encoded as classes) and fluvial class. Every
# other feature of FEATURES.txt is numeric and enters as is. region_match.py matches land cover
# exactly and leaves the other two out of its numeric features.
CATEGORICAL = ("landcover", "soil_floodfreq", "soil_fluvclass")

# The drop rule: a model is degenerate when it predicts flood for fewer than DEGENERATE_CUT or
# more than 1 - DEGENERATE_CUT of the ID test points. Exact fractions, so a share of exactly 5%
# or 95% is kept.
DEGENERATE_CUT = Fraction(1, 20)

# The hyperparameter grids (docs/TASK-ZOO-GRIDS-2026-11-06.md section 2, the paper's Table
# tab:models). Every combination of a family's axes is one model. Settings not named here are the
# library defaults, apart from the fixed ones in build. The families are fitted in sorted
# code-name order and each family's settings in itertools.product order over its axes as listed
# (first axis slowest), which is also the order of the estimator seeds (zoo_specs).
_FOREST_AXES = (("n_estimators", (10, 30, 100)),
                ("max_depth", (1, 2, 3, 5, 8, None)),
                ("max_features", (1, "sqrt", None)))
GRIDS: dict[str, tuple] = {
    "dt": (("max_depth", (1, 2, 3, 4, 5, 6, 8, 10, 12, None)),
           ("min_samples_leaf", (1, 20, 200, 2000))),
    "et": _FOREST_AXES,
    "hgb": (("learning_rate", (0.01, 0.03, 0.1, 0.5)),
            ("max_iter", (3, 10, 50, 300)),
            ("max_depth", (1, 3, None))),
    "lr": (("C", (0.001, 0.01, 0.1, 1, 100)),
           ("max_iter", (1, 2, 3, 5, 100)),
           ("class_weight", (None, "balanced"))),
    "mlp": (("hidden_layer_sizes", ((2,), (8,), (16,), (32,), (64, 32))),
            ("max_iter", (3, 20, 200)),
            ("learning_rate_init", (1e-4, 1e-3, 1e-2))),
    "nb": (("var_smoothing", (1e-9, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5, 1,
                              2, 3, 10)),
           ("priors", (None, (0.5, 0.5)))),
    "rf": _FOREST_AXES,
    "xgb": (("learning_rate", (0.01, 0.03, 0.1, 0.5)),
            ("n_estimators", (3, 10, 50, 300)),
            ("max_depth", (1, 3, 8))),
}

# Stored per OOD point: point_id joins back to the point table, lon/lat build the spatial graph,
# the rest are grouping keys for the restricted graphs and the characterisation.
SPATIAL_COLS = ["point_id", "lon", "lat", "block_id", "tile_50km", "county_fips",
                "meta_comid", "meta_source_cit"]
# Stored per ID point as <col>_id: enough to join and map the ID half without re-splitting.
ID_COLS = ["point_id", "lon", "lat"]


# ----------------------------------------------------------------- data
def read_features(path: Path) -> list[str]:
    """FEATURES.txt as column names: whitespace-separated; blank lines and lines starting with
    '#' are skipped."""
    names = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            names.extend(line.split())
    return names


def onehot_name(feature: str, level: float) -> str:
    """The input column of one class of a categorical feature, e.g. landcover_90."""
    return f"{feature}_{int(level)}"


def onehot_levels(df: pd.DataFrame, feats: list[str]) -> dict[str, list[float]]:
    """The classes of each CATEGORICAL feature in feats, sorted: every finite value over all rows
    of df (the full dataset), so every model of every cell sees the same columns. A class must be
    a whole number (its column is named after it)."""
    levels = {}
    for f in feats:
        if f not in CATEGORICAL:
            continue
        values = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        found = np.unique(values[np.isfinite(values)])
        if (found != np.round(found)).any():
            raise ValueError(f"{f}: a class that is not a whole number: {found[found != np.round(found)][:3]}")
        levels[f] = found.tolist()
    return levels


def model_inputs(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """(df with the one-hot columns added, the model's input columns in order).

    Each numeric feature is its own input column. Each CATEGORICAL feature becomes one 0/1 column
    per class of onehot_levels, in the feature's place in feats, with no reference column dropped;
    a row missing the feature (NaN) is NaN in all of its columns, so the imputers and the families
    that handle NaN treat it as missing, not as a class.
    """
    levels = onehot_levels(df, feats)
    inputs, added = [], {}
    for f in feats:
        if f not in levels:
            inputs.append(f)
            continue
        values = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(values)
        for level in levels[f]:
            column = (values == level).astype(float)
            column[missing] = np.nan
            name = onehot_name(f, level)
            if name in df.columns:
                raise ValueError(f"the dataset already has a column {name}")
            added[name] = column
            inputs.append(name)
    if added:
        df = pd.concat([df, pd.DataFrame(added, index=df.index)], axis=1)
    return df, inputs


def load(state: str, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """The state's point table with the one-hot columns added, and the model's input columns
    (model_inputs of the FEATURES.txt features the table has)."""
    feats = read_features(data_dir / "FEATURES.txt")
    df = pd.read_parquet(data_dir / f"dataset_{state}.parquet")
    feats = [f for f in feats if f in df.columns]
    return model_inputs(df, feats)


def subset(df: pd.DataFrame, domain: str, protocol: str) -> pd.DataFrame:
    """One zone's supervised set: its positives plus its negatives of the given protocol."""
    return df[(df.domain == domain) &
              ((df.label == 1) | (df.neg_protocol == protocol))]


def split_tiles(df: pd.DataFrame, seed: int, test_frac: float = 0.30) -> tuple[set, set]:
    """(training tiles, test tiles), shared by every model of the cell.

    Candidate test tiles without positives of both zones are dropped, not moved to training.
    """
    rng = np.random.default_rng(seed)
    tiles = np.array(sorted(df[TILE].dropna().unique()))
    rng.shuffle(tiles)
    n_test = max(3, int(round(len(tiles) * test_frac)))
    test, train = set(tiles[:n_test]), set(tiles[n_test:])

    positives = df[df.label == 1]
    test_positives = positives[positives[TILE].isin(test)]
    n_zones = test_positives.groupby(TILE).domain.nunique()
    test = set(n_zones[n_zones == 2].index)
    return train, test


# ----------------------------------------------------------------- zoo
def grid_settings() -> dict[str, list[dict]]:
    """Every setting of each family's grid (GRIDS), in population order: the families in sorted
    code-name order, and within a family itertools.product over its axes as GRIDS lists them
    (first axis slowest). Without xgboost the population has seven families."""
    settings = {}
    for kind in sorted(GRIDS):
        if kind == "xgb" and not _HAS_XGB:
            continue
        names = [name for name, _ in GRIDS[kind]]
        values = [axis for _, axis in GRIDS[kind]]
        settings[kind] = [dict(zip(names, combo)) for combo in itertools.product(*values)]
    return settings


def zoo_specs(seed: int) -> list[dict]:
    """One spec per grid setting, in population order: grid_idx (the setting's position), kind,
    kw (the setting) and seed (its estimator seed).

    The seeds are drawn in that order from one RNG seeded with the cell seed, so a setting keeps
    its seed whichever settings are later dropped, and xgb, the last family, leaves the others'
    seeds unchanged when xgboost is missing."""
    rng = np.random.default_rng(seed)
    specs = []
    for kind, settings in grid_settings().items():
        for kw in settings:
            specs.append(dict(grid_idx=len(specs), kind=kind, kw=kw,
                              seed=int(rng.integers(1 << 30))))
    return specs


def build(kind: str, kw: dict, seed: int, device: str = "cpu"):
    """An unfitted model of one family.

    xgb and hgb handle NaN natively; every other family imputes the training median, and lr and
    mlp standardize their inputs. random_state is the model's seed in every family but nb;
    settings not fixed here or in kw are the installed library's defaults. Any device other than
    "cpu" puts xgb on CUDA.
    """
    if kind == "xgb":
        xgb_device = "cpu" if device == "cpu" else "cuda"
        return XGBClassifier(random_state=seed, tree_method="hist", device=xgb_device,
                             verbosity=0, n_jobs=1, **kw)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **kw)
    imputer = SimpleImputer(strategy="median")
    if kind == "rf":
        return make_pipeline(imputer, RandomForestClassifier(random_state=seed, n_jobs=1, **kw))
    if kind == "et":
        return make_pipeline(imputer, ExtraTreesClassifier(random_state=seed, n_jobs=1, **kw))
    if kind == "dt":
        return make_pipeline(imputer, DecisionTreeClassifier(random_state=seed, **kw))
    if kind == "lr":
        return make_pipeline(imputer, StandardScaler(),
                             LogisticRegression(random_state=seed, **kw))
    if kind == "mlp":
        return make_pipeline(imputer, StandardScaler(), MLPClassifier(random_state=seed, **kw))
    if kind == "nb":
        return make_pipeline(imputer, GaussianNB(**kw))
    raise ValueError(kind)


def score_rows(model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(true-class scores, accuracy, AUC) of a fitted model on one set of rows.

    Every S row and every score_*, acc_* and auc_* column of the model table comes from here.
    The score is the probability of the observed class (p if y = 1, else 1 - p), not 0/1
    correctness: AUC has no per-example form, and a 0.5 threshold would mean different things in
    cells with different class ratios. Accuracy is argmax correctness, as in predict() and
    oodselect_paper.correctness: a negative at exactly p = 0.5 is correct.

    The AUC is ranked on p itself. p rebuilt from the scores (1 - (1 - p) for a negative) can be
    an ulp off and split the ties that trees, with few distinct p, make in most pairs; that AUC
    can be off by ~0.1. The clip to [0, 1] is for sklearn's trapezoid sum, which can land an ulp
    above 1 on a perfect separation and would fail audit_cell's range check.
    """
    p = model.predict_proba(X)[:, 1]
    scores = np.where(y == 1, p, 1.0 - p)
    correct = (scores > 0.5) | ((scores == 0.5) & (y == 0))
    accuracy = float(correct.mean())
    auc = float(np.clip(roc_auc_score(y, p), 0.0, 1.0))
    return scores, accuracy, auc


def predicts_flood(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Whether the model predicts flood (p > 0.5, a tie at 0.5 counting as dry) at each point,
    from its true-class scores (score_rows: p for a positive, 1 - p for a negative).

    At a negative, flood means S = 1 - p < 0.5. That is exactly p > 0.5: for p >= 0.5 the
    subtraction 1 - p is exact (Sterbenz), and for p < 0.5 it rounds to at least 0.5."""
    scores, y = np.asarray(scores), np.asarray(y)
    return np.where(y == 1, scores > 0.5, scores < 0.5)


def is_degenerate(n_flood: int, n_points: int) -> bool:
    """The drop rule: flood predicted for fewer than 5% or more than 95% of n_points points.
    Compared as exact fractions, so a share of exactly 5% or 95% is kept."""
    if n_points <= 0:
        raise ValueError("the drop rule needs at least one ID test point")
    share = Fraction(int(n_flood), int(n_points))
    return share < DEGENERATE_CUT or share > 1 - DEGENERATE_CUT


def hp_value(v):
    """A kw value as a model-table cell: None -> "none" (read_csv would turn "None" into NaN),
    a tuple or list -> its tuple repr, e.g. "(64, 32)"."""
    if v is None:
        return "none"
    if isinstance(v, (tuple, list)):
        return repr(tuple(v))
    return v


# ----------------------------------------------------------------- run
def extra_training_rows(df: pd.DataFrame, feats: list[str], extra_ids,
                        extra_repeat: int) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) of the --extra-train points, each row repeated extra_repeat times (at least once)."""
    by_id = df.set_index("point_id")
    missing = [p for p in extra_ids if p not in by_id.index]
    if missing:
        raise SystemExit(f"--extra-train: {len(missing)} point_ids not in the dataset, "
                         f"first {missing[:3]}")
    extra = by_id.loc[list(extra_ids)]
    reps = max(1, int(extra_repeat))
    X = np.repeat(extra[feats].to_numpy(float), reps, axis=0)
    y = np.repeat(extra.label.to_numpy(), reps)
    return X, y


def cell_data(df, feats, protocol, train_dom, seed, extra_ids=None, extra_repeat: int = 1) -> dict:
    """The cell's training rows and its ID and OOD test halves, as frames and float arrays.

    Shared by run_cell and audit_cell.py, so a refit sees the same rows in the same order and the
    same feature order. extra_ids (point_ids from any zone or tile) become arrays Xextra / yextra;
    those points also stay in whichever test half holds them, so readers must leave them out of
    any evaluation.
    """
    eval_dom = "AE" if train_dom == "A" else "A"
    tr_tiles, te_tiles = split_tiles(df, seed)

    train_zone = subset(df, train_dom, protocol)
    eval_zone = subset(df, eval_dom, protocol)
    tr = train_zone[train_zone[TILE].isin(tr_tiles)]
    id_te = train_zone[train_zone[TILE].isin(te_tiles)]
    ood_te = eval_zone[eval_zone[TILE].isin(te_tiles)]
    arrays = dict(Xtr=tr[feats].to_numpy(float), ytr=tr.label.to_numpy(),
                  Xid=id_te[feats].to_numpy(float), yid=id_te.label.to_numpy(),
                  Xod=ood_te[feats].to_numpy(float), yod=ood_te.label.to_numpy())
    if extra_ids is not None and len(extra_ids) > 0:
        arrays["Xextra"], arrays["yextra"] = extra_training_rows(df, feats, extra_ids,
                                                                 extra_repeat)
    return dict(eval_dom=eval_dom, tr_tiles=tr_tiles, te_tiles=te_tiles,
                tr=tr, id_te=id_te, ood_te=ood_te, arrays=arrays)


def varying_columns(X: np.ndarray) -> list[int]:
    """Columns of X with at least two distinct finite values."""
    keep = []
    for j in range(X.shape[1]):
        finite = X[np.isfinite(X[:, j]), j]
        if np.unique(finite).size >= 2:
            keep.append(j)
    return keep


def fit_one(sp: dict, A: dict, device: str = "cpu") -> dict | None:
    """Fit and score one model of the population; None when it cannot be fitted.

    A is cell_data's arrays. The model trains on every training row (with the extra rows, if
    any, appended) and every feature that has at least two distinct finite values in those rows.
    n_flood_id counts the ID test points it predicts flood (predicts_flood), for the drop rule.
    """
    Xs, ys = A["Xtr"], A["ytr"]
    if "Xextra" in A:
        Xs = np.vstack([Xs, A["Xextra"]])
        ys = np.concatenate([ys, A["yextra"]])
    if len(np.unique(ys)) < 2:
        return None
    cols = np.asarray(varying_columns(Xs), dtype=int)
    if len(cols) < 2:
        return None
    Xs = Xs[:, cols]

    try:
        model = build(sp["kind"], sp["kw"], sp["seed"], device=device)
        model.fit(Xs, ys)
        s_id, acc_id, auc_id = score_rows(model, A["Xid"][:, cols], A["yid"])
        s_od, acc_od, auc_od = score_rows(model, A["Xod"][:, cols], A["yod"])
        _, acc_sub, auc_sub = score_rows(model, Xs, ys)
        _, acc_all, auc_all = score_rows(model, A["Xtr"][:, cols], A["ytr"])
    except Exception as e:
        print(f"    [skip {sp['kind']}] {type(e).__name__}: {e}", flush=True)
        return None
    n_flood_id = int(predicts_flood(s_id, A["yid"]).sum())
    # Each metric pair is (accuracy, AUC).
    return dict(sp=sp, cols=cols, n_train=len(A["Xtr"]), s_id=s_id, s_od=s_od,
                n_flood_id=n_flood_id, id=(acc_id, auc_id), ood=(acc_od, auc_od),
                train_sub=(acc_sub, auc_sub), train_all=(acc_all, auc_all))


def hp_columns(kw: dict) -> dict:
    """hp_<name> -> hp_value(value) for each setting of a spec's kw."""
    return {f"hp_{name}": hp_value(value) for name, value in kw.items()}


def model_row(got: dict, feats: list[str], idx: int) -> dict:
    """One row of <tag>_models.csv from a fit_one result; idx is the model's row in the cell's
    score matrices (its position among the kept models).

    grid_idx is the setting's position in the grid (zoo_specs). features lists the columns the
    model used, after fit_one's constant-column drop. flood_share_id is the share of ID test
    points it predicts flood (the drop rule's statistic). The training metrics show whether a
    model interpolates its labels, which an inverse ID-OOD line from label noise requires (Sanyal
    et al. 2025): _train_sub is on the rows it was fitted on (extra rows included), _train_all on
    every training-tile row of the training zone (no extra rows); without extra rows the two are
    the same rows. hp_<name> holds each setting of the spec's kw; an empty cell means the family
    does not have that hyperparameter.
    """
    sp, cols = got["sp"], got["cols"]
    acc_id, auc_id = got["id"]
    acc_od, auc_od = got["ood"]
    row = dict(
        idx=idx, grid_idx=sp["grid_idx"], kind=sp["kind"], n_feat=len(cols),
        score_id=float(got["s_id"].mean()), score_ood=float(got["s_od"].mean()),
        acc_id=acc_id, acc_ood=acc_od, auc_id=auc_id, auc_ood=auc_od,
        flood_share_id=got["n_flood_id"] / len(got["s_id"]),
        seed=sp["seed"], n_train=got["n_train"],
        features=";".join(feats[j] for j in cols),
        acc_train_sub=got["train_sub"][0], auc_train_sub=got["train_sub"][1],
        acc_train_all=got["train_all"][0], auc_train_all=got["train_all"][1],
    )
    row.update(hp_columns(sp["kw"]))
    return row


DROPPED_COLUMNS = ["grid_idx", "kind", "seed", "reason", "flood_share_id", "acc_id", "acc_ood",
                   "duplicate_of"]


def dropped_row(sp: dict, got: dict | None, reason: str = "degenerate",
                duplicate_of: int | None = None) -> dict:
    """One row of <tag>_dropped.csv: a setting left out of the population, with why. A fitted
    model ("degenerate" or "duplicate") also gets its ID flood share and test accuracies, and a
    duplicate the grid_idx of the kept model whose predictions it repeats; a setting fit_one
    could not fit ("unusable") gets NaN for them."""
    row = dict(grid_idx=sp["grid_idx"], kind=sp["kind"], seed=sp["seed"],
               reason="unusable", flood_share_id=np.nan, acc_id=np.nan, acc_ood=np.nan,
               duplicate_of=np.nan)
    if got is not None:
        row.update(reason=reason, flood_share_id=got["n_flood_id"] / len(got["s_id"]),
                   acc_id=got["id"][0], acc_ood=got["ood"][0])
        if duplicate_of is not None:
            row["duplicate_of"] = int(duplicate_of)
    row.update(hp_columns(sp["kw"]))
    return row


def prediction_key(got: dict, y_id: np.ndarray, y_ood: np.ndarray) -> bytes:
    """A model's 0/1 predictions (predicts_flood) on every ID test point and every OOD point,
    packed: two models repeat each other exactly when their keys are equal."""
    flood = np.concatenate([predicts_flood(got["s_id"], y_id), predicts_flood(got["s_od"], y_ood)])
    return np.packbits(flood).tobytes() + len(flood).to_bytes(8, "little")


def hp_last(table: pd.DataFrame) -> pd.DataFrame:
    """table with its hp_<name> columns sorted and moved to the end."""
    hp_cols = sorted(c for c in table.columns if c.startswith("hp_"))
    other_cols = [c for c in table.columns if not c.startswith("hp_")]
    return table[other_cols + hp_cols]


def fit_population(specs: list[dict], A: dict, n_jobs: int, device: str) -> list:
    """fit_one for every spec, in spec order."""
    if n_jobs == 1:
        return [fit_one(sp, A, device) for sp in specs]
    # One BLAS/OpenMP thread per worker. joblib otherwise gives each worker cpu_count // n_jobs
    # threads, and Apple Accelerate (numpy's BLAS on the laptop, invisible to threadpoolctl)
    # moves LogisticRegression's scores by up to ~1e-11 between thread counts, so the cell
    # would depend on --n-jobs.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=n_jobs, verbose=0)(delayed(fit_one)(sp, A, device)
                                                  for sp in specs)


def split_population(specs: list[dict], fitted: list, y_id: np.ndarray,
                     y_ood: np.ndarray) -> tuple[list[dict], list[dict]]:
    """(kept fit_one results, dropped_row rows), both in grid order. A setting is dropped when
    fit_one could not fit it ("unusable"), when it is degenerate (is_degenerate on the ID test
    points), and then, among the rest, when its 0/1 predictions on every ID test point and every
    OOD point (y_id and y_ood, the labels behind its scores) equal those of a model earlier in
    grid order ("duplicate"; the first one in grid order is kept)."""
    kept, dropped, first_with = [], [], {}
    for sp, got in zip(specs, fitted):
        if got is None:
            dropped.append(dropped_row(sp, None))
        elif is_degenerate(got["n_flood_id"], len(got["s_id"])):
            dropped.append(dropped_row(sp, got, "degenerate"))
        else:
            key = prediction_key(got, y_id, y_ood)
            if key in first_with:
                dropped.append(dropped_row(sp, got, "duplicate", first_with[key]))
            else:
                first_with[key] = sp["grid_idx"]
                kept.append(got)
    return kept, dropped


DROP_REASONS = ("degenerate", "duplicate", "unusable")


def family_counts(specs: list[dict], kept: list[dict], dropped: list[dict]) -> pd.DataFrame:
    """Per family: grid settings, degenerate, duplicate, unusable and kept models."""
    table = pd.DataFrame(dict(kind=[sp["kind"] for sp in specs]))
    table = table.groupby("kind").size().rename("settings").to_frame()
    reasons = pd.DataFrame(dropped, columns=["kind", "reason"])
    for reason in DROP_REASONS:
        table[reason] = reasons[reasons.reason == reason].groupby("kind").size()
    table["kept"] = pd.Series([got["sp"]["kind"] for got in kept], dtype=object).value_counts()
    return table.fillna(0).astype(int)


def run_cell(df, feats, protocol, train_dom, seed,
             n_jobs: int = 1, device: str = "cpu", extra_ids=None, extra_repeat: int = 1):
    """Fit and score the population of one cell and apply the drop rule; None when the cell is
    unusable."""
    data = cell_data(df, feats, protocol, train_dom, seed, extra_ids, extra_repeat)
    tr, id_te, ood_te, A = data["tr"], data["id_te"], data["ood_te"], data["arrays"]
    if len(tr) < MIN_TRAIN or id_te.empty or ood_te.empty:
        return None

    print(f"  train {len(tr):,} | ID test {len(id_te):,} | OOD test {len(ood_te):,} "
          f"| tiles train {len(data['tr_tiles'])} test {len(data['te_tiles'])}", flush=True)
    if "Xextra" in A:
        print(f"  extra training rows: {len(extra_ids):,} point_ids x {extra_repeat} = "
              f"{len(A['yextra']):,} rows (positives {int(A['yextra'].sum()):,}) "
              f"added to every model", flush=True)
    # A protocol with no negatives in the dataset (shadedX on a band-only sample) leaves every
    # set with one label. No model can be fitted on it, and a 0-model cell would break every
    # reader, so it is refused here rather than after the fits.
    one_label = []
    for half, y in (("train", A["ytr"]), ("ID test", A["yid"]), ("OOD test", A["yod"])):
        if len(np.unique(y)) < 2:
            one_label.append(half)
    if one_label:
        print(f"  one label only in {', '.join(one_label)} (protocol {protocol}): "
              f"no model can be fitted or scored", flush=True)
        return None

    t0 = time.time()
    specs = zoo_specs(seed)
    fitted = fit_population(specs, A, n_jobs, device)
    kept, dropped = split_population(specs, fitted, A["yid"], A["yod"])
    fit_secs = time.time() - t0
    counts = family_counts(specs, kept, dropped)
    print(f"  {len(specs)} settings fitted in {fit_secs:.0f}s: {len(kept)} kept, "
          f"{int(counts.degenerate.sum())} degenerate (flood share of the ID test points "
          f"< {float(DEGENERATE_CUT):g} or > {float(1 - DEGENERATE_CUT):g}), "
          f"{int(counts.duplicate.sum())} duplicate (the predictions of a model earlier in "
          f"grid order), {int(counts.unusable.sum())} unusable", flush=True)
    print("  " + counts.to_string().replace("\n", "\n  "), flush=True)
    if not kept:                 # every setting unusable or degenerate: same refusal as above
        return None

    models = hp_last(pd.DataFrame([model_row(got, feats, i) for i, got in enumerate(kept)]))
    dropped_table = pd.DataFrame(dropped, columns=DROPPED_COLUMNS)
    if dropped:
        dropped_table = hp_last(pd.DataFrame(dropped))
    print(f"  training accuracy: median {models.acc_train_sub.median():.3f}"
          f" | >= 0.99 in {(models.acc_train_sub >= .99).sum()} of {len(models)} models",
          flush=True)

    # The per-point keys are taken here, where the row order still matches the columns of
    # S_ood and S_id. lon/lat are not features, so nothing else carries them through.
    spatial, id_cols, missing = {}, {}, []
    for c in SPATIAL_COLS:
        if c in ood_te:
            spatial[c] = ood_te[c].to_numpy()
        else:
            missing.append(c)
    for c in ID_COLS:
        if c in id_te:
            id_cols[f"{c}_id"] = id_te[c].to_numpy()
        else:
            missing.append(f"{c}_id")
    if missing:
        print(f"  note: not in this dataset, omitted from the cell: {missing}", flush=True)
    return dict(models=models, dropped=dropped_table, fit_secs=fit_secs,
                S_ood=np.asarray([got["s_od"] for got in kept]),
                S_id=np.asarray([got["s_id"] for got in kept]),
                y_ood=A["yod"], y_id=A["yid"], tiles_ood=ood_te[TILE].to_numpy(),
                spatial=spatial, id_cols=id_cols,
                protocol=protocol, train_dom=train_dom, eval_dom=data["eval_dom"])


def read_point_ids(path: Path) -> list[str]:
    """One point_id per line; blank lines and "point_id" header lines skipped."""
    ids = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if value and value != "point_id":
            ids.append(value)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="FL")
    ap.add_argument("--protocol", default="shadedX", choices=["band", "shadedX"])
    ap.add_argument("--train-dom", default="AE", choices=["A", "AE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/interim/aotil/cells")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel model fits; -1 uses every core")
    ap.add_argument("--device", default="cpu",
                    help="cpu, or cuda to put the XGBoost family on the GPU")
    ap.add_argument("--overwrite", action="store_true",
                    help="refit a cell whose .npz already exists")
    ap.add_argument("--extra-train", type=Path, default=None,
                    help="file of point_ids (one per line, optional 'point_id' header) "
                         "appended to every model's training rows; default none")
    ap.add_argument("--extra-train-repeat", type=int, default=1,
                    help="repeat each --extra-train row this many times (default 1)")
    args = ap.parse_args()

    tag = f"{args.state}_{args.protocol}_{args.train_dom}_s{args.seed}"
    out = Path(args.out)
    # Cells are expensive: an existing one is kept unless --overwrite, so an interrupted sweep
    # resumes by re-running the same command.
    if (out / f"{tag}.npz").exists() and not args.overwrite:
        print(f"{tag}: cell exists, skipping (--overwrite to refit)")
        return 0

    df, feats = load(args.state, Path(args.data))
    print(f"[{args.state}] {len(df):,} rows | {len(feats)} input columns: {', '.join(feats)}")
    print(f"cell: protocol={args.protocol} train={args.train_dom} seed={args.seed} "
          f"settings={len(zoo_specs(args.seed))} device={args.device}")
    extra_ids = None
    if args.extra_train is not None:
        extra_ids = read_point_ids(args.extra_train)
    cell = run_cell(df, feats, args.protocol, args.train_dom, args.seed,
                    n_jobs=args.n_jobs, device=args.device, extra_ids=extra_ids,
                    extra_repeat=args.extra_train_repeat)
    if cell is None:
        print("cell unusable: nothing written")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    # S_ood / S_id: (models x points) true-class scores of the kept models, rows in model-table
    # order. y_id lets readers recover exact argmax correctness on the ID half (ties at
    # p = 0.5). tiles_ood is kept for older readers. String columns are stored as fixed-width
    # str, not objects.
    payload = dict(S_ood=cell["S_ood"], S_id=cell["S_id"], y_ood=cell["y_ood"],
                   y_id=cell["y_id"], tiles_ood=cell["tiles_ood"].astype(str))
    per_point = dict(cell["spatial"])
    per_point.update(cell["id_cols"])
    for name, values in per_point.items():
        if values.dtype.kind in "OU":
            values = values.astype(str)
        payload[name] = values
    if extra_ids is not None:
        payload["extra_train_point_id"] = np.asarray(extra_ids, dtype=str)
        payload["extra_train_repeat"] = np.int64(args.extra_train_repeat)
    np.savez_compressed(out / f"{tag}.npz", **payload)
    cell["models"].to_csv(out / f"{tag}_models.csv", index=False)
    cell["dropped"].to_csv(out / f"{tag}_dropped.csv", index=False)
    print(f"saved {out}/{tag}.npz  (S_ood {cell['S_ood'].shape}, S_id {cell['S_id'].shape}; "
          f"keys: {', '.join(payload)})")
    print(f"saved {out}/{tag}_models.csv  ({len(cell['models'])} rows x "
          f"{cell['models'].shape[1]} columns)")
    print(f"saved {out}/{tag}_dropped.csv  ({len(cell['dropped'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
