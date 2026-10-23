#!/usr/bin/env python3
"""
Fit the model population of one cell and score it on the cell's ID and OOD test halves.

A cell is (state, negative protocol, training zone, seed). The 50 km tiles are split once, 70/30,
and every model sees that split. A test tile is kept only if it holds positives of both zones, so
the ID half (training zone) and the OOD half (other zone) cover the same ground. Each model draws
its feature subset, row subsample and estimator seed from its own spec seed, so any one model can
be refitted alone (audit_cell.py does this through cell_data and fit_one).

Inputs:  <data>/FEATURES.txt and <data>/dataset_<state>.parquet
Outputs: <out>/<tag>.npz (scores and per-point keys, see main) and <out>/<tag>_models.csv (one
         row per model, see model_row), with tag = <state>_<protocol>_<train-dom>_s<seed>

Run with one BLAS/OpenMP thread, as below, so a --n-jobs 1 run matches a parallel one (run_cell):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    .venv/bin/python -u aotil/zoo.py --state FL --protocol band --train-dom AE --seed 0 \\
        --n-models 300 --data data/processed --out data/interim/aotil/cells --n-jobs -1
"""
from __future__ import annotations

import argparse
import time
import warnings
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

# The low max_iter MLPs are deliberate: an undertrained model is a legitimate low-accuracy
# member of the population. sklearn warns on every one of them, which buries the real output.
warnings.filterwarnings("ignore")

TILE = "tile_50km"
MIN_TRAIN = 500        # rows: a model's smallest row subsample, and a cell's fewest training rows

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


def load(state: str, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """The state's point table and the FEATURES.txt columns it actually has."""
    feats = read_features(data_dir / "FEATURES.txt")
    df = pd.read_parquet(data_dir / f"dataset_{state}.parquet")
    feats = [f for f in feats if f in df.columns]
    return df, feats


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
def learner_settings() -> dict[str, list[dict]]:
    """Every hyperparameter setting of each learner family, in the order the population uses."""
    settings = {"hgb": [], "rf": [], "et": [], "lr": [], "mlp": [], "dt": [], "nb": [{}]}
    for depth in (2, 3, None):
        for n_iter in (30, 120, 400):
            for rate in (0.02, 0.10, 0.35):
                settings["hgb"].append(dict(max_depth=depth, max_iter=n_iter, learning_rate=rate))
    for depth in (3, 6, None):
        for n_trees in (40, 200):
            settings["rf"].append(dict(max_depth=depth, n_estimators=n_trees))
            settings["et"].append(dict(max_depth=depth, n_estimators=n_trees))
    for C in (0.003, 0.1, 3.0, 100.0):
        settings["lr"].append(dict(C=C))
    for hidden in ((8,), (48,), (64, 32)):
        for n_iter in (25, 120):
            settings["mlp"].append(dict(hidden_layer_sizes=hidden, max_iter=n_iter))
    for depth in (2, 4, 10):
        settings["dt"].append(dict(max_depth=depth))
    if _HAS_XGB:
        settings["xgb"] = []
        for depth in (2, 4, 8):
            for n_trees in (50, 300):
                settings["xgb"].append(dict(max_depth=depth, n_estimators=n_trees))
    return settings


def zoo_specs(n_target: int, seed: int) -> list[dict]:
    """A deliberately heterogeneous population of n_target model specs.

    Diversity comes from four axes at once (family, capacity, feature subset, row subsample):
    one family at one setting collapses to a single ID accuracy, and the ID-OOD correlation
    OODSelect needs is then undefined. Families are taken round-robin, not the flat list of
    settings, which would make the population 2/3 gradient boosting and leave too few models of
    other kinds for the family-disjoint controls. The specs are drawn in idx order from one RNG,
    so zoo_specs(n, seed) is a prefix of zoo_specs(N, seed) for N > n.
    """
    settings = learner_settings()
    kinds = sorted(settings)
    n_used = {kind: 0 for kind in kinds}
    rng = np.random.default_rng(seed)

    specs = []
    for i in range(n_target):
        kind = kinds[i % len(kinds)]
        options = settings[kind]
        kw = options[n_used[kind] % len(options)]
        n_used[kind] += 1
        feat_frac = float(rng.choice([0.3, 0.6, 1.0]))
        row_frac = float(rng.choice([0.15, 0.5, 1.0]))
        model_seed = int(rng.integers(1 << 30))
        specs.append(dict(idx=i, kind=kind, kw=kw, feat_frac=feat_frac, row_frac=row_frac,
                          seed=model_seed))
    return specs


def build(kind: str, kw: dict, seed: int, device: str = "cpu"):
    """An unfitted model of one family.

    xgb and hgb handle NaN natively; every other family imputes the training median.
    random_state is the model's seed in every family but nb; settings not fixed here or in kw
    are the installed library's defaults. Any device other than "cpu" puts xgb on CUDA.
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
                             LogisticRegression(max_iter=400, random_state=seed, **kw))
    if kind == "mlp":
        return make_pipeline(imputer, StandardScaler(), MLPClassifier(random_state=seed, **kw))
    if kind == "nb":
        return make_pipeline(imputer, GaussianNB())
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
    """Fit and score one model of the population; None when its draw is unusable.

    A is cell_data's arrays. The spec's seed draws the feature subset, then the row subsample.
    Extra rows are appended after the draw, so with or without them a model keeps the same base
    rows, features and seed. Columns without two distinct finite values in those training rows
    (extra rows included) are then dropped.
    """
    Xtr, ytr = A["Xtr"], A["ytr"]
    n_feats = Xtr.shape[1]
    rng = np.random.default_rng(sp["seed"])
    n_cols = max(3, int(n_feats * sp["feat_frac"]))
    cols = np.sort(rng.choice(n_feats, n_cols, replace=False))
    n_rows = max(MIN_TRAIN, int(len(Xtr) * sp["row_frac"]))
    row_idx = rng.choice(len(Xtr), min(n_rows, len(Xtr)), replace=False)
    Xs, ys = Xtr[np.ix_(row_idx, cols)], ytr[row_idx]
    if "Xextra" in A:
        Xs = np.vstack([Xs, A["Xextra"][:, cols]])
        ys = np.concatenate([ys, A["yextra"]])
    if len(np.unique(ys)) < 2:
        return None
    keep = varying_columns(Xs)
    if len(keep) < 2:
        return None
    cols, Xs = cols[keep], Xs[:, keep]

    try:
        model = build(sp["kind"], sp["kw"], sp["seed"], device=device)
        model.fit(Xs, ys)
        s_id, acc_id, auc_id = score_rows(model, A["Xid"][:, cols], A["yid"])
        s_od, acc_od, auc_od = score_rows(model, A["Xod"][:, cols], A["yod"])
        _, acc_sub, auc_sub = score_rows(model, Xs, ys)
        _, acc_all, auc_all = score_rows(model, Xtr[:, cols], ytr)
    except Exception as e:
        print(f"    [skip {sp['kind']}] {type(e).__name__}: {e}", flush=True)
        return None
    # Each metric pair is (accuracy, AUC).
    return dict(sp=sp, cols=cols, n_train=len(row_idx), s_id=s_id, s_od=s_od,
                id=(acc_id, auc_id), ood=(acc_od, auc_od),
                train_sub=(acc_sub, auc_sub), train_all=(acc_all, auc_all))


def model_row(got: dict, feats: list[str]) -> dict:
    """One row of <tag>_models.csv from a fit_one result.

    features lists the columns the model used, after fit_one's constant-column drop. The training
    metrics show whether a model interpolates its labels, which an inverse ID-OOD line from label
    noise requires (Sanyal et al. 2025): _train_sub is on the rows it was fitted on (extra rows
    included), _train_all on every training-tile row of the training zone (no extra rows).
    hp_<name> holds each setting of the spec's kw; an empty cell means the family does not vary it.
    """
    sp, cols = got["sp"], got["cols"]
    acc_id, auc_id = got["id"]
    acc_od, auc_od = got["ood"]
    row = dict(
        idx=sp["idx"], kind=sp["kind"], feat_frac=sp["feat_frac"], row_frac=sp["row_frac"],
        n_feat=len(cols),
        score_id=float(got["s_id"].mean()), score_ood=float(got["s_od"].mean()),
        acc_id=acc_id, acc_ood=acc_od, auc_id=auc_id, auc_ood=auc_od,
        seed=sp["seed"], n_train=got["n_train"],
        features=";".join(feats[j] for j in cols),
        acc_train_sub=got["train_sub"][0], auc_train_sub=got["train_sub"][1],
        acc_train_all=got["train_all"][0], auc_train_all=got["train_all"][1],
    )
    for name, value in sp["kw"].items():
        row[f"hp_{name}"] = hp_value(value)
    return row


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


def run_cell(df, feats, protocol, train_dom, n_models, seed,
             n_jobs: int = 1, device: str = "cpu", extra_ids=None, extra_repeat: int = 1):
    """Fit and score the population of one cell; None when the cell is unusable."""
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
    specs = zoo_specs(n_models, seed)
    fitted = fit_population(specs, A, n_jobs, device)

    rows, S_ood, S_id = [], [], []
    for got in fitted:
        if got is None:
            continue
        S_id.append(got["s_id"])
        S_ood.append(got["s_od"])
        rows.append(model_row(got, feats))
    print(f"  {len(rows)} models in {time.time() - t0:.0f}s", flush=True)
    if not rows:                 # every draw unusable: same refusal as above
        return None
    models = pd.DataFrame(rows)
    hp_cols = sorted(c for c in models.columns if c.startswith("hp_"))
    other_cols = [c for c in models.columns if not c.startswith("hp_")]
    models = models[other_cols + hp_cols]
    print(f"  training accuracy on own subsample: median {models.acc_train_sub.median():.3f}"
          f" | >= 0.99 in {(models.acc_train_sub >= .99).sum()} of {len(models)} models"
          f" | on all training rows: median {models.acc_train_all.median():.3f}",
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
    return dict(models=models,
                S_ood=np.asarray(S_ood), S_id=np.asarray(S_id),
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
    ap.add_argument("--n-models", type=int, default=300)
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
    print(f"[{args.state}] {len(df):,} rows | {len(feats)} features")
    print(f"cell: protocol={args.protocol} train={args.train_dom} seed={args.seed} "
          f"models={args.n_models} device={args.device}")
    extra_ids = None
    if args.extra_train is not None:
        extra_ids = read_point_ids(args.extra_train)
    cell = run_cell(df, feats, args.protocol, args.train_dom, args.n_models, args.seed,
                    n_jobs=args.n_jobs, device=args.device, extra_ids=extra_ids,
                    extra_repeat=args.extra_train_repeat)
    if cell is None:
        print("cell unusable: nothing written")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    # S_ood / S_id: (models x points) true-class scores, rows in model-table order. y_id lets
    # readers recover exact argmax correctness on the ID half (ties at p = 0.5). tiles_ood is
    # kept for older readers. String columns are stored as fixed-width str, not objects.
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
    print(f"saved {out}/{tag}.npz  (S_ood {cell['S_ood'].shape}, S_id {cell['S_id'].shape}; "
          f"keys: {', '.join(payload)})")
    print(f"saved {out}/{tag}_models.csv  ({len(cell['models'])} rows x "
          f"{cell['models'].shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
