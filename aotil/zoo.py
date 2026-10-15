#!/usr/bin/env python3
"""
Model zoo + AoTL gate for one state, one (protocol, train-domain) cell.

WHAT THIS ANSWERS
-----------------
OODSelect (Salaudeen et al. 2025) finds OOD subsets on which higher ID accuracy
predicts LOWER OOD accuracy. That is a SLOPE across a population of models. It
presupposes two things we have never checked on this data:

  (a) the model population spans a usable range of ID accuracy, and
  (b) the aggregate ID->OOD relationship is positive to begin with.

If either fails there is no "line" to invert and OODSelect is not applicable.
This script builds the population and reports both, before any selection is run.

SCORES
------
We use the per-example predicted probability of the TRUE class
(p if y=1 else 1-p) rather than binary correctness. AUC has no per-example
decomposition, and thresholding at 0.5 would mean different things in different
cells because the achieved positive:negative ratio varies. The continuous score
keeps OODSelect's objective intact and removes the threshold choice. Binary
accuracy at 0.5 is also computed, purely as a cross-check that the gate verdict
does not depend on this decision.

SPLIT
-----
Tiles (tile_50km) are split once, 70/30, and every model in the population sees
the identical split. Test tiles are restricted to those holding BOTH domains, so
ID and OOD are scored on the same geography -- the same commitment baselines.py
makes. Points are never split individually.

WHAT A CELL STORES
------------------
<tag>.npz
    S_ood, S_id    (n_models, n_points) true-class scores; rows in the order of
                   <tag>_models.csv, columns in dataset row order of each half.
    y_ood, y_id    labels of the two halves.
    SPATIAL_COLS   one value per OOD column: point_id, lon, lat, block_id,
                   tile_50km, county_fips, meta_comid, meta_source_cit (plus
                   tiles_ood, kept for older readers).
    <c>_id         point_id_id, lon_id, lat_id: one value per ID column.
point_id is the join key back to the point table and, through meta_fld_ar_id /
meta_src_row_id, to the polygons. Cells written before it was stored can only be
re-aligned by re-running split_tiles + subset on the identical dataset, or joined
on rounded lon/lat, and neither survives a rebuild of the sample.

<tag>_models.csv, one row per fitted model
    idx, kind, feat_frac, row_frac, n_feat, score/acc/auc on ID and OOD   (as before)
    seed           the model's own seed from zoo_specs
    n_train        rows in its training subsample
    features       ';'-joined names of the features it actually used (after the
                   constant-column drop in fit_one)
    acc/auc_train_sub   on its own training subsample
    acc/auc_train_all   on every training-tile row of the training domain
    hp_<name>      one column per key of its kw dict, i.e. per setting zoo_specs
                   varies. An empty cell means the parameter is not varied
                   through kw for that family: build() fixes it or the library
                   default applies (list below). None (unlimited depth) is
                   written "none" -- not "None", which pandas.read_csv turns into
                   NaN -- so it cannot be confused with an empty cell; tuples
                   are written as their repr, e.g. "(64, 32)".
The training metrics are for the label-noise check in Related Work: Sanyal et al.
(2025) get an inverse line only from models that INTERPOLATE noisy labels, so the
paper has to show whether ours do. _train_sub is what the model was fitted on;
_train_all adds the training rows it never saw, from the same tiles. Both use the
same conventions as the ID/OOD columns (score_rows: accuracy = score > 0.5, AUC
ranked on the raw p).

AUCs in cells written before 2026-10-14 (the 2026-09-24 v1 cells) were ranked
on p rebuilt from the scores, which breaks ties between positives and negatives
(score_rows says how); their dt rows are off by up to 0.095. The S matrices,
score_* and acc_* are unaffected.

Fixed in build(), the same for every model of a family. random_state is the
model's seed in every family but nb; anything not listed and not in kw is the
installed library's default (e.g. XGBoost's learning_rate, 0.3).
    xgb      tree_method="hist", n_jobs=1, verbosity=0, device from --device;
             NaN handled natively
    hgb      early_stopping=False; NaN handled natively
    rf, et   median imputation, n_jobs=1
    dt       median imputation
    lr       median imputation, StandardScaler, max_iter=400
    mlp      median imputation, StandardScaler
    nb       median imputation, GaussianNB defaults

DETERMINISM
-----------
Each model draws its feature subset, row subsample and estimator seed from its
own spec seed, so a cell is a pure function of (dataset, seed) and any single
model can be refitted alone: audit_cell.py does exactly that through cell_data +
fit_one. The training metrics are computed after the fit and draw no random
numbers, so adding them changed no stored score. Thread counts are the one
other input: parallel workers are pinned to one BLAS/OpenMP thread (run_cell),
and a --n-jobs 1 run needs the same from its environment, as below.

USAGE (repo root)
-----
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    .venv/bin/python -u aotil/zoo.py --state FL --protocol band --train-dom AE \\
        --seed 0 --n-models 300 --data data/processed \\
        --out data/interim/aotil/cells --n-jobs -1
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Low max_iter on the MLP family is deliberate: an undertrained model is a
# legitimate low-ID-accuracy member of the population, and without that spread
# the ID-OOD correlation OODSelect minimises is undefined. sklearn warns on
# every one of them, which at 300 models buries the real output.
warnings.filterwarnings("ignore")
TILE = "tile_50km"
MIN_TRAIN = 500

# Written into every cell alongside the scores. point_id is the join key back to
# the point table; lon/lat build the spatial graph; the rest are the grouping
# keys for the restricted graphs (block, catchment, provenance) and for the
# characterisation.
SPATIAL_COLS = ["point_id", "lon", "lat", "block_id", "tile_50km", "county_fips",
                "meta_comid", "meta_source_cit"]
# The ID half, stored as <col>_id. Enough to join it (point_id) and to map it
# (lon/lat) without rebuilding the split.
ID_COLS = ["point_id", "lon", "lat"]


# ----------------------------------------------------------------- data
def read_features(path: Path) -> list[str]:
    """FEATURES.txt -> column names (comments and blank lines dropped)."""
    feats = [l.strip() for l in Path(path).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return [f for tok in feats for f in tok.split()]


def load(state: str, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    feats = read_features(data_dir / "FEATURES.txt")
    df = pd.read_parquet(data_dir / f"dataset_{state}.parquet")
    feats = [f for f in feats if f in df.columns]
    return df, feats


def subset(df: pd.DataFrame, domain: str, protocol: str) -> pd.DataFrame:
    """One domain's supervised set: its positives plus its own negatives.
    Identical to baselines.subset, so the cells line up with the main results."""
    return df[(df.domain == domain) &
              ((df.label == 1) | (df.neg_protocol == protocol))]


def split_tiles(df: pd.DataFrame, seed: int, test_frac: float = 0.30):
    """One tile split shared by every model. Test tiles must hold both domains."""
    rng = np.random.default_rng(seed)
    tiles = np.array(sorted(df[TILE].dropna().unique()))
    rng.shuffle(tiles)
    n_test = max(3, int(round(len(tiles) * test_frac)))
    test, train = set(tiles[:n_test]), set(tiles[n_test:])

    pos = df[df.label == 1]
    both = pos[pos[TILE].isin(test)].groupby(TILE).domain.nunique() == 2
    test = set(both[both].index)
    return train, test


# ----------------------------------------------------------------- zoo
def zoo_specs(n_target: int, seed: int) -> list[dict]:
    """A deliberately heterogeneous population.

    Diversity is induced on four axes at once -- learner family, capacity,
    feature subset and training subsample -- because a population of one family
    at one setting collapses to a single point of ID accuracy and the
    correlation OODSelect needs becomes undefined. Teney et al. (2023) make the
    same point from the other direction: the ID-OOD trend itself depends on how
    diverse the population is.
    """
    rng = np.random.default_rng(seed)
    fams = []
    for d in (2, 3, None):
        for it in (30, 120, 400):
            for lr in (0.02, 0.10, 0.35):
                fams.append(("hgb", dict(max_depth=d, max_iter=it, learning_rate=lr)))
    for d in (3, 6, None):
        for n in (40, 200):
            fams.append(("rf", dict(max_depth=d, n_estimators=n)))
            fams.append(("et", dict(max_depth=d, n_estimators=n)))
    for C in (0.003, 0.1, 3.0, 100.0):
        fams.append(("lr", dict(C=C)))
    for h in ((8,), (48,), (64, 32)):
        for it in (25, 120):
            fams.append(("mlp", dict(hidden_layer_sizes=h, max_iter=it)))
    for d in (2, 4, 10):
        fams.append(("dt", dict(max_depth=d)))
    fams.append(("nb", {}))
    # XGBoost is the one family here that can use a GPU, and it widens the tree
    # side of the population with a different regularisation story than sklearn's
    # boosting. Skipped silently when xgboost is absent so the laptop path does
    # not need it.
    if _HAS_XGB:
        for d in (2, 4, 8):
            for n in (50, 300):
                fams.append(("xgb", dict(max_depth=d, n_estimators=n)))

    # Round-robin across LEARNER FAMILIES, not across the flat config list.
    # Cycling the flat list makes the population 2/3 gradient boosting, which
    # leaves too few models of any other kind to support an architecture-disjoint
    # control -- the ablation that separates a real effect from one the selection
    # manufactured out of a single axis of model quality.
    by_kind: dict[str, list] = {}
    for k, kw in fams:
        by_kind.setdefault(k, []).append((k, kw))
    kinds = sorted(by_kind)
    counters = {k: 0 for k in kinds}

    specs = []
    for i in range(n_target):
        k = kinds[i % len(kinds)]
        kind, kw = by_kind[k][counters[k] % len(by_kind[k])]
        counters[k] += 1
        specs.append(dict(
            idx=i, kind=kind, kw=kw,
            feat_frac=float(rng.choice([0.3, 0.6, 1.0])),
            row_frac=float(rng.choice([0.15, 0.5, 1.0])),
            seed=int(rng.integers(1 << 30)),
        ))
    return specs


def build(kind: str, kw: dict, seed: int, device: str = "cpu"):
    if kind == "xgb":
        return XGBClassifier(random_state=seed, tree_method="hist",
                             device=("cuda" if device != "cpu" else "cpu"),
                             verbosity=0, n_jobs=1, **kw)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **kw)
    imp = SimpleImputer(strategy="median")
    if kind == "rf":
        return make_pipeline(imp, RandomForestClassifier(random_state=seed, n_jobs=1, **kw))
    if kind == "et":
        return make_pipeline(imp, ExtraTreesClassifier(random_state=seed, n_jobs=1, **kw))
    if kind == "dt":
        return make_pipeline(imp, DecisionTreeClassifier(random_state=seed, **kw))
    if kind == "lr":
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegression(max_iter=400, random_state=seed, **kw))
    if kind == "mlp":
        return make_pipeline(imp, StandardScaler(),
                             MLPClassifier(random_state=seed, **kw))
    if kind == "nb":
        return make_pipeline(imp, GaussianNB())
    raise ValueError(kind)


def score_rows(model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(true-class scores, accuracy at 0.5, AUC) of a fitted model on one set of
    rows: the one convention every S row and every score_* / acc_* / auc_*
    column of the model table uses.

    The score is the predicted probability of the observed class (p if y=1 else
    1-p), higher = better. The AUC is ranked on p itself, never on p rebuilt
    from the scores: for a negative that is 1 - (1 - p), up to an ulp off p, so
    a positive and a negative that share a p stop being a tie (half a pair) and
    are counted as ordered. Trees predict few distinct p, so ties are most
    pairs: a depth-2 tree (idx 24 of the 2026-09-24 cell) has OOD AUC 0.812 on
    p and 0.717 on the rebuilt p. The scores themselves are unchanged by this.
    """
    p = model.predict_proba(X)[:, 1]
    s = np.where(y == 1, p, 1.0 - p)
    return s, float((s > .5).mean()), float(roc_auc_score(y, p))


def hp_value(v):
    """A kw value as a model-table cell: None -> "none" (read_csv keeps it a
    string; "None" would come back as NaN), tuple -> its repr."""
    if v is None:
        return "none"
    if isinstance(v, (tuple, list)):
        return repr(tuple(v))
    return v


# ----------------------------------------------------------------- run
def cell_data(df, feats, protocol, train_dom, seed) -> dict:
    """The cell's three halves and their arrays, in the row order the cell stores.

    Shared by run_cell and audit_cell.py, so a refit sees exactly the rows and
    the feature order the population was fitted on.
    """
    eval_dom = "AE" if train_dom == "A" else "A"
    tr_tiles, te_tiles = split_tiles(df, seed)

    tr = subset(df, train_dom, protocol); tr = tr[tr[TILE].isin(tr_tiles)]
    id_te = subset(df, train_dom, protocol); id_te = id_te[id_te[TILE].isin(te_tiles)]
    ood_te = subset(df, eval_dom, protocol); ood_te = ood_te[ood_te[TILE].isin(te_tiles)]
    arrays = dict(Xtr=tr[feats].to_numpy(float), ytr=tr.label.to_numpy(),
                  Xid=id_te[feats].to_numpy(float), yid=id_te.label.to_numpy(),
                  Xod=ood_te[feats].to_numpy(float), yod=ood_te.label.to_numpy())
    return dict(eval_dom=eval_dom, tr_tiles=tr_tiles, te_tiles=te_tiles,
                tr=tr, id_te=id_te, ood_te=ood_te, arrays=arrays)


def fit_one(sp: dict, A: dict, device: str = "cpu") -> dict | None:
    """One model of the population. Returns None when the draw is unusable.

    Module-level so the population can be fitted in parallel (the models are
    independent by construction and the fits are the whole wall-clock cost of a
    cell; the selection that follows is seconds) and so audit_cell.py can refit
    any single model through the identical code path. A is cell_data's arrays.
    """
    Xtr, ytr = A["Xtr"], A["ytr"]
    n_feats = Xtr.shape[1]
    rng = np.random.default_rng(sp["seed"])
    cols = np.sort(rng.choice(n_feats,
                              max(3, int(n_feats * sp["feat_frac"])), replace=False))
    n = max(MIN_TRAIN, int(len(Xtr) * sp["row_frac"]))
    ridx = rng.choice(len(Xtr), min(n, len(Xtr)), replace=False)
    Xs, ys = Xtr[np.ix_(ridx, cols)], ytr[ridx]
    if len(np.unique(ys)) < 2:
        return None
    keep = [j for j in range(Xs.shape[1])
            if np.isfinite(Xs[:, j]).sum() >= 2 and np.unique(Xs[np.isfinite(Xs[:, j]), j]).size >= 2]
    if len(keep) < 2:
        return None
    cols, Xs = cols[keep], Xs[:, keep]
    try:
        m = build(sp["kind"], sp["kw"], sp["seed"], device=device); m.fit(Xs, ys)
        s_id, *m_id = score_rows(m, A["Xid"][:, cols], A["yid"])
        s_od, *m_od = score_rows(m, A["Xod"][:, cols], A["yod"])
        # Training fit, after the test scores and without touching the RNG:
        # how close each model comes to interpolating its labels.
        _, *m_sub = score_rows(m, Xs, ys)
        _, *m_all = score_rows(m, Xtr[:, cols], ytr)
    except Exception as e:
        print(f"    [skip {sp['kind']}] {type(e).__name__}: {e}", flush=True)
        return None
    # Each metric pair is (accuracy at 0.5, AUC).
    return dict(sp=sp, cols=cols, n_train=len(ridx), s_id=s_id, s_od=s_od,
                id=tuple(m_id), ood=tuple(m_od),
                train_sub=tuple(m_sub), train_all=tuple(m_all))


def model_row(got: dict, feats: list[str]) -> dict:
    """One row of <tag>_models.csv from a fit_one result."""
    sp, cols, s_id, s_od = got["sp"], got["cols"], got["s_id"], got["s_od"]
    (acc_id, auc_id), (acc_od, auc_od) = got["id"], got["ood"]
    row = dict(
        idx=sp["idx"], kind=sp["kind"], feat_frac=sp["feat_frac"], row_frac=sp["row_frac"],
        n_feat=len(cols),
        score_id=float(s_id.mean()), score_ood=float(s_od.mean()),
        acc_id=acc_id, acc_ood=acc_od, auc_id=auc_id, auc_ood=auc_od,
        seed=sp["seed"], n_train=got["n_train"],
        features=";".join(feats[j] for j in cols),
        acc_train_sub=got["train_sub"][0], auc_train_sub=got["train_sub"][1],
        acc_train_all=got["train_all"][0], auc_train_all=got["train_all"][1],
    )
    row.update({f"hp_{k}": hp_value(v) for k, v in sp["kw"].items()})
    return row


def run_cell(df, feats, protocol, train_dom, n_models, seed,
             n_jobs: int = 1, device: str = "cpu"):
    d = cell_data(df, feats, protocol, train_dom, seed)
    tr, id_te, ood_te, A = d["tr"], d["id_te"], d["ood_te"], d["arrays"]
    if len(tr) < MIN_TRAIN or id_te.empty or ood_te.empty:
        return None

    print(f"  train {len(tr):,} | ID test {len(id_te):,} | OOD test {len(ood_te):,} "
          f"| tiles train {len(d['tr_tiles'])} test {len(d['te_tiles'])}", flush=True)
    # A protocol with no negatives in the dataset (shadedX on the band-only v2
    # sample) leaves every set positives-only. No model can be fitted or
    # scored on one label, and a cell with zero models would be written and
    # then break every reader, so it is refused here, before 300 empty fits.
    one_label = [h for h, y in (("train", A["ytr"]), ("ID test", A["yid"]),
                                ("OOD test", A["yod"])) if len(np.unique(y)) < 2]
    if one_label:
        print(f"  one label only in {', '.join(one_label)} (protocol {protocol}): "
              f"no model can be fitted or scored", flush=True)
        return None

    t0 = time.time()
    specs = zoo_specs(n_models, seed)
    if n_jobs == 1:
        fitted = [fit_one(sp, A, device) for sp in specs]
    else:
        # One BLAS/OpenMP thread per worker whatever n_jobs is. joblib otherwise
        # gives each worker cpu_count // n_jobs threads for every variable the
        # parent leaves unset, and Apple Accelerate (numpy's BLAS on the laptop,
        # invisible to threadpoolctl) moves LogisticRegression's scores by up to
        # ~1e-11 between thread counts, so the cell would depend on --n-jobs.
        # With 1 thread a --n-jobs 4 cell equals the 2026-09-24 --n-jobs -1 one.
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            fitted = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(fit_one)(sp, A, device) for sp in specs)

    rows, S_ood, S_id = [], [], []
    for got in fitted:
        if got is None:
            continue
        S_id.append(got["s_id"]); S_ood.append(got["s_od"])
        rows.append(model_row(got, feats))
    print(f"  {len(rows)} models in {time.time()-t0:.0f}s", flush=True)
    if not rows:                 # every draw unusable: same refusal as above
        return None
    models = pd.DataFrame(rows)
    hp = sorted(c for c in models.columns if c.startswith("hp_"))
    models = models[[c for c in models.columns if not c.startswith("hp_")] + hp]
    print(f"  training accuracy on own subsample: median {models.acc_train_sub.median():.3f}"
          f" | >= 0.99 in {(models.acc_train_sub >= .99).sum()} of {len(models)} models"
          f" | on all training rows: median {models.acc_train_all.median():.3f}",
          flush=True)

    # The spatial columns travel WITH the scores. The pilot stored only tiles,
    # which is why none of its cells can be used for the graph work: lon/lat are
    # join keys rather than features, so they are absent from FEATURES.txt and
    # were never carried through. Anything the graph or the characterisation
    # needs is written here, at the one point where the row order is still
    # guaranteed to match S_ood's (and S_id's) columns.
    spatial = {c: ood_te[c].to_numpy() for c in SPATIAL_COLS if c in ood_te}
    id_cols = {f"{c}_id": id_te[c].to_numpy() for c in ID_COLS if c in id_te}
    missing = ([c for c in SPATIAL_COLS if c not in ood_te]
               + [f"{c}_id" for c in ID_COLS if c not in id_te])
    if missing:
        print(f"  note: not in this dataset, omitted from the cell: {missing}",
              flush=True)
    return dict(models=models,
                S_ood=np.asarray(S_ood), S_id=np.asarray(S_id),
                y_ood=A["yod"], y_id=A["yid"], tiles_ood=ood_te[TILE].to_numpy(),
                spatial=spatial, id_cols=id_cols,
                protocol=protocol, train_dom=train_dom, eval_dom=d["eval_dom"])


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
    a = ap.parse_args()

    tag = f"{a.state}_{a.protocol}_{a.train_dom}_s{a.seed}"
    out = Path(a.out)
    # Cells are expensive and results append, so an existing cell is left alone
    # unless asked for. A half-finished sweep resumes by re-running the same
    # command rather than by editing the state list.
    if (out / f"{tag}.npz").exists() and not a.overwrite:
        print(f"{tag}: cell exists, skipping (--overwrite to refit)")
        return 0

    df, feats = load(a.state, Path(a.data))
    print(f"[{a.state}] {len(df):,} rows | {len(feats)} features")
    print(f"cell: protocol={a.protocol} train={a.train_dom} seed={a.seed} "
          f"models={a.n_models} device={a.device}")
    r = run_cell(df, feats, a.protocol, a.train_dom, a.n_models, a.seed,
                 n_jobs=a.n_jobs, device=a.device)
    if r is None:
        print("cell unusable: nothing written"); return 1

    out.mkdir(parents=True, exist_ok=True)
    # y_id lets a consumer recover exact argmax correctness on the ID half
    # (oodselect_paper.py needs it for ties at p = 0.5).
    payload = dict(S_ood=r["S_ood"], S_id=r["S_id"], y_ood=r["y_ood"], y_id=r["y_id"],
                   tiles_ood=r["tiles_ood"].astype(str))
    for c, v in {**r["spatial"], **r["id_cols"]}.items():
        payload[c] = v.astype(str) if v.dtype.kind in "OU" else v
    np.savez_compressed(out / f"{tag}.npz", **payload)
    r["models"].to_csv(out / f"{tag}_models.csv", index=False)
    print(f"saved {out}/{tag}.npz  (S_ood {r['S_ood'].shape}, S_id {r['S_id'].shape}; "
          f"keys: {', '.join(payload)})")
    print(f"saved {out}/{tag}_models.csv  ({len(r['models'])} rows x "
          f"{r['models'].shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
