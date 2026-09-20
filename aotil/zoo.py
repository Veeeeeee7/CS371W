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

# Written into every cell alongside the scores. lon/lat build the spatial graph;
# the rest are the grouping keys for the restricted graphs (block, catchment,
# provenance) and for the characterisation.
SPATIAL_COLS = ["lon", "lat", "block_id", "tile_50km", "county_fips",
                "meta_comid", "meta_source_cit"]


# ----------------------------------------------------------------- data
def load(state: str, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    feats = [l.strip() for l in (data_dir / "FEATURES.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    feats = [f for tok in feats for f in tok.split()]
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


def true_class_prob(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example predicted probability of the observed class. Higher = better."""
    p = model.predict_proba(X)[:, 1]
    return np.where(y == 1, p, 1.0 - p)


# ----------------------------------------------------------------- run
def run_cell(df, feats, protocol, train_dom, n_models, seed,
             n_jobs: int = 1, device: str = "cpu"):
    eval_dom = "AE" if train_dom == "A" else "A"
    tr_tiles, te_tiles = split_tiles(df, seed)

    tr = subset(df, train_dom, protocol); tr = tr[tr[TILE].isin(tr_tiles)]
    id_te = subset(df, train_dom, protocol); id_te = id_te[id_te[TILE].isin(te_tiles)]
    ood_te = subset(df, eval_dom, protocol); ood_te = ood_te[ood_te[TILE].isin(te_tiles)]
    if len(tr) < MIN_TRAIN or id_te.empty or ood_te.empty:
        return None

    Xtr_all = tr[feats].to_numpy(float); ytr = tr.label.to_numpy()
    Xid = id_te[feats].to_numpy(float);  yid = id_te.label.to_numpy()
    Xod = ood_te[feats].to_numpy(float); yod = ood_te.label.to_numpy()

    print(f"  train {len(tr):,} | ID test {len(id_te):,} | OOD test {len(ood_te):,} "
          f"| tiles train {len(tr_tiles)} test {len(te_tiles)}", flush=True)

    def fit_one(sp):
        """One model of the population. Returns None when the draw is unusable.

        Isolated into a function so the population can be fitted in parallel:
        the models are independent by construction and the fits are the whole
        wall-clock cost of a cell (the selection that follows is seconds).
        """
        rng = np.random.default_rng(sp["seed"])
        cols = np.sort(rng.choice(len(feats),
                                  max(3, int(len(feats) * sp["feat_frac"])), replace=False))
        n = max(MIN_TRAIN, int(len(tr) * sp["row_frac"]))
        ridx = rng.choice(len(tr), min(n, len(tr)), replace=False)
        Xs, ys = Xtr_all[np.ix_(ridx, cols)], ytr[ridx]
        if len(np.unique(ys)) < 2:
            return None
        keep = [j for j in range(Xs.shape[1])
                if np.isfinite(Xs[:, j]).sum() >= 2 and np.unique(Xs[np.isfinite(Xs[:, j]), j]).size >= 2]
        if len(keep) < 2:
            return None
        cols, Xs = cols[keep], Xs[:, keep]
        try:
            m = build(sp["kind"], sp["kw"], sp["seed"], device=device); m.fit(Xs, ys)
            s_id = true_class_prob(m, Xid[:, cols], yid)
            s_od = true_class_prob(m, Xod[:, cols], yod)
        except Exception as e:
            print(f"    [skip {sp['kind']}] {type(e).__name__}: {e}", flush=True)
            return None
        return sp, cols, s_id, s_od

    t0 = time.time()
    specs = zoo_specs(n_models, seed)
    if n_jobs == 1:
        fitted = [fit_one(sp) for sp in specs]
    else:
        from joblib import Parallel, delayed
        fitted = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(fit_one)(sp) for sp in specs)

    rows, S_ood, S_id = [], [], []
    for got in fitted:
        if got is None:
            continue
        sp, cols, s_id, s_od = got
        S_id.append(s_id); S_ood.append(s_od)
        rows.append(dict(
            idx=sp["idx"], kind=sp["kind"], feat_frac=sp["feat_frac"], row_frac=sp["row_frac"],
            n_feat=len(cols),
            score_id=float(s_id.mean()), score_ood=float(s_od.mean()),
            acc_id=float(((s_id > .5)).mean()), acc_ood=float(((s_od > .5)).mean()),
            auc_id=float(roc_auc_score(yid, np.where(yid == 1, s_id, 1 - s_id))),
            auc_ood=float(roc_auc_score(yod, np.where(yod == 1, s_od, 1 - s_od))),
        ))
    print(f"  {len(rows)} models in {time.time()-t0:.0f}s", flush=True)

    # The spatial columns travel WITH the scores. The pilot stored only tiles,
    # which is why none of its cells can be used for the graph work: lon/lat are
    # join keys rather than features, so they are absent from FEATURES.txt and
    # were never carried through. Anything the graph or the characterisation
    # needs is written here, at the one point where the row order is still
    # guaranteed to match S_ood's columns.
    spatial = {c: ood_te[c].to_numpy() for c in SPATIAL_COLS if c in ood_te}
    missing = [c for c in SPATIAL_COLS if c not in ood_te]
    if missing:
        print(f"  note: not in this dataset, omitted from the cell: {missing}",
              flush=True)
    return dict(models=pd.DataFrame(rows),
                S_ood=np.asarray(S_ood), S_id=np.asarray(S_id),
                y_ood=yod, tiles_ood=ood_te[TILE].to_numpy(),
                spatial=spatial,
                protocol=protocol, train_dom=train_dom, eval_dom=eval_dom)


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
        print("cell unusable"); return 1

    out.mkdir(parents=True, exist_ok=True)
    payload = dict(S_ood=r["S_ood"], S_id=r["S_id"], y_ood=r["y_ood"],
                   tiles_ood=r["tiles_ood"].astype(str))
    for c, v in r["spatial"].items():
        payload[c] = v.astype(str) if v.dtype.kind in "OU" else v
    np.savez_compressed(out / f"{tag}.npz", **payload)
    r["models"].to_csv(out / f"{tag}_models.csv", index=False)
    print(f"saved {out}/{tag}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
