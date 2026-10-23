#!/usr/bin/env python3
"""
o5_shift_decomp.py -- do the models fail on the regions through covariate or concept shift?
(docs/TASK-O5-REGIONS.md Part C; results in docs/O5-REGIONS-REPORT-2026-10-17.md, Part C.)

C1 support: how far region land lies from the Zone AE training set (kNN distance, propensity).
C2 label conflict (the primary): the share of a point's k nearest Zone AE training points with the
   other label, region minus terrain-matched controls, pooled over seeds.
C3 decomposition: the accuracy drop from Zone AE test land to region land, split by reweighting.

Reads   <data>/FEATURES.txt + dataset_FL.parquet, <cells-dir>/<cell>_s<seed>.npz + _models.csv,
        the regions and selections (through match_controls),
        <o5-dir>/regions_land.csv, S1_controls.csv, S2_null.parquet
Writes  <out-dir>/C_support.csv (C1), C_conflict.csv (C2), C_decomposition.csv (C3);
        --points-out: one row per scored point (optional)

    .venv/bin/python aotil/o5_shift_decomp.py --out-dir results/2026-10-17/o5 \\
        2>&1 | tee logs/2026-10-17/C_shift_decomp.log
"""
import argparse
import os
import sys
import time
import zlib
from pathlib import Path

# Thread defaults must be set before numpy loads: this runs next to the zoo builds.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import match_controls as mc  # noqa: E402
import oodselect_paper as op  # noqa: E402
import zoo  # noqa: E402
from cells import load_cell  # noqa: E402

RNG_SEED = 20261018
KS = (10, 25, 50)                       # neighbour counts of conflict_k; 25 is the primary
K_PRIMARY = 25
MISS_COLS = ("hand", "wetindexcat")     # missing on open water; each gets a 0/1 indicator column
WATER = 11                              # NLCD land-cover code of open water
N_FOLDS = 5
N_BOOT = 1000
CLIP_Q = 0.99                           # C3 clips the propensity odds at this quantile
TOP_Q = 15                              # C3's model set: the strongest quarter of the 60 test models
TIE_TOL = 1e-12                         # null values this close to the observed value count as ties
TYPES = ("positive-dominated", "negative-dominated")
HGB_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
                  early_stopping=False)
LABEL_SUBSETS = ("all", 1, 0)
CONFLICT = tuple(f"conflict_k{k}" for k in KS)
# Per-point measures: conflict_* and knn_dist_k25 exist for every test point, propensity only for
# the C1 evaluation set.
MEASURES = CONFLICT + ("knn_dist_k25", "propensity")
TERRAIN_Z = tuple(f"z_{v}" for v in mc.TERRAIN)     # terrain in seed-SD units, as S1 matched on
# match_controls.load_contexts: its point table, the selection budget S and the minimum region size
CONTEXT_TABLE = "data/processed/dataset_FL.parquet"
BUDGET = 500
MIN_REGION_SIZE = 20
C3_BOOT = ("covariate", "concept", "total", "acc_ae_w", "acc_region", "covariate_mu", "concept_mu",
           "covariate_mix", "concept_mix")
C3_MEAN_COLS = ("prop_auc", "mean_p_region", "ess", "max_abs_dmean_unw", "max_abs_dmean_w", "pos_share_region",
                "pos_share_ae", "pos_share_ae_w", "acc_ae", "acc_ae_w", "acc_region", "acc_controls", "acc_ae_mu",
                "acc_ae_w_mix", "ae_weight_mass", "region_point_share")
CSV_FLOAT = "%.6g"


# ----------------------------------------------------------------------------- feature space
class FeatureSpace:
    """The space all distances are measured in: the model features with training-median imputation,
    a 0/1 missingness column per MISS_COLS entry, every column standardized by the training mean and
    population SD (an SD-0 column is only centred). landcover stays the numeric NLCD class code, as
    the models see it, so "near" here is near in the models' own input."""

    def __init__(self, feats, miss_cols=MISS_COLS):
        self.feats = list(feats)
        self.miss_idx = [self.feats.index(c) for c in miss_cols]
        self.columns = self.feats + [f"miss_{c}" for c in miss_cols]

    def fit(self, Xraw: np.ndarray) -> "FeatureSpace":
        Xraw = np.asarray(Xraw, float)
        self.median = np.nanmedian(Xraw, axis=0)
        if not np.all(np.isfinite(self.median)):
            raise ValueError("a feature is missing on every training row")
        filled = self._fill(Xraw)
        self.mean = filled.mean(axis=0)
        sd = filled.std(axis=0)
        self.zero_sd = [c for c, s in zip(self.columns, sd) if s == 0]
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def _fill(self, Xraw: np.ndarray) -> np.ndarray:
        Xraw = np.asarray(Xraw, float)
        indicators = np.isnan(Xraw[:, self.miss_idx]).astype(float)
        X = np.where(np.isnan(Xraw), self.median[None, :], Xraw)
        return np.hstack([X, indicators])

    def transform(self, Xraw: np.ndarray) -> np.ndarray:
        return (self._fill(Xraw) - self.mean) / self.sd


def knn_stats(tree: cKDTree, ytr: np.ndarray, Xq: np.ndarray, yq: np.ndarray, ks=KS,
              k_dist: int = K_PRIMARY, workers: int = 2) -> dict:
    """Per query point: conflict_k{k}, the share of its k nearest training points whose label differs
    from yq, for each k in ks; and knn_dist_k{k_dist}, the mean distance to its k_dist nearest.
    Ties at the k-th distance are broken by cKDTree's exact search."""
    if len(Xq) == 0:
        out = {f"conflict_k{k}": np.zeros(0) for k in ks}
        out[f"knn_dist_k{k_dist}"] = np.zeros(0)
        return out
    dist, idx = tree.query(Xq, k=max(max(ks), k_dist), workers=workers)
    differs = ytr[idx] != np.asarray(yq)[:, None]
    out = {f"conflict_k{k}": differs[:, :k].mean(axis=1) for k in ks}
    out[f"knn_dist_k{k_dist}"] = dist[:, :k_dist].mean(axis=1)
    return out


def crossfit_propensity(X0: np.ndarray, X1: np.ndarray, seed: int, n_folds: int = N_FOLDS,
                        params: dict | None = None) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Cross-fitted P(class 1 | x) for the rows of X0 (class 0) and X1 (class 1).

    Stratified folds over the stacked rows; each row is scored by the model fitted on the other
    folds, so no model scores a row it saw. Returns (p0, p1, AUC of the cross-fitted p, fold of every
    stacked row).
    """
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(len(X0), int), np.ones(len(X1), int)]
    p = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (fit_rows, score_rows) in enumerate(folds.split(X, y)):
        model = HistGradientBoostingClassifier(random_state=seed, **(params or HGB_PARAMS))
        model.fit(X[fit_rows], y[fit_rows])
        p[score_rows] = model.predict_proba(X[score_rows])[:, 1]
        fold[score_rows] = f
    return p[:len(X0)], p[len(X0):], float(roc_auc_score(y, p)), fold


def composition(Xreg: np.ndarray, reg_unit: np.ndarray, Xq: np.ndarray, seed: int,
                params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """G[i, r] = P(region r | x_i, x is from some region): one multiclass classifier fitted on the
    region points (labels reg_unit), evaluated at Xq. Returns (G, the region labels in column order);
    the rows of G sum to 1."""
    units = np.unique(reg_unit)
    if len(units) == 1:
        return np.ones((len(Xq), 1)), units
    model = HistGradientBoostingClassifier(random_state=seed, **(params or HGB_PARAMS))
    model.fit(Xreg, reg_unit)
    # sklearn orders predict_proba's columns by the sorted labels, i.e. as `units`
    return model.predict_proba(Xq), units


def clip_odds(odds: np.ndarray, clip_q: float = CLIP_Q) -> tuple[np.ndarray, float]:
    """The propensity odds clipped at their clip_q quantile, and the cap."""
    cap = float(np.quantile(odds, clip_q))
    return np.minimum(odds, cap), cap


def c3_weights(oc: np.ndarray, G: np.ndarray, m: np.ndarray):
    """Normalised Zone AE weights oc * (G @ m) for region multiplicities m, and their Kish ESS.

    oc are the clipped odds of "region vs Zone AE"; since sum_r q_r is the pooled region density,
    oc_i * sum_r m_r g_ir is the density ratio to the m-weighted mixture of regions. With every
    m_r = 1 the factor is 1 and the weights are the clipped binary odds, normalised.
    """
    raw = oc * (G @ np.asarray(m, float))
    total = raw.sum()
    if not total > 0:
        return np.full(len(raw), np.nan), np.nan
    w = raw / total
    return w, float(1.0 / np.sum(w ** 2))


def midrank_percentile(obs: float, null, tol: float = TIE_TOL) -> float:
    """100 * (#{null < obs} + 0.5 #{|null - obs| <= tol}) / #null over the finite null values."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan
    tie = np.abs(null - obs) <= tol
    below = (null < obs) & ~tie
    return float(100.0 * (below.sum() + 0.5 * tie.sum()) / len(null))


def boot_multiplicities(n_units: int, n_boot: int, key: str, rng_seed: int = RNG_SEED) -> np.ndarray:
    """(n_boot, n_units) multiplicities of resampling the units with replacement, from a generator
    seeded by (rng_seed, crc32(key)) so each scope always gets the same draws."""
    rng = np.random.default_rng([rng_seed, zlib.crc32(key.encode())])
    picks = rng.integers(0, n_units, size=(n_boot, n_units))
    counts = [np.bincount(row, minlength=n_units) for row in picks]
    return np.stack(counts).astype(float)


# ----------------------------------------------------------------------------- paired accumulators (C1, C2)
def pair_acc(stats: pd.DataFrame, qids, cids, measures=MEASURES) -> dict:
    """Sufficient statistics of one set of (query, control) pairs; pooled statistics over any
    resample of pair sets are then sums of these.

    stats is indexed by point_id and holds the measures and TERRAIN_Z. Returns n (pairs); per
    measure, sum_diff (query - control summed over the pairs where both are finite) and n_m (their
    number); and for the terrain-adjusted OLS over the 2n stacked points with rows
    [1, is_query, terrain z], XtX and per measure Xty (NaN when a pair has a missing value).
    """
    qids, cids = list(qids), list(cids)
    out = dict(n=len(qids), sum_diff={}, n_m={}, Xty={})
    Q = stats.loc[qids] if qids else stats.iloc[:0]
    C = stats.loc[cids] if cids else stats.iloc[:0]
    Xq = np.column_stack([np.ones(len(Q)), np.ones(len(Q)), Q[list(TERRAIN_Z)].to_numpy(float)])
    Xc = np.column_stack([np.ones(len(C)), np.zeros(len(C)), C[list(TERRAIN_Z)].to_numpy(float)])
    X = np.vstack([Xq, Xc])
    if not np.all(np.isfinite(X)):
        raise ValueError("missing terrain on a matched point")
    out["XtX"] = X.T @ X
    for v in measures:
        q_vals = Q[v].to_numpy(float)
        c_vals = C[v].to_numpy(float)
        both = np.isfinite(q_vals) & np.isfinite(c_vals)
        out["sum_diff"][v] = float((q_vals[both] - c_vals[both]).sum())
        out["n_m"][v] = int(both.sum())
        if both.all():
            out["Xty"][v] = X.T @ np.r_[q_vals, c_vals]
        else:
            out["Xty"][v] = np.full(X.shape[1], np.nan)
    return out


def pooled_diff(accs: list[dict], m, v: str) -> float:
    """Pair-weighted mean query - control difference of measure v, pair set j counted m[j] times."""
    m = np.asarray(m, float)
    n_pairs = sum(mi * a["n_m"][v] for mi, a in zip(m, accs))
    if not n_pairs > 0:
        return np.nan
    total = sum(mi * a["sum_diff"][v] for mi, a in zip(m, accs))
    return float(total / n_pairs)


def pooled_adj(accs: list[dict], m, v: str) -> float:
    """Terrain-adjusted difference: the coefficient on the 0/1 region indicator of the OLS
    v ~ 1 + region + terrain over the pooled pairs (pair set j counted m[j] times). It guards
    against the terrain imbalance S1's caliper left."""
    m = np.asarray(m, float)
    n_pairs = sum(mi * a["n_m"][v] for mi, a in zip(m, accs))
    if not n_pairs > 0:
        return np.nan
    XtX = sum(mi * a["XtX"] for mi, a in zip(m, accs))
    Xty = sum(mi * a["Xty"][v] for mi, a in zip(m, accs))
    if not np.all(np.isfinite(Xty)):
        return np.nan
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    return float(beta[1])


def null_values(null_accs: dict, units: list, v: str, adjusted: bool, n_draws: int) -> np.ndarray:
    """The null distribution: draw d pools the d-th S2 patch of every unit, pair-weighted, like the
    observed statistic. A draw where a unit has no patch is NaN."""
    pooled = pooled_adj if adjusted else pooled_diff
    out = np.full(n_draws, np.nan)
    for d in range(n_draws):
        accs = [null_accs[(u, d)] for u in units if (u, d) in null_accs]
        if len(accs) == len(units):
            out[d] = pooled(accs, np.ones(len(accs)), v)
    return out


# ----------------------------------------------------------------------------- per seed
def check_cell_order(tag: str, od: pd.DataFrame, idt: pd.DataFrame, cell, ctx: mc.SeedContext) -> None:
    """Stop unless zoo.cell_data's test rows, the cell and the match_controls context agree."""
    od_ids = od.point_id.to_numpy().astype(str)
    id_ids = idt.point_id.to_numpy().astype(str)
    same_order = (np.array_equal(od_ids, np.asarray(cell.point_id).astype(str))
                  and np.array_equal(id_ids, np.asarray(cell.point_id_id).astype(str)))
    if not same_order:
        raise SystemExit(f"{tag}: zoo.cell_data rows do not reproduce the cell's point order")
    if not (np.array_equal(od.label.to_numpy(), cell.y_ood) and np.array_equal(idt.label.to_numpy(), cell.y_id)):
        raise SystemExit(f"{tag}: labels differ from the cell's")
    if not np.array_equal(od_ids, ctx.pts.point_id.to_numpy().astype(str)):
        raise SystemExit(f"{tag}: match_controls context is not in cell order")


def half_table(rows: pd.DataFrame, half: str, knn: dict) -> pd.DataFrame:
    """One test half's points: id, half ("A" = Zone A, "AE" = Zone AE), label, tile, land flag, kNN measures."""
    return pd.DataFrame({"point_id": rows.point_id.astype(str).to_numpy(), "half": half,
                         "label": rows.label.to_numpy(), "tile": rows.tile_50km.astype(str).to_numpy(),
                         "land": rows.landcover.to_numpy() != WATER, **knn})


def seed_points(s: int, df: pd.DataFrame, feats, ctx: mc.SeedContext, cells_dir: Path,
                cell_prefix: str, K: pd.DataFrame, threads: int, log=print):
    """Feature space, kNN measures and the C1 propensity for one seed.

    The Zone AE training rows are rebuilt with zoo.cell_data exactly as the 300 models saw them and
    are kept whole, water included, as the kNN reference. Returns (a table indexed by point_id of
    every Zone A and Zone AE test point, an info dict, the arrays c3_seed needs).
    """
    tag = f"{cell_prefix}_s{s}"
    halves = zoo.cell_data(df, feats, "band", "AE", s)
    tr, idt, od = halves["tr"], halves["id_te"], halves["ood_te"]
    cell = load_cell(cells_dir, tag)
    check_cell_order(tag, od, idt, cell, ctx)

    space = FeatureSpace(feats).fit(tr[feats].to_numpy(float))
    Xtr = space.transform(tr[feats].to_numpy(float))
    ytr = tr.label.to_numpy()
    Xod = space.transform(od[feats].to_numpy(float))
    Xid = space.transform(idt[feats].to_numpy(float))
    tree = cKDTree(Xtr)
    ood = half_table(od, "A", knn_stats(tree, ytr, Xod, od.label.to_numpy(), workers=threads))
    ae = half_table(idt, "AE", knn_stats(tree, ytr, Xid, idt.label.to_numpy(), workers=threads))
    for zc in TERRAIN_Z:
        ood[zc] = ctx.pts[zc].to_numpy()
        ae[zc] = np.nan                     # terrain z exists only for Zone A points

    P = pd.concat([ood, ae], ignore_index=True)
    if P.point_id.duplicated().any():
        raise SystemExit(f"{tag}: a point id is in both test halves")
    P["seed"] = s
    P["component"] = -1
    regions = ctx.regions
    in_region = P.point_id.isin(regions.point_id) & (P.half == "A")
    component_of = regions.set_index("point_id").component
    P.loc[in_region, "component"] = P.loc[in_region, "point_id"].map(component_of).to_numpy()
    matched = K[(K.seed == s) & (K.status == "matched")]
    P["is_control"] = P.point_id.isin(matched.control_point_id) & (P.half == "A")
    if (P.is_control & (P.component >= 0)).any():
        raise SystemExit(f"{tag}: a control is a region point")
    P["propensity"] = np.nan
    P["fold"] = -1

    # C1 propensity: Zone AE training rows (class 0) vs one evaluation pool (class 1) of the region
    # land points, their S1 controls and the land Zone AE test points. Analysis points are land only.
    zone_a_eval = (P.half == "A") & P.land & ((P.component >= 0) | P.is_control)
    in_eval = zone_a_eval | ((P.half == "AE") & P.land)
    X_test = np.vstack([Xod, Xid])
    p_train, p_eval, auc, fold = crossfit_propensity(Xtr, X_test[in_eval.to_numpy()], seed=RNG_SEED + s)
    P.loc[in_eval, "propensity"] = p_eval
    P.loc[in_eval, "fold"] = fold[len(Xtr):]

    n_eval = int(in_eval.sum())
    info = dict(seed=s, n_train=len(tr), n_train_water=int((tr.landcover == WATER).sum()),
                n_train_pos=int(ytr.sum()), n_ae_test=len(idt), n_ae_test_land=int(ae.land.sum()),
                n_ood=len(od), n_ood_land=int(ood.land.sum()), n_eval=n_eval,
                prop_auc_train_vs_eval=auc, mean_p_train=float(p_train.mean()),
                base_rate=float(in_eval.sum() / (in_eval.sum() + len(tr))), zero_sd=";".join(space.zero_sd),
                median=dict(zip(space.columns, np.r_[space.median, [np.nan] * len(MISS_COLS)])),
                miss_share_train={c: float(np.isnan(tr[c].to_numpy(float)).mean()) for c in MISS_COLS})
    missing = ", ".join(f"{c} {v:.3f}" for c, v in info["miss_share_train"].items())
    log(f"seed {s}: train {len(tr)} Zone AE rows ({info['n_train_water']} water, {info['n_train_pos']} positives); "
        f"missing in train: {missing}; Zone AE test {len(idt)} ({info['n_ae_test_land']} land); "
        f"Zone A test {len(od)} ({info['n_ood_land']} land); SD-0 columns: {info['zero_sd'] or 'none'}")
    # the base rate n_eval / (n_eval + n_train) is about 0.2-0.25: compare groups, not against 0.5
    log(f"  C1 propensity (train vs {n_eval} eval points: region land "
        f"{int(((P.component >= 0) & P.land).sum())}, controls {int(P.is_control.sum())}, AE test land "
        f"{info['n_ae_test_land']}): cross-fitted AUC {auc:.3f}; base rate {info['base_rate']:.3f}; "
        f"mean p on training rows {info['mean_p_train']:.3f}")
    c3 = dict(cell=cell, Xod=Xod, Xid=Xid, od=od, idt=idt)
    return P.set_index("point_id", drop=False), info, c3


# ----------------------------------------------------------------------------- C1 + C2
def label_subset(frame: pd.DataFrame, lab) -> pd.DataFrame:
    return frame if lab == "all" else frame[frame.label == lab]


def unit_tables(P_by_seed: dict, K: pd.DataFrame, rl: pd.DataFrame):
    """Observed pair accumulators of every region with land points (a unit is (seed, component)),
    over all its matched pairs and per label. Returns (units, {(unit, label subset): accumulator})."""
    units, accs = [], {}
    for _, r in rl[rl.n_land > 0].iterrows():
        unit = (int(r.seed), int(r.component))
        units.append(unit)
        pairs = K[(K.seed == unit[0]) & (K.component == unit[1]) & (K.status == "matched")]
        P = P_by_seed[unit[0]]
        for lab in LABEL_SUBSETS:
            sub = label_subset(pairs, lab)
            accs[(unit, lab)] = pair_acc(P, sub.point_id.astype(str), sub.control_point_id.astype(str))
    return units, accs


def null_tables(P_by_seed: dict, N: pd.DataFrame, units: list):
    """Null accumulators from S2's matched-growth patches: {label subset: {(unit, draw): accumulator}}.
    Each patch's land points are paired with their controls; unmatched patch points drop out, as
    unmatched region points do."""
    out = {lab: {} for lab in LABEL_SUBSETS}
    wanted = set(units)
    for row in N.itertuples(index=False):
        unit = (int(row.seed), int(row.component))
        if unit not in wanted:
            continue
        P = P_by_seed[unit[0]]
        qids = np.asarray(row.land_point_ids, dtype=object)
        cids = np.asarray(row.control_point_ids, dtype=object)
        matched = np.array([isinstance(x, str) for x in cids], bool)
        qids, cids = qids[matched].astype(str), cids[matched].astype(str)
        q_labels = P.loc[qids, "label"].to_numpy() if len(qids) else np.zeros(0, int)
        if len(qids) and not np.array_equal(q_labels, P.loc[cids, "label"].to_numpy()):
            raise SystemExit(f"S2 seed {unit[0]} comp {unit[1]} draw {row.draw}: control of another label")
        for lab in LABEL_SUBSETS:
            keep = np.ones(len(qids), bool) if lab == "all" else q_labels == lab
            # no propensity: patch points are outside the C1 evaluation set
            out[lab][(unit, int(row.draw))] = pair_acc(P, qids[keep], cids[keep], measures=MEASURES[:-1])
    return out


def units_of(regions: pd.DataFrame) -> list:
    return [(int(r.seed), int(r.component)) for _, r in regions.iterrows()]


def scopes(rl: pd.DataFrame, seeds) -> list[tuple[str, list, object]]:
    """(name, units, label subset) of every pooled scope with a unit; the first is the primary's."""
    land = rl[rl.n_land > 0]
    all_units = units_of(land)
    out = [("all", all_units, "all")]
    for t in TYPES:
        out.append((t, units_of(land[land.type == t]), "all"))
    for s in seeds:
        out.append((f"seed={s}", [u for u in all_units if u[0] == s], "all"))
    for lab in (1, 0):
        out.append((f"label={lab}", all_units, lab))
    return [scope for scope in out if scope[1]]


def finite_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def finite_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def concat_or_empty(parts: list) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0)


def descriptives(P_by_seed: dict, K: pd.DataFrame, units: list, lab, v: str) -> dict:
    """Levels of measure v over the units (point-weighted, label subset lab): all region land points,
    the matched region points, their controls, the land Zone AE test points and, as a local
    baseline, those in the regions' own 50 km tiles."""
    parts = {"region": [], "matched": [], "controls": [], "ae": [], "ae_tile": []}
    for s in sorted({u[0] for u in units}):
        P = P_by_seed[s]
        comps = [u[1] for u in units if u[0] == s]
        zone_a_land = P[(P.half == "A") & P.land]
        R = label_subset(zone_a_land[zone_a_land.component.isin(comps)], lab)
        pairs = label_subset(K[(K.seed == s) & K.component.isin(comps) & (K.status == "matched")], lab)
        AE = label_subset(P[(P.half == "AE") & P.land], lab)
        parts["region"].append(R[v].to_numpy(float))
        parts["matched"].append(P.loc[pairs.point_id.astype(str), v].to_numpy(float))
        parts["controls"].append(P.loc[pairs.control_point_id.astype(str), v].to_numpy(float))
        parts["ae"].append(AE[v].to_numpy(float))
        parts["ae_tile"].append(AE[AE.tile.isin(set(R.tile))][v].to_numpy(float))
    region = concat_or_empty(parts["region"])
    matched = concat_or_empty(parts["matched"])
    controls = concat_or_empty(parts["controls"])
    ae = concat_or_empty(parts["ae"])
    ae_tile = concat_or_empty(parts["ae_tile"])
    return dict(n_region_land=len(region), n_pairs=len(matched), n_ae_test=len(ae), n_ae_test_tile=len(ae_tile),
                region_all_mean=finite_mean(region), region_matched_mean=finite_mean(matched),
                controls_mean=finite_mean(controls), ae_test_mean=finite_mean(ae),
                ae_test_tile_mean=finite_mean(ae_tile), region_all_median=finite_median(region),
                controls_median=finite_median(controls), ae_test_median=finite_median(ae))


def contrast_summary(obs: float, boot: np.ndarray | None, null: np.ndarray) -> dict:
    """diff, its bootstrap interval (NaN without a bootstrap) and its place in the null."""
    if boot is None:
        ci_lo, ci_hi, n_boot = np.nan, np.nan, 0
    else:
        has_boot = np.isfinite(boot).any()
        ci = np.nanpercentile(boot, [2.5, 97.5]) if has_boot else [np.nan, np.nan]
        ci_lo, ci_hi, n_boot = float(ci[0]), float(ci[1]), int(np.isfinite(boot).sum())
    has_null = np.isfinite(null).any()
    return dict(diff=obs, ci_lo=ci_lo, ci_hi=ci_hi, n_boot=n_boot,
                null_mean=float(np.nanmean(null)) if has_null else np.nan,
                null_p2_5=float(np.nanpercentile(null, 2.5)) if has_null else np.nan,
                null_p97_5=float(np.nanpercentile(null, 97.5)) if has_null else np.nan,
                null_percentile=midrank_percentile(obs, null), n_null=int(np.isfinite(null).sum()))


def scope_seed(name: str) -> int:
    return int(name.split("=")[1]) if name.startswith("seed=") else -1


def contrast_rows(P_by_seed, K, rl, units, accs, nulls, seeds, measures, n_boot, n_draws,
                  with_adjusted: bool, primary_measure: str | None, label_col: str):
    """Rows of C_conflict (C2) or C_support (C1): every pooled scope, then every region.

    diff is region minus control over the matched pairs (raw), or pooled_adj (terrain_adjusted).
    Pooled rows get a 95% percentile interval from resampling the scope's regions; every row of a
    measure with a null (not propensity) gets the S2 null and the mid-rank percentile of diff in it.
    Per-region rows use that region's own patches and have no interval (a region bootstrap needs
    several regions). A region enters a label scope, its bootstrap and its null only if it has a
    matched pair of that label. met is the primary's rule: interval above 0 and null percentile
    above 95.
    """
    region_type = {(int(r.seed), int(r.component)): r.type for _, r in rl.iterrows()}
    analyses = (False, True) if with_adjusted else (False,)
    no_null = np.full(0, np.nan)
    rows = []
    for name, scope_units, lab in scopes(rl, seeds):
        if lab != "all":
            scope_units = [u for u in scope_units if accs[(u, lab)]["n"] > 0]
            if not scope_units:
                continue
        M = boot_multiplicities(len(scope_units), n_boot, name)
        scope_accs = [accs[(u, lab)] for u in scope_units]
        for v in measures:
            levels = descriptives(P_by_seed, K, scope_units, lab, v)
            for adjusted in analyses:
                pooled = pooled_adj if adjusted else pooled_diff
                obs = pooled(scope_accs, np.ones(len(scope_units)), v)
                boot = np.array([pooled(scope_accs, m, v) for m in M])
                if v == "propensity":
                    null = no_null
                else:
                    null = null_values(nulls[lab], scope_units, v, adjusted, n_draws)
                rows.append(dict(analysis="terrain_adjusted" if adjusted else "raw", scope_kind="pooled",
                                 scope=name, seed=scope_seed(name), component=-1,
                                 type=name if name in TYPES else "", label_subset=str(lab), **{label_col: v},
                                 n_regions=len(scope_units), **levels, **contrast_summary(obs, boot, null)))
    for u in units:
        for v in measures:
            levels = descriptives(P_by_seed, K, [u], "all", v)
            for adjusted in analyses:
                pooled = pooled_adj if adjusted else pooled_diff
                obs = pooled([accs[(u, "all")]], [1.0], v)
                if v == "propensity":
                    null = no_null
                else:
                    null = null_values(nulls["all"], [u], v, adjusted, n_draws)
                rows.append(dict(analysis="terrain_adjusted" if adjusted else "raw", scope_kind="region",
                                 scope="region", seed=u[0], component=u[1], type=region_type[u],
                                 label_subset="all", **{label_col: v}, n_regions=1, **levels,
                                 **contrast_summary(obs, None, null)))
    out = pd.DataFrame(rows)
    out["primary"] = False
    if primary_measure is not None:
        out["primary"] = ((out.analysis == "raw") & (out.scope_kind == "pooled") & (out.scope == "all")
                          & (out[label_col] == primary_measure))
    # written on every pooled row with a null; empty where the rule is undefined
    met = ((out.ci_lo > 0) & (out.null_percentile > 95)).astype(object)
    met[(out.scope_kind == "region") | out.null_percentile.isna()] = ""
    out["met"] = met
    return out


# ----------------------------------------------------------------------------- C3
def c3_model_sets(models: pd.DataFrame) -> dict:
    """The paper's 60 test models (op.model_split) and the strongest quarter of them: the TOP_Q with
    the highest auc_id, ties by index."""
    _, _, test = op.model_split(len(models))
    test = np.sort(test)
    by_auc = sorted(test, key=lambda i: (-models.auc_id.iloc[i], i))
    return {"top15": np.sort(np.array(by_auc[:TOP_Q])), "test60": test}


def c3_parts(acc_ae: float, st: dict) -> dict:
    """The decomposition of one target: covariate = acc_ae - acc_ae_w, concept = acc_ae_w - acc_region,
    and the same with the _mu and _mix sensitivities in place of acc_ae_w."""
    return dict(covariate=acc_ae - st["acc_ae_w"], concept=st["acc_ae_w"] - st["acc_region"],
                total=acc_ae - st["acc_region"], acc_ae_w=st["acc_ae_w"], acc_region=st["acc_region"],
                covariate_mu=acc_ae - st["acc_ae_mu"], concept_mu=st["acc_ae_mu"] - st["acc_region"],
                covariate_mix=acc_ae - st["acc_ae_w_mix"], concept_mix=st["acc_ae_w_mix"] - st["acc_region"])


def share_of_total(covariate: float, total: float) -> float:
    return covariate / total if abs(total) > 1e-9 else np.nan


def c3_seed(s: int, c3: dict, P: pd.DataFrame, K: pd.DataFrame, rl: pd.DataFrame, models: pd.DataFrame,
            n_boot: int, log=print) -> tuple[list[dict], dict]:
    """C3 for one seed: a row per (target, model set) and the bootstrap draws of the pooled and
    by-type targets. Targets: all regions, each region type, each region.

    acc_ae is the models' mean accuracy (op.correctness) on land Zone AE test points, acc_region on
    the target's region land points (all of them, matched or not, point-weighted), and acc_ae_w the
    Zone AE accuracy reweighted to the target's feature distribution by c3_weights. The odds come
    from one cross-fitted propensity "region vs Zone AE test land", clipped once; targets and
    bootstrap draws (regions resampled, nothing refitted) only change the multiplicities. Re-clipping
    each draw pulled the bootstrap below its point estimate on the real data.
    Sensitivities: the 60 test models; _mu, an outcome model of Zone AE accuracy averaged over the
    region points; _mix, per-region reweighted accuracies mixed by point counts (the pooled weights
    give each region its ae_weight_mass, not its region_point_share). Diagnostics: ESS, the largest
    |mean difference| of the standardized columns before / after weighting, label shares.
    """
    cell, Xod, Xid, od, idt = c3["cell"], c3["Xod"], c3["Xid"], c3["od"], c3["idt"]
    model_sets = c3_model_sets(models)
    Zo, _ = op.correctness(cell.S_ood, cell.y_ood)
    Zi, ties = op.correctness(cell.S_id, cell.y_id)

    land_ae = idt.landcover.to_numpy() != WATER
    land_a = od.landcover.to_numpy() != WATER
    pid_o = od.point_id.astype(str).to_numpy()
    comp_o = P.loc[pid_o, "component"].to_numpy()
    reg_mask = (comp_o >= 0) & land_a
    rl_s = rl[(rl.seed == s) & (rl.n_land > 0)].sort_values("component")
    comps = rl_s.component.to_numpy()
    reg_unit = comp_o[reg_mask]
    if set(reg_unit) != set(comps):
        raise SystemExit(f"seed {s}: region land points do not match regions_land.csv")
    counts = pd.Series(reg_unit).value_counts().reindex(comps).to_numpy()
    if not np.array_equal(counts, rl_s.n_land.to_numpy()):
        raise SystemExit(f"seed {s}: region land counts differ from regions_land.csv")

    Xae, Xreg = Xid[land_ae], Xod[reg_mask]
    y_ae = idt.label.to_numpy()[land_ae].astype(float)
    y_reg = od.label.to_numpy()[reg_mask].astype(float)
    p_ae, p_reg, auc, _ = crossfit_propensity(Xae, Xreg, seed=RNG_SEED + 100 + s)
    p_ae = np.clip(p_ae, 1e-12, 1 - 1e-12)
    oc, cap = clip_odds(p_ae / (1 - p_ae))
    G, units = composition(Xreg, reg_unit, Xae, seed=RNG_SEED + 200 + s)
    if not np.array_equal(units, comps):
        raise SystemExit(f"seed {s}: composition classes differ from the regions")
    if np.abs(G.sum(1) - 1).max() > 1e-9:
        raise SystemExit(f"seed {s}: composition rows do not sum to 1")

    # per-region sums over the region points (onehot: region point x region)
    onehot = (reg_unit[:, None] == comps[None, :]).astype(float)
    xsum_unit = onehot.T @ Xreg
    pos_unit = onehot.T @ y_reg
    preg_unit = onehot.T @ p_reg
    odds_by_unit = oc[:, None] * G
    mass_unit = odds_by_unit.sum(0) / oc.sum()                     # each region's share of the AE weight
    w_unit = odds_by_unit / odds_by_unit.sum(0)[None, :]           # AE weights for each region alone
    matched = K[(K.seed == s) & (K.status == "matched")]
    pos_o = pd.Series(np.arange(len(pid_o)), index=pid_o)
    log(f"  C3 seed {s}: {int(land_ae.sum())} land Zone AE test points vs {int(reg_mask.sum())} region land points "
        f"in {len(comps)} regions; propensity AUC {auc:.4f}; mean p on AE {p_ae.mean():.4f}, on regions "
        f"{p_reg.mean():.4f} (share > 0.9: {(p_reg > 0.9).mean():.3f}); unresolved ID ties {ties}; "
        f"top-15 models {model_sets['top15'].tolist()}")

    targets = [("all", comps)]
    for t in TYPES:
        of_type = rl_s[rl_s.type == t].component.to_numpy()
        if len(of_type):
            targets.append((t, of_type))
    for c in comps:
        targets.append((f"region_c{c}", np.array([c])))

    rows, draws = [], {}
    for set_name, idx in model_sets.items():
        acc_ae_pt = Zi[idx][:, land_ae].mean(0).astype(float)
        acc_o_pt = Zo[idx].mean(0).astype(float)
        acc_reg_unit = onehot.T @ acc_o_pt[reg_mask]
        mu = HistGradientBoostingRegressor(random_state=RNG_SEED + 300 + s, **HGB_PARAMS).fit(Xae, acc_ae_pt)
        mu_unit = onehot.T @ np.clip(mu.predict(Xreg), 0.0, 1.0)
        acc_ae = float(acc_ae_pt.mean())
        accw_unit = w_unit.T @ acc_ae_pt

        def target_stats(m):
            """Every C3 number of the region mixture with multiplicities m."""
            w, ess = c3_weights(oc, G, m)
            nm = float(np.sum(m * counts))
            region_mean_x = m @ xsum_unit / nm
            return dict(acc_ae_w=float(w @ acc_ae_pt), acc_region=float(m @ acc_reg_unit / nm),
                        acc_ae_mu=float(m @ mu_unit / nm), acc_ae_w_mix=float(np.sum(m * counts * accw_unit) / nm),
                        clip_value=cap, ess=ess,
                        pos_share_ae_w=float(w @ y_ae), pos_share_region=float(m @ pos_unit / nm),
                        mean_p_region=float(m @ preg_unit / nm),
                        max_abs_dmean_unw=float(np.abs(Xae.mean(0) - region_mean_x).max()),
                        max_abs_dmean_w=float(np.abs(w @ Xae - region_mean_x).max()))

        for target, target_comps in targets:
            base = np.isin(comps, target_comps).astype(float)
            st = target_stats(base)
            parts = c3_parts(acc_ae, st)
            intervals = {}
            if target in ("all",) + TYPES:
                drawn = np.flatnonzero(base)
                B = []
                for mb in boot_multiplicities(len(drawn), n_boot, f"C3:{s}:{target}"):
                    m = np.zeros(len(comps))
                    m[drawn] = mb
                    draw = c3_parts(acc_ae, target_stats(m))
                    B.append([draw[k] for k in C3_BOOT])
                B = np.array(B)
                draws[(set_name, target)] = B
                intervals = c3_intervals(B)
            control_ids = matched[matched.component.isin(target_comps)].control_point_id.astype(str)
            if len(control_ids):
                acc_controls = float(acc_o_pt[pos_o.loc[control_ids].to_numpy()].mean())
            else:
                acc_controls = np.nan
            rows.append(dict(seed=str(s), target=target, model_set=set_name, n_models=len(idx),
                             n_regions=int(base.sum()), n_region_land=int(np.sum(base * counts)),
                             n_ae_test_land=int(land_ae.sum()), n_controls=len(control_ids),
                             prop_auc=auc, mean_p_region=st["mean_p_region"], clip_value=st["clip_value"],
                             ess=st["ess"], max_abs_dmean_unw=st["max_abs_dmean_unw"],
                             max_abs_dmean_w=st["max_abs_dmean_w"],
                             pos_share_region=st["pos_share_region"], pos_share_ae=float(y_ae.mean()),
                             pos_share_ae_w=st["pos_share_ae_w"],
                             acc_ae=acc_ae, acc_ae_w=st["acc_ae_w"], acc_region=st["acc_region"],
                             acc_controls=acc_controls, total=parts["total"], covariate=parts["covariate"],
                             concept=parts["concept"],
                             covariate_share=share_of_total(parts["covariate"], parts["total"]),
                             acc_ae_mu=st["acc_ae_mu"], covariate_mu=parts["covariate_mu"],
                             concept_mu=parts["concept_mu"], acc_ae_w_mix=st["acc_ae_w_mix"],
                             covariate_mix=parts["covariate_mix"], concept_mix=parts["concept_mix"],
                             ae_weight_mass=float(base @ mass_unit),
                             region_point_share=float(base @ counts / counts.sum()), **intervals))
    return rows, draws


def c3_intervals(B: np.ndarray) -> dict:
    """Percentile 95% intervals of the C3_BOOT columns of the draws B and of the covariate share."""
    ci = {}
    for j, name in enumerate(C3_BOOT):
        lo, hi = np.percentile(B[:, j], [2.5, 97.5])
        ci[f"{name}_ci_lo"], ci[f"{name}_ci_hi"] = float(lo), float(hi)
    total = B[:, C3_BOOT.index("total")]
    covariate = B[:, C3_BOOT.index("covariate")]
    safe_total = np.where(total == 0, 1, total)         # avoids a divide warning; masked just after
    share = np.where(np.abs(total) > 1e-9, covariate / safe_total, np.nan)
    lo, hi = np.nanpercentile(share, [2.5, 97.5]) if np.isfinite(share).any() else (np.nan, np.nan)
    ci["covariate_share_ci_lo"], ci["covariate_share_ci_hi"] = float(lo), float(hi)
    return ci


def c3_pooled(rows: list[dict], draws_by_seed: dict, seeds) -> list[dict]:
    """Pooled C3 rows: the equal-weight mean over seeds of each seed's numbers, bootstrapped draw by
    draw (draw b of every seed). Needs at least two seeds."""
    R = pd.DataFrame(rows)
    out = []
    pooled_targets = R[R.target.isin(("all",) + TYPES)]
    for (target, set_name), g in pooled_targets.groupby(["target", "model_set"], sort=False):
        with_draws = [s for s in seeds if (set_name, target) in draws_by_seed.get(s, {})]
        if len(with_draws) < 2:
            continue
        g = g.set_index("seed").loc[[str(s) for s in with_draws]]
        B = np.mean([draws_by_seed[s][(set_name, target)] for s in with_draws], axis=0)
        d = dict(seed="pooled", target=target, model_set=set_name, n_models=int(g.n_models.iloc[0]),
                 n_regions=int(g.n_regions.sum()), n_region_land=int(g.n_region_land.sum()),
                 n_ae_test_land=int(g.n_ae_test_land.sum()), n_controls=int(g.n_controls.sum()),
                 clip_value=np.nan, **{c: float(g[c].mean()) for c in C3_MEAN_COLS})
        total = d["acc_ae"] - d["acc_region"]
        d.update(total=total, covariate=d["acc_ae"] - d["acc_ae_w"], concept=d["acc_ae_w"] - d["acc_region"],
                 covariate_share=share_of_total(d["acc_ae"] - d["acc_ae_w"], total),
                 covariate_mu=d["acc_ae"] - d["acc_ae_mu"], concept_mu=d["acc_ae_mu"] - d["acc_region"],
                 covariate_mix=d["acc_ae"] - d["acc_ae_w_mix"], concept_mix=d["acc_ae_w_mix"] - d["acc_region"],
                 seeds_pooled=";".join(map(str, with_draws)), **c3_intervals(B))
        out.append(d)
    return out


# ----------------------------------------------------------------------------- report
def print_report(C1: pd.DataFrame, C2: pd.DataFrame, C3: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    show = ["analysis", "scope", "seed", "component", "type", "measure", "n_regions", "n_pairs",
            "region_all_mean", "region_matched_mean", "controls_mean", "ae_test_mean", "ae_test_tile_mean",
            "diff", "ci_lo", "ci_hi", "null_mean", "null_percentile", "met"]
    print("\nC2 label conflict, pooled scopes (diff = region - controls over matched pairs; "
          "terrain_adjusted = OLS coefficient on the region indicator)")
    print(C2[C2.scope_kind == "pooled"][show].round(4).to_string(index=False))
    print("\nC2 per region, k = 25")
    print(C2[(C2.scope_kind == "region") & (C2.measure == "conflict_k25")][show].round(4).to_string(index=False))
    print("\nC1 support, pooled scopes")
    print(C1[C1.scope_kind == "pooled"][show].round(4).to_string(index=False))
    print("\nC1 support per region")
    print(C1[C1.scope_kind == "region"][show].round(4).to_string(index=False))
    split_cols = ["seed", "target", "model_set", "n_regions", "n_region_land", "acc_ae", "acc_ae_w",
                  "acc_region", "acc_controls", "total", "covariate", "covariate_ci_lo", "covariate_ci_hi",
                  "concept", "concept_ci_lo", "concept_ci_hi", "covariate_share", "covariate_mu", "concept_mu",
                  "covariate_mix", "concept_mix"]
    print("\nC3 decomposition (accuracy; covariate = acc_ae - acc_ae_w, concept = acc_ae_w - acc_region; "
          "_mu = outcome-model sensitivity)")
    print(C3[split_cols].round(4).to_string(index=False))
    diagnostic_cols = ["seed", "target", "model_set", "prop_auc", "mean_p_region", "ess", "clip_value",
                       "max_abs_dmean_unw", "max_abs_dmean_w", "ae_weight_mass", "region_point_share",
                       "pos_share_region", "pos_share_ae", "pos_share_ae_w"]
    print("\nC3 diagnostics (propensity separability, effective sample size, feature balance after "
          "weighting in training-SD units, label shares)")
    print(C3[C3.model_set == "top15"][diagnostic_cols].round(4).to_string(index=False))

    p = C2[C2.primary].iloc[0]
    adj = C2[(C2.analysis == "terrain_adjusted") & (C2.scope_kind == "pooled") & (C2.scope == "all")
             & (C2.measure == "conflict_k25")].iloc[0]
    print(f"\nPRIMARY (C2 conflict k=25, region - controls, pooled over seeds, {int(p.n_pairs)} pairs in "
          f"{int(p.n_regions)} regions): {p['diff']:+.4f} [{p.ci_lo:+.4f}, {p.ci_hi:+.4f}]; null mean "
          f"{p.null_mean:+.4f}, percentile {p.null_percentile:.1f} ({int(p.n_null)} draws); "
          f"predicted positive: {'MET' if p.met else 'NOT MET'}")
    print(f"terrain-adjusted: {adj['diff']:+.4f} [{adj.ci_lo:+.4f}, {adj.ci_hi:+.4f}]; null percentile "
          f"{adj.null_percentile:.1f}; {'interval excludes 0 and > 95' if adj.met else 'not met by the same rule'}")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/processed_w7"),
                    help="folder with FEATURES.txt and dataset_FL.parquet (zoo.load)")
    ap.add_argument("--cells-dir", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--cell-prefix", default="FL_band_AE")
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--o5-dir", type=Path, default=Path("results/2026-10-17/o5"),
                    help="where regions_land.csv, S1_controls.csv and S2_null.parquet are")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--points-out", type=Path, default=None, help="optional per-point table (parquet)")
    args = ap.parse_args()
    threadpool_limits(args.threads)
    t0 = time.time()

    rl = pd.read_csv(args.o5_dir / "regions_land.csv")
    rl = rl[rl.seed.isin(args.seeds)]
    K = pd.read_csv(args.o5_dir / "S1_controls.csv", dtype={"point_id": str, "control_point_id": str})
    K = K[K.seed.isin(args.seeds)]
    N = pd.read_parquet(args.o5_dir / "S2_null.parquet")
    N = N[N.seed.isin(args.seeds)]
    calipers = sorted(set(N.caliper.astype(str)))
    n_draws = int(N.draw.max()) + 1
    print(f"S1 controls: {int((K.status == 'matched').sum())} matched of {len(K)} region land points; "
          f"S2: {len(N)} rows, {n_draws} draws per region, caliper {calipers}")
    print("land rule (landcover != 11), points dropped per region:")
    for _, r in rl.iterrows():
        print(f"  seed {r.seed} c{r.component} ({r.type}): {r.n} points, {r.n_land} land, "
              f"{r.dropped_water} dropped")

    df, feats = zoo.load("FL", args.data)
    print(f"features ({len(feats)}): {', '.join(feats)}; space adds miss_hand, miss_wetindexcat; "
          f"landcover kept numeric (NLCD class code), as the models see it")
    ctxs = mc.load_contexts(args.seeds, args.cells_dir, args.cell_prefix, CONTEXT_TABLE, args.sel_root,
                            args.regions_dir, BUDGET, MIN_REGION_SIZE, verbose=False)
    P_by_seed, infos, c3_inputs, models_by_seed = {}, [], {}, {}
    for s in args.seeds:
        P, info, c3 = seed_points(s, df, feats, ctxs[s], args.cells_dir, args.cell_prefix, K, args.threads)
        P_by_seed[s], c3_inputs[s] = P, c3
        infos.append(info)
        models_by_seed[s] = pd.read_csv(args.cells_dir / f"{args.cell_prefix}_s{s}_models.csv")
    print(f"feature space, kNN and C1 propensities done ({time.time() - t0:.0f} s)")

    units, accs = unit_tables(P_by_seed, K, rl)
    nulls = null_tables(P_by_seed, N, units)
    print(f"accumulators: {len(units)} regions with land points; null patches "
          f"{len(nulls['all'])} ({time.time() - t0:.0f} s)")

    C2 = contrast_rows(P_by_seed, K, rl, units, accs, nulls, args.seeds, CONFLICT, args.n_boot, n_draws,
                       with_adjusted=True, primary_measure=f"conflict_k{K_PRIMARY}", label_col="measure")
    C1 = contrast_rows(P_by_seed, K, rl, units, accs, nulls, args.seeds, ("knn_dist_k25", "propensity"),
                       args.n_boot, n_draws, with_adjusted=False, primary_measure=None, label_col="measure")
    seed_info = pd.DataFrame(infos).set_index("seed")
    C1["prop_auc_train_vs_eval"] = C1.seed.map(seed_info.prop_auc_train_vs_eval)
    print(f"C1, C2 done ({time.time() - t0:.0f} s)")

    rows3, draws3 = [], {}
    for s in args.seeds:
        rows, draws = c3_seed(s, c3_inputs[s], P_by_seed[s], K, rl, models_by_seed[s], args.n_boot)
        rows3 += rows
        draws3[s] = draws
    rows3 += c3_pooled(rows3, draws3, args.seeds)
    C3 = pd.DataFrame(rows3)
    print(f"C3 done ({time.time() - t0:.0f} s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C1.to_csv(args.out_dir / "C_support.csv", index=False, float_format=CSV_FLOAT)
    C2.to_csv(args.out_dir / "C_conflict.csv", index=False, float_format=CSV_FLOAT)
    C3.to_csv(args.out_dir / "C_decomposition.csv", index=False, float_format=CSV_FLOAT)
    if args.points_out is not None:
        args.points_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(P_by_seed.values(), ignore_index=True).to_parquet(args.points_out, index=False)
        print(f"wrote {args.points_out}")

    print_report(C1, C2, C3)
    print(f"wrote {args.out_dir}/C_support.csv ({len(C1)} rows), C_conflict.csv ({len(C2)} rows), "
          f"C_decomposition.csv ({len(C3)} rows); {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
