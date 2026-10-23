#!/usr/bin/env python3
"""O5 Part B: how do the regions differ from their terrain twins, beyond terrain?

S1 (match_controls.py) gave every region land point a twin: an unselected Zone A test point of
the same seed, 50 km tile and label with the nearest terrain. S2 (region_null.py) grew 200
matched-growth null patches per region, each with its own twins. For about 40 non-terrain
descriptors and the behaviour of the strongest test models, this script compares region points
with their twins (SMD, class-share differences, total variation distance, terrain-adjusted SMD),
per region and pooled by region type, in label scopes all, positives and negatives, with a
bootstrap interval and a percentile among the null patches. Design and the fixed primary:
docs/TASK-O5-REGIONS.md Part B and results/2026-10-17/o5/PRIMARIES.md Part B.

Reads S1_controls.csv, S2_null.parquet, regions_land.csv, the point table, point_extras
(meta_edge_dist_m), the cells and, through match_controls.load_contexts, the selections and region
point tables. Writes <out-dir>/B_contrast.csv (one row per scope, label scope, variable [, class],
adjusted), B_balance.csv (terrain balance of the pairs used) and B_null.csv (every null draw of
every pooled statistic).

    .venv/bin/python aotil/o5_contrast.py --out-dir results/2026-10-17/o5 > logs/2026-10-17/B_contrast.log 2>&1
"""
import os

# Cap BLAS threads before numpy loads: the statistics are sums taken by matrix products, and the
# thread count can move their last bits.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import match_controls as mc  # noqa: E402
import oodselect_paper as op  # noqa: E402
from cells import load_cell  # noqa: E402

RNG_SEED = 20261018
N_BOOT = 1000                    # bootstrap draws
TOP_K = 15                       # strongest test models: a quarter of the paper's 60
CLASS_MIN_SHARE = 0.05           # a class is reported if it reaches this share somewhere
CLASS_MIN_PAIRS = 20             # pairs a pooled scope needs before it can nominate a class
TIE = 1e-12                      # null values this close to the observed value count as ties
MIN_PAIRS_ADJ = 5                # complete pairs the terrain-adjusted regression needs
TERRAIN = mc.TERRAIN
TYPES = ("positive-dominated", "negative-dominated")
# Twins share their point's label, so every pair lies within one label.
LABEL_SCOPES = (("all", (0, 1)), ("positives", (1,)), ("negatives", (0,)))
PRIMARY = "log10_dist_river"
PREDICTED = {"positive-dominated": +1, "negative-dominated": -1}
PRIMARY_PCT_HI, PRIMARY_PCT_LO = 95, 5          # null percentile the primary must pass
SECONDARY_LO, SECONDARY_HI = 2.5, 97.5           # null percentile that flags a secondary finding

# EPA StreamCat columns, averaged over the point's catchment (cat) or its whole upstream
# watershed (ws); NOAA Atlas 14 design-storm depths per 10 km block.
STREAMCAT = ("bfiws", "bficat", "hydrlcondws", "hydrlcondcat", "wetindexws", "kffactcat", "kffactws",
             "precip8110ws", "precip8110cat", "tmean8110ws", "tmean8110cat", "runoffws", "runoffcat",
             "clayws", "claycat", "sandws", "sandcat", "permws", "permcat", "meta_elevcat", "meta_elevws")
ATLAS14 = ("p100_24hr", "p100_60min", "p100_4day", "p2_24hr", "growth_100_2")
# Variable -> descriptor group (column meanings: docs/DATA-SUMMARY.md sections 2-3).
# meta_eff_date is used as a decimal year; meta_panel_scale (6000 / 12000 / 24000) as a number.
NUMERIC = {
    **dict.fromkeys(("wetindexcat", "log10_dist_river", "log10_dist_coast"), "model_feature"),
    **dict.fromkeys(("nearest_stream_order", "log10_upstream_area", "catch_area_km2", "drainage_density"),
                    "river_network"),
    **dict.fromkeys(STREAMCAT + ATLAS14, "soil_climate_rainfall"),
    **dict.fromkeys(("meta_eff_date", "meta_panel_scale", "meta_bfe_km_per_km2", "meta_poly_area_km2",
                     "meta_edge_dist_m"), "map_provenance"),
}
CATEGORICAL = {"landcover": "model_feature", "meta_nearest_ftype": "river_network",
               "meta_study_typ": "map_provenance", "meta_study_regime": "map_provenance",
               "meta_source_cit": "map_provenance", "dfirm_id": "map_provenance"}
# These come from the point's flood polygon, which only positives have, so they are compared in
# label scope positives only.
POSITIVE_ONLY = {"meta_study_typ", "meta_study_regime", "meta_source_cit", "meta_poly_area_km2",
                 "meta_edge_dist_m"}
DESCRIPTOR_GROUPS = ("model_feature", "river_network", "soil_climate_rainfall", "map_provenance")
MODEL_VARS = ("p_flood_top", "acc_top", "fn_share_top")
ALL_CLASSES = "(all classes)"
NLCD = {21: "developed, open space", 22: "developed, low", 23: "developed, medium",
        24: "developed, high", 31: "barren", 41: "deciduous forest", 42: "evergreen forest",
        43: "mixed forest", 52: "shrub/scrub", 71: "grassland", 81: "pasture/hay",
        82: "cultivated crops", 90: "woody wetlands", 95: "emergent herbaceous wetlands"}
# Columns of the per-pair model statistics M (_r region point, _c twin): 1 per pair, the top
# models' mean p(flood), their mean accuracy, and their error and false-negative counts.
MODEL_STAT_COLS = ("n", "p_r", "p_c", "acc_r", "acc_c", "err_r", "fn_r", "err_c", "fn_c")


# ------------------------------------------------------------------------------------ data
def decimal_year(dates: pd.Series) -> pd.Series:
    """Dates as decimal years; FEMA's 9999-09-09 placeholder (any year after 2100) becomes NaN."""
    parsed = pd.to_datetime(dates)
    year_length = np.where(parsed.dt.is_leap_year, 366.0, 365.0)
    years = parsed.dt.year + (parsed.dt.dayofyear - 1) / year_length
    return years.where(parsed.dt.year <= 2100).astype(float)


def landcover_name(code: float):
    """'<code> <NLCD class name>', or None when the code is missing."""
    if not np.isfinite(code):
        return None
    return f"{int(code)} {NLCD.get(int(code), 'other')}"


def category_or_none(x):
    """x as str; None, pd.NA and NaN are missing (None)."""
    if x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x)):
        return None
    return str(x)


def load_table(data, extras) -> pd.DataFrame:
    """The point table's Part B columns, indexed by point_id.

    Every categorical variable v gets a column cat_v of str values with None for missing
    (landcover as "<code> <NLCD name>")."""
    numeric = [v for v in NUMERIC if v != "meta_edge_dist_m"]       # that one is in extras
    categorical = [v for v in CATEGORICAL if v != "landcover"]
    columns = ["point_id", "domain", "label", "landcover", "tile_50km", *numeric, *categorical]
    table = pd.read_parquet(data, columns=columns)
    edge = pd.read_parquet(extras, columns=["point_id", "meta_edge_dist_m"])
    table = table.merge(edge, on="point_id", how="left", validate="one_to_one")
    table["meta_eff_date"] = decimal_year(table["meta_eff_date"])
    table["landcover_class"] = [landcover_name(code) for code in table["landcover"].to_numpy(float)]
    for v in CATEGORICAL:
        source = "landcover_class" if v == "landcover" else v
        values = [category_or_none(x) for x in table[source].astype(object)]
        table[f"cat_{v}"] = pd.Series(values, index=table.index, dtype=object)
    for v in NUMERIC:
        table[v] = table[v].astype(float)
    return table.set_index("point_id")


def model_scores(cells_dir, tag: str, top_k: int = TOP_K) -> dict:
    """The top_k test models by auc_id (Zone AE test AUC) among the paper's 60, and per OOD point
    (cell column order) their mean p(flood), error count and false-negative count."""
    cell = load_cell(cells_dir, tag)
    models = cell.models
    if not (models["idx"].to_numpy() == np.arange(len(models))).all():
        raise SystemExit(f"{tag}: models.csv idx is not 0..N-1 in row order")
    _, _, test_models = op.model_split(len(models))
    auc = models["auc_id"].to_numpy(float)
    top = test_models[np.argsort(-auc[test_models], kind="stable")[:top_k]]
    S = np.asarray(cell.S_ood, float)[top]
    y = np.asarray(cell.y_ood)
    # S is the probability of the true label. p(flood) is rounded to 9 decimals; it is only
    # averaged (p_flood_top), never ranked, so the rounding moves it by at most 5e-10.
    p_flood = np.round(np.where(y[None] == 1, S, 1 - S), 9)
    correct, _ = op.correctness(S, y)
    err = (1 - correct).sum(axis=0).astype(float)
    fn = np.where(y == 1, err, 0.0)
    return dict(top=np.sort(top), auc_top=auc[top], kinds=models["kind"].to_numpy()[top],
                p=p_flood.mean(axis=0), err=err, fn=fn, k=len(top),
                point_id=np.asarray(cell.point_id).astype(str))


@dataclass
class SeedData:
    """One seed's Zone A test points in cell column order, as the arrays Part B needs."""
    seed: int
    pid: np.ndarray
    label: np.ndarray
    tile: np.ndarray
    land: np.ndarray
    selected: np.ndarray
    component: np.ndarray        # region id, -1 outside every region
    num: np.ndarray              # (points, V) numeric variables minus Layout.center
    cat: np.ndarray              # (points, categorical vars) codes into Layout.cat_levels, -1 missing
    z: np.ndarray                # (points, 6) terrain in seed-SD units, as match_controls uses
    raw_terrain: np.ndarray      # (points, 6) terrain in its own units
    model: np.ndarray            # (points, 3) top models' mean p(flood), error count, FN count
    scores: dict                 # model_scores' output
    _pos: dict = field(default_factory=dict, repr=False)

    def pos(self, ids) -> np.ndarray:
        """Row positions of the given point ids (KeyError for an unknown id)."""
        if not self._pos:
            for i, p in enumerate(self.pid):
                self._pos[p] = i
        return np.array([self._pos[p] for p in ids], int)


@dataclass
class Layout:
    """The variables and reported classes, and where each categorical level sits on the one
    level axis that concatenates every categorical variable's levels."""
    num: list                    # numeric variable names
    center: np.ndarray           # subtracted before summing, for precision (the SMD is shift-invariant)
    scale: np.ndarray            # SD over land Zone A points; sets the zero tolerances
    cat_vars: list
    cat_levels: dict             # variable -> its levels; codes index this list
    classes: list = field(default_factory=list)   # reported classes: (var, class, var_index, code)

    @property
    def V(self):
        return len(self.num)

    @property
    def J(self):
        return len(self.classes)

    @property
    def offsets(self) -> np.ndarray:
        """Start of each categorical variable's levels on the level axis."""
        sizes = [len(self.cat_levels[v]) for v in self.cat_vars]
        return np.r_[0, np.cumsum(sizes)][:-1]

    @property
    def n_levels(self) -> int:
        return int(sum(len(self.cat_levels[v]) for v in self.cat_vars))

    @property
    def level_var(self) -> np.ndarray:
        """Categorical variable index of every level on the level axis."""
        return np.concatenate([np.full(len(self.cat_levels[v]), j) for j, v in enumerate(self.cat_vars)])

    @property
    def class_cols(self) -> np.ndarray:
        """Position of each reported class on the level axis."""
        offsets = self.offsets
        return np.array([offsets[var_index] + code for _, _, var_index, code in self.classes], int)


def make_layout(table: pd.DataFrame) -> Layout:
    """Variables, centres and scales (over land Zone A points) and every categorical level.
    The reported classes are filled in later by frequent_classes."""
    land_a = table[(table.domain == "A") & (table.landcover != mc.WATER)]
    center = np.array([np.nanmedian(land_a[v].to_numpy(float)) for v in NUMERIC])
    scale = np.array([np.nanstd(land_a[v].to_numpy(float)) for v in NUMERIC])
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    levels = {}
    for v in CATEGORICAL:
        present = {x for x in table[f"cat_{v}"] if x is not None}
        levels[v] = sorted(present)
    return Layout(list(NUMERIC), center, scale, list(CATEGORICAL), levels)


def seed_data(ctx: mc.SeedContext, table: pd.DataFrame, lay: Layout, scores: dict) -> SeedData:
    """One seed's arrays, after checking the cell's point order and the table's labels."""
    pts = ctx.pts
    pid = pts.point_id.to_numpy().astype(str)
    if not (scores["point_id"] == pid).all():
        raise SystemExit(f"{ctx.tag}: cell point_id order differs from the context's")
    rows = table.loc[pid]
    if not (rows.label.to_numpy() == pts.label.to_numpy()).all():
        raise SystemExit(f"{ctx.tag}: table labels differ from the context's")
    cat = np.full((len(pid), len(lay.cat_vars)), -1, int)
    for j, v in enumerate(lay.cat_vars):
        code_of = {level: i for i, level in enumerate(lay.cat_levels[v])}
        cat[:, j] = [code_of[x] if x is not None else -1 for x in rows[f"cat_{v}"]]
    return SeedData(seed=ctx.seed, pid=pid, label=pts.label.to_numpy().astype(int),
                    tile=pts.tile_50km.to_numpy().astype(str), land=pts.land.to_numpy(),
                    selected=pts.selected.to_numpy(), component=pts.component.to_numpy(),
                    num=rows[lay.num].to_numpy(float) - lay.center, cat=cat,
                    z=pts[[f"z_{v}" for v in TERRAIN]].to_numpy(float),
                    raw_terrain=pts[list(TERRAIN)].to_numpy(float),
                    model=np.c_[scores["p"], scores["err"], scores["fn"]], scores=scores)


# ------------------------------------------------------------------- sufficient statistics
def pair_stats(D: SeedData, ir: np.ndarray, ic: np.ndarray, lay: Layout) -> dict:
    """Per-pair sufficient statistics of the pairs (region row ir[p], twin row ic[p]).

    Every statistic of a set of pairs is a function of sums of these over the pairs, so pooled,
    bootstrapped, null and reweighted statistics all come from weighted sums (aggregate).
    One row per pair:
      npair   1
      C       (V) 1 if both values of the numeric variable are present (a complete pair)
      R1, R2  (V) region value and its square, 0 unless complete; K1, K2 the same for the twin
      AO      (categorical vars) 1 if both values are present
      AR, AC  (levels) region / twin one-hot over every level, 0 unless both present
      OUT     (64) X'X of the pair's two regression rows [1, indicator, z], flattened 8 x 8
      XY      (V, 8) X'y of the two rows
      M       (9) model behaviour, columns MODEL_STAT_COLS
    """
    n_pairs = len(ir)
    xr, xc = D.num[ir], D.num[ic]
    C = np.isfinite(xr) & np.isfinite(xc)
    xr0 = np.where(C, xr, 0.0)
    xc0 = np.where(C, xc, 0.0)
    st = dict(npair=np.ones((n_pairs, 1)), C=C.astype(float), R1=xr0, R2=xr0 ** 2, K1=xc0, K2=xc0 ** 2)

    cr, cc = D.cat[ir], D.cat[ic]
    both = (cr >= 0) & (cc >= 0)
    AR = np.zeros((n_pairs, lay.n_levels))
    AC = np.zeros((n_pairs, lay.n_levels))
    offsets = lay.offsets
    for j in range(len(lay.cat_vars)):
        rows = np.flatnonzero(both[:, j])
        AR[rows, offsets[j] + cr[rows, j]] = 1.0
        AC[rows, offsets[j] + cc[rows, j]] = 1.0
    st.update(AO=both.astype(float), AR=AR, AC=AC)

    # Regression rows: the region point [1, 1, z] and its twin [1, 0, z].
    one = np.ones((n_pairs, 1))
    x1 = np.hstack([one, one, D.z[ir]])
    x0 = np.hstack([one, 0 * one, D.z[ic]])
    if not (np.isfinite(x1).all() and np.isfinite(x0).all()):
        raise ValueError("matched pairs must have complete terrain")
    st["OUT"] = (x1[:, :, None] * x1[:, None, :] + x0[:, :, None] * x0[:, None, :]).reshape(n_pairs, 64)
    st["XY"] = x1[:, None, :] * xr0[:, :, None] + x0[:, None, :] * xc0[:, :, None]

    k = float(D.scores["k"])
    region, twin = D.model[ir], D.model[ic]
    st["M"] = np.c_[np.ones(n_pairs), region[:, 0], twin[:, 0], 1 - region[:, 1] / k, 1 - twin[:, 1] / k,
                    region[:, 1], region[:, 2], twin[:, 1], twin[:, 2]]
    return st


def aggregate(st: dict, W) -> dict:
    """Weighted sums over pairs: W is (B, pairs), dense or scipy sparse; each result is (B, ...).

    OUT becomes XTX (B, V, 8, 8): each numeric variable's regression uses its complete pairs only."""
    n_rows = W.shape[0]
    U = {}
    for key, a in st.items():
        if key == "OUT":
            continue
        flat = a.reshape(a.shape[0], -1)
        U[key] = np.asarray(W @ flat).reshape((n_rows,) + a.shape[1:])
    C, OUT = st["C"], st["OUT"]
    n_vars = C.shape[1]
    per_var = []
    for v in range(n_vars):
        per_var.append(np.asarray(W @ (C[:, [v]] * OUT)))
    U["XTX"] = np.stack(per_var, axis=1).reshape(n_rows, n_vars, 8, 8)
    return U


def group_sums(st: dict, groups: np.ndarray, n_groups: int) -> dict:
    """Sums of the pair statistics within each group; groups[p] is pair p's group."""
    n_pairs = len(groups)
    W = sp.csr_matrix((np.ones(n_pairs), (groups, np.arange(n_pairs))), shape=(n_groups, n_pairs))
    return aggregate(st, W)


def wsum(U: dict, W: np.ndarray) -> dict:
    """{key: W @ U[key]} along U's leading axis (W is (B, n))."""
    return {k: np.tensordot(W, a, axes=(1, 0)) for k, a in U.items()}


def take(U: dict, idx) -> dict:
    """U's rows idx along the leading axis."""
    return {k: a[idx] for k, a in U.items()}


def usum(U: dict, axis) -> dict:
    """U summed along axis."""
    return {k: a.sum(axis=axis) for k, a in U.items()}


def take_l(U: dict, L) -> dict:
    """The label slices L of null sums: (draws, 2, ...) -> (draws, len(L), ...)."""
    return {k: a[:, L] for k, a in U.items()}


def weighted_l(U: dict, w: np.ndarray) -> dict:
    """Null sums (draws, 2, ...) summed over labels with per-(draw, label) weights w."""
    return {k: np.einsum("dl,dl...->d...", w, a) for k, a in U.items()}


def observed_sums(obs_r: dict, labels: list) -> dict:
    """A region's observed sums (2 labels, ...) summed over the given labels."""
    return usum(take(obs_r, labels), 0)


def null_sums(null_r: dict, labels: list) -> dict:
    """A region's null sums (draws, 2 labels, ...) summed over the given labels, per draw."""
    return usum(take_l(null_r, labels), 1)


# ------------------------------------------------------------------------------- statistics
def standardize(diff, den, zero_den, tol):
    """diff / den; where both sides have zero spread, 0 if diff is zero too, else NaN."""
    zero_spread_value = np.where(np.abs(diff) <= tol, 0.0, np.nan)
    return np.where(zero_den, zero_spread_value, diff / den)


def indicator_coef(XTX, XY):
    """The region indicator's OLS coefficient from X'X and X'y (SVD pseudo-inverse), and whether
    the design has full rank."""
    u, s, vt = np.linalg.svd(XTX)
    full_rank = s[..., -1] > 1e-10 * s[..., 0]
    s_inv = np.where(s > 1e-12 * s[..., :1], 1 / s, 0.0)
    t = np.einsum("...kj,...k->...j", u, XY) * s_inv
    beta = np.einsum("...ji,...j->...i", vt, t)
    return beta[..., 1], full_rank


def numeric_stats(U: dict, lay: Layout) -> dict:
    """SMD and terrain-adjusted SMD of every numeric variable over its complete pairs.

    SMD = (mean region - mean twin) / sqrt((var region + var twin) / 2), sample variances, from
    >= 2 complete pairs. The adjusted difference is the indicator's coefficient in the OLS of the
    variable on [1, region indicator, six terrain z] over both rows of each complete pair, over
    the same denominator; it needs MIN_PAIRS_ADJ pairs and a full-rank design. There are no region
    fixed effects: each region has one twin per point, so the indicator is balanced within it."""
    C = U["C"]
    tol = 1e-9 * lay.scale
    mean_r, mean_c = U["R1"] / C, U["K1"] / C
    var_r = np.maximum(U["R2"] - U["R1"] ** 2 / C, 0) / (C - 1)
    var_c = np.maximum(U["K2"] - U["K1"] ** 2 / C, 0) / (C - 1)
    den = np.sqrt((var_r + var_c) / 2)
    diff = mean_r - mean_c
    zero_den = den <= tol
    smd = standardize(diff, den, zero_den, tol)
    smd = np.where(C >= 2, smd, np.nan)
    coef, full_rank = indicator_coef(U["XTX"], U["XY"])
    usable = full_rank & (C >= MIN_PAIRS_ADJ)
    adj = np.where(usable, standardize(coef, den, zero_den, tol), np.nan)
    coef = np.where(usable, coef, np.nan)
    return dict(smd=smd, diff=np.where(C >= 1, diff, np.nan), mean_r=mean_r + lay.center,
                mean_c=mean_c + lay.center, den=np.where(C >= 2, den, np.nan), n=C, adj=adj, adj_diff=coef)


def categorical_stats(U: dict, lay: Layout) -> dict:
    """Class shares among region points and among twins, over the pairs with both values present,
    and per variable the total variation distance: 0.5 * sum over all levels of |share difference|.

    A share row of a location label (dfirm_id, meta_source_cit) partly says where the region is;
    the TVD compares like with like, because each null patch is also compared with its own twins."""
    AO = U["AO"]
    level_var = lay.level_var
    share_r_all = U["AR"] / AO[..., level_var]
    share_c_all = U["AC"] / AO[..., level_var]
    abs_diff = np.abs(share_r_all - share_c_all)
    tvd_per_var = []
    for j in range(len(lay.cat_vars)):
        tvd_per_var.append(0.5 * abs_diff[..., level_var == j].sum(-1))
    tvd = np.where(AO > 0, np.stack(tvd_per_var, -1), np.nan)
    cols = lay.class_cols
    share_r, share_c = share_r_all[..., cols], share_c_all[..., cols]
    return dict(share=share_r - share_c, share_r=share_r, share_c=share_c, n_cat=AO[..., level_var[cols]],
                tvd=tvd, n_tvd=AO)


def model_stats(U: dict) -> dict:
    """Top models' mean p(flood), mean accuracy and share of errors that are false negatives, for
    region points and twins (a plain difference, not an SMD)."""
    M = {name: U["M"][..., i] for i, name in enumerate(MODEL_STAT_COLS)}
    region = np.stack([M["p_r"] / M["n"], M["acc_r"] / M["n"], M["fn_r"] / M["err_r"]], -1)
    twin = np.stack([M["p_c"] / M["n"], M["acc_c"] / M["n"], M["fn_c"] / M["err_c"]], -1)
    return dict(model=region - twin, model_r=region, model_c=twin, n_model=M["n"])


def finalize(U: dict, lay: Layout) -> dict:
    """Every statistic from summed sufficient statistics (U may have any leading shape)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numeric_stats(U, lay)
        out.update(categorical_stats(U, lay))
        out.update(model_stats(U))
    out["npair"] = U["npair"][..., 0]
    return out


def flat_stats(F: dict, lay: Layout) -> dict:
    """{(kind, index): array over the leading shape} for every reported statistic."""
    out = {}
    for i in range(lay.V):
        out[("num", i)] = F["smd"][..., i]
        out[("adj", i)] = F["adj"][..., i]
    for j in range(len(lay.cat_vars)):
        out[("tvd", j)] = F["tvd"][..., j]
    for j in range(lay.J):
        out[("cat", j)] = F["share"][..., j]
    for m in range(len(MODEL_VARS)):
        out[("model", m)] = F["model"][..., m]
    return out


# -------------------------------------------------------------------------------- inference
def midrank_pct(obs: float, null: np.ndarray, tie: float = TIE) -> tuple[float, int]:
    """Mid-rank percentile of obs among the finite null values (ties count half), and their number."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or not len(null):
        return np.nan, int(len(null))
    below = int((null < obs - tie).sum())
    ties = int((np.abs(null - obs) <= tie).sum())
    return 100.0 * (below + 0.5 * ties) / len(null), int(len(null))


def two_sided_p(pct: float) -> float:
    """Two-sided p of a null percentile: min(1, 2 * min(pct, 100 - pct) / 100)."""
    if not np.isfinite(pct):
        return np.nan
    return min(1.0, 2 * min(pct, 100 - pct) / 100)


def boot_counts(n_units: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_units) multiplicities of resampling the units with replacement."""
    draws = rng.integers(0, n_units, size=(n_boot, n_units))
    W = np.zeros((n_boot, n_units))
    for b in range(n_boot):
        W[b] = np.bincount(draws[b], minlength=n_units)
    return W


def interval(vals: np.ndarray) -> tuple[float, float, int]:
    """2.5 / 97.5 percentiles of the finite values, and their number."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, 0
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(len(v))


def label_weights(n_obs_r: np.ndarray, n_null_r: np.ndarray, labs) -> np.ndarray:
    """(draws, 2) weights giving each null patch the region's label mix: (region pairs of label l)
    / (patch pairs of label l) for l in labs, else 0.

    S2 patches are not matched on label composition (a patch near an all-positive seed-0 region
    can be 10-70% negatives), so for label scope all this reweighted null is reported beside the
    specified one as a sensitivity (null_pct_label_weighted)."""
    w = np.zeros_like(n_null_r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for label in labs:
            w[:, label] = np.where(n_null_r[:, label] > 0, n_obs_r[label] / n_null_r[:, label], 0.0)
    return w


# ------------------------------------------------------------------------------------ pairs
def observed_pairs(K: pd.DataFrame, data: dict[int, SeedData]) -> pd.DataFrame:
    """The S1 matched rows with row positions ir (region point) and ic (twin), after checking
    S1's rules: same label and tile, both on land, the twin unselected and outside every region,
    no twin used twice."""
    matched = K[K.status == "matched"]
    per_seed = []
    for s, rows in matched.groupby("seed"):
        D = data[s]
        ir = D.pos(rows.point_id.astype(str))
        ic = D.pos(rows.control_point_id.astype(str))
        failed = []
        if not (D.label[ir] == D.label[ic]).all() or not (D.label[ir] == rows.label.to_numpy()).all():
            failed.append("label")
        if not (D.tile[ir] == D.tile[ic]).all():
            failed.append("tile")
        if not (D.land[ir].all() and D.land[ic].all()):
            failed.append("land")
        if D.selected[ic].any() or (D.component[ic] >= 0).any():
            failed.append("control is selected")
        if not (D.component[ir] == rows.component.to_numpy()).all():
            failed.append("component")
        if len(set(ic)) != len(ic):
            failed.append("control reused")
        if failed:
            raise SystemExit(f"S1 controls, seed {s}: failed checks {failed}")
        per_seed.append(rows.assign(ir=ir, ic=ic, label=D.label[ir]))
    return pd.concat(per_seed, ignore_index=True)


def null_pairs(row, D: SeedData) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of an S2 patch's land points that have a twin, and of their twins."""
    land = np.asarray(row.land_point_ids).astype(object)
    twins = np.asarray(row.control_point_ids, dtype=object)
    has_twin = np.array([isinstance(t, str) for t in twins], bool)
    return D.pos(land[has_twin].astype(str)), D.pos(twins[has_twin].astype(str))


def frequent_classes(OP: pd.DataFrame, data: dict[int, SeedData], lay: Layout,
                     regions: pd.DataFrame) -> list:
    """The classes that get a share_diff row, most frequent first: (var, class, var_index, code)
    for every class reaching CLASS_MIN_SHARE among the region points or among the twins of some
    region (label scope all, or positives for positive-only variables) or of some pooled
    (type, label scope) with >= CLASS_MIN_PAIRS pairs."""
    by_seed = list(OP.groupby("seed"))
    pairs = pd.concat([seed_pairs[["seed", "component", "label"]] for _, seed_pairs in by_seed],
                      ignore_index=True)
    pairs = pairs.merge(regions[["seed", "component", "type"]], on=["seed", "component"], how="left")
    labels = pairs.label.to_numpy()
    scopes = list(pairs.groupby(["seed", "component"]).indices.values())
    for region_type in TYPES:
        for _, labs in LABEL_SCOPES:
            in_scope = (pairs.type == region_type).to_numpy() & pairs.label.isin(labs).to_numpy()
            rows = np.flatnonzero(in_scope)
            if len(rows) >= CLASS_MIN_PAIRS:
                scopes.append(rows)

    classes = []
    for j, v in enumerate(lay.cat_vars):
        region_codes = np.concatenate([data[s].cat[seed_pairs.ir.to_numpy(), j] for s, seed_pairs in by_seed])
        twin_codes = np.concatenate([data[s].cat[seed_pairs.ic.to_numpy(), j] for s, seed_pairs in by_seed])
        n_levels = len(lay.cat_levels[v])
        keep = set()
        for rows in scopes:
            rows = np.asarray(rows)
            if v in POSITIVE_ONLY:
                rows = rows[labels[rows] == 1]
            both = (region_codes[rows] >= 0) & (twin_codes[rows] >= 0)
            if not both.any():
                continue
            for side in (region_codes[rows][both], twin_codes[rows][both]):
                share = np.bincount(side, minlength=n_levels) / both.sum()
                keep |= set(np.flatnonzero(share >= CLASS_MIN_SHARE))
        present = np.concatenate([region_codes[region_codes >= 0], twin_codes[twin_codes >= 0]])
        total = np.bincount(present, minlength=n_levels)
        for code in sorted(keep, key=lambda c: -total[c]):
            classes.append((v, lay.cat_levels[v][code], j, int(code)))
    return classes


# ------------------------------------------------------------------------- B_contrast rows
def var_meta(kind: str, i: int, lay: Layout) -> dict:
    """The identifying columns of statistic (kind, i)."""
    if kind in ("num", "adj"):
        v = lay.num[i]
        stat = "adj_smd" if kind == "adj" else "smd"
        return {"variable": v, "group": NUMERIC[v], "kind": "numeric", "class": "", "stat": stat,
                "adjusted": kind == "adj"}
    if kind == "cat":
        v, class_name, _, _ = lay.classes[i]
        return {"variable": v, "group": CATEGORICAL[v], "kind": "categorical", "class": class_name,
                "stat": "share_diff", "adjusted": False}
    if kind == "tvd":
        v = lay.cat_vars[i]
        return {"variable": v, "group": CATEGORICAL[v], "kind": "categorical", "class": ALL_CLASSES,
                "stat": "tvd", "adjusted": False}
    v = MODEL_VARS[i]
    stat = "ratio_diff" if v == "fn_share_top" else "mean_diff"
    return {"variable": v, "group": "model_behaviour", "kind": "model", "class": "", "stat": stat,
            "adjusted": False}


def keep_row(meta: dict, lname: str, F: dict, kind: str, i: int) -> bool:
    """Whether statistic (kind, i) gets a row in label scope lname."""
    variable = meta["variable"]
    if variable in POSITIVE_ONLY and lname != "positives":
        return False
    # Within one label every error is a false negative (positives) or none is (negatives).
    if variable == "fn_share_top" and lname != "all":
        return False
    if kind == "cat" and not (F["share_r"][i] > 0 or F["share_c"][i] > 0):
        return False
    return True


def describe(F: dict, kind: str, i: int) -> dict:
    """The descriptive columns of statistic (kind, i)."""
    if kind in ("num", "adj"):
        return dict(diff_raw=F["adj_diff"][i] if kind == "adj" else F["diff"][i],
                    mean_region=F["mean_r"][i], mean_control=F["mean_c"][i], sd_pooled=F["den"][i],
                    n_pairs=int(F["n"][i]))
    if kind == "cat":
        return dict(diff_raw=F["share"][i], mean_region=F["share_r"][i], mean_control=F["share_c"][i],
                    sd_pooled=np.nan, n_pairs=int(F["n_cat"][i]))
    if kind == "tvd":
        return dict(diff_raw=F["tvd"][i], mean_region=np.nan, mean_control=np.nan,
                    sd_pooled=np.nan, n_pairs=int(F["n_tvd"][i]))
    return dict(diff_raw=F["model"][i], mean_region=F["model_r"][i], mean_control=F["model_c"][i],
                sd_pooled=np.nan, n_pairs=int(F["n_model"]))


def null_summary(value: float, draws: np.ndarray) -> dict:
    """Null percentile of the observed value, its two-sided p, and the null's median and 95% range."""
    pct, n_finite = midrank_pct(value, draws)
    finite = draws[np.isfinite(draws)]
    median = lo = hi = np.nan
    if len(finite):
        median = float(np.median(finite))
        lo = float(np.percentile(finite, 2.5))
        hi = float(np.percentile(finite, 97.5))
    return dict(null_pct=pct, null_p=two_sided_p(pct), null_n=n_finite, null_median=median,
                null_lo=lo, null_hi=hi)


def null_draws_frame(head: dict, lname: str, meta: dict, draws: np.ndarray,
                     draws_weighted: np.ndarray) -> pd.DataFrame:
    """B_null rows: every null draw of one pooled statistic."""
    return pd.DataFrame({"type": head["type"], "label_scope": lname, "variable": meta["variable"],
                         "group": meta["group"], "class": meta["class"], "stat": meta["stat"],
                         "adjusted": meta["adjusted"], "draw": np.arange(len(draws)), "null_value": draws,
                         "null_value_label_weighted": draws_weighted})


def make_rows(head: dict, lname: str, F: dict, Fb: dict | None, Fn: dict, Fw: dict | None,
              lay: Layout, null_sink: list | None = None) -> list[dict]:
    """B_contrast rows of one scope and label scope, from finalize outputs: F observed, Fb the
    bootstrap draws, Fn the null draws, Fw the label-weighted null draws (Fb, Fn, Fw may be None).
    With null_sink, every null draw is also appended to it as B_null rows."""
    observed = flat_stats(F, lay)
    boot = flat_stats(Fb, lay) if Fb is not None else None
    null = flat_stats(Fn, lay) if Fn is not None else None
    null_weighted = flat_stats(Fw, lay) if Fw is not None else None
    rows = []
    for (kind, i), value in observed.items():
        meta = var_meta(kind, i, lay)
        if not keep_row(meta, lname, F, kind, i):
            continue
        row = {**head, "label_scope": lname, **meta, "smd_or_share_diff": float(value), **describe(F, kind, i)}
        if boot is not None:
            lo, hi, n_used = interval(boot[(kind, i)])
        else:
            lo, hi, n_used = np.nan, np.nan, 0
        row.update(boot_lo=lo, boot_hi=hi, n_boot_used=n_used)
        if null is not None:
            draws = null[(kind, i)]
            row.update(null_summary(value, draws))
            if null_sink is not None:
                if null_weighted is not None:
                    draws_weighted = null_weighted[(kind, i)]
                else:
                    draws_weighted = np.full(len(draws), np.nan)
                null_sink.append(null_draws_frame(head, lname, meta, draws, draws_weighted))
        if null_weighted is not None:
            pct, n_finite = midrank_pct(value, null_weighted[(kind, i)])
            row.update(null_pct_label_weighted=pct, null_n_label_weighted=n_finite)
        rows.append(row)
    return rows


# -------------------------------------------------------------------------------------- run
def region_null_sums(patches: pd.DataFrame, D: SeedData, lay: Layout, n_draws: int, key) -> dict:
    """One region's S2 sums per draw and label: (n_draws, 2, ...), after checking that the draws
    are 0..n_draws-1 and every twin has its patch point's label and tile."""
    seed, comp = key
    if len(patches) != n_draws or not (patches.draw.to_numpy() == np.arange(n_draws)).all():
        raise SystemExit(f"S2 null for seed {seed} component {comp}: expected draws 0..{n_draws - 1}")
    irs, ics, units = [], [], []
    for d, patch in enumerate(patches.itertuples()):
        ir, ic = null_pairs(patch, D)
        if not ((D.label[ir] == D.label[ic]).all() and (D.tile[ir] == D.tile[ic]).all()):
            raise SystemExit(f"S2 null seed {seed} comp {comp} draw {d}: twin label/tile mismatch")
        irs.append(ir)
        ics.append(ic)
        units.append(2 * d + D.label[ir])
    ir, ic, unit = np.concatenate(irs), np.concatenate(ics), np.concatenate(units)
    sums = group_sums(pair_stats(D, ir, ic, lay), unit, 2 * n_draws)
    return {k: a.reshape((n_draws, 2) + a.shape[1:]) for k, a in sums.items()}


def build_unit_stats(OP: pd.DataFrame, N: pd.DataFrame, data: dict[int, SeedData], lay: Layout,
                     regions: pd.DataFrame, n_draws: int):
    """Per region (seed, component): observed sums by label (2, ...), null sums by draw and label
    (n_draws, 2, ...), the pair counts of both, and the per-pair statistics with their labels."""
    obs, null, n_obs, n_null, pair_st = {}, {}, {}, {}, {}
    for _, region in regions.iterrows():
        key = (int(region.seed), int(region.component))
        seed, comp = key
        D = data[seed]
        pairs = OP[(OP.seed == seed) & (OP.component == comp)]
        st = pair_stats(D, pairs.ir.to_numpy(), pairs.ic.to_numpy(), lay)
        labels = pairs.label.to_numpy()
        obs[key] = group_sums(st, labels, 2)
        n_obs[key] = np.bincount(labels, minlength=2)
        pair_st[key] = (st, labels)
        patches = N[(N.seed == seed) & (N.component == comp)].sort_values("draw")
        null[key] = region_null_sums(patches, D, lay, n_draws, key)
        n_null[key] = null[key]["npair"][..., 0]
    return obs, null, n_obs, n_null, pair_st


def region_rows(units, rc: pd.DataFrame, lay: Layout, n_boot: int, rng_seed: int) -> list[dict]:
    """B_contrast rows per region, from build_unit_stats' output (units). The interval resamples
    the region's pairs; they are spatially autocorrelated, so it is optimistic and only
    descriptive. The null uses the region's patches."""
    obs, null, n_obs, n_null, pair_st = units
    rows = []
    for key in obs:
        seed, comp = key
        head = dict(scope="region", seed=seed, component=comp, counties=rc.loc[key, "counties"],
                    tiles=rc.loc[key, "tiles"], type=rc.loc[key, "type"], n_regions=1, boot_unit="pairs")
        st, labels = pair_st[key]
        for li, (lname, labs) in enumerate(LABEL_SCOPES):
            scope_labels = list(labs)
            if n_obs[key][scope_labels].sum() == 0:
                continue
            F = finalize(observed_sums(obs[key], scope_labels), lay)
            in_scope = np.flatnonzero(np.isin(labels, scope_labels))
            rng = np.random.default_rng([rng_seed, 1, seed, comp, li])
            Wb = np.zeros((n_boot, len(labels)))
            Wb[:, in_scope] = boot_counts(len(in_scope), n_boot, rng)
            Fb = finalize(aggregate(st, Wb), lay)
            Fn = finalize(null_sums(null[key], scope_labels), lay)
            Fw = None
            if lname == "all":
                w = label_weights(n_obs[key], n_null[key], scope_labels)
                Fw = finalize(weighted_l(null[key], w), lay)
            rows += make_rows(head, lname, F, Fb, Fn, Fw, lay)
    return rows


def weighted_null_sums(null: dict, n_obs: dict, n_null: dict, keys: list, scope_labels: list) -> dict:
    """Label-weighted null sums (see label_weights) pooled over the regions in keys."""
    total = None
    for k in keys:
        part = weighted_l(null[k], label_weights(n_obs[k], n_null[k], scope_labels))
        if total is None:
            total = part
        else:
            total = {name: total[name] + part[name] for name in total}
    return total


def pooled_head(rc: pd.DataFrame, keys: list, region_type: str) -> dict:
    """The identifying columns of a pooled scope."""
    counties = set()
    tiles = set()
    for k in keys:
        counties.add(rc.loc[k, "counties"])
        tiles.update(rc.loc[k, "tiles"].split())
    return dict(scope="pooled", seed=-1, component=-1, counties=" | ".join(sorted(counties)),
                tiles=" ".join(sorted(tiles)), type=region_type, n_regions=len(keys), boot_unit="regions")


def pooled_rows(units, rc: pd.DataFrame, lay: Layout, n_boot: int, rng_seed: int,
                null_sink: list) -> list[dict]:
    """B_contrast rows pooled by region type, from build_unit_stats' output (units); every null
    draw is also appended to null_sink as B_null rows. A pool is the matched pairs of all seeds'
    regions of the type as one set, each pair weighted equally, so the SMD's variances include the
    spread between regions. There is no all-regions pool, as the primary predicts opposite signs
    for the two types. A region enters a (type, label scope) iff it has >= 1 pair of that label
    scope (seed-0 regions have no negatives). The interval resamples whole regions (a region drawn
    twice counts twice); null draw d pools the d-th patch of every entering region."""
    obs, null, n_obs, n_null, _ = units
    rows = []
    for ti, region_type in enumerate(TYPES):
        for li, (lname, labs) in enumerate(LABEL_SCOPES):
            scope_labels = list(labs)
            keys = [k for k in obs if rc.loc[k, "type"] == region_type and n_obs[k][scope_labels].sum() > 0]
            if not keys:
                continue
            per_region = [observed_sums(obs[k], scope_labels) for k in keys]
            per_region_null = [null_sums(null[k], scope_labels) for k in keys]
            stacked, null_total = {}, {}
            for name in per_region[0]:
                stacked[name] = np.stack([sums[name] for sums in per_region])
                null_total[name] = sum(sums[name] for sums in per_region_null)
            F = finalize(usum(stacked, 0), lay)
            rng = np.random.default_rng([rng_seed, 0, ti, li])
            Fb = finalize(wsum(stacked, boot_counts(len(keys), n_boot, rng)), lay)
            Fn = finalize(null_total, lay)
            Fw = None
            if lname == "all":
                Fw = finalize(weighted_null_sums(null, n_obs, n_null, keys, scope_labels), lay)
            rows += make_rows(pooled_head(rc, keys, region_type), lname, F, Fb, Fn, Fw, lay, null_sink)
    return rows


def flag_findings(contrast: pd.DataFrame) -> pd.DataFrame:
    """Add the primary, predicted_sign, met, test_family and secondary columns.

    Primary (fixed): the pooled, unadjusted SMD of PRIMARY in label scope all; met iff the
    bootstrap interval excludes 0 in the predicted direction and the null percentile is beyond 95
    (or below 5). Secondary: any other pooled, unadjusted descriptor row with a null percentile
    beyond 97.5 or below 2.5. Two test families are counted apart: 'variable' (one test per type,
    label scope and variable: the numeric SMD or the categorical TVD) and 'class' (the per-class
    share differences, many and often tied)."""
    contrast["primary"] = ((contrast.scope == "pooled") & (contrast.variable == PRIMARY)
                           & (contrast.label_scope == "all") & ~contrast.adjusted)
    contrast["predicted_sign"] = np.where(contrast.variable == PRIMARY, contrast.type.map(PREDICTED), 0)
    predicted_up = contrast.predicted_sign > 0
    met_up = (contrast.boot_lo > 0) & (contrast.null_pct > PRIMARY_PCT_HI)
    met_down = (contrast.boot_hi < 0) & (contrast.null_pct < PRIMARY_PCT_LO)
    contrast["met"] = contrast.primary & np.where(predicted_up, met_up, met_down)
    descriptor = (contrast.scope == "pooled") & ~contrast.adjusted & contrast.group.isin(DESCRIPTOR_GROUPS)
    per_variable = descriptor & ((contrast.kind == "numeric") | (contrast.stat == "tvd"))
    per_class = descriptor & (contrast.stat == "share_diff")
    contrast["test_family"] = np.select([per_variable, per_class], ["variable", "class"], default="")
    beyond = (contrast.null_pct > SECONDARY_HI) | (contrast.null_pct < SECONDARY_LO)
    contrast["secondary"] = (contrast.test_family != "") & ~contrast.primary & beyond
    return contrast


def balance(OP: pd.DataFrame, data: dict[int, SeedData], regions: pd.DataFrame) -> pd.DataFrame:
    """Terrain balance of the pairs used, per region (label scope all) and pooled by type and label
    scope: the mean region-minus-twin difference in seed-SD units, and the SMD in raw units."""
    by_seed = list(OP.groupby("seed", sort=False))
    z_region = np.concatenate([data[s].z[seed_pairs.ir] for s, seed_pairs in by_seed])
    z_twin = np.concatenate([data[s].z[seed_pairs.ic] for s, seed_pairs in by_seed])
    raw_region = np.concatenate([data[s].raw_terrain[seed_pairs.ir] for s, seed_pairs in by_seed])
    raw_twin = np.concatenate([data[s].raw_terrain[seed_pairs.ic] for s, seed_pairs in by_seed])
    pairs = pd.concat([seed_pairs for _, seed_pairs in by_seed], ignore_index=True)
    # regions_land.csv's type wins over the S1 file's own type column
    pairs = pairs.merge(regions[["seed", "component", "type"]], on=["seed", "component"], how="left",
                        suffixes=("_s1", ""))
    scopes = []
    for s, c, region_type in regions[["seed", "component", "type"]].itertuples(index=False):
        mask = ((pairs.seed == s) & (pairs.component == c)).to_numpy()
        scopes.append(("region", s, c, region_type, "all", mask))
    for region_type in TYPES:
        for lname, labs in LABEL_SCOPES:
            mask = ((pairs.type == region_type) & pairs.label.isin(labs)).to_numpy()
            scopes.append(("pooled", -1, -1, region_type, lname, mask))

    rows = []
    for scope, s, c, region_type, lname, mask in scopes:
        if not mask.any():
            continue
        n_pairs = int(mask.sum())
        for j, v in enumerate(TERRAIN):
            zr, zc = z_region[mask, j], z_twin[mask, j]
            raw_r, raw_c = raw_region[mask, j], raw_twin[mask, j]
            sd_matched = np.nan
            if n_pairs > 1:
                sd_matched = float(np.sqrt((np.var(zr, ddof=1) + np.var(zc, ddof=1)) / 2))
            rows.append(dict(scope=scope, seed=s, component=c, type=region_type, label_scope=lname,
                             variable=v, n_pairs=n_pairs, mean_region=float(raw_r.mean()),
                             mean_control=float(raw_c.mean()), diff_seed_sd=float((zr - zc).mean()),
                             smd=mc.smd(raw_r, raw_c), sd_matched_seed_sd=sd_matched))
    return pd.DataFrame(rows)


def load_seed_data(args, table: pd.DataFrame, lay: Layout) -> dict[int, SeedData]:
    """Every seed's SeedData, printing its top models."""
    ctxs = mc.load_contexts(args.seeds, args.cells_dir, args.cell_prefix, args.data, args.sel_root,
                            args.regions_dir, args.budget, args.min_size, verbose=False)
    data = {}
    for s, ctx in ctxs.items():
        scores = model_scores(args.cells_dir, ctx.tag, args.top_k)
        data[s] = seed_data(ctx, table, lay, scores)
        top = " ".join(str(m) for m in scores["top"])
        families = pd.Series(scores["kinds"]).value_counts().to_dict()
        print(f"seed {s}: {len(ctx.pts)} Zone A test points; top {scores['k']} test models by auc_id: "
              f"idx {top}; auc_id {scores['auc_top'].min():.4f}-{scores['auc_top'].max():.4f}; "
              f"families {families}", flush=True)
    return data


def check_region_types(regions: pd.DataFrame, N: pd.DataFrame) -> None:
    """Stop if S2_null.parquet types a region differently from regions_land.csv."""
    s2_types = N.groupby(["seed", "component"]).region_type.first().reset_index()
    both = regions.merge(s2_types, on=["seed", "component"], how="left")
    if not (both.type == both.region_type).all():
        raise SystemExit("region types in regions_land.csv and S2_null.parquet disagree")


def print_region_pairs(rl: pd.DataFrame, K: pd.DataFrame, OP: pd.DataFrame) -> None:
    """Per region: points, land points, water drops, unmatched land points and matched pairs."""
    status = K.groupby(["seed", "component"]).status.value_counts().unstack(fill_value=0)
    print("\nper region: land rule drops and matched pairs")
    rows = []
    for region in rl.itertuples():
        if (region.seed, region.component) in status.index:
            counts = status.loc[(region.seed, region.component)]
        else:
            counts = pd.Series(dtype=int)
        pairs = OP[(OP.seed == region.seed) & (OP.component == region.component)]
        rows.append(dict(seed=region.seed, component=region.component, type=region.type,
                         counties=region.counties, n=region.n, n_land=region.n_land,
                         dropped_water=region.dropped_water,
                         unmatched=int(region.n_land - counts.get("matched", 0)), pairs=len(pairs),
                         pairs_pos=int((pairs.label == 1).sum()), pairs_neg=int((pairs.label == 0).sum())))
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))


def print_classes(lay: Layout) -> None:
    """The variable counts and each categorical variable's reported classes."""
    print(f"\n{lay.V} numeric variables; {len(lay.cat_vars)} categorical variables with "
          f"{lay.J} reported classes (>= {CLASS_MIN_SHARE:.0%} somewhere):")
    for v in lay.cat_vars:
        names = [class_name for var, class_name, _, _ in lay.classes if var == v]
        print(f"  {v}: " + "; ".join(names))


def run(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B_contrast, B_balance and B_null for the parsed arguments, printing progress."""
    t0 = time.time()
    rl = pd.read_csv(args.regions_land)
    regions = rl[rl.type.isin(TYPES)].reset_index(drop=True)
    dropped = ", ".join(f"s{r.seed}c{r.component} ({r.type})" for r in rl[~rl.type.isin(TYPES)].itertuples())
    print(f"regions: {len(rl)} in {args.regions_land}; {len(regions)} with land points "
          f"({(regions.type == TYPES[0]).sum()} positive-dominated, {(regions.type == TYPES[1]).sum()} "
          f"negative-dominated); dropped: {dropped}")
    table = load_table(args.data, args.extras)
    lay = make_layout(table)
    data = load_seed_data(args, table, lay)
    K = pd.read_csv(args.controls, dtype={"point_id": str, "control_point_id": str})
    OP = observed_pairs(K, data)
    N = pd.read_parquet(args.null)
    calipers = set(N.caliper.astype(str))
    print(f"S1 controls: {len(K)} region land points, {len(OP)} matched pairs; S2 null: {len(N)} rows, "
          f"caliper {calipers}, max S1 distance {K.distance.max():.3f}")
    check_region_types(regions, N)
    print_region_pairs(rl, K, OP)

    lay.classes = frequent_classes(OP, data, lay, regions)
    print_classes(lay)
    units = build_unit_stats(OP, N, data, lay, regions, args.draws)
    print(f"sufficient statistics built ({time.time() - t0:.1f} s)", flush=True)

    rc = regions.set_index(["seed", "component"])
    rows = region_rows(units, rc, lay, args.n_boot, args.rng_seed)
    print(f"region rows done ({time.time() - t0:.1f} s)", flush=True)
    null_frames = []
    rows += pooled_rows(units, rc, lay, args.n_boot, args.rng_seed, null_frames)
    print(f"pooled rows done ({time.time() - t0:.1f} s)", flush=True)

    contrast = flag_findings(pd.DataFrame(rows))
    null_draws = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    return contrast, balance(OP, data, regions), null_draws


# ----------------------------------------------------------------------------------- report
def signed(x) -> str:
    """x as +0.000, or 'nan'."""
    if not np.isfinite(x):
        return "nan"
    return f"{x:+.3f}"


def wide_tables():
    """pandas display options for the report's tables."""
    return pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 40)


def report_primary(contrast: pd.DataFrame) -> None:
    print("\n=== PRIMARY: SMD of log10_dist_river, region minus twins, pooled over seeds, label scope all")
    for r in contrast[contrast.primary].itertuples():
        sign = "> 0" if r.predicted_sign > 0 else "< 0"
        verdict = "MET" if r.met else "NOT MET"
        print(f"  {r.type:<19} SMD {signed(r.smd_or_share_diff)} [{signed(r.boot_lo)}, {signed(r.boot_hi)}] "
              f"(bootstrap over {r.n_regions} regions, {r.n_boot_used} draws); means {r.mean_region:.3f} vs "
              f"{r.mean_control:.3f} log10 m on {r.n_pairs} pairs; null pct {r.null_pct:.1f} (n {r.null_n}; "
              f"label-weighted {r.null_pct_label_weighted:.1f}); predicted {sign}: {verdict}")
    pooled_primary = (contrast.scope == "pooled") & (contrast.variable == PRIMARY)
    print("  terrain-adjusted (OLS on indicator + 6 terrain, same denominator):")
    for r in contrast[pooled_primary & contrast.adjusted].itertuples():
        print(f"    {r.type:<19} {r.label_scope:<9} adj SMD {signed(r.smd_or_share_diff)} "
              f"[{signed(r.boot_lo)}, {signed(r.boot_hi)}], null pct {r.null_pct:.1f}; raw coef {r.diff_raw:+.4f}")
    print("  by label scope (unadjusted):")
    for r in contrast[pooled_primary & ~contrast.adjusted].itertuples():
        print(f"    {r.type:<19} {r.label_scope:<9} SMD {signed(r.smd_or_share_diff)} "
              f"[{signed(r.boot_lo)}, {signed(r.boot_hi)}], null pct {r.null_pct:.1f}, n {r.n_pairs}")
    per_region = contrast[(contrast.scope == "region") & (contrast.variable == PRIMARY) & ~contrast.adjusted
                          & (contrast.label_scope == "all")]
    print("  per region (label scope all; pair-bootstrap interval):")
    for r in per_region.itertuples():
        print(f"    s{r.seed}c{r.component} {r.type:<19} {r.counties:<12} SMD {signed(r.smd_or_share_diff)} "
              f"[{signed(r.boot_lo)}, {signed(r.boot_hi)}] null pct {r.null_pct:5.1f}  "
              f"means {r.mean_region:.3f} vs {r.mean_control:.3f}, n {r.n_pairs}")


def report_secondary(contrast: pd.DataFrame) -> None:
    """Secondary findings per test family, against the 5% expected by chance. The rows are
    correlated (cat/ws twins of one StreamCat measure, the Atlas 14 depths, the label scopes), so
    the count spreads wider than a binomial. A TVD below 2.5 means more alike than the null."""
    show = ["type", "label_scope", "variable", "class", "smd_or_share_diff", "boot_lo", "boot_hi",
            "null_pct", "null_pct_label_weighted", "mean_region", "mean_control", "n_pairs", "n_regions"]
    families = (("variable", "one test per variable and scope: numeric SMD or categorical "
                             "total variation distance"),
                ("class", "per-class share differences of the categorical variables"))
    print(f"\n=== SECONDARY findings (pooled, unadjusted, null pct > {SECONDARY_HI} or < {SECONDARY_LO})")
    for family, what in families:
        tested = contrast[(contrast.test_family == family) & ~contrast.primary & contrast.null_pct.notna()]
        found = tested[tested.secondary]
        print(f"\n  family '{family}' ({what}): {len(tested)} rows tested ({tested.variable.nunique()} variables "
              f"x type x label scope); expected by chance 0.05 x {len(tested)} = {0.05 * len(tested):.1f}; "
              f"found {len(found)}")
        for (region_type, lname), group in tested.groupby(["type", "label_scope"]):
            print(f"    {region_type:<19} {lname:<9}: {int(group.secondary.sum())} of {len(group)} "
                  f"(expected {0.05 * len(group):.1f})")
        if len(found):
            with wide_tables():
                ordered = found.sort_values(["type", "label_scope", "group", "variable"])
                print(ordered[show].round(3).to_string(index=False))
        excluding_zero = found[(found.boot_lo > 0) | (found.boot_hi < 0)]
        print(f"  of these, {len(excluding_zero)} also have a region-bootstrap interval excluding 0")
    per_region = contrast[(contrast.scope == "region") & ~contrast.adjusted
                          & contrast.group.isin(DESCRIPTOR_GROUPS) & contrast.null_pct.notna()
                          & ((contrast.kind == "numeric") | (contrast.stat == "tvd"))]
    n_beyond = int(((per_region.null_pct > SECONDARY_HI) | (per_region.null_pct < SECONDARY_LO)).sum())
    print(f"\n  per region, family 'variable' (not counted above): {n_beyond} of {len(per_region)} rows "
          f"beyond 2.5/97.5 (expected {0.05 * len(per_region):.1f})")


def report_model_behaviour(contrast: pd.DataFrame) -> None:
    print("\n=== MODEL BEHAVIOUR (top 15 of the 60 test models by auc_id), region minus twins")
    model = contrast[contrast.group == "model_behaviour"]
    pooled = model[model.scope == "pooled"]
    print(pooled[["type", "label_scope", "variable", "mean_region", "mean_control", "smd_or_share_diff",
                  "boot_lo", "boot_hi", "null_pct", "null_pct_label_weighted", "n_pairs", "n_regions"]]
          .round(3).to_string(index=False))
    print("  per region (label scope all):")
    per_region = model[(model.scope == "region") & (model.label_scope == "all")]
    print(per_region[["seed", "component", "type", "variable", "mean_region", "mean_control",
                      "smd_or_share_diff", "null_pct", "n_pairs"]].round(3).to_string(index=False))


def report_adjusted(contrast: pd.DataFrame) -> None:
    print("\n=== TERRAIN-ADJUSTED vs unadjusted (pooled, label scope all, numeric descriptors)")
    numeric = contrast[(contrast.scope == "pooled") & contrast.label_scope.isin(["all", "positives"])
                       & (contrast.kind == "numeric")]
    # positive-only variables have rows in label scope positives only
    numeric = numeric[(numeric.label_scope == "all") | numeric.variable.isin(POSITIVE_ONLY)]
    table = numeric.pivot_table(index=["type", "variable"], columns="adjusted",
                                values=["smd_or_share_diff", "null_pct"], aggfunc="first")
    table.columns = [f"{value}_{'adj' if adjusted else 'raw'}" for value, adjusted in table.columns]
    print(table.round(3).to_string())


def report_balance(bal: pd.DataFrame) -> None:
    print("\n=== BALANCE of the pairs used (terrain, mean difference in seed-SD units)")
    pooled = bal[bal.scope == "pooled"]
    table = pooled.pivot_table(index=["type", "label_scope"], columns="variable", values="diff_seed_sd",
                               aggfunc="first")
    print(table.round(3).to_string())
    per_region = bal[bal.scope == "region"]
    worst = per_region.diff_seed_sd.abs().idxmax()
    print(f"  max |diff| per region: {per_region.diff_seed_sd.abs().max():.3f} "
          f"(s{per_region.loc[worst, 'seed']}c{per_region.loc[worst, 'component']} "
          f"{per_region.loc[worst, 'variable']}); pooled max {pooled.diff_seed_sd.abs().max():.3f}")


def report(contrast: pd.DataFrame, bal: pd.DataFrame) -> None:
    report_primary(contrast)
    report_secondary(contrast)
    with wide_tables():
        report_model_behaviour(contrast)
        report_adjusted(contrast)
        report_balance(bal)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", type=Path, default=Path("results/2026-10-17/o5/S1_controls.csv"))
    ap.add_argument("--null", type=Path, default=Path("results/2026-10-17/o5/S2_null.parquet"))
    ap.add_argument("--regions-land", type=Path, default=Path("results/2026-10-17/o5/regions_land.csv"))
    ap.add_argument("--data", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    ap.add_argument("--extras", type=Path, default=Path("data/interim/analysis/point_extras_FL.parquet"))
    ap.add_argument("--cells-dir", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--cell-prefix", default="FL_band_AE")
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--draws", type=int, default=200, help="S2 draws per region")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--top-k", type=int, default=TOP_K, help="strongest test models whose behaviour is compared")
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="B")
    args = ap.parse_args()
    t0 = time.time()
    contrast, bal, null_draws = run(args)
    report(contrast, bal)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.prefix
    contrast.to_csv(f"{prefix}_contrast.csv", index=False)
    bal.to_csv(f"{prefix}_balance.csv", index=False)
    null_draws.to_csv(f"{prefix}_null.csv", index=False)
    print(f"\nwrote {prefix}_contrast.csv ({len(contrast)} rows), {prefix}_balance.csv ({len(bal)} rows), "
          f"{prefix}_null.csv ({len(null_draws)} rows) in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
