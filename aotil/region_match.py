#!/usr/bin/env python3
"""
The region analysis of the paper's Approach 3.6: caliper matching of selected regions to
held-out Zone AE points.

Each point of a region is matched to Zone AE test points with the same land cover and nearly the
same numeric inputs. A point with no match has inputs that Zone AE does not cover (covariate
shift); a point whose matches carry a different label has the same inputs labeled differently
(concept shift). In CS terms: a nearest-neighbour lookup in the standardized feature space,
restricted to one land-cover class and to tiles the region does not touch, with a maximum
distance (the caliper).

Definitions (3.6):
  z(x)      the q = 6 numeric model inputs (FEATURES.txt minus landcover), each standardized by
            its mean and standard deviation (ddof 1, as pandas computes it) over D_OOD, all Zone A
            test points.
  pool      the Zone AE test points (D_ID) with all q features present; the others are dropped
            and counted (missing values, not zeros).
  P_C       candidates for region C: the pool points in test tiles (tile_50km) that hold no point
            of C (--exclude region-tiles, the paper's rule). --exclude own-tile drops only each
            query point's own tile, the rule of the 2026-10-25 sandbox reference, for Stop S2 of
            docs/TASK-RERUN-ALL-2026-10-25.md.
  C'_x      the candidates with x's land-cover class at Euclidean distance <= eps in z, the
            --max-matches closest; x is matched when C'_x is not empty.
  eps       the caliper: every pool point is matched against the pool as a one-point region (its
            own tile and itself excluded); eps is the --caliper-pct percentile (numpy's linear
            rule) of the distance to the nearest candidate, infinite when the point's class has
            no candidate. So about 5% of Zone AE points are unmatched.
  eta       Eq. disagree: over the matched points of a set, the mean of the share of each
            point's matches whose label differs from its own (each matched point counts once).
  baseline  the pool matched against itself as one-point regions.

Nearest neighbours: one KD-tree per land-cover class. A query asks for QUERY_START candidates,
drops the ineligible ones (excluded tiles, the point itself) and asks again for QUERY_GROWTH
times as many until it has enough eligible candidates within eps, has seen every candidate
within eps, or has exhausted the class. (In the sandbox a fixed query of 200 left points of
dense tiles wrongly unmatched: all 200 lay in excluded tiles.)

Sets, one output row each:
  region          each region of --regions (pieces of >= --min-size points), named by its
                  counties as logs/2026-10-24/C5_regions_describe.py names them
  pooled          all regions together
  baseline        the Zone AE pool against itself
  reference       one-point regions: the other Zone A test points (D_OOD outside the
                  selection), the 500 hardest points (most errors of the 60 test models, ties at
                  the cut in a seeded random order), and the --classic (OODSelect) selection
  halo            Zone A test points outside the selection within 1, 2 and 3 km (straight line,
                  EPSG:3086 metres) of a region; each excludes its own tile and the tiles of
                  every region it is near
  extra_region    each region of --extra-regions (the GTV selection), for the report only
Columns: n, n_matched, unmatched, eta; pos_points (share of positives), pos_matched (among the
matched points), pos_matches (mean over matched points of the mean match label, the direction
of the disagreement); dist_median (over all match pairs); n_distinct_matches; smd_max and
smd_feature (the largest |mean z of the matched points - mean z of their matches|, each match
weighted 1/|C'_x|, in D_OOD standard deviations); landcover_top; and, as description, the 15
strongest test models' accuracy (acc15_*) and the share of their errors that are false negatives
(fn_share15), and the median height above drainage (hand_*) and
distance to the nearest river (river_km_*) of the points and of their matches (weighted
medians). These descriptive columns are not in 3.6.

Also written: <out>/region_match_kinds.csv (positive and negative regions, by the majority
label: the shares of positives among the points and their matches, the pooled unmatched share and
eta, the strongest models' FN and FP shares of their errors, corr_te over the 60 test models
and over those below 99% training accuracy) and <out>/region_match_sensitivity.csv (eps at other
percentiles, other match caps), and <out>/region_match.json (eps, pool counts).

    python aotil/region_match.py --cells-dir data/interim/aotil/cells_rerun_<date> \\
        --tag FL_band_AE_s0 --dataset data/processed_w7/dataset_FL.parquet \\
        --features data/processed_w7/FEATURES.txt \\
        --regions results/<date>/regions/FL_band_AE_s0_spatial_mu3_knn10_S500_points.parquet \\
        --classic results/<date>/regions/FL_band_AE_s0_classic_S500_points.parquet \\
        --extra-regions results/<date>/regions/FL_band_AE_s0_spatial_gtv_mu3_knn10_S500_points.parquet \\
        --out results/<date>/region_match
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import oodselect_paper as op  # noqa: E402
from regions import NLCD, point_county  # noqa: E402

LANDCOVER = "landcover"      # matched exactly, so not one of the numeric features
N_NUMERIC = 6                # q in the paper
MAX_MATCHES = 5
CALIPER_PCT = 95.0           # 5% of Zone AE points unmatched, after Stuart (2010)
MIN_REGION = 20              # c in the paper
STRONGEST = 15               # the strongest test models, as o1_shift.strongest_models picks them
INTERPOLATES = 0.99          # training accuracy at which a model interpolates (o5_interpolation)
HALO_KM = (1.0, 2.0, 3.0)
N_HARDEST = 500
QUERY_START = 64
QUERY_GROWTH = 4
EXCLUDE_RULES = ("region-tiles", "own-tile")
SENSITIVITY = [("caliper_pct", 90.0), ("caliper_pct", 99.0), ("max_matches", 1),
               ("max_matches", 10)]
COUNTIES = Path("data/interim/gis/counties_FL.parquet")


# ------------------------------------------------------------------ inputs
def numeric_features(path: str | Path) -> list[str]:
    """The model inputs of FEATURES.txt except land cover; there must be N_NUMERIC of them."""
    names = [line.strip() for line in Path(path).read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    numeric = [n for n in names if n != LANDCOVER]
    if LANDCOVER not in names or len(numeric) != N_NUMERIC:
        raise SystemExit(f"{path}: expected landcover and {N_NUMERIC} numeric features, got "
                         f"{names}")
    return numeric


def standardizer(ood: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """Mean and standard deviation (ddof 1) of each feature over D_OOD."""
    return ood[cols].mean(), ood[cols].std()


def strongest_test_models(models: pd.DataFrame) -> np.ndarray:
    """The STRONGEST test models of the 60/20/20 split by stored ID AUROC
    (o1_shift.strongest_models' rule)."""
    _, _, test = op.model_split(len(models))
    return test[np.argsort(-models.auc_id.to_numpy()[test])[:STRONGEST]]


def county_names(path: Path = COUNTIES) -> dict:
    """GEOID -> county name; an empty dict when the layer is missing (names fall back to FIPS)."""
    if not path.exists():
        return {}
    return dict(pd.read_parquet(path, columns=["GEOID", "NAME"]).values)


def region_name(points: pd.DataFrame, names: dict) -> str:
    """The counties a region's points lie in, by FIPS, joined by "/" (C5_regions_describe.py)."""
    fips = sorted(set(point_county(points).astype(str)))
    return "/".join(names.get(f, f) for f in fips)


# ------------------------------------------------------------------ matching
class Pool:
    """The Zone AE pool: standardized features, land cover, tile codes and labels, with one
    KD-tree per land-cover class."""

    def __init__(self, z: np.ndarray, landcover: np.ndarray, tile_code: np.ndarray,
                 y: np.ndarray):
        self.z = np.asarray(z, float)
        self.landcover = np.asarray(landcover)
        self.tile_code = np.asarray(tile_code, int)
        self.y = np.asarray(y)
        self.trees = {}
        for lc in np.unique(self.landcover):
            members = np.flatnonzero(self.landcover == lc)
            self.trees[lc] = (cKDTree(self.z[members]), members)

    def __len__(self) -> int:
        return len(self.z)


def match(pool: Pool, zq: np.ndarray, lcq: np.ndarray, excluded: np.ndarray,
          self_index: np.ndarray | None = None, k: int = MAX_MATCHES,
          eps: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The k closest eligible candidates of each query within eps (all distances when eps is
    None): (indices into the pool, -1 padded; distances, inf padded), each row sorted by distance.

    excluded[r, t] is True when query r may not match a candidate in tile code t; self_index[r]
    is the query's own pool index (-1 for a point outside the pool). A candidate is eligible when
    it has the query's land cover, lies in a tile that is not excluded, and is not the query."""
    zq = np.asarray(zq, float)
    lcq = np.asarray(lcq)
    nq = len(zq)
    if self_index is None:
        self_index = np.full(nq, -1)
    idx = np.full((nq, k), -1)
    dist = np.full((nq, k), np.inf)
    # scipy keeps neighbours strictly below its bound; the next float up keeps d == eps.
    bound = np.inf if eps is None else float(np.nextafter(eps, np.inf))
    for lc in np.unique(lcq):
        if lc not in pool.trees:
            continue                                   # no candidate of this class: unmatched
        tree, members = pool.trees[lc]
        n = len(members)
        pending = np.flatnonzero(lcq == lc)
        kq = min(QUERY_START, n)
        while len(pending):
            d, j = tree.query(zq[pending], k=kq, distance_upper_bound=bound)
            d = np.asarray(d).reshape(len(pending), kq)
            j = np.asarray(j).reshape(len(pending), kq)
            found = j < n                              # scipy pads missing neighbours with n
            cand = np.where(found, members[np.minimum(j, n - 1)], -1)
            ok = found & ~excluded[pending[:, None], pool.tile_code[np.maximum(cand, 0)]]
            ok &= cand != self_index[pending, None]
            # Every candidate within the bound was returned when a slot is left empty.
            done = (ok.sum(1) >= k) | (kq >= n) | ~found[:, -1]
            for r in np.flatnonzero(done):
                take = np.flatnonzero(ok[r])[:k]
                idx[pending[r], :len(take)] = cand[r, take]
                dist[pending[r], :len(take)] = d[r, take]
            pending = pending[~done]
            kq = min(n, kq * QUERY_GROWTH)
    return idx, dist


def own_tile_exclusion(tile_code: np.ndarray, n_tiles: int) -> np.ndarray:
    """excluded matrix of one-point regions: each query excludes its own tile only."""
    out = np.zeros((len(tile_code), n_tiles), bool)
    out[np.arange(len(tile_code)), tile_code] = True
    return out


def caliper(nearest: np.ndarray, pct: float = CALIPER_PCT) -> float:
    """The pct-th percentile (linear rule) of the pool's nearest-candidate distances, with an
    infinite distance for a point whose class has no candidate."""
    nearest = np.asarray(nearest, float)
    finite = np.isfinite(nearest)
    if finite.all():
        return float(np.percentile(nearest, pct))
    # The infinite distances sort last; the percentile is finite while it falls before them.
    stand_in = np.where(finite, nearest, nearest[finite].max() * 2 + 1)
    eps = float(np.percentile(stand_in, pct))
    if eps > nearest[finite].max():
        raise ValueError(f"the {pct}th percentile falls among the {int((~finite).sum())} "
                         f"points with no candidate")
    return eps


def disagreement(y: np.ndarray, idx: np.ndarray, pool_y: np.ndarray) -> np.ndarray:
    """Per query, the share of its matches whose label differs from its own (NaN unmatched)."""
    valid = idx >= 0
    differ = (pool_y[np.maximum(idx, 0)] != np.asarray(y)[:, None]) & valid
    n = valid.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, differ.sum(1) / n, np.nan)


def eta(y: np.ndarray, idx: np.ndarray, pool_y: np.ndarray) -> float:
    """Eq. disagree over the matched queries; NaN when none is matched."""
    share = disagreement(y, idx, pool_y)
    share = share[np.isfinite(share)]
    return float(share.mean()) if len(share) else np.nan


def match_means(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per query, the mean of a pool column over its matches (NaN unmatched)."""
    valid = idx >= 0
    v = np.where(valid[..., None] if values.ndim > 1 else valid,
                 values[np.maximum(idx, 0)], 0.0)
    n = valid.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = v.sum(1)
        return total / (n[:, None] if values.ndim > 1 else n)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The lower weighted median; NaN without values."""
    keep = np.isfinite(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return np.nan
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


# ------------------------------------------------------------------ one set
def set_row(name: str, kind: str, q: dict, idx: np.ndarray, dist: np.ndarray,
            ctx: dict, describe: bool = True) -> dict:
    """The output row of one matched set. q holds the queries' z, y, landcover, acc15, hand,
    river_m; ctx the pool, its acc15/hand/river_m and the feature names."""
    pool = ctx["pool"]
    matched = idx[:, 0] >= 0
    n, n_matched = len(idx), int(matched.sum())
    row = dict(set=name, kind=kind, n=n, n_matched=n_matched,
               unmatched=float(1 - n_matched / n) if n else np.nan,
               eta=eta(q["y"], idx, pool.y),
               pos_points=float(np.mean(q["y"])) if n else np.nan,
               pos_matched=float(np.mean(q["y"][matched])) if n_matched else np.nan,
               fn_share15=q["fn15"] / max(q["fn15"] + q["fp15"], 1))
    if not n_matched:
        return row
    m_idx, m_dist = idx[matched], dist[matched]
    valid = m_idx >= 0
    row["pos_matches"] = float(match_means(pool.y.astype(float), m_idx).mean())
    row["dist_median"] = float(np.median(m_dist[valid]))
    row["n_distinct_matches"] = int(len(np.unique(m_idx[valid])))
    smd = q["z"][matched].mean(0) - match_means(pool.z, m_idx).mean(0)
    worst = int(np.argmax(np.abs(smd)))
    row["smd_max"] = float(abs(smd[worst]))
    row["smd_feature"] = ctx["features"][worst]
    row["smd_signed"] = float(smd[worst])
    if not describe:
        return row
    shares = pd.Series(q["landcover"]).value_counts(normalize=True).head(3)
    row["landcover_top"] = "; ".join(f"{NLCD.get(int(c), c)} {s:.0%}" for c, s in shares.items())
    weights = np.where(valid, 1.0 / valid.sum(1, keepdims=True), 0.0).ravel()
    flat = np.maximum(m_idx, 0).ravel()
    row["acc15_points"] = float(np.mean(q["acc15"]))
    row["acc15_matched"] = float(np.mean(q["acc15"][matched]))
    row["acc15_matches"] = float(match_means(ctx["acc15"], m_idx).mean())
    row["hand_points"] = float(np.nanmedian(q["hand"]))
    row["hand_matches"] = weighted_median(ctx["hand"][flat], weights)
    row["river_km_points"] = float(np.nanmedian(q["river_m"])) / 1000
    row["river_km_matches"] = weighted_median(ctx["river_m"][flat], weights) / 1000
    return row


# ------------------------------------------------------------------ the run
def load_inputs(a) -> dict:
    """The cell, the dataset rows of both halves in cell order, the pool and the queries."""
    cell, Z_ood, acc, _ = op.load_paper_cell(a.cells_dir, a.tag)
    if cell.point_id is None or cell.point_id_id is None or cell.y_id is None:
        raise SystemExit(f"{a.tag}: the cell lacks point ids or y_id; rebuild it with zoo.py")
    Z_id, _ = op.correctness(cell.S_id, cell.y_id)
    features = numeric_features(a.features)
    cols = ["point_id", "label", "domain", "tile_50km", "lon", "lat", "county_fips",
            "meta_point_county", LANDCOVER, "hand", "meta_dist_to_river_m"] + features
    data = pd.read_parquet(a.dataset, columns=list(dict.fromkeys(cols))).set_index("point_id")
    ood = data.loc[cell.point_id].reset_index()
    idd = data.loc[cell.point_id_id].reset_index()
    for half, y in ((ood, cell.y_ood), (idd, cell.y_id)):
        if not np.array_equal(half.label.to_numpy(), np.asarray(y)):
            raise SystemExit("the dataset's labels disagree with the cell's")

    mean, sd = standardizer(ood, features)
    strong = strongest_test_models(cell.models)
    complete = idd[features].notna().all(axis=1).to_numpy()
    missing = {f: int(idd[f].isna().sum()) for f in features if idd[f].isna().any()}
    if ood[features].isna().any().any():
        raise SystemExit("Zone A test points with a missing feature: they cannot be matched")
    tiles = sorted(set(ood.tile_50km) | set(idd.tile_50km))
    code = {t: i for i, t in enumerate(tiles)}
    keep = np.flatnonzero(complete)
    pool_df = idd.iloc[keep].reset_index(drop=True)
    pool = Pool(((pool_df[features] - mean) / sd).to_numpy(), pool_df[LANDCOVER].to_numpy(),
                pool_df.tile_50km.map(code).to_numpy(), pool_df.label.to_numpy())
    return dict(cell=cell, Z_ood=Z_ood, acc=acc, features=features, mean=mean, sd=sd,
                ood=ood, ood_z=((ood[features] - mean) / sd).to_numpy(),
                ood_tile=ood.tile_50km.map(code).to_numpy(), n_tiles=len(tiles),
                ood_acc15=Z_ood[strong].mean(0), pool=pool, pool_df=pool_df,
                acc15=Z_id[strong][:, keep].mean(0), Z15_pool=Z_id[strong][:, keep],
                hand=pool_df.hand.to_numpy(float),
                river_m=pool_df.meta_dist_to_river_m.to_numpy(float), strong=strong,
                n_id=len(idd), n_dropped=int((~complete).sum()), missing=missing)


def queries(ctx: dict, rows: np.ndarray) -> dict:
    """The query arrays of the Zone A test points at the given OOD columns."""
    ood = ctx["ood"].iloc[rows]
    y = ood.label.to_numpy()
    wrong = ctx["Z_ood"][ctx["strong"]][:, rows] == 0
    return dict(z=ctx["ood_z"][rows], y=y, landcover=ood[LANDCOVER].to_numpy(),
                acc15=ctx["ood_acc15"][rows], hand=ood.hand.to_numpy(float),
                river_m=ood.meta_dist_to_river_m.to_numpy(float), tile=ctx["ood_tile"][rows],
                fn15=int((wrong & (y == 1)).sum()), fp15=int((wrong & (y == 0)).sum()))


def pool_queries(ctx: dict) -> dict:
    """The pool itself as queries, for the caliper and the baseline."""
    pool = ctx["pool"]
    wrong = ctx["Z15_pool"] == 0
    return dict(z=pool.z, y=pool.y, landcover=pool.landcover, acc15=ctx["acc15"],
                hand=ctx["hand"], river_m=ctx["river_m"], tile=pool.tile_code,
                fn15=int((wrong & (pool.y == 1)).sum()), fp15=int((wrong & (pool.y == 0)).sum()))


def selection_rows(ctx: dict, points: pd.DataFrame) -> np.ndarray:
    """OOD column of each point of a regions.py points table, in its order."""
    where = pd.Series(np.arange(len(ctx["ood"])), index=ctx["ood"].point_id)
    missing = ~points.point_id.isin(where.index)
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} selected points are not in the cell")
    return where.loc[points.point_id].to_numpy()


def regions_of(ctx: dict, points: pd.DataFrame, min_size: int, names: dict) -> list[dict]:
    """Each region (component of >= min_size points): its name, OOD columns and tile codes."""
    out = []
    for comp, g in points[points.component_size >= min_size].groupby("component", sort=True):
        rows = selection_rows(ctx, g)
        out.append(dict(component=int(comp), name=region_name(g, names), rows=rows,
                        tiles=np.unique(ctx["ood_tile"][rows]),
                        tile_names=" ".join(sorted(set(g.tile_50km.astype(str))))))
    return out


def region_exclusion(ctx: dict, region: dict, rule: str) -> np.ndarray:
    """excluded matrix for a region's points under --exclude."""
    tile = ctx["ood_tile"][region["rows"]]
    excluded = own_tile_exclusion(tile, ctx["n_tiles"])
    if rule == "region-tiles":
        excluded[:, region["tiles"]] = True
    return excluded


def hardest_rows(ctx: dict, n: int = N_HARDEST, seed: int = 0) -> np.ndarray:
    """The n Zone A points the 60 test models get wrong most often; ties at the cut in a seeded
    random order (report_paper.py averages over random orders instead)."""
    _, _, test = op.model_split(ctx["Z_ood"].shape[0])
    score = 1.0 - ctx["Z_ood"][test].mean(0)
    order = np.lexsort((np.random.default_rng(seed).permutation(len(score)), -score))
    return np.sort(order[:n])


def halo_sets(ctx: dict, regions: list[dict], selected: np.ndarray, rule: str):
    """(radius, OOD columns, excluded matrix) of the Zone A points outside the selection within
    each HALO_KM of a region; each excludes its own tile and, under region-tiles, the tiles of
    every region it is near."""
    xy = graph.project(ctx["ood"].lon.to_numpy(), ctx["ood"].lat.to_numpy())
    outside = np.flatnonzero(~selected)
    near = {}
    for region in regions:
        d, _ = cKDTree(xy[region["rows"]]).query(xy[outside], k=1)
        near[region["component"]] = d
    for km in HALO_KM:
        rows, excluded = [], []
        for r, col in enumerate(outside):
            close = [reg for reg in regions if near[reg["component"]][r] <= km * 1000]
            if not close:
                continue
            ex = np.zeros(ctx["n_tiles"], bool)
            ex[ctx["ood_tile"][col]] = True
            if rule == "region-tiles":
                for reg in close:
                    ex[reg["tiles"]] = True
            rows.append(col)
            excluded.append(ex)
        yield km, np.array(rows, int), np.array(excluded, bool).reshape(len(rows), ctx["n_tiles"])


def kinds_table(ctx: dict, regions: list[dict], idx_by_comp: dict) -> pd.DataFrame:
    """Positive and negative regions (majority label): the strongest models' FN and FP shares
    of their errors, corr_te over the 60 test models and over those below INTERPOLATES training
    accuracy, and the pooled unmatched share and eta."""
    _, _, test = op.model_split(ctx["Z_ood"].shape[0])
    models = ctx["cell"].models
    held_out = test[models.acc_train_sub.to_numpy()[test] < INTERPOLATES]
    y = ctx["ood"].label.to_numpy()
    rows = []
    for kind in ("positive", "negative"):
        members = [r for r in regions if (y[r["rows"]].mean() >= 0.5) == (kind == "positive")]
        if not members:
            continue
        cols = np.concatenate([r["rows"] for r in members])
        idx = np.concatenate([idx_by_comp[r["component"]] for r in members])
        mask = np.zeros(len(y), bool)
        mask[cols] = True
        wrong = ctx["Z_ood"][ctx["strong"]][:, cols] == 0
        fn = int((wrong & (y[cols] == 1)).sum())
        fp = int((wrong & (y[cols] == 0)).sum())
        S = len(cols)
        rows.append(dict(
            kind=kind, regions=len(members), points=S,
            names=", ".join(r["name"] for r in members),
            pos_points=float(y[cols].mean()),
            unmatched=float((idx[:, 0] < 0).mean()), eta=eta(y[cols], idx, ctx["pool"].y),
            pos_matches=float(np.nanmean(match_means(ctx["pool"].y.astype(float), idx))),
            errors15=fn + fp, fn_share=fn / max(fn + fp, 1), fp_share=fp / max(fn + fp, 1),
            corr_te=op.hard_corr(ctx["Z_ood"][test], ctx["acc"][test], mask.astype(np.float32),
                                 S, is_mask=True),
            n_test=len(test),
            corr_te_held_out=op.hard_corr(ctx["Z_ood"][held_out], ctx["acc"][held_out],
                                          mask.astype(np.float32), S, is_mask=True),
            n_held_out=len(held_out)))
    return pd.DataFrame(rows)


def run(a) -> dict:
    """Every set's row, the kinds table, the sensitivity table and the summary."""
    ctx = load_inputs(a)
    pool, n_tiles = ctx["pool"], ctx["n_tiles"]
    names = county_names(Path(a.counties))
    pq = pool_queries(ctx)
    own = own_tile_exclusion(pool.tile_code, n_tiles)
    self_index = np.arange(len(pool))
    nearest = match(pool, pool.z, pool.landcover, own, self_index, k=1)[1][:, 0]
    eps = caliper(nearest, a.caliper_pct)
    k = a.max_matches
    print(f"pool: {len(pool):,} Zone AE test points ({ctx['n_dropped']:,} of {ctx['n_id']:,} "
          f"dropped for a missing feature: {ctx['missing']}); "
          f"{int((~np.isfinite(nearest)).sum())} with no candidate in their class")
    print(f"caliper eps = {eps:.4f} standardized units ({a.caliper_pct:g}th percentile of the "
          f"nearest-candidate distance); exclusion rule {a.exclude}; at most {k} matches")

    rows = []
    idx_b, dist_b = match(pool, pool.z, pool.landcover, own, self_index, k=k, eps=eps)
    rows.append(set_row("Zone AE baseline", "baseline", pq, idx_b, dist_b, ctx))

    points = pd.read_parquet(a.regions)
    selected = np.zeros(len(ctx["ood"]), bool)
    selected[selection_rows(ctx, points)] = True
    regions = regions_of(ctx, points, a.min_size, names)
    idx_by_comp, dist_by_comp = {}, {}
    region_rows = []
    for region in regions:
        q = queries(ctx, region["rows"])
        idx, dist = match(pool, q["z"], q["landcover"], region_exclusion(ctx, region, a.exclude),
                          k=k, eps=eps)
        idx_by_comp[region["component"]], dist_by_comp[region["component"]] = idx, dist
        row = set_row(region["name"], "region", q, idx, dist, ctx)
        row.update(component=region["component"], tiles=region["tile_names"])
        region_rows.append(row)
    if regions:
        # Each region keeps its own exclusions; the pooled row only stacks the matches.
        q = queries(ctx, np.concatenate([r["rows"] for r in regions]))
        idx = np.concatenate([idx_by_comp[r["component"]] for r in regions])
        dist = np.concatenate([dist_by_comp[r["component"]] for r in regions])
        rows.append(set_row(f"All regions ({len(regions)})", "pooled", q, idx, dist, ctx))
    rows += region_rows

    references = [("Other Zone A points", np.flatnonzero(~selected)),
                  (f"{N_HARDEST} hardest points", hardest_rows(ctx))]
    if a.classic:
        classic = pd.read_parquet(a.classic)
        references.append((f"OODSelect S = {len(classic)}", selection_rows(ctx, classic)))
    for name, cols in references:
        q = queries(ctx, cols)
        idx, dist = match(pool, q["z"], q["landcover"], own_tile_exclusion(q["tile"], n_tiles),
                          k=k, eps=eps)
        rows.append(set_row(name, "reference", q, idx, dist, ctx))

    for km, cols, excluded in halo_sets(ctx, regions, selected, a.exclude):
        q = queries(ctx, cols)
        idx, dist = match(pool, q["z"], q["landcover"], excluded, k=k, eps=eps)
        rows.append(set_row(f"Halo {km:g} km", "halo", q, idx, dist, ctx))

    for path in a.extra_regions or []:
        extra = pd.read_parquet(path)
        label = str(extra.variant.iloc[0]) if "variant" in extra else Path(path).stem
        for region in regions_of(ctx, extra, a.min_size, names):
            q = queries(ctx, region["rows"])
            idx, dist = match(pool, q["z"], q["landcover"],
                              region_exclusion(ctx, region, a.exclude), k=k, eps=eps)
            row = set_row(f"{label}: {region['name']}", "extra_region", q, idx, dist, ctx)
            row.update(component=region["component"], tiles=region["tile_names"])
            rows.append(row)

    table = pd.DataFrame(rows)
    kinds = kinds_table(ctx, regions, idx_by_comp)
    sensitivity = sensitivity_table(ctx, a, regions, selected, nearest)
    summary = dict(cells_dir=str(a.cells_dir), tag=a.tag, regions=str(a.regions),
                   exclude=a.exclude, eps=eps, caliper_pct=a.caliper_pct, max_matches=k,
                   features=ctx["features"], pool=len(pool), id_points=ctx["n_id"],
                   pool_dropped=ctx["n_dropped"], pool_missing=ctx["missing"],
                   pool_no_candidate=int((~np.isfinite(nearest)).sum()),
                   baseline_unmatched=float(rows[0]["unmatched"]), baseline_eta=rows[0]["eta"],
                   strongest=ctx["strong"].tolist(),
                   standardization=dict(mean=ctx["mean"].round(6).to_dict(),
                                        sd=ctx["sd"].round(6).to_dict()))
    return dict(table=table, kinds=kinds, sensitivity=sensitivity, summary=summary)


def sensitivity_table(ctx: dict, a, regions: list[dict], selected: np.ndarray,
                      nearest: np.ndarray) -> pd.DataFrame:
    """The baseline, the pooled regions and the other Zone A points under each SENSITIVITY
    setting (the other settings at their command-line values)."""
    pool, n_tiles = ctx["pool"], ctx["n_tiles"]
    own = own_tile_exclusion(pool.tile_code, n_tiles)
    out = []
    for setting, value in [("as run", None)] + SENSITIVITY:
        pct = value if setting == "caliper_pct" else a.caliper_pct
        k = int(value) if setting == "max_matches" else a.max_matches
        eps = caliper(nearest, pct)
        row = dict(setting=setting, value=value, caliper_pct=pct, max_matches=k, eps=eps)
        idx, _ = match(pool, pool.z, pool.landcover, own, np.arange(len(pool)), k=k, eps=eps)
        row.update(baseline_unmatched=float((idx[:, 0] < 0).mean()),
                   baseline_eta=eta(pool.y, idx, pool.y))
        if regions:
            cols = np.concatenate([r["rows"] for r in regions])
            excluded = np.concatenate([region_exclusion(ctx, r, a.exclude) for r in regions])
            q = queries(ctx, cols)
            idx, _ = match(pool, q["z"], q["landcover"], excluded, k=k, eps=eps)
            row.update(regions_unmatched=float((idx[:, 0] < 0).mean()),
                       regions_eta=eta(q["y"], idx, pool.y))
        others = np.flatnonzero(~selected)
        q = queries(ctx, others)
        idx, _ = match(pool, q["z"], q["landcover"], own_tile_exclusion(q["tile"], n_tiles),
                       k=k, eps=eps)
        row.update(others_unmatched=float((idx[:, 0] < 0).mean()),
                   others_eta=eta(q["y"], idx, pool.y))
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--tag", required=True, help="cell tag, e.g. FL_band_AE_s0")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features", required=True, help="FEATURES.txt of the cell's models")
    ap.add_argument("--regions", required=True,
                    help="regions.py points table of the selection whose regions are analyzed")
    ap.add_argument("--classic", help="regions.py points table of the OODSelect selection")
    ap.add_argument("--extra-regions", nargs="*",
                    help="points tables of other selections whose regions are reported too")
    ap.add_argument("--exclude", choices=EXCLUDE_RULES, default="region-tiles",
                    help="candidates left out for a region's point: every tile of the region "
                         "(the paper) or only the point's own tile (the sandbox reference)")
    ap.add_argument("--min-size", type=int, default=MIN_REGION)
    ap.add_argument("--max-matches", type=int, default=MAX_MATCHES)
    ap.add_argument("--caliper-pct", type=float, default=CALIPER_PCT)
    ap.add_argument("--counties", default=str(COUNTIES))
    ap.add_argument("--out", required=True, help="output folder")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    out = run(a)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    out["table"].to_csv(folder / "region_match.csv", index=False)
    out["kinds"].to_csv(folder / "region_match_kinds.csv", index=False)
    out["sensitivity"].to_csv(folder / "region_match_sensitivity.csv", index=False)
    (folder / "region_match.json").write_text(json.dumps(out["summary"], indent=2))
    show = ["set", "n", "n_matched", "unmatched", "eta", "pos_points", "pos_matches",
            "dist_median", "smd_max", "smd_feature", "acc15_points", "acc15_matches"]
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_colwidth", 40):
        print(out["table"][[c for c in show if c in out["table"]]].round(3).to_string(index=False))
        print("\nkinds:")
        print(out["kinds"].round(3).to_string(index=False))
        print("\nsensitivity:")
        print(out["sensitivity"].round(3).to_string(index=False))
    print(f"\nwrote {folder}/region_match.csv, region_match_kinds.csv, "
          f"region_match_sensitivity.csv, region_match.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
