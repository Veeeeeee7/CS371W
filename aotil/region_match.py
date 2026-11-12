#!/usr/bin/env python3
"""
The region analysis (the paper's Region Analysis and Table tab:regions): caliper matching of each region's points
to held-out Zone AE points.

Each point of a region is matched to Zone AE test points with the same land cover and nearly the same numeric
inputs. A point with no match has inputs that Zone AE does not cover (covariate shift); a point whose matches
carry a different label has the same inputs labeled differently (concept shift). In CS terms: a nearest-neighbour
lookup in the standardized feature space, restricted to one land-cover class and to tiles the region does not
touch, with a maximum distance (the caliper).

Definitions:
  z(x)      the q numeric features (every feature of FEATURES.txt except the categorical ones of
            zoo.CATEGORICAL: land cover, matched exactly, and the two soil classes, not matched; q = 6), each
            standardized by its mean and standard deviation (ddof 1, as pandas computes it, skipping missing
            values) over D_OOD, all Zone A test points.
  pool      the Zone AE test points (D_ID) with all q features present; the others are dropped and counted
            (the paper's "75 Zone AE test points that lack a feature"). A Zone A point with a missing feature
            has no z either: it is left out of every matched set and counted (n_missing).
  P_C       candidates for region C: the pool points in test tiles (tile_50km) that hold no point of C.
  C'_x      the candidates with x's land-cover class at Euclidean distance <= eps in z, the 5 closest; x is
            matched when C'_x is not empty.
  eps       the caliper: every pool point is matched against the pool as a one-point region (its own tile and
            itself excluded); eps is the 95th percentile (numpy's linear rule) of the distance to the nearest
            candidate, so about 5% of Zone AE points are unmatched (Stuart 2010).
  eta       Eq. disagree: over the matched points of a set, the mean of the share of each point's matches whose
            label differs from its own (each matched point counts once).
Rows (Table tab:regions): each region; all regions pooled (each region keeps its own exclusions; the pooled row
stacks the matches); Other Zone A (the Zone A test points outside the subset, each matched as a one-point region,
excluding its own tile); the Zone AE baseline (the pool against itself, as for the caliper).

Nearest neighbours: one KD-tree per land-cover class. A query asks for QUERY_START candidates, drops the
ineligible ones (excluded tiles, the point itself) and asks again for QUERY_GROWTH times as many until it has
enough eligible candidates within eps, has seen every candidate within eps, or has exhausted the class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zoo import CATEGORICAL, read_features  # noqa: E402

LANDCOVER = "landcover"      # matched exactly, so not one of the numeric features
MAX_MATCHES = 5
CALIPER_PCT = 95.0           # 5% of Zone AE points unmatched, after Stuart (2010)
QUERY_START = 64
QUERY_GROWTH = 4


# ------------------------------------------------------------------ inputs
def numeric_features(path: str | Path) -> list[str]:
    """The numeric features of FEATURES.txt, in file order (q of them): every feature except the categorical
    ones (zoo.CATEGORICAL). Land cover is matched exactly instead; the soil classes are not matched."""
    names = read_features(path)
    numeric = [n for n in names if n not in CATEGORICAL]
    if LANDCOVER not in names or not numeric or len(set(names)) != len(names):
        raise SystemExit(f"{path}: expected landcover, at least one numeric feature and no "
                         f"repeats, got {names}")
    return numeric


def standardizer(ood: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """Mean and standard deviation (ddof 1) of each feature over D_OOD."""
    return ood[cols].mean(), ood[cols].std()


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
    v = np.where(valid, values[np.maximum(idx, 0)], 0.0)
    n = valid.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v.sum(1) / n


# ------------------------------------------------------------------ one set
def set_row(name: str, kind: str, y: np.ndarray, idx: np.ndarray, dist: np.ndarray, pool: Pool,
            n_missing: int = 0) -> dict:
    """The output row of one matched set: n (points with every feature), n_matched, unmatched, eta, the flood
    shares of the points (pos_points), of the matched points (pos_matched) and of their matches (pos_matches,
    each point's matches averaged first), the median match distance, and the points left out (n_missing)."""
    matched = idx[:, 0] >= 0
    n, n_matched = len(idx), int(matched.sum())
    row = dict(set=name, kind=kind, n=n, n_matched=n_matched,
               unmatched=float(1 - n_matched / n) if n else np.nan,
               eta=eta(y, idx, pool.y),
               pos_points=float(np.mean(y)) if n else np.nan,
               pos_matched=float(np.mean(y[matched])) if n_matched else np.nan,
               pos_matches=np.nan, dist_median=np.nan, n_missing=int(n_missing))
    if n_matched:
        m_idx, m_dist = idx[matched], dist[matched]
        row["pos_matches"] = float(match_means(pool.y.astype(float), m_idx).mean())
        row["dist_median"] = float(np.median(m_dist[m_idx >= 0]))
    return row


# ------------------------------------------------------------------ the analysis
def prepare(ood: pd.DataFrame, idd: pd.DataFrame, features: list[str]) -> dict:
    """The pool and the standardized Zone A points. ood / idd: the dataset rows of the OOD and ID test points in
    the cell's order (regions.ood_rows), with point_id, label, tile_50km, landcover and the numeric features."""
    mean, sd = standardizer(ood, features)
    complete = idd[features].notna().all(axis=1).to_numpy()
    ood_complete = ood[features].notna().all(axis=1).to_numpy()
    tiles = sorted(set(ood.tile_50km.astype(str)) | set(idd.tile_50km.astype(str)))
    code = {t: i for i, t in enumerate(tiles)}
    keep = np.flatnonzero(complete)
    pool_df = idd.iloc[keep].reset_index(drop=True)
    pool = Pool(((pool_df[features] - mean) / sd).to_numpy(), pool_df[LANDCOVER].to_numpy(),
                pool_df.tile_50km.astype(str).map(code).to_numpy(), pool_df.label.to_numpy())
    return dict(features=features, mean=mean, sd=sd, ood=ood.reset_index(drop=True),
                ood_z=((ood[features] - mean) / sd).to_numpy(),
                ood_tile=ood.tile_50km.astype(str).map(code).to_numpy(), n_tiles=len(tiles),
                ood_complete=ood_complete, pool=pool, pool_df=pool_df, n_id=len(idd),
                pool_dropped=int((~complete).sum()),
                pool_missing={f: int(idd[f].isna().sum()) for f in features if idd[f].isna().any()})


def run(ctx: dict, selected: np.ndarray, components: dict, k: int = MAX_MATCHES,
        pct: float = CALIPER_PCT) -> dict:
    """Table tab:regions. selected: the subset (bool over the OOD points); components: component id -> the OOD
    columns of its points (the regions, >= c points each). Returns dict(table=DataFrame of the rows,
    pairs=DataFrame of every region point's matches, point_summary=DataFrame per region point, summary=dict)."""
    pool, n_tiles = ctx["pool"], ctx["n_tiles"]
    own = own_tile_exclusion(pool.tile_code, n_tiles)
    self_index = np.arange(len(pool))
    nearest = match(pool, pool.z, pool.landcover, own, self_index, k=1)[1][:, 0]
    eps = caliper(nearest, pct)

    rows = []
    idx_b, dist_b = match(pool, pool.z, pool.landcover, own, self_index, k=k, eps=eps)
    baseline = set_row("Zone AE baseline", "baseline", pool.y, idx_b, dist_b, pool)

    y_ood = ctx["ood"].label.to_numpy()
    region_rows, stacked, pairs, per_point = [], [], [], []
    for cid, cols in components.items():
        cols = np.asarray(cols, int)
        tiles = np.unique(ctx["ood_tile"][cols])
        q = cols[ctx["ood_complete"][cols]]
        excluded = own_tile_exclusion(ctx["ood_tile"][q], n_tiles)
        excluded[:, tiles] = True
        idx, dist = match(pool, ctx["ood_z"][q], ctx["ood"][LANDCOVER].to_numpy()[q], excluded, k=k, eps=eps)
        row = set_row(str(cid), "region", y_ood[q], idx, dist, pool, n_missing=len(cols) - len(q))
        row["component"] = int(cid)
        region_rows.append(row)
        stacked.append((q, idx, dist, len(cols) - len(q)))
        share = disagreement(y_ood[q], idx, pool.y)
        for r, col in enumerate(q):
            per_point.append(dict(component=int(cid), point_id=ctx["ood"].point_id.iloc[col],
                                  n_matches=int((idx[r] >= 0).sum()), match_label_disagreement=share[r]))
            for rank in range(k):
                j = idx[r, rank]
                if j < 0:
                    break
                pairs.append(dict(component=int(cid), point_id=ctx["ood"].point_id.iloc[col], rank=rank + 1,
                                  match_point_id=ctx["pool_df"].point_id.iloc[j], distance=float(dist[r, rank]),
                                  label=int(y_ood[col]), match_label=int(pool.y[j])))
        for col in cols[~ctx["ood_complete"][cols]]:
            per_point.append(dict(component=int(cid), point_id=ctx["ood"].point_id.iloc[col], n_matches=0,
                                  match_label_disagreement=np.nan, missing_feature=True))
    if stacked:
        q = np.concatenate([s[0] for s in stacked])
        idx = np.concatenate([s[1] for s in stacked])
        dist = np.concatenate([s[2] for s in stacked])
        rows.append(set_row("All regions", "pooled", y_ood[q], idx, dist, pool,
                            n_missing=sum(s[3] for s in stacked)))
    rows = region_rows + rows

    others = np.flatnonzero(~np.asarray(selected, bool))
    q = others[ctx["ood_complete"][others]]
    idx, dist = match(pool, ctx["ood_z"][q], ctx["ood"][LANDCOVER].to_numpy()[q],
                      own_tile_exclusion(ctx["ood_tile"][q], n_tiles), k=k, eps=eps)
    rows.append(set_row("Other Zone A", "other", y_ood[q], idx, dist, pool, n_missing=len(others) - len(q)))
    rows.append(baseline)

    summary = dict(eps=eps, caliper_pct=pct, max_matches=k, features=ctx["features"], q=len(ctx["features"]),
                   pool=len(pool), id_points=ctx["n_id"], pool_dropped=ctx["pool_dropped"],
                   pool_missing=ctx["pool_missing"], pool_no_candidate=int((~np.isfinite(nearest)).sum()),
                   ood_points=len(ctx["ood"]), ood_dropped=int((~ctx["ood_complete"]).sum()),
                   standardization=dict(mean=ctx["mean"].round(6).to_dict(), sd=ctx["sd"].round(6).to_dict()))
    point_summary = pd.DataFrame(per_point)
    if "missing_feature" not in point_summary:
        point_summary["missing_feature"] = False
    point_summary["missing_feature"] = point_summary.missing_feature.fillna(False).astype(bool)
    return dict(table=pd.DataFrame(rows), pairs=pd.DataFrame(pairs), point_summary=point_summary,
                summary=summary)
