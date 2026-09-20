#!/usr/bin/env python3
"""
Spatial adjacency, the graph Laplacian, and the coherence metrics every subset
is reported with.

WHY A GRAPH AT ALL
------------------
Unconstrained OODSelect returns subsets that touch 15-22 of 23 evaluation tiles
at a 5-10% budget. A set like that cannot be handed to a GIS analysis, which
reads hydrological and land-use context off a PLACE. Adding a smoothness penalty
over a spatial graph turns the selection into regions, at a measured cost in
inversion strength -- the coherence/inversion tradeoff this project reports.

WHY kNN AND NOT A RADIUS
------------------------
Flood zones are ribbons roughly 110-130 m wide (Vermont's Zone A: 277 km of area
against 5,103 km of perimeter). A Euclidean-radius window centred on a corridor
point is mostly empty space; a kNN ball follows the corridor. The same fact is
why the sampler's separation is 200 m rather than 1 km.

WHY THE PENALTY IS NORMALISED
-----------------------------
P(w) = w' L w / (S_target * k) is the cut-edge count divided by the most edges a
budget-S selection can cut. A scattered selection of S points, none adjacent,
cuts about S*k edges, so P ~ 1; a compact region cuts only its perimeter, so
P << 1. That makes P an incoherence score in [0, ~1] and makes a given mu mean
the same thing in every cell, which a raw w'Lw does not.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Florida Albers (EPSG:3086) is the project's planar CRS for this state. The
# fallback below is an equirectangular approximation, which is accurate enough
# for a kNN ordering over a few hundred km but is NOT a substitute for a real
# projection if a distance is ever reported in metres.
EARTH_R = 6_371_008.8


def project(lon: np.ndarray, lat: np.ndarray, epsg: int | None = 3086) -> np.ndarray:
    """Longitude/latitude -> planar metres.

    kNN in raw degrees is anisotropic: a degree of longitude is ~0.74 of a degree
    of latitude at Florida's latitude, so an unprojected neighbour set is
    stretched east-west and the graph encodes that distortion.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if epsg is not None:
        try:
            from pyproj import Transformer
            tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            x, y = tf.transform(lon, lat)
            return np.column_stack([x, y])
        except Exception:
            pass                      # fall through to the approximation
    lat0 = np.deg2rad(np.nanmean(lat))
    x = EARTH_R * np.deg2rad(lon) * np.cos(lat0)
    y = EARTH_R * np.deg2rad(lat)
    return np.column_stack([x, y])


def _knn_pairs(xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i, j) of each point's k nearest neighbours, self excluded."""
    n = len(xy)
    k_eff = min(k, max(n - 1, 1))
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    keep = rows != cols                                  # drop self-matches
    return rows[keep], cols[keep]


def build_adjacency(lon, lat, k: int = 10, groups: np.ndarray | None = None,
                    epsg: int | None = 3086) -> sp.csr_matrix:
    """Symmetric binary kNN adjacency, optionally restricted within groups.

    `groups` restricts neighbours to points sharing a group value -- the same
    block, catchment or source study. Restriction always means kNN AMONG
    same-group points, never a clique: a 2,000-point county as a clique is two
    million edges that say nothing about locality.
    """
    xy = project(lon, lat, epsg)
    n = len(xy)
    if groups is None:
        rows, cols = _knn_pairs(xy, k)
    else:
        groups = np.asarray(groups)
        r_all, c_all = [], []
        for g in np.unique(groups[~_isnull(groups)]):
            member = np.where(groups == g)[0]
            if len(member) < 2:
                continue                                 # a singleton has no edges
            r, c = _knn_pairs(xy[member], k)
            r_all.append(member[r]); c_all.append(member[c])
        if not r_all:
            return sp.csr_matrix((n, n))
        rows = np.concatenate(r_all); cols = np.concatenate(c_all)

    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)               # symmetrise
    A.setdiag(0); A.eliminate_zeros()
    return A.tocsr()


def _isnull(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind in "OUS":
        return np.array([x is None or (isinstance(x, float) and np.isnan(x))
                         or x == "" or str(x).lower() == "nan" for x in a])
    return ~np.isfinite(a.astype(float, copy=False))


def laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """L = D - A, the combinatorial Laplacian. w'Lw is the cut-edge count."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(deg) - A).tocsr()


def incoherence(w: np.ndarray, L: sp.csr_matrix, s_target: float, k: int) -> float:
    """P(w) = w'Lw / (S*k) -- the normalised penalty, in [0, ~1]."""
    denom = max(s_target * k, 1.0)
    return float(w @ (L @ w) / denom)


# ---------------------------------------------------------------- metrics
def coherence_metrics(mask: np.ndarray, A: sp.csr_matrix, k: int,
                      tiles: np.ndarray | None = None,
                      blocks: np.ndarray | None = None,
                      values: np.ndarray | None = None) -> dict:
    """Every coherence number a subset is reported with.

    A random subset of the same size is the baseline for all of them, which is
    why the random control in the sweep doubles as the coherence null. Expected
    values under that baseline, with f = |M| / d:
        purity ~ f     P_hard ~ 1 - f     largest_cc_frac small     moran_I ~ 0
    """
    mask = np.asarray(mask, bool)
    d = len(mask)
    S = int(mask.sum())
    out = dict(k=S, frac=S / d if d else np.nan)
    if S == 0:
        return {**out, "purity": np.nan, "P_hard": np.nan, "n_cc": 0,
                "largest_cc_frac": np.nan, "tiles": 0, "blocks": 0,
                "moran_I": np.nan}

    deg = np.asarray(A.sum(axis=1)).ravel()
    inside = np.asarray(A[mask][:, mask].sum(axis=1)).ravel()   # neighbours also selected
    out["purity"] = float(np.mean(inside / np.maximum(deg[mask], 1)))

    # Cut edges: every selected point's neighbours that are NOT selected.
    cut = float(deg[mask].sum() - inside.sum())                 # counts each cut edge once
    out["P_hard"] = cut / max(S * k, 1)

    sub = A[mask][:, mask]
    n_cc, lab = connected_components(sub, directed=False)
    out["n_cc"] = int(n_cc)
    out["largest_cc_frac"] = float(np.bincount(lab).max() / S) if S else np.nan

    out["tiles"] = int(len(np.unique(tiles[mask]))) if tiles is not None else -1
    out["blocks"] = int(len(np.unique(blocks[mask]))) if blocks is not None else -1

    x = mask.astype(float) if values is None else np.asarray(values, float)
    out["moran_I"] = morans_i(x, A)
    return out


def morans_i(x: np.ndarray, A: sp.csr_matrix) -> float:
    """Moran's I of x under adjacency weights A. ~0 under spatial randomness."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.nan
    if not ok.all():                       # restrict to the finite subgraph
        A = A[ok][:, ok]; x = x[ok]
    z = x - x.mean()
    denom = float(z @ z)
    W = float(A.sum())
    if denom <= 0 or W <= 0:
        return np.nan
    return float(len(z) / W * (z @ (A @ z)) / denom)


def morans_i_null(x: np.ndarray, A: sp.csr_matrix, n_perm: int = 200,
                  seed: int = 0) -> dict:
    """Moran's I against a permutation null over POINTS.

    Used by the E0 go/no-go: if the per-point covariance map is inside its own
    null, it is spatial white noise and no mu > 0 can produce a region.
    """
    rng = np.random.default_rng(seed)
    obs = morans_i(x, A)
    x = np.asarray(x, float)
    null = np.array([morans_i(rng.permutation(x), A) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return dict(moran_I=obs, null_mean=np.nan, null_sd=np.nan,
                    z=np.nan, p_perm=np.nan, n_perm=0)
    sd = null.std(ddof=1)
    return dict(moran_I=float(obs), null_mean=float(null.mean()), null_sd=float(sd),
                z=float((obs - null.mean()) / sd) if sd > 0 else np.nan,
                # two-sided, +1 for the observation itself
                p_perm=float((np.abs(null - null.mean()) >= abs(obs - null.mean())).sum() + 1)
                       / (len(null) + 1),
                n_perm=int(len(null)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Jaccard index between two boolean masks.

    Divided by the value expected for two random masks of the same sizes, so 1.0
    means "no more overlap than chance" and higher means genuinely consistent.
    The paper uses this to show its subsets are stable across budgets.
    """
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = float((a & b).sum()); union = float((a | b).sum())
    if union == 0:
        return np.nan
    j = inter / union
    d = len(a); na, nb = a.sum(), b.sum()
    exp_inter = na * nb / d
    exp_union = na + nb - exp_inter
    j_rand = exp_inter / exp_union if exp_union > 0 else np.nan
    return float(j / j_rand) if j_rand and np.isfinite(j_rand) and j_rand > 0 else np.nan


def self_test() -> int:
    """Runnable check of the pieces every later stage depends on."""
    ok = fail = 0
    def chk(cond, msg):
        nonlocal ok, fail
        if cond: ok += 1
        else: fail += 1; print(f"  FAIL {msg}")

    rng = np.random.default_rng(0)
    # Two tight, well-separated clusters: coherence metrics have known answers.
    a = rng.normal([-81.0, 27.0], 0.002, size=(200, 2))
    b = rng.normal([-80.0, 28.0], 0.002, size=(200, 2))
    lon = np.r_[a[:, 0], b[:, 0]]; lat = np.r_[a[:, 1], b[:, 1]]
    k = 8
    A = build_adjacency(lon, lat, k=k)
    chk((A != A.T).nnz == 0, "adjacency symmetric")
    chk(A.diagonal().sum() == 0, "no self loops")

    L = laplacian(A)
    chk(abs(L.sum()) < 1e-9, "Laplacian rows sum to zero")
    chk(np.all(np.linalg.eigvalsh(L.toarray()[:50, :50]) > -1e-8), "L PSD (leading block)")

    one_cluster = np.zeros(400, bool); one_cluster[:200] = True
    scattered = np.zeros(400, bool); scattered[rng.choice(400, 200, replace=False)] = True
    mc = coherence_metrics(one_cluster, A, k)
    ms = coherence_metrics(scattered, A, k)
    chk(mc["purity"] > 0.9, f"compact purity high (got {mc['purity']:.3f})")
    chk(ms["purity"] < 0.75, f"scattered purity lower (got {ms['purity']:.3f})")
    chk(mc["P_hard"] < ms["P_hard"], "compact cuts fewer edges")
    chk(mc["n_cc"] <= 2, f"compact few components (got {mc['n_cc']})")
    chk(ms["n_cc"] > mc["n_cc"], "scattered more components")

    chk(morans_i(one_cluster.astype(float), A) > 0.8, "Moran's I high for a cluster")
    nul = morans_i_null(rng.normal(size=400), A, n_perm=100, seed=1)
    chk(abs(nul["z"]) < 3, f"white noise inside its null (z={nul['z']:.2f})")
    nul2 = morans_i_null(one_cluster.astype(float), A, n_perm=100, seed=1)
    chk(nul2["z"] > 5 and nul2["p_perm"] < 0.05, "clustered signal outside its null")

    chk(jaccard(one_cluster, one_cluster) > 1.5, "self-Jaccard above chance")
    r1 = np.zeros(400, bool); r1[rng.choice(400, 100, replace=False)] = True
    r2 = np.zeros(400, bool); r2[rng.choice(400, 100, replace=False)] = True
    chk(0.3 < jaccard(r1, r2) < 3.0, "two random masks near the chance ratio")

    grp = np.r_[np.zeros(200), np.ones(200)]
    Ag = build_adjacency(lon, lat, k=k, groups=grp)
    cross = Ag[:200][:, 200:].nnz
    chk(cross == 0, "grouped adjacency has no cross-group edges")

    # An incoherent selection must score near 1, a compact one well below.
    w_scat = scattered.astype(float); w_comp = one_cluster.astype(float)
    chk(incoherence(w_comp, L, 200, k) < incoherence(w_scat, L, 200, k),
        "incoherence orders compact below scattered")
    print(f"graph.py self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(self_test())
