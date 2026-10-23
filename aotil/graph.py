#!/usr/bin/env python3
"""
Spatial kNN adjacency over OOD points, its graph Laplacian, and the coherence
metrics every selected subset is reported with.

The Laplacian penalty P(w) = w'Lw / (S_target * k) turns an OODSelect subset into
regions. kNN is used rather than a radius because flood zones are narrow ribbons:
a kNN ball follows the corridor, a radius window is mostly empty. Dividing by
S_target * k (the most edges a budget-S selection can cut) puts P in [0, ~1], so a
given mu means the same in every cell. See docs/SPATIAL-OODSELECT.md sections 3.1,
3.3 and 4.

Run `python aotil/graph.py` for a self-test on synthetic clusters (exit 0 = pass).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Mean Earth radius in metres, for the equirectangular fallback in project(). The
# default CRS is Florida Albers (EPSG:3086), the project's planar CRS.
EARTH_R = 6_371_008.8


def project(lon: np.ndarray, lat: np.ndarray, epsg: int | None = 3086) -> np.ndarray:
    """Longitude/latitude in degrees -> (n, 2) planar x, y in metres.

    kNN on raw degrees would stretch neighbourhoods east-west (a degree of longitude
    is ~0.88 of a degree of latitude in Florida). Without pyproj, with an unknown
    CRS, or with epsg=None, an equirectangular approximation is used: fine for kNN
    ordering, not for distances reported in metres.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if epsg is not None:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            x, y = transformer.transform(lon, lat)
            return np.column_stack([x, y])
        except Exception:
            pass                                    # use the approximation below
    lat0 = np.deg2rad(np.nanmean(lat))
    x = EARTH_R * np.deg2rad(lon) * np.cos(lat0)
    y = EARTH_R * np.deg2rad(lat)
    return np.column_stack([x, y])


def _knn_pairs(xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i, j) of each point's k nearest neighbours, self excluded."""
    n = len(xy)
    k_eff = min(k, max(n - 1, 1))                   # at most n - 1 neighbours exist
    # Ask for one extra because the query normally returns the point itself.
    _, idx = cKDTree(xy).query(xy, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    # Drop self-matches by index. When more than k_eff + 1 points share one
    # coordinate, the query may not return the point itself, and then it keeps
    # k_eff + 1 neighbours.
    keep = rows != cols
    return rows[keep], cols[keep]


def _is_null_value(x) -> bool:
    """A missing group value: None, NaN, "" or the string "nan" in any case."""
    return bool(x is None or (isinstance(x, float) and np.isnan(x))
                or x == "" or str(x).lower() == "nan")


def _isnull(a: np.ndarray) -> np.ndarray:
    """Boolean mask of missing group values (NaN or +-inf in a numeric array)."""
    if a.dtype.kind in "OUS":
        return np.array([_is_null_value(x) for x in a])
    return ~np.isfinite(a.astype(float, copy=False))


def build_adjacency(lon, lat, k: int = 10, groups: np.ndarray | None = None,
                    epsg: int | None = 3086) -> sp.csr_matrix:
    """Symmetric 0/1 kNN adjacency over the projected points (CSR, float64).

    With `groups` (block, catchment or source study), each point's neighbours are its
    k nearest among points of the same group, never the whole group as a clique.
    Points with a missing group value, and singleton groups, get no edges.

    The directed kNN graph (row i = point i's own neighbours) is attached as
    `A.knn_directed`; coherence_metrics uses it for purity.
    """
    xy = project(lon, lat, epsg)
    n = len(xy)
    if groups is None:
        rows, cols = _knn_pairs(xy, k)
    else:
        groups = np.asarray(groups)
        row_parts = []
        col_parts = []
        for g in np.unique(groups[~_isnull(groups)]):
            members = np.where(groups == g)[0]
            if len(members) < 2:
                continue                            # a singleton has no neighbours
            r, c = _knn_pairs(xy[members], k)
            row_parts.append(members[r])
            col_parts.append(members[c])
        if not row_parts:
            return sp.csr_matrix((n, n))
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)

    directed = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    # i and j are adjacent when either is among the other's k nearest.
    A = ((directed + directed.T) > 0).astype(np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocsr()
    A.knn_directed = directed
    return A


def _row_sums(M: sp.spmatrix) -> np.ndarray:
    """Row sums of a sparse matrix as a flat array."""
    return np.asarray(M.sum(axis=1)).ravel()


def laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """L = D - A, the combinatorial Laplacian; w'Lw is the cut-edge count of a 0/1 w."""
    degree = _row_sums(A)
    return (sp.diags(degree) - A).tocsr()


def incoherence(w: np.ndarray, L: sp.csr_matrix, s_target: float, k: int) -> float:
    """P(w) = w'Lw / (S*k), the normalised penalty, in [0, ~1]."""
    denom = max(s_target * k, 1.0)
    return float(w @ (L @ w) / denom)


def _n_distinct(labels: np.ndarray | None, mask: np.ndarray) -> int:
    """Distinct labels among the selected points, or -1 when there are no labels."""
    if labels is None:
        return -1
    return int(len(np.unique(labels[mask])))


def coherence_metrics(mask: np.ndarray, A: sp.csr_matrix, k: int,
                      tiles: np.ndarray | None = None,
                      blocks: np.ndarray | None = None,
                      values: np.ndarray | None = None) -> dict:
    """Every coherence number a subset is reported with.

    Output key "k" is the subset size S. A random subset of the same size is the
    baseline; with f = S / d it gives purity ~ f, P_hard a little above 1 - f (the
    symmetrised degrees exceed k), moran_I ~ 0 and, at small f, a small
    largest_cc_frac. purity is the paper's Eq. (purity): the mean over selected
    points of the share of each point's own nearest neighbours (A.knn_directed; the
    rows of A when A lacks it) that are also selected. The share is over the
    neighbours found, which is k unless a group has k or fewer points or more than
    k + 1 points share one coordinate.
    """
    mask = np.asarray(mask, bool)
    d = len(mask)
    S = int(mask.sum())
    out = dict(k=S, frac=S / d if d else np.nan)
    if S == 0:
        return {**out, "purity": np.nan, "P_hard": np.nan, "n_cc": 0,
                "largest_cc_frac": np.nan, "tiles": 0, "blocks": 0,
                "moran_I": np.nan}

    knn_selected = getattr(A, "knn_directed", A)[mask]
    n_neighbours = _row_sums(knn_selected)
    n_selected_neighbours = _row_sums(knn_selected[:, mask])
    out["purity"] = float(np.mean(n_selected_neighbours / np.maximum(n_neighbours, 1)))

    # The degrees of the selected points count each cut edge once and each inside
    # edge twice; the induced subgraph's degrees count each inside edge twice.
    induced = A[mask][:, mask]
    degree = _row_sums(A)
    induced_degree = _row_sums(induced)
    cut = float(degree[mask].sum() - induced_degree.sum())
    out["P_hard"] = cut / max(S * k, 1)

    n_cc, component = connected_components(induced, directed=False)
    out["n_cc"] = int(n_cc)
    out["largest_cc_frac"] = float(np.bincount(component).max() / S)

    out["tiles"] = _n_distinct(tiles, mask)
    out["blocks"] = _n_distinct(blocks, mask)

    x = mask.astype(float) if values is None else np.asarray(values, float)
    out["moran_I"] = morans_i(x, A)
    return out


def morans_i(x: np.ndarray, A: sp.csr_matrix) -> float:
    """Moran's I of x under adjacency weights A; ~0 under spatial randomness.

    Non-finite values are dropped together with their rows and columns of A.
    """
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return np.nan
    if not finite.all():
        A = A[finite][:, finite]
        x = x[finite]
    z = x - x.mean()
    denom = float(z @ z)
    W = float(A.sum())
    if denom <= 0 or W <= 0:
        return np.nan
    return float(len(z) / W * (z @ (A @ z)) / denom)


def morans_i_null(x: np.ndarray, A: sp.csr_matrix, n_perm: int = 200,
                  seed: int = 0) -> dict:
    """Moran's I against a null of n_perm random permutations of x over the points.

    A map inside its own null is spatial white noise: no penalty weight can turn it
    into a region.
    """
    rng = np.random.default_rng(seed)
    obs = morans_i(x, A)
    x = np.asarray(x, float)
    null = []
    for _ in range(n_perm):
        null.append(morans_i(rng.permutation(x), A))
    null = np.array(null)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return dict(moran_I=obs, null_mean=np.nan, null_sd=np.nan,
                    z=np.nan, p_perm=np.nan, n_perm=0)

    null_mean = null.mean()
    null_sd = null.std(ddof=1)
    z = float((obs - null_mean) / null_sd) if null_sd > 0 else np.nan
    # Two-sided; the +1s count the observation as one of the permutations.
    n_as_extreme = (np.abs(null - null_mean) >= abs(obs - null_mean)).sum()
    p_perm = float(n_as_extreme + 1) / (len(null) + 1)
    return dict(moran_I=float(obs), null_mean=float(null_mean), null_sd=float(null_sd),
                z=z, p_perm=p_perm, n_perm=int(len(null)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index of two boolean masks, divided by its value for random masks.

    The random masks have the same sizes, so 1.0 means no more overlap than chance.
    NaN when both masks are empty or one of them is.
    """
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    intersection = float((a & b).sum())
    union = float((a | b).sum())
    if union == 0:
        return np.nan
    observed = intersection / union

    n_a = a.sum()
    n_b = b.sum()
    expected_intersection = n_a * n_b / len(a)
    expected_union = n_a + n_b - expected_intersection     # > 0 because union > 0
    expected = expected_intersection / expected_union
    if expected <= 0:
        return np.nan
    return float(observed / expected)


def _check(results: list, passed, message: str) -> None:
    """Record one self-test check and print it if it failed."""
    results.append(bool(passed))
    if not passed:
        print(f"  FAIL {message}")


def self_test() -> int:
    """Check the adjacency, Laplacian and metrics on two synthetic clusters."""
    results = []
    rng = np.random.default_rng(0)
    # Two tight, well-separated clusters, so the coherence metrics have known answers.
    a = rng.normal([-81.0, 27.0], 0.002, size=(200, 2))
    b = rng.normal([-80.0, 28.0], 0.002, size=(200, 2))
    lon = np.r_[a[:, 0], b[:, 0]]
    lat = np.r_[a[:, 1], b[:, 1]]
    k = 8
    A = build_adjacency(lon, lat, k=k)
    _check(results, (A != A.T).nnz == 0, "adjacency symmetric")
    _check(results, A.diagonal().sum() == 0, "no self loops")

    L = laplacian(A)
    _check(results, abs(L.sum()) < 1e-9, "Laplacian rows sum to zero")
    _check(results, np.all(np.linalg.eigvalsh(L.toarray()[:50, :50]) > -1e-8),
           "L PSD (leading block)")

    one_cluster = np.zeros(400, bool)
    one_cluster[:200] = True
    scattered = np.zeros(400, bool)
    scattered[rng.choice(400, 200, replace=False)] = True
    mc = coherence_metrics(one_cluster, A, k)
    ms = coherence_metrics(scattered, A, k)
    _check(results, mc["purity"] > 0.9, f"compact purity high (got {mc['purity']:.3f})")
    _check(results, ms["purity"] < 0.75, f"scattered purity lower (got {ms['purity']:.3f})")
    _check(results, mc["P_hard"] < ms["P_hard"], "compact cuts fewer edges")
    _check(results, mc["n_cc"] <= 2, f"compact few components (got {mc['n_cc']})")
    _check(results, ms["n_cc"] > mc["n_cc"], "scattered more components")

    _check(results, morans_i(one_cluster.astype(float), A) > 0.8,
           "Moran's I high for a cluster")
    noise_null = morans_i_null(rng.normal(size=400), A, n_perm=100, seed=1)
    _check(results, abs(noise_null["z"]) < 3,
           f"white noise inside its null (z={noise_null['z']:.2f})")
    cluster_null = morans_i_null(one_cluster.astype(float), A, n_perm=100, seed=1)
    _check(results, cluster_null["z"] > 5 and cluster_null["p_perm"] < 0.05,
           "clustered signal outside its null")

    _check(results, jaccard(one_cluster, one_cluster) > 1.5, "self-Jaccard above chance")
    r1 = np.zeros(400, bool)
    r1[rng.choice(400, 100, replace=False)] = True
    r2 = np.zeros(400, bool)
    r2[rng.choice(400, 100, replace=False)] = True
    _check(results, 0.3 < jaccard(r1, r2) < 3.0, "two random masks near the chance ratio")

    grp = np.r_[np.zeros(200), np.ones(200)]
    A_grouped = build_adjacency(lon, lat, k=k, groups=grp)
    _check(results, A_grouped[:200][:, 200:].nnz == 0,
           "grouped adjacency has no cross-group edges")

    # A scattered selection must score near 1, a compact one well below.
    w_compact = one_cluster.astype(float)
    w_scattered = scattered.astype(float)
    _check(results, incoherence(w_compact, L, 200, k) < incoherence(w_scattered, L, 200, k),
           "incoherence orders compact below scattered")

    n_passed = sum(results)
    n_failed = len(results) - n_passed
    print(f"graph.py self-test: {n_passed} passed, {n_failed} failed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(self_test())
