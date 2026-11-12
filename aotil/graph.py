#!/usr/bin/env python3
"""The kNN graph A on the OOD points, its Laplacian, and the coherence of a subset (the paper's Graph paragraph).

A connects two points when either is among the other's k = 10 nearest neighbours, measured in Florida Albers
(EPSG:3086, metres). kNN is used rather than a radius because flood zones are narrow ribbons: a kNN ball follows
the corridor, a radius window is mostly empty. Coherence of a subset:
  components  connected components of the subgraph the subset induces in A with at least c = 20 points
              (Algorithm alg:spatial), and the points they hold
  purity      the mean over the selected points of the share of each point's own k nearest neighbours (the
              directed kNN graph) that are also selected; a random subset of S of m points scores about S / m
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Mean Earth radius in metres, for the equirectangular fallback in project(). The
# default CRS is Florida Albers (EPSG:3086), the project's planar CRS.
EARTH_R = 6_371_008.8
C_MIN = 20                 # c, the minimum component size


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


def build_adjacency(lon, lat, k: int = 10, epsg: int | None = 3086) -> sp.csr_matrix:
    """Symmetric 0/1 kNN adjacency over the projected points (CSR, float64).

    The directed kNN graph (row i = point i's own neighbours) is attached as
    `A.knn_directed`; purity uses it.
    """
    xy = project(lon, lat, epsg)
    n = len(xy)
    rows, cols = _knn_pairs(xy, k)
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


def kth_neighbour_m(lon, lat, k: int = 10, epsg: int = 3086) -> np.ndarray:
    """Each point's distance in metres to its k-th nearest other point (the Graph paragraph's median)."""
    xy = project(lon, lat, epsg)
    d, _ = cKDTree(xy).query(xy, k=k + 1)
    return d[:, k]


def purity(mask: np.ndarray, A: sp.csr_matrix) -> float:
    """The mean over selected points of the share of each point's own nearest neighbours (A.knn_directed; the
    rows of A when A lacks it) that are also selected. NaN for an empty subset."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return float("nan")
    knn_selected = getattr(A, "knn_directed", A)[mask]
    n_neighbours = _row_sums(knn_selected)
    n_selected_neighbours = _row_sums(knn_selected[:, mask])
    return float(np.mean(n_selected_neighbours / np.maximum(n_neighbours, 1)))


def component_labels(mask: np.ndarray, A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Component id of each selected point (in OOD order), and the component sizes in id order.

    Ids are 0..n-1 by size, largest first, ties to the component holding the lowest OOD column
    index. scipy's own labels follow visiting order, which shifts with the cell's row order.
    """
    mask = np.asarray(mask, bool)
    n, labels = connected_components(A[mask][:, mask], directed=False)
    sizes = np.bincount(labels, minlength=n)
    first = np.full(n, mask.sum())
    np.minimum.at(first, labels, np.arange(len(labels)))    # lowest OOD index per component
    order = np.lexsort((first, -sizes))
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    return rank[labels], sizes[order]


def coherence(mask: np.ndarray, A: sp.csr_matrix, c: int = C_MIN) -> dict:
    """The coherence of a subset: components (>= c points), the points they hold, all connected pieces, purity."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return dict(components=0, comp_points=0, pieces=0, purity=float("nan"))
    _, sizes = component_labels(mask, A)
    big = sizes >= c
    return dict(components=int(big.sum()), comp_points=int(sizes[big].sum()), pieces=int(len(sizes)),
                purity=purity(mask, A))
