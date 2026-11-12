#!/usr/bin/env python3
"""Checks for graph.py, subsets.py and regions.py. Run from the repo root: .venv/bin/python aotil/test_subsets.py

1. The graph: symmetric 0/1 adjacency without self loops, each row of the directed kNN graph holds k neighbours,
   the Laplacian's rows sum to 0, and the k-th neighbour distance equals a brute-force search.
2. Purity: a compact cluster scores high, a scattered subset low, a random subset about S / m; empty -> NaN.
3. Components: ids ranked by size (largest first, ties to the lowest index), coherence counts the components of
   >= c points and the points they hold.
4. Smoothing (Equation eq:smooth): the kept points are the S largest (I + D)^-1 (I + A) sigmoid(theta) of a dense
   computation; an isolated selected point among unselected neighbours drops out, an unselected point surrounded
   by selected ones comes in; ties go to the lower index; computing on the complement avoids weights that round
   to 1.0 tying.
5. probit_r equals the search's torch loss; the Fisher interval; evaluate's fields.
6. Region names: the county holding most of the points; (west)/(east) by mean longitude when two share a county.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import regions  # noqa: E402
import subsets  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def two_clusters(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([-81.0, 27.0], 0.002, size=(n, 2))
    b = rng.normal([-80.0, 28.0], 0.002, size=(n, 2))
    return np.r_[a[:, 0], b[:, 0]], np.r_[a[:, 1], b[:, 1]]


def check_graph():
    print("1. graph")
    lon, lat = two_clusters()
    A = graph.build_adjacency(lon, lat, k=8)
    check((A != A.T).nnz == 0 and A.diagonal().sum() == 0 and set(np.unique(A.data)) == {1.0},
          "symmetric 0/1 adjacency, no self loops")
    check(set(np.asarray(A.knn_directed.sum(1)).ravel()) == {8.0}, "each point's own list holds k = 8 neighbours")
    L = graph.laplacian(A)
    check(abs(np.asarray(L.sum(1))).max() < 1e-12, "Laplacian rows sum to 0")
    xy = graph.project(lon, lat)
    d = np.sort(cdist(xy, xy), axis=1)[:, 8]
    check(np.allclose(graph.kth_neighbour_m(lon, lat, 8), d), "k-th neighbour distance = brute force")


def check_purity():
    print("2. purity")
    lon, lat = two_clusters()
    A = graph.build_adjacency(lon, lat, k=8)
    rng = np.random.default_rng(1)
    compact = np.zeros(400, bool)
    compact[:200] = True
    scattered = np.zeros(400, bool)
    scattered[rng.choice(400, 200, replace=False)] = True
    pc, ps = graph.purity(compact, A), graph.purity(scattered, A)
    check(pc > 0.99 and ps < 0.7, f"compact {pc:.2f}, scattered {ps:.2f}")
    rng = np.random.default_rng(2)
    lon2, lat2 = -82 + rng.uniform(0, 1, 3000), 28 + rng.uniform(0, 1, 3000)
    A2 = graph.build_adjacency(lon2, lat2, k=10)
    rand = np.zeros(3000, bool)
    rand[rng.choice(3000, 300, replace=False)] = True
    check(abs(graph.purity(rand, A2) - 0.1) < 0.03, f"random subset about S/m = 0.10 ({graph.purity(rand, A2):.3f})")
    check(np.isnan(graph.purity(np.zeros(3000, bool), A2)), "empty subset: NaN")


def check_components():
    print("3. components")
    # Three chains of points 100 m apart (60, 25 and 12 points), far from each other, and 5 isolated points.
    lon, lat = [], []
    for n, (x0, y0) in ((60, (-82.0, 28.0)), (25, (-81.0, 28.5)), (12, (-80.5, 27.5))):
        lon += [x0 + i * 0.001 for i in range(n)]
        lat += [y0] * n
    lon += [-83.0 + 0.3 * i for i in range(5)]
    lat += [29.5] * 5
    lon, lat = np.array(lon), np.array(lat)
    A = graph.build_adjacency(lon, lat, k=2)
    mask = np.ones(len(lon), bool)
    labels, sizes = graph.component_labels(mask, A)
    check(sizes[:3].tolist() == [60, 25, 12] and labels[0] == 0 and labels[60] == 1 and labels[85] == 2,
          f"ids by size, largest first (sizes {sizes.tolist()})")
    co = graph.coherence(mask, A, c=20)
    check(co["components"] == 2 and co["comp_points"] == 85 and co["pieces"] == len(sizes),
          f"coherence counts the 2 components of >= 20 points and their 85 points ({co})")
    two = np.zeros(len(lon), bool)
    two[60:63] = True
    two[85:88] = True
    lab, sz = graph.component_labels(two, A)
    check(sz.tolist() == [3, 3] and lab[0] == 0, "equal sizes: the component with the lowest index is 0")


def check_smoothing():
    print("4. smoothing")
    rng = np.random.default_rng(3)
    lon, lat = -82 + rng.uniform(0, 0.3, 300), 28 + rng.uniform(0, 0.3, 300)
    A = graph.build_adjacency(lon, lat, k=10)
    theta = rng.normal(0, 2, 300)
    S = 60
    mask, ties = subsets.smooth(theta, A, S)
    Ad = A.toarray()
    s_bar = (expit(theta) + Ad @ expit(theta)) / (1 + Ad.sum(1))
    ref = np.zeros(300, bool)
    ref[np.argsort(-s_bar, kind="stable")[:S]] = True
    check(mask.sum() == S and np.array_equal(mask, ref) and ties == 1, "the S largest smoothed weights (dense check)")
    nbrs = np.flatnonzero(Ad[0])
    theta2 = np.full(300, -8.0)
    theta2[0] = 8.0                                   # alone among unselected neighbours
    theta2[np.flatnonzero(Ad[1])] = 8.0               # point 1's neighbours selected, point 1 not
    theta2[1] = -8.0
    k = int(Ad[1].sum())
    m2, _ = subsets.smooth(theta2, A, k)
    check(m2[1] and (not m2[0] or 0 in np.flatnonzero(Ad[1])) and len(nbrs) > 0,
          "an unselected point among selected neighbours comes in; an isolated selected point drops out")
    big = np.full(300, 40.0)                          # sigmoid(40) rounds to 1.0: every weight would tie
    big[np.arange(0, 300, 7)] = 39.0
    m3, _ = subsets.smooth(big, A, 100)
    q = expit(-big)
    qbar = (q + A @ q) / (1 + np.asarray(A.sum(axis=1)).ravel())     # the sparse product, as smooth() sums
    s_naive = expit(big)
    naive = subsets.top_mask((s_naive + Ad @ s_naive) / (1 + Ad.sum(1)), 100)
    check(np.array_equal(m3, subsets.bottom_mask(qbar, 100)) and np.all(s_naive == 1.0)
          and not np.array_equal(m3, naive),
          "weights that round to 1.0 tie (the naive top 100 is just the first 100); the complement ranks them")
    t = np.zeros(10)
    check(subsets.top_mask(t, 3).tolist() == [True] * 3 + [False] * 7, "ties go to the lower index")


def check_corr():
    print("5. correlation and evaluate")
    import search
    rng = np.random.default_rng(4)
    Z = (rng.random((58, 500)) < 0.7).astype(np.float64)
    acc = rng.uniform(0.5, 0.8, 58)
    mask = np.zeros(500, bool)
    mask[:120] = True
    r = subsets.probit_r(Z, acc, mask)
    r_t = search.hard_corr(Z.astype(np.float32), acc.astype(np.float32), mask.astype(np.float32), 120, is_mask=True)
    check(abs(r - r_t) < 1e-5, f"probit_r {r:+.6f} = the search's float32 loss {r_t:+.6f}")
    lo, hi = subsets.fisher_ci(0.5, 58)
    z, h = np.arctanh(0.5), 1.959963984540054 / np.sqrt(55)
    check(np.allclose((lo, hi), (np.tanh(z - h), np.tanh(z + h))), "Fisher interval")
    arr = dict(Z=Z, acc=acc, test=np.arange(20), val=np.arange(20, 40), tiles=np.array(["t1", "t2"] * 250))
    A = graph.build_adjacency(-82 + rng.uniform(0, 1, 500), 28 + rng.uniform(0, 1, 500), k=10)
    ev = subsets.evaluate(mask, arr, A)
    check({"r_te", "ci_lo", "ci_hi", "r_val", "components", "comp_points", "purity", "pieces", "tiles"} <= set(ev)
          and ev["n"] == 120 and ev["tiles"] == 2, "evaluate's fields")


def check_names():
    print("6. region names")
    rows = []
    for i in range(30):          # region 0: 30 points, county 12095 (west part)
        rows.append(dict(lon=-81.2 + i * 1e-4, lat=28.5, county="12095" if i < 20 else "12117"))
    for i in range(25):          # region 1: 25 points, county 12095 (east part)
        rows.append(dict(lon=-80.9 + i * 1e-4, lat=28.5, county="12095"))
    for i in range(22):          # region 2: 22 points, county 12065
        rows.append(dict(lon=-83.0 + i * 1e-4, lat=30.4, county="12065"))
    df = pd.DataFrame(rows)
    df["meta_point_county"] = df.county
    df["county_fips"] = df.county
    df["label"] = 1
    df["tile_50km"] = "1_1"
    df["x_5070"], df["y_5070"] = np.arange(len(df)) * 10.0, 0.0
    lon, lat = df.lon.to_numpy(), df.lat.to_numpy()
    A = graph.build_adjacency(lon, lat, k=5)
    out = regions.regions(np.ones(len(df), bool), A, df, {"12095": "Orange", "12065": "Liberty", "12117": "Seminole"})
    names = out["components"].set_index("n_points").name.to_dict()
    check(names == {30: "Orange (west)", 25: "Orange (east)", 22: "Liberty"}, f"names {names}")


def main() -> int:
    check_graph()
    check_purity()
    check_components()
    check_smoothing()
    check_corr()
    check_names()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
