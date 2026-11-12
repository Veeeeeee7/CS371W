#!/usr/bin/env python3
"""
Checks for region_match.py (the caliper matching of the paper's Region Analysis, Table tab:regions). Run from the
repo root:

    .venv/bin/python aotil/test_region_match.py

Synthetic cases with known answers:
1. On a synthetic pool, exactly 5% of points (rounded) are unmatched against the pool.
2. A candidate in a tile that holds any point of the region is never a match, even when it is the
   closest; a one-point region (own-tile exclusion) leaves out only the query's own tile.
3. A closer candidate with a different land cover is never a match.
4. With 10 candidates inside eps, the 5 closest are kept, sorted by distance.
5. eta by hand: a positive point with matches labeled 1, 0, 0 scores 2/3; unmatched points are
   left out.
6. A query whose 600 nearest same-class candidates all lie in excluded tiles still finds its
   out-of-tile match.
7. q = the numeric features of FEATURES.txt (every feature but zoo.CATEGORICAL), in file order: 6 on
   data/processed_v4; a file without landcover stops.
8. prepare + run on a synthetic world: standardization over D_OOD, pool points missing a feature dropped and
   counted, one row per region + all regions + Other Zone A + the Zone AE baseline, the pooled eta = the regions'
   eta weighted by their matched points, Other Zone A = the points outside the subset, every pair within eps, in a
   tile outside its region, with the query's land cover.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region_match as rm  # noqa: E402

FAILS = []


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


def tiny_pool(z, landcover, tile, y=None):
    z = np.asarray(z, float).reshape(len(landcover), -1)
    y = np.zeros(len(landcover), int) if y is None else y
    return rm.Pool(z, np.asarray(landcover), np.asarray(tile), np.asarray(y))


def no_exclusion(nq, n_tiles):
    return np.zeros((nq, n_tiles), bool)


def check_caliper_share():
    print("1. 5% of a synthetic pool unmatched against itself")
    # Points on a line at c i^2, each in its own tile (so own-tile = itself): point i's nearest
    # distance is the gap to its left, c (2i - 1), distinct for every i >= 1, so no tie can sit
    # at the cut. Random points tie in pairs (mutual nearest neighbours share a distance).
    for n in (1000, 777):
        x = 1e-3 * np.arange(n, dtype=float) ** 2
        pool = tiny_pool(x[:, None], np.ones(n, int), np.arange(n))
        own = rm.own_tile_exclusion(pool.tile_code, n)
        nearest = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=1)[1][:, 0]
        eps = rm.caliper(nearest)
        idx, _ = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=5, eps=eps)
        unmatched = int((idx[:, 0] < 0).sum())
        check(unmatched == round(0.05 * n),
              f"n = {n}: {unmatched} unmatched = round(5% of n) = {round(0.05 * n)}")
    rng = np.random.default_rng(0)
    n = 777
    pool = tiny_pool(rng.normal(size=(n, 3)), np.ones(n, int), rng.integers(0, 4, n))
    own = rm.own_tile_exclusion(pool.tile_code, 4)
    nearest = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=1)[1][:, 0]
    eps = rm.caliper(nearest)
    idx, _ = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=5, eps=eps)
    unmatched = int((idx[:, 0] < 0).sum())
    check(abs(unmatched - round(0.05 * n)) <= 1
          and np.array_equal(idx[:, 0] >= 0, nearest <= eps),
          f"random pool, 4 tiles: {unmatched} unmatched (5% = {0.05 * n:.1f}, ties allow one "
          f"off); matched exactly when the nearest eligible candidate is within eps")


def check_region_tiles():
    print("2. tiles of the region are excluded")
    # Candidates: 0 in tile 1 (closest), 1 in tile 2 (next). The region holds points in tiles 0
    # and 1; the query lies in tile 0.
    pool = tiny_pool([[0.1], [0.5]], [1, 1], [1, 2])
    zq, lcq = np.array([[0.0]]), np.array([1])
    region_tiles = np.zeros((1, 3), bool)
    region_tiles[0, [0, 1]] = True
    idx, dist = rm.match(pool, zq, lcq, region_tiles, k=5, eps=1.0)
    check(idx[0].tolist() == [1, -1, -1, -1, -1] and np.isclose(dist[0, 0], 0.5),
          f"region-tiles: the closest candidate (in a region tile) is skipped: {idx[0].tolist()}")
    own = rm.own_tile_exclusion(np.array([0]), 3)
    idx, _ = rm.match(pool, zq, lcq, own, k=5, eps=1.0)
    check(idx[0].tolist()[:2] == [0, 1], f"one-point region (own tile only): the same candidate is kept: {idx[0].tolist()}")


def check_landcover():
    print("3. land cover must match")
    pool = tiny_pool([[0.01], [0.3], [0.2]], [90, 42, 90], [1, 1, 1])
    idx, _ = rm.match(pool, np.array([[0.0]]), np.array([42]), no_exclusion(1, 2), k=5, eps=1.0)
    check(idx[0].tolist() == [1, -1, -1, -1, -1],
          f"the closer class-90 candidates are never matches: {idx[0].tolist()}")
    idx, _ = rm.match(pool, np.array([[0.0]]), np.array([71]), no_exclusion(1, 2), k=5, eps=1.0)
    check(idx[0, 0] == -1, "a class with no candidate leaves the query unmatched")


def check_five_closest():
    print("4. the 5 closest of 10 candidates inside eps")
    d = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])
    pool = tiny_pool(d[:, None], np.ones(10, int), np.ones(10, int))
    idx, dist = rm.match(pool, np.array([[0.0]]), np.array([1]), no_exclusion(1, 2), k=5,
                         eps=1.0)
    check(idx[0].tolist() == [9, 1, 5, 3, 7] and np.allclose(dist[0], [0.05, 0.1, 0.2, 0.3, 0.4]),
          f"kept {idx[0].tolist()}, sorted by distance")
    idx, _ = rm.match(pool, np.array([[0.0]]), np.array([1]), no_exclusion(1, 2), k=5, eps=0.1)
    check(idx[0].tolist() == [9, 1, -1, -1, -1], "a candidate at exactly eps is a match")


def check_eta_by_hand():
    print("5. eta by hand")
    pool_y = np.array([1, 0, 0, 1])
    idx = np.array([[0, 1, 2], [-1, -1, -1], [3, -1, -1]])
    y = np.array([1, 0, 0])
    share = rm.disagreement(y, idx, pool_y)
    check(np.isclose(share[0], 2 / 3) and np.isnan(share[1]) and share[2] == 1.0,
          f"per point: {np.round(share, 3).tolist()}")
    check(np.isclose(rm.eta(y[:2], idx[:2], pool_y), 2 / 3),
          "the unmatched point is left out: eta = 2/3")
    check(np.isclose(rm.eta(y, idx, pool_y), (2 / 3 + 1) / 2), "each matched point counts once")


def check_query_expansion():
    print("6. the k-cap bug: 600 nearer candidates in excluded tiles")
    rng = np.random.default_rng(1)
    near = rng.uniform(0, 0.5, (600, 2))
    pool = tiny_pool(np.vstack([near, [[0.9, 0.9]]]), np.ones(601, int), np.r_[np.zeros(600), 1])
    idx, dist = rm.match(pool, np.array([[0.0, 0.0]]), np.array([1]),
                         rm.own_tile_exclusion(np.array([0]), 2), k=5, eps=2.0)
    check(idx[0, 0] == 600 and np.isclose(dist[0, 0], np.hypot(0.9, 0.9)),
          f"the out-of-tile candidate is found ({idx[0, 0]}, distance {dist[0, 0]:.3f})")


def check_features():
    print("7. the numeric features")
    v4 = Path("data/processed_v4/FEATURES.txt")
    if v4.exists():
        q = rm.numeric_features(v4)
        check(q == ["slope", "elev_sd_r300", "hand", "wetindexcat", "log10_dist_river", "log10_dist_coast"],
              f"v4: q = {len(q)} numeric features in file order")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "FEATURES.txt"
        f.write_text("slope\nhand\n")
        try:
            rm.numeric_features(f)
            ok = False
        except SystemExit:
            ok = True
        check(ok, "a file without landcover stops")


def synthetic_world(n_ood=400, n_id=400, seed=2):
    """OOD and ID rows with three numeric features, land cover, labels and tiles; 10 ID points miss a feature."""
    rng = np.random.default_rng(seed)

    def rows(n, prefix, shift):
        df = pd.DataFrame(dict(point_id=[f"{prefix}{i}" for i in range(n)], label=rng.integers(0, 2, n),
                               tile_50km=rng.choice(["0_0", "0_1", "1_0", "1_1", "2_0"], n),
                               landcover=rng.choice([42, 90], n), f1=rng.normal(shift, 1, n),
                               f2=rng.normal(0, 2, n), f3=rng.gamma(2, 1, n)))
        return df
    ood, idd = rows(n_ood, "A", 0.5), rows(n_id, "E", 0.0)
    idd.loc[rng.choice(n_id, 10, replace=False), "f2"] = np.nan
    return ood, idd


def check_run():
    print("8. prepare + run")
    ood, idd = synthetic_world()
    feats = ["f1", "f2", "f3"]
    ctx = rm.prepare(ood, idd, feats)
    check(np.allclose(ctx["mean"], ood[feats].mean()) and np.allclose(ctx["sd"], ood[feats].std()),
          "standardized by the D_OOD mean and SD")
    check(ctx["pool_dropped"] == 10 and len(ctx["pool"]) == 390, "10 pool points with a missing feature dropped")
    selected = np.zeros(len(ood), bool)
    regions = {0: np.flatnonzero(ood.tile_50km == "0_0")[:30], 1: np.flatnonzero(ood.tile_50km == "1_1")[:25]}
    for cols in regions.values():
        selected[cols] = True
    selected[np.flatnonzero(ood.tile_50km == "2_0")[:5]] = True        # selected, in no region
    out = rm.run(ctx, selected, regions)
    t = out["table"]
    check(t.kind.tolist() == ["region", "region", "pooled", "other", "baseline"], f"rows {t.kind.tolist()}")
    reg, pooled = t[t.kind == "region"], t[t.kind == "pooled"].iloc[0]
    check(int(pooled.n) == 55 and np.isclose(pooled.eta, (reg.eta * reg.n_matched).sum() / reg.n_matched.sum()),
          "all regions: 55 points, eta = the regions' eta weighted by matched points")
    other = t[t.kind == "other"].iloc[0]
    check(int(other.n) == len(ood) - int(selected.sum()), "Other Zone A = every point outside the subset")
    base = t[t.kind == "baseline"].iloc[0]
    check(int(base.n) == 390 and abs(base.unmatched - 0.05) < 0.01, f"baseline: the pool, {base.unmatched:.3f} unmatched")
    pairs = out["pairs"]
    eps = out["summary"]["eps"]
    tile_of = dict(zip(ctx["pool_df"].point_id, ctx["pool_df"].tile_50km))
    lc_pool = dict(zip(ctx["pool_df"].point_id, ctx["pool_df"].landcover))
    lc_ood = dict(zip(ood.point_id, ood.landcover))
    region_tiles = {c: set(ood.tile_50km.iloc[cols]) for c, cols in regions.items()}
    ok = all(r.distance <= eps and tile_of[r.match_point_id] not in region_tiles[r.component]
             and lc_pool[r.match_point_id] == lc_ood[r.point_id] for r in pairs.itertuples())
    check(len(pairs) > 0 and ok, f"{len(pairs)} pairs: within eps, outside the region's tiles, same land cover")
    ps = out["point_summary"]
    check(len(ps) == 55 and int(ps.n_matches.gt(0).sum()) == int(pooled.n_matched),
          "one summary row per region point; matched points = the pooled row's")


def main() -> int:
    check_caliper_share()
    check_region_tiles()
    check_landcover()
    check_five_closest()
    check_eta_by_hand()
    check_query_expansion()
    check_features()
    check_run()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
