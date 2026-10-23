#!/usr/bin/env python3
"""Checks for match_controls.py (S1) and region_null.py (S2); exit code 0 = all pass.

Synthetic, always run:
1. match(): the optimal assignment keeps a point greedy matching loses and equals a brute-force
   optimum on small problems; controls stay in their stratum and caliper and are used once;
   missing terrain, no caliper, the status codes, rescaling, overlapping ids.
2. smd(): a known value, NaNs skipped, integer weights equal repeated rows.
3. bfs_patch() and allowed_components() on a lattice cut by a wall.
4. null_region() on a synthetic seed context: patches avoid the selections, controls are valid,
   redraws are counted, reruns reproduce.
Real, when the outputs exist in --out-dir:
5. S1_controls.csv recomputed from the point table.
6. S2_null.parquet: patch size, connectivity, controls, and the first draws reproduced.

    .venv/bin/python aotil/test_match_controls.py --out-dir results/2026-10-17/o5 --caliper 1.0
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import match_controls as mc  # noqa: E402
import region_null as rn  # noqa: E402
from cells import load_cell  # noqa: E402

TERRAIN = list(mc.TERRAIN)
FAILS = []


def check(passed, message: str) -> None:
    """Print one check's result; a failure is also recorded in FAILS."""
    if passed:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILS.append(message)


# ------------------------------------------------------------------ 1. match()
def frame(ids, tile, label, X) -> pd.DataFrame:
    """A query or pool table. X holds the terrain columns, or only slope (the rest are 0)."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[1] == 1:
        X = np.c_[X, np.zeros((len(X), len(TERRAIN) - 1))]
    table = pd.DataFrame(X, columns=TERRAIN)
    table.insert(0, "point_id", ids)
    table["tile_50km"] = tile
    table["label"] = label
    return table


def random_frame(rng: np.random.Generator, prefix: str, n: int, tiles: np.ndarray,
                 sd: np.ndarray) -> pd.DataFrame:
    tile = rng.choice(tiles, n)
    label = rng.integers(0, 2, n)
    X = rng.normal(size=(n, len(TERRAIN))) * sd
    return frame([f"{prefix}{i}" for i in range(n)], tile, label, X)


def brute(D: np.ndarray, caliper: float) -> tuple:
    """(most matches, least total distance) over all injective assignments of rows to columns;
    (-1, inf) when there are more rows than columns."""
    best = (-1, np.inf)
    n_rows, n_cols = D.shape
    if n_cols < n_rows:
        return best
    for columns in itertools.permutations(range(n_cols), n_rows):
        n_matched = 0
        total = 0.0
        for i, j in enumerate(columns):
            if D[i, j] <= caliper:
                n_matched += 1
                total += D[i, j]
        if n_matched > best[0] or (n_matched == best[0] and total < best[1] - 1e-12):
            best = (n_matched, total)
    return best


def check_greedy_example() -> None:
    sd = np.ones(len(TERRAIN))
    q = frame(["q1", "q2"], "t", 1, [[0.0], [0.2]])
    p = frame(["c1", "c2"], "t", 1, [[0.15], [-0.2]])
    m = mc.match(q, p, sd, caliper=0.25)
    # Greedy in query order: q1 takes c1 (0.15 < 0.20), which leaves q2 only c2, at 0.40.
    check((m.status == "matched").all() and set(m.control_point_id) == {"c1", "c2"},
          f"both points matched where greedy keeps one "
          f"(q1->{m.control_point_id[0]}, q2->{m.control_point_id[1]})")
    check(abs(m.distance.sum() - 0.25) < 1e-12, f"total distance 0.25 (got {m.distance.sum():.3f})")


def check_brute_force(rng: np.random.Generator) -> None:
    sd = np.ones(len(TERRAIN))
    n_ok = 0
    for _ in range(60):
        n_query = rng.integers(1, 4)
        n_pool = rng.integers(1, 6)
        Xq = rng.normal(size=(n_query, len(TERRAIN))) * 0.15
        Xp = rng.normal(size=(n_pool, len(TERRAIN))) * 0.15
        q = frame([f"q{i}" for i in range(n_query)], "t", 0, Xq)
        p = frame([f"p{i}" for i in range(n_pool)], "t", 0, Xp)
        m = mc.match(q, p, sd, caliper=0.4)
        D = np.linalg.norm(Xq[:, None] - Xp[None], axis=2)
        if n_pool < n_query:
            D = D.T                    # brute() needs no more rows than columns
        best_n, best_total = brute(D, 0.4)
        matched = m.status == "matched"
        if matched.sum() == best_n and abs(m.distance[matched].sum() - best_total) < 1e-9:
            n_ok += 1
    check(n_ok == 60, f"equals the brute-force optimum on {n_ok}/60 random problems")


def check_random_problem(rng: np.random.Generator) -> None:
    """Strata, caliper, no replacement and distances on a 300 x 1500 problem."""
    tiles = np.array(["a", "b", "c"])
    sd = np.array([1.4, 1.1, 5.7, 0.7, 1.4, 2.7])
    q = random_frame(rng, "q", 300, tiles, sd)
    p = random_frame(rng, "p", 1500, tiles, sd)
    q.loc[[3, 7], "hand"] = np.nan
    p.loc[:40, "tpi_broad"] = np.nan
    caliper = 1.5
    m = mc.match(q, p, sd, caliper=caliper)
    matched = m[m.status == "matched"]
    queries = q.set_index("point_id").loc[matched.point_id]
    controls = p.set_index("point_id").loc[matched.control_point_id]

    check(matched.control_point_id.is_unique,
          f"without replacement ({len(matched)} matched, all controls distinct)")
    same_tile = (controls.tile_50km.to_numpy() == queries.tile_50km.to_numpy()).all()
    same_label = (controls.label.to_numpy() == queries.label.to_numpy()).all()
    check(same_tile and same_label, "every control in its point's (tile, label) stratum")
    distance = np.linalg.norm((queries[TERRAIN].to_numpy() - controls[TERRAIN].to_numpy()) / sd, axis=1)
    check(np.allclose(distance, matched.distance) and (matched.distance <= caliper).all(),
          f"distance = standardized Euclidean, all <= caliper (max {matched.distance.max():.3f})")
    check(set(m.loc[[3, 7], "status"]) == {"missing_terrain"}
          and m.loc[[3, 7], "control_point_id"].isna().all(),
          "query with missing terrain: missing_terrain, unmatched")
    check(not set(matched.control_point_id) & set(p.point_id[:41]),
          "pool rows with missing terrain never used")
    check(set(m.status) <= {"matched", "missing_terrain", "no_candidate", "lost_to_competition"},
          f"status codes {sorted(set(m.status))}")
    no_candidate = m[m.status == "no_candidate"]
    check((no_candidate.n_candidates == 0).all(),
          "no_candidate rows have no pool point within the caliper")
    check(m.equals(mc.match(q, p, sd, caliper=caliper)), "deterministic: two calls equal")

    q_scaled = q.copy()
    p_scaled = p.copy()
    q_scaled["hand"] *= 10
    p_scaled["hand"] *= 10
    sd_scaled = sd.copy()
    sd_scaled[TERRAIN.index("hand")] *= 10
    m_scaled = mc.match(q_scaled, p_scaled, sd_scaled, caliper=caliper)
    check((m_scaled.control_point_id.fillna("") == m.control_point_id.fillna("")).all(),
          "rescaling a column with its SD changes nothing")

    m_none = mc.match(q, p, sd, caliper=None)
    pool_complete = p[np.isfinite(p[TERRAIN].to_numpy()).all(axis=1)]
    query_complete = q[np.isfinite(q[TERRAIN].to_numpy()).all(axis=1)]
    expected = 0
    for (tile, label), group in query_complete.groupby(["tile_50km", "label"]):
        in_stratum = (pool_complete.tile_50km == tile) & (pool_complete.label == label)
        expected += min(len(group), in_stratum.sum())
    check((m_none.status == "matched").sum() == expected,
          f"caliper none: min(n_query, n_pool) per stratum ({expected})")


def check_statuses() -> None:
    sd = np.ones(len(TERRAIN))
    q = frame(["a", "b", "c"], ["t", "t", "u"], [1, 1, 1], [[0.0], [0.05], [0.0]])
    p = frame(["x", "y"], ["t", "t"], [1, 1], [[0.02], [3.0]])
    statuses = list(mc.match(q, p, sd, caliper=0.25).status)
    check(statuses in (["matched", "lost_to_competition", "empty_stratum"],
                       ["lost_to_competition", "matched", "empty_stratum"]),
          f"statuses with one shared candidate and an empty stratum: {statuses}")
    try:
        mc.match(q, frame(["a"], "t", 1, [[0.0]]), sd)
        raised = False
    except ValueError:
        raised = True
    check(raised, "overlapping query/pool ids raise")


def test_match() -> None:
    print("1. match(): optimal assignment, strata, caliper, missing values, statuses")
    check_greedy_example()
    rng = np.random.default_rng(1)
    check_brute_force(rng)
    check_random_problem(rng)
    check_statuses()


# ------------------------------------------------------------------ 2. smd()
def test_smd() -> None:
    print("2. smd()")
    a = np.array([1.0, 2, 3, 4])
    b = np.array([0.0, 1, 2, 3])
    expected = 1.0 / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    check(abs(mc.smd(a, b) - expected) < 1e-12, f"smd known value {mc.smd(a, b):.4f} = {expected:.4f}")
    check(abs(mc.smd(np.r_[a, np.nan], b) - expected) < 1e-12, "NaN skipped")
    weights = np.array([2, 1, 1, 3])
    check(abs(mc.smd(a, b, wb=weights) - mc.smd(a, np.repeat(b, weights))) < 1e-12,
          "integer weights = repetition")


# ------------------------------------------------------------------ 3. bfs_patch()
def lattice(nx, ny, k=8):
    """A jittered nx x ny lattice as lon/lat and its kNN adjacency."""
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    lon = -82 + gx.ravel() * 0.002 + rng.normal(0, 1e-5, nx * ny)
    lat = 28 + gy.ravel() * 0.002 + rng.normal(0, 1e-5, nx * ny)
    return lon, lat, graph.build_adjacency(lon, lat, k=k)


def hops_from(A, allowed: np.ndarray, start: int) -> np.ndarray:
    """Hops from start to each node through allowed nodes (inf when unreachable or not allowed)."""
    allowed_nodes = np.flatnonzero(allowed)
    start_in_subgraph = np.flatnonzero(allowed_nodes == start)[0]
    subgraph_hops = shortest_path(A[allowed][:, allowed], unweighted=True, indices=start_in_subgraph)
    hops = np.full(A.shape[0], np.inf)
    hops[allowed] = subgraph_hops
    return hops


def test_bfs() -> None:
    print("3. bfs_patch(), allowed_components()")
    lon, lat, A = lattice(30, 30)
    gx = np.tile(np.arange(30), 30)
    allowed = np.ones(A.shape[0], bool)
    # A wall two columns thick: a kNN edge here spans at most two lattice spacings, so a
    # one-column wall could be jumped.
    allowed[(gx == 15) | (gx == 16)] = False
    left_of_wall = np.flatnonzero(allowed & (gx < 15))
    rng = np.random.default_rng(5)
    for size in (1, 20, 57, 200):
        start = int(left_of_wall[rng.integers(0, 300)])
        patch = rn.bfs_patch(A, start, size, allowed, np.random.default_rng(size))
        n_parts = connected_components(A[patch][:, patch], directed=False)[0]
        hops = hops_from(A, allowed, start)
        farthest = hops[patch].max()
        closer = np.flatnonzero(hops < farthest)
        check(len(patch) == size and len(set(patch)) == size and patch[0] == start and n_parts == 1
              and allowed[patch].all() and set(closer) <= set(patch),
              f"size {size}: exact, distinct, starts at the seed, connected, allowed only, "
              f"all nodes < {int(farthest)} hops included")

    start = int(left_of_wall[0])
    reach = rn.allowed_components(A, allowed)
    check(rn.bfs_patch(A, start, int(reach[start]), allowed, rng) is not None
          and rn.bfs_patch(A, start, int(reach[start]) + 1, allowed, rng) is None,
          f"reachable set {reach[start]} (one side of the wall): that size grows, one more returns None")
    check(reach[start] == ((gx < 15) & allowed).sum() and (reach[~allowed] == 0).all(),
          "allowed_components = size of the allowed side; 0 for blocked nodes")
    first = rn.bfs_patch(A, start, 77, allowed, np.random.default_rng(9))
    again = rn.bfs_patch(A, start, 77, allowed, np.random.default_rng(9))
    patches_by_seed = set()
    for s in range(10):
        patch = rn.bfs_patch(A, start, 77, allowed, np.random.default_rng(s))
        patches_by_seed.add(tuple(sorted(patch)))
    check(np.array_equal(first, again) and len(patches_by_seed) > 1,
          "same rng seed -> same patch; seeds vary the last layer")


# ------------------------------------------------------------------ 4. null_region()
def synthetic_context(seed=0):
    """A fake SeedContext on a 40 x 40 lattice: two tiles, one 36-point region, two selections,
    and walled-off islands of unselected points. Returns (ctx, A, island mask)."""
    rng = np.random.default_rng(seed)
    lon, lat, A = lattice(40, 40, k=10)
    n = len(lon)
    gx = np.tile(np.arange(40), 40)
    gy = np.repeat(np.arange(40), 40)
    point_ids = np.array([f"P{i:05d}" for i in range(n)])
    label = (rng.random(n) < 0.4).astype(int)
    pts = pd.DataFrame(rng.normal(size=(n, len(TERRAIN))), columns=TERRAIN)
    pts.insert(0, "point_id", point_ids)
    pts["label"] = label
    pts["tile_50km"] = np.where(gx < 20, "t0", "t1")
    pts["lon"] = lon
    pts["lat"] = lat
    pts["landcover"] = np.where(rng.random(n) < 0.05, 11, 42).astype(float)    # 11: open water
    pts.loc[rng.choice(n, 10, replace=False), "hand"] = np.nan
    pts["land"] = pts.landcover != 11

    region = np.flatnonzero((gx >= 5) & (gx < 11) & (gy >= 5) & (gy < 11))     # 36 points, tile t0
    sel_spatial = np.zeros(n, bool)
    sel_spatial[region] = True
    sel_spatial[np.flatnonzero((gx >= 30) & (gy >= 30))[:40]] = True
    sel_classic = np.zeros(n, bool)
    sel_classic[rng.choice(n, 60, replace=False)] = True
    # Islands in the upper half of t0: 3 x 3 cells of unselected points inside classic-selected
    # walls two points thick, which a 10-NN edge cannot cross. A seed point there cannot grow
    # 36 points, so it is redrawn.
    upper_t0 = (gx < 20) & (gy >= 20)
    wall = upper_t0 & ((gx % 5 < 2) | (gy % 5 < 2) | (gx >= 15))      # gx >= 15: no leak into t1
    islands = upper_t0 & ~wall
    sel_classic[islands] = False
    sel_classic[wall] = True
    pts[f"sel_{mc.REGION_VARIANT}"] = sel_spatial
    pts["sel_classic"] = sel_classic
    pts["selected"] = sel_spatial | sel_classic
    pts["complete"] = np.isfinite(pts[TERRAIN].to_numpy()).all(axis=1)
    pts["component"] = -1
    pts.loc[region, "component"] = 0
    sd = pts.loc[pts.land, TERRAIN].std(ddof=1)
    for v in TERRAIN:
        pts[f"z_{v}"] = pts[v] / sd[v]

    region_columns = ["point_id", "component", "label", "tile_50km", "landcover", "land", "complete",
                      *TERRAIN]
    regions = pts.loc[region, region_columns]
    regions.insert(2, "component_size", len(region))
    ctx = mc.SeedContext(seed, "synthetic", pts, sd, regions.reset_index(drop=True))
    return ctx, A, islands


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Row-for-row equality, list columns included, compared as text. A scalar cell that is NaN
    or None stays missing after astype(str), so it never compares equal."""
    a = a.reset_index(drop=True).astype(str)
    b = b.reset_index(drop=True).astype(str)
    return a.shape == b.shape and bool((a.to_numpy() == b.to_numpy()).all())


def test_null() -> None:
    print("4. null_region() on a synthetic seed context")
    ctx, A, islands = synthetic_context()
    unselected = ~ctx.pts.selected.to_numpy()
    comp_size = rn.allowed_components(A, unselected)
    largest_island = comp_size[islands & unselected].max()
    check(largest_island < 36, f"an island reaches at most {largest_island} points (< 36)")

    R = pd.DataFrame(rn.null_region(ctx, A, comp_size, 0, 60, 1.0, rng_seed=3))
    position = {p: i for i, p in enumerate(ctx.pts.point_id.to_numpy())}
    selected = ctx.pts.selected.to_numpy()
    by_id = ctx.pts.set_index("point_id")
    size_ok = connected_ok = unselected_ok = land_ok = controls_ok = stratum_ok = True
    for _, row in R.iterrows():
        patch = np.array([position[p] for p in row.patch_point_ids])
        size_ok &= len(patch) == 36 == len(set(patch))
        connected_ok &= connected_components(A[patch][:, patch], directed=False)[0] == 1
        unselected_ok &= not selected[patch].any()
        land_ids = [p for p in row.patch_point_ids if by_id.loc[p, "land"]]
        land_ok &= sorted(land_ids) == sorted(row.land_point_ids)
        controls = [c for c in row.control_point_ids if c is not None]
        controls_ok &= (not set(controls) & set(row.patch_point_ids)
                        and not by_id.loc[controls, "selected"].any()
                        and len(set(controls)) == len(controls)
                        and by_id.loc[controls, "land"].all())
        for query, control, distance in zip(row.land_point_ids, row.control_point_ids, row.distances):
            if control is not None:
                stratum_ok &= (by_id.loc[query, "tile_50km"] == by_id.loc[control, "tile_50km"]
                               and by_id.loc[query, "label"] == by_id.loc[control, "label"]
                               and distance <= 1.0 + 1e-12)
    check(size_ok and connected_ok, "every patch: 36 distinct points, connected")
    check(unselected_ok, "no patch contains a selected point")
    check(land_ok, "land_point_ids = the patch's land points")
    check(controls_ok, "controls: outside the patch and the selections, land, distinct")
    check(stratum_ok, "controls share tile and label, distance <= caliper")
    check((R.seed_tile == "t0").all(), "seed points in the region's tile")
    n_island_seeds = (comp_size[ctx.index_of(R.seed_point_id)] < 36).sum()
    check(n_island_seeds == 0, "no accepted seed point is on an island")

    rerun = pd.DataFrame(rn.null_region(ctx, A, comp_size, 0, 60, 1.0, rng_seed=3))
    short_run = pd.DataFrame(rn.null_region(ctx, A, comp_size, 0, 7, 1.0, rng_seed=3))
    check(same(rerun, R), "rerun with the same rng seed: identical")
    check(same(short_run, R.iloc[:7]), "first 7 of 60 draws = a 7-draw run")
    other_seed = pd.DataFrame(rn.null_region(ctx, A, comp_size, 0, 7, 1.0, rng_seed=4))
    check(not same(other_seed, R.iloc[:7]), "another rng seed gives other patches")

    # A share f of the seed-point candidates is on islands, so each draw needs a geometric
    # number of redraws with mean f / (1 - f).
    in_t0 = (ctx.pts.tile_50km == "t0").to_numpy()
    candidates = np.flatnonzero(unselected & ctx.pts.land.to_numpy() & in_t0)
    island_share = (comp_size[candidates] < 36).mean()
    expected = island_share / (1 - island_share) * 60
    sd_redraws = np.sqrt(60 * island_share) / (1 - island_share)
    check(abs(R.n_redraws.sum() - expected) < 4 * sd_redraws + 1,
          f"redraws counted: {R.n_redraws.sum()} over 60 draws "
          f"(expected {expected:.1f}; island share {island_share:.3f})")


# ------------------------------------------------------------------ 5., 6. real outputs
def check_real_controls(K: pd.DataFrame, ctxs: dict, caliper: float) -> None:
    all_listed = in_pool = in_stratum = unique = distance_ok = True
    for seed, ctx in ctxs.items():
        k = K[K.seed == seed]
        by_id = ctx.pts.set_index("point_id")
        all_listed &= set(k.point_id) == set(ctx.regions.point_id[ctx.regions.land])
        matched = k[k.status == "matched"]
        controls = by_id.loc[matched.control_point_id]
        in_pool &= bool(controls.land.all() and (~controls.selected).all())
        query_tiles = by_id.loc[matched.point_id, "tile_50km"].to_numpy()
        in_stratum &= bool((controls.tile_50km.to_numpy() == matched.tile.to_numpy()).all()
                           and (controls.label.to_numpy() == matched.label.to_numpy()).all()
                           and (query_tiles == matched.tile.to_numpy()).all())
        unique &= matched.control_point_id.is_unique
        sd = ctx.sd[TERRAIN].to_numpy()
        z_query = by_id.loc[matched.point_id, TERRAIN].to_numpy() / sd
        z_control = controls[TERRAIN].to_numpy() / sd
        distance = np.linalg.norm(z_query - z_control, axis=1)
        distance_ok &= bool(np.allclose(distance, matched.distance, atol=1e-9)
                            and (matched.distance <= caliper + 1e-12).all())
    check(all_listed, "every region land point is listed")
    check(in_pool, "every control: land, outside the spatial and classic S = 500 selections")
    check(in_stratum, "every control: the region point's tile and label")
    check(unique, "no control used twice within a seed")
    check(distance_ok, f"distance recomputes from the point table, <= {caliper:g} (the run's caliper)")


def check_real_null(N: pd.DataFrame, ctxs: dict) -> None:
    size_ok = connected_ok = unselected_ok = controls_ok = land_ok = True
    reproduced = []
    for seed, ctx in ctxs.items():
        cell = load_cell(mc.CELLS_DIR, ctx.tag)
        A = graph.build_adjacency(cell.lon, cell.lat, k=10)
        position = {p: i for i, p in enumerate(ctx.pts.point_id.to_numpy())}
        by_id = ctx.pts.set_index("point_id")
        selected = ctx.pts.selected.to_numpy()
        region_sizes = ctx.regions.groupby("component").size()
        for _, row in N[N.seed == seed].iterrows():
            patch = np.array([position[p] for p in row.patch_point_ids])
            size_ok &= len(patch) == region_sizes[row.component] == len(set(patch))
            connected_ok &= connected_components(A[patch][:, patch], directed=False)[0] == 1
            unselected_ok &= not selected[patch].any()
            patch_land = [p for p in row.patch_point_ids if by_id.loc[p, "land"]]
            land_ok &= list(row.land_point_ids) == patch_land
            queries = []
            controls = []
            for query, control in zip(row.land_point_ids, row.control_point_ids):
                if control is not None:
                    queries.append(query)
                    controls.append(control)
            if controls:
                same_tile = (by_id.loc[controls, "tile_50km"].to_numpy()
                             == by_id.loc[queries, "tile_50km"].to_numpy())
                same_label = (by_id.loc[controls, "label"].to_numpy()
                              == by_id.loc[queries, "label"].to_numpy())
                controls_ok &= bool(not set(controls) & set(row.patch_point_ids)
                                    and not by_id.loc[controls, "selected"].any()
                                    and same_tile.all() and same_label.all()
                                    and np.nanmax(row.distances) <= float(row.caliper) + 1e-12)

        comp_size = rn.allowed_components(A, ~selected)
        caliper = mc.parse_caliper(str(N.caliper.iloc[0]))
        for component in sorted(ctx.regions.component.unique())[:1]:     # each seed's first region
            again = pd.DataFrame(rn.null_region(ctx, A, comp_size, component, 3, caliper,
                                                int(N.rng_seed.iloc[0])))
            stored = N[(N.seed == seed) & (N.component == component)]
            first = stored.sort_values("draw").head(3)
            same_patches = all(list(a) == list(b)
                               for a, b in zip(again.patch_point_ids, first.patch_point_ids))
            same_controls = all(list(a) == list(b)
                                for a, b in zip(again.control_point_ids, first.control_point_ids))
            reproduced.append(same_patches and same_controls)
    check(size_ok, "every patch has its region's size")
    check(connected_ok, "every patch is connected in the rebuilt 10-NN graph")
    check(unselected_ok, "no patch contains a point of either S = 500 selection")
    check(land_ok, "land lists = the patch's land points, in patch order")
    check(controls_ok, "controls: outside the patch and selections, same tile and label, within the caliper")
    check(all(reproduced), f"first 3 draws reproduced by a rerun ({len(reproduced)} regions)")


def test_real(out_dir: Path, caliper: float = 1.0):
    controls_path = out_dir / "S1_controls.csv"
    if not controls_path.exists():
        print(f"5./6. skipped: {controls_path} not found")
        return
    print(f"5. S1 controls in {out_dir} against the data")
    K = pd.read_csv(controls_path, dtype={"point_id": str, "control_point_id": str, "tile": str})
    ctxs = mc.load_contexts([0, 1, 2], mc.CELLS_DIR, mc.CELL_PREFIX, mc.DATA, mc.SEL_ROOT,
                            mc.REGIONS_DIR, mc.BUDGET, mc.MIN_SIZE, verbose=False)
    check_real_controls(K, ctxs, caliper)

    null_path = out_dir / "S2_null.parquet"
    if not null_path.exists():
        print(f"6. skipped: {null_path} not found")
        return
    print(f"6. S2 null in {out_dir}")
    check_real_null(pd.read_parquet(null_path), ctxs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/2026-10-17/o5", type=Path)
    ap.add_argument("--caliper", type=float, default=1.0,
                    help="the caliper the S1 outputs in --out-dir were made with (default 1.0, the "
                         "O5 run's, docs/O5-REGIONS-REPORT-2026-10-17.md; 0.25 for S1_caliper0.25/)")
    args = ap.parse_args()
    test_match()
    test_smd()
    test_bfs()
    test_null()
    test_real(args.out_dir, args.caliper)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
