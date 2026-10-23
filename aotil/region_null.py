#!/usr/bin/env python3
"""The matched-growth null for the regions (S2 of docs/TASK-O5-REGIONS.md).

Would any connected patch of unselected Zone A points, of a region's size and in its tiles, show
a region-vs-control difference as large as the region's? Points are spatially autocorrelated,
so the null draws whole patches, not single points (docs/REGION-ANALYSIS.md section 0.4). Each
draw starts at a random unselected land Zone A test point in one of the region's tiles, grows by
breadth-first search through unselected points of the seed's 10-NN graph (the graph OODSelect
was penalised on) to the region's size, and matches the patch's land points to S1 controls.
Patches are not matched on label composition; n_pos, n_land_pos and pos_share_land give it.

Reads   the inputs of match_controls.py (cells, both S = 500 selections, regions, point table).
Writes  <out> (parquet): one row per (seed, component, draw): the patch's point ids in BFS order,
        its land points with their controls and distances, and counts.

    .venv/bin/python aotil/region_null.py --out results/2026-10-17/o5/S2_null.parquet --caliper 1.0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import match_controls as mc  # noqa: E402
from cells import load_cell  # noqa: E402

RNG_SEED = 20261017
KNN = 10                 # neighbours per point: the graph of the spatial selection
DRAWS = 200              # null patches per region


def bfs_patch(A: sp.csr_matrix, start: int, size: int, allowed: np.ndarray,
              rng: np.random.Generator) -> np.ndarray | None:
    """A connected set of exactly `size` allowed nodes grown from `start` by BFS.

    Layers are added whole; the last one is subsampled uniformly at random. Returns node
    indices in BFS order (start first), or None when fewer than `size` allowed nodes are
    reachable from start through allowed nodes.
    """
    if not allowed[start]:
        raise ValueError("the start node is not allowed")
    seen = np.zeros(A.shape[0], bool)
    seen[start] = True
    layers = [np.array([start])]
    n_collected = 1
    while n_collected < size:
        neighbour_lists = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in layers[-1]]
        layer = np.unique(np.concatenate(neighbour_lists))
        layer = layer[allowed[layer] & ~seen[layer]]
        if len(layer) == 0:
            return None
        if n_collected + len(layer) > size:
            # Every node of this layer has a neighbour in the complete layer before it, so any
            # subset keeps the patch connected.
            layer = np.sort(rng.choice(layer, size - n_collected, replace=False))
        seen[layer] = True
        layers.append(layer)
        n_collected += len(layer)
    return np.concatenate(layers)


def allowed_components(A: sp.csr_matrix, allowed: np.ndarray) -> np.ndarray:
    """Size of each allowed node's connected component in the allowed-only subgraph (0 for a
    node that is not allowed): the most a BFS from it can reach."""
    nodes = np.flatnonzero(allowed)
    _, labels = connected_components(A[nodes][:, nodes], directed=False)
    sizes = np.bincount(labels)
    out = np.zeros(A.shape[0], int)
    out[nodes] = sizes[labels]
    return out


def seed_point_mask(ctx: mc.SeedContext, region: pd.DataFrame) -> np.ndarray:
    """Where a patch may start: unselected land OOD points in one of the region's tiles."""
    pts = ctx.pts
    unselected = ~pts.selected.to_numpy()
    in_region_tiles = pts.tile_50km.isin(set(region.tile_50km)).to_numpy()
    return unselected & pts.land.to_numpy() & in_region_tiles


def null_region(ctx: mc.SeedContext, A: sp.csr_matrix, comp_size: np.ndarray, component: int,
                draws: int, caliper, rng_seed: int = RNG_SEED, max_redraws: int = 100_000) -> list[dict]:
    """`draws` matched-growth patches for one region, each with its S1 controls.

    comp_size is allowed_components(A, unselected). A seed point whose unselected component is
    smaller than the region is redrawn; n_redraws counts these per draw.
    """
    pts = ctx.pts
    region = ctx.regions[ctx.regions.component == component]
    size = len(region)                    # all its points, water included, as in the region
    region_land = region[region.land]
    if len(region_land):
        pos_share = region_land.label.mean()
    else:
        pos_share = np.nan
    rtype = mc.region_type(pos_share)
    unselected = ~pts.selected.to_numpy()
    candidates = np.flatnonzero(seed_point_mask(ctx, region))
    # One generator per region, so a region's null does not change when other regions are
    # added or removed.
    rng = np.random.default_rng([rng_seed, ctx.seed, int(component)])
    point_ids = pts.point_id.to_numpy()
    land = pts.land.to_numpy()
    label = pts.label.to_numpy()
    tile = pts.tile_50km.to_numpy()

    rows = []
    if not (comp_size[candidates] >= size).any():
        largest = comp_size[candidates].max() if len(candidates) else 0
        print(f"  seed {ctx.seed} component {component}: no eligible seed point "
              f"({len(candidates)} candidates, largest reachable {largest} < {size})")
        return rows
    for draw in range(draws):
        redraws = 0
        while True:
            seed_point = int(rng.choice(candidates))
            if comp_size[seed_point] >= size:
                break
            redraws += 1
            if redraws > max_redraws:
                raise RuntimeError(f"seed {ctx.seed} component {component}: > {max_redraws} redraws")
        # The search moves only through unselected points, so a patch never contains part of a
        # region or of the classic selection. Water points may join, as in a region.
        patch = bfs_patch(A, seed_point, size, unselected, rng)
        assert patch is not None and len(patch) == size
        patch_land = patch[land[patch]]
        # The pool is S1's minus the patch itself.
        matches = mc.match_points(ctx, point_ids[patch_land], exclude_ids=point_ids[patch],
                                  caliper=caliper)
        is_matched = (matches.status == "matched").to_numpy()
        control_ids = []
        for control in matches.control_point_id:
            control_ids.append(control if isinstance(control, str) else None)
        if len(patch_land):
            matched_share = float(is_matched.mean())
            pos_share_land = float(label[patch_land].mean())
        else:
            matched_share = np.nan
            pos_share_land = np.nan
        patch_tiles = set(tile[patch])
        rows.append(dict(
            seed=ctx.seed, component=int(component), draw=draw, region_type=rtype,
            region_size=size, region_n_land=len(region_land),
            seed_point_id=point_ids[seed_point], seed_tile=tile[seed_point], n_redraws=redraws,
            patch_point_ids=point_ids[patch].tolist(), patch_size=len(patch),
            n_patch_water=int((~land[patch]).sum()),
            n_tiles=len(patch_tiles), tiles=" ".join(sorted(patch_tiles)),
            land_point_ids=matches.point_id.tolist(),
            control_point_ids=control_ids,
            distances=matches.distance.astype(float).tolist(),
            n_land=len(patch_land), n_matched=int(is_matched.sum()),
            matched_share=matched_share,
            n_missing_terrain=int((matches.status == "missing_terrain").sum()),
            n_pos=int(label[patch].sum()), n_land_pos=int(label[patch_land].sum()),
            pos_share_land=pos_share_land,
            caliper="none" if caliper is None else float(caliper), rng_seed=rng_seed))
    return rows


def check_graph(ctx: mc.SeedContext, A: sp.csr_matrix) -> None:
    """Raise SystemExit unless each region is exactly one connected component of A restricted
    to the spatial selection, i.e. unless A is the graph the selection was made on."""
    selected = ctx.pts[f"sel_{mc.REGION_VARIANT}"].to_numpy()
    nodes = np.flatnonzero(selected)
    _, labels = connected_components(A[nodes][:, nodes], directed=False)
    component_of = np.full(len(selected), -1)
    component_of[nodes] = labels
    for component, points in ctx.regions.groupby("component"):
        rows = ctx.index_of(points.point_id)
        found = set(component_of[rows])
        n_in_found = (component_of == component_of[rows[0]]).sum()
        if len(found) != 1 or n_in_found != len(points):
            raise SystemExit(f"{ctx.tag}: region {component} is not one component of the rebuilt graph")


def describe_null(ctx: mc.SeedContext, component: int, comp_size: np.ndarray,
                  draws: pd.DataFrame, seconds: float) -> str:
    """One log line on a region's null draws."""
    region = ctx.regions[ctx.regions.component == component]
    candidates = seed_point_mask(ctx, region)
    eligible_share = (comp_size[candidates] >= len(region)).mean()
    tiles = " ".join(sorted(set(region.tile_50km)))
    share = draws.matched_share
    return (f"  seed {ctx.seed} comp {component:>2} ({draws.region_type.iloc[0]}, n={len(region)}, "
            f"tiles {tiles}): {int(candidates.sum())} candidate seed points, "
            f"{eligible_share:.3f} eligible; redraws total {int(draws.n_redraws.sum())} "
            f"(max {int(draws.n_redraws.max())}); {draws.seed_point_id.nunique()} distinct seed "
            f"points; patch land {draws.n_land.mean():.1f}, pos share {draws.pos_share_land.mean():.2f}, "
            f"tiles {draws.n_tiles.mean():.2f}; matched share mean {share.mean():.3f} "
            f"[{share.quantile(.05):.3f}, {share.quantile(.95):.3f}] ({seconds:.1f} s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=mc.CELLS_DIR, type=Path)
    ap.add_argument("--cell-prefix", default=mc.CELL_PREFIX)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default=mc.DATA, type=Path)
    ap.add_argument("--sel-root", default=mc.SEL_ROOT, type=Path)
    ap.add_argument("--regions-dir", default=mc.REGIONS_DIR, type=Path)
    ap.add_argument("--budget", type=int, default=mc.BUDGET)
    ap.add_argument("--min-size", type=int, default=mc.MIN_SIZE)
    ap.add_argument("--knn", type=int, default=KNN)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--caliper", default="0.25", help="S1 caliper, standardized units, or 'none'")
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--components", type=int, nargs="*", default=None,
                    help="only these components (default: every region of each seed)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    start = time.time()
    caliper = mc.parse_caliper(args.caliper)
    ctxs = mc.load_contexts(args.seeds, args.cells_dir, args.cell_prefix, args.data, args.sel_root,
                            args.regions_dir, args.budget, args.min_size)
    rows = []
    for seed, ctx in ctxs.items():
        cell = load_cell(args.cells_dir, ctx.tag)
        # Symmetrised kNN graph on all OOD points, water included, in EPSG:3086 (metres;
        # build_adjacency's default): the graph the regions were cut from, as check_graph confirms.
        A = graph.build_adjacency(cell.lon, cell.lat, k=args.knn)
        check_graph(ctx, A)
        unselected = ~ctx.pts.selected.to_numpy()
        comp_size = allowed_components(A, unselected)
        n_sizes = len(np.unique(comp_size[unselected]))
        print(f"seed {seed}: {args.knn}-NN graph on {A.shape[0]} OOD points, {A.nnz // 2} edges; "
              f"regions are components of it (checked); unselected subgraph has {n_sizes} "
              f"distinct component sizes, largest {comp_size.max()}", flush=True)
        components = sorted(ctx.regions.component.unique())
        if args.components is not None:
            components = [c for c in components if c in args.components]
        for component in components:
            region_start = time.time()
            region_rows = null_region(ctx, A, comp_size, component, args.draws, caliper,
                                      args.rng_seed)
            rows += region_rows
            if region_rows:
                seconds = time.time() - region_start
                print(describe_null(ctx, component, comp_size, pd.DataFrame(region_rows), seconds),
                      flush=True)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_regions = out.groupby(["seed", "component"]).ngroups
    print(f"wrote {args.out}: {len(out)} rows ({n_regions} regions x {args.draws} draws), "
          f"{time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
