#!/usr/bin/env python3
"""
Do the regions of one seed come back in the other seeds? (docs/TASK-FULL-RUN-W7.md, C5.)

Reads <regions-dir>/<cell>_<variant>_S<S>_points.parquet of every seed (aotil/regions.py) and
each seed's cell for its test tiles. Writes <out>, a CSV with one row per region (component of
>= --min-size points) per seed. For a region of seed s and each other seed k:
  tested_s<k>         share of its points in seed k's test tiles. A region can only be found
                      again where its ground is in seed k's test half, so this is the denominator.
  near_s<k>           share of its points within --radius-m of a point of a seed-k region.
  near_s<k>_tested    the same share among its points in seed-k test tiles (NaN when none).
  near_s<k>_any       the same against every point seed k selected, components of any size.
A region is "recurring" when near_s<k> >= --recur-share for at least one other seed.
Distances are straight-line metres in EPSG:5070.

    python aotil/region_recurrence.py --regions-dir results/<date>/regions \\
        --cells-dir data/interim/aotil/cells_w7 \\
        --out results/<date>/regions/recurrence_spatial_S500.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import load_cell  # noqa: E402
from regions import point_county  # noqa: E402

# The regions tables' lon/lat (NAD83) to the pipeline's metric CRS, Albers equal-area.
_TR = Transformer.from_crs(4269, 5070, always_xy=True)


def xy(table: pd.DataFrame) -> np.ndarray:
    """(x, y) in EPSG:5070 metres of each point's lon/lat."""
    x, y = _TR.transform(table.lon.to_numpy(), table.lat.to_numpy())
    return np.c_[x, y]


def kd_trees(tables: dict) -> dict:
    """seed -> KD-tree of that seed's points; a seed with no points gets none."""
    trees = {}
    for seed, table in tables.items():
        if len(table):
            trees[seed] = cKDTree(xy(table))
    return trees


def output_columns(seeds: list[int]) -> list[str]:
    """Every column a row can have, for the header-only table."""
    cols = ["seed", "component", "n", "positives", "tiles", "counties"]
    for k in seeds:
        cols += [f"tested_s{k}", f"near_s{k}", f"near_s{k}_tested", f"near_s{k}_any"]
    return cols + ["recurring"]


def against_seed(k: int, region: pd.DataFrame, region_xy: np.ndarray, k_test_tiles: set,
                 k_tree, k_tree_any, radius_m: float) -> dict:
    """The four shares of one region against seed k (see the module docstring)."""
    tested = region.tile_50km.isin(k_test_tiles).to_numpy()
    if k_tree is None:
        near = np.zeros(len(region), bool)
    else:
        near = k_tree.query(region_xy)[0] <= radius_m
    if tested.any():
        near_tested = near[tested].mean()
    else:
        near_tested = np.nan
    if k_tree_any is None:
        near_any = 0.0
    else:
        near_any = (k_tree_any.query(region_xy)[0] <= radius_m).mean()
    return {f"tested_s{k}": tested.mean(), f"near_s{k}": near.mean(),
            f"near_s{k}_tested": near_tested, f"near_s{k}_any": near_any}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions-dir", required=True, type=Path)
    ap.add_argument("--cells-dir", required=True, type=Path)
    ap.add_argument("--cell-prefix", default="FL_band_AE", help="cell tag without _s<seed>")
    ap.add_argument("--variant", default="spatial_mu3_knn10")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--radius-m", type=float, default=2000.0)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--recur-share", type=float, default=0.5)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    region_pts, selected, test_tiles = {}, {}, {}
    for s in args.seeds:
        tag = f"{args.cell_prefix}_s{s}"
        points_p = args.regions_dir / f"{tag}_{args.variant}_S{args.budget}_points.parquet"
        table = pd.read_parquet(points_p)
        region_pts[s] = table[table.component_size >= args.min_size].reset_index(drop=True)
        selected[s] = table
        test_tiles[s] = set(np.unique(load_cell(args.cells_dir, tag).tile_50km).tolist())
    trees = kd_trees(region_pts)
    trees_any = kd_trees(selected)

    rows = []
    for s, table in region_pts.items():
        if not len(table):
            continue
        table_xy = xy(table)
        others = [k for k in args.seeds if k != s]
        for c, region in table.groupby("component"):
            row = dict(seed=s, component=int(c), n=len(region), positives=int(region.label.sum()),
                       tiles=" ".join(sorted(set(region.tile_50km))),
                       counties=" ".join(sorted(set(point_county(region).astype(str)))))
            for k in others:
                row.update(against_seed(k, region, table_xy[region.index.to_numpy()], test_tiles[k],
                                        trees.get(k), trees_any.get(k), args.radius_m))
            row["recurring"] = any(row[f"near_s{k}"] >= args.recur_share for k in others)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # No seed has a region this large (plausible for classic selections, whose components
        # are small). A header-only table keeps downstream readers working.
        pd.DataFrame(columns=output_columns(args.seeds)).to_csv(args.out, index=False)
        seeds = " ".join(map(str, args.seeds))
        print(f"no region of >= {args.min_size} points in any seed ({seeds}); "
              f"nothing to test for recurrence")
        print(f"wrote {args.out} (header only)")
        return 0

    out = pd.DataFrame(rows)
    out.round(3).to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.round(2).to_string(index=False))
    for s in args.seeds:
        of_seed = out[out.seed == s]
        print(f"seed {s}: {len(of_seed)} regions of >= {args.min_size} points, "
              f"{int(of_seed.recurring.sum())} recurring "
              f"(near share >= {args.recur_share} in another seed)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
