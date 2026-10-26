#!/usr/bin/env python3
"""
The regions table: each region of a selection, its land points and its type.

A region is a connected piece of at least --min-size points of a selection (regions.py). A land
point has an NLCD land-cover class other than 11 (open water); a point with no class is not land.
The type is the majority label of the region's land points: positive-dominated when at least half
are Zone A positives, negative-dominated otherwise, and "no land points" when it has none (the
paper calls the first two flood-labeled and dry-labeled). Counties are those the points lie in,
named as regions.py names them (meta_point_county, else the DFIRM's county).

Reads   <regions-dir>/<cell-prefix>_s<seed>_<variant>_S<budget>_points.parquet (regions.py)
        --dataset, for each point's land cover
Writes  --out: one row per region, the columns of results/2026-10-17/o5/regions_land.csv, which
        o5_interpolation.py, o5_contrast.py and o5_polygons.py read

    python aotil/regions_land.py --regions-dir results/<date>/regions --seeds 0 \\
        --dataset data/processed_w7/dataset_FL.parquet --out results/<date>/o5/regions_land.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regions import point_county  # noqa: E402

WATER = 11          # NLCD land-cover class "open water"
MIN_REGION = 20     # smallest region, in points (the paper's m)
COLUMNS = ["seed", "component", "n", "positives", "n_land", "positives_land", "dropped_water",
           "pos_share_land", "tiles", "counties", "dfirms", "type"]


def is_land(landcover: pd.Series) -> pd.Series:
    """A class other than open water; a missing class is not land."""
    return landcover.notna() & (landcover != WATER)


def region_type(n_land: int, positives_land: int) -> str:
    if n_land == 0:
        return "no land points"
    if positives_land / n_land >= 0.5:
        return "positive-dominated"
    return "negative-dominated"


def joined(values) -> str:
    """The distinct values, sorted and separated by spaces."""
    return " ".join(sorted(set(map(str, values))))


def region_rows(points: pd.DataFrame, landcover: pd.Series, seed: int,
                min_size: int = MIN_REGION) -> list[dict]:
    """One row per region of one seed's points table; landcover is indexed by point_id."""
    regions = points[points.component_size >= min_size].copy()
    missing = ~regions.point_id.isin(landcover.index)
    if missing.any():
        raise SystemExit(f"seed {seed}: {int(missing.sum())} region points are not in the "
                         f"dataset, e.g. {regions.point_id[missing].iloc[0]}")
    regions["land"] = is_land(landcover.reindex(regions.point_id)).to_numpy()
    regions["county"] = point_county(regions).to_numpy()
    rows = []
    for component, region in regions.groupby("component", sort=True):
        land = region[region.land]
        n_land = len(land)
        positives_land = int((land.label == 1).sum())
        share = positives_land / n_land if n_land else np.nan
        rows.append(dict(seed=seed, component=int(component), n=len(region),
                         positives=int((region.label == 1).sum()), n_land=n_land,
                         positives_land=positives_land, dropped_water=len(region) - n_land,
                         pos_share_land=share, tiles=joined(region.tile_50km),
                         counties=joined(region.county), dfirms=joined(region.dfirm_id),
                         type=region_type(n_land, positives_land)))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regions with their land points and type.")
    ap.add_argument("--regions-dir", type=Path, required=True)
    ap.add_argument("--cell-prefix", default="FL_band_AE")
    ap.add_argument("--variant", default="spatial_mu3_knn10")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--min-size", type=int, default=MIN_REGION)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    landcover = pd.read_parquet(args.dataset, columns=["point_id", "landcover"])
    landcover = landcover.set_index("point_id").landcover
    rows = []
    for seed in args.seeds:
        name = f"{args.cell_prefix}_s{seed}_{args.variant}_S{args.budget}_points.parquet"
        rows += region_rows(pd.read_parquet(args.regions_dir / name), landcover, seed,
                            args.min_size)
    table = pd.DataFrame(rows, columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    counts = table.type.value_counts()
    print(f"\n{len(table)} regions of >= {args.min_size} points: "
          + ", ".join(f"{n} {kind}" for kind, n in counts.items()) + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
