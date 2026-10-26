#!/usr/bin/env python3
"""
Checks for regions_land.py. Run from the repo root:

    python aotil/test_regions_land.py

1. A synthetic points table: regions below --min-size are left out; land excludes open water and
   a missing class; the type follows the land points' majority label, with an exact half counted
   positive-dominated and a region without land typed "no land points"; counties come from
   meta_point_county, falling back to county_fips per point.
2. A region point missing from the dataset stops the run.
3. From the 2026-10-16 regions and the archived v2 dataset, the script reproduces
   results/2026-10-17/o5/regions_land.csv byte for byte. Skipped when those files are missing.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regions_land as rl  # noqa: E402

FAILS = []
OLD_REGIONS = Path("results/2026-10-16/regions")
OLD_DATASET = Path("data/interim/archive/sampling-v2/data/processed/dataset_FL.parquet")
OLD_TABLE = Path("results/2026-10-17/o5/regions_land.csv")


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


def component(c, n, labels, landcover, tiles, county, county_fips, dfirm):
    return pd.DataFrame({
        "point_id": [f"c{c}_{i}" for i in range(n)], "component": c, "component_size": n,
        "label": labels, "tile_50km": tiles, "meta_point_county": county,
        "county_fips": county_fips, "dfirm_id": dfirm, "_landcover": landcover})


def synthetic():
    """Four components: mixed land (positive-dominated), all water, a tie, and one too small."""
    c0 = component(0, 25, [1] * 14 + [0] * 11, [42] * 20 + [11] * 3 + [np.nan] * 2,
                   ["9_10"] * 10 + ["10_9"] * 15, ["12001"] * 24 + [pd.NA], "12003",
                   ["12001C"] * 24 + ["12003C"])
    c1 = component(1, 22, [1] * 5 + [0] * 17, [11] * 22, "30_11", "12093", "12093", "12093C")
    c2 = component(2, 20, [1] * 10 + [0] * 10, [90] * 20, "26_17", "12007", "12007", "12007C")
    c3 = component(3, 19, [1] * 19, [42] * 19, "26_17", "12007", "12007", "12007C")
    points = pd.concat([c2, c0, c3, c1], ignore_index=True)
    # land among the first 23 of c0's points: 20 class 42, of which the first 14 are positives
    landcover = points.set_index("point_id")._landcover
    return points.drop(columns="_landcover"), landcover


def check_synthetic():
    print("1. synthetic regions")
    points, landcover = synthetic()
    table = pd.DataFrame(rl.region_rows(points, landcover, seed=4), columns=rl.COLUMNS)
    check(table.component.tolist() == [0, 1, 2], "components below 20 points are left out, "
          "rows in component order")
    c0 = table.iloc[0]
    check((c0.n, c0.positives, c0.n_land, c0.positives_land, c0.dropped_water) ==
          (25, 14, 20, 14, 5), "open water and a missing class are not land")
    check(np.isclose(c0.pos_share_land, 0.7) and c0.type == "positive-dominated",
          "majority of land points flood-labeled: positive-dominated")
    check(c0.tiles == "10_9 9_10" and c0.counties == "12001 12003" and
          c0.dfirms == "12001C 12003C",
          "tiles, counties and DFIRMs sorted; a null meta_point_county falls back to county_fips")
    c1 = table.iloc[1]
    check(c1.n_land == 0 and np.isnan(c1.pos_share_land) and c1.type == "no land points",
          "a region on open water only has no land points")
    check(table.iloc[2].type == "positive-dominated", "an exact half counts positive-dominated")
    check((table.seed == 4).all() and list(table.columns) == rl.COLUMNS,
          "seed column and column order")

    print("2. a region point missing from the dataset")
    try:
        rl.region_rows(points, landcover.drop(points.point_id.iloc[0]), seed=0)
        check(False, "stops the run")
    except SystemExit as err:
        check("not in the dataset" in str(err), "stops the run")


def check_reproduction():
    print("3. reproduce results/2026-10-17/o5/regions_land.csv")
    if not (OLD_REGIONS.is_dir() and OLD_DATASET.exists() and OLD_TABLE.exists()):
        print("  skipped: the 2026-10-16 regions, the archived v2 dataset or the 2026-10-17 "
              "table is missing")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "regions_land.csv"
        rl.main(["--regions-dir", str(OLD_REGIONS), "--seeds", "0", "1", "2",
                 "--dataset", str(OLD_DATASET), "--out", str(out)])
        same = out.read_bytes() == OLD_TABLE.read_bytes()
    check(same, "byte for byte, 18 rows")


if __name__ == "__main__":
    check_synthetic()
    check_reproduction()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall checks pass")
    sys.exit(1 if FAILS else 0)
