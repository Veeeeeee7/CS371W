#!/usr/bin/env python3
"""
Checks for pull_hydro.py's --processed (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md Part C: the v4
build merges into data/interim/v4_build, never data/processed). Run from the repo root:

    .venv/bin/python test_pull_hydro.py

1. set_processed points the dataset, FEATURES.txt and the merge lock at the folder.
2. --merge --processed DIR reads and rewrites DIR/dataset_FL.parquet and DIR/FEATURES.txt only,
   adding the three features (wetindexcat, log10_dist_river, log10_dist_coast), meta_comid and the
   raw distances meta_dist_to_river_m and meta_dist_to_coast_m.
3. Without --processed, main leaves PROC as it is: a caller that set PROC (as the private tests
   patch it) keeps it, so nothing is written to data/processed.
4. data/processed is byte-identical before and after.
"""
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest import mock

import geopandas as gpd
import pandas as pd

import pull_hydro as ph

FAILS = []
REAL = [Path("data/processed/dataset_FL.parquet"), Path("data/processed/FEATURES.txt")]


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def digest(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def write_world(folder: Path, hydro: Path):
    folder.mkdir(parents=True)
    hydro.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame({"point_id": ["FL0000000", "FL0000001"], "lon": [-82.0, -81.9],
                            "lat": [29.0, 29.1], "county_fips": ["12001", "12001"]},
                           geometry=gpd.points_from_xy([1.3e6, 1.31e6], [7.0e5, 7.0e5]), crs=5070)
    pts.to_parquet(folder / "dataset_FL.parquet")
    (folder / "FEATURES.txt").write_text("slope\n")
    pd.DataFrame({"point_id": pts.point_id, "comid": [11.0, 12.0]}).to_parquet(
        hydro / "comid_FL.parquet")
    pd.DataFrame({"point_id": pts.point_id, "dist_to_river_m": [0.5, 100.0],
                  "dist_to_coast_m": [1000.0, 10.0]}).to_parquet(hydro / "river_FL.parquet")
    pd.DataFrame({"comid": [11.0, 12.0], "wetindexcat": [550.0, 600.0]}).to_parquet(
        hydro / "streamcat_FL.parquet")


def main():
    before = digest(REAL)
    saved = (ph.PROC, ph.LOCK)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("1. set_processed")
        ph.set_processed(tmp / "x")
        check(ph.PROC == tmp / "x" and ph.LOCK == tmp / "x" / ".merge.lock",
              "PROC and the merge lock follow the folder")
        ph.PROC, ph.LOCK = saved

        print("2. --merge --processed")
        stage, hydro = tmp / "stage", tmp / "hydro"
        write_world(stage, hydro)
        with mock.patch.object(ph, "HYDRO", hydro):
            code = ph.main(["--merge", "FL", "--processed", str(stage)])
        out = pd.read_parquet(stage / "dataset_FL.parquet")
        check(code == 0 and out.wetindexcat.tolist() == [550.0, 600.0]
              and "wetindexcat" in (stage / "FEATURES.txt").read_text(),
              "the staging dataset and its FEATURES.txt gain the hydro columns")
        added = set(out) - {"point_id", "lon", "lat", "county_fips", "geometry"}
        check(added == {"meta_comid", "wetindexcat", "meta_dist_to_river_m",
                        "meta_dist_to_coast_m", "log10_dist_river", "log10_dist_coast"}
              and out.log10_dist_river.tolist() == [0.0, 2.0]
              and out.log10_dist_coast.tolist() == [3.0, 1.0],
              f"exactly the three features, meta_comid and the raw distances; log10 floored at "
              f"1 m: {sorted(added)}")
        ph.PROC, ph.LOCK = saved

        print("3. without --processed")
        other = tmp / "patched"
        write_world(other, hydro)
        with mock.patch.multiple(ph, HYDRO=hydro, PROC=other, LOCK=other / ".merge.lock"):
            code = ph.main(["--merge", "FL"])
            kept = ph.PROC
        check(code == 0 and kept == other
              and "wetindexcat" in pd.read_parquet(other / "dataset_FL.parquet"),
              "a PROC set by the caller is kept and merged into")
    print("4. data/processed")
    check(digest(REAL) == before and ph.PROC == ph.DEFAULT_PROC,
          "data/processed byte-identical; PROC back to data/processed")
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
