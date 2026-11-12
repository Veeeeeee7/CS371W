#!/usr/bin/env python3
"""
Checks for the terrain variability of pull_gee.py (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D3)
and its merge into another folder. Run from the repo root:

    .venv/bin/python test_pull_gee.py [--online]

Offline (no network):
1. sample_group's request, against a stand-in for the Earth Engine client: a group with buffer_m
   buffers every point by that radius (to BUFFER_ERROR_M) and reduces with its own reducer on the
   group's grid; a plain group reads the point with ee.Reducer.first().
2. derive: tri_300 (the SD inside the fixed 300 m cell) becomes meta_tri_300, elev_sd_r300 stays a
   feature, FEATURES names elev_sd_r300 and not tri_300.
3. merge --processed: the dataset and FEATURES.txt go to the given folder, data/processed is not
   touched, and stale shards are refused.
Online (--online, needs an Earth Engine login; skipped otherwise):
4. The 300 m radius on a synthetic DEM: on a ramp whose elevation is the EPSG:5070 x coordinate in
   metres, the standard deviation within a circle of radius r is r / 2 (150 m for 300 m), against
   300 / sqrt(12) = 86.6 m inside a 300 m square cell and 75 m for a circle of diameter 300 m.
   Then, on a flat synthetic DEM, 0.
"""
import argparse
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd

import pull_gee as pg

FAILS = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


# ----------------------------------------------------------------- 1: the request
class FakeGeometry:
    def __init__(self, coords, radius=None, error=None):
        self.coords, self.radius, self.error = coords, radius, error

    def buffer(self, radius, error):
        return FakeGeometry(self.coords, radius, error)


class FakeImage:
    """Records reduceRegions' arguments and answers with one value per feature."""

    def __init__(self):
        self.calls = []

    def reduceRegions(self, collection, reducer, crs, tileScale):
        self.calls.append(dict(collection=collection, reducer=reducer, crs=crs,
                               tileScale=tileScale))
        rows = [{"properties": {"pid": f.props["pid"], "elev_sd_r300": 1.5, "slope": 2.0}}
                for f in collection.features]
        return types.SimpleNamespace(getInfo=lambda: {"features": rows})


def fake_ee():
    feature = types.SimpleNamespace
    return types.SimpleNamespace(
        Geometry=types.SimpleNamespace(Point=lambda xy: FakeGeometry(tuple(xy))),
        Feature=lambda geometry, props: feature(geometry=geometry, props=props),
        FeatureCollection=lambda features: types.SimpleNamespace(features=features),
        Reducer=types.SimpleNamespace(first=lambda: "first"))


def check_request():
    print("1. sample_group's request")
    with mock.patch.object(pg, "ee", fake_ee()):
        image = FakeImage()
        group = dict(name="sd300", image=image, proj="native", bands=["elev_sd_r300"],
                     buffer_m=pg.SD_RADIUS_M, reducer="stdDev->elev_sd_r300")
        got = pg.sample_group(group, ["p1", "p2"], np.array([-82.0, -81.5]),
                              np.array([29.0, 29.5]))
        call = image.calls[0]
        geoms = [f.geometry for f in call["collection"].features]
        check(all(g.radius == 300.0 and g.error == pg.BUFFER_ERROR_M for g in geoms)
              and geoms[0].coords == (-82.0, 29.0),
              "each point buffered by 300 m (to 1 m), at its lon/lat")
        check(call["reducer"] == "stdDev->elev_sd_r300" and call["crs"] == "native",
              "reduced with the group's reducer on the group's grid")
        check(got.point_id.tolist() == ["p1", "p2"] and got.elev_sd_r300.tolist() == [1.5, 1.5],
              "one row per point, keyed by point_id")
        plain = FakeImage()
        pg.sample_group(dict(name="fine", image=plain, proj="native", bands=["slope"]),
                        ["p1"], np.array([-82.0]), np.array([29.0]))
        check(plain.calls[0]["reducer"] == "first"
              and plain.calls[0]["collection"].features[0].geometry.radius is None,
              "a group without buffer_m reads the point's own cell (ee.Reducer.first())")


# ----------------------------------------------------------------- 2: derive
def check_derive():
    print("2. derive")
    raw = pd.DataFrame({"elev": [3.0], "slope": [1.0], "aspect": [90.0], "tri_300": [0.4],
                        "elev_sd_r300": [0.6], "log1p_upa": [1.0], "block_id": ["10_20"]})
    out = pg.derive(raw)
    check("tri_300" not in out and out.meta_tri_300.tolist() == [0.4]
          and out.elev_sd_r300.tolist() == [0.6],
          "tri_300 -> meta_tri_300 (comparison only); elev_sd_r300 kept as is")
    check("elev_sd_r300" in pg.FEATURES and "tri_300" not in pg.FEATURES
          and "elev_sd_r300" in pg.ALL_BANDS and "tri_300" in pg.ALL_BANDS,
          "FEATURES names elev_sd_r300; both bands are pulled")
    check(out.tile_50km.tolist() == ["2_4"], "tiles still derived from block_id")


# ----------------------------------------------------------------- 3: merge --processed
def check_merge(tmp: Path):
    print("3. merge --processed")
    pts_dir, gee_dir, out = tmp / "points", tmp / "gee", tmp / "v4_build"
    (gee_dir / "FL").mkdir(parents=True)
    pts_dir.mkdir()
    points = gpd.GeoDataFrame(
        {"point_id": ["FL0000000", "FL0000001"], "lon": [-82.0, -81.5], "lat": [29.0, 29.5],
         "block_id": ["100_60", "101_60"]},
        geometry=gpd.points_from_xy([1.3e6, 1.31e6], [7e5, 7e5]), crs=5070)
    points.to_parquet(pts_dir / "points_FL.parquet")
    shard = pd.DataFrame({"point_id": points.point_id, "pt_lon": points.lon,
                          "pt_lat": points.lat})
    for band in pg.ALL_BANDS:
        shard[band] = 1.0
    shard.to_parquet(gee_dir / "FL" / "chunk_00000.parquet", index=False)
    processed_before = sorted(p.name for p in Path("data/processed").glob("*")) \
        if Path("data/processed").exists() else []
    with mock.patch.multiple(pg, PTS_DIR=pts_dir, GEE_DIR=gee_dir):
        code = pg.main(["--merge", "FL", "--processed", str(out)])
    data = pd.read_parquet(out / "dataset_FL.parquet")
    names = pg.FEATURES
    check(code == 0 and len(data) == 2 and "elev_sd_r300" in data and "meta_tri_300" in data
          and (out / "FEATURES.txt").read_text().count("elev_sd_r300") == 1,
          f"dataset and FEATURES.txt written to {out.name}, with elev_sd_r300 and meta_tri_300")
    after = sorted(p.name for p in Path("data/processed").glob("*")) \
        if Path("data/processed").exists() else []
    check(after == processed_before, "data/processed untouched")
    patched = tmp / "patched"
    real = Path("data/processed/dataset_FL.parquet")
    real_before = real.read_bytes() if real.exists() else None
    with mock.patch.multiple(pg, PTS_DIR=pts_dir, GEE_DIR=gee_dir, OUT_DIR=patched):
        pg.merge(["FL"])
        pg.main(["--merge", "FL"])
    check((patched / "dataset_FL.parquet").exists()
          and (real.read_bytes() if real.exists() else None) == real_before,
          "without --processed, merge() and main() write to OUT_DIR as it is when called "
          "(a patched folder), never to data/processed")
    shard.assign(pt_lon=shard.pt_lon + 0.01).to_parquet(gee_dir / "FL" / "chunk_00000.parquet",
                                                        index=False)
    with mock.patch.multiple(pg, PTS_DIR=pts_dir, GEE_DIR=gee_dir):
        code = pg.main(["--merge", "FL", "--processed", str(out)])
    check(code == 1, "shards from other locations are refused")
    check(len(names) == 13, "13 FEATURES as before (elev_sd_r300 in tri_300's place)")


# ----------------------------------------------------------------- 4: Earth Engine
def check_online():
    print("4. the 300 m radius on a synthetic DEM (Earth Engine)")
    ee = pg.ee
    try:
        ee.Initialize(project=os.environ.get("EE_PROJECT", "cs371w-gis"))
        ee.Number(1).getInfo()
    except Exception as e:
        print(f"  skip  no Earth Engine login ({str(e)[:80]})")
        return
    albers = ee.Projection("EPSG:5070")
    grid = albers.atScale(10)
    ramp = ee.Image.pixelCoordinates(albers).select("x").setDefaultProjection(grid)
    lon = np.array([-82.3, -81.0, -84.5])
    lat = np.array([29.6, 27.5, 30.4])
    ids = ["a", "b", "c"]
    got = pg.sample_group(pg.terrain_variability_group(ramp, grid), ids, lon, lat)
    sd = got.elev_sd_r300.to_numpy(float)
    check(np.allclose(sd, 150.0, atol=2.0),
          f"ramp z = x (m): SD within 300 m = {np.round(sd, 2).tolist()} (want 150 = r/2; a 300 m "
          f"cell gives 86.6, a 300 m diameter 75)")
    flat = ee.Image.constant(5).setDefaultProjection(grid)
    zero = pg.sample_group(pg.terrain_variability_group(flat, grid), ids, lon, lat)
    check(np.allclose(zero.elev_sd_r300.to_numpy(float), 0.0, atol=1e-6), "flat DEM: SD 0")
    small = pg.sample_group(pg.terrain_variability_group(ramp, grid, radius_m=100.0), ids,
                            lon, lat)
    check(np.allclose(small.elev_sd_r300.to_numpy(float), 50.0, atol=1.5),
          f"radius 100 m: {np.round(small.elev_sd_r300.to_numpy(float), 2).tolist()} (want 50)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="also run the Earth Engine check")
    a = ap.parse_args()
    check_request()
    check_derive()
    with tempfile.TemporaryDirectory() as tmp:
        check_merge(Path(tmp))
    if a.online:
        check_online()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
