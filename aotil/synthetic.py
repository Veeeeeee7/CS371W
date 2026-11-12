#!/usr/bin/env python3
"""Small synthetic inputs for the checks (aotil/test_*.py, test_entry_points.py); never used by a run.

  dataset(folder)   a Florida-like point table with the paper's nine features and every column the stages read:
                    dataset_FL.parquet (GeoParquet, EPSG:5070), FEATURES.txt, and counties.parquet (GEOID, NAME,
                    EPSG:4269 polygons). 30 tiles of 50 km on the EPSG:5070 grid, both zones and labels in each;
                    one corner of the Zone A points is labelled against the features, so a subset can invert.
  cell(folder)      a cell (npz + models.csv) without fitting any model: n_models models whose correctness on the OOD
                    points follows their ID accuracy, except on a block of points where it runs the other way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NINE = ["landcover", "soil_floodfreq", "soil_fluvclass", "slope", "elev_sd_r300", "hand", "wetindexcat",
        "log10_dist_river", "log10_dist_coast"]
X0, Y0 = 1_300_000, 600_000          # EPSG:5070 metres of the grid's corner (central Florida)
COUNTY_FIPS = ["12001", "12003", "12005", "12007", "12009", "12011"]
COUNTY_NAMES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]


def dataset(folder: Path, seed: int = 0, per_cell: int = 40) -> Path:
    """Write the synthetic dataset folder; returns it."""
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows, pid = [], 0
    for ti in range(6):
        for tj in range(5):
            for dom, lab in (("AE", 1), ("AE", 0), ("A", 1), ("A", 0)):
                cx, cy = X0 + ti * 50_000 + rng.uniform(5_000, 45_000), Y0 + tj * 50_000 + rng.uniform(5_000, 45_000)
                for _ in range(per_cell):
                    x = float(np.clip(cx + rng.normal(0, 4_000), X0 + ti * 50_000 + 10, X0 + (ti + 1) * 50_000 - 10))
                    y = float(np.clip(cy + rng.normal(0, 4_000), Y0 + tj * 50_000 + 10, Y0 + (tj + 1) * 50_000 - 10))
                    rows.append(dict(point_id=f"FL{pid:07d}", x=x, y=y, domain=dom, label=lab,
                                     neg_protocol=None if lab == 1 else "band", ti=ti, tj=tj))
                    pid += 1
    df = pd.DataFrame(rows)
    lon, lat = Transformer.from_crs(5070, 4269, always_xy=True).transform(df.x.to_numpy(), df.y.to_numpy())
    df["lon"], df["lat"] = lon, lat
    df["tile_50km"] = (np.floor(df.x / 50_000).astype(int).astype(str) + "_"
                       + np.floor(df.y / 50_000).astype(int).astype(str))
    n = len(df)
    y = df.label.to_numpy()
    flip = ((df.domain == "A") & (df.ti <= 1) & (df.tj <= 1)).to_numpy()   # Zone A labelled against the inputs
    sig = np.where(flip, 1 - y, y)
    df["landcover"] = np.where(sig == 1, rng.choice([90, 95, 42], n), rng.choice([42, 81, 21, 22, 82], n))
    df["soil_floodfreq"] = np.where(rng.random(n) < 0.03, np.nan, np.where(sig == 1, rng.choice([0., 4., 5.], n),
                                                                           rng.choice([0., 1., 2.], n)))
    df["soil_fluvclass"] = np.where(np.isnan(df.soil_floodfreq), np.nan, (rng.random(n) < 0.05 + 0.1 * sig) * 1.0)
    df["slope"] = np.abs(rng.normal(1.0 - 0.5 * sig, 0.5, n))
    df["elev_sd_r300"] = np.abs(rng.normal(1.0 - 0.4 * sig, 0.4, n))
    df["hand"] = np.abs(rng.normal(3.0 - 2.0 * sig, 1.5, n))
    df["wetindexcat"] = rng.normal(1100 + 200 * sig, 150, n)
    df["meta_dist_to_river_m"] = rng.gamma(2.0, 800 - 500 * sig, n)
    df["meta_dist_to_coast_m"] = rng.gamma(3.0, 10_000, n)
    df["log10_dist_river"] = np.log10(np.maximum(df.meta_dist_to_river_m, 1.0))
    df["log10_dist_coast"] = np.log10(np.maximum(df.meta_dist_to_coast_m, 1.0))
    county = COUNTY_FIPS
    df["meta_point_county"] = [county[ti] for ti in df.ti]
    df["county_fips"] = df.meta_point_county
    df["dfirm_id"] = df.meta_point_county + "C"
    df["meta_fld_ar_id"] = df.dfirm_id + "_" + (np.arange(n) // 7).astype(str)
    df["meta_src_row_id"] = np.arange(n) // 7
    df["block_id"] = "B" + df.tile_50km
    df["meta_comid"] = np.arange(n, dtype=float)
    df["meta_source_cit"] = "FEMA"
    df["state"] = "FL"
    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y", "ti", "tj"]), geometry=gpd.points_from_xy(df.x, df.y),
                           crs=5070)
    gdf.to_parquet(folder / "dataset_FL.parquet")
    (folder / "FEATURES.txt").write_text("# synthetic: the paper's nine features\n" + "\n".join(NINE) + "\n")
    polys = [box(X0 + ti * 50_000, Y0 - 20_000, X0 + (ti + 1) * 50_000, Y0 + 270_000) for ti in range(6)]
    counties = gpd.GeoDataFrame(dict(GEOID=COUNTY_FIPS, NAME=COUNTY_NAMES), geometry=polys, crs=5070).to_crs(4269)
    counties.to_parquet(folder / "counties.parquet")
    return folder


def cell(folder: Path, n_models: int = 60, d: int = 600, n_id: int = 400, seed: int = 0,
         tag: str = "FL_band_AE_s0") -> Path:
    """Write a synthetic cell to folder; models whose OOD correctness tracks ID accuracy except on points 0..39,
    where the better models are worse (an inversion a search can find). Points sit on a 30 x 20 grid of 1 km."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    skill = np.sort(rng.uniform(0.55, 0.9, n_models))
    y_ood = rng.integers(0, 2, d)
    y_id = rng.integers(0, 2, n_id)
    hard = np.zeros(d, bool)
    hard[:40] = True
    p_ok = np.where(hard[None, :], 1.0 - skill[:, None], skill[:, None])
    ok_ood = rng.random((n_models, d)) < p_ok
    ok_id = rng.random((n_models, n_id)) < skill[:, None]
    S_ood = np.where(ok_ood, rng.uniform(0.55, 1.0, (n_models, d)), rng.uniform(0.0, 0.45, (n_models, d)))
    S_id = np.where(ok_id, rng.uniform(0.55, 1.0, (n_models, n_id)), rng.uniform(0.0, 0.45, (n_models, n_id)))
    gx, gy = np.meshgrid(np.arange(30), np.arange(20))
    lon = -82.0 + gx.ravel()[:d] * 0.01
    lat = 28.0 + gy.ravel()[:d] * 0.009
    tiles = np.array([f"{int(a // 10)}_{int(b // 10)}" for a, b in zip(gx.ravel()[:d], gy.ravel()[:d])])
    np.savez_compressed(folder / f"{tag}.npz", S_ood=S_ood, S_id=S_id, y_ood=y_ood, y_id=y_id,
                        tiles_ood=tiles, point_id=np.array([f"P{i:05d}" for i in range(d)]), lon=lon, lat=lat,
                        tile_50km=tiles, point_id_id=np.array([f"Q{i:05d}" for i in range(n_id)]))
    pd.DataFrame(dict(idx=np.arange(n_models), kind=rng.choice(["rf", "lr", "nb"], n_models),
                      acc_id=ok_id.mean(1), acc_ood=ok_ood.mean(1), auc_id=skill, auc_ood=skill)).to_csv(
        folder / f"{tag}_models.csv", index=False)
    return folder
