#!/usr/bin/env python3
"""
Checks for variogram_ranges.py. Run from the repo root:

    .venv/bin/python aotil/test_variogram_ranges.py

1. The models: the spherical one reaches its sill at a and stays there; the exponential one reaches
   95% of its partial sill at a ln 20; effective_range gives a ln 20 and 0.811a.
2. empirical by hand: three points on a line, every pair once in its 1 km lag; a missing value
   drops its pairs for that column only.
3. A field with a known exponential covariance (range parameter 6 km, so an effective range of
   18 km) on 2,500 random points: the fitted effective range comes back within 40%, and a pure
   nugget field (no spatial structure) gets range 0, marked as a pure nugget.
4. columns_for: numeric features as they are; one-hot columns of the categorical features, those
   under 1% prevalence left out.
5. run end to end on a synthetic dataset: one row per column and per feature; a categorical
   feature's range is the largest of its columns; the summary's maximum is the largest feature.
"""
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variogram_ranges as vr  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def check_models():
    print("1. models")
    h = np.array([0.0, 5_000.0, 10_000.0, 20_000.0])
    sph = vr.spherical(h, 0.1, 1.0, 10_000.0)
    check(np.isclose(sph[0], 0.1) and np.isclose(sph[2], 1.1) and np.isclose(sph[3], 1.1),
          "spherical: the nugget at 0, the sill at a and beyond")
    a = 4_000.0
    r = vr.effective_range("exponential", a)
    check(np.isclose(r, a * np.log(20)) and np.isclose(vr.exponential(r, 0.2, 1.0, a), 0.2 + 0.95),
          f"exponential: 95% of the partial sill at a ln 20 = {r / a:.3f} a")
    rs = vr.effective_range("spherical", 10_000.0)
    check(abs(rs / 10_000.0 - 0.8113) < 1e-3 and np.isclose(vr.spherical(rs, 0.0, 1.0, 1e4), 0.95),
          f"spherical: 95% at {rs / 1e4:.4f} a")


def check_empirical():
    print("2. empirical by hand")
    xy = np.array([[0.0, 0.0], [500.0, 0.0], [1_500.0, 0.0]])
    Z = np.array([[0.0, 0.0], [1.0, np.nan], [3.0, 3.0]])
    h, gamma, pairs = vr.empirical(xy, Z, lag=1_000.0, max_lag=3_000.0, chunk=2)
    check(np.allclose(h, [500, 1500, 2500]), "lag centres")
    check(np.isclose(gamma[0, 0], 0.5) and np.isclose(gamma[1, 0], (9 + 4) / 4)
          and pairs[:, 0].tolist() == [1, 2, 0], "column 1: (1)/2 at 0.5 km, (9 + 4)/4 at 1.5 km")
    check(pairs[:, 1].tolist() == [0, 1, 0] and np.isclose(gamma[1, 1], 9 / 2),
          "column 2: the missing value drops its two pairs, not the third")


def field(n, a, seed, nugget_only=False):
    """n points on a 200 km square and a Gaussian field with exponential covariance exp(-h/a)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 200_000, (n, 2))
    if nugget_only:
        return xy, rng.normal(0, 1, n)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    L = np.linalg.cholesky(np.exp(-d / a) + 1e-8 * np.eye(n))
    return xy, L @ rng.normal(0, 1, n)


def check_recovery():
    print("3. a known range")
    xy, z = field(2_500, 6_000.0, 3)
    h, gamma, pairs = vr.empirical(xy, z[:, None])
    best, fits = vr.best_fit(h, gamma[:, 0], pairs[:, 0])
    got = fits["exponential"]["effective_range_m"]
    check(abs(got / 18_000.0 - 1) < 0.4,
          f"exponential field, effective range 18 km: fitted {got / 1000:.1f} km (best "
          f"{best['model']}, {best['effective_range_m'] / 1000:.1f} km)")
    xy, z = field(2_500, 1.0, 4, nugget_only=True)
    h, gamma, pairs = vr.empirical(xy, z[:, None])
    best, _ = vr.best_fit(h, gamma[:, 0], pairs[:, 0])
    check(best["effective_range_m"] == 0.0 and "nugget" in best["note"],
          f"no structure: range 0, a pure nugget (partial sill {best['psill']:.3f}, nugget "
          f"{best['nugget']:.3f})")


def synthetic_dataset(n=3_000, seed=5):
    rng = np.random.default_rng(seed)
    xy, smooth = field(n, 8_000.0, seed)
    df = gpd.GeoDataFrame({"point_id": [f"P{i}" for i in range(n)],
                           "slope": smooth + rng.normal(0, 0.1, n),
                           "hand": rng.normal(0, 1, n),
                           "landcover": np.where(smooth > 0.3, 90.0, np.where(smooth < -1, 42.0,
                                                                               21.0)),
                           "soil_fluvclass": np.where(rng.random(n) < 0.005, 1.0, 0.0)},
                          geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=5070)
    df.loc[rng.choice(n, 50, replace=False), "hand"] = np.nan
    return df


def check_columns():
    print("4. columns_for")
    df = synthetic_dataset()
    cols, owner = vr.columns_for(df, ["landcover", "slope", "hand", "soil_fluvclass"])
    check("slope" in cols and "hand" in cols and cols.hand.isna().sum() == 50,
          "numeric features as they are, missing values kept")
    check({c for c in cols if c.startswith("landcover_")} == {"landcover_21", "landcover_42",
                                                              "landcover_90"}
          and "soil_fluvclass_0" in cols and "soil_fluvclass_1" not in cols,
          "one-hot columns, the class under 1% prevalence left out")
    check(owner["landcover_90"] == "landcover" and owner["slope"] == "slope", "column -> feature")


def check_run(tmp: Path):
    print("5. run")
    df = synthetic_dataset()
    df.to_parquet(tmp / "dataset_FL.parquet")
    (tmp / "FEATURES.txt").write_text("landcover\nslope\nhand\n")
    out = vr.run(tmp, n_sample=2_000, seed=0)
    ranges, features = out["ranges"], out["features"]
    check(len(ranges) == 5 and features.feature.tolist() == ["landcover", "slope", "hand"],
          f"5 columns, 3 features: {ranges.column.tolist()}")
    lc = ranges[ranges.feature == "landcover"].range_km.max()
    check(np.isclose(features.set_index("feature").range_km["landcover"], lc),
          "a categorical feature takes its widest column's range")
    check(np.isclose(out["summary"]["max_range_km"], features.range_km.max())
          and out["summary"]["sample"] == 2_000, "the summary's maximum and sample size")
    check(ranges.set_index("column").range_km["slope"] > ranges.set_index("column").range_km["hand"],
          "the smooth feature has the longer range than the noise feature")


def main():
    check_models()
    check_empirical()
    check_recovery()
    check_columns()
    with tempfile.TemporaryDirectory() as tmp:
        check_run(Path(tmp))
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
