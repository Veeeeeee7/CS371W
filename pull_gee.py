#!/usr/bin/env python3
"""
pull_gee.py -- attach Earth Engine covariates to the sampled points.

Input   data/interim/points/points_{ST}.parquet   (878,336 points across 10 states)
Output  data/interim/gee/{ST}/chunk_{i:05d}.parquet   resumable shards
        data/processed/dataset_{ST}.parquet           after --merge

DESIGN NOTES
============

FOUR REQUESTS PER CHUNK, NOT ONE.  The bands come from rasters at 10 m, 30 m and
90 m. Sampling them all in one `reduceRegions(scale=10)` forces Earth Engine to
evaluate the 300 m and 900 m neighbourhood reductions on a 10 m grid, which is
both slow and the thing that makes requests time out. Each resolution group is
sampled at its own native scale and joined on `point_id` afterwards, so a failure
in one group does not cost the other three.

EVERY FEATURE IS CONSTANT-FREE.  TWI and SPI are absent on purpose: TWI's slope
floor decides the value at exactly the flat floodplain cells that carry the
positive label (Winooski bottom: tan(slope)=0.000611, floored to 0.001, so
TWI=11.08 came from the constant), and MERIT's upstream area is sub-cell at two of
three test points. See docs/COVARIATES.md section 4. Neighbourhood terrain is
computed by coarsening and differencing, which introduces a scale but no tunable
threshold, and both scales are reported so the choice is visible.

MULTI-SCALE TERRAIN.  Flood exposure is a neighbourhood property, not a pixel one:
a cell 2 m above a wide valley floor floods differently from one 2 m above a
narrow gully with identical slope. `tpi_300` (10 m DEM minus its 300 m mean) picks
up local position; `tpi_broad` (MERIT 90 m minus its 900 m mean) picks up position
within the valley. `tri_300` is the standard deviation of elevation in the same
300 m window.

WATER PROXIMITY IS A STOPGAP.  `water_frac_900m` -- the fraction of a 900 m
neighbourhood that JRC Global Surface Water has ever seen as water -- stands in
until NHDPlus V2 lands and a true distance-to-flowline is available. It is a
fraction, not a distance, so it saturates near large rivers; label it accordingly
in any result that leans on it.

RESUME AND FAILURE.  Each chunk writes its own parquet and finished chunks are
skipped, so an interrupted run costs only the chunk in flight. Earth Engine throws
transient quota and timeout errors under concurrency; every request retries with
exponential backoff and a failed chunk is recorded rather than killing the run.

USAGE
=====
    python pull_gee.py --selftest            # 3 Vermont points, prints the table
    python pull_gee.py VT                    # one state
    python pull_gee.py                       # all ten (about 1-2 hours)
    python pull_gee.py --workers 4           # back off if EE returns quota errors
    python pull_gee.py --merge               # shards -> data/processed/dataset_{ST}.parquet
    python pull_gee.py --status              # how far along, per state
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    sys.exit("pip install earthengine-api")

PTS_DIR = Path("data/interim/points")
GEE_DIR = Path("data/interim/gee")
OUT_DIR = Path("data/processed")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]

CHUNK = 2000        # points per request
WORKERS = 6         # concurrent EE requests
RETRIES = 5

SELFTEST_POINTS = [
    ("winooski bottom", -73.1550, 44.4900),
    ("hillslope",       -73.1805, 44.4812),
    ("upland",          -72.9851, 44.5203),
]


# =========================================================================
# the covariate stack, grouped by native resolution
# =========================================================================
def gsw_asset() -> str:
    """JRC Global Surface Water, whichever version this account can see."""
    for aid in ("JRC/GSW1_4/GlobalSurfaceWater", "JRC/GSW1_3/GlobalSurfaceWater"):
        try:
            ee.Image(aid).bandNames().getInfo()
            return aid
        except Exception:
            continue
    return ""


def build_groups() -> list[dict]:
    """Returns [{name, image, scale, bands}] -- one Earth Engine request each."""
    coll = ee.ImageCollection("USGS/3DEP/10m_collection")

    # mosaic() drops the projection and ee.Terrain silently returns null on an
    # image without one, so give the collection's native projection back first.
    native = coll.first().select("elevation").projection()
    dem = coll.mosaic().select("elevation").setDefaultProjection(native)

    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    curvature = dem.convolve(ee.Kernel.laplacian8()).rename("curvature")

    # 10 m -> 300 m is a 30x30 reduction, inside reduceResolution's pixel budget.
    p300 = native.atScale(300)
    dem300 = dem.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(p300)
    tpi300 = dem.subtract(dem300).rename("tpi_300")
    tri300 = (dem.reduceResolution(ee.Reducer.stdDev(), maxPixels=1024)
              .reproject(p300).rename("tri_300"))

    fine = (dem.rename("elev").addBands(slope).addBands(aspect)
            .addBands(curvature).addBands(tpi300).addBands(tri300))

    merit = ee.Image("MERIT/Hydro/v1_0_1")
    hnd = merit.select("hnd").rename("hand")
    upa = merit.select("upa")
    log_upa = upa.add(1).log().rename("log1p_upa")
    elv = merit.select("elv")
    p900 = elv.projection().atScale(900)          # 90 m -> 900 m, 10x10
    elv900 = elv.reduceResolution(ee.Reducer.mean(), maxPixels=256).reproject(p900)
    tpi_broad = elv.subtract(elv900).rename("tpi_broad")
    merit_img = hnd.addBands(log_upa).addBands(tpi_broad)

    nlcd = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
            .filter(ee.Filter.eq("system:index", "2021")).first()
            .select(["landcover", "impervious"]))

    groups = [
        dict(name="fine",  image=fine,      scale=10,
             bands=["elev", "slope", "aspect", "curvature", "tpi_300", "tri_300"]),
        dict(name="merit", image=merit_img, scale=90,
             bands=["hand", "log1p_upa", "tpi_broad"]),
        dict(name="nlcd",  image=nlcd,      scale=30,
             bands=["landcover", "impervious"]),
    ]

    aid = gsw_asset()
    if aid:
        gsw = ee.Image(aid)
        occ = gsw.select("occurrence").unmask(0).rename("gsw_occurrence")
        ext = gsw.select("max_extent").unmask(0)
        pw = ext.projection().atScale(900)         # 30 m -> 900 m, 30x30
        wfrac = (ext.reduceResolution(ee.Reducer.mean(), maxPixels=1024)
                 .reproject(pw).rename("water_frac_900m"))
        groups.append(dict(name="gsw", image=occ.addBands(wfrac), scale=30,
                           bands=["gsw_occurrence", "water_frac_900m"]))
    else:
        print("  WARNING: no JRC Global Surface Water asset reachable; "
              "water proximity will be missing until NHDPlus lands.")
    return groups


ALL_BANDS = ["elev", "slope", "aspect", "curvature", "tpi_300", "tri_300",
             "hand", "log1p_upa", "tpi_broad", "landcover", "impervious",
             "gsw_occurrence", "water_frac_900m"]


# =========================================================================
# request machinery
# =========================================================================
def sample_group(group: dict, ids: list[str], lon: np.ndarray,
                 lat: np.ndarray) -> pd.DataFrame:
    """One Earth Engine request. Raises on failure; the caller retries."""
    feats = [ee.Feature(ee.Geometry.Point([float(a), float(b)]), {"pid": p})
             for p, a, b in zip(ids, lon, lat)]
    fc = ee.FeatureCollection(feats)
    res = group["image"].reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=group["scale"],
        tileScale=4,                       # smaller tiles: fewer out-of-memory aborts
    ).getInfo()

    rows = []
    for f in res["features"]:
        p = f["properties"]
        rows.append({"point_id": p.get("pid"),
                     **{b: p.get(b) for b in group["bands"]}})
    return pd.DataFrame(rows)


def with_retries(fn, *a, label: str = "", **kw):
    delay = 4.0
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e)[:160].replace("\n", " ")
            if attempt == RETRIES:
                raise RuntimeError(f"{label}: {msg}") from e
            # EE's transient failures are quota, timeout and internal errors; a
            # jittered backoff keeps concurrent workers from retrying in lockstep.
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 120.0)


def do_chunk(state: str, i: int, sub: pd.DataFrame, groups: list[dict]) -> tuple[int, str]:
    out = GEE_DIR / state / f"chunk_{i:05d}.parquet"
    if out.exists():
        return i, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)

    ids = sub.point_id.tolist()
    lon = sub.lon.to_numpy()
    lat = sub.lat.to_numpy()

    merged = pd.DataFrame({"point_id": ids})
    for g in groups:
        df = with_retries(sample_group, g, ids, lon, lat,
                          label=f"{state} chunk {i} [{g['name']}]")
        merged = merged.merge(df, on="point_id", how="left")

    for b in ALL_BANDS:
        if b not in merged.columns:
            merged[b] = np.nan
    merged.to_parquet(out, index=False)
    return i, "ok"


def chunks_for(state: str) -> list[pd.DataFrame]:
    p = PTS_DIR / f"points_{state}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=["point_id", "lon", "lat", "block_id"])
    # Spatially coherent chunks: Earth Engine fetches far fewer raster tiles when
    # a request's points sit near each other, which is most of the wall clock.
    df = df.sort_values("block_id", kind="stable").reset_index(drop=True)
    return [df.iloc[i:i + CHUNK] for i in range(0, len(df), CHUNK)]


def pull_state(state: str, groups: list[dict], workers: int) -> None:
    parts = chunks_for(state)
    if not parts:
        print(f"[{state}] no points file -- skipping")
        return
    done = {int(f.stem.split("_")[1]) for f in (GEE_DIR / state).glob("chunk_*.parquet")} \
        if (GEE_DIR / state).exists() else set()
    todo = [(i, s) for i, s in enumerate(parts) if i not in done]
    n_pts = sum(len(s) for s in parts)
    print(f"\n[{state}] {n_pts:,} points, {len(parts)} chunks "
          f"({len(done)} already done, {len(todo)} to go)", flush=True)
    if not todo:
        return

    t0 = time.time()
    ok = fail = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_chunk, state, i, s, groups): i for i, s in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                _, status = fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                errors.append(str(e)[:200])
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / max(el, 1e-9)
                eta = (len(todo) - n) / max(rate, 1e-9)
                print(f"  {n}/{len(todo)} chunks  ok={ok} fail={fail}  "
                      f"{el/60:.1f} min elapsed, ~{eta/60:.1f} min left", flush=True)

    if errors:
        print(f"  {fail} chunks failed; first few:")
        for e in errors[:3]:
            print(f"    {e}")
        print("  rerun the same command -- finished chunks are skipped.")


# =========================================================================
# The feature set, decided by measurement rather than by what happened to be
# available. Written into every dataset file as a sidecar so downstream code
# cannot silently include something it should not.
FEATURES = ["slope", "aspect_sin", "aspect_cos", "curvature", "tpi_300", "tri_300",
            "hand", "log10_upa", "tpi_broad", "landcover", "impervious",
            "gsw_occurrence", "water_frac_900m"]

FLAT_SLOPE_DEG = 0.5      # below this, aspect is undefined rather than north-facing


def derive(out: pd.DataFrame) -> pd.DataFrame:
    """Turn raw bands into the modelling features, and quarantine the rest.

    Three changes, each forced by a measurement on the pulled data.

    ELEVATION IS A LOCATION FINGERPRINT, NOT A FLOOD FEATURE.  Absolute elevation
    alone identifies which of the ten states a point is in with 58.3% accuracy
    against a 10% baseline, and it supplies 0.264 of the 0.85 AUC at which a model
    can tell Zone A from Zone AE -- five times the next feature. Dropping it costs
    0.014 AUC on the primary task (positives vs band negatives, 0.932 -> 0.919) and
    takes domain separability from 0.860 down to 0.751. That trade is the whole
    experiment: what floods is RELATIVE elevation, which `hand`, `tpi_300` and
    `tpi_broad` already carry. So `elev` becomes `meta_elev` -- kept for
    diagnostics, never a feature. Orographic and climate effects, which are the only
    legitimate reason to want absolute elevation, come from StreamCat's
    precipitation and runoff metrics instead.

    log1p ON SQUARE KILOMETRES BARELY TRANSFORMS ANYTHING.  Half the points have an
    upstream area under 0.013 km2, where log1p(x) ~ x, so the band arrived with the
    bulk of its mass crushed against zero and a handful of Mississippi-scale
    outliers carrying all the range. `log10_upa` floors at one MERIT cell
    (0.0081 km2) and takes a base-10 log, which spreads the same information across
    roughly eight orders of magnitude. No re-pull needed -- the original value is
    recoverable as expm1(log1p_upa).

    ASPECT IS UNDEFINED ON FLAT GROUND.  6% of points came back with aspect exactly
    0, which sin/cos would encode as due north rather than as absent. On a
    floodplain -- the cells that carry the positive label -- flatness is the norm,
    so this would have written a spurious northward preference into precisely the
    wrong place. Aspect is masked to NaN below 0.5 degrees of slope.
    """
    # --- elevation out of the feature set -------------------------------
    if "elev" in out.columns:
        out = out.rename(columns={"elev": "meta_elev"})

    # --- upstream area on a usable scale --------------------------------
    if "log1p_upa" in out.columns:
        upa = np.expm1(pd.to_numeric(out["log1p_upa"], errors="coerce"))
        out["log10_upa"] = np.log10(upa.clip(lower=0.0081))
        out = out.rename(columns={"log1p_upa": "meta_log1p_upa"})

    # --- aspect: circular, and undefined where there is no slope --------
    if "aspect" in out.columns:
        slope = pd.to_numeric(out.get("slope"), errors="coerce")
        asp = pd.to_numeric(out["aspect"], errors="coerce")
        asp = asp.where(slope >= FLAT_SLOPE_DEG)
        rad = np.deg2rad(asp)
        out["aspect_sin"] = np.sin(rad)
        out["aspect_cos"] = np.cos(rad)
        out = out.rename(columns={"aspect": "meta_aspect_raw"})

    # --- spatially separated CV folds -----------------------------------
    # GroupKFold on block_id is NOT spatial cross-validation: two adjacent 10 km
    # blocks can land in different folds, so a model can memorise a neighbourhood
    # and be tested on it. Measured: A-vs-AE separability falls from 0.860 with
    # 10 km blocks to 0.693 with 100 km tiles, meaning most of what looked like a
    # domain difference was spatial memorisation. Coarser tiles are built here so
    # the fold column is in the data rather than reinvented per script.
    if "block_id" in out.columns:
        bx = out.block_id.str.split("_").str[0].astype(int)
        by = out.block_id.str.split("_").str[1].astype(int)
        for km in (30, 50, 100):
            out[f"tile_{km}km"] = ((bx // (km // 10)).astype(str) + "_" +
                                   (by // (km // 10)).astype(str))
    return out


def merge(states: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for state in states:
        d = GEE_DIR / state
        shards = sorted(d.glob("chunk_*.parquet")) if d.exists() else []
        if not shards:
            print(f"[{state}] no shards -- skipping")
            continue
        cov = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
        cov = cov.drop_duplicates("point_id")

        import geopandas as gpd
        pts = gpd.read_parquet(PTS_DIR / f"points_{state}.parquet")
        n_before = len(pts)
        out = pts.merge(cov, on="point_id", how="left")

        got = out[ALL_BANDS].notna().mean().mul(100).round(1)
        worst = got.min()
        missing = int(out["elev"].isna().sum()) if "elev" in out else 0
        print(f"\n[{state}] {len(out):,} rows (points {n_before:,}), "
              f"{missing:,} without elevation, worst band coverage {worst:.1f}%")

        out = derive(out)

        have = [f for f in FEATURES if f in out.columns]
        gone = [f for f in FEATURES if f not in out.columns]
        if gone:
            print(f"  WARNING missing features: {gone}")
        meta = [c for c in out.columns if c.startswith("meta_")]
        print(f"  {len(have)} features, {len(meta)} meta_ columns, "
              f"{out.tile_50km.nunique():,} 50 km tiles")

        f = OUT_DIR / f"dataset_{state}.parquet"
        out.to_parquet(f)
        (OUT_DIR / "FEATURES.txt").write_text(
            "# columns a model may use. everything else in dataset_*.parquet is an\n"
            "# identifier or provenance metadata and must stay out.\n"
            "# split on tile_50km (or tile_100km), never on block_id or at point level.\n"
            + "\n".join(FEATURES) + "\n")
        summary.append(dict(state=state, rows=len(out), feats=len(have),
                            tiles50=out.tile_50km.nunique(),
                            pct_complete=round(100*out[have].notna().all(axis=1).mean(), 1)))
        print(f"  -> {f}")

    if summary:
        print("\n" + pd.DataFrame(summary).to_string(index=False))
        print(f"\nfeature list written to {OUT_DIR/'FEATURES.txt'}")
        print("pct_complete = rows with every feature present (aspect is NaN on flat")
        print("ground by design, so use a model that handles NaN or impute at fit time)")


def status(states: list[str]) -> None:
    rows = []
    for s in states:
        parts = chunks_for(s)
        d = GEE_DIR / s
        done = len(list(d.glob("chunk_*.parquet"))) if d.exists() else 0
        rows.append(dict(state=s, points=sum(len(p) for p in parts),
                         chunks=len(parts), done=done,
                         pct=round(100 * done / max(len(parts), 1), 1)))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print(f"\n{t.done.sum()}/{t.chunks.sum()} chunks "
          f"({100*t.done.sum()/max(t.chunks.sum(),1):.1f}%)")


def selftest(groups: list[dict]) -> None:
    ids = [n for n, _, _ in SELFTEST_POINTS]
    lon = np.array([a for _, a, _ in SELFTEST_POINTS])
    lat = np.array([b for _, _, b in SELFTEST_POINTS])

    merged = pd.DataFrame({"point_id": ids})
    for g in groups:
        t0 = time.time()
        try:
            df = sample_group(g, ids, lon, lat)
            merged = merged.merge(df, on="point_id", how="left")
            print(f"  OK   {g['name']:<6} scale={g['scale']:>3} m  "
                  f"{time.time()-t0:5.1f}s  {g['bands']}")
        except Exception as e:
            print(f"  FAIL {g['name']:<6} {str(e)[:200]}")

    print()
    cols = [c for c in ALL_BANDS if c in merged.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(merged.set_index("point_id")[cols].round(3).to_string())

    nulls = merged[cols].isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        print(f"\n  NULL bands: {bad.to_dict()}")
        if "slope" in bad:
            print("  slope null -> setDefaultProjection did not take.")
    else:
        print("\n  All bands returned values at all three points.")
    print("\n  Sanity: the valley bottom should have the lowest hand and tpi_300,")
    print("  the upland the highest log1p_upa. Check that before launching the run.")


def main(argv: list[str]) -> int:
    global CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    a = ap.parse_args(argv)
    CHUNK = a.chunk

    states = a.states or STATES
    bad = [s for s in states if s not in STATES]
    if bad:
        sys.exit(f"unknown states: {bad}")

    if a.status:
        status(states); return 0
    if a.merge:
        merge(states); return 0

    try:
        ee.Initialize(project=a.project) if a.project else ee.Initialize()
    except Exception as e:
        print(f"Initialize failed: {str(e)[:300]}", file=sys.stderr)
        print("Run ee_raw_check.py to see the real error.", file=sys.stderr)
        return 1
    print("Earth Engine initialised")

    groups = build_groups()
    print(f"{len(groups)} request groups per chunk: "
          f"{[g['name'] for g in groups]}")

    if a.selftest:
        selftest(groups); return 0

    GEE_DIR.mkdir(parents=True, exist_ok=True)
    for s in states:
        pull_state(s, groups, a.workers)

    print()
    status(states)
    print("\nWhen every state is at 100%:  python pull_gee.py --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
