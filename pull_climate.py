#!/usr/bin/env python3
"""
pull_climate.py -- extreme-rainfall covariates, the last gap in the feature set.

WHY NOT NOAA ATLAS 14
=====================
The theoretically right variable for a 1%-annual-chance flood label is the
1%-annual-chance design storm: NOAA Atlas 14's 100-year 24-hour depth. Getting it
at point level means downloading Atlas 14's gridded products, which ship per volume
with different publication dates -- several volumes and several GB to cover these
ten states, each a separate grid per duration and return period, and no single
national file. That is days of plumbing for one column.

GRIDMET gives the same information from data already reachable through the Earth
Engine credentials that are working. It is 4 km daily precipitation from 1979, so
annual maxima can be computed directly and a Gumbel fit gives return-period depths
without a download:

    pr_annmax_mean   mean annual maximum 1-day rainfall, mm
    pr_annmax_sd     its standard deviation across years -- how variable the
                     extremes are, which is as hydrologically meaningful as the mean
    pr_100yr_gumbel  100-year 1-day depth from a Gumbel fit to the annual maxima
    pr_annual_mean   mean annual total, as a check against StreamCat's precip8110
    pr_wettest5day   mean annual maximum 5-day total -- multi-day storms drive
                     riverine flooding where single-day bursts drive flash flooding,
                     and Zone A and Zone AE sit on different parts of that split

The Gumbel value is an empirical fit to 46 years, not a regional frequency analysis
with Atlas 14's L-moment pooling, so it is noisier and should be described as a
proxy. At 4 km it is also coarser than the 800 m Atlas 14 grids. Both limitations
are worth stating in a limitations section; neither is worth days of download.

If the project later needs true Atlas 14 depths, the point estimates come from
https://hdsc.nws.noaa.gov/pfds/ one location at a time -- fine for a validation
sample of a few hundred points against this proxy, which is the cheap way to show
the substitution is sound.

RESOLUTION HONESTY
==================
4 km is coarse enough that every point inside a 4 km cell gets the same value, and
a 50 km CV tile holds roughly 150 such cells. So these behave like the StreamCat
catchment metrics: real covariates, but regional context rather than point
measurements, and capable of acting as a location fingerprint the way absolute
elevation did. Run the same leakage check on them -- if one starts dominating
A-vs-AE separability, quarantine it.

USAGE
=====
    python pull_climate.py --selftest       # 3 Vermont points, prints the table
    python pull_climate.py                  # all ten states, resumable
    python pull_climate.py --status
    python pull_climate.py --merge           # into data/processed/dataset_{ST}.parquet
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

PTS_DIR = Path("data/processed")
OUT_SHARDS = Path("data/interim/climate")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]

CHUNK = 4000       # 4 km cells: far fewer tiles to fetch, so chunks can be larger
WORKERS = 6
RETRIES = 5
YEAR0, YEAR1 = 1979, 2024

BANDS = ["pr_annmax_mean", "pr_annmax_sd", "pr_100yr_gumbel",
         "pr_annual_mean", "pr_wettest5day"]

SELFTEST_POINTS = [("winooski bottom", -73.1550, 44.4900),
                   ("hillslope", -73.1805, 44.4812),
                   ("upland", -72.9851, 44.5203)]


def pr_collection():
    """Daily precipitation, whichever asset this account can reach."""
    for aid, band, scale in (("IDAHO_EPSCOR/GRIDMET", "pr", 4000),
                             ("NASA/ORNL/DAYMET_V4", "prcp", 1000)):
        try:
            c = ee.ImageCollection(aid).select(band)
            c.first().bandNames().getInfo()
            return c, scale, aid
        except Exception:
            continue
    sys.exit("no daily precipitation collection reachable -- check EE access")


def build_image():
    coll, scale, aid = pr_collection()
    print(f"precipitation: {aid} at {scale} m, {YEAR0}-{YEAR1}")

    years = ee.List.sequence(YEAR0, YEAR1)

    def ann_max(y):
        y = ee.Number(y)
        s = coll.filter(ee.Filter.calendarRange(y, y, "year"))
        return s.max().set("year", y)

    def ann_sum(y):
        y = ee.Number(y)
        s = coll.filter(ee.Filter.calendarRange(y, y, "year"))
        return s.sum().set("year", y)

    maxima = ee.ImageCollection(years.map(ann_max))
    totals = ee.ImageCollection(years.map(ann_sum))

    mu = maxima.mean().rename("pr_annmax_mean")
    sd = maxima.reduce(ee.Reducer.stdDev()).rename("pr_annmax_sd")

    # Gumbel: x_T = mu - (sqrt(6)/pi) * sd * (0.5772 + ln(-ln(1 - 1/T))).
    # For T = 100 the bracket evaluates to a constant, so this is a closed form on
    # the mean and standard deviation of the annual maxima -- no per-pixel fitting.
    T = 100.0
    yT = -np.log(-np.log(1.0 - 1.0 / T))          # 4.6001 for T=100
    k = (np.sqrt(6.0) / np.pi) * (yT - 0.5772)
    gum = mu.add(sd.multiply(float(k))).rename("pr_100yr_gumbel")

    ann = totals.mean().rename("pr_annual_mean")

    # 5-day: convolve the daily series once, at native resolution, then take annual
    # maxima. Done with a moving sum over the whole record rather than per year, so
    # storms that straddle New Year are not cut in half.
    def max5(y):
        y = ee.Number(y)
        s = coll.filter(ee.Filter.calendarRange(y, y, "year")).toList(400)
        n = s.size()
        idx = ee.List.sequence(0, n.subtract(5))
        def win(i):
            return (ee.ImageCollection(s.slice(ee.Number(i), ee.Number(i).add(5)))
                    .sum())
        return ee.ImageCollection(idx.map(win)).max().set("year", y)

    w5 = ee.ImageCollection(years.map(max5)).mean().rename("pr_wettest5day")

    img = mu.addBands(sd).addBands(gum).addBands(ann).addBands(w5)
    return img, scale


def sample(img, scale: int, ids: list[str], lon, lat) -> pd.DataFrame:
    feats = [ee.Feature(ee.Geometry.Point([float(a), float(b)]), {"pid": p})
             for p, a, b in zip(ids, lon, lat)]
    res = img.reduceRegions(collection=ee.FeatureCollection(feats),
                            reducer=ee.Reducer.first(), scale=scale,
                            tileScale=4).getInfo()
    rows = []
    for f in res["features"]:
        p = f["properties"]
        rows.append({"point_id": p.get("pid"), **{b: p.get(b) for b in BANDS}})
    return pd.DataFrame(rows)


def with_retries(fn, *a, label="", **kw):
    delay = 4.0
    for i in range(1, RETRIES + 1):
        try:
            return fn(*a, **kw)
        except Exception as e:
            if i == RETRIES:
                raise RuntimeError(f"{label}: {str(e)[:160]}") from e
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 120.0)


def chunks_for(state: str):
    p = PTS_DIR / f"dataset_{state}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=["point_id", "lon", "lat", "tile_50km"])
    df = df.sort_values("tile_50km", kind="stable").reset_index(drop=True)
    return [df.iloc[i:i + CHUNK] for i in range(0, len(df), CHUNK)]


def do_chunk(state, i, sub, img, scale):
    out = OUT_SHARDS / state / f"chunk_{i:05d}.parquet"
    if out.exists():
        return i, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = with_retries(sample, img, scale, sub.point_id.tolist(),
                      sub.lon.to_numpy(), sub.lat.to_numpy(),
                      label=f"{state} chunk {i}")
    for b in BANDS:
        if b not in df.columns:
            df[b] = np.nan
    df.to_parquet(out, index=False)
    return i, "ok"


def pull(states, img, scale, workers):
    for state in states:
        parts = chunks_for(state)
        if not parts:
            print(f"[{state}] no dataset -- skipping")
            continue
        d = OUT_SHARDS / state
        done = ({int(f.stem.split("_")[1]) for f in d.glob("chunk_*.parquet")}
                if d.exists() else set())
        todo = [(i, s) for i, s in enumerate(parts) if i not in done]
        print(f"\n[{state}] {sum(len(s) for s in parts):,} points, {len(parts)} "
              f"chunks ({len(done)} done, {len(todo)} to go)", flush=True)
        if not todo:
            continue
        t0, ok, fail, errs = time.time(), 0, 0, []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(do_chunk, state, i, s, img, scale): i for i, s in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result(); ok += 1
                except Exception as e:
                    fail += 1; errs.append(str(e)[:180])
                if n % 5 == 0 or n == len(todo):
                    el = time.time() - t0
                    eta = (len(todo) - n) / max(n / max(el, 1e-9), 1e-9)
                    print(f"  {n}/{len(todo)}  ok={ok} fail={fail}  "
                          f"{el/60:.1f} min, ~{eta/60:.1f} min left", flush=True)
        if errs:
            print(f"  {fail} failed; first: {errs[0]}")
            print("  rerun the same command -- finished chunks are skipped.")


def status(states):
    rows = []
    for s in states:
        parts = chunks_for(s)
        d = OUT_SHARDS / s
        done = len(list(d.glob("chunk_*.parquet"))) if d.exists() else 0
        rows.append(dict(state=s, points=sum(len(p) for p in parts),
                         chunks=len(parts), done=done,
                         pct=round(100 * done / max(len(parts), 1), 1)))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print(f"\n{t.done.sum()}/{t.chunks.sum()} chunks")


def merge(states):
    import geopandas as gpd
    for state in states:
        d = OUT_SHARDS / state
        shards = sorted(d.glob("chunk_*.parquet")) if d.exists() else []
        if not shards:
            print(f"[{state}] no shards -- skipping")
            continue
        cov = pd.concat([pd.read_parquet(f) for f in shards],
                        ignore_index=True).drop_duplicates("point_id")
        p = PTS_DIR / f"dataset_{state}.parquet"
        g = gpd.read_parquet(p)
        g = g.drop(columns=[c for c in BANDS if c in g.columns])
        g = g.merge(cov, on="point_id", how="left")
        g.to_parquet(p)
        got = {b: round(100 * g[b].notna().mean(), 1) for b in BANDS}
        print(f"[{state}] {len(g):,} rows, coverage %: {got}")

    f = PTS_DIR / "FEATURES.txt"
    base = [l.strip() for l in f.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    new = [b for b in BANDS if b not in base]
    f.write_text(
        "# columns a model may use. everything else in dataset_*.parquet is an\n"
        "# identifier or provenance metadata and must stay out.\n"
        "# split on tile_50km (or tile_100km), never on block_id or point level.\n"
        "# StreamCat metrics are catchment areal means; pr_* are 4 km grid values.\n"
        "# Both are regional context, not point measurements -- run the elevation\n"
        "# leakage check on them before trusting a result that leans on one.\n"
        + "\n".join(base + new) + "\n")
    print(f"\nFEATURES.txt: {len(base)} -> {len(base)+len(new)}; added {new}")
    print("\nNext: python baselines.py   then   python baselines.py --null")


def selftest(img, scale):
    ids = [n for n, _, _ in SELFTEST_POINTS]
    lon = np.array([a for _, a, _ in SELFTEST_POINTS])
    lat = np.array([b for _, _, b in SELFTEST_POINTS])
    t0 = time.time()
    df = sample(img, scale, ids, lon, lat)
    print(f"\nsampled in {time.time()-t0:.1f}s\n")
    with pd.option_context("display.width", 200):
        print(df.set_index("point_id").round(2).to_string())
    nulls = df[BANDS].isna().sum()
    bad = nulls[nulls > 0]
    print(f"\n  NULL bands: {bad.to_dict()}" if len(bad)
          else "\n  All bands returned values.")
    print("\n  Sanity for Vermont: annual total ~1000-1400 mm, annual max 1-day")
    print("  ~50-90 mm, 100-year 1-day depth above that and below ~250 mm,")
    print("  5-day max above the 1-day max. Three points 20 km apart at 4 km")
    print("  resolution should be similar but not identical.")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)

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
        return 1
    print("Earth Engine initialised")
    img, scale = build_image()

    if a.selftest:
        selftest(img, scale); return 0

    OUT_SHARDS.mkdir(parents=True, exist_ok=True)
    pull(states, img, scale, a.workers)
    print()
    status(states)
    print("\nWhen every state is at 100%:  python pull_climate.py --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
