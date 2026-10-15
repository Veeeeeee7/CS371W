#!/usr/bin/env python3
"""
pull_alphaearth.py -- attach AlphaEarth Foundations satellite embeddings to the
sampled points.

Dataset  GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (Earth Engine), CC-BY 4.0.
         "The AlphaEarth Foundations Satellite Embedding dataset is produced by
         Google and Google DeepMind."  Brown et al. 2025, arXiv:2507.22291.
Input    data/interim/points/points_{ST}.parquet
Output   data/interim/aef/{YEAR}/{ST}/chunk_{i:05d}.parquet    resumable shards
         data/interim/aef/aef_{ST}_{YEAR}.parquet              after --merge
         data/interim/aef/aef_tiles_{ST}_{YEAR}.geojson        after --footprints

WHAT IS PULLED, PER POINT AND YEAR
==================================
    aef_pt_A00 .. aef_pt_A63       the 10 m embedding under the point
    aef_r{R}_A00 .. aef_r{R}_A63   mean of the 10 m embeddings within R m
    aef_pt_norm, aef_r{R}_norm     vector lengths, added at --merge

DESIGN NOTES
============

THE PIXEL ALREADY SEES AROUND THE POINT.  AlphaEarth computes every 10 m
embedding from a 1.28 km x 1.28 km input frame (spatial self-attention across
the frame), so the pixel under a point is not a context-free point measurement.
The window mean adds the ribbon-scale neighbourhood explicitly. R = 100 m by
default: a 200 m window is the sample spacing, so neighbouring points' windows
touch without much overlap, and flood ribbons here are ~110-130 m wide. Larger
windows smooth features across neighbouring points, which raises the spatial
autocorrelation OODSelect sees; add them with --radii only as a sensitivity arm.

UNIT VECTORS.  Pixel embeddings have length 1 (up to 8-bit quantisation), so
`aef_pt_norm` is a sanity check. A window mean is shorter than 1 by an amount
that measures how much the embeddings inside the window disagree -- 1 for a
homogeneous window, lower across an edge. It is kept rather than normalised
away; downstream code may L2-normalise the mean (sum-and-normalise, as in
Zhuang et al. 2026, arXiv:2608.12663).

WHAT THE EMBEDDING CAN KNOW.  At inference AlphaEarth sees only Sentinel-2,
Sentinel-1 and Landsat 8/9. Elevation (GLO-30), ERA5-Land climate, GEDI, GRACE
and NLCD were reconstruction targets during training, not inputs, so terrain is
encoded only as far as it can be inferred from imagery. Run the reconstruction
probe (can the embedding predict `hand`, `log10_dist_river`?) before dropping
the terrain covariates.

YEARS.  The Earth Engine catalogue lists 2017-2024 (the GCS copy runs to 2025).
Florida effective dates have median 2015 (Zone A 2014, Zone AE 2017). The
default pulls 2017, the year closest to when the maps were made, and 2024, the
current landscape; regions that recur under both are not landscape change since
mapping. 2017 has fewer inputs (no Landsat 9, Sentinel-2B from mid-2017).

PROJECTION AND TILES.  The collection is tiled in local UTM zones (~163.84 km
tiles; Florida spans 16N and 17N). Each year is mosaicked and sampled at 10 m in
EPSG:5070, the CRS the rest of the pipeline uses. The AEF paper reports visible
tile artifacts, so --footprints writes the tile outlines: R1 needs them as an
artifact break class next to the map-production and landscape breaks.

RESUME AND FAILURE.  As in pull_gee.py: one parquet per chunk, finished chunks
are skipped, every request retries with jittered exponential backoff, and a
failed chunk is reported rather than killing the run. Chunks are sorted by
block_id so a request's points sit near each other.

USAGE
=====
    python pull_alphaearth.py --selftest              # 3 FL points, both years
    python pull_alphaearth.py FL                      # Florida, 2017 and 2024
    python pull_alphaearth.py FL --years 2024         # one year
    python pull_alphaearth.py FL --radii 100 300      # add a 300 m window
    python pull_alphaearth.py FL --status
    python pull_alphaearth.py FL --merge
    python pull_alphaearth.py FL --footprints         # UTM tile outlines, for R1
    EE_PROJECT=<project> or --project <project>, as for pull_gee.py
"""
from __future__ import annotations

import argparse
import json
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

ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]

PTS_DIR = Path("data/interim/points")
AEF_DIR = Path("data/interim/aef")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]
YEARS = [2017, 2024]
RADII = [100]

CRS = "EPSG:5070"
SCALE = 10
CHUNK = 2000        # points per shard, and per point-sample request
CHUNK_WIN = 400     # points per window request (~314 pixels each at R = 100 m)
WORKERS = 6
RETRIES = 5


# =========================================================================
# the image
# =========================================================================
def year_collection(year: int) -> "ee.ImageCollection":
    return ee.ImageCollection(ASSET).filterDate(f"{year}-01-01", f"{year + 1}-01-01")


def year_image(year: int) -> "ee.Image":
    return year_collection(year).mosaic().select(BANDS)


def groups_for(year: int, radii: list[int]) -> list[dict]:
    img = year_image(year)
    groups = [dict(name="pt", image=img, radius=0, reducer=ee.Reducer.first())]
    for r in radii:
        groups.append(dict(name=f"r{r}", image=img, radius=r,
                           reducer=ee.Reducer.mean()))
    return groups


def columns_for(radii: list[int]) -> list[str]:
    names = ["pt"] + [f"r{r}" for r in radii]
    return [f"aef_{n}_{b}" for n in names for b in BANDS]


# =========================================================================
# request machinery
# =========================================================================
def sample(group: dict, ids: list[str], lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    """One Earth Engine request. Raises on failure; the caller retries."""
    feats = []
    for p, a, b in zip(ids, lon, lat):
        g = ee.Geometry.Point([float(a), float(b)])
        if group["radius"]:
            g = g.buffer(group["radius"], 1)
        feats.append(ee.Feature(g, {"pid": p}))
    res = (group["image"]
           .reduceRegions(collection=ee.FeatureCollection(feats),
                          reducer=group["reducer"], scale=SCALE, crs=CRS, tileScale=4)
           # drop the geometry: 400 buffered polygons would dominate the payload
           .select(["pid"] + BANDS, None, False)
           .getInfo())
    rows = []
    for f in res["features"]:
        p = f["properties"]
        rows.append({"point_id": p.get("pid"),
                     **{f"aef_{group['name']}_{b}": p.get(b) for b in BANDS}})
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
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 120.0)


def shard_path(year: int, state: str, i: int) -> Path:
    return AEF_DIR / str(year) / state / f"chunk_{i:05d}.parquet"


def do_chunk(year: int, state: str, i: int, sub: pd.DataFrame,
             groups: list[dict], radii: list[int]) -> tuple[int, str]:
    out = shard_path(year, state, i)
    if out.exists():
        return i, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)

    merged = pd.DataFrame({"point_id": sub.point_id.tolist()})
    for g in groups:
        step = CHUNK if g["radius"] == 0 else CHUNK_WIN
        parts = []
        for j in range(0, len(sub), step):
            s = sub.iloc[j:j + step]
            parts.append(with_retries(sample, g, s.point_id.tolist(),
                                      s.lon.to_numpy(), s.lat.to_numpy(),
                                      label=f"{year} {state} chunk {i} [{g['name']}]"))
        merged = merged.merge(pd.concat(parts, ignore_index=True),
                              on="point_id", how="left")

    cols = columns_for(radii)
    missing = [c for c in cols if c not in merged.columns]
    if missing:
        merged = pd.concat([merged, pd.DataFrame(np.nan, index=merged.index,
                                                 columns=missing)], axis=1)
    vals = pd.DataFrame(merged[cols].to_numpy(dtype="float32"), columns=cols)
    merged = pd.concat([merged[["point_id"]].reset_index(drop=True), vals,
                        pd.DataFrame({"aef_year": np.full(len(vals), year, "int16")})],
                       axis=1)
    merged.to_parquet(out, index=False)
    return i, "ok"


def chunks_for(state: str) -> list[pd.DataFrame]:
    p = PTS_DIR / f"points_{state}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=["point_id", "lon", "lat", "block_id"])
    df = df.sort_values("block_id", kind="stable").reset_index(drop=True)
    return [df.iloc[i:i + CHUNK] for i in range(0, len(df), CHUNK)]


def bbox_for(state: str) -> list[float]:
    df = pd.read_parquet(PTS_DIR / f"points_{state}.parquet", columns=["lon", "lat"])
    return [float(df.lon.min()), float(df.lat.min()), float(df.lon.max()), float(df.lat.max())]


def check_year(year: int, state: str) -> int:
    """Number of AEF tiles for this year over the state's points (0 = not published)."""
    return year_collection(year).filterBounds(
        ee.Geometry.Rectangle(bbox_for(state))).size().getInfo()


def pull_state(year: int, state: str, radii: list[int], workers: int) -> None:
    parts = chunks_for(state)
    if not parts:
        print(f"[{year} {state}] no points file -- skipping")
        return
    n_tiles = check_year(year, state)
    if n_tiles == 0:
        print(f"[{year} {state}] no AlphaEarth tiles for this year -- skipping")
        return
    groups = groups_for(year, radii)
    done = {i for i in range(len(parts)) if shard_path(year, state, i).exists()}
    todo = [(i, s) for i, s in enumerate(parts) if i not in done]
    print(f"\n[{year} {state}] {sum(len(s) for s in parts):,} points, {len(parts)} chunks, "
          f"{n_tiles} AEF tiles ({len(done)} done, {len(todo)} to go)", flush=True)
    if not todo:
        return

    t0 = time.time()
    ok = fail = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_chunk, year, state, i, s, groups, radii): i for i, s in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                errors.append(str(e)[:200])
            if n % 5 == 0 or n == len(todo):
                el = time.time() - t0
                eta = (len(todo) - n) * el / max(n, 1)
                print(f"  {n}/{len(todo)} chunks  ok={ok} fail={fail}  "
                      f"{el/60:.1f} min elapsed, ~{eta/60:.1f} min left", flush=True)
    if errors:
        print(f"  {fail} chunks failed; first few:")
        for e in errors[:3]:
            print(f"    {e}")
        print("  rerun the same command -- finished chunks are skipped.")


# =========================================================================
# merge, status, footprints
# =========================================================================
def add_norms(df: pd.DataFrame, radii: list[int]) -> pd.DataFrame:
    for n in ["pt"] + [f"r{r}" for r in radii]:
        cols = [f"aef_{n}_{b}" for b in BANDS]
        if all(c in df.columns for c in cols):
            df[f"aef_{n}_norm"] = np.sqrt((df[cols].astype("float64") ** 2)
                                          .sum(axis=1, min_count=64)).astype("float32")
    return df


def merge(states: list[str], years: list[int], radii: list[int]) -> None:
    rows = []
    for year in years:
        for state in states:
            d = AEF_DIR / str(year) / state
            shards = sorted(d.glob("chunk_*.parquet")) if d.exists() else []
            if not shards:
                continue
            n_expected = len(chunks_for(state))
            df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
            df = df.drop_duplicates("point_id")
            present = [n for n in ["pt"] + [f"r{r}" for r in radii]
                       if f"aef_{n}_A00" in df.columns]
            df = add_norms(df, [int(n[1:]) for n in present if n != "pt"])
            out = AEF_DIR / f"aef_{state}_{year}.parquet"
            df.to_parquet(out, index=False)
            row = dict(year=year, state=state, points=len(df),
                       shards=f"{len(shards)}/{n_expected}",
                       pt_complete=round(100 * df["aef_pt_A00"].notna().mean(), 2),
                       pt_norm_med=round(float(df["aef_pt_norm"].median()), 3))
            for n in present:
                if n != "pt":
                    row[f"{n}_norm_med"] = round(float(df[f"aef_{n}_norm"].median()), 3)
            rows.append(row)
            print(f"  -> {out}")
    if rows:
        print("\n" + pd.DataFrame(rows).to_string(index=False))
        print("\npt_norm_med should be ~1.00 (unit vectors, 8-bit quantised);")
        print("window norms below 1 measure disagreement inside the window.")
        print("Points with no pixel are usually open water or off the coast.")
    else:
        print("nothing to merge")


def status(states: list[str], years: list[int]) -> None:
    rows = []
    for year in years:
        for s in states:
            parts = chunks_for(s)
            done = sum(shard_path(year, s, i).exists() for i in range(len(parts)))
            rows.append(dict(year=year, state=s, points=sum(len(p) for p in parts),
                             chunks=len(parts), done=done,
                             pct=round(100 * done / max(len(parts), 1), 1)))
    print(pd.DataFrame(rows).to_string(index=False))


def footprints(states: list[str], years: list[int]) -> None:
    AEF_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        for state in states:
            coll = year_collection(year).filterBounds(ee.Geometry.Rectangle(bbox_for(state)))
            fc = ee.FeatureCollection(coll.map(lambda im: ee.Feature(
                im.geometry(), {"tile": im.get("system:index"),
                                "utm_zone": im.get("UTM_ZONE")})))
            gj = fc.getInfo()
            out = AEF_DIR / f"aef_tiles_{state}_{year}.geojson"
            out.write_text(json.dumps(gj))
            print(f"  {len(gj['features'])} tiles -> {out}")


def selftest(state: str, years: list[int], radii: list[int]) -> None:
    pts = chunks_for(state)[0].iloc[:3]
    ids = pts.point_id.tolist()
    for year in years:
        n_tiles = check_year(year, state)
        print(f"\n[{year}] {n_tiles} AEF tiles over {state}")
        if n_tiles == 0:
            continue
        merged = pd.DataFrame({"point_id": ids})
        for g in groups_for(year, radii):
            t0 = time.time()
            try:
                df = sample(g, ids, pts.lon.to_numpy(), pts.lat.to_numpy())
                merged = merged.merge(df, on="point_id", how="left")
                print(f"  OK   {g['name']:<5} {time.time() - t0:5.1f}s")
            except Exception as e:
                print(f"  FAIL {g['name']:<5} {str(e)[:200]}")
        merged = add_norms(merged, radii)
        ncols = [c for c in merged.columns if c.endswith("_norm")]
        print(merged.set_index("point_id")[ncols].round(3).to_string())
        pt = merged[[f"aef_pt_{b}" for b in BANDS]].to_numpy(dtype=float)
        if np.isfinite(pt).all():
            cos = pt @ pt.T
            print(f"  cosine between the three points: "
                  f"{cos[0, 1]:.3f} {cos[0, 2]:.3f} {cos[1, 2]:.3f}")
    print("\n  Expect aef_pt_norm ~1.00 and window norms in (0, 1].")
    print("  NaN at a point = masked pixel (open water / offshore) -- fine for one,")
    print("  a problem for all three.")


# =========================================================================
def main(argv: list[str]) -> int:
    global CHUNK, CHUNK_WIN
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--radii", type=int, nargs="*", default=RADII)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--footprints", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--chunk-win", type=int, default=CHUNK_WIN)
    a = ap.parse_args(argv)
    CHUNK, CHUNK_WIN = a.chunk, a.chunk_win

    states = a.states or ["FL"]
    bad = [s for s in states if s not in STATES]
    if bad:
        sys.exit(f"unknown states: {bad}")

    if a.status:
        status(states, a.years); return 0
    if a.merge:
        merge(states, a.years, a.radii); return 0

    try:
        ee.Initialize(project=a.project) if a.project else ee.Initialize()
    except Exception as e:
        print(f"Initialize failed: {str(e)[:300]}", file=sys.stderr)
        print("Run ee_raw_check.py to see the real error.", file=sys.stderr)
        return 1
    print("Earth Engine initialised")

    if a.selftest:
        selftest(states[0], a.years, a.radii); return 0
    if a.footprints:
        footprints(states, a.years); return 0

    for year in a.years:
        for s in states:
            pull_state(year, s, a.radii, a.workers)
    print()
    status(states, a.years)
    print(f"\nWhen every row is at 100%:  python pull_alphaearth.py "
          f"{' '.join(states)} --merge --years {' '.join(map(str, a.years))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
