#!/usr/bin/env python3
"""Attach the four Earth Engine model features to the points.

    slope         degrees, from the USGS 3DEP 10 m DEM (terrain_group)
    elev_sd_r300  terrain variability, the SD of the 3DEP 10 m elevations within 300 m
    hand          height above nearest drainage, MERIT Hydro, m (merit_group)
    landcover     NLCD 2021 land cover class, 30 m (nlcd_group)

Each 2,000-point chunk is sampled with one request per source resolution, each on its source's
native grid, and written as its own shard; finished shards are skipped, so a rerun resumes.
--merge joins the shards to the points and adds the 50 km cross-validation tile (derive()).

Terrain variability (elev_sd_r300, docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D3) is the standard
deviation of the 3DEP 10 m elevations within a circle of radius 300 m centred on the point: each
point is buffered by 300 m and the DEM pixels inside are reduced on the DEM's native grid
(terrain_variability_group).

Input   data/interim/points/points_FL.parquet
Output  data/interim/gee/FL/chunk_{i:05d}.parquet   one shard per chunk
        <processed>/dataset_FL.parquet              after --merge (default data/processed),
                                                    plus FEATURES.txt if absent

    python pull_gee.py --selftest --project $EE_PROJECT   # three Vermont points, prints the table
    python pull_gee.py FL --project $EE_PROJECT
    python pull_gee.py --status FL
    python pull_gee.py --merge FL --processed data/interim/v4_build
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    sys.exit("pip install earthengine-api")

PTS_DIR = Path("data/interim/points")
GEE_DIR = Path("data/interim/gee")
OUT_DIR = Path("data/processed")

STATES = ["FL"]

CHUNK = 2000        # points per request
WORKERS = 6         # concurrent Earth Engine requests
RETRIES = 5         # attempts per request

# The raw bands every shard carries, in shard column order.
ALL_BANDS = ["slope", "elev_sd_r300", "hand", "landcover"]

# The columns a model may use, the bands as pulled. Seeded into FEATURES.txt.
FEATURES = ["slope", "elev_sd_r300", "hand", "landcover"]

SD_RADIUS_M = 300.0     # terrain variability: SD of the 10 m elevations within this radius
BUFFER_ERROR_M = 1.0    # most a buffer circle's polygon may stray from the true circle
FEATURES_HEADER = (
    "# columns a model may use. everything else in dataset_*.parquet is an\n"
    "# identifier or provenance metadata and must stay out.\n"
    "# split on tile_50km, never on block_id or at point level.\n")

LOCATION_TOL_DEG = 1e-9     # a shard's pt_lon/pt_lat within this of the point is the same point

SELFTEST_POINTS = [         # (name, lon, lat)
    ("winooski bottom", -73.1550, 44.4900),
    ("hillslope",       -73.1805, 44.4812),
    ("upland",          -72.9851, 44.5203),
]


# ----------------------------------------------------------------------------- the covariate stack
# Each group is one Earth Engine request: name, image, scale (nominal metres, for printing),
# proj (the source's native grid, which sample_group samples on) and bands.

def dem_3dep():
    """(the USGS 3DEP 10 m DEM as one image, its native projection). mosaic() drops the
    projection and ee.Terrain silently returns null on an image without one, so the collection's
    native projection is put back."""
    coll = ee.ImageCollection("USGS/3DEP/10m_collection")
    native = coll.first().select("elevation").projection()
    return coll.mosaic().select("elevation").setDefaultProjection(native), native


def terrain_group() -> dict:
    """Slope from the USGS 3DEP 10 m elevation."""
    dem, native = dem_3dep()

    slope = ee.Terrain.slope(dem).rename("slope")

    return dict(name="fine", image=slope, scale=10, proj=native, bands=["slope"])


def terrain_variability_group(dem=None, proj=None, radius_m: float = SD_RADIUS_M) -> dict:
    """Terrain variability: the standard deviation of the DEM's elevations within radius_m of the
    point (D3). sample_group buffers each point into a circle of radius_m (to BUFFER_ERROR_M) and
    reduces the pixels inside with stdDev on the DEM's own grid (proj), each pixel weighted by the
    share of it inside the circle, as Earth Engine's reduceRegions does by default. dem and proj
    default to dem_3dep(); a test passes a synthetic image instead."""
    if dem is None:
        dem, proj = dem_3dep()
    return dict(name="sd300", image=dem.rename("elev_sd_r300"), scale=10, proj=proj,
                bands=["elev_sd_r300"], buffer_m=radius_m,
                reducer=ee.Reducer.stdDev().setOutputs(["elev_sd_r300"]))


def merit_group() -> dict:
    """MERIT Hydro (about 90 m): height above nearest drainage."""
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    hand = merit.select("hnd").rename("hand")

    return dict(name="merit", image=hand, scale=90, proj=merit.select("hnd").projection(),
                bands=["hand"])


def nlcd_group() -> dict:
    """NLCD 2021 land cover class, 30 m."""
    releases = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
    nlcd_2021 = releases.filter(ee.Filter.eq("system:index", "2021")).first()
    image = nlcd_2021.select(["landcover"])
    return dict(name="nlcd", image=image, scale=30, proj=image.select("landcover").projection(),
                bands=["landcover"])


def build_groups() -> list[dict]:
    """The request groups, one per source resolution, each sampled on its source's own grid;
    separate requests also mean one failure does not cost the others."""
    groups = [terrain_group(), terrain_variability_group(), merit_group(), nlcd_group()]
    return groups


# ----------------------------------------------------------------------------- requests
def sample_group(group: dict, ids: list[str], lon: np.ndarray,
                 lat: np.ndarray) -> pd.DataFrame:
    """One Earth Engine request: the group's bands at each point, keyed by point_id.

    By default each point reads the cell that contains it (ee.Reducer.first() on the point). A
    group with buffer_m reduces the circle of that radius around the point with its own reducer
    (terrain_variability_group). Raises on failure; the caller retries."""
    features = []
    for pid, x, y in zip(ids, lon, lat):
        geometry = ee.Geometry.Point([float(x), float(y)])
        if group.get("buffer_m"):
            geometry = geometry.buffer(group["buffer_m"], BUFFER_ERROR_M)
        features.append(ee.Feature(geometry, {"pid": pid}))
    # On the source's own grid (crs, no scale), so each point reads the cell that contains it.
    # A scale alone rescales the first band's grid (90 m against MERIT's 92.77 m) and moves
    # ~43% of points to a neighbouring cell.
    response = group["image"].reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=group.get("reducer") or ee.Reducer.first(),
        crs=group["proj"],
        tileScale=4,                       # smaller tiles: fewer out-of-memory aborts
    ).getInfo()

    rows = []
    for feature in response["features"]:
        props = feature["properties"]
        row = {"point_id": props.get("pid")}
        for band in group["bands"]:
            row[band] = props.get(band)
        rows.append(row)
    return pd.DataFrame(rows)


def with_retries(fn, *args, label: str = "", **kwargs):
    """Call fn(*args, **kwargs), up to RETRIES attempts, sleeping between them with a jittered
    backoff whose base starts at 4 s and doubles up to 120 s. Earth Engine's quota, timeout and
    internal errors are transient; the jitter keeps concurrent workers from retrying in lockstep."""
    delay_s = 4.0
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == RETRIES:
                message = str(e)[:160].replace("\n", " ")
                raise RuntimeError(f"{label}: {message}") from e
            time.sleep(delay_s + random.uniform(0, delay_s))
            delay_s = min(delay_s * 2, 120.0)


def do_chunk(state: str, i: int, chunk: pd.DataFrame, groups: list[dict]) -> tuple[int, str]:
    """Sample chunk i and write its shard. Returns (i, "ok"), or (i, "skip") if the shard
    already exists."""
    out = GEE_DIR / state / f"chunk_{i:05d}.parquet"
    if out.exists():
        return i, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)

    ids = chunk.point_id.tolist()
    lon = chunk.lon.to_numpy()
    lat = chunk.lat.to_numpy()

    merged = pd.DataFrame({"point_id": ids})
    for group in groups:
        label = f"{state} chunk {i} [{group['name']}]"
        values = with_retries(sample_group, group, ids, lon, lat, label=label)
        merged = merged.merge(values, on="point_id", how="left")

    # The location travels with the values, so merge() can tell these shards from another
    # sampling build that reissued the same point_ids.
    merged.insert(1, "pt_lon", lon)
    merged.insert(2, "pt_lat", lat)
    merged.to_parquet(out, index=False)
    return i, "ok"


def chunks_for(state: str) -> list[pd.DataFrame]:
    """The state's points in CHUNK-sized pieces, in block_id order; [] without a points file."""
    path = PTS_DIR / f"points_{state}.parquet"
    if not path.exists():
        return []
    points = pd.read_parquet(path, columns=["point_id", "lon", "lat", "block_id"])
    # Spatially coherent chunks: Earth Engine fetches far fewer raster tiles when a request's
    # points sit near each other, which is most of the wall clock.
    points = points.sort_values("block_id", kind="stable").reset_index(drop=True)
    chunks = []
    for start in range(0, len(points), CHUNK):
        chunks.append(points.iloc[start:start + CHUNK])
    return chunks


def shard_paths(state: str) -> list[Path]:
    """The state's finished shards, in chunk order."""
    return sorted((GEE_DIR / state).glob("chunk_*.parquet"))


def pull_state(state: str, groups: list[dict], workers: int) -> None:
    """Sample every chunk of a state that has no shard yet, `workers` chunks at a time.
    A failed chunk is reported and left for a rerun instead of stopping the others."""
    chunks = chunks_for(state)
    if not chunks:
        print(f"[{state}] no points file -- skipping")
        return

    done = set()
    for path in shard_paths(state):
        done.add(int(path.stem.split("_")[1]))
    todo = []
    for i, chunk in enumerate(chunks):
        if i not in done:
            todo.append((i, chunk))
    n_points = sum(len(chunk) for chunk in chunks)
    print(f"\n[{state}] {n_points:,} points, {len(chunks)} chunks "
          f"({len(done)} already done, {len(todo)} to go)", flush=True)
    if not todo:
        return

    t0 = time.time()
    ok = 0
    fail = 0
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for i, chunk in todo:
            futures.append(pool.submit(do_chunk, state, i, chunk, groups))
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                ok += 1
            except Exception as e:
                fail += 1
                errors.append(str(e)[:200])
            if n % 10 == 0 or n == len(todo):
                elapsed_s = time.time() - t0
                rate = n / max(elapsed_s, 1e-9)
                eta_s = (len(todo) - n) / max(rate, 1e-9)
                print(f"  {n}/{len(todo)} chunks  ok={ok} fail={fail}  "
                      f"{elapsed_s/60:.1f} min elapsed, ~{eta_s/60:.1f} min left", flush=True)

    if errors:
        print(f"  {fail} chunks failed; first few:")
        for error in errors[:3]:
            print(f"    {error}")
        print("  rerun the same command -- finished chunks are skipped.")


# ----------------------------------------------------------------------------- merge
def derive(out: pd.DataFrame) -> pd.DataFrame:
    """Add tile_50km, the spatial cross-validation tile of each point, from its block_id. The
    four bands are the model features as pulled."""
    # Coarser tiles for spatial cross-validation: with 10 km blocks, adjacent blocks land in
    # different folds and a model can memorise a neighbourhood. block_id is "{bx}_{by}" on
    # sample_points.py's 10 km EPSG:5070 grid, so a km-kilometre tile spans km // 10 blocks.
    if "block_id" in out.columns:
        block_xy = out.block_id.str.split("_")
        bx = block_xy.str[0].astype(int)
        by = block_xy.str[1].astype(int)
        for km in (50,):
            blocks_per_tile = km // 10
            tile_x = (bx // blocks_per_tile).astype(str)
            tile_y = (by // blocks_per_tile).astype(str)
            out[f"tile_{km}km"] = tile_x + "_" + tile_y
    return out


def count_stale_rows(cov: pd.DataFrame, pts: pd.DataFrame) -> int:
    """Shard rows that do not belong to these points: a point_id they do not have, or (for
    shards that carry pt_lon/pt_lat) a location more than LOCATION_TOL_DEG away."""
    stale = int((~cov.point_id.isin(pts.point_id)).sum())
    if "pt_lon" not in cov.columns:
        return stale
    sampled = pd.DataFrame({"point_id": pts.point_id.values, "lon": pts.lon.values,
                            "lat": pts.lat.values})
    pairs = cov[["point_id", "pt_lon", "pt_lat"]].merge(sampled, on="point_id", how="inner")
    same_lon = np.isclose(pairs.pt_lon, pairs.lon, rtol=0, atol=LOCATION_TOL_DEG)
    same_lat = np.isclose(pairs.pt_lat, pairs.lat, rtol=0, atol=LOCATION_TOL_DEG)
    stale += int((~(same_lon & same_lat)).sum())
    return stale


def merge(states: list[str], out_dir: Path | None = None) -> int:
    """Join each state's shards to its points, derive the features and write
    <out_dir>/dataset_<ST>.parquet (out_dir: OUT_DIR, read when called). Returns 1 if any state
    was refused, else 0."""
    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "FEATURES.txt"
    summary = []
    seeded = False
    refused = 0
    for state in states:
        shards = shard_paths(state)
        if not shards:
            print(f"[{state}] no shards -- skipping")
            continue
        shard_frames = [pd.read_parquet(path) for path in shards]
        cov = pd.concat(shard_frames, ignore_index=True)
        cov = cov.drop_duplicates("point_id")
        pts = gpd.read_parquet(PTS_DIR / f"points_{state}.parquet")
        n_points = len(pts)

        # point_ids are reissued on every sampling build, so shards from an earlier build would
        # join onto the wrong locations without an error; refuse the state instead.
        has_locations = "pt_lon" in cov.columns
        if not has_locations:
            print(f"[{state}] shards carry no pt_lon/pt_lat: only their point_ids can be checked")
        stale = count_stale_rows(cov, pts)
        if stale:
            print(f"[{state}] {stale:,} shard rows do not match "
                  f"points_{state}.parquet -- shards from another sampling build? "
                  f"Move {GEE_DIR / state} aside and re-pull. Not merged.")
            refused += 1
            continue
        if has_locations:
            cov = cov.drop(columns=["pt_lon", "pt_lat"])
        out = pts.merge(cov, on="point_id", how="left")

        coverage_pct = out[ALL_BANDS].notna().mean() * 100
        worst_pct = coverage_pct.round(1).min()
        print(f"\n[{state}] {len(out):,} rows (points {n_points:,}), "
              f"worst band coverage {worst_pct:.1f}%")

        out = derive(out)

        have = [f for f in FEATURES if f in out.columns]
        gone = [f for f in FEATURES if f not in out.columns]
        if gone:
            print(f"  WARNING missing features: {gone}")
        meta = [c for c in out.columns if c.startswith("meta_")]
        n_tiles = out.tile_50km.nunique()
        print(f"  {len(have)} features, {len(meta)} meta_ columns, {n_tiles:,} 50 km tiles")

        dataset_path = out_dir / f"dataset_{state}.parquet"
        out.to_parquet(dataset_path)
        # Seed the list on a first build only. A later merge must not cut it back to these 4:
        # pull_hydro extends it, and zoo.py takes the model's columns in its order.
        if not features_path.exists():
            features_path.write_text(FEATURES_HEADER + "\n".join(FEATURES) + "\n")
            seeded = True
        complete = out[have].notna().all(axis=1)
        summary.append(dict(state=state, rows=len(out), feats=len(have), tiles50=n_tiles,
                            pct_complete=round(100 * complete.mean(), 1)))
        print(f"  -> {dataset_path}")

    if summary:
        print("\n" + pd.DataFrame(summary).to_string(index=False))
        if seeded:
            print(f"\nfeature list written to {features_path}")
        else:
            print(f"\nfeature list left as is in {features_path}")
        print("pct_complete = rows with every feature present")
    if refused:
        return 1
    return 0


# ----------------------------------------------------------------------------- reports
def status(states: list[str]) -> None:
    """Print points, chunks and finished shards per state."""
    rows = []
    for state in states:
        chunks = chunks_for(state)
        n_done = len(shard_paths(state))
        rows.append(dict(state=state, points=sum(len(chunk) for chunk in chunks),
                         chunks=len(chunks), done=n_done,
                         pct=round(100 * n_done / max(len(chunks), 1), 1)))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    total_done = table.done.sum()
    total_chunks = table.chunks.sum()
    print(f"\n{total_done}/{total_chunks} chunks "
          f"({100 * total_done / max(total_chunks, 1):.1f}%)")


def selftest(groups: list[dict]) -> None:
    """Sample every group at the three SELFTEST_POINTS and print the values, to check the
    stack before a long run."""
    ids = []
    lon_deg = []
    lat_deg = []
    for name, x, y in SELFTEST_POINTS:
        ids.append(name)
        lon_deg.append(x)
        lat_deg.append(y)
    lon = np.array(lon_deg)
    lat = np.array(lat_deg)

    merged = pd.DataFrame({"point_id": ids})
    for group in groups:
        t0 = time.time()
        try:
            values = sample_group(group, ids, lon, lat)
            merged = merged.merge(values, on="point_id", how="left")
            print(f"  OK   {group['name']:<6} scale={group['scale']:>3} m  "
                  f"{time.time()-t0:5.1f}s  {group['bands']}")
        except Exception as e:
            print(f"  FAIL {group['name']:<6} {str(e)[:200]}")

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
    print("\n  Sanity: the valley bottom should have the lowest hand.")
    print("  Check that before launching the run.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Attach Earth Engine covariates to the points.")
    ap.add_argument("states", nargs="*", help="state codes (default: FL)")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                    help="Earth Engine Cloud project (default: $EE_PROJECT)")
    ap.add_argument("--selftest", action="store_true",
                    help="sample three Vermont points and print the values")
    ap.add_argument("--merge", action="store_true",
                    help="join the shards into <processed>/dataset_<ST>.parquet")
    ap.add_argument("--processed", type=Path, default=None,
                    help=f"folder --merge writes to (default {OUT_DIR})")
    ap.add_argument("--status", action="store_true", help="show finished chunks per state")
    args = ap.parse_args(argv)

    states = args.states or STATES
    unknown = [s for s in states if s not in STATES]
    if unknown:
        sys.exit(f"unknown states: {unknown}")

    if args.status:
        status(states)
        return 0
    if args.merge:
        return merge(states, args.processed)

    try:
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
    except Exception as e:
        print(f"Initialize failed: {str(e)[:300]}", file=sys.stderr)
        print("Run `earthengine authenticate` and check --project / EE_PROJECT.",
              file=sys.stderr)
        return 1
    print("Earth Engine initialised")

    groups = build_groups()
    group_names = [g["name"] for g in groups]
    print(f"{len(groups)} request groups per chunk: {group_names}")

    if args.selftest:
        selftest(groups)
        return 0

    GEE_DIR.mkdir(parents=True, exist_ok=True)
    for state in states:
        pull_state(state, groups, WORKERS)

    print()
    status(states)
    print(f"\nWhen every state is at 100%:  python pull_gee.py --merge {' '.join(states)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
