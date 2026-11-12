#!/usr/bin/env python3
"""TIGER/Line county polygons of one state, in EPSG:5070.

sample_points.py --state-polygons keeps band negatives inside them and records the county each
point lies in (meta_point_county, the GEOID); pull_land_mask.py takes its extent from them. Input:
the national TIGER/Line county zip, downloaded once into --raw-dir and reused only while its size
and every member's CRC-32 check out. Output, in EPSG:5070:

    <gis-dir>/counties_<ST>.parquet       the state's county polygons (STATEFP, GEOID, NAME)

    python pull_gis_lines.py --selftest              # a synthetic zipped shapefile, nothing kept
    python pull_gis_lines.py --state FL --dry-run    # URL, size and cache check only
    python pull_gis_lines.py --state FL
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import requests
import shapely

RAW_TIGER = Path("data/raw/tiger")            # TIGER download cache
GIS_DIR = Path("data/interim/gis")            # outputs

CRS_WORK = 5070             # NAD83 / CONUS Albers, metres
TIGER_YEAR = 2026           # tl_2026_us_county.zip: 83,378,454 bytes
# The national file: TIGER/Line has no per-state county file, and the per-state
# cartographic-boundary files are generalised and clipped to the shoreline.
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
MAX_DOWNLOAD = 2e9          # bytes; ask before downloading anything larger
MIN_FREE = 10e9             # bytes of disk that must stay free after the download
CHUNK_BYTES = 1 << 20       # download chunk
PROGRESS_BYTES = 20 << 20   # download progress line roughly this often

FIPS = {"FL": "12"}


# =========================================================================
# inputs and outputs
# =========================================================================
def head_size(url: str) -> int:
    """Content-Length of url in bytes, -1 when the server gives none."""
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("Content-Length", -1))


def zip_problem(path: Path) -> str | None:
    """None when the zip opens and every member decompresses to its stored CRC-32,
    else what is wrong with it.

    Size alone accepts a truncated-and-padded or corrupted copy; the CRC test catches
    both. testzip names a CRC or header mismatch, but a corrupted deflate stream raises
    zlib.error and a truncated one EOFError, so those are caught too.
    """
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as e:
        return f"{type(e).__name__}: {e}"
    if bad is None:
        return None
    return f"member {bad} fails its CRC-32 or header check"


def _stream_to(url: str, part: Path, size: int) -> None:
    """Download url into part, printing progress every PROGRESS_BYTES or so."""
    t0, got = time.time(), 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(CHUNK_BYTES):
                f.write(chunk)
                got += len(chunk)
                if got % PROGRESS_BYTES < CHUNK_BYTES:
                    print(f"    {got / 1e6:,.0f} / {size / 1e6:,.0f} MB "
                          f"({time.time() - t0:.0f}s)", flush=True)


def download(url: str, dest: Path, dry_run: bool) -> Path | None:
    """The zip at dest, fetched from url unless a verified copy is cached there.

    A cached zip is reused only if its size equals the server's Content-Length and it
    passes zip_problem; a fresh download must pass the same test before it replaces the
    cache. Delete the cached zip to force a fresh copy (--overwrite redoes the outputs,
    not the download). None on a dry run, which only reports.
    """
    size = head_size(url)
    print(f"  {url}\n  Content-Length {size:,} bytes ({size / 1e6:.1f} MB)", flush=True)
    if dest.exists() and dest.stat().st_size == size:
        problem = zip_problem(dest)
        if problem is None:
            print(f"  cached: {dest} (size and every member's CRC-32 check out)")
            if dry_run:
                print("  dry run: nothing built")
                return None
            return dest
        if dry_run:
            action = "a real run would download it again"
        else:
            action = "downloading it again"
        print(f"  cached {dest} has the right size but fails the zip test ({problem}); {action}")

    if size > MAX_DOWNLOAD:
        sys.exit(f"  {size / 1e9:.1f} GB is over the {MAX_DOWNLOAD / 1e9:.0f} GB limit: ask first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if free - size < MIN_FREE:
        sys.exit(f"  only {free / 1e9:.1f} GB free; the download would leave less than "
                 f"{MIN_FREE / 1e9:.0f} GB: ask first")
    if dry_run:
        print(f"  dry run: would download to {dest} ({free / 1e9:.1f} GB free); nothing built")
        return None

    t0 = time.time()
    part = dest.with_name(dest.name + ".part")
    _stream_to(url, part, size)
    got = part.stat().st_size
    if got != size:
        sys.exit(f"  incomplete download: {got:,} of {size:,} bytes; rerun")
    problem = zip_problem(part)
    if problem:
        sys.exit(f"  downloaded {part} fails the zip test ({problem}); rerun")
    part.replace(dest)
    print(f"  downloaded {dest} ({time.time() - t0:.0f}s)")
    return dest


def read_counties(zip_path: Path, state_fips: str):
    """The state's counties, EPSG:5070."""
    src = f"/vsizip/{Path(zip_path).resolve()}"
    cols = ["STATEFP", "GEOID", "NAME"]
    state = pyogrio.read_dataframe(src, columns=cols, where=f"STATEFP = '{state_fips}'")
    if not len(state):
        sys.exit(f"{zip_path}: no counties with STATEFP {state_fips}")
    print(f"  {len(state)} counties with STATEFP {state_fips}; source CRS "
          f"{state.crs.to_epsg()}", flush=True)
    return state.to_crs(CRS_WORK)


def write_geoparquet(gdf, path: Path, meta: dict) -> None:
    """GeoParquet with meta stored as JSON under the 'pull_gis_lines' schema key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    t = pq.read_table(path)
    metadata = dict(t.schema.metadata or {})
    metadata[b"pull_gis_lines"] = json.dumps(meta, sort_keys=True, default=str).encode()
    pq.write_table(t.replace_schema_metadata(metadata), path)


# =========================================================================
# the county layer
# =========================================================================
def run_counties(args, fips: str) -> None:
    """Write counties_<ST>.parquet unless it exists."""
    poly_out = args.gis_dir / f"counties_{args.state}.parquet"
    print(f"\n[{args.state}] counties (TIGER/Line {args.tiger_year})", flush=True)
    if poly_out.exists() and not args.overwrite:
        print(f"  {poly_out} exists -- skipping (use --overwrite)")
        return
    url = TIGER_URL.format(year=args.tiger_year)
    zip_path = download(url, args.raw_dir / Path(url).name, args.dry_run)
    if zip_path is None:
        return
    t0 = time.time()
    counties = read_counties(zip_path, fips)
    in_state = counties[counties.STATEFP == fips].reset_index(drop=True)
    write_geoparquet(in_state, poly_out,
                     dict(source=url, tiger_year=args.tiger_year, state=args.state))
    print(f"  wrote {poly_out}  ({time.time() - t0:.0f}s)")


# =========================================================================
# self-test: a zipped shapefile of three squares, through the same functions
# =========================================================================
def _check(what: str, got, want) -> bool:
    """Print a PASS or FAIL line; numbers are compared with np.isclose."""
    numeric = isinstance(want, (int, float)) and not isinstance(want, bool)
    if numeric:
        good = bool(np.isclose(got, want))
    else:
        good = got == want
    print(f"  {'PASS' if good else 'FAIL'}  {what}: {got} (want {want})")
    return bool(good)


def _three_squares() -> list:
    """A and B stacked; C spans both with no vertex where A meets B (a T-junction)."""
    return [shapely.box(0, 0, 1000, 1000), shapely.box(0, 1000, 1000, 2000),
            shapely.box(1000, 0, 2000, 2000)]


def _flip_byte_in_largest_member(zip_path: Path, out: Path) -> None:
    """A copy of zip_path with one byte flipped in the middle of its largest member."""
    with zipfile.ZipFile(zip_path) as z:
        member = max(z.infolist(), key=lambda info: info.compress_size)
    # The member's data follows its 30-byte local header, file name and extra field.
    start = member.header_offset + 30 + len(member.filename.encode()) + len(member.extra)
    raw = bytearray(zip_path.read_bytes())
    raw[start + member.compress_size // 2] ^= 0xFF
    out.write_bytes(bytes(raw))


def _selftest_read_counties(counties, tmp_root: Path) -> list[bool]:
    print("read_counties from a zipped shapefile, with a STATEFP filter:")
    lonlat = gpd.GeoDataFrame(counties.drop(columns="geometry"), geometry=[
        shapely.box(-82.02, 30.0, -82.01, 30.01), shapely.box(-82.02, 30.01, -82.01, 30.02),
        shapely.box(-82.01, 30.0, -82.0, 30.02)], crs=4269)
    far = gpd.GeoDataFrame(dict(STATEFP=["13"], GEOID=["13999"], NAME=["far"]),
                           geometry=[shapely.box(-84, 33, -83.9, 33.1)], crs=4269)
    lonlat = pd.concat([lonlat, far], ignore_index=True)
    results = []
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        shp = Path(td) / "tl_test_us_county.shp"
        pyogrio.write_dataframe(lonlat, shp)
        zip_path = Path(td) / "tl_test_us_county.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(td).glob("tl_test_us_county.*"):
                if f.suffix != ".zip":
                    z.write(f, f.name)
        got = read_counties(zip_path, "12")
        results.append(_check("zip test passes the intact zip", zip_problem(zip_path), None))
        broken = Path(td) / "broken.zip"
        _flip_byte_in_largest_member(zip_path, broken)
        results.append(_check("zip test flags one flipped byte in a deflated member",
                              zip_problem(broken) is not None, True))
    results.append(_check("counties read (the 2 in state; the other state's dropped)",
                          sorted(got.GEOID), ["12001", "12003"]))
    results.append(_check("CRS", got.crs.to_epsg(), CRS_WORK))
    return results


def selftest(tmp_root: Path) -> int:
    """Run the checks on synthetic squares; the only file written is a temp dir under
    tmp_root, removed afterwards. Returns the exit code."""
    counties = gpd.GeoDataFrame(dict(STATEFP=["12", "12", "13"],
                                     GEOID=["12001", "12003", "13049"], NAME=["A", "B", "C"]),
                                geometry=_three_squares(), crs=CRS_WORK)
    results = _selftest_read_counties(counties, tmp_root)
    ok = all(results)
    print("selftest " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="FL")
    parser.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_TIGER, help="TIGER download cache")
    parser.add_argument("--tiger-year", type=int, default=TIGER_YEAR)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the TIGER URL, size and cache state; build nothing")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        args.gis_dir.mkdir(parents=True, exist_ok=True)
        return selftest(args.gis_dir)
    if args.state not in FIPS:
        sys.exit(f"unknown state {args.state}")
    run_counties(args, FIPS[args.state])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
