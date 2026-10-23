#!/usr/bin/env python3
"""
pull_dem.py -- a 10 m DEM over each boundary-analysis bbox, then elevation z at every lobe-edge
vertex and z_in / z_out at --offset-m (default 100 m) inside and outside it along the edge normal.

    fetch     <prefix>_bboxes.parquet ...  -> <gis-dir>/dem/<prefix>_c<k>.tif + <prefix>_dem.json
    sample    <prefix>_vertices.parquet    -> results/<today>/boundary/<prefix>_vertices_z.parquet
    selftest  offline checks of the grid, tile and sampling code

Both inputs come from aotil/lobes.py; <prefix> (<cell>_<variant>_S<S>) is read off the file name.
Without --yes, `fetch` only lists rasters, tiles and sizes; pass the bboxes of every prefix in one
call so the size gate (--max-gb, --min-free-gb) sees the whole run. `sample` adds z, z_in, z_out
(NAVD88 m), dem_ok and, when <prefix>_lobes.parquet is present, PROBE_COLS. These feed the
level-set test in aotil/edge_layers.py; they are never model features.

Note 1, tile versions: pull_gee.py's elevation features come from Earth Engine's
USGS/3DEP/10m_collection, built from the same 1/3" tiles, but TNM republishes tiles and which
version Earth Engine served is not checked. Tiles published after REPUBLISHED_AFTER are flagged.

Example (docs/RUN-FL-V2.md section 13, step D3):
    python pull_dem.py fetch --bboxes $B/<p>_S250_bboxes.parquet $B/<p>_S500_bboxes.parquet \\
        --remote --yes --max-gb 3.5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import requests
import shapely
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GIS_DIR = Path("data/interim/gis")
RAW_DIR = Path("data/raw/3dep")           # tile and exportImage cache
RESULTS = Path("results")

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASET = "National Elevation Dataset (NED) 1/3 arc-second"
TNM_CURRENT = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
               "current/{tile}/USGS_13_{tile}.tif")
TILE_RE = re.compile(r"USGS_13_(n\d{2}w\d{3})")
IS_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

# Every raster is NAD83 / Conus Albers at exactly 10 m, pixel edges on multiples of 10 m: the
# lobes and vertices are EPSG:5070 metres, both sources land on one lattice, and the native 1/3"
# cell in Florida (10.3 m N-S by 8.9-9.3 m E-W) makes 10 m close to a 1:1 resample. NAD83 to
# NAD83 means no datum shift; heights stay NAVD88 metres.
CRS_OUT = "EPSG:5070"
CRS_TILE = "EPSG:4269"    # NAD83 geographic, as the 1/3" tiles ship
RES_M = 10.0
NODATA = -999999.0        # the 3DEP nodata value
SRC_PAD_PX = 2            # source pixels kept around a window for the bilinear kernel
IS_TILE_PX = 4000         # per exportImage request side; the server allows 8,000

# tnm (the default) clips the 3DEP 1/3" tiles, the product pull_gee.py's features came from.
# imageserver is a fallback: the service resamples the finest 3DEP source under each pixel (1 m
# lidar where there is any), which differs from tnm by decimetres (median 0.07 m, max 3.2 m on a
# test box), so a run should use one source throughout. The tag catches a raster from the other
# source as stale.
SOURCE_TAG = {"tnm": "tnm 3DEP 1/3 arc-second",
              "imageserver": "3DEP ImageServer best-available, bilinear"}
REPUBLISHED_AFTER = "2024-01-01"   # tnm tiles newer than this get a version warning (note 1)

NORMAL_EPS_M = 0.5        # step along the inward normal for the orientation check
NORMAL_SAMPLE = 5000      # vertices checked per component
NORMAL_MIN_SHARE = 0.9    # every component must reach it
UNIT_NORMAL_TOL = 1e-3    # |(nx, ny)| must be within this of 1

BBOX_COLS = ["component", "minx", "miny", "maxx", "maxy"]
VERTEX_COLS = ["lobe_id", "zone", "role", "component", "ring", "s",
               "x", "y", "lon", "lat", "nx", "ny"]
PROBE_COLS = ["in_probe_inside", "out_probe_outside", "out_probe_in_any_lobe"]
MISSES_COLS = ["lobe_id", "zone", "role", "component", "ring", "s", "x", "y",
               "lon", "lat", "z", "z_in", "z_out"]

RETRIES = 4
TIMEOUT = (30, 300)       # connect, read seconds
CHUNK_BYTES = 1 << 20
PROGRESS_EVERY_S = 15
REMOTE_SLACK = 2 * 16384  # bytes per block: GDAL /vsicurl/ reads whole 16 KB chunks
REMOTE_HEADER = 1 << 20   # bytes per tile pass: header, IFDs, tile offset tables

# GDAL settings for reading COG tiles over HTTPS (/vsicurl/)
GDAL_REMOTE = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                   GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                   GDAL_HTTP_MAX_RETRY=str(RETRIES), GDAL_HTTP_RETRY_DELAY="2")

SIZE_COLUMNS = {"tnm": ["whole tiles", "--remote <=", "rasters <="],
                "imageserver": ["exportImage <=", "rasters <="]}


# ---------------------------------------------------------------- small helpers
def session() -> requests.Session:
    """An HTTPS session that retries GET and HEAD on connection errors, 429 and 5xx."""
    retry = Retry(total=RETRIES, connect=RETRIES, read=RETRIES, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "HEAD"]), raise_on_status=False)
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


def gb(n: float) -> str:
    """A byte count as MB below 100 MB, else as GB."""
    if n < 1e8:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.3f} GB"


def prefix_of(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        sys.exit(f"{path.name} does not end in {suffix}; pass --prefix")
    return path.name[: -len(suffix)]


def dem_path(gis_dir: Path, prefix: str, comp: int) -> Path:
    return gis_dir / "dem" / f"{prefix}_c{comp}.tif"


def manifest_path(gis_dir: Path, prefix: str) -> Path:
    return gis_dir / "dem" / f"{prefix}_dem.json"


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def free_bytes(path: Path) -> int:
    """Free disk space at `path`, or at its nearest existing parent."""
    p = path.resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def read_bboxes(path: Path) -> pd.DataFrame:
    """The lobes.py bbox table sorted by component, exiting on missing columns, duplicate
    components or empty boxes."""
    bb = pd.read_parquet(path)
    miss = [c for c in BBOX_COLS if c not in bb.columns]
    if miss:
        sys.exit(f"{path}: missing columns {miss}")
    bb["component"] = bb["component"].astype(int)
    if bb["component"].duplicated().any():
        sys.exit(f"{path}: duplicate component ids")
    bad = bb[(bb.maxx <= bb.minx) | (bb.maxy <= bb.miny)]
    if len(bad):
        sys.exit(f"{path}: empty bboxes for components {bad.component.tolist()}")
    return bb.sort_values("component").reset_index(drop=True)


@dataclass
class Grid:
    """A north-up EPSG:5070 grid whose pixel edges sit on multiples of RES_M."""
    minx: float
    miny: float
    maxx: float
    maxy: float

    @classmethod
    def around(cls, minx, miny, maxx, maxy) -> "Grid":
        """The smallest such grid containing the box."""
        return cls(math.floor(minx / RES_M) * RES_M, math.floor(miny / RES_M) * RES_M,
                   math.ceil(maxx / RES_M) * RES_M, math.ceil(maxy / RES_M) * RES_M)

    @property
    def width(self) -> int:
        return int(round((self.maxx - self.minx) / RES_M))

    @property
    def height(self) -> int:
        return int(round((self.maxy - self.miny) / RES_M))

    @property
    def transform(self) -> Affine:
        return Affine(RES_M, 0.0, self.minx, 0.0, -RES_M, self.maxy)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

    def lonlat_bounds(self) -> tuple[float, float, float, float]:
        """Exact NAD83 envelope of the (curved, in lon/lat) Albers rectangle."""
        return transform_bounds(CRS_OUT, CRS_TILE, *self.bounds, densify_pts=21)


def grids_of(bb: pd.DataFrame) -> dict[int, Grid]:
    return {int(r.component): Grid.around(r.minx, r.miny, r.maxx, r.maxy)
            for r in bb.itertuples()}


@dataclass
class Job:
    """One raster `fetch` must end up with: component `comp` of `prefix`, labelled p1, p2, ...
    by its --bboxes file. state: "exists" (kept as it is), "stale" (exists but is not this grid
    from this source; `why` says what differs) or "todo"."""
    label: str
    prefix: str
    comp: int
    grid: Grid
    path: Path
    state: str = "todo"
    why: str = ""

    @property
    def key(self) -> tuple[float, float, float, float]:
        return self.grid.bounds

    @property
    def name(self) -> str:
        return f"{self.label}:c{self.comp}"


def raster_mismatch(path: Path, grid: Grid, source: str) -> str:
    """'' when `path` is exactly `grid` from `source`, else what differs.

    Prefixes repeat across runs (v1 and v2 are both <cell>_<variant>_S<S>) and the gis-dir is not
    dated, so without this check a new run's bboxes would reuse old rasters by name."""
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            return f"CRS {src.crs}"
        same_shape = (src.width, src.height) == (grid.width, grid.height)
        if not same_shape or not src.transform.almost_equals(grid.transform, precision=1e-6):
            return (f"bounds {tuple(round(v) for v in src.bounds)} != bbox grid "
                    f"{tuple(round(v) for v in grid.bounds)}")
        tag = src.tags().get("source")
        if tag != SOURCE_TAG[source]:
            return f"source {tag!r}, not {source}"
    return ""


def link_raster(src: Path, dst: Path) -> str:
    """Make `dst` the raster `src` is: a hardlink, which costs no disk, or a copy where the
    filesystem has none. Replaced by rename, like write_raster."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and os.path.samefile(src, dst):
        return "hardlink"
    tmp = dst.with_name(dst.stem + f".tmp{os.getpid()}.tif")
    try:
        os.link(src, tmp)
        kind = "hardlink"
    except OSError:
        shutil.copy2(src, tmp)
        kind = "copy"
    os.replace(tmp, dst)
    return kind


def tiles_for(lonlat: tuple[float, float, float, float]) -> list[str]:
    """1 x 1 degree 3DEP tile names covering a NAD83 envelope.

    Tile nYYwXXX spans latitude YY-1..YY and longitude -XXX..-XXX+1; its 6-pixel (2") overlap
    with the neighbours is what feeds the bilinear kernel at a seam, so a bbox that merely
    approaches a degree line does not pull the next tile.
    """
    west, south, east, north = lonlat
    out = []
    for lat0 in range(math.floor(south), math.ceil(north)):
        for lon0 in range(math.floor(west), math.ceil(east)):
            out.append(f"n{lat0 + 1:02d}w{-lon0:03d}")
    return sorted(out)


def window_for(src, lonlat, pad: int = SRC_PAD_PX) -> Window | None:
    """The source window covering a NAD83 envelope plus `pad` pixels, clipped to the dataset;
    None when they do not overlap."""
    t = src.transform
    west, south, east, north = lonlat
    c0 = math.floor((west - t.c) / t.a) - pad
    c1 = math.ceil((east - t.c) / t.a) + pad
    r0 = math.floor((north - t.f) / t.e) - pad
    r1 = math.ceil((south - t.f) / t.e) + pad
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, src.width), min(r1, src.height)
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def write_raster(arr: np.ndarray, grid: Grid, out: Path, tags: dict) -> None:
    """A float32 GeoTIFF on `grid`. It replaces `out` by rename, never rewriting it in place, so
    --overwrite on one prefix leaves a raster hardlinked under another prefix as it was."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.tif")
    profile = dict(driver="GTiff", dtype="float32", count=1, width=grid.width,
                   height=grid.height, crs=CRS_OUT, transform=grid.transform,
                   nodata=NODATA, tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, out)


def valid_frac(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr != NODATA))


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def update_manifest(path: Path, prefix: str, source: str, comp: int, entry: dict) -> None:
    """One JSON per prefix, one entry per component raster: what it is and where its numbers
    came from."""
    m = read_manifest(path)
    m.update(prefix=prefix, crs=CRS_OUT, res_m=RES_M, nodata=NODATA,
             vertical="NAVD88 metres", resampling="bilinear")
    components = m.setdefault("components", {})
    components[str(comp)] = {**entry, "source": source}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(json.dumps(m, indent=1, sort_keys=True), path)


# ---------------------------------------------------------------- fetch: tnm
def tnm_query(sess: requests.Session, lonlat) -> list[dict]:
    """Every TNM API item for the 1/3" GeoTIFF dataset over a NAD83 envelope, page by page."""
    items, offset = [], 0
    while True:
        r = sess.get(TNM_API, timeout=TIMEOUT, params=dict(
            datasets=TNM_DATASET, prodFormats="GeoTIFF", outputFormat="JSON",
            bbox=",".join(f"{v:.6f}" for v in lonlat), max=100, offset=offset))
        r.raise_for_status()
        d = r.json()
        if d.get("errors"):
            raise RuntimeError(f"TNM API errors: {d['errors']}")
        page = d.get("items", [])
        items += page
        offset += len(page)
        if not page or offset >= int(d.get("total", 0)):
            return items


def head(sess: requests.Session, url: str) -> tuple[int | None, str | None]:
    """(Content-Length, ETag) of `url`, or (None, None) unless HEAD answers 200."""
    r = sess.head(url, timeout=TIMEOUT, allow_redirects=True)
    if r.status_code != 200:
        return None, None
    return int(r.headers.get("Content-Length", 0)), r.headers.get("ETag")


def newest_tile_items(sess, grids: dict[tuple, Grid], wanted) -> dict[str, dict]:
    """The newest 1 x 1 degree API item of each wanted tile, by (publicationDate, lastUpdated).

    The API lists every historical version of a tile; on a tie the first one listed is kept."""
    newest: dict[str, dict] = {}
    newest_key: dict[str, tuple] = {}
    for g in grids.values():
        for item in tnm_query(sess, g.lonlat_bounds()):
            match = TILE_RE.search(item.get("downloadURL") or "")
            if not match or item.get("extent") != "1 x 1 degree":
                continue
            tile = match.group(1)
            if tile not in wanted:
                continue
            key = (item.get("publicationDate") or "", item.get("lastUpdated") or "")
            if tile not in newest or key > newest_key[tile]:
                newest[tile] = item
                newest_key[tile] = key
    return newest


def tnm_plan(sess, grids: dict[tuple, Grid], raw_dir: Path) -> list[dict]:
    """Per tile: the grids it serves, its newest version, true size, what is already cached,
    and the --remote block bound for each grid it serves.

    Sizes come from HEAD because the API's sizeInBytes is wrong for some tiles (491,502 bytes
    for a 503 MB one). current_same says whether the newest item's ETag is that of the file
    under .../13/TIFF/current/."""
    need: dict[str, list[tuple]] = {}
    for key, g in grids.items():
        for tile in tiles_for(g.lonlat_bounds()):
            need.setdefault(tile, []).append(key)
    newest = newest_tile_items(sess, grids, need)

    tiles = []
    for tile in sorted(need):
        item = newest.get(tile)
        if item is None:
            tiles.append(dict(tile=tile, grids=need[tile], url=None))
            continue
        url = item["downloadURL"]
        size, etag = head(sess, url)
        _, current_etag = head(sess, TNM_CURRENT.format(tile=tile))
        local = raw_dir / Path(url).name
        part = local.with_name(local.name + ".part")
        if local.exists():
            have = local.stat().st_size
        elif part.exists():
            have = part.stat().st_size
        else:
            have = 0
        if size is None:
            to_get, remote = 0, {}
        else:
            to_get = max(size - have, 0)
            remote = remote_blocks_bytes(url, {k: grids[k] for k in need[tile]})
        tiles.append(dict(
            tile=tile, grids=need[tile], url=url, title=item.get("title"),
            published=item.get("publicationDate") or "", size=size, etag=etag,
            current_same=(etag is not None and etag == current_etag), local=local,
            cached=(local.exists() and size is not None and have == size),
            to_get=to_get, remote=remote))
    return tiles


def remote_blocks_bytes(url: str, grids: dict[tuple, Grid]) -> dict[tuple, int]:
    """Upper bound on what --remote reads of one tile for each grid.

    The stored (compressed) size of every full-resolution block the grid's window touches, from
    the COG's own TileByteCounts, plus REMOTE_SLACK per block and REMOTE_HEADER per pass.
    Blocks x 4 bytes would count open water and nodata, which LZW stores in a few KB, at full
    size. Each grid is clipped in its own pass, so blocks two grids share are counted twice."""
    with rasterio.Env(**GDAL_REMOTE), rasterio.open("/vsicurl/" + url) as src:
        block_h, block_w = src.block_shapes[0]
        out = {}
        for key, g in grids.items():
            win = window_for(src, g.lonlat_bounds())
            if win is None:
                out[key] = 0
                continue
            first_row = win.row_off // block_h
            last_row = (win.row_off + win.height - 1) // block_h
            first_col = win.col_off // block_w
            last_col = (win.col_off + win.width - 1) // block_w
            total = REMOTE_HEADER
            for by in range(first_row, last_row + 1):
                for bx in range(first_col, last_col + 1):
                    stored = src.get_tag_item(f"BLOCK_SIZE_{bx}_{by}", "TIFF", bidx=1)
                    if stored:
                        total += int(stored) + REMOTE_SLACK
                    else:
                        total += block_w * block_h * 4 + REMOTE_SLACK
            out[key] = total
        return out


def download(sess: requests.Session, url: str, dest: Path, size: int) -> None:
    """Resumable: bytes go to <dest>.part, continued with a Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 200:
                    have = 0                 # the whole body (Range ignored): start over
                elif not (r.status_code == 206 and have):
                    raise RuntimeError(f"HTTP {r.status_code}")
                t0, last_print = time.time(), 0.0
                with open(part, "ab" if have else "wb") as f:
                    got = have
                    for chunk in r.iter_content(CHUNK_BYTES):
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last_print > PROGRESS_EVERY_S:
                            last_print = time.time()
                            print(f"    {dest.name}: {got / 1e6:,.0f}/{size / 1e6:,.0f} MB "
                                  f"({time.time() - t0:.0f}s)", flush=True)
        except (requests.RequestException, RuntimeError) as e:
            print(f"    {dest.name}: attempt {attempt} failed ({e}); resuming", flush=True)
            time.sleep(2 * attempt)
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: incomplete after {RETRIES} attempts")
    os.replace(part, dest)


def centre_in_cell(grid: Grid, rows: np.ndarray, cols: np.ndarray, lon0: int,
                   lat0: int) -> np.ndarray:
    """Whether each output pixel's centre lies in the 1 x 1 degree cell whose south-west corner
    is (lon0, lat0)."""
    lon, lat = warp_transform(CRS_OUT, CRS_TILE, grid.minx + (cols + 0.5) * RES_M,
                              grid.maxy - (rows + 0.5) * RES_M)
    return (np.floor(np.asarray(lon)) == lon0) & (np.floor(np.asarray(lat)) == lat0)


def clip_tiles(sources: list[str], grid: Grid) -> tuple[np.ndarray, list[str]]:
    """Reproject the part of each 1/3" tile under `grid` onto it, bilinear; returns the raster
    and the sources that overlap it.

    Only a window of each tile is read, so a 0.5 GB tile never sits in memory and --remote reads
    only the blocks it needs. Adjacent tiles overlap by 6 pixels (2") and can be different
    releases, so where two tiles both cover an output pixel, the tile whose own 1 x 1 degree cell
    holds the pixel centre wins; the overlap only feeds the kernel at the seam and fills pixels no
    owning tile covers.
    """
    dst = np.full((grid.height, grid.width), NODATA, np.float32)
    used = []
    remote = any(s.startswith("/vsicurl/") for s in sources)
    env = GDAL_REMOTE if remote else {}
    with rasterio.Env(**env):
        for source in sources:
            with rasterio.open(source) as src:
                if src.crs is None or src.crs.to_epsg() != 4269:
                    raise RuntimeError(f"{source}: expected EPSG:4269, got {src.crs}")
                win = window_for(src, grid.lonlat_bounds())
                if win is None:
                    continue
                arr = src.read(1, window=win)
                out = np.full_like(dst, NODATA)
                reproject(arr, out, src_transform=src.window_transform(win),
                          src_crs=src.crs, src_nodata=src.nodata,
                          dst_transform=grid.transform, dst_crs=CRS_OUT,
                          dst_nodata=NODATA, resampling=Resampling.bilinear,
                          init_dest_nodata=False)
                write = out != NODATA
                rows, cols = np.nonzero(write & (dst != NODATA))   # already written by another tile
                if len(rows):
                    # The tile's own cell is its extent less the overlap, which is symmetric.
                    lon0 = math.floor((src.bounds.left + src.bounds.right) / 2)
                    lat0 = math.floor((src.bounds.bottom + src.bounds.top) / 2)
                    write[rows, cols] = centre_in_cell(grid, rows, cols, lon0, lat0)
                np.copyto(dst, out, where=write)
                used.append(source)
    return dst, used


# ---------------------------------------------------------------- fetch: imageserver
def is_subgrids(grid: Grid) -> list[tuple[int, int, Grid]]:
    """Split a grid into requests of at most IS_TILE_PX a side, each on the same lattice, as
    (row offset, column offset, subgrid)."""
    out = []
    for r0 in range(0, grid.height, IS_TILE_PX):
        for c0 in range(0, grid.width, IS_TILE_PX):
            w = min(IS_TILE_PX, grid.width - c0)
            h = min(IS_TILE_PX, grid.height - r0)
            minx = grid.minx + c0 * RES_M
            maxy = grid.maxy - r0 * RES_M
            out.append((r0, c0, Grid(minx, maxy - h * RES_M, minx + w * RES_M, maxy)))
    return out


def is_cache_name(g: Grid) -> str:
    return f"is5070_10m_{g.minx:.0f}_{g.maxy:.0f}_{g.width}x{g.height}.tif"


def is_export(sess: requests.Session, g: Grid, dest: Path) -> None:
    """One exportImage request, clipped server side to `g` in EPSG:5070 at 10 m."""
    params = dict(bbox=",".join(f"{v:.0f}" for v in g.bounds), bboxSR=5070, imageSR=5070,
                  size=f"{g.width},{g.height}", format="tiff", pixelType="F32",
                  noData=NODATA, noDataInterpretation="esriNoDataMatchAny",
                  interpolation="RSP_BilinearInterpolation", compression="LZ77",
                  adjustAspectRatio="false", f="image")
    r = sess.get(IS_URL + "/exportImage", params=params, timeout=TIMEOUT)
    body = r.content
    if r.status_code != 200 or body[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"exportImage HTTP {r.status_code}: {body[:300]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    tmp.write_bytes(body)
    os.replace(tmp, dest)


def is_read(path: Path, g: Grid) -> np.ndarray:
    """Read one exportImage tile, refusing anything not on the requested grid."""
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            raise RuntimeError(f"{path.name}: CRS {src.crs}, expected EPSG:5070")
        same_shape = (src.width, src.height) == (g.width, g.height)
        if not same_shape or not src.transform.almost_equals(g.transform, precision=1e-6):
            raise RuntimeError(f"{path.name}: grid {src.transform} {src.width}x{src.height} "
                               f"!= requested {g.transform} {g.width}x{g.height}")
        a = src.read(1).astype(np.float32)
        if src.nodata is not None and src.nodata != NODATA:
            a[a == np.float32(src.nodata)] = NODATA
        return a


def is_catalog(sess: requests.Session, g: Grid) -> dict:
    """The primary 3DEP catalog items under a bbox that are visible at 10 m: the candidates the
    server mosaics from, best (finest) first."""
    r = sess.get(IS_URL + "/query", timeout=TIMEOUT, params=dict(
        geometry=",".join(f"{v:.0f}" for v in g.bounds), geometryType="esriGeometryEnvelope",
        inSR=5070, spatialRel="esriSpatialRelIntersects", where="Category = 1",
        outFields="Name,LowPS,MinPS,MaxPS", returnGeometry="false", f="json"))
    r.raise_for_status()
    d = r.json()
    rows = [f["attributes"] for f in d.get("features", [])]
    visible = [a for a in rows if (a.get("MinPS") or 0) <= RES_M <= (a.get("MaxPS") or 0)]
    visible.sort(key=lambda a: a.get("LowPS") or 0)
    items = [f"{a.get('Name')} (LowPS {a.get('LowPS') or 0:.2f} m)" for a in visible]
    return dict(items=items, truncated=bool(d.get("exceededTransferLimit")))


# ---------------------------------------------------------------- fetch
def plan_jobs(a) -> tuple[list[str], list[Job]]:
    """Every component raster of every --bboxes file, with its state."""
    if a.prefix and len(a.bboxes) > 1:
        sys.exit("--prefix needs exactly one --bboxes file")
    prefixes = [a.prefix or prefix_of(p, "_bboxes.parquet") for p in a.bboxes]
    if len(set(prefixes)) < len(prefixes):
        sys.exit(f"a prefix appears twice in --bboxes: {prefixes}")
    jobs = []
    for k, (path, prefix) in enumerate(zip(a.bboxes, prefixes), 1):
        for comp, g in grids_of(read_bboxes(path)).items():
            j = Job(f"p{k}", prefix, comp, g, dem_path(a.gis_dir, prefix, comp))
            if j.path.exists() and not a.overwrite:
                j.why = raster_mismatch(j.path, g, a.source)
                j.state = "stale" if j.why else "exists"
            jobs.append(j)
    return prefixes, jobs


def build_from_tiles(g: Grid, by_tile: dict, remote: bool) -> tuple[np.ndarray, dict, dict, str]:
    """Clip the tnm tiles under `g`: the downloaded files, or their COG blocks over HTTPS."""
    tiles =[by_tile[t] for t in tiles_for(g.lonlat_bounds()) if t in by_tile]
    if remote:
        sources = ["/vsicurl/" + t["url"] for t in tiles]
    else:
        sources = [str(t["local"]) for t in tiles]
    arr, used = clip_tiles(sources, g)
    inputs = [f"{t['title']} | {t['url']}" for t in tiles]
    published = {t["tile"]: t["published"] for t in tiles}
    entry = dict(remote=bool(remote), inputs=inputs, published=published)
    tags = dict(source=SOURCE_TAG["tnm"], inputs="; ".join(inputs))
    return arr, tags, entry, f"{len(used)} tile(s)"


def build_from_imageserver(sess, g: Grid, cache: Path) -> tuple[np.ndarray, dict, dict, str]:
    """Mosaic the exportImage requests for `g`, reusing those already in `cache`."""
    subgrids = is_subgrids(g)
    arr = np.full((g.height, g.width), NODATA, np.float32)
    for r0, c0, sub in subgrids:
        path = cache / is_cache_name(sub)
        if not path.exists():
            is_export(sess, sub, path)
        arr[r0:r0 + sub.height, c0:c0 + sub.width] = is_read(path, sub)
    catalog = is_catalog(sess, g)
    items = catalog["items"]
    entry = dict(requests=len(subgrids), service=IS_URL, catalog_items_at_10m=items,
                 catalog_truncated=catalog["truncated"])
    tags = dict(source=SOURCE_TAG["imageserver"], inputs="; ".join(items))
    best = items[0] if items else "?"
    return arr, tags, entry, f"{len(subgrids)} request(s); best catalog item {best}"


def build_one(a, sess, g: Grid, by_tile: dict) -> tuple[np.ndarray, dict, dict, str]:
    """One grid's raster from the selected source, with its GeoTIFF tags, its manifest entry and
    a note for the log."""
    if a.source == "tnm":
        return build_from_tiles(g, by_tile, a.remote)
    return build_from_imageserver(sess, g, a.raw_dir / "imageserver")


def pick_donors(jobs: list[Job]) -> dict[tuple, Job]:
    """Per grid, the job whose raster serves every job on that grid: an existing raster if the
    call has one, else the first job that builds it.

    lobes.py can give two selections the same component, and an identical grid means an identical
    raster, so it is built once and hardlinked under each other prefix: `sample` and later readers
    find a raster by its <prefix>_c<k>.tif name alone."""
    donors: dict[tuple, Job] = {}
    for j in jobs:
        if j.state == "exists":
            donors.setdefault(j.key, j)
    for j in jobs:
        donors.setdefault(j.key, j)
    return donors


def job_action(j: Job, donor: Job) -> str:
    """What `fetch` does for job j, as listed in the plan."""
    if j.state == "exists":
        return "exists, kept"
    if donor is j:
        action = "build"
    elif donor.state == "exists":
        action = f"same grid as {donor.name}: hardlink its existing raster"
    else:
        action = f"same grid as {donor.name}: hardlink once built"
    if j.state == "stale":
        action = f"STALE ({j.why}); {action}"
    return action


def print_plan(labels: list[str], prefixes: list[str], jobs: list[Job], donors: dict) -> None:
    for label, prefix in zip(labels, prefixes):
        mine = [j for j in jobs if j.label == label]
        n_exists = sum(j.state == "exists" for j in mine)
        n_stale = sum(j.state == "stale" for j in mine)
        n_todo = sum(j.state == "todo" for j in mine)
        print(f"[{label}] {prefix}: {len(mine)} bbox(es); {n_exists} raster(s) exist, "
              f"{n_stale} stale, {n_todo} to make", flush=True)
        for j in mine:
            g = j.grid
            west, south, east, north = g.lonlat_bounds()
            print(f"  c{j.comp}: {g.width} x {g.height} px ({g.width * RES_M / 1e3:.1f} x "
                  f"{g.height * RES_M / 1e3:.1f} km), lon {west:.4f}..{east:.4f}, "
                  f"lat {south:.4f}..{north:.4f}: {job_action(j, donors[j.key])}", flush=True)


def print_tiles(tiles: list[dict], jobs: list[Job], grids: dict[tuple, Grid]) -> None:
    """The tnm tile table, with a warning for each tile whose newest API item is not the file
    under current/."""
    users = {}
    for key in grids:
        users[key] = [j.name for j in jobs if j.state != "exists" and j.key == key]
    print(f"  {len(tiles)} tile(s):", flush=True)
    print(f"  {'tile':<8} {'published':<11} {'size':>10} {'to get':>10} {'remote <=':>10}  "
          f"{'cached':<6} {'=current':<8} used by", flush=True)
    for t in tiles:
        names = []
        for key in t["grids"]:
            names += users[key]
        used_by = " ".join(names)
        if t["url"] is None:
            print(f"  {t['tile']:<8} NO 3DEP 1/3\" TILE IN THE API   used by {used_by}",
                  flush=True)
            continue
        size = "?" if t["size"] is None else gb(t["size"])
        print(f"  {t['tile']:<8} {t['published']:<11} {size:>10} {gb(t['to_get']):>10} "
              f"{gb(sum(t['remote'].values())):>10}  {str(t['cached']):<6} "
              f"{str(t['current_same']):<8} {used_by}", flush=True)
        if not t["current_same"]:
            print(f"    WARNING {t['tile']}: newest API item's ETag differs from "
                  f"current/ (or current/ is absent); check {t['url']}", flush=True)


def warn_tiles(tiles: list[dict], by_tile: dict) -> None:
    """Warn about areas with no tile and about tiles republished after REPUBLISHED_AFTER."""
    missing = [t["tile"] for t in tiles if t["url"] is None]
    if missing:
        print(f"  WARNING: no tile for {missing}; that area stays nodata "
              f"(sample will report any vertex it costs)", flush=True)
    late = [t for t in by_tile.values() if t["published"] > REPUBLISHED_AFTER]
    if late:
        late_list = ", ".join(f"{t['tile']} ({t['published']})" for t in late)
        print(f"  WARNING: {len(late)} tile(s) published after {REPUBLISHED_AFTER}: {late_list}"
              ". pull_gee.py took elevation from Earth Engine's USGS/3DEP/10m_collection,"
              " whose version of these tiles is not checked; on them the level-set test may"
              " read a newer surface than the features saw (docstring note 1).", flush=True)


def fetched_alone(jobs: list[Job], donors: dict, label: str) -> set:
    """The grids prefix `label` would have to fetch were it run alone."""
    keys = set()
    for j in jobs:
        if j.label == label and j.state != "exists" and donors[j.key].state != "exists":
            keys.add(j.key)
    return keys


def mode_sizes(a, keys: set, grids: dict[tuple, Grid], by_tile: dict) -> list[int]:
    """Bytes to fetch the grids `keys`, one value per SIZE_COLUMNS[a.source] column. Whole
    tiles and exportImage requests that several grids need are counted once."""
    raster_bytes = sum(grids[k].width * grids[k].height * 4 for k in keys)
    if a.source == "tnm":
        whole = 0
        remote = 0
        for t in by_tile.values():
            if keys & set(t["grids"]):
                whole += t["to_get"]
            for key, n in t["remote"].items():
                if key in keys:
                    remote += n
        return [whole, remote, raster_bytes]
    # imageserver: the pixels requested times 4 bytes, before compression
    cache = a.raw_dir / "imageserver"
    request_bytes = {}
    for key in keys:
        for _, _, sub in is_subgrids(grids[key]):
            request_bytes[is_cache_name(sub)] = sub.width * sub.height * 4
    uncached = 0
    for name, n in request_bytes.items():
        if not (cache / name).exists():
            uncached += n
    return [uncached, raster_bytes]


def print_size_row(name: str, n, cells: list[str]) -> None:
    line = f"    {name:<6} {n:>5}"
    for cell in cells:
        line += f" {cell:>14}"
    print(line, flush=True)


def plan_sizes(a, sess, jobs: list[Job], donors: dict, grids: dict[tuple, Grid],
               labels: list[str], mode: str, free: int) -> tuple[dict, int, int] | None:
    """List the tiles (tnm) and the sizes per prefix and for the call; return the tiles by name,
    the network bytes and the disk bytes of the selected mode, or None when a tile cannot be
    sized. For tnm both modes are listed, so one dry run settles the choice."""
    tnm = a.source == "tnm"
    by_tile: dict[str, dict] = {}
    if tnm:
        tiles = tnm_plan(sess, grids, a.raw_dir)
        print_tiles(tiles, jobs, grids)
        if any(t["url"] and t["size"] is None for t in tiles):
            print("  ERROR: HEAD failed for a tile; cannot size the download", file=sys.stderr)
            return None
        by_tile = {t["tile"]: t for t in tiles if t["url"]}
        warn_tiles(tiles, by_tile)

    total = mode_sizes(a, set(grids), grids, by_tile)
    print("  size by prefix, each as if fetched alone; 'call' is this run, with shared tiles "
          "and identical grids counted once:", flush=True)
    print_size_row("", "grids", SIZE_COLUMNS[a.source])
    for label in labels:
        own = fetched_alone(jobs, donors, label)
        print_size_row(label, len(own), [gb(v) for v in mode_sizes(a, own, grids, by_tile)])
    print_size_row("call", len(grids), [gb(v) for v in total])

    remote = tnm and a.remote
    if remote:
        net = total[1]
        what = "upper bound on COG blocks read"
    elif tnm:
        net = total[0]
        what = "whole tiles still to download"
    else:
        net = total[0]
        what = f"upper bound, exportImage requests of <= {IS_TILE_PX} px"
    out_bytes = total[-1]
    disk = out_bytes if remote else out_bytes + net     # --remote stores no tiles
    print(f"  selected ({mode}): network {gb(net)} ({what}); disk {gb(disk)} incl. rasters "
          f"<= {gb(out_bytes)}; free now {gb(free)}", flush=True)
    return by_tile, net, disk


def build_rasters(a, sess, build: list[Job], by_tile: dict, t0: float) -> tuple[dict, set]:
    """Build and write each donor raster; returns the manifest entries by grid and the grids
    that came out all nodata (not written)."""
    entries: dict[tuple, dict] = {}
    failed: set[tuple] = set()
    for j in build:
        g = j.grid
        arr, tags, entry, note = build_one(a, sess, g, by_tile)
        vf = valid_frac(arr)
        if vf == 0:
            print(f"  {j.name}: ERROR every cell is nodata ({note}); raster not written",
                  file=sys.stderr, flush=True)
            failed.add(j.key)
            continue
        write_raster(arr, g, j.path, dict(tags, res_m=RES_M, vertical="NAVD88 m"))
        entries[j.key] = dict(bounds=g.bounds, width=g.width, height=g.height, valid_frac=vf,
                              created=dt.datetime.now().isoformat(timespec="seconds"), **entry)
        update_manifest(manifest_path(a.gis_dir, j.prefix), j.prefix, a.source, j.comp,
                        dict(entries[j.key], path=str(j.path)))
        print(f"  {j.name}: {j.path} {g.width}x{g.height}, valid {vf:.4f}, {note} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return entries, failed


def donor_entry(a, donor: Job, entries: dict) -> dict:
    """The manifest entry a linked raster copies: the one just built, else the donor's own
    manifest entry, else just its grid."""
    entry = entries.get(donor.key)
    if not entry:
        manifest = read_manifest(manifest_path(a.gis_dir, donor.prefix))
        entry = manifest.get("components", {}).get(str(donor.comp))
    if not entry:
        entry = dict(bounds=donor.grid.bounds, width=donor.grid.width, height=donor.grid.height)
    return {k: v for k, v in entry.items() if k not in ("path", "linked_from", "link", "source")}


def link_rasters(a, link: list[Job], donors: dict, entries: dict, failed: set) -> None:
    """Give each remaining job its donor's raster (a hardlink, or a copy) and manifest entry."""
    for j in link:
        if j.key in failed:
            continue
        donor = donors[j.key]
        entry = donor_entry(a, donor, entries)
        kind = link_raster(donor.path, j.path)
        update_manifest(manifest_path(a.gis_dir, j.prefix), j.prefix, a.source, j.comp,
                        dict(entry, path=str(j.path), linked_from=str(donor.path), link=kind))
        print(f"  {j.name}: {j.path.name} = {donor.name} {donor.path.name} ({kind})", flush=True)


def cmd_fetch(a) -> int:
    prefixes, jobs = plan_jobs(a)
    labels = [f"p{k}" for k in range(1, len(prefixes) + 1)]
    tnm = a.source == "tnm"
    if tnm and a.remote:
        mode = "--remote COG block reads"
    elif tnm:
        mode = "whole tiles"
    else:
        mode = "exportImage requests"
    print(f"fetch: {len(prefixes)} bboxes file(s), {len(jobs)} component raster(s), source "
          f"{a.source} ({mode}); rasters in {a.gis_dir / 'dem'}, cache {a.raw_dir}", flush=True)

    donors = pick_donors(jobs)
    build = [j for j in jobs if j.state != "exists" and donors[j.key] is j]
    link = [j for j in jobs if j.state != "exists" and donors[j.key] is not j]
    print_plan(labels, prefixes, jobs, donors)
    if not build and not link:
        print("nothing to do (use --overwrite to rebuild)", flush=True)
        return 0

    grids = {j.key: j.grid for j in build}
    sess = session()
    free = min(free_bytes(a.raw_dir), free_bytes(a.gis_dir))
    by_tile, net, disk = {}, 0, 0
    if grids:
        sized = plan_sizes(a, sess, jobs, donors, grids, labels, mode, free)
        if sized is None:
            return 1
        by_tile, net, disk = sized
    else:
        print(f"  nothing to fetch: the {len(link)} raster(s) to make are hardlinks of "
              f"existing ones", flush=True)

    # Nothing large without a yes: the gate applies to the whole call's total.
    stale = [j.name for j in jobs if j.state == "stale"]
    over = net > a.max_gb * 1e9
    low = bool(grids) and free - disk < a.min_free_gb * 1e9
    if not a.yes:
        if stale:
            print(f"  NOTE: {len(stale)} stale raster(s) {stale}; --yes refuses until "
                  f"--overwrite", flush=True)
        if over:
            print(f"  NOTE: exceeds --max-gb {a.max_gb}; --yes would refuse", flush=True)
        if low:
            print(f"  NOTE: would leave < {a.min_free_gb} GB free; --yes would refuse",
                  flush=True)
        print("  dry run: nothing downloaded or written. Re-run with --yes.", flush=True)
        return 0
    if stale:
        print(f"REFUSED: {len(stale)} existing raster(s) are not their bbox's grid from "
              f"{a.source}: {stale}. Pass --overwrite to rebuild them.", file=sys.stderr)
        return 2
    if over:
        hint = ", or use --remote" if tnm and not a.remote else ""
        print(f"REFUSED: {gb(net)} exceeds --max-gb {a.max_gb}. Raise --max-gb deliberately"
              f"{hint}.", file=sys.stderr)
        return 2
    if low:
        print(f"REFUSED: would leave {gb(free - disk)} free (< {a.min_free_gb} GB).",
              file=sys.stderr)
        return 2

    t0 = time.time()
    if tnm and not a.remote:
        for t in by_tile.values():
            if t["cached"]:
                print(f"  {t['tile']}: cached {t['local']}", flush=True)
                continue
            print(f"  {t['tile']}: downloading {gb(t['size'])} -> {t['local']}", flush=True)
            download(sess, t["url"], t["local"], t["size"])
    entries, failed = build_rasters(a, sess, build, by_tile, t0)
    link_rasters(a, link, donors, entries, failed)

    n_built = sum(j.key not in failed for j in build)
    n_linked = sum(j.key not in failed for j in link)
    print(f"fetch done: {n_built} raster(s) built, {n_linked} linked, in {a.gis_dir / 'dem'}; "
          f"manifests <prefix>_dem.json ({time.time() - t0:.0f}s)", flush=True)
    bad = [j.name for j in build + link if j.key in failed]
    if bad:
        print(f"FAILED: no elevation at all for {bad}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- sample
def bilinear(arr: np.ndarray, t: Affine, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear value at (x, y) from the four surrounding cell centres.

    `arr` holds NaN for nodata; a position with any NaN neighbour, or without four neighbours
    inside the array, gets NaN, so nothing is extrapolated. Nearest-neighbour would make z step
    along an edge; cubic would overshoot at breaks of slope (levees, canal banks) and could fake
    or hide a 0.3 m level set.
    """
    if t.b != 0 or t.d != 0:
        raise ValueError("rotated rasters are not supported")
    c = (np.asarray(x, float) - t.c) / t.a - 0.5
    r = (np.asarray(y, float) - t.f) / t.e - 0.5
    c0 = np.floor(c).astype(np.int64)
    r0 = np.floor(r).astype(np.int64)
    fc, fr = c - c0, r - r0
    h, w = arr.shape
    out = np.full(len(c), np.nan)
    inside = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < w) & (r0 + 1 < h)
    i = np.flatnonzero(inside)
    r0, c0, fc, fr = r0[i], c0[i], fc[i], fr[i]
    out[i] = (arr[r0, c0] * (1 - fc) * (1 - fr) + arr[r0, c0 + 1] * fc * (1 - fr)
              + arr[r0 + 1, c0] * (1 - fc) * fr + arr[r0 + 1, c0 + 1] * fc * fr)
    return out


def read_window(path: Path, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, Affine]:
    """The part of a raster around a set of positions, nodata as NaN."""
    with rasterio.open(path) as src:
        t = src.transform
        c0 = max(math.floor((xs.min() - t.c) / t.a) - SRC_PAD_PX, 0)
        c1 = min(math.ceil((xs.max() - t.c) / t.a) + SRC_PAD_PX, src.width)
        r0 = max(math.floor((ys.max() - t.f) / t.e) - SRC_PAD_PX, 0)
        r1 = min(math.ceil((ys.min() - t.f) / t.e) + SRC_PAD_PX, src.height)
        if c1 <= c0 or r1 <= r0:
            return np.full((1, 1), np.nan), t
        win = Window(c0, r0, c1 - c0, r1 - r0)
        a = src.read(1, window=win).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        a[~np.isfinite(a)] = np.nan
        return a, src.window_transform(win)


def read_vertices(path: Path) -> pd.DataFrame:
    """The lobes.py vertex table, exiting unless every position is finite and every normal is a
    unit vector."""
    vt = pd.read_parquet(path)
    miss = [c for c in VERTEX_COLS if c not in vt.columns]
    if miss:
        sys.exit(f"{path}: missing columns {miss}")
    vt["component"] = vt["component"].astype(int)
    bad = ~np.isfinite(vt[["x", "y", "nx", "ny"]].to_numpy(float)).all(axis=1)
    if bad.any():
        sys.exit(f"{path}: {bad.sum():,} vertices with non-finite x, y, nx or ny")
    norm = np.hypot(vt["nx"].to_numpy(float), vt["ny"].to_numpy(float))
    unit = np.abs(norm - 1) < UNIT_NORMAL_TOL
    if not np.all(unit):
        sys.exit(f"{path}: {(~unit).sum():,} normals are not unit vectors")
    return vt


def check_rasters(bb: pd.DataFrame, gis_dir: Path, prefix: str) -> tuple[dict, list[str]]:
    """Every bbox must have an EPSG:5070 raster that contains it."""
    paths, errors = {}, []
    tol = 1e-6
    for r in bb.itertuples():
        p = dem_path(gis_dir, prefix, int(r.component))
        if not p.exists():
            errors.append(f"c{r.component}: no raster {p}")
            continue
        with rasterio.open(p) as src:
            if src.crs is None or src.crs.to_epsg() != 5070:
                errors.append(f"c{r.component}: {p.name} is {src.crs}, expected EPSG:5070")
                continue
            b = src.bounds
            contains = (b.left <= r.minx + tol and b.bottom <= r.miny + tol
                        and b.right >= r.maxx - tol and b.top >= r.maxy - tol)
            if not contains:
                bbox = (round(r.minx), round(r.miny), round(r.maxx), round(r.maxy))
                errors.append(f"c{r.component}: {p.name} bounds {tuple(round(v) for v in b)} "
                              f"do not contain bbox {bbox}")
                continue
        paths[int(r.component)] = p
    return paths, errors


def load_lobes(lpath: Path, vt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lobe polygons, prepared: each vertex's own lobe (aligned with vt's rows), every lobe,
    and every lobe's component. Exits when the lobes file does not belong to the vertex table."""
    miss = [c for c in ("lobe_id", "component", "geometry")
            if c not in pq.ParquetFile(lpath).schema_arrow.names]
    if miss:
        sys.exit(f"FAILED: {lpath}: missing columns {miss}")
    lobes = gpd.read_parquet(lpath, columns=["lobe_id", "component", "geometry"])
    if lobes.crs is not None and lobes.crs.to_epsg() != 5070:
        lobes = lobes.to_crs(5070)
    if lobes["lobe_id"].duplicated().any():
        sys.exit(f"FAILED: {lpath}: lobe_id is not unique")
    ix = pd.Index(lobes["lobe_id"]).get_indexer(vt["lobe_id"])
    unknown = ix < 0
    if unknown.any():
        eg = pd.unique(vt["lobe_id"].to_numpy()[unknown])[:3].tolist()
        sys.exit(f"FAILED: {unknown.sum():,} of {len(vt):,} vertices have a lobe_id that is not "
                 f"in {lpath} (e.g. {eg}); the vertices and lobes are not from one lobes.py run")
    lobe_comp = lobes["component"].astype(int).to_numpy()
    off = lobe_comp[ix] != vt["component"].to_numpy()
    if off.any():
        sys.exit(f"FAILED: {off.sum():,} vertices sit in a lobe of another component in {lpath}")
    geoms = lobes.geometry.to_numpy()
    shapely.prepare(geoms)
    return geoms[ix], geoms, lobe_comp


def check_normals(vt: pd.DataFrame, own: np.ndarray) -> pd.DataFrame:
    """Per component, the share of up to NORMAL_SAMPLE seeded vertices whose point NORMAL_EPS_M
    along (nx, ny) is inside their own lobe: ~1 for inward normals, ~0 for a flipped orientation.

    A flipped ring would flip the sign of every dz, which nothing downstream could notice. The
    share is per component because a pooled one hides a flip confined to one component; corner
    vertices and sub-metre zig-zags fail legitimately, hence NORMAL_MIN_SHARE below 1."""
    rng = np.random.default_rng(0)
    comp = vt["component"].to_numpy()
    x = vt["x"].to_numpy(float)
    y = vt["y"].to_numpy(float)
    nx = vt["nx"].to_numpy(float)
    ny = vt["ny"].to_numpy(float)
    rows = []
    for c in np.unique(comp):
        i = np.flatnonzero(comp == c)
        if len(i) > NORMAL_SAMPLE:
            i = np.sort(rng.choice(i, NORMAL_SAMPLE, replace=False))
        inside = shapely.contains_xy(own[i], x[i] + NORMAL_EPS_M * nx[i],
                                     y[i] + NORMAL_EPS_M * ny[i])
        rows.append(dict(component=int(c), checked=len(i), share=float(inside.mean())))
    return pd.DataFrame(rows)


def probe_flags(own: np.ndarray, geoms: np.ndarray, lobe_comp: np.ndarray, comp: np.ndarray,
                xin, yin, xout, yout) -> dict[str, np.ndarray]:
    """Where the z_in / z_out probes land relative to the lobes (interior only: a probe on a
    boundary counts as outside):

        in_probe_inside        the inward probe is inside the vertex's own lobe
        out_probe_outside      the outward probe is not inside the vertex's own lobe
        out_probe_in_any_lobe  the outward probe is inside some lobe lobes.py wrote for the
                               vertex's component, any role, its own included

    A lobe narrower than the offset, a bend or a narrow gap puts probes on the wrong ground. That
    follows from the offset, so the probes are flagged, never dropped; aotil/edge_layers.py
    decides. The outward probes go into an STRtree that each lobe queries with "contains", so
    GEOS tests candidates against prepared polygons (the reverse query is ~40x slower)."""
    flags = dict(in_probe_inside=shapely.contains_xy(own, xin, yin),
                 out_probe_outside=~shapely.contains_xy(own, xout, yout),
                 out_probe_in_any_lobe=np.zeros(len(own), bool))
    for c in np.unique(comp):
        i = np.flatnonzero(comp == c)
        tree = shapely.STRtree(shapely.points(xout[i], yout[i]))
        hit = tree.query(geoms[lobe_comp == c], predicate="contains")[1]
        flags["out_probe_in_any_lobe"][i[hit]] = True
    return flags


def report_normals(nc: pd.DataFrame, lpath: Path) -> bool:
    """Print the normal check; False when a component fails it."""
    print(f"  normals ({lpath.name}): share of up to {NORMAL_SAMPLE:,} vertices per component "
          f"whose point {NORMAL_EPS_M} m along (nx, ny) is inside their lobe; each must "
          f"reach {NORMAL_MIN_SHARE}", flush=True)
    for r in nc.itertuples():
        flag = "" if r.share >= NORMAL_MIN_SHARE else "  FAIL"
        print(f"    c{r.component}: {r.share:.4f} of {r.checked:,}{flag}", flush=True)
    flipped = nc.component[nc.share < NORMAL_MIN_SHARE].tolist()
    if flipped:
        print(f"FAILED: (nx, ny) do not point into the lobes of components {flipped}; "
              f"z_in/z_out would be swapped", file=sys.stderr)
        return False
    return True


def warn_sources(manifest: Path, entries: dict, paths: dict) -> None:
    """Print the DEM source of the rasters sampled; warn when they mix sources or sit on tiles
    republished after REPUBLISHED_AFTER (note 1)."""
    used = sorted({str(entries.get(str(c), {}).get("source")) for c in paths})
    where = manifest.name if manifest.exists() else "no manifest"
    print(f"  DEM source(s): {', '.join(used)} ({where})", flush=True)
    if len(used) > 1:
        print("  WARNING: components come from different DEM sources; the two differ by "
              "decimetres (see SOURCE_TAG in pull_dem.py). Re-fetch with one --source and "
              "--overwrite.", flush=True)
    late = set()
    for c in paths:
        published = entries.get(str(c), {}).get("published", {})
        for tile, date in published.items():
            if date > REPUBLISHED_AFTER:
                late.add(f"{tile} ({date})")
    if late:
        print(f"  WARNING: tiles published after {REPUBLISHED_AFTER}: {', '.join(sorted(late))}; "
              f"which version pull_gee.py's Earth Engine elevation used is not checked "
              f"(docstring note 1)", flush=True)


def sample_elevations(paths: dict, comp: np.ndarray, x, y, xin, yin, xout, yout):
    """Bilinear z at the vertices and at both probes, read component by component from that
    component's raster."""
    z = np.full(len(comp), np.nan)
    z_in = np.full(len(comp), np.nan)
    z_out = np.full(len(comp), np.nan)
    for c in sorted(set(comp)):
        i = np.flatnonzero(comp == c)
        xs = np.concatenate([x[i], xin[i], xout[i]])
        ys = np.concatenate([y[i], yin[i], yout[i]])
        arr, t = read_window(paths[c], xs, ys)
        z[i] = bilinear(arr, t, x[i], y[i])
        z_in[i] = bilinear(arr, t, xin[i], yin[i])
        z_out[i] = bilinear(arr, t, xout[i], yout[i])
    return z, z_in, z_out


def print_probe_table(res: pd.DataFrame, offset_m: float) -> None:
    print(f"  probes at {offset_m:g} m, share of vertices: z_in probe inside own lobe | z_out "
          f"probe outside own lobe | z_out probe inside any lobe of the component", flush=True)
    print(f"  {'comp':>4} {'role':<10} {'vertices':>9} {'in_inside':>10} {'out_outside':>12} "
          f"{'out_any':>8}", flush=True)
    for (c, role), g in res.groupby(["component", "role"]):
        print(f"  {c:>4} {role:<10} {len(g):>9,} {g.in_probe_inside.mean():>10.3f} "
              f"{g.out_probe_outside.mean():>12.3f} {g.out_probe_in_any_lobe.mean():>8.3f}",
              flush=True)


def print_z_table(res: pd.DataFrame) -> None:
    print(f"  {'comp':>4} {'vertices':>9} {'ok':>9} {'miss z':>7} {'z_in':>6} {'z_out':>6}  "
          f"{'z min':>7} {'median':>7} {'max':>7}  {'dz med':>7}", flush=True)
    for c, g in res.groupby("component"):
        ok = g[g.dem_ok]
        dz = ok.z_out - ok.z_in
        if len(ok):
            z_range = (f"{ok.z.min():7.2f} {ok.z.median():7.2f} {ok.z.max():7.2f}  "
                       f"{dz.median():7.2f}")
        else:
            z_range = f"{'-':>7} {'-':>7} {'-':>7}  {'-':>7}"
        print(f"  {c:>4} {len(g):>9,} {len(ok):>9,} {g.z.isna().sum():>7,} "
              f"{g.z_in.isna().sum():>6,} {g.z_out.isna().sum():>6,}  {z_range}", flush=True)


def cmd_sample(a) -> int:
    prefix = a.prefix or prefix_of(a.vertices, "_vertices.parquet")
    out = a.out
    if out is None:
        out = RESULTS / dt.date.today().isoformat() / "boundary" / f"{prefix}_vertices_z.parquet"
    if out.exists() and not a.overwrite:
        print(f"[{prefix}] {out} exists -- skipping (use --overwrite to redo)", flush=True)
        return 0
    bpath = a.bboxes or a.vertices.with_name(f"{prefix}_bboxes.parquet")
    lpath = a.lobes or a.vertices.with_name(f"{prefix}_lobes.parquet")
    if a.lobes and not a.lobes.exists():
        sys.exit(f"FAILED: --lobes {a.lobes} does not exist")
    t0 = time.time()

    vt = read_vertices(a.vertices)
    bb = read_bboxes(bpath)
    print(f"[{prefix}] {len(vt):,} vertices in {vt.component.nunique()} component(s); "
          f"{len(bb)} bbox(es) from {bpath}; offset {a.offset_m:g} m", flush=True)

    errors = [f"c{c}: vertices but no bbox" for c in sorted(set(vt.component) - set(bb.component))]
    paths, raster_errors = check_rasters(bb, a.gis_dir, prefix)
    errors += raster_errors
    if errors:
        print("FAILED: bboxes not covered by their rasters:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    own = None
    if lpath.exists():
        own, geoms, lobe_comp = load_lobes(lpath, vt)
        normals = check_normals(vt, own)
        if not report_normals(normals, lpath):
            return 1
    else:
        print(f"  normals and probe flags: not computed ({lpath.name} not found)", flush=True)

    manifest = manifest_path(a.gis_dir, prefix)
    entries = read_manifest(manifest).get("components", {})
    warn_sources(manifest, entries, paths)

    x, y = vt["x"].to_numpy(float), vt["y"].to_numpy(float)
    nx, ny = vt["nx"].to_numpy(float), vt["ny"].to_numpy(float)
    d = a.offset_m
    xin, yin = x + d * nx, y + d * ny          # the normals point into the lobe
    xout, yout = x - d * nx, y - d * ny
    comp = vt["component"].to_numpy()
    z, z_in, z_out = sample_elevations(paths, comp, x, y, xin, yin, xout, yout)

    res = vt.copy()
    res["z"] = z
    res["z_in"] = z_in
    res["z_out"] = z_out
    res["dem_ok"] = np.isfinite(z) & np.isfinite(z_in) & np.isfinite(z_out)
    if own is not None:
        res = res.assign(**probe_flags(own, geoms, lobe_comp, comp, xin, yin, xout, yout))
        print_probe_table(res, d)
    print_z_table(res)

    out.parent.mkdir(parents=True, exist_ok=True)
    n_bad = int((~res.dem_ok).sum())
    if n_bad:
        mpath = out.with_name(out.stem + "_misses.csv")
        res.loc[~res.dem_ok, MISSES_COLS].to_csv(mpath, index=False)
        print(f"  {n_bad:,} vertices lack z, z_in or z_out; listed in {mpath}", flush=True)
        if not a.allow_missing:
            print(f"FAILED: {n_bad:,} of {len(res):,} vertices not fully covered "
                  f"(--allow-missing writes the table with dem_ok=False rows)", file=sys.stderr)
            return 1

    dem = {}
    for c, p in paths.items():
        dem[str(c)] = dict(path=str(p), source=entries.get(str(c), {}).get("source"))
    meta = dict(offset_m=d, interpolation="bilinear", crs=CRS_OUT, res_m=RES_M, dem=dem,
                lobes=None, probe_flags=[], normals=[],
                created=dt.datetime.now().isoformat(timespec="seconds"))
    if own is not None:
        meta.update(lobes=str(lpath), probe_flags=PROBE_COLS, normals=normals.to_dict("records"))
    table = pa.Table.from_pandas(res, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"pull_dem": json.dumps(meta).encode()})
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    print(f"[{prefix}] wrote {out}: {len(res):,} vertices, {int(res.dem_ok.sum()):,} with "
          f"z, z_in and z_out ({time.time() - t0:.1f}s)", flush=True)
    return 0


# ---------------------------------------------------------------- self-test (no network)
def self_test() -> int:
    """Offline checks of the grid, tile, bilinear and probe-flag code; 0 when all pass."""
    checks = []   # (passed, what)

    g = Grid.around(1315671.2, 857889.9, 1317669.0, 859890.1)
    checks.append((g.bounds == (1315670.0, 857880.0, 1317670.0, 859900.0),
                   f"grid snapping {g.bounds}"))
    checks.append(((g.width, g.height) == (200, 202), f"grid shape {g.width}x{g.height}"))
    checks.append((tiles_for((-82.31, 29.94, -82.29, 29.96)) == ["n30w083"], "one tile"))
    checks.append((tiles_for((-82.6, 29.7, -82.0, 30.2)) == ["n30w083", "n31w083"], "two tiles"))
    checks.append((tiles_for((-80.5, 25.0, -79.5, 25.5)) == ["n26w080", "n26w081"], "lon seam"))
    subs = is_subgrids(Grid(0, 0, 90000, 50000))
    covered = sum(s.width * s.height for _, _, s in subs)
    checks.append((len(subs) == 6 and covered == 9000 * 5000,
                   "imageserver split covers the grid once"))

    # bilinear is exact on a plane, NaN-strict, and honours cell centres
    t = Affine(10, 0, 1000, 0, -10, 2000)
    rr, cc = np.mgrid[0:50, 0:60]
    xc, yc = 1000 + (cc + 0.5) * 10, 2000 - (rr + 0.5) * 10
    plane = 0.01 * xc - 0.02 * yc + 5
    rng = np.random.default_rng(0)
    px = rng.uniform(1010, 1580, 500)
    py = rng.uniform(1520, 1990, 500)
    zb = bilinear(plane, t, px, py)
    checks.append((np.allclose(zb, 0.01 * px - 0.02 * py + 5, atol=1e-9),
                   "bilinear exact on a plane"))
    centre = bilinear(plane, t, np.r_[xc[3, 4]], np.r_[yc[3, 4]])[0]
    checks.append((np.isclose(centre, plane[3, 4]), "cell centre returns the cell value"))
    holed = plane.copy()
    holed[10, 10] = np.nan
    zz = bilinear(holed, t, np.r_[xc[10, 10] + 3, xc[30, 30]], np.r_[yc[10, 10] - 3, yc[30, 30]])
    checks.append((np.isnan(zz[0]) and np.isfinite(zz[1]),
                   "NaN neighbour -> NaN, others unaffected"))
    outside = bilinear(plane, t, np.r_[900.0], np.r_[1900.0])[0]
    checks.append((np.isnan(outside), "outside -> NaN"))

    # probe flags: a 60 m strip on a 500 m block, and a C-shaped lobe whose 80 m gap a z_out
    # probe crosses back into the lobe
    strip, block = shapely.box(0, 0, 1000, 60), shapely.box(0, -500, 1000, 0)
    c_shape = shapely.difference(shapely.box(2000, 0, 2300, 400),
                                 shapely.box(2050, 150, 2300, 230))
    geoms = np.array([strip, block, c_shape], dtype=object)
    vx, vy = np.r_[500.0, 500, 500, 2200], np.r_[60.0, 0, -500, 150]
    vny = np.r_[-1.0, 1, 1, -1]
    f = probe_flags(geoms[[0, 0, 1, 2]], geoms, np.zeros(3, int), np.zeros(4, int),
                    vx, vy + 100 * vny, vx, vy - 100 * vny)
    checks.append((f["in_probe_inside"].tolist() == [False, False, True, True],
                   f"in_probe_inside {f}"))
    checks.append((f["out_probe_outside"].tolist() == [True, True, True, False],
                   f"out_probe_outside {f}"))
    checks.append((f["out_probe_in_any_lobe"].tolist() == [False, True, False, True],
                   f"out_probe_in_any_lobe {f}"))
    f = probe_flags(geoms[[0, 0, 1, 2]], geoms, np.r_[1, 1, 1], np.zeros(4, int),
                    vx, vy + 100 * vny, vx, vy - 100 * vny)
    checks.append((not f["out_probe_in_any_lobe"].any(),
                   "out_probe_in_any_lobe ignores other components"))

    # the normal check is per component: one flipped component fails on its own
    vt = pd.DataFrame(dict(component=[0] * 50 + [1] * 50, x=np.tile(np.linspace(10, 990, 50), 2),
                           y=-500.0, nx=0.0, ny=np.r_[np.ones(50), -np.ones(50)]))
    nc = check_normals(vt, np.array([block] * 100, dtype=object))
    checks.append((nc.share.round(3).tolist() == [1.0, 0.0],
                   f"per-component normal check {nc.share.tolist()}"))

    failed = [what for passed, what in checks if not passed]
    for what in failed:
        print(f"  FAIL {what}")
    print(f"pull_dem.py self-test: {len(checks) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="DEM raster per bbox component")
    fetch.add_argument("--bboxes", type=Path, nargs="+", required=True,
                       help="one or more <prefix>_bboxes.parquet; the size gate applies to their "
                            "total, and shared tiles and identical grids are fetched once")
    fetch.add_argument("--source", choices=["tnm", "imageserver"], default="tnm")
    fetch.add_argument("--remote", action="store_true",
                       help="tnm: read only the needed COG blocks over HTTPS, no tile download")
    fetch.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    fetch.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="tile / request cache")
    fetch.add_argument("--prefix", help="default: from the bboxes file name (one --bboxes only)")
    fetch.add_argument("--max-gb", type=float, default=2.0,
                       help="refuse when the call's network total for the selected mode "
                            "exceeds it")
    fetch.add_argument("--min-free-gb", type=float, default=10.0,
                       help="refuse when the call would leave less free disk")
    fetch.add_argument("--yes", action="store_true", help="download; without it, list only")
    fetch.add_argument("--overwrite", action="store_true",
                       help="rebuild existing rasters, stale ones included (the --raw-dir cache "
                            "is still reused; delete it to re-download)")

    sample = sub.add_parser("sample", help="z, z_in, z_out at every vertex")
    sample.add_argument("--vertices", type=Path, required=True, help="<prefix>_vertices.parquet")
    sample.add_argument("--bboxes", type=Path,
                        help="default: <prefix>_bboxes.parquet beside --vertices")
    sample.add_argument("--lobes", type=Path,
                        help="default: <prefix>_lobes.parquet beside --vertices; when present, "
                             "normals are checked per component and the probe flags are added")
    sample.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    sample.add_argument("--prefix", help="default: from the vertices file name")
    sample.add_argument("--offset-m", type=float, default=100.0)
    sample.add_argument("--out", type=Path,
                        help="default: results/<today>/boundary/<prefix>_vertices_z.parquet")
    sample.add_argument("--allow-missing", action="store_true",
                        help="write the table even if some vertices lack z, z_in or z_out")
    sample.add_argument("--overwrite", action="store_true")

    sub.add_parser("selftest", help="offline checks of the grid, tiles and bilinear code")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        return cmd_fetch(a)
    if a.cmd == "sample":
        return cmd_sample(a)
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
