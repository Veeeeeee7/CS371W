#!/usr/bin/env python3
"""
pull_dem.py -- a 10 m DEM over each boundary-analysis bbox, then elevation at
every lobe-edge vertex and 100 m either side of it along the edge normal.

The boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md section 3.3; PROJECT-PLAN
O5, class E) asks of every 300 m window of Zone A edge whether it is an
elevation level set -- range within 0.76 m (half a 5-ft contour interval) or
0.3 m -- with a real drop across it, dz = z_out - z_in > 0. That needs z at each
vertex and 100 m inside and outside it. The displaced-edge null then moves every
edge by 300-1,000 m and asks again, so elevation is needed at positions nobody
knows in advance: a local raster per analysis bbox, not point samples.

Nothing here is a model feature. z, z_in and z_out live on the vertex table
under results/, never in dataset_FL.parquet; FEATURES.txt is untouched.

Two stages, run in order:

    fetch    <prefix>_bboxes.parquet ... -> <gis-dir>/dem/<prefix>_c<component>.tif
                                            + <gis-dir>/dem/<prefix>_dem.json
    sample   <prefix>_vertices.parquet   -> results/<today>/boundary/<prefix>_vertices_z.parquet

Both inputs come from aotil/lobes.py. `prefix` is <cell>_<variant>_S<S>, read off
the input file name. `fetch` takes the bboxes files of every prefix in one call,
so the size gate sees the whole run (note 5); `sample` runs once per prefix.

WHAT THIS SCRIPT DECIDES, AND WHY
=================================

1. THE SAME PRODUCT THE FEATURES SAW, NOT A CHECKED SAME VERSION.  The default
   source, `--source tnm`, is the USGS 3DEP 1/3 arc-second seamless DEM in
   1 x 1 degree GeoTIFF tiles from The National Map Access API. pull_gee.py
   computes elev/slope/curvature from Earth Engine's USGS/3DEP/10m_collection,
   which is built from the same 1/3" tiles. Same product, but not necessarily
   the same version: TNM republishes tiles, and 5 of the 16 the v1 bboxes need
   were republished in 2025-26 (n26w081 and n26w082 on 2026-02-25, n27w081 on
   2025-12-16, n30w082 on 2025-12-08, n28w081 on 2025-06-09). Which version
   Earth Engine served when pull_gee.py ran is not checked here (that needs
   Earth Engine). So `fetch` prints a WARNING naming every tile published after
   REPUBLISHED_AFTER (2024-01-01), the manifest records each raster's tile
   dates, and `sample` repeats the warning for the tiles under its rasters; on
   those tiles the level-set test may read a newer surface than the features
   were computed on.

   The API lists every historical version of a tile (n30w083 has four:
   2012-10-01, 2018-10-12, 2022-10-19, 2022-11-03). The newest publicationDate
   is kept. Checked: its ETag equals that of the tile under .../13/TIFF/current/,
   and the fetch repeats that check per tile and warns if they ever differ. The
   API's `sizeInBytes` is not trusted: it gives 491,502 bytes for the 2018
   n30w083 tile, which HEAD reports as 503,282,222. Sizes come from HEAD.

   A tile is 10,812 x 10,812 float32, up to about 0.5 GB (less where it is
   mostly sea: n25w081 is 4.6 MB). The tiles are Cloud Optimized GeoTIFFs
   (512 px blocks, LZW), so `--remote` reads only the blocks a bbox touches
   over HTTPS (GDAL /vsicurl/) instead of downloading whole tiles: same bytes,
   same result (checked: the 2 km box clipped from the downloaded n30w083 and
   via --remote gives identical arrays). Measured: a 2 km box read 3.3 MB
   (bound 4.4 MB) of a 0.495 GB tile; a 50 km box spanning n30w083 and n31w083
   is bounded by 0.13 GB against 0.975 GB of whole tiles. Without `--remote`
   tiles are downloaded whole into data/raw/3dep/ (resumable, size-checked) and
   reused across components and runs. The v1 bboxes are 45-138 km a side, and
   there the saving is smaller than on a 50 km box (sizes in note 5).

   `--source imageserver` is the smaller fallback: the 3DEP ImageServer's
   exportImage clips each bbox directly in EPSG:5070 at 10 m, in requests of at
   most 4,000 px a side, mosaicked here. It does NOT serve the 1/3" product: the
   service is a multi-resolution mosaic sorted by "Best", so at 10 m it
   resamples (bilinear, server side) the finest 3DEP source under each pixel.
   Where there is 1 m lidar that is the lidar: at the smoke-test box (Bradford
   County) the best catalog item is FL_Peninsular_FDEM_2018_D19_DRRA, ahead of
   the 1/3" tile n30w083. Measured there, imageserver minus tnm per 10 m cell:
   median |d| 0.07 m, 95th percentile 0.34 m, max 3.2 m; at the 134 test
   vertices the sign of z_out - z_in agreed at 133. So the two sources are not
   interchangeable for a 0.3 m test, and a run should use one source throughout.
   The manifest records the catalog items behind each bbox. A third route,
   Earth Engine's USGS/3DEP/10m clipped per bbox (with setDefaultProjection, see
   docs/COVARIATES.md section 2.1), is not implemented: it needs `earthengine
   authenticate` and gives nothing the tnm route does not.

2. OUTPUT GRID: EPSG:5070, 10 m, BILINEAR, ON ONE LATTICE.  Every raster is
   reprojected to NAD83 / Conus Albers at exactly 10 m, pixel edges on multiples
   of 10 m. Why not keep the native EPSG:4269 grid:
     - every other layer, the lobe polygons and the vertex table are EPSG:5070
       metres, and the displaced-edge null shifts edges by metres;
     - both sources land on the identical lattice, so switching source changes
       no downstream code and the two can be differenced cell by cell;
     - the native 1/3" cell in Florida (25-31 N) is 10.3 m N-S by 8.9-9.3 m E-W,
       so a 10 m grid is close to a 1:1 resample.
   The cost is a second interpolation (resample here, bilinear sample in
   `sample`), which smooths at the 10 m scale. Measured on a 1 km test square
   with 19 m of relief (steeper than most Zone A edges): sampling the 10 m
   raster instead of the native tile moves z, z_in and z_out by a median of
   0.009 m, 95th percentile 0.07 m, at most 0.18 m over 402 positions. That is
   5-8 times smaller than the gap between the two 3DEP sources at the same
   positions, and below the 0.3 m tolerance even at its maximum. NAD83 to NAD83
   means no datum shift; heights stay NAVD88 metres. nodata is -999999, the
   3DEP value.

3. BILINEAR SAMPLING, STRICT ABOUT NODATA.  z, z_in and z_out are bilinear in
   the four cell centres around each position. Nearest-neighbour would snap
   positions to the 10 m grid, so z along an edge sampled every 30 m would step
   with a discontinuity of up to half a cell times the slope; bilinear is
   continuous and exact on a plane. Cubic would overshoot at breaks of slope
   (levees, canal banks, the very drops the test looks for) and could fake or
   hide a 0.3 m level set. A position whose four neighbours are not all valid
   gets NaN: nothing is extrapolated, and the miss is reported.

4. COVERAGE IS CHECKED, NOT ASSUMED.  `sample` fails (exit 1) when a bbox has
   no raster, a raster is not EPSG:5070, or a raster does not contain its bbox.
   Every vertex must have z, z_in and z_out: misses are listed by component and
   written to <out>_misses.csv, and the table is not written unless
   `--allow-missing` (then `dem_ok` marks the gaps). When <prefix>_lobes.parquet
   sits next to the vertices, every vertex's lobe_id must be in it, under the
   same component, and the normals are checked per component: for up to 5,000
   seeded vertices of each, a point 0.5 m along (nx, ny) must fall inside the
   vertex's own lobe, and every component must reach 0.9 (corner vertices and
   sub-metre zig-zags fail legitimately; the 10 v1 components give 0.970-1.000).
   A flipped ring orientation would flip the sign of every dz, which nothing
   downstream could notice. The share is per component because a pooled one
   hides a flip confined to one: flipping all of a component holding 9% of the
   vertices takes a pooled share from 0.985 only to 0.89, and flipping 43% of
   it leaves 0.948, a pass.

5. NOTHING LARGE WITHOUT A YES, AND THE GATE SEES THE WHOLE RUN.  `fetch` takes
   the bboxes files of every prefix in one call. Without `--yes` it lists the
   rasters, the tiles and the sizes, per prefix (each as if fetched alone) and
   for the call, and exits; for tnm it gives both modes, whole tiles and the
   --remote bound, so one dry run settles the choice. With `--yes` it refuses
   when the selected mode's total for the call exceeds `--max-gb` (default 2.0,
   where the task brief says to ask first) or would leave less than
   `--min-free-gb` (default 10) free on the disk. A tile several prefixes need
   is counted and downloaded once. For `--remote` the size is an upper bound
   from the stored size of every block touched (remote_blocks_bytes); for
   `imageserver` it is the pixels requested times 4 bytes, before compression.
   Measured on the v1 bboxes (3 prefixes, 10 components, 8 distinct grids, 16
   tiles): whole tiles 4.534 / 3.307 / 4.543 GB per prefix (classic / S250 /
   S500), 5.30 GB for the call, plus rasters of at most 2.72 GB; --remote at
   most 1.468 / 0.972 / 1.409 GB, 3.10 GB for the call. Both are over 2 GB, so
   the real run needs a deliberate --max-gb.

   IDENTICAL GRIDS ARE BUILT ONCE.  lobes.py can give two selections the same
   component (v1: S250 c0 = S500 c1 and S250 c1 = S500 c3), and an identical
   Grid means an identical raster. It is built under the first prefix's name and
   HARDLINKED under each other one (copied where the filesystem has no
   hardlinks); that prefix's manifest entry records `linked_from` and `link`.
   A link rather than a manifest reference because `sample`, and every later
   reader, finds a raster by its <prefix>_c<k>.tif name alone, and a hardlink
   costs no disk. Rasters are only ever replaced by rename, never rewritten in
   place, so --overwrite on one prefix gives it a new file and leaves the
   other's as it was. An existing raster with the identical grid, under any
   prefix of the call, is linked without fetching anything.

   An existing raster is kept only if it is exactly its bbox's grid from the
   requested source (CRS, transform, size, `source` tag). One that differs is
   STALE: the prefix is the same for v1 and v2 (<cell>_<variant>_S<S>) and the
   gis-dir is not dated, so v2 bboxes would otherwise reuse v1 rasters by name.
   The dry run lists stale rasters and `--yes` refuses until `--overwrite`.

6. WHERE THE +-100 m PROBES LAND IS RECORDED, NOT FILTERED.  A lobe narrower
   than 100 m, or a bend, puts the z_in probe outside its own lobe, and a z_out
   probe can cross a narrow gap back into its lobe or into a neighbouring one;
   then dz compares the wrong ground. On the v1 vertices, per component, the
   z_in probe is inside its own lobe for 65-98% of the region-lobe vertices and
   44-74% of the AE ones; the z_out probe is back inside its own lobe for up to
   30% (classic c1, region) and inside some lobe of the component for 8-34% of
   region and 25-56% of AE vertices. That follows from the brief's 100 m, not
   from a coding error, so `sample` records it when the lobes file is present,
   from shapely.contains_xy against the lobe polygons (interior only: a probe
   on a boundary counts as outside):
       in_probe_inside        x + d n is inside the vertex's own lobe
       out_probe_outside      x - d n is not inside the vertex's own lobe
       out_probe_in_any_lobe  x - d n is inside some lobe of the vertex's
                              component, any role, its own included
   "Any lobe" means the lobes lobes.py wrote for that component (region,
   matched_A, AE), not all of the NFHL: a z_out probe in an unselected Zone A
   polygon or in Zone X is not flagged. Nothing here drops or shortens a probe;
   the level-set test (aotil/edge_layers.py, D7) decides. Like z, these are
   per-vertex metadata on a results table, never model features.

USAGE
=====
    python pull_dem.py selftest                               # no network
    python pull_dem.py fetch --bboxes results/$D/boundary/<p1>_bboxes.parquet \\
                                      results/$D/boundary/<p2>_bboxes.parquet ...
                                                              # rasters, tiles, sizes only
    python pull_dem.py fetch --bboxes ... --yes               # download tiles, clip
    python pull_dem.py fetch --bboxes ... --yes --remote      # read COG blocks only
    python pull_dem.py fetch --bboxes ... --source imageserver --yes
    python pull_dem.py sample --vertices results/$D/boundary/<prefix>_vertices.parquet \\
                              --out results/$D/boundary/<prefix>_vertices_z.parquet

The vertex table `sample` writes is the lobes.py vertex table plus z, z_in,
z_out (NAVD88 m), dem_ok and, when <prefix>_lobes.parquet is present,
in_probe_inside, out_probe_outside and out_probe_in_any_lobe (note 6).
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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

GIS_DIR = Path("data/interim/gis")
RAW_DIR = Path("data/raw/3dep")
RESULTS = Path("results")

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASET = "National Elevation Dataset (NED) 1/3 arc-second"
TNM_CURRENT = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
               "current/{tile}/USGS_13_{tile}.tif")
TILE_RE = re.compile(r"USGS_13_(n\d{2}w\d{3})")
IS_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

CRS_OUT = "EPSG:5070"
CRS_TILE = "EPSG:4269"    # NAD83 geographic, as the 1/3" tiles ship
RES_M = 10.0
NODATA = -999999.0        # the 3DEP nodata value
IS_TILE_PX = 4000         # per exportImage request side; the server allows 8,000
SRC_PAD_PX = 2            # source pixels kept around a window for the bilinear kernel
NORMAL_EPS_M = 0.5        # step along the inward normal for the orientation check
NORMAL_SAMPLE = 5000      # vertices checked per component
NORMAL_MIN_SHARE = 0.9    # every component must reach it
REPUBLISHED_AFTER = "2024-01-01"   # tnm tiles newer than this get a version warning (note 1)

SOURCE_TAG = {"tnm": "tnm 3DEP 1/3 arc-second",
              "imageserver": "3DEP ImageServer best-available, bilinear"}
PROBE_COLS = ["in_probe_inside", "out_probe_outside", "out_probe_in_any_lobe"]

DEFAULTS = dict(max_gb=2.0, min_free_gb=10.0, offset_m=100.0)

RETRIES = 4
CHUNK = 1 << 20
REMOTE_SLACK = 2 * 16384  # per block: GDAL /vsicurl/ reads whole 16 KB chunks
REMOTE_HEADER = 1 << 20   # per tile pass: header, IFDs, tile offset tables
TIMEOUT = (30, 300)       # connect, read seconds

BBOX_COLS = ["component", "minx", "miny", "maxx", "maxy"]
VERTEX_COLS = ["lobe_id", "zone", "role", "component", "ring", "s",
               "x", "y", "lon", "lat", "nx", "ny"]

GDAL_REMOTE = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                   GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                   GDAL_HTTP_MAX_RETRY=str(RETRIES), GDAL_HTTP_RETRY_DELAY="2")


# =========================================================================
# small helpers
# =========================================================================
def session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=RETRIES, connect=RETRIES, read=RETRIES, backoff_factor=1.5,
              status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET", "HEAD"]), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


def gb(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e9:.3f} GB"


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
    p = path.resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def read_bboxes(path: Path) -> pd.DataFrame:
    bb = pd.read_parquet(path)
    miss = [c for c in BBOX_COLS if c not in bb.columns]
    if miss:
        sys.exit(f"{path}: missing columns {miss}")
    bb = bb.copy()
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
        f = lambda v: math.floor(v / RES_M) * RES_M
        c = lambda v: math.ceil(v / RES_M) * RES_M
        return cls(f(minx), f(miny), c(maxx), c(maxy))

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
    """One raster `fetch` must end up with: component `comp` of `prefix`.

    state: "exists" (kept as it is), "stale" (exists but is not this grid from
    this source) or "todo"."""
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
    """'' when `path` is exactly `grid` from `source`, else what differs."""
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            return f"CRS {src.crs}"
        if (src.width, src.height) != (grid.width, grid.height) or \
                not src.transform.almost_equals(grid.transform, precision=1e-6):
            return (f"bounds {tuple(round(v) for v in src.bounds)} != bbox grid "
                    f"{tuple(round(v) for v in grid.bounds)}")
        tag = src.tags().get("source")
        if tag != SOURCE_TAG[source]:
            return f"source {tag!r}, not {source}"
    return ""


def link_raster(src: Path, dst: Path) -> str:
    """Make `dst` the raster `src` is: a hardlink, or a copy where the
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

    Tile nYYwXXX spans latitude YY-1..YY and longitude -XXX..-XXX+1; its 6-pixel
    (2") overlap with the neighbours is what feeds the bilinear kernel at a seam,
    so a bbox that merely approaches a degree line does not pull the next tile.
    """
    w, s, e, n = lonlat
    out = []
    for lat0 in range(math.floor(s), math.ceil(n)):
        for lon0 in range(math.floor(w), math.ceil(e)):
            out.append(f"n{lat0 + 1:02d}w{-lon0:03d}")
    return sorted(out)


def window_for(src, lonlat, pad: int = SRC_PAD_PX) -> Window | None:
    """The source window covering a NAD83 envelope plus `pad` pixels, clipped
    to the dataset; None when they do not overlap."""
    t = src.transform
    w, s, e, n = lonlat
    c0 = math.floor((w - t.c) / t.a) - pad
    c1 = math.ceil((e - t.c) / t.a) + pad
    r0 = math.floor((n - t.f) / t.e) - pad
    r1 = math.ceil((s - t.f) / t.e) + pad
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, src.width), min(r1, src.height)
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def write_raster(arr: np.ndarray, grid: Grid, out: Path, tags: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.tif")
    prof = dict(driver="GTiff", dtype="float32", count=1, width=grid.width,
                height=grid.height, crs=CRS_OUT, transform=grid.transform,
                nodata=NODATA, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, out)


def valid_frac(arr: np.ndarray) -> float:
    return float(np.mean(arr != NODATA)) if arr.size else 0.0


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def update_manifest(path: Path, prefix: str, source: str, comp: int, entry: dict) -> None:
    """One JSON per prefix, one entry per component raster: what it is and
    where its numbers came from."""
    m = read_manifest(path)
    m.update(prefix=prefix, crs=CRS_OUT, res_m=RES_M, nodata=NODATA,
             vertical="NAVD88 metres", resampling="bilinear")
    m.setdefault("components", {})[str(comp)] = {**entry, "source": source}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(json.dumps(m, indent=1, sort_keys=True), path)


# =========================================================================
# fetch: tnm
# =========================================================================
def tnm_query(sess: requests.Session, lonlat) -> list[dict]:
    items, offset = [], 0
    while True:
        r = sess.get(TNM_API, timeout=TIMEOUT, params=dict(
            datasets=TNM_DATASET, prodFormats="GeoTIFF", outputFormat="JSON",
            bbox=",".join(f"{v:.6f}" for v in lonlat), max=100, offset=offset))
        r.raise_for_status()
        d = r.json()
        if d.get("errors"):
            raise RuntimeError(f"TNM API errors: {d['errors']}")
        items += d.get("items", [])
        offset += len(d.get("items", []))
        if not d.get("items") or offset >= int(d.get("total", 0)):
            return items


def head(sess: requests.Session, url: str) -> tuple[int | None, str | None]:
    r = sess.head(url, timeout=TIMEOUT, allow_redirects=True)
    if r.status_code != 200:
        return None, None
    return int(r.headers.get("Content-Length", 0)), r.headers.get("ETag")


def tnm_plan(sess, grids: dict[tuple, Grid], raw_dir: Path) -> list[dict]:
    """Per tile: the grids it serves, its newest version, true size, what is
    already cached, and the --remote block bound for each grid it serves."""
    need: dict[str, list[tuple]] = {}
    for key, g in grids.items():
        for t in tiles_for(g.lonlat_bounds()):
            need.setdefault(t, []).append(key)

    best: dict[str, dict] = {}
    for g in grids.values():
        for it in tnm_query(sess, g.lonlat_bounds()):
            url = it.get("downloadURL") or ""
            m = TILE_RE.search(url)
            if not m or m.group(1) not in need or it.get("extent") != "1 x 1 degree":
                continue
            key = (it.get("publicationDate") or "", it.get("lastUpdated") or "")
            cur = best.get(m.group(1))
            if cur is None or key > cur["_key"]:
                best[m.group(1)] = dict(it, _key=key)

    tiles = []
    for t in sorted(need):
        it = best.get(t)
        if it is None:
            tiles.append(dict(tile=t, grids=need[t], url=None))
            continue
        size, etag = head(sess, it["downloadURL"])
        _, cur_etag = head(sess, TNM_CURRENT.format(tile=t))
        local = raw_dir / Path(it["downloadURL"]).name
        part = local.with_name(local.name + ".part")
        have = local.stat().st_size if local.exists() else (
            part.stat().st_size if part.exists() else 0)
        tiles.append(dict(
            tile=t, grids=need[t], url=it["downloadURL"], title=it.get("title"),
            published=it.get("publicationDate") or "", size=size, etag=etag,
            current_same=(etag is not None and etag == cur_etag), local=local,
            cached=(local.exists() and size is not None and local.stat().st_size == size),
            to_get=0 if size is None else max(size - have, 0),
            remote={} if size is None else remote_blocks_bytes(
                it["downloadURL"], {k: grids[k] for k in need[t]})))
    return tiles


def remote_blocks_bytes(url: str, grids: dict[tuple, Grid]) -> dict[tuple, int]:
    """Upper bound on what --remote reads of one tile for each grid.

    The stored (compressed) size of every full-resolution block the grid's
    window touches, read from the COG's own TileByteCounts, plus REMOTE_SLACK
    per block for GDAL's 16 KB read granularity and REMOTE_HEADER per pass for
    the header. Blocks x 4 bytes a pixel would count open water and nodata,
    which LZW stores in a few KB, at full size: on the v1 bboxes that gives
    4.58 GB for the call, this 3.10 GB. Each grid is clipped in its own pass,
    so blocks two grids share are read twice and counted twice."""
    with rasterio.Env(**GDAL_REMOTE), rasterio.open("/vsicurl/" + url) as src:
        bh, bw = src.block_shapes[0]
        out = {}
        for k, g in grids.items():
            win = window_for(src, g.lonlat_bounds())
            n = 0
            if win is not None:
                n = REMOTE_HEADER
                for by in range(win.row_off // bh, (win.row_off + win.height - 1) // bh + 1):
                    for bx in range(win.col_off // bw, (win.col_off + win.width - 1) // bw + 1):
                        size = src.get_tag_item(f"BLOCK_SIZE_{bx}_{by}", "TIFF", bidx=1)
                        n += (int(size) if size else bw * bh * 4) + REMOTE_SLACK
            out[k] = n
        return out


def download(sess: requests.Session, url: str, dest: Path, size: int) -> None:
    """Resumable: bytes go to <dest>.part, continued with a Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink(); have = 0
        if have == size:
            break
        hdr = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, headers=hdr, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 200:
                    have = 0                       # whole body (Range ignored): restart
                elif not (r.status_code == 206 and have):
                    raise RuntimeError(f"HTTP {r.status_code}")
                t0, last = time.time(), 0.0
                with open(part, "ab" if have else "wb") as f:
                    got = have
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk); got += len(chunk)
                        if time.time() - last > 15:
                            last = time.time()
                            print(f"    {dest.name}: {got / 1e6:,.0f}/{size / 1e6:,.0f} MB "
                                  f"({time.time() - t0:.0f}s)", flush=True)
        except (requests.RequestException, RuntimeError) as e:
            print(f"    {dest.name}: attempt {attempt} failed ({e}); resuming", flush=True)
            time.sleep(2 * attempt)
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: incomplete after {RETRIES} attempts")
    os.replace(part, dest)


def clip_tiles(sources: list[str], grid: Grid) -> tuple[np.ndarray, list[str]]:
    """Reproject the part of each 1/3" tile under `grid` onto it, bilinear.

    Only a window of each tile is read (the grid's footprint plus a pad for
    the kernel), so a 0.5 GB tile never sits in memory and --remote touches
    only the blocks it needs. Adjacent tiles overlap by 6 pixels (2"); there
    the later tile writes the same source values over the earlier one.
    """
    dst = np.full((grid.height, grid.width), NODATA, np.float32)
    used = []
    env = GDAL_REMOTE if any(s.startswith("/vsicurl/") for s in sources) else {}
    with rasterio.Env(**env):
        for s in sources:
            with rasterio.open(s) as src:
                if src.crs is None or src.crs.to_epsg() != 4269:
                    raise RuntimeError(f"{s}: expected EPSG:4269, got {src.crs}")
                win = window_for(src, grid.lonlat_bounds())
                if win is None:
                    continue
                arr = src.read(1, window=win)
                reproject(arr, dst, src_transform=src.window_transform(win),
                          src_crs=src.crs, src_nodata=src.nodata,
                          dst_transform=grid.transform, dst_crs=CRS_OUT,
                          dst_nodata=NODATA, resampling=Resampling.bilinear,
                          init_dest_nodata=False)
                used.append(s)
    return dst, used


# =========================================================================
# fetch: imageserver
# =========================================================================
def is_subgrids(grid: Grid) -> list[tuple[int, int, Grid]]:
    """Split a grid into <= IS_TILE_PX requests, each on the same lattice."""
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
        if (src.width, src.height) != (g.width, g.height) or \
                not src.transform.almost_equals(g.transform, precision=1e-6):
            raise RuntimeError(f"{path.name}: grid {src.transform} {src.width}x{src.height} "
                               f"!= requested {g.transform} {g.width}x{g.height}")
        a = src.read(1).astype(np.float32)
        if src.nodata is not None and src.nodata != NODATA:
            a[a == np.float32(src.nodata)] = NODATA
        return a


def is_catalog(sess: requests.Session, g: Grid) -> dict:
    """The primary 3DEP catalog items under a bbox that are visible at 10 m:
    the candidates the server mosaics from, best (finest) first."""
    r = sess.get(IS_URL + "/query", timeout=TIMEOUT, params=dict(
        geometry=",".join(f"{v:.0f}" for v in g.bounds), geometryType="esriGeometryEnvelope",
        inSR=5070, spatialRel="esriSpatialRelIntersects", where="Category = 1",
        outFields="Name,LowPS,MinPS,MaxPS", returnGeometry="false", f="json"))
    r.raise_for_status()
    d = r.json()
    rows = [f["attributes"] for f in d.get("features", [])]
    vis = [a for a in rows if (a.get("MinPS") or 0) <= RES_M <= (a.get("MaxPS") or 0)]
    vis.sort(key=lambda a: a.get("LowPS") or 0)
    return dict(items=[f"{a.get('Name')} (LowPS {a.get('LowPS') or 0:.2f} m)" for a in vis],
                truncated=bool(d.get("exceededTransferLimit")))


# =========================================================================
# fetch
# =========================================================================
def plan_jobs(a) -> tuple[list[str], list[Job]]:
    """Every component raster of every --bboxes file, with its state."""
    if a.prefix and len(a.bboxes) > 1:
        sys.exit("--prefix needs exactly one --bboxes file")
    prefixes = [a.prefix or prefix_of(p, "_bboxes.parquet") for p in a.bboxes]
    if len(set(prefixes)) < len(prefixes):
        sys.exit(f"a prefix appears twice in --bboxes: {prefixes}")
    jobs = []
    for k, (path, prefix) in enumerate(zip(a.bboxes, prefixes), 1):
        for c, g in grids_of(read_bboxes(path)).items():
            j = Job(f"p{k}", prefix, c, g, dem_path(a.gis_dir, prefix, c))
            if j.path.exists() and not a.overwrite:
                j.why = raster_mismatch(j.path, g, a.source)
                j.state = "stale" if j.why else "exists"
            jobs.append(j)
    return prefixes, jobs


def build_one(a, sess, g: Grid, by_tile: dict) -> tuple[np.ndarray, dict, dict, str]:
    """One grid's raster from the selected source, with its GeoTIFF tags, its
    manifest entry and a note for the log."""
    if a.source == "tnm":
        ts = [by_tile[t] for t in tiles_for(g.lonlat_bounds()) if t in by_tile]
        srcs = [("/vsicurl/" + t["url"]) if a.remote else str(t["local"]) for t in ts]
        arr, used = clip_tiles(srcs, g)
        inputs = [f"{t['title']} | {t['url']}" for t in ts]
        entry = dict(remote=bool(a.remote), inputs=inputs,
                     published={t["tile"]: t["published"] for t in ts})
        tags = dict(source=SOURCE_TAG["tnm"], inputs="; ".join(inputs))
        return arr, tags, entry, f"{len(used)} tile(s)"
    cache = a.raw_dir / "imageserver"
    subs = is_subgrids(g)
    arr = np.full((g.height, g.width), NODATA, np.float32)
    for r0, c0, sg in subs:
        p = cache / is_cache_name(sg)
        if not p.exists():
            is_export(sess, sg, p)
        arr[r0:r0 + sg.height, c0:c0 + sg.width] = is_read(p, sg)
    cat = is_catalog(sess, g)
    entry = dict(requests=len(subs), service=IS_URL, catalog_items_at_10m=cat["items"],
                 catalog_truncated=cat["truncated"])
    tags = dict(source=SOURCE_TAG["imageserver"], inputs="; ".join(cat["items"]))
    note = (f"{len(subs)} request(s); best catalog item "
            f"{cat['items'][0] if cat['items'] else '?'}")
    return arr, tags, entry, note


def plan_sizes(a, sess, jobs: list[Job], donor: dict, grids: dict[tuple, Grid],
               labels: list[str], mode: str, free: int) -> tuple[dict, int, int] | None:
    """List the tiles (tnm) and the sizes per prefix and for the call; return
    the tiles by name, the network bytes and the disk bytes of the selected
    mode, or None when a tile cannot be sized."""
    tnm = a.source == "tnm"
    # own[lab]: the grids prefix `lab` would have to fetch were it run alone
    own = {lab: {j.key for j in jobs if j.label == lab and j.state != "exists"
                 and donor[j.key].state != "exists"} for lab in labels}
    users = {k: [j.name for j in jobs if j.state != "exists" and j.key == k] for k in grids}

    def rasters(keys):
        return sum(grids[k].width * grids[k].height * 4 for k in keys)

    by_tile: dict[str, dict] = {}
    if tnm:
        tiles = tnm_plan(sess, grids, a.raw_dir)
        print(f"  {len(tiles)} tile(s):", flush=True)
        print(f"  {'tile':<8} {'published':<11} {'size':>10} {'to get':>10} {'remote <=':>10}  "
              f"{'cached':<6} {'=current':<8} used by", flush=True)
        for t in tiles:
            used_by = " ".join(n for k in t["grids"] for n in users[k])
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
        if any(t["url"] and t["size"] is None for t in tiles):
            print("  ERROR: HEAD failed for a tile; cannot size the download", file=sys.stderr)
            return None
        by_tile = {t["tile"]: t for t in tiles if t["url"]}
        missing = [t["tile"] for t in tiles if t["url"] is None]
        if missing:
            print(f"  WARNING: no tile for {missing}; that area stays nodata "
                  f"(sample will report any vertex it costs)", flush=True)
        late = [t for t in by_tile.values() if t["published"] > REPUBLISHED_AFTER]
        if late:
            print(f"  WARNING: {len(late)} tile(s) published after {REPUBLISHED_AFTER}: "
                  + ", ".join(f"{t['tile']} ({t['published']})" for t in late)
                  + ". pull_gee.py took elevation from Earth Engine's USGS/3DEP/10m_collection,"
                  " whose version of these tiles is not checked; on them the level-set test may"
                  " read a newer surface than the features saw (docstring note 1).", flush=True)

        def sizes(keys):
            whole = sum(t["to_get"] for t in by_tile.values() if keys & set(t["grids"]))
            remote = sum(b for t in by_tile.values() for k, b in t["remote"].items() if k in keys)
            return [whole, remote, rasters(keys)]
        cols = ["whole tiles", "--remote <=", "rasters <="]
    else:
        cache = a.raw_dir / "imageserver"

        def sizes(keys):
            req = {is_cache_name(sg): sg.width * sg.height * 4
                   for k in keys for _, _, sg in is_subgrids(grids[k])}
            return [sum(b for nm, b in req.items() if not (cache / nm).exists()), rasters(keys)]
        cols = ["exportImage <=", "rasters <="]

    total = sizes(set(grids))
    print("  size by prefix, each as if fetched alone; 'call' is this run, with shared tiles "
          "and identical grids counted once:", flush=True)
    def row(name, n, cells):
        print(f"    {name:<6} {n:>5}" + "".join(f" {v:>14}" for v in cells), flush=True)
    row("", "grids", cols)
    for lab in labels:
        row(lab, len(own[lab]), [gb(v) for v in sizes(own[lab])])
    row("call", len(grids), [gb(v) for v in total])

    net = total[1] if tnm and a.remote else total[0]
    out_bytes = total[-1]
    disk = out_bytes + (0 if tnm and a.remote else net)
    what = ("upper bound on COG blocks read" if tnm and a.remote
            else "whole tiles still to download" if tnm
            else f"upper bound, exportImage requests of <= {IS_TILE_PX} px")
    print(f"  selected ({mode}): network {gb(net)} ({what}); disk {gb(disk)} incl. rasters "
          f"<= {gb(out_bytes)}; free now {gb(free)}", flush=True)
    return by_tile, net, disk


def cmd_fetch(a) -> int:
    prefixes, jobs = plan_jobs(a)
    labels = [f"p{k}" for k in range(1, len(prefixes) + 1)]
    tnm = a.source == "tnm"
    mode = ("--remote COG block reads" if tnm and a.remote
            else "whole tiles" if tnm else "exportImage requests")
    print(f"fetch: {len(prefixes)} bboxes file(s), {len(jobs)} component raster(s), source "
          f"{a.source} ({mode}); rasters in {a.gis_dir / 'dem'}, cache {a.raw_dir}", flush=True)

    # donor[grid]: the job whose raster serves every job on that grid -- an
    # existing raster if the call has one, else the first job that builds it
    donor: dict[tuple, Job] = {}
    for j in jobs:
        if j.state == "exists":
            donor.setdefault(j.key, j)
    for j in jobs:
        donor.setdefault(j.key, j)
    build = [j for j in jobs if j.state != "exists" and donor[j.key] is j]
    link = [j for j in jobs if j.state != "exists" and donor[j.key] is not j]

    for lab, prefix in zip(labels, prefixes):
        mine = [j for j in jobs if j.label == lab]
        n = {s: sum(j.state == s for j in mine) for s in ("exists", "stale", "todo")}
        print(f"[{lab}] {prefix}: {len(mine)} bbox(es); {n['exists']} raster(s) exist, "
              f"{n['stale']} stale, {n['todo']} to make", flush=True)
        for j in mine:
            g, d = j.grid, donor[j.key]
            w, s, e, nn = g.lonlat_bounds()
            what = ("exists, kept" if j.state == "exists" else "build" if d is j else
                    f"same grid as {d.name}: hardlink "
                    f"{'its existing raster' if d.state == 'exists' else 'once built'}")
            if j.state == "stale":
                what = f"STALE ({j.why}); {what}"
            print(f"  c{j.comp}: {g.width} x {g.height} px ({g.width * RES_M / 1e3:.1f} x "
                  f"{g.height * RES_M / 1e3:.1f} km), lon {w:.4f}..{e:.4f}, "
                  f"lat {s:.4f}..{nn:.4f}: {what}", flush=True)
    if not build and not link:
        print("nothing to do (use --overwrite to rebuild)", flush=True)
        return 0

    grids = {j.key: j.grid for j in build}
    sess = session()
    free = min(free_bytes(a.raw_dir), free_bytes(a.gis_dir))
    by_tile, net, disk = {}, 0, 0
    if grids:
        sized = plan_sizes(a, sess, jobs, donor, grids, labels, mode, free)
        if sized is None:
            return 1
        by_tile, net, disk = sized
    else:
        print(f"  nothing to fetch: the {len(link)} raster(s) to make are hardlinks of "
              f"existing ones", flush=True)

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
        print(f"REFUSED: {gb(net)} exceeds --max-gb {a.max_gb}. Raise --max-gb deliberately"
              f"{', or use --remote' if tnm and not a.remote else ''}.", file=sys.stderr)
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
    for j in link:
        if j.key in failed:
            continue
        d = donor[j.key]
        e = (entries.get(j.key)
             or read_manifest(manifest_path(a.gis_dir, d.prefix)).get("components", {})
                .get(str(d.comp))
             or dict(bounds=d.grid.bounds, width=d.grid.width, height=d.grid.height))
        e = {k: v for k, v in e.items() if k not in ("path", "linked_from", "link", "source")}
        kind = link_raster(d.path, j.path)
        update_manifest(manifest_path(a.gis_dir, j.prefix), j.prefix, a.source, j.comp,
                        dict(e, path=str(j.path), linked_from=str(d.path), link=kind))
        print(f"  {j.name}: {j.path.name} = {d.name} {d.path.name} ({kind})", flush=True)

    bad = [j.name for j in build + link if j.key in failed]
    print(f"fetch done: {sum(j.key not in failed for j in build)} raster(s) built, "
          f"{sum(j.key not in failed for j in link)} linked, in {a.gis_dir / 'dem'}; "
          f"manifests <prefix>_dem.json ({time.time() - t0:.0f}s)", flush=True)
    if bad:
        print(f"FAILED: no elevation at all for {bad}", file=sys.stderr)
        return 1
    return 0


# =========================================================================
# sample
# =========================================================================
def bilinear(arr: np.ndarray, t: Affine, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear value at (x, y) from the four surrounding cell centres.

    `arr` holds NaN for nodata; a position with any NaN neighbour, or without
    four neighbours inside the array, gets NaN.
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
    ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < w) & (r0 + 1 < h)
    i = np.flatnonzero(ok)
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


def check_rasters(bb: pd.DataFrame, gis_dir: Path, prefix: str) -> tuple[dict, list[str]]:
    """Every bbox must have an EPSG:5070 raster that contains it."""
    paths, errors = {}, []
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
            tol = 1e-6
            if not (b.left <= r.minx + tol and b.bottom <= r.miny + tol
                    and b.right >= r.maxx - tol and b.top >= r.maxy - tol):
                errors.append(f"c{r.component}: {p.name} bounds {tuple(round(v) for v in b)} "
                              f"do not contain bbox {(round(r.minx), round(r.miny), round(r.maxx), round(r.maxy))}")
                continue
        paths[int(r.component)] = p
    return paths, errors


def load_lobes(lpath: Path, vt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lobe polygons, prepared: each vertex's own lobe (aligned with vt's
    rows), every lobe, and every lobe's component. Exits when the lobes file
    does not belong to the vertex table."""
    import geopandas as gpd
    import shapely
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
    if (ix < 0).any():
        eg = pd.unique(vt["lobe_id"].to_numpy()[ix < 0])[:3].tolist()
        sys.exit(f"FAILED: {(ix < 0).sum():,} of {len(vt):,} vertices have a lobe_id that is not "
                 f"in {lpath} (e.g. {eg}); the vertices and lobes are not from one lobes.py run")
    lobe_comp = lobes["component"].astype(int).to_numpy()
    off = lobe_comp[ix] != vt["component"].to_numpy()
    if off.any():
        sys.exit(f"FAILED: {off.sum():,} vertices sit in a lobe of another component in {lpath}")
    geoms = lobes.geometry.to_numpy()
    shapely.prepare(geoms)
    return geoms[ix], geoms, lobe_comp


def check_normals(vt: pd.DataFrame, own: np.ndarray) -> pd.DataFrame:
    """Per component, the share of up to NORMAL_SAMPLE seeded vertices whose
    point NORMAL_EPS_M along (nx, ny) is inside their own lobe: ~1 for inward
    normals, ~0 for a flipped orientation."""
    import shapely
    rng = np.random.default_rng(0)
    comp = vt["component"].to_numpy()
    x, y, nx, ny = (vt[c].to_numpy(float) for c in ("x", "y", "nx", "ny"))
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
    """Where the z_in / z_out probes land relative to the lobes (docstring
    note 6).

    For out_probe_in_any_lobe the probes go into an STRtree that each of the
    component's lobes queries with "contains", so GEOS tests each candidate
    against a prepared polygon; points querying a tree of lobes with "within"
    test each candidate against an unprepared ring (40x slower on v1)."""
    import shapely
    flags = dict(in_probe_inside=shapely.contains_xy(own, xin, yin),
                 out_probe_outside=~shapely.contains_xy(own, xout, yout),
                 out_probe_in_any_lobe=np.zeros(len(own), bool))
    for c in np.unique(comp):
        i = np.flatnonzero(comp == c)
        tree = shapely.STRtree(shapely.points(xout[i], yout[i]))
        hit = tree.query(geoms[lobe_comp == c], predicate="contains")[1]
        flags["out_probe_in_any_lobe"][i[hit]] = True
    return flags


def cmd_sample(a) -> int:
    prefix = a.prefix or prefix_of(a.vertices, "_vertices.parquet")
    out = a.out or RESULTS / dt.date.today().isoformat() / "boundary" / f"{prefix}_vertices_z.parquet"
    if out.exists() and not a.overwrite:
        print(f"[{prefix}] {out} exists -- skipping (use --overwrite to redo)", flush=True)
        return 0
    bpath = a.bboxes or a.vertices.with_name(f"{prefix}_bboxes.parquet")
    lpath = a.lobes or a.vertices.with_name(f"{prefix}_lobes.parquet")
    if a.lobes and not a.lobes.exists():
        sys.exit(f"FAILED: --lobes {a.lobes} does not exist")
    t0 = time.time()

    vt = pd.read_parquet(a.vertices)
    miss = [c for c in VERTEX_COLS if c not in vt.columns]
    if miss:
        sys.exit(f"{a.vertices}: missing columns {miss}")
    vt["component"] = vt["component"].astype(int)
    bad = ~np.isfinite(vt[["x", "y", "nx", "ny"]].to_numpy(float)).all(axis=1)
    if bad.any():
        sys.exit(f"{a.vertices}: {bad.sum():,} vertices with non-finite x, y, nx or ny")
    norm = np.hypot(vt["nx"].to_numpy(float), vt["ny"].to_numpy(float))
    if not np.all(np.abs(norm - 1) < 1e-3):
        sys.exit(f"{a.vertices}: {(np.abs(norm - 1) >= 1e-3).sum():,} normals are not unit vectors")
    bb = read_bboxes(bpath)
    print(f"[{prefix}] {len(vt):,} vertices in {vt.component.nunique()} component(s); "
          f"{len(bb)} bbox(es) from {bpath}; offset {a.offset_m:g} m", flush=True)

    errors = [f"c{c}: vertices but no bbox" for c in sorted(set(vt.component) - set(bb.component))]
    paths, errs = check_rasters(bb, a.gis_dir, prefix)
    errors += errs
    if errors:
        print("FAILED: bboxes not covered by their rasters:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    own = None
    if lpath.exists():
        own, geoms, lobe_comp = load_lobes(lpath, vt)
        nc = check_normals(vt, own)
        print(f"  normals ({lpath.name}): share of up to {NORMAL_SAMPLE:,} vertices per component "
              f"whose point {NORMAL_EPS_M} m along (nx, ny) is inside their lobe; each must "
              f"reach {NORMAL_MIN_SHARE}", flush=True)
        for r in nc.itertuples():
            print(f"    c{r.component}: {r.share:.4f} of {r.checked:,}"
                  f"{'' if r.share >= NORMAL_MIN_SHARE else '  FAIL'}", flush=True)
        flipped = nc.component[nc.share < NORMAL_MIN_SHARE].tolist()
        if flipped:
            print(f"FAILED: (nx, ny) do not point into the lobes of components {flipped}; "
                  f"z_in/z_out would be swapped", file=sys.stderr)
            return 1
    else:
        print(f"  normals and probe flags: not computed ({lpath.name} not found)", flush=True)

    man = manifest_path(a.gis_dir, prefix)
    src = read_manifest(man).get("components", {})
    used = sorted({str(src.get(str(c), {}).get("source")) for c in paths})
    print(f"  DEM source(s): {', '.join(used)} ({man.name if man.exists() else 'no manifest'})",
          flush=True)
    if len(used) > 1:
        print("  WARNING: components come from different DEM sources; the two differ by "
              "decimetres (see the docstring). Re-fetch with one --source and --overwrite.",
              flush=True)
    late = sorted({f"{t} ({p})" for c in paths
                   for t, p in src.get(str(c), {}).get("published", {}).items()
                   if p > REPUBLISHED_AFTER})
    if late:
        print(f"  WARNING: tiles published after {REPUBLISHED_AFTER}: {', '.join(late)}; "
              f"which version pull_gee.py's Earth Engine elevation used is not checked "
              f"(docstring note 1)", flush=True)

    x, y = vt["x"].to_numpy(float), vt["y"].to_numpy(float)
    nx, ny = vt["nx"].to_numpy(float), vt["ny"].to_numpy(float)
    d = a.offset_m
    xin, yin, xout, yout = x + d * nx, y + d * ny, x - d * nx, y - d * ny
    z = np.full(len(vt), np.nan); z_in = z.copy(); z_out = z.copy()
    comp = vt["component"].to_numpy()
    for c in sorted(set(comp)):
        i = np.flatnonzero(comp == c)
        arr, t = read_window(paths[c], np.r_[x[i], xin[i], xout[i]], np.r_[y[i], yin[i], yout[i]])
        z[i] = bilinear(arr, t, x[i], y[i])
        z_in[i] = bilinear(arr, t, xin[i], yin[i])
        z_out[i] = bilinear(arr, t, xout[i], yout[i])

    res = vt.copy()
    res["z"], res["z_in"], res["z_out"] = z, z_in, z_out
    res["dem_ok"] = np.isfinite(z) & np.isfinite(z_in) & np.isfinite(z_out)
    if own is not None:
        res = res.assign(**probe_flags(own, geoms, lobe_comp, comp, xin, yin, xout, yout))
        print(f"  probes at {d:g} m, share of vertices: z_in probe inside own lobe | z_out probe "
              f"outside own lobe | z_out probe inside any lobe of the component", flush=True)
        print(f"  {'comp':>4} {'role':<10} {'vertices':>9} {'in_inside':>10} {'out_outside':>12} "
              f"{'out_any':>8}", flush=True)
        for (c, role), g in res.groupby(["component", "role"]):
            print(f"  {c:>4} {role:<10} {len(g):>9,} {g.in_probe_inside.mean():>10.3f} "
                  f"{g.out_probe_outside.mean():>12.3f} {g.out_probe_in_any_lobe.mean():>8.3f}",
                  flush=True)

    print(f"  {'comp':>4} {'vertices':>9} {'ok':>9} {'miss z':>7} {'z_in':>6} {'z_out':>6}  "
          f"{'z min':>7} {'median':>7} {'max':>7}  {'dz med':>7}", flush=True)
    for c, g in res.groupby("component"):
        ok = g[g.dem_ok]
        dz = (ok.z_out - ok.z_in)
        zr = (f"{ok.z.min():7.2f} {ok.z.median():7.2f} {ok.z.max():7.2f}  {dz.median():7.2f}"
              if len(ok) else f"{'-':>7} {'-':>7} {'-':>7}  {'-':>7}")
        print(f"  {c:>4} {len(g):>9,} {len(ok):>9,} {g.z.isna().sum():>7,} "
              f"{g.z_in.isna().sum():>6,} {g.z_out.isna().sum():>6,}  {zr}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    n_bad = int((~res.dem_ok).sum())
    if n_bad:
        mpath = out.with_name(out.stem + "_misses.csv")
        res.loc[~res.dem_ok, ["lobe_id", "zone", "role", "component", "ring", "s", "x", "y",
                              "lon", "lat", "z", "z_in", "z_out"]].to_csv(mpath, index=False)
        print(f"  {n_bad:,} vertices lack z, z_in or z_out; listed in {mpath}", flush=True)
        if not a.allow_missing:
            print(f"FAILED: {n_bad:,} of {len(res):,} vertices not fully covered "
                  f"(--allow-missing writes the table with dem_ok=False rows)", file=sys.stderr)
            return 1

    meta = dict(offset_m=d, interpolation="bilinear", crs=CRS_OUT, res_m=RES_M,
                dem={str(c): dict(path=str(p), source=src.get(str(c), {}).get("source"))
                     for c, p in paths.items()},
                lobes=str(lpath) if own is not None else None,
                probe_flags=PROBE_COLS if own is not None else [],
                normals=nc.to_dict("records") if own is not None else [],
                created=dt.datetime.now().isoformat(timespec="seconds"))
    tbl = pa.Table.from_pandas(res, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}),
                                       b"pull_dem": json.dumps(meta).encode()})
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    pq.write_table(tbl, tmp)
    os.replace(tmp, out)
    print(f"[{prefix}] wrote {out}: {len(res):,} vertices, {int(res.dem_ok.sum()):,} with "
          f"z, z_in and z_out ({time.time() - t0:.1f}s)", flush=True)
    return 0


# =========================================================================
# self-test (no network)
# =========================================================================
def self_test() -> int:
    ok = fail = 0
    def chk(cond, msg):
        nonlocal ok, fail
        if cond: ok += 1
        else: fail += 1; print(f"  FAIL {msg}")

    g = Grid.around(1315671.2, 857889.9, 1317669.0, 859890.1)
    chk(g.bounds == (1315670.0, 857880.0, 1317670.0, 859900.0), f"grid snapping {g.bounds}")
    chk((g.width, g.height) == (200, 202), f"grid shape {g.width}x{g.height}")
    chk(tiles_for((-82.31, 29.94, -82.29, 29.96)) == ["n30w083"], "one tile")
    chk(tiles_for((-82.6, 29.7, -82.0, 30.2)) == ["n30w083", "n31w083"], "two tiles")
    chk(tiles_for((-80.5, 25.0, -79.5, 25.5)) == ["n26w080", "n26w081"], "lon seam")
    subs = is_subgrids(Grid(0, 0, 90000, 50000))
    chk(len(subs) == 6 and sum(s.width * s.height for _, _, s in subs) == 9000 * 5000,
        "imageserver split covers the grid once")

    # bilinear is exact on a plane, NaN-strict, and honours cell centres
    t = Affine(10, 0, 1000, 0, -10, 2000)
    rr, cc = np.mgrid[0:50, 0:60]
    xc, yc = 1000 + (cc + 0.5) * 10, 2000 - (rr + 0.5) * 10
    plane = 0.01 * xc - 0.02 * yc + 5
    rng = np.random.default_rng(0)
    px = rng.uniform(1010, 1580, 500); py = rng.uniform(1520, 1990, 500)
    zb = bilinear(plane, t, px, py)
    chk(np.allclose(zb, 0.01 * px - 0.02 * py + 5, atol=1e-9), "bilinear exact on a plane")
    chk(np.isclose(bilinear(plane, t, np.r_[xc[3, 4]], np.r_[yc[3, 4]])[0], plane[3, 4]),
        "cell centre returns the cell value")
    holed = plane.copy(); holed[10, 10] = np.nan
    zz = bilinear(holed, t, np.r_[xc[10, 10] + 3, xc[30, 30]], np.r_[yc[10, 10] - 3, yc[30, 30]])
    chk(np.isnan(zz[0]) and np.isfinite(zz[1]), "NaN neighbour -> NaN, others unaffected")
    chk(np.isnan(bilinear(plane, t, np.r_[900.0], np.r_[1900.0])[0]), "outside -> NaN")

    # probe flags: a 60 m strip A on a 500 m block B, and a C-shaped lobe whose
    # 80 m gap a z_out probe crosses back into the lobe
    import shapely
    A, B = shapely.box(0, 0, 1000, 60), shapely.box(0, -500, 1000, 0)
    C = shapely.difference(shapely.box(2000, 0, 2300, 400), shapely.box(2050, 150, 2300, 230))
    geoms = np.array([A, B, C], dtype=object)
    vx, vy = np.r_[500.0, 500, 500, 2200], np.r_[60.0, 0, -500, 150]
    vny = np.r_[-1.0, 1, 1, -1]
    f = probe_flags(geoms[[0, 0, 1, 2]], geoms, np.zeros(3, int), np.zeros(4, int),
                    vx, vy + 100 * vny, vx, vy - 100 * vny)
    chk(f["in_probe_inside"].tolist() == [False, False, True, True], f"in_probe_inside {f}")
    chk(f["out_probe_outside"].tolist() == [True, True, True, False], f"out_probe_outside {f}")
    chk(f["out_probe_in_any_lobe"].tolist() == [False, True, False, True],
        f"out_probe_in_any_lobe {f}")
    f = probe_flags(geoms[[0, 0, 1, 2]], geoms, np.r_[1, 1, 1], np.zeros(4, int),
                    vx, vy + 100 * vny, vx, vy - 100 * vny)
    chk(not f["out_probe_in_any_lobe"].any(), "out_probe_in_any_lobe ignores other components")

    # the normal check is per component: one flipped component fails on its own
    vt = pd.DataFrame(dict(component=[0] * 50 + [1] * 50, x=np.tile(np.linspace(10, 990, 50), 2),
                           y=-500.0, nx=0.0, ny=np.r_[np.ones(50), -np.ones(50)]))
    nc = check_normals(vt, np.array([B] * 100, dtype=object))
    chk(nc.share.round(3).tolist() == [1.0, 0.0], f"per-component normal check {nc.share.tolist()}")
    print(f"pull_dem.py self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="DEM raster per bbox component")
    f.add_argument("--bboxes", type=Path, nargs="+", required=True,
                   help="one or more <prefix>_bboxes.parquet; the size gate applies to their "
                        "total, and shared tiles and identical grids are fetched once")
    f.add_argument("--source", choices=["tnm", "imageserver"], default="tnm")
    f.add_argument("--remote", action="store_true",
                   help="tnm: read only the needed COG blocks over HTTPS, no tile download")
    f.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    f.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="tile / request cache")
    f.add_argument("--prefix", help="default: from the bboxes file name (one --bboxes only)")
    f.add_argument("--max-gb", type=float, default=DEFAULTS["max_gb"],
                   help="refuse when the call's network total for the selected mode exceeds it")
    f.add_argument("--min-free-gb", type=float, default=DEFAULTS["min_free_gb"],
                   help="refuse when the call would leave less free disk")
    f.add_argument("--yes", action="store_true", help="download; without it, list only")
    f.add_argument("--overwrite", action="store_true",
                   help="rebuild existing rasters, stale ones included (the --raw-dir cache "
                        "is still reused; delete it to re-download)")

    s = sub.add_parser("sample", help="z, z_in, z_out at every vertex")
    s.add_argument("--vertices", type=Path, required=True, help="<prefix>_vertices.parquet")
    s.add_argument("--bboxes", type=Path, help="default: <prefix>_bboxes.parquet beside --vertices")
    s.add_argument("--lobes", type=Path,
                   help="default: <prefix>_lobes.parquet beside --vertices; when present, "
                        "normals are checked per component and the probe flags are added")
    s.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    s.add_argument("--prefix", help="default: from the vertices file name")
    s.add_argument("--offset-m", type=float, default=DEFAULTS["offset_m"])
    s.add_argument("--out", type=Path,
                   help="default: results/<today>/boundary/<prefix>_vertices_z.parquet")
    s.add_argument("--allow-missing", action="store_true",
                   help="write the table even if some vertices lack z, z_in or z_out")
    s.add_argument("--overwrite", action="store_true")

    sub.add_parser("selftest", help="offline checks of the grid, tiles and bilinear code")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        return cmd_fetch(a)
    if a.cmd == "sample":
        return cmd_sample(a)
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
