#!/usr/bin/env python3
"""Download NLCD 2021 land cover for Florida as one local raster, for sample_points.py --land-mask.

The image is the one pull_gee.nlcd_group() samples (USGS/NLCD_RELEASES/2021_REL/NLCD, the 2021
release, band landcover), fetched on the asset's own CRS and 30 m pixel grid without resampling,
so the sampler tests the same pixel pull_gee.py later reads for each point. The extent is the
bounding box of the Florida counties, buffered by BUFFER_M and snapped outward to that grid: the
counties include coastal water, and band rings reach out over it. Masked pixels (offshore,
outside NLCD) arrive as 0.

Input   data/interim/gis/counties_FL.parquet
Output  data/interim/gis/nlcd2021_FL.tif      uint8 class codes, 0 = no data; tiled DEFLATE GeoTIFF
        data/interim/gis/nlcd2021_FL.json     CRS, transform, size, SHA-256, pixels per class
        data/interim/gis/nlcd2021_FL_tiles/   one file per tile; a rerun skips those on disk

    python pull_land_mask.py --project $EE_PROJECT
    python pull_land_mask.py --check --project $EE_PROJECT    # 2,000 points against pull_gee
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS as RasterCRS
from rasterio.transform import Affine
from rasterio.windows import Window

import pull_gee
from sample_points import MASK_NODATA, MASK_OPEN_WATER, LandMask, file_sha256

COUNTIES = Path("data/interim/gis/counties_FL.parquet")
OUT = Path("data/interim/gis/nlcd2021_FL.tif")
POINTS = Path("data/interim/points/points_FL.parquet")

SOURCE = "USGS/NLCD_RELEASES/2021_REL/NLCD"     # as pull_gee.nlcd_group()
BUFFER_M = 6000.0   # the 5 km band, the 100 m ring simplification and a margin
TILE_PX = 4096      # tile side: 16 MB of uint8, under computePixels' 48 MB per request
WORKERS = 4         # concurrent requests
BLOCK_PX = 512      # GeoTIFF block side; divides TILE_PX, so tiles write whole blocks

CHECK_POINTS = 2000
CHECK_MIN_AGREE = 0.999


@dataclass(frozen=True)
class Grid:
    """The mosaic: a window of the image's native pixel grid. Rows run south."""
    crs_wkt: str
    x0: float       # west edge of column 0, in the CRS's metres
    y0: float       # north edge of row 0
    res: float      # pixel side, metres
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y0)


def native_projection(projection_info: dict) -> tuple[str, float, float, float]:
    """(WKT, origin x, origin y, pixel side) of an image.projection().getInfo() result.

    Only a north-up grid of square pixels is accepted, which NLCD's is.
    """
    a, b, c, d, e, f = projection_info["transform"]
    if b != 0 or d != 0 or a <= 0 or e != -a:
        sys.exit(f"unexpected native transform {projection_info['transform']}: "
                 "want north-up square pixels")
    return projection_info["wkt"], float(c), float(f), float(a)


def window_on_grid(bounds, origin_x: float, origin_y: float, res: float,
                   buffer_m: float) -> tuple[int, int, int, int]:
    """(first column, first row, width, height) of the grid pixels covering bounds + buffer_m.

    Snapped outward: every pixel that the buffered box touches is in the window.
    """
    minx, miny, maxx, maxy = bounds
    col0 = math.floor((minx - buffer_m - origin_x) / res)
    col1 = math.ceil((maxx + buffer_m - origin_x) / res)
    row0 = math.floor((origin_y - (maxy + buffer_m)) / res)
    row1 = math.ceil((origin_y - (miny - buffer_m)) / res)
    return col0, row0, col1 - col0, row1 - row0


def florida_grid(projection_info: dict, counties: gpd.GeoDataFrame, buffer_m: float) -> Grid:
    """The mosaic grid: the counties' bounding box in the image's CRS, buffered and snapped."""
    wkt, origin_x, origin_y, res = native_projection(projection_info)
    bounds = counties.to_crs(wkt).total_bounds
    col0, row0, width, height = window_on_grid(bounds, origin_x, origin_y, res, buffer_m)
    return Grid(crs_wkt=wkt, x0=origin_x + col0 * res, y0=origin_y - row0 * res, res=res,
                width=width, height=height)


def tiles(grid: Grid, tile_px: int) -> list[tuple[str, int, int, int, int]]:
    """(name, column offset, row offset, width, height) of each tile, row by row."""
    out = []
    for row_off in range(0, grid.height, tile_px):
        for col_off in range(0, grid.width, tile_px):
            name = f"r{row_off // tile_px:03d}_c{col_off // tile_px:03d}.npz"
            out.append((name, col_off, row_off, min(tile_px, grid.width - col_off),
                        min(tile_px, grid.height - row_off)))
    return out


def tile_request_grid(grid: Grid, col_off: int, row_off: int, width: int, height: int) -> dict:
    """The computePixels pixel grid of one tile: the native grid, shifted to the tile."""
    return {"dimensions": {"width": width, "height": height},
            "affineTransform": {"scaleX": grid.res, "shearX": 0,
                                "translateX": grid.x0 + col_off * grid.res,
                                "shearY": 0, "scaleY": -grid.res,
                                "translateY": grid.y0 - row_off * grid.res},
            "crsWkt": grid.crs_wkt}


def save_tile(arr: np.ndarray, path: Path) -> None:
    """Write by rename, so a tile on disk is always complete."""
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, landcover=arr)
    os.replace(tmp, path)


def load_tile(path: Path) -> np.ndarray:
    with np.load(path) as stored:
        return stored["landcover"]


def download(fetch, grid: Grid, tiles_dir: Path, tile_px: int = TILE_PX,
             workers: int = WORKERS) -> int:
    """Fetch every tile not on disk yet; return how many failed.

    fetch(col_off, row_off, width, height) returns that tile as a (height, width) uint8 array.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    layout = tiles(grid, tile_px)
    todo = [t for t in layout if not (tiles_dir / t[0]).exists()]
    print(f"{len(layout)} tiles of up to {tile_px} x {tile_px} pixels "
          f"({len(layout) - len(todo)} on disk, {len(todo)} to fetch)", flush=True)
    if not todo:
        return 0

    def one(tile):
        name, col_off, row_off, width, height = tile
        arr = np.asarray(fetch(col_off, row_off, width, height))
        if arr.shape != (height, width):
            raise RuntimeError(f"{name}: got shape {arr.shape}, want {(height, width)}")
        save_tile(arr.astype(np.uint8), tiles_dir / name)

    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, tile): tile[0] for tile in todo}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as err:
                failed.append(f"{futures[future]}: {str(err)[:200]}")
            if n % 5 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} tiles, {len(failed)} failed, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    for line in failed[:3]:
        print(f"  FAILED {line}")
    if failed:
        print("  rerun the same command: tiles on disk are skipped")
    return len(failed)


def mosaic(grid: Grid, tiles_dir: Path, out: Path, tile_px: int = TILE_PX) -> np.ndarray:
    """Write the tiles into one GeoTIFF on `grid`; return the pixel count of each class 0-255."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.tif")
    profile = dict(driver="GTiff", dtype="uint8", count=1, width=grid.width,
                   height=grid.height, crs=RasterCRS.from_wkt(grid.crs_wkt),
                   transform=grid.transform, nodata=MASK_NODATA, tiled=True,
                   blockxsize=BLOCK_PX, blockysize=BLOCK_PX, compress="deflate",
                   BIGTIFF="IF_SAFER")
    counts = np.zeros(256, dtype=np.int64)
    with rasterio.open(tmp, "w", **profile) as dst:
        for name, col_off, row_off, width, height in tiles(grid, tile_px):
            arr = load_tile(tiles_dir / name)
            if arr.shape != (height, width):
                raise RuntimeError(f"{name}: shape {arr.shape}, want {(height, width)}")
            dst.write(arr, 1, window=Window(col_off, row_off, width, height))
            counts += np.bincount(arr.ravel(), minlength=256)
    os.replace(tmp, out)
    return counts


def sidecar(grid: Grid, out: Path, counts: np.ndarray, counties: Path, buffer_m: float,
            tile_px: int) -> dict:
    """What the raster is, for the record and for --land-mask users."""
    return {
        "source": SOURCE, "system:index": "2021", "band": "landcover",
        "unmask": MASK_NODATA,
        "crs_wkt": grid.crs_wkt,
        "transform": [grid.res, 0.0, grid.x0, 0.0, -grid.res, grid.y0],
        "width": grid.width, "height": grid.height,
        "extent": {"bounding_box_of": str(counties), "buffer_m": buffer_m,
                   "snapped": "outward to the native grid"},
        "tile_px": tile_px,
        "sha256": file_sha256(out),
        "pixels_by_class": {str(k): int(v) for k, v in enumerate(counts) if v},
        "written": dt.datetime.now().isoformat(timespec="seconds"),
    }


def print_counts(counts: np.ndarray) -> None:
    total = int(counts.sum())
    print(f"\n{total:,} pixels by class:")
    for code in np.nonzero(counts)[0]:
        print(f"  {code:>3}  {int(counts[code]):>13,}  {100 * counts[code] / total:6.2f}%")
    print(f"open water (class {MASK_OPEN_WATER}): {100 * counts[MASK_OPEN_WATER] / total:.2f}% "
          f"of pixels; no data (0): {100 * counts[MASK_NODATA] / total:.2f}%")


def check(mask_path: Path, points_path: Path, sample_ee, n: int = CHECK_POINTS,
          seed: int = 0) -> int:
    """Compare the raster with Earth Engine at n random points; 1 if they agree on too few.

    The raster is read through sample_points.LandMask, the sampler's own lookup, and Earth
    Engine through pull_gee's native-grid request, sample_ee(ids, lon, lat) -> a frame with
    point_id and landcover. Earth Engine's null (a masked pixel) counts as 0.
    """
    points = gpd.read_parquet(points_path, columns=["point_id", "lon", "lat", "geometry"])
    if points.crs is None or points.crs.to_epsg() != 5070:
        points = points.to_crs(5070)
    pick = points.sample(n=min(n, len(points)), random_state=seed).reset_index(drop=True)
    land = LandMask(mask_path)
    print(f"mask {land.describe()}")
    local = land.classes_at(pick.geometry.x.values, pick.geometry.y.values).astype(int)
    remote = sample_ee(pick.point_id.tolist(), pick.lon.to_numpy(), pick.lat.to_numpy())
    remote = remote.drop_duplicates("point_id").set_index("point_id")["landcover"]
    remote = pd.to_numeric(remote.reindex(pick.point_id), errors="coerce")
    remote = remote.fillna(MASK_NODATA).astype(int).to_numpy()

    agree = local == remote
    share = agree.mean()
    print(f"{int(agree.sum()):,} of {len(pick):,} points agree ({100 * share:.2f}%; "
          f"want at least {100 * CHECK_MIN_AGREE:.1f}%)")
    print(f"open water at these points: raster {int((local == MASK_OPEN_WATER).sum()):,}, "
          f"Earth Engine {int((remote == MASK_OPEN_WATER).sum()):,}")
    if not agree.all():
        rows = pick.loc[~agree, ["point_id", "lon", "lat"]].assign(
            raster=local[~agree], earth_engine=remote[~agree])
        print(rows.head(20).to_string(index=False))
    if share < CHECK_MIN_AGREE:
        print("STOP: the raster and Earth Engine disagree; the grids do not match")
        return 1
    print("grids match")
    return 0


def ee_fetcher(image, grid: Grid):
    """fetch() for download(): one computePixels request per tile, retried like pull_gee."""
    def fetch(col_off, row_off, width, height):
        request = {"expression": image, "fileFormat": "NUMPY_NDARRAY",
                   "grid": tile_request_grid(grid, col_off, row_off, width, height)}
        label = f"tile at column {col_off}, row {row_off}"
        values = pull_gee.with_retries(pull_gee.ee.data.computePixels, request, label=label)
        return values["landcover"]
    return fetch


def ee_sampler():
    """sample_ee() for check(): pull_gee's own NLCD request."""
    group = pull_gee.nlcd_group()

    def sample(ids, lon, lat):
        return pull_gee.with_retries(pull_gee.sample_group, group, ids, lon, lat,
                                     label="land mask check")
    return sample


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Download NLCD 2021 land cover for Florida as one "
                                             "raster on its native 30 m grid.")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                    help="Earth Engine Cloud project (default: $EE_PROJECT)")
    ap.add_argument("--counties", type=Path, default=COUNTIES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--buffer-m", type=float, default=BUFFER_M)
    ap.add_argument("--tile-px", type=int, default=TILE_PX)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--check", action="store_true",
                    help="compare the raster with Earth Engine at --n-check random --points")
    ap.add_argument("--points", type=Path, default=POINTS)
    ap.add_argument("--n-check", type=int, default=CHECK_POINTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.tile_px % BLOCK_PX:
        sys.exit(f"--tile-px must be a multiple of {BLOCK_PX}")

    try:
        if args.project:
            pull_gee.ee.Initialize(project=args.project)
        else:
            pull_gee.ee.Initialize()
    except Exception as err:
        print(f"Initialize failed: {str(err)[:300]}", file=sys.stderr)
        print("Run `earthengine authenticate` and check --project / EE_PROJECT.", file=sys.stderr)
        return 1

    if args.check:
        return check(args.out, args.points, ee_sampler(), args.n_check, args.seed)

    group = pull_gee.nlcd_group()
    image = group["image"].select("landcover").unmask(MASK_NODATA)
    grid = florida_grid(group["proj"].getInfo(), gpd.read_parquet(args.counties), args.buffer_m)
    print(f"grid: {grid.width:,} x {grid.height:,} pixels of {grid.res:g} m, "
          f"west {grid.x0:,.0f} north {grid.y0:,.0f} (native CRS)")

    tiles_dir = args.out.with_name(args.out.stem + "_tiles")
    if download(ee_fetcher(image, grid), grid, tiles_dir, args.tile_px, args.workers):
        return 1
    counts = mosaic(grid, tiles_dir, args.out, args.tile_px)
    meta = sidecar(grid, args.out, counts, args.counties, args.buffer_m, args.tile_px)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"-> {args.out} ({args.out.stat().st_size / 1e6:.0f} MB, sha256 {meta['sha256'][:12]}...)")
    print_counts(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
