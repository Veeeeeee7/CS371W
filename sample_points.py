#!/usr/bin/env python3
"""Sample a point-level flood-label dataset from FEMA NFHL flood-zone polygons.

Domain A draws its positives inside Zone A polygons (approximate study), domain AE inside Zone AE
polygons (detailed study). Both domains are built by one protocol, so they differ only in the
polygons behind them. Negatives are `band` (500-5,000 m from every SFHA, inside a FIRM panel) or
`shadedX` (inside Zone X 0.2%-chance polygons). Columns named `meta_*` record how the map was made
and are never features; every downstream split is made on `block_id`.

Each group (zone and label, docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D1) draws exactly
--candidates candidates that pass every requirement of its group: positives uniform by area over
the zone's polygons (no polygon cap, D2), band negatives inside the band and its masks. The
candidates are thinned to --min-sep-m within each group, then once over the whole state, and every
survivor is kept: the groups end a few hundred points apart, not at a round number.

With --land-mask (off by default), no point lies on open water: positives and shadedX negatives
are drawn over the land inside their polygons, band negatives over the land in the band and
outside NFHL OPEN WATER polygons. The mask is NLCD land cover from pull_land_mask.py.

Inputs:  data/interim/nfhl_states/{haz,pan,s_bfe}_FL.parquet
         data/interim/gis/nlcd2021_FL.tif (--land-mask only)
Outputs: data/interim/prep/{idx_FL,haz_FL_5070}.parquet (--prep, run once)
         data/interim/points/points_FL.parquet (the settings and the candidate and survivor counts
         per group in its parquet metadata: build_config, build_counts)

    python sample_points.py --prep FL
    python sample_points.py FL --protocols band --candidates 25000 \\
        --state-polygons data/interim/gis/counties_FL.parquet \\
        --land-mask data/interim/gis/nlcd2021_FL.tif
    python sample_points.py --audit FL
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import CRS, Transformer
from shapely import STRtree

IN_DIR = Path("data/interim/nfhl_states")
PREP_DIR = Path("data/interim/prep")
OUT_DIR = Path("data/interim/points")

STATES = ["FL"]     # state codes the CLI accepts; Florida is the only state prepped

CRS_SRC = 4269      # NAD83 geographic, as NFHL ships
CRS_WORK = 5070     # CONUS Albers Equal Area, metres

NODATA = -9999.0    # FEMA's null sentinel in STATIC_BFE and DEPTH
RG_ROWS = 10_000    # rows per row group in the prep repack; read_geoms relies on it

ATTRS = ["DFIRM_ID", "FLD_AR_ID", "STUDY_TYP", "FLD_ZONE", "ZONE_SUBTY",
         "SFHA_TF", "STATIC_BFE", "DEPTH", "SOURCE_CIT"]

# Every key is also a CLI option (--candidates, --min-sep-m, ...), in this order.
DEFAULTS = dict(
    candidates=25000,     # valid candidates per group (zone and label, or zone and protocol)
    min_sep_m=100.0,      # floor between any two points, any class or zone; de-cluttering only
    block_km=10.0,        # spatial CV block edge; spatial autocorrelation is handled by blocks
    band_min_m=500.0,     # band negatives: nearest SFHA at least this far
    band_max_m=5000.0,    # ... and at most this far
    protocols="band,shadedX",
    state_polygons="",    # optional polygon file (e.g. counties) band negatives must lie inside
    seed=20260918,
)
PROTOCOLS = ("band", "shadedX")

# First batch of proposals per valid candidate wanted (draw_valid): about the inverse of the share
# that passes (positives: 85-97% on land; band: about 8.5% in the 2026-10-23 build, most of the
# rest closer than 500 m to an SFHA). Later batches use the share actually seen.
FIRST_BATCH = {"positive": 1.25, "band": 13.0, "shadedX": 1.25}
MAX_BATCHES = 20
MAX_GROWTH = 10          # a batch proposes at most this many times the one before
MAX_BATCH = 5_000_000    # and at most this many candidates
EMPTY_AFTER = 100        # no valid candidate after this many proposals per one wanted: give up
UNASSESSED_ZONES = ("D", "AREA NOT INCLUDED")   # inside FIRM panels but never studied
OPEN_WATER_ZONE = "OPEN WATER"    # NFHL's mapped open water; --land-mask keeps band negatives out

# --land-mask raster codes (NLCD classes, pull_land_mask.py). A point is on land if its pixel
# holds a class other than these two; a point off the raster is not on land.
MASK_NODATA = 0       # no data: offshore, or outside NLCD
MASK_OPEN_WATER = 11  # NLCD open water

# Prefix of every column that records how the map was made. Never a model feature: STUDY_TYP
# alone predicts Zone A vs AE well above the base rate, and meta_dist_to_sfha_m is the distance
# to the labelled zone itself.
META_PREFIX = "meta_"

MAX_RINGS = 10_000        # distinct polygons buffered for the band (FL: ~4,400 per zone)
RING_SIMPLIFY_M = 100.0   # simplification before buffering; the exact distance filter decides

CONFIG_KEY = b"sample_points"   # parquet metadata key holding the build settings
COUNTS_KEY = b"sample_points_counts"   # ... and the candidate and survivor counts per group
LEAD_COLUMNS = ["point_id", "state", "county_fips", "dfirm_id", "block_id",
                "domain", "label", "neg_protocol", "lon", "lat"]


class SupplyError(RuntimeError):
    """A group could not reach its candidate count within MAX_BATCHES batches."""


# =========================================================================
# small helpers
# =========================================================================
def study_regime(study_typ: pd.Series) -> pd.Series:
    """Collapse the mapping partners' free-text STUDY_TYP into comparable study mechanisms.

    Zone A means different things under each (a Base Level Engineering risk class, a classic
    approximate study, ...), so the regime is kept apart. Later rules override earlier ones.
    """
    text = study_typ.fillna("").astype(str)
    text = text.str.strip().str.upper()
    regime = pd.Series("OTHER", index=text.index, dtype=object)
    regime[text.eq("NP") | text.eq("")] = "NOT_POPULATED"
    regime[text.str.contains("FLOOD RISK", na=False)] = "BLE_RISKCLASS"
    regime[text.str.contains("REDELINEATION", na=False)] = "REDELINEATION"
    regime[text.str.contains("DIGITAL CONVERSION", na=False)] = "DIGITAL_CONVERSION"
    regime[text.str.contains("WITHOUT BFE|UNPUBLISHED BFE", na=False)] = "NO_BFE"
    regime[text.str.contains("WITH BFE", na=False)] = "BFE_DETAIL"
    regime[text.str.contains("BLE AVAILABLE", na=False)] = "BLE_UNPUBLISHED"
    return regime


def clean_sentinel(values) -> np.ndarray:
    """Float copy of `values` with FEMA's -9999 sentinel, and anything non-numeric, as NaN."""
    numbers = pd.to_numeric(pd.Series(values), errors="coerce")
    out = np.array(numbers, dtype=float, copy=True)
    out[np.isclose(out, NODATA)] = np.nan
    return out


def _reproject(geoms: np.ndarray, transformer: Transformer) -> np.ndarray:
    """Apply a pyproj transformer to every coordinate of an array of shapely geometries."""
    def transform_xy(coords):
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geoms, transform_xy)


def _is_present(geoms) -> np.ndarray:
    """Mask of the geometries read_geoms found (it returns None for an unknown row id)."""
    return np.array([g is not None for g in geoms], dtype=bool)


def _query_mask(tree: STRtree, pts: np.ndarray, predicate: str) -> np.ndarray:
    """Mask of the points that satisfy `predicate` against at least one geometry in tree."""
    hit = np.zeros(len(pts), dtype=bool)
    hit[tree.query(pts, predicate=predicate)[0]] = True
    return hit


def _no_points():
    return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)


def _indented(text: str) -> str:
    return "    " + text.replace("\n", "\n    ")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# =========================================================================
# land mask (--land-mask)
# =========================================================================
class LandMask:
    """A land-cover raster with NLCD class codes, read whole (Florida: about 0.7 GB of uint8).

    Pixels are looked up from EPSG:5070 x, y directly when pyproj says the raster's CRS equals
    EPSG:5070; otherwise each point is transformed into the raster's CRS first.
    """

    def __init__(self, path):
        import rasterio     # only a build or audit with --land-mask reads a raster
        self.path = str(path)
        with rasterio.open(path) as src:
            self.classes = src.read(1)
            transform = src.transform
            self.crs = CRS.from_user_input(src.crs.to_wkt())
        if transform.b != 0 or transform.d != 0:
            raise ValueError(f"{path}: a rotated pixel grid is not supported")
        self.x0, self.dx = transform.c, transform.a
        self.y0, self.dy = transform.f, transform.e
        self.to_raster = None
        if self.crs != CRS.from_epsg(CRS_WORK):
            self.to_raster = Transformer.from_crs(CRS_WORK, self.crs, always_xy=True)
        self.sha256 = file_sha256(path)

    def describe(self) -> str:
        height, width = self.classes.shape
        if self.to_raster is None:
            lookup = f"EPSG:{CRS_WORK} x, y read directly"
        else:
            lookup = f"points transformed by '{self.to_raster.description}'"
        return f"{self.path} ({width:,} x {height:,} pixels; {lookup})"

    def classes_at(self, x, y) -> np.ndarray:
        """The class of the pixel holding each point (EPSG:5070 metres); MASK_NODATA off it."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.to_raster is not None and x.size:
            x, y = self.to_raster.transform(x, y)
        col = np.floor((x - self.x0) / self.dx)
        row = np.floor((y - self.y0) / self.dy)
        height, width = self.classes.shape
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        out = np.full(x.shape, MASK_NODATA, dtype=np.uint8)
        out[inside] = self.classes[row[inside].astype(np.int64), col[inside].astype(np.int64)]
        return out

    @staticmethod
    def is_land(classes) -> np.ndarray:
        classes = np.asarray(classes)
        return (classes != MASK_NODATA) & (classes != MASK_OPEN_WATER)


def on_land(land: LandMask, x, y, owner):
    """The candidates that lie on land per the mask: (x, y, owner, the class each one reads).

    It runs on place_points' output, before thinning. Dropping candidates from an area-uniform
    draw leaves an area-uniform draw over the land; rejecting water inside place_points' loop
    instead would keep each polygon's count proportional to its whole area, over-sampling the
    land of mostly-water polygons.
    """
    classes = land.classes_at(x, y)
    keep = LandMask.is_land(classes)
    return x[keep], y[keep], owner[keep], classes[keep]


# =========================================================================
# prep: stream once, write an index and a reprojected repack
# =========================================================================
def prep_state(state: str, force: bool = False) -> None:
    """Stream haz_{state}.parquet once into an attribute index and a reprojected repack.

    The source parquet is a single row group (Florida: 5.2 GB, whose geometry expands to tens of
    GB in memory), so it is never read whole. Later runs do their allocation arithmetic on the
    index and read geometry only for the polygons they sample, one row group at a time.
    """
    src = IN_DIR / f"haz_{state}.parquet"
    idx_path = PREP_DIR / f"idx_{state}.parquet"
    geo_path = PREP_DIR / f"haz_{state}_5070.parquet"
    if not src.exists():
        print(f"[{state}] missing {src} -- skipping")
        return
    if idx_path.exists() and geo_path.exists() and not force:
        print(f"[{state}] prepped already -- skipping (use --force)")
        return

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    source = pq.ParquetFile(src)
    total = source.metadata.num_rows
    print(f"[{state}] prep: {total:,} polygons from "
          f"{src.stat().st_size/1e9:.2f} GB", flush=True)

    transformer = Transformer.from_crs(CRS_SRC, CRS_WORK, always_xy=True)
    writer = None
    index_parts = []
    seen = 0
    n_fixed = 0
    for batch in source.iter_batches(batch_size=RG_ROWS, columns=ATTRS + ["geometry"]):
        geoms = shapely.from_wkb(np.asarray(batch.column("geometry")))
        geoms = _reproject(geoms, transformer)

        invalid = ~shapely.is_valid(geoms)
        if invalid.any():
            # Some NFHL polygons have self-touching rings from digitising. contains_xy is
            # undefined on an invalid polygon, and dropping them would bias toward simple shapes.
            geoms[invalid] = shapely.make_valid(geoms[invalid])
            n_fixed += int(invalid.sum())

        part = batch.select(ATTRS).to_pandas()
        part["row_id"] = np.arange(seen, seen + len(part), dtype=np.int64)
        part["area_m2"] = shapely.area(geoms)
        part[["minx", "miny", "maxx", "maxy"]] = shapely.bounds(geoms)
        index_parts.append(part)

        table = pa.table({"row_id": pa.array(part["row_id"].values),
                          "geometry": pa.array(shapely.to_wkb(geoms))})
        if writer is None:
            writer = pq.ParquetWriter(geo_path, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=RG_ROWS)

        seen += len(part)
        if seen % (RG_ROWS * 20) == 0:
            print(f"    {seen:,}/{total:,}  ({time.time()-t0:.0f}s)", flush=True)

    if writer is not None:
        writer.close()

    idx = pd.concat(index_parts, ignore_index=True)
    idx["study_regime"] = study_regime(idx["STUDY_TYP"])
    idx["STATIC_BFE"] = clean_sentinel(idx["STATIC_BFE"])
    idx["DEPTH"] = clean_sentinel(idx["DEPTH"])
    idx.to_parquet(idx_path)

    if n_fixed:
        print(f"    repaired {n_fixed:,} invalid polygons")
    print(f"[{state}] prep done: index {idx_path.stat().st_size/1e6:.0f} MB, "
          f"repack {geo_path.stat().st_size/1e9:.2f} GB  ({time.time()-t0:.0f}s)")


def load_index(state: str) -> pd.DataFrame:
    path = PREP_DIR / f"idx_{state}.parquet"
    if not path.exists():
        sys.exit(f"{path} missing -- run:  python sample_points.py --prep {state}")
    return pd.read_parquet(path)


def read_geoms(state: str, row_ids) -> np.ndarray:
    """Geometry (EPSG:5070) for the given row_ids, in their order; None for an unknown id.

    prep numbers rows in file order, so row id // RG_ROWS is the row group that holds it and only
    those row groups are read. RG_ROWS must be the value prep ran with.
    """
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.size == 0:
        return np.empty(0, dtype=object)
    repack = pq.ParquetFile(PREP_DIR / f"haz_{state}_5070.parquet")

    wanted = pd.unique(row_ids)
    geom_by_id = {}
    for group in np.unique(wanted // RG_ROWS):
        if group >= repack.num_row_groups:
            continue
        table = repack.read_row_group(int(group), columns=["row_id", "geometry"])
        group_ids = table.column("row_id").to_numpy()
        hit = np.isin(group_ids, wanted)
        if not hit.any():
            continue
        geoms = shapely.from_wkb(np.asarray(table.column("geometry"))[hit])
        for row_id, geom in zip(group_ids[hit], geoms):
            geom_by_id[int(row_id)] = geom
    return np.array([geom_by_id.get(int(r)) for r in row_ids], dtype=object)


# =========================================================================
# geometry helpers
# =========================================================================
def grid_thin(x: np.ndarray, y: np.ndarray, min_sep: float,
              rng: np.random.Generator) -> np.ndarray:
    """Greedy minimum-separation thinning. Returns the kept indices, in random visit order.

    Points are visited in random order and kept unless an already-kept point lies within min_sep.
    Kept points are hashed into square cells of edge min_sep, so only the point's own cell and its
    eight neighbours need checking: O(n) instead of O(n^2) over a million Florida candidates.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if min_sep <= 0:
        return np.arange(n, dtype=np.int64)

    order = rng.permutation(n)
    cell_x = np.floor(x / min_sep).astype(np.int64)
    cell_y = np.floor(y / min_sep).astype(np.int64)
    min_sep_sq = min_sep * min_sep

    kept_in_cell: dict[tuple[int, int], list[int]] = {}
    keep: list[int] = []
    for i in order:
        cell = (cell_x[i], cell_y[i])
        if _has_kept_neighbour(x[i], y[i], cell, x, y, kept_in_cell, min_sep_sq):
            continue
        keep.append(i)
        kept_in_cell.setdefault(cell, []).append(i)
    return np.asarray(keep, dtype=np.int64)


def _has_kept_neighbour(xi, yi, cell, x, y, kept_in_cell, min_sep_sq) -> bool:
    """True if a kept point in the 3x3 block of cells around `cell` is closer than min_sep."""
    for dx_cell in (-1, 0, 1):
        for dy_cell in (-1, 0, 1):
            for j in kept_in_cell.get((cell[0] + dx_cell, cell[1] + dy_cell), ()):
                dx = xi - x[j]
                dy = yi - y[j]
                if dx * dx + dy * dy < min_sep_sq:
                    return True
    return False


def sample_in_polygons(geoms: np.ndarray, weights, n: int, cap: int | None,
                       rng: np.random.Generator, max_rounds: int = 30):
    """n points over the union of geoms, each polygon drawn in proportion to its weight (its area:
    a uniform draw), at most cap per polygon (no cap when cap is None).

    Returns (x, y, index into geoms).
    """
    if n <= 0 or len(geoms) == 0:
        return _no_points()
    quota = polygon_quota(weights, n, cap, rng)
    return place_points(geoms, quota, rng, max_rounds)


def polygon_quota(weights, n: int, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Points per polygon: n draws with replacement in proportion to weight, at most cap each
    (no cap when cap is None).

    Drawing polygons by area and placing one uniform point in each is a uniform draw over their
    union; a fixed count per polygon would over-represent small polygons. Positives use no cap,
    so they stay uniform by area (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D2). The band's
    rings keep one (sample_band_negatives); draws it cuts off are redrawn once among the polygons
    below it. Needs no geometry, so it runs over a whole zone's index before any geometry is read.
    """
    weight = np.asarray(weights, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    if n <= 0 or weight.sum() <= 0:
        return np.zeros(len(weight), dtype=np.int64)

    quota = rng.multinomial(n, weight / weight.sum())
    if cap is None:
        return quota
    quota = np.minimum(quota, cap)
    deficit = n - int(quota.sum())
    if deficit > 0:
        room = cap - quota
        room_weight = np.where(room > 0, weight, 0.0)
        if room_weight.sum() > 0:
            redrawn = rng.multinomial(deficit, room_weight / room_weight.sum())
            quota = quota + np.minimum(redrawn, room)
    return quota


def place_points(geoms: np.ndarray, quota, rng: np.random.Generator,
                 max_rounds: int = 30):
    """quota[i] uniform points inside geoms[i], by rejection sampling from its bounding box.

    Returns (x, y, index into geoms). Each round proposes 2 / fill points per point still missing
    (fill = polygon area / bounding-box area), so a thin diagonal ribbon gets its full quota just
    as a square does. A very thin polygon can miss a whole round; max_rounds bounds the loop.
    """
    quota = np.asarray(quota, dtype=np.int64)
    live = np.nonzero(quota > 0)[0]
    if live.size == 0:
        return _no_points()

    polys = geoms[live]
    bounds = shapely.bounds(polys)
    need = quota[live]
    with np.errstate(divide="ignore", invalid="ignore"):
        bbox_area = (bounds[:, 2] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 1])
        fill = shapely.area(polys) / bbox_area
    # Floored at 0.002 so a degenerate polygon cannot ask for millions of proposals.
    fill = np.clip(np.nan_to_num(fill, nan=0.0, posinf=1.0), 0.002, 1.0)
    proposals_per_point = 2.0 / fill

    xs, ys, owners = [], [], []
    for _ in range(max_rounds):
        hot = np.nonzero(need > 0)[0]
        if hot.size == 0:
            break
        n_proposals = np.ceil(need[hot] * proposals_per_point[hot]).astype(np.int64)
        owner = np.repeat(hot, np.maximum(n_proposals, 1))
        box = bounds[owner]
        x = rng.uniform(box[:, 0], box[:, 2])
        y = rng.uniform(box[:, 1], box[:, 3])
        inside = shapely.contains_xy(polys[owner], x, y)
        if not inside.any():
            continue
        x, y, owner = x[inside], y[inside], owner[inside]

        # Keep the first need[k] hits of each polygon k. `owner` is already sorted: `hot` is
        # ascending and np.repeat and the mask keep that order.
        uniq, start, counts = np.unique(owner, return_index=True, return_counts=True)
        rank = np.arange(len(owner)) - np.repeat(start, counts)
        take = rank < np.repeat(need[uniq], counts)
        need[uniq] -= np.minimum(counts, need[uniq])

        xs.append(x[take])
        ys.append(y[take])
        owners.append(owner[take])

    if not xs:
        return _no_points()
    return np.concatenate(xs), np.concatenate(ys), live[np.concatenate(owners)]


def distances_to_sfha(state: str, pts: np.ndarray, sfha_idx: pd.DataFrame,
                      radius: float, tile_m: float = 50_000.0) -> np.ndarray:
    """Exact distance (m) from each point to the nearest SFHA polygon.

    Points are grouped into tiles of tile_m; for each tile only the SFHA polygons whose bounding
    box reaches within `radius` of it are read, and an STRtree gives each point its single nearest
    polygon. That is exact without loading the state's hazard layer, and avoids the tens of
    millions of candidate pairs a 5 km dwithin query gives against ribbon-shaped polygons. A point
    whose tile has no SFHA polygon within `radius` gets inf.
    """
    n = len(pts)
    if n == 0:
        return np.empty(0)
    if len(sfha_idx) == 0:
        return np.full(n, np.inf)

    px = shapely.get_x(pts)
    py = shapely.get_y(pts)
    dist = np.full(n, np.inf)

    min_x = sfha_idx["minx"].to_numpy()
    max_x = sfha_idx["maxx"].to_numpy()
    min_y = sfha_idx["miny"].to_numpy()
    max_y = sfha_idx["maxy"].to_numpy()
    row_ids = sfha_idx["row_id"].to_numpy()

    tile_x = np.floor(px / tile_m).astype(np.int64)
    tile_y = np.floor(py / tile_m).astype(np.int64)
    for tx, ty in set(zip(tile_x.tolist(), tile_y.tolist())):
        in_tile = (tile_x == tx) & (tile_y == ty)
        x0 = px[in_tile].min() - radius
        x1 = px[in_tile].max() + radius
        y0 = py[in_tile].min() - radius
        y1 = py[in_tile].max() + radius
        near = (max_x >= x0) & (min_x <= x1) & (max_y >= y0) & (min_y <= y1)
        if not near.any():
            continue
        geoms = read_geoms(state, row_ids[near])
        geoms = geoms[_is_present(geoms)]
        if len(geoms) == 0:
            continue
        tile_pts = pts[in_tile]
        nearest = STRtree(geoms).nearest(tile_pts)
        dist[in_tile] = shapely.distance(tile_pts, geoms[nearest])
    return dist


def draw_valid(propose, n: int, first: int, rng: np.random.Generator,
               max_batches: int = MAX_BATCHES) -> tuple[dict, dict]:
    """Exactly n valid candidates, in random order: (their columns, counts summed over batches).

    propose(k) proposes k independent candidates from one distribution and returns (the columns of
    the valid ones, a dict of counts with "proposed" and how many each test removed). The first
    batch proposes `first`; each later one the shortfall over the share valid so far, plus 20%,
    but at most MAX_GROWTH times the batch before and at most MAX_BATCH (a share near zero would
    otherwise ask for billions). With no valid candidate after EMPTY_AFTER proposals per candidate
    wanted, the group cannot be supplied.
    The valid candidates of every batch are independent draws from the valid region, whatever the
    number of batches, so a random n of the pool are too. counts gains "batches", "valid" (the
    pool, before the cut to n) and "candidates" (n). Raises SupplyError after max_batches.
    """
    parts, counts = [], {"proposed": 0}
    have, k = 0, max(int(first), 1)
    for _ in range(max_batches):
        cols, batch_counts = propose(k)
        for key, value in batch_counts.items():
            counts[key] = counts.get(key, 0) + int(value)
        counts["batches"] = counts.get("batches", 0) + 1
        parts.append(cols)
        have += len(cols["x"])
        if have >= n:
            break
        if have == 0 and counts["proposed"] >= EMPTY_AFTER * n:
            break
        share = max(have, 1) / counts["proposed"]
        k = min(int(np.ceil((n - have) / share * 1.2)), MAX_GROWTH * k, MAX_BATCH)
    if have < n:
        raise SupplyError(f"{have:,} valid candidates of the {n:,} wanted after {max_batches} "
                          f"batches ({counts['proposed']:,} proposed)")
    pooled = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    take = rng.permutation(have)[:n]
    counts.update(valid=have, candidates=n)
    return {key: values[take] for key, values in pooled.items()}, counts


# =========================================================================
# point frames
# =========================================================================
def _attr_frame(att: pd.DataFrame, x, y, dist) -> pd.DataFrame:
    """Point rows carrying the attributes of the index row (polygon) behind each point.

    meta_src_row_id is that polygon's row_id: FLD_AR_ID alone is not unique within a state.
    """
    if "DEPTH" in att:
        depth = att["DEPTH"].values
    else:
        depth = np.full(len(att), np.nan)
    return pd.DataFrame({
        "dfirm_id": att["DFIRM_ID"].values,
        "x": x, "y": y,
        "meta_fld_zone": att["FLD_ZONE"].values,
        "meta_zone_subty": att["ZONE_SUBTY"].values,
        "meta_study_typ": att["STUDY_TYP"].values,
        "meta_study_regime": att["study_regime"].values,
        "meta_static_bfe": att["STATIC_BFE"].values,
        "meta_depth": depth,
        "meta_source_cit": att["SOURCE_CIT"].values,
        "meta_fld_ar_id": att["FLD_AR_ID"].values,
        "meta_poly_area_km2": att["area_m2"].values / 1e6,
        "meta_dist_to_sfha_m": dist,
        "meta_src_row_id": att["row_id"].to_numpy(dtype=np.int64),
    })


def _blank_frame(dfirm, x, y, dist, src_row_id) -> pd.DataFrame:
    """Point rows for band negatives, which lie in no flood-zone polygon and so carry no zone
    attributes. meta_src_row_id is the positive polygon whose ring produced the point."""
    n = len(x)
    return pd.DataFrame({
        "dfirm_id": dfirm, "x": x, "y": y,
        "meta_fld_zone": _missing_strings(n), "meta_zone_subty": _missing_strings(n),
        "meta_study_typ": _missing_strings(n), "meta_study_regime": _missing_strings(n),
        "meta_static_bfe": np.full(n, np.nan), "meta_depth": np.full(n, np.nan),
        "meta_source_cit": _missing_strings(n), "meta_fld_ar_id": _missing_strings(n),
        "meta_poly_area_km2": np.full(n, np.nan),
        "meta_dist_to_sfha_m": dist,
        "meta_src_row_id": np.asarray(src_row_id, dtype=np.int64),
    })


def _missing_strings(n: int):
    return pd.array([pd.NA] * n, dtype="string")


def _tag(frame: pd.DataFrame, domain: str, label: int, protocol) -> pd.DataFrame:
    """Set the stratum columns of a block of sampled points."""
    frame["domain"] = domain
    frame["label"] = label
    frame["neg_protocol"] = protocol
    return frame


# =========================================================================
# negatives
# =========================================================================
def band_filter(state, x, y, sfha_idx, pan_tree, cfg, state_tree=None, unassessed_tree=None):
    """Which band candidates (EPSG:5070 x, y) pass the band and every mask, tested in the build's
    order: (valid mask, distance to the nearest SFHA, mask class or None, counts).

    A point counts as absence only where FEMA assessed the area: band_min_m to band_max_m from
    every SFHA (the exact distance), inside a FIRM panel (pan_tree), outside Zone D and AREA NOT
    INCLUDED (unassessed_tree), with --land-mask outside the NFHL OPEN WATER polygons
    (cfg["open_water_tree"]) and on land (cfg["land"]), and, given state_tree, inside the state,
    whose NFHL lacks the neighbours' floodplains. Each count is of the candidates that passed every
    earlier test.
    """
    pts = shapely.points(x, y)
    band_min, band_max = cfg["band_min_m"], cfg["band_max_m"]
    dist = distances_to_sfha(state, pts, sfha_idx, band_max)
    too_near = dist < band_min
    too_far = dist > band_max
    ok = ~too_near & ~too_far
    counts = dict(too_near=int(too_near.sum()), too_far=int(too_far.sum()))
    if pan_tree is not None:
        inside = _query_mask(pan_tree, pts, "within")
        counts["outside_panel"] = int((ok & ~inside).sum())
        ok &= inside
    if unassessed_tree is not None:
        unassessed = _query_mask(unassessed_tree, pts, "intersects")
        counts["unassessed"] = int((ok & unassessed).sum())
        ok &= ~unassessed
    mask_lc = None
    land = cfg.get("land")
    if land is not None:
        if cfg.get("open_water_tree") is not None:
            water = _query_mask(cfg["open_water_tree"], pts, "intersects")
            counts["open_water"] = int((ok & water).sum())
            ok &= ~water
        mask_lc = land.classes_at(x, y)
        land_ok = LandMask.is_land(mask_lc)
        counts["not_on_land"] = int((ok & ~land_ok).sum())
        ok &= land_ok
    if state_tree is not None:
        home = _query_mask(state_tree, pts, "within")
        counts["outside_state"] = int((ok & ~home).sum())
        ok &= home
    return ok, dist, mask_lc, counts


def _counts_text(counts: dict, labels: dict) -> str:
    """The counts named in labels, in its order, as "N label" phrases."""
    return ", ".join(f"{counts[key]:,} {label}" for key, label in labels.items() if key in counts)


BAND_LABELS = {"too_near": "< band_min", "too_far": "> band_max",
               "outside_panel": "outside a panel",
               "unassessed": "in " + " or ".join(UNASSESSED_ZONES),
               "open_water": f"in {OPEN_WATER_ZONE}", "not_on_land": "not on land per the mask",
               "outside_state": "outside the state polygons", "unplaced": "never placed"}


def sample_band_negatives(state, src_idx, src_geoms, sfha_idx, pan_tree, n, cfg, rng,
                          state_tree=None, unassessed_tree=None):
    """Exactly n band candidates (band_filter), thinned to min_sep_m with every survivor kept:
    (frame, counts), or None without source polygons.

    The band is a ring around the polygons behind the domain's positive candidates, so the
    protocol never reads FLD_ZONE and is identical for A and AE. The ring only bounds the
    candidates (band_filter decides), so it is built from simplified polygons and widened by the
    simplification tolerance; its 8-segment corners sit at most ~25 m inside the true buffer. A
    ring is drawn in proportion to its area, and no ring may supply more than four times an even
    share of a batch, so a few huge polygons cannot dominate; overlapping rings make the band
    denser where many polygons are near. Each kept point records its mask class in meta_mask_lc
    (with --land-mask). Raises SupplyError when the band cannot supply n.
    """
    if n <= 0 or len(src_geoms) == 0:
        return None

    ring_src = np.arange(len(src_geoms))
    flag = ""
    if len(ring_src) > MAX_RINGS:
        ring_src = rng.choice(ring_src, MAX_RINGS, replace=False)
        flag = "   <-- RING CAP BINDING: a random subset of rings defines the band"
    print(f"    band rings from {len(src_geoms):,} polygons "
          f"(cap {MAX_RINGS:,}){flag}", flush=True)
    simplified = shapely.simplify(src_geoms[ring_src], RING_SIMPLIFY_M)
    rings = shapely.buffer(simplified, cfg["band_max_m"] + RING_SIMPLIFY_M, quad_segs=8)
    ring_area = shapely.area(rings)

    def propose(k):
        cap = max(4, int(np.ceil(k / max(len(rings) // 4, 1))))
        x, y, ring_i = sample_in_polygons(rings, ring_area, k, cap, rng)
        ok, dist, mask_lc, counts = band_filter(state, x, y, sfha_idx, pan_tree, cfg,
                                                state_tree, unassessed_tree)
        counts.update(proposed=k, unplaced=k - len(x))
        cols = dict(x=x[ok], y=y[ok], ring=ring_i[ok], dist=dist[ok])
        if mask_lc is not None:
            cols["mask_lc"] = mask_lc[ok]
        return cols, counts

    cand, counts = draw_valid(propose, n, int(np.ceil(n * FIRST_BATCH["band"])), rng)
    thinned = grid_thin(cand["x"], cand["y"], cfg["min_sep_m"], rng)
    counts["after_group_thinning"] = len(thinned)
    counts["rings"] = len(rings)
    labels = dict(BAND_LABELS, too_near=f"< {cfg['band_min_m']:.0f} m",
                  too_far=f"> {cfg['band_max_m']:.0f} m")
    print(f"    band: {counts['proposed']:,} proposed in {counts['batches']} batches: "
          f"{_counts_text(counts, labels)}; {counts['valid']:,} valid, {n:,} candidates, "
          f"{len(thinned):,} after thinning", flush=True)
    src = ring_src[cand["ring"][thinned]]
    frame = _blank_frame(src_idx["DFIRM_ID"].to_numpy()[src], cand["x"][thinned],
                         cand["y"][thinned], cand["dist"][thinned],
                         src_idx["row_id"].to_numpy()[src])
    if "mask_lc" in cand:
        frame["meta_mask_lc"] = cand["mask_lc"][thinned]
    return frame, counts


def area_uniform_candidates(state, polys: pd.DataFrame, n: int, cfg: dict, rng, kind: str):
    """Exactly n candidates uniform by area over the land of polys (index rows with row_id and
    area_m2), with no polygon cap: (columns x, y, owner (the row of polys) and, with
    --land-mask, mask_lc; counts; the geometry of every polygon read, by row_id).

    Each batch draws its polygons in proportion to area over all of polys and reads geometry only
    for the polygons that drew a point, once. A placed candidate not on land per cfg["land"]
    fails (on_land): dropping points from an area-uniform draw leaves an area-uniform draw over
    the land, where rejecting water inside place_points would keep each polygon's count
    proportional to its whole area and over-sample the land of mostly-water polygons.
    """
    areas = polys["area_m2"].to_numpy()
    row_ids = polys["row_id"].to_numpy()
    cache: dict[int, object] = {}

    def propose(k):
        quota = polygon_quota(areas, k, None, rng)
        live = np.nonzero(quota > 0)[0]
        new = [int(r) for r in row_ids[live] if int(r) not in cache]
        if new:
            cache.update(zip(new, read_geoms(state, new)))
        geoms = np.array([cache[int(r)] for r in row_ids[live]], dtype=object)
        found = _is_present(geoms)
        x, y, i = place_points(geoms[found], quota[live][found], rng)
        owner = live[found][i]
        counts = dict(proposed=k, no_geometry=int(quota[live][~found].sum()),
                      unplaced=int(quota[live][found].sum()) - len(x))
        cols = dict(x=x, y=y, owner=owner)
        if cfg.get("land") is not None:
            x, y, owner, mask_lc = on_land(cfg["land"], x, y, owner)
            counts["not_on_land"] = len(cols["x"]) - len(x)
            cols = dict(x=x, y=y, owner=owner, mask_lc=mask_lc)
        return cols, counts

    cand, counts = draw_valid(propose, n, int(np.ceil(n * FIRST_BATCH[kind])), rng)
    return cand, counts, cache


def sample_shaded_x(state, domain, shaded_idx, src, sfha_idx, n, cfg, rng):
    """Exactly n shadedX candidates, uniform by area over the land of the Zone X 0.2%-chance
    polygons of the DFIRMs behind the domain's positives, thinned to min_sep_m with every
    survivor kept: (frame, counts), or None. They control for whether a place was mapped at all,
    which the distance band does not."""
    candidates = shaded_idx[shaded_idx.DFIRM_ID.isin(set(src.DFIRM_ID))]
    if len(candidates) == 0:
        print(f"  [{domain}] negatives shadedX none in these DFIRMs")
        return None
    cand, counts, _ = area_uniform_candidates(state, candidates, n, cfg, rng, "shadedX")
    thinned = grid_thin(cand["x"], cand["y"], cfg["min_sep_m"], rng)
    counts["after_group_thinning"] = len(thinned)
    x, y, owner = cand["x"][thinned], cand["y"][thinned], cand["owner"][thinned]
    dist = distances_to_sfha(state, shapely.points(x, y), sfha_idx, cfg["band_max_m"])
    frame = _tag(_attr_frame(candidates.iloc[owner], x, y, dist), domain, 0, "shadedX")
    if "mask_lc" in cand:
        frame["meta_mask_lc"] = cand["mask_lc"][thinned]
    print(f"  [{domain}] negatives shadedX: {counts['proposed']:,} proposed, "
          f"{counts.get('not_on_land', 0):,} not on land; {n:,} candidates, "
          f"{len(thinned):,} after thinning", flush=True)
    return frame, counts


def stratum_sizes(df: pd.DataFrame) -> pd.Series:
    """Rows per (domain, label, neg_protocol); positives' protocol shows as '-'."""
    return df.fillna({"neg_protocol": "-"}).groupby(["domain", "label", "neg_protocol"]).size()


def group_key(domain: str, label: int, protocol: str) -> str:
    """A group's name in the stored counts: "A/1/-", "AE/0/band"."""
    return f"{domain}/{int(label)}/{protocol}"


# =========================================================================
# per-point columns
# =========================================================================
def attach_panel_meta(df: pd.DataFrame, pan) -> pd.DataFrame:
    """Add the effective date, scale and type of the FIRM panel each point lies in (in place).

    Panels are whole printed sheets, so neighbouring counties' panels overlap and about a third of
    Florida's points lie in two to four of them. The point's own DFIRM is the map that labelled
    it, so its panel wins, then the lowest FIRM_PAN (the rule pull_point_extras.py uses for
    meta_firm_pan); another DFIRM's panel is used only when the point lies in none of its own.
    """
    for column in ("meta_eff_date", "meta_panel_scale", "meta_panel_typ"):
        df[column] = pd.NA
    if pan is None or not len(pan):
        return df
    pts = gpd.GeoDataFrame({"_dfirm": df["dfirm_id"].values},
                           geometry=gpd.points_from_xy(df.x, df.y), crs=CRS_WORK)
    cols = [c for c in ("EFF_DATE", "SCALE", "PANEL_TYP", "DFIRM_ID", "FIRM_PAN")
            if c in pan.columns]
    joined = gpd.sjoin(pts, pan[cols + ["geometry"]], how="left", predicate="within")
    if "DFIRM_ID" in joined.columns:
        own = (joined["DFIRM_ID"] == joined["_dfirm"]).fillna(False)
        joined = joined.assign(_own=own)
        if "FIRM_PAN" in joined.columns:
            joined = joined.sort_values(["_own", "FIRM_PAN"], ascending=[False, True],
                                        kind="stable")
        else:
            joined = joined.sort_values(["_own"], ascending=[False], kind="stable")
    first = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    if "EFF_DATE" in cols:
        df["meta_eff_date"] = first["EFF_DATE"].values
    if "SCALE" in cols:
        df["meta_panel_scale"] = pd.to_numeric(first["SCALE"], errors="coerce").values
    if "PANEL_TYP" in cols:
        df["meta_panel_typ"] = first["PANEL_TYP"].values
    return df


def point_counties(df: pd.DataFrame, polys) -> pd.api.extensions.ExtensionArray:
    """GEOID of the polygon in `polys` each point (df.x, df.y in EPSG:5070) lies in.

    NA for a point in none of them, and for every point when `polys` is None or has no GEOID
    column (a --state-polygons file that is not a county layer). The first polygon wins.
    """
    if polys is None or "GEOID" not in polys.columns or not len(df):
        return pd.array([pd.NA] * len(df), dtype="string")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.x, df.y), crs=CRS_WORK)
    right = polys[["GEOID", "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(pts, right, how="left", predicate="within")
    joined = joined.sort_values("index_right", kind="stable")
    first = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    return first["GEOID"].astype("string").array


def bfe_density_map(state: str, sfha_idx: pd.DataFrame) -> dict:
    """BFE line length per km2 of SFHA, per DFIRM: a continuous measure of study detail.

    Detailed studies publish their base flood elevations in S_BFE as lines, so line length per
    area grades how detailed a DFIRM's study was where the binary A/AE split cannot.
    """
    path = IN_DIR / f"s_bfe_{state}.parquet"
    if not path.exists():
        return {}
    try:
        bfe = gpd.read_parquet(path).to_crs(CRS_WORK)
        bfe["length_km"] = bfe.geometry.length / 1000
        km_by_dfirm = bfe.groupby("DFIRM_ID")["length_km"].sum()
        km2_by_dfirm = sfha_idx.groupby("DFIRM_ID")["area_m2"].sum() / 1e6
        density = (km_by_dfirm / km2_by_dfirm).replace([np.inf, -np.inf], np.nan)
        return density.dropna().to_dict()
    except Exception as err:
        print(f"  BFE density skipped: {type(err).__name__}: {err}")
        return {}


def block_ids(x: pd.Series, y: pd.Series, block_km: float) -> pd.Series:
    """'col_row' of the block_km grid cell each point (EPSG:5070 metres) lies in."""
    edge_m = block_km * 1000.0
    col = np.floor(x / edge_m).astype(np.int64).astype(str)
    row = np.floor(y / edge_m).astype(np.int64).astype(str)
    return col + "_" + row


def add_point_columns(df: pd.DataFrame, state: str, cfg: dict, bfe: dict,
                      state_polys) -> gpd.GeoDataFrame:
    """Add the identifying columns and lon/lat; return a GeoDataFrame in EPSG:5070."""
    df["state"] = state
    df["meta_bfe_km_per_km2"] = df["dfirm_id"].map(bfe)
    # Countywide DFIRM ids (all of Florida's) end in a letter and start with the county FIPS;
    # a community-based DFIRM keeps a null county_fips.
    last_char = df["dfirm_id"].astype(str).str[-1]
    df["county_fips"] = np.where(last_char.str.isalpha(), df["dfirm_id"].str[:5], None)
    # A band negative inherits its ring polygon's DFIRM and can lie up to 5 km into the next
    # county, so the county each point lies in is kept as well; regions are named by it.
    df["meta_point_county"] = point_counties(df, state_polys)
    if state_polys is None or "GEOID" not in state_polys.columns:
        print("  meta_point_county left null: no --state-polygons file with a GEOID column")

    # Spatial CV blocks. Every split downstream must be made on this column.
    df["block_id"] = block_ids(df.x, df.y, cfg["block_km"])
    df["point_id"] = [f"{state}{i:07d}" for i in range(len(df))]

    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=CRS_WORK)
    lonlat = points.geometry.to_crs(CRS_SRC)
    points["lon"] = lonlat.x
    points["lat"] = lonlat.y
    points = points.drop(columns=["x", "y"])

    rest = [c for c in points.columns if c not in LEAD_COLUMNS and c != "geometry"]
    return points[LEAD_COLUMNS + rest + ["geometry"]]


# =========================================================================
# per-state build
# =========================================================================
def parse_protocols(text: str) -> list[str]:
    protocols = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [p for p in protocols if p not in PROTOCOLS]
    if not protocols or unknown:
        sys.exit(f"--protocols must be a non-empty subset of {PROTOCOLS}, got {text!r}")
    return protocols


def read_panels(state: str):
    """The state's FIRM panels in EPSG:5070, or None when there is no panel file."""
    path = IN_DIR / f"pan_{state}.parquet"
    if not path.exists():
        return None
    return gpd.read_parquet(path).to_crs(CRS_WORK)


def read_state_polygons(path_text: str):
    """The --state-polygons file in EPSG:5070 (parquet, or anything geopandas reads), or None."""
    if not path_text:
        return None
    path = Path(path_text)
    if path.suffix == ".parquet":
        polys = gpd.read_parquet(path)
    else:
        polys = gpd.read_file(path)
    return polys.to_crs(CRS_WORK)


def read_unassessed(state: str, idx: pd.DataFrame) -> np.ndarray:
    """Geometry of the state's Zone D and AREA NOT INCLUDED polygons."""
    rows = idx[idx.FLD_ZONE.isin(list(UNASSESSED_ZONES))]
    geoms = read_geoms(state, rows["row_id"].to_numpy())
    return geoms[_is_present(geoms)]


def read_open_water(state: str, idx: pd.DataFrame) -> np.ndarray:
    """Geometry of the state's NFHL OPEN WATER polygons."""
    rows = idx[idx.FLD_ZONE == OPEN_WATER_ZONE]
    geoms = read_geoms(state, rows["row_id"].to_numpy())
    return geoms[_is_present(geoms)]


def sample_positives(state, domain, zone_idx, cfg, rng):
    """A domain's positives: exactly cfg["candidates"] candidates uniform by area over the land of
    the zone's polygons (area_uniform_candidates, no polygon cap), thinned to min_sep_m with every
    survivor kept.

    Returns (frame, src, geoms, counts), where src (index rows) and geoms are the polygons behind
    the candidates, which seed the band's rings, or None.
    """
    if len(zone_idx) == 0:
        print(f"  [{domain}] no polygons")
        return None
    n = cfg["candidates"]
    zone_idx = zone_idx.reset_index(drop=True)
    cand, counts, cache = area_uniform_candidates(state, zone_idx, n, cfg, rng, "positive")
    thinned = grid_thin(cand["x"], cand["y"], cfg["min_sep_m"], rng)
    counts["after_group_thinning"] = len(thinned)
    owner = cand["owner"][thinned]
    frame = _tag(_attr_frame(zone_idx.iloc[owner], cand["x"][thinned], cand["y"][thinned],
                             np.zeros(len(thinned))), domain, 1, pd.NA)
    if "mask_lc" in cand:
        frame["meta_mask_lc"] = cand["mask_lc"][thinned]

    used = np.unique(cand["owner"])
    src = zone_idx.iloc[used].reset_index(drop=True)
    geoms = np.array([cache[int(r)] for r in src["row_id"]], dtype=object)
    counts["polygons"] = len(used)
    print(f"  [{domain}] positives: {counts['proposed']:,} proposed in {counts['batches']} "
          f"batches, {_counts_text(counts, POSITIVE_LABELS)}; {counts['valid']:,} valid, "
          f"{n:,} candidates from {len(used):,} polygons, {src.DFIRM_ID.nunique()} DFIRMs; "
          f"{len(thinned):,} after thinning ({100 * len(thinned) / n:.1f}%)", flush=True)
    return frame, src, geoms, counts


POSITIVE_LABELS = {"not_on_land": "not on land per the mask", "unplaced": "never placed",
                   "no_geometry": "without geometry"}


def separate_whole_state(df: pd.DataFrame, min_sep_m: float, rng):
    """One thinning pass over every point, so no two are closer than min_sep_m whatever their
    class or zone. Returns the kept rows (in frame order) and the stratum sizes before the pass.

    A band negative is already band_min_m from every SFHA, so what this pass separates is A from
    AE (positives in adjacent polygons, band negatives in overlapping rings) and shadedX from the
    rest.
    """
    sizes_before = stratum_sizes(df)
    keep = grid_thin(df.x.values, df.y.values, min_sep_m, rng)
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    sizes_after = stratum_sizes(df).reindex(sizes_before.index, fill_value=0)
    lost = sizes_before - sizes_after
    parts = []
    for (domain, label, protocol), n in lost.items():
        parts.append(f"{domain}/{label}/{protocol} {n:,}")
    print("  whole-state separation dropped " + ", ".join(parts))
    return df, sizes_before


def group_table(groups: dict) -> pd.DataFrame:
    """One row per group: candidates, survivors of the group's own thinning, survivors of the
    whole-state pass (kept), the candidates proposed, and the share kept."""
    rows = []
    for key, c in groups.items():
        rows.append(dict(group=key, proposed=c.get("proposed"), candidates=c["candidates"],
                         after_group_thinning=c["after_group_thinning"], kept=c["kept"],
                         kept_share=round(c["kept"] / c["candidates"], 4)))
    table = pd.DataFrame(rows)
    total = dict(group="total", proposed=table.proposed.sum(), candidates=table.candidates.sum(),
                 after_group_thinning=table.after_group_thinning.sum(), kept=table.kept.sum())
    total["kept_share"] = round(total["kept"] / total["candidates"], 4)
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def write_points(points: gpd.GeoDataFrame, path: Path, cfg: dict,
                 groups: dict | None = None) -> None:
    """Write the points with the build settings and the per-group counts in the parquet
    metadata, so --audit tests what was built."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(path)
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    metadata[CONFIG_KEY] = json.dumps(cfg, sort_keys=True).encode()
    if groups is not None:
        metadata[COUNTS_KEY] = json.dumps(groups, sort_keys=True).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def build_state(state: str, cfg: dict):
    """Sample one state's points and write points_{state}.parquet; return them, or None.

    Raises SupplyError when a group cannot reach cfg["candidates"]; nothing is written then."""
    t0 = time.time()
    idx = load_index(state)
    rng = np.random.default_rng(cfg["seed"] + sum(map(ord, state)))
    protocols = parse_protocols(cfg["protocols"])

    sfha_idx = idx[idx.SFHA_TF == "T"]
    pos_idx = sfha_idx[sfha_idx.FLD_ZONE.isin(["A", "AE"])]
    if len(pos_idx) == 0:
        print(f"[{state}] no A/AE polygons -- skipping")
        return None
    subtype = idx["ZONE_SUBTY"].fillna("").astype(str).str.upper()
    shaded_idx = idx[(idx.FLD_ZONE == "X") & subtype.str.contains("0.2 PCT")]

    pan = read_panels(state)
    pan_tree = None
    if pan is not None and len(pan):
        pan_tree = STRtree(np.asarray(pan.geometry.values))
    state_polys = read_state_polygons(cfg.get("state_polygons"))
    state_tree = None
    if state_polys is not None:
        state_tree = STRtree(np.asarray(state_polys.geometry.values))
        print(f"  band negatives must also lie inside {len(state_polys):,} polygons "
              f"from {Path(cfg['state_polygons'])}")
    unassessed_tree = None
    if "band" in protocols:
        unassessed = read_unassessed(state, idx)
        if len(unassessed):
            unassessed_tree = STRtree(unassessed)
        print(f"  band negatives must also lie outside {len(unassessed):,} "
              f"{' / '.join(UNASSESSED_ZONES)} polygons")
    # The samplers read the mask and the OPEN WATER polygons from their cfg, next to the
    # settings; cfg itself keeps only what the points file stores (the path and its SHA-256).
    sample_cfg = cfg
    if cfg.get("land_mask"):
        land = LandMask(cfg["land_mask"])
        cfg = dict(cfg, land_mask_sha256=land.sha256)
        open_water = read_open_water(state, idx)
        open_water_tree = STRtree(open_water) if len(open_water) else None
        sample_cfg = dict(cfg, land=land, open_water_tree=open_water_tree)
        print(f"  every point must lie on land per {land.describe()}; band negatives also "
              f"outside {len(open_water):,} {OPEN_WATER_ZONE} polygons")

    n_panels = 0 if pan is None else len(pan)
    print(f"\n[{state}] A={int((pos_idx.FLD_ZONE == 'A').sum()):,}  "
          f"AE={int((pos_idx.FLD_ZONE == 'AE').sum()):,}  "
          f"shadedX={len(shaded_idx):,}  panels={n_panels:,}; "
          f"{cfg['candidates']:,} candidates per group", flush=True)
    bfe = bfe_density_map(state, sfha_idx)
    if bfe:
        print(f"  BFE density for {len(bfe)} DFIRMs")

    frames, groups = [], {}
    for domain in ("A", "AE"):
        positives = sample_positives(state, domain, pos_idx[pos_idx.FLD_ZONE == domain],
                                     sample_cfg, rng)
        if positives is None:
            continue
        pos, src, geoms, counts = positives
        frames.append(pos)
        groups[group_key(domain, 1, "-")] = counts

        if "band" in protocols:
            band = sample_band_negatives(state, src, geoms, sfha_idx, pan_tree,
                                         cfg["candidates"], sample_cfg, rng, state_tree,
                                         unassessed_tree)
            if band is not None:
                frame, counts = band
                frames.append(_tag(frame, domain, 0, "band"))
                groups[group_key(domain, 0, "band")] = counts

        if "shadedX" in protocols:
            shaded = sample_shaded_x(state, domain, shaded_idx, src, sfha_idx,
                                     cfg["candidates"], sample_cfg, rng)
            if shaded is not None:
                frame, counts = shaded
                frames.append(frame)
                groups[group_key(domain, 0, "shadedX")] = counts

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df, _ = separate_whole_state(df, cfg["min_sep_m"], rng)
    kept = stratum_sizes(df)
    for key, counts in groups.items():
        domain, label, protocol = key.split("/")
        counts["kept"] = int(kept.get((domain, int(label), protocol), 0))
    print("  candidates and survivors per group:\n"
          + _indented(group_table(groups).to_string(index=False)), flush=True)

    df = attach_panel_meta(df, pan)
    points = add_point_columns(df, state, cfg, bfe, state_polys)
    out = OUT_DIR / f"points_{state}.parquet"
    write_points(points, out, cfg, groups)
    print(f"[{state}] wrote {len(points):,} points, {points.block_id.nunique():,} blocks "
          f"-> {out}  ({time.time()-t0:.0f}s)")
    return points


def build_config(path: Path) -> dict | None:
    """The settings stored with a build, or None for a build that predates them."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(CONFIG_KEY)
    return json.loads(raw) if raw else None


def build_counts(path: Path) -> dict | None:
    """The per-group counts stored with a build ("A/1/-" -> candidates, after_group_thinning,
    kept, ...), or None for a build that predates them."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(COUNTS_KEY)
    return json.loads(raw) if raw else None


# =========================================================================
# audit and report
# =========================================================================
def audit(states: list[str], cfg: dict) -> None:
    """Check the invariants the design depends on, on point files already written."""
    for state in states:
        path = OUT_DIR / f"points_{state}.parquet"
        if not path.exists():
            print(f"[{state}] not built")
            continue
        audit_points(state, path, cfg)


def audit_points(state: str, path: Path, cfg: dict) -> None:
    """Print the invariant checks for one points file."""
    points = gpd.read_parquet(path)
    built = build_config(path)
    settings = built or cfg     # a build without stored settings is tested against cfg
    print(f"\n=== {state}: {len(points):,} points ===")
    print(f"  CRS {points.crs.to_epsg()} (want {CRS_WORK})")
    if built:
        print("  built with: " + json.dumps(built, sort_keys=True))
    else:
        print("  built with: no stored config (pre-v2 build); testing the defaults")

    strata = ["domain", "label", points.neg_protocol.fillna("-")]
    counts = points.groupby(strata).size()
    print("  strata:\n" + _indented(counts.to_string()))
    if built:
        _audit_groups(counts, settings, build_counts(path))

    _audit_spacing(points, strata, settings["min_sep_m"])

    pos = points[points.label == 1]
    print(f"  positives at distance 0 from SFHA: "
          f"{(pos.meta_dist_to_sfha_m == 0).mean()*100:.1f}% (want 100)")
    band = points[points.neg_protocol == "band"]
    if len(band):
        _audit_band(state, band, settings)
    # The mask the build used; --land-mask tests a build made without one.
    mask_path = settings.get("land_mask") or cfg.get("land_mask")
    if mask_path:
        _audit_land(state, points, mask_path, settings.get("land_mask_sha256"))

    # Builds from before meta_src_row_id identify a polygon by DFIRM and FLD_AR_ID.
    if "meta_src_row_id" in points.columns:
        polygon_key = ["meta_src_row_id"]
    else:
        polygon_key = ["dfirm_id", "meta_fld_ar_id"]
    for domain, domain_pos in pos.groupby("domain"):
        per_polygon = domain_pos.groupby(polygon_key).size()
        share = per_polygon.sort_values(ascending=False) / len(domain_pos)
        print(f"  [{domain}] positives from {len(share):,} polygons; largest "
              f"{100*share.iloc[0]:.2f}%, top 10 {100*share.iloc[:10].sum():.1f}% "
              f"(uniform by area, no polygon cap)")
    shaded = points[points.neg_protocol == "shadedX"]
    if len(shaded):
        print(f"  shadedX negatives inside an SFHA: "
              f"{int((shaded.meta_dist_to_sfha_m <= 0).sum())} (must be 0)")

    n_meta = len([col for col in points.columns if col.startswith(META_PREFIX)])
    print(f"  meta_ columns excluded from features: {n_meta}")
    print(f"  blocks: {points.block_id.nunique():,} "
          f"(median {points.groupby('block_id').size().median():.0f} points/block)")
    regime_share = pos.groupby("domain")["meta_study_regime"].value_counts(normalize=True)
    regimes = (regime_share * 100).round(1).unstack(fill_value=0)
    print("  positive study regimes (%):\n" + _indented(regimes.to_string()))


def group_problems(counts: pd.Series, settings: dict, groups: dict | None) -> list[str]:
    """How a build's groups break the candidate design (D1): every group of the stored counts
    drew exactly settings["candidates"] candidates, kept no more than its own thinning left and
    that no more than its candidates, and the file holds exactly the stored kept counts and no
    other group. counts are the file's rows per (domain, label, protocol)."""
    if groups is None:
        return ["no candidate counts stored with the build"]
    n = settings["candidates"]
    bad = []
    in_file = {group_key(d, label, p): int(v) for (d, label, p), v in counts.items()}
    for key in sorted(set(in_file) - set(groups)):
        bad.append(f"{key}: {in_file[key]:,} rows but no stored counts")
    for key, c in sorted(groups.items()):
        if c["candidates"] != n:
            bad.append(f"{key}: {c['candidates']:,} candidates, not {n:,}")
        if not c["kept"] <= c["after_group_thinning"] <= c["candidates"]:
            bad.append(f"{key}: kept {c['kept']:,} > thinned {c['after_group_thinning']:,} "
                       f"or thinned > candidates")
        if in_file.get(key, 0) != c["kept"]:
            bad.append(f"{key}: {in_file.get(key, 0):,} rows in the file, {c['kept']:,} stored")
    return bad


def _audit_groups(counts: pd.Series, settings: dict, groups: dict | None) -> None:
    """Print each group's candidates and survivors and whether they follow the design."""
    if groups is not None:
        print("  candidates and survivors:\n"
              + _indented(group_table(groups).to_string(index=False)))
    bad = group_problems(counts, settings, groups)
    print(f"  {settings['candidates']:,} candidates per group, every survivor kept: "
          f"{'YES' if not bad else 'NO  <-- ' + '; '.join(bad[:3])}")


def _audit_spacing(points: gpd.GeoDataFrame, strata: list, sep: float) -> None:
    """Every pair closer than sep, tested exactly in the working CRS, and nearest neighbours."""
    pts = shapely.points(points.geometry.x.values, points.geometry.y.values)
    tree = STRtree(pts)
    pairs = tree.query(pts, predicate="dwithin", distance=sep)
    pairs = pairs[:, pairs[0] < pairs[1]]
    pair_dist = shapely.distance(pts[pairs[0]], pts[pairs[1]])
    close = pair_dist < sep
    stratum = points.groupby(strata).ngroup().values
    same_stratum = int((stratum[pairs[0]] == stratum[pairs[1]])[close].sum())
    # exclusive=True skips geometries equal to the query, so exact duplicates are folded
    # back in from the pair distances.
    nearest_idx, nearest_dist = tree.query_nearest(pts, exclusive=True, return_distance=True)
    nearest = np.full(len(pts), np.inf)
    np.minimum.at(nearest, nearest_idx[0], nearest_dist)
    if close.any():
        np.minimum.at(nearest, pairs[0], pair_dist)
        np.minimum.at(nearest, pairs[1], pair_dist)
    print(f"  pairs closer than {sep:.0f} m: {int(close.sum()):,} "
          f"({same_stratum:,} within one stratum); minimum nearest-neighbour distance "
          f"{nearest.min():.3f} m, median {np.median(nearest):.0f} m")


def _audit_band(state: str, band: gpd.GeoDataFrame, settings: dict) -> None:
    """Band negatives: distance range, inside a FIRM panel, outside the unassessed zones."""
    band_min = settings["band_min_m"]
    band_max = settings["band_max_m"]
    dist = band.meta_dist_to_sfha_m
    off = int(((dist < band_min) | (dist > band_max)).sum())
    print(f"  band negatives distance range: {dist.min():.0f}-{dist.max():.0f} m "
          f"(want {band_min:.0f}-{band_max:.0f}; {off} outside)")
    band_pts = np.asarray(band.geometry.values)
    pan = read_panels(state)
    if pan is not None:
        inside = _query_mask(STRtree(np.asarray(pan.geometry.values)), band_pts, "within")
        print(f"  band negatives inside a FIRM panel: {int(inside.sum()):,}"
              f"/{len(band):,}")
    if (PREP_DIR / f"idx_{state}.parquet").exists():
        unassessed = read_unassessed(state, load_index(state))
        n_in = 0
        if len(unassessed):
            n_in = int(_query_mask(STRtree(unassessed), band_pts, "intersects").sum())
        print(f"  band negatives in {' or '.join(UNASSESSED_ZONES)}: "
              f"{n_in:,} (must be 0)")


def _audit_land(state: str, points: gpd.GeoDataFrame, mask_path: str, built_sha256) -> None:
    """Every point on land per the mask, and no band negative in an NFHL OPEN WATER polygon."""
    land = LandMask(mask_path)
    if built_sha256 and built_sha256 != land.sha256:
        print(f"  WARNING: {mask_path} is not the mask the build used (SHA-256 differs)")
    classes = land.classes_at(points.geometry.x.values, points.geometry.y.values)
    off_land = int((~LandMask.is_land(classes)).sum())
    print(f"  points not on land per the mask: {off_land:,} (must be 0)")
    band = points[points.neg_protocol == "band"]
    if len(band) and (PREP_DIR / f"idx_{state}.parquet").exists():
        open_water = read_open_water(state, load_index(state))
        n_in = 0
        if len(open_water):
            band_pts = np.asarray(band.geometry.values)
            n_in = int(_query_mask(STRtree(open_water), band_pts, "intersects").sum())
        print(f"  band negatives in NFHL {OPEN_WATER_ZONE}: {n_in:,} (must be 0)")


def report() -> None:
    """Points per state, domain, label and protocol over every points file written."""
    files = sorted(OUT_DIR.glob("points_*.parquet"))
    if not files:
        print(f"nothing in {OUT_DIR}")
        return
    rows = []
    for path in files:
        points = pd.read_parquet(path, columns=["state", "domain", "label", "neg_protocol"])
        strata = ["domain", "label", points.neg_protocol.fillna("-")]
        for (domain, label, protocol), stratum in points.groupby(strata, dropna=False):
            rows.append(dict(state=points.state.iloc[0], domain=domain, label=label,
                             protocol=protocol, n=len(stratum)))
    counts = pd.DataFrame(rows)
    table = counts.pivot_table(index=["state", "domain"], columns=["label", "protocol"],
                               values="n", aggfunc="sum", fill_value=0)
    print(table.to_string())
    print(f"\ntotal points: {counts.n.sum():,}")


LAND_MASK_HELP = ("--land-mask PATH: keep every point on land per a land-cover raster from "
                  "pull_land_mask.py (0 no data, 11 open water); band negatives also stay out of "
                  "NFHL OPEN WATER polygons. Off by default. With --audit, tests a build against "
                  "PATH when the build stored no mask.")


def main(argv: list[str]) -> int:
    # --land-mask is parsed apart from the main options, which stay one per DEFAULTS key plus
    # the commands; it enters cfg only when given.
    mask_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mask_parser.add_argument("--land-mask", default="")
    mask_args, argv = mask_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description="Sample a point-level flood-label dataset from NFHL flood-zone polygons.",
        epilog=LAND_MASK_HELP)
    parser.add_argument("states", nargs="*", help="state codes (default: FL)")
    parser.add_argument("--prep", action="store_true", help="one-time repack + index")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--force", action="store_true")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    options = vars(args)
    cfg = {key: options[key] for key in DEFAULTS}
    if mask_args.land_mask:
        cfg["land_mask"] = mask_args.land_mask
    parse_protocols(cfg["protocols"])
    states = args.states or STATES
    unknown = [state for state in states if state not in STATES]
    if unknown:
        sys.exit(f"unknown states: {unknown}")

    if args.report:
        report()
        return 0
    if args.audit:
        audit(states, cfg)
        return 0
    if args.prep:
        for state in states:
            prep_state(state, force=args.force)
        return 0

    print("config:", {key: cfg[key] for key in sorted(cfg)})
    failed = []
    for state in states:
        out = OUT_DIR / f"points_{state}.parquet"
        if out.exists() and not args.force:
            print(f"[{state}] exists -- skipping (use --force to rebuild)")
            continue
        try:
            if build_state(state, cfg) is None:
                print(f"[{state}] FAILED: nothing built", file=sys.stderr)
                failed.append(state)
        except SupplyError as err:
            print(f"[{state}] FAILED: {err}", file=sys.stderr)
            failed.append(state)
        except Exception:
            print(f"[{state}] FAILED", file=sys.stderr)
            traceback.print_exc()
            failed.append(state)

    print()
    report()
    if failed:
        print(f"\nfailed: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
