#!/usr/bin/env python3
"""Sample a point-level flood-label dataset from FEMA NFHL flood-zone polygons.

Domain A draws its positives inside Zone A polygons (approximate study), domain AE inside Zone AE
polygons (detailed study). Both domains are built by one protocol and trimmed to identical stratum
sizes, so they differ only in the polygons behind them. Negatives are `band` (500-5,000 m from
every SFHA, inside a FIRM panel) or `shadedX` (inside Zone X 0.2%-chance polygons). Columns named
`meta_*` record how the map was made and are never features; every downstream split is made on
`block_id`. Protocol v2 settings and their citations: docs/PROJECT-PLAN.md section 2a.

Inputs:  data/interim/nfhl_states/{haz,pan,s_bfe}_FL.parquet
Outputs: data/interim/prep/{idx_FL,haz_FL_5070}.parquet (--prep, run once)
         data/interim/points/points_FL.parquet

    python sample_points.py --prep FL
    python sample_points.py FL --protocols band --exact --ring-oversample 6
    python sample_points.py --audit FL
"""
import argparse
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
from pyproj import Transformer
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

# Every key is also a CLI option (--per-domain, --ring-oversample, ...), in this order.
DEFAULTS = dict(
    per_domain=20000,     # positives per domain (A, AE)
    neg_ratio=1.0,        # negatives per positive, per protocol
    min_sep_m=100.0,      # floor between any two points, any class or zone; de-cluttering only
    block_km=10.0,        # spatial CV block edge; spatial autocorrelation is handled by blocks
    poly_cap_frac=0.02,   # no polygon may supply more than this share of a domain's candidates
    band_min_m=500.0,     # band negatives: nearest SFHA at least this far
    band_max_m=5000.0,    # ... and at most this far
    oversample=3.0,       # candidates drawn per point wanted, before thinning
    ring_oversample=2.0,  # band candidates = n_neg * oversample * ring_oversample
    surplus=0.03,         # extra share per stratum, absorbed by the whole-state separation pass
    protocols="band,shadedX",
    state_polygons="",    # optional polygon file (e.g. counties) band negatives must lie inside
    seed=20260918,
)
PROTOCOLS = ("band", "shadedX")
UNASSESSED_ZONES = ("D", "AREA NOT INCLUDED")   # inside FIRM panels but never studied

# Prefix of every column that records how the map was made. Never a model feature: STUDY_TYP
# alone predicts Zone A vs AE well above the base rate, and meta_dist_to_sfha_m is the distance
# to the labelled zone itself.
META_PREFIX = "meta_"

MAX_RINGS = 10_000        # distinct polygons buffered for the band (FL: ~4,400 per zone)
RING_SIMPLIFY_M = 100.0   # simplification before buffering; the exact distance filter decides

CONFIG_KEY = b"sample_points"   # parquet metadata key holding the build settings
LEAD_COLUMNS = ["point_id", "state", "county_fips", "dfirm_id", "block_id",
                "domain", "label", "neg_protocol", "lon", "lat"]


class SupplyError(RuntimeError):
    """A stratum fell short of its exact target under --exact."""


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


def sample_in_polygons(geoms: np.ndarray, weights, n: int, cap: int,
                       rng: np.random.Generator, max_rounds: int = 30):
    """n points uniform over the union of geoms (weights: their areas), at most cap per polygon.

    Returns (x, y, index into geoms).
    """
    if n <= 0 or len(geoms) == 0:
        return _no_points()
    quota = polygon_quota(weights, n, cap, rng)
    return place_points(geoms, quota, rng, max_rounds)


def polygon_quota(weights, n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Points per polygon: n draws with replacement in proportion to weight, at most cap each.

    Drawing polygons by area and placing one uniform point in each is a uniform draw over their
    union; a fixed count per polygon would over-represent small polygons. The cap keeps a few huge
    polygons from dominating; draws it cut off are redrawn once among the polygons below it.
    Needs no geometry, so it runs over a whole zone's index before any geometry is read.
    """
    weight = np.asarray(weights, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    if n <= 0 or weight.sum() <= 0:
        return np.zeros(len(weight), dtype=np.int64)

    quota = np.minimum(rng.multinomial(n, weight / weight.sum()), cap)
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
def sample_band_negatives(state, src_idx, src_geoms, sfha_idx, pan_tree, n, cfg, rng,
                          state_tree=None, unassessed_tree=None):
    """Up to n band negatives, band_min_m to band_max_m from every SFHA, as a frame (or None).

    The band is a ring around the polygons behind the domain's positives, so the protocol never
    reads FLD_ZONE and is identical for A and AE. A point counts as absence only where FEMA
    assessed the area: inside a FIRM panel (pan_tree), outside Zone D and AREA NOT INCLUDED
    (unassessed_tree) and, given state_tree, inside the state, whose NFHL lacks the neighbours'
    floodplains. The ring only bounds the candidates (the exact distance filter decides), so it
    is built from simplified polygons and widened by the simplification tolerance; its 8-segment
    corners sit at most ~25 m inside the true buffer.
    """
    if n <= 0 or len(src_geoms) == 0:
        return None

    band_min, band_max = cfg["band_min_m"], cfg["band_max_m"]
    want = int(n * cfg["oversample"] * cfg["ring_oversample"])

    ring_src = np.arange(len(src_geoms))
    flag = ""
    if len(ring_src) > MAX_RINGS:
        ring_src = rng.choice(ring_src, MAX_RINGS, replace=False)
        flag = "   <-- RING CAP BINDING: a random subset of rings defines the band"
    print(f"    band rings from {len(src_geoms):,} polygons "
          f"(cap {MAX_RINGS:,}){flag}", flush=True)
    simplified = shapely.simplify(src_geoms[ring_src], RING_SIMPLIFY_M)
    rings = shapely.buffer(simplified, band_max + RING_SIMPLIFY_M, quad_segs=8)
    # No ring may supply more than four times an even share of the candidates.
    cap = max(4, int(np.ceil(want / max(len(rings) // 4, 1))))

    x, y, ring_i = sample_in_polygons(rings, shapely.area(rings), want, cap, rng)
    if len(x) == 0:
        return None

    pts = shapely.points(x, y)
    dist = distances_to_sfha(state, pts, sfha_idx, band_max)
    too_near = dist < band_min
    too_far = dist > band_max
    ok = ~too_near & ~too_far
    # Each mask below counts only the points that passed every earlier test.
    outside_panel = np.zeros(len(pts), dtype=bool)
    if pan_tree is not None:
        inside = _query_mask(pan_tree, pts, "within")
        outside_panel = ok & ~inside
        ok &= inside
    unassessed = np.zeros(len(pts), dtype=bool)
    if unassessed_tree is not None:
        in_unassessed = _query_mask(unassessed_tree, pts, "intersects")
        unassessed = ok & in_unassessed
        ok &= ~in_unassessed
    abroad = np.zeros(len(pts), dtype=bool)
    if state_tree is not None:
        home = _query_mask(state_tree, pts, "within")
        abroad = ok & ~home
        ok &= home
    if not ok.any():
        return None

    x, y, dist, ring_i = x[ok], y[ok], dist[ok], ring_i[ok]
    thinned = grid_thin(x, y, cfg["min_sep_m"], rng)
    mask_note = ""
    if state_tree is not None:
        mask_note = f", {int(abroad.sum()):,} outside the state polygons"
    print(f"    band candidates {len(pts):,}: {int(too_near.sum()):,} < {band_min:.0f} m, "
          f"{int(too_far.sum()):,} > {band_max:.0f} m, {int(outside_panel.sum()):,} outside a "
          f"panel, {int(unassessed.sum()):,} in {' or '.join(UNASSESSED_ZONES)}"
          f"{mask_note}; {len(x):,} valid, {len(thinned):,} after thinning",
          flush=True)
    if len(thinned) < n:
        suggested = cfg["ring_oversample"] * n / max(len(thinned), 1) * 1.1
        print(f"    band supply short: {len(thinned):,}/{n:,}; try --ring-oversample "
              f"{suggested:.1f}", flush=True)
    keep = thinned[:n]
    x, y, dist, ring_i = x[keep], y[keep], dist[keep], ring_i[keep]
    src = ring_src[ring_i]
    return _blank_frame(src_idx["DFIRM_ID"].to_numpy()[src], x, y, dist,
                        src_idx["row_id"].to_numpy()[src])


def sample_shaded_x(state, domain, shaded_idx, src, sfha_idx, n_neg, cfg, rng):
    """Up to n_neg shadedX negatives, inside Zone X 0.2%-chance polygons of the DFIRMs behind
    the domain's positives. They control for whether a place was mapped at all, which the
    distance band does not. Returns a frame, or None."""
    candidates = shaded_idx[shaded_idx.DFIRM_ID.isin(set(src.DFIRM_ID))]
    if len(candidates) == 0:
        print(f"  [{domain}] negatives shadedX none in these DFIRMs")
        return None

    over = cfg["oversample"]
    quota = polygon_quota(candidates["area_m2"].to_numpy(), int(n_neg * over),
                          int(_polygon_cap(cfg) * over), rng)
    live = np.nonzero(quota > 0)[0]
    polys = candidates.iloc[live]
    quota = quota[live]
    geoms = read_geoms(state, polys["row_id"].to_numpy())
    found = _is_present(geoms)
    polys, geoms, quota = polys[found], geoms[found], quota[found]

    x, y, pidx = place_points(geoms, quota, rng)
    if len(x) == 0:
        return None
    keep = grid_thin(x, y, cfg["min_sep_m"], rng)[:n_neg]
    x, y, pidx = x[keep], y[keep], pidx[keep]
    dist = distances_to_sfha(state, shapely.points(x, y), sfha_idx, cfg["band_max_m"])
    frame = _tag(_attr_frame(polys.iloc[pidx], x, y, dist), domain, 0, "shadedX")
    print(f"  [{domain}] negatives shadedX {len(x):,}/{n_neg:,}", flush=True)
    return frame


# =========================================================================
# strata: ranking, trimming, balancing
# =========================================================================
def rank_negatives(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add meta_neg_rank: each negative's position in a random ordering within its stratum.

    This makes the negative ratio a post hoc filter: `meta_neg_rank < r * n_pos` gives ratio r,
    and the samples nest (a 1:2 set contains the 1:1 set), so two ratios differ only in how many
    negatives they hold, not which. Positives get -1. Works on a copy.
    """
    df = df.copy()
    df["meta_neg_rank"] = -1
    is_neg = df.label == 0
    if not is_neg.any():
        return df
    df["_order"] = rng.permutation(len(df))     # a random position for every row
    df.loc[is_neg, "meta_neg_rank"] = _rank_within_strata(df[is_neg], "_order")
    return df.drop(columns="_order")


def dense_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber meta_neg_rank 0..n-1 within each stratum after rows were dropped (in place)."""
    is_neg = df.label == 0
    if is_neg.any():
        df.loc[is_neg, "meta_neg_rank"] = _rank_within_strata(df[is_neg], "meta_neg_rank")
    return df


def _rank_within_strata(neg: pd.DataFrame, column: str) -> pd.Series:
    """0-based rank of `column` within each (domain, neg_protocol) stratum of the negatives."""
    strata = [neg["domain"], neg["neg_protocol"].fillna("-")]
    ranks = neg.groupby(strata)[column].rank(method="first")
    return ranks.astype(int) - 1


def achieved_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Negatives actually obtained per positive, per stratum."""
    pos = df[df.label == 1].groupby("domain").size()
    rows = []
    for (domain, protocol), negs in df[df.label == 0].groupby(
            ["domain", df.neg_protocol.fillna("-")]):
        n_pos = int(pos.get(domain, 0))
        rows.append(dict(domain=domain, protocol=protocol, pos=n_pos, neg=len(negs),
                         ratio=round(len(negs) / max(n_pos, 1), 2)))
    return pd.DataFrame(rows)


def trim_to_targets(df: pd.DataFrame, cfg: dict, protocols: list[str]):
    """Cut every stratum to its target; return (df, shortfalls).

    Positives keep the first per_domain rows of each domain in frame order, which is grid_thin's
    random visit order, so the cut is a uniform subsample. Each requested negative protocol keeps
    the round(n_pos * neg_ratio) rows with the lowest meta_neg_rank, n_pos being the positives its
    domain kept. A short stratum is reported as (domain, label, protocol, have, want); positives
    are never trimmed to reach a ratio.
    """
    keep = []
    short = []
    for domain in ("A", "AE"):
        rows = df[df.domain == domain]
        pos = rows[rows.label == 1]
        if len(pos) < cfg["per_domain"]:
            short.append((domain, 1, "-", len(pos), cfg["per_domain"]))
        pos = pos.iloc[:cfg["per_domain"]]
        keep.append(pos)
        target = int(round(len(pos) * cfg["neg_ratio"]))
        for protocol in protocols:
            negs = rows[(rows.label == 0) & (rows.neg_protocol.fillna("-") == protocol)]
            if len(negs) < target:
                short.append((domain, 0, protocol, len(negs), target))
            keep.append(negs.nsmallest(target, "meta_neg_rank"))
    return pd.concat(keep, ignore_index=True), short


def balance_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Trim each (label, protocol) stratum to the smaller domain's size, so A and AE match.

    A transfer gap between unequal datasets would be uninterpretable.
    """
    df = df.copy()
    df["_p"] = df["neg_protocol"].fillna("-")
    sizes = df.groupby(["label", "_p", "domain"]).size().unstack("domain")
    if sizes.shape[1] < 2:
        return df.drop(columns="_p")
    target = sizes.min(axis=1)
    parts = []
    for (label, protocol), stratum in df.groupby(["label", "_p"]):
        n = int(target.get((label, protocol), 0))
        for _, rows in stratum.groupby("domain"):
            parts.append(rows.iloc[:n])
    out = pd.concat(parts, ignore_index=True).drop(columns="_p")
    if len(out) != len(df):
        print(f"  balanced A/AE: {len(df):,} -> {len(out):,}")
    return out


def stratum_sizes(df: pd.DataFrame) -> pd.Series:
    """Rows per (domain, label, neg_protocol); positives' protocol shows as '-'."""
    return df.fillna({"neg_protocol": "-"}).groupby(["domain", "label", "neg_protocol"]).size()


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


def _polygon_cap(cfg: dict) -> int:
    """Most points one polygon may supply to a domain, before oversampling."""
    return max(1, int(cfg["poly_cap_frac"] * cfg["per_domain"]))


def _with_surplus(n: int, surplus: float) -> int:
    """n plus the extra share drawn so the whole-state pass cannot leave a stratum short."""
    return n + int(np.ceil(n * surplus))


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


def sample_positives(state, domain, zone_idx, cfg, rng):
    """A domain's positives, thinned to min_sep_m, kept up to per_domain plus the surplus.

    Quotas are drawn over every polygon of the zone, and geometry is read only for the polygons
    that drew a point (about 9,000 per zone in Florida). Returns (frame, src, geoms, pidx), where
    src and geoms are those polygons and pidx maps each kept point to them, or None.
    """
    if len(zone_idx) == 0:
        print(f"  [{domain}] no polygons")
        return None

    n_target = cfg["per_domain"]
    over = cfg["oversample"]
    quota = polygon_quota(zone_idx["area_m2"].to_numpy(), int(n_target * over),
                          int(_polygon_cap(cfg) * over), rng)
    live = np.nonzero(quota > 0)[0]
    src = zone_idx.iloc[live]
    quota = quota[live]
    geoms = read_geoms(state, src["row_id"].to_numpy())
    found = _is_present(geoms)
    src, geoms, quota = src[found], geoms[found], quota[found]
    if len(geoms) == 0:
        print(f"  [{domain}] no geometry read")
        return None

    x, y, pidx = place_points(geoms, quota, rng)
    if len(x) == 0:
        print(f"  [{domain}] sampling produced nothing")
        return None
    drawn = len(x)
    thinned = grid_thin(x, y, cfg["min_sep_m"], rng)
    keep = thinned[:_with_surplus(n_target, cfg["surplus"])]
    x, y, pidx = x[keep], y[keep], pidx[keep]

    frame = _tag(_attr_frame(src.iloc[pidx], x, y, np.zeros(len(x))), domain, 1, pd.NA)
    supply_flag = "" if len(x) >= n_target else "   <-- SUPPLY-LIMITED"
    print(f"  [{domain}] positives {len(x):,} kept for {n_target:,} from "
          f"{len(src):,} polygons, {src.iloc[pidx].DFIRM_ID.nunique()} DFIRMs "
          f"(thinning kept {100*len(thinned)/drawn:.0f}% of {drawn:,}){supply_flag}",
          flush=True)
    return frame, src, geoms, pidx


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


def trim_and_balance(df: pd.DataFrame, cfg: dict, protocols: list[str],
                     sizes_before: pd.Series, state: str) -> pd.DataFrame:
    """Cut the strata to their targets, report each shortfall with its remedy, match A to AE.

    Raises SupplyError under --exact when any stratum is short, so a short build is never written.
    """
    n_before = len(df)
    df, short = trim_to_targets(df, cfg, protocols)
    print(f"  trimmed surplus to targets: {n_before:,} -> {len(df):,}")
    remedies = set()
    for domain, label, protocol, have, want in short:
        # Enough rows before the whole-state pass means the pass ate the surplus;
        # otherwise the stratum never had the candidates.
        if int(sizes_before.get((domain, label, protocol), 0)) >= want:
            why = "lost to the whole-state pass"
            fix = "--surplus"
        elif label == 1:
            why = "supply"
            fix = "--oversample"
        else:
            why = "supply"
            fix = "--ring-oversample"
        remedies.add(fix)
        what = "positives" if label == 1 else protocol
        print(f"  [{domain}] {what} {have:,}/{want:,}   <-- SHORT ({why}: raise {fix})")
    if short and cfg.get("exact"):
        raise SupplyError(
            f"[{state}] {len(short)} strata short of the exact targets; nothing "
            f"written. Raise {' and '.join(sorted(remedies))}.")
    return dense_ranks(balance_domains(df))


def print_achieved_ratios(df: pd.DataFrame, neg_ratio: float) -> None:
    """Print each stratum's negatives per positive, flagging those below the request."""
    ratios = achieved_ratios(df)
    short = ratios[ratios.ratio < neg_ratio - 0.01]
    print("  achieved negatives per positive:")
    for row in ratios.itertuples():
        flag = "   <-- SUPPLY-CAPPED" if row.ratio < neg_ratio - 0.01 else ""
        print(f"    {row.domain:<3} {row.protocol:<8} {row.neg:>7,}/{row.pos:<7,} "
              f"= {row.ratio:.2f}{flag}")
    if len(short):
        print(f"  {len(short)}/{len(ratios)} strata below the requested "
              f"{neg_ratio:g}:1 -- report the achieved ratio, not the request")


def write_points(points: gpd.GeoDataFrame, path: Path, cfg: dict) -> None:
    """Write the points with the build settings in the parquet metadata, so --audit tests what
    was built."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(path)
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    metadata[CONFIG_KEY] = json.dumps(cfg, sort_keys=True).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def build_state(state: str, cfg: dict):
    """Sample one state's points and write points_{state}.parquet; return them, or None."""
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

    n_panels = 0 if pan is None else len(pan)
    print(f"\n[{state}] A={int((pos_idx.FLD_ZONE == 'A').sum()):,}  "
          f"AE={int((pos_idx.FLD_ZONE == 'AE').sum()):,}  "
          f"shadedX={len(shaded_idx):,}  panels={n_panels:,}", flush=True)
    bfe = bfe_density_map(state, sfha_idx)
    if bfe:
        print(f"  BFE density for {len(bfe)} DFIRMs")

    frames = []
    for domain in ("A", "AE"):
        positives = sample_positives(state, domain, pos_idx[pos_idx.FLD_ZONE == domain],
                                     cfg, rng)
        if positives is None:
            continue
        pos, src, geoms, pidx = positives
        frames.append(pos)

        # Negatives are sized to the positives this stratum will keep, and only those
        # positives' polygons seed the ring: the surplus absorbs the whole-state pass,
        # it does not widen the band.
        n_pos = min(len(pos), cfg["per_domain"])
        n_neg = _with_surplus(int(round(n_pos * cfg["neg_ratio"])), cfg["surplus"])

        if "band" in protocols:
            used = np.unique(pidx[:n_pos])
            band = sample_band_negatives(state, src.iloc[used].reset_index(drop=True),
                                         geoms[used], sfha_idx, pan_tree, n_neg, cfg, rng,
                                         state_tree, unassessed_tree)
            if band is not None and len(band):
                frames.append(_tag(band, domain, 0, "band"))
                print(f"  [{domain}] negatives band    {len(band):,}/{n_neg:,}", flush=True)

        if "shadedX" in protocols:
            shaded = sample_shaded_x(state, domain, shaded_idx, src, sfha_idx, n_neg, cfg, rng)
            if shaded is not None:
                frames.append(shaded)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    df, sizes_before = separate_whole_state(df, cfg["min_sep_m"], rng)
    df = rank_negatives(df, rng)
    df = trim_and_balance(df, cfg, protocols, sizes_before, state)
    print_achieved_ratios(df, cfg["neg_ratio"])

    df = attach_panel_meta(df, pan)
    points = add_point_columns(df, state, cfg, bfe, state_polys)
    out = OUT_DIR / f"points_{state}.parquet"
    write_points(points, out, cfg)
    print(f"[{state}] wrote {len(points):,} points, {points.block_id.nunique():,} blocks "
          f"-> {out}  ({time.time()-t0:.0f}s)")
    return points


def build_config(path: Path) -> dict | None:
    """The settings stored with a build, or None for a build that predates them."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(CONFIG_KEY)
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
    if {"A", "AE"} <= set(points.domain.unique()):
        a = counts.xs("A", level=0)
        ae = counts.xs("AE", level=0)
        a_on_ae = a.reindex(ae.index).fillna(0).astype(int)
        same = a_on_ae.equals(ae.astype(int))
        print(f"  A/AE strata identical: {'YES' if same else 'NO  <-- BROKEN'}")
    if built:
        _audit_exact_targets(counts, settings)

    _audit_spacing(points, strata, settings["min_sep_m"])

    pos = points[points.label == 1]
    print(f"  positives at distance 0 from SFHA: "
          f"{(pos.meta_dist_to_sfha_m == 0).mean()*100:.1f}% (want 100)")
    band = points[points.neg_protocol == "band"]
    if len(band):
        _audit_band(state, band, settings)

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
              f"(cap {100*settings['poly_cap_frac']:.0f}% applies to candidates)")
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


def _audit_exact_targets(counts: pd.Series, settings: dict) -> None:
    """Whether every stratum holds exactly its target: per_domain positives, and
    round(per_domain * neg_ratio) negatives per protocol, in each domain."""
    n_pos = settings["per_domain"]
    n_neg = int(round(n_pos * settings["neg_ratio"]))
    want = {}
    for domain in ("A", "AE"):
        want[(domain, 1, "-")] = n_pos
    for domain in ("A", "AE"):
        for protocol in parse_protocols(settings["protocols"]):
            want[(domain, 0, protocol)] = n_neg
    got = {key: int(counts.get(key, 0)) for key in want}
    ok = got == want and len(counts) == len(want)
    print(f"  exact targets ({n_pos:,} positives, {n_neg:,} per "
          f"negative protocol, per domain): {'YES' if ok else 'NO'}")


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Sample a point-level flood-label dataset from NFHL flood-zone polygons.")
    parser.add_argument("states", nargs="*", help="state codes (default: FL)")
    parser.add_argument("--prep", action="store_true", help="one-time repack + index")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--exact", action="store_true",
                        help="fail, writing nothing, unless every stratum reaches "
                             "per_domain positives and per_domain*neg_ratio negatives")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    options = vars(args)
    cfg = {key: options[key] for key in DEFAULTS}
    cfg["exact"] = args.exact
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
            if build_state(state, cfg) is None and cfg["exact"]:
                print(f"[{state}] FAILED: nothing built under --exact", file=sys.stderr)
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
