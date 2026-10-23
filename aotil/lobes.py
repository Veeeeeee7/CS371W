#!/usr/bin/env python3
"""
Lobes and their 30 m edge vertices: the flood-zone boundaries the boundary analysis classifies
(docs/REGION-ANALYSIS-FLORIDA.md section 3.3 steps 1-2, docs/TASK-DATA-AUDIT.md D2).

A lobe is one single-part NFHL flood-hazard polygon, taken whole: clipping would add a straight
edge no map drew. Each region component of >= --min-size points (aotil/regions.py) gets three roles:
  region     Zone A parts of the positives' source polygons and of polygons within --near-m of
             a region point, kept where they meet the points' hull buffered by --hull-buffer-m
  AE         Zone AE parts in the component's 50 km tiles, the detailed-study contrast
  matched_A  Zone A parts in those tiles that hold no selected point, matched on perimeter
Only AE->A cells (OOD = Zone A). Outputs in --out, prefix <cell>_<variant>_S<S> from --regions:
  <prefix>_lobes.parquet     GeoParquet in EPSG:5070, one row per (component, role, lobe)
  <prefix>_vertices.parquet  one row per boundary vertex, with its inward normal and contacts
  <prefix>_bboxes.parquet    one row per component: the analysis bbox later layers are pulled for

    python aotil/lobes.py \\
        --regions results/<date>/regions/FL_band_AE_s0_spatial_mu3_knn10_S500_points.parquet \\
        --dataset data/processed/dataset_FL.parquet --out results/<date>/boundary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely import STRtree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # sample_points
import sample_points as sp  # noqa: E402

PREP_DIR = Path("data/interim/prep")
CRS_WORK = 5070            # every distance, area and perimeter here is in it
CRS_LONLAT_IN = 4269       # the dataset's lon/lat (sample_points.CRS_SRC)
CRS_LONLAT_OUT = 4326      # what the vertex and bbox tables carry
TILE_M = 50_000.0          # tile_50km edge
LONLAT_TOL = 1e-9          # degrees; regions lon/lat are copies of the dataset's

REGION_ZONE, CONTRAST_ZONE = "A", "AE"     # AE->A: OOD = Zone A
ROLES = ("region", "matched_A", "AE")
# NFHL codes a floodway as Zone AE with a ZONE_SUBTY containing this ('FLOODWAY', 'ADMINISTRATIVE
# FLOODWAY', ...). Floodway lobes stay in the AE role, flagged by is_floodway.
FLOODWAY = "FLOODWAY"

# Inward-normal check: probes PROBE_M either side of a vertex. It is asserted only on "clear"
# vertices, whose probes both lie >= CLEAR_FRAC * PROBE_M from the lobe's boundary (the edge runs
# straight past the vertex at the 1 m scale). An orientation error flips whole rings, far beyond
# both limits; sub-metre zig-zags in lidar-traced edges explain the few clear failures.
PROBE_M = 1.0
CLEAR_FRAC = 0.9
CHECK_MIN_PASS = 0.995
CHECK_MAX_FLIPPED = 0.005

REGION_COLS = ["point_id", "lon", "lat", "label", "component", "tile_50km",
               "meta_fld_ar_id", "dfirm_id"]
# part: index of the part in its polygon; matched_to: the region lobe_id a matched_A lobe was
# matched to; tiles: ';'-joined tiles the lobe meets; n_points: this component's selected points
# it covers; from_positive: a region lobe of a positive's source polygon; match_dlogp:
# |log P_matched - log P_region|; zone_subty: NFHL ZONE_SUBTY, blank -> null.
LOBE_COLS = ["lobe_id", "zone", "role", "component", "src_row_id", "fld_ar_id",
             "dfirm_id", "area_m2", "perimeter_m", "matched_to", "part", "tiles",
             "n_points", "from_positive", "match_dlogp", "zone_subty", "is_floodway"]
# ring: 0 = exterior, k = k-th hole; s: metres along the ring from its first coordinate;
# x, y in EPSG:5070, lon, lat in EPSG:4326; nx, ny: the unit inward normal.
VERTEX_COLS = ["lobe_id", "zone", "role", "component", "ring", "s", "x", "y",
               "lon", "lat", "nx", "ny"]
# See contacts(). contact_row_id is an idx row_id.
CONTACT_COLS = ["contact_row_id", "contact_zone", "contact_subty", "same_role_contact"]
IDX_COLS = ["row_id", "DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "ZONE_SUBTY", "SFHA_TF",
            "minx", "miny", "maxx", "maxy"]


# ----------------------------------------------------------------- inputs
def read_regions(points_p: Path, comps_p: Path, min_size: int):
    """The regions table, checked against its components table, and the ids of the components
    of >= min_size points."""
    pts = pd.read_parquet(points_p)
    missing = [c for c in REGION_COLS if c not in pts]
    if missing:
        raise SystemExit(f"{points_p}: missing columns {missing}")
    if "meta_src_row_id" not in pts:
        pts["meta_src_row_id"] = pd.array([pd.NA] * len(pts), dtype="Int64")
    pts = pts.reset_index(drop=True)
    sizes = pts.groupby("component").size()
    if not comps_p.exists():
        raise SystemExit(f"{comps_p} missing (aotil/regions.py writes it next to the points)")
    want = pd.read_csv(comps_p).set_index("component").n_points
    same_sizes = want.reindex(sizes.index).eq(sizes).all()
    if not same_sizes or len(want) != len(sizes):
        raise SystemExit(f"{points_p.name} and {comps_p.name} disagree on component sizes")
    big = [int(c) for c in sizes.index[sizes >= min_size]]
    return pts, big


def geometry_crs_id(pf: pq.ParquetFile) -> dict:
    """The CRS id GeoParquet metadata states for the geometry column, e.g.
    {"authority": "EPSG", "code": 5070}; {} when it states none."""
    metadata = pf.schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b"geo") or b"{}")
    crs = geo.get("columns", {}).get("geometry", {}).get("crs") or {}
    return crs.get("id", {})


def point_xy(pts: pd.DataFrame, dataset: Path | None) -> tuple[np.ndarray, str]:
    """(x, y) in EPSG:5070 of every selected point: the dataset's own geometry (the coordinates
    the sampler drew) when --dataset is given, else the regions lon/lat projected."""
    if dataset is None:
        tf = Transformer.from_crs(CRS_LONLAT_IN, CRS_WORK, always_xy=True)
        x, y = tf.transform(pts.lon.to_numpy(float), pts.lat.to_numpy(float))
        note = f"regions lon/lat projected EPSG:{CRS_LONLAT_IN} -> {CRS_WORK}"
        return np.column_stack([x, y]), note
    crs = geometry_crs_id(pq.ParquetFile(dataset))
    if not (crs.get("authority") == "EPSG" and int(crs.get("code", -1)) == CRS_WORK):
        raise SystemExit(f"{dataset}: geometry CRS is {crs or 'unstated'}, not EPSG:{CRS_WORK}")
    data = pq.read_table(dataset, columns=["point_id", "lon", "lat", "geometry"]).to_pandas()
    by_id = data.set_index("point_id")
    absent = ~pts.point_id.isin(by_id.index)
    if absent.any():
        raise SystemExit(f"{int(absent.sum())} selected point_ids are not in {dataset.name}; "
                         f"wrong --dataset?")
    rows = by_id.loc[pts.point_id]
    dlon = float(np.abs(rows.lon.to_numpy(float) - pts.lon.to_numpy(float)).max())
    dlat = float(np.abs(rows.lat.to_numpy(float) - pts.lat.to_numpy(float)).max())
    max_diff = max(dlon, dlat)
    if max_diff > LONLAT_TOL:
        raise SystemExit(f"lon/lat of the regions and {dataset.name} differ by up to "
                         f"{max_diff:.1e} deg")
    xy = shapely.get_coordinates(shapely.from_wkb(rows.geometry.to_numpy()))
    return xy, f"{dataset.name} geometry (EPSG:{CRS_WORK}); lon/lat agree to {max_diff:.0e} deg"


def source_candidates(pts: pd.DataFrame, idx: pd.DataFrame, among: np.ndarray) -> tuple[dict, str]:
    """Candidate source polygons (idx row_ids) of each positive in `among`, keyed by its row in
    pts: its meta_src_row_id when the table has one, else every idx row with its
    (dfirm_id, meta_fld_ar_id) key. FLD_AR_ID alone repeats, and even the pair can name several
    rows (39 in Florida). Band negatives are left out: their meta_src_row_id names the ring they
    came from, not a polygon they lie in, so they contribute through the near rule only."""
    positives = np.flatnonzero((pts.label.to_numpy() == 1) & among)
    src = pts.meta_src_row_id.astype("Int64")
    cand, rest = {}, []
    for i in positives:
        if pd.notna(src.iloc[i]):
            cand[int(i)] = [int(src.iloc[i])]
        elif pd.notna(pts.meta_fld_ar_id.iloc[i]):
            rest.append(i)
    n_direct = len(cand)

    n_by_key = 0
    n_repeated = 0
    if rest:
        keys = pd.DataFrame(dict(i=rest,
                                 DFIRM_ID=pts.dfirm_id.iloc[rest].astype(str).to_numpy(),
                                 FLD_AR_ID=pts.meta_fld_ar_id.iloc[rest].astype(str).to_numpy()))
        idx_keys = idx[["row_id", "DFIRM_ID", "FLD_AR_ID"]]
        idx_keys = idx_keys.astype({"DFIRM_ID": str, "FLD_AR_ID": str})
        hits = keys.merge(idx_keys, on=["DFIRM_ID", "FLD_AR_ID"], how="inner")
        for i, group in hits.groupby("i"):
            cand[int(i)] = group.row_id.astype(int).tolist()
            n_by_key += 1
            if len(group) > 1:
                n_repeated += 1
    n_none = len(positives) - len(cand)
    return cand, (f"{n_direct} from meta_src_row_id, {n_by_key} by (dfirm_id, meta_fld_ar_id) "
                  f"({n_repeated} with a repeated key), {n_none} with neither")


def covering_polygons(row_ids, geoms: dict, point) -> list:
    """The row_ids among row_ids whose polygon covers point."""
    hits = []
    for r in row_ids:
        g = geoms.get(r)
        if g is not None and g.covers(point):
            hits.append(r)
    return hits


def resolve_sources(cand: dict, xy: np.ndarray, geoms: dict) -> tuple[dict, int, int]:
    """One source row_id per positive: its only candidate, else the one candidate that covers
    it. Returns (row -> row_id, n covered by their source polygon, n unresolved)."""
    out, n_cover, n_unres = {}, 0, 0
    for i, rids in cand.items():
        covers = covering_polygons(rids, geoms, shapely.Point(xy[i]))
        if len(rids) == 1:
            out[i] = rids[0]
            if len(covers) == 1:
                n_cover += 1
        elif len(covers) == 1:
            out[i] = covers[0]
            n_cover += 1
        else:
            n_unres += 1
    return out, n_cover, n_unres


def tile_box(tile: str) -> tuple[float, float, float, float]:
    """The EPSG:5070 box of tile_50km "tx_ty": x in [tx, tx + 1) * 50 km, y likewise.
    tile_50km is (bx // 5, by // 5) of the 10 km blocks (pull_gee.derive)."""
    tx_str, ty_str = str(tile).split("_")
    tx, ty = int(tx_str), int(ty_str)
    return tx * TILE_M, ty * TILE_M, (tx + 1) * TILE_M, (ty + 1) * TILE_M


def bbox_hits(idx: pd.DataFrame, x0, y0, x1, y1) -> np.ndarray:
    """Which rows of idx have a bbox that meets the box (x0, y0, x1, y1), edges included."""
    return ((idx.maxx.to_numpy() >= x0) & (idx.minx.to_numpy() <= x1) &
            (idx.maxy.to_numpy() >= y0) & (idx.miny.to_numpy() <= y1))


def subty(s: pd.Series) -> pd.Series:
    """ZONE_SUBTY as a nullable string; NFHL's 16 blank values in Florida -> null."""
    s = s.astype("string").str.strip()
    return s.mask(s == "")


# ----------------------------------------------------------------- geometry
def geom_array(col) -> np.ndarray:
    """A writable object array of geometries. Under pandas copy-on-write,
    Series.to_numpy() is a read-only view, which shapely's ring extraction rejects."""
    return np.array(col.to_numpy(), dtype=object, copy=True)


def polygon_parts(g) -> list:
    """Single polygons of a Polygon / MultiPolygon / GeometryCollection, in order."""
    out = []
    for p in shapely.get_parts(g):
        if p.is_empty:
            continue
        if p.geom_type == "Polygon":
            out.append(p)
        elif p.geom_type in ("MultiPolygon", "GeometryCollection"):
            out.extend(polygon_parts(p))
    return out


def explode(geoms: dict, zone: dict) -> tuple[pd.DataFrame, int]:
    """Every polygon's single parts, numbered within the polygon and oriented so the interior
    lies left of every ring (exterior counter-clockwise, holes clockwise). An invalid part is
    repaired with make_valid, keeping its polygonal pieces. Returns (parts, n repaired)."""
    rows, n_fixed = [], 0
    for rid, g in geoms.items():
        k = 0
        for p in polygon_parts(g):
            pieces = [p]
            if not p.is_valid:
                pieces = polygon_parts(shapely.make_valid(p))
                n_fixed += 1
            for q in pieces:
                rows.append((rid, k, zone[rid], q))
                k += 1
    parts = pd.DataFrame(rows, columns=["row_id", "part", "zone", "geometry"])
    parts["geometry"] = shapely.orient_polygons(parts.geometry.to_numpy(), exterior_cw=False)
    return parts, n_fixed


def lobe_tiles(geoms: np.ndarray) -> list[str]:
    """';'-joined tile_50km ids each lobe intersects."""
    bounds = shapely.bounds(geoms)
    lowest = np.floor(bounds[:, :2] / TILE_M).astype(np.int64)
    highest = np.floor(bounds[:, 2:] / TILE_M).astype(np.int64)
    # Every (lobe, tile) pair inside the lobe's bbox, tested exactly below.
    lobe_ix, tx, ty = [], [], []
    for i in range(len(geoms)):
        for tile_x in range(lowest[i, 0], highest[i, 0] + 1):
            for tile_y in range(lowest[i, 1], highest[i, 1] + 1):
                lobe_ix.append(i)
                tx.append(tile_x)
                ty.append(tile_y)
    lobe_ix, tx, ty = np.array(lobe_ix, int), np.array(tx), np.array(ty)
    boxes = shapely.box(tx * TILE_M, ty * TILE_M, (tx + 1) * TILE_M, (ty + 1) * TILE_M)
    hit = shapely.intersects(geoms[lobe_ix], boxes)
    tiles = [[] for _ in geoms]
    for i, tile_x, tile_y in zip(lobe_ix[hit], tx[hit], ty[hit]):
        tiles[i].append(f"{tile_x}_{tile_y}")
    return [";".join(t) for t in tiles]


def densify(geoms: np.ndarray, step: float) -> dict:
    """Evenly spaced vertices and unit inward normals on every ring of the oriented polygons.

    A ring of length L gets n = ceil(L / step) vertices at s = k L / n, k = 0..n-1; the original
    vertices are not kept. Even spacing makes a share of edge length a vertex count. The tangent
    is the chord between the two neighbouring vertices, at s - L / n and s + L / n (L / 4 instead
    of L / n on rings of fewer than four vertices), so the normal is that of the step-spaced
    polyline (at a corner, the bisector); a sub-metre chord would follow the zig-zags of
    lidar-traced edges.
    The interior is on the left, so the inward normal is the tangent's left perpendicular.

    All rings lie end to end on one cumulative length axis, so each lookup is one searchsorted
    over all segments (GEOS's interpolate is quadratic on long rings). Segments are not copied
    out: ring r's coordinates are contiguous and closed, so global segment j of ring r starts at
    coordinate j + r. That keeps memory near six floats per coordinate.
    """
    rings, ring_lobe = shapely.get_rings(geoms, return_index=True)
    first_ring = np.searchsorted(ring_lobe, np.arange(len(geoms)))
    ring_no = np.arange(len(rings)) - first_ring[ring_lobe]
    coords = shapely.get_coordinates(rings)
    n_coords = shapely.get_num_coordinates(rings)
    n_segs = np.maximum(n_coords - 1, 0)                     # segments per ring
    first_seg = np.cumsum(n_segs) - n_segs                   # its first global segment
    seg_len = np.hypot(*np.diff(coords, axis=0).T)
    seg_len = np.delete(seg_len, (np.cumsum(n_coords) - 1)[:-1])   # drop steps between rings
    seg_start = np.concatenate([[0.0], np.cumsum(seg_len)])  # length axis at segment j's start
    ring_start = seg_start[first_seg]
    ring_len = seg_start[first_seg + n_segs] - seg_start[first_seg]

    n_vert = np.where(ring_len > 0, np.maximum(np.ceil(ring_len / step), 1), 0).astype(np.int64)
    v_ring = np.repeat(np.arange(len(rings)), n_vert)        # ring of each vertex
    v_k = np.arange(len(v_ring)) - np.repeat(np.cumsum(n_vert) - n_vert, n_vert)
    v_len = ring_len[v_ring]
    v_step = v_len / n_vert[v_ring]
    s = v_k * v_step

    def point_at(t):
        """Point at ring position t (wrapped) and the index of its segment's first coordinate."""
        pos = ring_start[v_ring] + np.mod(t, v_len)
        seg = np.searchsorted(seg_start, pos, side="right") - 1
        seg = np.clip(seg, first_seg[v_ring], first_seg[v_ring] + n_segs[v_ring] - 1)
        safe_len = np.where(seg_len[seg] > 0, seg_len[seg], 1.0)
        frac = np.clip((pos - seg_start[seg]) / safe_len, 0.0, 1.0)
        coord = seg + v_ring
        return coords[coord] + frac[:, None] * (coords[coord + 1] - coords[coord]), coord

    vertex, coord = point_at(s)
    half_chord = np.minimum(v_step, v_len / 4)
    tangent = point_at(s + half_chord)[0] - point_at(s - half_chord)[0]
    norm = np.hypot(*tangent.T)
    # At a spike the two neighbours coincide; fall back to the vertex's own segment.
    flat = norm < 1e-9
    tangent[flat] = (coords[coord + 1] - coords[coord])[flat]
    norm = np.hypot(*tangent.T)
    tangent /= np.where(norm > 0, norm, 1)[:, None]
    return dict(lobe=ring_lobe[v_ring], ring=ring_no[v_ring].astype(np.int32), s=s,
                x=vertex[:, 0], y=vertex[:, 1], nx=-tangent[:, 1], ny=tangent[:, 0])


def check_normals(geoms: np.ndarray, bounds: np.ndarray, verts: dict, sel: np.ndarray) -> dict:
    """Probe test of the normals at vertices `sel` (indices into verts, densify's output):
    x + PROBE_M n must be inside the lobe and x - PROBE_M n outside. bounds are the lobes'
    prepared boundaries."""
    lobe = verts["lobe"][sel]
    x, y = verts["x"][sel], verts["y"][sel]
    nx, ny = verts["nx"][sel], verts["ny"][sel]
    probe_in = shapely.points(x + PROBE_M * nx, y + PROBE_M * ny)
    probe_out = shapely.points(x - PROBE_M * nx, y - PROBE_M * ny)
    in_inside = shapely.contains(geoms[lobe], probe_in)
    out_inside = shapely.contains(geoms[lobe], probe_out)
    near_in = shapely.dwithin(bounds[lobe], probe_in, CLEAR_FRAC * PROBE_M)
    near_out = shapely.dwithin(bounds[lobe], probe_out, CLEAR_FRAC * PROBE_M)
    clear = ~(near_in | near_out)
    ok = in_inside & ~out_inside
    flipped = ~in_inside & out_inside
    return dict(n=len(sel), ok=int(ok.sum()), flipped=int(flipped.sum()),
                ambiguous=int((in_inside == out_inside).sum()), n_clear=int(clear.sum()),
                ok_clear=int((ok & clear).sum()), flipped_clear=int((flipped & clear).sum()))


def match_perimeter(reg_logp: np.ndarray, cand_logp: np.ndarray, n_rounds: int,
                    rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """Match candidates to region lobes on log perimeter, round-robin without replacement.

    In each of n_rounds rounds every region lobe, in a seeded random order, takes the free
    candidate nearest in log perimeter (ties by a seeded random key), so no lobe exhausts the
    good matches before the others get one. Log, because perimeters span five orders of
    magnitude. Returns (region index, candidate index, |dlog P|) per match.
    """
    free = np.ones(len(cand_logp), bool)
    tie = rng.random(len(cand_logp))
    order = rng.permutation(len(reg_logp))
    out = []
    for _ in range(n_rounds):
        for i in order:
            if not free.any():
                return out
            d = np.where(free, np.abs(cand_logp - reg_logp[i]), np.inf)
            j = int(np.lexsort((tie, d))[0])
            free[j] = False
            out.append((int(i), j, float(d[j])))
    return out


def lonlat_bounds(x0, y0, x1, y1) -> tuple[float, float, float, float]:
    """EPSG:4326 bounds of an EPSG:5070 box, over its densified edges, so the lon/lat box covers
    the whole projected box rather than only its corners."""
    tf = Transformer.from_crs(CRS_WORK, CRS_LONLAT_OUT, always_xy=True)
    return tf.transform_bounds(x0, y0, x1, y1, densify_pts=101)


# ----------------------------------------------------------------- build
def plan_component(c: int, pts: pd.DataFrame, xy: np.ndarray, idx: pd.DataFrame,
                   args) -> dict:
    """The polygons each role of component c may draw from, chosen on idx's bbox columns.
    The tiles are those of the component's own points: the ground the models were tested on."""
    mask = pts.component.to_numpy() == c
    comp_xy = xy[mask]
    tiles = sorted(pts.tile_50km[mask].astype(str).unique())
    zone = idx.FLD_ZONE.to_numpy()
    near_box = (comp_xy[:, 0].min() - args.near_m, comp_xy[:, 1].min() - args.near_m,
                comp_xy[:, 0].max() + args.near_m, comp_xy[:, 1].max() + args.near_m)
    near = (zone == REGION_ZONE) & bbox_hits(idx, *near_box)
    in_tiles = np.zeros(len(idx), bool)
    for t in tiles:
        in_tiles |= bbox_hits(idx, *tile_box(t))
    row_id = idx.row_id.to_numpy()
    hull = shapely.convex_hull(shapely.multipoints(comp_xy))
    return dict(mask=mask, xy=comp_xy, tiles=tiles,
                hull=shapely.buffer(hull, args.hull_buffer_m),
                near_ids=row_id[near],
                A_tile_ids=row_id[in_tiles & (zone == REGION_ZONE)],
                AE_tile_ids=row_id[in_tiles & (zone == CONTRAST_ZONE)])


def require_zone_a_sources(cand: dict, zone: dict) -> None:
    """Abort when a positive has no Zone A candidate source polygon: that would be an A->AE
    region, whose roles this script does not define."""
    other = set()
    for rids in cand.values():
        if all(zone[r] != REGION_ZONE for r in rids):
            other.update(rids)
    other = sorted(other)
    if other:
        raise SystemExit(f"{len(other)} source polygons of region positives are not Zone "
                         f"{REGION_ZONE} (e.g. row_id {other[0]}: {zone[other[0]]}); lobes.py "
                         f"handles AE->A regions only")


def polygons_to_read(cand: dict, plans: dict, big: list[int]) -> np.ndarray:
    """Sorted row_ids of the polygons build needs: the candidate source polygons, every
    component's near-rule polygons, and the tile polygons of the components in big."""
    need = set()
    for rids in cand.values():
        need.update(rids)
    for c, plan in plans.items():
        need.update(plan["near_ids"].tolist())
        if c in big:
            need.update(plan["A_tile_ids"].tolist())
            need.update(plan["AE_tile_ids"].tolist())
    return np.array(sorted(need), dtype=np.int64)


def region_parts(plan: dict, parts: pd.DataFrame, tree: STRtree, geoms: dict, src: dict,
                 near_m: float) -> tuple[np.ndarray, set, set, set]:
    """One component's region lobes, as indices into parts.

    A polygon qualifies when it is a positive's source polygon or lies within near_m of a region
    point; its parts are kept when they meet the buffered hull of the region's points. The test
    is on whole polygons first and parts second because an NFHL polygon can scatter parts tens of
    km apart. Returns (indices, the positives' source polygons, the polygons within near_m,
    all qualifying polygons).
    """
    from_pos = set()
    for i in np.flatnonzero(plan["mask"]):
        if i in src:
            from_pos.add(src[i])
    near_geoms = np.array([geoms[r] for r in plan["near_ids"]], dtype=object)
    hit = np.empty(0, dtype=np.int64)
    if len(near_geoms):
        near_tree = STRtree(near_geoms)
        hit = near_tree.query(shapely.points(plan["xy"]), predicate="dwithin",
                              distance=near_m)[1]
    near_set = set(plan["near_ids"][np.unique(hit)].tolist())
    polys = from_pos | near_set
    in_poly = parts.row_id.isin(polys).to_numpy()
    on_hull = np.zeros(len(parts), bool)
    on_hull[tree.query(plan["hull"], predicate="intersects")] = True
    return np.flatnonzero(in_poly & on_hull), from_pos, near_set, polys


def role_rows(parts: pd.DataFrame, part_geoms: np.ndarray, n_points: np.ndarray,
              ix: np.ndarray, role: str, c: int) -> pd.DataFrame:
    """Lobe rows for parts[ix] in one role of component c."""
    out = pd.DataFrame(dict(
        src_row_id=parts.row_id.to_numpy()[ix], part=parts.part.to_numpy()[ix],
        zone=parts.zone.to_numpy()[ix], geometry=part_geoms[ix], n_points=n_points[ix]))
    out["role"] = role
    out["component"] = c
    out["lobe_id"] = [f"c{c}_{role}_{r}_{k}" for r, k in zip(out.src_row_id, out.part)]
    return out


def add_attributes(lobes: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """The lobe table with its NFHL attributes, area, perimeter and tiles added."""
    meta = idx.set_index("row_id")[["FLD_AR_ID", "DFIRM_ID", "ZONE_SUBTY"]]
    lobes["fld_ar_id"] = meta.FLD_AR_ID.reindex(lobes.src_row_id).to_numpy()
    lobes["dfirm_id"] = meta.DFIRM_ID.reindex(lobes.src_row_id).to_numpy()
    lobes["zone_subty"] = subty(meta.ZONE_SUBTY.reindex(lobes.src_row_id)).array
    is_floodway = lobes.zone_subty.str.contains(FLOODWAY, regex=False)
    lobes["is_floodway"] = is_floodway.fillna(False).astype(bool)
    lobe_geoms = geom_array(lobes.geometry)
    lobes["area_m2"] = shapely.area(lobe_geoms)
    lobes["perimeter_m"] = shapely.length(lobe_geoms)
    lobes["tiles"] = lobe_tiles(lobe_geoms)
    lobes["from_positive"] = lobes.from_positive.astype("boolean")
    return lobes.astype({"component": np.int32, "src_row_id": np.int64, "part": np.int32,
                         "n_points": np.int32})


def build(pts: pd.DataFrame, xy: np.ndarray, big: list[int], idx: pd.DataFrame, a):
    """The lobes of the components in big, with `a` the parsed command line.
    Returns (lobe table, plans of big, row_id -> polygon of every polygon read)."""
    t0 = time.time()
    # Region lobes are found for every component of >= min_size points, not only big, so that
    # --only-components cannot let another component's region lobes into matched_A.
    sizes = pts.groupby("component").size()
    every = sorted(set(big) | {int(c) for c in sizes.index[sizes >= a.min_size]})
    cand, cand_note = source_candidates(pts, idx, pts.component.isin(every).to_numpy())
    print(f"  positives' source polygons (components of >= {a.min_size} points): {cand_note}",
          flush=True)
    zone = dict(zip(idx.row_id.to_numpy().tolist(), idx.FLD_ZONE.to_numpy().tolist()))
    require_zone_a_sources(cand, zone)
    plans = {c: plan_component(c, pts, xy, idx, a) for c in every}

    # haz_<ST>_5070.parquet is 5.4 GB for Florida, so only these polygons are read, in one pass
    # (read_geoms reads each row group at most once).
    need = polygons_to_read(cand, plans, big)
    n_groups = len(np.unique(need // sp.RG_ROWS))
    print(f"  reading {len(need):,} candidate polygons from {a.haz.name} ({n_groups} row groups)",
          flush=True)
    polys = sp.read_geoms(a.state, need)
    n_missing = sum(g is None for g in polys)
    if n_missing:
        raise SystemExit(f"{n_missing} candidate row_ids have no geometry in {a.haz}")
    geoms = dict(zip(need.tolist(), polys))
    print(f"    read in {time.time() - t0:.1f}s", flush=True)

    src, n_cover, n_unres = resolve_sources(cand, xy, geoms)
    bad_zone = sorted({r for r in src.values() if zone[r] != REGION_ZONE})
    if bad_zone:
        raise SystemExit(f"{len(bad_zone)} resolved source polygons are not Zone {REGION_ZONE} "
                         f"(e.g. row_id {bad_zone[0]}: {zone[bad_zone[0]]})")
    print(f"  {len(src)} positives with a source polygon: {n_cover} covered by it, "
          f"{len(src) - n_cover} not; {n_unres} unresolved", flush=True)

    parts, n_fixed = explode(geoms, zone)
    part_geoms = geom_array(parts.geometry)
    shapely.prepare(part_geoms)
    tree = STRtree(part_geoms)
    repaired = f"; {n_fixed} invalid parts repaired" if n_fixed else ""
    print(f"  {len(parts):,} single parts from {len(geoms):,} polygons{repaired}", flush=True)

    # Which parts cover (contain or have on their boundary) which selected points, all components.
    point_ix, part_ix = tree.query(shapely.points(xy), predicate="covered_by")
    covers_any = np.zeros(len(parts), bool)
    covers_any[part_ix] = True
    comp_of_point = pts.component.to_numpy()

    region, tile_parts = {}, {}
    for c, plan in plans.items():
        region[c], from_pos, near_set, qualifying = region_parts(plan, parts, tree, geoms, src,
                                                                 a.near_m)
        boxes = shapely.box(*np.array([tile_box(t) for t in plan["tiles"]]).T)
        tile_parts[c] = np.unique(tree.query(boxes, predicate="intersects")[1])
        plan.update(from_pos=from_pos, n_near=len(near_set), n_polys=len(qualifying))
    is_region_any = np.zeros(len(parts), bool)
    for ix in region.values():
        is_region_any[ix] = True
    log_perimeter = np.log(shapely.length(part_geoms))
    part_zone = parts.zone.to_numpy()

    tables = []
    for c in big:
        plan = plans[c]
        rng = np.random.default_rng([a.seed, c])
        n_here = np.bincount(part_ix[comp_of_point[point_ix] == c], minlength=len(parts))
        in_tiles = tile_parts[c]
        ae = in_tiles[part_zone[in_tiles] == CONTRAST_ZONE]
        # Eligible for matched_A: a Zone A part in the tiles that covers no selected point of
        # any component and is no component's region lobe.
        eligible = in_tiles[(part_zone[in_tiles] == REGION_ZONE) & ~covers_any[in_tiles]
                            & ~is_region_any[in_tiles]]
        reg = region[c]
        matches = match_perimeter(log_perimeter[reg], log_perimeter[eligible], a.n_matched, rng)

        region_df = role_rows(parts, part_geoms, n_here, reg, "region", c)
        region_df["from_positive"] = np.isin(parts.row_id.to_numpy()[reg], list(plan["from_pos"]))
        matched_ix = np.array([eligible[j] for _, j, _ in matches], dtype=np.int64)
        matched_df = role_rows(parts, part_geoms, n_here, matched_ix, "matched_A", c)
        matched_df["matched_to"] = [region_df.lobe_id.iloc[i] for i, _, _ in matches]
        matched_df["match_dlogp"] = [d for _, _, d in matches]
        ae_df = role_rows(parts, part_geoms, n_here, ae, "AE", c)
        by = ["src_row_id", "part"]
        tables += [region_df.sort_values(by), matched_df, ae_df.sort_values(by)]
        plan.update(n_region=len(region_df), n_matched=len(matched_df), n_eligible=len(eligible),
                    n_ae=len(ae_df), dlogp=np.array([d for _, _, d in matches]))

    lobes = add_attributes(pd.concat(tables, ignore_index=True), idx)
    if not lobes.lobe_id.is_unique:
        raise SystemExit("lobe_id is not unique; a part was taken twice in one role")
    big_plans = {c: plans[c] for c in big}
    return lobes[LOBE_COLS + ["geometry"]], big_plans, geoms


def vertices_and_checks(lobes: pd.DataFrame, plans: dict, args) -> tuple[pd.DataFrame, dict]:
    """The vertex table, and the inward-normal check on a seeded sample of each component's
    vertices; raises AssertionError when a component fails it."""
    lobe_geoms = geom_array(lobes.geometry)
    shapely.prepare(lobe_geoms)
    verts = densify(lobe_geoms, args.densify_m)
    bounds = shapely.boundary(lobe_geoms)
    shapely.prepare(bounds)
    comp = lobes.component.to_numpy()[verts["lobe"]]
    role = lobes.role.to_numpy()[verts["lobe"]]
    checks = {}
    for c in plans:
        rng = np.random.default_rng([args.seed, c, 1])
        here = np.flatnonzero(comp == c)
        sel = np.sort(rng.choice(here, min(args.check_n, len(here)), replace=False))
        result = {}
        for r in ROLES:
            result[r] = check_normals(lobe_geoms, bounds, verts, sel[role[sel] == r])
        total = {}
        for key in result["region"]:
            total[key] = sum(result[r][key] for r in ROLES)
        result["all"] = total
        checks[c] = result
        # An explicit raise rather than `assert`, so python -O cannot skip it.
        enough_pass = total["ok_clear"] >= CHECK_MIN_PASS * total["n_clear"]
        few_flipped = total["flipped_clear"] <= CHECK_MAX_FLIPPED * total["n_clear"]
        if not (enough_pass and few_flipped):
            raise AssertionError(
                f"component {c}: inward-normal check failed on clear edges: "
                f"{total['ok_clear']}/{total['n_clear']} pass, {total['flipped_clear']} flipped "
                f"(need >= {CHECK_MIN_PASS:.1%} and <= {CHECK_MAX_FLIPPED:.1%})")

    tf = Transformer.from_crs(CRS_WORK, CRS_LONLAT_OUT, always_xy=True)
    lon, lat = tf.transform(verts["x"], verts["y"])
    lobe = verts["lobe"]
    vt = pd.DataFrame(dict(
        lobe_id=lobes.lobe_id.to_numpy()[lobe], zone=lobes.zone.to_numpy()[lobe],
        role=role, component=comp.astype(np.int32), ring=verts["ring"], s=verts["s"],
        x=verts["x"], y=verts["y"], lon=lon, lat=lat, nx=verts["nx"], ny=verts["ny"]))
    return vt[VERTEX_COLS], checks


def bbox_table(lobes: pd.DataFrame, plans: dict, buf: float) -> tuple[pd.DataFrame, dict]:
    """Each component's analysis bbox: the union of all its lobes' extents, every role, buffered
    by buf so a 1 km displaced-edge null and the normal probes stay inside every layer pulled
    for it. Also returns, per component, the size the box would have from its region lobes
    alone (NaN without region lobes)."""
    bounds = shapely.bounds(geom_array(lobes.geometry))      # minx, miny, maxx, maxy per lobe
    comp = lobes.component.to_numpy()
    is_region = lobes.role.to_numpy() == "region"
    rows, region_only = [], {}
    for c in plans:
        in_comp = comp == c
        x0, y0 = bounds[in_comp, 0].min() - buf, bounds[in_comp, 1].min() - buf
        x1, y1 = bounds[in_comp, 2].max() + buf, bounds[in_comp, 3].max() + buf
        lon0, lat0, lon1, lat1 = lonlat_bounds(x0, y0, x1, y1)
        rows.append(dict(component=c, minx=x0, miny=y0, maxx=x1, maxy=y1, lon_min=lon0,
                         lat_min=lat0, lon_max=lon1, lat_max=lat1, width_km=(x1 - x0) / 1e3,
                         height_km=(y1 - y0) / 1e3, area_km2=(x1 - x0) * (y1 - y0) / 1e6,
                         n_lobes=int(in_comp.sum())))
        in_region = in_comp & is_region
        if in_region.any():
            width = bounds[in_region, 2].max() - bounds[in_region, 0].min() + 2 * buf
            height = bounds[in_region, 3].max() - bounds[in_region, 1].min() + 2 * buf
        else:
            width = height = np.nan
        region_only[c] = dict(region_km=(width / 1e3, height / 1e3),
                              region_km2=width * height / 1e6)
    return pd.DataFrame(rows).astype({"component": np.int32}), region_only


def contact_candidates(idx: pd.DataFrame, lobes: pd.DataFrame, bb: pd.DataFrame,
                       tol: float) -> dict:
    """SFHA row_ids per component whose idx bbox meets the component's analysis bbox and comes
    within tol of one of its lobes' bboxes."""
    sfha = idx[idx.SFHA_TF.to_numpy() == "T"]
    sfha_boxes = shapely.box(sfha.minx.to_numpy(), sfha.miny.to_numpy(), sfha.maxx.to_numpy(),
                             sfha.maxy.to_numpy())
    lobe_boxes = shapely.bounds(geom_array(lobes.geometry)) + np.array([-tol, -tol, tol, tol])
    comp = lobes.component.to_numpy()
    out = {}
    for box in bb.itertuples():
        in_bbox = np.flatnonzero(bbox_hits(sfha, box.minx, box.miny, box.maxx, box.maxy))
        comp_tree = STRtree(shapely.box(*lobe_boxes[comp == box.component].T))
        near = comp_tree.query(sfha_boxes[in_bbox], predicate="intersects")[0]
        out[int(box.component)] = sfha.row_id.to_numpy()[in_bbox[np.unique(near)]]
    return out


def nearest_contacts(tree: STRtree, vertex_pts: np.ndarray, polys: np.ndarray,
                     row_ids: np.ndarray, own_row_id: np.ndarray,
                     tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """For the vertices in tree, the nearest of polys within tol that is not the vertex's own
    polygon; ties go to the lower row_id. Returns (vertex indices, their contact row_ids, the
    number of vertices with several candidates)."""
    poly_ix, v_ix = tree.query(polys, predicate="dwithin", distance=tol)
    not_own = row_ids[poly_ix] != own_row_id[v_ix]
    poly_ix, v_ix = poly_ix[not_own], v_ix[not_own]
    # Exact distances are needed only where a vertex has several candidates.
    several = np.bincount(v_ix, minlength=len(vertex_pts))[v_ix] > 1
    n_several = len(np.unique(v_ix[several]))
    dist = np.zeros(len(v_ix))
    dist[several] = shapely.distance(vertex_pts[v_ix[several]], polys[poly_ix[several]])
    order = np.lexsort((row_ids[poly_ix], dist, v_ix))
    v_ix, poly_ix = v_ix[order], poly_ix[order]
    if len(v_ix):
        first = np.r_[True, v_ix[1:] != v_ix[:-1]]
    else:
        first = np.zeros(0, bool)
    return v_ix[first], row_ids[poly_ix[first]], n_several


def contacts(lobes: pd.DataFrame, vt: pd.DataFrame, bb: pd.DataFrame, idx: pd.DataFrame,
             geoms: dict, args) -> tuple[pd.DataFrame, str]:
    """The vertex table with CONTACT_COLS added, and a report line.

    Not all of a lobe's boundary is an SFHA limit: some of it is a contact with another SFHA
    polygon (a floodway/fringe line, a polygon split by source or at a sheet edge, an A/AE
    contact), and an edge shared by two lobes of one role is written once per lobe. So each vertex
    records the nearest other SFHA polygon (SFHA_TF 'T', any zone) within --contact-m, null when
    there is none, and same_role_contact: whether another lobe of its role and component lies that
    close. The analysis can then keep the SFHA-exterior edge, the duplicates, or both. "Other"
    excludes the vertex's whole own polygon, which loses no shared edge: the parts of a valid
    MultiPolygon touch only at points. Shared NFHL edges coincide to floating point, so
    --contact-m only absorbs round-off in the densified vertices. Candidate polygons that build
    did not read come in one more read_geoms pass.
    """
    t0 = time.time()
    tol = args.contact_m
    cand = contact_candidates(idx, lobes, bb, tol)
    every = np.unique(np.concatenate(list(cand.values())))
    already_read = np.fromiter(geoms, dtype=np.int64, count=len(geoms))
    new = np.setdiff1d(every, already_read)
    new_geoms = sp.read_geoms(args.state, new)
    n_missing = sum(g is None for g in new_geoms)
    if n_missing:
        raise SystemExit(f"{n_missing} SFHA row_ids have no geometry in {args.haz}")
    polygons = dict(geoms)
    polygons.update(zip(new.tolist(), new_geoms))

    lobe_of_vertex = pd.Index(lobes.lobe_id).get_indexer(vt.lobe_id)
    own_row_id = lobes.src_row_id.to_numpy()[lobe_of_vertex]
    role = lobes.role.to_numpy()
    lobe_geoms = geom_array(lobes.geometry)
    lobe_comp, vertex_comp = lobes.component.to_numpy(), vt.component.to_numpy()
    x, y = vt.x.to_numpy(), vt.y.to_numpy()
    contact_row = np.full(len(vt), -1, dtype=np.int64)
    same = np.zeros(len(vt), dtype=bool)
    n_several = 0
    for c, row_ids in cand.items():
        # Vertices go in the tree and the polygons query it: shapely prepares the query
        # geometry, so a vertex test against a huge polygon costs log n, not n.
        comp_vertex = np.flatnonzero(vertex_comp == c)
        vertex_pts = shapely.points(x[comp_vertex], y[comp_vertex])
        tree = STRtree(vertex_pts)
        polys = np.array([polygons[r] for r in row_ids], dtype=object)
        v_ix, contact, n = nearest_contacts(tree, vertex_pts, polys, row_ids,
                                            own_row_id[comp_vertex], tol)
        contact_row[comp_vertex[v_ix]] = contact
        n_several += n

        lobe_rows = np.flatnonzero(lobe_comp == c)
        lobe_ix, v_ix = tree.query(lobe_geoms[lobe_rows], predicate="dwithin", distance=tol)
        other_lobe = lobe_rows[lobe_ix]
        vertex_lobe = lobe_of_vertex[comp_vertex[v_ix]]
        same_role = (other_lobe != vertex_lobe) & (role[other_lobe] == role[vertex_lobe])
        same[comp_vertex[v_ix[same_role]]] = True

    has = contact_row >= 0
    lookup = np.where(has, contact_row, 0)       # row 0 stands in for none; masked below
    attrs = idx.set_index("row_id")
    zone = pd.Series(attrs.FLD_ZONE.reindex(lookup).to_numpy(), dtype="string")
    zone_subty = subty(pd.Series(attrs.ZONE_SUBTY.reindex(lookup).to_numpy()))
    out = vt.assign(
        contact_row_id=pd.Series(contact_row, dtype="Int64").mask(~has).array,
        contact_zone=zone.mask(~has).array,
        contact_subty=zone_subty.mask(~has).array,
        same_role_contact=same)
    note = (f"{len(every):,} SFHA polygons within {tol:g} m of a lobe bbox ({len(new):,} more "
            f"read); {int(has.sum()):,} of {len(vt):,} vertices have a contact, {n_several:,} "
            f"of them several (nearest taken); {int(same.sum()):,} lie on another lobe of "
            f"their role ({time.time() - t0:.1f}s)")
    return out[VERTEX_COLS + CONTACT_COLS], note


# ----------------------------------------------------------------- report
def print_lobes(c: int, plan: dict, comp_lobes: pd.DataFrame, pts: pd.DataFrame, args) -> None:
    """The header and the region, matched_A and AE lines of one component."""
    n_points = int(plan["mask"].sum())
    n_pos = int((pts.label[plan["mask"]] == 1).sum())
    perimeter_km = comp_lobes.groupby("role").perimeter_m.sum() / 1e3
    print(f"\n--- component {c}: {n_points} points ({n_pos} positive), "
          f"tiles {';'.join(plan['tiles'])}", flush=True)

    reg = comp_lobes[comp_lobes.role == "region"]
    near_only = reg[~reg.from_positive.fillna(False).astype(bool)]
    print(f"  region    {plan['n_region']:>6,} lobes from {reg.src_row_id.nunique():,} of "
          f"{plan['n_polys']:,} polygons ({len(plan['from_pos'])} positives' source polygons, "
          f"{plan['n_near']:,} within {args.near_m:g} m); perimeter "
          f"{perimeter_km.get('region', 0):,.0f} km, {near_only.perimeter_m.sum() / 1e3:,.0f} km "
          f"of it on {len(near_only):,} lobes of near-only polygons; "
          f"{int(reg.n_points.sum())} of {n_points} points inside")

    dlogp = plan["dlogp"]
    want = args.n_matched * plan["n_region"]
    spread = ""
    if len(dlogp):
        spread = f"; |dlog P| median {np.median(dlogp):.3f}, max {dlogp.max():.3f}"
    print(f"  matched_A {plan['n_matched']:>6,} lobes of {want:,} wanted ({args.n_matched} per "
          f"region lobe) from {plan['n_eligible']:,} eligible; perimeter "
          f"{perimeter_km.get('matched_A', 0):,.0f} km{spread}")

    floodway = comp_lobes[(comp_lobes.role == "AE") & comp_lobes.is_floodway]
    print(f"  AE        {plan['n_ae']:>6,} lobes; perimeter {perimeter_km.get('AE', 0):,.0f} km, "
          f"{floodway.perimeter_m.sum() / 1e3:,.0f} km of it on {len(floodway):,} floodway lobes")


def print_vertices(c: int, comp_vt: pd.DataFrame, n_vertices: pd.Series, contact_m: float) -> None:
    """Vertex counts by role, and each role's contact shares."""
    counts = [int(n_vertices.get((c, r), 0)) for r in ROLES]
    by_role = " | ".join(f"{r} {n:,}" for r, n in zip(ROLES, counts))
    print(f"  vertices  {by_role} | total {sum(counts):,}")
    for r in ROLES:
        role_vt = comp_vt[comp_vt.role == r]
        if not len(role_vt):
            continue
        has_contact = role_vt.contact_row_id.notna().to_numpy()
        zone_shares = role_vt.contact_zone[has_contact].value_counts(normalize=True)
        zones = ", ".join(f"{z} {share:.0%}" for z, share in zone_shares.head(4).items())
        floodway_share = 0
        if has_contact.any():
            is_floodway = role_vt.contact_subty[has_contact].str.contains(FLOODWAY, regex=False)
            floodway_share = is_floodway.fillna(False).mean()
        print(f"  contacts  {r:<9} {1 - has_contact.mean():6.1%} SFHA-exterior; "
              f"{has_contact.mean():.1%} on another SFHA polygon within {contact_m:g} m "
              f"({zones}; floodway {floodway_share:.0%}); "
              f"{role_vt.same_role_contact.mean():.1%} on another {r} lobe")


def print_normals(check: dict) -> None:
    """The inward-normal check of one component, overall and by role."""
    total = check["all"]
    print(f"  normals   {total['n']:,} sampled vertices, +-{PROBE_M:g} m probes: clear edges "
          f"{total['ok_clear']:,}/{total['n_clear']:,} pass "
          f"({total['ok_clear'] / max(total['n_clear'], 1):.3%}), "
          f"{total['flipped_clear']} flipped -> assertion passed; all sampled "
          f"{total['ok']:,}/{total['n']:,} pass ({total['ok'] / max(total['n'], 1):.2%}), "
          f"{total['ambiguous']:,} ambiguous, {total['flipped']:,} flipped")
    by_role = []
    for r in ROLES:
        by_role.append(f"{r} {check[r]['ok']:,}/{check[r]['n']:,}, "
                       f"{check[r]['ok_clear']:,}/{check[r]['n_clear']:,}")
    print("            by role (pass/n, clear pass/clear n): " + " | ".join(by_role))


def print_bbox(box, region_only: dict, bbox_buffer_m: float) -> None:
    """One component's analysis bbox, and the size of its region-lobes-only box."""
    width_km, height_km = region_only["region_km"]
    print(f"  bbox      {box.width_km:.1f} x {box.height_km:.1f} km = {box.area_km2:,.0f} km2 "
          f"(lon {box.lon_min:.4f}..{box.lon_max:.4f}, lat {box.lat_min:.4f}..{box.lat_max:.4f}); "
          f"region lobes alone + {bbox_buffer_m / 1e3:g} km: {width_km:.1f} x "
          f"{height_km:.1f} km = {region_only['region_km2']:,.0f} km2")


def summarise(lobes, vt, bb, region_only, plans, checks, args, pts) -> None:
    """Print the report of every component, then the bbox totals."""
    n_vertices = vt.groupby(["component", "role"]).size()
    for c, plan in plans.items():
        print_lobes(c, plan, lobes[lobes.component == c], pts, args)
        print_vertices(c, vt[vt.component == c], n_vertices, args.contact_m)
        print_normals(checks[c])
        print_bbox(bb[bb.component == c].iloc[0], region_only[c], args.bbox_buffer_m)
    boxes = shapely.box(bb.minx, bb.miny, bb.maxx, bb.maxy)
    union_km2 = shapely.union_all(boxes).area / 1e6
    region_only_km2 = sum(v["region_km2"] for v in region_only.values())
    print(f"\n  analysis bboxes: {len(bb)}, sum of areas {bb.area_km2.sum():,.0f} km2, area of "
          f"their union {union_km2:,.0f} km2; region-lobe-only boxes would sum to "
          f"{region_only_km2:,.0f} km2", flush=True)


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", required=True, type=Path,
                    help="<cell>_<variant>_S<S>_points.parquet from aotil/regions.py")
    ap.add_argument("--components", type=Path,
                    help="the matching _components.csv (default: next to --regions)")
    ap.add_argument("--dataset", type=Path,
                    help="dataset parquet the regions came from; its EPSG:5070 geometry gives "
                         "exact point coordinates (default: project the regions lon/lat)")
    ap.add_argument("--idx", type=Path,
                    help="idx_<ST>.parquet; haz_<ST>_5070.parquet must sit next to it "
                         "(default: data/interim/prep/idx_<ST>.parquet)")
    ap.add_argument("--state", default="FL")
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--hull-buffer-m", type=float, default=2000.0)
    ap.add_argument("--near-m", type=float, default=2000.0)
    ap.add_argument("--densify-m", type=float, default=30.0)
    ap.add_argument("--n-matched", type=int, default=3)
    ap.add_argument("--bbox-buffer-m", type=float, default=3000.0)
    ap.add_argument("--contact-m", type=float, default=0.5,
                    help="another SFHA polygon or same-role lobe this close to a vertex is a "
                         "contact")
    ap.add_argument("--check-n", type=int, default=20_000,
                    help="vertices per component sampled for the inward-normal check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-components", type=int, nargs="+",
                    help="build lobes for these components only (each must have >= --min-size "
                         "points); every selected point still counts as selected and every "
                         "component's region lobes are still found, so the other components' "
                         "polygons can never be matched_A. Default: all of them")
    ap.add_argument("--out", type=Path, help="default: results/<today>/boundary")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo when all three outputs exist")
    args = ap.parse_args()
    if not hasattr(shapely, "orient_polygons"):
        raise SystemExit(f"aotil/lobes.py needs shapely >= 2.1 (orient_polygons); this is "
                         f"{shapely.__version__}")

    name = args.regions.name
    if not name.endswith("_points.parquet"):
        raise SystemExit(f"{name}: expected <cell>_<variant>_S<S>_points.parquet")
    prefix = name[: -len("_points.parquet")]
    comps_p = args.components or args.regions.with_name(f"{prefix}_components.csv")
    args.idx = args.idx or PREP_DIR / f"idx_{args.state}.parquet"
    args.haz = args.idx.parent / f"haz_{args.state}_5070.parquet"
    out = args.out or Path("results") / date.today().isoformat() / "boundary"
    paths = {k: out / f"{prefix}_{k}.parquet" for k in ("lobes", "vertices", "bboxes")}
    if not args.regions.exists():
        raise SystemExit(f"{args.regions} missing")

    t0 = time.time()
    print(f"=== lobes {prefix} ===", flush=True)
    pts, big = read_regions(args.regions, comps_p, args.min_size)
    print(f"  {len(pts)} selected points in {pts.component.nunique()} components; "
          f"{len(big)} of >= {args.min_size} points: {big}", flush=True)
    if args.only_components:
        bad = sorted(set(args.only_components) - set(big))
        if bad:
            raise SystemExit(f"--only-components {bad}: not components of >= {args.min_size} "
                             f"points (those are {big})")
        big = [c for c in big if c in set(args.only_components)]
        print(f"  --only-components: building {big} only", flush=True)
    if not big:
        raise SystemExit("no component is large enough; nothing to do")
    if all(p.exists() for p in paths.values()) and not args.overwrite:
        # The file names do not say which components were built, so skip only when the
        # existing bboxes table (one row per component) holds exactly these.
        have = set(pq.read_table(paths["bboxes"], columns=["component"])
                   .column("component").to_pylist())
        if have != set(big):
            raise SystemExit(f"{prefix}: outputs in {out} hold components {sorted(have)}, not "
                             f"{big}; use another --out, or --overwrite to replace them")
        print(f"{prefix}: outputs exist in {out}; skipping (--overwrite to redo)")
        return 0
    for p in (args.idx, args.haz):
        if not p.exists():
            raise SystemExit(f"{p} missing")
    sp.PREP_DIR = args.idx.parent       # read_geoms opens <PREP_DIR>/haz_<ST>_5070.parquet
    xy, xy_note = point_xy(pts, args.dataset)
    print(f"  point coordinates: {xy_note}", flush=True)
    idx = pd.read_parquet(args.idx, columns=IDX_COLS)
    if not np.array_equal(idx.row_id.to_numpy(), np.arange(len(idx))):
        raise SystemExit(f"{args.idx.name}: row_id is not 0..n-1, which read_geoms' "
                         f"row-group lookup needs")

    lobes, plans, geoms = build(pts, xy, big, idx, args)
    print(f"  {len(lobes):,} lobes built ({time.time() - t0:.0f}s); densifying to "
          f"{args.densify_m:g} m", flush=True)
    vt, checks = vertices_and_checks(lobes, plans, args)
    bb, region_only = bbox_table(lobes, plans, args.bbox_buffer_m)
    vt, note = contacts(lobes, vt, bb, idx, geoms, args)
    print(f"  contacts: {note}", flush=True)
    summarise(lobes, vt, bb, region_only, plans, checks, args, pts)

    out.mkdir(parents=True, exist_ok=True)
    tables = dict(lobes=gpd.GeoDataFrame(lobes, geometry="geometry", crs=CRS_WORK),
                  vertices=vt, bboxes=bb)
    for k, table in tables.items():
        table.to_parquet(paths[k], index=False)
        size_mb = paths[k].stat().st_size / 1e6
        print(f"  wrote {paths[k]} ({len(table):,} rows, {size_mb:.1f} MB)")
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
