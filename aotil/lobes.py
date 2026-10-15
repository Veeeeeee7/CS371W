#!/usr/bin/env python3
"""
Lobes and the edge-vertex table: the polygon boundaries the G1 boundary analysis
classifies (docs/REGION-ANALYSIS-FLORIDA.md section 3.3 steps 1-2,
docs/TASK-DATA-AUDIT.md D2).

WHAT A LOBE IS
--------------
One single-part polygon of the NFHL flood-hazard layer. G1's verdict ("this
stretch of Zone A edge was traced from the soil survey") is about a part drawn in
one pass, and its statistic is a share of edge length, so the unit is the
exploded part, not the multipart polygon a FLD_AR_ID names. For every region
component of >= --min-size points (aotil/regions.py) three sets of lobes are
built, each tagged with a `role`:

    region     Zone A parts that hold the region: parts of the polygons the
               region's positives were drawn in, plus parts of every Zone A
               polygon within --near-m of any region point, kept when they
               intersect the convex hull of the region's points buffered by
               --hull-buffer-m.
    AE         every part of a Zone AE polygon that intersects one of the
               region's 50 km tiles. The rule contrast: a detailed-study edge
               should be an elevation level set.
    matched_A  Zone A parts in the same tiles that hold no selected point,
               matched to the region lobes on perimeter. Region specificity: if
               they look like the region lobes, Zone A is drawn the same way
               everywhere.

DECISIONS, EACH WITH ITS REASON
-------------------------------
1. NOTHING IS CLIPPED. A lobe cut at the hull or at a tile edge gains a straight
   edge that no map ever drew, and a straight edge is exactly what an
   administrative line looks like to G1. So a lobe is either taken whole or not
   at all; the hull and the tiles only decide WHICH parts are taken.
2. THE HULL TEST IS ON PARTS, the polygon test is on whole polygons. An NFHL
   polygon can scatter parts tens of km apart; "within 2 km of a region point"
   selects the polygon, "intersects the buffered hull" drops its far parts.
   Every polygon that covers a region point is within 0 m of it, so the near
   rule already contains the positives' own polygons; they are still looked up
   explicitly, to report and check them (decision 7). The near rule can bring in
   many small polygons that hold no point: on v1, 3,126 Zone A polygons of median
   0.4 ha around component 0 carry 54% of its region edge. `from_positive` and
   `n_points` keep them separable downstream.
3. TILES ARE THE SELECTION'S. tile_50km is (bx // 5, by // 5) of the 10 km
   EPSG:5070 blocks (pull_gee.derive), so tile (tx, ty) covers x in
   [tx * 50 km, (tx + 1) * 50 km) and y likewise. A component's tiles are the
   tiles of its own points: the contrast lobes come from the ground the models
   were evaluated on. `tiles` records every tile a lobe intersects.
4. MATCHED LOBES, PRECISELY. Eligible: a Zone A part intersecting one of the
   component's tiles that covers (contains or has on its boundary) no selected
   point of ANY component of this selection, and is not a region lobe of any
   component. Matched on perimeter, because the null's power depends on edge
   length; on log perimeter, because perimeters span five orders of magnitude
   and a 10 km lobe is as far from a 1 km lobe as a 100 km one is from a 10 km
   one. Round-robin without replacement: in each of --n-matched rounds every
   region lobe, in a seeded random order, takes the eligible part nearest in log
   perimeter (ties broken by a seeded random key), so no lobe exhausts the good
   matches before the others get one. A large region lobe in a tile without a
   comparable part still gets the nearest one; `match_dlogp` records
   |log P_matched - log P_region| so a poor match is visible, not hidden. On v1
   the point-holding region lobes have 345-3,552 km of edge, and in three of the
   four components their own tiles hold nothing comparable (median |dlog P| of
   their matches 2.6-4.2, a factor of 13-65 in perimeter).
5. EVEN 30 m SPACING. Each ring (exterior and every hole) gets
   n = ceil(L / --densify-m) vertices at s = k L / n, k = 0..n-1, where s is the
   distance along the ring from its first coordinate; the original vertices are
   not kept. Equal spacing makes "share of edge length" a vertex count and makes
   10 vertices the 300 m window of section 3.3 step 5. Interpolation is done in
   numpy over all rings at once: GEOS's interpolate walks the ring from its start
   for every call, which on a 241,000-vertex swamp polygon is quadratic.
6. INWARD NORMALS ARE THOSE OF THE 30 m BOUNDARY. Every part is oriented
   (exterior counter-clockwise, holes clockwise), so the lobe's interior is on
   the LEFT of every ring, holes included, and the inward normal is the left
   perpendicular of the tangent. The tangent is the chord between the vertex's two neighbours
   on the densified ring (s - ds to s + ds, ds = L / n; L / 4 on rings of fewer
   than four vertices), i.e. the normal of the 30 m polyline G1 analyses, which
   at a corner is the bisector rather than one segment's normal. A chord of
   +-0.5 m was tried first and rejected: the lidar-era Indian River (12061C)
   Zone AE edges are traced in 0.2-0.5 m steps that zig-zag about a metre either
   side of their course, and there 11% of +-0.5 m normals were more than 45 deg
   off the 30 m direction (v1, component 1); a +-100 m elevation probe (D3) along
   such a normal samples the wrong ground. The lobes are written oriented, so the
   vertex table can be rebuilt from the GeoParquet.
       Checked in code on a seeded sample of vertices per component: x + 1 m * n
   must be inside the lobe and x - 1 m * n outside. The test is asserted where
   it is a statement about orientation: on vertices whose two probes are both
   >= 0.9 m from the lobe's boundary, i.e. where the edge runs straight past the
   vertex at the 1 m scale and the normal is within ~25 deg of it. There the run
   aborts unless >= 99.5% pass and <= 0.5% are flipped. On v1 (ten components of
   three selections) the worst component had 99.90% pass and 0.10% flipped,
   from vertices where the 30 m direction and a sub-metre zig-zag disagree; an
   orientation error flips whole rings, and holes alone carry 7-48% of a
   component's vertices, so any such error is far above both thresholds.
   Elsewhere (a lobe narrower than 2 m, a hole or a zig-zag within 1 m of the
   vertex) a probe pair can land on one side ("ambiguous") or straddle a
   different piece of edge; those vertices are counted and printed, not
   asserted.
7. SOURCE POLYGONS. A positive's polygon is meta_src_row_id when the regions
   table carries it (v2 sampler, note 8 of sample_points.py). Otherwise it is
   looked up by (dfirm_id, meta_fld_ar_id) in idx_<ST>; FLD_AR_ID alone repeats,
   and where even the pair names several rows (39 in Florida) the candidate that
   covers the point is taken. A band negative's meta_src_row_id names the ring it
   came from, not a polygon it lies in, so negatives contribute through the near
   rule only. Every resolved source polygon must be Zone A and should cover its
   positive; the count is printed, and a source polygon of another zone aborts:
   that would be an A->AE region, whose roles this script does not define.
8. ANALYSIS BBOX. The union of the extents of all of a component's lobes, every
   role, buffered by --bbox-buffer-m (3 km) so a 1 km displaced-edge null and the
   +-100 m normal probes stay inside every layer pulled for it (D3-D6). Its
   EPSG:4326 bounds are taken over the densified box edges, so the lon/lat box
   covers the EPSG:5070 box rather than only its corners.
9. GEOMETRY ACCESS. haz_<ST>_5070.parquet is 5.4 GB for Florida and is never
   read whole: candidate polygons are chosen on idx_<ST>'s bbox columns, then
   all components' candidates are read in one pass of sample_points.read_geoms
   (one row group at a time, each row group at most once). Decision 10 needs
   SFHA polygons that only the finished lobes and bboxes determine, so the ones
   not already read come in one second pass. Parts that fail OGC validity are
   repaired with make_valid, and only the polygonal pieces of the result are
   kept (make_valid can return a GeometryCollection).
10. CONTACTS ARE FLAGGED, NOT DROPPED. Not all of a lobe's boundary is an SFHA
   limit. Some of it is a contact with another SFHA polygon, a line no map drew
   as the edge of the floodplain: the floodway/fringe line inside Zone AE, a
   Zone A polygon split by source or at a sheet edge, an A/AE/AH/VE contact.
   An edge shared by two lobes of one role is also written twice, once per
   lobe, with opposite normals. G1 must measure only the SFHA-exterior edge,
   and must count a shared edge once. So every vertex carries
       contact_row_id, contact_zone, contact_subty
           the nearest other SFHA polygon (SFHA_TF 'T', any zone) within
           --contact-m, or null when there is none. "Other" means not the
           vertex's own polygon (src_row_id). Excluding the whole polygon
           rather than only the lobe's own part loses no shared edge: the
           parts of a valid MultiPolygon touch only at points. "Nearest" is by
           exact distance, and ties go to the lower row_id;
       same_role_contact
           another lobe of the same role and component lies within
           --contact-m (a sibling part of the vertex's own polygon counts:
           4 vertices on v1 S=500, where two parts come within 0.5 m);
   and G1 can take the exterior edge, the duplicates, or both, without a
   rerun. On v1 S=500, 24% of component 0's vertices and 87% of component 3's
   have a contact. 0.5 m is not a tuning knob: shared NFHL edges coincide to
   floating point, so in component 0 the share is 23.7% at 1 mm, 24.5% at 0.5 m
   and 25.2% at 2 m (the other components are just as flat); going from 1 mm
   to 0.5 m adds 5,861 vertices of 1,052,316 where neighbours misfit slightly,
   68% of them against another DFIRM's polygon. At exactly 0 m, round-off in
   the interpolated vertices loses half the contacts (13.2%). The candidates are the SFHA
   polygons whose idx bbox meets the component's analysis bbox (decision 8)
   and comes within --contact-m of one of its lobes' bboxes. They feed one
   STRtree over the component's vertices, queried with the polygons. Shapely
   prepares the query geometry, so testing a vertex against a 241,000-vertex
   polygon costs log n, not n. The other way round (vertices queried against
   a tree of polygons) took 41 times longer on v1 component 2's 157,024
   vertices (21.2 s against 0.5 s), and the gap grows with the vertex count.
11. FLOODWAYS ARE AE LOBES, MARKED. NFHL codes a floodway as FLD_ZONE 'AE' with
   a ZONE_SUBTY containing 'FLOODWAY' ('FLOODWAY', 'ADMINISTRATIVE FLOODWAY',
   'RIVERINE FLOODWAY ...'), so it enters the AE role with the fringe polygons
   around it. Its encroachment line is set by an allowed surcharge, not by an
   elevation, and it runs against the fringe, so it is already a contact
   (decision 10). Where a floodway itself reaches the SFHA limit, that edge is
   the 1% limit like any other. So floodway lobes are kept, and `is_floodway` /
   `zone_subty` let G1 drop them or report them on their own.

Scope: the AE->A direction (cell FL_band_AE: OOD = Zone A), as the paper uses.
Nothing written here is a model feature: these are per-lobe and per-vertex
tables for the boundary analysis, keyed by NFHL identifiers.

OUTPUTS (prefix = <cell>_<variant>_S<S>, taken from the regions file name)
-------
<out>/<prefix>_lobes.parquet     GeoParquet, EPSG:5070, one row per (component,
                                 role, lobe): lobe_id, zone, role, component,
                                 src_row_id, fld_ar_id, dfirm_id, area_m2,
                                 perimeter_m, matched_to (region lobe_id, for
                                 matched_A), geometry; plus part (index of the
                                 part in its polygon), tiles (';'-joined),
                                 n_points (this component's selected points the
                                 lobe covers), from_positive (region lobes: the
                                 polygon is a positive's source polygon),
                                 match_dlogp (matched_A), zone_subty (NFHL
                                 ZONE_SUBTY, blank -> null), is_floodway
                                 (decision 11)
<out>/<prefix>_vertices.parquet  one row per vertex: lobe_id, zone, role,
                                 component, ring (0 = exterior, k = k-th hole), s,
                                 x, y (EPSG:5070), lon, lat (EPSG:4326), nx, ny
                                 (unit inward normal); plus contact_row_id (idx
                                 row_id), contact_zone, contact_subty,
                                 same_role_contact (decision 10)
<out>/<prefix>_bboxes.parquet    one row per component: component, minx, miny,
                                 maxx, maxy (EPSG:5070, buffered), lon_min,
                                 lat_min, lon_max, lat_max (EPSG:4326), width_km,
                                 height_km, area_km2, n_lobes

USAGE
-----
    python aotil/lobes.py \\
        --regions results/<date>/regions/FL_band_AE_s0_spatial_mu3_knn10_S500_points.parquet \\
        --dataset data/processed/dataset_FL.parquet
    ... --components <..._components.csv>   (default: next to --regions)
    ... --out results/<date>/boundary       (default: results/<today>/boundary)
    ... --overwrite                         (redo when all three outputs exist)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import geopandas as gpd
import shapely
from shapely import STRtree
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sample_points as sp  # noqa: E402

PREP_DIR = Path("data/interim/prep")
CRS_WORK = 5070            # every distance, area and perimeter here is in it
CRS_LONLAT_IN = 4269       # the dataset's lon/lat (sample_points.CRS_SRC)
CRS_LONLAT_OUT = 4326      # what the vertex and bbox tables carry
TILE_M = 50_000.0          # tile_50km edge
LONLAT_TOL = 1e-9          # degrees; regions lon/lat are copies of the dataset's

REGION_ZONE, CONTRAST_ZONE = "A", "AE"     # AE->A: OOD = Zone A
ROLES = ("region", "matched_A", "AE")
FLOODWAY = "FLOODWAY"      # substring of every floodway ZONE_SUBTY (decision 11)

# Normal check (decision 6): probes at PROBE_M either side; asserted on vertices
# whose probes are both at least CLEAR_FRAC * PROBE_M from the lobe's boundary.
PROBE_M = 1.0
CLEAR_FRAC = 0.9
CHECK_MIN_PASS = 0.995
CHECK_MAX_FLIPPED = 0.005

REGION_COLS = ["point_id", "lon", "lat", "label", "component", "tile_50km",
               "meta_fld_ar_id", "dfirm_id"]
LOBE_COLS = ["lobe_id", "zone", "role", "component", "src_row_id", "fld_ar_id",
             "dfirm_id", "area_m2", "perimeter_m", "matched_to", "part", "tiles",
             "n_points", "from_positive", "match_dlogp", "zone_subty", "is_floodway"]
VERTEX_COLS = ["lobe_id", "zone", "role", "component", "ring", "s", "x", "y",
               "lon", "lat", "nx", "ny"]
CONTACT_COLS = ["contact_row_id", "contact_zone", "contact_subty", "same_role_contact"]
IDX_COLS = ["row_id", "DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "ZONE_SUBTY", "SFHA_TF",
            "minx", "miny", "maxx", "maxy"]


# ----------------------------------------------------------------- inputs
def read_regions(points_p: Path, comps_p: Path, min_size: int):
    """The regions table, checked against its components table."""
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
    if not want.reindex(sizes.index).eq(sizes).all() or len(want) != len(sizes):
        raise SystemExit(f"{points_p.name} and {comps_p.name} disagree on component sizes")
    return pts, [int(c) for c in sizes.index[sizes >= min_size]]


def point_xy(pts: pd.DataFrame, dataset: Path | None) -> tuple[np.ndarray, str]:
    """(x, y) in EPSG:5070 for every selected point: the dataset's own geometry
    when --dataset is given (the coordinates the sampler drew), else the regions
    lon/lat projected."""
    if dataset is None:
        tf = Transformer.from_crs(CRS_LONLAT_IN, CRS_WORK, always_xy=True)
        x, y = tf.transform(pts.lon.to_numpy(float), pts.lat.to_numpy(float))
        return (np.column_stack([x, y]),
                f"regions lon/lat projected EPSG:{CRS_LONLAT_IN} -> {CRS_WORK}")
    pf = pq.ParquetFile(dataset)
    geo = json.loads((pf.schema_arrow.metadata or {}).get(b"geo", b"{}") or b"{}")
    crs = (geo.get("columns", {}).get("geometry", {}).get("crs") or {}).get("id", {})
    if not (crs.get("authority") == "EPSG" and int(crs.get("code", -1)) == CRS_WORK):
        raise SystemExit(f"{dataset}: geometry CRS is {crs or 'unstated'}, not EPSG:{CRS_WORK}")
    d = pq.read_table(dataset, columns=["point_id", "lon", "lat", "geometry"]).to_pandas()
    d = d.set_index("point_id")
    absent = ~pts.point_id.isin(d.index)
    if absent.any():
        raise SystemExit(f"{int(absent.sum())} selected point_ids are not in {dataset.name}; "
                         f"wrong --dataset?")
    d = d.loc[pts.point_id]
    dl = max(float(np.abs(d.lon.to_numpy(float) - pts.lon.to_numpy(float)).max()),
             float(np.abs(d.lat.to_numpy(float) - pts.lat.to_numpy(float)).max()))
    if dl > LONLAT_TOL:
        raise SystemExit(f"lon/lat of the regions and {dataset.name} differ by up to {dl:.1e} deg")
    xy = shapely.get_coordinates(shapely.from_wkb(d.geometry.to_numpy()))
    return xy, f"{dataset.name} geometry (EPSG:{CRS_WORK}); lon/lat agree to {dl:.0e} deg"


def source_candidates(pts: pd.DataFrame, idx: pd.DataFrame, among: np.ndarray) -> tuple[dict, str]:
    """Candidate source row_ids per positive in `among`, keyed by row position in
    pts (decision 7): one when known, several when the (dfirm_id, meta_fld_ar_id)
    key repeats."""
    pos = np.flatnonzero((pts.label.to_numpy() == 1) & among)
    src = pts.meta_src_row_id.astype("Int64")
    cand = {int(i): [int(src.iloc[i])] for i in pos if pd.notna(src.iloc[i])}
    n_direct = len(cand)
    rest = [i for i in pos if i not in cand and pd.notna(pts.meta_fld_ar_id.iloc[i])]
    n_key = n_rep = 0
    if rest:
        key = pd.DataFrame(dict(i=rest, DFIRM_ID=pts.dfirm_id.iloc[rest].astype(str).to_numpy(),
                                FLD_AR_ID=pts.meta_fld_ar_id.iloc[rest].astype(str).to_numpy()))
        keys = idx[["row_id", "DFIRM_ID", "FLD_AR_ID"]].astype({"DFIRM_ID": str, "FLD_AR_ID": str})
        hit = key.merge(keys, on=["DFIRM_ID", "FLD_AR_ID"], how="inner")
        for i, g in hit.groupby("i"):
            cand[int(i)] = g.row_id.astype(int).tolist()
            n_key += 1
            n_rep += len(g) > 1
    n_none = len(pos) - len(cand)
    return cand, (f"{n_direct} from meta_src_row_id, {n_key} by (dfirm_id, meta_fld_ar_id) "
                  f"({n_rep} with a repeated key), {n_none} with neither")


def resolve_sources(cand: dict, xy: np.ndarray, geoms: dict) -> tuple[dict, int, int]:
    """One source row_id per positive; returns (point -> row_id, n covering, n unresolved)."""
    out, n_cover, n_unres = {}, 0, 0
    for i, rids in cand.items():
        pt = shapely.Point(xy[i])
        covers = [r for r in rids if geoms.get(r) is not None and geoms[r].covers(pt)]
        if len(rids) == 1:
            out[i] = rids[0]
            n_cover += len(covers) == 1
        elif len(covers) == 1:
            out[i] = covers[0]
            n_cover += 1
        else:
            n_unres += 1
    return out, n_cover, n_unres


def tile_box(tile: str) -> tuple[float, float, float, float]:
    tx, ty = (int(v) for v in str(tile).split("_"))
    return tx * TILE_M, ty * TILE_M, (tx + 1) * TILE_M, (ty + 1) * TILE_M


def bbox_hits(idx: pd.DataFrame, x0, y0, x1, y1) -> np.ndarray:
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
    """Every polygon's single parts, oriented (decision 6); invalid parts repaired."""
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
    b = shapely.bounds(geoms)
    t0 = np.floor(b[:, :2] / TILE_M).astype(np.int64)
    t1 = np.floor(b[:, 2:] / TILE_M).astype(np.int64)
    li, tx, ty = [], [], []
    for i in range(len(geoms)):
        for a in range(t0[i, 0], t1[i, 0] + 1):
            for c in range(t0[i, 1], t1[i, 1] + 1):
                li.append(i); tx.append(a); ty.append(c)
    li, tx, ty = np.array(li, int), np.array(tx), np.array(ty)
    hit = shapely.intersects(geoms[li], shapely.box(tx * TILE_M, ty * TILE_M,
                                                    (tx + 1) * TILE_M, (ty + 1) * TILE_M))
    out = [[] for _ in geoms]
    for i, a, c in zip(li[hit], tx[hit], ty[hit]):
        out[i].append(f"{a}_{c}")
    return [";".join(t) for t in out]


def densify(geoms: np.ndarray, step: float) -> dict:
    """Evenly spaced vertices and unit inward normals on every ring (decisions 5-6).

    geoms are oriented Polygons. All rings are laid end to end on one cumulative
    length axis, so every interpolation is one searchsorted over all segments.
    Segments are not copied out: ring r's coordinates are contiguous and closed,
    so global segment j of ring r starts at coordinate j + r. That keeps the peak
    at about six floats per coordinate (Indian River's lidar-traced AE edges put
    6.4 M coordinates into one v1 component's lobes).
    """
    rings, gi = shapely.get_rings(geoms, return_index=True)
    first_ring = np.searchsorted(gi, np.arange(len(geoms)))
    ring_no = np.arange(len(rings)) - first_ring[gi]
    xy = shapely.get_coordinates(rings)
    nc = shapely.get_num_coordinates(rings)
    ns = np.maximum(nc - 1, 0)                           # segments per ring
    s0 = np.cumsum(ns) - ns                              # its first global segment
    seg = np.hypot(*np.diff(xy, axis=0).T)
    seg = np.delete(seg, (np.cumsum(nc) - 1)[:-1])       # drop the steps between rings
    cum = np.concatenate([[0.0], np.cumsum(seg)])        # cum[j] = start of segment j
    base, L = cum[s0], cum[s0 + ns] - cum[s0]
    r_all = np.arange(len(rings))

    n = np.where(L > 0, np.maximum(np.ceil(L / step), 1), 0).astype(np.int64)
    vr = np.repeat(r_all, n)
    k = np.arange(len(vr)) - np.repeat(np.cumsum(n) - n, n)
    s = k * (L[vr] / n[vr])

    def at(t):
        """Point at ring position t (wrapped) and the start coordinate of its segment."""
        g = base[vr] + np.mod(t, L[vr])
        j = np.clip(np.searchsorted(cum, g, side="right") - 1, s0[vr], s0[vr] + ns[vr] - 1)
        f = np.clip((g - cum[j]) / np.where(seg[j] > 0, seg[j], 1.0), 0.0, 1.0)
        c = j + vr
        return xy[c] + f[:, None] * (xy[c + 1] - xy[c]), c

    p, c = at(s)
    h = np.minimum(L[vr] / n[vr], L[vr] / 4)             # the neighbouring vertices
    tan = at(s + h)[0] - at(s - h)[0]
    norm = np.hypot(*tan.T)
    flat = norm < 1e-9                                   # a spike the chord cannot see
    tan[flat] = (xy[c + 1] - xy[c])[flat]
    norm = np.hypot(*tan.T)
    tan /= np.where(norm > 0, norm, 1)[:, None]
    return dict(lobe=gi[vr], ring=ring_no[vr].astype(np.int32), s=s,
                x=p[:, 0], y=p[:, 1], nx=-tan[:, 1], ny=tan[:, 0])


def check_normals(geoms: np.ndarray, bounds: np.ndarray, v: dict, sel: np.ndarray) -> dict:
    """Decision 6's probe test on the vertices `sel` (indices into v). `bounds`
    are the lobes' prepared boundaries; `clear` vertices have both probes at
    least CLEAR_FRAC * PROBE_M from them."""
    li = v["lobe"][sel]
    x, y, nx, ny = v["x"][sel], v["y"][sel], v["nx"][sel], v["ny"][sel]
    pin = shapely.points(x + PROBE_M * nx, y + PROBE_M * ny)
    pout = shapely.points(x - PROBE_M * nx, y - PROBE_M * ny)
    inn = shapely.contains(geoms[li], pin)
    out = shapely.contains(geoms[li], pout)
    clear = ~(shapely.dwithin(bounds[li], pin, CLEAR_FRAC * PROBE_M)
              | shapely.dwithin(bounds[li], pout, CLEAR_FRAC * PROBE_M))
    ok, flipped = inn & ~out, ~inn & out
    return dict(n=len(sel), ok=int(ok.sum()), flipped=int(flipped.sum()),
                ambiguous=int((inn == out).sum()), n_clear=int(clear.sum()),
                ok_clear=int((ok & clear).sum()), flipped_clear=int((flipped & clear).sum()))


def match_perimeter(reg_logp: np.ndarray, cand_logp: np.ndarray, n_rounds: int,
                    rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """Decision 4. Returns (region index, candidate index, |dlog P|) per match."""
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
    tf = Transformer.from_crs(CRS_WORK, CRS_LONLAT_OUT, always_xy=True)
    return tf.transform_bounds(x0, y0, x1, y1, densify_pts=101)


# ----------------------------------------------------------------- build
def plan_component(c: int, pts: pd.DataFrame, xy: np.ndarray, idx: pd.DataFrame,
                   a) -> dict:
    """Which polygons each role may draw from, on idx bbox columns alone."""
    m = pts.component.to_numpy() == c
    p = xy[m]
    tiles = sorted(pts.tile_50km[m].astype(str).unique())
    zone = idx.FLD_ZONE.to_numpy()
    near_box = (p[:, 0].min() - a.near_m, p[:, 1].min() - a.near_m,
                p[:, 0].max() + a.near_m, p[:, 1].max() + a.near_m)
    near = (zone == REGION_ZONE) & bbox_hits(idx, *near_box)
    in_tiles = np.zeros(len(idx), bool)
    for t in tiles:
        in_tiles |= bbox_hits(idx, *tile_box(t))
    rid = idx.row_id.to_numpy()
    return dict(mask=m, xy=p, tiles=tiles,
                hull=shapely.buffer(shapely.convex_hull(shapely.multipoints(p)), a.hull_buffer_m),
                near_ids=rid[near], A_tile_ids=rid[in_tiles & (zone == REGION_ZONE)],
                AE_tile_ids=rid[in_tiles & (zone == CONTRAST_ZONE)])


def build(pts: pd.DataFrame, xy: np.ndarray, big: list[int], idx: pd.DataFrame, a):
    t0 = time.time()
    cand, cand_note = source_candidates(pts, idx, pts.component.isin(big).to_numpy())
    print(f"  positives' source polygons (components of >= {a.min_size} points): {cand_note}",
          flush=True)
    zone = dict(zip(idx.row_id.to_numpy().tolist(), idx.FLD_ZONE.to_numpy().tolist()))
    other = sorted({r for rids in cand.values() if all(zone[r] != REGION_ZONE for r in rids)
                    for r in rids})
    if other:
        raise SystemExit(f"{len(other)} source polygons of region positives are not Zone "
                         f"{REGION_ZONE} (e.g. row_id {other[0]}: {zone[other[0]]}); lobes.py "
                         f"handles AE->A regions only")
    plans = {c: plan_component(c, pts, xy, idx, a) for c in big}

    need = {r for rids in cand.values() for r in rids}
    for pl in plans.values():
        for k in ("near_ids", "A_tile_ids", "AE_tile_ids"):
            need |= set(pl[k].tolist())
    need = np.array(sorted(need), dtype=np.int64)
    print(f"  reading {len(need):,} candidate polygons from "
          f"{a.haz.name} ({len(np.unique(need // sp.RG_ROWS))} row groups)", flush=True)
    g = sp.read_geoms(a.state, need)
    missing = int(sum(x is None for x in g))
    if missing:
        raise SystemExit(f"{missing} candidate row_ids have no geometry in {a.haz}")
    geoms = dict(zip(need.tolist(), g))
    meta = idx.set_index("row_id")[["FLD_AR_ID", "DFIRM_ID", "ZONE_SUBTY"]]
    print(f"    read in {time.time() - t0:.1f}s", flush=True)

    src, n_cover, n_unres = resolve_sources(cand, xy, geoms)
    bad_zone = sorted({r for r in src.values() if zone[r] != REGION_ZONE})
    if bad_zone:
        raise SystemExit(f"{len(bad_zone)} resolved source polygons are not Zone {REGION_ZONE} "
                         f"(e.g. row_id {bad_zone[0]}: {zone[bad_zone[0]]})")
    print(f"  {len(src)} positives with a source polygon: {n_cover} covered by it, "
          f"{len(src) - n_cover} not; {n_unres} unresolved", flush=True)

    parts, n_fixed = explode(geoms, zone)
    pg = geom_array(parts.geometry)
    shapely.prepare(pg)
    tree = STRtree(pg)
    print(f"  {len(parts):,} single parts from {len(geoms):,} polygons"
          + (f"; {n_fixed} invalid parts repaired" if n_fixed else ""), flush=True)

    # Which parts cover which selected points, for every component (decision 4).
    pi, pj = tree.query(shapely.points(xy), predicate="covered_by")
    covers_any = np.zeros(len(parts), bool)
    covers_any[pj] = True
    comp_of = pts.component.to_numpy()
    key = pd.MultiIndex.from_arrays([parts.row_id, parts.part])

    region, tile_parts = {}, {}
    for c, pl in plans.items():
        mine = [i for i in np.flatnonzero(pl["mask"]) if i in src]
        from_pos = {src[i] for i in mine}
        near_geoms = np.array([geoms[r] for r in pl["near_ids"]], dtype=object)
        hit = np.empty(0, dtype=np.int64)
        if len(near_geoms):
            hit = STRtree(near_geoms).query(shapely.points(pl["xy"]), predicate="dwithin",
                                            distance=a.near_m)[1]
        near_set = set(pl["near_ids"][np.unique(hit)].tolist())
        polys = from_pos | near_set
        in_poly = parts.row_id.isin(polys).to_numpy()
        on_hull = np.zeros(len(parts), bool)
        on_hull[tree.query(pl["hull"], predicate="intersects")] = True
        region[c] = np.flatnonzero(in_poly & on_hull)
        boxes = shapely.box(*np.array([tile_box(t) for t in pl["tiles"]]).T)
        tp = np.unique(tree.query(boxes, predicate="intersects")[1])
        tile_parts[c] = tp
        pl.update(from_pos=from_pos, n_near=len(near_set), n_polys=len(polys))

    region_keys = set(key[np.concatenate(list(region.values()))]) if region else set()
    is_region_any = np.array([k in region_keys for k in key])
    logp = np.log(shapely.length(pg))
    part_zone = parts.zone.to_numpy()

    lobes = []
    for c, pl in plans.items():
        rng = np.random.default_rng([a.seed, c])
        n_here = np.bincount(pj[comp_of[pi] == c], minlength=len(parts))
        tp = tile_parts[c]
        ae = tp[part_zone[tp] == CONTRAST_ZONE]
        elig = tp[(part_zone[tp] == REGION_ZONE) & ~covers_any[tp] & ~is_region_any[tp]]
        reg = region[c]
        matches = match_perimeter(logp[reg], logp[elig], a.n_matched, rng)

        def rows(ix, role, **extra):
            out = pd.DataFrame(dict(
                src_row_id=parts.row_id.to_numpy()[ix], part=parts.part.to_numpy()[ix],
                zone=part_zone[ix], geometry=pg[ix], n_points=n_here[ix]))
            out["role"] = role
            out["component"] = c
            out["lobe_id"] = [f"c{c}_{role}_{r}_{k}" for r, k in zip(out.src_row_id, out.part)]
            for k, v in extra.items():
                out[k] = v
            return out

        r_df = rows(reg, "region",
                    from_positive=np.isin(parts.row_id.to_numpy()[reg], list(pl["from_pos"])))
        m_ix = np.array([elig[j] for _, j, _ in matches], dtype=np.int64)
        m_df = rows(m_ix, "matched_A",
                    matched_to=[r_df.lobe_id.iloc[i] for i, _, _ in matches],
                    match_dlogp=[d for _, _, d in matches])
        e_df = rows(ae, "AE")
        by = ["src_row_id", "part"]
        lobes += [r_df.sort_values(by), m_df, e_df.sort_values(by)]
        pl.update(n_region=len(r_df), n_matched=len(m_df), n_eligible=len(elig), n_ae=len(e_df),
                  dlogp=np.array([d for _, _, d in matches]))

    lobes = pd.concat(lobes, ignore_index=True)
    lobes["fld_ar_id"] = meta.FLD_AR_ID.reindex(lobes.src_row_id).to_numpy()
    lobes["dfirm_id"] = meta.DFIRM_ID.reindex(lobes.src_row_id).to_numpy()
    lobes["zone_subty"] = subty(meta.ZONE_SUBTY.reindex(lobes.src_row_id)).array
    fw = lobes.zone_subty.str.contains(FLOODWAY, regex=False)
    lobes["is_floodway"] = fw.fillna(False).astype(bool)
    lg = geom_array(lobes.geometry)
    lobes["area_m2"] = shapely.area(lg)
    lobes["perimeter_m"] = shapely.length(lg)
    lobes["tiles"] = lobe_tiles(lg)
    for col, fill in (("matched_to", None), ("match_dlogp", np.nan), ("from_positive", pd.NA)):
        if col not in lobes:
            lobes[col] = fill
    lobes["from_positive"] = lobes.from_positive.astype("boolean")
    lobes = lobes.astype({"component": np.int32, "src_row_id": np.int64, "part": np.int32,
                          "n_points": np.int32})
    if not lobes.lobe_id.is_unique:
        raise SystemExit("lobe_id is not unique; a part was taken twice in one role")
    return lobes[LOBE_COLS + ["geometry"]], plans, geoms


def vertices_and_checks(lobes: pd.DataFrame, plans: dict, a) -> tuple[pd.DataFrame, dict]:
    lg = geom_array(lobes.geometry)
    shapely.prepare(lg)
    v = densify(lg, a.densify_m)
    bounds = shapely.boundary(lg)
    shapely.prepare(bounds)
    checks = {}
    comp = lobes.component.to_numpy()[v["lobe"]]
    role = lobes.role.to_numpy()[v["lobe"]]
    for c in plans:
        rng = np.random.default_rng([a.seed, c, 1])
        here = np.flatnonzero(comp == c)
        sel = np.sort(rng.choice(here, min(a.check_n, len(here)), replace=False))
        res = {r: check_normals(lg, bounds, v, sel[role[sel] == r]) for r in ROLES}
        tot = {k: sum(x[k] for x in res.values()) for k in res["region"]}
        res["all"] = tot
        checks[c] = res
        # An explicit raise rather than `assert`, so python -O cannot skip it.
        if not (tot["ok_clear"] >= CHECK_MIN_PASS * tot["n_clear"]
                and tot["flipped_clear"] <= CHECK_MAX_FLIPPED * tot["n_clear"]):
            raise AssertionError(
                f"component {c}: inward-normal check failed on clear edges: "
                f"{tot['ok_clear']}/{tot['n_clear']} pass, {tot['flipped_clear']} flipped "
                f"(need >= {CHECK_MIN_PASS:.1%} and <= {CHECK_MAX_FLIPPED:.1%})")

    tf = Transformer.from_crs(CRS_WORK, CRS_LONLAT_OUT, always_xy=True)
    lon, lat = tf.transform(v["x"], v["y"])
    vt = pd.DataFrame(dict(
        lobe_id=lobes.lobe_id.to_numpy()[v["lobe"]], zone=lobes.zone.to_numpy()[v["lobe"]],
        role=role, component=comp.astype(np.int32), ring=v["ring"], s=v["s"],
        x=v["x"], y=v["y"], lon=lon, lat=lat, nx=v["nx"], ny=v["ny"]))
    return vt[VERTEX_COLS], checks


def bbox_table(lobes: pd.DataFrame, plans: dict, buf: float) -> tuple[pd.DataFrame, dict]:
    b = shapely.bounds(geom_array(lobes.geometry))
    rows, alt = [], {}
    for c in plans:
        m = lobes.component.to_numpy() == c
        x0, y0 = b[m, 0].min() - buf, b[m, 1].min() - buf
        x1, y1 = b[m, 2].max() + buf, b[m, 3].max() + buf
        lo0, la0, lo1, la1 = lonlat_bounds(x0, y0, x1, y1)
        rows.append(dict(component=c, minx=x0, miny=y0, maxx=x1, maxy=y1, lon_min=lo0,
                         lat_min=la0, lon_max=lo1, lat_max=la1, width_km=(x1 - x0) / 1e3,
                         height_km=(y1 - y0) / 1e3, area_km2=(x1 - x0) * (y1 - y0) / 1e6,
                         n_lobes=int(m.sum())))
        rm = m & (lobes.role.to_numpy() == "region")
        if rm.any():
            rx = (b[rm, 2].max() - b[rm, 0].min() + 2 * buf,
                  b[rm, 3].max() - b[rm, 1].min() + 2 * buf)
        else:
            rx = (np.nan, np.nan)
        alt[c] = dict(region_km=(rx[0] / 1e3, rx[1] / 1e3), region_km2=rx[0] * rx[1] / 1e6)
    return pd.DataFrame(rows).astype({"component": np.int32}), alt


def contact_candidates(idx: pd.DataFrame, lobes: pd.DataFrame, bb: pd.DataFrame,
                       tol: float) -> dict:
    """SFHA row_ids per component whose idx bbox meets the component's analysis
    bbox and comes within tol of one of its lobes' bboxes (decision 10)."""
    sf = idx[idx.SFHA_TF.to_numpy() == "T"]
    sbox = shapely.box(sf.minx.to_numpy(), sf.miny.to_numpy(), sf.maxx.to_numpy(),
                       sf.maxy.to_numpy())
    lb = shapely.bounds(geom_array(lobes.geometry)) + np.array([-tol, -tol, tol, tol])
    comp = lobes.component.to_numpy()
    out = {}
    for r in bb.itertuples():
        inb = np.flatnonzero(bbox_hits(sf, r.minx, r.miny, r.maxx, r.maxy))
        near = STRtree(shapely.box(*lb[comp == r.component].T)).query(
            sbox[inb], predicate="intersects")[0]
        out[int(r.component)] = sf.row_id.to_numpy()[inb[np.unique(near)]]
    return out


def contacts(lobes: pd.DataFrame, vt: pd.DataFrame, bb: pd.DataFrame, idx: pd.DataFrame,
             geoms: dict, a) -> tuple[pd.DataFrame, str]:
    """Decision 10's columns for every vertex; reads the candidate SFHA polygons
    that build() did not, in one more read_geoms pass."""
    t0 = time.time()
    tol = a.contact_m
    cand = contact_candidates(idx, lobes, bb, tol)
    every = np.unique(np.concatenate(list(cand.values())))
    new = np.setdiff1d(every, np.fromiter(geoms, dtype=np.int64, count=len(geoms)))
    g = sp.read_geoms(a.state, new)
    if any(x is None for x in g):
        raise SystemExit(f"{sum(x is None for x in g)} SFHA row_ids have no geometry in {a.haz}")
    geoms.update(zip(new.tolist(), g))

    li = pd.Index(lobes.lobe_id).get_indexer(vt.lobe_id)
    own = lobes.src_row_id.to_numpy()[li]
    role = lobes.role.to_numpy()
    lg = geom_array(lobes.geometry)
    lcomp, vcomp = lobes.component.to_numpy(), vt.component.to_numpy()
    x, y = vt.x.to_numpy(), vt.y.to_numpy()
    crow = np.full(len(vt), -1, dtype=np.int64)
    same = np.zeros(len(vt), dtype=bool)
    n_multi = 0
    for c, rows in cand.items():
        vi = np.flatnonzero(vcomp == c)
        pts = shapely.points(x[vi], y[vi])
        tree = STRtree(pts)
        pg = np.array([geoms[r] for r in rows], dtype=object)
        pj, k = tree.query(pg, predicate="dwithin", distance=tol)
        keep = rows[pj] != own[vi[k]]
        pj, k = pj[keep], k[keep]
        multi = np.bincount(k, minlength=len(vi))[k] > 1
        n_multi += len(np.unique(k[multi]))
        d = np.zeros(len(k))
        d[multi] = shapely.distance(pts[k[multi]], pg[pj[multi]])
        o = np.lexsort((rows[pj], d, k))
        k, pj = k[o], pj[o]
        first = np.r_[True, k[1:] != k[:-1]] if len(k) else np.zeros(0, bool)
        crow[vi[k[first]]] = rows[pj[first]]

        lm = np.flatnonzero(lcomp == c)
        lj, k = tree.query(lg[lm], predicate="dwithin", distance=tol)
        mine = li[vi[k]]
        same[vi[k[(lm[lj] != mine) & (role[lm[lj]] == role[mine])]]] = True

    has = crow >= 0
    attrs = idx.set_index("row_id")
    at = np.where(has, crow, 0)
    out = vt.assign(
        contact_row_id=pd.Series(crow, dtype="Int64").mask(~has).array,
        contact_zone=pd.Series(attrs.FLD_ZONE.reindex(at).to_numpy(), dtype="string")
        .mask(~has).array,
        contact_subty=subty(pd.Series(attrs.ZONE_SUBTY.reindex(at).to_numpy())).mask(~has).array,
        same_role_contact=same)
    note = (f"{len(every):,} SFHA polygons within {tol:g} m of a lobe bbox ({len(new):,} more "
            f"read); {int(has.sum()):,} of {len(vt):,} vertices have a contact, {n_multi:,} "
            f"of them several (nearest taken); {int(same.sum()):,} lie on another lobe of "
            f"their role ({time.time() - t0:.1f}s)")
    return out[VERTEX_COLS + CONTACT_COLS], note


# ----------------------------------------------------------------- report
def summarise(lobes, vt, bb, alt, plans, checks, a, pts) -> None:
    nv = vt.groupby(["component", "role"]).size()
    for c, pl in plans.items():
        L = lobes[lobes.component == c]
        n_pos = int((pts.label[pl["mask"]] == 1).sum())
        per = L.groupby("role").perimeter_m.sum() / 1e3
        print(f"\n--- component {c}: {int(pl['mask'].sum())} points ({n_pos} positive), "
              f"tiles {';'.join(pl['tiles'])}", flush=True)
        reg = L[L.role == "region"]
        near_only = reg[~reg.from_positive.fillna(False).astype(bool)]
        print(f"  region    {pl['n_region']:>6,} lobes from {reg.src_row_id.nunique():,} of "
              f"{pl['n_polys']:,} polygons ({len(pl['from_pos'])} positives' source polygons, "
              f"{pl['n_near']:,} within {a.near_m:g} m); perimeter {per.get('region', 0):,.0f} km, "
              f"{near_only.perimeter_m.sum() / 1e3:,.0f} km of it on {len(near_only):,} lobes of "
              f"near-only polygons; {int(reg.n_points.sum())} of {int(pl['mask'].sum())} "
              f"points inside")
        d = pl["dlogp"]
        want = a.n_matched * pl["n_region"]
        q = (f"; |dlog P| median {np.median(d):.3f}, max {d.max():.3f}" if len(d) else "")
        print(f"  matched_A {pl['n_matched']:>6,} lobes of {want:,} wanted ({a.n_matched} per "
              f"region lobe) from {pl['n_eligible']:,} eligible; perimeter "
              f"{per.get('matched_A', 0):,.0f} km{q}")
        fw = L[(L.role == "AE") & L.is_floodway]
        print(f"  AE        {pl['n_ae']:>6,} lobes; perimeter {per.get('AE', 0):,.0f} km, "
              f"{fw.perimeter_m.sum() / 1e3:,.0f} km of it on {len(fw):,} floodway lobes")
        counts = [int(nv.get((c, r), 0)) for r in ROLES]
        print(f"  vertices  " + " | ".join(f"{r} {k:,}" for r, k in zip(ROLES, counts))
              + f" | total {sum(counts):,}")
        V = vt[vt.component == c]
        for r in ROLES:
            w = V[V.role == r]
            if not len(w):
                continue
            hc = w.contact_row_id.notna().to_numpy()
            zs = w.contact_zone[hc].value_counts(normalize=True)
            fwc = w.contact_subty[hc].str.contains(FLOODWAY, regex=False).fillna(False).mean()
            print(f"  contacts  {r:<9} {1 - hc.mean():6.1%} SFHA-exterior; {hc.mean():.1%} on "
                  f"another SFHA polygon within {a.contact_m:g} m ("
                  + ", ".join(f"{z} {s:.0%}" for z, s in zs.head(4).items())
                  + f"; floodway {fwc if hc.any() else 0:.0%}); {w.same_role_contact.mean():.1%}"
                  f" on another {r} lobe")
        ch, t = checks[c], checks[c]["all"]
        print(f"  normals   {t['n']:,} sampled vertices, +-{PROBE_M:g} m probes: clear edges "
              f"{t['ok_clear']:,}/{t['n_clear']:,} pass "
              f"({t['ok_clear'] / max(t['n_clear'], 1):.3%}), "
              f"{t['flipped_clear']} flipped -> assertion passed; all sampled "
              f"{t['ok']:,}/{t['n']:,} pass ({t['ok'] / max(t['n'], 1):.2%}), "
              f"{t['ambiguous']:,} ambiguous, {t['flipped']:,} flipped")
        print("            by role (pass/n, clear pass/clear n): " + " | ".join(
            f"{r} {ch[r]['ok']:,}/{ch[r]['n']:,}, {ch[r]['ok_clear']:,}/{ch[r]['n_clear']:,}"
            for r in ROLES))
        r = bb[bb.component == c].iloc[0]
        rk = alt[c]
        print(f"  bbox      {r.width_km:.1f} x {r.height_km:.1f} km = {r.area_km2:,.0f} km2 "
              f"(lon {r.lon_min:.4f}..{r.lon_max:.4f}, lat {r.lat_min:.4f}..{r.lat_max:.4f}); "
              f"region lobes alone + {a.bbox_buffer_m / 1e3:g} km: {rk['region_km'][0]:.1f} x "
              f"{rk['region_km'][1]:.1f} km = {rk['region_km2']:,.0f} km2")
    boxes = shapely.box(bb.minx, bb.miny, bb.maxx, bb.maxy)
    print(f"\n  analysis bboxes: {len(bb)}, sum of areas {bb.area_km2.sum():,.0f} km2, area of "
          f"their union {shapely.union_all(boxes).area / 1e6:,.0f} km2; region-lobe-only boxes "
          f"would sum to {sum(v['region_km2'] for v in alt.values()):,.0f} km2", flush=True)


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
                         "contact (decision 10)")
    ap.add_argument("--check-n", type=int, default=20_000,
                    help="vertices per component sampled for the inward-normal check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="default: results/<today>/boundary")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if not hasattr(shapely, "orient_polygons"):
        raise SystemExit(f"aotil/lobes.py needs shapely >= 2.1 (orient_polygons); this is "
                         f"{shapely.__version__}")

    name = a.regions.name
    if not name.endswith("_points.parquet"):
        raise SystemExit(f"{name}: expected <cell>_<variant>_S<S>_points.parquet")
    prefix = name[: -len("_points.parquet")]
    comps_p = a.components or a.regions.with_name(f"{prefix}_components.csv")
    a.idx = a.idx or PREP_DIR / f"idx_{a.state}.parquet"
    a.haz = a.idx.parent / f"haz_{a.state}_5070.parquet"
    out = a.out or Path("results") / date.today().isoformat() / "boundary"
    paths = {k: out / f"{prefix}_{k}.parquet" for k in ("lobes", "vertices", "bboxes")}
    if all(p.exists() for p in paths.values()) and not a.overwrite:
        print(f"{prefix}: outputs exist in {out}; skipping (--overwrite to redo)")
        return 0
    for p in (a.regions, a.idx, a.haz):
        if not p.exists():
            raise SystemExit(f"{p} missing")
    sp.PREP_DIR = a.idx.parent       # read_geoms opens <PREP_DIR>/haz_<ST>_5070.parquet

    t0 = time.time()
    print(f"=== lobes {prefix} ===", flush=True)
    pts, big = read_regions(a.regions, comps_p, a.min_size)
    print(f"  {len(pts)} selected points in {pts.component.nunique()} components; "
          f"{len(big)} of >= {a.min_size} points: {big}", flush=True)
    if not big:
        raise SystemExit("no component is large enough; nothing to do")
    xy, xy_note = point_xy(pts, a.dataset)
    print(f"  point coordinates: {xy_note}", flush=True)
    idx = pd.read_parquet(a.idx, columns=IDX_COLS)
    if not np.array_equal(idx.row_id.to_numpy(), np.arange(len(idx))):
        raise SystemExit(f"{a.idx.name}: row_id is not 0..n-1, which read_geoms' "
                         f"row-group lookup needs")

    lobes, plans, geoms = build(pts, xy, big, idx, a)
    print(f"  {len(lobes):,} lobes built ({time.time() - t0:.0f}s); densifying to "
          f"{a.densify_m:g} m", flush=True)
    vt, checks = vertices_and_checks(lobes, plans, a)
    bb, alt = bbox_table(lobes, plans, a.bbox_buffer_m)
    vt, note = contacts(lobes, vt, bb, idx, geoms, a)
    print(f"  contacts: {note}", flush=True)
    summarise(lobes, vt, bb, alt, plans, checks, a, pts)

    out.mkdir(parents=True, exist_ok=True)
    tables = dict(lobes=gpd.GeoDataFrame(lobes, geometry="geometry", crs=CRS_WORK),
                  vertices=vt, bboxes=bb)
    for k, t in tables.items():
        t.to_parquet(paths[k], index=False)
        print(f"  wrote {paths[k]} ({len(t):,} rows, {paths[k].stat().st_size / 1e6:.1f} MB)")
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
