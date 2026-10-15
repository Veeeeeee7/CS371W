#!/usr/bin/env python3
"""
Distance from every lobe-edge vertex to the nearest line of each GIS layer: the
last data step of the G1 boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md
section 3.3 step 4; docs/TASK-DATA-AUDIT.md D7).

This script measures and checks coverage. It does NO analysis: no level-set
classification, no displaced-edge null, no statistics. Those read the table it
writes. Nothing here is a model feature: the table lives under results/ and is
never joined onto dataset_<ST>.parquet; FEATURES.txt is untouched.

INPUTS (prefix = <cell>_<variant>_S<S>, read off the --vertices file name)
------
--vertices  <prefix>_vertices_z.parquet   pull_dem.py sample: the aotil/lobes.py
                                          vertex table (with its contact columns)
                                          + z, z_in, z_out, dem_ok and the probe
                                          flags (decision 9)
--lobes     <prefix>_lobes.parquet        aotil/lobes.py (only lobe_id, component,
                                          role, zone, dfirm_id are read)
--bboxes    <prefix>_bboxes.parquet       aotil/lobes.py analysis bboxes (EPSG:5070)
--gis-dir   data/interim/gis, holding
              ssurgo/<prefix>_mulines.parquet      pull_ssurgo.py (current version:
                                                   water_edge, areaname_a/_b)
              ssurgo/<prefix>_mupolygons.parquet   pull_ssurgo.py (coverage check)
              county_lines_<ST>.parquet            pull_gis_lines.py
              counties_<ST>.parquet                pull_gis_lines.py (coverage, optional)
              panel_edges_<ST>.parquet             pull_gis_lines.py
              optional layers, found by file name (below)
--lobes and --bboxes default to the files beside --vertices.

LAYERS (source -> layers; --skip-layer takes a source or a layer name)
------
ssurgo  soil            mulines kind == 'soil_line', of the vertex's component: the
                        lines of class S
        soil_water      mulines kind == 'water_edge' (a shore), same component;
                        never part of class S
        soil_join_county  mulines kind == 'survey_join' between two county
                        survey areas, same component
        soil_join_other   mulines kind == 'survey_join' with federal or tribal
                        land on a side, same component
county  county          county_lines kind == 'county' (two in-state counties)
        state_or_coast  county_lines kind == 'state_or_coast'; kind column = outer
                        ('state' or 'coast')
panel   panel_interior  panel_edges of the LOBE'S OWN DFIRM, kind 'interior' or
                        'overlap'; kind column says which
        panel_exterior  panel_edges of the lobe's own DFIRM, kind 'exterior'
nwi     nwi             *nwi* or *wetland* file under --gis-dir (polygon boundaries or lines)
plss    plss            *plss* or *cadnsdi* file (polygon boundaries or lines)
roads   roads           *road* file (lines)
canals  canals          *canal* file, or NHDPlus V2 NHDFlowline FTYPE == 'CanalDitch'
                        with --canals-from-nhdplus (explicit; never a silent fallback)
The three first sources are required: a missing file fails the run unless the
source is named in --skip-layer. Optional layers are used when found; a missing
one is reported as 'absent'. --layer-file NAME=PATH points a source at a file
explicitly (for ssurgo, the mulines file; mupolygons is read beside it).

DECISIONS, EACH WITH ITS REASON
-------------------------------
1. EXACT DISTANCE TO THE LINE, NOT ITS VERTICES. Distances are Euclidean in
   EPSG:5070 to the nearest point of the nearest line. Lines are cut into pieces
   of at most PIECE_SEGS segments before they go into the STRtree: a 40 km soil
   line has a 40 km envelope, and every vertex near it would test the whole line
   otherwise. The distance to a line is the minimum over its pieces, so the cut
   changes nothing but speed. Polygon layers (NWI, PLSS) are measured to their
   boundaries (exterior and holes).
2. TIES GO TO THE LOWEST LINE ID. A vertex equidistant from two lines (on a
   T-junction, or on a line shared by two features) gets the one listed first in
   the layer file, so the id and the soil diff_attrs do not depend on tree order.
3. SOIL LINES, SHORES AND SURVEY JOINS ARE SEPARATE LAYERS (pull_ssurgo.py
   note 5). `soil` is kind 'soil_line' only, the line a soil scientist drew
   between two soils: class S. `soil_water` is 'water_edge'. SSURGO maps open
   water as map units ("Water", "Waters of the Gulf of America"), so a shore is
   a two-sided line across which nearly every attribute differs, yet nobody drew
   it between two soils. It is written so G1 can see how far a vertex is from a
   shore, and it is never mixed into class S. `soil_join_*` are 'survey_join',
   the edge of a soil survey area: an administrative line, across which the map
   units can differ merely because the two surveys were made at different times
   (a real soil change is possible too; n_diff and diff_attrs say what changes).
   They are split by what the two survey areas are (areaname_a / areaname_b),
   because they are different lines. soil_join_county joins two county surveys:
   in Florida a county line or the state line, the line class M takes from
   TIGER. soil_join_other has federal or tribal land on a side: a National
   Forest, Park or Preserve, the Water Conservation Area or a reservation (on a
   3 x 5 km box at Biscayne Bay all 16 joins are the Biscayne National Park
   boundary). Two layers, not one with a kind column, so that the nearer of the
   two never hides the other.
   A survey area is a county survey when its areaname says County or Counties
   and none of National, Reservation, Conservation, Refuge (COUNTY_AREA,
   NOT_COUNTY_AREA). Checked on 2026-10-14 against SDA sacatalog: of the 74
   Florida survey areas and the 162 of Georgia and Alabama, exactly the seven
   Florida federal and tribal ones (FL609, FL615, FL616, FL618, FL622, FL623,
   FL624) are not county surveys. "County" alone is not enough: "Big Cypress
   Reservation, Broward County, Florida". A missing areaname is not a county
   survey. <layer>_areanames names the two survey areas of the nearest join.
4. PANEL EDGES OF THE LOBE'S OWN MAP ONLY. FIRM panels overlap across county
   DFIRMs, so a neighbouring county's sheet edges run across a lobe's ground
   without having anything to do with how it was drawn (pull_gis_lines.py
   decision 4). Vertices are grouped by their lobe's dfirm_id and measured only
   against that DFIRM's edges. 'overlap' edges (a panel's side inside another
   panel of the same DFIRM) are sheet edges in the map's interior, so they join
   panel_interior and carry kind = 'overlap'.
5. NaN MEANS "NONE THERE", NOT "NOT MEASURED". A skipped layer has no columns at
   all. A NaN distance means the layer was loaded and its coverage passed, but it
   has no line of that kind for the vertex: a bbox inside one county survey has
   no soil_join_county, an inland bbox may have no soil_water, a one-panel DFIRM
   has no interior edge.
6. EXACT UP TO THE BBOX MARGIN. For layers pulled per bbox (SSURGO, optional
   layers, canals) every line inside the bbox is loaded, so a distance d is exact
   when d <= the vertex's distance to its bbox edge (>= 3 km by lobes.py's
   buffer). A larger d is an upper bound and the margin a lower bound. The count
   of such vertices is in the coverage table (`beyond_margin`); with 30 m and
   100 m coincidence bands it does not matter, but it is not hidden. County and
   panel layers are complete for the state and for each DFIRM, so they are exact.
7. COVERAGE IS CHECKED, AND A FAILURE STOPS THE RUN (exit 1, no table written;
   the coverage JSON is written either way). Per component:
     vertices  every vertex's lobe_id is in --lobes with the same component, role
               and zone; every vertex lies inside its component's bbox.
     dem       every vertex has finite z, z_in and z_out and dem_ok
               (--allow-dem-missing downgrades this to a warning).
     ssurgo    both files come from the current pull_ssurgo.py: mulines has
               areaname_a/_b and both store the bboxes they were built for (its
               note 9). An older output fails: it called shores soil lines and
               did not name the survey areas. Each component's stored bbox must
               contain this one to TOL_M; the prefix is the same for v1 and v2
               and the gis-dir is not dated, so a v1 file would otherwise be
               read for v2 vertices, with other component numbers. Then: the
               component has mupolygons and at least one meets the bbox; no
               clip_edge line (the edge of what was fetched) enters the bbox
               shrunk by TOL_M, i.e. the fetched polygons reach past every side
               of the bbox and the clip_edge lines enclose it. no_data_edge
               inside the bbox (SSURGO has no polygon there: open water beyond a
               survey) is allowed and reported; soil_ok marks the vertices that
               lie in no map unit (they are warned about, not failed).
     county    the extent of county_lines contains the bbox. If it does not (a
               bbox reaching past the state line or the offshore limit), it
               passes only if counties_<ST>.parquet exists and every vertex lies
               inside a county or within --state-tol-m of one: the state line and
               the seaward limit are themselves lines of the layer, so no line
               beyond them can be nearer to an in-state vertex. NFHL and TIGER
               draw the water boundaries a little differently (the St. Marys and
               Perdido rivers: up to ~215 m on v2), so vertices just across the
               line are warned about and flagged `outside_state`; their county
               distances are to this state's lines only. Beyond the tolerance the
               layer does not cover the vertex and the run fails.
     panel     only the lobe's own DFIRM is measured, so every lobe's dfirm_id
               must have edges (else FAIL). Vertices outside their DFIRM's panel
               footprint (its edges polygonized) are warned about and flagged
               `outside_own_panels`: the panel layer itself is complete, the
               polygon just runs past its own map sheets (on v2, one Hendry
               12051C Zone AE lobe reaches into Lake Okeechobee, mapped on
               neighbouring counties' panels). Their distance is to the nearest
               edge of their own DFIRM, i.e. its exterior.
     optional  the layer's extent (the file's declared bounds when it has them)
               contains the bbox.
8. CRS. Every layer must state a CRS; a layer not in EPSG:5070 is reprojected,
   and the coverage note says so.
9. INPUT COLUMNS PASS THROUGH UNTOUCHED. Every column of --vertices is written
   back in input order with its values and type unchanged. That includes the
   lobes.py contact columns (contact_row_id, contact_zone, contact_subty,
   same_role_contact: the vertex lies on another SFHA polygon) and pull_dem's
   probe flags (in_probe_inside, out_probe_outside, out_probe_in_any_lobe: where
   the +-100 m probes landed). Nothing here reads them or drops a vertex by
   them; G1 decides which vertices count. Their absence (vertices from an older
   lobes.py, or pull_dem sample run without the lobes file) is a warning (the
   'inputs' coverage row), because G1 then cannot make that choice. The input's
   parquet metadata (pull_dem's b"pull_dem") is carried over as well. An input
   column with a name this script writes is an error, not an overwrite. No DEM
   raster is opened here: z comes from the table, so pull_dem sharing one
   raster between prefixes with identical grids (a hardlink) changes nothing,
   and the search for optional layers skips <gis-dir>/dem.

OUTPUT
------
<out> (default results/<today>/boundary/<prefix>_vertex_layers.parquet): one row
per input vertex, in input order, with every input column (lobe_id, zone, role,
component, ring, s, x, y, lon, lat, nx, ny, contact_*, same_role_contact, z,
z_in, z_out, dem_ok, the probe flags, ...; decision 9) plus
    dfirm_id                          the lobe's DFIRM (from --lobes)
    soil_ok                           the vertex lies in a SSURGO map unit (within
                                      TOL_M); a water map unit counts
    d_<layer>_m                       distance (m, EPSG:5070) to the nearest line
    <layer>_id                        that line's id: SSURGO line_id; NHDPlus COMID;
                                      for county_lines / panel_edges / files without
                                      an id column, the row number in the file
    <layer>_kind                      state_or_coast: outer; panel_interior: kind;
                                      optional layers: their kind-like column
    soil_n_diff, soil_diff_attrs      muaggatt attributes that change across the
                                      nearest soil line
    soil_join_county_n_diff, ..._diff_attrs, ..._areanames   the same for the
    soil_join_other_n_diff, ..._diff_attrs, ..._areanames    nearest join, and
                                      its two survey areas ('A | B', sorted)
All of these are metadata on a results table, not model features.
The parquet metadata key b"edge_layers" holds the inputs, layer files, skips
and the coverage table. <out stem>_coverage.json holds the same coverage table.

USAGE
-----
    python aotil/edge_layers.py \\
        --vertices results/<D>/boundary/<prefix>_vertices_z.parquet \\
        --out results/<D>/boundary/<prefix>_vertex_layers.parquet
    ... --canals-from-nhdplus           canals from data/raw/nhdplus NHDFlowline
    ... --skip-layer county             run without a required source (say why)
    ... --skip-layer soil_water         drop one layer of a source
    ... --layer-file nwi=path/to/x.parquet
    ... --overwrite                     redo when the output exists
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely import STRtree
from pyproj import CRS, Transformer

CRS_WORK = 5070
CRS_NHD = 4269            # NHDPlus V2 shapefiles ship in NAD83 geographic
GIS_DIR = Path("data/interim/gis")
NHD_DIR = Path("data/raw/nhdplus")
RESULTS = Path("results")

PIECE_SEGS = 16           # segments per STRtree piece (decision 1)
TOL_M = 1.0               # coverage tolerance (decision 7)
QUERY_CHUNK = 250_000     # vertices per nearest-line query

VERTEX_COLS = ["lobe_id", "zone", "role", "component", "ring", "s", "x", "y",
               "lon", "lat", "nx", "ny"]
DEM_COLS = ["z", "z_in", "z_out", "dem_ok"]
CONTACT_COLS = ["contact_row_id", "contact_zone", "contact_subty", "same_role_contact"]
PROBE_COLS = ["in_probe_inside", "out_probe_outside", "out_probe_in_any_lobe"]
BBOX_COLS = ["component", "minx", "miny", "maxx", "maxy"]
LOBE_COLS = ["lobe_id", "component", "role", "zone", "dfirm_id"]

SSURGO_COLS = ["line_id", "component", "kind", "areaname_a", "areaname_b", "n_diff", "diff_attrs"]
SSURGO_KINDS = ("soil_line", "survey_join", "water_edge", "clip_edge", "no_data_edge")
SSURGO_META = b"pull_ssurgo"                  # its note 9: the bboxes it was built for
COUNTY_AREA = re.compile(r"\bCount(?:y|ies)\b")                          # decision 3
NOT_COUNTY_AREA = re.compile(r"\b(?:National|Reservation|Conservation|Refuge)\b")

SOURCES = {
    "ssurgo": ("soil", "soil_water", "soil_join_county", "soil_join_other"),
    "county": ("county", "state_or_coast"),
    "panel": ("panel_interior", "panel_exterior"),
    "nwi": ("nwi",),
    "plss": ("plss",),
    "roads": ("roads",),
    "canals": ("canals",),
}
REQUIRED = ("ssurgo", "county", "panel")
OPTIONAL_PATTERNS = {
    "nwi": ("*nwi*", "*wetland*"),
    "plss": ("*plss*", "*cadnsdi*"),
    "roads": ("*road*",),
    "canals": ("*canal*",),
}
GEO_EXT = (".parquet", ".gpkg", ".shp", ".geojson", ".fgb")
SEARCH_SKIP_DIRS = {"dem", "ssurgo", "cache", "__pycache__"}
OTHER_PREFIX = re.compile(r"_S\d+_")          # <cell>_<variant>_S<S>_... of another prefix
ID_CANDIDATES = ("line_id", "COMID", "LINEARID", "FRSTDIVID", "PLSSID", "OBJECTID", "id")
KIND_CANDIDATES = ("kind", "WETLAND_TYPE", "ATTRIBUTE", "MTFCC", "FCODE", "FTYPE")
BBOX_LIMITED = SOURCES["ssurgo"] + ("nwi", "plss", "roads", "canals")   # decision 6


# ----------------------------------------------------------------- coverage table
class Coverage:
    """One row per (component, layer) check; 'all' for checks that are not per
    component. Status: ok, warn, FAIL, skipped, absent."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, comp, layer: str, status: str, note: str = "", **num) -> None:
        self.rows.append(dict(component=comp, layer=layer, status=status, note=note, **num))

    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]

    def skip(self, layer: str, note: str) -> None:
        """A skipped layer: its own rows go (a FAIL stays: it is about the shared file)."""
        self.rows = [r for r in self.rows if r["layer"] != layer or r["status"] == "FAIL"]
        self.add("all", layer, "skipped", note)

    def update(self, comp, layer: str, **num) -> None:
        for r in self.rows:
            if r["component"] == comp and r["layer"] == layer:
                r.update(num)

    def print(self) -> None:
        print(f"\n  {'comp':>4}  {'layer':<16} {'status':<7} {'lines':>8} {'vertices':>9} "
              f"{'beyond':>7}  note", flush=True)
        for r in self.rows:
            ln = r.get("n_lines")
            nv = r.get("n_vertices")
            bm = r.get("beyond_margin")
            print(f"  {str(r['component']):>4}  {r['layer']:<16} {r['status']:<7} "
                  f"{'-' if ln is None else f'{ln:,}':>8} {'-' if nv is None else f'{nv:,}':>9} "
                  f"{'-' if bm is None else f'{bm:,}':>7}  {r['note']}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ----------------------------------------------------------------- geometry
def explode_lines(geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single LineStrings of lines, multilines and polygon boundaries, with the
    index of the feature each came from."""
    g = np.array(geom, dtype=object, copy=True)
    poly = np.isin(shapely.get_type_id(g), (3, 6))
    if poly.any():
        g[poly] = shapely.boundary(g[poly])
    parts, parent = shapely.get_parts(g, return_index=True)
    keep = np.isin(shapely.get_type_id(parts), (1, 2)) & ~shapely.is_empty(parts)
    return parts[keep], parent[keep]


def line_pieces(geom: np.ndarray, k: int = PIECE_SEGS) -> tuple[np.ndarray, np.ndarray]:
    """Every line cut into pieces of at most k segments (decision 1); returns the
    pieces and the index of the feature each belongs to. 2-D only."""
    parts, parent = explode_lines(geom)
    if not len(parts):
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64)
    xy, gi = shapely.get_coordinates(parts, return_index=True)
    n = np.bincount(gi, minlength=len(parts))
    start = np.cumsum(n) - n
    nseg = np.maximum(n - 1, 0)
    nch = np.where(nseg > 0, -(-nseg // k), 0)
    li = np.repeat(np.arange(len(parts)), nch)
    c = np.arange(len(li)) - np.repeat(np.cumsum(nch) - nch, nch)
    a0 = start[li] + c * k
    a1 = start[li] + np.minimum((c + 1) * k, nseg[li])
    cnt = a1 - a0 + 1
    ci = np.repeat(np.arange(len(li)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pieces = shapely.linestrings(xy[np.repeat(a0, cnt) + off], indices=ci)
    return pieces, parent[li]


def nearest(x: np.ndarray, y: np.ndarray, geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each (x, y) to the nearest feature of `geom` and its index in
    `geom`; NaN and -1 when `geom` holds no line. Ties go to the lowest index."""
    d = np.full(len(x), np.nan)
    j = np.full(len(x), -1, dtype=np.int64)
    if not len(x) or not len(geom):
        return d, j
    pieces, owner = line_pieces(geom)
    if not len(pieces):
        return d, j
    tree = STRtree(pieces)
    for s in range(0, len(x), QUERY_CHUNK):
        sl = slice(s, s + QUERY_CHUNK)
        (qi, ti), dist = tree.query_nearest(shapely.points(x[sl], y[sl]),
                                            return_distance=True, all_matches=True)
        own = owner[ti]
        o = np.lexsort((own, qi))
        qi, own, dist = qi[o], own[o], dist[o]
        first = np.r_[True, qi[1:] != qi[:-1]]
        d[s + qi[first]] = dist[first]
        j[s + qi[first]] = own[first]
    return d, j


def box_contains(outer, inner, tol: float = TOL_M) -> bool:
    """(minx, miny, maxx, maxy) `outer` contains `inner`, to tol metres."""
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol
            and outer[2] >= inner[2] - tol and outer[3] >= inner[3] - tol)


def fmt_box(b) -> str:
    return "(" + ", ".join(f"{v:,.0f}" for v in b) + ")"


# ----------------------------------------------------------------- layers
@dataclass
class LineSet:
    """One layer's lines: geometry (EPSG:5070), the id and kind written for the
    nearest one, extra per-line columns, and the group (component or dfirm_id)
    each line belongs to when vertices are matched by group."""
    name: str
    geom: np.ndarray
    ids: pd.Series
    kind: pd.Series | None = None
    extra: dict = field(default_factory=dict)
    group: np.ndarray | None = None
    by: str | None = None                     # vertex column matched to `group`


def nullable(s: pd.Series) -> pd.Series:
    s = s.reset_index(drop=True)
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64")
    if pd.api.types.is_float_dtype(s):
        return s.astype("Float64")
    return s.astype("string")


def read_geo(path: Path, columns: list[str] | None = None):
    """A GeoDataFrame in EPSG:5070, and a note when it had to be reprojected."""
    import geopandas as gpd
    if path.suffix == ".parquet":
        g = gpd.read_parquet(path, columns=None if columns is None else columns + ["geometry"])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = gpd.read_file(path, columns=columns)
    if g.crs is None:
        raise SystemExit(f"{path}: no CRS stated; cannot place it in EPSG:{CRS_WORK}")
    note = ""
    if g.crs.to_epsg() != CRS_WORK:
        note = f"reprojected from {g.crs.to_string()}"
        g = g.to_crs(CRS_WORK)
    return g, note


def declared_bounds(path: Path, gdf) -> tuple[tuple, str]:
    """The file's own extent in EPSG:5070: GeoParquet 'bbox' or the OGR extent
    when present, else the bounds of its features."""
    try:
        if path.suffix == ".parquet":
            geo = json.loads(pq.read_schema(path).metadata.get(b"geo", b"{}"))
            col = geo.get("columns", {}).get(geo.get("primary_column", "geometry"), {})
            bb, crs = col.get("bbox"), col.get("crs")
            if bb and len(bb) == 4:
                src = CRS.from_user_input(crs) if crs else CRS.from_epsg(4326)
                if src.to_epsg() != CRS_WORK:
                    bb = Transformer.from_crs(src, CRS_WORK, always_xy=True).transform_bounds(
                        *bb, densify_pts=101)
                return tuple(bb), "declared GeoParquet bbox"
        else:
            import pyogrio
            info = pyogrio.read_info(path, force_total_bounds=True)
            bb, crs = info.get("total_bounds"), info.get("crs")
            if bb is not None and crs:
                src = CRS.from_user_input(crs)
                if src.to_epsg() != CRS_WORK:
                    bb = Transformer.from_crs(src, CRS_WORK, always_xy=True).transform_bounds(
                        *bb, densify_pts=101)
                return tuple(bb), "declared OGR extent"
    except Exception:          # noqa: BLE001  any unreadable metadata -> feature bounds
        pass
    return tuple(gdf.total_bounds), "extent of the features"


def pick_column(cols, candidates) -> str | None:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def find_optional(gis_dir: Path, source: str, prefix: str, state: str) -> tuple[Path | None, str]:
    """The file of an optional layer under gis_dir, by name (case-insensitive).
    A file named <prefix>_... wins; files that look like another prefix's are
    ignored; two or more candidates are an error (use --layer-file)."""
    if not gis_dir.exists():
        return None, f"{gis_dir} not found"
    pats = OPTIONAL_PATTERNS[source]
    req = (f"county_lines_{state}", f"counties_{state}", f"panel_edges_{state}")
    hits = []
    for root, dirs, files in os.walk(gis_dir):
        dirs[:] = [d for d in dirs if d.lower() not in SEARCH_SKIP_DIRS]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() not in GEO_EXT or p.stem.startswith(req):
                continue
            if any(fnmatch.fnmatch(p.stem.lower(), pat) for pat in pats):
                hits.append(p)
    mine = [p for p in hits if p.name.startswith(prefix + "_")]
    others = [p for p in hits if p not in mine and OTHER_PREFIX.search(p.name)]
    rest = [p for p in hits if p not in mine and p not in others]
    use = mine or rest
    if not use:
        return None, (f"no file matching {'/'.join(pats)} under {gis_dir}"
                      + (f" (ignored other prefixes' files: {[p.name for p in others]})"
                         if others else ""))
    if len(use) > 1:
        raise SystemExit(f"{source}: several candidate files {sorted(str(p) for p in use)}; "
                         f"choose one with --layer-file {source}=PATH")
    return use[0], "found by file name"


# ----------------------------------------------------------------- inputs
def prefix_of(path: Path) -> str:
    for suf in ("_vertices_z.parquet", "_vertices.parquet"):
        if path.name.endswith(suf):
            return path.name[: -len(suf)]
    raise SystemExit(f"{path.name}: expected <prefix>_vertices_z.parquet (or pass --prefix)")


def written_by_us(col: str) -> bool:
    """A column name this script writes (decision 9)."""
    layers = [l for v in SOURCES.values() for l in v]
    return (col in ("dfirm_id", "soil_ok")
            or any(col == f"d_{l}_m" or col.startswith(f"{l}_") for l in layers))


def read_inputs(a, cov: Coverage) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The vertex table as read (written back unchanged, decision 9), a working
    copy of the columns used here, and the bboxes."""
    raw = pd.read_parquet(a.vertices).reset_index(drop=True)
    miss = [c for c in VERTEX_COLS + DEM_COLS if c not in raw.columns]
    if miss:
        raise SystemExit(f"{a.vertices}: missing columns {miss} (run pull_dem.py sample first)")
    clash = [c for c in raw.columns if written_by_us(c)]
    if clash:
        raise SystemExit(f"{a.vertices}: has columns {clash} that this script writes "
                         f"(an edge_layers output given as --vertices?)")
    vt = raw[VERTEX_COLS + DEM_COLS].copy()
    vt["component"] = vt["component"].astype(np.int32)
    if not np.isfinite(vt[["x", "y"]].to_numpy(float)).all():
        raise SystemExit(f"{a.vertices}: non-finite x or y")
    extra = [c for c in raw.columns if c not in VERTEX_COLS + DEM_COLS]
    absent = [f"{what} ({why})" for what, cols, why in
              (("lobes.py contact columns", CONTACT_COLS, "an older lobes.py wrote these vertices"),
               ("pull_dem probe flags", PROBE_COLS, f"pull_dem.py sample writes them only when "
                f"{a.prefix}_lobes.parquet is beside the vertices"))
              if not set(cols) <= set(raw.columns)]
    cov.add("all", "inputs", "warn" if absent else "ok",
            (f"absent: {'; '.join(absent)}; G1 cannot select vertices by them. "
             if absent else "") + f"carried through unchanged: {extra or 'nothing'}")

    lb = pq.read_table(a.lobes_p, columns=LOBE_COLS).to_pandas()
    if not lb.lobe_id.is_unique:
        raise SystemExit(f"{a.lobes_p}: lobe_id is not unique")
    lb["component"] = lb["component"].astype(np.int32)
    bb = pd.read_parquet(a.bboxes_p)
    miss = [c for c in BBOX_COLS if c not in bb.columns]
    if miss:
        raise SystemExit(f"{a.bboxes_p}: missing columns {miss}")
    bb = bb.astype({"component": np.int32}).set_index("component")
    if not bb.index.is_unique:
        raise SystemExit(f"{a.bboxes_p}: duplicate component ids")

    # vertices <-> lobes
    key = vt.lobe_id.astype(str)
    lbi = lb.assign(lobe_id=lb.lobe_id.astype(str)).set_index("lobe_id")
    known = key.isin(lbi.index).to_numpy()
    vt["dfirm_id"] = pd.array(lbi.dfirm_id.reindex(key).to_numpy(), dtype="string")
    bad_meta = np.zeros(len(vt), bool)
    lbs = lbi[["component", "role", "zone"]].astype(str)   # before reindex: an unknown key
    for c in lbs.columns:                                  # must not turn 0 into "0.0"
        ref = lbs[c].reindex(key[known]).to_numpy()
        bad_meta[known] |= ref != vt[c].to_numpy()[known].astype(str)
    comps = sorted(vt.component.unique().tolist())
    for c in comps:
        m = (vt.component == c).to_numpy()
        n_unk, n_meta = int((m & ~known).sum()), int((m & bad_meta).sum())
        n_nodf = int((m & known & vt.dfirm_id.isna().to_numpy()).sum())
        if c not in bb.index:
            cov.add(c, "vertices", "FAIL", f"no bbox for component {c} in {a.bboxes_p.name}",
                    n_vertices=int(m.sum()))
            continue
        r = bb.loc[c]
        x, y = vt.x.to_numpy()[m], vt.y.to_numpy()[m]
        out = int(((x < r.minx) | (x > r.maxx) | (y < r.miny) | (y > r.maxy)).sum())
        probs = [f"{n_unk:,} lobe_ids not in {a.lobes_p.name}" if n_unk else "",
                 f"{n_meta:,} disagree with their lobe on component/role/zone" if n_meta else "",
                 f"{n_nodf:,} lobes without dfirm_id" if n_nodf else "",
                 f"{out:,} outside the bbox" if out else ""]
        probs = [p for p in probs if p]
        roles = vt.role[m].value_counts().to_dict()
        cov.add(c, "vertices", "FAIL" if probs else "ok",
                "; ".join(probs) if probs else
                f"bbox {fmt_box((r.minx, r.miny, r.maxx, r.maxy))}; roles {roles}",
                n_vertices=int(m.sum()))

    # DEM values (decision 7)
    zf = np.isfinite(vt[["z", "z_in", "z_out"]].to_numpy(float)).all(axis=1)
    ok = vt.dem_ok.fillna(False).astype(bool).to_numpy() & zf
    for c in comps:
        m = (vt.component == c).to_numpy()
        n_bad = int((m & ~ok).sum())
        status = "ok" if not n_bad else ("warn" if a.allow_dem_missing else "FAIL")
        cov.add(c, "dem", status,
                f"z, z_in and z_out at all {int(m.sum()):,} vertices" if not n_bad else
                f"{n_bad:,} vertices lack z, z_in or z_out (dem_ok False)"
                + ("; kept (--allow-dem-missing)" if a.allow_dem_missing else
                   "; rerun pull_dem.py fetch/sample, or --allow-dem-missing"),
                n_vertices=int(m.sum()))
    return raw, vt, bb


# ----------------------------------------------------------------- per source
def county_area(name) -> bool:
    """A county soil survey area, by its areaname (decision 3)."""
    return (isinstance(name, str) and COUNTY_AREA.search(name) is not None
            and NOT_COUNTY_AREA.search(name) is None)


def ssurgo_bboxes(path: Path) -> dict | None:
    """{component: (minx, miny, maxx, maxy)} a pull_ssurgo output was built for
    (its note 9); None when the file holds no such record."""
    md = pq.read_schema(path).metadata or {}
    rec = json.loads(md.get(SSURGO_META, b"{}")).get("bboxes")
    if not rec:
        return None
    return {int(r["component"]): (r["minx"], r["miny"], r["maxx"], r["maxy"]) for r in rec}


def load_ssurgo(a, vt, bb, cov: Coverage) -> list[LineSet]:
    lpath = a.files.get("ssurgo") or a.gis_dir / "ssurgo" / f"{a.prefix}_mulines.parquet"
    ppath = lpath.with_name(lpath.name.replace("_mulines", "_mupolygons"))
    missing = [p for p in (lpath, ppath) if not p.exists()]
    if missing:
        cov.add("all", "soil", "FAIL", f"{', '.join(map(str, missing))} missing: run "
                f"pull_ssurgo.py --bboxes {a.bboxes_p}, or --skip-layer ssurgo")
        return []
    rerun = f"rerun pull_ssurgo.py --bboxes {a.bboxes_p} --yes --overwrite"
    lines, note = read_geo(lpath)
    miss = [c for c in SSURGO_COLS if c not in lines.columns]
    built = {p: ssurgo_bboxes(p) for p in (lpath, ppath)}
    norec = [p.name for p, r in built.items() if r is None]
    if miss or norec:
        cov.add("all", "soil", "FAIL", "written by an older pull_ssurgo.py ("
                + "; ".join(([f"{lpath.name} lacks {miss}"] if miss else [])
                            + ([f"no bbox record in {', '.join(norec)}"] if norec else []))
                + f"), which called shores soil lines and did not name the survey areas: {rerun}")
        return []
    if built[lpath] != built[ppath]:
        cov.add("all", "soil", "FAIL", f"{lpath.name} and {ppath.name} were built for different "
                f"bboxes: {rerun}")
        return []
    built = built[lpath]
    unknown = sorted(set(lines.kind.astype(str)) - set(SSURGO_KINDS))
    if unknown:
        raise SystemExit(f"{lpath}: unknown kind(s) {unknown}; expected {SSURGO_KINDS}")
    polys, pnote = read_geo(ppath, columns=["component", "in_bbox"])
    a.layer_files["ssurgo"] = [str(lpath), str(ppath)]
    lcomp = lines.component.to_numpy().astype(int)
    pcomp = polys.component.to_numpy().astype(int)
    kind = lines.kind.astype(str).to_numpy()
    lg = np.asarray(lines.geometry.values)
    pg = np.asarray(polys.geometry.values)
    join = kind == "survey_join"
    both_county = (np.array([county_area(n) for n in lines.areaname_a.astype(object)])
                   & np.array([county_area(n) for n in lines.areaname_b.astype(object)]))
    names = pd.Series(pd.NA, index=lines.index, dtype="string")
    names[join] = [" | ".join(sorted((str(p), str(q))))
                   for p, q in zip(lines.areaname_a[join], lines.areaname_b[join])]
    lines = lines.assign(areanames=names)
    layer_mask = {"soil": kind == "soil_line", "soil_water": kind == "water_edge",
                  "soil_join_county": join & both_county, "soil_join_other": join & ~both_county}
    vt["soil_ok"] = False
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        box = (r.minx, r.miny, r.maxx, r.maxy)
        q = shapely.box(*box)
        q_in = shapely.box(r.minx + TOL_M, r.miny + TOL_M, r.maxx - TOL_M, r.maxy - TOL_M)
        pm = pcomp == c
        lm = lcomp == c
        nv = int((vt.component == c).sum())
        if c not in built or not box_contains(built[c], box):
            why = (f"{lpath.name} holds no component {c} (built for components {sorted(built)})"
                   if c not in built else
                   f"{lpath.name} was built for a component {c} bbox {fmt_box(built[c])} that "
                   f"does not contain this one {fmt_box(box)}")
            for lay in SOURCES["ssurgo"]:
                cov.add(c, lay, "FAIL", f"{why}: a file from another build (v1?); {rerun}"
                        if lay == "soil" else "same file as soil", n_vertices=nv)
            continue
        probs = []
        if not pm.any():
            probs.append(f"no mupolygons for component {c} in {ppath.name}")
        elif not polys.in_bbox.to_numpy()[pm].any():
            probs.append("no mupolygon meets the bbox")
        clip = lm & (kind == "clip_edge")
        clip_in = float(shapely.length(shapely.intersection(lg[clip], q_in)).sum()) if clip.any() else 0.0
        if clip_in > 0:
            probs.append(f"{clip_in:,.0f} m of clip_edge inside the bbox: SSURGO was fetched "
                         f"for a smaller or different box (rerun pull_ssurgo.py on {a.bboxes_p.name})")
        nd = lm & (kind == "no_data_edge")
        nd_in = float(shapely.length(shapely.intersection(lg[nd], q)).sum()) if nd.any() else 0.0
        iv = np.flatnonzero((vt.component == c).to_numpy())
        if pm.any():
            tree = STRtree(pg[pm])
            hit = tree.query(shapely.points(vt.x.to_numpy()[iv], vt.y.to_numpy()[iv]),
                             predicate="dwithin", distance=TOL_M)[0]
            ok = np.zeros(len(iv), bool)
            ok[hit] = True
            vt.loc[iv, "soil_ok"] = ok
        n_out = int((~vt.soil_ok.to_numpy()[iv]).sum())
        n = {lay: int((lm & m).sum()) for lay, m in layer_mask.items()}
        enclose = ("clip_edge encloses the bbox; the polygons cover all of it" if nd_in == 0 else
                   f"clip_edge stays outside the bbox; {nd_in:,.0f} m of no_data_edge inside "
                   f"it (no SSURGO polygon beyond: open water or a gap)")
        base = (f"{int(pm.sum()):,} mupolygons ({int(polys.in_bbox.to_numpy()[pm].sum()):,} "
                f"meet the bbox); {enclose}")
        if probs:
            st, nt = "FAIL", "; ".join(probs)
        elif n_out:
            st, nt = "warn", (f"{base}; {n_out:,} vertices in no map unit (soil_ok False; "
                              f"roles {vt.role.iloc[iv][~vt.soil_ok.to_numpy()[iv]].value_counts().to_dict()})")
        else:
            st, nt = "ok", base
        if note or pnote:
            nt += f"; {note or pnote}"
        cov.add(c, "soil", st, nt, n_lines=n["soil"], n_vertices=nv)
        others = sorted(set(names[lm & layer_mask["soil_join_other"]]))
        what = {"soil_water": ("shores (water_edge), never class S", "no water_edge"),
                "soil_join_county": ("joins of two county surveys", "no join of two county surveys"),
                "soil_join_other": (f"joins with federal or tribal land: {'; '.join(others)}",
                                    "no join with federal or tribal land")}
        for lay, (has, hasnt) in what.items():
            cov.add(c, lay, st, "same file and coverage as soil" if probs else
                    f"{has}; same coverage as soil" if n[lay] else
                    f"{hasnt} in this bbox (NaN distances)", n_lines=n[lay], n_vertices=nv)
    out = []
    for name, m in layer_mask.items():
        sub = lines[m].reset_index(drop=True)
        extra = {} if name == "soil_water" else {"n_diff": nullable(sub.n_diff),
                                                 "diff_attrs": nullable(sub.diff_attrs)}
        if name.startswith("soil_join"):
            extra["areanames"] = nullable(sub.areanames)
        out.append(LineSet(name, lg[m], nullable(sub.line_id), extra=extra,
                           group=lcomp[m], by="component"))
    return out


def load_county(a, vt, bb, cov: Coverage) -> list[LineSet]:
    path = a.files.get("county") or a.gis_dir / f"county_lines_{a.state}.parquet"
    if not path.exists():
        cov.add("all", "county", "FAIL", f"{path} missing: run pull_gis_lines.py --state "
                f"{a.state} --layers counties, or --skip-layer county")
        return []
    lines, note = read_geo(path)
    if "kind" not in lines.columns:
        raise SystemExit(f"{path}: no kind column")
    a.layer_files["county"] = [str(path)]
    ppath = path.with_name(path.name.replace("county_lines_", "counties_"))
    ext = tuple(lines.total_bounds)
    kind = lines.kind.astype(str).to_numpy()
    n_overlap = int((kind == "overlap").sum())
    counties = None
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        q = (r.minx, r.miny, r.maxx, r.maxy)
        iv = np.flatnonzero((vt.component == c).to_numpy())
        nv = len(iv)
        if box_contains(ext, q):
            st, nt = "ok", f"layer extent {fmt_box(ext)} contains the bbox"
        else:
            if counties is None and ppath.exists():
                cg, _ = read_geo(ppath, columns=[])
                counties = shapely.union_all(np.asarray(cg.geometry.values))
                shapely.prepare(counties)
            if counties is None:
                st, nt = "FAIL", (f"layer extent {fmt_box(ext)} does not contain the bbox "
                                  f"{fmt_box(q)}, and {ppath.name} is not there to show the "
                                  f"vertices are in-state")
            else:
                pts = shapely.points(vt.x.to_numpy()[iv], vt.y.to_numpy()[iv])
                inside = shapely.dwithin(counties, pts, TOL_M)
                n_out = int((~inside).sum())
                far = np.zeros(nv, dtype=bool)
                if n_out:
                    dout = shapely.distance(counties, pts[~inside])
                    far[~inside] = dout > a.state_tol_m
                    flag = a.flags.setdefault("outside_state", np.zeros(len(vt), dtype=bool))
                    flag[iv[~inside]] = True
                n_far = int(far.sum())
                st = "FAIL" if n_far else ("warn" if n_out else "ok")
                nt = (f"{n_far:,} vertices more than {a.state_tol_m:.0f} m outside the "
                      f"{a.state} counties: the layer does not cover them" if n_far else
                      f"{n_out:,} vertices up to {dout.max():.0f} m outside the {a.state} "
                      f"counties (NFHL and TIGER draw the line differently), flagged "
                      f"outside_state; their distances are to {a.state}'s lines only"
                      if n_out else
                      f"bbox reaches beyond the {a.state} extent, but every vertex is inside a "
                      f"{a.state} county and the state/coast line is in the layer")
        if n_overlap:
            nt += f"; {n_overlap} 'overlap' lines ignored"
        if note:
            nt += f"; {note}"
        cov.add(c, "county", st, nt, n_lines=int((kind == "county").sum()), n_vertices=nv)
        cov.add(c, "state_or_coast", st, "same file and coverage",
                n_lines=int((kind == "state_or_coast").sum()), n_vertices=nv)
    rows = pd.Series(np.arange(len(lines)), dtype="Int64")
    g = np.asarray(lines.geometry.values)
    out = []
    for name, k in (("county", "county"), ("state_or_coast", "state_or_coast")):
        m = kind == k
        kd = (nullable(lines.outer[m]) if (k == "state_or_coast" and "outer" in lines.columns)
              else None)
        out.append(LineSet(name, g[m], rows[m].reset_index(drop=True), kind=kd))
    return out


def load_panel(a, vt, bb, cov: Coverage) -> list[LineSet]:
    path = a.files.get("panel") or a.gis_dir / f"panel_edges_{a.state}.parquet"
    if not path.exists():
        cov.add("all", "panel", "FAIL", f"{path} missing: run pull_gis_lines.py --state "
                f"{a.state} --layers panels, or --skip-layer panel")
        return []
    edges, note = read_geo(path)
    for col in ("dfirm_id", "kind"):
        if col not in edges.columns:
            raise SystemExit(f"{path}: no {col} column")
    a.layer_files["panel"] = [str(path)]
    dfirm = edges.dfirm_id.astype(str).to_numpy()
    kind = edges.kind.astype(str).to_numpy()
    g = np.asarray(edges.geometry.values)
    b = shapely.bounds(g)
    vdf = vt.dfirm_id.fillna("").astype(str).to_numpy()   # "" = no lobe: a 'vertices' FAIL
    foot = {}
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        iv = np.flatnonzero((vt.component == c).to_numpy())
        probs, n_out, n_int, n_ext, notes = [], 0, 0, 0, []
        for d in sorted(set(vdf[iv]) - {""}):
            jv = iv[vdf[iv] == d]
            m = dfirm == d
            if not m.any():
                probs.append(f"DFIRM {d} ({len(jv):,} vertices) has no panel edges in {path.name}")
                continue
            n_int += int((m & np.isin(kind, ("interior", "overlap"))).sum())
            n_ext += int((m & (kind == "exterior")).sum())
            x, y = vt.x.to_numpy()[jv], vt.y.to_numpy()[jv]
            if d not in foot:
                faces = shapely.get_parts(shapely.polygonize(g[m]))
                foot[d] = shapely.union_all(faces) if len(faces) else None
                if foot[d] is not None:
                    shapely.prepare(foot[d])
            if foot[d] is None:
                notes.append(f"{d}: edges do not polygonize, footprint not checked")
                continue
            off = ~shapely.dwithin(foot[d], shapely.points(x, y), TOL_M)
            k = int(off.sum())
            n_out += k
            if k:
                flag = a.flags.setdefault("outside_own_panels", np.zeros(len(vt), dtype=bool))
                flag[jv[off]] = True
                notes.append(f"{d}: {k:,} vertices outside its panel footprint, flagged "
                             f"outside_own_panels (distance is to its exterior edge)")
        ds = sorted(set(vdf[iv]) - {""})
        if probs:
            st, nt = "FAIL", "; ".join(probs)
        else:
            st = "warn" if n_out else "ok"
            nt = (f"own-DFIRM edges for {', '.join(ds)}"
                  + (f"; {'; '.join(notes)}" if notes else "; every vertex inside its "
                     f"DFIRM's panel footprint"))
        if note:
            nt += f"; {note}"
        cov.add(c, "panel_interior", st, nt, n_lines=n_int, n_vertices=len(iv))
        cov.add(c, "panel_exterior", st, "same file and coverage", n_lines=n_ext,
                n_vertices=len(iv))
    rows = pd.Series(np.arange(len(edges)), dtype="Int64")
    mi = np.isin(kind, ("interior", "overlap"))
    me = kind == "exterior"
    return [LineSet("panel_interior", g[mi], rows[mi].reset_index(drop=True),
                    kind=nullable(edges.kind[mi]), group=dfirm[mi], by="dfirm_id"),
            LineSet("panel_exterior", g[me], rows[me].reset_index(drop=True),
                    group=dfirm[me], by="dfirm_id")]


def load_nhdplus_canals(a, vt, bb, cov: Coverage) -> list[LineSet]:
    import geopandas as gpd
    import pyogrio
    shp = sorted(a.nhdplus_dir.glob("*/NHDSnapshot/**/NHDFlowline.shp"))
    if not shp:
        cov.add("all", "canals", "FAIL", f"--canals-from-nhdplus: no */NHDSnapshot/**/"
                f"NHDFlowline.shp under {a.nhdplus_dir} (fetch_nhdplus_fl.sh extracts them)")
        return []
    a.layer_files["canals"] = [str(p) for p in shp]
    to_ll = Transformer.from_crs(CRS_WORK, CRS_NHD, always_xy=True)
    units = []
    for p in shp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = pyogrio.read_info(p, force_total_bounds=True)
        if CRS.from_user_input(info["crs"]).to_epsg() != CRS_NHD:
            raise SystemExit(f"{p}: CRS {info['crs']}, expected EPSG:{CRS_NHD}")
        units.append(shapely.box(*info["total_bounds"]))
    unit_cover = shapely.union_all(units)
    parts = []
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        env = to_ll.transform_bounds(r.minx, r.miny, r.maxx, r.maxy, densify_pts=101)
        nv = int((vt.component == c).sum())
        if not shapely.contains(unit_cover, shapely.box(*env)):
            cov.add(c, "canals", "FAIL", f"bbox (lon/lat {tuple(round(v, 4) for v in env)}) is "
                    f"not inside the extents of the {len(shp)} NHDPlus units", n_vertices=nv)
            continue
        got = []
        for p in shp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")       # Measured 3D LineString -> LineString Z
                f = pyogrio.read_dataframe(p, columns=["COMID", "FTYPE", "FCODE"],
                                           where="FTYPE = 'CanalDitch'", bbox=env)
            if len(f):
                got.append(f)
        n = sum(len(f) for f in got)
        parts += got
        cov.add(c, "canals", "ok", f"NHDPlus V2 NHDFlowline FTYPE CanalDitch (network and "
                f"non-network) from {len(shp)} units; every line meeting the bbox's lon/lat "
                f"envelope, kept whole", n_lines=n, n_vertices=nv)
    if not parts:
        return [LineSet("canals", np.empty(0, dtype=object), pd.Series([], dtype="Int64"),
                        kind=pd.Series([], dtype="string"))]
    f = pd.concat(parts, ignore_index=True).drop_duplicates("COMID").reset_index(drop=True)
    f = gpd.GeoDataFrame(f, geometry="geometry", crs=parts[0].crs).to_crs(CRS_WORK)
    g = shapely.force_2d(np.asarray(f.geometry.values))
    return [LineSet("canals", g, nullable(f.COMID), kind=nullable(f.FCODE.astype(str)))]


def load_optional(a, source: str, vt, bb, cov: Coverage) -> list[LineSet]:
    path = a.files.get(source)
    how = "--layer-file"
    if path is None:
        path, how = find_optional(a.gis_dir, source, a.prefix, a.state)
    if source == "canals" and a.canals_from_nhdplus:
        if path is not None:
            cov.add("all", "canals", "FAIL", f"--canals-from-nhdplus given but a canals file "
                    f"exists ({path}); drop the flag or point --layer-file canals= at it")
            return []
        return load_nhdplus_canals(a, vt, bb, cov)
    if path is None:
        cov.add("all", source, "absent", how)
        return []
    if not path.exists():
        cov.add("all", source, "FAIL", f"{path} not found")
        return []
    gdf, note = read_geo(path)
    a.layer_files[source] = [str(path)]
    ext, ext_how = declared_bounds(path, gdf)
    idc = pick_column(gdf.columns, ID_CANDIDATES)
    kdc = pick_column(gdf.columns, KIND_CANDIDATES)
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        q = (r.minx, r.miny, r.maxx, r.maxy)
        nv = int((vt.component == c).sum())
        n = int(shapely.intersects(np.asarray(gdf.geometry.values), shapely.box(*q)).sum())
        ok = box_contains(ext, q)
        cov.add(c, source, "ok" if ok else "FAIL",
                (f"{path.name} ({how}); {ext_how} {fmt_box(ext)} "
                 + ("contains the bbox" if ok else f"does not contain the bbox {fmt_box(q)}: "
                    f"re-pull it for these bboxes, or --skip-layer {source}"))
                + (f"; {note}" if note else ""), n_lines=n, n_vertices=nv)
    ids = (nullable(gdf[idc]) if idc else pd.Series(np.arange(len(gdf)), dtype="Int64"))
    kind = nullable(gdf[kdc].astype(str)) if kdc else None
    return [LineSet(source, shapely.force_2d(np.asarray(gdf.geometry.values)), ids, kind=kind)]


# ----------------------------------------------------------------- measure
def measure(vt: pd.DataFrame, ls: LineSet) -> dict:
    """The output columns of one layer."""
    x, y = vt.x.to_numpy(float), vt.y.to_numpy(float)
    d = np.full(len(vt), np.nan)
    j = np.full(len(vt), -1, dtype=np.int64)
    if ls.by is None:
        d, j = nearest(x, y, ls.geom)
    else:
        key = vt[ls.by].fillna("").astype(str).to_numpy()
        grp = ls.group.astype(str)
        for gval in sorted(set(key)):
            iv = np.flatnonzero(key == gval)
            il = np.flatnonzero(grp == gval)
            if not len(il):
                continue                       # no line of this layer in the group: NaN, decision 5
            dd, jj = nearest(x[iv], y[iv], ls.geom[il])
            d[iv] = dd
            j[iv] = np.where(jj >= 0, il[np.maximum(jj, 0)], -1)
    pick = lambda s: s.reindex(j).reset_index(drop=True)      # -1 -> <NA>
    cols = {f"d_{ls.name}_m": d, f"{ls.name}_id": pick(ls.ids)}
    if ls.kind is not None:
        cols[f"{ls.name}_kind"] = pick(ls.kind)
    for k, s in ls.extra.items():
        cols[f"{ls.name}_{k}"] = pick(s)
    return cols


def margins(vt: pd.DataFrame, bb: pd.DataFrame) -> np.ndarray:
    """Each vertex's distance to its own bbox edge (decision 6)."""
    r = bb.reindex(vt.component.to_numpy())
    x, y = vt.x.to_numpy(float), vt.y.to_numpy(float)
    return np.minimum.reduce([x - r.minx.to_numpy(), r.maxx.to_numpy() - x,
                              y - r.miny.to_numpy(), r.maxy.to_numpy() - y])


# ----------------------------------------------------------------- main
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vertices", required=True, type=Path,
                    help="<prefix>_vertices_z.parquet from pull_dem.py sample")
    ap.add_argument("--lobes", type=Path, help="default: <prefix>_lobes.parquet beside --vertices")
    ap.add_argument("--bboxes", type=Path, help="default: <prefix>_bboxes.parquet beside --vertices")
    ap.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--prefix", help="default: from the --vertices file name")
    ap.add_argument("--out", type=Path,
                    help="default: results/<today>/boundary/<prefix>_vertex_layers.parquet")
    ap.add_argument("--skip-layer", action="append", default=[], metavar="NAME",
                    help=f"skip a source ({', '.join(SOURCES)}) or one layer "
                         f"({', '.join(l for v in SOURCES.values() for l in v)}); repeatable")
    ap.add_argument("--layer-file", action="append", default=[], metavar="SOURCE=PATH",
                    help="explicit file for a source; repeatable")
    ap.add_argument("--canals-from-nhdplus", action="store_true",
                    help="canals = NHDPlus V2 NHDFlowline FTYPE CanalDitch under --nhdplus-dir "
                         "(only when no canals file exists)")
    ap.add_argument("--nhdplus-dir", type=Path, default=NHD_DIR)
    ap.add_argument("--allow-dem-missing", action="store_true",
                    help="keep vertices without z, z_in or z_out (dem_ok False) instead of failing")
    ap.add_argument("--state-tol-m", type=float, default=1000.0,
                    help="vertices up to this far outside the state's counties are flagged "
                         "outside_state, not failed (decision 7)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    a.prefix = a.prefix or prefix_of(a.vertices)
    a.lobes_p = a.lobes or a.vertices.with_name(f"{a.prefix}_lobes.parquet")
    a.bboxes_p = a.bboxes or a.vertices.with_name(f"{a.prefix}_bboxes.parquet")
    a.out = a.out or RESULTS / date.today().isoformat() / "boundary" / f"{a.prefix}_vertex_layers.parquet"
    layers = {l for v in SOURCES.values() for l in v}
    a.skip_sources, a.skip_layers = set(), set()
    for s in a.skip_layer:
        if s in SOURCES:
            a.skip_sources.add(s)
        elif s in layers:
            a.skip_layers.add(s)
        else:
            ap.error(f"--skip-layer {s}: not one of {sorted(SOURCES)} or {sorted(layers)}")
    a.files = {}
    for kv in a.layer_file:
        k, _, v = kv.partition("=")
        if k not in SOURCES or not v:
            ap.error(f"--layer-file {kv}: expected SOURCE=PATH with SOURCE in {sorted(SOURCES)}")
        a.files[k] = Path(v)
    a.layer_files = {}
    a.flags = {}                                     # per-vertex booleans written with the table
    return a


def main(argv=None) -> int:
    a = parse(argv)
    cov_path = a.out.with_name(a.out.stem + "_coverage.json")
    if a.out.exists() and not a.overwrite:
        print(f"[{a.prefix}] {a.out} exists; skipping (--overwrite to redo)")
        return 0
    for p in (a.vertices, a.lobes_p, a.bboxes_p):
        if not p.exists():
            raise SystemExit(f"{p} missing")
    t0 = time.time()
    print(f"=== edge_layers {a.prefix} ===", flush=True)
    cov = Coverage()
    raw, vt, bb = read_inputs(a, cov)
    print(f"  {len(vt):,} vertices in {vt.component.nunique()} component(s); "
          f"{len(bb)} bbox(es); gis-dir {a.gis_dir}", flush=True)

    loaders = dict(zip(REQUIRED, (load_ssurgo, load_county, load_panel)))
    sets: list[LineSet] = []
    for src in SOURCES:
        if src in a.skip_sources:
            for lay in SOURCES[src]:
                cov.add("all", lay, "skipped", f"--skip-layer {src}")
            continue
        t1 = time.time()
        got = (loaders[src](a, vt, bb, cov) if src in loaders
               else load_optional(a, src, vt, bb, cov))
        for ls in got:
            if ls.name in a.skip_layers:
                cov.skip(ls.name, f"--skip-layer {ls.name}")
            else:
                sets.append(ls)
        if got:
            print(f"  {src}: {', '.join(f'{ls.name} {len(ls.geom):,} lines' for ls in got)} "
                  f"[{time.time() - t1:.1f}s]", flush=True)

    meta = dict(prefix=a.prefix, state=a.state, vertices=str(a.vertices), lobes=str(a.lobes_p),
                bboxes=str(a.bboxes_p), gis_dir=str(a.gis_dir), layer_files=a.layer_files,
                skip=sorted(a.skip_layer), canals_from_nhdplus=a.canals_from_nhdplus,
                nhdplus_dir=str(a.nhdplus_dir) if a.canals_from_nhdplus else None,
                allow_dem_missing=a.allow_dem_missing, state_tol_m=a.state_tol_m,
                piece_segs=PIECE_SEGS, tol_m=TOL_M,
                crs=f"EPSG:{CRS_WORK}", created=datetime.now().isoformat(timespec="seconds"))
    fails = cov.failed()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if fails:
        cov.print()
        cov_path.write_text(json.dumps(jsonable(dict(status="FAILED", **meta, coverage=cov.rows)),
                                       indent=1))
        print(f"\nFAILED: {len(fails)} coverage check(s) failed; nothing measured, no table "
              f"written. Coverage: {cov_path}", file=sys.stderr)
        for r in fails:
            c = r["component"]
            print(f"  {'all components' if c == 'all' else f'component {c}'}, {r['layer']}: "
                  f"{r['note']}", file=sys.stderr)
        return 1

    res = raw.copy()                                   # decision 9: input columns as read
    res["dfirm_id"] = vt["dfirm_id"]
    if "soil_ok" in vt:
        res["soil_ok"] = vt["soil_ok"]
    for name in ("outside_state", "outside_own_panels"):
        res[name] = a.flags.get(name, np.zeros(len(vt), dtype=bool))
    marg = margins(vt, bb)
    for ls in sets:
        t1 = time.time()
        cols = measure(vt, ls)
        for k, v in cols.items():
            res[k] = v
        d = cols[f"d_{ls.name}_m"]
        if ls.name in BBOX_LIMITED:
            beyond = np.isnan(d) | (d > marg)
            for c in sorted(vt.component.unique()):
                m = (vt.component == c).to_numpy()
                cov.update(c, ls.name, beyond_margin=int((beyond & m).sum()))
        print(f"  measured {ls.name} [{time.time() - t1:.1f}s]", flush=True)

    cov.print()
    meta["coverage"] = cov.rows
    upstream = {k: v for k, v in (pq.read_schema(a.vertices).metadata or {}).items()
                if k != b"pandas"}                     # decision 9, e.g. b"pull_dem"
    tbl = pa.Table.from_pandas(res, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), **upstream,
                                       b"edge_layers": json.dumps(jsonable(meta)).encode()})
    tmp = a.out.with_name(a.out.name + f".tmp{os.getpid()}")
    pq.write_table(tbl, tmp)
    os.replace(tmp, a.out)
    cov_path.write_text(json.dumps(jsonable(dict(status="ok", **meta)), indent=1))
    n_warn = sum(r["status"] == "warn" for r in cov.rows)
    print(f"\n[{a.prefix}] wrote {a.out} ({len(res):,} rows, {len(res.columns)} columns, "
          f"{a.out.stat().st_size / 1e6:.1f} MB) and {cov_path.name}"
          + (f"; {n_warn} warning(s) above" if n_warn else "")
          + f" [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
