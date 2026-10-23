#!/usr/bin/env python3
"""
Distance from every lobe-edge vertex to the nearest line of each GIS layer, with
coverage checks: the last data step before aotil/boundary_stat.py (docs/TASK-DATA-AUDIT.md
D7). Nothing here is analysis or a model feature.

Inputs: --vertices <prefix>_vertices_z.parquet (pull_dem.py sample), --lobes and --bboxes
(aotil/lobes.py; default: beside --vertices) and, under --gis-dir, the required layers
ssurgo/<prefix>_mulines.parquet (+ _mupolygons), county_lines_<ST>.parquet (+ counties_<ST>)
and panel_edges_<ST>.parquet, plus optional NWI, PLSS, road and canal files found by name
(or NHDPlus canals with --canals-from-nhdplus).
Output: <prefix>_vertex_layers.parquet, every input column unchanged plus dfirm_id, soil_ok,
outside_state, outside_own_panels and, per layer, d_<layer>_m (metres, EPSG:5070), <layer>_id
and the layer's kind and attribute columns; and <stem>_coverage.json. Any FAIL in the
coverage table stops the run before anything is measured.

    python aotil/edge_layers.py --vertices results/<D>/boundary/<prefix>_vertices_z.parquet \\
        --canals-from-nhdplus --out results/<D>/boundary/<prefix>_vertex_layers.parquet
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

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import shapely
from shapely import STRtree
from pyproj import CRS, Transformer

CRS_WORK = 5070
CRS_NHD = 4269            # NHDPlus V2 shapefiles ship in NAD83 geographic
GIS_DIR = Path("data/interim/gis")
NHD_DIR = Path("data/raw/nhdplus")
RESULTS = Path("results")

PIECE_SEGS = 16           # segments per STRtree piece (see line_pieces)
TOL_M = 1.0               # coverage tolerance, metres
CANAL_MARGIN_M = 1000.0   # vertices + this must lie inside the NHDPlus units (load_nhdplus_canals)
M_PER_DEG_LAT = 111_320.0
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
SSURGO_META = b"pull_ssurgo"                  # parquet key holding the bboxes it was built for
COUNTY_AREA = re.compile(r"\bCount(?:y|ies)\b")
NOT_COUNTY_AREA = re.compile(r"\b(?:National|Reservation|Conservation|Refuge)\b")

# source -> its layers; --skip-layer takes either
SOURCES = {
    "ssurgo": ("soil", "soil_water", "soil_join_county", "soil_join_other"),
    "county": ("county", "state_or_coast"),
    "panel": ("panel_interior", "panel_exterior"),
    "nwi": ("nwi",),
    "plss": ("plss",),
    "roads": ("roads",),
    "canals": ("canals",),
}
REQUIRED = ("ssurgo", "county", "panel")      # a missing file fails the run unless skipped
OPTIONAL_PATTERNS = {
    "nwi": ("*nwi*", "*wetland*"),
    "plss": ("*plss*", "*cadnsdi*"),
    "roads": ("*road*",),
    "canals": ("*canal*",),
}
GEO_EXT = (".parquet", ".gpkg", ".shp", ".geojson", ".fgb")
# Never searched for optional layers. No DEM raster is opened here (z comes from the table).
SEARCH_SKIP_DIRS = {"dem", "ssurgo", "cache", "__pycache__"}
OTHER_PREFIX = re.compile(r"_S\d+_")          # <cell>_<variant>_S<S>_... of another prefix
ID_CANDIDATES = ("line_id", "COMID", "LINEARID", "FRSTDIVID", "PLSSID", "OBJECTID", "id")
KIND_CANDIDATES = ("kind", "WETLAND_TYPE", "ATTRIBUTE", "MTFCC", "FCODE", "FTYPE")
# Layers pulled per bbox: a distance beyond the vertex's bbox margin is an upper bound only.
BBOX_LIMITED = SOURCES["ssurgo"] + ("nwi", "plss", "roads", "canals")

SHAPELY_LINE_TYPES = (1, 2)       # LineString, LinearRing
SHAPELY_POLYGON_TYPES = (3, 6)    # Polygon, MultiPolygon


def layer_names() -> list[str]:
    """Every layer name, in SOURCES order."""
    names = []
    for layers in SOURCES.values():
        names.extend(layers)
    return names


# ----------------------------------------------------------------- coverage table
class Coverage:
    """One row per (component, layer) check; component 'all' for checks that are not
    per component. Status: ok, warn, FAIL, skipped, absent."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, comp, layer: str, status: str, note: str = "", **num) -> None:
        self.rows.append(dict(component=comp, layer=layer, status=status, note=note, **num))

    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]

    def skip(self, layer: str, note: str) -> None:
        """Replace a layer's rows by one 'skipped' row. FAIL rows stay: they are about
        the file the layer shares with the other layers of its source."""
        self.rows = [r for r in self.rows if r["layer"] != layer or r["status"] == "FAIL"]
        self.add("all", layer, "skipped", note)

    def update(self, comp, layer: str, **num) -> None:
        """Add or overwrite numbers in the rows of (comp, layer)."""
        for r in self.rows:
            if r["component"] == comp and r["layer"] == layer:
                r.update(num)

    def print(self) -> None:
        print(f"\n  {'comp':>4}  {'layer':<16} {'status':<7} {'lines':>8} {'vertices':>9} "
              f"{'beyond':>7}  note", flush=True)
        for r in self.rows:
            n_lines = count_text(r.get("n_lines"))
            n_vertices = count_text(r.get("n_vertices"))
            beyond = count_text(r.get("beyond_margin"))
            print(f"  {str(r['component']):>4}  {r['layer']:<16} {r['status']:<7} "
                  f"{n_lines:>8} {n_vertices:>9} {beyond:>7}  {r['note']}", flush=True)


def count_text(n) -> str:
    return "-" if n is None else f"{n:,}"


def jsonable(o):
    """Plain JSON values: numpy scalars as Python ones, NaN and inf as None, paths as str."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ----------------------------------------------------------------- geometry
def explode_lines(geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single LineStrings of lines, multilines and polygon boundaries (exterior and
    holes), with the index of the feature each came from. Points and empties are dropped."""
    g = np.array(geom, dtype=object, copy=True)
    is_polygon = np.isin(shapely.get_type_id(g), SHAPELY_POLYGON_TYPES)
    if is_polygon.any():
        g[is_polygon] = shapely.boundary(g[is_polygon])
    parts, parent = shapely.get_parts(g, return_index=True)
    keep = np.isin(shapely.get_type_id(parts), SHAPELY_LINE_TYPES) & ~shapely.is_empty(parts)
    return parts[keep], parent[keep]


def ranks_within(counts: np.ndarray) -> np.ndarray:
    """0, 1, ..., counts[i] - 1 for each i in turn, concatenated."""
    starts = np.cumsum(counts) - counts
    return np.arange(counts.sum()) - np.repeat(starts, counts)


def line_pieces(geom: np.ndarray, k: int = PIECE_SEGS) -> tuple[np.ndarray, np.ndarray]:
    """Every line cut into pieces of at most k segments, and the index of the feature
    each piece belongs to. 2-D only.

    A 40 km soil line has a 40 km STRtree envelope, so every vertex near it would test
    the whole line. The distance to a line is the minimum over its pieces, so the cut
    changes only the speed."""
    parts, parent = explode_lines(geom)
    if not len(parts):
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64)
    xy, coord_part = shapely.get_coordinates(parts, return_index=True)
    n_coords = np.bincount(coord_part, minlength=len(parts))
    first_coord = np.cumsum(n_coords) - n_coords          # where each part starts in xy
    n_segs = np.maximum(n_coords - 1, 0)
    n_pieces = np.where(n_segs > 0, -(-n_segs // k), 0)   # ceil(n_segs / k)

    # Piece c of a part spans the part's coordinates c*k .. min((c+1)*k, n_segs), so at
    # most k segments; lo and hi are their positions in xy (neighbouring pieces share one).
    piece_part = np.repeat(np.arange(len(parts)), n_pieces)
    c = ranks_within(n_pieces)
    lo = first_coord[piece_part] + c * k
    hi = first_coord[piece_part] + np.minimum((c + 1) * k, n_segs[piece_part])
    n_piece_coords = hi - lo + 1
    coord_piece = np.repeat(np.arange(len(piece_part)), n_piece_coords)
    coord_index = np.repeat(lo, n_piece_coords) + ranks_within(n_piece_coords)
    pieces = shapely.linestrings(xy[coord_index], indices=coord_piece)
    return pieces, parent[piece_part]


def nearest(x: np.ndarray, y: np.ndarray, geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each (x, y) to the nearest point of the nearest feature of `geom`,
    and that feature's index; NaN and -1 when `geom` holds no line. Ties (a T-junction,
    a line shared by two features) go to the lowest index, so the result does not
    depend on tree order."""
    d = np.full(len(x), np.nan)
    j = np.full(len(x), -1, dtype=np.int64)
    if not len(x) or not len(geom):
        return d, j
    pieces, owner = line_pieces(geom)
    if not len(pieces):
        return d, j
    tree = STRtree(pieces)
    for start in range(0, len(x), QUERY_CHUNK):
        chunk = slice(start, start + QUERY_CHUNK)
        points = shapely.points(x[chunk], y[chunk])
        (point_i, piece_i), dist = tree.query_nearest(points, return_distance=True,
                                                      all_matches=True)
        # all_matches returns every equidistant piece: keep the lowest feature per point
        line_i = owner[piece_i]
        order = np.lexsort((line_i, point_i))
        point_i, line_i, dist = point_i[order], line_i[order], dist[order]
        first = np.r_[True, point_i[1:] != point_i[:-1]]
        d[start + point_i[first]] = dist[first]
        j[start + point_i[first]] = line_i[first]
    return d, j


def box_contains(outer, inner, tol: float = TOL_M) -> bool:
    """(minx, miny, maxx, maxy) `outer` contains `inner`, to tol metres."""
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol
            and outer[2] >= inner[2] - tol and outer[3] >= inner[3] - tol)


def fmt_box(b) -> str:
    return "(" + ", ".join(f"{v:,.0f}" for v in b) + ")"


def length_inside(geoms: np.ndarray, box) -> float:
    """Total length (m) of geoms inside a shapely box."""
    if not len(geoms):
        return 0.0
    return float(shapely.length(shapely.intersection(geoms, box)).sum())


def near_any(geoms: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Whether each point lies within TOL_M of one of geoms."""
    tree = STRtree(geoms)
    pairs = tree.query(shapely.points(x, y), predicate="dwithin", distance=TOL_M)
    near = np.zeros(len(x), bool)
    near[pairs[0]] = True                      # pairs: (point index, geom index) per hit
    return near


# ----------------------------------------------------------------- layers
@dataclass
class LineSet:
    """One layer's lines in EPSG:5070, with the id, kind and extra columns written
    for the nearest one. When `by` is set, a vertex is measured only against the
    lines whose `group` equals its `by` column (component or dfirm_id)."""
    name: str
    geom: np.ndarray
    ids: pd.Series
    kind: pd.Series | None = None
    extra: dict = field(default_factory=dict)
    group: np.ndarray | None = None
    by: str | None = None


def nullable(s: pd.Series) -> pd.Series:
    """s with a 0..n-1 index and a pandas nullable dtype, so a vertex with no line
    gets <NA> in its id, kind and extra columns."""
    s = s.reset_index(drop=True)
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64")
    if pd.api.types.is_float_dtype(s):
        return s.astype("Float64")
    return s.astype("string")


def take(s: pd.Series, j: np.ndarray) -> pd.Series:
    """The values of s (0..n-1 index) at positions j; <NA> where j is -1."""
    return s.reindex(j).reset_index(drop=True)


def read_geo(path: Path, columns: list[str] | None = None):
    """A GeoDataFrame in EPSG:5070, and a note when it had to be reprojected.
    A file that states no CRS cannot be placed and stops the run."""
    if path.suffix == ".parquet":
        if columns is not None:
            columns = columns + ["geometry"]
        gdf = gpd.read_parquet(path, columns=columns)
    else:
        with warnings.catch_warnings(action="ignore"):
            gdf = gpd.read_file(path, columns=columns)
    if gdf.crs is None:
        raise SystemExit(f"{path}: no CRS stated; cannot place it in EPSG:{CRS_WORK}")
    note = ""
    if gdf.crs.to_epsg() != CRS_WORK:
        note = f"reprojected from {gdf.crs.to_string()}"
        gdf = gdf.to_crs(CRS_WORK)
    return gdf, note


def to_work_bounds(bounds, crs: CRS) -> tuple:
    """(minx, miny, maxx, maxy) in `crs` as a box in EPSG:5070. The edges are densified
    so the reprojected box still holds the curved edges of the original."""
    if crs.to_epsg() != CRS_WORK:
        transformer = Transformer.from_crs(crs, CRS_WORK, always_xy=True)
        bounds = transformer.transform_bounds(*bounds, densify_pts=101)
    return tuple(bounds)


def declared_bounds(path: Path, gdf) -> tuple[tuple, str]:
    """The file's own extent in EPSG:5070 and where it came from: the GeoParquet
    'bbox' or the OGR extent when the file declares one, else its features' bounds."""
    try:
        if path.suffix == ".parquet":
            geo = json.loads(pq.read_schema(path).metadata.get(b"geo", b"{}"))
            column = geo.get("columns", {}).get(geo.get("primary_column", "geometry"), {})
            bounds, crs = column.get("bbox"), column.get("crs")
            if bounds and len(bounds) == 4:
                # GeoParquet: no crs means lon/lat
                src = CRS.from_user_input(crs) if crs else CRS.from_epsg(4326)
                return to_work_bounds(bounds, src), "declared GeoParquet bbox"
        else:
            info = pyogrio.read_info(path, force_total_bounds=True)
            bounds, crs = info.get("total_bounds"), info.get("crs")
            if bounds is not None and crs:
                return to_work_bounds(bounds, CRS.from_user_input(crs)), "declared OGR extent"
    except Exception:          # noqa: BLE001  unreadable metadata: use the features' bounds
        pass
    return tuple(gdf.total_bounds), "extent of the features"


def pick_column(cols, candidates) -> str | None:
    """The first candidate in cols, matched case-insensitively, spelled as in cols."""
    by_lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in by_lower:
            return by_lower[c.lower()]
    return None


def find_optional(gis_dir: Path, source: str, prefix: str, state: str) -> tuple[Path | None, str]:
    """The file of an optional layer under gis_dir, by name (case-insensitive), and
    how it was found. A file named <prefix>_... wins; files that look like another
    prefix's are ignored; two or more candidates stop the run (use --layer-file)."""
    if not gis_dir.exists():
        return None, f"{gis_dir} not found"
    patterns = OPTIONAL_PATTERNS[source]
    required_stems = (f"county_lines_{state}", f"counties_{state}", f"panel_edges_{state}")
    hits = []
    for root, dirs, files in os.walk(gis_dir):
        dirs[:] = [d for d in dirs if d.lower() not in SEARCH_SKIP_DIRS]   # prunes the walk
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() not in GEO_EXT or p.stem.startswith(required_stems):
                continue
            if any(fnmatch.fnmatch(p.stem.lower(), pat) for pat in patterns):
                hits.append(p)
    mine, others, rest = [], [], []
    for p in hits:
        if p.name.startswith(prefix + "_"):
            mine.append(p)
        elif OTHER_PREFIX.search(p.name):
            others.append(p)
        else:
            rest.append(p)
    use = mine or rest
    if not use:
        note = f"no file matching {'/'.join(patterns)} under {gis_dir}"
        if others:
            note += f" (ignored other prefixes' files: {[p.name for p in others]})"
        return None, note
    if len(use) > 1:
        raise SystemExit(f"{source}: several candidate files {sorted(str(p) for p in use)}; "
                         f"choose one with --layer-file {source}=PATH")
    return use[0], "found by file name"


# ----------------------------------------------------------------- inputs
def prefix_of(path: Path) -> str:
    for suffix in ("_vertices_z.parquet", "_vertices.parquet"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise SystemExit(f"{path.name}: expected <prefix>_vertices_z.parquet (or pass --prefix)")


def written_by_us(col: str) -> bool:
    """A column name this script writes: such an input column is an error, not an overwrite."""
    if col in ("dfirm_id", "soil_ok"):
        return True
    for layer in layer_names():
        if col == f"d_{layer}_m" or col.startswith(f"{layer}_"):
            return True
    return False


def read_inputs(a, cov: Coverage) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The vertex table as read (written back unchanged), a working copy of the
    columns used here plus dfirm_id, and the bboxes indexed by component. Adds the
    'inputs', 'vertices' and 'dem' coverage rows."""
    raw, vt = read_vertices(a, cov)
    lobes = read_lobes(a.lobes_p)
    bb = read_bboxes(a.bboxes_p)
    vt["dfirm_id"] = pd.array(lobes.dfirm_id.reindex(vt.lobe_id.astype(str)).to_numpy(),
                              dtype="string")
    check_vertices(a, vt, lobes, bb, cov)
    check_dem(vt, a.allow_dem_missing, cov)
    return raw, vt, bb


def read_vertices(a, cov: Coverage) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The vertex table and a working copy of VERTEX_COLS + DEM_COLS.

    The other columns, such as lobes.py's contact columns and pull_dem's probe flags,
    pass through untouched: boundary_stat.py decides which vertices count, so their
    absence is a warning."""
    raw = pd.read_parquet(a.vertices).reset_index(drop=True)
    missing = [c for c in VERTEX_COLS + DEM_COLS if c not in raw.columns]
    if missing:
        raise SystemExit(f"{a.vertices}: missing columns {missing} (run pull_dem.py sample first)")
    clash = [c for c in raw.columns if written_by_us(c)]
    if clash:
        raise SystemExit(f"{a.vertices}: has columns {clash} that this script writes "
                         f"(an edge_layers output given as --vertices?)")
    vt = raw[VERTEX_COLS + DEM_COLS].copy()
    vt["component"] = vt["component"].astype(np.int32)
    if not np.isfinite(vt[["x", "y"]].to_numpy(float)).all():
        raise SystemExit(f"{a.vertices}: non-finite x or y")

    passed_through = [c for c in raw.columns if c not in VERTEX_COLS + DEM_COLS]
    absent = []
    if not set(CONTACT_COLS) <= set(raw.columns):
        absent.append("lobes.py contact columns (an older lobes.py wrote these vertices)")
    if not set(PROBE_COLS) <= set(raw.columns):
        absent.append(f"pull_dem probe flags (pull_dem.py sample writes them only when "
                      f"{a.prefix}_lobes.parquet is beside the vertices)")
    note = f"carried through unchanged: {passed_through or 'nothing'}"
    if absent:
        note = f"absent: {'; '.join(absent)}; G1 cannot select vertices by them. " + note
    cov.add("all", "inputs", "warn" if absent else "ok", note)
    return raw, vt


def read_lobes(path: Path) -> pd.DataFrame:
    """The lobes table indexed by lobe_id as a string."""
    lobes = pq.read_table(path, columns=LOBE_COLS).to_pandas()
    if not lobes.lobe_id.is_unique:
        raise SystemExit(f"{path}: lobe_id is not unique")
    lobes["component"] = lobes["component"].astype(np.int32)
    return lobes.assign(lobe_id=lobes.lobe_id.astype(str)).set_index("lobe_id")


def read_bboxes(path: Path) -> pd.DataFrame:
    bb = pd.read_parquet(path)
    missing = [c for c in BBOX_COLS if c not in bb.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    bb = bb.astype({"component": np.int32}).set_index("component")
    if not bb.index.is_unique:
        raise SystemExit(f"{path}: duplicate component ids")
    return bb


def check_vertices(a, vt: pd.DataFrame, lobes: pd.DataFrame, bb: pd.DataFrame,
                   cov: Coverage) -> None:
    """One 'vertices' row per component: every vertex's lobe is in --lobes with the
    same component, role and zone and has a dfirm_id, and the vertex lies in its
    component's bbox."""
    key = vt.lobe_id.astype(str)
    known = key.isin(lobes.index).to_numpy()
    disagrees = np.zeros(len(vt), bool)
    # Compared as strings, converted before the reindex so that an unknown key cannot
    # turn the integer 0 into "0.0".
    lobe_meta = lobes[["component", "role", "zone"]].astype(str)
    for col in lobe_meta.columns:
        ref = lobe_meta[col].reindex(key[known]).to_numpy()
        disagrees[known] |= ref != vt[col].to_numpy()[known].astype(str)
    no_dfirm = vt.dfirm_id.isna().to_numpy()

    for c in sorted(vt.component.unique().tolist()):
        in_comp = (vt.component == c).to_numpy()
        n_vertices = int(in_comp.sum())
        if c not in bb.index:
            cov.add(c, "vertices", "FAIL", f"no bbox for component {c} in {a.bboxes_p.name}",
                    n_vertices=n_vertices)
            continue
        r = bb.loc[c]
        x, y = vt.x.to_numpy()[in_comp], vt.y.to_numpy()[in_comp]
        n_unknown = int((in_comp & ~known).sum())
        n_disagree = int((in_comp & disagrees).sum())
        n_no_dfirm = int((in_comp & known & no_dfirm).sum())
        n_outside = int(((x < r.minx) | (x > r.maxx) | (y < r.miny) | (y > r.maxy)).sum())
        problems = []
        if n_unknown:
            problems.append(f"{n_unknown:,} lobe_ids not in {a.lobes_p.name}")
        if n_disagree:
            problems.append(f"{n_disagree:,} disagree with their lobe on component/role/zone")
        if n_no_dfirm:
            problems.append(f"{n_no_dfirm:,} lobes without dfirm_id")
        if n_outside:
            problems.append(f"{n_outside:,} outside the bbox")
        if problems:
            cov.add(c, "vertices", "FAIL", "; ".join(problems), n_vertices=n_vertices)
        else:
            roles = vt.role[in_comp].value_counts().to_dict()
            cov.add(c, "vertices", "ok",
                    f"bbox {fmt_box((r.minx, r.miny, r.maxx, r.maxy))}; roles {roles}",
                    n_vertices=n_vertices)


def check_dem(vt: pd.DataFrame, allow_missing: bool, cov: Coverage) -> None:
    """One 'dem' row per component: every vertex has finite z, z_in, z_out and dem_ok."""
    finite = np.isfinite(vt[["z", "z_in", "z_out"]].to_numpy(float)).all(axis=1)
    dem_ok = vt.dem_ok.fillna(False).astype(bool).to_numpy() & finite
    for c in sorted(vt.component.unique().tolist()):
        in_comp = (vt.component == c).to_numpy()
        n_vertices = int(in_comp.sum())
        n_bad = int((in_comp & ~dem_ok).sum())
        lacking = f"{n_bad:,} vertices lack z, z_in or z_out (dem_ok False)"
        if not n_bad:
            status, note = "ok", f"z, z_in and z_out at all {n_vertices:,} vertices"
        elif allow_missing:
            status, note = "warn", f"{lacking}; kept (--allow-dem-missing)"
        else:
            status = "FAIL"
            note = f"{lacking}; rerun pull_dem.py fetch/sample, or --allow-dem-missing"
        cov.add(c, "dem", status, note, n_vertices=n_vertices)


# ----------------------------------------------------------------- per source
def county_area(name) -> bool:
    """A county soil survey area, by its areaname: County or Counties, and none of
    National, Reservation, Conservation, Refuge ("County" alone is not enough: "Big
    Cypress Reservation, Broward County, Florida"). Against SDA sacatalog this leaves
    out exactly the seven Florida federal and tribal areas (FL609, FL615, FL616, FL618,
    FL622, FL623, FL624) of the 74 Florida and 162 Georgia and Alabama areas. A
    missing areaname is not a county survey."""
    return (isinstance(name, str) and COUNTY_AREA.search(name) is not None
            and NOT_COUNTY_AREA.search(name) is None)


def ssurgo_bboxes(path: Path) -> dict | None:
    """{component: (minx, miny, maxx, maxy)} a pull_ssurgo output was built for;
    None when the file holds no such record."""
    md = pq.read_schema(path).metadata or {}
    rec = json.loads(md.get(SSURGO_META, b"{}")).get("bboxes")
    if not rec:
        return None
    return {int(r["component"]): (r["minx"], r["miny"], r["maxx"], r["maxy"]) for r in rec}


def load_ssurgo(a, vt, bb, cov: Coverage) -> list[LineSet]:
    """The four SSURGO layers, each matched to the vertex's component:

    soil              kind 'soil_line', a line a soil scientist drew between two soils (class S)
    soil_water        kind 'water_edge', a shore. SSURGO maps open water as map units, so
                      nearly every attribute differs across it, yet nobody drew it between
                      two soils: never part of class S.
    soil_join_county  kind 'survey_join' between two county survey areas: an administrative
                      line (in Florida a county or the state line) across which map units can
                      differ merely because the surveys were made at different times (a real
                      soil change is possible too; n_diff and diff_attrs say what changes)
    soil_join_other   kind 'survey_join' with federal or tribal land on a side
    The joins are two layers, not one with a kind column, so the nearer never hides the
    other. Sets vt["soil_ok"] (within TOL_M of a map unit; water counts) and records the
    files in a.layer_files."""
    lpath = a.files.get("ssurgo") or a.gis_dir / "ssurgo" / f"{a.prefix}_mulines.parquet"
    ppath = lpath.with_name(lpath.name.replace("_mulines", "_mupolygons"))
    missing = [p for p in (lpath, ppath) if not p.exists()]
    if missing:
        cov.add("all", "soil", "FAIL", f"{', '.join(str(p) for p in missing)} missing: run "
                f"pull_ssurgo.py --bboxes {a.bboxes_p}, or --skip-layer ssurgo")
        return []
    rerun = f"rerun pull_ssurgo.py --bboxes {a.bboxes_p} --yes --overwrite"
    lines, note = read_geo(lpath)

    # An older pull_ssurgo.py called shores soil lines and did not record its bboxes.
    missing_cols = [c for c in SSURGO_COLS if c not in lines.columns]
    built = {p: ssurgo_bboxes(p) for p in (lpath, ppath)}
    no_record = [p.name for p, rec in built.items() if rec is None]
    if missing_cols or no_record:
        why = []
        if missing_cols:
            why.append(f"{lpath.name} lacks {missing_cols}")
        if no_record:
            why.append(f"no bbox record in {', '.join(no_record)}")
        cov.add("all", "soil", "FAIL", f"written by an older pull_ssurgo.py ({'; '.join(why)}), "
                f"which called shores soil lines and did not name the survey areas: {rerun}")
        return []
    if built[lpath] != built[ppath]:
        cov.add("all", "soil", "FAIL", f"{lpath.name} and {ppath.name} were built for different "
                f"bboxes: {rerun}")
        return []
    built_for = built[lpath]
    unknown = sorted(set(lines.kind.astype(str)) - set(SSURGO_KINDS))
    if unknown:
        raise SystemExit(f"{lpath}: unknown kind(s) {unknown}; expected {SSURGO_KINDS}")
    polys, pnote = read_geo(ppath, columns=["component", "in_bbox"])
    a.layer_files["ssurgo"] = [str(lpath), str(ppath)]

    line_comp = lines.component.to_numpy().astype(int)
    poly_comp = polys.component.to_numpy().astype(int)
    poly_meets_bbox = polys.in_bbox.to_numpy()
    kind = lines.kind.astype(str).to_numpy()
    line_geom = np.asarray(lines.geometry.values)
    poly_geom = np.asarray(polys.geometry.values)
    join = kind == "survey_join"
    county_a = np.array([county_area(n) for n in lines.areaname_a.astype(object)])
    county_b = np.array([county_area(n) for n in lines.areaname_b.astype(object)])
    both_county = county_a & county_b
    # The two survey areas of each join, 'A | B' sorted
    areanames = pd.Series(pd.NA, index=lines.index, dtype="string")
    pairs = []
    for name_a, name_b in zip(lines.areaname_a[join], lines.areaname_b[join]):
        pairs.append(" | ".join(sorted((str(name_a), str(name_b)))))
    areanames[join] = pairs
    lines = lines.assign(areanames=areanames)
    layer_mask = {"soil": kind == "soil_line", "soil_water": kind == "water_edge",
                  "soil_join_county": join & both_county, "soil_join_other": join & ~both_county}

    vt["soil_ok"] = False
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        box = (r.minx, r.miny, r.maxx, r.maxy)
        n_vertices = int((vt.component == c).sum())
        # Protocol versions share the file name and the gis-dir is not dated, so the
        # stored bboxes are what tie a file to these vertices and component numbers.
        if c not in built_for or not box_contains(built_for[c], box):
            if c not in built_for:
                why = (f"{lpath.name} holds no component {c} (built for components "
                       f"{sorted(built_for)})")
            else:
                why = (f"{lpath.name} was built for a component {c} bbox "
                       f"{fmt_box(built_for[c])} that does not contain this one {fmt_box(box)}")
            for layer in SOURCES["ssurgo"]:
                if layer == "soil":
                    layer_note = f"{why}: a file from another build (v1?); {rerun}"
                else:
                    layer_note = "same file as soil"
                cov.add(c, layer, "FAIL", layer_note, n_vertices=n_vertices)
            continue

        line_in_comp = line_comp == c
        poly_in_comp = poly_comp == c
        problems = []
        if not poly_in_comp.any():
            problems.append(f"no mupolygons for component {c} in {ppath.name}")
        elif not poly_meets_bbox[poly_in_comp].any():
            problems.append("no mupolygon meets the bbox")
        # clip_edge is the edge of what was fetched: none may enter the bbox (shrunk by
        # TOL_M), i.e. the fetched polygons reach past every side of it.
        shrunk = shapely.box(r.minx + TOL_M, r.miny + TOL_M, r.maxx - TOL_M, r.maxy - TOL_M)
        clip_m = length_inside(line_geom[line_in_comp & (kind == "clip_edge")], shrunk)
        if clip_m > 0:
            problems.append(f"{clip_m:,.0f} m of clip_edge inside the bbox: SSURGO was fetched "
                            f"for a smaller or different box (rerun pull_ssurgo.py on "
                            f"{a.bboxes_p.name})")
        # no_data_edge: SSURGO has no polygon beyond (open water past a survey); allowed
        no_data_m = length_inside(line_geom[line_in_comp & (kind == "no_data_edge")],
                                  shapely.box(*box))

        iv = np.flatnonzero((vt.component == c).to_numpy())
        if poly_in_comp.any():
            vt.loc[iv, "soil_ok"] = near_any(poly_geom[poly_in_comp], vt.x.to_numpy()[iv],
                                             vt.y.to_numpy()[iv])
        soil_ok = vt.soil_ok.to_numpy()[iv]
        n_no_unit = int((~soil_ok).sum())
        n_lines = {layer: int((line_in_comp & m).sum()) for layer, m in layer_mask.items()}

        if no_data_m == 0:
            enclose = "clip_edge encloses the bbox; the polygons cover all of it"
        else:
            enclose = (f"clip_edge stays outside the bbox; {no_data_m:,.0f} m of no_data_edge "
                       f"inside it (no SSURGO polygon beyond: open water or a gap)")
        base = (f"{int(poly_in_comp.sum()):,} mupolygons "
                f"({int(poly_meets_bbox[poly_in_comp].sum()):,} meet the bbox); {enclose}")
        if problems:
            status, soil_note = "FAIL", "; ".join(problems)
        elif n_no_unit:
            roles = vt.role.iloc[iv][~soil_ok].value_counts().to_dict()
            status, soil_note = "warn", (f"{base}; {n_no_unit:,} vertices in no map unit "
                                         f"(soil_ok False; roles {roles})")
        else:
            status, soil_note = "ok", base
        if note or pnote:
            soil_note += f"; {note or pnote}"
        cov.add(c, "soil", status, soil_note, n_lines=n_lines["soil"], n_vertices=n_vertices)

        federal = sorted(set(areanames[line_in_comp & layer_mask["soil_join_other"]]))
        descriptions = {   # layer: (note when it has lines here, note when it has none)
            "soil_water": ("shores (water_edge), never class S", "no water_edge"),
            "soil_join_county": ("joins of two county surveys", "no join of two county surveys"),
            "soil_join_other": (f"joins with federal or tribal land: {'; '.join(federal)}",
                                "no join with federal or tribal land"),
        }
        for layer, (has_lines, no_lines) in descriptions.items():
            if problems:
                layer_note = "same file and coverage as soil"
            elif n_lines[layer]:
                layer_note = f"{has_lines}; same coverage as soil"
            else:
                layer_note = f"{no_lines} in this bbox (NaN distances)"
            cov.add(c, layer, status, layer_note, n_lines=n_lines[layer], n_vertices=n_vertices)

    out = []
    for name, m in layer_mask.items():
        sub = lines[m].reset_index(drop=True)
        extra = {}
        if name != "soil_water":
            extra["n_diff"] = nullable(sub.n_diff)
            extra["diff_attrs"] = nullable(sub.diff_attrs)
        if name.startswith("soil_join"):
            extra["areanames"] = nullable(sub.areanames)
        out.append(LineSet(name, line_geom[m], nullable(sub.line_id), extra=extra,
                           group=line_comp[m], by="component"))
    return out


def load_county(a, vt, bb, cov: Coverage) -> list[LineSet]:
    """county (between two in-state counties) and state_or_coast (kind = outer: 'state'
    or 'coast'). The id is the row number in county_lines_<ST>.

    With counties_<ST>.parquet, every vertex must lie inside a county or within
    --state-tol-m of one; the state's rectangle would also cover south Georgia and
    Alabama. The state line and the coast are lines of the layer, so no line beyond
    them can be nearer to an in-state vertex. NFHL and TIGER draw the water boundaries
    a little differently (St. Marys and Perdido rivers, up to ~215 m), so vertices
    just across are flagged in a.flags["outside_state"] and measured to this state's
    lines only. Without the counties file the layer's extent must contain the bbox."""
    path = a.files.get("county") or a.gis_dir / f"county_lines_{a.state}.parquet"
    if not path.exists():
        cov.add("all", "county", "FAIL", f"{path} missing: run pull_gis_lines.py --state "
                f"{a.state} --layers counties, or --skip-layer county")
        return []
    lines, note = read_geo(path)
    if "kind" not in lines.columns:
        raise SystemExit(f"{path}: no kind column")
    a.layer_files["county"] = [str(path)]
    counties_path = path.with_name(path.name.replace("county_lines_", "counties_"))
    extent = tuple(lines.total_bounds)
    kind = lines.kind.astype(str).to_numpy()
    n_overlap = int((kind == "overlap").sum())
    n_county = int((kind == "county").sum())
    n_outer = int((kind == "state_or_coast").sum())
    counties = None
    if counties_path.exists():
        county_polys, _ = read_geo(counties_path, columns=[])
        counties = shapely.union_all(np.asarray(county_polys.geometry.values))
        shapely.prepare(counties)

    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        box = (r.minx, r.miny, r.maxx, r.maxy)
        iv = np.flatnonzero((vt.component == c).to_numpy())
        if counties is None:
            if box_contains(extent, box):
                status = "ok"
                county_note = (f"layer extent {fmt_box(extent)} contains the bbox; "
                               f"{counties_path.name} is not there, so outside_state is not tested")
            else:
                status = "FAIL"
                county_note = (f"layer extent {fmt_box(extent)} does not contain the bbox "
                               f"{fmt_box(box)}, and {counties_path.name} is not there to show "
                               f"the vertices are in-state")
        else:
            points = shapely.points(vt.x.to_numpy()[iv], vt.y.to_numpy()[iv])
            inside = shapely.dwithin(counties, points, TOL_M)
            n_out = int((~inside).sum())
            n_far = 0
            if n_out:
                dist_out = shapely.distance(counties, points[~inside])
                n_far = int((dist_out > a.state_tol_m).sum())
                flag = a.flags.setdefault("outside_state", np.zeros(len(vt), dtype=bool))
                flag[iv[~inside]] = True
            if n_far:
                status = "FAIL"
                county_note = (f"{n_far:,} vertices more than {a.state_tol_m:.0f} m outside "
                               f"the {a.state} counties: the layer does not cover them")
            elif n_out:
                status = "warn"
                county_note = (f"{n_out:,} vertices up to {dist_out.max():.0f} m outside the "
                               f"{a.state} counties (NFHL and TIGER draw the line differently), "
                               f"flagged outside_state; their distances are to {a.state}'s "
                               f"lines only")
            else:
                status = "ok"
                county_note = (f"every vertex is inside a {a.state} county and the state/coast "
                               f"line is in the layer")
        if n_overlap:
            county_note += f"; {n_overlap} 'overlap' lines ignored"
        if note:
            county_note += f"; {note}"
        cov.add(c, "county", status, county_note, n_lines=n_county, n_vertices=len(iv))
        cov.add(c, "state_or_coast", status, "same file and coverage", n_lines=n_outer,
                n_vertices=len(iv))

    row_ids = pd.Series(np.arange(len(lines)), dtype="Int64")
    geom = np.asarray(lines.geometry.values)
    is_county = kind == "county"
    is_outer = kind == "state_or_coast"
    outer_kind = nullable(lines.outer[is_outer]) if "outer" in lines.columns else None
    return [LineSet("county", geom[is_county], row_ids[is_county].reset_index(drop=True)),
            LineSet("state_or_coast", geom[is_outer], row_ids[is_outer].reset_index(drop=True),
                    kind=outer_kind)]


def panel_footprint(edges: np.ndarray):
    """The area a DFIRM's panels cover (its edges polygonized), prepared; None when
    the edges enclose nothing."""
    faces = shapely.get_parts(shapely.polygonize(edges))
    if not len(faces):
        return None
    footprint = shapely.union_all(faces)
    shapely.prepare(footprint)
    return footprint


def load_panel(a, vt, bb, cov: Coverage) -> list[LineSet]:
    """panel_interior ('interior' and 'overlap' edges; kind says which) and
    panel_exterior, of the lobe's own DFIRM only. The id is the row number in
    panel_edges_<ST>.

    FIRM panels overlap across county DFIRMs, so a neighbouring county's sheet edges
    run across a lobe without having anything to do with how it was drawn. 'overlap'
    edges (a panel's side inside another panel of the same DFIRM) are sheet edges in
    the map's interior. Every lobe's DFIRM must have edges. Vertices outside their
    DFIRM's panel footprint (e.g. a Hendry lobe reaching into Lake Okeechobee, mapped
    on neighbouring counties' panels) are flagged in a.flags["outside_own_panels"];
    their distance is to their own DFIRM's exterior."""
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
    edge_dfirm = edges.dfirm_id.astype(str).to_numpy()
    kind = edges.kind.astype(str).to_numpy()
    geom = np.asarray(edges.geometry.values)
    is_interior = np.isin(kind, ("interior", "overlap"))
    is_exterior = kind == "exterior"
    vertex_dfirm = vt.dfirm_id.fillna("").astype(str).to_numpy()   # "": no lobe, a 'vertices' FAIL
    footprints = {}

    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        iv = np.flatnonzero((vt.component == c).to_numpy())
        dfirms = sorted(set(vertex_dfirm[iv]) - {""})
        problems, notes = [], []
        n_out, n_interior, n_exterior = 0, 0, 0
        for d in dfirms:
            jv = iv[vertex_dfirm[iv] == d]
            own = edge_dfirm == d
            if not own.any():
                problems.append(f"DFIRM {d} ({len(jv):,} vertices) has no panel edges in "
                                f"{path.name}")
                continue
            n_interior += int((own & is_interior).sum())
            n_exterior += int((own & is_exterior).sum())
            if d not in footprints:
                footprints[d] = panel_footprint(geom[own])
            if footprints[d] is None:
                notes.append(f"{d}: edges do not polygonize, footprint not checked")
                continue
            points = shapely.points(vt.x.to_numpy()[jv], vt.y.to_numpy()[jv])
            off = ~shapely.dwithin(footprints[d], points, TOL_M)
            n_off = int(off.sum())
            n_out += n_off
            if n_off:
                flag = a.flags.setdefault("outside_own_panels", np.zeros(len(vt), dtype=bool))
                flag[jv[off]] = True
                notes.append(f"{d}: {n_off:,} vertices outside its panel footprint, flagged "
                             f"outside_own_panels (distance is to its exterior edge)")
        if problems:
            status, panel_note = "FAIL", "; ".join(problems)
        else:
            status = "warn" if n_out else "ok"
            panel_note = f"own-DFIRM edges for {', '.join(dfirms)}"
            if notes:
                panel_note += f"; {'; '.join(notes)}"
            else:
                panel_note += "; every vertex inside its DFIRM's panel footprint"
        if note:
            panel_note += f"; {note}"
        cov.add(c, "panel_interior", status, panel_note, n_lines=n_interior, n_vertices=len(iv))
        cov.add(c, "panel_exterior", status, "same file and coverage", n_lines=n_exterior,
                n_vertices=len(iv))

    row_ids = pd.Series(np.arange(len(edges)), dtype="Int64")
    return [LineSet("panel_interior", geom[is_interior],
                    row_ids[is_interior].reset_index(drop=True),
                    kind=nullable(edges.kind[is_interior]), group=edge_dfirm[is_interior],
                    by="dfirm_id"),
            LineSet("panel_exterior", geom[is_exterior],
                    row_ids[is_exterior].reset_index(drop=True), group=edge_dfirm[is_exterior],
                    by="dfirm_id")]


def lonlat_envelope(x: np.ndarray, y: np.ndarray, to_ll: Transformer, margin_m: float) -> tuple:
    """The lon/lat box around EPSG:5070 points, grown by about margin_m on every side."""
    lon, lat = to_ll.transform(x, y)
    dlat = margin_m / M_PER_DEG_LAT
    dlon = dlat / np.cos(np.radians(np.abs(lat).max() + dlat))   # at the poleward edge
    return (lon.min() - dlon, lat.min() - dlat, lon.max() + dlon, lat.max() + dlat)


def load_nhdplus_canals(a, vt, bb, cov: Coverage) -> list[LineSet]:
    """canals from NHDPlus V2 NHDFlowline FTYPE 'CanalDitch' (network and non-network):
    every line meeting a bbox's lon/lat envelope, kept whole. The id is the COMID and
    the kind the FCODE."""
    shapefiles = sorted(a.nhdplus_dir.glob("*/NHDSnapshot/**/NHDFlowline.shp"))
    if not shapefiles:
        cov.add("all", "canals", "FAIL", f"--canals-from-nhdplus: no */NHDSnapshot/**/"
                f"NHDFlowline.shp under {a.nhdplus_dir} (fetch_nhdplus_fl.sh extracts them)")
        return []
    a.layer_files["canals"] = [str(p) for p in shapefiles]
    to_ll = Transformer.from_crs(CRS_WORK, CRS_NHD, always_xy=True)
    unit_boxes = []
    for p in shapefiles:
        with warnings.catch_warnings(action="ignore"):
            info = pyogrio.read_info(p, force_total_bounds=True)
        if CRS.from_user_input(info["crs"]).to_epsg() != CRS_NHD:
            raise SystemExit(f"{p}: CRS {info['crs']}, expected EPSG:{CRS_NHD}")
        unit_boxes.append(shapely.box(*info["total_bounds"]))
    unit_cover = shapely.union_all(unit_boxes)

    parts = []
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        envelope = to_ll.transform_bounds(r.minx, r.miny, r.maxx, r.maxy, densify_pts=101)
        envelope_text = tuple(round(v, 4) for v in envelope)
        in_comp = (vt.component == c).to_numpy()
        n_vertices = int(in_comp.sum())
        status, margin_note = "ok", ""
        if not shapely.contains(unit_cover, shapely.box(*envelope)):
            # The bbox carries a 3 km buffer and its lon/lat envelope is wider still (a
            # 5070 rectangle is skewed in lon/lat), so it can reach past the last flowline,
            # e.g. into the Gulf south of Cape San Blas. Distances are measured from the
            # vertices: if they plus CANAL_MARGIN_M lie inside the units, every canal within
            # that margin of a vertex is loaded, and this is a warning, not a failure.
            near_vertices = lonlat_envelope(vt.x.to_numpy(float)[in_comp],
                                            vt.y.to_numpy(float)[in_comp], to_ll, CANAL_MARGIN_M)
            if not shapely.contains(unit_cover, shapely.box(*near_vertices)):
                cov.add(c, "canals", "FAIL", f"bbox (lon/lat {envelope_text}) and the vertices + "
                        f"{CANAL_MARGIN_M:g} m are not inside the extents of the "
                        f"{len(shapefiles)} NHDPlus units", n_vertices=n_vertices)
                continue
            status = "warn"
            margin_note = (f"; the bbox (lon/lat {envelope_text}) reaches past the NHDPlus "
                           f"units' extents, the vertices + {CANAL_MARGIN_M:g} m do not, so "
                           f"d_canals_m above {CANAL_MARGIN_M:g} m may miss lines beyond the "
                           f"extents")
        found = []
        for p in shapefiles:
            # pyogrio warns that it reads the Measured 3D lines as LineString Z
            with warnings.catch_warnings(action="ignore"):
                unit_canals = pyogrio.read_dataframe(p, columns=["COMID", "FTYPE", "FCODE"],
                                                     where="FTYPE = 'CanalDitch'", bbox=envelope)
            if len(unit_canals):
                found.append(unit_canals)
        n_lines = sum(len(unit_canals) for unit_canals in found)
        parts += found
        cov.add(c, "canals", status, f"NHDPlus V2 NHDFlowline FTYPE CanalDitch (network and "
                f"non-network) from {len(shapefiles)} units; every line meeting the bbox's "
                f"lon/lat envelope, kept whole" + margin_note,
                n_lines=n_lines, n_vertices=n_vertices)

    if not parts:
        return [LineSet("canals", np.empty(0, dtype=object), pd.Series([], dtype="Int64"),
                        kind=pd.Series([], dtype="string"))]
    flowlines = pd.concat(parts, ignore_index=True).drop_duplicates("COMID").reset_index(drop=True)
    flowlines = gpd.GeoDataFrame(flowlines, geometry="geometry", crs=parts[0].crs).to_crs(CRS_WORK)
    geom = shapely.force_2d(np.asarray(flowlines.geometry.values))
    return [LineSet("canals", geom, nullable(flowlines.COMID),
                    kind=nullable(flowlines.FCODE.astype(str)))]


def load_optional(a, source: str, vt, bb, cov: Coverage) -> list[LineSet]:
    """An optional layer (nwi, plss, roads, canals) from --layer-file or found by name;
    polygon layers are measured to their boundaries. Its extent (declared by the file
    when it can) must contain each bbox. A missing file is an 'absent' row."""
    path = a.files.get(source)
    how = "--layer-file"
    if path is None:
        path, how = find_optional(a.gis_dir, source, a.prefix, a.state)
    if source == "canals" and a.canals_from_nhdplus:
        # explicit only: NHDPlus is never a silent fallback for a canals file
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
    extent, extent_how = declared_bounds(path, gdf)
    id_col = pick_column(gdf.columns, ID_CANDIDATES)
    kind_col = pick_column(gdf.columns, KIND_CANDIDATES)
    geom = np.asarray(gdf.geometry.values)
    for c in sorted(vt.component.unique()):
        if c not in bb.index:
            continue
        r = bb.loc[c]
        box = (r.minx, r.miny, r.maxx, r.maxy)
        n_vertices = int((vt.component == c).sum())
        n_lines = int(shapely.intersects(geom, shapely.box(*box)).sum())
        covered = box_contains(extent, box)
        layer_note = f"{path.name} ({how}); {extent_how} {fmt_box(extent)} "
        if covered:
            layer_note += "contains the bbox"
        else:
            layer_note += (f"does not contain the bbox {fmt_box(box)}: re-pull it for these "
                           f"bboxes, or --skip-layer {source}")
        if note:
            layer_note += f"; {note}"
        cov.add(c, source, "ok" if covered else "FAIL", layer_note, n_lines=n_lines,
                n_vertices=n_vertices)
    if id_col:
        ids = nullable(gdf[id_col])
    else:
        ids = pd.Series(np.arange(len(gdf)), dtype="Int64")     # the row number
    kind = nullable(gdf[kind_col].astype(str)) if kind_col else None
    return [LineSet(source, shapely.force_2d(geom), ids, kind=kind)]


def load_source(a, source: str, vt, bb, cov: Coverage) -> list[LineSet]:
    if source == "ssurgo":
        return load_ssurgo(a, vt, bb, cov)
    if source == "county":
        return load_county(a, vt, bb, cov)
    if source == "panel":
        return load_panel(a, vt, bb, cov)
    return load_optional(a, source, vt, bb, cov)


# ----------------------------------------------------------------- measure
def measure(vt: pd.DataFrame, ls: LineSet) -> dict:
    """The output columns of one layer: d_<layer>_m, <layer>_id, <layer>_kind and
    <layer>_<extra>. A NaN distance (and <NA> id) means the layer was loaded and
    passed coverage but has no line for the vertex's group: e.g. a bbox inside one
    county survey has no soil_join_county, a one-panel DFIRM no interior edge."""
    x, y = vt.x.to_numpy(float), vt.y.to_numpy(float)
    d = np.full(len(vt), np.nan)
    j = np.full(len(vt), -1, dtype=np.int64)
    if ls.by is None:
        d, j = nearest(x, y, ls.geom)
    else:
        key = vt[ls.by].fillna("").astype(str).to_numpy()
        group = ls.group.astype(str)
        for value in sorted(set(key)):
            iv = np.flatnonzero(key == value)
            il = np.flatnonzero(group == value)
            if not len(il):
                continue
            dd, jj = nearest(x[iv], y[iv], ls.geom[il])
            d[iv] = dd
            j[iv] = np.where(jj >= 0, il[np.maximum(jj, 0)], -1)   # group index -> layer index
    cols = {f"d_{ls.name}_m": d, f"{ls.name}_id": take(ls.ids, j)}
    if ls.kind is not None:
        cols[f"{ls.name}_kind"] = take(ls.kind, j)
    for name, values in ls.extra.items():
        cols[f"{ls.name}_{name}"] = take(values, j)
    return cols


def margins(vt: pd.DataFrame, bb: pd.DataFrame) -> np.ndarray:
    """Each vertex's distance to the edge of its own bbox (>= 3 km by lobes.py's
    buffer). A per-bbox layer loads every line inside the bbox, so a distance up to
    the margin is exact; a larger one is an upper bound."""
    r = bb.reindex(vt.component.to_numpy())
    x, y = vt.x.to_numpy(float), vt.y.to_numpy(float)
    to_left = x - r.minx.to_numpy()
    to_right = r.maxx.to_numpy() - x
    to_bottom = y - r.miny.to_numpy()
    to_top = r.maxy.to_numpy() - y
    return np.minimum.reduce([to_left, to_right, to_bottom, to_top])


# ----------------------------------------------------------------- main
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vertices", required=True, type=Path,
                    help="<prefix>_vertices_z.parquet from pull_dem.py sample")
    ap.add_argument("--lobes", type=Path,
                    help="default: <prefix>_lobes.parquet beside --vertices")
    ap.add_argument("--bboxes", type=Path,
                    help="default: <prefix>_bboxes.parquet beside --vertices")
    ap.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--prefix", help="default: from the --vertices file name")
    ap.add_argument("--out", type=Path,
                    help="default: results/<today>/boundary/<prefix>_vertex_layers.parquet")
    ap.add_argument("--skip-layer", action="append", default=[], metavar="NAME",
                    help=f"skip a source ({', '.join(SOURCES)}) or one layer "
                         f"({', '.join(layer_names())}); repeatable. A missing file of a "
                         f"required source ({', '.join(REQUIRED)}) fails the run unless skipped")
    ap.add_argument("--layer-file", action="append", default=[], metavar="SOURCE=PATH",
                    help="explicit file for a source (for ssurgo the mulines file; mupolygons "
                         "is read beside it); repeatable")
    ap.add_argument("--canals-from-nhdplus", action="store_true",
                    help="canals = NHDPlus V2 NHDFlowline FTYPE CanalDitch under --nhdplus-dir "
                         "(only when no canals file exists)")
    ap.add_argument("--nhdplus-dir", type=Path, default=NHD_DIR)
    ap.add_argument("--allow-dem-missing", action="store_true",
                    help="keep vertices without z, z_in or z_out (dem_ok False) instead of failing")
    ap.add_argument("--state-tol-m", type=float, default=1000.0,
                    help="vertices up to this far outside the state's counties are flagged "
                         "outside_state, not failed")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    a.prefix = a.prefix or prefix_of(a.vertices)
    a.lobes_p = a.lobes or a.vertices.with_name(f"{a.prefix}_lobes.parquet")
    a.bboxes_p = a.bboxes or a.vertices.with_name(f"{a.prefix}_bboxes.parquet")
    if a.out is None:
        a.out = (RESULTS / date.today().isoformat() / "boundary"
                 / f"{a.prefix}_vertex_layers.parquet")
    layers = layer_names()
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
        source, _, path = kv.partition("=")
        if source not in SOURCES or not path:
            ap.error(f"--layer-file {kv}: expected SOURCE=PATH with SOURCE in {sorted(SOURCES)}")
        a.files[source] = Path(path)
    a.layer_files = {}                               # source -> files read, filled by the loaders
    a.flags = {}                                     # per-vertex booleans written with the table
    return a


def load_layers(a, vt, bb, cov: Coverage) -> list[LineSet]:
    """Every source's LineSets except the skipped ones, which get 'skipped' rows."""
    sets = []
    for source in SOURCES:
        if source in a.skip_sources:
            for layer in SOURCES[source]:
                cov.add("all", layer, "skipped", f"--skip-layer {source}")
            continue
        t1 = time.time()
        got = load_source(a, source, vt, bb, cov)
        for ls in got:
            if ls.name in a.skip_layers:
                cov.skip(ls.name, f"--skip-layer {ls.name}")
            else:
                sets.append(ls)
        if got:
            summary = ", ".join(f"{ls.name} {len(ls.geom):,} lines" for ls in got)
            print(f"  {source}: {summary} [{time.time() - t1:.1f}s]", flush=True)
    return sets


def run_metadata(a) -> dict:
    return dict(prefix=a.prefix, state=a.state, vertices=str(a.vertices), lobes=str(a.lobes_p),
                bboxes=str(a.bboxes_p), gis_dir=str(a.gis_dir), layer_files=a.layer_files,
                skip=sorted(a.skip_layer), canals_from_nhdplus=a.canals_from_nhdplus,
                nhdplus_dir=str(a.nhdplus_dir) if a.canals_from_nhdplus else None,
                allow_dem_missing=a.allow_dem_missing, state_tol_m=a.state_tol_m,
                piece_segs=PIECE_SEGS, tol_m=TOL_M,
                crs=f"EPSG:{CRS_WORK}", created=datetime.now().isoformat(timespec="seconds"))


def report_failures(cov: Coverage, meta: dict, cov_path: Path) -> None:
    cov.print()
    cov_path.write_text(json.dumps(jsonable(dict(status="FAILED", **meta, coverage=cov.rows)),
                                   indent=1))
    fails = cov.failed()
    print(f"\nFAILED: {len(fails)} coverage check(s) failed; nothing measured, no table "
          f"written. Coverage: {cov_path}", file=sys.stderr)
    for r in fails:
        where = "all components" if r["component"] == "all" else f"component {r['component']}"
        print(f"  {where}, {r['layer']}: {r['note']}", file=sys.stderr)


def build_table(raw, vt, bb, sets: list[LineSet], flags: dict, cov: Coverage) -> pd.DataFrame:
    """The input columns as read, then dfirm_id, soil_ok, the flags and every layer's
    columns. Adds beyond_margin to the coverage rows of the per-bbox layers."""
    table = raw.copy()
    table["dfirm_id"] = vt["dfirm_id"]
    if "soil_ok" in vt:
        table["soil_ok"] = vt["soil_ok"]
    for name in ("outside_state", "outside_own_panels"):
        table[name] = flags.get(name, np.zeros(len(vt), dtype=bool))
    margin = margins(vt, bb)
    for ls in sets:
        t1 = time.time()
        cols = measure(vt, ls)
        for col, values in cols.items():
            table[col] = values
        if ls.name in BBOX_LIMITED:
            # NaN counts too: no line inside the bbox says nothing about lines beyond it
            d = cols[f"d_{ls.name}_m"]
            beyond = np.isnan(d) | (d > margin)
            for c in sorted(vt.component.unique()):
                in_comp = (vt.component == c).to_numpy()
                cov.update(c, ls.name, beyond_margin=int((beyond & in_comp).sum()))
        print(f"  measured {ls.name} [{time.time() - t1:.1f}s]", flush=True)
    return table


def write_table(table: pd.DataFrame, vertices: Path, out: Path, meta: dict) -> None:
    """The table, carrying the input's parquet metadata (e.g. b"pull_dem") and ours
    under b"edge_layers", written to a temporary file and moved into place."""
    upstream = {}
    for key, value in (pq.read_schema(vertices).metadata or {}).items():
        if key != b"pandas":                   # pandas' entry describes the new table instead
            upstream[key] = value
    tbl = pa.Table.from_pandas(table, preserve_index=False)
    metadata = {**(tbl.schema.metadata or {}), **upstream,
                b"edge_layers": json.dumps(jsonable(meta)).encode()}
    tbl = tbl.replace_schema_metadata(metadata)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    pq.write_table(tbl, tmp)
    os.replace(tmp, out)


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
    sets = load_layers(a, vt, bb, cov)
    meta = run_metadata(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if cov.failed():
        report_failures(cov, meta, cov_path)
        return 1

    table = build_table(raw, vt, bb, sets, a.flags, cov)
    cov.print()
    meta["coverage"] = cov.rows
    write_table(table, a.vertices, a.out, meta)
    cov_path.write_text(json.dumps(jsonable(dict(status="ok", **meta)), indent=1))
    n_warn = sum(r["status"] == "warn" for r in cov.rows)
    print(f"\n[{a.prefix}] wrote {a.out} ({len(table):,} rows, {len(table.columns)} columns, "
          f"{a.out.stat().st_size / 1e6:.1f} MB) and {cov_path.name}"
          + (f"; {n_warn} warning(s) above" if n_warn else "")
          + f" [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
