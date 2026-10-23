#!/usr/bin/env python3
"""County lines and FIRM panel edges as directed lines, with the polygon on each side.

These boundaries were drawn for administrative or map-sheet reasons, not by the terrain;
aotil/edge_layers.py measures how far each lobe vertex lies from them (class M of
docs/REGION-ANALYSIS-FLORIDA.md section 3.3). Inputs: the national TIGER/Line county zip,
downloaded once into --raw-dir, and the FIRM panels in --panels. Outputs, in EPSG:5070:

    <gis-dir>/county_lines_<ST>.parquet   county boundaries, FIPS and name on each side
    <gis-dir>/counties_<ST>.parquet       the state's county polygons
    <gis-dir>/panel_edges_<ST>.parquet    panel edges within each DFIRM

    python pull_gis_lines.py --selftest              # synthetic squares, nothing kept
    python pull_gis_lines.py --state FL --dry-run    # URL, size and cache check only
    python pull_gis_lines.py --state FL              # both layers
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
from shapely import STRtree

RAW_TIGER = Path("data/raw/tiger")            # TIGER download cache
NFHL_DIR = Path("data/interim/nfhl_states")   # pan_<ST>.parquet
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
GRID_M = 0.001              # snap-rounding grid for noding, metres
# Near-duplicate tolerance, metres. In several Florida DFIRMs a panel's side and its
# neighbour's differ by up to 8 cm at one corner, so the shared edge is two lines 0-8 cm
# apart. 10 cm merges them but keeps open the narrowest real feature, a spur under 1 m
# wide on panel 12081C0333E (at 25-50 cm its two sides merge).
TOL_M = 0.10
NEIGHBOUR_BUFFER_DEG = 0.001   # reach for other states' counties; TIGER is in lon/lat

FIPS = {"FL": "12"}


# =========================================================================
# edge extraction
# =========================================================================
def _join_by_segment(seg: np.ndarray, poly: np.ndarray, labels: np.ndarray,
                     n: int) -> np.ndarray:
    """For each of n segments, the sorted ';'-joined labels of its polygons ('' for none).

    seg[k] and poly[k] are one (segment, polygon) pair; almost every segment has one.
    """
    out = np.full(n, "", dtype=object)
    if len(seg) == 0:
        return out
    count = np.bincount(seg, minlength=n)
    single = count[seg] == 1
    out[seg[single]] = labels[poly[single]]

    grouped = {}
    for segment, label in zip(seg[~single], labels[poly[~single]]):
        grouped.setdefault(segment, []).append(label)
    for segment, group in grouped.items():
        out[segment] = ";".join(sorted(group))
    return out


def _near_duplicates(p1: np.ndarray, p2: np.ndarray, seglen: np.ndarray,
                     tol: float) -> np.ndarray:
    """Mask of segments that run alongside a longer segment within tol.

    Segment i goes when both its ends lie within tol of segment j and at least half of
    i projects onto j: the second condition keeps a short segment that merely continues
    j from its end. Ties in length go to the lower index, so of two twins one survives.
    """
    segs = shapely.linestrings(np.stack([p1, p2], axis=1))
    i, j = STRtree(segs).query(segs, predicate="dwithin", distance=tol)
    shorter = seglen[i] < seglen[j]
    same_length_higher_index = (seglen[i] == seglen[j]) & (i > j)
    candidate = (i != j) & (shorter | same_length_higher_index)
    i, j = i[candidate], j[candidate]

    start_near = shapely.distance(shapely.points(p1[i]), segs[j]) <= tol
    end_near = shapely.distance(shapely.points(p2[i]), segs[j]) <= tol
    both_ends_near = start_near & end_near
    i, j = i[both_ends_near], j[both_ends_near]

    # Positions of i's ends along j, clipped to j: the overlap is hi - lo.
    direction = (p2[j] - p1[j]) / seglen[j][:, None]
    t1 = ((p1[i] - p1[j]) * direction).sum(axis=1)
    t2 = ((p2[i] - p1[j]) * direction).sum(axis=1)
    lo = np.clip(np.minimum(t1, t2), 0, seglen[j])
    hi = np.clip(np.maximum(t1, t2), 0, seglen[j])
    drop = np.zeros(len(p1), dtype=bool)
    drop[i[(hi - lo) >= 0.5 * seglen[i]]] = True
    return drop


def _polygonal(v):
    """The polygonal part of a make_valid result. make_valid turns a zero-width spike
    into a GeometryCollection, which has no boundary, so the rest would be lost."""
    polygon_types = (3, 6)    # Polygon, MultiPolygon
    if shapely.get_type_id(v) in polygon_types:
        return v
    parts = shapely.get_parts(v)
    return shapely.union_all(parts[np.isin(shapely.get_type_id(parts), polygon_types)])


def _snapped_polygons(geoms, ids, grid: float):
    """(valid polygons snapped to the grid with empties dropped, their ids as str,
    number of invalid polygons repaired)."""
    polys = np.asarray(geoms, dtype=object).copy()
    ids = np.asarray(ids).astype(str).astype(object)
    invalid = ~shapely.is_valid(polys)
    if invalid.any():
        polys[invalid] = [_polygonal(v) for v in shapely.make_valid(polys[invalid])]
    polys = shapely.set_precision(polys, grid)
    keep = ~shapely.is_empty(polys)
    return polys[keep], ids[keep], int(invalid.sum())


def _noded_segments(polys: np.ndarray, grid: float):
    """(start points, end points, lengths) of all boundaries as two-point segments.

    Neighbouring polygons rarely share vertex lists (a panel's side can abut two
    smaller panels with no vertex where they meet), so raw ring segments do not pair
    up. A union with snap-rounding at the grid nodes the boundaries at every junction
    and keeps each coincident piece once.
    """
    noded = shapely.union_all(shapely.boundary(polys), grid_size=grid)
    xy, part = shapely.get_coordinates(shapely.get_parts(noded), return_index=True)
    same_line = part[1:] == part[:-1]
    p1 = xy[:-1][same_line]
    p2 = xy[1:][same_line]
    delta = p2 - p1
    seglen = np.hypot(delta[:, 0], delta[:, 1])
    nonzero = seglen > 0
    return p1[nonzero], p2[nonzero], seglen[nonzero]


def _segment_sides(polys, ids, p1, p2, seglen, offset: float):
    """(left, right, cover, owners) of each segment.

    Each segment is probed `offset` metres to the left and right of its midpoint. A
    polygon holding only the left probe owns the segment on the left (and vice versa);
    one holding both probes covers it. left, right and cover are ';'-joined ids ('' for
    none) and owners counts the left and right owners.
    """
    n = len(p1)
    npoly = len(polys)
    mid = (p1 + p2) / 2
    delta = p2 - p1
    normal = np.column_stack([-delta[:, 1], delta[:, 0]]) / seglen[:, None]
    probes = shapely.points(np.concatenate([mid + offset * normal, mid - offset * normal]))
    # A tree of probe points queried with each polygon: point-in-polygon on a
    # 20,000-vertex county is then logarithmic instead of linear.
    poly_idx, probe_idx = STRtree(probes).query(polys, predicate="contains")

    # Probes 0..n-1 are the left ones. Each (segment, polygon) hit is encoded as the
    # integer segment * npoly + polygon, so numpy set operations compare the sides.
    on_left = probe_idx < n
    left_hits = np.unique(probe_idx[on_left] * npoly + poly_idx[on_left])
    right_hits = np.unique((probe_idx[~on_left] - n) * npoly + poly_idx[~on_left])
    left_seg, left_poly = np.divmod(np.setdiff1d(left_hits, right_hits), npoly)
    right_seg, right_poly = np.divmod(np.setdiff1d(right_hits, left_hits), npoly)
    cover_seg, cover_poly = np.divmod(np.intersect1d(left_hits, right_hits), npoly)

    left = _join_by_segment(left_seg, left_poly, ids, n)
    right = _join_by_segment(right_seg, right_poly, ids, n)
    cover = _join_by_segment(cover_seg, cover_poly, ids, n)
    owners = np.bincount(left_seg, minlength=n) + np.bincount(right_seg, minlength=n)
    return left, right, cover, owners


def _needs_swap(left: np.ndarray, right: np.ndarray, primary) -> np.ndarray:
    """Mask of segments to reverse so an owner is on the left; between two owners the
    primary one goes left, and between equals the lower id."""
    has_left = left != ""
    has_right = right != ""
    swap = ~has_left
    if primary is None:
        tie = has_left & has_right
    else:
        left_primary = np.array([x in primary for x in left])
        right_primary = np.array([x in primary for x in right])
        swap |= has_right & right_primary & ~left_primary
        tie = has_left & has_right & (left_primary == right_primary)
    swap |= tie & (left > right)
    return swap


def _lines_by_sides(p1, p2, left, right, cover, kind) -> pd.DataFrame:
    """Merge the segments of each (left, right, cover, kind) group into directed lines."""
    segs = shapely.linestrings(np.stack([p1, p2], axis=1))
    keys = pd.DataFrame(dict(left=left, right=right, cover=cover, kind_raw=kind))
    groups = keys.groupby(["left", "right", "cover", "kind_raw"], sort=True).indices
    rows, lines = [], []
    for key, members in groups.items():
        merged = shapely.line_merge(shapely.multilinestrings(segs[members]), directed=True)
        for line in shapely.get_parts(merged):
            rows.append(key)
            lines.append(line)
    out = pd.DataFrame(rows, columns=["left", "right", "cover", "kind_raw"])
    out["geometry"] = np.array(lines, dtype=object)
    out["length_m"] = shapely.length(out["geometry"].to_numpy())
    return out


def polygon_edges(geoms, ids, grid: float = GRID_M, tol: float = TOL_M,
                  primary=None) -> tuple[pd.DataFrame, dict]:
    """Boundary lines of a set of polygons with the polygon id on each side.

    Returns a frame with left, right, cover (';'-joined ids, '' for none), kind_raw
    ('shared', 'outer' or 'overlap'), geometry (LineStrings with the left owner on the
    left) and length_m, plus a dict of counts and checks. `primary` is a set of ids
    that goes on the left of an edge it shares with a non-primary id. Every polygon's
    perimeter should equal the length of the edges it owns, so perimeter_gap_m is ~0
    up to the dropped near-duplicates' offsets.
    """
    polys, ids, repaired = _snapped_polygons(geoms, ids, grid)
    stats = dict(polygons=len(polys), repaired=repaired, grid_m=grid, tol_m=tol)

    p1, p2, seglen = _noded_segments(polys, grid)
    stats["segments"] = len(p1)
    twin = _near_duplicates(p1, p2, seglen, tol)
    stats["near_duplicate_segments"] = int(twin.sum())
    stats["near_duplicate_length_m"] = float(seglen[twin].sum())
    p1, p2, seglen = p1[~twin], p2[~twin], seglen[~twin]

    # Probes sit beyond any dropped twin, and at least a few grid cells off the line.
    offset = max(2 * tol, 5 * grid)
    left, right, cover, owners = _segment_sides(polys, ids, p1, p2, seglen, offset)
    # A segment with the same polygons on both probes is not an edge at probe scale.
    edge = owners > 0
    stats["dropped_segments"] = int((~edge).sum())
    stats["dropped_length_m"] = float(seglen[~edge].sum())
    stats["perimeter_m"] = float(shapely.length(shapely.boundary(polys)).sum())
    stats["owned_length_m"] = float((seglen * owners).sum())
    p1, p2 = p1[edge], p2[edge]
    left, right, cover = left[edge], right[edge], cover[edge]

    swap = _needs_swap(left, right, primary)
    p1[swap], p2[swap] = p2[swap], p1[swap]
    left[swap], right[swap] = right[swap], left[swap]
    kind = np.where(cover != "", "overlap", "outer")
    kind = np.where(right != "", "shared", kind)

    out = _lines_by_sides(p1, p2, left, right, cover, kind)
    stats["lines"] = len(out)
    stats["perimeter_gap_m"] = stats["perimeter_m"] - stats["owned_length_m"]
    return out, stats


def _blank(s: pd.Series) -> pd.Series:
    """'' (no polygon on that side) as a missing value."""
    return s.replace("", None)


def county_lines(counties, state_fips: str, grid: float = GRID_M, tol: float = TOL_M):
    """County boundary lines of one state, with FIPS and names on each side.

    `counties` is a GeoDataFrame in EPSG:5070 with GEOID, NAME and STATEFP: the state's
    counties plus the other states' counties that touch them, so the outer edge splits
    into the state line (outer 'state') and the seaward limit (outer 'coast'). kind is
    'county' between two in-state counties, 'overlap' inside another county, else
    'state_or_coast'. In-state counties go on the left, the lower FIPS between two.
    TIGER county polygons include water, so 'coast' is the offshore limit, not the
    shoreline. Many Florida county lines follow rivers, so a county line can coincide
    with a floodplain for a hydrological reason.
    """
    geoid = counties.GEOID.astype(str).to_numpy()
    statefp = counties.STATEFP.astype(str).to_numpy()
    instate = set(geoid[statefp == state_fips])
    edges, stats = polygon_edges(np.asarray(counties.geometry.values), geoid, grid, tol,
                                 primary=instate)
    touches_state = edges.left.isin(instate) | edges.right.isin(instate)
    edges = edges[touches_state].reset_index(drop=True)

    name = dict(zip(geoid, counties.NAME.astype(str)))
    both_in_state = edges.left.isin(instate) & edges.right.isin(instate)
    overlap = edges.kind_raw == "overlap"
    has_right = edges.right != ""
    # Each np.where overrides the one before it, so the last condition wins.
    kind = np.where(overlap, "overlap", "state_or_coast")
    kind = np.where(both_in_state, "county", kind)
    outer = np.where(overlap, None, "coast")
    outer = np.where(has_right, "state", outer)
    outer = np.where(both_in_state, None, outer)

    out = pd.DataFrame({
        "fips_left": _blank(edges.left), "fips_right": _blank(edges.right),
        "name_left": edges.left.map(name), "name_right": edges.right.map(name),
        "kind": kind,
        "outer": outer,
        "fips_cover": _blank(edges.cover),
        "length_m": edges.length_m,
    })
    stats["counties_in_state"] = len(instate)
    stats["counties_neighbouring"] = len(geoid) - len(instate)
    return gpd.GeoDataFrame(out, geometry=edges.geometry.to_numpy(), crs=CRS_WORK), stats


def panel_edges(pan, grid: float = GRID_M, tol: float = TOL_M):
    """FIRM panel edges, extracted within each DFIRM.

    Florida's panels overlap across county DFIRMs (about 65,000 km2 of pairwise
    overlap), so one county's sheet edges run through the next county's map. An edge is
    a sheet edge only for its own map: every row carries dfirm_id, and coincident edges
    of different DFIRMs are kept once per DFIRM. kind is 'interior' (two panels),
    'exterior' (the outline of the DFIRM's panels) or 'overlap' (a panel's side inside
    another panel of the same DFIRM; pan_cover names it).
    """
    scale = {}
    if "SCALE" in pan.columns:
        scale = dict(zip(pan.FIRM_PAN.astype(str), pan.SCALE))
    parts, total = [], {}
    for dfirm, sub in pan.groupby("DFIRM_ID", sort=True):
        edges, stats = polygon_edges(np.asarray(sub.geometry.values), sub.FIRM_PAN.to_numpy(),
                                     grid, tol)
        edges.insert(0, "dfirm_id", dfirm)
        parts.append(edges)
        for key, value in stats.items():
            if key not in ("grid_m", "tol_m"):
                total[key] = total.get(key, 0) + value
    edges = pd.concat(parts, ignore_index=True)

    kind = edges.kind_raw.map({"shared": "interior", "outer": "exterior", "overlap": "overlap"})
    out = pd.DataFrame({
        "dfirm_id": edges.dfirm_id, "pan_left": _blank(edges.left),
        "pan_right": _blank(edges.right),
        "kind": kind, "pan_cover": _blank(edges.cover),
        "scale_left": edges.left.map(scale), "scale_right": edges.right.map(scale),
        "length_m": edges.length_m,
    })
    total["dfirms"] = int(pan.DFIRM_ID.nunique())
    total["grid_m"] = grid
    total["tol_m"] = tol
    return gpd.GeoDataFrame(out, geometry=edges.geometry.to_numpy(), crs=CRS_WORK), total


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
    """The state's counties and the other states' counties touching them, EPSG:5070."""
    src = f"/vsizip/{Path(zip_path).resolve()}"
    cols = ["STATEFP", "GEOID", "NAME"]
    state = pyogrio.read_dataframe(src, columns=cols, where=f"STATEFP = '{state_fips}'")
    if not len(state):
        sys.exit(f"{zip_path}: no counties with STATEFP {state_fips}")
    reach = shapely.buffer(shapely.union_all(np.asarray(state.geometry.values)),
                           NEIGHBOUR_BUFFER_DEG)
    other = pyogrio.read_dataframe(src, columns=cols, where=f"STATEFP <> '{state_fips}'",
                                   bbox=tuple(shapely.bounds(reach)))
    other = other[shapely.intersects(reach, np.asarray(other.geometry.values))]
    both = pd.concat([state, other], ignore_index=True)
    both = gpd.GeoDataFrame(both, geometry="geometry", crs=state.crs).to_crs(CRS_WORK)
    print(f"  {len(state)} counties with STATEFP {state_fips}, {len(other)} neighbouring "
          f"counties in {sorted(other.STATEFP.unique())}; source CRS {state.crs.to_epsg()}",
          flush=True)
    return both


def write_geoparquet(gdf, path: Path, meta: dict) -> None:
    """GeoParquet with meta stored as JSON under the 'pull_gis_lines' schema key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    t = pq.read_table(path)
    metadata = dict(t.schema.metadata or {})
    metadata[b"pull_gis_lines"] = json.dumps(meta, sort_keys=True, default=str).encode()
    pq.write_table(t.replace_schema_metadata(metadata), path)


def _print_checks(stats: dict) -> None:
    print(f"    {stats['segments']:,} noded segments -> {stats['lines']:,} lines; dropped "
          f"{stats['near_duplicate_segments']:,} near-duplicates "
          f"({stats['near_duplicate_length_m'] / 1e3:,.3f} km) and "
          f"{stats['dropped_segments']:,} segments ({stats['dropped_length_m']:.2f} m) with "
          f"the same polygons on both sides", flush=True)
    print(f"    perimeter check: sum of polygon perimeters {stats['perimeter_m'] / 1e3:,.3f} km, "
          f"edge length x owners {stats['owned_length_m'] / 1e3:,.3f} km, gap "
          f"{stats['perimeter_gap_m']:.3f} m", flush=True)
    if not np.isfinite(stats["perimeter_gap_m"]):
        print("    WARNING: the perimeter check failed: a polygon has no boundary, so its "
              "edges are missing", flush=True)


def _by_kind(gdf) -> str:
    """'<kind> <n> lines <length> km' for each kind, in kind order."""
    totals = gdf.groupby("kind").length_m.agg(["size", "sum"])
    parts = []
    for kind, row in totals.iterrows():
        parts.append(f"{kind} {int(row['size']):,} lines {row['sum'] / 1e3:,.1f} km")
    return ", ".join(parts)


# =========================================================================
# the two layers
# =========================================================================
def run_counties(args, fips: str) -> None:
    """Write county_lines_<ST>.parquet and counties_<ST>.parquet unless both exist."""
    out = args.gis_dir / f"county_lines_{args.state}.parquet"
    poly_out = args.gis_dir / f"counties_{args.state}.parquet"
    print(f"\n[{args.state}] county lines (TIGER/Line {args.tiger_year})", flush=True)
    if out.exists() and poly_out.exists() and not args.overwrite:
        print(f"  {out} exists -- skipping (use --overwrite)")
        return
    url = TIGER_URL.format(year=args.tiger_year)
    zip_path = download(url, args.raw_dir / Path(url).name, args.dry_run)
    if zip_path is None:
        return
    t0 = time.time()
    counties = read_counties(zip_path, fips)
    lines, stats = county_lines(counties, fips, args.grid_m, args.tol_m)
    _print_checks(stats)
    meta = dict(source=url, tiger_year=args.tiger_year, state=args.state, statefp=fips, **stats)
    write_geoparquet(lines, out, meta)
    in_state = counties[counties.STATEFP == fips].reset_index(drop=True)
    write_geoparquet(in_state, poly_out,
                     dict(source=url, tiger_year=args.tiger_year, state=args.state))
    print(f"  {_by_kind(lines)}")
    print(f"  wrote {out} and {poly_out}  ({time.time() - t0:.0f}s)")


def run_panels(args) -> None:
    """Write panel_edges_<ST>.parquet unless it exists."""
    out = args.gis_dir / f"panel_edges_{args.state}.parquet"
    print(f"\n[{args.state}] FIRM panel edges from {args.panels}", flush=True)
    if out.exists() and not args.overwrite:
        print(f"  {out} exists -- skipping (use --overwrite)")
        return
    t0 = time.time()
    pan = gpd.read_parquet(args.panels).to_crs(CRS_WORK)
    edges, stats = panel_edges(pan, args.grid_m, args.tol_m)
    _print_checks(stats)
    print(f"  {stats['dfirms']} DFIRMs, {len(pan):,} panels ({stats['repaired']} repaired "
          f"after reprojection); {_by_kind(edges)}")
    write_geoparquet(edges, out, dict(source=str(args.panels), state=args.state, **stats))
    print(f"  wrote {out}  ({time.time() - t0:.0f}s)")


# =========================================================================
# self-test: three squares with a T-junction, through the same functions
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


def _selftest_county_lines(counties) -> list[bool]:
    print("county_lines on three squares (C in another state):")
    lines, stats = county_lines(counties, "12")
    results = []
    length_by_kind = lines.groupby("kind").length_m.sum().to_dict()
    results.append(_check("county length", length_by_kind.get("county", 0), 1000.0))
    results.append(_check("state_or_coast length", length_by_kind.get("state_or_coast", 0),
                          6000.0))
    length_by_outer = lines.groupby("outer").length_m.sum().to_dict()
    results.append(_check("state-line length", length_by_outer.get("state", 0), 2000.0))
    results.append(_check("coast length", length_by_outer.get("coast", 0), 4000.0))
    results.append(_check("rows (A|B, A|C, B|C, coast A, coast B)", len(lines), 5))
    pairs = sorted(zip(lines.fips_left, lines.fips_right.fillna("-")))
    results.append(_check("sides", pairs, [("12001", "-"), ("12001", "12003"),
                                           ("12001", "13049"), ("12003", "-"),
                                           ("12003", "13049")]))

    # A point 1 m to the left of each line, 30% along it, must lie in the left county.
    county_by_fips = dict(zip(counties.GEOID, counties.geometry))
    sides_ok = True
    for row in lines.itertuples():
        p = shapely.line_interpolate_point(row.geometry, 0.3, normalized=True)
        q = shapely.line_interpolate_point(row.geometry, 0.3 + 1e-3, normalized=True)
        d = np.array([q.x - p.x, q.y - p.y])
        d /= np.hypot(*d)
        left_point = shapely.Point(p.x - d[1], p.y + d[0])
        sides_ok &= county_by_fips[row.fips_left].contains(left_point)
    results.append(_check("left FIPS is on the left of every line", bool(sides_ok), True))
    results.append(_check("perimeter gap (m)", round(stats["perimeter_gap_m"], 6), 0.0))
    return results


def _selftest_panel_edges() -> list[bool]:
    print("panel_edges on the same squares in one DFIRM, plus one panel of another:")
    pan = gpd.GeoDataFrame(dict(DFIRM_ID=["12001C"] * 3 + ["12003C"],
                                FIRM_PAN=["P1", "P2", "P3", "Q1"],
                                SCALE=["6000", "6000", "12000", "6000"]),
                           geometry=_three_squares() + [shapely.box(1500, 0, 2500, 1000)],
                           crs=CRS_WORK)
    edges, stats = panel_edges(pan)
    results = []
    length = edges.groupby(["dfirm_id", "kind"]).length_m.sum().to_dict()
    results.append(_check("12001C interior", length.get(("12001C", "interior"), 0), 3000.0))
    results.append(_check("12001C exterior", length.get(("12001C", "exterior"), 0), 8000.0))
    results.append(_check("12003C exterior (overlap with 12001C ignored)",
                          length.get(("12003C", "exterior"), 0), 4000.0))
    interior = edges[edges.kind == "interior"]
    results.append(_check("interior pairs", sorted(zip(interior.pan_left, interior.pan_right)),
                          [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]))
    results.append(_check("perimeter gap (m)", round(stats["perimeter_gap_m"], 6), 0.0))

    print("panel_edges on two panels whose shared side is 8 mm off at one corner:")
    wedge = gpd.GeoDataFrame(dict(DFIRM_ID=["12009C"] * 2, FIRM_PAN=["R1", "R2"]), geometry=[
        shapely.box(0, 0, 1000, 1000),
        shapely.Polygon([(1000, 0), (2000, 0), (2000, 1000), (1000.008, 1000)])], crs=CRS_WORK)
    edges, stats = panel_edges(wedge)
    length = edges.groupby("kind").length_m.sum().to_dict()
    results.append(_check("interior length, counted once",
                          round(length.get("interior", 0), 2), 1000.0))
    results.append(_check("exterior length", round(length.get("exterior", 0), 2), 5999.99))
    results.append(_check("near-duplicate segments dropped", stats["near_duplicate_segments"], 1))
    results.append(_check("perimeter gap (m)", round(stats["perimeter_gap_m"], 6), 0.0))
    return results


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
    results.append(_check("counties read (2 in state + 1 touching; the far one dropped)",
                          sorted(got.GEOID), ["12001", "12003", "13049"]))
    results.append(_check("CRS", got.crs.to_epsg(), CRS_WORK))
    return results


def selftest(tmp_root: Path) -> int:
    """Run the checks on synthetic squares; the only file written is a temp dir under
    tmp_root, removed afterwards. Returns the exit code."""
    counties = gpd.GeoDataFrame(dict(STATEFP=["12", "12", "13"],
                                     GEOID=["12001", "12003", "13049"], NAME=["A", "B", "C"]),
                                geometry=_three_squares(), crs=CRS_WORK)
    results = _selftest_county_lines(counties)
    results += _selftest_panel_edges()
    results += _selftest_read_counties(counties, tmp_root)
    ok = all(results)
    print("selftest " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="FL")
    parser.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_TIGER, help="TIGER download cache")
    parser.add_argument("--panels", type=Path,
                        help="default data/interim/nfhl_states/pan_<ST>.parquet")
    parser.add_argument("--tiger-year", type=int, default=TIGER_YEAR)
    parser.add_argument("--layers", default="counties,panels")
    parser.add_argument("--grid-m", type=float, default=GRID_M,
                        help="snap grid for noding, metres")
    parser.add_argument("--tol-m", type=float, default=TOL_M,
                        help="near-duplicate tolerance, metres")
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
    if args.panels is None:
        args.panels = NFHL_DIR / f"pan_{args.state}.parquet"
    layers = {x.strip() for x in args.layers.split(",") if x.strip()}
    if not layers or layers - {"counties", "panels"}:
        sys.exit(f"--layers must be a subset of counties,panels, got {args.layers!r}")

    if "counties" in layers:
        run_counties(args, FIPS[args.state])
    if "panels" in layers:
        if args.dry_run:
            print(f"\n[{args.state}] FIRM panel edges: dry run, nothing built")
        else:
            run_panels(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
