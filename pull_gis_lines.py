#!/usr/bin/env python3
"""
pull_gis_lines.py -- county lines and FIRM map-sheet edges as lines (audit brief D5).

Class M of the boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md section 3.3,
step 4): lines that were drawn for administrative or cartographic reasons and
that no covariate carries. A Zone A edge that runs along one of them was drawn by
the map's bookkeeping, not by the terrain. Both layers are written as lines in
EPSG:5070 with the polygon on each side, because D7 measures distances from lobe
vertices to the nearest line and reports what lies across it.

    <gis-dir>/county_lines_<ST>.parquet   TIGER/Line county boundaries
    <gis-dir>/counties_<ST>.parquet       the county polygons they came from
    <gis-dir>/panel_edges_<ST>.parquet    FIRM panel edges from pan_<ST>.parquet

DECISIONS, EACH WITH ITS REASON
===============================

1. EDGES ARE DERIVED FROM THE POLYGONS, ROBUSTLY. Neighbouring polygons rarely
   share vertex lists: a 1:12,000 panel's side abuts two 1:6,000 panels and has
   no vertex where they meet (a T-junction), so pairing raw ring segments would
   miss most shared edges. Instead:
     a. coordinates are snapped to a --grid-m grid (default 1 mm, set_precision),
        so vertices that should coincide do;
     b. all boundaries are unioned with snap-rounding at the same grid, which
        nodes them at every junction and keeps each coincident piece once
        (identical segments are merged by GEOS, on the rounded coordinates);
     c. the noded lines are split into two-point segments, and a segment that
        runs alongside a longer one within --tol-m (both ends within tol of it,
        at least half its length beside it) is dropped as a near-duplicate.
        This is needed: in several Florida DFIRMs a panel's side and its
        neighbour's differ by up to 7e-7 degrees (8 cm) at one corner, so the
        "shared" edge is two lines 0-8 cm apart over 3-7 km and is otherwise
        counted twice (52.7 km at a 5 cm tolerance);
     d. each segment's sides are read off by two probes, 2 x tol to its left
        and right of its midpoint (beyond any dropped twin): the polygons
        containing the left probe and not the right one own it on the left, and
        vice versa. That works whatever the vertex structure, and an edge
        appears once.
   tol is 10 cm: above the largest mismatch found, and below the narrowest real
   feature, a 3.1 km spur under 1 m wide on panel 12081C0333E (it is why that
   panel is invalid in EPSG:5070). At 25-50 cm the spur's two sides are merged
   and the check below fails by 6-20 km; at 10 cm it closes to 0.14 m.
   A segment whose probes see the same polygons on both sides is not an edge at
   probe scale and is dropped, counted. The output is checked: every polygon's
   perimeter must equal the length of the edges it owns, so sum(perimeters) =
   sum(length x owners), up to the dropped twins' offsets.
2. ORIENTATION CARRIES THE SIDES. Segments are oriented so the left owner is on
   the left, then merged into directed LineStrings per (left, right) pair. An
   outer edge has its owner on the left and nothing on the right; a county
   line between two in-state counties has the lower FIPS on the left; a state
   or coast edge has the in-state county on the left.
3. COUNTY LINES COME WITH THE NEIGHBOURING STATES' COUNTIES, so the outer edge
   can be split into the state line (another state's county across it) and the
   seaward limit (nothing across it). kind is 'county' for an edge between two
   in-state counties and 'state_or_coast' otherwise, with `outer` saying which.
   TIGER/Line county polygons include water: the seaward edge is the state's
   offshore limit, not the shoreline. Many Florida county lines follow rivers
   (St. Marys, Suwannee, Santa Fe, New River, Perdido), so a county line can
   coincide with a floodplain for a hydrological reason; D7 should report that
   share rather than read every coincidence as bookkeeping.
4. PANEL EDGES ARE PER DFIRM. Panels are whole printed rectangles and in Florida
   they overlap across county DFIRMs (about 65,000 km2 of pairwise overlap), so
   one county's panel edges run through the next county's map. An edge is a
   map-sheet edge only for its own map: edges are extracted within each DFIRM
   and every row carries dfirm_id, and D7 should compare a lobe only with its
   own DFIRM's edges. kind is 'interior' (two panels of the DFIRM), 'exterior'
   (the outline of the DFIRM's panel set) or 'overlap' (one panel's side inside
   another panel of the same DFIRM; pan_cover names it; a few in Florida).
   Coincident edges of different DFIRMs are kept once per DFIRM, on purpose.
5. THE TIGER FILE IS THE NATIONAL ONE, tl_<year>_us_county.zip (~83 MB, checked
   with HEAD before downloading and refused above 2 GB or below 10 GB free), cached
   under data/raw/tiger/ and read with a STATEFP filter. The per-state county
   file does not exist for TIGER/Line; the per-state "cartographic boundary"
   files are generalised and clipped to the shoreline, so they are not used.
   A cached zip is reused only if its size equals the server's Content-Length
   AND every member passes zipfile's CRC-32 test (under a second for 83 MB), and
   a fresh download must pass the same test before it replaces the cache. Size
   alone accepts a copy that was truncated and padded, or corrupted on disk;
   the CRC test catches both, because each member's checksum is stored in the
   zip itself. A same-size re-issue of the same year's file would pass; delete
   the cached zip to force a fresh copy (--overwrite redoes the outputs, not the
   download). A dry run only reports, the cache test included: it never builds
   or writes an output, whether the zip is cached or not.

USAGE
=====
    python pull_gis_lines.py --selftest                 # synthetic squares; only a temp
                                                        # dir under --gis-dir, removed
    python pull_gis_lines.py --state FL --dry-run       # URL and size only
    python pull_gis_lines.py --state FL                 # both layers
    python pull_gis_lines.py --state FL --layers panels --gis-dir /tmp/x/gis
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely import STRtree

RAW_TIGER = Path("data/raw/tiger")
NFHL_DIR = Path("data/interim/nfhl_states")
GIS_DIR = Path("data/interim/gis")

CRS_WORK = 5070
TIGER_YEAR = 2026     # TIGER2026 county file: 83,378,454 bytes, Last-Modified 2026-10-07
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
MAX_DOWNLOAD = 2e9    # audit brief: ask before anything larger
MIN_FREE = 10e9       # ... or anything that leaves less than this free
GRID_M = 0.001        # snap-rounding grid (decision 1a-b)
TOL_M = 0.10          # near-duplicate tolerance; probes sit 2 x this off the line (1c-d)
NEIGHBOUR_BUFFER_DEG = 0.001

FIPS = {"FL": "12", "TX": "48", "LA": "22", "IA": "19", "MN": "27",
        "MO": "29", "AZ": "04", "CO": "08", "VT": "50", "NC": "37"}


# =========================================================================
# edge extraction (decisions 1-2)
# =========================================================================
def _join_by_segment(seg: np.ndarray, poly: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """';'-joined labels per segment ('' when none): one label almost always."""
    out = np.full(n, "", dtype=object)
    if len(seg) == 0:
        return out
    cnt = np.bincount(seg, minlength=n)
    one = cnt[seg] == 1
    out[seg[one]] = labels[poly[one]]
    if (~one).any():
        s = (pd.Series(labels[poly[~one]]).groupby(seg[~one])
             .agg(lambda x: ";".join(sorted(x))))
        out[s.index.to_numpy()] = s.to_numpy()
    return out


def _near_duplicates(p1: np.ndarray, p2: np.ndarray, seglen: np.ndarray,
                     tol: float) -> np.ndarray:
    """Mask of segments that run alongside a longer segment within tol (decision 1c).

    Segment i goes when both its ends lie within tol of segment j and at least
    half of i projects onto j: the second condition keeps a short segment that
    merely continues j from its end, which is also within tol of it. Ties in
    length go to the lower index, so of two twins exactly one survives.
    """
    segs = shapely.linestrings(np.stack([p1, p2], axis=1))
    i, j = STRtree(segs).query(segs, predicate="dwithin", distance=tol)
    m = (i != j) & ((seglen[i] < seglen[j]) | ((seglen[i] == seglen[j]) & (i > j)))
    i, j = i[m], j[m]
    near = ((shapely.distance(shapely.points(p1[i]), segs[j]) <= tol) &
            (shapely.distance(shapely.points(p2[i]), segs[j]) <= tol))
    i, j = i[near], j[near]
    u = (p2[j] - p1[j]) / seglen[j][:, None]
    t1 = ((p1[i] - p1[j]) * u).sum(axis=1)
    t2 = ((p2[i] - p1[j]) * u).sum(axis=1)
    lo = np.clip(np.minimum(t1, t2), 0, seglen[j])
    hi = np.clip(np.maximum(t1, t2), 0, seglen[j])
    drop = np.zeros(len(p1), dtype=bool)
    drop[i[(hi - lo) >= 0.5 * seglen[i]]] = True
    return drop


def polygon_edges(geoms, ids, grid: float = GRID_M, tol: float = TOL_M,
                  primary=None) -> tuple[pd.DataFrame, dict]:
    """Boundary lines of a set of polygons with the polygon id on each side.

    Returns a frame with left, right, cover (';'-joined ids, '' for none),
    kind_raw ('shared', 'outer' or 'overlap'), geometry (directed LineStrings,
    left owner on the left) and length_m, plus a dict of checks. `primary` is a
    set of ids that goes on the left of an edge it shares with a non-primary id.
    """
    g = np.asarray(geoms, dtype=object).copy()
    ids = np.asarray(ids).astype(str).astype(object)
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.make_valid(g[bad])
    g = shapely.set_precision(g, grid)
    keep = ~shapely.is_empty(g)
    g, ids = g[keep], ids[keep]
    npoly = len(g)
    stats = dict(polygons=npoly, repaired=int(bad.sum()), grid_m=grid, tol_m=tol)

    noded = shapely.union_all(shapely.boundary(g), grid_size=grid)
    xy, part = shapely.get_coordinates(shapely.get_parts(noded), return_index=True)
    same = part[1:] == part[:-1]
    p1, p2 = xy[:-1][same], xy[1:][same]
    seglen = np.hypot(*(p2 - p1).T)
    p1, p2, seglen = p1[seglen > 0], p2[seglen > 0], seglen[seglen > 0]
    stats["segments"] = len(p1)
    twin = _near_duplicates(p1, p2, seglen, tol)
    stats["near_duplicate_segments"] = int(twin.sum())
    stats["near_duplicate_length_m"] = float(seglen[twin].sum())
    p1, p2, seglen = p1[~twin], p2[~twin], seglen[~twin]
    n = len(p1)

    eps = max(2 * tol, 5 * grid)
    mid = (p1 + p2) / 2
    nrm = np.column_stack([-(p2 - p1)[:, 1], (p2 - p1)[:, 0]]) / seglen[:, None]
    probes = shapely.points(np.concatenate([mid + eps * nrm, mid - eps * nrm]))
    # Prepared polygons against a tree of probe points: point-in-polygon on a
    # 20,000-vertex county is then logarithmic instead of linear.
    pi, qi = STRtree(probes).query(g, predicate="contains")
    on_left = qi < n
    code_l = np.unique(qi[on_left] * npoly + pi[on_left])
    code_r = np.unique((qi[~on_left] - n) * npoly + pi[~on_left])
    lo, ro = np.setdiff1d(code_l, code_r), np.setdiff1d(code_r, code_l)
    co = np.intersect1d(code_l, code_r)
    left = _join_by_segment(lo // npoly, lo % npoly, ids, n)
    right = _join_by_segment(ro // npoly, ro % npoly, ids, n)
    cover = _join_by_segment(co // npoly, co % npoly, ids, n)
    owners = (np.bincount(lo // npoly, minlength=n) + np.bincount(ro // npoly, minlength=n))

    edge = owners > 0
    stats["dropped_segments"] = int((~edge).sum())
    stats["dropped_length_m"] = float(seglen[~edge].sum())
    stats["perimeter_m"] = float(shapely.length(shapely.boundary(g)).sum())
    stats["owned_length_m"] = float((seglen * owners).sum())
    p1, p2, left, right, cover = p1[edge], p2[edge], left[edge], right[edge], cover[edge]

    # Orient: the owner goes left; between two owners the primary one, else the
    # lower id.
    swap = left == ""
    if primary is not None:
        prim = np.array([x in primary for x in left]), np.array([x in primary for x in right])
        swap |= (right != "") & prim[1] & ~prim[0]
        tie = (left != "") & (right != "") & (prim[0] == prim[1])
    else:
        tie = (left != "") & (right != "")
    swap |= tie & (left > right)
    p1[swap], p2[swap] = p2[swap].copy(), p1[swap].copy()
    left[swap], right[swap] = right[swap], left[swap].copy()
    kind = np.where(right != "", "shared", np.where(cover != "", "overlap", "outer"))

    segs = shapely.linestrings(np.stack([p1, p2], axis=1))
    key = pd.DataFrame(dict(left=left, right=right, cover=cover, kind_raw=kind))
    rows, geo = [], []
    for k, ix in key.groupby(["left", "right", "cover", "kind_raw"], sort=True).indices.items():
        merged = shapely.line_merge(shapely.multilinestrings(segs[ix]), directed=True)
        for part_ in shapely.get_parts(merged):
            rows.append(k)
            geo.append(part_)
    out = pd.DataFrame(rows, columns=["left", "right", "cover", "kind_raw"])
    out["geometry"] = np.array(geo, dtype=object)
    out["length_m"] = shapely.length(out.geometry.to_numpy())
    stats["lines"] = len(out)
    stats["perimeter_gap_m"] = stats["perimeter_m"] - stats["owned_length_m"]
    return out, stats


def _blank(s: pd.Series) -> pd.Series:
    return s.replace("", None)


def county_lines(counties, state_fips: str, grid: float = GRID_M, tol: float = TOL_M):
    """County boundary lines of one state, with FIPS and names on each side.

    `counties` is a GeoDataFrame in EPSG:5070 with GEOID, NAME and STATEFP: the
    state's counties plus the other states' counties that touch them (decision 3).
    """
    import geopandas as gpd
    geoid = counties.GEOID.astype(str).to_numpy()
    instate = set(geoid[counties.STATEFP.astype(str).to_numpy() == state_fips])
    e, stats = polygon_edges(np.asarray(counties.geometry.values), geoid, grid, tol,
                             primary=instate)
    e = e[e.left.isin(instate) | e.right.isin(instate)].reset_index(drop=True)
    name = dict(zip(geoid, counties.NAME.astype(str)))
    both = e.left.isin(instate) & e.right.isin(instate)
    out = pd.DataFrame({
        "fips_left": _blank(e.left), "fips_right": _blank(e.right),
        "name_left": e.left.map(name), "name_right": e.right.map(name),
        "kind": np.where(both, "county",
                         np.where(e.kind_raw == "overlap", "overlap", "state_or_coast")),
        "outer": np.where(both, None, np.where(e.right != "", "state",
                                               np.where(e.kind_raw == "overlap", None, "coast"))),
        "fips_cover": _blank(e.cover),
        "length_m": e.length_m,
    })
    stats["counties_in_state"] = len(instate)
    stats["counties_neighbouring"] = len(geoid) - len(instate)
    return gpd.GeoDataFrame(out, geometry=e.geometry.to_numpy(), crs=CRS_WORK), stats


def panel_edges(pan, grid: float = GRID_M, tol: float = TOL_M):
    """FIRM panel edges within each DFIRM (decision 4)."""
    import geopandas as gpd
    scale = dict(zip(pan.FIRM_PAN.astype(str), pan.SCALE)) if "SCALE" in pan else {}
    parts, total = [], {}
    for dfirm, sub in pan.groupby("DFIRM_ID", sort=True):
        e, st = polygon_edges(np.asarray(sub.geometry.values), sub.FIRM_PAN.to_numpy(), grid, tol)
        e.insert(0, "dfirm_id", dfirm)
        parts.append(e)
        for k, v in st.items():
            if isinstance(v, (int, float)) and k not in ("grid_m", "tol_m"):
                total[k] = total.get(k, 0) + v
    e = pd.concat(parts, ignore_index=True)
    kind = e.kind_raw.map({"shared": "interior", "outer": "exterior", "overlap": "overlap"})
    out = pd.DataFrame({
        "dfirm_id": e.dfirm_id, "pan_left": _blank(e.left), "pan_right": _blank(e.right),
        "kind": kind, "pan_cover": _blank(e.cover),
        "scale_left": e.left.map(scale), "scale_right": e.right.map(scale),
        "length_m": e.length_m,
    })
    total["dfirms"] = int(pan.DFIRM_ID.nunique())
    total["grid_m"], total["tol_m"] = grid, tol
    return gpd.GeoDataFrame(out, geometry=e.geometry.to_numpy(), crs=CRS_WORK), total


# =========================================================================
# inputs
# =========================================================================
def head_size(url: str) -> int:
    import requests
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("Content-Length", -1))


def zip_problem(path: Path) -> str | None:
    """None when the zip opens and every member decompresses to its stored
    CRC-32 (decision 5), else what is wrong with it. testzip reports a CRC or
    header mismatch by name, but a corrupted deflate stream raises zlib.error
    and a truncated one EOFError, so those are caught too."""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as e:
        return f"{type(e).__name__}: {e}"
    return None if bad is None else f"member {bad} fails its CRC-32 or header check"


def download(url: str, dest: Path, dry_run: bool) -> Path | None:
    """The zip at dest, fetched from url unless a verified copy is cached there
    (decision 5). None on a dry run, which reports and never builds anything."""
    size = head_size(url)
    print(f"  {url}\n  Content-Length {size:,} bytes ({size / 1e6:.1f} MB)", flush=True)
    if dest.exists() and dest.stat().st_size == size:
        bad = zip_problem(dest)
        if bad is None:
            print(f"  cached: {dest} (size and every member's CRC-32 check out)")
            if dry_run:
                print("  dry run: nothing built")
                return None
            return dest
        print(f"  cached {dest} has the right size but fails the zip test ({bad}); "
              f"{'a real run would download it again' if dry_run else 'downloading it again'}")
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
    import requests
    part = dest.with_name(dest.name + ".part")
    t0, got = time.time(), 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got % (20 << 20) < (1 << 20):
                    print(f"    {got / 1e6:,.0f} / {size / 1e6:,.0f} MB "
                          f"({time.time() - t0:.0f}s)", flush=True)
    if part.stat().st_size != size:
        sys.exit(f"  incomplete download: {part.stat().st_size:,} of {size:,} bytes; rerun")
    bad = zip_problem(part)
    if bad:
        sys.exit(f"  downloaded {part} fails the zip test ({bad}); rerun")
    part.replace(dest)
    print(f"  downloaded {dest} ({time.time() - t0:.0f}s)")
    return dest


def read_counties(zip_path: Path, state_fips: str):
    """The state's counties and the other states' counties touching them, EPSG:5070."""
    import geopandas as gpd
    import pyogrio
    src = f"/vsizip/{Path(zip_path).resolve()}"
    cols = ["STATEFP", "GEOID", "NAME"]
    st = pyogrio.read_dataframe(src, columns=cols, where=f"STATEFP = '{state_fips}'")
    if not len(st):
        sys.exit(f"{zip_path}: no counties with STATEFP {state_fips}")
    reach = shapely.buffer(shapely.union_all(np.asarray(st.geometry.values)),
                           NEIGHBOUR_BUFFER_DEG)
    other = pyogrio.read_dataframe(src, columns=cols, where=f"STATEFP <> '{state_fips}'",
                                   bbox=tuple(shapely.bounds(reach)))
    other = other[shapely.intersects(reach, np.asarray(other.geometry.values))]
    g = pd.concat([st, other], ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=st.crs).to_crs(CRS_WORK)
    print(f"  {len(st)} counties with STATEFP {state_fips}, {len(other)} neighbouring "
          f"counties in {sorted(other.STATEFP.unique())}; source CRS {st.crs.to_epsg()}",
          flush=True)
    return g


def write_geoparquet(gdf, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    t = pq.read_table(path)
    md = dict(t.schema.metadata or {})
    md[b"pull_gis_lines"] = json.dumps(meta, sort_keys=True, default=str).encode()
    pq.write_table(t.replace_schema_metadata(md), path)


def _print_checks(stats: dict) -> None:
    print(f"    {stats['segments']:,} noded segments -> {stats['lines']:,} lines; dropped "
          f"{stats['near_duplicate_segments']:,} near-duplicates "
          f"({stats['near_duplicate_length_m'] / 1e3:,.3f} km) and "
          f"{stats['dropped_segments']:,} segments ({stats['dropped_length_m']:.2f} m) with "
          f"the same polygons on both sides", flush=True)
    print(f"    perimeter check: sum of polygon perimeters {stats['perimeter_m'] / 1e3:,.3f} km, "
          f"edge length x owners {stats['owned_length_m'] / 1e3:,.3f} km, gap "
          f"{stats['perimeter_gap_m']:.3f} m", flush=True)


def _by_kind(gdf) -> str:
    s = gdf.groupby("kind").length_m.agg(["size", "sum"])
    return ", ".join(f"{k} {int(r['size']):,} lines {r['sum'] / 1e3:,.1f} km"
                     for k, r in s.iterrows())


# =========================================================================
def run_counties(a, fips: str) -> None:
    out = a.gis_dir / f"county_lines_{a.state}.parquet"
    poly_out = a.gis_dir / f"counties_{a.state}.parquet"
    print(f"\n[{a.state}] county lines (TIGER/Line {a.tiger_year})", flush=True)
    if out.exists() and poly_out.exists() and not a.overwrite:
        print(f"  {out} exists -- skipping (use --overwrite)")
        return
    url = TIGER_URL.format(year=a.tiger_year)
    zp = download(url, a.raw_dir / Path(url).name, a.dry_run)
    if zp is None:
        return
    t0 = time.time()
    cnty = read_counties(zp, fips)
    lines, stats = county_lines(cnty, fips, a.grid_m, a.tol_m)
    _print_checks(stats)
    meta = dict(source=url, tiger_year=a.tiger_year, state=a.state, statefp=fips, **stats)
    write_geoparquet(lines, out, meta)
    write_geoparquet(cnty[cnty.STATEFP == fips].reset_index(drop=True), poly_out,
                     dict(source=url, tiger_year=a.tiger_year, state=a.state))
    print(f"  {_by_kind(lines)}")
    print(f"  wrote {out} and {poly_out}  ({time.time() - t0:.0f}s)")


def run_panels(a) -> None:
    import geopandas as gpd
    out = a.gis_dir / f"panel_edges_{a.state}.parquet"
    print(f"\n[{a.state}] FIRM panel edges from {a.panels}", flush=True)
    if out.exists() and not a.overwrite:
        print(f"  {out} exists -- skipping (use --overwrite)")
        return
    t0 = time.time()
    pan = gpd.read_parquet(a.panels).to_crs(CRS_WORK)
    edges, stats = panel_edges(pan, a.grid_m, a.tol_m)
    _print_checks(stats)
    print(f"  {stats['dfirms']} DFIRMs, {len(pan):,} panels ({stats['repaired']} repaired "
          f"after reprojection); {_by_kind(edges)}")
    write_geoparquet(edges, out, dict(source=str(a.panels), state=a.state, **stats))
    print(f"  wrote {out}  ({time.time() - t0:.0f}s)")


# =========================================================================
# self-test: three squares with a T-junction, through the same functions
# =========================================================================
def selftest(tmp_root: Path) -> int:
    import geopandas as gpd
    import pyogrio
    ok = True

    def check(what: str, got, want) -> None:
        nonlocal ok
        numeric = isinstance(want, (int, float)) and not isinstance(want, bool)
        good = bool(np.isclose(got, want)) if numeric else got == want
        ok &= bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {what}: {got} (want {want})")

    # A and B are stacked; C spans both with no vertex where A meets B, so the
    # A|C and B|C edges exist only after noding.
    sq = [shapely.box(0, 0, 1000, 1000), shapely.box(0, 1000, 1000, 2000),
          shapely.box(1000, 0, 2000, 2000)]
    cn = gpd.GeoDataFrame(dict(STATEFP=["12", "12", "13"], GEOID=["12001", "12003", "13049"],
                               NAME=["A", "B", "C"]), geometry=sq, crs=CRS_WORK)
    print("county_lines on three squares (C in another state):")
    lines, st = county_lines(cn, "12")
    km = lines.groupby("kind").length_m.sum().to_dict()
    check("county length", km.get("county", 0), 1000.0)
    check("state_or_coast length", km.get("state_or_coast", 0), 6000.0)
    out = lines.groupby("outer").length_m.sum().to_dict()
    check("state-line length", out.get("state", 0), 2000.0)
    check("coast length", out.get("coast", 0), 4000.0)
    check("rows (A|B, A|C, B|C, coast A, coast B)", len(lines), 5)
    pairs = sorted(zip(lines.fips_left, lines.fips_right.fillna("-")))
    check("sides", pairs, [("12001", "-"), ("12001", "12003"), ("12001", "13049"),
                           ("12003", "-"), ("12003", "13049")])
    lookup = dict(zip(cn.GEOID, cn.geometry))
    sides_ok = True
    for r in lines.itertuples():
        p = shapely.line_interpolate_point(r.geometry, 0.3, normalized=True)
        q = shapely.line_interpolate_point(r.geometry, 0.3 + 1e-3, normalized=True)
        d = np.array([q.x - p.x, q.y - p.y]); d /= np.hypot(*d)
        lp = shapely.Point(p.x - d[1], p.y + d[0])
        sides_ok &= lookup[r.fips_left].contains(lp)
    check("left FIPS is on the left of every line", bool(sides_ok), True)
    check("perimeter gap (m)", round(st["perimeter_gap_m"], 6), 0.0)

    print("panel_edges on the same squares in one DFIRM, plus one panel of another:")
    pan = gpd.GeoDataFrame(dict(DFIRM_ID=["12001C"] * 3 + ["12003C"],
                                FIRM_PAN=["P1", "P2", "P3", "Q1"],
                                SCALE=["6000", "6000", "12000", "6000"]),
                           geometry=sq + [shapely.box(1500, 0, 2500, 1000)], crs=CRS_WORK)
    pe, pst = panel_edges(pan)
    k = pe.groupby(["dfirm_id", "kind"]).length_m.sum().to_dict()
    check("12001C interior", k.get(("12001C", "interior"), 0), 3000.0)
    check("12001C exterior", k.get(("12001C", "exterior"), 0), 8000.0)
    check("12003C exterior (overlap with 12001C ignored)", k.get(("12003C", "exterior"), 0), 4000.0)
    check("interior pairs", sorted(zip(pe[pe.kind == "interior"].pan_left,
                                       pe[pe.kind == "interior"].pan_right)),
          [("P1", "P2"), ("P1", "P3"), ("P2", "P3")])
    check("perimeter gap (m)", round(pst["perimeter_gap_m"], 6), 0.0)

    print("panel_edges on two panels whose shared side is 8 mm off at one corner:")
    wedge = gpd.GeoDataFrame(dict(DFIRM_ID=["12009C"] * 2, FIRM_PAN=["R1", "R2"]), geometry=[
        shapely.box(0, 0, 1000, 1000),
        shapely.Polygon([(1000, 0), (2000, 0), (2000, 1000), (1000.008, 1000)])], crs=CRS_WORK)
    we, wst = panel_edges(wedge)
    k = we.groupby("kind").length_m.sum().to_dict()
    check("interior length, counted once", round(k.get("interior", 0), 2), 1000.0)
    check("exterior length", round(k.get("exterior", 0), 2), 5999.99)
    check("near-duplicate segments dropped", wst["near_duplicate_segments"], 1)
    check("perimeter gap (m)", round(wst["perimeter_gap_m"], 6), 0.0)

    print("read_counties from a zipped shapefile, with a STATEFP filter:")
    ll = gpd.GeoDataFrame(cn.drop(columns="geometry"), geometry=[
        shapely.box(-82.02, 30.0, -82.01, 30.01), shapely.box(-82.02, 30.01, -82.01, 30.02),
        shapely.box(-82.01, 30.0, -82.0, 30.02)], crs=4269)
    ll = pd.concat([ll, gpd.GeoDataFrame(dict(STATEFP=["13"], GEOID=["13999"], NAME=["far"]),
                                         geometry=[shapely.box(-84, 33, -83.9, 33.1)],
                                         crs=4269)], ignore_index=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        shp = Path(td) / "tl_test_us_county.shp"
        pyogrio.write_dataframe(ll, shp)
        zp = Path(td) / "tl_test_us_county.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(td).glob("tl_test_us_county.*"):
                if f.suffix != ".zip":
                    z.write(f, f.name)
        got = read_counties(zp, "12")
        check("zip test passes the intact zip", zip_problem(zp), None)
        with zipfile.ZipFile(zp) as z:
            m = max(z.infolist(), key=lambda i: i.compress_size)
        start = m.header_offset + 30 + len(m.filename.encode()) + len(m.extra)
        raw = bytearray(zp.read_bytes())
        raw[start + m.compress_size // 2] ^= 0xFF
        broken = Path(td) / "broken.zip"
        broken.write_bytes(bytes(raw))
        check("zip test flags one flipped byte in a deflated member",
              zip_problem(broken) is not None, True)
    check("counties read (2 in state + 1 touching; the far one dropped)",
          sorted(got.GEOID), ["12001", "12003", "13049"])
    check("CRS", got.crs.to_epsg(), CRS_WORK)
    print("selftest " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    ap.add_argument("--raw-dir", type=Path, default=RAW_TIGER, help="TIGER download cache")
    ap.add_argument("--panels", type=Path, help="default data/interim/nfhl_states/pan_<ST>.parquet")
    ap.add_argument("--tiger-year", type=int, default=TIGER_YEAR)
    ap.add_argument("--layers", default="counties,panels")
    ap.add_argument("--grid-m", type=float, default=GRID_M,
                    help="snap grid for noding, metres (decision 1)")
    ap.add_argument("--tol-m", type=float, default=TOL_M,
                    help="near-duplicate tolerance, metres (decision 1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the TIGER URL and size, download nothing")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        a.gis_dir.mkdir(parents=True, exist_ok=True)
        return selftest(a.gis_dir)
    if a.state not in FIPS:
        sys.exit(f"unknown state {a.state}")
    a.panels = a.panels or NFHL_DIR / f"pan_{a.state}.parquet"
    layers = {x.strip() for x in a.layers.split(",") if x.strip()}
    if not layers or layers - {"counties", "panels"}:
        sys.exit(f"--layers must be a subset of counties,panels, got {a.layers!r}")
    if "counties" in layers:
        run_counties(a, FIPS[a.state])
    if "panels" in layers:
        if a.dry_run:
            print(f"\n[{a.state}] FIRM panel edges: dry run, nothing built")
        else:
            run_panels(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
