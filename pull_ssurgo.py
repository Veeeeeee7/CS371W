#!/usr/bin/env python3
"""
pull_ssurgo.py -- SSURGO soil map units and soil lines over the boundary-analysis bboxes.

Class S of the boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md section 3.3) asks whether a
Zone A boundary lies on a line a soil scientist drew. No model feature carries these lines, so
nothing written here is joined onto dataset_FL.parquet.

Input: lobes files <prefix>_bboxes.parquet (aotil/lobes.py; one row per region component, its
EPSG:5070 bounds and their WGS84 envelope). Output, under <gis-dir>/ssurgo/, GeoParquet in
EPSG:5070 that stores its input bboxes under the metadata key b"pull_ssurgo":
    <prefix>_mupolygons.parquet   map-unit polygons with their muaggatt attributes (POLY_COLS)
    <prefix>_mulines.parquet      map-unit boundaries as lines, tagged by kind and by which
                                  attributes differ across them (LINE_COLS)

    python pull_ssurgo.py --selftest --gis-dir $(mktemp -d)    # 2 km box, checks invariants
    python pull_ssurgo.py --bboxes results/$D/boundary/*_bboxes.parquet   # plan: size only
    python pull_ssurgo.py --bboxes ... --yes [--overwrite | --refetch]    # fetch and build
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import shapely
from pyproj import Transformer
from requests.adapters import HTTPAdapter

GIS_DIR = Path("data/interim/gis")
# NRCS Soil Data Access (SDA). A rejected query is HTTP 400 with an OGC ServiceException XML
# body; an empty result is the JSON {}.
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

CRS_GEO = 4326      # mupolygongeo (SQL Server geometry, x = longitude)
CRS_WORK = 5070     # CONUS Albers, metres, the project's GIS frame

# muaggatt map-unit aggregates: flooding frequency class (dominant condition / most frequent of
# any component), percent ponded, percent hydric, drainage class (dominant / wettest), and
# hydrologic soil group. 'None' in the flooding classes is a class (no flooding), not a null.
ATTRS = ["flodfreqdcd", "flodfreqmax", "pondfreqprs", "hydclprs",
         "drclassdcd", "drclasswettest", "hydgrpdcd"]
INT_ATTRS = ("pondfreqprs", "hydclprs")     # SDA sends varchar / int; written as Int64
BBOX_COLS = ["component", "minx", "miny", "maxx", "maxy",
             "lon_min", "lat_min", "lon_max", "lat_max"]

# Line kinds:
#   soil_line     two polygons of different mukey in one survey area, neither of them water
#   survey_join   two polygons in different survey areas, neither water: a county line or the
#                 edge of federal or tribal land (areaname_a / _b say which). A mukey belongs to
#                 one survey area, so mukey always changes across a join; the diff_ columns say
#                 whether the soil does.
#   water_edge    two polygons, at least one a water map unit: a shore, not a line between soils
#   clip_edge     one side only, outside the queried envelope: the edge of what was fetched
#   no_data_edge  one side only, inside the envelope: SSURGO has no polygon there
# Class S uses soil_line only.
KINDS = ("soil_line", "survey_join", "water_edge", "clip_edge", "no_data_edge")
TWO_SIDED_KINDS = ("soil_line", "survey_join", "water_edge")
# A water map unit's name starts with the word "Water" or "Waters", or is "Dredge channel". On
# every Florida map unit with a Water component, this picks exactly the units that are at least
# half water. It leaves out "Arents-Water complex", "Tidewater muck" and the subaqueous soils
# (soils mapped on a lagoon floor). mapunit.mukind cannot tell water from soil.
WATER_MU = re.compile(r"^(Waters?|Dredge channel)\b")
META_KEY = b"pull_ssurgo"   # parquet metadata key: the input bboxes an output was built for

POLY_COLS = (["component", "mupolygonkey", "mukey", "areasymbol", "areaname", "spatialversion",
              "saverest", "musym", "muname", "water_unit"] + ATTRS
             + ["area_m2", "in_bbox", "geometry"])
LINE_COLS = (["line_id", "component", "kind", "mupolygonkey_a", "mupolygonkey_b", "mukey_a",
              "mukey_b", "areasymbol_a", "areasymbol_b", "areaname_a", "areaname_b"]
             + [f"diff_{c}" for c in ATTRS] + ["n_diff", "diff_attrs", "length_m", "geometry"])

ROW_LIMIT = 100_000       # SDA rows per query; a response is also capped at 32 MB of JSON
BYTES_PER_PT = 36.3       # JSON bytes per vertex, measured: 150,145 bytes for 4,135 vertices
CACHE_B_PER_PT = 18.0     # disk bytes per vertex of geometry cache (17.3 on the self-test)
OUT_B_PER_PT = 37.0       # disk bytes per vertex of mupolygons + mulines (22.0 + 14.6)
RETRIES = 5
CONNECT_S = 30            # no connection in 30 s is an outage, not a big query
TIMEOUT_S = 300           # read timeout per request, seconds

DEFAULTS = dict(
    tile_deg=0.05,        # key-query tile edge, degrees (about 4.8 x 5.6 km at 30 N)
    workers=2,            # concurrent SDA requests; SDA is a shared government server
    max_pts=200_000,      # vertices per geometry request (about 7 MB of JSON)
    max_keys=1000,        # mupolygonkeys per geometry request
    snap_m=0.1,           # snap-rounding grid for noding the rings, metres (1/300 of the
                          # lobes' 30 m vertex spacing: it moves nothing the analysis sees)
    probe_m=0.5,          # side probe distance, metres; at least 2 x snap_m
    max_gb=2.0,           # --yes refuses a larger geometry download, GB
    min_free_gb=10.0,     # --yes refuses to leave less free disk, GB
)
# A 2 km box on the Bradford (FL007) / Union (FL125) survey join, in 0.01-degree tiles.
SELFTEST = dict(lon=-82.30, lat=29.95, half_m=1000.0, tile_deg=0.01)

# SDA_Get_Mupolygonkey_from_intersection_with_WktWgs84 returns the polygons a WGS84 WKT geometry
# intersects (its ..._Mukey_... sibling returns map units only, which is not enough for lines).
# `mupolygon` holds the SSURGO polygons; STATSGO is the separate gsmmupolygon.
KEY_COLS = ["mupolygonkey", "mukey", "areasymbol", "npts"]
KEY_SQL = """SELECT P.mupolygonkey, P.mukey, P.areasymbol, P.mupolygongeo.STNumPoints() AS npts
FROM SDA_Get_Mupolygonkey_from_intersection_with_WktWgs84('{wkt}') AS K
INNER JOIN mupolygon AS P ON P.mupolygonkey = K.mupolygonkey"""

GEOM_COLS = ["mupolygonkey", "mukey", "areasymbol", "spatialversion", "wkt"]
GEOM_SQL = """SELECT P.mupolygonkey, P.mukey, P.areasymbol, P.spatialversion,
P.mupolygongeo.STAsText() AS wkt
FROM mupolygon AS P WHERE P.mupolygonkey IN ({keys})"""

ATTR_COLS = ["mukey", "musym", "muname"] + ATTRS
ATTR_SQL = "SELECT " + ", ".join(ATTR_COLS) + " FROM muaggatt WHERE mukey IN ({keys})"
ATTR_BATCH = 1000         # keys per muaggatt or sacatalog request

# sacatalog names each survey area and dates its last export (saverest).
SA_COLS = ["areasymbol", "areaname", "saverest"]
SA_SQL = "SELECT " + ", ".join(SA_COLS) + " FROM sacatalog WHERE areasymbol IN ({keys})"


# =========================================================================
# Soil Data Access
# =========================================================================
class SDAError(RuntimeError):
    """SDA refused a query (bad SQL) or kept failing after the retries."""


class SDATooBig(SDAError):
    """SDA rejected a query for size, time or memory: split it and ask again."""


_TOO_BIG = re.compile(r"time ?out|timed out|memory|exceed|too large|too many|"
                      r"maximum|serializ", re.I)


def _service_message(text: str) -> str:
    """The text of an OGC ServiceException body, or else the start of the body."""
    m = re.search(r"<ServiceException>(.*?)</ServiceException>", text, re.S)
    if m:
        return html.unescape(m.group(1)).strip()
    return text.strip()[:300]


class SDA:
    """A polite POST client: retries with backoff, counts requests and bytes received."""

    def __init__(self, workers: int):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(pool_maxsize=max(2, 2 * workers)))
        self.lock = threading.Lock()        # the counters are shared by the worker threads
        self.n_req = 0
        self.n_bytes = 0

    def query(self, sql: str, cols: list[str]) -> pd.DataFrame:
        """The result of `sql`, whose columns must be `cols`. Raises SDATooBig when the query
        should be split, SDAError when it failed for another reason."""
        err: SDAError | None = None
        for attempt in range(RETRIES):
            if attempt:
                time.sleep(min(60.0, 2.0 ** attempt) + random.random())
            try:
                resp = self.session.post(SDA_URL,
                                         json={"query": sql, "format": "JSON+COLUMNNAME"},
                                         timeout=(CONNECT_S, TIMEOUT_S))
            except requests.ConnectTimeout as e:
                # the server is down: retry, but never split
                err = SDAError(f"no connection in {CONNECT_S}s: {e}")
                continue
            except requests.Timeout as e:
                # no answer in TIMEOUT_S: try once more, then split
                err = SDATooBig(f"no response in {TIMEOUT_S}s: {e}")
                if attempt >= 1:
                    break
                continue
            except requests.RequestException as e:
                err = SDAError(f"{type(e).__name__}: {e}")
                continue
            with self.lock:
                self.n_req += 1
                self.n_bytes += len(resp.content)

            if resp.status_code == 200:
                try:
                    table = resp.json().get("Table")
                except ValueError:
                    # a truncated body: retry, then split
                    err = SDATooBig(f"unparseable response, {len(resp.content):,} bytes")
                    continue
                if not table:
                    return pd.DataFrame(columns=cols)
                if table[0] != cols:        # JSON+COLUMNNAME: the first row holds the names
                    raise SDAError(f"unexpected columns {table[0]}, wanted {cols}")
                return pd.DataFrame(table[1:], columns=cols)

            msg = f"HTTP {resp.status_code}: {_service_message(resp.text)}"
            if _TOO_BIG.search(msg):
                # a timeout can be load, so ask once more; a size cap will not move
                err = SDATooBig(msg)
                if attempt >= 1:
                    break
            elif resp.status_code in (429, 500, 502, 503, 504):
                err = SDAError(msg)         # busy: retry
            else:
                raise SDAError(msg)         # bad SQL: retrying will not help
        assert err is not None
        raise err


# =========================================================================
# caching (under <gis-dir>/ssurgo/cache/, so an interrupted run resumes and
# overlapping bboxes never fetch a polygon twice)
# =========================================================================
def _write(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically, so a killed run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _digest(keys) -> str:
    """A short, stable cache-file name for a batch of keys."""
    return hashlib.sha1(",".join(map(str, keys)).encode()).hexdigest()[:16]


def _load_store(d: Path, pattern: str, key: str, cols: list[str]) -> pd.DataFrame | None:
    """All cached batches of one kind, or None. Where a key was fetched twice (--refetch) the
    newest file wins. A file without all of `cols` (an older cache layout) is ignored, so its
    keys are fetched again."""
    if not d.exists():
        return None
    files = sorted(d.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    files = [f for f in files if set(cols) <= set(pq.read_schema(f).names)]
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True).drop_duplicates(key)


def _stores(args) -> dict:
    """The cached geometry, muaggatt and sacatalog tables (all None under --refetch)."""
    if args.refetch:
        return {"geom": None, "attr": None, "sa": None}
    cache = args.gis_dir / "ssurgo" / "cache"
    geom_cols = GEOM_COLS[:-1] + ["wkb"]      # the cache holds WKB in place of SDA's WKT
    return {"geom": _load_store(cache / "geom", "g_*.parquet", "mupolygonkey", geom_cols),
            "attr": _load_store(cache / "attr", "attr_*.parquet", "mukey", ATTR_COLS),
            "sa": _load_store(cache / "attr", "sa_*.parquet", "areasymbol", SA_COLS)}


# =========================================================================
# step 1: keys by tile
# Fetching geometry tile by tile would download every polygon that crosses a
# tile edge once per tile, and one tile's WKT could near the 32 MB cap. So each
# tile is asked only for key, mukey, areasymbol and vertex count (a few kB,
# cached per tile), and step 2 fetches each unique polygon once.
# =========================================================================
def _outward(v: float, up: bool) -> float:
    """Round to 1e-6 deg (0.1 m) away from the box, so rounding never loses area."""
    if up:
        return math.ceil(v * 1e6) / 1e6
    return math.floor(v * 1e6) / 1e6


def _edges(a: float, b: float, step: float) -> list[float]:
    """a, then the lines of a global `step`-degree grid strictly between a and b, then b."""
    edges = [a]
    for k in range(math.floor(a / step) + 1, math.ceil(b / step)):
        v = round(k * step, 6)
        if a < v < b:
            edges.append(v)
    edges.append(b)
    return edges


def envelope(row) -> tuple[float, float, float, float]:
    """A bbox row's WGS84 envelope (lon_min, lat_min, lon_max, lat_max), rounded outward."""
    return (_outward(row.lon_min, False), _outward(row.lat_min, False),
            _outward(row.lon_max, True), _outward(row.lat_max, True))


def tiles_for(env: tuple, step: float) -> list[tuple]:
    """The envelope cut on a global grid of `step` degrees, row by row from the south-west.
    Edge tiles are clipped to the envelope, so a bbox never pulls polygons from beyond it."""
    xs = _edges(env[0], env[2], step)
    ys = _edges(env[1], env[3], step)
    tiles = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            tiles.append((x0, y0, x1, y1))
    return tiles


def _wkt(t: tuple) -> str:
    """A tile (x0, y0, x1, y1) as the WGS84 WKT polygon SDA expects: x = longitude, a
    counter-clockwise ring, 6 decimals (about 0.1 m)."""
    x0, y0, x1, y1 = (f"{v:.6f}" for v in t)
    return f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def _tile_name(t: tuple) -> str:
    """The cache file of a tile's keys."""
    return "_".join(f"{v:.6f}" for v in t) + ".parquet"


def query_tile(sda: SDA, t: tuple) -> pd.DataFrame:
    """The keys of every polygon meeting tile t. An answer that reaches the row cap may be cut
    short, so that tile is asked again as four quadrants (SW, SE, NW, NE)."""
    df = sda.query(KEY_SQL.format(wkt=_wkt(t)), KEY_COLS)
    if len(df) >= ROW_LIMIT:
        x0, y0, x1, y1 = t
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2
        quads = [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]
        parts = [query_tile(sda, q) for q in quads]
        df = pd.concat(parts).drop_duplicates("mupolygonkey")
    return df.astype({"mupolygonkey": "int64", "mukey": "int64", "npts": "int64"})


def tile_keys(sda: SDA, tiles: list[tuple], cache: Path, refetch: bool,
              workers: int) -> dict[tuple, pd.DataFrame]:
    """{tile: its keys}, read from the per-tile cache where present (unless refetch)."""
    out: dict[tuple, pd.DataFrame] = {}
    todo = []
    for t in tiles:
        f = cache / _tile_name(t)
        if f.exists() and not refetch:
            out[t] = pd.read_parquet(f)
        else:
            todo.append(t)
    print(f"  keys: {len(tiles)} tiles, {len(tiles) - len(todo)} cached, "
          f"{len(todo)} to query", flush=True)

    def query_and_cache(t):
        df = query_tile(sda, t)
        _write(df, cache / _tile_name(t))
        return t, df

    with ThreadPoolExecutor(workers) as ex:
        for done, (t, df) in enumerate(ex.map(query_and_cache, todo), 1):
            out[t] = df
            if done % 20 == 0 or done == len(todo):
                print(f"    {done}/{len(todo)} tiles", flush=True)
    return out


# =========================================================================
# step 2: geometry and attributes by key, each polygon fetched once and
# cached per batch
# =========================================================================
def _geom_batches(keys: pd.DataFrame, max_pts: int, max_keys: int) -> list[list[int]]:
    """Consecutive mupolygonkeys grouped into batches of at most max_pts vertices and max_keys
    keys; a polygon with more than max_pts vertices gets a batch of its own."""
    batches = []
    batch = []
    batch_pts = 0
    for key, n in zip(keys.mupolygonkey, keys.npts):
        if batch and (batch_pts + n > max_pts or len(batch) >= max_keys):
            batches.append(batch)
            batch = []
            batch_pts = 0
        batch.append(int(key))
        batch_pts += int(n)
    if batch:
        batches.append(batch)
    return batches


def _fetch_geom(sda: SDA, keys: list[int]) -> pd.DataFrame:
    """The geometry rows of `keys`, halving the batch for as long as SDA rejects it as too big."""
    try:
        return sda.query(GEOM_SQL.format(keys=",".join(map(str, keys))), GEOM_COLS)
    except SDATooBig as e:
        if len(keys) == 1:
            raise SDAError(f"mupolygonkey {keys[0]} alone is too big for SDA: {e}") from e
        half = len(keys) // 2
        print(f"    batch of {len(keys)} rejected ({e}); halving", flush=True)
        return pd.concat([_fetch_geom(sda, keys[:half]), _fetch_geom(sda, keys[half:])])


def fetch_geometry(sda: SDA, need: pd.DataFrame, store: pd.DataFrame | None,
                   cache: Path, args) -> pd.DataFrame:
    """Geometry (as WKB) of every polygon in `need`, fetching those not in `store`."""
    have = set() if store is None else set(store.mupolygonkey)
    miss = need[~need.mupolygonkey.isin(have)].sort_values("mupolygonkey")
    batches = _geom_batches(miss, args.max_pts, args.max_keys)
    print(f"  geometry: {len(need):,} polygons, {len(need) - len(miss):,} cached, "
          f"{len(miss):,} to fetch in {len(batches)} requests "
          f"(~{miss.npts.sum() * BYTES_PER_PT / 1e6:.1f} MB)", flush=True)

    def fetch_and_cache(keys):
        df = _fetch_geom(sda, keys)
        df = df.astype({"mupolygonkey": "int64", "mukey": "int64", "spatialversion": "int64"})
        wkt = df.pop("wkt").to_numpy()
        df["wkb"] = shapely.to_wkb(shapely.from_wkt(wkt))
        _write(df, cache / f"g_{_digest(keys)}.parquet")
        return df

    new = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, df in enumerate(ex.map(fetch_and_cache, batches), 1):
            new.append(df)
            print(f"    {i}/{len(batches)} geometry requests", flush=True)
    parts = new if store is None else [store] + new
    return pd.concat(parts, ignore_index=True).drop_duplicates("mupolygonkey")


def fetch_table(sda: SDA, values: list, store: pd.DataFrame | None, key: str,
                sql: str, cols: list[str], prefix: str, cache: Path,
                quote: bool) -> pd.DataFrame:
    """A small per-key table (muaggatt by mukey, sacatalog by areasymbol), fetching the keys
    not in `store`; `quote` puts each key in SQL quotes."""
    have = set() if store is None else set(store[key])
    miss = sorted(v for v in set(values) if v not in have)
    new = []
    for i in range(0, len(miss), ATTR_BATCH):
        chunk = miss[i:i + ATTR_BATCH]
        if quote:
            key_list = ",".join(f"'{v}'" for v in chunk)
        else:
            key_list = ",".join(str(v) for v in chunk)
        df = sda.query(sql.format(keys=key_list), cols)
        if key == "mukey":
            df["mukey"] = df.mukey.astype("int64")
        _write(df, cache / f"{prefix}_{_digest(chunk)}.parquet")
        new.append(df)
    if miss:
        print(f"  {prefix} table: {len(miss)} keys fetched", flush=True)
    parts = new if store is None else [store] + new
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).drop_duplicates(key)


def check_one_version(need: pd.DataFrame, stores: dict) -> None:
    """Raise SDAError unless the keys, geometry and attributes agree. SSURGO is refreshed every
    October and each cache is a snapshot of its fetch day; mixing versions would leave vanished
    polygons as holes, which become no_data_edge lines that look like open water."""
    geom = stores["geom"][stores["geom"].mupolygonkey.isin(need.mupolygonkey)]
    probs = []
    gone = sorted(set(need.mupolygonkey) - set(geom.mupolygonkey))
    if gone:
        probs.append(f"{len(gone):,} of {len(need):,} mupolygonkeys from the key step are "
                     f"not in the geometry (e.g. {gone[:5]})")
    pairs = need[["mupolygonkey", "mukey"]].merge(geom[["mupolygonkey", "mukey"]],
                                                  on="mupolygonkey", suffixes=("", "_g"))
    changed = pairs.mukey != pairs.mukey_g
    if changed.any():
        probs.append(f"{int(changed.sum()):,} polygons changed mukey "
                     f"between the key step and the geometry")
    n_versions = geom.groupby("areasymbol").spatialversion.nunique()
    mixed = n_versions.index[n_versions > 1]
    if len(mixed):
        probs.append(f"polygons of {', '.join(mixed)} carry more than one "
                     f"spatialversion")
    for table, wanted, have in (("muaggatt", need.mukey, stores["attr"].mukey),
                                ("sacatalog", need.areasymbol, stores["sa"].areasymbol)):
        miss = sorted(set(wanted) - set(have))
        if miss:
            probs.append(f"{len(miss):,} of {wanted.nunique():,} keys have no {table} row "
                         f"(e.g. {miss[:5]})")
    if probs:
        raise SDAError("the cached and fetched SSURGO data disagree: " + "; ".join(probs)
                       + ". The caches span an SSURGO refresh; re-run with --refetch so "
                       "keys, geometry and attributes come from one version.")


# =========================================================================
# polygons
# =========================================================================
def _reproject(geoms, transformer: Transformer):
    """Shapely geometries passed through a pyproj Transformer."""
    def project(xy):
        x, y = transformer.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geoms, project)


def _polygonal(g):
    """make_valid, keeping only the polygonal part of what it returns."""
    fixed = shapely.make_valid(g)
    polygonal = (3, 6)                  # shapely type ids of Polygon and MultiPolygon
    if shapely.get_type_id(fixed) in polygonal:
        return fixed
    parts = shapely.get_parts(fixed)
    return shapely.union_all(parts[np.isin(shapely.get_type_id(parts), polygonal)])


def build_polygons(comp: int, keys: pd.DataFrame, geom: pd.DataFrame,
                   attr: pd.DataFrame, sa: pd.DataFrame, bbox5070,
                   to_work: Transformer) -> tuple[gpd.GeoDataFrame, int]:
    """One component's polygons (POLY_COLS) in EPSG:5070, and how many invalid ones were
    repaired. Polygons crossing the bbox edge are kept whole, so every soil line inside the bbox
    is complete; in_bbox marks those that meet the EPSG:5070 bbox (the WGS84 envelope queried is
    a little larger)."""
    rows = geom[geom.mupolygonkey.isin(keys.mupolygonkey)].copy()
    shapes = _reproject(shapely.from_wkb(rows.pop("wkb").to_numpy()), to_work)
    invalid = ~shapely.is_valid(shapes)
    for i in np.nonzero(invalid)[0]:
        shapes[i] = _polygonal(shapes[i])
    rows = rows.merge(attr, on="mukey", how="left").merge(sa, on="areasymbol", how="left")
    for c in INT_ATTRS:
        rows[c] = pd.to_numeric(rows[c], errors="coerce").astype("Int64")
    rows["water_unit"] = rows.muname.fillna("").str.match(WATER_MU).astype(bool)
    rows.insert(0, "component", comp)
    out = gpd.GeoDataFrame(rows, geometry=shapes, crs=CRS_WORK)
    out["area_m2"] = out.geometry.area
    out["in_bbox"] = out.geometry.intersects(bbox5070)
    out = out[POLY_COLS].sort_values("mupolygonkey").reset_index(drop=True)
    return out, int(invalid.sum())


# =========================================================================
# lines
# =========================================================================
def _probe_frame(pieces: np.ndarray):
    """Midpoint and unit left-normal of each piece's longest segment."""
    xy, ii = shapely.get_coordinates(pieces, return_index=True)
    k = np.nonzero(ii[:-1] == ii[1:])[0]             # the coordinate rows that start a segment
    delta = xy[k + 1] - xy[k]
    seg_len = np.hypot(delta[:, 0], delta[:, 1])
    piece = ii[k]
    # sort by piece, longest segment first, and take the first segment of each piece
    order = np.lexsort((-seg_len, piece))
    sorted_piece = piece[order]
    first = order[np.r_[True, sorted_piece[1:] != sorted_piece[:-1]]]
    assert np.array_equal(piece[first], np.arange(len(pieces)))
    mid = xy[k[first]] + 0.5 * delta[first]
    tangent = delta[first] / seg_len[first, None]
    return mid, np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _probe(tree, geoms: np.ndarray, area: np.ndarray, rings: np.ndarray, pts: np.ndarray,
           mid: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """Index of the polygon containing each probe point (-1: none), and how many probes landed
    in more than one polygon. Where polygons overlap, the one whose ring passes within tol of
    the piece's midpoint wins, then the smallest. The tree only finds candidates by envelope:
    testing them on the prepared `geoms` is much faster than predicate="within", which scans
    the polygon's whole ring for every probe."""
    cand = tree.query(shapely.points(pts))          # rows: probe index, polygon index
    inside = shapely.contains_xy(geoms[cand[1]], pts[cand[0], 0], pts[cand[0], 1])
    cand = cand[:, inside]
    idx = np.full(len(pts), -1, dtype=np.int64)
    if not cand.shape[1]:
        return idx, 0
    hit = pd.DataFrame({"p": cand[0], "g": cand[1], "a": area[cand[1]]})
    n_multi = int((hit.p.value_counts() > 1).sum())
    if n_multi:
        hit["far"] = ~shapely.dwithin(rings[hit.g.to_numpy()],
                                      shapely.points(mid[hit.p.to_numpy()]), tol)
        hit = hit.sort_values(["p", "far", "a"])
    hit = hit.drop_duplicates("p")
    idx[hit.p.to_numpy()] = hit.g.to_numpy()
    return idx, n_multi


def _single_lines(geoms: np.ndarray, src: np.ndarray):
    """Explode to LineStrings, dropping the points and near-zero lines an intersection can
    leave; `src` says which input each part came from."""
    parts, j = shapely.get_parts(geoms, return_index=True)
    src = src[j]
    while (shapely.get_type_id(parts) >= 4).any():         # multi-parts and collections
        parts, j = shapely.get_parts(parts, return_index=True)
        src = src[j]
    keep = (shapely.get_type_id(parts) == 1) & (shapely.length(parts) > 1e-6)
    return parts[keep], src[keep]


def _assign_sides(pieces, tree, geoms, area, rings, snap_m, probe_m, tol, st) -> np.ndarray:
    """(n, 2) polygon index left and right of each piece, -1 where there is none; adds to the
    probe counters in st. A side whose polygon does not touch the piece (a polygon thinner than
    the probe), or that finds no polygon, is probed again closer."""
    mid, nrm = _probe_frame(pieces)
    side = np.empty((len(pieces), 2), dtype=np.int64)
    near = max(0.1 * probe_m, snap_m)       # the closer probe, still clear of the snap shift
    for s, sign in enumerate((1.0, -1.0)):  # left, then right
        idx, n_multi = _probe(tree, geoms, area, rings, mid + sign * probe_m * nrm, mid, tol)
        st["overlap_probes"] += n_multi
        found = idx >= 0
        far = np.ones(len(idx), bool)
        far[found] = ~shapely.dwithin(rings[idx[found]], shapely.points(mid[found]), tol)
        st["nonadjacent"] += int((far & found).sum())
        if far.any():
            idx2, n_multi = _probe(tree, geoms, area, rings,
                                   mid[far] + sign * near * nrm[far], mid[far], tol)
            st["overlap_probes"] += n_multi
            found2 = idx2 >= 0
            touches = found2.copy()
            touches[found2] = shapely.dwithin(rings[idx2[found2]],
                                              shapely.points(mid[far][found2]), tol)
            st["unresolved_sides"] += int((found2 & ~touches).sum())
            idx2[~touches] = -1
            idx[far] = idx2
        side[:, s] = idx
    return side


def _on_sides(pieces, side, rings, tol) -> np.ndarray:
    """True where every vertex and segment midpoint of a piece lies within tol of the ring of
    each polygon assigned to it. This fails only where a neighbour's boundary was not noded
    into the piece (a misfit beyond the snap), which a single probe cannot see."""
    xy, ii = shapely.get_coordinates(pieces, return_index=True)
    in_seg = ii[:-1] == ii[1:]
    pts = np.vstack([xy, 0.5 * (xy[:-1][in_seg] + xy[1:][in_seg])])
    own = np.concatenate([ii, ii[:-1][in_seg]])      # the piece of each point
    ok = np.ones(len(pieces), bool)
    for s in (0, 1):
        poly = side[own, s]
        has = poly >= 0
        near = np.ones(len(pieces), bool)
        np.logical_and.at(near, own[has], shapely.dwithin(rings[poly[has]],
                                                          shapely.points(pts[has]), tol))
        ok &= near
    return ok


def _segments(lines: np.ndarray) -> np.ndarray:
    """Every segment of `lines` as a two-point line, zero-length ones dropped."""
    xy, ii = shapely.get_coordinates(lines, return_index=True)
    in_seg = ii[:-1] == ii[1:]
    segs = shapely.linestrings(np.stack([xy[:-1][in_seg], xy[1:][in_seg]], axis=1))
    return segs[shapely.length(segs) > 0]


def _merge_pairs(pieces: np.ndarray, a: np.ndarray, b: np.ndarray):
    """The pieces of each polygon pair (a, b) merged into lines; returns lines, a, b."""
    group = pd.DataFrame({"a": a, "b": b}).groupby(["a", "b"], sort=True).ngroup().to_numpy()
    order = np.argsort(group, kind="stable")
    multi = shapely.multilinestrings(pieces[order], indices=group[order])
    pair = pd.DataFrame({"a": a[order], "b": b[order]}).drop_duplicates().to_numpy()
    merged = shapely.line_merge(multi)
    lines, li = _single_lines(merged, np.arange(len(merged)))
    return lines, pair[li, 0], pair[li, 1]


def _split_one_sided(lines: np.ndarray, la: np.ndarray, lb: np.ndarray, query_area):
    """Cut the one-sided lines (lb < 0) at the queried envelope, shrunk by 1 m: the parts
    inside are no_data_edge, those outside clip_edge. Returns geometry, side a, side b and kind
    of every line, two-sided lines first with kind '' (set by _line_table)."""
    one = lb < 0
    one_idx = np.nonzero(one)[0]
    q_in = shapely.buffer(query_area, -1.0)
    ins, si = _single_lines(shapely.intersection(lines[one], q_in), one_idx)
    out, so = _single_lines(shapely.difference(lines[one], q_in), one_idx)
    geom = np.concatenate([lines[~one], ins, out])
    ia = np.concatenate([la[~one], la[si], la[so]])
    ib = np.concatenate([lb[~one], lb[si], lb[so]])
    kind = np.array([""] * int((~one).sum()) + ["no_data_edge"] * len(ins)
                    + ["clip_edge"] * len(out), dtype=object)
    return geom, ia, ib, kind


def _differs(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Element-wise "the values differ", null counting as a value of its own."""
    na_a = pd.isna(va)
    na_b = pd.isna(vb)
    return (na_a != na_b) | (~na_a & ~na_b & (va.astype(str) != vb.astype(str)))


def _line_table(polys: pd.DataFrame, ia: np.ndarray, ib: np.ndarray, kind: np.ndarray,
                length_m: np.ndarray) -> pd.DataFrame:
    """The lines' attribute table: the polygon on side a and b (b is null on one-sided lines),
    kind (the two-sided kinds are decided here), diff_<attr> for each attribute, n_diff,
    diff_attrs and length_m. The diff columns are null on one-sided lines."""
    df = pd.DataFrame({"component": polys.component.iloc[0]}, index=np.arange(len(ia)))
    two = ib >= 0
    for c in ("mupolygonkey", "mukey", "areasymbol", "areaname"):
        is_text = c in ("areasymbol", "areaname")
        values = polys[c].to_numpy()
        df[f"{c}_a"] = values[ia]
        if is_text:
            df[f"{c}_a"] = df[f"{c}_a"].astype("string")
        side_b = pd.Series(pd.NA, index=df.index, dtype="string" if is_text else "Int64")
        side_b[two] = values[ib[two]]
        df[f"{c}_b"] = side_b

    wet = polys.water_unit.to_numpy(bool)
    water = two & (wet[ia] | wet[np.maximum(ib, 0)])
    join = df.areasymbol_a.to_numpy() != df.areasymbol_b.fillna("").to_numpy()
    kind = kind.copy()
    kind[two & ~water & ~join] = "soil_line"
    kind[two & ~water & join] = "survey_join"
    kind[two & water] = "water_edge"
    df["kind"] = kind

    for c in ATTRS:
        va = polys[c].to_numpy(dtype=object)[ia]
        vb = np.where(two, polys[c].to_numpy(dtype=object)[np.maximum(ib, 0)], None)
        df[f"diff_{c}"] = pd.array(np.where(two, _differs(va, vb), False), dtype="boolean")
        df.loc[~two, f"diff_{c}"] = pd.NA
    dcols = [f"diff_{c}" for c in ATTRS]
    df["n_diff"] = df[dcols].sum(axis=1).astype("Int64")
    df.loc[~two, "n_diff"] = pd.NA
    names = np.array(ATTRS, dtype=object)
    flags = df[dcols].fillna(False).to_numpy(bool)
    df["diff_attrs"] = pd.array([";".join(names[f]) for f in flags], dtype="string")
    df.loc[~two, "diff_attrs"] = pd.NA
    df["length_m"] = length_m
    return df


def build_lines(polys: gpd.GeoDataFrame, query_area, snap_m: float,
                probe_m: float) -> tuple[gpd.GeoDataFrame | None, dict]:
    """The map-unit boundaries of one component's polygons as lines, and diagnostic counts.

    SSURGO is topological (neighbours share identical vertices, also across survey joins), so
    the lines come from noding: all rings are unioned with snap-rounding at snap_m (absorbing
    float noise and sub-decimetre misfits), which dissolves each shared edge into one piece.
    Each piece gets its two polygons by probing probe_m left and right of its longest segment;
    a piece that does not run along those polygons is probed segment by segment. Pieces of the
    same polygon pair are merged into lines, and lines between two polygons of the same mukey
    are dropped. A misfit larger than snap_m is not repaired but shows in the counts: sideless
    pieces, probes in overlapping polygons, a perimeter balance (2 x two-sided + one-sided
    length) / perimeter off 1, or polygons owning over 1 m more line than their perimeter. A
    larger snap_m is no reliable repair: snap-rounding can leave two lines 1 m apart in
    neighbouring grid cells.
    """
    st = dict(overlap_probes=0, nonadjacent=0, unresolved_sides=0)
    geoms = polys.geometry.to_numpy()
    rings = shapely.boundary(geoms)
    # Unprepared, a point test scans the whole ring, and a V-vertex polygon gives ~V pieces to
    # test, so a 221,396-vertex water polygon would cost V^2. Prepared, each is an index lookup.
    shapely.prepare(geoms)
    shapely.prepare(rings)
    pieces = shapely.get_parts(shapely.union_all(rings, grid_size=snap_m))
    pieces = pieces[shapely.length(pieces) > 0]
    st["pieces"] = len(pieces)

    tree = shapely.STRtree(geoms)
    area = shapely.area(geoms)
    tol = 2 * snap_m + 1e-3
    side = _assign_sides(pieces, tree, geoms, area, rings, snap_m, probe_m, tol, st)
    misfit = ~_on_sides(pieces, side, rings, tol)
    st["split_pieces"] = int(misfit.sum())
    if misfit.any():
        segs = _segments(pieces[misfit])
        seg_side = _assign_sides(segs, tree, geoms, area, rings, snap_m, probe_m, tol, st)
        pieces = np.concatenate([pieces[~misfit], segs])
        side = np.vstack([side[~misfit], seg_side])

    # the polygon pair of each piece, a <= b; a one-sided piece has its polygon in a, b = -1
    lo = side.min(1)
    hi = side.max(1)
    a = np.where(lo >= 0, lo, hi)
    b = np.where(lo >= 0, hi, -1)
    plen = shapely.length(pieces)
    lost = a < 0                            # no polygon on either side
    inner = (a == b) & (a >= 0)             # a cut line inside one polygon
    st["lost_pieces"] = int(lost.sum())
    st["lost_m"] = float(plen[lost].sum())
    st["inner_pieces"] = int(inner.sum())
    st["inner_m"] = float(plen[inner].sum())
    keep = ~lost & ~inner
    pieces, a, b = pieces[keep], a[keep], b[keep]
    if not len(pieces):
        return None, st

    lines, la, lb = _merge_pairs(pieces, a, b)
    geom, ia, ib, kind = _split_one_sided(lines, la, lb, query_area)
    table = polys.reset_index(drop=True)      # ia / ib index its rows
    glen = shapely.length(geom)
    df = _line_table(table, ia, ib, kind, glen)

    two = ib >= 0
    same = two & (df.mukey_a.to_numpy() == df.mukey_b.fillna(-1).to_numpy())
    st["same_mukey_lines"] = int(same.sum())
    st["same_mukey_m"] = float(glen[same].sum())
    keep = ~same
    out = gpd.GeoDataFrame(df[keep].reset_index(drop=True), geometry=geom[keep], crs=CRS_WORK)

    st["perimeter_m"] = float(shapely.length(rings).sum())
    two_len = float(glen[two].sum()) + st["inner_m"]
    st["balance"] = (2 * two_len + float(glen[~two].sum())) / st["perimeter_m"]
    # A piece written twice makes its polygons own more line than their perimeter. Snap-rounding
    # changes a ring's length by well under 1 m, so more than that counts.
    owned = np.bincount(ia, glen, len(table)) + np.bincount(ib[two], glen[two], len(table))
    excess = owned - shapely.length(rings)
    over = excess > 1.0
    st["excess_polygons"] = int(over.sum())
    st["excess_m"] = float(excess[over].sum())
    return out, st


# =========================================================================
# per prefix
# =========================================================================
@dataclass
class Prefix:
    """One <prefix>_bboxes.parquet and what step 1 found for it."""
    path: Path
    prefix: str
    bb: pd.DataFrame
    f_poly: Path
    f_line: Path
    envs: dict = field(default_factory=dict)        # component -> WGS84 envelope
    tiles: dict = field(default_factory=dict)       # component -> its tiles
    comp_keys: dict = field(default_factory=dict)   # component -> its polygon keys

    @property
    def need(self) -> pd.DataFrame:
        """The unique polygon keys over all components."""
        return pd.concat(self.comp_keys.values()).drop_duplicates("mupolygonkey")


def load_prefix(path: Path, gis_dir: Path) -> Prefix:
    """Read and check one bboxes file; exits on a malformed one."""
    name = path.name
    if not name.endswith("_bboxes.parquet"):
        sys.exit(f"{path}: expected <prefix>_bboxes.parquet")
    prefix = name[: -len("_bboxes.parquet")]
    bb = pd.read_parquet(path)
    missing = [c for c in BBOX_COLS if c not in bb.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    if bb.empty or not bb.component.is_unique:
        sys.exit(f"{path}: needs one row per component")
    bb = bb.sort_values("component").reset_index(drop=True)
    out = gis_dir / "ssurgo"
    return Prefix(path, prefix, bb, out / f"{prefix}_mupolygons.parquet",
                  out / f"{prefix}_mulines.parquet")


def bbox_record(bb: pd.DataFrame) -> list[dict]:
    """The input bboxes as the outputs store them. JSON writes a float's repr, so the record
    reads back bit-exact."""
    records = []
    for row in bb[BBOX_COLS].itertuples(index=False):
        rec = {}
        for c, v in zip(BBOX_COLS, row):
            rec[c] = int(v) if c == "component" else float(v)
        records.append(rec)
    return records


def stored_bboxes(path: Path) -> list | None:
    """The bbox record of an existing output, [] when it has none; None: no file."""
    if not path.exists():
        return None
    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(META_KEY, b"{}")).get("bboxes", [])


def output_state(p: Prefix) -> tuple[str, str]:
    """Whether p's outputs were built for p's bboxes, and if not, why. The output names are
    reused between runs, so only stored bboxes say what an output holds. Returns 'absent',
    'same' (both outputs store these bboxes), 'partial' (one does and the other is missing: a
    run cut short, rebuilt without asking) or 'other' with the reason."""
    want = bbox_record(p.bb)
    stored = {f: stored_bboxes(f) for f in (p.f_poly, p.f_line)}
    records = list(stored.values())
    if all(r is None for r in records):
        return "absent", ""
    if all(r == want for r in records):
        return "same", ""
    if all(r is None or r == want for r in records):
        return "partial", ""
    return "other", _mismatch_reason(p, want, stored)


def _mismatch_reason(p: Prefix, want: list[dict], stored: dict) -> str:
    """One clause per distinct stored record that differs from `want`."""
    now = {r["component"]: r for r in want}
    reasons = []
    seen = []
    for record in stored.values():
        if record is None or record == want or record in seen:
            continue
        seen.append(record)
        names = " and ".join(f.name for f, r in stored.items() if r == record)
        if not record:
            reasons.append(f"{names}: no input record (written by an older pull_ssurgo)")
            continue
        old = {r["component"]: r for r in record}
        diff = sorted(c for c in now.keys() | old.keys() if now.get(c) != old.get(c))
        reasons.append(f"{names}: built for components {sorted(old)}, {p.path.name} has "
                       f"{sorted(now)}; the bboxes differ for component(s) {diff}")
    return "; ".join(reasons)


def plan_keys(p: Prefix, args, sda: SDA) -> None:
    """Step 1 for one prefix: the keys of each component, and a table of them."""
    print(f"\n== {p.prefix}: {len(p.bb)} components", flush=True)
    p.envs = {int(row.component): envelope(row) for row in p.bb.itertuples()}
    p.tiles = {c: tiles_for(env, args.tile_deg) for c, env in p.envs.items()}
    unique_tiles = set()
    for tiles in p.tiles.values():
        unique_tiles.update(tiles)
    cache = args.gis_dir / "ssurgo" / "cache" / "keys"
    keys_by_tile = tile_keys(sda, sorted(unique_tiles), cache, args.refetch, args.workers)
    p.comp_keys = {}
    for c, tiles in p.tiles.items():
        frames = [keys_by_tile[t] for t in tiles]
        p.comp_keys[c] = pd.concat(frames).drop_duplicates("mupolygonkey")

    print(f"  {'comp':>4} {'tiles':>5} {'polygons':>8} {'mukeys':>6} {'vertices':>10} "
          f"{'~MB':>7}  areasymbols")
    for c, keys in p.comp_keys.items():
        n_pts = keys.npts.sum()
        print(f"  {c:>4} {len(p.tiles[c]):>5} {len(keys):>8,} {keys.mukey.nunique():>6} "
              f"{n_pts:>10,} {n_pts * BYTES_PER_PT / 1e6:>7.1f}  "
              f"{','.join(sorted(keys.areasymbol.unique()))}", flush=True)
    need = p.need
    print(f"  {len(need):,} unique polygons, {need.npts.sum():,} vertices, "
          f"~{need.npts.sum() * BYTES_PER_PT / 1e6:.1f} MB of geometry", flush=True)


def gb(n: float) -> str:
    """A byte count as MB below 100 MB, else as GB."""
    if n < 1e8:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.3f} GB"


def free_bytes(path: Path) -> int:
    """Free disk space where `path` is, or will be, created."""
    p = path.resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def size_gate(jobs: list[Prefix], need: pd.DataFrame, args, stores: dict) -> int | None:
    """Print the network and disk estimate for the geometry not yet cached. Returns None to go
    on, else the exit code: 0 after a plan (no --yes, or --plan), 2 when --yes is refused for
    --max-gb or --min-free-gb. Outputs repeat a polygon for every component that holds it."""
    have = set() if stores["geom"] is None else set(stores["geom"].mupolygonkey)
    new = need[~need.mupolygonkey.isin(have)]
    new_pts = int(new.npts.sum())
    out_pts = 0
    for p in jobs:
        for keys in p.comp_keys.values():
            out_pts += int(keys.npts.sum())
    net = new_pts * BYTES_PER_PT
    disk = new_pts * CACHE_B_PER_PT + out_pts * OUT_B_PER_PT
    free = free_bytes(args.gis_dir)
    print(f"\nsize: {len(need):,} unique polygons in {len(jobs)} prefixes, "
          f"{len(need) - len(new):,} cached; to fetch {len(new):,} polygons, "
          f"{new_pts:,} vertices", flush=True)
    print(f"  network ~{gb(net)} of geometry JSON (plus attribute tables of a few kB); "
          f"disk ~{gb(disk)} (cache {gb(new_pts * CACHE_B_PER_PT)}, outputs "
          f"{gb(out_pts * OUT_B_PER_PT)}); free now {gb(free)}", flush=True)
    over = net > args.max_gb * 1e9
    low = free - disk < args.min_free_gb * 1e9
    if args.plan or not args.yes:
        if over:
            print(f"  NOTE: exceeds --max-gb {args.max_gb}; --yes would refuse", flush=True)
        if low:
            print(f"  NOTE: would leave < {args.min_free_gb} GB free; --yes would refuse",
                  flush=True)
        rerun_with = []
        if not args.yes:
            rerun_with.append("with --yes")
        if args.plan:
            rerun_with.append("without --plan")
        print(f"  plan only: no geometry fetched. Re-run {' and '.join(rerun_with)} to fetch "
              f"and build.", flush=True)
        return 0
    if over:
        print(f"REFUSED: ~{gb(net)} exceeds --max-gb {args.max_gb}. Ask before raising "
              f"--max-gb.", file=sys.stderr)
        return 2
    if low:
        print(f"REFUSED: would leave ~{gb(free - disk)} free (< --min-free-gb "
              f"{args.min_free_gb}).", file=sys.stderr)
        return 2
    return None


def _query_area_work(env: tuple, to_work: Transformer):
    """The queried WGS84 envelope in EPSG:5070, densified so its edges stay true."""
    return _reproject(shapely.segmentize(shapely.box(*env), 0.001), to_work)


def write_outputs(parts: list[tuple[gpd.GeoDataFrame, Path]], meta: dict) -> None:
    """GeoParquet with `meta` under META_KEY; each file appears whole or not at all."""
    for gdf, f in parts:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f.suffix + f".tmp{os.getpid()}")
        gdf.to_parquet(tmp)
        table = pq.read_table(tmp)
        metadata = {**(table.schema.metadata or {}), META_KEY: json.dumps(meta).encode()}
        pq.write_table(table.replace_schema_metadata(metadata), tmp)
        os.replace(tmp, f)


def _print_line_checks(lines: gpd.GeoDataFrame, st: dict, box5070) -> None:
    """One component's topology diagnostics, with a WARNING where the polygons are unclean."""
    print(f"    probes: {st['nonadjacent']} hit a non-adjacent polygon (re-probed closer), "
          f"{st['unresolved_sides']} still unresolved, {st['overlap_probes']} in "
          f"overlapping polygons, {st['split_pieces']} pieces split to segments; dropped: "
          f"{st['same_mukey_lines']} same-mukey lines ({st['same_mukey_m']:.0f} m), "
          f"{st['inner_pieces']} inner pieces ({st['inner_m']:.0f} m), "
          f"{st['lost_pieces']} sideless pieces ({st['lost_m']:.0f} m)")
    no_data = lines[lines.kind == "no_data_edge"]
    no_data_in_m = float(no_data.intersection(box5070).length.sum())
    print(f"    perimeter balance (2*two-sided + one-sided) / perimeter = "
          f"{st['balance']:.6f}; {st['excess_polygons']} polygons own over 1 m more line "
          f"than their perimeter ({st['excess_m']:.0f} m in all); no_data_edge inside "
          f"the 5070 bbox: {no_data_in_m:.0f} m")
    if (abs(st["balance"] - 1) > 1e-3 or st["overlap_probes"] or st["split_pieces"]
            or st["unresolved_sides"] or st["lost_pieces"] or st["excess_polygons"]):
        print("    WARNING: the polygons are not topologically clean here (a misfit "
              "beyond --snap-m or an overlap). Soil lines at the misfit are "
              "unpaired or doubled; inspect before using this component.")
    if no_data_in_m > 0:
        print("    NOTE: one-sided edges inside the bbox: SSURGO has no polygon on their "
              "far side (open water beyond a survey, or a gap). Expected only on a coast.")


def build_prefix(p: Prefix, args, stores: dict) -> None:
    """Build one prefix's polygons and lines from the fetched tables, and write them."""
    print(f"\n== {p.prefix}: building", flush=True)
    to_work = Transformer.from_crs(CRS_GEO, CRS_WORK, always_xy=True)
    polys_all = []
    lines_all = []
    for row in p.bb.itertuples():
        c = int(row.component)
        if p.comp_keys[c].empty:
            print(f"  component {c}: no SSURGO polygons in the envelope")
            continue
        t0 = time.time()
        box5070 = shapely.box(row.minx, row.miny, row.maxx, row.maxy)
        polys, n_fixed = build_polygons(c, p.comp_keys[c], stores["geom"], stores["attr"],
                                        stores["sa"], box5070, to_work)
        query_area = _query_area_work(p.envs[c], to_work)
        lines, st = build_lines(polys, query_area, args.snap_m, args.probe_m)
        polys_all.append(polys)
        n_lines = 0 if lines is None else len(lines)
        print(f"  component {c}: {len(polys):,} polygons ({int(polys.in_bbox.sum()):,} in the "
              f"5070 bbox, {int(polys.water_unit.sum()):,} water), {polys.mukey.nunique()} "
              f"mukeys, {polys.areasymbol.nunique()} survey areas, {n_fixed} invalid fixed; "
              f"{st['pieces']:,} noded pieces -> {n_lines:,} lines "
              f"[{time.time() - t0:.1f}s]", flush=True)
        if lines is not None:
            lines_all.append(lines)
            _print_line_checks(lines, st, box5070)

    if not lines_all:
        print("  nothing to write")
        return
    polys = pd.concat(polys_all, ignore_index=True)
    lines = pd.concat(lines_all, ignore_index=True)
    lines = lines.sort_values(["component", "kind", "mupolygonkey_a", "mupolygonkey_b",
                               "length_m"]).reset_index(drop=True)
    lines.insert(0, "line_id", np.arange(len(lines), dtype=np.int64))
    lines = gpd.GeoDataFrame(lines[LINE_COLS], geometry="geometry", crs=CRS_WORK)
    meta = dict(bboxes=bbox_record(p.bb), bboxes_file=str(p.path), tile_deg=args.tile_deg,
                snap_m=args.snap_m, probe_m=args.probe_m,
                created=dt.datetime.now().isoformat(timespec="seconds"))
    write_outputs([(polys, p.f_poly), (lines, p.f_line)], meta)
    summarize(polys, lines)
    print(f"  wrote {p.f_poly} ({len(polys):,} rows)\n  wrote {p.f_line} ({len(lines):,} rows)")


def summarize(polys: pd.DataFrame, lines: pd.DataFrame) -> None:
    """Print the lines by kind, the attributes that differ across them, and the survey joins."""
    print("  lines by kind (count, km):")
    for k in KINDS:
        is_kind = lines.kind == k
        print(f"    {k:<13} {int(is_kind.sum()):>6,} {lines.length_m[is_kind].sum() / 1e3:>9.2f}")
    two = lines.kind.isin(TWO_SIDED_KINDS)
    print("  two-sided lines on which each attribute differs "
          "(soil_line / survey_join / water_edge):")
    for c in ATTRS:
        differs = lines[f"diff_{c}"].fillna(False).astype(bool)
        counts = []
        for k in TWO_SIDED_KINDS:
            counts.append(f"{int((differs & (lines.kind == k)).sum()):>6,}")
        print(f"    {c:<15} " + " / ".join(counts))
    print(f"    no attribute differs: {int((two & (lines.n_diff == 0)).sum()):,} of "
          f"{int(two.sum()):,}")
    joins = lines[lines.kind == "survey_join"]
    if len(joins):
        print("  survey joins (km):")
        pair_names = joins.areaname_a + "  |  " + joins.areaname_b
        km = joins.length_m.groupby(pair_names).sum() / 1e3
        for names, length_km in km.sort_values(ascending=False).items():
            print(f"    {length_km:>9.2f}  {names}")


def run(args, sda: SDA) -> int:
    """Step 1 for every --bboxes file, the size gate, step 2, then the outputs; exit code."""
    jobs = []
    refused = []
    for path in args.bboxes:
        p = load_prefix(path, args.gis_dir)
        state, why = output_state(p)
        if args.overwrite or args.refetch:
            jobs.append(p)
        elif state == "same":
            print(f"{p.prefix}: outputs exist for exactly these bboxes, skipping "
                  f"(--overwrite to rebuild): {p.f_poly}, {p.f_line}")
        else:
            if state == "other":
                refused.append(f"{p.prefix}: {why}")
            jobs.append(p)
    if refused:
        msg = ("existing outputs were built for other bboxes:\n  "
               + "\n  ".join(refused) + "\n  --overwrite replaces them (the cache is kept).")
        if args.yes and not args.plan:
            print(f"REFUSED: {msg}", file=sys.stderr)
            return 2
        print(f"NOTE: --yes would refuse: {msg}")
    if not jobs:
        return 0

    stores = _stores(args)
    for p in jobs:
        plan_keys(p, args, sda)
    print(f"\nkey step: {sda.n_req} SDA requests, {sda.n_bytes / 1e6:.2f} MB", flush=True)
    need = pd.concat([p.need for p in jobs]).drop_duplicates("mupolygonkey")
    if need.empty:
        print("no SSURGO polygons in any envelope; nothing to do")
        return 0
    rc = size_gate(jobs, need, args, stores)
    if rc is not None:
        return rc

    cache = args.gis_dir / "ssurgo" / "cache"
    stores["geom"] = fetch_geometry(sda, need, stores["geom"], cache / "geom", args)
    stores["attr"] = fetch_table(sda, need.mukey.unique().tolist(), stores["attr"], "mukey",
                                 ATTR_SQL, ATTR_COLS, "attr", cache / "attr", quote=False)
    stores["sa"] = fetch_table(sda, need.areasymbol.unique().tolist(), stores["sa"],
                               "areasymbol", SA_SQL, SA_COLS, "sa", cache / "attr", quote=True)
    check_one_version(need, stores)
    for p in jobs:
        build_prefix(p, args, stores)
    return 0


# =========================================================================
# self-test
# =========================================================================
def _two_sided(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return lines[lines.kind.isin(TWO_SIDED_KINDS)]


def _sides_touch(polys: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> bool:
    """Every two-sided line's midpoint lies within 0.5 m of both its polygons' boundaries."""
    two = _two_sided(lines)
    geom_by_key = polys.set_index("mupolygonkey").geometry
    mid = shapely.line_interpolate_point(two.geometry.to_numpy(), 0.5, normalized=True)
    ring_a = shapely.boundary(geom_by_key.loc[two.mupolygonkey_a].to_numpy())
    ring_b = shapely.boundary(geom_by_key.loc[two.mupolygonkey_b.astype("int64")].to_numpy())
    da = shapely.distance(mid, ring_a)
    db = shapely.distance(mid, ring_b)
    return bool((np.maximum(da, db) < 0.5).all())


def _water_kinds(polys: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> bool:
    """A two-sided line is water_edge exactly when one of its sides is a water unit."""
    two = _two_sided(lines)
    wet = polys.set_index("mupolygonkey").water_unit
    wet_side = (wet.loc[two.mupolygonkey_a].to_numpy()
                | wet.loc[two.mupolygonkey_b.astype("int64")].to_numpy())
    return bool(((two.kind == "water_edge").to_numpy() == wet_side).all())


def selftest(args, sda: SDA) -> int:
    """Run the whole pipeline on the SELFTEST box (lobes bbox schema; small tiles, so the
    dedupe across tiles is exercised) and check the outputs. 0 when every check passes."""
    to_work = Transformer.from_crs(CRS_GEO, CRS_WORK, always_xy=True)
    to_geo = Transformer.from_crs(CRS_WORK, CRS_GEO, always_xy=True)
    cx, cy = to_work.transform(SELFTEST["lon"], SELFTEST["lat"])
    half = SELFTEST["half_m"]
    ring = shapely.segmentize(shapely.box(cx - half, cy - half, cx + half, cy + half), 50.0)
    lon, lat = to_geo.transform(*shapely.get_coordinates(ring).T)
    bb = pd.DataFrame([dict(component=0, minx=cx - half, miny=cy - half,
                            maxx=cx + half, maxy=cy + half,
                            lon_min=lon.min(), lat_min=lat.min(),
                            lon_max=lon.max(), lat_max=lat.max())])
    path = args.gis_dir / "ssurgo" / "selftest" / "SELFTEST_bboxes.parquet"
    _write(bb, path)
    args.bboxes = [path]
    args.tile_deg = SELFTEST["tile_deg"]
    args.overwrite = True
    args.yes = True
    args.plan = False
    args.min_free_gb = 0.0               # the self-test writes well under 1 MB
    rc = run(args, sda)
    if rc:
        return rc

    f_poly = args.gis_dir / "ssurgo" / "SELFTEST_mupolygons.parquet"
    f_line = args.gis_dir / "ssurgo" / "SELFTEST_mulines.parquet"
    polys = gpd.read_parquet(f_poly)
    lines = gpd.read_parquet(f_line)
    union = shapely.union_all(polys.geometry.to_numpy())
    box5070_in = shapely.box(bb.minx[0], bb.miny[0], bb.maxx[0], bb.maxy[0]).buffer(-1.0)
    checks = [
        ("polygons found", len(polys) > 0),
        ("mupolygonkey unique", polys.mupolygonkey.is_unique),
        ("no overlaps: union area == sum of areas (1e-6)",
         abs(shapely.area(union) / polys.area_m2.sum() - 1) < 1e-6),
        ("every polygon has muaggatt and a survey-area name",
         polys.muname.notna().all() and polys.areaname.notna().all()),
        ("survey join present (FL007 and FL125)",
         {"FL007", "FL125"} <= set(polys.areasymbol)),
        ("soil_line sides differ in mukey",
         (lines.mukey_a != lines.mukey_b)[lines.kind == "soil_line"].all()),
        ("survey_join sides differ in areasymbol",
         (lines.areasymbol_a != lines.areasymbol_b)[lines.kind == "survey_join"].all()),
        ("water_edge exactly where a side is a water unit", _water_kinds(polys, lines)),
        ("one-sided kinds have null side b",
         lines.mukey_b[lines.kind.isin(["clip_edge", "no_data_edge"])].isna().all()),
        ("no clip_edge inside the 5070 bbox",
         not lines[lines.kind == "clip_edge"].intersects(box5070_in).any()),
        ("two-sided lines lie on both polygons' boundaries", _sides_touch(polys, lines)),
        ("both outputs store the input bboxes",
         stored_bboxes(f_poly) == stored_bboxes(f_line) == bbox_record(bb)),
    ]
    print("\nself-test:")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 0 if all(ok for _, ok in checks) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bboxes", nargs="+", type=Path, default=[],
                    help="one or more <prefix>_bboxes.parquet from aotil/lobes.py")
    ap.add_argument("--gis-dir", type=Path, default=GIS_DIR,
                    help=f"outputs go to <gis-dir>/ssurgo/ (default {GIS_DIR})")
    ap.add_argument("--tile-deg", type=float, default=DEFAULTS["tile_deg"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--max-pts", type=int, default=DEFAULTS["max_pts"])
    ap.add_argument("--max-keys", type=int, default=DEFAULTS["max_keys"])
    ap.add_argument("--snap-m", type=float, default=DEFAULTS["snap_m"])
    ap.add_argument("--probe-m", type=float, default=DEFAULTS["probe_m"])
    ap.add_argument("--yes", action="store_true",
                    help="fetch and build; without it, keys and the size estimate only")
    ap.add_argument("--max-gb", type=float, default=DEFAULTS["max_gb"],
                    help="--yes refuses a geometry download estimated above this many GB")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULTS["min_free_gb"],
                    help="--yes refuses when the run would leave less free disk (GB)")
    ap.add_argument("--plan", action="store_true",
                    help="keys and the size estimate only, even with --yes")
    ap.add_argument("--overwrite", action="store_true", help="rebuild outputs (cache kept)")
    ap.add_argument("--refetch", action="store_true", help="ignore the cache, re-query SDA")
    ap.add_argument("--selftest", action="store_true",
                    help="2 km box at Bradford/Union (0.01 deg tiles), checks invariants")
    args = ap.parse_args(argv)
    if not (args.selftest or args.bboxes):
        ap.error("give --bboxes or --selftest")
    if not 0 < args.workers <= 4:
        ap.error("--workers must be 1-4 (SDA is a shared government server)")
    if args.probe_m < 2 * args.snap_m:
        ap.error("--probe-m must be at least 2 x --snap-m, or a probe can land on the "
                 "wrong side of a snapped line")

    t0 = time.time()
    sda = SDA(args.workers)
    try:
        if args.selftest:
            rc = selftest(args, sda)
        else:
            rc = run(args, sda)
    except SDAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        rc = 1
    print(f"\nSDA: {sda.n_req} requests, {sda.n_bytes / 1e6:.2f} MB received; "
          f"{time.time() - t0:.1f}s total", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
