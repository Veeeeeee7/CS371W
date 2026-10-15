#!/usr/bin/env python3
"""
pull_ssurgo.py -- SSURGO soil map units and soil lines over the boundary-analysis bboxes.

Class S of the boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md section 3.3,
docs/PROJECT-PLAN.md O5) asks whether a Zone A boundary inside a region lies on a
line a soil scientist drew. SSURGO map-unit lines are that layer. No covariate
carries them (the StreamCat soil columns are catchment averages), which is what
makes them one of the "held-out maps" the paper tests against. Nothing written
here is a model feature, and nothing here is joined onto dataset_FL.parquet.

For each lobes file `<prefix>_bboxes.parquet` (aotil/lobes.py: one row per region
component, EPSG:5070 bounds already buffered by 3 km, plus their WGS84 envelope)
this writes, under --gis-dir:

    ssurgo/<prefix>_mupolygons.parquet   map-unit polygons + muaggatt attributes
    ssurgo/<prefix>_mulines.parquet      map-unit boundaries as lines, each tagged
                                         with which attributes change across it

both GeoParquet in EPSG:5070, both with the `component` column of the bbox and
the input bboxes under the parquet metadata key b"pull_ssurgo" (note 9).

    mupolygons  component, mupolygonkey, mukey, areasymbol, areaname,
                spatialversion, saverest, musym, muname, water_unit, the seven
                attributes (note 3), area_m2, in_bbox, geometry
    mulines     line_id, component, kind (note 5), mupolygonkey_a/_b, mukey_a/_b,
                areasymbol_a/_b, areaname_a/_b, diff_<attr> for each attribute,
                n_diff, diff_attrs (note 6), length_m, geometry

areaname and water_unit are metadata for telling line kinds apart, not soil
attributes; nothing here is a model feature.

THE SOURCE, AS VERIFIED 2026-10-14
==================================
NRCS Soil Data Access (SDA), REST POST to
    https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest
with JSON {"query": <SQL>, "format": "JSON+COLUMNNAME"} (sdmdataaccess.nrcs.usda.gov/
WebServiceHelp.aspx). Checked against the live service:

- `mupolygon` holds the SSURGO polygons (STATSGO is the separate `gsmmupolygon`):
  `mupolygonkey`, `mukey`, `areasymbol` (soil survey area), `spatialversion`,
  `musym`, and the geometry twice, in `mupolygongeo` (SQL Server geometry, SRID
  4326, x = longitude) and `mupolygonproj` (spherical Web Mercator).
  documents/AdvancedQueries.html, section "Introduction".
- `sacatalog` names each survey area (`areaname`) and dates its last export
  (`saverest`). Florida has 74 survey areas (checked 2026-10-14). Most are one
  county, or the part of a county outside federal and tribal land ("Collier
  County Area", "Miami-Dade County Area"). Seven are federal or tribal land:
  FL609 Ocala National Forest, FL615 Water Conservation Area, FL616 Big Cypress
  Reservation, FL618 Miccosukee Indian Alligator Alley Reservation, FL622 Big
  Cypress National Preserve, FL623 Everglades National Park and FL624 Biscayne
  National Park. In every Florida survey area all polygons carry one
  `spatialversion`.
- `SDA_Get_Mupolygonkey_from_intersection_with_WktWgs84(wkt)` returns the
  mupolygonkeys of the polygons a WGS84 WKT geometry intersects
  (AdvancedQueries.html, "Tabular Functions"). Its sibling `..._Mukey_...`
  returns map units only, without polygons, so it is not enough for lines.
- Limits: at most 100,000 rows per query (WebServiceHelp.aspx) and 32 MB of
  serialised JSON (soilDB `SDA_query` documentation). A rejected query is HTTP
  400 with an OGC ServiceException XML body; an empty result is the JSON `{}`.
- `muaggatt` has the seven attributes below plus `musym`, `muname`, `mukey`.
  SDA types `pondfreqprs` as varchar and `hydclprs` as int; both are written as
  nullable integers.

WHAT THIS SCRIPT DECIDES, AND WHY
=================================
1. KEYS BY TILE, GEOMETRY BY KEY.  Fetching geometry tile by tile would download
   every polygon that crosses a tile edge once per tile, and one tile's WKT could
   approach the 32 MB cap where polygons are large and detailed. So step 1 asks
   each tile (--tile-deg, default 0.05 deg, about 4.8 x 5.6 km at 30 N) only for
   key, mukey, areasymbol and vertex count (`STNumPoints`), a few kB, and caches
   that per tile; a tile that reaches the 100,000-row cap is split into four.
   Step 2 fetches each unique polygon once, in batches capped at --max-pts
   vertices (default 200,000, about 7 MB of JSON at the measured 36 bytes per
   vertex) and --max-keys keys. A batch the server rejects for size, time or
   memory is halved and asked again. Geometry and attributes are cached per batch
   under ssurgo/cache/, so an interrupted run resumes and overlapping bboxes (the
   classic and spatial selections share ground) never fetch a polygon twice.
   Step 1 runs for every --bboxes file before step 2 starts, and the size gate
   (note 8) sits between them.

2. WHOLE POLYGONS, NOT CLIPPED.  A polygon that crosses the bbox edge is kept
   whole. Every soil line inside the bbox is then complete and the polygon on each
   side of it is known; the lines run past the bbox wherever the polygons do. The
   outer edge of the fetched set is not a soil line and is tagged `clip_edge`.
   `in_bbox` marks the polygons that meet the EPSG:5070 bbox itself (the WGS84
   envelope that is queried is slightly larger). A polygon inside two components'
   bboxes is written once per component.

3. ATTRIBUTES ARE MAP-UNIT AGGREGATES (`muaggatt`).  flodfreqdcd (flooding
   frequency class, dominant condition), flodfreqmax (most frequent class of any
   component), pondfreqprs (percent of the map unit with ponding), hydclprs
   (percent of the map unit that is hydric), drclassdcd (drainage class, dominant
   condition), drclasswettest (wettest drainage class of any component),
   hydgrpdcd (hydrologic soil group, dominant condition). 'None' in the flooding
   classes is a class (no flooding), not a null, and is kept as the string 'None'.

4. SOIL LINES FROM SHARED BOUNDARIES.  SSURGO is built topologically, and on the
   self-test box it is exactly so: neighbouring polygons share bit-identical
   vertices, also across the FL007/FL125 (Bradford/Union) survey-area join, and
   the union of the polygons has exactly the sum of their areas. So the lines come
   from noding, not from geometric matching. All polygon rings, in EPSG:5070, are
   unioned with GEOS snap-rounding at --snap-m (default 0.1 m): each shared edge
   dissolves into one line, T-junctions are noded, and float noise or a misfit
   under a decimetre at a survey join is absorbed. 0.1 m is 1/300 of the 30 m
   vertex spacing of the lobes, so the snap moves nothing the analysis can see.
   Each noded piece gets its two sides by probing --probe-m (default 0.5 m, at
   least 2 x --snap-m) left and right of the midpoint of its longest segment. A
   side whose polygon does not touch the piece (a polygon thinner than the probe),
   or that finds no polygon, is probed again closer. Then the whole piece is
   checked against the polygons it was given (every vertex and segment midpoint
   within 2 x --snap-m of their rings); a piece that fails is probed segment by
   segment. Pieces with the same pair of polygons are merged into lines.
   A misfit larger than the snap is NOT repaired, and a bigger --snap-m is not a
   reliable repair (snap-rounding can leave two lines 1 m apart in neighbouring
   grid cells). It is reported instead, per component: a gap leaves both edges
   one-sided (no_data_edge inside the bbox), an overlap shows as probes landing
   in two polygons, and the perimeter balance (2 x two-sided length + one-sided
   length = total polygon perimeter) moves off 1 when pieces are lost or doubled.
   Any of these prints a WARNING for that component.

5. LINE KINDS.
       soil_line     two polygons, different mukey, same survey area, neither of
                     them water
       survey_join   two polygons in different survey areas, neither of them
                     water. This is an administrative line: a county line, or the
                     edge of a National Forest, Park or Preserve, the Water
                     Conservation Area or a reservation (in the Miami-Dade bboxes
                     most joins are of that second sort). areaname_a / areaname_b
                     say which. It may also carry a real soil change. A mukey
                     belongs to one survey area, so a join always changes mukey,
                     even where the same soil runs across it (Foxworth fine sand is
                     mukey 631628 in FL007 and 1414581 in FL125, with identical
                     muaggatt values). The diff columns, not the mukey, say whether
                     the soil changes.
       water_edge    two polygons, at least one of them a water map unit, whatever
                     the survey areas: the shore of a lake, river, bay or the sea.
                     SSURGO maps open water as map units, so a shore is two-sided
                     and nearly every attribute differs across it, but it is not a
                     line drawn between two soils. Class S must exclude it, i.e.
                     use soil_line only.
       clip_edge     one side only, outside the queried envelope: the edge of what
                     was fetched, not a soil line
       no_data_edge  one side only, inside the queried envelope: SSURGO has no
                     polygon there (open water beyond the survey, an unmapped gap)
   Boundaries between two polygons of the same mukey are dropped and counted.
   A WATER MAP UNIT (`water_unit` on the polygons) is one whose muname starts with
   the word "Water" or "Waters" ("Water", "Waters of the Atlantic Ocean", "Waters
   of the Gulf of America", "Water-Udorthents complex"), or is a "Dredge channel".
   This rule was checked on 2026-10-14 against every Florida map unit with a
   component named Water. It picks exactly the units whose Water components make
   up at least half of the unit (55-100%). It also picks one "Waters of the Gulf
   of America" unit whose only component is named "Gulf of America". It leaves out
   "Arents-Water complex" (25-45% water, Arents named first),
   "Udorthents-Water-Urban land complex" (no Water component at all) and
   "Tidewater muck". It also leaves out the subaqueous soils ("Thalinectes sand, 1
   to 2 meter water depth"): those are soils mapped on a lagoon floor, so their
   lines are soil lines. Matching "Water" anywhere in the name would wrongly catch
   the first two. `mapunit.mukind` cannot tell water from soil, because water
   units are "Consociation" or null, the same as soil units.

6. WHAT "DIFFERS" MEANS.  diff_<attr> is True when the values on the two sides
   differ, with null treated as its own value (null against a value differs, null
   against null does not). n_diff counts the True ones and diff_attrs joins their
   names with ';' ('' when none). For one-sided kinds all of them are null. The
   values themselves join from the polygons table by mukey_a / mukey_b.

7. FALLBACK, NOT IMPLEMENTED: gSSURGO.  The Florida gSSURGO state file
   geodatabase (NRCS Geospatial Data Gateway / the NRCS gSSURGO download share)
   carries MUPOLYGON and the same muaggatt table, so the same outputs could be
   built from it offline. Its size was not checked; under the task rules Victor
   must be asked before any download over 2 GB. SDA served the v1 bboxes within
   the gate of note 8, so the fallback is only for an SDA outage.

8. NOTHING LARGE WITHOUT A YES.  The download is not small. On the three v1
   prefixes (bboxes 45-138 km a side), exact server-side counts gave per prefix
   69,500-118,121 polygons and 7.7-12.3 M vertices, about 278-445 MB of JSON. All
   three together, without duplicates, came to 195,794 polygons, 21.1 M vertices
   and about 767 MB. The largest single polygon is FL623 "Waters of the Gulf of
   America" with 221,396 vertices. Disk use is about 1.5 GB, measured on the
   self-test at ~17 B per vertex for the cache and ~37 B for the outputs; outputs
   repeat a polygon for every component and prefix that holds it. So every run
   first does step 1 for all --bboxes, which is a few kB per tile and is cached.
   It then prints the network and disk estimate for the geometry not yet cached.
   Without --yes (or with --plan) it stops there and exits 0. With --yes it
   refuses (exit 2) when the network estimate exceeds --max-gb (default 2.0, where
   the task brief says to ask first) or when the disk estimate would leave less
   than --min-free-gb (default 10) free. Run without --yes first: its total decides
   whether to ask.

9. A RERUN SKIPS ONLY ON THE SAME BBOXES.  The prefix (<cell>_<variant>_S<S>) is
   the same for v1 and v2, and the output folder is not dated, so a file name
   says nothing about which bboxes the file was built for. Each output therefore
   stores its input bboxes (component, minx..maxy, the lon/lat envelope) as JSON
   under the parquet metadata key b"pull_ssurgo". A rerun skips a prefix only when
   both outputs store exactly the bboxes of the file given now. If either one
   stores other bboxes, or none (it was written before this record existed), the
   run refuses before any query and asks for --overwrite, which rebuilds; so does
   --refetch.

10. ONE SSURGO VERSION.  SSURGO is refreshed every October, and each cache (keys,
    geometry, attributes) is a snapshot of the day it was fetched.
    `spatialversion` and `saverest` (the survey area's last export date) are
    carried on every polygon. After the fetch, these must all hold:
    - every mupolygonkey from step 1 comes back from the geometry query, with the
      same mukey;
    - every mukey has a muaggatt row (all 3,962 Florida mukeys did on
      2026-10-14);
    - every areasymbol has a sacatalog row;
    - each survey area's polygons carry a single spatialversion.
    If any fails, the caches span a refresh, and the run stops with an error that
    asks for --refetch, so that everything comes from one SSURGO version. Carrying
    on would leave the vanished polygons as holes, and those turn into
    no_data_edge lines, which on a coast look like open water.

USAGE
=====
    python pull_ssurgo.py --selftest --gis-dir /tmp/ssurgo_test
                                  # 2 km box at Bradford/Union, 0.01 deg tiles,
                                  # checks the invariants above
    python pull_ssurgo.py --bboxes results/$D/boundary/*_bboxes.parquet
                                  # plan: keys only, then the size and disk estimate
    python pull_ssurgo.py --bboxes ... --yes              # fetch and build
    python pull_ssurgo.py --bboxes ... --yes --overwrite  # rebuild outputs from the cache
    python pull_ssurgo.py --bboxes ... --yes --refetch    # ignore the cache, ask SDA again
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

import geopandas as gpd
import shapely
from pyproj import Transformer

GIS_DIR = Path("data/interim/gis")
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

CRS_GEO = 4326      # mupolygongeo
CRS_WORK = 5070     # CONUS Albers, metres, the project's GIS frame

ATTRS = ["flodfreqdcd", "flodfreqmax", "pondfreqprs", "hydclprs",
         "drclassdcd", "drclasswettest", "hydgrpdcd"]
INT_ATTRS = ("pondfreqprs", "hydclprs")
BBOX_COLS = ["component", "minx", "miny", "maxx", "maxy",
             "lon_min", "lat_min", "lon_max", "lat_max"]
KINDS = ("soil_line", "survey_join", "water_edge", "clip_edge", "no_data_edge")
WATER_MU = re.compile(r"^(Waters?|Dredge channel)\b")      # note 5
META_KEY = b"pull_ssurgo"                                  # note 9

ROW_LIMIT = 100_000       # SDA rows per query
BYTES_PER_PT = 36.3       # measured: 150,145 bytes of JSON for 4,135 vertices
CACHE_B_PER_PT = 18.0     # measured on the self-test: 17.3 B/vertex of geometry cache
OUT_B_PER_PT = 37.0       # and 22.0 + 14.6 B/vertex of mupolygons + mulines; note 8
RETRIES = 5
CONNECT_S = 30            # no connection in 30 s is an outage, not a big query
TIMEOUT_S = 300

DEFAULTS = dict(
    tile_deg=0.05,        # key-query tile edge, degrees; note 1
    workers=2,            # concurrent SDA requests; a government server
    max_pts=200_000,      # vertices per geometry request; note 1
    max_keys=1000,        # mupolygonkeys per geometry request
    snap_m=0.1,           # snap-rounding grid for noding the rings; note 4
    probe_m=0.5,          # side probe distance; note 4
    max_gb=2.0,           # network ceiling for --yes; note 8
    min_free_gb=10.0,     # free-disk floor for --yes; note 8
)
SELFTEST = dict(lon=-82.30, lat=29.95, half_m=1000.0, tile_deg=0.01)

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
ATTR_BATCH = 1000

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
    m = re.search(r"<ServiceException>(.*?)</ServiceException>", text, re.S)
    return html.unescape(m.group(1)).strip() if m else text.strip()[:300]


class SDA:
    """A polite POST client: retries with backoff, counts requests and bytes."""

    def __init__(self, workers: int):
        self.s = requests.Session()
        self.s.mount("https://", HTTPAdapter(pool_maxsize=max(2, 2 * workers)))
        self.lock = threading.Lock()
        self.n_req = 0
        self.n_bytes = 0

    def query(self, sql: str, cols: list[str]) -> pd.DataFrame:
        err: SDAError | None = None
        for attempt in range(RETRIES):
            if attempt:
                time.sleep(min(60.0, 2.0 ** attempt) + random.random())
            try:
                r = self.s.post(SDA_URL, json={"query": sql, "format": "JSON+COLUMNNAME"},
                                timeout=(CONNECT_S, TIMEOUT_S))
            except requests.ConnectTimeout as e:    # the server is down: retry, never split
                err = SDAError(f"no connection in {CONNECT_S}s: {e}")
                continue
            except requests.Timeout as e:     # no answer in TIMEOUT_S: once more, then split
                err = SDATooBig(f"no response in {TIMEOUT_S}s: {e}")
                if attempt >= 1:
                    break
                continue
            except requests.RequestException as e:
                err = SDAError(f"{type(e).__name__}: {e}")
                continue
            with self.lock:
                self.n_req += 1
                self.n_bytes += len(r.content)
            if r.status_code == 200:
                try:
                    table = r.json().get("Table")
                except ValueError:           # truncated body: retry, then split
                    err = SDATooBig(f"unparseable response, {len(r.content):,} bytes")
                    continue
                if not table:
                    return pd.DataFrame(columns=cols)
                if table[0] != cols:
                    raise SDAError(f"unexpected columns {table[0]}, wanted {cols}")
                return pd.DataFrame(table[1:], columns=cols)
            msg = f"HTTP {r.status_code}: {_service_message(r.text)}"
            if _TOO_BIG.search(msg):
                err = SDATooBig(msg)
                if attempt >= 1:             # a timeout can be load; a cap will not move
                    break
            elif r.status_code in (429, 500, 502, 503, 504):
                err = SDAError(msg)
            else:
                raise SDAError(msg)          # bad SQL: retrying will not help
        assert err is not None
        raise err


# =========================================================================
# caching
# =========================================================================
def _write(df: pd.DataFrame, path: Path) -> None:
    """Atomic: a killed run never leaves a half-written cache file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _digest(keys) -> str:
    return hashlib.sha1(",".join(map(str, keys)).encode()).hexdigest()[:16]


def _load_store(d: Path, pattern: str, key: str, cols: list[str]) -> pd.DataFrame | None:
    """All cached batches of one kind; where a key was fetched twice (--refetch),
    the newest file wins. A file without all of `cols` (written before a column
    was added) is ignored, so its keys are fetched again."""
    files = sorted(d.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True) \
        if d.exists() else []
    files = [f for f in files if set(cols) <= set(pq.read_schema(f).names)]
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files],
                     ignore_index=True).drop_duplicates(key)


# =========================================================================
# step 1: keys by tile
# =========================================================================
def _outward(v: float, up: bool) -> float:
    """Round to 1e-6 deg (0.1 m) away from the box, so rounding never loses area."""
    return (math.ceil(v * 1e6) if up else math.floor(v * 1e6)) / 1e6


def _edges(a: float, b: float, step: float) -> list[float]:
    inner = [round(k * step, 6) for k in range(math.floor(a / step) + 1, math.ceil(b / step))]
    return [a] + [v for v in inner if a < v < b] + [b]


def envelope(row) -> tuple[float, float, float, float]:
    return (_outward(row.lon_min, False), _outward(row.lat_min, False),
            _outward(row.lon_max, True), _outward(row.lat_max, True))


def tiles_for(env: tuple, step: float) -> list[tuple]:
    """The WGS84 envelope cut on a global grid of `step` degrees; tiles are clipped
    to the envelope so a bbox never pulls polygons from beyond it."""
    xs = _edges(env[0], env[2], step)
    ys = _edges(env[1], env[3], step)
    return [(x0, y0, x1, y1) for y0, y1 in zip(ys[:-1], ys[1:])
            for x0, x1 in zip(xs[:-1], xs[1:])]


def _wkt(t: tuple) -> str:
    x0, y0, x1, y1 = (f"{v:.6f}" for v in t)
    return f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def _tile_name(t: tuple) -> str:
    return "_".join(f"{v:.6f}" for v in t) + ".parquet"


def query_tile(sda: SDA, t: tuple) -> pd.DataFrame:
    df = sda.query(KEY_SQL.format(wkt=_wkt(t)), KEY_COLS)
    if len(df) >= ROW_LIMIT:                 # capped: the answer may be cut short
        xm, ym = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
        quads = [(t[0], t[1], xm, ym), (xm, t[1], t[2], ym),
                 (t[0], ym, xm, t[3]), (xm, ym, t[2], t[3])]
        df = pd.concat([query_tile(sda, q) for q in quads]).drop_duplicates("mupolygonkey")
    return df.astype({"mupolygonkey": "int64", "mukey": "int64", "npts": "int64"})


def tile_keys(sda: SDA, tiles: list[tuple], cache: Path, refetch: bool,
              workers: int) -> dict[tuple, pd.DataFrame]:
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

    def one(t):
        df = query_tile(sda, t)
        _write(df, cache / _tile_name(t))
        return t, df

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for t, df in ex.map(one, todo):
            out[t] = df
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"    {done}/{len(todo)} tiles", flush=True)
    return out


# =========================================================================
# step 2: geometry and attributes by key
# =========================================================================
def _geom_batches(keys: pd.DataFrame, max_pts: int, max_keys: int) -> list[list[int]]:
    out, cur, pts = [], [], 0
    for k, n in zip(keys.mupolygonkey, keys.npts):
        if cur and (pts + n > max_pts or len(cur) >= max_keys):
            out.append(cur)
            cur, pts = [], 0
        cur.append(int(k))
        pts += int(n)
    if cur:
        out.append(cur)
    return out


def _fetch_geom(sda: SDA, keys: list[int]) -> pd.DataFrame:
    try:
        return sda.query(GEOM_SQL.format(keys=",".join(map(str, keys))), GEOM_COLS)
    except SDATooBig as e:
        if len(keys) == 1:
            raise SDAError(f"mupolygonkey {keys[0]} alone is too big for SDA: {e}") from e
        h = len(keys) // 2
        print(f"    batch of {len(keys)} rejected ({e}); halving", flush=True)
        return pd.concat([_fetch_geom(sda, keys[:h]), _fetch_geom(sda, keys[h:])])


def fetch_geometry(sda: SDA, need: pd.DataFrame, store: pd.DataFrame | None,
                   cache: Path, a) -> pd.DataFrame:
    have = set() if store is None else set(store.mupolygonkey)
    miss = need[~need.mupolygonkey.isin(have)].sort_values("mupolygonkey")
    batches = _geom_batches(miss, a.max_pts, a.max_keys)
    print(f"  geometry: {len(need):,} polygons, {len(need) - len(miss):,} cached, "
          f"{len(miss):,} to fetch in {len(batches)} requests "
          f"(~{miss.npts.sum() * BYTES_PER_PT / 1e6:.1f} MB)", flush=True)

    def one(keys):
        df = _fetch_geom(sda, keys)
        df = df.astype({"mupolygonkey": "int64", "mukey": "int64", "spatialversion": "int64"})
        df["wkb"] = shapely.to_wkb(shapely.from_wkt(df.pop("wkt").to_numpy()))
        _write(df, cache / f"g_{_digest(keys)}.parquet")
        return df

    new = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, df in enumerate(ex.map(one, batches), 1):
            new.append(df)
            print(f"    {i}/{len(batches)} geometry requests", flush=True)
    parts = ([] if store is None else [store]) + new
    return pd.concat(parts, ignore_index=True).drop_duplicates("mupolygonkey")


def fetch_table(sda: SDA, values: list, store: pd.DataFrame | None, key: str,
                sql: str, cols: list[str], prefix: str, cache: Path,
                quote: bool) -> pd.DataFrame:
    """A small per-key table (muaggatt by mukey, sacatalog by areasymbol)."""
    have = set() if store is None else set(store[key])
    miss = sorted(v for v in set(values) if v not in have)
    new = []
    for i in range(0, len(miss), ATTR_BATCH):
        chunk = miss[i:i + ATTR_BATCH]
        lst = ",".join(f"'{v}'" if quote else str(v) for v in chunk)
        df = sda.query(sql.format(keys=lst), cols)
        if key == "mukey":
            df["mukey"] = df.mukey.astype("int64")
        _write(df, cache / f"{prefix}_{_digest(chunk)}.parquet")
        new.append(df)
    if miss:
        print(f"  {prefix} table: {len(miss)} keys fetched", flush=True)
    parts = ([] if store is None else [store]) + new
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).drop_duplicates(key)


def check_one_version(need: pd.DataFrame, stores: dict) -> None:
    """Note 10: the keys, geometry and attributes must come from one SSURGO version."""
    g = stores["geom"][stores["geom"].mupolygonkey.isin(need.mupolygonkey)]
    probs = []
    gone = sorted(set(need.mupolygonkey) - set(g.mupolygonkey))
    if gone:
        probs.append(f"{len(gone):,} of {len(need):,} mupolygonkeys from the key step are "
                     f"not in the geometry (e.g. {gone[:5]})")
    m = need[["mupolygonkey", "mukey"]].merge(g[["mupolygonkey", "mukey"]],
                                              on="mupolygonkey", suffixes=("", "_g"))
    if (m.mukey != m.mukey_g).any():
        probs.append(f"{int((m.mukey != m.mukey_g).sum()):,} polygons changed mukey "
                     f"between the key step and the geometry")
    sv = g.groupby("areasymbol").spatialversion.nunique()
    if (sv > 1).any():
        probs.append(f"polygons of {', '.join(sv.index[sv > 1])} carry more than one "
                     f"spatialversion")
    for what, vals, have in (("muaggatt", need.mukey, stores["attr"].mukey),
                             ("sacatalog", need.areasymbol, stores["sa"].areasymbol)):
        miss = sorted(set(vals) - set(have))
        if miss:
            probs.append(f"{len(miss):,} of {vals.nunique():,} keys have no {what} row "
                         f"(e.g. {miss[:5]})")
    if probs:
        raise SDAError("the cached and fetched SSURGO data disagree: " + "; ".join(probs)
                       + ". The caches span an SSURGO refresh; re-run with --refetch so "
                       "keys, geometry and attributes come from one version.")


# =========================================================================
# polygons
# =========================================================================
def _polygonal(g):
    """make_valid, keeping only the polygonal part of what it returns."""
    v = shapely.make_valid(g)
    if shapely.get_type_id(v) in (3, 6):
        return v
    parts = shapely.get_parts(v)
    return shapely.union_all(parts[np.isin(shapely.get_type_id(parts), (3, 6))])


def build_polygons(comp: int, keys: pd.DataFrame, geom: pd.DataFrame,
                   attr: pd.DataFrame, sa: pd.DataFrame, bbox5070,
                   to_work: Transformer) -> tuple[gpd.GeoDataFrame, int]:
    g = geom[geom.mupolygonkey.isin(keys.mupolygonkey)].copy()
    shp = shapely.from_wkb(g.pop("wkb").to_numpy())
    shp = shapely.transform(shp, lambda xy: np.column_stack(to_work.transform(xy[:, 0], xy[:, 1])))
    bad = ~shapely.is_valid(shp)
    for i in np.nonzero(bad)[0]:
        shp[i] = _polygonal(shp[i])
    g = g.merge(attr, on="mukey", how="left").merge(sa, on="areasymbol", how="left")
    for c in INT_ATTRS:
        g[c] = pd.to_numeric(g[c], errors="coerce").astype("Int64")
    g["water_unit"] = g.muname.fillna("").str.match(WATER_MU).astype(bool)
    g.insert(0, "component", comp)
    out = gpd.GeoDataFrame(g, geometry=shp, crs=CRS_WORK)
    out["area_m2"] = out.geometry.area
    out["in_bbox"] = out.geometry.intersects(bbox5070)
    cols = (["component", "mupolygonkey", "mukey", "areasymbol", "areaname", "spatialversion",
             "saverest", "musym", "muname", "water_unit"] + ATTRS
            + ["area_m2", "in_bbox", "geometry"])
    return out[cols].sort_values("mupolygonkey").reset_index(drop=True), int(bad.sum())


# =========================================================================
# lines
# =========================================================================
def _probe_frame(pieces: np.ndarray):
    """Midpoint and unit left-normal of each piece's longest segment."""
    xy, ii = shapely.get_coordinates(pieces, return_index=True)
    k = np.nonzero(ii[:-1] == ii[1:])[0]                  # segment start rows
    d = xy[k + 1] - xy[k]
    seg = np.hypot(d[:, 0], d[:, 1])
    piece = ii[k]
    order = np.lexsort((-seg, piece))
    first = order[np.r_[True, piece[order][1:] != piece[order][:-1]]]
    assert np.array_equal(piece[first], np.arange(len(pieces)))
    mid = xy[k[first]] + 0.5 * d[first]
    t = d[first] / seg[first, None]
    return mid, np.column_stack([-t[:, 1], t[:, 0]])


def _probe(tree, geoms: np.ndarray, area: np.ndarray, rings: np.ndarray, pts: np.ndarray,
           mid: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """Index of the polygon containing each probe point (-1: none). Where polygons
    overlap, the one whose ring touches the piece wins, then the smallest; also
    returns how many probes landed in more than one polygon. The tree only finds
    candidates by envelope; the point-in-polygon test runs on the prepared `geoms`
    (the tree's own predicate="within" prepares the point instead and scans the
    whole ring of the polygon for every probe)."""
    q = tree.query(shapely.points(pts))
    q = q[:, shapely.contains_xy(geoms[q[1]], pts[q[0], 0], pts[q[0], 1])]
    idx = np.full(len(pts), -1, dtype=np.int64)
    if not q.shape[1]:
        return idx, 0
    hit = pd.DataFrame({"p": q[0], "g": q[1], "a": area[q[1]]})
    n_multi = int((hit.p.value_counts() > 1).sum())
    if n_multi:
        hit["far"] = ~shapely.dwithin(rings[hit.g.to_numpy()],
                                      shapely.points(mid[hit.p.to_numpy()]), tol)
        hit = hit.sort_values(["p", "far", "a"])
    hit = hit.drop_duplicates("p")
    idx[hit.p.to_numpy()] = hit.g.to_numpy()
    return idx, n_multi


def _single_lines(geoms: np.ndarray, src: np.ndarray):
    """Explode to LineStrings, dropping the points an intersection can leave."""
    parts, j = shapely.get_parts(geoms, return_index=True)
    parts, src = parts, src[j]
    while (shapely.get_type_id(parts) >= 4).any():         # nested collections
        parts, j = shapely.get_parts(parts, return_index=True)
        src = src[j]
    keep = (shapely.get_type_id(parts) == 1) & (shapely.length(parts) > 1e-6)
    return parts[keep], src[keep]


def _assign_sides(pieces, tree, geoms, area, rings, snap_m, probe_m, tol, st) -> np.ndarray:
    """(n, 2) polygon index left and right of each piece, -1 where there is none.
    `geoms` and `rings` are prepared, so each test is a lookup, not a scan of the ring."""
    mid, nrm = _probe_frame(pieces)
    side = np.empty((len(pieces), 2), dtype=np.int64)
    for s, sign in enumerate((1.0, -1.0)):
        idx, nm = _probe(tree, geoms, area, rings, mid + sign * probe_m * nrm, mid, tol)
        st["overlap_probes"] += nm
        far = np.ones(len(idx), bool)
        got = idx >= 0
        far[got] = ~shapely.dwithin(rings[idx[got]], shapely.points(mid[got]), tol)
        st["nonadjacent"] += int((far & got).sum())
        if far.any():                    # a polygon thinner than the probe, or none
            near = max(0.1 * probe_m, snap_m)    # still clear of the snap shift
            idx2, nm = _probe(tree, geoms, area, rings, mid[far] + sign * near * nrm[far],
                              mid[far], tol)
            st["overlap_probes"] += nm
            ok = idx2 >= 0
            ok[ok] = shapely.dwithin(rings[idx2[ok]], shapely.points(mid[far][ok]), tol)
            st["unresolved_sides"] += int(((idx2 >= 0) & ~ok).sum())
            idx2[~ok] = -1
            idx[far] = idx2
        side[:, s] = idx
    return side


def _on_sides(pieces, side, rings, tol) -> np.ndarray:
    """True where every vertex and segment midpoint of a piece lies within tol of
    the ring of each polygon assigned to it. A probe reads one point; this checks
    that the whole piece runs along those polygons, which fails only where a
    neighbour's boundary was not noded into the piece (a misfit beyond the snap)."""
    xy, ii = shapely.get_coordinates(pieces, return_index=True)
    k = ii[:-1] == ii[1:]
    pts = np.vstack([xy, 0.5 * (xy[:-1][k] + xy[1:][k])])
    own = np.concatenate([ii, ii[:-1][k]])
    ok = np.ones(len(pieces), bool)
    for s in (0, 1):
        g = side[own, s]
        has = g >= 0
        near = np.ones(len(pieces), bool)
        np.logical_and.at(near, own[has], shapely.dwithin(rings[g[has]],
                                                          shapely.points(pts[has]), tol))
        ok &= near
    return ok


def build_lines(polys: gpd.GeoDataFrame, query_area, snap_m: float,
                probe_m: float) -> tuple[gpd.GeoDataFrame | None, dict]:
    st = dict(overlap_probes=0, nonadjacent=0, unresolved_sides=0)
    geoms = polys.geometry.to_numpy()
    rings = shapely.boundary(geoms)
    # Unprepared, a point-in-polygon or point-to-ring test scans the ring, and a
    # V-vertex polygon gives ~V pieces to test, so a 221,396-vertex water polygon
    # would cost V^2. Prepared, both are index lookups.
    shapely.prepare(geoms)
    shapely.prepare(rings)
    pieces = shapely.get_parts(shapely.union_all(rings, grid_size=snap_m))
    pieces = pieces[shapely.length(pieces) > 0]
    st["pieces"] = len(pieces)

    tree = shapely.STRtree(geoms)
    area = shapely.area(geoms)
    tol = 2 * snap_m + 1e-3
    side = _assign_sides(pieces, tree, geoms, area, rings, snap_m, probe_m, tol, st)
    bad = ~_on_sides(pieces, side, rings, tol)
    st["split_pieces"] = int(bad.sum())
    if bad.any():                        # probe those segment by segment instead
        xy, ii = shapely.get_coordinates(pieces[bad], return_index=True)
        k = ii[:-1] == ii[1:]
        segs = shapely.linestrings(np.stack([xy[:-1][k], xy[1:][k]], axis=1))
        segs = segs[shapely.length(segs) > 0]
        pieces = np.concatenate([pieces[~bad], segs])
        side = np.vstack([side[~bad],
                          _assign_sides(segs, tree, geoms, area, rings, snap_m, probe_m,
                                        tol, st)])

    a = np.where(side.min(1) >= 0, side.min(1), side.max(1))
    b = np.where(side.min(1) >= 0, side.max(1), -1)
    plen = shapely.length(pieces)
    lost = a < 0                                          # no polygon on either side
    inner = (a == b) & (a >= 0)                           # a cut line inside one polygon
    st["lost_pieces"], st["lost_m"] = int(lost.sum()), float(plen[lost].sum())
    st["inner_pieces"], st["inner_m"] = int(inner.sum()), float(plen[inner].sum())

    keep = ~lost & ~inner
    pieces, a, b = pieces[keep], a[keep], b[keep]
    if not len(pieces):
        return None, st
    grp = pd.DataFrame({"a": a, "b": b}).groupby(["a", "b"], sort=True).ngroup().to_numpy()
    order = np.argsort(grp, kind="stable")
    multi = shapely.multilinestrings(pieces[order], indices=grp[order])
    pair = pd.DataFrame({"a": a[order], "b": b[order]}).drop_duplicates().to_numpy()
    merged = shapely.line_merge(multi)
    lines, li = _single_lines(merged, np.arange(len(merged)))
    la, lb = pair[li, 0], pair[li, 1]

    one = lb < 0
    q_in = shapely.buffer(query_area, -1.0)
    ins, si = _single_lines(shapely.intersection(lines[one], q_in), np.nonzero(one)[0])
    out, so = _single_lines(shapely.difference(lines[one], q_in), np.nonzero(one)[0])
    geom = np.concatenate([lines[~one], ins, out])
    ia = np.concatenate([la[~one], la[si], la[so]])
    ib = np.concatenate([lb[~one], lb[si], lb[so]])
    kind = np.array([""] * int((~one).sum()) + ["no_data_edge"] * len(ins)
                    + ["clip_edge"] * len(out), dtype=object)

    P = polys.reset_index(drop=True)
    df = pd.DataFrame({"component": P.component.iloc[0]}, index=np.arange(len(geom)))
    two = ib >= 0
    for c in ("mupolygonkey", "mukey", "areasymbol", "areaname"):
        text = c in ("areasymbol", "areaname")
        df[f"{c}_a"] = P[c].to_numpy()[ia]
        vb = pd.Series(pd.NA, index=df.index, dtype="string" if text else "Int64")
        vb[two] = P[c].to_numpy()[ib[two]]
        df[f"{c}_b"] = vb
        if text:
            df[f"{c}_a"] = df[f"{c}_a"].astype("string")

    same = two & (df.mukey_a.to_numpy() == df.mukey_b.fillna(-1).to_numpy())
    glen = shapely.length(geom)
    st["same_mukey_lines"], st["same_mukey_m"] = int(same.sum()), float(glen[same].sum())
    wet = P.water_unit.to_numpy(bool)
    water = two & (wet[ia] | wet[np.maximum(ib, 0)])
    join = df.areasymbol_a.to_numpy() != df.areasymbol_b.fillna("").to_numpy()
    kind[two] = np.where(water, "water_edge", np.where(join, "survey_join", "soil_line"))[two]
    df["kind"] = kind

    for c in ATTRS:
        va = P[c].to_numpy(dtype=object)[ia]
        vb = np.where(two, P[c].to_numpy(dtype=object)[np.maximum(ib, 0)], None)
        na_a, na_b = pd.isna(va), pd.isna(vb)
        differs = (na_a != na_b) | (~na_a & ~na_b & (va.astype(str) != vb.astype(str)))
        df[f"diff_{c}"] = pd.array(np.where(two, differs, False), dtype="boolean")
        df.loc[~two, f"diff_{c}"] = pd.NA
    dcols = [f"diff_{c}" for c in ATTRS]
    df["n_diff"] = df[dcols].sum(axis=1).astype("Int64")
    df.loc[~two, "n_diff"] = pd.NA
    names = np.array(ATTRS, dtype=object)
    flags = df[dcols].fillna(False).to_numpy(bool)
    df["diff_attrs"] = pd.array([";".join(names[f]) for f in flags], dtype="string")
    df.loc[~two, "diff_attrs"] = pd.NA
    df["length_m"] = glen

    keep = ~same
    out_gdf = gpd.GeoDataFrame(df[keep].reset_index(drop=True),
                               geometry=geom[keep], crs=CRS_WORK)
    st["perimeter_m"] = float(shapely.length(rings).sum())
    two_len = float(glen[two].sum()) + st["inner_m"]
    st["balance"] = (2 * two_len + float(glen[~two].sum())) / st["perimeter_m"]
    return out_gdf, st


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
    envs: dict = field(default_factory=dict)
    tiles: dict = field(default_factory=dict)
    comp_keys: dict = field(default_factory=dict)

    @property
    def need(self) -> pd.DataFrame:
        return pd.concat(self.comp_keys.values()).drop_duplicates("mupolygonkey")


def load_prefix(path: Path, gis_dir: Path) -> Prefix:
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
    """The input bboxes as the outputs store them (note 9). JSON writes a float's
    repr, so the record reads back bit-exact."""
    return [{c: int(v) if c == "component" else float(v) for c, v in zip(BBOX_COLS, row)}
            for row in bb[BBOX_COLS].itertuples(index=False)]


def stored_bboxes(path: Path) -> list | None:
    """The bbox record of an existing output, [] when it has none; None: no file."""
    if not path.exists():
        return None
    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(META_KEY, b"{}")).get("bboxes", [])


def output_state(p: Prefix) -> tuple[str, str]:
    """'absent'; 'same' (both outputs store these bboxes); 'partial' (one does and
    the other is missing: a run cut short, rebuilt without asking); or 'other',
    with the reason."""
    want = bbox_record(p.bb)
    have = {f: stored_bboxes(f) for f in (p.f_poly, p.f_line)}
    if all(h is None for h in have.values()):
        return "absent", ""
    if all(h == want for h in have.values()):
        return "same", ""
    if all(h is None or h == want for h in have.values()):
        return "partial", ""
    why, seen = [], []
    now = {r["component"]: r for r in want}
    for h in have.values():
        if h is None or h == want or h in seen:
            continue
        seen.append(h)
        names = " and ".join(f.name for f, k in have.items() if k == h)
        if not h:
            why.append(f"{names}: no input record (written by an older pull_ssurgo)")
            continue
        old = {r["component"]: r for r in h}
        diff = sorted(c for c in now.keys() | old.keys() if now.get(c) != old.get(c))
        why.append(f"{names}: built for components {sorted(old)}, {p.path.name} has "
                   f"{sorted(now)}; the bboxes differ for component(s) {diff}")
    return "other", "; ".join(why)


def plan_keys(p: Prefix, a, sda: SDA) -> None:
    """Step 1 for one prefix, and its per-component table."""
    print(f"\n== {p.prefix}: {len(p.bb)} components", flush=True)
    p.envs = {int(r.component): envelope(r) for r in p.bb.itertuples()}
    p.tiles = {c: tiles_for(e, a.tile_deg) for c, e in p.envs.items()}
    uniq = sorted({t for ts in p.tiles.values() for t in ts})
    tk = tile_keys(sda, uniq, a.gis_dir / "ssurgo" / "cache" / "keys", a.refetch, a.workers)
    p.comp_keys = {c: pd.concat([tk[t] for t in ts]).drop_duplicates("mupolygonkey")
                   for c, ts in p.tiles.items()}
    print(f"  {'comp':>4} {'tiles':>5} {'polygons':>8} {'mukeys':>6} {'vertices':>10} "
          f"{'~MB':>7}  areasymbols")
    for c, k in p.comp_keys.items():
        print(f"  {c:>4} {len(p.tiles[c]):>5} {len(k):>8,} {k.mukey.nunique():>6} "
              f"{k.npts.sum():>10,} {k.npts.sum() * BYTES_PER_PT / 1e6:>7.1f}  "
              f"{','.join(sorted(k.areasymbol.unique()))}", flush=True)
    need = p.need
    print(f"  {len(need):,} unique polygons, {need.npts.sum():,} vertices, "
          f"~{need.npts.sum() * BYTES_PER_PT / 1e6:.1f} MB of geometry", flush=True)


def gb(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e9:.3f} GB"


def free_bytes(path: Path) -> int:
    p = path.resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def size_gate(jobs: list[Prefix], need: pd.DataFrame, a, stores: dict) -> int | None:
    """Note 8: None to go on; otherwise the exit code (0 after a plan, 2 refused)."""
    have = set() if stores["geom"] is None else set(stores["geom"].mupolygonkey)
    new = need[~need.mupolygonkey.isin(have)]
    new_pts = int(new.npts.sum())
    out_pts = sum(int(k.npts.sum()) for p in jobs for k in p.comp_keys.values())
    net = new_pts * BYTES_PER_PT
    disk = new_pts * CACHE_B_PER_PT + out_pts * OUT_B_PER_PT
    free = free_bytes(a.gis_dir)
    print(f"\nsize: {len(need):,} unique polygons in {len(jobs)} prefixes, "
          f"{len(need) - len(new):,} cached; to fetch {len(new):,} polygons, "
          f"{new_pts:,} vertices", flush=True)
    print(f"  network ~{gb(net)} of geometry JSON (plus attribute tables of a few kB); "
          f"disk ~{gb(disk)} (cache {gb(new_pts * CACHE_B_PER_PT)}, outputs "
          f"{gb(out_pts * OUT_B_PER_PT)}); free now {gb(free)}", flush=True)
    over = net > a.max_gb * 1e9
    low = free - disk < a.min_free_gb * 1e9
    if a.plan or not a.yes:
        if over:
            print(f"  NOTE: exceeds --max-gb {a.max_gb}; --yes would refuse", flush=True)
        if low:
            print(f"  NOTE: would leave < {a.min_free_gb} GB free; --yes would refuse",
                  flush=True)
        how = " and ".join(s for s, on in (("with --yes", not a.yes),
                                           ("without --plan", a.plan)) if on)
        print(f"  plan only: no geometry fetched. Re-run {how} to fetch and build.", flush=True)
        return 0
    if over:
        print(f"REFUSED: ~{gb(net)} exceeds --max-gb {a.max_gb}. Ask before raising "
              f"--max-gb.", file=sys.stderr)
        return 2
    if low:
        print(f"REFUSED: would leave ~{gb(free - disk)} free (< --min-free-gb "
              f"{a.min_free_gb}).", file=sys.stderr)
        return 2
    return None


def _query_area_work(env: tuple, to_work: Transformer):
    """The queried WGS84 envelope in EPSG:5070, densified so its edges stay true."""
    q = shapely.segmentize(shapely.box(*env), 0.001)
    return shapely.transform(q, lambda xy: np.column_stack(to_work.transform(xy[:, 0], xy[:, 1])))


def write_outputs(parts: list[tuple[gpd.GeoDataFrame, Path]], meta: dict) -> None:
    """GeoParquet with `meta` under META_KEY; each file appears whole or not at all."""
    for gdf, f in parts:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f.suffix + f".tmp{os.getpid()}")
        gdf.to_parquet(tmp)
        t = pq.read_table(tmp)
        pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}),
                                                  META_KEY: json.dumps(meta).encode()}), tmp)
        os.replace(tmp, f)


def build_prefix(p: Prefix, a, stores: dict) -> None:
    print(f"\n== {p.prefix}: building", flush=True)
    to_work = Transformer.from_crs(CRS_GEO, CRS_WORK, always_xy=True)
    polys_all, lines_all = [], []
    for r in p.bb.itertuples():
        c = int(r.component)
        if p.comp_keys[c].empty:
            print(f"  component {c}: no SSURGO polygons in the envelope")
            continue
        t0 = time.time()
        box5070 = shapely.box(r.minx, r.miny, r.maxx, r.maxy)
        polys, n_fixed = build_polygons(c, p.comp_keys[c], stores["geom"], stores["attr"],
                                        stores["sa"], box5070, to_work)
        lines, st = build_lines(polys, _query_area_work(p.envs[c], to_work), a.snap_m, a.probe_m)
        polys_all.append(polys)
        if lines is not None:
            lines_all.append(lines)
        print(f"  component {c}: {len(polys):,} polygons ({int(polys.in_bbox.sum()):,} in the "
              f"5070 bbox, {int(polys.water_unit.sum()):,} water), {polys.mukey.nunique()} "
              f"mukeys, {polys.areasymbol.nunique()} survey areas, {n_fixed} invalid fixed; "
              f"{st['pieces']:,} noded pieces -> {0 if lines is None else len(lines):,} lines "
              f"[{time.time() - t0:.1f}s]", flush=True)
        if lines is None:
            continue
        print(f"    probes: {st['nonadjacent']} hit a non-adjacent polygon (re-probed closer), "
              f"{st['unresolved_sides']} still unresolved, {st['overlap_probes']} in "
              f"overlapping polygons, {st['split_pieces']} pieces split to segments; dropped: "
              f"{st['same_mukey_lines']} same-mukey lines ({st['same_mukey_m']:.0f} m), "
              f"{st['inner_pieces']} inner pieces ({st['inner_m']:.0f} m), "
              f"{st['lost_pieces']} sideless pieces ({st['lost_m']:.0f} m)")
        nd = lines[lines.kind == "no_data_edge"]
        nd_in = float(nd.intersection(box5070).length.sum())
        print(f"    perimeter balance (2*two-sided + one-sided) / perimeter = "
              f"{st['balance']:.6f}; no_data_edge inside the 5070 bbox: {nd_in:.0f} m")
        if (abs(st["balance"] - 1) > 1e-3 or st["overlap_probes"] or st["split_pieces"]
                or st["unresolved_sides"] or st["lost_pieces"]):
            print("    WARNING: the polygons are not topologically clean here (a misfit "
                  "beyond --snap-m or an overlap). Soil lines at the misfit are "
                  "unpaired or doubled; inspect before using this component.")
        if nd_in > 0:
            print("    NOTE: one-sided edges inside the bbox: SSURGO has no polygon on their "
                  "far side (open water beyond a survey, or a gap). Expected only on a coast.")

    if not lines_all:
        print("  nothing to write")
        return
    polys = pd.concat(polys_all, ignore_index=True)
    lines = pd.concat(lines_all, ignore_index=True)
    lines = lines.sort_values(["component", "kind", "mupolygonkey_a", "mupolygonkey_b",
                               "length_m"]).reset_index(drop=True)
    lines.insert(0, "line_id", np.arange(len(lines), dtype=np.int64))
    cols = (["line_id", "component", "kind", "mupolygonkey_a", "mupolygonkey_b", "mukey_a",
             "mukey_b", "areasymbol_a", "areasymbol_b", "areaname_a", "areaname_b"]
            + [f"diff_{c}" for c in ATTRS] + ["n_diff", "diff_attrs", "length_m", "geometry"])
    lines = gpd.GeoDataFrame(lines[cols], geometry="geometry", crs=CRS_WORK)
    meta = dict(bboxes=bbox_record(p.bb), bboxes_file=str(p.path), tile_deg=a.tile_deg,
                snap_m=a.snap_m, probe_m=a.probe_m,
                created=dt.datetime.now().isoformat(timespec="seconds"))
    write_outputs([(polys, p.f_poly), (lines, p.f_line)], meta)
    summarize(polys, lines)
    print(f"  wrote {p.f_poly} ({len(polys):,} rows)\n  wrote {p.f_line} ({len(lines):,} rows)")


def summarize(polys: pd.DataFrame, lines: pd.DataFrame) -> None:
    print("  lines by kind (count, km):")
    for k in KINDS:
        m = lines.kind == k
        print(f"    {k:<13} {int(m.sum()):>6,} {lines.length_m[m].sum() / 1e3:>9.2f}")
    two = lines.kind.isin(["soil_line", "survey_join", "water_edge"])
    print("  two-sided lines on which each attribute differs "
          "(soil_line / survey_join / water_edge):")
    for c in ATTRS:
        s = lines[f"diff_{c}"].fillna(False).astype(bool)
        print(f"    {c:<15} " + " / ".join(f"{int((s & (lines.kind == k)).sum()):>6,}"
                                           for k in ("soil_line", "survey_join", "water_edge")))
    print(f"    no attribute differs: {int((two & (lines.n_diff == 0)).sum()):,} of "
          f"{int(two.sum()):,}")
    joins = lines[lines.kind == "survey_join"]
    if len(joins):
        print("  survey joins (km):")
        names = joins.areaname_a + "  |  " + joins.areaname_b
        for n, km in (joins.length_m.groupby(names).sum() / 1e3).sort_values(
                ascending=False).items():
            print(f"    {km:>9.2f}  {n}")


def run(a, sda: SDA) -> int:
    jobs, refused = [], []
    for path in a.bboxes:
        p = load_prefix(path, a.gis_dir)
        state, why = output_state(p)
        if a.overwrite or a.refetch:
            jobs.append(p)
        elif state == "same":
            print(f"{p.prefix}: outputs exist for exactly these bboxes, skipping "
                  f"(--overwrite to rebuild): {p.f_poly}, {p.f_line}")
        else:
            if state == "other":
                refused.append(f"{p.prefix}: {why}")
            jobs.append(p)
    if refused:
        msg = ("existing outputs were built for other bboxes (note 9):\n  "
               + "\n  ".join(refused) + "\n  --overwrite replaces them (the cache is kept).")
        if a.yes and not a.plan:
            print(f"REFUSED: {msg}", file=sys.stderr)
            return 2
        print(f"NOTE: --yes would refuse: {msg}")
    if not jobs:
        return 0

    stores = _stores(a)
    for p in jobs:
        plan_keys(p, a, sda)
    print(f"\nkey step: {sda.n_req} SDA requests, {sda.n_bytes / 1e6:.2f} MB", flush=True)
    need = pd.concat([p.need for p in jobs]).drop_duplicates("mupolygonkey")
    if need.empty:
        print("no SSURGO polygons in any envelope; nothing to do")
        return 0
    rc = size_gate(jobs, need, a, stores)
    if rc is not None:
        return rc

    cache = a.gis_dir / "ssurgo" / "cache"
    stores["geom"] = fetch_geometry(sda, need, stores["geom"], cache / "geom", a)
    stores["attr"] = fetch_table(sda, need.mukey.unique().tolist(), stores["attr"], "mukey",
                                 ATTR_SQL, ATTR_COLS, "attr", cache / "attr", quote=False)
    stores["sa"] = fetch_table(sda, need.areasymbol.unique().tolist(), stores["sa"],
                               "areasymbol", SA_SQL, SA_COLS, "sa", cache / "attr", quote=True)
    check_one_version(need, stores)
    for p in jobs:
        build_prefix(p, a, stores)
    return 0


# =========================================================================
# self-test
# =========================================================================
def selftest(a, sda: SDA) -> int:
    """A 2 km box straddling the Bradford (FL007) / Union (FL125) survey join,
    in the lobes bbox schema, cut into 0.01-degree tiles so dedupe is exercised."""
    to_work = Transformer.from_crs(CRS_GEO, CRS_WORK, always_xy=True)
    to_geo = Transformer.from_crs(CRS_WORK, CRS_GEO, always_xy=True)
    cx, cy = to_work.transform(SELFTEST["lon"], SELFTEST["lat"])
    h = SELFTEST["half_m"]
    ring = shapely.segmentize(shapely.box(cx - h, cy - h, cx + h, cy + h), 50.0)
    lon, lat = to_geo.transform(*shapely.get_coordinates(ring).T)
    bb = pd.DataFrame([dict(component=0, minx=cx - h, miny=cy - h, maxx=cx + h, maxy=cy + h,
                            lon_min=lon.min(), lat_min=lat.min(),
                            lon_max=lon.max(), lat_max=lat.max())])
    path = a.gis_dir / "ssurgo" / "selftest" / "SELFTEST_bboxes.parquet"
    _write(bb, path)
    a.bboxes = [path]
    a.tile_deg = SELFTEST["tile_deg"]
    a.overwrite = a.yes = True
    a.plan = False
    a.min_free_gb = 0.0                  # the self-test writes well under 1 MB
    rc = run(a, sda)
    if rc:
        return rc

    f_poly = a.gis_dir / "ssurgo" / "SELFTEST_mupolygons.parquet"
    f_line = a.gis_dir / "ssurgo" / "SELFTEST_mulines.parquet"
    polys, lines = gpd.read_parquet(f_poly), gpd.read_parquet(f_line)
    u = shapely.union_all(polys.geometry.to_numpy())
    checks = [
        ("polygons found", len(polys) > 0),
        ("mupolygonkey unique", polys.mupolygonkey.is_unique),
        ("no overlaps: union area == sum of areas (1e-6)",
         abs(shapely.area(u) / polys.area_m2.sum() - 1) < 1e-6),
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
         not lines[lines.kind == "clip_edge"].intersects(
             shapely.box(bb.minx[0], bb.miny[0], bb.maxx[0], bb.maxy[0]).buffer(-1.0)).any()),
        ("two-sided lines lie on both polygons' boundaries", _sides_touch(polys, lines)),
        ("both outputs store the input bboxes",
         stored_bboxes(f_poly) == stored_bboxes(f_line) == bbox_record(bb)),
    ]
    print("\nself-test:")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 0 if all(ok for _, ok in checks) else 1


def _two_sided(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return lines[lines.kind.isin(["soil_line", "survey_join", "water_edge"])]


def _sides_touch(polys: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> bool:
    two = _two_sided(lines)
    g = polys.set_index("mupolygonkey").geometry
    mid = shapely.line_interpolate_point(two.geometry.to_numpy(), 0.5, normalized=True)
    da = shapely.distance(mid, shapely.boundary(g.loc[two.mupolygonkey_a].to_numpy()))
    db = shapely.distance(mid, shapely.boundary(g.loc[two.mupolygonkey_b.astype("int64")].to_numpy()))
    return bool((np.maximum(da, db) < 0.5).all())


def _water_kinds(polys: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> bool:
    two = _two_sided(lines)
    wet = polys.set_index("mupolygonkey").water_unit
    w = (wet.loc[two.mupolygonkey_a].to_numpy()
         | wet.loc[two.mupolygonkey_b.astype("int64")].to_numpy())
    return bool(((two.kind == "water_edge").to_numpy() == w).all())


def _stores(a) -> dict:
    c = a.gis_dir / "ssurgo" / "cache"
    if a.refetch:
        return {"geom": None, "attr": None, "sa": None}
    return {"geom": _load_store(c / "geom", "g_*.parquet", "mupolygonkey",
                                GEOM_COLS[:-1] + ["wkb"]),
            "attr": _load_store(c / "attr", "attr_*.parquet", "mukey", ATTR_COLS),
            "sa": _load_store(c / "attr", "sa_*.parquet", "areasymbol", SA_COLS)}


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
                    help="--yes refuses a geometry download estimated above this (note 8)")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULTS["min_free_gb"],
                    help="--yes refuses when the run would leave less free disk (note 8)")
    ap.add_argument("--plan", action="store_true",
                    help="keys and the size estimate only, even with --yes")
    ap.add_argument("--overwrite", action="store_true", help="rebuild outputs (cache kept)")
    ap.add_argument("--refetch", action="store_true", help="ignore the cache, re-query SDA")
    ap.add_argument("--selftest", action="store_true",
                    help="2 km box at Bradford/Union (0.01 deg tiles), checks invariants")
    a = ap.parse_args(argv)
    if not (a.selftest or a.bboxes):
        ap.error("give --bboxes or --selftest")
    if not 0 < a.workers <= 4:
        ap.error("--workers must be 1-4 (SDA is a shared government server)")
    if a.probe_m < 2 * a.snap_m:
        ap.error("--probe-m must be at least 2 x --snap-m, or a probe can land on the "
                 "wrong side of a snapped line")

    t0 = time.time()
    sda = SDA(a.workers)
    try:
        rc = selftest(a, sda) if a.selftest else run(a, sda)
    except SDAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        rc = 1
    print(f"\nSDA: {sda.n_req} requests, {sda.n_bytes / 1e6:.2f} MB received; "
          f"{time.time() - t0:.1f}s total", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
