#!/usr/bin/env python3
"""
pull_hydro.py -- NHDPlus V2 + EPA StreamCat: the climate, soil and river covariates.

This closes the one categorical gap left in the feature set. The Earth Engine stack
covers terrain and land surface, but it has no rainfall, no soil, no drainage
density and no real distance to water -- and removing absolute elevation (which
alone identifies the state at 58.3% against a 10% baseline) left a climate-shaped
hole, because elevation was partly standing in for orography. A flood-susceptibility
model with no precipitation and no soil permeability is a gap a reviewer will find
before anything else.

WHAT EACH STAGE DOES
====================

--inspect     List the layers and fields in the extracted geodatabase. Run this
              FIRST. NHDPlus V2 names its catchment identifier `FEATUREID`, not
              `COMID`, and layer names have varied across releases, so nothing else
              here should run on an assumption about either.

--catchments  Per state, read only the catchments inside that state's point bounding
              box and spatially join points -> COMID. Bounding-box reads keep this
              off the 2.6M-polygon national layer; the same streaming discipline
              that got the NFHL subset working.

--flowlines   Per state, distance from each point to the nearest NHD flowline, plus
              drainage density (flowline km per km2) of its catchment. The distance
              is what `water_frac_900m` was standing in for: that band is the
              fraction of a 900 m neighbourhood JRC Global Surface Water has ever
              seen as water, so it saturates near large rivers and reads zero on any
              stream too small to detect -- and it is currently the LEADING
              discriminator between Zone A and Zone AE (median 0.000 vs 0.087). A
              real distance replaces a saturating proxy in the variable that matters
              most for the explanation.

--streamcat   Pull EPA StreamCat metrics. Queried by county FIPS, which the point
              data already carries for 99.86% of rows, so no COMID list has to be
              uploaded. Both `cat` (the point's own catchment) and `ws` (its whole
              upstream watershed) are fetched: the first is local conditions, the
              second is what arrives from upstream, and for flooding they are
              different questions.

--merge       Join everything onto data/processed/dataset_{ST}.parquet and rewrite
              FEATURES.txt.

A NOTE ON WHAT COUNTS AS A FEATURE
==================================
StreamCat metrics are catchment areal means, constant across a ~1-3 km2 catchment.
They are legitimate covariates -- catchment context genuinely drives flooding -- but
they are NOT point measurements, and a catchment is a spatial unit, so they can act
as a location fingerprint the way absolute elevation did. Folds are grouped on
`tile_50km` (50 km), which contains many catchments, so this is bounded rather than
eliminated. If a StreamCat variable turns out to dominate the A-vs-AE separability
the way `elev` did, quarantine it the same way.

USAGE
=====
    7z x data/raw/nhdplus/NHDPlusV21_*.7z -odata/raw/nhdplus     # ~20 GB
    python pull_hydro.py --inspect                  # confirm layers and fields
    python pull_hydro.py --catchments               # points -> COMID
    python pull_hydro.py --flowlines                # distance + drainage density
    python pull_hydro.py --streamcat                # EPA metrics by county
    python pull_hydro.py --merge                    # into dataset_{ST}.parquet
    python pull_hydro.py --catchments VT            # one state, for a dry run
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "ONLY_CCW")

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import geopandas as gpd
import pyogrio
import shapely
from shapely import STRtree

RAW = Path("data/raw/nhdplus")
PROC = Path("data/processed")
HYDRO = Path("data/interim/hydro")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]
CRS_GEO = 4269
CRS_WORK = 5070          # metres: distances and densities are computed here

# StreamCat metrics. `cat` is the point's own catchment, `ws` its upstream watershed.
SC_METRICS = ["precip8110", "tmean8110", "runoff", "wetindex", "bfi",
              "clay", "sand", "perm", "kffact", "hydrlcond", "elev"]
SC_URL = "https://api.epa.gov/StreamCat/streams/metrics"

# Confirmed against the extracted national geodatabase, 2026-09-19:
#   Catchment            2,647,454 features, EPSG:4269, id field FEATUREID
#   NHDFlowline_Network  2,691,339 features, EPSG:4269, id field COMID
CATCHMENT_LAYERS = ["Catchment", "CatchmentSP", "NHDPlusCatchment"]
FLOWLINE_LAYERS = ["NHDFlowline_Network", "NHDFlowline", "NHDFlowlineVAA"]
COMID_FIELDS = ["FEATUREID", "COMID", "ComID", "comid"]

# Fields worth taking off the flowline. The layer carries 157 of them -- most are
# monthly flow and velocity estimates -- so the subset matters for memory.
FLOW_FIELDS = ["COMID", "LENGTHKM", "StreamOrde", "TotDASqKM", "SLOPE", "FTYPE"]
CATCH_FIELDS = ["FEATUREID", "AreaSqKM"]

# FTYPE decides what "distance to river" means, and getting it wrong would have
# been invisible. NHDFlowline_Network includes Coastline features: the shoreline
# itself. Left in, every coastal point in Florida, Louisiana, North Carolina and
# Texas would read a distance to "river" of nearly zero, and since Zone AE contains
# 6,161 COASTAL FLOODPLAIN polygons the feature would have encoded coastal-versus-
# riverine directly into what is supposed to be a terrain covariate.
#
# So Coastline is excluded from the river distance and measured separately as
# `dist_to_coast_m`. That turns a contaminant into an honest feature: coastal and
# riverine flooding are different processes, and the A/AE contrast sits across that
# split. Pipelines are underground and excluded entirely.
RIVER_FTYPES = {"StreamRiver", "ArtificialPath", "Connector", "CanalDitch"}
COAST_FTYPES = {"Coastline"}


def session() -> requests.Session:
    s = requests.Session()
    # connect=6 is the parameter that survives the connection resets these
    # government endpoints hand out under sustained querying.
    r = Retry(total=6, connect=6, read=6, backoff_factor=1.5,
              status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET", "POST"]), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=8))
    return s


def find_gdb() -> str:
    hits = sorted(glob.glob(str(RAW / "**" / "*.gdb"), recursive=True))
    if not hits:
        sys.exit(
            f"No extracted .gdb under {RAW}.\n"
            f"    7z x {RAW}/NHDPlusV21_*.7z -o{RAW}\n"
            f"(~20 GB extracted. macOS cannot read .7z natively: brew install p7zip)")
    return hits[0]


def pick(layers: list[str], candidates: list[str]) -> str | None:
    low = {l.lower(): l for l in layers}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def pick_field(fields: list[str], candidates: list[str]) -> str | None:
    low = {f.lower(): f for f in fields}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


# =========================================================================
def inspect() -> int:
    gdb = find_gdb()
    print(f"gdb: {gdb}\n")
    layers = [l[0] for l in pyogrio.list_layers(gdb)]
    print(f"{len(layers)} layers:")
    for l in sorted(layers):
        print(f"  {l}")

    cat = pick(layers, CATCHMENT_LAYERS)
    flw = pick(layers, FLOWLINE_LAYERS)
    print(f"\ncatchment layer -> {cat}")
    print(f"flowline  layer -> {flw}")

    for name in (cat, flw):
        if not name:
            continue
        info = pyogrio.read_info(gdb, layer=name)
        print(f"\n--- {name} ---")
        print(f"  features : {info['features']:,}")
        print(f"  crs      : {info['crs']}")
        print(f"  geometry : {info['geometry_type']}")
        print(f"  fields   : {list(info['fields'])}")
        idf = pick_field(list(info["fields"]), COMID_FIELDS)
        print(f"  id field -> {idf}")

    print("\nIf the id field or either layer came back None, paste this and I will "
          "fix the candidate lists rather than you guessing.")
    return 0


def points_bbox(state: str) -> tuple:
    p = PROC / f"dataset_{state}.parquet"
    if not p.exists():
        return ()
    df = pd.read_parquet(p, columns=["lon", "lat"])
    pad = 0.05                     # ~5 km, so edge points still find a catchment
    return (df.lon.min() - pad, df.lat.min() - pad,
            df.lon.max() + pad, df.lat.max() + pad)


def catchments(states: list[str]) -> int:
    gdb = find_gdb()
    layers = [l[0] for l in pyogrio.list_layers(gdb)]
    lyr = pick(layers, CATCHMENT_LAYERS)
    if not lyr:
        sys.exit(f"no catchment layer among {layers[:20]} -- run --inspect")
    fields = list(pyogrio.read_info(gdb, layer=lyr)["fields"])
    idf = pick_field(fields, COMID_FIELDS)
    areaf = pick_field(fields, ["AreaSqKM"])
    if not idf:
        sys.exit(f"no COMID-like field in {lyr} -- run --inspect")
    cols = [c for c in (idf, areaf) if c]
    print(f"catchments from {lyr}, fields {cols}")

    HYDRO.mkdir(parents=True, exist_ok=True)
    for st in states:
        out = HYDRO / f"comid_{st}.parquet"
        if out.exists():
            print(f"[{st}] already joined -- skipping")
            continue
        bb = points_bbox(st)
        if not bb:
            print(f"[{st}] no dataset -- skipping")
            continue
        t0 = time.time()
        # Bounding-box read: never touch the 2.6M-polygon national extent.
        cat = gpd.read_file(gdb, layer=lyr, columns=cols, bbox=bb)
        if cat.crs is None:
            cat = cat.set_crs(CRS_GEO)
        cat = cat.to_crs(CRS_WORK)
        cat = cat[cat.geometry.notna() & ~cat.geometry.is_empty]
        bad = ~cat.geometry.is_valid
        if bad.any():
            cat.loc[bad, "geometry"] = cat.loc[bad, "geometry"].make_valid()
        # NHDPlus ships its own catchment area; use it rather than a reprojected
        # geometry area, so drainage density matches the published numbers.
        if areaf:
            cat["catch_area_km2"] = pd.to_numeric(cat[areaf], errors="coerce")
        else:
            cat["catch_area_km2"] = cat.geometry.area / 1e6

        pts = gpd.read_parquet(PROC / f"dataset_{st}.parquet",
                               columns=["point_id", "geometry"])
        j = gpd.sjoin(pts, cat[[idf, "catch_area_km2", "geometry"]],
                      how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")].reindex(pts.index)

        res = pd.DataFrame({"point_id": pts.point_id.values,
                            "comid": pd.to_numeric(j[idf], errors="coerce").values,
                            "catch_area_km2": j["catch_area_km2"].values})
        res.to_parquet(out, index=False)
        hit = 100 * res.comid.notna().mean()
        print(f"[{st}] {len(cat):,} catchments read, {len(res):,} points, "
              f"{hit:.1f}% matched a COMID, {res.comid.nunique():,} distinct "
              f"({time.time()-t0:.0f}s)", flush=True)
        if hit < 95:
            print(f"     NOTE only {hit:.1f}% matched. NHDPlus has no catchment "
                  f"over open water or tidal flats, so coastal states legitimately "
                  f"miss some; anything below ~85% is worth looking at.")
    return 0


def flowlines(states: list[str]) -> int:
    gdb = find_gdb()
    layers = [l[0] for l in pyogrio.list_layers(gdb)]
    lyr = pick(layers, FLOWLINE_LAYERS)
    if not lyr:
        sys.exit(f"no flowline layer among {layers[:20]} -- run --inspect")
    fields = list(pyogrio.read_info(gdb, layer=lyr)["fields"])
    cols = [f for f in FLOW_FIELDS if pick_field(fields, [f])]
    cols = [pick_field(fields, [f]) for f in cols]
    print(f"flowlines from {lyr}, fields {cols}")
    print("(the 'Measured (M) geometry' warning is expected and harmless -- the M")
    print(" values are river-mile measures and distances here are planar)")

    HYDRO.mkdir(parents=True, exist_ok=True)
    for st in states:
        out = HYDRO / f"river_{st}.parquet"
        if out.exists():
            print(f"[{st}] already computed -- skipping")
            continue
        bb = points_bbox(st)
        if not bb:
            print(f"[{st}] no dataset -- skipping")
            continue
        t0 = time.time()
        fl = gpd.read_file(gdb, layer=lyr, columns=cols, bbox=bb)
        if fl.crs is None:
            fl = fl.set_crs(CRS_GEO)
        fl = fl.to_crs(CRS_WORK)
        fl = fl[fl.geometry.notna() & ~fl.geometry.is_empty]
        if fl.empty:
            print(f"[{st}] no flowlines in bbox -- skipping")
            continue
        # The layer is MultiLineString Z; drop Z so nothing downstream has to
        # reason about whether a distance is planar or three-dimensional.
        fl["geometry"] = shapely.force_2d(fl.geometry.values)

        ft = fl["FTYPE"].astype(str) if "FTYPE" in fl.columns else pd.Series(
            "StreamRiver", index=fl.index)
        counts = ft.value_counts().to_dict()
        riv = fl[ft.isin(RIVER_FTYPES)]
        cst = fl[ft.isin(COAST_FTYPES)]
        print(f"[{st}] {len(fl):,} flowlines: {counts}")
        print(f"       {len(riv):,} river, {len(cst):,} coastline "
              f"(coastline measured separately, never as 'river')")
        if riv.empty:
            print(f"[{st}] no river-type flowlines -- skipping")
            continue

        pts = gpd.read_parquet(PROC / f"dataset_{st}.parquet",
                               columns=["point_id", "geometry"])
        P = shapely.points(pts.geometry.x.values, pts.geometry.y.values)

        G = np.asarray(riv.geometry.values)
        near = STRtree(G).nearest(P)
        res = pd.DataFrame({
            "point_id": pts.point_id.values,
            "dist_to_river_m": shapely.distance(P, G[near]),
        })
        # Attributes of the reach a point drains to. TotDASqKM is NHDPlus's total
        # upstream drainage area on the 1:100k network -- the honest version of what
        # MERIT's `upa` was standing in for, which read under one 90 m cell at two of
        # three test points. SLOPE is the channel gradient of that reach.
        for src, dst in (("StreamOrde", "nearest_stream_order"),
                         ("TotDASqKM", "upstream_area_km2"),
                         ("SLOPE", "channel_slope")):
            if src in riv.columns:
                v = pd.to_numeric(riv[src], errors="coerce").to_numpy()
                res[dst] = v[near]
        if "FTYPE" in riv.columns:
            res["meta_nearest_ftype"] = riv["FTYPE"].to_numpy()[near]

        if not cst.empty:
            C = np.asarray(cst.geometry.values)
            cn = STRtree(C).nearest(P)
            res["dist_to_coast_m"] = shapely.distance(P, C[cn])
        else:
            res["dist_to_coast_m"] = np.nan

        # Drainage density: in NHDPlus V2 each flowline pairs 1:1 with the catchment
        # of the same COMID, so reach length over catchment area is the standard
        # local density, using both published values rather than reprojected ones.
        cp = HYDRO / f"comid_{st}.parquet"
        if cp.exists() and "LENGTHKM" in fl.columns and "COMID" in fl.columns:
            cm = pd.read_parquet(cp)
            per = (fl.assign(_c=pd.to_numeric(fl["COMID"], errors="coerce"),
                             _l=pd.to_numeric(fl["LENGTHKM"], errors="coerce"))
                   .groupby("_c")._l.sum())
            res = res.merge(cm, on="point_id", how="left")
            res["drainage_density"] = (res.comid.map(per) /
                                       res.catch_area_km2.replace(0, np.nan))
            res = res.drop(columns=[c for c in ("comid", "catch_area_km2")
                                    if c in res.columns])

        res.to_parquet(out, index=False)
        d = res.dist_to_river_m
        print(f"       river distance median {np.median(d):,.0f} m, "
              f"p95 {np.percentile(d,95):,.0f} m; "
              f"coast distance available for "
              f"{100*res.dist_to_coast_m.notna().mean():.0f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
    return 0


def streamcat(states: list[str]) -> int:
    HYDRO.mkdir(parents=True, exist_ok=True)
    s = session()
    for st in states:
        out = HYDRO / f"streamcat_{st}.parquet"
        if out.exists():
            print(f"[{st}] already pulled -- skipping")
            continue
        p = PROC / f"dataset_{st}.parquet"
        if not p.exists():
            print(f"[{st}] no dataset -- skipping")
            continue
        fips = sorted(pd.read_parquet(p, columns=["county_fips"])
                      .county_fips.dropna().astype(str).unique())
        if not fips:
            print(f"[{st}] no county FIPS -- skipping")
            continue
        print(f"[{st}] {len(fips)} counties", flush=True)

        frames, failed = [], []
        t0 = time.time()
        for i, f in enumerate(fips, 1):
            # One county per request: the URL has an 8,192-character ceiling and a
            # single county already returns thousands of COMIDs.
            try:
                r = s.get(SC_URL, params={"name": ",".join(SC_METRICS),
                                          "areaOfInterest": "cat,ws",
                                          "county": f}, timeout=180)
                r.raise_for_status()
                items = r.json().get("items", [])
                if items:
                    frames.append(pd.DataFrame(items))
            except Exception as e:
                failed.append((f, str(e)[:90]))
            if i % 20 == 0 or i == len(fips):
                got = sum(len(x) for x in frames)
                print(f"    {i}/{len(fips)} counties, {got:,} COMIDs "
                      f"({time.time()-t0:.0f}s)", flush=True)

        if not frames:
            print(f"[{st}] StreamCat returned nothing")
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates("comid")
        df.columns = [c.lower() for c in df.columns]
        df.to_parquet(out, index=False)
        print(f"[{st}] {len(df):,} COMIDs, {df.shape[1]-1} metrics -> {out}")
        if failed:
            print(f"     {len(failed)} counties failed, e.g. {failed[:2]}")
            print(f"     rerun after deleting {out} to retry them")
    return 0


# =========================================================================
# Concurrency guard
# =========================================================================
# This script and pull_atlas14.py both finish by read-modify-writing the SAME
# dataset_{ST}.parquet files: each reads the whole file, adds its columns, writes
# the whole thing back. Two overlapping merges therefore mean the second write
# silently discards every column the first one added -- a failure that surfaces
# days later as "why is this feature missing".
#
# The FETCH stages are safe to run concurrently: they only read the datasets and
# write into their own directories. The MERGES are not. So an exclusive lockfile
# makes overlap fail loudly instead of quietly, and dataset writes go to a temp
# file renamed into place, which is atomic on POSIX -- a concurrent reader sees
# either the old file or the new one, never a half-written one.
LOCK = PROC / ".merge.lock"


class MergeLock:
    """Exclusive, cross-process, and released even if the merge raises."""

    def __enter__(self):
        PROC.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = (time.time() - LOCK.stat().st_mtime) / 60 if LOCK.exists() else 0.0
            sys.exit(
                f"\nAnother merge holds {LOCK} (age {age:.1f} min).\n"
                f"Merges rewrite the same dataset files, so they must not overlap.\n"
                f"Wait for it to finish. If you are certain nothing is running:\n"
                f"    rm {LOCK}\n")
        os.write(self.fd,
                 f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        LOCK.unlink(missing_ok=True)
        return False


def atomic_to_parquet(df, path) -> None:
    """Write to a temp file beside the target, then rename into place."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def merge(states: list[str]) -> int:
    with MergeLock():
        return _merge_body(states)


def _added_columns(st: str) -> set[str]:
    """Every column this merge will introduce for one state.

    Needed because the merge must be RE-RUNNABLE. The first version was not: a
    second run re-added `comid` beside the `meta_comid` the first run had renamed,
    pandas allowed the duplicate label, and the streamcat join then died on
    "The column label 'meta_comid' is not unique". Nothing was corrupted -- the
    atomic write meant the crash happened before any file was touched -- but the
    only way to recover was to know that. Now the merge drops whatever it is about
    to add before adding it, so running it twice is a no-op rather than a trap.

    The set is derived from the source files rather than hardcoded, because the
    last hardcoded list is exactly what silently dropped `channel_slope`.
    """
    import pyarrow.parquet as pq
    cols: set[str] = set()
    for name, key in ((f"comid_{st}", "point_id"),
                      (f"river_{st}", "point_id"),
                      (f"streamcat_{st}", "comid")):
        f = HYDRO / f"{name}.parquet"
        if f.exists():
            cols |= set(pq.read_schema(f).names) - {key}
    # plus the columns this function derives or renames into existence
    cols |= {"meta_comid", "log10_dist_river", "log10_dist_coast",
             "log10_upstream_area", "meta_elevcat", "meta_elevws"}
    cols |= {f"meta_{c}" for c in
             ("dist_to_river_m", "dist_to_coast_m", "upstream_area_km2")}
    return cols


def _merge_body(states: list[str]) -> int:
    feats_added: list[str] = []
    rows = []
    for st in states:
        p = PROC / f"dataset_{st}.parquet"
        if not p.exists():
            print(f"[{st}] no dataset -- skipping")
            continue
        d = gpd.read_parquet(p)
        n0 = len(d)

        # Idempotence: clear anything a previous run of THIS merge left behind, so
        # re-running replaces rather than duplicates. Columns from other scripts
        # (Earth Engine, Atlas 14) are untouched.
        stale = [c for c in _added_columns(st) if c in d.columns]
        if stale:
            d = d.drop(columns=stale)
        before_all = set(d.columns)

        cp = HYDRO / f"comid_{st}.parquet"
        if cp.exists():
            d = d.merge(pd.read_parquet(cp), on="point_id", how="left")
            d = d.rename(columns={"comid": "meta_comid"})

        rp = HYDRO / f"river_{st}.parquet"
        if rp.exists():
            d = d.merge(pd.read_parquet(rp), on="point_id", how="left")

        sp = HYDRO / f"streamcat_{st}.parquet"
        if sp.exists() and "meta_comid" in d.columns:
            sc = pd.read_parquet(sp).rename(columns={"comid": "meta_comid"})
            sc = sc.drop_duplicates("meta_comid")
            d = d.merge(sc, on="meta_comid", how="left")

        # StreamCat returns catchment and watershed mean ELEVATION, the same
        # variable quarantined from the Earth Engine stack for identifying the
        # state at 58.3% against a 10% baseline. Catchment-mean elevation varies
        # within a state and correlates with where detailed studies get done, so it
        # carries the identical risk and gets the identical treatment. (tmean8110
        # stays: mean annual temperature is near-constant within a state, so it
        # cannot separate A from AE in a within-state comparison -- but re-run the
        # elevation leakage check on it before trusting a result that leans on it.)
        for c in ("elevcat", "elevws"):
            if c in d.columns:
                d = d.rename(columns={c: f"meta_{c}"})

        # Distances and areas are heavy-tailed, so they go in on a log scale for
        # the same reason upstream area did: half the points sit in the first few
        # percent of the raw range and the tail otherwise carries everything.
        for raw, log in (("dist_to_river_m", "log10_dist_river"),
                         ("dist_to_coast_m", "log10_dist_coast"),
                         ("upstream_area_km2", "log10_upstream_area")):
            if raw in d.columns:
                v = pd.to_numeric(d[raw], errors="coerce")
                d[log] = np.log10(v.clip(lower=1.0 if raw.endswith("_m") else 0.01))
                d = d.rename(columns={raw: f"meta_{raw}"})

        # What actually landed, rather than a list that has to be kept in sync by
        # hand. The hand-kept version is what lost `channel_slope`: it was computed,
        # stored and 100% populated, but never reached FEATURES.txt, so no model
        # would ever have seen it.
        added = sorted(set(d.columns) - before_all)
        feature_cols = [c for c in added
                        if not c.startswith("meta_") and not c.endswith("_pctfull")]

        atomic_to_parquet(d, p)
        cov = {c: round(100 * d[c].notna().mean(), 1) for c in feature_cols}
        print(f"\n[{st}] {n0:,} rows, {len(added)} columns added "
              f"({len(feature_cols)} features, "
              f"{len(added)-len(feature_cols)} meta){'  [re-merge]' if stale else ''}")
        print(f"  coverage %: {cov}")
        rows.append(dict(state=st, features=len(feature_cols),
                         comid_matched=round(100 * d.meta_comid.notna().mean(), 1)
                         if "meta_comid" in d else np.nan,
                         coast=round(100 * d.log10_dist_coast.notna().mean(), 1)
                         if "log10_dist_coast" in d else np.nan))
        feats_added = feature_cols

    # rewrite the feature list, keeping provenance out of it
    if feats_added:
        base = [l.strip() for l in (PROC / "FEATURES.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        new = [f for f in feats_added if f not in base]
        allf = base + new
        atomic_write_text(
            "# columns a model may use. everything else in dataset_*.parquet is an\n"
            "# identifier or provenance metadata and must stay out.\n"
            "# split on tile_50km (or tile_100km), never on block_id or point level.\n"
            "# StreamCat metrics are catchment areal means, constant across a\n"
            "# ~1-3 km2 catchment -- catchment context, not point measurements.\n"
            + "\n".join(allf) + "\n", PROC / "FEATURES.txt")
        print(f"\nFEATURES.txt: {len(base)} -> {len(allf)} features")
        print(f"  added: {new if new else 'nothing new (already present)'}")

    if rows:
        print("\n" + pd.DataFrame(rows).to_string(index=False))
        print("\ncoast = % of points with a coastline within reach. Expect 0 for")
        print("landlocked states and well above 0 for FL, LA, NC, TX -- that is the")
        print("FTYPE split doing its job rather than being a no-op.")
        print("\nNext: python baselines.py   (and then --null again, since the")
        print("feature set changed and every number has to be re-earned)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--catchments", action="store_true")
    ap.add_argument("--flowlines", action="store_true")
    ap.add_argument("--streamcat", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)

    states = a.states or STATES
    bad = [s for s in states if s not in STATES]
    if bad:
        sys.exit(f"unknown states: {bad}")

    if a.inspect:
        return inspect()
    if a.catchments:
        return catchments(states)
    if a.flowlines:
        return flowlines(states)
    if a.streamcat:
        return streamcat(states)
    if a.merge:
        return merge(states)

    ap.print_help()
    print("\nOrder matters: --inspect, then --catchments, --flowlines, "
          "--streamcat, --merge")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
