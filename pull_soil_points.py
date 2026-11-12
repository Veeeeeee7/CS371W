#!/usr/bin/env python3
"""
pull_soil_points.py -- the two SSURGO soil predictors of Woznicki et al. (2019) at every point.

Woznicki et al. (2019, Table 2, p. 944) predict the floodplain from ten inputs; two come from the
county soil survey (SSURGO): the flooding frequency class (`fldfreq`) and the fluvial class
(`fluvclass`). This script looks up, for every point of a point table (--dataset: point_id, lon,
lat), the SSURGO map unit that contains it and codes the two predictors as the paper's Sources
paragraph and docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D4 fix them:

    soil_floodfreq  muaggatt.flodfreqdcd (flooding frequency, dominant condition) as an ordinal:
                    None 0, Very rare 1, Rare 2, Occasional 3, Frequent 4, Very frequent 5;
                    null -> NaN. "None" is a class (no flooding), not a missing value.
    soil_fluvclass  1 if the taxonomic subgroup (taxsubgrp) of the map unit's dominant component
                    contains "fluv" (any case), else 0: whether the area's main soil formed from
                    river deposits (binary; Woznicki's third class, water, is not used).
A point in no map unit, or in a water map unit (pull_ssurgo.WATER_MU on its name: a unit with no
soil to classify), gets NaN in both, also where a water unit carries a flood class.

Terms, in CS terms. SSURGO is a polygon map; each polygon belongs to one *map unit* (key mukey),
a mapped soil type, and a map unit is a mix of *components* (soil kinds), each a percentage of
the unit (comppct_r). The *dominant component* is the largest one; ties go to the one flagged as
a major component (majcompflag 'Yes'), then to the lowest cokey. A *survey area* (areasymbol, about
one county) is surveyed and versioned as a whole; sacatalog.saverest dates its current version.

Point to map unit: SDA_Get_Mukey_from_intersection_with_WktWgs84, applied with CROSS APPLY to a
VALUES table of (point_id, WKT point), a few hundred points per query (option (a) of the brief;
`--check-single N` compares it with one call per point, option (b)). A point exactly on an edge
between two map units returns both: it takes the smaller mukey, and n_mukeys counts them.

SDA is a shared government server: at most --workers (2) requests at a time, the retries and
backoff of pull_ssurgo.SDA, and every response cached under <soil-dir>/cache/, so a rerun resumes.

Stages:
    python pull_soil_points.py fetch --dataset data/interim/points/points_FL.parquet --check-single 20
    python pull_soil_points.py crosscheck --dataset data/interim/points/points_FL.parquet \
        --out results/$D/soil/crosscheck_FL.csv
    python pull_soil_points.py build --dataset data/interim/v4_build/dataset_FL.parquet \
        --out data/processed_v4                                  # FEATURES.txt + dataset_FL.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

from pull_ssurgo import (ATTR_BATCH, SA_COLS, SA_SQL, SDA, WATER_MU, SDAError, SDATooBig,
                         _digest, _load_store, _write, fetch_table)

PROCESSED = Path("data/processed")
SOIL_DIR = Path("data/interim/soil")
POLY_GLOB = "data/interim/gis/ssurgo/*_mupolygons.parquet"
META_KEY = b"pull_soil_points"
CRS_WORK = 5070         # CONUS Albers, metres: the CRS of the cached region-box polygons

# Flooding frequency classes in SSURGO order (Soil Survey Manual), coded as an ordinal.
FLOOD_CODES = {"None": 0, "Very rare": 1, "Rare": 2, "Occasional": 3, "Frequent": 4,
               "Very frequent": 5}
FLUVIAL = re.compile("fluv", re.I)
FLUVIAL_SOIL, NON_FLUVIAL = 1, 0
NEW_FEATURES = ["soil_floodfreq", "soil_fluvclass"]
# The nine model features of the paper's Sources paragraph, in its order (data/processed_v4).
V4_FEATURES = ["landcover", "soil_floodfreq", "soil_fluvclass", "slope", "elev_sd_r300", "hand",
               "wetindexcat", "log10_dist_river", "log10_dist_coast"]
V4_HEADER = [
    "# Model features (v4): the nine of the paper's Sources paragraph, in its order, as the",
    "# predictors of Woznicki et al. (2019), Table 2 (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md).",
    "# zoo.py one-hot encodes the first three (zoo.CATEGORICAL); the other six are numeric.",
    "#   land cover            -> landcover        (NLCD 2021 class code, 30 m; pull_gee.py)",
    "#   soil flood frequency  -> soil_floodfreq   (SSURGO muaggatt.flodfreqdcd, dominant condition;",
    "#       None 0, Very rare 1, Rare 2, Occasional 3, Frequent 4, Very frequent 5; pull_soil_points.py)",
    "#   fluvial class         -> soil_fluvclass   (dominant component's subgroup fluvial 1, not 0;",
    "#       NaN with soil_floodfreq in a water map unit or outside every map unit)",
    "#   slope                 -> slope            (degrees, 3DEP 10 m; pull_gee.py)",
    "#   terrain variability   -> elev_sd_r300     (SD of the 3DEP 10 m elevations within 300 m)",
    "#   height above drainage -> hand             (MERIT Hydro, m)",
    "#   topographic wetness   -> wetindexcat      (EPA StreamCat catchment mean, x100; pull_hydro.py)",
    "#   distance to river     -> log10_dist_river (log10 m to the nearest NHDPlus V2 river, floor 1 m)",
    "#   distance to coast     -> log10_dist_coast (log10 m to the NHDPlus V2 coastline, floor 1 m)",
]

POINT_COLS = ["point_id", "mukey"]
POINT_SQL = """SELECT P.point_id, M.mukey
FROM (VALUES {values}) AS P(point_id, wkt)
CROSS APPLY SDA_Get_Mukey_from_intersection_with_WktWgs84(P.wkt) AS M"""
SINGLE_SQL = "SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}')"
POINT_ID = re.compile(r"^[A-Za-z0-9_]+$")   # safe to put between SQL quotes

# The map unit's name, survey area and flooding class (LEFT JOIN: a unit without muaggatt keeps a
# null class rather than vanishing).
MU_COLS = ["mukey", "muname", "areasymbol", "flodfreqdcd"]
MU_SQL = """SELECT MU.mukey, MU.muname, L.areasymbol, A.flodfreqdcd
FROM mapunit AS MU INNER JOIN legend AS L ON L.lkey = MU.lkey
LEFT JOIN muaggatt AS A ON A.mukey = MU.mukey
WHERE MU.mukey IN ({keys})"""
COMP_COLS = ["mukey", "cokey", "compname", "compkind", "comppct_r", "majcompflag", "taxsubgrp"]
COMP_SQL = "SELECT " + ", ".join(COMP_COLS) + " FROM component WHERE mukey IN ({keys})"

OUT_COLS = ["point_id", "mukey", "n_mukeys", "areasymbol", "saverest", "muname", "flodfreqdcd",
            "dom_cokey", "dom_compname", "dom_comppct_r", "dom_taxsubgrp", "water_unit",
            "soil_floodfreq", "soil_fluvclass"]


# =========================================================================
# coding (pure functions)
# =========================================================================
def flood_code(values: pd.Series) -> np.ndarray:
    """flodfreqdcd as the ordinal of FLOOD_CODES, null as NaN. Any other value raises
    ValueError (a stop of the brief: a class outside the six)."""
    v = pd.Series(values, dtype=object)
    known = v.isna() | v.isin(list(FLOOD_CODES))
    if not known.all():
        raise ValueError(f"flooding frequency classes outside {list(FLOOD_CODES)}: "
                         f"{sorted(set(v[~known].astype(str)))}")
    return v.map(FLOOD_CODES).astype(float).to_numpy()


def dominant_components(comp: pd.DataFrame) -> pd.DataFrame:
    """One row per mukey: the component with the highest comppct_r (null lowest), ties broken by
    majcompflag 'Yes', then by the lowest cokey."""
    c = comp.copy()
    c["_pct"] = pd.to_numeric(c.comppct_r, errors="coerce").fillna(-1.0)
    c["_major"] = c.majcompflag.fillna("").str.strip().str.lower().eq("yes")
    c["_cokey"] = c.cokey.astype("int64")
    c = c.sort_values(["mukey", "_pct", "_major", "_cokey"],
                      ascending=[True, False, False, True], kind="mergesort")
    return c.drop_duplicates("mukey", keep="first").drop(columns=["_pct", "_major", "_cokey"])


def fluvial_code(taxsubgrp: pd.Series, water_unit: pd.Series, has_unit: pd.Series) -> np.ndarray:
    """Binary: 1 if the dominant component's subgroup contains "fluv" (any case), 0 if not (a
    null subgroup, as on miscellaneous areas such as "Urban land", is not fluvial); NaN where
    there is no map unit or the map unit is water (no soil to classify)."""
    tax = pd.Series(taxsubgrp, dtype=object).fillna("").astype(str)
    fluv = tax.str.contains(FLUVIAL).to_numpy()
    code = np.where(fluv, FLUVIAL_SOIL, NON_FLUVIAL).astype(float)
    water = pd.Series(water_unit).fillna(False).to_numpy(bool)
    code[water | ~pd.Series(has_unit).to_numpy(bool)] = np.nan
    return code


def resolve_mukeys(hits: pd.DataFrame, point_ids) -> pd.DataFrame:
    """One row per point, in the order of point_ids: the smallest mukey SDA returned for it
    (Int64, NA when none) and n_mukeys, the number of distinct map units (2+ = on an edge)."""
    h = hits.astype({"mukey": "int64"}).drop_duplicates(["point_id", "mukey"])
    g = h.groupby("point_id").mukey.agg(["min", "nunique"])
    out = pd.DataFrame({"point_id": list(point_ids)})
    out["mukey"] = out.point_id.map(g["min"]).astype("Int64")
    out["n_mukeys"] = out.point_id.map(g["nunique"]).fillna(0).astype("int64")
    return out


def code_points(resolved: pd.DataFrame, units: pd.DataFrame, comps: pd.DataFrame,
                catalog: pd.DataFrame) -> pd.DataFrame:
    """The output table (OUT_COLS), one row per resolved point, in its order."""
    dom = dominant_components(comps).rename(columns={
        "cokey": "dom_cokey", "compname": "dom_compname", "comppct_r": "dom_comppct_r",
        "taxsubgrp": "dom_taxsubgrp"})[["mukey", "dom_cokey", "dom_compname", "dom_comppct_r",
                                        "dom_taxsubgrp"]]
    u = units.astype({"mukey": "int64"}).drop_duplicates("mukey")
    out = resolved.merge(u, on="mukey", how="left", validate="many_to_one")
    out = out.merge(dom.astype({"mukey": "int64"}), on="mukey", how="left",
                    validate="many_to_one")
    out = out.merge(catalog[["areasymbol", "saverest"]].drop_duplicates("areasymbol"),
                    on="areasymbol", how="left", validate="many_to_one")
    has_unit = out.mukey.notna()
    missing = has_unit & out.muname.isna()
    if missing.any():
        raise SDAError(f"{int(missing.sum())} points have a mukey with no mapunit row "
                       f"(e.g. {out.mukey[missing].unique()[:5].tolist()})")
    out["water_unit"] = out.muname.fillna("").str.match(WATER_MU) & has_unit
    flood = np.array(flood_code(out.flodfreqdcd), dtype=float)   # a writable copy
    flood[out.water_unit.to_numpy(bool)] = np.nan       # a water unit has no soil, even if rated
    out["soil_floodfreq"] = flood
    out["soil_fluvclass"] = fluvial_code(out.dom_taxsubgrp, out.water_unit, has_unit)
    out["dom_comppct_r"] = pd.to_numeric(out.dom_comppct_r, errors="coerce")
    assert out.point_id.tolist() == resolved.point_id.tolist()
    return out[OUT_COLS]


# =========================================================================
# point -> mukey on SDA
# =========================================================================
def wkt_point(lon: float, lat: float) -> str:
    """A WGS84 point as SDA's WKT (x = longitude), 9 decimals (about 0.1 mm)."""
    return f"POINT({lon:.9f} {lat:.9f})"


def values_sql(batch: pd.DataFrame) -> str:
    """The VALUES rows (point_id, WKT) of a batch of points."""
    bad = ~batch.point_id.astype(str).str.match(POINT_ID)
    if bad.any():
        raise ValueError(f"point_id not safe for SQL: {batch.point_id[bad].tolist()[:3]}")
    return ",".join(f"('{p}','{wkt_point(x, y)}')"
                    for p, x, y in zip(batch.point_id, batch.lon, batch.lat))


def lookup_batch(sda: SDA, batch: pd.DataFrame) -> pd.DataFrame:
    """Every (point_id, mukey) SDA returns for the batch, halving it while SDA calls it too big."""
    try:
        return sda.query(POINT_SQL.format(values=values_sql(batch)), POINT_COLS)
    except SDATooBig as e:
        if len(batch) == 1:
            raise SDAError(f"point {batch.point_id.iloc[0]} alone is too big for SDA: {e}") from e
        half = len(batch) // 2
        print(f"    batch of {len(batch)} rejected ({e}); halving", flush=True)
        return pd.concat([lookup_batch(sda, batch.iloc[:half]),
                          lookup_batch(sda, batch.iloc[half:])], ignore_index=True)


def lookup_points(sda: SDA, points: pd.DataFrame, cache: Path, size: int, workers: int,
                  refetch: bool) -> pd.DataFrame:
    """(point_id, mukey) hits for every point, batch by batch, each batch cached under a name
    that hashes its point ids and coordinates."""
    batches = [points.iloc[i:i + size] for i in range(0, len(points), size)]
    names = [cache / f"pt_{_digest([values_sql(b)])}.parquet" for b in batches]
    todo = [i for i, f in enumerate(names) if refetch or not f.exists()]
    print(f"  point lookup: {len(points):,} points in {len(batches)} batches of {size}, "
          f"{len(batches) - len(todo)} cached, {len(todo)} to query", flush=True)
    t0 = time.time()

    def query_and_cache(i):
        df = lookup_batch(sda, batches[i]).astype({"mukey": "int64"})
        _write(df, names[i])
        return i

    with ThreadPoolExecutor(workers) as ex:
        for done, _ in enumerate(ex.map(query_and_cache, todo), 1):
            if done % 10 == 0 or done == len(todo):
                rate = (time.time() - t0) / done
                print(f"    {done}/{len(todo)} batches, {rate:.1f} s per batch, "
                      f"~{rate * (len(todo) - done) / 60:.1f} min left", flush=True)
    parts = [pd.read_parquet(f) for f in names]
    return pd.concat(parts, ignore_index=True)


def check_single(sda: SDA, points: pd.DataFrame, n: int, seed: int = 0) -> dict:
    """Option (a) against option (b) on n random points: the same smallest mukey and the same
    number of map units for every point, and the time each took."""
    sample = points.sample(n, random_state=seed)
    t0 = time.time()
    a = resolve_mukeys(lookup_batch(sda, sample), sample.point_id)
    t_a = time.time() - t0
    rows = []
    t0 = time.time()
    for p, x, y in zip(sample.point_id, sample.lon, sample.lat):
        df = sda.query(SINGLE_SQL.format(wkt=wkt_point(x, y)), ["mukey"])
        rows += [(p, int(m)) for m in df.mukey]
    t_b = time.time() - t0
    b = resolve_mukeys(pd.DataFrame(rows, columns=POINT_COLS), sample.point_id)
    same = (a.mukey.fillna(-1).to_numpy() == b.mukey.fillna(-1).to_numpy()) & \
           (a.n_mukeys.to_numpy() == b.n_mukeys.to_numpy())
    return {"n": n, "agree": int(same.sum()), "batched_s": round(t_a, 2),
            "single_s": round(t_b, 2)}


def fetch_components(sda: SDA, mukeys: list[int], cache: Path, refetch: bool) -> pd.DataFrame:
    """The component rows of every mukey (several per unit), fetched ATTR_BATCH keys per query and
    cached per batch."""
    keys = sorted(set(int(k) for k in mukeys))
    parts = []
    for i in range(0, len(keys), ATTR_BATCH):
        chunk = keys[i:i + ATTR_BATCH]
        f = cache / f"comp_{_digest(chunk)}.parquet"
        if f.exists() and not refetch:
            parts.append(pd.read_parquet(f))
            continue
        df = sda.query(COMP_SQL.format(keys=",".join(map(str, chunk))), COMP_COLS)
        df["mukey"] = df.mukey.astype("int64")
        _write(df, f)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=COMP_COLS)
    return pd.concat(parts, ignore_index=True)


def fetch(args) -> int:
    """Stage fetch: look up and code every point; write <soil-dir>/soil_points_<ST>.parquet."""
    dataset = args.dataset or PROCESSED / f"dataset_{args.state}.parquet"
    out = args.soil_dir / f"soil_points_{args.state}.parquet"
    cache = args.soil_dir / "cache"
    points = pd.read_parquet(dataset, columns=["point_id", "lon", "lat"])
    if not points.point_id.is_unique:
        raise SystemExit(f"{dataset}: point_id is not unique")
    if args.limit:
        points = points.iloc[:args.limit]
    sda = SDA(args.workers)
    t_start = time.time()

    meta = {"dataset": str(dataset), "n_points": len(points), "batch": args.batch,
            "workers": args.workers, "fetched": dt.date.today().isoformat(),
            "function": "SDA_Get_Mukey_from_intersection_with_WktWgs84 (CROSS APPLY)"}
    if args.check_single:
        res = check_single(sda, points, args.check_single)
        print(f"  check (a) batched vs (b) one call per point on {res['n']} points: "
              f"{res['agree']} agree; {res['batched_s']} s vs {res['single_s']} s", flush=True)
        meta["check_single"] = res
        if res["agree"] != res["n"]:
            raise SystemExit("the batched lookup disagrees with the one-point lookup")

    hits = lookup_points(sda, points, cache, args.batch, args.workers, args.refetch)
    resolved = resolve_mukeys(hits, points.point_id)
    t_lookup = time.time() - t_start
    mukeys = sorted(resolved.mukey.dropna().astype(int).unique())
    print(f"  {resolved.mukey.notna().sum():,} points in a map unit, "
          f"{int((resolved.n_mukeys > 1).sum())} on an edge; {len(mukeys):,} map units", flush=True)

    store = None if args.refetch else _load_store(cache, "mu_*.parquet", "mukey", MU_COLS)
    units = fetch_table(sda, mukeys, store, "mukey", MU_SQL, MU_COLS, "mu", cache, quote=False)
    comps = fetch_components(sda, mukeys, cache, args.refetch)
    areas = sorted(units[units.mukey.isin(mukeys)].areasymbol.dropna().unique())
    store = None if args.refetch else _load_store(cache, "sa_*.parquet", "areasymbol", SA_COLS)
    catalog = fetch_table(sda, areas, store, "areasymbol", SA_SQL, SA_COLS, "sa", cache,
                          quote=True)
    catalog = catalog[catalog.areasymbol.isin(areas)]
    if set(areas) - set(catalog.areasymbol):
        raise SDAError(f"survey areas without a sacatalog row: "
                       f"{sorted(set(areas) - set(catalog.areasymbol))}")

    table = code_points(resolved, units[units.mukey.isin(mukeys)], comps, catalog)
    meta.update({"seconds": round(time.time() - t_start, 1), "lookup_seconds": round(t_lookup, 1),
                 "sda_requests": sda.n_req, "sda_bytes": sda.n_bytes,
                 "saverest": dict(zip(catalog.areasymbol, catalog.saverest)),
                 "areaname": dict(zip(catalog.areasymbol, catalog.areaname))})
    pt = pa.Table.from_pandas(table, preserve_index=False)
    pt = pt.replace_schema_metadata({**(pt.schema.metadata or {}),
                                     META_KEY: json.dumps(meta).encode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(pt, tmp)
    tmp.replace(out)
    print(summary(table, meta), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


def summary(t: pd.DataFrame, meta: dict) -> str:
    """The counts the brief's stops read, as text."""
    n = len(t)
    land = t.mukey.notna() & ~t.water_unit
    saved = pd.to_datetime(pd.Series(list(meta["saverest"].values())), format="%m/%d/%Y %I:%M:%S %p")
    lines = [f"points {n:,}; no map unit {int(t.mukey.isna().sum()):,} "
             f"({t.mukey.isna().mean():.2%}); on an edge {int((t.n_mukeys > 1).sum())}; "
             f"water units {int(t.water_unit.sum()):,}",
             "flodfreqdcd: " + ", ".join(f"{k} {v:,}" for k, v in
                                         t.flodfreqdcd.fillna("<null>").value_counts().items()),
             f"non-water points with a null flood class: {int(t.flodfreqdcd[land].isna().sum()):,} "
             f"of {int(land.sum()):,} ({t.flodfreqdcd[land].isna().mean():.2%})",
             "soil_fluvclass: " + ", ".join(f"{k} {v:,}" for k, v in
                                            t.soil_fluvclass.value_counts(dropna=False).items()),
             f"survey areas {len(meta['saverest'])}; saverest from "
             f"{saved.min():%Y-%m-%d} to {saved.max():%Y-%m-%d}; "
             f"{meta['sda_requests']} SDA requests, {meta['sda_bytes'] / 1e6:.1f} MB, "
             f"{meta['seconds']} s"]
    return "\n".join(lines)


# =========================================================================
# cross-check against the cached region-box polygons
# =========================================================================
def local_mukeys(points: pd.DataFrame, polys: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-polygon on the cached EPSG:5070 polygons: for every point inside at least one,
    the smallest mukey of the polygons that contain or touch it, and how many distinct mukeys."""
    to_work = Transformer.from_crs(4326, CRS_WORK, always_xy=True)
    x, y = to_work.transform(points.lon.to_numpy(), points.lat.to_numpy())
    geoms = shapely.points(x, y)
    tree = shapely.STRtree(polys.geometry.to_numpy())
    ip, jp = tree.query(geoms, predicate="intersects")
    hits = pd.DataFrame({"point_id": points.point_id.to_numpy()[ip],
                         "mukey": polys.mukey.to_numpy()[jp]})
    inside = points.point_id[points.point_id.isin(hits.point_id)]
    return resolve_mukeys(hits, inside)


def crosscheck(args) -> int:
    """Stage crosscheck: the SDA mukey against a local point-in-polygon on the cached polygons."""
    soil = pd.read_parquet(args.soil_dir / f"soil_points_{args.state}.parquet")
    dataset = args.dataset or PROCESSED / f"dataset_{args.state}.parquet"
    points = pd.read_parquet(dataset, columns=["point_id", "lon", "lat"])
    files = sorted(Path().glob(args.polygons))
    if not files:
        raise SystemExit(f"no polygons match {args.polygons}")
    cols = ["mupolygonkey", "mukey", "areasymbol", "saverest", "geometry"]
    polys = pd.concat([gpd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    polys = gpd.GeoDataFrame(polys.drop_duplicates("mupolygonkey"), crs=CRS_WORK)
    loc = local_mukeys(points, polys)
    m = loc.merge(soil[["point_id", "mukey", "n_mukeys", "areasymbol", "saverest", "muname"]],
                  on="point_id", suffixes=("_local", "_sda"))
    m["agree"] = m.mukey_local.astype("Int64").eq(m.mukey_sda.astype("Int64")).fillna(False)
    share = m.agree.mean()
    local_ver = polys.drop_duplicates("areasymbol").set_index("areasymbol").saverest
    m["saverest_local"] = m.areasymbol.map(local_ver)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m[~m.agree].to_csv(args.out, index=False)
    print(f"cached polygons: {len(polys):,} from {len(files)} files; points inside: {len(m):,}")
    print(f"same mukey: {int(m.agree.sum()):,} of {len(m):,} ({share:.2%}); "
          f"mismatches written to {args.out}")
    if (~m.agree).any():
        print(m[~m.agree].head(20).to_string(index=False))
    print("cached versions: " + ", ".join(f"{a} {v}" for a, v in local_ver.items()))
    return 0 if share >= 0.99 else 1


# =========================================================================
# build data/processed_v4
# =========================================================================
def features_text(columns) -> str:
    """FEATURES.txt for v4: V4_HEADER, then the nine V4_FEATURES, which the dataset (columns)
    must all hold."""
    missing = [f for f in V4_FEATURES if f not in set(columns)]
    if missing:
        raise ValueError(f"the dataset lacks the model features {missing}")
    return "\n".join(V4_HEADER + V4_FEATURES) + "\n"


def with_soil(table: pa.Table, soil: pd.DataFrame) -> pa.Table:
    """The dataset table with the two soil columns appended, row for row by point_id. Every
    existing column, the row order and the schema metadata stay as they are; the pandas metadata
    gains the two columns."""
    if set(NEW_FEATURES) & set(table.column_names):
        raise ValueError("the dataset already has soil columns")
    pid = table.column("point_id").to_pandas()
    s = soil.drop_duplicates("point_id").set_index("point_id")
    if len(s) != len(soil) or set(s.index) != set(pid) or len(pid) != pid.nunique():
        raise ValueError("the soil table and the dataset do not hold the same unique point_ids")
    out = table
    meta = dict(table.schema.metadata or {})
    pmeta = json.loads(meta[b"pandas"]) if b"pandas" in meta else None
    for c in NEW_FEATURES:
        values = s[c].reindex(pid).to_numpy(dtype="float64")
        out = out.append_column(pa.field(c, pa.float64()), pa.array(values, mask=np.isnan(values)))
        if pmeta is not None:
            pmeta["columns"].append({"name": c, "field_name": c, "pandas_type": "float64",
                                     "numpy_type": "float64", "metadata": None})
    if pmeta is not None:
        meta[b"pandas"] = json.dumps(pmeta).encode()
    out = out.replace_schema_metadata(meta)
    assert out.select(table.column_names).equals(table)
    return out


def build(args) -> int:
    """Stage build: <out>/FEATURES.txt and <out>/dataset_<ST>.parquet (a new file, not a link)."""
    dataset = args.dataset or PROCESSED / f"dataset_{args.state}.parquet"
    soil_path = args.soil_dir / f"soil_points_{args.state}.parquet"
    if args.out.resolve() == PROCESSED.resolve() or args.out.resolve() == dataset.parent.resolve():
        raise SystemExit(f"refusing to write into {args.out}: it holds the input or the "
                         f"45-feature dataset")
    soil = pd.read_parquet(soil_path)
    table = pq.read_table(dataset)
    out = with_soil(table, soil)
    smeta = pq.read_schema(soil_path).metadata or {}
    meta = dict(out.schema.metadata)
    meta[META_KEY] = json.dumps({"soil": str(soil_path), "dataset": str(dataset),
                                 "soil_meta": json.loads(smeta.get(META_KEY, b"{}"))}).encode()
    out = out.replace_schema_metadata(meta)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"dataset_{args.state}.parquet"
    if target.is_symlink():
        raise SystemExit(f"{target} is a link; remove it first")
    pq.write_table(out, target)
    (args.out / "FEATURES.txt").write_text(features_text(out.column_names))

    back = pd.read_parquet(target)
    orig = pd.read_parquet(dataset)
    assert len(back) == len(orig) == len(soil)
    assert back.point_id.tolist() == orig.point_id.tolist()
    assert back[orig.columns].equals(orig), "a column of the dataset changed"
    print(f"wrote {target}: {len(back):,} rows, {back.shape[1]} columns "
          f"(+{', '.join(NEW_FEATURES)}); every other column equal to {dataset}")
    print(f"wrote {args.out / 'FEATURES.txt'}")
    for c in NEW_FEATURES:
        print(f"  {c}: NaN {back[c].isna().mean():.2%}; "
              + ", ".join(f"{k:g} {v:,}" for k, v in back[c].value_counts().sort_index().items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("fetch", "crosscheck", "build"):
        p = sub.add_parser(name)
        p.add_argument("--state", default="FL")
        p.add_argument("--dataset", type=Path, help="default data/processed/dataset_<ST>.parquet")
        p.add_argument("--soil-dir", type=Path, default=SOIL_DIR)
    f = sub.choices["fetch"]
    f.add_argument("--batch", type=int, default=500, help="points per SDA query")
    f.add_argument("--workers", type=int, default=2, help="concurrent SDA requests (at most 2)")
    f.add_argument("--check-single", type=int, default=0,
                   help="first compare the batched lookup with one call per point on N points")
    f.add_argument("--limit", type=int, default=0, help="only the first N points (a dry run)")
    f.add_argument("--refetch", action="store_true", help="ignore the cache")
    c = sub.choices["crosscheck"]
    c.add_argument("--polygons", default=POLY_GLOB)
    c.add_argument("--out", type=Path, required=True, help="CSV of the mismatches")
    b = sub.choices["build"]
    b.add_argument("--out", type=Path, required=True, help="e.g. data/processed_v4")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "fetch" and not 1 <= args.workers <= 2:
        raise SystemExit("SDA is a shared server: --workers must be 1 or 2")
    return {"fetch": fetch, "crosscheck": crosscheck, "build": build}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
