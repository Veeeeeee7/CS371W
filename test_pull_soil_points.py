#!/usr/bin/env python3
"""
Checks for pull_soil_points.py (the two SSURGO soil predictors, coded as
docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md D4 fixes them for data/processed_v4). Run from the repo
root:

    .venv/bin/python test_pull_soil_points.py

Synthetic rows with known answers, no network (a fake SDA answers the queries):
1. flood_code: each of the six classes, null -> NaN, any other class raises.
2. dominant_components: the highest comppct_r; a tie goes to majcompflag 'Yes', then to the lowest
   cokey; a null percentage ranks last.
3. fluvial_code (D4, binary): "Typic Fluvaquents" and "Fluvaquentic Endoaquepts" are fluvial 1
   (any case), other subgroups and a null subgroup 0; a water unit and no map unit are NaN.
4. resolve_mukeys: a point on an edge between two map units takes the smaller mukey (n_mukeys 2),
   a repeated mukey counts once, a point with no hit gets NA and 0.
5. code_points end to end: the water rule of pull_ssurgo.WATER_MU on the name ("Water", "Waters of
   the Gulf", "Dredge channel" are water; "Arents-Water complex" is not); a water unit is NaN in
   both soil features, also when it carries a flood class (D4); the survey versions.
6. The SDA side: the VALUES rows and their WKT, an unsafe point_id refused, a batch that SDA calls
   too big halved until it passes, the per-batch cache reused, the batched and one-point lookups
   compared.
7. local_mukeys: point-in-polygon on two squares that share an edge (inside, on the edge, outside).
8. The v4 folder: features_text (the nine features of the paper's Sources paragraph in its order,
   all present in the table) and with_soil (rows, order, every old column and the schema metadata
   unchanged; ids must match).
9. Real files (skipped when missing): data/interim/soil/soil_points_FL.parquet and
   data/processed_v4 against its input, data/interim/v4_build.
"""
import re
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

import pull_soil_points as sp
from pull_ssurgo import SDATooBig

FAILS = []


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def check_flood_code():
    print("1. flood_code")
    got = sp.flood_code(pd.Series(["None", "Very rare", "Rare", "Occasional", "Frequent",
                                   "Very frequent", None]))
    check(np.array_equal(got[:6], [0, 1, 2, 3, 4, 5]) and np.isnan(got[6]),
          "None 0 ... Very frequent 5, null NaN")
    check(raises(lambda: sp.flood_code(pd.Series(["None", "Common"])), ValueError),
          "a class outside the six raises")
    check(raises(lambda: sp.flood_code(pd.Series(["none"])), ValueError),
          "classes are matched exactly ('none' is not 'None')")


def comp_rows(rows):
    return pd.DataFrame(rows, columns=sp.COMP_COLS)


def check_dominant():
    print("2. dominant_components")
    comps = comp_rows([
        (1, "11", "A", "Series", "60", "Yes", "Typic Haplaquods"),
        (1, "12", "B", "Series", "40", "No", "Typic Fluvaquents"),
        # mukey 2: a 45/45 tie, broken by majcompflag
        (2, "21", "C", "Series", "45", "No", "Typic Fluvaquents"),
        (2, "22", "D", "Series", "45", "Yes", "Arenic Glossaqualfs"),
        (2, "23", "E", "Inclusion", "10", "No", None),
        # mukey 3: a full tie, broken by the lowest cokey (as a number: 9 < 10)
        (3, "10", "F", "Series", "50", "Yes", "Typic Fluvaquents"),
        (3, "9", "G", "Series", "50", "Yes", "Aeric Ochraqualfs"),
        # mukey 4: a null percentage ranks last
        (4, "41", "H", "Series", None, "Yes", "Typic Fluvaquents"),
        (4, "42", "I", "Series", "5", "No", "Aeric Ochraqualfs"),
    ])
    dom = sp.dominant_components(comps).set_index("mukey").cokey.to_dict()
    check(dom == {1: "11", 2: "22", 3: "9", 4: "42"},
          f"highest share, then majcompflag, then lowest cokey, null last: {dom}")


def check_fluvial():
    print("3. fluvial_code")
    tax = pd.Series(["Typic Fluvaquents", "Fluvaquentic Endoaquepts", "AQUIC UDIFLUVENTS",
                     "Typic Haplaquods", None, "Typic Fluvaquents", None])
    water = pd.Series([False, False, False, False, False, True, False])
    has = pd.Series([True, True, True, True, True, True, False])
    got = sp.fluvial_code(tax, water, has)
    check(np.array_equal(got[:5], [1, 1, 1, 0, 0]) and np.isnan(got[5:]).all(),
          f"binary: fluvial 1 (any case), not 0, null subgroup 0; a water unit (even a fluvial "
          f"subgroup) and no map unit NaN: {got.tolist()}")
    check(set(np.unique(got[np.isfinite(got)])) <= {0.0, 1.0}, "no third class")


def check_resolve():
    print("4. resolve_mukeys")
    hits = pd.DataFrame({"point_id": ["p1", "p2", "p2", "p3", "p3"],
                         "mukey": [7, 9, 4, 5, 5]})
    r = sp.resolve_mukeys(hits, ["p0", "p1", "p2", "p3"])
    check(r.point_id.tolist() == ["p0", "p1", "p2", "p3"]
          and r.mukey.tolist()[1:] == [7, 4, 5] and pd.isna(r.mukey.iloc[0])
          and r.n_mukeys.tolist() == [0, 1, 2, 1],
          "edge point: the smaller mukey, n_mukeys 2; a repeat counts once; no hit NA, 0")


def synthetic_tables():
    resolved = pd.DataFrame({"point_id": ["a", "b", "c", "d", "e", "f"],
                             "mukey": pd.array([1, 2, 3, 4, 5, None], dtype="Int64"),
                             "n_mukeys": [1, 2, 1, 1, 1, 0]})
    units = pd.DataFrame({"mukey": [1, 2, 3, 4, 5],
                          "muname": ["Pomona fine sand", "Waters of the Gulf", "Dredge channel",
                                     "Arents-Water complex", "Water"],
                          "areasymbol": ["FL001", "FL001", "FL003", "FL003", "FL003"],
                          "flodfreqdcd": ["Frequent", None, "Very frequent", "None", "None"]})
    comps = comp_rows([(1, "1", "Pomona", "Series", "90", "Yes", "Typic Fluvaquents"),
                       (2, "2", "Water", "Miscellaneous area", "100", "Yes", None),
                       (3, "3", "Dredge", "Miscellaneous area", "100", "Yes", None),
                       (4, "4", "Arents", "Taxadjunct", "60", "Yes", None),
                       (5, "5", "Water", "Miscellaneous area", "100", "Yes", None)])
    catalog = pd.DataFrame({"areasymbol": ["FL001", "FL003"], "areaname": ["X", "Y"],
                            "saverest": ["9/2/2026 3:58:00 PM", "8/28/2026 11:40:55 AM"]})
    return resolved, units, comps, catalog


def check_code_points():
    print("5. code_points")
    t = sp.code_points(*synthetic_tables())
    check(list(t.columns) == sp.OUT_COLS and t.point_id.tolist() == list("abcdef"),
          "the output columns, one row per point in order")
    check(t.water_unit.tolist() == [False, True, True, False, True, False],
          "water by WATER_MU: Waters of the Gulf, Dredge channel, Water; not Arents-Water complex")
    check(t.soil_floodfreq.tolist()[:1] == [4.0] and t.soil_floodfreq.iloc[3] == 0
          and t.soil_floodfreq.iloc[[1, 2, 4, 5]].isna().all(),
          "flood codes: Frequent 4, None 0 on land; null, no map unit, and water units NaN even "
          "when rated (Dredge channel 'Very frequent', Water 'None')")
    check(t.soil_fluvclass.tolist()[0] == 1.0 and t.soil_fluvclass.iloc[3] == 0.0
          and t.soil_fluvclass.iloc[[1, 2, 4, 5]].isna().all(),
          "fluvial codes: fluvial 1, null subgroup on land 0; water units and no map unit NaN")
    water = t.water_unit.to_numpy(bool)
    check(t.soil_floodfreq[water].isna().all() and t.soil_fluvclass[water].isna().all(),
          "every water-unit point is missing in both soil features")
    check(t.saverest.tolist()[:3] == ["9/2/2026 3:58:00 PM", "9/2/2026 3:58:00 PM",
                                      "8/28/2026 11:40:55 AM"] and pd.isna(t.saverest.iloc[5]),
          "each point carries its survey area's version")
    res, units, comps, cat = synthetic_tables()
    check(raises(lambda: sp.code_points(res, units[units.mukey != 1], comps, cat)),
          "a mukey without a mapunit row raises")


class FakeSDA:
    """Answers POINT_SQL and SINGLE_SQL from mukey_at(lon, lat) -> list of mukeys; rejects a
    VALUES batch larger than max_batch as too big."""

    def __init__(self, mukey_at, max_batch=10**9):
        self.mukey_at = mukey_at
        self.max_batch = max_batch
        self.n_req = 0
        self.n_bytes = 0
        self.batches = []

    def query(self, sql, cols):
        self.n_req += 1
        pts = re.findall(r"\('([^']+)','POINT\(([-\d.]+) ([-\d.]+)\)'\)", sql)
        if pts:
            if len(pts) > self.max_batch:
                raise SDATooBig("too many")
            self.batches.append(len(pts))
            rows = [(p, m) for p, x, y in pts for m in self.mukey_at(float(x), float(y))]
            return pd.DataFrame(rows, columns=cols)
        x, y = re.search(r"POINT\(([-\d.]+) ([-\d.]+)\)", sql).groups()
        return pd.DataFrame([(m,) for m in self.mukey_at(float(x), float(y))], columns=cols)


def check_sda_side(tmp: Path):
    print("6. the SDA side")
    pts = pd.DataFrame({"point_id": [f"FL{i:07d}" for i in range(25)],
                        "lon": -82.0 + np.arange(25) * 0.001, "lat": 28.0 + np.arange(25) * 0.001})
    vals = sp.values_sql(pts.iloc[:2])
    check(vals == "('FL0000000','POINT(-82.000000000 28.000000000)'),"
                  "('FL0000001','POINT(-81.999000000 28.001000000)')",
          "VALUES rows: (point_id, WKT with x = longitude, 9 decimals)")
    bad = pts.iloc[:1].assign(point_id="x'); DROP TABLE t; --")
    check(raises(lambda: sp.values_sql(bad), ValueError), "an unsafe point_id is refused")

    def at(x, y):          # every 5th point lies on an edge (two map units), the last in none
        i = int(round((x + 82.0) / 0.001))
        if i == 24:
            return []
        return [100 + i, 50 + i] if i % 5 == 0 else [100 + i]

    sda = FakeSDA(at, max_batch=4)
    hits = sp.lookup_batch(sda, pts.iloc[:10])
    check(max(sda.batches) <= 4 and sorted(hits.point_id.unique()) == pts.point_id[:10].tolist(),
          f"a batch SDA calls too big is halved until it passes: {sda.batches}")
    cache = tmp / "cache"
    sda = FakeSDA(at)
    first = sp.lookup_points(sda, pts, cache, size=10, workers=2, refetch=False)
    n_first = sda.n_req
    again = sp.lookup_points(sda, pts, cache, size=10, workers=2, refetch=False)
    check(n_first == 3 and sda.n_req == 3 and len(list(cache.glob("pt_*.parquet"))) == 3
          and first.sort_values(["point_id", "mukey"]).reset_index(drop=True).equals(
              again.sort_values(["point_id", "mukey"]).reset_index(drop=True)),
          "3 batches queried once, then read from the cache")
    r = sp.resolve_mukeys(first, pts.point_id)
    check(int((r.n_mukeys > 1).sum()) == 5 and r.mukey.iloc[5] == 55 and pd.isna(r.mukey.iloc[24]),
          "edges take the smaller mukey; the last point is in no map unit")
    res = sp.check_single(FakeSDA(at), pts, 12)
    check(res["n"] == 12 and res["agree"] == 12, f"batched and one-point lookups agree: {res}")


def check_local(tmp: Path):
    print("7. local_mukeys")
    # The squares are built around the projected points, so the edge point lies exactly on the
    # shared edge in the same projection local_mukeys uses.
    to_work = Transformer.from_crs(4326, sp.CRS_WORK, always_xy=True)
    lon = np.array([-82.0005, -81.9995, -82.0, -81.99])
    lat = np.full(4, 28.0)
    x, y = to_work.transform(lon, lat)
    xe, ye = x[2], y[2]
    left = shapely.box(xe - 200, ye - 50, xe, ye + 50)
    right = shapely.box(xe, ye - 50, xe + 200, ye + 50)
    polys = gpd.GeoDataFrame({"mukey": [20, 10]}, geometry=[left, right], crs=sp.CRS_WORK)
    pts = pd.DataFrame({"point_id": ["in_left", "in_right", "edge", "outside"], "lon": lon,
                        "lat": lat})
    inside = [left.contains(shapely.Point(x[0], y[0])), right.contains(shapely.Point(x[1], y[1]))]
    got = sp.local_mukeys(pts, polys).set_index("point_id")
    check(all(inside) and got.mukey.to_dict() == {"in_left": 20, "in_right": 10, "edge": 10}
          and got.n_mukeys.to_dict() == {"in_left": 1, "in_right": 1, "edge": 2},
          f"inside each square, on the shared edge (smaller mukey, n 2), outside left out: "
          f"{got.mukey.to_dict()}, {got.n_mukeys.to_dict()}")


def check_build(tmp: Path):
    print("8. features_text and with_soil")
    columns = ["point_id", "label"] + list(reversed(sp.V4_FEATURES)) + ["meta_tri_300"]
    text = sp.features_text(columns)
    names = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    check(names == ["landcover", "soil_floodfreq", "soil_fluvclass", "slope", "elev_sd_r300",
                    "hand", "wetindexcat", "log10_dist_river", "log10_dist_coast"]
          and "300 m" in text and "x100" in text,
          "the nine features of the Sources paragraph in its order; the header maps each")
    check(raises(lambda: sp.features_text(columns[:-3]), ValueError),
          "refuses a table that lacks one of the nine")

    df = pd.DataFrame({"point_id": ["c", "a", "b"], "label": [1, 0, 1], "x": [0.5, np.nan, 2.0]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**table.schema.metadata, b"geo": b"{}"})
    soil = pd.DataFrame({"point_id": ["a", "b", "c"], "soil_floodfreq": [0.0, np.nan, 4.0],
                         "soil_fluvclass": [np.nan, 0.0, 1.0]})
    out = sp.with_soil(table, soil)
    pq.write_table(out, tmp / "v4.parquet")
    back = pd.read_parquet(tmp / "v4.parquet")
    check(back.point_id.tolist() == ["c", "a", "b"] and back[df.columns].equals(df)
          and back.soil_floodfreq.tolist()[0] == 4.0 and np.isnan(back.soil_floodfreq[2])
          and back.soil_fluvclass.tolist()[0] == 1.0 and np.isnan(back.soil_fluvclass[1]),
          "rows and order kept, values joined by point_id, NaN stays NaN")
    check(out.schema.metadata[b"geo"] == b"{}" and out.select(table.column_names).equals(table),
          "schema metadata and every old column unchanged")
    check(raises(lambda: sp.with_soil(table, soil.iloc[:2]), ValueError)
          and raises(lambda: sp.with_soil(out, soil), ValueError),
          "refuses other point ids and a table that already has the soil columns")


def check_real():
    print("9. the real files")
    soil_path = sp.SOIL_DIR / "soil_points_FL.parquet"
    v4 = Path("data/processed_v4/dataset_FL.parquet")
    base = Path("data/interim/v4_build/dataset_FL.parquet")
    if not (soil_path.exists() and v4.exists() and base.exists()):
        print("  skip  the soil table or data/processed_v4 is missing")
        return
    soil = pd.read_parquet(soil_path)
    data = pd.read_parquet(base)
    check(set(soil.point_id) == set(data.point_id) and soil.point_id.is_unique,
          "one soil row per point of the v4 build")
    known = soil.flodfreqdcd.dropna().isin(list(sp.FLOOD_CODES)).all()
    check(known and set(soil.soil_fluvclass.dropna().unique()) <= {0.0, 1.0}
          and soil.mukey.isna().mean() <= 0.02,
          "flood classes within the six; fluvial binary; at most 2% without a map unit")
    water = soil.water_unit.to_numpy(bool)
    check(soil.soil_floodfreq[water].isna().all() and soil.soil_fluvclass[water].isna().all(),
          "water-unit points are missing in both soil features")
    land = soil.mukey.notna() & ~soil.water_unit
    check(soil.flodfreqdcd[land].isna().mean() <= 0.05,
          f"non-water points with a null flood class: {soil.flodfreqdcd[land].isna().mean():.2%}")
    recoded = sp.code_points(soil[["point_id", "mukey", "n_mukeys"]],
                             soil[["mukey", "muname", "areasymbol", "flodfreqdcd"]].dropna(
                                 subset=["mukey"]),
                             soil.dropna(subset=["mukey"]).rename(columns={
                                 "dom_cokey": "cokey", "dom_compname": "compname",
                                 "dom_comppct_r": "comppct_r", "dom_taxsubgrp": "taxsubgrp"}
                                 ).assign(compkind=None, majcompflag="Yes")[sp.COMP_COLS],
                             soil[["areasymbol", "saverest"]].dropna())
    check(np.allclose(recoded.soil_floodfreq, soil.soil_floodfreq, equal_nan=True)
          and np.allclose(recoded.soil_fluvclass, soil.soil_fluvclass, equal_nan=True),
          "the stored codes follow from the stored classes and subgroups")
    new = pd.read_parquet(v4)
    by_id = soil.set_index("point_id").reindex(new.point_id)
    check(not Path(v4).is_symlink() and new[data.columns].equals(data)
          and np.allclose(new.soil_floodfreq, by_id.soil_floodfreq, equal_nan=True)
          and np.allclose(new.soil_fluvclass, by_id.soil_fluvclass, equal_nan=True),
          "data/processed_v4: a new file; the v4 build plus the two soil columns")
    names = [ln.strip() for ln in Path("data/processed_v4/FEATURES.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    check(names == sp.V4_FEATURES, "FEATURES.txt: the nine features in the paper's order")


def main():
    check_flood_code()
    check_dominant()
    check_fluvial()
    check_resolve()
    check_code_points()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_sda_side(tmp)
        check_local(tmp)
        check_build(tmp)
    check_real()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
