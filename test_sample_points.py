#!/usr/bin/env python3
"""
Checks for the candidate design of sample_points.py (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md
D1 and D2). Run from the repo root:

    .venv/bin/python test_sample_points.py

Everything is synthetic (a small Florida in a temp folder), no network:
1. draw_valid: exactly n candidates over several batches, a random n of the pool (not the first
   batch's), counts summed over batches, SupplyError when the supply never reaches n.
2. polygon_quota without a cap is the plain multinomial by area: a polygon with 90% of the area
   gets about 90% of the draws (a 2% cap would have cut it to 2%); with a cap it still caps.
3. area_uniform_candidates: exactly n candidates, all on land per the mask, spread over the
   polygons in proportion to their land area (half of the big polygon is water).
4. band_filter: each test removes what it should (closer than 500 m, beyond 5 km, outside a panel,
   in Zone D, in NFHL OPEN WATER, on water per the mask, outside the state), counted in order.
5. build_state end to end: every group draws exactly --candidates candidates, its own thinning and
   the whole-state pass leave no two points closer than 100 m, every survivor is kept (no cut),
   and the counts are stored with the points; --audit reads them; group_problems names a group
   with the wrong number of candidates or rows.
"""
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import from_origin

import sample_points as sp

FAILS = []
X0, Y0 = 1_400_000.0, 700_000.0     # EPSG:5070 origin of the synthetic state (north Florida)
PIXEL = 30.0
LAND, WATER = 42, 11


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def box(x0, y0, x1, y1):
    return shapely.box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


# ----------------------------------------------------------------- the synthetic state
HAZARD = [  # DFIRM_ID, FLD_AR_ID, FLD_ZONE, SFHA_TF, geometry (metres from the origin)
    ("12001C", "A1", "A", "T", box(0, 0, 9_000, 1_000)),          # 9 km2, its east half water
    ("12001C", "A2", "A", "T", box(0, 3_000, 1_000, 4_000)),      # 1 km2
    ("12003C", "AE1", "AE", "T", box(20_000, 0, 24_000, 1_000)),
    ("12003C", "AE2", "AE", "T", box(20_000, 3_000, 21_000, 5_000)),
    ("12001C", "D1", "D", "F", box(3_000, 6_000, 8_000, 8_000)),
    ("12003C", "N1", "AREA NOT INCLUDED", "F", box(25_000, 7_000, 27_000, 9_000)),
    ("12001C", "W1", "OPEN WATER", "F", box(10_000, 2_000, 12_000, 3_000)),
    ("12001C", "X1", "X", "F", box(-3_000, -3_000, -2_000, -2_000)),
]
WATER_BOXES = [(4_500, -500, 9_500, 1_500),            # the mask's water: east half of A1,
               (12_000, 3_500, 13_000, 4_500)]           # and a pond in the band
EXTENT = (-7_000, -7_000, 33_000, 13_000)               # mask and panel extent
COUNTY = box(-6_500, -6_500, 27_000, 12_500)            # the state: band negatives must lie inside


def hazard_table() -> gpd.GeoDataFrame:
    rows = [dict(DFIRM_ID=d, FLD_AR_ID=f, FLD_ZONE=z, SFHA_TF=t, ZONE_SUBTY=None,
                 STUDY_TYP="NP", STATIC_BFE=-9999.0, DEPTH=-9999.0, SOURCE_CIT=d + "_FIS",
                 geometry=g) for d, f, z, t, g in HAZARD]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=sp.CRS_WORK).to_crs(sp.CRS_SRC)


def write_mask(path: Path) -> None:
    """A land-cover raster in EPSG:5070: land (42) everywhere but WATER_BOXES (11)."""
    x0, y0, x1, y1 = EXTENT
    width, height = int((x1 - x0) / PIXEL), int((y1 - y0) / PIXEL)
    classes = np.full((height, width), LAND, dtype=np.uint8)
    for wx0, wy0, wx1, wy1 in WATER_BOXES:
        cols = slice(int((wx0 - x0) / PIXEL), int((wx1 - x0) / PIXEL))
        rows = slice(int((y1 - wy1) / PIXEL), int((y1 - wy0) / PIXEL))
        classes[rows, cols] = WATER
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="uint8",
                   crs="EPSG:5070", transform=from_origin(X0 + x0, Y0 + y1, PIXEL, PIXEL))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(classes, 1)


@contextlib.contextmanager
def world():
    """The synthetic state in a temp dir, with sample_points' folders pointed at it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        w = types.SimpleNamespace(root=root, nfhl=root / "nfhl", prep=root / "prep",
                                  points=root / "points", mask=root / "mask.tif",
                                  counties=root / "counties.parquet")
        w.nfhl.mkdir()
        hazard_table().to_parquet(w.nfhl / "haz_FL.parquet")
        panels = gpd.GeoDataFrame(
            [dict(DFIRM_ID=d, FIRM_PAN=f"{d}0100F", PANEL_TYP="PRINTED", SCALE="12000",
                  EFF_DATE=pd.Timestamp("2014-05-02", tz="UTC"), geometry=box(*EXTENT))
             for d in ("12001C", "12003C")], geometry="geometry", crs=sp.CRS_WORK)
        panels.to_crs(sp.CRS_SRC).to_parquet(w.nfhl / "pan_FL.parquet")
        gpd.GeoDataFrame({"GEOID": ["12001"], "NAME": ["Synthetic"]}, geometry=[COUNTY],
                         crs=sp.CRS_WORK).to_parquet(w.counties)
        write_mask(w.mask)
        with mock.patch.multiple(sp, IN_DIR=w.nfhl, PREP_DIR=w.prep, OUT_DIR=w.points,
                                 RG_ROWS=3):
            with quiet():
                sp.prep_state("FL")
            yield w


def cfg_for(w, **overrides) -> dict:
    cfg = dict(sp.DEFAULTS, protocols="band", state_polygons=str(w.counties),
               land_mask=str(w.mask), candidates=60)
    cfg.update(overrides)
    return cfg


# ----------------------------------------------------------------- 1: draw_valid
def check_draw_valid():
    print("1. draw_valid")
    rng = np.random.default_rng(0)
    calls = []

    def propose(k):                       # 30% valid; owner = proposal number, so order shows
        calls.append(k)
        start = sum(calls[:-1])
        owner = np.arange(start, start + k)
        valid = rng.random(k) < 0.3
        return (dict(x=owner[valid].astype(float), y=owner[valid].astype(float),
                     owner=owner[valid]), dict(proposed=k, invalid=int((~valid).sum())))

    cols, counts = sp.draw_valid(propose, 500, 400, rng)
    check(len(cols["x"]) == 500 and counts["candidates"] == 500 and counts["batches"] >= 2,
          f"exactly 500 candidates from {counts['batches']} batches of {calls}")
    check(counts["proposed"] == sum(calls) and counts["valid"] >= 500
          and counts["invalid"] == counts["proposed"] - counts["valid"],
          "counts summed over the batches")
    check(len(np.unique(cols["owner"])) == 500 and (np.diff(cols["owner"]) < 0).any()
          and cols["owner"].max() >= calls[0],
          "a random 500 of the pool, from every batch, in random order")
    check(raises(lambda: sp.draw_valid(
        lambda k: (dict(x=np.empty(0)), dict(proposed=k)), 10, 5, rng, max_batches=3),
        sp.SupplyError), "SupplyError when the valid share is zero")
    sizes = []

    def scarce(k):                        # one valid candidate in 10,000 proposals
        sizes.append(k)
        valid = rng.random(k) < 1e-4
        return dict(x=np.zeros(int(valid.sum()))), dict(proposed=k)

    raises(lambda: sp.draw_valid(scarce, 50, 100, rng, max_batches=4), sp.SupplyError)
    check(all(b <= sp.MAX_GROWTH * a for a, b in zip(sizes, sizes[1:])),
          f"a batch is at most {sp.MAX_GROWTH} times the one before: {sizes}")
    sizes.clear()

    def empty(k):
        sizes.append(k)
        return dict(x=np.zeros(0)), dict(proposed=k)

    check(raises(lambda: sp.draw_valid(empty, 10, 50, rng), sp.SupplyError)
          and sum(sizes) < 2 * sp.EMPTY_AFTER * 10 * sp.MAX_GROWTH,
          f"no valid candidate at all: gives up after {sum(sizes):,} proposals ({sizes})")


# ----------------------------------------------------------------- 2: polygon_quota
def check_quota():
    print("2. polygon_quota without a cap")
    rng = np.random.default_rng(1)
    weights = np.array([90.0, 6.0, 4.0])
    quota = sp.polygon_quota(weights, 20_000, None, rng)
    share = quota[0] / quota.sum()
    sd = np.sqrt(0.9 * 0.1 / 20_000)
    check(quota.sum() == 20_000 and abs(share - 0.9) < 4 * sd,
          f"the 90% polygon draws {share:.4f} of 20,000 (uniform by area, D2)")
    capped = sp.polygon_quota(weights, 20_000, 400, np.random.default_rng(1))
    check(capped.max() <= 400, "a cap, where one is given (the band's rings), still caps")
    check(sp.polygon_quota(np.zeros(3), 10, None, rng).sum() == 0
          and sp.polygon_quota(weights, 0, None, rng).sum() == 0, "no weight or no draw: zero")


# ----------------------------------------------------------------- 3: positives
def check_area_uniform(w):
    print("3. area_uniform_candidates")
    idx = sp.load_index("FL")
    zone = idx[(idx.FLD_ZONE == "A") & (idx.SFHA_TF == "T")].reset_index(drop=True)
    land = sp.LandMask(w.mask)
    cfg = dict(cfg_for(w), land=land)
    rng = np.random.default_rng(2)
    with quiet():
        cand, counts, cache = sp.area_uniform_candidates("FL", zone, 4_000, cfg, rng, "positive")
    on_land = sp.LandMask.is_land(land.classes_at(cand["x"], cand["y"]))
    check(len(cand["x"]) == 4_000 and on_land.all() and counts["not_on_land"] > 0,
          f"exactly 4,000 candidates, all on land ({counts['not_on_land']:,} of "
          f"{counts['proposed']:,} proposals on water dropped)")
    inside = shapely.contains_xy(np.array([cache[int(r)] for r in zone.row_id[cand["owner"]]]),
                                 cand["x"], cand["y"])
    check(inside.all(), "every candidate lies in its polygon")
    big = float((zone.FLD_AR_ID.to_numpy()[cand["owner"]] == "A1").mean())
    # A1's land: 4.5 of its 9 km2 (the mask's water box covers x 4,500 to 9,000); A2: 1 km2.
    want = 4.5 / 5.5
    check(abs(big - want) < 4 * np.sqrt(want * (1 - want) / 4_000),
          f"shares follow land area: A1 {big:.3f} (want {want:.3f}); no polygon cap")


# ----------------------------------------------------------------- 4: band_filter
def check_band_filter(w):
    print("4. band_filter")
    from shapely import STRtree
    idx = sp.load_index("FL")
    sfha = idx[idx.SFHA_TF == "T"]
    pan = sp.read_panels("FL")
    cfg = dict(cfg_for(w), land=sp.LandMask(w.mask),
               open_water_tree=STRtree(sp.read_open_water("FL", idx)))
    pts = {  # name: (x, y) from the origin, and the test it should fail (None: kept)
        "kept": ((4_000, 2_500), None),             # 1.5 km from A1 and A2, on land
        "near": ((4_000, 1_200), "too_near"),       # 200 m from A1
        "far": ((15_000, 11_000), "too_far"),       # 7.8 km or more from every SFHA
        "zone_d": ((5_000, 6_500), "unassessed"),   # 4.7 km from A2, in Zone D
        "open_water": ((11_000, 2_500), "open_water"),  # 2.5 km from A1, in NFHL OPEN WATER
        "pond": ((12_500, 3_800), "not_on_land"),   # 4.5 km from A1, water per the mask
        "outside": ((28_000, 2_000), "outside_state"),  # 4.1 km from AE1, east of the county
    }
    x = np.array([X0 + v[0][0] for v in pts.values()])
    y = np.array([Y0 + v[0][1] for v in pts.values()])
    ok, dist, mask_lc, counts = sp.band_filter(
        "FL", x, y, sfha, STRtree(np.asarray(pan.geometry.values)), cfg,
        state_tree=STRtree(np.asarray(gpd.read_parquet(w.counties).geometry.values)),
        unassessed_tree=STRtree(sp.read_unassessed("FL", idx)))
    kept = [name for name, flag in zip(pts, ok) if flag]
    check(kept == ["kept"], f"only the point that passes every test is kept: {kept} "
                            f"(distances {dist.round(0).astype(int).tolist()})")
    want = {test: 1 for _, test in pts.values() if test}
    got = {k: v for k, v in counts.items() if v}
    check(got == want, f"each failure counted once, in order: {got}")
    check(mask_lc is not None and mask_lc[list(pts).index("pond")] == WATER,
          "the mask class is returned for every candidate")


# ----------------------------------------------------------------- 5: the build
def check_build(w):
    print("5. build_state")
    cfg = cfg_for(w)
    with quiet():
        points = sp.build_state("FL", cfg)
    path = w.points / "points_FL.parquet"
    groups = sp.build_counts(path)
    stored = sp.build_config(path)
    check(set(groups) == {"A/1/-", "A/0/band", "AE/1/-", "AE/0/band"},
          f"four groups: {sorted(groups)}")
    check(all(c["candidates"] == 60 for c in groups.values()),
          "every group drew exactly 60 candidates (--candidates)")
    rows = points.groupby(["domain", "label", points.neg_protocol.fillna("-")]).size()
    in_file = {sp.group_key(d, label, p): int(n) for (d, label, p), n in rows.items()}
    check(all(in_file[k] == c["kept"] and c["kept"] <= c["after_group_thinning"] <= 60
              for k, c in groups.items()) and len(points) == sum(in_file.values()),
          f"every survivor kept, no cut: kept {[(k, c['kept']) for k, c in groups.items()]}")
    xy = np.column_stack([points.geometry.x, points.geometry.y])
    tree = shapely.STRtree(shapely.points(xy))
    pairs = tree.query(shapely.points(xy), predicate="dwithin", distance=100.0)
    check(int((pairs[0] < pairs[1]).sum()) == 0, "no two points closer than 100 m")
    pos = points[points.label == 1]
    land = sp.LandMask(w.mask)
    check(sp.LandMask.is_land(land.classes_at(points.geometry.x, points.geometry.y)).all(),
          "every point on land per the mask")
    check(stored["candidates"] == 60 and "per_domain" not in stored and "surplus" not in stored
          and "exact" not in stored and "poly_cap_frac" not in stored,
          "the stored settings name the candidate count, not the retired targets")
    check(len(pos) > 0 and (pos.meta_dist_to_sfha_m == 0).all(), "positives sit in their zone")
    counts = rows
    check(sp.group_problems(counts, stored, groups) == [], "group_problems: none on the build")
    bad = {k: dict(c) for k, c in groups.items()}
    bad["A/1/-"]["candidates"] = 59
    check(any("59 candidates" in b for b in sp.group_problems(counts, stored, bad)),
          "group_problems names a group with 59 candidates")
    fewer = counts.copy()
    fewer.iloc[0] -= 1
    check(any("rows in the file" in b for b in sp.group_problems(fewer, stored, groups)),
          "group_problems names a group whose rows differ from its stored kept count")
    check(sp.group_problems(counts, stored, None) == ["no candidate counts stored with the build"],
          "a build without stored counts is named")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sp.audit(["FL"], cfg)
    text = out.getvalue()
    check("60 candidates per group, every survivor kept: YES" in text
          and "pairs closer than 100 m: 0" in text and "points not on land per the mask: 0" in text,
          "--audit reads the stored counts and passes")
    table = sp.group_table(groups)
    check(table.iloc[-1].group == "total" and table.iloc[-1].candidates == 240
          and table.iloc[-1].kept == len(points), "group_table totals the groups")
    return points


def main():
    check_draw_valid()
    check_quota()
    with world() as w:
        check_area_uniform(w)
        check_band_filter(w)
        check_build(w)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
