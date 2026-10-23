#!/usr/bin/env python3
"""
Checks for o5_polygons.py (docs/TASK-O5-REGIONS.md, Part E). Run from the repo root, with
--null-file the --null-out table of o5_polygons.py:

    .venv/bin/python aotil/test_o5_polygons.py --null-file "$SCRATCH/E_null_draws.parquet" \\
        [--polygons results/2026-10-17/o5/E_polygons.csv] \\
        [--summary results/2026-10-17/o5/E_summary.csv] [--log logs/2026-10-17/E_polygons.log]

Sections 1-6 and 8 run the code on synthetic inputs with known answers: straight runs on rings,
part metrics, part numbering against lobes.explode, perimeter matching against
lobes.match_perimeter, the lobe rules, the small statistics, the own-DFIRM panel rule and
summarise() end to end (point estimates, both bootstraps, the null, the sensitivities, the
primary flag), each recomputed independently. Section 7 checks the outputs, when they exist,
against values recomputed from E_polygons.csv, the stored lobe geometry and the raw panel, LOMR
and attribute layers, the log, and with --null-file the null percentiles and saved null draws.
Exit code 0 when every check passes.
"""
import argparse
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import geopandas as gpd       # noqa: E402
import shapely                # noqa: E402
import lobes as lb            # noqa: E402
import o5_polygons as e5      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def nclose(a, b, tol=1e-9):
    """close() that treats two NaNs as equal."""
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)
    return close(a, b, tol)


def midrank(obs, v):
    """Mid-rank percentile, written here independently of o5_polygons."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v) or not np.isfinite(obs):
        return np.nan
    below = (v < obs - 1e-12).sum()
    tied = (np.abs(v - obs) <= 1e-12).sum()
    return 100.0 * (below + 0.5 * tied) / len(v)


def txt(x):
    """A cell as text, with None and NaN as ''."""
    if x is None:
        return ""
    if not isinstance(x, str) and pd.isna(x):
        return ""
    return str(x)


def pct(v):
    """2.5 / 97.5 percentiles of the finite values; NaN for fewer than 10."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.array([np.nan, np.nan])
    return np.percentile(v, [2.5, 97.5])


def ring(pts):
    """A closed ring: the points with the first one repeated at the end."""
    pts = np.asarray(pts, float)
    return np.vstack([pts, pts[:1]])


def walk(start, legs):
    """Vertices from start, legs = [(length, left turn in degrees before the leg)]."""
    x, y = start
    heading = 0.0
    out = [(x, y)]
    for length, turn in legs:
        heading += np.deg2rad(turn)
        x, y = x + length * np.cos(heading), y + length * np.sin(heading)
        out.append((x, y))
    return out


# ------------------------------------------------------------------ 1-6: units with known answers
def random_ring(rng):
    """40 random vertices around a circle, each edge densified to 11 slightly jittered points."""
    ang = np.sort(rng.uniform(0, 2 * np.pi, 40))
    rad = 3000 * (1 + 0.3 * rng.random(40))
    pts = []
    for k in range(40):
        a0, a1 = ang[k], ang[(k + 1) % 40]
        p0 = np.array([rad[k] * np.cos(a0), rad[k] * np.sin(a0)])
        p1 = np.array([rad[(k + 1) % 40] * np.cos(a1), rad[(k + 1) % 40] * np.sin(a1)])
        for f in np.linspace(0, 1, 12)[:-1]:
            q = p0 + f * (p1 - p0)
            pts.append(q + rng.normal(0, 0.3, 2) * (f > 0))
    return np.array(pts)


def test_ring_runs():
    print("1. ring_runs")
    o = e5.ring_runs(ring([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]))
    check(o["n_seg"] == 4 and close(o["perim"], 4000), "1 km square: 4 vertices, P 4000")
    check(close(o["straight"], 4000) and close(o["straight_vertex"], 4000)
          and close(o["longest"], 1000), "1 km square: all 4000 m straight, longest 1000")
    o = e5.ring_runs(ring([(0, 0), (200, 0), (200, 200), (0, 200)]))
    check(o["straight"] == 0 and o["straight_vertex"] == 0, "200 m square: no straight run")
    side = np.linspace(0, 1000, 101)[:-1]
    dense = np.vstack([np.c_[side, 0 * side], np.c_[1000 + 0 * side, side],
                       np.c_[1000 - side, 1000 + 0 * side], np.c_[0 * side, 1000 - side]])
    o = e5.ring_runs(ring(dense))
    check(o["n_seg"] == 400 and close(o["straight"], 4000) and close(o["longest"], 1000),
          f"square at 10 m: 400 vertices, still 4000 m straight, longest {o['longest']:.1f}")
    o = e5.ring_runs(ring([(500, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)]))
    check(close(o["straight"], 4000) and close(o["longest"], 1000) and o["n_seg"] == 5,
          "start mid-side: the run across the ring's start is one 1000 m run")
    o = e5.ring_runs(ring([(0, 0), (1000, 0), (1000, 0), (1000, 1000), (0, 1000)]))
    check(o["n_seg"] == 4 and close(o["straight"], 4000), "repeated vertex is not a vertex")
    t = np.linspace(0, 2 * np.pi, 3601)[:-1]
    o = e5.ring_runs(ring(np.c_[1000 * np.cos(t), 1000 * np.sin(t)]))
    check(o["n_seg"] == 3600 and close(o["straight_vertex"], o["perim"], 1e-9),
          "circle, 0.1 deg per vertex: the per-vertex variant calls all of it straight")
    check(o["straight"] == 0, f"circle: the 1-degree band finds no straight run "
          f"({o['straight']:.1f} m)")
    o = e5.ring_runs(ring(np.c_[1e5 * np.cos(t), 1e5 * np.sin(t)]))
    check(close(o["straight"], o["perim"], 1e-9) and o["longest"] >= 1500,
          f"100 km circle (175 m segments, 0.1 deg turns): all straight, windows wrap the ring "
          f"start (longest {o['longest']:.0f} m)")
    # four 140 m legs with 0.6 deg bends: every window spanning < 1 deg is 280 m
    v = walk((0, 0), [(140, 0), (140, 0.6), (140, 0.6), (140, 0.6)])
    end = v[-1]
    o = e5.ring_runs(ring(v + [(end[0], end[1] - 2000), (0, -2000)]))
    check(close(o["straight_vertex"] - o["straight"], 560, 1e-9),
          f"gentle bends: the per-vertex variant counts 560 m more "
          f"({o['straight_vertex']:.3f} vs {o['straight']:.3f})")
    # neither the direction nor the start vertex matters
    pts = random_ring(np.random.default_rng(1))
    ref = e5.ring_runs(ring(pts))
    same = True
    for variant in (pts[::-1], np.roll(pts, 17, axis=0), np.roll(pts[::-1], 5, axis=0)):
        o = e5.ring_runs(ring(variant))
        for k in ("straight", "straight_vertex", "longest"):
            same &= close(o[k], ref[k], 1e-9)
    check(same and ref["straight"] > 0, f"reversed / rotated ring: same straight lengths "
          f"(band {ref['straight']:.1f}, per-vertex {ref['straight_vertex']:.1f} of "
          f"{ref['perim']:.1f})")


def test_part_metrics():
    print("2. part_metrics")
    g = shapely.Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
                        [[(400, 400), (500, 400), (500, 500), (400, 500)]])
    sq = shapely.box(0, 0, 500, 500)
    m = e5.part_metrics(np.array([g, sq], dtype=object))
    r = m.iloc[0]
    check(r.n_vertices == 8 and r.n_vertices_raw == 8 and r.n_rings == 2,
          "square with hole: 8 vertices on 2 rings")
    check(close(r.perimeter_m, 4400) and close(r.vertex_spacing_m, 550)
          and close(r.area_m2, 1e6 - 1e4), "P 4400, spacing 550, area 990,000")
    check(close(r.straight_share, 4000 / 4400), "straight share 4000/4400 (hole sides < 300 m)")
    check(close(r.shape_index, 2 * np.log(1100) / np.log(990000)), "shape index formula")
    check(close(m.iloc[1].shape_index, 1.0, 1e-12), "shape index of a square is 1")
    check(close(r.seg_len_median_m, 550), "median segment 550 (four 1000 m, four 100 m)")


def test_explode():
    print("3. explode_raw vs lobes.explode")
    bow = shapely.Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])          # invalid bow tie
    mp = shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(5, 5, 7, 7)])
    geoms = {3: mp, 7: bow, 9: shapely.box(0, 0, 3, 3)}
    ref, n_fixed = lb.explode(geoms, {3: "A", 7: "A", 9: "AE"})
    mine = e5.explode_raw(geoms)
    check(list(zip(ref.row_id, ref.part)) == list(zip(mine.row_id, mine.part)),
          f"same (row_id, part) numbering: {list(zip(mine.row_id, mine.part))}")
    eq = shapely.equals_exact(np.array(ref.geometry.to_numpy(), dtype=object),
                              np.array(mine.piece.to_numpy(), dtype=object), tolerance=0)
    check(bool(eq.all()), "identical stored pieces (oriented, repaired)")
    rep = mine[mine.row_id == 7]
    raw_is_bow = all(shapely.equals_exact(r, bow, 0) for r in rep.raw)
    check(n_fixed == 1 and rep.repaired.all() and len(rep) == 2 and raw_is_bow,
          "the bow tie: 2 repaired pieces, raw = the original invalid part")


def test_match():
    print("4. match_perimeter_fast == lobes.match_perimeter")
    ok = True
    for t in range(60):
        r = np.random.default_rng(t)
        nr, nc = int(r.integers(1, 40)), int(r.integers(1, 80))
        if t % 2:                       # odd cases: integer log P, so many ties
            reg = r.integers(0, 6, nr).astype(float)
            cand = r.integers(0, 6, nc).astype(float)
        else:
            reg = r.normal(size=nr)
            cand = r.normal(size=nc)
        a = lb.match_perimeter(reg, cand, 3, np.random.default_rng([0, t]))
        b = e5.match_perimeter_fast(reg, cand, 3, np.random.default_rng([0, t]))
        ok &= a == b
    check(ok, "60 random cases (half with tied log P, some exhausting the candidates)")


def part_keys(parts, ix):
    return [(int(parts.row_id.iat[i]), int(parts.part.iat[i])) for i in ix]


def test_rule():
    print("5. LobeRule on a synthetic map")
    # Zone A polygons: 1 covers the points; 2 is 1.5 km away (near); 3 is 5 km away;
    # 4 is a two-part polygon, one part on the hull, one 30 km away; 5 is AE
    geoms = {1: shapely.box(0, 0, 1000, 1000), 2: shapely.box(2500, 0, 3000, 500),
             3: shapely.box(6000, 0, 6500, 500),
             4: shapely.MultiPolygon([shapely.box(-1500, 0, -1200, 300),
                                      shapely.box(-31000, 0, -30500, 400)]),
             5: shapely.box(0, 3000, 900, 3900)}
    zone = {1: "A", 2: "A", 3: "A", 4: "A", 5: "AE"}
    parts = e5.explode_raw(geoms)
    pz = np.array([zone[r] for r in parts.row_id], dtype=object)
    polys_a = {r: g for r, g in geoms.items() if zone[r] == "A"}
    rule = e5.LobeRule(parts, pz, polys_a, ["0_0", "-1_0"])
    xy = np.array([[100.0, 100.0], [900.0, 900.0]])
    reg = rule.region_parts(xy)
    keys = part_keys(parts, reg)
    check(keys == [(1, 0), (2, 0), (4, 0)], f"near + hull: {keys} (polygon 3 too far, the far "
          f"part of 4 off the hull)")
    isreg = np.zeros(len(parts), bool)
    isreg[reg] = True
    ek = part_keys(parts, rule.eligible(["0_0"], rule.covers(xy) | isreg))
    check(ek == [(3, 0)], f"eligible in tile 0_0: {ek} (A only, not covering, not region)")


def test_small():
    print("6. mid_rank_pct, ci, source_kind, Columns")
    check(e5.mid_rank_pct(5.0, np.arange(5.0)) == 100.0, "above all -> 100")
    check(e5.mid_rank_pct(2.0, np.arange(5.0)) == 50.0, "middle with one tie -> (2 + 0.5)/5")
    check(e5.mid_rank_pct(-1.0, np.arange(5.0)) == 0.0, "below all -> 0")
    check(np.isnan(e5.mid_rank_pct(np.nan, np.arange(5.0))), "NaN observed -> NaN")
    lo, hi = e5.ci(np.arange(1001.0))
    check(close(lo, 25.0) and close(hi, 975.0), "ci = 2.5/97.5 percentiles")
    k = e5.source_kind(pd.Series(["12021C_STUDY6", "12086C_LOMC12", None, "12001C_BASE1"]))
    check(list(k.fillna("NA")) == ["STUDY", "LOMC", "NA", "BASE"], f"source_kind: {list(k)}")
    df = pd.DataFrame(dict(perimeter_m=[100.0, 300.0], n_vertices=[10, 10], straight_m=[0, 300.0],
                           straight_vertex_m=[0, 300.0], area_m2=[1.0, 3.0],
                           lomr_share=[1.0, 0.0], vertex_spacing_m=[10.0, 30.0]))
    C = e5.Columns(df, df.columns)
    ix = np.array([0, 1, 1])
    check(close(C.stat(ix, "vertex_spacing_m", "agg"), 700 / 30), "agg spacing = sum P / sum n")
    check(close(C.stat(ix, "straight_share", "agg"), 600 / 700), "agg straight = sum / sum P")
    check(close(C.stat(ix, "lomr_share", "agg"), 1 / 7), "agg LOMR = covered area / area")
    check(close(C.stat(ix, "vertex_spacing_m", "median"), 30.0), "median with repeats")
    df["flag"] = [1.0, np.nan]
    C = e5.Columns(df, df.columns)
    C01 = e5.Columns(df.assign(flag=[1.0, 0.0]), df.columns)
    check(close(C.stat(np.array([0, 1]), "flag", "pshare"), 1.0)
          and close(C01.stat(ix, "flag", "pshare"), 100 / 700),
          "pshare = perimeter-weighted share, NaN rows left out")


def test_panels():
    print("6b. attach_panels: the lobe's own DFIRM")
    pan = pd.DataFrame(dict(
        DFIRM_ID=["A", "B", "A"], FIRM_PAN=["A1", "B1", "A2"],
        EFF_DATE=pd.to_datetime(["2010-01-01", "2015-06-01", "2012-01-01"], utc=True),
        PRE_DATE=pd.to_datetime(["9999-09-09", "2001-01-01", "9999-09-09"], utc=True),
        SCALE=["24000", "6000", "12000"], PANEL_TYP=["x", "y", "z"]))
    # A1 and B1 cover the same square (two counties' maps overlap); A2 is east of it
    pg = np.array([shapely.box(0, 0, 100, 100), shapely.box(0, 0, 100, 100),
                   shapely.box(100, 0, 200, 100)], dtype=object)
    g = np.array([shapely.box(10, 10, 50, 50),       # DFIRM B: inside A1 and B1 -> B1
                  shapely.box(10, 10, 50, 50),       # DFIRM A: the same lobe -> A1
                  shapely.box(90, 10, 110, 20),      # DFIRM A: half on A1, half on A2 -> tie
                  shapely.box(10, 10, 20, 20),       # DFIRM C: no panel of its own -> fallback
                  shapely.box(500, 500, 510, 510)],  # on no panel
                 dtype=object)
    out, _ = e5.attach_panels(g, (pan, pg, shapely.STRtree(pg)), ["B", "A", "A", "C", "A"])
    fp = [txt(x) for x in out.pan_firm_pan]
    check(fp == ["B1", "A1", "A1", "A1", ""],
          f"dominant panel: own DFIRM, tie to the lower row, fallback to all: {fp}")
    check(list(out.pan_n) == [1, 1, 2, 2, 0] and list(out.pan_n_all) == [2, 2, 3, 2, 0],
          f"pan_n over the panels used {list(out.pan_n)}, pan_n_all {list(out.pan_n_all)}")
    check(list(out.pan_fallback) == [0, 0, 0, 1, 0], f"pan_fallback {list(out.pan_fallback)}")
    check(list(out.pan_scale.iloc[:2]) == [6000, 24000] and out.pan_pre_present.iat[0] == 1
          and out.pan_pre_present.iat[1] == 0 and bool(np.isnan(out.pan_pre_present.iat[4])),
          "scale and pre-date of the own DFIRM's panel")
    check(close(out.pan_cover.iat[1], 1.0) and close(out.pan_cover.iat[2], 1.0)
          and close(out.pan_cover.iat[3], 2.0), "pan_cover over the panels used (own: 1; fallback: 2)")
    check(out.pan_n_scales.iat[2] == 2 and close(out.pan_eff_year.iat[0], 2015 + 151 / 365.25),
          "pan_n_scales and pan_eff_year")


# ------------------------------------------------------------------ 8. summarise end to end
def synth_lobes(rng, n, shift):
    """n synthetic lobes whose vertex spacing is shifted by shift."""
    spacing = 10.0 + shift + rng.gamma(2.0, 2.0, n)
    nv = rng.integers(10, 200, n).astype(float)
    P = spacing * nv
    st = P * rng.uniform(0, 0.3, n) * (rng.random(n) < 0.5)
    lomr = np.where(rng.random(n) < 0.3, rng.random(n), 0.0)
    scale = rng.choice([6000.0, 12000.0, 24000.0], n)
    shape_index = rng.uniform(1.0, 1.3, n)
    area = rng.uniform(1e3, 1e6, n)
    eff_year = 2000.5 + rng.integers(0, 20, n)
    pre_present = (rng.random(n) < 0.3).astype(float)
    pan_n = rng.integers(1, 4, n).astype(float)
    study_typ = rng.choice(["DIGITAL CONVERSION", "NP"], n)
    study_regime = rng.choice(["detailed", "approximate"], n)
    kind = pd.Series(rng.choice(["STUDY", "FIRM", ""], n))
    return pd.DataFrame(dict(
        perimeter_m=P, n_vertices=nv, vertex_spacing_m=spacing, seg_len_median_m=spacing * 0.8,
        straight_m=st, straight_share=st / P, straight_vertex_m=st, straight_share_vertex=st / P,
        longest_straight_m=st / 2, shape_index=shape_index, area_m2=area, pan_eff_year=eff_year,
        pan_pre_present=pre_present, pan_scale=scale,
        pan_scale_cat=pd.Series(scale).astype("Int64").astype("string"), pan_n=pan_n,
        lomr_share=lomr, lomr_share_eff=lomr / 2, lomr_any=(lomr > 0).astype(float),
        study_typ=study_typ, study_regime=study_regime, source_kind=kind.mask(kind == "")))


def synth_regions(rng, regions, shift):
    """Lobes of 4 regions: region lobes shifted by the region's amount, matched_A unshifted, AE
    5 m finer; matched_A lobes matched to random region lobes of their region."""
    frames = []
    for k, reg in enumerate(regions):
        b, s, c = reg
        for role, n, sh in (("region", 7 + 3 * k, shift[reg]), ("matched_A", 18 + 2 * k, 0.0),
                            ("AE", 11 + k, -5.0)):
            d = synth_lobes(rng, n, sh)
            d.insert(0, "budget", b)
            d.insert(1, "seed", s)
            d.insert(2, "component", c)
            d.insert(3, "role", role)
            d["lobe_id"] = [f"c{c}_{role}_{k}_{i}" for i in range(n)]
            first_row = {"region": 0, "matched_A": 1000, "AE": 2000}[role]
            d["src_row_id"] = 10000 * k + first_row + np.arange(n)
            d["part"] = 0
            d["n_points"] = rng.integers(0, 3, n) if role == "region" else 0
            d["matched_to"] = None
            d["water_only"] = False
            frames.append(d)
    E = pd.concat(frames, ignore_index=True)
    for reg in regions:
        in_reg = (E.seed == reg[1]) & (E.component == reg[2])
        reg_ids = E.loc[in_reg & (E.role == "region"), "lobe_id"].to_numpy()
        is_matched = in_reg & (E.role == "matched_A")
        E.loc[is_matched, "matched_to"] = rng.choice(reg_ids, int(is_matched.sum()))
    return E


def synth_null(rng, regions, shift, n_draws):
    """n_draws null patches per region (region k's patch lobes shifted by half its amount);
    returns (the patches' lobes in one frame, null_tab indexing into it)."""
    frames, null_tab, offset = [], {}, 0
    for reg in regions:
        for d in range(n_draws):
            nr, nmt = int(rng.integers(4, 9)), int(rng.integers(12, 25))
            x = synth_lobes(rng, nr + nmt, 0.0)
            x.loc[: nr - 1, "vertex_spacing_m"] += shift[reg] * 0.5 + 0.1 * d
            frames.append(x)
            null_tab.setdefault((reg[1], reg[2]), {})[d] = dict(
                region_ix=np.arange(offset, offset + nr),
                matched_ix=np.arange(offset + nr, offset + nr + nmt))
            offset += nr + nmt
    return pd.concat(frames, ignore_index=True), null_tab


def test_summarise():
    print("8. summarise() end to end (synthetic)")
    rng = np.random.default_rng(11)
    regions = [(500, 0, 0), (500, 0, 1), (500, 1, 0), (500, 1, 2)]
    types = {regions[0]: "positive-dominated", regions[1]: "negative-dominated",
             regions[2]: "positive-dominated", regions[3]: "negative-dominated"}
    shift = {regions[0]: 4.0, regions[1]: -3.0, regions[2]: 9.0, regions[3]: 1.0}
    E = synth_regions(rng, regions, shift)
    D = 40
    NF, null_tab = synth_null(rng, regions, shift, D)
    null_tab["_frame"] = NF
    a = SimpleNamespace(budget=500, n_boot=60)
    keep = []
    S = e5.summarise(E.copy(), regions, types, null_tab, {}, a, lambda m: None, null_keep=keep)
    NK = pd.concat(keep, ignore_index=True)

    sp = E.vertex_spacing_m.to_numpy()
    grp = {}
    for _, s, c in regions:
        grp[(s, c)] = {}
        for r in e5.ROLES:
            grp[(s, c)][r] = sp[((E.seed == s) & (E.component == c) & (E.role == r)).to_numpy()]

    def pooled(pick, r):
        return np.median(np.concatenate([grp[(s, c)][r] for _, s, c in pick]))

    def row(scope, typ="", subset="all", metric="vertex_spacing_m", stat="median"):
        q = S[(S.scope == scope) & (S.subset == subset) & (S.metric == metric) & (S.stat == stat)
              & (S.region_type.fillna("") == typ)]
        return q.iloc[0] if len(q) == 1 else None

    nsp = NF.vertex_spacing_m.to_numpy()
    for scope, typ, key in (("pooled", "", [0]), ("type", "positive-dominated", [1, 0]),
                            ("type", "negative-dominated", [1, 1])):
        rs = [r for r in regions if not typ or types[r] == typ]
        q = row(scope, typ)
        if q is None:
            check(False, f"{scope} {typ}: one vertex_spacing_m/median row")
            continue
        dm = pooled(rs, "region") - pooled(rs, "matched_A")
        dA = pooled(rs, "region") - pooled(rs, "AE")
        lab = f"{scope} {typ or 'all'}"
        check(close(q.d_rm, dm, 1e-12) and close(q.d_rAE, dA, 1e-12),
              f"{lab}: point estimates {q.d_rm:+.4f}, {q.d_rAE:+.4f}")
        r_ = np.random.default_rng([e5.BOOT_SEED, *key])
        bm, ba = [], []
        for _ in range(a.n_boot):
            pick = [rs[i] for i in r_.integers(0, len(rs), len(rs))]
            bm.append(pooled(pick, "region") - pooled(pick, "matched_A"))
            ba.append(pooled(pick, "region") - pooled(pick, "AE"))
        (lo, hi), (alo, ahi) = pct(bm), pct(ba)
        check(close(q.d_rm_lo, lo, 1e-12) and close(q.d_rm_hi, hi, 1e-12)
              and close(q.d_rAE_lo, alo, 1e-12) and close(q.d_rAE_hi, ahi, 1e-12) and hi > lo,
              f"{lab}: region-bootstrap intervals of both contrasts "
              f"[{lo:+.3f}, {hi:+.3f}], [{alo:+.3f}, {ahi:+.3f}]")
        nv = []
        for d in range(D):
            Ir = np.concatenate([null_tab[(s, c)][d]["region_ix"] for _, s, c in rs])
            Im = np.concatenate([null_tab[(s, c)][d]["matched_ix"] for _, s, c in rs])
            nv.append(np.median(nsp[Ir]) - np.median(nsp[Im]))
        nv = np.array(nv)
        check(close(q.null_pct, midrank(q.d_rm, nv), 1e-12)
              and close(q.null_median, np.median(nv), 1e-12)
              and close(q.null_lo, np.percentile(nv, 2.5), 1e-12)
              and close(q.null_hi, np.percentile(nv, 97.5), 1e-12) and q.null_n == D,
              f"{lab}: null pools the d-th patch of every region (pct {q.null_pct:.1f}, "
              f"median {q.null_median:+.3f})")
        kk = NK[(NK.scope == scope) & (NK.region_type == typ) & (NK.metric == "vertex_spacing_m")
                & (NK.stat == "median")].sort_values("draw")
        check(len(kk) == D and np.allclose(kk.d_rm.to_numpy(), nv, rtol=0, atol=1e-12),
              f"{lab}: saved null draws = the recomputed ones")

    # one region's lobe bootstrap
    reg = regions[2]
    q = S[(S.scope == "region") & (S.seed == reg[1]) & (S.component == reg[2])
          & (S.metric == "vertex_spacing_m") & (S.stat == "median")].iloc[0]
    g = grp[(reg[1], reg[2])]
    r_ = np.random.default_rng([e5.BOOT_SEED, 99, *reg])
    bm = []
    for _ in range(a.n_boot):
        x = {r: g[r][r_.integers(0, len(g[r]), len(g[r]))] for r in e5.ROLES}
        bm.append(np.median(x["region"]) - np.median(x["matched_A"]))
    lo, hi = pct(bm)
    patches = null_tab[reg[1:]]
    nv = np.array([np.median(nsp[patches[d]["region_ix"]]) - np.median(nsp[patches[d]["matched_ix"]])
                   for d in range(D)])
    check(close(q.d_rm, np.median(g["region"]) - np.median(g["matched_A"]), 1e-12)
          and close(q.d_rm_lo, lo, 1e-12) and close(q.d_rm_hi, hi, 1e-12)
          and close(q.null_pct, midrank(q.d_rm, nv), 1e-12),
          f"region s{reg[1]} c{reg[2]}: estimate, lobe-bootstrap interval, null percentile")

    # other stats, pooled
    P_, n_ = E.perimeter_m.to_numpy(), E.n_vertices.to_numpy()
    rm = (E.role == "region").to_numpy()
    mmk = (E.role == "matched_A").to_numpy()
    q = row("pooled", stat="agg")
    check(close(q.d_rm, P_[rm].sum() / n_[rm].sum() - P_[mmk].sum() / n_[mmk].sum(), 1e-12),
          "pooled agg spacing = sum P / sum n per role")
    v = (E.pan_scale == 24000).to_numpy(float)
    q = row("pooled", metric="pan_scale_cat=24000", stat="pshare")
    check(q is not None and close(q.d_rm, (v * P_)[rm].sum() / P_[rm].sum()
                                  - (v * P_)[mmk].sum() / P_[mmk].sum(), 1e-12),
          "pooled share of edge on 1:24,000 panels")
    sk = E.source_kind
    v = np.where(sk.isna(), np.nan, (sk == "FIRM").astype(float))
    q = row("pooled", metric="source_kind=FIRM", stat="share")
    check(q is not None and close(q.d_rm, np.nanmean(v[rm]) - np.nanmean(v[mmk]), 1e-12),
          "pooled share of lobes, a missing category left out")

    # the point-lobe sensitivity
    holds = rm & (E.n_points > 0).to_numpy()
    pr = sp[holds]
    pm = sp[mmk & E.matched_to.isin(set(E.lobe_id[holds])).to_numpy()]
    q = row("pooled", subset="point_lobes")
    check(close(q.d_rm, np.median(pr) - np.median(pm), 1e-12),
          "point_lobes: lobes holding a point against the lobes matched to them")
    P1 = S[S.primary == True]  # noqa: E712
    ok = len(P1) == 1
    if ok:
        p1 = P1.iloc[0]
        ok = (p1.scope == "pooled" and p1.metric == "vertex_spacing_m" and p1.stat == "median"
              and p1.subset == "all" and bool(p1.met) == bool(p1.d_rm > 0 and p1.d_rm_lo > 0))
    check(ok, "one primary row (pooled, all, vertex_spacing_m, median); met = interval above 0")


# ------------------------------------------------------------------ 7. the outputs
def test_outputs(a):
    print("7. outputs")
    if not a.polygons.exists() or not a.summary.exists():
        print("  skip  (no outputs)")
        return
    E = pd.read_csv(a.polygons, low_memory=False)
    S = pd.read_csv(a.summary, low_memory=False)
    k = ["budget", "seed", "component", "role", "lobe_id"]
    check(not E.duplicated(k).any(), f"one row per lobe and role ({len(E):,} rows)")
    check(np.allclose(E.vertex_spacing_m, E.perimeter_m / E.n_vertices),
          "vertex_spacing_m = perimeter / n_vertices")
    check(bool((E.n_vertices <= E.n_vertices_raw).all()), "n_vertices <= n_vertices_raw")
    check(int(E.repaired.sum()) == 0 and np.allclose(E.area_m2, E.stored_area_m2, rtol=1e-9),
          "no repaired part; original-part area = stored lobe area")
    land = pd.read_csv("results/2026-10-17/o5/regions_land.csv")
    want = set()
    for r in land.itertuples():
        if r.type != "no land points":
            want.add((int(r.seed), int(r.component)))
    E5 = E[E.budget == 500]
    have = set()
    for (s, c), g in E5.groupby(["seed", "component"]):
        if {"region", "matched_A", "AE"} <= set(g.role):
            have.add((s, c))
    check(have == want, f"{len(have)} S = 500 regions with all three roles = the land regions")
    check_stored_geometry(E)

    # the primary
    P = S[S.primary == True]  # noqa: E712
    check(len(P) == 1, "exactly one primary row")
    p = P.iloc[0]
    regs = sorted(want)
    med = {r: E5[E5.role == r].vertex_spacing_m.median() for r in e5.ROLES}
    check(close(p.region, med["region"]) and close(p.matched_A, med["matched_A"])
          and close(p.d_rm, med["region"] - med["matched_A"]) and close(p.AE, med["AE"]),
          f"primary point estimate from E_polygons: {p.d_rm:+.4f}")
    grp = {}
    for (s, c), g in E5.groupby(["seed", "component"]):
        grp[(s, c)] = {r: g[g.role == r].vertex_spacing_m.to_numpy() for r in e5.ROLES}
    rng = np.random.default_rng([e5.BOOT_SEED, 0])
    d = []
    for _ in range(e5.N_BOOT):
        pick = [regs[i] for i in rng.integers(0, len(regs), len(regs))]
        d.append(np.median(np.concatenate([grp[x]["region"] for x in pick]))
                 - np.median(np.concatenate([grp[x]["matched_A"] for x in pick])))
    lo, hi = np.percentile(d, [2.5, 97.5])
    check(close(p.d_rm_lo, lo) and close(p.d_rm_hi, hi),
          f"primary region-bootstrap interval recomputed: [{lo:+.4f}, {hi:+.4f}]")
    met = (p.d_rm > 0) and (p.d_rm_lo > 0)
    check(str(p.met) == str(met), f"met = {p.met} follows the interval rule")
    R = S[(S.scope == "region") & (S.metric == "vertex_spacing_m") & (S.stat == "median")]
    R500 = R[R.budget == 500]
    errs = 0
    for r in R500.itertuples():
        gg = grp[(int(r.seed), int(r.component))]
        errs += not (close(r.region, np.median(gg["region"]))
                     and close(r.matched_A, np.median(gg["matched_A"]))
                     and close(r.AE, np.median(gg["AE"])))
    check(errs == 0, f"per-region medians recomputed ({len(R500)} regions)")

    # panels: the lobe's own DFIRM
    on = (E.pan_fallback == 0) & (E.pan_n > 0)
    fb = E.pan_fallback == 1
    check(bool((E.pan_dfirm_id[on] == E.dfirm_id[on]).all())
          and bool((E.pan_dfirm_id[fb] != E.dfirm_id[fb]).all())
          and bool((E.pan_n <= E.pan_n_all).all())
          and bool(((E.pan_n_all == 0) == (E.pan_n == 0)).all()),
          f"panels: {int(on.sum()):,} lobe rows take a panel of their own DFIRM, {int(fb.sum())} fall "
          f"back to another DFIRM's, {int((E.pan_n_all == 0).sum())} lie on none")
    if a.log.exists():
        check_log(a.log.read_text(), len(want), E, fb)
    test_type_pooled_intervals(E5, S, regs, land)
    test_raw_layers(E5, a)
    if a.null_file and a.null_file.exists():
        N = pd.read_parquet(a.null_file)
        errs = 0
        for r in R500.itertuples():
            g = N[(N.seed == r.seed) & (N.component == r.component)]
            v = (g.med_spacing_region - g.med_spacing_matched).to_numpy()
            errs += not close(e5.mid_rank_pct(r.d_rm, v), r.null_pct, 1e-9)
        check(errs == 0, "per-region null percentiles of the primary recomputed from the null file")
        test_pooled_null(S, N, a.null_file, regs, land)


def check_stored_geometry(E):
    """Up to 150 lobes per lobe file: metrics recomputed from the stored lobe geometry."""
    rng = np.random.default_rng(0)
    bad = 0
    n = 0
    for f, g in E.groupby("lobe_file"):
        L = gpd.read_parquet(f).set_index(["component", "role", "lobe_id"])
        smp = g.iloc[rng.choice(len(g), min(150, len(g)), replace=False)]
        geo = L.geometry.reindex(list(zip(smp.component, smp.role, smp.lobe_id))).to_numpy()
        m = e5.part_metrics(np.array(geo, dtype=object))
        for c in ("n_vertices", "perimeter_m", "straight_m", "straight_vertex_m", "shape_index",
                  "seg_len_median_m"):
            same = np.isclose(m[c].to_numpy(float), smp[c].to_numpy(float), rtol=1e-7,
                              equal_nan=True)
            bad += int((~same).sum())
        n += len(smp)
    check(bad == 0, f"{n} sampled lobes: metrics recomputed from the stored geometry agree")


def check_log(text, n_regions, E, fb):
    """The log shows every observed lobes.py build rebuilt identically and the panel fallback
    count of E_polygons."""
    n_id = text.count("-> identical")
    check(n_id == n_regions and "DIFFERENT" not in text,
          f"log: {n_id} observed lobes.py component builds rebuilt identically")
    m = re.search(r"fallback to all panels\): region (\d+) of [\d,]+, matched_A (\d+) of [\d,]+, "
                  r"AE (\d+) of", text)
    got = tuple(int(x) for x in m.groups()) if m else None
    check(got == tuple(int(fb[E.role == r].sum()) for r in e5.ROLES),
          f"log: panel fallback count per role {got}")


def region_arrays(E5):
    """Per (seed, component) and role: the columns the recomputed statistics need."""
    out = {}
    for (s, c), g in E5.groupby(["seed", "component"]):
        out[(s, c)] = {}
        for r in e5.ROLES:
            x = g[g.role == r]
            out[(s, c)][r] = dict(
                sp=x.vertex_spacing_m.to_numpy(float), P=x.perimeter_m.to_numpy(float),
                n=x.n_vertices.to_numpy(float), eff=x.pan_eff_year.to_numpy(float),
                pre=x.pan_pre_present.to_numpy(float), pn=x.pan_n.to_numpy(float),
                scale=x.pan_scale.to_numpy(float), lomr=x.lomr_share.to_numpy(float),
                area=x.area_m2.to_numpy(float))
    return out


RECOMP = [("vertex_spacing_m", "median"), ("vertex_spacing_m", "agg"), ("pan_eff_year", "median"),
          ("pan_pre_present", "share"), ("pan_n", "median"), ("pan_scale", "median"),
          ("pan_scale_cat=24000", "pshare"), ("pan_scale_cat=12000", "pshare"),
          ("pan_scale_cat=6000", "pshare"), ("pan_scale_cat=24000", "share"),
          ("lomr_share", "mean"), ("lomr_share", "agg")]
COL = dict(vertex_spacing_m="sp", pan_eff_year="eff", pan_pre_present="pre", pan_n="pn",
           pan_scale="scale", lomr_share="lomr")


def mval(c, metric, stat):
    """One statistic of one role's pooled lobes (written independently of Columns.stat)."""
    if (metric, stat) == ("vertex_spacing_m", "agg"):
        return c["P"].sum() / max(c["n"].sum(), 1)
    if (metric, stat) == ("lomr_share", "agg"):
        return (c["lomr"] * c["area"]).sum() / c["area"].sum()
    if metric.startswith("pan_scale_cat="):
        level = float(metric.split("=")[1])
        v = np.where(np.isnan(c["scale"]), np.nan, (c["scale"] == level).astype(float))
        ok = np.isfinite(v)
        if not ok.any():
            return np.nan
        if stat == "share":
            return v[ok].mean()
        return (v[ok] * c["P"][ok]).sum() / c["P"][ok].sum()
    v = c[COL[metric]]
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan
    if stat == "median":
        return np.median(v)
    return v.mean()


def test_type_pooled_intervals(E5, S, regs, land):
    """Type-level and pooled estimates and region-bootstrap intervals of BOTH contrasts
    (region - matched_A, region - AE), recomputed from E_polygons."""
    G = region_arrays(E5)
    typ = {(int(r.seed), int(r.component)): r.type for r in land.itertuples()}
    keys = list(G[regs[0]]["region"])

    def pool(pick, role):
        return {k: np.concatenate([G[x][role][k] for x in pick]) for k in keys}

    def ests(pick):
        pooled = {r: pool(pick, r) for r in e5.ROLES}
        out = np.empty((len(RECOMP), len(e5.ROLES)))
        for j, (m, st) in enumerate(RECOMP):
            for i, r in enumerate(e5.ROLES):
                out[j, i] = mval(pooled[r], m, st)
        return out

    for scope, ty, key in (("pooled", "", [0]), ("type", "positive-dominated", [1, 0]),
                           ("type", "negative-dominated", [1, 1])):
        rs = [r for r in regs if not ty or typ[r] == ty]
        v0 = ests(rs)
        rng = np.random.default_rng([e5.BOOT_SEED, *key])
        B = []
        for _ in range(e5.N_BOOT):
            B.append(ests([rs[i] for i in rng.integers(0, len(rs), len(rs))]))
        B = np.array(B)
        dm, da = B[:, :, 0] - B[:, :, 1], B[:, :, 0] - B[:, :, 2]
        bad = []
        for j, (m, st) in enumerate(RECOMP):
            q = S[(S.scope == scope) & (S.subset == "all") & (S.metric == m) & (S.stat == st)
                  & (S.region_type.fillna("") == ty)]
            if len(q) != 1:
                bad.append(f"{m}/{st}: {len(q)} rows")
                continue
            q = q.iloc[0]
            (lo, hi), (alo, ahi) = pct(dm[:, j]), pct(da[:, j])
            ok = (nclose(q.region, v0[j, 0]) and nclose(q.matched_A, v0[j, 1])
                  and nclose(q.AE, v0[j, 2])
                  and nclose(q.d_rm_lo, lo) and nclose(q.d_rm_hi, hi)
                  and nclose(q.d_rAE_lo, alo) and nclose(q.d_rAE_hi, ahi))
            if not ok:
                bad.append(f"{m}/{st}")
        check(not bad, f"{scope} {ty or 'all'} ({len(rs)} regions): estimates and region-bootstrap "
              f"intervals of region - matched_A and region - AE recomputed for {len(RECOMP)} metrics "
              f"(primary, panels, LOMR) {bad if bad else ''}")


# ------------------------------------------------------------------ 7. raw layers, no o5 code
def raw_lobe_sample(E5):
    """About 100 distinct lobes: 20 per role, 20 with LOMR overlap, 20 that also overlap another
    DFIRM's panels, 10 on the panel fallback; and their stored geometry."""
    U = E5.drop_duplicates(["src_row_id", "part"]).reset_index(drop=True)
    rng = np.random.default_rng(7)

    def take(mask, k):
        i = np.flatnonzero(mask.to_numpy())
        if not len(i):
            return i
        return i[rng.choice(len(i), min(k, len(i)), replace=False)]

    picks = [take(U.role == r, 20) for r in e5.ROLES]
    picks.append(take(U.lomr_share > 0, 20))
    picks.append(take((U.pan_n_all > U.pan_n) & (U.pan_fallback == 0), 20))
    picks.append(take(U.pan_fallback == 1, 10))
    X = U.iloc[np.unique(np.concatenate(picks))].reset_index(drop=True)
    geo = np.empty(len(X), dtype=object)
    for f, g in X.groupby("lobe_file"):
        L = gpd.read_parquet(f).set_index(["component", "role", "lobe_id"]).geometry
        geo[g.index.to_numpy()] = L.reindex(list(zip(g.component, g.role, g.lobe_id))).to_numpy()
    return X, geo


def raw_panels(path):
    """pan_FL in EPSG:5070 with each panel's DFIRM, FIRM_PAN, decimal effective year, pre-date
    flag and scale."""
    pan = gpd.read_parquet(path).to_crs(5070).reset_index(drop=True)
    geoms = shapely.make_valid(pan.geometry.to_numpy())
    eff = pd.to_datetime(pan.EFF_DATE, utc=True)
    pre = pd.to_datetime(pan.PRE_DATE, utc=True)
    firm_pan = pan.FIRM_PAN.to_numpy(dtype=object)
    return dict(
        geoms=geoms, tree=shapely.STRtree(geoms), dfirm=pan.DFIRM_ID.to_numpy(dtype=object),
        firm_pan=firm_pan, row={k: i for i, k in enumerate(firm_pan)},
        year=np.where(eff.dt.year < 9999, eff.dt.year + (eff.dt.dayofyear - 1) / 365.25, np.nan),
        pre=(pre.notna() & (pre.dt.year < 9999)).to_numpy(float),
        scale=pd.to_numeric(pan.SCALE, errors="coerce").to_numpy(float))


def panel_ok(r, g, pan):
    """Whether lobe row r has the own-DFIRM panel values recomputed from the raw panels; and
    whether, among all panels, its largest overlap is another DFIRM's."""
    j = pan["tree"].query(g, predicate="intersects")
    ar = shapely.area(shapely.intersection(g, pan["geoms"][j])) if len(j) else np.zeros(0)
    k = ar > 1.0
    j, ar = j[k], ar[k]
    own = pan["dfirm"][j] == r.dfirm_id
    use = own if own.any() else np.ones(len(j), bool)
    ju, au = j[use], ar[use]
    other = False
    if len(j):
        other = pan["dfirm"][j[np.argmax(ar)]] != r.dfirm_id
    ok =(int(r.pan_n) == len(ju) and int(r.pan_n_all) == len(j)
          and int(r.pan_fallback) == int(len(j) > 0 and not own.any()))
    if not len(ju):
        return ok and pd.isna(r.pan_firm_pan), other
    dominant = set(pan["firm_pan"][ju[au >= au.max() * (1 - 1e-9)]])
    q = pan["row"].get(r.pan_firm_pan, -1)
    ok &= (r.pan_firm_pan in dominant and q >= 0 and r.pan_dfirm_id == pan["dfirm"][q]
           and (r.pan_dfirm_id == r.dfirm_id) == bool(own.any())
           and nclose(r.pan_scale, pan["scale"][q]) and nclose(r.pan_eff_year, pan["year"][q])
           and nclose(r.pan_pre_present, pan["pre"][q]))
    return ok, other


def lomr_ok(r, g, lomr):
    """Whether lobe row r has the LOMR shares and count recomputed from the raw LOMRs."""
    j = lomr["tree"].query(g, predicate="intersects")
    area = shapely.area(g)
    share = 0.0
    if len(j):
        share = shapely.area(shapely.intersection(g, shapely.union_all(lomr["geoms"][j]))) / area
    je = j[lomr["effective"][j]]
    share_eff = 0.0
    if len(je):
        covered = shapely.intersection(g, shapely.union_all(lomr["geoms"][je]))
        share_eff = shapely.area(covered) / area
    return (abs(r.lomr_share - min(share, 1.0)) < 1e-9
            and abs(r.lomr_share_eff - min(share_eff, 1.0)) < 1e-9 and int(r.lomr_n) == len(j))


def attrs_ok(r, attrs):
    """Whether lobe row r has the STUDY_TYP, SOURCE_CIT and source kind of its attrs row."""
    try:
        w = attrs.loc[(r.dfirm_id, r.fld_ar_id)]
        study, cit = w.STUDY_TYP, w.SOURCE_CIT
    except KeyError:
        study = cit = None
    kind = None
    if isinstance(cit, str):
        kind = re.sub(r"\d+$", "", re.sub(r"^\d+[A-Z]?_", "", cit))
    return (txt(r.study_typ) == txt(study) and txt(r.source_cit) == txt(cit)
            and txt(r.source_kind) == txt(kind))


def test_raw_layers(E5, a):
    """About 100 lobes: own-DFIRM panel, LOMR shares, STUDY_TYP / SOURCE_CIT from the raw
    layers, without o5_polygons code."""
    X, geo = raw_lobe_sample(E5)
    pan = raw_panels(a.panels)
    lo = gpd.read_parquet(a.lomr).to_crs(5070).reset_index(drop=True)
    lomr_geoms = shapely.make_valid(lo.geometry.to_numpy())
    lomr = dict(geoms=lomr_geoms, tree=shapely.STRtree(lomr_geoms),
                effective=(lo.STATUS == "Effective").to_numpy())
    at = pd.read_parquet(a.attrs, columns=["DFIRM_ID", "FLD_AR_ID", "STUDY_TYP", "SOURCE_CIT",
                                           "STATE"])
    at = at[at.STATE == "FL"].drop_duplicates(["DFIRM_ID", "FLD_AR_ID"])
    at = at.set_index(["DFIRM_ID", "FLD_AR_ID"])
    bad_p, bad_l, bad_a, other = [], [], [], 0
    for i, r in X.iterrows():
        ok, other_dominant = panel_ok(r, geo[i], pan)
        other += other_dominant
        if not ok:
            bad_p.append(r.lobe_id)
        if not lomr_ok(r, geo[i], lomr):
            bad_l.append(r.lobe_id)
        if not attrs_ok(r, at):
            bad_a.append(r.lobe_id)
    n_lomr = int((X.lomr_share > 0).sum())
    check(not bad_p, f"{len(X)} lobes from the raw pan_FL: own-DFIRM dominant panel, pan_n, pan_n_all, "
          f"fallback, scale, date, pre-date ({other} of them would take another DFIRM's panel among "
          f"all panels) {bad_p[:5] if bad_p else ''}")
    check(not bad_l, f"{len(X)} lobes from the raw s_lomr_FL ({n_lomr} with LOMR overlap): lomr_share, "
          f"lomr_share_eff, lomr_n {bad_l[:5] if bad_l else ''}")
    check(not bad_a, f"{len(X)} lobes from the raw nfhl_haz_attrs_10state on (DFIRM_ID, FLD_AR_ID): "
          f"STUDY_TYP, SOURCE_CIT, source_kind {bad_a[:5] if bad_a else ''}")


# ------------------------------------------------------------------ 7. the null, with --null-file
def test_pooled_null(S, N, null_file, regs, land):
    """The pooled and type null: draws recomputed from the per-patch lobe spacings (primary), and
    every metric's percentile and summary from the saved draws."""
    typ = {(int(r.seed), int(r.component)): r.type for r in land.itertuples()}
    pth = null_file.with_name(null_file.stem + "_pooled.parquet")
    if "spacing_region" not in N or not pth.exists():
        check(False, f"null file has per-patch spacings and {pth.name} exists")
        return
    ok = True
    for x, m in zip(N.spacing_region, N.med_spacing_region):
        if len(x):
            ok = ok and close(np.median(x), m, 1e-12)
    check(ok, "per-patch spacings reproduce the per-patch medians")
    NP = pd.read_parquet(pth)
    NP["region_type"] = NP.region_type.fillna("")
    by = {(int(s), int(c)): g.set_index("draw") for (s, c), g in N.groupby(["seed", "component"])}
    for scope, ty in (("pooled", ""), ("type", "positive-dominated"), ("type", "negative-dominated")):
        rs = [r for r in regs if not ty or typ[r] == ty]
        draws = sorted(set.intersection(*[set(by[r].index) for r in rs]))
        v = []
        for d in draws:
            region = np.concatenate([by[r].at[d, "spacing_region"] for r in rs])
            matched = np.concatenate([by[r].at[d, "spacing_matched"] for r in rs])
            v.append(np.median(region) - np.median(matched))
        v = np.array(v)
        k = NP[(NP.scope == scope) & (NP.region_type == ty) & (NP.metric == "vertex_spacing_m")
               & (NP.stat == "median")].sort_values("draw")
        check(len(k) == len(draws) and np.allclose(k.d_rm.to_numpy(), v, rtol=0, atol=1e-9),
              f"{scope} {ty or 'all'}: {len(draws)} null draws of the primary recomputed by pooling "
              f"the d-th patch of all {len(rs)} regions")
    Sx = S[S.scope.isin(["pooled", "type"]) & (S.subset == "all")].copy()
    Sx["region_type"] = Sx.region_type.fillna("")
    grp = {}
    for key, g in NP.groupby(["scope", "region_type", "metric", "stat"]):
        grp[key] = g.sort_values("draw").d_rm.to_numpy()
    bad, n = [], 0
    for r in Sx.itertuples():
        v = grp.get((r.scope, r.region_type, r.metric, r.stat))
        if v is None:
            bad.append(f"{r.scope} {r.metric}/{r.stat}: no draws")
            continue
        f = v[np.isfinite(v)]
        n += 1
        if len(f):
            summary = (np.median(f), np.percentile(f, 2.5), np.percentile(f, 97.5))
        else:
            summary = (np.nan, np.nan, np.nan)
        ok = (nclose(r.null_pct, midrank(r.d_rm, v)) and int(r.null_n) == len(f)
              and nclose(r.null_median, summary[0]) and nclose(r.null_lo, summary[1])
              and nclose(r.null_hi, summary[2]))
        if not ok:
            bad.append(f"{r.scope} {r.region_type} {r.metric}/{r.stat}")
    check(not bad and n == len(Sx), f"{n} pooled and type rows: null percentile, n, median and "
          f"2.5 / 97.5 bounds recomputed from the saved draws {bad[:5] if bad else ''}")
    P = Sx[(Sx.scope == "pooled") & Sx.null_pct.notna()]
    nd = P[P.null_lo < P.null_hi]
    beyond = nd[(nd.null_pct < 2.5) | (nd.null_pct > 97.5)]
    edge = nd[(nd.null_pct == 2.5) | (nd.null_pct == 97.5)]
    print(f"  info  pooled rows with a null: {len(P)}, {len(nd)} with a non-degenerate null "
          f"(2.5% bound < 97.5% bound); {len(beyond)} with null_pct strictly below 2.5 or above "
          f"97.5, {len(edge)} exactly at 2.5 or 97.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygons", type=Path, default=Path("results/2026-10-17/o5/E_polygons.csv"))
    ap.add_argument("--summary", type=Path, default=Path("results/2026-10-17/o5/E_summary.csv"))
    ap.add_argument("--log", type=Path, default=Path("logs/2026-10-17/E_polygons.log"))
    ap.add_argument("--null-file", type=Path)
    ap.add_argument("--panels", type=Path, default=Path("data/interim/nfhl_states/pan_FL.parquet"))
    ap.add_argument("--lomr", type=Path, default=Path("data/interim/nfhl_states/s_lomr_FL.parquet"))
    ap.add_argument("--attrs", type=Path, default=Path("data/interim/nfhl_haz_attrs_10state.parquet"))
    a = ap.parse_args()
    test_ring_runs()
    test_part_metrics()
    test_explode()
    test_match()
    test_rule()
    test_small()
    test_panels()
    test_summarise()
    test_outputs(a)
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
