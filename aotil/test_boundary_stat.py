#!/usr/bin/env python3
"""
Checks for boundary_stat.py on synthetic lobes; exit code 0 = all passed. From the repo root:

    python aotil/test_boundary_stat.py

1. On a line: a 300 m square lobe lying on a synthetic soil line is on S along all its edge;
   moved 500 m through the null's own code (NullContext.draw -> edge_layers.measure) it is not.
2. Level set: a DEM flat along the edge and rising across it makes every window a level set;
   tilting it along the edge breaks the range test but not the detrended one.
3. Windows: 10 vertices each, < 8 usable drops one, and only the preview's rule spans a gap.
4. The detrended range of a linear ramp is 0.
5. Self-check: a zero displacement gives back the distances edge_layers.measure stored.
6. Helpers: NaN distance = no line, M = min of its layers, the mid-rank null percentile with
   ties a few ulp apart (boundary_stat.py decision 8).
7. Sensitivities (decision 10): testable windows only, and the within-component difference.
8. Covered null, paired (decision 6).
9. --calibrate-only --overwrite after a full run deletes the stale null.csv.
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boundary_stat as bs  # noqa: E402
import edge_layers as el  # noqa: E402

X0, Y0 = 1_400_000.0, 600_000.0          # somewhere in EPSG:5070 Florida
FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def square_ring(x0, y0, side, step=30.0):
    """Counter-clockwise square ring (interior on the left), densified like lobes.py:
    n = ceil(L / step), s = k L / n. Returns s, x, y and the unit inward normal."""
    L = 4 * side
    n = int(np.ceil(L / step))
    s = np.arange(n) * L / n
    side_i = np.minimum((s // side).astype(int), 3)
    t = s - side_i * side
    on_side = [side_i == 0, side_i == 1, side_i == 2, side_i == 3]
    x = np.select(on_side, [x0 + t, x0 + side, x0 + side - t, x0])
    y = np.select(on_side, [y0, y0 + t, y0 + side, y0 + side - t])
    tx = np.select(on_side, [1, 0, -1, 0])
    ty = np.select(on_side, [0, 1, 0, -1])
    return s, x, y, -ty.astype(float), tx.astype(float)      # left perpendicular = inward


def dem(poly, slope_along=0.0):
    """z(x, y): 10 m on the polygon's edge, rising 1 cm per metre outward and falling inward
    (signed distance), plus an optional tilt along x."""
    def z(x, y):
        p = shapely.points(x, y)
        d = shapely.distance(poly.boundary, p)
        inside = shapely.contains(poly, p)
        return 10.0 + 0.01 * np.where(inside, -d, d) + slope_along * (x - X0)
    return z


def line_sets(soil_geom):
    """The four null layers: soil = the given geometry (component 0); county and panel edges
    20 km away, so only S decides."""
    far = shapely.linestrings([[X0 - 20_000, Y0 - 20_000], [X0 - 20_000, Y0 + 20_000]])

    def one_id():
        return pd.Series([0], dtype="Int64")

    def far_line():
        return np.array([far], dtype=object)

    return {"soil": el.LineSet("soil", np.array([soil_geom], dtype=object), one_id(),
                               group=np.array([0]), by="component"),
            "county": el.LineSet("county", far_line(), one_id()),
            "panel_interior": el.LineSet("panel_interior", far_line(), one_id(),
                                         group=np.array(["D1"]), by="dfirm_id"),
            "panel_exterior": el.LineSet("panel_exterior", far_line(), one_id(),
                                         group=np.array(["D1"]), by="dfirm_id")}


def lobe_table(slope_along=0.0, soil_geom=None):
    """A one-lobe region vertex table shaped like edge_layers' output (a 300 m square), with its
    distances measured by edge_layers.measure. Also returns the four null layers."""
    side = 300.0
    probe = 100.0                         # m, the DEM probes inside and outside the edge
    s, x, y, nx, ny = square_ring(X0, Y0, side)
    poly = shapely.box(X0, Y0, X0 + side, Y0 + side)
    z = dem(poly, slope_along)
    n = len(s)
    xi, yi = x + probe * nx, y + probe * ny
    xo, yo = x - probe * nx, y - probe * ny
    v = pd.DataFrame({
        "lobe_id": "c0_region_0_0", "zone": "A", "role": "region", "component": 0, "ring": 0,
        "s": s, "x": x, "y": y, "dfirm_id": "D1",
        "contact_row_id": pd.array([pd.NA] * n, dtype="Int64"), "dem_ok": True,
        "outside_state": False, "outside_own_panels": False,
        "in_probe_inside": shapely.contains(poly, shapely.points(xi, yi)),
        "out_probe_outside": ~shapely.contains(poly, shapely.points(xo, yo)),
        "soil_ok": True, "z": z(x, y), "z_in": z(xi, yi), "z_out": z(xo, yo)})
    sets = line_sets(poly.boundary if soil_geom is None else soil_geom)
    pts = v[["x", "y", "component"]].assign(dfirm_id=pd.array(v.dfirm_id, dtype="string"))
    for name, ls in sets.items():
        v[f"d_{name}_m"] = el.measure(pts, ls)[f"d_{name}_m"]
    v["d_canals_m"] = np.nan
    return bs.index_vertices(v), sets


def windows_of(v, test="range"):
    """The ring-rule windows of v at the default tau, with each vertex's window and slot."""
    usable = bs.usable_mask(bs.drop_flags(v))
    return bs.build_windows(v, usable, "ring", test, bs.TAU)


def null_context(v, W, wid, slot, sets):
    """A NullContext whose bbox, map units, state and panels all cover a 10 km box."""
    bb = pd.DataFrame({"component": [0], "minx": [X0 - 5000], "miny": [Y0 - 5000],
                       "maxx": [X0 + 5000], "maxy": [Y0 + 5000]}).set_index("component")
    big = shapely.box(X0 - 5000, Y0 - 5000, X0 + 5000, Y0 + 5000)
    cover = {"mupolygons": {0: np.array([big], dtype=object)}, "state": big, "panels": {"D1": big}}
    return bs.NullContext(v, wid, slot, W, sets, bb, cover, ["level"])


def region_row(cal, stat):
    return cal[(cal.stat == stat) & (cal.role == "region")].iloc[0]


def max_stored_diff(r, ctx):
    """Largest |recomputed - stored| distance of a draw r over the four null layers."""
    diffs = []
    for name in bs.NULL_LAYERS:
        diffs.append(float(np.nanmax(np.abs(r["dist"][name] - ctx.stored[name]))))
    return max(diffs)


# ---------------------------------------------------------------- 1. on a line
print("1. on a line: square lobe on a soil line, and displaced 500 m")
v, sets = lobe_table()
W, wid, slot = windows_of(v)
check(len(W) == 4 and (W.n_usable == 10).all(),
      f"300 m square: 40 vertices -> 4 full windows (got {len(W)})")
check(np.allclose(v.d_soil_m, 0.0, atol=1e-9), "every vertex on the soil line (d_soil_m = 0)")
on = bs.pooled_share(W, W.on_S_d100)["region"]
check(on == 1.0, f"on-line share (S, d = 100 m) = 1 (got {on})")
check(bs.pooled_share(W, W.on_SM_d30)["region"] == 1.0, "on-line share (S or M, d = 30 m) = 1")
check(bs.pooled_share(W, W.on_M_d100)["region"] == 0.0, "M lines 20 km away: on-M share 0")
ctx = null_context(v, W, wid, slot, sets)
for dx, dy, what in ((500.0, 0.0, "east"), (500 / np.sqrt(2), 500 / np.sqrt(2), "north-east")):
    r = ctx.draw(np.array([dx]), np.array([dy]))
    check(r["u"][100.0]["onS"] == 0.0 and r["u"][30.0]["onS"] == 0.0,
          f"translated 500 m {what} through NullContext.draw: on-line share 0 at d = 100 and 30 m "
          f"(got {r['u'][100.0]['onS']}, {r['u'][30.0]['onS']})")
    check(np.nanmin(r["dist"]["soil"]) > 50.0,
          f"  ... min displaced soil distance {np.nanmin(r['dist']['soil']):.1f} m > 50 m")
    n_uncovered = sum(r["cov"][k] for k in ("outside_bbox", "no_map_unit", "outside_state",
                                            "outside_own_panels"))
    check(n_uncovered == 0, "  ... every displaced vertex inside the synthetic coverage")
v_off, _ = lobe_table(soil_geom=shapely.box(X0 + 500, Y0, X0 + 800, Y0 + 300).boundary)
W_off, _, _ = windows_of(v_off)
check(bs.pooled_share(W_off, W_off.on_S_d100)["region"] == 0.0,
      "the same square with the soil polygon 500 m east (measured directly): share 0")

# ---------------------------------------------------------------- 2. level set
print("2. level set: flat along the edge, rising across it")
good = (v.in_probe_inside & v.out_probe_outside).to_numpy()
check(good.sum() == 36 and (v.z_in[good] < v.z[good]).all() and (v.z[good] < v.z_out[good]).all(),
      "synthetic DEM: z_in < z < z_out wherever both probes land right (all but the 4 corners)")
check(np.allclose(v.z, 10.0), "flat along the edge (z = 10 m)")
for t in bs.TESTS:
    Wt, _, _ = windows_of(v, test=t)
    share = bs.pooled_share(Wt, Wt.level)["region"]
    check(share == 1.0, f"{t} <= 0.76 m + drop: level-set share 1 (got {share})")
check(bs.pooled_share(W, W.U_d100)["region"] == 0.0, "level set on the soil line: U = 0")
# 0.5 cm/m is 1.5 m over a 300 m side but less than the 1 cm/m outward rise, so the drop stays > 0
v_tilt, _ = lobe_table(slope_along=0.005)
W_tilt, _, _ = windows_of(v_tilt)
check(bs.pooled_share(W_tilt, W_tilt["level_range_t0.76"])["region"] < 1.0,
      "tilted 0.5 cm/m along x: the range test fails on the sloping sides")
check(bs.pooled_share(W_tilt, W_tilt["level_detrended_t0.76"])["region"] == 1.0,
      "tilted: the detrended test still passes every window (each window is one straight side)")
check(bs.pooled_share(W_tilt, W_tilt.U_d100)["region"] > 0.0,
      "tilted on the soil line with the range test: U > 0 (not a level set AND on S)")
v_rev = v.copy()                          # ground falls going out: a ridge, not a floodplain
v_rev["z_in"] = v.z_out
v_rev["z_out"] = v.z_in
W_rev, _, _ = windows_of(v_rev)
check(bs.pooled_share(W_rev, W_rev.level)["region"] == 0.0,
      "drop reversed (z_out < z_in): no level set")
check(bs.pooled_share(W_rev, W_rev.U_d100)["region"] == 1.0,
      "  ... and then U = 1 on the soil line")

# ---------------------------------------------------------------- 3. windows
print("3. windows: 10 vertices each, < 8 usable dropped")
rid = np.zeros(25, dtype=np.int64)
k = np.arange(25)
u = np.ones(25, dtype=bool)
wid, slot, wr, wl = bs.cut_windows(rid, k, u)
check(len(wr) == 2 and (wid[:20] == np.repeat([0, 1], 10)).all() and (wid[20:] == -1).all(),
      "25 vertices -> windows k 0-9 and 10-19; the 5-vertex tail is dropped")
check((slot[:20] == np.tile(np.arange(10), 2)).all(), "slots 0..9 within each window")
u2 = u.copy()
u2[[3, 4]] = False
wid2, _, _, _ = bs.cut_windows(rid, k, u2)
check((wid2[[0, 1, 2, 5, 6, 7, 8, 9]] == 0).all() and (wid2[[3, 4]] == -1).all(),
      "2 unusable vertices: window 0 kept with its 8 usable vertices")
u3 = u.copy()
u3[[3, 4, 5]] = False
wid3, _, wr3, wl3 = bs.cut_windows(rid, k, u3)
check((wid3[:10] == -1).all() and len(wr3) == 1 and wl3[0] == 1,
      "3 unusable vertices: window 0 has 7 < 8 and is dropped; only k 10-19 is left")
widf, _, wrf, _ = bs.cut_windows(rid, k, u3, mode="filtered")
check(len(wrf) == 2 and widf[2] == 0 and widf[6] == 0 and widf[12] == 0,
      "preview rule: the 22 usable vertices are re-cut, so window 0 spans the gap (k 0-2 and 6-12)")
rid4 = np.r_[np.zeros(9, int), np.ones(10, int)]
k4 = np.r_[np.arange(9), np.arange(10)]
wid4, _, wr4, _ = bs.cut_windows(rid4, k4, np.ones(19, bool))
check(len(wr4) == 2 and (wid4[:9] == 0).all() and (wid4[9:] == 1).all(),
      "windows never cross rings; a 9-vertex ring is one window of 9")
check(np.isclose(W.edge_m.sum(), 1200.0) and np.allclose(W.ds_m, 30.0),
      "edge length = usable vertices x ring spacing (4 x 300 m)")

# ---------------------------------------------------------------- 4. detrended range
print("4. detrended range")
S = np.tile(np.arange(10) * 30.0, (3, 1))
Z = 5.0 + np.array([[0.0], [0.02], [-0.05]]) * S
check(np.allclose(bs.detrended_range(S, Z), 0.0, atol=1e-9), "linear ramps (3 slopes): 0")
Z2 = Z.copy()
Z2[1, [2, 7]] = np.nan
check(np.allclose(bs.detrended_range(S, Z2), 0.0, atol=1e-9), "linear ramp with empty slots: 0")
Z3 = Z.copy()
Z3[0, 5] += 1.0
check(bs.detrended_range(S, Z3)[0] > 0.5, "a 1 m bump in a flat window is not removed")

# ---------------------------------------------------------------- 5. self-check
print("5. zero-displacement self-check")
zero = np.zeros(len(ctx.lobes))
r0 = ctx.draw(zero, zero)
diff = max_stored_diff(r0, ctx)
check(diff == 0.0, f"zero displacement reproduces d_soil/county/panel_interior/panel_exterior "
                   f"(max |diff| {diff})")
check(r0["u"][100.0]["onS"] == 1.0, "zero displacement: on-line share back to 1")
v2, sets2 = lobe_table(soil_geom=shapely.linestrings([[X0 - 50, Y0 - 70], [X0 + 410, Y0 + 260]]))
W2, wid2, slot2 = windows_of(v2)
ctx2 = null_context(v2, W2, wid2, slot2, sets2)
r2 = ctx2.draw(np.zeros(1), np.zeros(1))
d2 = max_stored_diff(r2, ctx2)
check(d2 <= bs.SELF_TOL_M, f"same on a diagonal soil line crossing the lobe (max |diff| {d2:.1e})")

# ---------------------------------------------------------------- 6. helpers
print("6. helpers")
Sv, Mv = bs.s_and_m({"soil": np.array([np.nan, 5.0]), "county": np.array([50.0, np.nan]),
                     "panel_interior": np.array([np.nan, np.nan]),
                     "panel_exterior": np.array([20.0, np.nan])})
check(np.isinf(Sv[0]) and Sv[1] == 5.0, "NaN soil distance -> +inf (no line there)")
check(Mv[0] == 20.0 and np.isinf(Mv[1]),
      "M = min of county / panel interior / exterior; all NaN -> +inf")
check(bs.null_percentile(5.0, np.arange(10.0))[0] == 55.0,
      "mid-rank percentile: 5 among 0..9 -> 55")
check(bs.null_percentile(100.0, np.arange(10.0)) == (100.0, 1 / 11),
      "above every draw: 100, p = 1/11")
obs = 0.3
below1 = np.nextafter(obs, 0.0)
below2 = np.nextafter(below1, 0.0)
pct, p = bs.null_percentile(obs, np.array([below1, below2, 0.1, 0.5]))
check(below1 != obs and pct == 50.0 and p == 4 / 5,
      f"null 1 and 2 ulp below obs are ties: pct 50 (exact == would say 75), p 4/5 (got {pct}, {p})")
pct, p = bs.null_percentile(obs, np.array([np.nextafter(obs, 1.0), 0.1]))
check(pct == 75.0 and p == 2 / 3,
      f"null 1 ulp above obs and 0.1: one tie, one below -> pct 75 (exact > would say 50), p 2/3 "
      f"(got {pct}, {p})")
noisy = 0.25 + np.array([0.0, 1, -1, 2, -2, 0, 3, -3, 1, -1]) * np.spacing(0.25)
pct, p = bs.null_percentile(0.25, noisy)
check(pct == 50.0 and p == 1.0, f"all 10 draws tie (within a few ulp): pct 50, p 1 (got {pct}, {p})")
s1 = np.sum(np.array([0.1, 0.2, 0.3, 0.4]) * 7.0) / 10.0      # the same value, two summation orders
s2 = (0.4 * 7.0 + 0.3 * 7.0 + 0.2 * 7.0 + 0.1 * 7.0) / 10.0
check(bs.null_percentile(s1, np.array([s2]))[0] == 50.0,
      f"the same share summed in two orders ({s1!r} vs {s2!r}) is a tie")
check(bs.null_percentile(0.5, np.array([0.5 - 1e-7])) == (100.0, 0.5)
      and bs.null_percentile(0.5, np.array([0.5 + 1e-7])) == (0.0, 1.0),
      "1e-7 apart (one window's weight in a large component) is NOT a tie")

# ---------------------------------------------------------------- 7. sensitivities
print("7. sensitivities: testable windows only, within-component difference")
# window 0 (k 0-9): the inside probes of k 1-5 fail too; with the corner k 0 that leaves 4 good probes
v_np = v.copy()
v_np.loc[1:5, "in_probe_inside"] = False
W_np, wid_np, slot_np = windows_of(v_np)
check(int(W_np.n_drop.iloc[0]) == 4 and (W_np.n_drop.iloc[1:] >= 5).all(),
      f"window 0 has 4 good probes, the others >= 5 (n_drop {W_np.n_drop.tolist()})")
check(not W_np.level.iloc[0] and W_np.level.iloc[1:].all() and W_np.U_d100.iloc[0],
      "  ... so window 0 is no level set (untestable) and counts in U on the soil line")
check(bs.pooled_share(W_np, W_np.U_d100)["region"] == 0.25, "U over all windows = 1/4")
Wt_np = W_np[W_np.n_drop >= bs.MIN_DROP]
check(bs.pooled_share(Wt_np, Wt_np.U_d100)["region"] == 0.0, "U over testable windows only = 0")
ctx_np = null_context(v_np, W_np, wid_np, slot_np, sets)
r_np = ctx_np.draw(np.zeros(1), np.zeros(1))
check(r_np["u"][100.0]["level"] == 0.25 and r_np["u_testable"][100.0]["level"] == 0.0,
      f"the null's code at zero displacement: U 1/4, testable-only U 0 "
      f"(got {r_np['u'][100.0]['level']}, {r_np['u_testable'][100.0]['level']})")
cal = bs.calibration_table(W_np)
check(region_row(cal, "drop_n").share_edge == 0.75 and "testable" in region_row(cal, "drop_n").test,
      "calibration: testable share of edge 3/4, labelled 'testable'")
check(region_row(cal, "drop_sign").share_edge == 1.0
      and region_row(cal, "drop_sign_probed").share_edge == 1.0
      and region_row(cal, "drop_sign_probed").n_windows == 4,
      "calibration: median drop > 0, over all windows and over windows with >= 1 good probe")
v_nop = v.copy()
v_nop.loc[0:9, "in_probe_inside"] = False                       # window 0: no good probe at all
cal0 = bs.calibration_table(windows_of(v_nop)[0])
check(region_row(cal0, "drop_sign").share_edge == 0.75
      and region_row(cal0, "drop_sign_probed").share_edge == 1.0
      and region_row(cal0, "drop_sign_probed").n_windows == 3,
      "a window with no good probe fails 'median drop > 0' but is left out of the '>= 1 good "
      "probe' row")
# within-component: component 0 has region U .5 on 100 m, matched_A .1 on 300 m, AE .5 on 10 m;
# component 1 has region U .2 on 300 m, matched_A .4 on 100 m, no AE
num = pd.DataFrame({"region": [50.0, 60.0], "matched_A": [30.0, 40.0], "AE": [5.0, 0.0]})
den = pd.DataFrame({"region": [100.0, 300.0], "matched_A": [300.0, 100.0], "AE": [10.0, 0.0]})
wc = bs.within_component(num, den)
pooled = num.sum() / den.sum()
check(np.isclose(pooled["region"] - pooled["matched_A"], 0.1)
      and np.isclose(wc["diff_matched_A"][0], -0.05),
      f"pooled region - matched_A +0.10 but within-component (100 x 0.4 + 300 x -0.2) / 400 = -0.05 "
      f"(got {wc['diff_matched_A'][0]:+.4f})")
check(np.isclose(wc["diff_AE"][0], 0.0),
      "region - AE: component 1 has no AE edge and is left out -> 0")
wb = bs.within_component(num, den, np.array([[0, 0], [1, 1], [0, 1]]))
check(np.allclose(wb["diff_matched_A"], [0.4, -0.2, -0.05])
      and np.isclose(wb["diff_AE"][0], 0.0) and np.isnan(wb["diff_AE"][1])
      and np.isclose(wb["diff_AE"][2], 0.0),
      f"bootstrap draws [0,0], [1,1], [0,1]: {wb['diff_matched_A']}, AE {wb['diff_AE']} "
      f"(NaN: no AE drawn)")
num_same = pd.DataFrame({"region": [50.0, 60.0], "matched_A": [10.0, 120.0], "AE": [5.0, 6.0]})
den_same = pd.DataFrame({"region": [100.0, 300.0], "matched_A": [100.0, 300.0], "AE": [10.0, 30.0]})
same = bs.within_component(num_same, den_same)
check(np.isclose(same["diff_matched_A"][0], 110 / 400 - 130 / 400),
      "control with the region's own mix of components: within-component = pooled difference")

# ---------------------------------------------------------------- 8. covered null, paired
print("8. covered-windows null, paired")
check(ctx_np.u_observed(np.ones(4, bool))[100.0] == 0.25
      and ctx_np.u_observed(np.array([False, True, True, True]))[100.0] == 0.0
      and ctx_np.u_observed(np.array([True, True, False, False]))[100.0] == 0.5,
      "observed U over the kept windows: all 1/4; window 0 left out 0; windows 0-1 only 1/2")
check(r_np["u_covered_obs"][100.0] == 0.25 and r_np["cov"]["windows_touched"] == 0,
      "zero displacement, full coverage: no window touched, paired observed = observed U")
# a hole in the state polygon at the corner (X0, Y0) puts window 0's first vertex outside the state
state_with_hole = shapely.box(X0 - 5000, Y0 - 5000, X0 + 5000, Y0 + 5000).difference(
    shapely.box(X0 - 100, Y0 - 100, X0 + 100, Y0 + 100))
ctx_cut = bs.NullContext(v_np, wid_np, slot_np, W_np, sets, ctx_np.bb,
                         dict(ctx_np.cover, state=state_with_hole), ["level"])
r_cut = ctx_cut.draw(np.zeros(1), np.zeros(1))
check(r_cut["cov"]["windows_touched"] >= 1 and r_cut["u_covered_obs"][100.0] < 0.25,
      f"a hole in the state polygon at window 0: {r_cut['cov']['windows_touched']} window(s) touched, "
      f"paired observed U {r_cut['u_covered_obs'][100.0]:.3f} drops the U window, like the null")

# ---------------------------------------------------------------- 9. calibrate-only
print("9. --calibrate-only --overwrite after a full run")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    parts = []
    for i, role in enumerate(bs.ROLES):            # the square once per role, 2 km apart
        p = v.drop(columns=["rid", "k", "ds", "dup_lobe", "is_floodway"]).copy()
        p["lobe_id"] = f"c0_{role}_{i}"
        p["role"] = role
        p["zone"] = "AE" if role == "AE" else "A"
        p["x"] = p.x + 2000.0 * i
        parts.append(p)
    pd.concat(parts, ignore_index=True).to_parquet(td / "synth_vertex_layers.parquet")
    pd.DataFrame({"lobe_id": [p.lobe_id.iloc[0] for p in parts], "component": 0,
                  "role": list(bs.ROLES), "zone": ["A", "A", "AE"], "dfirm_id": "D1",
                  "src_row_id": [0, 1, 2], "part": 0}).to_parquet(td / "synth_lobes.parquet")
    pd.DataFrame({"component": [0], "minx": [X0 - 5000], "miny": [Y0 - 5000], "maxx": [X0 + 9000],
                  "maxy": [Y0 + 5000]}).to_parquet(td / "synth_bboxes.parquet")
    out = td / "out"
    out.mkdir()
    stale = out / "synth_null.csv"
    stale.write_text("draw,U_region_d100\n0,0.5\n")             # left by an earlier full run
    with contextlib.redirect_stdout(io.StringIO()) as log:
        rc = bs.main(["--vertex-layers", str(td / "synth_vertex_layers.parquet"), "--out", str(out),
                      "--calibrate-only", "--overwrite"])
    report = (out / "synth_report.md").read_text()
    check(rc == 0 and not stale.exists(), "the stale null.csv is deleted")
    check("synth_null.csv" not in report.split("## Files")[-1] and "synth_summary.csv" in report,
          "the report's Files section lists only this run's files")
    sm = pd.read_csv(out / "synth_summary.csv")
    check({"testable_share", "within-component U(region)-U(matched_A)"} <= set(sm.quantity)
          and (sm.window_rule == "ring_testable").any(),
          "calibrate-only summary carries the sensitivity rows (observed values)")
    check("Sensitivities for the calibration stop" in report and "SENSITIVITY" in log.getvalue(),
          "  ... and the report and the printout show them")

print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED'}")
sys.exit(1 if FAILS else 0)
