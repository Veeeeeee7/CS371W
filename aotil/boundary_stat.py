#!/usr/bin/env python3
"""
The O5 boundary statistic (docs/TASK-FULL-RUN-W7.md B3): is the Zone A edge around the regions
where the models fail drawn along lines with no terrain meaning (S: soil-survey polygon edges;
M: county lines and FIRM panel edges, the sheet edges of the flood map) more often than other
Zone A edge nearby?

Input: the vertex table of aotil/edge_layers.py, with the lobes and bboxes tables beside it. A
lobe is one single-part flood-map polygon (aotil/lobes.py); its boundary rings are sampled every
~30 m into vertices (x, y in EPSG:5070 metres) carrying d_<layer>_m, the distance to the nearest
line of each layer. Roles: region (Zone A lobes of a selected region), matched_A (unselected
Zone A lobes in the same 50 km test tiles, matched on perimeter), AE (Zone AE lobes in the same
tiles).

Steps (the report cites them by number)
1. Keep the outer edge: drop vertices on an edge shared with another flood polygon, without a
   DEM value, or outside the state or the lobe's own map panels.
2. Cut each ring into windows of 10 vertices (300 m); drop windows with < 8 usable vertices.
3. Level set (contour-like): the spread of the ground elevation z along the window
   (--level-test: range, iqr or detrended range) is <= tau (default 0.76 m, half a 5-ft contour
   interval), and the ground rises going out: median z_out - z_in > 0 (DEM 100 m outside minus
   100 m inside) over >= 5 vertices whose probes landed on the right side.
4. On a line: the window's median distance <= d (100 m, and 30 m). S = d_soil_m; M = the min of
   the county and panel distances; canals (d_canals_m) are reported on their own.
5. U = share of edge length that is not a level set and lies on S or M. Primary
   U(region) - U(matched_A); secondary U(region) - U(AE); 2 x 2 tables per role and layer.
6. Displaced-edge null: every region lobe is shifted rigidly by its own random vector (300-1,000
   m, uniform direction), its S and M distances recomputed, its windows' level flags kept.
7. Bootstrap over components: 95% percentile intervals for every pooled U and difference.

Outputs in --out (prefix = <cell>_<variant>_S<S>, from the --vertex-layers name):
<prefix>_windows.parquet, _calibration.csv, _summary.csv, _null.csv (not with --calibrate-only)
and _report.md.

    python aotil/boundary_stat.py \\
        --vertex-layers results/<D>/boundary/<prefix>_vertex_layers.parquet \\
        --gis-dir data/interim/gis --null-draws 200 --seed 0 --out results/<D>/boundary_stat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_layers as el  # noqa: E402

# Choices the brief (docs/TASK-FULL-RUN-W7.md B3) left open. The report and
# docs/RUN-W7-REPORT-2026-10-16.md cite them as "decision N", so the numbers are fixed.
#  1. Windows are cut on the ring's own vertex sequence before any vertex is dropped, so a window
#     is one stretch of <= 300 m. Cutting the kept vertices instead (the rule of the preview,
#     results/2026-10-16/w7_first_look/u_preview.py, computed alongside) lets a window jump
#     across a dropped stretch. Windows do not wrap round the ring's seam.
#  2. A window weighs its edge length, n_usable x ds. A lobe built for two components counts once
#     per component, as the bootstrap unit requires; U with each physical lobe once is a check.
#  3. A NaN distance means the layer has no line there (a one-panel DFIRM has no interior edge):
#     it is +inf, i.e. not on the line.
#  4. The null loads S and M with edge_layers' own loaders and measures with edge_layers.measure.
#     A zero displacement must give back the stored distances to SELF_TOL_M, else the run exits 1.
#  5. Only the usable vertices of the region's kept windows are displaced, one vector per lobe and
#     draw. Per-draw seeds are spawned from --seed, so the draws do not depend on --jobs.
#  6. Coverage of the displaced vertices is counted, not fixed: outside the bbox (must be 0), soil
#     distance beyond the bbox margin, in no SSURGO map unit, outside the state or the own panels.
#     U(region) without the windows they touch is compared PAIRED with the observed U over the
#     same windows, because those windows sit near state and county lines, i.e. on M.
#  7. matched_A and AE are not displaced, so both differences share U(region)'s null percentile.
#  8. Null percentile = mid-rank; p_upper = (1 + #null >= obs) / (1 + #draws). Values within
#     TIE_TOL are ties: observed U (pandas groupby) and null U (numpy) sum in different orders and
#     can differ by a few ulp; distinct U values differ by at least one window's weight (~1e-7).
#  9. Floodway AE lobes are kept; is_floodway is carried into the windows.
# 10. Sensitivities for the calibration stop: (a) testable windows only (>= MIN_DROP good probes;
#     any other window can never be a level set, and the testable share differs by role); (b) the
#     within-component difference sum_c w_c (U_c(region) - U_c(control)), w_c = component c's
#     share of the region edge, so where each role's edge lies across components no longer enters.
# 11. --windows testable (the owner's choice at the calibration stop): the statistic, bootstrap and
#     null use only the testable windows; the calibration and windows.parquet still cover all. Its
#     null is another realisation than an all-windows run's: a lobe without a testable window is
#     not displaced, so the seeded vectors are handed out in a different order.

WIN = 10                        # vertices per window (300 m)
MIN_USABLE = 8                  # a window needs this many usable vertices
MIN_DROP = 5                    # vertices with good probes for the drop test
TAU = 0.76                      # m, half a 5-ft contour interval
TAUS = (0.3, 0.76, 1.5)         # m, calibration grid (lidar-era, FEMA, wide)
TESTS = ("range", "iqr", "detrended")
STAT_COL = {"range": "z_range", "iqr": "z_iqr", "detrended": "z_detrended_range"}
D_BANDS = (100.0, 30.0)         # m, on-a-line distances (main, tight)
SHIFT_M = (300.0, 1000.0)       # m, displaced-edge vector length range
SELF_TOL_M = 1e-6               # zero-displacement self-check tolerance
TIE_TOL = 1e-12                 # null percentile: values this close are ties (decision 8)
ROLES = ("region", "matched_A", "AE")
M_LAYERS = ("county", "panel_interior", "panel_exterior")
NULL_LAYERS = ("soil",) + M_LAYERS
LINE_LAYERS = ("S", "M", "SM", "canals")
SOURCES = ("ssurgo", "county", "panel")    # the edge_layers sources S and M come from
READ_COLS = ["lobe_id", "zone", "role", "component", "ring", "s", "x", "y", "dfirm_id",
             "contact_row_id", "dem_ok", "outside_state", "outside_own_panels",
             "in_probe_inside", "out_probe_outside", "soil_ok", "z", "z_in", "z_out",
             "d_soil_m", "d_county_m", "d_panel_interior_m", "d_panel_exterior_m", "d_canals_m"]
OPTIONAL_COLS = ("d_canals_m", "soil_ok")
DISTANCE_COLS = ("d_soil_m", "d_county_m", "d_panel_interior_m", "d_panel_exterior_m", "d_canals_m")
OUTPUT_NAMES = ("windows.parquet", "calibration.csv", "summary.csv", "null.csv", "report.md")
U_QUANTITIES = ("U", "U(region)-U(matched_A)", "U(region)-U(AE)")
DIFF_KEYS = {"U(region)-U(matched_A)": "diff_matched_A", "U(region)-U(AE)": "diff_AE"}


def variant(test: str, tau: float) -> str:
    """Name of a level test at a threshold, e.g. 'range_t0.76'."""
    return f"{test}_t{tau:g}"


def indented(text: str) -> str:
    return "  " + text.replace("\n", "\n  ")


# ----------------------------------------------------------------- vertices
def ring_index(lobe_id: np.ndarray, ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For vertices sorted by (lobe_id, ring, s): each vertex's ring number (0..R-1, in table
    order) and its position k along its ring."""
    n = len(lobe_id)
    starts_ring = np.ones(n, dtype=bool)
    if n > 1:
        starts_ring[1:] = (lobe_id[1:] != lobe_id[:-1]) | (ring[1:] != ring[:-1])
    rid = np.cumsum(starts_ring) - 1
    k = np.arange(n) - np.flatnonzero(starts_ring)[rid]
    return rid, k


def ring_spacing(rid: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Each vertex's ring spacing ds (lobes.py samples a ring at s = k * ds); NaN on one-vertex
    rings, which never form a window."""
    n_rings = rid.max() + 1 if len(rid) else 0
    ds = np.full(n_rings, np.nan)
    second = np.flatnonzero(k == 1)
    ds[rid[second]] = s[second] - s[second - 1]
    return ds[rid]


def drop_flags(v: pd.DataFrame) -> dict[str, np.ndarray]:
    """One boolean per reason of step 1 (a vertex can have several)."""
    def flag(col):
        return v[col].fillna(False).astype(bool).to_numpy()

    contact = v.contact_row_id.notna().to_numpy()
    has_dem = flag("dem_ok") & np.isfinite(v.z.to_numpy(float))
    return {"contact": contact,
            "no_dem": ~has_dem,
            "outside_state": flag("outside_state"),
            "outside_own_panels": flag("outside_own_panels")}


def usable_mask(flags: dict[str, np.ndarray]) -> np.ndarray:
    return ~np.logical_or.reduce(list(flags.values()))


# ----------------------------------------------------------------- windows
def cut_windows(rid: np.ndarray, k: np.ndarray, usable: np.ndarray, mode: str = "ring",
                win: int = WIN, min_usable: int = MIN_USABLE
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The window of every vertex, for vertices sorted by (lobe_id, ring, s).

    mode 'ring': window w of a ring holds its vertices k = win*w .. win*w + win-1 (decision 1),
    and only the usable ones join it. mode 'filtered': u_preview.py's rule, which numbers a
    ring's usable vertices 0, 1, 2, ... and cuts that sequence into runs of win. Windows with
    fewer than min_usable usable vertices are dropped.
    Returns wid (0..W-1 in table order, -1 = in no kept window), slot (0..win-1, the place in
    the window) and, per kept window, its ring number and its index along the ring."""
    usable = np.asarray(usable, dtype=bool)
    if mode == "ring":
        pos = np.asarray(k)
    elif mode == "filtered":
        n_before = np.cumsum(usable) - usable            # usable vertices before this one
        if len(rid):
            ring_start = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        else:
            ring_start = np.empty(0, int)
        pos = n_before - n_before[ring_start][rid]
    else:
        raise ValueError(f"mode {mode!r}")
    local = pos // win                                   # window index along the ring
    slot = (pos % win).astype(np.int64)
    wid = np.full(len(rid), -1, dtype=np.int64)
    u = np.flatnonzero(usable)
    if not len(u):
        return wid, slot, np.empty(0, np.int64), np.empty(0, np.int64)
    n_local = int(local.max()) + 1
    key = rid[u].astype(np.int64) * n_local + local[u]   # one key per (ring, window)
    keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts >= min_usable
    new_id = np.where(keep, np.cumsum(keep) - 1, -1)
    wid[u] = new_id[inverse]
    kept = keys[keep]
    return wid, slot, (kept // n_local).astype(np.int64), (kept % n_local).astype(np.int64)


def window_matrix(values: np.ndarray, wid: np.ndarray, slot: np.ndarray, n_windows: int,
                  win: int = WIN) -> np.ndarray:
    """(n_windows, win) array of a per-vertex value; NaN where a window has no usable vertex in
    that slot."""
    out = np.full((n_windows, win), np.nan)
    m = wid >= 0
    out[wid[m], slot[m]] = np.asarray(values, dtype=float)[m]
    return out


def nanmedian_rows(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN rows -> NaN
        return np.nanmedian(a, axis=1)


def detrended_range(S: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Per row: range of the residuals of the least-squares line Z ~ S over the row's finite
    entries (NaN = empty slot); 0 on an exact linear ramp. It removes the slope of a river's
    water surface along the reach."""
    ok = np.isfinite(Z) & np.isfinite(S)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_mean = np.where(ok, S, 0.0).sum(axis=1) / n
        z_mean = np.where(ok, Z, 0.0).sum(axis=1) / n
        s_dev = np.where(ok, S - s_mean[:, None], 0.0)
        z_dev = np.where(ok, Z - z_mean[:, None], 0.0)
        sxx = (s_dev * s_dev).sum(axis=1)
        safe_sxx = np.where(sxx > 0, sxx, 1.0)
        slope = np.where(sxx > 0, (s_dev * z_dev).sum(axis=1) / safe_sxx, 0.0)
        resid = np.where(ok, z_dev - slope[:, None] * s_dev, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(resid, axis=1) - np.nanmin(resid, axis=1)


def inf_nan(d) -> np.ndarray:
    """Decision 3: a NaN distance (no line of the layer there) is +inf."""
    d = np.asarray(d, dtype=float)
    return np.where(np.isnan(d), np.inf, d)


def s_and_m(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex distance to S and to M from the four layer distances."""
    to_s = inf_nan(d["soil"])
    to_m = np.minimum.reduce([inf_nan(d[name]) for name in M_LAYERS])
    return to_s, to_m


def window_stats(vx: dict, wid: np.ndarray, slot: np.ndarray, n_windows: int,
                 win: int = WIN) -> pd.DataFrame:
    """Per-window numbers of steps 3-4. vx holds per-vertex arrays s, z, z_in, z_out, probe_ok,
    ds, d_S, d_M and d_canals (distances already NaN -> inf)."""
    def per_window(values):
        return window_matrix(values, wid, slot, n_windows, win)

    Z = per_window(vx["z"])
    S = per_window(vx["s"])
    n_usable = np.isfinite(Z).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q25, q75 = np.nanpercentile(Z, [25, 75], axis=1)
        z_range = np.nanmax(Z, axis=1) - np.nanmin(Z, axis=1)
    drop = np.where(vx["probe_ok"], vx["z_out"] - vx["z_in"], np.nan)
    D = per_window(drop)
    n_drop = np.isfinite(D).sum(axis=1)
    drop_median = nanmedian_rows(D)
    ds = np.nanmax(per_window(vx["ds"]), axis=1)
    return pd.DataFrame({
        "n_usable": n_usable, "ds_m": ds, "edge_m": n_usable * ds,
        "s_first": np.nanmin(S, axis=1), "s_last": np.nanmax(S, axis=1),
        "z_median": nanmedian_rows(Z), "z_range": z_range, "z_iqr": q75 - q25,
        "z_detrended_range": detrended_range(S, Z),
        "drop_median": drop_median, "n_drop": n_drop,
        "drop_ok": (n_drop >= MIN_DROP) & (drop_median > 0),
        "d_S_median_m": nanmedian_rows(per_window(vx["d_S"])),
        "d_M_median_m": nanmedian_rows(per_window(vx["d_M"])),
        "d_canals_median_m": nanmedian_rows(per_window(vx["d_canals"])),
    })


def add_flags(W: pd.DataFrame, level_test: str, tau: float) -> pd.DataFrame:
    """Level-set flags for every test x tau (with the drop condition), the selected one as
    `level`, the on-line flags and U at each d."""
    for test in TESTS:
        taus = set(TAUS)
        if test == level_test:
            taus.add(tau)
        for t in sorted(taus):
            W[f"level_{variant(test, t)}"] = (W[STAT_COL[test]] <= t) & W.drop_ok
    W["level"] = W[f"level_{variant(level_test, tau)}"]
    for d in D_BANDS:
        sfx = f"_d{d:g}"
        W["on_S" + sfx] = W.d_S_median_m <= d
        W["on_M" + sfx] = W.d_M_median_m <= d
        W["on_SM" + sfx] = W["on_S" + sfx] | W["on_M" + sfx]
        W["on_canals" + sfx] = W.d_canals_median_m <= d
        W["U" + sfx] = ~W.level & W["on_SM" + sfx]
    return W


def build_windows(v: pd.DataFrame, usable: np.ndarray, mode: str, level_test: str,
                  tau: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The window table of one window rule, with each vertex's window and slot."""
    wid, slot, _, index_in_ring = cut_windows(v["rid"].to_numpy(), v["k"].to_numpy(), usable, mode)
    n_windows = len(index_in_ring)
    d_S, d_M = s_and_m({name: v[f"d_{name}_m"].to_numpy() for name in NULL_LAYERS})
    probe_ok = (v.in_probe_inside.fillna(False).astype(bool)
                & v.out_probe_outside.fillna(False).astype(bool)).to_numpy()
    vx = {"s": v.s.to_numpy(float), "z": v.z.to_numpy(float), "z_in": v.z_in.to_numpy(float),
          "z_out": v.z_out.to_numpy(float), "ds": v.ds.to_numpy(float), "d_S": d_S, "d_M": d_M,
          "d_canals": inf_nan(v.d_canals_m.to_numpy()), "probe_ok": probe_ok}
    stats = window_stats(vx, wid, slot, n_windows)

    # a window's labels are those of its first vertex (windows are numbered in table order)
    in_window = wid >= 0
    _, first = np.unique(wid[in_window], return_index=True)
    lab = v.iloc[np.flatnonzero(in_window)[first]]
    if mode == "ring":
        k_first = index_in_ring * WIN
    else:
        k_first = lab.k.to_numpy(np.int64)
    meta = pd.DataFrame({
        "lobe_id": lab.lobe_id.astype(str).to_numpy(), "zone": lab.zone.astype(str).to_numpy(),
        "role": lab.role.astype(str).to_numpy(), "component": lab.component.to_numpy(np.int64),
        "dfirm_id": lab.dfirm_id.astype(str).to_numpy(), "ring": lab.ring.to_numpy(np.int64),
        "window": index_in_ring, "k_first": k_first,
        "is_floodway": lab.is_floodway.to_numpy(bool), "dup_lobe": lab.dup_lobe.to_numpy(bool)})
    W = pd.concat([meta, stats], axis=1)
    return add_flags(W, level_test, tau), wid, slot


# ----------------------------------------------------------------- shares
def comp_sums(W: pd.DataFrame, flag, weight: str = "length") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted numerator (flag) and denominator per (component, role).
    weight 'length' = edge length (decision 2), 'count' = 1 per window."""
    if weight not in ("length", "count"):
        raise ValueError(f"weight {weight!r}")
    if weight == "length":
        w = W.edge_m.astype(float)
    else:
        w = pd.Series(1.0, index=W.index)
    f = np.asarray(flag, dtype=bool)

    def by_component_and_role(values):
        sums = values.groupby([W.component, W.role]).sum().unstack()
        return sums.reindex(columns=list(ROLES)).fillna(0.0)

    return by_component_and_role(w * f), by_component_and_role(w)


def pooled_share(W: pd.DataFrame, flag, weight: str = "length") -> pd.Series:
    """Share of each role's edge (or windows) with `flag`, summed over components."""
    num, den = comp_sums(W, flag, weight)
    total = den.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum() / total.where(total > 0)


def comp_share(W: pd.DataFrame, flag, weight: str = "length") -> pd.DataFrame:
    num, den = comp_sums(W, flag, weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den.where(den > 0)


def bootstrap(num: pd.DataFrame, den: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled U per role and both differences in each bootstrap draw. idx: (B, n_components)
    rows of num/den drawn with replacement. A draw without edge of a role gives NaN for what
    needs that role."""
    N = num.to_numpy()[idx].sum(axis=1)
    D = den.to_numpy()[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(D > 0, N / np.where(D > 0, D, 1.0), np.nan)
    out = {r: U[:, i] for i, r in enumerate(ROLES)}
    out["diff_matched_A"] = out["region"] - out["matched_A"]
    out["diff_AE"] = out["region"] - out["AE"]
    return out


def ci(a: np.ndarray) -> tuple[float, float, int]:
    """95% percentile interval of the finite values, and their number."""
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, 0
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi), len(a)


def null_percentile(obs: float, null: np.ndarray, tol: float = TIE_TOL) -> tuple[float, float]:
    """Decision 8: mid-rank percentile and one-sided upper p. A draw within tol of obs is a tie:
    half a rank in the percentile, and 'at least as large' in p_upper."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan, np.nan
    tie = np.abs(null - obs) <= tol
    pct = 100.0 * (((null < obs) & ~tie).sum() + 0.5 * tie.sum()) / len(null)
    p_upper = (1 + ((null >= obs) | tie).sum()) / (1 + len(null))
    return float(pct), float(p_upper)


def within_component(num: pd.DataFrame, den: pd.DataFrame, idx: np.ndarray | None = None
                     ) -> dict[str, np.ndarray]:
    """Decision 10b: sum_c w_c (U_c(region) - U_c(ctrl)) / sum_c w_c, w_c = component c's region
    edge length, for ctrl = matched_A and AE. num/den: comp_sums rows (components) x ROLES.
    Components where either role has no edge are left out. idx: (B, n_components) bootstrap
    rows as in bootstrap() (a component drawn twice counts twice); None = the observed value
    (an array of one)."""
    N, D = num.to_numpy(float), den.to_numpy(float)
    if idx is None:
        idx = np.arange(len(N))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(D > 0, N / np.where(D > 0, D, 1.0), np.nan)
    out = {}
    for j, key in ((1, "diff_matched_A"), (2, "diff_AE")):
        diff = U[:, 0] - U[:, j]                        # NaN where a role has no edge
        ok = np.isfinite(diff)
        weight = np.where(ok, D[:, 0], 0.0)[idx]        # region edge of each drawn component
        total = weight.sum(axis=1)
        weighted = (weight * np.where(ok, diff, 0.0)[idx]).sum(axis=1)
        out[key] = np.where(total > 0, weighted / np.where(total > 0, total, 1.0), np.nan)
    return out


# ----------------------------------------------------------------- null
def shift_vectors(n_lobes: int, rng: np.random.Generator, lo: float = SHIFT_M[0],
                  hi: float = SHIFT_M[1]) -> tuple[np.ndarray, np.ndarray]:
    """One translation per lobe: length uniform on [lo, hi], direction uniform."""
    r = rng.uniform(lo, hi, n_lobes)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_lobes)
    return r * np.cos(theta), r * np.sin(theta)


def displaced_distances(x: np.ndarray, y: np.ndarray, component: np.ndarray, dfirm: np.ndarray,
                        sets: dict) -> dict[str, np.ndarray]:
    """d_<layer>_m for the points (x, y), each against its own component's soil lines and its
    own DFIRM's panel edges, by edge_layers.measure (decision 4)."""
    pts = pd.DataFrame({"x": x, "y": y, "component": component,
                        "dfirm_id": pd.array(dfirm, dtype="string")})
    out = {}
    for name, ls in sets.items():
        out[name] = np.asarray(el.measure(pts, ls)[f"d_{name}_m"], dtype=float)
    return out


class NullContext:
    """What every null draw needs: the displaced vertices (usable vertices of the region's kept
    windows, decision 5) with their window and slot, the region windows' level flags (kept
    fixed) and edge lengths, the S and M LineSets, the bboxes and the coverage geometry.
    matched_A and AE do not move."""

    def __init__(self, v, wid, slot, W, sets, bb, cover, level_cols):
        region_windows = np.flatnonzero((W.role == "region").to_numpy())
        region_row = np.full(len(W), -1, dtype=np.int64)   # window -> row among region windows
        region_row[region_windows] = np.arange(len(region_windows))
        iv = np.flatnonzero((wid >= 0) & (v.role.astype(str).to_numpy() == "region"))
        self.iv = iv                                       # displaced vertices (rows of v)
        self.rwid = region_row[wid[iv]]
        self.slot = slot[iv]
        self.nW = len(region_windows)
        self.x = v.x.to_numpy(float)[iv]
        self.y = v.y.to_numpy(float)[iv]
        self.comp = v.component.to_numpy(np.int64)[iv]
        self.dfirm = v.dfirm_id.astype(str).to_numpy()[iv]
        self.lobes, self.lobe_ix = np.unique(v.lobe_id.astype(str).to_numpy()[iv],
                                             return_inverse=True)
        Wr = W.iloc[region_windows]
        self.w = Wr.edge_m.to_numpy(float)
        self.wcomp = Wr.component.to_numpy(np.int64)
        self.comps = np.unique(self.wcomp)
        self.wci = np.searchsorted(self.comps, self.wcomp)
        self.level_cols = level_cols
        self.L = Wr[level_cols].to_numpy(bool)               # (region windows, level variants)
        # observed U flags of the selected test: the paired side of the covered null
        self.Uobs = {d: Wr[f"U_d{d:g}"].to_numpy(bool) for d in D_BANDS}
        self.testable = Wr.n_drop.to_numpy() >= MIN_DROP     # decision 10a
        self.sets = sets
        self.bb = bb
        self.cover = cover
        box = bb[["minx", "miny", "maxx", "maxy"]].copy()
        box.index = box.index.astype(np.int64)
        self.margin_box = box.reindex(self.comp).to_numpy(float)    # NaN = no bbox: outside
        self.stored = {name: v[f"d_{name}_m"].to_numpy(float)[iv] for name in NULL_LAYERS}

    def window_medians(self, d: dict) -> tuple[np.ndarray, np.ndarray]:
        """Each region window's median distance to S and to M."""
        S, M = s_and_m(d)
        return (nanmedian_rows(window_matrix(S, self.rwid, self.slot, self.nW)),
                nanmedian_rows(window_matrix(M, self.rwid, self.slot, self.nW)))

    def coverage(self, x, y, d_soil) -> dict[str, np.ndarray]:
        """Per displaced vertex: the coverage problems of decision 6. Each test is 'within TOL_M
        of a polygon', as edge_layers tests soil_ok, the state and the panel footprint. The
        polygons are the QUERY side of an STRtree of the points: shapely then prepares a private
        copy of each polygon per call (fast point-in-polygon) and never touches the shared
        objects, so draws can run in threads without a lock."""
        b = self.margin_box
        margin = np.minimum.reduce([x - b[:, 0], b[:, 2] - x, y - b[:, 1], b[:, 3] - y])
        out = {"outside_bbox": ~(margin >= 0),
               "beyond_margin": ~(inf_nan(d_soil) <= margin)}
        pts = shapely.points(x, y)

        def near(jv: np.ndarray, polys) -> np.ndarray:
            """Which of the points jv lie within TOL_M of any of `polys`."""
            hit = np.zeros(len(jv), dtype=bool)
            polys = np.atleast_1d(np.asarray(polys, dtype=object))
            if len(jv) and len(polys):
                tree = STRtree(pts[jv])
                hit[tree.query(polys, predicate="dwithin", distance=el.TOL_M)[1]] = True
            return hit

        no_map_unit = np.ones(len(x), dtype=bool)
        for c, polys in self.cover["mupolygons"].items():
            jv = np.flatnonzero(self.comp == c)
            no_map_unit[jv] = ~near(jv, polys)
        out["no_map_unit"] = no_map_unit
        state = self.cover.get("state")
        if state is not None:
            out["outside_state"] = ~near(np.arange(len(x)), state)
        else:
            out["outside_state"] = np.zeros(len(x), dtype=bool)
        outside_panels = np.zeros(len(x), dtype=bool)
        for dfirm, footprint in self.cover["panels"].items():
            jv = np.flatnonzero(self.dfirm == dfirm)
            if footprint is not None:
                outside_panels[jv] = ~near(jv, footprint)
        out["outside_own_panels"] = outside_panels
        return out

    def u_region(self, dS, dM, keep=None) -> dict:
        """U(region) per level variant and the on-line shares at each d, pooled; per component
        for the selected test. With `keep` (bool per window) only those windows (NaN if none)."""
        w = self.w if keep is None else self.w * keep
        out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for d in D_BANDS:
                on = (dS <= d) | (dM <= d)
                u = (w[:, None] * (~self.L & on[:, None])).sum(axis=0) / w.sum()
                out[d] = dict(zip(self.level_cols, u))
                out[d]["onS"] = (w * (dS <= d)).sum() / w.sum()
                out[d]["onM"] = (w * (dM <= d)).sum() / w.sum()
                out[d]["onSM"] = (w * on).sum() / w.sum()
                n_comps = len(self.comps)
                num = np.bincount(self.wci, weights=w * (~self.L[:, 0] & on), minlength=n_comps)
                den = np.bincount(self.wci, weights=w, minlength=n_comps)
                out[d]["by_comp"] = num / den
        return out

    def u_observed(self, keep) -> dict[float, float]:
        """Observed U(region) (selected test) over the windows `keep`: what the covered null of
        the same draw is compared with (decision 6)."""
        wk = self.w * keep
        with np.errstate(invalid="ignore", divide="ignore"):
            return {d: (wk * self.Uobs[d]).sum() / wk.sum() for d in D_BANDS}

    def draw(self, dx_lobe: np.ndarray, dy_lobe: np.ndarray, with_coverage: bool = True):
        """One null draw: lobe i moves by (dx_lobe[i], dy_lobe[i])."""
        x = self.x + dx_lobe[self.lobe_ix]
        y = self.y + dy_lobe[self.lobe_ix]
        d = displaced_distances(x, y, self.comp, self.dfirm, self.sets)
        dS, dM = self.window_medians(d)
        res = {"dist": d, "dS": dS, "dM": dM, "u": self.u_region(dS, dM),
               "u_testable": self.u_region(dS, dM, keep=self.testable)}
        if with_coverage:
            cov = self.coverage(x, y, d["soil"])
            bad = np.logical_or.reduce(list(cov.values()))
            n_bad = np.bincount(self.rwid, weights=bad.astype(float), minlength=self.nW)
            covered = n_bad == 0
            res["cov"] = {k: int(a.sum()) for k, a in cov.items()}
            res["cov"]["windows_touched"] = int((n_bad > 0).sum())
            res["u_covered"] = self.u_region(dS, dM, keep=covered)
            res["u_covered_obs"] = self.u_observed(covered)
        return res


def coverage_geometry(a, comps, dfirms) -> dict:
    """What edge_layers checks coverage against, from the same files: the SSURGO mupolygons of
    each component (beside the mulines file), the union of the state's counties
    (counties_<ST>.parquet beside county_lines_<ST>; absent -> not checked) and each DFIRM's
    panel footprint (its edges polygonized). Deliberately not prepared (NullContext.coverage)."""
    mulines = a.files.get("ssurgo") or a.gis_dir / "ssurgo" / f"{a.prefix}_mulines.parquet"
    mupolygons_path = mulines.with_name(mulines.name.replace("_mulines", "_mupolygons"))
    polys, _ = el.read_geo(mupolygons_path, columns=["component"])
    poly_comp = polys.component.to_numpy().astype(int)
    poly_geom = np.asarray(polys.geometry.values)
    mupolygons = {int(c): poly_geom[poly_comp == c] for c in comps}

    county_lines = a.files.get("county") or a.gis_dir / f"county_lines_{a.state}.parquet"
    counties_path = county_lines.with_name(county_lines.name.replace("county_lines_", "counties_"))
    state = None
    if counties_path.exists():
        counties, _ = el.read_geo(counties_path, columns=[])
        state = shapely.union_all(np.asarray(counties.geometry.values))

    panel_path = a.files.get("panel") or a.gis_dir / f"panel_edges_{a.state}.parquet"
    edges, _ = el.read_geo(panel_path)
    edge_dfirm = edges.dfirm_id.astype(str).to_numpy()
    edge_geom = np.asarray(edges.geometry.values)
    panels = {}
    for dfirm in dfirms:
        faces = shapely.get_parts(shapely.polygonize(edge_geom[edge_dfirm == dfirm]))
        panels[dfirm] = shapely.union_all(faces) if len(faces) else None

    files = {"mupolygons": str(mupolygons_path),
             "counties": str(counties_path) if state is not None else None,
             "panels": str(panel_path)}
    return {"mupolygons": mupolygons, "state": state, "panels": panels, "files": files}


def load_lines(a, v: pd.DataFrame, bb: pd.DataFrame) -> tuple[dict, el.Coverage, dict]:
    """The S and M LineSets by edge_layers' own loaders, fed the region vertices; the coverage
    flags the loaders recompute are compared with the stored columns."""
    reg = v[v.role.astype(str) == "region"]
    vt = pd.DataFrame({c: reg[c].to_numpy() for c in ("lobe_id", "zone", "role", "x", "y")})
    vt["component"] = reg.component.to_numpy(np.int32)
    vt["dfirm_id"] = pd.array(reg.dfirm_id.astype(str).to_numpy(), dtype="string")
    ns = argparse.Namespace(files=a.files, gis_dir=a.gis_dir, prefix=a.prefix, state=a.state,
                            bboxes_p=a.bboxes_p, layer_files={}, flags={},
                            state_tol_m=a.state_tol_m)
    cov = el.Coverage()
    ssurgo = el.load_ssurgo(ns, vt, bb, cov)
    county = el.load_county(ns, vt, bb, cov)
    panel = el.load_panel(ns, vt, bb, cov)
    sets = {ls.name: ls for ls in ssurgo + county + panel if ls.name in NULL_LAYERS}
    not_flagged = np.zeros(len(vt), dtype=bool)   # a loader adds a flag only when a vertex has it
    flags = {"outside_state": ns.flags.get("outside_state", not_flagged),
             "outside_own_panels": ns.flags.get("outside_own_panels", not_flagged)}
    if "soil_ok" in vt:                                  # set by load_ssurgo
        flags["soil_ok"] = vt.soil_ok.to_numpy(bool)
    mismatch = {}
    for name, recomputed in flags.items():
        if name in reg:
            stored = reg[name].fillna(False).astype(bool).to_numpy()
            mismatch[name] = int((recomputed != stored).sum())
    return sets, cov, {"layer_files": ns.layer_files, "flag_mismatch": mismatch}


# ----------------------------------------------------------------- inputs
def prefix_of(path: Path) -> str:
    suffix = "_vertex_layers.parquet"
    if not path.name.endswith(suffix):
        raise SystemExit(f"{path.name}: expected <prefix>{suffix} (or pass --prefix)")
    return path.name[: -len(suffix)]


def index_vertices(v: pd.DataFrame) -> pd.DataFrame:
    """Sorted by (lobe_id, ring, s), with rid (ring number), k (position in the ring) and ds
    (ring spacing)."""
    v = v.sort_values(["lobe_id", "ring", "s"], kind="stable").reset_index(drop=True)
    rid, k = ring_index(v.lobe_id.astype(str).to_numpy(), v.ring.to_numpy())
    v["rid"] = rid
    v["k"] = k
    v["ds"] = ring_spacing(rid, k, v.s.to_numpy(float))
    for c in ("dup_lobe", "is_floodway"):
        if c not in v:
            v[c] = False
    return v


def check_lobes(lobes_p: Path, v: pd.DataFrame) -> pd.DataFrame:
    """The lobes table indexed by lobe_id, checked against the vertices as edge_layers checks
    it: every vertex's lobe is in it and agrees on component, role, zone and DFIRM."""
    want = ["lobe_id", "component", "role", "zone", "dfirm_id", "src_row_id", "part", "is_floodway"]
    names = pq.read_schema(lobes_p).names
    lb = pq.read_table(lobes_p, columns=[c for c in want if c in names]).to_pandas()
    if not lb.lobe_id.is_unique:
        raise SystemExit(f"{lobes_p}: lobe_id is not unique")
    lb["lobe_id"] = lb.lobe_id.astype(str)
    lb = lb.set_index("lobe_id")
    first = v.drop_duplicates("lobe_id")
    key = first.lobe_id.astype(str)
    known = key.isin(lb.index)
    if not known.all():
        raise SystemExit(f"{(~known).sum()} lobe_ids of the vertices are not in {lobes_p}")
    ref = lb.reindex(key)
    bad = []
    for c in ("component", "role", "zone", "dfirm_id"):
        if (ref[c].astype(str).to_numpy() != first[c].astype(str).to_numpy()).any():
            bad.append(c)
    if bad:
        raise SystemExit(f"vertices disagree with {lobes_p.name} on {bad}: not the lobes these "
                         f"vertices were built from")
    return lb


def lobe_flags(lb: pd.DataFrame) -> tuple[pd.Series, pd.Series, str | None]:
    """Per lobe: dup_lobe (decision 2: the same physical lobe (src_row_id, part) is already in
    an earlier component of its role), is_floodway, and a note when dup_lobe cannot be known."""
    note = None
    if {"src_row_id", "part"} <= set(lb.columns):
        physical = lb.src_row_id.astype(str) + "_" + lb.part.astype(str)
        ranked = lb.assign(phys=physical).sort_values(["role", "component"], kind="stable")
        dup = ranked.duplicated(["role", "phys"])
    else:
        dup = pd.Series(False, index=lb.index)
        note = "--lobes has no src_row_id/part: the deduplicated check is skipped"
    if "is_floodway" in lb:
        floodway = lb.is_floodway.fillna(False).astype(bool)
    else:
        floodway = pd.Series(False, index=lb.index)
    return dup, floodway, note


def read_vertices(a) -> tuple[pd.DataFrame, dict, list[str]]:
    """The vertex table indexed by index_vertices, with the lobe columns dup_lobe and
    is_floodway; edge_layers' metadata; notes for the report."""
    schema = pq.read_schema(a.vertex_layers)
    names = set(schema.names)
    missing = [c for c in READ_COLS if c not in names and c not in OPTIONAL_COLS]
    if missing:
        raise SystemExit(f"{a.vertex_layers}: missing columns {missing} (an edge_layers.py output "
                         f"from a pull_dem / lobes.py run that writes probe and contact columns?)")
    notes = []
    v = pd.read_parquet(a.vertex_layers, columns=[c for c in READ_COLS if c in names])
    for c in OPTIONAL_COLS:
        if c in v:
            continue
        if c == "d_canals_m":
            v[c] = np.nan
            notes.append(f"{c} absent from the table; canals reported as not measured")
        else:                                            # soil_ok: every vertex in a map unit
            v[c] = True
            notes.append(f"{c} absent from the table")
    meta = json.loads((schema.metadata or {}).get(b"edge_layers", b"{}"))
    v = index_vertices(v)
    dup, floodway, note = lobe_flags(check_lobes(a.lobes_p, v))
    if note:
        notes.append(note)
    key = v.lobe_id.astype(str)
    v["dup_lobe"] = dup.reindex(key).to_numpy(bool)
    v["is_floodway"] = floodway.reindex(key).to_numpy(bool)
    return v, meta, notes


def nan_distance_counts(v: pd.DataFrame, usable: np.ndarray) -> pd.DataFrame:
    """Per role and distance column: usable vertices with a NaN distance (decision 3)."""
    role = v.role.astype(str).to_numpy()
    counts = {}
    for col in DISTANCE_COLS:
        d = v[col].to_numpy(float)
        counts[col] = [int(np.isnan(d[usable & (role == r)]).sum()) for r in ROLES]
    return pd.DataFrame(counts, index=list(ROLES))


# ----------------------------------------------------------------- tables
def format_cell(x, fmt: str | None) -> str:
    """A markdown table cell: floats by fmt (default 3 decimals; blank if not finite), ints with
    thousands separators, anything else as str."""
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        if fmt:
            return fmt.format(x)
        return f"{x:.3f}"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{x:,}"
    return str(x)


def md_table(df: pd.DataFrame, fmt: dict | None = None, index: bool = True) -> str:
    """A markdown table (no tabulate dependency). fmt: column -> float format, '*' = all."""
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    values = [d[c].tolist() for c in d.columns]          # tolist keeps ints ints
    for row in zip(*values):
        cells = []
        for c, x in zip(d.columns, row):
            cells.append(format_cell(x, fmt.get(c, fmt.get("*"))))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def drop_table(v: pd.DataFrame, flags: dict, usable: np.ndarray, wid: np.ndarray,
               wid_f: np.ndarray) -> pd.DataFrame:
    """Per role: vertices, each drop reason, the usable ones and those in windows under both
    window rules."""
    role = v.role.astype(str).to_numpy()
    ds = np.nan_to_num(v.ds.to_numpy(float))
    rows = []
    for r in ROLES:
        m = role == r
        row = {"role": r, "vertices": int(m.sum())}
        for reason, f in flags.items():
            row[reason] = int((m & f).sum())
        row["dropped_any"] = int((m & ~usable).sum())
        row["usable"] = int((m & usable).sum())
        row["usable_km"] = ds[m & usable].sum() / 1e3
        row["in_windows"] = int((m & (wid >= 0)).sum())
        row["in_windows_km"] = ds[m & (wid >= 0)].sum() / 1e3
        row["in_windows_preview_rule"] = int((m & (wid_f >= 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("role")


def calibration_rows(W: pd.DataFrame, name: str, stat: str, tau: float, drop_required: bool,
                     flag, among: np.ndarray | None = None) -> list[dict]:
    """The calibration rows of one test: its share per role, of edge and of windows."""
    X = W
    f = np.asarray(flag, dtype=bool)
    if among is not None:
        X = W[among]
        f = f[among]
    share_edge = pooled_share(X, f)
    share_windows = pooled_share(X, f, "count")
    rows = []
    for r in ROLES:
        m = (X.role == r).to_numpy()
        rows.append(dict(test=name, stat=stat, tau=tau, drop_required=drop_required, role=r,
                         share_edge=share_edge.get(r, np.nan),
                         share_windows=share_windows.get(r, np.nan),
                         n_windows=int(m.sum()), edge_km=X.edge_m[m].sum() / 1e3))
    return rows


def calibration_table(W: pd.DataFrame, level_test: str = "range", tau: float = TAU) -> pd.DataFrame:
    """Level-set share per role (edge-weighted and by window count) for every test x tau, with
    and without the drop condition, and the drop alone; the selected test is added when it is
    not on the grid. The drop condition is split into its parts: >= 5 good probes (the
    'testable' windows, decision 10a) and median drop > 0, the latter over all windows (a window
    with no good probe has no median and fails) and over the windows with >= 1 good probe."""
    rows = []
    for test in TESTS:
        for t in TAUS:
            rows += calibration_rows(W, f"{test} <= {t:g} m + drop", test, t, True,
                                     (W[STAT_COL[test]] <= t) & W.drop_ok)
    if tau not in TAUS:
        rows += calibration_rows(W, f"{level_test} <= {tau:g} m + drop (selected)", level_test,
                                 tau, True, (W[STAT_COL[level_test]] <= tau) & W.drop_ok)
    for test in TESTS:
        for t in TAUS:
            rows += calibration_rows(W, f"{test} <= {t:g} m alone", test, t, False,
                                     W[STAT_COL[test]] <= t)
    rows += calibration_rows(W, "drop alone (>= 5 good probes, median > 0)", "drop", np.nan, True,
                             W.drop_ok)
    rows += calibration_rows(W, f"  >= {MIN_DROP} vertices with good probes (testable windows)",
                             "drop_n", np.nan, True, W.n_drop >= MIN_DROP)
    rows += calibration_rows(W, "  median drop > 0 (windows with no good probe count as failing)",
                             "drop_sign", np.nan, True, W.drop_median > 0)
    rows += calibration_rows(W, "  median drop > 0 among windows with >= 1 good probe",
                             "drop_sign_probed", np.nan, True, W.drop_median > 0,
                             among=(W.n_drop >= 1).to_numpy())
    return pd.DataFrame(rows)


def two_by_two(W: pd.DataFrame, d: float) -> pd.DataFrame:
    """Share of edge length in each cell of (level set) x (on layer), per role and layer, for
    the selected level test."""
    rows = []
    level = W.level.to_numpy(bool)
    for layer in LINE_LAYERS:
        on = W[f"on_{layer}_d{d:g}"].to_numpy(bool)
        cells = {"level & on": level & on, "level & off": level & ~on,
                 "not level & on": ~level & on, "not level & off": ~level & ~on}
        for name, f in cells.items():
            share = pooled_share(W, f)
            for r in ROLES:
                rows.append(dict(layer=layer, cell=name, role=r, d_m=d, share=share.get(r, np.nan)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- summary
class Summary:
    """The rows of <prefix>_summary.csv. Every row has every column; `put` fills the defaults:
    pooled over components, ring window rule, edge-length weights, the selected level test."""

    def __init__(self, level_test: str, windows: str):
        self.level_test = level_test
        self.windows = windows
        self.rows: list[dict] = []

    def put(self, **values) -> None:
        row = dict(quantity="", layer="", role="", component="pooled", d_m=np.nan,
                   level_test=self.level_test, window_rule="ring", windows=self.windows,
                   weight="length", value=np.nan, n_windows=np.nan, edge_km=np.nan,
                   boot_lo=np.nan, boot_hi=np.nan, boot_n=np.nan, null_mean=np.nan,
                   null_q025=np.nan, null_q975=np.nan, null_pct=np.nan, null_p_upper=np.nan)
        row.update(values)
        self.rows.append(row)


def level_columns(sel: str) -> tuple[list[str], dict[str, str]]:
    """The window columns of the selected level test ('level') and of every other test on the
    grid, and each one's variant name."""
    cols = ["level"]
    for test in TESTS:
        for t in TAUS:
            if variant(test, t) != sel:
                cols.append(f"level_{variant(test, t)}")
    names = {"level": sel}
    for c in cols[1:]:
        names[c] = c[len("level_"):]
    return cols, names


def put_u_rows(summary: Summary, X: pd.DataFrame, share: pd.Series, d: float, level_test: str,
               window_rule: str, weight: str = "length") -> None:
    """Rows U per role (with its windows and edge) and both differences."""
    for r in ROLES:
        m = (X.role == r).to_numpy()
        summary.put(quantity="U", role=r, d_m=d, level_test=level_test, window_rule=window_rule,
                    weight=weight, value=share[r], n_windows=int(m.sum()),
                    edge_km=X.edge_m[m].sum() / 1e3)
    summary.put(quantity="U(region)-U(matched_A)", d_m=d, level_test=level_test,
                window_rule=window_rule, weight=weight, value=share["region"] - share["matched_A"])
    summary.put(quantity="U(region)-U(AE)", d_m=d, level_test=level_test,
                window_rule=window_rule, weight=weight, value=share["region"] - share["AE"])


def put_observed(summary: Summary, W: pd.DataFrame, Wf: pd.DataFrame, level_cols: list[str],
                 var_name: dict) -> dict:
    """Pooled level-set shares, U and on-line shares under both window rules and weights (every
    level test only under the primary ring rule and edge-length weights).
    Returns (window rule, weight, d, level variant) -> U per role."""
    obs = {}
    for rule, X in (("ring", W), ("filtered", Wf)):
        for weight in ("length", "count"):
            level = pooled_share(X, X.level, weight)
            for r in ROLES:
                summary.put(quantity="level_share", role=r, window_rule=rule, weight=weight,
                            value=level[r])
            for d in D_BANDS:
                for lc in level_cols:
                    if (rule, weight) != ("ring", "length") and lc != "level":
                        continue
                    share = pooled_share(X, ~X[lc] & X[f"on_SM_d{d:g}"], weight)
                    obs[(rule, weight, d, var_name[lc])] = share
                    put_u_rows(summary, X, share, d, var_name[lc], rule, weight)
                for layer in LINE_LAYERS:
                    share = pooled_share(X, X[f"on_{layer}_d{d:g}"], weight)
                    for r in ROLES:
                        summary.put(quantity=f"on_{layer}", role=r, d_m=d, window_rule=rule,
                                    weight=weight, value=share[r])
    return obs


def put_per_component(summary: Summary, W: pd.DataFrame) -> dict:
    """U per role and both differences per component, then the level-set share per component.
    Returns d -> the per-component U table."""
    per_comp = {}
    for d in D_BANDS:
        flag = W[f"U_d{d:g}"]
        share = comp_share(W, flag)
        per_comp[d] = share
        _, den = comp_sums(W, flag)
        for c in share.index:
            for r in ROLES:
                summary.put(quantity="U", role=r, component=int(c), d_m=d, value=share.loc[c, r],
                            edge_km=den.loc[c, r] / 1e3,
                            n_windows=int(((W.component == c) & (W.role == r)).sum()))
            summary.put(quantity="U(region)-U(matched_A)", component=int(c), d_m=d,
                        value=share.loc[c, "region"] - share.loc[c, "matched_A"])
            summary.put(quantity="U(region)-U(AE)", component=int(c), d_m=d,
                        value=share.loc[c, "region"] - share.loc[c, "AE"])
    level = comp_share(W, W.level)
    for c in level.index:
        for r in ROLES:
            summary.put(quantity="level_share", role=r, component=int(c), value=level.loc[c, r])
    return per_comp


def put_dedup_and_two_by_two(summary: Summary, W: pd.DataFrame) -> pd.DataFrame:
    """U with each physical lobe counted once (window rule 'ring_dedup'), and the 2 x 2 cells.
    Returns the 2 x 2 table."""
    Wd = W[~W.dup_lobe]
    for d in D_BANDS:
        share = pooled_share(Wd, Wd[f"U_d{d:g}"])
        for r in ROLES:
            summary.put(quantity="U", role=r, d_m=d, window_rule="ring_dedup", value=share[r],
                        edge_km=Wd.edge_m[(Wd.role == r).to_numpy()].sum() / 1e3)
    tbt = pd.concat([two_by_two(W, d) for d in D_BANDS], ignore_index=True)
    for _, r in tbt.iterrows():
        summary.put(quantity=f"2x2 {r.cell}", layer=r.layer, role=r.role, d_m=r.d_m, value=r.share)
    return tbt


def put_sensitivities(summary: Summary, W: pd.DataFrame, Wt: pd.DataFrame,
                      t_share: pd.Series) -> tuple[dict, dict]:
    """Decision 10: the testable share of edge, U over the testable windows only (window rule
    'ring_testable') and the within-component differences. Returns d -> testable U per role and
    d -> the observed within-component differences."""
    for r in ROLES:
        m = (Wt.role == r).to_numpy()
        summary.put(quantity="testable_share", role=r, value=t_share[r], n_windows=int(m.sum()),
                    edge_km=Wt.edge_m[m].sum() / 1e3)
    obs_t, within_obs = {}, {}
    for d in D_BANDS:
        share = pooled_share(Wt, Wt[f"U_d{d:g}"])
        obs_t[d] = share
        put_u_rows(summary, Wt, share, d, summary.level_test, "ring_testable")
        within = within_component(*comp_sums(W, W[f"U_d{d:g}"]))
        within_obs[d] = {key: float(x[0]) for key, x in within.items()}
        for key, ctrl in (("diff_matched_A", "matched_A"), ("diff_AE", "AE")):
            summary.put(quantity=f"within-component U(region)-U({ctrl})", d_m=d,
                        value=within_obs[d][key])
    return obs_t, within_obs


# ----------------------------------------------------------------- printout
def print_inputs(v: pd.DataFrame, W: pd.DataFrame, Wf: pd.DataFrame, drops: pd.DataFrame,
                 nan_counts: pd.DataFrame) -> None:
    print(f"  {len(v):,} vertices, {v.lobe_id.nunique():,} lobes, {v.component.nunique()} "
          f"components; {len(W):,} windows (preview rule: {len(Wf):,})", flush=True)
    print("\n  vertices dropped (step 1; a vertex can have several reasons):")
    print(indented(drops.round(1).to_string()), flush=True)
    if nan_counts.to_numpy().sum():
        print("\n  NaN distances among usable vertices (counted as +inf: no line there):")
        print(indented(nan_counts.to_string()))
    else:
        print("  no NaN distance among the usable vertices")


def print_calibration(cal: pd.DataFrame, W: pd.DataFrame, level_test: str, tau: float) -> str | None:
    """Print the calibration table; return the brief's calibration-stop message when fewer than
    half of the AE edge passes the selected test."""
    table = cal.pivot_table(index="test", columns="role", values="share_edge", sort=False)[list(ROLES)]
    print("\n  CALIBRATION: level-set (contour-like) share of edge length per role")
    print(indented(table.round(3).to_string()), flush=True)
    share = pooled_share(W, W.level)
    count = pooled_share(W, W.level, "count")
    if not share["AE"] < 0.5:
        print(f"\n  calibration: {100 * share['AE']:.0f}% of AE edge passes the selected test")
        return None
    msg = (f"CALIBRATION: only {100 * share['AE']:.0f}% of AE windows "
           f"({100 * count['AE']:.0f}% by count) pass the level-set test "
           f"({level_test} <= {tau:g} m + drop); region {100 * share['region']:.0f}%, "
           f"matched_A {100 * share['matched_A']:.0f}% (brief: stop and ask)")
    print("\n" + "!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
    return msg


def keep_testable_windows(W: pd.DataFrame, Wf: pd.DataFrame, wid: np.ndarray):
    """Decision 11: drop the windows with < MIN_DROP good probes and renumber each vertex's
    window (-1 for the vertices of dropped windows). Also returns the testable share of ALL
    edge per role."""
    keep = (W.n_drop >= MIN_DROP).to_numpy()
    new_row = np.full(len(W), -1, dtype=np.int64)
    new_row[keep] = np.arange(int(keep.sum()))
    wid = np.where(wid >= 0, new_row[np.maximum(wid, 0)], -1)
    share_all = pooled_share(W, keep)
    W = W[keep].reset_index(drop=True)
    Wf = Wf[(Wf.n_drop >= MIN_DROP).to_numpy()].reset_index(drop=True)
    level = pooled_share(W, W.level)
    print(f"\n  --windows testable: {len(W):,} windows with >= {MIN_DROP} good probes are analysed "
          f"(share of edge: " + ", ".join(f"{r} {share_all[r]:.3f}" for r in ROLES) + "); level-set "
          f"share among them: " + ", ".join(f"{r} {level[r]:.3f}" for r in ROLES), flush=True)
    if level["AE"] >= 0.5:
        print(f"  calibration among the analysed windows: {100 * level['AE']:.0f}% of AE edge passes")
    return W, Wf, wid, share_all


def u_line(d: float, share: pd.Series) -> str:
    return (f"    d = {d:>3g} m: region {share['region']:.3f}  matched_A {share['matched_A']:.3f}  "
            f"AE {share['AE']:.3f}  | region - matched_A {share['region'] - share['matched_A']:+.3f}  "
            f"region - AE {share['region'] - share['AE']:+.3f}")


def print_observed(obs: dict, obs_t: dict, within_obs: dict, t_share: pd.Series, sel: str) -> None:
    print(f"\n  U (not a level set AND on an S or M line), share of edge length, {sel}:")
    for d in D_BANDS:
        print(u_line(d, obs[("ring", "length", d, sel)]))
    print("  preview rule (cut after the drop, by window count):")
    for d in D_BANDS:
        share = obs[("filtered", "count", d, sel)]
        print(f"    d = {d:>3g} m: region {share['region']:.3f}  matched_A {share['matched_A']:.3f}  "
              f"AE {share['AE']:.3f}  | region - matched_A {share['region'] - share['matched_A']:+.3f}")
    print(f"  SENSITIVITY, testable windows only (>= {MIN_DROP} good probes; testable share of edge: "
          + ", ".join(f"{r} {t_share[r]:.3f}" for r in ROLES) + "):")
    for d in D_BANDS:
        print(u_line(d, obs_t[d]))
    print("  SENSITIVITY, within-component difference (components weighted by region edge):")
    for d in D_BANDS:
        print(f"    d = {d:>3g} m: region - matched_A {within_obs[d]['diff_matched_A']:+.3f}  "
              f"region - AE {within_obs[d]['diff_AE']:+.3f}")


def preview_numbers(Wf: pd.DataFrame, obs: dict, sel: str, d: float, role: str) -> dict:
    """This script's numbers under u_preview.py's window rule and window-count weights."""
    m = (Wf.role == role).to_numpy()
    numbers = {"windows": int(m.sum()), "level_set": pooled_share(Wf, Wf.level, "count")[role]}
    for column, layer in (("on_S", "S"), ("on_M", "M"), ("on_S_or_M", "SM")):
        if d in D_BANDS:
            numbers[column] = pooled_share(Wf, Wf[f"on_{layer}_d{d:g}"], "count")[role]
        else:
            numbers[column] = np.nan
    numbers["U"] = obs.get(("filtered", "count", d, sel), {}).get(role, np.nan)
    return numbers


def preview_crosscheck(a, Wf: pd.DataFrame, obs: dict, sel: str) -> pd.DataFrame:
    """Compare with the u_preview.py outputs given by --preview-csv; one row per number."""
    rows = []
    for d, path in sorted(a.previews.items(), reverse=True):
        preview = pd.read_csv(path).set_index("role")
        for r in ROLES:
            for column, x in preview_numbers(Wf, obs, sel, d, r).items():
                if column in preview.columns and r in preview.index:
                    theirs = float(preview.loc[r, column])
                    rows.append(dict(d_m=d, role=r, column=column, preview=theirs,
                                     this_script=float(x), abs_diff=abs(theirs - float(x))))
    xcheck = pd.DataFrame(rows)
    if len(xcheck):
        max_diff = xcheck.abs_diff.max()
        verdict = "AGREES" if max_diff < 1e-9 else "DIFFERS"
        note = ""
        if not (a.level_test == "range" and a.tau == TAU):
            note = " (note: the preview used range <= 0.76 m; this run's level test differs)"
        print(f"\n  CROSS-CHECK vs u_preview.py ({', '.join(str(p) for p in a.previews.values())}), "
              f"preview window rule and window-count weights: max |diff| {max_diff:.2e} over "
              f"{len(xcheck)} numbers -> {verdict}" + note, flush=True)
    return xcheck


# ----------------------------------------------------------------- bootstrap and null
def bootstrap_components(W: pd.DataFrame, Wt: pd.DataFrame, level_cols: list[str], var_name: dict,
                         n_boot: int, seed: int) -> tuple[dict, dict, int]:
    """Bootstrap over components for every pooled U and difference, and for decision 10's
    sensitivities on the same draws. Returns the primary draws by (d, level variant), the
    sensitivities' by (d, 'testable' / 'within'), and the number of components."""
    comps = np.array(sorted(W.component.unique()))
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    idx = rng.integers(0, len(comps), (n_boot, len(comps)))

    def sums(X, flag):
        num, den = comp_sums(X, flag)
        return num.reindex(comps).fillna(0), den.reindex(comps).fillna(0)

    boots = {}
    for d in D_BANDS:
        for lc in level_cols:
            num, den = sums(W, ~W[lc] & W[f"on_SM_d{d:g}"])
            boots[(d, var_name[lc])] = bootstrap(num, den, idx)
    boots_s = {}
    for d in D_BANDS:
        num, den = sums(Wt, Wt[f"U_d{d:g}"])
        boots_s[(d, "testable")] = bootstrap(num, den, idx)
        num, den = sums(W, W[f"U_d{d:g}"])
        boots_s[(d, "within")] = within_component(num, den, idx)
    return boots, boots_s, len(comps)


def changed_layer_files(meta: dict, line_info: dict) -> dict:
    """Sources whose files differ from the ones edge_layers recorded: source -> (was, now)."""
    recorded = meta.get("layer_files", {})
    changed = {}
    for source, path in line_info["layer_files"].items():
        if source in SOURCES and recorded.get(source) != path:
            changed[source] = (recorded.get(source), path)
    return changed


def self_check(ctx: NullContext, obs_u: dict, obs_t: dict) -> tuple[pd.DataFrame, bool, dict]:
    """Decision 4: a zero displacement through the null's code must give back the stored
    distances and the observed U(region). Prints the check; returns its table, whether it
    passed and the coverage counts at zero displacement."""
    t1 = time.time()
    zero = np.zeros(len(ctx.lobes))
    r0 = ctx.draw(zero, zero)
    rows = []
    for name in NULL_LAYERS:
        got, stored = r0["dist"][name], ctx.stored[name]
        nan_mismatch = int((np.isnan(got) != np.isnan(stored)).sum())
        both = ~np.isnan(got) & ~np.isnan(stored)
        max_diff = float(np.abs(got[both] - stored[both]).max()) if both.any() else 0.0
        rows.append(dict(layer=name, vertices=len(stored), max_abs_diff_m=max_diff,
                         nan_mismatch=nan_mismatch))
    table = pd.DataFrame(rows)
    u_diff = max(abs(r0["u"][d]["level"] - obs_u[d]["region"]) for d in D_BANDS)
    testable_diffs = []
    for d in D_BANDS:
        if np.isfinite(obs_t[d]["region"]):              # NaN: no testable region edge
            testable_diffs.append(abs(r0["u_testable"][d]["level"] - obs_t[d]["region"]))
    ut_diff = max(testable_diffs, default=0.0)
    ok = bool((table.max_abs_diff_m <= SELF_TOL_M).all() and (table.nan_mismatch == 0).all()
              and u_diff < 1e-12 and ut_diff < 1e-12)
    print(f"\n  SELF-CHECK (zero displacement vs stored distances, {len(ctx.iv):,} vertices) "
          f"[{time.time() - t1:.1f}s]:")
    print(indented(table.to_string(index=False)))
    print(f"  U(region) from the recomputed distances - observed: {u_diff:.1e} "
          f"(testable windows only: {ut_diff:.1e})")
    print(f"  SELF-CHECK {'PASSED' if ok else 'FAILED'} (tolerance {SELF_TOL_M:g} m)", flush=True)
    return table, ok, r0["cov"]


def null_row(draw: int, dx: np.ndarray, dy: np.ndarray, r: dict, obs_u: dict, obs_t: dict,
             level_cols: list[str], var_name: dict, comps: np.ndarray) -> dict:
    """One row of <prefix>_null.csv from a NullContext.draw result r. obs_u / obs_t: the
    observed U per role over all / testable windows (the controls' side of the differences)."""
    row = {"draw": draw, "mean_shift_m": float(np.hypot(dx, dy).mean())}
    for d in D_BANDS:
        sfx = f"_d{d:g}"
        u = r["u"][d]
        row["U_region" + sfx] = u["level"]
        row["diff_matched_A" + sfx] = u["level"] - obs_u[d]["matched_A"]
        row["diff_AE" + sfx] = u["level"] - obs_u[d]["AE"]
        for k in ("onS", "onM", "onSM"):
            row[f"{k}_region{sfx}"] = u[k]
        for lc in level_cols[1:]:
            row[f"U_region_{var_name[lc]}{sfx}"] = u[lc]
        for c, x in zip(comps, u["by_comp"]):
            row[f"U_region_c{c}{sfx}"] = x
        row["U_region_covered" + sfx] = r["u_covered"][d]["level"]
    for d in D_BANDS:                     # decision 6 (paired side) and 10a
        sfx = f"_d{d:g}"
        row["U_region_covered_obs" + sfx] = r["u_covered_obs"][d]
        u_testable = r["u_testable"][d]["level"]
        row["U_region_testable" + sfx] = u_testable
        row["diff_matched_A_testable" + sfx] = u_testable - obs_t[d]["matched_A"]
        row["diff_AE_testable" + sfx] = u_testable - obs_t[d]["AE"]
    for k, n in r["cov"].items():
        row[f"n_{k}"] = n
    return row


def run_null(ctx: NullContext, n_draws: int, seed: int, jobs: int, obs_u: dict, obs_t: dict,
             level_cols: list[str], var_name: dict) -> tuple[pd.DataFrame, float]:
    """The displaced-edge null draws on `jobs` threads (GEOS releases the GIL); one row per draw,
    and the wall time."""
    seeds = np.random.SeedSequence(seed, spawn_key=(2,)).spawn(n_draws)
    lock = threading.Lock()
    done = 0
    t1 = time.time()

    def one(i):
        nonlocal done
        rng = np.random.default_rng(seeds[i])
        dx, dy = shift_vectors(len(ctx.lobes), rng)
        t = time.time()
        r = ctx.draw(dx, dy)
        row = null_row(i, dx, dy, r, obs_u, obs_t, level_cols, var_name, ctx.comps)
        row["seconds"] = time.time() - t
        with lock:
            done += 1
            if done % 20 == 0 or done == n_draws:
                print(f"    null draws done: {done}/{n_draws} [{time.time() - t1:.0f}s]", flush=True)
        return row

    print(f"\n  null: {n_draws} draws, {jobs} threads", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        null = pd.DataFrame(list(ex.map(one, range(n_draws))))
    return null, time.time() - t1


def add_within_null(null: pd.DataFrame, W: pd.DataFrame, per_comp: dict) -> pd.DataFrame:
    """Decision 10b in the null: only the region's U_c moves, the weights w_c stay fixed."""
    null = null.sort_values("draw").reset_index(drop=True)
    for d in D_BANDS:
        share = per_comp[d]
        _, den = comp_sums(W, W[f"U_d{d:g}"])
        for key, ctrl in (("diff_matched_A", "matched_A"), ("diff_AE", "AE")):
            ok = []
            for c in share.index:
                if (np.isfinite(share.loc[c, "region"] - share.loc[c, ctrl])
                        and f"U_region_c{c}_d{d:g}" in null):
                    ok.append(c)
            weight = den.loc[ok, "region"].to_numpy(float)
            u_null = null[[f"U_region_c{c}_d{d:g}" for c in ok]].to_numpy(float)
            if weight.sum() > 0:
                null[f"within_{key}_d{d:g}"] = ((u_null - share.loc[ok, ctrl].to_numpy(float))
                                                @ weight) / weight.sum()
            else:
                null[f"within_{key}_d{d:g}"] = np.nan
    return null


def null_stats(null: pd.DataFrame, col: str, observed: float) -> dict:
    """Mean, 95% range and percentile of `observed` in the null column `col` ({} if none)."""
    if col not in null or not len(null):
        return {}
    x = null[col].to_numpy(float)
    pct, p = null_percentile(observed, x)
    return dict(null_mean=float(np.nanmean(x)), null_q025=float(np.nanpercentile(x, 2.5)),
                null_q975=float(np.nanpercentile(x, 97.5)), null_pct=pct, null_p_upper=p)


def boot_key(row: dict) -> str:
    """The bootstrap() key of a U or difference row."""
    if row["quantity"] == "U":
        return row["role"]
    return DIFF_KEYS[row["quantity"]]


def is_pooled_primary(row: dict) -> bool:
    return (row["window_rule"] == "ring" and row["weight"] == "length"
            and row["component"] == "pooled")


def add_boot_and_null(summary: Summary, boots: dict, boots_s: dict, null: pd.DataFrame,
                      obs_u: dict, sel: str) -> None:
    """Fill the bootstrap and null columns of the summary rows they apply to, and add the
    covered-windows rows (decision 6)."""
    for row in summary.rows:
        q, d, vn = row["quantity"], row["d_m"], row["level_test"]
        if not is_pooled_primary(row) or q not in U_QUANTITIES or not np.isfinite(d):
            continue
        lo, hi, n = ci(boots[(d, vn)][boot_key(row)])
        row.update(boot_lo=lo, boot_hi=hi, boot_n=n)
        sfx = f"_d{d:g}"
        if q == "U" and row["role"] == "region":
            col = ("U_region" if vn == sel else f"U_region_{vn}") + sfx
            row.update(null_stats(null, col, row["value"]))
        elif q != "U" and vn == sel:
            row.update(null_stats(null, boot_key(row) + sfx, row["value"]))

    for row in summary.rows:              # per-component U(region) against its own null
        if (row["quantity"] == "U" and row["role"] == "region" and row["component"] != "pooled"
                and row["window_rule"] == "ring"):
            col = f"U_region_c{row['component']}_d{row['d_m']:g}"
            row.update(null_stats(null, col, row["value"]))

    online_col = {"on_S": "onS", "on_M": "onM", "on_SM": "onSM"}
    for row in summary.rows:              # on-line shares of the region edge against the null
        if (row["quantity"] in online_col and row["role"] == "region" and row["d_m"] in D_BANDS
                and is_pooled_primary(row)):
            col = f"{online_col[row['quantity']]}_region_d{row['d_m']:g}"
            row.update(null_stats(null, col, row["value"]))

    for row in summary.rows:              # decision 10
        q, sfx = row["quantity"], f"_d{row['d_m']:g}"
        if row["window_rule"] == "ring_testable" and q in U_QUANTITIES:
            key = boot_key(row)
            lo, hi, n = ci(boots_s[(row["d_m"], "testable")][key])
            row.update(boot_lo=lo, boot_hi=hi, boot_n=n)
            if q != "U":
                row.update(null_stats(null, f"{key}_testable" + sfx, row["value"]))
            elif row["role"] == "region":
                row.update(null_stats(null, "U_region_testable" + sfx, row["value"]))
        elif q.startswith("within-component "):
            key = "diff_matched_A" if q.endswith("(matched_A)") else "diff_AE"
            lo, hi, n = ci(boots_s[(row["d_m"], "within")][key])
            row.update(boot_lo=lo, boot_hi=hi, boot_n=n,
                       **null_stats(null, f"within_{key}{sfx}", row["value"]))

    # decision 6, paired: value = the observed U(region) over each draw's kept windows (mean
    # over draws); the percentile ranks null_kept - obs_kept against 0
    for d in D_BANDS:
        sfx = f"_d{d:g}"
        covered = {"value": obs_u[d]["region"]}
        if len(null):
            x = null["U_region_covered" + sfx].to_numpy(float)
            x_obs = null["U_region_covered_obs" + sfx].to_numpy(float)
            pct, p = null_percentile(0.0, x - x_obs)
            covered = dict(value=float(np.nanmean(x_obs)), null_mean=float(np.nanmean(x)),
                           null_q025=float(np.nanpercentile(x, 2.5)),
                           null_q975=float(np.nanpercentile(x, 97.5)), null_pct=pct, null_p_upper=p)
        summary.put(quantity="U(region), covered windows only (null)", role="region", d_m=d,
                    **covered)


@dataclass
class NullRun:
    """What the report and the final printout need from the displaced-edge null."""
    ctx: NullContext
    null: pd.DataFrame          # one row per draw
    seconds: float              # wall time of the draws
    selfcheck: pd.DataFrame     # zero-displacement check per layer
    ok_self: bool
    zero_cov: dict              # coverage counts at zero displacement
    line_info: dict             # load_lines' layer files and flag mismatches
    cover_files: dict           # files of the coverage geometry
    file_diff: dict             # layer files that differ from the ones edge_layers recorded


def print_final(S: pd.DataFrame, sel: str, run: NullRun) -> None:
    null, ctx = run.null, run.ctx
    m = ((S.window_rule == "ring") & (S.weight == "length") & (S.component == "pooled")
         & (S.level_test == sel))
    print(f"\n  RESULT ({sel}; edge-length weights; bootstrap 95% over components; null percentile):")
    for d in D_BANDS:
        for q in ("U(region)-U(matched_A)", "U(region)-U(AE)"):
            r = S[m & (S.quantity == q) & (S.d_m == d)].iloc[0]
            print(f"    d = {d:>3g} m  {q:<24} {r.value:+.3f}  [{r.boot_lo:+.3f}, {r.boot_hi:+.3f}]  "
                  f"null pct {r.null_pct:.1f}")
        r = S[m & (S.quantity == "U") & (S.role == "region") & (S.d_m == d)].iloc[0]
        print(f"    d = {d:>3g} m  U(region) {r.value:.3f}; null mean {r.null_mean:.3f} "
              f"[{r.null_q025:.3f}, {r.null_q975:.3f}]; percentile {r.null_pct:.1f}; "
              f"p_upper {r.null_p_upper:.3f}")
    base = (S.weight == "length") & (S.component == "pooled") & (S.level_test == sel)
    ts = S[base & (S.quantity == "testable_share")].set_index("role").value
    print(f"  SENSITIVITY (decision 10a, not the primary): testable windows only (>= {MIN_DROP} good "
          f"probes; testable share of edge " + ", ".join(f"{k} {ts[k]:.3f}" for k in ROLES) + ")")
    mt = base & (S.window_rule == "ring_testable")
    for d in D_BANDS:
        u = S[mt & (S.quantity == "U") & (S.d_m == d)].set_index("role")
        print(f"    d = {d:>3g} m  U region {u.value['region']:.3f}  matched_A "
              f"{u.value['matched_A']:.3f}  AE {u.value['AE']:.3f}")
        for q in ("U(region)-U(matched_A)", "U(region)-U(AE)"):
            r = S[mt & (S.quantity == q) & (S.d_m == d)].iloc[0]
            print(f"    d = {d:>3g} m  {q:<24} {r.value:+.3f}  [{r.boot_lo:+.3f}, {r.boot_hi:+.3f}]  "
                  f"null pct {r.null_pct:.1f}")
    print("  SENSITIVITY (decision 10b, not the primary): within-component difference, "
          "components weighted by region edge")
    for d in D_BANDS:
        for q in ("U(region)-U(matched_A)", "U(region)-U(AE)"):
            r = S[base & (S.quantity == f"within-component {q}") & (S.d_m == d)].iloc[0]
            print(f"    d = {d:>3g} m  {q:<24} {r.value:+.3f}  [{r.boot_lo:+.3f}, {r.boot_hi:+.3f}]  "
                  f"null pct {r.null_pct:.1f}")
    print("  SENSITIVITY (decision 6): covered windows only, paired (observed and null over the "
          "same kept windows):")
    for _, r in S[S.quantity == "U(region), covered windows only (null)"].iterrows():
        print(f"    d = {r.d_m:>3g} m  U(region) observed {r.value:.3f}; null mean {r.null_mean:.3f} "
              f"[{r.null_q025:.3f}, {r.null_q975:.3f}]; paired percentile {r.null_pct:.1f}; "
              f"p_upper {r.null_p_upper:.3f}")
    if len(null):
        total = len(ctx.iv) * len(null)
        counts = []
        for col in null.columns:
            if col.startswith("n_") and col != "n_windows_touched":
                counts.append(f"{col[2:]} {int(null[col].sum()):,}")
        print(f"  displaced-vertex coverage over {len(null)} draws ({total:,} vertex-draws): "
              + ", ".join(counts)
              + f"; windows touched per draw: mean {null.n_windows_touched.mean():.1f} of {ctx.nW:,}")
        print(f"  (at zero displacement: {run.zero_cov})")
        if null.n_outside_bbox.sum():
            print("  WARNING: displaced vertices fell outside their component's bbox: their soil "
                  "distances are upper bounds")


# ----------------------------------------------------------------- report
REPORT_DEFINITIONS = [
    "- A *lobe* is one single-part flood-map polygon; its boundary rings are sampled every "
    "~30 m into *vertices*. A *window* is 10 consecutive vertices of a ring (300 m of edge).",
    "- *Level set* (contour-like): along the window the ground elevation z stays within tau "
    "(statistic: range, IQR or detrended range) and the ground rises going out of the polygon "
    "(median z_out - z_in > 0 over >= 5 vertices whose probes landed right). That is what a flat "
    "water surface meeting the ground looks like.",
    "- *On a line*: the window's median distance to the layer is <= d (100 m main, 30 m tight). "
    "S = soil-survey polygon edges; M = county lines and the lobe's own FIRM panel (map sheet) "
    "edges; canals on their own.",
    "- **U** = share of edge length that is NOT a level set AND on an S or M line. Primary: "
    "U(region) - U(matched_A). Secondary: U(region) - U(AE).",
    "- Roles: region = Zone A lobes of a selected region; matched_A = unselected Zone A lobes in "
    "the same test tiles, matched on perimeter; AE = Zone AE lobes in the same tiles.",
    "- *Bootstrap*: resample components (regions) with replacement; 95% percentile interval. "
    "*Null*: each region lobe shifted 300-1,000 m in a random direction, its S/M distances "
    "recomputed, its windows' level flags kept; percentile = where the observed value falls "
    "among the null draws (mid-rank).\n",
]
ROLE_ROWS = [("U", "region"), ("U", "matched_A"), ("U", "AE"),
             ("U(region)-U(matched_A)", ""), ("U(region)-U(AE)", "")]


def first_row(S: pd.DataFrame, quantity: str, role: str = "") -> pd.Series:
    return S[(S.quantity == quantity) & (S.role == role)].iloc[0]


def edge_km(x: pd.Series) -> float:
    return x.sum() / 1e3


def report_intro(a, calib_msg: str | None, calibrate_only: bool) -> list[str]:
    if calibrate_only:
        n_null, n_boot, tail = 0, 0, " **--calibrate-only: no null, no bootstrap.**"
    else:
        n_null, n_boot, tail = a.null_draws, a.boot, ""
    lines = [f"# Boundary statistic (O5): {a.prefix}\n",
             f"Written by `aotil/boundary_stat.py` on {datetime.now().isoformat(timespec='seconds')}. "
             f"Input `{a.vertex_layers}`. Level test **{a.level_test} <= {a.tau:g} m + drop**; "
             f"seed {a.seed}; {n_null} null draws; {n_boot} bootstrap draws." + tail + "\n"]
    if a.windows == "testable":
        lines.append(f"**Windows: testable only** (`--windows testable`, decision 11): every table "
                     f"below except the calibration uses only the windows with >= {MIN_DROP} "
                     f"good-probe vertices.\n")
    if calib_msg:
        lines.append(f"> **{calib_msg}**\n")
    lines.append("## What the numbers are\n")
    return lines + REPORT_DEFINITIONS


def report_inputs(drops, nan_counts, notes, W, Wf) -> list[str]:
    lines = ["## Vertices dropped (step 1)\n",
             "A vertex can have several reasons, so the reason columns overlap; `dropped_any` is "
             "the union. `in_windows` = usable vertices in a kept window (this script's rule: "
             "windows cut on the ring's own sequence); `in_windows_preview_rule` = the same under "
             "u_preview.py's rule (cut after the drop).\n",
             md_table(drops, {"usable_km": "{:,.1f}", "in_windows_km": "{:,.1f}"})]
    if nan_counts.to_numpy().sum():
        nan_text = "\n\n" + md_table(nan_counts)
    else:
        nan_text = "none."
    lines.append("\nNaN distances among usable vertices (counted as +inf, 'no line of that layer "
                 "there'): " + nan_text + "\n")
    if notes:
        lines.append("Notes: " + "; ".join(notes) + "\n")
    lines.append("## Windows\n")
    table = W.groupby("role").agg(windows=("edge_m", "size"), edge_km=("edge_m", edge_km),
                                  lobes=("lobe_id", "nunique")).reindex(list(ROLES))
    table["windows_preview_rule"] = Wf.groupby("role").size().reindex(list(ROLES))
    lines += [md_table(table, {"edge_km": "{:,.1f}"}), ""]
    return lines


def report_calibration(cal: pd.DataFrame) -> list[str]:
    table = cal.copy()
    table["v"] = [f"{x:.3f} ({y:.3f})" for x, y in zip(table.share_edge, table.share_windows)]
    pivot = table.pivot_table(index="test", columns="role", values="v", aggfunc="first", sort=False)
    return ["## Calibration: level-set share per role\n",
            "Share of edge length (share of windows in brackets) that passes each test. Zone AE "
            "edges are the intended calibration: a detailed-study edge should be contour-like. "
            "`alone` rows drop the across-edge drop condition; the last four rows are the drop "
            "condition alone and its two parts: the windows with >= 5 good-probe vertices (the "
            "*testable* windows: only these can pass the drop test at all) and the sign of the "
            "median drop, once over all windows (a window with no good probe has no median and "
            "counts as failing) and once over the windows with >= 1 good probe only.\n",
            md_table(pivot[list(ROLES)]), ""]


def report_pooled(prim: pd.DataFrame, sel: str, calibrate_only: bool) -> list[str]:
    lines = ["## U per role (pooled over components by edge length)\n"]
    for d in D_BANDS:
        sub = prim[(prim.d_m == d) & (prim.level_test == sel)]
        rows = []
        for q, r in ROLE_ROWS:
            x = first_row(sub, q, r)
            rows.append({"quantity": q + (f" {r}" if r else ""), "value": x.value,
                         "boot 2.5%": x.boot_lo, "boot 97.5%": x.boot_hi,
                         "null mean": x.null_mean, "null 2.5%": x.null_q025,
                         "null 97.5%": x.null_q975, "null pct": x.null_pct,
                         "p_upper": x.null_p_upper, "edge km": x.edge_km})
        lines.append(f"**d = {d:g} m**\n")
        lines.append(md_table(pd.DataFrame(rows), {"null pct": "{:.1f}", "edge km": "{:,.1f}"},
                              index=False))
        lines.append("")
    if calibrate_only:
        return lines
    lines.append("The two null percentiles of a d are equal by construction: matched_A and AE are "
                 "not displaced, so the null difference is null U(region) minus a constant.\n")
    lines.append("On-line shares of the region edge, observed and in the null (the base rate: how "
                 "often a displaced copy of the same edge lands on a line):\n")
    rows = []
    for d in D_BANDS:
        for layer in ("S", "M", "SM"):
            x = first_row(prim[prim.d_m == d], f"on_{layer}", "region")
            rows.append({"d_m": d, "layer": layer, "observed": x.value, "null mean": x.null_mean,
                         "null 2.5%": x.null_q025, "null 97.5%": x.null_q975,
                         "null pct": x.null_pct})
    lines += [md_table(pd.DataFrame(rows), {"d_m": "{:g}", "null pct": "{:.1f}"}, index=False), ""]
    return lines


def report_components(S: pd.DataFrame, calibrate_only: bool) -> list[str]:
    lines = ["## U per component\n",
             "U per role for each component (selected region); `region null pct` = the "
             "component's U(region) against its own null draws.\n"]
    for d in D_BANDS:
        sub = S[(S.window_rule == "ring") & (S.component != "pooled") & (S.d_m == d)
                & (S.quantity == "U")]
        sub = sub.assign(component=sub.component.astype(int))
        table = sub.pivot_table(index="component", columns="role", values="value")[list(ROLES)]
        km = sub.pivot_table(index="component", columns="role", values="edge_km")[list(ROLES)]
        table["region-matched_A"] = table.region - table.matched_A
        table["region-AE"] = table.region - table.AE
        for r in ROLES:
            table[f"{r} km"] = km[r]
        if not calibrate_only:
            table["region null pct"] = sub[sub.role == "region"].set_index("component").null_pct
        fmt = {f"{r} km": "{:,.1f}" for r in ROLES}
        fmt["region null pct"] = "{:.1f}"
        lines += [f"**d = {d:g} m**\n", md_table(table, fmt), ""]
    level = S[(S.quantity == "level_share") & (S.component != "pooled")]
    level = level.assign(component=level.component.astype(int))
    level = level.pivot_table(index="component", columns="role", values="value")[list(ROLES)]
    lines += ["Level-set share per component (selected test):\n", md_table(level), ""]
    return lines


def report_two_by_two(tbt: pd.DataFrame) -> list[str]:
    lines = ["## 2 x 2 tables: level set x on the layer\n",
             "Share of each role's edge length in each cell (rows sum to 1 per layer and role), "
             "selected level test.\n"]
    for d in D_BANDS:
        table = tbt[tbt.d_m == d].pivot_table(index=["layer", "role"], columns="cell",
                                              values="share", sort=False)
        table = table[["level & on", "level & off", "not level & on", "not level & off"]]
        lines += [f"**d = {d:g} m** (layers: S soil, M map lines, SM either, canals)\n",
                  md_table(table), ""]
    return lines


def report_level_tests(prim: pd.DataFrame, sel: str) -> list[str]:
    lines = ["## Sensitivities: other level tests\n",
             "U with every level test and tau of the calibration grid; null percentile of "
             "U(region) (the level flags are kept fixed in the null, so every test shares the "
             "same draws).\n"]
    variants = [sel] + sorted(set(prim.level_test) - {sel})      # the selected test first
    for d in D_BANDS:
        rows = []
        for vn in variants:
            sub = prim[(prim.d_m == d) & (prim.level_test == vn)]
            u = first_row(sub, "U", "region")
            dm = first_row(sub, "U(region)-U(matched_A)")
            rows.append({"level test": vn + (" (selected)" if vn == sel else ""),
                         "U region": u.value, "U matched_A": first_row(sub, "U", "matched_A").value,
                         "U AE": first_row(sub, "U", "AE").value, "region-matched_A": dm.value,
                         "boot 2.5%": dm.boot_lo, "boot 97.5%": dm.boot_hi,
                         "region-AE": first_row(sub, "U(region)-U(AE)").value,
                         "region null pct": u.null_pct})
        lines += [f"**d = {d:g} m**\n",
                  md_table(pd.DataFrame(rows), {"region null pct": "{:.1f}"}, index=False), ""]
    return lines


def report_sensitivities(S: pd.DataFrame, prim: pd.DataFrame, sel: str) -> list[str]:
    base = S[(S.weight == "length") & (S.component == "pooled") & (S.level_test == sel)]
    primary = prim[prim.level_test == sel]
    ts = base[base.quantity == "testable_share"].set_index("role")
    lines = ["## Sensitivities for the calibration stop (decision 10; not the primary)\n",
             f"**Testable windows only.** A window with fewer than {MIN_DROP} good-probe vertices "
             f"cannot pass the drop test, so it is never a level set and counts in U whenever it "
             f"is on a line. Share of edge length in testable windows: "
             + ", ".join(f"{r} {ts.value[r]:.3f}" for r in ROLES)
             + ". Below, U and the differences over the testable windows only (same bootstrap "
               "draws; the null leaves the untestable region windows out of every draw); "
               "`primary` = the same quantity over all windows.\n"]
    for d in D_BANDS:
        rows = []
        for q, r in ROLE_ROWS:
            x = first_row(base[(base.window_rule == "ring_testable") & (base.d_m == d)], q, r)
            p0 = first_row(primary[primary.d_m == d], q, r)
            rows.append({"quantity": q + (f" {r}" if r else ""), "testable only": x.value,
                         "boot 2.5%": x.boot_lo, "boot 97.5%": x.boot_hi, "null pct": x.null_pct,
                         "p_upper": x.null_p_upper, "edge km": x.edge_km, "primary": p0.value})
        lines += [f"**d = {d:g} m**\n",
                  md_table(pd.DataFrame(rows), {"null pct": "{:.1f}", "edge km": "{:,.1f}"},
                           index=False), ""]
    lines.append("**Within-component difference.** sum over components c of w_c (U_c(region) - "
                 "U_c(control)), w_c = component c's share of the region's edge length: pooled "
                 "U(region) minus the control's U re-weighted to the region's mix of components, "
                 "so where each role's edge lies across components no longer enters. Components "
                 "without edge of the control are left out. Same bootstrap draws (the weights are "
                 "redrawn with the components); in the null only the region's U_c moves. "
                 "`primary` = the pooled difference.\n")
    rows = []
    for d in D_BANDS:
        for q in ("U(region)-U(matched_A)", "U(region)-U(AE)"):
            x = base[(base.quantity == f"within-component {q}") & (base.d_m == d)].iloc[0]
            p0 = first_row(primary[primary.d_m == d], q)
            rows.append({"d_m": d, "quantity": q, "within-component": x.value,
                         "boot 2.5%": x.boot_lo, "boot 97.5%": x.boot_hi, "null pct": x.null_pct,
                         "p_upper": x.null_p_upper, "primary": p0.value})
    lines += [md_table(pd.DataFrame(rows), {"d_m": "{:g}", "null pct": "{:.1f}"}, index=False), ""]
    return lines


def report_window_rules(S: pd.DataFrame, sel: str, xcheck: pd.DataFrame, previews: dict) -> list[str]:
    lines = ["## Window rule and weights vs the preview\n",
             "`ring` = this script (windows cut on the ring's own vertex sequence, so a window "
             "never spans a dropped stretch); `filtered` = u_preview.py (vertices dropped first, "
             "then cut every 10 kept vertices). `length` = edge-length weights; `count` = every "
             "window weighs 1 (the preview's). `ring_dedup` = ring/length with each physical lobe "
             "counted once (a lobe built for two components otherwise counts in both).\n"]
    pooled = S[S.component == "pooled"]
    rows = []
    for d in D_BANDS:
        for rule, weight in (("filtered", "count"), ("filtered", "length"), ("ring", "count"),
                             ("ring", "length"), ("ring_dedup", "length")):
            same = pooled[(pooled.window_rule == rule) & (pooled.weight == weight)]
            sub = same[(same.d_m == d) & (same.level_test == sel) & (same.quantity == "U")]
            if not len(sub):
                continue
            u = sub.set_index("role").value
            level = same[same.quantity == "level_share"].set_index("role").value
            rows.append({"d_m": d, "rule": rule, "weight": weight, "U region": u["region"],
                         "U matched_A": u["matched_A"], "U AE": u["AE"],
                         "region-matched_A": u["region"] - u["matched_A"],
                         "region-AE": u["region"] - u["AE"],
                         "level region": level.get("region", np.nan),
                         "level matched_A": level.get("matched_A", np.nan),
                         "level AE": level.get("AE", np.nan)})
    lines += [md_table(pd.DataFrame(rows), {"d_m": "{:g}"}, index=False), ""]
    if len(xcheck):
        lines.append(f"Cross-check against u_preview.py's own output "
                     f"({', '.join(f'`{p}`' for p in previews.values())}): max |difference| "
                     f"{xcheck.abs_diff.max():.2e} over {len(xcheck)} numbers (preview rule, "
                     f"window-count weights).\n")
    return lines


def report_null_checks(a, run: NullRun, S: pd.DataFrame, obs_u: dict) -> list[str]:
    ctx, null = run.ctx, run.null
    if run.file_diff:
        files_note = f" Differs from the files edge_layers recorded: {run.file_diff}."
    else:
        files_note = " Same files as edge_layers recorded."
    lines = ["## Displaced-edge null: checks\n",
             f"- Line files (edge_layers' loaders): `{run.line_info['layer_files']}`; coverage "
             f"geometry: `{run.cover_files}`." + files_note,
             f"- edge_layers flags recomputed on the region vertices vs the stored columns "
             f"(mismatching vertices): {run.line_info['flag_mismatch']}.",
             f"- Displaced per draw: {len(ctx.iv):,} usable vertices in {ctx.nW:,} region windows "
             f"of {len(ctx.lobes):,} lobes. {len(null)} draws in {run.seconds:.0f} s on "
             f"{a.jobs} threads.",
             f"- **Self-check** (zero displacement through the null's code vs the stored "
             f"distances): {'PASSED' if run.ok_self else 'FAILED'}.\n",
             md_table(run.selfcheck, {"max_abs_diff_m": "{:.2e}"}, index=False),
             "",
             "Coverage of the displaced vertices, summed over draws (vertex-draws; decision 6). "
             "outside_bbox: outside the component's analysis box, where soil lines were loaded; "
             "beyond_margin: soil distance larger than the distance to the box edge (then only an "
             "upper bound); no_map_unit: in no SSURGO polygon (open water beyond a survey: no soil "
             "line nearby); outside_state / outside_own_panels: beyond the county layer / the "
             "lobe's own map sheets.\n"]
    cols = [c for c in null.columns if c.startswith("n_")]
    cov = pd.DataFrame({"total over draws": null[cols].sum(), "mean per draw": null[cols].mean(),
                        "max per draw": null[cols].max()})
    cov["share of vertex-draws"] = cov["total over draws"] / (len(ctx.iv) * max(1, len(null)))
    cov.loc["n_windows_touched", "share of vertex-draws"] = np.nan
    cov.index = [c[2:] for c in cov.index]
    lines.append(md_table(cov, {"share of vertex-draws": "{:.5f}", "mean per draw": "{:,.1f}"}))
    lines.append(f"\n`windows_touched` = region windows with at least one such vertex in a draw "
                 f"(of {ctx.nW:,}). At zero displacement: {run.zero_cov}.\n")
    sub = S[S.quantity == "U(region), covered windows only (null)"]
    sub = sub.assign(observed_all=[obs_u[d]["region"] for d in sub.d_m])
    lines.append("U(region) with the touched windows left out of each draw, PAIRED: each draw's "
                 "null is compared with the observed U(region) over the same kept windows "
                 "(`value` = that observed U, averaged over draws; `observed_all` = over all "
                 "windows). `null_pct` = mid-rank share of draws whose null is below their own "
                 "observed value; `null_p_upper` = (1 + #draws with null >= observed) / "
                 "(1 + #draws).\n")
    lines.append(md_table(sub[["d_m", "observed_all", "value", "null_mean", "null_q025",
                               "null_q975", "null_pct", "null_p_upper"]],
                          {"d_m": "{:g}", "null_pct": "{:.1f}"}, index=False))
    lines.append("")
    return lines


def report_files(a, calibrate_only: bool) -> list[str]:
    names = ["windows.parquet", "calibration.csv", "summary.csv"]
    if not calibrate_only:
        names.append("null.csv")
    lines = ["## Files\n"]
    for name in names:
        p = a.out / f"{a.prefix}_{name}"
        if p.exists():
            lines.append(f"- `{p}`")
    return lines


def write_report(path: Path, a, sel: str, drops: pd.DataFrame, nan_counts: pd.DataFrame,
                 notes: list[str], W: pd.DataFrame, Wf: pd.DataFrame, cal: pd.DataFrame,
                 calib_msg: str | None, S: pd.DataFrame, obs_u: dict, tbt: pd.DataFrame,
                 xcheck: pd.DataFrame, run: NullRun | None = None) -> None:
    """The report: tables, each quantity explained in a line. run = None: --calibrate-only."""
    calibrate_only = run is None
    prim = S[(S.window_rule == "ring") & (S.weight == "length") & (S.component == "pooled")]
    lines = report_intro(a, calib_msg, calibrate_only)
    lines += report_inputs(drops, nan_counts, notes, W, Wf)
    lines += report_calibration(cal)
    lines += report_pooled(prim, sel, calibrate_only)
    lines += report_components(S, calibrate_only)
    lines += report_two_by_two(tbt)
    lines += report_level_tests(prim, sel)
    lines += report_sensitivities(S, prim, sel)
    lines += report_window_rules(S, sel, xcheck, a.previews)
    if not calibrate_only:
        lines += report_null_checks(a, run, S, obs_u)
    lines += report_files(a, calibrate_only)
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- main
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vertex-layers", required=True, type=Path,
                    help="<prefix>_vertex_layers.parquet from aotil/edge_layers.py")
    ap.add_argument("--lobes", type=Path, help="default: <prefix>_lobes.parquet beside --vertex-layers")
    ap.add_argument("--bboxes", type=Path, help="default: <prefix>_bboxes.parquet beside --vertex-layers")
    ap.add_argument("--gis-dir", type=Path, default=el.GIS_DIR,
                    help="as in edge_layers.py: ssurgo/<prefix>_mulines.parquet, "
                         "county_lines_<ST>.parquet, panel_edges_<ST>.parquet")
    ap.add_argument("--layer-file", action="append", default=[], metavar="SOURCE=PATH",
                    help=f"explicit file for a source ({', '.join(SOURCES)}), as in edge_layers.py; "
                         f"repeatable")
    ap.add_argument("--state", help="default: the state edge_layers recorded (else FL)")
    ap.add_argument("--state-tol-m", type=float,
                    help="edge_layers' --state-tol-m; default: the value it recorded (else 1000)")
    ap.add_argument("--prefix", help="default: from the --vertex-layers file name")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--level-test", choices=TESTS, default="range",
                    help="the level-set statistic U uses (default range)")
    ap.add_argument("--tau", type=float, default=TAU, help=f"level-set threshold, m (default {TAU})")
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000, help="component bootstrap draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="threads for the null draws (GEOS releases the GIL)")
    ap.add_argument("--windows", choices=("all", "testable"), default="all",
                    help="windows the statistic, bootstrap and null use: all (default) or only "
                         f"those with >= {MIN_DROP} good-probe vertices (decision 11); the "
                         "calibration is always over all windows")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="windows and the calibration table only; no null, no bootstrap")
    ap.add_argument("--preview-csv", action="append", default=[], metavar="D=PATH",
                    help="u_preview.py output at d = D m, compared with this script's numbers "
                         "under the preview's window rule; repeatable")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    a.prefix = a.prefix or prefix_of(a.vertex_layers)
    a.lobes_p = a.lobes or a.vertex_layers.with_name(f"{a.prefix}_lobes.parquet")
    a.bboxes_p = a.bboxes or a.vertex_layers.with_name(f"{a.prefix}_bboxes.parquet")
    a.files = {}
    for item in a.layer_file:
        source, _, path = item.partition("=")
        if source not in SOURCES or not path:
            ap.error(f"--layer-file {item}: expected SOURCE=PATH with SOURCE in {list(SOURCES)} "
                     f"(the sources of S and M)")
        a.files[source] = Path(path)
    a.previews = {}
    for item in a.preview_csv:
        d, _, path = item.partition("=")
        try:
            a.previews[float(d)] = Path(path)
        except ValueError:
            ap.error(f"--preview-csv {item}: expected D=PATH")
    if a.null_draws < 0 or a.boot < 0:
        ap.error("--null-draws and --boot must be >= 0")
    return a


def main(argv=None) -> int:
    a = parse(argv)
    t0 = time.time()
    outs = {name: a.out / f"{a.prefix}_{name}" for name in OUTPUT_NAMES}
    for p in (a.vertex_layers, a.lobes_p, a.bboxes_p):
        if not p.exists():
            raise SystemExit(f"{p} missing")
    existing = [p.name for p in outs.values() if p.exists()]
    if existing and not a.overwrite:
        raise SystemExit(f"{a.out}: {existing} exist; --overwrite to redo")
    a.out.mkdir(parents=True, exist_ok=True)
    sel = variant(a.level_test, a.tau)
    print(f"=== boundary_stat {a.prefix} ===", flush=True)
    print(f"  level test: {a.level_test} <= {a.tau:g} m + drop; d = "
          f"{', '.join(f'{d:g}' for d in D_BANDS)} m; seed {a.seed}", flush=True)

    # vertices, drops, windows
    v, meta, notes = read_vertices(a)
    a.state = a.state or meta.get("state", "FL")
    if a.state_tol_m is None:
        a.state_tol_m = float(meta.get("state_tol_m", 1000.0))
    bb = pd.read_parquet(a.bboxes_p)
    bb = bb.astype({"component": np.int32}).set_index("component")
    flags = drop_flags(v)
    usable = usable_mask(flags)
    W, wid, slot = build_windows(v, usable, "ring", a.level_test, a.tau)
    Wf, wid_f, _ = build_windows(v, usable, "filtered", a.level_test, a.tau)
    drops = drop_table(v, flags, usable, wid, wid_f)
    nan_counts = nan_distance_counts(v, usable)
    print_inputs(v, W, Wf, drops, nan_counts)

    # calibration, always over all windows
    cal = calibration_table(W, a.level_test, a.tau)
    cal.to_csv(outs["calibration.csv"], index=False)
    calib_msg = print_calibration(cal, W, a.level_test, a.tau)
    W.to_parquet(outs["windows.parquet"], index=False)
    t_share_all = None
    if a.windows == "testable":
        W, Wf, wid, t_share_all = keep_testable_windows(W, Wf, wid)

    # observed numbers
    level_cols, var_name = level_columns(sel)
    summary = Summary(sel, a.windows)
    obs = put_observed(summary, W, Wf, level_cols, var_name)
    obs_u = {d: obs[("ring", "length", d, sel)] for d in D_BANDS}     # the primary U per role
    per_comp = put_per_component(summary, W)
    tbt = put_dedup_and_two_by_two(summary, W)
    testable = (W.n_drop >= MIN_DROP).to_numpy()
    Wt = W[testable]
    if a.windows == "all":
        t_share = pooled_share(W, testable)
    else:
        t_share = t_share_all            # W holds only testable windows: the share of ALL edge
    obs_t, within_obs = put_sensitivities(summary, W, Wt, t_share)
    print_observed(obs, obs_t, within_obs, t_share, sel)
    xcheck = preview_crosscheck(a, Wf, obs, sel)

    if a.calibrate_only:
        if outs["null.csv"].exists():          # only reachable with --overwrite after a full run
            outs["null.csv"].unlink()
            print(f"\n  deleted the stale {outs['null.csv'].name} of an earlier full run "
                  f"(it does not belong to this run's windows and level test)")
        summary_df = pd.DataFrame(summary.rows)
        summary_df.to_csv(outs["summary.csv"], index=False)
        write_report(outs["report.md"], a, sel, drops, nan_counts, notes, W, Wf, cal, calib_msg,
                     summary_df, obs_u, tbt, xcheck)
        print(f"\n[{a.prefix}] --calibrate-only: wrote {outs['windows.parquet'].name}, "
              f"{outs['calibration.csv'].name}, {outs['summary.csv'].name}, "
              f"{outs['report.md'].name} in {a.out} [{time.time() - t0:.0f}s]")
        return 0

    # bootstrap over components
    t1 = time.time()
    boots, boots_s, n_comps = bootstrap_components(W, Wt, level_cols, var_name, a.boot, a.seed)
    print(f"\n  bootstrap: {a.boot} draws over {n_comps} components [{time.time() - t1:.1f}s]")

    # displaced-edge null: the lines, the coverage geometry and the self-check
    t1 = time.time()
    files = ""
    if a.files:
        files = f", files {', '.join(f'{k}={p}' for k, p in a.files.items())}"
    print(f"\n  null: loading S and M lines with edge_layers' loaders (gis-dir {a.gis_dir}"
          + files + ")", flush=True)
    sets, line_cov, line_info = load_lines(a, v, bb)
    failed = line_cov.failed()
    if failed:
        line_cov.print()
        print(f"\nFAILED: the line layers do not cover the region vertices "
              f"({len(failed)} edge_layers coverage check(s)); no null.", file=sys.stderr)
        return 1
    missing = [name for name in NULL_LAYERS if name not in sets]
    if missing:
        raise SystemExit(f"null: layers {missing} not loaded")
    file_diff = changed_layer_files(meta, line_info)
    for source, (was, now) in file_diff.items():
        print(f"  NOTE: {source} files differ from the ones edge_layers recorded: {was} vs {now}")
    print(f"  loaded {', '.join(f'{n} {len(ls.geom):,} lines' for n, ls in sets.items())}; files "
          f"{line_info['layer_files']} [{time.time() - t1:.0f}s]", flush=True)
    print(f"  edge_layers flags recomputed on region vertices: mismatches with the stored columns "
          f"{line_info['flag_mismatch']}", flush=True)
    t1 = time.time()
    region = v.role.astype(str).to_numpy() == "region"
    dfirms = sorted(set(v.dfirm_id.astype(str)[region]))
    cover = coverage_geometry(a, sorted(v.component[region].unique()), dfirms)
    ctx = NullContext(v, wid, slot, W, sets, bb, cover, level_cols)
    print(f"  coverage geometry loaded [{time.time() - t1:.0f}s]; {len(ctx.iv):,} region vertices "
          f"in {ctx.nW:,} windows of {len(ctx.lobes):,} lobes are displaced per draw", flush=True)
    selfcheck, ok_self, zero_cov = self_check(ctx, obs_u, obs_t)
    if not ok_self:
        print("FAILED: the null's distance code does not reproduce the stored distances; "
              "no null run.", file=sys.stderr)
        return 1

    # the null draws
    null, null_secs = run_null(ctx, a.null_draws, a.seed, a.jobs, obs_u, obs_t, level_cols, var_name)
    if len(null):
        null = add_within_null(null, W, per_comp)
    null.to_csv(outs["null.csv"], index=False)
    mean_thread_secs = null.seconds.mean() if len(null) else 0
    print(f"  null done in {null_secs:.0f}s ({null_secs / max(1, a.null_draws):.2f}s per draw "
          f"wall, {mean_thread_secs:.1f}s per draw per thread)", flush=True)

    add_boot_and_null(summary, boots, boots_s, null, obs_u, sel)
    summary_df = pd.DataFrame(summary.rows)
    summary_df.to_csv(outs["summary.csv"], index=False)
    run = NullRun(ctx=ctx, null=null, seconds=null_secs, selfcheck=selfcheck, ok_self=ok_self,
                  zero_cov=zero_cov, line_info=line_info, cover_files=cover["files"],
                  file_diff=file_diff)
    write_report(outs["report.md"], a, sel, drops, nan_counts, notes, W, Wf, cal, calib_msg,
                 summary_df, obs_u, tbt, xcheck, run)
    print_final(summary_df, sel, run)
    print(f"\n[{a.prefix}] wrote {', '.join(p.name for p in outs.values())} in {a.out} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
