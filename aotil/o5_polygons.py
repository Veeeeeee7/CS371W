#!/usr/bin/env python3
"""
o5_polygons.py -- were the Zone A polygons inside the OODSelect regions drawn differently from the
Zone A polygons around them? (docs/TASK-O5-REGIONS.md, Part E; results and a glossary of the map
terms in docs/O5-REGIONS-REPORT-2026-10-17.md, Part E and section 5.)

A lobe is one single-part flood-zone polygon. aotil/lobes.py builds them per region with the role
region, matched_A (Zone A in the region's 50 km tiles, matched on perimeter) or AE. Each lobe is
measured on its part as stored in the prep repack haz_FL_5070 (EPSG:5070 metres, not densified):
vertex spacing, straight runs, shape index, STUDY_TYP / SOURCE_CIT, the FIRM panels it lies on and
its LOMR cover. Every statistic is compared region - matched_A and region - AE per region, per
region type and pooled, with bootstrap intervals and a matched-growth null in which each S2 null
patch gets lobes by lobes.py's own rules.

Primary (results/2026-10-17/o5/PRIMARIES.md): median vertex spacing, region - matched_A, pooled
over the S = 500 regions with land points, 95% region-bootstrap interval; predicted positive.

Writes --out-polygons (one row per lobe, region and role), --out-summary (one row per region or
pooled scope, subset, metric and stat) and, with --null-out, one row per null patch plus
<stem>_pooled.parquet (the type and pooled null draws of every metric). The seed-1 c2, c5, c6 and
seed-2 c2, c3, c4 lobe tables in results/2026-10-17/o5/E_lobes were built by lobes.py
--only-components with the docs/RUN-FL-V2.md section 12 arguments. About 200 s; SCRATCH is a
directory outside the repo:

    OMP_NUM_THREADS=2 .venv/bin/python aotil/o5_polygons.py \\
        --lobes results/2026-10-16/boundary/FL_band_AE_s0_spatial_mu3_knn10_S500_lobes.parquet \\
                results/2026-10-16/boundary/FL_band_AE_s0_spatial_mu3_knn10_S250_lobes.parquet \\
                results/2026-10-16/boundary/FL_band_AE_s1_spatial_mu3_knn10_S500_lobes.parquet \\
                results/2026-10-16/boundary/FL_band_AE_s2_spatial_mu3_knn10_S500_lobes.parquet \\
                results/2026-10-17/o5/E_lobes/FL_band_AE_s1_spatial_mu3_knn10_S500_lobes.parquet \\
                results/2026-10-17/o5/E_lobes/FL_band_AE_s2_spatial_mu3_knn10_S500_lobes.parquet \\
        --out-polygons results/2026-10-17/o5/E_polygons.csv \\
        --out-summary results/2026-10-17/o5/E_summary.csv \\
        --null-out "$SCRATCH/E_null_draws.parquet" > logs/2026-10-17/E_polygons.log 2>&1
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

# Set before numpy loads, so its BLAS uses 2 threads unless the caller chose otherwise.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import shapely
from shapely import STRtree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lobes as lb          # noqa: E402
import sample_points as sp  # noqa: E402

TURN_DEG = 1.0          # a straight run turns by < 1 degree ...
RUN_MIN_M = 300.0       # ... over >= 300 m
SEG_EPS_M = 1e-6        # a segment shorter than this repeats a vertex
PANEL_MIN_M2 = 1.0      # a lobe lies on a panel when they overlap by more than this
NAT_YEAR = 9999         # NFHL writes 9999-09-09 for a missing date
WATER_CLASS = 11        # NLCD open water
N_BOOT = 1000
BOOT_SEED = 20261017
NULL_RNG = 20261018     # seeds the null patches' perimeter matching
TIE = 1e-12             # null values this close to the observed one count as ties
ROLES = ("region", "matched_A", "AE")
LOBE_RE = re.compile(r"(?P<cell>.+)_s(?P<seed>\d+)_(?P<variant>.+)_S(?P<S>\d+)_lobes\.parquet$")

# lobes.py arguments of every stored run (docs/RUN-FL-V2.md section 12)
NEAR_M = 2000.0
HULL_BUFFER_M = 2000.0
N_MATCHED = 3
LOBES_SEED = 0
MIN_SIZE = 20           # points; lobes.py keeps every such component's region lobes out of matched_A

# (column, stat) pairs of the summary, in row order. stat: 'median' over lobes; 'mean'; 'agg' a
# ratio of sums (see Columns.stat); 'share' the mean of a 0/1 column (share of lobes).
METRICS = [
    ("vertex_spacing_m", "median"), ("vertex_spacing_m", "agg"), ("seg_len_median_m", "median"),
    ("straight_share", "median"), ("straight_share", "agg"), ("straight_share_vertex", "agg"),
    ("shape_index", "median"),
    ("straight_share_vertex", "median"), ("longest_straight_m", "median"), ("area_m2", "median"),
    ("perimeter_m", "median"), ("n_vertices", "median"), ("pan_eff_year", "median"),
    ("pan_pre_present", "share"), ("pan_scale", "median"), ("pan_n", "median"),
    ("lomr_share", "mean"), ("lomr_share", "agg"), ("lomr_any", "share"),
    ("lomr_share_eff", "mean"),
]
# Each level of these becomes a 0/1 column, summarised as a share of lobes and of edge length.
CATEGORICAL = ["study_typ", "study_regime", "source_kind", "pan_scale_cat"]

SUMMARY_COLUMNS = ["primary", "predicted", "met", "scope", "subset", "budget", "seed", "component",
                   "region_type", "metric", "stat", "n_regions", "n_region", "n_matched", "n_AE",
                   "region", "matched_A", "AE", "d_rm", "d_rm_lo", "d_rm_hi", "null_pct", "null_n",
                   "null_median", "null_lo", "null_hi", "d_rAE", "d_rAE_lo", "d_rAE_hi", "ci_method"]
POLYGON_COLUMNS = [
    "budget", "seed", "component", "region_type", "role", "lobe_id", "src_row_id", "part", "zone",
    "fld_ar_id", "dfirm_id", "zone_subty", "is_floodway", "n_points", "from_positive",
    "matched_to", "match_dlogp", "tiles", "water_only", "repaired", "area_m2", "stored_area_m2",
    "perimeter_m", "n_rings", "n_vertices_raw", "n_vertices", "vertex_spacing_m",
    "seg_len_median_m", "straight_m", "straight_share", "straight_vertex_m",
    "straight_share_vertex", "longest_straight_m", "shape_index", "study_typ", "study_regime",
    "source_cit", "source_kind", "pan_n", "pan_n_all", "pan_fallback", "pan_n_1pct", "pan_cover",
    "pan_firm_pan", "pan_dfirm_id", "pan_eff_date", "pan_pre_date", "pan_eff_year",
    "pan_pre_present", "pan_scale", "pan_typ", "pan_dom_share", "pan_eff_min", "pan_eff_max",
    "pan_n_scales", "lomr_n", "lomr_share", "lomr_share_eff", "lomr_any", "lobe_file"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ======================================================================== geometry
def band_windows(headings: np.ndarray, seg_len: np.ndarray, lim: float, run_min: float,
                 n_start: int) -> tuple[float, float]:
    """Length covered by windows of consecutive segments whose headings span < lim and whose
    length is >= run_min, and the longest such window.

    Two pointers: for each start i the window [i, j) grows as far as the heading range allows
    (monotone deques hold its max and min), so every qualifying window lies inside one of these
    maximal windows. Starts are 0..n_start-1. A sequence longer than n_start is a ring laid out
    twice: its windows are at most n_start long and are marked modulo n_start."""
    h = headings.tolist()
    cum_len = np.r_[0.0, np.cumsum(seg_len)].tolist()
    n = len(h)
    m = n_start
    marks = np.zeros(m + 1)            # +1 where a covered stretch starts, -1 where it ends
    max_q, min_q = deque(), deque()
    j, longest = 0, 0.0
    for i in range(m):
        stop = min(i + m, n)
        while j < stop:
            hj = h[j]
            hi = max(h[max_q[0]], hj) if max_q else hj
            lo = min(h[min_q[0]], hj) if min_q else hj
            if hi - lo >= lim:
                break
            while max_q and h[max_q[-1]] <= hj:
                max_q.pop()
            max_q.append(j)
            while min_q and h[min_q[-1]] >= hj:
                min_q.pop()
            min_q.append(j)
            j += 1
        length = cum_len[j] - cum_len[i]
        if length >= run_min:
            longest = max(longest, length)
            if j <= m:
                marks[i] += 1
                marks[j] -= 1
            else:                      # the window wraps past the ring's first segment
                marks[i] += 1
                marks[m] -= 1
                marks[0] += 1
                marks[j - m] -= 1
        if max_q and max_q[0] == i:
            max_q.popleft()
        if min_q and min_q[0] == i:
            min_q.popleft()
    covered = np.cumsum(marks[:m]) > 0
    return float(np.asarray(seg_len[:m])[covered].sum()), longest


def ring_runs(xy: np.ndarray, turn_deg: float = TURN_DEG, run_min: float = RUN_MIN_M) -> dict:
    """Straight runs on one closed ring (xy: (n + 1, 2), first == last).

    Band version (the main one): a segment is straight when it lies in a chain of consecutive
    segments >= run_min long whose headings all lie within a turn_deg band; one segment >= run_min
    counts alone. A union of windows does not depend on where the ring starts or which way it
    runs. Per-vertex version: every vertex inside the run turns by < turn_deg, so on densely
    digitized edges (0.2-2 m spacing) a gentle curve passes too.
    Returns n_seg (non-zero segments = distinct vertices), perim, straight (band),
    straight_vertex, longest (band) and seg (the segment lengths)."""
    d = np.diff(np.asarray(xy, float), axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    keep = seg > SEG_EPS_M
    d, seg = d[keep], seg[keep]
    m = len(seg)
    out = dict(n_seg=m, perim=float(seg.sum()), straight=0.0, straight_vertex=0.0,
               longest=0.0, seg=seg)
    if m == 0:
        return out
    heading = np.arctan2(d[:, 1], d[:, 0])
    turn = np.mod(heading - np.roll(heading, 1) + np.pi, 2 * np.pi) - np.pi   # turn into segment k
    lim = np.deg2rad(turn_deg)
    breaks = np.abs(turn) >= lim
    cyclic = not breaks.any()          # a ring that never turns by turn_deg is one run
    if cyclic:
        order = np.arange(m)
    else:
        order = np.roll(np.arange(m), -int(np.flatnonzero(breaks)[0]))   # start after a turn
    s, t, b = seg[order], turn[order], breaks[order]
    b[0] = True

    # per-vertex runs: segments between two breaks
    run_id = np.cumsum(b) - 1
    run_len = np.bincount(run_id, weights=s)
    out["straight_vertex"] = float(run_len[run_len >= run_min].sum())

    # band version, within each per-vertex run long enough
    first = np.flatnonzero(b)
    last = np.r_[first[1:], m]
    straight, longest = 0.0, 0.0
    for a, z, length in zip(first, last, run_len):
        if length < run_min:
            continue
        h = np.cumsum(np.r_[0.0, t[a + 1:z]])         # heading relative to the run's first segment
        if h.max() - h.min() < lim:
            straight += length
            longest = max(longest, length)
            continue
        if cyclic:                                     # the run is the whole ring: let windows wrap
            h = np.cumsum(np.r_[0.0, t[1:], t[0], t[1:]])
            covered, run_longest = band_windows(h, np.r_[s, s], lim, run_min, m)
        else:
            covered, run_longest = band_windows(h, s[a:z], lim, run_min, z - a)
        straight += covered
        longest = max(longest, run_longest)
    out["straight"], out["longest"] = straight, longest
    return out


def part_metrics(geoms: np.ndarray) -> pd.DataFrame:
    """Edge metrics of single Polygons, summed over their rings (metres, m2):
    n_vertices        non-zero segments (exact repeated vertices do not count); n_vertices_raw
                      counts the coordinates as stored
    vertex_spacing_m  perimeter / n_vertices, the mean segment length: coarse spacing suggests hand
                      digitizing from a small-scale paper map. seg_len_median_m is robust to a few
                      long chords
    straight_share    straight_m / perimeter (ring_runs, band version); straight_share_vertex the
                      per-vertex version; longest_straight_m the longest band run
    shape_index       2 ln(P/4) / ln(A): 1 for a square, larger for convoluted edges; NaN if A <= 1
    """
    geoms = np.asarray(geoms, dtype=object)
    rings, geom_of_ring = shapely.get_rings(geoms, return_index=True)
    xy, ring_of_coord = shapely.get_coordinates(rings, return_index=True)
    cut = np.flatnonzero(np.diff(ring_of_coord)) + 1
    starts = np.r_[0, cut]
    stops = np.r_[cut, len(xy)]
    n = len(geoms)
    n_seg = np.zeros(n, np.int64)
    perim = np.zeros(n)
    straight = np.zeros(n)
    straight_vertex = np.zeros(n)
    longest = np.zeros(n)
    segs = [[] for _ in range(n)]
    for a, z in zip(starts, stops):
        if a == z:                     # no coordinates at all
            continue
        g = geom_of_ring[ring_of_coord[a]]
        runs = ring_runs(xy[a:z])
        n_seg[g] += runs["n_seg"]
        perim[g] += runs["perim"]
        straight[g] += runs["straight"]
        straight_vertex[g] += runs["straight_vertex"]
        longest[g] = max(longest[g], runs["longest"])
        segs[g].append(runs["seg"])
    area = shapely.area(geoms)
    n_coords = shapely.get_num_coordinates(geoms)
    n_rings = shapely.get_num_interior_rings(geoms) + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        spacing = np.where(n_seg > 0, perim / np.maximum(n_seg, 1), np.nan)
        shape = np.where(area > 1, 2 * np.log(perim / 4) / np.log(np.maximum(area, 1 + 1e-12)),
                         np.nan)
        share = np.where(perim > 0, straight / np.maximum(perim, 1e-300), np.nan)
        share_vertex = np.where(perim > 0, straight_vertex / np.maximum(perim, 1e-300), np.nan)
    seg_median = np.full(n, np.nan)
    for g in range(n):
        if segs[g] and sum(len(x) for x in segs[g]):
            seg_median[g] = np.median(np.concatenate(segs[g]))
    return pd.DataFrame(dict(area_m2=area, perimeter_m=perim, n_rings=n_rings,
                             n_vertices_raw=n_coords - n_rings, n_vertices=n_seg,
                             vertex_spacing_m=spacing, seg_len_median_m=seg_median,
                             straight_m=straight, straight_share=share,
                             straight_vertex_m=straight_vertex, straight_share_vertex=share_vertex,
                             longest_straight_m=longest, shape_index=shape))


def explode_raw(geoms: dict) -> pd.DataFrame:
    """lobes.explode's parts with the same (row_id, part) numbering, plus the part each piece
    comes from: `piece` is what lobes.py stores (oriented, repaired if invalid), `raw` the
    haz_FL_5070 part before that repair. The edge metrics are measured on `raw`."""
    rows = []
    for row_id, g in geoms.items():
        k = 0
        for p in lb.polygon_parts(g):
            pieces, repaired = [p], False
            if not p.is_valid:
                pieces, repaired = lb.polygon_parts(shapely.make_valid(p)), True
            for q in pieces:
                rows.append((row_id, k, q, p, repaired))
                k += 1
    parts = pd.DataFrame(rows, columns=["row_id", "part", "piece", "raw", "repaired"])
    parts["piece"] = shapely.orient_polygons(np.array(parts.piece.to_numpy(), dtype=object),
                                             exterior_cw=False)
    return parts


# ======================================================================== the lobe rules
def match_perimeter_fast(reg_logp: np.ndarray, cand_logp: np.ndarray, n_rounds: int,
                         rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """lobes.match_perimeter with the same rng draws and picks, without a full sort per pick:
    the nearest free candidate, ties to the smallest tie key, then to the lowest index (what
    lexsort((tie, d))[0] returns, np.lexsort being stable)."""
    free = np.ones(len(cand_logp), bool)
    n_free = len(cand_logp)
    tie = rng.random(len(cand_logp))
    order = rng.permutation(len(reg_logp))
    out = []
    for _ in range(n_rounds):
        for i in order:
            if n_free == 0:
                return out
            d = np.where(free, np.abs(cand_logp - reg_logp[i]), np.inf)
            nearest = np.flatnonzero(d == d.min())
            if len(nearest) > 1:
                j = int(nearest[np.argmin(tie[nearest])])
            else:
                j = int(nearest[0])
            free[j] = False
            n_free -= 1
            out.append((int(i), j, float(d[j])))
    return out


class LobeRule:
    """lobes.py's region and matched_A rules (lobes.region_parts, the matched_A eligibility in
    lobes.build, lobes.match_perimeter) on one seed's parts, for any set of points.

    parts: explode_raw output sorted by (row_id, part), as lobes.py's parts are. zone: FLD_ZONE
    per part. polys_A: row_id -> geometry of the Zone A polygons near the tiles."""

    def __init__(self, parts: pd.DataFrame, zone: np.ndarray, polys_A: dict, tiles: list[str]):
        self.part_row_id = parts.row_id.to_numpy()
        self.part_geoms = np.array(parts.piece.to_numpy(), dtype=object)
        shapely.prepare(self.part_geoms)
        self.tree = STRtree(self.part_geoms)
        self.logp = np.log(shapely.length(self.part_geoms))
        self.isA = zone == lb.REGION_ZONE
        self.poly_ids = np.array(sorted(polys_A), dtype=np.int64)
        self.poly_tree = STRtree(np.array([polys_A[r] for r in self.poly_ids], dtype=object))
        self.tile_parts = {}
        for t in tiles:
            box = shapely.box(*lb.tile_box(t))
            self.tile_parts[t] = np.unique(self.tree.query(box, predicate="intersects"))

    def covers(self, xy: np.ndarray) -> np.ndarray:
        """Mask over the parts: those that cover (contain or touch) one of the points."""
        mask = np.zeros(len(self.part_geoms), bool)
        if len(xy):
            mask[self.tree.query(shapely.points(xy), predicate="covered_by")[1]] = True
        return mask

    def region_parts(self, xy: np.ndarray, extra_polys=()) -> np.ndarray:
        """Parts of the Zone A polygons within NEAR_M of a point (plus extra_polys, the positives'
        source polygons) that meet the points' convex hull buffered by HULL_BUFFER_M."""
        hit = self.poly_tree.query(shapely.points(xy), predicate="dwithin", distance=NEAR_M)[1]
        polys = set(self.poly_ids[np.unique(hit)].tolist()) | set(extra_polys)
        hull = shapely.buffer(shapely.convex_hull(shapely.multipoints(xy)), HULL_BUFFER_M)
        on_hull = np.unique(self.tree.query(hull, predicate="intersects"))
        return on_hull[np.isin(self.part_row_id[on_hull], list(polys))]

    def eligible(self, tiles, exclude: np.ndarray) -> np.ndarray:
        """Zone A parts in the tiles that are not excluded: the matched_A candidates."""
        in_tiles = np.unique(np.concatenate([self.tile_parts[t] for t in tiles]))
        return in_tiles[self.isA[in_tiles] & ~exclude[in_tiles]]

    def match(self, reg: np.ndarray, elig: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        """N_MATCHED candidates per region part. Returns (matched parts, the region part each
        was matched to)."""
        matches = match_perimeter_fast(self.logp[reg], self.logp[elig], N_MATCHED, rng)
        matched = np.array([elig[j] for _, j, _ in matches], dtype=np.int64)
        matched_to = np.array([reg[i] for i, _, _ in matches], dtype=np.int64)
        return matched, matched_to


# ======================================================================== statistics
def mid_rank_pct(obs: float, null: np.ndarray) -> float:
    """Percentile of obs among the finite null values, ties (within TIE) counting half."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan
    below = np.sum(null < obs - TIE)
    tied = np.sum(np.abs(null - obs) <= TIE)
    return float(100.0 * (below + 0.5 * tied) / len(null))


class Columns:
    """A frame's columns as float arrays (self.a), for statistics over many sets of rows."""

    def __init__(self, df: pd.DataFrame, cols):
        self.a = {c: df[c].to_numpy(float) for c in cols}

    def stat(self, ix: np.ndarray, col: str, stat: str) -> float:
        """Statistic of column col over rows ix (a repeated row counts again); NaN for no rows.
        'median'; 'agg' a ratio of sums; 'pshare' the perimeter-weighted mean of a 0/1 column
        (share of edge length); anything else the mean. NaN values are left out."""
        if not len(ix):
            return np.nan
        a = self.a
        if stat == "agg":
            if col == "vertex_spacing_m":           # total perimeter / total vertices
                return float(a["perimeter_m"][ix].sum() / max(a["n_vertices"][ix].sum(), 1))
            if col == "straight_share":             # straight length / perimeter
                return float(a["straight_m"][ix].sum() / a["perimeter_m"][ix].sum())
            if col == "straight_share_vertex":
                return float(a["straight_vertex_m"][ix].sum() / a["perimeter_m"][ix].sum())
            if col == "lomr_share":                 # LOMR-covered area / area
                covered = (a["lomr_share"][ix] * a["area_m2"][ix]).sum()
                return float(covered / a["area_m2"][ix].sum())
            raise ValueError(col)
        if stat == "pshare":
            v, w = a[col][ix], a["perimeter_m"][ix]
            ok = np.isfinite(v)
            if not ok.any():
                return np.nan
            return float((v[ok] * w[ok]).sum() / w[ok].sum())
        v = a[col][ix]
        v = v[np.isfinite(v)]
        if not len(v):
            return np.nan
        if stat == "median":
            return float(np.median(v))
        return float(v.mean())


def ci(v: np.ndarray) -> tuple[float, float]:
    """2.5 / 97.5 percentiles of the finite values; NaN for fewer than 10."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ======================================================================== inputs
def parse_lobe_name(path: Path) -> dict:
    """cell, seed, variant and budget from a <cell>_s<seed>_<variant>_S<S>_lobes.parquet name."""
    m = LOBE_RE.match(path.name)
    if not m:
        raise SystemExit(f"{path.name}: expected <cell>_s<seed>_<variant>_S<S>_lobes.parquet")
    return dict(cell=m["cell"], seed=int(m["seed"]), variant=m["variant"], budget=int(m["S"]))


def load_lobes(files: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    """Every --lobes table in one frame (with cell, seed, variant, budget, lobe_file), and one run
    record per file: its name fields, file and components."""
    tables, runs = [], []
    for f in files:
        meta = parse_lobe_name(f)
        d = gpd.read_parquet(f)
        if str(d.crs.to_epsg()) != "5070":
            raise SystemExit(f"{f}: CRS {d.crs}, not EPSG:5070")
        components = sorted(d.component.unique().tolist())
        for k, v in meta.items():
            d[k] = v
        d["lobe_file"] = str(f)
        tables.append(d)
        runs.append(dict(meta, file=str(f), components=components))
    lobes = pd.concat(tables, ignore_index=True)
    dup = lobes.duplicated(["budget", "seed", "component", "role", "lobe_id"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} lobe rows appear in two --lobes files "
                         f"(same budget, seed, component, role, lobe_id)")
    return lobes, runs


def read_points(regions_dir: Path, cell: str, seed: int, variant: str, budget: int) -> pd.DataFrame:
    """The region points table (aotil/regions.py) of one selection."""
    return pd.read_parquet(regions_dir / f"{cell}_s{seed}_{variant}_S{budget}_points.parquet")


def read_selection(regions_dir: Path, cell: str, seed: int, variant: str,
                   budget: int) -> tuple[pd.DataFrame, list[int]]:
    """The region points table of one selection, read and checked as lobes.py reads it
    (lobes.read_regions), and its components of >= MIN_SIZE points."""
    stem = f"{cell}_s{seed}_{variant}_S{budget}"
    return lb.read_regions(regions_dir / f"{stem}_points.parquet",
                           regions_dir / f"{stem}_components.csv", MIN_SIZE)


def dataset_xy(dataset: Path) -> pd.DataFrame:
    """x, y (EPSG:5070), 50 km tile and NLCD landcover of every point, indexed by point_id."""
    table = pq.read_table(dataset, columns=["point_id", "geometry", "tile_50km", "landcover"])
    d = table.to_pandas().set_index("point_id")
    xy = shapely.get_coordinates(shapely.from_wkb(d.geometry.to_numpy()))
    return pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], tile=d.tile_50km.astype(str).to_numpy(),
                             landcover=d.landcover.to_numpy()), index=d.index)


# ======================================================================== attributes
def read_attrs(path: Path, state: str) -> tuple[pd.DataFrame, str]:
    """nfhl_haz_attrs_10state rows of the state, one per (DFIRM_ID, FLD_AR_ID). The composite key
    is a safeguard: in Florida FLD_AR_ID alone does not repeat across DFIRMs."""
    a = pd.read_parquet(path, columns=["DFIRM_ID", "FLD_AR_ID", "STUDY_TYP", "SOURCE_CIT",
                                       "STATE"])
    a = a[a.STATE == state]
    dups = a.duplicated(["DFIRM_ID", "FLD_AR_ID"], keep=False)
    conflict = a[dups].groupby(["DFIRM_ID", "FLD_AR_ID"])[["STUDY_TYP", "SOURCE_CIT"]].nunique()
    n_conflict = int((conflict > 1).any(axis=1).sum())
    note = (f"{int(dups.sum())} attrs rows of {state} share a (DFIRM_ID, FLD_AR_ID) key, "
            f"{n_conflict} of those keys with conflicting STUDY_TYP/SOURCE_CIT (first row kept)")
    return a.drop_duplicates(["DFIRM_ID", "FLD_AR_ID"]).drop(columns=["STATE"]), note


def attach_attrs(keys: pd.DataFrame, attrs: pd.DataFrame,
                 idx: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """keys: src_row_id, dfirm_id, fld_ar_id. Adds study_typ and source_cit (joined on DFIRM_ID,
    FLD_AR_ID as the brief asks), source_kind and study_regime (from idx_FL); stops unless the
    join agrees with idx_FL by row id."""
    renamed = attrs.rename(columns=dict(DFIRM_ID="dfirm_id", FLD_AR_ID="fld_ar_id",
                                        STUDY_TYP="study_typ", SOURCE_CIT="source_cit"))
    m = keys.merge(renamed, on=["dfirm_id", "fld_ar_id"], how="left")
    by_row = idx.set_index("row_id").reindex(m.src_row_id)
    same_typ = m.study_typ.fillna("").to_numpy() == by_row.STUDY_TYP.fillna("").to_numpy()
    same_cit = m.source_cit.fillna("").to_numpy() == by_row.SOURCE_CIT.fillna("").to_numpy()
    agree = same_typ & same_cit
    m["study_regime"] = by_row.study_regime.to_numpy()
    m["source_kind"] = source_kind(m.source_cit)
    note = (f"STUDY_TYP/SOURCE_CIT joined on (DFIRM_ID, FLD_AR_ID): {int(m.study_typ.notna().sum())}"
            f" of {len(m)} parts matched; {int(agree.sum())} agree with idx_FL by row id")
    if not agree.all():
        raise SystemExit("attrs join disagrees with idx_FL by row id: " + note)
    return m, note


def source_kind(cit: pd.Series) -> pd.Series:
    """The kind of document a SOURCE_CIT names, without its DFIRM prefix and number
    (12021C_STUDY6 -> STUDY). In Florida: STUDY (a flood study), BASE (traced on the base map),
    FIRM (carried over from the previous, usually paper, rate map), FIS (the Flood Insurance
    Study report), LOMC (a letter of map change)."""
    text = cit.astype("string")
    without_prefix = text.str.replace(r"^\d+[A-Z]?_", "", regex=True)
    kind = without_prefix.str.replace(r"\d+$", "", regex=True)
    return kind.mask(cit.isna())


def nfhl_dates(values: pd.Series) -> pd.Series:
    """UTC timestamps, with NFHL's 9999-09-09 'no date' as NaT."""
    t = pd.to_datetime(values, utc=True)
    return t.where(t.dt.year < NAT_YEAR)


def year_of(ts: pd.Series) -> pd.Series:
    """Decimal year of an NFHL date (NaN for none)."""
    t = nfhl_dates(ts)
    return t.dt.year + (t.dt.dayofyear - 1) / 365.25


def read_layer(path: Path) -> tuple[pd.DataFrame, np.ndarray, STRtree]:
    """A panel or LOMR layer in EPSG:5070, its geometries repaired by make_valid, and their tree."""
    d = gpd.read_parquet(path).to_crs(5070).reset_index(drop=True)
    g = shapely.make_valid(d.geometry.to_numpy())
    return d, g, STRtree(g)


def attach_panels(geoms: np.ndarray, layer, own_dfirm) -> tuple[pd.DataFrame, str]:
    """The FIRM panels (map sheets) each part lies on, overlapping it by > PANEL_MIN_M2.

    Only the panels of the part's own DFIRM (own_dfirm: its DFIRM_ID) count: a DFIRM is one
    county's flood map and neighbouring counties' panels overlap, so a part can lie inside two
    panels of two DFIRMs with different dates and scales. A part on no panel of its own DFIRM
    falls back to all panels (pan_fallback = 1). Over the panels used: pan_n, pan_n_1pct (>= 1% of
    the part's area), pan_cover (summed overlap / area), pan_eff_min / max, pan_n_scales, and the
    area-dominant panel's FIRM_PAN, DFIRM_ID, dates, scale, type and area share (ties: own DFIRM,
    then the lower panel row). pan_n_all counts the panels of any DFIRM. pan_pre_present: the
    dominant panel replaced an earlier one (it has a PRE_DATE)."""
    t0 = time.time()
    pan, pan_geoms, tree = layer
    n = len(geoms)
    area = shapely.area(geoms)
    part_ix, pan_ix = tree.query(geoms, predicate="intersects")
    overlap = np.zeros(0)
    if len(part_ix):
        overlap = shapely.area(shapely.intersection(geoms[part_ix], pan_geoms[pan_ix]))
    keep = overlap > PANEL_MIN_M2
    part_ix, pan_ix, overlap = part_ix[keep], pan_ix[keep], overlap[keep]

    own_dfirm = np.asarray(own_dfirm, dtype=object)
    own = pan.DFIRM_ID.to_numpy(dtype=object)[pan_ix] == own_dfirm[part_ix]
    n_all = np.bincount(part_ix, minlength=n)
    has_own = np.bincount(part_ix[own], minlength=n) > 0
    use = own | ~has_own[part_ix]
    part_ix, pan_ix, overlap, own = part_ix[use], pan_ix[use], overlap[use], own[use]

    eff = nfhl_dates(pan.EFF_DATE)
    pre = nfhl_dates(pan.PRE_DATE)
    pairs = pd.DataFrame(dict(part=part_ix, panel=pan_ix, overlap=overlap, own=own))
    out = pd.DataFrame(index=range(n))
    out["pan_n"] = np.bincount(part_ix, minlength=n)
    out["pan_n_all"] = n_all
    out["pan_fallback"] = ((n_all > 0) & ~has_own).astype(int)
    out["pan_n_1pct"] = np.bincount(part_ix[overlap >= 0.01 * area[part_ix]], minlength=n)
    out["pan_cover"] = np.bincount(part_ix, weights=overlap, minlength=n) / np.where(area > 0, area, 1)

    by_rank = pairs.sort_values(["part", "own", "overlap", "panel"],
                                ascending=[True, False, False, True])
    dom = by_rank.drop_duplicates("part")
    dom_part = dom.part.to_numpy()
    dom_panel = dom.panel.to_numpy()
    for col, src in (("pan_firm_pan", pan.FIRM_PAN), ("pan_dfirm_id", pan.DFIRM_ID),
                     ("pan_eff_date", eff), ("pan_pre_date", pre), ("pan_scale", pan.SCALE),
                     ("pan_typ", pan.PANEL_TYP)):
        values = pd.Series([pd.NA] * n, dtype=object)
        values.iloc[dom_part] = src.iloc[dom_panel].to_numpy()
        out[col] = values
    out["pan_dom_share"] = np.nan
    out.loc[dom_part, "pan_dom_share"] = dom.overlap.to_numpy() / area[dom_part]

    pairs["eff"] = eff.iloc[pairs.panel.to_numpy()].to_numpy()
    pairs["scale"] = pan.SCALE.iloc[pairs.panel.to_numpy()].to_numpy()
    per_part = pairs.groupby("part")
    out["pan_eff_min"] = pd.Series(per_part.eff.min()).reindex(range(n)).to_numpy()
    out["pan_eff_max"] = pd.Series(per_part.eff.max()).reindex(range(n)).to_numpy()
    n_scales = pd.Series(per_part.scale.nunique()).reindex(range(n))
    out["pan_n_scales"] = n_scales.fillna(0).astype(int)
    out["pan_eff_date"] = pd.to_datetime(out.pan_eff_date, utc=True)
    out["pan_pre_date"] = pd.to_datetime(out.pan_pre_date, utc=True)
    out["pan_eff_year"] = year_of(out.pan_eff_date)
    out["pan_pre_present"] = np.where(out.pan_n > 0, out.pan_pre_date.notna().astype(float), np.nan)
    out["pan_scale"] = pd.to_numeric(out.pan_scale, errors="coerce")

    also_other = (n_all > out.pan_n.to_numpy()) & has_own
    note = (f"{len(pan):,} panels; {int((out.pan_n > 0).sum()):,} of {n:,} parts lie on one "
            f"(> {PANEL_MIN_M2:g} m2), {int((out.pan_n > 1).sum()):,} on several of the panels "
            f"used, {int((n_all > 1).sum()):,} on several of any DFIRM; "
            f"{int(also_other.sum()):,} also overlap another "
            f"DFIRM's panels (ignored); {int(out.pan_fallback.sum()):,} overlap no panel of their "
            f"own DFIRM (fallback to all panels); median panel cover "
            f"{np.nanmedian(out.pan_cover):.3f} ({time.time() - t0:.0f}s)")
    return out, note


def attach_lomr(geoms: np.ndarray, layer) -> tuple[pd.DataFrame, str]:
    """Per part: lomr_n (LOMR polygons touching it), lomr_share (share of its area inside the
    union of all LOMRs, any STATUS), lomr_share_eff (Effective ones only), lomr_any."""
    t0 = time.time()
    lomr, lomr_geoms, tree = layer
    effective = (lomr.STATUS == "Effective").to_numpy()
    part_ix, lomr_ix = tree.query(geoms, predicate="intersects")
    area = shapely.area(geoms)
    n = len(geoms)
    share = np.zeros(n)
    share_eff = np.zeros(n)
    n_touching = np.bincount(part_ix, minlength=n)
    for i in np.unique(part_ix):
        touching = lomr_ix[part_ix == i]
        covered = shapely.intersection(geoms[i], shapely.union_all(lomr_geoms[touching]))
        share[i] = shapely.area(covered) / area[i]
        touching_eff = touching[effective[touching]]
        if len(touching_eff):
            covered = shapely.intersection(geoms[i], shapely.union_all(lomr_geoms[touching_eff]))
            share_eff[i] = shapely.area(covered) / area[i]
    out = pd.DataFrame(dict(lomr_n=n_touching, lomr_share=np.clip(share, 0, 1),
                            lomr_share_eff=np.clip(share_eff, 0, 1)))
    out["lomr_any"] = (out.lomr_share > 0).astype(float)
    note = (f"{len(lomr):,} LOMR polygons ({int(effective.sum())} Effective); "
            f"{int((n_touching > 0).sum()):,} of {n:,} parts touch one, "
            f"{int((share > 0).sum()):,} overlap one in area ({time.time() - t0:.0f}s)")
    return out, note


# ======================================================================== per seed
def seed_null_rows(null: pd.DataFrame, seed: int, seed_lobes: pd.DataFrame,
                   args) -> pd.DataFrame:
    """This seed's null patches of the regions that have lobes (the first --null-draws draws)."""
    rows = null[null.seed == seed]
    with_lobes = set(seed_lobes[seed_lobes.budget == args.budget].component.unique().tolist())
    rows = rows[rows.component.isin(with_lobes)]
    if args.null_draws:
        rows = rows[rows.draw < args.null_draws]
    return rows


def rule_tiles(sel: pd.DataFrame | None, null_rows: pd.DataFrame | None,
               pts_xy: pd.DataFrame) -> list[str]:
    """The 50 km tiles the lobe rules need: the region points' and the null patches'."""
    tiles = set()
    if sel is not None:
        tiles |= set(sel.tile_50km.astype(str))
    if null_rows is not None:
        for ids in null_rows.patch_point_ids:
            tiles |= set(pts_xy.tile.reindex(list(ids)).astype(str))
    return sorted(tiles)


def zone_a_near_tiles(idx: pd.DataFrame, tiles: list[str]) -> np.ndarray:
    """Mask over idx: Zone A polygons whose bounding box comes within NEAR_M of a tile."""
    near = np.zeros(len(idx), bool)
    for t in tiles:
        x0, y0, x1, y1 = lb.tile_box(t)
        near |= lb.bbox_hits(idx, x0 - NEAR_M, y0 - NEAR_M, x1 + NEAR_M, y1 + NEAR_M)
    return near & (idx.FLD_ZONE.to_numpy() == lb.REGION_ZONE)


def part_table(parts: pd.DataFrame, idx: pd.DataFrame, layers: dict, t0: float) -> pd.DataFrame:
    """Per part (every part, as the null patches can reach any): edge metrics of the original
    part, STUDY_TYP / SOURCE_CIT, panels and LOMR cover of the stored piece."""
    met = part_metrics(np.array(parts.raw.to_numpy(), dtype=object))
    met.insert(0, "row_id", parts.row_id.to_numpy())
    met.insert(1, "part", parts.part.to_numpy())
    met["repaired"] = parts.repaired.to_numpy()
    log(f"  edge metrics for all parts ({time.time() - t0:.0f}s)")

    by_row = idx.set_index("row_id")
    keys = pd.DataFrame(dict(src_row_id=parts.row_id.to_numpy(),
                             dfirm_id=by_row.DFIRM_ID.reindex(parts.row_id).to_numpy(),
                             fld_ar_id=by_row.FLD_AR_ID.reindex(parts.row_id).to_numpy()))
    attrs, note = attach_attrs(keys, layers["attrs"], idx)
    log("  " + note)
    pieces = np.array(parts.piece.to_numpy(), dtype=object)
    panels, note = attach_panels(pieces, layers["panels"], attrs.dfirm_id.to_numpy())
    log("  " + note)
    lomr, note = attach_lomr(pieces, layers["lomr"])
    log("  " + note)
    attr_cols = attrs[["study_typ", "source_cit", "source_kind", "study_regime"]]
    met = pd.concat([met, attr_cols, panels, lomr], axis=1)
    met["pan_scale_cat"] = met.pan_scale.astype("Int64").astype("string")
    return met


def check_stored_lobes(seed: int, seed_lobes: pd.DataFrame, parts: pd.DataFrame,
                       met: pd.DataFrame) -> np.ndarray:
    """Row in parts of every stored lobe's (src_row_id, part). Stops unless each stored lobe is
    its re-read part (geometry, coordinate count, area), except parts lobes.py repaired: their
    metrics are those of the part before the repair."""
    key = pd.MultiIndex.from_arrays([parts.row_id, parts.part])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([seed_lobes.src_row_id, seed_lobes.part]))
    if (pos < 0).any():
        raise SystemExit(f"seed {seed}: {int((pos < 0).sum())} lobes have no (src_row_id, part) "
                         f"in the re-exploded haz polygons")
    stored = np.array(seed_lobes.geometry.to_numpy(), dtype=object)
    raw = np.array(parts.raw.to_numpy(), dtype=object)[pos]
    piece = np.array(parts.piece.to_numpy(), dtype=object)[pos]
    same_n = shapely.get_num_coordinates(stored) == shapely.get_num_coordinates(raw)
    same_a = np.isclose(shapely.area(stored), met.area_m2.to_numpy()[pos], rtol=1e-9, atol=1e-3)
    same_g = shapely.equals_exact(stored, piece, tolerance=0)
    repaired = met.repaired.to_numpy()[pos]
    bad = ~(same_n & same_a & same_g) & ~repaired
    log(f"  stored lobes vs re-read parts: {len(seed_lobes):,} rows, {int(same_g.sum()):,} "
        f"identical geometry, {int(same_n.sum()):,} same coordinate count, {int(repaired.sum())} "
        f"from a repaired part; {int(bad.sum())} unrepaired lobes differ")
    if bad.any():
        raise SystemExit(f"seed {seed}: {int(bad.sum())} stored lobes differ from their "
                         f"haz_FL_5070 part")
    return pos


def source_row_ids(points: pd.DataFrame, mask) -> list[int]:
    """Source polygon row ids of the points in mask (lobes.py adds the positives' sources)."""
    return points.loc[mask, "meta_src_row_id"].dropna().astype(int).tolist()


def region_mask(rule: LobeRule, sel: pd.DataFrame, pts_xy: pd.DataFrame,
                components: list[int]) -> np.ndarray:
    """Mask over the parts: the region lobes of these components of sel, their positives'
    source polygons included as lobes.py includes them."""
    mask = np.zeros(len(rule.part_geoms), bool)
    for c in components:
        points = sel[sel.component == c]
        xy = pts_xy.loc[points.point_id, ["x", "y"]].to_numpy()
        mask[rule.region_parts(xy, source_row_ids(points, points.label == 1))] = True
    return mask


def rebuild_observed(seed: int, rule: LobeRule, parts: pd.DataFrame, sel: pd.DataFrame,
                     big: list[int], budget_lobes: pd.DataFrame, budget_runs: list[dict],
                     pos_of_lobe: pd.Series, pts_xy: pd.DataFrame,
                     covers_selected: np.ndarray) -> dict:
    """Rebuild every stored lobes.py run with LobeRule (the region points in place of a patch,
    the same rng) and stop unless its region and matched_A lobes come back exactly, in order, so
    the null patches get lobes by the code path that made the observed ones. big: the components
    of sel with >= MIN_SIZE points.
    Returns component -> parts of the region lobes that only a water point brings in."""
    water_only = {}
    for run in budget_runs:
        run_lobes = budget_lobes[budget_lobes.lobe_file == run["file"]]
        # lobes.build keeps out of matched_A the region lobes of the components it builds and of
        # every component of >= --min-size points, so an --only-components run keeps out the
        # same lobes as a run that builds them all.
        kept_out = sorted(set(run["components"]) | set(big))
        excluded = covers_selected | region_mask(rule, sel, pts_xy, kept_out)
        for c in run["components"]:
            points = sel[sel.component == c]
            xy = pts_xy.loc[points.point_id, ["x", "y"]].to_numpy()
            reg = rule.region_parts(xy, source_row_ids(points, points.label == 1))
            if not np.array_equal(reg, rule.region_parts(xy)):
                raise SystemExit(f"seed {seed} c{c}: the positives' source polygons add lobes the "
                                 f"near rule misses; the null (near rule only) would differ")
            comp_lobes = run_lobes[run_lobes.component == c]
            want_region = pos_of_lobe[comp_lobes.index[comp_lobes.role == "region"]].to_numpy()
            tiles = sorted(points.tile_50km.astype(str).unique())
            elig = rule.eligible(tiles, excluded)
            matched, matched_to = rule.match(reg, elig, np.random.default_rng([LOBES_SEED, c]))
            stored_matched = comp_lobes[comp_lobes.role == "matched_A"]
            want_matched = pos_of_lobe[stored_matched.index].to_numpy()
            # lobes.py's lobe_id of a region lobe: c<component>_region_<row_id>_<part>
            got_to = np.array([f"c{c}_region_{parts.row_id.iat[i]}_{parts.part.iat[i]}"
                               for i in matched_to], dtype=object)
            ok = (np.array_equal(np.sort(reg), np.sort(want_region))
                  and np.array_equal(matched, want_matched)
                  and np.array_equal(got_to, stored_matched.matched_to.to_numpy()))
            log(f"  rebuild s{seed} c{c} ({Path(run['file']).parent.name}): region {len(reg)} / "
                f"{len(want_region)}, matched {len(matched)} / {len(want_matched)} -> "
                f"{'identical' if ok else 'DIFFERENT'}")
            if not ok:
                raise SystemExit(f"seed {seed} component {c}: the lobe rules do not reproduce "
                                 f"{run['file']}; the null would not be comparable")

            # The land rule: lobes are built from all points; count those only water brings in.
            land = points.point_id.map(pts_xy.landcover).to_numpy() != WATER_CLASS
            if (~land).any():
                land_xy = xy[land]
                land_reg = np.empty(0, int)
                if len(land_xy):
                    land_sources = source_row_ids(points, (points.label == 1) & land)
                    land_reg = rule.region_parts(land_xy, land_sources)
                only_water = np.setdiff1d(reg, land_reg)
                water_only[c] = set(only_water.tolist())
                log(f"    land rule: {int((~land).sum())} water point(s); {len(only_water)} region "
                    f"lobe(s) only a water point brings in")
    return water_only


def log_cross_run_matches(seed: int, budget_lobes: pd.DataFrame, pos_of_lobe: pd.Series) -> None:
    """Note matched_A lobes that are another component's region lobe. lobes.py keeps out the
    region lobes of every component of >= --min-size points, so this happens only when another
    run built a component below this run's --min-size."""
    for c in sorted(budget_lobes.component.unique()):
        is_comp = budget_lobes.component == c
        matched = pos_of_lobe[budget_lobes.index[is_comp & (budget_lobes.role == "matched_A")]]
        other = pos_of_lobe[budget_lobes.index[~is_comp & (budget_lobes.role == "region")]]
        k = int(np.isin(matched.to_numpy(), other.to_numpy()).sum())
        if k:
            log(f"  note: s{seed} c{c}: {k} of {len(matched)} matched_A lobes are region lobes of "
                f"another component of this seed (built in a different lobes.py run)")


def null_patch_lobes(seed: int, rule: LobeRule, parts: pd.DataFrame, null_rows: pd.DataFrame,
                     pts_xy: pd.DataFrame, excluded: np.ndarray) -> list[dict]:
    """Region and matched_A lobes of every null patch by lobes.py's rules. The matched_A pool also
    excludes `excluded` (parts covering a selected point, the region lobes of every component of
    >= MIN_SIZE points or with stored lobes), the parts covering a patch point and the patch's own
    region lobes."""
    rows = []
    for (c, d), r in null_rows.set_index(["component", "draw"]).sort_index().iterrows():
        ids = list(r.patch_point_ids)
        xy = pts_xy.loc[ids, ["x", "y"]].to_numpy()
        reg = rule.region_parts(xy)
        is_region = np.zeros(len(parts), bool)
        is_region[reg] = True
        tiles = sorted(set(pts_xy.tile.reindex(ids).astype(str)))
        elig = rule.eligible(tiles, excluded | rule.covers(xy) | is_region)
        rng = np.random.default_rng([NULL_RNG, seed, int(c), int(d)])
        matched, _ = rule.match(reg, elig, rng)
        rows.append(dict(seed=seed, component=int(c), draw=int(d), region_ix=reg,
                         matched_ix=matched, n_region=len(reg), n_matched=len(matched),
                         n_eligible=len(elig), tiles=";".join(tiles)))
    return rows


def seed_pass(seed: int, lobes: pd.DataFrame, runs: list[dict], idx: pd.DataFrame,
              pts_xy: pd.DataFrame, layers: dict, null: pd.DataFrame | None, args) -> dict:
    """One seed: read its polygons once, measure every part, check the stored lobes against them,
    rebuild the observed S = --budget runs and give every null patch lobes. Returns metrics (one
    row per part), pos_of_lobe (lobe row -> part row), water_only and null (one dict per patch)."""
    t0 = time.time()
    seed_lobes = lobes[lobes.seed == seed]
    budget_runs = [r for r in runs if r["seed"] == seed and r["budget"] == args.budget]
    sel = None
    big = []                # its components of >= MIN_SIZE points
    if budget_runs:
        sel, big = read_selection(args.regions_dir, budget_runs[0]["cell"], seed,
                                  budget_runs[0]["variant"], args.budget)
    null_rows = None
    if null is not None:
        null_rows = seed_null_rows(null, seed, seed_lobes, args)

    tiles = rule_tiles(sel, null_rows, pts_xy)
    near_a = zone_a_near_tiles(idx, tiles)
    need = np.union1d(idx.row_id.to_numpy()[near_a], seed_lobes.src_row_id.unique())
    log(f"\n=== seed {seed}: {len(seed_lobes):,} lobe rows; {len(tiles)} tiles for the rules "
        f"({', '.join(tiles)}); reading {len(need):,} polygons "
        f"({int(near_a.sum()):,} Zone A candidates)")
    polys = sp.read_geoms(args.state, need)
    n_missing = sum(g is None for g in polys)
    if n_missing:
        raise SystemExit(f"seed {seed}: {n_missing} row ids have no geometry")
    geoms = dict(zip(need.tolist(), polys))
    parts = explode_raw(geoms)                    # sorted by (row_id, part), as need is sorted
    log(f"  {len(parts):,} parts from {len(geoms):,} polygons; {int(parts.repaired.sum())} "
        f"from invalid parts repaired by make_valid ({time.time() - t0:.0f}s)")

    metrics = part_table(parts, idx, layers, t0)
    pos = check_stored_lobes(seed, seed_lobes, parts, metrics)
    out = dict(metrics=metrics, pos_of_lobe=pd.Series(pos, index=seed_lobes.index),
               water_only={}, null=[])
    if sel is None:
        return out

    zone_of = dict(zip(idx.row_id.to_numpy().tolist(), idx.FLD_ZONE.to_numpy().tolist()))
    part_zone = np.array([zone_of[r] for r in parts.row_id.to_numpy().tolist()], dtype=object)
    polys_a = {r: geoms[r] for r in idx.row_id.to_numpy()[near_a].tolist()}
    rule = LobeRule(parts, part_zone, polys_a, tiles)
    covers_selected = rule.covers(pts_xy.loc[sel.point_id, ["x", "y"]].to_numpy())
    budget_lobes = seed_lobes[seed_lobes.budget == args.budget]
    out["water_only"] = rebuild_observed(seed, rule, parts, sel, big, budget_lobes, budget_runs,
                                         out["pos_of_lobe"], pts_xy, covers_selected)
    log_cross_run_matches(seed, budget_lobes, out["pos_of_lobe"])

    if null_rows is None or not len(null_rows):
        return out
    t1 = time.time()
    # As in every lobes.py run: no region lobe of a component of >= MIN_SIZE points, built or
    # not, can be matched_A; nor one of a component with stored lobes.
    built = set(budget_lobes.component.unique().tolist())
    any_region = region_mask(rule, sel, pts_xy, sorted(built | set(big)))
    out["null"] = null_patch_lobes(seed, rule, parts, null_rows, pts_xy,
                                   covers_selected | any_region)
    log(f"  null: {len(out['null'])} patches given lobes ({time.time() - t1:.0f}s)")
    return out


# ======================================================================== summary
def region_info(sel: pd.DataFrame, comp: int, names: dict, Lreg: pd.DataFrame) -> dict:
    """Counties (FIPS with point counts, and names) and DFIRMs of a region's points, and the
    DFIRMs and SOURCE_CITs of its region lobes (Lreg) by km of edge."""
    P = sel[sel.component == comp]
    # The county each point lies in; county_fips (the DFIRM's county) only for a points table
    # from before regions.py carried meta_point_county.
    if "meta_point_county" in P:
        county = P.meta_point_county.astype(object).where(P.meta_point_county.notna(),
                                                          P.county_fips)
    else:
        county = P.county_fips
    county_n = county.astype(str).value_counts()
    dfirm_n = P.dfirm_id.astype(str).value_counts()
    lobe_dfirm_m = Lreg.groupby("dfirm_id").perimeter_m.sum().sort_values(ascending=False)
    source_m = Lreg.groupby("source_cit").perimeter_m.sum().sort_values(ascending=False)
    return dict(
        counties=" ".join(f"{k}({v})" for k, v in county_n.items()),
        county_names="; ".join(f"{names.get(k, '?')}" for k in county_n.index),
        dfirms=" ".join(f"{k}({v})" for k, v in dfirm_n.items()),
        lobe_dfirms=" ".join(f"{k}:{v / 1e3:.0f}km" for k, v in lobe_dfirm_m.items()),
        region_source_cits=" ".join(f"{k}:{v / 1e3:.0f}km" for k, v in source_m.head(6).items()))


def add_category_columns(E: pd.DataFrame, null_frame: pd.DataFrame | None) -> list[tuple]:
    """Add a 0/1 column '<column>=<level>' (NaN where the column is missing) to E and the null
    frame for every CATEGORICAL level seen in E. Returns those columns as 'share' metrics."""
    cats = []
    for c in CATEGORICAL:
        for level in sorted(E[c].dropna().astype(str).unique()):
            col = f"{c}={level}"
            for frame in (E, null_frame):
                if frame is not None:
                    is_level = (frame[c].astype(str) == level).astype(float)
                    frame[col] = np.where(frame[c].isna(), np.nan, is_level)
            cats.append((col, "share"))
    return cats


def rows_by_region(E: pd.DataFrame, regions: list[tuple]) -> dict:
    """region (budget, seed, component) -> role -> positions of its rows in E."""
    positions = {}
    for i, (b, s, c, role) in enumerate(zip(E.budget, E.seed, E.component, E.role)):
        positions.setdefault((int(b), int(s), int(c), role), []).append(i)
    out = {}
    for reg in regions:
        out[reg] = {}
        for role in ROLES:
            out[reg][role] = np.array(positions.get((*reg, role), []), dtype=np.int64)
    return out


def pool(regs: list[tuple], index_sets: dict) -> dict:
    """Each role's rows over the regions in regs (a region drawn twice counts twice)."""
    out = {}
    for role in ROLES:
        if regs:
            out[role] = np.concatenate([index_sets[reg][role] for reg in regs])
        else:
            out[role] = np.empty(0, np.int64)
    return out


def role_stats(cols: Columns, pools: dict, metrics: list[tuple]) -> np.ndarray:
    """(n_metrics, 3): every metric over the region, matched_A and AE rows of pools."""
    out = np.empty((len(metrics), len(ROLES)))
    for j, (col, stat) in enumerate(metrics):
        for k, role in enumerate(ROLES):
            out[j, k] = cols.stat(pools[role], col, stat)
    return out


def region_bootstrap(cols: Columns, metrics: list[tuple], regs: list[tuple], index_sets: dict,
                     rng_key: list, n_boot: int):
    """Resample the regions with replacement, each bringing all its lobes. Returns (region -
    matched_A draws, region - AE draws, label), each (n_boot, n_metrics); None for < 2 regions."""
    if len(regs) < 2:
        return None
    rng = np.random.default_rng([BOOT_SEED, *rng_key])
    d_rm = np.empty((n_boot, len(metrics)))
    d_rae = np.empty((n_boot, len(metrics)))
    for d in range(n_boot):
        pick = [regs[k] for k in rng.integers(0, len(regs), len(regs))]
        v = role_stats(cols, pool(pick, index_sets), metrics)
        d_rm[d] = v[:, 0] - v[:, 1]
        d_rae[d] = v[:, 0] - v[:, 2]
    return d_rm, d_rae, f"region bootstrap ({len(regs)} regions, {n_boot} draws)"


def lobe_bootstrap(cols: Columns, metrics: list[tuple], base: dict, rng_key: list, n_boot: int):
    """Resample each role's lobes of one region with replacement (one region cannot be
    resampled over regions). Same return as region_bootstrap."""
    rng = np.random.default_rng([BOOT_SEED, 99, *rng_key])
    d_rm = np.empty((n_boot, len(metrics)))
    d_rae = np.empty((n_boot, len(metrics)))
    for d in range(n_boot):
        drawn = {}
        for role in ROLES:
            rows = base[role]
            if len(rows):
                drawn[role] = rows[rng.integers(0, len(rows), len(rows))]
            else:
                drawn[role] = rows
        v = role_stats(cols, drawn, metrics)
        d_rm[d] = v[:, 0] - v[:, 1]
        d_rae[d] = v[:, 0] - v[:, 2]
    return d_rm, d_rae, f"lobe bootstrap within region ({n_boot} draws)"


def null_differences(null_cols: Columns | None, metrics: list[tuple], null_tab: dict,
                     regs: list[tuple], budget: int):
    """Region - matched_A of every null draw; draw d pools the d-th patch of every region in regs,
    the way the observed value pools the regions. Returns (draws, (n_draws, n_metrics) array), or
    None unless every region in regs has a null."""
    if null_cols is None or not regs:
        return None
    for b, s, c in regs:
        if b != budget or (s, c) not in null_tab:
            return None
    common = set.intersection(*[set(null_tab[(s, c)]) for _, s, c in regs])
    draws = sorted(common)
    out = np.full((len(draws), len(metrics)), np.nan)
    for k, d in enumerate(draws):
        patches = [null_tab[(s, c)][d] for _, s, c in regs]
        region_rows = np.concatenate([p["region_ix"] for p in patches])
        matched_rows = np.concatenate([p["matched_ix"] for p in patches])
        pools = {"region": region_rows, "matched_A": matched_rows, "AE": matched_rows[:0]}
        v = role_stats(null_cols, pools, metrics)
        out[k] = v[:, 0] - v[:, 1]
    return np.array(draws), out


def null_draw_frame(scope: str, region_type: str, null: tuple, metrics: list[tuple]) -> pd.DataFrame:
    """A scope's null draws, one row per draw and metric."""
    draws, values = null
    n = len(metrics)
    return pd.DataFrame(dict(
        scope=scope, region_type=region_type, draw=np.repeat(draws, n),
        metric=[m for m, _ in metrics] * len(draws), stat=[st for _, st in metrics] * len(draws),
        d_rm=values.ravel()))


def summary_rows(label: dict, pools: dict, cols: Columns, metrics: list[tuple], boot, null,
                 extra: dict) -> list[dict]:
    """One row per metric: label, the per-role statistic, both contrasts, the bootstrap interval
    and null summary when given, then the extra columns."""
    v = role_stats(cols, pools, metrics)
    rows = []
    for j, (m, st) in enumerate(metrics):
        row = dict(label)
        row.update(metric=m, stat=st, n_region=len(pools["region"]),
                   n_matched=len(pools["matched_A"]), n_AE=len(pools["AE"]), region=v[j, 0],
                   matched_A=v[j, 1], AE=v[j, 2], d_rm=v[j, 0] - v[j, 1], d_rAE=v[j, 0] - v[j, 2])
        if boot is not None:
            lo, hi = ci(boot[0][:, j])
            ae_lo, ae_hi = ci(boot[1][:, j])
            row.update(d_rm_lo=lo, d_rm_hi=hi, d_rAE_lo=ae_lo, d_rAE_hi=ae_hi, ci_method=boot[2])
        if null is not None:
            nv = null[1][:, j]
            row.update(null_pct=mid_rank_pct(row["d_rm"], nv), null_n=int(np.isfinite(nv).sum()),
                       null_median=float(np.nanmedian(nv)),
                       null_lo=float(np.nanpercentile(nv, 2.5)),
                       null_hi=float(np.nanpercentile(nv, 97.5)))
        row.update(extra)
        rows.append(row)
    return rows


def point_lobe_rows(E: pd.DataFrame, regs: list[tuple], by_region: dict) -> dict:
    """Per region: the region lobes that hold a region point and the matched_A lobes matched to
    them (AE unchanged)."""
    n_points = E.n_points.to_numpy()
    lobe_id = E.lobe_id.to_numpy()
    matched_to = E.matched_to.to_numpy()
    out = {}
    for reg in regs:
        region = by_region[reg]["region"]
        matched = by_region[reg]["matched_A"]
        holds_point = n_points[region] > 0
        holding_ids = set(lobe_id[region][holds_point])
        out[reg] = {"region": region[holds_point],
                    "matched_A": matched[np.isin(matched_to[matched], list(holding_ids))],
                    "AE": by_region[reg]["AE"]}
    return out


def dedup_rows(E: pd.DataFrame, regs: list[tuple], by_region: dict) -> dict:
    """Per region: each role's lobes not already pooled from an earlier region, so a lobe shared
    by two regions enters once per role."""
    src_row_id = E.src_row_id.to_numpy()
    part = E.part.to_numpy()
    seen = {role: set() for role in ROLES}
    out = {}
    for reg in regs:
        out[reg] = {}
        for role in ROLES:
            rows = by_region[reg][role]
            keys = list(zip(src_row_id[rows].tolist(), part[rows].tolist()))
            new = np.array([key not in seen[role] for key in keys], bool)
            seen[role].update(keys)
            out[reg][role] = rows[new]
    return out


def land_lobe_rows(E: pd.DataFrame, regs: list[tuple], by_region: dict) -> dict:
    """Per region: without the region lobes that only a water point brings in."""
    water_only = E.water_only.to_numpy().astype(bool)
    out = {}
    for reg in regs:
        out[reg] = {}
        for role in ROLES:
            rows = by_region[reg][role]
            if role == "region":
                rows = rows[~water_only[rows]]
            out[reg][role] = rows
    return out


def summarise(E: pd.DataFrame, regions: list[tuple], types: dict, null_tab: dict, info: dict,
              a, log, null_keep: list | None = None) -> pd.DataFrame:
    """The summary table. E: one row per lobe and region with a RangeIndex; regions: (budget,
    seed, component); types: region -> region type; null_tab: (seed, component) -> draw ->
    region_ix / matched_ix into null_tab["_frame"]; info: region -> extra columns; a: budget and
    n_boot. If null_keep is a list, the type and pooled null draws are appended to it.

    Scopes: region (lobe bootstrap), type and pooled over the S = a.budget regions (region
    bootstrap), pooled_S<b> for other budgets. Pooled sensitivities (subset): point_lobes,
    dedup, land_lobes. The primary row is pooled / all / vertex_spacing_m / median; met when its
    interval lies above 0."""
    null_frame = null_tab.get("_frame") if null_tab else None
    cats = add_category_columns(E, null_frame)
    metrics = METRICS + cats + [(col, "pshare") for col, _ in cats]
    ratio_cols = {"perimeter_m", "n_vertices", "straight_m", "straight_vertex_m", "area_m2",
                  "lomr_share"}
    needed = sorted({m for m, _ in metrics} | ratio_cols)
    cols = Columns(E, needed)
    null_cols = Columns(null_frame, needed) if null_frame is not None else None
    by_region = rows_by_region(E, regions)
    n_boot, budget = a.n_boot, a.budget
    rows = []

    t0 = time.time()
    for reg in regions:
        b, s, c = reg
        label = dict(scope="region", subset="all", budget=b, seed=s, component=c,
                     region_type=types.get(reg, ""), n_regions=1)
        boot = lobe_bootstrap(cols, metrics, by_region[reg], [b, s, c], n_boot)
        null = null_differences(null_cols, metrics, null_tab, [reg], budget)
        rows += summary_rows(label, pool([reg], by_region), cols, metrics, boot, null,
                             info.get(reg, {}))
    log(f"  per-region rows ({time.time() - t0:.0f}s)")

    main_regs = [r for r in regions if r[0] == budget]
    scopes = [("type", "positive-dominated", [1, 0]), ("type", "negative-dominated", [1, 1]),
              ("pooled", "", [0])]
    for scope, typ, rng_key in scopes:
        if scope == "type":
            regs = [r for r in main_regs if types.get(r) == typ]
        else:
            regs = main_regs
        null = null_differences(null_cols, metrics, null_tab, regs, budget)
        if null_keep is not None and null is not None:
            null_keep.append(null_draw_frame(scope, typ, null, metrics))
        boot = region_bootstrap(cols, metrics, regs, by_region, rng_key, n_boot)
        label = dict(scope=scope, subset="all", budget=budget, seed="", component="",
                     region_type=typ, n_regions=len(regs))
        rows += summary_rows(label, pool(regs, by_region), cols, metrics, boot, null, {})
    log(f"  type and pooled rows ({time.time() - t0:.0f}s)")

    # Sensitivities of the pooled statistic; only point_lobes gets a region bootstrap.
    point_lobes = point_lobe_rows(E, main_regs, by_region)
    sensitivities = [("point_lobes", point_lobes,
                      region_bootstrap(cols, metrics, main_regs, point_lobes, [3], n_boot)),
                     ("dedup", dedup_rows(E, main_regs, by_region), None),
                     ("land_lobes", land_lobe_rows(E, main_regs, by_region), None)]
    for subset, index_sets, boot in sensitivities:
        label = dict(scope="pooled", subset=subset, budget=budget, seed="", component="",
                     region_type="", n_regions=len(main_regs))
        rows += summary_rows(label, pool(main_regs, index_sets), cols, metrics, boot, None, {})
    for b in sorted({r[0] for r in regions} - {budget}):
        regs = [r for r in regions if r[0] == b]
        boot = region_bootstrap(cols, metrics, regs, by_region, [2, b], n_boot)
        label = dict(scope=f"pooled_S{b}", subset="all", budget=b, seed="", component="",
                     region_type="", n_regions=len(regs))
        rows += summary_rows(label, pool(regs, by_region), cols, metrics, boot, None, {})
    log(f"  sensitivities ({time.time() - t0:.0f}s)")

    S = pd.DataFrame(rows)
    S["primary"] = ((S.scope == "pooled") & (S.subset == "all") & (S.metric == "vertex_spacing_m")
                    & (S.stat == "median"))
    S["predicted"] = np.where(S.primary, "positive", "")
    S["met"] = pd.Series(np.where(S.primary, (S.d_rm > 0) & (S.d_rm_lo > 0), None), dtype=object)
    for c in SUMMARY_COLUMNS:
        if c not in S:
            S[c] = np.nan
    return S[SUMMARY_COLUMNS + [c for c in S.columns if c not in SUMMARY_COLUMNS]]


# ======================================================================== main
def region_types(land: pd.DataFrame, regs: list[tuple], args) -> dict:
    """(budget, seed, component) -> region type from regions_land.csv; stops for lobes of a region
    without land points and warns about land regions without lobes."""
    types = {}
    for r in land.itertuples():
        types[(args.budget, int(r.seed), int(r.component))] = r.type
    missing = []
    for reg in regs:
        if reg[0] == args.budget and types.get(reg, "no land points") == "no land points":
            missing.append(reg)
    if missing:
        raise SystemExit(f"lobes for regions without land points: {missing}")
    want = set()
    for r in land.itertuples():
        if r.type != "no land points" and (not args.seeds or r.seed in args.seeds):
            want.add((args.budget, int(r.seed), int(r.component)))
    if want - set(regs):
        log(f"  WARNING: land regions without lobes: {sorted(want - set(regs))}")
    drops = []
    for r in land.itertuples():
        if r.dropped_water:
            drops.append(f"s{int(r.seed)} c{int(r.component)} drops {int(r.dropped_water)} "
                         f"of {int(r.n)}")
    log("  land rule (regions_land.csv): " + "; ".join(drops))
    return types


def water_only_mask(E: pd.DataFrame, passes: dict, seeds: list[int], budget: int) -> np.ndarray:
    """True for the region lobe rows that only a water point brings in."""
    mask = np.zeros(len(E), bool)
    for s in seeds:
        met = passes[s]["metrics"]
        for c, part_rows in passes[s]["water_only"].items():
            keys = {(met.row_id.iat[i], met.part.iat[i]) for i in part_rows}
            is_comp = ((E.seed == s) & (E.budget == budget) & (E.component == c)
                       & (E.role == "region")).to_numpy()
            is_key = np.array([(r, p) in keys for r, p in zip(E.src_row_id, E.part)])
            mask |= is_comp & is_key
    return mask


def lobe_table(lobes: pd.DataFrame, passes: dict, seeds: list[int], types: dict,
               budget: int) -> pd.DataFrame:
    """The lobe rows (RangeIndex) with their original part's metrics and attributes, the stored
    lobe's area, region_type and water_only. area_m2 and perimeter_m are the original part's."""
    per_seed = []
    for s in seeds:
        pos = passes[s]["pos_of_lobe"]
        per_seed.append(passes[s]["metrics"].iloc[pos.to_numpy()].set_axis(pos.index))
    metrics = pd.concat(per_seed).loc[lobes.index]
    E = pd.concat([lobes.drop(columns=["area_m2", "perimeter_m"]),
                   metrics.drop(columns=["row_id", "part"])], axis=1)
    E["stored_area_m2"] = shapely.area(np.array(lobes.geometry.to_numpy(), dtype=object))
    E = pd.DataFrame(E.drop(columns=["geometry"])).reset_index(drop=True)
    E["region_type"] = [types.get((b, s, c), "") for b, s, c in zip(E.budget, E.seed, E.component)]
    E["water_only"] = water_only_mask(E, passes, seeds, budget)
    return E


def null_table(passes: dict, seeds: list[int]) -> dict:
    """summarise's null_tab: the parts of every seed with null patches in one frame ("_frame"),
    and (seed, component) -> draw -> region_ix / matched_ix as rows of that frame."""
    null_tab, frames, offset = {}, [], 0
    for s in seeds:
        if not passes[s]["null"]:
            continue
        frames.append(passes[s]["metrics"])
        for r in passes[s]["null"]:
            draws = null_tab.setdefault((s, r["component"]), {})
            draws[r["draw"]] = dict(region_ix=r["region_ix"] + offset,
                                    matched_ix=r["matched_ix"] + offset)
        offset += len(passes[s]["metrics"])
    if frames:
        null_tab["_frame"] = pd.concat(frames, ignore_index=True)
    return null_tab


def write_null_patches(path: Path, passes: dict, seeds: list[int], null_tab: dict) -> None:
    """One row per null patch: lobe counts, tiles and the vertex spacings of its region and
    matched lobes (and their medians)."""
    spacing = null_tab["_frame"].vertex_spacing_m.to_numpy()
    rows = []
    for s in seeds:
        for r in passes[s]["null"]:
            ix = null_tab[(s, r["component"])][r["draw"]]
            region_spacing = spacing[ix["region_ix"]]
            matched_spacing = spacing[ix["matched_ix"]]
            med_region = np.median(region_spacing) if r["n_region"] else np.nan
            med_matched = np.median(matched_spacing) if r["n_matched"] else np.nan
            rows.append(dict(seed=s, component=r["component"], draw=r["draw"],
                             n_region=r["n_region"], n_matched=r["n_matched"],
                             n_eligible=r["n_eligible"], tiles=r["tiles"],
                             med_spacing_region=med_region, med_spacing_matched=med_matched,
                             spacing_region=region_spacing, spacing_matched=matched_spacing))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    log(f"  wrote {path} (per patch: vertex spacing of its region and matched lobes)")


def run_of(runs: list[dict], seed: int, budget: int) -> dict:
    """The first --lobes run of this seed and budget (all read the same points table)."""
    for run in runs:
        if run["seed"] == seed and run["budget"] == budget:
            return run
    raise SystemExit(f"no --lobes file for seed {seed}, S = {budget}")


def region_infos(regs: list[tuple], runs: list[dict], E: pd.DataFrame, land: pd.DataFrame,
                 args) -> dict:
    """Per region: region_info, its land-rule counts and n_water_only_lobes."""
    counties = gpd.read_parquet(args.counties)
    names = dict(zip(counties.GEOID.astype(str), counties.NAME.astype(str)))
    info = {}
    for b, s, c in regs:
        run = run_of(runs, s, b)
        sel = read_points(args.regions_dir, run["cell"], s, run["variant"], b)
        in_region = (E.budget == b) & (E.seed == s) & (E.component == c)
        reg_info = region_info(sel, c, names, E[in_region & (E.role == "region")])
        if b == args.budget:
            land_row = land[(land.seed == s) & (land.component == c)]
            if len(land_row):
                reg_info.update(n_points=int(land_row.n.iloc[0]), n_land=int(land_row.n_land.iloc[0]),
                                dropped_water=int(land_row.dropped_water.iloc[0]))
        reg_info["n_water_only_lobes"] = int(E[in_region].water_only.sum())
        info[(b, s, c)] = reg_info
    return info


def log_panel_fallback(E: pd.DataFrame) -> None:
    """Per role: lobe rows on no panel of their own DFIRM, and those that also overlap another
    DFIRM's panels."""
    also_other = (E.pan_n_all > E.pan_n) & (E.pan_fallback == 0)
    fallback, other = [], []
    for r in ROLES:
        is_role = E.role == r
        fallback.append(f"{r} {int(E.pan_fallback[is_role].sum())} of {int(is_role.sum()):,}")
        other.append(f"{r} {int(also_other[is_role].sum()):,}")
    log("panels: lobe rows whose own DFIRM has no panel under them (fallback to all panels): "
        + ", ".join(fallback)
        + f"; on no panel at all: {int((E.pan_n_all == 0).sum())}; also overlapping another "
          f"DFIRM's panels (ignored): "
        + ", ".join(other))


def log_primary(S: pd.DataFrame, budget: int) -> None:
    pr = S[S.primary].iloc[0]
    log(f"\nPRIMARY: median vertex spacing, region - matched_A, pooled over {int(pr.n_regions)} "
        f"regions (S = {budget}): {pr.region:.2f} - {pr.matched_A:.2f} = {pr.d_rm:+.2f} m "
        f"[{pr.d_rm_lo:+.2f}, {pr.d_rm_hi:+.2f}], null percentile {pr.null_pct:.1f} "
        f"(null median {pr.null_median:+.2f} [{pr.null_lo:+.2f}, {pr.null_hi:+.2f}]); predicted "
        f"positive: {'MET' if pr.met else 'NOT MET'}. Contrast region - AE: {pr.d_rAE:+.2f} m "
        f"[{pr.d_rAE_lo:+.2f}, {pr.d_rAE_hi:+.2f}] (AE median {pr.AE:.2f} m)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lobes", nargs="+", required=True, type=Path,
                    help="lobes.py tables, <cell>_s<seed>_<variant>_S<S>_lobes.parquet")
    ap.add_argument("--budget", type=int, default=500, help="the regions' budget S (primary)")
    ap.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--regions-land", type=Path,
                    default=Path("results/2026-10-17/o5/regions_land.csv"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    ap.add_argument("--idx", type=Path, default=Path("data/interim/prep/idx_FL.parquet"))
    ap.add_argument("--state", default="FL")
    ap.add_argument("--attrs", type=Path,
                    default=Path("data/interim/nfhl_haz_attrs_10state.parquet"))
    ap.add_argument("--panels", type=Path, default=Path("data/interim/nfhl_states/pan_FL.parquet"))
    ap.add_argument("--lomr", type=Path, default=Path("data/interim/nfhl_states/s_lomr_FL.parquet"))
    ap.add_argument("--counties", type=Path, default=Path("data/interim/gis/counties_FL.parquet"))
    ap.add_argument("--null", type=Path, default=Path("results/2026-10-17/o5/S2_null.parquet"),
                    help="S2 matched-growth null patches")
    ap.add_argument("--null-draws", type=int, default=0, help="0 = all draws")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", help="default: every seed in --lobes")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-polygons", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    ap.add_argument("--null-out", type=Path, help="per-patch null table (parquet), optional; "
                    "also writes <stem>_pooled.parquet (type and pooled null draws)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    sp.PREP_DIR = args.idx.parent           # sp.read_geoms reads haz_<state>_5070 from here
    lobes, runs = load_lobes(args.lobes)
    if args.seeds:
        lobes = lobes[lobes.seed.isin(args.seeds)]
        runs = [r for r in runs if r["seed"] in args.seeds]
    log(f"=== o5_polygons: {len(lobes):,} lobe rows from {len(runs)} files ===")
    for r in runs:
        log(f"  s{r['seed']} S{r['budget']} components {r['components']}: {r['file']}")
    regs = sorted({(int(b), int(s), int(c))
                   for b, s, c in zip(lobes.budget, lobes.seed, lobes.component)})
    land = pd.read_csv(args.regions_land)
    types = region_types(land, regs, args)

    idx = pd.read_parquet(args.idx,
                          columns=lb.IDX_COLS + ["STUDY_TYP", "SOURCE_CIT", "study_regime"])
    pts_xy = dataset_xy(args.dataset)
    null = None
    if not args.no_null and args.null.exists():
        null = pd.read_parquet(args.null, columns=["seed", "component", "draw",
                                                   "patch_point_ids", "region_type"])
    attrs, note = read_attrs(args.attrs, args.state)
    log("  " + note)
    layers = dict(attrs=attrs, panels=read_layer(args.panels), lomr=read_layer(args.lomr))

    seeds = sorted(lobes.seed.unique().tolist())
    passes = {}
    for s in seeds:
        passes[s] = seed_pass(s, lobes, runs, idx, pts_xy, layers, null, args)

    E = lobe_table(lobes, passes, seeds, types, args.budget)
    n_unique = len(E.drop_duplicates(["src_row_id", "part"]))
    per_role = ", ".join(f"{r} {int((E.role == r).sum()):,}" for r in ROLES)
    log(f"\n{n_unique:,} unique lobes (src_row_id, part) among {len(E):,} lobe rows; {per_role}")

    null_tab = null_table(passes, seeds)
    if args.null_out and "_frame" in null_tab:
        write_null_patches(args.null_out, passes, seeds, null_tab)
    info = region_infos(regs, runs, E, land, args)
    log_panel_fallback(E)

    log("\n--- summary")
    null_keep = [] if args.null_out else None
    summary = summarise(E, regs, types, null_tab, info, args, log, null_keep=null_keep)
    if null_keep:
        pooled_path = args.null_out.with_name(args.null_out.stem + "_pooled.parquet")
        pd.concat(null_keep, ignore_index=True).to_parquet(pooled_path, index=False)
        log(f"  wrote {pooled_path} (type and pooled null draws, every metric)")

    for path in (args.out_polygons, args.out_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    polygons = E[POLYGON_COLUMNS].sort_values(["budget", "seed", "component", "role", "lobe_id"])
    polygons.to_csv(args.out_polygons, index=False)
    summary.to_csv(args.out_summary, index=False)
    log(f"\nwrote {args.out_polygons} ({len(E):,} rows) and {args.out_summary} "
        f"({len(summary):,} rows)")

    log_primary(summary, args.budget)
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
