#!/usr/bin/env python3
"""
pull_point_extras.py -- per-point provenance sidecar for the boundary analysis.

The boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md section 3.3, audit brief
Part D1) needs three facts per point that the point table does not carry:

    meta_firm_pan          the FIRM panel (map sheet) the point lies in, and its
                           EFF_DATE
    meta_near_*            the nearest SFHA polygon: which one, its zone, how far
    meta_edge_dist_m       for a positive, how far it sits inside the polygon
                           that labelled it

They go in a sidecar keyed by point_id, not into dataset_<ST>.parquet, so that
the dataset and its 45 features stay exactly as the models saw them. Every
column is meta_: these are the "held-out maps" the paper tests the regions
against, and a model that could see them would make that test meaningless.

DECISIONS, EACH WITH ITS REASON
===============================

1. THE NEAREST SFHA POLYGON IS THE ONE sample_points.distances_to_sfha MEASURES.
   Same candidates (idx rows with SFHA_TF == 'T', every SFHA zone, not just A
   and AE), same geometry (haz_<ST>_5070.parquet), same exact shapely distance in
   EPSG:5070. The difference is that the polygon's identity is kept, and that
   the 5.4 GB hazard file is streamed one row group at a time, each read once,
   instead of once per 50 km tile. A polygon within --radius of a point has its
   bounding box within --radius of the point, so filtering candidates on the
   idx bbox columns first loses nothing: every distance <= --radius is exact.
   Beyond it the column is null. Under protocol v2 every point is <= 5,000 m
   from an SFHA (positives 0, band negatives 500-5,000 m), so 5,000 m is the
   default. On equal distances the lower row_id wins.

2. FOR A POSITIVE THE NEAREST POLYGON IS THE LABELLING POLYGON. A positive is at
   distance 0 from the polygon it was drawn in, but adjacent or overlapping SFHA
   polygons can also be at 0. So when the labelling polygon covers the point it
   is used outright, and the cases where the plain nearest search picked another
   polygon are counted.

3. THE LABELLING POLYGON. v2 points carry meta_src_row_id (the idx row_id of
   the polygon behind each point, sample_points.py note 8); it is used whenever
   it is set. v1 points do not, so a positive's polygon is recovered from
   (dfirm_id, meta_fld_ar_id) through idx_<ST>: the pair, because FLD_AR_ID
   alone repeats across DFIRMs, and where the pair itself names several rows (39
   rows in Florida) the one candidate that covers the point is taken, exactly as
   aotil/regions.py does. A v1 band negative's ring polygon was never recorded,
   which is fine: meta_edge_dist_m is defined for positives only.

4. meta_edge_dist_m is the distance to the BOUNDARY of the labelling polygon,
   holes included. That boundary includes edges the polygon shares with
   neighbouring SFHA polygons (an A/AE contact, a polygon split at a panel or
   county line), not only the edge to unmapped or Zone X ground. It is null for
   a positive that its polygon does not cover; there should be none.

5. THE PANEL IS THE ONE OF THE POINT'S OWN DFIRM. FIRM panels are full printed
   rectangles, and in Florida they overlap across county DFIRMs (about 65,000
   km2 of pairwise overlap: a Monroe panel can lie wholly inside a Miami-Dade
   one). A point inside several panels takes, among the panels whose DFIRM_ID
   equals the point's dfirm_id, the lowest FIRM_PAN; a point in none of its own
   DFIRM's panels takes the lowest FIRM_PAN of the others. meta_firm_pan_n
   records how many panels contain the point. This is the rule
   sample_points.attach_panel_meta applies to the dataset's meta_eff_date,
   meta_panel_scale and meta_panel_typ, with the same "within" in EPSG:5070, so
   both name one sheet; meta_firm_pan_eff_date is that sheet's EFF_DATE and
   equals meta_eff_date for a dataset built by the current sampler (v1 took
   the first of several panels). The one difference is that this script
   repairs the one Florida panel that reprojection leaves self-intersecting
   (12081C0333E, a 3.1 km spur under 1 m wide); the sampler tests the raw
   ring, and the two can disagree only inside that spur. Checked: the same
   panel row for all 98,578 v1 points and all 80,000 points of a v2 test build.

OUTPUT
======
<out> (default data/interim/analysis/point_extras_<ST>.parquet), one row per
dataset row, in dataset order:
    point_id
    meta_firm_pan, meta_firm_pan_n, meta_firm_pan_eff_date (naive UTC, as meta_eff_date)
    meta_near_src_row_id, meta_near_fld_ar_id, meta_near_fld_zone, meta_near_dist_m
    meta_edge_src_row_id   the labelling polygon whose edge meta_edge_dist_m measures
                           (meta_src_row_id where the dataset has it)
    meta_edge_dist_m
The settings and input paths travel in the parquet schema metadata.

USAGE
=====
    python pull_point_extras.py --state FL
    python pull_point_extras.py --state FL --dataset data/processed/dataset_FL.parquet \\
        --out /tmp/x/point_extras_FL.parquet --limit 2000      # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely import STRtree
from pyproj import Transformer

PROCESSED = Path("data/processed")
PREP_DIR = Path("data/interim/prep")
NFHL_DIR = Path("data/interim/nfhl_states")
OUT_DIR = Path("data/interim/analysis")

CRS_WORK = 5070        # CONUS Albers, metres: the sampler's working CRS
CRS_LL = 4269          # NAD83 lon/lat, as sample_points writes lon/lat
RADIUS_M = 5000.0      # band_max_m: no v2 point is farther than this from an SFHA
LIMIT_SEED = 0

POINT_COLS = ["point_id", "domain", "label", "dfirm_id", "meta_fld_ar_id", "lon", "lat"]
OPTIONAL_COLS = ["meta_src_row_id", "meta_dist_to_sfha_m", "meta_eff_date", "geometry"]
IDX_COLS = ["row_id", "DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "SFHA_TF",
            "minx", "miny", "maxx", "maxy"]


# =========================================================================
# inputs
# =========================================================================
def default_paths(state: str) -> dict:
    return dict(dataset=PROCESSED / f"dataset_{state}.parquet",
                idx=PREP_DIR / f"idx_{state}.parquet",
                haz=PREP_DIR / f"haz_{state}_5070.parquet",
                panels=NFHL_DIR / f"pan_{state}.parquet",
                out=OUT_DIR / f"point_extras_{state}.parquet")


def geo_crs(schema: pa.Schema) -> int | None:
    """EPSG code of a GeoParquet's primary geometry, or None if unstated."""
    geo = json.loads((schema.metadata or {}).get(b"geo", b"{}") or b"{}")
    col = geo.get("columns", {}).get(geo.get("primary_column", "geometry"), {})
    crs = col.get("crs") or {}
    cid = crs.get("id", {}) if isinstance(crs, dict) else {}
    return int(cid["code"]) if cid.get("authority") == "EPSG" else None


def load_points(path: Path, limit: int | None = None,
                cols: list[str] = POINT_COLS) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Point rows plus their (x, y) in EPSG:5070 and a note on where xy came from.

    The dataset's own point geometry is used when it is there: it holds the exact
    coordinates the sampler tested containment and distances with, which a
    lon/lat round trip reproduces only to ~1e-9 m.
    """
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    missing = [c for c in cols if c not in names]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    df = pq.read_table(path, columns=cols + [c for c in OPTIONAL_COLS
                                             if c in names and c not in cols]).to_pandas()
    if not df.point_id.is_unique:
        sys.exit(f"{path}: point_id is not unique")
    if limit is not None and limit < len(df):
        keep = np.sort(np.random.default_rng(LIMIT_SEED).choice(len(df), limit, replace=False))
        df = df.iloc[keep].reset_index(drop=True)

    epsg = geo_crs(pf.schema_arrow) if "geometry" in df else None
    if epsg is not None:
        xy = shapely.get_coordinates(shapely.from_wkb(df.pop("geometry").to_numpy()))
        how = f"dataset geometry (EPSG:{epsg})"
        if epsg != CRS_WORK:
            tf = Transformer.from_crs(epsg, CRS_WORK, always_xy=True)
            xy = np.column_stack(tf.transform(xy[:, 0], xy[:, 1]))
            how += f" -> EPSG:{CRS_WORK}"
    else:
        df = df.drop(columns="geometry", errors="ignore")
        tf = Transformer.from_crs(CRS_LL, CRS_WORK, always_xy=True)
        xy = np.column_stack(tf.transform(df.lon.to_numpy(float), df.lat.to_numpy(float)))
        how = f"lon/lat projected EPSG:{CRS_LL} -> {CRS_WORK}"
    if len(xy) != len(df):
        sys.exit(f"{path}: {len(xy)} coordinates for {len(df)} rows (empty or multi geometry)")
    return df, xy, how


def load_index(path: Path) -> pd.DataFrame:
    idx = pq.read_table(path, columns=IDX_COLS).to_pandas()
    for c in ("DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "SFHA_TF"):
        idx[c] = idx[c].astype(object)
    return idx


class HazReader:
    """Geometry from haz_<ST>_5070.parquet by row_id, one row group at a time.

    The file is 5.4 GB for Florida in 10,000-row groups and must never be read
    whole. prep_state writes row_id 0..n-1 in order, so row_id r sits at
    position r - start[g] of group g; that is checked on every read and a group
    whose ids are out of order falls back to matching them.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.pf = pq.ParquetFile(self.path)
        md = self.pf.metadata
        sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        self.start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n_rows = int(self.start[-1])

    def group_of(self, row_ids: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.start, row_ids, side="right") - 1

    def iter_groups(self, row_ids):
        """Yield (group, sorted row_ids found, geometries) for the wanted rows."""
        want = np.unique(np.asarray(row_ids, dtype=np.int64))
        want = want[(want >= 0) & (want < self.n_rows)]
        grp = self.group_of(want)
        for gi in np.unique(grp):
            sel = want[grp == gi]
            t = self.pf.read_row_group(int(gi), columns=["row_id", "geometry"])
            rid = t.column("row_id").to_numpy()
            pos = sel - self.start[gi]
            if pos.max() >= len(rid) or not np.array_equal(rid[pos], sel):
                order = np.argsort(rid)
                hit = np.searchsorted(rid[order], sel)
                ok = (hit < len(rid)) & (rid[order][np.minimum(hit, len(rid) - 1)] == sel)
                sel, pos = sel[ok], order[hit[ok]]
            wkb = t.column("geometry").take(pa.array(pos)).to_numpy(zero_copy_only=False)
            yield int(gi), sel, shapely.from_wkb(wkb)

    def read(self, row_ids) -> dict[int, object]:
        out: dict[int, object] = {}
        for _, sel, geo in self.iter_groups(row_ids):
            out.update(zip(sel.tolist(), geo))
        return out


# =========================================================================
# the labelling polygon (decision 3)
# =========================================================================
def resolve_label_polygons(df: pd.DataFrame, xy: np.ndarray, idx: pd.DataFrame,
                           haz: HazReader) -> tuple[np.ndarray, dict]:
    """idx row_id of the polygon that labelled each positive; -1 for the rest."""
    n = len(df)
    src = np.full(n, -1, dtype=np.int64)
    pos = df.label.to_numpy() == 1
    info = dict(positives=int(pos.sum()), from_column=0, key_unique=0,
                key_ambiguous=0, key_ambiguous_resolved=0, key_not_in_index=0,
                no_key=0, column_key_mismatch=0)

    if "meta_src_row_id" in df:
        s = pd.to_numeric(df.meta_src_row_id, errors="coerce").to_numpy(dtype=float)
        have = pos & np.isfinite(s)
        src[have] = s[have].astype(np.int64)
        info["from_column"] = int(have.sum())
        if have.any() and "meta_fld_ar_id" in df:
            ar = idx.set_index("row_id").FLD_AR_ID.reindex(src[have]).to_numpy()
            info["column_key_mismatch"] = int(
                (ar.astype(str) != df.meta_fld_ar_id.to_numpy()[have].astype(str)).sum())

    todo = pos & (src < 0)
    has_key = todo & df.meta_fld_ar_id.notna().to_numpy()
    info["no_key"] = int((todo & ~has_key).sum())
    if not has_key.any():
        return src, info

    key = pd.DataFrame(dict(i=np.flatnonzero(has_key),
                            DFIRM_ID=df.dfirm_id.to_numpy()[has_key].astype(str),
                            FLD_AR_ID=df.meta_fld_ar_id.to_numpy()[has_key].astype(str)))
    cand = key.merge(idx[["row_id", "DFIRM_ID", "FLD_AR_ID"]].astype(
        {"DFIRM_ID": str, "FLD_AR_ID": str}), on=["DFIRM_ID", "FLD_AR_ID"], how="inner")
    n_cand = cand.groupby("i").row_id.transform("size")
    one = cand[n_cand == 1]
    src[one.i.to_numpy()] = one.row_id.to_numpy()
    info["key_unique"] = len(one)
    info["key_not_in_index"] = int(has_key.sum()) - int(cand.i.nunique())

    amb = cand[n_cand > 1]
    info["key_ambiguous"] = int(amb.i.nunique())
    if len(amb):
        geoms = haz.read(amb.row_id.unique())
        for i, grp in amb.groupby("i"):
            pt = shapely.Point(xy[i])
            hits = [r for r in grp.row_id if geoms.get(r) is not None and geoms[r].covers(pt)]
            if len(hits) == 1:
                src[i] = hits[0]
                info["key_ambiguous_resolved"] += 1
    return src, info


# =========================================================================
# nearest SFHA polygon and edge distance, one pass over the hazard file
# =========================================================================
def candidate_rows(sfha: pd.DataFrame, xy: np.ndarray, radius: float) -> np.ndarray:
    """SFHA row_ids whose bbox lies within `radius` of some point (decision 1).

    Points are binned into radius-sized cells; a polygon is kept if its bbox meets
    an occupied cell grown by radius. That is a superset of the polygons within
    radius of any point, at the cost of a few hundred thousand box tests.
    """
    c = max(radius, 1000.0)
    cells = np.unique(np.floor(xy / c).astype(np.int64), axis=0)
    boxes = shapely.box(cells[:, 0] * c - radius, cells[:, 1] * c - radius,
                        (cells[:, 0] + 1) * c + radius, (cells[:, 1] + 1) * c + radius)
    pb = shapely.box(sfha.minx.to_numpy(), sfha.miny.to_numpy(),
                     sfha.maxx.to_numpy(), sfha.maxy.to_numpy())
    hit = STRtree(boxes).query(pb, predicate="intersects")
    return sfha.row_id.to_numpy()[np.unique(hit[0])]


def nearest_and_edges(haz: HazReader, sfha: pd.DataFrame, xy: np.ndarray,
                      src: np.ndarray, radius: float) -> dict:
    """Stream the hazard file once: nearest SFHA polygon for every point, and the
    covering test and boundary distance for every positive's labelling polygon."""
    n = len(xy)
    pts = shapely.points(xy)
    best_d = np.full(n, np.inf)
    best_r = np.full(n, -1, dtype=np.int64)
    covered = np.zeros(n, dtype=bool)
    edge = np.full(n, np.nan)

    cand = candidate_rows(sfha, xy, radius)
    want = np.union1d(cand, src[src >= 0])
    is_cand = np.zeros(haz.n_rows, dtype=bool)
    is_cand[cand] = True
    bb = sfha.set_index("row_id")[["minx", "miny", "maxx", "maxy"]]
    n_groups = len(np.unique(haz.group_of(want)))
    print(f"  {len(cand):,} of {len(sfha):,} SFHA polygons within {radius:.0f} m of a "
          f"point's cell; reading {n_groups} of {len(haz.start) - 1} row groups", flush=True)

    t0 = time.time()
    for k, (gi, sel, geo) in enumerate(haz.iter_groups(want), 1):
        ok = np.array([g is not None and not g.is_empty for g in geo])
        m = is_cand[sel] & ok
        if m.any():
            g_rows, g_geo = sel[m], geo[m]
            b = bb.loc[g_rows].to_numpy()
            x0, y0 = b[:, 0].min() - radius, b[:, 1].min() - radius
            x1, y1 = b[:, 2].max() + radius, b[:, 3].max() + radius
            near = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
                                  (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
            if near.size:
                (pi, gj), d = STRtree(g_geo).query_nearest(
                    pts[near], max_distance=radius, return_distance=True, all_matches=True)
                pi, r = near[pi], g_rows[gj]
                o = np.lexsort((r, d, pi))
                pi, d, r = pi[o], d[o], r[o]
                first = np.r_[True, pi[1:] != pi[:-1]]
                pi, d, r = pi[first], d[first], r[first]
                better = (d < best_d[pi]) | ((d == best_d[pi]) & (r < best_r[pi]))
                best_d[pi[better]] = d[better]
                best_r[pi[better]] = r[better]

        mine = np.flatnonzero(np.isin(src, sel))
        if mine.size:
            gpos = np.searchsorted(sel, src[mine])
            have = ok[gpos]
            mine, gpos = mine[have], gpos[have]
            covered[mine] = shapely.covers(geo[gpos], pts[mine])
            uniq, inv = np.unique(gpos, return_inverse=True)
            bnd = shapely.boundary(geo[uniq])
            edge[mine] = shapely.distance(pts[mine], bnd[inv])
        if k % 10 == 0 or k == n_groups:
            print(f"    row groups {k}/{n_groups}  ({time.time() - t0:.0f}s)", flush=True)

    edge[~covered] = np.nan
    return dict(near_d=best_d, near_r=best_r, covered=covered, edge=edge)


# =========================================================================
# FIRM panel (decision 5)
# =========================================================================
def load_panels(path: Path):
    """Panels in EPSG:5070, plus the mask of those reprojection left invalid
    (repaired here with make_valid)."""
    import geopandas as gpd
    pan = gpd.read_parquet(path)
    if pan.crs is None:
        sys.exit(f"{path}: no CRS")
    pan = pan.to_crs(CRS_WORK).reset_index(drop=True)
    g = np.asarray(pan.geometry.values)
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.make_valid(g[bad])
        pan = pan.set_geometry(g)
    return pan, bad


def panel_of(pan, xy: np.ndarray, dfirm: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(row of the chosen panel in pan or -1, number of containing panels, points
    in several panels that took their own DFIRM's). The choice is decision 5's."""
    n = len(xy)
    hits = STRtree(np.asarray(pan.geometry.values)).query(shapely.points(xy), predicate="within")
    count = np.bincount(hits[0], minlength=n)
    h = pd.DataFrame(dict(i=hits[0], row=hits[1], pan=pan.FIRM_PAN.to_numpy()[hits[1]],
                          own=pan.DFIRM_ID.to_numpy()[hits[1]] == dfirm[hits[0]]))
    h = h.sort_values(["i", "own", "pan"], ascending=[True, False, True], kind="stable")
    first = h.drop_duplicates("i")
    out = np.full(n, -1, dtype=np.int64)
    out[first.i.to_numpy()] = first.row.to_numpy()
    multi = count > 1
    by_dfirm = int(first.own.to_numpy()[multi[first.i.to_numpy()]].sum())
    return out, count, by_dfirm


def eff_dates(pan) -> np.ndarray:
    """Panel EFF_DATE as naive UTC datetime64, the form sample_points writes to
    meta_eff_date (its tz-aware EFF_DATE goes through .values)."""
    if "EFF_DATE" not in pan:
        return np.full(len(pan), np.datetime64("NaT", "ms"))
    return pd.to_datetime(pan.EFF_DATE, utc=True).dt.tz_localize(None).to_numpy()


# =========================================================================
def summarise(df: pd.DataFrame, res: pd.DataFrame, src_info: dict, near: dict,
              radius: float) -> None:
    pos = df.label.to_numpy() == 1
    n = len(df)
    cnt = res.meta_firm_pan_n.to_numpy()
    print(f"\n  panels: {int((cnt == 1).sum()):,} points in exactly one, "
          f"{int((cnt == 0).sum()):,} in none, {int((cnt > 1).sum()):,} in more than one "
          f"(max {int(cnt.max()) if n else 0})")
    if "meta_eff_date" in df:
        mine = pd.to_datetime(res.meta_firm_pan_eff_date)
        theirs = pd.to_datetime(df.meta_eff_date)
        both = mine.notna() & theirs.notna()
        same = int((mine[both] == theirs[both]).sum())
        print(f"  meta_firm_pan_eff_date equals the dataset's meta_eff_date for "
              f"{same:,}/{int(both.sum()):,} points; sample_points.attach_panel_meta picks "
              f"the panel by the same rule (decision 5), so any difference means the dataset "
              f"was built by an older sampler (v1 took the first of several panels) or "
              f"from another pan file")

    s = src_info
    print(f"  labelling polygon of {s['positives']:,} positives: {s['from_column']:,} from "
          f"meta_src_row_id, {s['key_unique']:,} by unique (dfirm_id, meta_fld_ar_id), "
          f"{s['key_ambiguous_resolved']}/{s['key_ambiguous']} ambiguous keys by "
          f"point-in-polygon, {s['key_not_in_index']} keys not in the index, "
          f"{s['no_key']} without a key")
    if s["column_key_mismatch"]:
        print(f"  WARNING: {s['column_key_mismatch']} positives whose meta_src_row_id "
              f"polygon has a different FLD_AR_ID from meta_fld_ar_id")
    has_src = res.meta_edge_src_row_id.notna().to_numpy()
    cov = near["covered"]
    print(f"  positives covered by their labelling polygon: {int(cov[pos].sum()):,}"
          f"/{int((pos & has_src).sum()):,} with a polygon"
          f"{'' if cov[pos & has_src].all() else '   <-- NOT ALL'}")
    if near.get("swapped") is not None:
        print(f"  positives whose plain nearest search found another polygon at the same "
              f"distance 0 (adjacent or overlapping SFHA): {near['swapped']:,}")

    d = res.meta_near_dist_m.to_numpy(float)
    found = np.isfinite(d)
    print(f"  nearest SFHA polygon found within {radius:.0f} m for {int(found.sum()):,}"
          f"/{n:,} points; zones: " +
          ", ".join(f"{z} {c:,}" for z, c in res.meta_near_fld_zone.value_counts().items()))
    for lab, name in ((1, "positives"), (0, "negatives")):
        m = (df.label.to_numpy() == lab) & found
        if m.any():
            q = np.percentile(d[m], [0, 50, 100])
            print(f"    {name:<9} distance min/median/max {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} m")
    if "meta_dist_to_sfha_m" in df:
        ref = df.meta_dist_to_sfha_m.to_numpy(float)
        both = found & np.isfinite(ref)
        diff = np.abs(d[both] - ref[both])
        print(f"  check against the dataset's meta_dist_to_sfha_m "
              f"(sample_points.distances_to_sfha): {int(both.sum()):,} points, max |diff| "
              f"{diff.max() if both.any() else float('nan'):.2e} m, "
              f"{int((diff > 1e-6).sum())} differ by more than 1e-6 m")

    e = res.meta_edge_dist_m.to_numpy(float)
    for dom, g in df[pos].groupby("domain"):
        v = e[g.index.to_numpy()]
        v = v[np.isfinite(v)]
        if len(v):
            q = np.percentile(v, [10, 50, 90])
            print(f"  meta_edge_dist_m [{dom}] n={len(v):,}: p10 {q[0]:.1f}, median "
                  f"{q[1]:.1f}, p90 {q[2]:.1f}, max {v.max():.1f} m")


def build(a) -> int:
    t0 = time.time()
    if a.out.exists() and not a.overwrite:
        print(f"{a.out} exists -- skipping (use --overwrite)")
        return 0
    for p in (a.dataset, a.idx, a.haz, a.panels):
        if not Path(p).exists():
            sys.exit(f"missing input {p}")

    df, xy, how = load_points(a.dataset, a.limit)
    print(f"[{a.state}] {len(df):,} points from {a.dataset}"
          f"{f' (random {a.limit:,}, seed {LIMIT_SEED})' if a.limit else ''}; xy: {how}",
          flush=True)
    idx = load_index(a.idx)
    haz = HazReader(a.haz)
    if haz.n_rows != len(idx):
        sys.exit(f"{a.haz} has {haz.n_rows:,} rows, {a.idx} {len(idx):,}: not the same prep")
    sfha = idx[idx.SFHA_TF == "T"]

    src, src_info = resolve_label_polygons(df, xy, idx, haz)
    print(f"  resolved labelling polygons ({time.time() - t0:.0f}s)", flush=True)

    near = nearest_and_edges(haz, sfha, xy, src, a.radius)
    pos = df.label.to_numpy() == 1
    use_src = pos & near["covered"]
    near["swapped"] = int((use_src & (near["near_r"] != src)).sum())
    near_r = np.where(use_src, src, near["near_r"])
    near_d = np.where(use_src, 0.0, near["near_d"])
    found = near_r >= 0

    attrs = idx.set_index("row_id")
    nr = pd.array(np.where(found, near_r, 0), dtype="Int64")
    nr[~found] = pd.NA
    er = pd.array(np.where(src >= 0, src, 0), dtype="Int64")
    er[src < 0] = pd.NA

    pan, fixed = load_panels(a.panels)
    pan_row, pan_n, by_dfirm = panel_of(pan, xy, df.dfirm_id.to_numpy().astype(object))
    in_pan = pan_row >= 0
    print(f"  {len(pan):,} panels ({int(fixed.sum())} repaired after reprojection); "
          f"{by_dfirm:,} of {int((pan_n > 1).sum()):,} multi-panel points took their own "
          f"DFIRM's panel", flush=True)

    res = pd.DataFrame({
        "point_id": df.point_id.to_numpy(),
        "meta_firm_pan": pd.array(np.where(in_pan, pan.FIRM_PAN.to_numpy()[pan_row], None),
                                  dtype="string"),
        "meta_firm_pan_n": pan_n.astype(np.int16),
        "meta_firm_pan_eff_date": np.where(in_pan, eff_dates(pan)[pan_row],
                                           np.datetime64("NaT", "ms")),
        "meta_near_src_row_id": nr,
        "meta_near_fld_ar_id": pd.array(np.where(
            found, attrs.FLD_AR_ID.reindex(np.where(found, near_r, -1)).to_numpy(), None),
            dtype="string"),
        "meta_near_fld_zone": pd.array(np.where(
            found, attrs.FLD_ZONE.reindex(np.where(found, near_r, -1)).to_numpy(), None),
            dtype="string"),
        "meta_near_dist_m": np.where(found, near_d, np.nan),
        "meta_edge_src_row_id": er,
        "meta_edge_dist_m": np.where(pos, near["edge"], np.nan),
    })
    summarise(df, res, src_info, near, a.radius)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(res, preserve_index=False)
    md = dict(t.schema.metadata or {})
    md[b"point_extras"] = json.dumps(dict(
        state=a.state, dataset=str(a.dataset), idx=str(a.idx), haz=str(a.haz),
        panels=str(a.panels), radius_m=a.radius, limit=a.limit, xy=how,
        labelling_polygon=src_info, written=time.strftime("%Y-%m-%d %H:%M:%S")),
        sort_keys=True).encode()
    pq.write_table(t.replace_schema_metadata(md), a.out)
    print(f"\n[{a.state}] wrote {len(res):,} rows x {res.shape[1]} columns -> {a.out}  "
          f"({time.time() - t0:.0f}s)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--idx", type=Path, help="prep/idx_<ST>.parquet")
    ap.add_argument("--haz", type=Path, help="prep/haz_<ST>_5070.parquet")
    ap.add_argument("--panels", type=Path, help="nfhl_states/pan_<ST>.parquet")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--radius", type=float, default=RADIUS_M,
                    help="nearest-SFHA search radius in metres (decision 1)")
    ap.add_argument("--limit", type=int, help="random subset of N points, for smoke tests")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    d = default_paths(a.state)
    if a.limit is not None and a.out is None:
        sys.exit("--limit writes a partial sidecar: give an explicit --out")
    for k, v in d.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return build(a)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
