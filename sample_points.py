#!/usr/bin/env python3
"""
sample_points.py -- NFHL flood-zone polygons -> a point-level supervised dataset.

The experiment needs two datasets built by an IDENTICAL protocol, differing only
in the provenance of the polygons that produced them:

    domain A   positives drawn inside Zone A  polygons (approximate study, no BFE)
    domain AE  positives drawn inside Zone AE polygons (detailed study, BFE)

Everything downstream -- OODSelect, the A->AE transfer gap, the explanation -- is
only meaningful if the two sides were built the same way. So every choice below is
applied symmetrically, recorded, and where it is contestable, exposed as a flag
rather than buried in a preprocessing step.

WHAT THIS SCRIPT DECIDES, AND WHY
=================================

1. AREA-PROPORTIONAL POSITIVES.  A fixed number of points per polygon would
   silently reweight the map: measured on Iowa, Zone A polygons average 1.42x the
   size of Zone AE, so the A share moves from 76.7% by count to 82.3% by area.
   Polygons are drawn with replacement in proportion to area and one uniform point
   placed in each -- exactly a uniform draw over their union -- with a per-polygon
   cap so a few enormous polygons cannot dominate a stratum.

2. SEPARATION IS A DE-DUPLICATION FLOOR, NOT AN AUTOCORRELATION FIX.  The first
   draft used 1 km and it was badly binding, because these zones are ribbons:
   measured on Vermont, Zone A is 277 km2 with 5,103 km of perimeter, a mean width
   of 109 m. The ceiling is corridor length, not area, and 1 km caps the entire
   state at ~1,400 points. The default is 200 m, which is what the covariates
   require -- MERIT Hydro 90 m, NLCD 30 m, 3DEP 10 m -- so every point sits in a
   distinct cell of every raster with margin.

   Spatial autocorrelation reaches far beyond any usable spacing, so it is handled
   where it belongs: `block_id` is a 10 km grid cell and EVERY split -- train/test,
   CV fold, OODSelect subset -- must be made at block level, never at point level
   (Roberts et al. 2017). A random point-split AUC here would reproduce the >0.95
   numbers the flood-susceptibility literature is criticised for.

3. TWO NEGATIVE PROTOCOLS, BOTH PROVENANCE-BLIND.
     band     a distance band outside every SFHA, inside the same map panel
     shadedX  inside Zone X '0.2 PCT ANNUAL CHANCE' polygons
   Tehrany & Jones (2017) found a 25-point non-monotonic AUC swing from the
   sampling choice alone on one flood footprint, and the labeling literature puts
   negative sampling at 0.13-0.24 AUC, second only to label definition. So the
   choice is not a hidden default: both are produced, tagged in `neg_protocol`, and
   the difference between them is a result rather than an assumption.

4. THE TWO DOMAINS ARE FORCED TO MATCH EXACTLY.  A transfer gap between unequal
   datasets is uninterpretable. Each stratum is trimmed so n_pos(A) == n_pos(AE)
   and n_neg(A, p) == n_neg(AE, p) per protocol. The first draft skipped this and
   the global separation pass ate 36% of AE positives against 5% of A -- an
   artefact that would have looked exactly like a finding.

5. LEAKAGE IS STRUCTURAL, NOT DISCIPLINARY.  Every column encoding how the map was
   made carries the `meta_` prefix and must never enter a model. This matters more
   than it looks. Across the ten states STUDY_TYP alone predicts Zone A vs AE at
   68.8% against a 56.4% base rate, and several of its levels are near-deterministic:

       SFHA WITH UNPUBLISHED BFE      8070 A /     8 AE   (99.9% A)
       SFHAS WITH LOW FLOOD RISK     42580 A /   153 AE   (99.6% A)
       SFHA WITH BFE AND FLOODWAY       42 A / 14105 AE   ( 0.3% A)

   `meta_dist_to_sfha_m` is the same category -- it is the distance to the thing
   being predicted. Recorded for diagnostics, excluded from features.

6. DFIRM_ID IS THE STRATUM, NOT THE COUNTY.  Provenance belongs to the map study.
   99.86% of rows sit in countywide DFIRMs (id ends in a letter), so DFIRM_ID[:5]
   is the county FIPS for almost all and `county_fips` is filled for those; the
   0.14% community-based DFIRMs keep a null county and remain usable.

7. STUDY REGIME IS CARRIED THROUGH.  Zone A does not mean the same thing
   everywhere. In Iowa 84% of Zone A carries a Base Level Engineering risk class
   and 50% of Zone AE is REDELINEATION; Vermont's STUDY_TYP is 100% unpopulated;
   Florida and Texas are ~78% unpopulated with the classic reading. Pooling those
   merges three mechanisms under one label, so `meta_study_regime` keeps them apart.

MEMORY
======
The per-state parquets were written as a single row group -- Florida is 5.2 GB in
one allocation, and its geometry decompresses to ~5.7 GB of WKB before geopandas
turns it into objects at 2-4x that again. Reading one whole is the same shape as
the segfault that killed the first subset attempt.

So `--prep` streams each state ONCE and writes two things:

    prep/idx_{ST}.parquet        attributes + area_m2 + bounds, no geometry
    prep/haz_{ST}_5070.parquet   geometry reprojected to EPSG:5070, 10k row groups

Every later run reads the index for all the allocation arithmetic and pulls
geometry only for the polygons it actually samples, one row group at a time. Peak
memory drops to a few hundred MB, and -- since the point of this project is to
re-run with different sampling parameters -- the cost is paid once.

USAGE
=====
    python sample_points.py --prep               # one-time, all states (do this first)
    python sample_points.py VT                   # build one state
    python sample_points.py                      # build all ten
    python sample_points.py --per-domain 40000
    python sample_points.py --neg-ratio 2        # build up to 1:2; 1:1 is a subset
    python sample_points.py --neg-ratio 2 --enforce-ratio   # exact, smaller
    python sample_points.py --audit VT           # verify invariants
    python sample_points.py --report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import geopandas as gpd
import shapely
from shapely import STRtree
from pyproj import Transformer

IN_DIR = Path("data/interim/nfhl_states")
OUT_DIR = Path("data/interim/points")
PREP_DIR = Path("data/interim/prep")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]

CRS_SRC = 4269      # NAD83 geographic, as NFHL ships
CRS_WORK = 5070     # CONUS Albers Equal Area, metres

NODATA = -9999.0    # FEMA's null sentinel in STATIC_BFE and DEPTH
RG_ROWS = 10_000    # rows per row group in the repack

ATTRS = ["DFIRM_ID", "FLD_AR_ID", "STUDY_TYP", "FLD_ZONE", "ZONE_SUBTY",
         "SFHA_TF", "STATIC_BFE", "DEPTH", "SOURCE_CIT"]

DEFAULTS = dict(
    per_domain=20000,     # positives per domain (A, AE) per state
    neg_ratio=2.0,        # MAXIMUM negatives per positive, per protocol; see note 8
    min_sep_m=200.0,      # de-duplication floor; see note 2
    block_km=10.0,        # spatial CV block edge
    poly_cap_frac=0.02,   # no polygon may supply more than this share of a domain
    band_min_m=500.0,     # negatives: nearest SFHA at least this far
    band_max_m=5000.0,    # ... and at most this far
    oversample=3.0,       # candidate multiplier before separation thinning
    seed=20260918,
)

META_PREFIX = "meta_"   # never a feature


# =========================================================================
# small helpers
# =========================================================================
def study_regime(s: pd.Series) -> pd.Series:
    """Collapse mapping-partner STUDY_TYP text into comparable mechanisms."""
    t = s.fillna("").astype(str).str.strip().str.upper()
    out = pd.Series("OTHER", index=t.index, dtype=object)
    out[t.eq("NP") | t.eq("")] = "NOT_POPULATED"
    out[t.str.contains("FLOOD RISK", na=False)] = "BLE_RISKCLASS"
    out[t.str.contains("REDELINEATION", na=False)] = "REDELINEATION"
    out[t.str.contains("DIGITAL CONVERSION", na=False)] = "DIGITAL_CONVERSION"
    out[t.str.contains("WITHOUT BFE|UNPUBLISHED BFE", na=False)] = "NO_BFE"
    out[t.str.contains("WITH BFE", na=False)] = "BFE_DETAIL"
    out[t.str.contains("BLE AVAILABLE", na=False)] = "BLE_UNPUBLISHED"
    return out


def clean_sentinel(a) -> np.ndarray:
    v = np.array(pd.to_numeric(pd.Series(a), errors="coerce"), dtype=float, copy=True)
    v[np.isclose(v, NODATA)] = np.nan
    return v


def _reproject(geoms: np.ndarray, tr) -> np.ndarray:
    return shapely.transform(
        geoms, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))


# =========================================================================
# prep: stream once, write an index and a reprojected repack
# =========================================================================
def prep_state(state: str, force: bool = False) -> None:
    src = IN_DIR / f"haz_{state}.parquet"
    idx_p = PREP_DIR / f"idx_{state}.parquet"
    geo_p = PREP_DIR / f"haz_{state}_5070.parquet"
    if not src.exists():
        print(f"[{state}] missing {src} -- skipping")
        return
    if idx_p.exists() and geo_p.exists() and not force:
        print(f"[{state}] prepped already -- skipping (use --force)")
        return

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    print(f"[{state}] prep: {total:,} polygons from "
          f"{src.stat().st_size/1e9:.2f} GB", flush=True)

    tr = Transformer.from_crs(CRS_SRC, CRS_WORK, always_xy=True)
    writer = None
    parts = []
    seen = 0
    n_fixed = 0

    for batch in pf.iter_batches(batch_size=RG_ROWS, columns=ATTRS + ["geometry"]):
        g = shapely.from_wkb(np.asarray(batch.column("geometry")))
        g = _reproject(g, tr)

        bad = ~shapely.is_valid(g)
        if bad.any():
            # A few NFHL polygons fail OGC validity (self-touching rings from
            # digitising). contains_xy on an invalid polygon is undefined, so
            # repair rather than drop -- dropping biases toward simple shapes.
            g[bad] = shapely.make_valid(g[bad])
            n_fixed += int(bad.sum())

        df = batch.select(ATTRS).to_pandas()
        df["row_id"] = np.arange(seen, seen + len(df), dtype=np.int64)
        df["area_m2"] = shapely.area(g)
        df[["minx", "miny", "maxx", "maxy"]] = shapely.bounds(g)
        parts.append(df)

        out = pa.table({"row_id": pa.array(df["row_id"].values),
                        "geometry": pa.array(shapely.to_wkb(g))})
        if writer is None:
            writer = pq.ParquetWriter(geo_p, out.schema, compression="zstd")
        writer.write_table(out, row_group_size=RG_ROWS)

        seen += len(df)
        if seen % (RG_ROWS * 20) == 0:
            print(f"    {seen:,}/{total:,}  ({time.time()-t0:.0f}s)", flush=True)

    if writer is not None:
        writer.close()

    idx = pd.concat(parts, ignore_index=True)
    idx["study_regime"] = study_regime(idx["STUDY_TYP"])
    idx["STATIC_BFE"] = clean_sentinel(idx["STATIC_BFE"])
    if "DEPTH" in idx:
        idx["DEPTH"] = clean_sentinel(idx["DEPTH"])
    idx.to_parquet(idx_p)

    if n_fixed:
        print(f"    repaired {n_fixed:,} invalid polygons")
    print(f"[{state}] prep done: index {idx_p.stat().st_size/1e6:.0f} MB, "
          f"repack {geo_p.stat().st_size/1e9:.2f} GB  ({time.time()-t0:.0f}s)")


def load_index(state: str) -> pd.DataFrame:
    p = PREP_DIR / f"idx_{state}.parquet"
    if not p.exists():
        sys.exit(f"{p} missing -- run:  python sample_points.py --prep {state}")
    return pd.read_parquet(p)


def read_geoms(state: str, row_ids) -> np.ndarray:
    """Geometry for the given row_ids, reading only the row groups they fall in."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.size == 0:
        return np.empty(0, dtype=object)
    pf = pq.ParquetFile(PREP_DIR / f"haz_{state}_5070.parquet")

    want = pd.unique(row_ids)
    out: dict[int, object] = {}
    for gi in np.unique(want // RG_ROWS):
        if gi >= pf.num_row_groups:
            continue
        t = pf.read_row_group(int(gi), columns=["row_id", "geometry"])
        rid = t.column("row_id").to_numpy()
        keep = np.isin(rid, want)
        if not keep.any():
            continue
        geo = shapely.from_wkb(np.asarray(t.column("geometry"))[keep])
        for r, gm in zip(rid[keep], geo):
            out[int(r)] = gm
    return np.array([out.get(int(r)) for r in row_ids], dtype=object)


# =========================================================================
# geometry helpers
# =========================================================================
def grid_thin(x: np.ndarray, y: np.ndarray, min_sep: float,
              rng: np.random.Generator) -> np.ndarray:
    """Greedy minimum-separation thinning on a hash grid. Returns kept indices.

    Points are visited in random order and kept only if no already-kept point lies
    within min_sep. With a cell edge of min_sep, checking the point's own cell plus
    its eight neighbours is sufficient, making this O(n) instead of the O(n^2) a
    pairwise distance matrix would cost -- which matters because Florida produces
    on the order of a million candidates.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if min_sep <= 0:
        return np.arange(n, dtype=np.int64)

    order = rng.permutation(n)
    cx = np.floor(x / min_sep).astype(np.int64)
    cy = np.floor(y / min_sep).astype(np.int64)

    occupied: dict[tuple[int, int], list[int]] = {}
    keep: list[int] = []
    s2 = min_sep * min_sep
    neigh = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]

    for i in order:
        a0, b0 = cx[i], cy[i]
        xi, yi = x[i], y[i]
        clash = False
        for da, db in neigh:
            for j in occupied.get((a0 + da, b0 + db), ()):
                dx = xi - x[j]
                dy = yi - y[j]
                if dx * dx + dy * dy < s2:
                    clash = True
                    break
            if clash:
                break
        if not clash:
            keep.append(i)
            occupied.setdefault((a0, b0), []).append(i)

    return np.asarray(keep, dtype=np.int64)


def sample_in_polygons(geoms: np.ndarray, weights, n: int, cap: int,
                       rng: np.random.Generator, max_rounds: int = 30):
    """Area-proportional uniform points inside polygons, with a per-polygon cap.

    Returns (x, y, index-into-geoms). Polygons are drawn with replacement in
    proportion to area and one uniform point placed in each, which is exactly a
    uniform draw over their union, so the work scales with the number of points
    requested rather than the number of polygons.
    """
    if n <= 0 or len(geoms) == 0:
        return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))

    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))

    quota = np.minimum(rng.multinomial(n, w / w.sum()), cap)
    deficit = n - int(quota.sum())
    if deficit > 0:
        room = cap - quota
        p = np.where(room > 0, w, 0.0)
        if p.sum() > 0:
            quota = quota + np.minimum(rng.multinomial(deficit, p / p.sum()), room)

    live = np.nonzero(quota > 0)[0]
    if live.size == 0:
        return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))

    g_live = geoms[live]
    b_live = shapely.bounds(g_live)
    need = quota[live].astype(np.int64)

    xs, ys, pidx = [], [], []
    for _ in range(max_rounds):
        hot = np.nonzero(need > 0)[0]
        if hot.size == 0:
            break
        sel = np.repeat(hot, np.maximum(need[hot] * 2, 1))
        b = b_live[sel]
        x = rng.uniform(b[:, 0], b[:, 2])
        y = rng.uniform(b[:, 1], b[:, 3])
        ok = shapely.contains_xy(g_live[sel], x, y)
        if not ok.any():
            break
        x, y, sel = x[ok], y[ok], sel[ok]

        order = np.argsort(sel, kind="stable")
        x, y, sel = x[order], y[order], sel[order]
        uniq, start, counts = np.unique(sel, return_index=True, return_counts=True)
        rank = np.arange(len(sel)) - np.repeat(start, counts)
        take = rank < np.repeat(need[uniq], counts)
        np.subtract.at(need, uniq, np.minimum(counts, need[uniq]))

        xs.append(x[take]); ys.append(y[take]); pidx.append(sel[take])

    if not xs:
        return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
    return (np.concatenate(xs), np.concatenate(ys), live[np.concatenate(pidx)])


def pick_polygons(idx: pd.DataFrame, n_points: int, cap: int,
                  rng: np.random.Generator, factor: int = 60,
                  budget: int = 60_000) -> pd.DataFrame:
    """Area-weighted subset of candidate polygons, so geometry stays bounded.

    With a per-polygon cap of `cap`, only ceil(n_points/cap) polygons are strictly
    required, but drawing that few would let a handful of polygons stand in for a
    whole state. `factor` widens the draw so the area distribution is represented
    faithfully; `budget` stops Florida's 147k Zone A polygons from being loaded at
    once. The draw is area-weighted, so it is the same distribution the points are
    ultimately sampled from.
    """
    need = min(int(np.ceil(n_points / max(cap, 1))) * factor, budget)
    if len(idx) <= need:
        return idx
    w = idx["area_m2"].to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return idx.head(need)
    take = np.unique(rng.choice(len(idx), size=min(need, len(idx)),
                                replace=True, p=w / w.sum()))
    return idx.iloc[take]


def distances_to_sfha(state: str, pts: np.ndarray, sfha_idx: pd.DataFrame,
                      radius: float, tile_m: float = 50_000.0) -> np.ndarray:
    """Exact distance from each point to the nearest SFHA polygon.

    Points are handled in spatial tiles, and for each tile only the SFHA polygons
    whose bounding box reaches into it are loaded. So this is exact, never
    materialises the state's full hazard layer, and -- because it asks the tree for
    the single nearest geometry rather than every geometry within the band -- costs
    O(n log m) instead of the tens of millions of candidate pairs a 5 km `dwithin`
    query produces against ribbon-shaped flood polygons.
    """
    n = len(pts)
    if n == 0:
        return np.empty(0)
    if len(sfha_idx) == 0:
        return np.full(n, np.inf)

    px = shapely.get_x(pts)
    py = shapely.get_y(pts)
    d = np.full(n, np.inf)

    smin_x = sfha_idx.minx.to_numpy(); smax_x = sfha_idx.maxx.to_numpy()
    smin_y = sfha_idx.miny.to_numpy(); smax_y = sfha_idx.maxy.to_numpy()
    rid_all = sfha_idx["row_id"].to_numpy()

    tx = np.floor(px / tile_m).astype(np.int64)
    ty = np.floor(py / tile_m).astype(np.int64)
    for key in {*zip(tx.tolist(), ty.tolist())}:
        m = (tx == key[0]) & (ty == key[1])
        x0, x1 = px[m].min() - radius, px[m].max() + radius
        y0, y1 = py[m].min() - radius, py[m].max() + radius
        near = (smax_x >= x0) & (smin_x <= x1) & (smax_y >= y0) & (smin_y <= y1)
        if not near.any():
            continue
        geoms = read_geoms(state, rid_all[near])
        ok = np.array([g is not None for g in geoms])
        if not ok.any():
            continue
        geoms = geoms[ok]
        sub = pts[m]
        j = STRtree(geoms).nearest(sub)
        d[np.nonzero(m)[0]] = shapely.distance(sub, geoms[j])
    return d


# =========================================================================
# per-state build
# =========================================================================
def _attr_frame(att: pd.DataFrame, x, y, dist) -> pd.DataFrame:
    return pd.DataFrame({
        "dfirm_id": att["DFIRM_ID"].values,
        "x": x, "y": y,
        "meta_fld_zone": att["FLD_ZONE"].values,
        "meta_zone_subty": att["ZONE_SUBTY"].values,
        "meta_study_typ": att["STUDY_TYP"].values,
        "meta_study_regime": att["study_regime"].values,
        "meta_static_bfe": att["STATIC_BFE"].values,
        "meta_depth": att["DEPTH"].values if "DEPTH" in att else np.full(len(att), np.nan),
        "meta_source_cit": att["SOURCE_CIT"].values,
        "meta_fld_ar_id": att["FLD_AR_ID"].values,
        "meta_poly_area_km2": att["area_m2"].values / 1e6,
        "meta_dist_to_sfha_m": dist,
    })


def _blank_frame(dfirm, x, y, dist) -> pd.DataFrame:
    n = len(x)
    na = lambda: pd.array([pd.NA] * n, dtype="string")
    return pd.DataFrame({
        "dfirm_id": dfirm, "x": x, "y": y,
        "meta_fld_zone": na(), "meta_zone_subty": na(), "meta_study_typ": na(),
        "meta_study_regime": na(), "meta_static_bfe": np.full(n, np.nan),
        "meta_depth": np.full(n, np.nan), "meta_source_cit": na(),
        "meta_fld_ar_id": na(), "meta_poly_area_km2": np.full(n, np.nan),
        "meta_dist_to_sfha_m": dist,
    })


MAX_RINGS = 1500          # distinct polygons buffered for the negative band
RING_SIMPLIFY_M = 100.0   # buffer is a candidate region; exactness is the distance filter


def sample_band_negatives(state, src_idx, src_geoms, sfha_idx, pan_tree, n, cfg, rng):
    """Negatives in a distance band outside every SFHA, inside the mapped area.

    The band is generated from the SAME polygons that produced the positives, so
    the protocol never reads FLD_ZONE and is identical on both sides -- which is
    the whole point.

    Two things make this affordable. Only DISTINCT source polygons are buffered:
    the positives repeat polygons many times over, and buffering 8,000 duplicated
    ribbons at 5 km took longer than the rest of the pipeline combined. And the
    ring is built from geometry simplified to 100 m with coarse corner segments,
    because the buffer only has to bound a candidate region -- the true distance
    filter below is what decides membership, so the ring's own precision is
    irrelevant. It is widened by the simplification tolerance so nothing valid is
    excluded.
    """
    if n <= 0 or len(src_geoms) == 0:
        return None

    bmin, bmax = cfg["band_min_m"], cfg["band_max_m"]
    want = int(n * cfg["oversample"] * 2)

    keep_i = np.arange(len(src_geoms))
    if len(keep_i) > MAX_RINGS:
        keep_i = rng.choice(keep_i, MAX_RINGS, replace=False)
    base = shapely.simplify(src_geoms[keep_i], RING_SIMPLIFY_M)
    ring = shapely.buffer(base, bmax + RING_SIMPLIFY_M, quad_segs=2)
    cap = max(4, int(np.ceil(want / max(len(ring) // 4, 1))))

    x, y, ri = sample_in_polygons(ring, shapely.area(ring), want, cap, rng)
    if len(x) == 0:
        return None

    pts = shapely.points(x, y)
    d = distances_to_sfha(state, pts, sfha_idx, bmax)
    ok = (d >= bmin) & (d <= bmax)
    if pan_tree is not None:
        # Outside a mapped panel we do not know the area was ever assessed, so
        # such a point is not evidence of absence.
        inside = np.zeros(len(pts), dtype=bool)
        hits = pan_tree.query(pts, predicate="within")
        if hits.size:
            inside[np.unique(hits[0])] = True
        ok &= inside
    if not ok.any():
        return None

    x, y, d, ri = x[ok], y[ok], d[ok], ri[ok]
    keep = grid_thin(x, y, cfg["min_sep_m"], rng)[:n]
    x, y, d, ri = x[keep], y[keep], d[keep], ri[keep]
    return _blank_frame(src_idx["DFIRM_ID"].to_numpy()[keep_i[ri]], x, y, d)


def rank_negatives(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Give every negative a position in a random ordering within its stratum.

    This is what turns the negative ratio into a post hoc filter. Ranks are
    assigned once, in random order, within each (domain, protocol) cell; a
    downstream analysis selecting `neg_rank < r * n_pos` gets ratio r, and the
    samples nest, so the 1:1 set is contained in the 1:2 set. Without this the
    two ratios would differ in WHICH negatives they contain as well as how many,
    and a comparison between them would not isolate the ratio.
    """
    df = df.copy()
    df["neg_rank"] = -1
    neg = df.label == 0
    if not neg.any():
        return df
    key = [df["domain"], df["neg_protocol"].fillna("-")]
    order = rng.permutation(len(df))
    df["_o"] = np.argsort(np.argsort(order))
    df.loc[neg, "neg_rank"] = (df[neg]
                               .groupby([df.loc[neg, "domain"],
                                         df.loc[neg, "neg_protocol"].fillna("-")])["_o"]
                               .rank(method="first").astype(int) - 1)
    return df.drop(columns="_o")


def achieved_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Negatives actually obtained per positive, per stratum."""
    pos = (df[df.label == 1].groupby("domain").size())
    rows = []
    for (dom, prot), g in df[df.label == 0].groupby(
            ["domain", df.neg_protocol.fillna("-")]):
        n = int(pos.get(dom, 0))
        rows.append(dict(domain=dom, protocol=prot, pos=n, neg=len(g),
                         ratio=round(len(g) / max(n, 1), 2)))
    return pd.DataFrame(rows)


def enforce_ratio(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """Trim POSITIVES so every stratum hits the target ratio exactly.

    The alternative -- letting the ratio float with supply -- leaves it varying
    by a factor of ten across states, which entangles it with the state effect
    this design is built to hold fixed. Trimming costs sample size and buys a
    clean comparison; which of those matters more is a choice, so it is a flag.
    """
    keep = []
    for dom, g in df.groupby("domain"):
        npos = int((g.label == 1).sum())
        cap = npos
        for prot, gg in g[g.label == 0].groupby(g.neg_protocol.fillna("-")):
            cap = min(cap, int(np.floor(len(gg) / target)))
        pos = g[g.label == 1].iloc[:cap]
        keep.append(pos)
        for prot, gg in g[g.label == 0].groupby(g.neg_protocol.fillna("-")):
            keep.append(gg.nsmallest(int(round(cap * target)), "neg_rank"))
    return pd.concat(keep, ignore_index=True)


def balance_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Trim each stratum so the A and AE sides are exactly parallel."""
    df = df.copy()
    df["_p"] = df["neg_protocol"].fillna("-")
    sizes = df.groupby(["label", "_p", "domain"]).size().unstack("domain")
    if sizes.shape[1] < 2:
        return df.drop(columns="_p")
    target = sizes.min(axis=1)
    out = []
    for (lab, prot), sub in df.groupby(["label", "_p"]):
        t = int(target.get((lab, prot), 0))
        for _, g in sub.groupby("domain"):
            out.append(g.iloc[:t] if len(g) > t else g)
    res = pd.concat(out, ignore_index=True).drop(columns="_p")
    if len(res) != len(df):
        print(f"  balanced A/AE: {len(df):,} -> {len(res):,}")
    return res


def attach_panel_meta(df: pd.DataFrame, pan) -> pd.DataFrame:
    for c in ("meta_eff_date", "meta_panel_scale", "meta_panel_typ"):
        df[c] = pd.NA
    if pan is None or not len(pan):
        return df
    pts = gpd.GeoDataFrame({"_i": np.arange(len(df))},
                           geometry=gpd.points_from_xy(df.x, df.y), crs=CRS_WORK)
    cols = [c for c in ("EFF_DATE", "SCALE", "PANEL_TYP") if c in pan.columns]
    j = gpd.sjoin(pts, pan[cols + ["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    if "EFF_DATE" in cols:
        df["meta_eff_date"] = j["EFF_DATE"].values
    if "SCALE" in cols:
        df["meta_panel_scale"] = pd.to_numeric(j["SCALE"], errors="coerce").values
    if "PANEL_TYP" in cols:
        df["meta_panel_typ"] = j["PANEL_TYP"].values
    return df


def bfe_density_map(state: str, sfha_idx: pd.DataFrame) -> dict:
    """BFE line length per km2 of SFHA, per DFIRM.

    In Zone AE the elevations live in S_BFE as line features -- STATIC_BFE on the
    polygon is -9999 about 98% of the time -- so line length per unit area is a
    continuous measure of study detail. Strictly more informative than the binary
    A/AE split, and the natural thing to regress the transfer gap against.
    """
    p = IN_DIR / f"s_bfe_{state}.parquet"
    if not p.exists():
        return {}
    try:
        bfe = gpd.read_parquet(p).to_crs(CRS_WORK)
        km = bfe.assign(_l=bfe.geometry.length / 1000).groupby("DFIRM_ID")._l.sum()
        km2 = sfha_idx.groupby("DFIRM_ID").area_m2.sum() / 1e6
        return (km / km2).replace([np.inf, -np.inf], np.nan).dropna().to_dict()
    except Exception as e:
        print(f"  BFE density skipped: {type(e).__name__}: {e}")
        return {}


def build_state(state: str, cfg: dict):
    t0 = time.time()
    idx = load_index(state)
    rng = np.random.default_rng(cfg["seed"] + sum(map(ord, state)))

    sfha_idx = idx[idx.SFHA_TF == "T"]
    pos_idx = sfha_idx[sfha_idx.FLD_ZONE.isin(["A", "AE"])]
    if len(pos_idx) == 0:
        print(f"[{state}] no A/AE polygons -- skipping")
        return None

    subty = idx["ZONE_SUBTY"].fillna("").astype(str).str.upper()
    shaded_idx = idx[(idx.FLD_ZONE == "X") & subty.str.contains("0.2 PCT")]

    pan_p = IN_DIR / f"pan_{state}.parquet"
    pan = gpd.read_parquet(pan_p).to_crs(CRS_WORK) if pan_p.exists() else None
    pan_tree = (STRtree(np.asarray(pan.geometry.values))
                if pan is not None and len(pan) else None)

    print(f"\n[{state}] A={int((pos_idx.FLD_ZONE=='A').sum()):,}  "
          f"AE={int((pos_idx.FLD_ZONE=='AE').sum()):,}  "
          f"shadedX={len(shaded_idx):,}  panels={0 if pan is None else len(pan):,}",
          flush=True)

    bfe = bfe_density_map(state, sfha_idx)
    if bfe:
        print(f"  BFE density for {len(bfe)} DFIRMs")

    frames = []
    for domain in ("A", "AE"):
        src_all = pos_idx[pos_idx.FLD_ZONE == domain]
        if len(src_all) == 0:
            print(f"  [{domain}] no polygons")
            continue

        n_target = cfg["per_domain"]
        cap = max(1, int(cfg["poly_cap_frac"] * n_target))
        over = cfg["oversample"]

        src = pick_polygons(src_all, int(n_target * over), int(cap * over), rng)
        geoms = read_geoms(state, src["row_id"].to_numpy())
        keepg = np.array([g is not None for g in geoms])
        src, geoms = src[keepg], geoms[keepg]
        if len(geoms) == 0:
            print(f"  [{domain}] no geometry read")
            continue

        x, y, pidx = sample_in_polygons(geoms, src["area_m2"].values,
                                        int(n_target * over), int(cap * over), rng)
        if len(x) == 0:
            print(f"  [{domain}] sampling produced nothing")
            continue
        drawn = len(x)
        keep = grid_thin(x, y, cfg["min_sep_m"], rng)[:n_target]
        x, y, pidx = x[keep], y[keep], pidx[keep]

        pos = _attr_frame(src.iloc[pidx], x, y, np.zeros(len(x)))
        pos["domain"], pos["label"], pos["neg_protocol"] = domain, 1, pd.NA
        frames.append(pos)
        sat = "" if len(x) >= n_target else "   <-- SUPPLY-LIMITED"
        print(f"  [{domain}] positives {len(x):,}/{n_target:,} from {len(src):,} "
              f"polygons, {src.iloc[pidx].DFIRM_ID.nunique()} DFIRMs "
              f"(thinning kept {100*len(keep)/drawn:.0f}%){sat}", flush=True)

        n_neg = int(len(x) * cfg["neg_ratio"])

        used = np.unique(pidx)
        band = sample_band_negatives(state, src.iloc[used].reset_index(drop=True),
                                     geoms[used], sfha_idx, pan_tree, n_neg, cfg, rng)
        if band is not None and len(band):
            band["domain"], band["label"], band["neg_protocol"] = domain, 0, "band"
            frames.append(band)
            print(f"  [{domain}] negatives band    {len(band):,}/{n_neg:,}", flush=True)

        # Zone X 0.2%-chance polygons in the same DFIRMs. These control for
        # 'was this place mapped at all', which the distance band does not.
        sx_all = shaded_idx[shaded_idx.DFIRM_ID.isin(set(src.DFIRM_ID))]
        if len(sx_all):
            sx = pick_polygons(sx_all, int(n_neg * over), int(cap * over), rng)
            sg = read_geoms(state, sx["row_id"].to_numpy())
            k2 = np.array([g is not None for g in sg])
            sx, sg = sx[k2], sg[k2]
            xs, ys, si = sample_in_polygons(sg, sx["area_m2"].values,
                                            int(n_neg * over), int(cap * over), rng)
            if len(xs):
                k = grid_thin(xs, ys, cfg["min_sep_m"], rng)[:n_neg]
                xs, ys, si = xs[k], ys[k], si[k]
                d = distances_to_sfha(state, shapely.points(xs, ys),
                                      sfha_idx, cfg["band_max_m"])
                f = _attr_frame(sx.iloc[si], xs, ys, d)
                f["domain"], f["label"], f["neg_protocol"] = domain, 0, "shadedX"
                frames.append(f)
                print(f"  [{domain}] negatives shadedX {len(xs):,}/{n_neg:,}", flush=True)
        else:
            print(f"  [{domain}] negatives shadedX none in these DFIRMs")

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    # One separation pass over the whole state, so a positive and a negative cannot
    # end up sharing a raster cell across strata. Half the within-stratum floor:
    # cross-label pairs need only be distinguishable, not independent.
    keep = grid_thin(df.x.values, df.y.values, cfg["min_sep_m"] * 0.5, rng)
    if len(keep) < len(df):
        print(f"  cross-stratum separation dropped {len(df)-len(keep):,}")
    df = df.iloc[np.sort(keep)].reset_index(drop=True)

    df = rank_negatives(df, rng)
    if cfg.get("enforce_ratio"):
        n0 = len(df)
        df = enforce_ratio(df, cfg["neg_ratio"])
        print(f"  enforced ratio {cfg['neg_ratio']:g}:1 exactly: {n0:,} -> {len(df):,}")
    df = balance_domains(df)

    ach = achieved_ratios(df)
    short = ach[ach.ratio < cfg["neg_ratio"] - 0.01]
    print("  achieved negatives per positive:")
    for r in ach.itertuples():
        flag = "   <-- SUPPLY-CAPPED" if r.ratio < cfg["neg_ratio"] - 0.01 else ""
        print(f"    {r.domain:<3} {r.protocol:<8} {r.neg:>7,}/{r.pos:<7,} "
              f"= {r.ratio:.2f}{flag}")
    if len(short):
        print(f"  {len(short)}/{len(ach)} strata below the requested "
              f"{cfg['neg_ratio']:g}:1 -- report the achieved ratio, not the request")

    df = attach_panel_meta(df, pan)

    df["state"] = state
    df["meta_bfe_km_per_km2"] = df["dfirm_id"].map(bfe)
    tail = df["dfirm_id"].astype(str).str[-1]
    df["county_fips"] = np.where(tail.str.isalpha(), df["dfirm_id"].str[:5], None)

    # Spatial CV blocks. Every split downstream must be made on this column.
    bm = cfg["block_km"] * 1000.0
    df["block_id"] = (np.floor(df.x / bm).astype(np.int64).astype(str) + "_" +
                      np.floor(df.y / bm).astype(np.int64).astype(str))
    df["point_id"] = [f"{state}{i:07d}" for i in range(len(df))]

    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=CRS_WORK)
    ll = g.geometry.to_crs(CRS_SRC)
    g["lon"], g["lat"] = ll.x, ll.y
    g = g.drop(columns=["x", "y"])

    lead = ["point_id", "state", "county_fips", "dfirm_id", "block_id",
            "domain", "label", "neg_protocol", "lon", "lat"]
    g = g[lead + [c for c in g.columns if c not in lead and c != "geometry"] + ["geometry"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"points_{state}.parquet"
    g.to_parquet(out)
    print(f"[{state}] wrote {len(g):,} points, {g.block_id.nunique():,} blocks "
          f"-> {out}  ({time.time()-t0:.0f}s)")
    return g


# =========================================================================
def audit(states: list[str], cfg: dict) -> None:
    """Verify the invariants the design depends on, on data already written."""
    for s in states:
        f = OUT_DIR / f"points_{s}.parquet"
        if not f.exists():
            print(f"[{s}] not built")
            continue
        g = gpd.read_parquet(f)
        print(f"\n=== {s}: {len(g):,} points ===")
        print(f"  CRS {g.crs.to_epsg()} (want {CRS_WORK})")

        t = g.groupby(["domain", "label", g.neg_protocol.fillna("-")]).size()
        print("  strata:\n    " + t.to_string().replace("\n", "\n    "))
        if {"A", "AE"} <= set(g.domain.unique()):
            a, e = t.xs("A", level=0), t.xs("AE", level=0)
            ok = a.reindex(e.index).fillna(0).astype(int).equals(e.astype(int))
            print(f"  A/AE strata identical: {'YES' if ok else 'NO  <-- BROKEN'}")

        pts = shapely.points(g.geometry.x.values, g.geometry.y.values)
        nn = STRtree(pts).query(pts, predicate="dwithin",
                                distance=cfg["min_sep_m"] * 0.999)
        off = nn[0] != nn[1]
        strat = g.groupby(["domain", "label", g.neg_protocol.fillna("-")]).ngroup().values
        same = int((strat[nn[0]][off] == strat[nn[1]][off]).sum() // 2)
        print(f"  pairs closer than {cfg['min_sep_m']:.0f} m: {int(off.sum()//2):,} "
              f"({same:,} within one stratum -- must be 0)")

        pos = g[g.label == 1]
        print(f"  positives at distance 0 from SFHA: "
              f"{(pos.meta_dist_to_sfha_m == 0).mean()*100:.1f}% (want 100)")
        b = g[g.neg_protocol == "band"]
        if len(b):
            print(f"  band negatives distance range: {b.meta_dist_to_sfha_m.min():.0f}"
                  f"-{b.meta_dist_to_sfha_m.max():.0f} m "
                  f"(want {cfg['band_min_m']:.0f}-{cfg['band_max_m']:.0f})")
        sx = g[g.neg_protocol == "shadedX"]
        if len(sx):
            print(f"  shadedX negatives inside an SFHA: "
                  f"{int((sx.meta_dist_to_sfha_m <= 0).sum())} (must be 0)")

        print(f"  meta_ columns excluded from features: "
              f"{len([c for c in g.columns if c.startswith(META_PREFIX)])}")
        print(f"  blocks: {g.block_id.nunique():,} "
              f"(median {g.groupby('block_id').size().median():.0f} points/block)")
        reg = (g[g.label == 1].groupby("domain").meta_study_regime
               .value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0))
        print("  positive study regimes (%):\n    " +
              reg.to_string().replace("\n", "\n    "))


def report() -> None:
    files = sorted(OUT_DIR.glob("points_*.parquet"))
    if not files:
        print(f"nothing in {OUT_DIR}")
        return
    rows = []
    for f in files:
        g = pd.read_parquet(f, columns=["state", "domain", "label", "neg_protocol"])
        for (dom, lab, prot), sub in g.groupby(
                ["domain", "label", g.neg_protocol.fillna("-")], dropna=False):
            rows.append(dict(state=g.state.iloc[0], domain=dom, label=lab,
                             protocol=prot, n=len(sub)))
    t = pd.DataFrame(rows)
    print(t.pivot_table(index=["state", "domain"], columns=["label", "protocol"],
                        values="n", aggfunc="sum", fill_value=0).to_string())
    print(f"\ntotal points: {t.n.sum():,}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*")
    ap.add_argument("--prep", action="store_true", help="one-time repack + index")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--enforce-ratio", action="store_true",
                    help="trim positives so every stratum hits "
                         "--neg-ratio exactly (uniform but smaller)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    a = ap.parse_args(argv)

    cfg = {k: getattr(a, k) for k in DEFAULTS}
    cfg["enforce_ratio"] = a.enforce_ratio
    states = a.states or STATES
    bad = [s for s in states if s not in STATES]
    if bad:
        sys.exit(f"unknown states: {bad}")

    if a.report:
        report(); return 0
    if a.audit:
        audit(states, cfg); return 0
    if a.prep:
        for s in states:
            prep_state(s, force=a.force)
        return 0

    print("config:", {k: cfg[k] for k in sorted(cfg)})
    for s in states:
        out = OUT_DIR / f"points_{s}.parquet"
        if out.exists() and not a.force:
            print(f"[{s}] exists -- skipping (use --force to rebuild)")
            continue
        try:
            build_state(s, cfg)
        except Exception:
            import traceback
            print(f"[{s}] FAILED", file=sys.stderr)
            traceback.print_exc()

    print()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
