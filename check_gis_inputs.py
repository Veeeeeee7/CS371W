#!/usr/bin/env python3
"""
check_gis_inputs.py -- what the boundary analysis can read today, and whether it
is what it claims to be (audit brief Part D0).

The boundary analysis (docs/REGION-ANALYSIS-FLORIDA.md section 3.3) measures
Zone A edges against layers the models never saw. Before any of that is built,
this script checks the inputs it rests on and writes the answers to a JSON file
(for scripts) and a Markdown file (for the audit report), one row per check
with a verdict: ok, missing, or warn (works but needs a decision).

WHAT IT CHECKS, AND WHY THAT WAY
================================

1. haz_<ST>_5070.parquet carries only row_id + WKB geometry. It has no GeoParquet
   metadata, so its CRS is not recorded anywhere in the file: EPSG:5070 is
   implied by sample_points.prep_state, which reprojects NFHL's EPSG:4269 and
   writes plain WKB. The script says so, and then tests the claim instead of
   trusting it: the decoded bounds of a random sample of row groups must equal
   the idx bbox columns that prep_state computed from the same geometry, and lie
   inside the FIRM panels' extent projected to EPSG:5070 (degrees would not).
   Validity is tested on whole row groups (default 3 of 60, ~30,000 polygons);
   the file is 5.4 GB and is never read whole.
2. FLD_AR_ID and FLD_ZONE live in idx_<ST>.parquet, joined on row_id. The join
   is checked (same row count, same row_id sequence), and so is the key that
   v1 points must fall back on: FLD_AR_ID alone is not unique in Florida, so
   duplicates of (DFIRM_ID, FLD_AR_ID) are counted too.
3. --regions: every region positive's polygon must be found and must cover the
   point. The polygon comes from meta_src_row_id when it is set, else from
   (dfirm_id, meta_fld_ar_id) exactly as pull_point_extras.py and
   aotil/regions.py recover it. The point is taken from the dataset's EPSG:5070
   geometry by point_id when --dataset exists, else from lon/lat (EPSG:4269).
   Every sampling build reissues point_ids from FL0000000, so a matching id
   proves nothing: the regions' lon/lat are copies of the dataset's and must
   equal them to 1e-9 deg (the tolerance aotil/lobes.py applies). A regions file
   with an id the dataset lacks, or with lon/lat that differ, is reported as
   coming from a different sampling build, and its containment is not tested,
   since every point would then "fail" for a reason that is not the polygon's.
4. pan_<ST>.parquet: FIRM panel polygons with FIRM_PAN. Also counted: overlaps.
   Panels are whole printed rectangles and overlap across county DFIRMs, so a
   point can lie in several; pull_point_extras.py resolves that by DFIRM.
5. Layers the analysis still needs (county lines, SSURGO, 10 m DEM, NWI, PLSS,
   roads, canals): a file-name search of data/raw, data/interim and --gis-dir.
   Canals: hydro/river_<ST>.parquet holds meta_nearest_ftype per point and no
   line geometry; NHDPlus V2 flowlines (FTYPE CanalDitch) are counted from the
   NHDSnapshot shapefiles if fetch_nhdplus_fl.sh has extracted them, both in
   total and intersecting the state's panel footprint.

USAGE
=====
    python check_gis_inputs.py --state FL --out results/$D/gis/check_gis_inputs_FL
    python check_gis_inputs.py --state FL --regions results/$D/regions/*_points.parquet \\
        --dataset data/processed/dataset_FL.parquet --out ... --overwrite
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely import STRtree

from pull_point_extras import (CRS_LL, CRS_WORK, HazReader, geo_crs, load_index,
                               load_panels, load_points, resolve_label_polygons)

PROCESSED = Path("data/processed")
PREP_DIR = Path("data/interim/prep")
NFHL_DIR = Path("data/interim/nfhl_states")
HYDRO_DIR = Path("data/interim/hydro")
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
GIS_DIR = Path("data/interim/gis")
NHD_DIR = Path("data/raw/nhdplus")

N_SAMPLE_GROUPS = 3
SEED = 0
LONLAT_TOL = 1e-9      # degrees; regions lon/lat are copies of the dataset's
SYMBOL = {"ok": "✓", "missing": "✗", "warn": "⚠", "info": "·"}

# File-name patterns per layer (case-insensitive, matched against file and
# directory names). Deliberately loose: a false hit is listed and read by eye.
LAYERS = {
    "county boundaries": ["tl_*_county*", "cb_*_county*", "county_lines_*", "counties_*"],
    "SSURGO soil map units": ["*ssurgo*", "*mupolygon*", "*muaggatt*", "ssurgo"],
    "10 m DEM (3DEP 1/3 arc-second)": ["*3dep*", "usgs_13_*", "usgs_1_3*", "dem"],
    "NWI wetlands": ["*wetland*", "*nwi*"],
    "PLSS sections": ["*plss*", "*cadnsdi*"],
    "roads": ["tl_*_roads*", "tl_*_prisecroads*", "*roads*"],
    "canals (other than NHDPlus)": ["*canal*"],
}
SKIP_DIRS = {"gee", "aef", "__pycache__"}       # thousands of per-point shards, no layers


class Report:
    """Checks in order, each with a verdict; rendered to JSON and Markdown."""

    def __init__(self, state: str):
        self.state = state
        self.rows: list[dict] = []

    def add(self, section: str, item: str, verdict: str, result: str, **data) -> None:
        self.rows.append(dict(section=section, item=item, verdict=verdict,
                              result=result, data=data))
        print(f"  {SYMBOL[verdict]} [{section}] {item}: {result}", flush=True)

    def write(self, stem: Path, inputs: dict) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        doc = dict(state=self.state, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   inputs={k: str(v) for k, v in inputs.items()}, checks=self.rows)
        stem.with_suffix(".json").write_text(json.dumps(doc, indent=1, default=_jsonable))

        lines = [f"# GIS inputs for the boundary analysis: {self.state}", "",
                 f"Written by `check_gis_inputs.py` on {doc['generated']}. "
                 f"✓ ok, ✗ missing or wrong, ⚠ works but needs a decision, · information.",
                 "", "Inputs:", ""]
        lines += [f"- `{k}`: `{v}`" for k, v in doc["inputs"].items()]
        for sec in dict.fromkeys(r["section"] for r in self.rows):
            lines += ["", f"## {sec}", "", "| check | | result |", "|---|---|---|"]
            for r in self.rows:
                if r["section"] == sec:
                    res = r["result"].replace("|", "\\|").replace("\n", "<br>")
                    lines.append(f"| {r['item']} | {SYMBOL[r['verdict']]} | {res} |")
        stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _gb(p: Path) -> str:
    return f"{p.stat().st_size / 1e9:.2f} GB"


# =========================================================================
# 1-2. hazard repack and its index
# =========================================================================
def check_haz(rep: Report, haz_p: Path, idx: pd.DataFrame | None, extent) -> HazReader | None:
    sec = "Hazard polygons (haz_5070 + idx)"
    if not haz_p.exists():
        rep.add(sec, "haz file", "missing", f"{haz_p} not found")
        return None
    haz = HazReader(haz_p)
    pf = haz.pf
    sizes = np.diff(haz.start)
    cols = pf.schema_arrow.names
    rep.add(sec, "haz file", "ok",
            f"{haz_p} ({_gb(haz_p)}): {haz.n_rows:,} rows in {len(sizes)} row groups of "
            f"{int(np.median(sizes)):,} (last {int(sizes[-1]):,})",
            rows=haz.n_rows, row_groups=len(sizes))
    rep.add(sec, "haz columns", "ok" if cols == ["row_id", "geometry"] else "warn",
            f"{cols} (geometry is plain WKB, {pf.schema_arrow.field('geometry').type})",
            columns=cols)
    md = pf.schema_arrow.metadata or {}
    rep.add(sec, "haz CRS", "ok" if b"geo" not in md else "info",
            "no GeoParquet metadata, so the file does not record its CRS: EPSG:5070 is "
            "implied by sample_points.prep_state, which reprojects NFHL's EPSG:4269 and "
            "writes plain WKB (tested below against the idx bounds and the panel extent)"
            if b"geo" not in md else f"geo metadata present: EPSG:{geo_crs(pf.schema_arrow)}")

    rid = pf.read(columns=["row_id"]).column("row_id").to_numpy()
    seq = bool(np.array_equal(rid, np.arange(haz.n_rows)))
    rep.add(sec, "haz row_id", "ok" if seq else "warn",
            f"row_id is 0..{haz.n_rows - 1} in file order: {seq}"
            f"{'; row_id // 10000 = row group' if seq and int(np.median(sizes)) == 10000 else ''}")

    rng = np.random.default_rng(SEED)
    groups = np.sort(rng.choice(len(sizes), min(N_SAMPLE_GROUPS, len(sizes)), replace=False))
    t0 = time.time()
    n = n_valid = n_empty = n_none = 0
    types: dict[str, int] = {}
    reasons: list[str] = []
    bdiff = 0.0
    bounds = []
    for gi in groups:
        t = pf.read_row_group(int(gi), columns=["row_id", "geometry"])
        r = t.column("row_id").to_numpy()
        g = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False))
        n += len(g)
        n_none += int(shapely.is_missing(g).sum())
        n_empty += int(shapely.is_empty(g).sum())
        v = shapely.is_valid(g)
        n_valid += int(v.sum())
        reasons += [shapely.is_valid_reason(x) for x in g[~v][:5]]
        for k, c in pd.Series(shapely.get_type_id(g)).value_counts().items():
            name = shapely.GeometryType(k).name if k >= 0 else "None"
            types[name] = types.get(name, 0) + int(c)
        b = shapely.bounds(g)
        bounds.append(b)
        if idx is not None:
            ib = idx.set_index("row_id").loc[r, ["minx", "miny", "maxx", "maxy"]].to_numpy()
            ok = np.isfinite(b).all(axis=1)
            bdiff = max(bdiff, float(np.abs(b[ok] - ib[ok]).max()) if ok.any() else 0.0)
    rep.add(sec, f"geometry validity ({len(groups)} random row groups)",
            "ok" if n_valid == n else "warn",
            f"row groups {groups.tolist()}: {n_valid:,}/{n:,} valid, {n_empty} empty, "
            f"{n_none} null; types {types}; {time.time() - t0:.0f}s"
            + (f"; invalid e.g. {reasons[:3]}" if reasons else ""),
            groups=groups, n=n, valid=n_valid, empty=n_empty, types=types)

    b = np.concatenate(bounds)
    b = b[np.isfinite(b).all(axis=1)]
    lo, hi = b[:, :2].min(axis=0), b[:, 2:].max(axis=0)
    inside = extent is None or (lo[0] >= extent[0] - 1e4 and lo[1] >= extent[1] - 1e4 and
                                hi[0] <= extent[2] + 1e4 and hi[1] <= extent[3] + 1e4)
    rep.add(sec, "implied CRS is EPSG:5070", "ok" if (bdiff < 1e-6 and inside) else "warn",
            f"sampled bounds x {lo[0]:,.0f}..{hi[0]:,.0f}, y {lo[1]:,.0f}..{hi[1]:,.0f} m; "
            f"inside the panel extent in EPSG:5070 (+10 km): {inside}; max |bounds - idx "
            f"bbox| {bdiff:.1e} m", max_bbox_diff=bdiff)

    if idx is None:
        return haz
    same = len(idx) == haz.n_rows and np.array_equal(np.sort(idx.row_id.to_numpy()), rid)
    rep.add(sec, "idx joins haz on row_id", "ok" if same else "missing",
            f"idx {len(idx):,} rows, haz {haz.n_rows:,}; identical row_id sets: {same}")
    for c in ("FLD_AR_ID", "FLD_ZONE", "SFHA_TF", "DFIRM_ID"):
        nn = int(idx[c].isna().sum())
        rep.add(sec, f"idx {c}", "ok" if nn == 0 else "warn", f"present, {nn:,} null")
    z = idx.FLD_ZONE.value_counts()
    sf = idx[idx.SFHA_TF == "T"].FLD_ZONE.value_counts()
    rep.add(sec, "FLD_ZONE", "info",
            "all polygons: " + ", ".join(f"{k} {v:,}" for k, v in z.head(8).items())
            + "; SFHA_TF == 'T': " + ", ".join(f"{k} {v:,}" for k, v in sf.items()),
            counts=z.to_dict(), sfha=sf.to_dict())
    d1 = idx.FLD_AR_ID.duplicated(keep=False)
    d2 = idx.duplicated(["DFIRM_ID", "FLD_AR_ID"], keep=False)
    n_keys = int(idx[d2].groupby(["DFIRM_ID", "FLD_AR_ID"]).ngroups)
    rep.add(sec, "polygon key uniqueness", "warn" if d2.any() else "ok",
            f"FLD_AR_ID alone repeats on {int(d1.sum()):,} rows; (DFIRM_ID, FLD_AR_ID) "
            f"repeats on {int(d2.sum()):,} rows ({n_keys} keys), resolved for a point by "
            f"point-in-polygon; row_id is the unique key (meta_src_row_id in v2)",
            fld_ar_id_dup_rows=int(d1.sum()), pair_dup_rows=int(d2.sum()), pair_dup_keys=n_keys)
    return haz


# =========================================================================
# 3. dataset keys and region positives
# =========================================================================
def check_dataset_keys(rep: Report, path: Path | None, idx: pd.DataFrame | None) -> None:
    sec = "Dataset positives -> polygons"
    if path is None or not path.exists():
        rep.add(sec, "dataset", "info", f"not checked ({path} not found)")
        return
    names = pq.read_schema(path).names
    cols = [c for c in ("point_id", "label", "dfirm_id", "meta_fld_ar_id", "meta_src_row_id")
            if c in names]
    df = pq.read_table(path, columns=cols).to_pandas()
    pos = df[df.label == 1]
    has = pos.meta_fld_ar_id.notna()
    rep.add(sec, "meta_fld_ar_id on positives", "ok" if has.all() else "warn",
            f"{int(has.sum()):,}/{len(pos):,} positives carry it ({path})")
    rep.add(sec, "meta_src_row_id", "ok" if "meta_src_row_id" in df else "info",
            f"present, {int(df.meta_src_row_id.notna().sum()):,}/{len(df):,} set"
            if "meta_src_row_id" in df else
            "absent (v1 sample): polygons are recovered from (dfirm_id, meta_fld_ar_id)")
    if idx is None:
        return
    k = idx.groupby(["DFIRM_ID", "FLD_AR_ID"]).size().rename("n").reset_index()
    m = pos[has].merge(k, left_on=["dfirm_id", "meta_fld_ar_id"],
                       right_on=["DFIRM_ID", "FLD_AR_ID"], how="left")
    rep.add(sec, "(dfirm_id, meta_fld_ar_id) in idx", "ok" if m.n.notna().all() else "missing",
            f"{int((m.n == 1).sum()):,} unique, {int((m.n > 1).sum()):,} ambiguous "
            f"(point-in-polygon decides), {int(m.n.isna().sum()):,} not found; "
            f"containment of every positive is reported by pull_point_extras.py")


def check_regions(rep: Report, paths: list[Path], dataset: Path | None, idx, haz) -> None:
    sec = "Region positives -> containing polygon"
    if not paths:
        rep.add(sec, "regions", "info", "not checked (no --regions given)")
        return
    if idx is None or haz is None:
        rep.add(sec, "regions", "missing", "needs idx and haz")
        return
    ds_xy = None
    if dataset is not None and dataset.exists():
        ds, xy, how = load_points(dataset, cols=["point_id", "lon", "lat"])
        ds_xy = (pd.Series(np.arange(len(ds)), index=ds.point_id), xy, ds, how)
    attrs = idx.set_index("row_id")
    for p in paths:
        item = p.name
        if not p.exists():
            rep.add(sec, item, "missing", f"{p} not found")
            continue
        r = pd.read_parquet(p)
        need = {"point_id", "lon", "lat", "label", "dfirm_id", "meta_fld_ar_id"}
        if need - set(r.columns):
            rep.add(sec, item, "missing", f"columns missing: {sorted(need - set(r.columns))}")
            continue
        pos = r[r.label == 1].reset_index(drop=True)
        how = "lon/lat projected EPSG:4269 -> 5070"
        if ds_xy is not None:
            row_of, ds_points, ds, ds_how = ds_xy
            known = r.point_id.isin(row_of.index).to_numpy()
            at = row_of.loc[r.point_id[known]].to_numpy()
            dll = float(max(np.abs(ds.lon.to_numpy()[at] - r.lon.to_numpy()[known]).max(),
                            np.abs(ds.lat.to_numpy()[at] - r.lat.to_numpy()[known]).max())
                        if len(at) else 0.0)
            if not known.all() or dll > LONLAT_TOL:
                why = [f"{int((~known).sum()):,} of {len(r):,} point_ids are not in "
                       f"{dataset.name}"] if not known.all() else []
                if dll > LONLAT_TOL:
                    why.append(f"lon/lat differ from {dataset.name}'s by up to {dll:.1e} deg "
                               f"for the same point_id (copies agree to {LONLAT_TOL:.0e})")
                rep.add(sec, item, "missing",
                        f"regions and --dataset come from different sampling builds: "
                        + "; ".join(why) + "; containment not checked: pass the dataset the "
                        f"regions were selected from",
                        rows=len(r), absent=int((~known).sum()), max_lonlat_diff_deg=dll)
                continue
            xy = ds_points[row_of.loc[pos.point_id].to_numpy()]
            how = f"{ds_how}, by point_id; lon/lat agree with it to {dll:.0e} deg"
        else:
            from pyproj import Transformer
            tf = Transformer.from_crs(CRS_LL, CRS_WORK, always_xy=True)
            xy = np.column_stack(tf.transform(pos.lon.to_numpy(float), pos.lat.to_numpy(float)))
        src, info = resolve_label_polygons(pos, xy, idx, haz)
        geoms = haz.read(src[src >= 0])
        covered = np.array([s >= 0 and geoms.get(int(s)) is not None and
                            geoms[int(s)].covers(shapely.Point(xy[i]))
                            for i, s in enumerate(src)], dtype=bool)
        zones = attrs.FLD_ZONE.reindex(src[src >= 0]).value_counts().to_dict()
        bad = pos.point_id[~covered].tolist()
        rep.add(sec, item, "ok" if covered.all() and len(pos) else "warn",
                f"{len(pos):,} positives of {len(r):,} rows; polygon found for "
                f"{int((src >= 0).sum()):,} ({info['from_column']:,} from meta_src_row_id, "
                f"{info['key_unique'] + info['key_ambiguous_resolved']:,} from the key); "
                f"point covered by it: {int(covered.sum()):,}; polygon zones {zones}; "
                f"xy: {how}" + (f"; NOT covered e.g. {bad[:5]}" if bad else ""),
                positives=len(pos), found=int((src >= 0).sum()), covered=int(covered.sum()),
                zones=zones, not_covered=bad[:50], resolution=info)


# =========================================================================
# 4. FIRM panels
# =========================================================================
def check_panels(rep: Report, path: Path):
    sec = "FIRM panels (pan)"
    if not path.exists():
        rep.add(sec, "panel file", "missing", f"{path} not found")
        return None
    import geopandas as gpd
    raw = gpd.read_parquet(path)
    epsg = raw.crs.to_epsg() if raw.crs is not None else None
    types = raw.geom_type.value_counts().to_dict()
    v0 = int(shapely.is_valid(np.asarray(raw.geometry.values)).sum())
    rep.add(sec, "panel polygons", "ok" if len(raw) and "FIRM_PAN" in raw else "missing",
            f"{path}: {len(raw):,} rows, CRS EPSG:{epsg}, types {types}, {v0:,}/{len(raw):,} "
            f"valid in EPSG:{epsg}", rows=len(raw), epsg=epsg, types=types)
    if "FIRM_PAN" in raw:
        rep.add(sec, "FIRM_PAN", "ok" if raw.FIRM_PAN.notna().all() and raw.FIRM_PAN.is_unique
                else "warn", f"{int(raw.FIRM_PAN.notna().sum()):,} set, unique: "
                f"{raw.FIRM_PAN.is_unique}; {raw.DFIRM_ID.nunique()} DFIRMs; columns "
                f"{[c for c in raw.columns if c != 'geometry']}")
    pan, fixed = load_panels(path)
    rep.add(sec, f"validity in EPSG:{CRS_WORK}", "ok" if not fixed.any() else "warn",
            f"{int(fixed.sum())} panel(s) invalid after reprojection "
            f"{pan.FIRM_PAN[fixed].tolist()[:5]}; repaired with make_valid by "
            f"pull_point_extras.py and pull_gis_lines.py")
    for c in ("PANEL_TYP", "SCALE"):
        if c in pan:
            rep.add(sec, c, "info", ", ".join(f"{k} {v:,}" for k, v in
                                              pan[c].value_counts().items()))
    g = np.asarray(pan.geometry.values)
    area = shapely.area(g)
    union = shapely.union_all(g)
    pr = STRtree(g).query(g, predicate="overlaps")
    pr = pr[:, pr[0] < pr[1]]
    ov = shapely.area(shapely.intersection(g[pr[0]], g[pr[1]]))
    same = pan.DFIRM_ID.to_numpy()[pr[0]] == pan.DFIRM_ID.to_numpy()[pr[1]]
    rep.add(sec, "overlaps", "warn" if (~same & (ov > 1)).any() else "ok",
            f"panel areas sum {area.sum() / 1e6:,.0f} km2, union {shapely.area(union) / 1e6:,.0f}"
            f" km2; overlapping pairs within a DFIRM {int((same & (ov > 1)).sum())} "
            f"({ov[same].sum() / 1e6:.2f} km2), across DFIRMs {int((~same & (ov > 1)).sum()):,} "
            f"({ov[~same].sum() / 1e6:,.0f} km2): panels are whole printed rectangles, so a "
            f"point near a county line can lie in several; take the panel of the point's own "
            f"DFIRM (pull_point_extras.py), and compare a lobe only with its own DFIRM's "
            f"panel edges (pull_gis_lines.py writes dfirm_id on every edge)",
            union_km2=shapely.area(union) / 1e6, overlap_pairs_same=int((same & (ov > 1)).sum()),
            overlap_pairs_cross=int((~same & (ov > 1)).sum()))
    return pan, union


# =========================================================================
# 5. layers still to collect
# =========================================================================
def find_files(roots: list[Path], patterns: list[str]) -> list[str]:
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for name in dirs + files:
                low = name.lower()
                if any(fnmatch.fnmatch(low, p) for p in patterns):
                    hits.append(str(Path(d) / name))
    return sorted(set(hits))


def check_layers(rep: Report, roots: list[Path], gis_dir: Path, state: str) -> None:
    sec = "Layers still to collect"
    for name, pats in LAYERS.items():
        hits = [h for h in find_files(roots, pats) if not h.endswith(".gdb.zip")]
        rep.add(sec, name, "ok" if hits else "missing",
                (f"{len(hits)} on disk: " + "; ".join(hits[:6]) + (" ..." if len(hits) > 6 else ""))
                if hits else f"none under {', '.join(str(r) for r in roots)}", files=hits)
    planned = [gis_dir / f"county_lines_{state}.parquet", gis_dir / f"panel_edges_{state}.parquet",
               gis_dir / "dem", gis_dir / "ssurgo"]
    rep.add(sec, "planned GIS outputs", "info",
            "; ".join(f"{p}: {'present' if p.exists() else 'not yet'}" for p in planned))


def check_canals(rep: Report, river_p: Path, nhd_dir: Path, footprint) -> None:
    sec = "Canals and ditches"
    if river_p.exists():
        names = pq.read_schema(river_p).names
        ft = (pd.read_parquet(river_p, columns=["meta_nearest_ftype"]).meta_nearest_ftype
              .value_counts().to_dict() if "meta_nearest_ftype" in names else {})
        rep.add(sec, river_p.name, "info",
                f"per-point table ({len(names)} columns, geometry column: "
                f"{'geometry' in names}); nearest flowline FTYPE {ft}. It names the nearest "
                f"flowline type per point but holds no line geometry, so it cannot give "
                f"distances from lobe vertices to canals", nearest_ftype=ft)
    else:
        rep.add(sec, str(river_p), "missing", "not found")

    shp = sorted(nhd_dir.glob("*/NHDSnapshot/**/NHDFlowline.shp"))
    units = (sorted(p for p in nhd_dir.iterdir()
                    if p.is_dir() and not p.suffix and p.name != "archives")
             if nhd_dir.exists() else [])
    # fetch_nhdplus_fl.sh touches <unit>/<layer>/.extracted once a layer is unpacked.
    marks = {u.name: {lay.name: (lay / ".extracted").exists()
                      for lay in sorted(u.iterdir()) if lay.is_dir()} for u in units}
    rep.add(sec, "NHDPlus V2 download", "ok" if shp else "missing",
            f"{len(shp)} NHDFlowline shapefiles under {nhd_dir}; extracted markers "
            f"{marks or 'none'}" if nhd_dir.exists() else f"{nhd_dir} not found", markers=marks)
    if not shp:
        return
    import pyogrio
    import geopandas as gpd
    fp_ll = None
    if footprint is not None:
        fp_ll = gpd.GeoSeries([footprint], crs=CRS_WORK).to_crs(CRS_LL).iloc[0]
        shapely.prepare(fp_ll)
    tot = dict(features=0, canal=0, canal_state=0, km_state=0.0)
    for p in shp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # "Measured 3D LineString" -> LineString Z
            info = pyogrio.read_info(p)
            ft = pyogrio.read_dataframe(p, columns=["FTYPE"], read_geometry=False).FTYPE
            n_canal = int((ft == "CanalDitch").sum())
            n_in, km = 0, 0.0
            if fp_ll is not None and n_canal:
                c = pyogrio.read_dataframe(p, columns=["COMID", "FTYPE"],
                                           where="FTYPE = 'CanalDitch'",
                                           bbox=tuple(shapely.bounds(fp_ll)))
                if len(c):
                    hit = shapely.intersects(fp_ll, np.asarray(c.geometry.values))
                    n_in = int(hit.sum())
                    km = float(c[hit].to_crs(CRS_WORK).length.sum() / 1000)
        unit = p.relative_to(nhd_dir).parts[0]
        rep.add(sec, f"NHDFlowline {unit}", "ok" if n_in else "info",
                f"{info['features']:,} flowlines ({info['crs']}); CanalDitch {n_canal:,}, of "
                f"which {n_in:,} ({km:,.0f} km) intersect the {rep.state} panel footprint; "
                f"FTYPE {ft.value_counts().to_dict()}", file=str(p), features=info["features"],
                canal=n_canal, canal_in_state=n_in, canal_km_in_state=km)
        tot["features"] += info["features"]; tot["canal"] += n_canal
        tot["canal_state"] += n_in; tot["km_state"] += km
    rep.add(sec, "CanalDitch flowlines in total", "ok" if tot["canal_state"] else "missing",
            f"{tot['canal_state']:,} CanalDitch flowlines ({tot['km_state']:,.0f} km) intersect "
            f"the {rep.state} panel footprint, of {tot['canal']:,} in the {len(shp)} units: the "
            f"canal layer for the optional class (D6) can be cut from these, no download needed",
            **tot)


# =========================================================================
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--regions", type=Path, nargs="*", default=[],
                    help="aotil/regions.py *_points.parquet file(s)")
    ap.add_argument("--dataset", type=Path, help="default data/processed/dataset_<ST>.parquet")
    ap.add_argument("--idx", type=Path)
    ap.add_argument("--haz", type=Path)
    ap.add_argument("--panels", type=Path)
    ap.add_argument("--river", type=Path)
    ap.add_argument("--nhdplus-dir", type=Path, default=NHD_DIR)
    ap.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    ap.add_argument("--out", type=Path,
                    help="output stem; writes <stem>.json and <stem>.md "
                         "(default results/<today>/gis/check_gis_inputs_<ST>)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    st = a.state
    a.dataset = a.dataset or PROCESSED / f"dataset_{st}.parquet"
    a.idx = a.idx or PREP_DIR / f"idx_{st}.parquet"
    a.haz = a.haz or PREP_DIR / f"haz_{st}_5070.parquet"
    a.panels = a.panels or NFHL_DIR / f"pan_{st}.parquet"
    a.river = a.river or HYDRO_DIR / f"river_{st}.parquet"
    a.out = a.out or Path("results") / date.today().isoformat() / "gis" / f"check_gis_inputs_{st}"
    if a.out.suffix in (".json", ".md"):
        a.out = a.out.with_suffix("")
    if a.out.with_suffix(".json").exists() and a.out.with_suffix(".md").exists() \
            and not a.overwrite:
        print(f"{a.out}.json/.md exist -- skipping (use --overwrite)")
        return 0

    t0 = time.time()
    print(f"[{st}] checking GIS inputs", flush=True)
    rep = Report(st)
    idx = load_index(a.idx) if a.idx.exists() else None
    if idx is None:
        rep.add("Hazard polygons (haz_5070 + idx)", "idx file", "missing", f"{a.idx} not found")
    pan = check_panels(rep, a.panels)
    extent = shapely.bounds(pan[1]) if pan is not None else None
    haz = check_haz(rep, a.haz, idx, extent)
    check_dataset_keys(rep, a.dataset, idx)
    check_regions(rep, a.regions, a.dataset, idx, haz)
    check_layers(rep, [RAW_DIR, INTERIM_DIR, a.gis_dir], a.gis_dir, st)
    check_canals(rep, a.river, a.nhdplus_dir, pan[1] if pan is not None else None)

    rep.write(a.out, dict(dataset=a.dataset, idx=a.idx, haz=a.haz, panels=a.panels,
                          river=a.river, nhdplus_dir=a.nhdplus_dir, gis_dir=a.gis_dir,
                          regions=";".join(map(str, a.regions)) or None))
    n = pd.Series([r["verdict"] for r in rep.rows]).value_counts().to_dict()
    print(f"\n[{st}] {len(rep.rows)} checks: " + ", ".join(f"{SYMBOL[k]} {k} {v}"
                                                         for k, v in n.items()))
    print(f"wrote {a.out}.json and {a.out}.md  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
