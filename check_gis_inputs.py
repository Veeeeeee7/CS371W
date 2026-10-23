#!/usr/bin/env python3
"""Check the inputs the boundary analysis rests on, and whether each is what it claims to be.

Writes <out>.json (for scripts) and <out>.md (for the audit report): one row per check, with a
verdict of ok, missing, warn (works but needs a decision) or info. The checks:
  - haz_<ST>_5070.parquet: row groups, columns, row_id order, geometry validity on a few random
    row groups, and its unrecorded CRS (EPSG:5070) tested against the idx bboxes and panel extent;
  - idx_<ST>.parquet: joins haz on row_id, and how often the polygon keys repeat;
  - dataset and --regions positives: every one finds the polygon that labelled it, and is covered;
  - pan_<ST>.parquet: FIRM panels, their validity in EPSG:5070 and their overlaps;
  - the layers still to collect (a file-name search), and canal flowlines in NHDPlus V2.
See docs/TASK-DATA-AUDIT.md, Part D0.

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

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import shapely
from shapely import STRtree

from pull_point_extras import (CRS_LL, CRS_WORK, HazReader, geo_crs, load_index,
                               load_panels, load_points, project_to_work,
                               resolve_label_polygons)

PROCESSED = Path("data/processed")
PREP_DIR = Path("data/interim/prep")
NFHL_DIR = Path("data/interim/nfhl_states")
HYDRO_DIR = Path("data/interim/hydro")
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
GIS_DIR = Path("data/interim/gis")
NHD_DIR = Path("data/raw/nhdplus")

N_SAMPLE_GROUPS = 3    # haz row groups decoded for the validity check (of ~60 in Florida)
SEED = 0               # RNG seed for picking them
LONLAT_TOL = 1e-9      # degrees; regions lon/lat are copies of the dataset's
EXTENT_PAD_M = 1e4     # slack when testing the haz bounds against the panel extent
BBOX_TOL_M = 1e-6      # largest |decoded bounds - idx bbox| that still confirms EPSG:5070
OVERLAP_MIN_M2 = 1     # two panels overlap when they share more than this
SYMBOL = {"ok": "✓", "missing": "✗", "warn": "⚠", "info": "·"}
HAZ_SECTION = "Hazard polygons (haz_5070 + idx)"
REGION_COLS = {"point_id", "lon", "lat", "label", "dfirm_id", "meta_fld_ar_id"}

# File-name patterns per layer (case-insensitive, matched against file and directory names).
# Deliberately loose: a false hit is listed and read by eye.
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
        for k, v in doc["inputs"].items():
            lines.append(f"- `{k}`: `{v}`")
        sections = []
        for row in self.rows:
            if row["section"] not in sections:
                sections.append(row["section"])
        for section in sections:
            lines += ["", f"## {section}", "", "| check | | result |", "|---|---|---|"]
            for row in self.rows:
                if row["section"] == section:
                    result = row["result"].replace("|", "\\|").replace("\n", "<br>")
                    lines.append(f"| {row['item']} | {SYMBOL[row['verdict']]} | {result} |")
        stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _gb(p: Path) -> str:
    return f"{p.stat().st_size / 1e9:.2f} GB"


def counts_text(counts) -> str:
    """'k1 n1, k2 n2, ...' from (key, count) pairs."""
    return ", ".join(f"{k} {v:,}" for k, v in counts)


# =========================================================================
# hazard repack and its index
# =========================================================================
def sample_row_groups(pf: pq.ParquetFile, groups: np.ndarray, idx: pd.DataFrame | None) -> dict:
    """Decode whole row groups: validity, geometry types, bounds, and the largest difference
    between the decoded bounds and the idx bbox columns."""
    stats = dict(n=0, valid=0, empty=0, none=0, types={}, reasons=[], bbox_diff=0.0, bounds=[])
    idx_bounds = None
    if idx is not None:
        idx_bounds = idx.set_index("row_id")[["minx", "miny", "maxx", "maxy"]]
    for gi in groups:
        table = pf.read_row_group(int(gi), columns=["row_id", "geometry"])
        row_ids = table.column("row_id").to_numpy()
        geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
        stats["n"] += len(geoms)
        stats["none"] += int(shapely.is_missing(geoms).sum())
        stats["empty"] += int(shapely.is_empty(geoms).sum())
        valid = shapely.is_valid(geoms)
        stats["valid"] += int(valid.sum())
        for geom in geoms[~valid][:5]:
            stats["reasons"].append(shapely.is_valid_reason(geom))
        for type_id, count in pd.Series(shapely.get_type_id(geoms)).value_counts().items():
            name = shapely.GeometryType(type_id).name if type_id >= 0 else "None"
            stats["types"][name] = stats["types"].get(name, 0) + int(count)
        bounds = shapely.bounds(geoms)
        stats["bounds"].append(bounds)
        if idx_bounds is not None:
            expected = idx_bounds.loc[row_ids].to_numpy()
            ok = np.isfinite(bounds).all(axis=1)
            diff = float(np.abs(bounds[ok] - expected[ok]).max()) if ok.any() else 0.0
            stats["bbox_diff"] = max(stats["bbox_diff"], diff)
    return stats


def check_idx(rep: Report, idx: pd.DataFrame, haz: HazReader, rid: np.ndarray) -> None:
    sec = HAZ_SECTION
    same = len(idx) == haz.n_rows and np.array_equal(np.sort(idx.row_id.to_numpy()), rid)
    rep.add(sec, "idx joins haz on row_id", "ok" if same else "missing",
            f"idx {len(idx):,} rows, haz {haz.n_rows:,}; identical row_id sets: {same}")
    for col in ("FLD_AR_ID", "FLD_ZONE", "SFHA_TF", "DFIRM_ID"):
        n_null = int(idx[col].isna().sum())
        rep.add(sec, f"idx {col}", "ok" if n_null == 0 else "warn", f"present, {n_null:,} null")

    zones = idx.FLD_ZONE.value_counts()
    sfha_zones = idx[idx.SFHA_TF == "T"].FLD_ZONE.value_counts()
    rep.add(sec, "FLD_ZONE", "info",
            "all polygons: " + counts_text(zones.head(8).items())
            + "; SFHA_TF == 'T': " + counts_text(sfha_zones.items()),
            counts=zones.to_dict(), sfha=sfha_zones.to_dict())

    # v1 points fall back on (DFIRM_ID, FLD_AR_ID); FLD_AR_ID alone repeats across DFIRMs
    ar_dup = idx.FLD_AR_ID.duplicated(keep=False)
    pair_dup = idx.duplicated(["DFIRM_ID", "FLD_AR_ID"], keep=False)
    n_keys = int(idx[pair_dup].groupby(["DFIRM_ID", "FLD_AR_ID"]).ngroups)
    rep.add(sec, "polygon key uniqueness", "warn" if pair_dup.any() else "ok",
            f"FLD_AR_ID alone repeats on {int(ar_dup.sum()):,} rows; (DFIRM_ID, FLD_AR_ID) "
            f"repeats on {int(pair_dup.sum()):,} rows ({n_keys} keys), resolved for a point by "
            f"point-in-polygon; row_id is the unique key (meta_src_row_id in v2)",
            fld_ar_id_dup_rows=int(ar_dup.sum()), pair_dup_rows=int(pair_dup.sum()),
            pair_dup_keys=n_keys)


def check_haz(rep: Report, haz_p: Path, idx: pd.DataFrame | None, extent) -> HazReader | None:
    """The haz file is only row_id + WKB, so its CRS is not recorded: EPSG:5070 is implied by
    sample_points.prep_state. The decoded bounds must equal the idx bboxes prep_state computed
    from the same geometry, and lie inside the panel extent in EPSG:5070 (degrees would not)."""
    sec = HAZ_SECTION
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
    if b"geo" not in (pf.schema_arrow.metadata or {}):
        rep.add(sec, "haz CRS", "ok",
                "no GeoParquet metadata, so the file does not record its CRS: EPSG:5070 is "
                "implied by sample_points.prep_state, which reprojects NFHL's EPSG:4269 and "
                "writes plain WKB (tested below against the idx bounds and the panel extent)")
    else:
        rep.add(sec, "haz CRS", "info", f"geo metadata present: EPSG:{geo_crs(pf.schema_arrow)}")

    rid = pf.read(columns=["row_id"]).column("row_id").to_numpy()
    in_order = bool(np.array_equal(rid, np.arange(haz.n_rows)))
    group_note = ""
    if in_order and int(np.median(sizes)) == 10000:
        group_note = "; row_id // 10000 = row group"
    rep.add(sec, "haz row_id", "ok" if in_order else "warn",
            f"row_id is 0..{haz.n_rows - 1} in file order: {in_order}{group_note}")

    rng = np.random.default_rng(SEED)
    groups = np.sort(rng.choice(len(sizes), min(N_SAMPLE_GROUPS, len(sizes)), replace=False))
    t0 = time.time()
    stats = sample_row_groups(pf, groups, idx)
    invalid_note = f"; invalid e.g. {stats['reasons'][:3]}" if stats["reasons"] else ""
    rep.add(sec, f"geometry validity ({len(groups)} random row groups)",
            "ok" if stats["valid"] == stats["n"] else "warn",
            f"row groups {groups.tolist()}: {stats['valid']:,}/{stats['n']:,} valid, "
            f"{stats['empty']} empty, {stats['none']} null; types {stats['types']}; "
            f"{time.time() - t0:.0f}s{invalid_note}",
            groups=groups, n=stats["n"], valid=stats["valid"], empty=stats["empty"],
            types=stats["types"])

    bounds = np.concatenate(stats["bounds"])
    bounds = bounds[np.isfinite(bounds).all(axis=1)]
    lo = bounds[:, :2].min(axis=0)
    hi = bounds[:, 2:].max(axis=0)
    inside = extent is None or (lo[0] >= extent[0] - EXTENT_PAD_M
                                and lo[1] >= extent[1] - EXTENT_PAD_M
                                and hi[0] <= extent[2] + EXTENT_PAD_M
                                and hi[1] <= extent[3] + EXTENT_PAD_M)
    bbox_diff = stats["bbox_diff"]
    rep.add(sec, "implied CRS is EPSG:5070",
            "ok" if (bbox_diff < BBOX_TOL_M and inside) else "warn",
            f"sampled bounds x {lo[0]:,.0f}..{hi[0]:,.0f}, y {lo[1]:,.0f}..{hi[1]:,.0f} m; "
            f"inside the panel extent in EPSG:5070 (+10 km): {inside}; max |bounds - idx "
            f"bbox| {bbox_diff:.1e} m", max_bbox_diff=bbox_diff)

    if idx is not None:
        check_idx(rep, idx, haz, rid)
    return haz


# =========================================================================
# dataset keys and region positives
# =========================================================================
def check_dataset_keys(rep: Report, path: Path | None, idx: pd.DataFrame | None) -> None:
    sec = "Dataset positives -> polygons"
    if path is None or not path.exists():
        rep.add(sec, "dataset", "info", f"not checked ({path} not found)")
        return
    names = pq.read_schema(path).names
    wanted = ("point_id", "label", "dfirm_id", "meta_fld_ar_id", "meta_src_row_id")
    cols = [c for c in wanted if c in names]
    df = pq.read_table(path, columns=cols).to_pandas()
    pos = df[df.label == 1]
    has = pos.meta_fld_ar_id.notna()
    rep.add(sec, "meta_fld_ar_id on positives", "ok" if has.all() else "warn",
            f"{int(has.sum()):,}/{len(pos):,} positives carry it ({path})")
    if "meta_src_row_id" in df:
        rep.add(sec, "meta_src_row_id", "ok",
                f"present, {int(df.meta_src_row_id.notna().sum()):,}/{len(df):,} set")
    else:
        rep.add(sec, "meta_src_row_id", "info",
                "absent (v1 sample): polygons are recovered from (dfirm_id, meta_fld_ar_id)")
    if idx is None:
        return
    key_sizes = idx.groupby(["DFIRM_ID", "FLD_AR_ID"]).size().rename("n").reset_index()
    matched = pos[has].merge(key_sizes, left_on=["dfirm_id", "meta_fld_ar_id"],
                             right_on=["DFIRM_ID", "FLD_AR_ID"], how="left")
    verdict = "ok" if matched.n.notna().all() else "missing"
    rep.add(sec, "(dfirm_id, meta_fld_ar_id) in idx", verdict,
            f"{int((matched.n == 1).sum()):,} unique, {int((matched.n > 1).sum()):,} ambiguous "
            f"(point-in-polygon decides), {int(matched.n.isna().sum()):,} not found; "
            f"containment of every positive is reported by pull_point_extras.py")


def max_lonlat_diff(ds: pd.DataFrame, at: np.ndarray, regions: pd.DataFrame,
                    known: np.ndarray) -> float:
    """Largest lon or lat difference between region rows and their dataset rows."""
    if len(at) == 0:
        return 0.0
    dlon = np.abs(ds.lon.to_numpy()[at] - regions.lon.to_numpy()[known]).max()
    dlat = np.abs(ds.lat.to_numpy()[at] - regions.lat.to_numpy()[known]).max()
    return float(max(dlon, dlat))


def build_mismatch(known: np.ndarray, dll: float, n_rows: int, dataset_name: str) -> list[str]:
    """Why a regions file does not come from the dataset's sampling build (empty if it does)."""
    why = []
    if not known.all():
        why.append(f"{int((~known).sum()):,} of {n_rows:,} point_ids are not in {dataset_name}")
    if dll > LONLAT_TOL:
        why.append(f"lon/lat differ from {dataset_name}'s by up to {dll:.1e} deg "
                   f"for the same point_id (copies agree to {LONLAT_TOL:.0e})")
    return why


def check_containment(rep: Report, sec: str, item: str, n_rows: int, pos: pd.DataFrame,
                      xy: np.ndarray, how: str, idx: pd.DataFrame, haz: HazReader) -> None:
    """Every region positive's labelling polygon is found and covers the point."""
    src, info = resolve_label_polygons(pos, xy, idx, haz)
    geoms = haz.read(src[src >= 0])
    covered = np.zeros(len(src), dtype=bool)
    for i, s in enumerate(src):
        if s >= 0 and geoms.get(int(s)) is not None:
            covered[i] = geoms[int(s)].covers(shapely.Point(xy[i]))
    zones = idx.set_index("row_id").FLD_ZONE.reindex(src[src >= 0]).value_counts().to_dict()
    bad = pos.point_id[~covered].tolist()
    n_found = int((src >= 0).sum())
    from_key = info["key_unique"] + info["key_ambiguous_resolved"]
    bad_note = f"; NOT covered e.g. {bad[:5]}" if bad else ""
    rep.add(sec, item, "ok" if covered.all() and len(pos) > 0 else "warn",
            f"{len(pos):,} positives of {n_rows:,} rows; polygon found for "
            f"{n_found:,} ({info['from_column']:,} from meta_src_row_id, "
            f"{from_key:,} from the key); "
            f"point covered by it: {int(covered.sum()):,}; polygon zones {zones}; "
            f"xy: {how}{bad_note}",
            positives=len(pos), found=n_found, covered=int(covered.sum()),
            zones=zones, not_covered=bad[:50], resolution=info)


def check_regions(rep: Report, paths: list[Path], dataset: Path | None, idx, haz) -> None:
    """Region positives against their labelling polygons.

    The points come from the dataset's EPSG:5070 geometry by point_id when --dataset exists,
    else from lon/lat. Every sampling build reissues point_ids from FL0000000, so a matching id
    proves nothing: the lon/lat must match too, or the regions come from another build and the
    containment test is skipped (every point would fail for a reason not the polygon's).
    """
    sec = "Region positives -> containing polygon"
    if not paths:
        rep.add(sec, "regions", "info", "not checked (no --regions given)")
        return
    if idx is None or haz is None:
        rep.add(sec, "regions", "missing", "needs idx and haz")
        return
    ds = None
    if dataset is not None and dataset.exists():
        ds, ds_xy, ds_how = load_points(dataset, cols=["point_id", "lon", "lat"])
        row_of = pd.Series(np.arange(len(ds)), index=ds.point_id)

    for path in paths:
        item = path.name
        if not path.exists():
            rep.add(sec, item, "missing", f"{path} not found")
            continue
        regions = pd.read_parquet(path)
        missing_cols = REGION_COLS - set(regions.columns)
        if missing_cols:
            rep.add(sec, item, "missing", f"columns missing: {sorted(missing_cols)}")
            continue
        pos = regions[regions.label == 1].reset_index(drop=True)

        if ds is None:
            xy = project_to_work(pos.lon.to_numpy(float), pos.lat.to_numpy(float), CRS_LL)
            how = "lon/lat projected EPSG:4269 -> 5070"
        else:
            known = regions.point_id.isin(row_of.index).to_numpy()
            at = row_of.loc[regions.point_id[known]].to_numpy()
            dll = max_lonlat_diff(ds, at, regions, known)
            why = build_mismatch(known, dll, len(regions), dataset.name)
            if why:
                rep.add(sec, item, "missing",
                        "regions and --dataset come from different sampling builds: "
                        + "; ".join(why) + "; containment not checked: pass the dataset the "
                        "regions were selected from",
                        rows=len(regions), absent=int((~known).sum()), max_lonlat_diff_deg=dll)
                continue
            xy = ds_xy[row_of.loc[pos.point_id].to_numpy()]
            how = f"{ds_how}, by point_id; lon/lat agree with it to {dll:.0e} deg"
        check_containment(rep, sec, item, len(regions), pos, xy, how, idx, haz)


# =========================================================================
# FIRM panels
# =========================================================================
def check_panel_overlaps(rep: Report, sec: str, pan) -> shapely.Geometry:
    """Panels are whole printed rectangles and overlap across county DFIRMs, so a point can lie
    in several. Returns the union of the panels (the state's footprint)."""
    geoms = np.asarray(pan.geometry.values)
    area = shapely.area(geoms)
    union = shapely.union_all(geoms)
    union_km2 = shapely.area(union) / 1e6
    # 'intersects', not 'overlaps': 'overlaps' is false when one panel contains the other
    pairs = STRtree(geoms).query(geoms, predicate="intersects")
    pairs = pairs[:, pairs[0] < pairs[1]]
    overlap = shapely.area(shapely.intersection(geoms[pairs[0]], geoms[pairs[1]]))
    dfirm = pan.DFIRM_ID.to_numpy()
    same = dfirm[pairs[0]] == dfirm[pairs[1]]
    real = overlap > OVERLAP_MIN_M2
    within = same & real
    across = ~same & real
    rep.add(sec, "overlaps", "warn" if across.any() else "ok",
            f"panel areas sum {area.sum() / 1e6:,.0f} km2, union {union_km2:,.0f}"
            f" km2; overlapping pairs within a DFIRM {int(within.sum())} "
            f"({overlap[within].sum() / 1e6:.2f} km2), across DFIRMs "
            f"{int(across.sum()):,} "
            f"({overlap[across].sum() / 1e6:,.0f} km2): panels are whole printed rectangles, so a "
            f"point near a county line can lie in several; take the panel of the point's own "
            f"DFIRM (pull_point_extras.py), and compare a lobe only with its own DFIRM's "
            f"panel edges (pull_gis_lines.py writes dfirm_id on every edge)",
            union_km2=union_km2, overlap_pairs_same=int(within.sum()),
            overlap_pairs_cross=int(across.sum()))
    return union


def check_panels(rep: Report, path: Path):
    """FIRM panel polygons with FIRM_PAN. Returns (panels in EPSG:5070, their union), or None."""
    sec = "FIRM panels (pan)"
    if not path.exists():
        rep.add(sec, "panel file", "missing", f"{path} not found")
        return None
    raw = gpd.read_parquet(path)
    epsg = raw.crs.to_epsg() if raw.crs is not None else None
    types = raw.geom_type.value_counts().to_dict()
    n_valid = int(shapely.is_valid(np.asarray(raw.geometry.values)).sum())
    rep.add(sec, "panel polygons", "ok" if len(raw) and "FIRM_PAN" in raw else "missing",
            f"{path}: {len(raw):,} rows, CRS EPSG:{epsg}, types {types}, {n_valid:,}/{len(raw):,} "
            f"valid in EPSG:{epsg}", rows=len(raw), epsg=epsg, types=types)
    if "FIRM_PAN" in raw:
        complete = raw.FIRM_PAN.notna().all() and raw.FIRM_PAN.is_unique
        other_cols = [c for c in raw.columns if c != "geometry"]
        rep.add(sec, "FIRM_PAN", "ok" if complete else "warn",
                f"{int(raw.FIRM_PAN.notna().sum()):,} set, unique: "
                f"{raw.FIRM_PAN.is_unique}; {raw.DFIRM_ID.nunique()} DFIRMs; columns "
                f"{other_cols}")
    pan, fixed = load_panels(path)
    rep.add(sec, f"validity in EPSG:{CRS_WORK}", "ok" if not fixed.any() else "warn",
            f"{int(fixed.sum())} panel(s) invalid after reprojection "
            f"{pan.FIRM_PAN[fixed].tolist()[:5]}; repaired with make_valid by "
            f"pull_point_extras.py and pull_gis_lines.py")
    for col in ("PANEL_TYP", "SCALE"):
        if col in pan:
            rep.add(sec, col, "info", counts_text(pan[col].value_counts().items()))
    union = check_panel_overlaps(rep, sec, pan)
    return pan, union


# =========================================================================
# layers still to collect
# =========================================================================
def find_files(roots: list[Path], patterns: list[str]) -> list[str]:
    """Paths under roots whose file or directory name matches a pattern (lower-cased)."""
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]     # prune the walk in place
            for name in dirs + files:
                low = name.lower()
                if any(fnmatch.fnmatch(low, p) for p in patterns):
                    hits.append(str(Path(d) / name))
    return sorted(set(hits))


def check_layers(rep: Report, roots: list[Path], gis_dir: Path, state: str) -> None:
    sec = "Layers still to collect"
    for name, patterns in LAYERS.items():
        hits = [h for h in find_files(roots, patterns) if not h.endswith(".gdb.zip")]
        if hits:
            more = " ..." if len(hits) > 6 else ""
            result = f"{len(hits)} on disk: " + "; ".join(hits[:6]) + more
        else:
            result = f"none under {', '.join(str(r) for r in roots)}"
        rep.add(sec, name, "ok" if hits else "missing", result, files=hits)
    planned = [gis_dir / f"county_lines_{state}.parquet", gis_dir / f"panel_edges_{state}.parquet",
               gis_dir / "dem", gis_dir / "ssurgo"]
    status = [f"{p}: {'present' if p.exists() else 'not yet'}" for p in planned]
    rep.add(sec, "planned GIS outputs", "info", "; ".join(status))


def extraction_markers(nhd_dir: Path) -> dict:
    """{unit: {layer: extracted?}}; fetch_nhdplus_fl.sh touches <unit>/<layer>/.extracted once a
    layer is unpacked."""
    if not nhd_dir.exists():
        return {}
    units = sorted(p for p in nhd_dir.iterdir()
                   if p.is_dir() and not p.suffix and p.name != "archives")
    marks = {}
    for unit in units:
        marks[unit.name] = {}
        for layer in sorted(unit.iterdir()):
            if layer.is_dir():
                marks[unit.name][layer.name] = (layer / ".extracted").exists()
    return marks


def count_canals(shp: Path, footprint_ll) -> tuple[dict, pd.Series, int, int, float]:
    """One NHDFlowline shapefile: its info, FTYPE column, CanalDitch count, and the CanalDitch
    flowlines intersecting the footprint (count and km)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # "Measured 3D LineString" -> LineString Z
        info = pyogrio.read_info(shp)
        ftype = pyogrio.read_dataframe(shp, columns=["FTYPE"], read_geometry=False).FTYPE
        n_canal = int((ftype == "CanalDitch").sum())
        n_in, km = 0, 0.0
        if footprint_ll is not None and n_canal:
            canals = pyogrio.read_dataframe(shp, columns=["COMID", "FTYPE"],
                                            where="FTYPE = 'CanalDitch'",
                                            bbox=tuple(shapely.bounds(footprint_ll)))
            if len(canals):
                hit = shapely.intersects(footprint_ll, np.asarray(canals.geometry.values))
                n_in = int(hit.sum())
                km = float(canals[hit].to_crs(CRS_WORK).length.sum() / 1000)
    return info, ftype, n_canal, n_in, km


def check_canals(rep: Report, river_p: Path, nhd_dir: Path, footprint) -> None:
    """hydro/river_<ST>.parquet names the nearest flowline type per point but holds no line
    geometry, so canals are counted from the NHDPlus V2 NHDSnapshot shapefiles instead."""
    sec = "Canals and ditches"
    if river_p.exists():
        names = pq.read_schema(river_p).names
        ftypes = {}
        if "meta_nearest_ftype" in names:
            river = pd.read_parquet(river_p, columns=["meta_nearest_ftype"])
            ftypes = river.meta_nearest_ftype.value_counts().to_dict()
        rep.add(sec, river_p.name, "info",
                f"per-point table ({len(names)} columns, geometry column: "
                f"{'geometry' in names}); nearest flowline FTYPE {ftypes}. It names the nearest "
                f"flowline type per point but holds no line geometry, so it cannot give "
                f"distances from lobe vertices to canals", nearest_ftype=ftypes)
    else:
        rep.add(sec, str(river_p), "missing", "not found")

    shapefiles = sorted(nhd_dir.glob("*/NHDSnapshot/**/NHDFlowline.shp"))
    marks = extraction_markers(nhd_dir)
    if nhd_dir.exists():
        result = (f"{len(shapefiles)} NHDFlowline shapefiles under {nhd_dir}; extracted markers "
                  f"{marks or 'none'}")
    else:
        result = f"{nhd_dir} not found"
    rep.add(sec, "NHDPlus V2 download", "ok" if shapefiles else "missing", result, markers=marks)
    if not shapefiles:
        return

    footprint_ll = None
    if footprint is not None:
        footprint_ll = gpd.GeoSeries([footprint], crs=CRS_WORK).to_crs(CRS_LL).iloc[0]
        shapely.prepare(footprint_ll)
    total = dict(features=0, canal=0, canal_state=0, km_state=0.0)
    for shp in shapefiles:
        info, ftype, n_canal, n_in, km = count_canals(shp, footprint_ll)
        unit = shp.relative_to(nhd_dir).parts[0]
        rep.add(sec, f"NHDFlowline {unit}", "ok" if n_in else "info",
                f"{info['features']:,} flowlines ({info['crs']}); CanalDitch {n_canal:,}, of "
                f"which {n_in:,} ({km:,.0f} km) intersect the {rep.state} panel footprint; "
                f"FTYPE {ftype.value_counts().to_dict()}", file=str(shp),
                features=info["features"], canal=n_canal, canal_in_state=n_in,
                canal_km_in_state=km)
        total["features"] += info["features"]
        total["canal"] += n_canal
        total["canal_state"] += n_in
        total["km_state"] += km
    rep.add(sec, "CanalDitch flowlines in total", "ok" if total["canal_state"] else "missing",
            f"{total['canal_state']:,} CanalDitch flowlines ({total['km_state']:,.0f} km) "
            f"intersect the {rep.state} panel footprint, of {total['canal']:,} in the "
            f"{len(shapefiles)} units: the canal layer for the optional class (D6) can be cut "
            f"from these, no download needed", **total)


# =========================================================================
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--regions", type=Path, nargs="*", default=[],
                    help="aotil/regions.py *_points.parquet file(s)")
    ap.add_argument("--dataset", type=Path, help="default data/processed/dataset_<ST>.parquet")
    ap.add_argument("--idx", type=Path, help="default data/interim/prep/idx_<ST>.parquet")
    ap.add_argument("--haz", type=Path, help="default data/interim/prep/haz_<ST>_5070.parquet")
    ap.add_argument("--panels", type=Path, help="default data/interim/nfhl_states/pan_<ST>.parquet")
    ap.add_argument("--river", type=Path, help="default data/interim/hydro/river_<ST>.parquet")
    ap.add_argument("--nhdplus-dir", type=Path, default=NHD_DIR)
    ap.add_argument("--gis-dir", type=Path, default=GIS_DIR)
    ap.add_argument("--out", type=Path,
                    help="output stem; writes <stem>.json and <stem>.md "
                         "(default results/<today>/gis/check_gis_inputs_<ST>)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    st = args.state
    args.dataset = args.dataset or PROCESSED / f"dataset_{st}.parquet"
    args.idx = args.idx or PREP_DIR / f"idx_{st}.parquet"
    args.haz = args.haz or PREP_DIR / f"haz_{st}_5070.parquet"
    args.panels = args.panels or NFHL_DIR / f"pan_{st}.parquet"
    args.river = args.river or HYDRO_DIR / f"river_{st}.parquet"
    default_out = Path("results") / date.today().isoformat() / "gis" / f"check_gis_inputs_{st}"
    args.out = args.out or default_out
    if args.out.suffix in (".json", ".md"):
        args.out = args.out.with_suffix("")
    both_exist = args.out.with_suffix(".json").exists() and args.out.with_suffix(".md").exists()
    if both_exist and not args.overwrite:
        print(f"{args.out}.json/.md exist -- skipping (use --overwrite)")
        return 0

    t0 = time.time()
    print(f"[{st}] checking GIS inputs", flush=True)
    rep = Report(st)
    idx = load_index(args.idx) if args.idx.exists() else None
    if idx is None:
        rep.add(HAZ_SECTION, "idx file", "missing", f"{args.idx} not found")
    panels = check_panels(rep, args.panels)
    footprint = panels[1] if panels is not None else None
    extent = shapely.bounds(footprint) if footprint is not None else None
    haz = check_haz(rep, args.haz, idx, extent)
    check_dataset_keys(rep, args.dataset, idx)
    check_regions(rep, args.regions, args.dataset, idx, haz)
    check_layers(rep, [RAW_DIR, INTERIM_DIR, args.gis_dir], args.gis_dir, st)
    check_canals(rep, args.river, args.nhdplus_dir, footprint)

    regions = ";".join(str(p) for p in args.regions) or None
    inputs = dict(dataset=args.dataset, idx=args.idx, haz=args.haz, panels=args.panels,
                  river=args.river, nhdplus_dir=args.nhdplus_dir, gis_dir=args.gis_dir,
                  regions=regions)
    rep.write(args.out, inputs)
    verdicts = pd.Series([r["verdict"] for r in rep.rows]).value_counts().to_dict()
    summary = ", ".join(f"{SYMBOL[k]} {k} {v}" for k, v in verdicts.items())
    print(f"\n[{st}] {len(rep.rows)} checks: {summary}")
    print(f"wrote {args.out}.json and {args.out}.md  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
