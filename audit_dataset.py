#!/usr/bin/env python3
"""
audit_dataset.py -- read-only audit of one state's point table, before a model sees it.

Every number the paper reports about its data -- how many points, how the seed-0
tiles split them, what share of each test half is positive -- and every join the
boundary analysis makes from a point back to its FEMA polygon is read off
`data/processed/dataset_{ST}.parquet`. This script checks that table and, given
the previous build (`--v1`), how it moved (TASK-DATA-AUDIT.md Part B and A4). It
writes only under --out and changes no input. Every check ends in a verdict:

    OK      correct as built
    WRONG   missing or broken: fix before the table is used
    DECIDE  works, but needs a decision or a sentence in the paper

WHAT IT CHECKS, AND WHY THIS WAY
================================
1. THE SPLIT IS zoo.py's OWN.  Train / ID-test / OOD-test counts come from
   importing zoo.split_tiles and zoo.subset and repeating run_cell's four lines,
   so they are the counts of the cell that will be fitted. split_tiles sees the
   whole table, as in run_cell (on v1 that includes the shadedX rows).

2. BASE RATE.  Plain accuracy compares the ID and OOD halves only if their
   positive shares match; v1 had 0.53 against 0.64. A gap above 0.05 is DECIDE:
   O1 then has to be reported as AUC.

3. MISSINGNESS IS CHECKED BY LABEL, NOT ONLY OVERALL.  HGB and XGBoost route NaN
   natively, so a feature missing far more often for negatives than for positives
   is a label proxy. Measured on v1: 30% of Zone AE band negatives lack HAND and
   every StreamCat metric against 0.1% of AE positives, and 35% of them sit on
   NLCD open water -- the 5 km ring around coastal AE polygons reaches into bays
   that lie inside FIRM panels. Flagged when |missing(neg) - missing(pos)| > 5
   points within a zone. aspect_sin/cos are excused: pull_gee.derive sets them
   to NaN below 0.5 degrees of slope, so their missingness is flatness, a real
   terrain signal.

4. POLYGON JOINS ARE CHECKED ON GEOMETRY.  FLD_AR_ID alone is not unique within
   a state (39 rows of idx_FL share a (DFIRM_ID, FLD_AR_ID) key; 190 v1
   positives match 3-4 polygons by key). A positive is joined through
   `meta_src_row_id` when the build wrote it, else by (dfirm_id, meta_fld_ar_id),
   and the join counts only if the polygon has the positive's zone and contains
   the point. Geometry comes from haz_{ST}_5070.parquet through
   sample_points.read_geoms, one 10k-row group at a time, never the 5.4 GB file.

5. A BAND NEGATIVE'S ZONE.  It is the zone of the positive polygon whose 5 km
   ring produced it, while its 500-5,000 m distance is to the nearest SFHA of ANY
   zone (sample_points.build_state -> sample_band_negatives -> distances_to_sfha).
   So a "Zone A" negative's nearest floodplain can be Zone AE. For every band
   negative (or a sample) the nearest SFHA polygon is recomputed exactly: the
   candidates are the polygons
   whose bounding box lies within the stored meta_dist_to_sfha_m (+1 m) of the
   point -- a polygon's box is never farther than the polygon -- and the
   recomputed distance must reproduce the stored one.

6. v2 AGAINST v1 (A4), on the band subset (positives + band negatives) by
   domain x label. A move is flagged when |delta median| > 5% of the v1 IQR (the
   v1 standard deviation where the IQR is 0) or the missing share changes by
   more than 2 points.

7. MERIT CELLS.  At 100 m spacing two points can share one ~90 m MERIT Hydro
   cell. Pairs closer than 125 m (EPSG:5070) whose hand and log10_upa are both
   identical are counted; pairs 125-250 m apart give the rate of identical values
   that a shared cell cannot explain (flat plateaus, catchment-wide values).

8. WHICH PANEL A DATE CAME FROM.  FIRM panels of adjacent countywide DFIRMs
   overlap: on v1, 37% of points lie in two to four panels. The v1 build kept
   whichever containing panel the spatial join returned first, so 18,394 v1
   points (19%) carry the EFF_DATE of another DFIRM's panel rather than their
   own DFIRM's. sample_points.attach_panel_meta now takes the panel of the
   point's own DFIRM (the lowest FIRM_PAN if several), else the lowest FIRM_PAN
   of the other panels. B14 restates that rule instead of importing it, so a
   change on either side shows up as WRONG, and accepts the first-match rule
   only on a v1 table (no meta_src_row_id): a v2 table that follows it was built
   before attach_panel_meta changed.

9. "INSIDE THE STATE" IS A POLYGON TEST.  Panel sheets run past the state line,
   so a band negative can sit up to 5 km into Alabama or Georgia: inside the
   Florida bbox, with a Florida county_fips (its DFIRM's county, so B13 cannot
   see it), and 500-5,000 m from an SFHA measured against Florida's floodplains
   only. Measured against the Census county polygons (TIGERweb, 2026-10-14):
   1,038 v1 band negatives (4.6%) and 1,761 of 40,000 in a scratch v2 build
   (4.4%) lie outside Florida, all across the Alabama and Georgia land lines; a
   31 deg N bbox edge catches only 54 and 119 of them. B7 therefore tests every
   point against the state's TIGER/Line county polygons, which include coastal
   water, so points in the state's waters stay in. A band negative outside is
   DECIDE (sample_points.py --state-polygons masks them, off by default). A
   positive more than 100 m outside is WRONG; closer, the NFHL and TIGER draw the
   line differently (one v1 Zone A positive lies about 17 m across the Perdido
   River). Without a boundary file B7 is a bbox test and says so. B7 also states the
   datum: lon/lat are NAD83 (sample_points.CRS_SRC), not the brief's EPSG:4326.
   pyproj's 5070 -> 4326 transform applies a null datum shift, so no numeric
   comparison can tell the two apart; the ~1 m gap is reported, not measured.

OUTPUT (under --out)
    audit.md          paste-ready markdown: key numbers, summary, one section per check
    audit.json        the same, machine readable
    missingness.csv   per feature, overall and by domain x label
    shift_v1.csv      per feature x domain x label, v2 against v1 (with --v1)

USAGE (from the repo root; sample_points.read_geoms reads data/interim/prep/)
=====
    python audit_dataset.py --state FL --dataset data/processed/dataset_FL.parquet \\
        --v1 data/interim/archive/sampling-v1/data/processed/dataset_FL.parquet \\
        --out results/$D/data_audit
    python audit_dataset.py ... --nearest-sample 1000  # nearest SFHA on 1,000 per zone
    python audit_dataset.py ... --nearest-sample 0     # skip it (all: ~20 s on v1)
    python audit_dataset.py ... --counties data/raw/tiger/tl_2026_us_county.zip
    python audit_dataset.py ... --overwrite            # redo an existing audit

B7's state polygons default to data/interim/gis/counties_<ST>.parquet
(pull_gis_lines.py --layers counties), else the newest
data/raw/tiger/tl_*_us_county.zip; with neither, B7 is a bbox test only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import geopandas as gpd
import shapely
from shapely import STRtree
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aotil"))
import zoo                    # noqa: E402  split_tiles / subset, exactly as run_cell
import sample_points as sp    # noqa: E402  read_geoms, CRS and band constants

DATA = Path("data/processed")
OUT_ROOT = Path("results")
GIS_DIR = Path("data/interim/gis")      # pull_gis_lines.py: counties_<ST>.parquet
RAW_TIGER = Path("data/raw/tiger")      # pull_gis_lines.py: tl_<year>_us_county.zip

FIPS = {"VT": "50", "CO": "08", "AZ": "04", "MO": "29", "MN": "27",
        "IA": "19", "LA": "22", "NC": "37", "TX": "48", "FL": "12"}
# lon/lat extent of the state, padded; other states fall back to the panel extent.
# A sanity check that lon/lat are degrees in the right place, not a state test.
BBOX = {"FL": (-87.70, 24.30, -79.80, 31.10)}
BAND = "band"           # the protocol drawn from the panel mask, so it can cross the line

# Columns that identify a point or carry its split; none may be a feature.
ID_COLS = {"point_id", "state", "county_fips", "dfirm_id", "block_id", "domain",
           "label", "neg_protocol", "lon", "lat", "x", "y", "geometry"}
EXPECTED_NEW = {"meta_src_row_id", "meta_neg_rank"}   # v2 adds these to v1's columns

BASE_RATE_GAP = 0.05    # |pos share ID - pos share OOD| above this -> DECIDE
LABEL_MISS_GAP = 5.0    # points of missingness between negatives and positives
WATER_GAP = 5.0         # points of open-water share between negatives and positives
NEAR_CONST = 0.99       # top-value share above this -> near-constant
SHIFT_IQR = 0.05        # |delta median| / v1 IQR above this -> flagged move
SHIFT_MISS = 2.0        # change in missing share, points
DIST_TOL_M = 1.0        # distance agreement, metres
LONLAT_TOL_M = 1.0      # lon/lat against the reprojected geometry, metres
STATE_TOL_M = 100.0     # a positive closer than this outside the county line: line mismatch
NAN_BY_DESIGN = {"aspect_sin", "aspect_cos"}   # NaN below 0.5 deg slope
NLCD_WATER = 11
VERDICTS = ("OK", "WRONG", "DECIDE")
# B14: which containing FIRM panel a point's meta_eff_date comes from (note 8)
RULE_V2 = "own DFIRM's panel, then lowest FIRM_PAN (sample_points.attach_panel_meta)"
RULE_V1 = "first spatial-join match (the v1 build)"


# =========================================================================
# reporting
# =========================================================================
class Audit:
    """Collects checks as they run, prints each, and writes md + json at the end."""

    def __init__(self):
        self.items: list[dict] = []

    def add(self, key, title, verdict, result, how, tables=(), data=None):
        assert verdict in VERDICTS, verdict
        self.items.append(dict(key=key, title=title, verdict=verdict, result=result,
                               how=how, tables=list(tables), data=data or {}))
        print(f"\n[{verdict}] {key}. {title}\n  {result}", flush=True)
        for cap, t in tables:
            body = t.head(25).to_string(index=False)
            more = f"\n    ... {len(t) - 25} more rows in audit.md" if len(t) > 25 else ""
            print(f"  {cap}:\n    " + body.replace("\n", "\n    ") + more, flush=True)


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v.is_integer() and abs(v) < 1e12:
            return f"{int(v):,}"
        return f"{v:,.1f}" if abs(v) >= 100 else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v).replace("|", "\\|")


def md_table(t: pd.DataFrame) -> str:
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in t.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def clean(o):
    """Make numpy / pandas values JSON-safe; NaN becomes null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def write_outputs(au: Audit, out: Path, meta: dict, numbers: pd.DataFrame) -> None:
    n = {v: sum(i["verdict"] == v for i in au.items) for v in VERDICTS}
    md = [f"# Data audit: {meta['state']} point table", "",
          f"Generated {meta['generated']} by `audit_dataset.py` "
          f"({meta['runtime_s']:.0f} s).", "", "```", meta["command"], "```", "",
          "## Inputs", "", md_table(pd.DataFrame(meta["inputs"])), "",
          "## Key numbers", "", md_table(numbers), "",
          f"## Summary: {n['OK']} OK, {n['WRONG']} WRONG, {n['DECIDE']} DECIDE", "",
          md_table(pd.DataFrame([dict(item=i["key"], check=i["title"],
                                      verdict=i["verdict"], result=i["result"])
                                 for i in au.items])), ""]
    for i in au.items:
        md += [f"### {i['key']}. {i['title']}: {i['verdict']}", "",
               f"**Result.** {i['result']}", "", f"**How.** {i['how']}", ""]
        for cap, t in i["tables"]:
            md += [f"*{cap}*", "", md_table(t), ""]
    (out / "audit.md").write_text("\n".join(md))

    js = dict(meta, numbers=numbers, verdicts=n,
              items=[dict(i, tables={cap: t for cap, t in i["tables"]})
                     for i in au.items])
    (out / "audit.json").write_text(json.dumps(clean(js), indent=1))


# =========================================================================
# shared pieces
# =========================================================================
def read_features(path: Path) -> list[str]:
    """FEATURES.txt parsed exactly as zoo.load does, before its silent column filter."""
    toks = [l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return [f for tok in toks for f in tok.split()]


def band_subset(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Positives plus this protocol's negatives, both zones: the union of zoo.subset."""
    return pd.concat([zoo.subset(df, d, protocol) for d in ("A", "AE")])


def split_frames(df, protocol, train_dom, seed):
    """run_cell's split, line for line: the frames a cell is fitted and scored on."""
    eval_dom = "AE" if train_dom == "A" else "A"
    tr_tiles, te_tiles = zoo.split_tiles(df, seed)
    tr = zoo.subset(df, train_dom, protocol); tr = tr[tr[zoo.TILE].isin(tr_tiles)]
    id_te = zoo.subset(df, train_dom, protocol); id_te = id_te[id_te[zoo.TILE].isin(te_tiles)]
    ood_te = zoo.subset(df, eval_dom, protocol); ood_te = ood_te[ood_te[zoo.TILE].isin(te_tiles)]
    return tr, id_te, ood_te, tr_tiles, te_tiles


def strata(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["domain", "label", df.neg_protocol.fillna("-")]).size()


def key_numbers(df: pd.DataFrame, a) -> dict:
    """The data numbers the paper quotes, for one table."""
    out = {"rows": len(df), "band-subset rows": len(band_subset(df, a.protocol))}
    for (d, l, p), n in strata(df).items():
        out[f"{d} " + ("positives" if l == 1 else f"{p} negatives")] = int(n)
    tr, idt, ood, trt, tet = split_frames(df, a.protocol, a.train_dom, a.seed)
    out.update({
        f"seed-{a.seed} train ({a.train_dom})": len(tr),
        f"seed-{a.seed} ID test ({a.train_dom})": len(idt),
        f"seed-{a.seed} OOD test ({'AE' if a.train_dom == 'A' else 'A'})": len(ood),
        "tiles train / test": f"{len(trt)} / {len(tet)}",
        "positive share train": round(float(tr.label.mean()), 4),
        "positive share ID test": round(float(idt.label.mean()), 4),
        "positive share OOD test": round(float(ood.label.mean()), 4),
    })
    return out


def match_polygons(pts: pd.DataFrame, idx: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Candidate source polygons per point: (i, row_id) pairs, i = position in pts.

    One pair per point through meta_src_row_id; through (dfirm_id, meta_fld_ar_id)
    on a v1 table, where a key can match several polygons.
    """
    i = np.arange(len(pts))
    if "meta_src_row_id" in pts.columns:
        rid = pd.to_numeric(pts["meta_src_row_id"], errors="coerce").to_numpy(float)
        ok = np.isfinite(rid) & (rid >= 0) & (rid < len(idx))
        return pd.DataFrame({"i": i[ok], "row_id": rid[ok].astype(np.int64)}), \
            "meta_src_row_id"
    k = pd.DataFrame({"i": i, "DFIRM_ID": pts["dfirm_id"].to_numpy(),
                      "FLD_AR_ID": pts["meta_fld_ar_id"].to_numpy()})
    m = k.merge(idx[["DFIRM_ID", "FLD_AR_ID", "row_id"]], on=["DFIRM_ID", "FLD_AR_ID"])
    return m[["i", "row_id"]], "(dfirm_id, meta_fld_ar_id)"


def geoms_for(state: str, row_ids: np.ndarray) -> np.ndarray:
    """Geometry per row_id (None if absent), each polygon decoded once."""
    uniq = np.unique(row_ids)
    g = sp.read_geoms(state, uniq)
    return g[np.searchsorted(uniq, row_ids)]


def xy(df: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df.geometry.x.to_numpy(), df.geometry.y.to_numpy()


def find_counties(state: str) -> Path | None:
    """pull_gis_lines.py's county polygons, else the national TIGER zip it downloads."""
    p = GIS_DIR / f"counties_{state}.parquet"
    if p.exists():
        return p
    zips = sorted(RAW_TIGER.glob("tl_*_us_county.zip"))
    return zips[-1] if zips else None


def state_polygons(path: Path, state: str) -> gpd.GeoDataFrame:
    """The state's county polygons (STATEFP filter) in EPSG:5070."""
    fips = FIPS[state]
    if path.suffix == ".zip":
        import pyogrio
        g = pyogrio.read_dataframe(f"/vsizip/{path.resolve()}", columns=["STATEFP", "GEOID"],
                                   where=f"STATEFP = '{fips}'")
    else:
        g = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    if "STATEFP" in g.columns:
        g = g[g.STATEFP.astype(str) == fips]
    if not len(g):
        sys.exit(f"{path}: no polygons with STATEFP {fips}")
    return g.to_crs(sp.CRS_WORK).reset_index(drop=True)


# =========================================================================
# Part B: the table on its own
# =========================================================================
def check_counts(au, df, a):
    t = strata(df)
    piv = t.unstack("domain", fill_value=0)
    n_neg = int(round(a.per_domain * a.neg_ratio))
    want = {(1, "-"): a.per_domain, (0, a.protocol): n_neg}
    tab = piv.reset_index()
    tab["target per zone"] = [want.get((l, p), 0) for l, p in zip(tab.label, tab.neg_protocol)]
    both = {"A", "AE"} <= set(piv.columns)
    identical = both and bool((piv["A"] == piv["AE"]).all())
    exact = both and set(piv.index) == set(want) and all(
        int(piv.loc[k, d]) == v for k, v in want.items() for d in ("A", "AE"))
    extra = sorted(p for _, p in piv.index if p not in ("-", a.protocol))

    rank_note, rank_ok = "meta_neg_rank absent", True
    if "meta_neg_rank" in df.columns:
        neg = df[df.label == 0]
        bad = [k for k, g in neg.groupby(["domain", neg.neg_protocol.fillna("-")])
               if not np.array_equal(np.sort(g.meta_neg_rank.to_numpy()), np.arange(len(g)))]
        pos_ok = bool((df.loc[df.label == 1, "meta_neg_rank"] == -1).all())
        rank_ok = not bad and pos_ok
        rank_note = ("meta_neg_rank dense 0..n-1 per stratum, -1 on positives" if rank_ok
                     else f"meta_neg_rank broken in {bad or 'positives'}")
    verdict = "OK" if identical and exact and rank_ok else "WRONG"
    result = (f"{len(df):,} rows; A/AE strata identical: {'yes' if identical else 'NO'}; "
              f"exact v2 targets ({a.per_domain:,} positives + {n_neg:,} {a.protocol} "
              f"negatives per zone, nothing else): {'yes' if exact else 'NO'}"
              + (f" (extra protocols: {', '.join(extra)})" if extra else "")
              + f"; {rank_note}")
    au.add("B1", "Counts by domain x label x neg_protocol", verdict, result,
           "`df.groupby(['domain','label',neg_protocol.fillna('-')]).size()` against "
           "--per-domain and round(--per-domain * --neg-ratio).",
           [("rows per stratum", tab)], dict(strata={f"{d}/{l}/{p}": int(n)
                                                     for (d, l, p), n in t.items()}))


def check_split(au, df, a):
    tr, idt, ood, trt, tet = split_frames(df, a.protocol, a.train_dom, a.seed)
    eval_dom = "AE" if a.train_dom == "A" else "A"
    rows = []
    for name, f, dom, nt in (("train", tr, a.train_dom, len(trt)),
                             ("ID test", idt, a.train_dom, len(tet)),
                             ("OOD test", ood, eval_dom, len(tet))):
        rows.append(dict(half=name, domain=dom, tiles=nt, n=len(f),
                         positives=int(f.label.sum()), negatives=int((f.label == 0).sum()),
                         positive_share=round(float(f.label.mean()), 4) if len(f) else np.nan,
                         tiles_with_rows=f[zoo.TILE].nunique()))
    tab = pd.DataFrame(rows)
    gap = abs(rows[1]["positive_share"] - rows[2]["positive_share"])
    usable = len(tr) >= zoo.MIN_TRAIN and len(idt) and len(ood)
    verdict = ("WRONG" if not usable else "DECIDE" if gap > BASE_RATE_GAP else "OK")
    result = (f"train {len(tr):,} | ID test {len(idt):,} | OOD test {len(ood):,} | "
              f"tiles train {len(trt)} test {len(tet)}; positive share ID "
              f"{rows[1]['positive_share']:.2f} vs OOD {rows[2]['positive_share']:.2f} "
              f"(gap {gap:.3f}, threshold {BASE_RATE_GAP})"
              + ("" if usable else "; run_cell would return None"))
    au.add("B2", f"Seed-{a.seed} 50 km tile split ({a.protocol}, train {a.train_dom})",
           verdict, result,
           "imports `zoo.split_tiles(df, seed)` and `zoo.subset`, and repeats "
           "`zoo.run_cell`'s train / ID / OOD lines on the whole table.",
           [("halves", tab)], dict(halves=tab, gap=gap))


def check_missing(au, b, feats, out, protocol):
    present = [f for f in feats if f in b.columns]
    miss = b[present].isna().groupby([b.domain, b.label]).mean().mul(100).T
    miss.columns = [f"{d}/{'pos' if l == 1 else 'neg'} %" for d, l in miss.columns]
    tab = miss.copy()
    tab.insert(0, "all %", b[present].isna().mean().mul(100))
    gaps = {}
    for d in ("A", "AE"):
        if f"{d}/neg %" in tab and f"{d}/pos %" in tab:
            tab[f"{d} neg-pos"] = tab[f"{d}/neg %"] - tab[f"{d}/pos %"]
            gaps[d] = tab[f"{d} neg-pos"]
    tab = tab.reset_index(names="feature").round(2)
    tab.to_csv(out / "missingness.csv", index=False)

    empty = tab.loc[(tab.filter(like="%") >= 100).any(axis=1), "feature"].tolist()
    proxy = sorted({f for d, g in gaps.items() for f, v in g.items()
                    if abs(v) > LABEL_MISS_GAP and f not in NAN_BY_DESIGN})
    excused = sorted({f for d, g in gaps.items() for f, v in g.items()
                      if abs(v) > LABEL_MISS_GAP and f in NAN_BY_DESIGN})
    worst = {d: (g.drop(list(NAN_BY_DESIGN), errors="ignore").abs().idxmax(),
                 float(g.drop(list(NAN_BY_DESIGN), errors="ignore").abs().max()))
             for d, g in gaps.items()}
    any_miss = tab[tab["all %"] > 0]
    verdict = "WRONG" if empty else "DECIDE" if proxy else "OK"
    result = (f"{len(any_miss)}/{len(present)} features have missing values; "
              f"entirely missing in some stratum: {empty or 'none'}; "
              f"label-correlated (|neg - pos| > {LABEL_MISS_GAP:g} points within a zone): "
              f"{len(proxy)} features"
              + (f" (largest: " + ", ".join(f"{d} {f} {v:.1f}" for d, (f, v) in worst.items())
                 + ")" if proxy else "")
              + (f"; excused by design: {', '.join(excused)}" if excused else ""))
    au.add("B3", "Per-feature missingness, overall and by domain x label", verdict,
           result, f"`b[feats].isna()` on the {protocol} subset (positives + {protocol} "
           "negatives), grouped by domain x label; full table in missingness.csv.",
           [("features with any missing value (%)", any_miss)],
           dict(label_correlated=proxy, entirely_missing=empty))


def check_water(au, b, protocol):
    need = {"landcover", "gsw_occurrence", "hand"}
    if not need <= set(b.columns):
        au.add("B4", "Negatives on open water", "DECIDE",
               f"{sorted(need - set(b.columns))} not in the table; check skipped", "-")
        return
    w = pd.DataFrame({"domain": b.domain, "label": b.label,
                      "open_water": b.landcover.eq(NLCD_WATER),
                      "gsw_ge_50": b.gsw_occurrence.ge(50), "hand_missing": b.hand.isna()})
    tab = (w.groupby(["domain", "label"])[["open_water", "gsw_ge_50", "hand_missing"]]
           .mean().mul(100).round(2).reset_index())
    tab.columns = ["domain", "label", "NLCD open water %", "GSW occurrence >= 50 %",
                   "hand missing %"]
    hm = w[(w.label == 0) & w.hand_missing]
    on_water = hm.groupby("domain").open_water.mean().mul(100).round(1).to_dict()
    gap = {d: float(g.set_index("label").loc[0, "NLCD open water %"] -
                    g.set_index("label").loc[1, "NLCD open water %"])
           for d, g in tab.groupby("domain") if {0, 1} <= set(g.label)}
    verdict = "DECIDE" if any(v > WATER_GAP for v in gap.values()) else "OK"
    result = ("open-water share, negatives minus positives: "
              + ", ".join(f"{d} {v:+.1f} points" for d, v in gap.items())
              + f" (threshold {WATER_GAP:g}); of {protocol} negatives with no HAND, on "
              f"open water: " + ", ".join(f"{d} {v:.0f}%" for d, v in on_water.items()))
    au.add("B4", f"Negatives on open water ({protocol}; explains label-correlated missingness)",
           verdict, result,
           f"NLCD `landcover == {NLCD_WATER}` and `gsw_occurrence >= 50` by domain x "
           f"label on the {protocol} subset; a band negative on water is a point the ring "
           "reached inside a FIRM panel but off land.", [("shares (%)", tab)],
           dict(gap=gap, hand_missing_on_water=on_water))


def check_constant(au, b, feats):
    rows = []
    for f in feats:
        if f not in b.columns:
            continue
        r = dict(feature=f)
        for name, g in [("all", b)] + [(d, b[b.domain == d]) for d in ("A", "AE")]:
            s = g[f].dropna()
            vc = s.value_counts(normalize=True)
            r[f"{name} n_unique"] = int(s.nunique())
            r[f"{name} top share"] = float(vc.iloc[0]) if len(vc) else np.nan
            if name == "all":
                r["top value"] = vc.index[0] if len(vc) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    nun = t.filter(like="n_unique").min(axis=1)
    top = t.filter(like="top share").max(axis=1)
    const = t.loc[nun <= 1, "feature"].tolist()
    near = t.loc[(top > NEAR_CONST) & (nun > 1), "feature"].tolist()
    show = t[top > 0.9].round(4)
    verdict = "WRONG" if const else "DECIDE" if near else "OK"
    result = (f"constant (<= 1 unique value, overall or within a zone): {const or 'none'}; "
              f"near-constant (top value share > {NEAR_CONST}): {near or 'none'}; "
              f"largest top share {top.max():.3f} ({t.loc[top.idxmax(), 'feature']})")
    au.add("B5", "Constant and near-constant features", verdict, result,
           "value counts per feature on the band subset, overall and per zone.",
           [("features whose top value covers > 90% of a group", show)],
           dict(constant=const, near_constant=near))


def check_features_txt(au, feats, df, n_expect):
    dup = sorted({f for f in feats if feats.count(f) > 1})
    absent = [f for f in feats if f not in df.columns]
    nonnum = [f for f in feats if f in df.columns and not pd.api.types.is_numeric_dtype(df[f])]
    banned = [f for f in feats
              if f.startswith(sp.META_PREFIX) or f in ID_COLS or f.startswith("tile_")
              or f.endswith("_id") or "elev" in f.lower()]
    known = set(feats) | ID_COLS
    other = [c for c in df.columns if c not in known and not c.startswith(sp.META_PREFIX)
             and not c.startswith("tile_")]
    bad = len(feats) != n_expect or dup or absent or nonnum or banned
    verdict = "WRONG" if bad else "DECIDE" if other else "OK"
    result = (f"{len(feats)} entries (want {n_expect}); duplicates {dup or 'none'}; "
              f"absent from the table {absent or 'none'}; non-numeric {nonnum or 'none'}; "
              f"forbidden names {banned or 'none'}; columns that are neither a feature, "
              f"an identifier, a tile nor meta_: {other or 'none'}")
    au.add("B6", "FEATURES.txt hygiene", verdict, result,
           "parsed as `zoo.load` does; forbidden: `meta_*`, `tile_*`, `*_id`, any name "
           "containing `elev`, and " + ", ".join(f"`{c}`" for c in sorted(ID_COLS)) + ".",
           data=dict(n=len(feats), absent=absent, banned=banned, unclassified=other))


def outside_state(df: gpd.GeoDataFrame, cn: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per point: outside every county polygon, and how far outside (m; 0 inside).

    One prepared polygon at a time over the points in its bbox, so the state is
    never unioned; a point on a county line counts as inside.
    """
    x, y = xy(df.to_crs(cn.crs))
    inside = np.zeros(len(df), dtype=bool)
    polys = np.asarray(cn.geometry.values)
    for g in polys:
        shapely.prepare(g)
        x0, y0, x1, y1 = g.bounds
        m = ~inside & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        inside[m] = shapely.intersects_xy(g, x[m], y[m])
    dist = np.zeros(len(df))
    if (~inside).any():
        pts = shapely.points(x[~inside], y[~inside])
        k, d = STRtree(polys).query_nearest(pts, return_distance=True, all_matches=False)
        dist[np.nonzero(~inside)[0][k[0]]] = d
    return ~inside, dist


def check_lonlat(au, df, pan, state, cn, cn_path):
    epsg = df.crs.to_epsg() if df.crs is not None else None
    lon, lat = df.lon.to_numpy(float), df.lat.to_numpy(float)
    finite = np.isfinite(lon) & np.isfinite(lat)
    x, y = xy(df)
    lo, la = Transformer.from_crs(epsg or sp.CRS_WORK, sp.CRS_SRC,
                                  always_xy=True).transform(x, y)
    dev = float(np.nanmax(np.hypot((lon - lo) * 111_320 * np.cos(np.deg2rad(lat)),
                                   (lat - la) * 110_574)))
    ext = tuple(pan.to_crs(sp.CRS_SRC).total_bounds) if pan is not None else None
    x0, y0, x1, y1 = BBOX.get(state) or ext or (-180, -90, 180, 90)
    out_bbox = int((~((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))).sum())
    out_pan = None
    if ext is not None:
        p0, q0, p1, q1 = ext
        out_pan = int((~((lon >= p0) & (lon <= p1) & (lat >= q0) & (lat <= q1))).sum())
    cf = df.county_fips.dropna()
    bad_fips = int((cf.str[:2] != FIPS[state]).sum())

    wrong, decide, tabs = [], [], []
    if epsg != sp.CRS_WORK:
        wrong.append(f"geometry CRS EPSG:{epsg}, want {sp.CRS_WORK}")
    if not finite.all():
        wrong.append(f"{int((~finite).sum())} non-finite lon/lat")
    if out_bbox or out_pan or bad_fips:
        wrong.append(f"{out_bbox} outside the {state} bbox, {out_pan} outside the FIRM panel "
                     f"extent, {bad_fips} county_fips not starting '{FIPS[state]}'")
    if not dev < LONLAT_TOL_M:
        wrong.append(f"lon/lat up to {dev:.3f} m from the geometry")
    if sp.CRS_SRC != 4326:
        decide.append(f"lon/lat are {CRS.from_epsg(sp.CRS_SRC).name} (EPSG:{sp.CRS_SRC}, "
                      "sample_points.CRS_SRC), not EPSG:4326 as the brief asks; the two differ "
                      "by about 1 m in Florida, and pyproj's EPSG:5070 -> 4326 transform applies "
                      "no datum shift, so no comparison here can tell them apart")

    n_out = n_band_out = None
    if cn is None:
        decide.append(f"bbox only, not a state test: no county polygons at "
                      f"{GIS_DIR}/counties_{state}.parquet or {RAW_TIGER}/tl_*_us_county.zip "
                      f"(pull_gis_lines.py --state {state} --layers counties)")
    else:
        out, dist = outside_state(df, cn)
        n_out = int(out.sum())
        prot = df.neg_protocol.fillna("-").to_numpy()
        st = pd.DataFrame({"domain": df.domain.to_numpy(), "label": df.label.to_numpy(),
                           "neg_protocol": prot, "outside_state": out,
                           "max_m_outside": np.where(out, dist, np.nan)})
        tab = (st.groupby(["domain", "label", "neg_protocol"])
               .agg(rows=("outside_state", "size"), outside_state=("outside_state", "sum"),
                    max_m_outside=("max_m_outside", "max")).reset_index())
        tabs = [(f"points outside the {len(cn)} {state} county polygons", tab)]
        band = out & (prot == BAND)
        n_band_out = int(band.sum())
        from_polys = out & (prot != BAND)   # drawn inside this state's own NFHL polygons
        for far, m in ((True, from_polys & (dist > STATE_TOL_M)),
                       (False, from_polys & (dist <= STATE_TOL_M))):
            if not m.any():
                continue
            g = st[m].groupby(["domain", "label"]).size()
            by = ", ".join(f"{d}/{l}: {n}" for (d, l), n in g.items())
            msg = (f"points drawn from {state}'s own polygons that lie outside {state}: "
                   f"{int(m.sum()):,} ({by}; up to {dist[m].max():.0f} m outside)")
            if far:
                wrong.append(msg)
            else:
                decide.append(msg + f", within {STATE_TOL_M:g} m: the NFHL and TIGER/Line "
                              "draw the state line differently there, not a sampling error")
        if band.any():
            # sample_points reads --state-polygons without a STATEFP filter, so point it at
            # pull_gis_lines.py's in-state file, never at the national TIGER zip.
            by = ", ".join(f"{d}: {n}" for d, n in st[band].groupby("domain").size().items())
            share = 100 * band.sum() / max((prot == BAND).sum(), 1)
            decide.append(f"{int(band.sum()):,} {BAND} negatives ({share:.1f}%) lie outside "
                          f"{state} ({by}; up to {dist[band].max():.0f} m outside): FIRM "
                          "panel sheets run past the state line and the neighbouring "
                          "state's floodplains are not in this NFHL, so their band distance "
                          "is measured against half the map; keep them, or rebuild with "
                          f"`sample_points.py --state-polygons {GIS_DIR}/counties_{state}"
                          f".parquet` (written by pull_gis_lines.py --state {state} "
                          "--layers counties)")

    verdict = "WRONG" if wrong else "DECIDE" if decide else "OK"
    state_test = (f"{n_out:,} of {len(df):,} points outside the {len(cn)} {state} county "
                  f"polygons of {cn_path}; " if cn is not None else "")
    result = ("".join(f"WRONG: {w}. " for w in wrong)
              + "".join(f"DECIDE: {d}. " for d in decide)
              + f"{state_test}geometry CRS EPSG:{epsg}; lon {np.nanmin(lon):.3f} to "
              f"{np.nanmax(lon):.3f}, lat {np.nanmin(lat):.3f} to {np.nanmax(lat):.3f}; "
              f"{out_bbox} outside the {state} bbox {(x0, y0, x1, y1)}; {out_pan} outside "
              f"the FIRM panel extent; {bad_fips} county_fips not starting '{FIPS[state]}'; "
              f"lon/lat vs the geometry reprojected to EPSG:{sp.CRS_SRC}: max {dev:.3f} m")
    state_how = (f"each point tested against the {state} county polygons (STATEFP "
                 f"{FIPS[state]}; TIGER/Line includes coastal water) with "
                 "`shapely.intersects_xy`, and its distance to the nearest one where it lies "
                 "outside. Points drawn inside the state's own NFHL polygons must be inside "
                 f"(WRONG beyond {STATE_TOL_M:g} m); {BAND} negatives come from the panel "
                 "mask and may not be."
                 if cn is not None else
                 "no county polygons, so no state test (--counties, or pull_gis_lines.py "
                 "--layers counties).")
    au.add("B7", f"lon/lat are degrees, their datum, and inside {state}", verdict, result,
           f"geometry (EPSG:5070) reprojected to EPSG:{sp.CRS_SRC} with pyproj and compared "
           "to lon/lat; the padded bbox and the FIRM panel extent as a degrees check; "
           + state_how, tabs,
           dict(max_dev_m=dev, lonlat_epsg=sp.CRS_SRC, outside_bbox=out_bbox,
                outside_panels=out_pan, outside_state=n_out, outside_state_band=n_band_out,
                state_polygons=str(cn_path) if cn is not None else None))


def check_point_id(au, df, state):
    pid = df.point_id
    pat = re.compile(rf"^{state}\d{{7}}$")
    bad = int((~pid.fillna("").map(lambda s: bool(pat.match(s)))).sum())
    ok = pid.notna().all() and pid.is_unique and bad == 0
    au.add("B8", "point_id unique", "OK" if ok else "WRONG",
           f"{pid.nunique():,} unique of {len(pid):,}; {int(pid.isna().sum())} null; "
           f"{bad} not of the form {state}NNNNNNN",
           "`df.point_id.is_unique` and a format check.")


def check_spatial_keys(au, df):
    x, y = xy(df)
    bm = sp.DEFAULTS["block_km"] * 1000.0
    bx, by = np.floor(x / bm).astype(np.int64), np.floor(y / bm).astype(np.int64)
    block = pd.Series(bx.astype(str), index=df.index) + "_" + by.astype(str)
    tile = pd.Series((bx // 5).astype(str), index=df.index) + "_" + (by // 5).astype(str)
    bad_b = int((block != df.block_id).sum())
    bad_t = int((tile != df.tile_50km).sum()) if "tile_50km" in df else len(df)
    ok = bad_b == 0 and bad_t == 0
    au.add("B9", "block_id and tile_50km agree with the geometry", "OK" if ok else "WRONG",
           f"{bad_b} block_id and {bad_t} tile_50km values differ from the geometry; "
           f"{df.block_id.nunique():,} blocks, {df.tile_50km.nunique():,} 50 km tiles",
           "block = floor(x / 10 km)_floor(y / 10 km) in EPSG:5070 "
           "(sample_points.build_state); tile_50km = block // 5 (pull_gee.derive).")


def check_invariants(au, df, a):
    pts = np.asarray(df.geometry.values)
    tree = STRtree(pts)
    i, j = tree.query(pts, predicate="dwithin", distance=a.min_sep)
    k = i < j
    close = int((shapely.distance(pts[i[k]], pts[j[k]]) < a.min_sep).sum())
    jj, dist = tree.query_nearest(pts, exclusive=True, return_distance=True)
    nearest = np.full(len(pts), np.inf)
    np.minimum.at(nearest, jj[0], dist)
    pos = df[df.label == 1]
    pos_zero = float((pos.meta_dist_to_sfha_m == 0).mean())
    bd = df.loc[df.neg_protocol == a.protocol, "meta_dist_to_sfha_m"]
    bmin, bmax = sp.DEFAULTS["band_min_m"], sp.DEFAULTS["band_max_m"]
    off = int(((bd < bmin) | (bd > bmax) | bd.isna()).sum())
    ok = close == 0 and pos_zero == 1.0 and off == 0
    au.add("B10", "Sampling invariants carried into the table", "OK" if ok else "WRONG",
           f"pairs closer than {a.min_sep:g} m: {close}; minimum nearest-neighbour "
           f"distance {nearest.min():.2f} m, median {np.median(nearest):.0f} m; positives "
           f"at distance 0 from an SFHA {100 * pos_zero:.1f}%; {a.protocol} negatives "
           f"{bd.min():.0f}-{bd.max():.0f} m from an SFHA, {off} outside "
           f"{bmin:.0f}-{bmax:.0f} m",
           "STRtree over the EPSG:5070 points (as `sample_points.audit`) and "
           "`meta_dist_to_sfha_m`; repeated here so a merge cannot have broken them.",
           data=dict(min_nn_m=float(nearest.min()), close_pairs=close))


def check_positive_polygons(au, df, idx, state):
    pos = df[df.label == 1]
    x, y = xy(pos)
    null_id = int(pos.meta_fld_ar_id.isna().sum())
    cand, how = match_polygons(pos, idx)
    rid = cand.row_id.to_numpy()
    att = idx.iloc[rid]
    pi = cand.i.to_numpy()
    zone_ok = att.FLD_ZONE.to_numpy() == pos.domain.to_numpy()[pi]
    key_ok = ((att.FLD_AR_ID.to_numpy() == pos.meta_fld_ar_id.to_numpy()[pi]) &
              (att.DFIRM_ID.to_numpy() == pos.dfirm_id.to_numpy()[pi]))
    sfha_ok = att.SFHA_TF.to_numpy() == "T"
    t0 = time.time()
    g = geoms_for(state, rid)
    have = np.array([v is not None for v in g])
    inside = np.zeros(len(rid), dtype=bool)
    inside[have] = shapely.contains_xy(g[have], x[pi[have]], y[pi[have]])
    good = zone_ok & key_ok & sfha_ok & inside
    n_cand = np.bincount(pi, minlength=len(pos))
    n_good = np.bincount(pi, weights=good, minlength=len(pos))
    rows = []
    for d in ("A", "AE"):
        m = (pos.domain == d).to_numpy()
        rows.append(dict(domain=d, positives=int(m.sum()),
                         null_fld_ar_id=int(pos.meta_fld_ar_id[m].isna().sum()),
                         no_match=int((n_cand[m] == 0).sum()),
                         several_by_key=int((n_cand[m] > 1).sum()),
                         resolved=int((n_good[m] == 1).sum()),
                         still_ambiguous=int((n_good[m] > 1).sum()),
                         unresolved=int((n_good[m] == 0).sum())))
    tab = pd.DataFrame(rows)
    ok = null_id == 0 and int(tab.unresolved.sum()) == 0 and int(tab.still_ambiguous.sum()) == 0
    result = (f"{len(pos):,} positives joined via {how}: {int(tab.resolved.sum()):,} resolve to "
              f"exactly one polygon of their zone that contains them; "
              f"{null_id} null meta_fld_ar_id; {int(tab.no_match.sum())} without a match; "
              f"{int(tab.several_by_key.sum())} matched several polygons by key, "
              f"{int(tab.still_ambiguous.sum())} still ambiguous after containment; "
              f"{int((~key_ok).sum())} FLD_AR_ID/DFIRM_ID disagreements, "
              f"{int((~zone_ok).sum())} zone disagreements, {int((~have).sum())} polygons "
              f"without geometry ({time.time() - t0:.0f} s of geometry)")
    au.add("B11", "meta_fld_ar_id joins every positive to its polygon (idx + haz geometry)",
           "OK" if ok else "WRONG", result,
           f"join to idx_{state} via {how}; a candidate counts if its FLD_ZONE equals the "
           "positive's domain, its FLD_AR_ID / DFIRM_ID equal the point's, SFHA_TF == 'T', "
           f"and its haz_{state}_5070 geometry contains the point "
           "(`sample_points.read_geoms` + `shapely.contains_xy`).",
           [("positives", tab)], dict(method=how))


def check_fld_zone(au, df, a):
    pos = df[df.label == 1]
    neg = df[df.neg_protocol == a.protocol]
    bad = int((pos.meta_fld_zone.fillna("<null>") != pos.domain).sum())
    neg_set = int(neg.meta_fld_zone.notna().sum())
    tab = (pos.groupby(["domain", pos.meta_fld_zone.fillna("<null>")]).size()
           .rename("positives").reset_index())
    ok = bad == 0 and neg_set == 0
    au.add("B12", "meta_fld_zone consistent with domain", "OK" if ok else "WRONG",
           f"{bad} positives whose meta_fld_zone differs from domain; {neg_set} "
           f"{a.protocol} negatives with a meta_fld_zone (want 0: a band negative carries "
           "no polygon attributes, its zone is `domain`)",
           "`meta_fld_zone == domain` on positives.", [("positives", tab)])


def check_county(au, df, state):
    cf = df.county_fips
    tail = df.dfirm_id.fillna("").str[-1]
    countywide = tail.str.isalpha().to_numpy()
    mism = int((countywide & (cf.fillna("") != df.dfirm_id.fillna("").str[:5]).to_numpy()).sum())
    pre = int((cf.dropna().str[:2] != FIPS[state]).sum())
    tab = (pd.DataFrame({"label": df.label, "null": cf.isna(),
                         "community_dfirm": ~countywide})
           .groupby("label")[["null", "community_dfirm"]].sum().reset_index())
    tab.insert(1, "rows", df.groupby("label").size().to_numpy())
    nulls = int(cf.isna().sum())
    verdict = "WRONG" if mism or pre else "DECIDE" if nulls else "OK"
    au.add("B13", "county_fips populated", verdict,
           f"{nulls} null ({int((~countywide).sum())} rows from community-based DFIRMs, "
           f"which keep a null by design); {mism} countywide rows where county_fips != "
           f"dfirm_id[:5]; {pre} not starting '{FIPS[state]}'; "
           f"{cf.nunique()} counties",
           "`county_fips` against `dfirm_id[:5]` for countywide DFIRMs (id ends in a letter).",
           [("by label", tab)])


def check_eff_date(au, df, pan, a):
    """meta_eff_date against the panels, including which county's panel it came from.

    Panels of adjacent countywide DFIRMs overlap -- measured on v1, 37% of points
    lie in two to four panels, always of different DFIRMs -- so the date depends
    on a rule (docstring note 8). The expected date is the build's current rule:
    the containing panel of the point's own DFIRM with the lowest FIRM_PAN, else
    the lowest FIRM_PAN of the other containing panels, ordered exactly as
    attach_panel_meta orders the join. The v1 rule, the first panel the join
    returns, is accepted only on a v1 table. Separately, a point that lies in a
    panel of its own DFIRM but carries none of those panels' dates is counted.
    """
    if pan is None:
        nul = df.groupby("label").meta_eff_date.apply(lambda s: int(s.isna().sum()))
        au.add("B14", "meta_eff_date populated", "DECIDE" if nul.sum() else "OK",
               f"nulls by label {nul.to_dict()}; no --panels, so not traced", "-")
        return
    eff = pd.to_datetime(pan.EFF_DATE)
    if eff.dt.tz is not None:
        eff = eff.dt.tz_convert(None)
    cols = ["DFIRM_ID"] + (["FIRM_PAN"] if "FIRM_PAN" in pan.columns else [])
    pts = gpd.GeoDataFrame({"_i": np.arange(len(df))}, geometry=df.geometry.values,
                           crs=df.crs)
    j = gpd.sjoin(pts, pan[cols + ["geometry"]].assign(EFF_DATE=eff), how="left",
                  predicate="within")
    i = j._i.to_numpy()
    hit = j.index_right.notna().to_numpy()
    own = hit & (j.DFIRM_ID.to_numpy() == df.dfirm_id.to_numpy()[i])
    n_in = np.bincount(i[hit], minlength=len(df))
    n_own = np.bincount(i[own], minlength=len(df))
    by = ["_own"] + cols[1:]
    ranked = j.assign(_own=own).sort_values(by, ascending=[False] + [True] * (len(by) - 1),
                                            kind="stable")
    rows = np.arange(len(df))
    expect = {rule: t[~t.index.duplicated(keep="first")].set_index("_i").EFF_DATE
              .reindex(rows).to_numpy()
              for rule, t in ((RULE_V2, ranked), (RULE_V1, j))}
    mine = pd.to_datetime(df.meta_eff_date).to_numpy()
    null = pd.isna(mine)
    miss = {rule: ~((null & pd.isna(d)) | (mine == d)) for rule, d in expect.items()}
    jd = j.EFF_DATE.to_numpy()
    same_own = own & ((null[i] & pd.isna(jd)) | (mine[i] == jd))
    other = (n_own > 0) & (np.bincount(i[same_own], minlength=len(df)) == 0)

    v1_table = "meta_src_row_id" not in df.columns
    if not miss[RULE_V2].any():
        rule, rule_note = RULE_V2, "reproduced for every point"
    elif not miss[RULE_V1].any() and v1_table:
        rule, rule_note = RULE_V1, (
            "reproduced for every point, as a v1 table should; the current rule would "
            f"change {int(miss[RULE_V2].sum()):,} dates")
    elif not miss[RULE_V1].any():
        rule, rule_note = None, (
            f"this v2 table (it has meta_src_row_id) follows the {RULE_V1}, and "
            f"{int(miss[RULE_V2].sum()):,} dates differ from the current rule: its points "
            "were built before attach_panel_meta changed; rebuild them")
    else:
        rule, rule_note = None, (
            f"meta_eff_date reproduces neither rule: {int(miss[RULE_V2].sum()):,} points "
            f"differ from the {RULE_V2}, {int(miss[RULE_V1].sum()):,} from the {RULE_V1}; "
            "compare attach_panel_meta with this check")
    d_exp = expect[rule or RULE_V2]
    stray = int((null & (n_in > 0) & pd.notna(d_exp)).sum())

    prot = df.neg_protocol.fillna("-").to_numpy()
    band_out = int(((prot == a.protocol) & (n_in == 0)).sum())
    st = pd.DataFrame({"domain": df.domain.to_numpy(), "label": df.label.to_numpy(),
                       "null_eff_date": null, "in_no_panel": n_in == 0,
                       "in_several_panels": n_in > 1, "no_own_dfirm_panel": n_own == 0,
                       "date_not_own_dfirm": other})
    tab = st.groupby(["domain", "label"]).sum().reset_index()
    tab.insert(2, "rows", st.groupby(["domain", "label"]).size().to_numpy())
    by_label = ", ".join(f"label {l}: {int(n)}"
                         for l, n in st.groupby("label").null_eff_date.sum().items())
    verdict = ("WRONG" if band_out or rule is None else
               "DECIDE" if null.any() or other.any() else "OK")
    au.add("B14", "meta_eff_date populated, traced to the FIRM panels", verdict,
           f"rule: {rule or 'NONE'}, {rule_note}; null meta_eff_date: {int(null.sum())} "
           f"({by_label}), {int((null & (n_in == 0)).sum())} of them in no panel and "
           f"{int((null & (n_in > 0) & pd.isna(d_exp)).sum())} where "
           f"{'that rule' if rule else 'the current rule'} picks a panel whose EFF_DATE is null"
           + (f", {stray} where it picks a dated panel" if stray else "")
           + f"; {band_out} {a.protocol} negatives outside every panel; "
           f"{int((n_in > 1).sum()):,} points lie in several panels, "
           f"{int((n_own == 0).sum()):,} in no panel of their own DFIRM, and "
           f"{int(other.sum()):,} lie in a panel of their own DFIRM but carry another DFIRM's "
           f"panel date",
           "`gpd.sjoin(points, pan, predicate='within')`; expected date: the joined rows "
           "sorted by (DFIRM_ID == the point's `dfirm_id` first, then FIRM_PAN ascending, "
           "stable), first row per point, as `sample_points.attach_panel_meta` does; the v1 "
           "build's unsorted first match is accepted only on a table without "
           "`meta_src_row_id`.", [("by domain x label", tab)],
           dict(rule=rule, mismatch_v2_rule=int(miss[RULE_V2].sum()),
                mismatch_v1_rule=int(miss[RULE_V1].sum()),
                several_panels=int((n_in > 1).sum()), not_own_dfirm=int(other.sum())))


def check_band_source(au, df, idx, state, a):
    neg = df[df.neg_protocol == a.protocol]
    if "meta_src_row_id" not in neg.columns:
        au.add("B15", "A band negative's zone is its ring polygon's zone", "DECIDE",
               "meta_src_row_id absent (v1 table): the ring polygon of a band negative is "
               "not recorded, so its zone can only be read from `domain`",
               "needs `meta_src_row_id` (sample_points v2).")
        return
    rid = pd.to_numeric(neg.meta_src_row_id, errors="coerce").to_numpy(float)
    okr = np.isfinite(rid) & (rid >= 0) & (rid < len(idx))
    ri = rid[okr].astype(np.int64)
    att = idx.iloc[ri]
    sub = neg[okr]
    zone_ok = att.FLD_ZONE.to_numpy() == sub.domain.to_numpy()
    dfirm_ok = att.DFIRM_ID.to_numpy() == sub.dfirm_id.to_numpy()
    g = geoms_for(state, ri)
    have = np.array([v is not None for v in g])
    d = np.full(len(ri), np.nan)
    d[have] = shapely.distance(np.asarray(sub.geometry.values)[have], g[have])
    hi = sp.DEFAULTS["band_max_m"] + 2 * sp.RING_SIMPLIFY_M
    off = int(((d < sp.DEFAULTS["band_min_m"] - DIST_TOL_M) | (d > hi)).sum())
    bad = int((~okr).sum()) + int((~zone_ok).sum()) + int((~dfirm_ok).sum())
    verdict = "WRONG" if bad else "DECIDE" if off else "OK"
    au.add("B15", "A band negative's zone is its ring polygon's zone", verdict,
           f"{len(neg):,} {a.protocol} negatives: {int((~okr).sum())} without a valid "
           f"meta_src_row_id, {int((~zone_ok).sum())} whose ring polygon's FLD_ZONE differs "
           f"from domain, {int((~dfirm_ok).sum())} whose dfirm_id differs from it; distance "
           f"to the ring polygon {np.nanmin(d):.0f}-{np.nanmax(d):.0f} m, {off} outside "
           f"{sp.DEFAULTS['band_min_m']:.0f}-{hi:.0f} m; {len(np.unique(ri)):,} ring polygons",
           "`idx.iloc[meta_src_row_id]`: FLD_ZONE == domain, DFIRM_ID == dfirm_id, and the "
           "point's distance to that polygon within the band plus the 2 x 100 m ring "
           "simplification slack.")


def check_nearest_zone(au, df, idx, state, a):
    if a.nearest_sample == 0:
        return
    neg = df[df.neg_protocol == a.protocol]
    rng = np.random.default_rng(a.seed)
    take = []
    for d in ("A", "AE"):
        ii = np.nonzero((neg.domain == d).to_numpy())[0]
        n = len(ii) if a.nearest_sample < 0 else min(a.nearest_sample, len(ii))
        take.append(np.sort(rng.choice(ii, n, replace=False)))
    s = neg.iloc[np.concatenate(take)]
    pts = np.asarray(s.geometry.values)
    d0 = s.meta_dist_to_sfha_m.to_numpy(float)
    d0 = np.where(np.isfinite(d0), d0, sp.DEFAULTS["band_max_m"])

    t0 = time.time()
    sfha = idx[idx.SFHA_TF == "T"]
    boxes = shapely.box(sfha.minx.to_numpy(), sfha.miny.to_numpy(),
                        sfha.maxx.to_numpy(), sfha.maxy.to_numpy())
    pi, bj = STRtree(boxes).query(pts, predicate="dwithin", distance=d0 + DIST_TOL_M)
    rid = sfha.row_id.to_numpy()[bj]
    zone = sfha.FLD_ZONE.to_numpy()[bj]
    g = geoms_for(state, rid)
    dist = shapely.distance(pts[pi], g)            # NaN where a geometry is missing
    dist = np.where(np.isfinite(dist), dist, np.inf)
    dmin = np.full(len(pts), np.inf)
    np.minimum.at(dmin, pi, dist)
    o = np.lexsort((dist, pi))
    first = o[np.r_[True, pi[o][1:] != pi[o][:-1]]]
    near_rid = np.full(len(pts), -1, dtype=np.int64)
    near_zone = np.full(len(pts), "<none>", dtype=object)
    near_rid[pi[first]] = rid[first]
    near_zone[pi[first]] = zone[first]
    tied = dist <= dmin[pi] + DIST_TOL_M
    zsets = pd.Series(zone[tied]).groupby(pi[tied]).agg(lambda z: "/".join(sorted(set(z))))
    label = near_zone.copy()
    multi = zsets[zsets.str.contains("/")]
    label[multi.index.to_numpy()] = "tie " + multi.to_numpy()

    agree = np.abs(dmin - s.meta_dist_to_sfha_m.to_numpy(float)) <= DIST_TOL_M
    tab = pd.crosstab(s.domain.to_numpy(), label, rownames=["negative's zone"],
                      colnames=["nearest SFHA"]).reset_index()
    share = {}
    for d in ("A", "AE"):
        m = (s.domain == d).to_numpy()
        other = "AE" if d == "A" else "A"
        share[d] = dict(n=int(m.sum()),
                        nearest_other_zone=float((label[m] == other).mean()),
                        nearest_own_zone=float((label[m] == d).mean()),
                        nearest_other_sfha=float(np.isin(label[m], ["AH", "AO", "VE"]).mean()),
                        tie=float(pd.Series(label[m]).str.startswith("tie").mean()))
    own = ""
    if "meta_src_row_id" in s.columns:
        same = near_rid == s.meta_src_row_id.to_numpy()
        own = f"; nearest polygon is the negative's own ring polygon for {100 * same.mean():.1f}%"
    verdict = "OK" if agree.all() else "WRONG"
    which = ("all" if a.nearest_sample < 0 or len(s) == len(neg)
             else f"a seed-{a.seed} sample of {a.nearest_sample:,} per zone:")
    result = (f"{which} {len(s):,} {a.protocol} negatives; recomputed "
              f"nearest-SFHA distance reproduces meta_dist_to_sfha_m within "
              f"{DIST_TOL_M:g} m for {int(agree.sum()):,}/{len(s):,}; Zone A negatives whose "
              f"nearest SFHA is Zone AE {100 * share['A']['nearest_other_zone']:.1f}%, Zone AE "
              f"negatives whose nearest SFHA is Zone A "
              f"{100 * share['AE']['nearest_other_zone']:.1f}% (AH/AO/VE: A "
              f"{100 * share['A']['nearest_other_sfha']:.1f}%, AE "
              f"{100 * share['AE']['nearest_other_sfha']:.1f}%; A/AE ties: A "
              f"{100 * share['A']['tie']:.1f}%, AE {100 * share['AE']['tie']:.1f}%){own} "
              f"({len(np.unique(rid)):,} candidate polygons, {time.time() - t0:.0f} s)")
    au.add("B16", "Nearest SFHA zone of band negatives", verdict, result,
           "for each sampled negative, every SFHA polygon (SFHA_TF == 'T', any zone) whose "
           "idx bounding box is within meta_dist_to_sfha_m + 1 m; exact "
           "`shapely.distance` to those geometries (read with `sample_points.read_geoms`); "
           "the nearest one's FLD_ZONE, with polygons tied within 1 m reported as ties.",
           [("negatives by nearest SFHA zone", tab)], dict(shares=share))


# =========================================================================
# Part A4: against v1
# =========================================================================
def compare_columns(au, df, v1, feats):
    c2, c1 = list(df.columns), list(v1.columns)
    added = [c for c in c2 if c not in c1]
    removed = [c for c in c1 if c not in c2]
    f2 = [f for f in feats if f not in c2]
    f1 = [f for f in feats if f not in c1]
    dt = [f"{c}: {v1[c].dtype} -> {df[c].dtype}" for c in c2
          if c in c1 and str(df[c].dtype) != str(v1[c].dtype)]
    unexpected = [c for c in added if c not in EXPECTED_NEW]
    verdict = ("WRONG" if removed or f2 or f1 else
               "DECIDE" if unexpected or dt else "OK")
    au.add("A4.1", "Columns against v1", verdict,
           f"{len(c2)} columns (v1 {len(c1)}); added {added or 'none'} (expected "
           f"{sorted(EXPECTED_NEW)}); removed {removed or 'none'}; features missing here "
           f"{f2 or 'none'}, in v1 {f1 or 'none'}; dtype changes {dt or 'none'}",
           "set difference of the column lists; v2 should add only meta_src_row_id and "
           "meta_neg_rank.", data=dict(added=added, removed=removed, dtype_changes=dt))


def strata_stats(b: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = []
    for (dom, lab), g in b.groupby(["domain", "label"]):
        X = g[feats].astype(float)
        q = X.quantile([0.25, 0.5, 0.75])
        sd, miss = X.std(), X.isna().mean() * 100
        for f in feats:
            rows.append(dict(feature=f, domain=dom, label=int(lab), n=len(g),
                             median=q.at[0.5, f], iqr=q.at[0.75, f] - q.at[0.25, f],
                             std=sd[f], miss_pct=miss[f]))
    return pd.DataFrame(rows)


def compare_distributions(au, b2, b1, feats, out, protocol):
    feats = [f for f in feats if f in b2.columns and f in b1.columns]
    s = strata_stats(b2, feats).merge(strata_stats(b1, feats),
                                      on=["feature", "domain", "label"], suffixes=("", "_v1"))
    s["delta_median"] = s["median"] - s["median_v1"]
    scale = s["iqr_v1"].where(s["iqr_v1"] > 0, s["std_v1"])
    rel = (s.delta_median.abs() / scale).where(scale > 0, np.where(s.delta_median.abs() > 0,
                                                                   np.inf, 0.0))
    s["delta_over_scale"] = rel
    s["delta_miss_pts"] = s["miss_pct"] - s["miss_pct_v1"]
    s["flag"] = (rel > SHIFT_IQR) | (s.delta_miss_pts.abs() > SHIFT_MISS)
    s.to_csv(out / "shift_v1.csv", index=False)
    f = s[s.flag].sort_values("delta_over_scale", ascending=False)
    show = f[["feature", "domain", "label", "n", "n_v1", "median", "median_v1",
              "delta_median", "iqr_v1", "delta_over_scale", "miss_pct", "miss_pct_v1",
              "delta_miss_pts"]].head(60)
    verdict = "DECIDE" if len(f) else "OK"
    au.add("A4.2", "Medians and missingness by domain x label, against v1", verdict,
           f"{len(s)} feature x domain x label cells on the {protocol} subset; {len(f)} "
           f"flagged ({int((rel > SHIFT_IQR).sum())} by median, "
           f"{int((s.delta_miss_pts.abs() > SHIFT_MISS).sum())} by missingness); max "
           f"|delta median| / scale {rel.max():.3f}, "
           f"max |delta missing| {s.delta_miss_pts.abs().max():.2f} points",
           f"per cell: median, IQR, std and missing share; flagged when |delta median| > "
           f"{SHIFT_IQR:g} x v1 IQR (v1 std where the IQR is 0) or |delta missing| > "
           f"{SHIFT_MISS:g} points. Full table in shift_v1.csv.",
           [("flagged cells (largest first, at most 60)", show)] if len(f) else [],
           dict(n_flagged=int(len(f))))


def merit_pairs(b: pd.DataFrame, near_m: float) -> dict:
    """Pairs closer than near_m with identical hand and log10_upa, plus a reference ring."""
    ref_m = 2 * near_m
    pts = np.asarray(b.geometry.values)
    i, j = STRtree(pts).query(pts, predicate="dwithin", distance=ref_m)
    k = i < j
    i, j = i[k], j[k]
    d = shapely.distance(pts[i], pts[j])
    h, u = b.hand.to_numpy(float), b.log10_upa.to_numpy(float)
    same = (h[i] == h[j]) & (u[i] == u[j])
    nan2 = np.isnan(h[i]) & np.isnan(h[j])
    lab = b.label.to_numpy()
    near, ring = d < near_m, (d >= near_m) & (d < ref_m)
    kind = np.where(lab[i] + lab[j] == 2, "pos-pos",
                    np.where(lab[i] + lab[j] == 1, "pos-neg", "neg-neg"))
    rows = []
    for kk in ("pos-pos", "pos-neg", "neg-neg", "all"):
        m = np.ones(len(d), bool) if kk == "all" else kind == kk
        rows.append({"pair": kk, f"pairs < {near_m:g} m": int((m & near).sum()),
                     "identical hand + log10_upa": int((m & near & same).sum()),
                     "hand missing on both": int((m & near & nan2).sum()),
                     f"pairs {near_m:g}-{ref_m:g} m": int((m & ring).sum()),
                     "identical (reference)": int((m & ring & same).sum())})
    pts_inv = len(np.unique(np.r_[i[near & same], j[near & same]]))
    return dict(table=pd.DataFrame(rows), n=int((near & same).sum()),
                n_near=int(near.sum()), points=pts_inv,
                ring_rate=float(same[ring].mean()) if ring.any() else np.nan,
                near_rate=float(same[near].mean()) if near.any() else np.nan)


def check_merit(au, b2, b1, a):
    r = merit_pairs(b2, a.merit_m)
    tabs = [("this table", r["table"])]
    ref = ""
    if b1 is not None:
        r1 = merit_pairs(b1, a.merit_m)
        tabs.append(("v1", r1["table"]))
        ref = (f"; v1: {r1['n']:,} of {r1['n_near']:,} pairs < {a.merit_m:g} m identical")
    verdict = "OK" if r["n"] == 0 else "DECIDE"
    au.add("A4.3", f"MERIT check: pairs < {a.merit_m:g} m with identical hand and log10_upa",
           verdict,
           f"{r['n']:,} of {r['n_near']:,} pairs closer than {a.merit_m:g} m have identical "
           f"hand and log10_upa ({100 * r['near_rate']:.1f}%), touching {r['points']:,} "
           f"points; at {a.merit_m:g}-{2 * a.merit_m:g} m the identical rate is "
           f"{100 * r['ring_rate']:.1f}%{ref}",
           f"STRtree `dwithin` on the EPSG:5070 points of the {a.protocol} subset; exact "
           "float equality of both values; a pair with HAND missing on both points is "
           "counted apart, not as identical.", tabs,
           dict(n=r["n"], n_near=r["n_near"], points=r["points"]))


# =========================================================================
def file_info(p: Path | None) -> dict | None:
    if p is None:
        return None
    st = p.stat()
    rows = pq.ParquetFile(p).metadata.num_rows if p.suffix == ".parquet" else None
    return dict(path=str(p), rows=rows, size_mb=round(st.st_size / 1e6, 1),
                modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--state", default="FL", choices=sorted(FIPS))
    ap.add_argument("--dataset", type=Path, help="default data/processed/dataset_<ST>.parquet")
    ap.add_argument("--v1", type=Path, help="previous build, for the A4 comparison")
    ap.add_argument("--features", type=Path, default=DATA / "FEATURES.txt")
    ap.add_argument("--idx", type=Path, help="default data/interim/prep/idx_<ST>.parquet")
    ap.add_argument("--panels", type=Path,
                    help="default data/interim/nfhl_states/pan_<ST>.parquet")
    ap.add_argument("--counties", type=Path,
                    help="state polygons for B7 (.parquet, or a TIGER tl_*_us_county.zip); "
                         "default data/interim/gis/counties_<ST>.parquet, else the newest "
                         "data/raw/tiger/tl_*_us_county.zip, else B7 is a bbox test")
    ap.add_argument("--seed", type=int, default=0, help="tile split seed (zoo.split_tiles)")
    ap.add_argument("--protocol", default="band", choices=list(sp.PROTOCOLS))
    ap.add_argument("--train-dom", default="AE", choices=["A", "AE"])
    ap.add_argument("--per-domain", type=int, default=20000, help="positives per zone (v2)")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="negatives per positive (v2)")
    ap.add_argument("--min-sep", type=float, default=100.0, help="spacing floor, metres (v2)")
    ap.add_argument("--n-features", type=int, default=45)
    ap.add_argument("--nearest-sample", type=int, default=-1,
                    help="band negatives per zone for the nearest-SFHA check; "
                         "-1 (default) takes all, 0 skips it")
    ap.add_argument("--merit-m", type=float, default=125.0)
    ap.add_argument("--out", type=Path,
                    default=OUT_ROOT / date.today().isoformat() / "data_audit")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    st = a.state
    a.dataset = a.dataset or DATA / f"dataset_{st}.parquet"
    a.idx = a.idx or sp.PREP_DIR / f"idx_{st}.parquet"
    a.panels = a.panels or sp.IN_DIR / f"pan_{st}.parquet"
    haz = sp.PREP_DIR / f"haz_{st}_5070.parquet"
    for p in (a.dataset, a.features, a.idx, haz, a.v1, a.counties):
        if p is not None and not p.exists():
            sys.exit(f"missing {p}" + (" -- run from the repo root" if p == haz else ""))
    if (a.out / "audit.json").exists() and not a.overwrite:
        print(f"{a.out}/audit.json exists -- skipping (use --overwrite)")
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = gpd.read_parquet(a.dataset)
    feats = read_features(a.features)
    idx = pd.read_parquet(a.idx)
    if (pq.ParquetFile(haz).metadata.num_rows != len(idx)
            or not np.array_equal(idx.row_id.to_numpy(), np.arange(len(idx)))):
        sys.exit(f"{a.idx} and {haz} are not from the same --prep run (row_id mismatch)")
    pan = (gpd.read_parquet(a.panels).to_crs(sp.CRS_WORK) if a.panels.exists() else None)
    v1 = gpd.read_parquet(a.v1) if a.v1 is not None else None
    a.counties = a.counties or find_counties(st)
    cn = state_polygons(a.counties, st) if a.counties is not None else None
    print(f"[{st}] {a.dataset}: {len(df):,} rows, {len(df.columns)} columns | "
          f"{len(feats)} features listed | idx {len(idx):,} polygons | "
          f"{0 if pan is None else len(pan):,} panels | "
          + (f"{len(cn)} county polygons from {a.counties}" if cn is not None
             else "no county polygons (B7 bbox only)")
          + (f" | v1 {a.v1}: {len(v1):,} rows" if v1 is not None else ""), flush=True)

    b = band_subset(df, a.protocol)
    au = Audit()
    check_counts(au, df, a)
    check_split(au, df, a)
    check_missing(au, b, feats, a.out, a.protocol)
    check_water(au, b, a.protocol)
    check_constant(au, b, feats)
    check_features_txt(au, feats, df, a.n_features)
    check_lonlat(au, df, pan, st, cn, a.counties)
    check_point_id(au, df, st)
    check_spatial_keys(au, df)
    check_invariants(au, df, a)
    check_positive_polygons(au, df, idx, st)
    check_fld_zone(au, df, a)
    check_county(au, df, st)
    check_eff_date(au, df, pan, a)
    check_band_source(au, df, idx, st, a)
    check_nearest_zone(au, df, idx, st, a)
    b1 = band_subset(v1, a.protocol) if v1 is not None else None
    if v1 is not None:
        compare_columns(au, df, v1, feats)
        compare_distributions(au, b, b1, feats, a.out, a.protocol)
    check_merit(au, b, b1, a)

    num = key_numbers(df, a)
    n1 = key_numbers(v1, a) if v1 is not None else None
    keys = list(num) + [k for k in (n1 or {}) if k not in num]
    numbers = pd.DataFrame({"quantity": keys, "this table": [num.get(k, "-") for k in keys]})
    if n1 is not None:
        numbers["v1"] = [n1.get(k, "-") for k in keys]

    meta = dict(state=st, generated=datetime.now().isoformat(timespec="seconds"),
                command="python " + " ".join([Path(sys.argv[0]).name] + argv),
                runtime_s=time.time() - t0,
                inputs=[dict(role=r, **file_info(p)) for r, p in
                        (("dataset", a.dataset), ("v1", a.v1), ("features", a.features),
                         ("idx", a.idx), ("haz", haz), ("panels", a.panels),
                         ("counties", a.counties))
                        if p is not None and p.exists()],
                args={k: str(v) for k, v in vars(a).items()})
    write_outputs(au, a.out, meta, numbers)

    print("\nkey numbers:\n  " + numbers.to_string(index=False).replace("\n", "\n  "))
    n = {v: [i["key"] for i in au.items if i["verdict"] == v] for v in VERDICTS}
    print(f"\n{len(au.items)} checks: {len(n['OK'])} OK, {len(n['WRONG'])} WRONG "
          f"{n['WRONG']}, {len(n['DECIDE'])} DECIDE {n['DECIDE']}")
    print(f"wrote {a.out}/audit.md, audit.json, missingness.csv"
          + (", shift_v1.csv" if v1 is not None else "") + f"  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
