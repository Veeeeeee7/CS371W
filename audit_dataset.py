#!/usr/bin/env python3
"""
Read-only audit of the Florida point table before a model sees it.

Checks data/processed/dataset_<ST>.parquet (Part B of docs/TASK-DATA-AUDIT.md) and, with --v1,
how it moved from the previous build (A4); docs/DATA-AUDIT-2026-10-14.md sections 4-5 discuss
the results. Each check ends in a verdict: OK (correct as built), WRONG (fix before the table is
used) or DECIDE (works, but needs a decision or a sentence in the paper). Inputs are only read.

Writes under --out: audit.md (paste-ready), audit.json (the same, machine readable),
missingness.csv and, with --v1, shift_v1.csv. Run from the repo root, because
sample_points.read_geoms reads data/interim/prep/:
    python audit_dataset.py --state FL --dataset data/processed/dataset_FL.parquet \\
        --v1 data/interim/archive/sampling-v1/data/processed/dataset_FL.parquet \\
        --out results/$D/data_audit
"""
import argparse
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import shapely
from pyproj import CRS, Transformer
from shapely import STRtree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aotil"))   # zoo.py lives in aotil/
import zoo                    # noqa: E402
import sample_points as sp    # noqa: E402

DATA = Path("data/processed")
OUT_ROOT = Path("results")
GIS_DIR = Path("data/interim/gis")      # pull_gis_lines.py writes counties_<ST>.parquet here
RAW_TIGER = Path("data/raw/tiger")      # ... and downloads tl_<year>_us_county.zip here

STATES = ["FL"]         # the --state choices; Florida is the only state built
FIPS = {"FL": "12"}     # state FIPS code
# Padded lon/lat extent of the state: checks that lon/lat are degrees in the right place, it is
# not a state test.
BBOX = {"FL": (-87.70, 24.30, -79.80, 31.10)}
BAND = "band"           # the protocol drawn from the panel mask, so it can cross the state line

# Columns that identify a point or carry its split; none may be a feature.
ID_COLS = {"point_id", "state", "county_fips", "dfirm_id", "block_id", "domain",
           "label", "neg_protocol", "lon", "lat", "x", "y", "geometry"}
EXPECTED_NEW = {"meta_src_row_id", "meta_neg_rank"}   # v2 adds these to v1's columns

BASE_RATE_GAP = 0.05    # |positive share ID - positive share OOD| above this is reported
LABEL_MISS_GAP = 5.0    # missingness gap between negatives and positives, percentage points
WATER_GAP = 5.0         # open-water share gap between negatives and positives, percentage points
NEAR_CONST = 0.99       # top-value share above this -> near-constant
SHIFT_IQR = 0.05        # |delta median| / v1 IQR above this -> flagged move
SHIFT_MISS = 2.0        # change in missing share, percentage points
DIST_TOL_M = 1.0        # distance agreement, metres
LONLAT_TOL_M = 1.0      # lon/lat against the reprojected geometry, metres
STATE_TOL_M = 100.0     # a positive closer than this outside the county line: line mismatch
NAN_BY_DESIGN = {"aspect_sin", "aspect_cos"}   # pull_gee.py leaves them NaN below 0.5 deg slope
NLCD_WATER = 11         # NLCD class: open water
VERDICTS = ("OK", "WRONG", "DECIDE")

# B14: the rules for which containing FIRM panel a point's meta_eff_date comes from.
RULE_V2 = "own DFIRM's panel, then lowest FIRM_PAN (sample_points.attach_panel_meta)"
RULE_V1 = "first spatial-join match (the v1 build)"

M_PER_DEG_LON = 111_320     # metres per degree of longitude at the equator
M_PER_DEG_LAT = 110_574     # metres per degree of latitude


# =========================================================================
# reporting
# =========================================================================
class Audit:
    """Collects the checks as they run, prints each one, and keeps them for write_outputs."""

    def __init__(self):
        self.items: list[dict] = []

    def add(self, key, title, verdict, result, how, tables=(), data=None):
        assert verdict in VERDICTS, verdict
        self.items.append(dict(key=key, title=title, verdict=verdict, result=result,
                               how=how, tables=list(tables), data=data or {}))
        print(f"\n[{verdict}] {key}. {title}\n  {result}", flush=True)
        for caption, table in tables:
            body = table.head(25).to_string(index=False)
            more = f"\n    ... {len(table) - 25} more rows in audit.md" if len(table) > 25 else ""
            print(f"  {caption}:\n    " + body.replace("\n", "\n    ") + more, flush=True)


def pick_verdict(wrong, decide) -> str:
    """WRONG if anything is wrong, else DECIDE if anything needs a decision, else OK."""
    if wrong:
        return "WRONG"
    if decide:
        return "DECIDE"
    return "OK"


def fmt(v) -> str:
    """One markdown table cell: thousands separators, up to 4 decimals below 100, pipes escaped."""
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
        if abs(v) >= 100:
            return f"{v:,.1f}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v).replace("|", "\\|")


def md_table(t: pd.DataFrame) -> str:
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in t.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def clean(o):
    """Make numpy / pandas values JSON-safe; NaN, NA and NaT become null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    # Before the datetime test: pd.NaT is a datetime too, and its isoformat() is "NaT".
    if o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    return o


def write_outputs(au: Audit, out: Path, meta: dict, numbers: pd.DataFrame) -> None:
    """audit.md (inputs, key numbers, summary, one section per check) and audit.json."""
    counts = {v: sum(i["verdict"] == v for i in au.items) for v in VERDICTS}
    summary = pd.DataFrame([dict(item=i["key"], check=i["title"], verdict=i["verdict"],
                                 result=i["result"]) for i in au.items])
    md = [f"# Data audit: {meta['state']} point table", "",
          f"Generated {meta['generated']} by `audit_dataset.py` ({meta['runtime_s']:.0f} s).", "",
          "```", meta["command"], "```", "",
          "## Inputs", "", md_table(pd.DataFrame(meta["inputs"])), "",
          "## Key numbers", "", md_table(numbers), "",
          f"## Summary: {counts['OK']} OK, {counts['WRONG']} WRONG, {counts['DECIDE']} DECIDE",
          "", md_table(summary), ""]
    for item in au.items:
        md += [f"### {item['key']}. {item['title']}: {item['verdict']}", "",
               f"**Result.** {item['result']}", "", f"**How.** {item['how']}", ""]
        for caption, table in item["tables"]:
            md += [f"*{caption}*", "", md_table(table), ""]
    (out / "audit.md").write_text("\n".join(md))

    items = [dict(item, tables=dict(item["tables"])) for item in au.items]
    js = dict(meta, numbers=numbers, verdicts=counts, items=items)
    (out / "audit.json").write_text(json.dumps(clean(js), indent=1))


# =========================================================================
# shared pieces
# =========================================================================
def read_features(path: Path) -> list[str]:
    """FEATURES.txt as zoo reads it, before zoo.load drops the names the table lacks."""
    return zoo.read_features(path)


def other_zone(zone: str) -> str:
    return "AE" if zone == "A" else "A"


def band_subset(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Positives plus this protocol's negatives, both zones: the union of zoo.subset."""
    return pd.concat([zoo.subset(df, d, protocol) for d in ("A", "AE")])


def split_frames(df, protocol, train_dom, seed):
    """The frames a zoo cell is fitted and scored on, built as zoo.cell_data builds them
    (split_tiles sees the whole table, so on v1 the shadedX rows too)."""
    tr_tiles, te_tiles = zoo.split_tiles(df, seed)
    train_zone = zoo.subset(df, train_dom, protocol)
    eval_zone = zoo.subset(df, other_zone(train_dom), protocol)
    train = train_zone[train_zone[zoo.TILE].isin(tr_tiles)]
    id_test = train_zone[train_zone[zoo.TILE].isin(te_tiles)]
    ood_test = eval_zone[eval_zone[zoo.TILE].isin(te_tiles)]
    return train, id_test, ood_test, tr_tiles, te_tiles


def strata(df: pd.DataFrame) -> pd.Series:
    """Row counts by domain x label x neg_protocol ('-' for positives)."""
    return df.groupby(["domain", "label", df.neg_protocol.fillna("-")]).size()


def key_numbers(df: pd.DataFrame, a) -> dict:
    """The data numbers the paper quotes, for one table."""
    numbers = {"rows": len(df), "band-subset rows": len(band_subset(df, a.protocol))}
    for (domain, label, protocol), n in strata(df).items():
        kind = "positives" if label == 1 else f"{protocol} negatives"
        numbers[f"{domain} {kind}"] = int(n)
    train, id_test, ood_test, tr_tiles, te_tiles = split_frames(df, a.protocol, a.train_dom,
                                                                a.seed)
    numbers[f"seed-{a.seed} train ({a.train_dom})"] = len(train)
    numbers[f"seed-{a.seed} ID test ({a.train_dom})"] = len(id_test)
    numbers[f"seed-{a.seed} OOD test ({other_zone(a.train_dom)})"] = len(ood_test)
    numbers["tiles train / test"] = f"{len(tr_tiles)} / {len(te_tiles)}"
    numbers["positive share train"] = round(float(train.label.mean()), 4)
    numbers["positive share ID test"] = round(float(id_test.label.mean()), 4)
    numbers["positive share OOD test"] = round(float(ood_test.label.mean()), 4)
    return numbers


def valid_row_ids(values: pd.Series, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """meta_src_row_id as int64 idx rows, and the mask of the entries that are valid rows."""
    rid = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(rid) & (rid >= 0) & (rid < n_rows)
    return rid[valid].astype(np.int64), valid


def match_polygons(pts: pd.DataFrame, idx: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Candidate source polygons per point: (i, row_id) pairs, i = position in pts.

    One pair per point through meta_src_row_id; through (dfirm_id, meta_fld_ar_id) on a v1
    table, where a key can match several polygons.
    """
    i = np.arange(len(pts))
    if "meta_src_row_id" in pts.columns:
        row_ids, valid = valid_row_ids(pts["meta_src_row_id"], len(idx))
        return pd.DataFrame({"i": i[valid], "row_id": row_ids}), "meta_src_row_id"
    keys = pd.DataFrame({"i": i, "DFIRM_ID": pts["dfirm_id"].to_numpy(),
                         "FLD_AR_ID": pts["meta_fld_ar_id"].to_numpy()})
    pairs = keys.merge(idx[["DFIRM_ID", "FLD_AR_ID", "row_id"]], on=["DFIRM_ID", "FLD_AR_ID"])
    return pairs[["i", "row_id"]], "(dfirm_id, meta_fld_ar_id)"


def geoms_for(state: str, row_ids: np.ndarray) -> np.ndarray:
    """Geometry per row_id (None if absent), each polygon decoded once."""
    uniq = np.unique(row_ids)
    g = sp.read_geoms(state, uniq)
    return g[np.searchsorted(uniq, row_ids)]


def xy(df: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df.geometry.x.to_numpy(), df.geometry.y.to_numpy()


def find_counties(state: str) -> Path | None:
    """pull_gis_lines.py's county polygons, else the newest national TIGER zip it downloads."""
    p = GIS_DIR / f"counties_{state}.parquet"
    if p.exists():
        return p
    zips = sorted(RAW_TIGER.glob("tl_*_us_county.zip"))
    return zips[-1] if zips else None


def state_polygons(path: Path, state: str) -> gpd.GeoDataFrame:
    """The state's county polygons (STATEFP filter) in EPSG:5070."""
    fips = FIPS[state]
    if path.suffix == ".zip":
        g = pyogrio.read_dataframe(f"/vsizip/{path.resolve()}", columns=["STATEFP", "GEOID"],
                                   where=f"STATEFP = '{fips}'")
    elif path.suffix == ".parquet":
        g = gpd.read_parquet(path)
    else:
        g = gpd.read_file(path)
    if "STATEFP" in g.columns:
        g = g[g.STATEFP.astype(str) == fips]
    if not len(g):
        sys.exit(f"{path}: no polygons with STATEFP {fips}")
    return g.to_crs(sp.CRS_WORK).reset_index(drop=True)


# =========================================================================
# Part B: the table on its own
# =========================================================================
def neg_rank_status(df: pd.DataFrame) -> tuple[bool, str]:
    """meta_neg_rank must run 0..n-1 within each negative stratum and be -1 on positives."""
    if "meta_neg_rank" not in df.columns:
        return True, "meta_neg_rank absent"
    neg = df[df.label == 0]
    broken = []
    for key, g in neg.groupby(["domain", neg.neg_protocol.fillna("-")]):
        ranks = np.sort(g.meta_neg_rank.to_numpy())
        if not np.array_equal(ranks, np.arange(len(g))):
            broken.append(key)
    positives_ok = bool((df.loc[df.label == 1, "meta_neg_rank"] == -1).all())
    if not broken and positives_ok:
        return True, "meta_neg_rank dense 0..n-1 per stratum, -1 on positives"
    return False, f"meta_neg_rank broken in {broken or 'positives'}"


def check_counts(au, df, a):
    """B1: both zones hold exactly the v2 targets and nothing else."""
    counts = strata(df)
    by_zone = counts.unstack("domain", fill_value=0)
    n_neg = int(round(a.per_domain * a.neg_ratio))
    targets = {(1, "-"): a.per_domain, (0, a.protocol): n_neg}
    tab = by_zone.reset_index()
    tab["target per zone"] = [targets.get((label, protocol), 0)
                              for label, protocol in zip(tab.label, tab.neg_protocol)]
    both_zones = {"A", "AE"} <= set(by_zone.columns)
    identical = both_zones and bool((by_zone["A"] == by_zone["AE"]).all())
    exact = both_zones and set(by_zone.index) == set(targets)
    if exact:
        for key, target in targets.items():
            if int(by_zone.loc[key, "A"]) != target or int(by_zone.loc[key, "AE"]) != target:
                exact = False
    extra = sorted(p for _, p in by_zone.index if p not in ("-", a.protocol))
    rank_ok, rank_note = neg_rank_status(df)

    verdict = "OK" if identical and exact and rank_ok else "WRONG"
    result = (f"{len(df):,} rows; A/AE strata identical: {'yes' if identical else 'NO'}; "
              f"exact v2 targets ({a.per_domain:,} positives + {n_neg:,} {a.protocol} "
              f"negatives per zone, nothing else): {'yes' if exact else 'NO'}")
    if extra:
        result += f" (extra protocols: {', '.join(extra)})"
    result += f"; {rank_note}"
    au.add("B1", "Counts by domain x label x neg_protocol", verdict, result,
           "`df.groupby(['domain','label',neg_protocol.fillna('-')]).size()` against "
           "--per-domain and round(--per-domain * --neg-ratio).",
           [("rows per stratum", tab)],
           dict(strata={f"{d}/{l}/{p}": int(n) for (d, l, p), n in counts.items()}))


def check_split(au, df, a):
    """B2: the seed's 50 km tile split and the gap between the ID and OOD positive shares. The
    gap is reported, flagged above BASE_RATE_GAP, and does not decide O1's metric: O1 reports
    accuracy, AUROC and F1, and its prevalence check (o1_shift.py, adjacc_*) shows what the gap
    does to accuracy by reweighting Zone A to the Zone AE share."""
    train, id_test, ood_test, tr_tiles, te_tiles = split_frames(df, a.protocol, a.train_dom,
                                                                a.seed)
    halves = [("train", train, a.train_dom, len(tr_tiles)),
              ("ID test", id_test, a.train_dom, len(te_tiles)),
              ("OOD test", ood_test, other_zone(a.train_dom), len(te_tiles))]
    rows = []
    for name, frame, zone, n_tiles in halves:
        share = round(float(frame.label.mean()), 4) if len(frame) else np.nan
        rows.append(dict(half=name, domain=zone, tiles=n_tiles, n=len(frame),
                         positives=int(frame.label.sum()),
                         negatives=int((frame.label == 0).sum()),
                         positive_share=share, tiles_with_rows=frame[zoo.TILE].nunique()))
    tab = pd.DataFrame(rows)
    id_share = rows[1]["positive_share"]
    ood_share = rows[2]["positive_share"]
    gap = abs(id_share - ood_share)
    usable = len(train) >= zoo.MIN_TRAIN and len(id_test) > 0 and len(ood_test) > 0

    verdict = pick_verdict(wrong=not usable, decide=False)
    above = "above" if gap > BASE_RATE_GAP else "within"
    result = (f"train {len(train):,} | ID test {len(id_test):,} | OOD test {len(ood_test):,} | "
              f"tiles train {len(tr_tiles)} test {len(te_tiles)}; positive share ID "
              f"{id_share:.2f} vs OOD {ood_share:.2f} "
              f"(gap {gap:.3f}, {above} the {BASE_RATE_GAP} threshold; reported, not a "
              f"decision: see the O1 prevalence check, o1_shift.py adjacc_*)")
    if not usable:
        result += ("; zoo.run_cell would return None (too few training rows or an empty "
                   "test half)")
    au.add("B2", f"Seed-{a.seed} 50 km tile split ({a.protocol}, train {a.train_dom})",
           verdict, result,
           "imports `zoo.split_tiles(df, seed)` and `zoo.subset`, and repeats the train / ID / "
           "OOD split of `zoo.cell_data` on the whole table.",
           [("halves", tab)], dict(halves=tab, gap=gap))


def check_missing(au, b, feats, out, protocol):
    """B3: missing shares per feature, overall and by domain x label.

    HGB and XGBoost route NaN natively, so a feature missing far more often for negatives than
    for positives within a zone is a label proxy. aspect_sin/cos are excused: their NaN marks
    flat ground.
    """
    present = [f for f in feats if f in b.columns]
    miss = b[present].isna().groupby([b.domain, b.label]).mean().mul(100).T
    miss.columns = [f"{d}/{'pos' if l == 1 else 'neg'} %" for d, l in miss.columns]
    tab = miss.copy()
    tab.insert(0, "all %", b[present].isna().mean().mul(100))
    gaps = {}
    for zone in ("A", "AE"):
        if f"{zone}/neg %" in tab and f"{zone}/pos %" in tab:
            tab[f"{zone} neg-pos"] = tab[f"{zone}/neg %"] - tab[f"{zone}/pos %"]
            gaps[zone] = tab[f"{zone} neg-pos"]
    # The checks below run on unrounded shares; only what is written out is rounded,
    # to 4 decimals so a single NaN in 80,000 rows still shows.
    tab = tab.reset_index(names="feature")
    tab.round(4).to_csv(out / "missingness.csv", index=False)

    empty = tab.loc[(tab.filter(like="%") >= 100).any(axis=1), "feature"].tolist()
    proxy, excused = set(), set()
    for gap in gaps.values():
        for feature, value in gap.items():
            if abs(value) > LABEL_MISS_GAP:
                if feature in NAN_BY_DESIGN:
                    excused.add(feature)
                else:
                    proxy.add(feature)
    proxy, excused = sorted(proxy), sorted(excused)
    largest = {}
    for zone, gap in gaps.items():
        counted = gap.drop(list(NAN_BY_DESIGN), errors="ignore").abs()
        largest[zone] = (counted.idxmax(), float(counted.max()))
    any_miss = tab[tab["all %"] > 0]

    verdict = pick_verdict(wrong=empty, decide=proxy)
    result = (f"{len(any_miss)}/{len(present)} features have missing values; "
              f"entirely missing in some stratum: {empty or 'none'}; "
              f"label-correlated (|neg - pos| > {LABEL_MISS_GAP:g} points within a zone): "
              f"{len(proxy)} features")
    if proxy:
        worst = [f"{zone} {feature} {value:.1f}" for zone, (feature, value) in largest.items()]
        result += " (largest: " + ", ".join(worst) + ")"
    if excused:
        result += f"; excused by design: {', '.join(excused)}"
    au.add("B3", "Per-feature missingness, overall and by domain x label", verdict,
           result, f"`b[feats].isna()` on the {protocol} subset (positives + {protocol} "
           "negatives), grouped by domain x label; full table in missingness.csv.",
           [("features with any missing value (%)", any_miss.round(4))],
           dict(label_correlated=proxy, entirely_missing=empty))


def check_water(au, b, protocol):
    """B4: negatives on open water, the usual cause of label-correlated missingness.

    The 5 km ring around a coastal AE polygon reaches into bays that lie inside FIRM panels.
    """
    need = {"landcover", "gsw_occurrence", "hand"}
    if not need <= set(b.columns):
        au.add("B4", "Negatives on open water", "DECIDE",
               f"{sorted(need - set(b.columns))} not in the table; check skipped", "-")
        return
    flags = pd.DataFrame({"domain": b.domain, "label": b.label,
                          "open_water": b.landcover.eq(NLCD_WATER),
                          "gsw_ge_50": b.gsw_occurrence.ge(50), "hand_missing": b.hand.isna()})
    shares = flags.groupby(["domain", "label"])[["open_water", "gsw_ge_50", "hand_missing"]].mean()
    tab = shares.mul(100).round(2).reset_index()
    tab.columns = ["domain", "label", "NLCD open water %", "GSW occurrence >= 50 %",
                   "hand missing %"]
    no_hand = flags[(flags.label == 0) & flags.hand_missing]
    on_water = no_hand.groupby("domain").open_water.mean().mul(100).round(1).to_dict()
    gap = {}
    for zone, g in tab.groupby("domain"):
        if {0, 1} <= set(g.label):
            water = g.set_index("label")["NLCD open water %"]
            gap[zone] = float(water.loc[0] - water.loc[1])

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
    """B5: features with a single value, or one value covering more than NEAR_CONST of a group."""
    groups = {"all": b, "A": b[b.domain == "A"], "AE": b[b.domain == "AE"]}
    rows = []
    for f in feats:
        if f not in b.columns:
            continue
        row = dict(feature=f)
        for name, group in groups.items():
            values = group[f].dropna()
            shares = values.value_counts(normalize=True)
            row[f"{name} n_unique"] = int(values.nunique())
            row[f"{name} top share"] = float(shares.iloc[0]) if len(shares) else np.nan
            if name == "all":
                row["top value"] = shares.index[0] if len(shares) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    n_unique = t.filter(like="n_unique").min(axis=1)
    top = t.filter(like="top share").max(axis=1)
    const = t.loc[n_unique <= 1, "feature"].tolist()
    near = t.loc[(top > NEAR_CONST) & (n_unique > 1), "feature"].tolist()
    show = t[top > 0.9].round(4)

    verdict = pick_verdict(wrong=const, decide=near)
    result = (f"constant (<= 1 unique value, overall or within a zone): {const or 'none'}; "
              f"near-constant (top value share > {NEAR_CONST}): {near or 'none'}; "
              f"largest top share {top.max():.3f} ({t.loc[top.idxmax(), 'feature']})")
    au.add("B5", "Constant and near-constant features", verdict, result,
           "value counts per feature on the band subset, overall and per zone.",
           [("features whose top value covers > 90% of a group", show)],
           dict(constant=const, near_constant=near))


def is_forbidden_feature(name: str) -> bool:
    """Identifiers, split keys, metadata and elevation must never be features."""
    return (name.startswith(sp.META_PREFIX) or name in ID_COLS or name.startswith("tile_")
            or name.endswith("_id") or "elev" in name.lower())


def check_features_txt(au, feats, df, n_expect):
    """B6: FEATURES.txt lists n_expect distinct numeric columns, none of them forbidden."""
    dup = sorted({f for f in feats if feats.count(f) > 1})
    absent = [f for f in feats if f not in df.columns]
    nonnum = [f for f in feats if f in df.columns and not pd.api.types.is_numeric_dtype(df[f])]
    banned = [f for f in feats if is_forbidden_feature(f)]
    known = set(feats) | ID_COLS
    other = [c for c in df.columns if c not in known and not c.startswith(sp.META_PREFIX)
             and not c.startswith("tile_")]

    bad = len(feats) != n_expect or dup or absent or nonnum or banned
    verdict = pick_verdict(wrong=bad, decide=other)
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

    One prepared polygon at a time over the points in its bbox, so the state is never unioned;
    a point on a county line counts as inside.
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
        outside_rows = np.nonzero(~inside)[0]
        pts = shapely.points(x[outside_rows], y[outside_rows])
        k, d = STRtree(polys).query_nearest(pts, return_distance=True, all_matches=False)
        dist[outside_rows[k[0]]] = d
    return ~inside, dist


def lonlat_error_m(df, epsg, lon, lat) -> float:
    """Largest distance (m) between the stored lon/lat and the geometry reprojected to CRS_SRC."""
    x, y = xy(df)
    to_lonlat = Transformer.from_crs(epsg or sp.CRS_WORK, sp.CRS_SRC, always_xy=True)
    lon_geom, lat_geom = to_lonlat.transform(x, y)
    dx = (lon - lon_geom) * M_PER_DEG_LON * np.cos(np.deg2rad(lat))
    dy = (lat - lat_geom) * M_PER_DEG_LAT
    return float(np.nanmax(np.hypot(dx, dy)))


def n_outside_box(lon, lat, box) -> int:
    """Points not inside (x0, y0, x1, y1); a non-finite coordinate counts as outside."""
    x0, y0, x1, y1 = box
    inside = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    return int((~inside).sum())


def outside_message(points: pd.DataFrame, mask, dist, state) -> str:
    """The finding for the points in mask: drawn from the state's own polygons, yet outside it."""
    counts = points[mask].groupby(["domain", "label"]).size()
    by = ", ".join(f"{d}/{l}: {n}" for (d, l), n in counts.items())
    return (f"points drawn from {state}'s own polygons that lie outside {state}: "
            f"{int(mask.sum()):,} ({by}; up to {dist[mask].max():.0f} m outside)")


def state_line_findings(df, cn, state):
    """B7's state test: (n outside, n band negatives outside, WRONG and DECIDE findings, table).

    Points drawn inside the state's own NFHL polygons must lie in the state; within STATE_TOL_M
    the NFHL and TIGER/Line just draw the line differently. Band negatives come from the panel
    mask, and panel sheets run past the state line.
    """
    outside, dist = outside_state(df, cn)
    protocol = df.neg_protocol.fillna("-").to_numpy()
    points = pd.DataFrame({"domain": df.domain.to_numpy(), "label": df.label.to_numpy(),
                           "neg_protocol": protocol, "outside_state": outside,
                           "max_m_outside": np.where(outside, dist, np.nan)})
    grouped = points.groupby(["domain", "label", "neg_protocol"])
    tab = grouped.agg(rows=("outside_state", "size"), outside_state=("outside_state", "sum"),
                      max_m_outside=("max_m_outside", "max")).reset_index()

    wrong, decide = [], []
    from_polygons = outside & (protocol != BAND)
    far = from_polygons & (dist > STATE_TOL_M)
    near = from_polygons & (dist <= STATE_TOL_M)
    if far.any():
        wrong.append(outside_message(points, far, dist, state))
    if near.any():
        decide.append(outside_message(points, near, dist, state)
                      + f", within {STATE_TOL_M:g} m: the NFHL and TIGER/Line "
                      "draw the state line differently there, not a sampling error")

    band = outside & (protocol == BAND)
    if band.any():
        counts = points[band].groupby("domain").size()
        by = ", ".join(f"{d}: {n}" for d, n in counts.items())
        share = 100 * band.sum() / max((protocol == BAND).sum(), 1)
        # sample_points reads --state-polygons without a STATEFP filter, so point it at
        # pull_gis_lines.py's in-state file, never at the national TIGER zip.
        decide.append(f"{int(band.sum()):,} {BAND} negatives ({share:.1f}%) lie outside "
                      f"{state} ({by}; up to {dist[band].max():.0f} m outside): FIRM "
                      "panel sheets run past the state line and the neighbouring "
                      "state's floodplains are not in this NFHL, so their band distance "
                      "is measured against half the map; keep them, or rebuild with "
                      f"`sample_points.py --state-polygons {GIS_DIR}/counties_{state}"
                      f".parquet` (written by pull_gis_lines.py --state {state} "
                      "--layers counties)")
    return int(outside.sum()), int(band.sum()), wrong, decide, tab


def check_lonlat(au, df, pan, state, cn, cn_path):
    """B7: lon/lat are degrees in the right place, their datum, and every point is in the state.

    A band negative can sit a few km into Alabama or Georgia while inside the bbox and carrying
    a Florida county_fips (its DFIRM's county), so only a test against the county polygons finds
    it. TIGER/Line county polygons include coastal water, so points in the state's waters stay in.
    """
    epsg = df.crs.to_epsg() if df.crs is not None else None
    lon = df.lon.to_numpy(float)
    lat = df.lat.to_numpy(float)
    finite = np.isfinite(lon) & np.isfinite(lat)
    dev = lonlat_error_m(df, epsg, lon, lat)
    panel_box = tuple(pan.to_crs(sp.CRS_SRC).total_bounds) if pan is not None else None
    bbox = BBOX[state]
    out_bbox = n_outside_box(lon, lat, bbox)
    out_pan = n_outside_box(lon, lat, panel_box) if panel_box is not None else None
    county_fips = df.county_fips.dropna()
    bad_fips = int((county_fips.str[:2] != FIPS[state]).sum())

    wrong, decide = [], []
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
    tabs = []
    if cn is None:
        decide.append(f"bbox only, not a state test: no county polygons at "
                      f"{GIS_DIR}/counties_{state}.parquet or {RAW_TIGER}/tl_*_us_county.zip "
                      f"(pull_gis_lines.py --state {state} --layers counties)")
    else:
        n_out, n_band_out, state_wrong, state_decide, tab = state_line_findings(df, cn, state)
        wrong += state_wrong
        decide += state_decide
        tabs = [(f"points outside the {len(cn)} {state} county polygons", tab)]

    verdict = pick_verdict(wrong, decide)
    state_test = ""
    if cn is not None:
        state_test = (f"{n_out:,} of {len(df):,} points outside the {len(cn)} {state} county "
                      f"polygons of {cn_path}; ")
    result = ("".join(f"WRONG: {w}. " for w in wrong)
              + "".join(f"DECIDE: {d}. " for d in decide)
              + f"{state_test}geometry CRS EPSG:{epsg}; lon {np.nanmin(lon):.3f} to "
              f"{np.nanmax(lon):.3f}, lat {np.nanmin(lat):.3f} to {np.nanmax(lat):.3f}; "
              f"{out_bbox} outside the {state} bbox {bbox}; {out_pan} outside "
              f"the FIRM panel extent; {bad_fips} county_fips not starting '{FIPS[state]}'; "
              f"lon/lat vs the geometry reprojected to EPSG:{sp.CRS_SRC}: max {dev:.3f} m")
    if cn is not None:
        state_how = (f"each point tested against the {state} county polygons (STATEFP "
                     f"{FIPS[state]}; TIGER/Line includes coastal water) with "
                     "`shapely.intersects_xy`, and its distance to the nearest one where it "
                     "lies outside. Points drawn inside the state's own NFHL polygons must be "
                     f"inside (WRONG beyond {STATE_TOL_M:g} m); {BAND} negatives come from the "
                     "panel mask and may not be.")
    else:
        state_how = ("no county polygons, so no state test (--counties, or pull_gis_lines.py "
                     "--layers counties).")
    au.add("B7", f"lon/lat are degrees, their datum, and inside {state}", verdict, result,
           f"geometry (EPSG:5070) reprojected to EPSG:{sp.CRS_SRC} with pyproj and compared "
           "to lon/lat; the padded bbox and the FIRM panel extent as a degrees check; "
           + state_how, tabs,
           dict(max_dev_m=dev, lonlat_epsg=sp.CRS_SRC, outside_bbox=out_bbox,
                outside_panels=out_pan, outside_state=n_out, outside_state_band=n_band_out,
                state_polygons=str(cn_path) if cn is not None else None))


def check_point_id(au, df, state):
    """B8: point_id is unique and of the form <ST> plus seven digits."""
    pid = df.point_id
    pattern = re.compile(rf"^{state}\d{{7}}$")
    bad = sum(1 for s in pid.fillna("") if not pattern.match(s))
    ok = pid.notna().all() and pid.is_unique and bad == 0
    au.add("B8", "point_id unique", "OK" if ok else "WRONG",
           f"{pid.nunique():,} unique of {len(pid):,}; {int(pid.isna().sum())} null; "
           f"{bad} not of the form {state}NNNNNNN",
           "`df.point_id.is_unique` and a format check.")


def check_spatial_keys(au, df):
    """B9: block_id and tile_50km agree with the EPSG:5070 geometry."""
    title = "block_id and tile_50km agree with the geometry"
    missing = sorted({"block_id", "tile_50km"} - set(df.columns))
    if missing:
        # zoo splits the table by tile_50km and stores both keys with every cell.
        au.add("B9", title, "WRONG", f"{missing} not in the table; check skipped", "-")
        return
    x, y = xy(df)
    block_m = sp.DEFAULTS["block_km"] * 1000.0
    bx = np.floor(x / block_m).astype(np.int64)
    by = np.floor(y / block_m).astype(np.int64)
    block = pd.Series(bx.astype(str), index=df.index) + "_" + by.astype(str)
    # A 50 km tile is 5 x 5 blocks of 10 km.
    tile = pd.Series((bx // 5).astype(str), index=df.index) + "_" + (by // 5).astype(str)
    bad_block = int((block != df.block_id).sum())
    bad_tile = int((tile != df.tile_50km).sum())
    ok = bad_block == 0 and bad_tile == 0
    au.add("B9", title, "OK" if ok else "WRONG",
           f"{bad_block} block_id and {bad_tile} tile_50km values differ from the geometry; "
           f"{df.block_id.nunique():,} blocks, {df.tile_50km.nunique():,} 50 km tiles",
           "block = floor(x / 10 km)_floor(y / 10 km) in EPSG:5070 "
           "(sample_points.build_state); tile_50km = block // 5 (pull_gee.derive).")


def check_invariants(au, df, a):
    """B10: the sampling invariants, rechecked so a later merge cannot have broken them."""
    pts = np.asarray(df.geometry.values)
    tree = STRtree(pts)
    i, j = tree.query(pts, predicate="dwithin", distance=a.min_sep)
    pair_once = i < j
    close = int((shapely.distance(pts[i[pair_once]], pts[j[pair_once]]) < a.min_sep).sum())
    nn, nn_dist = tree.query_nearest(pts, exclusive=True, return_distance=True)
    nearest = np.full(len(pts), np.inf)
    np.minimum.at(nearest, nn[0], nn_dist)
    pos = df[df.label == 1]
    pos_zero = float((pos.meta_dist_to_sfha_m == 0).mean())
    band_dist = df.loc[df.neg_protocol == a.protocol, "meta_dist_to_sfha_m"]
    band_min, band_max = sp.DEFAULTS["band_min_m"], sp.DEFAULTS["band_max_m"]
    off = int(((band_dist < band_min) | (band_dist > band_max) | band_dist.isna()).sum())
    ok = close == 0 and pos_zero == 1.0 and off == 0
    au.add("B10", "Sampling invariants carried into the table", "OK" if ok else "WRONG",
           f"pairs closer than {a.min_sep:g} m: {close}; minimum nearest-neighbour "
           f"distance {nearest.min():.2f} m, median {np.median(nearest):.0f} m; positives "
           f"at distance 0 from an SFHA {100 * pos_zero:.1f}%; {a.protocol} negatives "
           f"{band_dist.min():.0f}-{band_dist.max():.0f} m from an SFHA, {off} outside "
           f"{band_min:.0f}-{band_max:.0f} m",
           "STRtree over the EPSG:5070 points (as `sample_points.audit`) and "
           "`meta_dist_to_sfha_m`; repeated here so a merge cannot have broken them.",
           data=dict(min_nn_m=float(nearest.min()), close_pairs=close))


def check_positive_polygons(au, df, idx, state):
    """B11: every positive joins to exactly one polygon of its zone that contains it.

    FLD_AR_ID alone is not unique within a state (on v1, 190 positives matched 3-4 polygons by
    key), so a candidate counts only if it has the positive's zone and contains the point.
    """
    pos = df[df.label == 1]
    x, y = xy(pos)
    null_id = int(pos.meta_fld_ar_id.isna().sum())
    cand, how = match_polygons(pos, idx)
    rid = cand.row_id.to_numpy()
    att = idx.iloc[rid]
    pi = cand.i.to_numpy()
    zone_ok = att.FLD_ZONE.to_numpy() == pos.domain.to_numpy()[pi]
    # A missing key becomes None, which equals no polygon's key; pd.NA (a "string" column's
    # missing value) would make the comparison raise.
    fld_ar_id = pos.meta_fld_ar_id.to_numpy(dtype=object, na_value=None)
    dfirm_id = pos.dfirm_id.to_numpy(dtype=object, na_value=None)
    key_ok = ((att.FLD_AR_ID.to_numpy() == fld_ar_id[pi]) &
              (att.DFIRM_ID.to_numpy() == dfirm_id[pi]))
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
    for zone in ("A", "AE"):
        m = (pos.domain == zone).to_numpy()
        rows.append(dict(domain=zone, positives=int(m.sum()),
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
    """B12: a positive's meta_fld_zone is its domain; band negatives carry none."""
    pos = df[df.label == 1]
    neg = df[df.neg_protocol == a.protocol]
    pos_zone = pos.meta_fld_zone.fillna("<null>")
    bad = int((pos_zone != pos.domain).sum())
    neg_set = int(neg.meta_fld_zone.notna().sum())
    tab = pos.groupby(["domain", pos_zone]).size().rename("positives").reset_index()
    ok = bad == 0 and neg_set == 0
    au.add("B12", "meta_fld_zone consistent with domain", "OK" if ok else "WRONG",
           f"{bad} positives whose meta_fld_zone differs from domain; {neg_set} "
           f"{a.protocol} negatives with a meta_fld_zone (want 0: a band negative carries "
           "no polygon attributes, its zone is `domain`)",
           "`meta_fld_zone == domain` on positives.", [("positives", tab)])


def check_county(au, df, state):
    """B13: county_fips matches the DFIRM's county; community-based DFIRMs keep a null."""
    cf = df.county_fips
    dfirm = df.dfirm_id.fillna("")
    # A countywide DFIRM id is the county FIPS followed by a letter (e.g. 12086C).
    countywide = dfirm.str[-1].str.isalpha().to_numpy()
    mism = int((countywide & (cf.fillna("") != dfirm.str[:5]).to_numpy()).sum())
    pre = int((cf.dropna().str[:2] != FIPS[state]).sum())
    flags = pd.DataFrame({"label": df.label, "null": cf.isna(), "community_dfirm": ~countywide})
    tab = flags.groupby("label")[["null", "community_dfirm"]].sum().reset_index()
    tab.insert(1, "rows", df.groupby("label").size().to_numpy())
    nulls = int(cf.isna().sum())
    verdict = pick_verdict(wrong=mism or pre, decide=nulls)
    au.add("B13", "county_fips populated", verdict,
           f"{nulls} null ({int((~countywide).sum())} rows from community-based DFIRMs, "
           f"which keep a null by design); {mism} countywide rows where county_fips != "
           f"dfirm_id[:5]; {pre} not starting '{FIPS[state]}'; "
           f"{cf.nunique()} counties",
           "`county_fips` against `dfirm_id[:5]` for countywide DFIRMs (id ends in a letter).",
           [("by label", tab)])


def dates_match(mine, expected) -> np.ndarray:
    """Element-wise equal dates, two missing dates counting as equal."""
    return (pd.isna(mine) & pd.isna(expected)) | (mine == expected)


def first_date_per_point(joined: pd.DataFrame, n_points: int) -> np.ndarray:
    """EFF_DATE of each point's first row in `joined` (NaT for a point without one)."""
    first = joined[~joined.index.duplicated(keep="first")]
    return first.set_index("_i").EFF_DATE.reindex(np.arange(n_points)).to_numpy()


def pick_rule(miss: dict, v1_table: bool) -> tuple[str | None, str]:
    """The panel rule that meta_eff_date follows, and a note; None when it follows no valid rule.

    The v1 rule is valid only on a v1 table: a v2 table that follows it was built before
    attach_panel_meta changed.
    """
    n_v2 = int(miss[RULE_V2].sum())
    if not miss[RULE_V2].any():
        return RULE_V2, "reproduced for every point"
    if not miss[RULE_V1].any() and v1_table:
        return RULE_V1, ("reproduced for every point, as a v1 table should; the current rule "
                         f"would change {n_v2:,} dates")
    if not miss[RULE_V1].any():
        return None, (f"this v2 table (it has meta_src_row_id) follows the {RULE_V1}, and "
                      f"{n_v2:,} dates differ from the current rule: its points "
                      "were built before attach_panel_meta changed; rebuild them")
    return None, (f"meta_eff_date reproduces neither rule: {n_v2:,} points "
                  f"differ from the {RULE_V2}, {int(miss[RULE_V1].sum()):,} from the {RULE_V1}; "
                  "compare attach_panel_meta with this check")


def check_eff_date(au, df, pan, a):
    """B14: meta_eff_date against the FIRM panels, including which DFIRM's panel it came from.

    Panels of adjacent countywide DFIRMs overlap (on v1, 37% of points lie in two to four, always
    of different DFIRMs), so the date depends on which containing panel is used. The expected
    date restates sample_points.attach_panel_meta's rule rather than importing it, so a change on
    either side shows up as WRONG. A point that lies in a panel of its own DFIRM but carries none
    of those panels' dates is counted separately.
    """
    if pan is None:
        nulls_by_label = df.meta_eff_date.isna().groupby(df.label).sum()
        au.add("B14", "meta_eff_date populated", "DECIDE" if nulls_by_label.sum() else "OK",
               f"nulls by label {nulls_by_label.to_dict()}; no --panels, so not traced", "-")
        return
    eff = pd.to_datetime(pan.EFF_DATE)
    if eff.dt.tz is not None:
        eff = eff.dt.tz_convert(None)
    panel_cols = ["DFIRM_ID"]
    if "FIRM_PAN" in pan.columns:
        panel_cols.append("FIRM_PAN")
    points = gpd.GeoDataFrame({"_i": np.arange(len(df))}, geometry=df.geometry.values,
                              crs=df.crs)
    joined = gpd.sjoin(points, pan[panel_cols + ["geometry"]].assign(EFF_DATE=eff), how="left",
                       predicate="within")
    i = joined._i.to_numpy()                        # the point of each joined row
    hit = joined.index_right.notna().to_numpy()     # the row found a panel
    own = hit & (joined.DFIRM_ID.to_numpy() == df.dfirm_id.to_numpy()[i])
    n_in = np.bincount(i[hit], minlength=len(df))
    n_own = np.bincount(i[own], minlength=len(df))

    # The current rule: own DFIRM's panels first, then FIRM_PAN ascending, as attach_panel_meta.
    sort_by = ["_own"] + panel_cols[1:]
    ascending = [False] + [True] * (len(sort_by) - 1)
    ranked = joined.assign(_own=own).sort_values(sort_by, ascending=ascending, kind="stable")
    expect = {RULE_V2: first_date_per_point(ranked, len(df)),
              RULE_V1: first_date_per_point(joined, len(df))}
    mine = pd.to_datetime(df.meta_eff_date).to_numpy()
    null = pd.isna(mine)
    miss = {rule: ~dates_match(mine, expected) for rule, expected in expect.items()}
    same_own = own & dates_match(mine[i], joined.EFF_DATE.to_numpy())
    other = (n_own > 0) & (np.bincount(i[same_own], minlength=len(df)) == 0)

    rule, rule_note = pick_rule(miss, v1_table="meta_src_row_id" not in df.columns)
    expected = expect[rule or RULE_V2]
    stray = int((null & (n_in > 0) & pd.notna(expected)).sum())
    protocol = df.neg_protocol.fillna("-").to_numpy()
    band_out = int(((protocol == a.protocol) & (n_in == 0)).sum())

    flags = pd.DataFrame({"domain": df.domain.to_numpy(), "label": df.label.to_numpy(),
                          "null_eff_date": null, "in_no_panel": n_in == 0,
                          "in_several_panels": n_in > 1, "no_own_dfirm_panel": n_own == 0,
                          "date_not_own_dfirm": other})
    tab = flags.groupby(["domain", "label"]).sum().reset_index()
    tab.insert(2, "rows", flags.groupby(["domain", "label"]).size().to_numpy())
    nulls_by_label = flags.groupby("label").null_eff_date.sum()
    by_label = ", ".join(f"label {l}: {int(n)}" for l, n in nulls_by_label.items())

    verdict = pick_verdict(wrong=band_out or rule is None, decide=null.any() or other.any())
    picked = "that rule" if rule else "the current rule"
    result = (f"rule: {rule or 'NONE'}, {rule_note}; null meta_eff_date: {int(null.sum())} "
              f"({by_label}), {int((null & (n_in == 0)).sum())} of them in no panel and "
              f"{int((null & (n_in > 0) & pd.isna(expected)).sum())} where "
              f"{picked} picks a panel whose EFF_DATE is null")
    if stray:
        result += f", {stray} where it picks a dated panel"
    result += (f"; {band_out} {a.protocol} negatives outside every panel; "
               f"{int((n_in > 1).sum()):,} points lie in several panels, "
               f"{int((n_own == 0).sum()):,} in no panel of their own DFIRM, and "
               f"{int(other.sum()):,} lie in a panel of their own DFIRM but carry another "
               f"DFIRM's panel date")
    au.add("B14", "meta_eff_date populated, traced to the FIRM panels", verdict, result,
           "`gpd.sjoin(points, pan, predicate='within')`; expected date: the joined rows "
           "sorted by (DFIRM_ID == the point's `dfirm_id` first, then FIRM_PAN ascending, "
           "stable), first row per point, as `sample_points.attach_panel_meta` does; the v1 "
           "build's unsorted first match is accepted only on a table without "
           "`meta_src_row_id`.", [("by domain x label", tab)],
           dict(rule=rule, mismatch_v2_rule=int(miss[RULE_V2].sum()),
                mismatch_v1_rule=int(miss[RULE_V1].sum()),
                several_panels=int((n_in > 1).sum()), not_own_dfirm=int(other.sum())))


def check_band_source(au, df, idx, state, a):
    """B15: a band negative's ring polygon has its zone and DFIRM, and it lies in the band.

    A band negative takes the zone of the positive polygon whose 5 km ring produced it.
    """
    neg = df[df.neg_protocol == a.protocol]
    if "meta_src_row_id" not in neg.columns:
        au.add("B15", "A band negative's zone is its ring polygon's zone", "DECIDE",
               "meta_src_row_id absent (v1 table): the ring polygon of a band negative is "
               "not recorded, so its zone can only be read from `domain`",
               "needs `meta_src_row_id` (sample_points v2).")
        return
    row_ids, valid = valid_row_ids(neg.meta_src_row_id, len(idx))
    ring = idx.iloc[row_ids]
    sub = neg[valid]
    zone_ok = ring.FLD_ZONE.to_numpy() == sub.domain.to_numpy()
    dfirm_ok = ring.DFIRM_ID.to_numpy() == sub.dfirm_id.to_numpy()
    g = geoms_for(state, row_ids)
    have = np.array([v is not None for v in g])
    dist = np.full(len(row_ids), np.nan)
    dist[have] = shapely.distance(np.asarray(sub.geometry.values)[have], g[have])
    # The ring polygons were simplified by RING_SIMPLIFY_M before buffering.
    lo = sp.DEFAULTS["band_min_m"]
    hi = sp.DEFAULTS["band_max_m"] + 2 * sp.RING_SIMPLIFY_M
    off = int(((dist < lo - DIST_TOL_M) | (dist > hi)).sum())
    n_invalid = int((~valid).sum())
    bad = n_invalid + int((~zone_ok).sum()) + int((~dfirm_ok).sum())
    verdict = pick_verdict(wrong=bad, decide=off)
    au.add("B15", "A band negative's zone is its ring polygon's zone", verdict,
           f"{len(neg):,} {a.protocol} negatives: {n_invalid} without a valid "
           f"meta_src_row_id, {int((~zone_ok).sum())} whose ring polygon's FLD_ZONE differs "
           f"from domain, {int((~dfirm_ok).sum())} whose dfirm_id differs from it; distance "
           f"to the ring polygon {np.nanmin(dist):.0f}-{np.nanmax(dist):.0f} m, {off} outside "
           f"{lo:.0f}-{hi:.0f} m; {len(np.unique(row_ids)):,} ring polygons",
           "`idx.iloc[meta_src_row_id]`: FLD_ZONE == domain, DFIRM_ID == dfirm_id, and the "
           "point's distance to that polygon within the band plus the 2 x 100 m ring "
           "simplification slack.")


def sample_per_zone(neg: pd.DataFrame, n_per_zone: int, seed: int) -> np.ndarray:
    """Sorted row positions of n_per_zone negatives per zone, drawn with the seed; all if < 0."""
    rng = np.random.default_rng(seed)
    take = []
    for zone in ("A", "AE"):
        rows = np.nonzero((neg.domain == zone).to_numpy())[0]
        n = len(rows) if n_per_zone < 0 else min(n_per_zone, len(rows))
        take.append(np.sort(rng.choice(rows, n, replace=False)))
    return np.concatenate(take)


def joined_zones(zones) -> str:
    """The distinct zones, sorted and joined: 'A/AE'."""
    return "/".join(sorted(set(zones)))


def nearest_by_point(pi, dist, rid, zone, n_points):
    """Each point's nearest candidate polygon: its row_id, the distance, and a zone label.

    pi[k] is the point of candidate k. The label is the nearest polygon's FLD_ZONE, 'tie X/Y'
    when the polygons within DIST_TOL_M of the nearest have different zones, and '<none>' for a
    point without candidates.
    """
    dmin = np.full(n_points, np.inf)
    np.minimum.at(dmin, pi, dist)
    # Candidates sorted by point, then distance: the first of each point is its nearest.
    order = np.lexsort((dist, pi))
    sorted_pi = pi[order]
    starts_point = np.concatenate(([True], sorted_pi[1:] != sorted_pi[:-1]))
    first = order[starts_point]
    near_rid = np.full(n_points, -1, dtype=np.int64)
    label = np.full(n_points, "<none>", dtype=object)
    near_rid[pi[first]] = rid[first]
    label[pi[first]] = zone[first]

    tied = dist <= dmin[pi] + DIST_TOL_M
    zones_in_reach = pd.Series(zone[tied]).groupby(pi[tied]).agg(joined_zones)
    multi = zones_in_reach[zones_in_reach.str.contains("/")]
    label[multi.index.to_numpy()] = "tie " + multi.to_numpy()
    return near_rid, dmin, label


def has_a_and_ae(tie_label: str) -> bool:
    """For a 'tie ...' label: whether the tied zones include both A and AE."""
    return {"A", "AE"} <= set(tie_label[4:].split("/"))


def nearest_zone_shares(labels: np.ndarray, zone: str) -> dict:
    """Shares of one zone's negatives by the zone of their nearest SFHA polygon.

    `tie` is a tie between an A and an AE polygon; every other multi-zone tie (AE/AH, AE/VE,
    ...) is `other_tie`.
    """
    as_series = pd.Series(labels, dtype=object)
    is_tie = as_series.str.startswith("tie ")
    a_ae_tie = is_tie & as_series.map(has_a_and_ae)
    return dict(n=len(labels),
                nearest_other_zone=float((labels == other_zone(zone)).mean()),
                nearest_own_zone=float((labels == zone).mean()),
                nearest_other_sfha=float(np.isin(labels, ["AH", "AO", "VE"]).mean()),
                tie=float(a_ae_tie.mean()),
                other_tie=float((is_tie & ~a_ae_tie).mean()))


def check_nearest_zone(au, df, idx, state, a):
    """B16: the zone of each band negative's nearest SFHA polygon, with the distance recomputed.

    A band negative's 500-5,000 m distance is to the nearest SFHA of any zone, so a "Zone A"
    negative's nearest floodplain can be Zone AE. The candidates are the SFHA polygons whose
    bounding box lies within the stored distance + DIST_TOL_M of the point (a box is never
    farther than its polygon), and the recomputed distance must reproduce the stored one.
    """
    if a.nearest_sample == 0:
        return
    neg = df[df.neg_protocol == a.protocol]
    s = neg.iloc[sample_per_zone(neg, a.nearest_sample, a.seed)]
    pts = np.asarray(s.geometry.values)
    stored = s.meta_dist_to_sfha_m.to_numpy(float)
    reach = np.where(np.isfinite(stored), stored, sp.DEFAULTS["band_max_m"])

    t0 = time.time()
    sfha = idx[idx.SFHA_TF == "T"]
    boxes = shapely.box(sfha.minx.to_numpy(), sfha.miny.to_numpy(),
                        sfha.maxx.to_numpy(), sfha.maxy.to_numpy())
    pi, bj = STRtree(boxes).query(pts, predicate="dwithin", distance=reach + DIST_TOL_M)
    rid = sfha.row_id.to_numpy()[bj]
    zone = sfha.FLD_ZONE.to_numpy()[bj]
    g = geoms_for(state, rid)
    dist = shapely.distance(pts[pi], g)            # NaN where a geometry is missing
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near_rid, dmin, label = nearest_by_point(pi, dist, rid, zone, len(pts))

    agree = np.abs(dmin - stored) <= DIST_TOL_M
    tab = pd.crosstab(s.domain.to_numpy(), label, rownames=["negative's zone"],
                      colnames=["nearest SFHA"]).reset_index()
    share = {}
    for d in ("A", "AE"):
        share[d] = nearest_zone_shares(label[(s.domain == d).to_numpy()], d)
    own = ""
    if "meta_src_row_id" in s.columns:
        same = near_rid == s.meta_src_row_id.to_numpy()
        own = f"; nearest polygon is the negative's own ring polygon for {100 * same.mean():.1f}%"

    verdict = "OK" if agree.all() else "WRONG"
    if a.nearest_sample < 0 or len(s) == len(neg):
        which = "all"
    else:
        which = f"a seed-{a.seed} sample of {a.nearest_sample:,} per zone:"
    in_a, in_ae = share["A"], share["AE"]
    result = (f"{which} {len(s):,} {a.protocol} negatives; recomputed "
              f"nearest-SFHA distance reproduces meta_dist_to_sfha_m within "
              f"{DIST_TOL_M:g} m for {int(agree.sum()):,}/{len(s):,}; Zone A negatives whose "
              f"nearest SFHA is Zone AE {100 * in_a['nearest_other_zone']:.1f}%, Zone AE "
              f"negatives whose nearest SFHA is Zone A "
              f"{100 * in_ae['nearest_other_zone']:.1f}% (AH/AO/VE: A "
              f"{100 * in_a['nearest_other_sfha']:.1f}%, AE "
              f"{100 * in_ae['nearest_other_sfha']:.1f}%; A/AE ties: A "
              f"{100 * in_a['tie']:.1f}%, AE {100 * in_ae['tie']:.1f}%; other "
              f"multi-zone ties: A {100 * in_a['other_tie']:.1f}%, AE "
              f"{100 * in_ae['other_tie']:.1f}%){own} "
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
    """A4.1: v2 has v1's columns plus EXPECTED_NEW, with the same dtypes."""
    c2, c1 = list(df.columns), list(v1.columns)
    added = [c for c in c2 if c not in c1]
    removed = [c for c in c1 if c not in c2]
    feats_missing_here = [f for f in feats if f not in c2]
    feats_missing_v1 = [f for f in feats if f not in c1]
    dtypes = [f"{c}: {v1[c].dtype} -> {df[c].dtype}" for c in c2
              if c in c1 and str(df[c].dtype) != str(v1[c].dtype)]
    unexpected = [c for c in added if c not in EXPECTED_NEW]
    verdict = pick_verdict(wrong=removed or feats_missing_here or feats_missing_v1,
                           decide=unexpected or dtypes)
    au.add("A4.1", "Columns against v1", verdict,
           f"{len(c2)} columns (v1 {len(c1)}); added {added or 'none'} (expected "
           f"{sorted(EXPECTED_NEW)}); removed {removed or 'none'}; features missing here "
           f"{feats_missing_here or 'none'}, in v1 {feats_missing_v1 or 'none'}; dtype changes "
           f"{dtypes or 'none'}",
           "set difference of the column lists; v2 should add only meta_src_row_id and "
           "meta_neg_rank.", data=dict(added=added, removed=removed, dtype_changes=dtypes))


def strata_stats(b: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Median, IQR, std and missing share of each feature in each domain x label cell."""
    rows = []
    for (dom, lab), g in b.groupby(["domain", "label"]):
        values = g[feats].astype(float)
        q = values.quantile([0.25, 0.5, 0.75])
        sd = values.std()
        miss = values.isna().mean() * 100
        for f in feats:
            rows.append(dict(feature=f, domain=dom, label=int(lab), n=len(g),
                             median=q.at[0.5, f], iqr=q.at[0.75, f] - q.at[0.25, f],
                             std=sd[f], miss_pct=miss[f]))
    return pd.DataFrame(rows)


def compare_distributions(au, b2, b1, feats, out, protocol):
    """A4.2: per feature x domain x label, how far the median and missing share moved from v1."""
    shared = [f for f in feats if f in b2.columns and f in b1.columns]
    s = strata_stats(b2, shared).merge(strata_stats(b1, shared),
                                       on=["feature", "domain", "label"], suffixes=("", "_v1"))
    s["delta_median"] = s["median"] - s["median_v1"]
    moved = s.delta_median.abs()
    scale = s["iqr_v1"].where(s["iqr_v1"] > 0, s["std_v1"])    # v1 std where the IQR is 0
    no_scale = np.where(moved > 0, np.inf, 0.0)
    rel = (moved / scale).where(scale > 0, no_scale)
    s["delta_over_scale"] = rel
    s["delta_miss_pts"] = s["miss_pct"] - s["miss_pct_v1"]
    s["flag"] = (rel > SHIFT_IQR) | (s.delta_miss_pts.abs() > SHIFT_MISS)
    s.to_csv(out / "shift_v1.csv", index=False)
    flagged = s[s.flag].sort_values("delta_over_scale", ascending=False)
    show = flagged[["feature", "domain", "label", "n", "n_v1", "median", "median_v1",
                    "delta_median", "iqr_v1", "delta_over_scale", "miss_pct", "miss_pct_v1",
                    "delta_miss_pts"]].head(60)
    verdict = "DECIDE" if len(flagged) else "OK"
    au.add("A4.2", "Medians and missingness by domain x label, against v1", verdict,
           f"{len(s)} feature x domain x label cells on the {protocol} subset; {len(flagged)} "
           f"flagged ({int((rel > SHIFT_IQR).sum())} by median, "
           f"{int((s.delta_miss_pts.abs() > SHIFT_MISS).sum())} by missingness); max "
           f"|delta median| / scale {rel.max():.3f}, "
           f"max |delta missing| {s.delta_miss_pts.abs().max():.2f} points",
           f"per cell: median, IQR, std and missing share; flagged when |delta median| > "
           f"{SHIFT_IQR:g} x v1 IQR (v1 std where the IQR is 0) or |delta missing| > "
           f"{SHIFT_MISS:g} points. Full table in shift_v1.csv.",
           [("flagged cells (largest first, at most 60)", show)] if len(flagged) else [],
           dict(n_flagged=int(len(flagged))))


def merit_pairs(b: pd.DataFrame, near_m: float) -> dict:
    """Point pairs closer than near_m with identical hand and log10_upa.

    At 100 m spacing two points can share one ~90 m MERIT Hydro cell. Pairs near_m to 2 * near_m
    apart give the rate of identical values that a shared cell cannot explain (flat plateaus,
    catchment-wide values).
    """
    ref_m = 2 * near_m
    pts = np.asarray(b.geometry.values)
    i, j = STRtree(pts).query(pts, predicate="dwithin", distance=ref_m)
    pair_once = i < j
    i, j = i[pair_once], j[pair_once]
    d = shapely.distance(pts[i], pts[j])
    hand = b.hand.to_numpy(float)
    upa = b.log10_upa.to_numpy(float)
    same = (hand[i] == hand[j]) & (upa[i] == upa[j])
    both_missing = np.isnan(hand[i]) & np.isnan(hand[j])
    near = d < near_m
    ring = (d >= near_m) & (d < ref_m)
    labels = b.label.to_numpy()
    n_pos = labels[i] + labels[j]
    kind = np.select([n_pos == 2, n_pos == 1], ["pos-pos", "pos-neg"], default="neg-neg")
    rows = []
    for name in ("pos-pos", "pos-neg", "neg-neg", "all"):
        m = np.ones(len(d), bool) if name == "all" else kind == name
        rows.append({"pair": name, f"pairs < {near_m:g} m": int((m & near).sum()),
                     "identical hand + log10_upa": int((m & near & same).sum()),
                     "hand missing on both": int((m & near & both_missing).sum()),
                     f"pairs {near_m:g}-{ref_m:g} m": int((m & ring).sum()),
                     "identical (reference)": int((m & ring & same).sum())})
    involved = np.unique(np.concatenate([i[near & same], j[near & same]]))
    return dict(table=pd.DataFrame(rows), n=int((near & same).sum()),
                n_near=int(near.sum()), points=len(involved),
                ring_rate=float(same[ring].mean()) if ring.any() else np.nan,
                near_rate=float(same[near].mean()) if near.any() else np.nan)


def check_merit(au, b2, b1, a):
    """A4.3: close pairs that may share a MERIT Hydro cell, here and in v1."""
    r = merit_pairs(b2, a.merit_m)
    tabs = [("this table", r["table"])]
    ref = ""
    if b1 is not None:
        r1 = merit_pairs(b1, a.merit_m)
        tabs.append(("v1", r1["table"]))
        ref = f"; v1: {r1['n']:,} of {r1['n_near']:,} pairs < {a.merit_m:g} m identical"
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
# main
# =========================================================================
def file_info(p: Path | None) -> dict | None:
    if p is None:
        return None
    st = p.stat()
    rows = pq.ParquetFile(p).metadata.num_rows if p.suffix == ".parquet" else None
    return dict(path=str(p), rows=rows, size_mb=round(st.st_size / 1e6, 1),
                modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))


def numbers_table(df, v1, a) -> pd.DataFrame:
    """Key numbers of this table, and of v1 beside them when given ('-' where one lacks a row)."""
    num = key_numbers(df, a)
    num_v1 = key_numbers(v1, a) if v1 is not None else {}
    keys = list(num) + [k for k in num_v1 if k not in num]
    numbers = pd.DataFrame({"quantity": keys, "this table": [num.get(k, "-") for k in keys]})
    if v1 is not None:
        numbers["v1"] = [num_v1.get(k, "-") for k in keys]
    return numbers


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--state", default="FL", choices=STATES)
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
    return ap


def main(argv: list[str]) -> int:
    a = build_parser().parse_args(argv)
    st = a.state
    a.dataset = a.dataset or DATA / f"dataset_{st}.parquet"
    a.idx = a.idx or sp.PREP_DIR / f"idx_{st}.parquet"
    a.panels = a.panels or sp.IN_DIR / f"pan_{st}.parquet"
    haz = sp.PREP_DIR / f"haz_{st}_5070.parquet"
    for p in (a.dataset, a.features, a.idx, haz, a.v1, a.counties):
        if p is not None and not p.exists():
            hint = " -- run from the repo root" if p == haz else ""
            sys.exit(f"missing {p}{hint}")
    if (a.out / "audit.json").exists() and not a.overwrite:
        print(f"{a.out}/audit.json exists -- skipping (use --overwrite)")
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = gpd.read_parquet(a.dataset)
    feats = read_features(a.features)
    idx = pd.read_parquet(a.idx)
    same_prep = (pq.ParquetFile(haz).metadata.num_rows == len(idx)
                 and np.array_equal(idx.row_id.to_numpy(), np.arange(len(idx))))
    if not same_prep:
        sys.exit(f"{a.idx} and {haz} are not from the same --prep run (row_id mismatch)")
    pan = gpd.read_parquet(a.panels).to_crs(sp.CRS_WORK) if a.panels.exists() else None
    v1 = gpd.read_parquet(a.v1) if a.v1 is not None else None
    a.counties = a.counties or find_counties(st)
    cn = state_polygons(a.counties, st) if a.counties is not None else None

    loaded = (f"[{st}] {a.dataset}: {len(df):,} rows, {len(df.columns)} columns | "
              f"{len(feats)} features listed | idx {len(idx):,} polygons | "
              f"{0 if pan is None else len(pan):,} panels | ")
    if cn is not None:
        loaded += f"{len(cn)} county polygons from {a.counties}"
    else:
        loaded += "no county polygons (B7 bbox only)"
    if v1 is not None:
        loaded += f" | v1 {a.v1}: {len(v1):,} rows"
    print(loaded, flush=True)

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

    numbers = numbers_table(df, v1, a)
    inputs = []
    for role, path in (("dataset", a.dataset), ("v1", a.v1), ("features", a.features),
                       ("idx", a.idx), ("haz", haz), ("panels", a.panels),
                       ("counties", a.counties)):
        if path is not None and path.exists():
            inputs.append(dict(role=role, **file_info(path)))
    meta = dict(state=st, generated=datetime.now().isoformat(timespec="seconds"),
                command="python " + " ".join([Path(sys.argv[0]).name] + argv),
                runtime_s=time.time() - t0, inputs=inputs,
                args={k: str(v) for k, v in vars(a).items()})
    write_outputs(au, a.out, meta, numbers)

    print("\nkey numbers:\n  " + numbers.to_string(index=False).replace("\n", "\n  "))
    keys_by_verdict = {v: [i["key"] for i in au.items if i["verdict"] == v] for v in VERDICTS}
    wrong, decide = keys_by_verdict["WRONG"], keys_by_verdict["DECIDE"]
    print(f"\n{len(au.items)} checks: {len(keys_by_verdict['OK'])} OK, {len(wrong)} WRONG "
          f"{wrong}, {len(decide)} DECIDE {decide}")
    written = "audit.md, audit.json, missingness.csv" + (", shift_v1.csv" if v1 is not None else "")
    print(f"wrote {a.out}/{written}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
