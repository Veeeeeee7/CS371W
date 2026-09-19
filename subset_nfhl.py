#!/usr/bin/env python3
"""
subset_nfhl.py -- national NFHL geodatabase -> per-state GeoParquet.

Two passes, because they have very different cost:

  PASS 1  attributes only (ignore_geometry=True), all states at once.
          Seconds, no memory risk. Gives the go/no-go cross-check immediately.

  PASS 2  geometry, ONE STATE AT A TIME, written per state.
          Bounded memory. Resumable -- already-written states are skipped.

Why the rewrite: reading all ten states' geometry in a single call segfaults.
S_Fld_Haz_Ar across these states is on the order of a million features, some
with >100 parts, and GDAL materialises the lot before handing it back. Chunking
by state is the fix. Reading through /vsizip/ compounded it, so the archive now
has to be extracted first.

Extract with a ZIP64-capable tool -- NOT macOS unzip, which wraps at 2^32:

    ditto -x -k data/raw/nfhl/NFHL_Key_Layers.gdb.zip data/raw/nfhl/
    # or:  7z x data/raw/nfhl/NFHL_Key_Layers.gdb.zip -odata/raw/nfhl/

Usage:
    python subset_nfhl.py              # both passes
    python subset_nfhl.py --attrs      # pass 1 only
    python subset_nfhl.py FL TX        # geometry for specific states
"""
from __future__ import annotations
import os, sys, glob, time
from pathlib import Path

# Must be set before GDAL initialises. The >100-part polygons in this dataset
# make the default ring-organisation path pathologically slow; FileGDB follows
# the ESRI convention (outer ring CW, holes CCW), which is what ONLY_CCW assumes.
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "ONLY_CCW")

import geopandas as gpd
import pandas as pd
import pyogrio

RAW = Path("data/raw/nfhl")
OUT = Path("data/interim/nfhl_states"); OUT.mkdir(parents=True, exist_ok=True)

FIPS = {"FL": "12", "TX": "48", "LA": "22", "IA": "19", "MN": "27",
        "MO": "29", "AZ": "04", "CO": "08", "VT": "50", "NC": "37"}

EXPECTED_A_SHARE = {"IA": 76.7, "FL": 60.9, "LA": 41.1, "TX": 28.9, "MN": 28.3,
                    "AZ": 21.5, "MO": 21.4, "VT": 20.5, "CO": 12.3, "NC": 0.7}

# Everything we need downstream; skipping the rest cuts memory materially.
HAZ_COLS = ["DFIRM_ID", "FLD_AR_ID", "STUDY_TYP", "FLD_ZONE", "ZONE_SUBTY",
            "SFHA_TF", "STATIC_BFE", "DEPTH", "V_DATUM", "DUAL_ZONE", "SOURCE_CIT"]
PAN_COLS = ["DFIRM_ID", "FIRM_PAN", "PANEL", "SUFFIX", "PANEL_TYP",
            "EFF_DATE", "PRE_DATE", "ST_FIPS", "SCALE"]


def find_gdb() -> str:
    hits = sorted(glob.glob(str(RAW / "*.gdb")))
    if not hits:
        sys.exit(
            f"No extracted .gdb under {RAW}.\n"
            f"macOS unzip cannot read this ZIP64 archive -- use:\n"
            f"    ditto -x -k {RAW}/NFHL_Key_Layers.gdb.zip {RAW}/\n"
            f"    # or:  7z x {RAW}/NFHL_Key_Layers.gdb.zip -o{RAW}/\n"
            f"(~40 GB extracted; /vsizip/ was tried and segfaults on this data.)"
        )
    return hits[0]


def resolve_layers(gdb: str) -> tuple[str, str]:
    layers = [l[0] for l in pyogrio.list_layers(gdb)]
    def pick(want: str) -> str:
        for l in layers:
            if l.lower() == want.lower():
                return l
        sys.exit(f"layer {want!r} not in {layers}")
    return pick("S_Fld_Haz_Ar"), pick("S_FIRM_Pan")


def pass1_attributes(gdb: str, haz_layer: str) -> pd.DataFrame:
    """All states, attributes only. Fast and safe."""
    where = " OR ".join(f"DFIRM_ID LIKE '{f}%'" for f in FIPS.values())
    print(f"\n[pass 1] attributes for all 10 states...")
    t0 = time.time()
    df = gpd.read_file(gdb, layer=haz_layer, where=where,
                       columns=HAZ_COLS, ignore_geometry=True)
    inv = {v: k for k, v in FIPS.items()}
    df["STATE"] = df["DFIRM_ID"].str[:2].map(inv)
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s")

    df.to_parquet("data/interim/nfhl_haz_attrs_10state.parquet")

    sfha = df[df["SFHA_TF"] == "T"]
    tab = pd.crosstab(sfha["STATE"], sfha["FLD_ZONE"])
    tab = tab[[c for c in ("A", "AE") if c in tab.columns]]
    tab["A_share_%"] = (100 * tab["A"] / (tab["A"] + tab["AE"])).round(1)
    tab["expected_%"] = tab.index.map(EXPECTED_A_SHARE)
    tab["delta"] = (tab["A_share_%"] - tab["expected_%"]).round(1)

    print("\n" + "=" * 62)
    print("cross-check vs REST go/no-go")
    print("=" * 62)
    print(tab.to_string())
    worst = tab["delta"].abs().max()
    print(f"\nmax delta {worst:.1f}pt "
          f"({'OK' if worst <= 2 else 'INVESTIGATE -- filter or layer may be wrong'})")
    print("(national extract is 2026-03-03, REST is live, so ~1pt drift is normal)")
    return df


def pass2_geometry(gdb: str, haz_layer: str, pan_layer: str, states: list[str]) -> None:
    for abbr in states:
        fips = FIPS[abbr]
        haz_out = OUT / f"haz_{abbr}.parquet"
        pan_out = OUT / f"pan_{abbr}.parquet"
        if haz_out.exists() and pan_out.exists():
            print(f"[{abbr}] already written -- skipping")
            continue

        print(f"\n[{abbr}] reading geometry...", flush=True)
        t0 = time.time()
        try:
            haz = gpd.read_file(gdb, layer=haz_layer,
                                where=f"DFIRM_ID LIKE '{fips}%'",
                                columns=HAZ_COLS)
            haz["STATE"] = abbr
            haz.to_parquet(haz_out)

            pan = gpd.read_file(gdb, layer=pan_layer,
                                where=f"DFIRM_ID LIKE '{fips}%'",
                                columns=PAN_COLS)
            pan["STATE"] = abbr
            pan.to_parquet(pan_out)

            mb = (haz_out.stat().st_size + pan_out.stat().st_size) / 1e6
            print(f"  {len(haz):,} haz + {len(pan):,} panels, "
                  f"{mb:.0f} MB, {time.time()-t0:.0f}s  CRS={haz.crs}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"  -> rerun; completed states are skipped. If {abbr} keeps "
                  f"failing it is the biggest one; see the note below.",
                  file=sys.stderr)


def main(argv: list[str]) -> int:
    attrs_only = "--attrs" in argv
    argv = [a for a in argv if not a.startswith("--")]

    gdb = find_gdb()
    print(f"gdb: {gdb}")
    haz_layer, pan_layer = resolve_layers(gdb)
    print(f"layers: {haz_layer}, {pan_layer}")

    pass1_attributes(gdb, haz_layer)
    if attrs_only:
        return 0

    # Smallest first, so a crash surfaces early and cheaply.
    order = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]
    states = argv or order
    bad = [s for s in states if s not in FIPS]
    if bad:
        sys.exit(f"unknown states: {bad}")

    pass2_geometry(gdb, haz_layer, pan_layer, states)

    done = sorted(p.stem.split("_")[1] for p in OUT.glob("haz_*.parquet"))
    print(f"\ncomplete: {len(done)}/10 states -> {done}")
    if len(done) < 10:
        print("rerun to continue; finished states are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
