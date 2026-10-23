#!/usr/bin/env python3
"""Subset the national NFHL geodatabase into per-state GeoParquet files.

Pass 1 reads the hazard-area attributes (no geometry) of all ten states in one query, writes
data/interim/nfhl_haz_attrs_10state.parquet and checks each state's Zone A share against the
share FEMA's REST service reported. Pass 2 reads geometry one state at a time into
data/interim/nfhl_states/{haz,pan}_{ST}.parquet and skips states already written: reading every
state's geometry in one call segfaults (about a million features, some with >100 parts), and so
does reading through /vsizip/, so the archive must be extracted first, with a ZIP64-capable tool
(macOS unzip wraps at 2^32 bytes):

    ditto -x -k data/raw/nfhl/NFHL_Key_Layers.gdb.zip data/raw/nfhl/

    python subset_nfhl.py              # both passes, all ten states
    python subset_nfhl.py --attrs      # pass 1 only
    python subset_nfhl.py FL           # pass 1, then geometry for FL only
"""
import glob
import os
import sys
import time
from pathlib import Path

# Must be set before GDAL initialises. The default ring organisation is pathologically slow on
# the >100-part polygons here; FileGDB follows the ESRI convention (outer rings clockwise, holes
# counter-clockwise), which is what ONLY_CCW assumes.
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "ONLY_CCW")

import geopandas as gpd
import pandas as pd
import pyogrio

RAW = Path("data/raw/nfhl")
OUT = Path("data/interim/nfhl_states")
ATTRS_OUT = Path("data/interim/nfhl_haz_attrs_10state.parquet")   # read by aotil/o5_polygons.py

FIPS = {"FL": "12", "TX": "48", "LA": "22", "IA": "19", "MN": "27",
        "MO": "29", "AZ": "04", "CO": "08", "VT": "50", "NC": "37"}
# Pass 2 order: smallest first, so a crash surfaces early and cheaply.
GEOMETRY_ORDER = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]

# Zone A share (%) of the A + AE SFHA polygons per state, from FEMA's REST service.
EXPECTED_A_SHARE = {"IA": 76.7, "FL": 60.9, "LA": 41.1, "TX": 28.9, "MN": 28.3,
                    "AZ": 21.5, "MO": 21.4, "VT": 20.5, "CO": 12.3, "NC": 0.7}

# Only the columns needed downstream; skipping the rest cuts memory materially.
HAZ_COLS = ["DFIRM_ID", "FLD_AR_ID", "STUDY_TYP", "FLD_ZONE", "ZONE_SUBTY",
            "SFHA_TF", "STATIC_BFE", "DEPTH", "V_DATUM", "DUAL_ZONE", "SOURCE_CIT"]
PAN_COLS = ["DFIRM_ID", "FIRM_PAN", "PANEL", "SUFFIX", "PANEL_TYP",
            "EFF_DATE", "PRE_DATE", "ST_FIPS", "SCALE"]


def find_gdb() -> str:
    """The first extracted .gdb under RAW; exits with extraction instructions if there is none."""
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
    """The hazard-area and FIRM-panel layer names as the GDB spells them (matched ignoring case)."""
    names = [layer[0] for layer in pyogrio.list_layers(gdb)]
    return _pick_layer(names, "S_Fld_Haz_Ar"), _pick_layer(names, "S_FIRM_Pan")


def _pick_layer(names: list[str], want: str) -> str:
    for name in names:
        if name.lower() == want.lower():
            return name
    sys.exit(f"layer {want!r} not in {names}")


def pass1_attributes(gdb: str, haz_layer: str) -> pd.DataFrame:
    """Every state's hazard-area attributes in one query (no geometry, so fast and safe),
    written to ATTRS_OUT and checked against the expected Zone A shares."""
    clauses = [f"DFIRM_ID LIKE '{fips}%'" for fips in FIPS.values()]
    print(f"\n[pass 1] attributes for all {len(FIPS)} states...")
    t0 = time.time()
    df = gpd.read_file(gdb, layer=haz_layer, where=" OR ".join(clauses),
                       columns=HAZ_COLS, ignore_geometry=True)
    state_of_fips = {fips: abbr for abbr, fips in FIPS.items()}
    df["STATE"] = df["DFIRM_ID"].str[:2].map(state_of_fips)
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s")

    df.to_parquet(ATTRS_OUT)
    print_a_share_check(df)
    return df


def print_a_share_check(df: pd.DataFrame) -> None:
    """Zone A share of each state's A + AE SFHA polygons, against EXPECTED_A_SHARE."""
    sfha = df[df["SFHA_TF"] == "T"]
    tab = pd.crosstab(sfha["STATE"], sfha["FLD_ZONE"])[["A", "AE"]]
    tab["A_share_%"] = (100 * tab["A"] / (tab["A"] + tab["AE"])).round(1)
    tab["expected_%"] = tab.index.map(EXPECTED_A_SHARE)
    tab["delta"] = (tab["A_share_%"] - tab["expected_%"]).round(1)

    print("\n" + "=" * 62)
    print("cross-check vs REST go/no-go")
    print("=" * 62)
    print(tab.to_string())
    worst = tab["delta"].abs().max()
    verdict = "OK" if worst <= 2 else "INVESTIGATE -- filter or layer may be wrong"
    print(f"\nmax delta {worst:.1f}pt ({verdict})")
    print("(national extract is 2026-03-03, REST is live, so ~1pt drift is normal)")


def pass2_geometry(gdb: str, haz_layer: str, pan_layer: str, states: list[str]) -> None:
    """Hazard areas and FIRM panels with geometry, one state per read, each written at once.

    A state whose two files exist is skipped, so a rerun after a crash resumes.
    """
    for abbr in states:
        where = f"DFIRM_ID LIKE '{FIPS[abbr]}%'"
        haz_out = OUT / f"haz_{abbr}.parquet"
        pan_out = OUT / f"pan_{abbr}.parquet"
        if haz_out.exists() and pan_out.exists():
            print(f"[{abbr}] already written -- skipping")
            continue

        print(f"\n[{abbr}] reading geometry...", flush=True)
        t0 = time.time()
        try:
            haz = gpd.read_file(gdb, layer=haz_layer, where=where, columns=HAZ_COLS)
            haz["STATE"] = abbr
            haz.to_parquet(haz_out)

            pan = gpd.read_file(gdb, layer=pan_layer, where=where, columns=PAN_COLS)
            pan["STATE"] = abbr
            pan.to_parquet(pan_out)

            mb = (haz_out.stat().st_size + pan_out.stat().st_size) / 1e6
            print(f"  {len(haz):,} haz + {len(pan):,} panels, "
                  f"{mb:.0f} MB, {time.time()-t0:.0f}s  CRS={haz.crs}")
        except Exception as err:
            print(f"  FAILED: {type(err).__name__}: {err}", file=sys.stderr)
            print("  -> rerun; completed states are skipped.", file=sys.stderr)


def main(argv: list[str]) -> int:
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    attrs_only = "--attrs" in argv
    states = [arg for arg in argv if not arg.startswith("--")]    # any other --flag is ignored
    OUT.mkdir(parents=True, exist_ok=True)

    gdb = find_gdb()
    print(f"gdb: {gdb}")
    haz_layer, pan_layer = resolve_layers(gdb)
    print(f"layers: {haz_layer}, {pan_layer}")

    pass1_attributes(gdb, haz_layer)
    if attrs_only:
        return 0

    states = states or GEOMETRY_ORDER
    unknown = [state for state in states if state not in FIPS]
    if unknown:
        sys.exit(f"unknown states: {unknown}")

    pass2_geometry(gdb, haz_layer, pan_layer, states)

    done = sorted(p.stem.split("_")[1] for p in OUT.glob("haz_*.parquet"))
    print(f"\ncomplete: {len(done)}/{len(FIPS)} states -> {done}")
    if len(done) < len(FIPS):
        print("rerun to continue; finished states are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
