#!/usr/bin/env python3
"""Save two small NFHL layers from the extracted national geodatabase before it is deleted.

S_BFE   Base Flood Elevation lines. In Zone AE the elevations live here, not on the polygon
        (STATIC_BFE is -9999 in ~98% of AE polygons), so BFE line density measures how
        detailed a study was (sample_points.py reads it).
S_LOMR  Letters of Map Revision: where a map was amended after publication
        (aotil/o5_polygons.py reads it).

Input   data/raw/nfhl/*.gdb (the first in sorted order)
Output  data/interim/nfhl_states/s_bfe_FL.parquet, s_lomr_FL.parquet (existing ones are skipped)

Run it before deleting the geodatabase: neither layer can be recovered without re-extracting
26 GB. It prints whether deleting is safe.
    python pull_extras.py
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Set before GDAL loads. ONLY_CCW makes GDAL take counter-clockwise rings as holes instead of
# testing every ring against every other, which is slow on many-part polygons; FileGDB follows
# that ESRI ring convention (outer rings clockwise, holes counter-clockwise).
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "ONLY_CCW")

import geopandas as gpd
import pyogrio

RAW = Path("data/raw/nfhl")
OUT = Path("data/interim/nfhl_states")

STATE = "FL"
STATE_FIPS = "12"           # DFIRM_IDs start with the state's FIPS code

# Layer -> the fields to keep.
WANT = {
    "S_BFE":   ["DFIRM_ID", "BFE_LN_ID", "ELEV", "LEN_UNIT", "V_DATUM", "SOURCE_CIT"],
    "S_LOMR":  ["DFIRM_ID", "LOMR_ID", "EFF_DATE", "CASE_NO", "STATUS", "SCALE"],
}


def output_path(layer: str) -> Path:
    return OUT / f"{layer.lower()}_{STATE}.parquet"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gdbs = sorted(glob.glob(str(RAW / "*.gdb")))
    if not gdbs:
        sys.exit(f"No extracted .gdb under {RAW} -- nothing to pull from.")
    gdb = gdbs[0]
    print(f"gdb: {gdb}")

    # Layer names are matched case-insensitively.
    layer_names = {}
    for row in pyogrio.list_layers(gdb):      # rows of (name, geometry type)
        layer_names[row[0].lower()] = row[0]

    failed = []
    for want, cols in WANT.items():
        layer = layer_names.get(want.lower())
        if layer is None:
            print(f"\n{want}: not present -- skipping")
            failed.append(f"{want} (layer not in the GDB)")
            continue

        # Not every layer carries every field across FEMA vintages.
        fields = pyogrio.read_info(gdb, layer=layer)["fields"]
        present = [c for c in cols if c in fields]
        missing = [c for c in cols if c not in fields]
        if missing:
            print(f"\n{layer}: fields absent, skipping those -> {missing}")

        out = output_path(want)
        if out.exists():
            print(f"[{STATE}] {want} already written -- skipping")
            continue
        t0 = time.time()
        try:
            table = gpd.read_file(gdb, layer=layer, where=f"DFIRM_ID LIKE '{STATE_FIPS}%'",
                                  columns=present)
            table["STATE"] = STATE
            table.to_parquet(out)
            print(f"[{STATE}] {want}: {len(table):>8,} rows, "
                  f"{out.stat().st_size/1e6:6.1f} MB, {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"[{STATE}] {want} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(f"{want} {STATE} ({type(e).__name__})")

    # Advise deleting the GDB only when every expected output exists.
    absent = []
    for want in WANT:
        if not output_path(want).exists():
            absent.append(output_path(want).name)
    if failed or absent:
        print(f"\nNOT safe to delete the GDB {gdb}: failed {failed or 'none'}; "
              f"outputs missing from {OUT}: {absent or 'none'}")
        return 1
    print("\nDone. Safe to delete the extracted geodatabase now:")
    print(f"    rm -rf {gdb}")
    return 0


if __name__ == "__main__":
    # The script takes no options; the parser only gives it a --help.
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
