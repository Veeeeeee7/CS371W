#!/usr/bin/env python3
"""
pull_extras.py -- grab two small provenance layers before the GDB is deleted.

S_BFE   Base Flood Elevation lines. In Zone AE the elevations live here as
        line features, not on the polygon (STATIC_BFE is -9999 in ~98% of AE
        polygons). BFE line density inside a polygon is therefore a direct,
        continuous measure of how detailed the study was -- strictly more
        informative than the binary A/AE split, and a strong candidate for the
        alignment battery.

S_LOMR  Letters of Map Revision. Where a map has been amended after
        publication. A revision-activity signal that bears on vintage.

Both are line/small-polygon layers, so this is fast and cheap. Run it BEFORE
deleting the extracted geodatabase -- neither layer can be recovered afterwards
without re-extracting 26 GB.
"""
from __future__ import annotations
import os, sys, glob, time
from pathlib import Path

os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "ONLY_CCW")

import geopandas as gpd
import pyogrio

RAW = Path("data/raw/nfhl")
OUT = Path("data/interim/nfhl_states"); OUT.mkdir(parents=True, exist_ok=True)

FIPS = {"FL": "12", "TX": "48", "LA": "22", "IA": "19", "MN": "27",
        "MO": "29", "AZ": "04", "CO": "08", "VT": "50", "NC": "37"}

WANT = {
    "S_BFE":   ["DFIRM_ID", "BFE_LN_ID", "ELEV", "LEN_UNIT", "V_DATUM", "SOURCE_CIT"],
    "S_LOMR":  ["DFIRM_ID", "LOMR_ID", "EFF_DATE", "CASE_NO", "STATUS", "SCALE"],
}


def main() -> int:
    hits = sorted(glob.glob(str(RAW / "*.gdb")))
    if not hits:
        sys.exit(f"No extracted .gdb under {RAW} -- nothing to pull from.")
    gdb = hits[0]
    print(f"gdb: {gdb}")

    available = {l[0].lower(): l[0] for l in pyogrio.list_layers(gdb)}

    for want, cols in WANT.items():
        layer = available.get(want.lower())
        if layer is None:
            print(f"\n{want}: not present -- skipping")
            continue

        # Not every layer carries every field across FEMA vintages.
        info = pyogrio.read_info(gdb, layer=layer)
        present = [c for c in cols if c in info["fields"]]
        missing = [c for c in cols if c not in info["fields"]]
        if missing:
            print(f"\n{layer}: fields absent, skipping those -> {missing}")

        for abbr, fips in FIPS.items():
            out = OUT / f"{want.lower()}_{abbr}.parquet"
            if out.exists():
                print(f"[{abbr}] {want} already written -- skipping")
                continue
            t0 = time.time()
            try:
                g = gpd.read_file(gdb, layer=layer,
                                  where=f"DFIRM_ID LIKE '{fips}%'",
                                  columns=present)
                g["STATE"] = abbr
                g.to_parquet(out)
                print(f"[{abbr}] {want}: {len(g):>8,} rows, "
                      f"{out.stat().st_size/1e6:6.1f} MB, {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"[{abbr}] {want} FAILED: {type(e).__name__}: {e}",
                      file=sys.stderr)

    print("\nDone. Safe to delete the extracted geodatabase now:")
    print(f"    rm -rf {gdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
