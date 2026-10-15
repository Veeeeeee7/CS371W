#!/usr/bin/env python3
"""
prep_nhdplus_units.py -- NHDPlus V2.1 per-unit releases -> the one geodatabase
pull_hydro.py reads.

pull_hydro.py was written against the national seamless geodatabase (7.81 GB, about
20 GB extracted), which no longer fits on this disk. Florida needs only Vector
Processing Units 03N, 03S and 03W (547 MB, fetch_nhdplus_fl.sh). Those ship as
shapefiles plus attribute tables rather than a geodatabase, so this script rebuilds
the two layers pull_hydro.py looks for, with the national names and fields:

    Catchment            FEATUREID, AreaSqKM                     <- Catchment.shp
    NHDFlowline_Network  COMID, LENGTHKM, FTYPE                  <- NHDFlowline.shp
                         StreamOrde, TotDASqKM                   <- PlusFlowlineVAA.dbf
                         SLOPE                                   <- elevslope.dbf

WHY ONLY NETWORK FLOWLINES
==========================
The national NHDFlowline_Network layer is the flowlines that have a value-added
attribute (VAA) record; the rest sit in NHDFlowline_NonNetwork. The snapshot
shapefile holds both, so it is cut to the COMIDs in PlusFlowlineVAA. That is not
a detail in Florida: unit 03S carries 62,411 non-network CanalDitch lines (the
south Florida canal grid), which would otherwise all become "rivers" and change
dist_to_river for every point near them. Coastline survives the cut: 7,326 of its
8,103 features are network flowlines, which is what v1 measured coast distance
against.

SLOPE keeps NHDPlus's own no-data code (-9998) exactly as the national layer
did, so v1 and v2 treat it the same way; network flowlines with no elevslope row
get a null.

Features are kept whole if they intersect the state's FIRM-panel extent plus a
pad, so nothing near the state line is cut; pull_hydro.py applies its own
bounding box on top.

USAGE
=====
    ./fetch_nhdplus_fl.sh                     # download + extract the 9 archives
    python prep_nhdplus_units.py              # -> data/raw/nhdplus/NHDPlusV21_FL.gdb
    python prep_nhdplus_units.py --overwrite
    python pull_hydro.py --inspect            # must find exactly this .gdb
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

RAW = Path("data/raw/nhdplus")
PANELS = Path("data/interim/nfhl_states")
UNITS = {"FL": ["03N", "03S", "03W"]}
PAD_DEG = 0.25

# pyogrio reads the snapshot's measured lines as 3D and says so once per file.
warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")


def one(pattern: str) -> str:
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) != 1:
        sys.exit(f"expected exactly one match for {pattern}, found {hits} -- "
                 f"run ./fetch_nhdplus_fl.sh")
    return hits[0]


def state_bbox(state: str) -> tuple:
    pan = gpd.read_parquet(PANELS / f"pan_{state}.parquet").to_crs(4269)
    x0, y0, x1, y1 = pan.total_bounds
    return (x0 - PAD_DEG, y0 - PAD_DEG, x1 + PAD_DEG, y1 + PAD_DEG)


def read_unit(unit: str, bbox: tuple) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    root = RAW / unit
    cat = gpd.read_file(one(f"{root}/NHDPlusCatchment/**/Catchment.shp"),
                        columns=["FEATUREID", "AreaSqKM"], bbox=bbox)
    fl = gpd.read_file(one(f"{root}/NHDSnapshot/**/NHDFlowline.shp"),
                       columns=["COMID", "LENGTHKM", "FTYPE"], bbox=bbox)
    vaa = pyogrio.read_dataframe(one(f"{root}/NHDPlusAttributes/**/PlusFlowlineVAA.dbf"),
                                 columns=["ComID", "StreamOrde", "TotDASqKM"],
                                 read_geometry=False).rename(columns={"ComID": "COMID"})
    es = pyogrio.read_dataframe(one(f"{root}/NHDPlusAttributes/**/elevslope.dbf"),
                                columns=["COMID", "SLOPE"], read_geometry=False)

    n_all = len(fl)
    fl = fl[fl.COMID.isin(vaa.COMID)]
    fl = fl.merge(vaa, on="COMID", how="left").merge(es, on="COMID", how="left")
    fl["geometry"] = shapely.force_2d(fl.geometry.values)
    print(f"  {unit}: {len(cat):,} catchments, {len(fl):,} network flowlines "
          f"of {n_all:,} in the bbox; FTYPE {fl.FTYPE.value_counts().to_dict()}",
          flush=True)
    return cat, gpd.GeoDataFrame(fl, geometry="geometry", crs=cat.crs)


def build(state: str, overwrite: bool) -> int:
    out = RAW / f"NHDPlusV21_{state}.gdb"
    if out.exists() and not overwrite:
        print(f"{out} exists -- skipping (use --overwrite)")
        return 0
    others = [g for g in glob.glob(str(RAW / "**" / "*.gdb"), recursive=True)
              if Path(g) != out]
    if others:
        # pull_hydro.find_gdb() takes the first .gdb it finds.
        sys.exit(f"other geodatabases under {RAW}: {others} -- pull_hydro.py would "
                 f"read the first one; move them aside first")

    t0 = time.time()
    bbox = state_bbox(state)
    print(f"[{state}] units {UNITS[state]}, bbox {tuple(round(v, 2) for v in bbox)}")
    cats, fls = zip(*(read_unit(u, bbox) for u in UNITS[state]))
    cat = pd.concat(cats, ignore_index=True)
    fl = pd.concat(fls, ignore_index=True)

    # A feature on a unit boundary can appear in both units' files.
    dup_c, dup_f = int(cat.FEATUREID.duplicated().sum()), int(fl.COMID.duplicated().sum())
    cat = cat.drop_duplicates("FEATUREID")
    fl = fl.drop_duplicates("COMID")
    cat["FEATUREID"] = cat["FEATUREID"].astype(np.int64)
    fl["COMID"] = fl["COMID"].astype(np.int64)

    # Written outside RAW and moved in whole, so a failed run never leaves a
    # partial .gdb where find_gdb() would pick it up. promote_to_multi because the
    # FileGDB driver needs one geometry type per layer, as the national file has.
    tmp = RAW.parent / "nhdplus_staging" / out.name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(cat, tmp, layer="Catchment", driver="OpenFileGDB",
                            promote_to_multi=True)
    pyogrio.write_dataframe(fl, tmp, layer="NHDFlowline_Network", driver="OpenFileGDB",
                            promote_to_multi=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(tmp), str(out))
    tmp.parent.rmdir()

    slope = pd.to_numeric(fl.SLOPE, errors="coerce")
    print(f"[{state}] wrote {out}: {len(cat):,} catchments, {len(fl):,} flowlines "
          f"(dropped {dup_c} duplicate catchments, {dup_f} duplicate flowlines); "
          f"SLOPE null {int(slope.isna().sum()):,}, -9998 {int((slope == -9998).sum()):,}; "
          f"{sum(f.stat().st_size for f in out.iterdir())/1e6:.0f} MB "
          f"({time.time()-t0:.0f}s)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?", default="FL", choices=sorted(UNITS))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    return build(a.state, a.overwrite)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
