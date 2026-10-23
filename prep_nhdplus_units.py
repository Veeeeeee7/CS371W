#!/usr/bin/env python3
"""
Build the NHDPlus geodatabase pull_hydro.py reads from the per-unit NHDPlus V2.1 releases.

Florida lies in Vector Processing Units 03N, 03S and 03W, which fetch_nhdplus_fl.sh downloads
as shapefiles and .dbf tables under data/raw/nhdplus/<unit>/. This script writes the two
layers of the national geodatabase, with its names and fields, to
data/raw/nhdplus/NHDPlusV21_{ST}.gdb:

    Catchment            FEATUREID, AreaSqKM                <- Catchment.shp
    NHDFlowline_Network  COMID, LENGTHKM, FTYPE             <- NHDFlowline.shp
                         StreamOrde, TotDASqKM              <- PlusFlowlineVAA.dbf
                         SLOPE (-9998 = no data, kept)      <- elevslope.dbf

Features are kept whole if they touch the state's FIRM-panel extent plus PAD_DEG.
See docs/FEATURE-SOURCES.md section 8.

    python prep_nhdplus_units.py [--overwrite]
"""
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
CRS_GEO = 4269       # NAD83 lon/lat, as NHDPlus ships
PAD_DEG = 0.25       # degrees around the FIRM panels, so nothing near the state line is cut

# pyogrio reads the snapshot's measured lines as 3D and says so once per file.
warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")


def one(pattern: str) -> str:
    """The single file matching a recursive glob; exits if there are none or several."""
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) != 1:
        sys.exit(f"expected exactly one match for {pattern}, found {hits} -- "
                 f"run ./fetch_nhdplus_fl.sh")
    return hits[0]


def state_bbox(state: str) -> tuple:
    """The state's FIRM-panel extent in lon/lat, plus PAD_DEG."""
    panels = gpd.read_parquet(PANELS / f"pan_{state}.parquet").to_crs(CRS_GEO)
    x0, y0, x1, y1 = panels.total_bounds
    return (x0 - PAD_DEG, y0 - PAD_DEG, x1 + PAD_DEG, y1 + PAD_DEG)


def read_unit(unit: str, bbox: tuple) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """One unit's catchments and network flowlines (plus all coastline) inside bbox."""
    root = RAW / unit
    cat = gpd.read_file(one(f"{root}/NHDPlusCatchment/**/Catchment.shp"),
                        columns=["FEATUREID", "AreaSqKM"], bbox=bbox)
    fl = gpd.read_file(one(f"{root}/NHDSnapshot/**/NHDFlowline.shp"),
                       columns=["COMID", "LENGTHKM", "FTYPE"], bbox=bbox)
    vaa = pyogrio.read_dataframe(one(f"{root}/NHDPlusAttributes/**/PlusFlowlineVAA.dbf"),
                                 columns=["ComID", "StreamOrde", "TotDASqKM"],
                                 read_geometry=False)
    vaa = vaa.rename(columns={"ComID": "COMID"})
    slope = pyogrio.read_dataframe(one(f"{root}/NHDPlusAttributes/**/elevslope.dbf"),
                                   columns=["COMID", "SLOPE"], read_geometry=False)

    # The national NHDFlowline_Network holds only flowlines with a VAA record; the snapshot
    # also holds non-network lines (in 03S, the south Florida canal grid), which would
    # otherwise all count as rivers. Coastline is kept whether network or not, so the
    # Keys and barrier islands keep their own shoreline.
    n_in_bbox = len(fl)
    fl = fl[fl.COMID.isin(vaa.COMID) | (fl.FTYPE == "Coastline")]
    fl = fl.merge(vaa, on="COMID", how="left").merge(slope, on="COMID", how="left")
    fl["geometry"] = shapely.force_2d(fl.geometry.values)
    print(f"  {unit}: {len(cat):,} catchments, {len(fl):,} network flowlines and coastline "
          f"of {n_in_bbox:,} in the bbox; FTYPE {fl.FTYPE.value_counts().to_dict()}",
          flush=True)
    return cat, gpd.GeoDataFrame(fl, geometry="geometry", crs=cat.crs)


def write_geodatabase(cat: gpd.GeoDataFrame, fl: gpd.GeoDataFrame, out: Path) -> None:
    """Write both layers to a staging .gdb outside RAW, then move it to `out` whole.

    A failed run then never leaves a partial .gdb where pull_hydro.find_gdb() would pick it
    up. promote_to_multi because FileGDB needs one geometry type per layer.
    """
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


def build(state: str, overwrite: bool) -> int:
    out = RAW / f"NHDPlusV21_{state}.gdb"
    if out.exists() and not overwrite:
        print(f"{out} exists -- skipping (use --overwrite)")
        return 0
    others = []
    for gdb in glob.glob(str(RAW / "**" / "*.gdb"), recursive=True):
        if Path(gdb) != out:
            others.append(gdb)
    if others:
        sys.exit(f"other geodatabases under {RAW}: {others} -- pull_hydro.py reads the "
                 f"first .gdb it finds; move them aside first")

    t0 = time.time()
    bbox = state_bbox(state)
    x0, y0, x1, y1 = bbox
    print(f"[{state}] units {UNITS[state]}, bbox ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})")
    cat_parts, fl_parts = [], []
    for unit in UNITS[state]:
        unit_cat, unit_fl = read_unit(unit, bbox)
        cat_parts.append(unit_cat)
        fl_parts.append(unit_fl)
    cat = pd.concat(cat_parts, ignore_index=True)
    fl = pd.concat(fl_parts, ignore_index=True)

    # A feature on a unit boundary can appear in both units' files.
    dup_catchments = int(cat.FEATUREID.duplicated().sum())
    dup_flowlines = int(fl.COMID.duplicated().sum())
    cat = cat.drop_duplicates("FEATUREID")
    fl = fl.drop_duplicates("COMID")
    cat["FEATUREID"] = cat["FEATUREID"].astype(np.int64)
    fl["COMID"] = fl["COMID"].astype(np.int64)

    write_geodatabase(cat, fl, out)

    slope = pd.to_numeric(fl.SLOPE, errors="coerce")
    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"[{state}] wrote {out}: {len(cat):,} catchments, {len(fl):,} flowlines "
          f"(dropped {dup_catchments} duplicate catchments, {dup_flowlines} duplicate "
          f"flowlines); SLOPE null {int(slope.isna().sum()):,}, "
          f"-9998 {int((slope == -9998).sum()):,}; {size_mb:.0f} MB "
          f"({time.time() - t0:.0f}s)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-unit NHDPlus V2.1 -> NHDPlusV21_{ST}.gdb.")
    ap.add_argument("state", nargs="?", default="FL", choices=sorted(UNITS))
    ap.add_argument("--overwrite", action="store_true", help="rebuild an existing .gdb")
    args = ap.parse_args(argv)
    return build(args.state, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
