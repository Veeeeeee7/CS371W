#!/usr/bin/env python3
"""The regions of a subset: its components (connected components of the subset in the graph A with at least
c = 20 points; Algorithm alg:spatial), named as the paper names them.

Name of a region: the county that holds most of its points (meta_point_county, the county the point lies in; the
dataset's county_fips is the county of the point's flood-map panel, which a negative sampled near a county line
need not lie in), with the county names of data/interim/gis/counties_FL.parquet; ties go to the smaller FIPS code.
When two regions get the same name, the one with the smaller mean longitude is "(west)" and the other "(east)".

Component ids rank the components by size, largest first, ties to the component holding the lowest OOD column
(graph.component_labels), so the 8 regions of the paper's subset are components 0 to 7.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402

COUNTIES = Path("data/interim/gis/counties_FL.parquet")     # GEOID -> NAME
LONLAT_TOL = 1e-9          # degrees; the cell's lon/lat are copies of the dataset's
POINT_COLS = ["point_id", "domain", "label", "neg_protocol", "lon", "lat", "tile_50km", "county_fips",
              "meta_point_county", "dfirm_id", "meta_fld_ar_id", "meta_src_row_id", "meta_dist_to_river_m",
              "meta_dist_to_coast_m"]


def county_names(path: Path = COUNTIES) -> dict:
    """GEOID (5-digit FIPS) -> county name; an empty dict when the layer is missing (names fall back to FIPS)."""
    if not Path(path).exists():
        return {}
    return dict(pd.read_parquet(path, columns=["GEOID", "NAME"]).values)


def load_points(dataset: Path, features: list[str]) -> pd.DataFrame:
    """The dataset's point table indexed by point_id: POINT_COLS, the features, and x_5070 / y_5070 (metres,
    EPSG:5070, the dataset's own geometry; checked against its GeoParquet CRS)."""
    pf = pq.ParquetFile(dataset)
    geo = json.loads((pf.schema_arrow.metadata or {}).get(b"geo") or b"{}")
    crs = geo.get("columns", {}).get("geometry", {}).get("crs") or {}
    code = (crs.get("id") or {}).get("code")
    if code is not None and int(code) != 5070:
        raise SystemExit(f"{dataset}: geometry CRS EPSG:{code}, expected EPSG:5070")
    names = pf.schema_arrow.names
    cols = [c for c in dict.fromkeys(POINT_COLS + list(features) + ["geometry"]) if c in names]
    df = pq.read_table(dataset, columns=cols).to_pandas()
    xy = shapely.get_coordinates(shapely.from_wkb(df.pop("geometry").to_numpy()))
    df["x_5070"], df["y_5070"] = xy[:, 0], xy[:, 1]
    if not df.point_id.is_unique:
        raise SystemExit(f"{dataset}: point_id is not unique")
    return df.astype({"point_id": str}).set_index("point_id", drop=False)


def ood_rows(points: pd.DataFrame, z) -> pd.DataFrame:
    """The dataset rows of the cell's OOD points, in the cell's column order, checked against the cell."""
    pid = np.asarray(z["point_id"]).astype(str)
    missing = ~pd.Index(pid).isin(points.index)
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} of the cell's OOD points are not in the dataset (wrong dataset?)")
    rows = points.loc[pid].reset_index(drop=True)
    bad = (np.abs(rows.lon.to_numpy() - np.asarray(z["lon"], float)).max() > LONLAT_TOL
           or np.abs(rows.lat.to_numpy() - np.asarray(z["lat"], float)).max() > LONLAT_TOL
           or (rows.label.to_numpy() != np.asarray(z["y_ood"])).any())
    if bad:
        raise SystemExit("the dataset rows do not reproduce the cell's OOD points (lon, lat or label differ)")
    return rows


def point_county(t: pd.DataFrame) -> pd.Series:
    """The county each point lies in: meta_point_county, else county_fips."""
    return t.meta_point_county.astype(object).where(t.meta_point_county.notna(), t.county_fips).astype(str)


def regions(mask: np.ndarray, A, rows: pd.DataFrame, names: dict, c: int = graph.C_MIN) -> dict:
    """The selected points with their component, and one row per component of >= c points with its name.

    rows: the dataset rows in OOD order (ood_rows). Returns dict(points=DataFrame of every selected point,
    ordered by component then OOD column; components=DataFrame of the components of >= c points)."""
    mask = np.asarray(mask, bool)
    comp, sizes = graph.component_labels(mask, A)
    sel = rows[mask].copy()
    sel["ood_index"] = np.flatnonzero(mask)
    sel["component"] = comp.astype(np.int32)
    sel["component_size"] = sizes[comp].astype(np.int32)
    sel["in_component"] = sel.component_size >= c
    sel["county"] = point_county(sel)

    comps = []
    for cid, g in sel[sel.in_component].groupby("component", sort=True):
        counts = g.county.value_counts()
        ranked = sorted(counts.index, key=lambda f: (-counts[f], f))
        tiles = g.tile_50km.astype(str).value_counts()
        tiles = sorted(tiles.index, key=lambda t: (-tiles[t], t))
        dx, dy = np.ptp(g.x_5070.to_numpy()) / 1e3, np.ptp(g.y_5070.to_numpy()) / 1e3
        comps.append(dict(component=int(cid), name=names.get(ranked[0], ranked[0]), county_fips=ranked[0],
                          n_points=len(g), share_flood=float((g.label == 1).mean()),
                          counties=";".join(names.get(f, f) for f in ranked),
                          county_points=";".join(str(int(counts[f])) for f in ranked),
                          majority_tie=bool(len(ranked) > 1 and counts[ranked[0]] == counts[ranked[1]]),
                          tiles=";".join(tiles), n_tiles=len(tiles),
                          lon_center=float(g.lon.mean()), lat_center=float(g.lat.mean()),
                          extent_km=float(np.hypot(dx, dy))))
    comps = pd.DataFrame(comps)
    if len(comps):
        for name, g in comps.groupby("name"):
            if len(g) == 2:                       # same county: west / east by mean longitude
                order = g.sort_values("lon_center").index
                comps.loc[order[0], "name"] = f"{name} (west)"
                comps.loc[order[1], "name"] = f"{name} (east)"
            elif len(g) > 2:
                order = g.sort_values("lon_center").index
                for k, i in enumerate(order, 1):
                    comps.loc[i, "name"] = f"{name} ({k} from west)"
    name_of = dict(zip(comps.component, comps.name)) if len(comps) else {}
    sel["component_name"] = sel.component.map(name_of).fillna("")
    sel = sel.sort_values(["component", "ood_index"], kind="stable").reset_index(drop=True)
    return dict(points=sel, components=comps)
