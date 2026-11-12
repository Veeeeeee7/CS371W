#!/usr/bin/env python3
"""Export the regions (components) of one subset search for GIS error analysis: a GeoPackage, CSVs, a README, zipped.

The regions are the connected components of at least --min-size points of the subset a search selects (the paper's
analyzed method is GTV + smoothing at S = 500, mu = 3, c = 20). For each region point the export gives its location,
its FEMA attributes, the nine model features as the models see them (plus readable versions), the share of the 290
models that classify it correctly, and its caliper matches among the held-out Zone AE points (the paper's Region
Analysis). Output: exports/<method>_S<S>_mu<mu>_c<c>_<run date>/ and a .zip of it.

Examples:
  .venv/bin/python export_components.py --method "GTV+smoothing"                  # the paper's regions, latest run
  .venv/bin/python export_components.py --method Laplacian --S 1000 --run results/2026-11-12
Methods: OODSelect, OODSelect+smoothing, Laplacian, Laplacian+smoothing, GTV, GTV+smoothing (mu is ignored for
OODSelect, which is mu = 0). Needs a run folder with the stages zoo and search (and smooth for the stored masks).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aotil"))
import graph  # noqa: E402
import region_match  # noqa: E402
import regions  # noqa: E402
import subsets  # noqa: E402
from search import unit_name  # noqa: E402
from zoo import read_features  # noqa: E402

TAG = "FL_band_AE_s0"
METHODS = {"OODSelect": ("oodselect", False), "OODSelect+smoothing": ("oodselect", True),
           "Laplacian": ("laplacian", False), "Laplacian+smoothing": ("laplacian", True),
           "GTV": ("gtv", False), "GTV+smoothing": ("gtv", True)}
NLCD = {11: "Open Water", 12: "Perennial Ice/Snow", 21: "Developed, Open Space", 22: "Developed, Low Intensity",
        23: "Developed, Medium Intensity", 24: "Developed, High Intensity", 31: "Barren Land (Rock/Sand/Clay)",
        41: "Deciduous Forest", 42: "Evergreen Forest", 43: "Mixed Forest", 52: "Shrub/Scrub",
        71: "Grassland/Herbaceous", 81: "Pasture/Hay", 82: "Cultivated Crops", 90: "Woody Wetlands",
        95: "Emergent Herbaceous Wetlands"}
FLOODFREQ = {0: "None", 1: "Very rare", 2: "Rare", 3: "Occasional", 4: "Frequent", 5: "Very frequent"}
FEATURES9 = ["landcover", "soil_floodfreq", "soil_fluvclass", "slope", "elev_sd_r300", "hand", "wetindexcat",
             "log10_dist_river", "log10_dist_coast"]
BASE_COLS = ["component_id", "component_name", "point_id", "lon", "lat", "x_5070", "y_5070", "zone", "label",
             "fema_polygon_id", "dfirm_id", "county", "county_fips", "tile_50km", *FEATURES9, "landcover_name",
             "soil_floodfreq_name", "dist_river_m", "dist_coast_m", "share_models_correct"]
POINT_COLS = BASE_COLS + ["n_matches", "match_label_disagreement"]

DICTIONARY = [
    ("component_id", "Region number: 0 is the largest region, ties to the region holding the lowest point index. "
                     "In selected_not_in_component: -1."),
    ("component_name", "The county holding most of the region's points; (west)/(east) by mean longitude when two "
                       "regions share a county."),
    ("point_id", "The point's id in the project's dataset (data/processed_v4/dataset_FL.parquet)."),
    ("lon, lat", "Longitude and latitude in degrees, NAD83 (EPSG:4269), as in the dataset."),
    ("x_5070, y_5070", "The same point in metres, CONUS Albers Equal Area (EPSG:5070), the dataset's geometry."),
    ("zone", "FEMA flood zone the point was sampled for: A (approximate studies). All region points are Zone A test "
             "points."),
    ("label", "1 = inside a FEMA Zone A polygon (a sampled positive); 0 = a negative sampled 500 m to 5 km outside "
              "any mapped 1% annual chance floodplain."),
    ("fema_polygon_id", "FLD_AR_ID of the National Flood Hazard Layer flood-hazard polygon a positive (label 1) was "
                        "sampled inside; empty for a negative (label 0), which lies outside every floodplain."),
    ("dfirm_id", "DFIRM_ID of the FEMA flood map database of that polygon (for a negative: of the Zone A polygon "
                 "whose 500 m to 5 km band produced it)."),
    ("county, county_fips", "The county the point lies in (Census TIGER county polygons) and its 5-digit FIPS code. "
                            "For about 14% of negatives this differs from the county of their dfirm_id, the map "
                            "of the polygon they were sampled around."),
    ("tile_50km", "The 50 km tile of the train/test split, '<i>_<j>' = floor(x_5070 / 50,000)_floor(y_5070 / 50,000)."),
    ("landcover", "NLCD 2021 land cover class code of the 30 m cell holding the point (one-hot encoded for the "
                  "models); landcover_name is the NLCD legend name."),
    ("soil_floodfreq", "SSURGO flooding frequency class of the map unit (muaggatt.flodfreqdcd, dominant condition): "
                       "0 None, 1 Very rare, 2 Rare, 3 Occasional, 4 Frequent, 5 Very frequent; empty in a water map "
                       "unit or outside every map unit; soil_floodfreq_name is the class name."),
    ("soil_fluvclass", "1 if the map unit's dominant soil component is in a fluvial (river-deposit) subgroup, else 0; "
                       "empty where soil_floodfreq is empty."),
    ("slope", "Slope in degrees from the USGS 3DEP 10 m elevation model."),
    ("elev_sd_r300", "Terrain variability: the standard deviation, in metres, of the 3DEP 10 m elevations within "
                     "300 m of the point."),
    ("hand", "Height above the nearest drainage in metres, from MERIT Hydro (90 m)."),
    ("wetindexcat", "Topographic wetness index of the point's NHDPlus V2 catchment, the EPA StreamCat catchment mean, "
                    "times 100 (as StreamCat serves it; divide by 100 for the index); empty where the point is in no "
                    "catchment."),
    ("log10_dist_river, dist_river_m", "Straight-line distance in metres (EPSG:5070) to the nearest NHDPlus V2 river "
                                       "flowline (stream/river, artificial path, connector or canal/ditch); the model "
                                       "feature is log10 of it, floored at 1 m."),
    ("log10_dist_coast, dist_coast_m", "Straight-line distance in metres (EPSG:5070) to the nearest NHDPlus V2 "
                                       "coastline; the model feature is log10 of it, floored at 1 m."),
    ("share_models_correct", "Share of the 290 models (all model families) whose prediction (probability > 0.5) "
                             "matches the point's label."),
    ("n_matches", "Number of matched held-out Zone AE test points (0 to 5): same land cover, standardized "
                  "Euclidean distance on the six numeric features at most the caliper, in a test tile holding no "
                  "point of the region."),
    ("match_label_disagreement", "Share of the point's matches whose label differs from the point's; empty when "
                                 "the point has no match. Averaged over a region's matched points it is the "
                                 "paper's eta."),
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def latest_complete() -> Path:
    runs = [m.parent for m in Path("results").glob("*/manifest.json")
            if json.loads(m.read_text()).get("complete")]
    if not runs:
        raise SystemExit("no complete run under results/: run `run_experiments.py --stage all` first, or pass --run")
    return max(runs, key=lambda p: p.name)


def subset_mask(run: Path, method: str, smoothing: bool, mu: float, S: int, A) -> np.ndarray:
    """The selected subset: the stored mask of the smooth stage, else recomputed from the search's theta."""
    unit = unit_name(method, mu, S)
    key = unit + ("+smoothing" if smoothing else "")
    mp = run / "smooth" / "masks.npz"
    if mp.exists():
        m = np.load(mp)
        if key in m.files:
            return np.unpackbits(m[key])[: int(m["d"])].astype(bool)
    theta_path = run / "search" / unit / "theta.npy"
    if not theta_path.exists() or not (run / "search" / unit / "result.json").exists():
        raise SystemExit(f"{run}/search/{unit} is missing: run `run_experiments.py --stage search --only {unit} "
                         f"--out {run}`")
    theta = np.load(theta_path)
    return subsets.smooth(theta, A, S)[0] if smoothing else subsets.top_mask(theta, S)


def readable(df: pd.DataFrame, names: dict, correct: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(dict(
        point_id=df.point_id.astype(str).values, lon=df.lon.values, lat=df.lat.values, x_5070=df.x_5070.values,
        y_5070=df.y_5070.values, zone="A", label=df.label.astype(int).values,
        fema_polygon_id=df.meta_fld_ar_id.astype("string").fillna("").values,
        dfirm_id=df.dfirm_id.astype("string").fillna("").values,
        county_fips=df.county.astype(str).values, tile_50km=df.tile_50km.astype(str).values))
    out["county"] = out.county_fips.map(lambda f: names.get(f, f))
    for f in FEATURES9:
        out[f] = df[f].values
    out["landcover_name"] = out.landcover.map(lambda c: NLCD.get(int(c), "") if pd.notna(c) else "")
    out["soil_floodfreq_name"] = out.soil_floodfreq.map(lambda c: FLOODFREQ.get(int(c), "") if pd.notna(c) else "")
    out["dist_river_m"] = df.meta_dist_to_river_m.values
    out["dist_coast_m"] = df.meta_dist_to_coast_m.values
    out["share_models_correct"] = correct
    return out


def readme(info: dict) -> str:
    rows = "\n".join(f"| `{c}` | {t} |" for c, t in DICTIONARY)
    reg = "\n".join(f"| {r.component_id} | {r.component_name} | {r.n_points} | {r.unmatched_share:.2f} | {r.eta:.2f} |"
                    for r in info["summary"].itertuples())
    return f"""# Regions of {info['method']} at S = {info['S']} (mu = {info['mu']:g}, c = {info['c']})

## What these regions are

A *region* is a place in Florida where models that are better at predicting FEMA's detailed flood maps (Zone AE) are
*worse* at predicting FEMA's approximate flood maps (Zone A). We trained {info['n_models']} models on Zone AE points and
searched the {info['m']:,} Zone A test points for a subset of {info['S']} points on which the usual pattern (better models
do better everywhere) reverses; a spatial penalty and smoothing make the subset form connected groups of nearby points.
Each region here is one connected group of at least {info['c']} selected points (in the 10-nearest-neighbour graph of
the Zone A test points), {info['n_regions']} regions holding {info['n_points']} of the {info['S']} selected points; the
other {info['n_outside']} selected points belong to no region. For every region point, `matches.csv` lists held-out
Zone AE points with nearly the same inputs: when they carry a different label, the two mapping procedures label the
same kind of land differently there.

## How they were produced

- Method: {info['method']} (`{info['unit']}`), budget S = {info['S']}, regularization weight mu = {info['mu']:g}, minimum
  region size c = {info['c']} points, 10-nearest-neighbour graph in EPSG:3086.
- Run folder: `{info['run']}` (git commit `{info['commit']}`), produced by `run_experiments.py`; exported by
  `export_components.py` on {info['date']}.
- Matching: caliper eps = {info['eps']:.3f} standardized units on the six numeric features (slope, terrain variability,
  height above drainage, wetness, log distance to river and to coast, standardized over the Zone A test points), same
  land cover, at most 5 matches, candidates only in test tiles that hold no point of the region.

| component_id | name | points | unmatched share | eta (label disagreement) |
|---|---|---|---|---|
{reg}

## Coordinates

- `lon`, `lat`: degrees, NAD83 (EPSG:4269), as in the dataset.
- `x_5070`, `y_5070` and every layer of `regions.gpkg`: metres, CONUS Albers Equal Area (EPSG:5070).

## Files

- `regions.gpkg` (GeoPackage, EPSG:5070): layer `points` (every point in a region), layer `regions` (one convex-hull
  polygon per region, for display only: the hull is not a mapped boundary and can include land between the points),
  layer `selected_not_in_component` (selected points outside every region), layer `matches` (the matched Zone AE
  points, one row per match, with the region point they match).
- `points.csv`: one row per point in a region, sorted by region then point id.
- `components/<NN>_<name>.csv`: the same rows, one file per region (NN = rank by size).
- `components_summary.csv`: one row per region: id, name, number of points, counties, tiles, centroid lon/lat,
  extent (bounding-box diagonal in km, EPSG:5070), share labeled flood, unmatched share and eta (the paper's Table 3).
- `selected_not_in_component.csv`: the selected points outside every region (`component_id` = -1; `piece_id` and
  `piece_size` give the small connected piece each belongs to).
- `matches.csv`: one row per (region point, matched Zone AE point): the region point's id and label, the matched point's
  id, lon/lat (NAD83), x/y (EPSG:5070) and label, the feature distance (standardized units) and the match rank.

## Columns of the point files

| column | meaning |
|---|---|
{rows}

## Opening the files

- QGIS: Layer > Add Layer > Add Vector Layer, choose `regions.gpkg`, and tick the layers to add (or drag the file
  into the map). The CSVs: Layer > Add Layer > Add Delimited Text Layer, X field `lon`, Y field `lat`, CRS EPSG:4269
  (or X `x_5070`, Y `y_5070`, CRS EPSG:5070).
- ArcGIS Pro: Catalog > Add Folder Connection to this folder, expand `regions.gpkg` and drag a layer onto the map. For
  a CSV: Map > Add Data > XY Point Data (XY Table To Point), X `lon`, Y `lat`, coordinate system NAD 1983 (EPSG:4269).
- Python: `import geopandas as gpd; pts = gpd.read_file("regions.gpkg", layer="points")` (layers `points`, `regions`,
  `selected_not_in_component`, `matches`); `import pandas as pd; df = pd.read_csv("points.csv")`.
- Excel: open the CSVs directly (comma-separated, UTF-8, one header row).
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=list(METHODS), help="the search whose regions to export")
    ap.add_argument("--S", type=int, default=500, help="budget (default 500)")
    ap.add_argument("--mu", type=float, default=3.0, help="regularization weight of the Laplacian and GTV (default 3)")
    ap.add_argument("--min-size", type=int, default=20, help="c, the minimum region size (default 20)")
    ap.add_argument("--run", type=Path, help="run folder (default: the latest complete one under results/)")
    ap.add_argument("--out-root", type=Path, default=Path("exports"), help="parent folder (default exports/)")
    ap.add_argument("--data", type=Path, default=Path("data/processed_v4"), help=argparse.SUPPRESS)
    ap.add_argument("--counties", type=Path, default=regions.COUNTIES, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    run = a.run or latest_complete()
    man_path = run / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"{run} is not a run folder (no manifest.json)")
    man = json.loads(man_path.read_text())
    method, smoothing = METHODS[a.method]
    mu = 0.0 if method == "oodselect" else a.mu
    zoo_npz = run / "zoo" / f"{TAG}.npz"
    if not zoo_npz.exists():
        raise SystemExit(f"{zoo_npz} is missing: run `run_experiments.py --stage zoo --out {run}`")
    date = run.name if re.match(r"^\d{4}-\d{2}-\d{2}", run.name) else man.get("created", "")[:10]
    folder = a.out_root / f"{a.method}_S{a.S}_mu{mu:g}_c{a.min_size}_{date}"
    print(f"export_components: {a.method}, S = {a.S}, mu = {mu:g}, c = {a.min_size}, run {run} -> {folder}")

    z = dict(np.load(zoo_npz, allow_pickle=True))
    lon, lat = np.asarray(z["lon"], float), np.asarray(z["lat"], float)
    A = graph.build_adjacency(lon, lat, k=10)
    mask = subset_mask(run, method, smoothing, mu, a.S, A)
    feats = read_features(a.data / "FEATURES.txt")
    points = regions.load_points(a.data / "dataset_FL.parquet", feats)
    rows = regions.ood_rows(points, z)
    names = regions.county_names(a.counties)
    reg = regions.regions(mask, A, rows, names, a.min_size)
    pts, comps = reg["points"], reg["components"]
    arrays = subsets.cell_arrays(z)
    correct = arrays["Z"].mean(0)                       # share of all models correct at each OOD point

    idd = points.loc[np.asarray(z["point_id_id"]).astype(str)].reset_index(drop=True)
    members = {int(c): g.ood_index.to_numpy() for c, g in pts[pts.in_component].groupby("component")}
    match = region_match.run(region_match.prepare(rows, idd, region_match.numeric_features(a.data / "FEATURES.txt")),
                             mask, members) if members else None

    inside = pts[pts.in_component]
    out_pts = readable(inside, names, correct[inside.ood_index.to_numpy()])
    out_pts.insert(0, "component_name", inside.component_name.values)
    out_pts.insert(0, "component_id", inside.component.astype(int).values)
    if match is not None:
        ps = match["point_summary"].set_index("point_id")
        out_pts["n_matches"] = out_pts.point_id.map(ps.n_matches).fillna(0).astype(int)
        out_pts["match_label_disagreement"] = out_pts.point_id.map(ps.match_label_disagreement)
    else:
        out_pts["n_matches"], out_pts["match_label_disagreement"] = 0, np.nan
    out_pts = out_pts[POINT_COLS].sort_values(["component_id", "point_id"], kind="stable").reset_index(drop=True)

    outside = pts[~pts.in_component]
    out_rest = readable(outside, names, correct[outside.ood_index.to_numpy()])
    out_rest.insert(0, "component_name", "")
    out_rest.insert(0, "component_id", -1)
    out_rest = out_rest[BASE_COLS]
    out_rest["piece_id"] = outside.component.astype(int).values
    out_rest["piece_size"] = outside.component_size.astype(int).values
    out_rest = out_rest.sort_values(["piece_id", "point_id"], kind="stable").reset_index(drop=True)

    if match is not None:
        tab = match["table"]
        reg_rows = tab[tab.kind == "region"].set_index("component")
        pairs = match["pairs"]
        pool = points.loc[pairs.match_point_id.astype(str)]
        matches = pd.DataFrame(dict(component_id=pairs.component.astype(int).values,
                                    point_id=pairs.point_id.astype(str).values, label=pairs.label.values,
                                    match_rank=pairs["rank"].values, match_point_id=pairs.match_point_id.astype(str).values,
                                    match_lon=pool.lon.values, match_lat=pool.lat.values,
                                    match_x_5070=pool.x_5070.values, match_y_5070=pool.y_5070.values,
                                    match_label=pairs.match_label.values, match_zone="AE",
                                    feature_distance=pairs.distance.values))
        matches = matches.sort_values(["component_id", "point_id", "match_rank"], kind="stable").reset_index(drop=True)
        eps = match["summary"]["eps"]
    else:
        reg_rows, matches, eps = pd.DataFrame(), pd.DataFrame(columns=["component_id"]), float("nan")
    summary = pd.DataFrame([dict(
        component_id=int(c.component), component_name=c["name"], n_points=int(c.n_points), counties=c.counties,
        tiles=c.tiles, centroid_lon=c.lon_center, centroid_lat=c.lat_center, extent_km=c.extent_km,
        share_flood=c.share_flood,
        n_matched=int(reg_rows.loc[c.component, "n_matched"]) if len(reg_rows) else 0,
        unmatched_share=float(reg_rows.loc[c.component, "unmatched"]) if len(reg_rows) else np.nan,
        eta=float(reg_rows.loc[c.component, "eta"]) if len(reg_rows) else np.nan) for _, c in comps.iterrows()])

    # Against the run's Table 3 when this is the analyzed method.
    check = "not the run's analyzed method (no Table 3 to compare)"
    table3 = run / "match" / "table.csv"
    if (a.method, a.S, mu, a.min_size) == ("GTV+smoothing", 500, 3.0, 20) and table3.exists() and len(summary):
        t3 = pd.read_csv(table3)
        t3 = t3[t3.kind == "region"].set_index("component")
        same = all(int(t3.loc[r.component_id, "n"]) == r.n_points
                   and abs(t3.loc[r.component_id, "unmatched"] - r.unmatched_share) < 1e-12
                   and abs(t3.loc[r.component_id, "eta"] - r.eta) < 1e-12 for r in summary.itertuples())
        check = f"{'MATCH' if same and len(t3) == len(summary) else 'DIFFERENT'} with {table3}"
        if not same:
            raise SystemExit(f"the exported regions differ from {table3}; nothing written")

    if folder.exists():
        shutil.rmtree(folder)
    (folder / "components").mkdir(parents=True)
    out_pts.to_csv(folder / "points.csv", index=False)
    for cid, g in out_pts.groupby("component_id"):
        g.to_csv(folder / "components" / f"{cid + 1:02d}_{slug(g.component_name.iloc[0])}.csv", index=False)
    summary.to_csv(folder / "components_summary.csv", index=False)
    out_rest.to_csv(folder / "selected_not_in_component.csv", index=False)
    matches.to_csv(folder / "matches.csv", index=False)

    import geopandas as gpd
    import shapely
    gpkg = folder / "regions.gpkg"

    def gdf(df, x, y):
        return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[x], df[y]), crs=5070)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf(out_pts, "x_5070", "y_5070").to_file(gpkg, layer="points", driver="GPKG")
        hulls = [shapely.convex_hull(shapely.multipoints(g[["x_5070", "y_5070"]].to_numpy()))
                 for _, g in out_pts.groupby("component_id")]
        gpd.GeoDataFrame(summary.copy(), geometry=hulls, crs=5070).to_file(gpkg, layer="regions", driver="GPKG")
        gdf(out_rest, "x_5070", "y_5070").to_file(gpkg, layer="selected_not_in_component", driver="GPKG")
        if len(matches):
            gdf(matches, "match_x_5070", "match_y_5070").to_file(gpkg, layer="matches", driver="GPKG")

    info = dict(method=a.method, S=a.S, mu=mu, c=a.min_size, unit=unit_name(method, mu, a.S),
                n_models=int(arrays["Z"].shape[0]), m=int(arrays["Z"].shape[1]), n_regions=len(summary),
                n_points=len(out_pts), n_outside=len(out_rest), run=str(run),
                commit=man.get("git", {}).get("commit", "?")[:10], date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                eps=eps, summary=summary)
    (folder / "README.md").write_text(readme(info))
    zip_path = shutil.make_archive(str(folder), "zip", root_dir=folder.parent, base_dir=folder.name)

    # Read everything back.
    back = {layer: len(gpd.read_file(gpkg, layer=layer)) for layer in
            ("points", "regions", "selected_not_in_component") + (("matches",) if len(matches) else ())}
    csvs = {p.name: len(pd.read_csv(p)) for p in [folder / "points.csv", folder / "components_summary.csv",
                                                   folder / "selected_not_in_component.csv", folder / "matches.csv"]}
    assert back["points"] == csvs["points.csv"] == len(out_pts) and back["regions"] == len(summary)
    print(f"  {len(summary)} regions holding {len(out_pts)} of {int(mask.sum())} selected points; "
          f"{len(out_rest)} selected points outside; {len(matches)} matches; Table 3 check: {check}")
    print(f"  GeoPackage layers read back: {back}")
    print(f"  CSVs read back: {csvs}; {len(list((folder / 'components').glob('*.csv')))} component files")
    print(f"wrote {folder}/ (README.md, regions.gpkg, points.csv, components/, components_summary.csv, "
          f"selected_not_in_component.csv, matches.csv) and {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
