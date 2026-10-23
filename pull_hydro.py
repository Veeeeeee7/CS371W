#!/usr/bin/env python3
"""
Attach NHDPlus V2 and EPA StreamCat hydrology covariates to data/processed/dataset_{ST}.parquet.

Stages, run in this order (the three fetch stages write to data/interim/hydro/ and skip a
state whose output already exists; --merge can be rerun safely):
  --inspect     list the layers and id fields of the .gdb under data/raw/nhdplus
  --catchments  the catchment (COMID, area) containing each point    -> comid_{ST}.parquet
  --flowlines   distance to the nearest river reach and to the coast, attributes of that
                reach, drainage density of the point's catchment     -> river_{ST}.parquet
  --streamcat   StreamCat metrics by county for the catchment (cat) and its upstream
                watershed (ws): catchment context, not point values  -> streamcat_{ST}.parquet
  --merge       join the three files onto dataset_{ST}.parquet and add new features to
                FEATURES.txt

    python pull_hydro.py --catchments FL
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Set before GDAL loads. ONLY_CCW makes GDAL take counter-clockwise rings as holes instead
# of testing every ring against every other, which is slow on many-part catchments; FileGDB
# follows that ESRI ring convention.
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "ONLY_CCW")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import requests
import shapely
from requests.adapters import HTTPAdapter
from shapely import STRtree
from urllib3.util.retry import Retry

RAW = Path("data/raw/nhdplus")
PROC = Path("data/processed")
HYDRO = Path("data/interim/hydro")

STATES = ["FL"]
CRS_GEO = 4269           # NAD83 lon/lat, as NHDPlus ships
CRS_WORK = 5070          # CONUS Albers, metres: distances and densities are computed here
BBOX_PAD_DEG = 0.05      # ~5 km around the points, so edge points still find a catchment
# The nearest coastline can lie well outside the points' box (western-panhandle points
# are nearest Perdido Bay), so Coastline is read with this wider pad.
COAST_PAD_DEG = 1.0

# StreamCat metrics, each for the point's own catchment (cat) and its upstream watershed (ws).
SC_METRICS = ["precip8110", "tmean8110", "runoff", "wetindex", "bfi",
              "clay", "sand", "perm", "kffact", "hydrlcond", "elev"]
SC_URL = "https://api.epa.gov/StreamCat/streams/metrics"

# Layer and id-field names vary across NHDPlus releases; the first of each is what
# prep_nhdplus_units.py writes. Catchments call their COMID `FEATUREID`.
CATCHMENT_LAYERS = ["Catchment", "CatchmentSP", "NHDPlusCatchment"]
FLOWLINE_LAYERS = ["NHDFlowline_Network", "NHDFlowline", "NHDFlowlineVAA"]
COMID_FIELDS = ["FEATUREID", "COMID", "ComID", "comid"]

# The flowline layer has 157 fields, mostly monthly flow estimates; read only these.
FLOW_FIELDS = ["COMID", "LENGTHKM", "StreamOrde", "TotDASqKM", "SLOPE", "FTYPE"]

# Coastline is the shoreline itself. Counted as a river, every coastal point would read
# ~0 m to a "river" and the feature would encode coastal-versus-riverine flooding, so it
# is measured separately as dist_to_coast_m. Pipelines are underground and ignored.
RIVER_FTYPES = {"StreamRiver", "ArtificialPath", "Connector", "CanalDitch"}

# Attributes of the nearest river reach (not necessarily the reach the point drains to).
REACH_ATTRIBUTES = [("StreamOrde", "nearest_stream_order"),   # stream order
                    ("TotDASqKM", "upstream_area_km2"),       # total upstream drainage area
                    ("SLOPE", "channel_slope")]               # reach gradient, m/m
SLOPE_NO_DATA = -9998    # NHDPlus code for "no slope computed"

# Heavy-tailed distances and areas enter models as log10, floored so zero stays finite.
LOG_SCALED = [("dist_to_river_m", "log10_dist_river", 1.0),        # floor 1 m
              ("dist_to_coast_m", "log10_dist_coast", 1.0),        # floor 1 m
              ("upstream_area_km2", "log10_upstream_area", 0.01)]  # floor 0.01 km2
# StreamCat catchment and watershed mean elevation: a location fingerprint like the
# quarantined Earth Engine elevation, so kept only as meta_ columns. tmean8110 stays a
# feature: mean annual temperature is near-constant within a state.
STREAMCAT_ELEVATION = ["elevcat", "elevws"]

FEATURES_HEADER = (
    "# columns a model may use. everything else in dataset_*.parquet is an\n"
    "# identifier or provenance metadata and must stay out.\n"
    "# split on tile_50km (or tile_100km), never on block_id or point level.\n"
    "# StreamCat metrics are catchment areal means, constant across a\n"
    "# ~1-3 km2 catchment -- catchment context, not point measurements.\n")


# ----------------------------------------------------------------- shared helpers
def session() -> requests.Session:
    """HTTPS session that retries the resets and 5xx answers the EPA API gives under load."""
    retry = Retry(total=6, connect=6, read=6, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "POST"]), raise_on_status=False)
    http = requests.Session()
    http.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
    return http


def find_gdb() -> str:
    """The first .gdb under RAW, in sorted order."""
    hits = sorted(glob.glob(str(RAW / "**" / "*.gdb"), recursive=True))
    if not hits:
        sys.exit(f"No .gdb under {RAW}. Build it with ./fetch_nhdplus_fl.sh, then "
                 f"python prep_nhdplus_units.py")
    return hits[0]


def pick(names: list[str], candidates: list[str]) -> str | None:
    """The first candidate present in `names`, ignoring case, spelled as in `names`."""
    by_lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def pick_field(fields: list[str], candidates: list[str]) -> str | None:
    """pick() for field names."""
    return pick(fields, candidates)


def list_layers(gdb: str) -> list[str]:
    return [row[0] for row in pyogrio.list_layers(gdb)]


def layer_fields(gdb: str, layer: str) -> list[str]:
    return list(pyogrio.read_info(gdb, layer=layer)["fields"])


def find_layer(gdb: str, candidates: list[str], kind: str) -> str:
    """The first candidate layer present in the .gdb; exits if there is none."""
    layers = list_layers(gdb)
    layer = pick(layers, candidates)
    if not layer:
        sys.exit(f"no {kind} layer among {layers[:20]} -- run --inspect")
    return layer


def points_bbox(state: str) -> tuple:
    """Lon/lat box around the state's points plus BBOX_PAD_DEG; () if there is no dataset."""
    path = PROC / f"dataset_{state}.parquet"
    if not path.exists():
        return ()
    df = pd.read_parquet(path, columns=["lon", "lat"])
    return (df.lon.min() - BBOX_PAD_DEG, df.lat.min() - BBOX_PAD_DEG,
            df.lon.max() + BBOX_PAD_DEG, df.lat.max() + BBOX_PAD_DEG)


def pad_bbox(bbox: tuple, pad_deg: float) -> tuple:
    return (bbox[0] - pad_deg, bbox[1] - pad_deg, bbox[2] + pad_deg, bbox[3] + pad_deg)


def read_layer(gdb: str, layer: str, columns: list[str], bbox: tuple,
               where: str | None = None) -> gpd.GeoDataFrame:
    """Features inside `bbox` (degrees), in CRS_WORK, without null or empty geometries.

    Reading by bounding box keeps memory small: the layers cover more than the points do.
    """
    gdf = gpd.read_file(gdb, layer=layer, columns=columns, bbox=bbox, where=where)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_GEO)
    gdf = gdf.to_crs(CRS_WORK)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]


def read_points(state: str) -> gpd.GeoDataFrame:
    """point_id and geometry; dataset geometry is already in CRS_WORK."""
    return gpd.read_parquet(PROC / f"dataset_{state}.parquet", columns=["point_id", "geometry"])


def nearest(points: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of each point's nearest line, and the planar distance to it in metres."""
    index = STRtree(lines).nearest(points)
    return index, shapely.distance(points, lines[index])


# ----------------------------------------------------------------- --inspect
def inspect() -> int:
    gdb = find_gdb()
    print(f"gdb: {gdb}\n")
    layers = list_layers(gdb)
    print(f"{len(layers)} layers:")
    for name in sorted(layers):
        print(f"  {name}")

    catchment_layer = pick(layers, CATCHMENT_LAYERS)
    flowline_layer = pick(layers, FLOWLINE_LAYERS)
    print(f"\ncatchment layer -> {catchment_layer}")
    print(f"flowline  layer -> {flowline_layer}")

    for name in (catchment_layer, flowline_layer):
        if not name:
            continue
        info = pyogrio.read_info(gdb, layer=name)
        fields = list(info["fields"])
        print(f"\n--- {name} ---")
        print(f"  features : {info['features']:,}")
        print(f"  crs      : {info['crs']}")
        print(f"  geometry : {info['geometry_type']}")
        print(f"  fields   : {fields}")
        print(f"  id field -> {pick_field(fields, COMID_FIELDS)}")

    print("\nIf a layer or the id field is None, add its name to CATCHMENT_LAYERS, "
          "FLOWLINE_LAYERS or COMID_FIELDS.")
    return 0


# ----------------------------------------------------------------- --catchments
def catchments(states: list[str]) -> int:
    gdb = find_gdb()
    layer = find_layer(gdb, CATCHMENT_LAYERS, "catchment")
    fields = layer_fields(gdb, layer)
    id_field = pick_field(fields, COMID_FIELDS)
    area_field = pick_field(fields, ["AreaSqKM"])
    if not id_field:
        sys.exit(f"no COMID-like field in {layer} -- run --inspect")
    columns = [id_field]
    if area_field:
        columns.append(area_field)
    print(f"catchments from {layer}, fields {columns}")

    HYDRO.mkdir(parents=True, exist_ok=True)
    for state in states:
        out = HYDRO / f"comid_{state}.parquet"
        if out.exists():
            print(f"[{state}] already joined -- skipping")
            continue
        bbox = points_bbox(state)
        if not bbox:
            print(f"[{state}] no dataset -- skipping")
            continue
        join_catchments(state, bbox, out, gdb, layer, columns, id_field, area_field)
    return 0


def join_catchments(state, bbox, out, gdb, layer, columns, id_field, area_field) -> None:
    """Write point_id, comid and catch_area_km2 for every point of one state."""
    t0 = time.time()
    cat = read_layer(gdb, layer, columns, bbox)
    invalid = ~cat.geometry.is_valid
    if invalid.any():
        cat.loc[invalid, "geometry"] = cat.loc[invalid, "geometry"].make_valid()
    # Prefer NHDPlus's published area, so drainage density matches the published numbers.
    if area_field:
        cat["catch_area_km2"] = pd.to_numeric(cat[area_field], errors="coerce")
    else:
        cat["catch_area_km2"] = cat.geometry.area / 1e6

    pts = read_points(state)
    joined = gpd.sjoin(pts, cat[[id_field, "catch_area_km2", "geometry"]],
                       how="left", predicate="within")
    # A point inside overlapping catchments matches more than once: keep the first.
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    res = pd.DataFrame({"point_id": pts.point_id.values,
                        "comid": pd.to_numeric(joined[id_field], errors="coerce").values,
                        "catch_area_km2": joined["catch_area_km2"].values})
    res.to_parquet(out, index=False)
    hit = 100 * res.comid.notna().mean()
    print(f"[{state}] {len(cat):,} catchments read, {len(res):,} points, "
          f"{hit:.1f}% matched a COMID, {res.comid.nunique():,} distinct "
          f"({time.time() - t0:.0f}s)", flush=True)
    if hit < 95:
        print(f"     NOTE only {hit:.1f}% matched. NHDPlus has no catchment over open "
              f"water or tidal flats, so coastal points miss some; below ~85% is worth "
              f"a look.")


# ----------------------------------------------------------------- --flowlines
def flowlines(states: list[str]) -> int:
    gdb = find_gdb()
    layer = find_layer(gdb, FLOWLINE_LAYERS, "flowline")
    fields = layer_fields(gdb, layer)
    columns = []
    for name in FLOW_FIELDS:
        found = pick_field(fields, [name])
        if found:
            columns.append(found)
    print(f"flowlines from {layer}, fields {columns}")
    print("(a 'Measured (M) geometry' warning is harmless: M values are river-mile "
          "measures and distances here are planar)")

    HYDRO.mkdir(parents=True, exist_ok=True)
    for state in states:
        out = HYDRO / f"river_{state}.parquet"
        if out.exists():
            print(f"[{state}] already computed -- skipping")
            continue
        bbox = points_bbox(state)
        if not bbox:
            print(f"[{state}] no dataset -- skipping")
            continue
        measure_flowlines(state, bbox, out, gdb, layer, columns)
    return 0


def read_coastline(gdb: str, layer: str, bbox: tuple) -> gpd.GeoDataFrame:
    """Coastline features within COAST_PAD_DEG of the points' box."""
    return read_layer(gdb, layer, ["FTYPE"], pad_bbox(bbox, COAST_PAD_DEG),
                      where="FTYPE IN ('Coastline')")


def measure_flowlines(state, bbox, out, gdb, layer, columns) -> None:
    """Write river and coast distances, nearest-reach attributes and drainage density."""
    t0 = time.time()
    fl = read_layer(gdb, layer, columns, bbox)
    if fl.empty:
        print(f"[{state}] no flowlines in bbox -- skipping")
        return
    # The layer is MultiLineString Z; drop Z so every distance is planar.
    fl["geometry"] = shapely.force_2d(fl.geometry.values)

    if "FTYPE" in fl.columns:
        ftype = fl["FTYPE"].astype(str)
        coast = read_coastline(gdb, layer, bbox)
    else:
        # Without FTYPE every line counts as a river and there is no coastline.
        ftype = pd.Series("StreamRiver", index=fl.index)
        coast = fl.iloc[:0]
    rivers = fl[ftype.isin(RIVER_FTYPES)]
    print(f"[{state}] {len(fl):,} flowlines: {ftype.value_counts().to_dict()}")
    print(f"       {len(rivers):,} river, {len(coast):,} coastline "
          f"(coastline measured separately, never as 'river')")
    if rivers.empty:
        print(f"[{state}] no river-type flowlines -- skipping")
        return

    pts = read_points(state)
    points = shapely.points(pts.geometry.x.values, pts.geometry.y.values)
    near_river, river_dist = nearest(points, np.asarray(rivers.geometry.values))
    res = pd.DataFrame({"point_id": pts.point_id.values, "dist_to_river_m": river_dist})
    for source, column in REACH_ATTRIBUTES:
        if source in rivers.columns:
            values = pd.to_numeric(rivers[source], errors="coerce").to_numpy()
            if source == "SLOPE":
                values = np.where(values == SLOPE_NO_DATA, np.nan, values)
            res[column] = values[near_river]
    if "FTYPE" in rivers.columns:
        res["meta_nearest_ftype"] = rivers["FTYPE"].to_numpy()[near_river]

    if coast.empty:
        res["dist_to_coast_m"] = np.nan
    else:
        coast_lines = shapely.force_2d(np.asarray(coast.geometry.values))
        _, coast_dist = nearest(points, coast_lines)
        res["dist_to_coast_m"] = coast_dist

    comid_file = HYDRO / f"comid_{state}.parquet"
    if comid_file.exists() and "LENGTHKM" in fl.columns and "COMID" in fl.columns:
        res = add_drainage_density(res, fl, comid_file)

    res.to_parquet(out, index=False)
    dist = res.dist_to_river_m
    print(f"       river distance median {np.median(dist):,.0f} m, "
          f"p95 {np.percentile(dist, 95):,.0f} m; coast distance available for "
          f"{100 * res.dist_to_coast_m.notna().mean():.0f}% "
          f"({time.time() - t0:.0f}s)", flush=True)


def add_drainage_density(res: pd.DataFrame, fl: gpd.GeoDataFrame,
                         comid_file: Path) -> pd.DataFrame:
    """Add drainage_density (km of flowline per km2) of each point's catchment.

    Each NHDPlus flowline pairs 1:1 with the catchment of the same COMID, so reach length
    over catchment area is the local density, from published values only. A sink catchment
    (negative FEATUREID) has no flowline of its own, so its density is 0, not missing.
    """
    comid = pd.to_numeric(fl["COMID"], errors="coerce")
    length_km = pd.to_numeric(fl["LENGTHKM"], errors="coerce")
    length_per_comid = length_km.groupby(comid).sum()

    res = res.merge(pd.read_parquet(comid_file), on="point_id", how="left")
    length = res.comid.map(length_per_comid)
    is_sink = length.isna() & (res.comid < 0)
    length = length.mask(is_sink, 0.0)
    res["drainage_density"] = length / res.catch_area_km2.replace(0, np.nan)
    return res.drop(columns=["comid", "catch_area_km2"])


# ----------------------------------------------------------------- --streamcat
def fetch_county(http: requests.Session, county: str) -> list[dict]:
    """StreamCat cat and ws metrics for every COMID in one county."""
    # By county because the URL has an 8,192-character limit, too short for COMID lists;
    # one county already returns thousands of COMIDs.
    params = {"name": ",".join(SC_METRICS), "areaOfInterest": "cat,ws", "county": county}
    response = http.get(SC_URL, params=params, timeout=180)
    response.raise_for_status()
    return response.json().get("items", [])


def streamcat(states: list[str]) -> int:
    HYDRO.mkdir(parents=True, exist_ok=True)
    http = session()
    for state in states:
        out = HYDRO / f"streamcat_{state}.parquet"
        if out.exists():
            print(f"[{state}] already pulled -- skipping")
            continue
        path = PROC / f"dataset_{state}.parquet"
        if not path.exists():
            print(f"[{state}] no dataset -- skipping")
            continue
        county_fips = pd.read_parquet(path, columns=["county_fips"]).county_fips
        counties = sorted(county_fips.dropna().astype(str).unique())
        if not counties:
            print(f"[{state}] no county FIPS -- skipping")
            continue
        pull_streamcat(http, state, counties, out)
    return 0


def pull_streamcat(http: requests.Session, state: str, counties: list[str], out: Path) -> None:
    print(f"[{state}] {len(counties)} counties", flush=True)
    frames, failed = [], []
    n_rows = 0
    t0 = time.time()
    for i, county in enumerate(counties, 1):
        try:
            items = fetch_county(http, county)
            if items:
                frames.append(pd.DataFrame(items))
                n_rows += len(items)
        except Exception as error:   # keep going; failed counties are listed at the end
            failed.append((county, str(error)[:90]))
        if i % 20 == 0 or i == len(counties):
            print(f"    {i}/{len(counties)} counties, {n_rows:,} COMIDs "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if not frames:
        print(f"[{state}] StreamCat returned nothing")
        return
    df = pd.concat(frames, ignore_index=True).drop_duplicates("comid")
    df.columns = [c.lower() for c in df.columns]
    df.to_parquet(out, index=False)
    print(f"[{state}] {len(df):,} COMIDs, {df.shape[1] - 1} metrics -> {out}")
    if failed:
        print(f"     {len(failed)} counties failed, e.g. {failed[:2]}")
        print(f"     rerun after deleting {out} to retry them")


# ----------------------------------------------------------------- merge guard
# This script and pull_atlas14.py both end by read-modify-writing dataset_{ST}.parquet,
# so two overlapping merges would silently drop the first one's columns. The fetch stages
# are safe side by side; merges take an exclusive lockfile and write to a temp file that
# is renamed into place (atomic on POSIX), so readers never see a half-written file.
LOCK = PROC / ".merge.lock"


class MergeLock:
    """Exclusive, cross-process, and released even if the merge raises."""

    def __enter__(self):
        PROC.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = (time.time() - LOCK.stat().st_mtime) / 60 if LOCK.exists() else 0.0
            sys.exit(
                f"\nAnother merge holds {LOCK} (age {age:.1f} min).\n"
                f"Merges rewrite the same dataset files, so they must not overlap.\n"
                f"Wait for it to finish. If you are certain nothing is running:\n"
                f"    rm {LOCK}\n")
        os.write(self.fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        LOCK.unlink(missing_ok=True)
        return False


def atomic_to_parquet(df, path) -> None:
    """Write to a temp file beside the target, then rename into place."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------- --merge
def merge(states: list[str]) -> int:
    with MergeLock():
        return _merge_body(states)


def _added_columns(state: str) -> set[str]:
    """Every column the merge adds for one state, so a re-run can drop them first.

    Read from the hydro files' schemas rather than hardcoded, so a new column is never missed.
    """
    columns = set()
    for name, key in ((f"comid_{state}", "point_id"),
                      (f"river_{state}", "point_id"),
                      (f"streamcat_{state}", "comid")):
        path = HYDRO / f"{name}.parquet"
        if path.exists():
            columns |= set(pq.read_schema(path).names) - {key}
    # plus the columns the merge derives or renames into existence
    columns.add("meta_comid")
    for raw, log_column, _ in LOG_SCALED:
        columns.add(log_column)
        columns.add(f"meta_{raw}")
    for name in STREAMCAT_ELEVATION:
        columns.add(f"meta_{name}")
    return columns


def percent_filled(frame: pd.DataFrame, column: str) -> float:
    """Non-null share of a column in percent (one decimal); NaN if the column is absent."""
    if column not in frame.columns:
        return np.nan
    return round(100 * frame[column].notna().mean(), 1)


def join_hydro_files(dataset: gpd.GeoDataFrame, state: str) -> gpd.GeoDataFrame:
    """Left-join the comid, river and StreamCat files that exist; comid becomes meta_comid."""
    comid_file = HYDRO / f"comid_{state}.parquet"
    if comid_file.exists():
        dataset = dataset.merge(pd.read_parquet(comid_file), on="point_id", how="left")
        dataset = dataset.rename(columns={"comid": "meta_comid"})

    river_file = HYDRO / f"river_{state}.parquet"
    if river_file.exists():
        dataset = dataset.merge(pd.read_parquet(river_file), on="point_id", how="left")

    streamcat_file = HYDRO / f"streamcat_{state}.parquet"
    if streamcat_file.exists() and "meta_comid" in dataset.columns:
        sc = pd.read_parquet(streamcat_file).rename(columns={"comid": "meta_comid"})
        sc = sc.drop_duplicates("meta_comid")
        dataset = dataset.merge(sc, on="meta_comid", how="left")
    return dataset


def quarantine_and_log_scale(dataset: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rename StreamCat elevation to meta_; add log10 columns and keep the raw ones as meta_."""
    for name in STREAMCAT_ELEVATION:
        if name in dataset.columns:
            dataset = dataset.rename(columns={name: f"meta_{name}"})
    for raw, log_column, floor in LOG_SCALED:
        if raw in dataset.columns:
            values = pd.to_numeric(dataset[raw], errors="coerce")
            dataset[log_column] = np.log10(values.clip(lower=floor))
            dataset = dataset.rename(columns={raw: f"meta_{raw}"})
    return dataset


def merge_state(state: str) -> tuple[list[str], dict] | None:
    """Merge one state's hydro files into its dataset; return its features and a summary row.

    Returns None when the state has no dataset.
    """
    path = PROC / f"dataset_{state}.parquet"
    if not path.exists():
        print(f"[{state}] no dataset -- skipping")
        return None
    dataset = gpd.read_parquet(path)
    n_rows = len(dataset)

    # Drop what an earlier run of this merge added, so a re-run replaces rather than
    # duplicates. Columns from other scripts are untouched.
    stale = [c for c in _added_columns(state) if c in dataset.columns]
    if stale:
        dataset = dataset.drop(columns=stale)
    columns_before = set(dataset.columns)

    dataset = join_hydro_files(dataset, state)
    dataset = quarantine_and_log_scale(dataset)

    added = sorted(set(dataset.columns) - columns_before)
    features = [c for c in added if not c.startswith("meta_") and not c.endswith("_pctfull")]
    atomic_to_parquet(dataset, path)

    coverage = {c: percent_filled(dataset, c) for c in features}
    re_merge = "  [re-merge]" if stale else ""
    print(f"\n[{state}] {n_rows:,} rows, {len(added)} columns added "
          f"({len(features)} features, {len(added) - len(features)} meta){re_merge}")
    print(f"  coverage %: {coverage}")
    summary = {"state": state, "features": len(features),
               "comid_matched": percent_filled(dataset, "meta_comid"),
               "coast": percent_filled(dataset, "log10_dist_coast")}
    return features, summary


def update_features_txt(features: list[str]) -> None:
    """Append the features FEATURES.txt does not list yet, keeping the listed order.

    Comment lines are replaced by FEATURES_HEADER.
    """
    path = PROC / "FEATURES.txt"
    listed = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            listed.append(line.strip())
    new = [f for f in features if f not in listed]
    all_features = listed + new
    atomic_write_text(FEATURES_HEADER + "\n".join(all_features) + "\n", path)
    print(f"\nFEATURES.txt: {len(listed)} -> {len(all_features)} features")
    print(f"  added: {new if new else 'nothing new (already present)'}")


def _merge_body(states: list[str]) -> int:
    last_features: list[str] = []
    summaries = []
    for state in states:
        result = merge_state(state)
        if result is None:
            continue
        last_features, summary = result
        summaries.append(summary)

    if last_features:
        update_features_txt(last_features)
    if summaries:
        print("\n" + pd.DataFrame(summaries).to_string(index=False))
        print("\ncoast = % of points with a coastline distance; well above 0 shows the")
        print("FTYPE split doing its job.")
    return 0


# ----------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="NHDPlus V2 and StreamCat covariates.")
    ap.add_argument("states", nargs="*", help=f"default: {' '.join(STATES)}")
    ap.add_argument("--inspect", action="store_true", help="list geodatabase layers and fields")
    ap.add_argument("--catchments", action="store_true", help="points -> catchment COMID")
    ap.add_argument("--flowlines", action="store_true",
                    help="river and coast distance, reach attributes, drainage density")
    ap.add_argument("--streamcat", action="store_true", help="EPA StreamCat metrics by county")
    ap.add_argument("--merge", action="store_true", help="join everything into dataset_{ST}")
    args = ap.parse_args(argv)

    states = args.states or STATES
    unknown = [s for s in states if s not in STATES]
    if unknown:
        sys.exit(f"unknown states: {unknown}")

    if args.inspect:
        return inspect()
    if args.catchments:
        return catchments(states)
    if args.flowlines:
        return flowlines(states)
    if args.streamcat:
        return streamcat(states)
    if args.merge:
        return merge(states)

    ap.print_help()
    print("\nRun in order: --inspect, --catchments, --flowlines, --streamcat, --merge")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
