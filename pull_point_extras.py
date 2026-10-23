#!/usr/bin/env python3
"""Per-point provenance sidecar for the boundary analysis: FIRM panel, nearest SFHA polygon, edge.

Reads data/processed/dataset_<ST>.parquet, data/interim/prep/idx_<ST>.parquet and
haz_<ST>_5070.parquet, and data/interim/nfhl_states/pan_<ST>.parquet. Writes
data/interim/analysis/point_extras_<ST>.parquet, one row per dataset row in dataset order:
    point_id; meta_firm_pan, meta_firm_pan_n, meta_firm_pan_eff_date: the FIRM panel;
    meta_near_src_row_id, meta_near_fld_ar_id, meta_near_fld_zone, meta_near_dist_m: the nearest
        SFHA polygon within --radius;
    meta_edge_src_row_id, meta_edge_dist_m: for positives, the labelling polygon and the distance
        to its edge.
The columns live in a sidecar, not in dataset_<ST>.parquet, so the models never see them: they are
the held-out maps the regions are tested against. Settings and inputs go in the schema metadata.

    python pull_point_extras.py --state FL
    python pull_point_extras.py --state FL --out /tmp/x/point_extras_FL.parquet --limit 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely import STRtree

PROCESSED = Path("data/processed")
PREP_DIR = Path("data/interim/prep")
NFHL_DIR = Path("data/interim/nfhl_states")
OUT_DIR = Path("data/interim/analysis")

CRS_WORK = 5070        # CONUS Albers, metres: the sampler's working CRS
CRS_LL = 4269          # NAD83 lon/lat, as sample_points writes lon/lat
RADIUS_M = 5000.0      # band_max_m: no protocol v2 point is farther than this from an SFHA
LIMIT_SEED = 0         # RNG seed of the --limit subset
MIN_CELL_M = 1000.0    # smallest cell candidate_rows bins the points into

POINT_COLS = ["point_id", "domain", "label", "dfirm_id", "meta_fld_ar_id", "lon", "lat"]
OPTIONAL_COLS = ["meta_src_row_id", "meta_dist_to_sfha_m", "meta_eff_date", "geometry"]
IDX_COLS = ["row_id", "DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "SFHA_TF",
            "minx", "miny", "maxx", "maxy"]
BBOX_COLS = ["minx", "miny", "maxx", "maxy"]


# =========================================================================
# inputs
# =========================================================================
def default_paths(state: str) -> dict:
    return dict(dataset=PROCESSED / f"dataset_{state}.parquet",
                idx=PREP_DIR / f"idx_{state}.parquet",
                haz=PREP_DIR / f"haz_{state}_5070.parquet",
                panels=NFHL_DIR / f"pan_{state}.parquet",
                out=OUT_DIR / f"point_extras_{state}.parquet")


def project_to_work(x: np.ndarray, y: np.ndarray, epsg: int) -> np.ndarray:
    """(n, 2) coordinates reprojected from EPSG:`epsg` to EPSG:5070."""
    transformer = Transformer.from_crs(epsg, CRS_WORK, always_xy=True)
    return np.column_stack(transformer.transform(x, y))


def geo_crs(schema: pa.Schema) -> int | None:
    """EPSG code of a GeoParquet's primary geometry, or None if unstated."""
    metadata = schema.metadata or {}
    geo = json.loads(metadata.get(b"geo") or b"{}")
    primary = geo.get("primary_column", "geometry")
    column = geo.get("columns", {}).get(primary, {})
    crs = column.get("crs")
    if not isinstance(crs, dict):
        return None
    crs_id = crs.get("id", {})
    if crs_id.get("authority") != "EPSG":
        return None
    return int(crs_id["code"])


def load_points(path: Path, limit: int | None = None,
                cols: list[str] = POINT_COLS) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Point rows, their (x, y) in EPSG:5070, and a note on where the xy came from.

    The dataset's own point geometry is preferred: it holds the exact coordinates the sampler
    tested containment and distances with, which a lon/lat round trip reproduces only to ~1e-9 m.
    `limit` keeps a random subset of rows, in dataset order.
    """
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    missing = [c for c in cols if c not in names]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    optional = [c for c in OPTIONAL_COLS if c in names and c not in cols]
    df = pq.read_table(path, columns=cols + optional).to_pandas()
    if not df.point_id.is_unique:
        sys.exit(f"{path}: point_id is not unique")
    if limit is not None and limit < len(df):
        rng = np.random.default_rng(LIMIT_SEED)
        keep = np.sort(rng.choice(len(df), limit, replace=False))
        df = df.iloc[keep].reset_index(drop=True)

    epsg = geo_crs(pf.schema_arrow) if "geometry" in df else None
    if epsg is not None:
        points = shapely.from_wkb(df.pop("geometry").to_numpy())
        xy = shapely.get_coordinates(points)
        how = f"dataset geometry (EPSG:{epsg})"
        if epsg != CRS_WORK:
            xy = project_to_work(xy[:, 0], xy[:, 1], epsg)
            how += f" -> EPSG:{CRS_WORK}"
    else:
        # no geometry, or one without a stated CRS: use lon/lat
        df = df.drop(columns="geometry", errors="ignore")
        xy = project_to_work(df.lon.to_numpy(float), df.lat.to_numpy(float), CRS_LL)
        how = f"lon/lat projected EPSG:{CRS_LL} -> {CRS_WORK}"
    if len(xy) != len(df):
        sys.exit(f"{path}: {len(xy)} coordinates for {len(df)} rows (empty or multi geometry)")
    return df, xy, how


def load_index(path: Path) -> pd.DataFrame:
    """idx_<ST> polygon attributes and bboxes, with the string columns as object dtype."""
    idx = pq.read_table(path, columns=IDX_COLS).to_pandas()
    for col in ("DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "SFHA_TF"):
        idx[col] = idx[col].astype(object)
    return idx


def match_row_ids(rid: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The wanted ids present in the unordered id array `rid`, and their positions in it."""
    order = np.argsort(rid)
    sorted_ids = rid[order]
    hit = np.searchsorted(sorted_ids, wanted)
    clipped = np.minimum(hit, len(rid) - 1)
    found = (hit < len(rid)) & (sorted_ids[clipped] == wanted)
    return wanted[found], order[hit[found]]


class HazReader:
    """Geometry from haz_<ST>_5070.parquet by row_id, one row group at a time.

    The Florida file is 5.4 GB in 10,000-row groups and must never be read whole. prep_state
    writes row_id 0..n-1 in file order, so row_id r sits at position r - start[g] of group g.
    Every read checks that, and matches the ids explicitly in a group where it does not hold.
    """

    def __init__(self, path: Path):
        self.pf = pq.ParquetFile(path)
        meta = self.pf.metadata
        sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        self.start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n_rows = int(self.start[-1])

    def group_of(self, row_ids: np.ndarray) -> np.ndarray:
        """The row group each row_id sits in when ids follow file order."""
        return np.searchsorted(self.start, row_ids, side="right") - 1

    def iter_groups(self, row_ids):
        """Yield (group, sorted row_ids found, geometries) for each group holding wanted rows."""
        want = np.unique(np.asarray(row_ids, dtype=np.int64))
        want = want[(want >= 0) & (want < self.n_rows)]
        groups = self.group_of(want)
        for gi in np.unique(groups):
            sel = want[groups == gi]
            table = self.pf.read_row_group(int(gi), columns=["row_id", "geometry"])
            rid = table.column("row_id").to_numpy()
            pos = sel - self.start[gi]
            if not np.array_equal(rid[pos], sel):
                sel, pos = match_row_ids(rid, sel)
            wkb = table.column("geometry").take(pa.array(pos)).to_numpy(zero_copy_only=False)
            yield int(gi), sel, shapely.from_wkb(wkb)

    def read(self, row_ids) -> dict[int, object]:
        """{row_id: geometry} for the wanted rows the file holds."""
        out: dict[int, object] = {}
        for _, sel, geoms in self.iter_groups(row_ids):
            out.update(zip(sel.tolist(), geoms))
        return out


# =========================================================================
# the labelling polygon of each positive
# =========================================================================
def key_candidates(df: pd.DataFrame, has_key: np.ndarray, idx: pd.DataFrame) -> pd.DataFrame:
    """idx rows matching each keyed point's (dfirm_id, meta_fld_ar_id); i is the point's row."""
    keys = pd.DataFrame(dict(i=np.flatnonzero(has_key),
                             DFIRM_ID=df.dfirm_id.to_numpy()[has_key].astype(str),
                             FLD_AR_ID=df.meta_fld_ar_id.to_numpy()[has_key].astype(str)))
    idx_keys = idx[["row_id", "DFIRM_ID", "FLD_AR_ID"]].astype({"DFIRM_ID": str, "FLD_AR_ID": str})
    return keys.merge(idx_keys, on=["DFIRM_ID", "FLD_AR_ID"], how="inner")


def resolve_label_polygons(df: pd.DataFrame, xy: np.ndarray, idx: pd.DataFrame,
                           haz: HazReader) -> tuple[np.ndarray, dict]:
    """idx row_id of the polygon that labelled each positive (-1 for the rest), and counts of
    how each was found.

    meta_src_row_id names the polygon wherever it is set. Points without it (sampling v1) are
    matched on (dfirm_id, meta_fld_ar_id): the pair, because FLD_AR_ID alone repeats across
    DFIRMs. Where the pair names several idx rows, the single one that covers the point is taken,
    as aotil/regions.py does.
    """
    src = np.full(len(df), -1, dtype=np.int64)
    pos = df.label.to_numpy() == 1
    info = dict(positives=int(pos.sum()), from_column=0, key_unique=0,
                key_ambiguous=0, key_ambiguous_resolved=0, key_not_in_index=0,
                no_key=0, column_key_mismatch=0)

    if "meta_src_row_id" in df:
        column = pd.to_numeric(df.meta_src_row_id, errors="coerce").to_numpy(dtype=float)
        have = pos & np.isfinite(column)
        src[have] = column[have].astype(np.int64)
        info["from_column"] = int(have.sum())
        if have.any():
            # the named polygon should carry the point's own FLD_AR_ID
            polygon_ar = idx.set_index("row_id").FLD_AR_ID.reindex(src[have]).to_numpy()
            point_ar = df.meta_fld_ar_id.to_numpy()[have]
            mismatch = polygon_ar.astype(str) != point_ar.astype(str)
            info["column_key_mismatch"] = int(mismatch.sum())

    todo = pos & (src < 0)
    has_key = todo & df.meta_fld_ar_id.notna().to_numpy()
    info["no_key"] = int((todo & ~has_key).sum())
    if not has_key.any():
        return src, info

    cand = key_candidates(df, has_key, idx)
    n_cand = cand.groupby("i").row_id.transform("size")
    unique = cand[n_cand == 1]
    src[unique.i.to_numpy()] = unique.row_id.to_numpy()
    info["key_unique"] = len(unique)
    info["key_not_in_index"] = int(has_key.sum()) - int(cand.i.nunique())

    ambiguous = cand[n_cand > 1]
    info["key_ambiguous"] = int(ambiguous.i.nunique())
    if len(ambiguous):
        geoms = haz.read(ambiguous.row_id.unique())
        for i, group in ambiguous.groupby("i"):
            point = shapely.Point(xy[i])
            covering = []
            for r in group.row_id:
                if geoms.get(r) is not None and geoms[r].covers(point):
                    covering.append(r)
            if len(covering) == 1:
                src[i] = covering[0]
                info["key_ambiguous_resolved"] += 1
    return src, info


# =========================================================================
# nearest SFHA polygon and edge distance, one pass over the hazard file
# =========================================================================
def candidate_rows(sfha: pd.DataFrame, xy: np.ndarray, radius: float) -> np.ndarray:
    """SFHA row_ids whose bbox comes within `radius` of an occupied point cell.

    A polygon within radius of a point has its bbox within radius too, so this prefilter loses
    no distance <= radius. Binning the points into cells keeps it to a few hundred thousand box
    tests.
    """
    cell = max(radius, MIN_CELL_M)
    cells = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
    cell_boxes = shapely.box(cells[:, 0] * cell - radius, cells[:, 1] * cell - radius,
                             (cells[:, 0] + 1) * cell + radius, (cells[:, 1] + 1) * cell + radius)
    polygon_boxes = shapely.box(sfha.minx.to_numpy(), sfha.miny.to_numpy(),
                                sfha.maxx.to_numpy(), sfha.maxy.to_numpy())
    hits = STRtree(cell_boxes).query(polygon_boxes, predicate="intersects")
    return sfha.row_id.to_numpy()[np.unique(hits[0])]


def nearest_in_group(pts: np.ndarray, xy: np.ndarray, rows: np.ndarray, geoms: np.ndarray,
                     bboxes: np.ndarray, radius: float):
    """(point index, distance, row_id) of each point's nearest polygon among one row group's,
    within radius; the lower row_id wins on equal distances."""
    x0 = bboxes[:, 0].min() - radius
    y0 = bboxes[:, 1].min() - radius
    x1 = bboxes[:, 2].max() + radius
    y1 = bboxes[:, 3].max() + radius
    in_x = (xy[:, 0] >= x0) & (xy[:, 0] <= x1)
    in_y = (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    near = np.flatnonzero(in_x & in_y)
    if near.size == 0:
        return near, np.zeros(0), np.zeros(0, dtype=np.int64)

    (pi, gj), d = STRtree(geoms).query_nearest(
        pts[near], max_distance=radius, return_distance=True, all_matches=True)
    pi = near[pi]
    r = rows[gj]
    if pi.size == 0:
        # points inside the grown bbox can still all be farther than radius from every polygon
        return pi, d, r
    # order by point, distance, row_id and keep each point's first match
    order = np.lexsort((r, d, pi))
    pi, d, r = pi[order], d[order], r[order]
    first = np.r_[True, pi[1:] != pi[:-1]]
    return pi[first], d[first], r[first]


def label_polygon_edges(pts: np.ndarray, src: np.ndarray, rows: np.ndarray, geoms: np.ndarray,
                        usable: np.ndarray):
    """For the points whose labelling polygon is in this row group: their indices, whether the
    polygon covers each, and the distance to its boundary."""
    labelled = np.flatnonzero(np.isin(src, rows))
    if labelled.size == 0:
        return labelled, np.zeros(0, dtype=bool), np.zeros(0)
    at = np.searchsorted(rows, src[labelled])
    ok = usable[at]
    labelled, at = labelled[ok], at[ok]
    inside = shapely.covers(geoms[at], pts[labelled])
    uniq, inv = np.unique(at, return_inverse=True)
    boundaries = shapely.boundary(geoms[uniq])
    dist = shapely.distance(pts[labelled], boundaries[inv])
    return labelled, inside, dist


def nearest_and_edges(haz: HazReader, sfha: pd.DataFrame, xy: np.ndarray,
                      src: np.ndarray, radius: float) -> dict:
    """One pass over the hazard file: every point's nearest SFHA polygon within radius, and for
    every positive whether its labelling polygon covers it and how far its edge is.

    Same candidates (every SFHA_TF == 'T' zone), geometry and exact EPSG:5070 distance as
    sample_points.distances_to_sfha. The edge is the whole polygon boundary, holes and edges
    shared with neighbouring SFHA polygons included; it is NaN where the polygon does not cover
    the point.
    """
    n = len(xy)
    pts = shapely.points(xy)
    best_d = np.full(n, np.inf)
    best_r = np.full(n, -1, dtype=np.int64)
    covered = np.zeros(n, dtype=bool)
    edge = np.full(n, np.nan)

    cand = candidate_rows(sfha, xy, radius)
    want = np.union1d(cand, src[src >= 0])
    is_cand = np.zeros(haz.n_rows, dtype=bool)
    is_cand[cand] = True
    bboxes = sfha.set_index("row_id")[BBOX_COLS]
    n_groups = len(np.unique(haz.group_of(want)))
    print(f"  {len(cand):,} of {len(sfha):,} SFHA polygons within {radius:.0f} m of a "
          f"point's cell; reading {n_groups} of {len(haz.start) - 1} row groups", flush=True)

    t0 = time.time()
    for k, (gi, rows, geoms) in enumerate(haz.iter_groups(want), 1):
        usable = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
        keep = is_cand[rows] & usable
        if keep.any():
            group_bboxes = bboxes.loc[rows[keep]].to_numpy()
            pi, d, r = nearest_in_group(pts, xy, rows[keep], geoms[keep], group_bboxes, radius)
            better = (d < best_d[pi]) | ((d == best_d[pi]) & (r < best_r[pi]))
            best_d[pi[better]] = d[better]
            best_r[pi[better]] = r[better]

        labelled, inside, dist = label_polygon_edges(pts, src, rows, geoms, usable)
        covered[labelled] = inside
        edge[labelled] = dist
        if k % 10 == 0 or k == n_groups:
            print(f"    row groups {k}/{n_groups}  ({time.time() - t0:.0f}s)", flush=True)

    edge[~covered] = np.nan
    return dict(near_d=best_d, near_r=best_r, covered=covered, edge=edge)


# =========================================================================
# FIRM panel
# =========================================================================
def load_panels(path: Path):
    """FIRM panels in EPSG:5070, and the mask of those reprojection left invalid.

    Those are repaired with make_valid; in Florida it is one panel, 12081C0333E, whose 3.1 km
    spur under 1 m wide self-intersects after reprojection.
    """
    pan = gpd.read_parquet(path)
    if pan.crs is None:
        sys.exit(f"{path}: no CRS")
    pan = pan.to_crs(CRS_WORK).reset_index(drop=True)
    geoms = np.asarray(pan.geometry.values)
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms[bad] = shapely.make_valid(geoms[bad])
        pan = pan.set_geometry(geoms)
    return pan, bad


def panel_of(pan, xy: np.ndarray, dfirm: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Each point's panel: its row in pan (-1 for none), the number of panels containing the
    point, and how many points in several panels took one of their own DFIRM's.

    Panels are whole printed rectangles and overlap across county DFIRMs. A point takes the
    lowest FIRM_PAN among its own DFIRM's panels, else the lowest of the others: the rule
    sample_points.attach_panel_meta applies, so meta_firm_pan and meta_eff_date name one sheet.
    """
    n = len(xy)
    point_i, panel_row = STRtree(np.asarray(pan.geometry.values)).query(
        shapely.points(xy), predicate="within")
    count = np.bincount(point_i, minlength=n)
    hits = pd.DataFrame(dict(i=point_i, row=panel_row,
                             pan=pan.FIRM_PAN.to_numpy()[panel_row],
                             own=pan.DFIRM_ID.to_numpy()[panel_row] == dfirm[point_i]))
    hits = hits.sort_values(["i", "own", "pan"], ascending=[True, False, True], kind="stable")
    chosen = hits.drop_duplicates("i")
    out = np.full(n, -1, dtype=np.int64)
    out[chosen.i.to_numpy()] = chosen.row.to_numpy()
    in_several = count > 1
    by_dfirm = int(chosen.own.to_numpy()[in_several[chosen.i.to_numpy()]].sum())
    return out, count, by_dfirm


def eff_dates(pan) -> np.ndarray:
    """Panel EFF_DATE as naive UTC datetime64, the form sample_points writes to meta_eff_date."""
    if "EFF_DATE" not in pan:
        return np.full(len(pan), np.datetime64("NaT", "ms"))
    return pd.to_datetime(pan.EFF_DATE, utc=True).dt.tz_localize(None).to_numpy()


# =========================================================================
# output table and summary
# =========================================================================
def nullable_ids(row_ids: np.ndarray):
    """int64 row ids with -1 for none, as a pandas Int64 array with <NA>."""
    out = pd.array(np.where(row_ids >= 0, row_ids, 0), dtype="Int64")
    out[row_ids < 0] = pd.NA
    return out


def lookup_strings(column: pd.Series, row_ids: np.ndarray):
    """column[row_id] as a pandas string array, <NA> where row_id is -1."""
    found = row_ids >= 0
    values = column.reindex(np.where(found, row_ids, -1)).to_numpy()
    return pd.array(np.where(found, values, None), dtype="string")


def sidecar_frame(df: pd.DataFrame, idx: pd.DataFrame, near_r: np.ndarray, near_d: np.ndarray,
                  src: np.ndarray, edge: np.ndarray, pan, pan_row: np.ndarray,
                  pan_n: np.ndarray) -> pd.DataFrame:
    """The sidecar table: one row per point, in dataset order."""
    pos = df.label.to_numpy() == 1
    in_pan = pan_row >= 0
    attrs = idx.set_index("row_id")
    return pd.DataFrame({
        "point_id": df.point_id.to_numpy(),
        "meta_firm_pan": pd.array(np.where(in_pan, pan.FIRM_PAN.to_numpy()[pan_row], None),
                                  dtype="string"),
        "meta_firm_pan_n": pan_n.astype(np.int16),
        "meta_firm_pan_eff_date": np.where(in_pan, eff_dates(pan)[pan_row],
                                           np.datetime64("NaT", "ms")),
        "meta_near_src_row_id": nullable_ids(near_r),
        "meta_near_fld_ar_id": lookup_strings(attrs.FLD_AR_ID, near_r),
        "meta_near_fld_zone": lookup_strings(attrs.FLD_ZONE, near_r),
        "meta_near_dist_m": np.where(near_r >= 0, near_d, np.nan),
        "meta_edge_src_row_id": nullable_ids(src),
        "meta_edge_dist_m": np.where(pos, edge, np.nan),
    })


def print_panel_summary(df: pd.DataFrame, res: pd.DataFrame) -> None:
    count = res.meta_firm_pan_n.to_numpy()
    most = int(count.max()) if len(df) else 0
    print(f"\n  panels: {int((count == 1).sum()):,} points in exactly one, "
          f"{int((count == 0).sum()):,} in none, {int((count > 1).sum()):,} in more than one "
          f"(max {most})")
    if "meta_eff_date" not in df:
        return
    mine = pd.to_datetime(res.meta_firm_pan_eff_date)
    theirs = pd.to_datetime(df.meta_eff_date)
    both = mine.notna() & theirs.notna()
    same = int((mine[both] == theirs[both]).sum())
    print(f"  meta_firm_pan_eff_date equals the dataset's meta_eff_date for "
          f"{same:,}/{int(both.sum()):,} points (sample_points picks the panel by the same rule; "
          f"a difference means another pan file or the v1 sampler, which took the first panel)")


def print_label_summary(df: pd.DataFrame, res: pd.DataFrame, info: dict, near: dict) -> None:
    print(f"  labelling polygon of {info['positives']:,} positives: {info['from_column']:,} from "
          f"meta_src_row_id, {info['key_unique']:,} by unique (dfirm_id, meta_fld_ar_id), "
          f"{info['key_ambiguous_resolved']}/{info['key_ambiguous']} ambiguous keys by "
          f"point-in-polygon, {info['key_not_in_index']} keys not in the index, "
          f"{info['no_key']} without a key")
    if info["column_key_mismatch"]:
        print(f"  WARNING: {info['column_key_mismatch']} positives whose meta_src_row_id "
              f"polygon has a different FLD_AR_ID from meta_fld_ar_id")
    pos = df.label.to_numpy() == 1
    with_polygon = pos & res.meta_edge_src_row_id.notna().to_numpy()
    covered = near["covered"]
    flag = "" if covered[with_polygon].all() else "   <-- NOT ALL"
    print(f"  positives covered by their labelling polygon: {int(covered[pos].sum()):,}"
          f"/{int(with_polygon.sum()):,} with a polygon{flag}")
    if near.get("swapped") is not None:
        print(f"  positives whose plain nearest search found another polygon at the same "
              f"distance 0 (adjacent or overlapping SFHA): {near['swapped']:,}")


def print_nearest_summary(df: pd.DataFrame, res: pd.DataFrame, radius: float) -> None:
    d = res.meta_near_dist_m.to_numpy(float)
    found = np.isfinite(d)
    zone_counts = res.meta_near_fld_zone.value_counts()
    zones = ", ".join(f"{z} {c:,}" for z, c in zone_counts.items())
    print(f"  nearest SFHA polygon found within {radius:.0f} m for {int(found.sum()):,}"
          f"/{len(df):,} points; zones: {zones}")
    labels = df.label.to_numpy()
    for lab, name in ((1, "positives"), (0, "negatives")):
        mask = (labels == lab) & found
        if mask.any():
            q = np.percentile(d[mask], [0, 50, 100])
            print(f"    {name:<9} distance min/median/max {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} m")
    if "meta_dist_to_sfha_m" in df:
        ref = df.meta_dist_to_sfha_m.to_numpy(float)
        both = found & np.isfinite(ref)
        diff = np.abs(d[both] - ref[both])
        max_diff = diff.max() if both.any() else float("nan")
        print(f"  check against the dataset's meta_dist_to_sfha_m "
              f"(sample_points.distances_to_sfha): {int(both.sum()):,} points, max |diff| "
              f"{max_diff:.2e} m, {int((diff > 1e-6).sum())} differ by more than 1e-6 m")


def print_edge_summary(df: pd.DataFrame, res: pd.DataFrame) -> None:
    pos = df.label.to_numpy() == 1
    edge = res.meta_edge_dist_m.to_numpy(float)
    for domain, group in df[pos].groupby("domain"):
        dist = edge[group.index.to_numpy()]
        dist = dist[np.isfinite(dist)]
        if len(dist):
            q = np.percentile(dist, [10, 50, 90])
            print(f"  meta_edge_dist_m [{domain}] n={len(dist):,}: p10 {q[0]:.1f}, median "
                  f"{q[1]:.1f}, p90 {q[2]:.1f}, max {dist.max():.1f} m")


def summarise(df: pd.DataFrame, res: pd.DataFrame, src_info: dict, near: dict,
              radius: float) -> None:
    """Print the checks docs/RUN-FL-V2.md asks a reader of the build log to make."""
    print_panel_summary(df, res)
    print_label_summary(df, res, src_info, near)
    print_nearest_summary(df, res, radius)
    print_edge_summary(df, res)


def write_sidecar(res: pd.DataFrame, args: argparse.Namespace, how: str, src_info: dict) -> None:
    """Write the table to args.out, with the settings and input paths in the schema metadata."""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(res, preserve_index=False)
    settings = dict(state=args.state, dataset=str(args.dataset), idx=str(args.idx),
                    haz=str(args.haz), panels=str(args.panels), radius_m=args.radius,
                    limit=args.limit, xy=how, labelling_polygon=src_info,
                    written=time.strftime("%Y-%m-%d %H:%M:%S"))
    metadata = dict(table.schema.metadata or {})
    metadata[b"point_extras"] = json.dumps(settings, sort_keys=True).encode()
    pq.write_table(table.replace_schema_metadata(metadata), args.out)


def build(args: argparse.Namespace) -> int:
    t0 = time.time()
    if args.out.exists() and not args.overwrite:
        print(f"{args.out} exists -- skipping (use --overwrite)")
        return 0
    for path in (args.dataset, args.idx, args.haz, args.panels):
        if not path.exists():
            sys.exit(f"missing input {path}")

    df, xy, how = load_points(args.dataset, args.limit)
    subset = f" (random {args.limit:,}, seed {LIMIT_SEED})" if args.limit else ""
    print(f"[{args.state}] {len(df):,} points from {args.dataset}{subset}; xy: {how}", flush=True)
    idx = load_index(args.idx)
    haz = HazReader(args.haz)
    if haz.n_rows != len(idx):
        sys.exit(f"{args.haz} has {haz.n_rows:,} rows, {args.idx} {len(idx):,}: not the same prep")
    sfha = idx[idx.SFHA_TF == "T"]

    src, src_info = resolve_label_polygons(df, xy, idx, haz)
    print(f"  resolved labelling polygons ({time.time() - t0:.0f}s)", flush=True)
    near = nearest_and_edges(haz, sfha, xy, src, args.radius)

    # Adjacent or overlapping SFHA polygons can also be at distance 0 from a positive, so the
    # labelling polygon is its nearest whenever it covers the point.
    pos = df.label.to_numpy() == 1
    use_src = pos & near["covered"]
    near["swapped"] = int((use_src & (near["near_r"] != src)).sum())
    near_r = np.where(use_src, src, near["near_r"])
    near_d = np.where(use_src, 0.0, near["near_d"])

    pan, fixed = load_panels(args.panels)
    pan_row, pan_n, by_dfirm = panel_of(pan, xy, df.dfirm_id.to_numpy().astype(object))
    print(f"  {len(pan):,} panels ({int(fixed.sum())} repaired after reprojection); "
          f"{by_dfirm:,} of {int((pan_n > 1).sum()):,} multi-panel points took their own "
          f"DFIRM's panel", flush=True)

    res = sidecar_frame(df, idx, near_r, near_d, src, near["edge"], pan, pan_row, pan_n)
    summarise(df, res, src_info, near, args.radius)
    write_sidecar(res, args, how, src_info)
    print(f"\n[{args.state}] wrote {len(res):,} rows x {res.shape[1]} columns -> {args.out}  "
          f"({time.time() - t0:.0f}s)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="FL")
    ap.add_argument("--dataset", type=Path, help="default data/processed/dataset_<ST>.parquet")
    ap.add_argument("--idx", type=Path, help="default data/interim/prep/idx_<ST>.parquet")
    ap.add_argument("--haz", type=Path, help="default data/interim/prep/haz_<ST>_5070.parquet")
    ap.add_argument("--panels", type=Path, help="default data/interim/nfhl_states/pan_<ST>.parquet")
    ap.add_argument("--out", type=Path,
                    help="default data/interim/analysis/point_extras_<ST>.parquet")
    ap.add_argument("--radius", type=float, default=RADIUS_M,
                    help="nearest-SFHA search radius in metres")
    ap.add_argument("--limit", type=int, help="random subset of N points, for smoke tests")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.limit is not None and args.out is None:
        sys.exit("--limit writes a partial sidecar: give an explicit --out")
    defaults = default_paths(args.state)
    args.dataset = args.dataset or defaults["dataset"]
    args.idx = args.idx or defaults["idx"]
    args.haz = args.haz or defaults["haz"]
    args.panels = args.panels or defaults["panels"]
    args.out = args.out or defaults["out"]
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
