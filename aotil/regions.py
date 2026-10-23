#!/usr/bin/env python3
"""
Regions: the connected components of an OODSelect selection, one row per component.

A region is a connected component of the subgraph that the top-S points of a selection
(<cell>_S<S>_theta.npy) induce in the kNN graph the selection was penalised on. The graph is
rebuilt exactly as oodselect_paper.py built it, and each budget is checked against what the
selection run recorded (seed, variant, k, component and tile counts). A budget that fails a check
writes nothing and the run exits 1. See docs/REGION-ANALYSIS-FLORIDA.md sections 2-3.

Outputs in --out (prefix = <cell>_<variant>_S<S>):
  <prefix>_points.parquet   one row per selected point, ordered by component
  <prefix>_components.csv   one row per component; ge_min_size flags those of >= --min-size points

    python aotil/regions.py --cells-dir data/interim/aotil/cells --cell FL_band_AE_s0 \\
        --results-dir results/<date>/oodselect_paper/spatial_mu3_knn10 \\
        --budgets 250 500 --data data/processed/dataset_FL.parquet --out results/<date>/regions
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # sample_points
sys.path.insert(0, str(HERE))            # graph, cells, zoo
import graph  # noqa: E402
import sample_points as sp  # noqa: E402
from cells import load_cell  # noqa: E402

PREP_DIR = Path("data/interim/prep")
CRS_POINTS = 5070          # the sampler's metric CRS; every distance here is in it
LONLAT_TOL = 1e-9          # degrees; the cell's lon/lat are copies of the dataset's
DEFAULT_KNN = 10           # oodselect_paper.py's default, used when no record states k

DATA_COLS = ["point_id", "domain", "label", "neg_protocol", "tile_50km", "lon", "lat",
             "county_fips", "dfirm_id", "meta_fld_ar_id", "landcover", "hand",
             "meta_dist_to_river_m"]
OPTIONAL_COLS = ["meta_src_row_id", "meta_point_county", "geometry"]
# The points table; cell, variant, S and seed follow, so tables of several runs can be stacked.
POINT_COLS = ["point_id", "lon", "lat", "label", "component", "component_size",
              "tile_50km", "county_fips", "meta_point_county", "meta_fld_ar_id",
              "meta_src_row_id", "dfirm_id"]

# NLCD class names, for the dominant land cover of each component.
NLCD = {11: "open water", 12: "perennial ice/snow", 21: "developed, open space",
        22: "developed, low intensity", 23: "developed, medium intensity",
        24: "developed, high intensity", 31: "barren land", 41: "deciduous forest",
        42: "evergreen forest", 43: "mixed forest", 52: "shrub/scrub",
        71: "grassland/herbaceous", 81: "pasture/hay", 82: "cultivated crops",
        90: "woody wetlands", 95: "emergent herbaceous wetlands"}

TAG_RE = re.compile(r"^(?P<state>[A-Z]{2})_(?P<protocol>[A-Za-z]+)_(?P<train>AE|A)_s(?P<seed>\d+)$")


class BudgetFailed(RuntimeError):
    """One budget cannot be written faithfully; the others still run."""


# ----------------------------------------------------------------- inputs
def parse_tag(tag: str) -> dict:
    """FL_band_AE_s0 -> state, protocol, train and eval domains, cell seed."""
    match = TAG_RE.match(tag)
    if not match:
        raise SystemExit(f"cannot parse cell tag {tag!r} (expected <ST>_<protocol>_<A|AE>_s<seed>)")
    fields = match.groupdict()
    eval_domain = "AE" if fields["train"] == "A" else "A"
    return dict(state=fields["state"], protocol=fields["protocol"], train=fields["train"],
                eval=eval_domain, seed=int(fields["seed"]))


def geometry_crs_id(pf: pq.ParquetFile) -> dict:
    """The CRS id GeoParquet metadata states for the geometry column, e.g.
    {"authority": "EPSG", "code": 5070}; {} when it states none."""
    metadata = pf.schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b"geo") or b"{}")
    crs = geo.get("columns", {}).get("geometry", {}).get("crs") or {}
    return crs.get("id", {})


def load_dataset(path: Path) -> pd.DataFrame:
    """The dataset columns this script joins, with geometry only if it is EPSG:5070."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    missing = [c for c in DATA_COLS if c not in names]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    cols = DATA_COLS + [c for c in OPTIONAL_COLS if c in names]
    if "geometry" in cols:
        crs = geometry_crs_id(pf)
        if not (crs.get("authority") == "EPSG" and int(crs.get("code", -1)) == CRS_POINTS):
            print(f"  note: {path.name} geometry CRS is {crs or 'unstated'}, not "
                  f"EPSG:{CRS_POINTS}; projecting lon/lat instead", flush=True)
            cols.remove("geometry")
    df = pq.read_table(path, columns=cols).to_pandas()
    if not df.point_id.is_unique:
        raise SystemExit(f"{path}: point_id is not unique")
    if "meta_point_county" not in df:
        print(f"  note: {path.name} has no meta_point_county; counties fall back to "
              f"county_fips, the DFIRM's county, which a band negative need not lie in",
              flush=True)
        df["meta_point_county"] = pd.array([pd.NA] * len(df), dtype="string")
    return df


def ood_rows(cell, df: pd.DataFrame, info: dict) -> tuple[pd.DataFrame, str]:
    """Dataset rows in the cell's OOD column order, verified against the cell.

    A v2 cell stores the point_id of its OOD points. A v1 cell does not, so its OOD rows are
    rebuilt as zoo.py made them: the test tiles of zoo.split_tiles(cell seed), then the eval
    domain's rows of zoo.subset in dataset order. Either way lon/lat, labels and tiles must
    reproduce the cell's, or the run aborts (--data is not the dataset the cell was built from).
    """
    if cell.point_id is not None:
        pid = np.asarray(cell.point_id).astype(str)
        by_id = df.set_index("point_id")
        absent = ~pd.Index(pid).isin(by_id.index)
        if absent.any():
            raise SystemExit(f"{absent.sum():,} of the cell's {len(pid):,} point_ids are not "
                             f"in the dataset (e.g. {pid[absent][:3].tolist()}); wrong --data?")
        rows = by_id.loc[pid].reset_index()
        how = "the cell's point_id"
    else:
        import zoo                       # loads sklearn and xgboost, so only on this path
        _, test = zoo.split_tiles(df, info["seed"])
        rows = zoo.subset(df, info["eval"], info["protocol"])
        rows = rows[rows.tile_50km.isin(test)].reset_index(drop=True)
        how = (f"rebuilt (v1 cell): zoo.split_tiles(seed={info['seed']}) + "
               f"zoo.subset({info['eval']}, {info['protocol']}) on {len(test)} test tiles")

    if len(rows) != cell.d:
        raise SystemExit(f"OOD rows from the dataset: {len(rows):,}, cell columns: {cell.d:,}. "
                         f"The cell was not built from this dataset.")
    dlon = float(np.abs(rows.lon.to_numpy(float) - cell.lon.astype(float)).max())
    dlat = float(np.abs(rows.lat.to_numpy(float) - cell.lat.astype(float)).max())
    bad_y = int((rows.label.to_numpy() != np.asarray(cell.y_ood)).sum())
    if cell.tile_50km is None:
        bad_t = 0
    else:
        bad_t = int((rows.tile_50km.astype(str).to_numpy() != cell.tile_50km.astype(str)).sum())
    print(f"  OOD rows: {how}; max |dlon| {dlon:.1e}, |dlat| {dlat:.1e}, "
          f"label mismatches {bad_y}, tile mismatches {bad_t}", flush=True)
    if dlon > LONLAT_TOL or dlat > LONLAT_TOL or bad_y or bad_t:
        raise SystemExit("the dataset rows do not reproduce the cell's OOD order; aborting")
    return rows, how


def points_5070(rows: pd.DataFrame) -> tuple[np.ndarray, str]:
    """(x, y) in EPSG:5070, the CRS the sampler measured in: the dataset's own geometry, else
    lon/lat projected. Only distances use it; the graph stays on the selection's EPSG:3086."""
    if "geometry" in rows:
        xy = shapely.get_coordinates(shapely.from_wkb(rows.geometry.to_numpy()))
        if len(xy) == len(rows):
            return xy, "dataset geometry (EPSG:5070)"
    tf = Transformer.from_crs("EPSG:4269", f"EPSG:{CRS_POINTS}", always_xy=True)
    x, y = tf.transform(rows.lon.to_numpy(float), rows.lat.to_numpy(float))
    return np.column_stack([x, y]), f"lon/lat projected EPSG:4269 -> {CRS_POINTS}"


def covering_polygons(row_ids, geoms: dict, point) -> list:
    """The row_ids among row_ids whose polygon covers point."""
    hits = []
    for r in row_ids:
        g = geoms.get(r)
        if g is not None and g.covers(point):
            hits.append(r)
    return hits


def source_row_ids(sel: pd.DataFrame, xy: np.ndarray, state: str,
                   prep_dir: Path) -> tuple[pd.Series, str]:
    """meta_src_row_id of each selected point: the idx_<ST> row_id of the polygon the point was
    sampled from (for a band negative, the polygon whose ring produced it).

    A v2 dataset carries it. For v1 positives it is recovered from (dfirm_id, meta_fld_ar_id)
    through idx_<ST>, as FLD_AR_ID alone repeats; where even the pair names several rows (39 in
    Florida), from the candidate polygon that covers the point. v1 never recorded a band
    negative's ring polygon, so its id stays null. sel has a RangeIndex aligned with xy.
    """
    if "meta_src_row_id" in sel:
        s = sel.meta_src_row_id.astype("Int64")
        return s, f"from the dataset ({int(s.notna().sum())} of {len(s)} set)"
    out = pd.Series(pd.NA, index=sel.index, dtype="Int64")
    idx_p = prep_dir / f"idx_{state}.parquet"
    if not idx_p.exists():
        return out, f"unavailable: no meta_src_row_id in the dataset and {idx_p} missing"

    idx = pq.read_table(idx_p, columns=["row_id", "DFIRM_ID", "FLD_AR_ID"]).to_pandas()
    idx = idx.astype({"DFIRM_ID": str, "FLD_AR_ID": str})
    has_key = sel.meta_fld_ar_id.notna().to_numpy()
    keys = pd.DataFrame(dict(i=np.flatnonzero(has_key),
                             DFIRM_ID=sel.dfirm_id[has_key].astype(str).to_numpy(),
                             FLD_AR_ID=sel.meta_fld_ar_id[has_key].astype(str).to_numpy()))
    cand = keys.merge(idx, on=["DFIRM_ID", "FLD_AR_ID"], how="inner")
    n_cand = cand.groupby("i").row_id.transform("size")
    unique = cand[n_cand == 1]
    out.iloc[unique.i.to_numpy()] = unique.row_id.to_numpy()

    repeated = cand[n_cand > 1]
    n_repeated = int(repeated.i.nunique())
    n_resolved = 0
    if n_repeated:
        # read_geoms opens <sp.PREP_DIR>/haz_<ST>_5070.parquet; point it at prep_dir.
        sp.PREP_DIR = prep_dir
        rid = repeated.row_id.unique()
        geoms = dict(zip(rid.tolist(), sp.read_geoms(state, rid)))
        for i, grp in repeated.groupby("i"):
            hits = covering_polygons(grp.row_id, geoms, shapely.Point(xy[i]))
            if len(hits) == 1:
                out.iloc[i] = hits[0]
                n_resolved += 1
    n_unknown = int(has_key.sum()) - int(cand.i.nunique())
    n_null = int(out.isna().sum())
    neg_null = int((out.isna() & (sel.label == 0)).sum())
    return out, (f"recovered from (dfirm_id, meta_fld_ar_id) via {idx_p.name}: "
                 f"{len(unique)} points by a unique key, {n_resolved} of {n_repeated} points with "
                 f"a repeated key by point-in-polygon, {n_unknown} points whose key is not in "
                 f"the index; {n_null} null ({neg_null} negatives, whose source ring this "
                 f"dataset never recorded)")


def resolve_knn(results_dir: Path, stem: str, variant: str, cli_knn: int | None) -> tuple[int, str]:
    """k of the selection's graph and where it was read.

    Sources: the knn column of <stem>_result.csv, <stem>_meta.json, the variant name (..._knn10)
    and --knn as a cross-check. Every source present must agree: a different k would give
    different regions without raising any error.
    """
    found = {}
    res_p = results_dir / f"{stem}_result.csv"
    if res_p.exists():
        result = pd.read_csv(res_p)
        if "knn" in result and result.knn.notna().any():
            found["result.csv"] = int(result.knn.iloc[0])
    meta_p = results_dir / f"{stem}_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if "knn" in meta:
            found["meta.json"] = int(meta["knn"])
    k_in_name = re.search(r"knn(\d+)", variant)
    if k_in_name:
        found["variant name"] = int(k_in_name.group(1))
    if cli_knn is not None:
        found["--knn"] = int(cli_knn)
    if len(set(found.values())) > 1:
        raise BudgetFailed(f"k disagrees across sources: {found}")
    if not found:
        return DEFAULT_KNN, "oodselect_paper.py default (nothing recorded)"
    return list(found.values())[0], " = ".join(found)


def recorded(results_dir: Path, stem: str) -> dict:
    """What the selection run itself wrote about this budget, if anything."""
    res_p = results_dir / f"{stem}_result.csv"
    if not res_p.exists():
        return {}
    return pd.read_csv(res_p).iloc[0].to_dict()


def check_variant(rec: dict, variant: str, stem: str) -> str:
    """The outputs' variant name must be the one the run recorded; returns where it was found.

    Without this check, a spatial label pointed at the classic folder would pass every other
    check (they read the same folder) and write the classic fragments as the spatial regions.
    The variant column comes before the name built from mu and knn, because a suffixed variant
    (null-select, a family split) records its own name while its mu reads like the plain one's.
    """
    if pd.notna(rec.get("variant")):
        src, name = "result.csv variant", str(rec["variant"])
    elif pd.notna(rec.get("mu")) and pd.notna(rec.get("knn")):
        mu = float(rec["mu"])
        src = "result.csv mu/knn"
        if mu == 0:
            name = "classic"
        else:
            name = f"spatial_mu{mu:g}_knn{int(rec['knn'])}"
    else:
        return "unverified: no result.csv records it"
    if name != variant:
        raise BudgetFailed(f"{stem}_result.csv records variant {name!r} ({src}), but the outputs "
                           f"would be named {variant!r}; pass the --results-dir and --variant of "
                           f"the run these thetas came from")
    return src


# ----------------------------------------------------------------- regions
def top_mask(theta: np.ndarray, S: int) -> np.ndarray:
    """The selection as oodselect_paper.py and report_paper.py take it: the S largest theta,
    ties to the lower index (stable argsort)."""
    mask = np.zeros(len(theta), bool)
    mask[np.argsort(-theta, kind="stable")[:S]] = True
    return mask


def components(A, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Component id of each selected point, and the component sizes in id order.

    Ids are 0..n-1 by size, largest first, ties to the component holding the lowest OOD column
    index. scipy's own labels follow visiting order, which shifts with the cell's row order.
    """
    n, labels = connected_components(A[mask][:, mask], directed=False)
    sizes = np.bincount(labels, minlength=n)
    first = np.full(n, mask.sum())
    np.minimum.at(first, labels, np.arange(len(labels)))    # lowest OOD index per component
    order = np.lexsort((first, -sizes))
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    return rank[labels], sizes[order]


def point_county(t: pd.DataFrame) -> pd.Series:
    """The county each point lies in: meta_point_county, else county_fips.

    county_fips is the point's DFIRM's county, and a band negative takes the DFIRM
    of the polygon whose ring produced it, which can be the next county's.
    """
    if "meta_point_county" not in t:
        return t.county_fips
    return t.meta_point_county.astype(object).where(t.meta_point_county.notna(), t.county_fips)


def _ranked(s: pd.Series) -> list:
    """Distinct non-null values, most frequent first, ties by value."""
    vc = s.dropna().astype(str).value_counts()
    return sorted(vc.index, key=lambda v: (-vc[v], v))


def polygon_keys(sel: pd.DataFrame) -> np.ndarray:
    """A name for each point's source polygon: 'r<row_id>', else '<dfirm_id>|<fld_ar_id>',
    else '' when neither is known."""
    by_row_id = "r" + sel.meta_src_row_id.astype(str)
    by_pair = sel.dfirm_id.astype(str) + "|" + sel.meta_fld_ar_id.astype(str)
    pair_or_blank = np.where(sel.meta_fld_ar_id.notna(), by_pair, "")
    return np.where(sel.meta_src_row_id.notna(), by_row_id, pair_or_blank)


def nn_spacing_m(points_xy: np.ndarray) -> np.ndarray:
    """Each point's distance to its nearest other point; [nan] for a single point."""
    if len(points_xy) > 1:
        # k=2 because each point's nearest neighbour in the tree is itself.
        return cKDTree(points_xy).query(points_xy, k=2)[0][:, 1]
    return np.array([np.nan])


def dominant_landcover(landcover: pd.Series) -> tuple:
    """(NLCD code, class name, share of points) of the most common class, ties to the lower
    code; (NA, "", nan) when no point has one."""
    lc = landcover.dropna()
    if not len(lc):
        return pd.NA, "", np.nan
    vc = lc.round().astype(int).value_counts()
    code = int(sorted(vc.index, key=lambda v: (-vc[v], v))[0])
    return code, NLCD.get(code, ""), float(vc[code] / len(lc))


def component_table(sel: pd.DataFrame, xy: np.ndarray, comp: np.ndarray,
                    sizes: np.ndarray, min_size: int) -> pd.DataFrame:
    """One row per component. Distances are in EPSG:5070: extent_km is the diagonal of the
    points' bounding box, median_nn_spacing_m is within the component. tiles, counties and
    dfirm_ids are ';'-joined, most points first."""
    poly = polygon_keys(sel)
    rows = []
    for c in range(len(sizes)):
        members = comp == c
        pts, pts_xy = sel[members], xy[members]
        dx = np.ptp(pts_xy[:, 0]) / 1e3
        dy = np.ptp(pts_xy[:, 1]) / 1e3
        lc_code, lc_name, lc_frac = dominant_landcover(pts.landcover)
        tiles = _ranked(pts.tile_50km)
        counties = _ranked(point_county(pts))
        dfirms = _ranked(pts.dfirm_id)
        positive_keys = poly[members][(pts.label == 1).to_numpy()]
        rows.append(dict(
            component=c, n_points=int(members.sum()), n_positive=int((pts.label == 1).sum()),
            n_tiles=len(tiles), tiles=";".join(tiles),
            n_counties=len(counties), counties=";".join(counties),
            extent_km=float(np.hypot(dx, dy)),
            median_nn_spacing_m=float(np.median(nn_spacing_m(pts_xy))),
            dominant_landcover=lc_code,
            median_hand=float(pts.hand.median()),
            median_dist_to_river_m=float(pts.meta_dist_to_river_m.median()),
            ge_min_size=bool(members.sum() >= min_size),
            dominant_landcover_name=lc_name,
            dominant_landcover_frac=lc_frac,
            extent_x_km=float(dx), extent_y_km=float(dy),
            n_polygons=int(len(set(positive_keys[positive_keys != ""]))),
            dfirm_ids=";".join(dfirms),
            lon_center=float(pts.lon.mean()), lat_center=float(pts.lat.mean()),
        ))
    out = pd.DataFrame(rows)
    out["dominant_landcover"] = out.dominant_landcover.astype("Int64")
    return out


def print_components(tab: pd.DataFrame, min_size: int) -> None:
    big = tab[tab.ge_min_size]
    if big.empty:
        print(f"  no component of >= {min_size} points")
        return
    print(f"  {'cc':>3} {'n':>5} {'pos':>4} {'tiles':<16} {'counties':<18} {'ext_km':>6} "
          f"{'nn_m':>5} {'hand':>5} {'river_m':>7} {'polys':>5}  land cover")
    for r in big.itertuples():
        print(f"  {r.component:>3} {r.n_points:>5} {r.n_positive:>4} {r.tiles:<16} "
              f"{r.counties:<18} {r.extent_km:>6.1f} {r.median_nn_spacing_m:>5.0f} "
              f"{r.median_hand:>5.1f} {r.median_dist_to_river_m:>7.0f} {r.n_polygons:>5}  "
              f"{r.dominant_landcover_name} ({r.dominant_landcover_frac:.0%})")


def load_theta(th_p: Path, d: int, S: int) -> np.ndarray:
    """The selection's theta, one entry per OOD point, with S a proper budget."""
    if not th_p.exists():
        raise BudgetFailed(f"{th_p} missing")
    theta = np.load(th_p)
    if len(theta) != d:
        raise BudgetFailed(f"{th_p.name} has {len(theta):,} entries, the cell {d:,} OOD points")
    if not 0 < S < d:
        raise BudgetFailed(f"budget {S} outside (0, d = {d})")
    return theta


def coherence_check(rec: dict, n_cc: int, n_tiles: int) -> str:
    """The component and tile counts must equal those the selection run recorded; otherwise the
    graph or the mask is not the selection's. Returns the report's wording."""
    if "coh_n_cc" not in rec or "coh_tiles" not in rec:
        return "; no recorded coh_n_cc/coh_tiles to check against"
    want_cc, want_tiles = int(rec["coh_n_cc"]), int(rec["coh_tiles"])
    if want_cc != n_cc or want_tiles != n_tiles:
        raise BudgetFailed(f"{n_cc} components / {n_tiles} tiles here against "
                           f"{want_cc} / {want_tiles} recorded: "
                           f"the graph or the mask is not the selection's")
    return f"; the selection run recorded {want_cc} components, {want_tiles} tiles: match"


def run_budget(S: int, args, state: str, variant: str, cell, rows: pd.DataFrame, xy: np.ndarray,
               graphs: dict, paths: tuple[Path, Path]) -> None:
    """Find, check, report and write the regions of one budget. graphs caches k -> adjacency
    across budgets. Raises BudgetFailed, before writing anything, when a check fails."""
    stem = f"{args.cell}_S{S}"
    theta = load_theta(args.results_dir / f"{stem}_theta.npy", cell.d, S)
    rec = recorded(args.results_dir, stem)
    # The theta file name does not carry the selection seed, so it is checked here.
    if "seed" in rec and int(rec["seed"]) != args.seed:
        raise BudgetFailed(f"{stem}_result.csv records selection seed {int(rec['seed'])}, "
                           f"--seed is {args.seed}")
    v_src = check_variant(rec, variant, stem)
    knn, k_src = resolve_knn(args.results_dir, stem, variant, args.knn)
    if knn not in graphs:
        # oodselect_paper.py's call: build_adjacency's defaults (EPSG:3086, self excluded,
        # OR-symmetrised), so the components are the ones the penalty saw.
        graphs[knn] = graph.build_adjacency(cell.lon, cell.lat, k=knn)

    mask = top_mask(theta, S)
    comp, sizes = components(graphs[knn], mask)
    sel = rows[mask].reset_index(drop=True)
    sxy = xy[mask]
    src, src_note = source_row_ids(sel, sxy, state, args.prep_dir)
    sel["meta_src_row_id"] = src
    sel["component"] = comp.astype(np.int32)
    sel["component_size"] = sizes[comp].astype(np.int32)
    tab = component_table(sel, sxy, comp, sizes, args.min_size)

    n_cc = len(sizes)
    n_tiles = int(sel.tile_50km.nunique())
    n_pos = int((sel.label == 1).sum())
    print(f"\n--- S = {S}: {S:,} of {cell.d:,} OOD points ({n_pos} positive, "
          f"{S - n_pos} negative) | variant {variant} ({v_src}) | k = {knn} ({k_src})",
          flush=True)
    check = coherence_check(rec, n_cc, n_tiles)
    print(f"  {n_cc} components across {n_tiles} tiles{check}")
    big = tab[tab.ge_min_size]
    big_sizes = ", ".join(map(str, big.n_points)) or "-"
    print(f"  {len(big)} components of >= {args.min_size} points hold "
          f"{int(big.n_points.sum())} points ({big.n_points.sum() / S:.1%}); sizes {big_sizes}")
    print(f"  meta_src_row_id: {src_note}")
    print_components(tab, args.min_size)

    pts = sel[POINT_COLS].sort_values("component", kind="stable").reset_index(drop=True)
    pts = pts.astype({"point_id": str, "label": np.int8, "tile_50km": str})
    pts["cell"] = args.cell
    pts["variant"] = variant
    pts["S"] = np.int32(S)
    pts["seed"] = np.int32(args.seed)
    pts_p, comp_p = paths
    pts_p.parent.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(pts_p, index=False)
    tab.to_csv(comp_p, index=False)
    print(f"  wrote {pts_p} ({len(pts)} rows), {comp_p} ({len(tab)} rows)", flush=True)


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", required=True, type=Path)
    ap.add_argument("--cell", required=True, help="cell tag, e.g. FL_band_AE_s0")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="one variant directory, e.g. "
                         "results/<date>/oodselect_paper/spatial_mu3_knn10")
    ap.add_argument("--variant", help="name used in the output files, checked against "
                                      "*_result.csv (default: the --results-dir directory name)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 500])
    ap.add_argument("--seed", type=int, default=0,
                    help="selection seed (oodselect_paper.py --seed), checked against *_result.csv")
    ap.add_argument("--data", required=True, type=Path,
                    help="the dataset parquet the cell was built from")
    ap.add_argument("--prep-dir", type=Path, default=PREP_DIR,
                    help="idx_<ST>.parquet + haz_<ST>_5070.parquet, for v1 meta_src_row_id")
    ap.add_argument("--knn", type=int, help="cross-check only: must equal the recorded k")
    ap.add_argument("--out", type=Path, help="default: results/<today>/regions")
    ap.add_argument("--min-size", type=int, default=20,
                    help="components of at least this many points are flagged and printed")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo budgets whose two outputs exist")
    args = ap.parse_args()
    info = parse_tag(args.cell)
    variant = args.variant or args.results_dir.name
    out = args.out or Path("results") / date.today().isoformat() / "regions"

    todo = []
    for S in args.budgets:
        prefix = f"{args.cell}_{variant}_S{S}"
        paths = (out / f"{prefix}_points.parquet", out / f"{prefix}_components.csv")
        if all(p.exists() for p in paths) and not args.overwrite:
            print(f"{prefix}: outputs exist; skipping (--overwrite to redo)")
            continue
        todo.append((S, paths))
    if not todo:
        return 0

    t0 = time.time()
    print(f"=== regions {args.cell} | {variant} | budgets {[S for S, _ in todo]} ===", flush=True)
    cell = load_cell(args.cells_dir, args.cell)
    cell.require_spatial()
    df = load_dataset(args.data)
    print(f"  cell: {cell.n_models} models x {cell.d:,} OOD points | dataset "
          f"{args.data.name}: {len(df):,} rows", flush=True)
    rows, _ = ood_rows(cell, df, info)
    xy, xy_src = points_5070(rows)
    print(f"  distances from {xy_src}", flush=True)

    graphs, failed = {}, []
    for S, paths in todo:
        try:
            run_budget(S, args, info["state"], variant, cell, rows, xy, graphs, paths)
        except BudgetFailed as e:
            print(f"\n--- S = {S}: FAILED, nothing written: {e}", flush=True)
            failed.append(S)
    summary = f"{len(todo) - len(failed)} budget(s) written to {out}"
    if failed:
        summary += f"; FAILED {failed}"
    print(f"\ndone in {time.time() - t0:.0f}s: {summary}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
