#!/usr/bin/env python3
"""
Regions: the connected pieces of an OODSelect subset, written as two tables the
boundary analysis reads (docs/REGION-ANALYSIS-FLORIDA.md sections 2-3).

WHAT A REGION IS
----------------
The top-S points of one selection (*_S<S>_theta.npy), joined by the kNN graph
the selection was penalised on. A region is one connected component of the
subgraph those points induce. On v1, the spatial AE->A selection at S = 500 is 8
components, four of them basin-scale pieces of >= 20 points; the classic one is
186 fragments. Everything downstream (aotil/lobes.py, the GIS layers) works one
component at a time, so this script writes one row per component.

DECISIONS, EACH WITH ITS REASON
-------------------------------
1. THE GRAPH IS REBUILT, NOT APPROXIMATED. oodselect_paper.py calls
   graph.build_adjacency(cell.lon, cell.lat, k=knn) with its defaults (EPSG:3086,
   k nearest neighbours with self excluded, OR-symmetrised), and so does this
   script. That way the components are the ones the penalty saw. k comes from the
   run's own records: the knn column of *_result.csv, or *_meta.json if it ever
   stores one, then the variant name (..._knn10). It is never guessed; --knn is
   only a cross-check. Each budget is then checked against the component and tile
   counts the selection run recorded (coh_n_cc, coh_tiles), and a mismatch fails
   that budget. A different graph would give different regions without raising
   any error.
2. THE MASK IS THE PAPER'S. np.argsort(-theta, kind="stable")[:S], exactly as
   oodselect_paper.py and report_paper.py take it (ties go to the lower index).
3. COMPONENT IDS ARE 0..n-1 BY SIZE, LARGEST FIRST. Ties go to the component
   holding the lowest OOD column index. scipy's own labels follow visiting order,
   which shifts whenever the cell's row order does; "component 0 is the largest"
   survives a rebuild. REGION-ANALYSIS-FLORIDA.md section 2.1 numbers the v1
   components by scipy's labels: its cc 0, 1, 4, 2 are components 0, 1, 2, 3 here.
4. POINT_ID. A v2 cell stores the point_id of its OOD half (zoo.py). The join to
   --data is checked: every id must be found, lon/lat must equal the cell's to
   1e-9, and labels must equal y_ood. A v1 cell has no point_id, so its OOD rows
   are rebuilt the way zoo.run_cell made them: zoo.split_tiles(df, cell seed),
   then zoo.subset(df, eval domain, protocol) on the test tiles, in dataset
   order. The same checks apply and the script aborts if any fails. The cell
   seed comes from the tag (_s<seed>), so --data must be the dataset the cell
   was built from.
5. DISTANCES ARE IN EPSG:5070, taken from the dataset's own point geometry (the
   CRS the sampler measured its 100 m floor in). Lon/lat are projected only when
   a dataset has no 5070 geometry. The graph stays on 3086 because that is what
   the selection used.
       extent_km            diagonal of the component's bounding box, in km.
                            It matches the "extent" column of section 2.1 to
                            about 1 km (40.0 / 32.3 / 21.0 / 14.6 against 41 /
                            33 / 22 / 14); extent_x_km and extent_y_km give the
                            two spans.
       median_nn_spacing_m  median distance from each point to its nearest
                            neighbour WITHIN the component (NaN for one point).
6. SOURCE POLYGON. meta_src_row_id is the idx_<ST> row_id of the polygon behind
   a point (sample_points.py note 8), and it is read from the dataset when the
   dataset carries it. v1 does not. For v1 positives the id is recovered from
   (dfirm_id, meta_fld_ar_id) through prep/idx_<ST>.parquet; the pair is used
   because FLD_AR_ID alone repeats. Where the pair itself names several rows
   (39 in Florida), the id is taken from the candidate polygon that covers the
   point, reading only those polygons' row groups (sample_points.read_geoms). The
   ring polygon behind a v1 band negative was never recorded, so its id stays
   null.
7. --seed IS THE SELECTION SEED (oodselect_paper.py --seed). The theta file name
   does not carry it, so it is checked against each budget's *_result.csv.
8. THE VARIANT IS THE RUN'S, NOT THE FOLDER'S. The outputs are named after
   --variant (default: the --results-dir name), and that name must equal what
   each *_result.csv records: its variant column, or, in a file without one,
   the name oodselect_paper.py builds from its mu and knn columns ("classic" at
   mu = 0, else spatial_mu<mu>_knn<k>). The column comes first because a
   suffixed variant (null-select, a family split) would record its own name
   there while its mu reads like the plain variant's. A disagreement fails the
   budget. Without this, a typo that points a spatial label at the classic
   folder writes the 186 classic fragments as the spatial regions, every other
   check passes (they come from the same folder), and lobes.py analyses the
   wrong set. With no result.csv the name cannot be checked, and the run prints
   that it is unverified.

Nothing written here is a model feature: the outputs are per-region tables, and
their point columns are identifiers and meta_ provenance.

OUTPUTS (prefix = <cell>_<variant>_S<S>)
-------
<out>/<prefix>_points.parquet     one row per selected point, ordered by component:
                                  point_id, lon, lat, label, component,
                                  component_size, tile_50km, county_fips,
                                  meta_fld_ar_id, meta_src_row_id, dfirm_id,
                                  then cell, variant, S, seed (constant per
                                  file, so tables from several seeds or budgets
                                  can be stacked and still told apart)
<out>/<prefix>_components.csv     one row per component, ALL of them; ge_min_size
                                  flags those of >= --min-size points. tiles,
                                  counties and dfirm_ids are ';'-joined, most
                                  points first.

USAGE
-----
    python aotil/regions.py --cells-dir data/interim/aotil/cells --cell FL_band_AE_s0 \\
        --results-dir results/<date>/oodselect_paper/spatial_mu3_knn10 \\
        --budgets 250 500 --data data/processed/dataset_FL.parquet
    ... --out results/<date>/regions   (default: results/<today>/regions)
    ... --overwrite                    (redo budgets whose two outputs exist)
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
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph  # noqa: E402
from cells import load_cell  # noqa: E402

PREP_DIR = Path("data/interim/prep")
CRS_POINTS = 5070          # the sampler's metric CRS; every distance here is in it
LONLAT_TOL = 1e-9          # degrees; the cell's lon/lat are copies of the dataset's

DATA_COLS = ["point_id", "domain", "label", "neg_protocol", "tile_50km", "lon", "lat",
             "county_fips", "dfirm_id", "meta_fld_ar_id", "landcover", "hand",
             "meta_dist_to_river_m"]
OPTIONAL_COLS = ["meta_src_row_id", "geometry"]
POINT_COLS = ["point_id", "lon", "lat", "label", "component", "component_size",
              "tile_50km", "county_fips", "meta_fld_ar_id", "meta_src_row_id", "dfirm_id"]

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
    """cells.cell_tag in reverse: FL_band_AE_s0 -> state, protocol, domains, seed."""
    m = TAG_RE.match(tag)
    if not m:
        raise SystemExit(f"cannot parse cell tag {tag!r} (expected <ST>_<protocol>_<A|AE>_s<seed>)")
    d = m.groupdict()
    return dict(state=d["state"], protocol=d["protocol"], train=d["train"],
                eval="AE" if d["train"] == "A" else "A", seed=int(d["seed"]))


def load_dataset(path: Path) -> pd.DataFrame:
    """The dataset columns this script joins, with geometry only if it is EPSG:5070."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    missing = [c for c in DATA_COLS if c not in names]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    cols = DATA_COLS + [c for c in OPTIONAL_COLS if c in names]
    if "geometry" in cols:
        geo = json.loads((pf.schema_arrow.metadata or {}).get(b"geo", b"{}") or b"{}")
        crs = (geo.get("columns", {}).get("geometry", {}).get("crs") or {}).get("id", {})
        if not (crs.get("authority") == "EPSG" and int(crs.get("code", -1)) == CRS_POINTS):
            print(f"  note: {path.name} geometry CRS is {crs or 'unstated'}, not "
                  f"EPSG:{CRS_POINTS}; projecting lon/lat instead", flush=True)
            cols.remove("geometry")
    df = pq.read_table(path, columns=cols).to_pandas()
    if not df.point_id.is_unique:
        raise SystemExit(f"{path}: point_id is not unique")
    return df


def ood_rows(cell, df: pd.DataFrame, info: dict) -> tuple[pd.DataFrame, str]:
    """Dataset rows in the cell's OOD column order, verified against the cell."""
    pid = getattr(cell, "point_id", None)
    if pid is not None:
        pid = np.asarray(pid).astype(str)
        by_id = df.set_index("point_id")
        absent = ~pd.Index(pid).isin(by_id.index)
        if absent.any():
            raise SystemExit(f"{absent.sum():,} of the cell's {len(pid):,} point_ids are not "
                             f"in the dataset (e.g. {pid[absent][:3].tolist()}); wrong --data?")
        rows = by_id.loc[pid].reset_index()
        how = "the cell's point_id"
    else:
        import zoo                       # sklearn and xgboost load only on this path
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
    bad_t = (int((rows.tile_50km.astype(str).to_numpy() != cell.tile_50km.astype(str)).sum())
             if cell.tile_50km is not None else 0)
    print(f"  OOD rows: {how}; max |dlon| {dlon:.1e}, |dlat| {dlat:.1e}, "
          f"label mismatches {bad_y}, tile mismatches {bad_t}", flush=True)
    if dlon > LONLAT_TOL or dlat > LONLAT_TOL or bad_y or bad_t:
        raise SystemExit("the dataset rows do not reproduce the cell's OOD order; aborting")
    return rows, how


def points_5070(rows: pd.DataFrame) -> tuple[np.ndarray, str]:
    """(x, y) in EPSG:5070: the dataset's geometry, else lon/lat projected."""
    import shapely
    if "geometry" in rows:
        xy = shapely.get_coordinates(shapely.from_wkb(rows.geometry.to_numpy()))
        if len(xy) == len(rows):
            return xy, "dataset geometry (EPSG:5070)"
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4269", f"EPSG:{CRS_POINTS}", always_xy=True)
    x, y = tf.transform(rows.lon.to_numpy(float), rows.lat.to_numpy(float))
    return np.column_stack([x, y]), f"lon/lat projected EPSG:4269 -> {CRS_POINTS}"


def source_row_ids(sel: pd.DataFrame, xy: np.ndarray, state: str,
                   prep_dir: Path) -> tuple[pd.Series, str]:
    """meta_src_row_id for the selected points (decision 6). sel has a RangeIndex
    aligned with xy."""
    if "meta_src_row_id" in sel:
        s = sel.meta_src_row_id.astype("Int64")
        return s, f"from the dataset ({int(s.notna().sum())} of {len(s)} set)"
    out = pd.Series(pd.NA, index=sel.index, dtype="Int64")
    idx_p = prep_dir / f"idx_{state}.parquet"
    if not idx_p.exists():
        return out, f"unavailable: no meta_src_row_id in the dataset and {idx_p} missing"

    idx = pq.read_table(idx_p, columns=["row_id", "DFIRM_ID", "FLD_AR_ID"]).to_pandas()
    has = sel.meta_fld_ar_id.notna().to_numpy()
    key = pd.DataFrame(dict(i=np.flatnonzero(has),
                            DFIRM_ID=sel.dfirm_id[has].astype(str).to_numpy(),
                            FLD_AR_ID=sel.meta_fld_ar_id[has].astype(str).to_numpy()))
    cand = key.merge(idx.astype({"DFIRM_ID": str, "FLD_AR_ID": str}),
                     on=["DFIRM_ID", "FLD_AR_ID"], how="inner")
    n_cand = cand.groupby("i").row_id.transform("size")
    one = cand[n_cand == 1]
    out.iloc[one.i.to_numpy()] = one.row_id.to_numpy()

    amb = cand[n_cand > 1]
    n_amb, n_res = int(amb.i.nunique()), 0
    if n_amb:
        import shapely
        sys.path.insert(0, str(HERE.parent))
        import sample_points as sp
        # read_geoms opens <PREP_DIR>/haz_<ST>_5070.parquet; point it at --prep-dir.
        sp.PREP_DIR = prep_dir
        rid = amb.row_id.unique()
        geoms = dict(zip(rid.tolist(), sp.read_geoms(state, rid)))
        for i, grp in amb.groupby("i"):
            pt = shapely.Point(xy[i])
            hits = [r for r in grp.row_id if geoms.get(r) is not None and geoms[r].covers(pt)]
            if len(hits) == 1:
                out.iloc[i] = hits[0]
                n_res += 1
    n_none = int(has.sum()) - int(cand.i.nunique())
    n_null = int(out.isna().sum())
    neg_null = int((out.isna() & (sel.label == 0)).sum())
    return out, (f"recovered from (dfirm_id, meta_fld_ar_id) via {idx_p.name}: "
                 f"{len(one)} points by a unique key, {n_res} of {n_amb} points with a "
                 f"repeated key by point-in-polygon, {n_none} points whose key is not in "
                 f"the index; {n_null} null ({neg_null} negatives, whose source ring this "
                 f"dataset never recorded)")


def resolve_knn(results_dir: Path, stem: str, variant: str, cli_knn: int | None) -> tuple[int, str]:
    """k from the run's records (decision 1); every source present must agree."""
    found = {}
    res_p = results_dir / f"{stem}_result.csv"
    if res_p.exists():
        r = pd.read_csv(res_p)
        if "knn" in r and r.knn.notna().any():
            found["result.csv"] = int(r.knn.iloc[0])
    meta_p = results_dir / f"{stem}_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if "knn" in meta:
            found["meta.json"] = int(meta["knn"])
    m = re.search(r"knn(\d+)", variant)
    if m:
        found["variant name"] = int(m.group(1))
    if cli_knn is not None:
        found["--knn"] = int(cli_knn)
    if len(set(found.values())) > 1:
        raise BudgetFailed(f"k disagrees across sources: {found}")
    if not found:
        return 10, "oodselect_paper.py default (nothing recorded)"
    return next(iter(found.values())), " = ".join(found)


def recorded(results_dir: Path, stem: str) -> dict:
    """What the selection run itself wrote about this budget, if anything."""
    p = results_dir / f"{stem}_result.csv"
    if not p.exists():
        return {}
    return pd.read_csv(p).iloc[0].to_dict()


def check_variant(rec: dict, variant: str, stem: str) -> str:
    """The variant the run recorded must be the one the outputs are named after
    (decision 8). Returns where the name was confirmed."""
    if pd.notna(rec.get("variant")):
        src, name = "result.csv variant", str(rec["variant"])
    elif pd.notna(rec.get("mu")) and pd.notna(rec.get("knn")):
        mu = float(rec["mu"])
        src = "result.csv mu/knn"
        name = "classic" if mu == 0 else f"spatial_mu{mu:g}_knn{int(rec['knn'])}"
    else:
        return "unverified: no result.csv records it"
    if name != variant:
        raise BudgetFailed(f"{stem}_result.csv records variant {name!r} ({src}), but the outputs "
                           f"would be named {variant!r}; pass the --results-dir and --variant of "
                           f"the run these thetas came from")
    return src


# ----------------------------------------------------------------- regions
def top_mask(theta: np.ndarray, S: int) -> np.ndarray:
    """oodselect_paper.main's mask, verbatim."""
    m = np.zeros(len(theta), bool)
    m[np.argsort(-theta, kind="stable")[:S]] = True
    return m


def components(A, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Component id per selected point (decision 3) and the sizes in id order."""
    n, lab = connected_components(A[mask][:, mask], directed=False)
    sizes = np.bincount(lab, minlength=n)
    first = np.full(n, mask.sum())
    np.minimum.at(first, lab, np.arange(len(lab)))      # lowest OOD index per component
    order = np.lexsort((first, -sizes))
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    return rank[lab], sizes[order]


def _ranked(s: pd.Series) -> list:
    """Distinct non-null values, most frequent first, ties by value."""
    vc = s.dropna().astype(str).value_counts()
    return sorted(vc.index, key=lambda v: (-vc[v], v))


def component_table(sel: pd.DataFrame, xy: np.ndarray, comp: np.ndarray,
                    sizes: np.ndarray, min_size: int) -> pd.DataFrame:
    """One row per component (decision 5 for the distance columns)."""
    poly = np.where(sel.meta_src_row_id.notna(), "r" + sel.meta_src_row_id.astype(str),
                    np.where(sel.meta_fld_ar_id.notna(),
                             sel.dfirm_id.astype(str) + "|" + sel.meta_fld_ar_id.astype(str), ""))
    rows = []
    for c in range(len(sizes)):
        m = comp == c
        s, p = sel[m], xy[m]
        dx, dy = np.ptp(p[:, 0]) / 1e3, np.ptp(p[:, 1]) / 1e3
        nn = cKDTree(p).query(p, k=2)[0][:, 1] if len(p) > 1 else np.array([np.nan])
        lc = s.landcover.dropna()
        if len(lc):
            vc = lc.round().astype(int).value_counts()
            code = sorted(vc.index, key=lambda v: (-vc[v], v))[0]
            lc_code, lc_frac = int(code), float(vc[code] / len(lc))
        else:
            lc_code, lc_frac = pd.NA, np.nan
        tiles, counties, dfirms = _ranked(s.tile_50km), _ranked(s.county_fips), _ranked(s.dfirm_id)
        pk = poly[m][(s.label == 1).to_numpy()]
        rows.append(dict(
            component=c, n_points=int(m.sum()), n_positive=int((s.label == 1).sum()),
            n_tiles=len(tiles), tiles=";".join(tiles),
            n_counties=len(counties), counties=";".join(counties),
            extent_km=float(np.hypot(dx, dy)),
            median_nn_spacing_m=float(np.median(nn)),
            dominant_landcover=lc_code,
            median_hand=float(s.hand.median()),
            median_dist_to_river_m=float(s.meta_dist_to_river_m.median()),
            ge_min_size=bool(m.sum() >= min_size),
            dominant_landcover_name=NLCD.get(lc_code, "") if lc_code is not pd.NA else "",
            dominant_landcover_frac=lc_frac,
            extent_x_km=float(dx), extent_y_km=float(dy),
            n_polygons=int(len(set(pk[pk != ""]))),
            dfirm_ids=";".join(dfirms),
            lon_center=float(s.lon.mean()), lat_center=float(s.lat.mean()),
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


def run_budget(S: int, cell, rows, xy, a, variant: str, adj: dict, paths) -> None:
    stem = f"{a.cell}_S{S}"
    th_p = a.results_dir / f"{stem}_theta.npy"
    if not th_p.exists():
        raise BudgetFailed(f"{th_p} missing")
    theta = np.load(th_p)
    if len(theta) != cell.d:
        raise BudgetFailed(f"{th_p.name} has {len(theta):,} entries, the cell {cell.d:,} OOD points")
    if not 0 < S < cell.d:
        raise BudgetFailed(f"budget {S} outside (0, d = {cell.d})")
    rec = recorded(a.results_dir, stem)
    if rec and "seed" in rec and int(rec["seed"]) != a.seed:
        raise BudgetFailed(f"{stem}_result.csv records selection seed {int(rec['seed'])}, "
                           f"--seed is {a.seed}")
    v_src = check_variant(rec, variant, stem)
    knn, k_src = resolve_knn(a.results_dir, stem, variant, a.knn)
    if knn not in adj:
        adj[knn] = graph.build_adjacency(cell.lon, cell.lat, k=knn)

    mask = top_mask(theta, S)
    comp, sizes = components(adj[knn], mask)
    sel = rows[mask].reset_index(drop=True)
    sxy = xy[mask]
    src, src_note = source_row_ids(sel, sxy, a.info["state"], a.prep_dir)
    sel["meta_src_row_id"] = src
    sel["component"] = comp.astype(np.int32)
    sel["component_size"] = sizes[comp].astype(np.int32)
    tab = component_table(sel, sxy, comp, sizes, a.min_size)

    n_cc, n_tiles = len(sizes), int(sel.tile_50km.nunique())
    n_pos = int((sel.label == 1).sum())
    print(f"\n--- S = {S}: {S:,} of {cell.d:,} OOD points ({n_pos} positive, "
          f"{S - n_pos} negative) | variant {variant} ({v_src}) | k = {knn} ({k_src})",
          flush=True)
    check = ""
    if "coh_n_cc" in rec and "coh_tiles" in rec:
        ok = int(rec["coh_n_cc"]) == n_cc and int(rec["coh_tiles"]) == n_tiles
        check = (f"; the selection run recorded {int(rec['coh_n_cc'])} components, "
                 f"{int(rec['coh_tiles'])} tiles: {'match' if ok else 'MISMATCH'}")
        if not ok:
            raise BudgetFailed(f"{n_cc} components / {n_tiles} tiles here against "
                               f"{int(rec['coh_n_cc'])} / {int(rec['coh_tiles'])} recorded: "
                               f"the graph or the mask is not the selection's")
    else:
        check = "; no recorded coh_n_cc/coh_tiles to check against"
    print(f"  {n_cc} components across {n_tiles} tiles{check}")
    big = tab[tab.ge_min_size]
    print(f"  {len(big)} components of >= {a.min_size} points hold "
          f"{int(big.n_points.sum())} points ({big.n_points.sum() / S:.1%}); sizes "
          f"{', '.join(map(str, big.n_points)) or '-'}")
    print(f"  meta_src_row_id: {src_note}")
    print_components(tab, a.min_size)

    pts = sel[POINT_COLS].sort_values("component", kind="stable").reset_index(drop=True)
    pts = pts.astype({"point_id": str, "label": np.int8, "tile_50km": str})
    pts["cell"], pts["variant"] = a.cell, variant
    pts["S"], pts["seed"] = np.int32(S), np.int32(a.seed)
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
                    help="one variant directory, e.g. results/<date>/oodselect_paper/spatial_mu3_knn10")
    ap.add_argument("--variant", help="name used in the output files (default: the "
                                      "--results-dir directory name)")
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
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    a.info = parse_tag(a.cell)
    variant = a.variant or a.results_dir.name
    out = a.out or Path("results") / date.today().isoformat() / "regions"

    todo = []
    for S in a.budgets:
        prefix = f"{a.cell}_{variant}_S{S}"
        paths = (out / f"{prefix}_points.parquet", out / f"{prefix}_components.csv")
        if all(p.exists() for p in paths) and not a.overwrite:
            print(f"{prefix}: outputs exist; skipping (--overwrite to redo)")
            continue
        todo.append((S, paths))
    if not todo:
        return 0

    t0 = time.time()
    print(f"=== regions {a.cell} | {variant} | budgets {[S for S, _ in todo]} ===", flush=True)
    cell = load_cell(a.cells_dir, a.cell)
    cell.require_spatial()
    df = load_dataset(a.data)
    print(f"  cell: {cell.n_models} models x {cell.d:,} OOD points | dataset "
          f"{a.data.name}: {len(df):,} rows", flush=True)
    rows, _ = ood_rows(cell, df, a.info)
    xy, xy_src = points_5070(rows)
    print(f"  distances from {xy_src}", flush=True)

    adj, failed = {}, []
    for S, paths in todo:
        try:
            run_budget(S, cell, rows, xy, a, variant, adj, paths)
        except BudgetFailed as e:
            print(f"\n--- S = {S}: FAILED, nothing written: {e}", flush=True)
            failed.append(S)
    print(f"\ndone in {time.time() - t0:.0f}s: {len(todo) - len(failed)} budget(s) written "
          f"to {out}" + (f"; FAILED {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
