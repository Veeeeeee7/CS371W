#!/usr/bin/env python3
"""Terrain-matched controls for the region points (S1 of docs/TASK-O5-REGIONS.md).

Each land point of a region gets one control: an unselected Zone A test point of the same seed,
50 km tile and label with the nearest terrain (Euclidean distance on the six TERRAIN columns,
each divided by its SD over the seed's land Zone A test points). A region-vs-control difference
in any other variable is then not a terrain difference. Results and the caliper decision:
docs/O5-REGIONS-REPORT-2026-10-17.md, "Shared pieces S1 and S2".

Reads   the cells (<cells-dir>), the spatial and classic S = 500 theta files (<sel-root>),
        the region points (<regions-dir>) and the point table (<data>).
Writes  <out-dir>/<prefix>_controls.csv            one row per region land point
        <out-dir>/<prefix>_match_summary.csv       one row per region
        <out-dir>/<prefix>_balance.csv             SMD before / after, per region and pooled
        <out-dir>/<prefix>_caliper_sensitivity.csv when the Stop fires, or --sensitivity always

    .venv/bin/python aotil/match_controls.py --out-dir results/2026-10-17/o5 --caliper 1.0 --balance-rule seed_sd
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import load_cell  # noqa: E402

TERRAIN = ("slope", "tri_300", "hand", "curvature", "tpi_300", "tpi_broad")
STRATA = ("tile_50km", "label")              # matched exactly
WATER = 11                                   # NLCD class 11, open water: terrain is not meaningful
SEL_VARIANTS = ("spatial_mu3_knn10", "classic")   # a control is in neither selection
REGION_VARIANT = "spatial_mu3_knn10"
TABLE_COLUMNS = ["point_id", "domain", "label", "landcover", "lon", "lat", "tile_50km", *TERRAIN]

CELLS_DIR = "data/interim/aotil/cells_w7"
CELL_PREFIX = "FL_band_AE"
DATA = "data/processed/dataset_FL.parquet"
SEL_ROOT = "results/2026-10-16/oodselect_paper"
REGIONS_DIR = "results/2026-10-16/regions"
BUDGET = 500                                 # S of the OODSelect selections, in points
MIN_SIZE = 20                                # points; smaller components are not regions

SENS_CALIPERS = (0.25, 0.5, 0.75, 1.0, None)  # None = no caliper
STOP_SHARE = 0.8                             # the Stop: a region matches less than this share
STOP_SMD = 0.1                               # ... or pooled |imbalance| after matching >= this
POOLED_SCOPES = ("all", "positive-dominated", "negative-dominated", "label=1", "label=0")
CONTROLS_COLUMNS = ["seed", "component", "point_id", "control_point_id", "distance", "tile",
                    "label", "type", "status", "n_pool_stratum", "n_candidates"]


# ------------------------------------------------------------------ matching
def match(query: pd.DataFrame, pool: pd.DataFrame, sd, terrain=TERRAIN,
          caliper: float | None = 0.25, strata=STRATA, id_col: str = "point_id") -> pd.DataFrame:
    """Match each query row to at most one pool row of its stratum, without replacement.

    query and pool hold id_col, the strata columns and the raw terrain columns; their ids must be
    disjoint. sd is the standardizing SD per terrain column (a Series or dict by name, or a
    sequence in the order of `terrain`). caliper is the largest allowed standardized distance,
    or None. Returns one row per query row, in query order: point_id, control_point_id (None
    when unmatched), distance (NaN when unmatched), status, n_pool_stratum (complete-terrain pool
    points in the stratum) and n_candidates (those within the caliper). status is matched,
    missing_terrain, empty_stratum (no pool point in the stratum), no_candidate (none within the
    caliper) or lost_to_competition (the optimal assignment gave them to other queries).
    """
    terrain = list(terrain)
    if isinstance(sd, (pd.Series, dict)):
        sd = pd.Series(sd, index=terrain)
    sd = np.asarray(sd, float)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(f"standardizing SDs must be finite and > 0, got {sd}")
    query_ids = query[id_col].to_numpy()
    if set(query_ids) & set(pool[id_col].to_numpy()):
        raise ValueError("query and pool share point ids; the caller must exclude them")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query ids")

    n = len(query)
    out = pd.DataFrame({"point_id": query_ids,
                        "control_point_id": pd.Series([None] * n, dtype=object),
                        "distance": np.full(n, np.nan),
                        "status": "",
                        "n_pool_stratum": 0,
                        "n_candidates": 0})
    # A point with any terrain value missing has no distance: an unmatched query, never a control.
    Xq = query[terrain].to_numpy(float) / sd
    query_complete = np.isfinite(Xq).all(axis=1)
    out.loc[~query_complete, "status"] = "missing_terrain"
    Xp = pool[terrain].to_numpy(float) / sd
    pool_complete = np.isfinite(Xp).all(axis=1)
    pool = pool.loc[pool_complete]
    Xp = Xp[pool_complete]
    pool_ids = pool[id_col].to_numpy()

    pool_strata = _groups(pool, strata)
    for key, q_rows in _groups(query, strata).items():
        q_rows = q_rows[query_complete[q_rows]]
        if not len(q_rows):
            continue
        p_rows = pool_strata.get(key, np.array([], dtype=int))
        out.loc[q_rows, "n_pool_stratum"] = len(p_rows)
        if not len(p_rows):
            out.loc[q_rows, "status"] = "empty_stratum"
            continue
        D = cdist(Xq[q_rows], Xp[p_rows])
        if caliper is None:
            allowed = np.ones_like(D, bool)
        else:
            allowed = D <= caliper
        out.loc[q_rows, "n_candidates"] = allowed.sum(axis=1)
        # Pool points no query can use stay out of the assignment (smaller matrix, same optimum).
        usable = allowed.any(axis=0)
        if not usable.any():
            out.loc[q_rows, "status"] = "no_candidate"
            continue
        D = D[:, usable]
        allowed = allowed[:, usable]
        p_rows = p_rows[usable]

        rows, cols = optimal_pairs(D, allowed)
        matched = q_rows[rows]
        out.loc[matched, "control_point_id"] = pool_ids[p_rows[cols]]
        out.loc[matched, "distance"] = D[rows, cols]
        out.loc[matched, "status"] = "matched"
        is_matched = np.zeros(len(q_rows), bool)
        is_matched[rows] = True
        has_candidate = allowed.any(axis=1)
        out.loc[q_rows[~is_matched & has_candidate], "status"] = "lost_to_competition"
        out.loc[q_rows[~is_matched & ~has_candidate], "status"] = "no_candidate"
    return out


def optimal_pairs(D: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the allowed pairs of the assignment that matches the most rows
    and, among those, has the least total distance (scipy's Hungarian algorithm)."""
    # A disallowed pair costs more than all allowed distances together, so the optimum first
    # maximizes the number of allowed pairs. Unlike greedy nearest-neighbour matching it does not
    # lose a point whose only candidate an earlier point took.
    big = 10.0 * (float(D[allowed].sum()) + 1.0) * (len(D) + 1)
    rows, cols = linear_sum_assignment(np.where(allowed, D, big))
    keep = allowed[rows, cols]
    return rows[keep], cols[keep]


def _groups(df: pd.DataFrame, strata) -> dict:
    """{stratum key tuple: positional row indices}, keys sorted."""
    if not len(df):
        return {}
    indices = df.reset_index(drop=True).groupby(list(strata), sort=True).indices
    groups = {}
    for key, rows in indices.items():
        if not isinstance(key, tuple):
            key = (key,)
        groups[key] = np.asarray(rows)
    return groups


def smd(a, b, wb=None) -> float:
    """Standardized mean difference (mean_a - mean_b) / sqrt((var_a + var_b) / 2), with sample
    variances and missing values skipped. wb optionally gives b's rows frequency weights."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    if wb is None:
        wb = np.ones(len(b))
    wb = np.asarray(wb, float)
    keep = np.isfinite(b) & (wb > 0)
    b = b[keep]
    wb = wb[keep]
    if len(a) < 2 or wb.sum() <= 1:
        return np.nan
    mean_b = np.average(b, weights=wb)
    var_b = np.sum(wb * (b - mean_b) ** 2) / (wb.sum() - 1)
    var_a = a.var(ddof=1)
    den = np.sqrt((var_a + var_b) / 2)
    if den == 0:
        return 0.0 if a.mean() == mean_b else np.nan
    return float((a.mean() - mean_b) / den)


# ------------------------------------------------------------- seed context
@dataclass
class SeedContext:
    """One seed's OOD half (its Zone A test points), ready for matching.

    pts      one row per OOD point in cell order: the point-table columns, land, sel_<variant>
             per selection, selected (in any of them), complete (all terrain present),
             component (region id, -1 outside every region) and z_<terrain> (terrain / sd).
    sd       TERRAIN SDs over the seed's land OOD points.
    regions  the region points (component_size >= min_size), joined to pts.
    """
    seed: int
    tag: str
    pts: pd.DataFrame
    sd: pd.Series
    regions: pd.DataFrame
    sel_variants: tuple = SEL_VARIANTS
    _pos: dict = field(default_factory=dict, repr=False)

    def index_of(self, ids) -> np.ndarray:
        """Row positions in pts of the given point ids."""
        if not self._pos:
            for i, point_id in enumerate(self.pts.point_id.to_numpy()):
                self._pos[point_id] = i
        return np.array([self._pos[p] for p in ids], int)

    def pool_mask(self, exclude_ids=()) -> np.ndarray:
        """The control pool: land, unselected OOD points, minus exclude_ids (terrain may be missing)."""
        mask = (self.pts.land & ~self.pts.selected).to_numpy().copy()
        exclude_ids = list(exclude_ids)
        if exclude_ids:
            mask[self.index_of(exclude_ids)] = False
        return mask


def top_budget_mask(theta: np.ndarray, budget: int) -> np.ndarray:
    """The selection: the `budget` largest theta values, ties to the earlier point."""
    mask = np.zeros(len(theta), bool)
    mask[np.argsort(-theta, kind="stable")[:budget]] = True
    return mask


def load_seed_context(seed: int, cells_dir=CELLS_DIR, cell_prefix=CELL_PREFIX, data=DATA,
                      sel_root=SEL_ROOT, regions_dir=REGIONS_DIR, sel_variants=SEL_VARIANTS,
                      region_variant=REGION_VARIANT, budget: int = BUDGET, min_size: int = MIN_SIZE,
                      table: pd.DataFrame | None = None) -> SeedContext:
    """Join one seed's OOD half to the point table and the selections, checking the join.
    `table` is the point table (TABLE_COLUMNS) when the caller has already read it."""
    tag = f"{cell_prefix}_s{seed}"
    cell = load_cell(cells_dir, tag)
    ood_ids = np.asarray(cell.point_id).astype(str)
    if table is None:
        table = pd.read_parquet(data, columns=TABLE_COLUMNS)
    by_id = table.set_index("point_id")
    missing = set(ood_ids) - set(by_id.index)
    if missing:
        raise SystemExit(f"{tag}: {len(missing)} OOD point ids not in {data}")
    pts = by_id.loc[ood_ids].reset_index()
    if not (pts.domain.to_numpy() == "A").all():
        raise SystemExit(f"{tag}: OOD half is not all Zone A")
    if not (pts.label.to_numpy() == np.asarray(cell.y_ood)).all():
        raise SystemExit(f"{tag}: dataset labels differ from the cell's y_ood")
    if np.abs(pts.lon.to_numpy() - np.asarray(cell.lon)).max() > 1e-9:
        raise SystemExit(f"{tag}: dataset lon differs from the cell's")
    if not (pts.tile_50km.to_numpy() == np.asarray(cell.tile_50km).astype(str)).all():
        raise SystemExit(f"{tag}: dataset tile_50km differs from the cell's")
    pts["land"] = pts.landcover.notna().to_numpy() & (pts.landcover.to_numpy() != WATER)

    n_ood = len(pts)
    selected = np.zeros(n_ood, bool)
    for variant in sel_variants:
        theta = np.load(Path(sel_root) / variant / f"{tag}_S{budget}_theta.npy")
        if theta.shape != (n_ood,):
            raise SystemExit(f"{variant} theta for {tag} has shape {theta.shape}, "
                             f"cell has {n_ood} OOD points")
        mask = top_budget_mask(theta, budget)
        pts[f"sel_{variant}"] = mask
        selected |= mask
    pts["selected"] = selected
    pts["complete"] = np.isfinite(pts[list(TERRAIN)].to_numpy(float)).all(axis=1)

    regions = pd.read_parquet(Path(regions_dir) / f"{tag}_{region_variant}_S{budget}_points.parquet")
    regions = regions[regions.component_size >= min_size].reset_index(drop=True)
    position = {point_id: i for i, point_id in enumerate(ood_ids)}
    region_rows = np.array([position[p] for p in regions.point_id.astype(str)], int)
    region_mask = f"sel_{region_variant}"
    if region_mask in pts and not pts[region_mask].to_numpy()[region_rows].all():
        raise SystemExit(f"{tag}: region points outside the {region_variant} S={budget} mask")
    pts["component"] = -1
    pts.loc[region_rows, "component"] = regions.component.to_numpy()
    regions = regions[["point_id", "component", "component_size"]].merge(
        pts[["point_id", "label", "tile_50km", "landcover", "land", "complete", *TERRAIN]],
        on="point_id", how="left")

    # Over all land OOD points, selected or not, so the unit does not move when a caller
    # shrinks the pool (region_null removes each patch from it).
    sd = pts.loc[pts.land, list(TERRAIN)].std(ddof=1, skipna=True)
    for v in TERRAIN:
        pts[f"z_{v}"] = pts[v] / sd[v]
    return SeedContext(seed, tag, pts, sd, regions, tuple(sel_variants))


def describe_context(ctx: SeedContext, budget: int) -> str:
    """One log line on a seed's points, pool, regions and SDs."""
    pts = ctx.pts
    pool = ctx.pool_mask()
    pool_incomplete = int((pool & ~pts.complete.to_numpy()).sum())
    sds = ", ".join(f"{name}={value:.3g}" for name, value in ctx.sd.items())
    return (f"seed {ctx.seed}: {len(pts)} OOD points, {int(pts.land.sum())} land, "
            f"{int(pts.selected.sum())} in spatial or classic S={budget}, pool {int(pool.sum())} "
            f"(of which {pool_incomplete} with missing terrain); "
            f"{ctx.regions.component.nunique()} regions ({len(ctx.regions)} points); SD {sds}")


def load_contexts(seeds, cells_dir, cell_prefix, data, sel_root, regions_dir, budget, min_size,
                  verbose=True) -> dict[int, SeedContext]:
    """load_seed_context for each seed, reading the point table once."""
    table = pd.read_parquet(data, columns=TABLE_COLUMNS)
    ctxs = {}
    for seed in seeds:
        ctxs[seed] = load_seed_context(seed, cells_dir, cell_prefix, data, sel_root, regions_dir,
                                       budget=budget, min_size=min_size, table=table)
    if verbose:
        for ctx in ctxs.values():
            print(describe_context(ctx, budget), flush=True)
    return ctxs


def match_points(ctx: SeedContext, query_ids, exclude_ids=(), caliper: float | None = 0.25,
                 terrain=TERRAIN, land_only: bool = True) -> pd.DataFrame:
    """match() for OOD points of one seed against its pool, minus exclude_ids and the queries.
    Non-land queries are dropped first when land_only. Adds the seed, tile and label columns."""
    query = ctx.pts.iloc[ctx.index_of(list(query_ids))]
    if land_only:
        query = query[query.land]
    pool = ctx.pts.loc[ctx.pool_mask(list(exclude_ids) + list(query.point_id))]
    matches = match(query.reset_index(drop=True), pool, ctx.sd, terrain=terrain, caliper=caliper)
    matches.insert(0, "seed", ctx.seed)
    matches["tile"] = query.tile_50km.to_numpy()
    matches["label"] = query.label.to_numpy()
    return matches


# ------------------------------------------------------------------ balance
def strata_weights(region: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    """Per pool point, its (seed, tile, label) stratum's region-point count shared among the
    stratum's pool points: weighted, the pool shows the imbalance that matching on tile and
    label alone would leave."""
    key_cols = ["seed", "tile_50km", "label"]
    region_counts = region.groupby(key_cols).size().rename("w").reset_index()
    pool_counts = pool.merge(region_counts, on=key_cols, how="left")["w"]
    counts = pool_counts.fillna(0).to_numpy()
    stratum_sizes = pool.groupby(key_cols)["point_id"].transform("size").to_numpy()
    return counts / np.maximum(stratum_sizes, 1)


def nan_mean(values) -> float:
    return float(np.nanmean(values)) if len(values) else np.nan


def n_finite(values: pd.Series) -> int:
    return int(np.isfinite(values.to_numpy(float)).sum())


def balance_rows(region: pd.DataFrame, pool: pd.DataFrame, matched: pd.DataFrame,
                 controls: pd.DataFrame, terrain=TERRAIN) -> list[dict]:
    """Per terrain variable, the SMD before matching (region vs the whole pool, and vs the
    strata-weighted pool) and after (matched points vs their row-aligned controls).

    All frames carry seed, tile_50km, label and the raw terrain columns; matched and controls
    also carry the z_ columns of load_seed_context.
    """
    weights = strata_weights(region, pool)
    rows = []
    for v in terrain:
        z = f"z_{v}"
        has_z = z in matched
        # The same mean difference in the matching units (one unit = the seed's land SD). The SMD
        # divides by the matched sets' own SD instead, which is small where regions are flat.
        if len(matched) > 0 and has_z:
            diff_after = float(np.mean(matched[z]) - np.mean(controls[z]))
        else:
            diff_after = np.nan
        if len(matched) > 1 and has_z:
            spread = float(np.sqrt((np.var(matched[z], ddof=1) + np.var(controls[z], ddof=1)) / 2))
        else:
            spread = np.nan
        rows.append(dict(variable=v,
                         smd_before=smd(region[v], pool[v]),
                         smd_before_strata=smd(region[v], pool[v], wb=weights),
                         smd_after=smd(matched[v], controls[v]),
                         diff_after_seed_sd=diff_after,
                         sd_matched_seed_sd=spread,
                         mean_region=nan_mean(region[v]),
                         mean_pool=nan_mean(pool[v]),
                         mean_matched=nan_mean(matched[v]),
                         mean_controls=nan_mean(controls[v]),
                         n_region=n_finite(region[v]),
                         n_pool=n_finite(pool[v]),
                         n_pairs=len(matched)))
    return rows


def region_type(pos_share_land: float) -> str:
    """Positive-dominated iff at least half of the region's land points are positives."""
    if not np.isfinite(pos_share_land):
        return "no land points"
    return "positive-dominated" if pos_share_land >= 0.5 else "negative-dominated"


def _scope(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """The rows of df (with `type` and `label` columns) in one of POOLED_SCOPES."""
    if name == "all":
        return df
    if name.startswith("label="):
        return df[df.label == int(name.split("=")[1])]
    return df[df.type == name]


def _share(matched: np.ndarray, subset: np.ndarray) -> float:
    return float(matched[subset].mean()) if subset.any() else np.nan


def region_summary(seed: int, component, points: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """One row of the match summary: a region's counts, unmatched points by status, matched
    shares and distances. points are all its points, matches those of its land points."""
    land = points[points.land]
    n_land = len(land)
    positives_land = int(land.label.sum())
    pos_share = positives_land / n_land if n_land else np.nan
    matched = (matches.status == "matched").to_numpy()
    status_counts = matches.status.value_counts()
    labels = matches.label.to_numpy()
    matched_share = float(matched.mean()) if n_land else np.nan
    distances = matches.distance[matched]
    if len(matches):
        with_terrain = matches.status != "missing_terrain"
        median_candidates = float(matches.n_candidates[with_terrain].median())
        # max per stratum: a stratum's first query may lack terrain and carry n_pool_stratum 0
        pool_per_stratum = matches.groupby(["tile", "label"]).n_pool_stratum.max()
        n_pool_strata = int(pool_per_stratum.sum())
    else:
        median_candidates = np.nan
        n_pool_strata = 0
    return dict(seed=seed, component=int(component), type=region_type(pos_share),
                n=len(points), positives=int(points.label.sum()),
                n_land=n_land, positives_land=positives_land, dropped_water=len(points) - n_land,
                pos_share_land=pos_share, tiles=" ".join(sorted(set(points.tile_50km))),
                n_missing_terrain=int(status_counts.get("missing_terrain", 0)),
                n_no_candidate=int(status_counts.get("no_candidate", 0)),
                n_lost_to_competition=int(status_counts.get("lost_to_competition", 0)),
                n_empty_stratum=int(status_counts.get("empty_stratum", 0)),
                n_matched=int(matched.sum()),
                matched_share=matched_share,
                matched_share_pos=_share(matched, labels == 1),
                matched_share_neg=_share(matched, labels == 0),
                median_distance=float(distances.median()) if matched.any() else np.nan,
                max_distance=float(distances.max()) if matched.any() else np.nan,
                median_candidates=median_candidates,
                n_pool_strata=n_pool_strata,
                stop_share=bool(n_land > 0 and matched_share < STOP_SHARE))


def pooled_share_row(name: str, K: pd.DataFrame, S: pd.DataFrame) -> dict:
    """Matched share of the region land points in one pooled scope."""
    k = _scope(K, name)
    is_matched = k.status == "matched"
    row = dict(scope=name, n_land=len(k), n_matched=int(is_matched.sum()),
               matched_share=float(is_matched.mean()) if len(k) else np.nan)
    if name.startswith("label="):
        # a region is not of one label: no region counts
        row["n_regions"] = np.nan
        row["n_regions_below_0p8"] = np.nan
    else:
        regions = _scope(S, name)
        row["n_regions"] = int(len(regions))
        row["n_regions_below_0p8"] = int(regions.stop_share.sum())
    return row


def run_all(ctxs: dict[int, SeedContext], caliper: float | None):
    """S1 over every seed. All regions of a seed are matched together, so two regions in one
    tile never share a control.

    Returns (K, S, B, P): K one row per region land point (its match), S one row per region,
    B the balance table (per region, then pooled), P the pooled matched shares. Pooled rows put
    the points of all seeds together (a point in two seeds' pools counts twice).
    """
    controls, summary, balance = [], [], []
    region_parts, matched_parts, control_parts, pools = [], [], [], []
    for seed, ctx in ctxs.items():
        reg = ctx.regions
        matches = match_points(ctx, reg.loc[reg.land, "point_id"].tolist(), caliper=caliper)
        matches = matches.merge(reg[["point_id", "component"]], on="point_id", how="left")
        by_id = ctx.pts.set_index("point_id")
        pool = ctx.pts.loc[ctx.pool_mask()].assign(seed=seed)
        pools.append(pool)
        types = {}
        for component, points in reg.groupby("component"):
            region_matches = matches[matches.component == component]
            row = region_summary(seed, component, points, region_matches)
            summary.append(row)
            rtype = row["type"]
            types[component] = rtype
            ok = (region_matches.status == "matched").to_numpy()
            region = points[points.land].assign(seed=seed, type=rtype)
            matched = by_id.loc[region_matches.point_id[ok]].reset_index()
            matched = matched.assign(seed=seed, component=component, type=rtype)
            matched_controls = by_id.loc[region_matches.control_point_id[ok]].reset_index()
            matched_controls = matched_controls.assign(seed=seed, component=component, type=rtype)
            region_parts.append(region)
            matched_parts.append(matched)
            control_parts.append(matched_controls)
            for r in balance_rows(region, pool, matched, matched_controls):
                balance.append(dict(scope="region", seed=seed, component=int(component),
                                    type=rtype, **r))
        controls.append(matches.assign(type=matches.component.map(types)))

    S = pd.DataFrame(summary)
    K = pd.concat(controls, ignore_index=True)
    regions_all = pd.concat(region_parts, ignore_index=True)
    matched_all = pd.concat(matched_parts, ignore_index=True)
    controls_all = pd.concat(control_parts, ignore_index=True)
    pool_all = pd.concat(pools, ignore_index=True)
    pooled = []
    for name in POOLED_SCOPES:
        # A region-type scope is compared with the whole pool, which has no region type.
        if name.startswith("label="):
            pool = _scope(pool_all, name)
        else:
            pool = pool_all
        for r in balance_rows(_scope(regions_all, name), pool, _scope(matched_all, name),
                              _scope(controls_all, name)):
            balance.append(dict(scope="pooled", seed=-1, component=-1, type=name, **r))
        pooled.append(pooled_share_row(name, K, S))
    return K, S, pd.DataFrame(balance), pd.DataFrame(pooled)


def stop_check(S: pd.DataFrame, B: pd.DataFrame, rule: str = "smd") -> tuple[bool, list[str]]:
    """The S1 Stop: a region under STOP_SHARE matched, or pooled imbalance >= STOP_SMD.

    rule "smd" (the task's) measures imbalance as the pooled |SMD after|, which divides by the
    matched sets' own spread. rule "seed_sd" (the O5 run's, docs/O5-REGIONS-REPORT-2026-10-17.md,
    "Decision and rerun at caliper 1.0") uses the pooled |mean difference after| in seed-SD
    units, the units of the caliper. Per-region values over the threshold are printed as notes.
    """
    reasons = []
    for _, r in S[S.stop_share].iterrows():
        reasons.append(f"seed {r.seed} component {r.component} ({r.type}): matched share "
                       f"{r.matched_share:.3f} < {STOP_SHARE} ({r.n_matched}/{r.n_land} land points)")
    if rule == "smd":
        col, measure = "smd_after", "SMD after"
    else:
        col, measure = "diff_after_seed_sd", "mean difference after (seed-SD units)"
    for _, r in B[(B.scope == "pooled") & (B.type == "all")].iterrows():
        if np.isfinite(r[col]) and abs(r[col]) >= STOP_SMD:
            reasons.append(f"pooled |{measure}| of {r.variable} = {abs(r[col]):.3f} >= {STOP_SMD}")
    for _, r in B[B.scope == "region"].iterrows():
        if np.isfinite(r[col]) and abs(r[col]) >= STOP_SMD:
            print(f"  note (not a Stop): seed {r.seed} component {r.component} |{measure}| of "
                  f"{r.variable} = {abs(r[col]):.3f} on {int(r.n_pairs)} pairs")
    return bool(reasons), reasons


def smd_after_columns(balance: pd.DataFrame) -> dict:
    """smd_after_<variable> per row of balance, and their largest absolute value."""
    out = {}
    for variable, value in zip(balance.variable, balance.smd_after):
        out[f"smd_after_{variable}"] = value
    if balance.smd_after.notna().any():
        out["max_abs_smd_after"] = float(np.nanmax(np.abs(balance.smd_after)))
    else:
        out["max_abs_smd_after"] = np.nan
    return out


def sensitivity(ctxs: dict[int, SeedContext], calipers=SENS_CALIPERS) -> pd.DataFrame:
    """Matched share and SMD after matching, per region and pooled, at each caliper."""
    rows = []
    for caliper in calipers:
        _, S, B, P = run_all(ctxs, caliper)
        caliper_label = "none" if caliper is None else caliper
        for _, r in S.iterrows():
            region_balance = B[(B.scope == "region") & (B.seed == r.seed)
                               & (B.component == r.component)]
            row = dict(caliper=caliper_label, scope="region", seed=r.seed, component=r.component,
                       type=r.type, n_land=r.n_land, n_matched=r.n_matched,
                       matched_share=r.matched_share, median_distance=r.median_distance,
                       max_distance=r.max_distance, n_regions_below_0p8=np.nan)
            row.update(smd_after_columns(region_balance))
            rows.append(row)
        for _, p in P.iterrows():
            pooled_balance = B[(B.scope == "pooled") & (B.type == p.scope)]
            row = dict(caliper=caliper_label, scope="pooled", seed=-1, component=-1, type=p.scope,
                       n_land=p.n_land, n_matched=p.n_matched, matched_share=p.matched_share,
                       median_distance=np.nan, max_distance=np.nan,
                       n_regions_below_0p8=p.n_regions_below_0p8)
            row.update(smd_after_columns(pooled_balance))
            rows.append(row)
    return pd.DataFrame(rows)


def parse_caliper(x: str):
    """A caliper argument: a number, or none / inf / off for no caliper (None)."""
    return None if str(x).lower() in ("none", "inf", "off") else float(x)


# ------------------------------------------------------------------ report
def print_report(caliper_text: str, S: pd.DataFrame, P: pd.DataFrame, B: pd.DataFrame) -> None:
    region_cols = ["seed", "component", "type", "n", "n_land", "dropped_water", "n_missing_terrain",
                   "n_no_candidate", "n_lost_to_competition", "n_matched", "matched_share",
                   "median_distance", "median_candidates"]
    balance_cols = ["type", "variable", "smd_before", "smd_before_strata", "smd_after",
                    "diff_after_seed_sd", "sd_matched_seed_sd", "n_pairs"]
    pooled_balance = B[B.scope == "pooled"][balance_cols]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(f"\ncaliper {caliper_text}: per region")
        print(S[region_cols].round(3).to_string(index=False))
        print("\npooled matched share")
        print(P.round(3).to_string(index=False))
        print("\npooled balance (SMD)")
        print(pooled_balance.round(3).to_string(index=False))


def print_sensitivity(table: pd.DataFrame) -> None:
    pooled_cols = ["caliper", "type", "n_land", "n_matched", "matched_share", "n_regions_below_0p8",
                   *[f"smd_after_{v}" for v in TERRAIN], "max_abs_smd_after"]
    regions = table[table.scope == "region"].copy()
    regions["caliper"] = regions.caliper.astype(str)
    share_by_caliper = regions.pivot(index=["seed", "component", "type"], columns="caliper",
                                     values="matched_share")
    caliper_order = [str(c) for c in table.caliper.drop_duplicates()]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\ncaliper sensitivity, pooled")
        print(table[table.scope == "pooled"][pooled_cols].round(3).to_string(index=False))
        print("\nmatched share per region by caliper")
        print(share_by_caliper[caliper_order].round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=CELLS_DIR, type=Path)
    ap.add_argument("--cell-prefix", default=CELL_PREFIX)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default=DATA, type=Path)
    ap.add_argument("--sel-root", default=SEL_ROOT, type=Path)
    ap.add_argument("--regions-dir", default=REGIONS_DIR, type=Path)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--min-size", type=int, default=MIN_SIZE)
    ap.add_argument("--caliper", default="0.25", help="standardized units, or 'none'")
    ap.add_argument("--sensitivity", choices=("auto", "always", "never"), default="auto",
                    help="write the caliper-sensitivity file when the Stop fires (auto), always, or never")
    ap.add_argument("--sens-calipers", nargs="+",
                    default=["none" if c is None else str(c) for c in SENS_CALIPERS],
                    help="calipers of the sensitivity file (default: 0.25 0.5 0.75 1.0 none)")
    ap.add_argument("--balance-rule", choices=("smd", "seed_sd"), default="smd",
                    help="the Stop's balance measure: smd (the task's, default) or seed_sd "
                         "(mean difference in seed-SD units; see stop_check)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", default="S1")
    args = ap.parse_args()

    start = time.time()
    ctxs = load_contexts(args.seeds, args.cells_dir, args.cell_prefix, args.data, args.sel_root,
                         args.regions_dir, args.budget, args.min_size)
    K, S, B, P = run_all(ctxs, parse_caliper(args.caliper))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.prefix
    K[CONTROLS_COLUMNS].to_csv(f"{prefix}_controls.csv", index=False)
    S.to_csv(f"{prefix}_match_summary.csv", index=False)
    B.to_csv(f"{prefix}_balance.csv", index=False)
    print_report(args.caliper, S, P, B)

    stop, reasons = stop_check(S, B, args.balance_rule)
    print(f"\nSTOP {'FIRES' if stop else 'does not fire'}")
    for reason in reasons:
        print(f"  - {reason}")
    if args.sensitivity == "always" or (args.sensitivity == "auto" and stop):
        table = sensitivity(ctxs, [parse_caliper(c) for c in args.sens_calipers])
        table.to_csv(f"{prefix}_caliper_sensitivity.csv", index=False)
        print_sensitivity(table)
        print(f"wrote {prefix}_caliper_sensitivity.csv")
    print(f"wrote {prefix}_controls.csv, {prefix}_match_summary.csv, {prefix}_balance.csv "
          f"({time.time() - start:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
