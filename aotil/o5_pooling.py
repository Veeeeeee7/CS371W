#!/usr/bin/env python3
"""
Part F of docs/TASK-O5-REGIONS.md: does adding the region labels to training cost accuracy
on detailed (Zone AE) studies more than adding as many ordinary Zone A labels?

Populations per seed, 300 models each with the same specs (zoo.py --extra-train): base
(cells_w7), +region (the seed's region land points that have a terrain-matched control),
+matched (those controls: same count, labels and tiles), and both with every added row
repeated 10 times (_x10). All are scored on the same land test points (never the added ones):
    AE_far   Zone AE test points >= 5 km from every added point
    A_far    Zone A test points >= 5 km from every added point and outside every region
Primary: the per-model Zone AE AUC difference on AE_far, +region minus +matched, pooled over
the seeds, with a test-tile bootstrap. Predicted: negative.

Reads   --root/<population>/FL_band_AE_s<seed>.npz and _models.csv, --lists, --regions-dir,
        --dataset.  Writes --out: one row per (seed or "pooled", eval set, contrast).

    python aotil/o5_pooling.py --out results/2026-10-17/o5/F_pooling.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from o1_shift import rebuild_p, score_groups, weighted_aucs  # noqa: E402

# population -> its folder under --root
CONDS = {"base": "cells_w7", "region": "cells_w7_pool_region", "matched": "cells_w7_pool_matched",
         "region_x10": "cells_w7_pool_region_x10", "matched_x10": "cells_w7_pool_matched_x10"}
# (a, b): per-model AUC of population a minus that of b
CONTRASTS = [("region", "matched"), ("region_x10", "matched_x10"), ("region", "base"),
             ("matched", "base"), ("region_x10", "base"), ("matched_x10", "base")]
PRIMARY = ("AE_far", "region", "matched")
EVALS = ("AE_far", "A_far")
FAR_M = 5000.0             # straight-line distance in EPSG:5070 metres
SMALL = 0.002              # the brief asks whether both primaries are within +-0.002
WATER = 11                 # NLCD open water
MIN_REGION_POINTS = 20     # smaller connected components are not regions


def read_list(p: Path) -> list[str]:
    """The point ids of an F list file, without its header line."""
    ids = []
    for line in p.read_text().splitlines():
        point_id = line.strip()
        if point_id and point_id != "point_id":
            ids.append(point_id)
    return ids


def pflood(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p(flood) per model and point from the true-class scores S, rebuilt as O1 rebuilds it
    (o1_shift.rebuild_p). Not rounded, because rounding merges near-saturated p into ties and
    moves the AUCs."""
    return rebuild_p(S, y)


def same_specs(models: pd.DataFrame, base_models: pd.DataFrame) -> bool:
    """Same model family and seed, model by model."""
    return ((models.kind.to_numpy() == base_models.kind.to_numpy()).all()
            and (models.seed.to_numpy() == base_models.seed.to_numpy()).all())


def check_points_and_specs(z: dict, models: dict, conds: dict, tag: str) -> None:
    """Every population must score the same points with the same models as base."""
    base = z["base"]
    for c in conds:
        same_points = (np.array_equal(z[c]["point_id"], base["point_id"])
                       and np.array_equal(z[c]["point_id_id"], base["point_id_id"]))
        if not same_points or not same_specs(models[c], models["base"]):
            raise SystemExit(f"{conds[c]}/{tag}: points or model specs differ from cells_w7")


def check_added_rows(z: dict, conds: dict, tag: str, lists: Path, s: int,
                     reg_ids: list, mat_ids: list) -> None:
    """Every pooled population must store its F list as the added rows, repeated 10 times
    for the _x10 populations."""
    expected_ids = {"region": reg_ids, "matched": mat_ids,
                    "region_x10": reg_ids, "matched_x10": mat_ids}
    for c, ids in expected_ids.items():
        if c not in z:
            continue
        stored = [str(x) for x in z[c]["extra_train_point_id"]]
        repeat = int(z[c]["extra_train_repeat"])
        expected_repeat = 10 if c.endswith("x10") else 1
        if stored != ids or repeat != expected_repeat:
            raise SystemExit(f"{conds[c]}/{tag}: stored extra rows differ from {lists}/s{s}_*.txt")


def far_from_added(tree: cKDTree, xy: pd.DataFrame, pids: np.ndarray) -> np.ndarray:
    """True where a point is at least FAR_M from its nearest added point."""
    distance = tree.query(xy.loc[pids, ["x", "y"]].to_numpy())[0]
    return distance >= FAR_M


def seed_data(s: int, root: Path, lists: Path, regions_dir: Path, xy: pd.DataFrame, conds,
              land_only: bool = False) -> dict:
    """One seed's evaluation sets, with every population's score groups on them.

    Checks the populations against base and the F lists first (SystemExit on a mismatch).
    land_only drops open-water points (xy then needs a landcover column). Returns the tag,
    the list sizes, and per eval set its tiles, counts and score groups per population."""
    tag = f"FL_band_AE_s{s}"
    z = {}
    for c, folder in conds.items():
        z[c] = np.load(root / folder / f"{tag}.npz", allow_pickle=True)
    models = {}
    for c, folder in conds.items():
        models[c] = pd.read_csv(root / folder / f"{tag}_models.csv")
    check_points_and_specs(z, models, conds, tag)
    reg_ids = read_list(lists / f"s{s}_region.txt")
    mat_ids = read_list(lists / f"s{s}_matched.txt")
    check_added_rows(z, conds, tag, lists, s, reg_ids, mat_ids)

    base = z["base"]
    added = sorted(set(reg_ids) | set(mat_ids))
    tree = cKDTree(xy.loc[added, ["x", "y"]].to_numpy())
    id_pid = np.asarray(base["point_id_id"]).astype(str)
    od_pid = np.asarray(base["point_id"]).astype(str)
    keep_id = far_from_added(tree, xy, id_pid)
    keep_od = far_from_added(tree, xy, od_pid)
    if land_only:
        keep_id &= xy.loc[id_pid, "landcover"].to_numpy() != WATER
        keep_od &= xy.loc[od_pid, "landcover"].to_numpy() != WATER
    regions = pd.read_parquet(regions_dir / f"{tag}_spatial_mu3_knn10_S500_points.parquet")
    region_ids = regions[regions.component_size >= MIN_REGION_POINTS].point_id.to_numpy()
    keep_od &= ~np.isin(od_pid, region_ids)

    # eval set -> (kept points, the half's point ids, its labels, its scores in the npz)
    eval_sets = {"AE_far": (keep_id, id_pid, base["y_id"], "S_id"),
                 "A_far": (keep_od, od_pid, base["y_ood"], "S_ood")}
    out = dict(tag=tag, n_added=len(added), n_pairs=len(reg_ids), groups={}, tiles={}, n_eval={})
    for e, (keep, pids, labels, scores) in eval_sets.items():
        y = labels[keep]
        out["tiles"][e] = xy.loc[pids[keep], "tile_50km"].to_numpy()
        out["n_eval"][e] = dict(n=int(keep.sum()), pos=int(y.sum()), of=len(keep))
        for c in conds:
            S = z[c][scores]
            out["groups"][(e, c)] = score_groups(pflood(S[:, keep], y), y)
    return out


def aucs(sd: dict, e: str, c: str, w: np.ndarray | None = None) -> np.ndarray:
    """Per-model AUC of population c on eval set e of one seed, under point weights w
    (default: all 1)."""
    groups = sd["groups"][(e, c)]
    if w is None:
        w = np.ones(groups["n"])
    return weighted_aucs(groups, w)


def point_table(dataset: Path) -> pd.DataFrame:
    """x and y (EPSG:5070 metres), tile_50km and landcover per point_id."""
    points = pd.read_parquet(dataset, columns=["point_id", "geometry", "tile_50km", "landcover"])
    geometry = shapely.from_wkb(points.geometry.to_numpy())
    return pd.DataFrame(dict(x=shapely.get_x(geometry), y=shapely.get_y(geometry),
                             tile_50km=points.tile_50km.to_numpy(),
                             landcover=points.landcover.to_numpy()),
                        index=points.point_id.to_numpy())


def available_populations(root: Path, seeds: list) -> dict:
    """The populations of CONDS whose cells exist for every seed."""
    conds = {}
    for c, folder in CONDS.items():
        paths = [root / folder / f"FL_band_AE_s{s}.npz" for s in seeds]
        if all(path.exists() for path in paths):
            conds[c] = folder
    return conds


def print_seed(s: int, sd: dict) -> None:
    """One line per seed: list sizes and the evaluation sets' counts."""
    parts = []
    for e, counts in sd["n_eval"].items():
        parts.append(f"{e} {counts['n']:,} of {counts['of']:,} points "
                     f"({counts['pos']:,} positives)")
    print(f"  seed {s}: {sd['n_pairs']} pairs ({sd['n_added']} distinct added points); eval sets "
          + "; ".join(parts), flush=True)


def bootstrap_diffs(sds: dict, conds: dict, contrasts: list, n_boot: int, seed: int,
                    t0: float) -> dict:
    """(seed, eval set, a, b) -> per draw, the mean over models of AUC(a) - AUC(b). One
    resample of the union of all seeds' evaluation tiles per draw, applied to every seed:
    each point weighs as often as its tile was drawn."""
    all_tiles = []
    for sd in sds.values():
        for e in EVALS:
            all_tiles.append(sd["tiles"][e])
    union = np.unique(np.concatenate(all_tiles))
    rng = np.random.default_rng(seed)
    boot = {}
    for s in sds:
        for e in EVALS:
            for pop_a, pop_b in contrasts:
                boot[(s, e, pop_a, pop_b)] = []
    for b in range(n_boot):
        draw = rng.choice(union, len(union))
        draw_counts = pd.Series(draw).value_counts()
        for s, sd in sds.items():
            for e in EVALS:
                w = draw_counts.reindex(sd["tiles"][e]).fillna(0).to_numpy(float)
                draw_aucs = {}
                for c in conds:
                    draw_aucs[c] = aucs(sd, e, c, w)
                for pop_a, pop_b in contrasts:
                    diff = draw_aucs[pop_a] - draw_aucs[pop_b]
                    boot[(s, e, pop_a, pop_b)].append(np.nanmean(diff))
        if (b + 1) % 100 == 0:
            print(f"    bootstrap {b + 1}/{n_boot} [{time.time() - t0:.0f}s]", flush=True)
    return boot


def contrast_row(seed_label: str, e: str, contrast: str, diffs: np.ndarray,
                 boot_diffs: np.ndarray, n_eval: int, mean_auc_a: float,
                 mean_auc_b: float) -> dict:
    """One output row: the per-model differences summarised, with the bootstrap interval."""
    return dict(seed=seed_label, eval=e, contrast=contrast, mean_diff=float(np.mean(diffs)),
                median_diff=float(np.median(diffs)),
                share_models_negative=float((diffs < 0).mean()),
                ci_lo=float(np.nanpercentile(boot_diffs, 2.5)),
                ci_hi=float(np.nanpercentile(boot_diffs, 97.5)),
                n_models=len(diffs), n_eval=n_eval, mean_auc_a=mean_auc_a, mean_auc_b=mean_auc_b)


def run(a) -> pd.DataFrame:
    """The Part F table for main's arguments: per contrast and eval set, one row per seed and
    a pooled row (models of all seeds; interval from the per-draw mean over the seeds)."""
    t0 = time.time()
    xy = point_table(a.dataset)
    conds = available_populations(a.root, a.seeds)
    missing = sorted(set(CONDS) - set(conds))
    if missing:
        print(f"  missing populations (skipped): {missing}", flush=True)
    sds = {}
    for s in a.seeds:
        sds[s] = seed_data(s, a.root, a.lists, a.regions_dir, xy, conds, land_only=True)
    for s, sd in sds.items():
        print_seed(s, sd)
    contrasts = []
    for pop_a, pop_b in CONTRASTS:
        if pop_a in conds and pop_b in conds:
            contrasts.append((pop_a, pop_b))

    point_aucs = {}            # (seed, eval set, population) -> per-model AUC, unweighted
    for s, sd in sds.items():
        for e in EVALS:
            for c in conds:
                point_aucs[(s, e, c)] = aucs(sd, e, c)
    boot = bootstrap_diffs(sds, conds, contrasts, a.boot, a.seed, t0)

    rows = []
    for e in EVALS:
        for pop_a, pop_b in contrasts:
            contrast = f"{pop_a} - {pop_b}"
            seed_diffs, seed_boots, seed_auc_a, seed_auc_b = [], [], [], []
            for s in a.seeds:
                auc_a, auc_b = point_aucs[(s, e, pop_a)], point_aucs[(s, e, pop_b)]
                diffs = auc_a - auc_b
                boot_diffs = np.asarray(boot[(s, e, pop_a, pop_b)])
                mean_auc_a, mean_auc_b = float(np.mean(auc_a)), float(np.mean(auc_b))
                rows.append(contrast_row(str(s), e, contrast, diffs, boot_diffs,
                                         sds[s]["n_eval"][e]["n"], mean_auc_a, mean_auc_b))
                seed_diffs.append(diffs)
                seed_boots.append(boot_diffs)
                seed_auc_a.append(mean_auc_a)
                seed_auc_b.append(mean_auc_b)
            pooled_boot = np.nanmean(np.vstack(seed_boots), axis=0)
            n_eval = sum(sds[s]["n_eval"][e]["n"] for s in a.seeds)
            rows.append(contrast_row("pooled", e, contrast, np.concatenate(seed_diffs),
                                     pooled_boot, n_eval, float(np.mean(seed_auc_a)),
                                     float(np.mean(seed_auc_b))))

    out = pd.DataFrame(rows)
    primary_eval, primary_a, primary_b = PRIMARY
    out["primary"] = ((out.seed == "pooled") & (out["eval"] == primary_eval)
                      & (out.contrast == f"{primary_a} - {primary_b}"))
    out["within_0p002"] = out.mean_diff.abs() < SMALL
    out["n_boot"] = a.boot
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
    return out


def primary_line(out: pd.DataFrame) -> str:
    """The primary's verdict, with the x10 version beside it when it was run."""
    p = out[out.primary].iloc[0]
    x10 = out[(out.seed == "pooled") & (out["eval"] == "AE_far")
              & (out.contrast == "region_x10 - matched_x10")]
    if p.ci_hi < 0:
        verdict = "MET"
    elif p.mean_diff < 0:
        verdict = "negative but interval spans 0"
    else:
        verdict = "NOT MET"
    line = (f"\nPRIMARY: Zone AE AUC (eval 1), +region minus +matched, pooled: {p.mean_diff:+.4f} "
            f"[{p.ci_lo:+.4f}, {p.ci_hi:+.4f}]; predicted negative -> {verdict}")
    if len(x10):
        line += (f"; x10: {x10.mean_diff.iloc[0]:+.4f} "
                 f"[{x10.ci_lo.iloc[0]:+.4f}, {x10.ci_hi.iloc[0]:+.4f}]")
        if p.within_0p002 and bool(x10.within_0p002.iloc[0]):
            line += "; both within +-0.002"
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/interim/aotil"),
                        help="folder holding the population folders")
    parser.add_argument("--lists", type=Path, default=Path("results/2026-10-17/o5/F_lists"),
                        help="the F lists, s<seed>_region.txt and s<seed>_matched.txt")
    parser.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"),
                        help="the region point tables")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--boot", type=int, default=1000, help="tile-bootstrap draws")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(args.out, index=False)
    pooled = out[out.seed == "pooled"]
    columns = ["eval", "contrast", "mean_diff", "ci_lo", "ci_hi", "share_models_negative",
               "mean_auc_a", "mean_auc_b", "within_0p002"]
    with pd.option_context("display.width", 220, "display.float_format", "{:+.4f}".format):
        print(pooled[columns].to_string(index=False))
    print(primary_line(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
