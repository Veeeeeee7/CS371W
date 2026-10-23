#!/usr/bin/env python3
"""
Checks for o5_pooling.py (Part F of docs/TASK-O5-REGIONS.md). Run from the repo root:

    python aotil/test_o5_pooling.py [--csv results/2026-10-17/o5/F_pooling.csv]

1. The AUC path: rebuilt p(flood) + o1_shift.weighted_aucs on the whole ID half of the base
   cell reproduces the stored auc_id of every model, up to the ties that rebuilding p from S
   creates for saturated models (max 5e-4, median 0).
2. The pooled populations: same points and specs as cells_w7; the stored added rows equal the
   F lists (x1 and x10); the region and matched lists have the same length and the same labels,
   pair by pair, and each pair shares its tile.
3. The evaluation sets: every eval-1 / eval-2 point is >= 5 km from every added point (brute
   force), eval 2 holds no region point, and no added point is scored.
4. The written CSV: one primary row; pooled point estimates = the mean over the models of all
   seeds; every interval has ci_lo <= ci_hi.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o5_pooling as fp                                  # noqa: E402
from o1_shift import score_groups, weighted_aucs         # noqa: E402

FAILS = []


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


def xy_table(dataset):
    """x and y (EPSG:5070 metres), tile, label and land cover per point_id."""
    columns = ["point_id", "geometry", "tile_50km", "label", "landcover"]
    d = pd.read_parquet(dataset, columns=columns)
    geometry = shapely.from_wkb(d.geometry.to_numpy())
    return pd.DataFrame(dict(x=shapely.get_x(geometry), y=shapely.get_y(geometry),
                             tile_50km=d.tile_50km.to_numpy(), label=d.label.to_numpy(),
                             landcover=d.landcover.to_numpy()),
                        index=d.point_id.to_numpy())


def check_auc_path(root):
    print("1. AUC path reproduces the stored auc_id")
    z = np.load(root / "cells_w7/FL_band_AE_s0.npz", allow_pickle=True)
    models = pd.read_csv(root / "cells_w7/FL_band_AE_s0_models.csv")
    y = z["y_id"]
    got = weighted_aucs(score_groups(fp.pflood(z["S_id"], y), y), np.ones(len(y)))
    # auc_id was ranked on the raw p. Rebuilding p from S loses about 1e-16 for saturated
    # models, which can turn scores that differ by less than that into ties.
    diff = np.abs(got - models.auc_id.to_numpy())
    check(diff.max() < 5e-4 and np.median(diff) < 1e-9,
          f"AUC = stored auc_id up to ties from rebuilding p "
          f"(max {diff.max():.1e}, median {np.median(diff):.1e})")


def lists_are_pairs(reg, mat, xy) -> bool:
    """Same length, same label and tile pair by pair, all on land, no point in both lists."""
    if len(reg) != len(mat):
        return False
    same_label = (xy.loc[reg, "label"].to_numpy() == xy.loc[mat, "label"].to_numpy()).all()
    same_tile = (xy.loc[reg, "tile_50km"].to_numpy() == xy.loc[mat, "tile_50km"].to_numpy()).all()
    all_land = (xy.loc[reg + mat, "landcover"] != 11).all()
    disjoint = not set(reg) & set(mat)
    return bool(same_label and same_tile and all_land and disjoint)


def check_eval_sets(s, sd, root, regions_dir, xy, added):
    """Rebuild one seed's evaluation sets independently of seed_data and compare."""
    print(f"3. evaluation sets, seed {s}")
    base = np.load(root / f"cells_w7/FL_band_AE_s{s}.npz", allow_pickle=True)
    added_xy = xy.loc[added, ["x", "y"]].to_numpy()
    all_ok = True
    for e, half in (("AE_far", "point_id_id"), ("A_far", "point_id")):
        pids = np.asarray(base[half]).astype(str)
        nearest = cKDTree(added_xy).query(xy.loc[pids, ["x", "y"]].to_numpy())[0]
        keep = nearest >= fp.FAR_M
        if e == "A_far":
            regions = pd.read_parquet(
                regions_dir / f"FL_band_AE_s{s}_spatial_mu3_knn10_S500_points.parquet")
            keep &= ~np.isin(pids, regions[regions.component_size >= 20].point_id.to_numpy())
        # brute-force distances on about 200 of the kept points
        sample = np.flatnonzero(keep)[:: max(1, keep.sum() // 200)]
        sample_xy = xy.loc[pids[sample], ["x", "y"]].to_numpy()
        offsets = sample_xy[:, None, :] - added_xy[None, :, :]
        brute = np.sqrt((offsets ** 2).sum(-1)).min(1)
        far_ok = bool((brute >= fp.FAR_M).all())
        no_added_scored = not np.isin(pids[keep], added).any()
        counts_match = sd["n_eval"][e]["n"] == int(keep.sum())
        all_ok = all_ok and far_ok and no_added_scored and counts_match
    check(all_ok, f"seed {s}: eval points >= 5 km from every added point (brute force), "
                  f"no added point scored, counts match "
                  f"({sd['n_eval']['AE_far']['n']:,} / {sd['n_eval']['A_far']['n']:,})")


def check_seeds(root, lists, regions_dir, xy):
    print("2. pooled populations and lists")
    for s in (0, 1, 2):
        reg = fp.read_list(lists / f"s{s}_region.txt")
        mat = fp.read_list(lists / f"s{s}_matched.txt")
        check(lists_are_pairs(reg, mat, xy),
              f"seed {s}: {len(reg)} pairs, same label and tile pair by pair, all land, "
              "lists disjoint")
        conds = fp.available_populations(root, [s])
        try:
            sd = fp.seed_data(s, root, lists, regions_dir, xy, conds)
        except SystemExit as error:
            check(False, f"seed {s}: {error}")
            continue
        check(True, f"seed {s}: {sorted(conds)} share points and specs; "
                    "stored added rows = the lists")
        added = np.array(sorted(set(reg) | set(mat)))
        check_eval_sets(s, sd, root, regions_dir, xy, added)


def check_csv(csv):
    print("4. written CSV")
    if not csv or not Path(csv).exists():
        print("  skip  no CSV")
        return
    out = pd.read_csv(csv)
    out["seed"] = out.seed.astype(str)
    check(int(out.primary.sum()) == 1, "exactly one primary row")
    per_seed = out[out.seed != "pooled"]
    pooled = out[out.seed == "pooled"]
    worst = 0.0
    for _, row in pooled.iterrows():
        seeds = per_seed[(per_seed["eval"] == row["eval"]) & (per_seed.contrast == row.contrast)]
        model_weighted = (seeds.mean_diff * seeds.n_models).sum() / seeds.n_models.sum()
        worst = max(worst, abs(model_weighted - row.mean_diff))
    check(worst < 1e-5, f"pooled mean = model-weighted mean of the seeds (max |diff| {worst:.1e})")
    check((out.ci_lo <= out.ci_hi).all(), "every interval has lo <= hi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/interim/aotil"))
    parser.add_argument("--lists", type=Path, default=Path("results/2026-10-17/o5/F_lists"))
    parser.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    parser.add_argument("--csv", default=None, help="o5_pooling's output CSV (check 4)")
    args = parser.parse_args()
    xy = xy_table(args.dataset)

    check_auc_path(args.root)
    check_seeds(args.root, args.lists, args.regions_dir, xy)
    check_csv(args.csv)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
