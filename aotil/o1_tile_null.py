#!/usr/bin/env python3
"""
Null check of o1_shift.py's per-tile gap: does it read about 0 when there is no shift?

For each null seed, the zones of one cell's test points are shuffled within each (tile, label)
group: the group's Zone A and Zone AE points are pooled and randomly reassigned, as many to each
half as before (shuffle_zones). Points keep their tile, label, land cover and scores, and the
models stay the same. So every tile keeps its point counts and its flood shares, and the two
pseudo-zones differ only by chance within a tile and class. The per-tile columns are then computed
as o1_shift.py computes them, with the same bootstrap draws; the pooled accuracy and F1 gaps beside
them still carry the tile mix. No model is refit.

Reads   <cells_dir>/<tag>.npz and <tag>_models.csv, and the dataset (tile, block, land cover)
Writes  <out_csv>, one row per (null seed, scope)

    python aotil/o1_tile_null.py data/interim/aotil/cells_v3 \\
        data/processed_w7/dataset_FL.parquet FL_band_AE_s0 results/<date>/o1_tile_null.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o1_shift as o1s  # noqa: E402

NULL_SEEDS = (1, 2, 3)
# Printed per row; every tile_ column goes to the CSV.
SHOWN = ("pooled_acc_gap", "pooled_f1_gap", "tile_acc_gap", "tile_f1_gap", "tile_auc_gap",
         "tile_acc_raw_gap", "tile_f1_raw_gap", "tile_strong15_acc_gap")


def shuffle_zones(c: dict, seed: int) -> dict:
    """The cell with its zones shuffled within each (tile, label) group of test points, as many
    of the group's points going to each half as before. The keys tile_gaps, scopes and
    strongest_models read (yo, yi, Po, Pi, Zo, Zi, to, ti, o, i, N, auc_id, auc_ood); the
    strongest models stay those of the real cell."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([c["yo"], c["yi"]])
    tile = np.concatenate([c["to"], c["ti"]])
    was_a = np.arange(len(y)) < len(c["yo"])
    to_a = np.zeros(len(y), bool)
    groups = pd.DataFrame(dict(tile=tile, y=y)).groupby(["tile", "y"]).indices
    for key in sorted(groups):
        members = groups[key]
        to_a[rng.permutation(members)[:was_a[members].sum()]] = True
    a, i = np.flatnonzero(to_a), np.flatnonzero(~to_a)
    P = np.concatenate([c["Po"], c["Pi"]], axis=1)
    Z = np.concatenate([c["Zo"], c["Zi"]], axis=1)
    rows = pd.concat([c["o"], c["i"]])
    return dict(yo=y[a], yi=y[i], Po=P[:, a], Pi=P[:, i], Zo=Z[:, a], Zi=Z[:, i], to=tile[a],
                ti=tile[i], o=rows.iloc[a], i=rows.iloc[i], N=c["N"], auc_id=c["auc_id"],
                auc_ood=c["auc_ood"])


def null_rows(c: dict, tag: str, paired_blocks: set, null_seeds, n_boot: int,
              seed: int) -> list[dict]:
    """One row per (null seed, scope): point counts, the pooled accuracy and F1 gaps (mean over
    models, all points of each pseudo-zone) and o1_shift.tile_columns, with the real cell's
    bootstrap draws (the shuffle keeps each tile's points in each half)."""
    draws = o1s.tile_draws(c["to"], n_boot, seed)
    rows = []
    for null_seed in null_seeds:
        s = shuffle_zones(c, null_seed)
        for scope, mo, mi in o1s.scopes(s["o"], s["i"], paired_blocks):
            counts = o1s.scope_counts(s, mo, mi)
            pooled = o1s.count_metrics(counts, np.ones(len(counts["tiles"])))
            t = o1s.tile_gaps(s, mo, mi)
            strong = o1s.strongest_models(s) if scope == "all" else None
            row = dict(cell=tag, null_seed=null_seed, scope=scope, n_A=int(mo.sum()),
                       n_AE=int(mi.sum()),
                       pooled_acc_gap=float(np.mean(pooled["acc"][0] - pooled["acc"][1])),
                       pooled_f1_gap=float(np.mean(pooled["f1"][0] - pooled["f1"][1])))
            row.update(o1s.tile_columns(t, o1s.draw_matrix(draws, t["tiles"]), strong))
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells_dir")
    parser.add_argument("dataset")
    parser.add_argument("tag", help="one cell tag")
    parser.add_argument("out_csv")
    parser.add_argument("--null-seeds", type=int, nargs="+", default=list(NULL_SEEDS),
                        help="one shuffle per seed (default 1 2 3)")
    parser.add_argument("--boot", type=int, default=o1s.N_BOOT, help="tile-bootstrap draws")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args()

    points = pd.read_parquet(args.dataset).drop(columns=["geometry"], errors="ignore")
    points = points.set_index("point_id")
    c = o1s.cell_arrays(args.cells_dir, args.tag, points)
    rows = null_rows(c, args.tag, o1s.blocks_with_both_zones(points), args.null_seeds,
                     args.boot, args.seed)
    for row in rows:
        shown = {key: row[key] for key in ("null_seed", "scope")}
        shown.update({key: round(float(row[key]), 4) for key in SHOWN})
        print(shown, flush=True)
    table = pd.DataFrame(rows)
    every = table[table.scope == "all"]
    for metric in ("acc", "f1", "auc"):
        print(f"scope all, largest |tile_{metric}_gap| over the shuffles: "
              f"{every[f'tile_{metric}_gap'].abs().max():.4f}")
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
