#!/usr/bin/env python3
"""O5 Part D2: which feature carries the inversion? Leave-one-feature-out populations.

For each of the 7 model features f, zoo.py built a 300-model population without f
(<lofo-root>/cells_w7_minus_<f>). Model i has the same family, hyperparameters and seed in every
population; only the feature list differs. The masks are not re-selected: they are the fixed
spatial and classic S = 500 selections and regions of the 2026-10-16 run. On each mask the
paper's statistic (oodselect_paper.hard_corr on the 60 test models) gives
    delta_r(f) = r(without f) - r(with all 7),
so a positive delta_r means f carries part of the inversion. The CSV has one row per seed,
feature and mask, plus seed-mean (seed = mean3) rows. Primary: delta_r on the full spatial
selection, seed mean; met if log10_dist_river has the largest delta_r and r without it is above
-0.3. If no feature lifts r above -0.3, the pair D1's coefficients predict to help most is
printed. See docs/TASK-O5-REGIONS.md Part D.

    .venv/bin/python aotil/o5_lofo.py --out results/2026-10-17/o5/D2_lofo.csv
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op  # noqa: E402

FEATURES = ["landcover", "slope", "tri_300", "wetindexcat", "log10_dist_river",
            "log10_dist_coast", "hand"]
PREDICTED = "log10_dist_river"
ALTERNATIVE = "log10_dist_coast"
THRESHOLD = -0.3        # r without the predicted feature must rise above this; D1's pair is
                        # printed only when no feature's removal lifts r above it
S = 500                 # selection budget of the fixed masks
MIN_REGION = 20         # smallest region kept, in points (as regions.py)
WATER = 11              # NLCD land-cover class "open water"


def top_mask(theta: np.ndarray, k: int) -> np.ndarray:
    """The top k of theta by stable argsort (ties keep the earlier column), as regions.py."""
    mask = np.zeros(len(theta), bool)
    mask[np.argsort(-np.asarray(theta), kind="stable")[:k]] = True
    return mask


def masks_for_seed(seed: int, point_id: np.ndarray, sel_root: Path, regions_dir: Path,
                   land: np.ndarray) -> dict[str, np.ndarray]:
    """Mask name -> boolean mask over the cell's OOD columns; each also as <name>_land."""
    tag = f"FL_band_AE_s{seed}"
    masks = {}
    masks["spatial_full"] = top_mask(np.load(sel_root / "spatial_mu3_knn10" / f"{tag}_S{S}_theta.npy"), S)
    masks["classic_full"] = top_mask(np.load(sel_root / "classic" / f"{tag}_S{S}_theta.npy"), S)

    points = pd.read_parquet(regions_dir / f"{tag}_spatial_mu3_knn10_S{S}_points.parquet")
    points = points[points.component_size >= MIN_REGION]
    column_of = pd.Series(np.arange(len(point_id)), index=point_id)
    for component, group in points.groupby("component"):
        mask = np.zeros(len(point_id), bool)
        mask[column_of.loc[group.point_id.to_numpy()].to_numpy()] = True
        masks[f"region_c{int(component)}"] = mask
    region_masks = [mask for name, mask in masks.items() if name.startswith("region_c")]
    # with no region this is the scalar np.False_, which the _land step below broadcasts
    masks["regions_all"] = np.logical_or.reduce(region_masks)

    for name in list(masks):
        masks[name + "_land"] = masks[name] & land
    return masks


def r_on(cells_dir: Path, tag: str, masks: dict, te: np.ndarray) -> tuple[dict, dict]:
    """r of the test models te on every mask (NaN on an empty one), and the cell's points and
    model specs, so populations can be checked against each other."""
    cell, Z, acc, _ = op.load_paper_cell(cells_dir, tag)
    rs = {}
    for name, mask in masks.items():
        if mask.any():
            rs[name] = op.hard_corr(Z[te], acc[te], mask.astype(np.float32), int(mask.sum()), is_mask=True)
        else:
            rs[name] = np.nan
    meta = dict(point_id=np.asarray(cell.point_id).astype(str),
                point_id_id=np.asarray(cell.point_id_id).astype(str),
                kind=cell.models.kind.to_numpy(), seed=cell.models.seed.to_numpy())
    return rs, meta


def d1_pair(d1_csv: Path) -> dict | None:
    """The feature pair whose D1 coefficients sum most negative, or None without usable D1 rows.

    D1 regresses probit(selection accuracy) on "this model saw f" indicators, so the pair with
    the most negative sum is the one whose removal D1 predicts to help most. Rows: seed mean,
    spatial S = 500 selection, the 60 test models, the brief's formula.
    """
    if not d1_csv.exists():
        return None
    d1 = pd.read_csv(d1_csv)
    d1 = d1.assign(term=d1.term.astype(str).str.removeprefix("saw_"))   # D1 names them saw_<f>
    rows = d1[(d1.seed.astype(str) == "mean3") & (d1.response == "spatial_S500_all")
              & (d1.model_set == "test60") & (d1.term.isin(FEATURES))]
    if "formula" in d1:
        rows = rows[rows.formula.astype(str) == "id+family+features"]
    if rows.empty:
        return None
    coef = rows.set_index("term").coef
    best = min(combinations(FEATURES, 2), key=lambda pair: coef.get(pair[0], 0) + coef.get(pair[1], 0))
    if "formula" in rows:
        formula = str(rows.formula.iloc[0])
    else:
        formula = ""
    return dict(pair=" + ".join(best), coef_sum=float(coef[best[0]] + coef[best[1]]), formula=formula)


def check_same_population(meta: dict, meta_all7: dict, cells_dir: Path, base_dir: Path, tag: str) -> None:
    """Stop unless the minus-f cell has the base cell's points (so the fixed masks address the
    same points) and model specs (so model i is the same learner)."""
    same_points = (np.array_equal(meta["point_id"], meta_all7["point_id"])
                   and np.array_equal(meta["point_id_id"], meta_all7["point_id_id"]))
    same_specs = (np.array_equal(meta["kind"], meta_all7["kind"])
                  and np.array_equal(meta["seed"], meta_all7["seed"]))
    if not (same_points and same_specs):
        raise SystemExit(f"{cells_dir}/{tag}: points or specs differ from {base_dir}/{tag} "
                         f"(points {same_points}, specs {same_specs}); the fixed masks do not apply")


def seed_rows(seed: int, a, landcover: pd.Series) -> list[dict]:
    """One row per (feature, mask) of one seed; prints r without each feature on the full
    spatial selection."""
    tag = f"FL_band_AE_s{seed}"
    base_cell, *_ = op.load_paper_cell(a.cells_dir, tag)
    point_id = np.asarray(base_cell.point_id).astype(str)
    land = landcover.reindex(point_id).to_numpy() != WATER
    masks = masks_for_seed(seed, point_id, a.sel_root, a.regions_dir, land)
    te = op.model_split(len(base_cell.models))[2]
    r_all7, meta_all7 = r_on(a.cells_dir, tag, masks, te)

    rows = []
    progress = []
    for feature in FEATURES:
        cells_dir = Path(a.lofo_root) / f"cells_w7_minus_{feature}"
        if not (cells_dir / f"{tag}.npz").exists():
            print(f"  missing {cells_dir}/{tag}.npz; skipped", flush=True)
            continue
        r_without, meta = r_on(cells_dir, tag, masks, te)
        check_same_population(meta, meta_all7, cells_dir, a.cells_dir, tag)
        for name, mask in masks.items():
            rows.append(dict(seed=seed, feature=feature, subset=name.removesuffix("_land"),
                             land_only=name.endswith("_land"), n_points=int(mask.sum()),
                             r_all7=r_all7[name], r_without=r_without[name],
                             delta_r=r_without[name] - r_all7[name]))
        progress.append(f"{feature} {r_without['spatial_full']:+.3f}")
    print(f"  seed {seed}: spatial S=500 r(all 7) {r_all7['spatial_full']:+.3f}; r without each: "
          + ", ".join(progress), flush=True)
    return rows


def run(a) -> pd.DataFrame:
    """Per-seed rows for every feature and mask, then the seed-mean ("mean3") rows."""
    landcover = pd.read_parquet(a.dataset, columns=["point_id", "landcover"]).set_index("point_id").landcover
    rows = []
    for seed in a.seeds:
        rows += seed_rows(seed, a, landcover)
    out = pd.DataFrame(rows)

    # region_c<k> is numbered separately in each seed (a different place per seed), so the
    # per-region subsets get no seed mean; n_seeds counts the seeds with a finite delta_r.
    not_region = out[~out.subset.str.startswith("region_c")]
    mean = not_region.groupby(["feature", "subset", "land_only"], as_index=False).agg(
        r_all7=("r_all7", "mean"), r_without=("r_without", "mean"),
        delta_r=("delta_r", "mean"), n_seeds=("delta_r", "count"),
        n_points=("n_points", "sum"))
    mean.insert(0, "seed", "mean3")

    per_seed = out.assign(seed=out.seed.astype(str), n_seeds=1)
    out = pd.concat([per_seed, mean], ignore_index=True)
    out["primary"] = (out.seed == "mean3") & (out.subset == "spatial_full") & ~out.land_only
    return out


def verdict(out: pd.DataFrame) -> dict:
    """The primary's outcome from the seed-mean rows; complete=False unless every feature has
    all three seeds."""
    p = out[out.primary & (out.n_seeds == 3)].set_index("feature")
    if len(p) < len(FEATURES):
        return dict(complete=False)
    largest = p.delta_r.idxmax()
    r_without_largest = float(p.loc[largest, "r_without"])
    met = bool(largest == PREDICTED and r_without_largest > THRESHOLD)
    return dict(complete=True, largest=largest, largest_delta=float(p.loc[largest, "delta_r"]),
                r_without_largest=r_without_largest, r_all7=float(p.r_all7.iloc[0]),
                predicted_delta=float(p.loc[PREDICTED, "delta_r"]),
                predicted_r_without=float(p.loc[PREDICTED, "r_without"]),
                alternative_delta=float(p.loc[ALTERNATIVE, "delta_r"]), met=met,
                any_above=bool((p.r_without > THRESHOLD).any()))


def print_summary(out: pd.DataFrame, d1_csv: Path) -> None:
    """The seed-mean table, the primary's verdict and, if no feature lifts r enough, D1's pair."""
    v = verdict(out)
    shown = ((out.seed == "mean3") & (out.subset.isin(["spatial_full", "classic_full", "regions_all"]))
             & ~out.land_only)
    table = out[shown].pivot_table(index="feature", columns="subset", values=["r_without", "delta_r"])
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print("\nseed mean, all points:\n" + table.to_string())
    if not v.get("complete"):
        return
    outcome = "MET" if v["met"] else "NOT MET"
    print(f"\nPRIMARY: largest delta_r on the full spatial S=500 selection (seed mean) is {v['largest']} "
          f"({v['largest_delta']:+.3f}; r {v['r_all7']:+.3f} -> {v['r_without_largest']:+.3f}). "
          f"{PREDICTED}: delta_r {v['predicted_delta']:+.3f}, r without it {v['predicted_r_without']:+.3f}. "
          f"Prediction {outcome} (needs river largest and r without it > {THRESHOLD}).")
    if v["any_above"]:
        return
    pair = d1_pair(d1_csv)
    if pair:
        print(f"  no single feature turns r above {THRESHOLD}: D1's coefficients (seed mean, "
              f"spatial selection, test 60) predict the pair {pair['pair']} (coefficient sum "
              f"{pair['coef_sum']:+.3f}) as the two-feature combination with the largest delta_r")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--lofo-root", type=Path, default=Path("data/interim/aotil"))
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    ap.add_argument("--d1", type=Path, default=Path("results/2026-10-17/o5/D1_feature_inclusion.csv"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = run(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(a.out, index=False)
    print_summary(out, a.d1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
