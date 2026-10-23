#!/usr/bin/env python3
"""O5 Part A: is the inversion label-noise memorization?

Sanyal et al. (2025) get an inverse line when models interpolate (fit to ~100%) noisy labels.
Here a model interpolates when its accuracy on its own training rows (`acc_train_sub`, from
zoo.py) is >= 0.99. For each seed, subset and model set the CSV has one row with
oodselect_paper.hard_corr (the probit ID-OOD r; r < 0 is the inversion) and its Fisher 95% CI,
plus seed = "pooled" rows that average Fisher z over seeds for the full selections and unions.
Subsets: the spatial and classic S = 500 selections, each region (>= 20 points), the region
unions; each also land only (NLCD class 11, open water, dropped). Model sets: all300, test60,
test_lt099, all_lt099 (the primary), all_lt095, fam_<kind>. Only test60 and test_lt099 are held
out from the selection (column `circular`). See docs/TASK-O5-REGIONS.md Part A and
docs/O5-REGIONS-REPORT-2026-10-17.md Part A.

    .venv/bin/python aotil/o5_interpolation.py --cells-dir data/interim/aotil/cells_w7 \\
        --sel-root results/2026-10-16/oodselect_paper --regions-dir results/2026-10-16/regions \\
        --regions-land results/2026-10-17/o5/regions_land.csv \\
        --dataset data/processed/dataset_FL.parquet --out results/2026-10-17/o5/A_interpolation.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set before numpy loads, so its BLAS starts with 4 threads.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op  # noqa: E402

WATER = 11              # NLCD land-cover class "open water"
INTERP = 0.99           # acc_train_sub at or above which a model interpolates its training labels
STRICT_CUT = 0.95       # all_lt095: a stricter "does not interpolate" cut
MIN_REGION = 20         # smallest region kept, in points (as regions.py)
FAMILIES = ("dt", "et", "hgb", "lr", "mlp", "nb", "rf", "xgb")
# The selection was optimised on the 180 selection models and tuned on the 60 validation models,
# so a model set that contains them measures r partly in-sample. Unlisted sets (fam_*) are circular.
CIRCULAR = {"all300": True, "test60": False, "test_lt099": False,
            "all_lt099": True, "all_lt095": True}
PRIMARY = ("spatial_full", False, "all_lt099")   # (subset, land_only, model set), pooled row
PREDICTED_BELOW = -0.5  # Part A prediction: the primary's pooled r is below this
POOLED_SUBSETS = ("spatial_full", "classic_full", "regions_all", "regions_pos", "regions_neg")


def top_mask(theta: np.ndarray, S: int) -> np.ndarray:
    """The selection as regions.py takes it: top S of theta by stable argsort (ties keep the
    earlier column)."""
    mask = np.zeros(len(theta), bool)
    mask[np.argsort(-np.asarray(theta), kind="stable")[:S]] = True
    return mask


def model_sets(models: pd.DataFrame) -> dict[str, np.ndarray]:
    """Model-set name -> sorted model indices (rows of Z / acc)."""
    n = len(models)
    _, _, test = op.model_split(n)
    test = np.sort(test)
    acc_train_sub = models.acc_train_sub.to_numpy()
    kind = models.kind.astype(str).to_numpy()
    sets = {"all300": np.arange(n),
            "test60": test,
            "test_lt099": test[acc_train_sub[test] < INTERP],
            "all_lt099": np.where(acc_train_sub < INTERP)[0],
            "all_lt095": np.where(acc_train_sub < STRICT_CUT)[0]}
    for family in FAMILIES:
        sets[f"fam_{family}"] = np.where(kind == family)[0]
    return sets


def r_row(Z: np.ndarray, acc: np.ndarray, idx: np.ndarray, mask: np.ndarray) -> dict:
    """hard_corr of models idx on the masked points, with its Fisher CI; NaN when it is undefined."""
    n_points = int(mask.sum())
    if n_points == 0 or len(idx) < 2:
        if len(idx):
            mean_acc_id = float(acc[idx].mean())
        else:
            mean_acc_id = np.nan
        return dict(n_models=len(idx), r=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    mean_acc_id=mean_acc_id, mean_acc_subset=np.nan, n_clamped=0)
    subset_acc = Z[idx][:, mask].mean(1)
    r = op.hard_corr(Z[idx], acc[idx], mask.astype(np.float32), n_points, is_mask=True)
    lo, hi = op.fisher_ci(r, len(idx))
    # A subset accuracy of exactly 0 or 1 puts the probit on its eps clamp (+-4.75); on small
    # all-positive regions such near-constant values can dominate r.
    n_clamped = int(((subset_acc == 0) | (subset_acc == 1)).sum())
    return dict(n_models=len(idx), r=r, ci_lo=lo, ci_hi=hi,
                mean_acc_id=float(acc[idx].mean()), mean_acc_subset=float(subset_acc.mean()),
                n_clamped=n_clamped)


def pool_fisher(rs, ns, level: float = 0.95) -> dict:
    """Fisher-z average over seeds: r = tanh(mean z), var(mean z) = sum 1/(n_s - 3) / m^2.

    NaN r and n <= 3 are left out. Seeds share most points and all model specs, so the interval
    is narrower than an honest one: it describes the runs, not a population.
    """
    rs = np.asarray(rs, float)
    ns = np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns > 3)
    if not ok.any():
        return dict(r=np.nan, ci_lo=np.nan, ci_hi=np.nan, z_bar=np.nan, se_z=np.nan, n_seeds=0)
    z = np.arctanh(np.clip(rs[ok], -0.999999, 0.999999))
    m = int(ok.sum())
    z_bar = float(z.mean())
    se = float(np.sqrt(np.sum(1.0 / (ns[ok] - 3)) / m ** 2))
    half_width = norm.ppf(0.5 + level / 2) * se
    return dict(r=float(np.tanh(z_bar)), ci_lo=float(np.tanh(z_bar - half_width)),
                ci_hi=float(np.tanh(z_bar + half_width)), z_bar=z_bar, se_z=se, n_seeds=m)


def subsets_for_seed(seed: int, cell, sel_root: Path, regions_dir: Path, rl: pd.DataFrame,
                     S: int, variant: str, classic: str) -> list[dict]:
    """Every subset of one seed as dicts {subset, subset_type, component, region_type, mask}.

    The unions of positive- and negative-dominated regions are reported land only
    (`land_only_only`), because region types are defined on land points.
    """
    tag = cell.tag
    point_id = np.asarray(cell.point_id).astype(str)
    n = len(point_id)
    column_of = pd.Series(np.arange(n), index=point_id)
    theta_spatial = np.load(sel_root / variant / f"{tag}_S{S}_theta.npy")
    theta_classic = np.load(sel_root / classic / f"{tag}_S{S}_theta.npy")
    if len(theta_spatial) != n or len(theta_classic) != n:
        raise ValueError(f"{tag}: theta length != number of OOD points {n}")
    spatial = top_mask(theta_spatial, S)
    subsets = [dict(subset="spatial_full", subset_type="spatial_full", component=-1,
                    region_type="", mask=spatial),
               dict(subset="classic_full", subset_type="classic_full", component=-1,
                    region_type="", mask=top_mask(theta_classic, S))]

    points = pd.read_parquet(regions_dir / f"{tag}_{variant}_S{S}_points.parquet")
    points = points[points.component_size >= MIN_REGION]
    type_of = rl[rl.seed == seed].set_index("component")["type"]
    union = np.zeros(n, bool)
    union_by_type = {"positive-dominated": np.zeros(n, bool),
                     "negative-dominated": np.zeros(n, bool)}
    for component, group in points.groupby("component"):
        mask = np.zeros(n, bool)
        mask[column_of.loc[group.point_id.astype(str)].to_numpy()] = True
        if not np.all(spatial[mask]):
            raise ValueError(f"{tag} component {component}: region points outside the spatial top-{S}")
        region_type = str(type_of.get(int(component), ""))
        subsets.append(dict(subset=f"region_c{int(component)}", subset_type="region",
                            component=int(component), region_type=region_type, mask=mask))
        union |= mask
        if region_type in union_by_type:
            union_by_type[region_type] |= mask

    subsets.append(dict(subset="regions_all", subset_type="regions_all", component=-1,
                        region_type="", mask=union))
    for region_type, mask in union_by_type.items():
        if mask.any():
            name = "regions_pos" if region_type.startswith("positive") else "regions_neg"
            subsets.append(dict(subset=name, subset_type=name, component=-1, region_type=region_type,
                                mask=mask, land_only_only=True))
    return subsets


def seed_rows(seed: int, a, rl: pd.DataFrame, landcover: pd.Series) -> list[dict]:
    """One row per (subset, land_only, model set) of one seed; prints the seed's water and
    interpolating-model counts."""
    tag = f"{a.cell_prefix}_s{seed}"
    cell, Z, acc, _ = op.load_paper_cell(a.cells_dir, tag)
    models = pd.read_csv(a.cells_dir / f"{tag}_models.csv")
    if not np.array_equal(models.idx.to_numpy(), np.arange(len(models))) or len(models) != Z.shape[0]:
        raise ValueError(f"{tag}: models.csv idx is not 0..{Z.shape[0] - 1}")
    point_id = np.asarray(cell.point_id).astype(str)
    land = landcover.reindex(point_id).to_numpy() != WATER
    y = np.asarray(cell.y_ood)
    sets = model_sets(models)
    _, _, test = op.model_split(len(models))
    is_test = np.isin(np.arange(len(models)), test)
    interpolates = models.acc_train_sub.to_numpy() >= INTERP
    print(f"seed {seed}: {Z.shape[0]} models x {Z.shape[1]} OOD points; water OOD points "
          f"{int((~land).sum())}; interpolating (acc_train_sub >= {INTERP}) "
          f"{int(interpolates.sum())} of {len(models)}, {int((interpolates & is_test).sum())} of 60 test")

    rows = []
    subsets = subsets_for_seed(seed, cell, a.sel_root, a.regions_dir, rl, a.budget, a.variant, a.classic)
    for sub in subsets:
        full_mask = sub["mask"]
        for land_only in (False, True):
            if sub.get("land_only_only") and not land_only:
                continue
            if land_only:
                mask = full_mask & land
                n_water_dropped = int((full_mask & ~land).sum())
            else:
                mask = full_mask
                n_water_dropped = 0
            base = dict(seed=str(seed), subset=sub["subset"], subset_type=sub["subset_type"],
                        component=sub["component"], region_type=sub["region_type"],
                        land_only=land_only, n_points=int(mask.sum()), n_pos=int(y[mask].sum()),
                        n_water_dropped=n_water_dropped)
            for name, idx in sets.items():
                rows.append({**base, "model_set": name,
                             "circular": CIRCULAR.get(name, True),
                             "n_test_models": int(is_test[idx].sum()),
                             "n_interp_099": int(interpolates[idx].sum()),
                             **r_row(Z, acc, idx, mask),
                             "z_bar": np.nan, "se_z": np.nan, "n_seeds": 1,
                             "n_models_by_seed": str(len(idx))})
    return rows


def pooled_row(subset: str, land_only: bool, model_set: str, group: pd.DataFrame) -> dict:
    """The seed = "pooled" row of one (subset, land_only, model set) from its per-seed rows."""
    pooled = pool_fisher(group.r, group.n_models)
    first = group.iloc[0]
    return dict(seed="pooled", subset=subset, subset_type=first.subset_type, component=-1,
                region_type=first.region_type, land_only=land_only,
                n_points=int(group.n_points.sum()), n_pos=int(group.n_pos.sum()),
                n_water_dropped=int(group.n_water_dropped.sum()), model_set=model_set,
                circular=bool(first.circular), n_test_models=int(group.n_test_models.sum()),
                n_interp_099=int(group.n_interp_099.sum()), n_models=int(group.n_models.sum()),
                r=pooled["r"], ci_lo=pooled["ci_lo"], ci_hi=pooled["ci_hi"],
                mean_acc_id=float(group.mean_acc_id.mean()),
                mean_acc_subset=float(group.mean_acc_subset.mean()),
                n_clamped=int(group.n_clamped.sum()), z_bar=pooled["z_bar"], se_z=pooled["se_z"],
                n_seeds=pooled["n_seeds"],
                n_models_by_seed=";".join(str(int(n)) for n in group.n_models))


def run(a) -> pd.DataFrame:
    """Per-seed rows for every subset and model set, then the pooled rows, with `primary` marked."""
    import torch   # imported here so that --help starts fast; hard_corr runs in torch
    torch.set_num_threads(a.threads)
    rl = pd.read_csv(a.regions_land)
    landcover = pd.read_parquet(a.dataset, columns=["point_id", "landcover"])
    landcover = landcover.set_index("point_id")["landcover"]
    rows = []
    for seed in a.seeds:
        rows += seed_rows(seed, a, rl, landcover)
    per_seed = pd.DataFrame(rows)

    # Only the full selections and region unions pool: region_c<k> names a different place per
    # seed. A subset with no points in a seed (no land, or no such region) leaves that seed out.
    pooled = []
    to_pool = per_seed[per_seed.subset.isin(POOLED_SUBSETS)]
    groups = to_pool.groupby(["subset", "land_only", "model_set"], sort=False)
    for (subset, land_only, model_set), group in groups:
        group = group[group.n_points > 0]
        if len(group) < 2:
            continue
        pooled.append(pooled_row(subset, land_only, model_set, group))

    out = pd.concat([per_seed, pd.DataFrame(pooled)], ignore_index=True)
    out["primary"] = ((out.seed == "pooled") & (out.subset == PRIMARY[0])
                      & (out.land_only == PRIMARY[1]) & (out.model_set == PRIMARY[2]))
    return out


def interp_table(cells_dir: Path, prefix: str, seeds) -> pd.DataFrame:
    """Per seed and family: models, interpolating (>= 0.99), >= 0.95, test ones, test ones >= 0.99."""
    rows = []
    for s in seeds:
        models = pd.read_csv(cells_dir / f"{prefix}_s{s}_models.csv")
        is_test = np.isin(models.idx, op.model_split(len(models))[2])
        for kind, group in models.groupby("kind"):
            interpolates = group.acc_train_sub >= INTERP
            group_is_test = is_test[group.index]
            rows.append(dict(seed=s, family=kind, n=len(group), interp_099=int(interpolates.sum()),
                             ge_095=int((group.acc_train_sub >= STRICT_CUT).sum()),
                             test=int(group_is_test.sum()),
                             test_interp_099=int((interpolates & group_is_test).sum()),
                             median_acc_train_sub=float(group.acc_train_sub.median())))
    return pd.DataFrame(rows)


def print_summary(out: pd.DataFrame, budget: int) -> None:
    """Pooled rows, the per-seed full selections and regions, the primary and its held-out twin."""
    show = ["subset", "land_only", "model_set", "circular", "n_seeds", "n_models_by_seed",
            "n_points", "r", "ci_lo", "ci_hi"]
    per_seed = out.seed != "pooled"
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 500):
        print("POOLED rows (Fisher-z average over seeds):")
        print(out.loc[~per_seed, show].round(3).to_string(index=False))

        print("\nper-seed, full selections, model sets 1-4:")
        full = (per_seed & out.subset.isin(["spatial_full", "classic_full"])
                & out.model_set.isin(["all300", "test60", "test_lt099", "all_lt099", "all_lt095"]))
        columns = ["seed", "subset", "land_only", "model_set", "n_models", "n_points", "r", "ci_lo", "ci_hi"]
        print(out.loc[full, columns].round(3).to_string(index=False))

        print("\nper-seed regions, test60 / test_lt099 / all_lt099:")
        regions = (per_seed & (out.subset_type == "region")
                   & out.model_set.isin(["test60", "test_lt099", "all_lt099"]))
        columns = ["seed", "subset", "region_type", "land_only", "model_set", "n_points",
                   "n_water_dropped", "n_clamped", "r", "ci_lo", "ci_hi"]
        print(out.loc[regions, columns].round(3).to_string(index=False))

    # With a single seed nothing pools, so there is no primary row and this raises IndexError.
    p = out[out.primary].iloc[0]
    verdict = "MET" if p.r < PREDICTED_BELOW else "NOT MET"
    print(f"\nPRIMARY (pooled, spatial S={budget} all points, all models acc_train_sub < 0.99, "
          f"n = {p.n_models_by_seed}): r = {p.r:+.3f} [{p.ci_lo:+.3f}, {p.ci_hi:+.3f}]; "
          f"predicted below {PREDICTED_BELOW}: {verdict}")
    held_out = ((out.seed == "pooled") & (out.subset == "spatial_full") & ~out.land_only
                & (out.model_set == "test_lt099"))
    h = out[held_out].iloc[0]
    print(f"held-out companion (test models acc_train_sub < 0.99, n = {h.n_models_by_seed}): "
          f"r = {h.r:+.3f} [{h.ci_lo:+.3f}, {h.ci_hi:+.3f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--cell-prefix", default="FL_band_AE")
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--regions-land", type=Path, default=Path("results/2026-10-17/o5/regions_land.csv"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    ap.add_argument("--variant", default="spatial_mu3_knn10")
    ap.add_argument("--classic", default="classic")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    table = interp_table(a.cells_dir, a.cell_prefix, a.seeds)
    print("\ninterpolating models (acc_train_sub >= 0.99) by seed and family:")
    print(table.pivot(index="family", columns="seed", values="interp_099").to_string())
    print("of them in the 60 test models:")
    print(table.pivot(index="family", columns="seed", values="test_interp_099").to_string())
    print()

    out = run(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6g")
    print_summary(out, a.budget)
    print(f"wrote {a.out} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
