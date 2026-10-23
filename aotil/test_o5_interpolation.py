#!/usr/bin/env python3
"""Checks for o5_interpolation.py (docs/TASK-O5-REGIONS.md, Part A). Run from the repo root:

    .venv/bin/python aotil/test_o5_interpolation.py [--csv results/2026-10-17/o5/A_interpolation.csv]

1. pool_fisher: the pooled r and its interval, and which seeds are left out.
2. r_row matches a float64 re-derivation of the authors' statistic; n_clamped.
3. top_mask: a tie at the cut keeps the earlier column, as regions.py does.
4. model_sets: the test block, the acc_train_sub cuts, the family partition.
5. Real cells (skipped when missing): the stored 2026-10-16 r_test is reproduced, and the
   regions match results/2026-10-17/o5/regions_land.csv.
6. The written CSV (with --csv): one primary row; pooled rows are pool_fisher of their seeds.
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# library warnings would only clutter the check output
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import oodselect_paper as op  # noqa: E402
import o5_interpolation as o5  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def ref_r(Z, acc, mask, eps=1e-6):
    """The authors' statistic in float64 numpy, from the definition: subset accuracy, probit
    clamped at eps, population moments, + eps on both variances."""
    a = Z[:, mask].mean(1) * mask.sum() / (mask.sum() + 1e-6)
    x = norm.ppf(np.clip(a, eps, 1 - eps))
    y = norm.ppf(np.clip(acc, eps, 1 - eps))
    xc = x - x.mean()
    yc = y - y.mean()
    return float(np.mean(xc * yc) / np.sqrt((np.mean(xc ** 2) + eps) * (np.mean(yc ** 2) + eps)))


def test_pool():
    print("1. pool_fisher")
    p = o5.pool_fisher([-0.7, -0.7, -0.7], [60, 60, 60])
    check(abs(p["r"] + 0.7) < 1e-12, f"equal r pools to itself: {p['r']:.6f}")

    ns = np.array([271, 274, 273])
    p = o5.pool_fisher([-0.8, -0.6, -0.7], ns)
    z_bar = np.arctanh([-0.8, -0.6, -0.7]).mean()
    se = np.sqrt(np.sum(1 / (ns - 3)) / 9)
    check(abs(p["z_bar"] - z_bar) < 1e-12 and abs(p["se_z"] - se) < 1e-12,
          f"z_bar and se = sqrt(sum 1/(n_s-3)/9): {p['se_z']:.5f} vs {se:.5f}")
    check(abs(p["ci_lo"] - np.tanh(z_bar - 1.959964 * se)) < 1e-6
          and abs(p["ci_hi"] - np.tanh(z_bar + 1.959964 * se)) < 1e-6, "interval = tanh(z_bar -/+ 1.96 se)")

    p2 = o5.pool_fisher([-0.5, -0.3], [52, 51])
    se2 = np.sqrt((1 / 49 + 1 / 48) / 4)
    check(abs(p2["se_z"] - se2) < 1e-12 and p2["n_seeds"] == 2, "m = 2: se = sqrt(sum 1/(n_s-3)) / 2")

    p3 = o5.pool_fisher([-0.5, np.nan, -0.3], [52, 60, 3])
    check(p3["n_seeds"] == 1 and abs(p3["r"] + 0.5) < 1e-12, "NaN r and n <= 3 are left out")


def test_r_row():
    print("2. r_row vs a float64 re-derivation")
    # Away from the probit clamp torch's float32 path and float64 agree to ~1e-7. A model with
    # subset accuracy exactly 0 or 1 sits on the clamp, where float32 erfinv moves ~0.01 per ulp
    # (audit_selection.py, check 2), so those cases get the audit's 1e-3 tolerance.
    rng = np.random.default_rng(3)
    worst = 0.0
    worst_clamped = 0.0
    n_clamped_cases = 0
    for _ in range(8):
        N, d = int(rng.integers(20, 300)), int(rng.integers(300, 3000))
        skill = rng.uniform(0.5, 0.95, N)
        Z = (rng.random((N, d)) < skill[:, None]).astype(np.float32)
        acc = np.clip(skill + rng.normal(0, 0.02, N), 0.01, 0.99)
        for S in (20, 100, 500):
            mask = np.zeros(d, bool)
            mask[rng.choice(d, S, replace=False)] = True
            idx = np.sort(rng.choice(N, max(5, N // 2), replace=False))
            row = o5.r_row(Z, acc, idx, mask)
            diff = abs(row["r"] - ref_r(Z[idx], acc[idx], mask))
            if row["n_clamped"]:
                worst_clamped = max(worst_clamped, diff)
                n_clamped_cases += 1
            else:
                worst = max(worst, diff)
    check(worst < 1e-5,
          f"{24 - n_clamped_cases} cases without a clamped model: max |diff| = {worst:.2e} (< 1e-5)")
    check(worst_clamped < 1e-3,
          f"{n_clamped_cases} cases with a model on the clamp: max |diff| = {worst_clamped:.2e} (< 1e-3)")

    Z = np.zeros((6, 30), np.float32)
    Z[0] = 1
    Z[1, :10] = 1
    mask = np.zeros(30, bool)
    mask[:20] = True
    row = o5.r_row(Z, np.linspace(0.6, 0.9, 6), np.arange(6), mask)
    check(row["n_clamped"] == 5,
          f"n_clamped counts subset accuracy exactly 0 or 1: {row['n_clamped']} (want 5)")
    row = o5.r_row(Z, np.linspace(0.6, 0.9, 6), np.arange(6), np.zeros(30, bool))
    check(np.isnan(row["r"]), "empty mask -> NaN")


def test_top_mask():
    print("3. top_mask")
    theta = np.array([0.1, 0.9, 0.5, 0.5, 0.2, 0.5])
    m = o5.top_mask(theta, 3)
    check(m.tolist() == [False, True, True, True, False, False],
          f"tie at the cut keeps the earlier columns: {m.astype(int)}")


def test_model_sets():
    print("4. model_sets")
    rng = np.random.default_rng(0)
    n = 300
    models = pd.DataFrame(dict(idx=np.arange(n), kind=np.array(o5.FAMILIES)[np.arange(n) % 8],
                               acc_train_sub=rng.uniform(0.7, 1.0, n)))
    sets = o5.model_sets(models)
    test = np.sort(op.model_split(n)[2])
    acc_train_sub = models.acc_train_sub.to_numpy()
    check(np.array_equal(sets["test60"], test) and len(test) == 60, "test60 = op.model_split test block (60)")
    check(set(sets["test_lt099"]) <= set(sets["test60"])
          and np.all(acc_train_sub[sets["test_lt099"]] < 0.99), "test_lt099 inside test60, all < 0.99")
    check(set(sets["all_lt095"]) <= set(sets["all_lt099"]), "all_lt095 inside all_lt099")
    check(np.array_equal(sets["all_lt099"], np.where(acc_train_sub < 0.99)[0])
          and np.array_equal(sets["all_lt095"], np.where(acc_train_sub < 0.95)[0])
          and np.array_equal(np.sort(sets["test_lt099"]), np.sort(test[acc_train_sub[test] < 0.99])),
          "all_lt099 / all_lt095 / test_lt099 are exactly the < 0.99 / < 0.95 / test & < 0.99 models")

    at_cuts = models.copy()
    at_cuts.loc[0, "acc_train_sub"] = 0.99
    at_cuts.loc[1, "acc_train_sub"] = 0.95
    sets_at_cuts = o5.model_sets(at_cuts)
    check(0 not in set(sets_at_cuts["all_lt099"]) and 1 in set(sets_at_cuts["all_lt099"])
          and 1 not in set(sets_at_cuts["all_lt095"]),
          "the cut is strict: acc_train_sub == 0.99 interpolates, == 0.95 is not < 0.95")

    family_parts = [sets[f"fam_{kind}"] for kind in o5.FAMILIES]
    check(np.array_equal(np.sort(np.concatenate(family_parts)), np.arange(n)),
          "families partition the 300 models")


def check_against_written_csv(s, csv, subs, Z, acc, models, land):
    """r in the written CSV (land only, all_lt099) against an independent float64 r."""
    out = pd.read_csv(csv)
    out["seed"] = out.seed.astype(str)
    idx = np.where(models.acc_train_sub.to_numpy() < 0.99)[0]
    for key in ("spatial_full", "classic_full"):
        m = subs[key]["mask"] & land            # land only, rebuilt here from the point table
        want = ref_r(Z[idx], acc[idx], m)
        row = ((out.seed == str(s)) & (out.subset == key) & out.land_only.astype(bool)
               & (out.model_set == "all_lt099"))
        got = out[row].r.iloc[0]
        check(abs(got - want) < 1e-3, f"seed {s} {key} land-only all_lt099: CSV r {got:+.5f} "
                                      f"= independent float64 r {want:+.5f}")


def test_real(cells, sel_root, regions_dir, regions_land, dataset):
    print("5. real cells")
    if not (cells.exists() and regions_land.exists() and dataset.exists()):
        print("  skip  cells / regions_land / dataset missing")
        return
    rl = pd.read_csv(regions_land)
    landcover = pd.read_parquet(dataset, columns=["point_id", "landcover"]).set_index("point_id")["landcover"]
    # the CSV of the 2026-10-17 run, found from the repo root whatever the working directory
    # (separate from --csv, which test_csv checks)
    written_csv = Path(__file__).resolve().parents[1] / "results/2026-10-17/o5/A_interpolation.csv"
    for s in (0, 1, 2):
        tag = f"FL_band_AE_s{s}"
        if not (cells / f"{tag}.npz").exists():
            print(f"  skip  {tag} missing")
            continue
        cell, Z, acc, _ = op.load_paper_cell(cells, tag)
        models = pd.read_csv(cells / f"{tag}_models.csv")
        test = o5.model_sets(models)["test60"]
        subs = {}
        for d in o5.subsets_for_seed(s, cell, sel_root, regions_dir, rl, 500, "spatial_mu3_knn10", "classic"):
            subs[d["subset"]] = d

        for variant, key in (("spatial_mu3_knn10", "spatial_full"), ("classic", "classic_full")):
            stored = float(pd.read_csv(sel_root / variant / f"{tag}_S500_result.csv").r_test.iloc[0])
            got = o5.r_row(Z, acc, test, subs[key]["mask"])["r"]
            check(abs(got - stored) < 1e-6,
                  f"seed {s} {key}: test60 r {got:+.6f} = stored r_test {stored:+.6f}")

        land = landcover.reindex(np.asarray(cell.point_id).astype(str)).to_numpy() != o5.WATER
        spatial = subs["spatial_full"]["mask"]
        seed_regions = rl[rl.seed == s]
        bad = []
        for _, row in seed_regions.iterrows():
            m = subs[f"region_c{int(row.component)}"]["mask"]
            ok = (np.all(spatial[m]) and m.sum() == row.n and (m & land).sum() == row.n_land
                  and (m & ~land).sum() == row.dropped_water)
            if not ok:
                bad.append(int(row.component))
        mismatch = f" (mismatch: {bad})" if bad else ""
        check(not bad, f"seed {s}: {len(seed_regions)} regions inside the spatial top 500, sizes / land / "
                       f"water drops = regions_land.csv" + mismatch)

        if written_csv.exists():
            check_against_written_csv(s, written_csv, subs, Z, acc, models, land)

        region_points = 0
        for key, d in subs.items():
            if key.startswith("region_c"):
                region_points += d["mask"].sum()
        check(abs(region_points - subs["regions_all"]["mask"].sum()) == 0,
              f"seed {s}: regions_all = disjoint union of the regions")


def test_csv(path):
    print("6. written CSV")
    if not path or not Path(path).exists():
        print("  skip  no CSV")
        return
    out = pd.read_csv(path)
    out["seed"] = out.seed.astype(str)
    check(int(out.primary.sum()) == 1, f"exactly one primary row ({int(out.primary.sum())})")

    per = out[out.seed != "pooled"]
    pool = out[out.seed == "pooled"]
    worst = 0.0
    for _, r in pool.iterrows():
        g = per[(per.subset == r.subset) & (per.land_only == r.land_only) & (per.model_set == r.model_set)
                & (per.n_points > 0)]
        p = o5.pool_fisher(g.r, g.n_models)
        worst = max(worst, abs(p["r"] - r.r), abs(p["ci_lo"] - r.ci_lo), abs(p["ci_hi"] - r.ci_hi))
    check(worst < 1e-5,
          f"{len(pool)} pooled rows = pool_fisher of their per-seed rows (max |diff| {worst:.1e})")
    full = pool[pool.subset.isin(["spatial_full", "classic_full"])]
    check(len(full) and (full.n_seeds == 3).all(), "every full-selection pooled row is over 3 seeds")

    pr = out[out.primary.astype(bool)].iloc[0]
    primary_per_seed = per[(per.subset == "spatial_full") & ~per.land_only.astype(bool)
                           & (per.model_set == "all_lt099")]
    n_by_seed = []
    for s in ("0", "1", "2"):
        n_by_seed.append(str(int(primary_per_seed[primary_per_seed.seed == s].n_models.iloc[0])))
    want_n = ";".join(n_by_seed)
    check(pr.seed == "pooled" and pr.subset == "spatial_full" and not bool(pr.land_only)
          and pr.model_set == "all_lt099" and str(pr.n_models_by_seed).replace(" ", "") == want_n,
          f"the primary row is pooled / spatial_full / all points / all_lt099 over n = {want_n} "
          f"(got {pr.seed} / {pr.subset} / land {pr.land_only} / {pr.model_set} / {pr.n_models_by_seed})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--regions-dir", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--regions-land", type=Path, default=Path("results/2026-10-17/o5/regions_land.csv"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    test_pool()
    test_r_row()
    test_top_mask()
    test_model_sets()
    test_real(a.cells, a.sel_root, a.regions_dir, a.regions_land, a.dataset)
    test_csv(a.csv)
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
