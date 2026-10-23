#!/usr/bin/env python3
"""Checks for o5_lofo.py (Part D2 of docs/TASK-O5-REGIONS.md). Run from the repo root:

    python aotil/test_o5_lofo.py [--csv results/2026-10-17/o5/D2_lofo.csv]

1. top_mask: the largest theta, ties broken by column order.
2. verdict(): met only when log10_dist_river has the largest delta_r AND the seed-mean r without
   it is above -0.3; d1_pair() picks the pair with the most negative coefficient sum from the
   brief's formula.
3. The leave-one-feature-out inputs: each data/processed_w7_minus_<f>/ lists the other six
   features and links to the same dataset; each built cell has the same OOD / ID point_id
   columns and the same model kinds and seeds as cells_w7.
4. With all 7 features, r on the fixed masks reproduces the 2026-10-16 r_test.
5. The written CSV: seed-mean rows are the mean of the per-seed rows; one primary row per feature.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o5_lofo as lf          # noqa: E402
import oodselect_paper as op  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def read_features(path: Path) -> list[str]:
    """The feature names in a FEATURES.txt, without blank lines and # comments."""
    names = []
    with open(path) as fh:
        for line in fh:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    return names


def primary_table(deltas: dict, r_without: dict) -> pd.DataFrame:
    """Seed-mean primary rows with r(all 7) = -0.8 and the given delta_r per feature."""
    rows = []
    for f in lf.FEATURES:
        rows.append(dict(seed="mean3", feature=f, subset="spatial_full", land_only=False,
                         r_all7=-0.8, r_without=r_without.get(f, -0.8 + deltas[f]), delta_r=deltas[f],
                         n_seeds=3, primary=True))
    return pd.DataFrame(rows)


def test_top_mask():
    print("1. top_mask")
    m = lf.top_mask(np.array([1.0, 3.0, 3.0, 2.0, 3.0]), 2)
    check(m.tolist() == [False, True, True, False, False], f"ties keep the earlier columns: {m.astype(int)}")


def test_verdict_and_pair(tmp: Path):
    print("2. verdict and d1_pair")
    base = {f: 0.01 for f in lf.FEATURES}
    v = lf.verdict(primary_table({**base, "log10_dist_river": 0.6}, {}))
    check(v["met"] and v["largest"] == "log10_dist_river", "river largest and r above -0.3 -> met")
    v = lf.verdict(primary_table({**base, "log10_dist_river": 0.3}, {}))
    check(not v["met"] and v["largest"] == "log10_dist_river",
          "river largest but r stays below -0.3 -> not met")
    v = lf.verdict(primary_table({**base, "hand": 0.7, "log10_dist_river": 0.6}, {}))
    check(not v["met"] and v["largest"] == "hand", "another feature largest -> not met")

    rows = []
    for f, coef in zip(lf.FEATURES, [0.1, -0.2, 0.0, 0.05, -0.3, -0.25, 0.2]):
        rows.append(dict(seed="mean3", response="spatial_S500_all", model_set="test60",
                         formula="id+family+features", term=f"saw_{f}", coef=coef))
    # a row from another formula, which d1_pair must ignore
    rows.append(dict(seed="mean3", response="spatial_S500_all", model_set="test60", formula="family+features",
                     term="landcover", coef=-9.0))
    path = tmp / "d1.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    pair = lf.d1_pair(path)
    check(pair is not None and pair["pair"] == "log10_dist_river + log10_dist_coast"
          and abs(pair["coef_sum"] + 0.55) < 1e-12,
          f"d1_pair: most negative sum from the brief's formula only ({pair})")


def same_population(cells: Path, lofo_cells: Path, tag: str, feature: str) -> bool:
    """The minus-feature cell has cells_w7's points and model specs, and no model saw the feature."""
    cell_minus = np.load(lofo_cells / f"{tag}.npz", allow_pickle=True)
    cell_base = np.load(cells / f"{tag}.npz", allow_pickle=True)
    with cell_minus, cell_base:
        models_minus = pd.read_csv(lofo_cells / f"{tag}_models.csv")
        models_base = pd.read_csv(cells / f"{tag}_models.csv")
        return (np.array_equal(cell_minus["point_id"], cell_base["point_id"])
                and np.array_equal(cell_minus["point_id_id"], cell_base["point_id_id"])
                and (models_minus.kind.to_numpy() == models_base.kind.to_numpy()).all()
                and (models_minus.seed.to_numpy() == models_base.seed.to_numpy()).all()
                and not models_minus.features.str.contains(rf"(?:^|;){feature}(?:;|$)").any())


def test_inputs(cells: Path, lofo_root: Path):
    print("3. leave-one-feature-out inputs")
    base = read_features(Path("data/processed_w7/FEATURES.txt"))
    check(base == lf.FEATURES, f"data/processed_w7/FEATURES.txt = o5_lofo.FEATURES ({base})")
    target = Path("data/processed/dataset_FL.parquet").resolve()
    for f in lf.FEATURES:
        d = Path(f"data/processed_w7_minus_{f}")
        if not d.exists():
            print(f"  skip  {d} missing")
            continue
        others = [x for x in lf.FEATURES if x != f]
        check(read_features(d / "FEATURES.txt") == others and (d / "dataset_FL.parquet").resolve() == target,
              f"{d.name}: the six other features in order; dataset link -> {target.name}")
        lofo_cells = lofo_root / f"cells_w7_minus_{f}"
        for s in (0, 1, 2):
            tag = f"FL_band_AE_s{s}"
            if not (lofo_cells / f"{tag}.npz").exists():
                continue
            check(same_population(cells, lofo_cells, tag, f),
                  f"minus_{f} seed {s}: same points and specs as cells_w7; no model saw {f}")


def test_reproduce(cells: Path, sel_root: Path):
    print("4. all 7 features reproduce the 2026-10-16 r_test")
    for s in (0, 1, 2):
        tag = f"FL_band_AE_s{s}"
        cell, Z, acc, _ = op.load_paper_cell(cells, tag)
        test = op.model_split(len(cell.models))[2]
        for variant in ("spatial_mu3_knn10", "classic"):
            m = lf.top_mask(np.load(sel_root / variant / f"{tag}_S500_theta.npy"), 500)
            got = op.hard_corr(Z[test], acc[test], m.astype(np.float32), 500, is_mask=True)
            want = float(pd.read_csv(sel_root / variant / f"{tag}_S500_result.csv").r_test.iloc[0])
            check(abs(got - want) < 1e-6, f"seed {s} {variant}: {got:+.6f} = stored {want:+.6f}")


def test_csv(path):
    print("5. written CSV")
    if not path or not Path(path).exists():
        print("  skip  no CSV")
        return
    out = pd.read_csv(path)
    out["seed"] = out.seed.astype(str)
    per = out[out.seed != "mean3"]
    mean = out[out.seed == "mean3"]
    worst = 0.0
    for _, r in mean.iterrows():
        g = per[(per.feature == r.feature) & (per.subset == r.subset) & (per.land_only == r.land_only)]
        worst = max(worst, abs(g.delta_r.mean() - r.delta_r), abs(g.r_without.mean() - r.r_without))
    check(worst < 1e-5, f"seed-mean rows = mean of the per-seed rows (max |diff| {worst:.1e})")
    pr = out[out.primary.astype(bool)]
    check(len(pr) == len(lf.FEATURES) and set(pr.feature) == set(lf.FEATURES)
          and (pr.subset == "spatial_full").all() and (~pr.land_only.astype(bool)).all(),
          f"one primary row per feature, full spatial selection, all points ({len(pr)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--lofo-root", type=Path, default=Path("data/interim/aotil"))
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    test_top_mask()
    test_verdict_and_pair(Path(tempfile.mkdtemp()))
    test_inputs(a.cells, a.lofo_root)
    test_reproduce(a.cells, a.sel_root)
    test_csv(a.csv)
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
