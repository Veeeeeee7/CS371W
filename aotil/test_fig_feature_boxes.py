#!/usr/bin/env python3
"""
Checks for fig_feature_boxes.py. Run from the repo root:

    .venv/bin/python aotil/test_fig_feature_boxes.py

1. figure_sets: the three sets are zoo.cell_data's training rows, ID test half and OOD half.
2. top_classes: the k most common classes, most common first, ties by the smaller code.
3. class_shares: each listed class's share of the non-missing values; "other" takes the rest;
   the shares of a full list sum to 1.
4. draw writes a PDF and a PNG on a synthetic dataset.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_feature_boxes as fb  # noqa: E402
import zoo  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def synthetic(seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(20):
        for dom, lab, proto, n in (("AE", 1, None, 20), ("AE", 0, "band", 20),
                                   ("A", 1, None, 12), ("A", 0, "band", 12)):
            rows += [dict(tile_50km=f"{t // 4}_{t % 4}", domain=dom, label=lab, neg_protocol=proto)
                     for _ in range(n)]
    df = pd.DataFrame(rows)
    n = len(df)
    df["point_id"] = [f"P{i}" for i in range(n)]
    for col in ("slope", "elev_sd_r300", "hand", "wetness", "log10_dist_river", "log10_dist_coast"):
        df[col] = rng.gamma(2.0, 1.0, n)
    df["landcover"] = rng.choice([90.0, 42.0, 81.0, 95.0, 21.0, 82.0, 22.0, 52.0, 71.0], n,
                                 p=[.3, .2, .1, .1, .08, .07, .06, .05, .04])
    df["soil_floodfreq"] = np.where(rng.random(n) < 0.05, np.nan, rng.integers(0, 6, n))
    df["soil_fluvclass"] = np.where(rng.random(n) < 0.05, np.nan, (rng.random(n) < 0.1) * 1.0)
    return df


def check_sets():
    print("1. figure_sets")
    df = synthetic()
    groups = fb.figure_sets(df, 0)
    cell = zoo.cell_data(df, ["slope", "hand"], "band", "AE", 0)
    check(groups["Train"].index.equals(cell["tr"].index)
          and groups["ID test"].index.equals(cell["id_te"].index)
          and groups["OOD"].index.equals(cell["ood_te"].index),
          f"the sets are cell_data's halves: {[len(g) for g in groups.values()]}")


def check_classes():
    print("2. top_classes and 3. class_shares")
    values = pd.Series([90, 90, 90, 42, 42, 81, 81, 21, 22, 23, np.nan])
    check(fb.top_classes(values, 3) == [90, 42, 81], "most common first, ties by the smaller code")
    shares = fb.class_shares(values, [90, 42, "other"])
    check(np.allclose(shares, [0.3, 0.2, 0.5]) and np.isclose(sum(shares), 1.0),
          f"shares over the 10 non-missing values, other takes the rest: {shares}")
    check(fb.class_shares(pd.Series([1.0, 0.0, np.nan, 0.0]), [1]) == [1 / 3],
          "missing values are left out")


def check_draw():
    print("4. draw")
    df = synthetic()
    groups = fb.figure_sets(df, 0)
    with tempfile.TemporaryDirectory() as tmp:
        out, png = Path(tmp) / "f.pdf", Path(tmp) / "f.png"
        fb.draw(groups, fb.top_classes(df.landcover), out, png)
        check(out.stat().st_size > 5_000 and png.stat().st_size > 5_000, "PDF and PNG written")


def main():
    check_sets()
    check_classes()
    check_draw()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
