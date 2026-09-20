#!/usr/bin/env python3
"""
What IS the inverse-line subset?

This is the question OODSelect leaves open: the method locates subsets on which
higher in-distribution performance predicts lower out-of-distribution
performance, but its benchmarks carry no ground truth about why, so the subsets
cannot be described. Here they can, because the provenance of every label is
recorded and was quarantined from the features -- the model never saw it, so
any enrichment in the selected subset is a property of the subset rather than
something the model was told.

Reconstruction: the tile split in zoo.py is a pure function of (state, seed), so
the evaluation frame is rebuilt exactly rather than stored.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from oodselect import solve as select, probit

import zoo  # split_tiles / subset / load, so the frame matches byte for byte


def describe(df_sel: pd.DataFrame, df_rest: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df_sel:
            continue
        a, b = df_sel[c], df_rest[c]
        if pd.api.types.is_numeric_dtype(a):
            sd = pd.concat([a, b]).std()
            if not np.isfinite(sd) or sd == 0:
                continue
            rows.append(dict(variable=c, selected=a.mean(), rest=b.mean(),
                             std_diff=(a.mean() - b.mean()) / sd))
        else:
            top = pd.concat([a, b]).value_counts().head(4).index
            for lev in top:
                pa, pb = (a == lev).mean(), (b == lev).mean()
                rows.append(dict(variable=f"{c}={lev}", selected=pa, rest=pb,
                                 std_diff=(pa - pb) / max(np.sqrt(pb * (1 - pb)), 1e-6)))
    return (pd.DataFrame(rows)
            .assign(abs_d=lambda d: d.std_diff.abs())
            .sort_values("abs_d", ascending=False).drop(columns="abs_d"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="FL")
    ap.add_argument("--protocol", default="band")
    ap.add_argument("--train-dom", default="AE")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", default="data/interim/aotil/cells")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="smoothness weight; characterise the region, not the pilot subset")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--data", default="data/processed")
    a = ap.parse_args()

    tag = f"{a.state}_{a.protocol}_{a.train_dom}_s{a.seed}"
    eval_dom = "AE" if a.train_dom == "A" else "A"

    df, feats = zoo.load(a.state, Path(a.data))
    _, te_tiles = zoo.split_tiles(df, a.seed)
    ood = zoo.subset(df, eval_dom, a.protocol)
    ood = ood[ood[zoo.TILE].isin(te_tiles)].reset_index(drop=True)

    z = np.load(f"{a.cells}/{tag}.npz", allow_pickle=True)
    S = z["S_ood"]
    assert len(ood) == S.shape[1], (len(ood), S.shape)
    m = pd.read_csv(f"{a.cells}/{tag}_models.csv")
    x = probit(m.score_id.to_numpy())

    rng = np.random.default_rng(a.seed)
    i_sel = rng.permutation(len(m))[:int(.6 * len(m))]
    k = max(20, int(S.shape[1] * a.frac))
    L = None
    if a.mu > 0:
        import graph
        L = graph.laplacian(graph.build_adjacency(
            z["lon"], z["lat"], k=a.knn))
    masks, _ = select(x[i_sel], S[i_sel], k, L=L, mus=a.mu, knn=a.knn, seed=a.seed)
    mask = masks[0]

    print(f"### {tag}  OOD domain = Zone {eval_dom}  "
          f"subset = {mask.sum():,} of {len(ood):,} ({a.frac:.0%})")
    print(f"    tiles touched: {ood.loc[mask, zoo.TILE].nunique()} of {ood[zoo.TILE].nunique()}")
    print(f"    positives in subset: {ood.loc[mask,'label'].mean():.1%} "
          f"vs {ood.loc[~mask,'label'].mean():.1%} outside\n")

    cols = ["meta_panel_scale", "meta_bfe_km_per_km2", "meta_static_bfe",
            "meta_poly_area_km2", "meta_dist_to_sfha_m", "meta_elev",
            "meta_dist_to_river_m", "meta_dist_to_coast_m", "meta_upstream_area_km2",
            "hand", "impervious", "water_frac_900m", "slope", "log10_upa",
            "meta_study_typ", "meta_zone_subty", "meta_study_regime", "meta_panel_typ"]
    d = describe(ood[mask], ood[~mask], cols)
    print(d.head(14).round(3).to_string(index=False))
    print("\n  std_diff = (selected - rest) / pooled SD. |d| > 0.5 is a large shift.")
    print("  meta_ columns were quarantined from the feature matrix: the models never saw them.")


if __name__ == "__main__":
    main()
