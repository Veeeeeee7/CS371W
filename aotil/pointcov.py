#!/usr/bin/env python3
"""
E0 -- does any coherent inverse region exist? Run this BEFORE the mu sweep.

WHY THIS IS CHEAP AND DECISIVE
------------------------------
Covariance is additive over points:

    Cov_m(x_m, mean_{i in S} S_mi) = (1/|S|) sum_{i in S} Cov_m(x_m, S_mi)

so the per-point map c_i = Cov_m(x_m, S_mi), computed once on the SELECTION
models, already says which points pull the correlation negative -- no
optimisation needed. If c_i is spatial white noise, then no spatial smoothing
weight can assemble a coherent negative region out of it, and the whole mu sweep
is a week spent confirming that. Moran's I of c_i against a permutation null is
the test.

Pearson r also divides by the spread of the subset mean across models, which is
NOT additive, so the magnitudes here differ from what the optimiser reaches. The
SIGN is set by the covariance, and the sign is what this answers.

The map is also the first explanation object in its own right: c_i can be
regressed on GIS covariates directly, without any subset having been chosen.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph                                  # noqa: E402
from cells import load_cell, cell_tag         # noqa: E402


def point_covariance(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    """c_i = Cov_m(x_m, S_mi) across models, one value per OOD point."""
    x = np.asarray(x, float)
    S = np.asarray(S, float)
    xc = x - x.mean()
    return (xc @ (S - S.mean(axis=0, keepdims=True))) / max(len(x) - 1, 1)


def region_slope(c: np.ndarray, mask: np.ndarray, x: np.ndarray) -> float:
    """Slope of ID performance against subset mean for a given region.

    mean(c_i in region) / Var(x) -- the additive-covariance shortcut, useful for
    ranking candidate regions without re-running the optimiser on each.
    """
    v = np.var(np.asarray(x, float), ddof=1)
    return float(np.mean(c[mask]) / v) if v > 0 else np.nan


def analyse(cell, knn: int, n_perm: int, seed: int, graph_kind: str = "knn") -> tuple:
    cell.require_spatial()
    sel, _, _ = cell.splits(seed)["random-split"]
    c = point_covariance(cell.x[sel], cell.S_ood[sel])

    groups = None
    if graph_kind == "block":
        groups = cell.block_id
    elif graph_kind == "catchment":
        groups = cell.meta_comid
    elif graph_kind == "provenance":
        groups = cell.meta_source_cit if cell.meta_source_cit is not None else cell.county_fips
    A = graph.build_adjacency(cell.lon, cell.lat, k=knn, groups=groups)

    point = graph.morans_i_null(c, A, n_perm=n_perm, seed=seed)

    df = pd.DataFrame({
        "c": c, "lon": cell.lon, "lat": cell.lat,
        "block_id": cell.block_id if cell.block_id is not None else -1,
        "tile_50km": cell.tile_50km if cell.tile_50km is not None else "",
        "label": cell.y_ood,
    })

    # Block-level view: a region has to be more than a handful of adjacent
    # points, and blocks are the 10 km unit the sampler already works in.
    blk = df.groupby("block_id").c.agg(["mean", "size"]).reset_index()
    blk_neg = int((blk["mean"] < 0).sum())

    summary = dict(
        cell=cell.tag, graph=graph_kind, knn=knn,
        n_points=len(c), n_sel_models=len(sel),
        c_mean=float(c.mean()), c_sd=float(c.std()),
        frac_negative=float((c < 0).mean()),
        moran_I=point["moran_I"], null_mean=point["null_mean"],
        null_sd=point["null_sd"], z=point["z"], p_perm=point["p_perm"],
        n_blocks=int(len(blk)), n_blocks_negative=blk_neg,
        frac_blocks_negative=float(blk_neg / max(len(blk), 1)),
    )
    return df, summary


def verdict(summary: dict) -> str:
    """The go/no-go, stated in the terms the plan's outcome table uses."""
    if not np.isfinite(summary["z"]):
        return "UNDECIDED (Moran's I could not be computed)"
    if summary["p_perm"] < 0.05 and summary["moran_I"] > 0:
        return ("GO -- the per-point covariance is spatially clustered, so a "
                "coherent inverse region can exist. Proceed to the mu sweep.")
    return ("NO-GO for regions -- the covariance map is inside its permutation "
            "null, i.e. spatial white noise. No mu > 0 will assemble a region "
            "from it. This is outcome C in the plan: report the tradeoff curve "
            "and run the characterisation on the mu=0 subsets, and do not spend "
            "a week on the sweep expecting regions.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="data/interim/aotil/cells")
    ap.add_argument("--out", default="results/aotil")
    ap.add_argument("--state", default="FL")
    ap.add_argument("--protocol", default="band")
    ap.add_argument("--train-dom", default="AE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--graph", default="knn",
                    choices=["knn", "block", "catchment", "provenance"])
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()

    tag = cell_tag(a.state, a.protocol, a.train_dom, a.seed)
    cell = load_cell(a.cells, tag)
    df, summary = analyse(cell, a.knn, a.n_perm, a.seed, a.graph)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{tag}_{a.graph}_pointcov.parquet", index=False)
    pd.DataFrame([summary]).to_csv(out / f"{tag}_{a.graph}_pointcov_summary.csv",
                                   index=False)

    print(f"=== E0 point covariance: {tag} (graph={a.graph}, k={a.knn}) ===")
    for k in ["n_points", "n_sel_models", "c_mean", "c_sd", "frac_negative",
              "moran_I", "null_mean", "null_sd", "z", "p_perm",
              "n_blocks", "n_blocks_negative", "frac_blocks_negative"]:
        v = summary[k]
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
    print(f"\n{verdict(summary)}")
    print(f"\nwrote {out}/{tag}_{a.graph}_pointcov.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
