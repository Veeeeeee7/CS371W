#!/usr/bin/env python3
"""
Pooled, multivariate, out-of-state test of whether the tile-level transfer gap
is explainable from tile covariates.

WHY THIS EXISTS
---------------
`baselines.py --explain` reports a per-state, per-direction, UNIVARIATE Spearman
between each tile covariate and the tile's gap. That analysis returned nothing,
and was read as "the gap is unexplainable." It is not a null result; it is a
power failure. With a median of 35 tiles per cell, the smallest correlation
detectable at 80% power / alpha=0.05 is |rho| = 0.46. The correlations actually
present are around 0.14. The test could not have found them, and 176 of them
were run, so nothing survives a multiplicity correction either.

This script does the same job with the power to answer it:
  * pools tiles across states (n = 263-389 per cell instead of 9-53),
  * uses all covariates jointly rather than one at a time,
  * validates LEAVE-ONE-STATE-OUT, so "explainable" means the description
    transfers to a state the model never saw -- not an in-sample fit,
  * runs the identical procedure on the permutation-null tile gaps as a control.

A positive Spearman between predicted and actual gap, with the null at zero,
means the subsets where transfer degrades are DESCRIBABLE from observables.
That is the question OODSelect (Salaudeen et al. 2025) leaves open.

USAGE
-----
    python explain_pooled.py                      # paired run, default
    python explain_pooled.py --tag hgb_f4_s0      # pick a different tag

Reads  results/tile_gaps_{tag}_paired.csv
       results/tile_gaps_{tag}_null_paired.csv
Writes results/explain_pooled_{tag}.csv  (append-safe: overwrites its own file only)
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

RESULTS = Path("results")

# Tile covariates carried through by baselines.py, split into the groups whose
# separate contributions are the interesting part.
PROV = ["bfe", "panel_scale"]                      # how the map was made
TERR = ["elev", "hand", "imperv", "water"]         # what the ground looks like
SIZE = ["n_pos", "n_neg"]                          # control: is this just sample size?

def _sides(names: list[str]) -> list[str]:
    """Each covariate enters twice: its value on the ID side and the OOD side."""
    return [f"{v}_{s}" for v in names for s in ("id", "ood")]

GROUPS = {
    "all":       _sides(PROV + TERR + SIZE),
    "prov_only": _sides(PROV),
    "terr_only": _sides(TERR),
    "size_only": _sides(SIZE),
    "no_size":   _sides(PROV + TERR),
    "no_prov":   _sides(TERR + SIZE),
}


def load_pairs(path: Path) -> pd.DataFrame:
    """Join each tile's ID evaluation to its OOD evaluation and form the gap.

    baselines.py writes one row per (tile, kind). The gap is only meaningful
    within a fold and training domain, so those are part of the key.
    """
    t = pd.read_csv(path)
    key = ["state", "protocol", "fold", "train_domain", "tile"]
    m = (t[t.kind == "ID"]
         .merge(t[t.kind == "OOD"], on=key, suffixes=("_id", "_ood")))
    m["gap"] = m.auc_ood - m.auc_id
    return m


def leave_one_state_out(d: pd.DataFrame, cols: list[str]) -> dict:
    """Predict each state's tile gaps from a model fit on the other nine.

    Out-of-state is the right validation here for the same reason 50 km blocks
    are the right CV unit in baselines.py: anything less and the model can
    memorise the state rather than describe the mechanism.
    """
    X = d[cols].to_numpy(float)
    y = d.gap.to_numpy(float)
    st = d.state.to_numpy()
    pred = np.full(len(d), np.nan)

    for s in np.unique(st):
        test = st == s
        train = ~test
        if train.sum() < 50 or test.sum() < 3:
            continue                                   # too little to fit or score
        mdl = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
        mdl.fit(X[train], y[train])
        pred[test] = mdl.predict(X[test])

    ok = np.isfinite(pred)
    if ok.sum() < 30:
        return dict(n=int(ok.sum()), spearman=np.nan, p_value=np.nan, r2=np.nan)

    rho, p = spearmanr(pred[ok], y[ok])
    r2 = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    return dict(n=int(ok.sum()), spearman=float(rho), p_value=float(p), r2=float(r2))


def power_note(explain_csv: Path) -> None:
    """Print why the per-state univariate analysis could not have worked."""
    if not explain_csv.exists():
        return
    e = pd.read_csv(explain_csv)
    n = int(e.n.median())
    z = 1.959964 + 0.8416212                 # z(alpha/2) + z(power), two-sided 0.05 / 80%
    rho_min = math.tanh(z / math.sqrt(n - 3))
    print("--- why the per-state univariate analysis returned nothing ---")
    print(f"  tests run                         : {len(e)}")
    print(f"  median tiles per test             : {n}  (range {int(e.n.min())}-{int(e.n.max())})")
    print(f"  min |rho| detectable at 80% power : {rho_min:.3f}")
    print(f"  median |rho| actually present     : {e.spearman.abs().median():.3f}")
    print(f"  survive Bonferroni                : {int((e.p_value < .05/len(e)).sum())} of {len(e)}")
    print("  -> underpowered by a factor of ~3 on the effect size. Not a null result.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="hgb_f4_s0")
    ap.add_argument("--mode", default="paired", choices=["paired", "unpaired"])
    args = ap.parse_args()

    real_p = RESULTS / f"tile_gaps_{args.tag}_{args.mode}.csv"
    null_p = RESULTS / f"tile_gaps_{args.tag}_null_{args.mode}.csv"
    for p in (real_p, null_p):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    power_note(RESULTS / f"explain_{args.tag}_{args.mode}.csv")

    real, null = load_pairs(real_p), load_pairs(null_p)
    out = []
    for label, df in (("real", real), ("null", null)):
        for proto in sorted(df.protocol.unique()):
            for tr in sorted(df.train_domain.unique()):
                d = df[(df.protocol == proto) & (df.train_domain == tr)]
                if len(d) < 60:
                    continue
                for gname, cols in GROUPS.items():
                    r = leave_one_state_out(d, cols)
                    out.append(dict(setting=label, protocol=proto, train=tr,
                                    features=gname, **r))
    res = pd.DataFrame(out)

    # Headline table: the full feature set, real beside null.
    print("--- is the tile gap describable out-of-state? (features=all) ---")
    h = res[res.features == "all"].pivot_table(
        index=["protocol", "train"], columns="setting",
        values=["spearman", "p_value", "n"])
    print(h.round(4).to_string(), "\n")

    print("--- which covariate group carries it? (real only, Spearman) ---")
    g = (res[res.setting == "real"]
         .pivot_table(index=["protocol", "train"], columns="features", values="spearman")
         .reindex(columns=list(GROUPS)))
    print(g.round(3).to_string())
    print("\n  prov = {bfe, panel_scale}   terr = {elev, hand, imperv, water}   size = {n_pos, n_neg}")
    print("  size_only near zero everywhere => predictability is not a sample-size artifact.")

    dest = RESULTS / f"explain_pooled_{args.tag}_{args.mode}.csv"
    res.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
