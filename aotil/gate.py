#!/usr/bin/env python3
"""The AoTL gate: is there a line to invert?

Reports, per cell:
  * the spread of ID performance across the model population -- if every model
    lands in the same place, the correlation OODSelect minimises is undefined;
  * the aggregate ID->OOD correlation on the FULL OOD set. OODSelect looks for
    subsets where this goes negative. If it is not positive to begin with, the
    premise of accuracy-on-the-inverse-line does not hold here and the method
    has nothing to do.

Both the continuous per-example score and thresholded accuracy are reported, so
the verdict can be checked against the scoring decision.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.stats import norm, pearsonr, spearmanr


def convergence(x: np.ndarray, a: np.ndarray, sizes, n_draw=25, seed=0) -> pd.DataFrame:
    """The paper's model-count rule: add models until the correlation stops
    moving by more than 1%.

    Salaudeen et al. ran up to 4,200 models but state the criterion as "enough
    that adding more changes the correlation by less than 1%", and report that
    being about ten models for VLCS and six for WILDS-Camelyon. This table finds
    the number for a cell instead of guessing it: fitting 300 models when 120
    have converged is compute spent on nothing. The selection itself still
    benefits from more models, so this bounds the CORRELATION estimate, not the
    population you want for a sweep.
    """
    rng = np.random.default_rng(seed)
    N = len(x)
    full = pearsonr(x, a)[0]
    rows = []
    # With a small population every subsample is a large fraction of it, so the
    # table is mostly resampling noise and "converged" flickers. Below this the
    # answer is always "add models".
    if N < 40:
        return pd.DataFrame([dict(n_models=N, r_mean=float(full), r_sd=np.nan,
                                  abs_delta_vs_full=np.nan, converged=False,
                                  note="population too small to assess convergence")])
    for n in sizes:
        if n >= N or n < 4:
            continue   # n == N is one possible draw: sd 0, delta 0, no information
        rs = [pearsonr(x[i], a[i])[0]
              for i in (rng.choice(N, n, replace=False) for _ in range(n_draw))]
        rows.append(dict(n_models=n, r_mean=float(np.mean(rs)),
                         r_sd=float(np.std(rs)),
                         abs_delta_vs_full=float(abs(np.mean(rs) - full)),
                         converged=bool(abs(np.mean(rs) - full) < 0.01 * abs(full))))
    return pd.DataFrame(rows)

def probit(a, eps=1e-6):
    return norm.ppf(np.clip(a, eps, 1 - eps))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cells", default="data/interim/aotil/cells")
ap.add_argument("--sizes", default="25,50,75,100,150,225,300")
args = ap.parse_args()
SIZES = [int(v) for v in args.sizes.split(",")]

rows, conv = [], []
for f in sorted(glob.glob(f"{args.cells}/*_models.csv")):
    tag = os.path.basename(f).replace("_models.csv", "")
    m = pd.read_csv(f)
    for metric, lab in (("score", "prob"), ("acc", "acc@.5"), ("auc", "auc")):
        xi, yo = m[f"{metric}_id"].to_numpy(), m[f"{metric}_ood"].to_numpy()
        if lab != "auc":
            xi, yo = probit(xi), probit(yo)
        r, p = pearsonr(xi, yo)
        if lab == "prob":
            c = convergence(xi, yo, SIZES)
            c.insert(0, "cell", tag)
            conv.append(c)
        rows.append(dict(cell=tag, metric=lab, n_models=len(m),
                         id_min=m[f"{metric}_id"].min(), id_max=m[f"{metric}_id"].max(),
                         id_spread=m[f"{metric}_id"].max() - m[f"{metric}_id"].min(),
                         id_sd=m[f"{metric}_id"].std(),
                         pearson_probit=r, p_value=p))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== ID spread and aggregate ID->OOD correlation (full OOD set) ===")
print(r.assign(**{k: r[k].round(4) for k in ["id_min","id_max","id_spread","id_sd","pearson_probit"]},
               p_value=r.p_value.map(lambda v: f"{v:.2e}")).to_string(index=False))

print("\n=== GATE VERDICT (continuous per-example score) ===")
for _, g in r[r.metric == "prob"].iterrows():
    spread_ok = g.id_spread >= 0.03
    slope_ok = (g.pearson_probit > 0) and (g.p_value < 0.05)
    verdict = "OPEN" if (spread_ok and slope_ok) else "CLOSED"
    print(f"  {g.cell:22s} spread {g.id_spread:.3f} {'ok' if spread_ok else 'TOO NARROW':11s}"
          f" | r = {g.pearson_probit:+.3f} {'positive' if slope_ok else 'NOT POSITIVE':13s}"
          f" -> {verdict}")

if conv:
    cv = pd.concat(conv, ignore_index=True)
    print("\n=== model-count convergence (the paper's <1% rule) ===")
    print(cv.round(4).to_string(index=False))
    enough = (cv[cv.converged].groupby("cell").n_models.min()
              .rename("models_needed").reset_index())
    if len(enough):
        print("\nSmallest population within 1% of the full-population correlation:")
        print(enough.to_string(index=False))
        print("Past this point more models buy precision in the SELECTION, not "
              "in the correlation -- size the sweep on that, not on the paper's 4,200.")
    elif "note" in cv and cv.note.notna().any():
        print("\nPopulation too small to assess convergence. This table needs "
              "at least ~40 models\nbefore it says anything: below that every "
              "subsample is most of the population,\nso the spread collapses "
              "and 'converged' is an artifact of the draw, not a finding.")
    else:
        print("\nNo subsample is within 1% of the full population: the correlation "
              "is still moving, so ADD MODELS before trusting a sweep.")
