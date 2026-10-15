#!/usr/bin/env python3
"""
Summarise the paper-protocol OODSelect runs (oodselect_paper.py) the way the
paper reports them, per cell and variant, plus the classic-vs-spatial contrast.

Reads   <root>/<variant>/<cell>_S<budget>_{result.csv,theta.npy}
Writes  <root>/<variant>/<cell>_summary.csv     one row per budget + the full set
        <root>/<variant>/<cell>_jaccard.csv     the paper's Eq. 6 consistency
        <root>/<cell>_model_convergence.csv     the paper's model-count rule
        <root>/<cell>_comparison.csv            classic vs spatial, when both exist

WHAT EACH COLUMN IS, AND ITS SOURCE
-----------------------------------
r_test        OODSelect's subset on the held-out TEST models, the authors'
              TestSetFinder.test (probit Pearson with their eps). Fisher 95% CI
              over the test models (the paper's "Fisher z-intervals").
random_*      100 uniformly random subsets of the same size (Table 4: "we
              randomly select N subsets"): mean r, sd, and the Fisher-mean
              tanh(mean arctanh r) the authors' plotting code uses.
hard_r        The N most-misclassified points, ranked by 1 - mean correctness
              over the TEST models -- exactly generate_correlation_plots.py.
              At small N this is 0 BY CONSTRUCTION: the top-ranked points are
              wrong for every test model, so subset accuracy is constant, the
              probit of it has zero variance, and the eps-regularised r is 0.
              Read the paper's "most misclassified is near zero" with that in
              mind; the baseline only carries information once N is large
              enough to include points some test models get right.
hard_sel_r    The same ranking from the SELECTION models only. Not in the paper;
              it is this project's standing control (a ranking built from the
              models it is scored on is scored on models it has already seen).
aotil         r_test <= -0.3, the paper's threshold; the summary also prints the
              largest AoTIL budget, the quantity in the paper's Table 1.
coh_*         Coherence on the kNN graph (graph.coherence_metrics). Reported for
              the classic run too, which never sees the graph -- that is the
              baseline the spatial run is judged against.

NOT REPRODUCED FROM THE AUTHORS' PLOTTING SCRIPT: it reports the size-n point
using the selection trained at the NEXT larger size, with "error bars" from
permute_top_k, which permutes values within the top-n and so leaves the top-n
set unchanged. Here each budget is reported from its own selection, with a
Fisher interval over test models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op  # noqa: E402

N_RANDOM = 100


def fisher_mean(rs) -> float:
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999))))) if len(rs) else np.nan


def top_mask(score: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def normalised_jaccard(masks: list[np.ndarray], n_sim: int = 200, seed: int = 0) -> dict:
    """Eq. 6: (J_bar - J_min) / (J_max - J_min) over consecutive budgets.

    J_min: random selections of the same sizes. J_max: a nested sequence of the
    same sizes (z_k subset of z_k+1), for which J(z_k, z_k+1) = |z_k| / |z_k+1|.
    """
    masks = sorted(masks, key=lambda m: m.sum())
    if len(masks) < 2:
        return dict(J_bar=np.nan, J_min=np.nan, J_max=np.nan, J_norm=np.nan, T=len(masks))
    d = len(masks[0])
    sizes = [int(m.sum()) for m in masks]

    def J(a, b):
        u = (a | b).sum()
        return (a & b).sum() / u if u else np.nan

    j_bar = np.mean([J(masks[i], masks[i + 1]) for i in range(len(masks) - 1)])
    j_max = np.mean([sizes[i] / sizes[i + 1] for i in range(len(sizes) - 1)])
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sim):
        rm = [np.isin(np.arange(d), rng.choice(d, s, replace=False)) for s in sizes]
        sims.append(np.mean([J(rm[i], rm[i + 1]) for i in range(len(rm) - 1)]))
    j_min = float(np.mean(sims))
    return dict(J_bar=float(j_bar), J_min=j_min, J_max=float(j_max),
                J_norm=float((j_bar - j_min) / (j_max - j_min)) if j_max > j_min else np.nan,
                T=len(masks), sizes=" ".join(map(str, sizes)))


def model_convergence(acc_id, Z, step: int = 10, n_orders: int = 25, seed: int = 0):
    """The paper's model-count rule, on the full OOD set: add models until the
    ID-OOD correlation changes by less than 1%. Averaged over random orders."""
    from scipy.stats import norm
    x = norm.ppf(np.clip(acc_id, op.EPS, 1 - op.EPS))
    a = norm.ppf(np.clip(Z.mean(1), op.EPS, 1 - op.EPS))
    N = len(x)
    sizes = list(range(step, N + 1, step))
    if sizes[-1] != N:
        sizes.append(N)
    rng = np.random.default_rng(seed)
    R = np.full((n_orders, len(sizes)), np.nan)
    for o in range(n_orders):
        perm = rng.permutation(N)
        for j, n in enumerate(sizes):
            if n >= 4:
                R[o, j] = np.corrcoef(x[perm[:n]], a[perm[:n]])[0, 1]
    r_mean = np.nanmean(R, 0)
    rel = np.r_[np.nan, np.abs(np.diff(r_mean)) / np.maximum(np.abs(r_mean[:-1]), 1e-12)]
    df = pd.DataFrame(dict(n_models=sizes, r_mean=r_mean, r_sd=np.nanstd(R, 0), rel_change=rel))
    # first size from which every later step moves the correlation by < 1%
    ok = (df.rel_change < 0.01).to_numpy()
    n_needed = np.nan
    for j in range(1, len(ok)):
        if ok[j:].all():
            n_needed = sizes[j - 1]
            break
    df["n_needed"] = n_needed
    df["r_all_models"] = float(np.corrcoef(x, a)[0, 1])
    return df


def summarise(root: Path, variant: str, tag: str, cells_dir: str, knn: int | None):
    vdir = root / variant
    rows = [pd.read_csv(p) for p in sorted(vdir.glob(f"{tag}_S*_result.csv"))]
    if not rows:
        return None, {}
    res = pd.concat(rows, ignore_index=True).sort_values("budget").reset_index(drop=True)
    knn = int(res.knn.iloc[0]) if knn is None else knn

    cell, Z, acc, info = op.load_paper_cell(cells_dir, tag)
    tr, va, te = op.model_split(info["n_models"])
    d = info["d"]
    Zte, ate = Z[te], acc[te]
    A = None
    if cell.has_spatial():
        import graph
        A = graph.build_adjacency(cell.lon, cell.lat, k=knn)

    def coh(mask):
        if A is None:
            return {}
        import graph
        cm = graph.coherence_metrics(mask, A, knn, tiles=cell.tile_50km, blocks=cell.block_id)
        return cm

    rng = np.random.default_rng(0)
    hard_score = 1.0 - Zte.mean(0)                 # generate_correlation_plots.py
    hard_sel_score = 1.0 - Z[tr].mean(0)
    masks, out = {}, []
    for _, r in res.iterrows():
        S = int(r.budget)
        theta = np.load(vdir / f"{tag}_S{S}_theta.npy")
        m = top_mask(theta, S)
        masks[S] = m
        rr = [op.hard_corr(Zte, ate, top_mask(rng.random(d), S), S, is_mask=True)
              for _ in range(N_RANDOM)]
        rand_pur = [coh(top_mask(rng.random(d), S)).get("purity", np.nan) for _ in range(5)]
        mh = top_mask(hard_score, S)
        mhs = top_mask(hard_sel_score, S)
        row = r.to_dict()
        row.update(
            random_r_mean=float(np.mean(rr)), random_r_sd=float(np.std(rr)),
            random_r_fisher=fisher_mean(rr), random_purity=float(np.nanmean(rand_pur)),
            hard_r=op.hard_corr(Zte, ate, mh, S, is_mask=True),
            hard_sel_r=op.hard_corr(Zte, ate, mhs, S, is_mask=True),
            hard_purity=coh(mh).get("purity", np.nan),
            overlap_hard=float((m & mh).sum() / S),
        )
        out.append(row)
    full = dict(cell=tag, variant=variant, budget=d, d=d, frac=1.0,
                r_test=op.hard_corr(Zte, ate, np.ones(d, np.float32), d, is_mask=True),
                random_r_mean=np.nan, hard_r=np.nan, n_test=len(te))
    full["r_test_ci_lo"], full["r_test_ci_hi"] = op.fisher_ci(full["r_test"], len(te))
    summ = pd.DataFrame(out + [full])
    summ.to_csv(vdir / f"{tag}_summary.csv", index=False)

    jac = normalised_jaccard(list(masks.values()))
    pd.DataFrame([dict(cell=tag, variant=variant, **jac)]).to_csv(
        vdir / f"{tag}_jaccard.csv", index=False)

    aot = res[res.r_test <= -0.3]
    largest = int(aot.budget.max()) if len(aot) else 0
    print(f"\n=== {tag} | {variant} | d = {d} | test models {len(te)} ===")
    cols = ["budget", "frac", "r_test", "r_test_ci_lo", "r_test_ci_hi", "spearman_test",
            "random_r_mean", "hard_r", "hard_sel_r"]
    cols += [c for c in ("coh_purity", "coh_n_cc", "coh_largest_cc_frac", "coh_tiles")
             if c in summ]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:+.3f}".format):
        print(summ[cols].to_string(index=False))
    print(f"  largest AoTIL subset (r_test <= -0.3): {largest} of {d} "
          f"({largest / d:.1%})")
    print(f"  normalised Jaccard across budgets (Eq. 6): {jac['J_norm']:.3f} "
          f"(J_bar {jac['J_bar']:.3f}, random {jac['J_min']:.3f}, nested {jac['J_max']:.3f})")
    missing = [b for b in op.budgets_for(d) if b not in set(res.budget)]
    if missing:
        print(f"  MISSING budgets (task failed or still running): {missing}")
    return summ, masks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results/<date>/oodselect_paper")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="default: every subdirectory of --root")
    a = ap.parse_args()
    root = Path(a.root)
    variants = a.variants or sorted(p.name for p in root.iterdir() if p.is_dir())

    for tag in a.tags:
        cell, Z, acc, info = op.load_paper_cell(a.cells, tag)
        conv = model_convergence(acc, Z)
        conv.to_csv(root / f"{tag}_model_convergence.csv", index=False)
        print(f"\n##### {tag}: {info['n_models']} models, {info['d']} OOD points; "
              f"ID acc {acc.min():.3f}-{acc.max():.3f} (spread {acc.max() - acc.min():.3f}); "
              f"full-set ID->OOD r (all models) {conv.r_all_models.iloc[0]:+.3f}; "
              f"paper's <1% model-count rule met at {conv.n_needed.iloc[0]} models")

        summaries, masks = {}, {}
        for v in variants:
            s, m = summarise(root, v, tag, a.cells, None)
            if s is not None:
                summaries[v], masks[v] = s, m

        if "classic" in summaries and len(summaries) > 1:
            base = summaries["classic"].set_index("budget")
            rows = []
            for v, s in summaries.items():
                if v == "classic":
                    continue
                s = s.set_index("budget")
                for b in sorted(set(base.index) & set(s.index)):
                    if b not in masks["classic"] or b not in masks[v]:
                        continue
                    mc, mv = masks["classic"][b], masks[v][b]
                    row = dict(cell=tag, variant=v, budget=b, frac=b / info["d"],
                               r_test_classic=base.at[b, "r_test"],
                               r_test_spatial=s.at[b, "r_test"],
                               cost_of_coherence=s.at[b, "r_test"] - base.at[b, "r_test"],
                               jaccard_masks=float((mc & mv).sum() / (mc | mv).sum()),
                               random_r=base.at[b, "random_r_mean"],
                               hard_r=base.at[b, "hard_r"])
                    for c in ("coh_purity", "coh_n_cc", "coh_largest_cc_frac", "coh_tiles"):
                        if c in base and c in s:
                            row[f"{c}_classic"] = base.at[b, c]
                            row[f"{c}_spatial"] = s.at[b, c]
                    rows.append(row)
            if rows:
                comp = pd.DataFrame(rows)
                comp.to_csv(root / f"{tag}_comparison.csv", index=False)
                print(f"\n=== {tag}: classic vs spatial ===")
                show = [c for c in ("variant", "budget", "r_test_classic", "r_test_spatial",
                                    "cost_of_coherence", "coh_purity_classic",
                                    "coh_purity_spatial", "coh_n_cc_classic",
                                    "coh_n_cc_spatial", "coh_tiles_classic",
                                    "coh_tiles_spatial", "jaccard_masks") if c in comp]
                with pd.option_context("display.width", 220, "display.max_columns", 30,
                                       "display.float_format", "{:+.3f}".format):
                    print(comp[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
