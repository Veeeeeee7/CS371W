#!/usr/bin/env python3
"""Summarise the paper-protocol OODSelect runs (oodselect_paper.py) per cell and variant.

Reads   <root>/<variant>/<cell>_S<budget>_{result.csv,theta.npy}, and the cell from --cells
Writes  <root>/<variant>/<cell>_summary.csv   one row per budget, plus the full OOD set
        <root>/<variant>/<cell>_jaccard.csv   consistency across budgets (the paper's Eq. 6)
        <root>/<cell>_model_convergence.csv   the paper's model-count rule
        <root>/<cell>_comparison.csv          classic vs spatial per budget, when both exist
Control variants (_null<N>, _trees2others, _others2trees) are scored on the split their
result.csv records and are left out of the comparison, which needs one shared split.
Unlike the authors' plotting script, which plots size n from the selection trained at the
next larger size, each budget is reported from its own selection, with a Fisher interval
over the test models.

Example: python aotil/report_paper.py --root results/<date>/oodselect_paper \\
             --cells data/interim/aotil/cells --tags FL_band_AE_s0 FL_band_A_s0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import oodselect_paper as op  # noqa: E402

N_RANDOM = 100        # random subsets per budget (the paper's Table 4)
N_TIE_BREAKS = 100    # random tie orders for the most-misclassified baselines
N_RANDOM_PURITY = 5   # random subsets per budget for the purity baseline
COHERENCE_COLUMNS = ("coh_purity", "coh_n_cc", "coh_largest_cc_frac", "coh_tiles")


def fisher_mean(rs) -> float:
    """tanh(mean(arctanh r)) over the finite r, as the authors' plotting code averages."""
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))


def top_mask(score: np.ndarray, k: int) -> np.ndarray:
    """Mask of the k highest scores; ties go to the lower index."""
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def tie_broken_masks(score: np.ndarray, k: int, n: int, rng) -> list[np.ndarray]:
    """Top-k masks of score with ties at the k-th value broken in random order:
    one mask when nothing ties across the cut, otherwise n, one per order."""
    m = top_mask(score, k)
    if not (score[~m] == score[m].min()).any():
        return [m]
    masks = []
    for _ in range(n):
        order = rng.permutation(len(score))
        mask = np.zeros(len(score), bool)
        mask[order[np.argsort(-score[order], kind="stable")[:k]]] = True
        masks.append(mask)
    return masks


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Plain Jaccard index of two boolean masks (not graph.jaccard's chance-normalised one)."""
    union = (a | b).sum()
    if union == 0:
        return np.nan
    return (a & b).sum() / union


def normalised_jaccard(masks: list[np.ndarray], n_sim: int = 200, seed: int = 0) -> dict:
    """Eq. 6: (J_bar - J_min) / (J_max - J_min), J averaged over consecutive budgets.

    J_min: random selections of the same sizes. J_max: a nested sequence of the same sizes
    (z_k a subset of z_k+1), for which J(z_k, z_k+1) = |z_k| / |z_k+1|.
    """
    masks = sorted(masks, key=lambda m: m.sum())
    if len(masks) < 2:
        return dict(J_bar=np.nan, J_min=np.nan, J_max=np.nan, J_norm=np.nan, T=len(masks))
    d = len(masks[0])
    sizes = [int(m.sum()) for m in masks]
    n_pairs = len(masks) - 1

    j_bar = np.mean([jaccard(masks[i], masks[i + 1]) for i in range(n_pairs)])
    j_max = np.mean([sizes[i] / sizes[i + 1] for i in range(n_pairs)])
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sim):
        random_masks = []
        for s in sizes:
            random_masks.append(np.isin(np.arange(d), rng.choice(d, s, replace=False)))
        sims.append(np.mean([jaccard(random_masks[i], random_masks[i + 1])
                             for i in range(n_pairs)]))
    j_min = float(np.mean(sims))
    j_norm = np.nan
    if j_max > j_min:
        j_norm = float((j_bar - j_min) / (j_max - j_min))
    return dict(J_bar=float(j_bar), J_min=j_min, J_max=float(j_max), J_norm=j_norm,
                T=len(masks), sizes=" ".join(str(s) for s in sizes))


def model_convergence(acc_id, Z, step: int = 10, n_orders: int = 25, seed: int = 0):
    """The paper's model-count rule on the full OOD set: add models until the ID-OOD
    correlation changes by less than 1%. Averaged over random model orders."""
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
    step_change = np.abs(np.diff(r_mean)) / np.maximum(np.abs(r_mean[:-1]), 1e-12)
    rel = np.concatenate([[np.nan], step_change])
    df = pd.DataFrame(dict(n_models=sizes, r_mean=r_mean, r_sd=np.nanstd(R, 0), rel_change=rel))
    # The first size from which every later step moves the correlation by < 1%.
    ok = (df.rel_change < 0.01).to_numpy()
    n_needed = np.nan
    for j in range(1, len(ok)):
        if ok[j:].all():
            n_needed = sizes[j - 1]
            break
    df["n_needed"] = n_needed
    df["r_all_models"] = float(np.corrcoef(x, a)[0, 1])
    return df


def mask_purity(mask: np.ndarray, A, knn: int, cell) -> float:
    """Coherence purity of a mask on the kNN graph; NaN for a cell without lon/lat (A None)."""
    if A is None:
        return np.nan
    metrics = graph.coherence_metrics(mask, A, knn, tiles=cell.tile_50km, blocks=cell.block_id)
    return metrics["purity"]


def summarise(root: Path, variant: str, tag: str, cells_dir: str, knn: int | None):
    """Write and print <tag>_summary.csv and <tag>_jaccard.csv for one variant folder.
    Returns (summary, {budget: selected mask}), or (None, {}) when the folder has no runs."""
    vdir = root / variant
    frames = [pd.read_csv(p) for p in sorted(vdir.glob(f"{tag}_S*_result.csv"))]
    if not frames:
        return None, {}
    runs = pd.concat(frames, ignore_index=True)
    runs = runs.sort_values("budget").reset_index(drop=True)
    if knn is None:
        knn = int(runs.knn.iloc[0])

    cell, Z, acc, info = op.load_paper_cell(cells_dir, tag)
    controls = set()
    for _, result in runs.iterrows():
        rule, null_seed, run_seed = op.run_controls(result)
        if rule == "random":
            run_seed = 0             # the seed only moves a family split
        controls.add((rule, null_seed, run_seed))
    if len(controls) > 1:
        raise SystemExit(f"{vdir}: result.csv files disagree on (split, null seed, seed): "
                         f"{sorted(controls, key=str)}")
    rule, null_seed, run_seed = controls.pop()
    if "variant" in runs:
        named = set(runs.variant.astype(str))
    else:
        named = {variant}
    if named != {variant}:
        # A run copied into the wrong folder would be summarised as that folder's variant.
        raise SystemExit(f"{vdir}: result.csv files name variant(s) {sorted(named)}, "
                         f"not the folder's {variant!r}")
    selection, _, test = op.make_split(info["n_models"], rule, cell.models.kind.to_numpy(),
                                       run_seed)
    d = info["d"]
    Z_test, acc_test = Z[test], acc[test]
    A = None
    if cell.has_spatial():
        A = graph.build_adjacency(cell.lon, cell.lat, k=knn)

    # Baselines per budget:
    # random_*    N_RANDOM uniformly random subsets of the same size (the paper's Table 4):
    #             mean r, sd, and the Fisher mean the authors' plotting code uses.
    # hard_r      the S most-misclassified points, ranked by 1 - mean correctness over the test
    #             models, as generate_correlation_plots.py does. At small S it is 0 by
    #             construction: those points are wrong for every test model, so the subset
    #             accuracy is constant. The score is a multiple of 1/n_test, so many points tie
    #             at the cut, and the cell's columns put every y = 0 point first: hard_r,
    #             hard_purity and overlap_hard are means over N_TIE_BREAKS random tie orders.
    # hard_sel_r  the same ranking from the selection models (not in the paper), so it is
    #             scored on models its ranking has not seen.
    # coh_* (from result.csv) is reported for the classic run too, which never sees the graph:
    # it is the baseline the spatial run is judged against.
    rng = np.random.default_rng(0)
    tie_rng = np.random.default_rng(1)
    hard_score = 1.0 - Z_test.mean(0)
    hard_sel_score = 1.0 - Z[selection].mean(0)
    masks = {}
    rows = []
    for _, result in runs.iterrows():
        S = int(result.budget)
        theta = np.load(vdir / f"{tag}_S{S}_theta.npy")
        selected = top_mask(theta, S)
        masks[S] = selected
        random_rs = []
        for _ in range(N_RANDOM):
            random_mask = top_mask(rng.random(d), S)
            random_rs.append(op.hard_corr(Z_test, acc_test, random_mask, S, is_mask=True))
        random_purities = []
        for _ in range(N_RANDOM_PURITY):
            random_mask = top_mask(rng.random(d), S)
            random_purities.append(mask_purity(random_mask, A, knn, cell))
        hard_masks = tie_broken_masks(hard_score, S, N_TIE_BREAKS, tie_rng)
        hard_sel_masks = tie_broken_masks(hard_sel_score, S, N_TIE_BREAKS, tie_rng)
        hard_rs, hard_purities, overlaps = [], [], []
        for hard_mask in hard_masks:
            hard_rs.append(op.hard_corr(Z_test, acc_test, hard_mask, S, is_mask=True))
            hard_purities.append(mask_purity(hard_mask, A, knn, cell))
            overlaps.append((selected & hard_mask).sum() / S)
        hard_sel_rs = []
        for hard_mask in hard_sel_masks:
            hard_sel_rs.append(op.hard_corr(Z_test, acc_test, hard_mask, S, is_mask=True))
        row = result.to_dict()
        row.update(
            random_r_mean=float(np.mean(random_rs)), random_r_sd=float(np.std(random_rs)),
            random_r_fisher=fisher_mean(random_rs),
            random_purity=float(np.nanmean(random_purities)),
            hard_r=float(np.mean(hard_rs)), hard_sel_r=float(np.mean(hard_sel_rs)),
            hard_purity=float(np.mean(hard_purities)), overlap_hard=float(np.mean(overlaps)),
        )
        rows.append(row)
    full = dict(cell=tag, variant=variant, split=rule, null_seed=null_seed,
                budget=d, d=d, frac=1.0,
                r_test=op.hard_corr(Z_test, acc_test, np.ones(d, np.float32), d, is_mask=True),
                random_r_mean=np.nan, hard_r=np.nan, n_test=len(test))
    full["r_test_ci_lo"], full["r_test_ci_hi"] = op.fisher_ci(full["r_test"], len(test))
    summary = pd.DataFrame(rows + [full])
    summary.to_csv(vdir / f"{tag}_summary.csv", index=False)

    jac = normalised_jaccard(list(masks.values()))
    jaccard_row = dict(cell=tag, variant=variant, **jac)
    pd.DataFrame([jaccard_row]).to_csv(vdir / f"{tag}_jaccard.csv", index=False)

    aotil_runs = runs[runs.r_test <= op.AOTIL_THRESHOLD]
    largest = 0
    if len(aotil_runs):
        largest = int(aotil_runs.budget.max())
    split_note = f"split {rule}"
    if null_seed is not None:
        split_note += f", null seed {null_seed}"
    print(f"\n=== {tag} | {variant} | d = {d} | test models {len(test)} ({split_note}) ===")
    cols = ["budget", "frac", "r_test", "r_test_ci_lo", "r_test_ci_hi", "spearman_test",
            "random_r_mean", "hard_r", "hard_sel_r"]
    cols += [c for c in COHERENCE_COLUMNS if c in summary]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:+.3f}".format):
        print(summary[cols].to_string(index=False))
    # The largest AoTIL budget is the quantity in the paper's Table 1.
    print(f"  largest AoTIL subset (r_test <= {op.AOTIL_THRESHOLD}): {largest} of {d} "
          f"({largest / d:.1%})")
    print(f"  normalised Jaccard across budgets (Eq. 6): {jac['J_norm']:.3f} "
          f"(J_bar {jac['J_bar']:.3f}, random {jac['J_min']:.3f}, nested {jac['J_max']:.3f})")
    done = set(runs.budget)
    missing = [b for b in op.budgets_for(d) if b not in done]
    if missing:
        print(f"  MISSING budgets (task failed or still running): {missing}")
    return summary, masks


def is_control(summary: pd.DataFrame) -> bool:
    """True for a null-select or family-split run (oodselect_paper.py, TWO CONTROLS)."""
    return op.run_controls(summary.iloc[0])[:2] != ("random", None)


def comparison_rows(tag: str, d: int, summaries: dict, masks: dict) -> list[dict]:
    """classic vs each spatial variant on the random split, at every budget both selected."""
    classic = summaries["classic"].set_index("budget")
    rows = []
    for variant, summary in summaries.items():
        if variant == "classic" or is_control(summary):
            continue
        spatial = summary.set_index("budget")
        for b in sorted(set(classic.index) & set(spatial.index)):
            if b not in masks["classic"] or b not in masks[variant]:
                continue                   # the full-set row has no selection
            overlap = jaccard(masks["classic"][b], masks[variant][b])
            row = dict(cell=tag, variant=variant, budget=b, frac=b / d,
                       r_test_classic=classic.at[b, "r_test"],
                       r_test_spatial=spatial.at[b, "r_test"],
                       cost_of_coherence=spatial.at[b, "r_test"] - classic.at[b, "r_test"],
                       jaccard_masks=float(overlap),
                       random_r=classic.at[b, "random_r_mean"],
                       hard_r=classic.at[b, "hard_r"])
            for c in COHERENCE_COLUMNS:
                if c in classic and c in spatial:
                    row[f"{c}_classic"] = classic.at[b, c]
                    row[f"{c}_spatial"] = spatial.at[b, c]
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results/<date>/oodselect_paper")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="default: every subdirectory of --root")
    args = ap.parse_args()
    root = Path(args.root)
    variants = args.variants
    if not variants:
        variants = sorted(p.name for p in root.iterdir() if p.is_dir())

    for tag in args.tags:
        _, Z, acc, info = op.load_paper_cell(args.cells, tag)
        conv = model_convergence(acc, Z)
        conv.to_csv(root / f"{tag}_model_convergence.csv", index=False)
        print(f"\n##### {tag}: {info['n_models']} models, {info['d']} OOD points; "
              f"ID acc {acc.min():.3f}-{acc.max():.3f} (spread {acc.max() - acc.min():.3f}); "
              f"full-set ID->OOD r (all models) {conv.r_all_models.iloc[0]:+.3f}; "
              f"paper's <1% model-count rule met at {conv.n_needed.iloc[0]} models")

        summaries = {}
        masks = {}
        for variant in variants:
            summary, variant_masks = summarise(root, variant, tag, args.cells, None)
            if summary is not None:
                summaries[variant] = summary
                masks[variant] = variant_masks

        if "classic" not in summaries:
            continue
        if is_control(summaries["classic"]):
            print(f"  WARNING: {root}/classic holds a control run; no comparison written")
            continue
        if len(summaries) == 1:
            continue
        rows = comparison_rows(tag, info["d"], summaries, masks)
        if not rows:
            continue
        comparison = pd.DataFrame(rows)
        comparison.to_csv(root / f"{tag}_comparison.csv", index=False)
        print(f"\n=== {tag}: classic vs spatial ===")
        show = []
        for c in ("variant", "budget", "r_test_classic", "r_test_spatial", "cost_of_coherence",
                  "coh_purity_classic", "coh_purity_spatial", "coh_n_cc_classic",
                  "coh_n_cc_spatial", "coh_tiles_classic", "coh_tiles_spatial", "jaccard_masks"):
            if c in comparison:
                show.append(c)
        with pd.option_context("display.width", 220, "display.max_columns", 30,
                               "display.float_format", "{:+.3f}".format):
            print(comparison[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
