"""Every number and table the paper's Results and Analysis report, recomputed from a finished run.

Usage, from the repo root (after a run; no new searches are made):

    .venv/bin/python aotil/paper_numbers.py
    .venv/bin/python aotil/paper_numbers.py --run results/2026-11-07 \
        --cells data/interim/aotil/cells_v4_2026-11-07 --out results/2026-11-11/paper_numbers

It reads the run's stored files and writes, under --out:
  numbers.md      every number the prose states, keyed by paragraph, plus the self-checks
  budgets.csv     corr_te [95% CI], components (>= c = 20 points), pieces and purity for every search, budget and smoothing setting
  tab_shift.tex, tab_search.tex, tab_regions.tex, tab_mu.tex, tab_c.tex
                  paste-ready tabular blocks for the paper's tables (\\textbf{XX} where the run lacks the input)

What it computes, and from which inputs:
  * The shift (Table tab:shift, paragraph "The aggregate shift is small"): read from the run's
    o1_shift.csv (scope "all") and o1_tile_null.csv; per-tile accuracy and F1 are the unweighted
    ("raw") versions the paper reports. Palm Beach numbers from the cell and the dataset.
  * The searches (Tables tab:search, tab:mu, tab:ablation, Figure fig:budgets): for every stored
    weight file oodselect_paper/<variant>/<tag>_S<S>_theta.npy, the top-S subset, corr_te on the
    58 test models of the 60/20/20 split (probit Pearson, as aotil/audit_selection.probit_r_np),
    its Fisher 95% interval, and pieces and purity from aotil/graph.py on the k = 10 graph.
  * Smoothing (Equation eq:smooth): s_bar = (I + D)^-1 (I + A) s on the same graph A, applied to
    the final weights of each search; the S largest s_bar form the subset. Computed on the
    complement 1 - s = sigmoid(-theta) so that weights that round to 1.0 do not tie.
  * The regions (Table tab:regions, paragraph "Covariate and concept shift"): region_match.csv
    rows of the chosen variant; the pooled row is derived from the per-region rows (eta weighted by
    matched points), which reproduces the stored Laplacian pooled row exactly.
Self-checks (in numbers.md): the recomputed searches against <run>/gtv_vs_laplacian.csv, and the
old directed-10-NN smoothing against the selection audit, so a mismatch shows before a number is used.
Needs numpy, scipy, pandas, scikit-learn and pyproj (no torch).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.special import expit, ndtri
from scipy.stats import norm, rankdata, wilcoxon
from scipy.stats import t as t_dist
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402

TAG = "FL_band_AE_s0"
EPS = 1e-6                     # probit clip and variance floor (oodselect_paper.EPS)
KNN = 10
PALM_BEACH_TILE = "30_10"
VARIANTS = {"classic": "OODSelect", "spatial_mu3_knn10": "Laplacian",
            "spatial_gtv_mu3_knn10": "GTV"}
MU_VARIANTS = {1: "spatial_gtv_mu1_knn10", 3: "spatial_gtv_mu3_knn10", 10: "spatial_gtv_mu10_knn10"}
REPORTED_S = 500
C_MIN = 20                     # minimum component size c (Algorithm alg:spatial)
C_GRID = (10, 15, 20, 30)      # c values of Table tab:c
SEARCH_ROWS = [("OODSelect", ("OODSelect", False)), (r"\quad + smoothing", ("OODSelect", True)),
               ("Laplacian", ("Laplacian", False)), (r"\quad + smoothing", ("Laplacian", True)),
               ("GTV", ("GTV", False)), (r"\quad + smoothing", ("GTV", True))]
COUNTIES = Path("data/interim/gis/counties_FL.parquet")    # GEOID -> NAME, for region names
ABLATION_BUDGETS = [250, 500, 1000]
XX = r"\textbf{XX}"


# ------------------------------------------------------------------ helpers
def model_split(n_models: int):
    """The 60/20/20 selection / validation / test models (oodselect_paper.model_split)."""
    idx = np.arange(n_models)
    rest, test = train_test_split(idx, test_size=0.2, random_state=42)
    selection, validation = train_test_split(rest, test_size=0.25, random_state=42)
    return selection, validation, test


def correctness(S, y):
    """1 where a model's argmax prediction is right (oodselect_paper.correctness)."""
    y = np.asarray(y)
    return ((S > 0.5) | ((S == 0.5) & (y[None, :] == 0))).astype(np.float64)


def probit_r(Z, acc, mask) -> float:
    """Population Pearson r of probit(subset accuracy) on probit(ID accuracy) (probit_r_np)."""
    w = np.asarray(mask, np.float64)
    sub = (Z @ w) / (w.sum() + 1e-6)
    x = ndtri(np.clip(sub, EPS, 1 - EPS))
    y = ndtri(np.clip(acc, EPS, 1 - EPS))
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.mean(xc * yc) / np.sqrt((np.mean(xc ** 2) + EPS) * (np.mean(yc ** 2) + EPS)))


def fisher_ci(r: float, n: int, level: float = 0.95):
    z, se, q = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def top_mask(score, k):
    """The k highest scores; ties go to the lower index (report_paper.top_mask)."""
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def bottom_mask(score, k):
    m = np.zeros(len(score), bool)
    m[np.argsort(score, kind="stable")[:k]] = True
    return m


def smooth_graph(theta, A, S):
    """Equation eq:smooth on graph A: keep the S largest (I + D)^-1 (I + A) sigmoid(theta),
    computed as the S smallest smoothed sigmoid(-theta). Returns (mask, ties at the cut)."""
    q = expit(-np.asarray(theta, np.float64))
    deg = np.asarray(A.sum(axis=1)).ravel()
    q_bar = (q + A @ q) / (1.0 + deg)
    mask = bottom_mask(q_bar, S)
    cut = np.sort(q_bar)[S - 1]
    return mask, int((q_bar == cut).sum())


def smooth_knn_legacy(theta, xy, S, k=KNN):
    """The old smoothing (audit_selection.smoothed_classic): mean over each point's k nearest
    points, itself included. Only used to check this script against the stored audit."""
    q = expit(-np.asarray(theta, np.float64))
    _, idx = cKDTree(xy).query(xy, k=k)
    return bottom_mask(q[idx].mean(axis=1), S)


def sgn(x, nd=2):
    """'$-$0.97' / '$+$0.84' style for signed table cells (leading zero kept, Victor 2026-11-11)."""
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.{nd}f}"


def gap(x, nd=3):
    """'$-$0.034' / '0.017' style (minus sign only), as the shift tables print gaps."""
    return ("$-$" if x < 0 else "") + f"{abs(x):.{nd}f}"


def share(x, nd=2):
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------ parts
def load_cell(cells: Path):
    z = np.load(cells / f"{TAG}.npz", allow_pickle=True)
    Z_ood = correctness(z["S_ood"], z["y_ood"])
    Z_id = correctness(z["S_id"], z["y_id"])
    _, _, test = model_split(Z_ood.shape[0])
    return z, Z_ood, Z_id.mean(1), test


def components(mask, A, c=C_MIN):
    """Components of the subset: connected components of the subgraph the subset induces in A
    with at least c points. Returns their number and the share of selected points they hold."""
    from scipy.sparse.csgraph import connected_components
    idx = np.where(np.asarray(mask, bool))[0]
    if len(idx) == 0:
        return 0, float("nan")
    _, lab = connected_components(A[idx][:, idx], directed=False)
    sizes = np.bincount(lab)
    big = sizes >= c
    return int(big.sum()), float(sizes[big].sum() / len(idx))


def search_rows(run: Path, z, Z_ood, acc_id, test, A, budgets_only=None):
    """corr_te, interval, pieces and purity of every stored search, plain and smoothed."""
    rows = []
    Zt, at = Z_ood[test], acc_id[test]
    variants = dict(VARIANTS)
    for mu, v in MU_VARIANTS.items():
        variants.setdefault(v, f"GTV mu={mu}")
    for v, name in variants.items():
        vdir = run / "oodselect_paper" / v
        if not vdir.exists():
            continue
        for f in sorted(vdir.glob(f"{TAG}_S*_theta.npy")):
            S = int(re.search(r"_S(\d+)_theta", f.name).group(1))
            if budgets_only and S not in budgets_only:
                continue
            theta = np.load(f)
            for smoothed in (False, True):
                if smoothed:
                    mask, ties = smooth_graph(theta, A, S)
                else:
                    mask, ties = top_mask(theta, S), 0
                r = probit_r(Zt, at, mask)
                lo, hi = fisher_ci(r, len(test))
                coh = graph.coherence_metrics(mask, A, KNN)
                n_comp, in_comp = components(mask, A)
                by_c = {}
                for c in C_GRID:
                    n_c, f_c = components(mask, A, c)
                    by_c[f"comp_c{c}"], by_c[f"pts_c{c}"] = n_c, int(round(f_c * S)) if n_c else 0
                rows.append(dict(variant=v, search=name, smoothing=smoothed, S=S, r_te=r,
                                 ci_lo=lo, ci_hi=hi, components=n_comp, in_components=in_comp,
                                 comp_points=int(round(in_comp * S)) if n_comp else 0, **by_c,
                                 pieces=coh["n_cc"], purity=coh["purity"], ties_at_cut=ties))
    return pd.DataFrame(rows)


def checks(run: Path, rows: pd.DataFrame, z, Z_ood, acc_id, test, xy):
    """Recomputed numbers against the run's stored ones; returns markdown lines."""
    out = ["## Self-checks", ""]
    stored = run / "gtv_vs_laplacian.csv"
    if stored.exists():
        s = pd.read_csv(stored)
        m = rows[~rows.smoothing].merge(s, on=["search", "S"], suffixes=("", "_stored"))
        out.append(f"- Searches vs `{stored}` ({len(m)} rows): max |r_te diff| "
                   f"{(m.r_te - m.r_test).abs().max():.2e}, max |purity diff| "
                   f"{(m.purity - m.purity_stored).abs().max():.2e}, pieces equal: "
                   f"{bool((m.pieces == m.pieces_stored).all())}")
    audit = run / "oodselect_paper" / f"{TAG}_selection_audit.json"
    theta_dir = run / "oodselect_paper" / "classic"
    if audit.exists() and theta_dir.exists():
        j = json.loads(audit.read_text())
        found = []

        def walk(o):
            if isinstance(o, dict):
                if o.get("variant") == "smoothed_classic_knn10" and "coh_purity" in o:
                    found.append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(j)
        for o in found:
            S = int(o["budget"])
            theta = np.load(theta_dir / f"{TAG}_S{S}_theta.npy")
            mask = smooth_knn_legacy(theta, xy, S)
            r = probit_r(Z_ood[test], acc_id[test], mask)
            A = checks.A
            coh = graph.coherence_metrics(mask, A, KNN)
            out.append(f"- Old smoothing (directed 10-NN) at S = {S}: r_te {r:.4f} vs stored "
                       f"{o['r_test']:.4f}; purity {coh['purity']:.4f} vs {o['coh_purity']:.4f}; "
                       f"pieces {coh['n_cc']} vs {o['coh_n_cc']}")
    return out


def auc_rows(P, y):
    """ROC AUC of every model (row of P) on the given points, from average ranks (sklearn's
    roc_auc_score); NaN for every model when a class is missing."""
    pos = np.asarray(y) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.full(P.shape[0], np.nan)
    R = rankdata(P, axis=1)
    return (R[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def acc_f1_rows(Z, y):
    """Accuracy and flood-class F1 (2TP / (2TP + FP + FN)) of every model on the given points."""
    flood = np.asarray(y) == 1
    tp = (Z[:, flood]).sum(1)
    fn = (~Z[:, flood]).sum(1)
    fp = (~Z[:, ~flood]).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Z.mean(1), 2 * tp / (2 * tp + fp + fn)


def sign_flip_p(d, n_perm=100_000, seed=0):
    """Two-sided sign-flip permutation p of mean(d) against 0: under H0 (no shift) each tile's
    gap is as likely to be positive as negative, so its sign is flipped at random."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    hits = 0
    for start in range(0, n_perm, 20_000):
        signs = rng.choice([-1.0, 1.0], size=(min(20_000, n_perm - start), len(d)))
        hits += int((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum())
    return (hits + 1) / (n_perm + 1)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p values."""
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for rank, k in enumerate(keys):
        running = max(running, min(1.0, (m - rank) * pvals[k]))
        out[k] = running
    return out


def tile_bootstrap(to, n_boot=2000, seed=0):
    """o1_shift.tile_draws: n_boot resamples, with replacement, of the test tiles (those of the
    Zone A half), with numpy's default_rng(seed); returns (tiles, draws x tiles count matrix)."""
    test_tiles = np.unique(to)
    rng = np.random.default_rng(seed)
    draws = [rng.choice(test_tiles, len(test_tiles)) for _ in range(n_boot)]
    index = {tile: j for j, tile in enumerate(test_tiles)}
    K = np.zeros((n_boot, len(test_tiles)))
    for b, draw in enumerate(draws):
        for tile in draw:
            K[b, index[tile]] += 1
    return test_tiles, K


def t_interval(x, level=0.95):
    """mean +- t_{n-1} * sd / sqrt(n) over the tile values (NaN tiles left out)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, m = len(x), float(x.mean())
    h = float(t_dist.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m - h, m + h


def boot_interval(G, K):
    """o1_shift.tile_summary's interval: per draw, each model's mean of G (tiles x models) over the
    drawn tiles (a tile drawn twice counts twice), averaged over the models; 2.5th and 97.5th
    percentiles over the draws that hold a usable tile."""
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = ((K @ G) / K.sum(1, keepdims=True)).mean(1)
    boot = boot[np.isfinite(boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def shift_part(run: Path, z, test, dataset: Path, cells: Path, tile_ci: str = "t"):
    """Table tab:shift, the test of 'The aggregate shift is small', and its other numbers.

    Every model of the zoo (290), as aotil/o1_shift.py computes the shift. Pooled: each model's
    metric over all points of each half, averaged over the models. Per tile: each model's metric
    within each test tile, averaged over the models and then over the tiles where the metric is
    defined (accuracy: both halves present; F1: floods in both halves; AUROC: both classes in both
    halves). Per-tile 95% intervals are t-intervals over the tiles (each tile's value averaged over
    the models); --tile-ci bootstrap gives o1_shift.py's 2,000-draw tile bootstrap instead, and the
    bootstrap shift interval is always checked against o1_shift.csv. Pooled values carry no interval. The test: the tile is the unit (points within a
    tile are spatially correlated, so they are not independent), each tile's gap is averaged
    over the models, and a two-sided Wilcoxon signed-rank test asks whether the tile gaps are
    centred on 0, with Holm's correction across the three metrics. o1_shift.py computed the same
    signed-rank test before these results existed. A sign-flip permutation test of the mean tile
    gap is printed beside it: it is pulled toward 0 by a few tiles whose two zones have very
    different flood shares, which the signed-rank test is less sensitive to."""
    yo, yi = z["y_ood"], z["y_id"]
    Zo = (z["S_ood"] > 0.5) | ((z["S_ood"] == 0.5) & (yo[None] == 0))
    Zi = (z["S_id"] > 0.5) | ((z["S_id"] == 0.5) & (yi[None] == 0))
    Po = np.where(yo[None] == 1, 1 - (1 - z["S_ood"]), 1 - z["S_ood"])
    Pi = np.where(yi[None] == 1, 1 - (1 - z["S_id"]), 1 - z["S_id"])
    ds = pd.read_parquet(dataset, columns=["point_id", "tile_50km"]).set_index("point_id")
    to = np.asarray(z["tile_50km"])
    ti = ds.loc[z["point_id_id"], "tile_50km"].values
    models = pd.read_csv(cells / f"{TAG}_models.csv")

    # pooled (no interval)
    acc_i, f1_i = acc_f1_rows(Zi, yi)
    acc_o, f1_o = acc_f1_rows(Zo, yo)
    pooled = dict(acc=(acc_i.mean(), acc_o.mean()), f1=(f1_i.mean(), f1_o.mean()),
                  auc=(models.auc_id.mean(), models.auc_ood.mean()))

    # per tile
    tiles = sorted(set(to) | set(ti))
    per = {m: dict(ID=[], A=[]) for m in ("acc", "f1", "auc")}
    for tile in tiles:
        mo, mi = to == tile, ti == tile
        both = mo.any() and mi.any()
        ao, fo = acc_f1_rows(Zo[:, mo], yo[mo]) if mo.any() else (None, None)
        ai, fi = acc_f1_rows(Zi[:, mi], yi[mi]) if mi.any() else (None, None)
        floods = both and (yo[mo] == 1).any() and (yi[mi] == 1).any()
        classes = both and len(set(yo[mo])) == 2 and len(set(yi[mi])) == 2
        for m, ok, vi, vo in [("acc", both, ai, ao), ("f1", floods, fi, fo),
                              ("auc", classes, auc_rows(Pi[:, mi], yi[mi]) if classes else None,
                               auc_rows(Po[:, mo], yo[mo]) if classes else None)]:
            per[m]["ID"].append(vi if ok else np.full(Zo.shape[0], np.nan))
            per[m]["A"].append(vo if ok else np.full(Zo.shape[0], np.nan))
    s = pd.read_csv(run / "o1_shift.csv")
    stored = s[s.scope == "all"].iloc[0]
    ci_cols = dict(acc=("tile_acc_raw_ci_lo", "tile_acc_raw_ci_hi", "tile_acc_raw_gap", "tile_acc_raw_wilcoxon_p"),
                   auc=("tile_auc_ci_lo", "tile_auc_ci_hi", "tile_auc_gap", "tile_auc_wilcoxon_p"),
                   f1=("tile_f1_raw_ci_lo", "tile_f1_raw_ci_hi", "tile_f1_raw_gap", "tile_f1_raw_wilcoxon_p"))
    res, p_perm, checks_, d_tiles = {}, {}, [], {}
    boot_tiles, K_all = tile_bootstrap(to)
    col = {tile: j for j, tile in enumerate(boot_tiles)}
    for m in ("acc", "auc", "f1"):
        I = np.array(per[m]["ID"]).T                     # models x tiles
        A = np.array(per[m]["A"]).T
        keep = ~np.isnan(I).all(0)
        d_tile = np.nanmean(I[:, keep] - A[:, keep], axis=0)  # each tile's gap, mean over models
        K = K_all[:, [col[tile] for tile, k in zip(tiles, keep) if k]]
        boot_id, boot_a = boot_interval(I[:, keep].T, K), boot_interval(A[:, keep].T, K)
        boot_gap = boot_interval((I[:, keep] - A[:, keep]).T, K)
        if tile_ci == "t":                               # the paper's intervals (2026-11-11)
            ci_id = t_interval(np.nanmean(I[:, keep], axis=0))
            ci_a = t_interval(np.nanmean(A[:, keep], axis=0))
            ci_gap = t_interval(d_tile)
        else:                                            # o1_shift.py's tile bootstrap
            ci_id, ci_a, ci_gap = boot_id, boot_a, boot_gap
        res[m] = dict(ID=float(np.nanmean(I[:, keep])), A=float(np.nanmean(A[:, keep])),
                      gap=float(d_tile.mean()), n_tiles=int(keep.sum()),
                      tiles_pos=int((d_tile > 0).sum()), ci=ci_gap, ci_ID=ci_id, ci_A=ci_a,
                      wilcoxon_p=float(wilcoxon(d_tile).pvalue))
        checks_.append(f"- Per-tile {m} shift, bootstrap interval: recomputed [{boot_gap[0]:.4f}, {boot_gap[1]:.4f}] vs "
                       f"o1_shift.csv [{stored[ci_cols[m][0]]:.4f}, {stored[ci_cols[m][1]]:.4f}]")
        p_perm[m] = sign_flip_p(d_tile)
        d_tiles[m] = d_tile
        checks_.append(f"- Per-tile {m} gap: recomputed {res[m]['gap']:.4f} vs o1_shift.csv "
                       f"{stored[ci_cols[m][2]]:.4f}; Wilcoxon p {res[m]['wilcoxon_p']:.4f} vs "
                       f"{stored[ci_cols[m][3]]:.4f}")
    for m, (a, b) in [("acc", ("acc_ID", "acc_A")), ("f1", ("f1_ID", "f1_A")), ("auc", ("ID_AUC", "A_AUC"))]:
        checks_.append(f"- Pooled {m}: recomputed {pooled[m][0]:.4f} / {pooled[m][1]:.4f} vs "
                       f"o1_shift.csv {stored[a]:.4f} / {stored[b]:.4f}")
    p_wil = {m: res[m]["wilcoxon_p"] for m in res}
    p_holm = holm(p_wil)
    p_perm_holm = holm(p_perm)

    names = dict(acc="Accuracy", auc="AUROC", f1="F1")
    body = []
    for m in ("acc", "auc", "f1"):
        r = res[m]
        body.append(f"ID {names[m] if m != 'acc' else 'accuracy'} & {share(pooled[m][0], 3)} & {share(r['ID'], 3)} "
                    f"[{share(r['ci_ID'][0], 3)}, {share(r['ci_ID'][1], 3)}] \\\\")
        body.append(f"OOD {names[m] if m != 'acc' else 'accuracy'} & {share(pooled[m][1], 3)} & {share(r['A'], 3)} "
                    f"[{share(r['ci_A'][0], 3)}, {share(r['ci_A'][1], 3)}] \\\\")
        body.append(f"{names[m]} shift & {gap(pooled[m][0] - pooled[m][1])} & {gap(r['gap'])} "
                    f"[{gap(r['ci'][0])}, {gap(r['ci'][1])}] \\\\")
        if m != "f1":
            body.append(r"\midrule")
    tex = "\n".join([r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
                     r"\textbf{Metric} & \textbf{Pooled} & \textbf{Per tile} \\", r"\midrule",
                     *body, r"\bottomrule", r"\end{tabular}"])

    pb_id, pb_ood = ti == PALM_BEACH_TILE, to == PALM_BEACH_TILE
    nums = {}
    for m in ("acc", "auc", "f1"):
        r = res[m]
        nums[f"{names[m]}: pooled ID / OOD / shift"] = (pooled[m][0], pooled[m][1], pooled[m][0] - pooled[m][1])
        nums[f"{names[m]}: per-tile ID [95% CI]"] = (r["ID"], *r["ci_ID"])
        nums[f"{names[m]}: per-tile OOD [95% CI]"] = (r["A"], *r["ci_A"])
        nums[f"{names[m]}: per-tile shift [95% CI]"] = (r["gap"], *r["ci"])
        nums[f"{names[m]}: tiles used, tiles with a positive gap"] = (r["n_tiles"], r["tiles_pos"])
        nums[f"{names[m]}: Wilcoxon signed-rank p, Holm-adjusted p"] = (r["wilcoxon_p"], p_holm[m])
        nums[f"{names[m]}: shift at 95% confidence (Wilcoxon, Holm)?"] = "yes" if p_holm[m] < 0.05 else "no"
        nums[f"{names[m]}: check, sign-flip test of the mean tile gap: p, Holm-adjusted p"] = (p_perm[m], p_perm_holm[m])
        nums[f"{names[m]}: median tile gap"] = float(np.median(d_tiles[m]))
    nums["share of models worse on Zone A (per-tile accuracy)"] = float(stored["tile_acc_raw_worse"])
    nums["top-15 test models, per-tile accuracy shift [CI]"] = (stored["tile_strong15_acc_raw_gap"], stored["tile_strong15_acc_raw_ci_lo"], stored["tile_strong15_acc_raw_ci_hi"])
    nums["corr_te on all Zone A [CI] (58 test models)"] = (stored["r_full_test"], stored["r_full_test_ci_lo"], stored["r_full_test_ci_hi"])
    nums["Palm Beach: share of Zone AE / Zone A test points"] = (pb_id.mean(), pb_ood.mean())
    nums["Palm Beach: points Zone AE / Zone A"] = (int(pb_id.sum()), int(pb_ood.sum()))
    nums["Palm Beach: accuracy on its Zone AE / Zone A points (all models)"] = (Zi[:, pb_id].mean(), Zo[:, pb_ood].mean())
    return tex, nums, checks_


def regions_part(run: Path, variant: str, m_ood: int, match_csv: Path | None = None,
                 points_path: Path | None = None):
    """Table tab:regions and the numbers of 'Covariate and concept shift'.

    match_csv: a region_match.csv whose main variant (--regions) is `variant`; it has the variant's
    "Other Zone A points" row. Without it, the run's region_match.csv is used, where a variant other
    than the main one only has its regions (extra_region rows) and the Other Zone A row is XX."""
    path = match_csv if match_csv is not None else run / "region_match" / "region_match.csv"
    rm = pd.read_csv(path)
    meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    main_variant = variant in str(meta.get("regions", ""))
    pre = "" if main_variant else f"{variant}: "
    reg = (rm[rm.kind == "region"] if main_variant
           else rm[rm.set.str.startswith(pre) & (rm.kind == "extra_region")]).copy()
    reg["name"] = reg.set.str.replace(pre, "", regex=False) if pre else reg.set
    pts = pd.read_parquet(points_path or run / "regions" / f"{TAG}_{variant}_S{REPORTED_S}_points.parquet")
    lon = pts.groupby("component").lon.mean()
    # Name each region by the county that holds most of its points (region_match.py joins every
    # county with at least one point, so one stray point added a county to the name).
    county = dict(pd.read_parquet(COUNTIES, columns=["GEOID", "NAME"]).values)
    majority = pts.groupby("component").meta_point_county.agg(lambda c: c.astype(str).value_counts().index[0])
    reg["name_all_counties"] = reg["name"]
    reg["name"] = reg.component.map(majority).map(lambda f: county.get(f, f))
    for name, g in reg.groupby("name"):
        if len(g) > 1:                      # same county name: east / west by mean longitude
            order = g.component.map(lon).sort_values().index
            for side, i in zip(["west", "east"] if len(g) == 2 else range(len(g)), order):
                reg.loc[i, "name"] = f"{name} ({side})"
    n, nm = reg.n.sum(), reg.n_matched.sum()
    pooled = dict(n=int(n), unmatched=1 - nm / n, eta=float((reg.eta * reg.n_matched).sum() / nm),
                  pos_points=float((reg.pos_points * reg.n).sum() / n),
                  pos_matches=float((reg.pos_matches * reg.n_matched).sum() / nm))
    base = rm[rm.set == "Zone AE baseline"].iloc[0]
    other = rm[rm.set == f"{pre}Other Zone A points"]

    def line(name, n_, pp, pm, un, eta):   # pp, pm (flood shares) kept in numbers.md, not in the table
        return f"{name} & {n_:,} & {share(un)} & {share(eta)} \\\\"
    body = [line(x["name"], int(x.n), x.pos_points, x.pos_matches, x.unmatched, x.eta)
            for _, x in reg.sort_values("n", ascending=False).iterrows()]
    body.append(r"\midrule")
    body.append(line("All regions", pooled["n"], pooled["pos_points"], pooled["pos_matches"],
                     pooled["unmatched"], pooled["eta"]))
    if len(other):
        o = other.iloc[0]
        body.append(line("Other Zone A", int(o.n), o.pos_points, o.pos_matches, o.unmatched, o.eta))
    else:
        body.append(f"Other Zone A & {m_ood - REPORTED_S:,} & {XX} & {XX} \\\\")
    body.append(line("Zone AE baseline", int(base.n), base.pos_points, base.pos_matches, base.unmatched, base.eta))
    tex = "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
                     r"\textbf{Region} & \textbf{\# Points} & \textbf{Unmatched} & $\boldsymbol{\eta}$ \\",
                     r"\midrule", *body, r"\bottomrule", r"\end{tabular}"])
    nums = {"regions": len(reg), "points in regions": pooled["n"],
            "pooled unmatched": pooled["unmatched"], "largest region unmatched": float(reg.unmatched.max()),
            "pooled eta": pooled["eta"], "baseline eta": float(base.eta), "baseline unmatched": float(base.unmatched),
            "region eta range": (float(reg.eta.min()), float(reg.eta.max())),
            "regions above baseline eta": int((reg.eta > base.eta).sum()),
            "Other Zone A row": "present" if len(other) else "missing (XX)",
            "region names (majority county | every county)": dict(zip(reg.name, reg.name_all_counties)),
            "flood share, points / matches": {r["name"]: (round(r.pos_points, 2), round(r.pos_matches, 2))
                                              for _, r in reg.iterrows()},
            "Other Zone A unmatched, eta": ((float(other.iloc[0].unmatched), float(other.iloc[0].eta))
                                            if len(other) else "missing"),
            "source": str(path)}
    return tex, nums


def search_tables(rows: pd.DataFrame, full_zone_a):
    def pick(search, S, smoothed):
        g = rows[(rows.search == search) & (rows.S == S) & (rows.smoothing == smoothed)]
        return None if g.empty else g.iloc[0]

    def cells(x):
        if x is None:
            return f"{XX} & {XX} & {XX}"
        return (f"{sgn(x.r_te)} [{sgn(x.ci_lo)}, {sgn(x.ci_hi)}] & {int(x.components)} ({int(x.comp_points)}) & "
                f"{share(x.purity)}")
    r_all, lo_all, hi_all = full_zone_a
    # Row names as in the paper (Victor's style): the post-hoc baseline is "Smoothing".
    search = [f"All Zone A & {sgn(r_all)} [{sgn(lo_all)}, {sgn(hi_all)}] & -- & -- \\\\"]
    for label, (name, sm) in SEARCH_ROWS:
        search.append(f"{label} & {cells(pick(name, REPORTED_S, sm))} \\\\")
    head = r"\textbf{Points} & $\boldsymbol{\text{corr}_\text{te}}$ \textbf{[95\% CI]} & \textbf{\# Comps} & \textbf{Purity} \\"
    tab_search = "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule", head, r"\midrule", *search,
                            r"\bottomrule", r"\end{tabular}"])
    c_lines = []
    for label, (search_name, sm) in SEARCH_ROWS:
        x = pick(search_name, REPORTED_S, sm)
        cs = [XX if x is None else f"{int(x[f'comp_c{c}'])} ({int(x[f'pts_c{c}'])})" for c in C_GRID]
        c_lines.append(f"{label} & " + " & ".join(cs) + r" \\")
    head_c = r"\textbf{Search} & " + " & ".join(rf"$\boldsymbol{{c={c}}}$" for c in C_GRID) + r" \\"
    tab_c = "\n".join([r"\begin{tabular}{@{}l" + "c" * len(C_GRID) + "@{}}", r"\toprule", head_c,
                       r"\midrule", *c_lines, r"\bottomrule", r"\end{tabular}"])
    abl = [f"Laplacian & {cells(pick('Laplacian', REPORTED_S, False))} \\\\",
           rf"\quad + smoothing & {cells(pick('Laplacian', REPORTED_S, True))} \\",
           f"GTV & {cells(pick('GTV', REPORTED_S, False))} \\\\",
           rf"\quad + smoothing & {cells(pick('GTV', REPORTED_S, True))} \\"]
    head_s = head.replace(r"\textbf{Points}", r"\textbf{Search}")
    tab_abl = "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule", head_s, r"\midrule", *abl,
                         r"\bottomrule", r"\end{tabular}"])
    mu_lines = []
    for mu in (1, 3, 10):
        name = "GTV" if mu == 3 else f"GTV mu={mu}"
        cs = []
        for S in ABLATION_BUDGETS:
            x = pick(name, S, False)
            cs.append(f"{XX} / {XX}" if x is None else f"{sgn(x.r_te)} / {share(x.purity)}")
        mu_lines.append(f"{mu} & " + " & ".join(cs) + r" \\")
    tab_mu = "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
                        r"$\boldsymbol{\mu}$ & $\boldsymbol{S=250}$ & $\boldsymbol{S=500}$ & $\boldsymbol{S=1{,}000}$ \\",
                        r"\midrule", *mu_lines, r"\bottomrule", r"\end{tabular}"])
    return tab_search, tab_abl, tab_mu, tab_c


def budget_numbers(rows: pd.DataFrame, m_ood: int):
    p = rows[~rows.smoothing & rows.search.isin(["OODSelect", "Laplacian", "GTV"])]
    r = p.pivot(index="S", columns="search", values="r_te")
    pu = p.pivot(index="S", columns="search", values="purity")
    pc = p.pivot(index="S", columns="search", values="pieces")
    nc = p.pivot(index="S", columns="search", values="components")
    ic = p.pivot(index="S", columns="search", values="in_components")
    mid = [s for s in r.index if 20 <= s <= 10000]
    s50 = [s for s in r.index if 50 <= s <= 10000]
    s250 = [s for s in r.index if 250 <= s <= 10000]
    big = max(r.index)
    return {
        "every search inverts (r_te < -0.3) up to 10,000": bool((r.loc[:10000] < -0.3).all().all()),
        f"r_te at the largest budget {big} ({big / m_ood:.1%} of Zone A)": r.loc[big].round(2).to_dict(),
        "GTV has the highest purity at every budget 20-10,000": all(pu.loc[s, "GTV"] >= pu.loc[s].max() for s in mid),
        "GTV has the fewest pieces at every budget 20-10,000": all(pc.loc[s, "GTV"] <= pc.loc[s].min() for s in mid),
        "components (>= 20 points) by budget": nc.astype(int).to_dict("index"),
        "share of selected points in components by budget": ic.round(2).to_dict("index"),
        "GTV purity range, 50-10,000": (float(pu.loc[s50, "GTV"].min()), float(pu.loc[s50, "GTV"].max())),
        "GTV - Laplacian r_te gap range, 250-10,000": (float((r.loc[s250, "GTV"] - r.loc[s250, "Laplacian"]).min()),
                                                       float((r.loc[s250, "GTV"] - r.loc[s250, "Laplacian"]).max())),
        "Laplacian purity at 20 and 250": (float(pu.loc[20, "Laplacian"]), float(pu.loc[250, "Laplacian"])),
        "random purity at S = 500 (S/m)": REPORTED_S / m_ood,
    }


def fmt(v):
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=Path("results/2026-11-07"))
    ap.add_argument("--cells", type=Path, default=Path("data/interim/aotil/cells_v4_2026-11-07"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed_v4/dataset_FL.parquet"))
    # The paper analyzes the regions of GTV followed by smoothing (since 2026-11-11). For GTV alone:
    #   --regions-variant spatial_gtv_mu3_knn10 --region-match results/2026-11-11/region_match_gtv/region_match.csv
    #   --regions-points results/2026-11-07/regions/FL_band_AE_s0_spatial_gtv_mu3_knn10_S500_points.parquet
    ap.add_argument("--regions-variant", default="spatial_gtv_mu3_knn10_smoothed")
    ap.add_argument("--region-match", type=Path,
                    default=Path("results/2026-11-11/region_match_gtv_smoothed/region_match.csv"),
                    help="region_match.csv run with --regions-variant as its main variant (gives Other Zone A)")
    ap.add_argument("--regions-points", type=Path,
                    default=Path("results/2026-11-11/regions_gtv_smoothed/FL_band_AE_s0_spatial_gtv_mu3_knn10_smoothed_S500_points.parquet"),
                    help="regions.py points table of --regions-variant")
    ap.add_argument("--tile-ci", choices=["t", "bootstrap"], default="t",
                    help="per-tile intervals: t over the tiles (paper) or o1_shift's tile bootstrap")
    ap.add_argument("--out", type=Path, default=Path(f"results/{dt.date.today():%Y-%m-%d}/paper_numbers"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    z, Z_ood, acc_id, test = load_cell(a.cells)
    Z_id = correctness(z["S_id"], z["y_id"])
    A = graph.build_adjacency(z["lon"], z["lat"], k=KNN)
    checks.A = A
    xy = graph.project(z["lon"], z["lat"])
    rows = search_rows(a.run, z, Z_ood, acc_id, test, A)
    rows.to_csv(a.out / "budgets.csv", index=False)

    tab_shift, shift_nums, shift_checks = shift_part(a.run, z, test, a.dataset, a.cells, a.tile_ci)
    tab_regions, region_nums = regions_part(a.run, a.regions_variant, Z_ood.shape[1],
                                            a.region_match if a.region_match.exists() else None,
                                            a.regions_points)
    tab_search, tab_abl, tab_mu, tab_c = search_tables(rows, shift_nums["corr_te on all Zone A [CI] (58 test models)"])
    for name, tex in [("tab_shift", tab_shift), ("tab_search", tab_search), ("tab_regions", tab_regions),
                      ("tab_mu", tab_mu), ("tab_c", tab_c)]:  # tab_ablation merged into tab_search
        (a.out / f"{name}.tex").write_text(tex + "\n")

    def pick(search, S, sm):
        g = rows[(rows.search == search) & (rows.S == S) & (rows.smoothing == sm)]
        return None if g.empty else g.iloc[0]
    search_nums = {}
    for search in ["OODSelect", "Laplacian", "GTV"]:
        for sm in (False, True):
            x = pick(search, REPORTED_S, sm)
            if x is not None:
                search_nums[f"{search}{' + smoothing' if sm else ''} at S = 500: r_te [CI], components, share in components, all connected pieces, purity"] = (
                    x.r_te, x.ci_lo, x.ci_hi, int(x.components), x.in_components, int(x.pieces), x.purity)
    pts = pd.read_parquet(a.regions_points)
    search_nums[f"{a.regions_variant} S = 500: regions of >= 20 points, points in them"] = (
        int((pts.groupby('component').size() >= 20).sum()), int((pts.component_size >= 20).sum()))

    md = [f"# Paper numbers from `{a.run}` (cell `{a.cells}`)", "",
          f"Written {dt.datetime.now():%Y-%m-%d %H:%M} by aotil/paper_numbers.py. Tables: the .tex files here.", ""]
    for title, nums in [("The shift (Table tab:shift; test: two-sided Wilcoxon signed-rank over tiles, Holm across the three metrics)", shift_nums),
                        ("Searches at S = 500 (Tables tab:search, tab:ablation; paragraph Our loss)", search_nums),
                        ("Covariate and concept shift (Table tab:regions)", region_nums),
                        ("Sensitivity: budget (Figure fig:budgets)", budget_numbers(rows, Z_ood.shape[1]))]:
        md += [f"## {title}", ""] + [f"- {k}: {fmt(v)}" for k, v in nums.items()] + [""]
    missing = [f"GTV mu = {mu}" for mu, v in MU_VARIANTS.items() if not (a.run / "oodselect_paper" / v).exists()]
    md += ["## Missing inputs (XX in the tables)", ""] + ([f"- {m}: no weight files" for m in missing] or ["- none"])
    if region_nums["Other Zone A row"] != "present":
        md.append(f"- Other Zone A for {a.regions_variant}: no row in region_match.csv")
    md += [""] + checks(a.run, rows, z, Z_ood, acc_id, test, xy) + shift_checks
    (a.out / "numbers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {a.out}/numbers.md, budgets.csv and 5 tables (tab_shift, tab_search, tab_regions, tab_mu, tab_c)")


if __name__ == "__main__":
    main()
