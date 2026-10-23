#!/usr/bin/env python3
"""Audit one paper-protocol OODSelect run (oodselect_paper.py): every number the run stored must
follow from the stored selection and the cell. Also tabulates test r and coherence per budget.

Reads   <root>/<variant>/<tag>_S<S>_{theta.npy,result.csv,meta.json,trials.csv}
        <cells-dir>/<tag>.npz and <tag>_models.csv
Writes  <out>/<tag>_selection_audit.json   every check and every number
        <out>/<tag>_selection_audit.md     the same as tables (also printed)
Exits 1 when any check fails; warnings (a budget not run, a cell without point_id, a tie at the
cut) do not fail the audit.

Checks, per variant and budget:
  1. theta has one finite entry per OOD point; result.csv and meta.json name this cell, budget
     and model split.
  2. Test r recomputed with the selection's own code (to 1e-6) and re-derived in float64 numpy
     (to 1e-3: a test model whose subset accuracy is exactly 0 or 1 sits at the probit clamp,
     where torch's float32 erfinv moves ~0.01 per ulp).
  3. torch.topk and the stable argsort pick the same top-S set.
  4. The stored coh_* columns are reproduced on the rebuilt kNN graph.
  5. The ID->OOD r over all OOD points, re-derived in numpy, and each run's stored r_full_test.
--smoothed-classic adds the smoothed-classic ablation rows (see smoothed_classic).

Example:
    python aotil/audit_selection.py --root results/<date>/oodselect_paper \\
        --cells-dir data/interim/aotil/cells --tag FL_band_AE_s0 --smoothed-classic
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import expit, ndtri
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph                     # noqa: E402
import oodselect_paper as op     # noqa: E402

DEFAULT_VARIANTS = ["classic", "spatial_mu3_knn10"]
CHECK_BUDGETS = [250, 500, 1000]       # budgets in the table
RECOMPUTE_BUDGETS = [500, 1000]        # budgets whose stored r is recomputed (check 2)
MIN_COMPONENT = 20    # points; smaller components are listed only (docs/REGION-ANALYSIS.md 0.4)
TOL_SAME = 1e-6       # same code, same float32 arithmetic
TOL_NUMPY = 1e-3      # float64 re-derivation against float32 torch (check 2)
TOL_COH = 1e-9        # coherence recomputed with the same graph.py code
COH_KEYS = ("purity", "P_hard", "n_cc", "largest_cc_frac", "tiles", "blocks", "moran_I")
FILE_RE = r"_S(\d+)_(theta\.npy|result\.csv|meta\.json|trials\.csv)"   # follows the cell tag
MARK = {"pass": "✓", "warn": "⚠", "fail": "✗"}


# ------------------------------------------------------------------ helpers
class Checks:
    """Every check in the order it ran, each with a level: pass, warn or fail."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, check: str, level: str, detail: str, variant=None, budget=None):
        """Record a check and print it."""
        self.rows.append(dict(check=check, variant=variant, budget=budget,
                              level=level, detail=detail))
        where = ""
        if variant:
            where += f" {variant}"
        if budget:
            where += f" S={budget}"
        print(f"  [{level:4s}] {check}{where}: {detail}", flush=True)

    def n(self, level: str) -> int:
        """How many checks ended at this level."""
        return sum(1 for row in self.rows if row["level"] == level)


def scan_variant(vdir: Path, tag: str) -> dict[int, dict[str, Path]]:
    """Budget -> {theta, result, meta, trials: path} for the files one oodselect_paper.py run
    writes, in budget order."""
    pattern = re.compile(re.escape(tag) + FILE_RE)
    found: dict[int, dict[str, Path]] = {}
    for path in vdir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        budget = int(match.group(1))
        kind = match.group(2).split(".")[0]
        found.setdefault(budget, {})[kind] = path
    return dict(sorted(found.items()))


def top_mask(score: np.ndarray, S: int) -> np.ndarray:
    """The S largest scores, ties broken by column order: the mask oodselect_paper.py reports
    coherence on and regions.py analyses."""
    mask = np.zeros(len(score), bool)
    mask[np.argsort(-np.asarray(score), kind="stable")[:S]] = True
    return mask


def carry_field(w_from: np.ndarray, xy_from: np.ndarray, xy_to: np.ndarray,
                k: int = 10) -> np.ndarray:
    """Each point of xy_to takes the mean weight of its k nearest points of xy_from (all of them
    when there are fewer than k; zeros when there are none)."""
    k_eff = min(k, len(xy_from))
    if k_eff < 1:
        return np.zeros(len(xy_to))
    _, idx = cKDTree(xy_from).query(xy_to, k=k_eff)
    if idx.ndim == 1:          # query drops the neighbour axis when k == 1
        idx = idx[:, None]
    return np.asarray(w_from)[idx].mean(axis=1)


def smoothed_classic(theta: np.ndarray, xy: np.ndarray, S: int, k: int) -> tuple[np.ndarray, dict]:
    """The smoothed-classic ablation mask (docs/REGION-ANALYSIS.md section 0.4) and how sharp its
    cut is: each point takes the mean of sigmoid(theta) over its k nearest OOD points (itself
    included) and the S largest are kept. No search is run, so the smoothing is not tuned.

    sigmoid(theta) rather than theta, whose scale differs per trial. The ranking uses the
    complement expit(-theta), keeping the S smallest: sigmoid(theta) rounds to exactly 1.0 above
    theta ~37 (classic theta reaches ~350), and those ties would be cut by column order.
    """
    q = carry_field(expit(-np.asarray(theta, np.float64)), xy, xy, k=k)
    order = np.argsort(q, kind="stable")
    mask = np.zeros(len(q), bool)
    mask[order[:S]] = True
    cut = q[order[S - 1]]
    tied = q == cut
    next_value = q[order[S]] if S < len(q) else np.nan
    rel_gap = float((next_value - cut) / cut) if cut > 0 else np.nan
    return mask, dict(smooth_cut=float(cut), smooth_n_tied=int(tied.sum()),
                      smooth_n_tied_in=int((tied & mask).sum()), smooth_rel_gap=rel_gap)


def probit_r_np(Z: np.ndarray, acc: np.ndarray, mask: np.ndarray, eps: float = op.EPS) -> float:
    """op.pearson_r_loss re-derived in float64 numpy: the population Pearson r of probit(subset
    accuracy) against probit(ID accuracy), each clamped to [eps, 1 - eps] and each variance
    regularised by eps. It catches an error in the loss arithmetic, which the same-code path
    would reproduce; Z and acc come from op.load_paper_cell in both paths."""
    weights = np.asarray(mask, np.float64)
    subset_acc = (np.asarray(Z, np.float64) @ weights) / (weights.sum() + 1e-6)
    x = ndtri(np.clip(subset_acc, eps, 1 - eps))
    y = ndtri(np.clip(np.asarray(acc, np.float64), eps, 1 - eps))
    xc = x - x.mean()
    yc = y - y.mean()
    cov = np.mean(xc * yc)
    var_x = np.mean(xc ** 2) + eps
    var_y = np.mean(yc ** 2) + eps
    return float(cov / np.sqrt(var_x * var_y))


def components(mask: np.ndarray, A) -> list[int]:
    """Sizes of the connected components of the top-S mask, largest first."""
    if not mask.any():
        return []
    _, labels = connected_components(A[mask][:, mask], directed=False)
    return sorted(np.bincount(labels).tolist(), reverse=True)


def _close(a, b, tol: float) -> bool:
    """a and b agree to tol (relative once |b| > 1); two NaNs agree, a None never does."""
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(b))


def _clean(o):
    """JSON-safe: numpy scalars to Python, NaN to null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ------------------------------------------------------------------ the cell
class CellContext:
    """The cell every stored run is checked against, with the model splits and kNN graphs built
    from it (each built once)."""

    def __init__(self, tag: str, cell, Z, acc, d: int, n_models: int, r_full_random: float):
        self.tag = tag
        self.cell = cell
        self.Z = Z
        self.acc = acc
        self.d = d
        self.n_models = n_models
        self.kinds = cell.models.kind.to_numpy()
        self.r_full_random = r_full_random
        self.splits: dict[tuple, tuple] = {}
        self.graphs: dict[int, object] = {}

    def split(self, rule: str, null_seed, seed: int) -> tuple:
        """(selection, validation, test, the ID accuracies the search saw, full-set r on the test
        models) for a run with this split rule and null seed."""
        key = (rule, null_seed, seed if rule != "random" else 0)   # the random split has no seed
        if key not in self.splits:
            split = op.make_split(self.n_models, rule, self.kinds, seed)
            if rule == "random":
                r_full = self.r_full_random
            else:
                test = split[2]
                r_full = op.hard_corr(self.Z[test], self.acc[test], np.ones(self.d, np.float32),
                                      self.d, is_mask=True)
            acc_search = op.null_accuracies(self.acc, split, null_seed)
            self.splits[key] = (*split, acc_search, r_full)
        return self.splits[key]

    def adjacency(self, knn: int):
        """The knn-NN graph on the OOD points, symmetrised, as the selection built it."""
        if knn not in self.graphs:
            print(f"[audit] building the {knn}-NN graph on {self.d} OOD points", flush=True)
            self.graphs[knn] = graph.build_adjacency(self.cell.lon, self.cell.lat, k=knn)
        return self.graphs[knn]

    def coherence(self, mask: np.ndarray, knn: int) -> dict:
        """graph.coherence_metrics of a mask plus its component sizes; {} without lon/lat."""
        if not self.cell.has_spatial():
            return {}
        A = self.adjacency(knn)
        metrics = graph.coherence_metrics(mask, A, knn, tiles=self.cell.tile_50km,
                                          blocks=self.cell.block_id)
        metrics["cc_sizes"] = components(mask, A)
        return metrics


def check_point_id(ck: Checks, npz_path: Path, d: int) -> None:
    """theta is tied to points by cell column order; a unique point_id per column pins that."""
    with np.load(npz_path, allow_pickle=True) as npz:
        point_id = np.asarray(npz["point_id"]) if "point_id" in npz.files else None
    if point_id is None:
        ck.add("cell point_id", "warn", "absent (cell predates zoo.py's point_id); "
               "theta can only be tied to points by cell column order")
        return
    n_unique = len(pd.unique(point_id))
    ok = len(point_id) == d and n_unique == d
    ck.add("cell point_id", "pass" if ok else "fail",
           f"{len(point_id)} entries, {n_unique} unique, d = {d}")


def aggregate_r(ck: Checks, Z, acc, test, d: int, n_models: int) -> dict:
    """The ID->OOD r over all OOD points, both ways it is quoted: on the test models with the
    loss's eps-regularised probit Pearson (every run stores it as r_full_test; docs/PROJECT-PLAN.md
    section 3, O2), and on all models as a plain Pearson of probits (report_paper.py)."""
    Z_test, acc_test = Z[test], acc[test]
    r_full = op.hard_corr(Z_test, acc_test, np.ones(d, np.float32), d, is_mask=True)
    r_full_np = probit_r_np(Z_test, acc_test, np.ones(d, bool))
    probit_id = norm.ppf(np.clip(acc, op.EPS, 1 - op.EPS))
    probit_ood = norm.ppf(np.clip(Z.mean(1), op.EPS, 1 - op.EPS))
    r_all = float(np.corrcoef(probit_id, probit_ood)[0, 1])
    lo, hi = op.fisher_ci(r_full, len(test))
    ck.add("aggregate r (numpy re-derivation)",
           "pass" if _close(r_full_np, r_full, TOL_NUMPY) else "fail",
           f"{r_full:+.4f} torch vs {r_full_np:+.4f} numpy")
    return dict(r_full_test=r_full, r_full_test_ci=[lo, hi], r_full_test_numpy=r_full_np,
                r_all_models=r_all, n_test_models=len(test), n_models=n_models, d=d)


# ------------------------------------------------------------------ one stored run
def meta_matches(meta: dict, d: int, n_models: int, rule: str, null_seed, split: tuple) -> bool:
    """meta.json holds the d, model count, control and model lists the run should have used. The
    stored test r is only held out if the lists agree."""
    selection, validation, test = split
    lists = meta.get("split", {})
    return (meta.get("d") == d and meta.get("n_models") == n_models
            and meta.get("split_rule", "random") == rule
            and meta.get("null_seed") == null_seed
            and lists.get("selection") == selection.tolist()
            and lists.get("validation") == validation.tolist()
            and lists.get("test") == test.tolist())


def check_top_masks(ck: Checks, theta: np.ndarray, mask: np.ndarray, S: int, variant: str) -> bool:
    """Check 3: the run took r on torch.topk's top S and coherence on the stable argsort's, and
    regions.py takes argsort's. A tie straddling the cut is a warning even when the two agree.
    Returns whether they agree."""
    torch = op._torch()
    topk = op.binary_selection(torch.as_tensor(theta), S).numpy().astype(bool)
    tied = theta == theta[mask].min()
    tie_note = f"{int(tied.sum())} theta entries tie at the cut, {int((tied & mask).sum())} inside"
    if not np.array_equal(topk, mask):
        ck.add("top-S masks agree (torch.topk vs argsort)", "fail",
               f"{int((topk != mask).sum())} points differ; {tie_note}", variant=variant, budget=S)
        return False
    if (tied & ~mask).any():
        ck.add("top-S masks agree (torch.topk vs argsort)", "warn",
               f"one set, but {tie_note}: the selection rests on the tie rule",
               variant=variant, budget=S)
    return True


def recompute_r(ctx: CellContext, ck: Checks, res, theta, mask, S: int, split: tuple,
                acc_search, variant: str) -> dict:
    """Check 2: the stored train / validation / test r recomputed with the selection's own code
    (train and validation against the accuracies the search saw), test r re-derived in float64
    numpy, and the scipy Pearson / Spearman columns. Returns the row's recompute columns."""
    selection, validation, test = split
    Z, acc = ctx.Z, ctx.acc
    same = {
        "train": op.hard_corr(Z[selection], acc_search[selection], theta, S),
        "val": op.hard_corr(Z[validation], acc_search[validation], theta, S),
        "test": op.hard_corr(Z[test], acc[test], theta, S),
    }
    Z_test, acc_test = Z[test], acc[test]
    r_numpy = probit_r_np(Z_test, acc_test, mask)
    stats = op.plain_stats(Z_test, acc_test, mask)
    diffs = {f"r_{part}": abs(r - float(res[f"r_{part}"])) for part, r in same.items()}
    absdiff_numpy = abs(r_numpy - float(res.r_test))

    ok_same = all(x <= TOL_SAME for x in diffs.values())
    ck.add("test r recomputed (same code)", "pass" if ok_same else "fail",
           f"stored {res.r_test:+.6f}, recomputed {same['test']:+.6f}; largest abs diff "
           f"over train/val/test {max(diffs.values()):.1e}", variant=variant, budget=S)
    subset_acc = Z_test[:, mask].mean(1)
    n_clamp = int(((subset_acc <= 0) | (subset_acc >= 1)).sum())
    ck.add("test r recomputed (numpy float64)",
           "pass" if absdiff_numpy <= TOL_NUMPY else "fail",
           f"{r_numpy:+.6f}, abs diff {absdiff_numpy:.1e}; {n_clamp} test models "
           f"with subset accuracy 0 or 1", variant=variant, budget=S)
    ok_stats = (_close(stats["pearson"], res.pearson_test_scipy, TOL_SAME)
                and _close(stats["spearman"], res.spearman_test, TOL_SAME))
    ck.add("scipy Pearson / Spearman recomputed", "pass" if ok_stats else "fail",
           f"{stats['pearson']:+.6f} / {stats['spearman']:+.6f} vs stored "
           f"{res.pearson_test_scipy:+.6f} / {res.spearman_test:+.6f}",
           variant=variant, budget=S)
    return dict(r_test_same_code=same["test"], r_test_numpy=r_numpy,
                absdiff_same_code=diffs["r_test"], absdiff_numpy=absdiff_numpy)


def coherence_columns(metrics: dict) -> dict:
    """The coh_* table columns and the sizes of the components the region analysis tests."""
    columns = {f"coh_{key}": metrics[key] for key in COH_KEYS}
    columns["cc_sizes_ge_min"] = [size for size in metrics["cc_sizes"] if size >= MIN_COMPONENT]
    return columns


def check_coherence(ck: Checks, metrics: dict, res, variant: str, S: int) -> None:
    """Check 4: regions.py rebuilds the same graph, so a stored coh_* value that does not come
    back means its components are not the ones the selection saw."""
    stored = {key: res.get(f"coh_{key}") for key in COH_KEYS if f"coh_{key}" in res}
    if not stored:
        ck.add("coherence vs stored", "warn", "result.csv has no coh_* columns",
               variant=variant, budget=S)
        return
    wrong = {key: (metrics[key], value) for key, value in stored.items()
             if not _close(metrics[key], value, TOL_COH)}
    if wrong:
        ck.add("coherence vs stored", "fail", f"(recomputed, stored): {wrong}",
               variant=variant, budget=S)


def audit_run(ctx: CellContext, ck: Checks, variant: str, S: int, theta: np.ndarray, files: dict,
              recompute: bool) -> tuple[dict, np.ndarray, bool]:
    """Check one stored run against the cell; return its table row, its top-S mask and whether
    torch.topk and argsort agree on it. A control run (variant suffix _null<N>, _trees2others,
    _others2trees) is checked against the split and null seed its result.csv records."""
    res = pd.read_csv(files["result"]).iloc[0]
    rule, null_seed, run_seed = op.run_controls(res)
    selection, validation, test, acc_search, r_full_expected = ctx.split(rule, null_seed, run_seed)
    split = (selection, validation, test)

    want = dict(cell=ctx.tag, variant=variant, budget=S, d=ctx.d,
                n_sel=len(selection), n_val=len(validation), n_test=len(test))
    wrong = {key: res.get(key) for key, value in want.items() if res.get(key) != value}
    if wrong:
        ck.add("result.csv header", "fail", f"disagrees with the cell: {wrong} (want {want})",
               variant=variant, budget=S)
    knn = int(res.knn)

    if "meta" in files:
        meta = json.loads(files["meta"].read_text())
        if not meta_matches(meta, ctx.d, ctx.n_models, rule, null_seed, split):
            ck.add("meta.json split", "fail",
                   f"split / d / n_models / null seed differ from make_split({rule!r}, "
                   f"seed {run_seed}) on this cell", variant=variant, budget=S)

    mask = top_mask(theta, S)
    masks_agree = check_top_masks(ck, theta, mask, S, variant)
    control = rule != "random" or null_seed is not None
    row = dict(variant=variant, budget=S, knn=knn, split=rule, null_seed=null_seed,
               n_test=len(test), control=control, r_test=float(res.r_test),
               r_test_ci_lo=float(res.r_test_ci_lo), r_test_ci_hi=float(res.r_test_ci_hi),
               r_full_test_stored=float(res.r_full_test))
    if recompute:
        row.update(recompute_r(ctx, ck, res, theta, mask, S, split, acc_search, variant))
    metrics = ctx.coherence(mask, knn)
    if metrics:
        row.update(coherence_columns(metrics))
        check_coherence(ck, metrics, res, variant, S)
    row["r_full_test_expected"] = r_full_expected
    return row, mask, masks_agree


def check_variant_rows(ck: Checks, variant: str, vrows: list[dict]) -> None:
    """Every run of a variant stores the full-set r of its split, and all use one control."""
    if not vrows:
        return
    r_full_stored = [row["r_full_test_stored"] for row in vrows]
    r_full_want = sorted({row["r_full_test_expected"] for row in vrows})
    ok_full = all(_close(row["r_full_test_stored"], row["r_full_test_expected"], TOL_SAME)
                  for row in vrows)
    ck.add("stored r_full_test", "pass" if ok_full else "fail",
           f"{len(r_full_stored)} result.csv files, range {min(r_full_stored):+.6f} .. "
           f"{max(r_full_stored):+.6f}; recomputed "
           + ", ".join(f"{x:+.6f}" for x in r_full_want), variant=variant)
    controls = sorted({(row["split"], str(row["null_seed"])) for row in vrows})
    if len(controls) > 1:
        ck.add("one control per variant", "fail",
               f"result.csv files disagree on (split, null seed): {controls}", variant=variant)


# ------------------------------------------------------------------ ablation and report
def smoothed_classic_rows(ctx: CellContext, ck: Checks, budgets: list[int], thetas: dict,
                          masks: dict, rows: list[dict], test) -> list[dict]:
    """The smoothed-classic rows, scored on the random split's test models like a stored
    selection, at the classic run's knn."""
    new_rows = []
    # Projected to EPSG:3086 metres, as the selection's kNN graph is.
    xy = graph.project(ctx.cell.lon, ctx.cell.lat) if ctx.cell.has_spatial() else None
    Z_test, acc_test = ctx.Z[test], ctx.acc[test]
    for S in budgets:
        if ("classic", S) not in thetas or xy is None:
            ck.add("smoothed-classic", "warn", "needs the classic theta and lon/lat; skipped",
                   budget=S)
            continue
        knn = next(row["knn"] for row in rows if row["variant"] == "classic" and row["budget"] == S)
        name = f"smoothed_classic_knn{knn}"
        mask, cut = smoothed_classic(thetas[("classic", S)], xy, S, knn)
        n_tied, n_tied_in = cut["smooth_n_tied"], cut["smooth_n_tied_in"]
        n_out = n_tied - n_tied_in
        if n_tied > 1:
            verdict = ("the mask depends on the cell's column order" if n_out
                       else "all inside, so the mask does not depend on column order")
            ck.add("smoothed-classic cut", "warn",
                   f"{n_tied} points tie at the S-th smoothed value {cut['smooth_cut']:.3e} "
                   f"({n_tied_in} inside the cut, {n_out} outside): " + verdict,
                   variant=name, budget=S)
        else:
            ck.add("smoothed-classic cut", "pass",
                   f"unique S-th smoothed value {cut['smooth_cut']:.3e}; the next is "
                   f"{cut['smooth_rel_gap']:.1e} (relative) above", variant=name, budget=S)

        r = op.hard_corr(Z_test, acc_test, mask.astype(np.float32), S, is_mask=True)
        lo, hi = op.fisher_ci(r, len(test))
        metrics = ctx.coherence(mask, knn)
        classic_mask = masks[("classic", S)]
        jaccard = float((mask & classic_mask).sum() / (mask | classic_mask).sum())
        row = dict(variant=name, budget=S, knn=knn, r_test=r, r_test_ci_lo=lo, r_test_ci_hi=hi,
                   r_test_numpy=probit_r_np(Z_test, acc_test, mask),
                   jaccard_vs_classic=jaccard, **cut)
        row.update(coherence_columns(metrics))
        new_rows.append(row)
        print(f"[audit] {name} S={S}: test r {r:+.3f}, purity {metrics['purity']:.2f}, "
              f"{metrics['n_cc']} components, {metrics['tiles']} tiles", flush=True)
    return new_rows


def add_dr_vs_classic(rows: list[dict]) -> None:
    """Δr = r_variant - r_classic at the same budget, the cost of coherence (positive = a weaker
    inversion, the sign of report_paper.py's comparison.csv). Only for runs on the random split with
    the true accuracies; a control's r is read on its own."""
    classic_r = {row["budget"]: row["r_test"] for row in rows
                 if row["variant"] == "classic" and not row.get("control")}
    for row in rows:
        if row["variant"] != "classic" and not row.get("control") and row["budget"] in classic_r:
            row["dr_vs_classic"] = row["r_test"] - classic_r[row["budget"]]


def per_split_aggregate(rows: list[dict]) -> dict:
    """The full-set r of each family split a control ran on, with its Fisher 95% CI."""
    out = {}
    for row in rows:
        if row.get("split", "random") == "random":
            continue
        r_full, n_test = row["r_full_test_expected"], row["n_test"]
        out[row["split"]] = dict(r_full_test=r_full, n_test=n_test,
                                 ci=list(op.fisher_ci(r_full, n_test)))
    return out


# ------------------------------------------------------------------ audit
def audit(a) -> dict:
    """Run every check on the variants under a.root (a is the parsed command line); return the
    report main() writes."""
    t0 = time.time()
    root = Path(a.root)
    ck = Checks()
    torch = op._torch()
    torch.set_num_threads(max(1, a.threads))

    cell, Z, acc, info = op.load_paper_cell(a.cells_dir, a.tag)
    d, n_models = info["d"], info["n_models"]
    selection, validation, test = op.model_split(n_models)
    n_tiles = int(len(np.unique(cell.tile_50km))) if cell.tile_50km is not None else -1
    print(f"[audit] {a.tag}: {n_models} models (selection {len(selection)} | validation "
          f"{len(validation)} | test {len(test)}), d = {d} OOD points in {n_tiles} tiles",
          flush=True)

    check_point_id(ck, Path(a.cells_dir) / f"{a.tag}.npz", d)
    aggregate = aggregate_r(ck, Z, acc, test, d, n_models)
    ctx = CellContext(a.tag, cell, Z, acc, d, n_models, aggregate["r_full_test"])
    if not cell.has_spatial():
        ck.add("cell lon/lat", "warn", "absent: no graph, so no coherence numbers")

    rows, files_summary, masks, thetas = [], {}, {}, {}
    grid = op.budgets_for(d)
    for v in a.variants:
        vdir = root / v
        if not vdir.is_dir():
            ck.add("variant folder", "warn", f"{vdir} does not exist; skipped", variant=v)
            continue
        print(f"[audit] {v}: {vdir}", flush=True)
        found = scan_variant(vdir, a.tag)
        budgets = list(found)
        missing_grid = [b for b in grid if b not in found]
        files_summary[v] = dict(budgets=budgets, missing_from_grid=missing_grid)
        if not budgets:
            ck.add("budgets present", "warn", f"no {a.tag}_S*_ files", variant=v)
            continue
        detail = f"{len(budgets)} of the {len(grid)}-budget grid"
        if missing_grid:
            detail += f"; missing {missing_grid}"
        ck.add("budgets present", "warn" if missing_grid else "pass", detail, variant=v)
        off_grid = [b for b in budgets if b not in grid]
        if off_grid:
            ck.add("budgets present", "warn", f"budgets off the paper grid: {off_grid}", variant=v)

        theta_bad, knn_seen, mask_bad = [], set(), []
        for S, files in found.items():
            missing = [kind for kind in ("theta", "result", "meta", "trials") if kind not in files]
            if missing:
                level = "fail" if set(missing) & {"theta", "result"} else "warn"
                ck.add("files per budget", level, f"missing {missing}", variant=v, budget=S)
            if "theta" not in files or "result" not in files:
                continue
            # regions.py and the boundary analysis index the cell with theta, so a theta from
            # another cell would silently map the selection onto the wrong points.
            theta = np.load(files["theta"])
            if theta.ndim != 1 or theta.shape[0] != d or not np.isfinite(theta).all():
                n_nonfinite = int((~np.isfinite(theta)).sum())
                theta_bad.append(f"S={S}: shape {theta.shape}, {n_nonfinite} non-finite")
                continue
            thetas[(v, S)] = theta
            row, mask, masks_agree = audit_run(ctx, ck, v, S, theta, files,
                                               S in a.recompute_budgets)
            masks[(v, S)] = mask
            knn_seen.add(row["knn"])
            if not masks_agree:
                mask_bad.append(S)
            rows.append(row)

        n_ok = sum(1 for variant, _ in thetas if variant == v)
        detail = f"{n_ok} of {len(budgets)} theta files are finite ({d},) vectors"
        if theta_bad:
            detail += f"; bad: {theta_bad}"
        ck.add("theta has one entry per OOD point", "fail" if theta_bad else "pass", detail,
               variant=v)
        if n_ok and not mask_bad:
            ck.add("top-S masks agree (torch.topk vs argsort)", "pass",
                   f"one set at all {n_ok} budgets", variant=v)
        if len(knn_seen) > 1:
            ck.add("knn", "fail", f"result.csv files disagree on knn: {sorted(knn_seen)}",
                   variant=v)
        coherence_flagged = any(c["variant"] == v and c["check"] == "coherence vs stored"
                                for c in ck.rows)
        if cell.has_spatial() and n_ok and not coherence_flagged:
            ck.add("coherence vs stored", "pass",
                   f"every stored coh_* column reproduced at all {n_ok} budgets", variant=v)
        check_variant_rows(ck, v, [row for row in rows if row["variant"] == v])

    for v in a.variants:
        if v not in files_summary:
            continue
        for S in a.check_budgets:
            if (v, S) not in masks:
                ck.add("table budget", "warn", "not available; row left blank", variant=v, budget=S)
        for S in a.recompute_budgets:
            if (v, S) not in masks:
                ck.add("recompute budget", "warn", "not available; r not recomputed",
                       variant=v, budget=S)

    if a.smoothed_classic:
        rows += smoothed_classic_rows(ctx, ck, a.check_budgets, thetas, masks, rows, test)
    add_dr_vs_classic(rows)

    return dict(tag=a.tag, root=str(root), cells_dir=str(a.cells_dir),
                written=time.strftime("%Y-%m-%d %H:%M:%S"), secs=time.time() - t0,
                variants=a.variants, check_budgets=a.check_budgets,
                recompute_budgets=a.recompute_budgets, n_tiles_ood=n_tiles,
                split=dict(selection=len(selection), validation=len(validation), test=len(test)),
                per_split_aggregate=per_split_aggregate(rows),
                y_id_stored=info["y_id_stored"], aggregate=aggregate, files=files_summary,
                rows=rows, checks=ck.rows,
                n_fail=ck.n("fail"), n_warn=ck.n("warn"), n_pass=ck.n("pass"))


# ------------------------------------------------------------------ markdown
def table_row(row: dict | None, variant: str, S: int, files: dict, n_tiles: int) -> str:
    """One line of the per-budget table; a blank line when the run is missing or unusable."""
    if row is None:
        ran = S in files.get(variant, {}).get("budgets", [])
        status = "files unusable, see checks" if ran else "not run"
        return f"| {variant} | {S} | {status} | | | | | | | |"
    dr = f"{row['dr_vs_classic']:+.3f}" if "dr_vs_classic" in row else ""
    if "absdiff_same_code" in row:
        recomputed = f"{row['absdiff_same_code']:.0e} / {row['absdiff_numpy']:.0e}"
    elif variant.startswith("smoothed_classic"):
        recomputed = f"computed here (numpy {row['r_test_numpy']:+.3f})"
    else:
        recomputed = ""
    if "coh_purity" in row:
        big = row["cc_sizes_ge_min"]
        sizes = f"{len(big)} ({', '.join(map(str, big))})" if big else "none"
        coherence = (f"{row['coh_purity']:.2f} | {row['coh_n_cc']} | {sizes} | "
                     f"{row['coh_largest_cc_frac']:.2f} | {row['coh_tiles']} of {n_tiles}")
    else:
        coherence = " | | | | "
    label = variant
    if row.get("split", "random") != "random":
        label = f"{variant} (on {row['n_test']} test models)"
    return (f"| {label} | {S} | {row['r_test']:+.3f} [{row['r_test_ci_lo']:+.2f}, "
            f"{row['r_test_ci_hi']:+.2f}] | {dr} | {recomputed} | {coherence} |")


def to_markdown(rep: dict) -> str:
    """The report as markdown: the aggregate r, the per-budget table and every check."""
    ag, sp = rep["aggregate"], rep["split"]
    lines = [
        f"### Selection audit: `{rep['tag']}`",
        "",
        f"Run `{rep['root']}`; cell `{rep['cells_dir']}`. {ag['n_models']} models "
        f"(selection {sp['selection']} / validation {sp['validation']} / test {sp['test']}), "
        f"d = {ag['d']:,} OOD points in {rep['n_tiles_ood']} test tiles.",
        "",
        f"Aggregate ID→OOD r over all {ag['d']:,} OOD points: **{ag['r_full_test']:+.3f}** "
        f"[{ag['r_full_test_ci'][0]:+.2f}, {ag['r_full_test_ci'][1]:+.2f}] on the "
        f"{ag['n_test_models']} test models of the random 60/20/20 split (TestSetFinder.test, "
        f"stored as `r_full_test` by every random-split run); "
        f"{ag['r_all_models']:+.3f} on all {ag['n_models']} models (plain Pearson of probits, "
        f"`report_paper.py`).",
    ]
    for split_name, split_r in rep.get("per_split_aggregate", {}).items():
        lines.append(f"Family split {split_name}: full-set r **{split_r['r_full_test']:+.3f}** "
                     f"[{split_r['ci'][0]:+.2f}, {split_r['ci'][1]:+.2f}] on its "
                     f"{split_r['n_test']} test models (its runs' `r_full_test`).")
    lines += [
        "",
        "| variant | S | test r [95% CI] | Δr vs classic | recomputed, abs diff (same code / "
        "numpy) | kNN purity | components | components ≥ 20 pts (sizes) | largest | tiles |",
        "|---|---:|---|---:|---|---:|---:|---|---:|---:|",
    ]

    # Variants in --variants order, then the others (the smoothed-classic rows) by name; the
    # name breaks the tie, as set order changes with the hash seed.
    listed = {v: i for i, v in enumerate(rep["variants"])}
    present = {row["variant"] for row in rep["rows"]} | set(rep["files"])
    variants = sorted(present, key=lambda v: (listed.get(v, len(listed)), v))
    rows = {(row["variant"], row["budget"]): row for row in rep["rows"]}
    for S in rep["check_budgets"]:
        for v in variants:
            lines.append(table_row(rows.get((v, S)), v, S, rep["files"], rep["n_tiles_ood"]))

    lines += [
        "",
        f"Checks: {rep['n_pass']} ✓, {rep['n_warn']} ⚠, {rep['n_fail']} ✗.",
        "",
        "| check | variant | S | result | verdict |",
        "|---|---|---:|---|:---:|",
    ]
    for c in rep["checks"]:
        lines.append(f"| {c['check']} | {c['variant'] or ''} | {c['budget'] or ''} | "
                     f"{c['detail'].replace('|', '/')} | {MARK[c['level']]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results/<date>/oodselect_paper")
    ap.add_argument("--cells-dir", required=True, help="folder holding <tag>.npz and _models.csv")
    ap.add_argument("--tag", required=True, help="cell tag, e.g. FL_band_AE_s0")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    help="variant folders under --root, classic first; more spatial_mu<mu>_knn10 "
                         "folders make the table a mu sweep")
    ap.add_argument("--check-budgets", nargs="+", type=int, default=CHECK_BUDGETS,
                    help="budgets in the table")
    ap.add_argument("--recompute-budgets", nargs="+", type=int, default=RECOMPUTE_BUDGETS,
                    help="budgets whose stored r is recomputed from theta and the cell")
    ap.add_argument("--smoothed-classic", action="store_true",
                    help="add the smoothed-classic ablation rows at --check-budgets")
    ap.add_argument("--out", help="folder for the .json and .md (default: --root); the names "
                                  "carry only the tag, so a second audit of one cell needs its own")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads (the selections ran with 1)")
    ap.add_argument("--overwrite", action="store_true", help="redo an audit already written")
    a = ap.parse_args()

    out = Path(a.out or a.root)
    out.mkdir(parents=True, exist_ok=True)
    js = out / f"{a.tag}_selection_audit.json"
    md = out / f"{a.tag}_selection_audit.md"
    if js.exists() and md.exists() and not a.overwrite:
        print(f"{md} exists; skipping (--overwrite to redo)")
        return 0

    rep = audit(a)
    text = to_markdown(rep)
    js.write_text(json.dumps(_clean(rep), indent=1))
    md.write_text(text)
    print("\n" + text, flush=True)
    print(f"[audit] {rep['n_pass']} passed, {rep['n_warn']} warnings, {rep['n_fail']} FAILED "
          f"in {rep['secs']:.0f} s", flush=True)
    print(f"[audit] wrote {js}\n[audit] wrote {md}", flush=True)
    return 1 if rep["n_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
