#!/usr/bin/env python3
"""
audit_selection.py -- the C3 audit of one paper-protocol OODSelect run
(oodselect_paper.py): does every number the run stored follow from the stored
selection and the cell, and what are the headline numbers at S = 250, 500, 1000?

Reads   <root>/<variant>/<cell>_S<S>_{theta.npy,result.csv,meta.json,trials.csv}
        <cells-dir>/<cell>.npz + <cell>_models.csv
Writes  <out>/<cell>_selection_audit.json   every check and every number
        <out>/<cell>_selection_audit.md     the same as tables to paste into the
                                            audit report (also printed)
Exit 1 when any check FAILS; warnings (a budget not run, a v1 cell without
point_id, a tied cut) do not fail the audit.

WHAT IS CHECKED, AND WHY EACH ONE
---------------------------------
1. Files. Every *_theta.npy is a finite vector of exactly d = S_ood.shape[1]
   entries, one per OOD point in cell column order: regions.py and the
   boundary analysis index the cell with it, so a theta from another cell (a
   rebuilt cell, an A->AE run copied into the wrong folder) would silently map
   selections onto the wrong points. Each result.csv must name the same cell,
   budget, d and split sizes; each meta.json must hold exactly the selection /
   validation / test model lists that model_split() gives today. The stored
   test r is only "held out" if those lists agree.
2. Test r, recomputed two ways at --recompute-budgets (default 500 and 1000):
     same code    op.hard_corr on Z[test], acc[test], theta -- what
                  oodselect_paper.main ran. It must match to 1e-6 and catches a
                  stale or mismatched file.
     re-derived   float64 numpy, written from the definition (top-S of theta by
                  stable argsort, subset accuracy = Z @ m / (S + 1e-6), probit
                  with the clamp at eps = 1e-6 on both sides, population
                  moments, + eps on both variances). It must match to 1e-3. It
                  re-derives the loss arithmetic (mask, subset accuracy, probit,
                  moments) independently, so an error there, which the same-code
                  path would reproduce faithfully, shows up. Z and acc come from
                  op.load_paper_cell in both paths, so an error in the
                  correctness / ID-accuracy derivation (the p = 0.5 tie rule,
                  y_id) would not. Away from the clamp the two agree to ~1e-7
                  (v1, S >= 250: 1e-8 to 1e-7). A test model whose subset
                  accuracy is exactly 0 or 1 (S <= ~100) sits at the clamp,
                  where torch's float32 erfinv moves ~0.01 per ulp; on v1 that
                  put the two paths 2e-5 to 5e-5 apart at S = 10 and 100. The
                  tolerance allows for that; the number of such models is
                  printed with the check.
   The train and validation r, the scipy Pearson / Spearman columns and the
   full-set r are recomputed with the same code.
3. Top-S masks, at every stored budget: the run took r on torch.topk's top S
   and coherence on numpy's stable argsort, and regions.py takes argsort's.
   They must be one set, or the regions are not the selection whose r is
   reported. theta tying at the cut is where they can part; a tie that
   straddles the cut is a warning even when the two happen to agree.
4. Coherence, recomputed at every stored budget: the 10-NN graph is rebuilt
   exactly as the selection built it (graph.build_adjacency on the cell's
   lon/lat, EPSG:3086, k from result.csv, symmetrised) and
   graph.coherence_metrics must reproduce every stored coh_* column. regions.py
   rebuilds the same graph; if this disagrees, every component downstream is a
   different object from the one the selection saw.
5. The aggregate ID->OOD r over ALL OOD points, two ways, because both are
   quoted: on the 60 test models with the authors' eps-regularised probit
   Pearson (TestSetFinder.test; every result.csv stores it as r_full_test;
   PROJECT-PLAN section 3 quotes it as +0.78), and on all models with a plain
   Pearson of probits (report_paper.py's "full-set ID->OOD r (all models)").

THE TABLE (S = --check-budgets)
-------------------------------
Per variant and budget: stored test r with its Fisher 95% CI, the change
against classic at the same S (r_variant - r_classic, O4's cost of coherence:
positive = a weaker inversion, the sign of report_paper's comparison.csv), kNN
purity, connected components of the top-S mask, the sizes of components of
>= 20 points (the ones the region analysis tests; smaller ones are listed
only), the largest component's share, and the 50 km tiles touched out of the
tiles in the OOD half. Pass more --variants (spatial_mu1_knn10
spatial_mu10_knn10 ...) and the same table is the mu sweep; a variant run at
only some budgets is a warning, not a failure.

SMOOTHED-CLASSIC (--smoothed-classic; the ablation of PROJECT-PLAN O4)
----------------------------------------------------------------------
docs/REGION-ANALYSIS.md section 0.4: the mu = 0 weights passed through the E2
kNN-carry operator and thresholded to the same S. The definition:
    z = sigmoid(theta_classic)          the relaxed selection classic optimised
    s = transfer.carry_field(z, xy, xy, k)  each point takes the mean z of its
                                        k nearest OOD points in EPSG:3086; on the
                                        same point set the nearest one is the
                                        point itself, so that is itself + k - 1
    mask = the S largest s
scored on the test models exactly like a stored selection. z rather than theta
because theta is unbounded and its scale differs per trial (lr runs to 100),
so averaging raw theta lets one large entry decide a neighbourhood. No search
is run: the ablation asks what post-hoc smoothing of the classic answer buys,
so it must not be tuned.
It is computed on the complement, the same ranking without the rounding:
    q = expit(-theta) = 1 - z, in float64;  s_q = carry_field(q) = 1 - s;
    mask = the S smallest s_q (stable argsort).
z itself rounds to exactly 1.0 once theta > ~37, and classic theta from a
high-lr trial reaches |theta| ~350 (v1 S = 10, 20, 100). Thousands of points
then scored exactly 1.0 and tied at the cut, the stable argsort filled it by
cell column order, and reversing the columns changed every selected point and
moved r by up to 0.2. expit(-theta) stays representable to theta ~ 745. A cut
that still ties (more than one point at the S-th smoothed value) is a warning
with the tie count; otherwise the check gives the relative gap to the next
value, which is how far the ranking is from depending on column order.

USAGE
=====
    python aotil/audit_selection.py --root results/<date>/oodselect_paper \\
        --cells-dir data/interim/aotil/cells --tag FL_band_AE_s0
    python aotil/audit_selection.py ... --smoothed-classic        # + ablation rows
    python aotil/audit_selection.py ... --variants classic spatial_mu1_knn10 \\
        spatial_mu3_knn10 spatial_mu10_knn10 --out <dir>          # mu sweep
    python aotil/audit_selection.py ... --overwrite               # redo
The output names carry only the cell tag, so give a second audit of the same
cell (the mu sweep) its own --out.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph                     # noqa: E402
import oodselect_paper as op     # noqa: E402

DEFAULT_VARIANTS = ["classic", "spatial_mu3_knn10"]
CHECK_BUDGETS = [250, 500, 1000]
RECOMPUTE_BUDGETS = [500, 1000]
MIN_COMPONENT = 20               # REGION-ANALYSIS-FLORIDA.md: smaller ones are listed, not tested
TOL_SAME = 1e-6                  # same code, same float32 arithmetic
TOL_NUMPY = 1e-3                 # float64 re-derivation against float32 torch (docstring, 2)
TOL_COH = 1e-9
COH_KEYS = ("purity", "P_hard", "n_cc", "largest_cc_frac", "tiles", "blocks", "moran_I")
FILE_RE = r"_S(\d+)_(theta\.npy|result\.csv|meta\.json|trials\.csv)"
MARK = {"pass": "✓", "warn": "⚠", "fail": "✗"}


# ------------------------------------------------------------------ bookkeeping
class Checks:
    """Every check, in order, with a pass / warn / fail level."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, check: str, level: str, detail: str, variant=None, budget=None):
        self.rows.append(dict(check=check, variant=variant, budget=budget,
                              level=level, detail=detail))
        where = "".join(f" {x}" for x in (variant, f"S={budget}" if budget else None) if x)
        print(f"  [{level:4s}] {check}{where}: {detail}", flush=True)

    def n(self, level: str) -> int:
        return sum(r["level"] == level for r in self.rows)


def scan_variant(vdir: Path, tag: str) -> dict[int, dict[str, Path]]:
    """Budget -> the four files one oodselect_paper.py task writes."""
    out: dict[int, dict[str, Path]] = {}
    pat = re.compile(re.escape(tag) + FILE_RE)
    for p in vdir.iterdir():
        m = pat.fullmatch(p.name)
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2).split(".")[0]] = p
    return dict(sorted(out.items()))


def top_mask(score: np.ndarray, S: int) -> np.ndarray:
    """The mask oodselect_paper.main reports coherence on."""
    m = np.zeros(len(score), bool)
    m[np.argsort(-np.asarray(score), kind="stable")[:S]] = True
    return m


def smoothed_classic(theta: np.ndarray, xy: np.ndarray, S: int, k: int) -> tuple[np.ndarray, dict]:
    """The smoothed-classic mask (docstring, SMOOTHED-CLASSIC) and how sharp its cut is."""
    from scipy.special import expit
    from transfer import carry_field
    q = carry_field(expit(-np.asarray(theta, np.float64)), xy, xy, k=k)
    order = np.argsort(q, kind="stable")
    m = np.zeros(len(q), bool)
    m[order[:S]] = True
    cut = q[order[S - 1]]
    tied = q == cut
    nxt = q[order[S]] if S < len(q) else np.nan
    return m, dict(smooth_cut=float(cut), smooth_n_tied=int(tied.sum()),
                   smooth_n_tied_in=int((tied & m).sum()),
                   smooth_rel_gap=float((nxt - cut) / cut) if cut > 0 else np.nan)


def probit_r_np(Z: np.ndarray, acc: np.ndarray, mask: np.ndarray, eps: float = op.EPS) -> float:
    """TestSetFinder.test re-derived in float64 numpy (see the docstring, 2)."""
    from scipy.special import ndtri
    m = np.asarray(mask, np.float64)
    a = (np.asarray(Z, np.float64) @ m) / (m.sum() + 1e-6)
    x = ndtri(np.clip(a, eps, 1 - eps))
    y = ndtri(np.clip(np.asarray(acc, np.float64), eps, 1 - eps))
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.mean(xc * yc) / np.sqrt((np.mean(xc ** 2) + eps) * (np.mean(yc ** 2) + eps)))


def components(mask: np.ndarray, A) -> list[int]:
    """Sizes of the connected components of the top-S mask, largest first."""
    from scipy.sparse.csgraph import connected_components
    if not mask.any():
        return []
    _, lab = connected_components(A[mask][:, mask], directed=False)
    return sorted(np.bincount(lab).tolist(), reverse=True)


def _close(a, b, tol: float) -> bool:
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


# ------------------------------------------------------------------ audit
def audit(a) -> dict:
    t0 = time.time()
    root = Path(a.root)
    ck = Checks()
    torch = op._torch()
    torch.set_num_threads(max(1, a.threads))

    cell, Z, acc, info = op.load_paper_cell(a.cells_dir, a.tag)
    d, n_models = info["d"], info["n_models"]
    tr, va, te = op.model_split(n_models)
    Zte, ate = Z[te], acc[te]
    n_tiles = int(len(np.unique(cell.tile_50km))) if cell.tile_50km is not None else -1
    print(f"[audit] {a.tag}: {n_models} models (selection {len(tr)} | validation {len(va)} | "
          f"test {len(te)}), d = {d} OOD points in {n_tiles} tiles", flush=True)

    # The cell itself, only what the selection files depend on.
    npz = np.load(Path(a.cells_dir) / f"{a.tag}.npz", allow_pickle=True)
    if "point_id" in npz.files:
        pid = np.asarray(npz["point_id"])
        ok = len(pid) == d and len(pd.unique(pid)) == d
        ck.add("cell point_id", "pass" if ok else "fail",
               f"{len(pid)} entries, {len(pd.unique(pid))} unique, d = {d}")
    else:
        ck.add("cell point_id", "warn", "absent (cell predates zoo.py's point_id); "
               "theta can only be tied to points by cell column order")

    # Aggregate ID->OOD r over the full OOD set.
    from scipy.stats import norm
    r_full = op.hard_corr(Zte, ate, np.ones(d, np.float32), d, is_mask=True)
    r_full_np = probit_r_np(Zte, ate, np.ones(d, bool))
    r_all = float(np.corrcoef(norm.ppf(np.clip(acc, op.EPS, 1 - op.EPS)),
                              norm.ppf(np.clip(Z.mean(1), op.EPS, 1 - op.EPS)))[0, 1])
    lo, hi = op.fisher_ci(r_full, len(te))
    aggregate = dict(r_full_test=r_full, r_full_test_ci=[lo, hi], r_full_test_numpy=r_full_np,
                     r_all_models=r_all, n_test_models=len(te), n_models=n_models, d=d)
    ck.add("aggregate r (numpy re-derivation)",
           "pass" if _close(r_full_np, r_full, TOL_NUMPY) else "fail",
           f"{r_full:+.4f} torch vs {r_full_np:+.4f} numpy")

    graphs: dict[int, object] = {}

    def adjacency(knn: int):
        if knn not in graphs:
            print(f"[audit] building the {knn}-NN graph on {d} OOD points", flush=True)
            graphs[knn] = graph.build_adjacency(cell.lon, cell.lat, k=knn)
        return graphs[knn]

    def coherence(mask, knn):
        if not cell.has_spatial():
            return {}
        A = adjacency(knn)
        cm = graph.coherence_metrics(mask, A, knn, tiles=cell.tile_50km, blocks=cell.block_id)
        cm["cc_sizes"] = components(mask, A)
        return cm

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
        budgets = sorted(found)
        missing_grid = [b for b in grid if b not in found]
        files_summary[v] = dict(budgets=budgets, missing_from_grid=missing_grid)
        if not budgets:
            ck.add("budgets present", "warn", f"no {a.tag}_S*_ files", variant=v)
            continue
        ck.add("budgets present", "warn" if missing_grid else "pass",
               f"{len(budgets)} of the {len(grid)}-budget grid"
               + (f"; missing {missing_grid}" if missing_grid else ""), variant=v)
        off_grid = [b for b in budgets if b not in grid]
        if off_grid:
            ck.add("budgets present", "warn", f"budgets off the paper grid: {off_grid}", variant=v)

        theta_bad, knn_seen, mask_bad = [], set(), []
        for S, fp in found.items():
            lack = [k for k in ("theta", "result", "meta", "trials") if k not in fp]
            if lack:
                ck.add("files per budget", "fail" if set(lack) & {"theta", "result"} else "warn",
                       f"missing {lack}", variant=v, budget=S)
            if "theta" not in fp or "result" not in fp:
                continue

            theta = np.load(fp["theta"])
            if theta.ndim != 1 or theta.shape[0] != d or not np.isfinite(theta).all():
                theta_bad.append(f"S={S}: shape {theta.shape}, "
                                 f"{int((~np.isfinite(theta)).sum())} non-finite")
                continue
            thetas[(v, S)] = theta

            res = pd.read_csv(fp["result"]).iloc[0]
            want = dict(cell=a.tag, budget=S, d=d, n_sel=len(tr), n_val=len(va), n_test=len(te))
            bad = {k: res.get(k) for k, w in want.items() if res.get(k) != w}
            if bad:
                ck.add("result.csv header", "fail", f"disagrees with the cell: {bad} (want {want})",
                       variant=v, budget=S)
            knn = int(res.knn)
            knn_seen.add(knn)

            if "meta" in fp:
                meta = json.loads(fp["meta"].read_text())
                sp = meta.get("split", {})
                split_ok = (meta.get("d") == d and meta.get("n_models") == n_models
                            and sp.get("selection") == tr.tolist()
                            and sp.get("validation") == va.tolist()
                            and sp.get("test") == te.tolist())
                if not split_ok:
                    ck.add("meta.json split", "fail",
                           "split / d / n_models differ from model_split() on this cell",
                           variant=v, budget=S)

            mask = top_mask(theta, S)
            masks[(v, S)] = mask
            tk = op.binary_selection(torch.as_tensor(theta), S).numpy().astype(bool)
            tied = theta == theta[mask].min()
            tie_note = f"{int(tied.sum())} theta entries tie at the cut, {int((tied & mask).sum())} inside"
            if not np.array_equal(tk, mask):
                mask_bad.append(S)
                ck.add("top-S masks agree (torch.topk vs argsort)", "fail",
                       f"{int((tk != mask).sum())} points differ; {tie_note}", variant=v, budget=S)
            elif (tied & ~mask).any():
                ck.add("top-S masks agree (torch.topk vs argsort)", "warn",
                       f"one set, but {tie_note}: the selection rests on the tie rule",
                       variant=v, budget=S)
            row = dict(variant=v, budget=S, knn=knn, r_test=float(res.r_test),
                       r_test_ci_lo=float(res.r_test_ci_lo), r_test_ci_hi=float(res.r_test_ci_hi),
                       r_full_test_stored=float(res.r_full_test))

            if S in a.recompute_budgets:
                same = {p: op.hard_corr(Z[idx], acc[idx], theta, S)
                        for p, idx in (("train", tr), ("val", va), ("test", te))}
                r_np = probit_r_np(Zte, ate, mask)
                st = op.plain_stats(Zte, ate, mask)
                diffs = {f"r_{p}": abs(same[p] - float(res[f"r_{p}"])) for p in same}
                row.update(r_test_same_code=same["test"], r_test_numpy=r_np,
                           absdiff_same_code=diffs["r_test"],
                           absdiff_numpy=abs(r_np - float(res.r_test)))
                ok_same = all(x <= TOL_SAME for x in diffs.values())
                ck.add("test r recomputed (same code)", "pass" if ok_same else "fail",
                       f"stored {res.r_test:+.6f}, recomputed {same['test']:+.6f}; largest abs diff "
                       f"over train/val/test {max(diffs.values()):.1e}", variant=v, budget=S)
                sub = Zte[:, mask].mean(1)
                n_clamp = int(((sub <= 0) | (sub >= 1)).sum())
                ck.add("test r recomputed (numpy float64)",
                       "pass" if row["absdiff_numpy"] <= TOL_NUMPY else "fail",
                       f"{r_np:+.6f}, abs diff {row['absdiff_numpy']:.1e}; {n_clamp} test models "
                       f"with subset accuracy 0 or 1", variant=v, budget=S)
                ok_st = (_close(st["pearson"], res.pearson_test_scipy, TOL_SAME)
                         and _close(st["spearman"], res.spearman_test, TOL_SAME))
                ck.add("scipy Pearson / Spearman recomputed", "pass" if ok_st else "fail",
                       f"{st['pearson']:+.6f} / {st['spearman']:+.6f} vs stored "
                       f"{res.pearson_test_scipy:+.6f} / {res.spearman_test:+.6f}",
                       variant=v, budget=S)

            cm = coherence(mask, knn)
            if cm:
                row.update({f"coh_{k}": cm[k] for k in COH_KEYS},
                           cc_sizes_ge_min=[s for s in cm["cc_sizes"] if s >= MIN_COMPONENT])
                stored = {k: res.get(f"coh_{k}") for k in COH_KEYS if f"coh_{k}" in res}
                bad = {k: (cm[k], s) for k, s in stored.items() if not _close(cm[k], s, TOL_COH)}
                if not stored:
                    ck.add("coherence vs stored", "warn", "result.csv has no coh_* columns",
                           variant=v, budget=S)
                elif bad:
                    ck.add("coherence vs stored", "fail", f"(recomputed, stored): {bad}",
                           variant=v, budget=S)
            rows.append(row)

        n_ok = sum(1 for (vv, _) in thetas if vv == v)
        ck.add("theta has one entry per OOD point", "fail" if theta_bad else "pass",
               f"{n_ok} of {len(budgets)} theta files are finite ({d},) vectors"
               + (f"; bad: {theta_bad}" if theta_bad else ""), variant=v)
        if n_ok and not mask_bad:
            ck.add("top-S masks agree (torch.topk vs argsort)", "pass",
                   f"one set at all {n_ok} budgets", variant=v)
        if len(knn_seen) > 1:
            ck.add("knn", "fail", f"result.csv files disagree on knn: {sorted(knn_seen)}", variant=v)
        coh_other = [r for r in ck.rows if r["variant"] == v and r["check"] == "coherence vs stored"]
        if cell.has_spatial() and n_ok and not coh_other:
            ck.add("coherence vs stored", "pass",
                   f"every stored coh_* column reproduced at all {n_ok} budgets", variant=v)
        r_full_stored = [r["r_full_test_stored"] for r in rows if r["variant"] == v]
        if r_full_stored:
            ok_full = all(_close(x, r_full, TOL_SAME) for x in r_full_stored)
            ck.add("stored r_full_test", "pass" if ok_full else "fail",
                   f"{len(r_full_stored)} result.csv files, range {min(r_full_stored):+.6f} .. "
                   f"{max(r_full_stored):+.6f}; recomputed {r_full:+.6f}", variant=v)

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
        xy = graph.project(cell.lon, cell.lat) if cell.has_spatial() else None
        for S in a.check_budgets:
            if ("classic", S) not in thetas or xy is None:
                ck.add("smoothed-classic", "warn", "needs the classic theta and lon/lat; skipped",
                       budget=S)
                continue
            knn = next(r["knn"] for r in rows if r["variant"] == "classic" and r["budget"] == S)
            name = f"smoothed_classic_knn{knn}"
            mask, cut = smoothed_classic(thetas[("classic", S)], xy, S, knn)
            masks[(name, S)] = mask
            n_out = cut["smooth_n_tied"] - cut["smooth_n_tied_in"]
            if cut["smooth_n_tied"] > 1:
                ck.add("smoothed-classic cut", "warn",
                       f"{cut['smooth_n_tied']} points tie at the S-th smoothed value "
                       f"{cut['smooth_cut']:.3e} ({cut['smooth_n_tied_in']} inside the cut, {n_out} "
                       f"outside): " + ("the mask depends on the cell's column order" if n_out else
                                        "all inside, so the mask does not depend on column order"),
                       variant=name, budget=S)
            else:
                ck.add("smoothed-classic cut", "pass",
                       f"unique S-th smoothed value {cut['smooth_cut']:.3e}; the next is "
                       f"{cut['smooth_rel_gap']:.1e} (relative) above", variant=name, budget=S)
            r = op.hard_corr(Zte, ate, mask.astype(np.float32), S, is_mask=True)
            lo_s, hi_s = op.fisher_ci(r, len(te))
            cm = coherence(mask, knn)
            mc = masks[("classic", S)]
            rows.append(dict(variant=name, budget=S, knn=knn, r_test=r, r_test_ci_lo=lo_s,
                             r_test_ci_hi=hi_s, r_test_numpy=probit_r_np(Zte, ate, mask),
                             jaccard_vs_classic=float((mask & mc).sum() / (mask | mc).sum()),
                             **cut, **{f"coh_{k}": cm[k] for k in COH_KEYS},
                             cc_sizes_ge_min=[s for s in cm["cc_sizes"] if s >= MIN_COMPONENT]))
            print(f"[audit] {name} S={S}: test r {r:+.3f}, purity {cm['purity']:.2f}, "
                  f"{cm['n_cc']} components, {cm['tiles']} tiles", flush=True)

    base = {r["budget"]: r["r_test"] for r in rows if r["variant"] == "classic"}
    for r in rows:
        if r["variant"] != "classic" and r["budget"] in base:
            r["dr_vs_classic"] = r["r_test"] - base[r["budget"]]

    return dict(tag=a.tag, root=str(root), cells_dir=str(a.cells_dir),
                written=time.strftime("%Y-%m-%d %H:%M:%S"), secs=time.time() - t0,
                variants=a.variants, check_budgets=a.check_budgets,
                recompute_budgets=a.recompute_budgets, n_tiles_ood=n_tiles,
                split=dict(selection=len(tr), validation=len(va), test=len(te)),
                y_id_stored=info["y_id_stored"], aggregate=aggregate, files=files_summary,
                rows=rows, checks=ck.rows,
                n_fail=ck.n("fail"), n_warn=ck.n("warn"), n_pass=ck.n("pass"))


# ------------------------------------------------------------------ report
def to_markdown(rep: dict) -> str:
    ag, sp = rep["aggregate"], rep["split"]
    L = [f"### Selection audit: `{rep['tag']}`",
         "",
         f"Run `{rep['root']}`; cell `{rep['cells_dir']}`. {ag['n_models']} models "
         f"(selection {sp['selection']} / validation {sp['validation']} / test {sp['test']}), "
         f"d = {ag['d']:,} OOD points in {rep['n_tiles_ood']} test tiles.",
         "",
         f"Aggregate ID→OOD r over all {ag['d']:,} OOD points: **{ag['r_full_test']:+.3f}** "
         f"[{ag['r_full_test_ci'][0]:+.2f}, {ag['r_full_test_ci'][1]:+.2f}] on the "
         f"{ag['n_test_models']} test models (TestSetFinder.test, stored as `r_full_test`); "
         f"{ag['r_all_models']:+.3f} on all {ag['n_models']} models (plain Pearson of probits, "
         f"`report_paper.py`).",
         "",
         "| variant | S | test r [95% CI] | Δr vs classic | recomputed, abs diff (same code / numpy) "
         "| kNN purity | components | components ≥ 20 pts (sizes) | largest | tiles |",
         "|---|---:|---|---:|---|---:|---:|---|---:|---:|"]
    order = {v: i for i, v in enumerate(rep["variants"])}
    rows = {(r["variant"], r["budget"]): r for r in rep["rows"]}
    variants = sorted({r["variant"] for r in rep["rows"]} | set(rep["files"]),
                      key=lambda v: order.get(v, len(order)))
    for S in rep["check_budgets"]:
        for v in variants:
            r = rows.get((v, S))
            if r is None:
                ran = S in rep["files"].get(v, {}).get("budgets", [])
                L.append(f"| {v} | {S} | {'files unusable, see checks' if ran else 'not run'} "
                         f"| | | | | | | |")
                continue
            dr = f"{r['dr_vs_classic']:+.3f}" if "dr_vs_classic" in r else ""
            if "absdiff_same_code" in r:
                rec = f"{r['absdiff_same_code']:.0e} / {r['absdiff_numpy']:.0e}"
            elif v.startswith("smoothed_classic"):
                rec = f"computed here (numpy {r['r_test_numpy']:+.3f})"
            else:
                rec = ""
            if "coh_purity" in r:
                big = r["cc_sizes_ge_min"]
                sizes = f"{len(big)} ({', '.join(map(str, big))})" if big else "none"
                coh = (f"{r['coh_purity']:.2f} | {r['coh_n_cc']} | {sizes} | "
                       f"{r['coh_largest_cc_frac']:.2f} | {r['coh_tiles']} of {rep['n_tiles_ood']}")
            else:
                coh = " | | | | "
            L.append(f"| {v} | {S} | {r['r_test']:+.3f} [{r['r_test_ci_lo']:+.2f}, "
                     f"{r['r_test_ci_hi']:+.2f}] | {dr} | {rec} | {coh} |")
    L += ["",
          f"Checks: {rep['n_pass']} ✓, {rep['n_warn']} ⚠, {rep['n_fail']} ✗.",
          "",
          "| check | variant | S | result | verdict |",
          "|---|---|---:|---|:---:|"]
    for c in rep["checks"]:
        L.append(f"| {c['check']} | {c['variant'] or ''} | {c['budget'] or ''} | "
                 f"{c['detail'].replace('|', '/')} | {MARK[c['level']]} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results/<date>/oodselect_paper")
    ap.add_argument("--cells-dir", required=True, help="folder holding <tag>.npz and _models.csv")
    ap.add_argument("--tag", required=True, help="cell tag, e.g. FL_band_AE_s0")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    help="variant folders under --root, classic first")
    ap.add_argument("--check-budgets", nargs="+", type=int, default=CHECK_BUDGETS,
                    help="budgets in the table")
    ap.add_argument("--recompute-budgets", nargs="+", type=int, default=RECOMPUTE_BUDGETS,
                    help="budgets whose stored r is recomputed from theta and the cell")
    ap.add_argument("--smoothed-classic", action="store_true",
                    help="add the smoothed-classic ablation rows at --check-budgets")
    ap.add_argument("--out", help="folder for the .json and .md (default: --root)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads (the selections ran with 1)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out or a.root)
    out.mkdir(parents=True, exist_ok=True)
    js, md = out / f"{a.tag}_selection_audit.json", out / f"{a.tag}_selection_audit.md"
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
