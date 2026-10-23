#!/usr/bin/env python3
"""Checks for o5_contrast.py (O5 Part B). Run from the repo root:

    .venv/bin/python aotil/test_o5_contrast.py [--out-dir results/2026-10-17/o5]

1-6 (synthetic, no data needed): the summed sufficient statistics give the same SMD, class shares,
    TVD, terrain-adjusted coefficient (np.linalg.lstsq) and model behaviour as direct computation;
    bootstrap, pooling and label weights equal physically repeating pairs; midrank_pct,
    two_sided_p, zero spread and boot_counts.
7-8 (real inputs, skipped when missing): the S1 pairs pass observed_pairs' checks; the top 15 test
    models are the 15 highest auc_id among the paper's 60 test models.
9-11 (when <out-dir>/B_contrast.csv exists): the primary rows, region bootstraps, null values and
    null percentiles, an adjusted row, a TVD row, the balance, the flags and the model behaviour,
    recomputed independently from S1_controls, S2_null.parquet, the point table and the cells.
Exit code 0 when every check passes.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import match_controls as mc  # noqa: E402
import o5_contrast as contrast  # noqa: E402
import oodselect_paper as op  # noqa: E402

warnings.filterwarnings("ignore")

CELL_PREFIX = "FL_band_AE"
FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def close(a, b, tol=1e-9):
    """Same shape and equal within tol, NaN matching NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


def output(ok, text):
    """Suffix naming the module's output value when a check fails."""
    return "" if ok else f"; output has {text}"


# --------------------------------------------------------------------------------- synthetic
def fake(n=400, seed=0):
    """A random SeedData with 3 numeric and 2 categorical variables (some values missing), and its
    Layout with every level reported."""
    rng = np.random.default_rng(seed)
    V = 3
    num = rng.normal(size=(n, V)) * [1.0, 5.0, 0.2] + [0.0, 10.0, -3.0]
    num[rng.random((n, V)) < 0.08] = np.nan
    levels = {"a": ["x", "y", "z"], "b": ["p", "q", "r", "s"]}
    cat = np.c_[rng.integers(0, 3, n), rng.integers(0, 4, n)]
    cat[rng.random((n, 2)) < 0.05] = -1
    z = rng.normal(size=(n, 6))
    k = 15
    err = rng.integers(0, k + 1, n).astype(float)
    label = rng.integers(0, 2, n)
    model = np.c_[rng.random(n), err, np.where(label == 1, err, 0.0)]
    D = contrast.SeedData(0, np.array([f"P{i}" for i in range(n)]), label, np.array(["t"] * n),
                          np.ones(n, bool), np.zeros(n, bool), -np.ones(n, int), num, cat, z, z.copy(),
                          model, dict(k=k))
    lay = contrast.Layout([f"v{i}" for i in range(V)], np.zeros(V), np.ones(V), ["a", "b"], levels)
    for j, v in enumerate(["a", "b"]):
        for i, c in enumerate(levels[v]):
            lay.classes.append((v, c, j, i))
    return D, lay


def direct(D, lay, ir, ic):
    """The statistics computed directly on the pairs (ir[p], ic[p])."""
    out = dict(smd=[], adj=[], share=[], tvd=[])
    for v in range(lay.V):
        r, c = D.num[ir, v], D.num[ic, v]
        ok = np.isfinite(r) & np.isfinite(c)
        n_ok = ok.sum()
        out["smd"].append(mc.smd(r[ok], c[ok]))
        X = np.r_[np.c_[np.ones(n_ok), np.ones(n_ok), D.z[ir][ok]],
                  np.c_[np.ones(n_ok), np.zeros(n_ok), D.z[ic][ok]]]
        y = np.r_[r[ok], c[ok]]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        den = np.sqrt((r[ok].var(ddof=1) + c[ok].var(ddof=1)) / 2)
        out["adj"].append(beta[1] / den)
    for j, v in enumerate(lay.cat_vars):
        r, c = D.cat[ir, j], D.cat[ic, j]
        ok = (r >= 0) & (c >= 0)
        sr = np.bincount(r[ok], minlength=len(lay.cat_levels[v])) / ok.sum()
        sc = np.bincount(c[ok], minlength=len(lay.cat_levels[v])) / ok.sum()
        out["share"] += list(sr - sc)
        out["tvd"].append(0.5 * np.abs(sr - sc).sum())
    k = D.scores["k"]
    pr, pc = D.model[ir, 0].mean(), D.model[ic, 0].mean()
    ar, ac = 1 - D.model[ir, 1].mean() / k, 1 - D.model[ic, 1].mean() / k
    fr = D.model[ir, 2].sum() / D.model[ir, 1].sum()
    fc = D.model[ic, 2].sum() / D.model[ic, 1].sum()
    out["model"] = [pr - pc, ar - ac, fr - fc]
    return {name: np.array(values, float) for name, values in out.items()}


def first_row(U):
    """{key: U[key][0]}: the one row of an aggregate over a single weight row."""
    return {k: v[0] for k, v in U.items()}


def test_finalize():
    print("1. finalize(aggregate(pair_stats)) vs direct computation")
    D, lay = fake()
    ir, ic = np.arange(0, 200), np.arange(200, 400)
    st = contrast.pair_stats(D, ir, ic, lay)
    F = contrast.finalize(first_row(contrast.aggregate(st, np.ones((1, len(ir))))), lay)
    expected = direct(D, lay, ir, ic)
    check(close(F["smd"], expected["smd"], 1e-10), f"SMD per variable {np.round(F['smd'], 4)}")
    check(close(F["share"], expected["share"], 1e-12), "class share differences")
    check(close(F["tvd"], expected["tvd"], 1e-12), f"total variation distance {np.round(F['tvd'], 4)}")
    check(close(F["adj"], expected["adj"], 1e-8),
          f"terrain-adjusted SMD = lstsq / den {np.round(F['adj'], 4)}")
    check(close(F["model"], expected["model"], 1e-12), f"model behaviour {np.round(F['model'], 4)}")
    complete = [np.sum(np.isfinite(D.num[ir, v]) & np.isfinite(D.num[ic, v])) for v in range(3)]
    check(close(F["n"], complete), "n = complete pairs per variable")
    return D, lay


def test_weights(D, lay):
    print("2. weighted sums = the statistic of the resampled union")
    rng = np.random.default_rng(1)
    ir, ic = np.arange(0, 120), np.arange(200, 320)
    st = contrast.pair_stats(D, ir, ic, lay)
    W = contrast.boot_counts(len(ir), 5, rng)
    Fb = contrast.finalize(contrast.aggregate(st, W), lay)
    ok = True
    for b in range(5):
        rep = np.repeat(np.arange(len(ir)), W[b].astype(int))
        expected = direct(D, lay, ir[rep], ic[rep])
        ok &= close(Fb["smd"][b], expected["smd"], 1e-9) and close(Fb["adj"][b], expected["adj"], 1e-7)
        ok &= close(Fb["share"][b], expected["share"], 1e-12) and close(Fb["tvd"][b], expected["tvd"], 1e-12)
        ok &= close(Fb["model"][b], expected["model"], 1e-12)
    check(ok, "5 bootstrap draws: SMD, adjusted, shares, TVD, model equal repeating the pairs")
    # three units pooled by wsum with multiplicities (2, 0, 1)
    units = [np.arange(0, 40), np.arange(40, 90), np.arange(90, 120)]
    U = [contrast.aggregate(contrast.pair_stats(D, ir[u], ic[u], lay), np.ones((1, len(u)))) for u in units]
    stacked = {k: np.concatenate([u[k] for u in U]) for k in U[0]}
    Fp = contrast.finalize(first_row(contrast.wsum(stacked, np.array([[2.0, 0.0, 1.0]]))), lay)
    rep = np.r_[units[0], units[0], units[2]]
    expected = direct(D, lay, ir[rep], ic[rep])
    check(close(Fp["smd"], expected["smd"], 1e-9) and close(Fp["adj"], expected["adj"], 1e-7)
          and close(Fp["tvd"], expected["tvd"], 1e-12),
          "units pooled by wsum (2, 0, 1) = the union with repeats")


def test_label_weights(D, lay):
    print("3. label weights")
    ir, ic = np.arange(0, 150), np.arange(200, 350)
    st = contrast.pair_stats(D, ir, ic, lay)
    lab = D.label[ir]
    M = sp.csr_matrix((np.ones(len(ir)), (lab, np.arange(len(ir)))), shape=(2, len(ir)))
    U = contrast.aggregate(st, M)
    one_draw = {k: v[None] for k, v in U.items()}
    n_null = np.bincount(lab, minlength=2)[None].astype(float)
    w1 = contrast.label_weights(n_null[0], n_null, [0, 1])
    F1 = contrast.finalize(contrast.weighted_l(one_draw, w1), lay)
    F0 = contrast.finalize(contrast.usum(U, 0), lay)
    check(close(F1["smd"][0], F0["smd"], 1e-10) and close(F1["adj"][0], F0["adj"], 1e-8),
          "weights from equal label counts = the unweighted pool")
    w2 = contrast.label_weights(n_null[0] * [1, 3], n_null, [0, 1])   # positives count three times
    F2 = contrast.finalize(contrast.weighted_l(one_draw, w2), lay)
    rep = np.r_[np.arange(len(ir)), np.repeat(np.flatnonzero(lab == 1), 2)]
    expected = direct(D, lay, ir[rep], ic[rep])
    check(close(F2["smd"][0], expected["smd"], 1e-9) and close(F2["tvd"][0], expected["tvd"], 1e-12),
          "weight 3 on positives = repeating every positive pair three times")


def test_midrank():
    print("4. midrank_pct and two_sided_p")
    null = np.arange(10.0)
    check(contrast.midrank_pct(20, null)[0] == 100 and contrast.midrank_pct(-1, null)[0] == 0,
          "above all 100, below all 0")
    check(contrast.midrank_pct(4.0, null)[0] == 45.0, "tie at 4 of 0..9: (4 + 0.5) / 10 = 45")
    check(contrast.midrank_pct(4.0 + 1e-13, null)[0] == 45.0, "within 1e-12 counts as a tie")
    check(contrast.midrank_pct(4.5, np.r_[null, np.nan])[0:2] == (50.0, 10), "NaNs skipped, n = 10")
    check(np.isnan(contrast.midrank_pct(np.nan, null)[0]), "NaN observed -> NaN")
    check(contrast.two_sided_p(99.0) == 0.02 and contrast.two_sided_p(50) == 1.0,
          "two-sided p = 2 min(pct, 100 - pct)/100")


def test_zero_spread():
    print("5. zero spread")
    D, lay = fake(40)
    D.num[:, 0] = 7.0
    D.num[:20, 1], D.num[20:, 1] = 1.0, 2.0
    st = contrast.pair_stats(D, np.arange(20), np.arange(20, 40), lay)
    F = contrast.finalize(first_row(contrast.aggregate(st, np.ones((1, 20)))), lay)
    check(F["smd"][0] == 0.0 and np.isnan(F["smd"][1]), f"equal constants -> 0, unequal -> NaN: {F['smd'][:2]}")


def test_boot_counts():
    print("6. boot_counts")
    W1 = contrast.boot_counts(7, 50, np.random.default_rng([1, 2]))
    W2 = contrast.boot_counts(7, 50, np.random.default_rng([1, 2]))
    check((W1.sum(1) == 7).all() and (W1 == W2).all() and W1.shape == (50, 7),
          "rows sum to 7, same seed same draws")


# -------------------------------------------------------------------------------- real data
def real_inputs(args):
    need = [args.controls, args.null, args.regions_land, args.data, args.extras, args.cells_dir]
    return all(Path(p).exists() for p in need)


def test_real(args):
    print("7-8. real inputs")
    table = contrast.load_table(args.data, args.extras)
    lay = contrast.make_layout(table)
    ctxs = mc.load_contexts([0, 1, 2], args.cells_dir, CELL_PREFIX, args.data, args.sel_root,
                            args.regions_dir, 500, 20, verbose=False)
    data = {}
    for s, ctx in ctxs.items():
        scores = contrast.model_scores(args.cells_dir, ctx.tag, 15)
        data[s] = contrast.seed_data(ctx, table, lay, scores)
        models = pd.read_csv(Path(args.cells_dir) / f"{ctx.tag}_models.csv")
        _, _, test_models = op.model_split(len(models))
        ranked = models.iloc[test_models].sort_values("auc_id", ascending=False, kind="stable")
        check(set(scores["top"]) <= set(test_models) and set(scores["top"]) == set(ranked.idx.iloc[:15])
              and ranked.auc_id.iloc[14] > ranked.auc_id.iloc[15],
              f"seed {s}: top 15 are inside the 60 test models and are the 15 highest auc_id "
              f"(15th {ranked.auc_id.iloc[14]:.4f} > 16th {ranked.auc_id.iloc[15]:.4f})")
        npz = np.load(Path(args.cells_dir) / f"{ctx.tag}.npz", allow_pickle=True)
        y = npz["y_id"]
        diffs = []
        for m in scores["top"][:4]:
            S = npz["S_id"][m]
            p = np.round(np.where(y == 1, S, 1 - S), 9)
            diffs.append(abs(roc_auc_score(y, p) - models.auc_id.iloc[m]))
        # zoo.py ranks the unrounded p; rounding p to 9 decimals merges near-ties, so the AUCs agree
        # to ~1e-5, not to the last digit
        check(max(diffs) < 1e-4, f"seed {s}: auc_id recomputed from S_id for 4 top models "
                                 f"(max diff {max(diffs):.1e} < 1e-4)")
    K = pd.read_csv(args.controls, dtype={"point_id": str, "control_point_id": str})
    try:
        OP = contrast.observed_pairs(K, data)
        check(len(OP) == int((K.status == "matched").sum()), f"S1 pairs pass every check ({len(OP)} pairs)")
    except SystemExit as e:
        check(False, f"observed_pairs raised: {e}")


# ------------------------------------------------------------------------------------ outputs
def entering_regions(rl, pairs, region_type):
    """(seed, component) of the regions of this type with >= 1 of the given pairs, in rl order."""
    regs = []
    for s, c, t in rl[["seed", "component", "type"]].itertuples(index=False):
        if t == region_type and ((pairs.seed == s) & (pairs.component == c)).any():
            regs.append((s, c))
    return regs


def river(table, ids):
    """log10_dist_river of the given point ids."""
    return table.log10_dist_river.loc[ids].to_numpy()


def primary_met(row):
    """The primary rule: interval beyond 0 and null percentile beyond 95 (or 5), predicted way."""
    if contrast.PREDICTED[row.type] > 0:
        return row.boot_lo > 0 and row.null_pct > 95
    return row.boot_hi < 0 and row.null_pct < 5


def check_primary(prim, K, rl, table, S2, N):
    """Each type's primary SMD, region bootstrap, 200 null values, null percentile and verdict."""
    for ti, region_type in enumerate(contrast.TYPES):
        row = prim[prim.type == region_type].iloc[0]
        pairs = K[K.type == region_type]
        r, c = river(table, pairs.point_id), river(table, pairs.control_point_id)
        check(close(row.smd_or_share_diff, mc.smd(r, c), 1e-9) and close(row.mean_region, r.mean(), 1e-9)
              and close(row.mean_control, c.mean(), 1e-9) and row.n_pairs == len(pairs),
              f"{region_type}: primary SMD {row.smd_or_share_diff:+.4f} recomputed from S1 pairs + table "
              f"({len(pairs)} pairs)")
        # region bootstrap with the module's generator and region order
        regs = entering_regions(rl, K, region_type)
        rng = np.random.default_rng([contrast.RNG_SEED, 0, ti, 0])
        idx = rng.integers(0, len(regs), size=(contrast.N_BOOT, len(regs)))
        per = {k: K[(K.seed == k[0]) & (K.component == k[1])] for k in regs}
        vals = []
        for b in range(contrast.N_BOOT):
            resampled = pd.concat([per[regs[i]] for i in idx[b]])
            vals.append(mc.smd(river(table, resampled.point_id), river(table, resampled.control_point_id)))
        lo, hi = np.percentile(np.array(vals)[np.isfinite(vals)], [2.5, 97.5])
        check(close([row.boot_lo, row.boot_hi], [lo, hi], 1e-9),
              f"{region_type}: region bootstrap [{lo:+.4f}, {hi:+.4f}] recomputed by concatenating "
              f"resampled regions")
        # null values, every draw
        nv = N[(N.type == region_type) & (N.label_scope == "all") & (N.variable == contrast.PRIMARY)
               & ~N.adjusted]
        nv = nv.sort_values("draw").null_value.to_numpy()
        patches = S2.merge(pd.DataFrame(regs, columns=["seed", "component"]), on=["seed", "component"])
        ref = []
        for draw in range(200):
            in_draw = patches[patches.draw == draw]
            lp = np.concatenate([np.asarray(x, object) for x in in_draw.land_point_ids])
            cp = np.concatenate([np.asarray(x, object) for x in in_draw.control_point_ids])
            ok = np.array([isinstance(x, str) for x in cp])
            ref.append(mc.smd(river(table, lp[ok].astype(str)), river(table, cp[ok].astype(str))))
        check(len(nv) == 200 and close(nv, ref, 1e-9), f"{region_type}: 200 null values recomputed from S2 patches")
        ref = np.asarray(ref)
        obs = row.smd_or_share_diff
        pct = 100 * ((ref < obs - 1e-12).sum() + 0.5 * (np.abs(ref - obs) <= 1e-12).sum()) / len(ref)
        check(close(row.null_pct, pct, 1e-9), f"{region_type}: null percentile {pct:.2f} recomputed")
        check(bool(row.met) == bool(primary_met(row)), f"{region_type}: met = {bool(row.met)} follows the rule")


def check_adjusted(args, C, K, table):
    """The positive-dominated adjusted primary against lstsq on the stacked pairs."""
    region_type = contrast.TYPES[0]
    row = C[(C.scope == "pooled") & (C.type == region_type) & (C.label_scope == "all")
            & (C.variable == contrast.PRIMARY) & C.adjusted].iloc[0]
    pairs = K[K.type == region_type]
    zcols = [f"z_{v}" for v in contrast.TERRAIN]
    parts, z_region, z_twin = [], [], []
    for s, seed_pairs in pairs.groupby("seed"):
        ctx = mc.load_seed_context(s, args.cells_dir, CELL_PREFIX, args.data, args.sel_root,
                                   args.regions_dir)
        points = ctx.pts.set_index("point_id")
        parts.append(seed_pairs)
        z_region.append(points.loc[seed_pairs.point_id, zcols].to_numpy())
        z_twin.append(points.loc[seed_pairs.control_point_id, zcols].to_numpy())
    pairs_by_seed = pd.concat(parts)
    zr, zc = np.concatenate(z_region), np.concatenate(z_twin)
    r, c = river(table, pairs_by_seed.point_id), river(table, pairs_by_seed.control_point_id)
    n = len(pairs_by_seed)
    X = np.r_[np.c_[np.ones(n), np.ones(n), zr], np.c_[np.ones(n), np.zeros(n), zc]]
    beta = np.linalg.lstsq(X, np.r_[r, c], rcond=None)[0]
    den = np.sqrt((r.var(ddof=1) + c.var(ddof=1)) / 2)
    check(close(row.smd_or_share_diff, beta[1] / den, 1e-8) and close(row.diff_raw, beta[1], 1e-9),
          f"{region_type}: adjusted primary {row.smd_or_share_diff:+.4f} = lstsq on the stacked pairs")


def check_landcover_tvd(C, K, table):
    """The positive-dominated landcover TVD (label scope all) from the raw codes."""
    region_type = contrast.TYPES[0]
    row = C[(C.scope == "pooled") & (C.type == region_type) & (C.label_scope == "all")
            & (C.variable == "landcover") & (C.stat == "tvd")].iloc[0]
    pairs = K[K.type == region_type]
    lr = table.landcover.loc[pairs.point_id].astype(int).value_counts(normalize=True)
    lc = table.landcover.loc[pairs.control_point_id].astype(int).value_counts(normalize=True)
    codes = lr.index.union(lc.index)
    tvd = 0.5 * (lr.reindex(codes, fill_value=0) - lc.reindex(codes, fill_value=0)).abs().sum()
    check(close(row.smd_or_share_diff, tvd, 1e-12), f"{region_type}: landcover TVD {tvd:.4f} recomputed")


def check_balance(args, BAL):
    """B_balance per-region rows equal S1_balance.csv's 'after' columns (same pairs)."""
    S1 = pd.read_csv(Path(args.controls).parent / "S1_balance.csv")
    S1 = S1[S1.scope == "region"][["seed", "component", "variable", "diff_after_seed_sd", "smd_after"]]
    per_region = BAL[BAL.scope == "region"]
    m = per_region.merge(S1, on=["seed", "component", "variable"])
    check(len(m) == len(per_region) and close(m.diff_seed_sd, m.diff_after_seed_sd, 1e-9)
          and close(m.smd, m.smd_after, 1e-9), f"B_balance per-region rows equal S1_balance 'after' ({len(m)} rows)")


def check_flags_and_nulls(C, N):
    """Label-scope rules, secondary flags, and B_null's draws and finite counts."""
    po = C[C.variable.isin(contrast.POSITIVE_ONLY)]
    check((po.label_scope == "positives").all(), "positive-only variables only on label scope positives")
    check((C[C.variable == "fn_share_top"].label_scope == "all").all(), "fn_share_top only on label scope all")
    sec = (C.test_family.fillna("") != "") & ~C.primary & ((C.null_pct > 97.5) | (C.null_pct < 2.5))
    check((sec == C.secondary).all(), f"secondary flags follow the rule ({int(C.secondary.sum())} rows)")
    keys = ["type", "label_scope", "variable", "class", "adjusted"]
    cnt = N.groupby(keys).size()
    pooled = C[C.scope == "pooled"]
    check((cnt == 200).all() and len(cnt) == len(pooled),
          f"B_null: 200 draws for each of the {len(cnt)} pooled statistics")
    with_finite = N.assign(n_finite=np.isfinite(N.null_value.to_numpy(float)).astype(int))
    fin = with_finite.groupby(keys).n_finite.sum().reset_index()
    mm = pooled.merge(fin, on=keys, how="left", validate="one_to_one")
    check(len(mm) == len(pooled) and mm.n_finite.notna().all() and (mm.null_n == mm.n_finite).all(),
          f"pooled null_n = number of finite B_null values of that statistic, every row "
          f"({int((mm.null_n == 200).sum())} of {len(mm)} rows have all 200 finite)")


def test_outputs(args):
    print("9. outputs in", args.out_dir)
    C = pd.read_csv(Path(args.out_dir) / "B_contrast.csv", low_memory=False)
    N = pd.read_csv(Path(args.out_dir) / "B_null.csv", low_memory=False)
    BAL = pd.read_csv(Path(args.out_dir) / "B_balance.csv")
    C["class"] = C["class"].fillna("")
    N["class"] = N["class"].fillna("")
    prim = C[C.primary]
    check(len(prim) == 2 and set(prim.type) == set(contrast.TYPES) and (prim.variable == contrast.PRIMARY).all()
          and (prim.label_scope == "all").all() and (prim.scope == "pooled").all() and (~prim.adjusted).all(),
          "two primary rows: pooled, label scope all, log10_dist_river, unadjusted, one per type")

    # independent re-derivation from the raw tables
    rl = pd.read_csv(args.regions_land)
    columns = ["point_id", "log10_dist_river", "landcover", "label"]
    table = pd.read_parquet(args.data, columns=columns).set_index("point_id")
    K = pd.read_csv(args.controls, dtype={"point_id": str, "control_point_id": str})
    K = K[K.status == "matched"].drop(columns="type").merge(rl[["seed", "component", "type"]],
                                                              on=["seed", "component"])
    S2 = pd.read_parquet(args.null)
    check_primary(prim, K, rl, table, S2, N)
    check_adjusted(args, C, K, table)
    check_landcover_tvd(C, K, table)
    check_balance(args, BAL)
    check_flags_and_nulls(C, N)
    test_label_scopes(C, N, K, rl, table, S2)
    test_model_behaviour(args, C, K, table)


def s2_pairs(S2: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """One row per S2 patch pair (land point with a twin): seed, component, draw, the land point's
    and twin's table label and log10_dist_river."""
    n = S2.land_point_ids.map(len).to_numpy()
    if not (S2.control_point_ids.map(len).to_numpy() == n).all():
        raise AssertionError("S2 land_point_ids and control_point_ids differ in length")
    lp = np.concatenate([np.asarray(x, object) for x in S2.land_point_ids])
    cp = np.concatenate([np.asarray(x, object) for x in S2.control_point_ids])
    ok = np.array([isinstance(x, str) for x in cp])
    lp, cp = lp[ok].astype(str), cp[ok].astype(str)
    out = pd.DataFrame({k: np.repeat(S2[k].to_numpy(), n)[ok] for k in ("seed", "component", "draw")})
    out["label"] = table.label.loc[lp].to_numpy().astype(int)
    out["label_c"] = table.label.loc[cp].to_numpy().astype(int)
    out["r"] = table.log10_dist_river.loc[lp].to_numpy()
    out["c"] = table.log10_dist_river.loc[cp].to_numpy()
    return out


def test_label_scopes(C, N, K, rl, table, S2):
    """Primary variable in label scopes positives / negatives: entering regions, SMD, region
    bootstrap, null values and null_pct, from the raw tables."""
    print("10. log10_dist_river, label scopes positives and negatives")
    K = K.assign(lab=table.label.loc[K.point_id].to_numpy().astype(int))
    check((K.lab.to_numpy() == table.label.loc[K.control_point_id].to_numpy()).all(),
          "S1 twins have the point's label in the table")
    P2 = s2_pairs(S2, table)
    check((P2.label == P2.label_c).all(), f"S2 twins have the patch point's label ({len(P2)} patch pairs)")
    for ti, region_type in enumerate(contrast.TYPES):
        for li, (lname, labs) in enumerate(contrast.LABEL_SCOPES):
            if lname == "all":
                continue
            (L,) = labs
            row = C[(C.scope == "pooled") & (C.type == region_type) & (C.label_scope == lname)
                    & (C.variable == contrast.PRIMARY) & ~C.adjusted]
            if len(row) != 1:
                check(False, f"{region_type}, {lname}: expected one pooled {contrast.PRIMARY} row, found {len(row)}")
                continue
            row = row.iloc[0]
            pairs = K[(K.type == region_type) & (K.lab == L)]
            regs = entering_regions(rl, pairs, region_type)
            n_typed = int((rl.type == region_type).sum())
            r, c = river(table, pairs.point_id), river(table, pairs.control_point_id)
            ok = (row.n_regions == len(regs) and row.n_pairs == len(pairs)
                  and close(row.smd_or_share_diff, mc.smd(r, c), 1e-9))
            check(ok, f"{region_type}, {lname}: {len(regs)} of {n_typed} regions have >= 1 matched pair of "
                      f"label {L}; SMD {mc.smd(r, c):+.4f} on {len(pairs)} pairs recomputed"
                      + output(ok, f"n_regions {row.n_regions}, SMD {row.smd_or_share_diff:+.4f}, "
                                   f"n_pairs {row.n_pairs}"))
            # region bootstrap: the module's generator and region order, regions concatenated
            per = []
            for s, comp in regs:
                region_pairs = pairs[(pairs.seed == s) & (pairs.component == comp)]
                per.append((river(table, region_pairs.point_id),
                            river(table, region_pairs.control_point_id)))
            rng = np.random.default_rng([contrast.RNG_SEED, 0, ti, li])
            idx = rng.integers(0, len(regs), size=(contrast.N_BOOT, len(regs)))
            vals = []
            for resample in idx:
                region_values = np.concatenate([per[i][0] for i in resample])
                twin_values = np.concatenate([per[i][1] for i in resample])
                vals.append(mc.smd(region_values, twin_values))
            vals = np.array(vals)
            lo, hi = np.percentile(vals[np.isfinite(vals)], [2.5, 97.5])
            ok = close([row.boot_lo, row.boot_hi], [lo, hi], 1e-9)
            check(ok, f"{region_type}, {lname}: region bootstrap [{lo:+.4f}, {hi:+.4f}] recomputed"
                      + output(ok, f"[{row.boot_lo:+.4f}, {row.boot_hi:+.4f}]"))
            # null: draw d pools the d-th patch of every entering region, its label-L pairs only
            in_regions = pd.MultiIndex.from_frame(P2[["seed", "component"]]).isin(regs)
            label_pairs = P2[in_regions & (P2.label == L)]
            ref = np.array([mc.smd(x.r.to_numpy(), x.c.to_numpy()) for _, x in label_pairs.groupby("draw")])
            drawn = np.sort(label_pairs.draw.unique())
            nv = N[(N.type == region_type) & (N.label_scope == lname) & (N.variable == contrast.PRIMARY)
                   & ~N.adjusted]
            nv = nv.sort_values("draw")
            ok = (len(nv) == 200 and len(drawn) == 200 and (drawn == np.arange(200)).all()
                  and (nv.draw.to_numpy() == drawn).all() and close(nv.null_value.to_numpy(), ref, 1e-9))
            check(ok, f"{region_type}, {lname}: 200 null values recomputed from the S2 patches' label-{L} pairs "
                      f"(median {np.nanmedian(ref):+.4f})"
                      + output(ok, f"{len(nv)} values, median {np.nanmedian(nv.null_value):+.4f}"))
            fin = ref[np.isfinite(ref)]
            obs = row.smd_or_share_diff
            pct = 100 * ((fin < obs - 1e-12).sum() + 0.5 * (np.abs(fin - obs) <= 1e-12).sum()) / len(fin)
            ok = close(row.null_pct, pct, 1e-9) and row.null_n == len(fin)
            check(ok, f"{region_type}, {lname}: null percentile {pct:.2f} recomputed (n {len(fin)})"
                      + output(ok, f"{row.null_pct:.2f} (n {row.null_n})"))


def top_model_cells(args, seeds) -> dict:
    """Per seed, from the cell (without op.correctness): point positions, labels, and per point
    the top 15 test models' mean p(flood), mean accuracy and error count."""
    cells = {}
    for s in seeds:
        npz = np.load(Path(args.cells_dir) / f"{CELL_PREFIX}_s{s}.npz", allow_pickle=True)
        models = pd.read_csv(Path(args.cells_dir) / f"{CELL_PREFIX}_s{s}_models.csv")
        _, _, test_models = op.model_split(len(models))
        ranked = models.iloc[test_models].sort_values("auc_id", ascending=False, kind="stable")
        top = ranked.idx.iloc[:15].to_numpy()
        y = np.asarray(npz["y_ood"]).astype(int)
        S = np.asarray(npz["S_ood"][top], float)       # probability of the true label
        right = np.where(y[None] == 1, S > 0.5, S >= 0.5)
        positions = {p: i for i, p in enumerate(npz["point_id"].astype(str))}
        cells[s] = dict(pos=positions, y=y, p=np.round(np.where(y[None] == 1, S, 1 - S), 9).mean(0),
                        acc=right.mean(0), err=(~right).sum(0).astype(float))
    return cells


def gather(cells, pairs, column, key):
    """cells[seed][key] at the points pairs[column], seed by seed."""
    parts = []
    for s, rows in pairs.groupby("seed"):
        cell = cells[s]
        ix = np.array([cell["pos"][p] for p in rows[column]], int)
        parts.append(cell[key][ix])
    return np.concatenate(parts)


def test_model_behaviour(args, C, K, table):
    """p_flood_top, acc_top and fn_share_top, pooled by type and label scope, recomputed from the
    cells' S_ood."""
    print("11. model behaviour from the cells")
    cells = top_model_cells(args, sorted(K.seed.unique()))
    K = K.assign(lab=table.label.loc[K.point_id].to_numpy().astype(int))
    for region_type in contrast.TYPES:
        for lname, labs in contrast.LABEL_SCOPES:
            pairs = K[(K.type == region_type) & K.lab.isin(labs)]
            rows = C[(C.scope == "pooled") & (C.type == region_type) & (C.label_scope == lname)
                     & (C.group == "model_behaviour")].set_index("variable")
            for var, key in (("p_flood_top", "p"), ("acc_top", "acc")):
                mr = gather(cells, pairs, "point_id", key).mean()
                mcn = gather(cells, pairs, "control_point_id", key).mean()
                ok = var in rows.index and close([rows.loc[var, "mean_region"], rows.loc[var, "mean_control"],
                                                  rows.loc[var, "smd_or_share_diff"], rows.loc[var, "n_pairs"]],
                                                 [mr, mcn, mr - mcn, len(pairs)], 1e-9)
                suffix = ""
                if var in rows.index:
                    suffix = output(ok, f"{rows.loc[var, 'mean_region']:.4f} vs {rows.loc[var, 'mean_control']:.4f}")
                check(ok, f"{region_type}, {lname}: {var} {mr:.4f} vs {mcn:.4f} recomputed from the cells "
                          f"({len(pairs)} pairs)" + suffix)
            if lname != "all":
                continue
            # false-negative share: errors on positives / all errors, pooled over the pairs' points
            err_r = gather(cells, pairs, "point_id", "err")
            y_r = gather(cells, pairs, "point_id", "y")
            err_c = gather(cells, pairs, "control_point_id", "err")
            y_c = gather(cells, pairs, "control_point_id", "y")
            fr = err_r[y_r == 1].sum() / err_r.sum()
            fc = err_c[y_c == 1].sum() / err_c.sum()
            var = "fn_share_top"
            ok = var in rows.index and close([rows.loc[var, "mean_region"], rows.loc[var, "mean_control"],
                                              rows.loc[var, "smd_or_share_diff"]], [fr, fc, fr - fc], 1e-9)
            suffix = ""
            if var in rows.index:
                suffix = output(ok, f"{rows.loc[var, 'mean_region']:.4f} vs {rows.loc[var, 'mean_control']:.4f}")
            check(ok, f"{region_type}, {lname}: fn_share_top {fr:.4f} vs {fc:.4f} (errors on positives / all "
                      f"errors) recomputed from the cells" + suffix)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/2026-10-17/o5")
    ap.add_argument("--controls", default="results/2026-10-17/o5/S1_controls.csv")
    ap.add_argument("--null", default="results/2026-10-17/o5/S2_null.parquet")
    ap.add_argument("--regions-land", default="results/2026-10-17/o5/regions_land.csv")
    ap.add_argument("--data", default="data/processed/dataset_FL.parquet")
    ap.add_argument("--extras", default="data/interim/analysis/point_extras_FL.parquet")
    ap.add_argument("--cells-dir", default="data/interim/aotil/cells_w7")
    ap.add_argument("--sel-root", default="results/2026-10-16/oodselect_paper")
    ap.add_argument("--regions-dir", default="results/2026-10-16/regions")
    args = ap.parse_args()
    D, lay = test_finalize()
    test_weights(D, lay)
    test_label_weights(D, lay)
    test_midrank()
    test_zero_spread()
    test_boot_counts()
    if real_inputs(args):
        test_real(args)
    else:
        print("7-8. skipped (real inputs missing)")
    if (Path(args.out_dir) / "B_contrast.csv").exists():
        test_outputs(args)
    else:
        print("9. skipped (no B_contrast.csv in", args.out_dir, ")")
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
