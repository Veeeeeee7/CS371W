#!/usr/bin/env python3
"""
Checks for o5_feature_inclusion.py (Part D1). Run from the repo root:

    .venv/bin/python aotil/test_o5_feature_inclusion.py [--cells data/interim/aotil/cells_w7
        --sel-root results/2026-10-16/oodselect_paper --regions results/2026-10-16/regions
        --dataset data/processed/dataset_FL.parquet
        --regions-land results/2026-10-17/o5/regions_land.csv]

1. indicators() reads the `features` strings and refuses a name that is not a model feature.
2. probit() equals oodselect_paper.probit_t, clamp included (0 and 1 map to -/+4.753).
3. ols() matches lstsq, sklearn, the textbook slope SE and r2_score; a rank-deficient X raises.
4. vif() is 1 / (1 - corr^2) with two predictors.
5. design(): one column per non-reference family, probit_acc_id only in the id formula, and an
   absent reference family is refused.
6. A synthetic population drawn as zoo.py draws it, with one planted spurious feature: the fit
   and its bootstrap find it; at a fixed n_feat two indicators correlate at -1/6.
7. boot_draws() leaves a rank-deficient resample NaN.
8. Real cells (skipped when missing): the masks are the stored selection, the region land counts
   equal regions_land.csv, and the response's r equals oodselect_paper.hard_corr.
9. main() end to end on the real cells with 20 bootstrap draws into a scratch folder.
Exit code 0 when every check passes.
"""
import argparse
import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o5_feature_inclusion as fi     # noqa: E402
import oodselect_paper as op          # noqa: E402

warnings.filterwarnings("ignore")     # all-NaN percentiles and constant columns are expected here

FEATS = ["landcover", "slope", "tri_300", "wetindexcat", "log10_dist_river",
         "log10_dist_coast", "hand"]
FAMS = ["dt", "et", "hgb", "lr", "mlp", "nb", "rf", "xgb"]

FAILS = []


def check(cond, msg):
    """Print one check; a failure is remembered for the exit code."""
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def test_indicators():
    print("1. indicators()")
    I = fi.indicators(["landcover;hand", "slope;tri_300;wetindexcat", ";".join(FEATS)], FEATS)
    check(I.tolist() == [[1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0], [1] * 7],
          f"three lists read into the right columns: {I.tolist()}")
    try:
        fi.indicators(["landcover;dist_river"], FEATS)
        check(False, "an unknown feature name is refused")
    except SystemExit:
        check(True, "an unknown feature name is refused")


def test_probit():
    print("2. probit() vs oodselect_paper.probit_t")
    import torch          # slow to import, and only this check needs it
    x = np.array([0.0, 1e-9, 1e-6, 0.01, 0.3, 0.5, 0.77, 1 - 1e-6, 1.0])
    ref = op.probit_t(torch.as_tensor(x, dtype=torch.float64)).numpy()
    err = float(np.max(np.abs(fi.probit(x) - ref)))
    check(err < 1e-9, f"max |diff| = {err:.1e} on {len(x)} values incl. 0 and 1")
    low, high = float(fi.probit(0.0)), float(fi.probit(1.0))
    check(abs(low + 4.7534) < 1e-3 and abs(high - 4.7534) < 1e-3,
          f"clamp: probit(0) = {low:.4f}, probit(1) = {high:.4f}")


def test_ols():
    print("3. ols()")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(3)
    n = 80
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n), rng.normal(size=n)])
    Y = np.column_stack([X @ [0.5, 1.0, -0.3, 0.0] + rng.normal(0, 0.4, n),
                         X @ [-1, 0.2, 0.1, 0.7] + rng.normal(0, 0.1, n)])
    B, SE, R2, df = fi.ols(X, Y)
    ls = np.linalg.lstsq(X, Y, rcond=None)[0]
    sk = LinearRegression(fit_intercept=False).fit(X, Y).coef_.T
    check(np.allclose(B, ls, atol=1e-10) and np.allclose(B, sk, atol=1e-10),
          f"coef = lstsq = sklearn (max |diff| {np.abs(B - sk).max():.1e})")
    check(df == n - 4, f"residual df = {df}")

    # one predictor: the slope's SE is s / sqrt(sum (x - xbar)^2)
    x = X[:, 1]
    X1 = X[:, :2]
    b1, se1, r21, _ = fi.ols(X1, Y[:, 0])
    resid = Y[:, 0] - X1 @ b1[:, 0]
    s = np.sqrt(resid @ resid / (n - 2))
    se_text = s / np.sqrt(((x - x.mean()) ** 2).sum())
    check(abs(se1[1, 0] - se_text) < 1e-12, f"slope SE = textbook {se_text:.6f}")
    check(abs(r21[0] - r2_score(Y[:, 0], X1 @ b1[:, 0])) < 1e-12, "R2 = sklearn r2_score")
    try:
        fi.ols(np.column_stack([X, X[:, 1] * 2]), Y)
        check(False, "rank-deficient design raises")
    except np.linalg.LinAlgError:
        check(True, "rank-deficient design raises")


def test_vif():
    print("4. vif()")
    rng = np.random.default_rng(4)
    a = rng.normal(size=500)
    b = 0.6 * a + 0.8 * rng.normal(size=500)
    v = fi.vif(np.column_stack([np.ones(500), a, b]))
    want = 1 / (1 - np.corrcoef(a, b)[0, 1] ** 2)
    check(np.isnan(v[0]) and abs(v[1] - want) < 1e-9 and abs(v[2] - want) < 1e-9,
          f"VIF = 1/(1-corr^2) = {want:.4f}: {v[1]:.4f}, {v[2]:.4f}")


def test_design():
    print("5. design()")
    rng = np.random.default_rng(5)
    kinds = np.array([FAMS[i % 8] for i in range(40)])
    I = rng.integers(0, 2, (40, 7))
    acc = rng.uniform(0.6, 0.9, 40)
    X, terms = fi.design(acc, kinds, I, FEATS, "dt", with_id=True)
    fam_cols = [j for j, t in enumerate(terms) if t.startswith("family_")]
    check("family_dt" not in terms and len(fam_cols) == 7, f"7 family columns, no dt: {terms[:10]}")
    fam_sum = X[:, fam_cols].sum(1)
    check(((fam_sum == 0) == (kinds == "dt")).all() and set(fam_sum.tolist()) <= {0.0, 1.0},
          "family columns sum to 0 for dt models, 1 otherwise")
    check(terms[1] == "probit_acc_id" and np.allclose(X[:, 1], fi.probit(acc)),
          "probit_acc_id column in the id formula")
    X2, t2 = fi.design(acc, kinds, I, FEATS, "dt", with_id=False)
    check("probit_acc_id" not in t2 and X2.shape[1] == X.shape[1] - 1, "absent without it")
    check(t2[-7:] == [f"saw_{f}" for f in FEATS] and (X2[:, -7:] == I).all(),
          "saw_* columns are the indicators, in feature order")
    try:
        fi.design(acc, np.where(kinds == "dt", "et", kinds), I, FEATS, "dt", with_id=True)
        check(False, "absent reference family refused")
    except SystemExit:
        check(True, "absent reference family refused")


def zoo_like(rng, N):
    """Feature subsets drawn as zoo.py draws them (feat_frac 0.3 / 0.6 / 1.0, then a uniform
    subset of max(3, int(7 * frac)) features), families round-robin."""
    kinds = np.array([FAMS[i % 8] for i in range(N)])
    I = np.zeros((N, 7), int)
    nf = np.zeros(N, int)
    for i in range(N):
        frac = rng.choice([0.3, 0.6, 1.0])
        cols = rng.choice(7, max(3, int(7 * frac)), replace=False)
        I[i, cols] = 1
        nf[i] = len(cols)
    return kinds, I, nf


def test_synthetic():
    print("6. synthetic recovery of a planted spurious feature")
    rng = np.random.default_rng(6)
    N = 300
    kinds, I, nf = zoo_like(rng, N)
    fam_eff = {k: rng.normal(0, 0.1) for k in FAMS}
    true = np.array([0.0, 0.0, 0.0, 0.0, -0.30, 0.0, 0.0])        # log10_dist_river spurious
    river = 4
    z_id = 0.8 + 0.05 * I.sum(1) + 0.08 * I[:, river] + rng.normal(0, 0.1, N)
    acc_id = fi.norm.cdf(z_id)
    y = 0.2 - 0.6 * z_id + np.array([fam_eff[k] for k in kinds]) + I @ true + rng.normal(0, 0.1, N)
    X, terms = fi.design(acc_id, kinds, I, FEATS, "dt", with_id=True)
    B, SE, _, _ = fi.ols(X, y)
    saw = [terms.index(f"saw_{f}") for f in FEATS]
    z = (B[saw, 0] - true) / SE[saw, 0]
    max_z = np.abs(z).max()
    check(max_z < 3, f"every feature within 3 SE of the truth: max |z| = {max_z:.2f}")
    check(int(np.argmin(B[saw, 0])) == river,
          f"most negative = log10_dist_river ({B[saw[river], 0]:+.3f})")

    idx = np.random.default_rng(0).integers(0, N, (400, N))
    D = fi.boot_draws(X, y, idx)[:, :, 0]
    lo = np.nanpercentile(D, 2.5, 0)
    hi = np.nanpercentile(D, 97.5, 0)
    planted = saw[river]
    check(lo[planted] < -0.30 < hi[planted] and hi[planted] < 0,
          f"planted interval [{lo[planted]:+.3f}, {hi[planted]:+.3f}] covers -0.30, excludes 0")
    null_terms = [saw[k] for k in range(7) if k != river]
    cover0 = sum(lo[j] < 0 < hi[j] for j in null_terms)
    check(cover0 >= 5, f"null features: {cover0}/6 intervals cover 0")

    _, I2, nf2 = zoo_like(np.random.default_rng(7), 20000)
    c = np.corrcoef(I2[nf2 == 4].T)[np.triu_indices(7, 1)]
    check(abs(c.mean() + 1 / 6) < 0.01, f"at n_feat = 4 the pairwise corr is -1/6: {c.mean():+.4f}")


def test_boot_rank():
    print("7. boot_draws() drops rank-deficient resamples")
    rng = np.random.default_rng(8)
    kinds, I, _ = zoo_like(rng, 64)
    X, _ = fi.design(rng.uniform(0.6, 0.9, 64), kinds, I, FEATS, "dt", with_id=True)
    y = rng.normal(size=64)
    good = rng.integers(0, 64, 64)
    no_xgb = rng.choice(np.where(kinds != "xgb")[0], 64)
    D = fi.boot_draws(X, y, np.vstack([good, no_xgb]))
    check(np.isfinite(D[0]).all() and np.isnan(D[1]).all(),
          "a draw with every family is finite, a draw without xgb is NaN")


def have_real_data(a) -> bool:
    return (a.cells / "FL_band_AE_s0.npz").exists() and a.dataset.exists()


def region_number(response: str) -> int:
    """k of a 'region_<k>' response."""
    return int(response.split("_")[1])


def test_real(a):
    print("8. real cells: masks, land counts, response vs hard_corr")
    if not have_real_data(a):
        print("  skip  cells or dataset missing")
        return
    lc = fi.load_landcover(a.dataset)
    ns = argparse.Namespace(cells_dir=a.cells, sel_root=a.sel_root, regions_dir=a.regions,
                            cell_prefix="FL_band_AE", variant="spatial_mu3_knn10", budget=500,
                            min_size=20)
    rl = pd.read_csv(a.regions_land) if a.regions_land.exists() else None
    for s in (0, 1, 2):
        si = fi.seed_inputs(ns, s, lc, FEATS)
        check(True, f"seed {s}: seed_inputs accepted the regions file as the top-500 theta")
        if rl is not None:
            got = {}
            for c in si["counts"]:
                if c["response"].startswith("region_"):
                    got[region_number(c["response"])] = c["n_land"]
            want = rl[rl.seed == s].set_index("component").n_land.to_dict()
            check(got == want, f"seed {s}: region land counts = regions_land.csv {want}")
            with_land = {k for k, v in want.items() if v > 0}
            responses = {region_number(r) for r in si["masks"] if r.startswith("region_")}
            check(with_land == responses,
                  f"seed {s}: one region response per region with land points")

        # hard_corr adds 1e-6 to the variances and the mask sum and runs in float32, whose erfinv
        # moves ~0.01 per ulp at the clamp; the two paths differ by up to 1e-4 (seed 0, with 2
        # test models at the clamp), hence 1e-3, as in aotil/audit_selection.py.
        _, _, te = op.model_split(si["Z"].shape[0])
        m = si["masks"]["spatial_S500_all"]
        acc_r = si["Z"][te][:, m].mean(1)
        r_np = np.corrcoef(fi.probit(acc_r), fi.probit(si["acc_id"][te]))[0, 1]
        r_hc = op.hard_corr(si["Z"][te], si["acc_id"][te], m.astype(np.float32), 500, is_mask=True)
        n_cl = int(((acc_r == 0) | (acc_r == 1)).sum())
        check(abs(r_np - r_hc) < 1e-3, f"seed {s}: r from the response {r_np:+.5f} = hard_corr "
                                       f"{r_hc:+.5f} ({n_cl} test models at the clamp)")
        check((si["I"].sum(1) == si["n_feat"]).all(), f"seed {s}: features lists sum to n_feat")


def test_main(a):
    print("9. main() end to end, 20 draws")
    if not have_real_data(a):
        print("  skip  cells or dataset missing")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "d1.csv"
        saved_argv = sys.argv
        sys.argv = ["x", "--cells-dir", str(a.cells), "--sel-root", str(a.sel_root),
                    "--regions-dir", str(a.regions), "--dataset", str(a.dataset),
                    "--n-boot", "20", "--out", str(out)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = fi.main()
        finally:
            sys.argv = saved_argv
        check(rc == 0 and out.exists(), "main() returned 0 and wrote the CSV")

        df = pd.read_csv(out, dtype={"seed": str})
        need = ["seed", "response", "model_set", "term", "coef", "se", "boot_lo", "boot_hi",
                "n_models"]
        check(all(c in df.columns for c in need), f"brief's columns present: {need}")
        g = df[df.term.str.startswith("saw_")].groupby(["seed", "response", "model_set", "formula"])
        check((g.size() == 7).all(), f"{g.ngroups} fits, each with the 7 saw_* terms")
        check({"test60", "all300", "test60_noclamp", "all300_noclamp"} <= set(df.model_set),
              "the four model sets")
        k = ((df.response == "pooled_regions") & (df.model_set == "test60")
             & (df.formula == "id+family+features") & (df.term == "saw_hand"))
        per = df[k & df.seed.isin(["0", "1", "2"])].coef.mean()
        m3 = df[k & (df.seed == "mean3")].coef.iloc[0]
        check(abs(per - m3) < 1e-12, f"mean3 coef = mean of the seeds ({m3:+.4f})")
        stem = out.with_suffix("")
        check(Path(f"{stem}_seen.csv").exists() and Path(f"{stem}_corr.csv").exists(),
              "_seen and _corr written")
        t60 = df[(df.model_set == "test60") & (df.seed != "mean3")]
        check((t60.n_models == 60).all(), "per-seed test60 fits have 60 models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=Path("data/interim/aotil/cells_w7"))
    ap.add_argument("--sel-root", type=Path, default=Path("results/2026-10-16/oodselect_paper"))
    ap.add_argument("--regions", type=Path, default=Path("results/2026-10-16/regions"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_FL.parquet"))
    ap.add_argument("--regions-land", type=Path,
                    default=Path("results/2026-10-17/o5/regions_land.csv"))
    a = ap.parse_args()
    test_indicators()
    test_probit()
    test_ols()
    test_vif()
    test_design()
    test_synthetic()
    test_boot_rank()
    test_real(a)
    test_main(a)
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
