#!/usr/bin/env python3
"""
Self-test for the spatially regularized OODSelect implementation.

RUN THIS ON THE LAPTOP BEFORE SUBMITTING ANYTHING. It takes about a minute on
CPU, needs no data and no GPU, and exercises every code path the cluster run
uses. Its job is to catch the failures that are expensive to discover in a batch
job: a wrong gradient (which does not crash, it just optimises the wrong thing),
a backend that disagrees with the reference, a coherence metric with a sign
error, a spatial term that is silently a no-op.

    python aotil/test_aotil.py            # full
    python aotil/test_aotil.py --quick    # skip the slower end-to-end checks

The gradient checks are the load-bearing ones. Every gradient in oodselect.py is
hand-derived, so each is compared against a central finite difference of the
objective it claims to differentiate. If you change the objective, change these
in the same edit -- an untested gradient is a silently wrong experiment.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph          # noqa: E402
import oodselect as oo  # noqa: E402


class Checker:
    def __init__(self):
        self.ok = self.fail = 0
        self.failures: list[str] = []

    def __call__(self, cond: bool, msg: str, detail: str = ""):
        if cond:
            self.ok += 1
        else:
            self.fail += 1
            self.failures.append(f"{msg}{(' -- ' + detail) if detail else ''}")
            print(f"  FAIL  {msg}" + (f"  [{detail}]" if detail else ""))

    def section(self, name: str):
        print(f"\n--- {name} ---")

    def report(self) -> int:
        print(f"\n{'=' * 62}")
        print(f"test_aotil: {self.ok} passed, {self.fail} failed")
        if self.failures:
            print("\nFailures:")
            for f in self.failures:
                print(f"  - {f}")
            print("\nDo NOT submit the cluster run until these pass.")
        else:
            print("All checks passed. Safe to submit.")
        return 1 if self.fail else 0


def _fd_grad(fn, theta, h=1e-6):
    """Central finite difference of a scalar-per-row function at theta (1, d)."""
    theta = np.atleast_2d(theta).astype(float)
    g = np.zeros_like(theta)
    for j in range(theta.shape[1]):
        tp = theta.copy(); tp[0, j] += h
        tm = theta.copy(); tm[0, j] -= h
        g[0, j] = (fn(tp)[0] - fn(tm)[0]) / (2 * h)
    return g


def _toy(seed=0, N=25, d=60, k=6):
    """A small problem with real spatial structure: two clusters on a plane."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    S = rng.random((N, d)) * 0.8 + 0.1
    half = d // 2
    lon = np.r_[rng.normal(-81.0, 0.01, half), rng.normal(-80.0, 0.01, d - half)]
    lat = np.r_[rng.normal(27.0, 0.01, half), rng.normal(28.0, 0.01, d - half)]
    A = graph.build_adjacency(lon, lat, k=k)
    L = graph.laplacian(A)
    return x, S, A, L, lon, lat, k


def test_gradients(chk: Checker):
    chk.section("gradients vs finite differences (the load-bearing checks)")
    x, S, A, L, lon, lat, k = _toy()
    d = S.shape[1]
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 0.8, size=(1, d))
    s_target = 20

    for name, lam, mu, Lp in [
        ("correlation term only",      0.0,  0.0, None),
        ("+ cardinality penalty",      5e-2, 0.0, None),
        ("+ spatial penalty (mu=1)",   5e-2, 1.0, L),
        ("spatial term alone (mu=3)",  0.0,  3.0, L),
    ]:
        f = lambda th: oo.objective(th, x, S, Lp, s_target, lam, mu, k)
        ga = oo.objective_grad(theta, x, S, Lp, s_target, lam, mu, k)
        gn = _fd_grad(f, theta)
        scale = max(np.abs(gn).max(), 1e-12)
        rel = np.abs(ga - gn).max() / scale
        chk(rel < 1e-5, f"analytic gradient matches FD: {name}", f"rel err {rel:.2e}")

    # The spatial term must actually do something. A gradient that is silently
    # zero would leave every mu behaving like mu=0 and the whole sweep flat.
    g0 = oo.objective_grad(theta, x, S, None, s_target, 0.0, 0.0, k)
    g1 = oo.objective_grad(theta, x, S, L, s_target, 0.0, 1.0, k)
    chk(np.abs(g1 - g0).max() > 1e-8, "spatial gradient is not a no-op",
        f"max delta {np.abs(g1 - g0).max():.2e}")

    # P(w) must be the normalised cut count: scattered ~1, compact << 1.
    w_comp = np.zeros(d); w_comp[:s_target] = 1.0
    w_scat = np.zeros(d); w_scat[rng.choice(d, s_target, replace=False)] = 1.0
    P_comp = graph.incoherence(w_comp, L, s_target, k)
    P_scat = graph.incoherence(w_scat, L, s_target, k)
    chk(P_comp < P_scat, "P(compact) < P(scattered)", f"{P_comp:.3f} vs {P_scat:.3f}")
    chk(P_scat <= 1.5, "P of a scattered set is on the ~[0,1] scale",
        f"got {P_scat:.3f}")


def test_backends(chk: Checker):
    chk.section("numpy / torch backend agreement")
    if not oo._HAS_TORCH:
        print("  torch not installed -- skipping (the cluster run needs it for --device cuda)")
        return
    x, S, A, L, lon, lat, k = _toy(seed=2, N=30, d=80)
    kw = dict(iters=120, restarts=2, seed=7, knn=k)
    for mus, Lp, label in [(0.0, None, "mu=0"), ([0.0, 1.0], L, "mu sweep")]:
        mn, _ = oo.solve(x, S, 25, L=Lp, mus=mus, device="cpu", **kw)
        mt, _ = oo.solve(x, S, 25, L=Lp, mus=mus, device="torch-cpu",
                         dtype="float64", **kw)
        agree = float((mn == mt).mean())
        chk(agree > 0.98, f"backends select the same points ({label})",
            f"{agree:.1%} agreement")


def test_mu_behaviour(chk: Checker):
    chk.section("does mu do what it is supposed to do")
    rng = np.random.default_rng(3)
    N, d, k = 40, 400, 8
    # Plant a genuinely spatial signal: one cluster where better models do worse.
    half = d // 2
    lon = np.r_[rng.normal(-81.0, 0.02, half), rng.normal(-80.0, 0.02, d - half)]
    lat = np.r_[rng.normal(27.0, 0.02, half), rng.normal(28.0, 0.02, d - half)]
    A = graph.build_adjacency(lon, lat, k=k); L = graph.laplacian(A)
    x = rng.normal(size=N)
    S = rng.random((N, d)) * 0.3 + 0.5
    region = np.zeros(d, bool); region[:120] = True          # inside cluster 1
    S[:, region] -= 0.25 * (x[:, None] - x.mean())           # inverted there

    mus = np.array([0.0, 1.0, 10.0])
    masks, info = oo.solve(x, S, 100, L=L, mus=mus, knn=k, iters=400,
                           restarts=3, seed=0)
    mets = [graph.coherence_metrics(m, A, k) for m in masks]
    rs = [oo.evaluate(x, S, m)["r"] for m in masks]

    chk(all(m.sum() == 100 for m in masks), "cardinality respected at every mu",
        f"sizes {[int(m.sum()) for m in masks]}")
    chk(mets[-1]["purity"] > mets[0]["purity"],
        "purity rises with mu", f"{mets[0]['purity']:.3f} -> {mets[-1]['purity']:.3f}")
    chk(mets[-1]["P_hard"] < mets[0]["P_hard"],
        "incoherence falls with mu", f"{mets[0]['P_hard']:.3f} -> {mets[-1]['P_hard']:.3f}")
    chk(mets[-1]["n_cc"] <= mets[0]["n_cc"],
        "fewer components at high mu", f"{mets[0]['n_cc']} -> {mets[-1]['n_cc']}")
    chk(rs[0] < 0, "mu=0 finds the inversion", f"r={rs[0]:.3f}")
    chk(all(np.isfinite(r) for r in rs), "every mu returns a finite correlation")

    # With the inversion planted inside one cluster, constraining should find it.
    overlap = (masks[-1] & region).sum() / masks[-1].sum()
    chk(overlap > 0.5, "constrained selection lands in the planted region",
        f"{overlap:.1%} inside")


def test_controls(chk: Checker):
    chk.section("baselines and the selection null")
    x, S, A, L, lon, lat, k = _toy(seed=4, N=30, d=150)
    rng = np.random.default_rng(0)
    d = S.shape[1]

    rnd = oo.baseline_random(d, 40, rng)
    hard = oo.baseline_hard(S[:15], 40)
    chk(rnd.sum() == 40 and hard.sum() == 40, "baselines respect the budget")
    chk(not np.array_equal(rnd, hard), "baselines differ from each other")

    xs = oo.shuffled_x(x, np.random.default_rng(1))
    chk(np.allclose(np.sort(xs), np.sort(x)), "shuffled x is a permutation")
    chk(not np.allclose(xs, x), "shuffled x actually moved")

    # The hard baseline must be built from selection models only -- a version
    # using all models would be scored on models it had already seen.
    h_sel = oo.baseline_hard(S[:15], 40)
    h_all = oo.baseline_hard(S, 40)
    chk(not np.array_equal(h_sel, h_all),
        "hard baseline depends on which models built it")


def test_metrics(chk: Checker):
    chk.section("coherence metrics")
    rc = graph.self_test()
    chk(rc == 0, "graph.py self-test passes")

    x, S, A, L, lon, lat, k = _toy(seed=5, d=100)
    d = S.shape[1]
    rng = np.random.default_rng(0)
    f = 0.25
    m_rand = oo.baseline_random(d, int(d * f), rng)
    mets = graph.coherence_metrics(m_rand, A, k)
    # Under a random mask these have known expectations; a metric that misses
    # them by a wide margin has a sign or normalisation error.
    chk(abs(mets["purity"] - f) < 0.20, "random purity near f",
        f"{mets['purity']:.3f} vs f={f}")
    chk(abs(mets["P_hard"] - (1 - f)) < 0.35, "random P_hard near 1-f",
        f"{mets['P_hard']:.3f} vs {1-f}")
    chk(abs(mets["moran_I"]) < 0.25, "random mask Moran's I near zero",
        f"{mets['moran_I']:.3f}")

    j = graph.jaccard(m_rand, m_rand)
    chk(j > 1.0, "self-Jaccard above the chance ratio", f"{j:.2f}")


def test_pipeline(chk: Checker, quick: bool):
    chk.section("end-to-end on synthetic cells")
    if quick:
        print("  skipped (--quick)")
        return
    import pointcov, transfer                              # noqa: E402
    rng = np.random.default_rng(11)
    N, d, k = 60, 500, 8
    half = d // 2
    lon = np.r_[rng.normal(-81.0, 0.02, half), rng.normal(-80.0, 0.02, d - half)]
    lat = np.r_[rng.normal(27.0, 0.02, half), rng.normal(28.0, 0.02, d - half)]
    tiles = np.where(np.arange(d) < half, "t1", "t2")
    x = rng.normal(size=N)
    S = rng.random((N, d)) * 0.3 + 0.5
    S[:, :150] -= 0.25 * (x[:, None] - x.mean())

    A = graph.build_adjacency(lon, lat, k=k)
    cov = pointcov.point_covariance(x, S)
    chk(len(cov) == d, "one covariance per OOD point")
    chk(cov[:150].mean() < cov[150:].mean(),
        "covariance map finds the planted region",
        f"{cov[:150].mean():.3f} vs {cov[150:].mean():.3f}")
    nul = graph.morans_i_null(cov, A, n_perm=100, seed=0)
    chk(np.isfinite(nul["z"]), "Moran's I null returns a z")

    L = graph.laplacian(A)
    masks, _ = oo.solve(x, S, 100, L=L, mus=[0.0, 3.0], knn=k, iters=300,
                        restarts=2, seed=0)
    sel, rep = np.arange(0, N, 2), np.arange(1, N, 2)
    labels = (rng.random(d) < 0.5).astype(int)
    res = transfer.holdout_transfer(x[sel], S[sel], x[rep], S[rep],
                                    lon, lat, tiles, labels,
                                    frac=0.2, mu=3.0, knn=k, seed=0,
                                    iters=150, restarts=2)
    chk(np.isfinite(res["r_h2"]), "held-out-point transfer returns a correlation",
        f"r_h2={res['r_h2']:.3f}")
    chk(res["n_h2_selected"] > 0, "transfer selects points in the held-out half")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the slower end-to-end checks")
    a = ap.parse_args()

    t0 = time.time()
    chk = Checker()
    print("=" * 62)
    print("spatially regularized OODSelect -- self-test")
    print(f"numpy {np.__version__} | torch "
          f"{'present' if oo._HAS_TORCH else 'ABSENT'}")
    print("=" * 62)

    test_gradients(chk)
    test_backends(chk)
    test_mu_behaviour(chk)
    test_controls(chk)
    test_metrics(chk)
    test_pipeline(chk, a.quick)

    print(f"\nelapsed {time.time() - t0:.1f}s")
    return chk.report()


if __name__ == "__main__":
    raise SystemExit(main())
