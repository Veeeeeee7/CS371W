#!/usr/bin/env python3
"""Checks for search.py (the subset search). Run from the repo root: .venv/bin/python aotil/test_search.py

1. correctness: a p = 0.5 tie is correct for a negative, wrong for a positive.
2. The two spatial losses on a small kNN graph: z'Lz and its gradient 2Lz against dense torch; the total
   variation against sum |z_j - z_j'| over the undirected edges; on a 0/1 z both count the cut edges.
3. pearson_r_loss against numpy's Pearson r of the clipped probits (population moments, eps-regularized).
4. The squared budget term (S - total)^2.
5. train_one is deterministic: the same seed gives the same theta, another seed another theta; mu = 0 ignores L.
6. run_unit on a synthetic cell (3 trials, 300 steps): writes theta.npy, trials.csv and result.json; a second
   run gives the same theta bit for bit; the result's r_test is the top-S subset's; OODSelect refuses mu > 0 and the
   losses refuse mu = 0; unit names.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import search  # noqa: E402
import synthetic  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def check_correctness():
    print("1. correctness")
    S = np.array([[0.5, 0.5, 0.7, 0.3]])
    y = np.array([0, 1, 1, 0])
    Z, ties = search.correctness(S, y)
    check(Z.tolist() == [[1, 0, 1, 0]] and ties == 0 and Z.dtype == np.float32,
          "tie correct for a negative, wrong for a positive; true-class score 0.3 wrong; float32")


def small_graph(n=60, seed=1):
    rng = np.random.default_rng(seed)
    lon, lat = -82 + rng.uniform(0, 0.2, n), 28 + rng.uniform(0, 0.2, n)
    A = graph.build_adjacency(lon, lat, k=5)
    return A, graph.laplacian(A)


def check_losses():
    print("2. spatial losses")
    A, L = small_graph()
    n = A.shape[0]
    rng = np.random.default_rng(2)
    z = torch.tensor(rng.uniform(0, 1, n), dtype=torch.float32, requires_grad=True)
    q = search.spatial_term(L, "laplacian")(z)
    q.backward()
    Ld = torch.tensor(L.toarray(), dtype=torch.float32)
    ref = z.detach() @ Ld @ z.detach()
    check(abs(float(q.detach()) - float(ref)) < 1e-3 and torch.allclose(z.grad, 2 * Ld @ z.detach(), atol=1e-4),
          f"Laplacian: z'Lz = dense ({float(q):.4f}), gradient 2Lz")
    tv = search.spatial_term(L, "gtv")
    upper = sp.triu(A, k=1).tocoo()
    zn = z.detach().numpy().astype(np.float64)
    ref_tv = np.abs(zn[upper.row] - zn[upper.col]).sum()
    check(abs(float(tv(z.detach())) - ref_tv) < 1e-3 and tv.n_edges == upper.nnz,
          f"GTV: sum over the {upper.nnz} undirected edges of |z_j - z_j'|")
    b = torch.tensor((rng.random(n) < 0.3).astype(np.float32))
    cut = float(((b.numpy()[upper.row] != b.numpy()[upper.col])).sum())
    check(float(tv(b)) == cut and abs(float(search.spatial_term(L, "laplacian")(b)) - cut) < 1e-6,
          f"on a 0/1 z both count the {cut:.0f} cut edges")


def check_pearson():
    print("3. pearson_r_loss")
    rng = np.random.default_rng(3)
    X = (rng.random((40, 200)) < 0.6).astype(np.float32)
    y = rng.uniform(0.5, 0.9, 40).astype(np.float32)
    m = np.zeros(200, np.float32)
    m[:50] = 1
    r = float(search.pearson_r_loss(torch.tensor(X), torch.tensor(y), torch.tensor(m)))
    sub = X @ m / (m.sum() + 1e-6)
    a = ndtri(np.clip(sub, 1e-6, 1 - 1e-6))
    b = ndtri(np.clip(y, 1e-6, 1 - 1e-6))
    ac, bc = a - a.mean(), b - b.mean()
    ref = np.mean(ac * bc) / np.sqrt((np.mean(ac ** 2) + 1e-6) * (np.mean(bc ** 2) + 1e-6))
    check(abs(r - ref) < 1e-5, f"probit Pearson r {r:.6f} = numpy {ref:.6f}")


def check_budget():
    print("4. budget term")
    check(float(search.budget_term(torch.tensor(480.0), 500)) == 400.0
          and float(search.budget_term(torch.tensor(520.0), 500)) == 400.0, "(S - total)^2 on both sides")


def check_train_one():
    print("5. train_one")
    rng = np.random.default_rng(4)
    X = torch.tensor((rng.random((30, 80)) < 0.6).astype(np.float32))
    y = torch.tensor(rng.uniform(0.5, 0.9, 30).astype(np.float32))
    hp = dict(lr=0.5, eta_min=0.01, T_max=50, penalty=1e-3, penalty_anneal_iters=20)
    kw = dict(max_epochs=150, patience=40)
    a = search.train_one(X[:20], y[:20], X[20:], y[20:], 10, hp, seed=7, **kw)
    b = search.train_one(X[:20], y[:20], X[20:], y[20:], 10, hp, seed=7, **kw)
    c = search.train_one(X[:20], y[:20], X[20:], y[20:], 10, hp, seed=8, **kw)
    check(np.array_equal(a["theta"], b["theta"]) and not np.array_equal(a["theta"], c["theta"]),
          "same seed, same theta; another seed, another theta")
    _, L = small_graph(80)
    d = search.train_one(X[:20], y[:20], X[20:], y[20:], 10, hp, seed=7, L_t=search.spatial_term(L), mu=0.0, **kw)
    check(np.array_equal(a["theta"], d["theta"]) and np.isfinite(d["P_soft"]), "mu = 0 ignores the graph (P_soft logged)")


def check_run_unit():
    print("6. run_unit on a synthetic cell")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cell = synthetic.cell(tmp / "zoo")
        kw = dict(trials=3, max_steps=300, patience=50)
        out1, out2 = tmp / "a", tmp / "b"
        r1 = search.run_unit(cell, "FL_band_AE_s0", out1, 40, "gtv", 3.0, **kw)
        search.run_unit(cell, "FL_band_AE_s0", out2, 40, "gtv", 3.0, **kw)
        files = all((out1 / f).exists() for f in ("theta.npy", "trials.csv", "result.json"))
        check(files and json.loads((out1 / "result.json").read_text())["unit"] == "gtv_mu3_S40",
              "writes theta.npy, trials.csv and result.json")
        t1, t2 = np.load(out1 / "theta.npy"), np.load(out2 / "theta.npy")
        check(np.array_equal(t1, t2), "a second run gives the same theta bit for bit")
        _, Z, acc, _ = search.load_paper_cell(cell, "FL_band_AE_s0")
        _, _, test = search.model_split(len(acc))
        check(abs(search.hard_corr(Z[test], acc[test], t1, 40) - r1["r_test"]) < 1e-9 and r1["n_trials"] == 3,
              f"r_test is the top-40 subset's on the test models ({r1['r_test']:+.3f})")
        for method, mu in (("oodselect", 1.0), ("laplacian", 0.0)):
            try:
                search.run_unit(cell, "FL_band_AE_s0", tmp / "x", 40, method, mu, **kw)
                ok = False
            except ValueError:
                ok = True
            check(ok, f"{method} with mu = {mu:g} is refused")
    check(search.unit_name("oodselect", 0, 500) == "oodselect_S500"
          and search.unit_name("gtv", 0.01, 500) == "gtv_mu0.01_S500"
          and search.unit_name("laplacian", 3, 10000) == "laplacian_mu3_S10000", "unit names")


def main() -> int:
    torch.set_num_threads(1)
    check_correctness()
    check_losses()
    check_pearson()
    check_budget()
    check_train_one()
    check_run_unit()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
