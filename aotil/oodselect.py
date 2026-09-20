#!/usr/bin/env python3
"""
OODSelect (Salaudeen et al., arXiv:2510.24884) with a spatial smoothness term.

THE OBJECTIVE
-------------
Given N models with probit in-distribution performance x, and a per-example
score matrix S (N x d) on the OOD set, find a subset minimising the correlation
between x and the models' mean score on that subset:

    min_w  corr(x, a(w))  +  lambda (S_target - sum w)^2  +  mu * P(w)

    a(w) = S w / sum(w)                        the per-model subset mean
    P(w) = w' L w / (S_target * k)             normalised cut-edge count

with w = sigmoid(theta) as the relaxation of the binary selection vector, Adam,
cosine annealing on the learning rate, a 10x ramp on lambda, and random
restarts. mu = 0 is the paper's objective exactly. L is the Laplacian of a
spatial adjacency over the OOD points (see graph.py for why kNN and why P is
normalised).

TWO DEVIATIONS FROM THE PAPER, BOTH FORCED BY THIS DATA
-------------------------------------------------------
1. CONTINUOUS SCORES. The paper uses binary correctness Z_ij. AUC has no
   per-example decomposition, and thresholding at 0.5 would mean different
   things across cells whose achieved positive:negative ratios differ (0.18 to
   1.04 across this dataset). S_ij is the predicted probability of the OBSERVED
   class. The objective is unchanged; only the entries are continuous.
2. THE SPATIAL TERM. The paper's subsets are scattered by construction, which is
   fine for its benchmarks and fatal for a GIS explanation. mu trades inversion
   strength against spatial coherence.

GRADIENTS ARE HAND-WRITTEN
--------------------------
    dcorr/da  = xc / (|xc| |ac|) - r * ac / |ac|^2
    da_i/dw_j = (S_ij - a_i) / sum(w)
    =>  dcorr/dw = (g S - g.a) / sum(w)                one mat-vec
    dP/dw     = 2 L w / (S_target * k)                 one sparse mat-vec
    dw/dtheta = w (1 - w)

test_aotil.py checks all of these against finite differences. If you change the
objective, change that test in the same edit.

BATCHING IS THE POINT OF THE GPU
--------------------------------
Restarts and mu values are independent problems over the same S, so they run as
rows of a single (R, d) weight matrix: one batched mat-vec per iteration instead
of R*M sequential ones. The model-fitting in zoo.py is CPU-bound scikit-learn
and is the real wall-clock cost of a sweep; this part is what a GPU actually
accelerates, and only because it is batched.

BACKENDS
--------
numpy is the reference implementation and needs no new dependency. torch is
optional and used when available (`--device cuda`); the two agree to ~1e-10 in
float64, which test_aotil.py asserts. Pick float32 on the GPU only for the
sweep, never for a number that goes in the paper.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, pearsonr, spearmanr

EPS = 1e-12

try:
    import warnings
    import torch
    # The CSR path is flagged beta and re-warns on every tensor construction;
    # the sweep builds one per cell and the noise buries real output.
    warnings.filterwarnings("ignore", message=".*[Ss]parse.*",
                            category=UserWarning)
    _HAS_TORCH = True
except Exception:                                        # torch is optional
    torch = None
    _HAS_TORCH = False


# ------------------------------------------------------------------ helpers
def probit(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Probit-transform a performance measure, as the paper does.

    Linearises the ID-OOD relationship so a Pearson correlation is the right
    summary; without it the trend bends near the ceiling.
    """
    return norm.ppf(np.clip(np.asarray(a, float), eps, 1 - eps))


def evaluate(x: np.ndarray, S: np.ndarray, mask: np.ndarray) -> dict:
    """Correlation between ID performance and subset mean, on the given models.

    Spearman is reported alongside Pearson because a single outlying model can
    carry a Pearson correlation; the paper makes the same check.
    """
    mask = np.asarray(mask, bool)
    if mask.sum() < 2 or len(x) < 3:
        return dict(r=np.nan, p=np.nan, rho=np.nan, n_models=len(x), k=int(mask.sum()))
    a = S[:, mask].mean(1)
    if np.std(a) < EPS or np.std(x) < EPS:
        return dict(r=np.nan, p=np.nan, rho=np.nan, n_models=len(x), k=int(mask.sum()))
    r, p = pearsonr(x, a)
    rho, _ = spearmanr(x, a)
    return dict(r=float(r), p=float(p), rho=float(rho),
                n_models=int(len(x)), k=int(mask.sum()))


def _corr_and_grad(x: np.ndarray, a: np.ndarray):
    """Row-wise Pearson r between fixed x (N,) and each row of a (R, N).

    Returns r (R,) and dr/da (R, N). Centring is idempotent, so the derivative
    of the centring step drops out.
    """
    xc = x - x.mean()
    nx = np.sqrt((xc * xc).sum())
    ac = a - a.mean(axis=1, keepdims=True)
    na = np.sqrt((ac * ac).sum(axis=1))
    denom = np.maximum(nx * na, EPS)
    r = (ac @ xc) / denom
    g = xc[None, :] / denom[:, None] - r[:, None] * ac / np.maximum(na ** 2, EPS)[:, None]
    return r, g


# ------------------------------------------------------------------ core
def objective(theta, x, S, L, s_target, lam, mus, knn):
    """The full objective, per row of theta. Returns (R,).

    Written out separately from the optimiser so the hand-derived gradient below
    can be checked against finite differences of THIS function.
    """
    theta = np.atleast_2d(theta)
    mus = np.atleast_1d(np.asarray(mus, float))
    w = 1.0 / (1.0 + np.exp(-theta))
    W = np.maximum(w.sum(axis=1), EPS)
    a = (w @ S.T) / W[:, None]
    xc = x - x.mean()
    ac = a - a.mean(axis=1, keepdims=True)
    denom = np.maximum(np.sqrt((xc * xc).sum()) * np.sqrt((ac * ac).sum(axis=1)), EPS)
    J = (ac @ xc) / denom + lam * (s_target - W) ** 2
    if L is not None and np.any(mus > 0):
        P = np.einsum("rd,rd->r", w, (L @ w.T).T) / max(s_target * knn, 1.0)
        J = J + mus * P
    return J


def objective_grad(theta, x, S, L, s_target, lam, mus, knn):
    """d(objective)/d(theta), per row. The three terms are, in order:

        dcorr/dw  = (g S - g.a) / sum(w)      with g = dcorr/da
        dpenalty  = -2 lam (S_target - sum w)
        dP/dw     = 2 L w / (S_target k)

    then through the sigmoid, dw/dtheta = w (1 - w).
    """
    theta = np.atleast_2d(theta)
    mus = np.atleast_1d(np.asarray(mus, float))
    w = 1.0 / (1.0 + np.exp(-theta))
    W = np.maximum(w.sum(axis=1), EPS)
    a = (w @ S.T) / W[:, None]
    _, g = _corr_and_grad(x, a)
    gw = (g @ S - (g * a).sum(axis=1)[:, None]) / W[:, None]
    gw = gw - 2.0 * lam * (s_target - W)[:, None]
    if L is not None and np.any(mus > 0):
        gw = gw + mus[:, None] * 2.0 * (L @ w.T).T / max(s_target * knn, 1.0)
    return gw * w * (1 - w)


def _solve_numpy(x, S, L, s_target, mus, knn, iters, lr0, lam0, theta0):
    """Reference implementation. theta0 is (R, d); mus is (R,)."""
    theta = theta0.copy()
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    mus = np.asarray(mus, float)

    for t in range(iters):
        frac = t / max(iters - 1, 1)
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * frac))      # cosine anneal
        lam = lam0 * (1 + 9 * frac)                      # tighten the budget
        gt = objective_grad(theta, x, S, L, s_target, lam, mus, knn)

        m = 0.9 * m + 0.1 * gt
        v = 0.999 * v + 0.001 * gt ** 2
        mh = m / (1 - 0.9 ** (t + 1)); vh = v / (1 - 0.999 ** (t + 1))
        theta = np.clip(theta - lr * mh / (np.sqrt(vh) + 1e-8), -12, 12)
    return 1.0 / (1.0 + np.exp(-theta))


def _solve_torch(x, S, L, s_target, mus, knn, iters, lr0, lam0, theta0,
                 device="cuda", dtype="float64"):
    """Same arithmetic on a torch device. Kept line-for-line parallel to the
    numpy version so the two can be diffed when they disagree."""
    dt = torch.float64 if dtype == "float64" else torch.float32
    dev = torch.device(device)
    tx = torch.as_tensor(x, dtype=dt, device=dev)
    tS = torch.as_tensor(S, dtype=dt, device=dev)
    theta = torch.as_tensor(theta0, dtype=dt, device=dev).clone()
    tmus = torch.as_tensor(np.asarray(mus, float), dtype=dt, device=dev)
    m = torch.zeros_like(theta); v = torch.zeros_like(theta)

    tL = None
    if L is not None and bool((np.asarray(mus, float) > 0).any()):
        Lc = sp.csr_matrix(L)
        tL = torch.sparse_csr_tensor(
            torch.as_tensor(Lc.indptr, dtype=torch.int64, device=dev),
            torch.as_tensor(Lc.indices, dtype=torch.int64, device=dev),
            torch.as_tensor(Lc.data, dtype=dt, device=dev),
            size=Lc.shape)
    denom_P = max(s_target * knn, 1.0)

    xc = tx - tx.mean()
    nx = torch.sqrt((xc * xc).sum())

    for t in range(iters):
        frac = t / max(iters - 1, 1)
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * frac))
        lam = lam0 * (1 + 9 * frac)
        w = torch.sigmoid(theta)
        W = w.sum(dim=1).clamp_min(EPS)
        a = (w @ tS.T) / W[:, None]

        ac = a - a.mean(dim=1, keepdim=True)
        na = torch.sqrt((ac * ac).sum(dim=1))
        denom = (nx * na).clamp_min(EPS)
        r = (ac @ xc) / denom
        g = xc[None, :] / denom[:, None] - r[:, None] * ac / (na ** 2).clamp_min(EPS)[:, None]

        gw = (g @ tS - (g * a).sum(dim=1)[:, None]) / W[:, None]
        gw = gw - 2.0 * lam * (s_target - W)[:, None]
        if tL is not None:
            gw = gw + tmus[:, None] * 2.0 * (tL @ w.T).T / denom_P
        gt = gw * w * (1 - w)

        m = 0.9 * m + 0.1 * gt
        v = 0.999 * v + 0.001 * gt ** 2
        mh = m / (1 - 0.9 ** (t + 1)); vh = v / (1 - 0.999 ** (t + 1))
        theta = torch.clamp(theta - lr * mh / (torch.sqrt(vh) + 1e-8), -12, 12)
    return torch.sigmoid(theta).detach().cpu().numpy()


def solve(x, S, s_target, L=None, mus=0.0, knn=10, iters=400, restarts=4,
          lr0=0.35, lam0=1e-3, seed=0, device="cpu", dtype="float64"):
    """Run `restarts` restarts for each mu, batched, and return hard masks.

    Returns (masks, info) where masks is (M, d) boolean -- one per mu, the best
    restart by TRAINING objective -- and info is a list of dicts. Choosing among
    restarts by training correlation is the paper's procedure; the restart is
    re-chosen on validation models one level up, in fit().
    """
    x = np.asarray(x, float)
    S = np.asarray(S, float)
    d = S.shape[1]
    mus = np.atleast_1d(np.asarray(mus, float))
    M = len(mus)
    s_target = int(s_target)

    rng = np.random.default_rng(seed)
    # Initialise near the target budget so the cardinality term does not have to
    # drag the solution across the whole simplex before the correlation term
    # gets a say.
    bias = np.log(s_target / max(d - s_target, 1))
    theta0 = rng.normal(0.0, 0.1, size=(M * restarts, d)) + bias
    mus_rep = np.repeat(mus, restarts)

    if device != "cpu" and _HAS_TORCH and torch.cuda.is_available():
        w = _solve_torch(x, S, L, s_target, mus_rep, knn, iters, lr0, lam0,
                         theta0, device=device, dtype=dtype)
    elif device == "torch-cpu" and _HAS_TORCH:           # for backend agreement tests
        w = _solve_torch(x, S, L, s_target, mus_rep, knn, iters, lr0, lam0,
                         theta0, device="cpu", dtype=dtype)
    else:
        w = _solve_numpy(x, S, L, s_target, mus_rep, knn, iters, lr0, lam0, theta0)

    masks, info = [], []
    for i, mu in enumerate(mus):
        best, best_r, best_w = None, np.inf, None
        for rs in range(restarts):
            wi = w[i * restarts + rs]
            mk = np.zeros(d, bool)
            mk[np.argsort(-wi)[:s_target]] = True        # discretise: top S
            a = S[:, mk].mean(1)
            r = np.corrcoef(x, a)[0, 1] if np.std(a) > EPS else 0.0
            if r < best_r:
                best_r, best, best_w = r, mk, wi
        masks.append(best)
        info.append(dict(mu=float(mu), r_train=float(best_r), w=best_w))
    return np.array(masks), info


def fit(x_sel, S_sel, s_target, x_val=None, S_val=None, **kw):
    """Select on the selection models; choose the restart on validation models.

    The paper splits its model population 60/20/20 -- selection, restart choice,
    reporting -- so that a subset which merely overfits the models it was fit to
    is caught before it is reported. Passing no validation models falls back to
    choosing on the training objective, which is what solve() does.
    """
    restarts = kw.pop("restarts", 4)
    if x_val is None or S_val is None:
        return solve(x_sel, S_sel, s_target, restarts=restarts, **kw)

    # One restart per call so each candidate can be scored on the val models.
    seed0 = kw.pop("seed", 0)
    cands = [solve(x_sel, S_sel, s_target, restarts=1, seed=seed0 + 1000 * j, **kw)
             for j in range(restarts)]
    mus = np.atleast_1d(np.asarray(kw.get("mus", 0.0), float))
    masks, info = [], []
    for i, mu in enumerate(mus):
        best, best_r, best_inf = None, np.inf, None
        for mk_all, inf_all in cands:
            mk = mk_all[i]
            r_val = evaluate(x_val, S_val, mk)["r"]
            r_val = np.inf if not np.isfinite(r_val) else r_val
            if r_val < best_r:
                best_r, best, best_inf = r_val, mk, inf_all[i]
        masks.append(best)
        info.append({**best_inf, "r_val": float(best_r)})
    return np.array(masks), info


# ------------------------------------------------------------------ baselines
def baseline_random(d: int, s_target: int, rng) -> np.ndarray:
    m = np.zeros(d, bool)
    m[rng.choice(d, s_target, replace=False)] = True
    return m


def baseline_hard(S_sel: np.ndarray, s_target: int) -> np.ndarray:
    """The most-misclassified examples, from the SELECTION models only.

    Built from the selection models because OODSelect never sees the others; a
    version built from all models would be scored on models it had already used.
    The paper reports this baseline near zero, and that gap is what separates
    "models exploit a spurious feature here" from "these examples are hard".
    """
    d = S_sel.shape[1]
    m = np.zeros(d, bool)
    m[np.argsort(S_sel.mean(0))[:s_target]] = True
    return m


def shuffled_x(x: np.ndarray, rng) -> np.ndarray:
    """ID performance with the model labels shuffled -- the selection null.

    Run the identical optimisation against this and score the resulting subset
    against the TRUE x on held-out models. If a subset found against noise still
    inverts, the inversion is a property of the search and nothing else in the
    table means anything. Under a spatial constraint the null must carry the
    SAME mu, because the constraint changes what the optimiser can manufacture.
    """
    y = np.array(x, float, copy=True)
    rng.shuffle(y)
    return y
