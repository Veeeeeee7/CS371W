#!/usr/bin/env python3
"""OODSelect as the authors released it, plus an optional spatial smoothness penalty.

Reimplements the selection of Salaudeen et al., "Aggregation Hides Out-of-Distribution
Generalization Failures from Spurious Correlations" (NeurIPS 2025, arXiv:2510.24884) from
github.com/olawalesalaudeen/OODSELECT; each function names the part of that code it mirrors.
--mu > 0 adds mu * z'Lz / (S k) to the training loss, where L is the Laplacian of the kNN graph
on the OOD points (graph.py). One run is one (cell, budget S, variant).
--reg gtv replaces z'Lz by the graph total variation sum |z_j - z_j'| over the same edges (the
paper's Eq. gtv); see _TotalVariation. The default, laplacian, is the run before --reg existed.
--budget-penalty picks the budget term: softplus (default, the released code) or squared (the
published paper's Eq. 5, which the paper's Approach states); see budget_term and PENALTY_RANGE.

TWO CONTROLS (O3, docs/PROJECT-PLAN.md section 3), both off by default:
  --null-seed N  permute the selection + validation models' ID accuracies before the search
  --split RULE   trees->others / others->trees: select on one model family, test on the other

Reads   <cells>/<tag>.npz and <tag>_models.csv (aotil/zoo.py)
Writes  <out>/<tag>_S<S>_{result.csv,trials.csv,theta.npy,meta.json}; <out> is the variant folder
Example: python aotil/oodselect_paper.py --cells data/interim/aotil/cells --tag FL_band_AE_s0 \\
             --out results/<date>/oodselect_paper/classic --budget 250
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# torch, optuna, sklearn and scipy.stats are imported inside the functions that use them, so
# that the modules importing this one for its numpy helpers start fast.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
from cells import TREE_FAMILIES, load_cell  # noqa: E402

PAPER_BUDGETS = [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500,
                 10000, 15000, 20000, 25000, 50000]     # OODSelect_experiments.py
N_TRIALS = 100            # Optuna trials per budget
NUM_EPOCHS = 50_000       # maximum epochs per trial
PATIENCE = 500            # early-stopping patience, epochs
EPS = 1e-6                # probit clamp and variance regulariser
AOTIL_THRESHOLD = -0.3    # the paper's cut: r_test at or below it is accuracy-on-the-inverse-line

BUDGET_PENALTIES = ("softplus", "squared")
# Optuna's range (low, high, log) for the budget weight lambda under each budget term. softplus:
# the released code's uniform [0.1, 100]. squared: Eq. 5's term is in points squared (6.25e6 at
# S = 500 and a soft total of 3,000) while the correlation lies in [-1, 1], so lambda is searched
# log-uniform over a wider range (option b of docs/TODO-CLAUDE-CODE.md item 1). The range is
# ours, not the paper's.
PENALTY_RANGE = {"softplus": (1e-1, 1e2, False), "squared": (1e-6, 1e2, True)}

REGS = ("laplacian", "gtv")   # the spatial loss R under mu > 0 (paper Eq. laplacian, Eq. gtv)

SPLITS = ("random", "trees->others", "others->trees")
SPLIT_SUFFIX = {"random": "", "trees->others": "_trees2others", "others->trees": "_others2trees"}
FAMILY_VAL_FRAC = 0.25    # share of the selection family held out as validation


# ------------------------------------------------------------------ data
def correctness(S: np.ndarray, y: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Z[i, j] = 1 if model i's argmax prediction on point j is correct (extract_data).

    S is the stored true-class probability: p if y == 1 else 1 - p. argmax over [1-p, p] picks
    class 0 on a tie, so a y = 1 point is correct iff S > 0.5 and a y = 0 point iff S >= 0.5.
    Without y (cells written before y_id was stored) a tie counts as incorrect.
    Returns (Z, the number of ties left unresolved).
    """
    S = np.asarray(S, float)
    if y is None:
        return (S > 0.5).astype(np.float32), int((S == 0.5).sum())
    y = np.asarray(y)
    Z = (S > 0.5) | ((S == 0.5) & (y[None, :] == 0))
    return Z.astype(np.float32), 0


def load_paper_cell(cells_dir: str | Path, tag: str):
    """The cell as the paper sees it: (cell, binary Z on the OOD points, ID accuracy, info)."""
    cell = load_cell(cells_dir, tag)
    y_id = None
    with np.load(Path(cells_dir) / f"{tag}.npz", allow_pickle=True) as npz:
        if "y_id" in npz.files:
            y_id = npz["y_id"]
    Z_ood, _ = correctness(cell.S_ood, cell.y_ood)
    Z_id, ties = correctness(cell.S_id, y_id)
    acc_id = Z_id.mean(axis=1)
    info = dict(n_models=int(Z_ood.shape[0]), d=int(Z_ood.shape[1]),
                y_id_stored=y_id is not None, id_ties_unresolved=ties)
    return cell, Z_ood, acc_id, info


def model_split(n_models: int):
    """The paper's 60/20/20 selection / validation / test models (prepare_dataset).

    Two sklearn splits at random_state=42. Splitting np.arange(N) gives the same permutation as
    splitting (X, y), because train_test_split shuffles by index.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(n_models)
    rest, test = train_test_split(idx, test_size=0.2, random_state=42)
    selection, validation = train_test_split(rest, test_size=0.25, random_state=42)
    return np.asarray(selection), np.asarray(validation), np.asarray(test)


def family_split(kind, rule: str, seed: int):
    """trees->others selects on the tree families (cells.TREE_FAMILIES) and tests on the rest;
    any other rule does the reverse. The protocol needs a validation block (Optuna's objective,
    early stopping), so a seeded random FAMILY_VAL_FRAC of the selection family is held out.
    Returns (selection, validation, test), each sorted."""
    is_tree = np.isin(np.asarray(kind).astype(str), sorted(TREE_FAMILIES))
    if rule == "trees->others":
        in_selection_family = is_tree
    else:
        in_selection_family = ~is_tree
    family = np.where(in_selection_family)[0]
    test = np.where(~in_selection_family)[0]
    perm = np.random.default_rng(seed).permutation(family)
    n_val = int(round(FAMILY_VAL_FRAC * len(family)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val]), test


def make_split(n_models: int, rule: str = "random", kind=None, seed: int = 0):
    """(selection, validation, test) model indices under a --split rule."""
    if rule == "random":
        return model_split(n_models)
    if rule not in SPLITS:
        raise ValueError(f"unknown split rule {rule!r}; one of {SPLITS}")
    if kind is None or len(kind) != n_models:
        raise ValueError(f"split {rule!r} needs the model kinds (one per model)")
    return family_split(kind, rule, seed)


def null_accuracies(acc: np.ndarray, split, null_seed: int | None) -> np.ndarray:
    """The ID accuracies the search sees: acc itself, or under --null-seed a copy whose
    selection and validation entries are permuted together (the test entries are kept)."""
    if null_seed is None:
        return acc
    selection, validation, _ = split
    idx = np.concatenate([selection, validation])
    perm = np.random.default_rng(null_seed).permutation(len(idx))
    out = np.array(acc, copy=True)
    out[idx] = acc[idx][perm]
    return out


def variant_name(mu: float, knn: int, split: str = "random", null_seed: int | None = None,
                 reg: str = "laplacian") -> str:
    """classic, spatial_mu<mu>_knn<k> (Laplacian) or spatial_gtv_mu<mu>_knn<k>, then the split
    suffix, then _null<N>. At mu = 0 the loss has no spatial term, so reg does not enter."""
    if reg not in REGS:
        raise ValueError(f"unknown reg {reg!r}; one of {REGS}")
    if mu == 0:
        name = "classic"
    elif reg == "gtv":
        name = f"spatial_gtv_mu{mu:g}_knn{knn}"
    else:
        name = f"spatial_mu{mu:g}_knn{knn}"
    name += SPLIT_SUFFIX[split]
    if null_seed is not None:
        name += f"_null{null_seed}"
    return name


def run_controls(res) -> tuple[str, int | None, int]:
    """(split rule, null seed, seed) of a stored run, from one result.csv row. A row without
    the split or null_seed column (written before the controls existed) reads as random, None."""
    rule = res.get("split", "random")
    if not isinstance(rule, str) or rule == "":
        rule = "random"
    null_seed = res.get("null_seed", None)
    if null_seed is None or (isinstance(null_seed, float) and math.isnan(null_seed)):
        null_seed = None
    else:
        null_seed = int(null_seed)
    return rule, null_seed, int(res.get("seed", 0))


# ------------------------------------------------------------------ loss
def _torch():
    import torch
    return torch


def probit_t(x, eps: float = EPS):
    """probit_transform: the standard normal quantile of x clamped to [eps, 1 - eps]."""
    torch = _torch()
    x = torch.clamp(x, eps, 1 - eps)
    return math.sqrt(2) * torch.erfinv(2 * x - 1)


def pearson_r_loss(X, y, z, eps: float = EPS):
    """PearsonRLoss.forward, the same arithmetic: population Pearson r of probit(X z / sum z)
    against probit(y), each variance regularised by eps."""
    torch = _torch()
    z_sum = z.sum() + 1e-6
    x_z = (X @ z) / z_sum
    x_t = probit_t(x_z)
    y_t = probit_t(y)
    x_c = x_t - torch.mean(x_t)
    y_c = y_t - torch.mean(y_t)
    cov = torch.mean(x_c * y_c)
    x_var = torch.mean(x_c ** 2) + eps
    y_var = torch.mean(y_c ** 2) + eps
    return cov / torch.sqrt(x_var * y_var)


def budget_term(total, S: int, kind: str = "softplus"):
    """B_S of the soft total sum(z), before the weight lambda * warm-up.

    softplus: the released code's one-sided log(1 + exp(S - total)), ~0 once total > S.
    squared: the published paper's (S - total)^2 (Salaudeen et al. 2025, Eq. 5), which pulls the
    total toward S from either side.
    """
    torch = _torch()
    if kind == "softplus":
        return torch.nn.functional.softplus(S - total)
    if kind == "squared":
        return (S - total) ** 2
    raise ValueError(f"unknown budget penalty {kind!r}; one of {BUDGET_PENALTIES}")


def binary_selection(theta, k: int):
    """get_binary_selection: 1 on the top-k entries of the raw parameter."""
    torch = _torch()
    idx = torch.topk(theta.detach(), k).indices
    m = torch.zeros_like(theta)
    m[idx] = 1.0
    return m


def hard_corr(Z, acc, theta_or_mask, k: int, is_mask: bool = False) -> float:
    """TestSetFinder.test on numpy inputs: r of the top-k mask of theta (or of a given mask)."""
    torch = _torch()
    X = torch.as_tensor(np.asarray(Z, np.float32))
    y = torch.as_tensor(np.asarray(acc, np.float32))
    v = torch.as_tensor(np.asarray(theta_or_mask, np.float32))
    m = v if is_mask else binary_selection(v, k)
    return float(pearson_r_loss(X, y, m).item())


class _Quadratic:
    """z'Lz for a fixed symmetric sparse L, with gradient 2Lz.

    A scipy mat-vec inside a torch autograd Function: torch's CPU sparse mat-vec and its
    backward cost ~10 ms here against ~0.2 ms for scipy, once per epoch of every trial.
    aotil/test_oodselect_paper.py checks it against torch.sparse.
    """

    def __init__(self, L):
        import scipy.sparse as sp
        torch = _torch()
        L = sp.csr_matrix(L)
        if abs(L - L.T).max() > 0:
            raise ValueError("L must be symmetric")
        L = L.astype(np.float64)

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, z):
                zn = z.detach().cpu().numpy().astype(np.float64)
                Lz = L @ zn
                ctx.save_for_backward(torch.as_tensor(Lz, dtype=z.dtype))
                return torch.as_tensor(float(zn @ Lz), dtype=z.dtype)

            @staticmethod
            def backward(ctx, g):
                (Lz,) = ctx.saved_tensors
                return 2.0 * g * Lz

        self._fn = Fn

    def __call__(self, z):
        return self._fn.apply(z)


class _TotalVariation:
    """The graph total variation sum |z_j - z_j'| over the edges of L's graph, each undirected
    edge once, as z'Lz counts it (the paper's Eq. gtv before the division by S k).

    The edges are the strict upper triangle of L, whose off-diagonal entries are -1 on an edge
    (L = D - A with 0/1 A). On a 0/1 z both this and z'Lz count the cut edges. The gradient is
    autograd's through torch.abs, whose subgradient at a tie z_j = z_j' is 0 (paper 3.5).
    """

    def __init__(self, L):
        import scipy.sparse as sp
        torch = _torch()
        L = sp.csr_matrix(L)
        if abs(L - L.T).max() > 0:
            raise ValueError("L must be symmetric")
        upper = sp.triu(L, k=1).tocoo()
        upper.eliminate_zeros()
        if upper.nnz and not np.all(upper.data == -1):
            raise ValueError("L must be the Laplacian of a 0/1 adjacency (off-diagonals -1 or 0)")
        self.j = torch.as_tensor(upper.row, dtype=torch.int64)
        self.j_prime = torch.as_tensor(upper.col, dtype=torch.int64)
        self.n_edges = int(upper.nnz)

    def __call__(self, z):
        torch = _torch()
        z64 = z.double()
        return torch.abs(z64[self.j] - z64[self.j_prime]).sum().to(z.dtype)


def spatial_term(L, reg: str = "laplacian"):
    """The callable z -> numerator of R (before the division by S k) that train_one adds:
    z'Lz for reg = laplacian, the total variation for reg = gtv; None without a graph."""
    if reg not in REGS:
        raise ValueError(f"unknown reg {reg!r}; one of {REGS}")
    if L is None:
        return None
    if reg == "gtv":
        return _TotalVariation(L)
    return _Quadratic(L)


def laplacian_torch(L):
    """scipy CSR -> torch sparse CSR (float32), the reference _Quadratic is checked against."""
    torch = _torch()
    import scipy.sparse as sp
    Lc = sp.csr_matrix(L).astype(np.float32)
    return torch.sparse_csr_tensor(torch.as_tensor(Lc.indptr, dtype=torch.int64),
                                   torch.as_tensor(Lc.indices, dtype=torch.int64),
                                   torch.as_tensor(Lc.data), size=Lc.shape)


class TrialFailed(RuntimeError):
    """The training correlation went NaN. The released code raises and ends the whole search;
    here the trial is recorded as failed and the search goes on."""


def train_one(X_tr, y_tr, X_va, y_va, S: int, hp: dict, *, max_epochs: int,
              patience: int, seed: int, L_t=None, mu: float = 0.0, knn: int = 10,
              budget_penalty: str = "softplus"):
    """One fit of the authors' TestSetFinder, written as the loop PyTorch Lightning runs
    (Lightning's per-step overhead costs ~10x in wall clock and changes no arithmetic).
    budget_penalty picks the budget term (budget_term); softplus is the authors' code exactly.

    theta ~ N(0, 1) from a generator seeded by `seed` (the released code does not seed it).
    Returns theta at the epoch training stopped, as the released code keeps it (its best-epoch
    checkpoint is never read back), and the last validation r, which tune_hparams returns.
    L_t is a callable z -> the numerator of R (spatial_term: z'Lz, or the total variation under
    --reg gtv) or a torch sparse matrix L; the spatial penalty enters the training loss only, and
    only when mu > 0. P_soft is that R on the final soft weights.
    """
    torch = _torch()
    if L_t is None:
        quad = None
    elif callable(L_t):
        quad = L_t
    else:
        def quad(z):
            return z @ (L_t @ z)

    generator = torch.Generator().manual_seed(int(seed))
    theta = torch.randn(X_tr.shape[1], generator=generator).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=hp["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["T_max"],
                                                     eta_min=hp["eta_min"])
    # For a 0/1 mask z'Lz counts the cut edges; dividing by S * k makes it an incoherence score
    # in [0, ~1]. mu is fixed, never tuned: tuning it on validation r would drive it to zero.
    cut_norm = float(max(S * knn, 1))
    best, wait, best_epoch = math.inf, 0, -1
    val = math.nan
    epoch = 0
    # One epoch = one full-batch step, in Lightning's order: training_step, optimiser step,
    # validation r of the hard top-S mask, EarlyStopping check, CosineAnnealingLR step.
    for epoch in range(max_epochs):
        z = torch.sigmoid(theta)
        corr = pearson_r_loss(X_tr, y_tr, z)
        if torch.isnan(corr):
            raise TrialFailed(f"correlation NaN at epoch {epoch}")
        # The budget term, cosine-warmed as in the authors' code (epoch is Lightning's
        # global_step here): their one-sided softplus by default, or with
        # budget_penalty="squared" the (S - ||s||_1)^2 of the paper's Eq. 5.
        progress = min(max(epoch / float(hp["penalty_anneal_iters"]), 0.0), 1.0)
        warm = 0.5 * (1.0 - math.cos(math.pi * progress))
        loss = corr + hp["penalty"] * warm * budget_term(z.sum(), S, budget_penalty)
        if mu > 0 and quad is not None:
            loss = loss + mu * quad(z) / cut_norm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            val = float(pearson_r_loss(X_va, y_va, binary_selection(theta, S)).item())
        if not math.isfinite(val):             # EarlyStopping(check_finite=True)
            break
        if val < best:
            best, wait, best_epoch = val, 0, epoch
        else:
            wait += 1
            if wait >= patience:
                break
        sch.step()

    theta_np = theta.detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        z = torch.sigmoid(theta)
        soft_mass = float(z.sum().item())
        P_soft = math.nan
        if quad is not None:
            P_soft = float((quad(z) / cut_norm).item())
    return dict(theta=theta_np, val_last=val, val_best=best, best_epoch=best_epoch,
                epochs=epoch + 1, soft_mass=soft_mass, P_soft=P_soft)


# ------------------------------------------------------------------ search
def run_search(Z, acc, split, S: int, *, n_trials: int, max_epochs: int,
               patience: int, seed: int, L=None, mu: float = 0.0, knn: int = 10,
               log_every: int = 10, budget_penalty: str = "softplus", reg: str = "laplacian"):
    """The authors' Optuna search (tune_hparams, run_optuna): minimise the last validation r over
    their five hyperparameters. Returns (best trial, its theta, a DataFrame with one record per
    trial). Trial t trains from seed * 100_003 + t, and the TPE sampler is seeded with `seed`.
    The budget weight lambda ("penalty") is drawn from PENALTY_RANGE[budget_penalty]; reg picks
    the spatial loss on L's graph (spatial_term)."""
    import optuna
    torch = _torch()
    if budget_penalty not in PENALTY_RANGE:
        raise ValueError(f"unknown budget penalty {budget_penalty!r}; one of {BUDGET_PENALTIES}")
    penalty_low, penalty_high, penalty_log = PENALTY_RANGE[budget_penalty]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selection, validation, _ = split
    X_tr = torch.as_tensor(Z[selection])
    y_tr = torch.as_tensor(acc[selection].astype(np.float32))
    X_va = torch.as_tensor(Z[validation])
    y_va = torch.as_tensor(acc[validation].astype(np.float32))
    # Passed for both variants, so that P_soft is logged for the classic run too.
    quadratic = spatial_term(L, reg)

    records = []
    thetas = {}
    t_start = time.time()

    def objective(trial):
        hp = dict(
            lr=trial.suggest_float("lr", 1e-2, 1e2, log=True),
            eta_min=trial.suggest_float("eta_min", 1e-3, 1e0, log=True),
            T_max=trial.suggest_int("T_max", 10, 200),
            penalty=trial.suggest_float("penalty", penalty_low, penalty_high, log=penalty_log),
            penalty_anneal_iters=trial.suggest_int("penalty_anneal_iters", 10, 1000),
        )
        t0 = time.time()
        try:
            out = train_one(X_tr, y_tr, X_va, y_va, S, hp, max_epochs=max_epochs,
                            patience=patience, seed=seed * 100_003 + trial.number,
                            L_t=quadratic, mu=mu, knn=knn, budget_penalty=budget_penalty)
        except TrialFailed as e:
            records.append(dict(trial=trial.number, state="failed", note=str(e),
                                secs=time.time() - t0, **hp))
            raise optuna.TrialPruned(str(e))
        thetas[trial.number] = out.pop("theta")
        records.append(dict(trial=trial.number, state="complete",
                            secs=time.time() - t0, **hp, **out))
        if log_every and (trial.number % log_every == 0 or trial.number == n_trials - 1):
            completed = [r["val_last"] for r in records if r["state"] == "complete"]
            best_val = min(completed) if completed else math.nan
            print(f"    trial {trial.number:3d}  val {out['val_last']:+.3f}  "
                  f"(best {best_val:+.3f})  epochs {out['epochs']:6d}  "
                  f"soft total {out['soft_mass']:8.1f}  "
                  f"{time.time() - t0:5.1f}s  elapsed {time.time() - t_start:6.0f}s",
                  flush=True)
        return out["val_last"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, catch=())
    best = study.best_trial
    return best, thetas[best.number], pd.DataFrame(records)


# ------------------------------------------------------------------ stats
def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher z-interval of a Pearson r over n models; NaN when r is not finite or n <= 3."""
    from scipy.stats import norm
    if not np.isfinite(r) or n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    h = norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - h)), float(np.tanh(z + h))


def plain_stats(Z, acc, mask) -> dict:
    """scipy Pearson and Spearman of probit(ID acc) vs probit(subset acc)."""
    from scipy.stats import norm, pearsonr, spearmanr
    mask = np.asarray(mask, bool)
    a = Z[:, mask].mean(1)
    x = norm.ppf(np.clip(acc, EPS, 1 - EPS))
    yv = norm.ppf(np.clip(a, EPS, 1 - EPS))
    if np.std(yv) == 0 or np.std(x) == 0:
        return dict(pearson=np.nan, p=np.nan, spearman=np.nan)
    r, p = pearsonr(x, yv)
    rho, _ = spearmanr(acc, a)
    return dict(pearson=float(r), p=float(p), spearman=float(rho))


def budgets_for(d: int) -> list[int]:
    """The authors' grid, truncated below the OOD set size, as they do per dataset (S = d is the
    full set, which needs no search; report_paper.py scores it)."""
    return [b for b in PAPER_BUDGETS if b < d]


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--tag", required=True, help="cell tag, e.g. FL_band_AE_s0")
    ap.add_argument("--out", help="variant output directory (required unless --list-budgets)")
    ap.add_argument("--budget", type=int, help="S, the number of OOD points to select")
    ap.add_argument("--list-budgets", action="store_true",
                    help="print the budgets of the paper grid below this cell's OOD set size")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="spatial smoothness weight; 0 = the paper exactly")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--reg", choices=REGS, default="laplacian",
                    help="spatial loss under mu > 0: the graph Laplacian (default) or graph "
                         "total variation (folder spatial_gtv_mu<mu>_knn<k>)")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-seed", type=int, default=None,
                    help="null-select: permute the selection + validation models' ID "
                         "accuracies with this seed; default off")
    ap.add_argument("--split", choices=SPLITS, default="random",
                    help="model split: the paper's 60/20/20 (random) or family-disjoint")
    ap.add_argument("--budget-penalty", choices=BUDGET_PENALTIES, default="softplus",
                    help="budget term: softplus (the released code, default) or squared (the "
                         "published paper's Eq. 5, with lambda searched log-uniform over a "
                         "wider range)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo a budget whose result.csv exists")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    cell, Z, acc, info = load_paper_cell(args.cells, args.tag)
    d = info["d"]
    if args.list_budgets:
        print(" ".join(str(b) for b in budgets_for(d)))
        return 0
    if not args.out:
        ap.error("--out is required")
    if args.budget is None:
        ap.error("--budget is required")
    S = args.budget
    if S >= d:
        # The authors' scripts/OODSelect.py exits when the budget exceeds the OOD set; S == d
        # is the full set.
        print(f"{args.tag}: budget {S} >= d = {d}; nothing to select, skipping")
        return 0

    out = Path(args.out)
    variant = variant_name(args.mu, args.knn, args.split, args.null_seed, args.reg)
    is_control = args.split != "random" or args.null_seed is not None
    if is_control and out.name != variant:
        # A control run in the folder of the real run would be read as the real run.
        ap.error(f"--split {args.split} / --null-seed {args.null_seed} is variant {variant}; "
                 f"--out must be a folder of that name, not {out.name}")
    if args.mu > 0 and args.reg != "laplacian" and out.name != variant:
        # Likewise a GTV run in the Laplacian folder would be read as the Laplacian run.
        ap.error(f"--reg {args.reg} is variant {variant}; --out must be a folder of that name, "
                 f"not {out.name}")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{args.tag}_S{S}"
    if Path(f"{stem}_result.csv").exists() and not args.overwrite:
        print(f"{stem}_result.csv exists; skipping (--overwrite to redo)")
        return 0

    torch = _torch()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    split = make_split(info["n_models"], args.split, cell.models.kind.to_numpy(), args.seed)
    selection, validation, test = split
    acc_search = null_accuracies(acc, split, args.null_seed)
    ood_acc = Z.mean(1)
    print(f"=== OODSelect (paper protocol) {args.tag}  S={S}  d={d}  variant={variant} ===")
    print(f"  models {info['n_models']}: selection {len(selection)} | validation "
          f"{len(validation)} | test {len(test)}  (split {args.split})")
    if args.null_seed is not None:
        print(f"  NULL-SELECT: selection + validation ID accuracies permuted with seed "
              f"{args.null_seed}; test scored against the true accuracies")
    print(f"  ID accuracy {acc.min():.3f}-{acc.max():.3f} | OOD accuracy "
          f"{ood_acc.min():.3f}-{ood_acc.max():.3f}")
    print(f"  trials {args.trials}  max epochs {args.max_epochs}  patience {args.patience}  "
          f"threads {args.threads}  torch {torch.__version__}  device cpu")
    low, high, log = PENALTY_RANGE[args.budget_penalty]
    print(f"  budget penalty {args.budget_penalty}: lambda in [{low:g}, {high:g}] "
          f"{'log-uniform' if log else 'uniform'}")
    if args.reg != "laplacian":
        print(f"  spatial loss {args.reg} (mu {args.mu:g})")
    if not info["y_id_stored"]:
        print(f"  note: cell predates y_id; ID ties at p=0.5 counted incorrect "
              f"({info['id_ties_unresolved']} such entries)")

    # The graph is built for both variants: the classic run never sees it in its loss, but the
    # coherence of its subset on the graph is the baseline the spatial run is judged against.
    A = None
    L = None
    if cell.has_spatial():
        A = graph.build_adjacency(cell.lon, cell.lat, k=args.knn)
        L = graph.laplacian(A)
    elif args.mu > 0:
        cell.require_spatial()

    # One seed per (seed, budget): the paper selects each S independently and reads the overlap
    # across budgets (Eq. 6) as consistency. With one shared seed every budget would get the
    # same Optuna draws and theta init, and since the one-sided (softplus) budget term is ~0 once
    # sum(z) > S, the selections would come out nested by construction. The two variants share
    # the seed at a given S, which pairs them; so does null-select, so its theta inits and
    # Optuna's first 10 (random startup) draws are the real run's.
    seed_eff = args.seed * 1_000_003 + S
    t0 = time.time()
    best, theta, trials = run_search(Z, acc_search, split, S, n_trials=args.trials,
                                     max_epochs=args.max_epochs, patience=args.patience,
                                     seed=seed_eff, L=L, mu=args.mu, knn=args.knn,
                                     budget_penalty=args.budget_penalty, reg=args.reg)
    secs = time.time() - t0

    # r_train and r_val use the accuracies the search saw (shuffled under null-select); every
    # test number uses the true ones.
    mask = np.zeros(d, bool)
    mask[np.argsort(-theta, kind="stable")[:S]] = True
    r_train = hard_corr(Z[selection], acc_search[selection], theta, S)
    r_val = hard_corr(Z[validation], acc_search[validation], theta, S)
    r_test = hard_corr(Z[test], acc[test], theta, S)
    r_full_test = hard_corr(Z[test], acc[test], np.ones(d, np.float32), d, is_mask=True)
    stats = plain_stats(Z[test], acc[test], mask)
    lo, hi = fisher_ci(r_test, len(test))
    best_record = trials[trials.trial == best.number].iloc[0].to_dict()

    row = dict(cell=args.tag, variant=variant, split=args.split, null_seed=args.null_seed,
               mu=args.mu, knn=args.knn, budget_penalty=args.budget_penalty, reg=args.reg,
               budget=S, d=d,
               frac=S / d, n_sel=len(selection), n_val=len(validation), n_test=len(test),
               r_train=r_train, r_val=r_val, r_test=r_test,
               r_test_ci_lo=lo, r_test_ci_hi=hi,
               pearson_test_scipy=stats["pearson"], p_test=stats["p"],
               spearman_test=stats["spearman"], r_full_test=r_full_test,
               aotil=bool(r_test <= AOTIL_THRESHOLD),
               best_trial=best.number, epochs=int(best_record["epochs"]),
               best_epoch=int(best_record["best_epoch"]),
               soft_mass=best_record["soft_mass"], P_soft=best_record["P_soft"],
               n_trials=len(trials), n_failed=int((trials.state == "failed").sum()),
               search_secs=secs, threads=args.threads, seed=args.seed, seed_eff=seed_eff)
    for name, value in best.params.items():
        row[f"hp_{name}"] = value
    if A is not None:
        metrics = graph.coherence_metrics(mask, A, args.knn, tiles=cell.tile_50km,
                                          blocks=cell.block_id)
        for name, value in metrics.items():
            if name not in ("k", "frac"):
                row[f"coh_{name}"] = value

    np.save(f"{stem}_theta.npy", theta)
    trials.to_csv(f"{stem}_trials.csv", index=False)
    pd.DataFrame([row]).to_csv(f"{stem}_result.csv", index=False)
    # split_rule and null_seed (also in result.csv) let audit_selection.py and report_paper.py
    # rebuild the same split with make_split.
    meta = dict(info, variant=variant, budget=S, split_rule=args.split, null_seed=args.null_seed,
                budget_penalty=args.budget_penalty, reg=args.reg,
                split=dict(selection=selection.tolist(), validation=validation.tolist(),
                           test=test.tolist()))
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"  best trial {best.number}: val {r_val:+.3f} | TEST r = {r_test:+.3f} "
          f"[{lo:+.2f}, {hi:+.2f}]  rho {stats['spearman']:+.3f} | full OOD set "
          f"{r_full_test:+.3f} | {secs / 60:.1f} min")
    print(f"  soft total {row['soft_mass']:.1f} (S = {S}), stopped at epoch {row['epochs']}, "
          f"lambda {row['hp_penalty']:.3g}")
    if A is not None:
        print(f"  coherence: purity {row['coh_purity']:.2f}  components {row['coh_n_cc']}  "
              f"largest {row['coh_largest_cc_frac']:.2f}  tiles {row['coh_tiles']}")
    print(f"  wrote {stem}_result.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
