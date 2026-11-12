#!/usr/bin/env python3
"""The subset search of the paper (Algorithm alg:spatial): OODSelect, and OODSelect with the Laplacian or the
graph total variation (GTV) loss.

Reimplements the selection of Salaudeen et al., "Aggregation Hides Out-of-Distribution Generalization Failures
from Spurious Correlations" (NeurIPS 2025, arXiv:2510.24884) from github.com/olawalesalaudeen/OODSELECT; each
function names the part of that code it mirrors. The paper's settings:
  corr        Pearson correlation of probit accuracies, clipped to [1e-6, 1 - 1e-6] (pearson_r_loss)
  budget      the squared penalty (S - ||s||_1)^2 with weight lambda, warmed up along half a cosine wave over the
              first W steps (penalty_anneal_iters); lambda drawn log-uniformly from [1e-6, 1e2]
  spatial     mu * R(s) / (S k) on the symmetric kNN graph of the OOD points (graph.build_adjacency, k = 10):
              R = s'Ls for the Laplacian, sum |s_j - s_j'| over the edges for GTV; mu = 0 is OODSelect
  optimizer   full-batch Adam with a cosine learning rate (initial lr, minimum eta_min, half-period tau = T_max)
  trials      T = 100 Optuna TPE trials over lr, eta_min, tau, lambda and W; at most 50,000 steps per trial,
              patience P = 500 steps on the validation correlation of the top-S subset
  models      the 60/20/20 selection / validation / test split at random_state=42 (model_split)
Seeds: a search at budget S uses seed_eff = seed * 1,000,003 + S for Optuna's sampler, and trial t initializes
theta ~ N(0, I) from seed_eff * 100,003 + t, so every budget is searched independently and every search is
reproducible on one thread.

run_experiments.py starts one process per search (`python aotil/search.py --cell-dir ... --S 500 --method gtv
--mu 3 --out ...`), single-threaded; nothing else should call the command line. Writes, in --out:
  theta.npy     the final weights theta of the best trial (float32, one per OOD point)
  trials.csv    one row per trial (hyperparameters, steps, last and best validation correlation, soft total)
  result.json   the search's summary (written last: a unit is finished when it exists)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# torch, optuna and sklearn are imported inside the functions that use them, so the torch-free modules
# (subsets.py, make_paper.py) can import the numpy helpers here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
from cells import load_cell  # noqa: E402

N_TRIALS = 100            # T, Optuna trials per search
MAX_STEPS = 50_000        # at most this many full-batch steps per trial
PATIENCE = 500            # P, steps without a better validation correlation before a trial stops
KNN = 10                  # k of the graph
EPS = 1e-6                # probit clip and variance regularizer
METHODS = ("oodselect", "laplacian", "gtv")
# Optuna's range (low, high, log) for the budget weight lambda of the squared penalty. The term is in points
# squared (6.25e6 at S = 500 and a soft total of 3,000) while the correlation lies in [-1, 1], so lambda is
# searched log-uniformly over a wide range (the paper's Appendix).
PENALTY_RANGE = (1e-6, 1e2, True)


# ------------------------------------------------------------------ data
def correctness(S: np.ndarray, y: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Z[i, j] = 1 if model i's argmax prediction on point j is correct (extract_data).

    S is the stored true-class probability: p if y == 1 else 1 - p. argmax over [1-p, p] picks
    class 0 on a tie, so a y = 1 point is correct iff S > 0.5 and a y = 0 point iff S >= 0.5.
    Without y a tie counts as incorrect. Returns (Z as float32, the number of ties left unresolved).
    """
    S = np.asarray(S, float)
    if y is None:
        return (S > 0.5).astype(np.float32), int((S == 0.5).sum())
    y = np.asarray(y)
    Z = (S > 0.5) | ((S == 0.5) & (y[None, :] == 0))
    return Z.astype(np.float32), 0


def load_paper_cell(cells_dir: str | Path, tag: str):
    """The cell as the search sees it: (cell, binary Z on the OOD points, ID accuracy, info)."""
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


def unit_name(method: str, mu: float, S: int) -> str:
    """The run folder's name of one search: oodselect_S500, laplacian_mu3_S500, gtv_mu0.01_S500."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    if method == "oodselect":
        return f"oodselect_S{S}"
    return f"{method}_mu{mu:g}_S{S}"


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


def budget_term(total, S: int):
    """The squared budget penalty (S - ||s||_1)^2 of the soft total (Salaudeen et al. 2025, Eq. 5), before the
    weight lambda and its warm-up."""
    return (S - total) ** 2


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
    """z'Lz for a fixed symmetric sparse L, with gradient 2Lz (the Laplacian loss before the division by S k).

    A scipy mat-vec inside a torch autograd Function: torch's CPU sparse mat-vec and its
    backward cost ~10 ms here against ~0.2 ms for scipy, once per step of every trial.
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
    autograd's through torch.abs, whose subgradient at a tie z_j = z_j' is 0.
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
    """The callable z -> numerator of R (before the division by S k): z'Lz for the Laplacian, the total
    variation for GTV; None without a graph."""
    if reg not in ("laplacian", "gtv"):
        raise ValueError(f"unknown spatial loss {reg!r}")
    if L is None:
        return None
    if reg == "gtv":
        return _TotalVariation(L)
    return _Quadratic(L)


class TrialFailed(RuntimeError):
    """The training correlation went NaN. The released code raises and ends the whole search;
    here the trial is recorded as failed and the search goes on."""


def train_one(X_tr, y_tr, X_va, y_va, S: int, hp: dict, *, max_epochs: int,
              patience: int, seed: int, L_t=None, mu: float = 0.0, knn: int = KNN):
    """One fit of the authors' TestSetFinder, written as the loop PyTorch Lightning runs
    (Lightning's per-step overhead costs ~10x in wall clock and changes no arithmetic). One epoch is one
    full-batch step.

    theta ~ N(0, 1) from a generator seeded by `seed` (the released code does not seed it).
    Returns theta at the step training stopped, as the released code keeps it (its best-step
    checkpoint is never read back), and the last validation r, which run_search minimizes.
    L_t is a callable z -> the numerator of R (spatial_term); the spatial loss enters the training loss
    only, and only when mu > 0. P_soft is that R / (S k) on the final soft weights.
    """
    torch = _torch()
    quad = L_t

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
    # One step in Lightning's order: training_step, optimiser step, validation r of the hard top-S
    # mask, EarlyStopping check, CosineAnnealingLR step.
    for epoch in range(max_epochs):
        z = torch.sigmoid(theta)
        corr = pearson_r_loss(X_tr, y_tr, z)
        if torch.isnan(corr):
            raise TrialFailed(f"correlation NaN at step {epoch}")
        # The squared budget term, cosine-warmed over the first penalty_anneal_iters (W) steps.
        progress = min(max(epoch / float(hp["penalty_anneal_iters"]), 0.0), 1.0)
        warm = 0.5 * (1.0 - math.cos(math.pi * progress))
        loss = corr + hp["penalty"] * warm * budget_term(z.sum(), S)
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
    return dict(theta=theta_np, val_last=val, val_best=best, best_step=best_epoch,
                steps=epoch + 1, soft_mass=soft_mass, P_soft=P_soft)


# ------------------------------------------------------------------ search
def run_search(Z, acc, split, S: int, *, n_trials: int, max_epochs: int,
               patience: int, seed: int, L=None, mu: float = 0.0, knn: int = KNN,
               log_every: int = 10, reg: str = "laplacian"):
    """The authors' Optuna search (tune_hparams, run_optuna): minimise the last validation r over
    their five hyperparameters. Returns (best trial, its theta, a DataFrame with one record per
    trial). Trial t trains from seed * 100_003 + t, and the TPE sampler is seeded with `seed`.
    reg picks the spatial loss on L's graph (spatial_term)."""
    import optuna
    torch = _torch()
    penalty_low, penalty_high, penalty_log = PENALTY_RANGE
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selection, validation, _ = split
    X_tr = torch.as_tensor(Z[selection])
    y_tr = torch.as_tensor(acc[selection].astype(np.float32))
    X_va = torch.as_tensor(Z[validation])
    y_va = torch.as_tensor(acc[validation].astype(np.float32))
    # Passed for every method, so that P_soft is logged for OODSelect too.
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
                            L_t=quadratic, mu=mu, knn=knn)
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
                  f"(best {best_val:+.3f})  steps {out['steps']:6d}  "
                  f"soft total {out['soft_mass']:8.1f}  "
                  f"{time.time() - t0:5.1f}s  elapsed {time.time() - t_start:6.0f}s",
                  flush=True)
        return out["val_last"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, catch=())
    best = study.best_trial
    return best, thetas[best.number], pd.DataFrame(records)


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher z-interval of a Pearson r over n models; NaN when r is not finite or n <= 3."""
    from scipy.stats import norm
    if not np.isfinite(r) or n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    h = norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - h)), float(np.tanh(z + h))


# ------------------------------------------------------------------ one search
def run_unit(cell_dir: Path, tag: str, out: Path, S: int, method: str, mu: float, *,
             knn: int = KNN, trials: int = N_TRIALS, max_steps: int = MAX_STEPS,
             patience: int = PATIENCE, seed: int = 0, threads: int = 1) -> dict:
    """Search one (method, mu, S) on the cell and write theta.npy, trials.csv and result.json to out."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    if (method == "oodselect") != (mu == 0):
        raise ValueError("OODSelect is mu = 0; the Laplacian and GTV need mu > 0")
    reg = "gtv" if method == "gtv" else "laplacian"
    cell, Z, acc, info = load_paper_cell(cell_dir, tag)
    d = info["d"]
    if not 0 < S < d:
        raise ValueError(f"budget {S} outside (0, d = {d})")
    cell.require_spatial()

    torch = _torch()
    torch.set_num_threads(max(1, threads))
    torch.manual_seed(seed)

    split = model_split(info["n_models"])
    selection, validation, test = split
    print(f"=== {unit_name(method, mu, S)}: S={S} of d={d} OOD points, mu={mu:g}, k={knn} ===")
    print(f"  models {info['n_models']}: selection {len(selection)} | validation {len(validation)} | "
          f"test {len(test)}")
    print(f"  trials {trials}  max steps {max_steps}  patience {patience}  threads {threads}  "
          f"torch {torch.__version__}")

    # The graph is built for every method: OODSelect never sees it in its loss, but P_soft is logged.
    A = graph.build_adjacency(cell.lon, cell.lat, k=knn)
    L = graph.laplacian(A)

    # One seed per (seed, budget): every budget is searched independently, and the methods share the seed at a
    # given S, which pairs them.
    seed_eff = seed * 1_000_003 + S
    t0 = time.time()
    best, theta, records = run_search(Z, acc, split, S, n_trials=trials, max_epochs=max_steps,
                                      patience=patience, seed=seed_eff, L=L, mu=mu, knn=knn, reg=reg)
    secs = time.time() - t0

    r_train = hard_corr(Z[selection], acc[selection], theta, S)
    r_val = hard_corr(Z[validation], acc[validation], theta, S)
    r_test = hard_corr(Z[test], acc[test], theta, S)
    r_full_test = hard_corr(Z[test], acc[test], np.ones(d, np.float32), d, is_mask=True)
    lo, hi = fisher_ci(r_test, len(test))
    best_record = records[records.trial == best.number].iloc[0].to_dict()
    result = dict(unit=unit_name(method, mu, S), method=method, mu=mu, knn=knn, S=S, d=d,
                  n_sel=len(selection), n_val=len(validation), n_test=len(test),
                  r_train=r_train, r_val=r_val, r_test=r_test, r_test_ci_lo=lo, r_test_ci_hi=hi,
                  r_full_test=r_full_test, best_trial=best.number,
                  steps=int(best_record["steps"]), best_step=int(best_record["best_step"]),
                  soft_mass=best_record["soft_mass"], P_soft=best_record["P_soft"],
                  n_trials=len(records), n_failed=int((records.state == "failed").sum()),
                  trials=trials, max_steps=max_steps, patience=patience, budget_penalty="squared",
                  search_secs=secs, threads=threads, seed=seed, seed_eff=seed_eff,
                  torch=torch.__version__, hp={k: v for k, v in best.params.items()})

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "theta.npy", theta)
    records.to_csv(out / "trials.csv", index=False)
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out / "result.json")
    print(f"  best trial {best.number}: val {r_val:+.3f} | TEST r = {r_test:+.3f} [{lo:+.2f}, {hi:+.2f}] | "
          f"all OOD points {r_full_test:+.3f} | {secs / 60:.1f} min")
    print(f"  soft total {result['soft_mass']:.1f} (S = {S}), stopped at step {result['steps']}, "
          f"lambda {result['hp']['penalty']:.3g}")
    return result


def main() -> int:
    """The worker run_experiments.py starts for one search."""
    ap = argparse.ArgumentParser(description="One subset search (started by run_experiments.py).")
    ap.add_argument("--cell-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--S", required=True, type=int)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--knn", type=int, default=KNN)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    run_unit(a.cell_dir, a.tag, a.out, a.S, a.method, a.mu, knn=a.knn, trials=a.trials,
             max_steps=a.max_steps, patience=a.patience, seed=a.seed, threads=a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
