#!/usr/bin/env python3
"""
OODSelect as released -- Salaudeen et al., "Aggregation Hides Out-of-Distribution
Generalization Failures from Spurious Correlations" (NeurIPS 2025,
arXiv:2510.24884), reimplemented line-for-line from the authors' code at
github.com/olawalesalaudeen/OODSELECT (src/OODSelect_utils.py,
scripts/OODSelect.py, src/OODSelect_experiments.py) -- plus ONE optional term,
the spatial smoothness penalty, which is the only difference between this
project's two first-experiment runs.

One invocation = one (cell, budget S, variant). The submission scripts run one
Slurm array task per budget, which is how the authors launch it too
(launch_OODSelect.py: one job per num_OOD_samples).

WHAT IS TAKEN FROM THE RELEASED CODE, AND WHERE
-----------------------------------------------
Scores (extract_data)
    Z[i, j] = 1 if model i's argmax prediction on OOD point j is correct.
    ID performance = plain accuracy on held-out ID data. Binary, not the
    predicted-probability score aotil/oodselect.py uses.
Model split (prepare_dataset)
    train_test_split(test_size=0.2, random_state=42), then
    train_test_split(test_size=0.25, random_state=42) -> 60/20/20
    selection / validation / test models.
Loss (PearsonRLoss)
    a = Z_sel @ z / (sum z + 1e-6), z = sigmoid(theta)
    corr = cov(probit(a), probit(acc_ID)) / sqrt((var+1e-6)(var+1e-6)),
    probit with clamp eps = 1e-6 on BOTH sides, population moments.
Budget term (TestSetFinder.training_step)
    penalty * 0.5 * (1 - cos(pi * min(step / penalty_anneal_iters, 1)))
            * softplus(S - sum z)
    NOTE this is one-sided and cosine-warmed, not the (S - ||s||_1)^2 of the
    paper's Eq. 5. The code is what produced the paper's numbers, so the code
    wins.
Optimiser (configure_optimizers)
    Adam(lr), CosineAnnealingLR(T_max, eta_min) stepped once per epoch. One
    epoch = one full-batch step over all selection models.
Initialisation
    theta ~ N(0, 1).
Validation (validation_step, get_binary_selection)
    Every epoch, hard top-S mask on the validation models, same loss.
    EarlyStopping(monitor=val_correlation, patience=500, mode=min).
Hyperparameter search (tune_hparams, run_optuna)
    Optuna, minimise the validation correlation, over
        lr                    log-uniform [1e-2, 1e2]
        eta_min               log-uniform [1e-3, 1e0]
        T_max                 int [10, 200]
        penalty               uniform [0.1, 100]
        penalty_anneal_iters  int [10, 1000]
    N_TRIALS = 100 and num_epochs = 50,000 (OODSelect_experiments.py).
Selection kept (run_optuna, OODSelect.py)
    The BEST TRIAL's soft selection -- which is the parameter vector at the
    epoch training STOPPED, not the best-validation epoch (the ModelCheckpoint
    callback saves the best epoch but the code never reads it back). Replicated.
Test (TestSetFinder.test)
    Top-S mask of that vector, same loss, on the held-out test models.
Budget grid (OODSelect_experiments.py)
    10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, ... truncated at the
    OOD set size, exactly as the authors do per dataset.

DEVIATIONS, EACH FORCED
-----------------------
1. No PyTorch Lightning. The Trainer is replaced by the loop it runs (one
   full-batch step, then validation, then the scheduler step, then the early-
   stopping check). Lightning's per-step overhead would cost ~10x in wall clock
   at 50,000 epochs x 100 trials x 11 budgets and changes no arithmetic.
2. Seeds. The released code seeds neither theta nor the Optuna sampler. Both
   are seeded here so a rerun reproduces.
3. A trial whose correlation goes NaN is recorded as failed and the search
   continues; the released code raises and kills the whole job.
4. Ties at exactly p = 0.5 follow numpy argmax (class 0 wins), which needs the
   labels. Cells written before y_id was stored fall back to "correct iff
   S > 0.5", which differs only for ID negatives predicted at exactly 0.5; the
   count is printed.

THE ONE ADDITION (--mu > 0)
---------------------------
    + mu * z' L z / (S * k)
L is the combinatorial Laplacian of the symmetric kNN graph on the projected
OOD point locations (graph.py), so z'Lz counts cut edges and dividing by S*k
makes it an incoherence score in [0, ~1] -- the same normalisation as the
aotil/ sweep, so mu = 3 means what it meant in job 562664. It is added to the
training loss only. Validation, Optuna's objective, the kept selection and the
test are unchanged, so the two variants differ in exactly one line. mu is
fixed, never tuned: the penalty can only cost correlation, so tuning it on
validation correlation would drive it to zero.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import load_cell  # noqa: E402

PAPER_BUDGETS = [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500,
                 10000, 15000, 20000, 25000, 50000]
N_TRIALS = 100
NUM_EPOCHS = 50_000
PATIENCE = 500
EPS = 1e-6


# ------------------------------------------------------------------ data
def correctness(S: np.ndarray, y: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Binary correctness from the stored true-class probability.

    S = p if y == 1 else 1 - p. argmax over [1-p, p] picks class 0 on a tie, so
    a y=1 point is correct iff p > 0.5 (S > 0.5) and a y=0 point iff p <= 0.5
    (S >= 0.5). Returns (Z, n_ambiguous_ties).
    """
    S = np.asarray(S, float)
    if y is None:
        return (S > 0.5).astype(np.float32), int((S == 0.5).sum())
    y = np.asarray(y)
    Z = (S > 0.5) | ((S == 0.5) & (y[None, :] == 0))
    return Z.astype(np.float32), 0


def load_paper_cell(cells_dir: str | Path, tag: str):
    """The cell as the paper sees it: binary Z on OOD points, ID accuracy."""
    cell = load_cell(cells_dir, tag)
    npz = np.load(Path(cells_dir) / f"{tag}.npz", allow_pickle=True)
    y_id = npz["y_id"] if "y_id" in npz.files else None
    Z_ood, _ = correctness(cell.S_ood, cell.y_ood)
    Z_id, ties = correctness(cell.S_id, y_id)
    acc_id = Z_id.mean(axis=1)
    info = dict(n_models=int(Z_ood.shape[0]), d=int(Z_ood.shape[1]),
                y_id_stored=y_id is not None, id_ties_unresolved=ties)
    return cell, Z_ood, acc_id, info


def model_split(n_models: int):
    """prepare_dataset(): 60/20/20 by two sklearn splits at random_state=42.

    Splitting np.arange(N) gives the same row permutation as splitting (X, y),
    because train_test_split shuffles by index.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(n_models)
    tmp, te = train_test_split(idx, test_size=0.2, random_state=42)
    tr, va = train_test_split(tmp, test_size=0.25, random_state=42)
    return np.asarray(tr), np.asarray(va), np.asarray(te)


# ------------------------------------------------------------------ loss
def _torch():
    import torch
    return torch


def probit_t(x, eps: float = EPS):
    torch = _torch()
    x = torch.clamp(x, eps, 1 - eps)
    return math.sqrt(2) * torch.erfinv(2 * x - 1)


def pearson_r_loss(X, y, z, eps: float = EPS):
    """PearsonRLoss.forward, verbatim arithmetic."""
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


def binary_selection(theta, k: int):
    """get_binary_selection: 1 on the top-k entries of the raw parameter."""
    torch = _torch()
    idx = torch.topk(theta.detach(), k).indices
    m = torch.zeros_like(theta)
    m[idx] = 1.0
    return m


def hard_corr(Z, acc, theta_or_mask, k: int, is_mask: bool = False) -> float:
    """TestSetFinder.test on numpy inputs."""
    torch = _torch()
    X = torch.as_tensor(np.asarray(Z, np.float32))
    y = torch.as_tensor(np.asarray(acc, np.float32))
    v = torch.as_tensor(np.asarray(theta_or_mask, np.float32))
    m = v if is_mask else binary_selection(v, k)
    return float(pearson_r_loss(X, y, m).item())


class _Quadratic:
    """z' L z for a fixed symmetric sparse L, with d/dz = 2 L z.

    A scipy mat-vec inside a torch autograd Function. torch's own CPU sparse
    mat-vec plus its backward costs ~10 ms here against ~0.2 ms for scipy, and
    the spatial run does one per epoch for up to 50,000 epochs x 100 trials.
    Same arithmetic; test_oodselect_paper checks it against torch.sparse.
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


def laplacian_torch(L):
    """scipy CSR -> torch sparse CSR (float32). Reference path for tests."""
    torch = _torch()
    import scipy.sparse as sp
    Lc = sp.csr_matrix(L).astype(np.float32)
    return torch.sparse_csr_tensor(torch.as_tensor(Lc.indptr, dtype=torch.int64),
                                   torch.as_tensor(Lc.indices, dtype=torch.int64),
                                   torch.as_tensor(Lc.data), size=Lc.shape)


class TrialFailed(RuntimeError):
    pass


def train_one(X_tr, y_tr, X_va, y_va, S: int, hp: dict, *, max_epochs: int,
              patience: int, seed: int, L_t=None, mu: float = 0.0, knn: int = 10):
    """One Lightning fit of TestSetFinder, as the loop Lightning runs.

    Per epoch: training_step on the single full batch -> optimizer.step ->
    validation on the hard top-S mask -> EarlyStopping check ->
    CosineAnnealingLR.step. Returns the parameter vector at the stopping
    epoch and the LAST validation correlation (what tune_hparams returns).
    L_t is a callable z -> z'Lz (a _Quadratic) or a torch sparse matrix; it
    enters the loss only when mu > 0.
    """
    torch = _torch()
    quad = None
    if L_t is not None:
        quad = L_t if callable(L_t) else (lambda z: z @ (L_t @ z))
    d = X_tr.shape[1]
    g = torch.Generator().manual_seed(int(seed))
    theta = torch.randn(d, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=hp["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["T_max"],
                                                     eta_min=hp["eta_min"])
    denom_P = float(max(S * knn, 1))
    best, wait, best_epoch = math.inf, 0, -1
    val = math.nan
    epoch = 0
    for epoch in range(max_epochs):
        z = torch.sigmoid(theta)
        corr = pearson_r_loss(X_tr, y_tr, z)
        if torch.isnan(corr):
            raise TrialFailed(f"correlation NaN at epoch {epoch}")
        progress = min(max(epoch / float(hp["penalty_anneal_iters"]), 0.0), 1.0)
        warm = 0.5 * (1.0 - math.cos(math.pi * progress))
        loss = corr + hp["penalty"] * warm * torch.nn.functional.softplus(S - z.sum())
        if mu > 0 and quad is not None:
            loss = loss + mu * quad(z) / denom_P
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
        P_soft = float((quad(z) / denom_P).item()) if quad is not None else math.nan
    return dict(theta=theta_np, val_last=val, val_best=best, best_epoch=best_epoch,
                epochs=epoch + 1, soft_mass=soft_mass, P_soft=P_soft)


# ------------------------------------------------------------------ search
def run_search(Z, acc, split, S: int, *, n_trials: int, max_epochs: int,
               patience: int, seed: int, L=None, mu: float = 0.0, knn: int = 10,
               log_every: int = 10):
    import optuna
    torch = _torch()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va, _ = split
    X_tr = torch.as_tensor(Z[tr]); y_tr = torch.as_tensor(acc[tr].astype(np.float32))
    X_va = torch.as_tensor(Z[va]); y_va = torch.as_tensor(acc[va].astype(np.float32))
    L_t = _Quadratic(L) if L is not None else None   # P_soft is logged for both variants

    records, thetas = [], {}
    t_start = time.time()

    def objective(trial):
        hp = dict(
            lr=trial.suggest_float("lr", 1e-2, 1e2, log=True),
            eta_min=trial.suggest_float("eta_min", 1e-3, 1e0, log=True),
            T_max=trial.suggest_int("T_max", 10, 200),
            penalty=trial.suggest_float("penalty", 1e-1, 1e2),
            penalty_anneal_iters=trial.suggest_int("penalty_anneal_iters", 10, 1000),
        )
        t0 = time.time()
        try:
            out = train_one(X_tr, y_tr, X_va, y_va, S, hp, max_epochs=max_epochs,
                            patience=patience, seed=seed * 100_003 + trial.number,
                            L_t=L_t, mu=mu, knn=knn)
        except TrialFailed as e:
            records.append(dict(trial=trial.number, state="failed", note=str(e),
                                secs=time.time() - t0, **hp))
            raise optuna.TrialPruned(str(e))
        thetas[trial.number] = out.pop("theta")
        records.append(dict(trial=trial.number, state="complete",
                            secs=time.time() - t0, **hp, **out))
        if log_every and (trial.number % log_every == 0 or trial.number == n_trials - 1):
            done = [r for r in records if r["state"] == "complete"]
            bv = min(r["val_last"] for r in done) if done else math.nan
            print(f"    trial {trial.number:3d}  val {out['val_last']:+.3f}  "
                  f"(best {bv:+.3f})  epochs {out['epochs']:6d}  "
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
    """The authors' grid, truncated below the OOD set size (S = d is the
    full set, which needs no optimisation and is reported by the summary)."""
    return [b for b in PAPER_BUDGETS if b < d]


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--tag", required=True, help="cell tag, e.g. FL_band_AE_s0")
    ap.add_argument("--out", help="variant output directory (required unless --list-budgets)")
    ap.add_argument("--budget", type=int, help="S, the number of OOD points to select")
    ap.add_argument("--budget-index", type=int,
                    help="index into the paper grid (Slurm array task id)")
    ap.add_argument("--list-budgets", action="store_true")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="spatial smoothness weight; 0 = the paper exactly")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cell, Z, acc, info = load_paper_cell(a.cells, a.tag)
    d = info["d"]
    grid = budgets_for(d)
    if a.list_budgets:
        print(" ".join(map(str, grid)))
        return 0
    if not a.out:
        ap.error("--out is required")
    if a.budget is None and a.budget_index is None:
        ap.error("--budget or --budget-index is required")
    if a.budget is None:
        if a.budget_index >= len(PAPER_BUDGETS):
            print(f"budget index {a.budget_index} is past the grid; nothing to do")
            return 0
        a.budget = PAPER_BUDGETS[a.budget_index]
    S = int(a.budget)
    if S >= d:
        # scripts/OODSelect.py exits when num_OOD_samples > X.shape[1]; S == d
        # is the full set.
        print(f"{a.tag}: budget {S} >= d = {d}; nothing to select, skipping")
        return 0

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.tag}_S{S}"
    if Path(f"{stem}_result.csv").exists() and not a.overwrite:
        print(f"{stem}_result.csv exists; skipping (--overwrite to redo)")
        return 0

    torch = _torch()
    torch.set_num_threads(max(1, a.threads))
    torch.manual_seed(a.seed)

    split = model_split(info["n_models"])
    tr, va, te = split
    variant = "classic" if a.mu == 0 else f"spatial_mu{a.mu:g}_knn{a.knn}"
    print(f"=== OODSelect (paper protocol) {a.tag}  S={S}  d={d}  variant={variant} ===")
    print(f"  models {info['n_models']}: selection {len(tr)} | validation {len(va)} | test {len(te)}")
    print(f"  ID accuracy {acc.min():.3f}-{acc.max():.3f} | OOD accuracy "
          f"{Z.mean(1).min():.3f}-{Z.mean(1).max():.3f}")
    print(f"  trials {a.trials}  max epochs {a.max_epochs}  patience {a.patience}  "
          f"threads {a.threads}  torch {torch.__version__}  device cpu")
    if not info["y_id_stored"]:
        print(f"  note: cell predates y_id; ID ties at p=0.5 counted incorrect "
              f"({info['id_ties_unresolved']} such entries)")

    # The graph is built for BOTH variants: the classic run uses it only to
    # measure how coherent its (unconstrained) subset happens to be.
    A = L = None
    if cell.has_spatial():
        import graph
        A = graph.build_adjacency(cell.lon, cell.lat, k=a.knn)
        L = graph.laplacian(A)
    elif a.mu > 0:
        cell.require_spatial()

    # One seed per (seed, budget). The paper selects every S INDEPENDENTLY and
    # then offers cross-budget overlap (Eq. 6) as evidence of consistency. With
    # a shared seed, every budget would get the same Optuna draws and the same
    # theta init -- and because the one-sided budget term is ~0 once sum(z) > S,
    # the trajectories barely depend on S, so the budgets would come out nested
    # by construction. The two variants DO share the seed at a given S: that
    # pairs them, which is what the comparison wants.
    seed_eff = a.seed * 1_000_003 + S
    t0 = time.time()
    best, theta, trials = run_search(Z, acc, split, S, n_trials=a.trials,
                                     max_epochs=a.max_epochs, patience=a.patience,
                                     seed=seed_eff, L=L, mu=a.mu, knn=a.knn)
    secs = time.time() - t0

    mask = np.zeros(d, bool)
    mask[np.argsort(-theta, kind="stable")[:S]] = True
    r = {part: hard_corr(Z[idx], acc[idx], theta, S)
         for part, idx in (("train", tr), ("val", va), ("test", te))}
    r_full_test = hard_corr(Z[te], acc[te], np.ones(d, np.float32), d, is_mask=True)
    st = plain_stats(Z[te], acc[te], mask)
    lo, hi = fisher_ci(r["test"], len(te))
    brow = trials[trials.trial == best.number].iloc[0].to_dict()

    row = dict(cell=a.tag, variant=variant, mu=a.mu, knn=a.knn, budget=S, d=d,
               frac=S / d, n_sel=len(tr), n_val=len(va), n_test=len(te),
               r_train=r["train"], r_val=r["val"], r_test=r["test"],
               r_test_ci_lo=lo, r_test_ci_hi=hi,
               pearson_test_scipy=st["pearson"], p_test=st["p"],
               spearman_test=st["spearman"], r_full_test=r_full_test,
               aotil=bool(r["test"] <= -0.3),
               best_trial=best.number, epochs=int(brow.get("epochs", -1)),
               best_epoch=int(brow.get("best_epoch", -1)),
               soft_mass=brow.get("soft_mass", np.nan), P_soft=brow.get("P_soft", np.nan),
               n_trials=len(trials), n_failed=int((trials.state == "failed").sum()),
               search_secs=secs, threads=a.threads, seed=a.seed, seed_eff=seed_eff,
               **{f"hp_{k}": v for k, v in best.params.items()})
    if A is not None:
        import graph
        cm = graph.coherence_metrics(mask, A, a.knn, tiles=cell.tile_50km,
                                     blocks=cell.block_id)
        row.update({f"coh_{k}": v for k, v in cm.items() if k not in ("k", "frac")})

    np.save(f"{stem}_theta.npy", theta)
    trials.to_csv(f"{stem}_trials.csv", index=False)
    pd.DataFrame([row]).to_csv(f"{stem}_result.csv", index=False)
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(dict(info, variant=variant, budget=S, split=dict(
            selection=tr.tolist(), validation=va.tolist(), test=te.tolist())), f)

    print(f"  best trial {best.number}: val {r['val']:+.3f} | TEST r = {r['test']:+.3f} "
          f"[{lo:+.2f}, {hi:+.2f}]  rho {st['spearman']:+.3f} | full OOD set "
          f"{r_full_test:+.3f} | {secs / 60:.1f} min")
    if A is not None:
        print(f"  coherence: purity {row['coh_purity']:.2f}  components {row['coh_n_cc']}  "
              f"largest {row['coh_largest_cc_frac']:.2f}  tiles {row['coh_tiles']}")
    print(f"  wrote {stem}_result.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
