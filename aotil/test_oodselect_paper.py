#!/usr/bin/env python3
"""Checks for oodselect_paper.py. Run from the repo root:

    python aotil/test_oodselect_paper.py [--cells data/interim/aotil/cells --tag FL_band_AE_s0]

1. Equivalence with the authors' code. TestSetFinder from OODSELECT/src/OODSelect_utils.py is
   copied below (plus the one mu line, for the spatial case) and trained under a real PyTorch
   Lightning Trainer with EarlyStopping; oodselect_paper.train_one must reach the same stopping
   epoch, the same final validation r and the same top-S set from the same initial theta.
   Needs pytorch_lightning (pip install pytorch_lightning); skipped without it.
2. The scipy quadratic (_Quadratic) against torch.sparse, value and gradient.
3. Budgets are seeded independently (why: the seeding comment in oodselect_paper.main).
4. Correctness from the true-class probability matches argmax, ties included.
5. The two controls: make_split, null_accuracies, variant_name, run_controls.
6. The budget penalty (--budget-penalty): budget_term's value and its gradient against central
   differences; the wider log-uniform lambda range applies only under squared; squared pulls the
   soft total to S in train_one; main writes the choice into result.csv and meta.json.
7. Graph total variation (--reg gtv): on a 0/1 s it equals the Laplacian term and the cut-edge
   count; its gradient matches central differences where no two neighbours tie; a tied edge
   adds 0 to the gradient; mu = 0 gives the classic run exactly whatever --reg says; the default
   is the Laplacian run; main writes the choice and the spatial_gtv_ folder name.
Check 1, the slowest, runs last. Check 1 covers softplus only, the authors' term. Uses a real cell when --cells/--tag point at one; otherwise
builds a synthetic cell so the checks still run anywhere. Exit code 0 = every check passed.
"""
import argparse
import contextlib
import io
import json
import logging
import math
import sys
import tempfile
import warnings
from pathlib import Path
from unittest import mock

# Silenced before torch and Lightning are imported, since both warn on import.
warnings.filterwarnings("ignore")
for logger_name in ("pytorch_lightning", "lightning.pytorch", "lightning"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
import oodselect_paper as op  # noqa: E402
from cells import TREE_FAMILIES  # noqa: E402

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    _HAS_PL = True
except Exception:
    _HAS_PL = False

FAILS = []


def check(cond, msg):
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


# ------- from OODSELECT/src/OODSelect_utils.py, one statement per line; code unchanged -------
def probit_transform(x, eps=1e-6):
    x = torch.clamp(x, eps, 1-eps)
    return math.sqrt(2) * torch.erfinv(2 * x - 1)


class PearsonRLoss(nn.Module):
    def __init__(self, transform=probit_transform, eps=1e-6):
        super().__init__()
        self.transform = transform
        self.eps = eps

    def forward(self, X, y, z, eps=1e-6):
        z_sum = z.sum() + 1e-6
        x_z = (X @ z) / z_sum
        x_t = self.transform(x_z)
        y_t = self.transform(y)
        x_z_c = x_t - torch.mean(x_t)
        y_c = y_t - torch.mean(y_t)
        cov = torch.mean(x_z_c * y_c)
        x_var = torch.mean(x_z_c ** 2) + self.eps
        y_var = torch.mean(y_c ** 2) + self.eps
        corr = cov / torch.sqrt(x_var * y_var)
        if torch.isnan(corr):
            raise ValueError("Correlation is NaN")
        return corr


if _HAS_PL:
    class TestSetFinder(pl.LightningModule):
        def __init__(self, num_samples, num_OOD_samples, penalty=0., penalty_anneal_iters=1000,
                     lr=0.01, eta_min=0.01, T_max=100, L=None, mu=0.0, knn=10):
            super().__init__()
            self.selection = nn.Parameter(torch.randn(num_samples))
            self.loss_fn = PearsonRLoss()
            self.c = 1
            self.penalty = penalty
            self.penalty_anneal_iters = penalty_anneal_iters
            self.eta_min = eta_min
            self.lr = lr
            self.T_max = T_max
            self.num_OOD_samples = num_OOD_samples
            self.L = L                    # L, mu, knn: the one addition, for the spatial check
            self.mu = mu
            self.knn = knn

        def forward(self, X, y):
            return self.loss_fn(X, y, torch.sigmoid(self.selection))

        def training_step(self, batch, batch_idx):
            X, y = batch
            corr = self(X, y)
            progress = self.global_step / float(self.penalty_anneal_iters)
            progress = min(max(progress, 0.0), 1.0)
            warmup_ratio = 0.5 * (1.0 - math.cos(math.pi * progress))
            current_penalty = self.penalty * warmup_ratio
            loss = self.c*corr + current_penalty * nn.functional.softplus(
                self.num_OOD_samples - torch.sigmoid(self.selection).sum())
            if self.mu > 0:
                z = torch.sigmoid(self.selection)
                loss = loss + self.mu * (z @ (self.L @ z)) / (self.num_OOD_samples * self.knn)
            self.log('train_correlation', corr)
            return loss

        def validation_step(self, batch, batch_idx):
            X, y = batch
            selection = TestSetFinder.get_binary_selection(self.selection, self.num_OOD_samples)
            corr = self.loss_fn(X, y, selection).item()
            self.log('val_correlation', corr)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam([self.selection], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max,
                                                                   eta_min=self.eta_min)
            return [optimizer], [scheduler]

        @staticmethod
        def get_binary_selection(selection, num_OOD_samples):
            k = num_OOD_samples
            selection = selection.detach()
            topk_indices = torch.topk(selection, k).indices
            binary_mask = torch.zeros_like(selection)
            binary_mask[topk_indices] = 1.0
            return binary_mask
# --------------------------------------------------------------------------------------------


def synthetic_cell(d=1500, n_models=120, seed=0):
    """A cell with a planted inverted corner and spatial structure, written to a new temporary
    folder. Returns (cells_dir, tag)."""
    n_id = 400
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0, 1, n_models)
    lon = rng.uniform(-82, -81, d)
    lat = rng.uniform(27, 28, d)
    inverted = (lon < -81.8) & (lat < 27.2)        # better models do worse in this corner
    sign = np.where(inverted, -1, 1)
    base = rng.uniform(0.2, 0.8, d)
    noise = rng.normal(0, .15, (n_models, d))
    S_ood = np.clip(base[None, :] + 0.25 * sign[None, :] * (quality[:, None] - .5) + noise,
                    0.01, 0.99)
    y_ood = rng.integers(0, 2, d)
    id_noise = rng.normal(0, .2, (n_models, n_id))
    S_id = np.clip(0.55 + 0.35 * quality[:, None] + id_noise, 0.01, 0.99)
    y_id = rng.integers(0, 2, n_id)
    tiles = np.array([f"t{int((x + 82) * 3)}" for x in lon])

    tmp = Path(tempfile.mkdtemp())
    np.savez_compressed(tmp / "SYN.npz", S_ood=S_ood, S_id=S_id, y_ood=y_ood, y_id=y_id,
                        tiles_ood=tiles, lon=lon, lat=lat, block_id=np.arange(d) // 50,
                        tile_50km=tiles)
    kinds = np.array(["hgb", "lr", "mlp", "rf"])[np.arange(n_models) % 4]
    models = pd.DataFrame(dict(idx=np.arange(n_models), kind=kinds, score_id=S_id.mean(1)))
    models.to_csv(tmp / "SYN_models.csv", index=False)
    return str(tmp), "SYN"


def check_correctness():
    print("4. correctness from true-class probability")
    rng = np.random.default_rng(1)
    p = rng.choice([0.2, 0.5, 0.7], size=(5, 50))
    y = rng.integers(0, 2, 50)
    S = np.where(y == 1, p, 1 - p)
    pred = np.stack([1 - p, p], -1).argmax(-1)
    check(np.array_equal(op.correctness(S, y)[0], (pred == y).astype(np.float32)),
          "matches argmax([1-p, p]) == y, including p = 0.5")


def check_quadratic(L, d):
    print("2. scipy quadratic vs torch.sparse")
    quadratic = op._Quadratic(L)
    L_sparse = op.laplacian_torch(L)
    z1 = torch.rand(d, dtype=torch.float32, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    v1 = quadratic(z1)
    v1.backward()
    v2 = z2 @ (L_sparse @ z2)
    v2.backward()
    check(abs(v1.item() - v2.item()) / max(abs(v2.item()), 1) < 1e-4, "value")
    check(torch.allclose(z1.grad, z2.grad, rtol=1e-4, atol=1e-3), "gradient")


def check_budget_seeding(Z, acc, split):
    print("3. per-budget seeding")
    # The seeds are oodselect_paper.main's per-budget seed at --seed 0.
    trials_50 = op.run_search(Z, acc, split, 50, n_trials=2, max_epochs=40, patience=10,
                              seed=0 * 1_000_003 + 50, log_every=0)[2]
    trials_100 = op.run_search(Z, acc, split, 100, n_trials=2, max_epochs=40, patience=10,
                               seed=0 * 1_000_003 + 100, log_every=0)[2]
    check(not np.isclose(trials_50.lr.iloc[0], trials_100.lr.iloc[0]),
          "different budgets draw different trials")


def check_controls(cell, acc, n_models):
    print("5. the two controls")
    kind = cell.models.kind.to_numpy().astype(str)
    is_tree = np.isin(kind, sorted(TREE_FAMILIES))
    for rule, in_family in (("trees->others", is_tree), ("others->trees", ~is_tree)):
        sel, val, test = op.make_split(n_models, rule, kind, seed=0)
        sel_again, val_again, _ = op.make_split(n_models, rule, kind, seed=0)
        sel_seed1, _, _ = op.make_split(n_models, rule, kind, seed=1)
        n_family = int(in_family.sum())
        blocks_ok = (np.array_equal(test, np.where(~in_family)[0])
                     and np.array_equal(np.sort(np.r_[sel, val]), np.where(in_family)[0])
                     and len(np.intersect1d(sel, val)) == 0
                     and len(val) == int(round(op.FAMILY_VAL_FRAC * n_family)))
        family_name = "tree" if rule == "trees->others" else "other"
        check(blocks_ok, f"{rule}: test = the other family ({len(test)}), selection {len(sel)} + "
                         f"validation {len(val)} = the {n_family} {family_name} models")
        seeded = (np.array_equal(sel, sel_again) and np.array_equal(val, val_again)
                  and not np.array_equal(sel, sel_seed1))
        check(seeded, f"{rule}: the validation draw follows --seed")
    pairs = zip(op.make_split(n_models), op.model_split(n_models))
    check(all(np.array_equal(x, y) for x, y in pairs), "make_split('random') is model_split")

    split = op.model_split(n_models)
    shuffled = op.null_accuracies(acc, split, 1)
    sel_val = np.r_[split[0], split[1]]
    check(op.null_accuracies(acc, split, None) is acc, "null seed off: the accuracies themselves")
    null_ok = (np.array_equal(shuffled[split[2]], acc[split[2]])
               and np.array_equal(np.sort(shuffled[sel_val]), np.sort(acc[sel_val]))
               and not np.array_equal(shuffled, acc)
               and np.array_equal(shuffled, op.null_accuracies(acc, split, 1))
               and not np.array_equal(shuffled, op.null_accuracies(acc, split, 2)))
    check(null_ok, "null-select: selection + validation permuted (seeded), test untouched")

    names = [op.variant_name(0, 10), op.variant_name(3, 10),
             op.variant_name(0, 10, "trees->others"), op.variant_name(3, 10, "others->trees", 2),
             op.variant_name(0, 10, null_seed=1)]
    check(names == ["classic", "spatial_mu3_knn10", "classic_trees2others",
                    "spatial_mu3_knn10_others2trees_null2", "classic_null1"],
          f"variant names {names}")
    old_row = pd.Series(dict(seed=4))
    nan_row = pd.Series(dict(split=np.nan, null_seed=np.nan, seed=0))
    new_row = pd.Series(dict(split="trees->others", null_seed=3.0, seed=1))
    check(op.run_controls(old_row) == ("random", None, 4)
          and op.run_controls(nan_row) == ("random", None, 0)
          and op.run_controls(new_row) == ("trees->others", 3, 1),
          "run_controls reads old and new result.csv rows")


def check_budget_penalty(Z, acc, split, X_tr, y_tr, X_va, y_va, cells_dir, tag):
    print("6. the budget penalty")
    S = 500
    for total, want in ((510.0, {"softplus": math.log1p(math.exp(-10.0)), "squared": 100.0}),
                        (490.0, {"softplus": 10.0 + math.log1p(math.exp(-10.0)),
                                 "squared": 100.0})):
        got = {kind: op.budget_term(torch.tensor(total, dtype=torch.float64), S, kind).item()
               for kind in op.BUDGET_PENALTIES}
        check(all(abs(got[k] - want[k]) < 1e-9 for k in want),
              f"value at a soft total of {total:g} (S = {S}): {got}")
    try:
        op.budget_term(torch.tensor(1.0), S, "cubic")
        raised = False
    except ValueError:
        raised = True
    check(raised, "an unknown budget penalty raises ValueError")

    # The gradient by central differences, on a total away from S (the squared term's gradient
    # is -2 (S - total) on every weight).
    z0 = np.random.default_rng(3).uniform(0, 1, 40)
    S_small, h = 15, 1e-6
    for kind in op.BUDGET_PENALTIES:
        z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
        op.budget_term(z.sum(), S_small, kind).backward()

        def value(v):
            return op.budget_term(torch.tensor(v, dtype=torch.float64).sum(), S_small, kind).item()

        fd = np.array([(value(z0 + h * e) - value(z0 - h * e)) / (2 * h) for e in np.eye(len(z0))])
        check(np.allclose(z.grad.numpy(), fd, rtol=1e-5, atol=1e-6),
              f"{kind}: gradient matches central differences (first {z.grad[0].item():+.4f})")
    z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
    op.budget_term(z.sum(), S_small, "squared").backward()
    check(np.allclose(z.grad.numpy(), -2 * (S_small - z0.sum())),
          "squared: gradient is -2 (S - total) on every weight")

    # The lambda range Optuna draws from.
    check(op.PENALTY_RANGE == {"softplus": (0.1, 100.0, False), "squared": (1e-6, 100.0, True)},
          "PENALTY_RANGE: softplus keeps the released code's uniform [0.1, 100]; squared is "
          "log-uniform over [1e-6, 100]")
    for kind in op.BUDGET_PENALTIES:
        best, _, records = op.run_search(Z, acc, split, 50, n_trials=3, max_epochs=20,
                                         patience=5, seed=11, log_every=0, budget_penalty=kind)
        dist = best.distributions["penalty"]
        low, high, log = op.PENALTY_RANGE[kind]
        drawn = records.penalty.to_numpy()
        check((dist.low, dist.high, dist.log) == (low, high, log)
              and bool(np.all((drawn >= low) & (drawn <= high))),
              f"{kind}: lambda drawn from [{dist.low:g}, {dist.high:g}] log={dist.log}, "
              f"draws {np.round(drawn, 6).tolist()}")

    # train_one: squared pulls the soft total to S (here S = 100 of 1,500 synthetic points).
    hp = dict(lr=0.5, eta_min=0.01, T_max=50, penalty=0.01, penalty_anneal_iters=50)
    out = op.train_one(X_tr, y_tr, X_va, y_va, 100, hp, max_epochs=400, patience=400, seed=7,
                       budget_penalty="squared")
    check(abs(out["soft_mass"] - 100) <= 2,
          f"squared: soft total {out['soft_mass']:.1f} ends within 2 of S = 100")

    # main writes the choice; without the flag it is softplus.
    with tempfile.TemporaryDirectory() as tmp:
        for kind, extra in (("softplus", []), ("squared", ["--budget-penalty", "squared"])):
            out_dir = Path(tmp) / kind / "classic"
            argv = ["oodselect_paper.py", "--cells", cells_dir, "--tag", tag, "--out",
                    str(out_dir), "--budget", "50", "--trials", "2", "--max-epochs", "20",
                    "--patience", "5"] + extra
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                status = op.main()
            result = pd.read_csv(out_dir / f"{tag}_S50_result.csv").iloc[0]
            with open(out_dir / f"{tag}_S50_meta.json") as f:
                meta = json.load(f)
            check(status == 0 and result.budget_penalty == kind and meta["budget_penalty"] == kind,
                  f"main {' '.join(extra) or '(no flag)'}: result.csv and meta.json say {kind}")


def check_gtv(Z, acc, split, L, cell, cells_dir, tag):
    print("7. graph total variation (--reg gtv)")
    d = L.shape[0]
    gtv = op.spatial_term(L, "gtv")
    quad = op.spatial_term(L, "laplacian")
    check(isinstance(gtv, op._TotalVariation) and isinstance(quad, op._Quadratic)
          and op.spatial_term(None, "gtv") is None, "spatial_term picks the loss by reg")
    check(gtv.n_edges * 2 == int((L != 0).sum() - (L.diagonal() != 0).sum()),
          f"each undirected edge counted once ({gtv.n_edges} edges)")

    # On a 0/1 s, GTV = Laplacian = the cut edges (graph.coherence_metrics' P_hard * S k).
    A = graph.build_adjacency(cell.lon, cell.lat, k=10)
    rng = np.random.default_rng(5)
    for S in (min(50, d // 4), min(200, d // 2)):
        mask = np.zeros(d, bool)
        mask[rng.choice(d, S, replace=False)] = True
        s = torch.tensor(mask.astype(np.float32))
        cut = graph.coherence_metrics(mask, A, 10)["P_hard"] * S * 10
        v_gtv, v_quad = gtv(s).item(), quad(s).item()
        check(v_gtv == v_quad and abs(v_gtv - cut) < 1e-6,
              f"0/1 s with S = {S}: GTV {v_gtv:g} = Laplacian {v_quad:g} = cut edges {cut:g}")

    # The gradient by central differences where no two neighbours tie.
    sub = np.arange(min(60, d))
    L_small = graph.laplacian(graph.build_adjacency(cell.lon[sub], cell.lat[sub], k=4))
    gtv_small = op._TotalVariation(L_small)
    z0 = np.random.default_rng(6).uniform(0, 1, len(sub))
    gap = np.abs(z0[gtv_small.j.numpy()] - z0[gtv_small.j_prime.numpy()]).min()
    h = min(1e-6, gap / 10)
    z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
    gtv_small(z).backward()

    def value(v):
        return gtv_small(torch.tensor(v, dtype=torch.float64)).item()

    fd = np.array([(value(z0 + h * e) - value(z0 - h * e)) / (2 * h) for e in np.eye(len(z0))])
    check(gap > 0 and np.allclose(z.grad.numpy(), fd, rtol=1e-6, atol=1e-6),
          f"gradient matches central differences (smallest neighbour gap {gap:.1e})")

    # A tied edge adds 0: the path 0 - 1 - 2 with s = (0.5, 0.5, 0.2) has gradient (0, 1, -1).
    path = sp.csr_matrix(np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], float))
    z = torch.tensor([0.5, 0.5, 0.2], dtype=torch.float64, requires_grad=True)
    op._TotalVariation(path)(z).backward()
    check(z.grad.tolist() == [0.0, 1.0, -1.0],
          f"a tied edge adds 0 to the gradient: {z.grad.tolist()}")
    try:
        op._TotalVariation(sp.csr_matrix(np.array([[2.0, -2.0], [-2.0, 2.0]])))
        raised = False
    except ValueError:
        raised = True
    check(raised, "a weighted graph raises ValueError (the paper's edges weigh 1)")

    # mu = 0: the classic run exactly, whatever --reg says; the default is the Laplacian.
    S_run = min(60, d // 2)
    common = dict(n_trials=2, max_epochs=30, patience=8, seed=13, log_every=0,
                  budget_penalty="squared")
    runs = {}
    for name, kw in (("classic", dict(mu=0.0)), ("classic gtv", dict(mu=0.0, reg="gtv")),
                     ("spatial", dict(mu=3.0)), ("spatial laplacian", dict(mu=3.0, reg="laplacian")),
                     ("spatial gtv", dict(mu=3.0, reg="gtv"))):
        runs[name] = op.run_search(Z, acc, split, S_run, L=L, **common, **kw)
    same = (np.array_equal(runs["classic"][1], runs["classic gtv"][1])
            and runs["classic"][2].drop(columns=["secs", "P_soft"]).equals(
                runs["classic gtv"][2].drop(columns=["secs", "P_soft"])))
    check(same, "mu = 0 with --reg gtv: theta and every trial record equal the classic run's")
    check(np.array_equal(runs["spatial"][1], runs["spatial laplacian"][1]),
          "reg defaults to the Laplacian (theta bit-identical)")
    check(not np.array_equal(runs["spatial"][1], runs["spatial gtv"][1]),
          "mu = 3 with --reg gtv changes the run")
    out = op.train_one(torch.tensor(Z[split[0]]), torch.tensor(acc[split[0]], dtype=torch.float32),
                       torch.tensor(Z[split[1]]), torch.tensor(acc[split[1]], dtype=torch.float32),
                       S_run, dict(lr=0.5, eta_min=0.01, T_max=50, penalty=0.01,
                                penalty_anneal_iters=50),
                       max_epochs=40, patience=40, seed=3, L_t=gtv, mu=3.0, knn=10,
                       budget_penalty="squared")
    s_end = torch.sigmoid(torch.tensor(out["theta"]))
    check(abs(out["P_soft"] - gtv(s_end).item() / (S_run * 10)) < 1e-6,
          f"P_soft is R_GTV of the final soft weights ({out['P_soft']:.4f})")
    try:
        op.variant_name(3, 10, reg="tv")
        raised = False
    except ValueError:
        raised = True
    check(op.variant_name(3, 10, reg="gtv") == "spatial_gtv_mu3_knn10"
          and op.variant_name(3, 10, "random", 2, "gtv") == "spatial_gtv_mu3_knn10_null2"
          and op.variant_name(0, 10, reg="gtv") == "classic" and raised,
          "variant names under --reg gtv")

    # main writes the choice, and a GTV run must sit in its own folder.
    with tempfile.TemporaryDirectory() as tmp:
        base = ["oodselect_paper.py", "--cells", cells_dir, "--tag", tag, "--budget", "50",
                "--trials", "2", "--max-epochs", "20", "--patience", "5", "--mu", "3"]
        written = {}
        for reg, folder in (("laplacian", "spatial_mu3_knn10"), ("gtv", "spatial_gtv_mu3_knn10")):
            out_dir = Path(tmp) / folder
            argv = base + ["--out", str(out_dir)] + ([] if reg == "laplacian" else ["--reg", reg])
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                status = op.main()
            result = pd.read_csv(out_dir / f"{tag}_S50_result.csv").iloc[0]
            with open(out_dir / f"{tag}_S50_meta.json") as f:
                meta = json.load(f)
            written[reg] = (status, result.reg, meta["reg"], result.variant)
        check(written == {"laplacian": (0, "laplacian", "laplacian", "spatial_mu3_knn10"),
                          "gtv": (0, "gtv", "gtv", "spatial_gtv_mu3_knn10")},
              f"main writes reg and the variant: {written}")
        argv = base + ["--out", str(Path(tmp) / "spatial_mu3_knn10"), "--reg", "gtv"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            try:
                op.main()
                refused = False
            except SystemExit:
                refused = True
        check(refused, "a GTV run into the Laplacian folder is refused")


def check_lightning(X_tr, y_tr, X_va, y_va, L, d):
    print("1. equivalence with the authors' TestSetFinder under Lightning")
    if not _HAS_PL:
        print("  skip  pytorch_lightning not installed")
        return
    from torch.utils.data import DataLoader, TensorDataset
    L_sparse = op.laplacian_torch(L)
    seed = 7
    batch_size = len(X_tr) + len(X_va)      # one batch holds every model
    # (S, hyperparameters, max epochs, patience, mu)
    cases = [
        (min(500, d // 3),
         dict(lr=1.5, eta_min=0.14, T_max=125, penalty=40.0, penalty_anneal_iters=300),
         300, 500, 0.0),
        (min(100, d // 10),
         dict(lr=0.03, eta_min=0.002, T_max=30, penalty=5.0, penalty_anneal_iters=50),
         1200, 60, 0.0),
        (min(500, d // 3),
         dict(lr=0.5, eta_min=0.01, T_max=80, penalty=20.0, penalty_anneal_iters=200),
         500, 100, 3.0),
    ]
    for S, hp, max_epochs, patience, mu in cases:
        theta0 = torch.randn(d, generator=torch.Generator().manual_seed(seed))
        model = TestSetFinder(d, S, L=L_sparse, mu=mu, **hp)
        with torch.no_grad():
            model.selection.copy_(theta0)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
        early_stopping = EarlyStopping(monitor="val_correlation", patience=patience, mode="min")
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stopping], logger=False,
                             enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, accelerator="cpu")
        trainer.fit(model, train_loader, val_loader)
        ref = model.selection.detach().numpy()
        mine = op.train_one(X_tr, y_tr, X_va, y_va, S, hp, max_epochs=max_epochs,
                            patience=patience, seed=seed, L_t=op._Quadratic(L), mu=mu, knn=10)
        top_ref = set(np.argsort(-ref)[:S])
        top_mine = set(np.argsort(-mine["theta"])[:S])
        overlap = len(top_ref & top_mine)
        val_ref = float(trainer.callback_metrics["val_correlation"])
        same = (trainer.current_epoch == mine["epochs"]
                and abs(val_ref - mine["val_last"]) < 1e-4
                and overlap >= S - 2)
        check(same, f"S={S} mu={mu}: epochs {trainer.current_epoch}/{mine['epochs']}, "
                    f"val {val_ref:+.5f}/{mine['val_last']:+.5f}, top-S overlap {overlap}/{S}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells")
    ap.add_argument("--tag")
    args = ap.parse_args()
    if args.cells and args.tag and (Path(args.cells) / f"{args.tag}.npz").exists():
        cells_dir, tag = args.cells, args.tag
    else:
        cells_dir, tag = synthetic_cell()
        print("(using a synthetic cell; pass --cells/--tag for a real one)")
    cell, Z, acc, info = op.load_paper_cell(cells_dir, tag)
    split = op.model_split(info["n_models"])
    selection, validation, _ = split
    d = Z.shape[1]
    X_tr = torch.tensor(Z[selection])
    y_tr = torch.tensor(acc[selection], dtype=torch.float32)
    X_va = torch.tensor(Z[validation])
    y_va = torch.tensor(acc[validation], dtype=torch.float32)
    L = graph.laplacian(graph.build_adjacency(cell.lon, cell.lat, k=10))

    check_correctness()
    check_quadratic(L, d)
    check_budget_seeding(Z, acc, split)
    check_controls(cell, acc, info["n_models"])
    check_budget_penalty(Z, acc, split, X_tr, y_tr, X_va, y_va, cells_dir, tag)
    check_gtv(Z, acc, split, L, cell, cells_dir, tag)
    check_lightning(X_tr, y_tr, X_va, y_va, L, d)

    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
