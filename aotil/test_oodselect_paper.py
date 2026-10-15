#!/usr/bin/env python3
"""
Checks for oodselect_paper.py. Run from the repo root:

    python aotil/test_oodselect_paper.py [--cells data/interim/aotil/cells --tag FL_band_AE_s0]

1. EQUIVALENCE with the authors' code. TestSetFinder from
   OODSELECT/src/OODSelect_utils.py is copied verbatim below (plus the one mu
   line, for the spatial case) and trained under a real PyTorch Lightning
   Trainer with EarlyStopping; oodselect_paper.train_one must reach the same
   stopping epoch, the same final validation r and the same top-S set from the
   same initial theta. Needs pytorch_lightning (pip install pytorch_lightning);
   skipped without it.
2. The scipy quadratic (_Quadratic) against torch.sparse, value and gradient.
3. Budgets are seeded independently (the nesting bug this guards against is
   described in oodselect_paper.main).
4. Correctness from true-class probability matches argmax, ties included.

Uses a real cell when --cells/--tag point at one; otherwise builds a synthetic
cell so the checks still run anywhere.
"""
import argparse, math, sys, tempfile, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
for n in ("pytorch_lightning", "lightning.pytorch", "lightning"):
    logging.getLogger(n).setLevel(logging.ERROR)
import numpy as np, pandas as pd, torch, torch.nn as nn
import oodselect_paper as op, graph

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    _HAS_PL = True
except Exception:
    _HAS_PL = False
    pl = None

FAILS = []
def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)

# ---------------- verbatim from OODSELECT/src/OODSelect_utils.py ----------------
def probit_transform(x, eps=1e-6):
    x = torch.clamp(x, eps, 1-eps)
    return math.sqrt(2) * torch.erfinv(2 * x - 1)

class PearsonRLoss(nn.Module):
    def __init__(self, transform=probit_transform, eps=1e-6):
        super().__init__(); self.transform = transform; self.eps = eps
    def forward(self, X, y, z, eps=1e-6):
        z_sum = z.sum() + 1e-6
        x_z = (X @ z) / z_sum
        x_t = self.transform(x_z); y_t = self.transform(y)
        x_z_c = x_t - torch.mean(x_t); y_c = y_t - torch.mean(y_t)
        cov = torch.mean(x_z_c * y_c)
        x_var = torch.mean(x_z_c ** 2) + self.eps; y_var = torch.mean(y_c ** 2) + self.eps
        corr = cov / torch.sqrt(x_var * y_var)
        if torch.isnan(corr): raise ValueError("Correlation is NaN")
        return corr

if _HAS_PL:
  class TestSetFinder(pl.LightningModule):
      def __init__(self, num_samples, num_OOD_samples, penalty=0., penalty_anneal_iters=1000,
                   lr=0.01, eta_min=0.01, T_max=100, L=None, mu=0.0, knn=10):
          super().__init__()
          self.selection = nn.Parameter(torch.randn(num_samples))
          self.loss_fn = PearsonRLoss(); self.c = 1
          self.penalty = penalty; self.penalty_anneal_iters = penalty_anneal_iters
          self.eta_min = eta_min; self.lr = lr; self.T_max = T_max
          self.num_OOD_samples = num_OOD_samples
          self.L = L; self.mu = mu; self.knn = knn       # the one addition, for the spatial check
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
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
          return [optimizer], [scheduler]
      @staticmethod
      def get_binary_selection(selection, num_OOD_samples):
          k = num_OOD_samples; selection = selection.detach()
          topk_indices = torch.topk(selection, k).indices
          binary_mask = torch.zeros_like(selection); binary_mask[topk_indices] = 1.0
          return binary_mask
# ---------------------------------------------------------------------------------



def synthetic_cell(d=1500, n_models=120, seed=0):
    """Scores with a planted inverse subset and spatial structure."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0, 1, n_models)                       # model quality
    lon = rng.uniform(-82, -81, d); lat = rng.uniform(27, 28, d)
    inv = (lon < -81.8) & (lat < 27.2)                    # a corner inverts
    base = rng.uniform(0.2, 0.8, d)
    p = np.clip(base[None, :] + 0.25 * np.where(inv, -1, 1)[None, :] * (q[:, None] - .5)
                + rng.normal(0, .15, (n_models, d)), 0.01, 0.99)
    y_ood = rng.integers(0, 2, d)
    S_ood = p
    S_id = np.clip(0.55 + 0.35 * q[:, None] + rng.normal(0, .2, (n_models, 400)), 0.01, 0.99)
    y_id = rng.integers(0, 2, 400)
    tmp = Path(tempfile.mkdtemp())
    np.savez_compressed(tmp / "SYN.npz", S_ood=S_ood, S_id=S_id, y_ood=y_ood, y_id=y_id,
                        tiles_ood=np.array([f"t{int((a+82)*3)}" for a in lon]),
                        lon=lon, lat=lat, block_id=np.arange(d) // 50,
                        tile_50km=np.array([f"t{int((a+82)*3)}" for a in lon]))
    kinds = np.array(["hgb", "lr", "mlp", "rf"])[np.arange(n_models) % 4]
    pd.DataFrame(dict(idx=np.arange(n_models), kind=kinds,
                      score_id=S_id.mean(1))).to_csv(tmp / "SYN_models.csv", index=False)
    return str(tmp), "SYN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells"); ap.add_argument("--tag")
    a = ap.parse_args()
    if a.cells and a.tag and (Path(a.cells) / f"{a.tag}.npz").exists():
        cells, tag = a.cells, a.tag
    else:
        cells, tag = synthetic_cell()
        print(f"(using a synthetic cell; pass --cells/--tag for a real one)")
    cell, Z, acc, info = op.load_paper_cell(cells, tag)
    tr, va, te = op.model_split(info["n_models"])
    d = Z.shape[1]
    Xtr, ytr = torch.tensor(Z[tr]), torch.tensor(acc[tr], dtype=torch.float32)
    Xva, yva = torch.tensor(Z[va]), torch.tensor(acc[va], dtype=torch.float32)
    A = graph.build_adjacency(cell.lon, cell.lat, k=10)
    L = graph.laplacian(A)

    print("4. correctness from true-class probability")
    rng = np.random.default_rng(1)
    p1 = rng.choice([0.2, 0.5, 0.7], size=(5, 50)); yy = rng.integers(0, 2, 50)
    S = np.where(yy == 1, p1, 1 - p1)
    pred = np.stack([1 - p1, p1], -1).argmax(-1)
    check(np.array_equal(op.correctness(S, yy)[0], (pred == yy).astype(np.float32)),
          "matches argmax([1-p, p]) == y, including p = 0.5")

    print("2. scipy quadratic vs torch.sparse")
    Q = op._Quadratic(L); Lt = op.laplacian_torch(L)
    z1 = torch.rand(d, dtype=torch.float32, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    v1 = Q(z1); v1.backward(); v2 = z2 @ (Lt @ z2); v2.backward()
    check(abs(v1.item() - v2.item()) / max(abs(v2.item()), 1) < 1e-4, "value")
    check(torch.allclose(z1.grad, z2.grad, rtol=1e-4, atol=1e-3), "gradient")

    print("3. per-budget seeding")
    t1 = op.run_search(Z, acc, (tr, va, te), 50, n_trials=2, max_epochs=40, patience=10,
                       seed=0 * 1_000_003 + 50, log_every=0)[2]
    t2 = op.run_search(Z, acc, (tr, va, te), 100, n_trials=2, max_epochs=40, patience=10,
                       seed=0 * 1_000_003 + 100, log_every=0)[2]
    check(not np.isclose(t1.lr.iloc[0], t2.lr.iloc[0]), "different budgets draw different trials")

    print("1. equivalence with the authors' TestSetFinder under Lightning")
    if not _HAS_PL:
        print("  skip  pytorch_lightning not installed")
    else:
        from torch.utils.data import TensorDataset, DataLoader
        cases = [(min(500, d // 3), dict(lr=1.5, eta_min=0.14, T_max=125, penalty=40.0, penalty_anneal_iters=300), 300, 500, 0.0),
                 (min(100, d // 10), dict(lr=0.03, eta_min=0.002, T_max=30, penalty=5.0, penalty_anneal_iters=50), 1200, 60, 0.0),
                 (min(500, d // 3), dict(lr=0.5, eta_min=0.01, T_max=80, penalty=20.0, penalty_anneal_iters=200), 500, 100, 3.0)]
        for S, hp, E, P, mu in cases:
            seed = 7
            theta0 = torch.randn(d, generator=torch.Generator().manual_seed(seed))
            m = TestSetFinder(d, S, L=Lt, mu=mu, **hp)
            with torch.no_grad():
                m.selection.copy_(theta0)
            bs = len(tr) + len(va)
            tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
            vl = DataLoader(TensorDataset(Xva, yva), batch_size=bs, shuffle=False)
            trn = pl.Trainer(max_epochs=E, callbacks=[EarlyStopping(monitor="val_correlation", patience=P, mode="min")],
                             logger=False, enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, accelerator="cpu")
            trn.fit(m, tl, vl)
            ref = m.selection.detach().numpy()
            mine = op.train_one(Xtr, ytr, Xva, yva, S, hp, max_epochs=E, patience=P, seed=seed,
                                L_t=op._Quadratic(L), mu=mu, knn=10)
            top_ref = set(np.argsort(-ref)[:S]); top_mine = set(np.argsort(-mine["theta"])[:S])
            vr = float(trn.callback_metrics["val_correlation"])
            check(trn.current_epoch == mine["epochs"] and abs(vr - mine["val_last"]) < 1e-4
                  and len(top_ref & top_mine) >= S - 2,
                  f"S={S} mu={mu}: epochs {trn.current_epoch}/{mine['epochs']}, "
                  f"val {vr:+.5f}/{mine['val_last']:+.5f}, top-S overlap {len(top_ref & top_mine)}/{S}")

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
