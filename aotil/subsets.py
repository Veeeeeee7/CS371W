#!/usr/bin/env python3
"""A selected subset of the OOD points, evaluated (Table tab:search, Figures fig:budgets and fig:mu), and the post-hoc
smoothing of Equation eq:smooth. numpy and scipy only (no torch).

  corr        Pearson correlation of probit accuracies (Equation eq:corr), accuracies clipped to [1e-6, 1 - 1e-6],
              each variance regularized by 1e-6 as the search's loss does; corr_te on the 58 test models with a
              Fisher 95% interval, the validation correlation on the 58 validation models
  smoothing   s_bar = (I + D)^-1 (I + A) s on the search's graph A, applied to the final weights s = sigmoid(theta)
              of the search, keeping the S points with the largest s_bar. It is computed on the complement
              q = sigmoid(-theta), keeping the S smallest smoothed q (stable sort), so that weights that round to
              1.0 in floating point do not tie.
  coherence   components (>= c = 20 points), the points they hold, purity (graph.coherence), and the test tiles
              the subset touches
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtri
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph  # noqa: E402
from search import correctness, model_split  # noqa: E402,F401  (torch-free helpers)

EPS = 1e-6


def cell_arrays(z) -> dict:
    """The arrays every subset is scored with, from a cell's npz: Z (models x OOD points, 0/1 float64), the ID
    accuracies (float64), the selection / validation / test models, the tiles and the coordinates."""
    Z = correctness(z["S_ood"], z["y_ood"])[0].astype(np.float64)
    acc = correctness(z["S_id"], z["y_id"])[0].astype(np.float64).mean(1)
    sel, val, test = model_split(Z.shape[0])
    return dict(Z=Z, acc=acc, sel=sel, val=val, test=test, tiles=np.asarray(z["tile_50km"]).astype(str),
                lon=np.asarray(z["lon"], float), lat=np.asarray(z["lat"], float), y=np.asarray(z["y_ood"]))


def probit_r(Z, acc, mask) -> float:
    """Population Pearson r of probit(subset accuracy) on probit(ID accuracy), as the search's loss computes it."""
    w = np.asarray(mask, np.float64)
    sub = (Z @ w) / (w.sum() + 1e-6)
    x = ndtri(np.clip(sub, EPS, 1 - EPS))
    y = ndtri(np.clip(acc, EPS, 1 - EPS))
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.mean(xc * yc) / np.sqrt((np.mean(xc ** 2) + EPS) * (np.mean(yc ** 2) + EPS)))


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher's 95% interval of a correlation over n models."""
    z, se, q = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def top_mask(score, k: int) -> np.ndarray:
    """The k highest scores; ties go to the lower index (stable sort), as the search takes its subset."""
    m = np.zeros(len(score), bool)
    m[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return m


def bottom_mask(score, k: int) -> np.ndarray:
    """The k lowest scores; ties go to the lower index."""
    m = np.zeros(len(score), bool)
    m[np.argsort(np.asarray(score), kind="stable")[:k]] = True
    return m


def smooth(theta, A, S: int) -> tuple[np.ndarray, int]:
    """Equation eq:smooth on graph A: the S largest (I + D)^-1 (I + A) sigmoid(theta), computed as the S smallest
    smoothed sigmoid(-theta). Returns (mask, the number of points tied at the cut)."""
    q = expit(-np.asarray(theta, np.float64))
    deg = np.asarray(A.sum(axis=1)).ravel()
    q_bar = (q + A @ q) / (1.0 + deg)
    mask = bottom_mask(q_bar, S)
    cut = np.sort(q_bar)[S - 1]
    return mask, int((q_bar == cut).sum())


def evaluate(mask, arrays: dict, A, c: int = graph.C_MIN) -> dict:
    """corr_te [95% CI] on the test models, the validation correlation, coherence and tiles of one subset."""
    mask = np.asarray(mask, bool)
    Z, acc, test, val = arrays["Z"], arrays["acc"], arrays["test"], arrays["val"]
    r_te = probit_r(Z[test], acc[test], mask)
    lo, hi = fisher_ci(r_te, len(test))
    out = dict(n=int(mask.sum()), r_te=r_te, ci_lo=lo, ci_hi=hi, r_val=probit_r(Z[val], acc[val], mask))
    out.update(graph.coherence(mask, A, c))
    out["tiles"] = int(len(np.unique(arrays["tiles"][mask])))
    return out
