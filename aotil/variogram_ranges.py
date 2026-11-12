#!/usr/bin/env python3
"""
How far each model feature stays spatially correlated: the effective range of its semivariogram,
for the Split paragraph's "at most XX km" (docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md Part C4).

In CS terms: the semivariogram gamma(h) is half the mean squared difference of a feature between
two points h apart. Nearby points have similar values, so gamma starts low and rises with h until
the points no longer inform each other, where it levels off at the sill. The range is the distance
where it levels off; tiles wider than every feature's range keep a test tile's points from being
near-copies of training points (Roberts et al. 2017; Valavi et al. 2019, blockCV).

Method, as blockCV fits its ranges: a random 20,000-point subsample of the dataset (seed 0); for
each of the six numeric features and each one-hot column (zoo.model_inputs) with at least 1%
prevalence, the empirical semivariogram in 1 km lags out to 200 km (pairs with both values present;
straight-line distance in EPSG:5070 metres); a spherical and an exponential model, each with a
nugget, fitted by weighted least squares with gstat's default weights N_j / h_j^2 (pairs in lag j
over the squared lag distance); the fit with the smaller weighted residual is kept. The effective
range is the distance where the fitted model's rise above its nugget reaches 95% of its partial
sill: a ln 20 (about 3a) for the exponential model, 0.811a for the spherical one, a being the
fitted range parameter. A fit whose partial sill is under 1% of its total sill has no spatial
structure (a pure nugget: neighbours are no more alike than any two points) and gets range 0; a
fit whose range parameter sits at its upper bound is still rising at 200 km, and its range is
marked as beyond the lags. A categorical feature's range is the largest over its columns.

    .venv/bin/python aotil/variogram_ranges.py --data data/processed_v4 \\
        --out results/$D/variogram
Writes ranges.csv (one row per column, with both fits), features.csv (one row per feature, its
range and the maximum), empirical.csv (gamma and the pair count per lag and column).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zoo  # noqa: E402

N_SAMPLE = 20_000
LAG_M = 1_000.0
MAX_LAG_M = 200_000.0
MIN_PREVALENCE = 0.01
SILL_SHARE = 0.95
MIN_STRUCTURE = 0.01            # partial sill under this share of the total sill: no structure
CHUNK = 400                     # rows of the distance matrix per step


# ------------------------------------------------------------------ models
def spherical(h, nugget, psill, a):
    """gamma(h) of the spherical model: nugget + psill (1.5 h/a - 0.5 (h/a)^3) up to a."""
    x = np.minimum(np.asarray(h, float) / a, 1.0)
    return nugget + psill * (1.5 * x - 0.5 * x ** 3)


def exponential(h, nugget, psill, a):
    """gamma(h) of the exponential model: nugget + psill (1 - exp(-h / a))."""
    return nugget + psill * (1.0 - np.exp(-np.asarray(h, float) / a))


MODELS = {"spherical": spherical, "exponential": exponential}


def effective_range(model: str, a: float, share: float = SILL_SHARE) -> float:
    """The distance where the model's rise above its nugget reaches share of its partial sill."""
    if model == "exponential":
        return float(a * np.log(1.0 / (1.0 - share)))
    if model == "spherical":
        return float(a * brentq(lambda x: 1.5 * x - 0.5 * x ** 3 - share, 0.0, 1.0))
    raise ValueError(model)


# ------------------------------------------------------------------ empirical
def empirical(xy: np.ndarray, Z: np.ndarray, lag: float = LAG_M,
              max_lag: float = MAX_LAG_M, chunk: int = CHUNK):
    """(lag centres, gamma (lags x columns), pairs (lags x columns)): every pair of points once,
    by straight-line distance, binned [k lag, (k + 1) lag) up to max_lag; a pair counts for a
    column when both values are present. gamma = sum (z_i - z_j)^2 / (2 N)."""
    n_lags = int(round(max_lag / lag))
    n, m = Z.shape
    sums = np.zeros((n_lags, m))
    counts = np.zeros((n_lags, m))
    present = np.isfinite(Z)
    Zf = np.where(present, Z, 0.0)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        d = cdist(xy[rows], xy[start:])            # pairs (i, j) with j >= i, each once below
        i_loc, j_loc = np.nonzero(d < max_lag)
        j = j_loc + start
        i = rows[i_loc]
        keep = j > i
        i, j, dist = i[keep], j[keep], d[i_loc[keep], j_loc[keep]]
        bins = np.minimum((dist // lag).astype(np.int64), n_lags - 1)
        both = present[i] & present[j]
        sq = (Zf[i] - Zf[j]) ** 2
        for c in range(m):
            w = both[:, c]
            sums[:, c] += np.bincount(bins[w], weights=sq[w, c], minlength=n_lags)
            counts[:, c] += np.bincount(bins[w], minlength=n_lags)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = sums / (2.0 * counts)
    centres = (np.arange(n_lags) + 0.5) * lag
    return centres, gamma, counts


# ------------------------------------------------------------------ fits
def fit(model: str, h: np.ndarray, gamma: np.ndarray, pairs: np.ndarray) -> dict:
    """Weighted least squares of one model on the lags with pairs: weights pairs / h^2 (gstat's
    fit.method 7), nugget and partial sill >= 0, range a in [lag / 10, 10 max lag]."""
    ok = (pairs > 0) & np.isfinite(gamma)
    h, g, n = h[ok], gamma[ok], pairs[ok]
    w = np.sqrt(n) / h
    f = MODELS[model]
    top = float(np.max(g))
    start = [max(float(g[0]), 0.0), max(top - float(g[0]), top * 0.5, 1e-12), float(h[-1]) / 3]
    lo = [0.0, 0.0, LAG_M / 10]
    hi = [max(top, 1e-12) * 10, max(top, 1e-12) * 10, MAX_LAG_M * 10]
    start = [min(max(s, l), u) for s, l, u in zip(start, lo, hi)]
    res = least_squares(lambda p: w * (f(h, *p) - g), start, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=20_000)
    nugget, psill, a = (float(v) for v in res.x)
    note, rng = "", effective_range(model, a)
    if psill < MIN_STRUCTURE * (nugget + psill):
        note, rng = "pure nugget (no spatial structure)", 0.0
    elif a >= 0.99 * hi[2] or rng > MAX_LAG_M:
        note = "beyond the lags (still rising at the largest lag)"
    return dict(model=model, nugget=nugget, psill=psill, a_m=a, wsse=float(np.sum(res.fun ** 2)),
                effective_range_m=rng, converged=bool(res.success), note=note)


def best_fit(h, gamma, pairs) -> tuple[dict, dict]:
    """(the better of the two fits by weighted residual, both fits by model)."""
    fits = {model: fit(model, h, gamma, pairs) for model in MODELS}
    best = min(fits.values(), key=lambda r: r["wsse"])
    return best, fits


# ------------------------------------------------------------------ the dataset
def columns_for(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """The variogram columns: the numeric features, and the one-hot columns (zoo.model_inputs)
    with at least MIN_PREVALENCE ones among the rows that have the feature. Returns (the columns
    as floats, column -> feature)."""
    full, inputs = zoo.model_inputs(df, feats)
    out, owner = {}, {}
    for f in feats:
        if f not in zoo.CATEGORICAL:
            out[f] = full[f].to_numpy(float)
            owner[f] = f
            continue
        for c in [c for c in inputs if c.startswith(f + "_")]:
            values = full[c].to_numpy(float)
            if np.nanmean(values) >= MIN_PREVALENCE:
                out[c] = values
                owner[c] = f
    return pd.DataFrame(out, index=df.index), owner


def run(data_dir: Path, n_sample: int = N_SAMPLE, seed: int = 0, state: str = "FL") -> dict:
    df = gpd.read_parquet(data_dir / f"dataset_{state}.parquet")
    if df.crs is None or df.crs.to_epsg() != 5070:
        raise SystemExit("the dataset's geometry must be EPSG:5070 metres")
    feats = [f for f in zoo.read_features(data_dir / "FEATURES.txt") if f in df.columns]
    take = np.sort(np.random.default_rng(seed).choice(len(df), min(n_sample, len(df)),
                                                      replace=False))
    sub = df.iloc[take].reset_index(drop=True)
    xy = np.column_stack([sub.geometry.x.to_numpy(), sub.geometry.y.to_numpy()])
    cols, owner = columns_for(sub, feats)
    h, gamma, pairs = empirical(xy, cols.to_numpy(float))
    rows, emp = [], []
    for k, c in enumerate(cols.columns):
        best, fits = best_fit(h, gamma[:, k], pairs[:, k])
        rows.append(dict(column=c, feature=owner[c], n_points=int(np.isfinite(cols[c]).sum()),
                         prevalence=(float(np.nanmean(cols[c])) if c != owner[c] else np.nan),
                         best=best["model"], range_km=best["effective_range_m"] / 1000,
                         note=best["note"],
                         **{f"{m}_{key}": fits[m][key] for m in MODELS
                            for key in ("nugget", "psill", "a_m", "wsse", "effective_range_m",
                                        "converged")}))
        emp.append(pd.DataFrame(dict(column=c, h_km=h / 1000, gamma=gamma[:, k],
                                     pairs=pairs[:, k].astype(np.int64))))
    ranges = pd.DataFrame(rows)
    features = (ranges.groupby("feature", sort=False)
                .agg(range_km=("range_km", "max"), columns=("column", "count"),
                     widest=("column", lambda s: s.iloc[int(np.argmax(
                         ranges.loc[s.index, "range_km"]))]))
                .reset_index())
    return dict(ranges=ranges, features=features, empirical=pd.concat(emp, ignore_index=True),
                summary=dict(points=len(df), sample=len(sub), seed=seed, lag_km=LAG_M / 1000,
                             max_lag_km=MAX_LAG_M / 1000, min_prevalence=MIN_PREVALENCE,
                             sill_share=SILL_SHARE, max_range_km=float(features.range_km.max()),
                             max_feature=str(features.feature[features.range_km.idxmax()])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="folder with FEATURES.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = run(a.data, a.n_sample, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    out["ranges"].to_csv(a.out / "ranges.csv", index=False)
    out["features"].to_csv(a.out / "features.csv", index=False)
    out["empirical"].to_csv(a.out / "empirical.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(out["summary"], indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out["ranges"][["column", "feature", "n_points", "prevalence", "best", "range_km",
                             "spherical_effective_range_m", "exponential_effective_range_m",
                             "note"]].round(3).to_string(index=False))
        print()
        print(out["features"].round(2).to_string(index=False))
    s = out["summary"]
    print(f"\nlargest range: {s['max_range_km']:.1f} km ({s['max_feature']}); "
          f"{'ABOVE' if s['max_range_km'] > 50 else 'within'} the 50 km tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
