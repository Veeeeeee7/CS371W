#!/usr/bin/env python3
"""
o5_feature_inclusion.py -- which feature carries the inversion? (docs/TASK-O5-REGIONS.md, Part D1)

zoo.py gave every model a random subset of the 7 features, so "model i saw feature f" is assigned
like a treatment in a randomized trial. Per seed, response and model set this fits, by OLS,
    probit(acc_R) ~ 1 + probit(acc_ID) + family + sum_f saw_f
where acc_R is a model's accuracy on the response's points (the seed's regions, pooled and one by
one, and the spatial S = 500 selection) and acc_ID its accuracy on the Zone AE test half. A
negative saw_f means models that saw f sit further below the line on those points at equal Zone AE
accuracy and family, which is what a spurious feature (Salaudeen et al. 2025) predicts.

Reads   <cells-dir>/<cell>_s<seed>.npz and _models.csv, the selection's theta under <sel-root>,
        the regions' points under <regions-dir>, and <dataset> (land cover by point_id)
Writes  <out>           one row per (seed, response, model_set, formula, term)
        <out>_seen.csv  how often each feature was seen, per seed and model set
        <out>_corr.csv  correlations between the feature indicators and n_feat

    .venv/bin/python aotil/o5_feature_inclusion.py --cells-dir data/interim/aotil/cells_w7 \\
        --sel-root results/2026-10-16/oodselect_paper --regions-dir results/2026-10-16/regions \\
        --dataset data/processed/dataset_FL.parquet \\
        --out results/2026-10-17/o5/D1_feature_inclusion.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op     # noqa: E402

WATER = 11                       # NLCD class 11, open water; every other class counts as land
# "id+family+features" is the brief's fit (primary). "family+features" drops probit(acc_ID): a
# spurious feature raises acc_ID and lowers acc_R, so conditioning on acc_ID absorbs part of its
# effect, and the second formula shows the feature's total effect on the response.
FORMULAS = ("id+family+features", "family+features")
# Every model_set of the output: model_sets()'s two and the _noclamp fits fit_seed adds to them.
MODEL_SETS = ("test60", "all300", "test60_noclamp", "all300_noclamp")
# The responses every seed has, so they can be averaged over seeds (regions differ by seed).
MEAN_RESPONSES = ("pooled_regions", "spatial_S500_all", "spatial_S500_land")


# ------------------------------------------------------------------ inputs
def read_features(path: Path) -> list[str]:
    """Feature names of a FEATURES.txt, comments and blank lines skipped."""
    names = []
    for line in Path(path).read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names


def load_landcover(path: Path) -> pd.Series:
    """The dataset's NLCD land cover, indexed by point_id as text."""
    lc = pd.read_parquet(path, columns=["point_id", "landcover"])
    lc = lc.assign(point_id=lc.point_id.astype(str))
    return lc.set_index("point_id").landcover


def indicators(feature_lists, feats: list[str]) -> np.ndarray:
    """(n_models, len(feats)) 0/1 matrix: model i saw feats[j]. Each entry of feature_lists is a
    model's ';'-joined `features` from models.csv. An unknown name is refused, so a renamed
    column cannot silently read as 'not seen'."""
    column_of = {f: j for j, f in enumerate(feats)}
    I = np.zeros((len(feature_lists), len(feats)), int)
    for i, entry in enumerate(feature_lists):
        for name in str(entry).split(";"):
            name = name.strip()
            if not name:
                continue
            if name not in column_of:
                raise SystemExit(f"model {i}: feature {name!r} is not in the feature list {feats}")
            I[i, column_of[name]] = 1
    return I


def probit(a) -> np.ndarray:
    """The selections' probit: clamp to [EPS, 1 - EPS], then the inverse normal CDF."""
    return norm.ppf(np.clip(np.asarray(a, float), op.EPS, 1 - op.EPS))


def at_clamp(acc: np.ndarray) -> np.ndarray:
    """True where an accuracy is 0 or 1, i.e. at the probit clamp (|probit| = 4.75)."""
    return (acc <= op.EPS) | (acc >= 1 - op.EPS)


def top_mask(theta: np.ndarray, S: int) -> np.ndarray:
    """The selection: the S largest theta, stable argsort (regions.py's rule)."""
    mask = np.zeros(len(theta), bool)
    mask[np.argsort(-np.asarray(theta), kind="stable")[:S]] = True
    return mask


def model_sets(n_models: int) -> dict:
    """test60: the paper's 60 test models, which took no part in choosing the regions (primary).
    all300: every model; partly circular, since the other 240 drove the search on these points."""
    _, _, test = op.model_split(n_models)
    return {"test60": np.sort(test), "all300": np.arange(n_models)}


# ------------------------------------------------------------------ design and fit
def design(acc_id, kinds, I, feats, ref: str, with_id: bool):
    """Design matrix and term names: intercept, [probit_acc_id], family_<k> for every family but
    the reference ref, saw_<f> for every feature."""
    kinds = np.asarray(kinds).astype(str)
    families = sorted(set(kinds.tolist()))
    if ref not in families:
        raise SystemExit(f"reference family {ref!r} has no model in this set")
    columns = [np.ones(len(kinds))]
    terms = ["intercept"]
    if with_id:
        columns.append(probit(acc_id))
        terms.append("probit_acc_id")
    for family in families:
        if family != ref:
            columns.append((kinds == family).astype(float))
            terms.append(f"family_{family}")
    for j, f in enumerate(feats):
        columns.append(I[:, j].astype(float))
        terms.append(f"saw_{f}")
    return np.column_stack(columns), terms


def ols(X: np.ndarray, Y: np.ndarray):
    """OLS of every column of Y on X. Returns (coef p x R, se p x R, r2 of length R, residual df),
    se being the classical sqrt(diag((X'X)^-1) * RSS / (n - p)). Raises LinAlgError when X is
    rank-deficient, as a coefficient would not be identified."""
    n, p = X.shape
    if np.linalg.matrix_rank(X) < p:
        raise np.linalg.LinAlgError("rank-deficient design")
    Y = Y.reshape(n, -1)
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    resid = Y - X @ B
    rss = (resid ** 2).sum(0)
    df = n - p
    se = np.sqrt(np.outer(np.diag(XtX_inv), rss / df))
    tss = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):    # a constant response has tss = 0
        r2 = 1 - rss / tss
    return B, se, r2, df


def vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column: 1 / (1 - R2) of the column regressed on the
    others. NaN for the intercept (column 0), inf for a column the others fit exactly."""
    out = np.full(X.shape[1], np.nan)
    for j in range(1, X.shape[1]):
        target = X[:, j]
        others = np.delete(X, j, axis=1)
        coef = np.linalg.lstsq(others, target, rcond=None)[0]
        resid = target - others @ coef
        rss = resid @ resid
        tss = ((target - target.mean()) ** 2).sum()
        if rss == 0:
            out[j] = np.inf
        else:
            out[j] = tss / rss
    return out


def boot_draws(X: np.ndarray, Y: np.ndarray, idx: np.ndarray):
    """Coefficients refitted on each row of idx (a resample of the models): n_boot x p x R.
    A resample whose design is rank-deficient (a family absent, or a feature seen by all or none
    of the drawn models) cannot identify every coefficient and stays NaN."""
    Y = Y.reshape(len(Y), -1)
    n_params = X.shape[1]
    draws = np.full((len(idx), n_params, Y.shape[1]), np.nan)
    for b, rows in enumerate(idx):
        Xb = X[rows]
        if np.linalg.matrix_rank(Xb) < n_params:
            continue
        draws[b] = np.linalg.lstsq(Xb, Y[rows], rcond=None)[0]
    return draws


# ------------------------------------------------------------------ one seed
def land_mask(point_id: np.ndarray, lc: pd.Series, tag: str) -> np.ndarray:
    """True for the cell's OOD points that are not open water."""
    if pd.Index(point_id).has_duplicates:
        raise SystemExit(f"{tag}: duplicate point_id among the OOD points")
    missing = ~pd.Index(point_id).isin(lc.index)
    if missing.any():
        raise SystemExit(f"{tag}: {missing.sum()} OOD points not in the dataset")
    return lc.reindex(point_id).to_numpy() != WATER     # a missing land cover counts as land


def selection_mask(a, tag: str, n_points: int) -> np.ndarray:
    """The spatial selection: the top a.budget points by the stored theta."""
    theta = np.load(Path(a.sel_root) / a.variant / f"{tag}_S{a.budget}_theta.npy")
    if theta.shape != (n_points,):
        raise SystemExit(f"{tag}: theta has shape {theta.shape}, the cell has {n_points} points")
    return top_mask(theta, a.budget)


def region_points(a, tag: str, point_id: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    """The regions file's points with their column in the cell (cell_column), checked to be
    exactly the selection, keeping the components of at least a.min_size points."""
    pts = pd.read_parquet(Path(a.regions_dir) / f"{tag}_{a.variant}_S{a.budget}_points.parquet")
    column_of = pd.Series(np.arange(len(point_id)), index=point_id)
    pts_id = pts.point_id.astype(str)
    if not pts_id.isin(column_of.index).all():
        raise SystemExit(f"{tag}: region points not in the cell")
    columns = column_of.reindex(pts_id).to_numpy()
    selected_columns = set(np.where(selected)[0].tolist())
    if len(columns) != a.budget or set(columns.tolist()) != selected_columns:
        raise SystemExit(f"{tag}: the regions file's points are not the top-{a.budget} theta")
    pts = pts.assign(cell_column=columns)
    return pts[pts.component_size >= a.min_size]


def count_row(seed, response: str, mask: np.ndarray, land: np.ndarray) -> dict:
    """A response's points before and after the land rule."""
    n = int(mask.sum())
    n_land = int((mask & land).sum())
    return dict(seed=seed, response=response, n=n, n_land=n_land, dropped_water=n - n_land)


def seed_inputs(a, seed: int, lc: pd.Series, feats: list[str]):
    """What the fits of one seed need: Z (models x OOD points, 1 = correct), acc_id, the models'
    kinds, indicators I and n_feat, the response masks and their point counts.

    Responses: pooled_regions (the land points of all regions, a region being a component of at
    least a.min_size points of the selection), region_<k> (one region's land points; none for a
    region without land), spatial_S500_all (the whole selection) and spatial_S500_land. The
    names keep "S500", the routine's budget, whatever a.budget is.
    """
    tag = f"{a.cell_prefix}_s{seed}"
    cell, Z, acc_id, _ = op.load_paper_cell(a.cells_dir, tag)
    models = cell.models
    n_models, n_points = Z.shape
    if len(models) != n_models or not (models.idx.to_numpy() == np.arange(n_models)).all():
        raise SystemExit(f"{tag}: models.csv rows are not models 0..{n_models - 1} in order")
    point_id = np.asarray(cell.point_id).astype(str)
    land = land_mask(point_id, lc, tag)
    selected = selection_mask(a, tag, n_points)
    regions = region_points(a, tag, point_id, selected)

    pooled = np.zeros(n_points, bool)
    region_masks, region_counts = {}, []
    for k, group in regions.groupby("component"):
        in_region = np.zeros(n_points, bool)
        in_region[group.cell_column.to_numpy()] = True
        pooled |= in_region
        row = count_row(seed, f"region_{k}", in_region, land)
        region_counts.append(row)
        if row["n_land"] > 0:
            region_masks[f"region_{k}"] = in_region & land

    masks = {"pooled_regions": pooled & land}
    masks.update(region_masks)
    masks["spatial_S500_all"] = selected
    masks["spatial_S500_land"] = selected & land

    n_selected = int(selected.sum())
    counts = [count_row(seed, "pooled_regions", pooled, land)]
    counts += region_counts
    # spatial_S500_all keeps its water points, so the land rule drops nothing from it
    counts.append(dict(seed=seed, response="spatial_S500_all", n=n_selected, n_land=n_selected,
                       dropped_water=0))
    counts.append(count_row(seed, "spatial_S500_land", selected, land))

    I = indicators(models.features.tolist(), feats)
    return dict(tag=tag, Z=Z, acc_id=acc_id, kinds=models.kind.astype(str).to_numpy(), I=I,
                n_feat=models.n_feat.to_numpy(), masks=masks, counts=counts)


def fit_block(si: dict, feats, a, seed: int, model_set: str, models: np.ndarray,
              acc_r: np.ndarray, names: list[str], rng) -> tuple[list[dict], dict]:
    """Both formulas on one model set, for every response in names (the columns of acc_r,
    models x responses, already restricted to `models`).

    se is the classical OLS standard error; boot_lo/hi the percentile 95% interval over a.n_boot
    resamples of the models, shared by every response and formula here so that rows are
    comparable; n_boot_used counts the resamples that were not rank-deficient; vif is the term's
    variance inflation within the design. Returns (rows, draws) with
    draws[(model_set, formula, response)] = (terms, n_boot x p draws).
    """
    rows, draws = [], {}
    idx = rng.integers(0, len(models), size=(a.n_boot, len(models)))
    Y = probit(acc_r)
    n_clamped = at_clamp(acc_r).sum(0)
    for formula in FORMULAS:
        X, terms = design(si["acc_id"][models], si["kinds"][models], si["I"][models], feats,
                          a.ref_family, with_id=(formula == "id+family+features"))
        try:
            B, SE, R2, df = ols(X, Y)
        except np.linalg.LinAlgError:
            print(f"[D1] seed {seed} {model_set} {formula} {names}: design rank-deficient on "
                  f"{len(models)} models (a family or a feature constant); no rows", flush=True)
            continue
        D = boot_draws(X, Y, idx)
        n_boot_used = int((~np.isnan(D[:, 0, 0])).sum())
        lo = np.nanpercentile(D, 2.5, axis=0)
        hi = np.nanpercentile(D, 97.5, axis=0)
        V = vif(X)
        for r, response in enumerate(names):
            draws[(model_set, formula, response)] = (terms, D[:, :, r])
            for j, term in enumerate(terms):
                t_value = B[j, r] / SE[j, r] if SE[j, r] > 0 else np.nan
                rows.append(dict(
                    seed=seed, response=response, model_set=model_set, formula=formula, term=term,
                    coef=B[j, r], se=SE[j, r], boot_lo=lo[j, r], boot_hi=hi[j, r],
                    n_models=len(models), p_ols=2 * t_dist.sf(abs(t_value), df),
                    vif=V[j], r2=R2[r], resid_df=df, n_boot_used=n_boot_used,
                    n_points=int(si["masks"][response].sum()), n_clamped=int(n_clamped[r])))
    return rows, draws


def fit_seed(si: dict, feats, a, seed: int):
    """All fits of one seed. Returns (rows, draws), draws keyed (model_set, formula, response).

    Besides test60 and all300, each response gets a <set>_noclamp fit without the models whose
    accuracy on it is exactly 0 or 1. Such a model sits at the probit clamp, far from the rest,
    so one or two can decide a fit of 60 models (on an all-positive region, a weak model that
    calls every point flood is 100% correct). The brief fixes the clamp; these rows show its
    effect.
    """
    names = list(si["masks"])
    acc_r = np.column_stack([si["Z"][:, si["masks"][r]].mean(1) for r in names])
    clamped = at_clamp(acc_r)                                   # models x responses
    rows, draws = [], {}
    # Each block draws from its own RNG stream keyed by (boot seed, seed, model set[, response]).
    for code, (model_set, models) in enumerate(model_sets(si["Z"].shape[0]).items()):
        rng = np.random.default_rng([a.boot_seed, seed, code])
        block_rows, block_draws = fit_block(si, feats, a, seed, model_set, models, acc_r[models],
                                            names, rng)
        rows += block_rows
        draws.update(block_draws)
        for r, response in enumerate(names):
            keep = models[~clamped[models, r]]
            rng = np.random.default_rng([a.boot_seed, seed, code, 1, r])
            block_rows, block_draws = fit_block(si, feats, a, seed, f"{model_set}_noclamp", keep,
                                                acc_r[keep][:, [r]], [response], rng)
            rows += block_rows
            draws.update(block_draws)
    return rows, draws


def clamped_models(si: dict, seed: int) -> list[dict]:
    """The models whose accuracy on a response is exactly 0 or 1, one row each (for the log)."""
    test60 = model_sets(si["Z"].shape[0])["test60"]
    out = []
    for response, mask in si["masks"].items():
        acc_r = si["Z"][:, mask].mean(1)
        for i in np.where(at_clamp(acc_r))[0]:
            out.append(dict(seed=seed, response=response, model=int(i), kind=si["kinds"][i],
                            in_test60=bool(np.isin(i, test60)), acc_R=acc_r[i],
                            acc_id=si["acc_id"][i], n_feat=int(si["n_feat"][i])))
    return out


def mean_rows(all_draws: dict, rows: pd.DataFrame, seeds, model_set: str, formula: str,
              response: str) -> list[dict]:
    """The seed-averaged rows of one fit; none when a seed lacks it or the seeds' terms differ
    (a family absent in one seed)."""
    keys = [(s, model_set, formula, response) for s in seeds]
    if not all(k in all_draws for k in keys):
        return []
    terms = all_draws[keys[0]][0]
    if any(all_draws[k][0] != terms for k in keys):
        return []
    stack = np.stack([all_draws[k][1] for k in keys])           # seeds x n_boot x p
    kept = ~np.isnan(stack[:, :, 0]).any(0)                     # draws kept in every seed
    mean_draws = stack[:, kept].mean(0)
    in_fit = ((rows.model_set == model_set) & (rows.formula == formula)
              & (rows.response == response) & rows.seed.isin(seeds))
    fit_rows = rows[in_fit]
    out = []
    for j, term in enumerate(terms):
        per_seed = fit_rows[fit_rows.term == term]
        out.append(dict(
            seed="mean3", response=response, model_set=model_set, formula=formula, term=term,
            coef=per_seed.coef.mean(), se=np.sqrt((per_seed.se ** 2).sum()) / len(per_seed),
            boot_lo=np.percentile(mean_draws[:, j], 2.5),
            boot_hi=np.percentile(mean_draws[:, j], 97.5),
            n_models=int(per_seed.n_models.sum()), n_boot_used=int(kept.sum()),
            n_points=int(per_seed.n_points.sum()), n_clamped=int(per_seed.n_clamped.sum())))
    return out


def seed_means(all_draws: dict, rows: pd.DataFrame, seeds) -> list[dict]:
    """Seed-averaged rows (seed = "mean3") of the MEAN_RESPONSES. coef is the seeds' mean and
    se = sqrt(sum se^2) / n_seeds, the seeds being separate populations on separate test tiles;
    the interval is the percentiles of the seeds' bootstrap draws averaged draw by draw."""
    out = []
    for model_set in MODEL_SETS:
        for formula in FORMULAS:
            for response in MEAN_RESPONSES:
                out += mean_rows(all_draws, rows, seeds, model_set, formula, response)
    return out


def share_seen_among(seen: np.ndarray, n_feat: np.ndarray, k: int):
    """Share of the models with n_feat == k that saw the feature; NaN when there are none."""
    in_group = n_feat == k
    if not in_group.any():
        return np.nan
    return seen[in_group].mean()


def inclusion_tables(si: dict, feats, seed: int):
    """_seen.csv and _corr.csv rows of one seed.

    Under zoo.py's uniform draws a model with n_feat = 3 or 4 sees each feature with probability
    3/7 or 4/7 (n_feat = 7 sees all). A larger n_feat makes every indicator likelier to be 1, a
    positive correlation, while at a fixed n_feat the indicators sum to a constant, which makes
    every pair correlate at exactly -1/6.
    """
    seen, corr = [], []
    for model_set, models in model_sets(si["Z"].shape[0]).items():
        I = si["I"][models]
        n_feat = si["n_feat"][models]
        if not (I.sum(1) == n_feat).all():
            raise SystemExit(f"seed {seed}: the features lists do not have n_feat entries")
        for j, f in enumerate(feats):
            seen.append(dict(seed=seed, model_set=model_set, feature=f, n_models=len(models),
                             n_seen=int(I[:, j].sum()), share_seen=I[:, j].mean(),
                             share_seen_nfeat3=share_seen_among(I[:, j], n_feat, 3),
                             share_seen_nfeat4=share_seen_among(I[:, j], n_feat, 4),
                             n_nfeat3=int((n_feat == 3).sum()), n_nfeat4=int((n_feat == 4).sum()),
                             n_nfeat7=int((n_feat == 7).sum())))
        C = np.corrcoef(np.column_stack([I, n_feat]).T)
        names = feats + ["n_feat"]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                corr.append(dict(seed=seed, model_set=model_set, a=names[i], b=names[j],
                                 corr=C[i, j]))
    return seen, corr


# ------------------------------------------------------------------ report
def summary(df: pd.DataFrame) -> str:
    """The primary formula's saw_* coefficients on pooled_regions and spatial_S500_all."""
    lines = []
    for model_set in ("test60", "test60_noclamp", "all300"):
        for response in ("pooled_regions", "spatial_S500_all"):
            chosen = ((df.model_set == model_set) & (df.response == response)
                      & (df.formula == "id+family+features") & df.term.str.startswith("saw_"))
            lines.append(f"\n{response}, {model_set}, id+family+features: "
                         f"coef [boot 95%] (OLS se)")
            for seed, g in df[chosen].groupby("seed", sort=False):
                g = g.set_index("term")
                worst = g.coef.idxmin()
                lines.append(f"  seed {seed}: most negative {worst} = {g.coef[worst]:+.3f} "
                             f"[{g.boot_lo[worst]:+.3f}, {g.boot_hi[worst]:+.3f}]")
                cells = []
                for term in g.index:
                    cells.append(f"{term.removeprefix('saw_')} {g.coef[term]:+.3f} "
                                 f"[{g.boot_lo[term]:+.3f},{g.boot_hi[term]:+.3f}] "
                                 f"({g.se[term]:.3f})")
                lines.append("    " + "  ".join(cells))
    return "\n".join(lines)


def distinct_values(values) -> str:
    """A group's distinct accuracies as sorted text, e.g. '0,1'."""
    return ",".join(sorted({f"{x:g}" for x in values}))


def print_report(df: pd.DataFrame, counts: list, clamps: list, seen: list, corr: list,
                 feats: list[str]) -> None:
    """The run's log: land rule counts, clamped models, inclusion shares and correlations, and
    the summary of the primary fits."""
    print("\n[D1] land rule (landcover != 11): points per response")
    print(pd.DataFrame(counts).to_string(index=False))

    print("\n[D1] models at the probit clamp (accuracy exactly 0 or 1 on a response)")
    clamp_table = pd.DataFrame(clamps)
    if len(clamp_table):
        per_response = clamp_table.groupby(["seed", "response"]).agg(
            n_all300=("model", "size"), n_test60=("in_test60", "sum"),
            acc_R=("acc_R", distinct_values), median_acc_id=("acc_id", "median"))
        print(per_response.to_string())

    print("\n[D1] share of models that saw each feature")
    shares = pd.DataFrame(seen).pivot_table(index=["seed", "model_set"], columns="feature",
                                            values="share_seen")
    print(shares[feats].round(3).to_string())

    corr_table = pd.DataFrame(corr)
    feature_pairs = corr_table[(corr_table.a != "n_feat") & (corr_table.b != "n_feat")]
    print("\n[D1] indicator correlations (pairs of features): min / mean / max; "
          "corr with n_feat: min / max")
    for (seed, model_set), g in feature_pairs.groupby(["seed", "model_set"]):
        with_n_feat = corr_table[(corr_table.seed == seed) & (corr_table.model_set == model_set)
                                 & (corr_table.b == "n_feat")]
        pairs = g["corr"]
        print(f"  seed {seed} {model_set:6s}: "
              f"{pairs.min():+.3f} / {pairs.mean():+.3f} / {pairs.max():+.3f};"
              f"  {with_n_feat['corr'].min():+.3f} / {with_n_feat['corr'].max():+.3f}")

    print(summary(df))


# ------------------------------------------------------------------ main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", required=True, type=Path)
    ap.add_argument("--sel-root", required=True, type=Path, help="results/<date>/oodselect_paper")
    ap.add_argument("--regions-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path, help="point table with landcover")
    ap.add_argument("--features-file", type=Path, default=Path("data/processed_w7/FEATURES.txt"))
    ap.add_argument("--cell-prefix", default="FL_band_AE", help="cell tag without _s<seed>")
    ap.add_argument("--variant", default="spatial_mu3_knn10")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-size", type=int, default=20, help="smallest component that is a region")
    ap.add_argument("--ref-family", default="dt", help="family without its own column")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    feats = read_features(args.features_file)
    print(f"[D1] features ({len(feats)}): {', '.join(feats)}", flush=True)
    lc = load_landcover(args.dataset)

    rows, seen, corr, counts, clamps, all_draws = [], [], [], [], [], {}
    for seed in args.seeds:
        si = seed_inputs(args, seed, lc, feats)
        seed_rows, seed_draws = fit_seed(si, feats, args, seed)
        rows += seed_rows
        for key, value in seed_draws.items():
            all_draws[(seed, *key)] = value
        clamps += clamped_models(si, seed)
        seed_seen, seed_corr = inclusion_tables(si, feats, seed)
        seen += seed_seen
        corr += seed_corr
        counts += si["counts"]
        print(f"[D1] seed {seed}: {len(si['masks'])} responses, {len(seed_rows)} rows", flush=True)

    df = pd.DataFrame(rows)
    if len(args.seeds) > 1:
        df = pd.concat([df, pd.DataFrame(seed_means(all_draws, df, args.seeds))],
                       ignore_index=True)
    df["seed"] = df.seed.astype(str)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    stem = args.out.with_suffix("")
    pd.DataFrame(seen).to_csv(f"{stem}_seen.csv", index=False)
    pd.DataFrame(corr).to_csv(f"{stem}_corr.csv", index=False)

    print_report(df, counts, clamps, seen, corr, feats)
    print(f"\n[D1] wrote {args.out}, {stem}_seen.csv, {stem}_corr.csv in {time.time() - t0:.0f} s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
