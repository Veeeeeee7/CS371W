#!/usr/bin/env python3
"""
baselines.py -- the transfer measurement this project exists to make.

Input   data/processed/dataset_{ST}.parquet  (from pull_gee.py --merge)
Output  results/baselines_{tag}.csv          per (state, protocol, direction)
        results/tile_gaps_{tag}.csv          per test tile -- the OODSelect input
        results/explain_{tag}.csv            what predicts a tile's gap

THE MEASUREMENT
===============

For each state and each negative protocol, a model is trained on ONE domain's
points in a set of held-out-by-tile training folds, then scored twice on the SAME
held-out test tiles:

    ID    the training domain's points in those tiles
    OOD   the other domain's points in those tiles

gap = AUC(OOD) - AUC(ID).

Scoring both on the same tiles is the whole point. If ID and OOD were measured in
different places, the gap would mix provenance with geography, and measurement on
this data showed that mixture is large: Zone A covers up to 3.8x more 10 km blocks
than Zone AE, and in eight of ten states fewer than a third of blocks hold both
domains. Same-tile scoring removes that by construction -- what differs between the
two numbers is which process drew the labels, not where.

`--paired` goes further and keeps only test tiles that contain both domains, so
every tile contributes to both sides of the gap. That is the primary result. The
unpaired run is reported beside it; the difference between them is the geographic
component of the gap, which is the decomposition DISDE (arXiv:2303.02011) formalises.

WHAT IS HELD FIXED
==================

Within-state only. State floodplain-mapping policy swings the Zone A share from
0.7% (NC) to 76.7% (IA), a 100x spread that would dominate any pooled comparison.

Folds are grouped by `tile_50km`, not `block_id`. Measured: A-vs-AE separability
falls from 0.860 with 10 km blocks to 0.693 with 100 km tiles, so most of what
looked like a domain difference under block grouping was a model memorising a
neighbourhood and being scored on it.

Features come from `data/processed/FEATURES.txt` and nothing else. Absolute
elevation is excluded because it alone identifies the state at 58.3% against a 10%
baseline; `lon`/`lat` are excluded for the same reason; every `meta_` column is
excluded because it encodes how the map was made, which is what is being predicted.

THE NULL
========

`--null` shuffles the domain label within each tile, preserving per-tile counts and
every feature value. A real provenance effect must vanish under it. If a gap
survives the null, the pipeline is manufacturing it -- which is exactly the failure
mode "Shift is Good" (arXiv:2510.25108) and Sanyal et al. (arXiv:2406.19049)
describe, where a transfer gap appears with no transfer effect present. Run it
before believing any number here.

USAGE
=====
    python baselines.py                     # all states, both protocols, paired + unpaired
    python baselines.py VT FL IA            # a subset
    python baselines.py --null              # the shuffled-domain control
    python baselines.py --folds 5 --seed 1  # stability check
    python baselines.py --neg-ratio 0.5     # ratio sensitivity, no rebuild needed
    python baselines.py --summary           # re-print the last run's tables
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATA = Path("data/processed")
RESULTS = Path("results")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]
PROTOCOLS = ["band", "shadedX"]
TILE = "tile_50km"

MIN_TRAIN = 500          # below this a fold's model is not worth fitting
MIN_EVAL_PER_CLASS = 25  # below this a tile's AUC is noise
CATEGORICAL = ["landcover"]


def load_features() -> list[str]:
    f = DATA / "FEATURES.txt"
    if not f.exists():
        sys.exit(f"{f} missing -- run: python pull_gee.py --merge")
    return [l.strip() for l in f.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def make_model(kind: str, seed: int):
    if kind == "logreg":
        # A linear baseline needs imputation and scaling; the tree does not.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=False, random_state=seed)


def usable_columns(X: np.ndarray) -> np.ndarray:
    """Indices of columns a histogram booster can actually bin.

    sklearn bins a feature with sliding_window_view(distinct_values, 2), which
    needs at least two distinct finite values. A column that is entirely NaN, or
    constant, raises "window shape cannot be larger than input array shape" from
    deep inside a joblib worker, where the traceback names numpy rather than the
    offending feature.

    This is not a defensive nicety here -- it is load-bearing. `log10_dist_coast`
    is NaN for every point in a landlocked state, because the FTYPE split found no
    Coastline features there. That is the covariate behaving CORRECTLY: distance to
    the coast is undefined in Vermont. So the feature set is legitimately
    state-dependent, 45 columns in FL/LA/NC/TX and 44 inland, and the fold has to
    adapt rather than the data being patched to suit the model.

    Chosen on the TRAINING fold only and applied unchanged to every evaluation, so
    the model never sees a column at scoring time that it was not fitted on.
    """
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2 and np.unique(col).size >= 2:
            keep.append(j)
    return np.asarray(keep, dtype=int)


def apply_neg_ratio(df: pd.DataFrame, ratio: float | None,
                    seed: int) -> pd.DataFrame:
    """Subset negatives to `ratio` per positive, within each domain and protocol.

    THE RATIO IS A CEILING AND IT IS OFTEN NOT REACHED. Jia et al. (2026)
    recommend two negatives per positive. Measured on this data, the achieved
    ratio at a requested 1:1 already runs from 0.18 (Vermont, shadedX) to 1.04,
    because Zone X 0.2%-chance polygons cover only 31 km2 in Vermont and no
    request can produce more of them. The shadedX protocol reaches 1:1 in 0 of 10
    states; the band protocol reaches it in 7. The two therefore cannot share a
    ratio, and a target of 2 would widen the achieved spread to 0.18-2.09 WITHIN
    the study -- entangling the ratio with state, which is the confounder this
    design exists to hold fixed.

    So the ratio is treated as a sweepable analysis parameter rather than a
    property of the build. Negatives are ordered once, deterministically, within
    each (state, domain, protocol) cell; selecting rank < ratio * n_pos yields
    that ratio, and the subsets nest, so a 0.5 sample is contained in the 1.0
    sample. Sweeping DOWNWARD costs nothing and needs no re-sampling: if the
    transfer gap is stable across ratios, the ratio is not driving the result and
    the question is settled without rebuilding the dataset.

    `neg_rank` is used when the sampler wrote it; otherwise it is reconstructed
    here from a fixed seed, so existing datasets work unchanged.
    """
    if ratio is None:
        return df
    if "neg_rank" not in df.columns:
        rng = np.random.default_rng(seed)
        df = df.copy()
        df["neg_rank"] = -1
        neg = df.label == 0
        if neg.any():
            o = pd.Series(rng.permutation(int(neg.sum())), index=df.index[neg])
            df.loc[neg, "neg_rank"] = (
                o.groupby([df.loc[neg, "domain"],
                           df.loc[neg, "neg_protocol"].fillna("-")])
                 .rank(method="first").astype(int) - 1)

    npos = df[df.label == 1].groupby("domain").size()
    keep = [df[df.label == 1]]
    for (dom, prot), g in df[df.label == 0].groupby(
            ["domain", df[df.label == 0].neg_protocol.fillna("-")]):
        keep.append(g[g.neg_rank < ratio * int(npos.get(dom, 0))])
    return pd.concat(keep, ignore_index=False).sort_index()


def report_ratios(df: pd.DataFrame, state: str) -> None:
    """Print the ratio actually obtained, never the one requested."""
    npos = df[df.label == 1].groupby("domain").size()
    out = []
    for (dom, prot), g in df[df.label == 0].groupby(
            ["domain", df[df.label == 0].neg_protocol.fillna("-")]):
        n = int(npos.get(dom, 0))
        out.append(f"{dom}/{prot} {len(g)/max(n,1):.2f}")
    print(f"    [{state}] achieved negatives per positive: {'  '.join(out)}",
          flush=True)


def load_state(state: str, feats: list[str]) -> pd.DataFrame:
    p = DATA / f"dataset_{state}.parquet"
    if not p.exists():
        return pd.DataFrame()
    cols = (["point_id", "state", "domain", "label", "neg_protocol", TILE,
             "block_id", "dfirm_id", "meta_bfe_km_per_km2", "meta_panel_scale",
             "meta_study_regime", "meta_eff_date", "meta_dist_to_sfha_m",
             "meta_elev", "meta_poly_area_km2"] + feats)
    df = pd.read_parquet(p, columns=[c for c in cols if c in
                                     pd.read_parquet(p, columns=None).columns[:0].union(
                                         pd.read_parquet(p).columns)])
    return df


def load_state_fast(state: str, feats: list[str]) -> pd.DataFrame:
    """Read only the columns we need, tolerating ones a state may lack."""
    import pyarrow.parquet as pq
    p = DATA / f"dataset_{state}.parquet"
    if not p.exists():
        return pd.DataFrame()
    have = set(pq.read_schema(p).names)
    want = (["point_id", "state", "domain", "label", "neg_protocol", TILE,
             "block_id", "dfirm_id", "meta_bfe_km_per_km2", "meta_panel_scale",
             "meta_study_regime", "meta_eff_date", "meta_dist_to_sfha_m",
             "meta_elev", "meta_poly_area_km2", "neg_rank"] + feats)
    df = pd.read_parquet(p, columns=[c for c in want if c in have])
    for c in CATEGORICAL:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def subset(df: pd.DataFrame, domain: str, protocol: str) -> pd.DataFrame:
    """One domain's supervised set: its positives plus its own negatives."""
    return df[(df.domain == domain) &
              ((df.label == 1) | (df.neg_protocol == protocol))]


def shuffle_domains(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The null: permute `domain` within each tile x label x protocol cell.

    Counts, features and geography are untouched, so any surviving gap is
    manufactured by the pipeline rather than by provenance.
    """
    out = df.copy()
    key = [TILE, "label", out.neg_protocol.fillna("-")]
    out["domain"] = (out.groupby(key, observed=True)["domain"]
                     .transform(lambda s: rng.permutation(s.values)))
    return out


def run_state(df: pd.DataFrame, state: str, feats: list[str], protocol: str,
              folds: int, seed: int, paired: bool, model_kind: str) -> list[dict]:
    """Returns one record per (fold, train_domain, eval_domain, test tile)."""
    recs: list[dict] = []
    dropped: set[str] = set()
    tiles = df[TILE].dropna().unique()
    if len(tiles) < folds:
        return recs

    # Fold assignment is on tiles, once, so both domains see the same geography.
    tile_fold = pd.Series(
        np.arange(len(tiles)) % folds,
        index=pd.Index(np.random.default_rng(seed).permutation(tiles), name=TILE))

    for fold in range(folds):
        test_tiles = set(tile_fold[tile_fold == fold].index)
        train_tiles = set(tile_fold[tile_fold != fold].index)

        if paired:
            # Keep only test tiles where BOTH domains have positives, so every
            # tile contributes to both sides of the gap.
            pos = df[df.label == 1]
            in_test = pos[pos[TILE].isin(test_tiles)]
            both = (in_test.groupby(TILE).domain.nunique() == 2)
            test_tiles = set(both[both].index)
            if not test_tiles:
                continue

        for train_dom in ("A", "AE"):
            tr = subset(df, train_dom, protocol)
            tr = tr[tr[TILE].isin(train_tiles)]
            if len(tr) < MIN_TRAIN or tr.label.nunique() < 2:
                continue
            Xtr = tr[feats].to_numpy(dtype=float)
            cols = usable_columns(Xtr)
            if cols.size < 2:
                continue
            dropped.update(feats[j] for j in range(len(feats)) if j not in set(cols))
            m = make_model(model_kind, seed)
            m.fit(Xtr[:, cols], tr.label.to_numpy())

            for eval_dom in ("A", "AE"):
                te = subset(df, eval_dom, protocol)
                te = te[te[TILE].isin(test_tiles)]
                if te.empty:
                    continue
                p = m.predict_proba(
                    te[feats].to_numpy(dtype=float)[:, cols])[:, 1]
                te = te.assign(_p=p)

                for tile, g in te.groupby(TILE):
                    npos = int((g.label == 1).sum())
                    nneg = int((g.label == 0).sum())
                    if min(npos, nneg) < MIN_EVAL_PER_CLASS:
                        continue
                    recs.append(dict(
                        state=state, protocol=protocol, fold=fold,
                        train_domain=train_dom, eval_domain=eval_dom,
                        kind="ID" if eval_dom == train_dom else "OOD",
                        tile=tile, n_pos=npos, n_neg=nneg,
                        auc=roc_auc_score(g.label, g._p),
                        ap=average_precision_score(g.label, g._p),
                        bfe=g.loc[g.label == 1, "meta_bfe_km_per_km2"].median()
                            if "meta_bfe_km_per_km2" in g else np.nan,
                        panel_scale=g.loc[g.label == 1, "meta_panel_scale"].median()
                            if "meta_panel_scale" in g else np.nan,
                        elev=g["meta_elev"].median() if "meta_elev" in g else np.nan,
                        hand=g["hand"].median() if "hand" in g else np.nan,
                        imperv=g["impervious"].median() if "impervious" in g else np.nan,
                        water=g["water_frac_900m"].median()
                            if "water_frac_900m" in g else np.nan,
                    ))
    if dropped:
        # Surfaced rather than swallowed: a feature that silently disappears is
        # the same failure mode that lost channel_slope from the feature list.
        print(f"    [{state}/{protocol}] not usable in this state, excluded: "
              f"{sorted(dropped)}", flush=True)
    return recs


# =========================================================================
def tile_bootstrap(vals: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    """Percentile CI resampling TILES, because points inside a tile are not
    independent -- 20,000 points per domain come from a few hundred polygons and
    the largest 1% of them supply 8-26% of the sample."""
    if len(vals) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def summarise(recs: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Per (state, protocol, train_domain): ID AUC, OOD AUC, gap with a CI."""
    rows = []
    for (st, pr, td), g in recs.groupby(["state", "protocol", "train_domain"]):
        idv = g[g.kind == "ID"].set_index(["fold", "tile"]).auc
        oov = g[g.kind == "OOD"].set_index(["fold", "tile"]).auc
        common = idv.index.intersection(oov.index)
        if len(common) < 3:
            continue
        d = (oov.loc[common] - idv.loc[common]).to_numpy()
        lo, hi = tile_bootstrap(d)
        rows.append(dict(
            state=st, protocol=pr, train=td,
            test="AE" if td == "A" else "A",
            n_tiles=len(common),
            auc_id=idv.loc[common].mean(), auc_ood=oov.loc[common].mean(),
            gap=d.mean(), gap_lo=lo, gap_hi=hi,
            pct_tiles_ood_higher=100 * (d > 0).mean(),
            signif="yes" if (lo > 0 or hi < 0) else "no",
        ))
    return pd.DataFrame(rows)


def explain(recs: pd.DataFrame) -> pd.DataFrame:
    """Does provenance metadata explain a tile's gap better than terrain does?

    This is the claim the project turns on. OODSelect can find the subsets where
    transfer inverts but cannot say what they are; here each tile's gap is
    regressed on provenance variables (BFE line density, map scale) and on terrain
    variables (HAND, imperviousness, water proximity, elevation) separately, and the
    two R-squared values are compared. Spearman, because neither relationship has
    any reason to be linear.
    """
    from scipy.stats import spearmanr
    rows = []
    for (st, pr, td), g in recs.groupby(["state", "protocol", "train_domain"]):
        idv = g[g.kind == "ID"].set_index(["fold", "tile"])
        oov = g[g.kind == "OOD"].set_index(["fold", "tile"])
        common = idv.index.intersection(oov.index)
        if len(common) < 8:
            continue
        gap = (oov.loc[common].auc - idv.loc[common].auc)
        meta = idv.loc[common]
        for var, group in [("bfe", "provenance"), ("panel_scale", "provenance"),
                           ("hand", "terrain"), ("imperv", "terrain"),
                           ("water", "terrain"), ("elev", "terrain")]:
            x = pd.to_numeric(meta[var], errors="coerce")
            ok = x.notna() & gap.notna()
            if ok.sum() < 8 or x[ok].nunique() < 3:
                continue
            rho, p = spearmanr(x[ok], gap[ok])
            rows.append(dict(state=st, protocol=pr, train=td, group=group,
                             variable=var, n=int(ok.sum()),
                             spearman=rho, p_value=p))
    return pd.DataFrame(rows)


def print_tables(summ: pd.DataFrame, exp: pd.DataFrame, label: str) -> None:
    if summ.empty:
        print(f"\n[{label}] nothing to summarise")
        return
    pd.set_option("display.width", 220)
    show = summ.copy()
    for c in ("auc_id", "auc_ood", "gap", "gap_lo", "gap_hi"):
        show[c] = show[c].round(3)
    show["pct_tiles_ood_higher"] = show["pct_tiles_ood_higher"].round(0)
    print(f"\n{'='*104}\n{label}\n{'='*104}")
    print(show.to_string(index=False))

    print(f"\n  pooled over states, by protocol and direction:")
    p = (summ.groupby(["protocol", "train"])
         .apply(lambda g: pd.Series({
             "n_tiles": int(g.n_tiles.sum()),
             "auc_id": g.auc_id.mean().round(3),
             "auc_ood": g.auc_ood.mean().round(3),
             "gap": g.gap.mean().round(3),
             "states_signif": f"{(g.signif=='yes').sum()}/{len(g)}",
         })))
    print("    " + p.to_string().replace("\n", "\n    "))

    if not exp.empty:
        print(f"\n  what predicts a tile's gap (mean |Spearman| by variable group):")
        agg = (exp.assign(absr=exp.spearman.abs())
               .groupby(["group", "variable"])
               .agg(mean_abs_rho=("absr", "mean"), n_cells=("absr", "size"),
                    pct_p05=("p_value", lambda s: 100*(s < .05).mean()))
               .round(3).sort_values("mean_abs_rho", ascending=False))
        print("    " + agg.to_string().replace("\n", "\n    "))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
    ap.add_argument("--protocols", nargs="*", default=PROTOCOLS)
    ap.add_argument("--null", action="store_true",
                    help="shuffle domain within tiles -- the gap must vanish")
    ap.add_argument("--unpaired-only", action="store_true")
    ap.add_argument("--neg-ratio", type=float, default=None,
                    help="negatives per positive; subsets what the "
                         "build already contains. Omit to use all.")
    ap.add_argument("--summary", action="store_true", help="re-print the last run")
    a = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    tag = (f"{a.model}_f{a.folds}_s{a.seed}"
           + (f"_r{a.neg_ratio:g}" if a.neg_ratio is not None else "")
           + ("_null" if a.null else ""))

    if a.summary:
        for mode in ("paired", "unpaired"):
            f = RESULTS / f"tile_gaps_{tag}_{mode}.csv"
            if f.exists():
                r = pd.read_csv(f)
                print_tables(summarise(r, tag), explain(r), f"{mode.upper()} ({tag})")
        return 0

    feats = load_features()
    states = a.states or STATES
    print(f"features ({len(feats)}): {feats}")
    print(f"model={a.model} folds={a.folds} seed={a.seed} "
          f"null={a.null} tile column={TILE}\n")

    modes = [False] if a.unpaired_only else [True, False]
    rng = np.random.default_rng(a.seed)

    for paired in modes:
        mode = "paired" if paired else "unpaired"
        all_recs: list[dict] = []
        t0 = time.time()
        for st in states:
            df = load_state_fast(st, feats)
            if df.empty:
                print(f"  [{st}] no dataset -- skipping")
                continue
            if a.neg_ratio is not None:
                df = apply_neg_ratio(df, a.neg_ratio, a.seed)
            if paired:
                report_ratios(df, st)
            if a.null:
                df = shuffle_domains(df, rng)
            n = 0
            for pr in a.protocols:
                r = run_state(df, st, feats, pr, a.folds, a.seed, paired, a.model)
                all_recs += r
                n += len(r)
            print(f"  [{mode}] {st}: {n:,} tile evaluations "
                  f"({time.time()-t0:.0f}s cumulative)", flush=True)

        if not all_recs:
            print(f"[{mode}] produced nothing")
            continue
        recs = pd.DataFrame(all_recs)
        recs.to_csv(RESULTS / f"tile_gaps_{tag}_{mode}.csv", index=False)
        summ = summarise(recs, tag)
        summ.to_csv(RESULTS / f"baselines_{tag}_{mode}.csv", index=False)
        exp = explain(recs)
        exp.to_csv(RESULTS / f"explain_{tag}_{mode}.csv", index=False)
        print_tables(summ, exp, f"{mode.upper()}  ({tag})")

    print(f"\nwrote results/ -> baselines_{tag}_*.csv, tile_gaps_{tag}_*.csv, "
          f"explain_{tag}_*.csv")
    if not a.null:
        print("\nNow run the control before believing any of it:")
        print(f"    python baselines.py --null")
        print("Every gap should collapse to roughly zero with a CI straddling it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
