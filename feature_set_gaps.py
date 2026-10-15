#!/usr/bin/env python3
"""
feature_set_gaps.py -- how the AE->A transfer gap depends on the feature set.

The measurement is baselines.py's, unchanged: paired test tiles (both domains
present), tile_50km folds, per-tile AUC, tile-bootstrap CI, and the null that
shuffles `domain` within tile x label x protocol. Only the feature list changes.
Florida, band protocol, both training directions.

WHAT IS REPORTED PER FEATURE SET
    gap              OOD AUC - ID AUC on the same tiles (negative = transfer loss)
    null_gap         the same number after the domain shuffle; what the pipeline
                     and the geography produce with provenance removed
    provenance_gap   gap - null_gap: the part only label provenance can explain
    separability     A-vs-AE AUC on positives: how much the two domains are
                     different PLACES in this feature space

WHY NOT JUST TAKE THE LARGEST GAP
    A feature set can widen the gap by letting the AE model fit places Zone A
    does not share (hydrologic position, regional climate), which is covariate
    shift, not label provenance. Choosing features by the raw gap is selection on
    the outcome, the same reason Florida was never chosen by gap size. The rule
    used here, fixed before the numbers were read:
      1. provenance_gap on AE->A (seed 0) must have a CI that excludes 0 and a
         null CI that contains 0;
      2. among those, the largest provenance_gap wins, unless its separability
         is more than 0.05 above the next candidate's, in which case the
         next candidate wins;
      3. the winner is confirmed on seeds 1-2 (new tile partitions), which
         played no part in choosing it.
    Every feature set is reported, so the choice is auditable.

USAGE
    python feature_set_gaps.py                 # all sets, seed 0
    python feature_set_gaps.py --seeds 1 2 --sets aef24+hydro old
Output: results/<date>/feature_set_gaps/{gaps.csv, per_tile.parquet}
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B                                   # noqa: E402
from aef_checks import load, OLD_GROUPS, HYDRO4         # noqa: E402

STATE, PROTOCOL, FOLDS = "FL", "band", 4


def feature_sets(sets: dict) -> dict[str, list[str]]:
    old, aef24, aef17 = sets["old"], sets["aef24"], sets["aef17"]
    streamcat = [c for c in old if c.endswith(("ws", "cat"))]
    local = [c for c in old if c not in streamcat + OLD_GROUPS["atlas14"]]
    terrain_hydro = [c for c in OLD_GROUPS["terrain"] + OLD_GROUPS["hydro"] if c in old]
    win = [c for c in sets["aef24+r100"] if "_r100_" in c]
    return {
        "old": old,
        "old-local": local,                          # old minus StreamCat + Atlas 14
        "aef24": aef24,
        "aef24+hydro": aef24 + HYDRO4,
        "aef24+terrain+hydro": aef24 + terrain_hydro,
        "aef24+old-local": aef24 + local,
        "aef24+old": aef24 + old,
        "aef24+r100+hydro": aef24 + win + HYDRO4,
        "aef17+hydro": aef17 + HYDRO4,
    }


def separability(df: pd.DataFrame, feats: list[str], seed: int) -> float:
    pos = df[df.label == 1]
    X, y, g = pos[feats].to_numpy(float), (pos.domain == "AE").to_numpy().astype(int), pos[B.TILE]
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(FOLDS).split(X, y, g):
        m = B.make_model("hgb", seed); m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def one(df: pd.DataFrame, name: str, feats: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    recs = pd.DataFrame(B.run_state(df, STATE, feats, PROTOCOL, FOLDS, seed, True, "hgb"))
    null = pd.DataFrame(B.run_state(B.shuffle_domains(df, np.random.default_rng(seed)),
                                    STATE, feats, PROTOCOL, FOLDS, seed, True, "hgb"))
    s = B.summarise(recs, name)
    n = B.summarise(null, name)[["train", "gap", "gap_lo", "gap_hi"]].rename(
        columns={"gap": "null_gap", "gap_lo": "null_lo", "gap_hi": "null_hi"})
    s = s.merge(n, on="train")
    s["provenance_gap"] = s.gap - s.null_gap
    s.insert(0, "features", name); s.insert(1, "n_features", len(feats)); s.insert(2, "seed", seed)
    s["separability"] = separability(df, feats, seed)
    s["seconds"] = time.time() - t0
    recs.insert(0, "features", name); recs.insert(1, "seed", seed)
    return s, recs


def common_propensity(df: pd.DataFrame, feats: list[str], seed: int,
                      clip: float = 0.05) -> pd.Series:
    """Out-of-fold P(AE | x) on the band rows, grouped by tile.

    One propensity model on the richest description of place (AEF + every old
    covariate) is used to reweight every feature set, so "the same places" means
    the same thing in every row of the comparison."""
    band = df[(df.label == 1) | (df.neg_protocol == PROTOCOL)]
    X, y = band[feats].to_numpy(float), (band.domain == "AE").to_numpy().astype(int)
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(FOLDS).split(X, y, band[B.TILE]):
        m = B.make_model("hgb", seed); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return pd.Series(np.clip(p, clip, 1 - clip), index=band.index)


def reweighted_gap(df: pd.DataFrame, name: str, feats: list[str], seed: int,
                   prop: pd.Series, train: str = "AE") -> dict:
    """The transfer gap with the covariate shift between domains removed.

    Same folds, paired tiles and per-tile AUC as baselines.run_state. The eval
    domain's points are importance-weighted by P(train|x)/P(eval|x), so within
    each tile they are distributed like the training domain in feature space.
    What survives is the part of the gap that differences in PLACE cannot
    explain -- the Y|x (label) component in DISDE's decomposition
    (arXiv:2303.02011). Tiles whose weighted effective sample size is below
    MIN_EVAL_PER_CLASS in either class are dropped.
    """
    ev = "A" if train == "AE" else "AE"
    band = df[(df.label == 1) | (df.neg_protocol == PROTOCOL)]
    w_all = (prop / (1 - prop)) if train == "AE" else ((1 - prop) / prop)
    tiles = df[B.TILE].dropna().unique()
    tile_fold = pd.Series(np.arange(len(tiles)) % FOLDS,
                          index=pd.Index(np.random.default_rng(seed).permutation(tiles)))
    rows = []
    pos = df[df.label == 1]
    for fold in range(FOLDS):
        test = set(tile_fold[tile_fold == fold].index)
        trn = set(tile_fold[tile_fold != fold].index)
        both = pos[pos[B.TILE].isin(test)].groupby(B.TILE).domain.nunique() == 2
        test = set(both[both].index)
        tr = band[(band.domain == train) & band[B.TILE].isin(trn)]
        Xtr = tr[feats].to_numpy(float)
        cols = B.usable_columns(Xtr)
        m = B.make_model("hgb", seed); m.fit(Xtr[:, cols], tr.label.to_numpy())
        te = band[band[B.TILE].isin(test)]
        te = te.assign(_p=m.predict_proba(te[feats].to_numpy(float)[:, cols])[:, 1],
                       _w=w_all.reindex(te.index).to_numpy())
        for tile, g in te.groupby(B.TILE):
            gi, go = g[g.domain == train], g[g.domain == ev]
            if min((gi.label == 1).sum(), (gi.label == 0).sum(),
                   (go.label == 1).sum(), (go.label == 0).sum()) < B.MIN_EVAL_PER_CLASS:
                continue
            ess = min(_ess(go._w[go.label == c]) for c in (0, 1))
            rows.append(dict(tile=tile, fold=fold,
                             auc_id=roc_auc_score(gi.label, gi._p),
                             auc_ood=roc_auc_score(go.label, go._p),
                             auc_ood_w=(roc_auc_score(go.label, go._p, sample_weight=go._w)
                                        if ess >= B.MIN_EVAL_PER_CLASS else np.nan),
                             ess=ess))
    t = pd.DataFrame(rows)
    d, dw = (t.auc_ood - t.auc_id).to_numpy(), (t.auc_ood_w - t.auc_id).dropna().to_numpy()
    ok = t.auc_ood_w.notna()
    d_same = (t.auc_ood - t.auc_id)[ok].to_numpy()        # unweighted, same tiles as dw
    lo, hi = B.tile_bootstrap(d); wlo, whi = B.tile_bootstrap(dw)
    return dict(features=name, n_features=len(feats), seed=seed, train=train, test=ev,
                n_tiles=len(t), n_tiles_weighted=int(np.isfinite(t.auc_ood_w).sum()),
                gap=d.mean(), gap_lo=lo, gap_hi=hi,
                gap_same_tiles=d_same.mean(),
                gap_reweighted=dw.mean(), gap_rw_lo=wlo, gap_rw_hi=whi,
                covariate_share=1 - dw.mean() / d_same.mean() if d_same.mean() else np.nan,
                median_ess=float(t.ess.median()))


def _ess(w: pd.Series) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--reweight", action="store_true",
                    help="also compute the covariate-shift-adjusted gap (reweighted.csv)")
    ap.add_argument("--out", default=f"results/{date.today().isoformat()}/feature_set_gaps")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df, base = load()
    fs = feature_sets(base)
    names = a.sets or list(fs)
    if a.reweight:
        for seed in a.seeds:
            prop = common_propensity(df, fs["aef24+old"], seed)
            for name in names:
                t0 = time.time()
                rows = [reweighted_gap(df, name, fs[name], seed, prop, tr) for tr in ("AE", "A")]
                rp = out / "reweighted.csv"
                pd.DataFrame(rows).to_csv(rp, mode="a", header=not rp.exists(), index=False)
                r = rows[0]
                print(f"s{seed} {name:22s} AE->A gap {r['gap']:+.3f} (same tiles "
                      f"{r['gap_same_tiles']:+.3f}) | reweighted "
                      f"{r['gap_reweighted']:+.3f} [{r['gap_rw_lo']:+.3f},{r['gap_rw_hi']:+.3f}] "
                      f"covariate share {r['covariate_share']:.2f} tiles {r['n_tiles_weighted']}/"
                      f"{r['n_tiles']} ({time.time() - t0:.0f}s)", flush=True)
        print(f"\nwrote {out}")
        return 0
    for seed in a.seeds:
        for name in names:
            s, recs = one(df, name, fs[name], seed)
            gp = out / "gaps.csv"
            s.to_csv(gp, mode="a", header=not gp.exists(), index=False)
            tp = out / f"per_tile_{name}_s{seed}.parquet"
            recs.to_parquet(tp, index=False)
            r = s[s.train == "AE"].iloc[0]
            print(f"s{seed} {name:22s} AE->A gap {r.gap:+.3f} [{r.gap_lo:+.3f},{r.gap_hi:+.3f}] "
                  f"null {r.null_gap:+.3f} prov {r.provenance_gap:+.3f} "
                  f"sep {r.separability:.3f}  ({r.seconds:.0f}s)", flush=True)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
