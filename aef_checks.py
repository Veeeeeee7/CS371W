#!/usr/bin/env python3
"""
aef_checks.py -- is anything wrong with using AlphaEarth embeddings as the features?

Run after `pull_alphaearth.py FL --merge`. Every test puts the embeddings next to
the current covariate stack (data/processed/FEATURES.txt) on the same points, the
same tile_50km folds and the same model, so each number has a reference.

    T1 integrity       coverage, NaN, vector norms
    T2 stability       2017 vs 2024 cosine at the same point, by domain
    T3 reconstruction  how much of each old covariate the embedding carries
    T4 viability       flood AUC, A-vs-AE separability, AE->A transfer gap
      T4b              which old covariate groups separate A from AE
      T4c              AEF plus four hydrologic covariates
    T5 fingerprint     how geographically close a point's nearest feature-neighbour is
    T6 seams           similarity of close pairs across vs within AEF tile edges
    T7 smoothness      feature distance to k geographic neighbours / to random points
    T8 gate + E0       small model zoo on FL band AE: ID spread, aggregate r,
                       Moran's I of the per-point covariance, proxy selection

Inputs   data/processed/dataset_FL.parquet, data/processed/FEATURES.txt
         data/interim/aef/aef_FL_{2017,2024}.parquet
         data/interim/aef/aef_tiles_FL_{2017,2024}.geojson
Output   results/<date>/aef_checks/T*.csv and summary.txt

USAGE
    python aef_checks.py                     # everything (T8 is the slow one)
    python aef_checks.py --tests T1 T3 T4    # a subset
    python aef_checks.py --n-models 90       # zoo size for T8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm, pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "aotil"))

STATE = "FL"
DATA = Path("data/processed")
AEF = Path("data/interim/aef")
BANDS = [f"A{i:02d}" for i in range(64)]
TILE = "tile_50km"
SEED = 0

# Old covariates the probe tries to recover from the embedding. The first four
# are the ones flood susceptibility leans on and imagery can only infer.
PROBE_TARGETS = ["hand", "log10_dist_river", "tpi_300", "log10_upa",
                 "water_frac_900m", "gsw_occurrence", "slope", "impervious",
                 "log10_dist_coast", "precip8110cat", "channel_slope",
                 "drainage_density", "curvature"]
TERRAIN = ["hand", "log10_dist_river"]      # kept beside AEF in the hybrid set
PROFILE = ["water_frac_900m", "gsw_occurrence", "hand", "log10_dist_river",
           "log10_upa", "impervious", "meta_dist_to_sfha_m"]
OUT = Path(".")                             # set in main()


# ============================================================ data
def old_features() -> list[str]:
    toks = [l.strip() for l in (DATA / "FEATURES.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return [t for l in toks for t in l.split()]


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_parquet(DATA / f"dataset_{STATE}.parquet")
    df = df.drop(columns=[c for c in ["geometry"] if c in df.columns])
    for y in (2017, 2024):
        a = pd.read_parquet(AEF / f"aef_{STATE}_{y}.parquet")
        a = a.rename(columns={c: c.replace("aef_", f"e{y % 100}_")
                              for c in a.columns if c.startswith("aef_")})
        a = a.drop(columns=[c for c in a.columns if c.endswith("_year")])
        df = df.merge(a, on="point_id", how="left")
    old = [f for f in old_features() if f in df.columns]
    pt = lambda y: [f"e{y}_pt_{b}" for b in BANDS]
    win = lambda y: [f"e{y}_r100_{b}" for b in BANDS] + [f"e{y}_r100_norm"]
    sets = {
        "old": old,
        "aef24": pt(24),
        "aef17": pt(17),
        "aef24+r100": pt(24) + win(24),
        "aef24+terrain": pt(24) + TERRAIN,
    }
    return df, sets


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def standardise(X: np.ndarray) -> np.ndarray:
    X = X.astype(float).copy()
    med = np.nanmedian(X, axis=0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    sd = X.std(0); sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def feature_matrix(df: pd.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    """Distance-ready features: unit AEF vectors, standardised old covariates."""
    X = df[cols].to_numpy(float)
    return unit(X) if kind.startswith("aef") and len(cols) == 64 else standardise(X)


def xy(df: pd.DataFrame) -> np.ndarray:
    import graph
    return graph.project(df.lon.to_numpy(), df.lat.to_numpy())


def hgb_clf(seed=SEED):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          random_state=seed)


# ============================================================ T1 / T2
def t1_integrity(df, sets):
    rows = []
    for y in (17, 24):
        pt = df[[f"e{y}_pt_{b}" for b in BANDS]].to_numpy(float)
        ok = np.isfinite(pt).all(1)
        rows.append(dict(year=2000 + y, points=len(df), pt_complete=ok.mean(),
                         r100_complete=df[f"e{y}_r100_A00"].notna().mean(),
                         pt_norm_min=df[f"e{y}_pt_norm"].min(),
                         pt_norm_median=df[f"e{y}_pt_norm"].median(),
                         pt_norm_max=df[f"e{y}_pt_norm"].max(),
                         r100_norm_p01=df[f"e{y}_r100_norm"].quantile(.01),
                         r100_norm_median=df[f"e{y}_r100_norm"].median(),
                         distinct_pt_vectors=np.unique(np.round(pt[ok], 3), axis=0).shape[0]))
    return pd.DataFrame(rows)


def t2_stability(df, sets):
    a, b = df[sets["aef17"]].to_numpy(float), df[sets["aef24"]].to_numpy(float)
    cos = (unit(a) * unit(b)).sum(1)
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, len(df), (2, 20000))
    rand = np.median((unit(b)[i] * unit(b)[j]).sum(1))
    g = df.assign(cos=cos, part=np.where(df.label == 1, "pos", df.neg_protocol))
    out = (g.groupby(["domain", "part"]).cos
           .agg(n="size", median="median", share_below_0p8=lambda s: (s < .8).mean())
           .reset_index())
    out.loc[len(out)] = dict(domain="all", part="all", n=len(g), median=np.median(cos),
                             share_below_0p8=(cos < .8).mean())
    out.loc[len(out)] = dict(domain="random pairs", part="2024", n=20000,
                             median=rand, share_below_0p8=np.nan)
    return out


# ============================================================ T3
def t3_reconstruction(df, sets, n_sample=40000):
    rng = np.random.default_rng(SEED)
    sub = df.iloc[rng.choice(len(df), min(n_sample, len(df)), replace=False)]
    rows = []
    for target in PROBE_TARGETS:
        if target not in sub.columns:
            continue
        y = sub[target].to_numpy(float)
        ok = np.isfinite(y)
        for name in ("aef24", "aef24+r100", "old-minus-target"):
            cols = ([c for c in sets["old"] if c != target] if name == "old-minus-target"
                    else sets[name])
            X = sub[cols].to_numpy(float)[ok]
            yy, groups = y[ok], sub[TILE].to_numpy()[ok]
            pred = np.full(len(yy), np.nan)
            for tr, te in GroupKFold(4).split(X, yy, groups):
                m = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
                m.fit(X[tr], yy[tr]); pred[te] = m.predict(X[te])
            rows.append(dict(target=target, predictors=name, n=int(ok.sum()),
                             r2_tile_cv=r2_score(yy, pred)))
        print(f"  T3 {target:18s} " + "  ".join(
            f"{r['predictors']}={r['r2_tile_cv']:.3f}" for r in rows[-3:]), flush=True)
    out = pd.DataFrame(rows).pivot(index="target", columns="predictors",
                                   values="r2_tile_cv").reset_index()
    return out[["target", "aef24", "aef24+r100", "old-minus-target"]]


# ============================================================ T4
def oof_auc(X, y, groups, seed=SEED):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(4).split(X, y, groups):
        m = hgb_clf(seed); m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pred), pred


def transfer_gap(df, cols, protocol="band", train="AE", seed=SEED):
    """AE -> A on the same held-out tiles, as baselines.py --paired does."""
    ev = "A" if train == "AE" else "AE"
    sub = df[(df.label == 1) | (df.neg_protocol == protocol)]
    tiles = sub[TILE].to_numpy()
    ids, oods = [], []
    for k, (tr, te) in enumerate(GroupKFold(4).split(sub, groups=tiles)):
        trd, ted = sub.iloc[tr], sub.iloc[te]
        both = ted[ted.label == 1].groupby(TILE).domain.nunique() == 2
        ted = ted[ted[TILE].isin(both[both].index)]
        fit = trd[trd.domain == train]
        m = hgb_clf(seed); m.fit(fit[cols].to_numpy(float), fit.label.to_numpy())
        for dom, store in ((train, ids), (ev, oods)):
            t = ted[ted.domain == dom]
            if t.label.nunique() == 2:
                store.append(roc_auc_score(t.label, m.predict_proba(t[cols].to_numpy(float))[:, 1]))
    return float(np.mean(ids)), float(np.mean(oods))


def t4_viability(df, sets, keep_pred: dict):
    band = df[(df.label == 1) | (df.neg_protocol == "band")]
    pos = df[df.label == 1]
    rows = []
    for name, cols in sets.items():
        auc_flood, pred = oof_auc(band[cols].to_numpy(float), band.label.to_numpy(),
                                  band[TILE].to_numpy())
        keep_pred[name] = pd.Series(pred, index=band.point_id.to_numpy())
        auc_dom_pos, _ = oof_auc(pos[cols].to_numpy(float),
                                 (pos.domain == "AE").to_numpy().astype(int), pos[TILE].to_numpy())
        auc_dom_all, _ = oof_auc(band[cols].to_numpy(float),
                                 (band.domain == "AE").to_numpy().astype(int), band[TILE].to_numpy())
        id_auc, ood_auc = transfer_gap(df, cols)
        rows.append(dict(features=name, n_features=len(cols),
                         flood_auc=auc_flood,
                         A_vs_AE_positives=auc_dom_pos, A_vs_AE_band_rows=auc_dom_all,
                         margin=auc_flood - auc_dom_pos,
                         AE_to_A_id_auc=id_auc, AE_to_A_ood_auc=ood_auc,
                         AE_to_A_gap=ood_auc - id_auc))
        print(f"  T4 {name:14s} flood {auc_flood:.3f} | A/AE pos {auc_dom_pos:.3f} "
              f"all {auc_dom_all:.3f} | AE->A ID {id_auc:.3f} OOD {ood_auc:.3f}", flush=True)
    # out-of-fold flood probabilities, reused by T6 for the prediction-jump check
    pd.DataFrame(keep_pred).rename_axis("point_id").reset_index().to_parquet(
        OUT / "T4_oof_flood_pred.parquet", index=False)
    return pd.DataFrame(rows)


OLD_GROUPS = {
    "atlas14": ["p100_24hr", "p100_60min", "p100_4day", "p2_24hr", "growth_100_2"],
    "terrain": ["slope", "aspect_sin", "aspect_cos", "curvature", "tpi_300", "tri_300",
                "hand", "tpi_broad"],
    "hydro": ["log10_upa", "catch_area_km2", "nearest_stream_order", "drainage_density",
              "channel_slope", "log10_dist_river", "log10_upstream_area"],
    "water": ["gsw_occurrence", "water_frac_900m"],
    "coast+landcover": ["log10_dist_coast", "landcover", "impervious"],
}


def t4b_old_groups(df, sets):
    """Why the old stack separates Zone A from AE positives: each group alone,
    and everything but that group."""
    old = sets["old"]
    groups = {"streamcat": [c for c in old if c.endswith(("ws", "cat"))], **OLD_GROUPS}
    pos = df[df.label == 1]
    y = (pos.domain == "AE").to_numpy().astype(int)
    g = pos[TILE].to_numpy()
    rows = []
    for k, cols in groups.items():
        cols = [c for c in cols if c in old]
        rest = [c for c in old if c not in cols]
        rows.append(dict(group=k, n=len(cols),
                         alone=oof_auc(pos[cols].to_numpy(float), y, g)[0],
                         all_but=oof_auc(pos[rest].to_numpy(float), y, g)[0]))
    rest = [c for c in old if c not in groups["streamcat"] + groups["atlas14"]]
    rows.append(dict(group="old minus streamcat+atlas14", n=len(rest),
                     alone=oof_auc(pos[rest].to_numpy(float), y, g)[0], all_but=np.nan))
    return pd.DataFrame(rows)


HYDRO4 = ["hand", "log10_dist_river", "log10_upa", "nearest_stream_order"]


def t4c_hydro(df, sets):
    """AEF plus the four hydrologic-position covariates it reconstructs worst."""
    band = df[(df.label == 1) | (df.neg_protocol == "band")]
    pos = df[df.label == 1]
    rows = []
    for name, cols in (("aef24+hydro", sets["aef24"] + HYDRO4), ("hydro4 alone", HYDRO4)):
        f = oof_auc(band[cols].to_numpy(float), band.label.to_numpy(), band[TILE].to_numpy())[0]
        s = oof_auc(pos[cols].to_numpy(float), (pos.domain == "AE").to_numpy().astype(int),
                    pos[TILE].to_numpy())[0]
        r = dict(features=name, n_features=len(cols), flood_auc=f, A_vs_AE_positives=s,
                 margin=f - s)
        if name != "hydro4 alone":
            i, o = transfer_gap(df, cols)
            r.update(AE_to_A_id_auc=i, AE_to_A_ood_auc=o, AE_to_A_gap=o - i)
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================ T5
def t5_fingerprint(df, sets, n_query=10000):
    P = xy(df)
    rng = np.random.default_rng(SEED)
    q = rng.choice(len(df), n_query, replace=False)
    rows = []
    for name in ("old", "aef24", "aef17", "aef24+terrain"):
        F = feature_matrix(df, sets[name], name)
        nn = NearestNeighbors(n_neighbors=2).fit(F)
        _, idx = nn.kneighbors(F[q])
        j = np.where(idx[:, 0] == q, idx[:, 1], idx[:, 0])
        dkm = np.linalg.norm(P[q] - P[j], axis=1) / 1000
        rows.append(dict(features=name, median_km=np.median(dkm),
                         within_1km=(dkm < 1).mean(), within_10km=(dkm < 10).mean(),
                         same_tile50=(df[TILE].to_numpy()[q] == df[TILE].to_numpy()[j]).mean(),
                         same_county=(df.county_fips.to_numpy()[q] ==
                                      df.county_fips.to_numpy()[j]).mean()))
    i, j = rng.integers(0, len(df), (2, n_query))
    dkm = np.linalg.norm(P[i] - P[j], axis=1) / 1000
    rows.append(dict(features="random pairs", median_km=np.median(dkm),
                     within_1km=(dkm < 1).mean(), within_10km=(dkm < 10).mean(),
                     same_tile50=(df[TILE].to_numpy()[i] == df[TILE].to_numpy()[j]).mean(),
                     same_county=(df.county_fips.to_numpy()[i] == df.county_fips.to_numpy()[j]).mean()))
    return pd.DataFrame(rows)


# ============================================================ T6
def tile_of(df, year: int) -> np.ndarray:
    from shapely import STRtree, points
    from shapely.geometry import shape
    from pyproj import Transformer
    # Earth Engine returns each tile's footprint in the tile's own UTM zone
    # (metres), not lon/lat, so points are projected per zone before the test.
    gj = json.loads((AEF / f"aef_tiles_{STATE}_{year}.geojson").read_text())
    out = np.full(len(df), "", dtype=object)
    n_in = np.zeros(len(df), int)
    feats = gj["features"]
    for zone in sorted({f["properties"]["utm_zone"] for f in feats}):
        epsg = 32600 + int(zone[:-1]) if zone.endswith("N") else 32700 + int(zone[:-1])
        tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = tf.transform(df.lon.to_numpy(), df.lat.to_numpy())
        zf = [f for f in feats if f["properties"]["utm_zone"] == zone]
        tree = STRtree([shape(f["geometry"]) for f in zf])
        pi, gi = tree.query(points(x, y), predicate="within")
        n_in += np.bincount(pi, minlength=len(df))
        # a point inside two overlapping footprints (the 16N/17N overlap) keeps
        # the first tile it was assigned; overlaps are counted separately
        for p, g in zip(pi, gi):
            if out[p] == "":
                out[p] = zf[g]["properties"]["tile"]
    return out, n_in


def t6_seams(df, sets, keep_pred: dict, max_m=400):
    if not keep_pred and (OUT / "T4_oof_flood_pred.parquet").exists():
        kp = pd.read_parquet(OUT / "T4_oof_flood_pred.parquet").set_index("point_id")
        keep_pred.update({c: kp[c] for c in kp.columns})
    P = xy(df)
    pairs = cKDTree(P).query_pairs(max_m, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(P[i] - P[j], axis=1)
    rows = []
    for year in (2017, 2024):
        tile, n_in = tile_of(df, year)
        cross = tile[i] != tile[j]
        dis = {}
        for name in (f"aef{year % 100}", "old"):
            F = feature_matrix(df, sets[name], name)
            dis[name] = np.linalg.norm(F[i] - F[j], axis=1)
        if "aef24" in keep_pred:
            pr = keep_pred["aef24"].reindex(df.point_id).to_numpy()
            dis["pred_aef24"] = np.abs(pr[i] - pr[j])
        bins = np.digitize(dist, [100, 200, 300])
        for b in np.unique(bins):
            sel = bins == b
            r = dict(year=year, dist_bin=["<100", "100-200", "200-300", "300-400"][b],
                     n_same=int((sel & ~cross).sum()), n_cross=int((sel & cross).sum()),
                     points_in_overlap=int((n_in > 1).sum()),
                     points_outside=int((n_in == 0).sum()))
            for name, v in dis.items():
                a, c = v[sel & ~cross], v[sel & cross]
                a, c = a[np.isfinite(a)], c[np.isfinite(c)]
                key = "aef" if name.startswith("aef") else name
                r[f"{key}_same"] = a.mean() if len(a) else np.nan
                r[f"{key}_cross"] = c.mean() if len(c) else np.nan
                r[f"{key}_ratio"] = (c.mean() / a.mean()) if len(a) and len(c) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================ T7
def t7_smoothness(df, sets, k=10):
    P = xy(df)
    _, nb = cKDTree(P).query(P, k=k + 1)
    nb = nb[:, 1:]
    geo_m = np.median(np.linalg.norm(P[:, None, :] - P[nb], axis=2))
    rng = np.random.default_rng(SEED)
    rnd = rng.integers(0, len(df), (len(df), k))
    rows = []
    for name in ("old", "aef24", "aef17", "aef24+terrain"):
        F = feature_matrix(df, sets[name], name)
        near = np.linalg.norm(F[:, None, :] - F[nb], axis=2).mean()
        far = np.linalg.norm(F[:, None, :] - F[rnd], axis=2).mean()
        rows.append(dict(features=name, k=k, median_neighbour_m=geo_m,
                         dist_to_knn=near, dist_to_random=far, ratio=near / far))
    return pd.DataFrame(rows)


# ============================================================ T8
def probit(a, eps=1e-6):
    return norm.ppf(np.clip(a, eps, 1 - eps))


def t8_gate(df, sets, n_models, frac=0.05, n_perm=100):
    import zoo, graph
    rows, masks = [], {}
    for name in ("old", "aef24", "aef24+terrain"):
        t0 = time.time()
        r = zoo.run_cell(df, sets[name], "band", "AE", n_models, SEED, n_jobs=-1)
        m = r["models"]
        x = probit(m.score_id.to_numpy())
        S = r["S_ood"]
        agg = pearsonr(x, probit(S.mean(1)))[0]
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(m))
        sel, rep = perm[:int(.6 * len(m))], perm[int(.8 * len(m)):]
        xs = x[sel]
        c = ((xs - xs.mean()) @ (S[sel] - S[sel].mean(0))) / (len(sel) - 1)
        sp = r["spatial"]
        A = graph.build_adjacency(sp["lon"], sp["lat"], k=10)
        mi = graph.morans_i_null(c, A, n_perm=n_perm, seed=SEED)
        # Proxy selection, not OODSelect: the frac most-negative c_i on the
        # selection models, scored on the held-out reporting models.
        n_s = int(frac * S.shape[1])
        mask = np.zeros(S.shape[1], bool); mask[np.argsort(c)[:n_s]] = True
        r_sel = pearsonr(x[rep], probit(S[rep][:, mask].mean(1)))[0]
        rmask = np.zeros_like(mask); rmask[rng.choice(len(mask), n_s, replace=False)] = True
        r_rand = pearsonr(x[rep], probit(S[rep][:, rmask].mean(1)))[0]
        coh = graph.coherence_metrics(mask, A, 10, tiles=r["tiles_ood"])
        rows.append(dict(features=name, n_models=len(m),
                         id_spread=m.score_id.max() - m.score_id.min(),
                         aggregate_r=agg,
                         moran_I_c=mi["moran_I"], moran_z=mi["z"],
                         frac_c_negative=(c < 0).mean(),
                         proxy_sel_r_reporting=r_sel, random_r_reporting=r_rand,
                         proxy_n_cc=coh["n_cc"], proxy_largest_cc=coh["largest_cc_frac"],
                         proxy_tiles=coh["tiles"], proxy_purity=coh["purity"],
                         seconds=time.time() - t0))
        print(f"  T8 {name:14s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in rows[-1].items() if k != "features"), flush=True)
        masks[name] = mask

    # Who the proxy-selected inverse points are, in the old covariates, and
    # whether the feature sets pick the same points. zoo.run_cell builds the OOD
    # rows exactly like this, so the order matches S_ood's columns.
    _, te_tiles = zoo.split_tiles(df, SEED)
    ood = zoo.subset(df, "A", "band"); ood = ood[ood[TILE].isin(te_tiles)]
    prof = []
    for name, mask in masks.items():
        assert len(mask) == len(ood)
        for part, sel in (("selected", mask), ("rest", ~mask)):
            g = ood[sel]
            prof.append(dict(features=name, part=part, n=len(g),
                             positive_share=(g.label == 1).mean(),
                             **{f"mean_{c}": g[c].mean() for c in PROFILE}))
    names = list(masks)
    for a_ in range(len(names)):
        for b_ in range(a_ + 1, len(names)):
            ma, mb = masks[names[a_]], masks[names[b_]]
            jac = (ma & mb).sum() / max((ma | mb).sum(), 1)
            f = ma.mean()
            prof.append(dict(features=f"{names[a_]} vs {names[b_]}", part="jaccard",
                             n=int((ma & mb).sum()), positive_share=np.nan,
                             **{f"mean_{c}": np.nan for c in PROFILE},
                             jaccard=jac, jaccard_over_chance=jac / (f / (2 - f))))
    pd.DataFrame(prof).to_csv(OUT / "T8b_proxy_profile.csv", index=False)
    print(pd.DataFrame(prof).round(3).to_string(index=False), flush=True)
    return pd.DataFrame(rows)


# ============================================================ main
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="*",
                    default=["T1", "T2", "T3", "T4", "T4b", "T4c", "T5", "T6", "T7", "T8"])
    ap.add_argument("--n-models", type=int, default=90)
    ap.add_argument("--out", default=f"results/{date.today().isoformat()}/aef_checks")
    a = ap.parse_args(argv)
    global OUT
    out = OUT = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df, sets = load()
    print(f"{len(df):,} points; feature sets: " +
          ", ".join(f"{k}={len(v)}" for k, v in sets.items()), flush=True)
    keep_pred: dict = {}
    funcs = dict(T1=lambda: t1_integrity(df, sets), T2=lambda: t2_stability(df, sets),
                 T3=lambda: t3_reconstruction(df, sets),
                 T4=lambda: t4_viability(df, sets, keep_pred),
                 T4b=lambda: t4b_old_groups(df, sets), T4c=lambda: t4c_hydro(df, sets),
                 T5=lambda: t5_fingerprint(df, sets), T6=lambda: t6_seams(df, sets, keep_pred),
                 T7=lambda: t7_smoothness(df, sets), T8=lambda: t8_gate(df, sets, a.n_models))
    names = dict(T1="integrity", T2="stability", T3="reconstruction", T4="viability",
                 T4b="old_separability_by_group", T4c="aef_plus_hydro",
                 T5="fingerprint", T6="seams", T7="smoothness", T8="gate_e0")
    for t in a.tests:
        t0 = time.time()
        res = funcs[t]()
        res.to_csv(out / f"{t}_{names[t]}.csv", index=False)
        print(f"\n=== {t} {names[t]} ({time.time() - t0:.0f}s) ===\n" +
              res.round(4).to_string(index=False), flush=True)
    write_summary(out)
    print(f"\nwrote {out}")
    return 0


def write_summary(out: Path) -> None:
    """Rebuild summary.txt from every table in the folder, so partial reruns
    (--tests T8) leave one current copy of each result rather than duplicates."""
    with open(out / "summary.txt", "w") as log:
        for f in sorted(out.glob("T*.csv")):
            log.write(f"\n=== {f.stem} ===\n" +
                      pd.read_csv(f).round(4).to_string(index=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
