#!/usr/bin/env python3
"""
E2 -- does the region hold on points the selection never saw?

THE CRITIQUE THIS ANSWERS
-------------------------
OODSelect's subsets are fit to the very points they are then evaluated on. The
disjoint MODEL splits control for overfitting to models; nothing in the method
controls for overfitting to points. A subset of 500 points chosen to
anti-correlate with model quality, and then scored on those same 500 points, has
one obvious trivial explanation.

A subset that is a PLACE does not have that problem: if the inversion belongs to
a region, then points of that region which were held out of the selection should
invert too. So:

  1. split the OOD points in half within each tile, stratified by label;
  2. select on H1 only, producing weights w over H1;
  3. carry w to H2 by the spatial field -- each H2 point takes the mean w of its
     k nearest H1 points -- and take the top f as H2's region;
  4. score the correlation on H2's region, on the reporting models.

Unconstrained selection (mu = 0) has no spatial structure to carry, so its
transfer should collapse toward the random control. That collapse is a result in
its own right. A place-organised inversion should survive.

Pseudo-replication note: points are 200 m apart inside corridors and the largest
1% of source polygons supply 8-26% of a domain, so anything reported at the
point level is bootstrapped at the TILE level, never the point level. Inside a
compact region this is worse, not better, which is exactly why this test exists.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph                          # noqa: E402
import oodselect as oo                # noqa: E402


def split_points(tiles: np.ndarray, labels: np.ndarray, seed: int = 0):
    """Half the OOD points per tile, stratified by label.

    Splitting within tile rather than globally keeps both halves spread over the
    same geography, so a difference between them cannot be a difference of place.
    """
    rng = np.random.default_rng(seed)
    h1 = np.zeros(len(tiles), bool)
    tiles = np.asarray(tiles)
    labels = np.asarray(labels)
    for t in np.unique(tiles):
        for lab in np.unique(labels):
            idx = np.where((tiles == t) & (labels == lab))[0]
            if len(idx) < 2:
                h1[idx] = True                     # too few to split; keep in H1
                continue
            take = rng.permutation(idx)[: len(idx) // 2]
            h1[take] = True
    return h1, ~h1


def carry_field(w_h1: np.ndarray, xy_h1: np.ndarray, xy_h2: np.ndarray,
                k: int = 10) -> np.ndarray:
    """Each H2 point takes the mean selection weight of its k nearest H1 points."""
    k_eff = min(k, len(xy_h1))
    if k_eff < 1:
        return np.zeros(len(xy_h2))
    _, idx = cKDTree(xy_h1).query(xy_h2, k=k_eff)
    idx = np.atleast_2d(idx.T).T if idx.ndim == 1 else idx
    return np.asarray(w_h1)[idx].mean(axis=1)


def holdout_transfer(x_sel, S_sel, x_rep, S_rep, lon, lat, tiles, labels,
                     frac=0.10, mu=0.0, knn=10, graph_kind="knn",
                     groups=None, seed=0, iters=400, restarts=3,
                     device="cpu") -> dict:
    """Select on H1, carry the field to H2, score H2's region on reporting models.

    x_sel/S_sel are the selection models; x_rep/S_rep the reporting models. All
    arrays are over the full OOD point set; the halving happens inside.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    h1, h2 = split_points(tiles, labels, seed)
    i1, i2 = np.where(h1)[0], np.where(h2)[0]
    if len(i1) < 50 or len(i2) < 50:
        return dict(r_h2=np.nan, r_h1=np.nan, n_h2_selected=0,
                    note="one half too small to split")

    g1 = None if groups is None else np.asarray(groups)[i1]
    A1 = graph.build_adjacency(lon[i1], lat[i1], k=knn, groups=g1)
    L1 = graph.laplacian(A1)

    s_target = max(20, int(len(i1) * frac))
    masks, info = oo.solve(x_sel, S_sel[:, i1], s_target, L=L1, mus=mu, knn=knn,
                           iters=iters, restarts=restarts, seed=seed, device=device)
    w1 = info[0]["w"]

    xy = graph.project(lon, lat)
    w2 = carry_field(w1, xy[i1], xy[i2], k=knn)
    k2 = max(20, int(len(i2) * frac))
    sel2_local = np.argsort(-w2)[:k2]

    m1 = np.zeros(S_rep.shape[1], bool); m1[i1[masks[0]]] = True
    m2 = np.zeros(S_rep.shape[1], bool); m2[i2[sel2_local]] = True

    rng = np.random.default_rng(seed + 77)
    rnd2 = np.zeros(S_rep.shape[1], bool)
    rnd2[rng.choice(i2, k2, replace=False)] = True

    A2 = graph.build_adjacency(lon[i2], lat[i2], k=knn,
                               groups=None if groups is None else np.asarray(groups)[i2])
    mets = graph.coherence_metrics(np.isin(i2, i2[sel2_local]), A2, knn)

    return dict(
        r_h1=oo.evaluate(x_rep, S_rep, m1)["r"],
        r_h2=oo.evaluate(x_rep, S_rep, m2)["r"],
        r_h2_random=oo.evaluate(x_rep, S_rep, rnd2)["r"],
        r_full=oo.evaluate(x_rep, S_rep, np.ones(S_rep.shape[1], bool))["r"],
        n_h1=len(i1), n_h2=len(i2),
        n_h1_selected=int(masks[0].sum()), n_h2_selected=int(k2),
        mu=float(mu), frac=float(frac), knn=int(knn), graph=graph_kind,
        purity_h2=mets["purity"], n_cc_h2=mets["n_cc"],
    )


def tile_bootstrap_r(x: np.ndarray, S: np.ndarray, mask: np.ndarray,
                     tiles: np.ndarray, n: int = 500, seed: int = 0) -> tuple:
    """Percentile CI for the correlation, resampling TILES not points.

    Points inside a tile are not independent: 20,000 points per domain come from
    a few hundred source polygons and the largest 1% of them supply 8-26% of the
    sample. A point-level bootstrap would report an interval several times too
    narrow.
    """
    tiles = np.asarray(tiles)
    uniq = np.unique(tiles[mask])
    if len(uniq) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        m = np.zeros(len(mask), bool)
        for t in pick:
            m |= mask & (tiles == t)
        if m.sum() >= 2:
            r = oo.evaluate(x, S, m)["r"]
            if np.isfinite(r):
                out.append(r)
    if len(out) < 10:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ------------------------------------------------------------------ CLI
def main() -> int:
    import argparse
    import pandas as pd
    from cells import load_cell, cell_tag

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="data/interim/aotil/cells")
    ap.add_argument("--out", default="results/aotil")
    ap.add_argument("--state", default="FL")
    ap.add_argument("--protocol", default="band")
    ap.add_argument("--train-dom", default="AE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mus", default="0,3")
    ap.add_argument("--fracs", default="0.10")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--graph", default="knn",
                    choices=["knn", "block", "catchment", "provenance"])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    tag = cell_tag(a.state, a.protocol, a.train_dom, a.seed)
    cell = load_cell(a.cells, tag)
    cell.require_spatial()

    groups = None
    if a.graph == "block":
        groups = cell.block_id
    elif a.graph == "catchment":
        groups = cell.meta_comid
    elif a.graph == "provenance":
        groups = (cell.meta_source_cit if cell.meta_source_cit is not None
                  else cell.county_fips)

    sel, _, rep = cell.splits(a.seed)["random-split"]
    rows = []
    print(f"=== E2 held-out-point transfer: {tag} (graph={a.graph}) ===")
    for f in [float(v) for v in a.fracs.split(",")]:
        for mu in [float(v) for v in a.mus.split(",")]:
            r = holdout_transfer(cell.x[sel], cell.S_ood[sel],
                                 cell.x[rep], cell.S_ood[rep],
                                 cell.lon, cell.lat, cell.tile_50km, cell.y_ood,
                                 frac=f, mu=mu, knn=a.knn, graph_kind=a.graph,
                                 groups=groups, seed=a.seed, iters=a.iters,
                                 restarts=a.restarts, device=a.device)
            r.update(cell=tag, state=a.state, protocol=a.protocol,
                     train_dom=a.train_dom, seed=a.seed)
            rows.append(r)
            print(f"  f={f:.2f} mu={mu:<5g} r(H1)={r['r_h1']:+.3f} "
                  f"r(H2)={r['r_h2']:+.3f} random(H2)={r['r_h2_random']:+.3f} "
                  f"purity(H2)={r['purity_h2']:.3f}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{tag}_{a.graph}_transfer.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)

    print("\nReading it: r(H1) is the selection's own points, r(H2) points it "
          "never saw.\nA point-type inversion collapses toward the random "
          "control at H2; a\nplace-organised one survives. That collapse is "
          "itself the answer to the\nstanding critique that OODSelect's subsets "
          "are fit to the points they are\nscored on.")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
