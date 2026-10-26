#!/usr/bin/env python3
"""
Checks for region_match.py (the caliper matching of the paper's Approach 3.6). Run from the repo
root:

    python aotil/test_region_match.py

Synthetic cases with known answers (docs/TASK-RERUN-ALL-2026-10-25.md, Part C):
1. On a synthetic pool, exactly 5% of points (rounded) are unmatched against the pool.
2. A candidate in a tile that holds any point of the region is never a match, even when it is the
   closest; under --exclude own-tile only the query's own tile is left out.
3. A closer candidate with a different land cover is never a match.
4. With 10 candidates inside eps, the 5 closest are kept, sorted by distance.
5. eta by hand: a positive point with matches labeled 1, 0, 0 scores 2/3; unmatched points are
   left out.
6. A query whose 500 nearest same-class candidates all lie in excluded tiles still finds its
   out-of-tile match (the fixed-k bug of the sandbox).
7. Standardization uses the D_OOD mean and SD, not the pool's.
8. Pool points missing a feature are dropped and counted.
Helpers: the caliper with points that have no candidate, the weighted median, the feature list,
region names, the exclusion matrices, the hardest points, the halo exclusions, the kinds table
and an end-to-end run on a synthetic cell.
9. The real check (Stop S2): on the 2026-10-23 cell and regions with --exclude own-tile, eps and
   the sandbox's eta and unmatched shares to 0.01. Skipped when those files are missing.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodselect_paper as op  # noqa: E402
import region_match as rm  # noqa: E402

FAILS = []
FEATURES = ["slope", "tri_300", "wetindexcat", "log10_dist_river", "log10_dist_coast", "hand"]
REAL = dict(cells="data/interim/aotil/cells_v3",
            dataset="data/processed_w7/dataset_FL.parquet",
            features="data/processed_w7/FEATURES.txt",
            regions="results/2026-10-23/regions/FL_band_AE_s0_spatial_mu3_knn10_S500_points.parquet",
            classic="results/2026-10-23/regions/FL_band_AE_s0_classic_S500_points.parquet")
# The sandbox's numbers (docs/TASK-RERUN-ALL-2026-10-25.md, Part C): (unmatched, eta).
SANDBOX = {"Zone AE baseline": (0.05, 0.297), "Other Zone A points": (0.047, 0.254),
           "All regions (7)": (None, 0.654), "Liberty": (0.00, 0.73), "Levy": (0.00, 0.74),
           "Dixie": (0.00, 0.66), "Hendry": (0.03, 0.87), "Palm Beach": (0.00, 0.48),
           "Hamilton": (0.10, 0.71), "Hendry/Palm Beach": (0.02, 0.48)}


def check(cond, msg):
    """Print ok or FAIL with the message; remember the failures."""
    if cond:
        print("  ok    " + msg)
    else:
        print("  FAIL  " + msg)
        FAILS.append(msg)


def tiny_pool(z, landcover, tile, y=None):
    z = np.asarray(z, float).reshape(len(landcover), -1)
    y = np.zeros(len(landcover), int) if y is None else y
    return rm.Pool(z, np.asarray(landcover), np.asarray(tile), np.asarray(y))


def no_exclusion(nq, n_tiles):
    return np.zeros((nq, n_tiles), bool)


def check_caliper_share():
    print("1. 5% of a synthetic pool unmatched against itself")
    # Points on a line at c i^2, each in its own tile (so own-tile = itself): point i's nearest
    # distance is the gap to its left, c (2i - 1), distinct for every i >= 1, so no tie can sit
    # at the cut. Random points tie in pairs (mutual nearest neighbours share a distance).
    for n in (1000, 777):
        x = 1e-3 * np.arange(n, dtype=float) ** 2
        pool = tiny_pool(x[:, None], np.ones(n, int), np.arange(n))
        own = rm.own_tile_exclusion(pool.tile_code, n)
        nearest = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=1)[1][:, 0]
        eps = rm.caliper(nearest)
        idx, _ = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=5, eps=eps)
        unmatched = int((idx[:, 0] < 0).sum())
        check(unmatched == round(0.05 * n),
              f"n = {n}: {unmatched} unmatched = round(5% of n) = {round(0.05 * n)}")
    rng = np.random.default_rng(0)
    n = 777
    pool = tiny_pool(rng.normal(size=(n, 3)), np.ones(n, int), rng.integers(0, 4, n))
    own = rm.own_tile_exclusion(pool.tile_code, 4)
    nearest = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=1)[1][:, 0]
    eps = rm.caliper(nearest)
    idx, _ = rm.match(pool, pool.z, pool.landcover, own, np.arange(n), k=5, eps=eps)
    unmatched = int((idx[:, 0] < 0).sum())
    check(abs(unmatched - round(0.05 * n)) <= 1
          and np.array_equal(idx[:, 0] >= 0, nearest <= eps),
          f"random pool, 4 tiles: {unmatched} unmatched (5% = {0.05 * n:.1f}, ties allow one "
          f"off); matched exactly when the nearest eligible candidate is within eps")


def check_region_tiles():
    print("2. tiles of the region are excluded")
    # Candidates: 0 in tile 1 (closest), 1 in tile 2 (next). The region holds points in tiles 0
    # and 1; the query lies in tile 0.
    pool = tiny_pool([[0.1], [0.5]], [1, 1], [1, 2])
    zq, lcq = np.array([[0.0]]), np.array([1])
    region_tiles = np.zeros((1, 3), bool)
    region_tiles[0, [0, 1]] = True
    idx, dist = rm.match(pool, zq, lcq, region_tiles, k=5, eps=1.0)
    check(idx[0].tolist() == [1, -1, -1, -1, -1] and np.isclose(dist[0, 0], 0.5),
          f"region-tiles: the closest candidate (in a region tile) is skipped: {idx[0].tolist()}")
    own = rm.own_tile_exclusion(np.array([0]), 3)
    idx, _ = rm.match(pool, zq, lcq, own, k=5, eps=1.0)
    check(idx[0].tolist()[:2] == [0, 1], f"own-tile: the same candidate is kept: {idx[0].tolist()}")


def check_landcover():
    print("3. land cover must match")
    pool = tiny_pool([[0.01], [0.3], [0.2]], [90, 42, 90], [1, 1, 1])
    idx, _ = rm.match(pool, np.array([[0.0]]), np.array([42]), no_exclusion(1, 2), k=5, eps=1.0)
    check(idx[0].tolist() == [1, -1, -1, -1, -1],
          f"the closer class-90 candidates are never matches: {idx[0].tolist()}")
    idx, _ = rm.match(pool, np.array([[0.0]]), np.array([71]), no_exclusion(1, 2), k=5, eps=1.0)
    check(idx[0, 0] == -1, "a class with no candidate leaves the query unmatched")


def check_five_closest():
    print("4. the 5 closest of 10 candidates inside eps")
    d = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])
    pool = tiny_pool(d[:, None], np.ones(10, int), np.ones(10, int))
    idx, dist = rm.match(pool, np.array([[0.0]]), np.array([1]), no_exclusion(1, 2), k=5,
                         eps=1.0)
    check(idx[0].tolist() == [9, 1, 5, 3, 7] and np.allclose(dist[0], [0.05, 0.1, 0.2, 0.3, 0.4]),
          f"kept {idx[0].tolist()}, sorted by distance")
    idx, _ = rm.match(pool, np.array([[0.0]]), np.array([1]), no_exclusion(1, 2), k=5, eps=0.1)
    check(idx[0].tolist() == [9, 1, -1, -1, -1], "a candidate at exactly eps is a match")


def check_eta_by_hand():
    print("5. eta by hand")
    pool_y = np.array([1, 0, 0, 1])
    idx = np.array([[0, 1, 2], [-1, -1, -1], [3, -1, -1]])
    y = np.array([1, 0, 0])
    share = rm.disagreement(y, idx, pool_y)
    check(np.isclose(share[0], 2 / 3) and np.isnan(share[1]) and share[2] == 1.0,
          f"per point: {np.round(share, 3).tolist()}")
    check(np.isclose(rm.eta(y[:2], idx[:2], pool_y), 2 / 3),
          "the unmatched point is left out: eta = 2/3")
    check(np.isclose(rm.eta(y, idx, pool_y), (2 / 3 + 1) / 2), "each matched point counts once")


def check_query_expansion():
    print("6. the k-cap bug: 600 nearer candidates in excluded tiles")
    rng = np.random.default_rng(1)
    near = rng.uniform(0, 0.5, (600, 2))
    pool = tiny_pool(np.vstack([near, [[0.9, 0.9]]]), np.ones(601, int), np.r_[np.zeros(600), 1])
    idx, dist = rm.match(pool, np.array([[0.0, 0.0]]), np.array([1]),
                         rm.own_tile_exclusion(np.array([0]), 2), k=5, eps=2.0)
    check(idx[0, 0] == 600 and np.isclose(dist[0, 0], np.hypot(0.9, 0.9)),
          f"the out-of-tile candidate is found ({idx[0, 0]}, distance {dist[0, 0]:.3f})")


def synthetic_world(tmp: Path, n_ood=400, n_id=400, n_models=20, n_missing=10, seed=2):
    """A cell, its dataset, FEATURES.txt and two regions tables in tmp. Returns the paths."""
    rng = np.random.default_rng(seed)
    tiles = np.array(["0_0", "0_1", "1_0", "1_1"])

    def half(n, prefix, domain, shift):
        tile = tiles[rng.integers(0, 4, n)]
        row, col = np.array([t.split("_") for t in tile], int).T
        frame = pd.DataFrame(dict(
            point_id=[f"{prefix}{i}" for i in range(n)], label=rng.integers(0, 2, n),
            domain=domain, tile_50km=tile,
            lon=-82 + 0.5 * col + rng.uniform(0, 0.45, n),
            lat=28 + 0.5 * row + rng.uniform(0, 0.45, n),
            county_fips=np.where(col == 0, "12001", "12003"), meta_point_county=None,
            landcover=rng.choice([42, 90], n), meta_dist_to_river_m=rng.uniform(10, 3000, n)))
        for j, f in enumerate(FEATURES):
            frame[f] = rng.normal(shift * (j + 1), 1 + j * 0.5, n)
        return frame

    ood, idd = half(n_ood, "a", "A", 1.0), half(n_id, "e", "AE", 0.0)
    idd.loc[:n_missing - 1, "wetindexcat"] = np.nan
    idd.loc[0, "hand"] = np.nan                         # missing in both features: one point
    pd.concat([ood, idd]).to_parquet(tmp / "dataset.parquet")
    (tmp / "FEATURES.txt").write_text("# test\nlandcover\n" + "\n".join(FEATURES) + "\n")
    S_ood = rng.uniform(0, 1, (n_models, n_ood))
    S_id = rng.uniform(0, 1, (n_models, n_id))
    np.savez_compressed(tmp / "SYN.npz", S_ood=S_ood, S_id=S_id, y_ood=ood.label.to_numpy(),
                        y_id=idd.label.to_numpy(), point_id=ood.point_id.to_numpy(),
                        point_id_id=idd.point_id.to_numpy(), lon=ood.lon.to_numpy(),
                        lat=ood.lat.to_numpy(), tile_50km=ood.tile_50km.to_numpy())
    pd.DataFrame(dict(idx=np.arange(n_models), kind=["rf", "lr"] * (n_models // 2),
                      auc_id=rng.uniform(0.6, 0.9, n_models),
                      acc_train_sub=np.r_[np.ones(3), rng.uniform(0.6, 0.9, n_models - 3)])
                 ).to_csv(tmp / "SYN_models.csv", index=False)
    # Regions: 25 points of tiles 0_0 and 0_1, 22 of tile 1_1, and a piece of 5 (too small).
    pick = []
    for want, size in ((["0_0", "0_1"], 25), (["1_1"], 22), (["1_0"], 5)):
        pool = ood.index[ood.tile_50km.isin(want) & ~ood.index.isin(sum(pick, []))]
        pick.append(list(rng.choice(pool, size, replace=False)))
    points = pd.concat([ood.loc[p].assign(component=c, component_size=len(p))
                        for c, p in enumerate(pick)])
    points["variant"] = "spatial_mu3_knn10"
    points.to_parquet(tmp / "regions.parquet")
    ood.sample(30, random_state=0).assign(component=0, component_size=30).to_parquet(
        tmp / "classic.parquet")
    return ood, idd


def args_for(tmp: Path, **kw):
    a = dict(cells_dir=str(tmp), tag="SYN", dataset=str(tmp / "dataset.parquet"),
             features=str(tmp / "FEATURES.txt"), regions=str(tmp / "regions.parquet"),
             classic=str(tmp / "classic.parquet"), extra_regions=[str(tmp / "regions.parquet")],
             exclude="region-tiles", min_size=20, max_matches=5, caliper_pct=95.0,
             counties=str(tmp / "no_counties.parquet"), out=str(tmp / "out"))
    a.update(kw)
    return argparse.Namespace(**a)


def check_inputs(tmp: Path, ood: pd.DataFrame, idd: pd.DataFrame):
    print("7-8. standardization over D_OOD; pool points missing a feature")
    ctx = rm.load_inputs(args_for(tmp))
    mean, sd = ood[FEATURES].mean(), ood[FEATURES].std()
    pool_mean = idd[FEATURES].dropna().mean()
    check(np.allclose(ctx["mean"], mean) and np.allclose(ctx["sd"], sd)
          and not np.allclose(ctx["mean"], pool_mean),
          "mean and SD are D_OOD's (ddof 1), not the pool's")
    kept = idd[idd[FEATURES].notna().all(axis=1)]
    check(np.allclose(ctx["pool"].z, ((kept[FEATURES] - mean) / sd).to_numpy()),
          "pool z = (x - D_OOD mean) / D_OOD SD")
    check(ctx["n_dropped"] == 10 and len(ctx["pool"]) == len(idd) - 10
          and ctx["missing"] == {"wetindexcat": 10, "hand": 1},
          f"10 pool points dropped and counted: {ctx['n_dropped']}, {ctx['missing']}")
    check(rm.numeric_features(tmp / "FEATURES.txt") == FEATURES, "the 6 numeric features")
    (tmp / "BAD.txt").write_text("landcover\nslope\n")
    try:
        rm.numeric_features(tmp / "BAD.txt")
        raised = False
    except SystemExit:
        raised = True
    check(raised, "a feature list without 6 numeric features stops the run")
    return ctx


def check_helpers(tmp: Path, ctx: dict):
    print("helpers")
    check(rm.caliper(np.array([0.1, 0.2, 0.3, np.inf]), 50) == 0.25,
          "caliper: an infinite distance sorts last")
    try:
        rm.caliper(np.array([0.1, np.inf, np.inf]), 95)
        raised = False
    except ValueError:
        raised = True
    check(raised, "caliper: a percentile among the points without a candidate raises")
    check(rm.weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0
          and rm.weighted_median(np.array([1.0, 2.0]), np.array([0.5, 0.5])) == 1.0
          and np.isnan(rm.weighted_median(np.array([np.nan]), np.array([1.0]))),
          "weighted median (lower) and NaN without values")
    vals = np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
    means = rm.match_means(vals, np.array([[0, 1], [2, -1], [-1, -1]]))
    check(np.allclose(means[:2], [[2.0, 4.0], [8.0, 10.0]]) and np.isnan(means[2]).all(),
          "match_means averages over each query's matches")
    pts = pd.DataFrame(dict(meta_point_county=[None, "12099", None],
                            county_fips=["12051", "12051", "12051"]))
    check(rm.region_name(pts, {"12051": "Hendry", "12099": "Palm Beach"}) == "Hendry/Palm Beach",
          "region name: counties by FIPS")
    models = pd.DataFrame(dict(auc_id=np.arange(20) / 20))
    _, _, test = op.model_split(20)
    check(set(rm.strongest_test_models(models)) == set(test),
          "strongest models are test models (all 4 of 4 here)")

    points = pd.read_parquet(tmp / "regions.parquet")
    regions = rm.regions_of(ctx, points, 20, {})
    check([r["component"] for r in regions] == [0, 1] and len(regions[0]["rows"]) == 25,
          "regions_of keeps the pieces of >= 20 points")
    ex = rm.region_exclusion(ctx, regions[0], "region-tiles")
    own = rm.region_exclusion(ctx, regions[0], "own-tile")
    tiles = ctx["ood_tile"][regions[0]["rows"]]
    check(ex[:, regions[0]["tiles"]].all() and ex.sum(1).max() == 2
          and np.array_equal(own, rm.own_tile_exclusion(tiles, ctx["n_tiles"])),
          "region-tiles excludes both of the region's tiles; own-tile only the point's")

    q = rm.queries(ctx, regions[0]["rows"])
    wrong = ctx["Z_ood"][ctx["strong"]][:, regions[0]["rows"]] == 0
    y0 = ctx["ood"].label.to_numpy()[regions[0]["rows"]]
    check(q["fn15"] == int((wrong & (y0 == 1)).sum()) and q["fp15"] == int((wrong & (y0 == 0)).sum())
          and q["fn15"] + q["fp15"] == int(wrong.sum()),
          f"queries count the strongest models' FNs and FPs ({q['fn15']} and {q['fp15']})")
    hard = rm.hardest_rows(ctx, n=50)
    _, _, test = op.model_split(ctx["Z_ood"].shape[0])
    score = 1 - ctx["Z_ood"][test].mean(0)
    check(len(hard) == 50 and score[hard].min() >= np.sort(score)[-50],
          "the hardest rows have the highest error share of the test models")

    selected = np.zeros(len(ctx["ood"]), bool)
    selected[rm.selection_rows(ctx, points)] = True
    halos = list(rm.halo_sets(ctx, regions, selected, "region-tiles"))
    km, rows, excluded = halos[-1]
    ok = not selected[rows].any() and len(rows) > 0
    for r, col in enumerate(rows):
        ok &= excluded[r, ctx["ood_tile"][col]]
    sizes = [len(h[1]) for h in halos]
    check(ok and sizes == sorted(sizes),
          f"halos: outside the selection, own tile excluded, growing with the radius {sizes}")

    kinds = rm.kinds_table(ctx, regions, {r["component"]: np.full((len(r["rows"]), 5), -1)
                                          for r in regions})
    below = int((ctx["cell"].models.acc_train_sub.to_numpy()[test] < rm.INTERPOLATES).sum())
    check(np.allclose(kinds.fn_share + kinds.fp_share, 1) and (kinds.n_test == len(test)).all()
          and (kinds.n_held_out == below).all() and (kinds.unmatched == 1).all(),
          f"kinds: FN + FP shares = 1; the held-out models are the {below} test models below "
          f"99% training accuracy")


def check_run(tmp: Path):
    print("end to end on the synthetic cell")
    out = rm.run(args_for(tmp))
    t = out["table"]
    kinds = t.kind.value_counts().to_dict()
    check(kinds == {"region": 2, "reference": 3, "halo": 3, "extra_region": 2, "baseline": 1,
                    "pooled": 1}, f"rows per kind: {kinds}")
    base = t[t.kind == "baseline"].iloc[0]
    check(abs(base.unmatched - 0.05) < 0.01, f"baseline unmatched {base.unmatched:.3f}")
    pooled = t[t.kind == "pooled"].iloc[0]
    reg = t[t.kind == "region"]
    check(pooled.n == reg.n.sum() and pooled.n_matched == reg.n_matched.sum()
          and np.isclose(pooled.eta, (reg.eta * reg.n_matched).sum() / reg.n_matched.sum()),
          "the pooled row stacks the regions' own matches")
    check(len(out["sensitivity"]) == 1 + len(rm.SENSITIVITY), "sensitivity rows for every setting")
    check(t.fn_share15.between(0, 1).all()
          and out["kinds"].pos_matches.dropna().between(0, 1).all(),
          "FN shares of the strongest models' errors and the kinds' match positives are shares")
    p90 = out["sensitivity"].set_index("setting").loc["caliper_pct"]
    check(np.allclose(p90.baseline_unmatched.to_numpy(), [0.10, 0.01], atol=0.01),
          f"eps at the 90th and 99th percentiles leave 10% and 1% unmatched "
          f"{np.round(p90.baseline_unmatched.to_numpy(), 3).tolist()}")
    own = rm.run(args_for(tmp, exclude="own-tile"))["table"]
    r0 = own[own.kind == "region"].iloc[0]
    check(r0.n_matched >= reg.iloc[0].n_matched,
          "own-tile never matches fewer points than region-tiles")


def check_real():
    print("9. the sandbox numbers (Stop S2), --exclude own-tile on the 2026-10-23 cell")
    if not all(Path(p).exists() for p in REAL.values()):
        print("  skip  2026-10-23 cell or regions missing")
        return
    a = argparse.Namespace(cells_dir=REAL["cells"], tag="FL_band_AE_s0", dataset=REAL["dataset"],
                           features=REAL["features"], regions=REAL["regions"],
                           classic=REAL["classic"], extra_regions=None, exclude="own-tile",
                           min_size=20, max_matches=5, caliper_pct=95.0,
                           counties=str(rm.COUNTIES), out="unused")
    out = rm.run(a)
    t = out["table"].set_index("set")
    check(abs(out["summary"]["eps"] - 1.391) <= 0.01, f"eps {out['summary']['eps']:.4f} (1.391)")
    for name, (unmatched, eta) in SANDBOX.items():
        got_u, got_e = t.loc[name, "unmatched"], t.loc[name, "eta"]
        ok = abs(got_e - eta) <= 0.01 and (unmatched is None or abs(got_u - unmatched) <= 0.01)
        check(ok, f"{name}: unmatched {got_u:.3f} ({unmatched}), eta {got_e:.3f} ({eta})")


def main():
    check_caliper_share()
    check_region_tiles()
    check_landcover()
    check_five_closest()
    check_eta_by_hand()
    check_query_expansion()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ood, idd = synthetic_world(tmp)
        ctx = check_inputs(tmp, ood, idd)
        check_helpers(tmp, ctx)
        check_run(tmp)
    check_real()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
