#!/usr/bin/env python3
"""
E1 -- the coherence/inversion tradeoff curve. This is the core result.

For every cell, model split, budget f and smoothness weight mu, run the
selection and score it on models the selection never saw, alongside the three
controls, and record the coherence of every mask.

THE CONTROLS ARE THE RESULT
---------------------------
OODSelect always returns a subset; the controls decide whether it means
anything.

  random       a random subset of the same size. Must stay POSITIVE -- that is
               what makes aggregate accuracy-on-the-line an averaging artifact
               rather than a fact about every example. It doubles as the
               coherence null, since its purity and cut fraction are the values
               a structureless mask of that size gets.
  hard         the most-misclassified examples, built from the SELECTION models
               only. Near zero or positive means the inversion is not merely
               difficulty; strongly negative means it is, and the cell is
               reported as contaminated.
  null-select  the identical optimisation against SHUFFLED ID performance,
               scored against true ID performance on the reporting models. Run
               AT THE SAME MU, because the spatial constraint changes what the
               optimiser can manufacture out of noise, so a null that omits it
               is not the right null. If this inverts, nothing else in the row
               means anything.

Model splits: the paper's 60/20/20, plus two family-disjoint splits. A random
split does not make the halves independent when the population varies mainly
along overall quality; disjoint learner families break that shared axis.

MU IS NEVER CHOSEN BY TEST r. That would be selection on the outcome. This
script writes the whole curve, and the operating point is the smallest mu whose
PURITY clears a threshold. Purity is a property of the mask and the spatial
graph -- no model is involved in computing it -- so choosing on it cannot leak
the reported correlation, whichever models end up scoring the subset. That is
the reason the coherence metrics, not r, define the operating point.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph                                              # noqa: E402
import oodselect as oo                                    # noqa: E402
from cells import load_cell, cell_tag, result_tag         # noqa: E402


def build_graph(cell, knn: int, graph_kind: str):
    """Adjacency for one cell, with the group restriction the kind implies."""
    groups = None
    if graph_kind == "block":
        groups = cell.block_id
    elif graph_kind == "catchment":
        groups = cell.meta_comid
    elif graph_kind == "provenance":
        src = cell.meta_source_cit
        ok = src is not None and np.mean(
            [s not in ("", "nan", "<NA>", "None") for s in np.asarray(src).astype(str)]) > 0.5
        if not ok:
            print("  note: meta_source_cit is mostly missing -- falling back to "
                  "county_fips for the provenance graph", flush=True)
            groups = cell.county_fips
        else:
            groups = src
    A = graph.build_adjacency(cell.lon, cell.lat, k=knn, groups=groups)
    return A, graph.laplacian(A)


def sweep_cell(cell, mus, fracs, knn, graph_kind, seed, iters, restarts,
               device, n_perm_moran=0) -> list[dict]:
    cell.require_spatial()
    A, L = build_graph(cell, knn, graph_kind)
    x_all = cell.x
    S = cell.S_ood
    d = S.shape[1]
    tiles = cell.tile_50km
    blocks = cell.block_id
    rows = []
    keep_masks: dict = {}          # (split, frac) -> {mu: mask}, for the Jaccard

    for split_name, (i_sel, i_val, i_rep) in cell.splits(seed).items():
        if len(i_sel) < 10 or len(i_rep) < 5:
            continue
        x_sel, S_sel = x_all[i_sel], S[i_sel]
        x_val = x_all[i_val] if len(i_val) else None
        S_val = S[i_val] if len(i_val) else None
        x_rep, S_rep = x_all[i_rep], S[i_rep]
        r_full = oo.evaluate(x_rep, S_rep, np.ones(d, bool))

        for f in fracs:
            k_target = max(20, int(d * f))
            t0 = time.time()

            # All mu values for this (split, f) solve as one batch -- they are
            # independent problems over the same S, so the GPU sees one big
            # mat-vec per iteration instead of len(mus) small ones.
            masks, info = oo.fit(x_sel, S_sel, k_target, x_val=x_val, S_val=S_val,
                                 L=L, mus=mus, knn=knn, iters=iters,
                                 restarts=restarts, seed=seed, device=device)

            rng = np.random.default_rng(seed + 991)
            x_sh = oo.shuffled_x(x_sel, rng)
            null_masks, _ = oo.solve(x_sh, S_sel, k_target, L=L, mus=mus, knn=knn,
                                     iters=iters, restarts=restarts, seed=seed,
                                     device=device)

            hard = oo.baseline_hard(S_sel, k_target)
            rnd = oo.baseline_random(d, k_target, rng)

            for j, mu in enumerate(np.atleast_1d(mus)):
                for method, mask in (("oodselect", masks[j]),
                                     ("nullsel", null_masks[j]),
                                     ("hard", hard), ("random", rnd)):
                    ev = oo.evaluate(x_rep, S_rep, mask)
                    mt = graph.coherence_metrics(
                        mask, A, knn, tiles=tiles, blocks=blocks,
                        values=None)
                    rows.append(dict(
                        cell=cell.tag, tag=result_tag(*cell.tag.split("_")[:3],
                                                      int(cell.tag.split("_s")[-1]),
                                                      float(mu), graph_kind),
                        split=split_name, graph=graph_kind, knn=knn,
                        frac=f, k=k_target, mu=float(mu), method=method,
                        n_sel_models=len(i_sel), n_rep_models=len(i_rep),
                        r_full=r_full["r"], r=ev["r"], p=ev["p"], rho=ev["rho"],
                        r_train=info[j]["r_train"] if method == "oodselect" else np.nan,
                        r_val=info[j].get("r_val", np.nan) if method == "oodselect" else np.nan,
                        purity=mt["purity"], P_hard=mt["P_hard"],
                        n_cc=mt["n_cc"], largest_cc_frac=mt["largest_cc_frac"],
                        tiles=mt["tiles"], blocks=mt["blocks"],
                        moran_I=mt["moran_I"],
                    ))
            keep_masks[(split_name, f)] = {float(mu): masks[j]
                                           for j, mu in enumerate(np.atleast_1d(mus))}
            print(f"    [{split_name}] f={f:.2f} k={k_target:,} "
                  f"{len(np.atleast_1d(mus))} mu in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    return _add_jaccard(df, keep_masks, fracs).to_dict("records")


def _add_jaccard(df: pd.DataFrame, keep_masks: dict, fracs) -> pd.DataFrame:
    """Subset stability across budgets, as the paper reports it.

    Each selected subset is compared against the SMALLEST-budget subset from the
    same split and mu, using the chance-normalised Jaccard index in graph.py: 1.0
    means no more overlap than two random masks of those sizes, and higher means
    genuinely the same points. A method that returns a different set every time
    the budget moves is not locating a stable property of the data, whatever its
    correlation is.
    """
    df["jaccard_vs_smallest"] = np.nan
    if not keep_masks:
        return df
    f0 = min(fracs)
    for (split_name, f), by_mu in keep_masks.items():
        base = keep_masks.get((split_name, f0))
        if base is None:
            continue
        for mu, mk in by_mu.items():
            if mu not in base:
                continue
            j = graph.jaccard(mk, base[mu])
            sel = ((df.split == split_name) & (df.frac == f)
                   & (df.mu == mu) & (df.method == "oodselect"))
            df.loc[sel, "jaccard_vs_smallest"] = j
    return df


def coherent_operating_point(df: pd.DataFrame, purity_min: float = 0.5) -> pd.DataFrame:
    """Smallest mu whose SELECTION-side purity clears the threshold, per cell.

    Chosen on coherence, never on test r. Everything reported at this point is
    still the test-model number; only the CHOICE is made on the selection side.
    """
    out = []
    sub = df[(df.method == "oodselect") & (df.split == "random-split")]
    for (cell, f), g in sub.groupby(["cell", "frac"]):
        g = g.sort_values("mu")
        ok = g[g.purity >= purity_min]
        row = (ok.iloc[0] if len(ok) else g.iloc[-1]).to_dict()
        row["reached_purity"] = bool(len(ok))
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="data/interim/aotil/cells")
    ap.add_argument("--out", default="results/aotil")
    ap.add_argument("--state", default="FL")
    ap.add_argument("--protocol", default="band")
    ap.add_argument("--train-dom", default="AE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mus", default="0,0.1,0.3,1,3,10")
    ap.add_argument("--fracs", default="0.05,0.10,0.25")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--graph", default="knn",
                    choices=["knn", "block", "catchment", "provenance"])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--purity-min", type=float, default=0.5)
    a = ap.parse_args()

    mus = [float(v) for v in a.mus.split(",")]
    fracs = [float(v) for v in a.fracs.split(",")]
    tag = cell_tag(a.state, a.protocol, a.train_dom, a.seed)
    cell = load_cell(a.cells, tag)

    print(f"=== E1 mu sweep: {tag} ===")
    print(f"  {cell.n_models} models | {cell.d:,} OOD points | graph={a.graph} "
          f"k={a.knn} | device={a.device}")
    rows = sweep_cell(cell, mus, fracs, a.knn, a.graph, a.seed, a.iters,
                      a.restarts, a.device)
    df = pd.DataFrame(rows)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{tag}_{a.graph}_sweep.csv"
    df.to_csv(dest, index=False)

    pd.set_option("display.width", 220)
    show = df[df.split == "random-split"].pivot_table(
        index=["frac", "mu"], columns="method",
        values=["r", "purity", "n_cc"]).round(3)
    print("\n" + show.to_string())

    cop = coherent_operating_point(df, a.purity_min)
    if len(cop):
        print(f"\n--- coherent operating point (smallest mu with purity >= "
              f"{a.purity_min}; purity is model-free, so this is not selection "
              f"on the outcome) ---")
        print(cop[["cell", "frac", "mu", "r", "purity", "n_cc",
                   "largest_cc_frac", "tiles", "reached_purity"]]
              .round(3).to_string(index=False))
        cop.to_csv(out / f"{tag}_{a.graph}_operating_point.csv", index=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
