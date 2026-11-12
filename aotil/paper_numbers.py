#!/usr/bin/env python3
"""Every table and number of the paper's Experiments, Analysis and Appendix, from one run folder of
run_experiments.py (no torch, no new computation beyond arithmetic on the stored results). make_paper.py calls it.

  load(run)        the stage outputs of a run folder (a missing stage reads as None)
  tables(R)        Table tab:shift, tab:search, tab:regions and tab:models: the paper's tabular blocks, exactly in its
                   format (leading zeros, $-$ signs, \\quad + smoothing rows, # Comps cells like 7 (383)), and the
                   same rows as plain text for the terminal
  numbers(R)       every number the prose states, by paragraph, as the paper prints it, each with the phrase the
                   paper should contain (checked by paper_check)
  mu_sweep(R)      the mu sweep table (loss x mu, alone and + smoothing) and the values of the "Regularization
                   weight" paragraph, with the answers to its % CHECK
  checks(R)        self-checks of the run (PASS / FAIL)
  paper_check(...) the generated tables and phrases against CS371_Cho_Li/acl_latex.tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPORTED_S, MU, C_MIN = 500, 3, 20
SEARCHES = ("OODSelect", "Laplacian", "GTV")
SEARCH_ROWS = [("OODSelect", "OODSelect", False), (r"\quad + smoothing", "OODSelect", True),
               ("Laplacian", "Laplacian", False), (r"\quad + smoothing", "Laplacian", True),
               ("GTV", "GTV", False), (r"\quad + smoothing", "GTV", True)]
PAPER_SETTINGS = dict(trials=100, max_steps=50_000, patience=500, knn=10, c=20, mu=3, search_seed=0,
                      reported_S=500, budget_penalty="squared")
# Appendix Table tab:models: each family's grid as the paper words it (zoo.GRIDS), in the paper's row order.
MODEL_ROWS = [
    ("dt", "Decision tree", r"maximum depth $\in\{1,2,3,4,5,6,8,10,12,\text{none}\}$ $\times$ minimum leaf size "
                            r"$\in\{1,20,200,2000\}$"),
    ("rf", "Random forest", r"trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ "
                            r"$\times$ features per split $\in\{1,5,28\}$"),
    ("et", "Extra trees", r"trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ "
                          r"$\times$ features per split $\in\{1,5,28\}$"),
    ("hgb", "Histogram gradient boosting", r"learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ iterations "
                                           r"$\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,\text{none}\}$, "
                                           r"without early stopping"),
    ("xgb", "XGBoost", r"learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ trees $\in\{3,10,50,300\}$ $\times$ "
                       r"maximum depth $\in\{1,3,8\}$, histogram tree method"),
    ("lr", "Logistic regression", r"inverse regularization strength $C\in\{0.001,0.01,0.1,1,100\}$ $\times$ "
                                  r"iterations $\in\{1,2,3,5,100\}$ $\times$ class weights "
                                  r"$\in\{\text{none},\text{balanced}\}$"),
    ("mlp", "Multilayer perceptron", r"hidden layers $\in\{(2),(8),(16),(32),(64,32)\}$ $\times$ epochs "
                                     r"$\in\{3,20,200\}$ $\times$ learning rate $\in\{10^{-4},10^{-3},10^{-2}\}$"),
    ("nb", "Gaussian naive Bayes", r"variance smoothing $\in\{10^{-9},10^{-5},10^{-4},10^{-3},0.003,0.01,0.03,"
                                   r"\allowbreak 0.1,0.2,0.3,0.5,1,2,3,10\}$ $\times$ class priors "
                                   r"$\in\{\text{fitted},\text{equal}\}$"),
]
NLCD_NAMES = {11: "open water", 21: "developed open space", 22: "low-intensity developed",
              23: "medium-intensity developed", 24: "high-intensity developed", 31: "barren land",
              41: "deciduous forest", 42: "evergreen forest", 43: "mixed forest", 52: "shrub and scrub",
              71: "grassland", 81: "pasture and hay", 82: "crops", 90: "woody wetland", 95: "herbaceous wetland"}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve"}


# ------------------------------------------------------------------ formats
def sgn(x, nd=2, tex=True):
    """'$-$0.97' / '$+$0.84' (signed cells; leading zero kept)."""
    m, p = ("$-$", "$+$") if tex else ("-", "+")
    return (m if x < 0 else p) + f"{abs(x):.{nd}f}"


def gap(x, nd=3, tex=True):
    """'$-$0.034' / '0.017' (minus sign only), as the shift table prints Delta."""
    return (("$-$" if tex else "-") if x < 0 else "") + f"{abs(x):.{nd}f}"


def share(x, nd=2):
    return f"{x:.{nd}f}"


def signed_prose(x, nd=2):
    """'$-0.97$' inside math, as the prose writes negative correlations."""
    return f"${x:.{nd}f}$" if x < 0 else f"{x:.{nd}f}"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def word(n: int) -> str:
    return NUMBER_WORDS.get(int(n), str(int(n)))


def thousands(n: int) -> str:
    return f"{int(n):,}"


def tex_thousands(n: int) -> str:
    """10,000 -> 10{,}000 as the paper writes numbers in math mode."""
    return f"{int(n):,}".replace(",", "{,}")


# ------------------------------------------------------------------ loading
def load(run: Path) -> dict:
    """Every stage output of a run folder; a missing one is None."""
    run = Path(run)

    def js(p):
        p = run / p
        return json.loads(p.read_text()) if p.exists() else None

    def csv(p):
        p = run / p
        return pd.read_csv(p) if p.exists() else None
    R = dict(run=run, manifest=js("manifest.json"), zoo=js("zoo/zoo.json"), data=js("data/data.json"),
             shift=js("shift/summary.json"), per_tile=csv("shift/per_tile.csv"), subsets=csv("smooth/subsets.csv"),
             components=csv("regions/components.csv"), match=csv("match/table.csv"),
             match_summary=js("match/summary.json"))
    if R["subsets"] is not None:
        R["subsets"]["smoothing"] = R["subsets"].smoothing.astype(bool)
    return R


def missing_stages(R: dict) -> list[str]:
    need = dict(zoo="zoo", data="data", shift="shift", subsets="smooth", components="regions", match="match")
    return [stage for key, stage in need.items() if R.get(key) is None]


def pick(R, search, S=REPORTED_S, smoothing=False, mu=None):
    """One subset row: the search at S (mu = 3 for the two losses unless given)."""
    t = R["subsets"]
    if t is None:
        return None
    if mu is None:
        mu = 0 if search == "OODSelect" else MU
    g = t[(t.search == search) & (t.S == S) & (t.smoothing == smoothing) & np.isclose(t.mu, mu)]
    return None if g.empty else g.iloc[0]


def n_ood(R) -> int | None:
    """m, the number of Zone A test points (from the zoo, else from the data stage)."""
    if R.get("zoo"):
        return int(R["zoo"]["n_ood"])
    if R.get("data"):
        return int(R["data"]["sets"]["OOD"]["n"])
    return None


def budget_grid(R) -> pd.DataFrame | None:
    t = R["subsets"]
    if t is None:
        return None
    return t[t.group == "budget grid"]


# ------------------------------------------------------------------ tables
def table_shift(R):
    s = R["shift"]["metrics"]
    names = dict(acc="accuracy", auc="AUROC", f1="F1")
    head = ["Metric", "Pooled", "Per tile"]
    tex_rows, txt_rows = [], []
    for k, m in enumerate(("acc", "auc", "f1")):
        r = s[m]
        nm = names[m]
        cap = nm[0].upper() + nm[1:]
        for label, pooled, tile, ci, is_gap in [
                (f"ID {nm}", r["pooled_id"], r["tile_id"], r["ci_id"], False),
                (f"OOD {nm}", r["pooled_ood"], r["tile_ood"], r["ci_ood"], False),
                (f"$\\Delta$ {cap}", r["pooled_delta"], r["tile_delta"], r["ci_delta"], True)]:
            f = (lambda x, t=True: gap(x, 3, t)) if is_gap else (lambda x, t=True: share(x, 3))
            tex_rows.append(f"{label} & {f(pooled)} & {f(tile)} [{f(ci[0])}, {f(ci[1])}] \\\\")
            txt_rows.append([label.replace("$\\Delta$", "Delta"), f(pooled, False),
                             f"{f(tile, False)} [{f(ci[0], False)}, {f(ci[1], False)}]"])
        if k < 2:
            tex_rows.append(r"\midrule")
            txt_rows.append(None)
    tex = "\n".join([r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
                     r"\textbf{Metric} & \textbf{Pooled} & \textbf{Per tile} \\", r"\midrule", *tex_rows,
                     r"\bottomrule", r"\end{tabular}"])
    return dict(name="tab_shift", label="tab:shift",
                title=f"Table 1 (tab:shift): shift from Zone AE (ID) to Zone A (OOD), mean over all "
                      f"{R['shift']['n_models']} models; per tile: mean over {s['acc']['n_tiles']} test tiles "
                      f"({s['auc']['n_tiles']} for AUROC) with 95% t-intervals",
                head=head, rows=txt_rows, tex=tex)


def table_search(R):
    r_all, (lo_all, hi_all) = R["shift"]["corr_te_all"], R["shift"]["corr_te_all_ci"]
    tex_rows = [f"All Zone A & {sgn(r_all)} [{sgn(lo_all)}, {sgn(hi_all)}] & -- & -- \\\\"]
    txt_rows = [["All Zone A", f"{sgn(r_all, 2, False)} [{sgn(lo_all, 2, False)}, {sgn(hi_all, 2, False)}]", "--", "--"]]
    for label, search, sm in SEARCH_ROWS:
        x = pick(R, search, REPORTED_S, sm)
        if x is None:
            tex_rows.append(f"{label} & \\textbf{{XX}} & \\textbf{{XX}} & \\textbf{{XX}} \\\\")
            txt_rows.append([label.replace(r"\quad ", "  "), "missing", "missing", "missing"])
            continue
        tex_rows.append(f"{label} & {sgn(x.r_te)} [{sgn(x.ci_lo)}, {sgn(x.ci_hi)}] & {int(x.components)} "
                        f"({int(x.comp_points)}) & {share(x.purity)} \\\\")
        txt_rows.append([label.replace(r"\quad ", "  "),
                         f"{sgn(x.r_te, 2, False)} [{sgn(x.ci_lo, 2, False)}, {sgn(x.ci_hi, 2, False)}]",
                         f"{int(x.components)} ({int(x.comp_points)})", share(x.purity)])
    head = r"\textbf{Points} & $\boldsymbol{\text{corr}_\text{te}}$ \textbf{[95\% CI]} & \textbf{\# Comps} & \textbf{Purity} \\"
    tex = "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule", head, r"\midrule", *tex_rows,
                     r"\bottomrule", r"\end{tabular}"])
    return dict(name="tab_search", label="tab:search",
                title=f"Table 2 (tab:search): subset search at S = {REPORTED_S}; corr_te on the "
                      f"{R['shift']['n_test_models']} test models; # Comps = components of >= {C_MIN} points (points "
                      f"they hold)", head=["Points", "corr_te [95% CI]", "# Comps", "Purity"], rows=txt_rows, tex=tex)


def region_rows(R) -> pd.DataFrame:
    m = R["match"].copy()
    reg = m[m.kind == "region"].copy()
    reg["n_sort"] = -reg.n
    return reg.sort_values(["n_sort", "name"]).drop(columns="n_sort")


def table_regions(R):
    m = R["match"]
    tex_rows, txt_rows = [], []

    def line(name, n, un, eta):
        tex_rows.append(f"{name} & {n:,} & {share(un)} & {share(eta)} \\\\")
        txt_rows.append([name, f"{n:,}", share(un), share(eta)])
    for _, x in region_rows(R).iterrows():
        line(x["name"], int(x.n), x.unmatched, x.eta)
    tex_rows.append(r"\midrule")
    txt_rows.append(None)
    for kind, label in (("pooled", "All regions"), ("other", "Other Zone A"), ("baseline", "Zone AE baseline")):
        x = m[m.kind == kind].iloc[0]
        line(label, int(x.n), x.unmatched, x.eta)
    tex = "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
                     r"\textbf{Region} & \textbf{\# Points} & \textbf{Unmatched} & $\boldsymbol{\eta}$ \\",
                     r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}"])
    s = R["match_summary"]
    return dict(name="tab_regions", label="tab:regions",
                title=f"Table 3 (tab:regions): matched inputs of the regions of GTV + smoothing at S = {REPORTED_S} "
                      f"(caliper eps = {s['eps']:.2f} on q = {s['q']} features, at most {s['max_matches']} matches)",
                head=["Region", "# Points", "Unmatched", "eta"], rows=txt_rows, tex=tex)


def table_models(R):
    fam = R["zoo"]["families"]
    tex_rows, txt_rows = [], []
    for code, name, grid in MODEL_ROWS:
        f = fam.get(code, dict(settings=0, models=0))
        tex_rows.append(f"{name} & {grid} & {f['settings']} & {f['models']} \\\\")
        txt_rows.append([name, str(f["settings"]), str(f["models"])])
    tex_rows += [r"\midrule", f"Total & & {R['zoo']['settings']} & {R['zoo']['kept']} \\\\"]
    txt_rows += [None, ["Total", str(R["zoo"]["settings"]), str(R["zoo"]["kept"])]]
    tex = "\n".join([r"\begin{tabular}{@{}lp{0.6\textwidth}rr@{}}", r"\toprule",
                     r"\textbf{Family} & \textbf{Grid} & \textbf{Settings} & \textbf{Models} \\", r"\midrule",
                     *tex_rows, r"\bottomrule", r"\end{tabular}"])
    return dict(name="tab_models", label="tab:models",
                title="Appendix Table tab:models: grid settings and models kept per family",
                head=["Family", "Settings", "Models"], rows=txt_rows, tex=tex)


def tables(R) -> list[dict]:
    out = []
    if R["shift"] is not None:
        out.append(table_shift(R))
        if R["subsets"] is not None:
            out.append(table_search(R))
    if R["match"] is not None and R["match_summary"] is not None:
        out.append(table_regions(R))
    if R["zoo"] is not None:
        out.append(table_models(R))
    return out


def render(table: dict) -> str:
    """A table as aligned plain text with its title."""
    rows = [r for r in table["rows"] if r is not None]
    widths = [max(len(str(c)) for c in col) for col in zip(table["head"], *rows)]
    sep = "  ".join("-" * w for w in widths)

    def fmt(r):
        return "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    lines = [table["title"], sep, fmt(table["head"]), sep]
    for r in table["rows"]:
        lines.append(sep if r is None else fmt(r))
    lines.append(sep)
    return "\n".join(lines)


# ------------------------------------------------------------------ the mu sweep
def mu_sweep(R) -> dict | None:
    """The mu sweep at S = 500: one row per (loss, mu, smoothing), the OODSelect reference, the XX values of the
    "Regularization weight" paragraph and the answers to its % CHECK."""
    t = R["subsets"]
    if t is None:
        return None
    sw = t[(t.S == REPORTED_S) & t.search.isin(["Laplacian", "GTV"])].sort_values(["search", "smoothing", "mu"])
    if sw.empty:
        return None
    ref = t[(t.search == "OODSelect") & (t.S == REPORTED_S)].set_index("smoothing")
    out = dict(table=sw, reference=ref)
    plain = sw[~sw.smoothing]
    grid = sorted(plain.mu.unique())
    out["mus"] = grid
    out["complete"] = all(len(plain[plain.search == s]) == 9 for s in ("Laplacian", "GTV"))
    trend = {}
    for s in ("GTV", "Laplacian"):
        g = plain[plain.search == s].sort_values("mu")
        if len(g) < 2:
            continue
        from scipy.stats import spearmanr
        trend[s] = dict(mu_lo=float(g.mu.iloc[0]), mu_hi=float(g.mu.iloc[-1]),
                        purity_lo=float(g.purity.iloc[0]), purity_hi=float(g.purity.iloc[-1]),
                        r_lo=float(g.r_te.iloc[0]), r_hi=float(g.r_te.iloc[-1]),
                        purity_rises=bool(np.all(np.diff(g.purity.to_numpy()) >= 0)),
                        purity_spearman=float(spearmanr(g.mu, g.purity)[0]),
                        purity_drops=[f"{a:g}->{b:g}" for a, b, d in zip(g.mu[:-1], g.mu[1:],
                                                                         np.diff(g.purity.to_numpy())) if d < 0],
                        r_weakens=bool(np.all(np.diff(g.r_te.to_numpy()) >= 0)),
                        r_spearman=float(spearmanr(g.mu, g.r_te)[0]),
                        r_strengthens=[f"{a:g}->{b:g}" for a, b, d in zip(g.mu[:-1], g.mu[1:],
                                                                          np.diff(g.r_te.to_numpy())) if d < 0])
    out["trend"] = trend
    out["matched"] = matched_purity(plain)
    return out


def matched_purity(plain: pd.DataFrame) -> dict | None:
    """corr_te of the two losses at the same purity: each loss's corr_te interpolated linearly in purity (points
    sorted by purity) over the purities both reach; the mean and range of GTV minus Laplacian."""
    curves = {}
    for s in ("GTV", "Laplacian"):
        g = plain[plain.search == s].sort_values("purity")
        if len(g) < 2:
            return None
        curves[s] = (g.purity.to_numpy(), g.r_te.to_numpy())
    lo = max(c[0].min() for c in curves.values())
    hi = min(c[0].max() for c in curves.values())
    if hi <= lo:
        return dict(overlap=None)
    grid = np.linspace(lo, hi, 101)
    diff = np.interp(grid, *curves["GTV"]) - np.interp(grid, *curves["Laplacian"])
    stronger = "GTV" if diff.mean() < 0 else "Laplacian"
    at_points = []
    for s, other in (("GTV", "Laplacian"), ("Laplacian", "GTV")):
        p, r = curves[s]
        for pi, ri in zip(p, r):
            if lo <= pi <= hi:
                at_points.append(dict(search=s, purity=float(pi), r_te=float(ri),
                                      other_r_te=float(np.interp(pi, *curves[other]))))
    return dict(overlap=(float(lo), float(hi)), mean_gtv_minus_lap=float(diff.mean()),
                min_gtv_minus_lap=float(diff.min()), max_gtv_minus_lap=float(diff.max()),
                share_gtv_stronger=float((diff < 0).mean()), stronger=stronger, at_points=at_points)


# ------------------------------------------------------------------ numbers
def numbers(R) -> list[tuple[str, list[dict]]]:
    """(paragraph, items); item = dict(label, value, phrase): value as the paper prints it, phrase the text the
    paper should contain with this value (None: nothing to check)."""
    secs = []

    def item(label, value, phrase=None, raw=None):
        return dict(label=label, value=value, phrase=phrase, raw=raw)
    z, d, s = R["zoo"], R["data"], R["shift"]
    gtv_sm = pick(R, "GTV", REPORTED_S, True)
    ood = pick(R, "OODSelect", REPORTED_S, False)
    reg = region_rows(R) if R["match"] is not None else None
    pooled = R["match"][R["match"].kind == "pooled"].iloc[0] if R["match"] is not None else None

    if z and s and gtv_sm is not None and ood is not None and pooled is not None:
        a = s["metrics"]["acc"]
        secs.append(("Introduction", [
            item("models", str(z["kept"]), f"We train {z['kept']} tabular models"),
            item("per-tile accuracy loss on Zone A", gap(a["tile_delta"], 3, False),
                 f"models lose {gap(a['tile_delta'], 3, False)} accuracy on Zone A"),
            item("corr on all Zone A", share(s["corr_te_all"]), f"a correlation of {share(s['corr_te_all'])}"),
            item("OODSelect corr_te", f"{ood.r_te:.2f}", f"inverts to ${ood.r_te:.2f}$"),
            item("OODSelect components (>= 20 points)", str(int(ood.components)),
                 "none of its points form a region of at least 20 points" if int(ood.components) == 0 else None),
            item("GTV + smoothing: points in regions, regions", f"{int(gtv_sm.comp_points)} of {REPORTED_S}, "
                 f"{int(gtv_sm.components)}",
                 f"places {int(gtv_sm.comp_points)} of its {REPORTED_S} points in {word(gtv_sm.components)} such regions"),
            item("GTV + smoothing corr_te", f"{gtv_sm.r_te:.2f}", f"the correlation still inverts to ${gtv_sm.r_te:.2f}$"),
            item("regions' label disagreement", pct(pooled.eta), f"disagree with their matches {pct(pooled.eta)} of the time"),
        ]))
    if d and z:
        sets = d["sets"]
        v = d["variogram"]
        above = v["above_50km"]
        t50, t110 = d["tiles"]["50km"], d["tiles"]["110km"]
        secs.append(("Data Construction: Sampling, Split, Data statistics, Figure 1", [
            item("points", thousands(d["points"]), f"which leaves {thousands(d['points'])} points"),
            item("points by zone and label", ", ".join(f"{k} {v_:,}" for k, v_ in d["points_by_zone_label"].items())),
            item("median variogram range", f"{v['median_km']:.0f} km", f"({v['median_km']:.0f} km)"),
            item("features correlated beyond 50 km", ", ".join(f"{k} {x:.1f} km" for k, x in above.items()),
                 f"over {min(above.values()):.0f} to {max(above.values()):.0f} km" if above else None),
            item("110 km tiles: test tiles", str(t110["test_tiles"]),
                 f"110 km tiles would leave only {t110['test_tiles']} test tiles"),
            item("50 km tiles: tiles, test tiles", f"{t50['tiles']}, {t50['test_tiles']}",
                 f"This leaves {t50['test_tiles']} test tiles"),
            item("training tiles holding Zone AE training points (all training tiles)",
                 f"{z['train_tiles_with_points']} ({z['train_tiles']})",
                 f"Zone AE points of {z['train_tiles_with_points']} training tiles"),
            item("model inputs", str(z["n_inputs"]), f"which gives {z['n_inputs']} model inputs"),
            item("training / ID test / OOD points", f"{z['n_train']:,} / {z['n_id']:,} / {z['n_ood']:,}",
                 f"The training set holds {z['n_train']:,} Zone AE points, the ID test set {z['n_id']:,}, and the "
                 f"OOD set holds {z['n_ood']:,} Zone A points"),
            item("flood shares (train, ID, OOD)", ", ".join(share(sets[k]["flood_share"]) for k in ("Train", "ID test", "OOD")),
                 "with flood shares of " + ", ".join(share(sets[k]["flood_share"]) for k in ("Train", "ID test")) +
                 f", and {share(sets['OOD']['flood_share'])}"),
            item("missing: soil flood frequency, fluvial class",
                 f"{100 * d['missing_share']['soil_floodfreq']:.1f}%, {100 * d['missing_share']['soil_fluvclass']:.1f}%",
                 f"Soil flood frequency is missing for {100 * d['missing_share']['soil_floodfreq']:.1f}\\% of points and "
                 f"fluvial class for {100 * d['missing_share']['soil_fluvclass']:.1f}\\%"),
            item("Figure 1: the seven most common land cover classes",
                 ", ".join(f"{c} {NLCD_NAMES.get(c, '?')}" for c in d["landcover_top7"]),
                 ", ".join(f"{c} {NLCD_NAMES.get(c, '?')}" for c in d["landcover_top7"][:6]) +
                 f", and {d['landcover_top7'][6]} {NLCD_NAMES.get(d['landcover_top7'][6], '?')}"),
            item("50 km grid reproduces the dataset's tile_50km", str(d["grid_50km_reproduces_tile_50km"])),
        ]))
    if s:
        lo, hi = s["tile_flood_share_range"]
        secs.append(("Metrics", [
            item("flood share within a tile (min, max over test tiles and both sets)", f"{lo:.2f} to {hi:.2f}",
                 f"ranges from {lo:.2f} to {hi:.2f}"),
            item("AUROC tiles", f"{s['metrics']['auc']['n_tiles']} of {s['metrics']['acc']['n_tiles']}",
                 f"covers {s['metrics']['auc']['n_tiles']} of the {s['metrics']['acc']['n_tiles']} tiles"),
        ]))
    if d and s and R["subsets"] is not None:
        m = n_ood(R)
        bn = budget_numbers(R)
        it = [item("S = 500 as a share of the OOD points", f"{100 * REPORTED_S / m:.1f}%",
                   f"which is {100 * REPORTED_S / m:.1f}\\% of the OOD points")]
        if bn:
            it.append(item("largest budget at which all three searches invert (r_te < -0.3)", bn["largest_all_invert"],
                           f"holds up to $S={tex_thousands(bn['largest_all_invert'])}$" if bn["largest_all_invert"] else None))
        secs.append(("Budget penalty", it))
        secs.append(("Graph", [
            item("median distance to the 10th nearest neighbour (Zone A test points)", f"{d['median_10nn_km']:.2f} km",
                 f"is {d['median_10nn_km']:.2f} km"),
            item("random-subset purity S/m at S = 500", share(REPORTED_S / m), f"which is {share(REPORTED_S / m)} at $S=500$"),
        ]))
    if z:
        secs.append(("Models", [
            item("N", str(z["kept"]), f"The $N={z['kept']}$ models"),
            item("selection / validation / test models", f"{z['n_selection']} / {z['n_validation']} / {z['n_test']}",
                 f"into {z['n_selection']} selection, {z['n_validation']} validation, and {z['n_test']} test models"),
        ]))
    if R["match_summary"]:
        ms = R["match_summary"]
        secs.append(("Matching", [
            item("caliper eps, q", f"{ms['eps']:.2f}, {ms['q']}", f"$\\epsilon={ms['eps']:.2f}$ on $q={ms['q']}$ features"),
            item("Zone AE test points without a feature (not matched)", str(ms["pool_dropped"]),
                 f"The {ms['pool_dropped']} Zone AE test points that lack a feature are not matched"),
        ]))
    if s:
        a, u, f = (s["metrics"][k] for k in ("acc", "auc", "f1"))
        pb = s["palm_beach"]
        worst = s["most_negative_acc_tiles"]
        secs.append(("Results: Aggregation hides the shift", [
            item("pooled Delta (accuracy, AUROC, F1)", ", ".join(gap(x["pooled_delta"], 3, False) for x in (a, u, f)),
                 None),
            item("all three pooled Delta negative", str(all(x["pooled_delta"] < 0 for x in (a, u, f))),
                 "Pooled, all three shifts are negative" if all(x["pooled_delta"] < 0 for x in (a, u, f)) else None),
            item("Palm Beach tile: share of Zone AE / Zone A test points", f"{pb['share_id']:.0%} / {pb['share_ood']:.0%}",
                 f"holds {pb['share_id']:.0%}\\% of the Zone AE test points but only {pb['share_ood']:.0%}\\% of the "
                 f"Zone A test points".replace("%\\%", "\\%")),
            item("Palm Beach tile: models' accuracy on its Zone AE points (all models)", share(pb["acc_id"]),
                 f"the models average {share(pb['acc_id'])} accuracy on its Zone AE points"),
            item("Palm Beach tile: id, county, points ID / OOD, accuracy on its Zone A points",
                 f"{pb['tile']}, {pb['county_fips']}, {pb['n_id']:,} / {pb['n_ood']:,}, {pb['acc_ood']:.3f}"),
            item("tiles where models do worse on Zone A (accuracy, AUROC, F1)",
                 f"{a['tiles_worse']}/{a['n_tiles']}, {u['tiles_worse']}/{u['n_tiles']}, {f['tiles_worse']}/{f['n_tiles']}",
                 f"{a['tiles_worse']} of {a['n_tiles']} in accuracy, {u['tiles_worse']} of {u['n_tiles']} in AUROC, "
                 f"and {f['tiles_worse']} of {f['n_tiles']} in F1"),
            item("Wilcoxon p, Holm-adjusted (AUROC, F1, accuracy)",
                 f"{u['holm_p']:.3f}, {f['holm_p']:.3f}, {a['holm_p']:.3f}",
                 f"shift in AUROC ($p={u['holm_p']:.3f}$) and F1 ($p={f['holm_p']:.3f}$) at 95\\% confidence, but not "
                 f"in accuracy ($p={a['holm_p']:.3f}$)"),
            item("Wilcoxon p, raw (accuracy, AUROC, F1)", f"{a['wilcoxon_p']:.4f}, {u['wilcoxon_p']:.4f}, {f['wilcoxon_p']:.4f}"),
            item("shift at 95% (Holm) in AUROC / F1 / accuracy",
                 " / ".join("yes" if x["holm_p"] < 0.05 else "no" for x in (u, f, a))),
            item("per-tile Delta intervals include zero (accuracy, AUROC, F1)",
                 ", ".join(str(x["ci_delta"][0] < 0 < x["ci_delta"][1]) for x in (a, u, f))),
            item("most negative per-tile accuracy Delta (tile, county, Delta, flood share ID / OOD)",
                 "; ".join(f"{w['tile']} {w['county']} {w['acc_delta']:+.2f} ({w['flood_id']:.2f} / {w['flood_ood']:.2f})"
                           for w in worst)),
            item("corr_te on all Zone A", sgn(s["corr_te_all"], 2, False), f"$\\text{{corr}}_\\text{{te}}=+{s['corr_te_all']:.2f}$"),
        ]))
    if R["subsets"] is not None and s:
        rows = []
        o, lap, gtv = (pick(R, n) for n in SEARCHES)
        osm = pick(R, "OODSelect", REPORTED_S, True)
        if o is not None:
            rows += [item("OODSelect corr_te", f"{o.r_te:.2f}", f"OODSelect finds an inversion of ${o.r_te:.2f}$"),
                     item("OODSelect: test tiles touched", f"{int(o.tiles)} of {s['metrics']['acc']['n_tiles']}",
                          f"spread across {int(o.tiles)} of the {s['metrics']['acc']['n_tiles']} test tiles"),
                     item("OODSelect components", str(int(o.components)))]
        if gtv_sm is not None and gtv is not None and lap is not None and osm is not None:
            rows += [
                item("GTV + smoothing corr_te [CI]", f"{gtv_sm.r_te:.2f} [{gtv_sm.ci_lo:.2f}, {gtv_sm.ci_hi:.2f}]",
                     f"keeps an inversion of ${gtv_sm.r_te:.2f}$ [${gtv_sm.ci_lo:.2f}$, ${gtv_sm.ci_hi:.2f}$]"),
                item("GTV + smoothing points in components, components, purity",
                     f"{int(gtv_sm.comp_points)}, {int(gtv_sm.components)}, {share(gtv_sm.purity)}",
                     f"places {int(gtv_sm.comp_points)} of the {REPORTED_S} points in {word(gtv_sm.components)} components, "
                     f"with a purity of {share(gtv_sm.purity)}"),
                item("GTV alone corr_te, points, components, purity",
                     f"{gtv.r_te:.2f}, {int(gtv.comp_points)}, {int(gtv.components)}, {share(gtv.purity)}",
                     f"GTV alone inverts more (${gtv.r_te:.2f}$) but holds {int(gtv.comp_points)} points in "
                     f"{word(gtv.components)} components, with a purity of {share(gtv.purity)}"),
                item("Laplacian corr_te, components, points, purity",
                     f"{lap.r_te:.2f}, {int(lap.components)}, {int(lap.comp_points)}, {share(lap.purity)}",
                     f"The Laplacian inverts more still (${lap.r_te:.2f}$), but its {word(lap.components)} components hold "
                     f"only {int(lap.comp_points)} points, and its purity is {share(lap.purity)}"),
                item("OODSelect + smoothing corr_te, points, purity",
                     f"{osm.r_te:.2f}, {int(osm.comp_points)}, {share(osm.purity)}",
                     f"with an inversion of ${osm.r_te:.2f}$, but its components hold only {int(osm.comp_points)} points, "
                     f"and its purity is {share(osm.purity)}"),
                item("GTV + smoothing: test tiles touched", str(int(gtv_sm.tiles))),
                item("ties at the cut of GTV + smoothing (stable sort)", str(int(gtv_sm.ties_at_cut))),
            ]
        if rows:
            secs.append(("Results: OODSelect inverts / Spatial regularization forces regions / Figure 2", rows))
    if R["match"] is not None and reg is not None:
        base = R["match"][R["match"].kind == "baseline"].iloc[0]
        other = R["match"][R["match"].kind == "other"].iloc[0]
        secs.append(("Results: Concept shift without covariate shift (Table 3)", [
            item("regions", str(len(reg)), f"matching results for the {word(len(reg))} regions"),
            item("pooled unmatched vs baseline", f"{100 * pooled.unmatched:.1f}% vs {100 * base.unmatched:.0f}%",
                 f"Pooled, {100 * pooled.unmatched:.1f}\\% of the regions' points have no match, against "
                 f"{100 * base.unmatched:.0f}\\% in the Zone AE baseline"),
            item("largest region unmatched share", f"{reg.unmatched.max():.3f}",
                 "no region exceeds 5\\%" if reg.unmatched.max() <= 0.05 else None),
            item("pooled eta vs baseline", f"{pooled.eta:.2f} vs {base.eta:.2f}",
                 f"$\\eta={pooled.eta:.2f}$, against {base.eta:.2f} in the Zone AE baseline"),
            item("regions above the baseline eta", f"{int((reg.eta > base.eta).sum())} of {len(reg)}",
                 "every region lies above this baseline" if (reg.eta > base.eta).all() else None),
            item("region eta range", f"{reg.eta.min():.2f} to {reg.eta.max():.2f}",
                 f"from {reg.eta.min():.2f} to {reg.eta.max():.2f}"),
            item("Other Zone A unmatched, eta", f"{100 * other.unmatched:.0f}%, {other.eta:.2f}",
                 f"with {100 * other.unmatched:.0f}\\% unmatched and $\\eta={other.eta:.2f}$"),
            item("flood share of the points / of their matches",
                 "; ".join(f"{x['name']} {x.pos_points:.2f} / {x.pos_matches:.2f}" for _, x in reg.iterrows())
                 + f"; all regions {pooled.pos_points:.2f} / {pooled.pos_matches:.2f}; Other Zone A "
                   f"{other.pos_points:.2f} / {other.pos_matches:.2f}; baseline {base.pos_points:.2f} / "
                   f"{base.pos_matches:.2f}"),
        ]))
    mu = mu_sweep(R)
    if mu:
        tr = mu["trend"]
        it = []
        if "GTV" in tr and "Laplacian" in tr:
            g, l = tr["GTV"], tr["Laplacian"]
            it.append(item("XX 1-2: GTV purity at the smallest -> largest mu", f"{share(g['purity_lo'])} to {share(g['purity_hi'])}"))
            it.append(item("XX 3-4: Laplacian purity at the smallest -> largest mu", f"{share(l['purity_lo'])} to {share(l['purity_hi'])}"))
            it.append(item("XX 5-6: GTV corr_te at the smallest -> largest mu", f"{g['r_lo']:.2f} to {g['r_hi']:.2f}"))
            it.append(item("XX 7-8: Laplacian corr_te at the smallest -> largest mu", f"{l['r_lo']:.2f} to {l['r_hi']:.2f}"))
            mp = mu["matched"]
            if mp and mp.get("overlap"):
                it.append(item("XX 9 and 10: the loss that inverts more strongly at the same purity", mp["stronger"]))
                it.append(item("matched purity: overlap, mean / min / max of GTV minus Laplacian corr_te, share of "
                               "the overlap where GTV inverts more",
                               f"[{mp['overlap'][0]:.2f}, {mp['overlap'][1]:.2f}], {mp['mean_gtv_minus_lap']:+.3f} / "
                               f"{mp['min_gtv_minus_lap']:+.3f} / {mp['max_gtv_minus_lap']:+.3f}, "
                               f"{mp['share_gtv_stronger']:.0%}"))
            for name in ("GTV", "Laplacian"):
                x = tr[name]
                it.append(item(f"CHECK {name}: purity rises with mu (every step; Spearman)",
                               f"{x['purity_rises']} ({x['purity_spearman']:+.2f}; drops at {', '.join(x['purity_drops']) or 'none'})"))
                it.append(item(f"CHECK {name}: corr_te weakens with mu (every step; Spearman)",
                               f"{x['r_weakens']} ({x['r_spearman']:+.2f}; strengthens at {', '.join(x['r_strengthens']) or 'none'})"))
        it.append(item("grid complete (9 mu per loss)", str(mu["complete"])))
        secs.append(("Analysis: Regularization weight (Figure fig:mu; the XX of its paragraph)", it))
    bn = budget_numbers(R)
    if bn:
        secs.append(("Analysis: Budget (Figure fig:budgets)", [
            item("budgets in the grid", ", ".join(str(x) for x in bn["budgets"])),
            item("without smoothing, every search inverts (r_te < -0.3) at every budget up to 10,000",
                 str(bn["plain_all_invert_to_10000"]),
                 "all three searches invert at every budget up to 10,000 points" if bn["plain_all_invert_to_10000"] else None),
            item("with smoothing, the smallest budget from which all three invert up to 10,000", str(bn["smoothed_from"]),
                 f"with smoothing, they do from {bn['smoothed_from']} points up" if bn["smoothed_from"] else None),
            item("r_te at the largest budget (share of Zone A), alone / + smoothing",
                 f"S = {bn['largest']:,} ({bn['largest_share']:.0%}): " + "; ".join(
                     f"{k} {v[0]:+.2f} / {v[1]:+.2f}" for k, v in bn["r_largest"].items()),
                 f"At {bn['largest']:,} points, which is {bn['largest_share']:.0%}".replace("%", "\\%") + " of Zone A, none does"
                 if bn["none_inverts_at_largest"] else None),
            item("GTV has the highest purity at every budget 20-10,000 (alone, + smoothing)",
                 f"{bn['gtv_most_coherent'][0]}, {bn['gtv_most_coherent'][1]}",
                 "GTV is the most coherent search at every budget from 20 to 10,000 points" if all(bn["gtv_most_coherent"]) else None),
            item("GTV + smoothing purity, 50-10,000", f"{bn['gtv_sm_purity'][0]:.2f} to {bn['gtv_sm_purity'][1]:.2f}",
                 f"its purity stays between {bn['gtv_sm_purity'][0]:.2f} and {bn['gtv_sm_purity'][1]:.2f} from 50 points up"),
            item("GTV + smoothing inverts less than Laplacian + smoothing, 250-10,000 (r_te gap)",
                 f"{bn['gtv_vs_lap_sm'][0]:.2f} to {bn['gtv_vs_lap_sm'][1]:.2f}",
                 f"it inverts {bn['gtv_vs_lap_sm'][0]:.2f} to {bn['gtv_vs_lap_sm'][1]:.2f} less than the Laplacian with "
                 f"smoothing from 250 points up"),
            item("Laplacian alone, purity at 20 and 250", f"{bn['lap_purity_20']:.2f}, {bn['lap_purity_250']:.2f}",
                 f"its purity drops to {bn['lap_purity_20']:.2f} at 20 points and to {bn['lap_purity_250']:.2f} at 250 points"),
            item("r_te by budget, alone", bn["r_plain"]),
            item("r_te by budget, + smoothing", bn["r_smoothed"]),
        ]))
    if R["subsets"] is not None:
        rows = []
        trio = [(n, pick(R, n), pick(R, n, REPORTED_S, True)) for n in SEARCHES]
        if all(p is not None and q is not None for _, p, q in trio):
            rows.append(item("purity alone -> + smoothing (OODSelect, Laplacian, GTV)",
                             ", ".join(f"{share(p.purity)} to {share(q.purity)}" for _, p, q in trio),
                             f"from {share(trio[0][1].purity)} to {share(trio[0][2].purity)} for OODSelect, from "
                             f"{share(trio[1][1].purity)} to {share(trio[1][2].purity)} for the Laplacian, and from "
                             f"{share(trio[2][1].purity)} to {share(trio[2][2].purity)} for GTV"))
            # Read from Table 2 (the paper's Ablation reads its rounded values): -0.85 -> -0.80 is 0.05.
            w = [round(q.r_te, 2) - round(p.r_te, 2) for _, p, q in trio]
            w_raw = [q.r_te - p.r_te for _, p, q in trio]
            rows.append(item("corr_te weakens by (OODSelect, Laplacian, GTV), from Table 2's rounded values "
                             "(unrounded in parentheses)",
                             ", ".join(f"{x:.2f} ({y:.3f})" for x, y in zip(w, w_raw)),
                             f"weakens by {min(w):.2f} to {max(w):.2f}"))
            lap_sm, gtv_al = trio[1][2], trio[2][1]
            rows.append(item("Laplacian + smoothing vs GTV alone: purity, points in components",
                             f"{share(lap_sm.purity)} vs {share(gtv_al.purity)}, {int(lap_sm.comp_points)} vs "
                             f"{int(gtv_al.comp_points)}",
                             f"with a purity of {share(lap_sm.purity)} against {share(gtv_al.purity)}, but its components "
                             f"hold fewer points, {int(lap_sm.comp_points)} against {int(gtv_al.comp_points)}"))
            g = trio[2][2]
            rows.append(item("GTV + smoothing: points in components, components", f"{int(g.comp_points)}, {int(g.components)}",
                             f"with {int(g.comp_points)} points in {word(g.components)} components"))
        if rows:
            secs.append(("Analysis: Ablation (smoothing)", rows))
    if z:
        fam_secs = R["manifest"]["stages"].get("zoo", {}).get("fit_secs") if R.get("manifest") else None
        searches = R["subsets"]
        med = None
        if searches is not None and R.get("manifest"):
            secs_ = [u.get("secs") for k, u in R["manifest"].get("units", {}).items() if u.get("secs")]
            med = float(np.median(secs_)) if secs_ else None
        secs.append(("Appendix", [
            item("settings, degenerate, duplicates, N", f"{z['settings']}, {z['degenerate']}, {z['duplicate']}, {z['kept']}",
                 f"Of the {z['settings']} settings, {z['degenerate']} predicts one class for more than 95\\% of the ID test "
                 f"points, and {z['duplicate']} repeat the predictions of an earlier setting, which leaves $N={z['kept']}$ models"),
            item("training points", f"{z['n_train']:,}", f"all {z['n_train']:,} training points"),
            item("time to fit the 369 settings (this run)", f"{fam_secs:.0f} s" if fam_secs else "n/a"),
            item("median search wall time (this run, searches run in parallel)", f"{med / 60:.1f} min" if med else "n/a"),
        ]))
    return secs


def budget_numbers(R) -> dict | None:
    t = budget_grid(R)
    if t is None or t.empty:
        return None
    full = all(len(t[(t.search == s) & (t.smoothing == sm)]) == 13 for s in SEARCHES for sm in (False, True))
    if not full:
        return None
    m = n_ood(R)
    r = {sm: t[t.smoothing == sm].pivot(index="S", columns="search", values="r_te") for sm in (False, True)}
    pu = {sm: t[t.smoothing == sm].pivot(index="S", columns="search", values="purity") for sm in (False, True)}
    budgets = sorted(r[False].index)
    largest = budgets[-1]
    upto = [b for b in budgets if b <= 10000]
    inv = r[False] < -0.3
    largest_all = max((b for b in budgets if inv.loc[:b].all().all()), default=None)
    sm_inv = r[True] < -0.3
    sm_from = None
    for b in upto:
        if sm_inv.loc[[x for x in upto if x >= b]].all().all():
            sm_from = b
            break
    mid = [b for b in budgets if 20 <= b <= 10000]
    s50 = [b for b in budgets if 50 <= b <= 10000]
    s250 = [b for b in budgets if 250 <= b <= 10000]
    gap_sm = (r[True].loc[s250, "GTV"] - r[True].loc[s250, "Laplacian"])
    return dict(budgets=budgets, largest_all_invert=largest_all,
                plain_all_invert_to_10000=bool(inv.loc[upto].all().all()), smoothed_from=sm_from,
                largest=largest, largest_share=largest / m,
                r_largest={s: (float(r[False].loc[largest, s]), float(r[True].loc[largest, s])) for s in SEARCHES},
                none_inverts_at_largest=bool((r[False].loc[largest] >= -0.3).all() and (r[True].loc[largest] >= -0.3).all()),
                gtv_most_coherent=tuple(bool(all(pu[sm].loc[b, "GTV"] >= pu[sm].loc[b].max() for b in mid))
                                        for sm in (False, True)),
                gtv_sm_purity=(float(pu[True].loc[s50, "GTV"].min()), float(pu[True].loc[s50, "GTV"].max())),
                gtv_vs_lap_sm=(float(gap_sm.min()), float(gap_sm.max())),
                lap_purity_20=float(pu[False].loc[20, "Laplacian"]), lap_purity_250=float(pu[False].loc[250, "Laplacian"]),
                r_plain="; ".join(f"S={b}: " + ", ".join(f"{s} {r[False].loc[b, s]:+.2f}" for s in SEARCHES) for b in budgets),
                r_smoothed="; ".join(f"S={b}: " + ", ".join(f"{s} {r[True].loc[b, s]:+.2f}" for s in SEARCHES) for b in budgets))


# ------------------------------------------------------------------ self-checks
def checks(R) -> list[tuple[str, bool, str]]:
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))
    man = R.get("manifest") or {}
    st = man.get("settings", {})
    diff = {k: (st.get(k), v) for k, v in PAPER_SETTINGS.items() if st.get(k) != v}
    add("run settings are the paper's (T, steps, patience, k, c, mu, seed, S, penalty)", not diff,
        "; ".join(f"{k}: run {a} vs paper {b}" for k, (a, b) in diff.items()) or "all equal")
    git = man.get("git", {})
    inv = man.get("invocations", [])
    commits = sorted({i["commit"][:10] for i in inv}) or [git.get("commit", "?")[:10]]
    dirty = sorted({f for i in inv for f in i.get("dirty_files", [])} | set(git.get("dirty_files", [])))
    add("the run's code was committed (no uncommitted code change in any invocation)", not dirty,
        f"commit(s) {', '.join(commits)}; uncommitted: {', '.join(dirty[:8]) or 'none'}")
    add("every invocation of the run used the same commit", len(commits) == 1, ", ".join(commits))
    z = R.get("zoo")
    if z:
        add("zoo: kept + degenerate + duplicate + unusable = settings",
            z["kept"] + z["degenerate"] + z["duplicate"] + z["unusable"] == z["settings"],
            f"{z['kept']} + {z['degenerate']} + {z['duplicate']} + {z['unusable']} = {z['settings']}")
        add("zoo: selection + validation + test = N", z["n_selection"] + z["n_validation"] + z["n_test"] == z["kept"],
            f"{z['n_selection']} + {z['n_validation']} + {z['n_test']} = {z['kept']}")
        add("zoo: per-family models add up to N", sum(f["models"] for f in z["families"].values()) == z["kept"])
    d = R.get("data")
    if d and z:
        add("data: the 50 km grid reproduces the dataset's tiles", d["grid_50km_reproduces_tile_50km"])
        add("data: Figure 1's sets are the cell's sets",
            (d["sets"]["Train"]["n"], d["sets"]["ID test"]["n"], d["sets"]["OOD"]["n"]) == (z["n_train"], z["n_id"], z["n_ood"]),
            f"{d['sets']['Train']['n']}, {d['sets']['ID test']['n']}, {d['sets']['OOD']['n']}")
        add("data: test tiles of the 50 km grid = the cell's test tiles", d["tiles"]["50km"]["test_tiles"] == z["test_tiles"])
    s = R.get("shift")
    if s:
        add("shift: Palm Beach tile's main county is Palm Beach (12099)", s["palm_beach"]["county_fips"] == "12099",
            f"tile {s['palm_beach']['tile']}, county {s['palm_beach']['county_fips']}")
        if z:
            add("shift: tiles with accuracy = the cell's test tiles", s["metrics"]["acc"]["n_tiles"] == z["test_tiles"])
    t = R.get("subsets")
    if t is not None:
        p = t[~t.smoothing]
        g = float((p.r_te - p.r_test_search).abs().max())
        add("searches: corr_te (float64) equals the search's own value (float32) within 1e-3", g < 1e-3, f"max |diff| {g:.1e}")
        add("subsets: every subset has exactly S points", bool((t.n == t.S).all()))
    m, comps = R.get("match"), R.get("components")
    if m is not None and comps is not None:
        reg = m[m.kind == "region"]
        pooled = m[m.kind == "pooled"].iloc[0]
        other = m[m.kind == "other"].iloc[0]
        base = m[m.kind == "baseline"].iloc[0]
        add("match: one row per region", len(reg) == len(comps), f"{len(reg)} rows, {len(comps)} regions")
        add("match: no region point lacks a feature", int(reg.n_missing.sum()) == 0 and int(pooled.n_missing) == 0)
        add("match: region points = the components' points", int(reg.n.sum()) == int(comps.n_points.sum()) == int(pooled.n))
        eta_w = float((reg.eta * reg.n_matched).sum() / reg.n_matched.sum())
        add("match: pooled eta = the regions' eta weighted by matched points", abs(eta_w - pooled.eta) < 1e-12,
            f"{pooled.eta:.6f} vs {eta_w:.6f}")
        if z:
            add("match: Other Zone A = the Zone A test points outside the subset",
                int(other.n) + int(other.n_missing) == z["n_ood"] - REPORTED_S,
                f"{int(other.n)} + {int(other.n_missing)} missing = {z['n_ood']} - {REPORTED_S}")
            ms = R["match_summary"]
            add("match: baseline = the Zone AE test points with every feature",
                int(base.n) == z["n_id"] - ms["pool_dropped"], f"{int(base.n)} = {z['n_id']} - {ms['pool_dropped']}")
        add("regions: no tie for the majority county", not bool(comps.majority_tie.any()),
            ", ".join(comps.name[comps.majority_tie]) or "none")
    return out


# ------------------------------------------------------------------ the paper
def paper_tables(tex: str) -> dict:
    """label -> the tabular block of each table environment of the paper."""
    out = {}
    for env in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        body = env.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", body)
        tab = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", body, re.S)
        if lab and tab:
            out[lab.group(1)] = tab.group(0)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def paper_check(tabs: list[dict], secs: list, tex: str) -> list[tuple[str, str, str]]:
    """(what, PASS / FAIL / NOT IN PAPER, detail) for every generated table row and every phrase."""
    out = []
    ptabs = paper_tables(tex)
    for t in tabs:
        if t["label"] not in ptabs:
            out.append((f"{t['label']}", "NOT IN PAPER", "the paper has no table with this label"))
            continue
        mine = [_norm(x) for x in t["tex"].splitlines()]
        theirs = [_norm(x) for x in ptabs[t["label"]].splitlines()]
        for i, row in enumerate(mine):
            ok = row in theirs
            out.append((f"{t['label']} row {i + 1}", "PASS" if ok else "FAIL", row if ok else f"generated: {row}"))
        extra = [r for r in theirs if r not in mine]
        if extra:
            out.append((f"{t['label']}: paper rows not generated", "FAIL", " | ".join(extra)))
    flat = _norm(tex)
    for title, items in secs:
        for it in items:
            if it.get("phrase"):
                ok = _norm(it["phrase"]) in flat
                out.append((f"{title}: {it['label']}", "PASS" if ok else "FAIL", it["phrase"]))
    return out
