#!/usr/bin/env python3
"""Checks for paper_numbers.py (the tables and numbers make_paper.py writes). Run from the repo root:
.venv/bin/python aotil/test_paper_numbers.py

1. The paper's number formats: signed cells ($-$0.97, $+$0.84), Delta cells ($-$0.034, 0.017), shares, words.
2. paper_tables finds each table's tabular block by its label; paper_check passes rows the paper has, fails the
   others, and lists the paper rows that were not generated; a phrase passes only when the paper contains it.
3. The Table 2 block from a hand-made subsets table, row for row in the paper's format.
4. The mu sweep: trends (purity rising, corr_te weakening) and the matched-purity comparison of the two losses.
5. budget_numbers on a hand-made budget grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_numbers as pn  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def check_formats():
    print("1. formats")
    check(pn.sgn(-0.9716) == "$-$0.97" and pn.sgn(0.838) == "$+$0.84" and pn.sgn(-0.5, tex=False) == "-0.50",
          "signed cells")
    check(pn.gap(-0.0338) == "$-$0.034" and pn.gap(0.0172) == "0.017", "Delta cells")
    check(pn.share(0.7864) == "0.79" and pn.word(8) == "eight" and pn.tex_thousands(10000) == "10{,}000",
          "shares, number words, math-mode thousands")


TEX = r"""
\begin{table}[t]
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Points} & $\boldsymbol{\text{corr}_\text{te}}$ \textbf{[95\% CI]} & \textbf{\# Comps} & \textbf{Purity} \\
\midrule
All Zone A & $+$0.84 [$+$0.74, $+$0.90] & -- & -- \\
OODSelect & $-$0.97 [$-$0.98, $-$0.95] & 0 (0) & 0.11 \\
\quad + smoothing & $-$0.91 [$-$0.95, $-$0.85] & 7 (203) & 0.65 \\
Laplacian & $-$0.92 [$-$0.95, $-$0.86] & 7 (227) & 0.50 \\
\quad + smoothing & $-$0.86 [$-$0.92, $-$0.78] & 8 (343) & 0.81 \\
GTV & $-$0.85 [$-$0.91, $-$0.75] & 7 (383) & 0.79 \\
\quad + smoothing & $-$0.80 [$-$0.88, $-$0.69] & 8 (421) & 0.86 \\
\bottomrule
\end{tabular}
\caption{Subset search.}
\label{tab:search}
\end{table}
OODSelect finds an inversion of $-0.97$ on test models it never saw.
"""


def subsets_table():
    rows = []
    vals = {("OODSelect", False): (-0.9716, -0.984, -0.954, 0, 0, 0.1064), ("OODSelect", True): (-0.911, -0.946, -0.853, 7, 203, 0.6546),
            ("Laplacian", False): (-0.918, -0.951, -0.865, 7, 227, 0.5038), ("Laplacian", True): (-0.865, -0.918, -0.781, 8, 343, 0.810),
            ("GTV", False): (-0.846, -0.906, -0.753, 7, 383, 0.7864), ("GTV", True): (-0.803, -0.879, -0.688, 8, 421, 0.858)}
    for (search, sm), (r, lo, hi, c, p, pu) in vals.items():
        rows.append(dict(search=search, smoothing=sm, S=500, mu=0 if search == "OODSelect" else 3, group="budget grid",
                         r_te=r, ci_lo=lo, ci_hi=hi, components=c, comp_points=p, purity=pu))
    return pd.DataFrame(rows)


def check_tables():
    print("2-3. tables against a paper")
    R = dict(shift=dict(corr_te_all=0.8379, corr_te_all_ci=(0.7404, 0.9010), n_test_models=58),
             subsets=subsets_table())
    t = pn.table_search(R)
    check(pn.paper_tables(TEX)["tab:search"].count("\\\\") == 8, "the paper's tab:search block is found by its label")
    res = pn.paper_check([t], [], TEX)
    check(all(s == "PASS" for _, s, _ in res) and len(res) == len(t["tex"].splitlines()),
          f"every generated row is in the paper ({len(res)} rows PASS)")
    R["subsets"].loc[(R["subsets"].search == "GTV") & R["subsets"].smoothing, "r_te"] = -0.79
    res = pn.paper_check([pn.table_search(R)], [], TEX)
    fails = [w for w, s, _ in res if s == "FAIL"]
    check(len(fails) == 2 and any("not generated" in w for w in fails), f"a changed cell fails, and the paper's row is listed: {fails}")
    secs = [("Results", [dict(label="OODSelect", value="-0.97", phrase="OODSelect finds an inversion of $-0.97$"),
                         dict(label="wrong", value="-0.96", phrase="OODSelect finds an inversion of $-0.96$")])]
    res = pn.paper_check([], secs, TEX)
    check([s for _, s, _ in res] == ["PASS", "FAIL"], "a phrase passes only when the paper contains it")


def check_mu():
    print("4. the mu sweep")
    mus = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    rows = []
    for i, mu in enumerate(mus):
        rows.append(dict(search="GTV", mu=mu, S=500, smoothing=False, group="mu sweep", r_te=-0.95 + 0.02 * i,
                         purity=0.2 + 0.08 * i, ci_lo=0, ci_hi=0, components=0, comp_points=0, r_val=-0.9))
        rows.append(dict(search="Laplacian", mu=mu, S=500, smoothing=False, group="mu sweep", r_te=-0.97 + 0.03 * i,
                         purity=0.1 + 0.06 * i, ci_lo=0, ci_hi=0, components=0, comp_points=0, r_val=-0.9))
    rows.append(dict(search="OODSelect", mu=0, S=500, smoothing=False, group="budget grid", r_te=-0.97, purity=0.11,
                     ci_lo=0, ci_hi=0, components=0, comp_points=0, r_val=-0.98))
    sw = pn.mu_sweep(dict(subsets=pd.DataFrame(rows)))
    g, l = sw["trend"]["GTV"], sw["trend"]["Laplacian"]
    check(g["purity_rises"] and g["r_weakens"] and l["purity_rises"] and l["r_weakens"] and sw["complete"],
          "both losses: purity rises and corr_te weakens with mu")
    check(np.isclose(g["purity_lo"], 0.2) and np.isclose(g["purity_hi"], 0.84) and np.isclose(l["r_lo"], -0.97),
          "the end points of the paragraph")
    # At purity p, GTV's corr_te = -0.95 + (p - 0.2) / 4, the Laplacian's = -0.97 + (p - 0.1) / 2: GTV inverts more
    # from p = 0.26 up.
    mp = sw["matched"]
    check(mp["stronger"] == "GTV" and mp["overlap"] == (0.2, 0.58), f"matched purity: {mp['stronger']} over {mp['overlap']}")


def check_budgets():
    print("5. budget numbers")
    budgets = [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000]
    rows = []
    for s_i, search in enumerate(pn.SEARCHES):
        for sm in (False, True):
            for b in budgets:
                r = 0.5 if b == 15000 else (-0.2 if (sm and b < 50) else -0.8)
                pur = 0.9 if search == "GTV" else 0.5
                rows.append(dict(search=search, smoothing=sm, S=b, mu=0 if s_i == 0 else 3, group="budget grid",
                                 r_te=r - (0.05 if search == "Laplacian" and sm else 0), purity=pur))
    R = dict(subsets=pd.DataFrame(rows), zoo=dict(n_ood=17346), data=None)
    bn = pn.budget_numbers(R)
    check(bn["largest_all_invert"] == 10000 and bn["plain_all_invert_to_10000"] and bn["smoothed_from"] == 50,
          "inverts up to 10,000 alone, from 50 with smoothing")
    check(bn["none_inverts_at_largest"] and round(bn["largest_share"], 2) == 0.86 and all(bn["gtv_most_coherent"]),
          "none inverts at 15,000 (86% of Zone A); GTV the most coherent")
    check(np.isclose(bn["gtv_vs_lap_sm"][0], 0.05), "GTV + smoothing inverts 0.05 less than the Laplacian + smoothing")


def main() -> int:
    check_formats()
    check_tables()
    check_mu()
    check_budgets()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
