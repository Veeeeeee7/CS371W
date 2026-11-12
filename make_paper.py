#!/usr/bin/env python3
"""Turn one run folder of run_experiments.py into the paper's tables, numbers and figures (no torch; seconds).

  tables   prints Table 1 (tab:shift), Table 2 (tab:search), Table 3 (tab:regions) and the Appendix table
           (tab:models) in the terminal, and writes each as a paste-ready LaTeX tabular block in the paper's exact
           format to <run>/paper/tables/<name>.tex
  numbers  writes <run>/paper/numbers.md: every number the prose states, by paragraph (with the XX of the
           "Regularization weight" paragraph and the answers to its % CHECK), the mu sweep table, the self-checks
           of the run (PASS / FAIL), and a check of every generated table row and number against the paper's text
           (CS371_Cho_Li/acl_latex.tex; PASS when the paper already says it)
  figures  draws fig_feature_boxes.pdf (Figure 1), fig_maps.pdf (Figure 2), fig_budgets.pdf (Figure 3) and
           fig_mu.pdf (the mu figure) into <run>/paper/figures/ with PNG previews, and copies each PDF to
           CS371_Cho_Li/figures/ under its plain name; a replaced file moves to CS371_Cho_Li/figures/old/<name>_vN.pdf
           (the next free N) first, and an identical file is left alone

Examples:
  .venv/bin/python make_paper.py                                   # the latest complete run under results/
  .venv/bin/python make_paper.py --run results/2026-11-12-pilot --no-copy
  .venv/bin/python make_paper.py --only tables numbers            # no figures
"""
from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aotil"))
import paper_numbers as pn  # noqa: E402

PAPER = Path("CS371_Cho_Li/acl_latex.tex")
FIG_DIR = Path("CS371_Cho_Li/figures")
COUNTIES = Path("data/interim/gis/counties_FL.parquet")
TAG = "FL_band_AE_s0"


def latest_complete(results: Path = Path("results")) -> Path:
    runs = []
    for m in results.glob("*/manifest.json"):
        try:
            if json.loads(m.read_text()).get("complete"):
                runs.append(m.parent)
        except json.JSONDecodeError:
            continue
    if not runs:
        raise SystemExit("no complete run under results/: run `run_experiments.py --stage all` first, or pass "
                         "--run <folder> for an incomplete one")
    return max(runs, key=lambda p: (p.name, p.stat().st_mtime))


def install(src: Path, fig_dir: Path) -> str:
    """Copy src into fig_dir under its plain name; an existing different file moves to fig_dir/old/<stem>_vN<ext>
    with the next free N first."""
    dest = fig_dir / src.name
    if dest.exists():
        if filecmp.cmp(src, dest, shallow=False):
            return f"{dest} unchanged"
        old = fig_dir / "old"
        old.mkdir(parents=True, exist_ok=True)
        pat = re.compile(rf"^{re.escape(src.stem)}_v(\d+){re.escape(src.suffix)}$")
        n = max([int(m.group(1)) for f in old.iterdir() if (m := pat.match(f.name))], default=0) + 1
        moved = old / f"{src.stem}_v{n}{src.suffix}"
        shutil.move(str(dest), moved)
        shutil.copy2(src, dest)
        return f"{dest} replaced (old one -> {moved})"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return f"{dest} written (new)"


# ------------------------------------------------------------------ figures
def figures(R: dict, run: Path, copy: bool, fig_dir: Path, counties: Path) -> list[str]:
    import figures as fg
    import fig_feature_boxes as ffb
    import graph
    out_dir = run / "paper" / "figures"
    lines, made = [], []
    if R["data"] is not None and (run / "data" / "figure1.parquet").exists():
        d = pd.read_parquet(run / "data" / "figure1.parquet")
        d["wetness"] = d.wetindexcat / 100            # StreamCat serves the index times 100
        groups = {k: d[d.set == k] for k in ("Train", "ID test", "OOD")}
        ffb.draw(groups, [int(c) for c in R["data"]["landcover_top7"]], out_dir / "fig_feature_boxes.pdf",
                 out_dir / "fig_feature_boxes.png")
        made.append(out_dir / "fig_feature_boxes.pdf")
    else:
        lines.append("fig_feature_boxes: skipped (run `run_experiments.py --stage data`)")
    masks_path = run / "smooth" / "masks.npz"
    want = ("oodselect_S500", "gtv_mu3_S500+smoothing")
    m = np.load(masks_path) if masks_path.exists() else None
    if m is not None and all(k in m.files for k in want) and (run / "zoo" / f"{TAG}.npz").exists():
        z = np.load(run / "zoo" / f"{TAG}.npz", allow_pickle=True)
        lon, lat, tiles = np.asarray(z["lon"], float), np.asarray(z["lat"], float), np.asarray(z["tile_50km"]).astype(str)
        A = graph.build_adjacency(lon, lat, k=10)
        panels = []
        for key, title in zip(want, ("(a) OODSelect", "(b) GTV + smoothing")):
            mask = np.unpackbits(m[key])[: int(m["d"])].astype(bool)
            comp, sizes = graph.component_labels(mask, A)
            panels.append((title, pd.DataFrame(dict(lon=lon[mask], lat=lat[mask], component=comp,
                                                    in_component=sizes[comp] >= pn.C_MIN))))
        lines += fg.fig_maps(counties, (lon, lat), sorted(set(tiles)), panels, out_dir / "fig_maps.pdf",
                             out_dir / "fig_maps.png")
        made.append(out_dir / "fig_maps.pdf")
    else:
        lines.append("fig_maps: skipped (needs the S = 500 OODSelect and GTV searches and `--stage smooth`)")
    bg = pn.budget_grid(R)
    if bg is not None and pn.budget_numbers(R) is not None:
        fg.fig_budgets(bg, out_dir / "fig_budgets.pdf", out_dir / "fig_budgets.png")
        made.append(out_dir / "fig_budgets.pdf")
    else:
        n = 0 if bg is None else len(bg[~bg.smoothing])
        lines.append(f"fig_budgets: skipped (budget grid incomplete: {n} of 39 searches; run "
                     f"`run_experiments.py --stage search smooth`)")
    sw = pn.mu_sweep(R)
    if sw and sw["complete"]:
        plain = sw["table"][~sw["table"].smoothing]
        ref = sw["reference"].loc[False]
        fg.fig_mu(plain, dict(r_te=float(ref.r_te), purity=float(ref.purity)), out_dir / "fig_mu.pdf",
                  out_dir / "fig_mu.png")
        made.append(out_dir / "fig_mu.pdf")
    else:
        lines.append("fig_mu: skipped (mu sweep incomplete; run `run_experiments.py --stage search smooth`)")
    for f in made:
        lines.append(f"drew {f} and {f.with_suffix('.png')}")
        if copy:
            lines.append(install(f, fig_dir))
    return lines


# ------------------------------------------------------------------ numbers.md
def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return out


def mu_sweep_md(R) -> list[str]:
    sw = pn.mu_sweep(R)
    if not sw:
        return ["## The mu sweep", "", "Not in this run (run `run_experiments.py --stage search smooth`).", ""]
    rows = []
    t = sw["table"]
    ref = sw["reference"]
    for sm in (False, True):
        if sm in ref.index:
            x = ref.loc[sm]
            rows.append(dict(loss="OODSelect", mu="0", smoothing="yes" if sm else "no",
                             corr_te=f"{x.r_te:+.3f} [{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]", purity=f"{x.purity:.3f}",
                             comps=f"{int(x.components)} ({int(x.comp_points)})", r_val=f"{x.r_val:+.3f}"))
        for loss in ("Laplacian", "GTV"):
            for _, x in t[(t.search == loss) & (t.smoothing == sm)].sort_values("mu").iterrows():
                rows.append(dict(loss=loss, mu=f"{x.mu:g}", smoothing="yes" if sm else "no",
                                 corr_te=f"{x.r_te:+.3f} [{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]", purity=f"{x.purity:.3f}",
                                 comps=f"{int(x.components)} ({int(x.comp_points)})", r_val=f"{x.r_val:+.3f}"))
    out = ["## The mu sweep at S = 500 (Figure fig:mu)", "",
           "corr_te on the 58 test models with its Fisher 95% interval; purity; # Comps (points they hold); r_val, "
           "the correlation on the 58 validation models of the same subset.", ""]
    out += md_table(pd.DataFrame(rows)) + [""]
    tr = sw["trend"]
    mp = sw["matched"]
    if "GTV" in tr and "Laplacian" in tr and mp and mp.get("overlap"):
        g, l = tr["GTV"], tr["Laplacian"]
        stronger = mp["stronger"]
        art = "the Laplacian" if stronger == "Laplacian" else "GTV"
        out += ["Paragraph \"Regularization weight\" with its XX filled (Victor and Claude chat edit the prose):", "",
                f"> As $\\mu$ grows, purity rises from {g['purity_lo']:.2f} to {g['purity_hi']:.2f} for GTV and from "
                f"{l['purity_lo']:.2f} to {l['purity_hi']:.2f} for the Laplacian, while $\\text{{corr}}_\\text{{te}}$ "
                f"weakens from ${g['r_lo']:.2f}$ to ${g['r_hi']:.2f}$ and from ${l['r_lo']:.2f}$ to ${l['r_hi']:.2f}$. "
                f"At the same purity, {art} inverts more strongly. A user who needs a stronger inversion can therefore "
                f"lower $\\mu$ or choose {art}, at the cost of coherence.", "",
                "Answers to the % CHECK (see the Analysis section of numbers above): purity rises with mu at every "
                f"step for GTV: {g['purity_rises']}, for the Laplacian: {l['purity_rises']}; corr_te weakens at every "
                f"step for GTV: {g['r_weakens']}, for the Laplacian: {l['r_weakens']}. Over the purities both reach "
                f"({mp['overlap'][0]:.2f} to {mp['overlap'][1]:.2f}), GTV minus Laplacian corr_te at equal purity is "
                f"{mp['mean_gtv_minus_lap']:+.3f} on average (range {mp['min_gtv_minus_lap']:+.3f} to "
                f"{mp['max_gtv_minus_lap']:+.3f}; GTV inverts more on {mp['share_gtv_stronger']:.0%} of that range).",
                ""]
    return out


def write_numbers(R, run: Path, tabs: list[dict], paper: Path) -> tuple[Path, int, int, int]:
    secs = pn.numbers(R)
    chk = pn.checks(R)
    man = R.get("manifest") or {}
    out = [f"# Paper numbers from `{run}`", "",
           f"Written {dt.datetime.now():%Y-%m-%d %H:%M} by make_paper.py from the run of git commit "
           f"`{man.get('git', {}).get('commit', '?')[:10]}`. Tables: `{run}/paper/tables/*.tex`. A value shows as the "
           f"paper prints it; [PASS] means the paper's text already contains it, [FAIL] that it does not (the number "
           f"or the wording differs).", ""]
    missing = pn.missing_stages(R)
    if missing:
        out += [f"Missing stages (their numbers are left out): {', '.join(missing)}.", ""]
    out += ["## Self-checks", ""]
    for name, ok, detail in chk:
        out.append(f"- [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    n_fail_checks = sum(not ok for _, ok, _ in chk)
    pc = []
    if paper.exists():
        pc = pn.paper_check(tabs, secs, paper.read_text())
        n_pass = sum(s == "PASS" for _, s, _ in pc)
        out += ["", f"## Against the paper ({paper}): {n_pass} of {len(pc)} PASS", ""]
        for what, status, detail in pc:
            if status != "PASS" or what.startswith("tab:"):
                out.append(f"- [{status}] {what}: `{detail}`")
    status_of = {}
    for what, status, _ in pc:
        status_of[what] = status
    out.append("")
    for title, items in secs:
        out += [f"## {title}", ""]
        for it in items:
            tag = status_of.get(f"{title}: {it['label']}")
            out.append(f"- {it['label']}: **{it['value']}**" + (f" [{tag}]" if tag else ""))
        out.append("")
    out += mu_sweep_md(R)
    t = R.get("subsets")
    if t is not None:
        cols = ["unit", "smoothing", "r_te", "ci_lo", "ci_hi", "r_val", "components", "comp_points", "pieces",
                "purity", "tiles", "ties_at_cut"]
        show = t[cols].copy()
        for c in ("r_te", "ci_lo", "ci_hi", "r_val", "purity"):
            show[c] = show[c].map(lambda v: f"{v:+.3f}" if c != "purity" else f"{v:.3f}")
        out += ["## Every subset of the run (alone and + smoothing)", ""] + md_table(show) + [""]
    path = run / "paper" / "numbers.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    n_fail_paper = sum(s == "FAIL" for _, s, _ in pc)
    return path, n_fail_checks, n_fail_paper, len(pc)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="run folder (default: the latest complete one under results/)")
    ap.add_argument("--only", nargs="+", choices=["tables", "numbers", "figures"],
                    help="only these parts (default all three)")
    ap.add_argument("--no-copy", action="store_true", help=f"do not touch {FIG_DIR}/")
    ap.add_argument("--paper", type=Path, default=PAPER, help=f"the paper to check against (default {PAPER})")
    ap.add_argument("--figures-dir", type=Path, default=FIG_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--counties", type=Path, default=COUNTIES, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    run = a.run or latest_complete()
    if not (run / "manifest.json").exists():
        raise SystemExit(f"{run} is not a run folder (no manifest.json): run `run_experiments.py --out {run}` first")
    parts = a.only or ["tables", "numbers", "figures"]
    R = pn.load(run)
    print(f"make_paper: run {run}" + (f" (missing stages: {', '.join(pn.missing_stages(R))})"
                                      if pn.missing_stages(R) else ""))
    written = []
    tabs = pn.tables(R)
    if "tables" in parts:
        tdir = run / "paper" / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        for t in tabs:
            print()
            print(pn.render(t))
            (tdir / f"{t['name']}.tex").write_text(t["tex"] + "\n")
            written.append(tdir / f"{t['name']}.tex")
    if "numbers" in parts:
        path, n_fail, n_fail_paper, n_pc = write_numbers(R, run, tabs, a.paper)
        written.append(path)
        print(f"\nself-checks: {'all PASS' if not n_fail else f'{n_fail} FAIL'}; against the paper: "
              f"{n_pc - n_fail_paper} of {n_pc} PASS" + (f", {n_fail_paper} FAIL (see {path})" if n_fail_paper else ""))
    if "figures" in parts:
        print()
        for line in figures(R, run, not a.no_copy, a.figures_dir, a.counties):
            print(line)
    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    if "figures" in parts:
        print(f"  {run / 'paper' / 'figures'}/ (PDF and PNG)" + ("" if a.no_copy else f", and the PDFs in {a.figures_dir}/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
