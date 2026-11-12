#!/usr/bin/env python3
"""Checks for the three entry points on a small synthetic dataset (aotil/synthetic.py; about a minute).
Run from the repo root: .venv/bin/python test_entry_points.py

1. --help of each script exits 0 and shows its examples; run_experiments.py --dry-run lists the 55 searches.
2. run_experiments.py --stage all with three S = 500 searches (2 trials, 60 steps): every stage writes its outputs
   and marks itself in manifest.json (git commit, settings, package versions, input hashes, timings); the run is not
   "complete" (52 searches missing); a second call skips every finished unit; a missing input names the stage to run.
3. make_paper.py --no-copy on that run: Tables 1-3 and the Appendix table as .tex blocks in the paper's format,
   numbers.md with self-checks and the check against a paper file, the maps and feature figures; the budget and mu
   figures are skipped (their searches are missing); the figure install keeps old versions as figures/old/<name>_vN.
4. export_components.py --method GTV+smoothing on that run: the folder and its zip, the GeoPackage layers and the CSVs
   with the documented columns, counts that agree with the run's Table 3.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aotil"))
import synthetic  # noqa: E402

FAILS = []
PY = sys.executable


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def quiet(fn, *args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = fn(*args)
    return code, buf.getvalue()


def check_help():
    print("1. --help and --dry-run")
    for script in ("run_experiments.py", "make_paper.py", "export_components.py"):
        got = subprocess.run([PY, script, "--help"], cwd=ROOT, capture_output=True, text=True)
        check(got.returncode == 0 and "Examples" in got.stdout, f"{script} --help")
    with tempfile.TemporaryDirectory() as tmp:
        got = subprocess.run([PY, "run_experiments.py", "--dry-run", "--out", f"{tmp}/r", "--jobs", "8"], cwd=ROOT,
                             capture_output=True, text=True)
        check(got.returncode == 0 and "55 of 55 to run" in got.stdout and "estimated total" in got.stdout
              and "gtv_mu0.01_S500" in got.stdout and "oodselect_S15000" in got.stdout,
              "--dry-run: the 55 searches (39 budget grid + 16 mu sweep) and an estimate")


def check_pipeline(tmp: Path) -> Path:
    print("2. run_experiments.py")
    import run_experiments as rx
    data = synthetic.dataset(tmp / "data")
    run = tmp / "run"
    args = ["--stage", "all", "--out", str(run), "--logs", str(tmp / "logs"), "--data", str(data),
            "--counties", str(data / "counties.parquet"), "--jobs", "3", "--trials", "2", "--max-steps", "60",
            "--patience", "20", "--only", "oodselect_S500", "laplacian_mu3_S500", "gtv_mu3_S500"]
    code, out = quiet(rx.main, args)
    man = json.loads((run / "manifest.json").read_text())
    check(code == 0 and all(man["stages"][s]["status"] == "done" for s in rx.STAGES), "every stage done")
    check(man["git"]["commit"] and man["packages"]["torch"] and man["settings"]["trials"] == 2
          and any(k.endswith("dataset_FL.parquet") for k in man["inputs"]) and man["complete"] is False,
          "manifest: commit, packages, settings, input hashes; not complete (52 searches missing)")
    files = ["zoo/FL_band_AE_s0.npz", "zoo/zoo.json", "data/data.json", "data/figure1.parquet",
             "data/variogram/features.csv", "shift/summary.json", "shift/per_tile.csv", "smooth/subsets.csv",
             "smooth/masks.npz", "regions/points.parquet", "regions/components.csv", "match/table.csv",
             "match/pairs.parquet", "match/summary.json"] + [f"search/{u}/result.json" for u in
                                                            ("oodselect_S500", "laplacian_mu3_S500", "gtv_mu3_S500")]
    missing = [f for f in files if not (run / f).exists()]
    check(not missing, f"every stage's outputs ({len(files)} files){': missing ' + str(missing) if missing else ''}")
    sub = pd.read_csv(run / "smooth" / "subsets.csv")
    check(len(sub) == 6 and set(sub.S) == {500} and (sub.n == 500).all(), "6 subsets of 500 points (3 searches x smoothing)")
    t = pd.read_csv(run / "match" / "table.csv")
    check(t.kind.tolist()[-3:] == ["pooled", "other", "baseline"] and (t.kind == "region").sum() >= 1,
          f"Table 3 rows: {(t.kind == 'region').sum()} regions, all regions, Other Zone A, baseline")
    log_ok = (tmp / "logs" / "run_experiments.log").exists() and (tmp / "logs" / "search" / "gtv_mu3_S500.log").exists()
    check(log_ok and "search: done gtv_mu3_S500" in out, "logs: the run's and one per search; progress lines")
    theta = np.load(run / "search" / "gtv_mu3_S500" / "theta.npy")
    code, out = quiet(rx.main, args)
    check(code == 0 and "all 3 searches finished" in out and "zoo: finished before; skipped" in out
          and np.array_equal(theta, np.load(run / "search" / "gtv_mu3_S500" / "theta.npy")),
          "a second call skips the finished units")
    try:
        quiet(rx.main, ["--stage", "shift", "--out", str(tmp / "empty"), "--logs", str(tmp / "logs2"),
                        "--data", str(data)])
        ok = False
    except SystemExit as e:
        ok = "--stage zoo" in str(e)
    check(ok, "a missing cell names the stage to run (zoo)")
    return run


def check_make_paper(tmp: Path, run: Path):
    print("3. make_paper.py")
    import make_paper
    paper = tmp / "paper.tex"
    paper.write_text("We train 290 tabular models.\n")
    code, out = quiet(make_paper.main, ["--run", str(run), "--no-copy", "--paper", str(paper),
                                         "--counties", str(tmp / "data" / "counties.parquet")])
    tabs = sorted(p.name for p in (run / "paper" / "tables").glob("*.tex"))
    check(code == 0 and tabs == ["tab_models.tex", "tab_regions.tex", "tab_search.tex", "tab_shift.tex"],
          f"the four tables: {tabs}")
    search = (run / "paper" / "tables" / "tab_search.tex").read_text().splitlines()
    check(search[0] == r"\begin{tabular}{@{}lccc@{}}" and search[4].startswith("All Zone A & ")
          and search[6].startswith(r"\quad + smoothing & ") and len(search) == 13, "tab_search in the paper's layout")
    numbers = (run / "paper" / "numbers.md").read_text()
    check("## Self-checks" in numbers and "## Against the paper" in numbers and "[FAIL] run settings" in numbers,
          "numbers.md: self-checks (the test's 2 trials fail the settings check) and the paper check")
    figs = sorted(p.name for p in (run / "paper" / "figures").glob("*.pdf"))
    check(figs == ["fig_feature_boxes.pdf", "fig_maps.pdf"] and "fig_budgets: skipped" in out and "fig_mu: skipped" in out,
          f"figures {figs}; budgets and mu skipped while their searches are missing")
    fig_dir = tmp / "figures"
    src = run / "paper" / "figures" / "fig_maps.pdf"
    first = make_paper.install(src, fig_dir)
    same = make_paper.install(src, fig_dir)
    other = tmp / "other" / "fig_maps.pdf"
    other.parent.mkdir()
    other.write_bytes(src.read_bytes() + b"%")
    third = make_paper.install(other, fig_dir)
    fourth = make_paper.install(src, fig_dir)
    check("written (new)" in first and "unchanged" in same and (fig_dir / "old" / "fig_maps_v1.pdf").exists()
          and (fig_dir / "old" / "fig_maps_v2.pdf").exists() and "replaced" in third and "replaced" in fourth,
          "install: new, unchanged, then each replacement moves the old file to old/fig_maps_vN.pdf")


def check_export(tmp: Path, run: Path):
    print("4. export_components.py")
    import export_components as ec
    code, out = quiet(ec.main, ["--method", "GTV+smoothing", "--run", str(run), "--out-root", str(tmp / "exports"),
                                "--data", str(tmp / "data"), "--counties", str(tmp / "data" / "counties.parquet")])
    folder = next(p for p in (tmp / "exports").glob("GTV+smoothing_S500_mu3_c20_*") if p.is_dir())
    zp = folder.with_suffix(".zip")
    check(code == 0 and zp.exists() and "Table 3 check: MATCH" in out, f"{folder.name} and its zip; Table 3 check MATCH")
    names = zipfile.ZipFile(zp).namelist()
    check(any(n.endswith("regions.gpkg") for n in names) and any(n.endswith("README.md") for n in names),
          "the zip holds the GeoPackage and the README")
    import geopandas as gpd
    layers = {lay: gpd.read_file(folder / "regions.gpkg", layer=lay) for lay in
              ("points", "regions", "selected_not_in_component", "matches")}
    check(all(g.crs.to_epsg() == 5070 for g in layers.values()), "every layer in EPSG:5070")
    pts = pd.read_csv(folder / "points.csv")
    check(list(pts.columns) == ec.POINT_COLS and len(pts) == len(layers["points"]), "points.csv has the documented columns")
    summ = pd.read_csv(folder / "components_summary.csv")
    t3 = pd.read_csv(run / "match" / "table.csv")
    t3 = t3[t3.kind == "region"]
    check(sorted(summ.n_points) == sorted(t3.n) and len(list((folder / "components").glob("*.csv"))) == len(summ)
          and len(pts) + len(layers["selected_not_in_component"]) == 500, "counts agree with Table 3 and the 500 points")


def main() -> int:
    check_help()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = check_pipeline(tmp)
        check_make_paper(tmp, run)
        check_export(tmp, run)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
