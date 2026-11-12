#!/usr/bin/env python3
"""Run every computation behind the paper's Experiments and Analysis into one run folder.

Stages (each resumable: a finished unit is skipped unless --force; each skippable with --stage):
  zoo      fit the 369 settings of the model grids on the Zone AE training points, drop 1 degenerate and 78
           duplicate models, keep N = 290; score them on the ID (Zone AE) and OOD (Zone A) test points
  data     the data numbers of Data Construction and Figure 1: set sizes and flood shares, missing values, model
           inputs, tile counts (50 km and 110 km), variogram ranges, the median 10-NN distance, Figure 1's points
  shift    Table 1: pooled and per-tile accuracy, AUROC and F1, t-intervals, Wilcoxon + Holm, Palm Beach,
           corr_te on all Zone A test points
  search   the subset searches, one single-threaded process each, --jobs at a time: OODSelect, Laplacian (mu 3)
           and GTV (mu 3) at the 13 budgets S = 10 ... 15,000 (39 searches), and the mu sweep at S = 500
           (Laplacian and GTV at mu in {0.01, 0.03, 0.1, 0.3, 1, 10, 30, 100}; 16 more searches)
  smooth   every search's subset with and without smoothing (Equation eq:smooth): corr_te [95% CI], validation
           correlation, components, purity
  regions  the components of GTV + smoothing at S = 500 (the analyzed method) and their county names
  match    Table 3: caliper matching of each region, all regions, Other Zone A and the Zone AE baseline
  all      every stage in the order zoo, data, shift, search, smooth, regions, match

Writes (under --out, default results/<today>): zoo/, data/, shift/, search/<unit>/, smooth/, regions/, match/ and
manifest.json (git commit, settings, seeds, package versions, timings, input hashes). Logs go to
logs/<run folder name>/ (run_experiments.log and one log per search). make_paper.py then turns the run folder into
the paper's tables, numbers and figures; export_components.py exports the regions for GIS.

Examples:
  .venv/bin/python run_experiments.py --dry-run                      # the plan and its estimated time
  caffeinate -i .venv/bin/python run_experiments.py --stage all      # everything into results/<today>
  .venv/bin/python run_experiments.py --stage all --out results/2026-11-12-pilot \\
      --only oodselect_S500 laplacian_mu3_S500 gtv_mu3_S500           # the pilot: only the three S = 500 searches
  .venv/bin/python run_experiments.py --stage search smooth --jobs 6  # resume the searches, then smooth
  .venv/bin/python run_experiments.py --stage shift --force           # redo one stage
"""
from __future__ import annotations

import os

# One BLAS/OpenMP thread per process before numpy loads: the zoo's fits and every search are then the same
# whether they run alone or in parallel (Apple Accelerate otherwise moves LogisticRegression's scores by ~1e-11).
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _var in THREAD_VARS:
    os.environ[_var] = "1"

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import fnmatch  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aotil"))

# ------------------------------------------------------------------ the paper's settings
STATE, PROTOCOL, TRAIN_DOM, CELL_SEED = "FL", "band", "AE", 0
TAG = f"{STATE}_{PROTOCOL}_{TRAIN_DOM}_s{CELL_SEED}"
DATA = Path("data/processed_v4")
COUNTIES = Path("data/interim/gis/counties_FL.parquet")
BUDGETS = (10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000)
MU_GRID = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
MU = 3                        # mu of both losses in every main result
REPORTED_S = 500
KNN, C_MIN = 10, 20
TRIALS, MAX_STEPS, PATIENCE, SEED = 100, 50_000, 500, 0
ANALYZED = ("gtv", MU, REPORTED_S, True)     # GTV + smoothing at S = 500: the regions of Table 3
STAGES = ("zoo", "data", "shift", "search", "smooth", "regions", "match")
SEARCH_NAMES = {"oodselect": "OODSelect", "laplacian": "Laplacian", "gtv": "GTV"}
# Seconds per search, measured on the 2026-11-07 run (Apple M4 Pro, 8 searches at a time), for --dry-run and
# the progress line's estimate. The mu sweep uses the S = 500 time of its loss.
EST_SECS = {
    "oodselect": dict(zip(BUDGETS, (100, 158, 204, 262, 297, 271, 312, 284, 318, 1166, 1350, 356, 1989))),
    "laplacian": dict(zip(BUDGETS, (114, 183, 258, 273, 333, 611, 777, 963, 1379, 1330, 1798, 1432, 1753))),
    "gtv": dict(zip(BUDGETS, (224, 232, 382, 579, 737, 520, 1011, 1260, 1350, 1495, 1476, 308, 763))),
}
EST_STAGE_SECS = dict(zoo=60, data=150, shift=30, smooth=60, regions=10, match=60)


# ------------------------------------------------------------------ small helpers
class Tee:
    """Write everything printed to the terminal and to the run's log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def fmt_secs(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        except Exception:
            return ""
    # Uncommitted code: modified tracked files, and untracked Python or shell files (not docs or outputs).
    dirty = [line[3:] for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    dirty += [line[3:] for line in run("status", "--porcelain").splitlines()
              if line.startswith("??") and line.rstrip().endswith((".py", ".sh"))]
    return dict(commit=run("rev-parse", "HEAD").strip(), branch=run("rev-parse", "--abbrev-ref", "HEAD").strip(),
                dirty_files=dirty)


def versions() -> dict:
    out = dict(python=platform.python_version())
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "xgboost", "torch", "optuna", "pyproj", "geopandas",
                "shapely", "pyarrow", "matplotlib", "joblib"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def default_jobs() -> int:
    try:
        perf = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                  text=True).stdout.strip())
    except Exception:
        perf = (os.cpu_count() or 4)
    return max(1, perf - 2)


# ------------------------------------------------------------------ the run folder
class Run:
    """Paths, settings and the manifest of one run folder."""

    def __init__(self, a):
        self.out = Path(a.out)
        self.logs = Path(a.logs) if a.logs else Path("logs") / self.out.name
        self.data_dir = Path(a.data)
        self.dataset = self.data_dir / f"dataset_{STATE}.parquet"
        self.features_txt = self.data_dir / "FEATURES.txt"
        self.counties = Path(a.counties)
        self.zoo_dir = self.out / "zoo"
        self.settings = dict(state=STATE, protocol=PROTOCOL, train_dom=TRAIN_DOM, cell_seed=CELL_SEED, tag=TAG,
                             data=str(self.data_dir), budgets=list(BUDGETS), mu_grid=list(MU_GRID), mu=MU,
                             reported_S=REPORTED_S, knn=KNN, c=C_MIN, trials=a.trials, max_steps=a.max_steps,
                             patience=a.patience, search_seed=SEED, budget_penalty="squared",
                             lambda_range=[1e-6, 1e2], lr_range=[1e-2, 1e2], eta_min_range=[1e-3, 1.0],
                             tau_range=[10, 200], W_range=[10, 1000], model_split="60/20/20, random_state=42",
                             tile_split="50 km tiles, 70/30, seed 0", smoothing="(I + D)^-1 (I + A) s on graph A",
                             analyzed="GTV + smoothing, mu 3, S 500")
        self.path = self.out / "manifest.json"
        self.manifest = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.git = git_info()                  # the code of this invocation, read once at its start
        self.started = dt.datetime.now().isoformat(timespec="seconds")

    def save(self, **updates) -> None:
        m = self.manifest
        m.setdefault("run", str(self.out))
        m.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
        m["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        m["git"] = self.git
        runs = m.setdefault("invocations", [])
        if not runs or runs[-1].get("started") != self.started:
            runs.append(dict(started=self.started, commit=self.git["commit"], dirty_files=self.git["dirty_files"],
                             argv=sys.argv[1:]))
        m["settings"] = self.settings
        m["packages"] = versions()
        m["machine"] = dict(platform=platform.platform(), processor=platform.processor(), cpus=os.cpu_count())
        m.setdefault("stages", {})
        m.setdefault("units", {})
        m.update(updates)
        write_json(self.path, m)

    def stage_done(self, stage: str) -> bool:
        return self.manifest.get("stages", {}).get(stage, {}).get("status") == "done"

    def mark(self, stage: str, secs: float, **info) -> None:
        self.manifest.setdefault("stages", {})[stage] = dict(
            status="done", secs=round(secs, 1), finished=dt.datetime.now().isoformat(timespec="seconds"), **info)
        self.save()

    def require(self, *paths: Path, stage: str) -> None:
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)}: run `run_experiments.py --stage {stage} "
                             f"--out {self.out}` first")


# ------------------------------------------------------------------ the search plan
def search_plan() -> list[dict]:
    """The 55 searches: the budget grid (OODSelect, Laplacian mu 3, GTV mu 3 at 13 budgets), then the mu sweep at
    S = 500 (Laplacian and GTV at every mu of MU_GRID but 3, which the grid has)."""
    from search import unit_name
    units = []
    for method, mu in (("oodselect", 0), ("laplacian", MU), ("gtv", MU)):
        for S in BUDGETS:
            units.append(dict(method=method, mu=mu, S=S, group="budget grid"))
    for method in ("laplacian", "gtv"):
        for mu in MU_GRID:
            if mu != MU:
                units.append(dict(method=method, mu=mu, S=REPORTED_S, group="mu sweep"))
    for u in units:
        u["name"] = unit_name(u["method"], u["mu"], u["S"])
        u["est"] = EST_SECS[u["method"]][u["S"]]
    return units


def select_units(units: list[dict], only: list[str] | None) -> list[dict]:
    if not only:
        return units
    keep = [u for u in units if any(fnmatch.fnmatch(u["name"], p) for p in only)]
    if not keep:
        raise SystemExit(f"--only {' '.join(only)} matches no search; names look like oodselect_S500, "
                         f"laplacian_mu3_S500, gtv_mu0.01_S500")
    return keep


def makespan(secs: list[float], jobs: int) -> float:
    """Wall time of running the units largest first on `jobs` workers."""
    loads = [0.0] * max(1, jobs)
    for s in sorted(secs, reverse=True):
        i = loads.index(min(loads))
        loads[i] += s
    return max(loads)


# ------------------------------------------------------------------ stages
def stage_zoo(run: Run, a) -> None:
    import zoo
    run.require(run.dataset, run.features_txt, stage="zoo (the dataset is built by the data-build scripts)")
    t0 = time.time()
    df, feats = zoo.load(STATE, run.data_dir)
    log(f"zoo: {len(df):,} points, {len(feats)} model inputs; fitting {len(zoo.zoo_specs(CELL_SEED))} settings")
    cell = zoo.run_cell(df, feats, PROTOCOL, TRAIN_DOM, CELL_SEED, n_jobs=-1, device="cpu")
    if cell is None:
        raise SystemExit("zoo: the cell is unusable (see above); nothing written")
    paths = zoo.write_cell(cell, run.zoo_dir, TAG)
    specs = zoo.zoo_specs(CELL_SEED)
    settings = pd.Series([s["kind"] for s in specs]).value_counts()
    kept = cell["models"].kind.value_counts()
    reasons = cell["dropped"].reason.value_counts() if len(cell["dropped"]) else pd.Series(dtype=int)
    data = zoo.cell_data(df, feats, PROTOCOL, TRAIN_DOM, CELL_SEED)
    from search import model_split
    sel, val, test = model_split(len(cell["models"]))
    families = {k: dict(settings=int(settings.get(k, 0)), models=int(kept.get(k, 0)),
                        degenerate=int(((cell["dropped"].kind == k) & (cell["dropped"].reason == "degenerate")).sum())
                        if len(cell["dropped"]) else 0,
                        duplicate=int(((cell["dropped"].kind == k) & (cell["dropped"].reason == "duplicate")).sum())
                        if len(cell["dropped"]) else 0) for k in sorted(settings.index)}
    info = dict(settings=len(specs), kept=len(cell["models"]), degenerate=int(reasons.get("degenerate", 0)),
                duplicate=int(reasons.get("duplicate", 0)), unusable=int(reasons.get("unusable", 0)),
                families=families, inputs=feats, n_inputs=len(feats), n_train=len(data["tr"]),
                n_id=len(data["id_te"]), n_ood=len(data["ood_te"]), train_tiles=len(data["tr_tiles"]),
                train_tiles_with_points=int(data["tr"].tile_50km.nunique()), test_tiles=len(data["te_tiles"]),
                n_selection=len(sel), n_validation=len(val), n_test=len(test), fit_secs=cell["fit_secs"],
                cell_sha256={p.name: sha256(p) for p in paths})
    if info["kept"] + info["degenerate"] + info["duplicate"] + info["unusable"] != info["settings"]:
        raise SystemExit("zoo: kept + dropped settings do not add up to the grid")
    write_json(run.zoo_dir / "zoo.json", info)
    run.mark("zoo", time.time() - t0, kept=info["kept"], fit_secs=round(cell["fit_secs"], 1))
    log(f"zoo: {info['kept']} models kept ({info['degenerate']} degenerate, {info['duplicate']} duplicate) "
        f"-> {run.zoo_dir}")


def stage_data(run: Run, a) -> None:
    import fig_feature_boxes as ffb
    import graph
    import variogram_ranges
    import zoo
    from pyproj import Transformer
    run.require(run.dataset, run.features_txt, stage="data (the dataset is built by the data-build scripts)")
    t0 = time.time()
    out = run.out / "data"
    df = pd.read_parquet(run.dataset)
    feats = [f for f in zoo.read_features(run.features_txt) if f in df.columns]
    _, inputs = zoo.model_inputs(df, feats)
    sets = ffb.figure_sets(df, CELL_SEED, PROTOCOL)
    counts = df.groupby(["domain", "label"]).size()
    info = dict(points=len(df), features=feats, n_inputs=len(inputs),
                points_by_zone_label={f"{d}_{int(l)}": int(n) for (d, l), n in counts.items()},
                sets={k: dict(n=len(v), flood_share=float((v.label == 1).mean())) for k, v in sets.items()},
                missing_share={f: float(df[f].isna().mean()) for f in feats},
                landcover_top7=ffb.top_classes(df.landcover))
    # Tiles: the 50 km grid of the dataset, and the 110 km grid the Split paragraph compares (EPSG:5070).
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(df.lon.to_numpy(), df.lat.to_numpy())
    tiles = {}
    for size_km in (50, 110):
        ids = (np.floor(x / (size_km * 1000)).astype(int).astype(str) + "_"
               + np.floor(y / (size_km * 1000)).astype(int).astype(str))
        if size_km == 50:
            info["grid_50km_reproduces_tile_50km"] = bool((ids == df.tile_50km.astype(str).to_numpy()).all())
        g = df[["domain", "label", "neg_protocol"]].copy()
        g[zoo.TILE] = ids
        train, test = zoo.split_tiles(g, CELL_SEED)
        tiles[f"{size_km}km"] = dict(tiles=int(len(set(ids))), train_tiles=len(train), test_tiles=len(test))
    info["tiles"] = tiles
    # The graph's spatial scale: each Zone A test point's distance to its 10th nearest neighbour (EPSG:3086).
    ood = sets["OOD"]
    d10 = graph.kth_neighbour_m(ood.lon.to_numpy(), ood.lat.to_numpy(), KNN)
    info["median_10nn_km"] = float(np.median(d10) / 1000)
    vg = variogram_ranges.run(run.data_dir)
    vdir = out / "variogram"
    vdir.mkdir(parents=True, exist_ok=True)
    vg["ranges"].to_csv(vdir / "ranges.csv", index=False)
    vg["features"].to_csv(vdir / "features.csv", index=False)
    vg["empirical"].to_csv(vdir / "empirical.csv", index=False)
    write_json(vdir / "summary.json", vg["summary"])
    r = vg["features"].set_index("feature").range_km
    info["variogram"] = dict(median_km=float(r.median()), ranges_km=r.round(3).to_dict(),
                             above_50km=r[r > 50].sort_values().round(3).to_dict())
    fig = pd.concat([v.assign(set=k) for k, v in sets.items()])[["set", "point_id", *feats]]
    fig.to_parquet(out / "figure1.parquet", index=False)
    write_json(out / "data.json", info)
    run.mark("data", time.time() - t0)
    log(f"data: {len(df):,} points; median variogram range {info['variogram']['median_km']:.1f} km; "
        f"test tiles 50 km {tiles['50km']['test_tiles']}, 110 km {tiles['110km']['test_tiles']} -> {out}")


def load_npz(run: Run):
    run.require(run.zoo_dir / f"{TAG}.npz", run.zoo_dir / f"{TAG}_models.csv", stage="zoo")
    return dict(np.load(run.zoo_dir / f"{TAG}.npz", allow_pickle=True)), pd.read_csv(run.zoo_dir / f"{TAG}_models.csv")


def stage_shift(run: Run, a) -> None:
    import regions
    import shift
    t0 = time.time()
    z, models = load_npz(run)
    ds = pd.read_parquet(run.dataset, columns=["point_id", "tile_50km", "meta_point_county"])
    ds = ds.astype({"point_id": str}).set_index("point_id")
    out = shift.shift(z, models, ds, regions.county_names(run.counties))
    sdir = run.out / "shift"
    sdir.mkdir(parents=True, exist_ok=True)
    out["per_tile"].to_csv(sdir / "per_tile.csv", index=False)
    out["per_model"].to_csv(sdir / "per_model.csv", index=False)
    write_json(sdir / "summary.json", out["summary"])
    run.mark("shift", time.time() - t0)
    s = out["summary"]["metrics"]["acc"]
    log(f"shift: accuracy Delta pooled {s['pooled_delta']:+.3f}, per tile {s['tile_delta']:+.3f}; "
        f"corr_te all Zone A {out['summary']['corr_te_all']:+.3f} -> {sdir}")


def stage_search(run: Run, a, units: list[dict]) -> None:
    run.require(run.zoo_dir / f"{TAG}.npz", stage="zoo")
    d = int(np.load(run.zoo_dir / f"{TAG}.npz", allow_pickle=True)["y_ood"].shape[0])
    todo, skipped = [], []
    for u in units:
        unit_dir = run.out / "search" / u["name"]
        if u["S"] >= d:
            skipped.append(f"{u['name']} (S >= d = {d})")
            continue
        if (unit_dir / "result.json").exists() and not a.force:
            continue
        todo.append(u)
    for s in skipped:
        log(f"search: skipping {s}")
    n_done = len(units) - len(todo) - len(skipped)
    if not todo:
        log(f"search: all {n_done} searches finished")
        run.mark("search", 0.0, units=n_done)
        return
    todo.sort(key=lambda u: -u["est"])
    log(f"search: {len(todo)} to run ({n_done} already finished), {a.jobs} at a time; estimated "
        f"{fmt_secs(makespan([u['est'] for u in todo], a.jobs))} wall")
    (run.logs / "search").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{v: "1" for v in THREAD_VARS})
    running, finished, failed = {}, [], []
    pending = list(todo)
    t0 = time.time()

    def stop_all(*_):
        for p, _u, _t, f in running.values():
            p.terminate()
        raise SystemExit("search: interrupted; finished searches are kept, rerun the same command to resume")
    old_int, old_term = signal.signal(signal.SIGINT, stop_all), signal.signal(signal.SIGTERM, stop_all)
    try:
        while pending or running:
            while pending and len(running) < a.jobs:
                u = pending.pop(0)
                unit_dir = run.out / "search" / u["name"]
                if a.force and unit_dir.exists():
                    for f in ("result.json", "theta.npy", "trials.csv"):
                        (unit_dir / f).unlink(missing_ok=True)
                cmd = [sys.executable, "-u", str(ROOT / "aotil" / "search.py"), "--cell-dir", str(run.zoo_dir),
                       "--tag", TAG, "--out", str(unit_dir), "--S", str(u["S"]), "--method", u["method"],
                       "--mu", f"{u['mu']:g}", "--knn", str(KNN), "--trials", str(a.trials),
                       "--max-steps", str(a.max_steps), "--patience", str(a.patience), "--seed", str(SEED),
                       "--threads", "1"]
                f = open(run.logs / "search" / f"{u['name']}.log", "w")
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
                running[p.pid] = (p, u, time.time(), f)
                log(f"search: start {u['name']} ({u['group']}, est. {fmt_secs(u['est'])})")
            time.sleep(2)
            for pid in list(running):
                p, u, ts, f = running[pid]
                if p.poll() is None:
                    continue
                f.close()
                del running[pid]
                secs = time.time() - ts
                res_path = run.out / "search" / u["name"] / "result.json"
                if p.returncode == 0 and res_path.exists():
                    res = json.loads(res_path.read_text())
                    finished.append(u)
                    run.manifest.setdefault("units", {})[u["name"]] = dict(
                        status="done", secs=round(secs, 1), r_test=res["r_test"], r_val=res["r_val"])
                    run.save()
                    msg = f"r_te {res['r_test']:+.3f}, val {res['r_val']:+.3f}"
                else:
                    failed.append(u)
                    msg = f"FAILED (exit {p.returncode}; see {run.logs / 'search' / (u['name'] + '.log')})"
                left = [x["est"] for x in pending]
                eta = makespan(left + [max(0.0, x[1]["est"] - (time.time() - x[2])) for x in running.values()],
                               a.jobs)
                log(f"search: done {u['name']} in {fmt_secs(secs)}: {msg} | {len(finished) + len(failed)}/"
                    f"{len(todo)} | running {len(running)}: {', '.join(x[1]['name'] for x in running.values())} "
                    f"| pending {len(pending)} | elapsed {fmt_secs(time.time() - t0)}, about {fmt_secs(eta)} left")
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    if failed:
        raise SystemExit(f"search: {len(failed)} failed: {', '.join(u['name'] for u in failed)}; rerun to retry")
    run.mark("search", time.time() - t0, units=n_done + len(finished))


def finished_searches(run: Run, units: list[dict]) -> list[dict]:
    out = []
    for u in units:
        if (run.out / "search" / u["name"] / "result.json").exists():
            out.append(u)
    return out


def stage_smooth(run: Run, a, units: list[dict]) -> None:
    import graph
    import subsets
    t0 = time.time()
    z, _ = load_npz(run)
    arrays = subsets.cell_arrays(z)
    A = graph.build_adjacency(arrays["lon"], arrays["lat"], k=KNN)
    done = finished_searches(run, units)
    if not done:
        raise SystemExit(f"smooth: no finished search: run `run_experiments.py --stage search --out {run.out}`")
    sdir = run.out / "smooth"
    sdir.mkdir(parents=True, exist_ok=True)
    rows, masks = [], {}
    for u in done:
        res = json.loads((run.out / "search" / u["name"] / "result.json").read_text())
        theta = np.load(run.out / "search" / u["name"] / "theta.npy")
        S = u["S"]
        for smoothing in (False, True):
            if smoothing:
                mask, ties = subsets.smooth(theta, A, S)
            else:
                mask, ties = subsets.top_mask(theta, S), 0
            ev = subsets.evaluate(mask, arrays, A, C_MIN)
            rows.append(dict(unit=u["name"], search=SEARCH_NAMES[u["method"]], method=u["method"], mu=u["mu"], S=S,
                             group=u["group"], smoothing=smoothing, **ev, ties_at_cut=ties,
                             r_test_search=res["r_test"], r_val_search=res["r_val"],
                             best_trial=res["best_trial"], soft_mass=res["soft_mass"]))
            masks[f"{u['name']}{'+smoothing' if smoothing else ''}"] = np.packbits(mask)
    table = pd.DataFrame(rows)
    plain = table[~table.smoothing]
    gap = float((plain.r_te - plain.r_test_search).abs().max())
    table.to_csv(sdir / "subsets.csv", index=False)
    np.savez_compressed(sdir / "masks.npz", d=np.int64(len(arrays["y"])), **masks)
    run.mark("smooth", time.time() - t0, units=len(done), max_abs_r_te_vs_search=gap)
    log(f"smooth: {len(done)} searches x (alone, + smoothing) -> {sdir / 'subsets.csv'}; corr_te here vs the "
        f"search's own float32 value: max |diff| {gap:.1e}")


def analyzed_mask(run: Run) -> np.ndarray:
    from search import unit_name
    method, mu, S, smoothing = ANALYZED
    name = unit_name(method, mu, S) + ("+smoothing" if smoothing else "")
    run.require(run.out / "smooth" / "masks.npz", stage="smooth")
    m = np.load(run.out / "smooth" / "masks.npz")
    if name not in m.files:
        raise SystemExit(f"{name} is not in {run.out / 'smooth/masks.npz'}: run the search {unit_name(method, mu, S)} "
                         f"and then `--stage smooth`")
    return np.unpackbits(m[name])[: int(m["d"])].astype(bool)


def stage_regions(run: Run, a) -> None:
    import graph
    import regions
    import zoo
    t0 = time.time()
    z, _ = load_npz(run)
    mask = analyzed_mask(run)
    A = graph.build_adjacency(np.asarray(z["lon"], float), np.asarray(z["lat"], float), k=KNN)
    points = regions.load_points(run.dataset, zoo.read_features(run.features_txt))
    rows = regions.ood_rows(points, z)
    out = regions.regions(mask, A, rows, regions.county_names(run.counties), C_MIN)
    rdir = run.out / "regions"
    rdir.mkdir(parents=True, exist_ok=True)
    out["points"].to_parquet(rdir / "points.parquet", index=False)
    out["components"].to_csv(rdir / "components.csv", index=False)
    comps = out["components"]
    run.mark("regions", time.time() - t0, regions=len(comps), points=int(comps.n_points.sum()) if len(comps) else 0)
    log(f"regions: {len(comps)} regions of >= {C_MIN} points holding "
        f"{int(comps.n_points.sum()) if len(comps) else 0} of {int(mask.sum())}: "
        f"{', '.join(f'{n} {k}' for n, k in zip(comps.name, comps.n_points))} -> {rdir}")


def stage_match(run: Run, a) -> None:
    import region_match
    import regions
    import zoo
    t0 = time.time()
    run.require(run.out / "regions" / "points.parquet", run.out / "regions" / "components.csv", stage="regions")
    z, _ = load_npz(run)
    mask = analyzed_mask(run)
    feats = region_match.numeric_features(run.features_txt)
    points = regions.load_points(run.dataset, zoo.read_features(run.features_txt))
    ood = regions.ood_rows(points, z)
    idd = points.loc[np.asarray(z["point_id_id"]).astype(str)].reset_index(drop=True)
    if not np.array_equal(idd.label.to_numpy(), np.asarray(z["y_id"])):
        raise SystemExit("the dataset's ID labels disagree with the cell's")
    pts = pd.read_parquet(run.out / "regions" / "points.parquet")
    comps = pd.read_csv(run.out / "regions" / "components.csv")
    members = {int(c): g.ood_index.to_numpy() for c, g in pts[pts.in_component].groupby("component")}
    ctx = region_match.prepare(ood, idd, feats)
    out = region_match.run(ctx, mask, members)
    names = dict(zip(comps.component.astype(str), comps.name))
    table = out["table"]
    table["name"] = table.set.map(names).fillna(table.set)
    mdir = run.out / "match"
    mdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(mdir / "table.csv", index=False)
    out["pairs"].to_parquet(mdir / "pairs.parquet", index=False)
    out["point_summary"].to_parquet(mdir / "point_summary.parquet", index=False)
    write_json(mdir / "summary.json", out["summary"])
    run.mark("match", time.time() - t0, eps=out["summary"]["eps"])
    pooled = table[table.kind == "pooled"].iloc[0]
    base = table[table.kind == "baseline"].iloc[0]
    log(f"match: eps {out['summary']['eps']:.3f}; regions pooled unmatched {pooled.unmatched:.3f}, eta "
        f"{pooled.eta:.3f}; baseline {base.unmatched:.3f} / {base.eta:.3f} -> {mdir}")


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["all"], choices=list(STAGES) + ["all"],
                    help="stage(s) to run (default all)")
    ap.add_argument("--out", default=f"results/{dt.date.today():%Y-%m-%d}",
                    help="run folder (default results/<today>)")
    ap.add_argument("--jobs", type=int, default=default_jobs(),
                    help="searches at a time, one single-threaded process each (default: performance cores - 2)")
    ap.add_argument("--only", nargs="+", metavar="PATTERN",
                    help="only the searches whose names match (shell patterns, e.g. '*_S500' or gtv_mu3_S500)")
    ap.add_argument("--dry-run", action="store_true", help="print what would run and its estimated time")
    ap.add_argument("--force", action="store_true", help="redo finished units of the selected stages")
    ap.add_argument("--data", default=str(DATA), help=f"dataset folder (default {DATA})")
    ap.add_argument("--counties", default=str(COUNTIES), help=f"county polygons for region names (default {COUNTIES})")
    ap.add_argument("--logs", default=None, help="log folder (default logs/<run folder name>)")
    ap.add_argument("--trials", type=int, default=TRIALS, help=argparse.SUPPRESS)        # tests only
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--patience", type=int, default=PATIENCE, help=argparse.SUPPRESS)    # tests only
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    stages = list(STAGES) if "all" in a.stage else [s for s in STAGES if s in a.stage]
    run = Run(a)
    units = select_units(search_plan(), a.only)

    if a.dry_run:
        print(f"run folder {run.out} (logs {run.logs}); stages {', '.join(stages)}; {a.jobs} searches at a time")
        total = 0.0
        for st in stages:
            if st == "search":
                todo = [u for u in units if a.force or not (run.out / "search" / u["name"] / "result.json").exists()]
                wall = makespan([u["est"] for u in todo], a.jobs)
                total += wall
                print(f"  search: {len(todo)} of {len(units)} to run, {sum(u['est'] for u in todo) / 3600:.1f} "
                      f"CPU hours, about {fmt_secs(wall)} wall")
                for g in ("budget grid", "mu sweep"):
                    names = [u["name"] for u in todo if u["group"] == g]
                    if names:
                        print(f"    {g} ({len(names)}): {' '.join(names)}")
            else:
                done = run.stage_done(st) and not a.force
                total += 0 if done else EST_STAGE_SECS[st]
                print(f"  {st}: {'finished (skipped)' if done else 'to run, about ' + fmt_secs(EST_STAGE_SECS[st])}")
        print(f"estimated total: about {fmt_secs(total)}")
        return 0

    stdout = sys.stdout
    sys.stdout = Tee(run.logs / "run_experiments.log")
    try:
        return run_stages(run, a, stages, units)
    finally:
        sys.stdout.file.close()
        sys.stdout = stdout


def run_stages(run: Run, a, stages: list[str], units: list[dict]) -> int:
    run.out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for p in (run.dataset, run.features_txt, run.counties):
        if Path(p).exists():
            inputs[str(p)] = dict(sha256=sha256(Path(p)), bytes=Path(p).stat().st_size)
    run.save(inputs=inputs)
    log(f"run_experiments: stages {', '.join(stages)} -> {run.out} (logs {run.logs}); git "
        f"{run.manifest['git']['commit'][:10]}{' + uncommitted changes' if run.manifest['git']['dirty_files'] else ''}")
    t_all = time.time()
    for st in stages:
        if st in ("search", "smooth"):
            pass                         # unit-level resumption inside the stage
        elif run.stage_done(st) and not a.force:
            log(f"{st}: finished before; skipped (--force to redo)")
            continue
        log(f"--- {st}")
        if st == "zoo":
            stage_zoo(run, a)
        elif st == "data":
            stage_data(run, a)
        elif st == "shift":
            stage_shift(run, a)
        elif st == "search":
            stage_search(run, a, units)
        elif st == "smooth":
            stage_smooth(run, a, search_plan())
        elif st == "regions":
            stage_regions(run, a)
        elif st == "match":
            stage_match(run, a)
    planned = search_plan()
    complete = (all(run.stage_done(s) for s in STAGES)
                and len(finished_searches(run, planned)) == len(planned))
    run.save(complete=complete)
    log(f"finished in {fmt_secs(time.time() - t_all)}. Wrote {run.out}: "
        + ", ".join(f"{s}/" for s in stages) + " and manifest.json"
        + ("" if complete else " (the run is not complete yet: some stages or searches are missing)"))
    log(f"next: .venv/bin/python make_paper.py --run {run.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
