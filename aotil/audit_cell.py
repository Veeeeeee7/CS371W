#!/usr/bin/env python3
"""
Audit one model-zoo cell: is it, row for row and model for model, what zoo.py says it is?

A misaligned cell (a column of S_ood that belongs to another point, an ID half from other tiles,
a model that cannot be refitted) still gives plausible numbers downstream, so the audit
re-derives the cell from the dataset and the code and compares. The checks, in order: shapes,
models (the table against zoo_specs and the score matrices), rows (every column against the
dataset), tiles (the split_tiles rule) and determinism (refits through zoo.cell_data and
zoo.fit_one). Each is PASS, FAIL, WARN (works, but a v1-format limit or something to look at)
or INFO (a number, no verdict); the exit code is 1 on any FAIL. A v1 cell (no point_id) is
audited against the archived v1 dataset and FEATURES.txt, passed as --data and --features.

    .venv/bin/python -u aotil/audit_cell.py --cells-dir data/interim/aotil/cells \\
        --tag FL_band_AE_s0 --data data/processed/dataset_FL.parquet \\
        --out results/$D/audit/FL_band_AE_s0_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# The refits' BLAS/OpenMP thread count must be set before numpy loads: Apple Accelerate
# (numpy's BLAS on the laptop) reads it once, at load, and threadpoolctl cannot change it. The
# count moves LogisticRegression scores by up to ~1e-11, and cells are fitted with one thread
# per worker (zoo.run_cell), so --threads 1 reproduces them digit for digit.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def set_thread_env() -> None:
    """Set every THREAD_VARS variable to --threads from the command line (default 1)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--threads", type=int, default=1)
    threads = pre.parse_known_args()[0].threads
    for var in THREAD_VARS:
        os.environ[var] = str(threads)


if __name__ == "__main__":
    set_thread_env()

import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zoo                                   # noqa: E402
from cells import load_cell, SPATIAL_COLS    # noqa: E402

TAG_RE = re.compile(r"^(?P<state>[A-Z]{2})_(?P<protocol>band|shadedX)_"
                    r"(?P<train_dom>A|AE)_s(?P<seed>\d+)$")
FAMILIES = "dt,et,hgb,lr,mlp,nb,rf,xgb"                     # zoo_specs' families
V2_KEYS = ["point_id", "point_id_id", "lon_id", "lat_id"]   # npz keys only a v2 cell has
V2_TABLE = ["seed", "n_train", "features", "acc_train_sub", "auc_train_sub",
            "acc_train_all", "auc_train_all"]              # model-table columns of a v2 cell
MISSING_STR = {"", "nan", "<NA>", "None", "NaN"}            # spellings of a missing string


# ----------------------------------------------------------------- helpers
class Report:
    """The PASS/FAIL table, in the order the checks ran."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, group: str, check: str, status: str, detail: str) -> None:
        self.rows.append(dict(group=group, check=check, status=status, detail=detail))
        print(f"  [{status:4s}] {group:11s} {check}: {detail}", flush=True)

    def ok(self, group: str, check: str, cond: bool, detail: str) -> None:
        self.add(group, check, "PASS" if cond else "FAIL", detail)

    @property
    def n_fail(self) -> int:
        return sum(row["status"] == "FAIL" for row in self.rows)


def print_table(rows: list[dict]) -> None:
    """The checks as one fixed-width table."""
    width = max(len(f"{row['group']} / {row['check']}") for row in rows)
    print("\n" + "=" * 100)
    print(f"{'check':{width}s}  status  detail")
    print("-" * 100)
    for row in rows:
        name = f"{row['group']} / {row['check']}"
        print(f"{name:{width}s}  {row['status']:6s}  {row['detail']}")
    print("=" * 100)


def as_cell_array(v: np.ndarray) -> np.ndarray:
    """A dataset column as zoo.main writes it into the npz."""
    return v.astype(str) if v.dtype.kind in "OU" else v


def blank_missing(values: np.ndarray) -> np.ndarray:
    """values as strings, with every spelling of a missing value replaced by ""."""
    return np.array(["" if s in MISSING_STR else s for s in values.astype(str)])


def same_values(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """Element-wise agreement of a cell array and a dataset column.

    Strings compare with every spelling of a missing value as "", so a pandas version that
    renders NA as 'nan' instead of '<NA>' is not a mismatch. Numbers compare NaN-aware, exactly
    up to 1e-12.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, f"length {a.shape} vs {b.shape}"
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        bad = int((blank_missing(a) != blank_missing(b)).sum())
        return bad == 0, f"{bad} differ"
    a, b = a.astype(float), b.astype(float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    diff = np.where(np.isnan(diff), np.inf, diff)       # NaN on one side only
    max_diff = float(diff.max()) if diff.size else 0.0
    return max_diff <= 1e-12, f"max |diff| {max_diff:.2e}"


def hp_same(stored, want) -> bool:
    """A model-table hp_<name> cell against zoo.hp_value(kw[name])."""
    if stored is None or (isinstance(stored, float) and np.isnan(stored)):
        return False
    try:
        return float(stored) == float(want)
    except (TypeError, ValueError):
        return str(stored) == str(want)


def pick_refits(models: pd.DataFrame, n: int, required: list[str], seed: int) -> list[int]:
    """Row positions to refit: one per required family, one per other family, then random
    extras, up to n in all, drawn with a seeded RNG so the audit repeats."""
    rng = np.random.default_rng(seed)
    kinds = models.kind.astype(str).to_numpy()
    order = required + sorted(set(kinds) - set(required))
    chosen: list[int] = []
    for kind in order:
        rows = np.flatnonzero(kinds == kind)
        if len(rows) > 0 and len(chosen) < n:
            chosen.append(int(rng.choice(rows)))
    rest = np.setdiff1d(np.arange(len(models)), chosen)
    n_extra = min(max(n - len(chosen), 0), len(rest))
    for i in rng.choice(rest, n_extra, replace=False):
        chosen.append(int(i))
    return sorted(chosen)


def jsonable(o):
    """o with numpy scalars turned into Python ones, recursively through dicts and lists."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def audit_key(a, data: Path) -> dict:
    """Everything a verdict depends on besides the files' contents. Paths are resolved, so the
    same files named from another directory still match."""
    return dict(tag=a.tag, cells_dir=str(Path(a.cells_dir).resolve()),
                data=str(data.resolve()), features=str(Path(a.features).resolve()),
                seed=a.seed_used, n_refit=a.n_refit, refit_idx=a.refit_idx,
                require_refit=a.require_refit, tol=a.tol, table_tol=a.table_tol,
                expect_models=a.expect_models, expect_families=a.expect_families,
                threads=a.threads, device=a.device)


def stored_report(out: Path, key: dict, inputs: list[Path]) -> tuple[dict | None, str]:
    """The report at out if it is about these inputs as they are now, else (None, why not).

    Current means written with the same key and later than every input: zoo.py --overwrite
    rebuilds a cell in place under the same name, and the old verdict must not be shown for it.
    """
    try:
        prev = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        return None, f"unreadable ({type(e).__name__})"
    if not set(key) <= set(prev):
        return None, "written by an older audit_cell.py, which did not record its settings"
    differ = [k for k in key if prev.get(k) != key[k]]
    if differ:
        settings = [f"{k} ({prev.get(k)!r}, now {key[k]!r})" for k in differ]
        return None, "written for another " + ", ".join(settings)
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        return None, f"{', '.join(missing)} not found"
    written = out.stat().st_mtime_ns
    newer = [p.name for p in inputs if p.stat().st_mtime_ns >= written]
    if newer:
        return None, f"{', '.join(newer)} written after it"
    return prev, ""


def stored_extra_rows(npz) -> tuple[list[str] | None, int]:
    """The --extra-train point_ids and repeat a cell was built with; (None, 1) if none."""
    extra_ids = None
    if "extra_train_point_id" in npz.files:
        extra_ids = [str(x) for x in npz["extra_train_point_id"]]
    extra_repeat = 1
    if "extra_train_repeat" in npz.files:
        extra_repeat = int(npz["extra_train_repeat"])
    return extra_ids, extra_repeat


# ----------------------------------------------------------------- checks
def check_shapes(report: Report, cell, npz_keys: list[str]) -> bool:
    """Every array and the model table agree on the number of models and points, and the scores
    are finite in [0, 1]. Returns whether the cell is v2 (stores point_id for both halves)."""
    n_models, n_ood = cell.S_ood.shape
    n_id = cell.S_id.shape[1]
    missing_v2 = [k for k in V2_KEYS if k not in npz_keys]
    v2 = not missing_v2
    if v2:
        report.add("shapes", "cell format", "PASS",
                   "v2 (point_id, point_id_id, lon_id, lat_id stored)")
    else:
        report.add("shapes", "cell format", "WARN",
                   f"v1-format cell, missing {missing_v2}: columns are mapped by "
                   f"re-deriving the split, not by point_id")

    bad = []
    if cell.S_id.shape[0] != n_models:
        bad.append(f"S_id has {cell.S_id.shape[0]} rows")
    if len(cell.y_ood) != n_ood:
        bad.append(f"y_ood {len(cell.y_ood)}")
    if cell.y_id is not None and len(cell.y_id) != n_id:
        bad.append(f"y_id {len(cell.y_id)}")
    for c in SPATIAL_COLS + ["tiles_ood"]:
        values = getattr(cell, c, None)
        if values is not None and len(values) != n_ood:
            bad.append(f"{c} {len(values)}")
    for c in ("point_id_id", "lon_id", "lat_id"):
        values = getattr(cell, c, None)
        if values is not None and len(values) != n_id:
            bad.append(f"{c} {len(values)}")
    if len(cell.models) != n_models:
        bad.append(f"models.csv {len(cell.models)} rows")
    y_id_state = "stored" if cell.y_id is not None else "absent"
    if bad:
        outcome = f"; MISMATCH: {', '.join(bad)}"
    else:
        outcome = "; every per-point key matches"
    report.ok("shapes", "array shapes", not bad,
              f"S_ood {n_models}x{n_ood}, S_id {n_models}x{n_id}, "
              f"models.csv {len(cell.models)} rows, y_id {y_id_state}" + outcome)

    in_range = all(np.isfinite(S).all() and S.min() >= 0 and S.max() <= 1
                   for S in (cell.S_ood, cell.S_id))
    report.ok("shapes", "scores finite in [0, 1]", in_range,
              f"S_ood [{cell.S_ood.min():.3g}, {cell.S_ood.max():.3g}], "
              f"S_id [{cell.S_id.min():.3g}, {cell.S_id.max():.3g}]")
    return v2


def spec_mismatches(models: pd.DataFrame, specs: list[dict], has_v2: bool) -> list[str]:
    """Model-table rows that differ from zoo_specs: kind, feat_frac, row_frac and, in a v2
    table, seed and every hp_<name>."""
    hp_cols = [c for c in models.columns if c.startswith("hp_")]
    bad = []
    for _, row in models.iterrows():
        i = int(row.idx)
        if i >= len(specs):
            bad.append(f"idx {i} beyond zoo_specs")
            continue
        spec = specs[i]
        if (str(row.kind) != spec["kind"] or float(row.feat_frac) != spec["feat_frac"]
                or float(row.row_frac) != spec["row_frac"]):
            bad.append(f"idx {i} kind/feat_frac/row_frac")
            continue
        if not has_v2:
            continue
        if int(row.seed) != spec["seed"]:
            bad.append(f"idx {i} seed")
        for name, value in spec["kw"].items():
            column = f"hp_{name}"
            if column not in models.columns or not hp_same(row[column], zoo.hp_value(value)):
                bad.append(f"idx {i} {column}")
        for column in hp_cols:
            if column.removeprefix("hp_") not in spec["kw"] and not pd.isna(row[column]):
                bad.append(f"idx {i} {column} set but not a {spec['kind']} parameter")
    return bad


def v2_column_problems(models: pd.DataFrame, feats: list[str], n_tr: int) -> list[str]:
    """Rows whose features, n_train or training metrics are not what fit_one writes."""
    bad = []
    for _, row in models.iterrows():
        names = str(row.features).split(";")
        name_set = set(names)
        if len(names) != int(row.n_feat) or not name_set <= set(feats):
            bad.append(f"idx {row.idx} features")
        elif names != [f for f in feats if f in name_set]:
            bad.append(f"idx {row.idx} feature order")
        n_want = min(max(zoo.MIN_TRAIN, int(n_tr * float(row.row_frac))), n_tr)
        if int(row.n_train) != n_want:
            bad.append(f"idx {row.idx} n_train {row.n_train} != {n_want}")
        metrics = row[["acc_train_sub", "auc_train_sub", "acc_train_all", "auc_train_all"]]
        if metrics.isna().any():
            bad.append(f"idx {row.idx} training metrics")
        else:
            values = metrics.astype(float)
            if (values < 0).any() or (values > 1).any():
                bad.append(f"idx {row.idx} training metrics")
    return bad


def table_vs_matrices(cell, tol: float) -> tuple[dict[str, float], set[str]]:
    """Max |diff| of each score_* and acc_* column against its S row, and the families with a
    row off by more than tol.

    XGBoost returns float32 probabilities, so its score_* are float32 means, ~1e-8 off the
    float64 mean of the (exactly upcast) stored row: hence check_models' looser --table-tol, and
    the families above tol are named. auc_* is left to the refits: it is ranked on the raw p,
    which S (1 - p for negatives) does not hold exactly (zoo.score_rows).
    """
    models = cell.models
    kinds = models.kind.astype(str).to_numpy()
    diffs, loose = {}, set()
    for half, S, y in (("id", cell.S_id, cell.y_id), ("ood", cell.S_ood, cell.y_ood)):
        # zoo.score_rows' argmax rule: a negative at exactly S = 0.5 is correct.
        if y is None:
            correct = S > 0.5
        else:
            negative = np.asarray(y)[None, :] == 0
            correct = (S > 0.5) | ((S == 0.5) & negative)
        for column, values in ((f"score_{half}", S.mean(1)), (f"acc_{half}", correct.mean(1))):
            err = np.abs(values - models[column].to_numpy())
            diffs[column] = float(err.max())
            loose |= set(kinds[err > tol])
    return diffs, loose


def check_models(report: Report, cell, args, specs: list[dict], feats: list[str],
                 n_tr: int) -> None:
    """The model table: model count and families as expected, idx contiguous (no fit skipped),
    every row equal to zoo_specs, score_* / acc_* equal to the matrices and, in a v2 table, the
    features, n_train and training metrics consistent."""
    models = cell.models
    n_models = len(models)
    report.ok("models", "model count", n_models == args.expect_models,
              f"{n_models} models (expected {args.expect_models})")

    counts = models.kind.astype(str).value_counts().sort_index()
    want = sorted(f for f in args.expect_families.split(",") if f)
    got = sorted(counts.index)
    balanced = counts.max() - counts.min() <= 1
    detail = " ".join(f"{kind} {n}" for kind, n in counts.items())
    if got != want:
        detail += f"; expected {want}"
    if not balanced:
        detail += "; NOT round-robin balanced"
    report.ok("models", "families", got == want and balanced, detail)

    idx = models.idx.to_numpy()
    expected_idx = np.arange(n_models)
    if np.array_equal(idx, expected_idx):
        report.ok("models", "idx contiguous", True, "idx = 0..M-1 in order (no fit skipped)")
    else:
        first_gap = int(np.argmax(idx != expected_idx))
        report.ok("models", "idx contiguous", False,
                  f"idx not 0..{n_models - 1}: first gap at {first_gap}")

    has_v2 = all(c in models.columns for c in V2_TABLE)
    bad = spec_mismatches(models, specs, has_v2)
    if has_v2:
        what = "kind, feat_frac, row_frac, seed and every hp_<name>"
    else:
        what = "kind, feat_frac, row_frac (v1 table: no seed or hp_ columns)"
    if bad:
        detail = f"{len(bad)} mismatches, first: {bad[:3]}"
    else:
        detail = f"{what} match for all {n_models} rows"
    report.ok("models", "table == zoo_specs", not bad, detail)

    diffs, loose = table_vs_matrices(cell, args.tol)
    note = ""
    if loose and loose <= {"xgb"}:
        note = f" (> {args.tol:g} in {sorted(loose)}: xgb float32 means)"
    elif loose:
        note = f" (> {args.tol:g} in {sorted(loose)})"
    max_diffs = ", ".join(f"{column} {value:.1e}" for column, value in diffs.items())
    report.ok("models", "table == matrices", max(diffs.values()) <= args.table_tol,
              "max |diff| " + max_diffs + note + "; auc_* checked by the refits")

    if not has_v2:
        absent = ", ".join(c for c in V2_TABLE if c not in models.columns)
        report.add("models", "v2 columns", "WARN",
                   f"absent ({absent}, hp_*): cell written before the extended model table")
        return
    bad = v2_column_problems(models, feats, n_tr)
    if bad:
        detail = f"{len(bad)} problems, first: {bad[:3]}"
    else:
        detail = (f"features (names in FEATURES.txt order, count == n_feat), n_train, "
                  f"training metrics in [0, 1] for all {n_models} rows")
    report.ok("models", "v2 columns", not bad, detail)
    report.add("models", "training fit", "INFO",
               f"acc on own subsample: median {models.acc_train_sub.median():.3f}, "
               f"max {models.acc_train_sub.max():.3f}, "
               f">= 0.99 in {(models.acc_train_sub >= .99).sum()}; "
               f"on all training rows: median {models.acc_train_all.median():.3f}, "
               f"max {models.acc_train_all.max():.3f}")


def stored_point_keys(cell, half: str) -> list[tuple[str, np.ndarray, str]]:
    """(name, array stored in the cell, dataset column it must equal) for one half."""
    pairs = []
    if half == "OOD":
        for c in SPATIAL_COLS:
            if c != "point_id" and getattr(cell, c) is not None:
                pairs.append((c, getattr(cell, c), c))
        if cell.tiles_ood is not None:
            pairs.append(("tiles_ood", cell.tiles_ood, "tile_50km"))
    else:
        for c in ("lon_id", "lat_id"):
            if getattr(cell, c) is not None:
                pairs.append((c, getattr(cell, c), c.removesuffix("_id")))
    return pairs


def row_disagreements(sub: pd.DataFrame, y, pairs: list, dom: str, protocol: str,
                      te_tiles) -> tuple[list[str], list[str]]:
    """(problems, names checked) of one half's dataset rows against what the cell stores."""
    problems, checked = [], []
    if y is not None:
        checked.append("label")
        if not np.array_equal(sub.label.to_numpy(), y):
            problems.append(f"label ({int((sub.label.to_numpy() != y).sum())} differ)")
    for name, stored, column in pairs:
        checked.append(name)
        ok, msg = same_values(stored, as_cell_array(sub[column].to_numpy()))
        if not ok:
            problems.append(f"{name} ({msg})")
    checked += ["domain", "protocol", "test tile"]
    other_zone = sub.domain.astype(str) != dom
    if other_zone.any():
        problems.append(f"domain ({int(other_zone.sum())} not {dom})")
    in_protocol = (sub.label == 1) | (sub.neg_protocol.astype(str) == protocol)
    if not in_protocol.all():
        problems.append(f"protocol ({int((~in_protocol).sum())} negatives not {protocol})")
    if not sub.tile_50km.isin(te_tiles).all():
        problems.append("tile (some columns not in a test tile)")
    return problems, checked


def check_same_split(report: Report, half: str, rows: np.ndarray, derived: np.ndarray,
                     n: int) -> None:
    """A v2 half's dataset rows against the rows split_tiles + subset re-derive."""
    check = f"{half} half == re-derived split"
    if set(rows.tolist()) != set(derived.tolist()):
        n_differ = len(set(rows.tolist()) ^ set(derived.tolist()))
        report.ok("rows", check, False,
                  f"re-derived {len(derived):,} points, cell {n:,}, {n_differ} differ")
        return
    same_order = np.array_equal(rows, derived)
    order = "same order" if same_order else "DIFFERENT ORDER"
    report.ok("rows", check, True,
              f"split_tiles + subset re-derive the same {len(derived):,} points, {order}")
    if not same_order:
        report.add("rows", f"{half} order", "WARN",
                   "same points in another order: the dataset was re-sorted "
                   "after the cell was built (harmless for point_id joins)")


def positions_in(derived: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
    """Position of each of rows within derived, or None if one is not there."""
    where = pd.Series(np.arange(len(derived)), index=derived)
    perm = where.reindex(rows).to_numpy()
    if np.isnan(perm).any():
        return None
    return perm.astype(int)


def check_rows(report: Report, cell, df: pd.DataFrame, rebuilt: dict, protocol: str,
               train_dom: str, v2: bool) -> dict:
    """Map each half's columns to dataset rows and check label, keys, zone, protocol and tile.

    A v2 cell is joined through point_id and must also equal the re-derived split. A v1 cell
    has no point_id: its column order is recovered by re-running zoo.split_tiles + zoo.subset,
    accepted only if every stored key agrees, and its ID half stores only labels. df has a
    RangeIndex, so a row position is its index. Returns, per half, the dataset row of every
    column ('ood_rows', 'id_rows') and the column's position in cell_data's frame, the order a
    refit scores in ('ood_perm', 'id_perm'); None where the mapping failed.
    """
    pid = df.point_id.astype(str)
    report.ok("rows", "dataset point_id unique", pid.is_unique,
              f"{len(df):,} rows, {pid.nunique():,} distinct point_id")
    index = pd.Index(pid)

    halves = [("OOD", rebuilt["eval_dom"], rebuilt["ood_te"], cell.point_id, cell.y_ood,
               cell.S_ood.shape[1]),
              ("ID", train_dom, rebuilt["id_te"], cell.point_id_id, cell.y_id,
               cell.S_id.shape[1])]
    out = {}
    for half, dom, frame, point_ids, y, n in halves:
        h = half.lower()
        out[f"{h}_rows"] = None
        out[f"{h}_perm"] = None
        check = f"{half} columns -> dataset"
        derived = frame.index.to_numpy()
        if v2 and pid.is_unique:
            rows = index.get_indexer(np.asarray(point_ids).astype(str))
            if (rows < 0).any():
                report.add("rows", check, "FAIL",
                           f"{int((rows < 0).sum())} of {n} point_id not in the dataset")
                continue
            how = "joined through point_id"
        elif len(derived) != n:
            report.add("rows", check, "FAIL",
                       f"v1 order recovery impossible: the split re-derives "
                       f"{len(derived):,} points, the cell has {n:,}")
            continue
        else:
            rows = derived
            how = "v1: order RECOVERED by re-running zoo.split_tiles + zoo.subset"

        problems, checked = row_disagreements(df.iloc[rows], y, stored_point_keys(cell, half),
                                              dom, protocol, rebuilt["te_tiles"])
        if problems:
            detail = f"{how}; DISAGREE: {'; '.join(problems)}"
        else:
            detail = f"{how}; all {n:,} columns agree on {', '.join(checked)}"
        report.ok("rows", check, not problems, detail)
        if half == "ID" and not v2:
            report.add("rows", "ID columns: v1 limit", "WARN",
                       "v1 cell stores no point_id or lon/lat for the ID half: only its "
                       "labels and count are checked against the re-derived split")
        if v2:
            check_same_split(report, half, rows, derived, n)
        out[f"{h}_rows"] = rows
        out[f"{h}_perm"] = positions_in(derived, rows)

    if v2:
        ood_ids = set(np.asarray(cell.point_id).astype(str))
        id_ids = set(np.asarray(cell.point_id_id).astype(str))
        shared = ood_ids & id_ids
        unique = len(ood_ids) == len(cell.point_id) and len(id_ids) == len(cell.point_id_id)
        report.ok("rows", "point_id unique / disjoint", unique and not shared,
                  f"OOD {len(ood_ids):,} distinct of {len(cell.point_id):,}, "
                  f"ID {len(id_ids):,} distinct of {len(cell.point_id_id):,}, "
                  f"shared {len(shared)}")
    return out


def check_tiles(report: Report, df: pd.DataFrame, rebuilt: dict, ood_rows, id_rows) -> None:
    """ID and OOD come from exactly the split's test tiles, none a training tile, and every
    test tile holds positives of both zones (the split_tiles rule)."""
    if ood_rows is None or id_rows is None:
        report.add("tiles", "same test tiles", "FAIL", "row mapping failed; not checkable")
        return
    ood, idd = df.iloc[ood_rows], df.iloc[id_rows]
    t_ood, t_id = set(ood.tile_50km), set(idd.tile_50km)
    te, tr = set(rebuilt["te_tiles"]), set(rebuilt["tr_tiles"])
    in_training = (t_ood | t_id) & tr
    same = t_ood == t_id and t_id == te and not in_training
    report.ok("tiles", "same test tiles", same,
              f"OOD {len(t_ood)} tiles, ID {len(t_id)}, split's test tiles {len(te)}, "
              f"shared {len(t_ood & t_id)}, in a training tile {len(in_training)}")
    per = pd.DataFrame({
        "pos_id": idd[idd.label == 1].groupby("tile_50km").size(),
        "pos_ood": ood[ood.label == 1].groupby("tile_50km").size(),
        "n_id": idd.groupby("tile_50km").size(),
        "n_ood": ood.groupby("tile_50km").size(),
    }).reindex(sorted(te)).fillna(0).astype(int)
    both = ((per.pos_id > 0) & (per.pos_ood > 0)).all()
    report.ok("tiles", "every test tile holds both zones", bool(both),
              f"{len(per)} tiles; per tile min: ID {per.n_id.min()} pts "
              f"({per.pos_id.min()} pos), OOD {per.n_ood.min()} pts ({per.pos_ood.min()} pos)")
    train = rebuilt["tr"]
    report.add("tiles", "split sizes", "INFO",
               f"train {len(train):,} | ID test {len(idd):,} | OOD test {len(ood):,} | "
               f"tiles train {len(tr)} test {len(te)} | positive share ID "
               f"{idd.label.mean():.3f}, OOD {ood.label.mean():.3f}, "
               f"train {train.label.mean():.3f}")


def choose_refits(models: pd.DataFrame, args, required: list[str]) -> list[int]:
    """Row positions to refit: the --refit-idx models, or pick_refits' seeded choice."""
    if args.refit_idx:
        idx = models.idx.to_numpy()
        want = [int(v) for v in args.refit_idx.split(",")]
        absent = set(want) - set(idx.tolist())
        if absent:
            raise SystemExit(f"--refit-idx {sorted(absent)} not in the cell")
        positions = []
        for i in want:
            positions.append(int(np.flatnonzero(idx == i)[0]))
        return positions
    kinds = set(models.kind.astype(str))
    present = [f for f in required if f in kinds]
    return pick_refits(models, args.n_refit, present, args.seed_used)


def check_required_families(report: Report, kinds: np.ndarray, picks: list[int],
                            required: list[str], by_hand: bool) -> None:
    """Every required family is among the refits. Checked on the picks themselves, so a small
    --n-refit or a hand-written --refit-idx cannot quietly drop one."""
    present = set(kinds)
    picked = set(kinds[picks])
    absent = [f for f in required if f not in present]
    unpicked = [f for f in required if f not in absent and f not in picked]
    how = "add an idx of each to --refit-idx" if by_hand else "raise --n-refit"
    problems = []
    if absent:
        problems.append(f"no {', '.join(absent)} model in the cell")
    if unpicked:
        problems.append(f"{', '.join(unpicked)} not among the refits ({how})")
    if problems:
        detail = "; ".join(problems)
    else:
        detail = f"{', '.join(required)} among the {len(picks)} refits"
    report.ok("determinism", "required families refitted", not problems, detail)


def refit_metrics(v2_table: bool) -> tuple[list[str], list[tuple[str, str, int]], list[str]]:
    """(fit_one metric keys, table metrics as (column, fit_one key, position in its pair), the
    columns that count toward the verdict). A v1 table's AUCs were ranked on p rebuilt from S,
    so they are compared but not judged."""
    metric_sets = ["id", "ood"]
    if v2_table:
        metric_sets += ["train_sub", "train_all"]
    metrics = []
    for name in metric_sets:
        metrics.append((f"acc_{name}", name, 0))
        metrics.append((f"auc_{name}", name, 1))
    judged = [column for column, _, _ in metrics if v2_table or not column.startswith("auc_")]
    return metric_sets, metrics, judged


def report_refit_summary(report: Report, records: list[dict], tol: float, v2_table: bool,
                         metric_sets: list[str], thread_note: str) -> None:
    """The refits' verdict, with the worst score difference per family and, for a v1 table,
    the AUC difference left out of it."""
    worst_per_kind: dict[str, float] = {}
    for rec in records:
        worst = max(rec.get("max_diff_S_id", np.inf), rec.get("max_diff_S_ood", np.inf))
        worst_per_kind[rec["kind"]] = max(worst_per_kind.get(rec["kind"], 0.0), worst)
    worst_table = max((rec.get("max_diff_table", np.inf) for rec in records), default=np.inf)
    per_kind = ", ".join(f"{kind} {v:.1e}" for kind, v in sorted(worst_per_kind.items()))
    which = "acc/auc" if v2_table else "acc"
    all_ok = bool(records) and all(rec["ok"] for rec in records)
    report.ok("determinism", f"{len(records)} refits match to {tol:g}", all_ok,
              f"max |diff| of the scores per family: {per_kind}; of the table's {which} on "
              f"{', '.join(metric_sets)}: {worst_table:.1e} ({thread_note})")
    if not v2_table and records:
        worst = max(records, key=lambda rec: rec.get("v1_auc_diff", -1.0))
        report.add("determinism", "v1 table AUCs", "WARN",
                   f"auc_id/auc_ood of a table written before 2026-10-14 are ranked on p "
                   f"rebuilt from S, which breaks ties (zoo.score_rows); against the "
                   f"refits' raw-p AUC: max |diff| {worst.get('v1_auc_diff', np.nan):.1e} "
                   f"(idx {worst['idx']} {worst['kind']})")


def check_refits(report: Report, cell, args, specs, rebuilt, maps: dict, feats,
                 v2: bool) -> list[dict]:
    """Refit models through zoo.cell_data + zoo.fit_one, the code path zoo.py uses, and compare
    their scores and table metrics with the stored ones to --tol. Returns one record per refit.
    A required family missing from the refits is a FAIL however they were chosen."""
    models = cell.models
    kinds = models.kind.astype(str).to_numpy()
    required = [f for f in args.require_refit.split(",") if f]
    picks = choose_refits(models, args, required)
    if required:
        check_required_families(report, kinds, picks, required, bool(args.refit_idx))
    p_od, p_id = maps["ood_perm"], maps["id_perm"]
    if p_od is None or p_id is None:
        report.add("determinism", "refits", "FAIL", "a half is unmapped; scores not comparable")
        return []
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    veclib = os.environ.get("VECLIB_MAXIMUM_THREADS", "unset")
    thread_note = f"threads OMP_NUM_THREADS={omp}, VECLIB_MAXIMUM_THREADS={veclib}"
    print(f"  refitting {len(picks)} models ({thread_note})", flush=True)

    v2_table = all(c in models.columns for c in V2_TABLE)
    metric_sets, metrics, judged = refit_metrics(v2_table)
    records = []
    for r in picks:
        row = models.iloc[r]
        spec = specs[int(row.idx)]
        t0 = time.time()
        got = zoo.fit_one(spec, rebuilt["arrays"], args.device)
        seconds = time.time() - t0
        if got is None:
            records.append(dict(idx=int(row.idx), kind=spec["kind"], kw=str(spec["kw"]),
                                ok=False, note="fit_one returned None"))
            print(f"    idx {int(row.idx):3d} {spec['kind']:4s} refit returned None", flush=True)
            continue
        d_id = float(np.abs(got["s_id"][p_id] - cell.S_id[r]).max())
        d_od = float(np.abs(got["s_od"][p_od] - cell.S_ood[r]).max())
        diff = {}
        for column, key, j in metrics:
            diff[column] = abs(got[key][j] - float(row[column]))
        d_table = {c: diff[c] for c in judged}
        v1_auc = {c: v for c, v in diff.items() if c not in d_table}
        rec = dict(idx=int(row.idx), kind=spec["kind"], kw=str(spec["kw"]),
                   max_diff_S_id=d_id, max_diff_S_ood=d_od,
                   max_diff_table=max(d_table.values()), seconds=round(seconds, 2))
        if v1_auc:
            rec["v1_auc_diff"] = max(v1_auc.values())
        notes = []
        if v2 and "features" in models.columns:
            if ";".join(feats[j] for j in got["cols"]) != str(row.features):
                notes.append("features differ")
            if int(row.n_train) != got["n_train"]:
                notes.append("n_train differs")
        off = [f"{c} {v:.1e}" for c, v in d_table.items() if v > args.tol]
        if off:
            notes.append("table differs: " + ", ".join(off))
        rec["ok"] = max(d_id, d_od) <= args.tol and not notes
        rec["note"] = "; ".join(notes)
        records.append(rec)
        line = (f"    idx {rec['idx']:3d} {rec['kind']:4s} {rec['kw']:58s} "
                f"max|dS_id| {d_id:.1e}  max|dS_ood| {d_od:.1e}  "
                f"max|dtable| {rec['max_diff_table']:.1e}  {seconds:5.1f}s")
        if v1_auc:
            line += f"  v1 auc {rec['v1_auc_diff']:.1e}"
        if notes:
            line += f"  {rec['note']}"
        print(line, flush=True)

    report_refit_summary(report, records, args.tol, v2_table, metric_sets, thread_note)
    return records


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default="data/interim/aotil/cells")
    ap.add_argument("--tag", required=True, help="e.g. FL_band_AE_s0")
    ap.add_argument("--data", default="data/processed",
                    help="the dataset parquet the cell was built from, or the directory "
                         "holding dataset_<state>.parquet")
    ap.add_argument("--features", default="data/processed/FEATURES.txt")
    ap.add_argument("--seed", type=int, default=None,
                    help="split/zoo seed; default: the tag's")
    ap.add_argument("--n-refit", type=int, default=10)
    ap.add_argument("--refit-idx", default="",
                    help="comma-separated model idx to refit instead of the automatic pick")
    ap.add_argument("--require-refit", default="xgb,mlp,hgb",
                    help="families the refits must include, also with --refit-idx "
                         "(FAIL otherwise); '' for none")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="refitted scores and table metrics vs the stored ones")
    ap.add_argument("--table-tol", type=float, default=1e-6,
                    help="model-table summaries vs the matrices (xgb means are float32)")
    ap.add_argument("--expect-models", type=int, default=300)
    ap.add_argument("--expect-families", default=FAMILIES)
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS/OpenMP threads for the refits, set before numpy loads "
                         "(1 = how cells are fitted)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="optional JSON report")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-audit even when --out is a current report of this cell")
    args = ap.parse_args()

    match = TAG_RE.match(args.tag)
    if not match:
        raise SystemExit(f"cannot parse tag {args.tag!r} (want e.g. FL_band_AE_s0)")
    state, protocol, train_dom = match["state"], match["protocol"], match["train_dom"]
    tag_seed = int(match["seed"])
    # Stored on args because audit_key and choose_refits read it from there.
    args.seed_used = tag_seed if args.seed is None else args.seed
    if args.seed is not None and args.seed != tag_seed:
        print(f"note: --seed {args.seed} overrides the tag's seed {match['seed']}")
    data = Path(args.data)
    if data.is_dir():
        data = data / f"dataset_{state}.parquet"
    cells_dir = Path(args.cells_dir)
    npz_path = cells_dir / f"{args.tag}.npz"

    key = audit_key(args, data)
    if args.out and Path(args.out).exists() and not args.overwrite:
        inputs = [npz_path, cells_dir / f"{args.tag}_models.csv", data, Path(args.features)]
        prev, why = stored_report(Path(args.out), key, inputs)
        if prev is not None:
            print(f"{args.out} is current (same audit, later than the cell and the dataset), "
                  f"showing it (--overwrite to re-audit)")
            print_table(prev["checks"])
            print(f"verdict: {prev['verdict']}")
            return 0 if prev["verdict"] == "PASS" else 1
        print(f"{args.out} exists but is stale: {why}. Re-auditing.", flush=True)

    t0 = time.time()
    print(f"[audit] {args.tag} | cells {args.cells_dir} | data {data} | "
          f"features {args.features} | seed {args.seed_used}", flush=True)
    cell = load_cell(args.cells_dir, args.tag)
    npz = np.load(npz_path, allow_pickle=True)
    npz_keys = list(npz.files)
    # Positions, not labels, identify a row from here on.
    df = pd.read_parquet(data).reset_index(drop=True)
    feats = [f for f in zoo.read_features(Path(args.features)) if f in df.columns]
    print(f"  dataset {len(df):,} rows | {len(feats)} features | "
          f"npz keys: {', '.join(npz_keys)}", flush=True)
    # A cell built with zoo.py --extra-train stores its added rows; the refits need them.
    extra_ids, extra_repeat = stored_extra_rows(npz)
    if extra_ids is not None:
        print(f"  extra training rows stored in the cell: {len(extra_ids):,} point_ids "
              f"x {extra_repeat}", flush=True)
    rebuilt = zoo.cell_data(df, feats, protocol, train_dom, args.seed_used,
                            extra_ids, extra_repeat)
    # zoo_specs(n) is a prefix of zoo_specs(N > n), so this reproduces the specs of the run
    # that built the cell whatever its --n-models was.
    specs = zoo.zoo_specs(int(cell.models.idx.max()) + 1, args.seed_used)

    report = Report()
    v2 = check_shapes(report, cell, npz_keys)
    check_models(report, cell, args, specs, feats, len(rebuilt["tr"]))
    maps = check_rows(report, cell, df, rebuilt, protocol, train_dom, v2)
    check_tiles(report, df, rebuilt, maps["ood_rows"], maps["id_rows"])
    refits = check_refits(report, cell, args, specs, rebuilt, maps, feats, v2)

    verdict = "PASS" if report.n_fail == 0 else "FAIL"
    n_warn = sum(row["status"] == "WARN" for row in report.rows)
    print_table(report.rows)
    print(f"verdict: {verdict} ({report.n_fail} FAIL, {n_warn} WARN) "
          f"in {time.time() - t0:.0f}s")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = dict(**key, cell_format="v2" if v2 else "v1",
                      checks=report.rows, refits=refits, verdict=verdict)
        out.write_text(json.dumps(jsonable(result), indent=1))
        print(f"saved {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
