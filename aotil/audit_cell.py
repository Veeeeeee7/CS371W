#!/usr/bin/env python3
"""
Audit one model-zoo cell: is it, row for row and model for model, what zoo.py says it is?

A misaligned cell (a column of S_ood that belongs to another point, an ID half from other tiles,
a model that cannot be refitted) still gives plausible numbers downstream, so the audit
re-derives the cell from the dataset and the code and compares. The checks, in order: shapes,
models (the kept models and <tag>_dropped.csv against the grid of zoo_specs, the drop rules
(degenerate, then duplicate predictions), and the table against the score matrices), rows (every
column against the dataset), tiles (the split_tiles rule) and determinism (refits of kept and
dropped settings through zoo.cell_data and zoo.fit_one, on the inputs of zoo.model_inputs). Each is PASS, FAIL, WARN (works, but a v1-format limit or something to look at)
or INFO (a number, no verdict); the exit code is 1 on any FAIL. A model table written before the
grid population (2026-11-06, no grid_idx) fails the model checks, since zoo.py no longer builds
it. A v1 cell (no point_id) has its rows checked against the archived v1 dataset and
FEATURES.txt, passed as --data and --features.

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
# The model-table columns zoo.py writes for the grid population.
GRID_TABLE = ["grid_idx", "seed", "n_train", "features", "flood_share_id", "acc_train_sub",
              "auc_train_sub", "acc_train_all", "auc_train_all"]
DROP_REASONS = zoo.DROP_REASONS                             # zoo.split_population's reasons
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
                expect_families=a.expect_families, threads=a.threads, device=a.device)


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


def spec_mismatches(table: pd.DataFrame, specs: list[dict], label: str = "idx") -> list[str]:
    """Rows of a model table (or of <tag>_dropped.csv) that differ from the zoo_specs setting at
    their grid_idx: kind, seed and every hp_<name>. label names the column that identifies a row
    in the messages."""
    hp_cols = [c for c in table.columns if c.startswith("hp_")]
    bad = []
    for _, row in table.iterrows():
        name = f"{label} {row[label]}"
        g = int(row.grid_idx)
        if not 0 <= g < len(specs):
            bad.append(f"{name} grid_idx {g} outside the grid")
            continue
        spec = specs[g]
        if str(row.kind) != spec["kind"]:
            bad.append(f"{name} kind")
            continue
        if int(row.seed) != spec["seed"]:
            bad.append(f"{name} seed")
        for key, value in spec["kw"].items():
            column = f"hp_{key}"
            if column not in table.columns or not hp_same(row[column], zoo.hp_value(value)):
                bad.append(f"{name} {column}")
        for column in hp_cols:
            if column.removeprefix("hp_") not in spec["kw"] and not pd.isna(row[column]):
                bad.append(f"{name} {column} set but not a {spec['kind']} parameter")
    return bad


def grid_partition_problems(models: pd.DataFrame, dropped: pd.DataFrame,
                            n_settings: int) -> list[str]:
    """How the kept models' and the dropped settings' grid_idx fail to cover the grid 0..n-1
    exactly once, in grid order within each table."""
    kept_g = models.grid_idx.to_numpy(int)
    dropped_g = dropped.grid_idx.to_numpy(int)
    bad = []
    if (np.diff(kept_g) <= 0).any():
        bad.append("model table not in grid order")
    if (np.diff(dropped_g) <= 0).any():
        bad.append("dropped table not in grid order")
    both = np.intersect1d(kept_g, dropped_g)
    if both.size:
        bad.append(f"{both.size} settings both kept and dropped, first {int(both[0])}")
    covered = np.union1d(kept_g, dropped_g)
    missing = np.setdiff1d(np.arange(n_settings), covered)
    if missing.size:
        bad.append(f"{missing.size} settings neither kept nor dropped, first {int(missing[0])}")
    outside = covered[(covered < 0) | (covered >= n_settings)]
    if outside.size:
        bad.append(f"grid_idx outside 0..{n_settings - 1}: {outside[:3].tolist()}")
    return bad


def stored_flood_count(share, n_id: int) -> int | None:
    """The number of ID test points a stored flood_share_id stands for: share * n_id rounded,
    or None when the share is not such a count over n_id (to 1e-12; a share read back from a CSV
    can be an ulp off the count's quotient)."""
    share = float(share)
    if not np.isfinite(share):
        return None
    n_flood = int(round(share * n_id))
    if abs(share - n_flood / n_id) > 1e-12:
        return None
    return n_flood


def stored_keys(cell) -> list[bytes]:
    """zoo.prediction_key of every kept model, from its stored S_id and S_ood rows."""
    return [zoo.prediction_key(dict(s_id=cell.S_id[r], s_od=cell.S_ood[r]), cell.y_id,
                               cell.y_ood) for r in range(len(cell.models))]


def drop_rule_problems(cell, dropped: pd.DataFrame) -> list[str]:
    """How the kept and dropped models break the drop rules: degenerate (zoo.is_degenerate on the
    ID test points), then duplicate (zoo.prediction_key, the 0/1 predictions on both test halves).
    A kept model's flood share is recounted from S_id and y_id (zoo.predicts_flood) and must equal
    its flood_share_id and pass the rule, and no two kept models may share their predictions; a
    dropped row must name a known reason, a degenerate one must carry a flood share the rule
    rejects, and a duplicate must name, in duplicate_of, a kept model earlier in grid order with
    its ID flood share (its refit is compared with that model's predictions in check_dropped_refits)."""
    if cell.y_id is None:
        return ["y_id not stored: the ID flood shares cannot be recounted"]
    models, n_id = cell.models, len(cell.y_id)
    bad = []
    for r in range(len(models)):
        n_flood = int(zoo.predicts_flood(cell.S_id[r], cell.y_id).sum())
        name = f"idx {models.idx.iloc[r]}"
        if zoo.is_degenerate(n_flood, n_id):
            bad.append(f"{name} kept but degenerate ({n_flood} of {n_id} predicted flood)")
        if stored_flood_count(models.flood_share_id.iloc[r], n_id) != n_flood:
            bad.append(f"{name} flood_share_id {models.flood_share_id.iloc[r]} != "
                       f"{n_flood}/{n_id}")
    keys = stored_keys(cell)
    first = {}
    for r, key in enumerate(keys):
        if key in first:
            bad.append(f"idx {models.idx.iloc[r]} kept but repeats the predictions of idx "
                       f"{models.idx.iloc[first[key]]}")
        first.setdefault(key, r)
    kept_share = dict(zip(models.grid_idx.astype(int), models.flood_share_id))
    for _, row in dropped.iterrows():
        name = f"grid_idx {row.grid_idx}"
        if row.reason not in DROP_REASONS:
            bad.append(f"{name} reason {row.reason!r}")
        elif row.reason == "degenerate":
            n_flood = stored_flood_count(row.flood_share_id, n_id)
            if n_flood is None or not zoo.is_degenerate(n_flood, n_id):
                bad.append(f"{name} dropped as degenerate with flood share "
                           f"{row.flood_share_id}")
        elif row.reason == "duplicate":
            of = row.get("duplicate_of")
            if of is None or pd.isna(of) or int(of) not in kept_share:
                bad.append(f"{name} duplicate of {of}, not a kept model")
            elif int(of) >= int(row.grid_idx):
                bad.append(f"{name} duplicate of the later grid_idx {int(of)}")
            elif stored_flood_count(row.flood_share_id, n_id) != \
                    stored_flood_count(kept_share[int(of)], n_id):
                bad.append(f"{name} flood share differs from its model's (grid_idx {int(of)})")
    return bad


def column_problems(models: pd.DataFrame, used: list[str], n_tr: int) -> list[str]:
    """Rows whose features, n_train or training metrics are not what fit_one writes. used is
    the input columns every model trains on: those of zoo.model_inputs with two distinct finite
    values in the training rows (zoo.varying_columns), in their order; n_tr is the training rows."""
    bad = []
    for _, row in models.iterrows():
        names = str(row.features).split(";")
        if names != used or len(names) != int(row.n_feat):
            bad.append(f"idx {row.idx} features")
        if int(row.n_train) != n_tr:
            bad.append(f"idx {row.idx} n_train {row.n_train} != {n_tr}")
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


def check_models(report: Report, cell, args, specs: list[dict], dropped: pd.DataFrame | None,
                 used: list[str], n_tr: int) -> bool:
    """The model table and <tag>_dropped.csv (None if absent) against the grid: the families as
    expected, idx contiguous, the kept and dropped settings covering the grid once, every row
    equal to its zoo_specs setting, the drop rule, score_* / acc_* equal to the matrices, and the
    features, n_train and training metrics as fit_one writes them. used and n_tr are
    column_problems'. Returns whether the table is in the grid format (refits need grid_idx)."""
    models = cell.models
    n_models = len(models)
    absent = [c for c in GRID_TABLE if c not in models.columns]
    if absent:
        report.ok("models", "grid table", False,
                  f"model table written before the grid population (no {', '.join(absent)}); "
                  f"zoo.py no longer builds such a cell")
        return False
    if dropped is None:
        report.ok("models", "dropped settings", False,
                  f"{args.tag}_dropped.csv not found next to the cell")
        return False

    by_kind = pd.Series([sp["kind"] for sp in specs]).value_counts()
    counts = models.kind.astype(str).value_counts()
    want = sorted(f for f in args.expect_families.split(",") if f)
    got = sorted(counts.index)
    detail = " ".join(f"{kind} {counts.get(kind, 0)}/{by_kind[kind]}"
                      for kind in sorted(by_kind.index))
    detail = "kept/settings: " + detail
    if got != want:
        detail += f"; expected {want}"
    report.ok("models", "families", got == want, detail)

    idx = models.idx.to_numpy()
    expected_idx = np.arange(n_models)
    if np.array_equal(idx, expected_idx):
        report.ok("models", "idx contiguous", True, "idx = 0..M-1 in order")
    else:
        first_gap = int(np.argmax(idx != expected_idx))
        report.ok("models", "idx contiguous", False,
                  f"idx not 0..{n_models - 1}: first gap at {first_gap}")

    bad = grid_partition_problems(models, dropped, len(specs))
    detail = (f"{n_models} kept + {len(dropped)} dropped = {n_models + len(dropped)} of the "
              f"{len(specs)} settings")
    if bad:
        detail += f"; {'; '.join(bad[:3])}"
    else:
        detail += ", each once, in grid order"
    report.ok("models", "kept + dropped = grid", not bad, detail)

    bad = spec_mismatches(models, specs) + spec_mismatches(dropped, specs, label="grid_idx")
    if bad:
        detail = f"{len(bad)} mismatches, first: {bad[:3]}"
    else:
        detail = (f"kind, seed and every hp_<name> match the setting at grid_idx for all "
                  f"{n_models} kept and {len(dropped)} dropped rows")
    report.ok("models", "table == zoo_specs", not bad, detail)

    bad = drop_rule_problems(cell, dropped)
    reasons = dropped.reason.value_counts()
    detail = (f"kept: ID flood share recounted from S_id in [{float(zoo.DEGENERATE_CUT):g}, "
              f"{float(1 - zoo.DEGENERATE_CUT):g}] and equal to flood_share_id, no two with the "
              f"same predictions on both test halves; dropped: "
              + (", ".join(f"{n} {reason}" for reason, n in reasons.items()) or "none"))
    if bad:
        detail += f"; {len(bad)} problems, first: {bad[:3]}"
    report.ok("models", "drop rule", not bad, detail)

    diffs, loose = table_vs_matrices(cell, args.tol)
    note = ""
    if loose and loose <= {"xgb"}:
        note = f" (> {args.tol:g} in {sorted(loose)}: xgb float32 means)"
    elif loose:
        note = f" (> {args.tol:g} in {sorted(loose)})"
    max_diffs = ", ".join(f"{column} {value:.1e}" for column, value in diffs.items())
    report.ok("models", "table == matrices", max(diffs.values()) <= args.table_tol,
              "max |diff| " + max_diffs + note + "; auc_* checked by the refits")

    bad = column_problems(models, used, n_tr)
    if bad:
        detail = f"{len(bad)} problems, first: {bad[:3]}"
    else:
        detail = (f"every model on all {len(used)} varying features and all {n_tr:,} training "
                  f"rows; training metrics in [0, 1] for all {n_models} rows")
    report.ok("models", "features and rows", not bad, detail)
    report.add("models", "training fit", "INFO",
               f"acc on the fitted rows: median {models.acc_train_sub.median():.3f}, "
               f"max {models.acc_train_sub.max():.3f}, "
               f">= 0.99 in {(models.acc_train_sub >= .99).sum()}; "
               f"on all training rows: median {models.acc_train_all.median():.3f}, "
               f"max {models.acc_train_all.max():.3f}")
    return True


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


# fit_one's metric pairs and the model-table columns they fill: (column, fit_one key, position).
REFIT_METRICS = [(f"{metric}_{name}", name, j)
                 for name in ("id", "ood", "train_sub", "train_all")
                 for j, metric in enumerate(("acc", "auc"))]


def report_refit_summary(report: Report, records: list[dict], tol: float,
                         thread_note: str) -> None:
    """The refits' verdict, with the worst score difference per family."""
    worst_per_kind: dict[str, float] = {}
    for rec in records:
        worst = max(rec.get("max_diff_S_id", np.inf), rec.get("max_diff_S_ood", np.inf))
        worst_per_kind[rec["kind"]] = max(worst_per_kind.get(rec["kind"], 0.0), worst)
    worst_table = max((rec.get("max_diff_table", np.inf) for rec in records), default=np.inf)
    per_kind = ", ".join(f"{kind} {v:.1e}" for kind, v in sorted(worst_per_kind.items()))
    all_ok = bool(records) and all(rec["ok"] for rec in records)
    report.ok("determinism", f"{len(records)} refits match to {tol:g}", all_ok,
              f"max |diff| of the scores per family: {per_kind}; of the table's acc/auc on "
              f"id, ood, train_sub, train_all: {worst_table:.1e} ({thread_note})")


def thread_note() -> str:
    """The BLAS/OpenMP thread settings the refits run under."""
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    veclib = os.environ.get("VECLIB_MAXIMUM_THREADS", "unset")
    return f"threads OMP_NUM_THREADS={omp}, VECLIB_MAXIMUM_THREADS={veclib}"


def check_refits(report: Report, cell, args, specs, rebuilt, maps: dict, feats) -> list[dict]:
    """Refit kept models through zoo.cell_data + zoo.fit_one, the code path zoo.py uses, and
    compare their scores and table metrics with the stored ones to --tol. Returns one record per
    refit. A required family missing from the refits is a FAIL however they were chosen."""
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
    note = thread_note()
    print(f"  refitting {len(picks)} models ({note})", flush=True)

    records = []
    for r in picks:
        row = models.iloc[r]
        spec = specs[int(row.grid_idx)]
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
        d_table = {column: abs(got[key][j] - float(row[column]))
                   for column, key, j in REFIT_METRICS}
        rec = dict(idx=int(row.idx), kind=spec["kind"], kw=str(spec["kw"]),
                   max_diff_S_id=d_id, max_diff_S_ood=d_od,
                   max_diff_table=max(d_table.values()), seconds=round(seconds, 2))
        notes = []
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
        line = (f"    idx {rec['idx']:3d} {rec['kind']:4s} {rec['kw']:80s} "
                f"max|dS_id| {d_id:.1e}  max|dS_ood| {d_od:.1e}  "
                f"max|dtable| {rec['max_diff_table']:.1e}  {seconds:5.1f}s")
        if notes:
            line += f"  {rec['note']}"
        print(line, flush=True)

    report_refit_summary(report, records, args.tol, note)
    return records


def check_dropped_refits(report: Report, specs, dropped: pd.DataFrame, rebuilt,
                         device: str, cell=None, maps: dict | None = None) -> list[dict]:
    """Refit every dropped setting and confirm why it was dropped: a degenerate one again
    predicts flood for its stored share of the ID test points, a share the drop rule rejects; a
    duplicate again predicts, on every ID test point and OOD point, what its duplicate_of model's
    stored scores predict (mapped to the refit's order through maps); an unusable one again
    cannot be fitted. Returns one record per dropped setting."""
    arrays = rebuilt["arrays"]
    n_id = len(arrays["yid"])
    kept_keys = {}
    if cell is not None and maps is not None and maps.get("id_perm") is not None \
            and maps.get("ood_perm") is not None:
        inv_id = np.argsort(maps["id_perm"])
        inv_od = np.argsort(maps["ood_perm"])
        for r, g in enumerate(cell.models.grid_idx.astype(int)):
            kept_keys[g] = zoo.prediction_key(
                dict(s_id=cell.S_id[r][inv_id], s_od=cell.S_ood[r][inv_od]), arrays["yid"],
                arrays["yod"])
    records = []
    for _, row in dropped.iterrows():
        spec = specs[int(row.grid_idx)]
        got = zoo.fit_one(spec, arrays, device)
        rec = dict(grid_idx=int(row.grid_idx), kind=spec["kind"], kw=str(spec["kw"]),
                   reason=str(row.reason))
        if row.reason == "unusable":
            rec["ok"] = got is None
        elif got is None:
            rec["ok"] = False
        elif row.reason == "duplicate":
            rec["n_flood_id"] = got["n_flood_id"]
            of = int(row.duplicate_of) if pd.notna(row.get("duplicate_of")) else None
            key = zoo.prediction_key(got, arrays["yid"], arrays["yod"])
            rec["ok"] = (of is not None and of in kept_keys and key == kept_keys[of]
                         and not zoo.is_degenerate(got["n_flood_id"], n_id))
        else:
            rec["n_flood_id"] = got["n_flood_id"]
            rec["ok"] = (stored_flood_count(row.flood_share_id, n_id) == got["n_flood_id"]
                         and zoo.is_degenerate(got["n_flood_id"], n_id))
        records.append(rec)
        print(f"    grid_idx {rec['grid_idx']:3d} {rec['kind']:4s} {rec['kw']:80s} "
              f"{rec['reason']}: {rec.get('n_flood_id', '-')} of {n_id} flood, "
              f"{'ok' if rec['ok'] else 'MISMATCH'}", flush=True)
    bad = [f"grid_idx {rec['grid_idx']}" for rec in records if not rec["ok"]]
    detail = (f"all {len(records)} refitted: degenerate ones with the same flood share, "
              f"outside the band; duplicates with the predictions of their kept model")
    if not records:
        detail = "none dropped"
    elif bad:
        detail = f"{len(bad)} of {len(records)} do not refit as stored, first {bad[:3]}"
    report.ok("determinism", "dropped settings refit as stored", not bad, detail)
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
        inputs = [npz_path, cells_dir / f"{args.tag}_models.csv",
                  cells_dir / f"{args.tag}_dropped.csv", data, Path(args.features)]
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
    raw = [f for f in zoo.read_features(Path(args.features)) if f in df.columns]
    df, feats = zoo.model_inputs(df, raw)
    print(f"  dataset {len(df):,} rows | {len(raw)} features, {len(feats)} input columns | "
          f"npz keys: {', '.join(npz_keys)}", flush=True)
    # A cell built with zoo.py --extra-train stores its added rows; the refits need them.
    extra_ids, extra_repeat = stored_extra_rows(npz)
    if extra_ids is not None:
        print(f"  extra training rows stored in the cell: {len(extra_ids):,} point_ids "
              f"x {extra_repeat}", flush=True)
    rebuilt = zoo.cell_data(df, feats, protocol, train_dom, args.seed_used,
                            extra_ids, extra_repeat)
    specs = zoo.zoo_specs(args.seed_used)
    dropped_path = cells_dir / f"{args.tag}_dropped.csv"
    dropped = pd.read_csv(dropped_path) if dropped_path.exists() else None
    # The features every model trains on: fit_one drops only the constant ones.
    arrays = rebuilt["arrays"]
    X_fit = arrays["Xtr"]
    if "Xextra" in arrays:
        X_fit = np.vstack([X_fit, arrays["Xextra"]])
    used = [feats[j] for j in zoo.varying_columns(X_fit)]

    report = Report()
    v2 = check_shapes(report, cell, npz_keys)
    grid_table = check_models(report, cell, args, specs, dropped, used, len(rebuilt["tr"]))
    maps = check_rows(report, cell, df, rebuilt, protocol, train_dom, v2)
    check_tiles(report, df, rebuilt, maps["ood_rows"], maps["id_rows"])
    refits, dropped_refits = [], []
    if grid_table:
        refits = check_refits(report, cell, args, specs, rebuilt, maps, feats)
        print(f"  refitting the {len(dropped)} dropped settings", flush=True)
        dropped_refits = check_dropped_refits(report, specs, dropped, rebuilt, args.device,
                                              cell, maps)

    verdict = "PASS" if report.n_fail == 0 else "FAIL"
    n_warn = sum(row["status"] == "WARN" for row in report.rows)
    print_table(report.rows)
    print(f"verdict: {verdict} ({report.n_fail} FAIL, {n_warn} WARN) "
          f"in {time.time() - t0:.0f}s")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = dict(**key, cell_format="v2" if v2 else "v1", checks=report.rows,
                      refits=refits, dropped_refits=dropped_refits, verdict=verdict)
        out.write_text(json.dumps(jsonable(result), indent=1))
        print(f"saved {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
