#!/usr/bin/env python3
"""
Audit one model-zoo cell: is it, row for row and model for model, what zoo.py
says it is?

WHY
---
Every downstream number (the aggregate ID->OOD r, the OODSelect subsets, the
regions and their polygons) is read off a cell's two score matrices. A cell
that is silently misaligned with the point table -- a column of S_ood that
belongs to a different point, an ID half drawn from other tiles, a model that
cannot be refitted -- would still produce plausible numbers. So the audit
re-derives the cell from the dataset and the code and compares, instead of
inspecting the cell on its own.

WHAT IT CHECKS
--------------
shapes      S_ood (M x n_ood), S_id (M x n_id), labels and every per-point key
            of matching length, the model table with M rows, scores finite in
            [0, 1], point_id unique in each half and disjoint across halves.
models      M == --expect-models, the families present == --expect-families,
            idx contiguous (no fit was skipped), each row's kind / feat_frac /
            row_frac (and, in v2 tables, seed and every hp_<name>) equal to
            zoo_specs, the table's score_* and acc_* equal to the matrices (to
            --table-tol: xgb probabilities are float32, so its score_* are
            float32 means), and the v2 columns (features, n_train, training
            metrics) consistent. auc_* is not re-derived from the matrices: S
            holds 1 - p for negatives, and p rebuilt from it breaks ties
            (zoo.score_rows), so the refits check it on the raw p instead.
rows        every column of S_ood / S_id is joined to the dataset through
            point_id and must agree on label, lon/lat, tile, domain, protocol
            and every stored per-point key; the halves must also equal the ones
            split_tiles + subset re-derive. A v1 cell has no point_id: its
            column order is RECOVERED by re-running zoo.split_tiles + zoo.subset
            on the dataset, accepted only if lon/lat, labels and the stored keys
            agree for every column, and the table says so. The v1 ID half
            stores only labels, so only its labels and count can be checked.
tiles       ID and OOD come from the same test tiles, those are exactly the
            split's test tiles, none is a training tile, and every test tile
            holds positives of both zones in the cell (the split_tiles rule).
determinism --n-refit models (one per --require-refit family, default xgb, mlp
            and hgb, then one per remaining family, then random extras; or the
            --refit-idx list) are refitted through zoo.cell_data + zoo.fit_one
            -- the code path zoo.py itself uses. Their scores must match the
            stored rows to --tol, and so must their table metrics: acc/auc on
            ID and OOD and, in v2 tables, the training metrics. A required
            family missing from the refits is a FAIL, whichever way they were
            chosen. A v1 table's AUCs were ranked on the rebuilt p, so there
            the AUC difference is reported (WARN), not failed. The refits run
            with BLAS/OpenMP capped at --threads (default 1, the setting
            run_oodselect_local.sh builds cells with), because a different
            thread count is the one legitimate source of last-digit
            differences. The cap is set through the environment before numpy
            loads, since Apple Accelerate (numpy's BLAS on the laptop) ignores
            threadpoolctl.

Statuses: PASS, FAIL, WARN (works, but a v1 limitation or something to look
at), INFO (a number for the report, no verdict). Exit code 1 on any FAIL.

An existing --out report is shown again instead of re-auditing only when it is
about this cell as it is now: written by the same audit (tag, cells dir,
dataset, feature list, seed and every audit setting) and later than the cell's
.npz, its _models.csv, the dataset and the feature list. zoo.py --overwrite
rebuilds a cell in place under the same name, and the old cell's verdict must
not be shown as the new one's. Otherwise the audit runs and replaces the report.

USAGE (repo root)
-----
    .venv/bin/python -u aotil/audit_cell.py --cells-dir data/interim/aotil/cells \\
        --tag FL_band_AE_s0 --data data/processed/dataset_FL.parquet \\
        --out results/$D/audit/FL_band_AE_s0_audit.json
    # a v1 cell against the archived v1 dataset and feature list
    .venv/bin/python -u aotil/audit_cell.py \\
        --cells-dir data/interim/archive/sampling-v1/data/interim/aotil/cells \\
        --tag FL_band_AE_s0 \\
        --data data/interim/archive/sampling-v1/data/processed/dataset_FL.parquet \\
        --features data/interim/archive/sampling-v1/data/processed/FEATURES.txt
    # a small test cell: relax the population expectations
    ... --expect-models 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# The refits' BLAS/OpenMP thread count is fixed BEFORE numpy loads: Apple
# Accelerate (numpy's BLAS on the laptop) reads it once, at load, and
# threadpoolctl cannot see it. Its thread count moves LogisticRegression scores
# by up to ~1e-11. Cells are fitted with one thread per worker (zoo.run_cell),
# so --threads 1 is the setting that reproduces them digit for digit.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--threads", type=int, default=1)
    for _v in THREAD_VARS:
        os.environ[_v] = str(_pre.parse_known_args()[0].threads)

import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zoo                                   # noqa: E402
from cells import load_cell, SPATIAL_COLS    # noqa: E402

TAG_RE = re.compile(r"^(?P<state>[A-Z]{2})_(?P<protocol>band|shadedX)_"
                    r"(?P<train_dom>A|AE)_s(?P<seed>\d+)$")
FAMILIES = "dt,et,hgb,lr,mlp,nb,rf,xgb"
V2_KEYS = ["point_id", "point_id_id", "lon_id", "lat_id"]
V2_TABLE = ["seed", "n_train", "features", "acc_train_sub", "auc_train_sub",
            "acc_train_all", "auc_train_all"]
MISSING_STR = {"", "nan", "<NA>", "None", "NaN"}


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
        return sum(r["status"] == "FAIL" for r in self.rows)


def print_table(rows: list[dict]) -> None:
    w_c = max(len(f"{r['group']} / {r['check']}") for r in rows)
    print("\n" + "=" * 100)
    print(f"{'check':{w_c}s}  status  detail")
    print("-" * 100)
    for r in rows:
        print(f"{r['group'] + ' / ' + r['check']:{w_c}s}  {r['status']:6s}  {r['detail']}")
    print("=" * 100)


def as_cell_array(v: np.ndarray) -> np.ndarray:
    """A dataset column as zoo.main writes it into the npz."""
    return v.astype(str) if v.dtype.kind in "OU" else v


def same_values(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """Element-wise agreement of a cell array and a dataset column.

    Strings compare after mapping every spelling of a missing value to "", so
    a pandas version that renders NA as 'nan' instead of '<NA>' is not a
    mismatch. Numbers compare NaN-aware, exactly up to 1e-12.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, f"length {a.shape} vs {b.shape}"
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        ca = np.array(["" if s in MISSING_STR else s for s in a.astype(str)])
        cb = np.array(["" if s in MISSING_STR else s for s in b.astype(str)])
        bad = int((ca != cb).sum())
        return bad == 0, f"{bad} differ"
    a, b = a.astype(float), b.astype(float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    diff = np.where(np.isnan(diff), np.inf, diff)
    mx = float(diff.max()) if diff.size else 0.0
    return mx <= 1e-12, f"max |diff| {mx:.2e}"


def hp_same(stored, want) -> bool:
    """A model-table hp_<name> cell against zoo.hp_value(kw[name])."""
    if stored is None or (isinstance(stored, float) and np.isnan(stored)):
        return False
    try:
        return float(stored) == float(want)
    except (TypeError, ValueError):
        return str(stored) == str(want)


def pick_refits(models: pd.DataFrame, n: int, required: list[str], seed: int) -> list[int]:
    """Row positions to refit: one per required family, one per other family,
    then random extras, all drawn with a seeded RNG so the audit repeats."""
    rng = np.random.default_rng(seed)
    kinds = models.kind.astype(str).to_numpy()
    order = required + sorted(set(kinds) - set(required))
    chosen: list[int] = []
    for k in order:
        rows = np.where(kinds == k)[0]
        if len(rows) and len(chosen) < n:
            chosen.append(int(rng.choice(rows)))
    rest = np.setdiff1d(np.arange(len(models)), chosen)
    k_extra = min(max(n - len(chosen), 0), len(rest))
    chosen += [int(i) for i in rng.choice(rest, k_extra, replace=False)]
    return sorted(chosen)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def audit_key(a, data: Path) -> dict:
    """Everything a verdict depends on besides the files' contents. Paths are
    resolved, so the same files named from another directory still match."""
    return dict(tag=a.tag, cells_dir=str(Path(a.cells_dir).resolve()),
                data=str(data.resolve()), features=str(Path(a.features).resolve()),
                seed=a.seed_used, n_refit=a.n_refit, refit_idx=a.refit_idx,
                require_refit=a.require_refit, tol=a.tol, table_tol=a.table_tol,
                expect_models=a.expect_models, expect_families=a.expect_families,
                threads=a.threads, device=a.device)


def stored_report(out: Path, key: dict, inputs: list[Path]) -> tuple[dict | None, str]:
    """The report at out if it is about these inputs as they are now, else
    (None, why not)."""
    try:
        prev = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        return None, f"unreadable ({type(e).__name__})"
    if not set(key) <= set(prev):
        return None, "written by an older audit_cell.py, which did not record its settings"
    differ = [k for k in key if prev.get(k) != key[k]]
    if differ:
        return None, "written for another " + ", ".join(
            f"{k} ({prev.get(k)!r}, now {key[k]!r})" for k in differ)
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        return None, f"{', '.join(missing)} not found"
    t = out.stat().st_mtime_ns
    newer = [p.name for p in inputs if p.stat().st_mtime_ns >= t]
    if newer:
        return None, f"{', '.join(newer)} written after it"
    return prev, ""


# ----------------------------------------------------------------- checks
def check_shapes(R: Report, cell, npz_keys: list[str]) -> dict:
    M, n_ood = cell.S_ood.shape
    n_id = cell.S_id.shape[1]
    missing_v2 = [k for k in V2_KEYS if k not in npz_keys]
    v2 = not missing_v2
    if v2:
        R.add("shapes", "cell format", "PASS", "v2 (point_id, point_id_id, lon_id, lat_id stored)")
    else:
        R.add("shapes", "cell format", "WARN",
              f"v1-format cell, missing {missing_v2}: columns are mapped by "
              f"re-deriving the split, not by point_id")

    bad = []
    if cell.S_id.shape[0] != M:
        bad.append(f"S_id has {cell.S_id.shape[0]} rows")
    if len(cell.y_ood) != n_ood:
        bad.append(f"y_ood {len(cell.y_ood)}")
    if cell.y_id is not None and len(cell.y_id) != n_id:
        bad.append(f"y_id {len(cell.y_id)}")
    for c in SPATIAL_COLS + ["tiles_ood"]:
        v = getattr(cell, c, None)
        if v is not None and len(v) != n_ood:
            bad.append(f"{c} {len(v)}")
    for c in ("point_id_id", "lon_id", "lat_id"):
        v = getattr(cell, c, None)
        if v is not None and len(v) != n_id:
            bad.append(f"{c} {len(v)}")
    if len(cell.models) != M:
        bad.append(f"models.csv {len(cell.models)} rows")
    R.ok("shapes", "array shapes", not bad,
         f"S_ood {M}x{n_ood}, S_id {M}x{n_id}, models.csv {len(cell.models)} rows, "
         f"y_id {'stored' if cell.y_id is not None else 'absent'}"
         + (f"; MISMATCH: {', '.join(bad)}" if bad else "; every per-point key matches"))

    fin = all(np.isfinite(S).all() and S.min() >= 0 and S.max() <= 1
              for S in (cell.S_ood, cell.S_id))
    R.ok("shapes", "scores finite in [0, 1]", fin,
         f"S_ood [{cell.S_ood.min():.3g}, {cell.S_ood.max():.3g}], "
         f"S_id [{cell.S_id.min():.3g}, {cell.S_id.max():.3g}]")
    return dict(M=M, n_ood=n_ood, n_id=n_id, v2=v2)


def check_models(R: Report, cell, a, specs: list[dict], feats: list[str],
                 n_tr: int) -> None:
    m = cell.models
    M = len(m)
    R.ok("models", "model count", M == a.expect_models,
         f"{M} models (expected {a.expect_models})")

    counts = m.kind.astype(str).value_counts().sort_index()
    want = sorted(f for f in a.expect_families.split(",") if f)
    got = sorted(counts.index)
    balanced = counts.max() - counts.min() <= 1
    R.ok("models", "families", got == want and balanced,
         " ".join(f"{k} {v}" for k, v in counts.items())
         + ("" if got == want else f"; expected {want}")
         + ("" if balanced else "; NOT round-robin balanced"))

    idx = m.idx.to_numpy()
    R.ok("models", "idx contiguous", np.array_equal(idx, np.arange(M)),
         "idx = 0..M-1 in order (no fit skipped)" if np.array_equal(idx, np.arange(M))
         else f"idx not 0..{M-1}: first gap at {int(np.argmax(idx != np.arange(M)))}")

    # The table against the code that claims to have produced it.
    hp_cols = [c for c in m.columns if c.startswith("hp_")]
    has_v2 = all(c in m.columns for c in V2_TABLE)
    bad = []
    for _, row in m.iterrows():
        i = int(row.idx)
        if i >= len(specs):
            bad.append(f"idx {i} beyond zoo_specs"); continue
        sp = specs[i]
        if (str(row.kind) != sp["kind"] or float(row.feat_frac) != sp["feat_frac"]
                or float(row.row_frac) != sp["row_frac"]):
            bad.append(f"idx {i} kind/feat_frac/row_frac"); continue
        if has_v2:
            if int(row.seed) != sp["seed"]:
                bad.append(f"idx {i} seed")
            for k, v in sp["kw"].items():
                if f"hp_{k}" not in m.columns or not hp_same(row[f"hp_{k}"], zoo.hp_value(v)):
                    bad.append(f"idx {i} hp_{k}")
            for c in hp_cols:
                if c[3:] not in sp["kw"] and not pd.isna(row[c]):
                    bad.append(f"idx {i} {c} set but not a {sp['kind']} parameter")
    what = ("kind, feat_frac, row_frac, seed and every hp_<name>" if has_v2
            else "kind, feat_frac, row_frac (v1 table: no seed or hp_ columns)")
    R.ok("models", "table == zoo_specs", not bad,
         f"{what} match for all {M} rows" if not bad
         else f"{len(bad)} mismatches, first: {bad[:3]}")

    # The table against the matrices it is supposed to summarise. XGBoost
    # returns float32 probabilities, so its rows' score_* are float32 means,
    # ~1e-8 off the float64 mean of the (exactly upcast) stored row; hence a
    # separate, looser tolerance here, and the families that need it are named.
    # auc_* is left to the refits: it is ranked on the raw p, which S (1 - p
    # for negatives) does not hold exactly.
    diffs, loose = {}, set()
    kinds = m.kind.astype(str).to_numpy()
    for h, S in (("id", cell.S_id), ("ood", cell.S_ood)):
        for c, v in ((f"score_{h}", S.mean(1)), (f"acc_{h}", (S > .5).mean(1))):
            e = np.abs(v - m[c].to_numpy())
            diffs[c] = float(e.max())
            loose |= set(kinds[e > a.tol])
    note = ""
    if loose:
        note = f" (> {a.tol:g} in {sorted(loose)}" + (
            ": xgb float32 means)" if loose <= {"xgb"} else ")")
    R.ok("models", "table == matrices", max(diffs.values()) <= a.table_tol,
         "max |diff| " + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()) + note
         + "; auc_* checked by the refits")

    if not has_v2:
        R.add("models", "v2 columns", "WARN",
              f"absent ({', '.join(c for c in V2_TABLE if c not in m.columns)}, hp_*): "
              f"cell written before the extended model table")
        return
    bad = []
    for _, row in m.iterrows():
        names = str(row.features).split(";")
        if len(names) != int(row.n_feat) or not set(names) <= set(feats):
            bad.append(f"idx {row.idx} features")
        elif names != [f for f in feats if f in set(names)]:
            bad.append(f"idx {row.idx} feature order")
        n_want = min(max(zoo.MIN_TRAIN, int(n_tr * float(row.row_frac))), n_tr)
        if int(row.n_train) != n_want:
            bad.append(f"idx {row.idx} n_train {row.n_train} != {n_want}")
        tm = row[["acc_train_sub", "auc_train_sub", "acc_train_all", "auc_train_all"]]
        if tm.isna().any() or (tm.astype(float) < 0).any() or (tm.astype(float) > 1).any():
            bad.append(f"idx {row.idx} training metrics")
    R.ok("models", "v2 columns", not bad,
         f"features (names in FEATURES.txt order, count == n_feat), n_train, "
         f"training metrics in [0, 1] for all {M} rows" if not bad
         else f"{len(bad)} problems, first: {bad[:3]}")
    R.add("models", "training fit", "INFO",
          f"acc on own subsample: median {m.acc_train_sub.median():.3f}, "
          f"max {m.acc_train_sub.max():.3f}, >= 0.99 in {(m.acc_train_sub >= .99).sum()}; "
          f"on all training rows: median {m.acc_train_all.median():.3f}, "
          f"max {m.acc_train_all.max():.3f}")


def check_rows(R: Report, cell, df: pd.DataFrame, d: dict, protocol: str,
               train_dom: str, v2: bool) -> dict:
    """Map each half's columns to dataset rows (df has a RangeIndex, so a row
    position is its index). Returns, per half, the dataset row of every column
    ('ood_rows', 'id_rows') and the column's position in cell_data's frame,
    which is the order a refit scores in ('ood_perm', 'id_perm'); None where
    the mapping failed."""
    pid = df.point_id.astype(str)
    R.ok("rows", "dataset point_id unique", pid.is_unique,
         f"{len(df):,} rows, {pid.nunique():,} distinct point_id")
    index = pd.Index(pid)

    out = {}
    for half, dom, frame in (("OOD", d["eval_dom"], d["ood_te"]),
                             ("ID", train_dom, d["id_te"])):
        h = half.lower()
        out[f"{h}_rows"] = out[f"{h}_perm"] = None
        key = cell.point_id if half == "OOD" else cell.point_id_id
        y = cell.y_ood if half == "OOD" else cell.y_id
        n = (cell.S_ood if half == "OOD" else cell.S_id).shape[1]
        derived = frame.index.to_numpy()
        if v2 and pid.is_unique:
            rows = index.get_indexer(np.asarray(key).astype(str))
            if (rows < 0).any():
                R.add("rows", f"{half} columns -> dataset", "FAIL",
                      f"{int((rows < 0).sum())} of {n} point_id not in the dataset")
                continue
            how = "joined through point_id"
        elif len(derived) != n:
            R.add("rows", f"{half} columns -> dataset", "FAIL",
                  f"v1 order recovery impossible: the split re-derives "
                  f"{len(derived):,} points, the cell has {n:,}")
            continue
        else:
            rows = derived
            how = "v1: order RECOVERED by re-running zoo.split_tiles + zoo.subset"

        # (name, array stored in the cell, dataset column it must equal)
        if half == "OOD":
            pairs = [(c, getattr(cell, c), c) for c in SPATIAL_COLS
                     if c != "point_id" and getattr(cell, c) is not None]
            if cell.tiles_ood is not None:
                pairs.append(("tiles_ood", cell.tiles_ood, "tile_50km"))
        else:
            pairs = [(c, getattr(cell, c), c[:-3]) for c in ("lon_id", "lat_id")
                     if getattr(cell, c) is not None]
        sub = df.iloc[rows]
        problems, checked = [], []
        if y is not None:
            checked.append("label")
            if not np.array_equal(sub.label.to_numpy(), y):
                problems.append(f"label ({int((sub.label.to_numpy() != y).sum())} differ)")
        for name, v, col in pairs:
            checked.append(name)
            ok, msg = same_values(v, as_cell_array(sub[col].to_numpy()))
            if not ok:
                problems.append(f"{name} ({msg})")
        checked += ["domain", "protocol", "test tile"]
        if not (sub.domain.astype(str) == dom).all():
            problems.append(f"domain ({int((sub.domain.astype(str) != dom).sum())} not {dom})")
        in_proto = (sub.label == 1) | (sub.neg_protocol.astype(str) == protocol)
        if not in_proto.all():
            problems.append(f"protocol ({int((~in_proto).sum())} negatives not {protocol})")
        if not sub.tile_50km.isin(d["te_tiles"]).all():
            problems.append("tile (some columns not in a test tile)")
        R.ok("rows", f"{half} columns -> dataset", not problems,
             f"{how}; all {n:,} columns agree on {', '.join(checked)}" if not problems
             else f"{how}; DISAGREE: {'; '.join(problems)}")
        if half == "ID" and not v2:
            R.add("rows", "ID columns: v1 limit", "WARN",
                  "v1 cell stores no point_id or lon/lat for the ID half: only its "
                  "labels and count are checked against the re-derived split")
        if v2:
            same_set = set(rows.tolist()) == set(derived.tolist())
            if same_set:
                order = "same order" if np.array_equal(rows, derived) else "DIFFERENT ORDER"
                R.ok("rows", f"{half} half == re-derived split", True,
                     f"split_tiles + subset re-derive the same {len(derived):,} points, {order}")
                if order != "same order":
                    R.add("rows", f"{half} order", "WARN",
                          "same points in another order: the dataset was re-sorted "
                          "after the cell was built (harmless for point_id joins)")
            else:
                R.ok("rows", f"{half} half == re-derived split", False,
                     f"re-derived {len(derived):,} points, cell {n:,}, "
                     f"{len(set(rows.tolist()) ^ set(derived.tolist()))} differ")
        where = pd.Series(np.arange(len(derived)), index=derived)
        perm = where.reindex(rows).to_numpy()
        out[f"{h}_rows"] = rows
        out[f"{h}_perm"] = None if np.isnan(perm).any() else perm.astype(int)

    if v2:
        a_ = set(np.asarray(cell.point_id).astype(str))
        b_ = set(np.asarray(cell.point_id_id).astype(str))
        R.ok("rows", "point_id unique / disjoint",
             len(a_) == len(cell.point_id) and len(b_) == len(cell.point_id_id)
             and not (a_ & b_),
             f"OOD {len(a_):,} distinct of {len(cell.point_id):,}, "
             f"ID {len(b_):,} distinct of {len(cell.point_id_id):,}, shared {len(a_ & b_)}")
    return out


def check_tiles(R: Report, df: pd.DataFrame, d: dict, ood_rows, id_rows) -> None:
    if ood_rows is None or id_rows is None:
        R.add("tiles", "same test tiles", "FAIL", "row mapping failed; not checkable")
        return
    ood, idd = df.iloc[ood_rows], df.iloc[id_rows]
    t_ood, t_id = set(ood.tile_50km), set(idd.tile_50km)
    te, tr = set(d["te_tiles"]), set(d["tr_tiles"])
    R.ok("tiles", "same test tiles", t_ood == t_id == te and not (t_ood | t_id) & tr,
         f"OOD {len(t_ood)} tiles, ID {len(t_id)}, split's test tiles {len(te)}, "
         f"shared {len(t_ood & t_id)}, in a training tile {len((t_ood | t_id) & tr)}")
    per = pd.DataFrame({
        "pos_id": idd[idd.label == 1].groupby("tile_50km").size(),
        "pos_ood": ood[ood.label == 1].groupby("tile_50km").size(),
        "n_id": idd.groupby("tile_50km").size(),
        "n_ood": ood.groupby("tile_50km").size(),
    }).reindex(sorted(te)).fillna(0).astype(int)
    both = ((per.pos_id > 0) & (per.pos_ood > 0)).all()
    R.ok("tiles", "every test tile holds both zones", bool(both),
         f"{len(per)} tiles; per tile min: ID {per.n_id.min()} pts "
         f"({per.pos_id.min()} pos), OOD {per.n_ood.min()} pts ({per.pos_ood.min()} pos)")
    R.add("tiles", "split sizes", "INFO",
          f"train {len(d['tr']):,} | ID test {len(idd):,} | OOD test {len(ood):,} | "
          f"tiles train {len(tr)} test {len(te)} | positive share ID "
          f"{idd.label.mean():.3f}, OOD {ood.label.mean():.3f}, train {d['tr'].label.mean():.3f}")


def check_refits(R: Report, cell, a, specs, d, maps: dict, feats, v2: bool) -> list[dict]:
    m = cell.models
    kinds = m.kind.astype(str).to_numpy()
    required = [f for f in a.require_refit.split(",") if f]
    if a.refit_idx:
        idx = m.idx.to_numpy()
        want = [int(v) for v in a.refit_idx.split(",")]
        if not set(want) <= set(idx.tolist()):
            raise SystemExit(f"--refit-idx {sorted(set(want) - set(idx.tolist()))} not in the cell")
        picks = [int(np.where(idx == i)[0][0]) for i in want]
    else:
        picks = pick_refits(m, a.n_refit, [f for f in required if f in set(kinds)],
                            a.seed_used)
    if required:
        # Enforced on the picks themselves, so a small --n-refit or a
        # hand-written --refit-idx cannot quietly drop a required family.
        absent = [f for f in required if f not in set(kinds)]
        unpicked = [f for f in required if f not in absent and f not in set(kinds[picks])]
        how = "add an idx of each to --refit-idx" if a.refit_idx else "raise --n-refit"
        problems = (([f"no {', '.join(absent)} model in the cell"] if absent else [])
                    + ([f"{', '.join(unpicked)} not among the refits ({how})"] if unpicked else []))
        R.ok("determinism", "required families refitted", not problems,
             "; ".join(problems) or f"{', '.join(required)} among the {len(picks)} refits")
    p_od, p_id = maps["ood_perm"], maps["id_perm"]
    if p_od is None or p_id is None:
        R.add("determinism", "refits", "FAIL", "a half is unmapped; scores not comparable")
        return []
    thread_note = ("threads " + ", ".join(f"{v}={os.environ.get(v, 'unset')}"
                                          for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")))
    print(f"  refitting {len(picks)} models ({thread_note})", flush=True)

    # The table metrics a refit reproduces, as (column, fit_one key, position).
    # A v1 table ranked auc_* on p rebuilt from S, so its AUCs are compared
    # but kept out of the verdict.
    v2_table = all(c in m.columns for c in V2_TABLE)
    metrics = [(f"{t}_{k}", k, j)
               for k in ("id", "ood") + (("train_sub", "train_all") if v2_table else ())
               for j, t in enumerate(("acc", "auc"))]
    judged = [c for c, _, _ in metrics if v2_table or not c.startswith("auc_")]
    out = []
    for r in picks:
        row = m.iloc[r]
        sp = specs[int(row.idx)]
        t0 = time.time()
        got = zoo.fit_one(sp, d["arrays"], a.device)
        dt = time.time() - t0
        if got is None:
            out.append(dict(idx=int(row.idx), kind=sp["kind"], kw=str(sp["kw"]),
                            ok=False, note="fit_one returned None"))
            print(f"    idx {int(row.idx):3d} {sp['kind']:4s} refit returned None", flush=True)
            continue
        d_id = float(np.abs(got["s_id"][p_id] - cell.S_id[r]).max())
        d_od = float(np.abs(got["s_od"][p_od] - cell.S_ood[r]).max())
        diff = {c: abs(got[k][j] - float(row[c])) for c, k, j in metrics}
        d_tab = {c: diff[c] for c in judged}
        v1_auc = {c: v for c, v in diff.items() if c not in d_tab}
        rec = dict(idx=int(row.idx), kind=sp["kind"], kw=str(sp["kw"]),
                   max_diff_S_id=d_id, max_diff_S_ood=d_od,
                   max_diff_table=max(d_tab.values()), seconds=round(dt, 2))
        if v1_auc:
            rec["v1_auc_diff"] = max(v1_auc.values())
        notes = []
        if v2 and "features" in m.columns:
            if ";".join(feats[j] for j in got["cols"]) != str(row.features):
                notes.append("features differ")
            if int(row.n_train) != got["n_train"]:
                notes.append("n_train differs")
        off = [f"{c} {v:.1e}" for c, v in d_tab.items() if v > a.tol]
        if off:
            notes.append("table differs: " + ", ".join(off))
        rec["ok"] = max(d_id, d_od) <= a.tol and not notes
        rec["note"] = "; ".join(notes)
        out.append(rec)
        print(f"    idx {rec['idx']:3d} {rec['kind']:4s} {rec['kw']:58s} "
              f"max|dS_id| {d_id:.1e}  max|dS_ood| {d_od:.1e}  "
              f"max|dtable| {rec['max_diff_table']:.1e}  {dt:5.1f}s"
              + (f"  v1 auc {rec['v1_auc_diff']:.1e}" if v1_auc else "")
              + (f"  {rec['note']}" if notes else ""), flush=True)

    fam: dict[str, float] = {}
    for rec in out:
        v = max(rec.get("max_diff_S_id", np.inf), rec.get("max_diff_S_ood", np.inf))
        fam[rec["kind"]] = max(fam.get(rec["kind"], 0.0), v)
    tab = max((rec.get("max_diff_table", np.inf) for rec in out), default=np.inf)
    R.ok("determinism", f"{len(out)} refits match to {a.tol:g}",
         bool(out) and all(r["ok"] for r in out),
         "max |diff| of the scores per family: "
         + ", ".join(f"{k} {v:.1e}" for k, v in sorted(fam.items()))
         + f"; of the table's {'acc/auc' if v2_table else 'acc'} on "
           f"{', '.join(dict.fromkeys(k for _, k, _ in metrics))}: {tab:.1e} ({thread_note})")
    if not v2_table and out:
        worst = max(out, key=lambda rec: rec.get("v1_auc_diff", -1.0))
        R.add("determinism", "v1 table AUCs", "WARN",
              f"auc_id/auc_ood of a table written before 2026-10-14 are ranked on p "
              f"rebuilt from S, which breaks ties (zoo.score_rows); against the "
              f"refits' raw-p AUC: max |diff| {worst.get('v1_auc_diff', np.nan):.1e} "
              f"(idx {worst['idx']} {worst['kind']})")
    return out


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
    a = ap.parse_args()

    mt = TAG_RE.match(a.tag)
    if not mt:
        raise SystemExit(f"cannot parse tag {a.tag!r} (want e.g. FL_band_AE_s0)")
    state, protocol, train_dom = mt["state"], mt["protocol"], mt["train_dom"]
    a.seed_used = int(mt["seed"]) if a.seed is None else a.seed
    if a.seed is not None and a.seed != int(mt["seed"]):
        print(f"note: --seed {a.seed} overrides the tag's seed {mt['seed']}")
    data = Path(a.data)
    if data.is_dir():
        data = data / f"dataset_{state}.parquet"

    key = audit_key(a, data)
    if a.out and Path(a.out).exists() and not a.overwrite:
        cells = Path(a.cells_dir)
        prev, why = stored_report(Path(a.out), key,
                                  [cells / f"{a.tag}.npz", cells / f"{a.tag}_models.csv",
                                   data, Path(a.features)])
        if prev is not None:
            print(f"{a.out} is current (same audit, later than the cell and the dataset), "
                  f"showing it (--overwrite to re-audit)")
            print_table(prev["checks"])
            print(f"verdict: {prev['verdict']}")
            return 0 if prev["verdict"] == "PASS" else 1
        print(f"{a.out} exists but is stale: {why}. Re-auditing.", flush=True)

    t0 = time.time()
    print(f"[audit] {a.tag} | cells {a.cells_dir} | data {data} | features {a.features} "
          f"| seed {a.seed_used}", flush=True)
    cell = load_cell(a.cells_dir, a.tag)
    npz_keys = list(np.load(Path(a.cells_dir) / f"{a.tag}.npz", allow_pickle=True).files)
    # Positions, not labels, identify a row from here on.
    df = pd.read_parquet(data).reset_index(drop=True)
    feats = [f for f in zoo.read_features(Path(a.features)) if f in df.columns]
    print(f"  dataset {len(df):,} rows | {len(feats)} features | npz keys: {', '.join(npz_keys)}",
          flush=True)
    d = zoo.cell_data(df, feats, protocol, train_dom, a.seed_used)
    # zoo_specs(n) is a prefix of zoo_specs(N > n): each spec's draws and its
    # round-robin slot depend only on its position, so this reproduces the
    # specs of the run that built the cell whatever its --n-models was.
    specs = zoo.zoo_specs(int(cell.models.idx.max()) + 1, a.seed_used)

    R = Report()
    shp = check_shapes(R, cell, npz_keys)
    check_models(R, cell, a, specs, feats, len(d["tr"]))
    maps = check_rows(R, cell, df, d, protocol, train_dom, shp["v2"])
    check_tiles(R, df, d, maps["ood_rows"], maps["id_rows"])
    refits = check_refits(R, cell, a, specs, d, maps, feats, shp["v2"])

    verdict = "PASS" if R.n_fail == 0 else "FAIL"
    print_table(R.rows)
    print(f"verdict: {verdict} ({R.n_fail} FAIL, "
          f"{sum(r['status'] == 'WARN' for r in R.rows)} WARN) in {time.time() - t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(jsonable(dict(
            **key, cell_format="v2" if shp["v2"] else "v1",
            checks=R.rows, refits=refits, verdict=verdict)), indent=1))
        print(f"saved {a.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
