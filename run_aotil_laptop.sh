#!/usr/bin/env bash
# =============================================================================
# run_aotil_laptop.sh -- everything you can check WITHOUT the cluster.
#
# =============================================================================
# Run this before ./submit_aotil_all.sh. It takes a couple of minutes on a
# laptop CPU and catches the failures that are expensive to find in a batch job:
# a wrong hand-written gradient (which does not crash, it optimises the wrong
# thing), a backend that disagrees with the reference, a cell missing the
# spatial columns, a coherence metric with a sign error, a dataset whose columns
# moved. A cluster sweep that dies three hours in on any of these costs a day.
#
# Two stages:
#   1. SELF-TEST      no data, no GPU. Finite-difference checks on every
#                     gradient, numpy/torch agreement, mu behaviour, controls,
#                     coherence metrics, end-to-end on synthetic cells.
#   2. TINY REAL RUN  one cell, a small model population, two mu values, on the
#                     actual Florida parquet. Proves the data path -- column
#                     names, spatial columns in the .npz, the graph, the sweep,
#                     the transfer -- with numbers that are indicative only.
#
# Usage:
#   ./run_aotil_laptop.sh                 both stages
#   ./run_aotil_laptop.sh --test-only     stage 1 only (no data needed)
#   N_MODELS=60 ./run_aotil_laptop.sh     a bigger tiny run
#
# THE NUMBERS FROM STAGE 2 ARE NOT RESULTS. A 24-model population is far below
# what the correlation needs; gate.py's convergence table on the real run is
# what says how many models a cell actually requires.
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="${DATA_DIR:-data/processed}"
CELLS_DIR="${CELLS_DIR:-data/interim/aotil/cells-laptop}"
OUT_DIR="${OUT_DIR:-results/aotil-laptop}"
N_MODELS="${N_MODELS:-24}"
PY="${PY:-python3}"

TEST_ONLY=0
[ "${1:-}" = "--test-only" ] && TEST_ONLY=1

echo "============================================================"
echo " stage 1 -- self-test (no data, no GPU)"
echo "============================================================"
$PY - <<'PYEOF'
import importlib.util, sys
need = ["numpy", "scipy", "sklearn", "pandas"]
opt = {"torch": "GPU backend and --device cuda",
       "pyarrow": "reading the parquet in stage 2",
       "joblib": "parallel model fitting",
       "xgboost": "the XGBoost family in the model population",
       "pyproj": "projecting lon/lat properly (a fallback is used without it)"}
missing = [m for m in need if not importlib.util.find_spec(m)]
if missing:
    print(f"MISSING REQUIRED: {missing}\n  pip install " + " ".join(missing))
    sys.exit(1)
for m, why in opt.items():
    print(f"  {'ok     ' if importlib.util.find_spec(m) else 'ABSENT '} {m:9s} ({why})")
PYEOF

$PY aotil/test_aotil.py

if [ "$TEST_ONLY" = "1" ]; then
    echo
    echo "Self-test only. Re-run without --test-only to exercise the data path."
    exit 0
fi

echo
echo "============================================================"
echo " stage 2 -- tiny real run on ${DATA_DIR}"
echo "============================================================"
if [ ! -f "${DATA_DIR}/dataset_FL.parquet" ]; then
    echo "SKIPPED: ${DATA_DIR}/dataset_FL.parquet not found."
    echo "Stage 1 passed, which is the part that gates the cluster run."
    exit 0
fi

rm -rf "$CELLS_DIR"; mkdir -p "$CELLS_DIR" "$OUT_DIR"
CELL=(--cells "$CELLS_DIR" --out "$OUT_DIR" --state FL --protocol band --train-dom AE --seed 0)

echo "--- zoo (${N_MODELS} models) ---"
$PY aotil/zoo.py --state FL --protocol band --train-dom AE --seed 0 \
    --n-models "$N_MODELS" --data "$DATA_DIR" --out "$CELLS_DIR" --n-jobs -1

echo
echo "--- the cell must carry the spatial columns ---"
$PY - "$CELLS_DIR" <<'PYEOF'
import sys, numpy as np
from pathlib import Path
z = np.load(Path(sys.argv[1]) / "FL_band_AE_s0.npz", allow_pickle=True)
need = ["lon", "lat", "block_id", "tile_50km"]
missing = [c for c in need if c not in z.files]
if missing:
    print(f"  FAIL: cell is missing {missing} -- the graph cannot be built.")
    print("  This is what the pilot cells were missing. Check zoo.SPATIAL_COLS "
          "against the dataset's columns.")
    sys.exit(1)
print(f"  ok -- {z['S_ood'].shape[0]} models x {z['S_ood'].shape[1]:,} OOD points, "
      f"lon/lat present, {len(np.unique(z['tile_50km']))} tiles")
PYEOF

echo
echo "--- gate ---"
$PY aotil/gate.py --cells "$CELLS_DIR" --sizes 8,16,24

echo
echo "--- E0 (the go/no-go for regions) ---"
$PY aotil/pointcov.py "${CELL[@]}" --knn 10 --n-perm 50

echo
echo "--- E1 (two mu values, short) ---"
$PY aotil/analyse.py "${CELL[@]}" --mus 0,10 --fracs 0.05 --iters 150 --restarts 2

echo
echo "--- E2 ---"
$PY aotil/transfer.py "${CELL[@]}" --mus 0,10 --fracs 0.10 --iters 150 --restarts 2

echo
echo "============================================================"
echo " Both stages passed. The data path works end to end."
echo
echo " The stage-2 numbers are NOT results -- ${N_MODELS} models is far below"
echo " what the correlation needs. What they prove is that nothing will fail"
echo " on the cluster for a reason a laptop could have caught."
echo
echo " Next:"
echo "   ./upload_to_cluster.sh && ./upload_data_to_cluster.sh"
echo "   LIMIT=1 ./submit_aotil_all.sh --date \$(date +%F)   # watch one"
echo "   ./submit_aotil_all.sh --date \$(date +%F)           # then the rest"
echo "============================================================"
