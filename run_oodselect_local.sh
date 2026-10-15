#!/bin/bash
# =============================================================================
# run_oodselect_local.sh -- the first experiment (classic + spatial OODSelect,
# paper protocol) on this laptop instead of the cluster.
#
# Same code, same cells, same budgets, same outputs as the two submit_* scripts:
#   results/<date>/oodselect_paper/{classic,spatial_mu3_knn10}/...
#   logs/<date>/oodselect_paper/...
# Only the scheduler changes: Slurm arrays become `xargs -P`, one budget per
# process, one thread each (the selection is dispatch-bound, so N single-thread
# processes beat one N-thread process).
#
# Usage, from the repo root:
#   ./run_oodselect_local.sh --date 2026-09-24
#   JOBS=8 ./run_oodselect_local.sh --date 2026-09-24
#   VARIANTS=classic ./run_oodselect_local.sh --date 2026-09-24     # one version
#   TRIALS=3 MAX_EPOCHS=500 ./run_oodselect_local.sh --date 2000-01-01   # smoke test
#
# Resumable: a budget whose _result.csv exists is skipped, so re-running the
# same command after an interruption only does what is missing.
#
# Needs in the venv: torch, optuna (and xgboost for the full 300-model family):
#   .venv/bin/pip install torch optuna xgboost      # + `brew install libomp` for xgboost
# Keep the Mac plugged in; the script wraps itself in `caffeinate -i` so it
# does not idle-sleep (closing the lid on battery still will).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Stay awake for the duration (macOS only; re-exec once under caffeinate).
if command -v caffeinate >/dev/null 2>&1 && [ -z "${_OODSEL_CAFFEINATED:-}" ]; then
    export _OODSEL_CAFFEINATED=1
    exec caffeinate -i "$0" "$@"
fi

DATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --date) DATE="$2"; shift 2 ;;
        --date=*) DATE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
if ! echo "$DATE" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'; then
    echo "usage: $0 --date YYYY-MM-DD" >&2; exit 1
fi

PY="${PY:-.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/processed}"
CELLS_DIR="${CELLS_DIR:-data/interim/aotil/cells}"
CELLS="${CELLS:-FL:band:AE:0 FL:band:A:0}"
VARIANTS="${VARIANTS:-classic spatial}"
MU_SPATIAL="${MU_SPATIAL:-3}"
KNN="${KNN:-10}"
N_MODELS="${N_MODELS:-300}"
TRIALS="${TRIALS:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-50000}"
PATIENCE="${PATIENCE:-500}"
SEED="${SEED:-0}"
# Default: the performance cores (Apple silicon), leaving the efficiency cores
# for the rest of the machine.
if [ -z "${JOBS:-}" ]; then
    JOBS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)
    [ -n "$JOBS" ] || JOBS=$(( $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4) - 2 ))
    [ "$JOBS" -ge 1 ] || JOBS=1
fi

ROOT="results/${DATE}/oodselect_paper"
LOG_DIR="logs/${DATE}/oodselect_paper"
mkdir -p "$ROOT" "$LOG_DIR" "$CELLS_DIR"

# ---------------------------------------------------------------- preflight
if ! "$PY" -c "import optuna, torch, sklearn, scipy, pandas, pyarrow" 2>/dev/null; then
    echo "ERROR: $PY cannot import optuna/torch/sklearn/scipy/pandas/pyarrow." >&2
    echo "       $PY -m pip install torch optuna" >&2
    exit 1
fi
if ! "$PY" -c "import xgboost" 2>/dev/null; then
    echo "WARNING: xgboost not importable -- the population will lack the XGBoost"
    echo "         family, so it will NOT match the cluster's 300 models."
    echo "         $PY -m pip install xgboost   (and: brew install libomp)"
    if [ "${ALLOW_NO_XGB:-0}" != "1" ]; then
        echo "         ALLOW_NO_XGB=1 to run anyway." >&2; exit 1
    fi
fi

variant_dir() {   # variant_dir <mu>
    if [ "$(printf '%g' "$1")" = "0" ]; then echo classic
    else echo "spatial_mu$(printf '%g' "$1")_knn${KNN}"; fi
}

echo "[local] date=$DATE cells=[$CELLS] variants=[$VARIANTS] jobs=$JOBS"
echo "[local] models=$N_MODELS trials=$TRIALS max_epochs=$MAX_EPOCHS patience=$PATIENCE"
T0=$(date +%s)

# ---------------------------------------------------------------- 1. zoo
TAGS=""
for spec in $CELLS; do
    IFS=':' read -r st pr td sd <<< "$spec"
    tag="${st}_${pr}_${td}_s${sd}"
    TAGS="$TAGS $tag"
    echo "[zoo] $tag"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u aotil/zoo.py --state "$st" --protocol "$pr" --train-dom "$td" --seed "$sd" \
        --n-models "$N_MODELS" --data "$DATA_DIR" --out "$CELLS_DIR" \
        --n-jobs -1 --device cpu >>"${LOG_DIR}/zoo_${tag}_local.out" 2>>"${LOG_DIR}/zoo_${tag}_local.err"
    tail -1 "${LOG_DIR}/zoo_${tag}_local.out"
done

# ---------------------------------------------------------------- 2. select
# One line per (variant, cell, budget), largest budgets first: they take the
# longest, and starting them first keeps the tail of the run short.
TASKS="${LOG_DIR}/local_tasks.txt"
: > "$TASKS"
for v in $VARIANTS; do
    case "$v" in
        classic) mu=0 ;;
        spatial) mu="$MU_SPATIAL" ;;
        *) echo "unknown variant '$v' (classic|spatial)" >&2; exit 1 ;;
    esac
    vd=$(variant_dir "$mu")
    mkdir -p "$ROOT/$vd"
    for tag in $TAGS; do
        for b in $("$PY" aotil/oodselect_paper.py --cells "$CELLS_DIR" --tag "$tag" --list-budgets); do
            echo "$b $mu $vd $tag" >> "$TASKS"
        done
    done
done
sort -k1,1nr -o "$TASKS" "$TASKS"
N_TASKS=$(wc -l < "$TASKS" | tr -d ' ')
echo "[select] $N_TASKS tasks, $JOBS at a time (progress: ls $ROOT/*/*_result.csv | wc -l)"

export PY CELLS_DIR ROOT LOG_DIR KNN TRIALS MAX_EPOCHS PATIENCE SEED
# shellcheck disable=SC2016
xargs -P "$JOBS" -L 1 bash -c '
    b=$0; mu=$1; vd=$2; tag=$3
    log="${LOG_DIR}/${vd}_${tag}_S${b}_local"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u aotil/oodselect_paper.py --cells "$CELLS_DIR" --tag "$tag" \
        --out "$ROOT/$vd" --budget "$b" --mu "$mu" --knn "$KNN" \
        --trials "$TRIALS" --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
        --seed "$SEED" --threads 1 >>"$log.out" 2>>"$log.err" \
      && echo "  done  $vd $tag S=$b  $(grep -o "TEST r = [^|]*" "$log.out" | tail -1)" \
      || echo "  FAIL  $vd $tag S=$b  (see $log.err)"
' < "$TASKS"

# ---------------------------------------------------------------- 3. report
echo "[report]"
"$PY" aotil/report_paper.py --root "$ROOT" --cells "$CELLS_DIR" --tags $TAGS \
    2>>"${LOG_DIR}/report_local.err" | tee "${LOG_DIR}/report_local.out"
echo "[local] finished in $(( ($(date +%s) - T0) / 60 )) min. Results: $ROOT"
