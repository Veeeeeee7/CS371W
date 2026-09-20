#!/bin/bash
#SBATCH --job-name=run_aotil
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --partition=rp6b-1-gm96-c8-m64
#SBATCH --output=logs/sbatch/%x_%j.out
#SBATCH --error=logs/sbatch/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --time=08:00:00

# =============================================================================
# aotil -- spatially regularized OODSelect, one job per CELL
#
# =============================================================================
# A CELL is one (state, protocol, train-domain, tile-split seed): a population of
# heterogeneous models trained on one FEMA study-type domain, scored per example
# on the in-domain and out-of-domain halves of the SAME held-out 50 km tiles.
# Everything downstream -- the covariance map, the mu sweep, the held-out-point
# transfer -- runs inside one cell, so a cell is the natural unit of work and of
# failure. Every output and log is per-cell, so cells run as concurrent jobs.
#
# PHASES, in order, each skippable:
#   RUN_ZOO=1   fit the model population and write the cell .npz  (the wall clock)
#   RUN_GATE=1  ID spread, aggregate ID->OOD correlation, the model-count rule
#   RUN_E0=1    per-point covariance + Moran's I -- the GO/NO-GO for regions
#   RUN_E1=1    the mu sweep with every control at every mu   (the core result)
#   RUN_E2=1    held-out-point transfer inside the region
#
# E0 IS A GATE, NOT A FORMALITY. If the per-point covariance map is inside its
# permutation null it is spatial white noise, no mu > 0 can assemble a region
# from it, and the sweep will spend a day confirming that. STOP_ON_NOGO=1 makes
# the job stop there instead. Read the verdict line in the log before queueing
# the rest of the sweep.
#
# WHERE THE COMPUTE ACTUALLY GOES. The model population is scikit-learn on CPU
# and is ~95% of the wall clock; the selection is seconds. The GPU earns its
# place only because restarts and mu values batch into one mat-vec
# (DEVICE=cuda), and because the XGBoost family can use it. If the queue for a
# GPU node is long, GPUS=0 DEVICE=cpu costs little.
#
# Usage:
#   sbatch run_aotil.sh --date YYYY-MM-DD                  # every cell in this job
#   ONLY_CELL="FL:band:AE:0" sbatch run_aotil.sh --date YYYY-MM-DD
#   RUN_ZOO=0 RUN_E1=1 sbatch run_aotil.sh --date YYYY-MM-DD   # re-sweep existing cells
#
# ./submit_aotil_all.sh --date YYYY-MM-DD is the intended entry point.
# =============================================================================

conda init bash > /dev/null 2>&1
source ~/.bashrc

AOTIL_ENV="${AOTIL_ENV:-aotil}"
PY=(conda run -n "$AOTIL_ENV" python3 -u)

# ---------------------------------------------------------------- where things live
# CODE and RESULTS live in the repo checkout; DATA and CELLS live on scratch.
#
# The split is by size and by replaceability, not by convenience. The code push
# is a few hundred KB. results/ and logs/ are CSVs and a parquet per cell -- a
# few MB, and they are what comes back down. data/processed is ~220 MB and the
# cells are ~20-40 MB each, which is what fills a home quota.
#
# SCRATCH IS PURGED. Treat data/ and cells/ there as reconstructible: the
# parquets re-upload from the laptop in minutes and a cell refits from them.
# NOTHING that is a result may live on scratch -- results/ and logs/ stay in the
# repo tree on purpose, so a purge costs compute and never costs a number.
#
# If the scratch root is not writable (running off-cluster, or a quota problem),
# both fall back to repo-local paths with a loud note rather than dying.
# Resolving the scratch root. An explicitly set AOTIL_ARTIFACT_ROOT is trusted
# and created. The DEFAULT is used only when /scratch already exists, because
# "mkdir -p succeeded" is not evidence of being on the cluster -- on a machine
# with a writable /, it succeeds everywhere and would quietly put the laptop's
# cells somewhere nobody looks again.
if [ -n "${AOTIL_ARTIFACT_ROOT:-}" ]; then
    ART="$AOTIL_ARTIFACT_ROOT"
elif [ -d /scratch ]; then
    ART="/scratch/${USER:-$(id -un)}/cs371w"
else
    ART=""
fi
if [ -n "$ART" ] && ! mkdir -p "$ART/aotil/cells" 2>/dev/null; then
    echo "NOTE: scratch root '$ART' is not writable; using repo-local paths." >&2
    ART=""
fi

if [ -n "$ART" ]; then
    CELLS_DIR="${CELLS_DIR:-$ART/aotil/cells}"
    # The data has to already BE there; the guard below reports it if not.
    # A repo-local checkout that was never pushed to scratch still runs.
    if [ -z "${DATA_DIR:-}" ]; then
        if [ -d "$ART/data/processed" ]; then DATA_DIR="$ART/data/processed"
        else DATA_DIR="data/processed"; fi
    fi
else
    CELLS_DIR="${CELLS_DIR:-data/interim/aotil/cells}"
    DATA_DIR="${DATA_DIR:-data/processed}"
fi

# The model fits are parallelised one model per worker, so the learners are
# pinned to one thread each; letting BLAS also grab every core oversubscribes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RESULTS_DATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --date)   RESULTS_DATE="$2"; shift 2 ;;
        --date=*) RESULTS_DATE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1 (only --date YYYY-MM-DD is accepted)" >&2; exit 1 ;;
    esac
done
if [[ ! "$RESULTS_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: --date YYYY-MM-DD is required, e.g. sbatch $0 --date $(date +%F)" >&2
    exit 1
fi

# CELLS are "state:protocol:traindom:seed". The two band cells first: band is the
# primary negative protocol, and shadedX is the sensitivity arm only, because the
# Florida pilot found the inversion inseparable from difficulty under shadedX.
CELLS=(
  "FL:band:AE:0" "FL:band:A:0"
  "FL:band:AE:1" "FL:band:A:1"
  "FL:band:AE:2" "FL:band:A:2"
  "FL:shadedX:AE:0" "FL:shadedX:A:0"
)
if [ -n "${ONLY_CELL:-}" ]; then
    read -ra CELLS <<< "$ONLY_CELL"
fi

N_MODELS="${N_MODELS:-300}"
MUS="${MUS:-0,0.1,0.3,1,3,10}"
FRACS="${FRACS:-0.05,0.10,0.25}"
KNN="${KNN:-10}"
GRAPH="${GRAPH:-knn}"
ITERS="${ITERS:-400}"
RESTARTS="${RESTARTS:-4}"
N_PERM="${N_PERM:-200}"
DEVICE="${DEVICE:-cuda}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

RUN_ZOO="${RUN_ZOO:-1}"; RUN_GATE="${RUN_GATE:-1}"; RUN_E0="${RUN_E0:-1}"
RUN_E1="${RUN_E1:-1}";   RUN_E2="${RUN_E2:-1}"
STOP_ON_NOGO="${STOP_ON_NOGO:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

# Fail fast on missing data: data/ is gitignored and is NOT part of the code
# push, so a cluster checkout without ./upload_data_to_cluster.sh looks exactly
# like a code bug three hours into the queue.
missing=()
for spec in "${CELLS[@]}"; do
    st="${spec%%:*}"
    f="${DATA_DIR}/dataset_${st}.parquet"
    [ -f "$f" ] || missing+=("$f")
done
[ -f "${DATA_DIR}/FEATURES.txt" ] || missing+=("${DATA_DIR}/FEATURES.txt")
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: data not found under DATA_DIR='${DATA_DIR}' (cwd: $(pwd)). Missing:" >&2
    printf '         %s\n' "${missing[@]}" >&2
    echo "       DATA_DIR resolved from AOTIL_ARTIFACT_ROOT='$ART'." >&2
    echo "       Run ./upload_data_to_cluster.sh from the laptop, or set" >&2
    echo "       DATA_DIR/AOTIL_ARTIFACT_ROOT if the data is somewhere else." >&2
    exit 1
fi

OUTDIR="results/${RESULTS_DATE}/aotil"
LOG_DIR="logs/${RESULTS_DATE}/aotil"
mkdir -p "$OUTDIR" "$LOG_DIR" "$CELLS_DIR"
# ONLY_CELL may name several cells, which would otherwise put a space and a
# colon in the log path.
LOG_TAG=$(printf '%s' "${ONLY_CELL:-all}" | tr -s '[:space:]:' '--' | tr -cd '[:alnum:]-')
exec >>"${LOG_DIR}/run_aotil_${LOG_TAG}_${SLURM_JOB_ID:-manual}.out" \
    2>>"${LOG_DIR}/run_aotil_${LOG_TAG}_${SLURM_JOB_ID:-manual}.err"

DRY_RUN=${DRY_RUN:-0}
run() { if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi; }

FAILED=()
phase() {   # phase <name> <command...>
    local name="$1"; shift
    echo "---- $name ----"
    if run "$@"; then
        return 0
    fi
    echo "!! FAILED: $name"
    FAILED+=("$name")
    [ "$STOP_ON_FAIL" = "1" ] && exit 1
    return 1
}

echo "[aotil] date=$RESULTS_DATE cells=${CELLS[*]}"
echo "[aotil] scratch=$ART"
echo "[aotil] data=$DATA_DIR  cells_dir=$CELLS_DIR  results=results/${RESULTS_DATE}/aotil"
echo "[aotil] models=$N_MODELS device=$DEVICE n_jobs=$N_JOBS graph=$GRAPH knn=$KNN"
echo "[aotil] mus=$MUS fracs=$FRACS iters=$ITERS restarts=$RESTARTS"
echo "[aotil] phases: zoo=$RUN_ZOO gate=$RUN_GATE E0=$RUN_E0 E1=$RUN_E1 E2=$RUN_E2"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[aotil] no GPU visible"

for spec in "${CELLS[@]}"; do
    IFS=':' read -r ST PR TD SD <<< "$spec"
    TAG="${ST}_${PR}_${TD}_s${SD}"
    echo
    echo "================ cell $TAG ================"
    COMMON=(--cells "$CELLS_DIR" --out "$OUTDIR" --state "$ST" --protocol "$PR"
            --train-dom "$TD" --seed "$SD")

    if [ "$RUN_ZOO" = "1" ]; then
        phase "zoo $TAG" "${PY[@]}" aotil/zoo.py \
            --state "$ST" --protocol "$PR" --train-dom "$TD" --seed "$SD" \
            --n-models "$N_MODELS" --data "$DATA_DIR" --out "$CELLS_DIR" \
            --n-jobs "$N_JOBS" --device "$DEVICE" || continue
    fi

    if [ "$RUN_E0" = "1" ]; then
        phase "E0 $TAG" "${PY[@]}" aotil/pointcov.py "${COMMON[@]}" \
            --knn "$KNN" --graph "$GRAPH" --n-perm "$N_PERM"
        # The verdict line is the go/no-go. Reading it here keeps a NO-GO from
        # silently turning into a day of sweep that confirms it.
        SUMM="${OUTDIR}/${TAG}_${GRAPH}_pointcov_summary.csv"
        if [ "$STOP_ON_NOGO" = "1" ] && [ -f "$SUMM" ]; then
            if ! "${PY[@]}" -c "
import pandas as pd, sys
s = pd.read_csv('$SUMM').iloc[0]
sys.exit(0 if (s.p_perm < 0.05 and s.moran_I > 0) else 1)"; then
                echo "E0 says NO-GO for regions on $TAG; STOP_ON_NOGO=1, skipping the sweep."
                continue
            fi
        fi
    fi

    if [ "$RUN_E1" = "1" ]; then
        phase "E1 $TAG" "${PY[@]}" aotil/analyse.py "${COMMON[@]}" \
            --mus "$MUS" --fracs "$FRACS" --knn "$KNN" --graph "$GRAPH" \
            --iters "$ITERS" --restarts "$RESTARTS" --device "$DEVICE"
    fi

    if [ "$RUN_E2" = "1" ]; then
        phase "E2 $TAG" "${PY[@]}" aotil/transfer.py "${COMMON[@]}" \
            --mus "$MUS" --fracs "$FRACS" --knn "$KNN" --graph "$GRAPH" \
            --iters "$ITERS" --restarts "$RESTARTS" --device "$DEVICE"
    fi
done

if [ "$RUN_GATE" = "1" ]; then
    echo
    phase "gate (all cells present)" "${PY[@]}" aotil/gate.py --cells "$CELLS_DIR"
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED phases (${#FAILED[@]}): ${FAILED[*]}"
    exit 1
fi
echo "Done."
echo "  results (repo, comes down with --date):  $OUTDIR"
echo "  cells   (scratch, purgeable, refittable): $CELLS_DIR"
