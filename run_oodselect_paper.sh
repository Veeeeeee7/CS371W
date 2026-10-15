#!/bin/bash
#SBATCH --job-name=oodsel_paper
#SBATCH --account=general
#SBATCH --partition=l4-4-gm96-c96-m384
#SBATCH --nodes=1
#SBATCH --output=logs/sbatch/%x_%A_%a.out
#SBATCH --error=logs/sbatch/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00

# =============================================================================
# run_oodselect_paper.sh -- the worker behind submit_oodselect_classic.sh and
# submit_oodselect_spatial.sh. Do not sbatch this by hand; the submit scripts
# set its environment and chain its three stages with dependencies.
#
#   STAGE=zoo     fit the 300-model population for one cell (zoo.py skips a
#                 cell that already exists)
#   STAGE=select  ONE budget of OODSelect for one cell -- the Slurm array index
#                 is the index into the paper's budget grid
#   STAGE=report  summarise every budget, both variants if present
#
# Everything the two variants share lives here, so they cannot drift apart: the
# only thing a submit script chooses is MU (0 = the paper; > 0 = spatial).
#
# NO GPU IS REQUESTED, on purpose. The selection is a (180 x ~9,400) mat-vec per
# epoch -- dispatch-bound, ~2.5 ms on 4 CPU threads -- and torch in the aotil
# env cannot initialise CUDA on these nodes anyway ("driver too old", job
# 562664). Ask for one with GPUS=1 only if that is fixed and you have a reason.
# =============================================================================

conda init bash > /dev/null 2>&1
source ~/.bashrc
AOTIL_ENV="${AOTIL_ENV:-aotil}"
PY=(conda run --no-capture-output -n "$AOTIL_ENV" python3 -u)

RESULTS_DATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --date)   RESULTS_DATE="$2"; shift 2 ;;
        --date=*) RESULTS_DATE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
if [[ ! "$RESULTS_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: --date YYYY-MM-DD is required" >&2; exit 1
fi

# ---------------------------------------------------------------- paths
# Same resolution as run_aotil.sh: data and cells on scratch when /scratch
# exists, results and logs in the repo checkout (scratch is purged; nothing that
# is a result lives there).
if [ -n "${AOTIL_ARTIFACT_ROOT:-}" ]; then ART="$AOTIL_ARTIFACT_ROOT"
elif [ -d /scratch ]; then ART="/scratch/${USER:-$(id -un)}/cs371w"
else ART=""; fi
if [ -n "$ART" ] && ! mkdir -p "$ART/aotil/cells" 2>/dev/null; then ART=""; fi
if [ -n "$ART" ]; then
    CELLS_DIR="${CELLS_DIR:-$ART/aotil/cells}"
    if [ -z "${DATA_DIR:-}" ]; then
        if [ -d "$ART/data/processed" ]; then DATA_DIR="$ART/data/processed"
        else DATA_DIR="data/processed"; fi
    fi
else
    CELLS_DIR="${CELLS_DIR:-data/interim/aotil/cells}"
    DATA_DIR="${DATA_DIR:-data/processed}"
fi

STAGE="${STAGE:?STAGE must be zoo, select or report}"
MU="${MU:-0}"
KNN="${KNN:-10}"
N_MODELS="${N_MODELS:-300}"
TRIALS="${TRIALS:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-50000}"
PATIENCE="${PATIENCE:-500}"
SEED="${SEED:-0}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"

if [ "$(printf '%g' "$MU")" = "0" ]; then VARIANT="classic"
else VARIANT="spatial_mu$(printf '%g' "$MU")_knn${KNN}"; fi

ROOT="results/${RESULTS_DATE}/oodselect_paper"
LOG_DIR="logs/${RESULTS_DATE}/oodselect_paper"
mkdir -p "$ROOT/$VARIANT" "$LOG_DIR" "$CELLS_DIR"

# The selection is dispatch-bound; let torch have the allocation, not BLAS
# threads fighting it.
export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS"

spec_to_tag() { local st pr td sd; IFS=':' read -r st pr td sd <<< "$1"; echo "${st}_${pr}_${td}_s${sd}"; }

case "$STAGE" in
# --------------------------------------------------------------------- zoo
zoo)
    TAG=$(spec_to_tag "$CELL_SPEC")
    IFS=':' read -r ST PR TD SD <<< "$CELL_SPEC"
    exec >>"${LOG_DIR}/zoo_${TAG}_${SLURM_JOB_ID:-manual}.out" \
        2>>"${LOG_DIR}/zoo_${TAG}_${SLURM_JOB_ID:-manual}.err"
    echo "[zoo] $TAG  models=$N_MODELS  cpus=$CPUS  cells=$CELLS_DIR  data=$DATA_DIR"
    for f in "$DATA_DIR/dataset_${ST}.parquet" "$DATA_DIR/FEATURES.txt"; do
        [ -f "$f" ] || { echo "ERROR: missing $f -- run ./upload_data_to_cluster.sh" >&2; exit 1; }
    done
    # The submit scripts never queue two zoo jobs for one cell (a second
    # submission waits on the first one's job id instead), and zoo.py skips a
    # cell that already exists, so no file lock is needed here.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    "${PY[@]}" aotil/zoo.py \
        --state "$ST" --protocol "$PR" --train-dom "$TD" --seed "$SD" \
        --n-models "$N_MODELS" --data "$DATA_DIR" --out "$CELLS_DIR" \
        --n-jobs "$CPUS" --device cpu || exit 1
    "${PY[@]}" -c "
import numpy as np; z = np.load('${CELLS_DIR}/${TAG}.npz')
print('[zoo] cell ${TAG}:', z['S_ood'].shape[0], 'models x', z['S_ood'].shape[1], 'OOD points;',
      'y_id stored' if 'y_id' in z.files else 'y_id NOT stored (older cell; ID ties at p=0.5 counted incorrect)')"
    ;;
# ------------------------------------------------------------------ select
select)
    TAG=$(spec_to_tag "$CELL_SPEC")
    IDX="${SLURM_ARRAY_TASK_ID:?select must run as an array task}"
    exec >>"${LOG_DIR}/${VARIANT}_${TAG}_b${IDX}_${SLURM_ARRAY_JOB_ID:-manual}.out" \
        2>>"${LOG_DIR}/${VARIANT}_${TAG}_b${IDX}_${SLURM_ARRAY_JOB_ID:-manual}.err"
    echo "[select] $VARIANT $TAG budget-index $IDX  mu=$MU knn=$KNN trials=$TRIALS" \
         "max_epochs=$MAX_EPOCHS patience=$PATIENCE cpus=$CPUS host=$(hostname)"
    "${PY[@]}" aotil/oodselect_paper.py --cells "$CELLS_DIR" --tag "$TAG" \
        --out "$ROOT/$VARIANT" --budget-index "$IDX" --mu "$MU" --knn "$KNN" \
        --trials "$TRIALS" --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
        --seed "$SEED" --threads "$CPUS"
    ;;
# ------------------------------------------------------------------ report
report)
    TAGS=()
    # ";"-separated: a space cannot travel through sbatch --export.
    for spec in ${CELL_SPECS//;/ }; do TAGS+=("$(spec_to_tag "$spec")"); done
    exec >>"${LOG_DIR}/report_${VARIANT}_${SLURM_JOB_ID:-manual}.out" \
        2>>"${LOG_DIR}/report_${VARIANT}_${SLURM_JOB_ID:-manual}.err"
    echo "[report] root=$ROOT cells=${TAGS[*]} (triggered by $VARIANT)"
    "${PY[@]}" aotil/report_paper.py --root "$ROOT" --cells "$CELLS_DIR" --tags "${TAGS[@]}"
    ;;
*)
    echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
