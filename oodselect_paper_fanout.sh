# =============================================================================
# oodselect_paper_fanout.sh -- shared fan-out for the two first-experiment
# submissions. SOURCED by submit_oodselect_classic.sh and
# submit_oodselect_spatial.sh, which differ only in MU. Not executable alone.
#
# Per cell it queues three things, chained by dependency:
#
#   zoo      1 job     fit 300 models (skipped when the cell already exists, and
#                      never queued twice: a pending/running zoo job for the same
#                      cell -- say, from the other submission -- is reused)
#   select   1 array   one task per budget in the paper's grid (10 ... 7500),
#                      each a full 100-trial Optuna search. afterok:zoo
#   report   1 job     for all cells, afterany:every select array, so a single
#                      failed budget still gets a report that names it missing
#
# Knobs (env; all optional):
#   CELLS        "FL:band:AE:0 FL:band:A:0"   state:protocol:traindom:seed
#   N_MODELS 300  TRIALS 100  MAX_EPOCHS 50000  PATIENCE 500  SEED 0  KNN 10
#   ACCOUNT general   PARTITION (worker header: l4-4-gm96-c96-m384)   QOS
#   SEL_CPUS 4  SEL_MEM 8G  SEL_TIME 06:00:00  MAX_PARALLEL 13
#   ZOO_CPUS 16 ZOO_MEM 32G ZOO_TIME 01:00:00
#   GPUS         unset = no GPU (see run_oodselect_paper.sh for why)
#   DRY_RUN=1    print the sbatch commands, submit nothing
#   ALLOW_EXISTING=1   submit into a date whose variant folder already holds
#                results. Finished budgets are skipped, not redone, so this is
#                also how to re-run only the tasks that failed.
# =============================================================================

oodsel_submit() {   # oodsel_submit <mu> "$@"
    local MU_IN="$1"; shift
    if [ "${1:-}" != "--date" ] || [ -z "${2:-}" ]; then
        echo "usage: $0 --date YYYY-MM-DD" >&2; return 1
    fi
    local DATE="$2"
    if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "ERROR: --date must be YYYY-MM-DD, got '$DATE'" >&2; return 1
    fi

    local KNN="${KNN:-10}"
    local VARIANT
    if [ "$(printf '%g' "$MU_IN")" = "0" ]; then VARIANT="classic"
    else VARIANT="spatial_mu$(printf '%g' "$MU_IN")_knn${KNN}"; fi
    local ROOT="results/${DATE}/oodselect_paper"

    if [ -d "$ROOT/$VARIANT" ] && \
       [ -n "$(find "$ROOT/$VARIANT" -name '*_result.csv' -print -quit 2>/dev/null)" ] && \
       [ "${ALLOW_EXISTING:-0}" != "1" ]; then
        echo "ERROR: $ROOT/$VARIANT already holds results." >&2
        echo "       Use a fresh --date, or ALLOW_EXISTING=1 to fill in only the" >&2
        echo "       budgets that are missing (finished ones are skipped)." >&2
        return 1
    fi

    # Same scratch resolution as the worker, so "does the cell exist" is asked
    # of the place the jobs will look.
    local ART CELLS_DIR
    if [ -n "${AOTIL_ARTIFACT_ROOT:-}" ]; then ART="$AOTIL_ARTIFACT_ROOT"
    elif [ -d /scratch ]; then ART="/scratch/${USER:-$(id -un)}/cs371w"
    else ART=""; fi
    if [ -n "$ART" ]; then CELLS_DIR="${CELLS_DIR:-$ART/aotil/cells}"
    else CELLS_DIR="${CELLS_DIR:-data/interim/aotil/cells}"; fi

    local AOTIL_ENV="${AOTIL_ENV:-aotil}"
    local PYCHK=(conda run -n "$AOTIL_ENV" python3)

    # PREFLIGHT. optuna is new to this env; finding that out 22 tasks deep is
    # the expensive way.
    if [ "${DRY_RUN:-0}" != "1" ]; then
        if ! "${PYCHK[@]}" -c "import optuna, torch, sklearn, scipy, pandas" >/dev/null 2>&1; then
            echo "ERROR: the '$AOTIL_ENV' env cannot import optuna/torch/sklearn/scipy/pandas." >&2
            echo "       conda run -n $AOTIL_ENV pip install 'optuna>=3.0'" >&2
            return 1
        fi
    fi

    mkdir -p logs/sbatch "logs/${DATE}/oodselect_paper"
    local SB=(sbatch --parsable --account "${ACCOUNT:-general}")
    [ -n "${QOS:-}" ]       && SB+=(--qos "$QOS")
    [ -n "${PARTITION:-}" ] && SB+=(--partition "$PARTITION")
    [ -n "${GPUS:-}" ]      && SB+=(--gpus "$GPUS")

    local CELLS_LIST="${CELLS:-FL:band:AE:0 FL:band:A:0}"
    local EXPORTS="ALL,MU=${MU_IN},KNN=${KNN},N_MODELS=${N_MODELS:-300},TRIALS=${TRIALS:-100},MAX_EPOCHS=${MAX_EPOCHS:-50000},PATIENCE=${PATIENCE:-500},SEED=${SEED:-0}"

    sub() {   # sub <sbatch args...> -- echo the job id (or a fake one on DRY_RUN)
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[dry-run] ${SB[*]} $*" >&2; echo "DRY$RANDOM"
        else
            "${SB[@]}" "$@"
        fi
    }

    echo "[oodselect-paper] variant=$VARIANT date=$DATE cells=$CELLS_LIST"
    echo "[oodselect-paper] trials=${TRIALS:-100} max_epochs=${MAX_EPOCHS:-50000}" \
         "patience=${PATIENCE:-500} models=${N_MODELS:-300} cells_dir=$CELLS_DIR"

    local SEL_IDS=() spec tag zoo_dep zoo_id arr n_idx sel_id st pr td sd
    for spec in $CELLS_LIST; do
        IFS=':' read -r st pr td sd <<< "$spec"
        tag="${st}_${pr}_${td}_s${sd}"
        zoo_dep=""

        if [ -f "$CELLS_DIR/${tag}.npz" ] && [ -f "$CELLS_DIR/${tag}_models.csv" ]; then
            echo "  $tag: cell exists, no zoo job"
        else
            zoo_id=$(squeue -h -u "${USER:-$(id -un)}" -n "oodsel_zoo_${tag}" -o %i 2>/dev/null | head -1)
            if [ -n "$zoo_id" ]; then
                echo "  $tag: zoo job $zoo_id already queued (other submission); waiting on it"
            else
                zoo_id=$(sub --job-name "oodsel_zoo_${tag}" \
                    --cpus-per-task "${ZOO_CPUS:-16}" --mem "${ZOO_MEM:-32G}" \
                    --time "${ZOO_TIME:-01:00:00}" \
                    --export "${EXPORTS},STAGE=zoo,CELL_SPEC=${spec}" \
                    run_oodselect_paper.sh --date "$DATE") || return 1
                echo "  $tag: zoo job $zoo_id"
            fi
            zoo_dep="--dependency=afterok:${zoo_id}"
        fi

        # Array range: the exact grid when the cell is already on disk (the
        # grid stops below the OOD set size); otherwise indices 0-12, which
        # covers d up to 15,000 -- tasks past d exit in seconds.
        arr="${BUDGET_INDICES:-}"
        if [ -z "$arr" ]; then
            arr="0-12"
            if [ -f "$CELLS_DIR/${tag}.npz" ] && [ "${DRY_RUN:-0}" != "1" ]; then
                n_idx=$("${PYCHK[@]}" aotil/oodselect_paper.py --cells "$CELLS_DIR" \
                        --tag "$tag" --list-budgets 2>/dev/null | tail -1 | wc -w)
                [ "${n_idx:-0}" -gt 0 ] && arr="0-$((n_idx - 1))"
            fi
        fi
        sel_id=$(sub --job-name "oodsel_${VARIANT}_${tag}" ${zoo_dep} \
            --array "${arr}%${MAX_PARALLEL:-13}" \
            --cpus-per-task "${SEL_CPUS:-4}" --mem "${SEL_MEM:-8G}" \
            --time "${SEL_TIME:-06:00:00}" \
            --export "${EXPORTS},STAGE=select,CELL_SPEC=${spec}" \
            run_oodselect_paper.sh --date "$DATE") || return 1
        echo "  $tag: select array $sel_id (budget indices $arr)"
        SEL_IDS+=("$sel_id")
    done

    local dep; dep=$(IFS=:; echo "${SEL_IDS[*]}")
    local rep_id
    rep_id=$(sub --job-name "oodsel_report_${VARIANT}" --dependency "afterany:${dep}" \
        --cpus-per-task 2 --mem 8G --time 00:30:00 \
        --export "${EXPORTS},STAGE=report,CELL_SPECS=${CELLS_LIST// /;}" \
        run_oodselect_paper.sh --date "$DATE") || return 1
    echo "  report job $rep_id (after all arrays)"
    echo
    echo "Watch:   squeue -u \$USER -o '%.12i %.45j %.9T %.10M %R' | grep oodsel"
    echo "Logs:    logs/${DATE}/oodselect_paper/"
    echo "Results: ${ROOT}/${VARIANT}/   (summary: <cell>_summary.csv; both variants: <cell>_comparison.csv)"
    echo "Pull:    ./download_from_cluster.sh ${DATE}"
}
