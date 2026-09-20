#!/bin/bash
# =============================================================================
# submit_aotil_all.sh -- fan out the AoTIL suite, ONE JOB PER CELL
#
# =============================================================================
# A cell is (state, protocol, train-domain, tile-split seed). Every output, cell
# .npz and log is per-cell, so the jobs are independent and safe to run
# concurrently -- nothing appends to a shared file.
#
# ORDER MATTERS. The band cells go first: band is the primary negative protocol,
# and the Florida pilot found the inversion inseparable from difficulty under
# shadedX, which is why shadedX is the sensitivity arm rather than half the
# result. If the queue only gets through part of the list, you want it to have
# been the part the paper is built on.
#
# Run from the repo root on the login node:
#   ./submit_aotil_all.sh --date YYYY-MM-DD
#   SEEDS="0" ./submit_aotil_all.sh --date YYYY-MM-DD       # one seed first
#   PROTOCOLS="band" ./submit_aotil_all.sh --date YYYY-MM-DD
#   LIMIT=1 ./submit_aotil_all.sh --date YYYY-MM-DD         # watch one before the sweep
#
# WATCH ONE JOB BEFORE SUBMITTING THE SWEEP. The failure that costs a day is a
# silent one -- a cgroup SIGKILL under --mem shows up as a MISSING cell, not a
# failed job -- so LIMIT=1, check the log and `sacct -j <id> --format=MaxRSS`,
# then submit the rest.
#
# Env (DATA_DIR, N_MODELS, MUS, FRACS, KNN, GRAPH, DEVICE, RUN_*, DRY_RUN,
# AOTIL_ENV) is forwarded to every job via --export=ALL.
# =============================================================================
set -euo pipefail

if [ "${1:-}" != "--date" ] || [ -z "${2:-}" ]; then
    echo "usage: $0 --date YYYY-MM-DD   (the results tree to write into)" >&2
    exit 1
fi
DATE="$2"
if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: --date must be YYYY-MM-DD, got '$DATE'" >&2; exit 1
fi

# Results append and are never deduped, so a date whose tree already holds rows
# would silently accumulate a second copy of every cell.
if [ -d "results/${DATE}/aotil" ] && \
   [ -n "$(find "results/${DATE}/aotil" -name '*.csv' -print -quit 2>/dev/null)" ]; then
    if [ "${ALLOW_EXISTING:-0}" != "1" ]; then
        echo "ERROR: results/${DATE}/aotil already holds CSVs." >&2
        echo "       Use a fresh --date, or ALLOW_EXISTING=1 to append anyway." >&2
        exit 1
    fi
    echo "warning: appending into an existing tree (ALLOW_EXISTING=1)"
fi

source ./sbatch_overrides.sh
sb_overrides_banner "aotil"

STATES="${STATES:-FL}"
PROTOCOLS="${PROTOCOLS:-band shadedX}"
DOMAINS="${DOMAINS:-AE A}"
SEEDS="${SEEDS:-0 1 2}"

# band before shadedX, whatever order PROTOCOLS names them in.
CELLS=()
for pr in band shadedX; do
    case " $PROTOCOLS " in *" $pr "*) ;; *) continue ;; esac
    for st in $STATES; do
        for td in $DOMAINS; do
            for sd in $SEEDS; do
                # shadedX is the sensitivity arm: seed 0 only, unless asked.
                if [ "$pr" = "shadedX" ] && [ "$sd" != "0" ] \
                   && [ "${SHADEDX_ALL_SEEDS:-0}" != "1" ]; then
                    continue
                fi
                CELLS+=("${st}:${pr}:${td}:${sd}")
            done
        done
    done
done

if [ -n "${LIMIT:-}" ]; then
    CELLS=("${CELLS[@]:0:$LIMIT}")
fi

echo "[aotil] submitting ${#CELLS[@]} cells for results date $DATE"
for spec in "${CELLS[@]}"; do
    name=$(printf '%s' "$spec" | tr ':' '_')
    echo "  -> $spec"
    sbatch "${SB_OVERRIDES[@]}" \
        --job-name="aotil_${name}" \
        --export=ALL,ONLY_CELL="${spec}" \
        run_aotil.sh --date "$DATE"
done

echo
echo "Submitted ${#CELLS[@]} jobs."
echo "Watch the first log for the E0 verdict line before trusting the sweep:"
echo "  tail -f logs/${DATE}/aotil/run_aotil_*-\$(echo \"${CELLS[0]}\" | tr ':' '-')*.out"
echo "Then pull it down:  ./download_from_cluster.sh ${DATE}"
