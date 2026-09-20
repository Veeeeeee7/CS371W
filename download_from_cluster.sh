#!/usr/bin/env bash
# =============================================================================
# Pull RESULTS and LOGS for ONE date back down from the cluster, and optionally
# the cell .npz files the jobs wrote.
#
# results/ and logs/ are date-partitioned by the driver's required --date and
# live in the HOME checkout, so you pass the date and only that day's tree comes
# down. Cells live on SCRATCH, are not date-partitioned (they are inputs to
# every later analysis and are reused across dates), and are ~20-40 MB each, so
# they come down only when asked for, with --cells.
#
# You need the cells locally to re-run a sweep or the characterisation without
# refitting the model population. Pull them once the run finishes: scratch is
# purged, and refitting a cell is cheap but not free.
#
# Usage:
#   ./download_from_cluster.sh YYYY-MM-DD              results + logs
#   ./download_from_cluster.sh YYYY-MM-DD --cells      also the cell .npz (~20 MB each)
#   ./download_from_cluster.sh YYYY-MM-DD --dry-run    preview, copies nothing
#   ./download_from_cluster.sh YYYY-MM-DD --delete     mirror deletions -- a remote
#                                     # tree with fewer files WILL delete local-only
#                                     # files under that date
# =============================================================================
set -euo pipefail

REMOTE_USER="${CS371_REMOTE_USER:-vmli3}"
REMOTE_HOST="${CS371_REMOTE_HOST:-cirrostratus.it.emory.edu}"
REMOTE_DIR="${CS371_REMOTE_DIR:-/users/vmli3/cs371w}"
REMOTE_SCRATCH="${CS371_REMOTE_SCRATCH:-/scratch/${CS371_REMOTE_USER:-vmli3}/cs371w}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

DATE="${1:-}"
if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "usage: $(basename "$0") YYYY-MM-DD [--cells] [extra rsync flags]" >&2
    exit 1
fi
shift

WANT_CELLS=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--cells" ]; then WANT_CELLS=1; else ARGS+=("$a"); fi
done

SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/cs371_sync-%C -o ControlPersist=60s"

for sub in logs results; do
    echo "--- pull: ${REMOTE}:${REMOTE_DIR}/${sub}/${DATE}/ -> ${LOCAL_DIR}/${sub}/${DATE}/ ---"
    mkdir -p "${LOCAL_DIR}/${sub}/${DATE}"
    rsync -avzh --progress -e "ssh ${SSH_MUX}" \
        ${ARGS[@]+"${ARGS[@]}"} \
        "${REMOTE}:${REMOTE_DIR}/${sub}/${DATE}/" \
        "${LOCAL_DIR}/${sub}/${DATE}/"
done

if [ "$WANT_CELLS" = "1" ]; then
    echo "--- pull: cells (from scratch) ---"
    mkdir -p "${LOCAL_DIR}/data/interim/aotil/cells"
    rsync -avzh --progress -e "ssh ${SSH_MUX}" \
        ${ARGS[@]+"${ARGS[@]}"} \
        "${REMOTE}:${REMOTE_SCRATCH}/aotil/cells/" \
        "${LOCAL_DIR}/data/interim/aotil/cells/"
fi

echo "Pulled ${DATE} logs/ + results/${WANT_CELLS:+ (+ cells)} -> ${LOCAL_DIR}"
