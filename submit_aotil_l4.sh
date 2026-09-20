#!/bin/bash
# =============================================================================
# submit_aotil_l4.sh -- the AoTIL sweep on the L4 partition
#
# =============================================================================
# Same driver and same fan-out as submit_aotil_all.sh; only the allocation
# changes, through sbatch CLI overrides that beat the driver's #SBATCH header.
# Use it when the rp6b nodes are busy.
#
#   partition  l4-4-gm96-c96-m384-dy-g6-24xlarge-1
#   per job    1 GPU, 16 CPUs, 64G  -- one clean quarter of a g6.24xlarge
#                                      (4x L4 24GB, 96 vCPU, 384G), so four of
#                                      these jobs fit one node with no contention
#   walltime   04:00:00
#
# THE `-dy-` IN THE NAME IS A DYNAMIC NODE. It is powered on on demand, so the
# first job into an idle partition sits in CF/CONFIGURING for several minutes
# before it starts. That is not a hang and not a queue problem -- do not cancel
# and resubmit, which just restarts the boot. `squeue -j <id> -o "%T %R"` shows
# CONFIGURING while it happens.
#
# WALLTIME IS SHORT ON PURPOSE. Priority here is 1000*Age + 10000*FairShare +
# 1000*QOS with PriorityWeightJobSize=0, so a shorter walltime does NOT raise a
# job's rank -- but it does make it fit backfill windows, which is the whole
# reason to be on this partition at all when the preferred nodes are occupied. A
# cell is minutes of work on 16 cores; 4 hours is already generous. Raise
# WALLTIME rather than letting Slurm kill a cell at the limit, because a killed
# cell leaves no .npz at all.
#
# Usage, from the repo root on the login node:
#   ./submit_aotil_l4.sh --date YYYY-MM-DD
#   LIMIT=1 ./submit_aotil_l4.sh --date YYYY-MM-DD      # watch one first
#   CPUS=32 MEM=128G ./submit_aotil_l4.sh --date YYYY-MM-DD   # half a node
#
# Any knob set in your shell wins: this script only supplies defaults.
# =============================================================================
set -euo pipefail

PARTITION="${PARTITION:-l4-4-gm96-c96-m384}"
CPUS="${CPUS:-16}"
MEM="${MEM:-64G}"
GPUS="${GPUS:-1}"
WALLTIME="${WALLTIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-general}"
export PARTITION CPUS MEM GPUS WALLTIME ACCOUNT

# PREFLIGHT: a partition can exist, show idle nodes, and still refuse every
# account you hold -- the submit then fails with "Invalid account or
# account/partition combination specified" after you have already fanned out.
# Check the association first and say which accounts DO work, rather than
# letting eight submits fail one after another.
if command -v sacctmgr >/dev/null 2>&1; then
    assoc=$(sacctmgr -nP show assoc user="$USER" format=account,partition 2>/dev/null || true)
    if [ -n "$assoc" ]; then
        # An association with an EMPTY partition field grants every partition.
        if ! printf '%s\n' "$assoc" | grep -qE "^${ACCOUNT}\|(${PARTITION})?$"; then
            echo "WARNING: no association found for account='${ACCOUNT}' on" >&2
            echo "         partition='${PARTITION}'. Submits may be refused." >&2
            echo "         Your associations:" >&2
            printf '%s\n' "$assoc" | sed 's/^/           /' >&2
            echo "         Set ACCOUNT=<one that covers this partition> and retry," >&2
            echo "         or continue anyway if the table above looks permissive." >&2
            if [ "${FORCE:-0}" != "1" ]; then
                echo "         (FORCE=1 to submit regardless.)" >&2
                exit 1
            fi
        fi
    fi
fi

if command -v sinfo >/dev/null 2>&1; then
    echo "--- partition state ---"
    sinfo -p "$PARTITION" -o "%P %a %l %D %t %G %c %m" 2>/dev/null \
        || echo "  (sinfo could not read '$PARTITION' -- check the name)"
    echo
fi

exec ./submit_aotil_all.sh "$@"
