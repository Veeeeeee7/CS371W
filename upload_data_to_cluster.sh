#!/usr/bin/env bash
# =============================================================================
# Push data/processed/ (the per-state parquets + FEATURES.txt) to cluster SCRATCH.
#
# Separate from the code push because it is ~220 MB and changes rarely, while
# code changes every session. Run it when the dataset is regenerated, and only
# then -- the jobs read it but never write it.
#
# IT GOES TO SCRATCH, NOT HOME. Home quota is the binding constraint here, and
# the parquets plus the cells the jobs write are the only large things involved.
# Scratch is periodically PURGED, which is survivable precisely because both are
# reconstructible: the parquets re-upload from the laptop in minutes and a cell
# refits from them. Results and logs stay in the home checkout for the same
# reason -- a purge must cost compute, never a number.
#
# ONLY data/processed/ GOES UP. data/raw/ is a 19 GB NFHL archive plus 7.8 GB of
# NHDPlus, and data/interim/prep/ is another 20 GB; none of it is read by any
# job on the cluster. The cells the jobs WRITE come back down via
# ./download_from_cluster.sh --cells, they do not go up.
#
# The guard below refuses the push if FEATURES.txt and the parquets disagree,
# because a cluster running a stale feature list against fresh data produces
# rows that look fine and are not comparable to anything.
#
# Usage:
#   ./upload_data_to_cluster.sh
#   ./upload_data_to_cluster.sh --dry-run
# =============================================================================
set -euo pipefail

REMOTE_USER="${CS371_REMOTE_USER:-vmli3}"
REMOTE_HOST="${CS371_REMOTE_HOST:-cirrostratus.it.emory.edu}"
REMOTE_DIR="${CS371_REMOTE_DIR:-/users/vmli3/cs371w}"
# Where the DATA lives on the cluster. Must match AOTIL_ARTIFACT_ROOT in
# run_aotil.sh, which defaults to /scratch/$USER/cs371w.
REMOTE_SCRATCH="${CS371_REMOTE_SCRATCH:-/scratch/${CS371_REMOTE_USER:-vmli3}/cs371w}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SRC="${LOCAL_DIR}/data/processed"

[ -d "$SRC" ] || { echo "ERROR: $SRC does not exist" >&2; exit 1; }
[ -f "$SRC/FEATURES.txt" ] || { echo "ERROR: $SRC/FEATURES.txt missing" >&2; exit 1; }

echo "--- guard: features vs parquet columns ---"
python3 - "$SRC" <<'PY'
import sys, glob, os
from pathlib import Path
import pandas as pd
src = Path(sys.argv[1])
feats = [f for line in (src / "FEATURES.txt").read_text().splitlines()
         if line.strip() and not line.startswith("#") for f in line.split()]
bad = 0
for p in sorted(src.glob("dataset_*.parquet")):
    cols = set(pd.read_parquet(p, columns=None).columns[:0].union(
        pd.read_parquet(p).columns))
    missing = [f for f in feats if f not in cols]
    # lon/lat are join keys rather than features, but the spatial work cannot
    # run without them and a dataset regenerated without them fails only after
    # the model population has already been fitted.
    for need in ("lon", "lat", "tile_50km", "block_id", "domain", "label",
                 "neg_protocol"):
        if need not in cols:
            missing.append(need)
    if missing:
        print(f"  {p.name}: MISSING {missing}")
        bad += 1
if bad:
    print(f"\nRefusing the push: {bad} file(s) disagree with FEATURES.txt.")
    sys.exit(1)
print(f"  ok -- {len(feats)} features present in every dataset_*.parquet")
PY

SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/cs371_sync-%C -o ControlPersist=60s"
ssh ${SSH_MUX} "${REMOTE}" "mkdir -p '${REMOTE_SCRATCH}/data/processed' '${REMOTE_SCRATCH}/aotil/cells'"

echo "--- push: data/processed -> ${REMOTE}:${REMOTE_SCRATCH}/data/processed ---"
rsync -avzh --progress -e "ssh ${SSH_MUX}" "$@" \
  "${SRC}/" "${REMOTE}:${REMOTE_SCRATCH}/data/processed/"

echo
echo "Pushed $(du -sh "$SRC" | cut -f1) of processed data to SCRATCH:"
echo "  ${REMOTE_SCRATCH}/data/processed   <- the jobs read this"
echo "  ${REMOTE_SCRATCH}/aotil/cells      <- the jobs write cells here"
echo "Code and results stay in ${REMOTE_DIR}."
echo "Re-run this after a scratch purge; nothing here is a result."
