#!/usr/bin/env bash
# =============================================================================
# Push this repository's CODE to the compute cluster over rsync/ssh.
#
# Code moves up; results, logs and cells move down. Experiments are written by
# Slurm on the remote, so pull them with ./download_from_cluster.sh rather than
# letting rsync mirror both ways -- a two-way sync over a results tree that
# running jobs are appending to is how rows get lost.
#
# ALLOW-LIST, NOT A BLOCK-LIST. Everything is refused by the trailing
# --exclude='*'; only the patterns named above it are sent. A block-list fails
# the other way: every new local directory ships until someone remembers to add
# it, and this repo holds a 19 GB NFHL archive and a 20 GB interim tree that
# have no business on shared cluster storage. An allow-list cannot fail that
# way -- a new directory is excluded by construction, not by memory. If you add
# a file type the jobs need at runtime, add its --include here or the cluster
# runs without it.
#
# data/ is pushed separately by ./upload_data_to_cluster.sh, and it goes to
# SCRATCH rather than here -- home quota is the binding constraint, and the
# parquets and the cells are the only large things in play. This script creates
# both trees so the first job does not fail on a missing directory.
#
# The split: code and results in ${CS371_REMOTE_DIR}, data and cells in
# ${CS371_REMOTE_SCRATCH}. Scratch is purged and everything on it is
# reconstructible; nothing that is a result is ever written there.
#
# Usage:
#   ./upload_to_cluster.sh              push for real
#   ./upload_to_cluster.sh --dry-run    preview; copies nothing
#   ./upload_to_cluster.sh --delete     mirror deletions on the remote code tree
#
# Point at a different cluster with CS371_REMOTE_USER / _HOST / _DIR.
# =============================================================================
set -euo pipefail

REMOTE_USER="${CS371_REMOTE_USER:-vmli3}"
REMOTE_HOST="${CS371_REMOTE_HOST:-cirrostratus.it.emory.edu}"
REMOTE_DIR="${CS371_REMOTE_DIR:-/users/vmli3/cs371w}"
REMOTE_SCRATCH="${CS371_REMOTE_SCRATCH:-/scratch/${CS371_REMOTE_USER:-vmli3}/cs371w}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# One shared, reused ssh connection for mkdir + rsync, so you enter your
# passphrase/Duo once per run rather than three times.
SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/cs371_sync-%C -o ControlPersist=60s"

# logs/sbatch/ holds the #SBATCH --output/--error stubs -- everything a driver
# emits BEFORE it redirects into the dated folder, which is where the fail-fast
# guards and slurmstepd's own OOM messages land. Slurm does NOT create
# intermediate directories, so a missing logs/sbatch/ fails the job instantly.
ssh ${SSH_MUX} "${REMOTE}" \
    "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/logs/sbatch' \
              '${REMOTE_SCRATCH}/data/processed' '${REMOTE_SCRATCH}/aotil/cells'"

echo "--- push: code -> ${REMOTE}:${REMOTE_DIR} ---"
rsync -avzh --progress -e "ssh ${SSH_MUX}" "$@" \
  --exclude='data/' \
  --exclude='refs/' \
  --include='/*.py' \
  --include='/*.sh' \
  --include='/*.md' \
  --include='/requirements.txt' \
  --include='/aotil/' \
  --include='/aotil/*.py' \
  --include='/aotil/*.md' \
  --include='/docs/' \
  --include='/docs/*.md' \
  --exclude='*' \
  "${LOCAL_DIR}/" \
  "${REMOTE}:${REMOTE_DIR}/"

echo
echo "Pushed code -> ${REMOTE}:${REMOTE_DIR}"
echo "  code + results + logs : ${REMOTE_DIR}"
echo "  data + cells          : ${REMOTE_SCRATCH}   (purgeable, reconstructible)"
echo "If the dataset changed, or scratch was purged: ./upload_data_to_cluster.sh"
