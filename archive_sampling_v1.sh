#!/usr/bin/env bash
# Move Florida's sampling-v1 files aside before the v2 build (docs/RUN-FL-V2.md step 1).
#
# Protocol v2 reissues point_ids FL0000000... for different locations, and the pull scripts treat
# an existing output as done, so a v1 file keyed by point_id left in place would be joined onto
# the v2 points silently. Each such file is MOVED (same volume, no disk cost) to
#     data/interim/archive/sampling-v1/<same relative path>
# The Atlas 14 block cache and FEATURES.txt are COPIED instead: the v2 run updates both in place,
# and moving them would lose the entries it does not rewrite.
#
# Before anything moves, sha256_before.txt records every dataset_*.parquet, FEATURES.txt and
# blocks.parquet; MANIFEST.txt lists what was moved or copied, with sizes. Safe to rerun: an entry
# already in the archive is skipped, and nothing in the archive is ever overwritten.
#
#     ./archive_sampling_v1.sh

set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "usage: ./archive_sampling_v1.sh   (from the repo root; docs/RUN-FL-V2.md step 1)"
  exit 0
fi

ARCH="data/interim/archive/sampling-v1"

MOVE="
data/interim/points/points_FL.parquet
data/processed/dataset_FL.parquet
data/interim/gee/FL
data/interim/hydro/comid_FL.parquet
data/interim/hydro/river_FL.parquet
data/interim/aotil/cells/FL_band_AE_s0.npz
data/interim/aotil/cells/FL_band_AE_s0_models.csv
data/interim/aotil/cells/FL_band_A_s0.npz
data/interim/aotil/cells/FL_band_A_s0_models.csv
data/interim/aef/aef_FL_2017.parquet
data/interim/aef/aef_FL_2024.parquet
data/interim/aef/2017/FL
data/interim/aef/2024/FL
data/interim/aef/aef_tiles_FL_2017.geojson
data/interim/aef/aef_tiles_FL_2024.geojson
"
COPY="
data/interim/atlas14/blocks.parquet
data/processed/FEATURES.txt
"

[ -f sample_points.py ] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p "$ARCH"

if [ ! -f "$ARCH/sha256_before.txt" ]; then
    shasum -a 256 data/processed/dataset_*.parquet data/processed/FEATURES.txt \
        data/interim/atlas14/blocks.parquet > "$ARCH/sha256_before.txt"
    echo "checksums -> $ARCH/sha256_before.txt"
fi
df -h . | tail -1 | awk '{print "free before: " $4}' | tee -a "$ARCH/MANIFEST.txt"

for p in $MOVE; do
    dest="$ARCH/$p"
    if [ -e "$dest" ]; then
        echo "archived already: $p"
        continue
    fi
    if [ ! -e "$p" ]; then
        echo "MISSING, not moved: $p" | tee -a "$ARCH/MANIFEST.txt"
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    size=$(du -sh "$p" | cut -f1)
    mv "$p" "$dest"
    echo "moved  $size  $p" | tee -a "$ARCH/MANIFEST.txt"
done

for p in $COPY; do
    dest="$ARCH/$p"
    if [ -e "$dest" ]; then
        echo "copied already: $p"
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    cp -p "$p" "$dest"
    echo "copied $(du -sh "$p" | cut -f1)  $p" | tee -a "$ARCH/MANIFEST.txt"
done

echo
echo "archive: $ARCH ($(du -sh "$ARCH" | cut -f1))"
