#!/usr/bin/env bash
# fetch_nhdplus.sh -- NHDPlus V2 national seamless geodatabase, resumably.
#
# 7.81 GB. Needed for two things the current covariate stack cannot supply:
#   Catchment  -> COMID per point, which is the join key for every StreamCat metric
#   Flowline   -> true distance-to-river and drainage density
#
# Same single-connection resume loop as fetch_nfhl.sh. That pattern exists because
# the parallel-chunk downloader failed on the FEMA archive: 5 of 6 concurrent range
# requests succeeded and the one failure aborted the whole transfer. The server
# here advertises Accept-Ranges, so an interrupted run resumes instead of
# restarting.
#
# Run it in the background and get on with the Earth Engine pull:
#     nohup ./fetch_nhdplus.sh > nhdplus_download.log 2>&1 &
#     tail -f nhdplus_download.log

set -uo pipefail

URL="https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/NationalData/NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z"
DEST_DIR="data/raw/nhdplus"
DEST="${DEST_DIR}/NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z"
EXPECT=7812330106

mkdir -p "$DEST_DIR"

have_bytes() { [ -f "$DEST" ] && wc -c < "$DEST" | tr -d ' ' || echo 0; }

echo "target : $DEST"
echo "size   : $(printf "%'d" $EXPECT) bytes"
echo

attempt=0
while : ; do
    have=$(have_bytes)
    if [ "$have" -ge "$EXPECT" ]; then
        echo "complete: $(printf "%'d" "$have") bytes"
        break
    fi
    attempt=$((attempt + 1))
    echo "[attempt $attempt] have $(printf "%'d" "$have") / $(printf "%'d" $EXPECT)"

    # -C -        resume at the current byte offset
    # --speed-*   give up on a connection stalled under 10 KB/s for 60 s, so a
    #             dead socket is retried rather than hanging the whole night
    curl -L -C - \
         --speed-limit 10240 --speed-time 60 \
         --retry 5 --retry-delay 10 \
         -o "$DEST" "$URL"

    if [ "$attempt" -ge 40 ]; then
        echo "giving up after $attempt attempts" >&2
        exit 1
    fi
    sleep 5
done

echo
echo "verifying archive..."
if command -v 7z >/dev/null 2>&1; then
    7z t "$DEST" >/dev/null 2>&1 \
        && echo "archive OK" \
        || { echo "archive FAILED its integrity test -- delete it and rerun" >&2; exit 1; }
    echo
    echo "extract with:"
    echo "    7z x $DEST -o$DEST_DIR"
else
    echo "7z is not installed, so the archive could not be tested."
    echo "macOS cannot open .7z natively:"
    echo "    brew install p7zip"
    echo "    7z t $DEST        # test"
    echo "    7z x $DEST -o$DEST_DIR"
fi

echo
echo "Extracted it is roughly 20 GB. Only two feature classes are needed:"
echo "    Catchment   COMID + geometry, for the StreamCat join"
echo "    NHDFlowline for distance-to-river and drainage density"
echo "Subset those to the ten states, then delete the extracted geodatabase --"
echo "the same pattern that took the NFHL from 26 GB down to what we kept."
