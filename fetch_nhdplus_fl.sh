#!/usr/bin/env bash
# fetch_nhdplus_fl.sh -- download and extract NHDPlus V2.1 for Florida: VPUs 03N, 03S, 03W.
#
# Every Florida catchment lies in Region 03, split into units 03N, 03S and 03W (547 MB in
# nine archives). Per unit, only the files pull_hydro.py needs are extracted:
#   NHDSnapshot        NHDFlowline: COMID, FTYPE, LENGTHKM + geometry
#   NHDPlusAttributes  PlusFlowlineVAA (StreamOrde, TotDASqKM), elevslope (SLOPE)
#   NHDPlusCatchment   Catchment: FEATUREID, AreaSqKM + geometry
# into data/raw/nhdplus/<unit>/<layer>/; prep_nhdplus_units.py then builds the geodatabase.
# Finished downloads and extractions are skipped on a rerun.
#
#     ./fetch_nhdplus_fl.sh

set -uo pipefail

BASE="https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusSA"
DEST_DIR="data/raw/nhdplus"

# unit  layer           archive                                     bytes (S3 Content-Length)
FILES="
03N NHDSnapshot       NHDPlusV21_SA_03N_NHDSnapshot_04.7z         56396271
03N NHDPlusAttributes NHDPlusV21_SA_03N_NHDPlusAttributes_07.7z   13780514
03N NHDPlusCatchment  NHDPlusV21_SA_03N_NHDPlusCatchment_01.7z   143642141
03S NHDSnapshot       NHDPlusV21_SA_03S_NHDSnapshot_06.7z         42252892
03S NHDPlusAttributes NHDPlusV21_SA_03S_NHDPlusAttributes_07.7z    5449377
03S NHDPlusCatchment  NHDPlusV21_SA_03S_NHDPlusCatchment_01.7z    68475716
03W NHDSnapshot       NHDPlusV21_SA_03W_NHDSnapshot_04.7z         57629089
03W NHDPlusAttributes NHDPlusV21_SA_03W_NHDPlusAttributes_07.7z   13677712
03W NHDPlusCatchment  NHDPlusV21_SA_03W_NHDPlusCatchment_01.7z   145355423
"

command -v 7z >/dev/null 2>&1 || { echo "7z not found: brew install p7zip" >&2; exit 1; }

members() {
    case "$1" in
        NHDSnapshot)       echo "NHDFlowline.*" ;;
        NHDPlusAttributes) echo "PlusFlowlineVAA.dbf elevslope.dbf" ;;
        NHDPlusCatchment)  echo "Catchment.*" ;;
    esac
}

# Resume the download until the file has its known size; give up after 20 attempts.
fetch() {   # url dest bytes
    local url=$1 dest=$2 expect=$3 attempt=0 have
    while : ; do
        have=$([ -f "$dest" ] && wc -c < "$dest" | tr -d ' ' || echo 0)
        [ "$have" -ge "$expect" ] && return 0
        attempt=$((attempt + 1))
        echo "  [attempt $attempt] have $(printf "%'d" "$have") / $(printf "%'d" "$expect")"
        curl -sS -L -C - --speed-limit 10240 --speed-time 60 \
             --retry 5 --retry-delay 10 -o "$dest" "$url"
        [ "$attempt" -ge 20 ] && { echo "giving up on $dest" >&2; return 1; }
        sleep 2
    done
}

echo "$FILES" | while read -r unit layer file bytes; do
    [ -z "$unit" ] && continue
    dl="$DEST_DIR/archives"
    out="$DEST_DIR/$unit/$layer"
    mkdir -p "$dl" "$out"
    echo "$file ($(printf "%'d" "$bytes") bytes)"
    fetch "$BASE/NHDPlus$unit/$file" "$dl/$file" "$bytes" || exit 1
    # .extracted marks a finished extraction (check_gis_inputs.py reads it too).
    if [ -f "$out/.extracted" ]; then
        echo "  extracted already"
        continue
    fi
    7z t "$dl/$file" >/dev/null || { echo "  archive FAILED its test: delete it and rerun" >&2; exit 1; }
    # The member patterns are 7z wildcards, so keep the shell from globbing them.
    set -f
    # shellcheck disable=SC2046
    7z x -y -r -bso0 -bsp0 "$dl/$file" -o"$out" $(members "$layer") \
        || { echo "  extraction failed" >&2; exit 1; }
    set +f
    touch "$out/.extracted"
    echo "  -> $out"
done || exit 1

echo
du -sh "$DEST_DIR"/archives "$DEST_DIR"/03?
echo "next: .venv/bin/python prep_nhdplus_units.py"
