#!/usr/bin/env bash
# fetch_nfhl.sh -- resume-until-done downloader for the national NFHL geodatabase.
#
# Why not aria2: hazards.fema.gov drops roughly 1 in 6 concurrent connections.
# Measured: 3/3 single connections returned HTTP 206; 5/6 parallel ones did, and
# aria2 aborts the whole transfer when any one of its 8 streams fails a TLS
# handshake. One connection, resumed forever, is slower per-second and far
# faster in wall-clock terms because it actually finishes.
#
# Safe to Ctrl-C and rerun -- it picks up at the byte it stopped on.

set -uo pipefail

DEST_DIR="${1:-data/raw/nfhl}"
ENTRY="https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip"
OUT="$DEST_DIR/NFHL_Key_Layers.gdb.zip"

mkdir -p "$DEST_DIR"

# Resolve the redirect once. The filename carries a date that changes when FEMA
# republishes, and re-resolving mid-download could silently switch files.
echo "resolving..."
URL=$(curl -sSI -L --retry 8 --retry-all-errors --retry-delay 3 --max-time 120 "$ENTRY" \
      | awk 'tolower($1)=="location:"{print $2}' | tr -d '\r' | tail -1)
[ -z "$URL" ] && URL="$ENTRY"
echo "url:  $URL"

EXPECTED=$(curl -sSI --retry 8 --retry-all-errors --retry-delay 3 --max-time 120 "$URL" \
           | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
if ! [[ "$EXPECTED" =~ ^[0-9]+$ ]]; then
  echo "could not read content-length; aborting" >&2; exit 1
fi
echo "size: $EXPECTED bytes ($(echo "$EXPECTED" | awk '{printf "%.2f GB", $1/1e9}'))"
echo

have() { [ -f "$OUT" ] && stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT" 2>/dev/null || echo 0; }

attempt=0
while :; do
  cur=$(have); cur=${cur:-0}
  if [ "$cur" -ge "$EXPECTED" ]; then
    echo; echo "complete: $cur bytes"; break
  fi
  attempt=$((attempt+1))
  pct=$(awk -v c="$cur" -v e="$EXPECTED" 'BEGIN{printf "%.1f", 100*c/e}')
  echo "[attempt $attempt] at ${pct}% (${cur}/${EXPECTED}) -- resuming"

  # -C -  resume from wherever the file currently ends
  # single connection on purpose; --speed-* kills a stalled socket so the loop retries
  curl -L -C - \
       --retry 10 --retry-all-errors --retry-delay 5 \
       --speed-limit 10240 --speed-time 60 \
       --connect-timeout 30 \
       -o "$OUT" "$URL"
  rc=$?
  [ $rc -ne 0 ] && { echo "  curl exit $rc -- backing off 15s"; sleep 15; }
done

echo
echo "verifying archive integrity (this takes a few minutes)..."
if unzip -t "$OUT" > /tmp/nfhl_unzip_test.log 2>&1; then
  tail -2 /tmp/nfhl_unzip_test.log
  echo "OK"
else
  echo "ARCHIVE CORRUPT -- delete $OUT and rerun" >&2
  tail -5 /tmp/nfhl_unzip_test.log >&2
  exit 1
fi
