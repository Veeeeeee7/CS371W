#!/usr/bin/env bash
# Download the national NFHL geodatabase (~20 GB) over one connection, resuming until it is
# complete, then test the archive. Safe to Ctrl-C and rerun: it resumes at the byte it stopped on.
#
# One connection on purpose: hazards.fema.gov drops roughly 1 in 6 concurrent connections, and
# aria2 aborts the whole transfer when any one of its streams fails a TLS handshake.
#
#     ./fetch_nfhl.sh [DEST_DIR]        # default data/raw/nfhl

set -uo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "usage: ./fetch_nfhl.sh [DEST_DIR]   (default data/raw/nfhl)"
  exit 0
fi

DEST_DIR="${1:-data/raw/nfhl}"
ENTRY="https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip"
OUT="$DEST_DIR/NFHL_Key_Layers.gdb.zip"

mkdir -p "$DEST_DIR"

# Resolve the redirect once: the target filename carries a date that changes when FEMA
# republishes, and re-resolving mid-download could silently switch files.
echo "resolving..."
URL=$(curl -sSI -L --retry 8 --retry-all-errors --retry-delay 3 --max-time 120 "$ENTRY" \
      | awk 'tolower($1)=="location:"{print $2}' | tr -d '\r' | tail -1)
if [ -z "$URL" ]; then
  URL="$ENTRY"
fi
echo "url:  $URL"

EXPECTED=$(curl -sSI --retry 8 --retry-all-errors --retry-delay 3 --max-time 120 "$URL" \
           | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
if ! [[ "$EXPECTED" =~ ^[0-9]+$ ]]; then
  echo "could not read content-length; aborting" >&2
  exit 1
fi
echo "size: $EXPECTED bytes ($(echo "$EXPECTED" | awk '{printf "%.2f GB", $1/1e9}'))"
echo

# Bytes on disk so far: BSD stat (macOS) first, then GNU stat (Linux).
have() {
  if [ ! -f "$OUT" ]; then
    echo 0
    return
  fi
  stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT" 2>/dev/null || echo 0
}

attempt=0
while :; do
  cur=$(have)
  cur=${cur:-0}
  if [ "$cur" -ge "$EXPECTED" ]; then
    echo
    echo "complete: $cur bytes"
    break
  fi
  attempt=$((attempt+1))
  pct=$(awk -v c="$cur" -v e="$EXPECTED" 'BEGIN{printf "%.1f", 100*c/e}')
  echo "[attempt $attempt] at ${pct}% (${cur}/${EXPECTED}) -- resuming"

  # -C - resumes from wherever the file ends; --speed-limit/--speed-time kill a stalled
  # socket so the loop retries.
  curl -L -C - \
       --retry 10 --retry-all-errors --retry-delay 5 \
       --speed-limit 10240 --speed-time 60 \
       --connect-timeout 30 \
       -o "$OUT" "$URL"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  curl exit $rc -- backing off 15s"
    sleep 15
  fi
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
