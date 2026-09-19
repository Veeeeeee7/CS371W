#!/usr/bin/env python3
"""
nfhl_zone_stats.py — week-1 go/no-go, computed server-side.

Replaces discover_nfhl.py. Downloads nothing: FEMA's ArcGIS REST service
aggregates flood-zone counts for us, so the premise check needs no GDBs.

Two bugs in the original script this fixes:
  1. No retry. hazards.fema.gov and msc.fema.gov reset TLS connections
     constantly (SSL_ERROR_SYSCALL / ECONNRESET 54). Roughly one call in
     three dies. This is FEMA, not your machine.
  2. msc.fema.gov/portal/advanceSearch returns HTML, not JSON, for a
     state-only POST -- r.json() would have thrown even on a clean connection.

Usage:
    python nfhl_zone_stats.py                # all 10 states
    python nfhl_zone_stats.py VT IA LA       # subset
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LAYER = ("https://hazards.fema.gov/arcgis/rest/services"
         "/public/NFHL/MapServer/28/query")

STATES = {"FL": "12", "LA": "22", "IA": "19", "CO": "08", "AZ": "04",
          "NC": "37", "VT": "50", "TX": "48", "MN": "27", "MO": "29"}

OUT = Path("data/interim/nfhl_zone_stats")
TIMEOUT = (30, 300)          # (connect, read) -- big states are slow
MAX_ATTEMPTS = 8


def make_session() -> requests.Session:
    """Session that survives FEMA's connection resets."""
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,           # <-- this is the one that catches ECONNRESET
        read=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "CS371W-research/1.0"})
    return s


def query_counts(session, where: str) -> dict[str, int] | None:
    """Server-side COUNT(*) grouped by FLD_ZONE. None if it never succeeds."""
    params = {
        "where": where,
        "groupByFieldsForStatistics": "FLD_ZONE",
        "outStatistics": json.dumps([{
            "statisticType": "count",
            "onStatisticField": "FLD_AR_ID",
            "outStatisticFieldName": "n",
        }]),
        "f": "json",
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = session.get(LAYER, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                print(f"    server error: {d['error'].get('message')}", file=sys.stderr)
                return None
            return {f["attributes"]["FLD_ZONE"] or "<null>": f["attributes"]["n"]
                    for f in d.get("features", [])}
        except (requests.exceptions.RequestException, ValueError) as e:
            wait = min(2 ** attempt, 60)
            print(f"    attempt {attempt}/{MAX_ATTEMPTS} failed ({type(e).__name__}); "
                  f"retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def main(argv: list[str]) -> int:
    targets = argv or list(STATES)
    bad = [s for s in targets if s not in STATES]
    if bad:
        print(f"unknown state(s): {bad}; known: {list(STATES)}", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    session = make_session()
    summary = {}

    for abbr in targets:
        fips = STATES[abbr]
        print(f"\n=== {abbr} ({fips}) ===", flush=True)

        # DFIRM_ID is the county/community FIRM id; its first two chars are the
        # state FIPS. SFHA_TF='T' restricts to the Special Flood Hazard Area.
        base = f"DFIRM_ID LIKE '{fips}%' AND SFHA_TF='T'"

        total = query_counts(session, base)
        if total is None:
            print("  FAILED -- skipping", file=sys.stderr)
            continue

        # Real (non-sentinel) base flood elevations. -9999 is FEMA's null.
        with_bfe = query_counts(session, base + " AND STATIC_BFE <> -9999") or {}

        print(f"  {'zone':<10}{'polygons':>10}{'real BFE':>10}{'pct':>8}")
        for z in sorted(total, key=lambda k: -total[k]):
            n, b = total[z], with_bfe.get(z, 0)
            print(f"  {z:<10}{n:>10,}{b:>10,}{100 * b / n:>7.1f}%")

        a, ae = total.get("A", 0), total.get("AE", 0)
        share = 100 * a / (a + ae) if (a + ae) else float("nan")
        print(f"  Zone A share of A+AE: {share:.1f}%")

        summary[abbr] = {"total": total, "with_real_bfe": with_bfe,
                         "zone_a_share_pct": round(share, 2)}
        (OUT / f"{abbr}.json").write_text(json.dumps(summary[abbr], indent=2))

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 46)
    print(f"{'state':<8}{'Zone A':>10}{'Zone AE':>10}{'A share':>12}")
    for abbr, s in summary.items():
        t = s["total"]
        print(f"{abbr:<8}{t.get('A', 0):>10,}{t.get('AE', 0):>10,}"
              f"{s['zone_a_share_pct']:>11.1f}%")
    print(f"\nwrote {OUT}/_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
