#!/usr/bin/env python3
"""
pull_ssurgo.py -- the Soil Data Access (SDA) helpers pull_soil_points.py uses.

NRCS Soil Data Access serves SSURGO, the county soil survey, as SQL over HTTPS. This module holds
the polite client (SDA: retries with backoff, request and byte counts), the per-batch parquet
cache (_write, _digest, _load_store), the per-key table fetch (fetch_table), the sacatalog query
(SA_COLS, SA_SQL) and WATER_MU, the name rule that tells a water map unit from a soil. It has no
command line: pull_soil_points.py looks up the points and codes the two soil features.
"""
from __future__ import annotations

import hashlib
import html
import os
import random
import re
import threading
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

# NRCS Soil Data Access (SDA). A rejected query is HTTP 400 with an OGC ServiceException XML
# body; an empty result is the JSON {}.
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

# A water map unit's name starts with the word "Water" or "Waters", or is "Dredge channel". On
# every Florida map unit with a Water component, this picks exactly the units that are at least
# half water. It leaves out "Arents-Water complex", "Tidewater muck" and the subaqueous soils
# (soils mapped on a lagoon floor). mapunit.mukind cannot tell water from soil.
WATER_MU = re.compile(r"^(Waters?|Dredge channel)\b")

RETRIES = 5
CONNECT_S = 30            # no connection in 30 s is an outage, not a big query
TIMEOUT_S = 300           # read timeout per request, seconds

ATTR_BATCH = 1000         # keys per muaggatt or sacatalog request

# sacatalog names each survey area and dates its last export (saverest).
SA_COLS = ["areasymbol", "areaname", "saverest"]
SA_SQL = "SELECT " + ", ".join(SA_COLS) + " FROM sacatalog WHERE areasymbol IN ({keys})"


# =========================================================================
# Soil Data Access
# =========================================================================
class SDAError(RuntimeError):
    """SDA refused a query (bad SQL) or kept failing after the retries."""


class SDATooBig(SDAError):
    """SDA rejected a query for size, time or memory: split it and ask again."""


_TOO_BIG = re.compile(r"time ?out|timed out|memory|exceed|too large|too many|"
                      r"maximum|serializ", re.I)


def _service_message(text: str) -> str:
    """The text of an OGC ServiceException body, or else the start of the body."""
    m = re.search(r"<ServiceException>(.*?)</ServiceException>", text, re.S)
    if m:
        return html.unescape(m.group(1)).strip()
    return text.strip()[:300]


class SDA:
    """A polite POST client: retries with backoff, counts requests and bytes received."""

    def __init__(self, workers: int):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(pool_maxsize=max(2, 2 * workers)))
        self.lock = threading.Lock()        # the counters are shared by the worker threads
        self.n_req = 0
        self.n_bytes = 0

    def query(self, sql: str, cols: list[str]) -> pd.DataFrame:
        """The result of `sql`, whose columns must be `cols`. Raises SDATooBig when the query
        should be split, SDAError when it failed for another reason."""
        err: SDAError | None = None
        for attempt in range(RETRIES):
            if attempt:
                time.sleep(min(60.0, 2.0 ** attempt) + random.random())
            try:
                resp = self.session.post(SDA_URL,
                                         json={"query": sql, "format": "JSON+COLUMNNAME"},
                                         timeout=(CONNECT_S, TIMEOUT_S))
            except requests.ConnectTimeout as e:
                # the server is down: retry, but never split
                err = SDAError(f"no connection in {CONNECT_S}s: {e}")
                continue
            except requests.Timeout as e:
                # no answer in TIMEOUT_S: try once more, then split
                err = SDATooBig(f"no response in {TIMEOUT_S}s: {e}")
                if attempt >= 1:
                    break
                continue
            except requests.RequestException as e:
                err = SDAError(f"{type(e).__name__}: {e}")
                continue
            with self.lock:
                self.n_req += 1
                self.n_bytes += len(resp.content)

            if resp.status_code == 200:
                try:
                    table = resp.json().get("Table")
                except ValueError:
                    # a truncated body: retry, then split
                    err = SDATooBig(f"unparseable response, {len(resp.content):,} bytes")
                    continue
                if not table:
                    return pd.DataFrame(columns=cols)
                if table[0] != cols:        # JSON+COLUMNNAME: the first row holds the names
                    raise SDAError(f"unexpected columns {table[0]}, wanted {cols}")
                return pd.DataFrame(table[1:], columns=cols)

            msg = f"HTTP {resp.status_code}: {_service_message(resp.text)}"
            if _TOO_BIG.search(msg):
                # a timeout can be load, so ask once more; a size cap will not move
                err = SDATooBig(msg)
                if attempt >= 1:
                    break
            elif resp.status_code in (429, 500, 502, 503, 504):
                err = SDAError(msg)         # busy: retry
            else:
                raise SDAError(msg)         # bad SQL: retrying will not help
        assert err is not None
        raise err


# =========================================================================
# caching (pull_soil_points.py keeps it under <soil-dir>/cache/, so an interrupted run
# resumes)
# =========================================================================
def _write(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically, so a killed run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _digest(keys) -> str:
    """A short, stable cache-file name for a batch of keys."""
    return hashlib.sha1(",".join(map(str, keys)).encode()).hexdigest()[:16]


def _load_store(d: Path, pattern: str, key: str, cols: list[str]) -> pd.DataFrame | None:
    """All cached batches of one kind, or None. Where a key was fetched twice (--refetch) the
    newest file wins. A file without all of `cols` (an older cache layout) is ignored, so its
    keys are fetched again."""
    if not d.exists():
        return None
    files = sorted(d.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    files = [f for f in files if set(cols) <= set(pq.read_schema(f).names)]
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True).drop_duplicates(key)


def fetch_table(sda: SDA, values: list, store: pd.DataFrame | None, key: str,
                sql: str, cols: list[str], prefix: str, cache: Path,
                quote: bool) -> pd.DataFrame:
    """A small per-key table (muaggatt by mukey, sacatalog by areasymbol), fetching the keys
    not in `store`; `quote` puts each key in SQL quotes."""
    have = set() if store is None else set(store[key])
    miss = sorted(v for v in set(values) if v not in have)
    new = []
    for i in range(0, len(miss), ATTR_BATCH):
        chunk = miss[i:i + ATTR_BATCH]
        if quote:
            key_list = ",".join(f"'{v}'" for v in chunk)
        else:
            key_list = ",".join(str(v) for v in chunk)
        df = sda.query(sql.format(keys=key_list), cols)
        if key == "mukey":
            df["mukey"] = df.mukey.astype("int64")
        _write(df, cache / f"{prefix}_{_digest(chunk)}.parquet")
        new.append(df)
    if miss:
        print(f"  {prefix} table: {len(miss)} keys fetched", flush=True)
    parts = new if store is None else [store] + new
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).drop_duplicates(key)
