#!/usr/bin/env python3
"""
pull_atlas14.py -- NOAA Atlas 14 design-storm depths, the real ones.

This replaces the GRIDMET Gumbel proxy in pull_climate.py, which turned out to be
measurably wrong in a way the selftest exposed.

WHY THE PROXY WAS RETIRED
=========================
The proxy's arithmetic was right and its mean annual maximum was accurate. Its
100-year extrapolation was not. Checked against Atlas 14 at the two Vermont test
points:

                        proxy    Atlas 14   error
    mean annual max     46.50      48       -3%     <- fine
    100-yr 24-hr        96.65     123      -21%
    100-yr 24-hr       100.04     144      -31%     (upland point)

The cause is the distribution, not the data. Atlas 14's growth from the 1-year to
the 100-year depth is a factor of 2.56 and 2.72 at those two points; a Gumbel fit
gives 2.08 and 2.11. Gumbel's tail is fixed and too light for Northeastern
convective extremes, and no amount of record length fixes that. Two smaller biases
push the same way: GRIDMET's daily field is a calendar day while Atlas 14's "24-hour"
is a sliding window (worth about 1.13x), and 4 km spatial averaging smooths
convective peaks.

A 25-30% error in the one variable that is supposed to correspond to the 1%-annual-
chance label is not a limitation to note, it is a wrong number. And the proxy was
slow: 102 seconds for three points, because the sliding 5-day maximum evaluates
roughly 361 windows a year across 46 years.

Atlas 14 is directly reachable, covers all ten states, and returns the full
duration-by-return-period matrix in one request. Measured: ~1.4 s per request, 10
points in 2.6 s with 8 workers.

WHAT IS PULLED
==============
    p100_24hr      100-year 24-hour depth, mm. The design storm that corresponds to
                   the 1%-annual-chance flood the labels encode. This is the
                   variable the whole covariate set was missing.
    p100_60min     100-year 1-hour. Flash flooding is driven by short-duration
                   intensity, riverine flooding by accumulation, and Zone A and
                   Zone AE sit differently across that split.
    p100_4day      100-year 4-day, the multi-day accumulation end.
    p2_24hr        2-year 24-hour, the frequent-event depth.
    growth_100_2   p100_24hr / p2_24hr, dimensionless. How extreme-prone a place is
                   RELATIVE to its ordinary storms. Wetness and extremeness are
                   different things and this separates them -- a wet place with a
                   flat growth curve behaves nothing like a dry place with a steep
                   one, and the raw depths alone cannot tell them apart.

Range across the ten states is wide enough to matter: the 100-year 24-hour depth
runs from 87 mm in Arizona to 432 mm in Texas.

GRANULARITY
===========
Requests are made once per 10 km `block_id`, at the mean position of that block's
own points rather than the grid centroid, so every request lands inside the data.
That is 16,397 requests, about 70 minutes, instead of 878,336.

Atlas 14's own grids are 800 m, so this coarsens them roughly 12x. `--validate`
measures what that costs: it pulls a sample of individual points and compares each
to its block value, and reports the error distribution rather than assuming it is
small. Run it before relying on the numbers.

USAGE
=====
    python pull_atlas14.py --selftest       # 2 known points, checked against print
    python pull_atlas14.py                  # all blocks, resumable
    python pull_atlas14.py --status
    python pull_atlas14.py --validate 200   # block value vs true point value
    python pull_atlas14.py --merge          # into data/processed/dataset_{ST}.parquet
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROC = Path("data/processed")
CACHE = Path("data/interim/atlas14")

STATES = ["VT", "CO", "AZ", "MO", "MN", "IA", "LA", "NC", "TX", "FL"]

URL = "https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"
WORKERS = 8                    # measured safe; this is a government server
RETRIES = 4

# The CSV's ARI columns, in order.
ARIS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

# (csv duration label, ARI years) -> output column
WANT = {("24-hr", 100): "p100_24hr",
        ("60-min", 100): "p100_60min",
        ("4-day", 100): "p100_4day",
        ("24-hr", 2): "p2_24hr"}

BANDS = ["p100_24hr", "p100_60min", "p100_4day", "p2_24hr", "growth_100_2"]

SELFTEST = [("winooski bottom", 44.4900, -73.1550, 123),
            ("upland", 44.5203, -72.9851, 144)]


def session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=RETRIES, connect=RETRIES, read=RETRIES, backoff_factor=1.5,
              status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET"]), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=WORKERS * 2))
    return s


def parse(text: str) -> dict:
    """Pull the wanted duration/ARI cells out of the frequency matrix."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line or "," not in line:
            continue
        label, _, rest = line.partition(":")
        label = label.strip()
        if not any(label == d for d, _ in WANT):
            continue
        vals = [v.strip() for v in rest.split(",") if v.strip()]
        if len(vals) < len(ARIS):
            continue
        for (dur, ari), col in WANT.items():
            if dur != label:
                continue
            try:
                out[col] = float(vals[ARIS.index(ari)])
            except (ValueError, IndexError):
                pass
    if out.get("p2_24hr") and out.get("p100_24hr"):
        out["growth_100_2"] = out["p100_24hr"] / out["p2_24hr"]
    return out


def fetch(s: requests.Session, lat: float, lon: float) -> dict:
    r = s.get(URL, params={"lat": f"{lat:.4f}", "lon": f"{lon:.4f}",
                           "type": "pf", "data": "depth",
                           "units": "metric", "series": "pds"}, timeout=90)
    r.raise_for_status()
    if "PRECIPITATION FREQUENCY ESTIMATES" not in r.text:
        # Outside Atlas 14 coverage, or the server returned an error page.
        return {}
    return parse(r.text)


def block_targets() -> pd.DataFrame:
    """One request location per 10 km block: the mean position of its own points.

    A grid centroid could fall in the ocean or outside Atlas 14's project area; the
    mean of the block's actual points cannot.
    """
    frames = []
    for st in STATES:
        p = PROC / f"dataset_{st}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["state", "block_id", "lon", "lat"])
        g = (df.groupby(["state", "block_id"])
             .agg(lon=("lon", "mean"), lat=("lat", "mean"), n=("lon", "size"))
             .reset_index())
        frames.append(g)
    if not frames:
        sys.exit("no datasets in data/processed -- run pull_gee.py --merge first")
    return pd.concat(frames, ignore_index=True)


def cache_path() -> Path:
    return CACHE / "blocks.parquet"


def load_cache() -> pd.DataFrame:
    p = cache_path()
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=["state", "block_id"] + BANDS)


def pull(limit: int | None, workers: int) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    tgt = block_targets()
    have = load_cache()
    done = set(zip(have.state, have.block_id)) if len(have) else set()
    todo = tgt[~tgt.apply(lambda r: (r.state, r.block_id) in done, axis=1)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(tgt):,} blocks total, {len(done):,} cached, {len(todo):,} to fetch")
    if todo.empty:
        return 0

    s = session()
    rows, failed = [], 0
    t0 = time.time()

    def one(r):
        try:
            d = fetch(s, r.lat, r.lon)
            if not d:
                return None
            return {"state": r.state, "block_id": r.block_id, **d}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, r) for r in todo.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            v = f.result()
            if v:
                rows.append(v)
            else:
                failed += 1
            if i % 250 == 0 or i == len(futs):
                el = time.time() - t0
                eta = (len(futs) - i) / max(i / max(el, 1e-9), 1e-9)
                print(f"  {i}/{len(futs)}  ok={len(rows)} fail={failed}  "
                      f"{el/60:.1f} min, ~{eta/60:.1f} min left", flush=True)
            # Checkpoint, so an interrupted run keeps what it fetched.
            if len(rows) and len(rows) % 1000 == 0:
                out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
                out.drop_duplicates(["state", "block_id"]).to_parquet(cache_path(),
                                                                      index=False)

    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out = out.drop_duplicates(["state", "block_id"])
    out.to_parquet(cache_path(), index=False)
    print(f"\ncached {len(out):,} blocks ({failed:,} failed this run)")
    if failed:
        print("rerun the same command -- cached blocks are skipped. Persistent")
        print("failures are usually points outside an Atlas 14 project area.")
    return 0


def status() -> int:
    tgt = block_targets()
    have = load_cache()
    if not len(have):
        print("nothing cached yet")
        return 0
    m = tgt.merge(have[["state", "block_id", "p100_24hr"]],
                  on=["state", "block_id"], how="left")
    t = (m.groupby("state")
         .agg(blocks=("block_id", "size"),
              fetched=("p100_24hr", lambda s: int(s.notna().sum())),
              points=("n", "sum"),
              p100_min=("p100_24hr", "min"), p100_med=("p100_24hr", "median"),
              p100_max=("p100_24hr", "max"))
         .assign(pct=lambda d: (100 * d.fetched / d.blocks).round(1)))
    print(t.to_string())
    print(f"\n{int(t.fetched.sum()):,}/{int(t.blocks.sum()):,} blocks "
          f"({100*t.fetched.sum()/t.blocks.sum():.1f}%)")
    return 0


def validate(n: int, workers: int) -> int:
    """Block value vs the true value at individual points inside that block."""
    have = load_cache()
    if not len(have):
        sys.exit("nothing cached -- run the pull first")
    frames = []
    for st in STATES:
        p = PROC / f"dataset_{st}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=["state", "block_id", "lon", "lat"]).sample(
                    min(n, 20000), random_state=0))
    pts = pd.concat(frames, ignore_index=True).sample(n, random_state=0)
    pts = pts.merge(have[["state", "block_id", "p100_24hr"]],
                    on=["state", "block_id"], how="inner")
    if pts.empty:
        sys.exit("no sampled points fall in a cached block")

    s = session()
    print(f"fetching true values at {len(pts)} individual points...")
    truth = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, s, r.lat, r.lon): r.Index
                for r in pts.itertuples()}
        for f in as_completed(futs):
            i = futs[f]
            try:
                d = f.result()
                truth.append((i, d.get("p100_24hr")))
            except Exception:
                truth.append((i, None))
    tv = pd.Series(dict(truth), name="true_p100")
    pts = pts.join(tv)
    ok = pts.dropna(subset=["true_p100", "p100_24hr"])
    err = ok.p100_24hr - ok.true_p100
    pct = 100 * err / ok.true_p100

    print(f"\n=== block value vs true point value, n={len(ok)} ===")
    print(f"  mean absolute error : {err.abs().mean():.1f} mm "
          f"({pct.abs().mean():.1f}%)")
    print(f"  median absolute     : {err.abs().median():.1f} mm "
          f"({pct.abs().median():.1f}%)")
    print(f"  90th percentile     : {err.abs().quantile(.9):.1f} mm "
          f"({pct.abs().quantile(.9):.1f}%)")
    print(f"  max                 : {err.abs().max():.1f} mm")
    print(f"  bias (signed mean)  : {err.mean():+.1f} mm")
    print(f"  correlation         : {ok.p100_24hr.corr(ok.true_p100):.4f}")
    print("\n  by state (mean abs % error):")
    print("    " + (100 * (ok.p100_24hr - ok.true_p100).abs() / ok.true_p100)
          .groupby(ok.state).mean().round(1).to_string().replace("\n", "\n    "))
    print("\n  Quote these numbers as the spatial-aggregation error of the")
    print("  covariate. If the 90th percentile is under about 10%, block-level")
    print("  resolution is defensible; if not, refetch at tile_50km's finer")
    print("  siblings or per point for the states that fail.")
    return 0


# =========================================================================
# Concurrency guard -- see the same block in pull_hydro.py
# =========================================================================
# Both scripts end by read-modify-writing dataset_{ST}.parquet, so their merges
# must not overlap or the later write drops the earlier one's columns. Fetch
# stages are safe to run side by side; merges take an exclusive lock and write
# atomically.
LOCK = PROC / ".merge.lock"


class MergeLock:
    def __enter__(self):
        PROC.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = (time.time() - LOCK.stat().st_mtime) / 60 if LOCK.exists() else 0.0
            sys.exit(
                f"\nAnother merge holds {LOCK} (age {age:.1f} min).\n"
                f"Merges rewrite the same dataset files, so they must not overlap.\n"
                f"Wait for it to finish. If you are certain nothing is running:\n"
                f"    rm {LOCK}\n")
        os.write(self.fd,
                 f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        LOCK.unlink(missing_ok=True)
        return False


def atomic_to_parquet(df, path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def merge() -> int:
    with MergeLock():
        return _merge_body()


def _merge_body() -> int:
    have = load_cache()
    if not len(have):
        sys.exit("nothing cached -- run the pull first")
    for st in STATES:
        p = PROC / f"dataset_{st}.parquet"
        if not p.exists():
            continue
        import geopandas as gpd
        g = gpd.read_parquet(p)
        g = g.drop(columns=[c for c in BANDS if c in g.columns])
        sub = have[have.state == st][["block_id"] + [b for b in BANDS if b in have]]
        g = g.merge(sub, on="block_id", how="left")
        atomic_to_parquet(g, p)
        cov = {b: round(100 * g[b].notna().mean(), 1) for b in BANDS if b in g}
        print(f"[{st}] {len(g):,} rows, coverage %: {cov}")

    f = PROC / "FEATURES.txt"
    base = [l.strip() for l in f.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    # retire the proxy columns if a previous pull_climate.py run added them
    proxy = {"pr_annmax_mean", "pr_annmax_sd", "pr_100yr_gumbel",
             "pr_annual_mean", "pr_wettest5day"}
    base = [b for b in base if b not in proxy]
    new = [b for b in BANDS if b not in base]
    atomic_write_text(
        "# columns a model may use. everything else in dataset_*.parquet is an\n"
        "# identifier or provenance metadata and must stay out.\n"
        "# split on tile_50km (or tile_100km), never on block_id or point level.\n"
        "# p* are NOAA Atlas 14 depths resolved per 10 km block; StreamCat metrics\n"
        "# are catchment areal means. Both are regional context, not point\n"
        "# measurements -- run the elevation leakage check before trusting a\n"
        "# result that leans on one.\n"
        + "\n".join(base + new) + "\n", f)
    print(f"\nFEATURES.txt: {len(base)+len(new)} features; added {new}")
    if proxy & set(base):
        print(f"retired GRIDMET proxy columns: {sorted(proxy)}")
    print("\nNext: python baselines.py   then   python baselines.py --null")
    return 0


def selftest(workers: int) -> int:
    s = session()
    print(f"{'point':<18}{'p100_24hr':>11}{'expected':>10}{'p100_60min':>12}"
          f"{'p100_4day':>11}{'p2_24hr':>9}{'growth':>8}")
    allok = True
    for name, lat, lon, expect in SELFTEST:
        t0 = time.time()
        d = fetch(s, lat, lon)
        if not d:
            print(f"{name:<18}  FAILED -- no frequency matrix returned")
            allok = False
            continue
        got = d.get("p100_24hr")
        flag = "" if got == expect else f"  <-- expected {expect}"
        print(f"{name:<18}{got:>11.0f}{expect:>10}{d.get('p100_60min',0):>12.0f}"
              f"{d.get('p100_4day',0):>11.0f}{d.get('p2_24hr',0):>9.0f}"
              f"{d.get('growth_100_2',0):>8.2f}{flag}  ({time.time()-t0:.1f}s)")
        if got != expect:
            allok = False
    print("\n  Parsed values match the published Atlas 14 numbers."
          if allok else "\n  MISMATCH -- the CSV layout may have changed; send me the raw output.")
    print("  growth_100_2 near 2.2 for Vermont: the 100-year storm is a bit over")
    print("  twice the 2-year storm. Expect higher in convective regions.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--validate", type=int, metavar="N")
    ap.add_argument("--limit", type=int, help="fetch only N blocks, for a dry run")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(a.workers)
    if a.status:
        return status()
    if a.merge:
        return merge()
    if a.validate:
        return validate(a.validate, a.workers)
    rc = pull(a.limit, a.workers)
    print()
    status()
    print("\nThen:  python pull_atlas14.py --validate 200")
    print("       python pull_atlas14.py --merge")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
