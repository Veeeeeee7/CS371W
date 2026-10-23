#!/usr/bin/env python3
"""NOAA Atlas 14 design-storm depths per 10 km block, merged into the dataset.

One PFDS request per block_id, at the mean position of the block's own points (a grid
centroid could fall offshore or outside Atlas 14's area). Cached blocks are never fetched
again, so a block keeps the position of the build that first fetched it. Atlas 14's own
grid is 800 m; --validate measures what the block resolution costs. Columns:
    p100_24hr, p100_60min, p100_4day   100-year depth over 24 h, 1 h and 4 days, mm
    p2_24hr                            2-year 24-hour depth, mm
    growth_100_2                       p100_24hr / p2_24hr: extreme vs ordinary storms
Cache: data/interim/atlas14/blocks.parquet. --merge adds the columns to
data/processed/dataset_<ST>.parquet and to FEATURES.txt.

    python pull_atlas14.py FL                    # fetch missing blocks; resumable
    python pull_atlas14.py --status FL
    python pull_atlas14.py --validate 200 FL     # block value vs per-point value
    python pull_atlas14.py --merge FL
"""
import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROC = Path("data/processed")
CACHE = Path("data/interim/atlas14")
# pull_hydro.py takes the same lock: both merges rewrite dataset_<ST>.parquet.
LOCK = PROC / ".merge.lock"

STATES = ["FL"]

URL = "https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"
WORKERS = 8                    # measured safe; this is a government server
RETRIES = 4
NO_MATRIX = "no frequency matrix in the response"

ARIS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]   # the CSV's return-period columns, years

# (CSV duration label, return period in years) -> output column
WANT = {("24-hr", 100): "p100_24hr",
        ("60-min", 100): "p100_60min",
        ("4-day", 100): "p100_4day",
        ("24-hr", 2): "p2_24hr"}

BANDS = ["p100_24hr", "p100_60min", "p100_4day", "p2_24hr", "growth_100_2"]

# Two Vermont points with their published 100-year 24-hour depths, mm.
SELFTEST = [("winooski bottom", 44.4900, -73.1550, 123),
            ("upland", 44.5203, -72.9851, 144)]

FEATURES_HEADER = (
    "# columns a model may use. everything else in dataset_*.parquet is an\n"
    "# identifier or provenance metadata and must stay out.\n"
    "# split on tile_50km (or tile_100km), never on block_id or point level.\n"
    "# p* are NOAA Atlas 14 depths resolved per 10 km block; StreamCat metrics\n"
    "# are catchment areal means. Both are regional context, not point\n"
    "# measurements -- run the elevation leakage check before trusting a\n"
    "# result that leans on one.\n")


# =========================================================================
# requests
# =========================================================================
def session() -> requests.Session:
    """An HTTPS session that retries GETs with backoff on connection errors and 429/5xx."""
    s = requests.Session()
    retry = Retry(total=RETRIES, connect=RETRIES, read=RETRIES, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS * 2))
    return s


def parse(text: str) -> dict:
    """The WANT cells of a PFDS frequency matrix, plus growth_100_2 when both depths
    are there. A matrix row looks like '24-hr:, 50.1,60.2,...' with one cell per ARI."""
    durations = {duration for duration, _ in WANT}
    out = {}
    for line in text.splitlines():
        if ":" not in line or "," not in line:
            continue
        label, _, rest = line.partition(":")
        label = label.strip()
        if label not in durations:
            continue
        cells = [v.strip() for v in rest.split(",") if v.strip()]
        if len(cells) < len(ARIS):
            continue
        for (duration, ari), column in WANT.items():
            if duration != label:
                continue
            try:
                out[column] = float(cells[ARIS.index(ari)])
            except ValueError:
                pass
    if out.get("p2_24hr") and out.get("p100_24hr"):
        out["growth_100_2"] = out["p100_24hr"] / out["p2_24hr"]
    return out


def fetch(http: requests.Session, lat: float, lon: float) -> dict:
    """Atlas 14 depths at one point; {} when the reply has no frequency matrix."""
    params = {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "type": "pf", "data": "depth",
              "units": "metric", "series": "pds"}
    reply = http.get(URL, params=params, timeout=90)
    reply.raise_for_status()
    if "PRECIPITATION FREQUENCY ESTIMATES" not in reply.text:
        # Outside Atlas 14 coverage, or the server returned an error page.
        return {}
    return parse(reply.text)


# =========================================================================
# blocks and the cache
# =========================================================================
def block_targets(states: list[str]) -> pd.DataFrame:
    """One request location per 10 km block: the mean position of its own points."""
    frames = []
    for state in states:
        path = PROC / f"dataset_{state}.parquet"
        if not path.exists():
            continue
        points = pd.read_parquet(path, columns=["state", "block_id", "lon", "lat"])
        blocks = points.groupby(["state", "block_id"]).agg(
            lon=("lon", "mean"), lat=("lat", "mean"), n=("lon", "size"))
        frames.append(blocks.reset_index())
    if not frames:
        sys.exit("no datasets in data/processed -- run pull_gee.py --merge first")
    return pd.concat(frames, ignore_index=True)


def cache_path() -> Path:
    return CACHE / "blocks.parquet"


def load_cache() -> pd.DataFrame:
    path = cache_path()
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["state", "block_id"] + BANDS)


def _save_cache(have: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Write the cache plus the new rows (the first copy of a block wins); return it."""
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out = out.drop_duplicates(["state", "block_id"])
    out.to_parquet(cache_path(), index=False)
    return out


def _fetch_block(http: requests.Session, block) -> dict | str:
    """The block's cache row, or why it failed: the HTTP status, the exception type,
    or NO_MATRIX for a successful reply without the frequency matrix."""
    try:
        depths = fetch(http, block.lat, block.lon)
    except Exception as error:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return f"HTTP {status_code}"
        return type(error).__name__
    if not depths:
        return NO_MATRIX
    return {"state": block.state, "block_id": block.block_id, **depths}


def _print_progress(done: int, total: int, ok: int, failed: int, t0: float) -> None:
    """One progress line: blocks done, ok and failed, minutes elapsed and left."""
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    print(f"  {done}/{total}  ok={ok} fail={failed}  "
          f"{elapsed / 60:.1f} min, ~{eta / 60:.1f} min left", flush=True)


def pull(states: list[str], limit: int | None, workers: int) -> int:
    """Fetch every block that is not cached yet (at most `limit`), in parallel."""
    CACHE.mkdir(parents=True, exist_ok=True)
    targets = block_targets(states)
    have = load_cache()
    done = set(zip(have.state, have.block_id))
    cached = [(state, block) in done for state, block in zip(targets.state, targets.block_id)]
    todo = targets[~np.array(cached, dtype=bool)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(targets):,} blocks total, {len(done):,} cached, {len(todo):,} to fetch")
    if todo.empty:
        return 0

    http = session()
    rows, failed = [], 0
    causes = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for block in todo.itertuples():
            futures.append(pool.submit(_fetch_block, http, block))
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if isinstance(result, dict):
                rows.append(result)
            else:
                failed += 1
                causes[result] += 1
            if i % 250 == 0 or i == len(futures):
                _print_progress(i, len(futures), len(rows), failed, t0)
            # Checkpoint, so an interrupted run keeps what it fetched.
            if len(rows) and len(rows) % 1000 == 0:
                _save_cache(have, rows)

    out = _save_cache(have, rows)
    print(f"\ncached {len(out):,} blocks ({failed:,} failed this run)")
    if failed:
        print("failure causes: " + ", ".join(f"{c} x{n:,}" for c, n in causes.most_common()))
        print("rerun the same command -- cached blocks are skipped.")
        if NO_MATRIX in causes:
            print(f"'{NO_MATRIX}' is usually a point outside an Atlas 14 project area.")
    return 0


def status(states: list[str]) -> int:
    """Per state: blocks, blocks fetched, points and the range of p100_24hr."""
    targets = block_targets(states)
    have = load_cache()
    if not len(have):
        print("nothing cached yet")
        return 0
    blocks = targets.merge(have[["state", "block_id", "p100_24hr"]],
                           on=["state", "block_id"], how="left")
    blocks["fetched"] = blocks.p100_24hr.notna()
    table = blocks.groupby("state").agg(
        blocks=("block_id", "size"), fetched=("fetched", "sum"), points=("n", "sum"),
        p100_min=("p100_24hr", "min"), p100_med=("p100_24hr", "median"),
        p100_max=("p100_24hr", "max"))
    table["pct"] = (100 * table.fetched / table.blocks).round(1)
    print(table.to_string())
    fetched, total = table.fetched.sum(), table.blocks.sum()
    print(f"\n{int(fetched):,}/{int(total):,} blocks ({100 * fetched / total:.1f}%)")
    return 0


# =========================================================================
# validation: block value vs the value at the point itself
# =========================================================================
def _point_values(pts: pd.DataFrame, workers: int) -> pd.Series:
    """p100_24hr fetched at each point of pts, by pts' index (None where it failed)."""
    http = session()
    truth = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for point in pts.itertuples():
            futures[pool.submit(fetch, http, point.lat, point.lon)] = point.Index
        for future in as_completed(futures):
            index = futures[future]
            try:
                truth[index] = future.result().get("p100_24hr")
            except Exception:
                truth[index] = None
    return pd.Series(truth, name="true_p100")


def validate(states: list[str], n: int, workers: int) -> int:
    """Block value vs the true value at n individual points inside cached blocks."""
    have = load_cache()
    if not len(have):
        sys.exit("nothing cached -- run the pull first")
    frames = []
    for state in states:
        path = PROC / f"dataset_{state}.parquet"
        if path.exists():
            points = pd.read_parquet(path, columns=["state", "block_id", "lon", "lat"])
            frames.append(points.sample(n, random_state=0))
    # n per state, then n of those overall.
    pts = pd.concat(frames, ignore_index=True).sample(n, random_state=0)
    pts = pts.merge(have[["state", "block_id", "p100_24hr"]],
                    on=["state", "block_id"], how="inner")
    if pts.empty:
        sys.exit("no sampled points fall in a cached block")

    print(f"fetching true values at {len(pts)} individual points...")
    pts = pts.join(_point_values(pts, workers))
    ok = pts.dropna(subset=["true_p100", "p100_24hr"])
    err = ok.p100_24hr - ok.true_p100
    abs_err = err.abs()
    abs_pct = (100 * err / ok.true_p100).abs()

    print(f"\n=== block value vs true point value, n={len(ok)} ===")
    print(f"  mean absolute error : {abs_err.mean():.1f} mm ({abs_pct.mean():.1f}%)")
    print(f"  median absolute     : {abs_err.median():.1f} mm ({abs_pct.median():.1f}%)")
    print(f"  90th percentile     : {abs_err.quantile(.9):.1f} mm "
          f"({abs_pct.quantile(.9):.1f}%)")
    print(f"  max                 : {abs_err.max():.1f} mm")
    print(f"  bias (signed mean)  : {err.mean():+.1f} mm")
    print(f"  correlation         : {ok.p100_24hr.corr(ok.true_p100):.4f}")
    print("\n  by state (mean abs % error):")
    by_state = abs_pct.groupby(ok.state).mean().round(1)
    print("    " + by_state.to_string().replace("\n", "\n    "))
    print("\n  Quote these numbers as the spatial-aggregation error of the")
    print("  covariate. If the 90th percentile is under about 10%, block-level")
    print("  resolution is defensible; if not, fetch per point instead.")
    return 0


# =========================================================================
# merge into the dataset
# =========================================================================
class MergeLock:
    """Exclusive lock file for merges; released even if the merge raises."""

    def __enter__(self):
        PROC.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = 0.0
            if LOCK.exists():
                age = (time.time() - LOCK.stat().st_mtime) / 60
            sys.exit(
                f"\nAnother merge holds {LOCK} (age {age:.1f} min).\n"
                f"Merges rewrite the same dataset files, so they must not overlap.\n"
                f"Wait for it to finish. If you are certain nothing is running:\n"
                f"    rm {LOCK}\n")
        os.write(self.fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        LOCK.unlink(missing_ok=True)
        return False


def atomic_to_parquet(df, path: Path) -> None:
    """Write to a temp file, then rename, so readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def merge(states: list[str]) -> int:
    with MergeLock():
        return _merge_body(states)


def _add_bands_to_features(path: Path) -> None:
    """Append the Atlas 14 bands that FEATURES.txt does not list yet."""
    features = []
    for line in path.read_text().splitlines():
        # Only a '#' in the first column marks a comment.
        if line.strip() and not line.startswith("#"):
            features.append(line.strip())
    new = [band for band in BANDS if band not in features]
    if not new:
        print("\nFEATURES.txt already lists every Atlas 14 band -- left as is")
        return
    atomic_write_text(FEATURES_HEADER + "\n".join(features + new) + "\n", path)
    print(f"\nFEATURES.txt: {len(features) + len(new)} features; added {new}")


def _merge_body(states: list[str]) -> int:
    """Replace the band columns of each dataset with the cached block values, by block_id."""
    have = load_cache()
    if not len(have):
        sys.exit("nothing cached -- run the pull first")
    for state in states:
        path = PROC / f"dataset_{state}.parquet"
        if not path.exists():
            continue
        dataset = gpd.read_parquet(path)
        dataset = dataset.drop(columns=[c for c in BANDS if c in dataset.columns])
        cached_bands = [b for b in BANDS if b in have.columns]
        blocks = have[have.state == state][["block_id"] + cached_bands]
        dataset = dataset.merge(blocks, on="block_id", how="left")
        atomic_to_parquet(dataset, path)
        coverage = {b: round(100 * dataset[b].notna().mean(), 1)
                    for b in BANDS if b in dataset.columns}
        print(f"[{state}] {len(dataset):,} rows, coverage %: {coverage}")
    _add_bands_to_features(PROC / "FEATURES.txt")
    return 0


# =========================================================================
# self-test and command line
# =========================================================================
def selftest(workers: int) -> int:
    """Fetch the SELFTEST points and compare p100_24hr with the published values.
    workers is unused (two requests, one after the other). Always returns 0."""
    http = session()
    print(f"{'point':<18}{'p100_24hr':>11}{'expected':>10}{'p100_60min':>12}"
          f"{'p100_4day':>11}{'p2_24hr':>9}{'growth':>8}")
    all_ok = True
    for name, lat, lon, expect in SELFTEST:
        t0 = time.time()
        depths = fetch(http, lat, lon)
        if not depths:
            print(f"{name:<18}  FAILED -- no frequency matrix returned")
            all_ok = False
            continue
        got = depths.get("p100_24hr")
        flag = ""
        if got != expect:
            flag = f"  <-- expected {expect}"
            all_ok = False
        print(f"{name:<18}{got:>11.0f}{expect:>10}{depths.get('p100_60min', 0):>12.0f}"
              f"{depths.get('p100_4day', 0):>11.0f}{depths.get('p2_24hr', 0):>9.0f}"
              f"{depths.get('growth_100_2', 0):>8.2f}{flag}  ({time.time() - t0:.1f}s)")
    if all_ok:
        print("\n  Parsed values match the published Atlas 14 numbers.")
    else:
        print("\n  MISMATCH -- the CSV layout may have changed; check the raw reply.")
    print("  growth_100_2 near 2.2 for Vermont: the 100-year storm is a bit over")
    print("  twice the 2-year storm. Expect higher in convective regions.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states", nargs="*", help="default: FL")
    parser.add_argument("--selftest", action="store_true",
                        help="check the parser against two published values")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--validate", type=int, metavar="N")
    parser.add_argument("--limit", type=int, help="fetch only N blocks, for a dry run")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    states = args.states or STATES
    unknown = [s for s in states if s not in STATES]
    if unknown:
        sys.exit(f"unknown states: {unknown}")

    if args.selftest:
        return selftest(args.workers)
    if args.status:
        return status(states)
    if args.merge:
        return merge(states)
    if args.validate:
        return validate(states, args.validate, args.workers)
    rc = pull(states, args.limit, args.workers)
    print()
    status(states)
    print(f"\nThen:  python pull_atlas14.py --validate 200 {' '.join(states)}")
    print(f"       python pull_atlas14.py --merge {' '.join(states)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
