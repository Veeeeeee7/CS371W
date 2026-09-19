#!/usr/bin/env python3
"""
check_ee.py -- prove Earth Engine works end to end, and check the covariates are
real numbers rather than artefacts of our own preprocessing.

HISTORY OF THIS FILE
--------------------
v1  slope/TWI/SPI came back null: ee.Terrain needs the image to carry a
    projection and .mosaic() drops it. Fixed with setDefaultProjection().
v2  every band populated, so the stack looked finished. It was not. Checking the
    numbers rather than the nulls showed TWI was being set by an arbitrary
    constant of mine, at exactly the cells the project is about.

WHY TWI AND SPI ARE GONE
------------------------
TWI = ln(a / tan(beta)) blows up as slope goes to zero, so every implementation
floors tan(beta). v2 used max(tan(beta), 0.001). At the Winooski valley-bottom
test point:

    slope        = 0.035 degrees
    tan(slope)   = 0.000611      <-- the real value
    used         = 0.001000      <-- the floor
    TWI          = 11.098        <-- determined by the floor, not the terrain

The floodplain is where slope is smallest, so the floor binds hardest exactly on
the cells that carry the flood label. A feature whose value is decided by a
preprocessing constant, on the cells that define the positive class, is the
failure Steger et al. (2017) named: the model learns our pipeline, not the
landscape. Sweeping the floor would move the feature and therefore the result.

The second half of the problem is the numerator. MERIT Hydro's upstream area is
90 m, and at the three test points it reads 0.006 / 0.006 / 0.104 km2 -- 0.74,
0.74 and 12.84 cells. A focal maximum over a 3-cell radius changes it by 1x, 3x
and 1x, so this is not channel misalignment that a nudge would fix: these points
genuinely carry near-zero upstream area in MERIT. At that resolution `upa` is
closer to a binary 'am I in a mapped channel' flag than a continuous measure of
contributing area, and it is the input TWI and SPI are both built from.

Earth Engine has no flow-accumulation algorithm for an arbitrary DEM, so a 10 m
TWI from 3DEP is not available either. Rather than ship a feature we would have
to caveat in every result, the wetness signal is split into three honest pieces:

    HAND (hnd)          point-level, from MERIT. Height above nearest drainage is
                        the strongest single predictor in the flood-susceptibility
                        literature and involves no division by a floored slope.
    StreamCat wetindex  catchment-level. The EPA ships a precomputed Composite
                        Topographic Index averaged over each NHDPlusV2 catchment
                        (`WetIndexCat`) and its full upstream watershed
                        (`WetIndexWs`). It is NOT a point TWI -- it is constant
                        across a ~1-3 km2 catchment -- so it enters the model as
                        catchment context and must be named that way in the
                        write-up.
    slope, curvature    point-level terrain, straight from 3DEP at 10 m, with no
                        constants of ours in them.

`upa` is still sampled, but as log1p and labelled a channel-proximity proxy; the
real distance-to-water covariate comes from NHDPlus flowlines.

Usage:
  python check_ee.py [project-id]
"""
from __future__ import annotations
import os, sys, math

try:
    import ee
except ImportError:
    sys.exit("pip install earthengine-api")

# Vermont: a valley bottom, a hillslope, and an upland.
TEST_POINTS = [
    ("winooski bottom", -73.1550, 44.4900),
    ("hillslope",       -73.1805, 44.4812),
    ("upland",          -72.9851, 44.5203),
]

OLD_SLOPE_FLOOR = 0.001   # kept only to reproduce the finding above


def init() -> None:
    project = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("EE_PROJECT")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as e:
        print(f"Initialize failed: {str(e)[:300]}", file=sys.stderr)
        print("Run ee_raw_check.py to see the real error.", file=sys.stderr)
        sys.exit(1)


def build_stack():
    """The covariate stack. Returns (image, native_projection)."""
    coll = ee.ImageCollection("USGS/3DEP/10m_collection")

    # mosaic() produces an image with no fixed projection and ee.Terrain silently
    # returns null on such images, so restore the collection's native projection
    # before computing anything topographic.
    native = coll.first().select("elevation").projection()
    dem = coll.mosaic().select("elevation").setDefaultProjection(native)

    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")

    # Curvature from the 10 m DEM: second derivative of elevation, no constants.
    k = ee.Kernel.laplacian8()
    curvature = dem.convolve(k).rename("curvature")

    merit = ee.Image("MERIT/Hydro/v1_0_1")
    upa = merit.select("upa")                     # upstream area, km2 (90 m)
    hnd = merit.select("hnd").rename("HAND")      # height above nearest drainage, m

    log_upa = upa.add(1).log().rename("log1p_upa")

    nlcd = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
            .filter(ee.Filter.eq("system:index", "2021")).first())

    stack = (dem.rename("elev")
             .addBands(slope).addBands(aspect).addBands(curvature)
             .addBands(hnd).addBands(upa).addBands(log_upa)
             .addBands(nlcd.select(["landcover", "impervious"])))
    return stack, native


def check_assets() -> None:
    print("\n--- asset access ---")
    checks = [
        ("3DEP elevation", lambda: ee.ImageCollection("USGS/3DEP/10m_collection").size().getInfo()),
        ("MERIT Hydro",    lambda: len(ee.Image("MERIT/Hydro/v1_0_1").bandNames().getInfo())),
        ("NLCD 2021",      lambda: ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").size().getInfo()),
        # community catalog ships this as a COLLECTION, not an Image
        ("gNATSGO mukey",  lambda: ee.ImageCollection(
            "projects/sat-io/open-datasets/gNATSGO/raster/mukey").size().getInfo()),
    ]
    for label, fn in checks:
        try:
            print(f"  OK   {label:<18} -> {fn()}")
        except Exception as e:
            note = ("   (not required -- hydrologic soil group comes from the "
                    "SDA REST API)") if "gNATSGO" in label else ""
            print(f"  FAIL {label:<18} {str(e)[:90]}{note}")


def twi_postmortem(rows: list[dict]) -> None:
    """Reproduce the artefact, so the reason TWI was dropped stays checkable."""
    print("\n--- why TWI is not in the stack ---")
    print(f"  {'site':<17}{'slope_deg':>11}{'tan':>11}{'used':>11}"
          f"{'TWI_would_be':>14}  floored")
    for r in rows:
        s = r.get("slope")
        u = r.get("upa")
        if s is None or u is None:
            continue
        tan = math.tan(math.radians(s))
        used = max(tan, OLD_SLOPE_FLOOR)
        a = u * 1e6 / 92.77                       # specific catchment area, m
        twi = math.log(a / used) if a > 0 else float("nan")
        print(f"  {r['site']:<17}{s:>11.3f}{tan:>11.6f}{used:>11.6f}"
              f"{twi:>14.3f}  {'YES <-- value set by the constant' if used > tan else 'no'}")
    print("\n  Wetness is carried instead by HAND (point-level) and StreamCat's")
    print("  WetIndexCat/WetIndexWs (catchment-level, from the StreamCat REST API:")
    print("  https://api.epa.gov/StreamCat/streams/metrics?name=wetindex ).")


def smoke_test() -> None:
    stack, native = build_stack()
    try:
        crs = native.crs().getInfo()
        nominal = native.nominalScale().getInfo()
        print(f"\nDEM projection restored: {crs} @ {nominal:.2f} m")
    except Exception:
        print("\ncould not read DEM projection")

    feats = [ee.Feature(ee.Geometry.Point([lon, lat]), {"site": name})
             for name, lon, lat in TEST_POINTS]

    print("\n--- end-to-end sample (Vermont) ---")
    try:
        res = stack.reduceRegions(
            collection=ee.FeatureCollection(feats),
            reducer=ee.Reducer.first(),
            scale=10,                       # match 3DEP, not 30
        ).getInfo()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
        sys.exit(1)

    cols = ["elev", "slope", "curvature", "HAND", "upa", "log1p_upa",
            "landcover", "impervious"]
    print(f"\n  {'site':<17}" + "".join(f"{c:>12}" for c in cols))
    nulls: dict[str, int] = {}
    rows = []
    for f in res["features"]:
        p = f["properties"]
        rows.append(p)
        row = f"  {p.get('site',''):<17}"
        for c in cols:
            v = p.get(c)
            if v is None:
                nulls[c] = nulls.get(c, 0) + 1
                row += f"{'None':>12}"
            else:
                row += f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}"
        print(row)

    print()
    if nulls:
        print(f"  NULLS by band: {nulls}")
        if "slope" in nulls:
            print("  slope null -> the setDefaultProjection fix did not take.")
    else:
        print("  All bands returned values.")

    # A populated band is not the same as a usable one. MERIT's 90 m cell is
    # 8,100 m2, so report how many cells the upstream area actually amounts to.
    print("\n  upa in MERIT cells (~8,100 m2 each):")
    for r in rows:
        u = r.get("upa")
        if u is not None:
            cells = u * 1e6 / 8100
            flag = "  <-- sub-cell: treat as a channel flag, not an area" if cells < 2 else ""
            print(f"    {r['site']:<17}{cells:>8.2f} cells{flag}")

    twi_postmortem(rows)


def main() -> int:
    init()
    print("Earth Engine initialised")
    check_assets()
    smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
