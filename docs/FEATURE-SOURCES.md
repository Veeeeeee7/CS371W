# Feature sources: where each of the 45 covariates comes from

Written 2026-10-14 for `docs/TASK-DATA-AUDIT.md` Part B. Every row was checked against the
code of 2026-10-14; since the 2026-10-21 refactor the code column names functions instead of
line numbers, and notes marked 2026-10-21 record what that day's fixes changed. Every quoted
number was measured read-only on the **v1** Florida table, on the **band subset** the paper
uses (positives + band negatives, 62,578 rows of `dataset_FL.parquet`). The v2 rebuild changes the points, not the sources or
the methods, except for the NHDPlus input (§8). §9 re-runs the quoted numbers on v2.

> **2026-10-16 (plan v5):** the models now use seven of these 45, the Woznicki et al. (2019)
> predictor set (`data/processed_w7/FEATURES.txt`): rows 10 `landcover`, 1 `slope`, 6 `tri_300`,
> 21 `wetindexcat`, 37 `log10_dist_river`, 38 `log10_dist_coast`, 7 `hand` of the table in §3.
> They come from five sources: NLCD, 3DEP, EPA StreamCat, NHDPlus V2.1 and MERIT Hydro.
> The source, transform and missingness of each are as documented below. §7 items 1
> (`channel_slope`) and the Atlas 14 gap no longer touch the models; item 2 (open-water
> missingness) still does, through `hand` and `wetindexcat`; item 7 (`landcover` read as a
> number) still applies.

---

## 1. Summary

- **45 features from 7 sources**: USGS 3DEP 6, MERIT Hydro 3, NLCD 2021 2, JRC Global
  Surface Water 2, NHDPlus V2.1 7, EPA StreamCat 20, NOAA Atlas 14 5. The paper's "45
  covariates from seven public sources" holds. The caveat: StreamCat's 20 columns are
  catchment summaries of six other products (PRISM, STATSGO and four USGS/EPA layers), §4.
- **27 of 45 are regional context**, constant across an NHDPlus catchment, its upstream
  watershed or a 10 km block. 18 are measured at or near the point (§5).
- `FEATURES.txt` contains no `meta_*`, id, lon/lat, tile or elevation column (§6).
- Five things need a fix or a decision (§7): `channel_slope` keeps NHDPlus's −9998 no-value
  code as a number; band negatives on open water make MERIT and NHDPlus missingness a label
  cue; Florida catchments are much larger than the "~1–3 km²" the code comments state; the
  NLCD citation in `DATA-SOURCES.md` is for a different product; the GSW version is not logged.
  (2026-10-21: the first is fixed, `pull_hydro.py` now writes NaN for −9998, and
  `DATA-SOURCES.md` §6 now names the NLCD 2021 release.)

---

## 2. The seven sources

| source | product, version | access | native resolution | n | cite |
|---|---|---|---|---:|---|
| USGS 3DEP | 1/3 arc-second seamless DEM | Earth Engine `USGS/3DEP/10m_collection`, band `elevation`, mosaicked with the native projection restored (`setDefaultProjection`) | 1/3″ (≈10 m) | 6 | USGS 3D Elevation Program; Stoker & Miller 2022, doi:10.3390/rs14040940 |
| MERIT Hydro | v1.0.1 | Earth Engine `MERIT/Hydro/v1_0_1`, bands `hnd`, `upa`, `elv` | 3″ (92.8 m at the equator; ≈82 × 93 m in Florida) | 3 | Yamazaki et al. 2019, *WRR* 55:5053, doi:10.1029/2019WR024873 |
| NLCD | 2021 release (the epoch product, not Annual NLCD) | Earth Engine `USGS/NLCD_RELEASES/2021_REL/NLCD`, image `2021`, bands `landcover`, `impervious` | 30 m | 2 | Dewitz 2023, USGS data release, doi:10.5066/P9JZ7AO3 |
| JRC Global Surface Water | v1.4 (1984–2021); v1.3 only if 1.4 is not visible to the account | Earth Engine `JRC/GSW1_4/GlobalSurfaceWater`, bands `occurrence`, `max_extent` | 30 m | 2 | Pekel et al. 2016, *Nature* 540:418, doi:10.1038/nature20584 |
| NHDPlus V2.1 | medium resolution (1:100k NHD). v1: national seamless geodatabase `_07`. v2: VPU 03N/03S/03W releases (§8) | local files, read by `pull_hydro.py` | flowlines 1:100,000; catchments from a 30 m grid | 7 | McKay et al. 2012, *NHDPlus Version 2: User Guide* |
| EPA StreamCat | REST API, pulled 2026-09-19 (`streamcat_FL.parquet`) | `api.epa.gov/StreamCat/streams/metrics`, `areaOfInterest=cat,ws`, one county FIPS per request | NHDPlus V2.1 catchment (cat), upstream watershed (ws) | 20 | Hill et al. 2016, *JAWRA* 52:120, doi:10.1111/1752-1688.12372 |
| NOAA Atlas 14 | Volume 9 Version 2, Southeastern States (PFDS response header, checked 2026-10-14) | PFDS `hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv`, `type=pf`, `data=depth`, `units=metric`, `series=pds` | 30″ grid (≈0.8 km) | 5 | Perica et al. 2013, NOAA Atlas 14 Vol. 9 (check the title page for the author list) |

---

## 3. The 45 features, in `FEATURES.txt` order

How a value is attached to a point:

- **pixel**: the raster cell containing the point. `pull_gee.py:sample_group` runs
  `reduceRegions` with `ee.Reducer.first()` on the source's own grid (`crs` = the source's
  projection, no `scale`). 2026-10-21: before this fix the pull passed a scale (10 m for 3DEP,
  90 m for MERIT, 30 m for NLCD and GSW), which made Earth Engine rescale the 3DEP and MERIT
  grids: about 42% of points read a neighbouring MERIT cell, 3DEP reads moved by up to about
  6.6 m and slope came out about 3% high. NLCD and GSW were already on their native grid. The
  stored dataset keeps the old values until the Earth Engine pull is rerun.
- **grid window W**: a statistic over the fixed W-sized grid cell that contains the point
  (`reduceResolution` then `reproject`, read back at the point). The window is a grid cell,
  not a window centred on the point, so the point can sit anywhere inside it. W is nominal:
  both grids are geographic, and Earth Engine measures their scale at the equator, so in
  Florida the east–west extent is about 0.86–0.91 W.
- **nearest reach**: distance to, or an attribute of, the nearest NHDPlus flowline of the
  river FTYPEs (StreamRiver, ArtificialPath, Connector, CanalDitch; `pull_hydro.py`
  `RIVER_FTYPES`),
  planar in EPSG:5070 (`STRtree.nearest`). The search is not routed: the nearest reach need
  not be the reach the point drains to.
- **catchment**: the NHDPlus catchment containing the point (`sjoin`, predicate `within`,
  EPSG:5070). Its FEATUREID is `meta_comid`.
- **cat / ws**: StreamCat areal mean over that catchment (cat), or over its whole upstream
  watershed including it (ws), joined on `meta_comid`.
- **block**: one Atlas 14 request per 10 km `block_id`, at the mean lon/lat of the block's
  own points (`pull_atlas14.py:block_targets`); every point in the block gets it.

NaN is passed to the models as NaN: hgb and xgb route it natively, and the other six families
impute the training median (`aotil/zoo.py:build`, a `SimpleImputer(strategy="median")` in each
pipeline). "NaN x%" is the band-subset share.

| # | feature | source (asset / field) | code | native res. | attached as | units | transform, NaN |
|---:|---|---|---|---|---|---|---|
| 1 | `slope` | 3DEP `elevation` → `ee.Terrain.slope` | `pull_gee.py:terrain_group` | 10 m | pixel (10 m) | degrees | none. NaN 0% |
| 2 | `aspect_sin` | 3DEP → `ee.Terrain.aspect` | `terrain_group`; `derive` | 10 m | pixel (10 m) | unitless | sin(aspect); NaN where slope < 0.5° (`FLAT_SLOPE_DEG`). NaN 64.1% |
| 3 | `aspect_cos` | 3DEP → `ee.Terrain.aspect` | as 2 | 10 m | pixel (10 m) | unitless | cos(aspect), same mask. NaN 64.1% |
| 4 | `curvature` | 3DEP → `convolve(ee.Kernel.laplacian8())` | `terrain_group` | 10 m | 3×3 kernel at the pixel | m | sum of the 8 neighbours minus 8 × centre, not divided by cell area; positive in hollows (Spearman −0.31 with `tpi_300`). NaN 0% |
| 5 | `tpi_300` | 3DEP | `terrain_group` | 10 m | pixel minus mean of grid window 300 m | m | none. NaN 0% |
| 6 | `tri_300` | 3DEP | `terrain_group` | 10 m | std. dev. of 10 m elevation in grid window 300 m | m | a standard deviation, not Riley's TRI. NaN 0% |
| 7 | `hand` | MERIT `hnd` | `merit_group` | 3″ | pixel (90 m) | m | none; values on 0.1 m steps, 19.2% exactly 0. NaN where MERIT has no land cell (open water): 5.82% |
| 8 | `log10_upa` | MERIT `upa` | `merit_group`; `derive` | 3″ | pixel (90 m) | log10 km² | EE returns log1p(upa); the merge takes expm1, clips at ≥ 0.0081 km² (one cell, `MERIT_CELL_KM2`), log10. 47.5% sit at the floor (−2.09). NaN as `hand` |
| 9 | `tpi_broad` | MERIT `elv` | `merit_group` | 3″ | pixel minus mean of grid window 900 m (10 × 10 cells) | m | none. NaN as `hand` |
| 10 | `landcover` | NLCD 2021 `landcover` | `nlcd_group` | 30 m | pixel (30 m) | class code | none; 15 codes (11–95) in Florida. A number to every model: `zoo.py` has no one-hot, so lr, mlp and nb read it as ordinal. NaN 0% |
| 11 | `impervious` | NLCD 2021 `impervious` | `nlcd_group` | 30 m | pixel (30 m) | % | none. NaN 0% |
| 12 | `gsw_occurrence` | GSW `occurrence` | `surface_water_group` (asset from `gsw_asset`) | 30 m | pixel (30 m) | % of valid observations with water | `unmask(0)`: never-water and no-data both read 0, so never NaN |
| 13 | `water_frac_900m` | GSW `max_extent` | `surface_water_group` | 30 m | mean of grid window 900 m | fraction 0–1 | `unmask(0)`, then the share of 30 m pixels ever seen as water. Never NaN |
| 14 | `catch_area_km2` | NHDPlus `Catchment.AreaSqKM` | `pull_hydro.py:join_catchments` | catchment | catchment | km² | NHDPlus's published area, no transform. NaN outside every catchment: 7.17% |
| 15 | `nearest_stream_order` | NHDPlus `PlusFlowlineVAA.StreamOrde` | `measure_flowlines` (`REACH_ATTRIBUTES`) | 1:100k | nearest reach | Strahler order | none; 1–8 in Florida (−9 occurs only on Coastline, which is excluded). NaN 0% |
| 16 | `drainage_density` | NHDPlus `LENGTHKM` / `AreaSqKM` | `add_drainage_density` | catchment | catchment | km/km² | length of the one flowline whose COMID is the catchment's, over the catchment's area; 0 for a sink catchment (negative FEATUREID: an area that drains into a closed sink and has no flowline of its own; since 2026-10-21, NaN before). NaN 8.78% on v1. On the whole v2 table the sink catchments were the whole excess over `catch_area_km2` (11.27% against 9.30% NaN; 1,582 points in 159 catchments), so the fix takes the rate to 9.30% |
| 17 | `bfiws` | StreamCat `BFIWs` | `pull_streamcat`; joined in `join_hydro_files` | watershed | ws | % | none. NaN 7.77% |
| 18 | `bficat` | StreamCat `BFICat` | as 17 | catchment | cat | % | none. NaN 7.86% |
| 19 | `hydrlcondws` | StreamCat `HydrlCondWs` | as 17 | watershed | ws | µm/s | none. NaN 7.76% |
| 20 | `hydrlcondcat` | StreamCat `HydrlCondCat` | as 17 | catchment | cat | µm/s | none. NaN 7.78% |
| 21 | `wetindexcat` | StreamCat `WetIndexCat` | as 17 | catchment | cat | unitless (served ×100: 698–2,208) | none. NaN 7.78% |
| 22 | `wetindexws` | StreamCat `WetIndexWs` | as 17 | watershed | ws | unitless (×100) | none. NaN 7.76% |
| 23 | `kffactcat` | StreamCat `KffactCat` | as 17 | catchment | cat | unitless | none. NaN 7.76% |
| 24 | `kffactws` | StreamCat `KffactWs` | as 17 | watershed | ws | unitless | none. NaN 7.76% |
| 25 | `precip8110ws` | StreamCat `Precip8110Ws` | as 17 | watershed | ws | mm/yr | none. NaN 7.76% |
| 26 | `tmean8110ws` | StreamCat `Tmean8110Ws` | as 17 | watershed | ws | °C | none. NaN 7.76% |
| 27 | `tmean8110cat` | StreamCat `Tmean8110Cat` | as 17 | catchment | cat | °C | none. NaN 7.76% |
| 28 | `precip8110cat` | StreamCat `Precip8110Cat` | as 17 | catchment | cat | mm/yr | none. NaN 7.76% |
| 29 | `runoffws` | StreamCat `RunoffWs` | as 17 | watershed | ws | mm/yr | none. NaN 8.10% |
| 30 | `runoffcat` | StreamCat `RunoffCat` | as 17 | catchment | cat | mm/yr | none. NaN 8.15% |
| 31 | `clayws` | StreamCat `ClayWs` | as 17 | watershed | ws | % | none. NaN 7.77% |
| 32 | `sandcat` | StreamCat `SandCat` | as 17 | catchment | cat | % | none. NaN 8.21% |
| 33 | `claycat` | StreamCat `ClayCat` | as 17 | catchment | cat | % | none. NaN 8.21% |
| 34 | `sandws` | StreamCat `SandWs` | as 17 | watershed | ws | % | none. NaN 7.77% |
| 35 | `permcat` | StreamCat `PermCat` | as 17 | catchment | cat | cm/h | none. NaN 8.21% |
| 36 | `permws` | StreamCat `PermWs` | as 17 | watershed | ws | cm/h | none. NaN 7.77% |
| 37 | `log10_dist_river` | NHDPlus flowline geometry | `measure_flowlines`; `quarantine_and_log_scale` | 1:100k | distance to nearest reach | log10 m | clip at ≥ 1 m (50 points), log10; raw value in `meta_dist_to_river_m`. NaN 0% |
| 38 | `log10_dist_coast` | NHDPlus Coastline flowlines | `measure_flowlines` (`read_coastline`); `quarantine_and_log_scale` | 1:100k | distance to nearest Coastline flowline (since 2026-10-21 island shorelines are kept and the search box is wider, §8) | log10 m | clip at ≥ 1 m (2 points), log10; raw in `meta_dist_to_coast_m`. NaN 0% |
| 39 | `log10_upstream_area` | NHDPlus `PlusFlowlineVAA.TotDASqKM` | `measure_flowlines` (`REACH_ATTRIBUTES`); `quarantine_and_log_scale` | 1:100k | nearest reach | log10 km² | clip at ≥ 0.01 km² (170 points at −2, 69 of them raw 0), log10; raw in `meta_upstream_area_km2`. NaN 0% |
| 40 | `p100_24hr` | Atlas 14, 100-yr 24-hr depth | `pull_atlas14.py:parse` (`WANT`); merged by `pull_atlas14.py:_merge_body` | 30″ | block | mm | none; whole mm as served. NaN where the block's request failed: 1 point (block 147_40) on v1. On v2, 191 points in 35 blocks (7 A negatives, 176 AE negatives, 8 AE positives) after the 2026-10-14 PFDS outage; the refill is still pending (`DATA-AUDIT-2026-10-14.md` §3 and §12) |
| 41 | `p100_60min` | Atlas 14, 100-yr 60-min depth | as 40 | 30″ | block | mm | as 40 |
| 42 | `p100_4day` | Atlas 14, 100-yr 4-day depth | as 40 | 30″ | block | mm | as 40 |
| 43 | `p2_24hr` | Atlas 14, 2-yr 24-hr depth | as 40 | 30″ | block | mm | as 40 |
| 44 | `growth_100_2` | Atlas 14, derived from 40 and 43 | `parse` | 30″ | block | ratio | `p100_24hr / p2_24hr` of the whole-mm values. As 40 |
| 45 | `channel_slope` | NHDPlus `elevslope.SLOPE` | `measure_flowlines` (`REACH_ATTRIBUTES`, `SLOPE_NO_DATA`) | 1:100k | nearest reach | m/m | v1: **none, NHDPlus's −9998 no-value code passed through as a number** (30 points). Since 2026-10-21 −9998 becomes NaN (§7 item 1). 1e-5 is NHDPlus's smallest assigned slope; 36.7% of points sit at it. NaN 0% on v1 |

Atlas 14 in numbers: 1,683 blocks hold the band subset, median 32 points each, and
`p100_24hr` takes 194 distinct values.

---

## 4. "Seven public sources": does the sentence hold?

**Yes, counted as the datasets the pipeline reads.** 6 + 3 + 2 + 2 + 7 + 20 + 5 = 45. The
expected seven are exactly the seven in §2. `log10_dist_coast` comes from NHDPlus Coastline
flowlines, not from an eighth source. `pull_climate.py` (the retired GRIDMET proxy, removed
2026-10-21) and
`pull_extras.py` (NFHL BFE and LOMR lines; `sample_points.py:bfe_density_map` turns the BFE
lines into `meta_bfe_km_per_km2`) produce none of the 45.

**StreamCat is a distributor, not a measurement.** Its 20 columns are catchment and watershed
means of six other products. From StreamCat's own metric dictionary
(`api.epa.gov/StreamCat/streams/variable_info`, read 2026-10-14):

| StreamCat metric | underlying product | features |
|---|---|---:|
| `Precip8110`, `Tmean8110` | PRISM 1981–2010 climate normals | 4 |
| `Clay`, `Sand`, `Perm` | STATSGO soils (Carlisle et al. 2009 summaries) | 6 |
| `Kffact` | STATSGO K-factor raster (Penn State Soil Information) | 2 |
| `BFI` | USGS base-flow index grid (Wolock 2003, OFR 03-263) | 2 |
| `Runoff` | McCabe & Wolock 2011 water-balance runoff | 2 |
| `WetIndex` | EnviroAtlas composite topographic index | 2 |
| `HydrlCond` | USGS lithological hydraulic conductivity (StreamCat table `GeoChemPhys3`) | 2 |

Counting those separately gives 12 primary products (6 direct sources + 6 behind StreamCat,
STATSGO counted once). Seven is the honest count of what we pulled. The paper should not imply
seven independent measurements. Suggested wording:

> We attach 45 covariates from seven public sources: terrain from the USGS 3DEP 10 m DEM and
> MERIT Hydro, land cover from NLCD 2021, surface water from JRC Global Surface Water, the river
> network and catchments from NHDPlus V2.1, climate, soil, runoff, base-flow and geology
> summaries from EPA StreamCat (itself derived from PRISM, STATSGO and USGS layers), and
> design-storm depths from NOAA Atlas 14. Twenty-seven of the 45 are regional context, constant
> across an NHDPlus catchment, its upstream watershed or a 10 km block, rather than point
> measurements.

---

## 5. Point measurements and regional context

| attached as | features | n |
|---|---|---:|
| pixel at the point | slope, aspect_sin, aspect_cos, curvature, hand, log10_upa, landcover, impervious, gsw_occurrence | 9 |
| grid window | tpi_300, tri_300 (300 m); tpi_broad, water_frac_900m (900 m) | 4 |
| distance to the nearest NHDPlus line | log10_dist_river, log10_dist_coast | 2 |
| attribute of the nearest reach | nearest_stream_order, log10_upstream_area, channel_slope | 3 |
| NHDPlus catchment | catch_area_km2, drainage_density, the 10 StreamCat `cat` | 12 |
| upstream watershed | the 10 StreamCat `ws` | 10 |
| 10 km block | the 5 Atlas 14 columns | 5 |

The last three rows are the **27 regional-context features**. The three reach attributes sit
in between: each is constant over the area for which that reach is the nearest one.

**Florida catchments are large.** The `FEATURES.txt` header (written from `pull_hydro.py`'s
`FEATURES_HEADER`) calls StreamCat values "constant across a ~1–3 km² catchment". In the band
subset the median distinct catchment is 3.7 km², but points are drawn in proportion to area, so they land
in big catchments: the median **per point** is 24.1 km² (quartiles 6.8 and 92.2 km²). Three
catchments over 1,000 km² hold 3,668 points (5.9%): COMID 89748 (7,984 km², 1,993 points in
six counties, mostly 12099, 12086 and 12051), 16770632 (1,893 km², 945) and 16770182
(1,155 km², 730). All 1,993 points in COMID 89748 share every `cat` value, `catch_area_km2`
and `drainage_density`.
Do not carry "~1–3 km²" into the paper; quote the per-point numbers. `FEATURES.txt` keeps its
comment (it is never edited).

---

## 6. Columns that are not features

**Derived during the covariate pull, quarantined with `meta_`:**

| column | made by | what it is | why not a feature |
|---|---|---|---|
| `meta_elev` | `pull_gee.py:derive` | 3DEP 10 m elevation, m | identifies the state at 58.3% against a 10% baseline (the ten-state covariate audit of 2026-09-19, §2.1; archived in `Claude outputs/removed-2026-10-21/docs/COVARIATE-AUDIT.md`) |
| `meta_aspect_raw` | `derive` | aspect, degrees, unmasked | replaced by the masked sin/cos |
| `meta_log1p_upa` | `derive` | MERIT log1p(upa) as pulled | replaced by `log10_upa` |
| `meta_comid` | `pull_hydro.py:join_hydro_files` | FEATUREID of the point's catchment | an identifier |
| `meta_dist_to_river_m`, `meta_dist_to_coast_m`, `meta_upstream_area_km2` | `quarantine_and_log_scale` | raw values behind the three log10 features | kept for diagnostics |
| `meta_nearest_ftype` | `measure_flowlines` | FTYPE of the nearest reach | provenance of the distance |
| `meta_elevcat`, `meta_elevws` | `quarantine_and_log_scale` (`STREAMCAT_ELEVATION`) | StreamCat catchment / watershed mean elevation (from NHDPlusV2) | same fingerprint risk as `meta_elev` |

**From sampling (`sample_points.py`), describing how FEMA made the map:** `meta_fld_zone`,
`meta_zone_subty`, `meta_study_typ`, `meta_study_regime`, `meta_static_bfe`, `meta_depth`,
`meta_source_cit`, `meta_fld_ar_id`, `meta_poly_area_km2`, `meta_dist_to_sfha_m`,
`meta_eff_date`, `meta_panel_scale`, `meta_panel_typ`, `meta_bfe_km_per_km2`; v2 adds
`meta_src_row_id` and `meta_neg_rank`. Since 2026-10-21 the sampler also writes
`meta_point_county` (the county polygon the point lies in, from `--state-polygons`); it reaches
the dataset when the points are rebuilt.

**Identifiers, labels and split columns:** `point_id`, `state`, `county_fips`, `dfirm_id`,
`block_id`, `domain`, `label`, `neg_protocol`, `lon`, `lat`, `geometry`, `tile_30km`,
`tile_50km`, `tile_100km`.

None of these is in `FEATURES.txt` (checked against its 45 lines).

---

## 7. Discrepancies found (v1, read-only checks)

1. **`channel_slope` keeps −9998 as a value.** NHDPlus marks reaches with no computed slope
   −9998; `pull_hydro.py` copies it unchanged, so 30 band-subset points (61 in the whole table:
   50 nearest an ArtificialPath, 8 StreamRiver, 2 Connector, 1 CanalDitch) read a slope of
   −9998 m/m where every real value lies between 1e-5 and 0.073. Tree models only see a low
   threshold; lr and mlp standardise it, so those 30 points set the column's variance and
   compress every real slope towards zero, and nb fits a Gaussian through it. Across the three
   Florida VPUs 3,998 of 323,091 river-type network flowlines (1.2%) carry the code.
   `prep_nhdplus_units.py` keeps it on purpose so v1 and v2 match. **Decision:** map −9998 to
   NaN in `pull_hydro.py --merge` (one line; v1 is then not reproducible exactly), or keep it and
   say so in Methods. Separately, 36.7% of points sit at NHDPlus's 1e-5 floor.
   **Done 2026-10-21:** `pull_hydro.py` maps −9998 (`SLOPE_NO_DATA`) to NaN in
   `measure_flowlines`, when it writes `river_FL.parquet` (the `--flowlines` stage, not
   `--merge`). The geodatabase still holds −9998. The stored dataset changes once
   `river_FL.parquet` is rebuilt and merged again. `channel_slope` is not a W7 feature.
2. **Band negatives on open water make missingness a label cue.** In the band subset (% of each
   stratum):

   | domain, label | `hand` NaN | `catch_area_km2` NaN | NLCD open water |
   |---|---:|---:|---:|
   | A, negative | 2.18 | 3.28 | 5.10 |
   | A, positive | 0.00 | 0.00 | 3.05 |
   | AE, negative | **29.83** | **32.14** | **35.33** |
   | AE, positive | 0.09 | 2.37 | 16.15 |

   These points sit in bays, estuaries and lakes inside FIRM panels (median `meta_elev` 0 m;
   top counties Monroe, Miami-Dade, Lee). MERIT has no land cell there, and NHDPlus has no
   catchment, so `hand`, `log10_upa`, `tpi_broad`, `catch_area_km2`, `drainage_density` and
   all 20 StreamCat columns go NaN together, almost only for negatives, and mostly for AE
   negatives, which are in the training domain. hgb and xgb can split on the NaN itself; the
   others impute the median. This is a sampling property, not a provenance error, and v2 keeps
   the same band rule (`PROJECT-PLAN.md` §2a), so expect it again. **Decision for Victor:** drop
   NLCD-water negatives (a sampling change not listed in §2a), or keep them and report the
   missingness by stratum.
3. **Catchment size.** "~1–3 km²" is wrong for Florida points; see §5.
4. **MERIT cells against 100 m spacing.** A MERIT 3″ cell is about 82 × 93 m in Florida, so
   two points 100 m apart can share one. v1 (200 m within a class, 100 m across classes) has 54
   band-subset pairs under 125 m, and 3 of them share both `hand` and `log10_upa`. v2 puts every
   pair at ≥ 100 m, so expect more; Part A4 counts them (the §9 check prints it).
5. **"Drains to" is "nearest".** A comment in `pull_hydro.py` called the reach attributes
   those of "the reach a point drains to". The code takes the nearest river-type flowline by
   planar distance. Methods should say nearest. (The comment above `REACH_ATTRIBUTES` now says
   "nearest river reach (not necessarily the reach the point drains to)".)
6. **`log10_upa` floor.** 0.0081 km² is a 90 m square; a Florida MERIT cell is about
   0.0074–0.0078 km², so the clip also lifts single-cell values slightly. 47.5% of points sit at the floor, which is
   the "channel proximity, not contributing area" reading in `COVARIATES.md` §4.
7. **`landcover` is a code read as a number** by all eight families (§3 row 10). A model-side
   choice, recorded here because Methods will describe the feature.
8. **GSW version is not logged.** `gsw_asset()` takes GSW 1.4 and falls back to 1.3 silently;
   nothing records which one the v1 pull used. Check once in Earth Engine with
   `ee.Image("JRC/GSW1_4/GlobalSurfaceWater").bandNames().getInfo()` under the pulling account.
9. **NLCD citation.** `DATA-SOURCES.md` §6 says to cite Fleckenstein et al. 2026 (Annual NLCD),
   but `pull_gee.py` reads the NLCD 2021 epoch release. Cite Dewitz 2023
   (doi:10.5066/P9JZ7AO3). (`DATA-SOURCES.md` §6 says so since 2026-10-21.)
10. **StreamCat gaps beyond missing catchments.** 377 band-subset points have a COMID but no
    StreamCat row: 130 COMIDs, 121 of them absent from the county responses and 9 served with
    null values. Some metrics have more nulls (with a COMID: `runoffcat` 616, `sandcat` 653,
    `bficat` 433 points).
11. **Atlas 14 positions.** v1 fetched each block at the mean of all its points, shadedX
    included. `archive_sampling_v1.sh` copies `blocks.parquet` rather than moving it, and the
    pull skips cached `(state, block_id)` keys, so a v2 block that existed in v1 keeps the
    value fetched at the v1 position; only new blocks are fetched at the v2 mean. Both positions
    lie inside the same 10 km block. `pull_atlas14.py --validate 200 FL` measures the
    within-block error.
12. **lon/lat datum.** `sample_points.py` writes lon/lat in NAD83 (EPSG:4269); Earth Engine and
    PFDS read them as WGS84. The difference is about 1 m, well under a 10 m pixel.
13. **StreamCat's own unit fields** list `HydrlCond` and `Perm` as "Percent"; the metric
    descriptions say µm/s and cm/h. §3 uses the descriptions.

---

## 8. What v2 changes

- **NHDPlus input.** v1 read `Catchment` and `NHDFlowline_Network` from the national seamless
  geodatabase. v2 downloads only VPUs 03N, 03S and 03W (`fetch_nhdplus_fl.sh`, 547 MB, archives
  NHDSnapshot `_04`/`_06`, NHDPlusAttributes `_07`, NHDPlusCatchment `_01`), and
  `prep_nhdplus_units.py` rebuilds the same two layers, with the national names and fields, in
  `data/raw/nhdplus/NHDPlusV21_FL.gdb`. `pull_hydro.py` reads that file unchanged.
- **Network flowlines only.** The national `NHDFlowline_Network` holds the flowlines with a VAA
  record; the snapshot shapefiles also hold the non-network ones. Across the three units that
  is 86,199 extra lines, 76,905 of them CanalDitch (62,411 in 03S, the south Florida canal
  grid). `prep_nhdplus_units.py` cuts to the VAA COMIDs, which reproduces v1's network. Checked
  on the extracted files: every river-type network flowline has an `elevslope` row, so v2
  introduces no new NaN in `channel_slope`; the only network flowlines without one are the
  7,326 Coastline features, which never feed a slope.
- **Coastline (2026-10-21).** Island and barrier-island shorelines are non-network lines, so
  the VAA cut above dropped them, and v1's national layer lacked them too: points on the Keys
  and barrier islands read their distance to a farther coastline (one Boca Chica point read
  34,679 m instead of 3.9 m). `prep_nhdplus_units.py` now keeps every Coastline
  line whether or not it has a VAA record, and `pull_hydro.py` (`read_coastline`) searches for
  the coast 1° (`COAST_PAD_DEG`) beyond the points' box instead of the rivers' 0.05°, because
  western-panhandle points are nearest a coast outside that box. Checked on a scratch rebuild:
  917 points get a smaller `meta_dist_to_coast_m` and none a larger one. 858 come from the island
  shorelines (6 A negatives, 710 AE negatives, 142 AE positives; up to 57 km, 641 of them by more
  than 0.3 in log10, all AE and mostly in Monroe County), 59 western-panhandle Zone A points from
  the wider box (up to 12.6 km). The river attributes do not change. The stored dataset changes
  once the geodatabase, `river_FL.parquet` and the merge are rebuilt (`docs/RUN-FL-V2.md`
  step 5); `log10_dist_coast` is a W7 feature, so the W7 models and everything built on them
  change on the rerun.
- **COMIDs are NHDPlus V2.1 in both**, so the cached `streamcat_FL.parquet` (left in place by
  `archive_sampling_v1.sh`) still joins. A v2 point in a county that v1 never queried would get
  no StreamCat row; the §9 check counts them.
- Earth Engine assets, the StreamCat API and the Atlas 14 endpoint are unchanged.
- **Open for v2 when this was written, both decided since:** whether `channel_slope` keeps
  −9998 (§7.1; NaN since 2026-10-21), and whether band negatives that fall outside Florida stay
  in (1,038 of 22,652 v1 band negatives, 4.6%, lie in Alabama or Georgia by the Census county
  polygons; `sample_points.py --state-polygons` is off by default; `audit_dataset.py` B7 counts
  them). The second was decided on 2026-10-14 (decision 1(b) in `DATA-AUDIT-2026-10-14.md`):
  v2 is built with `--state-polygons` and the 67 TIGER/Line 2026 Florida counties, and has 0
  band negatives outside Florida.

---

## 9. Re-check on v2

Prints the numbers this document quotes. Read-only, a few seconds. Run it on the v2 table after
the Part A merges, then update §3, §5 and §7:

```bash
.venv/bin/python - data/processed/dataset_FL.parquet <<'EOF'
import sys, numpy as np, geopandas as gpd
from scipy.spatial import cKDTree
d = gpd.read_parquet(sys.argv[1])
b = d[(d.neg_protocol == "band") | (d.label == 1)]
print(f"rows {len(d):,}, band subset {len(b):,}")
cs = b.channel_slope
print(f"channel_slope -9998: {int((cs == -9998).sum())}, at 1e-5: {np.isclose(cs, 1e-5).mean():.1%}")
print(f"log10_upa at the 0.0081 km2 floor: {np.isclose(b.log10_upa, np.log10(0.0081)).mean():.1%}")
print("NaN hand / catch_area / open water (NLCD 11), % by domain x label:")
t = b.assign(hand_na=b.hand.isna(), catch_na=b.catch_area_km2.isna(), water=b.landcover == 11)
print((100 * t.groupby(["domain", "label"])[["hand_na", "catch_na", "water"]].mean()).round(2).to_string())
c = b.catch_area_km2
print(f"catchment km2 per point: median {c.median():.1f}; per distinct COMID: "
      f"{b.drop_duplicates('meta_comid').catch_area_km2.median():.1f}; "
      f"points in catchments > 1000 km2: {int((c > 1000).sum()):,}")
xy = np.c_[b.geometry.x, b.geometry.y]
pr = cKDTree(xy).query_pairs(125.0, output_type="ndarray")
h, u = b.hand.to_numpy(), b.log10_upa.to_numpy()
same = (h[pr[:, 0]] == h[pr[:, 1]]) & (u[pr[:, 0]] == u[pr[:, 1]])
print(f"pairs < 125 m: {len(pr):,}, sharing hand and log10_upa: {int(same.sum()):,}")
print(f"Atlas 14 NaN rows: {int(b.p100_24hr.isna().sum())}; "
      f"COMID but no StreamCat: {int((b.meta_comid.notna() & b.bfiws.isna()).sum())}")
EOF
```

After the 2026-10-21 `channel_slope` fix the −9998 count is 0 on a rebuilt table; those
reaches read NaN. On v1, pass
`data/interim/archive/sampling-v1/data/processed/dataset_FL.parquet` once it has been
archived. v1 output (2026-10-14): channel_slope −9998 30, at 1e-5 36.7%; `log10_upa` at the
floor 47.5%; the table in §7.2; catchment median 24.1 / 3.7 km², 3,668 points over 1,000 km²;
54 pairs under 125 m, 3 sharing; Atlas 14 NaN 1; COMID but no StreamCat 377.
