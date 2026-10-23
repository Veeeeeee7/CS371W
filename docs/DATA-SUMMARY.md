# Data summary: Florida, sampling protocol v2 (built 2026-10-14)

What the project's data is, what each part measures, and what it is used for. Sources and
transforms for each feature: `docs/FEATURE-SOURCES.md`. How it was built and checked:
`docs/DATA-AUDIT-2026-10-14.md`. How to rebuild it: `docs/RUN-FL-V2.md`.

**Status, 2026-10-22.** The table on disk was built on 2026-10-14. The code review of 2026-10-21
fixed bugs in the sampler and in the NHDPlus and Earth Engine covariates. The fixes reach the data
only once `dataset_FL.parquet` is rebuilt (`docs/RUN-FL-V2.md`, "Still to run"); §7 lists what
each changes. Every number in §1–§6 describes the file on disk.

---

## 1. The point table: `data/processed/dataset_FL.parquet`

**80,000 points, 85 columns** (86 once rebuilt: `meta_point_county`, §7). Each point is a location with a binary label: inside
the FEMA 1% annual-chance floodplain (1) or not (0). It belongs to one of two **domains**, the
two mapping procedures the paper compares:

| | positives (label 1) | band negatives (label 0) |
|---|---:|---:|
| Zone AE (detailed study, base flood elevations) | 20,000 | 20,000 |
| Zone A (approximate study) | 20,000 | 20,000 |

- **Positives** are meant to be drawn uniformly by area inside the zone's FEMA polygons (NFHL
  `S_Fld_Haz_Ar`). No polygon may supply more than 2% of a zone's candidates. The points on disk
  are not uniform: the sampler weighted polygons by area twice, so small polygons are almost
  absent (fixed once rebuilt, §7).
- **Band negatives** are drawn inside 5 km buffers of the polygons that supplied that zone's
  positives (a buffer is chosen in proportion to its area, so where buffers overlap the
  sampling is denser). A candidate is kept only if its nearest floodplain polygon of any zone
  is 500 m–5 km away, it lies inside a FIRM map panel, and it lies inside Florida. A negative
  takes the zone of the buffer that produced it. All 40,000 lie inside Florida. 4 Zone AE
  positives lie 6–214 m outside the TIGER state line, where FEMA's polygons extend past it in
  Perdido Bay and on the St. Marys. 2,129 band negatives on disk lie in Zone D or AREA NOT
  INCLUDED, land FEMA did not assess; the fixed sampler keeps candidates out of both (§7).
- **Spacing:** every two points are at least 100 m apart (EPSG:5070).
- **Blocks and tiles:** points sit in 10 km blocks and 50 km tiles; every train/test split is
  made by tile.

**How the experiment uses it.** Models train on Zone AE and are evaluated on Zone A. Seed 0 draws
31 of the 102 tiles for test and keeps the 24 that hold positives of both zones. The other 71
are training tiles (70 contain Zone AE rows); the 7 dropped tiles (1,663 points) are not used:

| set | points | positive share |
|---|---:|---:|
| train: Zone AE, 71 training tiles | 28,109 | 0.555 |
| ID test: Zone AE, test tiles | 10,235 | 0.403 |
| OOD test: Zone A, the same test tiles | 12,771 | 0.476 |

## 2. The 45 features (`data/processed/FEATURES.txt`)

> **Since 2026-10-16 (plan v5) only seven of these are model inputs**: the predictor set of
> Woznicki et al. (2019) minus its two soil variables, listed in `data/processed_w7/FEATURES.txt`:
> `landcover`, `slope`, `tri_300`, `wetindexcat`, `log10_dist_river`, `log10_dist_coast`, `hand`.
> The mapping from their predictors to these columns is in `docs/PROJECT-PLAN.md` §0a. The other
> 38 stay in the table as descriptive variables for the regions; no model sees them. Cells built
> with the seven features live in `data/interim/aotil/cells_w7/`. The rest of this section, and
> §4, describe the 45-feature build of 2026-10-14.

These were the only columns a model could use. **All 45 fed the 300-model zoo**: each model sees a
random fraction of them (`feat_frac`) and a random fraction of the training rows. Below, each
group says what it measures and why it is there. "% missing" is over all 80,000 points.

### Terrain at the point: USGS 3DEP, 10 m (6 features)

Local landform. Floodplain edges in detailed studies follow terrain (they are where a computed
water surface meets the ground), so these let a model learn "low, flat, concave ground floods".
Values on disk were read from a rescaled 10 m grid, so they shift slightly once re-pulled (§7).

| feature | measures | % missing |
|---|---|---:|
| `slope` | ground slope, degrees | 0.0 |
| `aspect_sin`, `aspect_cos` | which way the slope faces (NaN on flat ground, slope < 0.5°, by design) | 61.9 |
| `curvature` | Laplacian of elevation (8 neighbours minus 8 × centre): positive in hollows, negative on crests | 0.0 |
| `tpi_300` | elevation minus the mean of its 300 m cell: a local low or high | 0.0 |
| `tri_300` | roughness: std. dev. of elevation in the 300 m cell | 0.0 |

### Position in the drainage network: MERIT Hydro, 90 m (3)

Physically motivated predictors of flooding: how high a point sits above the nearest drainage
(HAND), and how much land drains through it. On disk, about 42% of points carry the values of a
neighbouring 90 m cell instead of their own; the re-pull reads the cell that contains the point (§7).

| feature | measures | % missing |
|---|---|---:|
| `hand` | height above nearest drainage, m | 8.1 |
| `log10_upa` | upstream drainage area at the point, log10 km² | 8.1 |
| `tpi_broad` | MERIT elevation minus the mean of its 900 m grid cell | 8.1 |

### Land cover and surface water: NLCD 2021 and JRC Global Surface Water v1.4, 30 m (4)

Wetland, open water and development. Florida's approximate Zone A is largely wetland.

| feature | measures | % missing |
|---|---|---:|
| `landcover` | NLCD class code (read as a number by every model; no one-hot) | 0.0 (1 offshore point) |
| `impervious` | % impervious surface | 0.0 (1 offshore point) |
| `gsw_occurrence` | % of satellite observations with water, 1984–2021 | 0.0 |
| `water_frac_900m` | share of the 900 m cell ever covered by water | 0.0 |

### Stream network: NHDPlus V2.1, 1:100k (7)

Distance to and character of the nearest river and the coast, and the point's catchment.

| feature | measures | % missing |
|---|---|---:|
| `log10_dist_river` | distance to the nearest river flowline, log10 m | 0.0 |
| `log10_dist_coast` | distance to the coastline, log10 m (kept apart from rivers); too large for 917 points until rebuilt (§7) | 0.0 |
| `nearest_stream_order` | Strahler order of the nearest reach | 0.0 |
| `log10_upstream_area` | that reach's total drainage area, log10 km² | 0.0 |
| `channel_slope` | that reach's channel slope, m/m (43 points carry NHDPlus's −9998 no-data code; missing once rebuilt, §7) | 0.0 |
| `catch_area_km2` | area of the NHDPlus catchment the point lies in | 9.3 |
| `drainage_density` | flowline length per catchment area, km/km² | 11.3 (9.3 with the fix: 0 in sink catchments, §7) |

### Catchment and watershed context: EPA StreamCat (20)

Areal means over the point's NHDPlus catchment (`…cat`) and its whole upstream watershed
(`…ws`): soils, geology, climate and runoff. These are **regional context**, constant across a
catchment (median 24.7 km² per point), not point measurements. They are the only soil
information the models see, which is why the SSURGO soil-survey lines in §5 count as a "held-out
map" for the boundary analysis.

| features (cat and ws) | measures | % missing |
|---|---|---:|
| `bficat`, `bfiws` | base-flow index: share of streamflow from groundwater, % | 9.3–9.4 |
| `hydrlcondcat`, `hydrlcondws` | lithological hydraulic conductivity, µm/s | 9.3 |
| `wetindexcat`, `wetindexws` | topographic wetness index (×100) | 9.3 |
| `kffactcat`, `kffactws` | soil erodibility (K factor) | 9.3 |
| `clayws`, `claycat`, `sandws`, `sandcat` | soil clay and sand content, % | 9.3–9.8 |
| `permcat`, `permws` | soil permeability, cm/h | 9.3–9.8 |
| `precip8110cat`, `precip8110ws` | mean annual precipitation 1981–2010, mm/yr | 9.3 |
| `tmean8110cat`, `tmean8110ws` | mean annual temperature 1981–2010, °C | 9.3 |
| `runoffcat`, `runoffws` | mean annual runoff, mm/yr | 9.7 |

### Extreme rainfall: NOAA Atlas 14 Volume 9, per 10 km block (5)

The design storms behind the 1% annual-chance label.

| feature | measures | % missing |
|---|---|---:|
| `p100_24hr` | 100-year 24-hour rainfall depth, mm | 0.2 |
| `p100_60min` | 100-year 1-hour depth (flash flooding) | 0.2 |
| `p100_4day` | 100-year 4-day depth (accumulation) | 0.2 |
| `p2_24hr` | 2-year 24-hour depth (ordinary storms) | 0.2 |
| `growth_100_2` | `p100_24hr / p2_24hr`: how extreme-prone a place is relative to its ordinary storms | 0.2 |

The 0.2% (191 points in 35 blocks) could not be fetched: every request failed during the pull,
and NOAA's server answered HTTP 503 to every manual check from 23:12 to 00:14 (`DATA-AUDIT` §3,
§12). 83% of these points are on open water, so some blocks may keep failing even when the
server is back.

**About the missing values.** The 8% gap in the MERIT columns is almost all band negatives on
open water (99%). The 9–10% StreamCat and catchment gap is about 90% such points, and the 11%
`drainage_density` gap about 77%. 34.8% of Zone AE band negatives are open water, against 15.8%
of AE positives, and NHDPlus and MERIT have no values there. So, inside Zone AE, "missing" leans
toward "negative". The hgb and xgb models (75 of 300) see the NaN and can learn that; the other
families see the training median (decision 2 in the audit).

## 3. Columns that are not features (never given to a model)

| columns | what they are | used for |
|---|---|---|
| `point_id` | unique id, `FL0000000…FL0079999` | the join key for every sidecar, cell and region table |
| `domain`, `label`, `neg_protocol` | zone (A / AE), class, `band` | defining the train / ID / OOD sets |
| `lon`, `lat` (NAD83), `geometry` (EPSG:5070) | location | lon/lat, projected to EPSG:3086, build the kNN graph for spatial OODSelect; the 5070 geometry is used for spacing, distances and all GIS joins |
| `block_id`, `tile_30km`, `tile_50km`, `tile_100km` | spatial cells | tile-held-out splits (50 km); spatial cross-validation |
| `county_fips`, `dfirm_id`, `state` | the FEMA map study (DFIRM, usually one county's digital flood map) and its county: `county_fips` is `dfirm_id[:5]`, so for a band negative it is the county of the map whose polygon's ring produced it, not always the county the point lies in | matching each point to its own county's FIRM panels; describing regions (until rebuilt) |
| `meta_point_county` (once rebuilt, §7) | the TIGER county the point lies in | naming the counties of each region |
| `meta_fld_ar_id`, `meta_src_row_id` | FEMA polygon id, and the exact polygon row that labelled a positive (or seeded a negative's ring) | finding the Zone A / AE polygons behind each region (lobes, edges); `meta_edge_dist_m` |
| `meta_fld_zone`, `meta_zone_subty`, `meta_study_typ`, `meta_study_regime`, `meta_source_cit` | how the polygon was mapped. Often unrecorded: 80% of Zone A and 62% of Zone AE positives have STUDY_TYP 'NP' (`NOT_POPULATED`) | describing **why** a region differs: label provenance |
| `meta_static_bfe`, `meta_bfe_km_per_km2` (`meta_depth` is empty in Florida) | static base flood elevation (64% of AE positives, no Zone A positives); BFE-line density of the map study | `meta_bfe_km_per_km2` is the continuous measure of study detail |
| `meta_eff_date`, `meta_panel_scale`, `meta_panel_typ` | the map panel's effective date, scale and type: the point's own DFIRM's panel where it lies in one (2,309 points lie only in neighbouring DFIRMs' panels; 55 dates are null) | map vintage and scale per region |
| `meta_dist_to_sfha_m`, `meta_poly_area_km2`, `meta_neg_rank` | sampling diagnostics | checking the sampling rules |
| `meta_elev`, `meta_aspect_raw`, `meta_log1p_upa`, `meta_comid`, `meta_dist_to_river_m`, `meta_dist_to_coast_m`, `meta_upstream_area_km2`, `meta_nearest_ftype`, `meta_elevcat`, `meta_elevws` | raw values behind the features, and the NHDPlus catchment id | region descriptions (HAND, distance to river); the StreamCat join. Elevation is kept out of the features on purpose |

## 4. What the models and selections produce

| data | where | used for |
|---|---|---|
| **cell** `FL_band_AE_s0`: 300 models' per-point scores on the ID and OOD test sets, with `point_id` for both halves | `data/interim/aotil/cells/FL_band_AE_s0.npz` | O1 (the shift), O2 (aggregate accuracy-on-the-line), input to OODSelect |
| **model table**: family, every hyperparameter, seed, features used, `n_train`, ID/OOD accuracy and AUC, training accuracy and AUC | `…/FL_band_AE_s0_models.csv` | O1 per-model AUC drop (mean 0.020); the label-noise / interpolation check (42 of 300 models reach ≥ 0.99 accuracy on their own training subsample, 16 on all training rows); the Methods model-grid appendix |
| **selections**: OODSelect weights θ per budget S, classic and spatial (μ = 3, 10-NN graph) | `results/2026-10-14/oodselect_paper/{classic,spatial_mu3_knn10}/` | O3 (inverted subsets: classic S = 500 r −0.889, scattered over 23 tiles) and O4 (spatial: r −0.763, 20 components in 12 tiles) |
| **regions**: connected components of the top-S points on the same 10-NN graph; each row carries `cell`, `variant`, `S` and `seed` | `results/2026-10-14/regions/` | the regions O5 analyses. Spatial S = 500 has 8 regions of ≥ 20 points, named by `county_fips`: Hamilton, Miami-Dade, Palm Beach, Dixie, Okeechobee, Indian River, Marion, Nassau–Duval. Some are mostly band negatives (S = 500 Dixie: 6 of 42 positive; S = 250 Palm Beach: 0 of 36) |

## 5. GIS data for the boundary analysis (O5, Contribution 3)

The analysis asks whether the Zone A boundaries inside each region follow **terrain** (as a
detailed study's would) or **pre-existing lines the models never saw**: soil-survey lines,
county lines, map-sheet edges.

| data | where | used for |
|---|---|---|
| **per-point extras**: the FIRM panel each point lies in, its nearest floodplain polygon, and a positive's distance to its polygon's edge | `data/interim/analysis/point_extras_FL.parquet` | map-sheet membership; how deep inside its polygon a positive sits |
| **lobes**: the single-part Zone A polygons of each region; matched unselected Zone A lobes in the same tiles (the comparison); Zone AE lobes in those tiles (the rule contrast) | `results/2026-10-14/boundary/*_lobes.parquet` | the objects whose edges are tested. The target of 3 matched lobes per region lobe is not met in 7 of 12 components |
| **edge vertices** every 30 m along every lobe boundary, with the inward normal and flags for edges shared with another floodplain polygon | `…/*_vertices.parquet` | the unit of measurement (S = 500: 1.9 M vertices) |
| **10 m DEM** (3DEP 1/3″), elevation at each vertex and 100 m inside and outside it | `data/interim/gis/dem/`, `…/*_vertices_z.parquet` | class E: is the edge a level set of the terrain (a water surface)? |
| **SSURGO soil map units** with flooding, ponding, hydric-soil, drainage and hydrologic-group attributes; their boundaries as lines, tagged with which attributes change | `data/interim/gis/ssurgo/` | class S: does the edge follow a line a soil scientist drew? |
| **county lines** (TIGER 2026) and **FIRM panel edges** (per county map) | `data/interim/gis/county_lines_FL.parquet`, `panel_edges_FL.parquet` | class M: does the edge follow an administrative or map-sheet line? |
| **vertex-layer table**: per vertex, the distance to and id of the nearest soil line, shore, survey-area join, county line, state or coast line, own-map panel edge and canal | `results/2026-10-14/boundary/*_vertex_layers.parquet` | the input to the boundary statistic and the displaced-edge null (the boundary statistic ran on these tables on 2026-10-16 as a comparison, `results/2026-10-16/boundary_stat_45feat/`; the main run uses the seven-feature regions, `docs/RUN-W7-REPORT-2026-10-16.md`) |

## 6. Caveats to carry into the paper

- **Atlas 14 gap:** 191 points (0.24%) lack rainfall values; every request failed while NOAA's
  server returned HTTP 503. Refill and rebuild when it is back (audit §12).
- **Open-water negatives:** missing values lean toward the negative class in Zone AE (§2).
- **Class balance of the test halves depends on the split:** seed 0 has positive share 0.403
  (ID) vs 0.476 (OOD), but seeds 1–2 reverse the sign, and over 20 split seeds the gap
  averages about 0 (sd 0.05) in both v1 and v2. 1:1 did not change this. Report O1 in AUC.
- **`channel_slope` no-data:** the −9998 code is kept for 43 points (v1 band subset: 30); it
  becomes missing once rebuilt (§7).
- **Close pairs and MERIT:** 43 of the 1,179 pairs closer than 125 m have identical `hand` and
  `log10_upa` (3.6%, against 2.1% at 125–250 m). At most 27 of them also match on raw upstream
  area, so only a minority can be two points in one ~90 m MERIT cell. Counted on the values on
  disk, which for about 42% of points come from a neighbouring MERIT cell (§7); the step-7 audit
  of `docs/RUN-FL-V2.md` recounts it after the re-pull.
- **State line:** 4 Zone AE positives lie 6–214 m outside the TIGER Florida boundary (FEMA and
  the Census draw the water boundary differently).
- **Matched lobes:** the comparison lobes fall short of 3 per region lobe in 7 of 12 components.
- **Datum:** lon/lat are NAD83, about 1 m from WGS84.
- **The label is a FEMA map, not observed flooding.** Band negatives sit 500 m–5 km from any
  floodplain by construction.

## 7. What the code review of 2026-10-21 changes (applies once `dataset_FL.parquet` is rebuilt)

The file on disk predates these fixes. Each applies to the table that `docs/RUN-FL-V2.md` ("Still
to run") rebuilds. The numbers are the review's: measured on the current points, or, where marked,
in its scratch rebuild of the sampler, which wrote nothing under `data/`.

| change | what it means | size on the current points |
|---|---|---|
| **New column `meta_point_county`** | the GEOID (county code) of the TIGER county polygon the point lies in, from `sample_points.py --state-polygons`; null for a point in no county. `county_fips` is unchanged (the map study's county, §3). `aotil/regions.py`, `region_recurrence.py` and `o5_polygons.py` name a region's counties from `meta_point_county` when the table has it. 86 columns | scratch rebuild: `county_fips` is not the containing county for 12.4% of negatives (4,944 of 40,000); 6 points lie in no county. The current split-3 "Collier" region lies in Collier (64 points), Broward (43) and Hendry (39) |
| **`drainage_density` is 0, not missing, in sink catchments** | an NHDPlus sink catchment (negative id) drains into a closed depression and has no stream of its own, so its stream length per area is 0 | 1,582 points in 159 catchments (Zone A: 1,186 negatives, 224 positives; Zone AE: 159, 13). Missing falls from 11.27% to 9.30%, the rate of `catch_area_km2` |
| **`channel_slope`'s −9998 becomes missing** | NHDPlus's no-data code is no longer kept as a slope | the 43 points of §2 |
| **Coast distances include island coastlines and the full Coastline layer** | `prep_nhdplus_units.py` keeps every NHDPlus Coastline feature, including island and barrier-island shores, which are not part of the river network and were cut with it; `pull_hydro.py` reads the coastline over the points' box padded by 1° instead of 0.05°, so western-panhandle points find their nearest coast | 917 points get a smaller `meta_dist_to_coast_m`, none a larger one. 858 from the missing shores (6 Zone A negatives, 710 Zone AE negatives, 142 Zone AE positives; 641 move by more than 0.3 in log10, mostly in Monroe County, the largest by 57 km); 59 Zone A points in the western panhandle from the padding (0.04–12.6 km). `log10_dist_coast` is one of the seven model features |
| **Positives uniform by area** | one area-proportional draw over all of a zone's polygons, still with the 2% cap; before, polygons were weighted by area twice. Thin, ribbon-shaped polygons now also get their full share | positives from polygons under 1 km²: Zone A 0.6% now, 16.5% in the scratch rebuild (those polygons hold 14.3% of the area); Zone AE 0.6% and 13.3% (13.5%). The kept positives come from 4,300 (A) and 4,329 (AE) polygons instead of 592 and 950, and the band negatives move with their rings |
| **Band negatives exclude Zone D and AREA NOT INCLUDED** | Zone D (flood hazard undetermined) and AREA NOT INCLUDED are unassessed land, so a point there is no evidence of dry ground; band candidates in them are dropped, as candidates outside a FIRM panel already were. Open water stays. The 5 km rings are also drawn rounder: their arcs are approximated to about 24 m instead of 381 m | 2,129 negatives (5.3%) are replaced, mostly in Miami-Dade, Charlotte, Broward, Hardee, Palm Beach and Hendry. The split-3 "Collier" region (78 of its 79 dry points are in these zones) and the split-2 "Collier/Hendry" region may change or disappear |
| **MERIT Hydro and 3DEP sampled in the cell that contains the point** | Earth Engine reads each layer on its own grid instead of resampling it onto a request grid | `hand`, `log10_upa` (`meta_log1p_upa`) and `tpi_broad` change for about 42% of points, which read a MERIT cell about 93 m away. `slope`, `aspect_*`, `curvature`, `tpi_300` and `tri_300` shift slightly: the old 10 m grid moved a point by up to about 6.6 m and inflated slope by about 3%. The error was jitter, not an A-vs-AE bias. Land cover and surface water are unchanged |

Downstream, everything built from the table is stale: the cells, selections, regions, lobes and
boundary tables of §4–§5, and the seven-feature runs, since `data/processed_w7/dataset_FL.parquet`
links to this table. How far the regions, r values and twins numbers move is unknown until the
rerun. Separately, rebuilt cells count a negative predicted at exactly p = 0.5 as correct in the
model table's `acc_*` columns (dt, et and rf models); the AUC columns do not change.
