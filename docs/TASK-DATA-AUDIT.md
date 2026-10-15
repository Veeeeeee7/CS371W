# Task for Claude Code: rebuild the Florida sample, audit it, and collect what is missing

Written 2026-10-13, revised the same evening for sampling protocol v2 (`docs/PROJECT-PLAN.md`
§2a). Run from the repo root (`CS371W/`). Work through Parts A → E in order.

**Goal.**
1. Rebuild the Florida point sample with sampling protocol v2 (100 m spacing, exactly 1:1 band
   negatives, exactly 20,000 positives per zone) and re-pull its covariates.
2. Confirm that every piece of data the paper's experiments need is collected, stored and
   correct on the new sample. The paper (`CS371_Cho_Li/acl_latex.tex`) commits to five
   anticipated outcomes and three contributions; `docs/PROJECT-PLAN.md` v4 maps them to
   experiments and data, and **its §4 is your checklist**.
3. Re-run the seed-0 models and selections on the new sample, extract the regions, and collect
   the GIS layers the boundary analysis needs. Where something is missing, write the code that
   collects it and run it.

---

## Read first

1. `docs/PROJECT-PLAN.md` v4: **§2a (sampling protocol v2: exactly what changes and what stays
   the same)**, then §2–§4.
2. `CS371_Cho_Li/acl_latex.tex`: the Introduction and Related Work. Its numbers (98,578 points,
   45 covariates, seven public sources, 300 models, eight families, train on Zone AE and evaluate
   on Zone A) describe the **old (v1) sample**. You will report the new numbers.
3. `docs/REGION-ANALYSIS-FLORIDA.md` §3.3: the boundary analysis whose inputs you collect. The
   regions in its §2 come from the v1 sample and will change.
4. The pipeline: `sample_points.py` (all of it, especially the docstring notes and `DEFAULTS`),
   `subset_nfhl.py`, `pull_gee.py`, `pull_hydro.py`, `pull_atlas14.py`, `pull_climate.py`,
   `pull_extras.py`, and `docs/COVARIATES.md`, `docs/COVARIATE-AUDIT.md`,
   `docs/SAMPLING-RESULTS.md`, `docs/DATA-SOURCES.md`, `docs/DOWNLOAD-GUIDE.md`,
   `docs/ASSEMBLY-SHORTCUTS.md`.
5. The models: `aotil/README.md`, `aotil/zoo.py`, `aotil/cells.py`, `aotil/graph.py`,
   `aotil/oodselect_paper.py`, `aotil/report_paper.py`, `run_oodselect_local.sh`.

---

## Rules

- **Florida only.** Do not rebuild or rewrite any other state's files. If a script's `--merge`
  step rewrites every state, add a state argument. Record checksums of the other nine
  `data/processed/dataset_*.parquet` before you start and confirm they are unchanged at the end.
- **Archive, don't delete.** Before regenerating anything, **move** (not copy: the disk is 95%
  full, about 26 GB free) every Florida file the rebuild will overwrite into
  `data/interim/archive/sampling-v1/`, keeping relative paths. At least:
  `data/interim/points/points_FL.parquet`, the Florida shards in `data/interim/gee/FL/`,
  `data/interim/hydro/comid_FL.parquet`, `data/processed/dataset_FL.parquet`, and
  `data/interim/aotil/cells/FL_*`. Write the list into the audit report. `results/2026-09-*` stay
  where they are; they are now the v1 results.
- **Nothing you add is a model feature.** `FEATURES.txt` keeps exactly the same 45 columns.
  Per-point columns you add get the `meta_` prefix. Soil lines, county lines and map-sheet edges
  are the "held-out maps" the paper tests against; if a model could see them, the test would mean
  nothing.
- **Scope: cell `FL_band_AE`** (train on Zone AE, evaluate on Zone A, band negatives). Nothing
  for Iowa/Missouri, shadedX or the A→AE direction.
- **Large files.** `data/interim/prep/haz_FL_5070.parquet` is 5.4 GB. Read it with a bbox,
  row-group or column filter (check whether `data/interim/prep/idx_FL.parquet` already indexes
  it). Never load it whole.
- **Environment.** Use `.venv/bin/python` (see `run_oodselect_local.sh`). Add any new dependency
  to `requirements.txt`.
- **Code style.** Delete superseded code instead of commenting it out; git keeps history. Every
  new script gets argparse, caching (skip when the output exists, `--overwrite` to redo), and a
  smoke test on a small piece before the full run.
- **Stop and ask Victor before:**
  - any download over 2 GB, or anything that would leave less than 10 GB free on the disk;
  - changing any file outside Florida, or any sampling setting not listed in PLAN §2a;
  - installing something that is not pip-installable into `.venv`;
  - anything that needs Earth Engine if `ee.Initialize()` fails (authentication is his to do);
  - continuing past Part A if the rebuild cannot reach exactly 20,000 positives and 20,000 band
    negatives per zone.
- **No git writes.** Do not run `git add`, `git commit`, `git push`, `git stash`, `git checkout`,
  `git reset`, `git mv` or anything else that changes the repository's git state. Move files
  with plain `mv`. Read-only git (`git status`, `git diff`, `git log`) is fine. Leave every
  change uncommitted: Victor reviews all the code and commits it himself.

---

## Part A: rebuild the Florida sample (protocol v2)

### A0. Record v1 and archive

Record the v1 Florida counts by `domain × label × neg_protocol`, the checksums of all ten
`dataset_*.parquet`, and the free disk space. Then move the Florida files into the archive as
described in Rules.

### A1. Change `sample_points.py`

- `min_sep_m`: 200 → **100**.
- The whole-state cross-stratum separation pass uses the full `min_sep_m` (100 m) instead of
  half of it, so **every point is at least 100 m from every other point**, whatever its class or
  zone.
- `neg_ratio`: 2 → **1**. Band negatives must end at **exactly** the number of positives in each
  zone. Trim any surplus with `neg_rank` (lowest ranks kept). If supply is short, raise the
  candidate oversampling for the ring; that changes speed, not the distribution. **Never trim
  positives to reach the ratio.**
- Positives: **exactly 20,000 per zone** after thinning and balancing (`per_domain` stays
  20000; oversample more if the final count falls short).
- Add a way to build Florida with band negatives only (for example `--protocols band`). The
  default for other states stays as it is.
- Rewrite the docstring notes that state the old values (note 2's 200 m rationale, the ratio
  note) with the v2 values and the reasons in PLAN §2a.
- Leave everything else unchanged: the area-proportional draw, the 2% per-polygon cap, the
  500–5,000 m band, the FIRM-panel mask, 10 km blocks, and the seed.

### A2. Build and check

Run it for FL, then `--audit FL`. Check and record:
- 20,000 positives and 20,000 band negatives per zone, 80,000 points in total.
- The smallest distance between any two points is ≥ 100 m, measured in EPSG:5070.
- Every band negative is 500–5,000 m from the nearest SFHA and inside a FIRM panel.
- The largest share of a zone's positives that comes from a single polygon. The cap is applied
  before thinning, so report the real maximum and the share supplied by the top 10 polygons.
- A and AE are balanced exactly.

### A3. Re-pull the covariates (Florida only)

Run the pipeline for Florida in the order its docstrings and `docs/` describe (`pull_gee.py FL`
→ the `pull_hydro.py` stages → `pull_atlas14.py` → the merges). Check whether `pull_climate.py`
or `pull_extras.py` produce any of the 45 features and run them only if they do.

- **NHDPlus V2 raw data is not on disk** (`data/raw/` holds only `nfhl/`), and
  `pull_hydro.py --catchments` / `--flowlines` read it. Work out exactly which layers are needed.
  Prefer downloading only the Florida units (NHDPlus V2 South Atlantic 03N, 03S, 03W: catchments
  and flowlines) over the national file. Report the size first; ask if it is over 2 GB. Check
  whether `data/interim/hydro/river_FL.parquet` already carries flowline geometry that makes part
  of that stage reusable.
- StreamCat (`data/interim/hydro/streamcat_FL.parquet`) and Atlas 14
  (`data/interim/atlas14/blocks.parquet`) are cached by COMID / county and by block. Reuse what is
  there; fetch only what the new points need.
- Earth Engine: if `ee.Initialize()` fails, stop and ask.

### A4. Compare v2 with v1

- The same 45 feature columns and the same metadata columns as v1; `point_id` unique.
- Per-feature missingness and medians by `domain × label`, v2 against v1. Flag anything that
  moved by more than a few percent and explain it if you can.
- At 100 m spacing two points can sit in the same ~90 m MERIT Hydro cell. Count the pairs closer
  than 125 m whose `hand` and `log10_upa` are identical, and report the number.

---

## Part B: audit the new point table (read-only)

Write `docs/DATA-AUDIT-<today>.md` as you go (`date +%F`). For every item record what you
checked, how (the command or a short code excerpt), the result, and a verdict: ✓ correct,
✗ missing or wrong, ⚠ works but needs a decision.

- Counts by `domain × label × neg_protocol`. Train / ID-test / OOD-test counts after the seed-0
  50 km tile split (`zoo.split_tiles`, `zoo.subset`). **Positive share in the ID and OOD
  halves**: v1 was 0.53 vs 0.64; check that 1:1 brought them close to each other.
- How a band negative gets its zone. PLAN §2 has a draft; confirm it against the code and write
  it in two or three plain sentences that could go in Methods.
- **Feature provenance table**: for each of the 45 features, the source dataset, the script that
  produces it, native resolution, and whether it is point-sampled or a catchment/watershed mean.
  Count distinct sources and check the paper's "seven public sources" (expected: USGS 3DEP,
  MERIT Hydro, NLCD, JRC Global Surface Water, NHDPlus V2, EPA StreamCat, NOAA Atlas 14). Save it
  as `docs/FEATURE-SOURCES.md`.
- Missingness per feature; constant or near-constant features; no `meta_*`, `lon`, `lat`,
  elevation or id column in `FEATURES.txt`; lon/lat in EPSG:4326 and inside Florida.
- Columns the boundary analysis needs: `meta_fld_ar_id` populated for every positive and
  matching an `FLD_AR_ID` in `haz_FL_5070.parquet`; `meta_fld_zone` consistent with `domain`;
  `county_fips`; `meta_eff_date`.

---

## Part C: models, selections and regions on the new sample

### C1. Make `zoo.py` store what the analyses need

Store these directly in every new cell:
- `point_id` for the OOD half (add it to `SPATIAL_COLS`), plus `point_id` and lon/lat for the ID
  half;
- an extended model table: each hyperparameter in `kw` as its own column, the per-model seed,
  the names of the features used, `n_train`, and training accuracy and AUC both on the model's
  own training subsample and on all training-tile rows.

The training metrics are for the label-noise check in Related Work. Sanyal et al. (2025) get an
inverse line only from models that interpolate noisy labels, so the paper has to show whether
ours do. Do not change `zoo_specs`, `split_tiles`, `subset` or `build`; show this with
`git diff` on those functions.

### C2. Build and audit the seed-0 cell

Generate `FL_band_AE_s0` with 300 models on the new dataset. Then check:
- array shapes; 300 models; family counts (expect eight: dt, et, hgb, lr, mlp, nb, rf, xgb);
- every column of `S_ood` / `S_id` maps to the right dataset row through `point_id`;
- the ID and OOD halves come from the same test tiles, and every test tile holds both zones;
- **determinism**: refit 10 models (at least one xgb, one mlp and one hgb) and confirm their
  scores match the stored rows to 1e-9.

### C3. Selections

Run paper-protocol OODSelect at seed 0, classic and spatial (μ = 3, k = 10), over the full
budget grid, with `run_oodselect_local.sh` or its equivalent. Write into
`results/<today>/oodselect_paper/` and run `report_paper.py`. The v1 run took about 40 minutes on
this laptop. Check:
- each `*_theta.npy` has one entry per OOD point;
- the stored test r can be recomputed from the theta file and the cell, for two budgets;
- the aggregate ID→OOD r, and the classic and spatial r, purity, components and tiles at
  S = 250, 500 and 1,000.

### C4. `aotil/regions.py`

`--cells-dir`, `--cell`, `--results-dir`, `--variant`, `--budgets`, `--seed`. It rebuilds the
10-NN graph exactly as the selection did (projection, k, symmetrization; see `aotil/graph.py`
and `aotil/oodselect_paper.py`), takes the top-S mask from each theta file, finds connected
components, and writes:
- `results/<today>/regions/<cell>_<variant>_S<S>_points.parquet`: `point_id`, lon, lat, label,
  component, component size, `tile_50km`, `county_fips`, `meta_fld_ar_id`;
- `results/<today>/regions/<cell>_<variant>_S<S>_components.csv`: per component, n points,
  positives, tiles, counties, extent in km, median nearest-neighbour spacing, dominant
  `landcover`, median `hand`, median `meta_dist_to_river_m`.

**Test it on v1 first.** Pointed at the archived v1 cell and
`results/2026-09-24/oodselect_paper/spatial_mu3_knn10/`, it must reproduce
`REGION-ANALYSIS-FLORIDA.md` §2.1: at S = 500, 8 components across 5 tiles, and components of
≥ 20 points of sizes 198, 142, 102 and 27. Then run it on v2 for spatial S = 250 and 500, and for
classic S = 500 as the scattered comparison.

### Still to run (list only, do not run)

Seeds 1–2 cells and their classic and spatial selections; null-select; the family-disjoint split
(trees→others, others→trees) under the paper protocol; the paper-protocol μ sweep (0, 1, 3, 10)
at S = 250, 500, 1,000; the smoothed-classic ablation. For each, give the command or say which
script is missing.

---

## Part D: GIS inputs for the boundary analysis

Build these for the **v2** regions from C4. If time runs short, stop after D5 and report.

### D0. Checks

- `haz_FL_5070.parquet`: columns, CRS, geometry validity on a sample; `FLD_AR_ID` and
  `FLD_ZONE` present; the polygons containing the region positives are found.
- `data/interim/nfhl_states/pan_FL.parquet` carries FIRM panel polygons (expected yes:
  `subset_nfhl.py` writes `S_FIRM_Pan` with geometry and `sample_points.py` uses it as the
  panel mask). Confirm.
- County boundaries, SSURGO, a 10 m DEM, NWI, PLSS, roads and canals: expect none on disk.
  Check whether `river_FL.parquet` already has canal/ditch flowlines (`FTYPE`).

### D1. Per-point extras sidecar

`data/interim/analysis/point_extras_FL.parquet`, keyed by `point_id`, for every row of the new
`dataset_FL.parquet`:
- `meta_firm_pan`: the FIRM panel id containing the point;
- `meta_near_fld_ar_id`: the nearest SFHA polygon (mainly for negatives);
- `edge_dist_m`: for positives, distance to the edge of the polygon that labelled them.

### D2. Lobes and the edge-vertex table

`aotil/lobes.py`, following `REGION-ANALYSIS-FLORIDA.md` §3.3 steps 1–2. For each region
component with ≥ 20 points:
- **Region lobes:** explode the Zone A polygons that the region's positives fall in, plus Zone A
  polygons within 2 km of any region point, into single parts. Keep the parts that intersect the
  region's convex hull buffered by 2 km.
- **Zone AE lobes** in the region's `tile_50km` tiles (the rule contrast).
- **Matched unselected Zone A lobes** in the same tiles: not touching any region point, matched
  on perimeter, at least three per region lobe where the tiles allow.
- Densify every lobe boundary to 30 m and write a vertex table: `lobe_id`, `zone`, `role`
  (region / matched_A / AE), `component`, `s` (position along the edge), x and y in EPSG:5070,
  lon, lat, and the inward normal as a unit vector. Write the lobe polygons as GeoParquet.
- Write each component's analysis bbox: the union of its lobe extents, buffered by 3 km so a
  1 km displaced-edge null still falls inside every layer. D3–D6 pull data for these bboxes.

### D3. 10 m DEM over the analysis bboxes

The displaced-edge null needs elevation at displaced positions too, so get a local raster, not
just point samples.
- Preferred: USGS 3DEP 1/3 arc-second GeoTIFF tiles from The National Map API. Report the total
  size first; ask if it is over 2 GB.
- Fallback: export `USGS/3DEP/10m` clipped to each bbox from Earth Engine. Call
  `setDefaultProjection` before any terrain operation (`docs/COVARIATES.md` §2.1).
- Sample with rasterio: elevation at each vertex and at ±100 m along its normal, as `z`, `z_in`,
  `z_out` on a copy of the vertex table.

### D4. SSURGO soil map units

`pull_ssurgo.py`: for each bbox, get the map-unit polygons from the NRCS Soil Data Access web
service, with their `muaggatt` attributes (`flodfreqdcd`, `flodfreqmax`, `pondfreqprs`,
`hydclprs`, `drclassdcd`, `drclasswettest`, `hydgrpdcd`).
- Check the current SDA documentation for the spatial-query functions (the plan names
  `SDA_Get_Mukey_from_intersection_with_WktWgs84` and the `mupolygon` geometry).
- Tile large bboxes to stay under SDA's response limits.
- Fallback: the gSSURGO Florida geodatabase; ask first if it is over 2 GB.
- Write GeoParquet in EPSG:5070, plus the map-unit boundaries as lines, each segment tagged with
  which `muaggatt` attributes differ across it.

### D5. County lines and map-sheet edges

TIGER/Line county boundaries for Florida (STATEFP 12) and the FIRM panel edges from
`pan_FL.parquet`, both as lines in EPSG:5070 under `data/interim/gis/`.

### D6. Optional layers (only after D1–D5 pass)

NWI wetland polygons for the bboxes (USFWS Wetlands Mapper service or the Florida download;
check the size first), PLSS sections (BLM CadNSDI), TIGER roads, and canal/ditch flowlines (only
if D0 found none in `river_FL.parquet`).

### D7. Distances from vertices to layers

`aotil/edge_layers.py`: for each vertex from D2, the distance to the nearest line of each layer
from D4–D6, and for soil lines which `muaggatt` attributes change across the nearest one. Write a
new vertex-layer table under `results/<today>/boundary/`. This is the last data step. **Do not
do the analysis**: no level-set classification, no displaced-edge null, no statistics. Do make
sure every layer covers every bbox and every vertex has `z`, `z_in` and `z_out`.

---

## Part E: report back

1. Finish `docs/DATA-AUDIT-<today>.md`:
   - the archive list;
   - **protocol v2 as implemented**: setting, value, achieved;
   - Part B results, one row per item with its verdict;
   - a table: experiment (O1–O5, regions, controls) → data → status (✓ built today /
     ✗ missing / ⚠ problem) → file path;
   - **every number in the paper that the rebuild changes**: quote the sentence from
     `acl_latex.tex` and give the new value (total points; train / ID / OOD counts; aggregate r;
     classic and spatial results at S = 500);
   - each new script with its run command and runtime on this machine;
   - what stays manual: the Flood Insurance Study reports for the counties the v2 regions fall
     in (FEMA Map Service Center); USGS OFR 74-1111 if a region overlaps the Osceola National
     Forest area; and the PDFs Victor must fetch for the new citations (`collins2022`,
     `jia2026`, see PLAN §2a);
   - the still-to-run list from Part C.
2. In `docs/PROJECT-PLAN.md`, fill in the "achieved" column of §2a and update the status column
   of §4. Change nothing else there.
3. Confirm the other nine `dataset_*.parquet` checksums are unchanged.
4. End your session with a short summary for Victor: what is verified, what you built, what is
   broken, and the decisions he needs to make.
