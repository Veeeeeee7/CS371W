# Task: land-only rebuild (v3) and the paper's experiment on one split

Brief for a Claude Code session in this repo, written 2026-10-23. Victor comes from CS, not GIS.
Explain GIS terms in CS terms in anything you write for him.

The 2026-10-21 code review (`docs/CODE-REVIEW-2026-10-21.md`) fixed bugs that change the points
and the covariates, so the dataset must be rebuilt. This task:
1. adds a water mask to the sampler, so no point lies on open water;
2. rebuilds the Florida points and covariates with it;
3. reruns, on one split, only what the paper reports;
4. writes a report that gives every paper number old and new.

---

## 0. Decisions (Victor, 2026-10-23)

**No point on open water, enforced inside the sampler.**
- In the current data, 35% of Zone AE band negatives and 16% of Zone AE positives sit on NLCD open
  water (class 11). For Zone A the shares are 5% and 3%.
- Height above drainage (`hand`) and topographic wetness (`wetindexcat`) are missing on most of
  those points. That makes "missing" a cue for the label.
- Samela 2017 (p. 4) and Wing 2017 (p. 4) leave open water out. Woznicki 2019 codes it as
  floodplain. None of them uses open water as a dry point.
- The region analysis is already land-only. With the mask, every analysis uses the same points.
- Positives are drawn uniformly by area over the land inside their zone's polygons. Band
  negatives are drawn uniformly over the land in the band. The 20,000 per class per zone and the
  1:1 ratio stay exact.

**One split only: seed 0.** There is no time for more.
- Seeds 1 and 2 are not built.
- Nothing that needs several splits is run: region recurrence, the "four other places" check
  and pooling over splits.
- Seed 0 is the default split, and the μ sweep and the ablations always ran on it. It was chosen
  before any rebuilt result existed.

**Only what the paper uses.**
- Out of scope:
  - the boundary statistic (`RUN-FL-V2.md` steps 11 to 15: per-point extras, input checks,
    lobes, DEM, SSURGO, vertex distances; `boundary_stat.py`);
  - O5 Parts B to F, including the shared pieces S1 and S2;
  - the leave-one-feature-out data folders `data/processed_w7_minus_*`.
- The paper keeps only O5 Part A (memorized noise) and the twins.

---

## 1. Rules

- `export D=<today>`. Outputs go under `results/$D/`, logs under `logs/$D/`.
- Do not change anything under:
  - `results/2026-10-14/` through `results/2026-10-19/`;
  - `data/interim/aotil/cells/` or `data/interim/aotil/cells_w7/`;
  - `data/interim/archive/sampling-v1/`.
  These are the records of earlier runs.
- New cells go to a new folder, `data/interim/aotil/cells_v3/`.
- Do not edit the paper (`CS371_Cho_Li/`). Do not commit or push.
- New behavior goes behind new flags, and defaults stay as they are.
  - With the new flags off, `tests/characterization/test_sampling.py` must still pass unchanged.
  - Run `.venv/bin/python -m unittest tests.characterization.test_sampling -v` before and after
    your sampler change.
- Every new script gets a docstring and a test.
- Use `.venv/bin/python`. Respect each **Stop**: report what happened and wait for Victor.
- `docs/RUN-FL-V2.md` holds the exact commands for every data step. This brief says what
  changes. If the two disagree, this brief wins for this run.

---

## 2. What the paper will claim (fixed before the run)

Do not change these after seeing results. If one cannot be computed as written, say so in the
report and give both versions. "Holds" uses r < −0.3, where a correlation stops being weak
(`salaudeen2025oodselect` p. 19, already used in `results/2026-10-17/o5/PRIMARIES.md`).

| # | Claim | Statistic | Holds if |
|---|---|---|---|
| C1 | The shift is small in aggregate | Mean over 300 models of ID accuracy minus Zone A accuracy (primary; a model predicts flood iff p > 0.5, so a tie counts as dry), with AUROC and F1 of the flood class as secondary metrics. 95% intervals from the same 2,000 bootstrap draws of whole test tiles. Beside them: the gaps of the 15 strongest test models, and Zone A accuracy reweighted to the Zone AE flood share. **Headline scope: all points**, which under the mask are land points. The paired-land scope stays secondary. *Amended on 2026-10-23, before any v3 result existed: the statistic was the AUROC gap alone.* | Reported whatever its sign or interval. No other scope may replace the headline. |
| C2 | Accuracy-on-the-line holds in aggregate | r on all Zone A test points, 60 test models | r > +0.3 |
| C3 | OODSelect inverts, but scattered | classic S = 500 r_te; pieces, tiles, purity | r_te < −0.3, and below all 3 shuffled-classic runs at S = 500 |
| C4 | The penalty makes regions at a small cost | spatial (μ = 3, k = 10) S = 500 r_te, purity, pieces; smoothed-classic purity at S = 250, 500, 1,000 | r_te < −0.3, purity ≥ 0.5, pieces ≤ classic pieces / 5, and purity above smoothed-classic at all three budgets |
| C5 | Not memorized label noise | r on the **held-out** test models with `acc_train_sub` < 0.99, on the spatial S = 500 selection | r < −0.3. The trees→others family-disjoint r at S = 500 is reported beside it, and so is the reverse direction. |
| C6 | Same inputs, different labels | Over the regions with land points: input distance d and accuracy gap g (strongest 15 test models), against the pooled reference patches | g is above the 95th percentile of both patch kinds, and d is not above the 95th percentile of the Zone AE patches |

Readings:
- **C6 fails on d:** the regions are unfamiliar land, which is covariate shift. Report that
  reading.
- **C6 fails on g:** there is no concept-shift claim. The regions are described, not explained.
- **Stop** if the spatial S = 500 selection gives fewer than 3 regions of at least 20 land
  points. The twins analysis would be too thin, and Victor decides what to do.

---

## Part A. The water mask (an hour or two of code, minutes of download)

### A1. `pull_land_mask.py` (new, repo root)

Download NLCD 2021 land cover for Florida from Earth Engine as one local raster, on the same
30 m grid the `landcover` feature is read from.

**Source.** Use the same image as `pull_gee.nlcd_group()`: `USGS/NLCD_RELEASES/2021_REL/NLCD`,
`system:index` 2021, band `landcover`.

**Grid.** Use the asset's native projection: the CRS and the 30 m affine transform from
`image.projection().getInfo()`. Do not resample. The sampler then tests the same pixel that
`pull_gee.py` later reads for each point.

**Extent.** The bounding box of `data/interim/gis/counties_FL.parquet`, buffered by 6 km and
snapped outward to the native grid. The buffer covers the 5 km band, the 100 m ring
simplification and a margin. Florida's counties include coastal water, so band rings reach out
there.

**Download.**
- Call `image.unmask(0)` first, so masked pixels (offshore, outside NLCD) arrive as 0.
- Fetch in tiles small enough for one request, for example 4096 × 4096 uint8 through
  `ee.data.computePixels`. Skip tiles already on disk so a rerun resumes.
- Mosaic into `data/interim/gis/nlcd2021_FL.tif`: uint8 class codes, 0 = no data, tiled
  DEFLATE GeoTIFF.
- Write the CRS, transform, size and SHA-256 to `data/interim/gis/nlcd2021_FL.json`.

**Print.** Pixel counts by class, the share of class 11 and the share of 0.

**Self-check.**
- Take 2,000 random points from the current `data/interim/points/points_FL.parquet` (the v2
  points, before B0 archives them).
- Read their land cover through `pull_gee.sample_group(nlcd_group(), ...)`, the post-review
  native-grid request, and compare with the raster at the same points.
- **Stop** if they agree on fewer than 99.9% of points: the grids do not match.
- Do not compare with the archived `landcover` column. It was read before the review's
  native-grid fix.

**Earth Engine login.**
- If Earth Engine asks for authentication, **Stop** and ask Victor to run
  `.venv/bin/earthengine authenticate`. It opens a browser, so you cannot do it yourself.
- Ask him for `EE_PROJECT` if it is not set.

### A2. `sample_points.py --land-mask PATH` (default off)

**Land.** A point is on land if its raster pixel holds a valid class (not 0) other than 11. A
point off the raster is not on land.
- Read the raster CRS. If pyproj says it equals EPSG:5070 (the sampler's working CRS), look
  pixels up from x, y directly. Otherwise transform first.
- Load the raster once. It is about 0.6 GB as uint8; a windowed or memory-mapped read is fine.

**Positives: drop non-land candidates after placement, before thinning.**
- Apply the test to the output of `place_points`, before `grid_thin`.
- Do not reject water inside `place_points`'s per-polygon loop. That keeps each polygon's count
  proportional to its total area, which over-samples the land of mostly-water polygons.
- Dropping candidates from an area-uniform draw leaves an area-uniform draw over the land.
- The 2% polygon cap stays as it is, on candidates.
- If `--exact` reports positives short, raise `--oversample`.

**Band negatives: two new tests in `sample_band_negatives`'s filter chain, after the
unassessed test.**
1. Drop candidates inside NFHL OPEN WATER polygons (`FLD_ZONE == "OPEN WATER"`, 117 in
   Florida). Read them the way `read_unassessed` reads Zone D.
2. Drop candidates that are not on land per the mask.

Count and print each test like the existing ones. Zone AE loses the most. The review's scratch
build kept 47,099 AE band candidates after thinning against 20,600 needed, so
`--ring-oversample 6` should still be enough. If it is short, the script says what to raise.

**Record.**
- Add the mask class read for each point as a new column, `meta_mask_lc`.
- Add the mask path and its SHA-256 to the build config stored in the parquet metadata.

**`--audit`.** Add two checks, each "must be 0":
- points not on land per the mask;
- band negatives inside NFHL OPEN WATER.

**Tests** (new file in `tests/`, synthetic raster and polygons):
- No sampled point lies on a water pixel or off the raster.
- **Uniform over land.** Polygon P1 is 1 km², all land. P2 is 1 km², half water. Land points
  must split about 2:1 between them. This is the test that catches the per-polygon mistake.
- Band candidates in a synthetic OPEN WATER polygon are dropped.
- With the flag off, the characterization pins still pass.

### A3. Docs

- Add `--land-mask data/interim/gis/nlcd2021_FL.tif` to the step 3 (b) command in
  `docs/RUN-FL-V2.md`, with a sentence on why.
- Add a row to the sampling part of `docs/DECISION-PROVENANCE.md` for the mask. Support:
  Victor's decision of 2026-10-23, `samela2017` p. 4 and `wing2017` p. 4.
- Mark the old "Band negatives are not screened for open water" row as replaced.

---

## Part B. Rebuild the data (about 45 min, mostly sampling)

**B0. Archive v2.**
- Move these into `data/interim/archive/sampling-v2/`, keeping their relative paths:
  - `data/interim/points/points_FL.parquet`
  - `data/interim/gee/FL/`
  - `data/interim/hydro/comid_FL.parquet`
  - `data/interim/hydro/river_FL.parquet`
  - `data/processed/dataset_FL.parquet`
- Write their SHA-256 to `data/interim/archive/sampling-v2/sha256.txt` first.
- Keep the StreamCat cache (`streamcat_FL.parquet`).
- `data/processed_w7/dataset_FL.parquet` is a link to `data/processed/dataset_FL.parquet`. It
  points at the new file once B3 merges it.

**B1. Sample.**
- Run `RUN-FL-V2.md` step 3 (b) with `--force --land-mask data/interim/gis/nlcd2021_FL.tif`,
  then `--audit`.
- **Stop** on `FAILED` or on any failed audit check.
- Record per zone and label how many candidates the mask and the OPEN WATER test dropped.

**B2. Earth Engine covariates.**
- Run step 4: the selftest, the pull, `--status` at 40/40, then the merge.
- Then check:
  - `meta_mask_lc` equals `landcover` on at least 99.9% of points;
  - at most 80 points (0.1%) have `landcover` 11 or missing.
  Small disagreements can come from Earth Engine reading the NAD83 longitude and latitude as
  WGS84, about 1 m.
- Keep any such points, so the counts stay exact, and list them in the report. The land rules
  downstream drop them from the region analysis.
- **Stop** if either check fails.

**B3. Atlas 14, then NHDPlus and StreamCat.**
- Run step 5 as written, with `prep_nhdplus_units.py --overwrite` (the coastline fix).
- Atlas 14 is not a model input. If NOAA's server fails, record the failure causes and go on.
- Stop at the StreamCat coverage check if its count is far above 400.

**B4.** Run step 6, then step 7 (`audit_dataset.py --overwrite`, into `results/$D/data_audit/`).
Check that the open-water rows (B4) now show about 0%. Check that `hand` and `wetindexcat` are
missing on only a few hundred points; on land in v2 it was 45 and 421.

---

## Part C. The experiment, seed 0 only (about 3 to 4 h of compute)

Commands as in `docs/TASK-FULL-RUN-W7.md` Part C and `docs/RUN-FL-V2.md` "Extra selection runs",
with these settings throughout:
- `DATA_DIR=data/processed_w7` (the seven Woznicki features);
- `CELLS_DIR=data/interim/aotil/cells_v3` (and `--cells` / `--cells-dir` to match);
- `CELLS="FL:band:AE:0"`, `SEED=0`.

**C1. Cell.**
- Run `aotil/zoo.py` as in `RUN-FL-V2.md` step 8 (`--protocol band --train-dom AE --seed 0
  --n-models 300`), but with `--data data/processed_w7 --out data/interim/aotil/cells_v3`.
- Then `aotil/audit_cell.py`. **Stop** unless it ends `verdict: PASS`.

**C2. O1 and O2.** Run `aotil/o1_shift.py` on `FL_band_AE_s0`, writing
`results/$D/o1_shift.csv`. Check that the land scope drops 0 or almost 0 points.

**C3. Selections.**
- Run `run_oodselect_local.sh` for classic and spatial over the full budget grid.
- Then `aotil/audit_selection.py --smoothed-classic`.

**C4. Controls at S = 250, 500 and 1,000.**
- Null-select: `NULL_SEEDS="1 2 3"`, classic and spatial.
- Family-disjoint: `SPLITS="trees->others others->trees"`, classic and spatial.
- μ sweep: μ = 1 and 10.
- Audit each set into its own `--out`.

**C5. Regions.**
- Run `aotil/regions.py` for spatial S = 250 and 500 and classic S = 500, into
  `results/$D/regions/`.
- Write `results/$D/o5/regions_land.csv` with the same columns as
  `results/2026-10-17/o5/regions_land.csv`.
  - No script writes that file yet. Add one, for example `aotil/regions_land.py`, with a test.
  - The type is the majority label of a region's land points.
  - Check that the new script reproduces the 2026-10-17 file exactly from
    `results/2026-10-16/regions/` (18 rows).
- Then check the Stop in §2.

**C6. Memorized noise (O5 Part A).**
- Run `aotil/o5_interpolation.py --seeds 0` with this run's cells, selections, regions,
  `regions_land.csv` and dataset, into `results/$D/o5/A_interpolation.csv`.
- In the report, keep the held-out model sets apart from the circular ones, which include
  models the search was fit on (review §5 item 14).
- Family-level r on all of a family's models is circular. Label it as such.

**C7. Twins and figures.**
- Copy `results/2026-10-19/paper_figures/*.py` to `results/$D/paper_figures/`, and leave the
  2026-10-19 folder untouched.
- In the copies, replace the hard-coded `cells_w7`, `results/2026-10-16`,
  `results/2026-10-17/o5/regions_land.csv` and the seed lists `[0, 1, 2]` with this run's paths
  and `[0]`. Use flags or constants in `common.py`.
- Run `region_twins_ref.py 0 200` and `region_twins_summary.py`.
  - The pooled row is now split 0 alone.
  - Keep the bootstrap over regions (1,000 draws).
- Run `fig_twins.py`, `fig_maps_combined.py` (panel c shows split 0's regions) and
  `fig_aotl.py` (split 0).
- Skip `region_recurrence_ref.py`: it needs several splits.
- Do not copy figures into `CS371_Cho_Li/`.

---

## Part D. The report: `docs/RUN-V3-REPORT-$D.md`

1. **§2 claims table.** For each claim: the statistic, its value, and holds / does not hold.
2. **Paper numbers.** Walk `CS371_Cho_Li/acl_latex.tex` from Experiments to Discussion. Give
   one row per number: the paper's value, the value from split 1 of 2026-10-16 where the paper
   gives per-split values, the new value, and the file it comes from.
   - Mark numbers that existed only because of several splits: pooled rows, per-split ranges,
     "only two places recur" and the "four other places" sentence. Mark them "gone with one
     split".
3. **The sample.**
   - Points dropped by the mask and the OPEN WATER test, per zone and label.
   - The supply margins.
   - How many positives now come from polygons under 1 km² (v2: 0.7% of Zone A positives).
   - The `meta_mask_lc` vs `landcover` agreement, with any disagreeing points.
4. **Regions.**
   - For each new region: name (counties), size, type, how many Zone A polygons it draws from,
     and the area of the largest.
   - Whether any old seed-0 region of 2026-10-16 is found again, judged by footprint: at least
     half of a new region's points within 2 km of the old region's points. The old regions were
     Dixie/Lafayette, Hamilton, Bradford (two), Lafayette/Taylor and Gulf.
5. **Twins.** Per-region and pooled d and g with their percentiles. Give Table 3 of the paper
   (by kind) on the new data.
6. **Deviations** from this brief, the code you added, and the tests.

Tell Victor the date `D` and the report path when you finish or reach a Stop. Do not edit the
paper.
