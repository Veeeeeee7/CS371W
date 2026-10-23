# Running the Florida v2 rebuild

The full command routine for `docs/TASK-DATA-AUDIT.md` Parts A–D. The whole routine ran on the
real data on 2026-10-14/15 (D=2026-10-14; results in `docs/DATA-AUDIT-2026-10-14.md`). Expected
values below are from that run unless marked as scratch. To rerun, start from the repo root and
go in order. Each step says what to check before going
on. **Stop** marks the points where a failure means do not continue.

The code review of 2026-10-21 fixed bugs in the sampler and in the Earth Engine and NHDPlus
covariates, so the points and covariates on disk predate the current code. "Still to run" at the
end says what to rerun and what changes (`docs/DATA-SUMMARY.md` §7 for the data).

Times are for this laptop. The raw-input steps, steps 2, 4–5 and 13–14 download data or call web services; sizes are given where they matter.

## 0. Every new shell

```bash
cd /Users/victorli/Documents/GitHub/CS371W
```

The first time, set the results date to today and note what it printed:

```bash
export D=$(date +%F) && echo "D=$D"
```

In every later shell, even on a later day, set the same value by hand (for example
`export D=2026-10-15`), so all outputs land under one `results/$D/`.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 && mkdir -p logs/$D results/$D && df -h .
```

Keep at least 10 GB free throughout. About 122 GB were free on 2026-10-14; the DEM in step 13
is the largest download.

## Raw inputs (done 2026-09-16 to 09-18; rerun only to start from a newer flood map)

Everything below starts from FEMA's National Flood Hazard Layer (NFHL), the national flood map, cut
down to Florida: `data/interim/nfhl_states/{pan,s_bfe,s_lomr}_FL.parquet` and
`data/interim/prep/{idx_FL,haz_FL_5070}.parquet`, all on disk. These four steps rebuild them; each
skips outputs that exist, so for a newer map move the old Florida files aside first. They replace
the old ten-state download guide (moved to `Claude outputs/removed-2026-10-21/docs/DOWNLOAD-GUIDE.md`).

1. The national geodatabase (19.6 GB zip; one connection, resumes after Ctrl-C, then checks the
   archive). The copy on disk is FEMA's extract of 2026-03-03; a new download gets whatever FEMA
   last published:
   ```bash
   ./fetch_nfhl.sh
   ```
2. Extract it with a ZIP64-capable tool (macOS `unzip` cannot read it; `subset_nfhl.py` puts the
   extracted size at about 40 GB), then cut out Florida:
   ```bash
   ditto -x -k data/raw/nfhl/NFHL_Key_Layers.gdb.zip data/raw/nfhl/ && .venv/bin/python subset_nfhl.py FL
   ```
   Pass 1 writes the zone attributes of all ten original states to
   `data/interim/nfhl_haz_attrs_10state.parquet` (`aotil/o5_polygons.py` reads it) and checks each
   state's Zone A share against FEMA's (`max delta ... (OK)` means within 2 points). Pass 2 writes
   `data/interim/nfhl_states/haz_FL.parquet` (the flood-zone polygons) and `pan_FL.parquet` (the
   FIRM panels, the printed map sheets).
3. Before deleting the extracted geodatabase, save the base-flood-elevation lines (`s_bfe_FL`) and
   the map revisions (LOMRs, `s_lomr_FL`). It prints whether deleting the geodatabase is safe:
   ```bash
   .venv/bin/python pull_extras.py
   ```
4. Repack the polygons once into an attribute index and an EPSG:5070 geometry file
   (`idx_FL.parquet`, `haz_FL_5070.parquet`, 5.4 GB). It skips when both exist (`--force` redoes it):
   ```bash
   .venv/bin/python sample_points.py --prep FL
   ```

NHDPlus (step 5, `fetch_nhdplus_fl.sh`), the TIGER county file (step 2, `pull_gis_lines.py`) and the
Earth Engine layers (step 4) are fetched by their own steps.

## 1. Record v1 and archive it (A0, about 1 min)

```bash
.venv/bin/python sample_points.py --audit FL 2>&1 | tee logs/$D/A0_audit_v1.log
```

```bash
./archive_sampling_v1.sh 2>&1 | tee logs/$D/A0_archive.log
```

Check: 15 `moved` lines and 2 `copied` lines. `data/interim/archive/sampling-v1/` now holds
`sha256_before.txt` and `MANIFEST.txt`.

## 2. County polygons, before sampling (D5, 83 MB download, under 1 min)

TIGER/Line 2026 counties and the FIRM panel edges. This comes first because the audit's
"inside Florida" test uses the county polygons, and so does the optional state mask in step 3.

```bash
.venv/bin/python pull_gis_lines.py --selftest
```

```bash
.venv/bin/python pull_gis_lines.py --state FL --dry-run
```

```bash
.venv/bin/python pull_gis_lines.py --state FL --tiger-year 2026 --panels data/interim/nfhl_states/pan_FL.parquet --gis-dir data/interim/gis --raw-dir data/raw/tiger 2>&1 | tee logs/$D/D5_gis_lines.log
```

Check: the perimeter check line reports a gap near 0 m for counties and panels. Outputs:
`data/interim/gis/{county_lines_FL,counties_FL,panel_edges_FL}.parquet`.

## 3. Build the v2 sample (A1–A2, about 25 min)

**Decision 1 first.** Florida's FIRM panel sheets cross the state line, so in the scratch build
1,761 band negatives (4.4%; 1,390 of them Zone A) landed on land in Alabama or Georgia, where the
500 m test sees no floodplain map. Choose one command:

(a) Protocol v2 exactly as approved (the scratch builds used this):
```bash
.venv/bin/python -u sample_points.py FL --protocols band --exact --ring-oversample 6 2>&1 | tee logs/$D/A2_sample_points_FL.log
```

(b) Recommended: also keep band negatives inside Florida. This is a sampling change, which is
why it is opt-in:
```bash
.venv/bin/python -u sample_points.py FL --protocols band --exact --ring-oversample 6 --state-polygons data/interim/gis/counties_FL.parquet 2>&1 | tee logs/$D/A2_sample_points_FL.log
```

(b) also fills `meta_point_county`, the TIGER county each point lies in; with (a) it stays null.
`points_FL.parquet` exists, so a rebuild needs `--force` (without it the script skips).

**Stop** if it prints `FAILED` (exit code 1, nothing written). The message names the setting to
raise.

Expected for (b) with the current sampler, from the code review's scratch build (22.5 min):
- `[A] positives 20,600 kept for 20,000 from 9,335 polygons` (the polygons the area-proportional
  draw placed points in)
- band rings from 4,300 (A) and 4,328 (AE) polygons (cap 10,000), the polygons the kept positives
  came from
- band candidates in Zone D or AREA NOT INCLUDED dropped: 1,532 (A) and 2,137 (AE)
- 28,375 (A) and 47,099 (AE) band candidates after thinning
- `whole-state separation dropped A/0/band 178, A/1/- 0, AE/0/band 177, AE/1/- 2`
- `[FL] wrote 80,000 points`

Before the 2026-10-21 fixes (the points on disk), (b) gave, in the real build of 2026-10-14 (23 min):
- A band: `2,609 outside the state polygons`, 32,998 after thinning
- AE band: `1,092 outside the state polygons`, 48,212 after thinning
- `whole-state separation dropped A/0/band 159, A/1/- 2, AE/0/band 179, AE/1/- 0`
- `[FL] wrote 80,000 points`

and (a), in the scratch builds: `[A] positives 20,600 kept for 20,000 from 930 polygons`;
`band candidates 370,063: … 35,506 after thinning`, and 49,998 for AE;
`whole-state separation dropped A/0/band 136, A/1/- 0, AE/0/band 139, AE/1/- 0`. The mask changes
how many random draws the A band uses, so (b)'s AE positives are a different realization of the
same distribution, not the same points as (a).

```bash
.venv/bin/python sample_points.py --audit FL 2>&1 | tee logs/$D/A2_audit_FL.log
```

Check:
- `exact targets … YES` and `A/AE strata identical: YES`
- `pairs closer than 100 m: 0`
- band distance between 500 and 5,000 m, and `40,000/40,000` inside a FIRM panel
- `band negatives in D or AREA NOT INCLUDED: 0 (must be 0)`. Zone D (flood hazard undetermined)
  and AREA NOT INCLUDED are unassessed land, so a point there is no evidence of dry ground. The
  points on disk have 2,129 there.

Expected values with the current sampler (scratch build): band 500–4,993 m, largest polygon
2.12% (A) and 2.02% (AE). Before the fixes: (a) minimum nearest-neighbour 100.009 m, largest
polygon 2.23% (A) and 2.18% (AE); (b) 100.114 m, 2.23% and 2.09%.

**Stop** if a check fails.

## 4. Covariates: Earth Engine (A3, about 3 min)

```bash
.venv/bin/earthengine authenticate
```

```bash
export EE_PROJECT=your-cloud-project-id        # the Cloud project registered for Earth Engine
```

```bash
.venv/bin/python pull_gee.py --selftest --project $EE_PROJECT
```

Check: the selftest returns values for all bands at the three Vermont points (slope not null).
Since the 2026-10-21 fix, each layer is read on its own native grid, so a point gets the MERIT
Hydro or 3DEP cell that contains it. The review could not run the selftest (Earth Engine asked for
authorization), so this is the first live check that Earth Engine accepts that request.

Finished chunks are skipped, so for a re-pull move `data/interim/gee/FL` aside first. The merge
rewrites `dataset_FL.parquet` from the points and the chunks alone, so steps 5–7 must follow it.

```bash
.venv/bin/python -u pull_gee.py FL --project $EE_PROJECT 2>&1 | tee logs/$D/A3_pull_gee_FL.log
```

```bash
.venv/bin/python pull_gee.py --status FL
```

**Stop** unless status shows 40/40 chunks. Then merge, always naming the state:

```bash
.venv/bin/python pull_gee.py --merge FL 2>&1 | tee logs/$D/A3_merge_gee_FL.log
```

Check: `80,000 rows (points 80,000)`, `feature list left as is`, and no `Not merged` line.

## 5. Covariates: Atlas 14, then NHDPlus + StreamCat (A3, a few minutes)

```bash
.venv/bin/python pull_atlas14.py FL 2>&1 | tee logs/$D/A3_atlas14_FL.log
```

```bash
.venv/bin/python pull_atlas14.py --status FL && .venv/bin/python pull_atlas14.py --merge FL
```

Check: the new blocks are fetched, FL near 100%, and `FEATURES.txt already lists every Atlas
14 band -- left as is`. On 2026-10-14 every request for the 35 new blocks failed (NOAA's
server answered HTTP 503 to manual checks from 23:12 to 00:14), leaving 191 points (0.24%)
without values; status showed 97.9% of blocks. Once it answers, rerun `pull_atlas14.py FL`,
`--merge FL`, then `pull_hydro.py --merge FL`; blocks over open water may still fail as outside
Atlas 14's area. The fetch now ends with a `failure causes:` line (e.g. `HTTP 503`), so the next
run says why blocks failed. Then rerun everything from step 7 into a new `D`. Add `--overwrite` to
`audit_dataset.py`, `zoo.py` (the cell path is not dated), `pull_point_extras.py` and
`check_gis_inputs.py`, or move their old outputs aside: each script skips an existing output.

NHDPlus is already on disk, so `fetch_nhdplus_fl.sh` skips. The `NHDPlusV21_FL.gdb` on disk predates
the 2026-10-21 fix that keeps every Coastline feature, including island and barrier-island shores,
which are not part of the river network and were cut with it. Rebuild it once with `--overwrite`
(the review's rebuild: 53,094 flowlines against 52,351, and 5,377 Coastline features against 4,634;
later runs can drop the flag). `--inspect` must find `NHDPlusV21_FL.gdb` with layers `Catchment` and
`NHDFlowline_Network`.

```bash
./fetch_nhdplus_fl.sh && .venv/bin/python prep_nhdplus_units.py --overwrite && .venv/bin/python pull_hydro.py --inspect
```

The next three stages skip an existing output. For new points, move
`data/interim/hydro/comid_FL.parquet` and `river_FL.parquet` aside first; for the 2026-10-21 fixes
alone (coastline, drainage density, channel slope), `river_FL.parquet` only. `streamcat_FL.parquet` is
a per-county cache keyed by catchment id (COMID) and can stay: the count below says whether it covers
the points.

```bash
.venv/bin/python pull_hydro.py --catchments FL && .venv/bin/python pull_hydro.py --flowlines FL && .venv/bin/python pull_hydro.py --streamcat FL
```

```bash
.venv/bin/python -c "import pandas as pd; c=pd.read_parquet('data/interim/hydro/comid_FL.parquet'); s=pd.read_parquet('data/interim/hydro/streamcat_FL.parquet'); m=c.comid.notna()&~c.comid.isin(s.comid); print(f'{int(m.sum()):,} of {len(c):,} points have a COMID with no StreamCat row (v1: 368; v2 run: 0)')"
```

If that count is far above about 400, tell Claude before merging: the StreamCat cache would need
a per-COMID top-up. (`docs/FEATURE-SOURCES.md` §9 counts something else, points with a COMID
but no `bfiws`: v1 band subset 377, v2 12.)

```bash
.venv/bin/python pull_hydro.py --merge FL 2>&1 | tee logs/$D/A3_merge_hydro_FL.log
```

## 6. Florida-only checks (A3, E3)

```bash
cmp data/processed/FEATURES.txt data/interim/archive/sampling-v1/data/processed/FEATURES.txt && echo "FEATURES.txt identical"
```

```bash
grep -v -e dataset_FL.parquet -e blocks.parquet data/interim/archive/sampling-v1/sha256_before.txt | shasum -a 256 -c
```

```bash
.venv/bin/python -c "import pandas as pd; k=['state','block_id']; a=pd.read_parquet('data/interim/archive/sampling-v1/data/interim/atlas14/blocks.parquet'); b=pd.read_parquet('data/interim/atlas14/blocks.parquet'); f=lambda t: t[t.state!='FL'].sort_values(k).reset_index(drop=True); print('non-FL Atlas 14 rows unchanged:', f(a).astype(str).equals(f(b).astype(str)))"
```

**Stop** unless every checksum line says `OK` and the last line prints `True`. Since the other
nine states were deleted (Florida-only scope), the checksum check prints a single line,
`data/processed/FEATURES.txt: OK`.

## 7. Audit the v2 point table (A4, Part B, under 1 min)

```bash
.venv/bin/python audit_dataset.py --state FL --dataset data/processed/dataset_FL.parquet --v1 data/interim/archive/sampling-v1/data/processed/dataset_FL.parquet --features data/processed/FEATURES.txt --idx data/interim/prep/idx_FL.parquet --panels data/interim/nfhl_states/pan_FL.parquet --counties data/interim/gis/counties_FL.parquet --seed 0 --protocol band --train-dom AE --out results/$D/data_audit 2>&1 | tee logs/$D/B_audit_dataset.log
```

```bash
bash <(awk '/^```bash$/{f=1;next} /^```$/{f=0} f' docs/FEATURE-SOURCES.md) | tee logs/$D/B_feature_sources_v2.log
```

Check `results/$D/data_audit/audit.md`:
- seed-0 split. (a): 27,032 / 11,137 / 10,632, positive shares 0.531 (ID) and 0.601 (OOD).
  (b): 28,109 / 10,235 / 12,771 on 71 / 24 tiles, shares 0.403 and 0.476;
- the v2-vs-v1 shift table;
- the MERIT identical-pair count.

The second command re-measures every number quoted in `docs/FEATURE-SOURCES.md` on v2.

## 8. The seed-0 cell (C1–C2, about 1 min)

`--protocol band` is required: the default (`shadedX`) has no negatives in v2.

```bash
mkdir -p logs/$D/oodselect_paper results/$D/audit && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u aotil/zoo.py --state FL --protocol band --train-dom AE --seed 0 --n-models 300 --data data/processed --out data/interim/aotil/cells --n-jobs -1 --device cpu > logs/$D/oodselect_paper/zoo_FL_band_AE_s0_local.out 2> logs/$D/oodselect_paper/zoo_FL_band_AE_s0_local.err; tail -5 logs/$D/oodselect_paper/zoo_FL_band_AE_s0_local.out
```

Check: the split matches step 7 and `300 models`. (b), the real run: `train 28,109 | ID test
10,235 | OOD test 12,771 | tiles train 71 test 24` (300 models in 22 s). (a), scratch:
`train 27,032 | ID test 11,137 | OOD test 10,632 | tiles train 71 test 23`.

```bash
.venv/bin/python -u aotil/audit_cell.py --cells-dir data/interim/aotil/cells --tag FL_band_AE_s0 --data data/processed/dataset_FL.parquet --features data/processed/FEATURES.txt --n-refit 10 --tol 1e-9 --out results/$D/audit/FL_band_AE_s0_audit.json --overwrite | tee logs/$D/C2_audit_cell.log
```

**Stop** unless it ends `verdict: PASS`. The audit covers shapes, 300 models in eight families,
`point_id` joins for both halves, the same test tiles with both zones, and 10 refits matching to
1e-9. A cell built before 2026-10-21 fails `table == matrices`: `zoo.py` now counts a negative
predicted at exactly p = 0.5 as correct (the argmax rule the selection uses), which raises the
`acc_*` columns of the dt, et and rf models that output exactly 0.5. Rebuild it with `--overwrite`.

## 9. Selections (C3, about 40 min)

`CELLS` is explicit so that the out-of-scope A→AE cell is not built. The cell from step 8 is
reused. `report_paper.py` runs at the end, and its log is copied aside so later runs cannot
overwrite it.

```bash
CELLS="FL:band:AE:0" JOBS=8 caffeinate -i ./run_oodselect_local.sh --date $D && cp logs/$D/oodselect_paper/report_local.out logs/$D/oodselect_paper/report_local_FL_band_AE_s0.out
```

```bash
.venv/bin/python aotil/audit_selection.py --root results/$D/oodselect_paper --cells-dir data/interim/aotil/cells --tag FL_band_AE_s0 --variants classic spatial_mu3_knn10 --check-budgets 250 500 1000 --recompute-budgets 250 500 1000 --smoothed-classic | tee logs/$D/C3_audit_selection.log
```

**Stop** if it exits 1. `results/$D/oodselect_paper/FL_band_AE_s0_selection_audit.md` holds:
- the aggregate ID→OOD r;
- the classic and spatial r, purity, components and tiles at S = 250, 500 and 1,000;
- the smoothed-classic ablation rows.

Purity is now the paper's definition: the share of each selected point's own 10 nearest
neighbours that are also selected. Runs from before 2026-10-21 stored it on the symmetrised graph
(two points count as neighbours if either is among the other's 10 nearest), so their audit fails
the stored-coherence check until the selections are rerun (a new `--date`).
The most-misclassified baseline now averages over random orders of the points tied at the cut;
`report_paper.py` alone refreshes it.

## 10. Regions (C4, seconds)

First the v1 test the brief asks for, on the archived v1 cell. It must report 8 components across
5 tiles at S = 500, with components of 198, 142, 102 and 27 points.

```bash
V1=data/interim/archive/sampling-v1 && .venv/bin/python aotil/regions.py --cells-dir $V1/data/interim/aotil/cells --cell FL_band_AE_s0 --results-dir results/2026-09-24/oodselect_paper/spatial_mu3_knn10 --variant spatial_mu3_knn10 --budgets 250 500 --data $V1/data/processed/dataset_FL.parquet --out results/$D/regions_v1 | tee logs/$D/C4_regions_v1.log
```

Then v2, spatial S = 250 and 500, and classic S = 500 as the scattered comparison:

```bash
.venv/bin/python aotil/regions.py --cells-dir data/interim/aotil/cells --cell FL_band_AE_s0 --results-dir results/$D/oodselect_paper/spatial_mu3_knn10 --variant spatial_mu3_knn10 --budgets 250 500 --data data/processed/dataset_FL.parquet --out results/$D/regions | tee logs/$D/C4_regions_spatial.log
```

```bash
.venv/bin/python aotil/regions.py --cells-dir data/interim/aotil/cells --cell FL_band_AE_s0 --results-dir results/$D/oodselect_paper/classic --variant classic --budgets 500 --data data/processed/dataset_FL.parquet --out results/$D/regions | tee logs/$D/C4_regions_classic.log
```

Each budget is checked against the component and tile counts the selection recorded, and against
the variant in `result.csv`; a mismatch writes nothing and exits 1.

## 11. Per-point extras and input checks (D1, D0, about 4 min)

```bash
.venv/bin/python pull_point_extras.py --state FL --dataset data/processed/dataset_FL.parquet --idx data/interim/prep/idx_FL.parquet --haz data/interim/prep/haz_FL_5070.parquet --panels data/interim/nfhl_states/pan_FL.parquet --out data/interim/analysis/point_extras_FL.parquet 2>&1 | tee logs/$D/D1_point_extras.log
```

Check:
- every positive is covered by its labelling polygon;
- max |diff| against `meta_dist_to_sfha_m` is 0;
- `meta_firm_pan_eff_date` equals the dataset's `meta_eff_date` for every point in a panel.

Columns: `meta_firm_pan`, `meta_firm_pan_eff_date`, `meta_near_fld_ar_id`, `meta_near_src_row_id`,
`meta_near_fld_zone`, `meta_near_dist_m`, `meta_edge_dist_m`.

```bash
.venv/bin/python check_gis_inputs.py --state FL --dataset data/processed/dataset_FL.parquet --regions results/$D/regions/FL_band_AE_s0_spatial_mu3_knn10_S250_points.parquet results/$D/regions/FL_band_AE_s0_spatial_mu3_knn10_S500_points.parquet results/$D/regions/FL_band_AE_s0_classic_S500_points.parquet --river data/interim/hydro/river_FL.parquet --nhdplus-dir data/raw/nhdplus --gis-dir data/interim/gis --out results/$D/gis/check_gis_inputs_FL 2>&1 | tee logs/$D/D0_check_gis_inputs.log
```

## 12. Lobes and the 30 m edge-vertex table (D2, about 30 s and 3 GB of memory per budget)

```bash
for S in 250 500; do .venv/bin/python aotil/lobes.py --regions results/$D/regions/FL_band_AE_s0_spatial_mu3_knn10_S${S}_points.parquet --components results/$D/regions/FL_band_AE_s0_spatial_mu3_knn10_S${S}_components.csv --dataset data/processed/dataset_FL.parquet --idx data/interim/prep/idx_FL.parquet --state FL --min-size 20 --hull-buffer-m 2000 --near-m 2000 --densify-m 30 --n-matched 3 --bbox-buffer-m 3000 --contact-m 0.5 --seed 0 --out results/$D/boundary 2>&1 | tee logs/$D/D2_lobes_S${S}.log; done
```

Check: the normal-direction assertion passes in every component. Read the bbox sizes it prints:
they decide how big steps 13 and 14 are (v1 S = 500: 32,131 km²). Outputs per budget:
`<prefix>_lobes.parquet`, `<prefix>_vertices.parquet` and `<prefix>_bboxes.parquet`, with
`<prefix>` = `FL_band_AE_s0_spatial_mu3_knn10_S{250,500}`. The vertex table carries contact flags
(`contact_row_id`, `contact_zone`, `contact_subty`, `same_role_contact`), so the analysis can keep
only the SFHA-exterior edge and drop shared edges.

When the three outputs exist, `lobes.py` skips only if they hold the same components as this call
(`--only-components`, or all of them); a different set stops with an error, so use another `--out`
or `--overwrite`.

## 13. The 10 m DEM (D3). Decision: the download is over 2 GB

The dry run makes one metadata call per tile and prints the totals for both budgets together.

```bash
B=results/$D/boundary && P=FL_band_AE_s0_spatial_mu3_knn10 && .venv/bin/python pull_dem.py selftest && .venv/bin/python pull_dem.py fetch --bboxes $B/${P}_S250_bboxes.parquet $B/${P}_S500_bboxes.parquet 2>&1 | tee logs/$D/D3_dem_dry.log
```

v1-sized areas needed about 3.1 GB read with `--remote`, which reads only the needed blocks of
the 1/3″ COG tiles and stores no tiles. Whole tiles would be 5.3 GB plus up to 2.7 GB of rasters.
**This is over your 2 GB rule**, so the script refuses unless you raise `--max-gb` deliberately.
Read the dry run's call total for `--remote`, then:

```bash
B=results/$D/boundary && P=FL_band_AE_s0_spatial_mu3_knn10 && .venv/bin/python pull_dem.py fetch --bboxes $B/${P}_S250_bboxes.parquet $B/${P}_S500_bboxes.parquet --remote --yes --max-gb 3.5 2>&1 | tee logs/$D/D3_dem_fetch.log
```

It still refuses if free disk would fall below 10 GB. `fetch` keeps an existing raster whose grid
and source match, so add `--overwrite` to rebuild rasters made before the 2026-10-21 fix: where
two 1×1° tiles of different releases meet, each pixel now comes from the tile that owns it, which
removes a false 0.17–0.43 m step near the seam. Then elevation at every vertex and ±100 m along its
inward normal:

```bash
for S in 250 500; do P=FL_band_AE_s0_spatial_mu3_knn10_S${S}; .venv/bin/python pull_dem.py sample --vertices results/$D/boundary/${P}_vertices.parquet --out results/$D/boundary/${P}_vertices_z.parquet 2>&1 | tee logs/$D/D3_dem_sample_${P}.log; done
```

If a coastal component misses DEM values over open water, it exits 1 and lists the misses. Rerun
`sample` with `--allow-missing` only if you accept `dem_ok=False` rows. The output
also flags whether each ±100 m probe really lands inside or outside its lobe
(`in_probe_inside`, `out_probe_outside`, `out_probe_in_any_lobe`). In v1 only 81–88% of region
inward probes stayed inside, so the analysis should filter on these.

## 14. SSURGO soil map units (D4; v1-sized areas: about 770 MB from Soil Data Access, 1.5 GB on disk; runtime not measured at this scale)

```bash
.venv/bin/python pull_ssurgo.py --selftest --gis-dir $(mktemp -d)
```

Plan (keys only):

```bash
B=results/$D/boundary && P=FL_band_AE_s0_spatial_mu3_knn10 && .venv/bin/python pull_ssurgo.py --bboxes $B/${P}_S250_bboxes.parquet $B/${P}_S500_bboxes.parquet 2>&1 | tee logs/$D/D4_ssurgo_plan.log
```

If the plan reports more than 2 GB, decide before going on. Otherwise:

```bash
B=results/$D/boundary && P=FL_band_AE_s0_spatial_mu3_knn10 && .venv/bin/python pull_ssurgo.py --bboxes $B/${P}_S250_bboxes.parquet $B/${P}_S500_bboxes.parquet --yes --workers 2 2>&1 | tee logs/$D/D4_ssurgo.log
```

Outputs: `data/interim/gis/ssurgo/<prefix>_mupolygons.parquet` and `<prefix>_mulines.parquet`.
Lines are tagged:
- `kind`: `soil_line`, `survey_join`, `water_edge` or `clip_edge`, or `no_data_edge` (one-sided,
  inside the queried area, where SSURGO has no polygon);
- `areaname_a` / `areaname_b`, so county survey joins can be told from National Park, Forest
  and Conservation Area joins;
- `diff_<attr>`: which `muaggatt` attributes change across the line.

Each component's perimeter-balance line now also prints how many polygons own over 1 m more line
than their perimeter: a join written twice where two polygons overlap slightly. A nonzero count
raises the component's topology WARNING; the doubled lines are flagged, not removed. To get these
lines for an existing pull, rerun the `--yes` command with `--overwrite` (it rebuilds from the cache).

If it stops with "re-run with --refetch" (SSURGO was refreshed mid-run), do exactly that.

## 15. Distances from vertices to every layer (D7, about 1.5 min for S = 250 and 4.5 min for S = 500)

```bash
for S in 250 500; do P=FL_band_AE_s0_spatial_mu3_knn10_S${S}; .venv/bin/python aotil/edge_layers.py --vertices results/$D/boundary/${P}_vertices_z.parquet --lobes results/$D/boundary/${P}_lobes.parquet --bboxes results/$D/boundary/${P}_bboxes.parquet --gis-dir data/interim/gis --state FL --canals-from-nhdplus --out results/$D/boundary/${P}_vertex_layers.parquet 2>&1 | tee logs/$D/D7_edge_layers_${P}.log; done
```

If step 13 needed `--allow-missing`, add `--allow-dem-missing` here. Vertices just past the
state line (within `--state-tol-m`, 1 km) and vertices of a lobe that runs past its own county's
panels are measured and flagged (`outside_state`, `outside_own_panels`) rather than failing the
run. Since 2026-10-21 every vertex is checked against `counties_FL.parquet`, not only those of a
box that reaches past the county layer's extent. The seed-0 tables of 2026-10-14 and 2026-10-16
miss `outside_state` on some vertices that lie outside (add `--overwrite` to rebuild them). The
coverage table must show `ok` or an explained `warn` for:
- the DEM;
- soil (`soil_line` only), `soil_water`, `soil_join_county`, `soil_join_other`;
- county and state-or-coast lines;
- panel edges, which are the point's own DFIRM only;
- canals (NHDPlus CanalDitch).

NWI, PLSS and roads have no collector (D6, optional) and show as absent. **This is the last data
step: no analysis.**

## 16. Final checks and what to send back

```bash
grep -v -e dataset_FL.parquet -e blocks.parquet data/interim/archive/sampling-v1/sha256_before.txt | shasum -a 256 -c && df -h .
```

Then tell Claude the date `D`, so the report's pending numbers can be filled in from:
- `logs/$D/`
- `results/$D/data_audit/audit.md`
- `results/$D/audit/FL_band_AE_s0_audit.json`
- `results/$D/oodselect_paper/FL_band_AE_s0_selection_audit.md`
- `results/$D/regions/*_components.csv`
- `results/$D/gis/check_gis_inputs_FL.md`
- the coverage JSONs under `results/$D/boundary/`

## Extra selection runs: seeds 1–2, the μ sweep and the controls

These ran on 2026-10-16 on the seven-feature cells (`docs/TASK-FULL-RUN-W7.md` C3–C4; results in
`docs/RUN-W7-REPORT-2026-10-16.md`). C4 points to the μ-sweep command below under this section's
old name, "Still to run". As written the commands use the 45-feature cell; for the seven-feature
cells pass `--cells-dir`/`--cells data/interim/aotil/cells_w7` (and, to the script,
`DATA_DIR=data/processed_w7 CELLS_DIR=data/interim/aotil/cells_w7`).

- **Seeds 1–2**: one call per seed, so the Optuna search seed follows the tile-split seed, with
  the report log copied aside each time.
  ```bash
  CELLS="FL:band:AE:1" SEED=1 JOBS=8 caffeinate -i ./run_oodselect_local.sh --date $D && cp logs/$D/oodselect_paper/report_local.out logs/$D/oodselect_paper/report_local_FL_band_AE_s1.out && CELLS="FL:band:AE:2" SEED=2 JOBS=8 caffeinate -i ./run_oodselect_local.sh --date $D && cp logs/$D/oodselect_paper/report_local.out logs/$D/oodselect_paper/report_local_FL_band_AE_s2.out
  ```
  ```bash
  for s in 1 2; do .venv/bin/python aotil/audit_selection.py --root results/$D/oodselect_paper --cells-dir data/interim/aotil/cells --tag FL_band_AE_s$s --recompute-budgets 250 500 1000 --smoothed-classic; done
  ```
  Then `aotil/regions.py --cell FL_band_AE_s{1,2} --seed {1,2}` as in step 10, and
  `aotil/region_recurrence.py` (`docs/TASK-FULL-RUN-W7.md` C5) to see which regions recur.
- **Paper-protocol μ sweep** (μ = 0 and 3 come from step 9; this adds μ = 1 and 10 at S = 250, 500
  and 1,000, paired by `--seed 0`):
  ```bash
  for mu in 1 10; do for S in 250 500 1000; do echo "$mu $S"; done; done | caffeinate -i xargs -P 6 -L 1 bash -c 'mu=$0; S=$1; vd=spatial_mu${mu}_knn10; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u aotil/oodselect_paper.py --cells data/interim/aotil/cells --tag FL_band_AE_s0 --out results/$D/oodselect_paper/$vd --budget $S --mu $mu --knn 10 --seed 0 --threads 1 > logs/$D/oodselect_paper/${vd}_FL_band_AE_s0_S${S}_local.out 2>&1 && echo "done $vd S=$S" || echo "FAIL $vd S=$S"'
  ```
  ```bash
  .venv/bin/python aotil/audit_selection.py --root results/$D/oodselect_paper --cells-dir data/interim/aotil/cells --tag FL_band_AE_s0 --variants classic spatial_mu1_knn10 spatial_mu3_knn10 spatial_mu10_knn10 --check-budgets 250 500 1000 --recompute-budgets 250 500 1000 --out results/$D/selection_audit/mu_sweep
  ```
- **Smoothed-classic ablation**: produced by step 9 (`--smoothed-classic`), post hoc. The ranking
  uses `expit(−θ)`, because `sigmoid(θ)` rounds to 1.0 and made the result depend on column order.
- **Null-select and the family-disjoint splits**: `--null-seed N` and
  `--split {trees->others,others->trees}` of `aotil/oodselect_paper.py`, run through the script's
  `NULL_SEEDS="1 2 3"` or `SPLITS="trees->others others->trees"` knobs, usually with
  `BUDGETS="250 500 1000"`. The runs go into suffixed folders (`classic_null1`,
  `spatial_mu3_knn10_trees2others`, ...); audit them with `audit_selection.py --variants` naming those
  folders, classic first, and its own `--out`. `aotil/README.md`, "Controls", says what each does.

## Still to run: not part of this routine

1. **Rebuild after the 2026-10-21 code review.** The review fixed the sampler (positives uniform by
   area, band negatives kept out of Zone D and AREA NOT INCLUDED, thin polygons filled, finer ring
   buffers, the new `meta_point_county`), the Earth Engine reads (the MERIT Hydro and 3DEP cell
   that contains the point) and the NHDPlus covariates (island and panhandle coastlines, drainage
   density 0 in sink catchments, `channel_slope`'s −9998 as missing). `docs/DATA-SUMMARY.md` §7 says
   what each changes. Everything from the points on is stale, including `data/processed_w7/` and
   `data/processed_w7_minus_*/`, which link to `data/processed/dataset_FL.parquet`. In a new `D`:
   - Move the v2 files keyed by `point_id` aside, as step 1 did for v1 (`archive_sampling_v1.sh`
     handles only v1 and skips what it already archived): `data/interim/gee/FL/`,
     `data/interim/hydro/comid_FL.parquet` and `river_FL.parquet`.
   - Step 3 (b) with `--force`, and its `--audit`; step 4 (re-authenticate first); step 5 with
     `prep_nhdplus_units.py --overwrite`; step 6.
   - Steps 7–15 with `--overwrite` where a script skips an existing output (`audit_dataset.py`,
     `zoo.py`, `pull_point_extras.py`, `check_gis_inputs.py`, `pull_dem.py fetch`,
     `aotil/edge_layers.py`).
   - Then `docs/TASK-FULL-RUN-W7.md` Part C, `docs/TASK-O5-REGIONS.md` (S1, S2, Parts A–F),
     `docs/TASK-REGION-TWINS.md` and the figures in `results/2026-10-19/paper_figures/`.

   On the current points the review measured: 917 coast distances shrink and none grow (641 by more
   than 0.3 in log10, all Zone AE, mostly in Monroe County); 1,582 drainage densities go from missing
   to 0; 2,129 band negatives lie in Zone D or AREA NOT INCLUDED. The new sampler's kept positives
   come from 4,300 (A) and 4,329 (AE) polygons instead of 592 and 950. How far the regions, r values
   and twins numbers move is unknown until the rerun.

2. **Results the review made stale without new points.** All of these are redone by item 1; each
   can also be refreshed alone.

   | fix | stale | rerun |
   |---|---|---|
   | most-misclassified ties broken at random (100 orders) | `hard_r`, `hard_sel_r`, `hard_purity`, `overlap_hard` in `results/2026-10-16/oodselect_paper/` (paper's Hardest row at S = 500: −.13/−.14/−.31 becomes −.21/−.19/−.36) | `report_paper.py` alone: `.venv/bin/python aotil/report_paper.py --root results/2026-10-16/oodselect_paper --cells data/interim/aotil/cells_w7 --tags FL_band_AE_s0 FL_band_AE_s1 FL_band_AE_s2` |
   | purity as the paper defines it | every stored `coh_purity` (S = 500: .08/.10/.09 becomes .09/.11/.10 classic, .75/.81/.81 becomes .76/.82/.81 spatial) | new selections (W7 C3–C4, new `--date`), then their audits and the report |
   | p = 0.5 ties in the `acc_*` columns | every `<tag>_models.csv` | the cells (step 8, W7 C1, O5 D2 and F populations) with `--overwrite`, then O5 Part A |
   | O1: land excludes missing land cover, p not rounded, 2,000 bootstrap draws | `results/2026-10-16/o1_shift*.csv` | W7 C2 |
   | O5 Part C label scopes; D2 seed means; F land-only evaluation, p not rounded, 1,000 draws | `C_conflict.csv`, `C_support.csv`; `D2_lofo.csv`; `F_pooling.csv` (primary −0.0018 becomes about −0.0030) | `docs/TASK-O5-REGIONS.md` Part C; D2 step 4; F evaluation |
   | twins: split 2's strongest models ranked on the stored ID AUROC; figure axes from the data | split-2 twins outputs, `fig_twins.pdf` | `docs/TASK-REGION-TWINS.md` Part A |
   | DEM seams; `outside_state` on every vertex | DEM rasters of every seed and the vertex tables built on them; seed-0 `outside_state`; the boundary statistic | step 13 `fetch --overwrite` and `sample`, step 15 `--overwrite`, W7 C7 |
   | audit counts (B3 missingness, B16 ties, panel overlaps) | `results/2026-10-14/data_audit/`, `results/2026-10-14/gis/check_gis_inputs_FL.*` | steps 7 and 11 with `--overwrite` |
   | doubled soil joins reported | the SSURGO pull logs only | step 14 with `--overwrite` |

3. **Atlas 14 gap**: 191 points in 35 blocks have no rainfall values (step 5). A rebuild fetches
   blocks for its own points anyway.
4. **`docs/TASK-REGION-TWINS.md`** has not run: Part A (reproduce the paper's twins numbers),
   Part B (port them into `aotil/`) and its report.
