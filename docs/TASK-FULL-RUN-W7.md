# Task: the full experiment with the Woznicki feature set (plan v5)

Brief for a Claude Code session in this repo, written 2026-10-16. It runs every experiment the
paper's five anticipated outcomes need, with the model features switched to the predictor set of
Woznicki et al. (2019). It ends with one report that states each outcome.

Read first, in this order:
1. `docs/PROJECT-PLAN.md` §0a (the feature decision), §1, §3 (one experiment per outcome), §7.
2. `docs/RUN-FL-V2.md`: the command style, the checks and the stop rules this brief reuses. Its
   steps 8–15 are the template for Part C here.
3. `docs/REGION-ANALYSIS-FLORIDA.md` §3.3–3.5: the boundary analysis (O5).
4. `docs/DATA-SUMMARY.md` §1, §3–§5: what the data and the GIS layers are.

Victor comes from CS, not GIS. Explain GIS terms in CS terms in anything you write for him.

---

## 0. What is already known (first look, seed 0–2 cells, run 2026-10-16 in Claude's sandbox)

Claude ran a reduced version in its sandbox: the three cells, OODSelect at S = 250/500/1,000 on
seed 0 and S = 500 on seeds 1–2, and the regions. Everything is in
`results/2026-10-16/w7_first_look/` (its `README.md` has the tables and what was left out). Use
these numbers as the expected values for your run. Differences above about 0.02 in r or AUC mean
something differs; find out what before going on.

| outcome | first look | reading |
|---|---|---|
| O1 shift | AUC gap 0.004 / 0.072 / 0.063 (seeds 0/1/2); tile-bootstrap intervals exclude 0 on seeds 1 and 2; 56% / 94% / 95% of models worse on Zone A | met on two of three seeds; seed 0 is a low-gap split for every feature set |
| O2 aggregate line | r on the 60 test models +0.729 / +0.863 / +0.872 | met |
| O3 classic | S = 500 r −0.889 / −0.936 / −0.914, against +0.70 to +0.85 for random subsets and −0.10 to −0.29 for the hardest points; 303–325 pieces over 21–24 tiles | met, pending null-select and the family split |
| O4 spatial | S = 500 r −0.888 / −0.838 / −0.748 (cost 0.00 / 0.10 / 0.17); 13 / 26 / 10 pieces over 10 / 15 / 8 tiles | met |
| O5 boundaries | not run. A preview of the statistic on the 45-feature regions without its null gives U(region) − U(matched Zone A) = −0.11 at d = 100 m and −0.08 at 30 m, both intervals spanning 0 | open; see the calibration stop in B3 |

Seed-0 regions (spatial, S = 500): Big Bend (Dixie, Lafayette, Taylor; 135 points), Bradford–Union
(108), Palm Beach (53, mostly band negatives), Miami-Dade (38), Hamilton (37 and 25),
Gilchrist–Levy (37), Marion (28). Four of the eight are within 2 km of a 45-feature region, so the
DEM and SSURGO pulls are mostly new.

(Updated 2026-10-21.) Two fixes move numbers this brief reports, on any rerun. The
hardest-points r is now the mean over 100 random orders of the points that tie at the S-th
score (`aotil/report_paper.py`); on the stored 2026-10-16 runs this moved S = 500 from
−0.13/−0.14/−0.31 to −0.21/−0.19/−0.36. kNN purity now follows the paper's equation on each
point's own k nearest neighbours (`aotil/graph.py` `coherence_metrics`), which raised it by up
to 0.015 at S = 500. Other 2026-10-21 fixes (sampling, the Earth Engine pull, the coast and
drainage-density columns) change the data once `docs/RUN-FL-V2.md` is rerun; after that no
number in this brief is an expected value.

---

## 1. Rules

- **Do not change the data.** `data/processed/dataset_FL.parquet`, `data/processed/FEATURES.txt`
  (the 45-feature list) and everything under `data/interim/` stay as they are. Sampling is not
  redesigned.
- **The model features are `data/processed_w7/FEATURES.txt`** (seven columns):
  `landcover, slope, tri_300, wetindexcat, log10_dist_river, log10_dist_coast, hand`.
  `data/processed_w7/dataset_FL.parquet` is a relative link to `../processed/dataset_FL.parquet`.
  `aotil/zoo.py` reads `<--data>/FEATURES.txt`, and `run_oodselect_local.sh` passes `DATA_DIR`, so
  no code change is needed to switch features.
- **Separate cells.** Write the new cells to `data/interim/aotil/cells_w7/`. Never write into
  `data/interim/aotil/cells/` (the 45-feature cells): the cell path is not dated, so a mistake
  there overwrites the old seed-0 cell.
- **One new results date** for everything: `export D=<today>`, outputs under `results/$D/` and
  `logs/$D/`. The 45-feature outputs in `results/2026-10-14/` stay untouched; they are the
  comparison.
- **Code changes** (Part B): add flags, keep every default as it is, write a test or a smoke run
  for each, and keep each change small. Say in the report what changed and why.
- **Stop points** are marked **Stop**. At a stop, write what you found into the report and ask
  Victor before going on.
- Use `.venv/bin/python`. Keep at least 10 GB free (`df -h .`).

---

## Part A. Set-up checks (minutes)

A1. The feature directory:

```bash
cd /Users/victorli/Documents/GitHub/CS371W && export D=$(date +%F) && echo "D=$D" && mkdir -p logs/$D results/$D
```

```bash
grep -v '^#' data/processed_w7/FEATURES.txt | grep -c . && ls -l data/processed_w7/ && shasum -a 256 data/processed/dataset_FL.parquet data/processed_w7/dataset_FL.parquet
```

Check: 7 features; the link resolves; the two checksums are identical. If the link is missing,
recreate it: `ln -s ../processed/dataset_FL.parquet data/processed_w7/dataset_FL.parquet`.

A2. Missingness of the seven features by stratum, for the report:

```bash
.venv/bin/python -c "import pandas as pd; d=pd.read_parquet('data/processed_w7/dataset_FL.parquet'); f=[l.strip() for l in open('data/processed_w7/FEATURES.txt') if l.strip() and not l.startswith('#')]; print((100*d[f].isna().groupby([d.domain,d.label]).mean()).round(2).to_string())" | tee logs/$D/A2_missingness_w7.log
```

Expected: `hand` and `wetindexcat` missing for about 30% of Zone AE band negatives and under 3%
of positives (open water). Everything else 0.

---

## Part B. Code changes (before the run)

### B1. Null-select and the family-disjoint split in `aotil/oodselect_paper.py`

Both are controls the paper needs for O3 (`PROJECT-PLAN.md` §3, "Controls"). Neither exists.

- `--null-seed N` (default: off). Before the search, permute the ID accuracies of the selection
  and validation models with `np.random.default_rng(N)`. Run the identical search. Score the
  kept θ on the test models against their **true** ID accuracies. A real effect gives r near 0
  or positive here; an inversion as strong as the real one means the optimizer manufactures it.
  Output folder suffix `_null{N}` (for example `classic_null1/`), and a `null_seed` column in
  `*_result.csv`.
- `--split {random,trees->others,others->trees}` (default `random`, the current 60/20/20).
  Use the family groups in `aotil/cells.py` (`TREE_FAMILIES`). The paper protocol needs a
  validation block for Optuna and early stopping, and the family splits have none, so hold out a
  random 25% of the selection-family models as validation (seeded by `--seed`). Report r on the
  other family. Output suffix `_trees2others` / `_others2trees`.
- `run_oodselect_local.sh`: add `NULL_SEEDS` and `SPLITS` knobs (default empty and `random`),
  passed through to the variant folder names. `aotil/audit_selection.py` already accepts
  suffixed variants; check that it does.
- Test: a smoke run with `TRIALS=3 MAX_EPOCHS=500` on the seed-0 cell for each new flag, and
  `test_oodselect_paper.py` still passes.

### B2. The O1/O2 script: adopt `results/2026-10-16/w7_first_look/o1o2.py` as `aotil/o1_shift.py`

It computes, per cell: per-model AUC on the Zone AE and Zone A halves; the mean gap with a 95%
interval from a bootstrap over test tiles; the share of models worse on Zone A; the
strongest-quartile gap; the slope of Zone A AUC on ID AUC; the same on land points (NLCD ≠ 11)
and on land points in paired blocks (10 km blocks holding positives of both zones); and the
aggregate probit-accuracy r. Two fixes when adopting it:
- its `r_acc_test60` uses its own random 60 models, not the selection's test models. Take the
  paper's test-model r from `report_paper.py` (`r_full_test`) instead, and drop or rename the
  column.
- rebuild p(flood) from the true-class scores without rounding: p = 1 − (1 − S) for positives
  and 1 − S for negatives (`o1_shift.rebuild_p`), so both classes go through the same
  floating-point round trip and ties between a positive and a negative survive. On the
  all-points scope, report the stored `auc_id` / `auc_ood` of `<tag>_models.csv`, which `zoo.py`
  ranks on p itself. (Updated 2026-10-21: this bullet said to keep `np.round(..., 9)`, which the
  2026-10-16 run did; rounding merges near-saturated p into ties and moves the AUCs.)

(Updated 2026-10-21.) Two more differences from the 2026-10-16 run: the tile bootstrap uses
2,000 draws (`o1_shift.py --boot`; 200 then), and a land point needs a land-cover class that is
present and not 11 (`o1_shift.is_land`). On the stored cells the 2026-10-21 changes move the
gap estimates by about 1e-5 or less; the intervals also change with the draw count.

### B3. The boundary statistic (O5): new `aotil/boundary_stat.py`

The vertex tables exist, but the analysis was never written (`DATA-SUMMARY.md` §5: "the analysis
itself is not done yet"). Spec, from `PROJECT-PLAN.md` §3 O5 and `REGION-ANALYSIS-FLORIDA.md`
§3.3–3.5:

**Input.** `results/$D/boundary/<prefix>_vertex_layers.parquet` (from `aotil/edge_layers.py`;
one row per 30 m boundary vertex; columns include `lobe_id, role, component, ring, s, x, y,
z, z_in, z_out, dem_ok, in_probe_inside, out_probe_outside, contact_row_id, outside_state,
outside_own_panels, d_soil_m, d_county_m, d_panel_interior_m, d_panel_exterior_m, d_canals_m`),
`<prefix>_lobes.parquet`, and the line layers under `data/interim/gis/` for the null.
`role` is `region` (Zone A lobes inside a selected region), `matched_A` (unselected Zone A lobes
in the same tiles, matched on perimeter) or `AE` (Zone AE lobes in the same tiles).

**Steps.**
1. Keep the floodplain's outer edge: drop vertices on an edge shared with another floodplain
   polygon (`contact_row_id` not null), vertices with `dem_ok` false, and vertices flagged
   `outside_state` or `outside_own_panels` (count them in the report).
2. Windows: walk each ring in order of `s` and cut it into windows of 10 consecutive vertices
   (300 m). Drop windows with fewer than 8 usable vertices.
3. **Level set** (does the edge follow a contour line, the way a computed water surface meets
   the ground?): the window's elevation range `max(z) − min(z)` ≤ τ, τ = 0.76 m (half of a 5-ft
   contour interval: the ±½-interval standard NRC 2007 Table 3.2 gives for risk class C, which
   covers both Zones A and AE; updated 2026-10-21, this said "FEMA's boundary standard for
   approximate studies"), **and** a real drop
   across the edge: the median of `z_out − z_in` over vertices whose probes landed where they
   should (`in_probe_inside & out_probe_outside`) is > 0, with at least 5 such vertices.
   Sensitivities: τ = 0.3 m; the IQR instead of the range.
4. **On a line**: the window's median distance to the layer ≤ d, d = 100 m (main) and 30 m.
   S = soil-survey lines (`d_soil_m`). M = map lines: min of `d_county_m`, `d_panel_interior_m`,
   `d_panel_exterior_m`. Also report canals (`d_canals_m`) separately.
5. **The primary statistic, fixed before looking**: U = share of edge length (windows) that is
   **not** a level set **and** lies on an S or M line. Report U for `region`, `matched_A` and
   `AE`, pooled over components by edge length and per component. Primary:
   **U(region) − U(matched_A)**. Secondary: U(region) − U(AE), and the full 2 × 2 table
   (level set × on line) per role and per layer.
6. **Displaced-edge null** (how often a curve lands near a soil or map line by chance; soil lines
   are dense in Florida): for each region lobe, translate its vertices by a random vector of length
   uniform on 300–1,000 m and uniform direction, recompute the distances to the S and M layers,
   and keep each window's level-set flag. 200 draws, seeded. Report where the observed U(region)
   and the observed difference fall in the null distribution (percentile). Reuse the distance
   code in `aotil/edge_layers.py` where you can. The translation stays inside the pulled boxes,
   which carry a 3 km buffer.
7. Uncertainty: a bootstrap over components (resample components with replacement, 1,000
   draws) for the primary difference.

**Outputs.** `results/$D/boundary_stat/<prefix>_windows.parquet` (one row per window with its
flags), `<prefix>_summary.csv`, `<prefix>_null.csv`, `<prefix>_report.md`.

**Calibration stop.** Before the null, print the contour-like share for each role. Zone AE
edges are the intended calibration: a detailed study's edge is where a computed water surface
meets the ground, so most AE windows should pass step 3. In the 45-feature preview only 30% of
AE windows passed (region 58%, matched Zone A 49%). A likely cause is that a river's water
surface slopes along the reach, so a 300 m window of a correct riverine edge can rise or fall
more than 0.76 m. **Stop** if fewer than half the AE windows pass, and report the share with τ =
0.76 m, with τ = 1.5 m, and with a detrended range (the range of the residuals after removing a
linear trend in `z` along `s`). Victor decides which test to use before the null runs.

**Test.** A synthetic square lobe whose edge lies exactly on a synthetic soil line gives
on-line share 1 and, translated 500 m, share 0. A synthetic DEM that is flat along the edge and
drops across it gives level-set share 1. Then run it on the existing 45-feature S = 250 table
(`results/2026-10-14/boundary/FL_band_AE_s0_spatial_mu3_knn10_S250_vertex_layers.parquet`) as a
smoke run, and put those numbers in the report labelled as 45-feature numbers.

---

## Part C. The run

Each step follows `docs/RUN-FL-V2.md`'s routine with the paths changed. Times are from that
routine on this laptop.

(Updated 2026-10-21.) The expected counts below are for the 2026-10-14 points; they change once
the points are rebuilt with the 2026-10-21 `sample_points.py`.

### C1. Cells, seeds 0–2 (about 1 min each)

```bash
for s in 0 1 2; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u aotil/zoo.py --state FL --protocol band --train-dom AE --seed $s --n-models 300 --data data/processed_w7 --out data/interim/aotil/cells_w7 --n-jobs -1 --device cpu > logs/$D/zoo_w7_s$s.out 2> logs/$D/zoo_w7_s$s.err; tail -4 logs/$D/zoo_w7_s$s.out; done
```

Check: `[FL] 80,000 rows | 7 features`; 300 models; seed 0 split `train 28,109 | ID test
10,235 | OOD test 12,771 | tiles train 71 test 24` (the tile split does not depend on the
features, so it equals the 45-feature run); seed 1 `29,395 | 10,479 | 15,036` on 71/25; seed 2
`25,670 | 11,426 | 14,357` on 71/23.

```bash
for s in 0 1 2; do .venv/bin/python -u aotil/audit_cell.py --cells-dir data/interim/aotil/cells_w7 --tag FL_band_AE_s$s --data data/processed_w7/dataset_FL.parquet --features data/processed_w7/FEATURES.txt --n-refit 10 --tol 1e-9 --out results/$D/audit/FL_band_AE_s${s}_audit.json --overwrite | tee logs/$D/C1_audit_cell_s$s.log; done
```

**Stop** unless each ends `verdict: PASS`.

(Updated 2026-10-21.) `zoo.py` keeps a cell whose `.npz` exists. Cells built before 2026-10-21
count a negative predicted at exactly p = 0.5 as wrong in the `acc_*` columns; `zoo.py` now
counts it as right (the argmax rule of `oodselect_paper.correctness`), so `audit_cell.py` fails
their "table == matrices" check (by up to 0.017). Add `--overwrite` to the `zoo.py` loop to
rebuild them; the scores S are unchanged.

### C2. O1 and O2 (minutes)

```bash
.venv/bin/python aotil/o1_shift.py data/interim/aotil/cells_w7 data/processed_w7/dataset_FL.parquet FL_band_AE_s0,FL_band_AE_s1,FL_band_AE_s2 results/$D/o1_shift.csv | tee logs/$D/C2_o1_shift.log
```

(Argument order as in the first-look script; adjust if you give it flags.) Also run it on the
45-feature cell `data/interim/aotil/cells/FL_band_AE_s0` for the side-by-side row.

### C3. Selections, classic and spatial, seeds 0–2 (about 40 min per seed)

```bash
for s in 0 1 2; do DATA_DIR=data/processed_w7 CELLS_DIR=data/interim/aotil/cells_w7 CELLS="FL:band:AE:$s" SEED=$s JOBS=8 caffeinate -i ./run_oodselect_local.sh --date $D && cp logs/$D/oodselect_paper/report_local.out logs/$D/oodselect_paper/report_local_FL_band_AE_s$s.out; done
```

The zoo step inside the script finds the existing cells and skips them. Then:

```bash
for s in 0 1 2; do .venv/bin/python aotil/audit_selection.py --root results/$D/oodselect_paper --cells-dir data/interim/aotil/cells_w7 --tag FL_band_AE_s$s --variants classic spatial_mu3_knn10 --check-budgets 250 500 1000 --recompute-budgets 250 500 1000 --smoothed-classic | tee logs/$D/C3_audit_selection_s$s.log; done
```

**Stop** if an audit exits 1.

### C4. Controls at S = 250, 500, 1,000 (seed 0; seeds 1–2 if time)

- Null-select, three null seeds, classic and spatial: `NULL_SEEDS="1 2 3"` (B1).
- Family-disjoint, both directions, classic and spatial: `SPLITS="trees->others others->trees"`.
- μ sweep, μ = 1 and 10 (μ = 0 and 3 come from C3): the command in `docs/RUN-FL-V2.md`, "Still
  to run", with `--cells data/interim/aotil/cells_w7` and `results/$D/...`.
- Audit each with `aotil/audit_selection.py` (suffixed variants).

### C5. Regions, seeds 0–2 (seconds)

Spatial S = 250 and 500, classic S = 500, as in `docs/RUN-FL-V2.md` step 10, with
`--cells-dir data/interim/aotil/cells_w7`, `--data data/processed_w7/dataset_FL.parquet`, the
seed's selection folder, and `--out results/$D/regions`. Then the **recurrence check** (write a
few lines if no script exists): for each seed-0 spatial S = 500 region of ≥ 20 points, the share
of its points within 2 km of a region point of seed 1 and of seed 2. A region that recurs in at
least one other seed is "recurring". The tile splits differ by seed, so a region can only recur
where the tiles are test tiles in both seeds; report that denominator too.

### C6. Lobes, DEM, SSURGO, distances (seed 0 spatial S = 250 and 500; then any recurring regions of seeds 1–2)

`docs/RUN-FL-V2.md` steps 12–15 unchanged except the region paths under `results/$D/`. The new
regions are in different places from the 45-feature ones, so the DEM and SSURGO pulls are new.
`pull_ssurgo.py` reuses its cache where boxes overlap. `pull_dem.py` reuses a raster only when
its grid is identical (hardlinked across prefixes); with `--remote`, partly overlapping boxes
re-read their COG blocks (updated 2026-10-21: this said both scripts reuse their caches). A
`fetch` into a folder that already holds a raster for the same grid keeps it unless given
`--overwrite`; rasters built before 2026-10-21 carry a false step of 0.17–0.43 m a few pixels
from seams between tiles of different releases, which the 2026-10-21 `pull_dem.py` removes.

**Stop** at the DEM dry run and the SSURGO plan if either reports more than 3.5 GB.

### C7. The boundary statistic

(Updated 2026-10-21: the command below is the one the 2026-10-16 run used, from
`logs/2026-10-16/C7_run.sh`. The first version of this step had no `--windows` flag, whose
default `all` gives only the sensitivity. After the calibration Stop, Victor chose the windows
with at least 5 good probes, `--windows testable`, as the primary, with all windows as the
sensitivity. The SSURGO and DEM files are under `results/$D/gis`, the county and panel lines
under `data/interim/gis`.)

```bash
for S in 250 500; do P=FL_band_AE_s0_spatial_mu3_knn10_S${S}; for m in testable all; do out=results/$D/boundary_stat; [ $m = all ] && out=results/$D/boundary_stat_allwindows; .venv/bin/python aotil/boundary_stat.py --vertex-layers results/$D/boundary/${P}_vertex_layers.parquet --lobes results/$D/boundary/${P}_lobes.parquet --gis-dir results/$D/gis --layer-file county=data/interim/gis/county_lines_FL.parquet --layer-file panel=data/interim/gis/panel_edges_FL.parquet --windows $m --null-draws 200 --seed 0 --jobs 8 --out $out > logs/$D/C7_boundary_stat_${P}_windows-${m}.log 2>&1; done; done
```

The recurring seed-1 and seed-2 regions (S = 500) run the same way
(`logs/2026-10-16/C7_run_recurring.sh`).

### C8. The Florida map (Contribution 3), if time

One figure: Florida with Zone A and Zone AE polygons (light), the test tiles of each seed, the
spatial S = 500 regions of all three seeds (recurring ones emphasised), and, for seed 0, each
region's Zone A edge windows coloured by class (level set / on S or M line / neither). Save
`results/$D/figures/florida_map.{pdf,png}` with its script.

---

## Part D. The report: `docs/RUN-W7-REPORT-$D.md`

Lead with one line per outcome: **met / partly met / not met**, and the one number that decides
it. Then the tables below, every number with its file. Compare with the 45-feature seed-0 numbers
(`results/2026-10-14/`) in one column. (Updated 2026-10-21: the purities and hardest-points r
stored there use the definitions from before 2026-10-21, see the note in §0; recompute them or
label that column as the old definitions.)

**O1, the shift** ("a model trained on Zone AE scores lower on Zone A than on held-out Zone AE").
Per seed and pooled: mean ID AUC, mean Zone A AUC, gap with its tile-bootstrap 95% interval,
share of models worse on Zone A, strongest-quartile gap, slope of Zone A AUC on ID AUC. Rows: all
points, land points, paired land blocks. Met if the pooled all-points gap is positive and its
interval excludes 0; say which rows meet it.

**O2, aggregate accuracy-on-the-line.** Per seed: `r_full_test` on the 60 test models with its
Fisher interval, and the model-count convergence rule. Met if r > 0.3 in every seed (the
benchmark-misspecification threshold the plan cites) and strongly met if r > 0.7.

**O3, classic OODSelect finds inverted but scattered subsets.** Per seed at S = 250, 500, 1,000:
test r with interval; the random-subset and hardest-points baselines; null-select (all three null
seeds); both family-disjoint directions; kNN purity, connected components, tiles touched. Met if
r is clearly negative, null-select is not, the family-disjoint splits keep the sign, and the
subset touches most test tiles.

**O4, the penalty makes regions at a small cost.** Spatial vs classic at each budget: Δr, purity,
components, tiles; the μ sweep; the smoothed-classic ablation. Met if r stays clearly negative,
coherence rises sharply, and spatial beats smoothed-classic on coherence at similar r.

**O5, the regions differ from nearby Zone A beyond terrain.** Seed 0 spatial S = 250 and 500:
U(region), U(matched_A), U(AE); the primary difference with its component-bootstrap interval and
displaced-edge null percentile; the per-layer 2 × 2 tables; per component. Region recurrence
across seeds. Read it with the table in `docs/REGION-ANALYSIS-FLORIDA.md` §8 and say which row
the result matches. Met if the primary difference is positive, its interval excludes 0, and the
observed U(region) is above the 95th percentile of its null.

Also: the code changes made (B1–B3), the decisions taken at stop points, and anything that
disagreed with this brief or with `docs/PROJECT-PLAN.md`.

When done, tell Victor the date `D` and the report path, so Claude can read the results with him.
