# Code review, 2026-10-21

What was reviewed, what was found and fixed, what was removed, and what you need to do next.
The decision-by-decision record (what supports every coding choice) is in
`docs/DECISION-PROVENANCE.md`.

## 1. Summary

**Scope.** Every script at the repo root and in `aotil/`, and the paper-figure and twins scripts in
`results/2026-10-19/paper_figures/` (about 33,000 lines before the cleanup).

**How.**
- Independent bug hunts per module with three checklists: geospatial (CRS and units, axis order,
  raster sampling, geometry validity, spatial joins, tiles), logic, and statistics (model splits,
  bootstrap units, correlations, AUC).
- Every reported bug was reproduced before it was fixed and confirmed fixed afterwards. Findings
  that could not be reproduced went to two skeptical second readers.
- Every coding decision (443 of them) was traced to a published work, the method being
  reproduced, one of your recorded decisions, a constraint of the data or tools, or a convention.
  Citations were checked against the PDFs in `refs/`.
- Legacy code was removed and every remaining file was refactored for readability, with its
  behaviour checked before and after on synthetic and real inputs.

**Results.**
- 33 verified bugs fixed (1 high, 5 medium, 27 low), 5 more found and fixed during the refactor,
  and 4 changes adopted from the provenance review.
- **The dataset has to be rebuilt.** The sampler weighted positives by polygon area twice, so the
  points are not uniform by area as the paper says (bug `sampling_01`). Together with three other
  data bugs (band negatives in unassessed areas, island coastlines missing from
  `log10_dist_coast`, MERIT Hydro read from a neighbouring cell), every number in the paper will
  move when the pipeline is rerun. Nothing has been rerun.
- 24 places where the paper's text does not match what the code does (section 5).
- 32 legacy code files removed and six superseded or backup docs archived; every remaining file
  refactored for readability.

## 2. What you need to do

### (a) Review and commit

Nothing was committed. Suggested commands, to run yourself after reviewing `git diff`:

```bash
git add -u
```

`git add -u` stages every edit and every removal. The removed files are the 35 tracked files that
`git status` lists as deleted (the cluster scripts, the old aotil sweep, AlphaEarth, the
Earth Engine diagnostics, `pull_climate.py`, `baselines.py` and friends, and four superseded
docs). They are in git history and, locally, in `Claude outputs/removed-2026-10-21/`
(restore any one with `mv`).

```bash
git add aotil/*.py docs/*.md
```

That adds the code and docs that were never tracked: `aotil/boundary_stat.py`,
`aotil/match_controls.py`, `aotil/o1_shift.py`, the `aotil/o5_*.py` modules and their check
scripts, `aotil/region_null.py`, `aotil/region_recurrence.py`, the W7 and O5 reports and briefs,
and this report and `docs/DECISION-PROVENANCE.md`.

```bash
git rm -r --cached logs/2026-09-19 results/2026-09-19 results/aotil-laptop results/*.csv refs
```

That untracks files that sit in gitignored folders but were committed before the ignore rules
(old sweep logs and results, and the reference PDFs, which are copyrighted). They stay on disk.

### (b) The rebuild

Use a new date `D` for every output, so the 2026-10-14/16/17/19 results stay as records. Steps
marked EE need your Earth Engine login; steps marked NET need the network.

| # | Step | Routine | Why it must rerun |
|---|---|---|---|
| 1 | `sample_points.py FL` rebuild of the points | `docs/RUN-FL-V2.md` step 3 | sampling_01, sampling_02, sampling_03, sampling_06, ring buffer |
| 2 | `prep_nhdplus_units.py` with `--overwrite`, delete `data/interim/hydro/river_FL.parquet`, then `pull_hydro.py` (catchments, flowlines, StreamCat, merge) and `pull_atlas14.py` (NET) | step 5 | hydro_01-03, channel slope; Atlas 14 also refills the 191-point gap if PFDS is up |
| 3 | `pull_gee.py FL` pull, then `--merge FL` (EE) | step 4 | gee_01 (native-grid sampling) |
| 4 | `pull_point_extras.py`, `audit_dataset.py`, `check_gis_inputs.py` with `--overwrite` | steps 6, 7, 11 | new points; audits_01-03 |
| 5 | The cells for seeds 0-2 (`zoo.py` via `run_oodselect_local.sh`), then `audit_cell.py` | `docs/TASK-FULL-RUN-W7.md` C1 | new points; zoo_01 |
| 6 | The searches, `report_paper.py`, `audit_selection.py` | W7 C3-C4 | selection_01 (hardest-points ties), selection_02 (purity) |
| 7 | `aotil/o1_shift.py` | W7 C2 | stats_01, stats_02, 2,000 bootstrap draws |
| 8 | `regions.py`, `region_recurrence.py` | W7 C5 | new selections |
| 9 | `lobes.py`, `pull_dem.py` (NET), `pull_ssurgo.py` (NET), `edge_layers.py`, `boundary_stat.py` | W7 C6-C7 | dem_01, boundary_01, new regions |
| 10 | O5 Parts A-F, including the leave-one-feature-out and pooled cells | `docs/TASK-O5-REGIONS.md` | o5_contrast_01, o5_features_01, Part F land-only and 1,000 draws |
| 11 | Twins: `region_twins_ref.py` for seeds 0-2, `region_twins_summary.py`, `fig_twins.py`, `fig_maps_combined.py`; copy the figures into `CS371_Cho_Li/figures/` | `docs/TASK-REGION-TWINS.md` Part A | paper_twins_01, paper_twins_02, fig_twins axes |

After step 10, `aotil/test_o5_shift_decomp.py` passes again (it currently reports one failure
because the stored Part C outputs predate the label-scope fix).

### (c) Paper numbers to update after the rebuild

Every number in Results and Analysis comes from the rebuilt data, so all of them move. Estimates
of the fixes alone, computed on the current data:
- Table 2, "Hardest" row: about −.13 / −.14 / −.31 becomes −.21 / −.19 / −.36.
- Table 2 purities: OODSelect .08 / .10 / .09 becomes .09 / .11 / .10; ours .75 / .81 / .81
  becomes .76 / .82 / .81.
- Twins: pooled p(flood) 0.56 becomes 0.57; g stays +0.36.
- Table 1 (O1): the pooled gap stays 0.046 on the current data.
- Figure 4 (twins) now shows all 6,400 reference patches; the old fixed axes hid 318.

## 3. Bugs found and fixed

**High**

| Bug | What was wrong | Where | Fix |
|---|---|---|---|
| sampling_01 | Positives were weighted by polygon area twice (a pre-draw of polygons by area, then an area-proportional draw), so small polygons were nearly never sampled: polygons under 1 km² hold 14% of a zone's area but supplied 0.6% of positives. The paper says "uniformly by area". | `sample_points.py`, `polygon_quota` / `place_points` | One capped area-proportional draw over every polygon; geometry is read only for polygons that receive points. |

**Medium**

| Bug | What was wrong | Where | Fix |
|---|---|---|---|
| sampling_02 | 2,129 band negatives (5.3%) lay in Zone D (undetermined hazard) or AREA NOT INCLUDED, i.e. unassessed land labelled dry. | `sample_points.py`, `sample_band_negatives` | Candidates inside those polygons are dropped. |
| hydro_01 | Cutting NHDPlus to the reaches with value-added attributes also dropped 743 coastline features (924 km), so `log10_dist_coast`, a model feature, was too large on the Keys and barrier islands. | `prep_nhdplus_units.py`, `read_unit` | Coastline features are always kept. |
| selection_01 | The most-misclassified baseline broke ties at the cut by column order, which is label order. | `aotil/report_paper.py`, `summarise` | Mean over 100 random tie-breaks. |
| paper_twins_01 | The 15 strongest models were ranked by AUROC on p rebuilt from the stored scores, not by the stored ID AUROC. | `region_twins_ref.py` | Rank by the stored `auc_id`. |
| gee_01 | MERIT Hydro (and 3DEP) were sampled on a 90 m request grid instead of the native 92.77 m cells, so about 42% of points read HAND from a neighbouring cell. | `pull_gee.py`, `sample_group` | Sample in each source's native projection. |

**Low**

| Bug | What was wrong | Fix |
|---|---|---|
| sampling_03 | Rejection sampling stopped after 30 rounds and under-filled ribbon-shaped polygons. | Proposals scale with each polygon's fill of its bounding box. |
| sampling_06 | A band negative's `county_fips` is its source polygon's county, which named regions after the wrong county. | New column `meta_point_county` (the county containing the point), used for region names. |
| hydro_02 | `drainage_density` was NaN instead of 0 in sink catchments with no flowline. | 0 for negative COMIDs. |
| hydro_03 | The 0.05° read box cut off the nearest coastline for western-panhandle points. | Coastline read with a 1° pad. |
| dem_01 | A tile's 6-pixel overlap overwrote its neighbour's interior, leaving a false elevation step at seams between DEM releases. | Each tile writes only its own 1°×1° cell. |
| zoo_01 | Table accuracies counted p = 0.5 as wrong for negatives, unlike the argmax rule. | Same rule as `oodselect_paper.correctness`. |
| selection_02 | Purity divided by the symmetrised degree, not by k as in the paper's equation. | The paper's equation on the directed k-NN graph. |
| regions_01-03 | `--only-components` let other components' region lobes become matched controls, and reused stale outputs for a different component set. | Exclusion covers every component; the skip checks the component set. |
| boundary_01 | Out-of-state boundary vertices were flagged only when a bounding box left the county layer's extent. | Every vertex is tested against the counties. |
| stats_01-03 | Missing land cover counted as land in O1; 9-decimal rounding of p changed AUCs of saturated models; a stratum count read the wrong row. | NaN is not land; no rounding; maximum per stratum. |
| o5_contrast_01 | Label scopes pooled regions that had no pairs of that label. | Only regions with pairs. |
| o5_features_01, -24 | Seed means averaged unrelated regions; Part F scored open-water points. | Region rows excluded from seed means; Part F on land points. |
| paper_twins_02, -27 | The summary crashed with any draw count other than 200; the figure's fixed axes hid 318 patches. | Draw count from the data; axes from the data. |
| audits_01-03 | Missingness rounded before testing; panel overlaps missed containment; every multi-zone tie counted as an A/AE tie. | Unrounded tests; `intersects`; separate counts. |
| point_extras_01-03 | Atlas 14 errors were swallowed; `pull_extras.py` said the GDB was safe to delete after failures; `pull_point_extras.py` crashed when a row group had no polygon within the radius. | Error causes counted; "not safe" with a non-zero exit; empty result. |
| ssurgo_lines_01-02 | Survey joins with small overlaps were written twice unflagged; an invalid polygon repaired into a GeometryCollection lost its boundary. | Excess line length reported; keep the polygonal part. |
| Follow-ups | `o5_polygons.py` used the old lobe-exclusion rule; `o5_pooling.py` still rounded p; `audit_selection.py` ordered rows by set iteration; `audit_dataset.py` crashed on missing values in B9/B11 and wrote "NaT" into JSON. | Same rule as `lobes.py`; no rounding; sorted; reported as failed checks; null. |

## 4. Changes adopted from the provenance review

- O1 tile bootstrap: 200 → 2,000 draws; Part F: 500 → 1,000 (percentile intervals need about
  1,000-2,000 draws, Carpenter and Bithell 2000).
- Band-ring buffer: 8 segments per quarter circle instead of 2, so its corners sit at most 24 m
  (not 381 m) inside the true 5 km band.
- NHDPlus's `-9998` channel-slope code is read as missing.
- `requirements.txt` lists `earthengine-api`; `pull_extras.py` has a `--help`.

## 5. Paper text to correct

The code is right (or follows a recorded decision) and the paper's wording is not. The paper was not edited.

| # | Where in the code | What differs, and suggested wording |
|---|---|---|
| 1 | `aotil/zoo.py` | The code assigns configs round-robin by family (zoo_specs lines 240-250): 8 kinds over 300 models gives 38 hgb models over 27 configs. Checked in all three tables: 11 hgb configs appear twice and 16 once. Only feat_frac, row_frac and the seed are drawn at random. Paper line 229: 'Each model draws its capacity from a small grid, such as tree depth, ...'. Corrected wording: 'Each family cycles through a small grid of capacities, such as tree depth, so that every setting is used, and each model draws its feature and training-row fractions at random from ...'. |
| 2 | `aotil/zoo.py` | Same as finding zoo_03. With 7 features, max(3, int(7*f)) gives 3, 4 or 7 columns (confirmed in all three cells_w7 tables). Paper line 229: '... and its feature and training-row fractions from $\{0.3, 0.6, 1.0\}$ and $\{0.15, 0.5, 1.0\}$'. Corrected wording: '... and a random subset of 3, 4 or all 7 features and 15%, 50% or 100% of the training rows'. |
| 3 | `aotil/oodselect_paper.py` | The code follows the released code, and paper line 253 already says the penalty is 'a one-sided softplus with a warm-up'. But line 120, 'where the budget penalty $B_S$ pushes $\/\mathbf{s}\/_1$ toward $S$', is wrong: softplus(S - sum z) only penalizes mass below S. The stored S=500 soft_mass values are 7169/3999/3850 (classic) and 4208/10060/14281 (spatial), far above S. Separately, aotil/README.md 'Two things to know' still gives the soft mass as 2,800-5,300 and the spatial purity as ~0.85-0.93. Those are figures from an earlier run; W7 gives 3,850-14,281 and 0.75-0.81. Suggested: Paper line 120: 'where the budget penalty $B_S$ penalizes $\/\mathbf{s}\/_1$ falling below $S$; the soft mass may exceed $S$, and the $S$ largest weights form the subset.' In aotil/README.md, update the soft-mass and purity ranges to the W7 values. |
| 4 | `aotil/audit_selection.py` | The code follows its documented definition: audit_selection.py docstring lines 81-90, 'so that is itself + k - 1', and REGION-ANALYSIS.md section 0.4. The Table 2 caption, 'Smoothing averages OODSelect's weights over each point's 10 neighbors.', is inaccurate, and so is RUN-W7-REPORT O4 line 216, 'smoothed over each point's 10 neighbours'. Suggested: Caption: 'Smoothing averages OODSelect's weights over each point and its nine nearest neighbors.' Make the same correction in docs/RUN-W7-REPORT-2026-10-16.md O4. |
| 5 | `fetch_nfhl.sh` | CS371_Cho_Li/*.bib entry fema_nfhl has `note = {Accessed October 2026}`. The labels actually come from the national NFHL_Key_Layers extract dated 2026-03-03 (DOWNLOAD-GUIDE l.85: NFHL_Key_Layers_20260303.gdb.zip). The zip was downloaded 2026-09-16 (file mtime) and subset on 2026-09-18 (pan_FL/haz prep files). Corrected bib note: `National NFHL\_Key\_Layers extract of 3 March 2026, downloaded September 2026`. Optionally add to the Data paragraph (l.203): 'The labels come from the 3 March 2026 national extract of FEMA's National Flood Hazard Layer.' Suggested: Edit the bib note as above (paper only). |
| 6 | `sample_points.py` | The code applies the recorded decision (DATA-AUDIT section 2, decision 1(b); section 9 'Methods also needs the state rule'), so this is paper-wording. The paper's l.195 sentence, 'We also sample 20,000 negatives per zone at 500\,m to 5\,km from the nearest floodplain of any zone, inside FEMA's mapped area.', omits the state mask. Corrected: 'We also sample 20,000 negatives per zone at 500\,m to 5\,km from the nearest floodplain of any zone, inside a FIRM panel and inside Florida's counties, because neighbouring states' floodplains are not in Florida's map.' Positives are not masked (4 AE positives lie 6-214 m outside the TIGER line). Suggested: Edit acl_latex.tex l.195 as above. |
| 7 | `audit_dataset.py` | The code thresholds are fine. PROJECT-PLAN v5 section 0a says 'Report O1 on all points and on land points', and RESULTS-SUMMARY-W7 lists the missingness cue among the limitations to state. The paper reports only 'Pooled over splits, models trained on Zone AE lose 0.046 AUROC on Zone A (Table~\ref{tab:splits}).' Line 203 mentions open-water NaN only to justify land-only regions. Corrected wording: add 'On land points alone the gap is 0.025 [-0.042, 0.070].' Add to Limitations: 'Height above drainage and topographic wetness are missing on open water, almost only for dry-labeled points. The gradient-boosting families (hgb, xgb) route missing values natively and can read missing as dry; the other families see the median imputed from training.' |
| 8 | `audit_dataset.py` | The code enforces the state rule: sample_points --state-polygons with the 67 TIGER 2026 Florida counties is in the RUN-FL-V2 step, and DATA-AUDIT section 5 records 0 band negatives outside Florida. The paper says 'We also sample 20,000 negatives per zone at 500\,m to 5\,km from the nearest floodplain of any zone, inside FEMA's mapped area.' Corrected: '... inside FEMA's mapped area and inside Florida's county boundaries.' DATA-AUDIT line 251 says Methods needs this rule. There is also a doc-stale part: docs/FEATURE-SOURCES.md section 8, 'Still open for v2' bullet (lines 307-310), should say this was decided 2026-10-14 (decision 1(b)): --state-polygons on, 0 band negatives outside Florida. |
| 9 | `audit_dataset.py` | DATA-AUDIT section 4 row 113 marks the A4.3 result '⚠ disclose': 43 of 1,179 pairs under 125 m (86 points) have identical hand and log10_upa, against a 2.1% reference rate. The paper says only 'Height above the nearest drainage ... comes from MERIT Hydro, a 90\,m global river map' and 'any two points are at least 100\,m apart'. Suggested addition: 'At 100 m spacing a few points can share one 90 m MERIT cell; 43 of the 1,179 pairs closer than 125 m (86 points, 0.1%) have identical height above drainage.' |
| 10 | `prep_nhdplus_units.py` | Network-only flowlines are a recorded decision (prep_nhdplus_units.py docstring 'WHY ONLY NETWORK FLOWLINES'; FEATURE-SOURCES §8) for the river distance. The paper's 'NHDPlus streams' is too loose given that 36% of points lie nearer to an excluded line. Paper Sec. Experiments (acl_latex.tex L203): 'Distances to the nearest river and to the coast are straight-line distances to NHDPlus streams and coastline \citep{mckay2012nhdplus}, on a log scale.' Suggested: 'Distances to the nearest river and to the coast are straight-line distances, on a log scale, to the nearest NHDPlus V2.1 network flowline (streams, canals and flow paths through lakes) and to the NHDPlus coastline.' For the coastline half, the cut is a code bug (finding hydro_01). Suggested: Edit acl_latex.tex L203 as above. |
| 11 | `pull_hydro.py` | RIVER_FTYPES (StreamRiver, ArtificialPath, Connector, CanalDitch) is documented in FEATURE-SOURCES §1 ('nearest reach'). The nearest 'river' is an ArtificialPath for 28.4% of points and a CanalDitch for 5.9%, so 'NHDPlus streams' (L203, same sentence as hydro-03) understates it. Use the hydro-03 wording '(streams, canals and flow paths through lakes)'. Suggested: Same L203 edit as hydro-03. |
| 12 | `pull_hydro.py` | FEATURES.txt and PROJECT-PLAN L31 map Woznicki's OFD/HOFD (distance to channel, with coastlines counting as channels) onto two features. Paper L193: 'We use their predictors: ... height above the nearest drainage channel, and distances to the nearest river and to the coast.' Suggested: '... height above the nearest drainage channel, and the distance to the nearest channel, which we split into distances to the nearest river and to the coast.' Suggested: Edit acl_latex.tex L193 as above. |
| 13 | `pull_hydro.py` | The code drops these points anyway. region_twins_ref.py L46 filters landcover != 11 and then dropna(subset=NUM). 422 non-water points lack a numeric feature (421 wetindexcat, outside every NHDPlus catchment; 45 hand), all Zone AE (50 neg, 372 pos), so regions (Zone A) are unaffected and only the AE twin pool shrinks. Paper L203: 'Height above drainage and topographic wetness are missing on open water, so we analyze regions on land points only.' Suggested: 'Height above drainage and topographic wetness are missing on open water (and, for a few hundred coastal Zone AE points, outside every NHDPlus catchment), so we analyze land points with all features present.' Suggested: Edit acl_latex.tex L203 as above. |
| 14 | `aotil/o5_interpolation.py` | The code labels the circular and held-out model sets correctly. The paper reports the circular numbers under a held-out claim. (a) Paper sec:perf, 'Memorized noise': 'Without them, $r$ on our selected points is $-0.82$ [$-0.84$, $-0.79$].' This is the pooled all_lt099 row (271/274/273 models, about 80% of them the selection and validation models the search was fit on). It contradicts line 229, 'every reported correlation uses 60 models that the search never saw'. Corrected: 'Without them, $r$ on our selected points is $-0.81$ [$-0.86$, $-0.74$] on the 51 to 52 held-out test models that do not interpolate ($-0.82$ over all 271 to 274 non-interpolating models, most of which the search used).' (b) The family sentence uses fam_* sets, which contain the selection and validation models. Corrected: 'Within single families (all of a family's models, including those the search used), $r$ ranges from $-0.52$ for naive Bayes to $-0.86$ for random forests'. The same applies to 'invert within seven of the eight model families'. (c) Table tab:errors, row '$r$, test models & $-$.70 & $+$.01': these are the pooled land-only test_lt099 values (-0.696 / +0.006). The 60 test models give -0.706 / -0.024. Either relabel the row 'r, non-interpolating test models', or use -.71 / -.02 and change '($r = -0.70$)' in 'Two kinds of failure' to '$-0.71$'. (d) The pooled bracket treats the three seeds as independent (docstring lines 67-74), so call it 'pooled over splits' rather than presenting it as a plain 95% interval. Suggested: Paper text only (acl_latex.tex lines 229 or 318, the tab:errors row at line 332, and line 345). |
| 15 | `aotil/o5_polygons.py` | The text is in the report (docs/O5-REGIONS-REPORT-2026-10-17.md, Part E, 'What a lobe is'), not in the paper; the paper does not mention vertex counts. Quote: 'The stored lobe geometry is identical to the polygon parts in the prep table `haz_FL_5070`, so vertex counts are those of the published map, not densified.' The repack reprojects each vertex one to one (sample_points._reproject), but sample_points.prep_state replaced the polygons that failed validity with make_valid output. The pre-repair source is not on disk, so the number repaired cannot be checked. Corrected: '...so vertex counts are those of the NFHL polygons as reprojected vertex by vertex to EPSG:5070 in the prep repack, not densified; the few polygons the repack repaired with make_valid carry their post-repair counts.' The practical effect on the medians is likely negligible. Suggested: Report wording only. |
| 16 | `aotil/region_recurrence.py` | The paper says (Limitations): 'For robustness, we study one state, and only two places recur across splits.' No rule is given. The count rests on region_recurrence.py's rule (near share within 2 km >= 0.5): Collier/Hendry 0.71/0.62 and Sarasota 1.0/0.84. region_recurrence_ref.py's 80%-coverage table gives the same two (Collier 84%/55% selected, Sarasota 100%/65%), so the claim holds under both rules. docs/REGION-ANALYSIS.md §0.4 'Seeds' (chance-normalised Jaccard, graph.py) is an old plan and is stale. Suggested wording: 'only two places, Collier and Sarasota, are found again by a second split (at least half of their points within 2 km of that split's regions).' |
| 17 | `results/2026-10-19/paper_figures/region_twins_ref.py` | The code's pooled g_all is 0.1947 (acc_twin_all 0.643 minus acc_set_all 0.448), which rounds to 0.19. Paper Limitations: 'With all 60 test models instead of the strongest 15, the gap shrinks to 0.20'. Corrected wording: '...the gap shrinks to 0.19'. docs/TASK-REGION-TWINS.md section 4 item 1 ('g with all 60 test models: 0.20') should also say 0.19. Finding 01 does not affect g_all. Suggested: Change 0.20 to 0.19 in the paper's Limitations paragraph and in TASK-REGION-TWINS section 4 item 1. |
| 18 | `results/2026-10-19/paper_figures/region_twins_ref.py` | The code follows the brief: docs/TASK-REGION-TWINS.md section 0 says 'Both kinds are grown by breadth-first search on a 10-nearest-neighbor graph (EPSG:5070)'. A separate graph is needed anyway, because the OOD region graph holds no Zone AE points. The paper's 'by breadth-first search on the neighbor graph' points to the EPSG:3086 OOD graph of the search. Corrected wording: 'We grow random patches of the same size as each region by breadth-first search on a 10-nearest-neighbor graph of each pool (held-out Zone AE land points, or unselected Zone A land points).' Suggested: Paper wording only. |
| 19 | `results/2026-10-19/paper_figures/region_twins_ref.py` | The brief (section 0: 'starting in the region's tiles') matches the code, which constrains only the start point. 2-5% of patch points lie outside the tiles. The start-anywhere fallback never fires (see paper_twins_03). Paper: 'Patches of unselected OOD points from the region's tiles show what an ordinary place in pi_tgt gives.' Corrected wording: 'Patches of unselected OOD points, started in the region's tiles, show what an ordinary place in pi_tgt gives.' Suggested: Paper wording only. |
| 20 | `results/2026-10-19/paper_figures/fig_maps_combined.py` | The code draws the two no-land Okeechobee regions in gray, with the legend 'Open water (excluded)'. They are excluded from the analysis, not from the map. Caption: 'Two regions on the open water of Lake Okeechobee are left out.' Corrected wording: 'Two regions on the open water of Lake Okeechobee (gray) have no land points and are left out of the analysis.' Suggested: Caption wording only. Alternatively, drop the gray scatter and its legend entry from fig_maps_combined.py lines 49 and 66. |
| 21 | `pull_gee.py` | Paper L203: 'where terrain variability is the standard deviation of elevation within a 300\,m cell.' Corrected wording: 'within the fixed grid cell of about 300\,m that contains the point (0.0027 degrees, about 270 x 300\,m in Florida)'. Also change O5 report glossary L891 '300 m window' to 'fixed ~300 m grid cell'. Same issue as finding gee_02; no numbers change. |
| 22 | `pull_gee.py` | Paper L203: 'Height above drainage and topographic wetness are missing on open water, so we analyze regions on land points only.' Data: of the 11,798 NLCD open-water points, hand is NaN for 54.3% and wetindexcat for 59.7%. On land, hand is NaN for 45 points and wetindexcat for 421. Corrected wording: 'Height above drainage and topographic wetness are missing for most open-water points, where MERIT Hydro and NHDPlus have no land cell or catchment, so we analyze regions on land points only (NLCD class other than open water).' |
| 23 | `` | Proved by reading the code and checked against the tables. fit_one (zoo.py:361) takes max(3, int(7*f)): f=0.3 gives int(2.1)=2, raised to 3; f=0.6 gives int(4.2)=4. In all three cells_w7 tables, feat_frac 0.3 maps to n_feat 3 exactly (89/103/100 models), 0.6 to 4 (108/112/110) and 1.0 to 7. With 6 features, 0.3 and 0.6 both give 3. The code behaves as intended (a floor of 3 features). The mismatch is the paper's sentence at line 229, so this is a description error, not a computation error. The O5 report (Part D2) notes the 6-feature collapse. No paper, CHANGES or PROJECT-PLAN text records the floor. No test was written because the fix is to the wording. Suggested: Paper line 229 (no code change): 'Each model uses a random subset of 3, 4 or all 7 features and a random 15%, 50% or 100% of the training rows.' The alternative 'feature fractions 0.3, 0.6, 1.0, with at least three features' also works. |
| 24 | `pull_gee.py` | Proof by reading the code. p300 = native.atScale(300) on the EPSG:4269 3DEP projection, and atScale measures metres at the identity point (the equator), so a cell is 300/111319.49 = 0.002695 deg. At 24.5-31 N that is about 257-273 m E-W by about 299 m N-S, roughly 11% less area than a 300 x 300 m cell. FEATURE-SOURCES section 3 records that W is nominal, but that is a fact noted in the doc, not a decision. The paper (L203) and the processed_w7/FEATURES.txt header still say a 300 m cell. The same applies to tpi_300 and to the 900 m windows (tpi_broad, water_frac_900m), which the paper does not use. Suggested: Wording only. Paper L203: 'the standard deviation of elevation within the fixed grid cell of about 300 m that contains the point (0.0027 degrees, about 270 x 300 m in Florida)'. Make the same change in the data/processed_w7/FEATURES.txt header comment. |

## 6. Reported problems that turned out not to be bugs

| Finding | Why it is not a bug |
|---|---|
| Cell.splits 'random-split' is not the paper's 60/20/20 model split, and Cell.x is not the paper's ID accuracy (`aotil/cells.py`) | The facts check out: model_split(300) and default_rng(0).permutation share 11 of 60 test models, and 40 of the cells test models are paper selection models. |
| Smoothed-classic averages over the point itself plus 9 neighbours, but the paper says 10 neighbours (`aotil/audit_selection.py`) | The operator is intentional and documented. |
| carry_field counts each point as its own neighbour when given the same point set twice, so the paper's 'smoothing over 1 (`transfer.py`) | Same issue as selection_03. |
| Band candidates are drawn over overlapping 5 km rings, so their density scales with how many rings overlap (`sample_points.py`) | The behaviour is real: rings are weighted by their own area, so overlap k gives about k times the density. |
| A missing pan_{state}.parquet silently turns off the 'inside a FIRM panel' rule (`sample_points.py`) | This is unreachable in the current pipeline. |
| channel_slope copies NHDPlus's -9998 no-data code as a value (`pull_hydro.py`) | This is a known open decision, kept on purpose: prep_nhdplus_units.py docstring ('SLOPE keeps NHDPlus's own no-data code (-9998) exactly as the national layer did'), DATA-AUDIT-2026-10-14 §12 decision 3 ('Keep (v1 parity, disclose), or map it to NaN'), FEATURE-SOURCES §7.1 and §8. |
| Nearest-reach attributes take an arbitrary reach when several reaches are equally near (confluences) (`pull_hydro.py`) | FEATURE-SOURCES §1 ('nearest reach': nearest river-FTYPE flowline by planar STRtree.nearest, not routed) and §7.5 ('"Drains to" is "nearest"') define the attribute as that of a nearest reach. |
| dataset_xy takes point x/y from --dataset geometry without checking its CRS (risk of treating lon/lat as metres) (`aotil/o5_polygons.py`) | Both data/processed/dataset_FL.parquet and data/processed_w7/dataset_FL.parquet store their geometry as EPSG:5070 (I checked the geo metadata). |
| STUDY_TYP/SOURCE_CIT join on (dfirm_id, fld_ar_id) could drop or duplicate rows through a dtype mismatch or repeated key (`aotil/o5_polygons.py`) | read_attrs removes duplicate keys before the left merge, so the merge keeps the row count and order. |
| ring_runs / band_windows straight-run lengths (start, direction and cyclic wrap) (`aotil/o5_polygons.py`) | Fuzzed against a brute-force enumeration of every cyclic window (headings within 1 degree, length >= 300 m, union coverage, longest run, per-vertex runs). |
| A part that make_valid repairs gives its pre-repair metrics (area, perimeter, vertices) to every piece it splits into (`aotil/o5_polygons.py`) | Documented in the module docstring, lines 62-71: 'repaired' marks a part lobes.py's make_valid had to fix, whose metrics are those of the part before that repair. |
| Pooled Fisher interval treats the three seeds as independent samples of models (`aotil/o5_interpolation.py`) | Documented in the o5_interpolation.py docstring, lines 67-74: the seeds share most points and all 300 model specs, so the interval is 'narrower than an honest one' and only describes the three runs. |
| Missing landcover counts as land (NaN != 11) in the land-only subsets and in the water_only lobe rule (`aotil/o5_interpolation.py`) | o5_polygons and the report state that missing landcover counts as land. |
| ref_A fallback to start-anywhere silently changes the Zone A null (`results/2026-10-19/paper_figures/region_twins_ref.py`) | If patch() fails 1,000 starts, starts is reset to the whole pool, and every later draw of that region would start anywhere. |
| match() leaves np.empty values for a point with NaN land cover (`results/2026-10-19/paper_figures/region_twins_ref.py`) | A point with NaN land cover passes landcover!=11, and in match() it would never equal any lc in the per-class loop, so out_d/out_t would keep np.empty garbage. |

## 7. Cleanup

**Removed** (now in `Claude outputs/removed-2026-10-21/` and in git history): the cluster and SLURM
drivers, the earlier aotil sweep (`aotil/oodselect.py`, `analyse.py`, `gate.py`, `pointcov.py`, `transfer.py`,
`characterise.py`, `test_aotil.py`), the AlphaEarth and Earth Engine diagnostics, the ten-state baselines,
`pull_climate.py` (retired GRIDMET proxy), `fetch_nhdplus.sh`, `nfhl_zone_stats.py`, two unused paper-figure
scripts, and the superseded docs. The two pieces active code still used were kept: `carry_field` now lives in
`aotil/audit_selection.py`; the dead `Cell.x`/`Cell.splits` helpers went with `oodselect.py`.

**Files removed:** `aef_checks.py`, `aotil/analyse.py`, `aotil/characterise.py`, `aotil/gate.py`, `aotil/oodselect.py`, `aotil/pointcov.py`, `aotil/test_aotil.py`, `aotil/transfer.py`, `baselines.py`, `check_ee.py`, `docs/SPATIAL-OODSELECT.md.2026-09-26.bak`, `docs/TASK-MATCHING-MODEL-TERRAIN.md`, `download_from_cluster.sh`, `ee_diagnose.py`, `ee_raw_check.py`, `explain_pooled.py`, `feature_set_gaps.py`, `fetch_nhdplus.sh`, `nfhl_zone_stats.py`, `oodselect_paper_fanout.sh`, `pull_alphaearth.py`, `pull_climate.py`, `results/2026-10-19/paper_figures/fig_conflict.py`, `results/2026-10-19/paper_figures/fig_maps.py`, `run_aotil.sh`, `run_aotil_laptop.sh`, `run_oodselect_paper.sh`, `sbatch_overrides.sh`, `submit_aotil_all.sh`, `submit_aotil_l4.sh`, `submit_oodselect_classic.sh`, `submit_oodselect_spatial.sh`, `upload_data_to_cluster.sh`, `upload_to_cluster.sh`.

**Refactor.** Every file was rewritten for readability: long docstrings, numbered decision essays, history
comments, ten-state leftovers, unused functions and flags removed; dense one-liners split into named steps.
Behaviour was checked before and after on synthetic and real inputs. Many files grew, because dense code became
plain code; the long explanations now live in `docs/DECISION-PROVENANCE.md`.

| File | Lines before | Lines after |
|---|---|---|
| `aotil/audit_cell.py` | 722 | 838 |
| `aotil/audit_selection.py` | 646 | 727 |
| `aotil/boundary_stat.py` | 1764 | 1968 |
| `aotil/cells.py` | 77 | 76 |
| `aotil/edge_layers.py` | 1190 | 1291 |
| `aotil/graph.py` | 300 | 337 |
| `aotil/lobes.py` | 965 | 986 |
| `aotil/match_controls.py` | 629 | 674 |
| `aotil/o1_shift.py` | 326 | 337 |
| `aotil/o5_contrast.py` | 1027 | 1151 |
| `aotil/o5_feature_inclusion.py` | 530 | 565 |
| `aotil/o5_interpolation.py` | 365 | 343 |
| `aotil/o5_lofo.py` | 219 | 240 |
| `aotil/o5_polygons.py` | 1349 | 1448 |
| `aotil/o5_pooling.py` | 230 | 344 |
| `aotil/o5_shift_decomp.py` | 940 | 882 |
| `aotil/oodselect_paper.py` | 635 | 586 |
| `aotil/region_null.py` | 274 | 256 |
| `aotil/region_recurrence.py` | 134 | 149 |
| `aotil/regions.py` | 558 | 569 |
| `aotil/report_paper.py` | 319 | 366 |
| `aotil/test_boundary_stat.py` | 368 | 400 |
| `aotil/test_match_controls.py` | 426 | 511 |
| `aotil/test_o1_shift.py` | 173 | 211 |
| `aotil/test_o5_contrast.py` | 555 | 600 |
| `aotil/test_o5_feature_inclusion.py` | 293 | 319 |
| `aotil/test_o5_interpolation.py` | 234 | 275 |
| `aotil/test_o5_lofo.py` | 140 | 175 |
| `aotil/test_o5_polygons.py` | 838 | 937 |
| `aotil/test_o5_pooling.py` | 123 | 168 |
| `aotil/test_o5_shift_decomp.py` | 765 | 742 |
| `aotil/test_oodselect_paper.py` | 250 | 341 |
| `aotil/zoo.py` | 558 | 508 |
| `archive_sampling_v1.sh` | 86 | 86 |
| `audit_dataset.py` | 1271 | 1403 |
| `check_gis_inputs.py` | 554 | 639 |
| `fetch_nfhl.sh` | 68 | 87 |
| `fetch_nhdplus_fl.sh` | 88 | 82 |
| `prep_nhdplus_units.py` | 165 | 162 |
| `pull_atlas14.py` | 490 | 461 |
| `pull_dem.py` | 1372 | 1442 |
| `pull_extras.py` | 96 | 106 |
| `pull_gee.py` | 547 | 554 |
| `pull_gis_lines.py` | 631 | 726 |
| `pull_hydro.py` | 670 | 658 |
| `pull_point_extras.py` | 578 | 622 |
| `pull_ssurgo.py` | 1281 | 1280 |
| `requirements.txt` | 25 | 22 |
| `results/2026-10-19/paper_figures/common.py` | 45 | 105 |
| `results/2026-10-19/paper_figures/fig_aotl.py` | 26 | 89 |
| `results/2026-10-19/paper_figures/fig_maps_combined.py` | 68 | 151 |
| `results/2026-10-19/paper_figures/fig_twins.py` | 36 | 85 |
| `results/2026-10-19/paper_figures/region_recurrence_ref.py` | 25 | 74 |
| `results/2026-10-19/paper_figures/region_twins_ref.py` | 111 | 212 |
| `results/2026-10-19/paper_figures/region_twins_summary.py` | 54 | 149 |
| `run_oodselect_local.sh` | 176 | 197 |
| `sample_points.py` | 1222 | 1295 |
| `subset_nfhl.py` | 176 | 179 |

**Kept on purpose:** the ten-state output of `subset_nfhl.py` (`aotil/o5_polygons.py` reads it); the shadedX
protocol (still the sampler's default, and `audit_dataset.py` uses it); the `--source imageserver` fallback of
`pull_dem.py`; the AlphaEarth lines of `archive_sampling_v1.sh` (RUN-FL-V2 step 1 expects them).

## 8. Known limitations left as they are

- Band negatives within 500 m of the Alabama/Georgia line can be near out-of-state floodplain the NFHL extract
  does not hold (at most about 58 of 40,000 points).
- `aotil/test_o5_shift_decomp.py` reports one failing check until O5 Part C is rerun (the stored outputs predate
  the label-scope fix). The stored cells' model tables predate the p = 0.5 tie rule, so `audit_cell.py` flags their
  accuracy columns until the cells are rebuilt.
- 34 decisions have no published support (listed at the end of each part of `docs/DECISION-PROVENANCE.md`, each
  with a recommendation); the main ones are the per-family capacity grids and the feature fractions in `zoo.py`.
- Small quirks the reviewers left for you to decide: `o5_contrast.py` still rounds p(flood) to 9 decimals where it
  is only averaged; `pull_dem.py` keeps an unused imageserver route; a few module docstrings run past 15 lines.
