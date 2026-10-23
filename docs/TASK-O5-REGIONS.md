# Task: what the found regions are, and why the models fail there (O5, second round)

Brief for a Claude Code session in this repo, written 2026-10-17. It follows the full run of
2026-10-16 (`docs/RUN-W7-REPORT-2026-10-16.md`). That run treated O3 as met for this round. Its
boundary analysis of O5 found that the selected regions' Zone A edges look like the Zone A edges
around them. This brief runs six analyses (A–F) on the same regions. Each asks a different
question about why models trained on Zone AE fail there. Each has a primary statistic fixed
here, before anyone looks, and a stated reading for either outcome.

Read first:
1. `docs/RUN-W7-REPORT-2026-10-16.md`, sections O3–O5 and C5 (the regions and their recurrence).
2. `docs/PROJECT-PLAN.md` §0a and §3 (the outcomes) and §7 (limitations).
3. `docs/REGION-ANALYSIS-FLORIDA.md` §4 (Story 1 vs Story 2) and §8 (how to read outcomes together).
4. `docs/DATA-SUMMARY.md` §3 (the non-feature columns you will use as descriptors).

Victor comes from CS, not GIS. Explain GIS terms in CS terms in anything you write for him.

---

## 0. What is known

- The models use 7 features (`data/processed_w7/FEATURES.txt`): `landcover, slope, tri_300,
  wetindexcat, log10_dist_river, log10_dist_coast, hand`. The point table has 78 more columns.
  None of them is a model input, so all are available as descriptors (`DATA-SUMMARY.md` §3).
- Regions are the connected pieces (≥ 20 points) of the spatial OODSelect selection (μ = 3,
  k = 10, S = 500) on each seed's 10-nearest-neighbour graph:
  `results/2026-10-16/regions/FL_band_AE_s{0,1,2}_spatial_mu3_knn10_S500_{points.parquet,components.csv}`.
  That is 6 + 7 + 5 = 18 regions.
  - **Seed 0:** all six are Zone A floodplain points (positives) in north Florida. These are
    Big Bend (Dixie, Lafayette, Taylor; two pieces), Hamilton, Bradford (two pieces) and Gulf.
  - **Seeds 1 and 2:** they share two recurring regions, Collier/Hendry (about half positives)
    and Sarasota (83 and 128 points, only 2 and 7 positives).
  - **Seed 1 also has two regions on Lake Okeechobee open water** (tile 30_11), where `hand` and
    `wetindexcat` are missing.
- **There are two kinds of region**, and every analysis reports them separately:
  - *positive-dominated*: Zone A says flood, and the better Zone AE models increasingly say dry;
  - *negative-dominated*: land 0.5–5 km outside Zone A, where the better Zone AE models
    increasingly say flood.

  The first kind is "Zone A over-drawn, or the models miss floodplain". The second is "the map
  leaves out floodplain, or the models over-predict" (`REGION-ANALYSIS.md` R4).
- Boundary analysis (done): U(region) − U(matched Zone A) is +0.025 [−0.007, +0.058] at
  S = 250 and −0.059 at S = 500. The region edges are not special. That result stands; the
  analyses below ask other questions.

---

## 1. Rules

- Do not change `data/processed/`, `data/processed_w7/`, `data/interim/aotil/cells_w7/`, or
  anything under `results/2026-10-14/` or `results/2026-10-16/`.
- One new results date: `export D=<today>`, with outputs under `results/$D/o5/` and logs under
  `logs/$D/`.
- New populations go to new folders under `data/interim/aotil/`, one per condition (named below).
  Never write into `cells/` or `cells_w7/`.
- Every new script goes in `aotil/`, with a docstring that says what it computes and why, and a
  test or a smoke run. Add flags without changing defaults.
- **Fix every primary statistic and its predicted direction exactly as written here, before
  running the analysis.** If you must change one, say so in the report and give both versions.
- **Analysis points:** land only (NLCD `landcover` ≠ 11), unless an analysis says otherwise.
  Report how many region points this drops per region. The two Lake Okeechobee regions will
  mostly drop out; that is intended.
- **Uncertainty:** a bootstrap over regions (resample regions with replacement, 1,000 draws) for
  pooled numbers. Where an analysis compares regions with other places, also use the
  matched-growth null below.
- Use `.venv/bin/python`. Respect each **Stop**. Do not commit or push.

### Shared pieces (build once, used by B, C, E, F)

**S1. Terrain-matched controls: `aotil/match_controls.py`.** For each region point, pick one
unselected Zone A point that has:
- the same seed, the same 50 km test tile and the same label;
- the nearest terrain, by Euclidean distance on standardized `slope, tri_300, hand, curvature,
  tpi_300, tpi_broad`;
- no membership in that seed's spatial S = 500 or classic S = 500 selection.

Match without replacement, with a caliper of 0.25 standard deviations. Report:
- the matched share per region;
- balance: the standardized mean difference (SMD) of each terrain variable, which should be
  below 0.1.

**Stop** if fewer than 80% of a region's land points match, or any terrain SMD is ≥ 0.1 pooled.
Then report and propose a looser caliper.

(Updated 2026-10-21.) This Stop fired on 2026-10-17, and Victor changed both rules
(`results/2026-10-17/o5/PRIMARIES.md`, change log 2026-10-17 21:20): caliper **1.0**, and
balance judged as |mean difference| < 0.1 in units of the seed's SD on the pooled values;
per-region values are reported, not enforced. The 0.25 outputs are kept in
`results/2026-10-17/o5/S1_caliper0.25/`. The script defaults are still the brief's, so pass both
flags: `.venv/bin/python aotil/match_controls.py --out-dir results/$D/o5 --caliper 1.0
--balance-rule seed_sd`.

These controls make "beyond terrain" concrete: any difference between a region and its controls
is not a terrain difference.

**S2. Matched-growth null: `aotil/region_null.py`.** This implements the null specified in
`docs/REGION-ANALYSIS.md` (search "matched-growth"). Each draw:
1. picks a random unselected Zone A seed point in the same tile;
2. grows a connected set by breadth-first search on the same 10-NN graph to the region's size;
3. gets terrain-matched controls by S1.

Use 200 draws per region. The null's distribution of each statistic says how big a
region-vs-control difference arises for any connected patch of Zone A in that tile.
(Updated 2026-10-21: `aotil/region_null.py` also defaults to the brief's caliper 0.25; pass
`--caliper 1.0`, as the 2026-10-17 run did.)

---

## Part A. Is the inversion label-noise memorization? (hours)

**Question.** Sanyal et al. (2025, `refs/cited/sanyal2025.pdf`) get an inverse line when models
interpolate noisy labels. The paper's Related Work promises to test whether models memorising
noisy Zone AE training labels explain the inversion. (Updated 2026-10-21: this said "whether
lower-quality Zone A labels explain the inversion". The models train on Zone AE only, so this
part tests memorised noise in their own training labels; it cannot show whether the Zone A
evaluation labels are noisy.)

**Data.** `data/interim/aotil/cells_w7/FL_band_AE_s*.npz` and `*_models.csv`
(`acc_train_sub`, `acc_train_all`), plus the selections in `results/2026-10-16/oodselect_paper/`.

**Procedure.** For each seed, recompute the probit ID–OOD r on these subsets:
- the full spatial S = 500 selection;
- the full classic S = 500 selection;
- each region.

Compute it for these sets of models:
1. the paper's 60 test models;
2. the test models with `acc_train_sub` < 0.99, which do not interpolate their own training
   sample;
3. all models with `acc_train_sub` < 0.99;
4. all models with `acc_train_sub` < 0.95;
5. each family on its own (all models of that family).

Use `oodselect_paper.hard_corr` with masks, so the arithmetic matches the selections.

**Primary (fixed).** r on all non-interpolating models (`acc_train_sub` < 0.99), full spatial
S = 500 selection, pooled over seeds by averaging Fisher z. **Predicted: below −0.5.**

**Reading.**
- If the inversion holds among models that cannot memorize their labels, it is not Sanyal's
  label-noise mechanism. The Related Work sentence gets its answer.
- If it collapses to ≥ 0, the inversion needs interpolation, and memorised training-label noise
  becomes the leading explanation (updated 2026-10-21: this said "Zone A label noise").

**Output.** `results/$D/o5/A_interpolation.csv`.

---

## Part B. What differs beyond terrain? (half a day)

**Question.** O5 as written: do the regions differ from the Zone A around them, beyond terrain?
This describes each region by attributes, the way slice-discovery work does (Slice Finder, Chung
et al. 2019; Domino, Eyuboglu et al. 2022, `refs/cited/eyuboglu2022domino.pdf`). Unlike those,
it compares against terrain-matched controls and a spatial null.

**Data.** The point table, `data/interim/analysis/point_extras_FL.parquet`, the regions, and the
S1/S2 controls.

**Variables, in four groups:**
- **non-terrain model features:** `landcover` (as class shares), `wetindexcat`,
  `log10_dist_river`, `log10_dist_coast`;
- **river-network context:** `nearest_stream_order`, `log10_upstream_area`, `catch_area_km2`,
  `drainage_density`, `meta_nearest_ftype`;
- **soil, climate, rainfall:** the StreamCat soil/climate/runoff columns and the Atlas 14
  columns;
- **map provenance:** `meta_study_typ`, `meta_study_regime`, `meta_source_cit`,
  `meta_eff_date`, `meta_panel_scale`, `meta_bfe_km_per_km2`, `meta_poly_area_km2`,
  `meta_edge_dist_m` (how deep inside its polygon a positive sits), `dfirm_id`.

**Procedure.**
1. For each region and pooled, compute the SMD of every variable, region minus its matched
   controls. For categorical variables, use the difference in class shares.
2. Do positives and negatives separately.
3. For each, give a region bootstrap interval and the matched-growth null percentile.
4. Also describe what the models do there: the mean predicted p(flood) of the strongest quarter
   of test models on region points vs controls; and the error direction, i.e. the share of
   errors that are false negatives.

**Primary (fixed).** The SMD of `log10_dist_river`, region minus controls, pooled over seeds.
- **Predicted: positive for positive-dominated regions** (Zone A floodplain farther from mapped
  rivers than terrain-identical Zone A).
- **Predicted: negative for negative-dominated regions** (non-floodplain land closer to mapped
  rivers).
- Met if the pooled interval excludes 0 in the predicted direction and the null percentile is
  beyond 95 (or below 5).

**Reading.** A met primary supports this sentence: on the same terrain, the regions are where the
Zone A map and the river network disagree. That is O5's "beyond terrain", carried by the network
rather than by boundary tracing. List the other variables whose null percentile is beyond 97.5
or below 2.5 as secondary findings. With about 40 variables, say how many you would expect by
chance.

**Output.** `results/$D/o5/B_contrast.csv`, `B_balance.csv`, `B_null.csv`.

---

## Part C. Covariate shift or concept shift? (half a day to a day)

**Question.** Cai, Namkoong and Yadlowsky (2023, DISDE, arXiv:2303.02011) split a performance
drop into two parts:
- the inputs moved (covariate shift): the region looks unlike anything in Zone AE training;
- the input-to-label rule changed (concept shift): Zone AE training points that look the same
  carry the other label.

The two readings mean different things for the paper. A concept shift means the two procedures
label the same land differently. A covariate shift means detailed studies never covered land
like this.

**Data.** The cells' training rows (rebuild them with `aotil/zoo.py`'s `cell_data`, so the rows
are exactly the training set), the region points, the S1 controls, and Zone AE test points as
the calibration.

**Procedure.** Work in the standardized 7-feature space. Use the training median for missing
values, plus a missingness indicator for `hand` and `wetindexcat`.
- **C1, support.** Two measures:
  - the mean distance to the k = 25 nearest Zone AE training points;
  - a cross-fitted propensity: a classifier (gradient boosting, 5 folds) separating Zone AE
    training points from the evaluation set.

  Report both for regions, controls and Zone AE test points.
- **C2, label conflict.** Among the k = 25 nearest Zone AE training points, the share whose
  label differs from the point's own label. Zone AE test points give the baseline: how often
  Zone AE disagrees with itself. Use k = 10 and 50 as sensitivities.
- **C3, decomposition.**
  1. Take the strongest quarter of test models.
  2. Reweight Zone AE test points so their feature distribution matches the region's, using the
     propensity from C1 (clip weights at the 99th percentile).
  3. The drop from unweighted to reweighted Zone AE accuracy is the part a covariate shift
     explains. The rest of the drop down to region accuracy is the concept-shift part.
  4. Report both parts per seed with region-bootstrap intervals.

**Primary (fixed).** C2 label-conflict rate, region minus controls, pooled over seeds.
**Predicted: positive** (region points disagree with similar-looking Zone AE land more often than
terrain-matched Zone A does). Met if the interval excludes 0 and the matched-growth null
percentile is above 95.

**Reading.**
- **C2 met and C3 mostly concept:** in the regions, the two procedures give different labels to
  land the models cannot tell apart. That is label provenance in its strongest form.
- **C2 not met and C1/C3 mostly covariate:** the regions are land the Zone AE training set does
  not cover. That is Story 2, "where detailed studies were done", and also a provenance finding.

Either reading is reportable. Say which.

**Output.** `results/$D/o5/C_support.csv`, `C_conflict.csv`, `C_decomposition.csv`.

---

## Part D. Which feature carries the inversion? (D1 hours; D2 2–3 hours of compute)

**Question.** Salaudeen et al. (2025) explain accuracy-on-the-inverse-line by spurious
correlations: a feature that helps in-distribution and misleads out of distribution. Their method
cannot say which feature it is. Our population lets us test it.

**D1, a natural experiment in the existing population (no new training).** Each model saw a
random subset of the 7 features (`features` column of `*_models.csv`, from `feat_frac` 0.3 / 0.6 /
1.0), so feature inclusion is randomized across models. Fit, per seed:

probit(region accuracy) ~ probit(ID accuracy) + family + one indicator per feature
("this model saw f")

Region accuracy is the mean correctness over the region's land points, pooled over the seed's
regions and also per region. Fit it on the 60 test models (primary) and on all 300 (secondary;
partly circular, because the selection and validation models chose the regions). Report each
feature's coefficient with a bootstrap over models.

**D2, leave-one-feature-out populations.**
1. For each feature f and each seed, make `data/processed_w7_minus_<f>/` holding a
   `FEATURES.txt` without f and a relative link to `../processed/dataset_FL.parquet`.
2. Build the 300-model population with `aotil/zoo.py --data data/processed_w7_minus_<f> --out
   data/interim/aotil/cells_w7_minus_<f>`.
3. Audit each with `aotil/audit_cell.py`. **Stop** if one does not pass. (Updated 2026-10-21:
   populations built before 2026-10-21 fail its "table == matrices" check, because `zoo.py` now
   counts a negative predicted at exactly p = 0.5 as correct in the `acc_*` columns; rebuild them
   with `zoo.py --overwrite`. The scores S do not change.)
4. On the **fixed** masks (the regions and the full spatial and classic S = 500 selections from
   `results/2026-10-16/`), recompute the probit ID–OOD r on the paper's 60 test models. The same
   indices give the same specs.

That is 21 populations, about 1–3 minutes each.

**Primary (fixed).** In D2, Δr = r(without f) − r(with all 7) on the full spatial S = 500
selection, averaged over seeds.
- **Predicted: the largest Δr is for `log10_dist_river`**, with `log10_dist_coast` the
  alternative.
- Met if that feature's Δr is the largest of the seven and turns r above −0.3.

D1 is the supporting evidence: that feature's indicator should have a negative coefficient.

**Reading.** A met primary supports this sentence: the inverse line on these regions is carried
by distance to the mapped river network. Models that use it do better on detailed studies, which
FEMA carried out along mapped rivers, and worse on these Zone A regions. This is the explanation
OODSelect's authors could not give. If no single feature carries it, report the two-feature
combination with the largest Δr from D1's coefficients, without building more populations.

**Output.** `results/$D/o5/D1_feature_inclusion.csv`, `D2_lofo.csv`.

---

## Part E. Map-production signatures of the region polygons (half a day to a day)

**Question.** Steger et al. (2017, `refs/cited/steger2017.pdf`) show that a systematically
incomplete or differently made inventory creates relationships a model learns as if they were
real. Here we ask whether the Zone A polygons inside the regions were made differently from the
Zone A polygons around them. This is about how each polygon was drawn and revised, not where its
edge runs.

**Data.**
- Lobes, i.e. single-part polygons with role `region`, `matched_A` or `AE`: they exist for seed
  0 (S = 250, 500) and the recurring seed-1/2 regions in `results/2026-10-16/boundary/*_lobes.parquet`.
  Build the other regions' lobes with `aotil/lobes.py` (polygons only, no downloads).
- `data/interim/nfhl_haz_attrs_10state.parquet` (`STUDY_TYP, SOURCE_CIT, ZONE_SUBTY, DFIRM_ID`,
  joined on `FLD_AR_ID`).
- `data/interim/nfhl_states/pan_FL.parquet` (`EFF_DATE, PRE_DATE, SCALE`).
- `data/interim/nfhl_states/s_lomr_FL.parquet`. A LOMR is a Letter of Map Revision, a later,
  local correction of the map.

**Per lobe:**
- area and perimeter;
- **vertex spacing**: perimeter divided by the number of vertices in the original polygon,
  not densified. Coarse spacing suggests hand digitizing from a small-scale paper map;
- the share of edge length in straight runs: turning angle < 1° sustained over ≥ 300 m;
- the perimeter–area shape index 2·ln(P/4)/ln(A);
- `STUDY_TYP`, `SOURCE_CIT`;
- the effective date, pre-FIRM date and scale of the panels it lies on;
- the share of its area inside any LOMR polygon.

**Primary (fixed).** Median vertex spacing, region lobes minus matched Zone A lobes, pooled by
region bootstrap. **Predicted: positive** (region polygons were digitized more coarsely). Report
Zone AE lobes as the contrast.

**Reading.**
- If met: the regions sit in Zone A polygons made by a coarser procedure than the Zone A around
  them, a signature of how the map was produced.
- If not met, together with the boundary result: the regions' polygons are ordinary Zone A, and
  the explanation lies in B, C or D.

Victor will read the Flood Insurance Study reports for the region counties by hand. List the
counties and DFIRM ids in the report so he knows which to fetch.

**Output.** `results/$D/o5/E_polygons.csv` (one row per lobe), `E_summary.csv`.

---

## Part F. What does pooling these labels do? (a day)

**Question.** Prior work pools Zone A and Zone AE into one label (Woznicki et al. 2019; Wu et al.
2026; Related Work paragraph 1). If the region labels follow a different rule, adding them to
training should hurt accuracy on detailed studies more than adding the same number of ordinary
Zone A labels. That would be Story 1, a label conflict. If adding them does not hurt Zone AE but
helps on other Zone A, pooling is safe there, which is Story 2.

**Code.** Add `--extra-train <file of point_ids>` to `aotil/zoo.py` (default: none). It appends
those rows, whatever their domain, to the training set of every model, and writes the list into
the cell. Everything else stays the same.

**Conditions per seed** (300 models each, same specs):
1. Zone AE only: the existing `cells_w7`, so no new build;
2. **+region**: plus the seed's region land points;
3. **+matched**: plus their S1 controls, the same count with the same labels;
4. as a sensitivity, 2 and 3 with each added row repeated 10 times (`--extra-train-repeat 10`).

Output folders: `data/interim/aotil/cells_w7_pool_{region,matched}[_x10]/`.

**Evaluation.** Done afterwards, from the stored point ids, so the cells need no other change.
1. Zone AE test points at least 5 km from any added point.
2. Zone A test points at least 5 km from any added point and outside every region.

The added points themselves are never scored.

**Primary (fixed).** The paired per-model difference in Zone AE AUC (evaluation 1), +region
minus +matched, averaged over 300 models and three seeds, with a test-tile bootstrap.
**Predicted: negative.**

**Reading.**
- Negative with the interval excluding 0: the region labels conflict with the detailed-study rule,
  and pooling them, as prior work does, costs detailed-study accuracy.
- Near 0, with evaluation 2 improving: pooling these labels is harmless or helpful.

Effects may be small (the regions add about 400 points per seed to about 27,000). Report the
×10 sensitivity beside the primary, and say so if both are within ±0.002.

**Output.** `results/$D/o5/F_pooling.csv`.

---

## Part G. The report: `docs/O5-REGIONS-REPORT-$D.md`

Lead with one line per part (A–F): the primary statistic with its interval and null
percentile, whether the prediction was met, and the one-sentence claim it supports. Then the
tables, every number with its file path.

Close with:
1. **A proposed new wording of O5** that the results support, and a one-paragraph summary that
   ties together the boundary result of 2026-10-16 and A–F. Mark both as proposals for Victor; do
   not edit the paper.
2. Which region types (positive- or negative-dominated) and which regions (recurring or not)
   drive each result.
3. Every deviation from this brief, with the reason.
4. The code you added or changed.

Tell Victor the date `D` and the report path when you finish or reach a stop.
