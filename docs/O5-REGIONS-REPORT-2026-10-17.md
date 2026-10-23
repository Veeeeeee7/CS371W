# O5, second round: what the found regions are, and why the models fail there. Report, D = 2026-10-17

Brief: `docs/TASK-O5-REGIONS.md`. Outputs: `results/2026-10-17/o5/`. Logs: `logs/2026-10-17/`. The
primary statistics were copied verbatim into `results/2026-10-17/o5/PRIMARIES.md` before anything
was computed. Its change log records the one decision taken since (the S1 caliper) and one later
correction to how that entry described the terrain spread. A glossary of the map terms, in CS
terms, is at the end (§5).

**Status: done.** One Stop fired, at S1; Victor decided on 2026-10-17 at 21:20 EDT. The remaining
steps ran from 03:00 EDT on 2026-10-18, at his request. Nothing is committed.

How the work was checked:
- Every part was built by one agent and checked by an independent one that recomputed the
  numbers without importing the module.
- The verifiers' findings were fixed and checked again, including mutation tests: deliberately
  broken copies of the code must fail the tests.
- The numbers in this report were then checked against the files.

## One line per part

| part | primary [95% interval]; null percentile | prediction | claim it supports |
|---|---|---|---|
| **A** label-noise memorization | r on non-interpolating models, full spatial S = 500, pooled: **−0.817 [−0.839, −0.793]** (held-out test models only: −0.808) | below −0.5: **met** | The inversion is not the label-noise interpolation of Sanyal et al.: models that cannot memorize their labels invert just as strongly. |
| **B** what differs beyond terrain | SMD of `log10_dist_river`, region − terrain-matched controls. Positive-dominated: **−0.058 [−0.346, +0.212]**, null 33. Negative-dominated: **−0.181 [−0.720, +0.0008]**, null 1.0 | + / −: **not met** for either type (the negative one misses excluding 0 by 0.0008) | On the same terrain, the regions are not where the Zone A map and the river network disagree. What does differ is land cover (positive-dominated regions: forest where matched Zone A is wetland). |
| **C** covariate vs concept shift | label conflict (k = 25), region − controls, pooled: **+0.124 [+0.092, +0.170]**; null **100** | positive: **met** | In the regions, similar-looking Zone AE land carries the other label. The drop is mostly concept shift (0.29 of 0.45), so this is label provenance, not missing coverage. |
| **D** which feature carries it | D2: Δr from removing a feature, full spatial S = 500, seed mean. Largest is **landcover +0.073** (r −0.819 → −0.746); river +0.027 | river largest and r > −0.3: **not met** | No single feature carries the inversion. Removing any one leaves the seed-mean r at −0.746 or below. Land cover carries most; the river distance carries part of it on the regions themselves (+0.081) and in seeds 1–2. |
| **E** map-production signatures | median vertex spacing, region − matched Zone A lobes: **−5.99 m [−6.98, +0.40]**; null **0.0** | positive: **not met**, with the opposite sign | The region polygons are not cruder digitizations. Their vertices are closer together than in the Zone A around them (and farther apart than in Zone AE). The interval includes 0; the reversal rests on the null. |
| **F** pooling the labels | Zone AE AUC, +region − +matched, 900 model pairs: **−0.0018 [−0.0032, −0.0007]**; ×10: −0.0123 [−0.0174, −0.0069] | negative: **met** (small) | Adding the region labels to training costs detailed-study accuracy more than adding as many terrain-matched ordinary Zone A labels (Story 1), and gains nothing on other Zone A. |

The sections below give every number with its file. The closing sections give the proposed O5
wording, which regions drive each result, the deviations from the brief, and the code.

---

## Shared regions table

`results/2026-10-17/o5/regions_land.csv`. There are 18 regions: components of ≥ 20 points of the
spatial S = 500 selection, 6 / 7 / 5 per seed. Land only means NLCD land-cover code ≠ 11, i.e.
not open water.

Points dropped by the land rule:
- seed 1 c3 and c4, the two Lake Okeechobee regions: all of their 35 and 32 points, as the brief
  intends;
- seed 1 c0: 1 point;
- seed 2 c1: 1 point;
- every other region: none.

That leaves 16 regions with land points:
- **11 positive-dominated** (≥ 50% of land points are Zone A positives), 681 land points. These
  are all six seed-0 regions, seed 1 c0, c2, c5 and seed 2 c3, c4.
- **5 negative-dominated**, 433 land points: seed 1 c1, c6 and seed 2 c0, c1, c2.

(An earlier version of this report said 10 and 6; the member lists were right.)

Collier/Hendry, the recurring region, has 56% positives in seed 1 (positive-dominated) and 46% in
seed 2 (negative-dominated).

## Shared pieces S1 and S2: terrain-matched controls and the matched-growth null

Code: `aotil/match_controls.py` (S1), `aotil/region_null.py` (S2), test
`aotil/test_match_controls.py` (48 checks pass). Outputs at the chosen caliper 1.0:
`results/2026-10-17/o5/S1_controls.csv`, `S1_match_summary.csv`, `S1_balance.csv`, `S2_null.parquet`.
Outputs at the brief's caliper 0.25, with the caliper sensitivity:
`results/2026-10-17/o5/S1_caliper0.25/`. An independent verifier reproduced
every match, and every SMD, from scratch.

**What S1 does.** Each region land point gets one control: an unselected Zone A point in the
same seed, 50 km tile and label, with the most similar terrain. Similarity is Euclidean distance
on six terrain variables (slope, tri_300, hand, curvature, tpi_300, tpi_broad), each divided by
its standard deviation over the seed's land Zone A test points. Matching is without replacement,
by optimal assignment within each tile and label group. A pair is allowed only if the distance is
≤ 0.25, the brief's caliper.

**At the brief's caliper (0.25) the Stop fired on both conditions.** Those outputs are kept in
`results/2026-10-17/o5/S1_caliper0.25/`.

1. **Matched share.** 14 of the 16 regions match fewer than 80% of their land points. Pooled:
   674 of 1,114 = 60.5%.
   - 427 points have no candidate within 0.25.
   - 13 lost their candidate to another region point.
   - The worst regions: seed 2 c3 at 2 of 36 (0.06), seed 2 c2 at 0.29, seed 2 c4 at 0.29,
     seed 1 c2 at 0.31.
2. **Balance.** The pooled SMD after matching is 0.124 for `tri_300`, over 0.1. (SMD is the mean
   difference divided by the pooled within-group SD.) Per region it reaches 0.55, and 1.13 for
   seed 2 c3 on 2 pairs.

In the table, "all pairs" is the pooled scope over every pair. "Any pooled subset" is the largest
value over the pooled scopes: all pairs, by region type, and by label.

| caliper | pooled matched share | regions < 80% | largest pooled SMD after (all pairs) | largest pooled mean difference after, seed-SD: all pairs / any pooled subset | largest per-region mean difference, seed-SD |
|---|---:|---:|---:|---:|---:|
| **0.25** (brief) | 0.605 | 14 | 0.124 | 0.012 / 0.020 | 0.087 |
| 0.5 | 0.880 | 3 (s1c6 0.64, s2c3 0.47, s2c4 0.71) | 0.126 | 0.023 / 0.027 | 0.115 (s2c3) |
| 0.75 | 0.942 | 1 (s2c3 0.67) | 0.119 | 0.027 / 0.036 | 0.070 |
| **1.0** | **0.966** | **0** (lowest: s2c0 0.86) | 0.120 | **0.029 / 0.037** | 0.138 (s2c3) |
| none | 0.984 | 0 | 0.126 | 0.038 / 0.055 | 0.157 (s1c0) |

Files:
- matched shares and SMDs for every caliper: `S1_caliper0.25/S1_caliper_sensitivity.csv`;
- mean differences in seed-SD units (column `diff_after_seed_sd`):
  - `S1_caliper0.25/S1_balance.csv` (0.25);
  - `S1_balance.csv` (1.0);
  - `S1_caliper_sensitivity/S1_caliper{0.5,0.75,1000}_balance.csv` (0.5, 0.75, none). These
    were rerun on 2026-10-18 so that every row cites a file; caliper 1000 is "none". Their
    matched shares and SMDs equal the sensitivity file's.

**Why the SMD rule cannot be met, at any caliper.**
- The matched sets are terrain-homogeneous. At 0.25, over all pairs pooled, the spread of the
  matched region points and their controls is only 0.08–0.31 of the seed's SD, depending on the
  variable (`sd_matched_seed_sd` in `S1_caliper0.25/S1_balance.csv`). At 1.0 it is 0.15–0.50.
  (An earlier version called this "the regions' own SD". It is the matched sets' SD.)
- The SMD divides by that small spread, so a mean gap of 0.01 seed-SD becomes an SMD of 0.12.
- In the units the matching uses (seed SD), regions and controls differ by at most 0.012 over all
  pairs at 0.25 (0.020 in any pooled subset), and 0.029 at 1.0 (0.037).

**The low match rate is not special to the regions.** S2's random null patches match just as
badly at 0.25: 0.51 per patch, 0.57 weighted by points. Each region's matched share lies inside
its null's 5–95% range, except two:
- seed 1 c0 is harder to match than any patch;
- seed 2 c0 is easier than 98.5% of patches.

A 0.25 Euclidean ball over 6 standardized variables is simply very small: a median of 0–2
candidates per point in most regions. The common per-variable reading of "0.25 SD" (every
variable within 0.25 SD) also fails: pooled 0.787, with 7 regions under 80%.

**S2 at caliper 0.25** (superseded; kept in `S1_caliper0.25/`).
- 200 draws per region, 3,600 in total, 27 s.
- Each draw starts at a random unselected land Zone A point in the region's tile(s). It grows by
  breadth-first search through unselected points on the same 10-nearest-neighbour graph to the
  region's size, then gets S1 controls.
- 187 redraws, 155 of them for seed 2 c0.
- One weakness: seed 2 c0's patches leave its tile. On average 26% of each patch's points lie
  outside tile 30_9, so that null is only partly tile-matched. The other regions leak less than
  8%.
- **S2 has to be rerun at whatever caliper is chosen.** That takes about 30 s.

### Decision and rerun at caliper 1.0

- **Decision (Victor, 2026-10-17 21:20 EDT).** Caliper 1.0. Balance is judged as the mean
  difference in seed-SD units, below 0.1. Recorded in PRIMARIES.md's change log.
- **S1 at 1.0** (`aotil/match_controls.py --caliper 1.0 --balance-rule seed_sd`; the flag was
  added for this, and its default is the brief's SMD rule).
  - Every region matches ≥ 80% of its land points: pooled 1,076 of 1,114 = 96.6%, lowest seed 2
    c0 at 86.3%.
  - The pooled terrain mean difference is at most 0.029 seed-SD over all pairs and 0.037 in any
    pooled subset, so the Stop does not fire.
  - Per region, three values in two regions are just over 0.1: seed 2 c3 (`tri_300` 0.138 and
    `slope` 0.122, on 34 pairs) and seed 1 c6 (`tpi_broad` 0.110, on 21 pairs). The proposal
    mentioned only seed 2 c3.
  - Files: `results/2026-10-17/o5/S1_controls.csv`, `S1_match_summary.csv`, `S1_balance.csv`; log
    `logs/2026-10-17/S1_match_controls.log`.
- **S2 at 1.0** (`aotil/region_null.py --caliper 1.0`): 200 draws × 18 regions in 20 s, the same
  seed-point and patch rules as before. Output `results/2026-10-17/o5/S2_null.parquet`, log
  `logs/2026-10-17/S2_region_null.log`.
- **Tests:** all 48 checks pass on the caliper-1.0 outputs
  (`logs/2026-10-17/S1S2_test_caliper1.log`; the test gained a `--caliper` argument).


---

## Part A. Is the inversion label-noise memorization?

Code: `aotil/o5_interpolation.py` (test `aotil/test_o5_interpolation.py`, 38 checks pass; log
`logs/2026-10-17/A_interpolation_test.log`). Output: `results/2026-10-17/o5/A_interpolation.csv`,
871 rows. Log: `logs/2026-10-17/A_interpolation.log`. An independent verifier recomputed every
number in float64 without importing the module.

**What "interpolating" means.** A model *interpolates* when it fits its own training sample almost
perfectly. Here that is `acc_train_sub` ≥ 0.99, accuracy on the rows it was trained on.
Sanyal et al. get an inverse line only from models that memorize noisy labels. So if the
inversion survives among models that cannot memorize, it is not their mechanism.

Interpolating models per seed (0/1/2): 29 / 26 / 27 of 300, and 9 / 9 / 8 of the 60 test
models. Only et (11/11/11), rf (12/10/12), xgb (5/4/3) and hgb (1/1/1) ever interpolate;
dt, lr, mlp and nb never do.

**Primary:** r = **−0.817 [−0.839, −0.793]**. Per seed: −0.824, −0.827, −0.798, on 271 / 274 /
273 models. **Below −0.5: met.**

Pooled over seeds (tanh of the mean Fisher z):

| subset | all 300 | 60 test | test, < 0.99 (held out) | all, < 0.99 (**primary set**) | all, < 0.95 |
|---|---:|---:|---|---:|---:|
| spatial S = 500, all points | −0.823 | −0.820 | −0.808 [−0.858, −0.744] | **−0.817** | −0.814 |
| spatial S = 500, land only | | | −0.794 | −0.811 [−0.833, −0.786] | |
| classic S = 500, all points | −0.921 | −0.926 | −0.925 [−0.946, −0.898] | −0.921 | −0.921 |
| all regions | −0.703 | −0.693 | −0.677 [−0.755, −0.578] | −0.690 | −0.679 |
| positive-dominated regions, land | −0.710 | −0.706 | −0.696 [−0.771, −0.602] | −0.709 | −0.707 |
| negative-dominated regions, land (seeds 1–2) | −0.058 | −0.024 | **+0.006 [−0.191, +0.202]** | −0.027 | +0.008 |

Each family on its own (all models of the family), spatial S = 500, pooled:

| dt | et | hgb | lr | mlp | nb | rf | xgb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −0.795 | −0.824 | −0.811 | −0.758 | −0.766 | −0.522 | −0.859 | −0.841 |

Reading:
- The inversion holds among models that cannot memorize their labels. On the held-out test
  models it is −0.808, and the four families that never interpolate are all inverted (−0.52 to
  −0.79). **The Sanyal label-noise mechanism does not explain it.** That answers the Related
  Work sentence.
- **Circularity.** The primary model set includes about 220 of the 240 selection and validation
  models, whose r the optimiser drove down when it chose the selection. The primary is therefore
  mostly in-sample. The held-out set (test models with `acc_train_sub` < 0.99) gives the same
  answer.
- **Which regions carry it.** The positive-dominated regions are inverted (−0.70). The
  negative-dominated regions are not, across the population: +0.006 on held-out models. Within
  each family they are inverted (−0.30 to −0.71), except nb at +0.41. nb has both the lowest ID
  accuracy and the lowest accuracy on these regions, and that between-family gap cancels the
  within-family slopes.
- Small all-positive regions put some models' subset accuracy at exactly 0 or 1, where the
  probit clamp acts (up to 74 of 271 models on one seed-0 region). Read per-region r with the
  `n_clamped` column.
- The pooled interval treats seeds as independent. They share points and model specs, so the
  interval is optimistic.

## Part B. What differs beyond terrain?

Code: `aotil/o5_contrast.py`, test `aotil/test_o5_contrast.py`. Outputs:
`results/2026-10-17/o5/B_contrast.csv`, `B_balance.csv`, `B_null.csv`. Log:
`logs/2026-10-17/B_contrast.log`.

**Method.** Every region land point that has an S1 control forms a pair: 1,076 pairs in 16 regions
(11 positive-dominated, 667 pairs; 5 negative-dominated, 409 pairs). For each variable the
contrast is the SMD, region minus control: the mean difference divided by the pooled standard
deviation of the two sets. For a category it is the difference in class shares, plus one
total-variation distance per variable (the share of the category mix that differs).

The interval is a bootstrap over regions. The null percentile is where the observed value falls
among 200 matched-growth null draws. Each draw pools, for every region, one random connected patch
of unselected Zone A the same size in the same tile, with its own terrain-matched controls.

Terrain balance of the pairs used (`B_balance.csv`), under the balance rule chosen at the S1 Stop:
- at most 0.069 seed-SD in any pooled subset;
- at most 0.138 per region. Three per-region values, in two regions, exceed 0.1: seed 2 c3
  (`tri_300` 0.138, `slope` 0.122) and seed 1 c6 (`tpi_broad` 0.110).

**Primary: SMD of `log10_dist_river`, region minus controls, pooled over seeds.**

| region type | SMD [95% over regions] | null percentile | prediction | terrain-adjusted SMD [95%], null % |
|---|---|---|---|---|
| positive-dominated (11 regions, 667 pairs) | −0.058 [−0.346, +0.212] | 33.0 | > 0: **not met** | +0.004 [−0.307, +0.277], 45.5 |
| negative-dominated (5 regions, 409 pairs) | −0.181 [−0.720, +0.0008] | 1.0 | < 0: **not met** (the interval misses excluding 0 by 0.0008) | −0.145 [−0.719, +0.138], 2.5 |

Notes:
- The negative-dominated result meets the null part (1.0 < 5) but misses the interval part by
  0.0008. The exact bootstrap over all 3,125 resamples of 5 regions has 3.9% at or above 0, so
  the verdict does not depend on the random seed (verifier script
  `results/2026-10-17/o5/verification/B_verify.py`, output `B_verify.out`).
- One region pulls against the prediction. Collier seed 2 c0 is +1.67; the other four
  negative-dominated regions are −0.56 to −0.76.
- Positive-dominated regions split by sign, from −0.78 to +0.92.

Per-region SMD, all labels:

| region | s0 c0 | s0 c1 | s0 c2 | s0 c3 | s0 c4 | s0 c5 | s1 c0 | s1 c1 | s1 c2 | s1 c5 | s1 c6 | s2 c0 | s2 c1 | s2 c2 | s2 c3 | s2 c4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| type | pos | pos | pos | pos | pos | pos | pos | neg | pos | pos | neg | neg | neg | neg | pos | pos |
| SMD | −0.62 | +0.28 | −0.78 | −0.13 | +0.74 | +0.67 | +0.92 | −0.76 | +0.41 | +0.01 | −0.73 | +1.67 | −0.56 | −0.59 | +0.59 | +0.20 |

**Secondary findings** (pooled, null percentile above 97.5 or below 2.5):
- 28 of 242 variable-and-scope tests fire, against 12.1 expected by chance; 16 of the 28 also have
  an interval excluding 0.
- 51 of 207 class-share rows fire, against 10.4 expected.
- The clearest:
  - **Positive-dominated regions, land cover:** total-variation distance 0.336 [0.213, 0.450],
    null 100. Region positives are 32% woody wetland against 67% for their terrain-matched
    controls, and 39% evergreen forest against 17%.
  - **Negative-dominated regions, `drainage_density`:** +0.397 [+0.147, +0.753], null 100. This
    is the density of mapped streams in the catchment.
  - **Negative-dominated regions, `growth_100_2`:** −0.091 [−0.296, −0.025], null 0. This is the
    ratio of 100-year to 2-year rainfall.
  - **Positive-dominated negatives, `meta_bfe_km_per_km2`:** +1.32 [+0.17, +2.02], null 99, on 55
    pairs. This is the density of base-flood-elevation lines, i.e. the detail of the map study.
  - **Negative-dominated positives, `meta_edge_dist_m`:** −0.442 [−1.053, −0.299], null 0.
    Region positives sit 3.3 km inside their Zone A polygon against 5.2 km for controls. This
    scope has 66 pairs, 50 of them Collier.
- 6 of the 28 are "more alike than the null": a total-variation percentile below 2.5.

**What the models do there** (the strongest quarter of the 60 test models, 15 by ID AUC; region vs
controls):

| | positive-dominated | negative-dominated |
|---|---|---|
| mean predicted p(flood) | 0.460 vs 0.591: −0.131 [−0.182, −0.088], null 0 | 0.735 vs 0.646: +0.089 [+0.057, +0.122], null 100 |
| accuracy | 0.413 vs 0.662, null 0 | 0.239 vs 0.366, null 0 |
| share of errors that are false negatives | 0.913 vs 0.857 | 0.044 vs 0.054 |

In positive-dominated regions the strong models call Zone A floodplain dry more often than on
terrain-identical Zone A elsewhere. In negative-dominated regions they call land outside Zone A
wet more often.

**Reading.**
- The predicted network signature is not there. On the same terrain, the regions are not
  systematically farther from mapped rivers (positive-dominated) or closer to them
  (negative-dominated) than ordinary Zone A. Only the negative-dominated null percentile points
  the predicted way, and its interval just reaches 0.
- What does differ beyond terrain is **land cover** in the positive-dominated regions. They are
  evergreen-forest Zone A where terrain-matched Zone A is mostly woody wetland, and the models
  call them dry. That fits Part D's finding that landcover is the largest single carrier of the
  inversion.
- The negative-dominated regions sit in denser mapped stream networks.

## Part C. Covariate shift or concept shift?

Code: `aotil/o5_shift_decomp.py`, test `aotil/test_o5_shift_decomp.py`. Outputs:
`results/2026-10-17/o5/C_support.csv`, `C_conflict.csv`, `C_decomposition.csv`. Log:
`logs/2026-10-17/C_shift_decomp.log`.

**The question in CS terms.** A model trained on Zone AE can fail on a region for two reasons:
- **covariate shift:** the region's inputs look unlike anything in the training set;
- **concept shift:** training points that look the same carry the other label.

**Feature space.** Each seed's exact Zone AE training rows (rebuilt with `zoo.cell_data`), the 7
features plus missing-value flags for `hand` and `wetindexcat`, all standardized by the training
statistics. `landcover` is kept as the numeric code the models see.

**Primary: C2 label conflict.** For each point, take its 25 nearest Zone AE training points and the
share of them with the other label. The statistic is region minus controls, pooled over the 1,076
pairs of all seeds.

**+0.1245 [+0.0916, +0.1700], null percentile 100. Predicted positive: met.**
- The null's 97.5th percentile is +0.068.
- Terrain-adjusted: +0.122 [+0.086, +0.168], null 100.

| scope | region − controls | null % |
|---|---|---|
| k = 10 / k = 50 | +0.133 / +0.117 | 100 / 100 |
| positive-dominated | +0.146 [0.106, 0.210] | 100 |
| negative-dominated | +0.089 [0.041, 0.127] | 99.5 |
| seed 0 / 1 / 2 | +0.148 / +0.135 / +0.089 | 91.5 / 100 / 100 |
| each of the 16 regions | +0.016 to +0.380, all positive | |

Levels (share of the 25 neighbours with the other label):
- regions 0.629 over all their land points, 0.636 over the 1,076 that have a control;
- terrain-matched controls 0.512, so the primary is 0.636 − 0.512 = 0.124;
- Zone AE test points 0.256, which is how often Zone AE disagrees with itself (0.309 in the
  regions' own tiles).

**C1, support.** Distance to the 25 nearest Zone AE training points, in standardized units:
- regions 0.534 over all their land points (0.521 for those with a control), controls 0.461,
  Zone AE test points 0.712;
- region − controls (0.521 − 0.461) +0.060 [0.036, 0.084], null 99.5.

So the regions are not out of the training set's support. They lie closer to Zone AE training land
than held-out Zone AE land does. A cross-fitted classifier separating training points from the
evaluation points is mostly geographic (AUC 0.93–0.95), so its propensities (region 0.719, controls
0.607, Zone AE 0.626) are read only as a comparison between groups.

**C3, decomposition** (the 15 strongest test models):

| | covariate part | concept part |
|---|---|---|
| **pooled** (accuracy drop 0.454 [0.377, 0.529]) | 0.164 [0.104, 0.205] | **0.290 [0.216, 0.372]** |
| seed 0 / 1 / 2 | 0.050 / 0.177 / 0.265 | 0.309 / 0.279 / 0.280 |
| positive- / negative-dominated | 0.097 / 0.268 | 0.333 / 0.326 |

Pooled accuracies: Zone AE land 0.805; Zone AE reweighted to look like the regions 0.641; regions
0.352. The covariate part is 36% of the drop.

**Caveats**, all reported by the implementer and the verifier:
1. **C2 is partly built in.** The regions were chosen where the better models are wrong, and those
   models learned Zone AE's local labels. The evidence that does not depend on the selection is the
   *controls'* conflict: 0.512 against Zone AE's own 0.26. Unselected Zone A on the same terrain
   already disagrees with similar-looking Zone AE twice as often as Zone AE disagrees with itself.
   The regions are where that disagreement is strongest.
2. **C3 has poor overlap.** The regions are almost perfectly separable from Zone AE test land
   (AUC 0.98–1.00). The 99th-percentile clip binds and carries 60–71% of the weight, and the
   effective sample size is 136–176. So the covariate part is a **lower bound** and the concept
   part absorbs what the weights cannot reach. With another fold seed the split moves to
   0.175 / 0.278. Both numbers come from the verifier's scripts, saved with their outputs in
   `results/2026-10-17/o5/verification/` (`C_v_clipmass.py`, `C_v_c3.py`).
3. Two cross-check estimators give the same split: 0.133 / 0.321 and 0.147 / 0.306.

**Reading.** The brief's first reading holds: C2 is met, and C3 is mostly concept. In the regions,
the two procedures give different labels to land the models cannot tell apart. In the brief's
words that is label provenance in its strongest form. The covariate part is real but secondary,
and larger in the negative-dominated regions and in seeds 1–2.

## Part D. Which feature carries the inversion?

### D1. Natural experiment in the existing population (supporting)

Code: `aotil/o5_feature_inclusion.py` (test `aotil/test_o5_feature_inclusion.py`, 44 checks pass).
Output: `results/2026-10-17/o5/D1_feature_inclusion.csv`, plus `_seen.csv` and `_corr.csv`. Log:
`logs/2026-10-17/D1_feature_inclusion.log`. An independent verifier reproduced every coefficient
to 1e-6.

**The idea.** Each model saw a random subset of the 7 features (3, 4 or 7 of them). So "did this
model see feature f" is randomized, as in an experiment. Per seed, I fit OLS:

probit(region accuracy) ~ probit(Zone AE accuracy) + family + one 0/1 indicator per feature.

A negative coefficient means that models which saw f do worse on the regions than equally good
models which did not.

Response: all region land points of the seed together. Primary fit on the 60 test models;
bootstrap 95% over models (936–945 of 1,000 draws usable, because some resamples lose the lr
family, which has only 3 test models).

| feature | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| landcover | −0.461 [−1.286, +0.080] | −0.087 [−0.195, +0.033] | −0.039 [−0.113, +0.033] |
| slope | +0.301 [−0.151, +0.889] | −0.015 [−0.115, +0.113] | +0.040 [−0.015, +0.099] |
| tri_300 | +0.623 [+0.078, +1.184] | +0.001 [−0.101, +0.111] | −0.057 [−0.117, +0.002] |
| wetindexcat | +0.332 [−0.480, +0.931] | +0.119 [+0.014, +0.245] | −0.022 [−0.085, +0.053] |
| **log10_dist_river** | +0.209 [−0.358, +0.779] | **−0.213 [−0.361, −0.066]** | **−0.109 [−0.181, −0.048]** |
| log10_dist_coast | −0.104 [−0.504, +0.365] | +0.068 [−0.023, +0.170] | +0.041 [−0.015, +0.089] |
| hand | −0.518 [−1.255, +0.195] | −0.045 [−0.161, +0.061] | +0.020 [−0.043, +0.079] |

Reading:
- In seeds 1 and 2, `log10_dist_river` has the most negative coefficient, and its interval
  excludes 0. On the full spatial S = 500 selection (D2's mask) the effect is −0.083 [−0.167,
  −0.002] in seed 1, a borderline interval, and −0.066 [−0.111, −0.027] in seed 2.
- **Seed 0 shows no river effect** under any treatment of the probit clamp:
  - +0.209 with the brief's clamp;
  - +0.072 with a continuity correction that keeps all models (verifier script
    `verification/D1_verify3.py`, output `D1_verify3.out`);
  - −0.046 [−0.399, +0.371] after dropping the 2 test models at accuracy 0 or 1. That last
    variant selects models on their outcome, so it is shown only for completeness.
- Seed 0's most negative feature changes with the treatment: hand, coast or landcover.
- Seed 0 is all positive-dominated regions in north Florida. Seeds 1–2 contain the recurring
  southern regions and all five negative-dominated ones.
- Averaged over seeds, the most negative feature is landcover (−0.196 [−0.474, −0.014]), driven
  by seed 0. The total-effect sensitivity, the same fit without probit(Zone AE accuracy), also
  points at landcover in seed 0 and on average, and at river in seeds 1–2.
- On all 300 models (partly circular), river is the most negative on average: −0.108
  [−0.155, −0.060].
- The indicators are only mildly correlated with each other: VIF 1.2–2.2. Each feature was seen
  by 57–83% of the test models; the design share is 2/3.

So D1 supports D2's predicted feature in seeds 1–2 and not in seed 0.

### D2. Leave-one-feature-out populations (primary)

For each of the 7 features and each seed, a 300-model population was built from the same model
specs as `cells_w7`, without that feature:
- inputs `data/processed_w7_minus_<f>/`: an 8-line `FEATURES.txt` (2 comment lines and the 6
  features) plus a relative link to the same dataset;
- cells `data/interim/aotil/cells_w7_minus_<f>/`.

All 21 populations pass `aotil/audit_cell.py`, 0 fail and 0 warn (`results/2026-10-17/o5/D2_audit/`,
logs `logs/2026-10-17/D2/`), so the D2 Stop did not fire. Each build took about 20 s (runner
`logs/2026-10-17/D2F_runs.sh`, log `D2F_runs.log`).

Model i keeps its family, hyperparameters, feature and row fractions, and random seed in every
population. It is not, though, the same model with one column deleted:
- Each model draws its feature subset and then its training-row subsample from its seed
  (`zoo.fit_one`). With 6 features instead of 7 both draws change.
- The feature subset is redrawn. Models with feature fraction 0.6 get 3 features instead of 4,
  and only 100–174 of the 300 models per seed end up with exactly their old subset minus f.
- 113–133 of the 300 models per seed get a different training-row subsample of the same size.
  The rest use all rows, or happen to draw the same rows.

So Δr measures removing f *together with* redrawing these subsets. The per-seed spread in the
table (for example tri_300: +0.061 in seed 1, −0.058 in seed 2) is of the same order as most
effects, so only landcover's lead is clear.

The masks are the fixed ones from 2026-10-16, so no selection is rerun. The statistic is
the paper's probit ID–OOD r on the 60 test models. With all seven features it reproduces the
stored r_test exactly.

Code: `aotil/o5_lofo.py`; test `aotil/test_o5_lofo.py`, 42 checks pass, among them the points and
specs matching `cells_w7` and no model seeing the dropped feature. Output:
`results/2026-10-17/o5/D2_lofo.csv`; log `logs/2026-10-17/D2_lofo.log`.

Δr = r(without f) − r(all 7); positive means removing f weakens the inversion. Full spatial
S = 500 selection, all 500 points:

| feature removed | seed 0 | seed 1 | seed 2 | **seed mean Δr** | seed-mean r without it (all 7: −0.819) |
|---|---:|---:|---:|---:|---:|
| landcover | +0.072 | +0.122 | +0.025 | **+0.073** | −0.746 |
| hand | −0.031 | +0.051 | +0.061 | +0.027 | −0.792 |
| log10_dist_river | −0.055 | +0.093 | +0.043 | +0.027 | −0.792 |
| slope | −0.020 | +0.072 | +0.014 | +0.022 | −0.797 |
| tri_300 | +0.038 | +0.061 | −0.058 | +0.014 | −0.805 |
| wetindexcat | −0.049 | +0.013 | +0.047 | +0.004 | −0.815 |
| log10_dist_coast | −0.055 | +0.033 | +0.010 | −0.004 | −0.823 |

**Primary: not met.** The largest seed-mean Δr belongs to `landcover` (+0.073), not
`log10_dist_river` (+0.027, tied with hand). No removal brings r above −0.3: the weakest
remaining inversion is −0.746.

Because no single feature carries it, the brief asks for the two-feature combination D1's
coefficients point to. D1's seed-mean coefficients on the same mask (test 60, the brief's formula)
sum most negative for **`landcover + hand`** (−0.170 − 0.174 = −0.345). On this mask, D1's single
most negative coefficient is hand (−0.174), with landcover nearly tied (−0.170). On the pooled
regions D1 puts landcover first (−0.196). So D1 and D2 agree that landcover is among the
strongest carriers, but D1 does not separate it from hand.

Secondary masks (seed mean Δr; files as above):
- **Union of the regions:** `log10_dist_river` has the largest Δr, +0.081 (r −0.672 → −0.591),
  then wetindexcat +0.057 and hand +0.048.
- **Classic S = 500:** the largest are tri_300, landcover and hand, all +0.05. River is +0.003.
- **Spatial S = 500, land only:** landcover +0.056 is largest; river is +0.022.

Reading:
- The inversion does not sit in any one feature. Take any one away and the population still
  inverts at r ≈ −0.75 to −0.82, so the features are redundant carriers.
- Distance to the river network carries part of the inversion on the regions themselves, and in
  seeds 1–2. That matches D1, but it is not the dominant carrier of the full selection.
- The brief's sentence "the inverse line is carried by distance to the mapped river network" is
  **not supported** as stated.

## Part E. Map-production signatures of the region polygons

Code: `aotil/o5_polygons.py`; test `aotil/test_o5_polygons.py`, 84 checks pass with
`--null-file results/2026-10-17/o5/E_null_draws.parquet` (`logs/2026-10-17/E_polygons_test.log`).
Outputs in `results/2026-10-17/o5/`:
- `E_polygons.csv`: one row per lobe × region × role × budget (S = 250 or 500), 35,696 rows
  covering 23,432 distinct map polygons;
- `E_summary.csv`, 1,782 rows;
- `E_null_draws.parquet` and `E_null_draws_pooled.parquet`: the metrics of every null patch,
  per region and pooled.

Log: `logs/2026-10-17/E_polygons.log`.

**What a lobe is.** A lobe is one single-part flood-zone polygon. Its role is `region` (Zone A near
a region's points), `matched_A` (unselected Zone A in the same tiles, matched on perimeter), or `AE`.
- The seed-0 lobes and the recurring seed-1/2 regions' lobes are from 2026-10-16
  (`results/2026-10-16/boundary/`).
- The other six regions' lobes (seed 1 c2, c5, c6; seed 2 c2, c3, c4) were built with
  `aotil/lobes.py --only-components` into `results/2026-10-17/o5/E_lobes/`. This used polygons only
  and the 2026-10-16 arguments.
- The stored lobe geometry is identical to the polygon parts in the prep table `haz_FL_5070`, so
  vertex counts are those of the published map, not densified.

**Primary: median vertex spacing, region lobes minus matched Zone A lobes**, pooled by region
bootstrap over the 16 regions with land points. Vertex spacing is perimeter divided by the number of
vertices; a coarse spacing suggests hand-digitizing from a small-scale paper map.
- Region lobes 6.22 m, matched Zone A lobes 12.21 m: **−5.99 m [−6.98, +0.40]**.
- **Predicted positive (coarser): not met; the sign is the opposite.** The region polygons are
  digitized *more finely* than the Zone A around them.
- **Null percentile 0.0** (null median +0.66, 95% range [−1.55, +2.66]).
  - The brief defines no null for E.
  - Each S2 null patch was given its own region and matched lobes by `lobes.py`'s rules, and the
    observed value is below every one of the 200 draws.
- **Zone AE contrast:** AE lobes 2.92 m, region − AE = **+3.30 m [+0.42, +12.40]**. Region Zone A
  is digitized more coarsely than detailed-study Zone AE, but more finely than nearby ordinary
  Zone A.

| scope | region / matched A / AE, median m | region − matched A [95%] | null % |
|---|---|---|---|
| pooled (16 regions) | 6.22 / 12.21 / 2.92 | −5.99 [−6.98, +0.40] | 0.0 |
| positive-dominated (11) | 6.70 / 12.06 / 4.53 | −5.36 [−6.65, +0.75] | 0.0 |
| negative-dominated (5) | 3.25 / 15.26 / 1.41 | −12.01 [−13.70, −1.29] | 0.0 |

Per region, region − matched A in metres. The intervals resample lobes within the region.
- **Finer, with the interval excluding 0 (9 regions):**
  - seed 0: c2 −1.96, c4 −0.97, c5 −2.09;
  - seed 1: c0 −5.40, **c1 Sarasota −12.20**, c2 −5.74;
  - seed 2: **c1 Sarasota −12.54**, c2 −9.18, c3 −7.76.
- **Coarser (4 regions):** seed 0 c1 +0.83, seed 0 c3 +13.08, seed 1 c6 +28.61 (Liberty), seed 2
  c4 +3.38 (Calhoun).
- **Interval spans 0:** seed 0 c0, seed 1 c5, seed 2 c0 (Collier).

The pooled medians mix very different per-region distributions, so the pooled difference is
larger than most single-region ones. The edge-weighted version (total perimeter / total vertices)
is −3.84 [−8.36, +1.45], null 2.5.

**Other signatures** (pooled, region vs matched Zone A, region-bootstrap 95%, null percentile; Zone
AE in brackets):

| signature | region | matched A | difference [95%] | null % |
|---|---|---|---|---|
| share of edge in straight runs (heading within 1° over ≥ 300 m) | 0.049 | 0.042 | +0.007 [−0.007, +0.023] | 1.5 |
| shape index 2·ln(P/4)/ln(A), median (higher = more convoluted) | 1.079 | 1.033 | +0.046 [+0.006, +0.076] | 100 |
| STUDY_TYP "DIGITAL CONVERSION", share of lobes | 10.2% | 0.9% | +0.092 [−0.007, +0.402] | 99 |
| source "FIRM" (traced from the old paper map), share of edge | 32% | 25% | +0.069 [−0.217, +0.371] (AE 2%) | 78.5 |
| source "STUDY", share of edge | 37% | 47% | −0.106 [−0.408, +0.146] (AE 90%) | 41.5 |
| edge on 1:24,000-scale panels | 19% | 34% | −0.146 [−0.308, +0.060] (AE 7%) | 15.0 |
| edge on 1:6,000-scale panels | 2% | 6% | −0.044 [−0.076, −0.012] (AE 61%) | 85.0 |
| effective year of the lobe's main panel, median | 2012.3 | 2012.3 | 0.00 [−1.13, +7.37] (AE 2018.9) | 69.0 |
| area inside a LOMR (a later local map correction), mean | 0.07% | 0.11% | −0.0004 (AE 2.85%) | 36.0 |

- The panel rows use each lobe's own county map panels. That is a fix the verifier found
  necessary (Deviations, item 9).
- Negative-dominated regions: 71% of their lobes are BLE risk-class polygons, all of them the "low
  flood risk" class, against 22% of matched lobes in any BLE risk class (null 100). By edge length
  the shares of any BLE class are equal (11% vs 11%), so these are many small polygons.
- Of 59 pooled metrics with a non-constant null, 12 fall strictly beyond 2.5/97.5, against about
  3 expected by chance. They are strongly correlated with each other (spacing and segment length;
  the same category under two column names).

**Reading.**
- **Not met, and the point estimate has the opposite sign.** The region polygons have their
  vertices closer together than the Zone A around them.
  - The interval includes 0: the bootstrap has only 16 regions, and the per-region differences
    run from −12.5 to +28.6 m.
  - So the reversal rests on the null: no random Zone A patch's polygons are this finely drawn.
- They are also more convoluted (shape index, null 100).
- They are more often marked "digital conversion" by lobe count (10.2% vs 0.9%, null 99), but not
  by edge length (1.2% vs 3.9%).
- Their panels are no older and not at a coarser scale.
- So the brief's "if not met" reading, that the polygons are ordinary Zone A, only half applies.
  They are not a coarser product. They are not ordinary either: they differ in the other
  direction, with finer and more convoluted linework. Where that linework comes from is a
  question for the Flood Insurance Study reports below.
- The explanation for the models' failure lies in B, C and D. That fits C: the region polygons
  are carefully drawn lines that disagree with the detailed-study rule, not careless ones.

**Counties and DFIRM ids for the Flood Insurance Study reports.** A DFIRM is FEMA's digital flood map
for one county; its id is the county FIPS code plus "C".

| region | counties | DFIRM ids |
|---|---|---|
| seed 0 c0 | Dixie (12029), Lafayette (12067) | 12029C, 12067C |
| seed 0 c1 | Hamilton (12047) | 12047C |
| seed 0 c2 | Bradford (12007) | 12007C (lobes also in 12125C, Union) |
| seed 0 c3 | Lafayette, Taylor (12123) | 12067C, 12123C |
| seed 0 c4 | Gulf (12045) | 12045C |
| seed 0 c5 | Bradford | 12007C (lobes also in 12003C, Baker) |
| seed 1 c0 | Collier (12021), Hendry (12051) | 12021C, 12051C |
| seed 1 c1 | Sarasota (12115) | 12115C |
| seed 1 c2 | Alachua (12001) | 12001C (most lobe edge in 12007C) |
| seed 1 c3, c4 | Okeechobee (12093), Lake Okeechobee | 12093C (water only, so not in E's analysis) |
| seed 1 c5 | Flagler (12035) | 12035C |
| seed 1 c6 | Liberty (12077) | 12077C |
| seed 2 c0 | Collier | 12021C (lobes also in 12051C) |
| seed 2 c1 | Sarasota | 12115C |
| seed 2 c2 | Liberty | 12077C |
| seed 2 c3 | Marion (12083) | 12083C |
| seed 2 c4 | Calhoun (12013) | 12013C |

That is 17 DFIRMs in all:
- 14 hold region land points: 12001C, 12007C, 12013C, 12021C, 12029C, 12035C, 12045C, 12047C,
  12051C, 12067C, 12077C, 12083C, 12115C and 12123C;
- 1 holds only the two water-only regions: 12093C (Okeechobee);
- 2 are reached only by nearby lobes: 12003C (Baker) and 12125C (Union).

## Part F. What does pooling these labels do?

**Code change.** `aotil/zoo.py` gained `--extra-train FILE` and `--extra-train-repeat N`; the
default is none, so nothing changes without them. Every model keeps its usual random sample of
Zone AE training rows, plus **all** listed rows. The listed rows are appended after that sample is
drawn, so each model keeps exactly its base rows, features and seed and gains only the added rows.
The cell stores the ids and N. `aotil/audit_cell.py` reads them back, so refits of a pooled cell
still reproduce it exactly.

Smoke test (`logs/2026-10-17/F_zoo_extra_train_smoke.log`):
- without the flag, the seed-0 cell is identical to `cells_w7`;
- with it, the cell passes its audit.

**Populations** (300 models per seed, the same specs as `cells_w7`):
- **+region:** the seed's region land points that have an S1 control;
- **+matched:** those points' S1 controls, aligned row by row. Each pair has the same label and
  tile and terrain within caliper 1.0.

Pairs per seed: 415 / 301 / 360. Region points without a control were left out of both lists, so
the lists have equal length: 3 / 7 / 28 points (`results/2026-10-17/o5/F_lists/`). Each condition
was also built with every added row repeated 10 times. Folders:
`data/interim/aotil/cells_w7_pool_{region,matched}[_x10]/`. All 12 cells pass `audit_cell.py`
(`results/2026-10-17/o5/F_audit/`).

**Evaluation** (`aotil/o5_pooling.py`, test `aotil/test_o5_pooling.py`, 13 checks pass with `--csv`;
output `results/2026-10-17/o5/F_pooling.csv`, log `logs/2026-10-17/F_pooling.log`):
- **eval 1:** Zone AE test points ≥ 5 km (straight line, in metres) from every added point of
  either list. That keeps 9,841 / 9,535 / 10,692 of 10,235 / 10,479 / 11,426 points.
- **eval 2:** Zone A test points ≥ 5 km from every added point and outside every region: 9,283 /
  11,110 / 11,653 points.

The added points are never scored. All conditions are scored on the same points, model by model.
The interval is a test-tile bootstrap, 500 draws: one resample of the union of the seeds' 50 km
test tiles per draw, applied to every seed.

**Primary:** Zone AE AUC on eval 1, +region minus +matched, mean over 900 model pairs =
**−0.0018 [−0.0032, −0.0007]**. 63% of the model pairs are negative. **Predicted negative: met.**

| contrast | Zone AE far from added points (eval 1) | other Zone A (eval 2) |
|---|---|---|
| **+region − +matched** | **−0.0018 [−0.0032, −0.0007]** | −0.0013 [−0.0024, +0.0002] |
| ×10: +region − +matched | −0.0123 [−0.0174, −0.0069] | −0.0137 [−0.0181, −0.0082] |
| +region − Zone AE only | −0.0013 [−0.0025, −0.0002] | −0.0012 [−0.0030, +0.0006] |
| +matched − Zone AE only | +0.0005 [−0.0009, +0.0023] | +0.0002 [−0.0010, +0.0012] |
| ×10: +region − Zone AE only | −0.0214 [−0.0296, −0.0130] | −0.0240 [−0.0325, −0.0149] |
| ×10: +matched − Zone AE only | −0.0091 [−0.0143, −0.0035] | −0.0103 [−0.0163, −0.0046] |

Per seed, +region − +matched on eval 1:
- seed 0: −0.0001 [−0.0017, +0.0016];
- seed 1: −0.0026 [−0.0054, −0.0006];
- seed 2: −0.0027 [−0.0048, −0.0010].

×10: −0.0027, −0.0132, −0.0210.

Reading:
- **How big.** The primary is negative with an interval that excludes 0, but it is small. Its
  point estimate is inside ±0.002, as the brief warned it might be. The ×10 sensitivity is not
  (−0.012), so the two are not both inside ±0.002.
- **The comparison that isolates the region labels.** Adding ordinary Zone A labels with the same
  terrain, labels and tiles (+matched) costs nothing at ×1. It costs −0.009 at ×10. Adding the
  region labels costs −0.001 at ×1 and −0.021 at ×10, more than twice as much.
- **No gain elsewhere.** Pooling the region labels does not improve other Zone A either: eval 2 is
  −0.0013 [−0.0024, +0.0002] against +matched. So the brief's Story 2 condition ("near 0, with
  evaluation 2 improving") is not met.
- **Which regions drive it.** Seeds 1–2 carry the effect. Seed 0, whose regions are all Zone A
  floodplain positives in north Florida, shows none (−0.0001). Seeds 1–2 add the
  negative-dominated regions and Collier/Hendry: their added rows are 50% and 71% negatives.
- **Conclusion.** This supports the brief's Story 1 reading: the region labels conflict with the
  detailed-study rule, and pooling them as prior work does costs detailed-study accuracy. It is a
  small cost at the regions' real size: 301–415 added rows per seed, against a median model's
  12,800–14,700 Zone AE training rows, i.e. about 2–3%.

---

## 1. Proposed new wording of O5 (a proposal for Victor; the paper is not edited)

> **O5 (proposed).** The selected regions are places in the Zone A-mapped area whose labels
> disagree with those of similar-looking Zone AE land, more often than other Zone A land on the
> same terrain does. Most are Zone A floodplain; five are land just outside it. The disagreement
> is not memorized label noise and is not carried by any single feature. The region polygons are
> not cruder digitizations; if anything they are drawn more finely. Adding these labels to
> training costs a little detailed-study accuracy.

**One-paragraph summary (proposal).** The boundary analysis of 2026-10-16 found that the selected
regions' Zone A edges look like the Zone A edges around them: no extra tracing along soil or map
lines. This round asked why AE-trained models still fail there.
- **A:** it is not label noise the models memorize. Models that cannot interpolate their labels
  invert just as strongly.
- **E:** it is not a cruder map either. The region polygons are digitized more finely than nearby
  Zone A (beyond every null draw, though the region interval includes 0), and their panels are no
  older or coarser.
- **C:** what differs is the labelling rule. On terrain-matched land, the region points disagree
  with their 25 nearest Zone AE training points 12 points more often than ordinary Zone A does,
  and even ordinary Zone A disagrees twice as often as Zone AE disagrees with itself. Most of the
  accuracy drop is concept shift rather than unfamiliar inputs.
- **D:** no single feature carries the inversion. Land cover carries the most, and distance to the
  mapped river network carries part of it on the regions and in seeds 1–2.
- **B:** the predicted river-network signature is absent once terrain is matched. The
  positive-dominated regions differ from terrain-identical Zone A mainly in land cover: forest
  where matched Zone A is wetland.
- **F:** pooling the region labels into training, as prior work does, lowers Zone AE AUC slightly
  more than pooling the same number of terrain-matched Zone A labels (−0.002, or −0.012 when
  repeated ×10).

Together: the regions are places where the two mapping procedures assign different labels to land
the models cannot tell apart. That is label provenance in the brief's strongest form (C), with a
measurable cost when the labels are pooled (F, Story 1). The brief's covariate-shift reading
(Story 2), "detailed studies never covered land like this", is not what the data show: the regions
lie inside the Zone AE training set's feature support (C1).

## 2. Which regions drive each result

The 16 regions with land points are 11 positive-dominated and 5 negative-dominated. The recurring
regions are Collier/Hendry (seed 1 c0, seed 2 c0) and Sarasota (seed 1 c1, seed 2 c1).

| part | positive-dominated | negative-dominated | recurring regions |
|---|---|---|---|
| A | carry the inversion (−0.70 on held-out models) | not inverted across the population (+0.006); inverted within families except nb | both inside the selection; per-region r in `A_interpolation.csv` |
| B | river SMD splits by sign (−0.78 to +0.92); the land-cover difference is here (seed 0 especially) | null part met (1.0); the interval just reaches 0 because Collier s2 c0 goes the other way (+1.67) | Collier s2 c0 against the prediction; Sarasota s1 c1 −0.76 with it |
| C | larger conflict (+0.146) | smaller (+0.089); larger covariate part (0.27) | all positive (Collier and Sarasota among the 16 positive regions) |
| D | seed 0, all positive-dominated: no river effect. D2: landcover +0.072, then tri_300 +0.038 (hand −0.031). D1: hand and landcover most negative | seeds 1–2 hold all of them: river carries part (D1 coefficients, D2 Δr) | Collier and Sarasota are in seeds 1–2, where the river shows up |
| E | finer in 7 of 11 point estimates, 6 with intervals excluding 0 (−5.36 m pooled) | finer by much more (−12.0 m); 71% of lobes are BLE low-risk-class polygons (22% of matched lobes in any BLE class) | **Sarasota is finest (−12.2 and −12.5 m)**; Collier is finer in seed 1 (−5.4 m, interval excludes 0) and spans 0 in seed 2 |
| F | seed 0 (only positives added): no effect (−0.0001) | seeds 1–2 carry the effect (−0.0026, −0.0027) | both are in the seeds that carry it |

## 3. Deviations from the brief, with reasons

1. **S1 caliper 1.0 and the seed-SD balance rule**, instead of caliper 0.25 and the SMD rule.
   This was Victor's decision at the S1 Stop. At 0.25 only 60.5% of points matched, and the SMD
   rule cannot be met at any caliper, because the matched sets' terrain spread is 8–31% of the
   seed's (pooled, at 0.25). The 0.25 outputs are kept in `results/2026-10-17/o5/S1_caliper0.25/`.
   At 1.0, three values in two regions are just over 0.1 seed-SD: seed 2 c3 (tri_300 0.138, slope
   0.122) and seed 1 c6 (tpi_broad 0.110). The proposal named only the first region; B and C
   report terrain-adjusted contrasts alongside.
2. **Region-type counts.** The table has 11 positive- and 5 negative-dominated regions. The
   progress report said 10 and 6 by a counting slip; the member lists were right.
3. **"Full spatial S = 500 selection"** is read as all 500 points in the A and D2 primaries, as
   recorded in PRIMARIES.md before computing. Land-only versions are reported beside them and
   give the same answers.
4. **Part A's primary set is mostly in-sample,** as the brief fixes it: it includes the selection
   and validation models. The held-out set is reported beside it and agrees (−0.808).
5. **F lists use matched pairs only,** so +region and +matched have the same length and labels.
   This drops 3 / 7 / 28 region land points per seed that had no control.
6. **F's "5 km from any added point"** excludes the union of both lists, so every condition is
   scored on the same points. That makes the per-model difference paired. The F bootstrap uses
   500 tile draws; the brief gives no number.
7. **Order.** The D2 and F population builds (~20 s each) ran in the background *while* B, C and
   E were built, rather than after them. Nothing depended on the order.
8. **E's null.** The brief defines none for E. Each S2 null patch got its own lobes by
   `lobes.py`'s rules, so E's metrics have a null percentile like B's and C's.
9. **E's panel join** (a fix found in verification). Panels are matched within each lobe's own
   county map. Before the fix, neighbouring counties' overlapping panels were picked almost at
   random, and two signatures (scale 1:24,000, effective year) came out as artefacts. The
   primary did not change.
10. **S2 details.** Growth goes through unselected points only. Seed 2 c0's null patches leave
    its tile (26% of their points on average), so that region's null is only partly tile-matched.
11. **Extra sensitivities** beyond the brief. B: a total-variation distance per categorical
    variable, and a label-weighted null. C: cross-check estimators for C3. E: an edge-weighted
    spacing and a point-holding-lobes subset. They are labelled as such and never replace a
    primary.
12. **C3 caveat.** The 99th-percentile weight clip binds and carries 60–71% of the weight, so the
    covariate part is a lower bound. The fold seed moves the split by about 0.01 (verifier
    scripts in `results/2026-10-17/o5/verification/`).
13. **D2.** No single feature turns r above −0.3. As the brief asks, the two-feature
    combination is read from D1's coefficients (landcover + hand), without building more
    populations.
14. **D2 populations redraw the feature and row subsets.** The brief's "only the feature list
    changes" cannot hold exactly with `zoo.py`'s seeded draws: with 6 features, each model's
    feature subset and, for 113–133 of 300 models per seed, its row subsample are redrawn. The
    specs (family, hyperparameters, fractions, seed) are unchanged. Δr therefore includes this
    resampling (Part D2).

## 4. Code added or changed (nothing committed)

New, each with a docstring and a test or smoke run:
- `aotil/match_controls.py` (S1; `--balance-rule` flag added after the Stop, default the brief's
  SMD rule), `aotil/region_null.py` (S2), `aotil/test_match_controls.py` (48 checks);
- `aotil/o5_interpolation.py` + test (A, 38 checks with `--csv`);
- `aotil/o5_contrast.py` + test (B, 76 checks);
- `aotil/o5_shift_decomp.py` + test (C, 89 checks);
- `aotil/o5_feature_inclusion.py` + test (D1, 44 checks);
- `aotil/o5_lofo.py` + test (D2, 42 checks with `--csv`);
- `aotil/o5_polygons.py` + test (E, 84 checks with `--null-file`);
- `aotil/o5_pooling.py` + test (F, 13 checks with `--csv`).

Changed, flags added and defaults unchanged:
- `aotil/zoo.py`: `--extra-train FILE`, `--extra-train-repeat N`. Without the flags the seed-0
  cell is identical to `cells_w7`.
- `aotil/audit_cell.py`: reads the stored extra rows, so refits of a pooled cell reproduce it.
  Cells without them are unaffected.

New data, in the folders the brief names; nothing else was written under `data/`:
- `data/processed_w7_minus_<f>/` for 7 features (an 8-line `FEATURES.txt`, 2 comment lines and
  6 features, plus a relative link to the dataset);
- `data/interim/aotil/cells_w7_minus_<f>/` (7 × 3 cells);
- `data/interim/aotil/cells_w7_pool_{region,matched}[_x10]/` (4 × 3 cells).

All 33 new cells pass `aotil/audit_cell.py`. `data/processed/`, `data/processed_w7/`, `cells/`,
`cells_w7/`, `results/2026-10-14/` and `results/2026-10-16/` are unchanged.

Runners and logs: `logs/2026-10-17/D2F_runs.sh` and `D2F_runs.log`, and per-part logs in
`logs/2026-10-17/`.

Added during the check of this report (2026-10-18):
- `results/2026-10-17/o5/verification/`: the verifier scripts behind four numbers cited above,
  with their outputs (`*.out`):
  - B exact bootstrap: `B_verify.py`; `B_verify2.py` and `B_verify3.py` are its other checks;
  - C3 clip mass: `C_v_clipmass.py`;
  - C3 fold seed: `C_v_c3.py`;
  - D1 continuity correction: `D1_verify3.py`.
- `results/2026-10-17/o5/S1_caliper_sensitivity/`: S1 rerun at calipers 0.5, 0.75 and none, for
  the caliper table's balance columns.
- `results/2026-10-17/o5/E_null_draws*.parquet`: E's null draws, read by its test.

## 5. Glossary: the map terms in CS terms

Definitions of the features follow `docs/FEATURE-SOURCES.md`.

**Labels and maps**
- **Zone A / Zone AE.** Two kinds of FEMA flood-zone polygon. Both mark the land with a 1% chance
  of flooding each year. Zone AE comes from a detailed engineering study with computed water
  heights; Zone A comes from an approximate method. Here they are two labelling procedures:
  models train on Zone AE labels and are tested on Zone A.
- **Positive / negative.** A point inside the flood-zone polygon (label 1) or outside it (label 0).
- **DFIRM; FIPS.** A DFIRM is the digital flood-map database for one county. Its id is the
  county's FIPS code (a 5-digit federal county id, e.g. 12115 = Sarasota) plus "C".
- **Flood Insurance Study (FIS).** The report that documents how a county's flood map was made.
- **FIRM panel.** The printed map is split into sheets ("panels"), each covering one rectangle, and
  each has its own scale and dates. Think of tiles of a map, each released separately.
- **Scale (1:6,000, 1:24,000).** The drawing scale of a panel: 1 cm on paper is 60 m or 240 m on
  the ground. A larger second number means coarser linework.
- **Effective date.** The date a panel took legal effect: its release version.
- **Digital conversion.** A study type meaning the polygons were traced from an older paper map
  rather than re-studied.
- **BLE.** Base Level Engineering: a newer, automated, large-area flood model. Its polygons carry a
  risk class (low, medium, high) instead of a computed water height.
- **BFE.** Base Flood Elevation: the computed flood water height, drawn as lines across the
  floodplain in detailed studies. `meta_bfe_km_per_km2` (line length per unit area) is a proxy
  for how detailed the study is.
- **LOMR.** Letter of Map Revision: an official later edit to a small part of a published map,
  like a patch applied on top of a release.

**Polygons (Part E)**
- **Single-part polygon; lobe.** A map record can hold several disjoint shapes (a multi-part
  polygon, like a list of shapes). A single-part polygon is one connected shape. A lobe is one
  such shape of flood zone, the unit of Part E.
- **Vertex spacing.** A polygon's boundary is stored as a list of vertices joined by straight
  segments. Perimeter / number of vertices is the mean segment length: the resolution at which
  the line was digitized.
- **EPSG:5070.** A map projection (CONUS Albers equal-area) in which coordinates are metres on a
  flat plane, so lengths and areas can be computed directly. The prep table `haz_FL_5070` stores
  polygons in it.

**Features and units**
- **50 km tile.** The 50 km × 50 km grid square a point falls in. The train/test split assigns whole
  tiles, as in grouped cross-validation.
- **NLCD land cover (`landcover`).** A national 30 m pixel grid that gives each pixel a land-cover
  class: e.g. 11 open water, 42 evergreen forest, 90 woody wetlands. The models read the class
  code as a number.
- **HAND (`hand`).** Height Above Nearest Drainage: how many metres a point sits above the stream
  it drains to.
- **`tri_300`.** The standard deviation of 10 m elevation in a 300 m window: a local roughness
  score. (Despite the name, not the classic Terrain Ruggedness Index.)
- **TPI (`tpi_300`, `tpi_broad`).** Topographic Position Index: a point's elevation minus the mean
  elevation around it (300 m window, or 900 m for `tpi_broad`). Negative is a hollow, positive a
  rise.
- **Slope, curvature.** The steepness of the elevation surface, and a discrete Laplacian of it
  (positive in hollows).
- **Catchment.** The patch of land that drains into one stream segment, like a basin assigned to
  each edge of the river graph.
- **`drainage_density`.** The length of the catchment's stream segment divided by the catchment's
  area.
- **`wetindexcat`.** The catchment's mean topographic wetness index: how readily water collects,
  from upslope area and slope.
- **Mapped river network; `log10_dist_river`.** The NHDPlus stream lines (a national stream
  graph drawn at 1:100,000). The feature is log10 of the distance in metres to the nearest line.
- **`growth_100_2`.** The 100-year 24-hour rainfall divided by the 2-year one (NOAA Atlas 14): how
  much heavier the rare storm is than the common one.
