# Approach, experiments and results: full run with the seven Woznicki features (D = 2026-10-16)

Short version of `docs/RUN-W7-REPORT-2026-10-16.md`, written 2026-10-17. Every number below was
checked against the result files under `results/2026-10-16/`.

## Approach

**Data.** The data are 80,000 points across Florida. Each zone has 20,000 positives inside its
FEMA 1% annual-chance flood polygons and 20,000 negatives. The negatives are band negatives:
points drawn 500 m to 5 km from any mapped floodplain, inside a map panel and inside Florida.
Every pair of points is at least 100 m apart. Points are grouped into 50 km tiles, and every
split is made by tile.

**Features.** The models use the predictor set of Woznicki et al. (2019), mapped onto our
columns:
- land cover (NLCD 2021);
- slope and local terrain variability (3DEP, 10 m);
- topographic wetness (EPA StreamCat, catchment mean);
- distance to the nearest river and to the coast (NHDPlus V2.1);
- height above the nearest drainage (MERIT Hydro).

We leave out their two soil-survey predictors, because soil-survey lines are one of the held-out
maps in the boundary analysis.

**Splits and models.** For each of three seeds, about 30% of the tiles are drawn as test tiles.
Only those holding positives of both zones are kept, which leaves 23–25 test tiles and 71
training tiles. Models train on Zone AE points in the training tiles. The in-distribution (ID)
set is Zone AE in the test tiles. The out-of-distribution (OOD) set is Zone A in the same test
tiles.

| seed | train | ID test | OOD test | test tiles |
|---|---:|---:|---:|---:|
| 0 | 28,109 | 10,235 | 12,771 | 24 |
| 1 | 29,395 | 10,479 | 15,036 | 25 |
| 2 | 25,670 | 11,426 | 14,357 | 23 |

Each seed trains 300 models from eight families: decision trees, extra trees, gradient boosting,
logistic regression, small neural networks, naive Bayes, random forests and XGBoost. The models
vary in capacity, in the share of features they see (30%, 60% or 100%) and in the share of
training rows (15%, 50% or 100%).

**Subset search.** OODSelect is run as the authors released it. It searches for a set of S Zone A
points on which ID accuracy and OOD accuracy are negatively correlated across models.
- The 300 models are split 60/20/20 into selection, validation and test models.
- Each search runs 100 Optuna trials of up to 50,000 epochs.
- The reported r is on the 60 test models, which the search never saw.

The spatial variant adds a graph-Laplacian penalty, μ·zᵀLz/(S·k). It is computed on a
10-nearest-neighbor graph of the Zone A points, with μ = 3, and penalizes selecting a point
whose neighbors are not selected. Regions are the connected pieces of the selected set on the
same graph, keeping pieces of at least 20 points.

**Boundary analysis (outcome 5).** For each region we take the Zone A polygons it covers. We also
take unselected Zone A polygons in the same tiles, matched on perimeter, and the Zone AE
polygons in those tiles. Each boundary is sampled every 30 m and cut into 300 m windows. Two
tests are applied to each window:
- **Contour-like:** elevation along the edge varies by at most 0.76 m, and the ground rises out
  of the floodplain. Elevation comes from the 3DEP 10 m DEM. This is how a computed water surface
  meets the ground.
- **On a line:** the window's median distance to a soil-survey line (SSURGO), a county line or a
  FIRM map-panel edge is at most 100 m (30 m as a second setting).

U is the share of edge that is not contour-like but lies on such a line, meaning edge that looks
traced from a map rather than from terrain. The primary statistic is U(region) − U(matched Zone
A), with a bootstrap over regions. A displaced-edge null shifts each region polygon 300–1,000 m
in a random direction 200 times. It shows how often a curve of the same shape lands on a line by
chance.

## Experiments

For each seed the run covered:
- the 300-model population;
- classic and spatial OODSelect over the full budget grid (S = 10 to 10,000);
- at S = 250, 500 and 1,000, two controls:
  - null-select: the same search with the ID accuracies shuffled, three shuffles;
  - family-disjoint splits: select on tree models and test on the others, and the reverse;
- regions, and whether they recur across seeds.

On seed 0 it added a μ sweep (μ = 0, 1, 3, 10) and a smoothed-classic ablation (the classic
selection smoothed over neighbors after the fact). That makes 54 null runs, 36 family-split runs
and 6 μ-sweep runs, all with 100 trials and none failed.

The boundary statistic ran on the seed-0 regions at S = 250 and 500, and on the two regions that
recur in seeds 1 and 2. The 45-feature seed-0 results serve as the comparison.

**What the agent built and decided.** It added:
- null-select and family-split options to the OODSelect code;
- a script for outcomes 1 and 2 with a bootstrap over test tiles;
- the boundary-statistic script (64 tests);
- a region-recurrence script;
- the Florida map figure.

Every default was left unchanged. There were two stops, both decided by Victor:
- **AUC rounding.** One training AUC came out at 1 + 2e−16, so AUCs are now clipped to [0, 1].
  No score changed.
- **Testable windows.** Only about half of Zone AE edge could be tested, because the 100 m probes
  often fall outside narrow polygons. The primary statistic therefore uses testable windows only.
  On those, 77–88% of Zone AE edge is contour-like, which is the calibration the test needed.

The agent also fixed one bug carried over from the preview. The preview's bootstrap interval used
every 10th model, which covers only 4 of the 8 families. It now uses all 300.

## Results

**O1: the shift.** Mean AUC over 300 models. The interval is a bootstrap over test tiles.

| | ID AUC | Zone A AUC | gap [95%] | models worse on Zone A |
|---|---:|---:|---|---:|
| seed 0 | 0.829 | 0.826 | 0.004 [−0.050, 0.060] | 56% |
| seed 1 | 0.828 | 0.755 | 0.072 [0.014, 0.120] | 94% |
| seed 2 | 0.852 | 0.789 | 0.063 [0.007, 0.113] | 95% |
| **pooled** | 0.836 | 0.790 | **0.046 [0.004, 0.094]** | 82% |
| pooled, land only | 0.812 | 0.787 | 0.025 [−0.042, 0.070] | 72% |
| pooled, land in blocks with both zones | 0.842 | 0.791 | 0.051 [0.017, 0.078] | 90% |
| 45 features, seed 0 | 0.858 | 0.839 | 0.020 [−0.040, 0.078] | 71% |

The pooled gap is positive and its interval excludes zero, but barely. It does not hold on land
points alone.

**O2: aggregate accuracy-on-the-line.** The correlation between ID and Zone A accuracy on the 60
test models is +0.729, +0.863 and +0.873 for seeds 0–2 (45 features, seed 0: +0.880).

**O3: classic OODSelect.** At S = 500:
- test r is −0.888, −0.948 and −0.931, against +0.71 to +0.84 for random subsets of the same
  size and −0.13 to −0.31 for the points the models get wrong most often;
- the subsets are scattered: 301–309 pieces across 21–24 of the 23–25 test tiles, with neighbor
  purity 0.08–0.10.

The controls are mixed:
- **Null-select.** The real search is more negative than every null run (54 of 54). But the null
  runs are not near zero: the classic median is +0.10, 9 of 27 fall below −0.3, and the lowest is
  −0.864.
- **Family-disjoint split.** Selecting on trees and testing on the other families keeps the sign
  everywhere (−0.74 to −0.93). The reverse loses it in seed 0 (+0.09 at S = 500).

**O4: the spatial penalty.**

| S = 500 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| spatial r | −0.815 | −0.842 | −0.801 |
| cost vs. classic | +0.07 | +0.11 | +0.13 |
| purity, classic → spatial | 0.08 → 0.75 | 0.10 → 0.81 | 0.09 → 0.81 |
| pieces | 309 → 20 | 304 → 23 | 301 → 29 |
| tiles | 24 → 5 | 23 → 15 | 21 → 11 |

Across all budgets and seeds, the spatial variant beats smoothed-classic on purity in 9 of 9
settings and on r in 8 of 9. Coherence rises with μ, and r stays between −0.81 and −0.90 for
μ ≥ 1. One spatial S = 500 result can move by about 0.07 in r with the search path alone. The
agent traced this to tiny machine-to-machine differences in tree-ensemble scores.

**Regions.** All 418 points in seed 0's six regions (≥ 20 points each) are Zone A floodplain
points in north Florida forest and wetland: Big Bend (Dixie, Lafayette, Taylor), Hamilton, two in
Bradford, and Gulf. None of them recurs in another seed. Three lie in tiles that no other seed
tests, and the three that could recur did not. Seeds 1 and 2 share two regions in southwest
Florida, Collier/Hendry and Sarasota.

**O5: the boundary analysis.** Testable windows, d = 100 m:

| | seed 0, S = 250 | seed 0, S = 500 | seeds 1 & 2, recurring regions |
|---|---|---|---|
| U region / matched Zone A / Zone AE | 0.32 / 0.29 / 0.11 | 0.29 / 0.35 / 0.21 | 0.16 / 0.22 / 0.15 and 0.15 / 0.31 / 0.15 |
| **primary** U(region) − U(matched Zone A) | +0.025 [−0.007, 0.058] | −0.059 [−0.226, 0.044] | −0.062 and −0.160 |
| U(region) against the displaced-edge null | above all 200 draws | above all 200 draws | above all 200 draws |

The primary is never positive with an interval that excludes zero. At 30 m the verdict is the
same. Region edges lie on soil or map lines more often than shifted copies of themselves, but no
more often than nearby unselected Zone A edges. The S = 500 negative comes mostly from Gulf
County, whose region edges are 91% contour-like. Collier/Hendry leans the predicted way (+0.07
and +0.13), but it is one region per seed.

## Discussion

**Outcomes 1, 2 and 4 hold.**
- The shift between the zones is real but small, and it depends on which tiles are held out. That
  fits the paper's argument: on average, a model trained on detailed studies loses little on
  approximate maps, and the failures hide in subsets.
- Accuracy-on-the-line holds strongly in aggregate.
- The Laplacian penalty does what it was built for. It turns 300 scattered pieces into 20–30
  regions for about 0.07–0.13 in r, and it beats smoothing the classic answer afterward.

**Outcome 3 holds in direction, but the controls weaken it.** The real search always beats the
shuffled one. Still, a third of the shuffled searches also land on strongly inverted subsets, so
with a 300-model population such subsets are easy to find. The meaningful comparison is real
against null at the same setting, not null against zero. The plan's wording ("null-select must
not invert") should change. The seed-0 failure when selecting on non-tree models and testing on
trees suggests part of the inversion rides on differences between model families rather than on
the data.

**Outcome 5 is not met as written.** Zone A edges in general are less contour-like than Zone AE
edges and often follow soil-survey lines. But the selected regions' edges look like the Zone A
edges around them. Whatever makes the regions hard for models trained on Zone AE does not show in
how their boundaries were drawn. In the terms of `docs/REGION-ANALYSIS-FLORIDA.md` §8, this
matches the row where unselected Zone A looks like the regions: the explanation moves toward
landscape the Zone AE models never saw, which this run did not test. Sarasota, whose region edges
are more contour-like than Zone AE's, fits the reading that the approximate map may be right
there and the models wrong.

**Region identity depends on the split.** None of seed 0's regions recurs, and only the two
southwest Florida regions recur across seeds. Any claim about specific places should rest on
those two.

**For the paper:**
- word outcome 1 as a small but real shift that most models show;
- report the null-select comparison as real against null at matched settings;
- reframe outcome 5 and contribution 3 around "the regions are not explained by boundary
  provenance";
- report the search-path variance alongside the spatial results.

**Limitations to state:**
- Florida only, and three seeds.
- The label is a FEMA map, not observed flooding.
- Height above drainage and topographic wetness are missing on open water, almost always for
  negatives, so a missing value leans toward "negative".
- Some DEM tiles were republished after the features were pulled.
- Untestable windows were left out of the primary boundary statistic.

Full report: `docs/RUN-W7-REPORT-2026-10-16.md`. Map: `results/2026-10-16/figures/florida_map.pdf`.
