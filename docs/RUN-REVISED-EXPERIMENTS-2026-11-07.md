# Revised Experiments: code against the text, the v4 rebuild, the full run (2026-11-07)

Brief: `docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md` (prompt: `docs/PROMPT-REVISED-EXPERIMENTS-2026-11-07.md`), item 7
of `docs/TODO-CLAUDE-CODE.md`. Run by Claude Code with `D=2026-11-07`. Outputs `results/2026-11-07/` (the w9
experiment of the same day stays in `results/2026-11-07/w9/`), logs `logs/2026-11-07/`, data `data/processed_v4/`,
cell `data/interim/aotil/cells_v4_2026-11-07/`. Nothing committed or pushed.

## 1. Summary

This run applied your decisions D1 to D7 to the code (each with a test), rebuilt the dataset as `data/processed_v4`
(96,467 points; nine features, the seven of before plus Woznicki's two soil features, one-hot encoded), fitted a new
model population (290 models after dropping 1 degenerate and 78 duplicate settings), and reran every experiment of
2026-11-06 on it with the same scripts, seeds and settings. That is 84 searches, not 81: Zone A now has 17,346 test
points, so the code's budget grid adds S = 15,000. Then it updated the paper's numbers, placeholders and figures
without rewording a sentence. Part A found no gap between the text and the code beyond D1 to D10, plus 11 wording
notes (§2.2). **Claims that hold:** accuracy-on-the-line on all of Zone A (+0.84, C2); OODSelect's inversion, still
scattered (−0.97 in 293 pieces, C3); the inversion is not memorized noise (−0.92 without the interpolating models,
C5); the regions' inputs are familiar but their labels differ (1.3% unmatched; disagreement 0.76 against 0.31 among
Zone AE points; C6a, C6b); the μ = 3 rule; and the inversion up to S = 10,000 (was 7,500). **Claims that weakened or
fail:** the per-tile accuracy gap shrank to +0.017 and is no longer distinguishable from the shuffled-zone null (C1,
flag F1). C4 fails more clearly than before: our Laplacian loss keeps the inversion (−0.92) but in 105 pieces, and
smoothing OODSelect's weights now keeps it too (−0.91) in fewer pieces (48), so "coherence must be part of the
objective" no longer holds (F7). The hardest-points baseline is 0 by construction (F6). The regions changed: there are
7 instead of 8, and only Liberty and Hendry are the same places as before, so the region names and the Case studies
paragraph are flagged (F9, F10). §8.2 lists ten flagged sentences for you to rewrite. Every number in the paper
matches the result files (`G3_check_paper_numbers.py`: ALL MATCH). The paper compiles to 10 pages; the only overfull
box is the existing one in tab:models, and the only undefined citation is `ssurgo`. Nothing was committed.

## 2. Part A: the code against the text

Every statement of Approach 3.1 to 3.5 with Algorithm 1, of Experiments (Implementation Details, Data
Construction, Region Analysis, and the method statements in the Results captions) and of Appendix A, with the code
that implements it, as the code stands after this run. **Verdicts:** *agrees*; *changed here (Dn)*, where the code
was changed to follow the text; *wording note*, where the text leaves out or blurs a rule the code applies (no code
change, listed for your prose in §2.2). **No row disagrees in a way D1 to D10 do not cover, so Part A found no Stop
and no bug.** Plain-language terms: an *SFHA* is FEMA's 1% annual-chance floodplain polygon (Zones A, AE, AH, AO, VE,
...); a *FIRM panel* is one printed map sheet; a *map unit* is one soil type of the county soil survey (SSURGO).

### 2.1 The table

| # | Statement (paper) | Code (`file:line`) | Verdict |
|---|---|---|---|
| **Approach** | | | |
| 1 | 3.1: features u(x) in R^d; a model f: R^d -> [0,1] | the d input columns are `zoo.model_inputs` (`aotil/zoo.py:158`), d = 28 on v4 (6 numeric + 14 land-cover + 6 flood + 2 fluvial columns) | agrees |
| 2 | 3.1: positives from R_pi, negatives from outside it | positives: `sample_points.sample_positives` (`sample_points.py:979`); band negatives 500 m to 5 km from every SFHA (`band_filter`, `:638-652`) | agrees |
| 3 | 3.1: points grouped into spatial tiles; tiles split into D_train and D_test | `zoo.split_tiles` (`aotil/zoo.py:202-217`) on `tile_50km` (`pull_gee.derive`, `pull_gee.py:383-387`) | agrees |
| 4 | 3.1: y-hat(x) = threshold of f(u(x)) | `zoo.predicts_flood` (`aotil/zoo.py:304`), `oodselect_paper.correctness` (`aotil/oodselect_paper.py:65`): flood iff p > 0.5 | agrees |
| 5 | 3.1: N models trained on the pi_AE points of D_train; D_ID = pi_AE points of D_test; D_OOD = pi_A points of the same tiles | `zoo.cell_data` (`aotil/zoo.py:349-372`); checked by `aotil/audit_cell.py` (PASS, §6) | agrees |
| 6 | 3.1: the shift is the mean over the N models of metric(ID) - metric(OOD) | `aotil/o1_shift.py:564-587` (`per_model`, `summary`), per tile `tile_gaps` / `tile_summary` (`:378`, `:440`) | agrees |
| 7 | 3.2: Z_ij = 1[y-hat_i(x_j) = y_j]; OOD accuracy (Zs)_i / ||s||_1 | `aotil/oodselect_paper.py:65-78`, `:195-208` (`pearson_r_loss`, `X @ z / (z.sum() + 1e-6)`) | agrees (the released code adds 1e-6 to ||s||_1) |
| 8 | 3.2: corr = a correlation between ID accuracies and subset accuracies | `:188-208` (`probit_t`, `pearson_r_loss`), `:235` (`hard_corr`) | agrees |
| 9 | 3.2: s_j = sigma(theta_j), theta trained by gradient descent | `:360` (`torch.randn`), `:373` (`torch.sigmoid(theta)`), `:361` (Adam) | agrees |
| 10 | 3.2: disjoint selection, validation and test models | `model_split` (`:96-106`, sklearn `random_state=42`, 60/20/20) | agrees |
| 11 | 3.2 Eq. oodselect and Alg. 1 line 7: L = corr_sel + lambda Omega_S + mu R | `:374-385`; Omega_S enters as `hp["penalty"] * warm * budget_term(...)` (`:381-382`) | agrees; **wording note W1**: the code multiplies lambda by a warm-up weight w_e that rises along half a cosine wave over W steps (`:381`), and the text never defines W, although Search names it |
| 12 | 3.2: T trials; h_t proposed from earlier trials | `run_search` (`:413`), `TPESampler(seed)` (`:469`), `study.optimize(n_trials)` (`:471`) | agrees |
| 13 | 3.2: Q_S(s) = the S points with the largest weights | `binary_selection` (`:226-232`, top S of theta; sigma is increasing) | agrees |
| 14 | 3.2 and Alg. 1 lines 9-10: corr_val of Q_S after every step; stop after P steps without improvement | `:389-398` (`wait >= patience`, `:397`); `PATIENCE = 500` (`:45`) | agrees |
| 15 | 3.2 and Alg. 1 line 13: the trial with the lowest final value gives M | objective = the last validation r; `study.best_trial` (`:472`) | agrees |
| 16 | 3.2: corr_te on the test models | `hard_corr` on the test rows (`aotil/report_paper.py`, `aotil/oodselect_paper.py:640-652`) | agrees |
| 17 | 3.3: R depends on s and V; mu = 0 gives OODSelect; R only in the training loss | `spatial_term` (`:309`), `--mu`; validation and test use `pearson_r_loss` alone | agrees |
| 18 | 3.3 and Alg. 1 line 14: regions = components of the subgraph M induces in A with >= c points | `aotil/regions.py:308` (`components`), `:395` (`ge_min_size`), `--min-size 20` (`:523`) | agrees |
| 19 | 3.4: A is a kNN graph in which every point has at least k edges | `aotil/graph.py:80-116` (`build_adjacency`: directed kNN, OR-symmetrised, 0/1, no self loops) | agrees |
| 20 | 3.4 Eq. laplacian: R = s^T L s / (Sk) | `aotil/graph.py:126` (`laplacian`), `aotil/oodselect_paper.py:245-276` (`_Quadratic`), `:366` (S k) | agrees |
| 21 | 3.5 Eq. gtv: R_GTV = sum over edges of abs(s_j - s_j') / (Sk); a subgradient at ties | `aotil/oodselect_paper.py:280-318` (`_TotalVariation`, `spatial_term`) | agrees |
| **Implementation Details** | | | |
| 22 | Metrics: y-hat = 1[f > 0.5]; accuracy, AUROC and F1 | row 4; `aotil/o1_shift.py` | agrees |
| 23 | Metrics: the shift per test tile, then pooled across test tiles | `tile_gaps` / `tile_summary` (`aotil/o1_shift.py:378`, `:440`); the paper reads the unweighted `tile_*_raw` columns (decided 2026-11-04) | agrees |
| 24 | Metrics: 95% intervals from 2,000 bootstrap draws of test tiles | `N_BOOT = 2000` (`aotil/o1_shift.py:55`), `tile_draws` (`:347`) | agrees |
| 25 | Correlation: Pearson rho of probits, accuracies clipped to [1e-6, 1 - 1e-6] (Eq. corr) | `EPS = 1e-6` (`aotil/oodselect_paper.py:46`), `probit_t` (`:188-192`) | agrees (as in the released code, 1e-6 is also added to each variance) |
| 26 | Correlation: Fisher intervals for corr_te | `fisher_ci` (`:477`) over the test models (after D6 no two of them share their predictions) | agrees |
| 27 | Budget penalty: Omega_S = (S - ||s||_1)^2 | `budget_term(..., "squared")` (`:211`); `BUDGET_PENALTY=squared` (`run_oodselect_local.sh:78`) | agrees |
| 28 | Budget penalty: S = 500, 4.0% of the OOD points; the inversion holds up to S = 7,500 | numbers; recomputed in §6 | agrees (numbers updated) |
| 29 | Graph: A joins two points when either is among the other's k nearest; k = 10 | row 19; `KNN=10` (`run_oodselect_local.sh:70`) | agrees |
| 30 | Graph: the median distance to a point's 10 nearest neighbors | the median over the Zone A test points of the distance to the 10th nearest (EPSG:3086 metres, `graph.project`), as on 2026-11-04 | agrees (number updated) |
| 31 | Search: theta ~ N(0, I) per trial; full-batch Adam; cosine learning rate between an initial and a minimum value with half-period tau | `:360-363` (`torch.randn`, `Adam`, `CosineAnnealingLR(T_max=tau, eta_min)`) | agrees |
| 32 | Search: TPE tunes lambda, W, the two learning rates and tau | `:441-445` (five `suggest_*`), `PENALTY_RANGE["squared"]` (`:55`) | agrees; **wording note W2**: the ranges (lambda log-uniform in [1e-6, 100], learning rate [1e-2, 1e2], minimum [1e-3, 1], tau 10 to 200, W 10 to 1,000) are no longer in the text |
| 33 | Search: T = 100 trials of at most 50,000 steps, patience P = 500 | `N_TRIALS = 100` (`:43`), `NUM_EPOCHS = 50_000` (`:44`), `PATIENCE = 500` (`:45`) | agrees |
| 34 | Search: c = 20; mu = 3 chosen from a sensitivity analysis | `--min-size 20`; `MU_SPATIAL=3` (`run_oodselect_local.sh:69`); the mu sweep (D8, §6.5) | agrees; D8: mu stays 3, the sweep is reported |
| 35 | Models: eight families | `GRIDS` (`aotil/zoo.py:96-116`) | agrees |
| 36 | Models: each model is one hyperparameter setting | `zoo_specs` (`:235-248`) | agrees |
| 37 | Models: drop models that predict one class for more than 95% of the ID test points | `is_degenerate` (`:314`), `DEGENERATE_CUT = 1/20` (`:86`) | agrees |
| 38 | Models: drop models whose predictions repeat an earlier model's | `prediction_key` (`:477`), `split_population` (`:504-524`) | **changed here (D6)** |
| 39 | Models: split the remaining models at random 60/20/20 | `model_split` (`aotil/oodselect_paper.py:96-106`) over the kept models | agrees |
| **Data Construction** | | | |
| 40 | Sources: Florida; labels from FEMA's NFHL | `STATES = ["FL"]`; `prep_state` reads the NFHL hazard layer (`sample_points.py:244`) | agrees |
| 41 | Sources: land cover = the point's NLCD 2021 class | `pull_gee.nlcd_group` (`pull_gee.py:144-150`, `USGS/NLCD_RELEASES/2021_REL/NLCD`, native 30 m grid) | agrees |
| 42 | Sources: NLCD uses 16 classes (1 water, 2 developed, 4 forest, ...) | descriptive; v4 holds 14 of them (no open water by design, no 12 ice) | agrees |
| 43 | Sources: soil flood frequency and fluvial class from SSURGO | `pull_soil_points.py` (SDA point lookup, `:194-230`) | agrees |
| 44 | Sources: flood frequency in six ordered classes from none to very frequent | `FLOOD_CODES` (`pull_soil_points.py:70`), `flood_code` (`:120`) on `muaggatt.flodfreqdcd` | agrees; wording note W3: the class is the map unit's dominant condition |
| 45 | Sources: fluvial class marks whether the area's main soil formed from river deposits | `fluvial_code` (`:143-153`): 1 if the dominant component's taxonomic subgroup contains "fluv", else 0 | **changed here (D4)**: binary; water map units NaN in both soil features (`:186`) |
| 46 | Sources: slope in degrees from 3DEP 10 m | `ee.Terrain.slope` on the 3DEP 10 m mosaic (`pull_gee.py:93-97`) | agrees |
| 47 | Sources: terrain variability = SD of the 3DEP elevations within 300 m of the point | `terrain_variability_group` (`pull_gee.py:115-125`), `sample_group` (`:194-212`): stdDev over a 300 m circle on the DEM's native grid; `elev_sd_r300` | **changed here (D3)** (was the SD inside the fixed 300 m cell, now `meta_tri_300`, `:356`) |
| 48 | Sources: HAND from MERIT Hydro | `merit_group` (`pull_gee.py:128-141`, band `hnd`) | agrees |
| 49 | Sources: topographic wetness (Beven) from StreamCat | `wetindexcat` (`pull_hydro.py:58`, `streamcat`) | agrees; D10: StreamCat's catchment mean, served x100 |
| 50 | Sources: distances to the nearest river and the coast, straight-line, log, from NHDPlus | `measure_flowlines` (`pull_hydro.py:322`), `LOG_SCALED` (`:83`) | agrees; D10: floored at 1 m; **wording note W4**: "river" is every NHDPlus flowline of type stream/river, artificial path (through lakes and wide rivers), connector or canal/ditch (`RIVER_FTYPES`, `:74`) |
| 51 | Sampling: positives uniformly by area inside each zone's polygons | `area_uniform_candidates` with no cap (`sample_points.py:751-786`; `polygon_quota`, `:410-426`) | **changed here (D2)** |
| 52 | Sampling: negatives from a band 500 m to 5 km from any floodplain around its polygons | `band_filter` (`:638-652`) on rings around the polygons behind the zone's positive candidates (`sample_band_negatives`, `:694-748`; `used`, `:1001`) | agrees; **wording note W5**: negatives are not uniform over the band (rings drawn by area, at most 4x an even share each, `:723`; overlapping rings make the band denser where floodplains crowd) |
| 53 | Sampling: inside a FEMA map panel and outside Zone D | `band_filter` (`:657`, `:661`; `UNASSESSED_ZONES`, `:85`) | agrees; D10: also outside AREA NOT INCLUDED |
| 54 | Sampling: remove points on open water, NLCD class 11 | `LandMask.is_land` (`:226`; classes 11 and 0 are not land, `:90-91`), `on_land` for positives, `band_filter` (`:668-672`) | agrees; D10: class 0 too, and NFHL OPEN WATER polygons for negatives; negatives also inside Florida's county polygons (`:676`) |
| 55 | Sampling: no two points closer than 100 m | `grid_thin` (`:355`) within each group, then `separate_whole_state` (`:1016`) | agrees; **wording note W6**: two passes (each group, then the whole state), and a point is dropped when a point already kept lies within 100 m |
| 56 | Sampling: 100,000 candidates that meet every requirement, 25,000 per zone and label | `draw_valid` (`:540-575`), `DEFAULTS["candidates"]` (`:66`), counts stored with the points (`write_points`, `build_counts`) | **changed here (D1)** |
| 57 | Sampling: drop points in random order until the spacing holds, which leaves XX points | `grid_thin` visits in random order and keeps every survivor; no cut | **changed here (D1)**; W6 |
| 58 | Split: 50 km tiles, wider than every feature's autocorrelation range (at most XX km) | `tile_50km` (`pull_gee.py:383-387`); ranges by `aotil/variogram_ranges.py` (new, §5) | agrees (number filled) |
| 59 | Split: 70%/30%, dropping test tiles that do not represent both procedures | `split_tiles` (`aotil/zoo.py:202-217`) | agrees; **D9 wording note** (§2.2) |
| 60 | Split: no evaluation set, models not tuned | `zoo.py` fits each setting once, no validation split | agrees |
| 61 | Data statistics: six numeric; land cover and fluvial categorical; flood frequency ordinal | `V4_FEATURES` (`pull_soil_points.py:76`), `zoo.CATEGORICAL` (`aotil/zoo.py:81`) | agrees |
| 62 | Data statistics: one-hot encode the three categorical and ordinal features | `model_inputs` (`aotil/zoo.py:137-192`) | **changed here (D5)** |
| 63 | Data statistics: XX% lack the soil features, XX% lack topographic wetness | numbers, §4 | agrees (numbers filled); **wording note W7**: HAND is also missing at a few points (§4.4) |
| 64 | Data statistics: models that cannot handle missing values fill the training median | `build` (`aotil/zoo.py:251-279`): a median imputer before dt, rf, et, lr, mlp and nb; hgb and xgb take NaN | agrees with D5; **wording note W8**: in scikit-learn 1.9.1 trees, random forests and extra trees could take NaN too, but they get the imputer; "all families but HGB and XGBoost" is the exact rule |
| **Region Analysis** | | | |
| 65 | Matching: z(x) = the q numeric features, standardized over D_OOD; q = 6 | `numeric_features` (`aotil/region_match.py:114-124`, every feature but `zoo.CATEGORICAL`), `standardizer` (`:127`) | **changed here (D7)** |
| 66 | Matching: candidates = pi_AE points of D_ID in test tiles with no point of C | `region_exclusion` (`:414`, `region-tiles`) | agrees; wording note W9: D_ID points missing a numeric feature are left out of the pool (and Zone A points missing one are not matched) |
| 67 | Matching: same land-cover class, within eps, up to the 5 closest | `match` (`:172-213`), `MAX_MATCHES = 5` (`:98`) | agrees |
| 68 | Matching: eps so that 5% of pi_AE points have no match (Stuart 2010) | `caliper` (`:223`), `CALIPER_PCT = 95` (`:99`) | agrees (eps updated) |
| 69 | Covariate shift: far more than 5% unmatched | the unmatched share per region (`set_row`) | agrees |
| 70 | Concept shift: eta (Eq. disagree); baseline by matching pi_AE to itself | `eta` (`:248`), the baseline row | agrees |
| 71 | Concept shift: "seven features do not determine the label" | nine features since this run | number to flag (F3) |
| 72 | Describing the regions: share of positives against the matches; features against the matches | `region_match.py` columns pos_points, pos_matches, hand_*, river_km_* | agrees |
| **Results captions (method statements)** | | | |
| 73 | Table tab:search: Random = the mean of 100 draws; hardest = the points the test models get wrong most often | `N_RANDOM = 100`, `N_TIE_BREAKS = 100` (`aotil/report_paper.py:31-32`, `:219`) | agrees |
| 74 | Table tab:search: pieces = components in A; tiles = test tiles they touch | `graph.coherence_metrics` (`aotil/graph.py:145`) | agrees |
| 75 | Table tab:search: smoothing averages OODSelect's final weights over each point's 10 nearest neighbors | `carry_field`, `smoothed_classic` (`aotil/audit_selection.py:106-136`) | agrees; wording note W10: the 10 nearest include the point itself |
| 76 | Controls: shuffled ID accuracies of the selection and validation models; family-disjoint split; models with 99% training accuracy left out | `null_accuracies` (`aotil/oodselect_paper.py:137`), `family_split` (`:109`), `INTERP = 0.99` (`aotil/o5_interpolation.py:40`) | agrees |
| **Appendix A** | | | |
| 77 | Table tab:models lists the grids; each combination is one model with a fixed seed | `GRIDS` and `zoo_specs` (`aotil/zoo.py:93-116`, `:235-248`); every value checked against the table | agrees |
| 78 | Every model trains on all seven features and all 29,161 training points | `fit_one` (`:385`): every training row, every input column with two distinct values | numbers updated (nine features, one-hot) |
| 79 | Random forests and extra trees draw a random subset of features at each split; forests resample rows | `max_features` axis; sklearn's `bootstrap=True` (rf), `False` (et) | agrees; wording note W11: with "all" features per split there is no subset |
| 80 | LR and MLP standardize their inputs | `StandardScaler` (`aotil/zoo.py:273`, `:276`) | agrees |
| 81 | Defaults of scikit-learn 1.9.1 and XGBoost 3.4.1 | the `.venv` versions (checked: 1.9.1 and 3.4.1) | agrees |
| 82 | Of the 369 settings, X predict one class for more than 95%, which leaves N | `split_population`; numbers, §6 | numbers updated; the duplicates of D6 are not in this sentence (flag F5) |
| 83 | Table tab:models: features per split in {1, 2, 7} | sklearn "sqrt" = floor(sqrt(input columns)); "None" = all | numbers updated: {1, 5, 28} on 28 input columns |

### 2.2 Wording notes for your prose (no code change)

These are rules the code applies that the text leaves out or states loosely. None changes a method; each is a sentence
you may want in Overleaf.

- **W1 (Search, 3.2): the warm-up W.** The code multiplies lambda by a weight w_e that rises from 0 to 1 along half a
  cosine wave over the first W steps (`aotil/oodselect_paper.py:381`), so Algorithm 1's lambda Omega_S is really
  lambda w_e Omega_S. Search names W among the tuned hyperparameters, but no sentence defines it. Possible wording:
  "the penalty weight rises from 0 to lambda along half a cosine wave over the first W steps".
- **W2 (Search): the TPE ranges** are no longer in the text: lambda log-uniform in [10^-6, 10^2], the initial
  learning rate in [10^-2, 10^2] and the minimum in [10^-3, 1] (both log), tau in 10 to 200, W in 10 to 1,000.
- **W3 (Sources): flood frequency** is the map unit's *dominant condition* (`muaggatt.flodfreqdcd`): the class that
  covers the largest share of the map unit, not a measurement at the point.
- **W4 (Sources): "river"** is every NHDPlus flowline of type stream/river, artificial path (the flow path drawn
  through a lake or a wide river), connector or canal/ditch; the coastline is measured apart.
- **W5 (Sampling): the band is not sampled uniformly.** Band candidates are drawn in 5.1 km rings around the polygons
  that hold the zone's positive candidates (4,513 Zone A and 5,065 Zone AE polygons), each ring chosen in proportion
  to its area but never more than four times an even share, so places that several rings cover are denser. The text
  says only "from a band ... around its polygons", which is accurate; it just does not claim uniformity.
- **W6 (Sampling): the thinning.** "We then drop points in random order until the spacing holds" reads as one pass.
  The code makes two passes, first within each group and then over all points, and each visits the points in random
  order and drops a point that lies within 100 m of a point already kept. A literal reading of "drop points in random
  order until the spacing holds" could also mean dropping random members of close pairs until none is left, which
  keeps a different number. Possible wording: "We then visit the candidates in random order and drop any that lies
  within 100 m of one already kept, first within each zone and label and then across all of them".
- **W7 (Data statistics): HAND is missing too**, at a few points (§4.4), besides the soil features and wetness.
- **W8 (Data statistics): which models impute.** In scikit-learn 1.9.1, decision trees, random forests and extra trees
  accept missing values natively, but the code gives them the median imputer, as it does logistic regression, the
  MLP and naive Bayes; only HGB and XGBoost take NaN. "Models that cannot handle missing values" is therefore not the
  exact rule; "All families except HGB and XGBoost fill missing values with the training median" is. (D5 keeps this
  pipeline.) With one-hot columns, the median of a 0/1 column is its majority value, so a point missing a soil
  feature is filled with the most common class.
- **W9 (Matching): points with a missing numeric feature.** Zone AE test points missing a numeric feature (mostly
  wetness) are left out of the candidate pool, and Zone A points missing one are not matched; both are counted (75 Zone AE test points this run, §7).
  Table tab:regions' caption already says this for the pool.
- **W10 (Table tab:search caption): smoothing** averages over each point's 10 nearest points including the point itself
  (itself and its 9 nearest others).
- **W11 (Appendix A): features per split.** With "all" features per split (`max_features=None`), random forests and
  extra trees consider every feature at each split, so "still draw a random subset of features at each split" holds
  only for the other two values.
- **D9 (Split): the exact rule.** The tiles are the 50 km squares of the EPSG:5070 (Albers) grid that hold at least one
  point of the dataset. 30% of them, rounded (at least 3), are drawn at random (seed 0) as candidate test tiles; a drawn
  tile stays a test tile only if it holds positives of both zones, and a dropped one goes to neither set (it is not
  moved to training). The rest are training tiles, and the models train on the Zone AE points among them. The 70/30
  is a split of tiles by count, not of points. Numbers in §4.3.
- **D10 (Sampling, Sources): the rules the text leaves out.** Band negatives also lie outside AREA NOT INCLUDED (FEMA's
  "not studied" polygons, like Zone D) and outside NFHL OPEN WATER polygons, and inside Florida's county polygons
  (without that mask, negatives would fall in Alabama and Georgia, whose floodplains are not loaded); NLCD class 0 (no
  data) counts as not land; both distances are floored at 1 m before the log; topographic wetness is StreamCat's mean
  over the point's NHDPlus catchment (a small watershed of about 1 to 3 km²), served times 100.

### Plain-language terms (GIS in CS terms)

- **NFHL / SFHA / zones.** FEMA's National Flood Hazard Layer is a set of polygons; an SFHA (Special Flood Hazard Area)
  polygon is the 1% annual-chance floodplain. Zone A was mapped by an approximate study, Zone AE by a detailed one.
  Zone D and AREA NOT INCLUDED mean "not studied". A FIRM panel is one printed map sheet.
- **EPSG:5070 / EPSG:3086.** Map projections: they turn longitude and latitude into planar x, y in metres (5070 for
  the whole US, 3086 for Florida), so Euclidean distances are true distances to within a fraction of a percent.
- **Native grid.** A raster is a 2-D array of pixels; reading it "on its native grid" means asking for the pixel that
  contains the point, without resampling the array first.
- **Buffer.** The polygon of all points within a distance of a geometry; a 300 m buffer of a point is a disk.
- **Map unit / component (SSURGO).** The county soil survey is a polygon map; each polygon belongs to a map unit (a
  mapped soil type), which is a mix of components (soil kinds) with percentages; the dominant component is the largest.
- **Semivariogram.** For a feature z and a distance h, gamma(h) is half the mean of (z_i - z_j)^2 over the point pairs
  about h apart. It starts low (neighbours are alike) and levels off (the sill) where points stop informing each
  other; that distance is the range. It is the spatial analogue of an autocorrelation function.
- **Catchment (NHDPlus).** The small area that drains into one stream segment; StreamCat reports averages over it.

## 3. Part B: the code changes

Backups of every changed file and of all private characterization goldens, from before the change:
`Claude outputs/goldens-before-2026-11-07-revised/` (`code/`, `golden/`).

### 3.1 D1 to D7

| # | Change | Where |
|---|---|---|
| D1 | **Candidates, no cut.** Each group (zone and label) draws exactly `--candidates` (25,000) candidates that pass every requirement of its group, in batches: `draw_valid` pools the valid candidates of every batch and keeps a random 25,000 (the first batch proposes about the inverse of the expected valid share; later ones the shortfall over the share seen, at most 10 times the batch before and 5 million; a group with no valid candidate after 100 proposals per one wanted stops with `SupplyError`). The candidates are thinned at 100 m within the group (`grid_thin`), then over the whole state (`separate_whole_state`), and every survivor is kept. The candidate, thinning and survivor counts of every group are stored with the points (`write_points`, parquet metadata key `sample_points_counts`; read by `build_counts`), printed (`group_table`) and checked by `--audit` (`group_problems`). Retired and deleted: `per_domain`, `neg_ratio`, `poly_cap_frac`, `oversample`, `ring_oversample`, `surplus`, `--exact`, `trim_to_targets`, `balance_domains`, `trim_and_balance`, `rank_negatives`, `dense_ranks`, `achieved_ratios`, `print_achieved_ratios`, `_polygon_cap`, `_with_surplus`, `_audit_exact_targets`, and the `meta_neg_rank` column. `audit_dataset.py`: B1 now checks the stored counts and that the table holds exactly the points file's points; new B17 retests every sampling mask; `--per-domain` and `--neg-ratio` gave way to `--points`, `--candidates` and `--land-mask`. | `sample_points.py:65-84, 540-575, 638-748, 979-1170, 1251-1281`; `audit_dataset.py:296-352, 777-845` |
| D2 | **No polygon cap.** `polygon_quota(..., cap=None)` is the plain multinomial draw by polygon area, so the positives (and shadedX) are uniform by area over the land of their polygons (`area_uniform_candidates`: polygons drawn by area, a uniform point in each, then the land test, which keeps the draw uniform over the land). The band's rings keep their cap (W5). | `sample_points.py:410-426, 751-786` |
| D3 | **Terrain variability within 300 m.** New request group `sd300` (`terrain_variability_group`): every point is buffered into a 300 m circle (to 1 m) and the 3DEP 10 m pixels inside are reduced with `stdDev` on the DEM's native grid (Earth Engine weights a pixel by its share inside the circle). New column `elev_sd_r300` (the model feature); `tri_300`, the SD inside the fixed 300 m cell, becomes `meta_tri_300`. `--merge --processed` writes the merge to a given folder (the v4 staging folder `data/interim/v4_build`). | `pull_gee.py:62-63, 84-125, 194-212, 356, 406-411, 553` |
| D4 | **Binary fluvial class.** `fluvial_code` gives 1 (fluvial) or 0; a point in a soil-survey water map unit or in no map unit is NaN in both soil features, also when the water unit carries a flood class (`code_points`). `build` writes `data/processed_v4/FEATURES.txt`: the nine features of the Sources paragraph in its order (`V4_FEATURES`). | `pull_soil_points.py:72-92, 143-153, 182-188, 420-470` |
| D5 | **One-hot encoding.** `zoo.CATEGORICAL` = land cover, soil flood frequency, fluvial class. `model_inputs` adds one 0/1 column per class present in the full dataset (`onehot_levels`, sorted, no reference column dropped), NaN in all of a feature's columns where it is missing; `load` returns the model's input columns, so `cell_data`, `fit_one` and the model table use them. The imputers and hgb/xgb are unchanged. `audit_cell.py` refits on the same inputs. | `aotil/zoo.py:81, 137-192`; `aotil/audit_cell.py:945` |
| D6 | **Duplicates dropped.** After the degenerate drop, `split_population` drops a model whose 0/1 predictions on every ID test point and every OOD point (`prediction_key`) equal those of a model earlier in grid order; `<tag>_dropped.csv` gains `duplicate_of`. `audit_cell.py` checks that no two kept models share their predictions, that every duplicate names an earlier kept model with its flood share, and refits every duplicate against that model's stored predictions. | `aotil/zoo.py:459-530`; `aotil/audit_cell.py:332-385, 823-866` |
| D7 | **q = 6.** `region_match.numeric_features` is every feature but `zoo.CATEGORICAL`: land cover stays the exact-match key, the soil features are not matched; the caliper keeps the 5% rule. On the 2026-11-06 inputs the outputs are byte-identical (test 12). | `aotil/region_match.py:95, 114-124` |
| D8 to D10 | No code change (§2.2, §6.5). | |

### 3.2 Other changes

- `pull_hydro.py --processed DIR` (`set_processed`): the stages read the points from, and `--merge` rewrites, the dataset
  in DIR, so the v4 build never touches `data/processed`. **A bug I introduced and fixed in this run:** the first
  version reset the module's folder to `data/processed` whenever `--processed` was not given, which overrode the folder
  a caller had set. A private hydro test patches that folder, so while I ran the private suite its synthetic merge
  rewrote the real `data/processed/dataset_FL.parquet` (by about 17:45; noticed at 18:05). I restored it at 18:06 from
  the committed copy (`git show HEAD:...`, SHA-256 `9a27d7f8...`), checked it against `data/processed_w9` minus its two
  soil columns (equal, as asserted when w9 was built), and `git status` shows it clean. `FEATURES.txt` was rewritten
  with the same bytes. Nothing read the damaged file. The fix: `--processed` only moves the stages when it is given
  (`pull_hydro.py:654-661`), and `pull_gee.merge` reads its default folder when called (`pull_gee.py:406-411`). New
  `test_pull_hydro.py` and a new check in `test_pull_gee.py` pin both, with `data/processed` byte-identical before and
  after.
- New `aotil/variogram_ranges.py` (Part C4) and `aotil/fig_feature_boxes.py` (Part F).
- `aotil/README.md`: the rows of `zoo.py`, `audit_cell.py`, `region_match.py` and the two new scripts.

### 3.3 Tests

| Test | What it checks | Result |
|---|---|---|
| `test_sample_points.py` (new) | D1: `draw_valid` gives exactly n over several batches, a random n of the pool, sums the counts, stops on a zero share, caps the batch growth; D2: the uncapped quota is the plain multinomial (a 90% polygon draws 0.901 of 20,000); `area_uniform_candidates` places exactly n candidates, all on land, split by land area (0.811 against 0.818 expected); `band_filter` removes and counts each failure in order (7 synthetic points, one per test); `build_state` end to end: four groups of exactly 60 candidates, every survivor kept, no pair under 100 m, all on land, counts stored and read back by `--audit`; `group_problems` names a wrong candidate count, a missing row and a build without counts; `audit_dataset` B1 OK / WRONG and B17 OK / WRONG on the built table | ALL PASSED |
| `test_pull_gee.py` (new) | D3: the request buffers each point by 300 m (to 1 m) and uses the group's reducer on the group's grid; `derive` keeps `elev_sd_r300` and moves `tri_300` to `meta_tri_300`; `--merge --processed` and the default folder; online (`--online`): on a synthetic DEM whose elevation is the EPSG:5070 x coordinate, the SD within 300 m is r/2 = 150 m (a 300 m cell would give 86.6 m, a 300 m diameter 75 m), 50 m within 100 m, 0 on a flat DEM | offline ALL PASSED; online §4.1 |
| `test_pull_hydro.py` (new) | `--processed` moves the stages only when given; `data/processed` byte-identical | ALL PASSED |
| `test_pull_soil_points.py` | D4: binary fluvial class, water units NaN in both features even when rated; the nine-feature FEATURES.txt | ALL PASSED |
| `aotil/test_zoo.py` | D5: one column per class present, fixed and complete, NaN in all of a missing feature's columns, numeric features unchanged, a non-integer class or a clashing column refused; hgb and xgb receive the NaN, every other family has the median imputer first; D6: duplicates dropped with the first in grid order kept (also when only the scores differ), a model that agrees on one half only kept, a degenerate model never counted as a first; `audit_cell` passes a clean cell (every dropped setting, duplicates included, refits as stored) and fails a kept duplicate | ALL PASSED |
| `aotil/test_region_match.py` | D7: q = 6 of a nine-feature file (no soil, no land cover in z), q = 6 on w7, w9 and v4; the 2026-11-06 outputs byte for byte | ALL PASSED |
| `aotil/test_variogram_ranges.py` (new) | the two models and their 95% points; the empirical semivariogram by hand; a synthetic field with an 18 km effective range recovered (16.7 km); a pure nugget gets range 0 | ALL PASSED |
| `aotil/test_fig_feature_boxes.py` (new) | the sets are `zoo.cell_data`'s halves; the seven most common classes; the class shares | ALL PASSED |

## 4. Part C: the v4 dataset

### 4.0 Archive (step 1)

Every interim file keyed by point id moved (not deleted) to `data/interim/archive/sampling-v3/<same path>`, with the
SHA-256 of all 206 files recorded first (`sha256.txt`) and verified after the move (`MANIFEST.txt`): the v3 points
file, the 40 Earth Engine shards, the hydro files (`comid_FL`, `river_FL`, `streamcat_FL`), the soil table, the stale
`point_extras_FL` and the 160 per-batch soil caches (`pt_*.parquet`, keyed by a hash of point ids and coordinates).
Left in place, keyed by something else: the SDA map-unit, component and survey-area caches (`mu_*`, `comp_*`,
`sa_*`), the Atlas 14 block cache, the NFHL prep files and the land mask. (The auto-mode check refused my archive
script as one opaque `bash` call; the same moves as explicit `mv` commands went through.) The StreamCat pull of this
run returned exactly the archived table (36,497 catchments, every value equal), so its source did not change.

### 4.1 The points (D1, D2)

`sample_points.py FL --protocols band --candidates 25000 --state-polygons data/interim/gis/counties_FL.parquet
--land-mask data/interim/gis/nlcd2021_FL.tif` (`logs/2026-11-07/C2_sample_points.log`, 1,291 s). In CS terms: each
group draws random points until exactly 25,000 pass every test of the group; then a random-order sweep drops any
point within 100 m of a point already kept, first inside the group and then over all points.

| Group | Proposed | Failed a test | Valid | Candidates | After the group's thinning | Kept (after the statewide pass) | Kept / candidates |
|---|---:|---|---:|---:|---:|---:|---:|
| Zone A positives | 31,250 | 857 not on land | 30,393 | 25,000 | 24,731 | **24,729** | 98.9% |
| Zone A band negatives | 325,000 | 293,535 closer than 500 m, 22 beyond 5 km, 1,676 outside a panel, 1,428 in Zone D or AREA NOT INCLUDED, 452 in NFHL OPEN WATER, 1,039 not on land, 1,778 outside Florida, 43 lost to the ring cap | 25,027 | 25,000 | 23,926 | **23,574** | 94.3% |
| Zone AE positives | 31,250 | 4,581 not on land | 26,669 | 25,000 | 24,634 | **24,634** | 98.5% |
| Zone AE band negatives | 325,000 | 273,763 closer than 500 m, 16 beyond 5 km, 4,569 outside a panel, 2,046 in Zone D or AREA NOT INCLUDED, 7,155 in NFHL OPEN WATER, 10,226 not on land, 525 outside Florida, 22 lost to the ring cap | 26,678 | 25,000 | 23,895 | **23,530** | 94.1% |
| **Total** | 712,500 | | | **100,000** | 97,186 | **96,467** | 96.5% |

Each group needed one batch. The statewide pass dropped 352 Zone A band, 2 Zone A positive, 365 Zone AE band and
0 Zone AE positive points. Flood share per zone: Zone A 0.512, Zone AE 0.512 (0.5115). The positives come from
4,511 Zone A and 5,063 Zone AE polygons; with no cap, the largest Zone A polygon supplies 5.2% of Zone A positives
(its share of the land area) and the ten largest 29.5% (Zone AE: 2.4% and 13.3%); polygons under 1 km² supply 14.2%
(A) and 14.4% (AE). "Lost to the ring cap": draws the band's ring cap cut and its one redraw did not place (0.01%).

`sample_points.py --audit FL` (`logs/2026-11-07/C2_sample_points_audit.log`): 25,000 candidates per group, every
survivor kept: YES; 0 pairs closer than 100 m (minimum nearest-neighbour distance 100.02 m, median 408 m); positives
at distance 0 from an SFHA 100%; band negatives 500 to 4,990 m from the nearest SFHA; 47,104 of 47,104 inside a FIRM
panel; 0 in Zone D or AREA NOT INCLUDED; 0 not on land; 0 in NFHL OPEN WATER.

### 4.2 Soil (D4)

`pull_soil_points.py fetch --dataset data/interim/points/points_FL.parquet --check-single 20` (`logs/2026-11-07/C4_soil_fetch.log`, 895 s, 220 SDA requests at most 2 at a time): 96,461 of 96,467 points in a map unit (6 in none, 0.01%; limit 2%), 0 on an edge, 3,438 map units in 75 survey areas (versions 2026-08-26 to 2026-09-11). 1,928 points lie in a soil-survey water map unit and are NaN in both soil features. Non-water points with no flood class: 764 of 94,533 (0.81%; limit 5%). Fluvial class: 0 for 91,758 points, 1 for 2,775, missing for 1,934 (the 1,928 water-unit points and the 6 in no map unit). The batched lookup agreed with one call per point on 20 of 20 points, and the cross-check against the cached region-box soil polygons agreed on 37,114 of 37,114 points (`results/2026-11-07/soil/crosscheck_FL.csv`, no mismatch). Hydro fetch (`logs/2026-11-07/C3_hydro_*.log`): 99.1% of the points fall in an NHDPlus catchment (13,631 distinct); river distance median 1,539 m.

### 4.3 The split and the sets

The 50 km tiles that hold at least one point: 99. Drawn for testing (30%, seed 0): 30; dropped because they hold positives of only one zone: 3 (17_16, 28_8, 31_12); test tiles: 27; training tiles: 69, of which 66 hold Zone AE points (the other 3 hold only Zone A points, so no training row comes from them).

| Set | Points | Positives | Flood share | Tiles |
|---|---:|---:|---:|---:|
| Train | 36,758 | 18,417 | 0.501 | 66 |
| ID test | 10,986 | 5,838 | 0.531 | 27 |
| OOD | 17,346 | 8,675 | 0.500 | 27 |
| All | 96,467 | 49,363 | 0.512 | 99 |

Train = the Zone AE points of the training tiles; ID test = the Zone AE points of the test tiles; OOD = the Zone A points of the test tiles; All = every point of the dataset (both zones, all 99 tiles).

### 4.4 Everything the Data statistics paragraph needs

**Numeric features** (topographic wetness as the index, the StreamCat value / 100; distances in log10 m, as the models see them; terrain variability and HAND in m; slope in degrees):

| Feature | Set | n | Mean | SD | Median | 5th pct | 95th pct | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Slope | Train | 36,758 | 0.954 | 1.405 | 0.504 | 0.028 | 3.473 | 0.000 | 33.663 |
| Slope | ID test | 10,986 | 1.008 | 1.753 | 0.428 | 0.027 | 4.078 | 0.000 | 32.537 |
| Slope | OOD | 17,346 | 1.157 | 1.825 | 0.523 | 0.036 | 4.521 | 0.000 | 24.904 |
| Slope | All | 96,467 | 0.983 | 1.496 | 0.494 | 0.030 | 3.612 | 0.000 | 46.286 |
| Terrain variability | Train | 36,758 | 0.967 | 1.213 | 0.572 | 0.066 | 3.479 | 0.000 | 19.008 |
| Terrain variability | ID test | 10,986 | 1.088 | 1.701 | 0.442 | 0.082 | 4.691 | 0.000 | 16.571 |
| Terrain variability | OOD | 17,346 | 1.475 | 1.963 | 0.651 | 0.070 | 5.692 | 0.000 | 24.165 |
| Terrain variability | All | 96,467 | 1.116 | 1.496 | 0.552 | 0.074 | 4.257 | 0.000 | 24.165 |
| HAND | Train | 36,719 | 2.909 | 4.494 | 1.400 | 0.000 | 11.400 | 0.000 | 48.500 |
| HAND | ID test | 10,985 | 3.122 | 5.603 | 1.100 | 0.000 | 14.100 | 0.000 | 58.500 |
| HAND | OOD | 17,346 | 4.271 | 6.615 | 1.600 | 0.000 | 18.900 | 0.000 | 62.700 |
| HAND | All | 96,427 | 3.344 | 5.311 | 1.400 | 0.000 | 14.100 | 0.000 | 62.700 |
| Wetness | Train | 36,035 | 12.215 | 2.715 | 11.493 | 8.814 | 17.598 | 7.402 | 21.409 |
| Wetness | ID test | 10,911 | 12.396 | 2.820 | 11.844 | 8.414 | 16.648 | 6.936 | 21.233 |
| Wetness | OOD | 17,346 | 11.624 | 2.863 | 10.780 | 8.196 | 16.766 | 6.936 | 21.470 |
| Wetness | All | 95,577 | 12.011 | 2.732 | 11.311 | 8.523 | 17.039 | 6.936 | 22.515 |
| Dist. to river | Train | 36,758 | 3.053 | 0.658 | 3.117 | 1.859 | 4.030 | 0.000 | 5.084 |
| Dist. to river | ID test | 10,986 | 3.136 | 0.737 | 3.175 | 1.837 | 4.305 | 0.000 | 4.623 |
| Dist. to river | OOD | 17,346 | 3.265 | 0.762 | 3.257 | 1.935 | 4.536 | 0.000 | 4.724 |
| Dist. to river | All | 96,467 | 3.145 | 0.692 | 3.187 | 1.894 | 4.246 | 0.000 | 5.084 |
| Dist. to coast | Train | 36,758 | 4.128 | 0.622 | 4.255 | 2.998 | 4.886 | 0.000 | 4.998 |
| Dist. to coast | ID test | 10,986 | 4.445 | 0.549 | 4.651 | 3.318 | 4.951 | 0.141 | 5.074 |
| Dist. to coast | OOD | 17,346 | 4.636 | 0.308 | 4.694 | 4.035 | 4.974 | 2.365 | 5.069 |
| Dist. to coast | All | 96,467 | 4.377 | 0.532 | 4.524 | 3.327 | 4.932 | 0.000 | 5.074 |

**Categorical features** (share of each set's points; soil shares over the points with a soil class):

| Feature | Class | Train | ID test | OOD | All |
|---|---|---:|---:|---:|---:|
| Land cover | 90 woody wetland | 31.6% | 27.6% | 30.2% | 31.7% |
| Land cover | 42 evergreen forest | 13.9% | 9.9% | 19.5% | 17.0% |
| Land cover | 81 pasture and hay | 7.8% | 8.5% | 10.9% | 9.5% |
| Land cover | 95 herbaceous wetland | 9.3% | 12.0% | 11.0% | 9.1% |
| Land cover | 82 crops | 4.2% | 18.8% | 8.9% | 7.2% |
| Land cover | 21 developed open space | 8.8% | 6.7% | 4.9% | 6.8% |
| Land cover | 22 low-intensity developed | 8.9% | 5.8% | 2.5% | 5.6% |
| Land cover | the other classes | 15.4% | 10.6% | 12.1% | 13.0% |
| Land cover | woody or herbaceous wetland (90, 95) | 41.0% | 39.6% | 41.2% | 40.8% |
| Land cover | forest (41, 42, 43) | 14.6% | 10.2% | 20.4% | 17.8% |
| Flood frequency | 1 Very rare | 0.0% | 0.1% | 0.0% | 0.0% |
| Flood frequency | 2 Rare | 1.8% | 0.4% | 0.5% | 1.2% |
| Flood frequency | 3 Occasional | 2.9% | 3.5% | 0.7% | 2.2% |
| Flood frequency | 4 Frequent | 12.5% | 12.9% | 12.8% | 12.1% |
| Flood frequency | 5 Very frequent | 7.8% | 1.2% | 0.0% | 3.3% |
| Flood frequency | Rare or above | 25.0% | 18.0% | 14.0% | 18.7% |
| Fluvial | 1 fluvial | 3.2% | 2.1% | 3.5% | 2.9% |

Land cover holds 14 of NLCD's 16 classes (21-24, 31, 41-43, 52, 71, 81, 82, 90, 95; no 11 open water by design and no 12 ice). The seven most common over all points: 90, 42, 81, 95, 82, 21, 22 (82 crops now ranks above 21 developed open space).

**Missing values** (count and share of all points; share per set; the reasons):

| Feature | Missing | Share | Train | ID test | OOD | Reasons |
|---|---:|---:|---:|---:|---:|---|
| landcover | 0 | 0.00% | 0.00% | 0.00% | 0.00% | - |
| soil_floodfreq | 2,698 | 2.80% | 3.09% | 8.17% | 1.57% | 1,928 in a soil-survey water map unit; 6 in no soil map unit; 764 a land map unit with no flood class |
| soil_fluvclass | 1,934 | 2.00% | 2.03% | 5.98% | 1.38% | 1,928 in a soil-survey water map unit; 6 in no soil map unit |
| slope | 0 | 0.00% | 0.00% | 0.00% | 0.00% | - |
| elev_sd_r300 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | - |
| hand | 40 | 0.04% | 0.11% | 0.01% | 0.00% | 40 no MERIT Hydro value at the point |
| wetindexcat | 890 | 0.92% | 1.97% | 0.68% | 0.00% | 889 outside every NHDPlus catchment; 1 a catchment without a StreamCat value |
| log10_dist_river | 0 | 0.00% | 0.00% | 0.00% | 0.00% | - |
| log10_dist_coast | 0 | 0.00% | 0.00% | 0.00% | 0.00% | - |
| either soil feature | 2,698 | 2.80% | 3.09% | 8.17% | 1.57% | 1,928 in a soil-survey water map unit; 6 in no soil map unit; 764 a land map unit with no flood class |

**The qualitative claims of the draft Data statistics paragraph** (docs/PENDING-CHANGES.md item 5), on the v4 data:
- flood share near one half in every set: 0.501 / 0.531 / 0.500 (Train / ID test / OOD): holds;
- slope, terrain variability and HAND small with long right tails: medians 0.49°, 0.55 m and 1.4 m against 95th
  percentiles of 3.6°, 4.3 m and 14.1 m and maxima of 46°, 24 m and 63 m (all points): holds;
- most points in wetlands or forest: 40.8% wetland (90, 95) and 17.9% forest (41-43), 58.7% together: holds;
- most points on soil rated as never flooding: 81.3% "None" among points with a class: holds;
- Zone A points farther from the coast than Zone AE: median log10 distance 4.69 (OOD) against 4.65 (ID test) and
  4.26 (Train); means 4.64 against 4.45 and 4.13: holds;
- missing soil features mostly in soil-survey water areas: 1,928 of the 2,698 (71%): holds;
- missing wetness mostly outside NHDPlus catchments: 889 of the 890: holds.
Two things the current paragraph does not say: HAND is missing too (40 points, 0.04%; all Zone AE, 34 of them herbaceous
wetland near the coast), and the ID test set holds far more crops (18.7%) than Train (4.2%) or OOD (8.9%).

### 4.5 The audits (Part C step 3)

`audit_dataset.py --n-features 9` (`results/2026-11-07/data_audit/audit.md`, `logs/2026-11-07/C7_audit_dataset.log`):
**14 OK, 0 WRONG, 4 DECIDE**. The new and changed checks: B1 (25,000 candidates per group, every survivor kept, the
table equal to the points file: OK), B10 (no pair under 100 m, minimum 100.02 m; positives at distance 0; band
negatives 500 to 4,990 m: OK), B17 (new: 0 points off land per the mask, 0 with land cover 11, 0 or missing, and of
the 47,104 band negatives 0 outside 500 m to 5 km, 0 outside every panel, 0 in Zone D or AREA NOT INCLUDED, 0 in NFHL
OPEN WATER, 0 outside the 67 Florida county polygons: OK). B16 recomputes every band negative's nearest-SFHA distance
and reproduces the stored one within 1 m for 47,104 of 47,104. The DECIDE items are of the kinds earlier builds had:
B6 (the dataset keeps 36 non-feature columns such as curvature and the StreamCat extras), B7 (lon/lat are NAD83,
about 1 m from WGS84; **0** points now lie outside the Florida county polygons, against 2 positives in v3), B14 (82
band negatives whose panel has no effective date), A4.3 (46 of 1,823 point pairs closer than 125 m share their HAND and
upstream area, 2.5%, as MERIT's 90 m cells predict). **A bug fixed here:** B6 first called `elev_sd_r300` a forbidden
name, because its rule bans any name containing "elev" to keep absolute elevation (a location fingerprint) out of the
features. The SD of elevation within 300 m is relative relief, like the old `tri_300`, so the rule now exempts
`elev_sd_*` (`audit_dataset.py:514-526`, test 7 of `test_sample_points.py`); absolute elevation stays banned.

**Part C stops:** the smallest group keeps 23,530 points (limit 20,000); the audit has no WRONG; 0.01% of points lie
in no soil map unit (limit 2%); 0.81% of non-water points lack a flood class (limit 5%). **No stop.**

## 5. Spatial autocorrelation ranges (Part C step 4)

`aotil/variogram_ranges.py --data data/processed_v4` (`results/2026-11-07/variogram/`, `logs/2026-11-07/C8_variogram.log`).
Method, as blockCV fits its ranges (Valavi et al. 2019): a random 20,000-point subsample of all 96,467 points (seed 0);
for each numeric feature and each one-hot column with at least 1% prevalence, the empirical semivariogram (half the
mean squared difference between two points, by distance) in 1 km lags out to 200 km, on the EPSG:5070 coordinates and
the pairs where both values are present; a spherical and an exponential model, each with a nugget (the jump at zero
distance), fitted by weighted least squares with gstat's default weights (pairs over squared lag distance); the fit with
the smaller weighted residual kept. The effective range is the distance where the fitted model's rise above its
nugget reaches 95% of its partial sill (a ln 20, about 3a, for the exponential; 0.81a for the spherical; a = the fitted
range parameter). A column with no structure beyond the first 1 km lag (partial sill under 1% of the total) gets range
0. A categorical feature's range is the largest over its columns.

| Feature | Range (km) | Widest column | Model |
|---|---:|---|---|
| landcover | 72.7 | landcover_81 | exponential |
| soil_floodfreq | 27.4 | soil_floodfreq_5 | exponential |
| soil_fluvclass | 9.5 | soil_fluvclass_0 | exponential |
| slope | 7.8 | slope | exponential |
| elev_sd_r300 | 24.5 | elev_sd_r300 | exponential |
| hand | 14.3 | hand | exponential |
| wetindexcat | 106.9 | wetindexcat | exponential |
| log10_dist_river | 27.7 | log10_dist_river | exponential |
| log10_dist_coast | 95.2 | log10_dist_coast | spherical |

Per column (one-hot columns under 1% prevalence left out: land cover 31, 41, 43; flood class 1, Very rare): land cover 21,
22, 23, 24, 52 and 71 show no structure beyond 1 km (range 0); 42 evergreen forest 10.2 km, 90 woody wetland 13.8,
95 herbaceous wetland 14.6, 82 crops 44.7 and 81 pasture and hay 72.7; flood classes None 13.4, Rare 9.0,
Occasional 4.4, Frequent 13.7, Very frequent 27.4; fluvial 9.5 (both columns are one indicator and its complement).

**The largest range is 106.9 km (topographic wetness), and three features exceed 50 km: wetness 106.9, distance to the
coast 95.2 and land cover 72.7 (pasture).** Per the brief, the tiles stay as they are and the Split sentence is flagged
(F2, §8.2): the 50 km tiles are not wider than every feature's range. The fits follow the empirical curves (wetness: 0.35
of its sill at 10 km, 0.72 at 50, 0.94 at 100, 1.23 at 200 km, still rising, a statewide gradient; coast distance: 0.62 at 50
km, 0.93 at 100). Six of the nine features (slope 7.8, fluvial 9.5, HAND 14.3, terrain variability 24.5, flood frequency
27.4, river distance 27.7 km) stay under 50 km.

## 6. Part D: the cell, and Part E: the full run

### 6.1 The cell (Part D)

`aotil/zoo.py --data data/processed_v4 --out data/interim/aotil/cells_v4_2026-11-07` (`logs/2026-11-07/D1_zoo.log`, the 369 settings fitted in 32 s) and `aotil/audit_cell.py` (`logs/2026-11-07/D2_audit_cell.log`, `results/2026-11-07/audit/FL_band_AE_s0_audit.json`): **verdict PASS** (0 FAIL, 0 WARN): the 290 kept and 79 dropped settings cover the grid once each in grid order; no kept model is degenerate and no two kept models share their predictions on both test halves; 10 refits (every family) reproduce the stored scores exactly (max difference 0.0); all 79 dropped settings refit as stored (the degenerate one with its flood share, every duplicate with the predictions of the kept model it repeats); every column of both halves matches the dataset through point_id, and split_tiles + subset re-derive the same points in the same order.

| Family | Settings | Degenerate | Duplicate | Kept | ID accuracy | Interpolating (train acc >= 0.99) |
|---|---:|---:|---:|---:|---|---:|
| dt | 40 | 0 | 18 | 22 | 0.619 to 0.768 | 1 |
| et | 54 | 0 | 9 | 45 | 0.622 to 0.776 | 9 |
| hgb | 48 | 0 | 10 | 38 | 0.671 to 0.797 | 1 |
| lr | 50 | 0 | 18 | 32 | 0.724 to 0.737 | 0 |
| mlp | 45 | 1 | 0 | 44 | 0.532 to 0.785 | 0 |
| nb | 30 | 0 | 2 | 28 | 0.536 to 0.684 | 0 |
| rf | 54 | 0 | 7 | 47 | 0.671 to 0.778 | 9 |
| xgb | 48 | 0 | 14 | 34 | 0.721 to 0.788 | 1 |
| Total | 369 | 1 | 78 | 290 | 0.532 to 0.797 | 21 |

- **Model inputs:** the 9 features become 28 input columns after one-hot (6 numeric, 14 land-cover
  classes, 6 flood classes, 2 fluvial classes); every model uses all 28 (each varies in the training rows).
  Random forests and extra trees therefore draw 1, 5 (sqrt) or
  28 (all) input columns per split.
- **Kept N = 290**; the 60/20/20 split (random_state 42): **174 selection, 58
  validation, 58 test models**. The one degenerate setting is an MLP (grid_idx 193) that predicts dry
  everywhere. The 78 duplicates are settings that reach the same predictions as an earlier one: decision trees
  with large leaves or little depth (9 of the 18 with leaves of at least 2,000 points), random forests and extra trees
  of depth 1 to 3 that consider every feature at each split (15), boosting runs of depth 1 to 3 with few iterations or
  a small learning rate (hgb 10, xgb 14), logistic regressions that end at the same fit (9 of the 18 with C = 100, 8
  stopped after one iteration), and 2 naive Bayes settings.
- **ID accuracy** 0.532 to 0.797, SD 0.064; quantiles 0/10/25/50/75/90/100%:
  0.532 0.622 0.727 0.743 0.761 0.775 0.797. Below 0.70: 52 models
  (nb 28, et 10, mlp 5, rf 5, dt 3, hgb 1): the weak end still leans on naive Bayes.
- **Probit r(ID, OOD)** on all Zone A test points: **+0.905** over all 290 kept models;
  **+0.838** [+0.740, +0.901] over the 58 test models.
  Without naive Bayes (N1): +0.847 over all, +0.772 over the 56 test models.
- **OODSelect's model-count rule** (`n_models_needed`): **20** models before r moves by less than 1%.
- All 58 test models have distinct predictions (D6), so the Fisher interval counts 58 distinct
  observations. Interpolating models: 21 of 290 (3 of the test models).
- **Part D stops:** kept N 290 (limit 200); lowest ID accuracy 0.532 (stop above 0.60); smallest
  family dt with 22 models (limit 10); probit r +0.905 (limit 0.5). **No stop.**


### 6.2 What ran (Part E)

Every command, setting and seed of 2026-11-06, on the v4 cell (`logs/2026-11-07/T_chain.sh`, `C_chain.sh`,
`D_chain.sh`, copied from `logs/2026-11-06/` with only the paths changed: `cells_v4_2026-11-07`,
`data/processed_v4`, outputs under `results/2026-11-07/`):
- the shift, pooled and per tile (`o1_shift.py`), its table (`o1_table.py`) and the tile-shuffle null
  (`o1_tile_null.py`, seeds 1 to 3);
- the searches, all with the squared budget penalty, T = 100 TPE trials, at most 50,000 steps, patience 500,
  search seed 0, k = 10: OODSelect (classic), Laplacian (μ = 3) and GTV (μ = 3) over the budget grid; the
  shuffled-accuracy controls of all three at S = 250, 500 and 1,000 (null seeds 1 to 3: 27); both family-disjoint
  directions for classic and Laplacian (12); μ = 1 and μ = 10 (6).
  **84 searches, not 81.** The budget grid is the code's rule, every budget of the authors' list below the number of
  OOD points (`aotil/oodselect_paper.py:501-504`, `budgets_for`). With 12,531 Zone A test points on 2026-11-06 that
  stopped at S = 10,000 (12 budgets, 36 grid searches, 81 in all); with 17,346 now it also takes **S = 15,000** (13
  budgets, 39 grid searches, 84 in all). No setting changed; the three extra searches are the same command at one more
  budget. The analysis scripts read the grid from the same rule (`op.budgets_for`) instead of a typed list.
- the selection audits, smoothed-classic, the regions (Laplacian S = 250 and 500, classic and GTV S = 500),
  `regions_land.py`, the region description, the memorized-noise check (`o5_interpolation.py`), `region_match.py`
  and both figures.

### 6.3 The shift (C1) and the tile-shuffle null

`logs/2026-11-07/T1_o1_shift.log`, `results/2026-11-07/o1_shift.csv` (scope all), 58 test models, 27 test tiles,
2,000 tile-bootstrap draws. Gap = ID (held-out Zone AE) minus Zone A; a positive gap means the models do worse on
Zone A.

| Statistic | 2026-11-07 (v4) | 2026-11-06 |
|---|---|---|
| Pooled accuracy gap | −0.034 [−0.111, +0.037] | −0.008 [−0.065, +0.048] |
| Pooled AUROC gap | −0.038 [−0.122, +0.033] | −0.016 [−0.074, +0.039] |
| Pooled F1 gap | −0.016 [−0.145, +0.100] | +0.019 [−0.067, +0.092] |
| Per tile (unweighted), accuracy | **+0.017 [−0.042, +0.067]**; worse on Zone A in 19 of 27 tiles; 75% of models worse; Wilcoxon p = 0.091 | +0.046 [+0.021, +0.074]; 22 of 28; 94% |
| Per tile, AUROC (25 tiles: in 2 tiles, Columbia 25_18 and Miami-Dade 31_8, one half has a single class) | +0.029 [−0.041, +0.081] | +0.043 [+0.007, +0.079] |
| Per tile, F1 | +0.068 [−0.012, +0.131]; 22 of 27 tiles | +0.093 [+0.053, +0.139] |
| Per tile, the 15 strongest test models, accuracy | +0.059 [+0.007, +0.107]; 23 of 27 tiles; all 15 worse | +0.079 |
| Tile-shuffle null (zones shuffled within tile and label, seeds 1 to 3), accuracy | −0.022, −0.015, −0.004 (full precision −0.022487, −0.015240, −0.004261) | within 0.005 |
| Null, AUROC | −0.001, −0.008, +0.012 | within 0.023 |
| Null, F1 | +0.013, +0.024, +0.032 | +0.035 to +0.041 |
| corr_te on all Zone A test points (C2) | **+0.838** [+0.740, +0.901] | +0.806 [+0.707, +0.874] |
| The Palm Beach tile (30_10) | 1,502 Zone AE and 193 Zone A test points; mean accuracy 0.488 | 1,264 and 192; 0.514 |

**Reading.** Pooled, the models are slightly *more* accurate on Zone A (−0.034), and every pooled interval contains
0, as before. Per tile, the accuracy gap shrank from +0.046 to +0.017 and **its interval now contains 0**; it is also
smaller than the largest gap the tile-shuffle null produces by chance (0.022 in absolute value), so the per-tile
accuracy gap is no longer distinguishable from no shift. AUROC (+0.029) also has an interval containing 0, but it is
above its null (at most 0.012); F1 (+0.068) is above its null (0.013 to 0.032) with an interval that just reaches 0
(−0.012). The 15 strongest test models still do worse on Zone A in accuracy, +0.059 [+0.007, +0.107]. Full-precision
null values: `logs/2026-11-07/E_null_precision.log` (`E_null_precision.py` reruns `o1_tile_null.null_rows` with the
same seeds; the CSV keeps 4 decimals).

### 6.4 Claims C1 to C7

As fixed in `docs/TASK-RERUN-ALL-2026-10-25.md` §2, checked by `logs/2026-11-07/E_claims.py`
(`logs/2026-11-07/E_claims.log`, `results/2026-11-07/claims.json`). Two terms: **pieces** are the connected
components of the subset in the 10-nearest-neighbor graph (how many separate blobs it falls into); **purity** is the
mean, over the selected points, of the share of a point's graph neighbors that are also selected (1 for solid blobs,
about S/m for a random subset; S/m = 0.029 at S = 500).

| Claim | Test (as fixed) | 2026-11-07 (v4) | Verdict | 2026-11-06 |
|---|---|---|---|---|
| C1 shift | Pooled and per-tile gaps (reported) | Pooled accuracy −0.034 [−0.111, +0.037]; per tile **+0.017 [−0.042, +0.067]**, 19 of 27 tiles, 75% of models; tile-shuffle null up to 0.022 (§6.3) | **Reported**; the per-tile accuracy gap is no longer distinguishable from the null (flag F1) | Pooled −0.008; per tile +0.046 [+0.021, +0.074], 22 of 28, 94% |
| C2 accuracy-on-the-line | r on all Zone A test points, test models, > +0.3 | **+0.838** [+0.740, +0.901] | **Holds** | +0.806 |
| C3 OODSelect inverts, scattered | Classic r_te < −0.3 and below all three shuffled classic runs at S = 500 | **−0.972** [−0.984, −0.954]; 293 pieces, 25 of 27 tiles, purity 0.106. Shuffled: −0.035, −0.116, +0.401 | **Holds** | −0.977; 291 pieces, 27 of 28 tiles |
| C4 regions at small cost | Laplacian r_te < −0.3, purity ≥ 0.5, pieces ≤ classic pieces / 5, purity above smoothed-classic at S = 250, 500 and 1,000 | **−0.918** [−0.951, −0.865], purity 0.504, **105 pieces (classic/5 = 59)**. Purity, ours vs smoothing: **0.295 vs 0.573, 0.504 vs 0.655, 0.658 vs 0.713**; smoothing r_te −0.920, −0.913, −0.835 | **Does not hold**: r_te and purity hold; pieces and the smoothing comparison fail | Does not hold (71 pieces vs 58; purity below smoothing) |
| C5 not memorized noise | r on the held-out test models below 99% training accuracy < −0.3 (Laplacian S = 500) | **−0.915** [−0.950, −0.858], 55 models. Family-disjoint at S = 500: Laplacian trees→others −0.739, others→trees −0.562 (classic −0.869, −0.704) | **Holds** | −0.905, 67 models; −0.758, −0.561 |
| C6a familiar inputs | Pooled unmatched share of the regions ≤ 0.10 | **0.013** (3 of 227 points); per region 0 to 0.050; none above 0.10 | **Holds** | 0.029; Hamilton 0.115 above 0.10 |
| C6b different labels | Pooled η ≥ the Zone AE baseline η + 0.10 | **0.758 against 0.307** (needs ≥ 0.407); every region above the baseline (lowest Wakulla 0.371) | **Holds** | 0.649 against 0.297 |
| C7 GTV (descriptive) | GTV S = 500 | −0.846 [−0.906, −0.752]; purity 0.786; **29 pieces**; 10 tiles; 7 regions holding 383 points | Reported | −0.858; purity 0.916; 15 pieces; 11 tiles; 9 regions, 440 points |

C4 fails more clearly than on 2026-11-06. Smoothing OODSelect's final weights over each point's 10 nearest neighbors
(`smoothed_classic_knn10`, computed by the selection audit) now keeps the inversion (−0.913 at S = 500, against
−0.519 on 2026-11-06) with fewer pieces than our loss (48 against 105) and higher purity at all three budgets. The
paper's argument that "coherence must be part of the objective" rests on smoothing losing the inversion, so it no
longer holds (flag F7). The Laplacian search's own coherence also dropped: purity 0.504 at S = 500 (0.671 before) and
0.295 at S = 250 (0.651 before).

### 6.5 The paper's rule statements

- **μ rule (holds, no flag).** Purity at S = 500: μ = 1 **0.305**, μ = 3 **0.504**, μ = 10 0.626. The smallest μ of
  {1, 3, 10} with purity ≥ 0.5 is **μ = 3**, so "choose μ = 3 from a sensitivity analysis" stands. The margin is thin
  (0.504 against 0.5), and at S = 250 none of the three reaches 0.5. The sweep at S = 250 / 500 / 1,000:

| μ | Purity | r_te | Pieces | Tiles |
|---:|---|---|---|---|
| 0 (OODSelect) | .057 / .106 / .167 | −.968 / −.972 / −.946 | 184 / 293 / 446 | 25 / 25 / 27 |
| 1 | .262 / .305 / .486 | −.937 / −.926 / −.949 | 98 / 175 / 222 | 20 / 23 / 24 |
| 3 (the paper) | .295 / .504 / .658 | −.935 / −.918 / −.919 | 110 / 105 / 130 | 20 / 22 / 21 |
| 10 | .470 / .626 / .822 | −.882 / −.894 / −.885 | 73 / 71 / 46 | 25 / 19 / 17 |

- **Largest inverted budget.** The largest grid budget with corr_te ≤ −0.3 is **S = 10,000** for OODSelect (−0.919),
  the Laplacian (−0.839) and GTV (−0.746); at 15,000 none inverts (+0.466, +0.476, +0.433), and every budget from 10 to
  10,000 inverts for all three. The Budget penalty sentence now says 10,000 (was 7,500). S = 15,000 is 86% of the
  17,346 Zone A test points, so a subset that large is close to all of Zone A (+0.838).
- **N1, without naive Bayes (report only).** Probit r(ID, OOD) on all Zone A test points without the 28 naive Bayes
  models: **+0.847** over 262 models (with them +0.905); over the 56 test models that are not nb, +0.772. corr_te of
  the Laplacian S = 500 subset over those 56 test models: **−0.897** [−0.938, −0.829] (all 58: −0.918). Neither the
  aggregate trend nor the inversion leans on naive Bayes, although nb holds 28 of the 52 models below 0.70 ID accuracy.

### 6.6 The searches across budgets

corr_te, then (purity, pieces); random purity is S/m with m = 17,346 (`logs/2026-11-07/E_gtv_vs_laplacian.log`,
`E_budget_table.log`). The paper has no budget table; this is for the record.

| S | Random purity | OODSelect | Laplacian | GTV |
|---:|---:|---|---|---|
| 10 | .00 | −.76 (.00, 10) | −.50 (.58, 2) | −.65 (.02, 9) |
| 20 | .00 | −.85 (.00, 20) | −.88 (.03, 18) | −.75 (.43, 2) |
| 50 | .00 | −.94 (.01, 47) | −.70 (.67, 8) | −.83 (.96, 2) |
| 100 | .01 | −.97 (.02, 89) | −.80 (.71, 13) | −.84 (.88, 4) |
| 250 | .01 | −.97 (.06, 184) | −.94 (.30, 110) | −.82 (.92, 8) |
| 500 | .03 | −.97 (.11, 293) | −.92 (.50, 105) | −.85 (.79, 29) |
| 750 | .04 | −.97 (.13, 404) | −.93 (.67, 73) | −.84 (.87, 28) |
| 1,000 | .06 | −.95 (.17, 446) | −.92 (.66, 130) | −.82 (.94, 20) |
| 2,500 | .14 | −.96 (.34, 554) | −.92 (.69, 246) | −.82 (.89, 77) |
| 5,000 | .29 | −.95 (.40, 499) | −.92 (.72, 254) | −.79 (.92, 125) |
| 7,500 | .43 | −.94 (.55, 246) | −.87 (.81, 197) | −.76 (.94, 102) |
| 10,000 | .58 | −.92 (.73, 172) | −.84 (.90, 135) | −.75 (.95, 79) |
| 15,000 | .86 | +.47 (.92, 31) | +.48 (.95, 46) | +.43 (.95, 41) |

**Reading.** Every search inverts at every budget up to 10,000, more strongly than on 2026-11-06 at the large budgets
(OODSelect −0.92 at 10,000 against +0.07 before). The Laplacian's coherence is uneven: at S = 20 (purity 0.03, 18
pieces of 20 points) and S = 250 (0.30, 110 pieces) it is far less coherent than at the budgets around them (0.67 to
0.71 at S = 50 and 100, 0.50 to 0.67 at 500 to 1,000). GTV is the most coherent search at every budget from 20 to
10,000 (2 to 125 pieces) and inverts 0.07 to 0.13 less than the Laplacian from S = 250 to 10,000. The largest gap
between OODSelect and the Laplacian is 0.262 (S = 10); from S = 100 up it is at most 0.163. At S = 500, GTV and the
Laplacian share 273 of their 500 points (Jaccard overlap 0.376).

### 6.7 The controls in detail

**Shuffled ID accuracies** (the selection and validation models' Zone AE accuracies permuted before the search, null
seeds 1 to 3; a search that still finds an inversion here is finding noise). corr_te on the 58 test models:

| Search | S = 250 (seeds 1, 2, 3) | S = 500 | S = 1,000 | Median of 9 | Most negative | Real run lower in |
|---|---|---|---|---:|---:|---:|
| OODSelect | +0.130, +0.032, −0.252 | −0.035, −0.116, +0.401 | −0.169, −0.032, +0.026 | **−0.032** | −0.252 | 9 of 9 |
| Laplacian | +0.762, +0.450, −0.352 | +0.859, +0.594, −0.403 | +0.845, +0.599, −0.155 | **+0.594** | **−0.403** | **9 of 9** |
| GTV | +0.736, +0.642, +0.017 | +0.858, +0.622, −0.104 | +0.845, +0.631, −0.138 | +0.631 | −0.138 | 9 of 9 |

On 2026-11-06 the medians were +0.495, −0.662 and +0.251 and the most negative runs −0.779, −0.883 and −0.674; 8 of
the 27 shuffled runs reached −0.6 or lower. Now none does. The real searches are far below their shuffled runs at
every budget and seed. The paper's "Some shuffled runs still reach −0.40, so a negative correlation alone is weak
evidence" is still true as written, but its support is weaker (−0.40 instead of −0.88; §8.3).

**Memorized noise** (`results/2026-11-07/o5/A_interpolation.csv`). 21 of the 290 models reach 99% training accuracy
(dt 1, et 9, hgb 1, rf 9, xgb 1), 3 of them test models. Laplacian S = 500: all 58 test models −0.918 [−0.951, −0.865];
**without the 3, −0.915 [−0.950, −0.858]** (55 models); over all 290 models −0.959 (circular: it includes the models
the search selected with). The three families with no model at 99% are lr, mlp and nb; over all their models
(circular) the subset gives −0.984 (32 models), −0.896 (44) and −0.952 (28). OODSelect S = 500 without the 3:
−0.972.

**Family-disjoint** (select with the tree families dt, rf, et, hgb and xgb, 186 models, and test on lr, mlp and nb,
104 models, then the reverse). corr_te at S = 250 / 500 / 1,000: Laplacian trees→others −0.680 / **−0.739** / −0.718,
others→trees −0.541 / **−0.562** / −0.589; OODSelect −0.872 / −0.869 / −0.874 and −0.708 / −0.704 / −0.685. On all Zone A,
the 104 non-tree models give r = +0.950 and the 186 tree models +0.758, so both test sets follow the line in aggregate.

### 6.8 Every number the paper states, 2026-11-06 → 2026-11-07

Each was recomputed from the result files by `logs/2026-11-07/G3_check_paper_numbers.py` (ALL MATCH,
`logs/2026-11-07/G3_check_paper_numbers.log`); the edits themselves are in the Appendix.

| Where | Number | 2026-11-06 | 2026-11-07 |
|---|---|---|---|
| Introduction, Appendix A | N | 361 | **290** |
| Models | N; selection / validation / test models | XX; XX / XX / XX | **290; 174 / 58 / 58** |
| tab:search caption | test models | 73 | **58** |
| Introduction | per-tile accuracy gap; aggregate r | 0.046; 0.81 | **0.017; 0.84** |
| Introduction, tab:search | OODSelect r_te; pieces | −0.98; 291 | **−0.97; 293** |
| Introduction, Results | Laplacian pieces; cost of coherence | 71; 0.077 | **105; 0.054** |
| Introduction | regions' disagreement with their matches | 65% | **76%** |
| Sampling | points after thinning | XX | **96,467** |
| Split | largest autocorrelation range | XX | **107 km** |
| Data statistics | missing soil / wetness | XX / XX | **2.8% / 0.9%** |
| Budget penalty | S = 500 as share of the OOD points; largest inverted budget | 4.0%; 7,500 | **2.9%; 10,000** |
| Graph | median distance to the 10th neighbor | 2.54 km | **2.17 km** |
| Matching | ε | 1.39 | **1.30** |
| Pooling hides the shift | pooled accuracy gap | −0.008 [−0.065, 0.048] | **−0.034 [−0.111, 0.037]** |
| Pooling hides the shift | Palm Beach tile: Zone AE / Zone A points; accuracy | 1,264 / 192; 0.51 | **1,502 / 193; 0.49** |
| Pooling hides the shift | tiles worse on Zone A | 22 of 28 | **19 of 27** |
| Pooling hides the shift | per-tile gap: accuracy [CI], AUROC, F1; models worse | 0.046 [0.021, 0.074], 0.043, 0.093; 94% | **0.017 [−0.042, 0.067], 0.029, 0.068; 75%** |
| Pooling hides the shift | null: accuracy and AUROC within; F1 | 0.023; 0.035 to 0.041 | **0.022; 0.013 to 0.032** |
| Results, tab:search | All Zone A r [CI]; tiles | +.81 [.71, .87]; 28 | **+.84 [.74, .90]; 27** |
| tab:search, Results | Random; Hardest | +.79; −.27 | **+.78; .00** |
| tab:search | OODSelect [CI]; tiles | [−.99, −.96]; 27 | **[−.98, −.95]; 25** |
| tab:search, Results | Laplacian r_te [CI]; pieces; tiles | −.90 [−.94, −.84]; 71; 20 | **−.92 [−.95, −.86]; 105; 22** |
| tab:search, Results | GTV r_te [CI]; pieces; tiles | −.86 [−.91, −.78]; 15; 11 | **−.85 [−.91, −.75]; 29; 10** |
| tab:search, Results | Smoothing r_te [CI]; pieces; tiles | −.52 [−.67, −.33]; 51; 24 | **−.91 [−.95, −.86]; 48; 17** |
| Results, fig:maps | regions of ≥ 20 points; points in them | 8; 343 | **7; 227** |
| fig:maps | test tiles; OODSelect tiles and pieces; Laplacian pieces and tiles | 28; 27, 291; 71, 20 | **27; 25, 293; 105, 22** |
| Controls | shuffled medians, OODSelect / Laplacian; most negative | +0.49 / −0.66; −0.88 | **−0.03 / +0.59; −0.40** |
| Controls | interpolating test models; r without them [CI] | 6; −0.91 [−0.94, −0.85] | **3; −0.92 [−0.95, −0.86]** |
| Controls | never-interpolating families, range | −0.81 to −0.98 | **−0.90 to −0.98** |
| Controls | trees→others; others→trees | −0.76; −0.56 | **−0.74; −0.56** |
| Same inputs | pooled η; baseline; other Zone A | 0.65; 0.30; 0.25 | **0.76; 0.31; 0.29** |
| Same inputs | positive regions: count, points / matches positive | three, 84% / 35% | **two, 98% / 45%** |
| Same inputs | negative regions: count, points / matches positive | five, 19% / 64% | **five, 4% / 85%** |
| tab:regions | every row; Zone AE points lacking a feature | 8 regions; 50 | **7 regions; 75** |
| Appendix A, tab:models | features; training points; degenerate settings; features per split; Models column | seven; 29,161; 8; {1, 2, 7}; 40, 54, 54, 45, 45, 50, 43, 30 | **nine; 36,758; 1; {1, 5, 28}; 22, 47, 45, 38, 34, 32, 44, 28** |
| fig:maps | file | fig_maps_v5.pdf | **fig_maps_v6.pdf** |

The 2026-11-06 column is the paper as it stood before this run (backup:
`Claude outputs/paper-before-2026-11-07-revised/acl_latex.tex`); XX marks a placeholder.

## 7. The regions and the region analysis (Approach 3.6)

### 7.1 The regions changed

The Laplacian S = 500 subset now holds **7 regions** of at least 20 points (227 of the 500 points), against 8 (343) on
2026-11-06 (`logs/2026-11-07/D3_regions_describe.log`). A region of 2026-11-06 is *found again* when at least half of
its points lie within 2 km of this run's region points. Two are: **Liberty** (81% of its points) and **Hendry** (100%).
Hamilton, Manatee/Sarasota, Dixie, Highlands and Levy are gone (nearest region point 67 to 164 km away), and so is the
old negative Hendry/Palm Beach (12.9 km away). Five regions are new: two in Orange County, the positive
Hendry/Palm Beach, Wakulla and Seminole. Region names come from `aotil/region_match.py` (`region_name`, the
counties a region touches), so two different regions are both called "Orange" (flag F9).

| Region | n | Kind | Positives | Zone A polygons (largest, km²) | Ring polygons | Tile | Land cover | Median HAND (m) | Median river (km) | Extent (km) |
|---|---:|---|---:|---|---:|---|---|---:|---:|---:|
| Liberty | 51 | negative | 0.02 | 1 (1.8) | 15 | 21_17 | woody wetlands 75% | 1.7 | 0.6 | 15.4 |
| Orange | 43 | negative | 0.00 | 0 | 8 | 29_14 | woody wetlands 91% | 1.4 | 2.0 | 11.7 |
| Orange | 32 | negative | 0.06 | 2 (2.5) | 14 | 29_14 | emergent herbaceous wetlands 56% | 1.9 | 0.3 | 5.7 |
| Hendry/Palm Beach | 31 | positive | 0.97 | 1 (156.0) | 1 | 30_10 | cultivated crops 84% | 0.7 | 3.8 | 12.2 |
| Wakulla | 28 | positive | 1.00 | 1 (440.1) | 0 | 21_17 | woody wetlands 89% | 1.1 | 2.0 | 8.9 |
| Seminole | 22 | negative | 0.14 | 3 (0.7) | 7 | 29_14 | woody wetlands 59% | 1.1 | 0.8 | 12.2 |
| Hendry | 20 | negative | 0.00 | 0 | 10 | 29_10 | woody wetlands 65% | 0.8 | 16.4 | 6.4 |

Columns: *Zone A polygons* counts the floodplain polygons the region's positives were sampled from (with the largest
one's area); *ring polygons* counts the polygons around which its negatives were sampled (all Zone A). HAND is the
height above the nearest drainage. Both positive regions sit inside one very large Zone A polygon (156 and 440 km²),
so all their positives come from a single approximate-study boundary. The negative regions are mostly band points
(negatives sampled 0.5 to 5 km from Zone A polygons), mostly in wetlands, with 0 to 14% positives.

### 7.2 Every set, with the case-study descriptors

Points / matches for positives, HAND and river distance; the land cover is the points'; Acc15 is the mean accuracy of
the 15 strongest test models on the points / on their matches; the largest SMD is the largest standardized mean
difference between the points and their matches over the six numeric features, with the feature in parentheses
(river and coast: the log distances; wetness: topographic wetness) (`logs/2026-11-07/E_regions_table.log`).

| Set | n | Unmatched | η | Positives | HAND (m) | River (km) | Land cover (top 3) | Acc15 | Largest SMD |
|---|---:|---:|---:|---|---|---|---|---|---|
| Zone AE baseline | 10,911 | 0.050 | 0.307 | 0.53 / 0.50 | 1.10 / 1.00 | 1.48 / 1.52 | woody wetlands 28%, cultivated crops 19%, emergent herbaceous wetlands 12% | 0.78 / 0.79 | 0.03 (slope) |
| All regions (7) | 227 | 0.013 | 0.758 | 0.28 / 0.75 | 1.00 / 0.90 | 1.29 / 1.25 | woody wetlands 62%, emergent herbaceous wetlands 11%, cultivated crops 11% | 0.17 / 0.81 | 0.03 (river) |
| Liberty | 51 | 0.000 | 0.910 | 0.02 / 0.92 | 1.70 / 1.60 | 0.58 / 0.57 | woody wetlands 75%, evergreen forest 22%, developed open space 4% | 0.09 / 0.85 | 0.08 (coast) |
| Orange (43) | 43 | 0.000 | 0.705 | 0.00 / 0.70 | 1.40 / 0.90 | 2.04 / 1.73 | woody wetlands 91%, grassland/herbaceous 5%, mixed forest 2% | 0.35 / 0.72 | 0.12 (coast) |
| Orange (32) | 32 | 0.000 | 0.894 | 0.06 / 0.94 | 1.85 / 2.20 | 0.34 / 0.35 | emergent herbaceous wetlands 56%, woody wetlands 38%, evergreen forest 3% | 0.14 / 0.95 | 0.10 (coast) |
| Hendry/Palm Beach | 31 | 0.032 | 0.697 | 0.97 / 0.28 | 0.70 / 0.90 | 3.77 / 2.91 | cultivated crops 84%, developed open space 10%, developed low intensity 6% | 0.16 / 0.75 | 0.25 (coast) |
| Wakulla | 28 | 0.000 | 0.371 | 1.00 / 0.63 | 1.15 / 0.90 | 2.03 / 1.94 | woody wetlands 89%, evergreen forest 11% | 0.19 / 0.76 | 0.05 (wetness) |
| Seminole | 22 | 0.045 | 0.711 | 0.14 / 0.77 | 1.10 / 0.50 | 0.77 / 0.79 | woody wetlands 59%, emergent herbaceous wetlands 32%, developed open space 5% | 0.12 / 0.77 | 0.16 (slope) |
| Hendry | 20 | 0.050 | 0.958 | 0.00 / 0.96 | 0.75 / 0.40 | 16.45 / 21.09 | woody wetlands 65%, pasture/hay 25%, grassland/herbaceous 5% | 0.05 / 0.92 | 0.13 (coast) |
| Other Zone A points | 16,846 | 0.060 | 0.286 | 0.51 / 0.47 | 1.60 / 1.40 | 1.82 / 1.76 | woody wetlands 29%, evergreen forest 20%, pasture/hay 11% | 0.81 / 0.79 | 0.04 (river) |
| 500 hardest points | 500 | 0.090 | 0.797 | 0.99 / 0.21 | 3.05 / 2.50 | 1.93 / 1.82 | evergreen forest 51%, pasture/hay 14%, developed open space 9% | 0.00 / 0.84 | 0.05 (river) |
| OODSelect S = 500 | 500 | 0.058 | 0.586 | 0.44 / 0.64 | 1.50 / 1.20 | 1.20 / 1.23 | woody wetlands 38%, evergreen forest 18%, emergent herbaceous wetlands 15% | 0.27 / 0.73 | 0.04 (HAND) |
| Halo 1 km | 116 | 0.026 | 0.621 | 0.16 / 0.66 | 1.15 / 0.90 | 1.06 / 1.13 | woody wetlands 47%, evergreen forest 16%, pasture/hay 15% | 0.53 / 0.71 | 0.05 (coast) |
| Halo 2 km | 239 | 0.013 | 0.567 | 0.27 / 0.63 | 1.30 / 1.10 | 1.22 / 1.21 | woody wetlands 52%, evergreen forest 13%, pasture/hay 12% | 0.54 / 0.72 | 0.05 (HAND) |
| Halo 3 km | 356 | 0.008 | 0.537 | 0.31 / 0.62 | 1.30 / 1.00 | 1.22 / 1.22 | woody wetlands 54%, evergreen forest 11%, pasture/hay 10% | 0.57 / 0.71 | 0.05 (coast) |

The *halos* are the Zone A test points within 1, 2 and 3 km of a region but outside the subset: their η falls from
0.62 to 0.54 with distance, between the regions (0.76) and the other Zone A points (0.29). The 500 hardest points are
99% positives that the 15 strongest models all get wrong (Acc15 0.00); their η (0.80) is high too, but they are
scattered (purity 0.145) and 9% have no match.

**By kind** (majority rule; `region_match_kinds.csv`):

| Kind | Regions (points) | Positives, points / matches | Unmatched | η | Strongest-15 errors: missed floods / false floods | corr_te, test models | corr_te, held-out below 99% |
|---|---|---|---:|---:|---|---:|---:|
| Positive | Hendry/Palm Beach, Wakulla (59) | 0.98 / 0.45 | 0.017 | 0.540 | 1.00 / 0.00 (733 errors) | −0.621 (58) | −0.604 (55) |
| Negative | Liberty, Orange, Orange, Seminole, Hendry (168) | 0.04 / 0.85 | 0.012 | 0.834 | 0.03 / 0.97 (2,100 errors) | −0.718 (58) | −0.715 (55) |

In the positive regions the strong models' errors are all missed floods (they predict dry where Zone A says flood);
in the negative regions 97% are false floods (they predict flood where Zone A says dry, as Zone AE would around such
land).

**Sensitivity of the matching** (`region_match_sensitivity.csv`): the regions' η stays 0.745 to 0.759 and their
unmatched share 0 to 0.022 across calipers at the 90th to 99th percentile (ε 1.02 to 2.31) and 1 to 10 matches.

### 7.3 The GTV regions (descriptive)

GTV at S = 500 gives 7 regions holding 383 points (`region_match.csv`, rows `spatial_gtv_mu3_knn10: ...`): Orange 98
(negative, η 0.746), Liberty 85 (negative, 0.845), Hendry/Palm Beach 56 (positive, 0.464), Brevard/Indian River 39
(positive, 0.458), Wakulla 38 (positive, 0.453), Orange/Seminole 35 (negative, 0.520) and Orange 32 (negative,
0.686). All are above the baseline (0.307) and at most 2.6% unmatched. Apart from Brevard/Indian River they lie in the
counties of the Laplacian regions.

### 7.4 The maps figure

`results/2026-11-07/paper_figures/fig_maps_combined.py` (copied from 2026-11-06; only the paths and the
hand-placed region labels `PLACE_LABELS` changed) wrote `fig_maps_combined.pdf`/`.png`, copied to
`CS371_Cho_Li/figures/fig_maps_v6.pdf`; `fig_maps_v5.pdf` is left in place. Panel (c) labels the 7 regions with the
table's names, so it shows "Orange" twice. The region files it reads are byte-identical to those of the final chain
(the checksums of the classic and Laplacian region files match; `region_match.csv` differs only by the 7 GTV rows
appended by the final chain). *2026-11-08:* redrawn as `fig_maps_v7.pdf` by
`results/2026-11-08/paper_figures/fig_maps_combined.py`, which labels the two regions Orange (west) and Orange (east),
as Table tab:regions now does; the paper uses v7, and v6 is left in place.

## 8. Parts F and G: the paper (the data figure, then numbers only)

`CS371_Cho_Li/acl_latex.tex` before this run is backed up in
`Claude outputs/paper-before-2026-11-07-revised/acl_latex.tex`. Only numbers, placeholders, figure files and the two
insertions of Part F changed; no sentence was reworded. Every edit, before and after, is in the Appendix.

### 8.1 What changed

- **Part F, the data figure.** `aotil/fig_feature_boxes.py` (new; adapted from the approved
  `Claude outputs/fig-feature-boxes-2026-11-07/fig_feature_boxes.py`: the sets come from `zoo.split_tiles` and
  `zoo.subset`, terrain variability is `elev_sd_r300`, the water-unit recoding is gone, the layout, colors and 5th/95th
  percentile whiskers are the design's) wrote `CS371_Cho_Li/figures/fig_feature_boxes.pdf` (preview
  `results/2026-11-07/paper_figures/fig_feature_boxes.png`). The figure block replaces `% INSERT FIGURE HERE`, and
  Data statistics now cites `Figure~\ref{fig:data}` instead of `Table~\ref{tab:data}`. The seven most common
  land-cover classes of the v4 data are those of the brief's caption, but 82 crops is now more common than 21
  developed open space, so the caption lists them in that order.
- **Part G, first batch** (`logs/2026-11-07/G1_paper_edits.py`, after Part D): every `\textbf{XX}` (96,467 points;
  107 km; 2.8% and 0.9%; N = 290 and 174 / 58 / 58), the S = 500 share (2.9%), the Graph distance (2.17 km), the model
  counts (Introduction, tab:search caption, Appendix A, tab:models with features per split {1, 5, 28}).
- **Part G, second batch** (`logs/2026-11-07/G2_paper_edits.py`, 37 exact replacements, after the full run): the
  shift, tab:search, the Results paragraphs, the controls, the region analysis (ε, tab:regions), the largest inverted
  budget, and the maps figure (`fig_maps_v6.pdf` with its caption numbers).
- **Check:** `logs/2026-11-07/G3_check_paper_numbers.py` recomputes each number from the dataset, the cell and the
  result files and finds the exact phrase in the paper: **ALL MATCH** (`logs/2026-11-07/G3_check_paper_numbers.log`).

### 8.2 Flagged, not rewritten

Each quotes the paper as it stood on 2026-11-07. *2026-11-08:* draft rewrites for all ten, with Before/After
blocks, are in `docs/DRAFT-REWRITES-2026-11-08.md`. At Victor's go-ahead, its 15 recommended edits were applied to the
paper the same day (listed in `CS371_Cho_Li/CHANGES-2026-11-08.md`); the optional and alternative ones were not.

- **F1, the shift (Introduction; Pooling hides the shift).** "First, we found that the shift exists and models lose
  up to 0.017 accuracy when evaluated on Zone A instead of Zone AE." and the paragraph "Pooling hides the shift",
  which ends the comparison with "the accuracy and AUROC gaps fall to within 0.022 of zero, but the F1 gap stays at
  0.013 to 0.032". *Reason:* the per-tile accuracy gap is +0.017 [−0.042, +0.067]: its interval contains 0 and it is
  smaller than the largest accuracy gap the tile-shuffle null produces by chance (0.022). AUROC (+0.029, null at
  most 0.012) and F1 (+0.068, null 0.013 to 0.032) are still above their nulls, but their intervals also contain 0
  ([−0.041, +0.081], [−0.012, +0.131]). The 15 strongest test models still do worse on Zone A, +0.059 [+0.007,
  +0.107] in accuracy (§6.3). "Lose up to 0.017" also reads as a maximum; it is the mean gap over tiles.
- **F2, the tiles (Split).** "We divide Florida into 50 km tiles, which are wider than the distance over which each
  feature stays spatially correlated (at most 107 km)". *Reason:* the number is filled as asked, but 107 km is more
  than 50 km, so the tiles are not wider than every feature's range (wetness 107 km, §5).
- **F3, "seven features" (Concept shift).** "Labels also disagree within one procedure, since seven features do not
  determine the label." *Reason:* the models now use nine features. The matching uses the six numeric features plus
  exact land cover, which is seven, so the number depends on which set the sentence means.
- **F4, the tab:models caption.** "Models counts the models left after dropping those that predict one class for
  more than 95\% of the ID test points." *Reason:* since D6 the Models column also drops the 78 settings whose
  predictions repeat an earlier setting's (369 − 1 − 78 = 290). The Models paragraph mentions this drop; the caption
  does not.
- **F5, Appendix A.** "Of the 369 settings, 1 predict one class for more than 95\% of the ID test points, which leaves
  $N=290$ models." *Reason:* 369 − 1 is 368; the 78 duplicates are missing from the sentence. With one setting, the verb
  should be "predicts".
- **F6, the hardest-points baseline (OODSelect inverts, but scatters; tab:search).** "much stronger than on the 500
  points the test models get wrong most often ($0.00$)" and the row "Hardest & .00". *Reason:* 579 Zone A test points
  (99% of them positives) are wrong for all 58 test models, so the 500 hardest all come from them. Every test
  model's accuracy on the subset is then 0, and the correlation of a constant is undefined. The code's ε-stabilized
  correlation returns exactly 0 (`aotil/report_paper.py:189-194`), so the comparison is against a degenerate baseline. The same ranking made
  with the selection models and scored on the test models (`hard_sel_r`, not in the paper) gives +0.205. The caption's
  "only indicative" already hedges this.
- **F7, smoothing and coherence (Our loss turns the subset into regions).** "Smoothing OODSelect's final weights over
  each point's neighbors also gives a coherent subset (48 pieces), but it loses the inversion ($-0.91$), so coherence
  must be part of the objective." *Reason:* smoothing now keeps the inversion (−0.913, against −0.918 for the
  Laplacian) with fewer pieces (48 against 105) and tiles (17 against 22) and higher purity at S = 250, 500 and 1,000
  (C4 fails, §6.4). The paragraph's title and the contribution "show that the Laplacian and Graph Total Variation turn
  scattered subsets into regions" are still true against OODSelect (293 pieces against 105 and 29), but not against
  smoothing at this budget.
- **F8, "Only Hamilton" (Same inputs, different labels).** "Only Hamilton, with 11\% unmatched points, may partly
  reflect covariate shift." *Reason:* Hamilton is no longer a region, and no region has more than 5% unmatched points
  (largest: Hendry 5.0%, Seminole 4.5%), so by the paper's rule ("far more than 5%") no region shows covariate shift.
- **F9, the region names (tab:regions; fig:maps panel c).** Two different regions are both "Orange" (43 and 32 points,
  both in test tile 29_14, about 6 km apart in east Orange County), because `region_match.region_name` names a region
  by the counties it touches. "Hendry/Palm Beach" is now a positive region, while on 2026-11-06 a different, negative
  region 12.9 km away had the same name. The names need disambiguating (for example by a nearby river or town). That
  is your call, and the figure's labels follow the table.
- **F10, Case studies.** The paragraph describes Levy, which is no longer a region, and a Liberty of 123 points with 6%
  positives against 72% of its matches. Now the largest positive region is **Hendry/Palm Beach**: 31 points, 97%
  positives against 28% of its matches, 84% cultivated crops, median HAND 0.7 m, 3.8 km from a river, all its positives
  inside one 156 km² Zone A polygon. The largest negative region is still **Liberty**: 51 points, 2% positives against 92% of
  its matches, 75% woody wetlands and 22% evergreen forest, median HAND 1.7 m, 0.6 km from a river; 81% of the
  2026-11-06 Liberty lies within 2 km of it. The closing claim "Both lie on low, wooded land near streams" does not fit
  the new positive region (cropland, 3.8 km from a river). §7.1 and §7.2 have the descriptors of every region.
- **The wording notes of Part A** (W1 to W11, D9 the exact split rule, D10 the sampling rules the text leaves out):
  listed in §2.2. Each is a sentence you may want to add or change; none changes a number.

### 8.3 Still true, but worth a look

- **Controls:** "Some shuffled runs still reach $-0.40$, so a negative correlation alone is weak evidence." True, but
  the most negative shuffled run is now −0.40 (it was −0.88) and no shuffled run reaches −0.6 (§6.7). The sentence's
  point is weaker than before.
- **Search:** "choose $\mu=3$ from a sensitivity analysis" holds by the rule (§6.5), with purity 0.504 against the 0.5
  cut-off.
- **Budget penalty:** "the inversion from the subset search holds up to $S=10{,}000$" holds for both searches; at the next
  grid budget, 15,000 (86% of the Zone A test points), nothing inverts.
- **Same inputs, different labels:** every claim holds: pooled unmatched 1.3% and no region above 10% (largest 5.0%);
  every region above the baseline (lowest Wakulla 0.37 against 0.31); other Zone A lower (0.29); both directions
  (two positive regions, five negative ones).

### 8.4 Compile

A copy in the scratchpad (`pdflatex`, `bibtex`, `pdflatex` twice): **10 pages**. The only overfull box is the existing
one in Table tab:models (15.4 pt, the same as before this run), and the only undefined reference is the `ssurgo`
citation. Floats: Figure 1 (fig:data), Table 1 (tab:search) and Table 2 (tab:regions) on page 6, Figure 2 (fig:maps)
on page 7, Table 3 (tab:models) on page 10. The Part F insertion moved fig:maps from page 6 to page 7.

## 9. Timings

| Step | Wall clock |
|---|---|
| Part B: the full test suite, final run | 3 min 42 s |
| Part C: the points (`logs/2026-11-07/C2_build_points.sh`; `sample_points.py` 1,291 s of it) | 39 min 18 s |
| Part C: soil from SDA (220 requests, at most 2 at a time, 3.6 MB) | 895 s (15 min) |
| Part C: the dataset (Earth Engine, hydrology, merge; `C5_build_dataset.sh`) | 14 min 1 s |
| Part D: fit the cell and audit it (`D_fit_cell.sh`; the 369 fits take 32 s) | 1 min 34 s |
| Part E: the shift and its null (`T_chain.sh`, run alongside the searches) | 8 min 9 s |
| Part E: the 84 searches (`C_chain.sh`, 19:47:30 to 21:34:19; 49,012 CPU s) | 1 h 46 min 49 s |
| · grid (classic and Laplacian, 26 searches) / GTV (13) | 39 min / 25 min |
| · shuffled classic and Laplacian (18) / shuffled GTV (9) | 8 min / 6 min |
| · family-disjoint (12) / μ = 1 (3) / μ = 10 (3) | 12 min / 9 min / 8 min |
| Part E: audits, regions, region analysis, figure (`D_chain.sh`) | 20 s |
| Part G: paper check and compile | under 1 min |

## 10. Docs updated

- This report.
- `docs/DECISION-PROVENANCE.md`: rows for D1 to D7 (support "Your decision", source this brief), written with Part B.
- `docs/TODO-CLAUDE-CODE.md`: item 7 moved to Done (2026-11-07, this report).
- Paper: `CS371_Cho_Li/acl_latex.tex` (numbers, placeholders, figure files, Part F's two insertions),
  `CS371_Cho_Li/figures/fig_feature_boxes.pdf` and `fig_maps_v6.pdf` (new; `fig_maps_v5.pdf` kept).
- Nothing committed or pushed.

## Appendix: every paper edit, before → after

Generated by `logs/2026-11-07/H_paper_edits.py` (`logs/2026-11-07/H_paper_edits.md`): one block per changed source
line, with its location, the changed words and the whole line before and after, ready to paste over the same line in
Overleaf. Rows of one table that do not pair up line by line, and an inserted block (Part F's figure), form one block.

#### Edit 1, near: Introduction

Changed: `361` -> `290`; `0.046` -> `0.017`; `0.81.` -> `0.84.`; `$-0.98$,` -> `$-0.97$,`; `291` -> `293`; `71` -> `105`; `0.077` -> `0.054`; `65\%` -> `76\%`

Before:

```latex
We train 361 tabular models on Zone AE across Florida and evaluate on Zone A. First, we found that the shift exists and models lose up to 0.046 accuracy when evaluated on Zone A instead of Zone AE. Second, accuracy-on-the-line still holds, with a correlation of 0.81. Third, OODSelect finds a subset of Zone A where the correlation inverts to $-0.98$, but it is split into 291 pieces. Fourth, our spatially regularized loss reduces these to 71 pieces at a cost of 0.077 in correlation. Lastly, the regions' inputs are familiar, yet their labels disagree with their matches 65\% of the time.
```

After:

```latex
We train 290 tabular models on Zone AE across Florida and evaluate on Zone A. First, we found that the shift exists and models lose up to 0.017 accuracy when evaluated on Zone A instead of Zone AE. Second, accuracy-on-the-line still holds, with a correlation of 0.84. Third, OODSelect finds a subset of Zone A where the correlation inverts to $-0.97$, but it is split into 293 pieces. Fourth, our spatially regularized loss reduces these to 105 pieces at a cost of 0.054 in correlation. Lastly, the regions' inputs are familiar, yet their labels disagree with their matches 76\% of the time.
```

#### Edit 2, near: Budget penalty.

Changed: `4.0\%` -> `2.9\%`; `$S=7{,}500$.` -> `$S=10{,}000$.`

Before:

```latex
\paragraph{Budget penalty.} We use the squared penalty following \citet{salaudeen2025oodselect}, $\Omega_S(\mathbf{s})=(S-\|\mathbf{s}\|_1)^2$. We report $S=500$, which is 4.0\% of the OOD points and small enough to form regions, although the inversion from the subset search holds up to $S=7{,}500$. 
```

After:

```latex
\paragraph{Budget penalty.} We use the squared penalty following \citet{salaudeen2025oodselect}, $\Omega_S(\mathbf{s})=(S-\|\mathbf{s}\|_1)^2$. We report $S=500$, which is 2.9\% of the OOD points and small enough to form regions, although the inversion from the subset search holds up to $S=10{,}000$. 
```

#### Edit 3, near: Graph.

Changed: `2.54` -> `2.17`

Before:

```latex
\paragraph{Graph.} The graph $\mathbf{A}$ connects two points when either is among the other's $k$ nearest neighbors. We set $k=10$ to keep neighbors local, and the median distance to a point's 10 nearest neighbors is 2.54 km.
```

After:

```latex
\paragraph{Graph.} The graph $\mathbf{A}$ connects two points when either is among the other's $k$ nearest neighbors. We set $k=10$ to keep neighbors local, and the median distance to a point's 10 nearest neighbors is 2.17 km.
```

#### Edit 4, near: Models.

Changed: `$N=\textbf{XX}$` -> `$N=290$`; `\textbf{XX}` -> `174`; `\textbf{XX}` -> `58`; `\textbf{XX}` -> `58`

Before:

```latex
\paragraph{Models.} The $N=\textbf{XX}$ models come from eight families: decision tree, random forest, extremely randomized trees, histogram gradient boosting, XGBoost, logistic regression, multilayer perceptron, and Gaussian naive Bayes \citep{pedregosa2011sklearn,chen2016xgboost}. Each model is one hyperparameter setting. We drop models that predict one class for more than 95\% of the ID test points, since their accuracy on a subset only reflects its share of positives. We also drop models whose predictions repeat an earlier model's. Following \citet{salaudeen2025oodselect}, we split the remaining models at random 60/20/20 into \textbf{XX} selection, \textbf{XX} validation, and \textbf{XX} test models.
```

After:

```latex
\paragraph{Models.} The $N=290$ models come from eight families: decision tree, random forest, extremely randomized trees, histogram gradient boosting, XGBoost, logistic regression, multilayer perceptron, and Gaussian naive Bayes \citep{pedregosa2011sklearn,chen2016xgboost}. Each model is one hyperparameter setting. We drop models that predict one class for more than 95\% of the ID test points, since their accuracy on a subset only reflects its share of positives. We also drop models whose predictions repeat an earlier model's. Following \citet{salaudeen2025oodselect}, we split the remaining models at random 60/20/20 into 174 selection, 58 validation, and 58 test models.
```

#### Edit 5 (replace), near: Sampling.

Before:

```latex
\paragraph{Sampling.} For each zone, we sample positives uniformly by area inside its polygons \citep{woznicki2019,tehrany2017} and negatives from a band 500 m to 5 km from any floodplain around its polygons. The inner limit is a common buffer that keeps negatives off the floodplain's edge \citep{zhang2025,gu2024}. The outer limit keeps them on land that FEMA studied, since far from a mapped floodplain can indicate unstudied areas \citep{wing2017}. Additionally, we force negatives to lie inside a FEMA map panel and outside unstudied areas (Zone D) \citep{samela2017,nrc2007}. Following \citet{samela2017}, we also remove points on open water, points in NLCD class 11. Furthermore, no two points can lie closer than 100 m of each other \citep{mundayatt2026}. In total, we draw 100,000 candidates that meet every requirement above, 25,000 per zone and label. We then drop points in random order until the spacing holds, which leaves \textbf{XX} points.
\paragraph{Split.} Since nearby points look alike, we split by tiles. We divide Florida into 50 km tiles, which are wider than the distance over which each feature stays spatially correlated (at most \textbf{XX} km) \citep{roberts2017, valavi2019blockcv}. We use a 70\%/30\% split as in \citet{gu2024}, dropping any test tiles that do not represent both procedures. There is no need for an evaluation set, as our models are not tuned.
\paragraph{Data statistics.} Table~\ref{tab:data} summarizes each feature of our dataset. Six features are numeric, land cover and fluvial class are categorical, and soil flood frequency is ordinal. We one-hot encode the three categorical and ordinal features. Additionally, some values are missing: \textbf{XX}\% of points lack the soil features, and \textbf{XX}\% lack topographic wetness. Models that cannot handle missing values fill them with the median value from the training set.
% INSERT FIGURE HERE
```

After:

```latex
\paragraph{Sampling.} For each zone, we sample positives uniformly by area inside its polygons \citep{woznicki2019,tehrany2017} and negatives from a band 500 m to 5 km from any floodplain around its polygons. The inner limit is a common buffer that keeps negatives off the floodplain's edge \citep{zhang2025,gu2024}. The outer limit keeps them on land that FEMA studied, since far from a mapped floodplain can indicate unstudied areas \citep{wing2017}. Additionally, we force negatives to lie inside a FEMA map panel and outside unstudied areas (Zone D) \citep{samela2017,nrc2007}. Following \citet{samela2017}, we also remove points on open water, points in NLCD class 11. Furthermore, no two points can lie closer than 100 m of each other \citep{mundayatt2026}. In total, we draw 100,000 candidates that meet every requirement above, 25,000 per zone and label. We then drop points in random order until the spacing holds, which leaves 96,467 points.
\paragraph{Split.} Since nearby points look alike, we split by tiles. We divide Florida into 50 km tiles, which are wider than the distance over which each feature stays spatially correlated (at most 107 km) \citep{roberts2017, valavi2019blockcv}. We use a 70\%/30\% split as in \citet{gu2024}, dropping any test tiles that do not represent both procedures. There is no need for an evaluation set, as our models are not tuned.
\paragraph{Data statistics.} Figure~\ref{fig:data} summarizes each feature of our dataset. Six features are numeric, land cover and fluvial class are categorical, and soil flood frequency is ordinal. We one-hot encode the three categorical and ordinal features. Additionally, some values are missing: 2.8\% of points lack the soil features, and 0.9\% lack topographic wetness. Models that cannot handle missing values fill them with the median value from the training set.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_feature_boxes.pdf}
\caption{Each feature by set (Train and ID test are Zone AE; OOD is Zone A). Top: the box spans the 25th to 75th percentile, the line is the median, and the whiskers reach the 5th and 95th percentiles. Bottom: the share of each set's points in each class. Land cover shows its seven most common classes, 90 woody wetland, 42 evergreen forest, 81 pasture and hay, 95 herbaceous wetland, 82 crops, 21 developed open space, and 22 low-intensity developed, and merges the rest. Soil shares are over points with a soil class.}
\label{fig:data}
\end{figure*}
```

#### Edit 6, near: Matching.

Changed: `$\epsilon=1.39$` -> `$\epsilon=1.30$`

Before:

```latex
\paragraph{Matching.} Let $z(x)\in\mathbb{R}^q$ be the $q$ numeric features of $x$, standardized over $\mathcal{D}_\text{OOD}$. For a region $\mathcal{C}$, the candidates are the $\pi_\text{AE}$ points of $\mathcal{D}_\text{ID}$ in the test tiles that contain no point of $\mathcal{C}$. The matches $\mathcal{C}'_x$ of a point $x\in\mathcal{C}$ are the candidates with its land-cover class within a Euclidean distance $\epsilon$ of $z(x)$, up to the 5 closest. Following \citet{stuart2010}, we set this caliper $\epsilon$ so that 5\% of $\pi_\text{AE}$ points have no match when matched to other $\pi_\text{AE}$ points, which gives $\epsilon=1.39$ on $q=6$ features. Let $\tilde{\mathcal{C}}$ be the points of $\mathcal{C}$ with at least one match.
```

After:

```latex
\paragraph{Matching.} Let $z(x)\in\mathbb{R}^q$ be the $q$ numeric features of $x$, standardized over $\mathcal{D}_\text{OOD}$. For a region $\mathcal{C}$, the candidates are the $\pi_\text{AE}$ points of $\mathcal{D}_\text{ID}$ in the test tiles that contain no point of $\mathcal{C}$. The matches $\mathcal{C}'_x$ of a point $x\in\mathcal{C}$ are the candidates with its land-cover class within a Euclidean distance $\epsilon$ of $z(x)$, up to the 5 closest. Following \citet{stuart2010}, we set this caliper $\epsilon$ so that 5\% of $\pi_\text{AE}$ points have no match when matched to other $\pi_\text{AE}$ points, which gives $\epsilon=1.30$ on $q=6$ features. Let $\tilde{\mathcal{C}}$ be the points of $\mathcal{C}$ with at least one match.
```

#### Edit 7, near: fig:maps

Changed: `\includegraphics[width=\textwidth]{figures/fig_maps_v5.pdf}` -> `\includegraphics[width=\textwidth]{figures/fig_maps_v6.pdf}`

Before:

```latex
\includegraphics[width=\textwidth]{figures/fig_maps_v5.pdf}
```

After:

```latex
\includegraphics[width=\textwidth]{figures/fig_maps_v6.pdf}
```

#### Edit 8, near: fig:maps

Changed: `28` -> `27`; `27` -> `25`; `291` -> `293`; `71` -> `105`; `20` -> `22`; `8` -> `7`

Before:

```latex
\caption{Selected points and regions at $S=500$. Gray dots are Zone A test points, and squares are the 28 test tiles of 50\,km. (a) OODSelect spreads its points over 27 test tiles in 291 pieces. (b) Our search groups them into 71 pieces in 20 tiles. (c) The 8 regions of at least 20 points, colored by whether most of their points are positives or negatives.}
```

After:

```latex
\caption{Selected points and regions at $S=500$. Gray dots are Zone A test points, and squares are the 27 test tiles of 50\,km. (a) OODSelect spreads its points over 25 test tiles in 293 pieces. (b) Our search groups them into 105 pieces in 22 tiles. (c) The 7 regions of at least 20 points, colored by whether most of their points are positives or negatives.}
```

#### Edit 9, near: tab:search

Changed: `$+$.81 [.71, .87]` -> `$+$.84 [.74, .90]`; `28` -> `27`

Before:

```latex
All Zone A & $+$.81 [.71, .87] & -- & 28 \\
```

After:

```latex
All Zone A & $+$.84 [.74, .90] & -- & 27 \\
```

#### Edit 10, near: tab:search

Changed: `$+$.79` -> `$+$.78`

Before:

```latex
Random & $+$.79 & -- & -- \\
```

After:

```latex
Random & $+$.78 & -- & -- \\
```

#### Edit 11, near: tab:search

Changed: `$-$.27` -> `.00`

Before:

```latex
Hardest & $-$.27 & -- & -- \\
```

After:

```latex
Hardest & .00 & -- & -- \\
```

#### Edit 12, near: tab:search

Changed: `$-$.98 [$-$.99, $-$.96]` -> `$-$.97 [$-$.98, $-$.95]`; `291` -> `293`; `27` -> `25`

Before:

```latex
OODSelect & $-$.98 [$-$.99, $-$.96] & 291 & 27 \\
```

After:

```latex
OODSelect & $-$.97 [$-$.98, $-$.95] & 293 & 25 \\
```

#### Edit 13, near: tab:search

Changed: `$-$.90 [$-$.94, $-$.84]` -> `$-$.92 [$-$.95, $-$.86]`; `71` -> `105`; `20` -> `22`

Before:

```latex
Laplacian (ours) & $-$.90 [$-$.94, $-$.84] & 71 & 20 \\
```

After:

```latex
Laplacian (ours) & $-$.92 [$-$.95, $-$.86] & 105 & 22 \\
```

#### Edit 14, near: tab:search

Changed: `$-$.86` -> `$-$.85`; `$-$.78]` -> `$-$.75]`; `15` -> `29`; `11` -> `10`

Before:

```latex
GTV (ours) & $-$.86 [$-$.91, $-$.78] & 15 & 11 \\
```

After:

```latex
GTV (ours) & $-$.85 [$-$.91, $-$.75] & 29 & 10 \\
```

#### Edit 15, near: tab:search

Changed: `$-$.52 [$-$.67, $-$.33]` -> `$-$.91 [$-$.95, $-$.86]`; `51` -> `48`; `24` -> `17`

Before:

```latex
Smoothing & $-$.52 [$-$.67, $-$.33] & 51 & 24 \\
```

After:

```latex
Smoothing & $-$.91 [$-$.95, $-$.86] & 48 & 17 \\
```

#### Edit 16, near: tab:search

Changed: `73` -> `58`

Before:

```latex
\caption{Subset search at $S=500$. $\text{corr}_\text{te}$ uses the 73 test models, and a more negative value is a stronger inversion. Pieces are the connected components of the subset in $\mathbf{A}$, and tiles are the test tiles they touch. Random is the mean of 100 draws. The hardest points are those the test models get wrong most often, so their $\text{corr}_\text{te}$ is only indicative. Smoothing averages OODSelect's final weights over each point's 10 nearest neighbors.}
```

After:

```latex
\caption{Subset search at $S=500$. $\text{corr}_\text{te}$ uses the 58 test models, and a more negative value is a stronger inversion. Pieces are the connected components of the subset in $\mathbf{A}$, and tiles are the test tiles they touch. Random is the mean of 100 draws. The hardest points are those the test models get wrong most often, so their $\text{corr}_\text{te}$ is only indicative. Smoothing averages OODSelect's final weights over each point's 10 nearest neighbors.}
```

#### Edit 17 (table rows), near: tab:regions

Changed: the rows Dixie, Highlands, Levy, Liberty, Hamilton, Manatee/Sarasota, Hendry/Palm Beach, Hendry, All regions, Other Zone A, Zone AE baseline -> Hendry/Palm Beach, Wakulla, Liberty, Orange, Orange, Seminole, Hendry, All regions, Other Zone A, Zone AE baseline

Before:

```latex
Dixie & 31 & .55 / .30 & .00 & .46 \\
Highlands & 31 & 1.00 / .59 & .00 & .41 \\
Levy & 28 & 1.00 / .13 & .00 & .87 \\
\multicolumn{5}{@{}l}{\textit{Negative regions}} \\
Liberty & 107 & .06 / .73 & .00 & .71 \\
Hamilton & 61 & .43 / .63 & .11 & .64 \\
Manatee/Sarasota & 38 & .34 / .34 & .03 & .61 \\
Hendry/Palm Beach & 27 & .07 / .62 & .04 & .60 \\
Hendry & 20 & .10 / .80 & .05 & .83 \\
\midrule
All regions & 343 & .36 / .56 & .03 & .65 \\
Other Zone A & 12,031 & .52 / .51 & .05 & .25 \\
Zone AE baseline & 10,785 & .55 / .53 & .05 & .30 \\
```

After:

```latex
Hendry/Palm Beach & 31 & .97 / .28 & .03 & .70 \\
Wakulla & 28 & 1.00 / .63 & .00 & .37 \\
\multicolumn{5}{@{}l}{\textit{Negative regions}} \\
Liberty & 51 & .02 / .92 & .00 & .91 \\
Orange & 43 & .00 / .70 & .00 & .70 \\
Orange & 32 & .06 / .94 & .00 & .89 \\
Seminole & 22 & .14 / .77 & .05 & .71 \\
Hendry & 20 & .00 / .96 & .05 & .96 \\
\midrule
All regions & 227 & .28 / .75 & .01 & .76 \\
Other Zone A & 16,846 & .51 / .47 & .06 & .29 \\
Zone AE baseline & 10,911 & .53 / .50 & .05 & .31 \\
```

#### Edit 18, near: tab:regions

Changed: `50` -> `75`

Before:

```latex
\caption{Matched inputs. Positives is the share of positives among the points and among their matches. Unmatched is the share of points without a match, which is .05 for Zone AE points by construction. $\eta$ is the label disagreement (Equation~\ref{eq:disagree}). Other Zone A points are the Zone A test points outside the subset. The 50 Zone AE test points that lack a feature are not matched.}
```

After:

```latex
\caption{Matched inputs. Positives is the share of positives among the points and among their matches. Unmatched is the share of points without a match, which is .05 for Zone AE points by construction. $\eta$ is the label disagreement (Equation~\ref{eq:disagree}). Other Zone A points are the Zone A test points outside the subset. The 75 Zone AE test points that lack a feature are not matched.}
```

#### Edit 19, near: Pooling hides the shift.

Changed: `$-0.008$ [$-0.065$, $0.048$].` -> `$-0.034$ [$-0.111$, $0.037$].`; `1,264` -> `1,502`; `192` -> `193`; `0.51` -> `0.49`; `22` -> `19`; `28` -> `27`; `0.046 [0.021, 0.074]` -> `0.017 [$-0.042$, 0.067]`; `0.043` -> `0.029`; `0.093` -> `0.068`; `94\%` -> `75\%`; `0.023` -> `0.022`; `0.035` -> `0.013`; `0.041.` -> `0.032.`; `$\text{corr}_\text{te}=+0.81$.` -> `$\text{corr}_\text{te}=+0.84$.`

Before:

```latex
\paragraph{Pooling hides the shift.} Pooled over the test tiles, models trained on Zone AE are as accurate on Zone A as on held-out Zone AE, with a gap of $-0.008$ [$-0.065$, $0.048$]. However, the two test sets weight the tiles differently. For example, the tile around Palm Beach holds 1,264 Zone AE points but only 192 Zone A points, and the models average 0.51 accuracy there. Within a tile, models are less accurate on Zone A in 22 of the 28 tiles. Averaged over tiles, the gap is 0.046 [0.021, 0.074] in accuracy, 0.043 in AUROC, and 0.093 in F1, and 94\% of models do worse on Zone A. When we shuffle which test points belong to each zone within each tile and label, the accuracy and AUROC gaps fall to within 0.023 of zero, but the F1 gap stays at 0.035 to 0.041. Accuracy-on-the-line still holds on all of Zone A, with $\text{corr}_\text{te}=+0.81$.
```

After:

```latex
\paragraph{Pooling hides the shift.} Pooled over the test tiles, models trained on Zone AE are as accurate on Zone A as on held-out Zone AE, with a gap of $-0.034$ [$-0.111$, $0.037$]. However, the two test sets weight the tiles differently. For example, the tile around Palm Beach holds 1,502 Zone AE points but only 193 Zone A points, and the models average 0.49 accuracy there. Within a tile, models are less accurate on Zone A in 19 of the 27 tiles. Averaged over tiles, the gap is 0.017 [$-0.042$, 0.067] in accuracy, 0.029 in AUROC, and 0.068 in F1, and 75\% of models do worse on Zone A. When we shuffle which test points belong to each zone within each tile and label, the accuracy and AUROC gaps fall to within 0.022 of zero, but the F1 gap stays at 0.013 to 0.032. Accuracy-on-the-line still holds on all of Zone A, with $\text{corr}_\text{te}=+0.84$.
```

#### Edit 20, near: OODSelect inverts, but scatters.

Changed: `($+0.79$),` -> `($+0.78$),`; `$-0.98$` -> `$-0.97$`; `($-0.27$).` -> `($0.00$).`; `291` -> `293`; `27` -> `25`; `28` -> `27`

Before:

```latex
\paragraph{OODSelect inverts, but scatters.} Random subsets of 500 points follow the aggregate trend ($+0.79$), so small subsets do not invert by chance (Table~\ref{tab:search}). OODSelect finds an inversion of $-0.98$ on test models it never saw, much stronger than on the 500 points the test models get wrong most often ($-0.27$). However, its subset is split into 291 pieces across 27 of the 28 test tiles (Figure~\ref{fig:maps}a), so it cannot be studied as a set of places.
```

After:

```latex
\paragraph{OODSelect inverts, but scatters.} Random subsets of 500 points follow the aggregate trend ($+0.78$), so small subsets do not invert by chance (Table~\ref{tab:search}). OODSelect finds an inversion of $-0.97$ on test models it never saw, much stronger than on the 500 points the test models get wrong most often ($0.00$). However, its subset is split into 293 pieces across 25 of the 27 test tiles (Figure~\ref{fig:maps}a), so it cannot be studied as a set of places.
```

#### Edit 21, near: Our loss turns the subset into regions.

Changed: `$-0.90$ [$-0.94$, $-0.84$]` -> `$-0.92$ [$-0.95$, $-0.86$]`; `71` -> `105`; `20` -> `22`; `Eight` -> `Seven`; `343` -> `227`; `0.077` -> `0.054`; `(51` -> `(48`; `($-0.15$),` -> `($-0.91$),`; `($-0.86$)` -> `($-0.85$)`; `15` -> `29`

Before:

```latex
\paragraph{Our loss turns the subset into regions.} With the Laplacian loss, the search keeps an inversion of $-0.90$ [$-0.94$, $-0.84$] in 71 pieces over 20 tiles (Figure~\ref{fig:maps}b). Eight pieces of at least 20 points hold 343 of the 500 points, and these are the regions we analyze (Figure~\ref{fig:maps}c). The cost of coherence is 0.077 in correlation. Smoothing OODSelect's final weights over each point's neighbors also gives a coherent subset (51 pieces), but it loses the inversion ($-0.15$), so coherence must be part of the objective. GTV inverts less ($-0.86$) and leaves 15 pieces at this budget.
```

After:

```latex
\paragraph{Our loss turns the subset into regions.} With the Laplacian loss, the search keeps an inversion of $-0.92$ [$-0.95$, $-0.86$] in 105 pieces over 22 tiles (Figure~\ref{fig:maps}b). Seven pieces of at least 20 points hold 227 of the 500 points, and these are the regions we analyze (Figure~\ref{fig:maps}c). The cost of coherence is 0.054 in correlation. Smoothing OODSelect's final weights over each point's neighbors also gives a coherent subset (48 pieces), but it loses the inversion ($-0.91$), so coherence must be part of the objective. GTV inverts less ($-0.85$) and leaves 29 pieces at this budget.
```

#### Edit 22, near: Controls.

Changed: `$+0.49$` -> `$-0.03$`; `$-0.66$` -> `$+0.59$`; `$-0.88$,` -> `$-0.40$,`; `6` -> `3`; `$-0.91$ [$-0.94$, $-0.85$],` -> `$-0.92$ [$-0.95$, $-0.86$],`; `$-0.81$` -> `$-0.90$`; `$-0.76$,` -> `$-0.74$,`

Before:

```latex
\paragraph{Controls.} Three controls support the inversion. First, we rerun each search three times at budgets of 250, 500, and 1,000 points with the ID accuracies of the selection and validation models shuffled. The median correlation of these nine runs is $+0.49$ for OODSelect and $-0.66$ for the Laplacian, and the real Laplacian search beats 9 of its 9. Some shuffled runs still reach $-0.88$, so a negative correlation alone is weak evidence. Second, noisy labels invert the trend only when models interpolate them \citep{sanyal2025}. Without the 6 test models that reach 99\% training accuracy, $\text{corr}_\text{te}$ is $-0.91$ [$-0.94$, $-0.85$], and the trend also inverts within each of the three families that never interpolate, from $-0.81$ to $-0.98$. Third, selecting with the tree-based families and testing on the others gives $-0.76$, and the reverse gives $-0.56$.
```

After:

```latex
\paragraph{Controls.} Three controls support the inversion. First, we rerun each search three times at budgets of 250, 500, and 1,000 points with the ID accuracies of the selection and validation models shuffled. The median correlation of these nine runs is $-0.03$ for OODSelect and $+0.59$ for the Laplacian, and the real Laplacian search beats 9 of its 9. Some shuffled runs still reach $-0.40$, so a negative correlation alone is weak evidence. Second, noisy labels invert the trend only when models interpolate them \citep{sanyal2025}. Without the 3 test models that reach 99\% training accuracy, $\text{corr}_\text{te}$ is $-0.92$ [$-0.95$, $-0.86$], and the trend also inverts within each of the three families that never interpolate, from $-0.90$ to $-0.98$. Third, selecting with the tree-based families and testing on the others gives $-0.74$, and the reverse gives $-0.56$.
```

#### Edit 23, near: Same inputs, different labels.

Changed: `$\eta=0.65$,` -> `$\eta=0.76$,`; `0.30` -> `0.31`; `($\eta=0.25$),` -> `($\eta=0.29$),`; `three` -> `two`; `84\%` -> `98\%`; `35\%` -> `45\%`; `19\%` -> `4\%`; `64\%` -> `85\%`

Before:

```latex
\paragraph{Same inputs, different labels.} The regions are not unfamiliar land (Table~\ref{tab:regions}). Pooled, 1.3\% of their points have no match, against 5\% of Zone AE points, and no region exceeds 10\%. Their labels, however, differ. The pooled label disagreement is $\eta=0.65$, against 0.30 among Zone AE points, and every region lies above this baseline. Other Zone A points disagree less ($\eta=0.25$), so the disagreement is local to the regions. It runs in both directions. In the three positive regions, 84\% of the points are positives against 35\% of their matches, so Zone A marks floodplain where Zone AE would not. In the five negative regions, 19\% are positives against 64\% of their matches, so Zone A leaves out land that Zone AE would map. Only Hamilton, with 11\% unmatched points, may partly reflect covariate shift.
```

After:

```latex
\paragraph{Same inputs, different labels.} The regions are not unfamiliar land (Table~\ref{tab:regions}). Pooled, 1.3\% of their points have no match, against 5\% of Zone AE points, and no region exceeds 10\%. Their labels, however, differ. The pooled label disagreement is $\eta=0.76$, against 0.31 among Zone AE points, and every region lies above this baseline. Other Zone A points disagree less ($\eta=0.29$), so the disagreement is local to the regions. It runs in both directions. In the two positive regions, 98\% of the points are positives against 45\% of their matches, so Zone A marks floodplain where Zone AE would not. In the five negative regions, 4\% are positives against 85\% of their matches, so Zone A leaves out land that Zone AE would map. Only Hamilton, with 11\% unmatched points, may partly reflect covariate shift.
```

#### Edit 24, near: Model Settings

Changed: `seven` -> `nine`; `29,161` -> `36,758`; `8` -> `1`; `$N=361$` -> `$N=290$`

Before:

```latex
Table~\ref{tab:models} lists the hyperparameter grids. Each combination in a family's grid is one model with a fixed seed, and every model trains on all seven features and all 29,161 training points. The grids are our own. They run from deliberately weak settings, such as a few training iterations, one feature per split, or heavy variance smoothing, to typical ones. Random forests and extra trees still draw a random subset of features at each split, and random forests also resample the rows of each tree, as their algorithms do. Logistic regression and the multilayer perceptrons standardize their inputs. Settings not listed take the defaults of scikit-learn 1.9.1 and XGBoost 3.4.1. Of the 369 settings, 8 predict one class for more than 95\% of the ID test points, which leaves $N=361$ models.
```

After:

```latex
Table~\ref{tab:models} lists the hyperparameter grids. Each combination in a family's grid is one model with a fixed seed, and every model trains on all nine features and all 36,758 training points. The grids are our own. They run from deliberately weak settings, such as a few training iterations, one feature per split, or heavy variance smoothing, to typical ones. Random forests and extra trees still draw a random subset of features at each split, and random forests also resample the rows of each tree, as their algorithms do. Logistic regression and the multilayer perceptrons standardize their inputs. Settings not listed take the defaults of scikit-learn 1.9.1 and XGBoost 3.4.1. Of the 369 settings, 1 predict one class for more than 95\% of the ID test points, which leaves $N=290$ models.
```

#### Edit 25, near: tab:models

Changed: `40` -> `22`

Before:

```latex
Decision tree & maximum depth $\in\{1,2,3,4,5,6,8,10,12,\text{none}\}$ $\times$ minimum leaf size $\in\{1,20,200,2000\}$ & 40 & 40 \\
```

After:

```latex
Decision tree & maximum depth $\in\{1,2,3,4,5,6,8,10,12,\text{none}\}$ $\times$ minimum leaf size $\in\{1,20,200,2000\}$ & 40 & 22 \\
```

#### Edit 26, near: tab:models

Changed: `$\in\{1,2,7\}$` -> `$\in\{1,5,28\}$`; `54` -> `47`

Before:

```latex
Random forest & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,2,7\}$ & 54 & 54 \\
```

After:

```latex
Random forest & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,5,28\}$ & 54 & 47 \\
```

#### Edit 27, near: tab:models

Changed: `$\in\{1,2,7\}$` -> `$\in\{1,5,28\}$`; `54` -> `45`

Before:

```latex
Extra trees & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,2,7\}$ & 54 & 54 \\
```

After:

```latex
Extra trees & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,5,28\}$ & 54 & 45 \\
```

#### Edit 28, near: tab:models

Changed: `45` -> `38`

Before:

```latex
Histogram gradient boosting & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ iterations $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,\text{none}\}$, without early stopping & 48 & 45 \\
```

After:

```latex
Histogram gradient boosting & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ iterations $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,\text{none}\}$, without early stopping & 48 & 38 \\
```

#### Edit 29, near: tab:models

Changed: `45` -> `34`

Before:

```latex
XGBoost & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ trees $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,8\}$, histogram tree method & 48 & 45 \\
```

After:

```latex
XGBoost & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ trees $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,8\}$, histogram tree method & 48 & 34 \\
```

#### Edit 30, near: tab:models

Changed: `50` -> `32`

Before:

```latex
Logistic regression & inverse regularization strength $C\in\{0.001,0.01,0.1,1,100\}$ $\times$ iterations $\in\{1,2,3,5,100\}$ $\times$ class weights $\in\{\text{none},\text{balanced}\}$ & 50 & 50 \\
```

After:

```latex
Logistic regression & inverse regularization strength $C\in\{0.001,0.01,0.1,1,100\}$ $\times$ iterations $\in\{1,2,3,5,100\}$ $\times$ class weights $\in\{\text{none},\text{balanced}\}$ & 50 & 32 \\
```

#### Edit 31, near: tab:models

Changed: `43` -> `44`

Before:

```latex
Multilayer perceptron & hidden layers $\in\{(2),(8),(16),(32),(64,32)\}$ $\times$ epochs $\in\{3,20,200\}$ $\times$ learning rate $\in\{10^{-4},10^{-3},10^{-2}\}$ & 45 & 43 \\
```

After:

```latex
Multilayer perceptron & hidden layers $\in\{(2),(8),(16),(32),(64,32)\}$ $\times$ epochs $\in\{3,20,200\}$ $\times$ learning rate $\in\{10^{-4},10^{-3},10^{-2}\}$ & 45 & 44 \\
```

#### Edit 32, near: tab:models

Changed: `30` -> `28`

Before:

```latex
Gaussian naive Bayes & variance smoothing $\in\{10^{-9},10^{-5},10^{-4},10^{-3},0.003,0.01,0.03,\allowbreak 0.1,0.2,0.3,0.5,1,2,3,10\}$ $\times$ class priors $\in\{\text{fitted},\text{equal}\}$ & 30 & 30 \\
```

After:

```latex
Gaussian naive Bayes & variance smoothing $\in\{10^{-9},10^{-5},10^{-4},10^{-3},0.003,0.01,0.03,\allowbreak 0.1,0.2,0.3,0.5,1,2,3,10\}$ $\times$ class priors $\in\{\text{fitted},\text{equal}\}$ & 30 & 28 \\
```

#### Edit 33, near: tab:models

Changed: `361` -> `290`

Before:

```latex
Total & & 369 & 361 \\
```

After:

```latex
Total & & 369 & 290 \\
```

