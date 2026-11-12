# The grid model population and a full rerun on it, D = 2026-11-06

Victor's request of 2026-11-06: build the model population from fixed hyperparameter grids (models differ
only by family and setting; all features and all training points; degenerate models dropped, duplicates kept),
do a full run of the subset search on the new models, rerun every result in the paper on it, and update the
paper's numbers. Brief: `docs/TASK-ZOO-GRIDS-2026-11-06.md` (prompt: `docs/PROMPT-ZOO-GRIDS-2026-11-06.md`).
D = 2026-11-06. Outputs are in `results/2026-11-06/`, logs in `logs/2026-11-06/`, and the cell in
`data/interim/aotil/cells_grid_2026-11-06/`. Nothing is committed or pushed. The Mac was on AC power for the
whole run.

**In one paragraph.** The model population is now one model per setting of fixed hyperparameter grids: 369
settings, 8 dropped as degenerate (they predict one class for more than 95% of the ID test points), **361 kept**,
split 216 / 72 / 73. Part B passed with no Stop (audit PASS; ID accuracy 0.510 to 0.804; probit r(ID, OOD) +0.849
over all kept models). All 81 searches of the 2026-10-25 plan ran on the new cell with the same settings and seeds
(classic, Laplacian and GTV over the whole budget grid, the shuffled and family-disjoint controls, the μ sweep), then
everything downstream. **No Stop fired:** C6a and C6b hold (pooled unmatched 0.029; η 0.649 against the Zone AE
baseline 0.297), and the μ rule still picks μ = 3. The inversion holds: OODSelect −0.98 in 291 pieces, our
Laplacian loss −0.90 in 71 pieces at a cost of 0.077, GTV −0.86 in 15 pieces, all three inverting up to S = 7,500.
**Three results now contradict the paper's text:**
1. *The regions changed*: 8 instead of 6. Liberty, Hamilton and Dixie are found again; Levy shrank to 28 points;
   Palm Beach and Martin are gone; Manatee/Sarasota, Highlands, Hendry/Palm Beach and Hendry are new.
2. *Smoothing OODSelect's weights now keeps an inversion* (−0.52 at S = 500), with higher purity than our search.
   So C4 fails (also on pieces: 71 > 291/5).
3. *Hamilton has 11.5% unmatched points*, so "no region exceeds 10%" no longer holds.

In the paper I changed only numbers, the 17 placeholders and the map figure, and I flag four places (§5.2) for you.

**Update 2026-11-07.** At your go-ahead, Table tab:regions now lists the 8 regions of this run (F4 applied, §5.2;
paste block: Appendix, Edit 13). Three flagged places remain (F1 to F3). The paper still compiles to 10 pages with
every float on its page; the taller table moves the heading 4.4 Results from page 5 to page 6 (§5.4).

| Claim | Result (2026-11-06) | Holds? |
|---|---|---|
| C1 shift | pooled accuracy gap −0.008 [−0.065, +0.048]; per tile (unweighted) +0.046 [+0.021, +0.074], 22 of 28 tiles, 94% of models | reported |
| C2 accuracy-on-the-line | r_te = +0.806 on all Zone A points | **holds** |
| C3 OODSelect inverts, scattered | −0.977; 291 pieces, 27 tiles, purity 0.126; below all 3 shuffled runs at S = 500 | **holds** |
| C4 regions at small cost | −0.900, purity 0.671, 71 pieces (needs ≤ 58); purity below smoothing at S = 250, 500 and 1,000 | **does not hold** |
| C5 not memorized noise | −0.905 [−0.941, −0.850] on the 67 held-out test models below 99% training accuracy | **holds** |
| C6a familiar inputs | pooled unmatched 0.029 (≤ 0.10); Hamilton 0.115 is above 0.10 | **holds** |
| C6b different labels | pooled η 0.649 against the baseline 0.297 (needs ≥ 0.397) | **holds** |
| C7 GTV (descriptive) | −0.858 [−0.909, −0.782]; purity 0.916; 15 pieces; 11 tiles; 9 regions, 440 points | reported |

**Read first:** §5.2 (the flagged places in the paper; F4 is applied) and §4.1 (the regions).

---

## Terms used here, in CS terms

- **Cell, population.** One cell is one train/test split (seed 0) and the models fitted on it. The *population*
  is the set of models. Each model's per-point outputs are stored as a models × points matrix.
- **Grid, setting.** Each model family (decision tree, random forest, extra trees, histogram gradient boosting,
  XGBoost, logistic regression, multilayer perceptron, Gaussian naive Bayes) has a fixed hyperparameter grid; a
  *setting* is one point of that grid (one combination of the axes). One model per setting: 369 settings.
- **Degenerate model.** A model that predicts flood for fewer than 5% or more than 95% of the held-out Zone AE
  test points. Its accuracy on a subset then mostly measures the subset's share of positives (the base rate), so
  it is dropped before the 60/20/20 model split.
- **Duplicate model.** A kept model whose 0/1 predictions on every test point equal another model's. Kept
  (Victor's decision).
- **Tile.** The map is cut into 50 km squares (`tile_50km`). Whole tiles go to training or testing, like grouped
  cross-validation where a group is a square of land. The 28 test tiles hold both test sets.
- **Zone A, Zone AE.** Two FEMA labeling procedures for the same label (the 1% annual chance floodplain). Zone AE
  is the detailed one; the models train on it. Zone A is the approximate one, the out-of-distribution (OOD) test
  set. ID = held-out Zone AE points in the same test tiles.
- **kNN graph, piece, region, purity.** Each Zone A test point is linked to its 10 nearest Zone A test points
  (straight-line distance in metres on EPSG:3086, a flat map projection of Florida). A *piece* is a connected
  component of the selected points in that graph; a *region* is a piece of at least c = 20 points. *Purity* is
  the share of a selected point's 10 nearest neighbours that are also selected, averaged over the selected
  points (1 for one solid blob; S/m for random points).
- **Land cover (NLCD), HAND.** NLCD is the National Land Cover Database class of the 30 m pixel under a point
  (90 = woody wetlands, 42 = evergreen forest, ...), a categorical feature. HAND is the height above the nearest
  drainage: how many metres a point sits above the nearest stream cell.
- **Caliper matching, η.** Each region point looks for held-out Zone AE points with the same land-cover class
  whose 6 standardized numeric features lie within a Euclidean distance ε (the caliper), keeping at most the 5
  closest, never from a test tile that holds a point of the region. *Unmatched* = nothing inside ε (covariate
  shift: unusual inputs). η is the share of a matched point's matches whose label differs from its own, averaged
  (concept shift: the same inputs labeled differently).

---

## 1. Part A: the code

Backups of every changed file and of all private characterization goldens, from before the change:
`Claude outputs/goldens-before-2026-11-06/` (`code/`, `golden/`).

| File | Change |
|---|---|
| `aotil/zoo.py` | `GRIDS` holds the brief's §2 grids (the paper's Table tab:models). `grid_settings()` expands them in the brief's order (families in sorted code-name order; `itertools.product`, first axis slowest), and `zoo_specs(seed)` returns one spec per setting: `grid_idx`, `kind`, `kw`, `seed`, the seeds drawn in that order from `default_rng(cell seed)`. Deleted: `learner_settings`, the round robin (`i mod 8`), `feat_frac`, `row_frac`, and the feature and row draws of `fit_one`. `fit_one` now trains every model on all training rows (plus any `--extra-train` rows) and every column with two distinct finite values (the `varying_columns` guard stays), and counts the ID test points it predicts flood. New: `predicts_flood` (p > 0.5, a tie dry, read from the stored true-class scores), `is_degenerate` (the drop rule, in exact fractions so 5% and 95% exactly are kept), `split_population` (drops degenerate and unusable settings, keeps duplicates), `dropped_row`, `family_counts`, `hp_columns`, `hp_last`. `build`: lr no longer fixes `max_iter=400`; nb now takes its grid setting (`GaussianNB(**kw)`). `model_row` writes `idx` (row in the score matrices, 0..N−1), `grid_idx` and `flood_share_id`. `run_cell` has no model count; `main` writes `<tag>_dropped.csv`, and `--n-models` is gone (argparse refuses it). |
| `aotil/audit_cell.py` | The model checks follow the grid: the 8 families present (no balance check), `idx` contiguous, kept + dropped = the grid exactly once in grid order, every kept and dropped row equal to the setting at its `grid_idx` (kind, seed, every hp), the drop rule (each kept model's flood share recounted from S_id passes it and equals `flood_share_id`; each degenerate row fails it), every model on all varying features and all training rows. Refits use `specs[grid_idx]`, and every dropped setting is refitted too and must reproduce its stored flood share. A table without `grid_idx` (any cell built before today) fails and is not refitted. Removed: `--expect-models` and the v1-table AUC branch (retired with the old population). |
| `aotil/o1_shift.py` | The `MODEL_STRIDE` comment and `--model-stride` help no longer describe the round robin (families are now contiguous blocks in grid order). No computation changed. |
| `aotil/o5_interpolation.py` | The printout counted the test models as a hard-coded 60; it now prints the real count. The model-set labels `all300` and `test60` are kept (so every run's CSV reads the same) and documented as "all models" and "the test models". |
| `aotil/o5_feature_inclusion.py` | New `require_varying_inclusion`: refuses a cell where every model saw every feature (any grid cell), since the analysis needs random feature subsets. Not in the paper. |
| `aotil/o5_pooling.py` | `same_specs` compares lengths first and also `grid_idx`, so two populations that dropped different settings are refused instead of raising. Not in the paper. |
| `aotil/region_match.py`, `aotil/o5_shift_decomp.py` | Docstrings only: "the 60 test models" and "the 300 models" became "the test models" and "the cell's models". |
| `run_oodselect_local.sh` | `N_MODELS` and the zoo's `--n-models` are gone; the comments no longer say 300 models. |
| `aotil/test_zoo.py` (new) | The owner checks of the brief's Part A step 4 (below). |
| `aotil/test_o5_feature_inclusion.py`, `aotil/test_o5_pooling.py` | One new check each, for the two guards above. |
| `results/2026-11-06/paper_figures/` | Copies of `common.py`, `fig_aotl.py` and `fig_maps_combined.py` of 2026-10-25, pointed at this run (`RUN`, `CELLS_DIR`); the map's place labels follow the new regions. |

Readers checked and left unchanged because they already work for any population size: `oodselect_paper.py`
(`model_split`, `family_split`, `make_split`), `report_paper.py`, `audit_selection.py`, `o1_shift.py`'s
`strongest_models`, `region_match.py`, `regions.py`, `cells.py`.

**Tests.**
- `aotil/test_zoo.py`: **ALL PASSED** (`logs/2026-11-06/A_test_zoo.log`). It checks every family's grid size and
  order against the brief's values written out again, 369 settings, every (family, setting) unique, the seeds in
  grid order, xgb last (so a laptop without xgboost keeps the other seeds), `build` (lr's `max_iter` from the
  grid, nb's setting); the drop rule on synthetic predictions with ties exactly at 5% (1 of 20, 2 of 40) and 95%
  (19 of 20, 38 of 40) kept and one point beyond dropped, and on the real ID size (541 and 542 of 10,835);
  `predicts_flood` (a tie dry at either label); identical predictions kept; `fit_one` on all rows and varying
  columns; `run_cell` and `main` end to end on a synthetic dataset (idx 0..N−1, grid order, kept + dropped = 369,
  serial = parallel, `--n-models` refused); `model_split` over 300, 311, 360 and 369 models (360 gives 216/72/72);
  the family split; the audit helpers; and `audit_cell.py` on the synthetic cell (PASS) and on five damaged copies
  (a kept degenerate model, no dropped table, a setting neither kept nor dropped, a changed seed, a pre-grid
  table: each FAILs on the right check). Check 10 runs on the real cell.
- `aotil/test_o5_feature_inclusion.py` and `aotil/test_o5_pooling.py`: the new checks pass; their other real-data
  checks fail as they did before today (they read the 2026-10-16/17 outputs made on the v2 data, the known data
  pairing of 2026-10-23).
- **Private characterization goldens.** The zoo goldens were regenerated (11 files changed, listed with reasons in
  §7). Outside the zoo, 4 golden files changed, each only as `Claude outputs/goldens-before-2026-11-06/compare_goldens.py`
  allows (`logs/2026-11-06/A_compare_goldens.log`; every other golden file is byte-identical):
  the runner's recorded zoo calls lost `--n-models` (and the settings line its number), the interpolation
  printout's hard-coded "60" became the real test count, and two owner check scripts report more passing checks
  (13 → 19 and 22 → 24, the new checks above). The private test of the zoo's tie rule now runs on the grid cell
  and passes; before today it failed on the retired cells.
- **The whole private suite**: see §7.

---

## 2. Part B: the new cell (no Stop)

`aotil/zoo.py --state FL --protocol band --train-dom AE --seed 0 --data data/processed_w7 --out
data/interim/aotil/cells_grid_2026-11-06 --n-jobs -1` with one BLAS thread per worker: all 369 settings fitted
in 20 s (`logs/2026-11-06/B1_zoo.log`). `aotil/audit_cell.py`: **verdict PASS**, 0 FAIL, 0 WARN
(`logs/2026-11-06/B2_audit_cell.log`): the 361 kept and 8 dropped settings cover the grid once, every row equals
its setting, the drop rule holds, every model uses all 7 features and all 29,161 training rows, 10 refits match
to 0.0 (one per family plus xgb, mlp, hgb) and all 8 dropped settings refit to the same flood share. The data
split is unchanged (train 29,161; ID test 10,835; OOD test 12,531; 28 test tiles). Summary script:
`logs/2026-11-06/B3_cell_summary.py` (log beside it, `results/2026-11-06/cell_summary.json`).

| Family | Settings | Degenerate (dropped) | Kept | ID accuracy of the kept (min, median, max) | Kept that repeat an earlier model's predictions | Preview: dropped / kept / repeat |
|---|---:|---:|---:|---|---:|---|
| dt (decision tree) | 40 | 0 | 40 | .734, .769, .794 | 16 | 0 / 40 / 16 |
| et (extra trees) | 54 | 0 | 54 | .692, .771, .804 | 0 | 1 / 53 / 0 |
| hgb (hist. gradient boosting) | 48 | 3 | 45 | .662, .778, .803 | 9 | 3 / 45 / 9 |
| lr (logistic regression) | 50 | 0 | 50 | .712, .723, .741 | 13 | 0 / 50 / 13 |
| mlp (multilayer perceptron) | 45 | 2 | 43 | .510, .765, .787 | 0 | 2 / 43 / 0 |
| nb (Gaussian naive Bayes) | 30 | 0 | 30 | .520, .609, .757 | 0 | 0 / 30 / 0 |
| rf (random forest) | 54 | 0 | 54 | .644, .773, .803 | 5 | 0 / 54 / 4 |
| xgb (XGBoost) | 48 | 3 | 45 | .660, .777, .804 | 8 | 3 / 45 / 10 |
| **Total** | **369** | **8** | **361** | **.510, .766, .804** | **51** | 9 / 360 / 52 |

"Repeat an earlier model's predictions": the model's 0/1 predictions on all 23,366 test points (both halves)
equal those of a kept model earlier in grid order. The preview (`Claude outputs/zoo-grid-preview-2026-11-06/`)
used other library versions and seeds, so small differences were expected: one extra-trees setting it dropped
is kept here, and two duplicate counts differ by one or two.

The 8 dropped settings (`data/interim/aotil/cells_grid_2026-11-06/FL_band_AE_s0_dropped.csv`): seven predict no
flood at any ID test point (hgb and xgb at learning rate 0.01 with 3 rounds, at every depth; mlp (2,) at 3 epochs
and learning rate 1e-3), so their accuracy equals the dry share, 0.453; one predicts flood at 99.7% of them (mlp
(16,) at 3 epochs and learning rate 1e-4).

| Statistic | Value |
|---|---|
| Kept N | **361** |
| 60/20/20 split (sklearn, random_state 42) | **216 selection, 72 validation, 73 test** |
| ID accuracy | 0.510 to 0.804; sd 0.050; quantiles 0 / 10 / 25 / 50 / 75 / 90 / 100%: .510, .711, .732, .766, .778, .792, .804 |
| Kept models below 0.70 ID accuracy | 32: nb 20, mlp 5, et 3, rf 2, hgb 1, xgb 1 |
| Distinct prediction patterns among the 73 test models | **69** on both test halves (68 on the Zone A half alone): 4 test models repeat another test model |
| Distinct prediction patterns among all 361 | 310 |
| Probit r(ID, OOD) on all Zone A test points | **+0.849** over all 361 kept models; **+0.806** [+0.707, +0.874] over the 73 test models |
| Without naive Bayes (N1, first half) | +0.822 over the 331 others; +0.769 over the 67 test models that are not nb |
| OODSelect's model-count rule (`report_paper.model_convergence`: add models in steps of 10 until r moves by < 1%, 25 random orders) | **40** models |
| Interpolating models (training accuracy ≥ 0.99) | 21 of 361 (et 9, rf 9, dt 1, hgb 1, xgb 1); **6 of the 73 test models** |
| Test models by family | dt 9, et 15, hgb 9, lr 8, mlp 7, nb 6, rf 9, xgb 10 |

**Stops (none fired):** kept N = 361 ≥ 200; the lowest kept ID accuracy is 0.510 ≤ 0.60; every family keeps at
least 30 models (≥ 10); probit r(ID, OOD) over all kept models is 0.849 ≥ 0.5. So Part C started without
waiting.

Against the 300-model cell of 2026-10-25: ID accuracy 0.547 to 0.807 before, 0.510 to 0.804 now; probit r over
all models 0.88 before (the brief), 0.849 now; the model-count rule needs 40 models now against 10 before.

---

## 3. Part C: the full run of the search, and the claims

### 3.1 What ran

Every command, setting and seed of 2026-10-25, on the new cell (`logs/2026-11-06/T_chain.sh`, `C_chain.sh`,
`D_chain.sh`; the only differences from `logs/2026-10-25/` are the date and the cell folder):
- the shift, pooled and per tile (`o1_shift.py`), its table (`o1_table.py`) and the tile-shuffle null
  (`o1_tile_null.py`, seeds 1 to 3);
- **81 searches**, all with the squared budget penalty, T = 100 TPE trials, at most 50,000 steps, patience 500,
  search seed 0, k = 10: OODSelect (classic), Laplacian (μ = 3) and GTV (μ = 3) over the full budget grid (10 to
  10,000: 36 searches); the shuffled-accuracy controls of all three at S = 250, 500 and 1,000 (null seeds 1 to 3:
  27); both family-disjoint directions for classic and Laplacian (12); μ = 1 and μ = 10 (6). **All 81 finished; none
  failed** (`logs/2026-11-06/C_chain.log`);
- the selection audits (`audit_selection.py`: 346 checks, 0 FAILED; the 17 warnings are tie notices at the
  smoothing cut), smoothed-classic, the regions (Laplacian S = 250 and 500, classic and GTV S = 500),
  `regions_land.py`, the region description, the memorized-noise check (`o5_interpolation.py`), `region_match.py`
  and both figures (`results/2026-11-06/paper_figures/`).

Scripts that turn the outputs into the numbers below, all in `logs/2026-11-06/` with their logs: `E_claims.py`
(C1 to C7, the μ rule, the largest inverted budget, N1; `results/2026-11-06/claims.json`), `E_paper_numbers.py` and
`E_paper_numbers_extra.py` (every paper number, 2026-10-25 beside 2026-11-06), `E_budget_table.py`,
`E_gtv_vs_laplacian.py` (`results/2026-11-06/gtv_vs_laplacian.csv`) and `F_check_paper_numbers.py` (recomputes the
numbers written into the paper from the result files and checks each phrase in the edited file: 73 checks since
the full Table tab:regions went in on 2026-11-07, 61 before; ALL MATCH).

### 3.2 Claims C1 to C7

C1 to C5 as fixed in `docs/TASK-REBUILD-V3.md` §2 (C1 on the unweighted per-tile columns, as the brief says),
C6 and C7 as fixed in `docs/TASK-RERUN-ALL-2026-10-25.md` §2. Split 1 (seed 0), 361 models, 73 test models,
S = 500 unless stated.

| # | Statistic | 2026-11-06 | Holds? | 2026-10-25 (300 models) |
|---|---|---|---|---|
| C1 | Pooled shift (mean over the models of ID minus Zone A; tile bootstrap) and the per-tile gap (unweighted) | Pooled: accuracy −0.008 [−0.065, +0.048], AUROC −0.016 [−0.074, +0.039], F1 +0.019 [−0.067, +0.092]. Per tile: accuracy **+0.046 [+0.021, +0.074]**, worse on Zone A in 22 of 28 tiles, 94% of models worse; AUROC +0.043 [+0.007, +0.079] (27 tiles); F1 +0.093 [+0.053, +0.139]; strongest 15: accuracy +0.079. Null (zones shuffled within tile and label, 3 seeds): accuracy and AUROC within 0.023 of 0, F1 +0.035 to +0.041 | Reported | Pooled −0.007; per tile +0.050 [+0.026, +0.079], 22 of 28, 92%; AUROC +0.060; F1 +0.104; null within 0.019, F1 +0.036 to +0.043 |
| C2 | r on all Zone A test points, test models | **+0.806** [+0.707, +0.874] | **Holds** (> +0.3) | +0.849 |
| C3 | Classic r_te; pieces, tiles, purity; the 3 shuffled classic runs at S = 500 | **−0.977** [−0.986, −0.964]; 291 pieces, 27 of 28 tiles, purity 0.126. Shuffled: +0.332, +0.743, −0.028 | **Holds** | −0.981; 275 pieces, 25 tiles |
| C4 | Laplacian r_te, purity ≥ 0.5, pieces ≤ classic pieces / 5, purity above smoothed-classic at S = 250, 500, 1,000 | **−0.900** [−0.936, −0.845], purity 0.671, **71 pieces (classic/5 = 58)**. Purity, ours vs smoothing: **0.651 vs 0.700, 0.671 vs 0.692, 0.724 vs 0.727** | **Does not hold**: r_te and purity hold; pieces fail (71 > 58), and smoothing's purity is higher at all three budgets. Smoothing now keeps an inversion: r_te −0.648, −0.519, −0.337 | Did not hold either (purity at 250 and 1,000); smoothing r_te −0.549, −0.152, +0.096 |
| C5 | r on the held-out test models below 99% training accuracy, Laplacian S = 500; family-disjoint at S = 500 | **−0.905** [−0.941, −0.850], 67 models. Laplacian trees→others −0.758, others→trees −0.561 (classic −0.809, −0.659) | **Holds** | −0.910, 51 models; −0.710, −0.935 |
| C6a | Pooled unmatched share of the regions | **0.029** (10 of 343 points); per region 0 to 0.115; above 0.10: **Hamilton 0.115** (possible covariate shift) | **Holds** (≤ 0.10) | 0.013; none above 0.10 |
| C6b | Pooled η of the regions against the Zone AE baseline η | **0.649 against 0.297** (needs ≥ 0.397); every region above the baseline (lowest Highlands 0.413) | **Holds** | 0.664 vs 0.297 |
| C7 | GTV S = 500 (descriptive) | −0.858 [−0.909, −0.782]; purity **0.916**; **15 pieces**; 11 tiles; 9 regions holding 440 points | Reported | −0.877; purity 0.671; 122 pieces; 24 tiles; 7 regions, 345 points |

No Stop fired: Part B's four conditions (§2), the grids needed no change, the μ rule picks μ = 3 (§3.3), and C6a
and C6b hold.

### 3.3 The paper's rule statements

- **μ rule (no Stop).** Purity at S = 500: μ = 1 **0.429**, μ = 3 **0.671**, μ = 10 0.911. The smallest μ of
  {1, 3, 10} with purity ≥ 0.5 is still **μ = 3**, so the Search paragraph's sentence stands unchanged. (μ sweep at
  S = 250 / 500 / 1,000: purity 0.448 / 0.429 / 0.506 at μ = 1, 0.651 / 0.671 / 0.724 at μ = 3, 0.793 / 0.911 /
  0.731 at μ = 10; r_te −0.930 / −0.944 / −0.914, −0.909 / −0.900 / −0.908, −0.896 / −0.904 / −0.860.)
- **Largest inverted budget.** The largest grid budget with corr_te ≤ −0.3 is **S = 7,500** for OODSelect
  (−0.635), the Laplacian (−0.488) and GTV (−0.374); at 10,000 none inverts (+0.066, +0.102, +0.109), and every budget
  from 10 to 7,500 inverts for all three. The Budget penalty sentence ("holds up to S = 7,500") stands unchanged.
- **N1 (report only).** Probit r(ID, OOD) on all Zone A points without the 30 naive Bayes models: **+0.822** over 331
  models (with them +0.849); over the 67 test models that are not nb, +0.769. corr_te of the Laplacian S = 500 subset
  over those 67 test models: **−0.915** [−0.947, −0.865] (all 73: −0.900). So the aggregate line and the inversion do
  not lean on naive Bayes, although 20 of the 32 models below 0.70 ID accuracy are nb.

### 3.4 The searches across budgets

corr_te (purity in parentheses); random purity is S/m (`logs/2026-11-06/E_budget_table.log`). The 2026-11-06 paper
no longer has this table; it is here for the record.

| S | Random purity | OODSelect | Laplacian | GTV |
|---:|---:|---|---|---|
| 10 | .00 | −.81 (.01) | −.84 (.56) | −.89 (.02) |
| 20 | .00 | −.95 (.01) | −.82 (.60) | −.84 (.40) |
| 50 | .00 | −.94 (.02) | −.85 (.03) | −.89 (.14) |
| 100 | .01 | −.96 (.04) | −.91 (.53) | −.88 (.80) |
| 250 | .02 | −.98 (.07) | −.91 (.65) | −.88 (.92) |
| 500 | .04 | −.98 (.13) | −.90 (.67) | −.86 (.92) |
| 750 | .06 | −.97 (.17) | −.90 (.68) | −.82 (.88) |
| 1,000 | .08 | −.97 (.19) | −.91 (.72) | −.83 (.92) |
| 2,500 | .20 | −.96 (.31) | −.84 (.68) | −.68 (.85) |
| 5,000 | .40 | −.82 (.54) | −.71 (.80) | −.58 (.92) |
| 7,500 | .60 | −.64 (.75) | −.49 (.89) | −.37 (.93) |
| 10,000 | .80 | +.07 (.88) | +.10 (.90) | +.11 (.91) |

**GTV against the Laplacian** (`results/2026-11-06/gtv_vs_laplacian.csv`). GTV has higher purity at 10 of the 12
budgets (not at S = 10 and 20) and inverts less at every budget from 100 up: by 0.03 at 100 to 250, 0.04 at 500,
0.08 at 750 and 1,000, and 0.11 to 0.16 from 2,500 to 7,500. Its relaxed weights are sharper, as Approach 3.5
expects: at S = 500, 8.0% of its final weights lie strictly between 0.05 and 0.95, against 20.4% for the Laplacian.
On 2026-10-25 GTV was the *less* coherent of the two at S = 500 (122 pieces, purity 0.671, against 31 and 0.768);
now it is the more coherent (15 pieces, purity 0.916, against 71 and 0.671).

### 3.5 The controls in detail

| Control | OODSelect | Laplacian | GTV |
|---|---|---|---|
| Shuffled, S = 250 (seeds 1/2/3; real) | +0.495 / +0.798 / −0.222 (−0.977) | +0.184 / −0.669 / −0.883 (−0.909) | +0.251 / +0.599 / −0.674 (−0.884) |
| Shuffled, S = 500 | +0.332 / +0.743 / −0.028 (−0.977) | +0.408 / −0.662 / −0.780 (−0.900) | −0.008 / +0.723 / −0.151 (−0.858) |
| Shuffled, S = 1,000 | +0.496 / +0.773 / −0.779 (−0.969) | +0.715 / −0.606 / −0.754 (−0.908) | +0.299 / +0.774 / −0.275 (−0.831) |
| Median of the 9; how many ≤ −0.3; real run lower at its budget | **+0.495**; 1; **9 of 9** | **−0.662**; 6; **9 of 9** | +0.251; 1; 9 of 9 |
| 2026-10-25 | −0.147; 3; 9 of 9 | −0.540; 5; 8 of 9 | +0.479; 1; 9 of 9 |
| Family-disjoint, trees→others (S = 250 / 500 / 1,000) | −0.854 / −0.809 / −0.824 | −0.812 / −0.758 / **−0.025** | |
| Family-disjoint, others→trees | −0.720 / −0.659 / −0.544 | −0.728 / −0.561 / −0.573 | |
| Memorization: the 67 held-out test models below 99% training accuracy | −0.980 [−0.987, −0.967] | −0.905 [−0.941, −0.850] | |
| Within each family that never interpolates (lr, mlp, nb; circular) | | lr −0.937, mlp −0.805, nb −0.975 | |

The real Laplacian search now beats all 9 of its shuffled runs (8 of 9 on 2026-10-25), but 6 of the 9 shuffled
Laplacian runs still invert, down to −0.88, so a negative corr_te alone stays weak evidence; the shuffled Laplacian
runs that invert are also coherent (purity 0.42 to 0.79). The trees→others Laplacian run at S = 1,000 does not
invert (−0.025; not in the paper, which reports S = 500). Six test models interpolate their training labels now (5
extra trees, 1 random forest), against 9 before; three families never interpolate (lr, mlp, nb), against four.

### 3.6 Every number the paper states, 2026-10-25 → 2026-11-06

The paper's numbers before this run were those of 2026-10-25 (the 300-model cell). Values as the paper rounds them;
**bold** marks a number whose sentence I did not edit because its claim no longer holds (§5.2).

| Where | Number | 2026-10-25 | 2026-11-06 |
|---|---|---|---|
| Introduction | models | XX (300) | 361 |
| Introduction, Results | per-tile accuracy gap ("lose up to") | 0.050 | 0.046 |
| Introduction, Results, tab:search | corr on all Zone A, test models | 0.85 [.76, .91] | 0.81 [.71, .87] |
| Introduction, Results, tab:search | OODSelect r_te; pieces; tiles | −0.98 [−.99, −.97]; 275; 25 | −0.98 [−.99, −.96]; 291; 27 |
| Introduction, Results, tab:search | Laplacian r_te; pieces; tiles | −0.92 [−.95, −.86]; 31; 15 | −0.90 [−.94, −.84]; 71; 20 |
| Introduction, Results | cost of coherence | 0.065 | 0.077 |
| Introduction, Results | pooled η (as a %) | 66% (0.66) | 65% (0.65) |
| Models | N; selection / validation / test | XX; XX / XX / XX | 361; 216 / 72 / 73 |
| fig:maps | regions | 6 | 8 |
| tab:search | Random; Hardest | +.84; −.68 | +.79; −.27 |
| tab:search, Results | GTV r_te; pieces; tiles | −.88 [−.92, −.80]; 122; 24 | −.86 [−.91, −.78]; 15; 11 |
| tab:search | Smoothing r_te; pieces; tiles | −.15 [−.39, .11]; 51; 18 | −.52 [−.67, −.33]; 51; 24 (**the sentence still says −0.15**) |
| tab:search caption | test models | XX (60) | 73 |
| Results | regions; points in them | Six; 381 | Eight; 343 |
| Results, Pooling | pooled gap | −0.007 [−0.060, 0.045] | −0.008 [−0.065, 0.048] |
| Results, Pooling | Palm Beach tile accuracy | 0.52 | 0.51 |
| Results, Pooling | tiles where Zone A is worse | 22 of 28 | 22 of 28 |
| Results, Pooling | per tile: accuracy; AUROC; F1; models worse | 0.050 [0.026, 0.079]; 0.060; 0.104; 92% | 0.046 [0.021, 0.074]; 0.043; 0.093; 94% |
| Results, Pooling | null: accuracy and AUROC within; F1 range | 0.019; 0.036 to 0.043 | 0.023; 0.035 to 0.041 |
| Results, Controls | shuffled medians (OODSelect, Laplacian); real beats | −0.15, −0.54; 8 of 9 | +0.49, −0.66; 9 of 9 |
| Results, Controls | most negative shuffled run (Laplacian) | −0.67 | −0.88 |
| Results, Controls | interpolating test models; corr_te without them | 9; −0.91 [−0.95, −0.85] | 6; −0.91 [−0.94, −0.85] |
| Results, Controls | families that never interpolate; their range | four; −0.70 to −0.97 | three; −0.81 to −0.98 |
| Results, Controls | trees→others; reverse | −0.71; −0.93 | −0.76; −0.56 |
| Results, Same inputs | pooled unmatched | **1.3%** | 2.9% (**sentence not edited**) |
| Results, Same inputs | positive regions: count, points / matches | two, 98% / 34% | three, 84% / 35% |
| Results, Same inputs | negative regions: count, points / matches | four, 12% / 66% | five, 19% / 64% |
| Results, Same inputs | Hamilton unmatched | 9% | 11% |
| tab:regions | region rows; All regions | 6 regions: Levy 114, Dixie/Levy 41, Liberty 123, Hamilton 54, Palm Beach 26, Martin 23; All regions 381 | 8 regions (§4.2; applied 2026-11-07): Dixie 31, Highlands 31, Levy 28, Liberty 107, Hamilton 61, Manatee/Sarasota 38, Hendry/Palm Beach 27, Hendry 20; All regions 343 |
| Case studies | Levy 114 points, 0.35 m, 35%; Liberty 123 points, 0.7 km, 6% vs 72% | | **paragraph not edited** (§5.2) |
| Appendix A, tab:models | dropped; N; per family; total | XX | 8; 361; 40, 54, 54, 45, 45, 50, 43, 30; 361 |
| Unchanged | S = 7,500; μ = 3 rule; ε = 1.39; 50 Zone AE points without a feature; 22 of 28 tiles; Other Zone A and Zone AE baseline rows; 2.5 km; 4.0% | | |

---

## 4. The regions and the region analysis (Approach 3.6)

### 4.1 The regions changed

The Laplacian S = 500 selection now has **8 regions** (pieces of at least 20 points) holding **343 of the 500
points** (2026-10-25: 6 regions, 381 points). `logs/2026-11-06/D3_regions_describe.log` names each region by its
counties and tests whether each 2026-10-25 region is *found again* (at least half of its points within 2 km of a
region of this run, the footprint rule of `docs/TASK-REBUILD-V3.md` Part D):

| 2026-10-25 region | Points | Share within 2 km of this run's regions | Found again? | Its successor in this run |
|---|---:|---:|---|---|
| Liberty | 123 | 0.91 | yes | Liberty (107 points) |
| Levy | 114 | 0.32 | no | Levy (28 points), a small piece inside it (all 28 within 2 km of the old Levy) |
| Hamilton | 54 | 1.00 | yes | Hamilton (61 points) |
| Dixie/Levy | 41 | 0.68 | yes | Dixie (31 points; its points now lie in Dixie county only) |
| Palm Beach | 26 | 0.00 | no | none (nearest region point 71 km away) |
| Martin | 23 | 0.00 | no | none (85 km) |

New regions: **Manatee/Sarasota** (38 points), **Highlands** (31), **Hendry/Palm Beach** (27; the cultivated
cropland south of Lake Okeechobee that the softplus run of 2026-10-23 and the GTV run of 2026-10-25 had found) and
**Hendry** (20). So the brief's "if the regions change" case applies: I did not rewrite the case studies or any
region name in the paper, and I flag the paragraphs (§6). Below are the new regions with the descriptors the case
studies use.

### 4.2 Every set, with the case-study descriptors

The paper's rule: a region's candidate matches exclude every test tile that holds a point of the region.
**Caliper ε = 1.391 standardized units** (unchanged: ε depends only on the Zone AE points, not on the models);
**50** Zone AE test points are left out for a missing feature (unchanged). `results/2026-11-06/region_match/`,
`logs/2026-11-06/D5_region_match.log`.

Columns: n = points; Unmatched = share with no Zone AE point within ε; η = label disagreement (Equation
disagree); Positives = share of positives among the points / among their matches; HAND = median height above the
nearest drainage of the points / of their matches (m); River = median distance to the nearest NHDPlus stream (km),
points / matches; Land cover = the top classes of the points; Acc15 = the 15 strongest test models' accuracy on the
points / on their matches (descriptive, not in 3.6).

| Set | n | Unmatched | η | Positives | HAND (m) | River (km) | Land cover (top 3) | Acc15 |
|---|---:|---:|---:|---|---|---|---|---|
| Liberty | 107 | 0.000 | 0.712 | 0.06 / 0.73 | 1.6 / 1.8 | 0.65 / 0.63 | woody wetlands 58%, evergreen forest 29%, developed open space 6% | 0.29 / 0.84 |
| Hamilton | 61 | **0.115** | 0.640 | 0.43 / 0.63 | 0.7 / 1.3 | 1.88 / 1.63 | evergreen forest 33%, woody wetlands 25%, emergent herbaceous wetlands 13% | 0.30 / 0.75 |
| Manatee/Sarasota | 38 | 0.026 | 0.605 | 0.34 / 0.34 | 0.9 / 0.9 | 0.36 / 0.52 | developed low intensity 34%, pasture/hay 16%, woody wetlands 16% | 0.32 / 0.70 |
| Dixie | 31 | 0.000 | 0.465 | 0.55 / 0.30 | 2.3 / 1.2 | 2.66 / 2.20 | woody wetlands 55%, evergreen forest 23%, grassland 6% | 0.32 / 0.65 |
| Highlands | 31 | 0.000 | 0.413 | 1.00 / 0.59 | 0.3 / 1.1 | 2.18 / 1.75 | pasture/hay 35%, shrub/scrub 29%, woody wetlands 23% | 0.53 / 0.85 |
| Levy | 28 | 0.000 | 0.871 | 1.00 / 0.13 | 0.25 / 1.0 | 1.69 / 1.49 | evergreen forest 68%, woody wetlands 18%, shrub/scrub 14% | 0.15 / 0.81 |
| Hendry/Palm Beach | 27 | 0.037 | 0.600 | 0.07 / 0.62 | 0.3 / 0.4 | 23.6 / 24.8 | cultivated crops 93%, developed open space 4%, developed low intensity 4% | 0.30 / 0.78 |
| Hendry | 20 | 0.050 | 0.832 | 0.10 / 0.80 | 0.65 / 0.4 | 16.2 / 19.3 | woody wetlands 55%, pasture/hay 20%, developed open space 10% | 0.25 / 0.71 |
| **All regions (8)** | 343 | **0.029** | **0.649** | 0.36 / 0.56 | 0.8 / 1.2 | 1.24 / 1.31 | woody wetlands 36%, evergreen forest 22%, cultivated crops 7% | 0.31 / 0.78 |
| Zone AE baseline | 10,785 | 0.050 | **0.297** | 0.55 / 0.53 | 1.2 / 1.2 | 1.39 / 1.37 | woody wetlands 29%, cultivated crops 15%, evergreen forest 13% | 0.79 / 0.80 |
| Other Zone A points | 12,031 | 0.046 | 0.253 | 0.52 / 0.51 | 1.7 / 1.6 | 1.52 / 1.59 | woody wetlands 33%, evergreen forest 21%, cultivated crops 10% | 0.81 / 0.80 |
| 500 hardest points | 500 | 0.050 | 0.816 | 0.95 / 0.23 | 1.9 / 1.9 | 1.60 / 1.59 | evergreen forest 56%, developed open space 15%, shrub/scrub 6% | 0.00 / 0.81 |
| OODSelect S = 500 | 500 | 0.048 | 0.737 | 0.45 / 0.57 | 1.5 / 1.5 | 1.08 / 1.16 | woody wetlands 38%, evergreen forest 20%, pasture/hay 13% | 0.18 / 0.78 |
| Halo 1 km | 57 | 0.000 | 0.409 | 0.30 / 0.57 | 0.9 / 1.0 | 0.80 / 0.99 | woody wetlands 23%, pasture/hay 21%, developed open space 14% | 0.67 / 0.74 |
| Halo 2 km | 137 | 0.000 | 0.399 | 0.49 / 0.59 | 0.6 / 1.0 | 2.13 / 2.11 | woody wetlands 30%, evergreen forest 13%, pasture/hay 13% | 0.67 / 0.75 |
| Halo 3 km | 232 | 0.009 | 0.400 | 0.59 / 0.57 | 0.6 / 1.0 | 2.11 / 1.91 | woody wetlands 29%, evergreen forest 15%, cultivated crops 14% | 0.68 / 0.76 |

The largest standardized mean difference between a region's matched points and their matches (each match
weighted 1/|matches|) is 0.65 SD for Hendry/Palm Beach (distance to the coast) and 0.26 for Hamilton (distance to
the coast); every other region is below 0.18 (`region_match.csv`, `smd_max`). Hendry/Palm Beach's matches come
from only 10 distinct Zone AE points.

**Reading.** Pooled, the regions are matched more often than Zone AE points are (unmatched 0.029 against 0.050), so
their inputs are familiar (C6a). Their η is 2.2 times the baseline's (0.649 against 0.297), and every region lies
above the baseline (lowest: Highlands 0.413), so the same inputs carry different labels (C6b). The other Zone A
points sit below the baseline (0.253), so the disagreement is local; the halos lie between (0.40 to 0.41). The 500
hardest points reach the highest η (0.816) and OODSelect's subset 0.737: points chosen for errors have high η
whatever their coherence, as the paper's Limitations caveat says. **Hamilton** is the only region above 0.10
unmatched (0.115): possible covariate shift. **Manatee/Sarasota** holds as many positives as its matches (0.34 /
0.34), so its disagreement runs both ways inside the region rather than in one direction.

**Two kinds of failure** (majority label, `region_match_kinds.csv`):

| Kind | Regions (points) | Positives, points / matches | Unmatched | η | Strongest-15 errors: FN / FP | corr_te, 73 test models | corr_te, 67 test models below 99% training accuracy |
|---|---|---|---:|---:|---|---:|---:|
| Positive | Dixie, Highlands, Levy (90) | 0.84 / 0.35 | 0.000 | 0.573 | 0.83 / 0.17 (895 errors) | −0.593 | −0.622 |
| Negative | Liberty, Hamilton, Manatee/Sarasota, Hendry/Palm Beach, Hendry (253) | 0.19 / 0.64 | 0.040 | 0.677 | 0.16 / 0.84 (2,673 errors) | −0.776 | −0.763 |

**Sensitivity** (report only):

| Setting | ε | Baseline unmatched / η | Regions unmatched / η | Other Zone A unmatched / η |
|---|---:|---|---|---|
| As run (95th percentile, at most 5 matches) | 1.391 | 0.050 / 0.297 | 0.029 / 0.649 | 0.046 / 0.253 |
| ε at the 90th percentile | 1.070 | 0.100 / 0.302 | 0.058 / 0.650 | 0.093 / 0.256 |
| ε at the 99th percentile | 2.290 | 0.010 / 0.294 | 0.015 / 0.648 | 0.010 / 0.251 |
| At most 1 match | 1.391 | 0.050 / 0.287 | 0.029 / 0.673 | 0.046 / 0.253 |
| At most 10 matches | 1.391 | 0.050 / 0.295 | 0.029 / 0.647 | 0.046 / 0.251 |

C6a and C6b hold under every setting: the regions' unmatched share stays at or below the baseline's, and their η
stays at 0.65 or more against a baseline of at most 0.30.

**GTV's regions (C7, report only).** 9 regions of at least 20 points holding 440 of its 500 points: Liberty (131
and 20 points), Levy (70 and 22), Highlands (65), Citrus/Levy (46), Manatee (40), Volusia (25) and Dixie (21). Their
unmatched shares are 0 to 0.025 and their η 0.36 to 0.89 (`region_match.csv`, kind `extra_region`).

---

## 5. Part D: the paper (numbers only)

Backup of the file before the edits: `Claude outputs/paper-before-2026-11-06-zoo-grids/acl_latex.tex` (and
`custom.bib`, unchanged). In the local `CS371_Cho_Li/acl_latex.tex` I changed only numbers, the 17 `\textbf{XX}`
placeholders and one figure file. The edits are scripted in `logs/2026-11-06/D_paper_edits.py` (52 exact
replacements, each matching once). On 2026-11-07, at your go-ahead, `logs/2026-11-06/G_regions_table.py` also
replaced the region rows of Table tab:regions with the 8 regions of this run (F4; backup before it:
`Claude outputs/paper-before-2026-11-07-regions-table/acl_latex.tex`). `logs/2026-11-06/F_check_paper_numbers.py`
recomputes every new number from the result files and finds each phrase in the edited file (73 checks, ALL MATCH).
The Appendix below has all 28 edits as Before/After blocks to paste over the same lines in Overleaf
(`logs/2026-11-06/F_paper_edits.md`): 27 edited lines and one block for the rows of Table tab:regions.

### 5.1 What changed

| Where | Change | Edits |
|---|---|---|
| Introduction, results paragraph | XX → 361 models; 0.050 → 0.046; 0.85 → 0.81; 275 → 291 pieces; 31 → 71 pieces; 0.065 → 0.077; 66% → 65% | 1 |
| Implementation details, Models | N = XX → 361; XX / XX / XX → 216 / 72 / 73 | 2 |
| Figure fig:maps | `fig_maps_v4.pdf` → `fig_maps_v5.pdf` (new file; v3 and v4 kept): the 8 new regions, all labeled; caption 25 → 27 tiles, 275 → 291 pieces, 31 → 71 pieces, 15 → 20 tiles, 6 → 8 regions | 3, 4 |
| Table tab:search | every row (§3.6) and the caption's XX → 73 | 5 to 12 |
| Table tab:regions | **2026-11-07, at your go-ahead (F4):** the region rows are those of this run, the positive regions Dixie, Highlands and Levy, then the negative regions Liberty, Hamilton, Manatee/Sarasota, Hendry/Palm Beach and Hendry, each kind by size; All regions pools the 8. The 2026-10-25 rows Dixie/Levy, Palm Beach and Martin are gone. Other Zone A and the Zone AE baseline round to the same values | 13 (one block) |
| Results: Pooling hides the shift | pooled gap, Palm Beach accuracy, per-tile gaps, models worse, the null's bounds, corr_te | 14 |
| Results: OODSelect inverts, but scatters | +0.84 → +0.79, −0.68 → −0.27, 275 → 291 pieces, 25 → 27 tiles | 15 |
| Results: Our loss turns the subset into regions | Laplacian numbers, Six → Eight pieces holding 381 → 343 points, 0.065 → 0.077, GTV −0.88 → −0.86 and 122 → 15 pieces. The smoothing sentence is **unchanged** (F1) | 16 |
| Results: Controls | shuffled medians, 8 → 9 of 9, −0.67 → −0.88, 9 → 6 interpolating test models, CI, four → three families and their range, family-disjoint −0.71 → −0.76 and −0.93 → −0.56 | 17 |
| Results: Same inputs, different labels | η 0.66 → 0.65, two → three positive regions at 84% / 35%, four → five negative regions at 19% / 64%, Hamilton 9% → 11%. The first sentence is **unchanged** (F2) | 18 |
| Appendix A and Table tab:models | XX → 8 dropped and N = 361; the Models column 40, 54, 54, 45, 45, 50, 43, 30; total 361 | 19 to 28 |

Unchanged, because their numbers do not move or their claims still hold as written: the Budget penalty sentence
("holds up to S = 7,500"), the Search paragraph's μ rule (§3.3), the Graph paragraph, Data construction and Table
tab:data, the Region Analysis subsection (ε = 1.39; ε depends only on the Zone AE points), and the Case studies
paragraph (flagged, F3). The local Abstract is empty, so the brief's "Abstract's model count" placeholder is the one
in the Introduction ("We train 361 tabular models"); if your Overleaf abstract has an XX, it is 361.

### 5.2 Flagged: sentences whose claim no longer holds (left as they are)

**F1. Results, "Our loss turns the subset into regions", the smoothing sentence.** It says: "Smoothing OODSelect's
final weights over each point's neighbors also gives a coherent subset (51 pieces), but it loses the inversion
($-0.15$), so coherence must be part of the objective." Now smoothing keeps an inversion: r_te **−0.52 [−0.67,
−0.33]**, 51 pieces in 24 tiles, purity 0.692, which is above our Laplacian's 0.671 (71 pieces). Across budgets:
−0.65, −0.52, −0.34 at S = 250, 500, 1,000 (purity 0.700, 0.692, 0.727; ours 0.651, 0.671, 0.724). What still holds:
smoothing inverts much less than our search (−0.52 against −0.90 at S = 500), so its inversion is weaker, not gone.
Table tab:search already shows the new smoothing row, so the table and this sentence now disagree until you rewrite
the sentence. This is also why C4 fails (§3.2).

**F2. Results, "Same inputs, different labels", second sentence.** It says: "Pooled, 1.3\% of their points have no
match, against 5\% of Zone AE points, and no region exceeds 10\%." Now the pooled share is **2.9%** (10 of 343
points) and **Hamilton has 11.5%**, so "no region exceeds 10%" no longer holds. The paragraph's last sentence,
"Only Hamilton, with 11% unmatched points, may partly reflect covariate shift", holds and was updated.

**F3. Results, "Case studies" (the whole paragraph).** The regions changed (§4.1), so per the brief I did not touch
the case studies or the names. It says "Levy is the largest positive region. All 114 of its points are positives, on
woody wetland and evergreen forest at a median of 0.35\,m above the nearest drainage, but only 35\% of its matches
are." Now Levy is a 28-point piece of the old region and **not the largest positive region** (Dixie and Highlands
have 31 points; Highlands is all positives). The new Levy: all 28 points positive; evergreen forest 68%, woody
wetlands 18%, shrub/scrub 14%; median 0.25 m above the nearest drainage; 1.7 km from a mapped river; 13% of its
matches are positives; η 0.87. Liberty still holds as written with new numbers: the largest negative region, 107
points, woody wetlands 58% (evergreen forest 29%), 0.65 km from a mapped river, 6% positives against 73% of its
matches, η 0.71. "Both lie on low, wooded land near streams" still roughly holds (median height above drainage
0.25 m and 1.6 m; river 1.7 km and 0.65 km). The new regions with the same descriptors are in §4.2.

**F4. Table tab:regions, region rows. Applied on 2026-11-07, at your go-ahead.** On 2026-11-06 I updated only
the rows of Levy, Liberty, Hamilton and All regions, which left the rows Dixie/Levy, Palm Beach and Martin stale:
they are no longer regions (Dixie/Levy's successor is "Dixie", 31 points; Palm Beach and Martin have no region within
70 km), and the five new regions had no rows. The table now lists the 8 regions of this run, the positive ones first
and each kind by size, as tested on 2026-11-06 (`logs/2026-11-06/G_regions_table.py`; paste block: Appendix,
Edit 13). `F_check_paper_numbers.py` recomputes every row from `results/2026-11-06/region_match/region_match.csv`
and checks the order, the kinds (a region is positive when most of its points are positives, as in
`region_match.py`) and that no old row is left. The figure already labels these 8 names. The Case studies paragraph,
which names the old regions, is still flagged (F3).

### 5.3 Still true, but worth a look

- **GTV versus the Laplacian reversed.** "GTV inverts less ($-0.86$) and leaves 15 pieces at this budget" is true,
  but on 2026-10-25 GTV was the less coherent of the two (122 pieces against 31), and now it is far more coherent
  (15 pieces and purity 0.92 against 71 and 0.67), as Approach 3.5 expects (§3.4). You may want the sentence to say so.
- **"the real Laplacian search beats 9 of its 9"** is the numbers-only edit; "beats all 9" reads better.
- **OODSelect's shuffled median is now positive** (+0.49 against −0.15), and only 1 of its 9 shuffled runs inverts.
- **"models lose up to 0.046 accuracy"** (Introduction) is the per-tile mean gap, as before.
- **C4 does not hold** (§3.2): the paper does not state C4's conditions, but F1 is its visible part.
- **Manatee/Sarasota**, a new row of Table tab:regions, is a negative region by the majority rule (34% of its
  points are positives), but its matches are positives at the same rate (.34 / .34). Its disagreement (η .61) runs
  in both directions. "Zone A leaves out land that Zone AE would map" pools the five negative regions and still
  holds (19% against 64%).
- **A pre-existing overfull box**, 15.4 pt, in Table tab:models (lines 283 to 299 of the current file), present
  before and after these edits. Changing `\begin{tabular}{@{}lp{0.6\textwidth}rr@{}}` to `p{0.56\textwidth}`
  removes it (tested; 0.57 does not). Not applied, since it is not a number.

### 5.4 Compile

pdflatex, bibtex, pdflatex ×3 on a copy: **10 pages, as before**; every labeled float (Tables 1 to 4, Figure 1,
Algorithm 1, the equations) on the same page as before; no undefined reference or citation; the only BibTeX
warning is the old empty year in `fema_sg_unit3`; one overfull box, the pre-existing one above. Victor's notes say
the main text ends on page 7 and Table tab:models is on page 10; the floats did not move.

**Again on 2026-11-07, with the 8-region table:** 10 pages; every labeled float on the same page as before; the
references still start on page 7 and Table tab:models is on page 10; no undefined reference or citation; the same
BibTeX warning and the same overfull box. One new warning, which the test of 2026-11-06 also had and I did not
report: the table is two rows taller, so the right column of page 5 no longer has room for the heading 4.4 Results
with the two lines that must follow it. LaTeX moves the heading to page 6 and stretches the spacing in that column
(`Underfull \vbox (badness 10000) has occurred while \output is active`), which shows as larger gaps before "Since
matched points" and before "Describing the regions". About three fewer lines of text on page 5 would bring the
heading back. I changed nothing for it, since it is not a number, and your Overleaf text may already differ.

---

## 6. Timings

- **Cell:** all 369 settings fitted in 20 s (22.5 s wall clock, every core, one BLAS thread per worker). Audit: 4 s.
- **Shift chain** (`o1_shift`, `o1_table`, `o1_tile_null`): 8 min 21 s.
- **Searches:** 21:50:52 to 23:32:11, **1 h 41 min** wall clock for the 81 searches, 8 at a time (2026-10-25:
  1 h 32 min). By phase: classic and Laplacian grid 40 min 56 s; GTV grid 25 min 20 s; shuffled controls 6 min 54 s;
  GTV's shuffled controls 4 min 46 s; family splits 9 min 37 s; μ = 1 8 min 43 s; μ = 10 5 min 3 s. The first phase
  overlapped my private-test runs (load average 31 to 38 on 14 cores until about 22:10), so it ran slower than it
  would alone; the results are seeded and do not depend on load. Summed single-search time of the classic and
  Laplacian grid: 312 minutes (196 on 2026-10-25). At S = 500 one search took 6.7 min (OODSelect), 8.2 min
  (Laplacian) and 15.7 min (GTV).
- **Downstream chain** (audits, regions, memorization check, region analysis, Figure 1): 20 s.

---

## 7. Tests and goldens

- **Owner check scripts** (`aotil/test_*.py`, `logs/2026-11-06/F_owner_*.log`): 8 pass (`test_zoo`, `test_oodselect_paper`,
  `test_region_match`, `test_o1_shift`, `test_regions_land`, `test_o5_lofo`, `test_o5_polygons`, `test_boundary_stat`).
  6 fail, the same 6 as on 2026-10-25 and for the same reason: their real-data checks read the 2026-10-16/17 outputs
  made on the v2 data ("dataset lon differs from the cell's", region land counts, pair lists). The new checks inside
  `test_o5_feature_inclusion` and `test_o5_pooling` pass.
- **The private test suite** (`logs/2026-11-06/F_tests_all.log`): **1,209 tests, 2 failures and 1 error**, all
  known and unrelated (a documented call in `docs/TASK-REBUILD-V3.md`, and two real-data pins of the v2 era). Before
  any change today it was 1,208 tests with 3 failures and 1 error (`logs/2026-11-06/A_tests_baseline.log`, the same
  as on 2026-10-25): the zoo's tie-rule bug test now passes on the grid cell, and the new test is the drop-rule pin.
- **Zoo goldens that changed** (11; `dataset.parquet` and `cell_data.*` did not):

| Golden | Why it changed |
|---|---|
| `specs.json` | `zoo_specs` is now the 369 grid settings (`grid_idx`, `kind`, `kw`, `seed`) for seeds 0 to 2, and 321 without xgboost; before, 300 round-robin specs with `feat_frac` and `row_frac` |
| `pure.json` | `build`: lr no longer fixes `max_iter=400` (the pinned lr now shows the library default, 100); three new cases (rf, lr and nb with grid settings; nb now takes its setting) |
| `fit_one.json`, `fit_one.npz` | `fit_one` has no feature or row draw (all varying columns, all rows) and returns `n_flood_id`; `model_row` writes `idx`, `grid_idx` and `flood_share_id` instead of `feat_frac` and `row_frac`; the pinned specs are each family's first grid setting and hand-written grid settings |
| `run_cell.json`, `run_cell.npz` | `run_cell` fits the whole grid (no model count), applies the drop rule, and returns the dropped table and the fit time |
| `main.json`, `main_basic.npz`, `main_extra.npz` | `zoo.main` writes the grid population and `<tag>_dropped.csv`; `--n-models` is gone |
| `audit.json` | `audit_cell`'s model checks follow the grid (kept + dropped = grid, the drop rule, refits of the dropped settings); `--expect-models` is gone; four new damaged-cell cases (a kept degenerate model, no dropped table, a missing setting, a changed seed) |
| `parser.json` | `zoo.py` lost `--n-models`, `audit_cell.py` lost `--expect-models` |

- **Goldens outside the zoo**: 4 files changed, each value by value only as allowed (§1); every other golden file
  is byte-identical (`logs/2026-11-06/A_compare_goldens.log`). The private test of the o1 check scripts has one
  inline count changed with them (13 → 19 passing checks of `test_o5_pooling.py`).

---

## 8. Notes and deviations

- **Order of the parts.** Part B's checks passed before any search started. To save wall clock, the searches ran
  while I updated the private pins (the last item of Part A); `zoo.py` did not change after the cell was fitted, and
  the audit and `aotil/test_zoo.py` pass on the cell as it stands. The paper was edited only after C6 (from the
  region analysis) and the μ rule (from the μ = 1 run) had both passed.
- **Early runs.** The regions, the smoothing audit and the region analysis ran once early (to check C6), then again
  in `logs/2026-11-06/D_chain.sh`; the outputs are deterministic, and the region-analysis table is identical line by
  line.
- **Soft totals** (for the parked Limitations caveat in `docs/TODO-CLAUDE-CODE.md`). The main runs at S = 250, 500
  and 1,000 end with Σs within 0.4 points of S. 11 of the 81 runs end more than 6% from S: in the grids, OODSelect
  S = 10 (3.0), the Laplacian at S = 10, 20 and 750 (61.5, 22.0, 695) and GTV at S = 20 (13.0); μ = 10 at S = 250
  (214); and 5 shuffled controls (up to 7,728 at S = 500). As on 2026-10-24/25, their chosen trials have a tiny λ or
  stopped before the budget term pulled the total to S; the subset is the top S of θ in every run.
- **The memorization check** exits with code 1 after writing its CSV: the known one-seed printout issue.
- **Model-set labels.** `o5_interpolation.py` still names its sets `all300` and `test60`, now meaning all models
  (361) and the test models (73); `logs/2026-11-06/E_paper_numbers.py` reads them by those names.
- **Figure 1** (`fig_aotl`, r = +0.81 and −0.90 on the 73 test models) is regenerated in
  `results/2026-11-06/paper_figures/` but not copied into the paper, which no longer includes it.
- **Two stale private pins were not regenerated.** The `real/` pins of two O5 test files rewrote themselves during
  a regeneration (their inputs changed on 2026-10-23, so they had been skipped); I restored them from the backup,
  since they are unrelated to this change.
- **No agent fan-out.** Ultracode was on, but your standing instruction (2026-10-23) is to avoid agent fan-outs
  even then, so every check here ran as a script in this session.

---

## 9. Docs updated

- `aotil/README.md`: the `zoo.py` and `audit_cell.py` rows describe the grid population and its audit; run order
  item 6 points at this run.
- `docs/DECISION-PROVENANCE.md`: the zoo rows for the population size, the grids, the order (replacing the round
  robin), the feature share and the row share (now all features and all rows), the seeds, the undertrained settings,
  the fixed settings (lr's `max_iter`), unusable settings, the training-fit counts, the cell contents and the pooling
  experiment are rewritten with "Your decision" and this brief as the source; two new rows record the degenerate
  drop rule and keeping duplicates; the audit rows (hard checks, v1 cells) follow; the overview counts are updated
  (462 rows).
- `docs/TODO-CLAUDE-CODE.md`: item 5 moved to Done with this report; a dated note on the soft totals added to the
  parked Limitations caveat; on 2026-11-07, a line for the 8-region Table tab:regions.

---

## Appendix: every paper edit, before → after

Generated by `logs/2026-11-06/F_paper_edits.py` from the backup and the edited file: one block per edited line (each paragraph, caption and table row of the source is one line), with the words that changed. Paste the After text over the same line in Overleaf. The flagged sentences of §5.2 are inside some of these paragraphs and are unchanged there. Edit 13 is one block for the rows of Table tab:regions (2026-11-07), because rows were added and removed: paste its After over the lines of its Before, from the Levy row to the All regions row.

#### Edit 1, near: Introduction

Changed: `\textbf{XX}` -> `361`; `0.050` -> `0.046`; `0.85.` -> `0.81.`; `275` -> `291`; `31` -> `71`; `0.065` -> `0.077`; `66\%` -> `65\%`

Before:

```latex
We train \textbf{XX} tabular models on Zone AE across Florida and evaluate on Zone A. First, we found that the shift exists and models lose up to 0.050 accuracy when evaluated on Zone A instead of Zone AE. Second, accuracy-on-the-line still holds, with a correlation of 0.85. Third, OODSelect finds a subset of Zone A where the correlation inverts to $-0.98$, but it is split into 275 pieces. Fourth, our spatially regularized loss reduces these to 31 pieces at a cost of 0.065 in correlation. Lastly, the regions' inputs are familiar, yet their labels disagree with their matches 66\% of the time.
```

After:

```latex
We train 361 tabular models on Zone AE across Florida and evaluate on Zone A. First, we found that the shift exists and models lose up to 0.046 accuracy when evaluated on Zone A instead of Zone AE. Second, accuracy-on-the-line still holds, with a correlation of 0.81. Third, OODSelect finds a subset of Zone A where the correlation inverts to $-0.98$, but it is split into 291 pieces. Fourth, our spatially regularized loss reduces these to 71 pieces at a cost of 0.077 in correlation. Lastly, the regions' inputs are familiar, yet their labels disagree with their matches 65\% of the time.
```

#### Edit 2, near: Models.

Changed: `$N=\textbf{XX}$` -> `$N=361$`; `\textbf{XX}` -> `216`; `\textbf{XX}` -> `72`; `\textbf{XX}` -> `73`

Before:

```latex
\paragraph{Models.} The $N=\textbf{XX}$ models come from eight families: decision trees, random forests, extra trees, histogram gradient boosting, XGBoost, logistic regression, small multilayer perceptrons, and Gaussian naive Bayes \citep{pedregosa2011sklearn,chen2016xgboost}. Each model is one hyperparameter setting of its family, and all models train on all features and all training points. The settings include deliberately weak ones, such as a few training iterations or heavy smoothing. These spread ID accuracy, which a correlation across models needs (Table~\ref{tab:models}). We drop models that predict one class for more than 95\% of the ID test points, since their accuracy on a subset only reflects its share of positives. Following \citet{salaudeen2025oodselect}, we split the remaining models at random 60/20/20 into \textbf{XX} selection, \textbf{XX} validation, and \textbf{XX} test models.
```

After:

```latex
\paragraph{Models.} The $N=361$ models come from eight families: decision trees, random forests, extra trees, histogram gradient boosting, XGBoost, logistic regression, small multilayer perceptrons, and Gaussian naive Bayes \citep{pedregosa2011sklearn,chen2016xgboost}. Each model is one hyperparameter setting of its family, and all models train on all features and all training points. The settings include deliberately weak ones, such as a few training iterations or heavy smoothing. These spread ID accuracy, which a correlation across models needs (Table~\ref{tab:models}). We drop models that predict one class for more than 95\% of the ID test points, since their accuracy on a subset only reflects its share of positives. Following \citet{salaudeen2025oodselect}, we split the remaining models at random 60/20/20 into 216 selection, 72 validation, and 73 test models.
```

#### Edit 3, near: fig:maps

Changed: `\includegraphics[width=\textwidth]{figures/fig_maps_v4.pdf}` -> `\includegraphics[width=\textwidth]{figures/fig_maps_v5.pdf}`

Before:

```latex
\includegraphics[width=\textwidth]{figures/fig_maps_v4.pdf}
```

After:

```latex
\includegraphics[width=\textwidth]{figures/fig_maps_v5.pdf}
```

#### Edit 4, near: fig:maps

Changed: `25` -> `27`; `275` -> `291`; `31` -> `71`; `15` -> `20`; `6` -> `8`

Before:

```latex
\caption{Selected points and regions at $S=500$. Gray dots are Zone A test points, and squares are the 28 test tiles of 50\,km. (a) OODSelect spreads its points over 25 test tiles in 275 pieces. (b) Our search groups them into 31 pieces in 15 tiles. (c) The 6 regions of at least 20 points, colored by whether most of their points are positives or negatives.}
```

After:

```latex
\caption{Selected points and regions at $S=500$. Gray dots are Zone A test points, and squares are the 28 test tiles of 50\,km. (a) OODSelect spreads its points over 27 test tiles in 291 pieces. (b) Our search groups them into 71 pieces in 20 tiles. (c) The 8 regions of at least 20 points, colored by whether most of their points are positives or negatives.}
```

#### Edit 5, near: tab:search

Changed: `$+$.85 [.76, .91]` -> `$+$.81 [.71, .87]`

Before:

```latex
All Zone A & $+$.85 [.76, .91] & -- & 28 \\
```

After:

```latex
All Zone A & $+$.81 [.71, .87] & -- & 28 \\
```

#### Edit 6, near: tab:search

Changed: `$+$.84` -> `$+$.79`

Before:

```latex
Random & $+$.84 & -- & -- \\
```

After:

```latex
Random & $+$.79 & -- & -- \\
```

#### Edit 7, near: tab:search

Changed: `$-$.68` -> `$-$.27`

Before:

```latex
Hardest & $-$.68 & -- & -- \\
```

After:

```latex
Hardest & $-$.27 & -- & -- \\
```

#### Edit 8, near: tab:search

Changed: `$-$.97]` -> `$-$.96]`; `275` -> `291`; `25` -> `27`

Before:

```latex
OODSelect & $-$.98 [$-$.99, $-$.97] & 275 & 25 \\
```

After:

```latex
OODSelect & $-$.98 [$-$.99, $-$.96] & 291 & 27 \\
```

#### Edit 9, near: tab:search

Changed: `$-$.92 [$-$.95, $-$.86]` -> `$-$.90 [$-$.94, $-$.84]`; `31` -> `71`; `15` -> `20`

Before:

```latex
Laplacian (ours) & $-$.92 [$-$.95, $-$.86] & 31 & 15 \\
```

After:

```latex
Laplacian (ours) & $-$.90 [$-$.94, $-$.84] & 71 & 20 \\
```

#### Edit 10, near: tab:search

Changed: `$-$.88 [$-$.92, $-$.80]` -> `$-$.86 [$-$.91, $-$.78]`; `122` -> `15`; `24` -> `11`

Before:

```latex
GTV (ours) & $-$.88 [$-$.92, $-$.80] & 122 & 24 \\
```

After:

```latex
GTV (ours) & $-$.86 [$-$.91, $-$.78] & 15 & 11 \\
```

#### Edit 11, near: tab:search

Changed: `$-$.15 [$-$.39, .11]` -> `$-$.52 [$-$.67, $-$.33]`; `18` -> `24`

Before:

```latex
Smoothing & $-$.15 [$-$.39, .11] & 51 & 18 \\
```

After:

```latex
Smoothing & $-$.52 [$-$.67, $-$.33] & 51 & 24 \\
```

#### Edit 12, near: tab:search

Changed: `\textbf{XX}` -> `73`

Before:

```latex
\caption{Subset search at $S=500$. $\text{corr}_\text{te}$ uses the \textbf{XX} test models, and a more negative value is a stronger inversion. Pieces are the connected components of the subset in $\mathbf{A}$, and tiles are the test tiles they touch. Random is the mean of 100 draws. The hardest points are those the test models get wrong most often, so their $\text{corr}_\text{te}$ is only indicative. Smoothing averages OODSelect's final weights over each point's 10 nearest neighbors.}
```

After:

```latex
\caption{Subset search at $S=500$. $\text{corr}_\text{te}$ uses the 73 test models, and a more negative value is a stronger inversion. Pieces are the connected components of the subset in $\mathbf{A}$, and tiles are the test tiles they touch. Random is the mean of 100 draws. The hardest points are those the test models get wrong most often, so their $\text{corr}_\text{te}$ is only indicative. Smoothing averages OODSelect's final weights over each point's 10 nearest neighbors.}
```

#### Edit 13 (table rows), near: tab:regions

Changed: the rows Levy, Dixie/Levy, Liberty, Hamilton, Palm Beach, Martin, All regions -> Dixie, Highlands, Levy, Liberty, Hamilton, Manatee/Sarasota, Hendry/Palm Beach, Hendry, All regions

Before:

```latex
Levy & 114 & 1.00 / .35 & .00 & .65 \\
Dixie/Levy & 41 & .93 / .32 & .00 & .70 \\
\multicolumn{5}{@{}l}{\textit{Negative regions}} \\
Liberty & 123 & .06 / .72 & .00 & .70 \\
Hamilton & 54 & .35 / .60 & .09 & .67 \\
Palm Beach & 26 & .00 / .45 & .00 & .45 \\
Martin & 23 & .00 / .74 & .00 & .74 \\
\midrule
All regions & 381 & .47 / .53 & .01 & .66 \\
```

After:

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
```

#### Edit 14, near: Pooling hides the shift.

Changed: `$-0.007$ [$-0.060$, $0.045$].` -> `$-0.008$ [$-0.065$, $0.048$].`; `0.52` -> `0.51`; `0.050 [0.026, 0.079]` -> `0.046 [0.021, 0.074]`; `0.060` -> `0.043`; `0.104` -> `0.093`; `92\%` -> `94\%`; `0.019` -> `0.023`; `0.036` -> `0.035`; `0.043.` -> `0.041.`; `$\text{corr}_\text{te}=+0.85$.` -> `$\text{corr}_\text{te}=+0.81$.`

Before:

```latex
\paragraph{Pooling hides the shift.} Pooled over the test tiles, models trained on Zone AE are as accurate on Zone A as on held-out Zone AE, with a gap of $-0.007$ [$-0.060$, $0.045$]. However, the two test sets weight the tiles differently. For example, the tile around Palm Beach holds 1,264 Zone AE points but only 192 Zone A points, and the models average 0.52 accuracy there. Within a tile, models are less accurate on Zone A in 22 of the 28 tiles. Averaged over tiles, the gap is 0.050 [0.026, 0.079] in accuracy, 0.060 in AUROC, and 0.104 in F1, and 92\% of models do worse on Zone A. When we shuffle which test points belong to each zone within each tile and label, the accuracy and AUROC gaps fall to within 0.019 of zero, but the F1 gap stays at 0.036 to 0.043. Accuracy-on-the-line still holds on all of Zone A, with $\text{corr}_\text{te}=+0.85$.
```

After:

```latex
\paragraph{Pooling hides the shift.} Pooled over the test tiles, models trained on Zone AE are as accurate on Zone A as on held-out Zone AE, with a gap of $-0.008$ [$-0.065$, $0.048$]. However, the two test sets weight the tiles differently. For example, the tile around Palm Beach holds 1,264 Zone AE points but only 192 Zone A points, and the models average 0.51 accuracy there. Within a tile, models are less accurate on Zone A in 22 of the 28 tiles. Averaged over tiles, the gap is 0.046 [0.021, 0.074] in accuracy, 0.043 in AUROC, and 0.093 in F1, and 94\% of models do worse on Zone A. When we shuffle which test points belong to each zone within each tile and label, the accuracy and AUROC gaps fall to within 0.023 of zero, but the F1 gap stays at 0.035 to 0.041. Accuracy-on-the-line still holds on all of Zone A, with $\text{corr}_\text{te}=+0.81$.
```

#### Edit 15, near: OODSelect inverts, but scatters.

Changed: `($+0.84$),` -> `($+0.79$),`; `($-0.68$).` -> `($-0.27$).`; `275` -> `291`; `25` -> `27`

Before:

```latex
\paragraph{OODSelect inverts, but scatters.} Random subsets of 500 points follow the aggregate trend ($+0.84$), so small subsets do not invert by chance (Table~\ref{tab:search}). OODSelect finds an inversion of $-0.98$ on test models it never saw, much stronger than on the 500 points the test models get wrong most often ($-0.68$). However, its subset is split into 275 pieces across 25 of the 28 test tiles (Figure~\ref{fig:maps}a), so it cannot be studied as a set of places.
```

After:

```latex
\paragraph{OODSelect inverts, but scatters.} Random subsets of 500 points follow the aggregate trend ($+0.79$), so small subsets do not invert by chance (Table~\ref{tab:search}). OODSelect finds an inversion of $-0.98$ on test models it never saw, much stronger than on the 500 points the test models get wrong most often ($-0.27$). However, its subset is split into 291 pieces across 27 of the 28 test tiles (Figure~\ref{fig:maps}a), so it cannot be studied as a set of places.
```

#### Edit 16, near: Our loss turns the subset into regions.

Changed: `$-0.92$ [$-0.95$, $-0.86$]` -> `$-0.90$ [$-0.94$, $-0.84$]`; `31` -> `71`; `15` -> `20`; `Six` -> `Eight`; `381` -> `343`; `0.065` -> `0.077`; `($-0.88$)` -> `($-0.86$)`; `122` -> `15`

Before:

```latex
\paragraph{Our loss turns the subset into regions.} With the Laplacian loss, the search keeps an inversion of $-0.92$ [$-0.95$, $-0.86$] in 31 pieces over 15 tiles (Figure~\ref{fig:maps}b). Six pieces of at least 20 points hold 381 of the 500 points, and these are the regions we analyze (Figure~\ref{fig:maps}c). The cost of coherence is 0.065 in correlation. Smoothing OODSelect's final weights over each point's neighbors also gives a coherent subset (51 pieces), but it loses the inversion ($-0.15$), so coherence must be part of the objective. GTV inverts less ($-0.88$) and leaves 122 pieces at this budget.
```

After:

```latex
\paragraph{Our loss turns the subset into regions.} With the Laplacian loss, the search keeps an inversion of $-0.90$ [$-0.94$, $-0.84$] in 71 pieces over 20 tiles (Figure~\ref{fig:maps}b). Eight pieces of at least 20 points hold 343 of the 500 points, and these are the regions we analyze (Figure~\ref{fig:maps}c). The cost of coherence is 0.077 in correlation. Smoothing OODSelect's final weights over each point's neighbors also gives a coherent subset (51 pieces), but it loses the inversion ($-0.15$), so coherence must be part of the objective. GTV inverts less ($-0.86$) and leaves 15 pieces at this budget.
```

#### Edit 17, near: Controls.

Changed: `$-0.15$` -> `$+0.49$`; `$-0.54$` -> `$-0.66$`; `8` -> `9`; `$-0.67$,` -> `$-0.88$,`; `9` -> `6`; `[$-0.95$,` -> `[$-0.94$,`; `four` -> `three`; `$-0.70$` -> `$-0.81$`; `$-0.97$.` -> `$-0.98$.`; `$-0.71$,` -> `$-0.76$,`; `$-0.93$.` -> `$-0.56$.`

Before:

```latex
\paragraph{Controls.} Three controls support the inversion. First, we rerun each search three times at budgets of 250, 500, and 1,000 points with the ID accuracies of the selection and validation models shuffled. The median correlation of these nine runs is $-0.15$ for OODSelect and $-0.54$ for the Laplacian, and the real Laplacian search beats 8 of its 9. Some shuffled runs still reach $-0.67$, so a negative correlation alone is weak evidence. Second, noisy labels invert the trend only when models interpolate them \citep{sanyal2025}. Without the 9 test models that reach 99\% training accuracy, $\text{corr}_\text{te}$ is $-0.91$ [$-0.95$, $-0.85$], and the trend also inverts within each of the four families that never interpolate, from $-0.70$ to $-0.97$. Third, selecting with the tree-based families and testing on the others gives $-0.71$, and the reverse gives $-0.93$.
```

After:

```latex
\paragraph{Controls.} Three controls support the inversion. First, we rerun each search three times at budgets of 250, 500, and 1,000 points with the ID accuracies of the selection and validation models shuffled. The median correlation of these nine runs is $+0.49$ for OODSelect and $-0.66$ for the Laplacian, and the real Laplacian search beats 9 of its 9. Some shuffled runs still reach $-0.88$, so a negative correlation alone is weak evidence. Second, noisy labels invert the trend only when models interpolate them \citep{sanyal2025}. Without the 6 test models that reach 99\% training accuracy, $\text{corr}_\text{te}$ is $-0.91$ [$-0.94$, $-0.85$], and the trend also inverts within each of the three families that never interpolate, from $-0.81$ to $-0.98$. Third, selecting with the tree-based families and testing on the others gives $-0.76$, and the reverse gives $-0.56$.
```

#### Edit 18, near: Same inputs, different labels.

Changed: `$\eta=0.66$,` -> `$\eta=0.65$,`; `two` -> `three`; `98\%` -> `84\%`; `34\%` -> `35\%`; `four` -> `five`; `12\%` -> `19\%`; `66\%` -> `64\%`; `9\%` -> `11\%`

Before:

```latex
\paragraph{Same inputs, different labels.} The regions are not unfamiliar land (Table~\ref{tab:regions}). Pooled, 1.3\% of their points have no match, against 5\% of Zone AE points, and no region exceeds 10\%. Their labels, however, differ. The pooled label disagreement is $\eta=0.66$, against 0.30 among Zone AE points, and every region lies above this baseline. Other Zone A points disagree less ($\eta=0.25$), so the disagreement is local to the regions. It runs in both directions. In the two positive regions, 98\% of the points are positives against 34\% of their matches, so Zone A marks floodplain where Zone AE would not. In the four negative regions, 12\% are positives against 66\% of their matches, so Zone A leaves out land that Zone AE would map. Only Hamilton, with 9\% unmatched points, may partly reflect covariate shift.
```

After:

```latex
\paragraph{Same inputs, different labels.} The regions are not unfamiliar land (Table~\ref{tab:regions}). Pooled, 1.3\% of their points have no match, against 5\% of Zone AE points, and no region exceeds 10\%. Their labels, however, differ. The pooled label disagreement is $\eta=0.65$, against 0.30 among Zone AE points, and every region lies above this baseline. Other Zone A points disagree less ($\eta=0.25$), so the disagreement is local to the regions. It runs in both directions. In the three positive regions, 84\% of the points are positives against 35\% of their matches, so Zone A marks floodplain where Zone AE would not. In the five negative regions, 19\% are positives against 64\% of their matches, so Zone A leaves out land that Zone AE would map. Only Hamilton, with 11\% unmatched points, may partly reflect covariate shift.
```

#### Edit 19, near: Model Settings

Changed: `\textbf{XX}` -> `8`; `$N=\textbf{XX}$` -> `$N=361$`

Before:

```latex
Table~\ref{tab:models} lists the hyperparameter grids. Each combination in a family's grid is one model with a fixed seed, and every model trains on all seven features and all 29,161 training points. The grids are our own. They run from deliberately weak settings, such as a few training iterations, one feature per split, or heavy variance smoothing, to typical ones. Random forests and extra trees still draw a random subset of features at each split, and random forests also resample the rows of each tree, as their algorithms do. Logistic regression and the multilayer perceptrons standardize their inputs. Settings not listed take the defaults of scikit-learn 1.9.1 and XGBoost 3.4.1. Of the 369 settings, \textbf{XX} predict one class for more than 95\% of the ID test points, which leaves $N=\textbf{XX}$ models.
```

After:

```latex
Table~\ref{tab:models} lists the hyperparameter grids. Each combination in a family's grid is one model with a fixed seed, and every model trains on all seven features and all 29,161 training points. The grids are our own. They run from deliberately weak settings, such as a few training iterations, one feature per split, or heavy variance smoothing, to typical ones. Random forests and extra trees still draw a random subset of features at each split, and random forests also resample the rows of each tree, as their algorithms do. Logistic regression and the multilayer perceptrons standardize their inputs. Settings not listed take the defaults of scikit-learn 1.9.1 and XGBoost 3.4.1. Of the 369 settings, 8 predict one class for more than 95\% of the ID test points, which leaves $N=361$ models.
```

#### Edit 20, near: tab:models

Changed: `\textbf{XX}` -> `40`

Before:

```latex
Decision tree & maximum depth $\in\{1,2,3,4,5,6,8,10,12,\text{none}\}$ $\times$ minimum leaf size $\in\{1,20,200,2000\}$ & 40 & \textbf{XX} \\
```

After:

```latex
Decision tree & maximum depth $\in\{1,2,3,4,5,6,8,10,12,\text{none}\}$ $\times$ minimum leaf size $\in\{1,20,200,2000\}$ & 40 & 40 \\
```

#### Edit 21, near: tab:models

Changed: `\textbf{XX}` -> `54`

Before:

```latex
Random forest & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,2,7\}$ & 54 & \textbf{XX} \\
```

After:

```latex
Random forest & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,2,7\}$ & 54 & 54 \\
```

#### Edit 22, near: tab:models

Changed: `\textbf{XX}` -> `54`

Before:

```latex
Extra trees & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,2,7\}$ & 54 & \textbf{XX} \\
```

After:

```latex
Extra trees & trees $\in\{10,30,100\}$ $\times$ maximum depth $\in\{1,2,3,5,8,\text{none}\}$ $\times$ features per split $\in\{1,2,7\}$ & 54 & 54 \\
```

#### Edit 23, near: tab:models

Changed: `\textbf{XX}` -> `45`

Before:

```latex
Histogram gradient boosting & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ iterations $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,\text{none}\}$, without early stopping & 48 & \textbf{XX} \\
```

After:

```latex
Histogram gradient boosting & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ iterations $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,\text{none}\}$, without early stopping & 48 & 45 \\
```

#### Edit 24, near: tab:models

Changed: `\textbf{XX}` -> `45`

Before:

```latex
XGBoost & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ trees $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,8\}$, histogram tree method & 48 & \textbf{XX} \\
```

After:

```latex
XGBoost & learning rate $\in\{0.01,0.03,0.1,0.5\}$ $\times$ trees $\in\{3,10,50,300\}$ $\times$ maximum depth $\in\{1,3,8\}$, histogram tree method & 48 & 45 \\
```

#### Edit 25, near: tab:models

Changed: `\textbf{XX}` -> `50`

Before:

```latex
Logistic regression & inverse regularization strength $C\in\{0.001,0.01,0.1,1,100\}$ $\times$ iterations $\in\{1,2,3,5,100\}$ $\times$ class weights $\in\{\text{none},\text{balanced}\}$ & 50 & \textbf{XX} \\
```

After:

```latex
Logistic regression & inverse regularization strength $C\in\{0.001,0.01,0.1,1,100\}$ $\times$ iterations $\in\{1,2,3,5,100\}$ $\times$ class weights $\in\{\text{none},\text{balanced}\}$ & 50 & 50 \\
```

#### Edit 26, near: tab:models

Changed: `\textbf{XX}` -> `43`

Before:

```latex
Multilayer perceptron & hidden layers $\in\{(2),(8),(16),(32),(64,32)\}$ $\times$ epochs $\in\{3,20,200\}$ $\times$ learning rate $\in\{10^{-4},10^{-3},10^{-2}\}$ & 45 & \textbf{XX} \\
```

After:

```latex
Multilayer perceptron & hidden layers $\in\{(2),(8),(16),(32),(64,32)\}$ $\times$ epochs $\in\{3,20,200\}$ $\times$ learning rate $\in\{10^{-4},10^{-3},10^{-2}\}$ & 45 & 43 \\
```

#### Edit 27, near: tab:models

Changed: `\textbf{XX}` -> `30`

Before:

```latex
Gaussian naive Bayes & variance smoothing $\in\{10^{-9},10^{-5},10^{-4},10^{-3},0.003,0.01,0.03,\allowbreak 0.1,0.2,0.3,0.5,1,2,3,10\}$ $\times$ class priors $\in\{\text{fitted},\text{equal}\}$ & 30 & \textbf{XX} \\
```

After:

```latex
Gaussian naive Bayes & variance smoothing $\in\{10^{-9},10^{-5},10^{-4},10^{-3},0.003,0.01,0.03,\allowbreak 0.1,0.2,0.3,0.5,1,2,3,10\}$ $\times$ class priors $\in\{\text{fitted},\text{equal}\}$ & 30 & 30 \\
```

#### Edit 28, near: tab:models

Changed: `\textbf{XX}` -> `361`

Before:

```latex
Total & & 369 & \textbf{XX} \\
```

After:

```latex
Total & & 369 & 361 \\
```
