# Rerun of everything with the final Approach, D = 2026-10-25

Victor's request of 2026-10-25: rerun every result in the paper from the model cell on, so that it
matches the final Approach (3.1 to 3.6 and Algorithm 1), add GTV and the new region analysis, and
update the paper. Brief: `docs/TASK-RERUN-ALL-2026-10-25.md`. D = 2026-10-25. Outputs are in
`results/2026-10-25/`, logs in `logs/2026-10-25/`, and the refit cell in
`data/interim/aotil/cells_rerun_2026-10-25/`. Nothing is committed. The Mac was on AC power with the lid
open for the whole run (no sleep gap in any log).

**In one paragraph.** The audit found no disagreement between the code and the Approach, so no code
was fixed. GTV (`--reg gtv`) and the caliper matching (`aotil/region_match.py`) are implemented and
tested. No Stop fired. **S1:** the refit cell is identical to `cells_v3`. **S2:** the sandbox numbers are
reproduced to 0.005. **C6a and C6b** both hold. All 81 searches ran: the 60 that also ran on 2026-10-24
gave identical θ, bit for bit, and every downstream number that existed on 2026-10-24 is unchanged. So
**no paper number differs from the 2026-10-24 squared-penalty run** (§6). The paper now reports that
run, plus GTV and the region analysis, and no longer contains the softplus numbers or the twins.

| Claim | Result | Holds? |
|---|---|---|
| C1 shift | pooled accuracy gap −0.007 [−0.060, +0.045]; per tile +0.052 [+0.033, +0.072], 24 of 28 tiles, 97% of models | reported (as fixed) |
| C2 accuracy-on-the-line | r_te = +0.849 on all Zone A points | **holds** |
| C3 OODSelect inverts, scattered | −0.981; 275 pieces, 25 tiles, purity 0.131; below all 3 shuffled runs at S = 500 | **holds** |
| C4 regions at small cost | −0.916, purity 0.768, 31 pieces (≤ 55); purity above smoothing at S = 500 only | **does not hold as fixed** (purity at S = 250 and 1,000), as on 2026-10-24 |
| C5 not memorized noise | −0.910 [−0.948, −0.846] on the 51 held-out test models below 99% training accuracy | **holds** |
| C6a familiar inputs | pooled unmatched share 0.013 (≤ 0.10); no region above 0.10 (Hamilton 0.093) | **holds** |
| C6b different labels | pooled η 0.664 against the Zone AE baseline 0.297 (needs ≥ 0.397) | **holds** |
| C7 GTV (descriptive) | −0.877 [−0.925, −0.802]; purity 0.671; 122 pieces; 24 tiles; 7 regions holding 345 points | reported |

**Read first:** §8, the claims the paper now states differently. **Not in the paper, for Victor:** the
proposed ninth decision (§9), and a few notes (§11).

---

## Terms used here, in CS terms

- **Tile.** The map is cut into 50 km squares (`tile_50km`). Whole tiles go to training or testing,
  like grouped cross-validation where a group is a square of land. The 28 test tiles hold both test sets.
- **Zone A, Zone AE.** Two FEMA labeling procedures for the same label (the 1% annual chance
  floodplain). Zone AE is the detailed one; the models train on it. Zone A is the approximate one,
  the out-of-distribution (OOD) test set. ID = held-out Zone AE points in the same test tiles.
- **kNN graph.** Each Zone A test point is linked to its 10 nearest Zone A test points (and to every
  point that has it among its 10 nearest). Distances are straight lines on a flat map projection in
  metres (**EPSG:3086**, Florida Albers: a fixed formula from longitude/latitude to x/y metres).
- **Piece, region.** A piece is a connected component of the selected points in that graph. A region
  is a piece of at least c = 20 points.
- **Purity.** Of a selected point's 10 nearest neighbours, the share that is also selected, averaged
  over the selected points; 1 for one solid blob, S/m for random points.
- **Land cover (NLCD).** The National Land Cover Database class of the 30 m pixel under a point
  (e.g. 90 = woody wetlands, 42 = evergreen forest, 82 = cultivated crops). A categorical feature.
- **HAND.** Height above the nearest drainage: how many metres a point sits above the nearest stream
  cell of a river network. Low HAND means close to the water level of a stream.
- **Caliper matching.** Nearest-neighbour lookup with a maximum distance. Each region point looks for
  held-out Zone AE points with the same land-cover class whose 6 standardized numeric features are
  within a Euclidean distance ε (the caliper). A point with nothing inside ε is *unmatched*.
- **η (label disagreement).** For a matched point, the share of its matches whose label differs from
  its own; η of a set averages this over its matched points. Since a point and its matches have nearly
  the same inputs, a high η means the two procedures label the same inputs differently.
- **Covariate shift / concept shift.** A change in P(x) (the inputs themselves are unusual) versus a
  change in P(y | x) (the same inputs get different labels).
- **Halo.** Unselected Zone A test points within 1, 2 or 3 km of a region.

---

## 1. Part A: the code against the Approach

Every statement of Approach 3.1 to 3.6, Algorithm 1, Experimental Settings and Evaluation Metrics
that the code implements, with the code that implements it. Line numbers are for the code as it
stands after this run. **No statement disagrees, so Part A changed no code.** Four details agree
only up to a numerical or wording nuance; they are marked "agrees, note" and none changes a number
or a method choice.

| # | Statement (paper) | Code | Agrees? |
|---|---|---|---|
| 1 | 3.1: ŷ(x) = 1[f(u(x)) > 0.5] | `aotil/oodselect_paper.py:65-78` (`correctness`: a point is right iff p > 0.5 for a positive, p ≤ 0.5 for a negative, from the stored true-class probability); `aotil/o1_shift.py:216` (flood iff p > 0.5) | Agrees |
| 2 | 3.1: N models trained on the π_AE points of 𝒟_train; 𝒟_ID = π_AE points of 𝒟_test; 𝒟_OOD = π_A points of the same tiles | `aotil/zoo.py` (cell), checked by `aotil/audit_cell.py` (verdict PASS, `logs/2026-10-25/D1_audit_cell.log`: same 28 test tiles, both zones in each) | Agrees |
| 3 | 3.1 and Evaluation: the shift Δ is the mean over the N models of metric(ID) − metric(OOD) | `aotil/o1_shift.py:564-587` (`per_model`, `summary`) | Agrees |
| 4 | 3.2: Z_ij = 1[ŷ_i(x_j) = y_j]; OOD accuracy (Z s)_i / ‖s‖₁ | `aotil/oodselect_paper.py:77`; `:195-208` (`pearson_r_loss`: `X @ z / (z.sum() + 1e-6)`) | Agrees, note: the released code adds 1e-6 to ‖s‖₁ |
| 5 | 3.2: corr is the Pearson correlation of probits, accuracies clipped to [10⁻⁶, 1 − 10⁻⁶] | `aotil/oodselect_paper.py:188-208` (`probit_t` clamps to [1e-6, 1 − 1e-6]; population Pearson) | Agrees, note: as in the released code, 1e-6 is added to each variance; this moves r by under 4e-5 at S = 500 (DECISION-PROVENANCE) |
| 6 | 3.2: s_j = σ(θ_j); Algorithm 1 line 4: θ ~ 𝒩(0, I), per trial | `aotil/oodselect_paper.py:359-360` (`torch.randn` from a generator seeded per trial), `:373` (`torch.sigmoid(theta)`) | Agrees |
| 7 | 3.2: three disjoint model groups (selection, validation, test), 180/60/60 | `aotil/oodselect_paper.py:96-106` (`model_split`, sklearn at `random_state=42`) | Agrees |
| 8 | 3.2: Ω_S(s) = (S − ‖s‖₁)², weight λ > 0 | `aotil/oodselect_paper.py:211-223` (`budget_term(..., "squared")`); every run of this rerun passes `--budget-penalty squared` (`run_oodselect_local.sh:77`) | Agrees |
| 9 | 3.2: the weight at step e is λ w_e, w_e rising from 0 to 1 along half a cosine wave over the first W steps | `aotil/oodselect_paper.py:380-382`: w_e = ½(1 − cos(π·min(e/W, 1))), e from 0 | Agrees |
| 10 | 3.2: full-batch Adam; learning rate on a cosine between an initial and a minimum value with half-period τ | `aotil/oodselect_paper.py:361-363` (`Adam(lr)`, `CosineAnnealingLR(T_max=τ, eta_min)`; one step per epoch on all points) | Agrees |
| 11 | 3.2: five hyperparameters h = (λ, W, two learning rates, τ); T trials; TPE proposes h_t from the earlier trials | `aotil/oodselect_paper.py:440-446` (the five `suggest_*`), `:469-471` (`TPESampler(seed)`, `n_trials`); `N_TRIALS = 100` (`:43`) | Agrees, note: Optuna's TPE draws its first 10 trials at random before it models the earlier ones (its default) |
| 12 | Settings: lr ∈ [10⁻², 10²], min lr ∈ [10⁻³, 1], λ ∈ [10⁻⁶, 10²], all log; τ ∈ {10..200}; W ∈ {10..1000} | `aotil/oodselect_paper.py:441-445`; `PENALTY_RANGE["squared"] = (1e-6, 1e2, log)` (`:55`) | Agrees |
| 13 | 3.2: 𝒬_S(s) = the S points with the largest weights | `aotil/oodselect_paper.py:226-232` (`binary_selection`: top S of θ), `:633` (stored mask: stable argsort of −θ) | Agrees, note: ranking θ instead of σ(θ) gives the same order (σ is increasing) without the ties of a saturated float32 sigmoid |
| 14 | 3.2 and Algorithm 1 line 9: corr_val(𝒬_S(σ(θ))) after every step | `aotil/oodselect_paper.py:389-390`, computed after `opt.step()` | Agrees |
| 15 | 3.2 and Algorithm 1 line 10: stop once corr_val has not improved for P steps; keep s from that last step | `aotil/oodselect_paper.py:391-398` (`wait` counts steps without a strictly lower value; stop at `wait >= P`), `:401` (θ at the stopping step); `PATIENCE = 500`, at most 50,000 steps (`:44-45`) | Agrees (it also stops on a non-finite corr_val, the released code's guard) |
| 16 | 3.2 and Algorithm 1 line 13: the trial with the lowest final corr_val gives ℳ | `aotil/oodselect_paper.py:467` (objective = the last validation r), `:472` (`study.best_trial`, minimised) | Agrees |
| 17 | 3.3 Eq. spatial and Algorithm 1 line 7: ℓ = corr_sel(s) + λ w_e Ω_S(s) + μ R(s; 𝒱); μ = 0 is OODSelect | `aotil/oodselect_paper.py:374-385` | Agrees |
| 18 | 3.3: μ fixed, not tuned; R only in the training loss | `--mu` is a fixed argument (3 in `run_oodselect_local.sh`); validation and test use `pearson_r_loss` alone (`:390`, `hard_corr`) | Agrees |
| 19 | 3.3: the graph joins two points when either is among the other's k nearest, weight 1, on 𝒱 | `aotil/graph.py:80-118` (`build_adjacency`: OR-symmetrised directed kNN, 0/1, self excluded) | Agrees |
| 20 | Settings: coordinates projected to metres | `aotil/graph.py:27-48` (EPSG:3086, Florida Albers) | Agrees |
| 21 | 3.4: R = sᵀ L s / (S k), L = D − A, sum over edges of (s_j − s_j′)² | `aotil/graph.py:126-129` (`laplacian`); `aotil/oodselect_paper.py:244-276` (`_Quadratic`: sᵀLs, gradient 2Ls), `:366` (S·k) | Agrees |
| 22 | 3.4: every point has at least k edges | min degree 10, mean 12.08 on this cell (`graph.build_adjacency`, k = 10) | Agrees |
| 23 | 3.5: R_GTV = Σ_edges \|s_j − s_j′\| / (S k), same graph; subgradient where neighbours tie | `aotil/oodselect_paper.py:280-318` (`_TotalVariation`, `spatial_term`), new in Part B | Agrees (new) |
| 24 | 3.3 and Algorithm 1 line 14: regions = connected components of the subgraph ℳ induces, ≥ c points | `aotil/regions.py:308-321` (`components`), `:395` (`ge_min_size`), `--min-size 20` (`:523`) | Agrees |
| 25 | 3.6: z(x) standardized by the mean and SD over 𝒟_OOD; q = 6 | `aotil/region_match.py:107-121`, `:330-345` (new in Part C) | Agrees (new) |
| 26 | 3.6: 𝒫_𝒞 = π_AE points of 𝒟_ID in test tiles with no point of 𝒞 | `aotil/region_match.py:382-388` (`region_exclusion`, `region-tiles`) | Agrees (new) |
| 27 | 3.6: matches with the same land-cover class within ε, at most the 5 closest; matched iff 𝒞′_x ≠ ∅ | `aotil/region_match.py:163-204` (`match`) | Agrees (new) |
| 28 | 3.6: ε set so that 5% of π_AE points are unmatched when matched to other π_AE points the same way | `aotil/region_match.py:214-227` (`caliper`), `:470-472` | Agrees (new) |
| 29 | 3.6 Eq. disagree: η(𝒞) = mean over matched x of the share of x's matches with a different label; the baseline is π_AE against π_AE | `aotil/region_match.py:230-243`; `:474-476` | Agrees (new) |
| 30 | Evaluation: Δ_tile = mean over models and test tiles of the gap within the tile; ID positives weighted ȳ_b^OOD/ȳ_b^ID, ID negatives (1 − ȳ_b^OOD)/(1 − ȳ_b^ID); AUROC unweighted on tiles where both sets hold both classes | `aotil/o1_shift.py:378-424` (`tile_gaps`; weights at `:404-405`), `:440-471` (`tile_summary`) | Agrees: 28 tiles for accuracy and F1, 27 for AUROC |
| 31 | Evaluation: intervals from 2,000 bootstrap draws of whole test tiles; for Δ_tile each draw averages the drawn tiles' gaps | `aotil/o1_shift.py:55` (`N_BOOT`), `:347-352` (`tile_draws`), `:463` | Agrees |
| 32 | Evaluation: corr_te on the 60 test models with a 95% interval from Fisher's transformation; inverted when ≤ −0.3 | `aotil/oodselect_paper.py:477-484` (`fisher_ci`: tanh(arctanh r ± 1.96/√(n − 3))), `AOTIL_THRESHOLD = -0.3` | Agrees |
| 33 | Evaluation Eq. purity: mean over selected points of \|𝒦_j ∩ ℳ\| / k, with 𝒦_j the k nearest | `aotil/graph.py:169-172` (rows of the directed kNN graph) | Agrees: every point has exactly 10 directed neighbours on this cell, so the code's denominator (neighbours found) is k |
| 34 | Evaluation: pieces and the test tiles they touch | `aotil/graph.py:182-186` (`n_cc`, `tiles`) | Agrees |
| 35 | Subset searches: random = mean of 100 draws; hardest = the points the test models get wrong most often | `aotil/report_paper.py:31; 202-226` (100 random subsets; hardest ranked by the test models' error share, averaged over 100 random orders of the ties at the cut) | Agrees |
| 36 | Subset searches: smoothing averages OODSelect's final weights over each point's 10 nearest neighbours and takes the S largest | `aotil/audit_selection.py:106-136` (`carry_field`, `smoothed_classic`) | Agrees, note: the 10 nearest include the point itself (itself and its 9 nearest others). A wording nuance in Experiments, not in the Approach; left as is |
| 37 | Controls: shuffled = the selection and validation models' ID accuracies permuted; family-disjoint = trees vs the others; memorization = models with ≥ 99% accuracy on their own training rows removed | `aotil/oodselect_paper.py:137-147` (`null_accuracies`), `:109-123` (`family_split`; 25% of the selection family held out for validation, which the paper does not mention); `aotil/o5_interpolation.py:38; 64-68` | Agrees |

Bug fixes: none were needed. Method choices: none were changed (no Stop).

---

## 2. Code changes and tests

Nothing is committed. Backups of the files before the change: `Claude outputs/goldens-before-2026-10-25/`
(oodselect_paper.py, run_oodselect_local.sh, the private oodselect goldens and their test file) and
`Claude outputs/paper-before-2026-10-25-rerun/` (the paper and its bib).

| File | Change |
|---|---|
| `aotil/oodselect_paper.py` | **Part B, GTV.** `--reg {laplacian,gtv}` (default `laplacian`). `_TotalVariation` takes the edges from the strict upper triangle of L (each undirected edge once, as sᵀLs counts it; a weighted graph raises) and returns Σ\|s_j − s_j′\| in float64 through `torch.abs`, whose subgradient at a tie is 0. `spatial_term(L, reg)` picks the loss; `train_one` is unchanged (it divides the term by S·k, so `P_soft` is R_GTV under GTV). `variant_name` gives `spatial_gtv_mu<μ>_knn<k>` (μ = 0 stays `classic`). `main` writes `reg` into `result.csv` and `meta.json` and refuses a GTV run whose `--out` folder is not its variant name, so it cannot be read as the Laplacian run. |
| `run_oodselect_local.sh` | `REG` (default `laplacian`) is passed as `--reg` to the spatial variant; the classic variant always runs with the default. The task lines carry the reg as a seventh word; `variant_dir` names the GTV folders. |
| `aotil/region_match.py` (new) | **Part C**, Approach 3.6. See its docstring and §4. `--exclude own-tile` reproduces the sandbox rule for S2. |
| `aotil/test_oodselect_paper.py` | Check 7 (GTV): on a 0/1 s, GTV = Laplacian = the cut-edge count of `graph.coherence_metrics` (S = 50 and 200 on the synthetic cell); the gradient against central differences where no two neighbours tie; a tied edge adds 0 (path 0–1–2 with s = (0.5, 0.5, 0.2) gives gradient (0, 1, −1)); a weighted graph raises; μ = 0 with `--reg gtv` gives the classic run's θ and every trial record exactly; the default is the Laplacian (θ bit-identical); μ = 3 with GTV changes the run; `P_soft` is R_GTV of the final weights; variant names; `main` writes `reg` and refuses a GTV run in the Laplacian folder. |
| `aotil/test_region_match.py` (new) | The brief's 8 cases, helper checks, an end-to-end run on a synthetic cell, and the real S2 check (§3). |
| `aotil/README.md` | `region_match.py` row, the `--reg` flag, `BUDGET_PENALTY` and `REG` knobs, the squared penalty as the paper's budget term, the twins scripts marked superseded, run order item 5. |
| `docs/DECISION-PROVENANCE.md` | Rows for the squared penalty in the paper and GTV (μ = 3, not tuned); a new `aotil/region_match.py` table (11 rows: features and scale, pool, region-tile exclusion, the 5% caliper after Stuart 2010, at most 5 matches, η, baseline, the KD-tree query rule, halos, the hardest-points tie order, the descriptive strongest-15 accuracy); the twins section marked replaced; the softplus row marked replaced in the paper; overview counts updated. |
| `results/2026-10-25/paper_figures/` | Copies of `common.py`, `fig_aotl.py` and `fig_maps_combined.py` of 2026-10-24, pointed at this run (`RUN`, `CELLS_DIR`). The map legend says "Positive region" and "Negative region"; panel (c) labels every region. |

**Tests.**
- `aotil/test_oodselect_paper.py` (synthetic cell): all checks pass, including the 14 of check 7
  (`logs/2026-10-25/B_test_oodselect_paper_synthetic.log`). Check 1 (Lightning) is skipped, as before:
  `.venv` has no `pytorch_lightning`.
- **The default reproduces the old outputs bit for bit** (`logs/2026-10-25/B_old_vs_new.py` and `.log`):
  the code before the change (the backup) and after it, run through `main()` on the v3 cell at S = 500
  (3 trials, 400 steps, squared), give identical θ and trial records for classic and spatial; `result.csv`
  and `meta.json` differ only by the added `reg` field. The full rerun confirms it at scale: every
  Laplacian and classic search of this run has the same θ as on 2026-10-24 (§6).
- The private oodselect characterization goldens changed by additions only:
  `Claude outputs/goldens-before-2026-10-25/compare_goldens_oodselect.py` (adapted from 2026-10-24) reports
  "ADDITIONS ONLY" (`logs/2026-10-25/B_compare_goldens_oodselect.log`): a `reg` entry in 8 meta records, a
  `reg` column in 8 result tables, the `--reg laplacian` flag in 34 shell calls, a seventh word in 34 task
  lines, and 28 more ok lines and 56 more printed numbers from check 7. Every old array, table column and
  value is unchanged; the API pins gained `REGS`, `spatial_term`, `_TotalVariation` and the `reg` parameters.
  81 pins pass after the regeneration.
- `aotil/test_region_match.py`: all checks pass (`logs/2026-10-25/C_test_region_match.log`).

---

## 3. Stops S1 and S2

**S1, the refit cell (passes).** `aotil/zoo.py --protocol band --train-dom AE --seed 0 --n-models 300
--data data/processed_w7` into `data/interim/aotil/cells_rerun_2026-10-25` (300 models in 17 s,
`logs/2026-10-25/D1_zoo.log`). `aotil/audit_cell.py`: **verdict PASS** (0 FAIL, 0 WARN; 10 refits match to
1e-9; `logs/2026-10-25/D1_audit_cell.log`). Against `cells_v3` (`logs/2026-10-25/D1_compare_cells.py`):
every array of the `.npz` is identical (point ids of both halves, labels, coordinates, tiles, counties,
both halves' scores, max |diff| 0) and so is the models table.

**S2, the sandbox numbers (passes).** `aotil/region_match.py --exclude own-tile` on `cells_v3` and the
2026-10-23 softplus regions (`results/2026-10-25/region_match_check_own_tile/`,
`logs/2026-10-25/C_region_match_check_own_tile.log`; the same check is check 9 of the test file):

| Set | Sandbox unmatched | Ours | Sandbox η | Ours |
|---|---|---|---|---|
| ε (standardized units) | 1.391 | 1.3914 | | |
| Zone AE baseline | 0.05 | 0.050 | 0.297 | 0.297 |
| Other Zone A points | 0.047 | 0.047 | 0.254 | 0.254 |
| 7 regions pooled | | 0.016 | 0.654 | 0.654 |
| Liberty | 0.00 | 0.000 | 0.73 | 0.725 |
| Levy | 0.00 | 0.000 | 0.74 | 0.737 |
| Dixie | 0.00 | 0.000 | 0.66 | 0.662 |
| Hendry | 0.03 | 0.027 | 0.87 | 0.869 |
| Palm Beach | 0.00 | 0.000 | 0.48 | 0.482 |
| Hamilton | 0.10 | 0.098 | 0.71 | 0.712 |
| Hendry/Palm Beach | 0.02 | 0.018 | 0.48 | 0.482 |
| Halo 3 km | | 0.043 | 0.475 on 138 points | 0.469 on 140 points (134 matched) |

Every difference is at most 0.005. The halo holds 2 more points because the sandbox measured distances
on an equirectangular approximation at latitude 28°, and `region_match.py` on EPSG:3086 metres, the
kNN graph's projection; with the sandbox's projection the 3 km halo holds exactly 138 points
(`logs/2026-10-25/C_halo_projection_check.py` and `.log`). Under the paper's rule (all of a region's
tiles excluded, `results/2026-10-25/region_match_check_region_tiles/`) the same old regions give a pooled
η of 0.635; Dixie drops from 0.66 to 0.47, because under the sandbox rule its matches came partly from the
neighbouring tile that also holds region points. The 3 km halo gives η 0.475 under the paper's rule.

---

## 4. The claims, C1 to C7

Split 1 (seed 0), 300 models, 60 test models, S = 500 unless stated. C1 to C5 are fixed as in
`docs/TASK-REBUILD-V3.md` §2 (C1 with the per-tile gap of 2026-10-24); C6 and C7 as in the brief §2.

| # | Statistic | Value (2026-10-25) | Holds? |
|---|---|---|---|
| C1 | Pooled shift, mean over 300 models of ID minus Zone A (primary: accuracy), tile bootstrap; per-tile gap | Accuracy −0.007 [−0.060, +0.045], AUROC −0.018 [−0.074, +0.040], F1 +0.028 [−0.051, +0.099]. Per tile: accuracy +0.052 [+0.033, +0.072] (24 of 28 tiles, 97% of models), AUROC +0.060 (27 tiles), F1 +0.070. Strongest 15: +0.074 [+0.049, +0.100]. Null (zones shuffled within tile and label): accuracy and F1 within 0.006 of 0, AUROC within 0.019 | Reported, as fixed. `results/2026-10-25/o1_shift.csv`, `o1_tile_null.csv` are identical to 2026-10-24 (every column) |
| C2 | r on all Zone A points, 60 test models | +0.849 [+0.758, +0.907] | **Holds** (> +0.3) |
| C3 | Classic r_te; pieces, tiles, purity; the 3 shuffled classic runs at S = 500 | −0.981 [−0.989, −0.969]; 275 pieces, 25 of 28 tiles, purity 0.131. Shuffled: −0.789, −0.147, +0.160 | **Holds** |
| C4 | Laplacian r_te, purity, pieces; smoothing's purity at S = 250, 500, 1,000 | −0.916 [−0.949, −0.863], purity 0.768, 31 pieces (classic/5 = 55). Purity, ours vs smoothing: 0.355 vs 0.648, 0.768 vs 0.649, 0.646 vs 0.726 | **Does not hold as fixed**: the first three conditions hold; purity above smoothing fails at S = 250 and 1,000. Smoothing still loses the inversion (r_te −0.549, −0.152, +0.096 against −0.942, −0.916, −0.917) |
| C5 | r on the held-out test models with `acc_train_sub` < 0.99, Laplacian S = 500; family-disjoint r at S = 500 | −0.910 [−0.948, −0.846], 51 models. Laplacian trees→others −0.710, others→trees −0.935 (classic −0.870, −0.944) | **Holds** |
| C6a | Pooled unmatched share of the regions (components of ≥ 20 points, Laplacian S = 500) | 0.013 (5 of 381 points); per region 0 to 0.093 (Hamilton) | **Holds** (≤ 0.10). No region above 0.10 |
| C6b | Pooled η of the regions against the Zone AE baseline η | 0.664 against 0.297 | **Holds** (≥ 0.397) |
| C7 | GTV S = 500: r_te, purity, pieces, tiles, regions | −0.877 [−0.925, −0.802]; purity 0.671; 122 pieces; 24 tiles; 7 regions of ≥ 20 points holding 345 points (69%); pooled unmatched 0.003, η 0.581 | Descriptive |

**Why C4 fails** is unchanged from `docs/RUN-SQUARED-PENALTY-2026-10-24.md` §3: the search keeps the
single trial with the lowest validation r, and near-ties in that r decide coherence (at S = 250, 13
trials end within 0.02 of the best, with soft incoherence from 0.04 to 0.84).

---

## 5. The region analysis (Approach 3.6)

`results/2026-10-25/region_match/` (region_match.csv, _kinds.csv, _sensitivity.csv, .json),
`logs/2026-10-25/D5_region_match.log`. The paper's rule: a region's candidates exclude every test tile
that holds a point of the region.

**Caliper.** ε = **1.391** standardized units (1.3914). The pool is 10,785 Zone AE test points; **50** of
the 10,835 were left out for a missing feature (`wetindexcat` 50, `hand` 2; two points miss both). No pool
point lacks a candidate of its land-cover class. Standardization: mean and SD (ddof 1) of each feature
over the 12,531 Zone A test points. The 15 strongest test models (by stored ID AUROC) are models 233, 119,
33, 215, 7, 175, 63, 92, 295, 266, 73, 25, 226, 164 and 182.

**All sets.** n = points; matched = points with at least one match inside ε; Positives = share of positives
among the points / among their matches (mean over matched points of the mean match label); distance =
median over match pairs, in standardized units; distinct = distinct Zone AE points used as matches; SMD =
the largest |mean z of the matched points − mean z of their matches| in Zone A standard deviations, and its
feature; Acc15 = the strongest 15 test models' accuracy on the points / on their matches (descriptive,
not in 3.6); FN = the share of those models' errors on the points that are false negatives.

| Set | n | Matched | Unmatched | η | Positives | Distance | Distinct | SMD (feature) | Acc15 | FN |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---:|
| Zone AE baseline | 10,785 | 10,246 | 0.050 | 0.297 | 0.55 / 0.53 | 0.60 | 7,832 | 0.03 (slope) | 0.78 / 0.78 | 0.62 |
| All regions (6) | 381 | 376 | 0.013 | 0.664 | 0.47 / 0.53 | 0.44 | 829 | 0.04 (log10_dist_coast) | 0.35 / 0.77 | 0.45 |
| Liberty | 123 | 123 | 0.000 | 0.697 | 0.06 / 0.72 | 0.44 | 262 | 0.06 (wetindexcat) | 0.36 / 0.82 | 0.03 |
| Levy | 114 | 114 | 0.000 | 0.647 | 1.00 / 0.35 | 0.40 | 274 | 0.06 (log10_dist_river) | 0.39 / 0.77 | 1.00 |
| Hamilton | 54 | 49 | 0.093 | 0.668 | 0.35 / 0.60 | 0.69 | 138 | 0.26 (log10_dist_coast) | 0.27 / 0.75 | 0.40 |
| Dixie/Levy | 41 | 41 | 0.000 | 0.702 | 0.93 / 0.32 | 0.51 | 106 | 0.09 (hand) | 0.35 / 0.79 | 0.93 |
| Palm Beach | 26 | 26 | 0.000 | 0.446 | 0.00 / 0.45 | 0.32 | 56 | 0.12 (log10_dist_river) | 0.37 / 0.57 | 0.00 |
| Martin | 23 | 23 | 0.000 | 0.739 | 0.00 / 0.74 | 0.47 | 71 | 0.17 (log10_dist_river) | 0.27 / 0.76 | 0.00 |
| Other Zone A points | 12,031 | 11,464 | 0.047 | 0.253 | 0.52 / 0.51 | 0.59 | 7,176 | 0.04 (tri_300) | 0.80 / 0.80 | 0.75 |
| 500 hardest points | 500 | 474 | 0.052 | 0.808 | 0.84 / 0.33 | 0.59 | 1,289 | 0.07 (tri_300) | 0.01 / 0.81 | 0.84 |
| OODSelect S = 500 | 500 | 487 | 0.026 | 0.673 | 0.40 / 0.54 | 0.53 | 1,572 | 0.03 (slope) | 0.25 / 0.76 | 0.45 |
| Halo 1 km | 49 | 49 | 0.000 | 0.380 | 0.41 / 0.53 | 0.58 | 205 | 0.14 (slope) | 0.76 / 0.75 | 0.37 |
| Halo 2 km | 121 | 121 | 0.000 | 0.377 | 0.53 / 0.50 | 0.56 | 450 | 0.08 (slope) | 0.76 / 0.74 | 0.63 |
| Halo 3 km | 226 | 223 | 0.013 | 0.397 | 0.62 / 0.49 | 0.54 | 733 | 0.06 (tri_300) | 0.74 / 0.73 | 0.71 |
| GTV: Liberty | 105 | 105 | 0.000 | 0.722 | 0.04 / 0.74 | 0.43 | 233 | 0.06 (wetindexcat) | 0.31 / 0.84 | 0.01 |
| GTV: Levy | 71 | 70 | 0.014 | 0.477 | 0.82 / 0.37 | 0.49 | 192 | 0.05 (slope) | 0.59 / 0.69 | 1.00 |
| GTV: Hendry | 43 | 43 | 0.000 | 0.577 | 0.02 / 0.60 | 1.04 | 21 | 0.64 (log10_dist_coast) | 0.47 / 0.76 | 0.01 |
| GTV: Dixie | 39 | 39 | 0.000 | 0.482 | 0.54 / 0.27 | 0.55 | 107 | 0.13 (hand) | 0.34 / 0.69 | 0.54 |
| GTV: Palm Beach | 34 | 34 | 0.000 | 0.406 | 0.00 / 0.41 | 0.37 | 93 | 0.06 (log10_dist_river) | 0.50 / 0.61 | 0.00 |
| GTV: Levy (2) | 30 | 30 | 0.000 | 0.487 | 1.00 / 0.51 | 0.40 | 118 | 0.11 (wetindexcat) | 0.61 / 0.74 | 1.00 |
| GTV: Levy (3) | 23 | 23 | 0.000 | 0.817 | 1.00 / 0.18 | 0.50 | 45 | 0.19 (log10_dist_river) | 0.18 / 0.86 | 1.00 |

The six Laplacian regions are exactly the 2026-10-24 regions (`logs/2026-10-25/D3_regions_describe.log`:
every point of each is within 2 km of the same 2026-10-24 region). Land cover, median HAND and river
distance of each region are in `region_match.csv` and in that log. GTV's 7 regions pool to 0.003 unmatched
and η 0.581; three of them lie in Levy, and one (Hendry, 43 points, cultivated crops) is in the farmland
south of Lake Okeechobee that the softplus run of 2026-10-23 had found and the Laplacian runs of
2026-10-24/25 do not.

**Reading.** The Zone AE baseline's unmatched share is 0.050 by construction. The regions are matched more
often than that (0.013), so their inputs are not unusual for Zone AE (no covariate shift, C6a). Their η is
2.2 times the baseline's (0.664 against 0.297), and every region is above the baseline, so the same inputs
carry different labels (concept shift, C6b). The other Zone A points sit below the baseline (0.253), so
the disagreement is local to the regions. The 500 hardest points reach the highest η (0.808), and
OODSelect's scattered subset 0.673: points chosen for errors have high η whatever their coherence, which is
the "built into the selection" caveat of the Limitations. The halos sit between the baseline and the regions
(0.38 to 0.40).

**Two kinds of failure** (majority label; `region_match_kinds.csv`; the corr_te values equal
`results/2026-10-25/o5/A_interpolation.csv`, rows regions_pos and regions_neg):

| Kind | Regions (points) | Positives, points / matches | Unmatched | η | Strongest-15 errors: FN / FP | corr_te, 60 test models | corr_te, 51 held-out below 99% training accuracy |
|---|---|---|---:|---:|---|---:|---:|
| Positive | Levy, Dixie/Levy (155) | 0.98 / 0.34 | 0.000 | 0.662 | 0.98 / 0.02 (1,437 errors) | −0.403 | −0.405 |
| Negative | Liberty, Hamilton, Palm Beach, Martin (226) | 0.12 / 0.66 | 0.022 | 0.665 | 0.12 / 0.88 (2,273 errors) | −0.611 | −0.560 |

**Sensitivity** (report only; the other settings as run):

| Setting | ε | Baseline unmatched | Baseline η | Regions unmatched | Regions η | Other Zone A unmatched | Other Zone A η |
|---|---:|---:|---:|---:|---:|---:|---:|
| As run (95th percentile, at most 5 matches) | 1.391 | 0.050 | 0.297 | 0.013 | 0.664 | 0.047 | 0.253 |
| ε at the 90th percentile | 1.070 | 0.100 | 0.302 | 0.047 | 0.660 | 0.093 | 0.257 |
| ε at the 99th percentile | 2.290 | 0.010 | 0.294 | 0.008 | 0.662 | 0.010 | 0.252 |
| At most 1 match | 1.391 | 0.050 | 0.287 | 0.013 | 0.652 | 0.047 | 0.254 |
| At most 10 matches | 1.391 | 0.050 | 0.295 | 0.013 | 0.666 | 0.047 | 0.252 |

C6a and C6b hold under every setting: the regions' unmatched share stays below the baseline's, and their η
stays above 0.65 against a baseline of at most 0.30.

---

## 6. Every paper number, 2026-10-24 (squared) → 2026-10-25

`logs/2026-10-25/E_paper_numbers.log` (made by `logs/2026-10-25/E_paper_numbers.py 2026-10-24 2026-10-25`
from the result files) prints every search-dependent number for both runs: 369 rows. **Every number present
in both runs is identical**, as the identical θ predicts. The only difference is the summed search time
(1,926 CPU minutes on 2026-10-24, which include 3 h 48 min of lid-closed sleep, against 196 today). Q1 and
Table 1 are identical too (`results/2026-10-25/o1_*.csv` against `results/2026-10-24/`, every column; the
regenerated `o1_table.tex` differs only in the log path of its comment line). So Part D step 5 finds **no
paper number that differs by more than 0.005**, and there is nothing to explain.

The numbers the paper now shows are therefore those of `docs/RUN-SQUARED-PENALTY-2026-10-24.md` §4 (its
"Squared" column), and they come from this run. The numbers new in the paper:

| Paper text | Value | Source |
|---|---|---|
| Settings: "The caliper is ε = 1.39 standardized units, and the 50 Zone AE test points that lack a feature are left out" | 1.3914; 50 | `region_match.json` |
| Settings: "the main searches take 30 minutes" | 30 min 7 s wall clock for the 24 searches of the classic and Laplacian grid, 8 at a time (18:14:57 to 18:45:04, `logs/2026-10-25/C_chain.log` and the result-file times). For reference: the GTV grid took 22 min; S = 500 alone took 7.9 min (classic) and 3.3 min (Laplacian) of single-thread time | `results/2026-10-25/oodselect_paper/*/…_result.csv` (search_secs) |
| Development: "On the current data, the same rule picks μ = 3 again" | purity at S = 500: μ = 1 0.418, μ = 3 0.768 | E_paper_numbers.log |
| Table tab:search, GTV row: −.88 [−.92, −.80], .67, 122, 24; Shuffled GTV: +.20, .96, 7, 8 | −0.877 [−0.925, −0.802]; the shuffled run with the median r_te (null 3: +0.199, purity 0.959) | result.csv files |
| Table tab:budgets, GTV column, S = 100 to 10,000 | −.89 (.50), −.94 (.25), −.88 (.67), −.82 (.88), −.79 (.92), −.69 (.89), −.44 (.86), +.28 (.92); S = 10, 20, 50: −0.880, −0.780, −0.884 | `logs/2026-10-25/E_budget_table.py` |
| GTV paragraph: "1% of its final weights lie between 0.05 and 0.95, against 15%"; "higher purity at 10 of the 12 budgets, but not at 250 and 500"; "weaker at every budget from 500 to 5,000, by 0.04 to 0.13" | 1.3% vs 15.0%; GTV purity below the Laplacian's only at S = 250 (0.253 vs 0.355) and 500 (0.671 vs 0.768); r_te gaps 0.039, 0.111, 0.093, 0.130, 0.124 at 500, 750, 1,000, 2,500, 5,000 | `results/2026-10-25/gtv_vs_laplacian.csv` (`logs/2026-10-25/E_gtv_vs_laplacian.py`) |
| Shuffled control: GTV beats all 9; median +0.48; 1 of 9 below −0.3 | +0.479; one run (−0.503 at S = 250) | E_paper_numbers.log |
| Table tab:errors, the region analysis paragraphs, Limitations | §5 | `region_match*.csv` |
| Abstract: "2.2 times as often" | 0.664 / 0.297 = 2.24 | §5 |

**GTV's soft totals.** Under the squared penalty most runs end with ‖s‖₁ near S, but three GTV runs of the
grid did not: S = 20 (2.0), S = 250 (0.5) and S = 750 (647). Their chosen trials have a tiny λ, so the
budget term never pulled the total up before early stopping; the subset is still the top S of θ, as for
every run. The same happened to four control or μ-sweep runs on 2026-10-24 (report §5 item 9). This does
not change a number in the paper, but it bears on the parked Limitations caveat in `docs/TODO-CLAUDE-CODE.md`.

---

## 7. Paper edits (`CS371_Cho_Li/acl_latex.tex`)

Backup before the edits: `Claude outputs/paper-before-2026-10-25-rerun/acl_latex.tex`. The Approach (from
`\section{Approach}` to `\section{Experiments}`), Related Work, Introduction paragraphs 1 to 4 and 6, and the
team contributions are untouched. Every `% STALE` comment is fixed and deleted. The full before → after text
of every edit, ready to paste, is the Appendix (generated by `logs/2026-10-25/F_paper_edits.py`, also saved
as `logs/2026-10-25/F_paper_edits.md`). Summary, with what is more than a prose change:

| Where | Change | More than prose? |
|---|---|---|
| Abstract | −0.89 → −0.92; 20 → 31 pieces; "about 300" → 275; the twins sentence replaced by the brief's sentence with "2.2 times" | numbers; new claim from §5 |
| Introduction, results paragraph | −0.96 → −0.98; "about 300" → 275; 20 → 31 pieces; 0.08 → 0.07; 65% → 66% (30% unchanged). "as often as Zone AE points do" kept (pooled unmatched 1.3% ≤ 5%) | numbers |
| Subset searches | "...the Laplacian loss of Equation (laplacian) or the GTV loss of Equation (gtv), both with μ = 3 and k = 10. The searches share every other setting." | new method in the experiments |
| Controls | "reruns both searches" → "reruns each search" | GTV added to the control |
| Settings | caliper and missing-feature sentence; 31 → 30 minutes | new numbers |
| Development, fifth decision | "would pick μ = 1" → "picks μ = 3 again" | claim changed (§8) |
| Table tab:search | OODSelect, Laplacian, Smoothing and both shuffled rows updated; row "GTV (μ = 3)" and "Shuffled, GTV" added; "Spatial (μ = 3)" → "Laplacian (μ = 3)" under the *Ours* header, "Shuffled, ours" → "Shuffled, Laplacian" (the brief's "Spatial, GTV" label widened the table by 25 pt, so the two rows sit under *Ours* without "Spatial,"); caption: "show the median of three runs" → "show the run with the median corr_te of three" (the row is one run, not a column-wise median) | new rows; caption precision |
| Figure fig:aotl | `fig_aotl_v3.pdf` → `fig_aotl_v4.pdf` (r = −0.92 in panel b) | new figure |
| Figure fig:maps | `fig_maps_v3.pdf` → `fig_maps_v4.pdf`; caption 27/298 → 25/275, 20/11 → 31/15, 7 → 6 regions, the "two unnamed" sentence removed (all six are labeled; legend "Positive region", "Negative region") | new figure |
| Q2, Q3 paragraphs | numbers (§6); "Seven pieces ... 426 ... (85%)" → "Six pieces ... 381 ... (76%)"; hardest gap 0.28 → 0.30 | numbers |
| Table tab:families | r column: −.97, −.95, −.97, −.93, −.95, −.92, −.96, −.70 | numbers |
| Breakdown by model family | "−0.66 for naive Bayes to −0.96 for decision trees and multilayer perceptrons" → "−0.70 ... −0.97 for decision trees and extra trees" | numbers |
| Table tab:budgets | rebuilt with a GTV column; header "Random purity" → "Random" (caption says it is the purity); caption names the two losses | new column |
| Breakdown by budget | "Both searches" → "All three searches"; "within about 0.1 ... up to 7,500" → "within 0.12 ... from 100 to 7,500 points, and within 0.17 below 100" | claim changed (§8) |
| Comparison with the baselines | 36% → 23%, 11% → 13%, −0.96 → −0.98 | numbers |
| Ablation of the weight μ | rewritten (§8) | claim changed |
| Ablation of the loss | rewritten (§8) | claim changed |
| GTV against the Laplacian (new paragraph, 3 sentences) | after the loss ablation | new result |
| Shuffled-accuracy control | rewritten with the new numbers and GTV (§8) | claim changed |
| Family-disjoint splits | −0.65 → −0.71; reverse −0.87, −0.80 → −0.94, −0.93 | numbers |
| Memorized noise | −0.89 [−0.93, −0.81] → −0.91 [−0.95, −0.85]; −0.66 to −0.96 → −0.70 to −0.97 | numbers |
| Error Analysis | rewritten from §5: new Table tab:errors (per region n, positives, unmatched, η; pooled; other Zone A; baseline); paragraphs Two kinds of failure, Familiar inputs, Different labels, Exceptions, Case studies. Figure fig:twins and its references deleted (`fig_twins_v3.pdf` kept). d, g, p(flood) and the twins are gone everywhere | new analysis |
| Limitations | the twins clause and sentences replaced by the brief's four drafts, filled in (η 0.81 for the hardest points; Hamilton 0.26 SD in distance to the coast; halo η 0.40 on 226 points). One wording change to the fourth draft: "We report η and the unmatched shares as point estimates without intervals", since the paper does report intervals elsewhere | claims from §5 |

**Grep after the edits.** No `STALE`, `twin`, `Hendry`, `flood-labeled`, `dry-labeled`, `p(flood)`,
`input distance`, `298` or `about 300` is left. The softplus-era numbers −0.89, 0.86, 20 pieces and 0.08 are gone.

---

## 8. Claims changed

Each sentence below said something the new numbers contradict; each was rewritten to what the data show.

1. **Ablation of the weight μ.** Was: "purity rises from 0.11 at μ = 0 to 0.78, 0.86, 0.89 ... Most of the
   coherence comes with the first unit of weight". Now: purity 0.13, 0.42, 0.77, 0.67 at S = 500; "Purity does
   not rise steadily with μ" (at S = 250: 0.64, 0.35, 0.79 for μ = 1, 3, 10), "no single weight gives the most
   coherent subset at every budget", every weight inverts and r_te weakens by at most 0.10 from μ = 0 to 10.
   The conclusion ("a reasonable fixed choice rather than a tuned optimum") is kept.
2. **Ablation of the loss (and C4).** Was: "Smoothing ... fails at every budget". Now: smoothing "also makes
   subsets coherent" (purity 0.65, 0.65, 0.73 against ours 0.35, 0.77, 0.65), "However, it loses the
   inversion" (−0.55, −0.15, +0.10 against −0.94, −0.92, −0.92). The conclusion (coherence must be part of
   the objective) is kept: it rests on the lost inversion, not on purity.
3. **Shuffled-accuracy control.** Was: "beats all 18 shuffled runs", medians +0.16 and +0.59, "4 of 9 ... 3 of
   9", one run at −0.69. Now: all 9 for OODSelect and for GTV, 8 of 9 for the Laplacian (at S = 1,000 a
   shuffled run reaches −0.921 against the real −0.917); medians −0.15, −0.54, +0.48; 3, 5 and 1 of 9 below
   −0.3; two shuffled Laplacian runs at S = 500 reach −0.56 and −0.67. The conclusion (a negative r_te alone
   is weak evidence) is kept and is stronger.
4. **Breakdown by budget.** Was: "within about 0.1 of OODSelect at every budget up to 7,500" (0.115 over
   100–7,500 and 0.164 at S = 20 under the squared penalty). Now "within 0.12 ... from 100 to 7,500 points, and
   within 0.17 below 100"; "Both searches" → "All three searches" (GTV also inverts from 10 to 7,500).
5. **Development, fifth decision.** Was: "the same rule would pick μ = 1". Now: "picks μ = 3 again".
6. **Error Analysis and Abstract/Introduction result 4.** The twins claims (d, g = +0.34, "the models predict
   the same", "0.34 less accurate") are gone; the claim is now the matched-input one (C6a, C6b).
7. **Figure 3 caption.** The two unnamed regions in Hendry and Palm Beach no longer exist; all six regions are
   named in panel (c).

---

## 9. Proposed ninth decision (not added to the paper)

Development now lists eight decisions. For Victor, as the brief drafts it:

> Ninth, we first summarized each region by its input distance and accuracy gap to matched patches of Zone
> AE points. We replaced this with per-point matching, which separates unfamiliar inputs from different
> labels directly.

It is accurate as written. If it goes in, "Eight decisions shaped these experiments" becomes "Nine".

---

## 10. Page count

Compiled in a scratch copy (pdflatex, bibtex, pdflatex ×2): **15 pages**, as before the edits. The body (to
the end of the Discussion) ends on **page 12**, as before; the references now start on page 13 (page 12
before). So the paper is still about 4 pages over the 8-page limit. No undefined reference; the only
undefined citation is `wang2016trend`, which Victor adds himself. One overfull box, 11.6 pt in Table
tab:search, which was there before the edits (the new rows did not widen it); no new overfull box.

---

## 11. Deviations and notes

- **Order of the parts.** The brief's order was kept for every decision point: S1 passed before any search
  started, and S2 passed before `region_match.py` ran on the new regions. But to save wall clock, the 81
  searches (Part D) ran in the background while Part C was written and tested. The searches do not depend on
  Part C.
- **Early runs.** Regions, the region analysis, the smoothing audit and Figure 1 ran once early (to check C6
  before any Abstract, Introduction or Error Analysis edit, and while the controls searched) and again in
  `logs/2026-10-25/D_chain.sh`; the outputs are deterministic and identical. The chain's D1 audit at first
  skipped because the early outputs existed, so it was rerun with `--overwrite` (and the chain now passes it).
- **Halo projection.** `region_match.py` measures halo distances in EPSG:3086 metres (the kNN graph's
  projection), not the sandbox's equirectangular approximation; the 3 km halo holds 140 points instead of 138
  (§3).
- **Hardest points.** The region analysis needs one set, so ties at the cut are broken in one seeded random
  order; `report_paper.py`'s hard_r averages over 100 orders. Its η (0.808) is a single draw.
- **Figure 1 panel title.** Panel (b) of `fig_aotl` is titled "Our selected regions" (unchanged script), but it
  plots all 500 selected points, of which 381 lie in regions. The caption is right ("the 500 points our method
  selects"); the panel title could say "Our selected points". Not changed.
- **Table numbers.** The brief's "Table 2", "Table 3" and "Table 5" are tab:search, tab:errors and tab:budgets.
  In the compiled paper they are Tables 3, 6 and 5 (Table 1 is the data split).
- **Load.** Each single-thread search process used about 1.4 cores (PyTorch's own threads), as on 2026-10-24;
  results are seeded and unaffected.
- **The whole private suite** (`logs/2026-10-25/F_tests_all.log`): 1,208 tests, the same 3 failures and 1
  error as on 2026-10-24 (`logs/2026-10-24/A_tests_all.log`), all known and unrelated: a documented call in
  `docs/TASK-REBUILD-V3.md`, and data pairing between older cells and the v3 dataset.
- **The owner check scripts** (`aotil/test_*.py`, `logs/2026-10-25/F_owner_tests.log`): 7 pass, including
  `test_oodselect_paper.py` and the new `test_region_match.py`. 6 fail (`test_match_controls`, `test_o5_contrast`,
  `test_o5_feature_inclusion`, `test_o5_interpolation`, `test_o5_pooling`, `test_o5_shift_decomp`). By default
  their real-data checks read the 2026-10-16/17 outputs, which were made on the v2 data, against the v3
  dataset ("dataset lon differs from the cell's", region land counts of seeds 1 and 2). None of them calls the
  code this run changed (whose default path is bit-identical anyway). This is the data pairing noted on
  2026-10-23; rerunning those old analyses on v3 would clear it, and the paper does not use them.

---

## 12. Docs updated

- `aotil/README.md`: `region_match.py` row, the `--reg` flag, the squared penalty, the `BUDGET_PENALTY` and
  `REG` knobs, run order item 5; the twins scripts marked superseded.
- `docs/DECISION-PROVENANCE.md`: rows for the squared penalty in the paper and GTV (μ = 3, not tuned), a new
  `aotil/region_match.py` table (the 5% caliper after Stuart 2010, at most 5 matches, region tiles excluded,
  and eight more rows), the twins section and the softplus row marked replaced, the overview counts updated
  (460 rows).
- `docs/TODO-CLAUDE-CODE.md`: the standing rules now say Approach 3.1 to 3.6 is final; item 4 moved to Done
  with this report.

---

## Appendix: every paper edit, before → after

Generated by `logs/2026-10-25/F_paper_edits.py` from the backup and the edited file; each edit is one
or more consecutive lines (each paragraph of the source is one line). "near" names the closest section,
paragraph, caption or label above it.

#### Edit 1 (replace), near: start

Before:

```latex
% STALE (abstract): -0.89 and 20 pieces come from the softplus-penalty run (squared rerun: -0.92, 31 pieces, docs/RUN-SQUARED-PENALTY-2026-10-24.md), and the 0.34 sentence comes from the removed twins analysis. Update after the region-analysis rerun.
Machine learning models trained on FEMA flood maps are used to extend flood hazard maps. These models pool two labels drawn by different procedures, approximate Zone A and detailed Zone AE, assuming that the models absorb the difference. We test this assumption by training 300 models on Zone AE and searching Zone A for subsets where accuracy-on-the-line inverts. Since OODSelect, the existing search, returns scattered points that cannot be studied as places, we add a spatial regularization loss that turns them into coherent regions. We then match each region to held-out Zone AE points with similar inputs, which separates unfamiliar inputs from different labels. On 80,000 points across Florida, the pooled accuracy gap is only $-0.007$, because the test points of the two zones lie in different places. Within the same tiles, models lose 0.052 accuracy on Zone A. Our search finds regions where the correlation inverts ($-0.89$) in 20 pieces instead of about 300. The models predict the same for the regions and their matched points, yet the strongest models are 0.34 less accurate on the regions, because the two procedures label the same inputs differently. To the best of our knowledge, this is the first work to treat FEMA's labeling procedures as a distribution shift and to locate where they disagree.
```

After:

```latex
Machine learning models trained on FEMA flood maps are used to extend flood hazard maps. These models pool two labels drawn by different procedures, approximate Zone A and detailed Zone AE, assuming that the models absorb the difference. We test this assumption by training 300 models on Zone AE and searching Zone A for subsets where accuracy-on-the-line inverts. Since OODSelect, the existing search, returns scattered points that cannot be studied as places, we add a spatial regularization loss that turns them into coherent regions. We then match each region to held-out Zone AE points with similar inputs, which separates unfamiliar inputs from different labels. On 80,000 points across Florida, the pooled accuracy gap is only $-0.007$, because the test points of the two zones lie in different places. Within the same tiles, models lose 0.052 accuracy on Zone A. Our search finds regions where the correlation inverts ($-0.92$) in 31 pieces instead of 275. Their points have close matches among held-out Zone AE points, yet they disagree in label with those matches 2.2 times as often as Zone AE points do, because the two procedures label the same inputs differently. To the best of our knowledge, this is the first work to treat FEMA's labeling procedures as a distribution shift and to locate where they disagree.
```

#### Edit 2 (replace), near: Introduction

Before:

```latex
% STALE (intro): -0.96, about 300, 20 pieces, and 0.08 are softplus numbers (squared rerun: -0.98, 275, 31, 0.06). The fourth result uses pilot numbers on the softplus regions (eta 0.65 vs baseline 0.30). Recompute both in the region-analysis rerun.
Our experiments show four results. First, pooling hides the shift. Pooled, models are as accurate on Zone A as on held-out Zone AE (a gap of $-0.007$), because more of the Zone AE test points lie where models are weak. Within the same tiles, models lose 0.052 accuracy on Zone A, yet accuracy-on-the-line holds with a correlation of 0.85. Second, OODSelect finds a subset of Zone A where the correlation inverts to $-0.96$, but it is split into about 300 pieces. Third, our loss reduces these to 20 pieces at a cost of 0.08 in correlation. Fourth, the regions are not unfamiliar land. Their points find matches among held-out Zone AE points as often as Zone AE points do. Yet a region's point and its matches disagree in label 65\% of the time, against 30\% between Zone AE points, so the two procedures label the same inputs differently. The inversion does not come from memorized label noise.
```

After:

```latex
Our experiments show four results. First, pooling hides the shift. Pooled, models are as accurate on Zone A as on held-out Zone AE (a gap of $-0.007$), because more of the Zone AE test points lie where models are weak. Within the same tiles, models lose 0.052 accuracy on Zone A, yet accuracy-on-the-line holds with a correlation of 0.85. Second, OODSelect finds a subset of Zone A where the correlation inverts to $-0.98$, but it is split into 275 pieces. Third, our loss reduces these to 31 pieces at a cost of 0.07 in correlation. Fourth, the regions are not unfamiliar land. Their points find matches among held-out Zone AE points as often as Zone AE points do. Yet a region's point and its matches disagree in label 66\% of the time, against 30\% between Zone AE points, so the two procedures label the same inputs differently. The inversion does not come from memorized label noise.
```

#### Edit 3 (replace), near: Subset searches.

Before:

```latex
\paragraph{Subset searches.} We compare five ways of choosing $S$ OOD points, grouped in Table~\ref{tab:search}. Three are baselines: all Zone A points, random subsets (the mean of 100 draws), and the hardest points, which the test models get wrong most often \citep{salaudeen2025oodselect}. Since the hardest points are ranked by the test models themselves, their $\text{corr}_\text{te}$ is only indicative. The state of the art is OODSelect ($\mu=0$), and our method adds the Laplacian loss of Equation~\ref{eq:laplacian} ($\mu=3$ and $k=10$). The two searches share every other setting. As an ablation, smoothing after search averages OODSelect's final weights over each point's 10 nearest neighbors and takes the $S$ largest, which makes a subset coherent without optimizing it.
\paragraph{Controls.} Three controls test whether an inversion is real. The shuffled control reruns both searches three times per budget with the ID accuracies of the selection and validation models permuted, which keeps $Z$ but removes any link between ID and OOD accuracy. The family-disjoint control selects with the tree-based families and tests on the others, and the reverse. The memorization control removes the models that reach 99\% accuracy on their own training rows, since \citet{sanyal2025} show that models that fit noisy labels can invert the trend.
```

After:

```latex
\paragraph{Subset searches.} We compare five ways of choosing $S$ OOD points, grouped in Table~\ref{tab:search}. Three are baselines: all Zone A points, random subsets (the mean of 100 draws), and the hardest points, which the test models get wrong most often \citep{salaudeen2025oodselect}. Since the hardest points are ranked by the test models themselves, their $\text{corr}_\text{te}$ is only indicative. The state of the art is OODSelect ($\mu=0$), and our method adds the Laplacian loss of Equation~\ref{eq:laplacian} or the GTV loss of Equation~\ref{eq:gtv}, both with $\mu=3$ and $k=10$. The searches share every other setting. As an ablation, smoothing after search averages OODSelect's final weights over each point's 10 nearest neighbors and takes the $S$ largest, which makes a subset coherent without optimizing it.
\paragraph{Controls.} Three controls test whether an inversion is real. The shuffled control reruns each search three times per budget with the ID accuracies of the selection and validation models permuted, which keeps $Z$ but removes any link between ID and OOD accuracy. The family-disjoint control selects with the tree-based families and tests on the others, and the reverse. The memorization control removes the models that reach 99\% accuracy on their own training rows, since \citet{sanyal2025} show that models that fit noisy labels can invert the trend.
```

#### Edit 4 (replace), near: Experimental Settings

Before:

```latex
Following the released OODSelect code, each search runs $T=100$ trials of at most 50,000 full-batch steps with patience $P=500$. TPE draws the initial learning rate from $[10^{-2},10^{2}]$, the minimum learning rate from $[10^{-3},1]$, and $\lambda$ from $[10^{-6},10^{2}]$, all on a log scale, $\tau$ from $\{10,\dots,200\}$, and $W$ from $\{10,\dots,1000\}$. We report $S=500$, which is 4.0\% of the OOD points, and regions of at least $c=20$ points. The ablations add $S=250$ and $1{,}000$, and a sweep runs over budgets from 10 to 10,000 points. The $k$-nearest-neighbor graph uses coordinates projected to meters. Matching uses the $q=6$ numeric features, matches land cover exactly, and keeps at most 5 matches per point.
Everything runs on one laptop with an Apple M4 Pro CPU, using Python 3.12, PyTorch 2.14, scikit-learn 1.9, XGBoost 3.4, and Optuna 5.0 \citep{paszke2019pytorch,pedregosa2011sklearn,chen2016xgboost,akiba2019optuna}. The raster features are read with Google Earth Engine. The 300 models train in 17 seconds, and the main searches take 31 minutes. The tile split, the models, the trials, and the bootstrap use fixed seeds, and refitting a model reproduces its predictions exactly.
```

After:

```latex
Following the released OODSelect code, each search runs $T=100$ trials of at most 50,000 full-batch steps with patience $P=500$. TPE draws the initial learning rate from $[10^{-2},10^{2}]$, the minimum learning rate from $[10^{-3},1]$, and $\lambda$ from $[10^{-6},10^{2}]$, all on a log scale, $\tau$ from $\{10,\dots,200\}$, and $W$ from $\{10,\dots,1000\}$. We report $S=500$, which is 4.0\% of the OOD points, and regions of at least $c=20$ points. The ablations add $S=250$ and $1{,}000$, and a sweep runs over budgets from 10 to 10,000 points. The $k$-nearest-neighbor graph uses coordinates projected to meters. Matching uses the $q=6$ numeric features, matches land cover exactly, and keeps at most 5 matches per point. The caliper is $\epsilon=1.39$ standardized units, and the 50 Zone AE test points that lack a feature are left out of matching.
Everything runs on one laptop with an Apple M4 Pro CPU, using Python 3.12, PyTorch 2.14, scikit-learn 1.9, XGBoost 3.4, and Optuna 5.0 \citep{paszke2019pytorch,pedregosa2011sklearn,chen2016xgboost,akiba2019optuna}. The raster features are read with Google Earth Engine. The 300 models train in 17 seconds, and the main searches take 30 minutes. The tile split, the models, the trials, and the bootstrap use fixed seeds, and refitting a model reproduces its predictions exactly.
```

#### Edit 5 (replace), near: Development

Before:

```latex
Eight decisions shaped these experiments. First, we replaced 45 covariates of our own choosing with the published predictors of \citet{woznicki2019}, so that our models resemble those used to extend FEMA's maps. Second, the released OODSelect code differs from its paper in the budget penalty, its warm-up, and the step it keeps. We follow the paper for the penalty and the code for the warm-up and the kept step. Since the squared penalty is measured in points squared, we widened the range of $\lambda$. Third, our first datasets held points on open water, where two features are undefined, and two regions lay on Lake Okeechobee. We added the land mask and rebuilt the data. The rebuild also fixed a bug that counted polygon area twice when sampling positives, so that polygons under 1\,km$^2$ had supplied only 0.6\% of the positives. Fourth, rerunning one split on another machine moved $\text{corr}_\text{te}$ by 0.07, because the search depends on its path. We therefore do not read differences below 0.1, and we test inversions with controls rather than by their size. Fifth, we set $\mu=3$ in a pilot on an earlier version of the data, as the smallest weight with a purity of at least 0.5, and kept it fixed. On the current data, the same rule would pick $\mu=1$. Sixth, we first compared regions with Zone A points of similar terrain. Terrain covers only three of the seven inputs, so we now match each region to Zone AE points on all seven. Seventh, we first measured the shift in AUROC. Since OODSelect works with accuracy and the three metrics can disagree in sign, we made accuracy the primary metric before computing any result on the rebuilt data. Eighth, the pooled gap was near zero on the rebuilt data. Since the two test sets hold the tiles in different proportions, we added the per-tile gap after seeing this result.
% STALE (results): Tables 2 and 5, the r column of Table 4, Figures 1 to 3, and the Q2, Q3, and ablation numbers come from the softplus-penalty run. The squared-penalty rerun is in docs/RUN-SQUARED-PENALTY-2026-10-24.md (sections 4 and 5). Q1 and Table 1 do not depend on the search.
```

After:

```latex
Eight decisions shaped these experiments. First, we replaced 45 covariates of our own choosing with the published predictors of \citet{woznicki2019}, so that our models resemble those used to extend FEMA's maps. Second, the released OODSelect code differs from its paper in the budget penalty, its warm-up, and the step it keeps. We follow the paper for the penalty and the code for the warm-up and the kept step. Since the squared penalty is measured in points squared, we widened the range of $\lambda$. Third, our first datasets held points on open water, where two features are undefined, and two regions lay on Lake Okeechobee. We added the land mask and rebuilt the data. The rebuild also fixed a bug that counted polygon area twice when sampling positives, so that polygons under 1\,km$^2$ had supplied only 0.6\% of the positives. Fourth, rerunning one split on another machine moved $\text{corr}_\text{te}$ by 0.07, because the search depends on its path. We therefore do not read differences below 0.1, and we test inversions with controls rather than by their size. Fifth, we set $\mu=3$ in a pilot on an earlier version of the data, as the smallest weight with a purity of at least 0.5, and kept it fixed. On the current data, the same rule picks $\mu=3$ again. Sixth, we first compared regions with Zone A points of similar terrain. Terrain covers only three of the seven inputs, so we now match each region to Zone AE points on all seven. Seventh, we first measured the shift in AUROC. Since OODSelect works with accuracy and the three metrics can disagree in sign, we made accuracy the primary metric before computing any result on the rebuilt data. Eighth, the pooled gap was near zero on the rebuilt data. Since the two test sets hold the tiles in different proportions, we added the per-tile gap after seeing this result.
```

#### Edit 6 (replace), near: tab:shift

Before:

```latex
\includegraphics[width=\columnwidth]{figures/fig_aotl_v3.pdf}
```

After:

```latex
\includegraphics[width=\columnwidth]{figures/fig_aotl_v4.pdf}
```

#### Edit 7 (replace), near: fig:aotl

Before:

```latex
OODSelect ($\mu = 0$) & $-$.96 [$-$.98, $-$.94] & .11 & 298 & 27 \\
```

After:

```latex
OODSelect ($\mu = 0$) & $-$.98 [$-$.99, $-$.97] & .13 & 275 & 25 \\
```

#### Edit 8 (replace), near: fig:aotl

Before:

```latex
\textbf{Spatial ($\mu = 3$)} & $-$.89 [$-$.93, $-$.82] & \textbf{.86} & \textbf{20} & \textbf{11} \\
```

After:

```latex
\textbf{Laplacian ($\mu = 3$)} & $-$.92 [$-$.95, $-$.86] & \textbf{.77} & \textbf{31} & \textbf{15} \\
GTV ($\mu = 3$) & $-$.88 [$-$.92, $-$.80] & .67 & 122 & 24 \\
```

#### Edit 9 (replace), near: fig:aotl

Before:

```latex
Smoothing & $-$.22 [$-$.45, .04] & .72 & 51 & 21 \\
Shuffled, OODSelect & $-$.49 & .08 & 330 & 28 \\
Shuffled, ours & $+$.59 & .15 & 242 & 25 \\
```

After:

```latex
Smoothing & $-$.15 [$-$.39, .11] & .65 & 51 & 18 \\
Shuffled, OODSelect & $-$.15 & .06 & 355 & 27 \\
Shuffled, Laplacian & $-$.56 & .53 & 150 & 25 \\
Shuffled, GTV & $+$.20 & .96 & 7 & 8 \\
```

#### Edit 10 (replace), near: fig:aotl

Before:

```latex
\caption{Subset search at $S = 500$, 4.0\% of the Zone A points (Q2 and Q3). $\text{corr}_\text{te}$ (Equation~\ref{eq:corr}) is computed on the 60 test models, which the search never saw, and a more negative value is a stronger inversion. Purity (Equation~\ref{eq:purity}) is 1 for one solid region and .04 for random points. Pieces are the connected components of the subset on the 10-nearest-neighbor graph, and tiles are the test tiles they touch, out of 28. Random is the mean of 100 draws. The hardest points are ranked by the test models' errors, so their $\text{corr}_\text{te}$ is indicative. Smoothing averages OODSelect's weights over each point's 10 neighbors after the search. Shuffled rows rerun the search with the ID accuracies of the selection and validation models shuffled, and show the median of three runs.}
```

After:

```latex
\caption{Subset search at $S = 500$, 4.0\% of the Zone A points (Q2 and Q3). $\text{corr}_\text{te}$ (Equation~\ref{eq:corr}) is computed on the 60 test models, which the search never saw, and a more negative value is a stronger inversion. Purity (Equation~\ref{eq:purity}) is 1 for one solid region and .04 for random points. Pieces are the connected components of the subset on the 10-nearest-neighbor graph, and tiles are the test tiles they touch, out of 28. Random is the mean of 100 draws. The hardest points are ranked by the test models' errors, so their $\text{corr}_\text{te}$ is indicative. Smoothing averages OODSelect's weights over each point's 10 neighbors after the search. Shuffled rows rerun the search with the ID accuracies of the selection and validation models shuffled, and show the run with the median $\text{corr}_\text{te}$ of three.}
```

#### Edit 11 (replace), near: tab:search

Before:

```latex
\includegraphics[width=\textwidth]{figures/fig_maps_v3.pdf}
\caption{Selected points and regions at $S = 500$. Gray dots are Zone A test points, and squares are the 28 test tiles of 50\,km. (a) OODSelect spreads its points over 27 test tiles in 298 pieces. (b) Our search groups them into 20 pieces in 11 tiles. (c) The 7 regions of at least 20 points, colored by whether most of their points are positives or negatives. The two unnamed negative regions lie in Hendry (west) and Palm Beach (east).}
```

After:

```latex
\includegraphics[width=\textwidth]{figures/fig_maps_v4.pdf}
\caption{Selected points and regions at $S = 500$. Gray dots are Zone A test points, and squares are the 28 test tiles of 50\,km. (a) OODSelect spreads its points over 25 test tiles in 275 pieces. (b) Our search groups them into 31 pieces in 15 tiles. (c) The 6 regions of at least 20 points, colored by whether most of their points are positives or negatives.}
```

#### Edit 12 (replace), near: OODSelect finds inversions, but scattered (Q2).

Before:

```latex
\paragraph{OODSelect finds inversions, but scattered (Q2).} Random subsets follow the aggregate trend ($+0.84$), so small subsets do not invert by chance (Table~\ref{tab:search}). The hardest points invert ($-0.68$), but they are ranked by the test models themselves, and ranking them by the selection models instead gives $-0.62$. OODSelect finds a much stronger inversion of $-0.96$ on test models it never saw. However, its subset is split into 298 pieces across 27 of the 28 test tiles (Figure~\ref{fig:maps}a). With a purity of 0.11, a selected point has about 9 unselected points among its 10 nearest neighbors, so the subset cannot be described as a set of places.
\paragraph{Our loss turns subsets into regions (Q3).} Our search keeps a strong inversion of $-0.89$ [$-0.93$, $-0.82$] (Figure~\ref{fig:aotl}b), with a purity of 0.86 in 20 pieces over 11 tiles (Figure~\ref{fig:maps}b). Seven pieces of at least 20 points hold 426 of the 500 points (85\%), and these are the regions we analyze (Figure~\ref{fig:maps}c). The cost of coherence is 0.08 in $\text{corr}_\text{te}$, within the 0.1 that path dependence can move it and well below the gap between OODSelect and the hardest points (0.28). The cost is not zero, however, since the two intervals do not overlap. Together, these results answer Q1 to Q3. Pooled, the two procedures look interchangeable. Yet Zone A is harder within the same tiles, and it holds coherent places where better Zone AE models do worse. We answer Q4 by analyzing the errors on these regions.
```

After:

```latex
\paragraph{OODSelect finds inversions, but scattered (Q2).} Random subsets follow the aggregate trend ($+0.84$), so small subsets do not invert by chance (Table~\ref{tab:search}). The hardest points invert ($-0.68$), but they are ranked by the test models themselves, and ranking them by the selection models instead gives $-0.62$. OODSelect finds a much stronger inversion of $-0.98$ on test models it never saw. However, its subset is split into 275 pieces across 25 of the 28 test tiles (Figure~\ref{fig:maps}a). With a purity of 0.13, a selected point has about 9 unselected points among its 10 nearest neighbors, so the subset cannot be described as a set of places.
\paragraph{Our loss turns subsets into regions (Q3).} Our search keeps a strong inversion of $-0.92$ [$-0.95$, $-0.86$] (Figure~\ref{fig:aotl}b), with a purity of 0.77 in 31 pieces over 15 tiles (Figure~\ref{fig:maps}b). Six pieces of at least 20 points hold 381 of the 500 points (76\%), and these are the regions we analyze (Figure~\ref{fig:maps}c). The cost of coherence is 0.07 in $\text{corr}_\text{te}$, within the 0.1 that path dependence can move it and well below the gap between OODSelect and the hardest points (0.30). The cost is not zero, however, since the two intervals do not overlap. Together, these results answer Q1 to Q3. Pooled, the two procedures look interchangeable. Yet Zone A is harder within the same tiles, and it holds coherent places where better Zone AE models do worse. We answer Q4 by analyzing the errors on these regions.
```

#### Edit 13 (replace), near: Performance Analysis

Before:

```latex
Decision tree & 38 & .720 & .730 & .046 & $-$.96 & 0 \\
Random forest & 37 & .734 & .740 & .055 & $-$.93 & 12 \\
Extra trees & 38 & .736 & .751 & .036 & $-$.94 & 12 \\
Hist.\ grad.\ boosting & 38 & .729 & .741 & .054 & $-$.94 & 1 \\
XGBoost & 37 & .745 & .740 & .077 & $-$.90 & 6 \\
Logistic regression & 38 & .686 & .670 & .054 & $-$.85 & 0 \\
```

After:

```latex
Decision tree & 38 & .720 & .730 & .046 & $-$.97 & 0 \\
Random forest & 37 & .734 & .740 & .055 & $-$.95 & 12 \\
Extra trees & 38 & .736 & .751 & .036 & $-$.97 & 12 \\
Hist.\ grad.\ boosting & 38 & .729 & .741 & .054 & $-$.93 & 1 \\
XGBoost & 37 & .745 & .740 & .077 & $-$.95 & 6 \\
Logistic regression & 38 & .686 & .670 & .054 & $-$.92 & 0 \\
```

#### Edit 14 (replace), near: Performance Analysis

Before:

```latex
Naive Bayes & 37 & .692 & .724 & .027 & $-$.66 & 0 \\
```

After:

```latex
Naive Bayes & 37 & .692 & .724 & .027 & $-$.70 & 0 \\
```

#### Edit 15 (replace), near: tab:families

Before:

```latex
\begin{tabular}{@{}rccc@{}}
```

After:

```latex
\begin{tabular}{@{}rcccc@{}}
```

#### Edit 16 (replace), near: tab:families

Before:

```latex
$\boldsymbol{S}$ & \textbf{Random purity} & \textbf{OODSelect} & \textbf{Ours} \\
```

After:

```latex
$\boldsymbol{S}$ & \textbf{Random} & \textbf{OODSelect} & \textbf{Laplacian} & \textbf{GTV} \\
```

#### Edit 17 (replace), near: tab:families

Before:

```latex
100 & .01 & $-$.90 (.04) & $-$.82 (.63) \\
250 & .02 & $-$.95 (.07) & $-$.89 (.69) \\
500 & .04 & $-$.96 (.11) & $-$.89 (.86) \\
1,000 & .08 & $-$.96 (.18) & $-$.88 (.86) \\
2,500 & .20 & $-$.97 (.33) & $-$.90 (.77) \\
5,000 & .40 & $-$.92 (.54) & $-$.82 (.82) \\
7,500 & .60 & $-$.62 (.73) & $-$.54 (.84) \\
10,000 & .80 & $+$.21 (.88) & $+$.28 (.93) \\
```

After:

```latex
100 & .01 & $-$.97 (.04) & $-$.95 (.22) & $-$.89 (.50) \\
250 & .02 & $-$.98 (.08) & $-$.94 (.35) & $-$.94 (.25) \\
500 & .04 & $-$.98 (.13) & $-$.92 (.77) & $-$.88 (.67) \\
1,000 & .08 & $-$.98 (.20) & $-$.92 (.65) & $-$.82 (.88) \\
2,500 & .20 & $-$.97 (.32) & $-$.92 (.73) & $-$.79 (.92) \\
5,000 & .40 & $-$.93 (.54) & $-$.82 (.82) & $-$.69 (.89) \\
7,500 & .60 & $-$.62 (.73) & $-$.52 (.84) & $-$.44 (.86) \\
10,000 & .80 & $+$.21 (.88) & $+$.25 (.91) & $+$.28 (.92) \\
```

#### Edit 18 (replace), near: tab:families

Before:

```latex
\caption{The search across budgets, as $\text{corr}_\text{te}$ with purity in parentheses. Random purity is $S/m$, the purity of a random subset of $S$ of the 12,531 Zone A points. Budgets of 10 to 50 points, not shown, also invert.}
```

After:

```latex
\caption{The search across budgets, as $\text{corr}_\text{te}$ with purity in parentheses. Random is the purity of a random subset of $S$ of the 12,531 Zone A points, $S/m$. Laplacian and GTV are our search with each loss. Budgets of 10 to 50 points, not shown, also invert.}
```

#### Edit 19 (replace), near: Breakdown by model family.

Before:

```latex
\paragraph{Breakdown by model family.} Every family follows the overall pattern (Table~\ref{tab:families}). Pooled, its Zone A accuracy lies within 0.032 of its ID accuracy. Per tile, every family does worse on Zone A, from 0.027 for naive Bayes to 0.077 for XGBoost. On our selected points, the trend inverts within every family, from $-0.66$ for naive Bayes to $-0.96$ for decision trees and multilayer perceptrons. The inversion is therefore not a contrast between families, such as trees against linear models, but holds among models of the same kind.
\paragraph{Breakdown by budget.} Both searches invert at every budget from 10 to 7,500 points (Table~\ref{tab:budgets}). At 7,500 points, 60\% of Zone A, OODSelect still finds $-0.62$, so the positive trend on all of Zone A hides a trend that runs the other way on most of it. Our search stays within about 0.1 of OODSelect at every budget up to 7,500, while its purity stays well above that of random subsets. OODSelect's purity stays within 0.15 of the random level at every budget, so it becomes more coherent only by selecting a larger share of the points.
\paragraph{Comparison with the baselines.} OODSelect's subset shares 36\% of its points with the 500 hardest points, and ours shares 11\%. Neither search simply returns the points that models get wrong. They return points whose errors grow with ID accuracy, which the hardest points capture only in part ($-0.68$ against $-0.96$).
\paragraph{Ablation of the weight $\mu$.} At $S=500$, purity rises from 0.11 at $\mu=0$ to 0.78 at $\mu=1$, 0.86 at $\mu=3$, and 0.89 at $\mu=10$, while $\text{corr}_\text{te}$ is $-0.96$, $-0.85$, $-0.89$, and $-0.89$. Most of the coherence comes with the first unit of weight, and beyond it $\text{corr}_\text{te}$ changes by less than path dependence allows. The weight $\mu=3$ is therefore a reasonable fixed choice rather than a tuned optimum.
\paragraph{Ablation of the loss.} Smoothing OODSelect's weights after the search is simpler than our loss, but it fails at every budget. At $S = 250$, 500, and 1,000, its purity is 0.67, 0.72, and 0.71 against 0.69, 0.86, and 0.86 for ours, and its $\text{corr}_\text{te}$ is $-0.31$, $-0.22$, and $+0.13$ against $-0.89$, $-0.89$, and $-0.88$. The points around OODSelect's scattered selections are not the ones on which better models fail, so coherence must be part of the objective rather than added afterwards.
\paragraph{Shuffled-accuracy control.} An inversion could come from the search alone, so we rerun both searches with shuffled ID accuracies at $S = 250$, 500, and 1,000. The real search beats all 18 shuffled runs at its own budget. However, the shuffled runs have medians of $+0.16$ for OODSelect and $+0.59$ for ours, and 4 of 9 OODSelect runs and 3 of 9 of ours fall below $-0.3$. One shuffled run of ours at $S=500$ even reaches $-0.69$ with a purity of 0.67. A negative $\text{corr}_\text{te}$ alone is therefore weak evidence, so our claims rest on this comparison and on the region analysis.
\paragraph{Family-disjoint splits.} When the search selects with the tree-based families and tests on the others, OODSelect reaches $-0.87$ and ours $-0.65$. In the reverse direction, they reach $-0.87$ and $-0.80$. The inversion transfers across families, although our coherent subsets transfer less well when trees guide the search.
\paragraph{Memorized noise.} Noisy labels invert accuracy-on-the-line only when models interpolate them \citep{sanyal2025}. Only 31 of the 300 models reach 99\% training accuracy, all of them tree ensembles, and 9 of them are test models. Without these 9, $\text{corr}_\text{te}$ on our selected points is $-0.89$ [$-0.93$, $-0.81$] on the remaining 51 test models. The four families that never interpolate also invert, from $-0.66$ to $-0.96$ (Table~\ref{tab:families}). The mechanism of \citet{sanyal2025} therefore does not explain the inversion.
% STALE: Error Analysis (Table 3, Figure 4, the d/g twins numbers) describes the removed twins analysis. Replace after the region-analysis rerun.
```

After:

```latex
\paragraph{Breakdown by model family.} Every family follows the overall pattern (Table~\ref{tab:families}). Pooled, its Zone A accuracy lies within 0.032 of its ID accuracy. Per tile, every family does worse on Zone A, from 0.027 for naive Bayes to 0.077 for XGBoost. On our selected points, the trend inverts within every family, from $-0.70$ for naive Bayes to $-0.97$ for decision trees and extra trees. The inversion is therefore not a contrast between families, such as trees against linear models, but holds among models of the same kind.
\paragraph{Breakdown by budget.} All three searches invert at every budget from 10 to 7,500 points (Table~\ref{tab:budgets}). At 7,500 points, 60\% of Zone A, OODSelect still finds $-0.62$, so the positive trend on all of Zone A hides a trend that runs the other way on most of it. Our Laplacian search stays within 0.12 of OODSelect at every budget from 100 to 7,500 points, and within 0.17 below 100, while its purity stays well above that of random subsets. OODSelect's purity stays within 0.15 of the random level at every budget, so it becomes more coherent only by selecting a larger share of the points.
\paragraph{Comparison with the baselines.} OODSelect's subset shares 23\% of its points with the 500 hardest points, and ours shares 13\%. Neither search simply returns the points that models get wrong. They return points whose errors grow with ID accuracy, which the hardest points capture only in part ($-0.68$ against $-0.98$).
\paragraph{Ablation of the weight $\mu$.} At $S=500$, purity is 0.13 at $\mu=0$, 0.42 at $\mu=1$, 0.77 at $\mu=3$, and 0.67 at $\mu=10$, while $\text{corr}_\text{te}$ is $-0.98$, $-0.95$, $-0.92$, and $-0.88$. Purity does not rise steadily with $\mu$. At $S=250$, it is 0.64, 0.35, and 0.79 for $\mu=1$, 3, and 10, so no single weight gives the most coherent subset at every budget. At all three budgets, every weight inverts, and $\text{corr}_\text{te}$ weakens by at most 0.10 from $\mu=0$ to $\mu=10$. The weight $\mu=3$ is therefore a reasonable fixed choice rather than a tuned optimum.
\paragraph{Ablation of the loss.} Smoothing OODSelect's weights after the search is simpler than our loss, and it also makes subsets coherent. At $S = 250$, 500, and 1,000, its purity is 0.65, 0.65, and 0.73 against 0.35, 0.77, and 0.65 for ours. However, it loses the inversion, with a $\text{corr}_\text{te}$ of $-0.55$, $-0.15$, and $+0.10$ against $-0.94$, $-0.92$, and $-0.92$ for ours. The points around OODSelect's scattered selections are not the ones on which better models fail, so coherence must be part of the objective rather than added afterwards.
\paragraph{GTV against the Laplacian.} As we expected, GTV makes the relaxed weights sharper: at $S=500$, 1\% of its final weights lie between 0.05 and 0.95, against 15\% for the Laplacian. Its subsets have a higher purity than the Laplacian's at 10 of the 12 budgets, but not at 250 and 500 points, and at $S=500$ it reaches $-0.88$ with a purity of 0.67 in 122 pieces (Table~\ref{tab:search}). Its inversions are weaker at every budget from 500 to 5,000 points, by 0.04 to 0.13 (Table~\ref{tab:budgets}).
\paragraph{Shuffled-accuracy control.} An inversion could come from the search alone, so we rerun each search with shuffled ID accuracies at $S = 250$, 500, and 1,000. The real search beats all 9 shuffled runs at its own budget for OODSelect and for GTV, and 8 of 9 for the Laplacian: at $S=1{,}000$, one shuffled Laplacian run reaches $-0.921$, just below the real run's $-0.917$. The shuffled runs have medians of $-0.15$ for OODSelect, $-0.54$ for the Laplacian, and $+0.48$ for GTV, and 3, 5, and 1 of their 9 runs fall below $-0.3$. Two shuffled Laplacian runs at $S=500$ reach $-0.56$ and $-0.67$, with purities of 0.53 and 0.58. A negative $\text{corr}_\text{te}$ alone is therefore weak evidence, so our claims rest on this comparison and on the region analysis.
\paragraph{Family-disjoint splits.} When the search selects with the tree-based families and tests on the others, OODSelect reaches $-0.87$ and ours $-0.71$. In the reverse direction, they reach $-0.94$ and $-0.93$. The inversion transfers across families, although our coherent subsets transfer less well when trees guide the search.
\paragraph{Memorized noise.} Noisy labels invert accuracy-on-the-line only when models interpolate them \citep{sanyal2025}. Only 31 of the 300 models reach 99\% training accuracy, all of them tree ensembles, and 9 of them are test models. Without these 9, $\text{corr}_\text{te}$ on our selected points is $-0.91$ [$-0.95$, $-0.85$] on the remaining 51 test models. The four families that never interpolate also invert, from $-0.70$ to $-0.97$ (Table~\ref{tab:families}). The mechanism of \citet{sanyal2025} therefore does not explain the inversion.
```

#### Edit 20 (replace), near: Error Analysis

Before:

```latex
\begin{tabular}{@{}p{3.6cm}cc@{}}
```

After:

```latex
\begin{tabular}{@{}lrccc@{}}
```

#### Edit 21 (replace), near: Error Analysis

Before:

```latex
& \textbf{Positive} & \textbf{Negative} \\
```

After:

```latex
\textbf{Points} & \textbf{n} & \textbf{Positives} & \textbf{Unmatched} & $\boldsymbol{\eta}$ \\
```

#### Edit 22 (replace), near: Error Analysis

Before:

```latex
Regions (points) & 3 (219) & 4 (207) \\
$\text{corr}_\text{te}$ & $-$.49 & $-$.55 \\
Input distance $d$ & .09 (.17) & .15 (.16) \\
Strongest models' $p$(flood) & .47 / .48 & .58 / .58 \\
Share of positives & .67 / .70 & .10 / .74 \\
Strongest models' accuracy & .39 / .64 & .31 / .74 \\
Accuracy gap $g$ & \textbf{+.25} ($-$.10) & \textbf{+.43} (.02) \\
```

After:

```latex
\multicolumn{5}{@{}l}{\textit{Positive regions}} \\
Levy & 114 & 1.00 / .35 & .00 & .65 \\
Dixie/Levy & 41 & .93 / .32 & .00 & .70 \\
\multicolumn{5}{@{}l}{\textit{Negative regions}} \\
Liberty & 123 & .06 / .72 & .00 & .70 \\
Hamilton & 54 & .35 / .60 & .09 & .67 \\
Palm Beach & 26 & .00 / .45 & .00 & .45 \\
Martin & 23 & .00 / .74 & .00 & .74 \\
\midrule
All regions & 381 & .47 / .53 & .01 & .66 \\
Other Zone A points & 12,031 & .52 / .51 & .05 & .25 \\
Zone AE baseline & 10,785 & .55 / .53 & .05 & .30 \\
```

#### Edit 23 (replace), near: Error Analysis

Before:

```latex
\caption{Positive and negative regions. Pairs are region / twins. $\text{corr}_\text{te}$ uses each kind's points and the 60 test models. In parentheses, the median over pooled reference patches: Zone AE patches for $d$ and Zone A patches for $g$.}
```

After:

```latex
\caption{Matched inputs (Q4). Positives is the share of positives among the points and among their matches. Unmatched is the share of points with no Zone AE point within the caliper $\epsilon$, which is .05 for Zone AE points by construction. $\eta$ is the label disagreement (Equation~\ref{eq:disagree}). Other Zone A points are the Zone A test points outside $\mathcal{M}$.}
```

#### Edit 24 (replace), near: Two kinds of failure.

Before:

```latex
\paragraph{Two kinds of failure.} We group the 7 regions by the label of most of their points (Table~\ref{tab:errors} and Figure~\ref{fig:maps}c). In the 3 positive regions, with 219 points, 70\% of the strongest models' errors are FNs. In the 4 negative regions, with 207 points, 89\% of their errors are FPs. Both kinds invert across the test models ($-0.49$ and $-0.55$), and so do the 51 test models that do not interpolate ($-0.45$ and $-0.47$). The failures therefore run in both directions, and neither kind of region explains the inversion alone.
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_twins_v3.pdf}
\caption{Input distance $d$ against accuracy gap $g$ for the 7 regions, sized by points. Gray dots are random patches of Zone AE and Zone A points of the same sizes, each with its own twins. Covariate shift would place regions to the right, and concept shift places them high at an ordinary distance.}
\label{fig:twins}
\end{figure}
\paragraph{Same inputs, different labels.} The regions are not unfamiliar land (Figure~\ref{fig:twins}). Pooled, they differ from their twins by $d = 0.12$ [0.08, 0.18] standard deviations per feature, which is lower than 95.5\% of pooled Zone AE patches (median 0.17). The regions are closer to their twins than ordinary Zone AE land is to its own. The strongest models also predict the same flood probability for the regions and their twins (0.52 and 0.53). Yet their accuracy is 0.69 on the twins and 0.35 on the regions, a gap of $g = +0.34$ [0.14, 0.50]. This gap exceeds all 200 pooled patches of either kind (medians $-0.005$ for Zone AE and $-0.04$ for Zone A), and it holds for both kinds of region ($+0.25$ and $+0.43$). Since the inputs and predictions match, the gap comes from the labels. In the negative regions, 10\% of the points are positives, against 74\% of their twins, which carry Zone AE labels. The positive column is dominated by Hendry and Palm Beach, whose points are half positives (below). In the other two positive regions, 83\% of the points are positives, against only 42\% of their twins. The two procedures label the same inputs differently, which is concept shift.
\paragraph{Exceptions.} Every region's $g$ is at or above the 95th percentile of its own Zone A patches. Two regions are less clear. The largest region, Hendry and Palm Beach (114 points), lies in the farmland south of Lake Okeechobee and is 94\% cultivated crops. Its inputs are the most familiar of all regions ($d=0.06$), and 96\% of its twins are positives, but only half of its own points are. The strongest models' errors on it split evenly between FNs and FPs. The strongest models are near chance on both the region and its twins (0.50 and 0.54), so its gap is small ($g=+0.04$, the 68th percentile of its Zone AE patches). The two procedures label this farmland differently, but the models also find it hard. Hamilton has the rarest inputs ($d=0.29$, the 95th percentile of its Zone AE patches), so part of its large gap ($+0.44$) may come from covariate shift.
\paragraph{Case studies.} Levy is the clearest positive region. All 63 of its points are positives, on land that is mostly evergreen forest (54\%), at a median of 0.3\,m above the nearest drainage and 1.5\,km from a mapped river. Only 29\% of its twins are positives, and the strongest models are right on 25\% of the region against 82\% of its twins ($g=+0.58$), so every error on Levy is an FN. Liberty, in the panhandle, is the clearest negative region. Only 2\% of its 107 points are positives, although it is mostly woody wetland (59\%), 0.7\,km from a mapped river, and 1.7\,m above the drainage. In contrast, 79\% of its twins are positives, and the strongest models are right on 31\% of the region against 85\% of its twins ($g=+0.54$). Almost all of their errors on Liberty are FPs. The two cases fail in opposite directions, by FNs and by FPs, on low, wooded land near streams. An approximate study draws its boundary from maps and records of past floods rather than computed flood elevations, so on such land its line can fall on either side of where a detailed study would draw it.
```

After:

```latex
\paragraph{Two kinds of failure.} We group the 6 regions by the label of most of their points (Table~\ref{tab:errors} and Figure~\ref{fig:maps}c). In the 2 positive regions, with 155 points, 98\% of the 15 strongest test models' errors are FNs. In the 4 negative regions, with 226 points, 88\% of their errors are FPs. Both kinds invert across the test models ($-0.40$ and $-0.61$), and so do the 51 test models that do not interpolate ($-0.41$ and $-0.56$). The failures therefore run in both directions, and neither kind of region explains the inversion alone.
\paragraph{Familiar inputs.} The regions are not unfamiliar land. Pooled, 1.3\% of their points have no Zone AE match within the caliper, against the 5\% of Zone AE points, and no region exceeds 10\%. Held-out Zone AE points cover the inputs of the regions, so covariate shift does not explain the failures.
\paragraph{Different labels.} The labels, however, differ. Pooled, the label disagreement of the regions is $\eta=0.66$, against 0.30 when Zone AE points are matched to each other, and every region lies above this baseline, from 0.45 for Palm Beach to 0.74 for Martin. The disagreement follows each region's label. In the positive regions, 98\% of the points are positives against 34\% of their matches, so Zone A marks floodplain where Zone AE does not. In the negative regions, 12\% of the points are positives against 66\% of their matches, so Zone A leaves out land that Zone AE would map. The other Zone A test points disagree with their matches less than Zone AE points do ($\eta=0.25$), so the disagreement is local to the regions rather than spread over Zone A. In the regions, the two procedures label the same inputs differently, which is concept shift.
\paragraph{Exceptions.} Two regions are less clear. Hamilton has the most unmatched points (9\%, against the 5\% expected), and it differs from its matches by 0.26 standard deviations in distance to the coast, so part of its failure may come from covariate shift. Palm Beach has the lowest disagreement ($\eta=0.45$), because its matches are themselves mixed, with 45\% positives, while all of its own points are negatives.
\paragraph{Case studies.} Levy is the largest positive region. All 114 of its points are positives, on woody wetland (44\%) and evergreen forest (42\%), at a median of 0.35\,m above the nearest drainage and 1.6\,km from a mapped river. Only 35\% of its matches are positives, at a median of 0.8\,m above the drainage and 1.5\,km from a river. The strongest models are right on 39\% of the region against 77\% of its matches, and every error on Levy is an FN. Liberty, in the Panhandle, is the largest negative region. Only 6\% of its 123 points are positives, although it is mostly woody wetland (55\%), 0.7\,km from a mapped river, and 1.7\,m above the drainage. In contrast, 72\% of its matches are positives, at 1.9\,m above the drainage and 0.65\,km from a river, and the strongest models are right on 36\% of the region against 83\% of its matches. Almost all of their errors on Liberty (97\%) are FPs. The two cases fail in opposite directions, by FNs and by FPs, on low, wooded land near streams. An approximate study draws its boundary from maps and records of past floods rather than computed flood elevations, so on such land its line can fall on either side of where a detailed study would draw it.
```

#### Edit 25 (replace), near: Limitations.

Before:

```latex
% STALE (limitations): the twins sentences and the 0.22 / 0.08 numbers come from the removed twins analysis. Replace after the region-analysis rerun.
\paragraph{Limitations.} All results come from one split of the rebuilt data. On an earlier version of the data, which held points on open water, three splits showed the same pattern of a small aggregate shift, strong inversions, coherent regions, and a large gap to the twins. However, the rebuild changed the test tiles (only 10 of 28 are shared), and none of the earlier regions reappears, so we cannot yet say which places are robust. The search also depends on its path, and shuffled searches can reach strong negative correlations, so a single correlation is weak evidence. The accuracy gap is partly built into the selection, since the regions are where the strongest models fail. With all 60 test models instead of the strongest 15, the gap shrinks to 0.22, because weaker models do better on the regions and worse on the twins. The twins match the inputs only approximately, with imbalances of up to 0.08 standard deviations per feature. In the data, topographic wetness is missing for 3.4\% of the Zone AE positives but almost never for negatives, which gives models a cue inside Zone AE that Zone A lacks. Land cover enters as a class code, which imposes an arbitrary order on the classes for the linear, neural, and naive Bayes models. For robustness, we study one state. For applicability, FEMA's map is a regulatory label, not observed flooding, so we show that the two procedures disagree, but not which one is right.
```

After:

```latex
\paragraph{Limitations.} All results come from one split of the rebuilt data. On an earlier version of the data, which held points on open water, three splits showed the same pattern of a small aggregate shift, strong inversions, and coherent regions. However, the rebuild changed the test tiles (only 10 of 28 are shared), and none of the earlier regions reappears, so we cannot yet say which places are robust. The search also depends on its path, and shuffled searches can reach strong negative correlations, so a single correlation is weak evidence. The label disagreement is partly built into the selection. The search picks points where better models fail, and a model trained on Zone AE fails where a point's label differs from Zone AE labels at similar inputs. The 500 hardest points, chosen by errors alone, reach $\eta=0.81$. The caliper compares points one at a time, so a region can match point by point and still differ on average. Hamilton differs from its matches by 0.26 standard deviations in distance to the coast. Zone A points within 3\,km of a region also disagree more than the baseline ($\eta=0.40$ on 226 points), so the regions may be the cores of larger areas. We report $\eta$ and the unmatched shares as point estimates without intervals, and the description of each region does not test a cause. In the data, topographic wetness is missing for 3.4\% of the Zone AE positives but almost never for negatives, which gives models a cue inside Zone AE that Zone A lacks. Land cover enters as a class code, which imposes an arbitrary order on the classes for the linear, neural, and naive Bayes models. For robustness, we study one state. For applicability, FEMA's map is a regulatory label, not observed flooding, so we show that the two procedures disagree, but not which one is right.
```


