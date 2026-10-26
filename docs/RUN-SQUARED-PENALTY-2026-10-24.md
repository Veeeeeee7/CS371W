# The squared budget penalty: rerun of the paper's selections, D = 2026-10-24

Brief: `docs/TODO-CLAUDE-CODE.md` item 1 and Victor's prompt of 2026-10-24 (option b for the λ range).
Data and split: those of `docs/RUN-V3-REPORT-2026-10-23.md` (the v3 cell `FL_band_AE_s0` in
`data/interim/aotil/cells_v3/`, seed 0, 300 models; the cell was reused, not refit). Outputs:
`results/2026-10-24/`. Logs: `logs/2026-10-24/`. Earlier results and the cell folders are untouched.
Nothing is committed, and no number in the paper was changed.

**Status: done; one Stop reached as planned (no paper number changed; Victor decides).** Every
selection the paper reports was rerun with the squared penalty: classic and spatial over the full
budget grid (10 to 10,000), the μ sweep, the shuffled-accuracy controls and both family-disjoint
directions at S = 250, 500 and 1,000 (60 searches, none failed), then smoothed-classic, the regions,
the memorized-noise check, the twins with their reference patches, the three figures and the edge
count of Approach 3.4. The quick check passed (§2). **C3, C5 and C6 hold; C4 does not hold as fixed**:
one of its four conditions, purity above smoothed-classic at all three budgets, fails at S = 250 and
S = 1,000 (§3).

**In one paragraph.** With λ searched log-uniform, Eq. 5 behaves as the Approach now says: the soft
total ends at S (within 0.3 points in the six main runs at S = 250, 500, 1,000; under softplus it ended
at 262 to 11,464), so the parked Limitations caveat no longer applies. The inversions get stronger
everywhere: OODSelect −0.98 (was −0.96), ours −0.92 (was −0.89), both family-disjoint directions,
and the held-out non-interpolating models −0.91 (was −0.89). Our search is less coherent at the
same μ = 3: purity 0.77 in 31 pieces at S = 500 (was 0.86 in 20), and 0.35 and 0.65 at S = 250 and
1,000 (was 0.70 and 0.86), so smoothed-classic now has the higher purity at those two budgets,
although it still loses the inversion (−0.15 and +0.10 at S = 500 and 1,000). The shuffled-accuracy
control is weaker: shuffled runs go negative more often (median of our nine −0.54, was +0.59), and one
shuffled run of ours at S = 1,000 (−0.921) beats the real run there (−0.917). The regions are mostly
the same places: 6 regions of at least 20 points, 5 of them found again from the softplus run (Liberty,
Levy, Hamilton, Dixie, Palm Beach); the two Hendry regions, including the mixed farmland region that was
the main exception, are gone, and Martin is new. The twins result is stronger: g = +0.41 [0.34, 0.46]
(was +0.34 [0.14, 0.50]), above every reference patch, with d at the 28.5th percentile of Zone AE
patches (was the 4.5th), so the regions are as close to their twins as ordinary land, no longer
closer than almost all of it. C1, C2, the hardest points and random subsets do not depend on the
search and are unchanged (checked: identical values).

**Timing caveat.** The laptop lid was closed from 12:13 to 16:01 during the grid (`pmset -g log`:
"Clamshell Sleep" on battery; `caffeinate -i` does not prevent lid-close sleep on battery). The
processes were suspended, not killed: the results are unaffected (early stopping counts epochs, not
seconds), but the wall clock of the grid (4 h 09 min) and the `search_secs` of the grid runs include
the sleep. Awake time for the grid was about 21 minutes.

---

## Terms used here, in CS terms

- **Weights s, soft total.** The search gives each of the m = 12,531 Zone A test points a weight
  s_j = sigmoid(θ_j) in (0, 1). The **soft total** is ‖s‖₁ = Σ_j s_j, a continuous stand-in for "how
  many points are selected". The reported subset is always the S points with the largest weights.
- **Budget penalty B_S.** The loss term that ties the soft total to the budget S.
  - **softplus** (the released OODSelect code): log(1 + e^(S − ‖s‖₁)). One-sided: it pushes the total
    up when it is below S and is about 0 above S, so totals can end far above S.
  - **squared** (the published paper's Eq. 5, which Approach 3.2 now states): (S − ‖s‖₁)². Two-sided:
    it pulls the total to S from both sides. It is measured in points squared, so it can be millions,
    while the correlation it is added to lies in [−1, 1].
- **λ (lambda)** weights the penalty; **W** is the warm-up length (the penalty weight rises from 0 to λ
  over the first W steps). Optuna's TPE (a sequential model-based hyperparameter search) draws λ, W
  and the two learning rates per trial. **Option b** keeps Eq. 5 as written and widens λ's range.
- **r_te** is the paper's corr_te: the Pearson correlation, across the 60 test models the search never
  saw, between probit ID accuracy and probit accuracy on the subset. Below −0.3 counts as an inversion.
- **Pieces, purity, tiles.** Pieces are connected components of the subset on the 10-nearest-neighbour
  graph of the Zone A points (an edge joins two points when either is among the other's 10 nearest).
  Purity is the mean share of a selected point's 10 nearest neighbours that are also selected (1 for
  one solid blob, S/m for random points). A **tile** is a 50 km × 50 km grid square; the test set is
  28 tiles. A **region** is a piece of at least 20 points.
- **Twins, d, g.** Each region's points are matched one-to-one to held-out Zone AE points with the same
  land-cover class, at least 5 km away, with the closest six numeric inputs. d is the mean input
  difference per feature in standard deviations; g is the 15 strongest test models' accuracy on the
  twins minus on the region. Reference **patches** are random contiguous sets of the same size (grown
  by breadth-first search on the neighbour graph), matched the same way, which say what d and g look
  like on ordinary land.
- **GIS terms.** Zone AE: floodplain mapped by a detailed engineering study; Zone A: floodplain mapped
  by an approximate study. A county is a polygon (an area outline) from the Census; NLCD land cover is
  a 30 m raster (a pixel grid) of land-use classes; HAND is a point's height above the nearest stream.

---

## 1. The code

| File | Change |
|---|---|
| `aotil/oodselect_paper.py` | `--budget-penalty {softplus,squared}`, default `softplus`. `budget_term(total, S, kind)` returns softplus(S − total) or (S − total)². `PENALTY_RANGE` gives Optuna's λ range per term: softplus keeps the released code's uniform [0.1, 100]; squared draws λ log-uniform over [10⁻⁶, 100] (option b). `train_one` and `run_search` take `budget_penalty` (keyword, default softplus); everything else in the search is unchanged (warm-up, full-batch Adam with CosineAnnealingLR, TPE and its 100 trials, early stopping on the validation r of the top-S mask with patience 500 keeping the last step's θ, the top-S selection, the μ term). The choice is written to `result.csv` (column `budget_penalty`) and `meta.json`. The soft total is printed for every logged trial and for the chosen trial, with its stopping epoch and λ; `trials.csv` already kept it for every trial (`soft_mass`). |
| `run_oodselect_local.sh` | `BUDGET_PENALTY` (default `squared`) is passed as `--budget-penalty` to every selection, so every run for the paper uses Eq. 5. `BUDGET_PENALTY=softplus` reproduces the runs made before 2026-10-24. The header line prints the choice. |
| `aotil/test_oodselect_paper.py` | New check 6 (12 checks), see §8. |
| `tests/characterization/test_oodselect.py` (private) | Pins extended by the additions only: constants, the option table, the routine command lines, the signatures of `train_one`, `run_search` and `budget_term`; goldens regenerated, see §8. |

Under softplus the arithmetic is unchanged: the regenerated characterization goldens keep every θ
vector, mask and trials table bit for bit (§8).

---

## 2. The quick check at S = 500

The full protocol (100 trials, at most 50,000 epochs, patience 500), squared penalty, classic and
spatial (μ = 3, k = 10), on the v3 cell and seed-0 split (`logs/2026-10-24/B_quick_check.sh`, logs
`logs/2026-10-24/oodselect_paper/*_S500_local.out`). The full rerun reused these two runs.

| | Classic, softplus (2026-10-23) | Classic, squared | Spatial, softplus (2026-10-23) | Spatial, squared |
|---|---|---|---|---|
| Soft total ‖s‖₁ at the end of the chosen trial (S = 500) | 3,096.6 | **500.0** | 1,009.3 | **500.3** |
| Stopping epoch of the chosen trial | 1,560 | 2,632 | 1,887 | 879 |
| λ of the chosen trial | 27.6 | 4.5 × 10⁻⁴ | 87.1 | 4.5 × 10⁻⁴ |
| r_te [95% CI] | −0.962 [−0.98, −0.94] | **−0.981** [−0.99, −0.97] | −0.886 [−0.93, −0.82] | **−0.916** [−0.95, −0.86] |
| Purity; pieces; tiles | 0.11; 298; 27 | 0.13; 275; 25 | 0.86; 20; 11 | 0.77; 31; 15 |
| Wall clock | 5.2 min | 3.2 min | 3.2 min | 1.7 min |

**Verdict: go on.** The soft total ends at S in both runs, and the test correlation still inverts,
more strongly than under softplus.

The 100 trials show where the penalty swamps the correlation (`trials.csv`, completed trials, none
failed):

| λ | Classic: trials | median validation r | median soft total | Spatial: trials | median validation r | median soft total |
|---|---:|---:|---:|---:|---:|---:|
| 10⁻⁶ to 10⁻⁴ | 37 | −0.980 | 500.0 | 42 | −0.817 | 493.3 |
| 10⁻⁴ to 10⁻³ | 42 | −0.982 | 500.0 | 35 | −0.856 | 500.3 |
| 10⁻³ to 10⁻² | 13 | −0.976 | 500.0 | 17 | −0.852 | 499.7 |
| 10⁻² to 10⁻¹ | 4 | −0.660 | 500.0 | 2 | +0.715 | 3,296.5 |
| 10⁻¹ to 1 | 2 | −0.678 | 500.0 | 2 | +0.057 | 500.0 |
| 1 to 100 | 2 | −0.061 | 3,402.5 | 2 | +0.706 | 3,152.0 |

Above λ ≈ 10⁻² the squared term swamps the correlation: the validation r goes to about zero or turns
positive, and some runs stop early with the soft total near 3,300. TPE learned this and put 92 (classic)
and 94 (spatial) of the 100 trials below 10⁻². So the wider range is what makes Eq. 5 usable here: the
released code's range [0.1, 100] lies entirely in the swamped region, which is why option a would not
have worked.

---

## 3. Claims C3 to C6 (fixed in `docs/TASK-REBUILD-V3.md` §2, unchanged here)

Split 1 (seed 0), 300 models, 60 test models, S = 500 unless stated. "Holds" uses the thresholds fixed
in that brief. Softplus values are those of `docs/RUN-V3-REPORT-2026-10-23.md`.

| # | Statistic | Softplus (2026-10-23) | Squared (2026-10-24) | Holds? |
|---|---|---|---|---|
| C3 | Classic r_te; pieces, tiles, purity; the 3 shuffled-classic runs at S = 500 | −0.962 [−0.977, −0.936]; 298 pieces, 27 of 28 tiles, purity 0.114. Shuffled: −0.890, −0.485, +0.366 | **−0.981** [−0.989, −0.969]; 275 pieces, 25 of 28 tiles, purity 0.131. Shuffled: −0.789, −0.147, +0.160 | **Holds** (below −0.3 and below all three; the closest shuffled run is 0.192 above) |
| C4 | Spatial r_te, purity, pieces; smoothed-classic purity at S = 250, 500, 1,000 | −0.886, purity 0.864, 20 pieces (classic/5 = 59.6). Purity spatial vs smoothed: 0.695 vs 0.672, 0.864 vs 0.717, 0.865 vs 0.709 | **−0.916** [−0.949, −0.863], purity **0.768**, **31 pieces** (classic/5 = 55). Purity spatial vs smoothed: **0.355 vs 0.648**, 0.768 vs 0.649, **0.646 vs 0.726** | **Does not hold as fixed.** Three conditions hold (r_te < −0.3, purity ≥ 0.5, pieces ≤ classic/5); "purity above smoothed-classic at all three budgets" fails at S = 250 and 1,000. Smoothed-classic still loses the inversion: r_te −0.549, −0.152, +0.096 against ours −0.942, −0.916, −0.917 |
| C5 | r on the held-out test models with `acc_train_sub` < 0.99, spatial S = 500 selection; family-disjoint r at S = 500 | −0.886 [−0.934, −0.808], 51 models. Spatial trees→others −0.652, others→trees −0.795 (classic −0.868, −0.869) | **−0.910** [−0.948, −0.846], 51 models. Spatial trees→others −0.710, others→trees −0.935 (classic −0.870, −0.944) | **Holds** |
| C6 | Pooled d and g (strongest 15 test models) against the pooled reference patches | g = +0.336 [0.139, 0.502], 100th percentile of both kinds; d = 0.119 [0.084, 0.175], 4.5th percentile of the Zone AE patches | **g = +0.413** [0.341, 0.458], above all 200 draws of both kinds (100th percentile); **d = 0.145** [0.119, 0.204], 28.5th percentile of the Zone AE patches | **Holds** (g above the 95th percentile of both kinds; d not above the 95th percentile of the Zone AE patches) |

The Stop of that brief (fewer than 3 regions of at least 20 land points) did not fire: 6 regions.
C1 and C2 do not depend on the search: accuracy gap −0.007 [−0.060, +0.045] and r = +0.849, as on
2026-10-23. So do the hardest points (−0.682, purity 0.131; −0.622 ranked by the selection models) and
random subsets (+0.836, purity 0.040); `logs/2026-10-24/E_paper_numbers.log` prints them for both runs
and they are identical.

**Why C4 fails, as far as the data show.** Across the 100 trials of a search, a more negative
validation r goes with lower soft incoherence (the soft Laplacian term at the end, correlation +0.41
to +0.82 across trials), so the trials do not trade correlation against coherence in general. But the
search keeps the single trial with the lowest validation r, and near-ties decide coherence. At
S = 250, 13 trials end within 0.02 of the best validation r, with soft incoherence from 0.04 to 0.84;
the chosen trial (validation r −0.927) is at 0.70, and the runner-up, only 0.004 worse (−0.923), is
the most coherent of the 13, at 0.04. That is the path dependence the paper already describes, here
deciding a claim. No mechanism tied to the squared term is established; one candidate, a weaker relative pull of
μR once the soft total is S rather than thousands, does not fit S = 250, where the softplus total was
already near S (262).

---

## 4. The paper's numbers, softplus (in the paper now) and squared

`CS371_Cho_Li/acl_latex.tex` as synced on 2026-10-24, one row per number that depends on the search.
Source for every new value: `logs/2026-10-24/E_paper_numbers.log` (made by
`logs/2026-10-24/E_paper_numbers.py 2026-10-23 2026-10-24` from the result files), unless another file
is named. Numbers that do not depend on the search (the data and split, the model zoo, C1, C2, Table 2,
the hardest and random rows of Table 3, Table 4's accuracy and Interp. columns, the 31 interpolating
models) are unchanged and not repeated.

**Abstract and Introduction**

| Paper text | Softplus | Squared |
|---|---|---|
| Abstract: "regions where the correlation inverts (−0.89) in 20 pieces instead of about 300" | −0.89; 20; about 300 (298) | **−0.92; 31; 275** |
| Abstract and Introduction: "the strongest models are 0.34 less accurate on the regions" | 0.34 | **0.41** |
| Introduction: "OODSelect finds a subset of Zone A where the correlation inverts to −0.96, but it is split into about 300 pieces" | −0.96; about 300 | **−0.98; 275** |
| Introduction: "our loss reduces these to 20 pieces at a cost of 0.08 in correlation" | 20; 0.08 | **31; 0.07** (0.065) |

**Approach 3.2 and 3.4** (Victor's final text)

| Paper text | Softplus | Squared |
|---|---|---|
| 3.2: "B_S(s) = (S − ‖s‖₁)²", "pushes ‖s‖₁ toward S" | the code used softplus; totals 262 to 11,464 at the end of the six main runs at S = 250, 500, 1,000 | **now true of the code**: the totals end at S (within 0.3) in those six runs, and within 6% of S in all 24 runs of the budget grid |
| 3.4: "at least Sk edges (about 1.2Sk on our data)" | 1.231 Sk | **1.228 Sk**: still "about 1.2Sk" |
| 3.4: "about 1.3mk" nonzero entries of L | 1.31 mk | unchanged (the graph does not depend on the search) |

**Experiments: Settings and Development**

| Paper text | Softplus | Squared |
|---|---|---|
| Settings: "TPE draws ... λ from [0.1, 100]" | uniform [0.1, 100] | **log-uniform [10⁻⁶, 100]** (the chosen trials' λ: 1.6 × 10⁻⁴, 4.5 × 10⁻⁴, 6 × 10⁻⁶ for classic and 5.6 × 10⁻³, 4.5 × 10⁻⁴, 5.4 × 10⁻³ for spatial at S = 250, 500, 1,000) |
| Settings: "the main searches take 31 minutes" | 31 min for the 24 grid searches, 8 at a time | **about 21 min awake** for 22 grid searches, plus 3.2 min for the two S = 500 runs of the quick check (wall clock 4 h 09 min, of which 3 h 48 min the laptop slept with the lid closed) |
| Development: "Second, the released OODSelect code differs from its paper in the budget penalty, its warm-up, and the step it keeps, and we follow the code." | true | **false for the budget penalty** (see §5) |
| Development: "On the current data, the same rule would pick μ = 1" (the smallest μ in {1, 3, 10} with purity ≥ 0.5 at S = 500) | μ = 1 (purity 0.78) | **μ = 3** (μ = 1 gives 0.42, μ = 3 gives 0.77) |

**Results: Table 3 (S = 500), Figure 2 and the text**

| Paper text | Softplus | Squared |
|---|---|---|
| Table 3, OODSelect: r_te [CI], purity, pieces, tiles | −.96 [−.98, −.94], .11, 298, 27 | **−.98 [−.99, −.97], .13, 275, 25** |
| Table 3, Ours: the same | −.89 [−.93, −.82], .86, 20, 11 | **−.92 [−.95, −.86], .77, 31, 15** |
| Table 3, Smoothing: the same | −.22 [−.45, .04], .72, 51, 21 | **−.15 [−.39, .11], .65, 51, 18** |
| Table 3, Shuffled OODSelect (median of 3): r_te, purity, pieces, tiles | −.49, .08, 330, 28 | **−.15, .06, 355, 27** |
| Table 3, Shuffled ours (median of 3) | +.59, .15, 242, 25 | **−.56, .53, 150, 25** |
| Figure 1(b) (redrawn, `results/2026-10-24/paper_figures/fig_aotl.pdf`) | r = −0.89 | **r = −0.92** |
| Figure 2 caption: "(a) OODSelect spreads its points over 27 test tiles in 298 pieces" | 27; 298 | **25; 275** |
| Figure 2 caption: "(b) Our search groups them into 20 pieces in 11 tiles" | 20; 11 | **31; 15** |
| Figure 2 caption: "(c) The 7 regions of at least 20 points"; "The two unnamed dry-labeled regions lie in Hendry (west) and Palm Beach (east)" | 7; Hendry, Palm Beach | **6; the new figure names all six**, so the last sentence goes (`fig_maps_combined.pdf`) |
| "OODSelect finds a much stronger inversion of −0.96" | −0.96 | **−0.98** |
| "split into 298 pieces across 27 of the 28 test tiles" | 298; 27 | **275; 25** |
| "With a purity of 0.11, a selected point has about 9 unselected points among its 10 nearest neighbors" | 0.11; about 9 | **0.13; about 9** (8.7) |
| "a strong inversion of −0.89 [−0.93, −0.82]" | | **−0.92 [−0.95, −0.86]** |
| "a purity of 0.86 in 20 pieces over 11 tiles" | | **0.77; 31; 15** |
| "Seven pieces of at least 20 points hold 426 of the 500 points (85%)" | 7; 426; 85% | **Six; 381; 76%** |
| "The cost of coherence is 0.08" | 0.08 (0.075) | **0.07** (0.065) |
| "well below the gap between OODSelect and the hardest points (0.28)" | 0.28 | **0.30** |
| "the two intervals do not overlap" | [−.98, −.94] vs [−.93, −.82] | still true: [−.989, −.969] vs [−.949, −.863] |

**Performance analysis: Table 4 (r on our 500 points, all of a family's models, circular)**

| Family | Softplus | Squared |
|---|---|---|
| Decision tree | −.96 | **−.97** |
| Random forest | −.93 | **−.95** |
| Extra trees | −.94 | **−.97** |
| Hist. grad. boosting | −.94 | **−.93** |
| XGBoost | −.90 | **−.95** |
| Logistic regression | −.85 | **−.92** |
| Multilayer perceptron | −.96 | **−.96** |
| Naive Bayes | −.66 | **−.70** |

**Performance analysis: Table 5, r_te with purity in parentheses**

| S | Random purity | OODSelect, softplus | OODSelect, squared | Ours, softplus | Ours, squared |
|---:|---:|---|---|---|---|
| 100 | .01 | −.90 (.04) | **−.97 (.04)** | −.82 (.63) | **−.95 (.22)** |
| 250 | .02 | −.95 (.07) | **−.98 (.08)** | −.89 (.69) | **−.94 (.35)** |
| 500 | .04 | −.96 (.11) | **−.98 (.13)** | −.89 (.86) | **−.92 (.77)** |
| 1,000 | .08 | −.96 (.18) | **−.98 (.20)** | −.88 (.86) | **−.92 (.65)** |
| 2,500 | .20 | −.97 (.33) | **−.97 (.32)** | −.90 (.77) | **−.92 (.73)** |
| 5,000 | .40 | −.92 (.54) | **−.93 (.54)** | −.82 (.82) | **−.82 (.82)** |
| 7,500 | .60 | −.62 (.73) | **−.62 (.73)** | −.54 (.84) | **−.52 (.84)** |
| 10,000 | .80 | +.21 (.88) | **+.21 (.88)** | +.28 (.93) | **+.25 (.91)** |

Budgets of 10 to 50 points still invert (OODSelect −0.91, −0.94, −0.97; ours −0.81, −0.78, −0.87).

**Performance analysis: the text**

| Paper text | Softplus | Squared |
|---|---|---|
| "Its Zone A accuracy lies within 0.032 of its ID accuracy ... from −0.66 for naive Bayes to −0.96 for decision trees and multilayer perceptrons" | −0.66 to −0.96 (dt, mlp) | **−0.70 (nb) to −0.97 (dt, et)**; the accuracy part is unchanged |
| "Both searches invert at every budget from 10 to 7,500 points" | true | still true |
| "At 7,500 points, 60% of Zone A, OODSelect still finds −0.62" | −0.62 | **−0.62** (−0.619) |
| "Our search stays within about 0.1 of OODSelect at every budget up to 7,500" | 0.101 over 100–7,500; **0.130 at S = 10** | **0.115** over 100–7,500 (at S = 5,000); **0.164 at S = 20**. The old sentence was already slightly off below 100 (see §5) |
| "OODSelect's purity stays within 0.15 of the random level at every budget" | 0.138 | **0.143**: still true |
| "OODSelect's subset shares 36% of its points with the 500 hardest points, and ours shares 11%" | 36%; 11% | **23%; 13%** |
| "(−0.68 against −0.96)" | −0.96 | **−0.98** |
| μ ablation: "purity rises from 0.11 at μ = 0 to 0.78 at μ = 1, 0.86 at μ = 3, and 0.89 at μ = 10, while r_te is −0.96, −0.85, −0.89, and −0.89" | as quoted | **purity 0.13, 0.42, 0.77, 0.67; r_te −0.98, −0.95, −0.92, −0.88**. Not monotone in μ any more (see §5) |
| Smoothing: "its purity is 0.67, 0.72, and 0.71 against 0.69, 0.86, and 0.86 for ours" | as quoted | **0.65, 0.65, 0.73 against 0.35, 0.77, 0.65** |
| Smoothing: "its r_te is −0.31, −0.22, and +0.13 against −0.89, −0.89, and −0.88" | as quoted | **−0.55, −0.15, +0.10 against −0.94, −0.92, −0.92** |
| Shuffled: "The real search beats all 18 shuffled runs at its own budget" | 18 of 18 | **17 of 18**: at S = 1,000 a shuffled run of ours reaches −0.921, the real run −0.917 |
| Shuffled: "medians of +0.16 for OODSelect and +0.59 for ours" | +0.16; +0.59 | **−0.15; −0.54** |
| Shuffled: "4 of 9 OODSelect runs and 3 of 9 of ours fall below −0.3" | 4; 3 | **3; 5** |
| Shuffled: "One shuffled run of ours at S = 500 even reaches −0.69 with a purity of 0.67" | −0.69; 0.67 | **two do: −0.56 (purity 0.53) and −0.67 (0.58)**; at S = 1,000 one reaches −0.92 (0.46) |
| Family-disjoint: "OODSelect reaches −0.87 and ours −0.65" (trees→others) | −0.87; −0.65 | **−0.87; −0.71** |
| Family-disjoint: "In the reverse direction, they reach −0.87 and −0.80" | −0.87; −0.80 | **−0.94; −0.93** |
| Memorized noise: "r_te on our selected points is −0.89 [−0.93, −0.81] on the remaining 51 test models" | −0.89 [−0.93, −0.81] | **−0.91 [−0.95, −0.85]** (`results/2026-10-24/o5/A_interpolation.csv`) |
| Memorized noise: "The four families that never interpolate also invert, from −0.66 to −0.96" | −0.66 to −0.96 | **−0.70 to −0.97** (dt −0.97, lr −0.92, mlp −0.96, nb −0.70) |

**Error analysis: Table 6** (pairs are region / twins; in parentheses the median over pooled reference
patches: Zone AE patches for d, Zone A patches for g; `results/2026-10-24/paper_figures/region_twins_out/pooled_summary.json`
and `results/2026-10-24/o5/A_interpolation.csv`)

| | Flood-labeled, softplus | Flood-labeled, squared | Dry-labeled, softplus | Dry-labeled, squared |
|---|---|---|---|---|
| Regions (points) | 3 (219) | **2 (155)** | 4 (207) | **4 (226)** |
| r_te, 60 test models | −.49 | **−.40** | −.55 | **−.61** |
| (r, 51 held-out non-interpolating) | (−.45) | **(−.41)** | (−.47) | **(−.56)** |
| Input distance d | .09 (.17) | **.12 (.17)** | .15 (.16) | **.16 (.17)** |
| Strongest models' p(flood) | .47 / .48 | **.44 / .45** | .58 / .58 | **.57 / .56** |
| Flood-labeled share | .67 / .70 | **.98 / .43** | .10 / .74 | **.12 / .68** |
| Strongest models' accuracy | .39 / .64 | **.38 / .78** | .31 / .74 | **.33 / .76** |
| Accuracy gap g | +.25 (−.10) | **+.39 (−.06)** | +.43 (.02) | **+.43 (.04)** |

**Error analysis: the text**

| Paper text | Softplus | Squared |
|---|---|---|
| "We group the 7 regions"; "In 3 flood-labeled regions with 219 points"; "In 4 dry-labeled regions with 207 points" | 7; 3, 219; 4, 207 | **6; 2, 155; 4, 226** |
| "Both kinds invert across the test models (−0.49 and −0.55), and so do the 51 ... (−0.45 and −0.47)" | as quoted | **(−0.40 and −0.61); (−0.41 and −0.56)** |
| "d = 0.12 [0.08, 0.18] ... lower than 95.5% of pooled Zone AE patches (median 0.17)" | as quoted | **d = 0.15 [0.12, 0.20], lower than 71.5% (the 28.5th percentile), median 0.17** |
| "predict the same flood probability for the regions and their twins (0.52 and 0.53)" | 0.52; 0.53 | **0.52; 0.52** (0.518, 0.519) |
| "accuracy is 0.69 on the twins and 0.35 on the regions, a gap of g = +0.34 [0.14, 0.50]" | as quoted | **0.76; 0.35; +0.41 [0.34, 0.46]** |
| "exceeds all 200 pooled patches of either kind (medians −0.005 for Zone AE and −0.04 for Zone A)" | all 200; −0.005; −0.04 | **all 200; −0.005; −0.002** |
| "it holds for both kinds of region (+0.25 and +0.43)" | +0.25; +0.43 | **+0.39; +0.43** |
| "In the dry-labeled regions, Zone A marks 10% of the land as floodplain, while Zone AE marks 74% of the twins" | 10%; 74% | **12%; 68%** |
| "In the other two flood-labeled regions, Zone A marks 83% ... and Zone AE only 42% of the twins" | 83%; 42% | **the flood-labeled regions are now Levy and Dixie/Levy alone: 98%; 43%** |
| "Every region's g is at or above the 95th percentile of its own Zone A patches" | 7 of 7 | **5 of 6**: Dixie/Levy is at the 94th |
| "Hendry and Palm Beach (114 points) ... 94% cultivated crops ... d = 0.06 ... 96% of its twins ... 0.50 and 0.54 ... g = +0.04, the 68th percentile" | as quoted | **region gone** (§6) |
| "Hamilton has the rarest inputs (d = 0.29, the 95th percentile ...), so part of its large gap (+0.44)" | 0.29; 95th; +0.44 | **0.28; 90th; +0.39** (still the rarest inputs) |
| Levy: "All 63 of its points ... evergreen forest (54%) ... 0.3 m ... 1.5 km ... 29% of its twins ... 25% of the region against 82% ... (g = +0.58)" | as quoted | **all 114 points; woody wetland (44%); 0.3 m; 1.6 km; 43%; 39% against 80%; g = +0.41** |
| Liberty: "Only 2% of its 107 points ... woody wetland (59%), 0.7 km ... 1.7 m ... 79% of its twins ... 31% of the region against 85% ... (g = +0.54)" | as quoted | **6% of its 123 points; woody wetland (55%); 0.7 km; 1.7 m; 75%; 36% against 83%; g = +0.47** |
| Figure 3 caption: "for the 7 regions" | 7 | **6** (`fig_twins.pdf`) |

**Discussion**

| Paper text | Softplus | Squared |
|---|---|---|
| "With all 60 test models instead of the strongest 15, the gap shrinks to 0.22" | 0.22 | **0.25** |
| "imbalances of up to 0.08 standard deviations per feature" (size-weighted, region minus twin) | 0.08 (coast distance) | **0.06** (coast +0.060, river +0.058) |
| "only 10 of 28 are shared", "none of the earlier regions reappears" (v2 against v3) | | unchanged: these compare data versions, not penalties |

---

## 5. Text that no longer reads true, beyond its numbers

Not applied; listed for Victor's decision.

1. **Development, second decision.** "the released OODSelect code differs from its paper in the budget
   penalty, its warm-up, and the step it keeps, and we follow the code." Now the search follows the
   paper for the penalty (Eq. 5) and the code for the warm-up and the kept step, and the λ range is
   ours. The Settings sentence on λ needs the new range.
2. **The μ ablation.** "Most of the coherence comes with the first unit of weight" is false: μ = 1
   gives purity 0.42 at S = 500. Purity is not monotone in μ (μ = 10 gives 0.67 < 0.77 at S = 500; at
   S = 250, μ = 1 gives 0.64 and μ = 3 gives 0.35). On the other hand the pilot rule now picks μ = 3,
   the value the paper uses, so the Development sentence becomes simpler.
3. **The smoothing ablation and C4.** "it fails at every budget" no longer holds for purity: smoothing
   is more coherent than our search at S = 250 and 1,000. It still loses the inversion at S = 500 and
   1,000 (−0.15, +0.10) and weakens it at 250 (−0.55 against −0.94). C4 as fixed in the brief does not
   hold (§3).
4. **The shuffled control.** "The real search beats all 18 shuffled runs" is now 17 of 18, and the
   shuffled medians are negative. The paper's conclusion ("A negative r_te alone is therefore weak
   evidence, so our claims rest on this comparison and on the region analysis") is, if anything,
   stronger; the supporting sentences need the new numbers.
5. **The twins.** "The regions are closer to their twins than ordinary Zone AE land is to its own" was
   true at the 4.5th percentile; at the 28.5th it is "about as close as ordinary land". The concept-shift
   reading (C6) still holds.
6. **The flood-labeled column and the exceptions.** The mixed farmland region (Hendry/Palm Beach) is
   gone, so "The flood-labeled column is dominated by Hendry and Palm Beach" and the first half of the
   Exceptions paragraph describe a region that no longer exists. The new less clear cases: Palm Beach
   (the smallest gap, g = +0.18, at the 90th percentile of its Zone AE patches) and Dixie/Levy (g at the
   94th percentile of its Zone A patches).
7. **Figure 2's caption.** All six regions are named in the new panel (c); the sentence about two
   unnamed regions goes.
8. **The budget sentence.** "within about 0.1 of OODSelect at every budget up to 7,500" was already
   0.13 at S = 10 under softplus (my wording of 2026-10-24 morning; it holds from S = 100). Squared:
   0.115 over 100–7,500 and 0.164 at S = 20.
9. **The parked Limitations caveat** (`docs/TODO-CLAUDE-CODE.md`, Parked). Its premise ("B_S only
   bounds their total from below") is gone: in all 24 grid runs the total ends within 6% of S, and
   within 0.3 points at S = 250, 500 and 1,000. The four runs that end more than 6% from S are three
   shuffled-accuracy runs and one μ-sweep run (μ = 10, S = 1,000) where TPE chose λ ≈ 10⁻⁶ (soft
   totals 5,527, 309, 288 and 892), too weak to pull the total in before early stopping. A weaker caveat would still be true: the weights are soft during the search and the
   total reaches S only after the warm-up.
10. **Abstract and Introduction.** The numbers in §4; "about 300" for OODSelect's pieces would become
    275.

---

## 6. Regions (squared, spatial S = 500)

Pieces of at least 20 points: 6, holding 381 of the 500 points (76%), all on land. Files:
`results/2026-10-24/regions/`, `results/2026-10-24/o5/regions_land.csv`; log
`logs/2026-10-24/D3_regions_describe.log` (script `logs/2026-10-24/C5_regions_describe.py`). "Found
again" uses the footprint rule of the v3 brief: at least half of the region's points within 2 km of a
2026-10-23 (softplus) region's points.

| # | Region (counties) | Points | Type: flood-labeled share | Zone A polygons of its positives (largest, km²) | Ring polygons of its negatives | Land cover | Median HAND, river distance | Softplus region found again (share within 2 km) |
|---|---|---:|---|---|---:|---|---|---|
| 0 | Liberty | 123 | dry-labeled, 6% | 7 (827.7) | 24 | woody wetlands 55% | 1.7 m, 0.7 km | Liberty (0.89) |
| 1 | Levy | 114 | flood-labeled, 100% | 25 (117.6) | 0 | woody wetlands 44% | 0.3 m, 1.6 km | Levy (0.53) |
| 2 | Hamilton | 54 | dry-labeled, 35% | 10 (156.5) | 18 | evergreen forest 35% | 0.8 m, 1.9 km | Hamilton (0.78) |
| 3 | Dixie, Levy | 41 | flood-labeled, 93% | 4 (128.4) | 2 | woody wetlands 41% | 2.1 m, 3.4 km | Dixie (0.88) |
| 4 | Palm Beach | 26 | dry-labeled, 0% | 0 | 2 | woody wetlands 46% | 0.4 m, 6.2 km | Palm Beach (0.88) |
| 5 | Martin | 23 | dry-labeled, 0% | 0 | 3 | woody wetlands 61% | 0.4 m, 4.7 km | none (new) |

(A ring polygon is the Zone A polygon whose 0.5–5 km band produced a negative. HAND is the height above
the nearest drainage channel.) In the other direction, 5 of the 7 softplus regions are found again
(Liberty 0.99 of its points within 2 km, Levy 0.87, Dixie 0.62, Hamilton 0.98, Palm Beach 1.00); the
two Hendry regions (114 and 37 points, the farmland south of Lake Okeechobee) are not (nearest new
region point 67 and 85 km away).

---

## 7. Twins (squared)

`results/2026-10-24/paper_figures/region_twins_out/` (`twins_s0.csv`, `regions_summary.csv`,
`pooled_summary.json`); logs `logs/2026-10-24/D5_region_twins_*.log`. 200 Zone AE and 200 Zone A
patches per region, 1,000 bootstrap draws over regions; definitions as on 2026-10-23.

| Region | Kind | n | d | d pct, AE / A patches | Accuracy region / twins | g | g pct, AE / A patches | p(flood) region / twins | Flood-labeled share region / twins |
|---|---|---:|---:|---|---|---:|---|---|---|
| Liberty | dry | 123 | 0.125 | 20.5 / 34.5 | .36 / .83 | +0.474 | 100 / 100 | .60 / .59 | .06 / .75 |
| Levy | flood | 114 | 0.117 | 20.0 / 0 | .39 / .80 | +0.411 | 100 / 100 | .46 / .47 | 1.00 / .43 |
| Hamilton | dry | 54 | 0.275 | 90.0 / 100 | .27 / .66 | +0.389 | 100 / 100 | .52 / .50 | .35 / .61 |
| Dixie, Levy | flood | 41 | 0.139 | 41.5 / 21.5 | .35 / .70 | +0.348 | 99 / 94 | .40 / .41 | .93 / .44 |
| Palm Beach | dry | 26 | 0.104 | 23.0 / 0 | .37 / .55 | +0.179 | 89.5 / 100 | .52 / .55 | .00 / .46 |
| Martin | dry | 23 | 0.149 | 49.5 / 0 | .27 / .80 | +0.530 | 100 / 100 | .58 / .60 | .00 / .74 |

**Pooled (size-weighted over the 6 regions, 381 points):** d = 0.145 [0.119, 0.204], the 28.5th
percentile of the pooled Zone AE patches (median 0.173) and the 2nd of the Zone A patches; g = +0.413
[0.341, 0.458], above all 200 pooled patches of both kinds (medians −0.005 Zone AE, −0.002 Zone A);
accuracy 0.35 on the regions and 0.76 on the twins; p(flood) 0.518 and 0.519; with all 60 test
models, g = 0.248. Balance, region minus twin in standard deviations: coast distance +0.060, river
distance +0.058, wetness +0.045, slope +0.037, HAND −0.024, terrain variability +0.023; all within 0.1.

Figures (not copied into the paper): `results/2026-10-24/paper_figures/fig_aotl.pdf` (panel b:
r = −0.92), `fig_maps_combined.pdf` (275 and 31 pieces; panel c names all six regions; its place labels
were moved to the new regions), `fig_twins.pdf`.

---

## 8. Deviations, code and tests

### 8.1 Deviations and notes

1. **The full budget grid was rerun**, not only S = 250, 500 and 1,000: the paper's Table 5 and the
   "budgets of 10 to 50" sentence report the grid, as the 2026-10-23 run did. The controls and the μ
   sweep used S = 250, 500 and 1,000, as listed.
2. **The quick check's two S = 500 runs are the paper's runs**; the grid step skipped them.
3. **The lid was closed** from 12:13 to 16:01 (see the timing caveat above). Results are unaffected;
   timings of the grid are not usable.
4. **`run_oodselect_local.sh` now defaults to `squared`.** The Python flag defaults to softplus, as the
   doc asks; the script passes `--budget-penalty squared` unless `BUDGET_PENALTY=softplus` is set.
5. **`aotil/o5_interpolation.py --seeds 0` exits 1 after writing its CSV**, the known one-seed printout
   issue (same traceback as on 2026-10-23); the CSV is complete (260 rows).
6. **Audit warnings:** the control variants have 3 of the 12 grid budgets (by design), and
   smoothed-classic has a 2-point tie at its cut at S = 250 (one in, one out) and S = 1,000 (both in).
7. **Not done, as asked:** no number or sentence of the paper changed, no figure copied into
   `CS371_Cho_Li/`, nothing committed.

### 8.2 Code

| File | What |
|---|---|
| `aotil/oodselect_paper.py` | §1: `BUDGET_PENALTIES`, `PENALTY_RANGE`, `budget_term`, `train_one(..., budget_penalty)`, `run_search(..., budget_penalty)`, `--budget-penalty`, `budget_penalty` in `result.csv` and `meta.json`, the soft total in the printed lines |
| `run_oodselect_local.sh` | `BUDGET_PENALTY` (default `squared`), passed to every selection and printed |
| `results/2026-10-24/paper_figures/*.py` | Copies of the 2026-10-23 scripts with `RUN = 'results/2026-10-24/'`; `fig_maps_combined.py` labels the new regions |
| `logs/2026-10-24/*.sh`, `C5_regions_describe.py`, `E_paper_numbers.py` | The run chains (quick check, selections, downstream), the region description against 2026-10-23, and the side-by-side extraction of every number in §3–§7 |

### 8.3 Tests

| Test | Checks | Result |
|---|---|---|
| `aotil/test_oodselect_paper.py`, new check 6 (12 checks) | `budget_term` values at totals 510 and 490 (S = 500) for both terms; an unknown term raises; both gradients against central differences; the squared gradient is −2(S − total) on every weight; `PENALTY_RANGE`; Optuna's recorded distribution and draws for λ under each term; `train_one` with squared ends within 2 of S = 100; `main` writes the choice into `result.csv` and `meta.json` (softplus without the flag) | all pass on the synthetic cell and on the v3 cell (`logs/2026-10-24/A_test_oodselect_paper_*.log`); check 1 (equivalence with the authors' Lightning code, softplus) is kept and still skipped here because `pytorch_lightning` is not installed |
| `tests.characterization.test_oodselect` (private) | Inline pins extended by the additions only (constants, option table, routine command lines, `train_one`/`run_search`/`budget_term` signatures); goldens regenerated | 81 OK. 34 pins failed before the regeneration, all where expected; `Claude outputs/goldens-before-2026-10-24/compare_goldens_oodselect.py` shows the new goldens only add (8 `meta` entries `budget_penalty=softplus`, 8 table columns, 34 shell lines gaining the flag, the owner script's 24 new ok lines and their numbers); **every θ vector, mask and trials table under softplus is bit-identical** (`logs/2026-10-24/A_compare_goldens_oodselect.log`; old goldens in `Claude outputs/goldens-before-2026-10-24/`) |
| The full private suite | 1,208 tests | 3 failures and 1 error, **the same four as on 2026-10-23** (the brief's prose line and three v2/v3 data-pairing tests), none from this change (`logs/2026-10-24/A_tests_all.log`) |
