# Rerun everything for the paper, with the final Approach (2026-10-25)

Decided by Victor on 2026-10-25. Written by Claude in a claude.ai chat. Victor comes from CS, not
GIS. Explain GIS terms in CS terms in anything you write for him.

## 0. What changed and what Victor decided

- **The Approach is final.** Approach 3.1 to 3.6 and Algorithm 1 in the local
  `CS371_Cho_Li/acl_latex.tex` (as of 2026-10-25) are Victor's final text. Treat them as the
  specification. When the code and the Approach disagree, change the code, not the Approach. Do
  not edit anything from `\section{Approach}` to `\section{Experiments}`.
- **Three things are new since the last paper numbers.**
  1. *The squared budget penalty* Ω_S(**s**) = (S − ‖**s**‖₁)² (3.2). It is already in the code
     (`--budget-penalty squared`, run on 2026-10-24, `docs/RUN-SQUARED-PENALTY-2026-10-24.md`), but
     the paper still reports the softplus numbers.
  2. *The region analysis* (3.6). Per-point caliper matching replaces the twins (d, g, reference
     patches). The twins scripts are superseded. Do not run them, and remove the twins results from
     the paper.
  3. *Graph total variation (GTV)* (3.5). This is a second spatial loss. It has never been run.
- **Victor's decision: rerun everything from the model cell on**, so that no number in the paper
  can come from an older run. The dataset is an input, not a result, so it is not rebuilt. Use the
  same dataset, features and split as the v3 cell (`docs/TASK-REBUILD-V3.md` Part C,
  `logs/2026-10-24/C_chain.sh` and `D_chain.sh`).
- **Notation.** The paper's symbols were renamed on 2026-10-25 to remove collisions
  (`docs/NOTATION-2026-10-25.md`). Use only symbols the paper defines. If you need a new one, check
  that the notation table does not already use it. Never reuse d, g or p(flood): they belonged to
  the twins.

## 1. Rules

- `export D=<today>`. Outputs go under `results/$D/` and logs under `logs/$D/`. Refit the cell into
  a new folder, `data/interim/aotil/cells_rerun_$D`. Do not change earlier results or cell folders.
- Use `.venv/bin/python`. Every new or changed function gets a test. Private characterization
  goldens may change by additions only, as on 2026-10-24, so show with a comparison script that
  every old value is unchanged.
- Do not commit or push.
- Keep the Mac plugged in with the lid open. On 2026-10-24, lid-closed sleep added 3 h 48 min to the
  wall clock. The paper's timing sentence will come from this run.
- Paper rules: see the standing rules in `docs/TODO-CLAUDE-CODE.md` (`algorithmic` package, no
  section labels or `Section~\ref`, no new bib entries; give Victor the landing-page URL instead).
  Style: American English, few em dashes, short sentences, keep Victor's tone. Write "Zone A" and
  "Zone AE" without "~". Sets are in `\mathcal`.

## 2. Claims (fixed before the run)

C1 to C5 stay as fixed in `docs/TASK-REBUILD-V3.md` §2, with the per-tile gap of 2026-10-24 added
to C1. Report each one as holds or does not hold, whatever the result. Under the squared penalty,
C4 did not hold as fixed on 2026-10-24 (smoothed-classic purity at S = 250 and 1,000). Report it
the same way if that is still true.

C6 replaces the twins version. It applies to the regions (components of at least c = 20 points) of
the spatial (Laplacian, μ = 3, k = 10) S = 500 selection:

| # | Claim | Statistic | Holds if |
|---|---|---|---|
| C6a | The regions are not unfamiliar land | Pooled share of region points with no match inside the caliper (3.6) | ≤ 0.10 (twice the 5% expected for Zone AE inputs). Report each region above 0.10 as possible covariate shift. |
| C6b | Same inputs, different labels | Pooled η of the regions (Eq. disagree) against the Zone AE baseline η | Pooled η ≥ baseline η + 0.10 |
| C7 | GTV (descriptive) | GTV S = 500: corr_te, purity, pieces, tiles, regions | No threshold. Report it. |

Readings: if C6a fails, the regions are unfamiliar land, which is covariate shift. If C6b fails,
there is no concept-shift claim. **Stop** in either case before rewriting the Abstract,
Introduction or Error Analysis, and report.

---

## Part A. Audit the code against the Approach (no new results)

Go through Approach 3.1 to 3.6, Algorithm 1 and Evaluation Metrics, statement by statement, and
find the code that implements each one. Put a table in the report with the statement, `file:line`,
and whether they agree. `docs/CODE-REVIEW-2026-10-21.md` already covers much of this. At least
check:
- the probit is clipped to [10⁻⁶, 1 − 10⁻⁶], and the correlation is Pearson;
- s = σ(θ) and θ ~ N(0, I);
- the penalty is Ω_S with weight λ w_e, where w_e rises along half a cosine wave over W steps from
  e = 0;
- full-batch Adam, with a cosine learning rate of half-period τ;
- TPE, T = 100 trials, over the five hyperparameters with the ranges in Settings;
- the validation correlation of the top-S mask after every step; patience P counts steps without a
  new best; the run keeps the last θ, and the trial with the lowest final corr_val wins;
- the kNN graph joins two points when either is among the other's k nearest, with weight 1, on
  coordinates in meters;
- R = **s**ᵀL**s**/(Sk);
- regions are the connected components of the induced subgraph with ≥ c points;
- purity uses the k nearest neighbors (directed) divided by k;
- Δ_tile and the per-tile label weights; the Fisher interval.

If the code disagrees and the fix is a bug fix, fix it and add a test. If the fix would change a
method choice, **stop** and ask Victor.

## Part B. GTV (code)

- In `aotil/oodselect_paper.py`, add `--reg {laplacian,gtv}`, with the default `laplacian`, so that
  earlier runs reproduce. GTV is R_GTV(**s**) = Σ_{(j,j′)∈ℰ} |s_j − s_j′| / (Sk) (Eq. gtv). It uses
  the same edge set and the same S·k normalization as the Laplacian term, counting each undirected
  edge exactly as the Laplacian term does, so that the two are equal on a binary **s**. Use
  autograd on `torch.abs`, whose subgradient at a tie is 0 (3.5).
- The variant folder is `spatial_gtv_mu3_knn10`. Pass `REG` through `run_oodselect_local.sh`. Write
  the choice into `result.csv` and `meta.json`.
- μ = 3 and k = 10, as for the Laplacian. Since the paper fixes μ rather than tuning it (3.3), do
  not tune μ for GTV.
- Tests:
  - on a binary **s**, GTV = Laplacian = the cut edges / (Sk);
  - the gradient matches finite differences at a point with no ties;
  - an edge with a tie contributes 0 to the gradient;
  - μ = 0 gives exactly the classic run;
  - the default reproduces the old outputs bit for bit.

## Part C. The region analysis (code): `aotil/region_match.py`

Implement Approach 3.6 exactly. A reference implementation that Claude ran in its sandbox is at
`Claude outputs/region-analysis-2026-10-25/a9_concept_reference.py`. Do not import it. It differs
from 3.6 in one way: its pool excludes only the query point's own tile, not every tile that holds a
point of the region.

**Definitions (3.6).**
- **Features.** q = 6 numeric features: every model input except `landcover` (read them from
  `FEATURES.txt` and assert that there are 6). Standardize each by its mean and standard deviation
  over 𝒟_OOD (all Zone A test points), giving z(x). Land cover must match exactly.
- **Pool.** The Zone AE test points (𝒟_ID). Leave out the ones that are missing any of the q
  features, and count them. On 2026-10-23 about 690 were missing `wetindexcat` or `hand`. These are
  missing values, not zeros.
- **Pool for a region 𝒞.** 𝒫_𝒞 is the pool points in test tiles (`tile_50km`) that hold no point
  of 𝒞.
- **Matches.** For x ∈ 𝒞, 𝒞′_x is the points of 𝒫_𝒞 with the same land cover and a Euclidean
  distance in z of at most ε, keeping the 5 closest. x is matched if 𝒞′_x is not empty.
- **Caliper ε.** Match every pool point against the other pool points in the same way, treating
  each point as a one-point region (so only its own tile is excluded), and take its distance to the
  nearest match. ε is the 95th percentile of these distances, so 5% of Zone AE points are
  unmatched. A point whose land-cover class has no candidate counts as an infinite distance.
- **Unmatched share.** The share of a set's points that are not matched.
- **η (Eq. disagree).** Over the matched points of a set, the mean of the share of each point's
  matches whose label differs from its own. Each matched point counts once.
- **Baseline.** η of the pool matched against itself in the same way (one-point regions). Its
  unmatched share is 0.05 by construction. Report the actual value.

**Nearest-neighbor search.** Use a KD-tree per land-cover class. Watch for the bug Claude hit in its
sandbox: a fixed query size of k = 200 returned no out-of-tile candidate for points in dense tiles,
so they were wrongly unmatched. Expand the query (×4 each time) until enough eligible candidates are
found or the class is exhausted.

**Sets to compute** (one output row each):
1. Each region of the Laplacian S = 500 selection (the paper's regions). Name it by its dominant
   counties, as `logs/2026-10-24/C5_regions_describe.py` does.
2. All regions pooled.
3. The Zone AE baseline.
4. Reference sets, each matched point by point (one-point regions):
   - the other Zone A test points (𝒟_OOD outside ℳ);
   - the 500 hardest points (the paper's baseline);
   - OODSelect's (μ = 0) S = 500 subset.
5. Halos: Zone A test points outside ℳ within 1, 2 and 3 km (straight-line, in meters) of a
   region's points. For these, exclude the region's tiles and the point's own tile.
6. The regions of the GTV S = 500 selection, for the report only.

**Columns for each row:**
- number of points, number matched, unmatched share, η;
- share of positives among the points, and among their matches (the mean over matched points of
  the mean match label; this gives the direction of the disagreement in 3.6);
- median match distance; number of distinct match points;
- the largest standardized mean difference between the points and their matches, and its feature
  (each match weighted 1/|𝒞′_x|);
- the top land-cover classes;
- for regions only: the 15 strongest test models' accuracy on the region and on its matches. This
  is descriptive, for the case studies, and is not in 3.6.

**Also, for "Two kinds of failure":** group the regions by the majority label (positive or negative
regions) and give, for each kind:
- the share of the 15 strongest test models' errors that are FNs and that are FPs;
- corr_te on that kind's points, over the 60 test models and over the 51 test models below 99%
  training accuracy.

**Sensitivity (report only):** ε at the 90th and 99th percentiles; at most 1 and at most 10 matches.

**Tests** (synthetic cases with known answers):
1. On a synthetic pool, exactly 5% of points (rounded) are unmatched against the pool.
2. A candidate in a tile that holds any point of the region is never a match, even when it is the
   closest.
3. A closer candidate with a different land cover is never a match.
4. With 10 candidates inside ε, the 5 closest are kept.
5. η by hand: a positive point with matches labeled 1, 0, 0 scores 2/3. Unmatched points are
   excluded.
6. A query whose 500 nearest same-class candidates all lie in excluded tiles still finds its
   out-of-tile match (the k-cap bug).
7. Standardization uses the 𝒟_OOD mean and SD, not the pool's.
8. Pool points missing a feature are dropped and counted.

**Check against the sandbox (Stop S2).** Add a flag `--exclude own-tile` that excludes only the
query point's own tile, as the reference does. Run it on the 2026-10-23 cell and the softplus
regions (`results/2026-10-23/regions/FL_band_AE_s0_spatial_mu3_knn10_S500_points.parquet`, the 7
components with id < 7). Claude's numbers:

| Set | Unmatched | η |
|---|---|---|
| ε | 1.391 standardized units | |
| Zone AE baseline | 0.05 | 0.297 |
| Other Zone A points | 0.047 | 0.254 |
| 7 regions pooled | | 0.654 |
| Liberty | 0.00 | 0.73 |
| Levy | 0.00 | 0.74 |
| Dixie | 0.00 | 0.66 |
| Hendry | 0.03 | 0.87 |
| Palm Beach | 0.00 | 0.48 |
| Hamilton | 0.10 | 0.71 |
| Hendry/Palm Beach | 0.02 | 0.48 |

Halo at 3 km (in own-tile mode, each halo point excludes only its own tile): η 0.475 on 138
points. **Stop** if ε differs by more than 0.01 or any η or unmatched
share differs by more than 0.01. The paper itself uses the default rule (3.6: all of the region's
tiles excluded).

## Part D. Rerun everything

1. **Cell.** Refit `FL_band_AE_s0` with `aotil/zoo.py` exactly as in `docs/TASK-REBUILD-V3.md` C1,
   into `data/interim/aotil/cells_rerun_$D`. Run `aotil/audit_cell.py`. Compare it with
   `data/interim/aotil/cells_v3`: point ids, labels, and every model's scores on both halves.
   **Stop** (S1) unless the audit passes and the predictions are identical.
2. **Shift (Q1).** `aotil/o1_shift.py`, `aotil/o1_tile_null.py` (seeds 1 to 3) and
   `aotil/o1_table.py` on the new cell.
3. **Selections**, all with `BUDGET_PENALTY=squared` on the new cell (the commands of
   `logs/2026-10-24/C_chain.sh`):
   - classic and spatial (Laplacian) over the full budget grid;
   - the shuffled-accuracy controls (`NULL_SEEDS="1 2 3"`) at S = 250, 500 and 1,000;
   - both family-disjoint directions;
   - μ = 1 and 10;
   - GTV over the full budget grid, and its shuffled-accuracy controls at S = 250, 500 and 1,000.
4. **Downstream** (the commands of `logs/2026-10-24/D_chain.sh`, without the twins):
   - the selection audits and smoothed-classic;
   - regions: Laplacian S = 250 and 500, classic S = 500, GTV S = 500;
   - `regions_land.py` and the region description;
   - the memorized-noise check;
   - `report_paper.py` (random subsets and the hardest points);
   - `region_match.py` (Part C);
   - `fig_aotl.py` and `fig_maps_combined.py`, in a new `results/$D/paper_figures/`. In
     `fig_maps_combined.py`, change the legend's "Flood-labeled region" and "Dry-labeled region" to
     "Positive region" and "Negative region", and label every region.
   - Do not run `region_twins_ref.py`, `region_twins_summary.py` or `fig_twins.py`.
5. **Compare with 2026-10-24.** The same code, seeds and machine should reproduce the 2026-10-24
   squared results. In the report, list every paper number that differs by more than 0.005, and
   explain the difference. Always use the new run.

## Part E. Update the paper (`CS371_Cho_Li/acl_latex.tex`)

Back up the file first to `Claude outputs/paper-before-$D-rerun/`. Victor authorizes every change
below, including in the Abstract and Introduction. Leave the Approach, Related Work, Introduction
paragraphs 1 to 4 and 6, and the team contributions as they are, except for the numbers named here.

1. **Every number.** Walk the file from the abstract to the end, including tables and captions.
   For each number, give its source file and replace it with the rerun value
   (`logs/2026-10-24/E_paper_numbers.py` is a starting point). Each `% STALE` comment marks text to
   fix. Fix it, then delete the comment.
2. **Abstract.**
   - Update −0.89, "20 pieces" and "about 300".
   - Replace the last results sentence ("The models predict the same for the regions and their
     matched points, yet the strongest models are 0.34 less accurate on the regions, because the
     two procedures label the same inputs differently.") with: "Their points have close matches
     among held-out Zone AE points, yet they disagree in label with those matches [ratio] times as
     often as Zone AE points do, because the two procedures label the same inputs differently."
     Write the ratio in words if it is close to a round number ("twice").
3. **Introduction, results paragraph.**
   - Update results 2 and 3.
   - In result 4, update 65% and 30%. Keep "as often as Zone AE points do" only if the pooled
     unmatched share is at most about 5%. Otherwise reword it to what the data show.
4. **Experiments.**
   - *Settings:* add the value of ε in standardized units and the number of Zone AE points left out
     of matching for a missing feature. Update the timing sentence from this run.
   - *Subset searches:* add GTV, e.g. "...and our method adds the Laplacian loss of
     Equation~\ref{eq:laplacian} or the GTV loss of Equation~\ref{eq:gtv}, both with $\mu=3$ and
     $k=10$."
   - *Development:* recheck the fifth decision (the μ pilot rule) on the new run. Propose, but do
     not add, a ninth decision for Victor: "Ninth, we first summarized each region by its input
     distance and accuracy gap to matched patches of Zone AE points. We replaced this with
     per-point matching, which separates unfamiliar inputs from different labels directly."
5. **Results and Analysis.**
   - Table 2: add a row "Spatial, GTV ($\mu = 3$)" under *Ours*.
   - Table 5: add a GTV column (corr_te with purity in parentheses), if it fits the column.
   - Add one short paragraph, at most three sentences, comparing GTV with the Laplacian on what the
     data show. 3.5 expects GTV to give sharper, more piecewise-constant regions; say whether it
     does.
   - Recheck every sentence that `docs/RUN-SQUARED-PENALTY-2026-10-24.md` §5 lists (items 2 to 8:
     the μ ablation, the smoothing ablation and C4, the shuffled control, the budget sentence, the
     Figure 3 caption). Rewrite each to what the new data show.
   - Never keep a claim that the numbers contradict. List every such rewrite in the report under
     "claims changed".
6. **Figures.** Copy the new figures into `CS371_Cho_Li/figures/` as `fig_aotl_v4.pdf` and
   `fig_maps_v4.pdf`, and point `\includegraphics` at them. Keep the old files. Update the
   Figure 3 caption: the number of regions, and the names if all are now labeled.
7. **Error Analysis.** Rewrite it from the Part C output. Delete Figure 4 (`fig:twins`) and every
   reference to it, but keep the PDF. Remove d, g and the twins everywhere. Structure:
   - *Table 3* (`tab:errors`), template below.
   - *Two kinds of failure.* Keep the paragraph, with the recomputed FN/FP shares and corr_te
     values.
   - *Familiar inputs.* Give the unmatched shares against the 5% expected, and name any region
     above 0.10.
   - *Different labels.* Give the pooled η against the baseline, then per region. Give the
     direction: in positive regions, Zone A marks floodplain where the matched Zone AE points do
     not, and in negative regions the reverse. The other Zone A points' η against the baseline
     shows whether the disagreement is local.
   - *Exceptions.* Name the region with the highest unmatched share and the one with the lowest η.
   - *Case studies.* Keep Levy and Liberty, if both are still regions. Compare each region with its
     matches (share of positives, land cover, height above drainage, distance to river, the
     strongest models' accuracy on each), and close with how approximate studies are drawn, as the
     current paragraph does. "Describing the regions" in 3.6 sets the scope.

   ```latex
   \begin{table}[t]
   \centering
   \small
   \setlength{\tabcolsep}{4pt}
   \begin{tabular}{@{}lrccc@{}}
   \toprule
   \textbf{Points} & \textbf{n} & \textbf{Positives} & \textbf{Unmatched} & $\boldsymbol{\eta}$ \\
   \midrule
   Liberty & 123 & .06 / .79 & .00 & .73 \\
   % ... one row per region
   \midrule
   All regions & ... & ... & ... & ... \\
   Other Zone A points & ... & ... & ... & ... \\
   Zone AE baseline & ... & ... & .05 & ... \\
   \bottomrule
   \end{tabular}
   \caption{Matched inputs (Q4). Positives is the share of positives among the points and among their matches. Unmatched is the share of points with no Zone AE point within the caliper $\epsilon$, which is .05 for Zone AE points by construction. $\eta$ is the label disagreement (Equation~\ref{eq:disagree}). Other Zone A points are the Zone A test points outside $\mathcal{M}$.}
   \label{tab:errors}
   \end{table}
   ```
   (The numbers in the template are placeholders.)
8. **Limitations.** Replace the twins sentences ("a large gap to the twins"; "With all 60 test
   models instead of the strongest 15, the gap shrinks to 0.22 ..."; "The twins match the inputs
   only approximately ...") with these drafts, filled in from Part C:
   - "The caliper compares points one at a time, so a region can match point by point and still
     differ on average. [Region] differs from its matches by [x] standard deviations in [feature]."
   - "The label disagreement is partly built into the selection. The search picks points where
     better models fail, and a model trained on Zone AE fails where a point's label differs from
     Zone AE labels at similar inputs. The 500 hardest points, chosen by errors alone, reach
     $\eta=[x]$."
   - "Zone A points within 3\,km of a region also disagree more than the baseline ($\eta=[x]$ on
     [n] points), so the regions may be the cores of larger areas."
   - "We report point estimates without intervals, and the description of each region does not
     test a cause."

   In "three splits showed the same pattern of a small aggregate shift, strong inversions, coherent
   regions, and a large gap to the twins", drop the twins clause. Keep the other Limitations
   sentences.
9. **Compile.** There should be no undefined references other than `wang2016trend` (Victor adds
   that entry himself) and no new overfull boxes. Report the page count (the limit is 8 pages
   excluding references; the body was about 12 pages before).

## Part F. Docs and report

- Write `docs/RUN-RERUN-ALL-$D.md`:
  - the Part A audit table;
  - the code changes and tests;
  - the S1 and S2 checks;
  - the claims table, C1 to C7;
  - every paper number, 2026-10-24 squared → new, and the differences from Part D step 5;
  - the region-analysis tables (all sets, kinds, sensitivity);
  - every paper edit as before → after text that Victor can paste into Overleaf, flagging what is
    more than a prose change;
  - "claims changed";
  - the proposed ninth decision;
  - the page count.
- `aotil/README.md`: add `region_match.py` and the `--reg` flag. Mark the twins scripts superseded.
- `docs/DECISION-PROVENANCE.md`: add rows for the squared penalty in the paper, the caliper
  matching (5% rule after Stuart 2010, at most 5 matches, region tiles excluded), and GTV with
  μ = 3 not tuned. Mark the twins as replaced.
- `docs/TODO-CLAUDE-CODE.md`: update the standing rules to say Approach 3.1 to 3.6 is final, and
  move this item to Done with the date and the report.

## Stops

- **S1** The refit cell does not reproduce `cells_v3`.
- **S2** `region_match.py --exclude own-tile` misses the sandbox numbers by more than 0.01.
- **C6** C6a or C6b fails. Stop before the Abstract, Introduction and Error Analysis edits.
- **Part A** A fix would change a method choice.

## Effort

About 3 hours of code and tests (most of it Part C). For compute, the 2026-10-24 budget grid took
about 21 minutes awake for classic and spatial together, plus the controls and the μ sweep. GTV adds
a third variant, so expect 1 to 2 hours awake. The paper edits and report take about 2 hours.
