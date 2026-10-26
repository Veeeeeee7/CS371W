# To-do for Claude Code

This file lists changes Victor wants made in this repo. Read it before starting any task. Work from the top down. When an item is finished, move it to Done with the date and the report it produced. Do not delete it. Victor comes from CS, not GIS, so explain GIS terms in CS terms in anything you write for him.

Last updated: 2026-10-25 (items 1 to 4 done; no open item).

## Standing rules for the paper (`CS371_Cho_Li/`)

- Overleaf is the source of truth. Victor has revised the Introduction and Related Work himself, and Approach 3.1 to 3.6 with Algorithm 1 is final (2026-10-25): it is the specification, and when the code disagrees, the code changes. Do not rewrite them. Change them only where an item below says so.
- Algorithms use the `algorithmic` package (`\REQUIRE`, `\ENSURE`, `\STATE`, `\FOR`/`\ENDFOR`, `\REPEAT`/`\UNTIL`, `\RETURN`), not `algpseudocode`. `\RETURN` already starts its own line, so never write `\STATE \RETURN`.
- No section references. Do not put `\label` on sections or subsections, and do not write `Section~\ref{...}`. Labels on equations, figures, tables, and algorithms are fine.
- New citations: do not write bib entries. Give Victor the paper's landing-page URL, and he adds the entry himself.

## Open items

None.

## Parked

- **Caveat for Limitations.** Victor removed this sentence from Approach 3.4 and wants to use it in the Limitations paragraph later: "During the search, however, the weights are soft and spread over all m points, and B_S only bounds their total from below, so the strength of the penalty can still vary with S." It describes the softplus penalty. Revisit it after item 1. If the soft total ends near S under the squared penalty, the caveat may no longer apply. *2026-10-24 rerun:* the soft total ends within 0.3 points of S in the six main runs at S = 250, 500 and 1,000, and within 6% of S in all 24 grid runs (under softplus: 262 to 11,464), so the caveat's premise ("only bounds their total from below") is gone; four control or μ-sweep runs with λ ≈ 1e-6 end far from S. See `docs/RUN-SQUARED-PENALTY-2026-10-24.md` §5 item 9. *2026-10-25:* three GTV grid runs also end far from S (S = 20: 2.0; 250: 0.5; 750: 647), for the same reason (`docs/RUN-RERUN-ALL-2026-10-25.md` §6). Victor decides.

## Done

### 4. Rerun everything with the final Approach

Decided by Victor on 2026-10-25. The brief is `docs/TASK-RERUN-ALL-2026-10-25.md` (prompt: `docs/PROMPT-RERUN-ALL-2026-10-25.md`). Approach 3.1 to 3.6 and Algorithm 1 in the local `CS371_Cho_Li/acl_latex.tex` are final and are the specification. The brief covers an audit of the code against the Approach, GTV (`--reg gtv`), the new region analysis (`aotil/region_match.py`, caliper matching, replacing the twins), a rerun of everything from the model cell on with the squared penalty, and the paper update. Notation: `docs/NOTATION-2026-10-25.md`.

**Done 2026-10-25 by Claude Code.** Report: `docs/RUN-RERUN-ALL-2026-10-25.md`. Outputs `results/2026-10-25/`, logs `logs/2026-10-25/`, refit cell `data/interim/aotil/cells_rerun_2026-10-25/`. Nothing committed. No Stop fired.
- Part A: no statement of the Approach disagrees with the code; no code was fixed (report §1, 37 rows).
- Part B: `--reg gtv` in `aotil/oodselect_paper.py` and `REG` in `run_oodselect_local.sh`; check 7 of `aotil/test_oodselect_paper.py`; the default is bit-identical to the old code; the private goldens changed by additions only.
- Part C: `aotil/region_match.py` and `aotil/test_region_match.py`; S2 reproduced the sandbox to 0.005 (ε 1.391).
- Part D: S1 passed (the refit cell is identical to `cells_v3`). 81 searches; the 60 that also ran on 2026-10-24 gave identical θ, so no paper number changed from the 2026-10-24 squared run. C1, C2, C3, C5, C6a, C6b hold; C4 does not hold as fixed (smoothing's purity at S = 250 and 1,000), as on 2026-10-24. Region analysis: pooled unmatched 0.013, η 0.664 against the Zone AE baseline 0.297. GTV at S = 500: −0.877, purity 0.671, 122 pieces.
- Part E: the paper reports this run, with GTV and the region analysis; the twins and every `% STALE` note are gone; 15 pages, body to page 12, as before. The proposed ninth decision is in the report (§9), not in the paper.

### 1. Use the squared budget penalty from the published OODSelect paper

Decided by Victor on 2026-10-24.

**What.** In `aotil/oodselect_paper.py` (`train_one`), the budget term is currently the released code's one-sided softplus:
`hp["penalty"] * warm * softplus(S - z.sum())`.
Change it to the squared penalty of Eq. 5 in Salaudeen et al. (2025):
`hp["penalty"] * warm * (S - z.sum()) ** 2`.

**Why.** The paper's Approach now states $B_S(\mathbf{s})=(S-\|\mathbf{s}\|_1)^2$. It follows the published paper, not the released code. Until the reruns below are done, the text and the reported numbers disagree.

**Keep unchanged.** The paper describes all of these, so do not change them:
- the cosine warm-up of the penalty weight (`warm`);
- full-batch Adam with CosineAnnealingLR;
- the Optuna TPE search and its number of trials;
- early stopping on the validation correlation of the top-S mask, with patience P, keeping θ from the last step;
- the top-S selection;
- the μ term.

**Watch out for scale.** The squared term is measured in points squared. At S = 500 and a soft total of 3,000 (typical under softplus), it equals 6.25 million, while the correlation lies in [−1, 1]. With Optuna's current λ range of [0.1, 100], the penalty will swamp the correlation. Options, easiest first:

| Option | Change | Difficulty | Risk |
|---|---|---|---|
| a | Eq. 5 as written, λ range unchanged | one line | The penalty dominates, and the search may ignore the correlation. |
| b | Eq. 5 as written, λ searched log-uniform over a wider range, e.g. [1e-6, 1e2] | two lines | None for the text. The λ range is ours, not the paper's. |
| c | Normalized penalty λ((S − ‖s‖₁)/S)² | two lines | Departs from Eq. 5, so the paper text would have to change. |

Recommended: b. Ask Victor before using c. Before the full reruns, do one quick check at S = 500: the soft total, the stopping epoch, and the test correlation.

**How.**
- Add a flag `--budget-penalty {softplus,squared}`. The default stays `softplus`, so earlier runs remain reproducible. Every run for the paper passes `--budget-penalty squared`, including the runs started by `run_oodselect_local.sh`. Write the choice into `meta.json` and `result.csv`.
- Tests. `aotil/test_oodselect_paper.py` checks equivalence with the authors' `TestSetFinder`, which uses softplus. Keep that test for `softplus`. Add tests for `squared`: the penalty value, the gradient by finite differences, and, under option b, that the wider λ range applies only under `squared`.
- Log the soft total ‖s‖₁ for every run. Under the squared penalty it should end near S. If it does, dividing the Laplacian by S·k means the same thing at every budget during the search, which was not true under softplus.

**Rerun (new results).** Use the same data and split as `docs/RUN-V3-REPORT-2026-10-23.md`: the v3 cells, seed 0 only, 300 models. The models do not change, so reuse the cells. Rerun every selection the paper reports:
- classic and spatial (μ = 3, k = 10) at S = 250, 500, and 1,000;
- the μ sweep (μ = 1 and 10) at the same budgets;
- the shuffled-accuracy controls (3 seeds, both variants);
- both family-disjoint directions (trees→others and others→trees), both variants.

Then rerun everything downstream:
- smoothed-classic;
- regions;
- twins and reference patches;
- the memorized-noise check (C5);
- the three figures;
- the "about 1.2Sk" edge count in Approach 3.4.

C1 (the aggregate shift), C2 (accuracy-on-the-line on all points) and the hardest-points baseline do not depend on the search, so they do not need rerunning.

**Report.** Write `docs/RUN-SQUARED-PENALTY-<date>.md`. Give every paper number old (softplus) and new (squared) in the format of `RUN-V3-REPORT-2026-10-23.md` §2, and say whether C3 to C6 still hold. **Stop** before changing any number in the paper. Victor decides.

**Effort.** About an hour for the code and tests. The searches took about 31 minutes per variant at S = 500 on the laptop, so the full set takes several hours. About an hour for the report.

**Done 2026-10-24 by Claude Code.** Report: `docs/RUN-SQUARED-PENALTY-2026-10-24.md`. Outputs `results/2026-10-24/`, logs `logs/2026-10-24/`. No paper number changed (the Stop above); nothing committed.
- Code as specified, with option b: `--budget-penalty {softplus,squared}` (default softplus); squared draws λ log-uniform over [1e-6, 100]; `run_oodselect_local.sh` passes `squared` by default (`BUDGET_PENALTY=softplus` reproduces the old runs); the choice is in `result.csv` and `meta.json`; the soft total is logged. Tests: check 6 of `aotil/test_oodselect_paper.py`; the private characterization goldens changed by additions only, every softplus array bit-identical; the full suite matches 2026-10-23 (same 3 failures and 1 error).
- Quick check at S = 500 passed: soft totals 500.0 (classic) and 500.3 (spatial), r_te −0.981 and −0.916. Above λ ≈ 1e-2 the squared term swamps the correlation; the released code's range [0.1, 100] lies entirely there.
- Rerun: 60 searches (the full grid for classic and spatial, the μ sweep, shuffled and family-disjoint controls), then smoothed-classic, regions, memorized noise, twins, the three figures and the 1.2Sk edge count (now 1.228Sk).
- **C3, C5 and C6 hold; C4 does not hold as fixed**: purity above smoothed-classic fails at S = 250 and 1,000 (0.355 vs 0.648, 0.646 vs 0.726), while smoothed-classic still loses the inversion. Other changes for Victor to decide (report §5): stronger inversions, less coherence at μ = 3 (0.77 in 31 pieces at S = 500), the pilot rule now picks μ = 3, the shuffled control weaker (17 of 18 beaten), 6 regions (5 found again; the two Hendry regions gone, Martin new), twins g = +0.41 with d at the 28.5th percentile.
- The grid's wall clock (4 h 09 min) includes 3 h 48 min of lid-closed sleep; results are unaffected.

### 2. Sync the local paper's Approach with Overleaf

**What.** In `CS371_Cho_Li/acl_latex.tex`, replace Approach 3.1–3.4 with `CS371_Cho_Li/approach-3.1-3.4-2026-10-24.tex`. This is Victor's final version, read and revised on 2026-10-24, and it matches Overleaf. The file runs from `\section{Approach}` to the end of 3.4 Laplacian Regularization. Paste it verbatim over the same span of the local file, which ends just before `\subsection{Region Analysis}`. Victor has not reviewed 3.5 Region Analysis yet, so leave it as it is.

**Also.**
- Preamble: replace `\usepackage[noend]{algpseudocode}` and the `\algrenewcommand\alglinenumber` line with `\usepackage{algorithmic}`. The algorithm then prints "end for" and numbers its lines "7:". The text says "line 7", which stays correct.
- Remove every `\label` on a section or subsection, and every `Section~\ref` or `Sections~\ref`. On 2026-10-24 the local file had nine of these references, all in Experiments and Results. Rewrite each sentence so it reads without the pointer.
- Check that `kingma2015adam` and `akiba2019optuna` are in `custom.bib`. Victor adds them to Overleaf himself.
- Compile, and check that there are no undefined references.

**Effort.** Under an hour, with no new results.

**Done 2026-10-24 by Claude Code.** No report (text only, no new results).
- The span from `\section{Approach}` to just before `\subsection{Region Analysis}` is now byte-identical to `approach-3.1-3.4-2026-10-24.tex`, plus one blank line before 3.5. The file before the sync is kept outside the repo, in Claude's scratchpad.
- Preamble: `\usepackage{algorithmic}` replaces the `algpseudocode` package and the `\algrenewcommand` line. Algorithm 1 now numbers its lines "7:" and prints "end for"; line 7 is still the loss.
- Removed 14 section/subsection labels (the 4 Approach labels went with the splice; 3.5's label went too, its text is unchanged) and the 9 `Section(s)~\ref`s. The sentences now read: "The $N=300$ models come from eight families"; "... and 60 test models." (pointer dropped); "OODSelect ($\mu=0$), and our method adds the Laplacian loss of Equation~\ref{eq:laplacian} ($\mu=3$ and $k=10$)"; "the connected pieces of $M$ (Algorithm~\ref{alg:spatial})"; "partly built in by the selection, since the search picks places where the strongest models fail"; "its warm-up, and the step it keeps, and we follow the code"; "the same rule would pick $\mu=1$." (pointer dropped); "We answer Q4 by analyzing the errors on these regions."
- `custom.bib` has `kingma2015adam` and `akiba2019optuna`. Every cited key resolves; the compile has no undefined references or citations and no overfull boxes (14 pages; the only BibTeX warning is the old empty year in `fema_sg_unit3`). Claude wrote the local `kingma2015adam` entry on 2026-10-24, before the no-bib-entries rule; landing page: https://arxiv.org/abs/1412.6980.
- Two sentences outside 3.1-3.4 now disagree with the final Approach. They were left alone because they depend on item 1: Development's "the released OODSelect code differs from its paper in the budget penalty, its warm-up, and the step it keeps, and we follow the code", and Settings' "$\lambda$ from $[0.1,100]$".

### 3. Report the shift per tile

Decided by Victor on 2026-10-24.

**Background.** The paper's aggregate shift (Q1) is the pooled gap: each model's metric on all ID test points (held-out Zone AE) minus its metric on all OOD points (Zone A), averaged over the 300 models. On the v3 data, split 1, this gap is about zero: accuracy −0.007, AUROC −0.018, F1 +0.028 (`results/2026-10-23/o1_shift.csv`). Claude found in a claude.ai chat that this is a Simpson's paradox. The two test sets cover the same 28 tiles, but in very different proportions. For example, tile 30_10 (Palm Beach County) holds 1,264 Zone AE test points and 192 Zone A points, and the models do badly there. Most Zone A points sit in Panhandle tiles where models do well. Within a tile, models are worse on Zone A in most tiles. The analysis, its reference script, and the expected numbers are in `Claude outputs/aggregate-shift-2026-10-24/` (README.md, a8_per_tile_mean.py, a8_per_tile.csv, a8_per_tile_mean.txt) and in the claude.ai Project doc claude/aggregate-shift.md. The scripts ran in Claude's sandbox and use its paths. Use them as a reference and reimplement inside `aotil/o1_shift.py`. Do not import from that folder.

Victor's decision: the paper keeps the pooled gap and adds a per-tile gap. The per-tile gap computes each gap within each test tile and takes a plain mean over the tiles (each tile counts once).

**Part 1: code (`aotil/o1_shift.py`).** For each scope row the script already writes (all, land, paired land), add per-tile columns with the prefix `tile_`. Keep every existing column and value unchanged.

Definitions, for model i and test tile t (ID_t and OOD_t are the ID and OOD test points in t):
- Predictions use the existing rule: flood iff p > 0.5, a tie at 0.5 counts as dry.
- Label weights within a tile. Let rho_ID(t) and rho_OOD(t) be the shares of positives of ID_t and OOD_t. Weight each ID positive in t by rho_OOD(t)/rho_ID(t) and each ID negative by (1 − rho_OOD(t))/(1 − rho_ID(t)). OOD points have weight 1. If a class is present in OOD_t but absent from ID_t, the weighted metric is undefined for that tile. Drop the tile for that metric and count it. (In the v3 data no tile has this. Tile 30_8 has no negatives in either half.)
- Accuracy and F1 per tile use these weights: weighted correct/total, and 2·TP_w / (2·TP_w + FP_w + FN_w). Also write the unweighted ("raw") per-tile accuracy and F1 as secondary columns. Raw per-tile F1 is biased here, see the null in Part 2.
- AUROC per tile is unweighted, because AUROC does not depend on the share of positives. Compute it only on tiles where both ID_t and OOD_t hold both classes, and record the number of tiles used (27 in the v3 data). Rank on p exactly as the existing AUROC code does.
- gap_i(t) = metric(f_i, ID_t, weighted) − metric(f_i, OOD_t).
- Columns per metric: the mean over models of the mean over tiles (the point estimate); the mean over tiles of each half's metric (the ID and OOD values for the table); a 95% interval from 2,000 bootstrap draws of whole tiles; the share of models whose tile-mean gap is > 0 ("worse"); the number of tiles whose mean-over-models gap is > 0; the median tile gap; and a Wilcoxon signed-rank p-value over the tiles' mean-over-models gaps. Each bootstrap draw resamples tiles with replacement and averages the drawn tiles' gaps, with the same draws for every metric. The per-tile metrics do not change between draws, so compute the [tile × model] gap matrix once and resample its rows.
- Also give the same columns for the 15 strongest test models (the existing strong15 rule).
- Write a per-tile table, one row per test tile: n ID and n OOD points, flood shares of each half, mean accuracy of the 300 models on the tile's points (both halves together), its dominant county, and the mean-over-models gap per metric (weighted and raw). Also write a per-family table of the tile-mean accuracy gap.

Tests (`aotil/test_o1_shift.py`, synthetic cases with known answers, plus the real cell):
1. One tile only: the per-tile gap equals the pooled gap with the same label weights.
2. Two tiles whose ID and OOD halves are identical inside each tile, but in different proportions: per-tile gap = 0 exactly, pooled gap ≠ 0.
3. Inside a tile, the ID half has flood share 0.8 and the OOD half 0.5, with the same TPR and TNR: weighted accuracy gap = 0, raw accuracy gap ≠ 0.
4. A tile missing a class in one half is dropped from AUROC and counted.
5. If every tile is drawn exactly once, the bootstrap mean equals the point estimate. A tile drawn twice counts twice.
6. A class present in OOD_t and absent from ID_t: that tile's weighted metric is NaN, dropped, and counted.

The private characterization pins (`tests/characterization/test_o1.py`) may change by additions only. Regenerate the goldens and show with a comparison script, as on 2026-10-23, that every old value is unchanged.

**Part 2: run, check, null.** Run `o1_shift.py` on the v3 cell (`data/interim/aotil/cells_v3`, `FL_band_AE_s0`, dataset `data/processed_w7/dataset_FL.parquet`), seed 0, 300 models. Outputs go to `results/<date>/` and logs to `logs/<date>/`. Regenerate the LaTeX table that produced `results/2026-10-23/o1_table.tex` (see `logs/2026-10-23/D_o1_table.log`) with a second block of per-tile rows.

Expected values (Claude's sandbox, same stored cell, scope "all", 2,000 tile draws with a different RNG, so the intervals may move by about 0.005):

| Column | Point estimate | 95% CI | Tiles > 0 | Models worse |
|---|---|---|---|---|
| accuracy, weighted | +0.0522 | [+0.0332, +0.0720] | 24/28 | 97% |
| F1, weighted | +0.0702 | [+0.0449, +0.0976] | 23/28 | 95% |
| AUROC (27 tiles) | +0.0599 | [+0.0288, +0.0917] | 22/27 | 97% |
| accuracy, raw | +0.0502 | [+0.0262, +0.0788] | 22/28 | 92% |
| F1, raw | +0.1043 | [+0.0663, +0.1490] | 24/28 | 99% |

The per-tile table should show tile 30_10: 1,264 ID and 192 OOD points, mean accuracy about 0.52. The Panhandle tiles 16_18, 18_17, 18_18 and 20_18 should show about 0.83 to 0.86.

**Stop** and report before Part 3 if any of these happens: a point estimate differs from the table by more than 0.003 (accuracy, F1) or 0.005 (AUROC); a tiles > 0 count differs; any pre-existing column changes.

Null check (no refit needed). For seeds 1, 2 and 3, within each (tile, label) group of test points, randomly reassign which points are ID and which are OOD, keeping the group's counts. Then recompute the per-tile columns with the same 300 models. This keeps where the points lie and the class mix, and removes any difference between the two zones within a tile. So it checks that the estimator reads about 0 when there is no shift. Write the results to `results/<date>/`. In Claude's sandbox, a refit version of this null gave weighted accuracy and F1 within ±0.007 of 0, AUROC within ±0.022, and raw F1 +0.035 to +0.038, which is the bias the weights remove. **Stop** if weighted accuracy or weighted F1 is more than 0.01 from 0 in any shuffle.

**Part 3: update the paper prose (`CS371_Cho_Li/acl_latex.tex`).** Victor authorizes the edits below, including in the abstract and introduction. Do not touch Approach 3.1–3.4. Do not change any number from the subset searches; the squared-penalty numbers are still Victor's decision. Every number in the drafts below is from Claude's sandbox. Replace each one with your computed value, and list each draft number against your number in the report. Style: American English, few em dashes, short sentences, keep Victor's tone. Write "Zone A" and "Zone AE" without "~". No section labels or Section~\ref. No new citations. Do not use \pi for a share of positives, because \pi is the labeling procedure in 3.1.

1. Abstract (line ~38). Replace "On 80,000 points across Florida, there is no aggregate shift (an accuracy gap of $-0.007$), but our search finds regions where" with "On 80,000 points across Florida, the pooled accuracy gap is only $-0.007$, because the test points of the two zones lie in different places. Within the same tiles, models lose 0.052 accuracy on Zone A. Our search finds regions where". Keep the rest of the sentence.
2. Introduction, results paragraph (line ~50). Replace "First, there is no shift in aggregate. Models are as accurate on Zone A as on held-out Zone AE (a gap of $-0.007$), and accuracy-on-the-line holds with a correlation of 0.85." with "First, pooling hides the shift. Pooled, models are as accurate on Zone A as on held-out Zone AE (a gap of $-0.007$), because most Zone AE test points lie where every model is weak. Within the same tiles, models lose 0.052 accuracy on Zone A, yet accuracy-on-the-line holds with a correlation of 0.85."
3. Evaluation Metrics, Shift (Q1) (line ~209). Replace the two sentences from "Accuracy depends on the share of positives" through "that a model gets right." with: "Both test sets cover the same tiles, but in different proportions, so a pooled gap mixes the shift with where the points lie. We therefore also compute the gap within each test tile and average over the $T$ test tiles, $\Delta_\text{tile}=\frac{1}{N}\sum_i\frac{1}{T}\sum_t[\text{metric}(f_i,\mathcal{D}_\text{ID}\cap\mathcal{D}_t)-\text{metric}(f_i,\mathcal{D}_\text{OOD}\cap\mathcal{D}_t)]$. Within a tile, the two sets can still differ in their share of positives, and accuracy and F1 depend on it. F1 in particular rises with the share of positives. We therefore weight the ID positives of tile $t$ by $\rho_t^\text{OOD}/\rho_t^\text{ID}$ and its ID negatives by $(1-\rho_t^\text{OOD})/(1-\rho_t^\text{ID})$, where $\rho_t$ is a set's share of positives in tile $t$. AUROC does not depend on this share, so we compute it without weights on the tiles where both sets hold both classes." After "because points in one tile are not independent.", add "For $\Delta_\text{tile}$, each draw averages the gaps of the drawn tiles." Extend the last sentence to "We also report the share of models with a positive gap, and for $\Delta_\text{tile}$ the number of tiles with a positive gap." Use a display equation if the inline one breaks the column.
4. Table 1 (tab:shift). Add a "Per tile" block under the pooled rows (accuracy, AUROC, F1; ID and OOD = the mean over tiles of each half's metric). In the caption, add: "The per-tile rows compute each gap within a test tile and average over the 28 tiles (27 for AUROC), with the ID points of each tile weighted to its OOD share of positives."
5. Results, Q1 paragraph (line ~291). Replace the whole paragraph with: "\paragraph{Pooling hides the shift (Q1).} Pooled over the test tiles, models trained on Zone AE do as well on Zone A as on held-out Zone AE (Table~\ref{tab:shift}). The accuracy gap is $-0.007$ [$-0.060$, $0.045$], and the AUROC and F1 gaps of $-0.018$ and $+0.028$ have opposite signs. However, the two test sets do not weight the tiles equally. The Zone AE points lie mostly in south Florida, where all models are weak. For example, the tile around Palm Beach holds 1,264 Zone AE points and 192 Zone A points, and the models average 0.52 accuracy there. The Zone A points lie mostly in the Panhandle, where the models average 0.83 to 0.86. Within a tile, models are less accurate on Zone A in 24 of the 28 tiles. Averaged over tiles, the accuracy gap is 0.052 [0.033, 0.072], and 97\% of models do worse on Zone A. AUROC and F1 agree, with gaps of $+0.060$ and $+0.070$, and the 15 strongest test models lose [your value]. When we shuffle which test points belong to each zone within each tile and label, the per-tile gaps stay within [your bound] of zero. Accuracy-on-the-line still holds, with $\text{corr}_\text{te}=+0.85$ on all Zone A points (Figure~\ref{fig:aotl}a). The shift is real, but a pooled measure hides it. This is the setting in which aggregation can hide failures \citep{salaudeen2025oodselect}, which is why we search for subsets."
6. Results, end of Q3 (line ~295). Replace "The two procedures look interchangeable in aggregate, yet Zone A holds coherent places where better Zone AE models do worse." with "Pooled, the two procedures look interchangeable. Yet Zone A is harder within the same tiles, and it holds coherent places where better Zone AE models do worse."
7. Breakdown by model family (line ~346) and Table tab:families. Add a per-tile accuracy-gap column to the table. Rewrite the second sentence as "Pooled, its Zone A accuracy lies within 0.032 of its ID accuracy. Per tile, [every family / N of 8 families] does worse on Zone A, from [smallest gap and family] to [largest gap and family]." Keep the rest of the paragraph.
8. Development (line ~230). Change "Seven decisions" to "Eight decisions", and add at the end of the paragraph: "Eighth, the pooled gap was near zero on the rebuilt data. Since the two test sets hold the tiles in different proportions, we added the per-tile gap after seeing this result."
9. Leave the Q1 question (line ~160) and the Limitations sentence about the earlier data (line ~402) unchanged. Grep the file for "aggregate", "no shift", "harmless", "interchangeable" and "0.007". In the report, list every hit with what you did.

Approach 3.1 defines the shift as the pooled mean. Do not edit it. If you think it now conflicts with the paper, propose one sentence for Victor in the report, for example "The shift is the mean over the $N$ models of this difference, which we compute within each test tile and then average over the tiles."

Compile and check that there are no undefined references and no new overfull boxes. Report the page count (the limit is 8 pages excluding references).

**Part 4: docs and report.**
- `aotil/README.md`: update the o1_shift.py row.
- `docs/DECISION-PROVENANCE.md`: add a row for the per-tile gap (Victor, 2026-10-24, chosen after seeing the pooled result) and mark the prevalence reweighting as replaced in the paper (keep its adjacc_ columns).
- Write `docs/RUN-PER-TILE-SHIFT-<date>.md`: what changed in the code; expected vs got; the null; the per-tile and per-family tables; every paper edit as before → after text that Victor can paste into Overleaf, flagging each number that came from the new run; the grep list; the 3.1 proposal; and the page count. Explain GIS terms in CS terms.

**Effort.** About 1–2 hours. Everything runs in seconds. Do not commit.

**Done 2026-10-24 by Claude Code.** Report: `docs/RUN-PER-TILE-SHIFT-2026-10-24.md`. Outputs `results/2026-10-24/o1_shift*.csv`, `o1_table.tex`, `o1_tile_null.csv`; logs `logs/2026-10-24/T*`. Nothing committed.
- Code: `aotil/o1_shift.py` writes 101 `tile_` columns per per-cell scope row plus `<stem>_tiles.csv` and `<stem>_families.csv`; the 60 old columns are identical to `results/2026-10-23/o1_shift.csv`. New: `aotil/o1_tile_null.py` (the null) and `aotil/o1_table.py` (Table 1 as LaTeX). Checks 7-14 in `aotil/test_o1_shift.py` pass; the private O1 goldens changed by additions only (0 problems in `Claude outputs/goldens-before-2026-10-24-tile/compare_goldens.py`; 42 pins OK). Pooled (multi-tag) rows leave `tile_` empty.
- Run: matches the sandbox to 4 decimals (accuracy +0.0522 [+0.0332, +0.0720], 24/28 tiles, 97.3% of models; AUROC +0.0599 on 27 tiles); no Stop. Strongest 15: +0.074 [+0.049, +0.100]. Per tile, every family does worse on Zone A (+0.027 naive Bayes to +0.077 XGBoost).
- Null (zones shuffled within tile and label, 3 seeds, no refit): weighted accuracy and F1 within 0.006 of 0, AUROC up to 0.0181; raw F1 +0.036 to +0.043 (the bias the weights remove); the pooled gap reads about −0.05 from the tile mix alone.
- Paper: all edits made in `CS371_Cho_Li/acl_latex.tex` (backup in `Claude outputs/paper-before-2026-10-24-tile/`). **Three draft sentences were reworded because the data does not support them** (report §6.1): Zone AE is not "mostly" in south Florida, Zone A is not "mostly" in the Panhandle, and "most Zone AE test points lie where every model is weak" overstates it. Victor kept the rewordings (2026-10-24). The families table is wrapped in `\resizebox` to fit seven columns. No undefined references, no new overfull box; 14 pages, body ends on page 12 as before (about 4 pages over the 8-page limit before and after). Still for Victor: the $T$ notation clash (trials vs test tiles) and a proposed 3.1 sentence (report §8).
