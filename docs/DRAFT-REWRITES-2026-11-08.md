# Draft rewrites for flags F1 to F10 (2026-11-08)

Drafts for the ten sentences flagged in `docs/RUN-REVISED-EXPERIMENTS-2026-11-07.md` §8.2. Each edit is a Before →
After pair in LaTeX, ready to paste. Before is the paper's text as it stood on 2026-11-07, and After is the draft.

**Applied 2026-11-08, at Victor's go-ahead.** The 15 recommended edits are in `CS371_Cho_Li/acl_latex.tex`, and
`CS371_Cho_Li/CHANGES-2026-11-08.md` lists them, in order, to carry to Overleaf, which does not have them yet. Backup
of the paper before them: `Claude outputs/paper-before-2026-11-08-rewrites/acl_latex.tex`. The 2 optional edits
(F1d, F5b) and the 3 alternatives (F2b, F6b, F9b) are not applied. The Befores of F1d and F5b still occur once in the
paper, so either can still be pasted in. R1 now stops at F1a, whose Before is gone, and writes nothing. Then, at
Victor's request, the maps figure was redrawn with the F9a names as `figures/fig_maps_v7.pdf`, and the paper now uses
it (the last entry of the CHANGES file).

- **Recommended** edits are the drafts. **Optional** edits add a sentence you may not want, and an **alternative**
  replaces the recommended edit above it.
- **Numbers.** Every number in an After is recomputed from the 2026-11-07 results by
  `logs/2026-11-08/R2_check_rewrite_numbers.py`: 44 checks, **ALL MATCH** (`logs/2026-11-08/R2_check_rewrite_numbers.log`).
- **Compile.** `logs/2026-11-08/R1_apply_rewrites.py` reads the blocks of this file, applies them to a copy of the
  paper and refuses unless each Before occurs exactly once.
  Applied to a copy, the 15 recommended edits compile to 10 pages, and so do the 15 plus the 2 optional ones. Every
  float stays on its page (fig:data, tab:search and tab:regions on page 6, fig:maps on 7, tab:models on 10), and the
  main text still ends on page 7. As now, the only overfull box is the one in tab:models (15.4 pt), and the only
  undefined citation is `ssurgo`. Before the edits, each of the 3 alternatives' Before also occurred once in the paper.
- **After applying.** The edited paper is byte-identical to the copy compiled above, and a fresh compile gives the same
  pages. `logs/2026-11-07/G3_check_paper_numbers.py` passes 66 of its 74 checks. The other 8 look for phrases the edits
  replaced, and `logs/2026-11-08/R3_map_g3_failures.py` matches each to its edit
  (`logs/2026-11-08/R3_map_g3_failures.log`: ALL REPLACED BY THE APPLIED EDITS).

## At a glance

| Flag | Where | What the draft does | Your call |
|---|---|---|---|
| F1 | Introduction; "Pooling hides the shift" | States that the aggregate shift is small and within chance in accuracy. Gives the AUROC and F1 intervals, and retitles the paragraph. | Optional sentence on the 15 strongest models |
| F2 | Split | Compares the tiles with the median correlation range (27 km), the block size blockCV suggests, and names the three long-range features | Second sentence optional |
| F3 | Concept shift | "seven features" → "the features we match on" | none |
| F4 | tab:models caption | Adds the duplicate drop | none |
| F5 | Appendix A | Adds the 78 duplicates; "predict" → "predicts" | Optional sentence on why repeats exist |
| F6 | OODSelect paragraph; tab:search | Drops the degenerate "Hardest" baseline. Says instead that OODSelect does not pick the 579 points every test model gets wrong. | Alternative: just delete |
| F7 | "Our loss turns the subset into regions" | Reports smoothing as it is: same inversion, fewer pieces, weaker only at S = 1,000 | How to frame the contribution (notes) |
| F8 | "Same inputs, different labels" | Deletes "Only Hamilton…" and tightens 10% to 5% | Alternative: delete only |
| F9 | tab:regions; fig:maps (c) | "Orange (east)" / "Orange (west)" | The names; the figure must be redrawn |
| F10 | Case studies | Rewritten for Hendry/Palm Beach and the new Liberty | none |

---

## F1. The shift

**Why.** Averaged over test tiles, the accuracy gap is +0.017 [−0.042, +0.067]. The interval contains 0, and the gap is
smaller than what shuffling the zones produces by chance (up to 0.022). "Lose up to 0.017" also reads as a maximum,
but 0.017 is the mean over tiles. AUROC and F1 still beat their shuffled values, but their intervals contain 0 too.

The draft keeps the paper's logic. Pooling and per-tile averaging still disagree in sign (−0.034 against +0.017), so
the Palm Beach example stays. Only the claim that a shift exists goes. The story still holds: the line holds in
aggregate and inverts on a subset.

#### F1a. Introduction (recommended)

Before:
```latex
First, we found that the shift exists and models lose up to 0.017 accuracy when evaluated on Zone A instead of Zone AE.
```

After:
```latex
First, we found that the shift is small in aggregate: averaged over test tiles, models lose 0.017 accuracy on Zone A instead of Zone AE, and the 95\% interval includes zero.
```

#### F1b. Paragraph title (recommended)

Before:
```latex
\paragraph{Pooling hides the shift.}
```

After:
```latex
\paragraph{The aggregate shift is small.}
```

#### F1c. Per-tile gaps and the shuffle (recommended)

Before:
```latex
Averaged over tiles, the gap is 0.017 [$-0.042$, 0.067] in accuracy, 0.029 in AUROC, and 0.068 in F1, and 75\% of models do worse on Zone A. When we shuffle which test points belong to each zone within each tile and label, the accuracy and AUROC gaps fall to within 0.022 of zero, but the F1 gap stays at 0.013 to 0.032.
```

After:
```latex
Averaged over tiles, the gap is 0.017 [$-0.042$, 0.067] in accuracy, 0.029 [$-0.041$, 0.081] in AUROC, and 0.068 [$-0.012$, 0.131] in F1, and 75\% of models do worse on Zone A, but all three intervals include zero. When we shuffle which test points belong to each zone within each tile and label, the accuracy gap still reaches 0.022 in magnitude, more than the observed 0.017, while the AUROC and F1 gaps reach at most 0.012 and 0.032, below the observed 0.029 and 0.068.
```

#### F1d. The strongest models (optional)

Only the strongest test models show a gap whose interval excludes 0. "Strongest" means the 15 of the 58 test models
with the highest ID AUROC, as `aotil/o1_shift.py` picks them. The subgroup was fixed before this run (it was +0.079 on
2026-11-06), but it is new to the paper's text, and a reader may see it as a subgroup chosen after the fact.

Before:
```latex
below the observed 0.029 and 0.068. Accuracy-on-the-line still holds
```

After:
```latex
below the observed 0.029 and 0.068. The 15 test models with the highest ID AUROC all do worse on Zone A, by 0.059 [0.007, 0.107] in accuracy. Accuracy-on-the-line still holds
```

**Facts** (`results/2026-11-07/o1_shift.csv`, scope all, the unweighted `_raw` columns as decided on 2026-11-04;
`o1_tile_null.csv`):

- Per-tile accuracy is +0.0172 [−0.0417, +0.0673].
- AUROC is +0.0292 [−0.0407, +0.0808], over 25 tiles: in 2 tiles one half has a single class.
- F1 is +0.0676 [−0.0119, +0.1313].
- Shuffled gaps over seeds 1 to 3:
  - accuracy −0.0225, −0.0152, −0.0043;
  - AUROC −0.0005, −0.0078, +0.0123;
  - F1 +0.0133, +0.0235, +0.0321.
- The 15 strongest models: +0.0593 [+0.0069, +0.1073], worse in 23 of 27 tiles, all 15 worse.
- The per-tile Wilcoxon p for accuracy is 0.091, not stated in the paper.

---

## F2. The tiles

**Why.** The sentence says the 50 km tiles are wider than every feature's correlation range, but the largest range is
107 km. The fix keeps the tiles.

The blockCV documentation says its suggested block size "is the median of all calculated ranges"
(https://rdrr.io/cran/blockCV/man/cv_spatial_autocor.html), and blockCV is the paper's `valavi2019blockcv`. The median
of our nine ranges is 27 km, so the 50 km tiles meet that rule. The second sentence says plainly which features they
do not cover, and why wider tiles are impractical. 110 km tiles would leave 7 test tiles instead of 27, too few for the
per-tile intervals.

#### F2a. Split (recommended)

Before:
```latex
We divide Florida into 50 km tiles, which are wider than the distance over which each feature stays spatially correlated (at most 107 km) \citep{roberts2017, valavi2019blockcv}.
```

After:
```latex
We divide Florida into 50 km tiles, which are wider than the median distance over which a feature stays spatially correlated (27 km) \citep{roberts2017, valavi2019blockcv}. Topographic wetness, distance to the coast, and land cover stay correlated over 73 to 107 km, but 110 km tiles would leave only 7 test tiles.
```

#### F2b. Split, one sentence (alternative to F2a)

Before:
```latex
We divide Florida into 50 km tiles, which are wider than the distance over which each feature stays spatially correlated (at most 107 km) \citep{roberts2017, valavi2019blockcv}.
```

After:
```latex
We divide Florida into 50 km tiles, which are wider than the median distance over which a feature stays spatially correlated (27 km) \citep{roberts2017, valavi2019blockcv}.
```

**Facts.** The ranges come from `results/2026-11-07/variogram/features.csv` and report §5. A range is the distance at
which two points' values stop being more alike than two random points' (the spatial analogue of an autocorrelation
function's decay length).

| Group | Features and ranges |
|---|---|
| Under 28 km (six features) | slope 7.8, fluvial class 9.5, HAND 14.3, terrain variability 24.5, flood frequency 27.4, river distance 27.7 km |
| 73 to 107 km (three features) | land cover 72.7 (from its pasture column), coast distance 95.2, wetness 106.9 km |

The median of the nine ranges is 27.4 km.

The test-tile counts come from `zoo.split_tiles` with seed 0 on the same EPSG:5070 grid (a planar projection in metres).
50 km tiles give 99 tiles and keep 27 as test tiles; 110 km tiles give 28 and keep 7. The blockCV page also notes
that ranges computed from raw features are not ranges of model residuals, which it prefers when they are available.
The paper's Split sentence never claimed otherwise.

---

## F3. "seven features"

**Why.** The models use nine features. "Seven" was the matched features: the six numeric ones plus exact land cover.
The draft names the set the sentence means, without a number.

#### F3a. Concept shift (recommended)

Before:
```latex
Labels also disagree within one procedure, since seven features do not determine the label.
```

After:
```latex
Labels also disagree within one procedure, since the features we match on do not determine the label.
```

---

## F4. The tab:models caption

**Why.** Since D6 the Models column also drops the 78 settings whose predictions repeat an earlier setting's
(369 − 1 − 78 = 290). The draft uses the wording of the Models paragraph ("repeat an earlier model's").

#### F4a. Caption (recommended)

Before:
```latex
Settings counts each grid, and Models counts the models left after dropping those that predict one class for more than 95\% of the ID test points.
```

After:
```latex
Settings counts each grid, and Models counts the models left after dropping those that predict one class for more than 95\% of the ID test points and those whose predictions repeat an earlier model's.
```

---

## F5. Appendix A

**Why.** 369 − 1 is 368, not 290: the 78 duplicates are missing. With one setting, the verb is "predicts".

#### F5a. Model Settings (recommended)

Before:
```latex
Of the 369 settings, 1 predict one class for more than 95\% of the ID test points, which leaves $N=290$ models.
```

After:
```latex
Of the 369 settings, 1 predicts one class for more than 95\% of the ID test points, and 78 repeat the predictions of an earlier setting, which leaves $N=290$ models.
```

#### F5b. Why repeats exist (optional)

Before:
```latex
78 repeat the predictions of an earlier setting, which leaves $N=290$ models.
```

After:
```latex
78 repeat the predictions of an earlier setting, which leaves $N=290$ models. Most repeats are shallow tree models or logistic regressions run for a few iterations. For example, a decision tree of depth 1 predicts the same for every minimum leaf size.
```

**Facts** (`data/interim/aotil/cells_v4_2026-11-07/FL_band_AE_s0_dropped.csv`).

- The one degenerate setting is a multilayer perceptron (2 hidden units, 3 epochs) that predicts flood for none of
  the ID test points.
- A duplicate has the same 0/1 prediction as an earlier setting at every ID test point and every OOD point. The first
  one in grid order is kept.

| Family | Duplicates dropped |
|---|---:|
| Decision tree | 18 |
| Logistic regression | 18 |
| XGBoost | 14 |
| Histogram gradient boosting | 10 |
| Extra trees | 9 |
| Random forest | 7 |
| Naive Bayes | 2 |

Of the 78, 48 are tree models of depth 3 or less, and 15 are logistic regressions of 3 or fewer iterations. Some
repeat a model of another family. Extra trees and random forests that use all features per split repeat decision trees
of the same depth, and XGBoost models repeat histogram gradient boosting models.

---

## F6. The hardest-points baseline

**Why.** 579 Zone A test points, 99% of them positives, are wrong for every one of the 58 test models. So the 500
hardest points all come from them, and every test model's accuracy on that subset is 0. The correlation of a
constant is undefined, and the code's stabilized version returns 0. The row compares OODSelect with a number that
carries no information.

The replacement I first considered, ranking the hardest points by the selection models (`hard_sel_r`, +0.205), is
also near-degenerate. On it, 47 of the 58 test models still have accuracy 0, so the correlation rests on 11
models. The draft therefore drops the row and keeps the point the row was meant to make, with a fact that is not
degenerate. OODSelect's own subset contains only 3 of the 579 points, and no test model
has accuracy 0 on it (median 0.31).

#### F6a. OODSelect paragraph (recommended)

Before:
```latex
OODSelect finds an inversion of $-0.97$ on test models it never saw, much stronger than on the 500 points the test models get wrong most often ($0.00$).
```

After:
```latex
OODSelect finds an inversion of $-0.97$ on test models it never saw. It is not driven by the hardest points: 579 Zone A test points, 99\% of them positives, are wrong for every test model, so a subset of them shows no trend, and OODSelect selects only 3 of them.
```

#### F6b. OODSelect paragraph, delete only (alternative to F6a)

Before:
```latex
OODSelect finds an inversion of $-0.97$ on test models it never saw, much stronger than on the 500 points the test models get wrong most often ($0.00$).
```

After:
```latex
OODSelect finds an inversion of $-0.97$ on test models it never saw.
```

#### F6c. tab:search, the row (recommended, with F6a or F6b)

Before:
```latex
Random & $+$.78 & -- & -- \\
Hardest & .00 & -- & -- \\
```

After:
```latex
Random & $+$.78 & -- & -- \\
```

#### F6d. tab:search, the caption (recommended, with F6a or F6b)

Before:
```latex
Random is the mean of 100 draws. The hardest points are those the test models get wrong most often, so their $\text{corr}_\text{te}$ is only indicative. Smoothing averages
```

After:
```latex
Random is the mean of 100 draws. Smoothing averages
```

**Facts.**

- The 579 points and their 98.8% positives come from the correctness matrix of the v4 cell (`op.load_paper_cell`), rows
  of the 58 test models.
- The OODSelect overlap uses the S = 500 weights in `results/2026-11-07/oodselect_paper/classic/`. The Laplacian
  subset holds 5 of the 579 and GTV's holds 1.
- These points are Zone A floodplain that every model trained on Zone AE calls dry, which fits the concept-shift
  story. But they are scattered (purity 0.145 for the 500 hardest, report §7.2) and 9% of them have no match, so they
  are not regions.

---

## F7. Smoothing and coherence

**Why.** Smoothing OODSelect's final weights over each point's 10 nearest neighbors no longer loses the inversion.

| Budget | Smoothing | Laplacian |
|---|---|---|
| S = 250 | −0.920, purity 0.573 | −0.935, purity 0.295 |
| S = 500 | −0.913 in 48 pieces over 17 tiles, purity 0.655 | −0.918 in 105 pieces over 22 tiles, purity 0.504 |
| S = 1,000 | −0.835, purity 0.713 | −0.919, purity 0.658 |

So "so coherence must be part of the objective" does not follow any more. The draft reports the comparison as it is.
It moves the S = 1,000 point after the GTV sentence, so that "at this budget" still means S = 500.

#### F7a. Our loss turns the subset into regions (recommended)

Before:
```latex
Smoothing OODSelect's final weights over each point's neighbors also gives a coherent subset (48 pieces), but it loses the inversion ($-0.91$), so coherence must be part of the objective. GTV inverts less ($-0.85$) and leaves 29 pieces at this budget.
```

After:
```latex
Smoothing OODSelect's final weights over each point's neighbors also gives a coherent subset, with a similar inversion ($-0.91$) in fewer pieces (48 over 17 tiles). GTV inverts less ($-0.85$) and leaves 29 pieces at this budget. At 1,000 points, smoothing inverts less than the Laplacian ($-0.83$ against $-0.92$).
```

**Notes for you.** These are not drafted, because they are framing decisions.

- **The contribution still holds against OODSelect.** The Introduction's "show that the Laplacian and Graph Total
  Variation turn scattered subsets into regions" is true against OODSelect: 293 pieces against 105 and 29. At S = 500
  smoothing does about as well as the Laplacian, so the paper no longer shows that the loss beats a post-hoc fix
  there.
- **Where the losses still win.**
  - GTV is more coherent than smoothing (29 pieces and purity 0.786, against 48 and 0.655) but inverts less (−0.85).
  - The Laplacian inverts more strongly only at S = 1,000.
- **Rounding.** −0.8348 rounds to −0.83, not −0.84. The report's "−0.835" is the same number at three decimals.

---

## F8. "Only Hamilton"

**Why.** Hamilton is no longer a region, and no region has more than 5% unmatched points (largest: Hendry 1 of 20 =
5.0%, Seminole 1 of 22 = 4.5%). By the paper's rule ("far more than 5%"), no region shows covariate shift. The draft
deletes the sentence and moves the conclusion into the sentence on the unmatched shares, tightening 10% to 5%.

#### F8a. The unmatched share (recommended)

Before:
```latex
Pooled, 1.3\% of their points have no match, against 5\% of Zone AE points, and no region exceeds 10\%.
```

After:
```latex
Pooled, 1.3\% of their points have no match, against 5\% of Zone AE points, and no region exceeds 5\%, so none shows covariate shift.
```

#### F8b. Delete "Only Hamilton" (recommended; on its own, the alternative to F8a)

Before:
```latex
so Zone A leaves out land that Zone AE would map. Only Hamilton, with 11\% unmatched points, may partly reflect covariate shift.
```

After:
```latex
so Zone A leaves out land that Zone AE would map.
```

---

## F9. The region names

**Why.** Two different regions are both "Orange". `aotil/region_match.region_name` names a region by the counties it
touches, and both lie in east Orange County, in test tile 29_14. Their centres are 5.9 km apart and their closest
points 1.8 km. They are separate pieces of the graph. The 43-point region is the eastern one (centre 80.96° W) and
the 32-point region the western one (81.02° W).

#### F9a. tab:regions (recommended)

Before:
```latex
Orange & 43 & .00 / .70 & .00 & .70 \\
Orange & 32 & .06 / .94 & .00 & .89 \\
```

After:
```latex
Orange (east) & 43 & .00 / .70 & .00 & .70 \\
Orange (west) & 32 & .06 / .94 & .00 & .89 \\
```

#### F9b. tab:regions, by land cover (alternative to F9a)

The eastern region is 91% woody wetlands, which are forested swamps. The western one is 56% emergent herbaceous
wetlands, which are open marsh, and 38% woody wetlands.

Before:
```latex
Orange & 43 & .00 / .70 & .00 & .70 \\
Orange & 32 & .06 / .94 & .00 & .89 \\
```

After:
```latex
Orange (swamp) & 43 & .00 / .70 & .00 & .70 \\
Orange (marsh) & 32 & .06 / .94 & .00 & .89 \\
```

**Notes.**

- **Other ways to name them.** Of the streams NHDPlus names, both regions are nearest to Tootoosahatchee Creek (all 32 points of the western one,
  26 of the 43 of the eastern one; the other 17 are nearest Jim Creek). A creek name would not tell them apart, and the
  other rows use county names.
- **The figure needs redrawing.** Panel (c) of fig:maps labels both regions "Orange". Its labels are hard-coded in
  `PLACE_LABELS` in `results/2026-11-07/paper_figures/fig_maps_combined.py`: the label at (−81.02, 28.48) is the
  western region and the one at (−80.96, 28.49) the eastern one. Once you pick names, I can redraw it as
  `fig_maps_v7.pdf` and leave v6 in place. *Done 2026-11-08:* `results/2026-11-08/paper_figures/fig_maps_combined.py`
  drew `fig_maps_v7.pdf` with the F9a names, and the paper now uses it.
- **Hendry/Palm Beach.** The positive region shares its name with a different negative region of the 2026-11-06 run,
  12.9 km away. That matters only when comparing runs, so nothing changes in the paper.

---

## F10. Case studies

**Why.** The paragraph describes Levy, which is no longer a region, and a Liberty of 123 points. Its closing claim,
"Both lie on low, wooded land near streams", does not fit the new largest positive region, which is cropland.

The draft keeps the paragraph's structure, the Liberty sentence's form and the closing sentence on approximate
studies. It replaces "Both lie…" with what the two regions show: each is labeled the opposite way from Zone AE points
with the same inputs.

#### F10a. Case studies (recommended)

Before:
```latex
\paragraph{Case studies.} Levy is the largest positive region. All 114 of its points are positives, on woody wetland and evergreen forest at a median of 0.35\,m above the nearest drainage, but only 35\% of its matches are. Liberty, in the Panhandle, is the largest negative region. It is mostly woody wetland 0.7\,km from a mapped river, yet only 6\% of its 123 points are positives, against 72\% of its matches. Both lie on low, wooded land near streams. An approximate study draws its boundary from maps and records of past floods rather than computed flood elevations, so on such land its line can fall on either side of where a detailed study would draw it.
```

After:
```latex
\paragraph{Case studies.} Hendry/Palm Beach, south of Lake Okeechobee, is the largest positive region. It is mostly cropland at a median of 0.7\,m above the nearest drainage, and 30 of its 31 points are positives, but only 28\% of its matches are. Liberty, in the Panhandle, is the largest negative region. It is mostly woody wetland 0.6\,km from a mapped river, yet only 2\% of its 51 points are positives, against 92\% of its matches. An approximate study draws its boundary from maps and records of past floods rather than computed flood elevations, so its line can fall on either side of where a detailed study would draw it. Here it takes in low farmland that detailed studies of similar land mostly leave out, and it leaves out wetland along small streams that they mostly map.
```

**Facts.**

| Region | Points | Positives | Matches positive | Land cover | HAND | River |
|---|---:|---:|---:|---|---:|---:|
| Hendry/Palm Beach | 31 | 30 | 28% | cultivated crops 84% | 0.7 m | see notes |
| Liberty | 51 | 1 (2%) | 92% | woody wetlands 75%, evergreen forest 22% | 1.7 m | 0.6 km |

- **Hendry/Palm Beach.** It lies south of Lake Okeechobee, in Hendry and Palm Beach counties. Its northernmost point is
  at 26.744° N, where the flow paths NHDPlus draws through the lake end (26.740° N). Its points are 1.2 to 8.5 km from
  those paths.
- **Liberty.** 44 of its 51 points are nearest a first-order stream, the smallest kind: one with no tributaries,
  usually a headwater creek. Its negatives lie 0.5 to 2.2 km from the nearest mapped floodplain.
- **"Similar land".** "Detailed studies of similar land" means the matches: Zone AE test points in other tiles with
  the same land cover and nearly the same six numeric features.

**Notes for you** (things I left out of the paper text on purpose):

- **The river distance is really the lake.** Hendry/Palm Beach is "3.8 km from a river" (report §7.1), but that
  "river" is the NHDPlus artificial path through Lake Okeechobee. An artificial path is a line drawn through a lake so
  the stream network stays connected. So the models and the matching see this land as 3.8 km from a river. Nothing
  in the nine features tells the models it lies beside Florida's largest lake, and its matches are mostly cropland a
  median of 2.9 km from a river. "Same inputs" is true in feature space, but the inputs miss the lake. This is worth a sentence in
  Limitations if you have one. The draft avoids "near streams" for this region and does not state its river distance.
- **The 156 km² polygon is typical.** All 30 positives of Hendry/Palm Beach come from one Zone A polygon of 156 km², and
  Wakulla's 28 come from one of 440 km². That looks like approximate studies drawing huge polygons. But positives are
  sampled uniformly by area, so big polygons hold most of them anyway: the median Zone A test positive lies in a
  polygon of 118 km², and 47% lie in polygons of at least 156 km². So it is not distinctive, and the draft leaves it out.
- **Wakulla.** The second positive region (28 points, all positives, 63% of matches) is woody wetland 89%, in one
  440 km² polygon. η is 0.37, the closest to the 0.31 baseline.

---

## Not drafted here

- **The Part A wording notes** (W1 to W11, D9 the exact split rule, D10 the sampling rules the text leaves out),
  report §2.2.
- **The sentences of report §8.3.** They are still true as written: the controls' −0.40, μ = 3 by a thin margin, and
  the inversion up to S = 10,000.
- **The figure labels of F9.** They wait for your choice of names.
