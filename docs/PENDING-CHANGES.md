# Pending changes for the next run (not a brief)

**2026-11-07: items 1 to 5 are now briefed in `docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md` (D1 to D7 and Parts C, F, G).
Item 1 uses option (a); item 2 uses one column per class, NaN in all of a feature's columns when it is missing,
and matching on the six numeric features only; item 3 uses 100,000 candidates.**

Victor's running list, started 2026-11-07. These are decisions he has made or is weighing before the next
full run. **Claude Code: do not implement anything here.** Victor will turn the list into a brief when he is
ready. Each item says what changes, why, what it touches, and whether it is decided.

---

## 1. Fluvial class: drop the water class

- **Status:** decided by Victor, 2026-11-07. One detail open (below); the paper text (2026-11-07) already
  assumes (a).
- **What:** `soil_fluvclass` becomes binary (0 non-fluvial, 1 fluvial). There is no third "water" class.
- **Why:** the dataset masks open water (NLCD land only), so a water class contradicts the design. The
  1,556 points in the class are land in NLCD 2021 but lie in SSURGO water map units (mostly marsh: 53%
  emergent herbaceous and 23% woody wetland; 1,414 of them flood points). Woznicki et al. (2019, p. 945)
  have the same three classes but give no reason; their design keeps open water and labels it floodplain
  (Table 1, p. 944), so the class fits theirs and not ours.
- **Open detail:** what those 1,556 points get instead.
  - (a) missing in both soil features (they have no soil to classify; models that cannot handle missing
    values fill in the training median, as for the other features). Six of these points currently have a
    flood class (three None, three Very frequent) and would become missing too.
  - (b) coded non-fluvial (0), flood class left as is.
- **Touches:** the soil coding in `pull_soil_points.py` and its tests; `data/processed_w9`; the w9 cell and
  the full suite rerun.
- **Paper (after the run):** the fluvial sentence in Sources; "fluvial class are categorical" in Data
  overview (becomes binary); the Fluvial class row of Table `tab:features` and its caption's missing counts.

## 2. One-hot encode every categorical feature

- **Status:** decided by Victor, 2026-11-07.
- **What:** in the model inputs, one-hot encode land cover (14 classes present), soil flood frequency (six
  classes, even though they are ordered) and fluvial class (binary after item 1), so each class becomes its
  own 0/1 input. The six numeric features stay as they are.
- **Why:** today every family reads a class code as a number (90 > 42), which imposes an order and spacing
  that the classes do not have.
- **Open details for the brief:**
  - one column per class, or drop one per feature (matters for logistic regression and the MLP, not trees);
  - a missing class: all-zero row, or its own "missing" column (today: training-median fill, or native NaN
    in hgb and xgb);
  - the model input count rises from 9 to about 27 columns (6 numeric + 14 + 6 + 1 or 2), which touches the
    `varying_columns` guard, the per-feature importance in the report (sum the dummies back to their
    feature), and Table `tab:models` if it states a feature count;
  - region matching (`aotil/region_match.py`) keeps working on the original nine features: land cover as the
    exact-match key, the rest in z. Decide whether the soil classes enter z as codes or as dummies.
- **Touches:** `aotil/zoo.py` (the preprocessing pipeline of every family), its tests and goldens, and a full
  rerun of the suite (all results change, not only the soil ones).
- **Paper (after the run):** Data overview already says "We one-hot encode all three" (2026-11-07); check the
  Sampling paragraph's sentence on median filling.

## 3. Sample a fixed number of candidates and keep what survives the 100 m spacing

- **Status:** decided by Victor, 2026-11-07. One number open (below).
- **What:** replace the exact 20,000 per zone and label (oversample with a 3% surplus, thin, cut to 20,000)
  with: draw a fixed number of valid candidates per zone and label (after the land mask, the 500 m to 5 km
  band, the panel and Zone D tests), thin at 100 m within each group and then over the whole state, and keep
  every survivor. No cut to a round number. The 100 m spacing itself does not change.
- **Why:** simpler to describe and no arbitrary cut; Victor does not need a round 80,000.
- **What to expect (from the 2026-10-23 build, `docs/RUN-V3-REPORT-2026-10-23.md` §3.2):** thinning kept 97%
  of positive candidates and 95% of band candidates, and the statewide pass dropped about 500 more. So
  100,000 candidates (25,000 per group) should leave about 96,000 to 98,000 points, not about 80,000. The
  groups will differ by a few hundred points (positives lose fewer than band negatives), so each zone's
  flood share will sit near, not at, 0.50.
- **Open number:** 100,000 candidates (about 97,000 points) or about 83,000 candidates (about 20,750 per
  group, about 80,000 points).
- **Touches:** `sample_points.py` (the surplus, the cut and `--exact` give way to a candidate count per group,
  and the build reports kept counts per group) and its tests; then the whole data pipeline for the new points
  (`pull_gee.py`, `pull_hydro.py` and its StreamCat pull, `pull_soil_points.py`, the merge, `audit_dataset.py`),
  the cell and a full rerun. This is a new point table, so every number in the paper changes.
- **Paper (after the run):** fill the `\textbf{XX}` in Sampling; the first Sampling sentence ("we sample 20,000
  positives ... and 20,000 negatives"); Data overview ("20,000 flood and 20,000 dry points per zone"); the
  captions of Table `tab:features` and Figure `fig:features` ("20,000 per column", "20,000 points per group");
  Table `tab:data`.

## 4. Report training tiles as the tiles that hold Zone AE training points

- **Status:** found 2026-11-07; a reporting fix, no new results.
- **What:** Table `tab:data` listed 69 training tiles, but 3 of the 69 hold only Zone A points, so the training rows
  come from 66 tiles (seed 0 on `data/processed`). The paper now says 66 (2026-11-07). Any script that writes the
  paper's split numbers (for example the coder's paper-edit scripts) should count tiles with at least one Zone AE
  training point, not all non-test tiles.
- **After the rebuild (item 3):** recompute all three tile counts; the test-tile and dropped-tile counts may move too.

## 5. Fill the Data statistics paragraph and Table `tab:data` after the rebuild

- **Status:** text and table written 2026-11-07 with `\textbf{XX}` placeholders; Victor's Data Construction
  subsection (Sources, Sampling, Split) is otherwise final and should not be reworded.
- **Fill, from the rebuilt dataset and the seed-0 split:**
  - Sampling: the number of points left after thinning. Split: the largest spatial-autocorrelation range over the
    features (variogram range, as in blockCV / Valavi et al. 2019), in km. If any feature's range is above 50 km,
    flag the Split sentence instead of filling it.
  - Data statistics: total points; % of points missing the soil features; % missing topographic wetness.
  - Table `tab:data`, per set (Train = Zone AE points of training tiles; ID test = Zone AE points of test tiles;
    OOD = Zone A points of test tiles): tiles (Train counts only tiles that hold Zone AE training points, item 4),
    points, flood share; % wetland (NLCD 90 or 95); % on soil that floods at least rarely (class Rare or above,
    among points with a class); % fluvial (among points with a class); mean ± SD for slope (degrees), terrain
    variability (m), height above drainage (m), topographic wetness (StreamCat value / 100), and both distances
    (log10 m, as the models see them). Range (min to max) and missing share are over all points. Land cover's
    type gives the number of NLCD classes present.
- **Check each qualitative claim in the paragraph still holds:** flood share near one half in every set; slope,
  terrain variability and height above drainage small with long right tails; most points in wetlands or forest
  (NLCD 9x or 4x); most points on soil rated as never flooding; Zone A points farther from the coast than Zone AE;
  missing soil features mostly in soil-survey water areas; missing wetness mostly outside NHDPlus catchments.
- **Optional figure:** `Claude outputs/fig-features-2026-11-07/fig_features.py` draws per-feature distributions
  (it reads `data/processed_w9`; point it at the rebuilt dataset).
- **Sample box-plot figure (2026-11-07, not in the paper yet):** `Claude outputs/fig-feature-boxes-2026-11-07/fig_feature_boxes.py`
  draws box plots for the six numeric features and one stacked bar per set for the three categorical ones, by set of
  the seed-0 split. If Victor adopts it, redraw it on the rebuilt dataset.

## 6. Smoothing ablation: each search with and without post-hoc smoothing

- **Status:** decided 2026-11-11 (Victor); prose written (Approach, Post-hoc Smoothing, Eq. eq:smooth); not briefed.
- **What:** apply smoothing to the final weights of OODSelect, the Laplacian and GTV, and compare each with and
  without it (corr_te on the test models, pieces, tiles, purity). Report it as an ablation in the Analysis section;
  the Smoothing row of tab:search and the smoothing sentences in Results move there.
- **Definition change:** the paper defines smoothing on the graph A, s_bar = (I + D)^-1 (I + A) s, with s the final
  sigmoid weights of the chosen trial, then the top S of s_bar. The current code (`aotil/audit_selection.py`,
  `smoothed_classic`) averages over each point and its 9 nearest neighbors (directed 10-NN including itself) and
  only runs on OODSelect. Switch it to graph A and run it on all three searches.
- **No new searches needed:** it is post-processing on the stored theta files (S = 250, 500, 1,000 at least;
  optionally the full budget grid).
- **Where results go (Victor, 2026-11-11):** OODSelect + smoothing stays in Results (tab:search row "OODSelect +
  smoothing" and the smoothing sentences of "Our loss turns the subset into regions"); the Laplacian and GTV with and
  without smoothing go to the Analysis ablation. Recompute the OODSelect + smoothing row (-.91, 48 pieces, 17 tiles)
  and the S = 1,000 comparison (-0.83) with the graph-A definition.
- **Random baseline:** removed from the paper 2026-11-11; `report_paper.py` may keep computing it, but no paper
  number reads it.

## 7. GTV as the proposed method (decided 2026-11-11 by Victor; paper switched, runs below still to do)

- **Why:** at S = 250 / 500 / 1,000, GTV is the most coherent search (purity .92 / .79 / .94; 8 / 29 / 20 pieces) and
  the only one more coherent than OODSelect + smoothing (purity .57 / .65 / .71), while still inverting (corr_te
  -.82 / -.85 / -.82). The Laplacian is less coherent than smoothing at 250 and 500 (purity .30 / .50) at about the
  same inversion. The choice rests on coherence, the paper's stated goal, not on corr_te.
- **Already in hand:** GTV searches at every budget; shuffled-accuracy controls (median +0.63, real run lower in 9 of
  9); the GTV S = 500 regions (7 regions, 383 of 500 points, every region above the eta baseline, at most 2.6%
  unmatched, `region_match.csv` rows `spatial_gtv_mu3_knn10`).
- **Needs new results, easiest first:**
  1. No new searches: the region table, case-study descriptors and pooled eta for the GTV S = 500 regions; the
     memorized-noise control (corr_te without the 3 interpolating test models) for GTV; fig:maps with GTV in panels
     (b) and (c). Existing scripts, minutes.
  2. A GTV mu sweep (mu = 1 and 10 at S = 250, 500, 1,000; 6 searches): mu = 3 was picked on the Laplacian by the
     rule "smallest mu with purity >= 0.5". If GTV at mu = 1 already passes, the rule picks mu = 1 and the GTV
     grid reruns at mu = 1 (13 searches).
  3. Family-disjoint controls for GTV (2 directions x 3 budgets; 6 searches).
- **Paper:** Intro numbers (29 pieces, cost 0.126, the regions' eta), Results paragraph "Our loss turns the subset
  into regions", the Analysis region table and case studies, fig:maps.

## 8. Numbers the paper now waits for (\textbf{XX}), 2026-11-11

- **Table tab:mu (Analysis, Regularization weight):** GTV at mu = 1 and mu = 10, S = 250, 500, 1,000 (6 searches,
  same settings as the grid): corr_te on the 58 test models and purity. Then check the rule sentence: mu = 3 is "the
  smallest weight whose purity reaches 0.5 at 500 points"; if mu = 1 already reaches it, flag it to Victor.
- **Table tab:ablation (Analysis, Smoothing):** the Laplacian and GTV S = 500 final weights with and without
  smoothing on graph A (item 6): corr_te with its 95% Fisher interval, pieces, purity. Recompute OODSelect + smoothing
  (tab:search, Results) with the same graph-A definition.
- **Table tab:regions, row Other Zone A (Results):** the Zone A test points outside the GTV S = 500 subset, matched
  as the region analysis matches (positives points / matches, unmatched share, eta).
- No new searches beyond the 6 mu runs; the rest is post-processing of stored weights and region_match.

- **2026-11-11 update to item 8:** Table tab:ablation and the OODSelect + smoothing row are filled by
  `aotil/paper_numbers.py` (smoothing on graph A, post-processing only). Still open: Table tab:mu (GTV mu = 1, 10;
  6 searches) and the Other Zone A row of tab:regions. After those runs, rerun `aotil/paper_numbers.py` and paste
  its tables.

- 2026-11-11: the GTV mu sweep is replanned at S = 500 only (one variable at a time): docs/TASK-GTV-MU-SWEEP-2026-11-11.md. It replaces the mu = 1, 10 at S = 250, 500, 1,000 plan above.
