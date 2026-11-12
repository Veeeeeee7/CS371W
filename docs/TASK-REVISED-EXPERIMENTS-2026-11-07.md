# Make the code match the revised Experiments section, rebuild the data, rerun everything (2026-11-07)

Decided by Victor on 2026-11-07. Written by Claude in a claude.ai chat. Victor comes from CS, not GIS.
Explain GIS terms in CS terms in anything you write for him.

## 0. What Victor decided

- **The paper text is the specification.** Victor synced the local `CS371_Cho_Li/acl_latex.tex` with his
  Overleaf version on 2026-11-07 (file time 22:04). The code must do exactly what these parts say:
  Approach 3.1 to 3.5 with Algorithm 1 (final since 2026-10-25), and all of Experiments: Implementation Details
  (Metrics, Correlation, Budget penalty, Graph, Search, Models), Data Construction (Sources, Sampling, Split,
  Data statistics), Region Analysis (Matching, Covariate shift, Concept shift, Describing the regions), and
  Appendix A (Model Settings, Table `tab:models`). Part A audits this statement by statement.
- **After this run, Victor rewrites the prose with Claude in claude.ai** from your report. So in the paper you
  change only numbers, the `\textbf{XX}` placeholders, figure files, and the two insertions in Part F. Put every
  change in the report as a Before/After block, and flag (do not rewrite) every sentence whose claim stops
  holding.
- **The model features are the nine of the Sources paragraph:** land cover, soil flood frequency, fluvial class,
  slope, terrain variability, height above the nearest drainage, topographic wetness, distance to the nearest
  river, distance to the coast.

### Where the text differs from the current code, and what to do

Claude compared the text with the code before writing this brief. These are decided: make the code follow the
text as described. Anything else Part A finds goes through Part A's rules.

| # | The text says | The code does now | Do this |
|---|---|---|---|
| D1 | Sampling: "we draw 100,000 candidates that meet every requirement above, 25,000 per zone and label. We then drop points in random order until the spacing holds, which leaves XX points." | Oversamples, thins, then cuts each group to exactly 20,000 (`per_domain`, `surplus`, `--exact`). | For each zone and label, draw exactly 25,000 candidates that pass every requirement (positives: inside the zone's polygons and on land; negatives: the band, the panel, not in Zone D, not on water, on land, and the masks of D10). Thin at 100 m within each group, then over the whole state, as now. Keep every survivor; no cut. Retire `per_domain`, `surplus` and `--exact`. Report candidates and survivors per group. |
| D2 | "we sample positives uniformly by area inside its polygons" | Caps any polygon at 2% of a zone's candidates (`poly_cap_frac`), which is not uniform by area. | Remove the cap. Positives are uniform by area over the land of the zone's polygons. |
| D3 | "Terrain variability is the standard deviation of the 3DEP elevations within 300 m of the point." | `tri_300`: the standard deviation inside the fixed 300 m grid cell that contains the point. | Compute the standard deviation of the 10 m 3DEP elevations within a circle of radius 300 m centered on the point (in Earth Engine, for example, reduce over each point buffered by 300 m at the DEM's native scale). New column `elev_sd_r300`; it replaces `tri_300` as the model feature. |
| D4 | "Fluvial class marks whether the area's main soil formed from river deposits." Data statistics: "XX% of points lack the soil features." | Three classes (0 non-fluvial, 1 fluvial, 2 water). | Binary: 0 non-fluvial, 1 fluvial. A point in a soil-survey water map unit (`water_unit`) gets missing values in both soil features, including the few water units that carry a flood class. |
| D5 | "We one-hot encode the three categorical and ordinal features. ... Models that cannot handle missing values fill them with the median value from the training set." | Every family reads land cover and the soil codes as numbers. | One-hot encode land cover, soil flood frequency and fluvial class: one 0/1 column per class present in the full dataset (a fixed list, so every model sees the same columns; no dropped reference column). A point missing a feature gets NaN in all of that feature's columns, and the existing pipeline handles it: the training-median imputer for the families that have one, native NaN handling for hgb and xgb. The six numeric features are unchanged (distances as log10 m). |
| D6 | Models: "We also drop models whose predictions repeat an earlier model's." | Duplicates are kept (Victor's 2026-11-06 decision, now superseded by the text). | After the degenerate drop and before the 60/20/20 split, drop a model whose 0/1 predictions on every ID test point and every OOD point equal those of a model earlier in grid order. Write the dropped settings to `<tag>_dropped.csv` with the reason. |
| D7 | Matching: "Let z(x) be the q numeric features of x ... the candidates with its land-cover class" | On w9, q = 8 (the soil codes were in z). | q = the six numeric features. Land cover stays the only exact-match key; the soil features are not matched. Recalibrate the caliper with the same 5% rule. |
| D8 | Search: "choose μ = 3 from a sensitivity analysis" | μ = 3; on w9 the purity rule picked μ = 1. | Keep μ = 3. Run the μ sweep as before (μ = 1, 3, 10 at S = 250, 500, 1,000) and report what it shows; if it does not support 3, flag the sentence. Do not change μ. |
| D9 | Split: "We use a 70%/30% split ..., dropping any test tiles that do not represent both procedures." | 30% of tiles drawn for testing; a test tile needs positives of both zones; dropped test tiles are not moved to training. | No code change. Report the exact rule as a wording note for Victor. |
| D10 | The text does not mention these rules. | Negatives also avoid AREA NOT INCLUDED and NFHL OPEN WATER polygons and must lie inside Florida's county polygons; NLCD class 0 (no data) is not land; distances are floored at 1 m; wetness is StreamCat's catchment mean, served ×100. | Keep them all (removing the state mask would put negatives in Alabama and Georgia where floodplains are not loaded). List each as an omission for Victor's prose. |

Everything else stays as on 2026-11-06: the grids of Table `tab:models`, the degenerate rule (one class for
more than 95% of the ID test points), the prediction rule (flood iff p > 0.5), the 60/20/20 model split
(`random_state` 42), the search settings (squared penalty, T = 100, at most 50,000 steps, patience 500, the
budget grid, the same seeds), S = 500, k = 10, c = 20, every control, the unweighted per-tile columns, cell
seed 0, one split.

## 1. Rules

- `export D=<today>`. Outputs go under `results/$D/`, logs under `logs/$D/`. The new dataset goes in
  `data/processed_v4/` (`dataset_FL.parquet`, and `FEATURES.txt` listing the nine raw features), the cell in
  `data/interim/aotil/cells_v4_$D`. Do not change earlier results, cells, `data/processed/`,
  `data/processed_w7/` or `data/processed_w9/`.
- **The new points reuse point ids, and every pull script treats an existing output as done.** Before the
  build, move (do not delete) every interim file keyed by point id (Earth Engine shards, hydro and StreamCat
  files, the soil table, anything else you find) into `data/interim/archive/sampling-v3/`, as
  `archive_sampling_v1.sh` did, and record checksums. Caches keyed by something else (SDA map units, Atlas 14
  blocks) may stay.
- Use `.venv/bin/python`. Every new or changed function gets a test. List every changed golden with its reason.
- Repo policy: delete code you retire; do not comment it out.
- Do not commit or push. Keep the Mac plugged in with the lid open.
- Paper: numbers, placeholders, figure files and the two Part F insertions only. Standing rules in
  `docs/TODO-CLAUDE-CODE.md` (no section labels or `Section~\ref`, no new bib entries, `algorithmic`). Write
  "Zone A" and "Zone AE" without "~".
- SDA is a shared government server: at most 2 concurrent requests, retries, cache.

## Part A. Audit the code against the text (no new results)

Go through every statement of the parts listed in §0, sentence by sentence, including every number and every
"we do X". For each, give a table row: the statement, the `file:line` that implements it, and one of
**agrees**, **changed here (D1 to D10)**, or **disagrees**. `docs/RUN-RERUN-ALL-2026-10-25.md` §1 has the
earlier audit of the Approach; redo it, since the code has changed since.

For a **disagrees** row:
- a bug (the code clearly means to do what the text says) → fix it and add a test;
- the text describes a choice the code makes differently, and D1 to D10 do not cover it → **stop** and list
  these before changing any method;
- the text is vague or leaves out a rule the code applies → no code change; list it as a wording note.

## Part B. Code changes

Implement D1 to D7 in `sample_points.py`, `pull_gee.py`, `pull_soil_points.py`, `aotil/zoo.py`,
`aotil/region_match.py` and every reader they affect (`audit_dataset.py`, `aotil/audit_cell.py`, the chain
scripts, the figure scripts, `aotil/report_paper.py`, anything that assumes 20,000 per group, 80,000 points,
seven features, numeric land cover, or kept duplicates). Tests at least: the candidate count and keep-all
thinning (D1); uniform-by-area placement without a cap (D2); the 300 m radius on a synthetic DEM (D3); the soil
coding with a water unit (D4); the one-hot columns are fixed, complete, and NaN-propagating, and hgb and xgb
receive NaN (D5); duplicate detection, including that the first in grid order is kept (D6); q = 6 (D7).

## Part C. Rebuild the dataset

1. Archive the point-keyed interim files (§1).
2. `sample_points.py` with D1 and D2 → points. Then `pull_gee.py` (with D3), `pull_hydro.py` (catchments,
   flowlines, StreamCat, merge), `pull_soil_points.py` (D4), and the merges, into `data/processed_v4/`.
3. Run `audit_dataset.py` on the new table and add checks: no two points closer than 100 m; every point on
   land; every negative 500 m to 5 km from the nearest SFHA, inside a panel, outside Zone D, AREA NOT INCLUDED
   and OPEN WATER, inside Florida; candidates per group exactly 25,000; per-group survivor counts.
4. **Spatial autocorrelation ranges** (for the Split paragraph's "at most XX km"). As blockCV does (Valavi et al.
   2019): for each of the six numeric features and each one-hot column with at least 1% prevalence, compute
   an empirical semivariogram on a random 20,000-point subsample (seed 0) with 1 km lags up to 200 km, fit a
   spherical and an exponential model by weighted least squares, keep the better fit, and report the
   effective range (the distance where the fitted model reaches 95% of its sill). Report each feature's range
   (for a categorical feature, the largest over its columns) and the maximum. If any range exceeds 50 km,
   flag the Split sentence; do not change the tiles.

**Stop** before Part D if: any zone-and-label group keeps fewer than 20,000 points after thinning; the audit
fails; more than 2% of points fall in no soil map unit; or more than 5% of non-water points lack a flood
class.

## Part D. Fit the cell and check it (stop point)

Fit `FL_band_AE_s0` from `data/processed_v4` into `data/interim/aotil/cells_v4_$D` and run
`aotil/audit_cell.py` (PASS). Report the per-family table (settings, degenerate, duplicate, kept, ID accuracy
range), the number of model input columns after one-hot, kept N, the 60/20/20 sizes, ID accuracy range,
probit r(ID, OOD) over all kept models and over the test models, and `n_models_needed`.

**Stop** if: kept N < 200; the lowest kept ID accuracy is above 0.60; a family keeps fewer than 10 models; or
probit r(ID, OOD) over all kept models is below 0.5.

## Part E. Full run

Repeat Part C of `docs/TASK-ZOO-GRIDS-2026-11-06.md` on the new cell with the same scripts, seeds and settings
(copy `logs/2026-11-06/C_chain.sh` and `D_chain.sh` into `logs/$D/` and change only the paths): the shift
(pooled and per tile) and its tile-shuffle null; all 81 searches (classic, Laplacian and GTV over the budget
grid, the shuffled and family-disjoint controls, the μ sweep); the selection audit; the interpolation control;
the regions (c = 20); `region_match.py`; and the maps figure (a new file, `fig_maps_v6.pdf`). Check claims C1 to
C7 as fixed in `docs/TASK-RERUN-ALL-2026-10-25.md` §2 and report each as holds or does not hold. No stop on
C4, C6a, C6b or the μ sweep: report them.

## Part F. The data figure

1. Adapt `Claude outputs/fig-feature-boxes-2026-11-07/fig_feature_boxes.py` (Victor approved this design) into a
   repo script, for example `aotil/fig_feature_boxes.py`: read `data/processed_v4`, take the sets from
   `zoo.split_tiles` and `zoo.subset` instead of its copied split code, drop its water-unit recoding (D4 makes it
   unnecessary), keep its layout, colors and percentile whiskers, and draw terrain variability from
   `elev_sd_r300`. Land cover shows the seven most common classes of the new data plus "other". Write
   `CS371_Cho_Li/figures/fig_feature_boxes.pdf`.
2. In the paper, replace the line `% INSERT FIGURE HERE` with this block, updating the land-cover class list in
   the caption if the seven most common classes change:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_feature_boxes.pdf}
\caption{Each feature by set (Train and ID test are Zone AE; OOD is Zone A). Top: the box spans the 25th to 75th percentile, the line is the median, and the whiskers reach the 5th and 95th percentiles. Bottom: the share of each set's points in each class. Land cover shows its seven most common classes, 90 woody wetland, 42 evergreen forest, 81 pasture and hay, 95 herbaceous wetland, 21 developed open space, 82 crops, and 22 low-intensity developed, and merges the rest. Soil shares are over points with a soil class.}
\label{fig:data}
\end{figure*}
```

3. In the Data statistics paragraph, change `Table~\ref{tab:data}` to `Figure~\ref{fig:data}` (the table no
   longer exists).

## Part G. Paper numbers

Numbers only. Fill every `\textbf{XX}`: the points left after thinning (Sampling), the largest autocorrelation
range in km (Split), the two missing shares (Data statistics; for "the soil features", give the share missing
either one and report each separately), N and the 60/20/20 sizes (Models). Update every number that depends on
the data or the models: Budget penalty (the 4.0% share of OOD points, the largest inverted budget), Graph (the
median distance to the 10 nearest neighbors), Matching (ε; q stays 6), all of Results, Tables `tab:search`,
`tab:regions` and `tab:models` (counts, and "features per split" as 1, the square root and the total of the
model input columns), the maps figure, and Appendix A ("all seven features and all 29,161 training points").
Compile a copy: no overfull boxes beyond the existing one in Table `tab:models`, no undefined references other
than the `ssurgo` citation, which Victor will add.

Flag, do not rewrite: every sentence whose claim stops holding; the region names and the Case studies paragraph
if the regions change; "choose μ = 3 from a sensitivity analysis" if the sweep does not support it; "seven
features do not determine the label" (Concept shift); the Table `tab:models` caption (it mentions only the
degenerate drop); and the wording notes from Part A, D9 and D10.

## Part H. Report and docs

`docs/RUN-REVISED-EXPERIMENTS-$D.md`. Victor will hand it to Claude to write the prose, so make it complete:

1. A one-paragraph summary in plain language: what changed and which claims hold.
2. The Part A audit table, and every code change (D1 to D10 and any bug fix) with `file:line`.
3. The data build: candidates and survivors per zone and label, the total; tiles (all, drawn for testing,
   dropped, test, training, training tiles that hold Zone AE points); points and flood share per set (Train,
   ID test, OOD).
4. **Everything the Data statistics prose needs**, per set and over all points: for each numeric feature the
   mean, SD, median, 5th and 95th percentiles, minimum and maximum; for each categorical feature the class
   shares (the seven most common land-cover classes with their NLCD names, all six flood classes, fluvial);
   the missing count and share of every feature, and of "either soil feature", with the main reason (for
   example soil-survey water units, outside every NHDPlus catchment).
5. The autocorrelation ranges per feature and the method.
6. Part D's tables, claims C1 to C7, the μ sweep, the largest inverted budget, N1 (without naive Bayes).
7. Every paper change as a Before/After block, and every flagged sentence with the reason.
8. Timings.

Then update `docs/DECISION-PROVENANCE.md` (rows for D1 to D7, support "Your decision" or the cited source, source:
this brief) and move item 7 of `docs/TODO-CLAUDE-CODE.md` to Done with the date and the report.

## Stops (summary)

1. Part A: the text describes a method the code does differently and D1 to D10 do not cover it.
2. Part C: a zone-and-label group keeps fewer than 20,000 points; the dataset audit fails; more than 2% of points
   in no soil map unit; more than 5% of non-water points without a flood class.
3. Part D: kept N < 200; lowest kept ID accuracy above 0.60; a family below 10 models; probit r(ID, OOD) below 0.5.
4. A grid keyword or value would have to change.
