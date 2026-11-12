# Final pipeline: match the code to the finished paper, three entry points, run the last experiments

Written 2026-11-11 for Claude Code, by Claude (chat) for Victor. Read all of it before changing anything.

The Experiments and Analysis sections of the paper are final. This task makes the code match them exactly. It removes
every experiment and function the paper no longer uses, rebuilds the code around three user-facing scripts, runs the
experiments the paper still needs, and exports the regions for the GIS error analysis.

## 0. Ground rules

- **Source of truth:** `CS371_Cho_Li/acl_latex.tex` (Experiments, Analysis, Appendix) and the decisions in §1. If code
  and paper disagree on a method or setting, do not change either one: stop and report the disagreement to Victor.
- **Do not edit** `CS371_Cho_Li/acl_latex.tex`. Victor and Claude (chat) edit the prose; you produce numbers and files.
- **Do not delete data.** Nothing under `data/` is deleted, including the files of features the paper no longer uses
  (Victor will delete those when the project ends). Do not delete `results/`, `logs/`, `refs/` or `docs/` history.
- **Delete code, do not comment it out.** Code the paper does not use is deleted (git keeps it).
- **Git:** first `git switch -c final-pipeline` and commit the current working tree as
  `snapshot before final pipeline` (there are about 45 uncommitted changes, including the paper). Then commit in
  logical steps. Never push.
- **Environment:** the Mac, `.venv` (torch, optuna, xgboost, scikit-learn, geopandas, pyproj). Long runs go in the
  background under `caffeinate -i`, log to `logs/<date>/`, and must be resumable.
- **No new methods or settings.** Do not tune, change a hyperparameter, or add an experiment beyond §5. If something
  looks wrong, report it instead of fixing it silently.

## 1. What the paper now contains (the spec)

Read the paper for the wording; this is the checklist of everything the code must produce.

**Data (Experiments, Data Construction; Figure 1).** `data/processed_v4/dataset_FL.parquet`, the nine features of
`data/processed_v4/FEATURES.txt` (land cover, soil flood frequency, fluvial class, slope, terrain variability, height
above nearest drainage, topographic wetness, log distance to river, log distance to coast). 96,467 points; 50 km tiles;
70/30 tile split; 27 test tiles; 36,758 Zone AE training points, 10,986 ID test points, 17,346 Zone A (OOD) points.
The Split paragraph's spatial-correlation ranges (median 27 km; 73 to 107 km for wetness, distance to coast, land
cover; 110 km tiles would leave 7 test tiles) come from `aotil/variogram_ranges.py` (check that it reproduces them). Figure 1 is
`aotil/fig_feature_boxes.py` (`figures/fig_feature_boxes.pdf`).

**Models (Implementation Details, Models; Appendix Table tab:models).** Eight families, the grids of the appendix
table; 369 settings; drop 1 that predicts one class for > 95% of ID test points and 78 that repeat an earlier
setting's predictions; N = 290. One-hot for the three categorical/ordinal features (28 inputs); median imputation except
histogram gradient boosting and XGBoost. Model split 60/20/20 with `random_state=42`: 174 selection, 58 validation,
58 test models.

**Table 1 (tab:shift) and the "Aggregation hides the shift" paragraph.** Accuracy, AUROC, F1; ID = held-out Zone AE
test points, OOD = Zone A test points; Delta = ID minus OOD; averaged over all 290 models. Pooled over all test points,
and per tile (27 tiles; 25 for AUROC) averaged over the models, then over the tiles, with 95% t-intervals over the tiles.
Two-sided Wilcoxon signed-rank test of the per-tile Delta, Holm across the three metrics. Also: tiles where models do
worse on Zone A per metric, the Palm Beach tile numbers (its share of each test set, models' accuracy on its Zone AE
points), all-Zone-A corr_te with its Fisher interval.

**Correlation.** corr = Pearson correlation of probit-transformed accuracies (clip 1e-6); corr_te on the 58 test
models; 95% intervals by Fisher's transformation.

**Graph and coherence.** Symmetric kNN graph A, k = 10, EPSG:3086 (`graph.build_adjacency`). Median 10-NN distance
2.17 km. Components = connected components of the subset in A with at least c = 20 points; report their number and the
points they hold. Purity = mean over selected points of the share of the point's own 10 nearest neighbors that are
selected (directed kNN). Random-subset purity S/m is quoted once in the Graph paragraph.

**Searches (Approach; Implementation Details, Search; Algorithm 1; Appendix).** OODSelect (mu = 0); Laplacian and GTV
losses with mu = 3, R normalized by S*k; squared budget penalty; T = 100 TPE trials (Optuna), at most 50,000 steps,
patience 500; full-batch Adam with a cosine learning rate; lambda ramp over W steps; TPE ranges as in the Appendix;
seed 0. Smoothing (Equation eq:smooth): s_bar = (I + D)^-1 (I + A) s on the same graph, applied to the final weights
of the search. It is computed on the complement q = sigmoid(-theta), keeping the S points with the smallest smoothed q
(the largest s_bar; stable sort), so that weights that round to 1.0 do not tie (`paper_numbers.smooth_graph`).
"+ smoothing" = smoothing after a search.

**Table 2 (tab:search), S = 500.** All Zone A, then OODSelect, + smoothing, Laplacian, + smoothing, GTV, + smoothing:
corr_te [95% CI], # Comps with points held in parentheses (e.g. `7 (383)`), purity.

**Regions, Figure 2, Table 3 (tab:regions).** The method we analyze is **GTV + smoothing at S = 500** (8 components,
421 points). Region names: the county (by `meta_point_county`, names from `data/interim/gis/counties_FL.parquet`) that
holds most of the region's points; east/west by mean longitude when two regions share a county. Matching
(`aotil/region_match.py`): the six numeric features standardized over the OOD set, exact land cover; candidates are
Zone AE test points in test tiles that contain no point of the region; caliper = the 95th percentile of the Zone AE to
Zone AE nearest-candidate distance (1.30); up to 5 matches; the 75 Zone AE test points missing a feature are not
matched. Table 3 rows: each region (# Points, Unmatched, eta), All regions (pooled), Other Zone A (the Zone A test
points outside the subset, matched the same way), Zone AE baseline. Figure 2: (a) OODSelect, (b) GTV + smoothing, both
S = 500; violet #6250d6 = selected and in a component, red #e34948 = selected, not in one
(`results/2026-11-11/paper_figures/fig_maps.py`).

**Analysis, Sensitivity.** Two parts, mu first, then S.
- **mu (Figure fig:mu, new; paragraph "Regularization weight").** Laplacian and GTV at S = 500 with
  mu in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100} (half-decade log grid; 3 is a grid point), OODSelect (mu = 0) as the
  reference. Report corr_te [CI], purity, # Comps (points), and the validation correlation, each loss with and without
  smoothing. The figure follows `fig_budgets.py`'s style: x = mu (log), top corr_te, bottom purity, one line per loss,
  dotted line at mu = 3. Its paragraph has XX the numbers must fill (see `% CHECK` in the .tex).
- **Budget (Figure 3, fig:budgets).** S in {10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000} for
  OODSelect, Laplacian (mu 3), GTV (mu 3), 2 x 2 small multiples: left alone, right + smoothing; top corr_te, bottom
  purity; no random-subset line (`results/2026-11-11/paper_figures/fig_budgets.py`).
- **Ablation:** read from Table 2 (no separate table).

**Not in the paper any more (delete their code):** null-select controls (`*_null<N>`), family-disjoint splits, the
cell trained on Zone A (`FL:band:A:0`), the softplus budget penalty, the old directed-10-NN smoothing, the O5 analyses
(interpolation, LOFO, feature inclusion, pooling, polygons, shift decomposition, contrast), boundary statistics, lobes,
edge layers, terrain-matched controls, the matched-growth null, region recurrence, regions_land, halo rows, the
"500 hardest points" and classic-reference rows, the kinds table and the matching sensitivity of `region_match.py`,
the per-tile null check, the c sensitivity table, every feature beyond the nine, and the 45-feature and w7/w9 cells.

## 2. Target structure

Three entry points at the repo root; everything else is a library module they import. Keep module names where they
still fit (`aotil/zoo.py`, `aotil/graph.py`, `aotil/cells.py`); rename where a name now misleads (e.g.
`oodselect_paper.py` -> `search.py`, `o1_shift.py` -> `shift.py`). Delete `run_oodselect_local.sh` once
`run_experiments.py` replaces it.

1. **`run_experiments.py`** — all computation (the only script that needs torch). Stages, each resumable and
   skippable: `zoo` (fit the 369 settings, keep 290), `shift` (pooled and per-tile metrics per model, the test),
   `search` (the budget grid and the mu sweep; one single-threaded process per search, `--jobs` at a time),
   `smooth` (smoothed subsets of every search), `regions` (components and names for the analyzed method), `match`
   (Table 3 including Other Zone A and the baseline), `data` (variogram ranges and anything Figure 1 needs), `all`.
   Flags: `--out results/<date>` (default today), `--stage`, `--jobs`, `--dry-run` (print what would run and its
   estimated time), `--force` (redo a finished unit). It writes everything the report needs to disk (arrays, CSV,
   parquet, JSON), plus `manifest.json` (git commit, settings, seeds, package versions, timings, input file hashes).
   Progress lines show what is running and what is left.
2. **`make_paper.py`** — no torch, seconds to run, reads one run folder (`--run`, default the latest complete one).
   - Prints every table in the terminal in a readable layout (aligned columns, titles, units).
   - Writes paste-ready LaTeX tabular blocks in exactly the paper's current format (leading zeros, `$-$` signs,
     `\quad + smoothing` rows, `\# Comps` cells like `7 (383)`, Region / \# Points headers) to
     `<run>/paper/tables/*.tex`.
   - Writes `<run>/paper/numbers.md`: every number the prose states, keyed by paragraph, including the XX of the
     "Regularization weight" paragraph, and the self-checks with PASS/FAIL.
   - Draws all figures (feature boxes, maps, budgets, mu) into `CS371_Cho_Li/figures/` under their plain names, moving a
     replaced file to `CS371_Cho_Li/figures/old/<name>_vN.pdf` (next free N; see `docs/TODO-CLAUDE-CODE.md`), plus PNG
     previews in `<run>/paper/figures/`.
   - Flags: `--only tables|numbers|figures`, `--no-copy` (do not touch `CS371_Cho_Li/figures/`).
   - Reuse `aotil/paper_numbers.py` (its tables and self-checks match the paper today) and the three figure scripts;
     fold them in rather than rewriting from scratch.
3. **`export_components.py`** — see §4.

Every script: `--help` with examples, clear errors when an input is missing (say which stage to run), and a short
summary at the end of what was written where. Add a root `README.md` with the three commands, what each produces, and
how long each takes.

**Data build.** Keep `sample_points.py` and only the pull/fetch scripts and functions that produce the points and the
nine features (trace each FEATURES.txt column back to its code; `docs/FEATURE-SOURCES.md` maps the old 45). Delete the
code that gathers any other feature (e.g. Atlas 14, BFE/LOMR extras, the extra GEE/StreamCat/NHDPlus columns,
AlphaEarth). Do not rerun the pulls (GEE access, hours); keep the code path of the nine unchanged and show in the
report that it is.

**Tests.** Delete tests of deleted code. Keep and update the rest; add tests for the new entry points (tiny synthetic
inputs) and keep `paper_numbers.py`'s self-checks as tests or as `make_paper.py` checks.

## 3. Verification ladder (do these in order; stop and report on any failure)

1. Tests pass.
2. **Pilot:** `run_experiments.py` on a fresh folder with stages `zoo`, `shift` and the three S = 500 searches. The zoo
   must reproduce `data/interim/aotil/cells_v4_2026-11-07` (same 290 models, same predictions), and each S = 500 theta
   must reproduce `results/2026-11-07/oodselect_paper/{classic,spatial_mu3_knn10,spatial_gtv_mu3_knn10}/` (identical
   top-500 masks; corr_te within 1e-4). Then `make_paper.py` on the pilot must print exactly the S = 500 numbers now in
   the paper: Table 1 (all cells), Table 2 (all rows), Table 3 (8 regions, Other Zone A 16,846 / 0.06 / 0.29, baseline
   10,911 / 0.05 / 0.31), and the Results paragraphs' numbers (Palm Beach 14% / 1% / 0.50, tile counts 19/27, 19/25,
   22/27, Wilcoxon p 0.044 / 0.016 / 0.091). A table of paper value vs new value goes in the report.
3. Only then start the full run (§5).

## 4. Component export for the GIS error analysis

`export_components.py --method "GTV+smoothing"`; `--method` takes OODSelect, OODSelect+smoothing, Laplacian,
Laplacian+smoothing, GTV, GTV+smoothing; `--S 500`, `--mu 3`, `--min-size 20` and `--run` default to the paper's
settings and the latest run. Output folder `exports/<method>_S<S>_mu<mu>_c<c>_<run date>/` and a `.zip` of it, so
Victor can send one file to his teammate:

- `README.md` — what the regions are (one paragraph, plain words), how they were produced (method, S, mu, c, run
  folder, git commit), the CRS of every coordinate, a dictionary of every column (meaning, units, codes), and how to
  open the files in QGIS, ArcGIS and Python.
- `regions.gpkg` (GeoPackage, EPSG:5070) — layer `points`: every point in a component; layer `regions`: one convex-hull
  polygon per component (for display only; say so); layer `selected_not_in_component`; layer `matches`: the matched
  Zone AE points of each region point.
- `points.csv` — one row per point in a component, sorted by component then point id.
- `components/<NN>_<county-slug>.csv` — one file per component, the same columns (NN = rank by size).
- `components_summary.csv` — one row per component: id, name, # points, counties, tiles, centroid lon/lat, extent in
  km, share labeled flood, unmatched share, eta (the Table 3 values).
- `selected_not_in_component.csv`, `matches.csv` (region point id, matched Zone AE point id, its lon/lat, label,
  feature distance).

Columns of the point files: `component_id`, `component_name`, `point_id`, `lon`, `lat` (NAD83, EPSG:4269, as in the
dataset), `x_5070`, `y_5070` (EPSG:5070 metres), `zone` (A), `label` (1 = inside a Zone A polygon, 0 = sampled
negative 500 m to 5 km outside), `fema_polygon_id` (`meta_fld_ar_id`), `dfirm_id`, `county`, `county_fips`,
`tile_50km`; the nine features exactly as the models see them (`landcover`, `soil_floodfreq`, `soil_fluvclass`,
`slope`, `elev_sd_r300`, `hand`, `wetindexcat`, `log10_dist_river`, `log10_dist_coast`) plus readable versions
(`landcover_name` from the NLCD legend, `soil_floodfreq_name` from None to Very frequent, `dist_river_m`,
`dist_coast_m`); units exactly as `FEATURES.txt` and the pull code define them (check, do not guess). Model results,
in clearly named columns after the features: `share_models_correct` (of the 290 models), `n_matches`,
`match_label_disagreement` (the point's share of matches with a different label). No other features or columns from
older runs. Run it once for GTV+smoothing and check the CSVs open in Excel, the GeoPackage in QGIS (or with
`geopandas.read_file` per layer), and that the counts match Table 3.

## 5. Experiments to run (one fresh run folder, `results/<date>/`)

After the pilot passes, run `run_experiments.py --stage all` in the background:
- zoo; shift;
- budget grid: OODSelect, Laplacian (mu 3), GTV (mu 3) at the 13 budgets (39 searches);
- mu sweep at S = 500: Laplacian and GTV at {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100} (mu 3 is shared with the grid;
  16 new searches); OODSelect is the mu = 0 reference;
- smoothing of every search; regions and matching for GTV+smoothing S = 500 (Table 3, Other Zone A, baseline).
Then `make_paper.py --run results/<date>` and `export_components.py --method "GTV+smoothing"`.

Print a time estimate before starting (the S = 500 searches took about 10 minutes each on one core; large budgets
take longer). `docs/TASK-GTV-MU-SWEEP-2026-11-11.md` is the earlier plan for the sweep; this task supersedes it.

## 6. Report

Write `docs/RUN-FINAL-PIPELINE-<date>.md`:
1. The new structure and the three commands.
2. Everything deleted (files and functions), each with one line on why (not in the paper), and the data-build check.
3. The verification ladder: paper value vs new value for every number in §3 step 2, then the same for the full run.
   Any difference beyond the tolerances is listed with its cause; nothing is silently accepted.
4. The mu sweep: a table, loss x mu, with corr_te [CI], purity, # Comps (points), validation correlation, with and
   without smoothing; and the values for every XX of the "Regularization weight" paragraph. Answer the `% CHECK`:
   does purity rise and corr_te weaken as mu grows, for both losses; at equal purity, which loss inverts more strongly.
5. Where the export is and its file list.
6. Anything that looked wrong, unclear, or that you did not do.

Also append the run to `CS371_Cho_Li/CHANGES-2026-11-11.md`'s successor for the day you run it
(`CS371_Cho_Li/CHANGES-<date>.md`), listing the new figure files and tables Victor must paste.
