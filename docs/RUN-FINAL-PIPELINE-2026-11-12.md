# Final pipeline: the code matches the paper, three entry points, the last experiments (2026-11-12)

Report for `docs/PROMPT-FINAL-PIPELINE-2026-11-11.md`, run by Claude Code on 2026-11-12. Branch `final-pipeline`
(never pushed). Paper: `CS371_Cho_Li/acl_latex.tex`, not edited.

**Summary.** The code now matches the paper's Experiments and Analysis and runs from three commands
(`run_experiments.py`, `make_paper.py`, `export_components.py`); 19 analysis modules with their 10 tests, the old
runner and 7 data-build scripts that the paper no longer uses are deleted (the branch removes about 29,600 lines and
adds 4,700), and the data build keeps only the code of the points and the nine features (an offline rebuild
reproduces `data/processed_v4` exactly). Every check script passes.
The pilot reproduced the 2026-11-07 cell and the three S = 500 searches bit for bit and printed exactly the paper's
numbers (90 of 90 table cells, 54 of 54 prose numbers). The full run (`results/2026-11-12`, 1 h 12 min) reproduced all
41 searches the 2026-11-07 run also made bit for bit, ran the 16 new mu-sweep searches, and every paper number again
matches (126 of 126 checks, Budget paragraph included); no number disagrees with the paper. **The mu sweep does not
give the tidy picture its paragraph expects** (section 4): the Laplacian behaves as expected, but GTV is not monotone
in mu (purity peaks at mu = 10, and two runs are weak: mu = 1, where the tuned budget weight left the budget
penalty almost off (soft total 248 of 500), and mu = 100, where most trials do not invert), and at equal purity neither loss consistently inverts
more, so two XX of that paragraph have no clean answer. The figures are in `CS371_Cho_Li/figures/` (fig_mu.pdf new),
the tables in `results/2026-11-12/paper/tables/`, every number in `results/2026-11-12/paper/numbers.md`, and the GIS
export in `exports/GTV+smoothing_S500_mu3_c20_2026-11-12.zip`.

## 1. The new structure and the three commands

Three scripts at the repo root; everything else in `aotil/` is a library they import (`README.md` and
`aotil/README.md` describe both).

| Command | What it produces | Time here |
|---|---|---|
| `.venv/bin/python run_experiments.py --stage all` | every computation into `results/<today>/` | 1 h 12 min with `--jobs 10` (searches 71 min) |
| `.venv/bin/python make_paper.py` | tables, `numbers.md`, figures, from the latest complete run | about 15 s |
| `.venv/bin/python export_components.py --method "GTV+smoothing"` | `exports/<method>_S500_mu3_c20_<run date>/` and its `.zip` | about 3 s |

`run_experiments.py` (the only script that imports torch) has seven resumable stages, each with its own folder in the
run folder:

| Stage | Folder | What it holds | Library module |
|---|---|---|---|
| `zoo` | `zoo/` | the cell `FL_band_AE_s0.npz`, `_models.csv`, `_dropped.csv`, and `zoo.json` (counts per family, split sizes, fit time, file hashes) | `zoo.py` |
| `data` | `data/` | `data.json` (points, sets, flood shares, missing shares, model inputs, tiles at 50 and 110 km, median 10-NN distance, the seven most common land-cover classes), `variogram/`, `figure1.parquet` | `variogram_ranges.py`, `fig_feature_boxes.py` |
| `shift` | `shift/` | `summary.json` (Table 1, the test, Palm Beach, corr_te on all Zone A), `per_tile.csv`, `per_model.csv` | `shift.py` |
| `search` | `search/<unit>/` | `theta.npy`, `trials.csv`, `result.json` per search; units `oodselect_S<S>`, `laplacian_mu<mu>_S<S>`, `gtv_mu<mu>_S<S>` | `search.py` |
| `smooth` | `smooth/` | `subsets.csv` (every search alone and + smoothing: corr_te [95% CI], r_val, components, points in them, pieces, purity, tiles, ties at the cut), `masks.npz` | `subsets.py`, `graph.py` |
| `regions` | `regions/` | `points.parquet` (the 500 selected points with component, size, name), `components.csv` | `regions.py` |
| `match` | `match/` | `table.csv` (Table 3), `pairs.parquet` (every match), `point_summary.parquet`, `summary.json` (eps, pool) | `region_match.py` |

Flags: `--out` (default `results/<today>`), `--stage`, `--jobs` (searches at a time, default performance cores
minus 2), `--only` (search names, shell patterns), `--dry-run` (the plan and its estimated time), `--force`. Each
search is its own single-threaded process (`aotil/search.py` as a worker); every process runs with one BLAS/OpenMP
thread, so results do not depend on `--jobs`. `manifest.json` holds the git commit (and any uncommitted file), the
settings, seeds, package versions, input hashes, stage and search timings, and `complete`. Logs:
`logs/<run folder name>/run_experiments.log` and `logs/<run folder name>/search/<unit>.log`.

`make_paper.py` (no torch) reads one run folder (`--run`, default the latest complete one), prints Tables 1-3 and the
Appendix table, writes them as paste-ready tabular blocks to `<run>/paper/tables/`, writes `<run>/paper/numbers.md`
(every number of the prose by paragraph, the mu sweep table and the filled "Regularization weight" paragraph, the
self-checks, and a PASS/FAIL check of every table row and number against the paper's text), and draws the four
figures into `CS371_Cho_Li/figures/` (a replaced file moves to `figures/old/<name>_vN.pdf`; an identical one is left
alone; `--no-copy` leaves the folder alone) with PNG previews in `<run>/paper/figures/`. `--only tables|numbers|figures`.

`export_components.py`: section 5.

Checks: `test_entry_points.py` runs the three scripts end to end on a synthetic dataset; `aotil/test_*.py` check each
library module; the data-build scripts keep their own `test_pull_*.py` and `test_sample_points.py`.

## 2. What was deleted, and the data-build check

Every file below was deleted with `git rm` (the snapshot commit `1f3fd61` keeps it). Nothing under `data/`,
`results/`, `logs/`, `refs/` or `docs/` was deleted.

### 2.1 Analyses the paper no longer uses (aotil/)

| File (lines) | Why |
|---|---|
| `audit_cell.py` (993) | re-derived a cell from the code; the zoo stage now checks kept + dropped = 369 itself and `test_zoo.py` checks the drop rules |
| `audit_selection.py` (730) | recomputed stored selections and held the old directed-10-NN smoothing, which the paper replaced by smoothing on graph A |
| `report_paper.py` (365) | random-subset and "500 hardest points" baselines and the model-count rule, not in the paper |
| `o1_table.py` (95), `o1_tile_null.py` (118) | the old Table 1 builder and the per-tile null check; Table 1 now comes from `shift.py` |
| `o5_interpolation.py`, `o5_lofo.py`, `o5_feature_inclusion.py`, `o5_pooling.py`, `o5_polygons.py`, `o5_shift_decomp.py`, `o5_contrast.py` (5,045 in all) | the O5 analyses (interpolation, leave-one-feature-out, feature inclusion, pooling, polygons, shift decomposition, contrast) |
| `boundary_stat.py` (1,968), `lobes.py` (990), `edge_layers.py` (1,293) | boundary statistics, lobes, edge layers |
| `match_controls.py` (674), `region_null.py` (258) | terrain-matched controls and the matched-growth null |
| `region_recurrence.py` (149), `regions_land.py` (111) | region recurrence across seeds and the regions_land table |
| the tests of all of these (`test_boundary_stat.py`, `test_match_controls.py`, `test_o5_*.py` x 7, `test_regions_land.py`; 4,298 lines) | tests of deleted code |
| `run_oodselect_local.sh` (root, 218) | replaced by `run_experiments.py` |
| `_to_delete/` (root) | a copied `common.py` and a `.pyc` the 2026-11-11 session set aside for deletion |

### 2.2 Inside the kept and renamed modules

- `oodselect_paper.py` -> `search.py`: deleted the null-select control (`--null-seed`, `null_accuracies`), the
  family-disjoint splits (`--split`, `family_split`, `make_split`, `TREE_FAMILIES` in `cells.py`), the softplus budget
  penalty (`--budget-penalty`; the squared penalty is the only one), the released budget grid helpers
  (`PAPER_BUDGETS`, `budgets_for`), `plain_stats`, `AOTIL_THRESHOLD`, `variant_name`/`run_controls` and the
  `torch.sparse` reference. The arithmetic of the search (loss, budget term, spatial terms, optimizer, Optuna search,
  seeds) is the old code line for line; the pilot reproduces the stored thetas (section 3.2).
- `o1_shift.py` -> `shift.py`: rewritten around what Table 1 reports (from the 2026-11-11 `paper_numbers.shift_part`,
  same arithmetic); deleted the land and paired-land scopes, the label-weighted per-tile metrics, the tile bootstrap,
  the 15 strongest models, the per-family table and the multi-seed pooling.
- `region_match.py`: deleted the halo rows, the "500 hardest points" and classic-reference rows, the kinds table, the
  matching sensitivity table, the descriptive columns (strongest-model accuracy, HAND and river medians, SMD) and the
  command line; added the per-pair and per-point outputs the export needs. Matching itself is unchanged.
- `regions.py`: deleted the v1-dataset recovery (`source_row_ids`, the `sample_points` import), the checks against
  `oodselect_paper`'s result files and the command line; names regions as the paper does (the old "/"-joined names are
  gone).
- `graph.py`: deleted grouped graphs, Moran's I and its null, Jaccard, `incoherence`, the self-test (now
  `aotil/test_subsets.py`).
- `zoo.py`: deleted `--extra-train` (training on extra points), the command line and the Zone A training direction
  (the cell `FL:band:A:0`); `write_cell` writes the same three files `main` wrote.
- `paper_numbers.py`: rebuilt to read a run folder; deleted Table tab:mu (S x mu), Table tab:c (the c sensitivity), the
  old-smoothing check against the selection audit, the sign-flip permutation test and the tile bootstrap.
- `variogram_ranges.py`, `fig_feature_boxes.py`: only their command lines.

### 2.3 Data build (repo root): only the points and the nine features

| Deleted | Why |
|---|---|
| `pull_atlas14.py` (464) | NOAA Atlas 14 rainfall depths, not a model feature |
| `pull_dem.py` (1,442), `pull_point_extras.py` (624), `check_gis_inputs.py` (644) | the boundary analysis's 10 m DEM along region edges, a provenance sidecar, and checks of those inputs; none produces a dataset column |
| `pull_extras.py` (111) | FEMA BFE lines (only an optional sampler input with no effect on the draws) and LOMR polygons (only for `o5_polygons.py`) |
| `archive_sampling_v1.sh` (86) | a one-off archive of the v1 sample |
| `audit_dataset.py` (1,504) | the audit of the old 45-feature table; the build chain does not need it |

Inside the kept scripts:

- `pull_gee.py`: the Earth Engine bands other than the four features (elevation, aspect, curvature, tpi_300, tri_300,
  upstream area, tpi_broad, impervious, the JRC surface-water group) and the columns derived from them, the 30 and
  100 km tiles.
- `pull_hydro.py`: catchment area, the nearest reach's stream order, upstream area and channel slope, the nearest
  FTYPE, drainage density, every StreamCat metric but the wetness index (now requested for the catchment only), the
  StreamCat elevation quarantine.
- `prep_nhdplus_units.py`, `fetch_nhdplus_fl.sh`: the NHDPlus fields those attributes needed (AreaSqKM, LENGTHKM,
  StreamOrde, TotDASqKM, SLOPE, `elevslope.dbf`).
- `pull_ssurgo.py`: the SSURGO polygon and soil-line download (boundary analysis); the Soil Data Access helpers
  `pull_soil_points.py` imports stay verbatim.
- `pull_gis_lines.py`: county lines and map-panel edges (boundary analysis); the TIGER download and
  `counties_FL.parquet` stay.
- Unchanged: `sample_points.py` (same SHA-256), `pull_soil_points.py`, `pull_land_mask.py`, `subset_nfhl.py`,
  `fetch_nfhl.sh`.

Each kept column traced to its code:

| Column(s) | Script: function | Source |
|---|---|---|
| point_id, state, county_fips, block_id, lon, lat, geometry, meta_point_county | `sample_points.py`: `add_point_columns` | the sample; counties from `counties_FL` (`pull_gis_lines.py`) |
| domain, label, neg_protocol, dfirm_id, meta_fld_ar_id, meta_source_cit | `sample_points.py`: `_tag`, `_attr_frame` | NFHL flood-hazard polygons (`fetch_nfhl.sh`, `subset_nfhl.py`, `sample_points.py --prep`) |
| tile_50km | `pull_gee.py`: `derive` | the point's EPSG:5070 coordinates |
| slope, elev_sd_r300, hand, landcover | `pull_gee.py`: `terrain_group`, `terrain_variability_group`, `merit_group`, `nlcd_group` | Earth Engine: 3DEP 10 m, MERIT Hydro, NLCD 2021 |
| meta_comid; meta_dist_to_river_m, meta_dist_to_coast_m, log10_dist_river, log10_dist_coast; wetindexcat | `pull_hydro.py`: `join_catchments`; `measure_flowlines` + `quarantine_and_log_scale`; `pull_streamcat` + `join_hydro_files` | NHDPlus V2.1 (`fetch_nhdplus_fl.sh`, `prep_nhdplus_units.py`); EPA StreamCat |
| soil_floodfreq, soil_fluvclass | `pull_soil_points.py`: `fetch`, `build` | SSURGO via Soil Data Access |

**Check that the code path of the nine is unchanged.** (a) Every kept function on that path is the snapshot's text
(`git diff 1f3fd61`), or differs only by deleted lines for other features: each Earth Engine group's image is now
its kept band alone (same expression, same projection, same reducer), the StreamCat request asks for `wetindex` at
`areaOfInterest=cat` instead of every metric at `cat,ws`, and `derive` makes only the 50 km tile. (b) Offline rebuild
from the cached inputs, into a scratch folder: `pull_gee.py --merge` and `pull_hydro.py --merge` on a copy of
`points_FL.parquet`, then `pull_soil_points.py build`: all 18 kept columns and the nine features equal
`data/processed_v4/dataset_FL.parquet` exactly (NaN-aware, same dtypes, same row order, same geometry, CRS 5070), and
`FEATURES.txt` is byte-identical. The trimmed catchment and flowline stages on the NHDPlus geodatabase on disk, a
rebuild of that geodatabase with the trimmed `prep_nhdplus_units.py`, a replay of the soil lookup from a copy of its
Soil Data Access cache (no request sent) and `read_counties` on the cached TIGER file each reproduce their stored
output. No pull was rerun and nothing under `data/` was written. Not testable offline: the Earth Engine and StreamCat
requests themselves, which now ask for fewer bands and metrics; re-pulling one chunk and one county into a scratch
folder would confirm the values.

## 3. The verification ladder

### 3.1 Tests

All 13 check scripts pass (`logs/2026-11-12/T1_tests.log`, one log per script beside it):

| Script | Time | Checks |
|---|---|---|
| `test_entry_points.py` | 15 s | the three scripts end to end on a synthetic dataset (`--help`, `--dry-run`, every stage's outputs and manifest, resume, a missing input names its stage; make_paper's tables, numbers.md, figures and the figure versioning; the export's folder, zip, GeoPackage layers, columns and Table 3 counts) |
| `aotil/test_zoo.py` | 16 s | grids, seeds, drop rules, one-hot inputs, `run_cell` and `write_cell` on a synthetic dataset, the real cell |
| `aotil/test_search.py` | 2 s | correctness ties, the Laplacian and GTV losses against dense torch, the probit Pearson loss, the squared penalty, `train_one` determinism, `run_unit` writes and reproduces a search |
| `aotil/test_shift.py` | 1 s | accuracy, F1, AUROC (against sklearn), Holm, t-intervals, identical halves give a per-tile Delta of 0, a one-class tile leaves AUROC, the real cell's Table 1 |
| `aotil/test_subsets.py` | 1 s | the graph, purity, components, smoothing against a dense computation, the rounding case, region names |
| `aotil/test_region_match.py` | 1 s | the caliper, tile exclusion, land cover, the 5 closest, eta, the query expansion, q = 6, the Table 3 rows on a synthetic world |
| `aotil/test_paper_numbers.py` | 1 s | the paper's number formats, the paper check, the mu-sweep trends and matched purity, the budget numbers |
| `aotil/test_variogram_ranges.py`, `aotil/test_fig_feature_boxes.py` | 3 s | unchanged |
| `test_pull_gee.py`, `test_pull_hydro.py`, `test_pull_soil_points.py`, `test_sample_points.py` | 2 s | the data build, offline |

### 3.2 The pilot (`results/2026-11-12-pilot`, commit `6c2d24e`)

`run_experiments.py --stage all --out results/2026-11-12-pilot --only oodselect_S500 laplacian_mu3_S500
gtv_mu3_S500`: 7 min 36 s (zoo 30 s of fits, data 30 s, the three searches at once 3.4 to 6.5 min).

**The zoo reproduces `data/interim/aotil/cells_v4_2026-11-07` exactly**: all 16 arrays of the npz are identical
(scores `S_ood` 290 x 17,346 and `S_id` 290 x 10,986 bit for bit, labels, point ids, coordinates, tiles), and
`_models.csv` (290 x 30) and `_dropped.csv` (79 x 20) are identical. Same 290 models, same predictions
(`logs/2026-11-12/P_compare_pilot.log`).

**The three S = 500 searches reproduce `results/2026-11-07/oodselect_paper/` exactly**: the theta of each is bit for
bit the stored one, so the top-500 masks are identical and corr_te differs by 0:

| Search | corr_te new / 2026-11-07 | Best trial | r_val | lambda |
|---|---|---|---|---|
| OODSelect (`classic`) | -0.972470 / -0.972470 | 22 / 22 | -0.987148 / -0.987148 | 6.36e-06 / 6.36e-06 |
| Laplacian (`spatial_mu3_knn10`) | -0.918006 / -0.918006 | 76 / 76 | -0.962933 / -0.962933 | 0.01617 / 0.01617 |
| GTV (`spatial_gtv_mu3_knn10`) | -0.846217 / -0.846217 | 18 / 18 | -0.920886 / -0.920886 | 0.02666 / 0.02666 |

**`make_paper.py --run results/2026-11-12-pilot --no-copy` prints exactly the paper's numbers**: 90 of 90 table cells
and 54 of 54 numbers of the prose are the paper's (118 of 118 PASS in `results/2026-11-12-pilot/paper/numbers.md`).
Every self-check passes but one, which is bookkeeping: the pilot's manifest re-read `git status` on every save, so it
listed `aotil/paper_numbers.py` and `run_experiments.py`, which I edited while the pilot's searches were running (the
fix of that very behaviour, commit `64ce67a`), and the draft of this report; the code that ran was commit `6c2d24e`.

Paper value vs new value, every table row (cells separated by "/"):

**tab:shift**

| Row | Paper | New | Same |
|---|---|---|---|
| ID accuracy | 0.725 / 0.762 [0.710, 0.813] | 0.725 / 0.762 [0.710, 0.813] | yes |
| OOD accuracy | 0.759 / 0.744 [0.693, 0.796] | 0.759 / 0.744 [0.693, 0.796] | yes |
| Delta Accuracy | -0.034 / 0.017 [-0.042, 0.076] | -0.034 / 0.017 [-0.042, 0.076] | yes |
| ID AUROC | 0.811 / 0.817 [0.749, 0.886] | 0.811 / 0.817 [0.749, 0.886] | yes |
| OOD AUROC | 0.849 / 0.788 [0.725, 0.851] | 0.849 / 0.788 [0.725, 0.851] | yes |
| Delta AUROC | -0.038 / 0.029 [-0.038, 0.096] | -0.038 / 0.029 [-0.038, 0.096] | yes |
| ID F1 | 0.711 / 0.723 [0.651, 0.794] | 0.711 / 0.723 [0.651, 0.794] | yes |
| OOD F1 | 0.727 / 0.655 [0.579, 0.732] | 0.727 / 0.655 [0.579, 0.732] | yes |
| Delta F1 | -0.016 / 0.068 [-0.008, 0.143] | -0.016 / 0.068 [-0.008, 0.143] | yes |

**tab:search**

| Row | Paper | New | Same |
|---|---|---|---|
| All Zone A | +0.84 [+0.74, +0.90] / -- / -- | +0.84 [+0.74, +0.90] / -- / -- | yes |
| OODSelect | -0.97 [-0.98, -0.95] / 0 (0) / 0.11 | -0.97 [-0.98, -0.95] / 0 (0) / 0.11 | yes |
| + smoothing | -0.91 [-0.95, -0.85] / 7 (203) / 0.65 | -0.91 [-0.95, -0.85] / 7 (203) / 0.65 | yes |
| Laplacian | -0.92 [-0.95, -0.86] / 7 (227) / 0.50 | -0.92 [-0.95, -0.86] / 7 (227) / 0.50 | yes |
| + smoothing  | -0.86 [-0.92, -0.78] / 8 (343) / 0.81 | -0.86 [-0.92, -0.78] / 8 (343) / 0.81 | yes |
| GTV | -0.85 [-0.91, -0.75] / 7 (383) / 0.79 | -0.85 [-0.91, -0.75] / 7 (383) / 0.79 | yes |
| + smoothing   | -0.80 [-0.88, -0.69] / 8 (421) / 0.86 | -0.80 [-0.88, -0.69] / 8 (421) / 0.86 | yes |

**tab:regions**

| Row | Paper | New | Same |
|---|---|---|---|
| Orange (east) | 103 / 0.00 / 0.70 | 103 / 0.00 / 0.70 | yes |
| Liberty | 95 / 0.00 / 0.78 | 95 / 0.00 / 0.78 | yes |
| Palm Beach | 50 / 0.02 / 0.50 | 50 / 0.02 / 0.50 | yes |
| Wakulla | 48 / 0.00 / 0.49 | 48 / 0.00 / 0.49 | yes |
| Brevard | 42 / 0.02 / 0.45 | 42 / 0.02 / 0.45 | yes |
| Orange (west) | 35 / 0.00 / 0.59 | 35 / 0.00 / 0.59 | yes |
| Seminole | 25 / 0.00 / 0.56 | 25 / 0.00 / 0.56 | yes |
| Dixie | 23 / 0.04 / 0.61 | 23 / 0.04 / 0.61 | yes |
| All regions | 421 / 0.01 / 0.62 | 421 / 0.01 / 0.62 | yes |
| Other Zone A | 16,846 / 0.06 / 0.29 | 16,846 / 0.06 / 0.29 | yes |
| Zone AE baseline | 10,911 / 0.05 / 0.31 | 10,911 / 0.05 / 0.31 | yes |

**tab:models**

| Row | Paper | New | Same |
|---|---|---|---|
| Decision tree | 40 / 22 | 40 / 22 | yes |
| Random forest | 54 / 47 | 54 / 47 | yes |
| Extra trees | 54 / 45 | 54 / 45 | yes |
| Histogram gradient boosting | 48 / 38 | 48 / 38 | yes |
| XGBoost | 48 / 34 | 48 / 34 | yes |
| Logistic regression | 50 / 32 | 50 / 32 | yes |
| Multilayer perceptron | 45 / 44 | 45 / 44 | yes |
| Gaussian naive Bayes | 30 / 28 | 30 / 28 | yes |
| Total | 369 / 290 | 369 / 290 | yes |

Every number of the prose that make_paper checks (the phrase it looks for is in `numbers.md`):

| Paragraph | Number | New value | Paper says it |
|---|---|---|---|
| Introduction | models | 290 | yes |
| Introduction | per-tile accuracy loss on Zone A | 0.017 | yes |
| Introduction | corr on all Zone A | 0.84 | yes |
| Introduction | OODSelect corr_te | -0.97 | yes |
| Introduction | OODSelect components (>= 20 points) | 0 | yes |
| Introduction | GTV + smoothing: points in regions, regions | 421 of 500, 8 | yes |
| Introduction | GTV + smoothing corr_te | -0.80 | yes |
| Introduction | regions' label disagreement | 62\% | yes |
| Data Construction | points | 96,467 | yes |
| Data Construction | median variogram range | 27 km | yes |
| Data Construction | features correlated beyond 50 km | landcover 72.7 km, log10_dist_coast 95.2 km, wetindexcat 106.9 km | yes |
| Data Construction | 110 km tiles: test tiles | 7 | yes |
| Data Construction | 50 km tiles: tiles, test tiles | 99, 27 | yes |
| Data Construction | training tiles holding Zone AE training points (all training tiles) | 66 (69) | yes |
| Data Construction | model inputs | 28 | yes |
| Data Construction | training / ID test / OOD points | 36,758 / 10,986 / 17,346 | yes |
| Data Construction | flood shares (train, ID, OOD) | 0.50, 0.53, 0.50 | yes |
| Data Construction | missing: soil flood frequency, fluvial class | 2.8%, 2.0% | yes |
| Data Construction | Figure 1: the seven most common land cover classes | 90 woody wetland, 42 evergreen forest, 81 pasture and hay, 95 herbaceous wetland, 82 crops, 21 developed open space, 22 low-intensity developed | yes |
| Metrics | flood share within a tile (min, max over test tiles and both sets) | 0.04 to 1.00 | yes |
| Metrics | AUROC tiles | 25 of 27 | yes |
| Budget penalty | S = 500 as a share of the OOD points | 2.9% | yes |
| Graph | median distance to the 10th nearest neighbour (Zone A test points) | 2.17 km | yes |
| Graph | random-subset purity S/m at S = 500 | 0.03 | yes |
| Models | N | 290 | yes |
| Models | selection / validation / test models | 174 / 58 / 58 | yes |
| Matching | caliper eps, q | 1.30, 6 | yes |
| Matching | Zone AE test points without a feature (not matched) | 75 | yes |
| Results | all three pooled Delta negative | True | yes |
| Results | Palm Beach tile: share of Zone AE / Zone A test points | 14% / 1% | yes |
| Results | Palm Beach tile: models' accuracy on its Zone AE points (all models) | 0.50 | yes |
| Results | tiles where models do worse on Zone A (accuracy, AUROC, F1) | 19/27, 19/25, 22/27 | yes |
| Results | Wilcoxon p, Holm-adjusted (AUROC, F1, accuracy) | 0.044, 0.016, 0.091 | yes |
| Results | corr_te on all Zone A | +0.84 | yes |
| Results | OODSelect corr_te | -0.97 | yes |
| Results | OODSelect: test tiles touched | 25 of 27 | yes |
| Results | GTV + smoothing corr_te [CI] | -0.80 [-0.88, -0.69] | yes |
| Results | GTV + smoothing points in components, components, purity | 421, 8, 0.86 | yes |
| Results | GTV alone corr_te, points, components, purity | -0.85, 383, 7, 0.79 | yes |
| Results | Laplacian corr_te, components, points, purity | -0.92, 7, 227, 0.50 | yes |
| Results | OODSelect + smoothing corr_te, points, purity | -0.91, 203, 0.65 | yes |
| Results | regions | 8 | yes |
| Results | pooled unmatched vs baseline | 0.7% vs 5% | yes |
| Results | largest region unmatched share | 0.043 | yes |
| Results | pooled eta vs baseline | 0.62 vs 0.31 | yes |
| Results | regions above the baseline eta | 8 of 8 | yes |
| Results | region eta range | 0.45 to 0.78 | yes |
| Results | Other Zone A unmatched, eta | 6%, 0.29 | yes |
| Analysis | purity alone -> + smoothing (OODSelect, Laplacian, GTV) | 0.11 to 0.65, 0.50 to 0.81, 0.79 to 0.86 | yes |
| Analysis | corr_te weakens by (OODSelect, Laplacian, GTV), from Table 2's rounded values (unrounded in parentheses) | 0.06 (0.062), 0.06 (0.053), 0.05 (0.043) | yes |
| Analysis | Laplacian + smoothing vs GTV alone: purity, points in components | 0.81 vs 0.79, 343 vs 383 | yes |
| Analysis | GTV + smoothing: points in components, components | 421, 8 | yes |
| Appendix | settings, degenerate, duplicates, N | 369, 1, 78, 290 | yes |
| Appendix | training points | 36,758 | yes |

One number is read from Table 2's rounded values, as the prompt says the Ablation is: "In exchange, corr_te weakens by
0.05 to 0.06" holds for the table's values (-0.97 to -0.91, -0.92 to -0.86, -0.85 to -0.80); unrounded the weakening is
0.062, 0.053 and 0.043, which would read "0.04 to 0.06". Not a disagreement, but the GTV value rounds to 0.04.

### 3.3 The full run (`results/2026-11-12`, commit `64ce67a`)

`caffeinate -i .venv/bin/python run_experiments.py --stage all --jobs 10` (estimate printed before: about 1 h 10 min),
01:07 to 02:20: zoo 32 s (the 369 fits 30 s), data 30 s, shift 1 s, the 55 searches 71 min with 10 at a time (11.1 CPU
hours; median 9.5 min, 2.1 to 37.1 min; S = 500: 4.1 to 21.5 min, mean 9.7), smoothing, regions and matching 2 s. One
invocation, on committed code (no uncommitted file), `complete: true`; no search failed; log
`logs/2026-11-12/run_experiments.log`, one log per search in `logs/2026-11-12/search/`.

- The zoo and the three S = 500 thetas again equal the 2026-11-07 cell and thetas bit for bit
  (`logs/2026-11-12/F_compare_full.log`, ALL MATCH).
- Every search of the full run that the 2026-11-07 run also made (the 39 of the budget grid and the Laplacian at
  mu = 1 and 10) has the same theta bit for bit, so the same subsets (`logs/2026-11-12/F_compare_grid.log`: 41 of 41;
  corr_te differs by at most 1e-16). Figure 3 and the Budget paragraph therefore come out as before.
- `make_paper.py --run results/2026-11-12`: every self-check PASS (`results/2026-11-12/paper/numbers.md`), and 126 of
  126 PASS against the paper: the 90 table cells of section 3.2, identical row for row, and 62 numbers of the prose,
  the 54 of section 3.2 plus these 8 from the budget grid:

| Paragraph | Number | New value | Paper says it |
|---|---|---|---|
| Budget penalty | largest budget at which all three searches invert (corr_te < -0.3) | 10,000 | yes |
| Budget | without smoothing, every search inverts at every budget up to 10,000 | yes | yes |
| Budget | with smoothing, the smallest budget from which all three invert | 50 | yes |
| Budget | corr_te at 15,000 points (86% of Zone A), alone / + smoothing | OODSelect +0.47 / +0.66; Laplacian +0.48 / +0.64; GTV +0.43 / +0.63 | yes ("none does") |
| Budget | GTV the most coherent at every budget 20 to 10,000, alone and + smoothing | yes, yes | yes |
| Budget | GTV + smoothing purity from 50 points up | 0.86 to 0.97 | yes |
| Budget | GTV + smoothing inverts less than the Laplacian + smoothing, 250 points up | 0.05 to 0.16 | yes |
| Budget | Laplacian alone, purity at 20 and 250 points | 0.03, 0.30 | yes |

The Appendix's timing sentence is environment-dependent and not checked as text: this run fitted the 369 settings in
30 s (paper: 32 s), and an S = 500 search took 9.7 min on average on one core (paper: about 10 minutes).


## 4. The mu sweep (S = 500; Figure fig:mu)

Laplacian and GTV at mu in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}, OODSelect (mu = 0) the reference; mu = 3 is
the budget grid's search. corr_te on the 58 test models with its Fisher 95% interval; # Comps = components of at least
20 points (points they hold); r_val = the same subset's correlation on the 58 validation models; soft total = the sum of
the search's final weights (the budget penalty pulls it to 500).

| Loss | mu | corr_te [95% CI] | Purity | # Comps (points) | r_val | + smoothing: corr_te [95% CI] | Purity | # Comps (points) | r_val | soft total |
|---|---|---|---|---|---|---|---|---|---|---|
| OODSelect | **0** | -0.972 [-0.98, -0.95] | 0.106 | 0 (0) | -0.987 | -0.911 [-0.95, -0.85] | 0.653 | 7 (203) | -0.914 | 500 |
| Laplacian | 0.01 | -0.979 [-0.99, -0.96] | 0.210 | 1 (20) | -0.982 | -0.900 [-0.94, -0.84] | 0.732 | 9 (295) | -0.938 | 500 |
| Laplacian | 0.03 | -0.959 [-0.98, -0.93] | 0.205 | 0 (0) | -0.983 | -0.848 [-0.91, -0.75] | 0.675 | 8 (279) | -0.884 | 500 |
| Laplacian | 0.1 | -0.962 [-0.98, -0.94] | 0.183 | 2 (49) | -0.978 | -0.868 [-0.92, -0.79] | 0.712 | 7 (249) | -0.921 | 499 |
| Laplacian | 0.3 | -0.960 [-0.98, -0.93] | 0.205 | 2 (66) | -0.976 | -0.884 [-0.93, -0.81] | 0.674 | 8 (285) | -0.917 | 500 |
| Laplacian | 1 | -0.926 [-0.96, -0.88] | 0.305 | 3 (92) | -0.970 | -0.893 [-0.94, -0.82] | 0.776 | 11 (396) | -0.942 | 500 |
| Laplacian | **3** | -0.918 [-0.95, -0.86] | 0.504 | 7 (227) | -0.963 | -0.865 [-0.92, -0.78] | 0.810 | 8 (343) | -0.932 | 500 |
| Laplacian | 10 | -0.894 [-0.94, -0.83] | 0.626 | 7 (313) | -0.950 | -0.840 [-0.90, -0.74] | 0.857 | 8 (414) | -0.925 | 500 |
| Laplacian | 30 | -0.866 [-0.92, -0.78] | 0.779 | 11 (398) | -0.936 | -0.842 [-0.90, -0.75] | 0.858 | 10 (425) | -0.923 | 500 |
| Laplacian | 100 | -0.833 [-0.90, -0.73] | 0.813 | 9 (378) | -0.928 | -0.791 [-0.87, -0.67] | 0.864 | 8 (391) | -0.909 | 500 |
| GTV | 0.01 | -0.963 [-0.98, -0.94] | 0.150 | 0 (0) | -0.984 | -0.692 [-0.81, -0.53] | 0.629 | 8 (249) | -0.740 | 500 |
| GTV | 0.03 | -0.970 [-0.98, -0.95] | 0.173 | 0 (0) | -0.980 | -0.875 [-0.92, -0.80] | 0.697 | 12 (322) | -0.916 | 495 |
| GTV | 0.1 | -0.955 [-0.97, -0.93] | 0.299 | 4 (101) | -0.975 | -0.889 [-0.93, -0.82] | 0.718 | 9 (315) | -0.930 | 501 |
| GTV | 0.3 | -0.919 [-0.95, -0.87] | 0.592 | 8 (296) | -0.971 | -0.882 [-0.93, -0.81] | 0.798 | 9 (365) | -0.942 | 500 |
| GTV | 1 | -0.938 [-0.96, -0.90] | 0.193 | 1 (23) | -0.958 | -0.862 [-0.92, -0.78] | 0.715 | 6 (203) | -0.916 | 248 |
| GTV | **3** | -0.846 [-0.91, -0.75] | 0.786 | 7 (383) | -0.921 | -0.803 [-0.88, -0.69] | 0.858 | 8 (421) | -0.920 | 498 |
| GTV | 10 | -0.757 [-0.85, -0.62] | 0.916 | 7 (475) | -0.912 | -0.673 [-0.79, -0.50] | 0.775 | 7 (339) | -0.820 | 500 |
| GTV | 30 | -0.798 [-0.88, -0.68] | 0.695 | 4 (439) | -0.924 | -0.801 [-0.88, -0.68] | 0.912 | 4 (479) | -0.926 | 500 |
| GTV | 100 | -0.695 [-0.81, -0.53] | 0.725 | 9 (325) | -0.693 | -0.543 [-0.70, -0.33] | 0.852 | 9 (363) | -0.443 | 500 |

`CS371_Cho_Li/figures/fig_mu.pdf` draws the left half (each search alone): corr_te on top, purity below, one line per
loss, OODSelect dashed, dotted line at mu = 3.

**The XX of "Regularization weight"** (end points mu = 0.01 and mu = 100):

> As $\mu$ grows, purity rises from **0.15** to **0.73** for GTV and from **0.21** to **0.81** for the Laplacian, while
> $\text{corr}_\text{te}$ weakens from **$-0.96$** to **$-0.70$** and from **$-0.98$** to **$-0.83$**. At the same
> purity, **XX** inverts more strongly. A user who needs a stronger inversion can therefore lower $\mu$ or choose
> **XX**, at the cost of coherence.

The first eight are the end points as the sentence asks for them; the last two have no clean answer (below), so I left
them open. **Answers to the % CHECK:**

1. *Does purity rise and corr_te weaken as mu grows?* **For the Laplacian, yes from mu = 0.3 up**: its purity is flat at
   0.18 to 0.21 for mu up to 0.3, then rises at every step (0.31, 0.50, 0.63, 0.78, 0.81); corr_te weakens from -0.98
   to -0.83, with one step of 0.003 the other way (0.03 to 0.1). **For GTV, only as a trend** (Spearman +0.82 for
   purity, +0.95 for corr_te), not step by step: purity climbs from 0.15 to 0.92 at mu = 10, except at mu = 1 (0.19),
   then falls to 0.70 at mu = 30 and 0.73 at mu = 100; corr_te weakens from -0.96 to -0.70 with bumps at mu = 0.03, 1
   and 30.
   So "purity rises from 0.15 to 0.73 for GTV" is true of the end points but hides its peak of 0.92 at mu = 10.
2. *Two GTV runs explain most of the irregularity, and neither is a code fault*; both come from the specified search
   (the best of 100 TPE trials by validation correlation): at **mu = 1** the winning trial drew the budget weight
   lambda = 4e-6, near the bottom of its range (1e-6), so the squared penalty was almost off and the soft total ended at 248
   instead of 500 (the top 500 of theta then include many low-weight points); this is the case the Parked caveat of
   `docs/TODO-CLAUDE-CODE.md` describes (runs with lambda near 1e-6 end far from S). At **mu = 100** the spatial term
   dominates: most of the 100 trials did not invert on the validation models (median validation r +0.66; the best,
   -0.69, is far weaker than at any other mu). Every other S = 500 search ended within 5 points of S.
3. *At the same purity, which loss inverts more strongly?* **Neither, consistently.** Interpolating each loss's corr_te
   over purity, over the purities both reach (0.18 to 0.81) GTV inverts more on 55% of the range and the Laplacian on
   45%; the mean difference is 0.014. Point by point the two are within 0.04 of each other, except GTV at mu = 30 and
   100, which invert 0.08 and 0.18 less than the Laplacian at their purity. At the operating point the paper uses,
   GTV at mu = 3 (purity 0.79, -0.85) and the Laplacian at mu = 30 (0.78, -0.87) are 0.02 apart.

Not applied, for Victor and Claude (chat), who write the prose: one option for the last two sentences is "At the same
purity, the two losses invert about equally (within 0.04), except GTV at the largest weights, where its search no
longer finds a strong inversion. A user who needs a stronger inversion can therefore lower $\mu$, at the cost of
coherence." The mu = 3 choice itself is supported: it is the smallest weight at which both losses form regions with a
purity of at least 0.5 while corr_te stays below -0.84.


## 5. The component export

`.venv/bin/python export_components.py --method "GTV+smoothing"` (S = 500, mu = 3, c = 20, run `results/2026-11-12`):
`exports/GTV+smoothing_S500_mu3_c20_2026-11-12/` and `exports/GTV+smoothing_S500_mu3_c20_2026-11-12.zip` (381 KB), the one
file to send:

| File | Contents |
|---|---|
| `README.md` | what the regions are (plain words), how they were produced (method, S, mu, c, run folder, git commit, caliper), the CRS of every coordinate, a dictionary of every column, how to open the files in QGIS, ArcGIS, Python and Excel |
| `regions.gpkg` (EPSG:5070) | layers `points` (421), `regions` (8 convex hulls, for display only, as the README says), `selected_not_in_component` (79), `matches` (2,066 matched Zone AE points) |
| `points.csv` | the 421 region points, sorted by region then point id |
| `components/01_orange-east.csv` ... `08_dixie.csv` | one file per region, the same columns |
| `components_summary.csv` | per region: id, name, points, counties, tiles, centroid, extent (km), share labeled flood, matched points, unmatched share, eta |
| `selected_not_in_component.csv` | the 79 selected points outside every region, with the small piece each belongs to |
| `matches.csv` | region point, its label, match rank, matched Zone AE point, its lon/lat (and EPSG:5070 x/y), label, feature distance |

Point columns: `component_id`, `component_name`, `point_id`, `lon`, `lat` (NAD83, EPSG:4269), `x_5070`, `y_5070`,
`zone` (A), `label`, `fema_polygon_id` (`meta_fld_ar_id`; empty for negatives, which lie outside every polygon),
`dfirm_id`, `county`, `county_fips`, `tile_50km`, the nine features as the models see them, `landcover_name`,
`soil_floodfreq_name`, `dist_river_m`, `dist_coast_m`, then `share_models_correct` (of the 290 models), `n_matches`,
`match_label_disagreement`. Units were checked against `FEATURES.txt` and the pull code: slope in degrees, terrain
variability and HAND in metres, wetness as StreamCat serves it (index x 100), distances in metres in EPSG:5070 with the
model features their log10 floored at 1 m (the stored columns satisfy log10_dist = log10(max(dist_m, 1)) exactly).

Checks: the 8 regions' points, matched points, unmatched shares and eta equal the run's Table 3 (`match/table.csv`)
exactly; every GeoPackage layer reads back with `geopandas.read_file` in EPSG:5070 with the CSV's row counts; Excel
opened `points.csv` (422 rows with the header, 30 columns, the right headers and values) and closed it unsaved.
Region rows: Orange (east) 103, Liberty 95, Palm Beach 50, Wakulla 48, Brevard 42, Orange (west) 35, Seminole 25,
Dixie 23.


## 6. What looked wrong or unclear, and what I did not do

1. **The paper is not in the snapshot commit.** `CS371_Cho_Li/` is in `.gitignore`, so `git add -A` left it out (the
   prompt expected the paper among the uncommitted changes). I did not force-add it: the folder is ignored on purpose
   (Overleaf). `acl_latex.tex` was not edited; `make_paper.py` only wrote the four figures (old versions moved to
   `figures/old/`) and this run's `CHANGES-2026-11-12.md`.
2. **The Ablation's "0.05 to 0.06"** is read from Table 2's rounded values (section 3.2); unrounded, GTV weakens by
   0.043.
3. **Two points tie at the GTV + smoothing cut** (their smoothed weights are equal in floating point, as on
   2026-11-11). The stable sort keeps the one with the lower index, as the spec says; the other tie-break would swap
   one of the 500 points.
4. **The data build's network requests changed** (fewer Earth Engine bands per request, StreamCat asked for the
   wetness index only). Every computation of the nine is unchanged and an offline rebuild reproduces the dataset
   exactly, but no pull was rerun, as instructed. Also from the data-build trim: `pull_soil_points.py crosscheck`
   still reads the cached SSURGO polygons, whose download code is now deleted (the cache stays under `data/`);
   `subset_nfhl.py` is unchanged, so its pass 1 still writes a ten-state attribute table that only the deleted
   `o5_polygons.py` read.
5. **The GIS Error Analysis placeholder** in the paper says "categorize the 29 regions"; the analyzed method has 8.
6. **Older documents name deleted scripts** (`docs/DECISION-PROVENANCE.md`, `docs/RUN-FL-V2.md`, the earlier task and
   run reports, `logs/`). They are history and were left as they are.
7. **Figure fig:mu has one column** (each search alone), as its caption describes (top corr_te, bottom purity); it
   adds OODSelect (mu = 0) as a dashed reference line, which the caption does not mention yet. The smoothed values
   are in the table of section 4 and in `numbers.md`; a second column like Figure 3's is a small change if wanted.
8. **The export's `county_fips`** is the county the point lies in (`meta_point_county`), so that it agrees with
   `county`; the dataset's own `county_fips` is the county of the point's flood-map database, which differs for 14%
   of negatives. The README says so.
9. **Timings in the Appendix** ("32 seconds", "about 10 minutes on one CPU core"): this run fitted the 369 settings in
   30 s and an S = 500 search took 9.7 min on average (4.1 to 21.5) with 10 running at once; budgets of 5,000 and more
   took up to 37 min. Both sentences still hold.
10. **`exports/` is new and untracked**; I added it to `.gitignore` with `results/` (generated outputs).
11. **Figures**: `make_paper.py` moved the replaced figures to `CS371_Cho_Li/figures/old/` (`fig_feature_boxes_v2.pdf`,
    `fig_maps_v10.pdf`, `fig_budgets_v2.pdf`) and wrote the new ones; the three redrawn figures show the same data as
    before. `fig_mu.pdf` was drawn twice (I tightened its corr_te axis after the first draw), so its first draft is
    `figures/old/fig_mu_v1.pdf`.
12. **Not done**: no network pull, no edit of the paper's prose, nothing pushed. The prompt's Overleaf steps (upload
    the figures, paste the tables and numbers) are Victor's (`CS371_Cho_Li/CHANGES-2026-11-12.md`).
