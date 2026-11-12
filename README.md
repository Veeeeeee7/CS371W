# Mapping generalization failures in FEMA flood maps (CS371W)

Code for the paper in `CS371_Cho_Li/acl_latex.tex`: models trained on FEMA Zone AE flood-map points (detailed
studies) are tested on Zone A points (approximate studies); OODSelect, and OODSelect with a Laplacian or graph total
variation (GTV) spatial loss, search for Zone A subsets where accuracy-on-the-line inverts, and the regions those
subsets form are matched to held-out Zone AE points to separate covariate shift from concept shift.

Everything runs from the repo root in `.venv` (Python 3.12, torch, optuna, xgboost, scikit-learn, geopandas, pyproj).

## The three commands

| Command | What it does | Time (Apple M4 Pro) |
|---|---|---|
| `.venv/bin/python run_experiments.py --stage all` | every computation, into `results/<today>/` | about 1.5 h with 8 searches at a time |
| `.venv/bin/python make_paper.py` | tables, numbers and figures from the latest complete run | about 15 s |
| `.venv/bin/python export_components.py --method "GTV+smoothing"` | the regions as a GeoPackage + CSVs + README, zipped | a few seconds |

Each has `--help` with examples.

### 1. `run_experiments.py`: the computation (the only script that needs torch)

Stages, each resumable (a finished unit is skipped; `--force` redoes it) and selectable with `--stage`:

| Stage | Output in the run folder | Paper |
|---|---|---|
| `zoo` | `zoo/`: the 369 grid settings fitted on the Zone AE training points, 290 kept (1 degenerate, 78 duplicates), scored on the ID and OOD test points | Models, Table tab:models |
| `data` | `data/`: set sizes, flood shares, missing values, tile counts (50 and 110 km), variogram ranges, the 10-NN distance, Figure 1's points | Data Construction, Figure 1 |
| `shift` | `shift/`: pooled and per-tile accuracy, AUROC and F1, t-intervals, Wilcoxon + Holm, Palm Beach, corr_te on all Zone A | Table 1, "Aggregation hides the shift" |
| `search` | `search/<unit>/`: 55 searches, one single-threaded process each, `--jobs` at a time: OODSelect, Laplacian (mu 3), GTV (mu 3) at 13 budgets (10 to 15,000), and the mu sweep at S = 500 (Laplacian and GTV at mu 0.01 to 100) | Table 2, Figures 3 and fig:mu |
| `smooth` | `smooth/`: every search's subset alone and + smoothing: corr_te [95% CI], validation correlation, components, purity | Table 2, Ablation, Figures 3 and fig:mu |
| `regions` | `regions/`: the components of GTV + smoothing at S = 500 and their county names | Figure 2, Table 3 |
| `match` | `match/`: caliper matching of each region, all regions, Other Zone A, Zone AE baseline | Table 3 |

`manifest.json` records the git commit, the settings, seeds, package versions, input file hashes and timings.
Logs go to `logs/<run folder name>/`. Useful flags: `--out`, `--jobs`, `--dry-run` (the plan and its estimated
time), `--only` (search names such as `gtv_mu3_S500` or `'*_S500'`). Run the full thing in the background with
`caffeinate -i` so the Mac does not sleep.

### 2. `make_paper.py`: tables, numbers, figures (no torch)

Reads one run folder (`--run`, default the latest complete one) and

- prints Tables 1-3 and the Appendix table in the terminal, and writes paste-ready LaTeX tabular blocks in the paper's
  format to `<run>/paper/tables/`;
- writes `<run>/paper/numbers.md`: every number the prose states, by paragraph (including the XX of the
  "Regularization weight" paragraph), the mu sweep table, the run's self-checks, and a check of every table row and
  number against `CS371_Cho_Li/acl_latex.tex` (PASS when the paper already says it);
- draws `fig_feature_boxes.pdf`, `fig_maps.pdf`, `fig_budgets.pdf` and `fig_mu.pdf` into `CS371_Cho_Li/figures/`
  (a replaced file moves to `CS371_Cho_Li/figures/old/<name>_vN.pdf` first; `--no-copy` leaves the paper's folder
  alone), with PNG previews in `<run>/paper/figures/`.

### 3. `export_components.py`: the regions for GIS

`--method` is one of OODSelect, OODSelect+smoothing, Laplacian, Laplacian+smoothing, GTV, GTV+smoothing; `--S`
(500), `--mu` (3), `--min-size` (20) and `--run` (latest complete) default to the paper's settings. Writes
`exports/<method>_S<S>_mu<mu>_c<c>_<run date>/` (README, `regions.gpkg` in EPSG:5070 with layers points, regions,
selected_not_in_component and matches, `points.csv`, one CSV per region, `components_summary.csv`,
`selected_not_in_component.csv`, `matches.csv`) and a `.zip` of it.

## Layout

- `aotil/`: the library the three scripts import (`aotil/README.md` lists the modules).
- Data build (run once, needs Earth Engine and network access; not part of the three commands): `fetch_nfhl.sh`,
  `subset_nfhl.py`, `pull_gis_lines.py`, `pull_land_mask.py`, `sample_points.py`, `fetch_nhdplus_fl.sh`,
  `prep_nhdplus_units.py`, `pull_hydro.py`, `pull_soil_points.py` (with `pull_ssurgo.py`'s Soil Data Access helpers)
  and `pull_gee.py` produce the points and the nine features of `data/processed_v4/dataset_FL.parquet`
  (`data/processed_v4/FEATURES.txt` names each feature's source).
- `docs/`: task briefs and run reports, newest first by date in their names.

## Checks

Each check script exits 0 when every check passes:

```bash
for f in test_entry_points.py aotil/test_*.py test_pull_*.py test_sample_points.py; do .venv/bin/python "$f" > /dev/null 2>&1 && echo "ok    $f" || echo "FAIL  $f"; done
```
