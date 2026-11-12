# aotil: the library behind the three entry points

`run_experiments.py`, `make_paper.py` and `export_components.py` (repo root) import these modules; none of them is
run on its own (search.py's command line is the worker `run_experiments.py` starts for each search). Models trained
on FEMA Zone AE points (detailed studies) are scored on held-out Zone AE points (in-distribution, ID) and on Zone A
points (approximate studies, out-of-distribution, OOD) in the same 50 km test tiles. Terms: a *cell* is one model
population, stored as `<tag>.npz` and `<tag>_models.csv` (the paper's is `FL_band_AE_s0`); a *component* (region)
is a connected piece of at least c = 20 selected points in the 10-nearest-neighbour graph of the Zone A test points.
EPSG:5070 and EPSG:3086 are projections in metres.

| Module | What it does | Used by |
|---|---|---|
| `zoo.py` | the 369 settings of the model grids (Table tab:models), one-hot inputs for land cover and the two soil classes, the drop rules (degenerate: flood predicted for < 5% or > 95% of the ID test points; duplicate: the same 0/1 predictions as an earlier setting), the 70/30 tile split; writes the cell | stage zoo, data |
| `cells.py` | `load_cell`, the one way every stage reads a cell | search, stages |
| `shift.py` | Table tab:shift: pooled and per-tile accuracy, AUROC and F1 over all models, t-intervals over the tiles, Wilcoxon signed-rank + Holm, Palm Beach, corr_te on all Zone A | stage shift |
| `search.py` | the subset search (Algorithm alg:spatial): OODSelect as its authors released it (probit Pearson loss, squared budget penalty, Adam with a cosine learning rate, 100 Optuna TPE trials, patience 500), plus the Laplacian or GTV loss mu R(s) / (S k); one process per search | stage search |
| `graph.py` | the symmetric 10-NN graph in EPSG:3086, its Laplacian, purity, components (ids by size) | search, smooth, regions, figures |
| `subsets.py` | a subset evaluated: corr_te with its Fisher interval, validation correlation, components, purity, tiles; the post-hoc smoothing (I + D)^-1 (I + A) s (Equation eq:smooth), computed on the complement sigmoid(-theta) | stage smooth, export |
| `regions.py` | the components of a subset and their names (the county holding most points; west/east when two share one) | stage regions, export |
| `region_match.py` | the Region Analysis: caliper matching (95th percentile of the Zone AE nearest-match distance, same land cover, up to 5 matches, outside the region's tiles) and eta; rows for each region, all regions, Other Zone A and the Zone AE baseline | stage match, export |
| `variogram_ranges.py` | the effective range of each feature's semivariogram (the Split paragraph's 27 km median) | stage data |
| `fig_feature_boxes.py` | Figure 1 (the features by set) | stage data, make_paper |
| `figures.py` | Figure 2 (maps), Figure 3 (budgets), Figure fig:mu | make_paper |
| `paper_numbers.py` | every table and number of the paper from a run folder, the self-checks and the check against the paper's text | make_paper |
| `synthetic.py` | small synthetic datasets and cells for the checks | tests only |

## How `search.py` follows the authors' code

It reimplements the selection from the authors' release (github.com/olawalesalaudeen/OODSELECT); each function names
the part it mirrors. As in the release: correctness of each model's argmax prediction (a p = 0.5 tie goes to class 0);
models split 60/20/20 into selection, validation and test groups by sklearn `train_test_split` at `random_state=42`;
the Pearson r of probit accuracies clamped to [1e-6, 1 - 1e-6]; weights sigmoid(theta), the subset the top S of theta;
Adam with `CosineAnnealingLR`; early stopping after 500 steps without a better validation r (at most 50,000); the theta
at the stopping step kept; Optuna over the five released hyperparameters; Fisher 95% intervals. Where it departs: the
budget term is the published paper's squared penalty (S - ||s||_1)^2 with lambda drawn log-uniformly from [1e-6, 1e2];
mu > 0 adds the spatial loss to the training loss only (validation and the Optuna objective are unchanged; mu is fixed,
never tuned); a plain loop replaces the PyTorch Lightning Trainer (same arithmetic); a trial whose r turns NaN is
recorded as failed and the search goes on; theta's start is seeded, and each budget gets its own seed
(seed * 1,000,003 + S), so budgets are selected independently and every search is reproducible on one thread.

## Checks

`aotil/test_<module>.py` checks a module on synthetic cases with known answers (and on the stored 2026-11-07 cell
when it exists); `test_entry_points.py` (repo root) runs the three scripts on a synthetic dataset. Exit code 0 means
every check passed.
