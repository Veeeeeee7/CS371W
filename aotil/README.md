# aotil: OODSelect with a spatial penalty, and the analyses of its regions

The model and selection code behind the paper (`CS371_Cho_Li/acl_latex.tex`). Models trained on FEMA Zone AE
points (detailed studies) are scored on held-out Zone AE points (the in-distribution half, ID) and on Zone A
points (approximate studies, the out-of-distribution half, OOD) in the same 50 km tiles. OODSelect (Salaudeen
et al. 2025) searches for Zone A subsets where accuracy-on-the-line inverts: across models, higher ID accuracy
goes with lower accuracy on the subset. A graph-Laplacian penalty on a 10-nearest-neighbour graph of the Zone A
points (or graph total variation, `--reg gtv`) makes those subsets connected regions, and the other modules
describe the regions.

Terms: a *cell* is one model population, (state, negative protocol, training zone, seed), stored as `<tag>.npz`
and `<tag>_models.csv` (e.g. `FL_band_AE_s0`). A *lobe* is one single-part flood-zone polygon. A *level set* is
a curve of constant ground elevation, like a contour line. EPSG:5070 and EPSG:3086 are projections in metres.

## Modules

| file | what it does |
|---|---|
| `zoo.py` | fits a cell's 300 models (dt, et, hgb, lr, mlp, nb, rf, xgb) and stores each model's probability of the true class on every ID and OOD point |
| `cells.py` | `load_cell`, the one way every stage reads a cell; the tree families of the family-disjoint split |
| `audit_cell.py` | re-derives a cell from the point table and the code (shapes, model table, rows, tiles, refits); PASS or FAIL |
| `graph.py` | the kNN graph (EPSG:3086), its Laplacian and the coherence metrics (purity, cut share, components, tiles); `python aotil/graph.py` runs its built-in checks |
| `oodselect_paper.py` | OODSelect as its authors released it, plus the optional spatial term (`--reg laplacian`, the default, or `--reg gtv`, graph total variation on the same edges, folder `spatial_gtv_mu<mu>_knn<k>`); one run = (cell, budget S, variant) |
| `report_paper.py` | per cell and variant: r on the held-out models with a Fisher interval, the random and most-misclassified baselines, cross-budget overlap (the paper's Eq. 6), the model-count rule, classic vs spatial |
| `audit_selection.py` | recomputes every number a selection run stored from its θ and the cell; `--smoothed-classic` adds that ablation |
| `o1_shift.py` | O1, the shift from the Zone AE half to the Zone A half in accuracy (primary), AUROC and F1 (all points, land, paired land; tile bootstrap), twice: pooled over the test points, and per tile (`tile_` columns: each gap within a test tile, ID points weighted to the tile's Zone A share of positives, averaged over tiles); writes a per-tile and a per-family table beside the CSV; and O2, the aggregate ID-OOD correlation |
| `o1_tile_null.py` | null check of the per-tile gap: zones shuffled within each (tile, label) group, same models, no refit |
| `o1_table.py` | Table 1 (tab:shift) as LaTeX, pooled and per-tile rows, from an `o1_shift.py` log |
| `regions.py` | the connected components of a selection's top S points on the same kNN graph, one row per region |
| `region_match.py` | the paper's region analysis (Approach 3.6): matches each region point to held-out Zone AE points with the same land cover within a caliper on the 6 standardized numeric inputs (at most 5, never from a tile of the region), and reports the unmatched share (covariate shift) and the label disagreement η (concept shift) per region, pooled, for the Zone AE baseline, reference sets and halos; `--exclude own-tile` reproduces the 2026-10-25 sandbox rule |
| `region_recurrence.py` | whether a seed's regions come back in the other seeds |
| `lobes.py` | per region: its Zone A lobes, matched unselected Zone A lobes and Zone AE lobes in its tiles, with edge vertices every 30 m |
| `edge_layers.py` | distance from every edge vertex to soil-survey lines, county lines, map-panel edges and canals, with coverage checks |
| `boundary_stat.py` | O5 boundary statistic: share of Zone A edge that is not a level set and lies on a soil or map line, with a displaced-edge null |
| `match_controls.py` | O5 S1: a terrain-matched unselected Zone A control for every region land point |
| `region_null.py` | O5 S2: random connected patches of unselected Zone A points, grown to each region's size |
| `o5_interpolation.py` | O5 Part A: does the inversion survive without the models that memorize their training labels? |
| `o5_contrast.py` | O5 Part B: regions vs their terrain-matched controls on about 40 non-terrain descriptors |
| `o5_shift_decomp.py` | O5 Part C: covariate shift or concept shift (support, label conflict, reweighting) |
| `o5_feature_inclusion.py` | O5 Part D1: which features the models furthest below the line saw |
| `o5_lofo.py` | O5 Part D2: leave-one-feature-out populations scored on the fixed selections |
| `o5_polygons.py` | O5 Part E: how the region polygons were drawn (vertex spacing, straight runs, study type, panels, map revisions) |
| `o5_pooling.py` | O5 Part F: what adding the region labels to training costs on Zone AE |

Not in the paper: the boundary statistic, Part E and the family-disjoint split (`CS371_Cho_Li/CHANGES-2026-10-19.md`);
the terrain-matched controls, the matched-growth null, label conflict, pooling and leave-one-feature-out
(`CHANGES-2026-10-19b.md` §2). The paper's region analysis is `region_match.py` (since 2026-10-25). It replaces the
*twins* (patches of held-out Zone AE points matched on all seven inputs, with the distance d and accuracy gap g),
whose scripts `results/<date>/paper_figures/region_twins_ref.py`, `region_twins_summary.py` and `fig_twins.py` are
superseded: do not run them for the paper.

## Run order

1. `docs/RUN-FL-V2.md`: raw inputs and the Florida point table, then a 45-feature seed-0 pass (cell, selections,
   regions, lobes, DEM, soil lines, distances).
2. `docs/TASK-FULL-RUN-W7.md` Part C, the seven-feature run: cells for seeds 0-2, O1/O2 (`o1_shift.py`), selections
   and controls, regions and their recurrence, lobes and distances (RUN-FL-V2 steps 12-15), `boundary_stat.py`.
3. `docs/TASK-O5-REGIONS.md`: S1 (`match_controls.py`), S2 (`region_null.py`), then Parts A-F (`o5_*.py`).
4. `docs/TASK-REGION-TWINS.md`: the twins analysis (superseded by `region_match.py`).
5. `docs/TASK-RERUN-ALL-2026-10-25.md`: the paper's current numbers: the refit cell, every selection with the squared
   budget penalty (Laplacian and GTV), the downstream analyses and `region_match.py`; the commands are in
   `logs/2026-10-25/C_chain.sh` and `D_chain.sh`.

`run_oodselect_local.sh` (repo root) builds each cell in `CELLS` with `zoo.py` (skipped when it exists), runs
`oodselect_paper.py` for every (variant, budget) as single-threaded processes, `JOBS` at a time, then
`report_paper.py`. A budget whose `_result.csv` exists is skipped, so redo a run under a new `--date`. Its defaults
are the 45-feature table and both directions, so the routines always set `CELLS` (the seven-feature run also
`DATA_DIR=data/processed_w7 CELLS_DIR=data/interim/aotil/cells_w7`). Other knobs: `SEED`, `VARIANTS`, `MU_SPATIAL`
(3), `KNN` (10), `BUDGETS`, `NULL_SEEDS`, `SPLITS`, `BUDGET_PENALTY` (squared) and `REG` (laplacian; `REG=gtv`
with `VARIANTS=spatial` runs GTV). Outputs the 2026-10-21 code review made stale: `docs/RUN-FL-V2.md`.

## How `oodselect_paper.py` follows the authors' code

It reimplements the selection from the authors' release (github.com/olawalesalaudeen/OODSELECT); each function
names the part it mirrors. Where the paper and the code differ, it follows the code, which produced the paper's
numbers. As in the release:
- score: correctness of each model's argmax prediction (`extract_data`), rebuilt from the stored true-class
  probability, a p = 0.5 tie going to class 0; ID performance is plain accuracy;
- models split 60/20/20 into selection, validation and held-out groups by sklearn `train_test_split` at
  `random_state=42` (`prepare_dataset`);
- loss: the Pearson r of probit accuracies, clamped to [1e-6, 1 - 1e-6] (`PearsonRLoss`); weights `sigmoid(theta)`;
  the subset is the top S of theta (`get_binary_selection`);
- budget term: by default a one-sided `softplus(S - sum of weights)` with a cosine warm-up, not the quadratic
  penalty of the paper's Eq. 5 (Salaudeen et al. 2025, p. 4). The soft mass then ends far above S (S = 500, W7 run:
  3,850-7,169 classic, 4,208-14,281 spatial), and the S largest weights form the subset. Every run for our paper
  passes `--budget-penalty squared` (Eq. 5, which our Approach states; `run_oodselect_local.sh` does so by default),
  with lambda searched log-uniform over [1e-6, 100]; the soft total then ends near S;
- Adam with `CosineAnnealingLR`, early stopping after 500 epochs without a better validation r (at most 50,000),
  the theta at the stopping epoch kept; Optuna, 100 trials over the five released hyperparameters;
- the authors' budget grid below the OOD set size, Fisher 95% intervals, r <= -0.3 counted as an inversion.

Where it departs:
- `--mu > 0` adds mu * z'Lz / (S k) to the training loss. For a 0/1 subset z'Lz counts the graph edges it cuts, so
  the term is an incoherence score in [0, ~1]. Validation, the Optuna objective and the held-out scoring are
  unchanged; mu = 0 is the authors' objective, and mu is fixed at 3, never tuned on r.
- `--reg gtv` replaces z'Lz by the graph total variation, the sum of |z_j - z_j'| over the same edges (each
  undirected edge once), with the same S k normalization; on a 0/1 subset both count the cut edges. Its gradient is
  autograd's through `torch.abs`, whose subgradient at a tie is 0. mu stays 3, not tuned.
- A plain loop replaces the PyTorch Lightning Trainer (same arithmetic, about 10x faster). A trial whose r turns
  NaN is recorded as failed and the search goes on; the release stops.
- theta's start is seeded (the release does not seed it), and each budget gets its own seed, seed * 1,000,003 + S,
  so budgets are selected independently, as the paper's overlap measure assumes.
- `report_paper.py` reports each budget from its own selection; the authors' plots show size n from the selection
  made at the next larger size.

Check 1 of `test_oodselect_paper.py` runs the authors' `TestSetFinder` (copied verbatim) under Lightning against
`train_one`. It is skipped without `pytorch_lightning`, which `.venv` lacks, so it has never run here.

## Controls

- **Random subsets** (`report_paper.py`; the paper's "Random"): 100 random subsets of size S per budget, scored on
  the held-out models. They should follow the aggregate trend.
- **Most-misclassified points** (`hard_r`; the paper's "Hardest"): the S points with the highest 1 - mean
  correctness over the 60 held-out models, averaged over 100 random orders of the many points tied at the cut.
  `hard_sel_r` ranks on the 180 selection models (not in the paper). An r near 0 separates an inversion from
  plain difficulty.
- **Null-select** (`--null-seed N`; `NULL_SEEDS="1 2 3"`, folders `<variant>_null<N>`; the paper's "Shuffled"):
  the selection and validation models' ID accuracies are permuted, the same search runs with the same seed, and
  the kept subset is scored against the held-out models' true accuracies.
- **Family-disjoint splits** (`--split trees->others` or `others->trees`; `SPLITS`, folders `<variant>_trees2others`,
  `<variant>_others2trees`): select on hgb, rf, et, dt and xgb and score on lr, mlp and nb, or the reverse; 25% of
  the selection family is held out for validation. A random split lets both groups share one quality axis.
- **Smoothed classic** (`audit_selection.py --smoothed-classic`; the paper's "Smoothing after search"): each point
  takes the mean of the classic weights over its 10 nearest points (itself included) and the top S are kept, so
  better regions can be credited to the penalty rather than to smoothing.

## The owner's check scripts

Each `aotil/test_<module>.py` checks one module on synthetic cases with known answers, then recomputes its stored
outputs where they exist; exit code 0 means every check passed. Run them from the repo root; by default they read
the stored outputs of 2026-10-16/17, and they write nothing under `data/`, `results/` or `logs/`:

```bash
for f in aotil/test_*.py; do .venv/bin/python "$f" > /dev/null 2>&1 && echo "ok    $f" || echo "FAIL  $f"; done
```

Rerun a failing one alone to see the failed check. `--csv` (Parts A, D2, F), `--out-dir results/$D/o5` (S1/S2, B, C)
and `--null-file` (E) point them at new outputs. After the review all exit 0 except `test_o5_shift_decomp.py`, whose
stored Part C label-scope rows predate a fix; rerunning Part C of `docs/TASK-O5-REGIONS.md` clears it.
