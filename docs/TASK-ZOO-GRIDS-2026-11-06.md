# Rebuild the model population from fixed hyperparameter grids (2026-11-06)

Decided by Victor on 2026-11-06. Written by Claude in a claude.ai chat. Victor comes from CS, not
GIS. Explain GIS terms in CS terms in anything you write for him.

## 0. What Victor decided

- **Models differ only by family and hyperparameter setting.** Every model trains on all 7
  features and all training points of the cell. The per-model random feature share
  (`feat_frac`) and row share (`row_frac`) go away.
- **One model per grid setting.** Each family has the fixed grid in §2. Every combination is
  one model, fitted once. No setting repeats, and the families no longer take turns
  (`i mod 8`). Some settings are deliberately weak, to spread ID accuracy.
- **One drop rule, applied after fitting and before the 60/20/20 model split.** It looks only at
  predictions, never at labels. *Degenerate:* the model predicts flood for fewer than 5% or more
  than 95% of the ID test points. Such a model's accuracy on any subset only reflects the
  subset's share of positives, which the search could exploit. In Claude's preview, keeping the
  degenerate models turned the correlation on random 500-point dry-only subsets from +0.19 to
  -0.47, a free inversion.
- **Duplicates stay (Victor, 2026-11-06).** Models whose 0/1 predictions match another model's
  are different settings and stay in the population. Do not drop them. Report how many there are
  (Part B).
- **Everything else stays as on 2026-10-25:** dataset `data/processed_w7`, features,
  tile split (cell seed 0), prediction rule (flood iff p > 0.5, a tie counts as dry), the
  60/20/20 model split (`random_state` 42, now over the kept models), the search settings
  (squared penalty, T = 100, patience 500, the budget grid), S = 500, k = 10, c = 20, mu = 3, and
  every control.
- **The per-tile metrics in the paper are unweighted** (decided 2026-11-04): the
  `tile_acc_raw_*` and `tile_f1_raw_*` columns of `aotil/o1_shift.py`, and the same columns of
  `aotil/o1_tile_null.py`. Per-tile AUROC was never weighted. Do not change the code's weighting;
  just read these columns for every paper number and claim.

**Claude's preview.** Claude fitted the grid once in a scratch script on Victor's machine (VM
Python with scikit-learn 1.7.2 and XGBoost 3.2.0, not the `.venv` versions 1.9.1 and 3.4.1, and
not the zoo's seeds). Script and per-setting results: `Claude outputs/zoo-grid-preview-2026-11-06/`.
Expect small differences from your fit.

| Family | Settings | Degenerate (dropped) | Kept | Of the kept, repeat an earlier model's predictions |
|---|---|---|---|---|
| dt | 40 | 0 | 40 | 16 |
| et | 54 | 1 | 53 | 0 |
| hgb | 48 | 3 | 45 | 9 |
| lr | 50 | 0 | 50 | 13 |
| mlp | 45 | 2 | 43 | 0 |
| nb | 30 | 0 | 30 | 0 |
| rf | 54 | 0 | 54 | 4 |
| xgb | 48 | 3 | 45 | 10 |
| **Total** | **369** | **9** | **360** | **52** |

The 360 kept models span ID accuracy 0.510 to 0.806. Probit r(ID, OOD) over them: 0.84 (the
current 300-model cell: 0.88). With 360 models the 60/20/20 split gives 216/72/72. Of the kept
models below 0.70 ID accuracy, most are naive Bayes, so the weak end leans on one family (see
Part C, check N1).

## 1. Rules

- `export D=<today>`. Outputs go under `results/$D/` and logs under `logs/$D/`. Fit the new cell
  into `data/interim/aotil/cells_grid_$D`. Do not change earlier results or cell folders.
- Use `.venv/bin/python`. Every new or changed function gets a test.
- Repo policy: delete code you retire; do not comment it out.
- Do not commit or push.
- Keep the Mac plugged in with the lid open.
- **Victor is editing the paper in Overleaf while you work.** Overleaf is the source of truth.
  In the local `CS371_Cho_Li/acl_latex.tex`, change only numbers, the `\textbf{XX}`
  placeholders, and figure files. Do not rewrite prose. Every change also goes in the report as a
  Before/After block he can paste. When a sentence's claim stops holding, flag it in the report
  and leave the sentence as it is.
- Standing paper rules: `docs/TODO-CLAUDE-CODE.md` (no section labels or `Section~\ref`, no new
  bib entries, `algorithmic` package). Write "Zone A" and "Zone AE" without "~".

## 2. The grids

Families in this order (the sorted code names, as now): dt, et, hgb, lr, mlp, nb, rf, xgb.
Within a family, settings follow `itertools.product` over the axes in the order listed (first
axis slowest). Each setting's seed is drawn from `np.random.default_rng(cell seed)` in that
order, as `zoo_specs` draws seeds now.

| Family | Axes (keyword: values) | Count |
|---|---|---|
| dt | `max_depth`: 1, 2, 3, 4, 5, 6, 8, 10, 12, None × `min_samples_leaf`: 1, 20, 200, 2000 | 40 |
| rf | `n_estimators`: 10, 30, 100 × `max_depth`: 1, 2, 3, 5, 8, None × `max_features`: 1, "sqrt", None | 54 |
| et | same axes as rf | 54 |
| hgb | `learning_rate`: 0.01, 0.03, 0.1, 0.5 × `max_iter`: 3, 10, 50, 300 × `max_depth`: 1, 3, None | 48 |
| xgb | `learning_rate`: 0.01, 0.03, 0.1, 0.5 × `n_estimators`: 3, 10, 50, 300 × `max_depth`: 1, 3, 8 | 48 |
| lr | `C`: 0.001, 0.01, 0.1, 1, 100 × `max_iter`: 1, 2, 3, 5, 100 × `class_weight`: None, "balanced" | 50 |
| mlp | `hidden_layer_sizes`: (2,), (8,), (16,), (32,), (64, 32) × `max_iter`: 3, 20, 200 × `learning_rate_init`: 1e-4, 1e-3, 1e-2 | 45 |
| nb | `var_smoothing`: 1e-9, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 10 × `priors`: None, (0.5, 0.5) | 30 |

Fixed settings, unchanged from `zoo.build`: hgb `early_stopping=False`; xgb
`tree_method="hist"`, `n_jobs=1`, CPU; rf and et `n_jobs=1` and their default `bootstrap`;
median imputation for every family except hgb and xgb; `StandardScaler` before lr and mlp;
`random_state` = the setting's seed (not for nb). **One change:** lr no longer fixes
`max_iter=400`, because `max_iter` is now a grid axis. Everything else is the library default.
The paper's Table `tab:models` lists exactly these grids. If any keyword or value must change,
stop and ask.

## Part A. Change the code

1. `aotil/zoo.py`: replace `learner_settings` and `zoo_specs` with the grids of §2 and one spec
   per setting. Delete `feat_frac`, `row_frac`, the round-robin and the feature and row draws in
   `fit_one`. Every model now uses all columns of the cell. The `varying_columns` guard stays.
   `--n-models` no longer sets the population size; remove it, or make it an error if given.
2. Apply the degenerate drop rule of §0 after fitting. Do not drop duplicates. The cell's
   `S_ood`, `S_id` and models table hold the kept models only, in grid order. Add a `grid_idx`
   column (the setting's position in §2 order), and write the dropped settings to
   `<tag>_dropped.csv` with their flood share on the ID test points.
3. Find every reader that assumes 300 models, eight equal families, round-robin order, or the
   `feat_frac` and `row_frac` columns, and update it. At least check: `aotil/audit_cell.py` (the
   refit and the 8-family check), the `MODEL_STRIDE` comment in `aotil/o1_shift.py`,
   `strongest_models`, the family-disjoint split in `aotil/oodselect_paper.py`, `report_paper.py`,
   `audit_selection.py`, `run_oodselect_local.sh`, and any `--n-models 300` in the chain scripts.
4. Tests: every family's grid size and order; all specs unique; the degenerate rule on synthetic
   predictions (including ties at the 5% and 95% cuts); identical predictions are kept; the model
   split over a population that is not 300. The zoo's characterization goldens will change. List each changed golden in the report
   with the reason. Every other golden may change by additions only.

## Part B. Fit the cell and check it (stop point)

Fit `FL_band_AE_s0` into `data/interim/aotil/cells_grid_$D` and run `aotil/audit_cell.py`
(verdict PASS). Report:
- a per-family table like the preview (settings, degenerate, kept, ID accuracy range, and how
  many kept models repeat an earlier model's predictions);
- the kept N, the 60/20/20 split sizes, the overall ID accuracy range, sd and quantiles;
- the number of distinct prediction patterns among the test models (the Fisher interval on
  corr_te counts every test model as a separate observation);
- probit r(ID, OOD) over all kept models and over the test models;
- OODSelect's model-count rule (`n_models_needed`).

**Stop** and report before Part C if: kept N < 200; the lowest kept ID accuracy is above 0.60;
any family keeps fewer than 10 models; or probit r(ID, OOD) over all kept models is below 0.5.
Otherwise continue without waiting.

## Part C. Full run of the search, and everything after it, on the new models

This is a full run, not a spot check: every budget in the grid for the classic (OODSelect),
Laplacian and GTV searches, with the same settings as 2026-10-25 (squared penalty, T = 100 TPE
trials, at most 50,000 steps, patience 500, the same seeds). Repeat Part D of `docs/TASK-RERUN-ALL-2026-10-25.md` on the new cell with the same scripts, seeds
and settings (`logs/2026-10-25/C_chain.sh` and what follows it, `docs/RUN-RERUN-ALL-2026-10-25.md`):
the shift (pooled and per tile) and its tile-shuffle null; the classic, Laplacian and GTV
searches over the budget grid; the selection audit; the mu sweep (mu = 1, 3, 10 at S = 250, 500
and 1,000); the three controls (shuffled accuracies, interpolating models, family splits); the
regions (c = 20); `region_match.py`; and the figures. Check claims C1 to C7 as fixed in
`docs/TASK-RERUN-ALL-2026-10-25.md` §2 (C1 on the unweighted per-tile columns). Report each as
holds or does not hold, whatever the result.

Also recompute the paper's rule statements:
- **mu rule.** The paper says mu = 3 is the smallest of {1, 3, 10} at which the S = 500 selection
  has purity of at least 0.5 (the share of a selected point's own 10 nearest neighbours that are
  also selected, averaged over the selected points). If the rule no longer picks 3, **stop** and
  report. Do not change mu.
- **Largest inverted budget.** The paper says the inversion holds up to S = 7,500: the largest
  grid budget with corr_te at or below -0.3, for both the classic and the Laplacian search.
  Recompute it.
- **N1 (report only).** Because the weak end leans on naive Bayes: probit r(ID, OOD) without the nb
  models, and corr_te of the Laplacian S = 500 subset on the test models without nb (score the
  existing subset; no new search).

**Stop** if C6a or C6b fails, before any paper edit (as in the 2026-10-25 brief).

## Part D. Update the paper (numbers only)

- Fill the 17 `\textbf{XX}` placeholders: the Abstract's model count, the Models paragraph in
  Implementation Details (N and the split sizes), the Table `tab:search` caption (test models),
  and Appendix A (the dropped count, N, and the Models column of Table `tab:models`).
- Update every number that depends on the models: the Abstract, all of Results (shift, search,
  controls, matching), Table `tab:search`, Table `tab:regions`, the regions figure, the S = 7,500
  sentence and the Correlation and Search paragraphs if their numbers move.
- **If the regions change** (different places, or not 6 regions), do not rewrite the case
  studies (Levy, Liberty) or region names. Report the new regions with the same descriptors the
  case studies use (size, share of positives, matches' share of positives, land cover, height
  above the nearest drainage, distance to river) and flag the paragraph for Victor.
- Compile a copy: no overfull boxes, no undefined references.

## Part E. Report and docs

- `docs/RUN-ZOO-GRIDS-$D.md`: Part B's tables; claims C1 to C7; the mu rule; the largest inverted
  budget; N1; every paper change as a Before/After block; flagged sentences; and timings (fit
  time, search wall clock).
- `docs/DECISION-PROVENANCE.md`: replace the zoo rows for the grids, the feature share, the row
  share and the round-robin with the new design (support: "Your decision", source: this brief),
  and add a row for the degenerate drop rule and for keeping duplicates.
- `docs/TODO-CLAUDE-CODE.md`: move item 5 to Done with the date and the report.

## Stops (summary)

1. Part B: kept N < 200, lowest ID accuracy above 0.60, a family below 10 models, or probit
   r(ID, OOD) below 0.5.
2. A grid keyword or value has to change (§2).
3. The mu rule no longer picks mu = 3.
4. C6a or C6b fails (before any paper edit).
