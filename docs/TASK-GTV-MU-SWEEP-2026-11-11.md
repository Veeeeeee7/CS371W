# Task: mu sweep at S = 500 for both regularized searches (sensitivity analysis, one variable at a time) — 2026-11-11 (v3)

v1 (GTV only, {0.3, 1, 10, 30}) and v2 (decade grid, mu picked by a rule) are in Claude outputs/. v3 (Victor,
2026-11-11 23:11): mu stays fixed at 3 for both losses; the sweep is a sensitivity analysis on a half-decade grid,
so 3 is a grid point.

## Why
mu = 3 was never chosen for GTV. It started as the operating point of an early pipeline version and was later
justified on the Laplacian by "smallest mu whose purity reaches 0.5 at S = 500" (current cell: Laplacian purity at
S = 500 is 0.31 / 0.50 / 0.63 at mu = 1 / 3 / 10). GTV copied it. Validation tuning (or the one-standard-error rule)
cannot choose mu here: it trades against the validation correlation, so it would pick mu near 0 (OODSelect r_val
-0.987; one Fisher-z SE with 58 validation models already excludes Laplacian mu = 1, r_val -0.970).

## Grid
- Log-spaced over four decades, as glmnet builds its lambda path (Friedman, Hastie and Tibshirani 2010; default
  lambda_min / lambda_max = 1e-4), at half-decade steps (the rounded 1, 3, 10 sequence) so that the paper's mu = 3 is a
  grid point: mu in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}, plus mu = 0 (OODSelect).
- The same grid for both losses. Their weights are not on the same scale, so the paper compares them at matched purity
  (Wang et al. 2016 compare graph trend filtering and Laplacian smoothing at matched degrees of freedom).

## What to vary and what to hold fixed
Vary mu only. Hold S = 500, k = 10, c = 20, seed 0, the 60/20/20 model split, T = 100, P = 500, squared budget penalty.
Existing in results/2026-11-07/oodselect_paper: classic (mu 0), spatial_mu{1,3,10}_knn10, spatial_gtv_mu3_knn10.
New searches: GTV at 0.01, 0.03, 0.1, 0.3, 1, 10, 30, 100 (8); Laplacian at 0.01, 0.03, 0.1, 0.3, 30, 100 (6).
14 searches, about 2.5 hours (overnight is fine).

## Command (on the Mac, repo root, .venv with torch + optuna)
One run_oodselect_local.sh at a time (they share logs/<date>/oodselect_paper/local_tasks.txt); the zoo step skips
the existing cell; each S = 500 search is one single-thread process of about 10 minutes.

    D=2026-11-07
    sweep() {  # sweep <reg> <mu...>
      local reg=$1; shift
      for MU in "$@"; do
        env VARIANTS=spatial REG=$reg MU_SPATIAL=$MU BUDGETS=500 \
            DATA_DIR=data/processed_v4 CELLS_DIR=data/interim/aotil/cells_v4_$D CELLS="FL:band:AE:0" \
            SEED=0 JOBS=8 BUDGET_PENALTY=squared ./run_oodselect_local.sh --date $D \
          > logs/2026-11-11/${reg}_mu${MU}_S500.log 2>&1
        echo "$reg mu=$MU exit $?"
      done
    }
    sweep gtv 0.01 0.03 0.1 0.3 1 10 30 100
    sweep laplacian 0.01 0.03 0.1 0.3 30 100

Outputs: results/2026-11-07/oodselect_paper/spatial_gtv_mu{0.01,0.03,0.1,0.3,1,10,30,100}_knn10/ and
spatial_mu{0.01,0.03,0.1,0.3,30,100}_knn10/, each with FL_band_AE_s0_S500_{theta.npy,result.csv,...}.

## After the run (no torch needed; Claude can do this)
Decided 2026-11-11 23:11 (Victor): mu = 3 for both losses in every main result; the Sensitivity Analysis has two
paragraphs, mu first (this sweep), then the budget.
1. For every (loss, mu): corr_te [95% CI] on the 58 test models, purity, # Comps (points); smoothing optional.
2. Figure fig_mu.pdf (same style as fig_budgets.py): x = mu (log), top corr_te, bottom purity, one line per loss,
   OODSelect (mu = 0) as a reference line, dotted line at mu = 3. Replace the \fbox placeholder in the paper.
3. Fill the XX of the "Regularization weight." paragraph and check its CHECK comment (direction of the trends; which
   loss inverts more at equal purity).
