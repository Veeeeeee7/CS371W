# Full run with the Woznicki feature set: report, D = 2026-10-16

Brief: `docs/TASK-FULL-RUN-W7.md`. Plan: `docs/PROJECT-PLAN.md` v5. Outputs: `results/2026-10-16/`,
logs: `logs/2026-10-16/`. The first-look folder `results/2026-10-16/w7_first_look/` was already there
and is untouched. The models use the seven features in `data/processed_w7/FEATURES.txt`. The
45-feature comparison numbers (seed 0) come from two places. The selections (O3, O4) are from
`results/2026-10-14/`, which is untouched. O1, O2 and O5 were recomputed on the 45-feature cell
and tables with this run's code, into `results/2026-10-16/o1_shift_45feat_s0.csv` and
`results/2026-10-16/boundary_stat_45feat/`.

**Status: done.** Parts A–D and C8 (the map) all ran. There were two stops, each decided by Victor:
- Stop 1 (C1): a seed-2 cell audit fell one float rounding step outside [0, 1].
- Stop 2 (C7): the level-set calibration.

Nothing is committed; every code change is left for review.

## Verdicts

| outcome | verdict | the number that decides it |
|---|---|---|
| **O1** the shift | **met, narrowly** | pooled all-points AUC gap **0.046 [0.004, 0.094]** (tile bootstrap over all 300 models × 3 seeds). Seeds 1 and 2 meet it alone; seed 0 does not (0.004 [−0.050, 0.060]) |
| **O2** aggregate accuracy-on-the-line | **strongly met** | `r_full_test` **+0.729 / +0.863 / +0.873** (seeds 0/1/2), every one > 0.7 |
| **O3** classic OODSelect: inverted but scattered | **partly met** | classic S = 500 r **−0.888 / −0.948 / −0.931**, scattered over 21–24 of the cell's 23–25 test tiles, and more negative than all 27 null-select runs. But null-select itself reaches **−0.864**, with a median of +0.10, and the others→trees split loses the sign in seed 0 (**+0.094** at S = 500) |
| **O4** the spatial penalty makes regions at a small cost | **met** | spatial S = 500 r **−0.815 / −0.842 / −0.801** at a cost Δr of +0.07 / +0.11 / +0.13. kNN purity rises from 0.08–0.10 to 0.75–0.81, and the subset shrinks from 301–309 to 20–29 components. Spatial beats smoothed-classic on purity in 9 of 9 settings and on r in 8 of 9 |
| **O5** regions differ from nearby Zone A beyond terrain | **not met** | primary U(region) − U(matched Zone A) on testable windows, d = 100 m: **+0.025 [−0.007, +0.058]** (S = 250) and **−0.059 [−0.226, +0.044]** (S = 500). Only the null part holds: pooled U(region) is above every displaced-edge draw (percentile 100, p = 0.005) |

---

# Results by outcome

## O1. The shift: a model trained on Zone AE scores lower on Zone A than on held-out Zone AE

Files: `results/2026-10-16/o1_shift.csv` (7 features) and `results/2026-10-16/o1_shift_45feat_s0.csv`
(45 features, seed 0, the same code). Logs: `logs/2026-10-16/C2_o1_shift*.log`. Script: `aotil/o1_shift.py` (B2).

What the columns mean:
- **ID AUC / Zone A AUC:** the mean over 300 models of the AUC on the held-out Zone AE half and on
  the Zone A half of the same 50 km test tiles.
- **Gap:** ID − Zone A. Positive means worse on Zone A.
- **95% interval:** a bootstrap over test tiles. Whole 50 km squares are resampled, so points that
  share a tile stay together. 200 draws, all 300 models.
- **Worse:** the share of models with a positive gap.
- **Strongest quarter:** the 75 models with the highest ID AUC.
- **Slope:** the least-squares slope of Zone A AUC on ID AUC across models.

The rows:
- "land" leaves out open water (NLCD land-cover code 11).
- "paired land" keeps land points in 10 km blocks that hold positives of both zones, so both
  halves are compared on the same ground.
- "pooled" is the mean over the three seeds. Its interval resamples the union of the seeds' test
  tiles once per draw for all seeds together, because a tile can be a test tile in more than one
  seed.

| cell | rows | ID AUC | Zone A AUC | gap [95%] | worse | strongest quarter: gap (worse) | slope of Zone A AUC on ID AUC |
|---|---|---:|---:|---|---:|---|---:|
| seed 0 | all | 0.829 | 0.826 | 0.004 [−0.050, 0.060] | 56% | 0.013 (72%) | 0.72 |
| seed 0 | land | 0.800 | 0.821 | −0.021 [−0.099, 0.056] | 33% | −0.010 (27%) | 0.78 |
| seed 0 | paired land | 0.848 | 0.823 | 0.026 [−0.021, 0.058] | 74% | 0.039 (100%) | 0.72 |
| seed 1 | all | 0.828 | 0.755 | 0.072 [0.014, 0.120] | 94% | 0.088 (100%) | 0.66 |
| seed 1 | land | 0.799 | 0.757 | 0.042 [−0.037, 0.101] | 90% | 0.048 (100%) | 0.81 |
| seed 1 | paired land | 0.824 | 0.765 | 0.058 [0.014, 0.093] | 98% | 0.064 (100%) | 0.83 |
| seed 2 | all | 0.852 | 0.789 | 0.063 [0.007, 0.113] | 95% | 0.079 (100%) | 0.64 |
| seed 2 | land | 0.839 | 0.784 | 0.055 [−0.011, 0.114] | 94% | 0.071 (100%) | 0.66 |
| seed 2 | paired land | 0.854 | 0.784 | 0.070 [0.025, 0.113] | 97% | 0.082 (100%) | 0.77 |
| **pooled** | **all** | 0.836 | 0.790 | **0.046 [0.004, 0.094]** | 82% | 0.060 (91%) | 0.67 |
| pooled | land | 0.812 | 0.787 | 0.025 [−0.042, 0.070] | 72% | 0.036 (76%) | 0.75 |
| pooled | paired land | 0.842 | 0.791 | 0.051 [0.017, 0.078] | 90% | 0.062 (100%) | 0.77 |
| 45 features, seed 0 | all | 0.858 | 0.839 | 0.020 [−0.040, 0.078] | 71% | 0.016 (89%) | 0.95 |
| 45 features, seed 0 | land | 0.831 | 0.831 | 0.000 [−0.075, 0.068] | 44% | −0.004 (39%) | 0.97 |
| 45 features, seed 0 | paired land | 0.879 | 0.851 | 0.028 [−0.015, 0.059] | 85% | 0.035 (100%) | 0.94 |

The point values equal the first look's to within 0.001. Most intervals (11 of 13) are wider than
the first look's, because they now cover all 300 models instead of a 30-model, 4-family subset.
The pooled all-points lower bound would have been 0.015 on the first look's 30 models; it is 0.004
here. Seed 0 paired land is slightly narrower.

**Verdict: met, narrowly.** The pooled all-points gap is positive, and its interval excludes 0,
with a lower bound of only 0.004.
- Rows that meet it: pooled all points, and pooled paired land (0.051 [0.017, 0.078]).
- Pooled land points do not (0.025 [−0.042, 0.070]).
- Seeds 1 and 2 meet it on all points and on paired land. Seed 0 does not meet it on any row. As
  the first look said, seed 0 is a low-gap split for every feature set: the 45-feature seed-0 gap
  is 0.020 [−0.040, 0.078].

## O2. Aggregate accuracy-on-the-line

The paper's aggregate correlation: probit(ID accuracy) against probit(accuracy on all OOD points),
across the 60 test models of the 60/20/20 model split, with the authors' eps-regularised Pearson
(`r_full_test`, the number every selection stores). The 95% interval is Fisher's, over the 60
models. "Models needed" is the paper's convergence rule as `report_paper.model_convergence` codes
it: add models 10 at a time, and take the first size from which every later step moves r by less
than 1%.
Files: `results/2026-10-16/o1_shift.csv` and `results/2026-10-16/oodselect_paper/FL_band_AE_s*_model_convergence.csv`.

| cell | `r_full_test` [95%] | r, all 300 models | models needed |
|---|---|---:|---:|
| seed 0 | +0.729 [+0.583, +0.830] | +0.710 | 60 |
| seed 1 | +0.863 [+0.779, +0.916] | +0.821 | 20 |
| seed 2 | +0.873 [+0.795, +0.922] | +0.820 | 20 |
| 45 features, seed 0 | +0.880 [+0.807, +0.927] | +0.829 | 30 |

**Verdict: strongly met.** r > 0.7 in every seed. Seed 0 is the weakest, with an interval down
to 0.58. The 7-feature population tracks the line a little less tightly than the 45-feature one
(+0.729 vs +0.880 on seed 0).

## O3. Classic OODSelect finds inverted but scattered subsets

Test r on the 60 held-out test models [Fisher 95%] and coherence, per seed and budget S.
- **Random subsets:** 100 random subsets of the same size; their r is the line O2 measures.
- **Hardest points:** the S points the test models get wrong most often.
- **kNN purity:** the share of a selected point's 10 nearest OOD neighbours that are also
  selected; 1 means one solid blob.
- **Components:** connected pieces of the selection on the 10-nearest-neighbour graph.
- **Tiles:** how many of the cell's 50 km test tiles the selection touches.

Files: `results/2026-10-16/oodselect_paper/<variant>/FL_band_AE_s<seed>_summary.csv`. Audits:
`logs/2026-10-16/C3_audit_selection_s{0,1,2}.log`, 0 fail.

| seed | variant | S | test r [95%] | random subsets r (sd) | hardest points r | purity | components | largest | tiles |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| 0 | classic | 250 | −0.935 [−0.961, −0.894] | +0.677 (0.116) | −0.175 | 0.10 | 164 | 0.05 | 24 of 24 |
| 0 | classic | 500 | −0.888 [−0.932, −0.818] | +0.706 (0.073) | −0.126 | 0.08 | 309 | 0.03 | 24 of 24 |
| 0 | classic | 1,000 | −0.905 [−0.942, −0.845] | +0.721 (0.047) | −0.300 | 0.21 | 348 | 0.03 | 24 of 24 |
| 0 | spatial | 250 | −0.825 [−0.892, −0.722] | | | 0.72 | 11 | 0.29 | 3 of 24 |
| 0 | spatial | 500 | −0.815 [−0.886, −0.708] | | | 0.75 | 20 | 0.36 | 5 of 24 |
| 0 | spatial | 1,000 | −0.883 [−0.928, −0.810] | | | 0.80 | 38 | 0.34 | 11 of 24 |
| 1 | classic | 250 | −0.937 [−0.962, −0.896] | +0.808 (0.057) | +0.046 | 0.06 | 187 | 0.03 | 23 of 25 |
| 1 | classic | 500 | −0.948 [−0.969, −0.914] | +0.835 (0.036) | −0.135 | 0.10 | 304 | 0.04 | 23 of 25 |
| 1 | classic | 1,000 | −0.953 [−0.972, −0.923] | +0.846 (0.033) | −0.221 | 0.17 | 468 | 0.04 | 25 of 25 |
| 1 | spatial | 250 | −0.787 [−0.867, −0.666] | | | 0.90 | 8 | 0.42 | 8 of 25 |
| 1 | spatial | 500 | −0.842 [−0.903, −0.748] | | | 0.81 | 23 | 0.24 | 15 of 25 |
| 1 | spatial | 1,000 | −0.827 [−0.893, −0.725] | | | 0.88 | 29 | 0.27 | 16 of 25 |
| 2 | classic | 250 | −0.886 [−0.931, −0.816] | +0.830 (0.057) | −0.287 | 0.05 | 191 | 0.04 | 21 of 23 |
| 2 | classic | 500 | −0.931 [−0.959, −0.887] | +0.844 (0.041) | −0.310 | 0.09 | 301 | 0.03 | 21 of 23 |
| 2 | classic | 1,000 | −0.944 [−0.966, −0.908] | +0.864 (0.023) | −0.245 | 0.17 | 442 | 0.06 | 22 of 23 |
| 2 | spatial | 250 | −0.845 [−0.905, −0.753] | | | 0.77 | 23 | 0.42 | 11 of 23 |
| 2 | spatial | 500 | −0.801 [−0.876, −0.686] | | | 0.81 | 29 | 0.29 | 11 of 23 |
| 2 | spatial | 1,000 | −0.897 [−0.937, −0.833] | | | 0.87 | 34 | 0.24 | 14 of 23 |

Classic and spatial share their random-subset and hardest-point baselines, since the budgets and
the test models are the same. (`report_paper.py` draws the random subsets from one stream across
the budget grid. The control folders, which cover only 3 budgets, therefore get slightly
different random baselines, e.g. +0.659 vs +0.677 at seed 0, S = 250. The hardest-point
baseline does not change.) "Hardest points" means the S points the test models get wrong most
often. 45-feature seed 0 for comparison (`results/2026-10-14/oodselect_paper/`): classic
S = 250 / 500 / 1,000 r −0.841 / −0.889 / −0.886 (purity 0.16 / 0.21 / 0.31, 143 / 221 / 291
components, 23–24 tiles); random subsets +0.84 to +0.87; hardest points −0.37 to −0.42.

**Null-select.** The same search, with the ID accuracies the search sees shuffled (null seeds
1, 2, 3). It is scored on the test models against their true accuracies. Files:
`results/2026-10-16/oodselect_paper/{classic,spatial_mu3_knn10}_null{1,2,3}/`, audits
`logs/2026-10-16/C4_audit_null_s{0,1,2}.log`, 0 fail.

| seed | S | classic: real | null 1 | null 2 | null 3 | spatial: real | null 1 | null 2 | null 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 250 | **−0.935** | +0.788 | −0.288 | +0.096 | **−0.825** | +0.721 | +0.840 | +0.199 |
| 0 | 500 | **−0.888** | +0.763 | −0.568 | +0.195 | **−0.815** | +0.588 | +0.813 | +0.571 |
| 0 | 1,000 | **−0.905** | +0.766 | −0.559 | −0.233 | **−0.883** | −0.466 | +0.446 | +0.617 |
| 1 | 250 | **−0.937** | −0.327 | −0.119 | −0.582 | **−0.787** | −0.415 | +0.776 | +0.171 |
| 1 | 500 | **−0.948** | −0.540 | +0.114 | −0.582 | **−0.842** | +0.022 | +0.842 | +0.182 |
| 1 | 1,000 | **−0.953** | −0.864 | −0.030 | −0.747 | **−0.827** | −0.250 | +0.012 | −0.677 |
| 2 | 250 | **−0.886** | +0.517 | +0.186 | +0.608 | **−0.845** | +0.270 | +0.288 | −0.180 |
| 2 | 500 | **−0.931** | +0.666 | +0.451 | +0.701 | **−0.801** | +0.368 | +0.250 | −0.723 |
| 2 | 1,000 | **−0.944** | +0.690 | −0.714 | +0.585 | **−0.897** | +0.320 | −0.097 | +0.364 |

- classic null runs (27): median r +0.096, range −0.864 to +0.788; 9 of 27 below −0.3; 13 of 27 below 0. Validation r against the shuffled accuracies: median −0.164 (the real runs: −0.932).
- spatial null runs (27): median r +0.270, range −0.723 to +0.842; 4 of 27 below −0.3; 7 of 27 below 0. Validation r against the shuffled accuracies: median −0.214 (the real runs: −0.835).
- In all 54 null runs the real run at the same seed, S and variant is more negative.

- Classic null runs (27): median r +0.096, range −0.864 to +0.788; 9 of 27 below −0.3; 13 of 27
  below 0. Spatial null runs (27): median +0.270, range −0.723 to +0.842; 4 of 27 below −0.3.
- **In all 54 null runs, the real run at the same seed, S and variant is more negative.**
- Against the shuffled accuracies they were tuned on, the null searches reach only a validation r
  of about −0.16 (classic median; spatial −0.21), against −0.93 / −0.83 for the real runs. So
  their test r is not a fitted inversion. It says where a weakly guided selection ends up, and it
  shows that strongly anti-correlated subsets are easy to land in. In seed 1, all three classic
  nulls are negative at S = 1,000 (−0.86, −0.03, −0.75).

**Family-disjoint split.** Select on one learner family, report r on the other. Trees = hgb, rf,
et, dt, xgb (188 models); others = lr, mlp, nb (112). Files: `..._trees2others/`,
`..._others2trees/`; audits `logs/2026-10-16/C4_audit_splits_s{0,1,2}.log`, 0 fail.

| seed | S | classic: random split | trees→others (112 test) | others→trees (188 test) | spatial: random split | trees→others | others→trees |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 250 | −0.935 | −0.889 | −0.028 | −0.825 | −0.878 | −0.630 |
| 0 | 500 | −0.888 | −0.927 | +0.094 | −0.815 | −0.849 | −0.231 |
| 0 | 1,000 | −0.905 | −0.926 | −0.037 | −0.883 | −0.872 | −0.477 |
| 1 | 250 | −0.937 | −0.771 | −0.495 | −0.787 | +0.689 | −0.523 |
| 1 | 500 | −0.948 | −0.779 | −0.600 | −0.842 | −0.280 | −0.286 |
| 1 | 1,000 | −0.953 | −0.750 | −0.680 | −0.827 | −0.581 | −0.493 |
| 2 | 250 | −0.886 | −0.737 | −0.518 | −0.845 | +0.299 | −0.460 |
| 2 | 500 | −0.931 | −0.745 | −0.591 | −0.801 | −0.303 | −0.429 |
| 2 | 1,000 | −0.944 | −0.795 | −0.652 | −0.897 | −0.584 | −0.357 |

The full-set r of each test family (the aggregate line on those models): trees→others +0.681 /
+0.813 / +0.786; others→trees +0.804 / +0.836 / +0.824 (seeds 0 / 1 / 2).
- Classic trees→others keeps the sign everywhere (−0.74 to −0.93).
- Classic others→trees keeps it in seeds 1–2 (−0.49 to −0.68) but **not in seed 0** (−0.03,
  +0.09, −0.04).
- Spatial others→trees keeps the sign everywhere (−0.23 to −0.63). Spatial trees→others keeps it
  at S = 500 and 1,000 but **loses it at S = 250 in seeds 1 and 2** (+0.689, +0.299).

**Verdict: partly met.**
- r is clearly negative: −0.886 to −0.953 at every seed and budget, against +0.68 to +0.86 for
  random subsets and +0.05 to −0.31 for the hardest points.
- The subsets are scattered: purity 0.05–0.21, 164–468 components, 21–25 of the cell's 23–25 tiles.
- Two of the brief's conditions fail in part:
  - Null-select is not near 0. It is weaker than the real search in 54 of 54 runs, but a third
    of the classic nulls fall below −0.3.
  - The family split does not always keep the sign. For classic, others→trees loses it in seed 0;
    for spatial, trees→others loses it at S = 250 in seeds 1–2.
- The real inversion is stronger than every null-select run, but the nulls show that much of its
  size could come from the search alone.

## O4. The spatial penalty makes regions at a small cost

Spatial (μ = 3, k = 10) against classic at each budget; the full spatial rows are in O3's table.
"Smoothed-classic" is the classic answer smoothed over each point's 10 neighbours after the fact
and cut back to S points, with no new search; it is the ablation (`--smoothed-classic` in the
audits).

| seed | S | Δr (spatial − classic) | purity classic → spatial | components | tiles | smoothed-classic: r, purity, components, tiles |
|---|---:|---:|---|---|---|---|
| 0 | 250 | +0.111 | 0.10 → 0.72 | 164 → 11 | 24 → 3 | −0.500, 0.61, 29, 13 |
| 0 | 500 | +0.072 | 0.08 → 0.75 | 309 → 20 | 24 → 5 | −0.863, 0.53, 63, 13 |
| 0 | 1,000 | +0.022 | 0.21 → 0.80 | 348 → 38 | 24 → 11 | −0.852, 0.67, 63, 14 |
| 1 | 250 | +0.150 | 0.06 → 0.90 | 187 → 8 | 23 → 8 | −0.632, 0.59, 34, 16 |
| 1 | 500 | +0.106 | 0.10 → 0.81 | 304 → 23 | 23 → 15 | −0.663, 0.59, 61, 17 |
| 1 | 1,000 | +0.126 | 0.17 → 0.88 | 468 → 29 | 25 → 16 | −0.648, 0.64, 88, 22 |
| 2 | 250 | +0.041 | 0.05 → 0.77 | 191 → 23 | 21 → 11 | −0.283, 0.58, 28, 9 |
| 2 | 500 | +0.131 | 0.09 → 0.81 | 301 → 29 | 21 → 11 | −0.263, 0.65, 48, 16 |
| 2 | 1,000 | +0.047 | 0.17 → 0.87 | 442 → 34 | 22 → 14 | −0.221, 0.67, 88, 22 |
| 45 feat., 0 | 500 | +0.126 | 0.21 → 0.81 | 221 → 20 | 23 → 12 | −0.553, 0.69, 43, 18 |

Files: `results/2026-10-16/oodselect_paper/FL_band_AE_s*_selection_audit.md`; the 45-feature row
is from `results/2026-10-14/oodselect_paper/FL_band_AE_s0_selection_audit.md`.

μ sweep, seed 0: r; purity, components, tiles. μ = 0 and 3 come from C3, μ = 1 and 10 from C4.
Audit: `logs/2026-10-16/C4_audit_mu_sweep_s0.log`, 0 fail.

| S | μ = 0 (classic) | μ = 1 | μ = 3 | μ = 10 |
|---:|---|---|---|---|
| 250 | −0.935; purity 0.10, 164 comp., 24 tiles | −0.848; purity 0.47, 29 comp., 7 tiles | −0.825; purity 0.72, 11 comp., 3 tiles | −0.808; purity 0.86, 4 comp., 2 tiles |
| 500 | −0.888; purity 0.08, 309 comp., 24 tiles | −0.858; purity 0.49, 46 comp., 10 tiles | −0.815; purity 0.75, 20 comp., 5 tiles | −0.847; purity 0.91, 4 comp., 3 tiles |
| 1,000 | −0.905; purity 0.21, 348 comp., 24 tiles | −0.901; purity 0.82, 36 comp., 15 tiles | −0.883; purity 0.80, 38 comp., 11 tiles | −0.845; purity 0.92, 8 comp., 4 tiles |

**Verdict: met.**
- Spatial r stays clearly negative (−0.787 to −0.897, every interval below −0.66) at a cost Δr of
  +0.02 to +0.15.
- Coherence rises sharply:
  - purity from 0.05–0.21 to 0.72–0.90;
  - components from 164–468 to 8–38;
  - tiles from 21–25 to 3–16.
- Coherence rises with μ at S = 250 and 500. At S = 1,000, μ = 1 → 3 dips slightly (purity 0.82 → 0.80, 36 → 38 components) before μ = 10 (0.92, 8). For μ ≥ 1, r stays between −0.81 and −0.90.
- Spatial beats smoothed-classic on purity in every setting (0.72–0.90 vs 0.53–0.67). It also
  beats it on r in 8 of 9; the exception is seed 0, S = 500 (−0.815 vs −0.863).

Two caveats:
- The S = 500 spatial result moves by about 0.07 with the search path (see the run log, C3), so
  the cost of coherence at a single setting is uncertain to that degree.
- Spatial null-select also produces coherent subsets, some of them inverted: seed 1 null 3 at
  S = 1,000 gives r −0.677 at purity 0.79, and seed 2 null 3 at S = 500 gives −0.723 at purity
  0.86. So coherence alone is no evidence of a real effect.

## O5. The regions differ from nearby Zone A beyond terrain

The statistic (B3; `aotil/boundary_stat.py`):
- Each flood-zone boundary is cut into 300 m windows.
- A window is **contour-like** if two things hold:
  - its ground elevation varies by at most 0.76 m along the edge;
  - the ground rises out of the floodplain, i.e. the outside probe is higher than the inside one.
- **U** is the share of edge length that is *not* contour-like **and** lies (median distance) within
  d of a soil-survey line or a map line (county line, FIRM panel edge). Such edge looks traced
  from a map rather than from terrain.
- **Primary statistic:** U(region) − U(matched Zone A). Here `region` is Zone A edge inside a
  selected region, and `matched Zone A` is unselected Zone A edge in the same tiles, matched on
  perimeter.

As Victor decided at Stop 2, the primary uses only **testable windows**: those with ≥ 5 vertices
whose 100 m probes landed on the correct side. The full-edge version is a sensitivity
(`results/2026-10-16/boundary_stat_allwindows/`).

Calibration on testable windows (AE should mostly pass):
- seed-0 regions: 88% of AE edge passes at S = 250 and 77% at S = 500;
- recurring regions: 81% in both seed 1 and seed 2.

The null shifts each region lobe rigidly by 300–1,000 m in a random direction, 200 times, and
recomputes the line distances with the stored-distance code. The zero-shift self-check passed in
all 8 runs.

Files: `results/2026-10-16/boundary_stat/<prefix>_{summary.csv,null.csv,windows.parquet,report.md}`.
Logs: `logs/2026-10-16/C7_boundary_stat_<prefix>_windows-{testable,all}.log`.

**Main table.** Testable windows, edge-length weights. The bootstrap is over components, 1,000 draws. The null percentile is the rank of observed U(region) among 200 displaced draws, with one-sided p in brackets.

| | seed 0, S = 250 (4 components) | seed 0, S = 500 (6 components) | seed 1 recurring, S = 500 (2: Collier/Hendry, Sarasota) | seed 2 recurring, S = 500 (2: the same) |
|---|---|---|---|---|
| testable share of edge, region / matched A / AE | 0.73 / 0.75 / 0.50 | 0.71 / 0.77 / 0.54 | 0.71 / 0.76 / 0.32 | 0.68 / 0.79 / 0.33 |
| contour-like share, region / matched A / AE | 0.658 / 0.676 / 0.879 | 0.682 / 0.620 / 0.769 | 0.834 / 0.766 / 0.813 | 0.848 / 0.671 / 0.814 |
| **d = 100 m** U region / matched A / AE | 0.316 / 0.291 / 0.108 | 0.288 / 0.346 / 0.208 | 0.156 / 0.218 / 0.147 | 0.146 / 0.306 / 0.147 |
| **primary U(region) − U(matched A)** | **+0.025 [−0.007, +0.058]** | **−0.059 [−0.226, +0.044]** | **−0.062 [−0.192, +0.070]** | **−0.160 [−0.221, +0.125]** |
| secondary U(region) − U(AE) | +0.208 [+0.013, +0.265] | +0.079 [−0.076, +0.207] | +0.010 [−0.069, +0.065] | −0.001 [−0.069, +0.303] |
| U(region) observed vs null mean [2.5–97.5%] | 0.316 vs 0.298 [0.291, 0.303] | 0.288 vs 0.273 [0.270, 0.277] | 0.156 vs 0.136 [0.128, 0.146] | 0.146 vs 0.119 [0.110, 0.132] |
| null percentile of U(region) (p) | 100 (0.005) | 100 (0.005) | 100 (0.005) | 100 (0.005) |
| **d = 30 m** U region / matched A / AE | 0.181 / 0.178 / 0.053 | 0.157 / 0.194 / 0.105 | 0.083 / 0.148 / 0.073 | 0.079 / 0.206 / 0.073 |
| primary | +0.003 [−0.029, +0.029] | −0.037 [−0.141, +0.029] | −0.065 [−0.138, +0.010] | −0.126 [−0.157, +0.015] |
| secondary | +0.128 [−0.020, +0.174] | +0.051 [−0.056, +0.138] | +0.010 [−0.030, +0.081] | +0.007 [−0.030, +0.226] |
| U(region) vs null; percentile (p) | 0.181 vs 0.129 [0.125, 0.133]; 100 (0.005) | 0.157 vs 0.117 [0.114, 0.121]; 100 (0.005) | 0.083 vs 0.059 [0.050, 0.069]; 100 (0.005) | 0.079 vs 0.054 [0.044, 0.069]; 98.5 (0.020) |

Sensitivities of the primary at d = 100 m, together with the 45-feature seed-0 smoke numbers
(`results/2026-10-16/boundary_stat_45feat/`). Those were run with the brief's all-windows rule, and
their testable-only version is a sensitivity row.

| primary U(region) − U(matched A), d = 100 m | seed 0, S = 250 | seed 0, S = 500 | seed 1 rec. | seed 2 rec. |
|---|---|---|---|---|
| testable windows (**primary**) | +0.025 [−0.007, +0.058] | −0.059 [−0.226, +0.044] | −0.062 [−0.192, +0.070] | −0.160 [−0.221, +0.125] |
| all windows (the brief's rule, failed calibration) | +0.044 [−0.042, +0.108] | −0.011 [−0.124, +0.081] | −0.003 [−0.026, +0.007] | −0.033 [−0.035, +0.007] |
| within-component (each component's own difference, weighted by its region edge) | +0.034 [+0.002, +0.062] | −0.041 [−0.225, +0.066] | −0.065 [−0.192, +0.070] | −0.163 [−0.221, +0.125] |
| each shared lobe counted once (no interval) | +0.015 | −0.112 | −0.062 | −0.160 |
| 45 features, all windows | −0.042 [−0.189, +0.217] | −0.111 [−0.242, +0.123] | | |
| 45 features, testable windows | −0.006 [−0.145, +0.246] | −0.067 [−0.165, +0.135] | | |

**2 × 2 tables**, seed 0, testable windows, d = 100 m. Each cell is a share of the role's edge
length; the four cells add to 1. "level" = contour-like; "on" = within 100 m of the layer
(window median).

| layer | role | S = 250: level & on / level & off / not level & on / not level & off | S = 500: the same four |
|---|---|---|---|
| S (soil-survey lines) | region | 0.624 / 0.033 / 0.313 / 0.029 | 0.628 / 0.054 / 0.285 / 0.033 |
| | matched A | 0.638 / 0.038 / 0.287 / 0.037 | 0.582 / 0.037 / 0.343 / 0.038 |
| | AE | 0.720 / 0.159 / 0.104 / 0.017 | 0.661 / 0.108 / 0.205 / 0.025 |
| M (county lines, panel edges) | region | 0.039 / 0.619 / 0.025 / 0.317 | 0.044 / 0.638 / 0.023 / 0.295 |
| | matched A | 0.045 / 0.631 / 0.024 / 0.300 | 0.042 / 0.577 / 0.024 / 0.356 |
| | AE | 0.091 / 0.788 / 0.014 / 0.107 | 0.076 / 0.694 / 0.022 / 0.209 |
| S or M | region | 0.627 / 0.030 / 0.316 / 0.026 | 0.631 / 0.051 / 0.288 / 0.030 |
| | matched A | 0.642 / 0.034 / 0.291 / 0.033 | 0.586 / 0.034 / 0.346 / 0.034 |
| | AE | 0.741 / 0.138 / 0.108 / 0.013 | 0.673 / 0.096 / 0.208 / 0.023 |
| canals (NHDPlus, reported separately) | region | 0.001 / 0.657 / 0.001 / 0.341 | 0.004 / 0.678 / 0.002 / 0.316 |
| | matched A | 0.004 / 0.672 / 0.004 / 0.320 | 0.004 / 0.616 / 0.003 / 0.378 |
| | AE | 0.004 / 0.875 / 0.001 / 0.120 | 0.010 / 0.759 / 0.008 / 0.223 |

**Per component**, seed 0, S = 500, testable windows, d = 100 m:

| cc | where | region edge, km | U region / matched A / AE | region − matched A | contour-like region / matched A / AE | null percentile of U(region) |
|---:|---|---:|---|---:|---|---:|
| 0 | Dixie, Lafayette (Big Bend) | 2,097 | 0.285 / 0.297 / 0.076 | −0.013 | 0.690 / 0.679 / 0.912 | 100 |
| 1 | Hamilton | 538 | 0.344 / 0.299 / 0.787 | +0.046 | 0.611 / 0.644 / 0.167 | 100 |
| 2 | Bradford | 1,077 | 0.390 / 0.349 / 0.331 | +0.041 | 0.554 / 0.614 / 0.633 | 100 |
| 3 | Lafayette, Taylor | 3,854 | 0.351 / 0.275 / 0.076 | +0.075 | 0.626 / 0.710 / 0.913 | 100 |
| 4 | Gulf | 1,974 | 0.074 / 0.464 / 0.199 | **−0.390** | **0.906** / 0.513 / 0.780 | 20.5 |
| 5 | Bradford | 614 | 0.358 / 0.324 / 0.331 | +0.034 | 0.576 / 0.639 / 0.633 | 99.5 |

At S = 250, all four components have region ≥ matched A (+0.020, +0.003, +0.002, +0.068).
Components 1–3 at S = 250, and 0 and 3 at S = 500, share one set of AE lobes, because their Big
Bend boxes overlap. Hamilton's AE edge (111 km) is small and is mostly *not* contour-like.

Per component, recurring regions (seeds 1 and 2), testable windows, d = 100 m:

| seed | cc | where | region edge, km | U region / matched A / AE | region − matched A | contour-like region / matched A / AE |
|---:|---:|---|---:|---|---:|---|
| 1 | 0 | Collier, Hendry | 362 | 0.242 / 0.172 / 0.176 | +0.070 | 0.744 / 0.812 / 0.624 |
| 1 | 1 | Sarasota | 388 | 0.077 / 0.268 / 0.146 | −0.192 | 0.917 / 0.716 / 0.821 |
| 2 | 0 | Collier, Hendry | 90 | 0.486 / 0.361 / 0.183 | +0.125 | 0.504 / 0.569 / 0.565 |
| 2 | 1 | Sarasota | 441 | 0.076 / 0.298 / 0.146 | −0.221 | 0.918 / 0.686 / 0.821 |

**Recurrence across seeds** (C5, `results/2026-10-16/regions/recurrence_spatial_S500.csv`):
- None of the 6 seed-0 regions recurs. 3 of them (Big Bend cc 0 and 3, Hamilton cc 1) lie in
  tiles that are test tiles in neither other seed, so they could not recur.
- The two Bradford regions (testable in seed 1) and Gulf (testable in seed 2) could have
  recurred and did not.
- Seeds 1 and 2 share two recurring regions: Collier/Hendry (71% / 62% of points within 2 km of
  the other seed's region) and Sarasota (100% / 84%). Both are in the tables above.

**Reading it with `REGION-ANALYSIS-FLORIDA.md` §8.**

*Seed 0 (north Florida).* The pattern matches the first half of §8's last row, "non-region Zone A
lobes look like region lobes". Region and matched Zone A edges are alike on every boundary
quantity:
- contour-like share 0.66–0.68 vs 0.62–0.68;
- share within 100 m of a soil-survey line 0.91–0.94 vs 0.92;
- U 0.29–0.32 vs 0.29–0.35;
- per component, region − matched A between −0.013 and +0.075, except Gulf.

That row's interpretation is "Zone A is a W/S label everywhere", where W = wetland/land-cover
lines and S = soil lines. It is only weakly supported here:
- Zone A edges are less contour-like than Zone AE edges (0.62–0.68 vs 0.77–0.88), but most Zone
  A edge is still contour-like.
- Wetland lines (W) were not measured.
- U(region) − U(AE) excludes 0 only at S = 250 (+0.208 [+0.013, +0.265]).

The safer statement is that the selected regions' edges are not special compared with the Zone A
around them. What makes the regions hard for the AE-trained models does not show in how their
boundaries were drawn. In §8's terms the explanation of the regions moves to Story 2 and R7
(terrain the AE models never saw), which this run did not test.

*Recurring regions (seeds 1–2).* The two regions differ:
- **Sarasota.** Its region edges are 92% contour-like, more than Zone AE (82%) and nearby Zone A
  (69–72%). Region − matched A is −0.19 / −0.22. That is the G1 half of §8's **Story 2** row:
  region edges are level sets like AE's.
- **Collier/Hendry.** Its region edges are less contour-like than nearby Zone A (0.74 vs 0.81;
  0.50 vs 0.57), and region − matched A is **+0.070 / +0.125**, the direction O5 predicts. But
  this is one component per seed, on 362 / 90 km of region edge.

The pooled recurring primary is negative because Sarasota has more region edge and the stronger
contrast. No case shows §8's "H2 with mechanism" pattern, in which region edges are mostly
off-contour and on soil or map lines while AE edges are contour lines. Collier/Hendry points that
way, but not by that much: its region edge is still 50–74% contour-like.

**Verdict: not met.** The primary is +0.025 [−0.007, +0.058] at S = 250, −0.059 at S = 500, and
−0.06 / −0.16 on the recurring regions. It is never both positive and with an interval excluding
0. The null part holds for every pooled U(region) (percentile ≥ 98.5). Per component it fails for
Gulf (20.5), and for seed-1 Sarasota at d = 30 m (94.0).

Notes on reading O5:
1. At d = 100 m the "on a line" test hardly discriminates. 91–94% of Zone A edge and 82–87% of
   AE edge lies within 100 m of a soil-survey line, so U at 100 m is mostly "not contour-like".
   d = 30 m tells the roles apart better and gives the same verdict.
2. The null says region edges sit on lines more often than shifted copies of themselves. The null
   is narrow (±0.004–0.013), so a small excess gives percentile 100. Matched Zone A was not given
   its own null, so this does not single out the regions.
3. The S = 500 negative is driven by Gulf County (cc 4). Its region edges are 91% contour-like,
   and the nearby Zone A is not. Without it, 4 of the 5 other components are positive (+0.03 to
   +0.08; Big Bend cc 0 is −0.013), and the pooled primary would be +0.021.
4. Counting each shared lobe once, instead of once per component, moves the seed-0 primary:
   S = 250 from +0.025 to +0.015, S = 500 from −0.059 to −0.112. The matched Zone A lobes that
   components 0 and 3 share then weigh more. The verdict is the same.
5. Seed-0 Bradford components use DEM tile n30w082, and both recurring regions use four tiles,
   that were republished after the features were pulled. There the level-set test reads a
   possibly newer surface than the features saw.

---

# Run log

## Part A. Set-up checks: all pass

| check | result | file |
|---|---|---|
| A1 feature list | 7 features: `landcover, slope, tri_300, wetindexcat, log10_dist_river, log10_dist_coast, hand` | `data/processed_w7/FEATURES.txt` |
| A1 link | `data/processed_w7/dataset_FL.parquet -> ../processed/dataset_FL.parquet` resolves | |
| A1 checksums | both SHA-256 `3f724dac…d621f5c71`, identical | `logs/2026-10-16/A1_checksums.log` |
| A2 missingness | as expected, below | `logs/2026-10-16/A2_missingness_w7.log` |
| disk | 121 GB free | |

A2, percent missing by stratum (domain × label; label 1 = inside the flood zone, 0 = band
negative 500–5,000 m outside it):

| domain | label | wetindexcat | hand | other five |
|---|---:|---:|---:|---:|
| A | 0 | 3.56 | 2.44 | 0 |
| A | 1 | 0.00 | 0.00 | 0 |
| AE | 0 | 31.29 | 29.72 | 0 |
| AE | 1 | 2.46 | 0.10 | 0 |

Matches the brief: ~30% of Zone AE band negatives and < 3% of positives missing `hand` and
`wetindexcat`.

## C1. Cells (built before Part B; see Disagreements, item 8)

Command as in the brief, with `caffeinate -i` added. Logs `logs/2026-10-16/zoo_w7_s{0,1,2}.{out,err}`
(the `.err` files hold only sklearn's MLP "maximum iterations reached" warnings, 37 per seed, as in
every earlier run). Cells in `data/interim/aotil/cells_w7/`. `data/interim/aotil/cells/` was not
written (its seed-0 cell still has its 2026-10-14 23:18 timestamp).

| seed | header | train / ID test / OOD test | tiles train / test | brief expects | audit |
|---|---|---|---|---|---|
| 0 | `[FL] 80,000 rows \| 7 features`, 300 models | 28,109 / 10,235 / 12,771 | 71 / 24 | same | **PASS** (0 fail, 0 warn) |
| 1 | same | 29,395 / 10,479 / 15,036 | 71 / 25 | same | **PASS** (0 fail, 0 warn) |
| 2 | same | 25,670 / 11,426 / 14,357 | 71 / 23 | same | FAIL (1 fail), then after the fix below **PASS** (0 fail, 0 warn) |

Audit files: `results/2026-10-16/audit/FL_band_AE_s{0,1,2}_audit.json`, logs
`logs/2026-10-16/C1_audit_cell_s{0,1,2}.log`.

Against the first look (`results/2026-10-16/w7_first_look/models/*_models.csv`), model by model: the
same 300 specs. hgb, lr, mlp, nb and xgb give identical numbers (xgb to 1e-7). So do the decision
trees, except one: seed 2 idx 208 differs by 0.0025 in OOD AUC. Random forests and extra trees
differ by at most 0.0035 in ID AUC (seed 0, et), well inside the 0.02 tolerance. Most likely
cause: the first look ran on a different machine, and tree ensembles that average many trees round
differently there. The audit's refits on this laptop reproduce the stored scores exactly (max
difference 0 to 1e-16), so the cells are deterministic here.

### Stop 1: the seed-2 cell audit fails one check

```
[FAIL] models  v2 columns: 1 problems, first: ['idx 142 training metrics']
```

What it is. `audit_cell.py` (line 376) checks that the four training-fit columns of every model
(`acc_train_sub, auc_train_sub, acc_train_all, auc_train_all`) lie in [0, 1]. Model 142 of seed 2
is a random forest with unlimited depth and 200 trees, trained on all 25,670 training rows. It
fits them perfectly, and sklearn's AUC on them comes out as `1.0000000000000002`, one rounding step
of a 64-bit float above 1. (sklearn computes AUC as the area under the ROC curve with the trapezoid
rule, a sum of many small floats, and the last bit of that sum depends on the machine.) The first
look stored exactly `1.0` for the same model. The check has no tolerance, so it fails.

Why it is harmless. These four columns describe how well each model fits its own training data.
Nothing downstream reads them: the selection uses the per-point test scores (`S_id`, `S_ood`), and
every check of those passed, including the exact refits. The other 20 rows of the seed-2 audit
(18 PASS, 2 INFO) are fine.

Options:
1. **Add a 1e-12 tolerance to that one range check** in `aotil/audit_cell.py` (a one-line change,
   written up here), re-run the seed-2 audit, and continue if it then passes. Recommended: the check
   exists to catch real out-of-range values (a NaN, a negative), and a 2e-16 overshoot is not one.
2. Accept the seed-2 cell as it is, with this FAIL written up, and continue.
3. Clip AUCs to [0, 1] in `aotil/zoo.py` and rebuild the seed-2 cell. Not recommended: it changes
   the code that makes the data, for no change in any number that matters.

Decision (Victor): option 3. `aotil/zoo.py` `score_rows` now clips the AUC to [0, 1] (one line,
with the reason in its docstring), and the seed-2 cell was rebuilt with `--overwrite` into
`cells_w7/`. Against the cell before the rebuild: `S_ood` and `S_id` byte-identical; in the model
table only `auc_train_sub` and `auc_train_all` of idx 142 changed, by 2.2e-16 (now exactly 1.0).
Re-audit: **PASS** (0 fail, 0 warn), `logs/2026-10-16/C1_audit_cell_s2.log`. Seeds 0 and 1 have no
value outside [0, 1] in any acc/AUC column, so the clip changes nothing in them and they were not
rebuilt.

## C2. O1 and O2 (run notes)

The brief's command was used as is: `aotil/o1_shift.py` with the three 7-feature cells into
`results/2026-10-16/o1_shift.csv`, and the 45-feature seed-0 cell into `o1_shift_45feat_s0.csv`.
Logs: `logs/2026-10-16/C2_o1_shift*.log`.

It ran three times, as the code was fixed:
1. Stride 10, as in the first look. It matched the first look to within 0.001.
2. With the pooled rows added.
3. Final: the bootstrap over all 300 models (the review's fix, B2).

The point values never changed; only the intervals did. The tables are in O1 and O2.

## C3. Selections: notes while running

Seed 0 finished in 23 min (21:30–21:54; log `logs/2026-10-16/C3_selections.log`). Audit:
`logs/2026-10-16/C3_audit_selection_s0.log`, 32 pass, 1 warning (a tie at the smoothed-classic cut,
all inside the cut, so the mask does not depend on point order), 0 fail.

**Against the first look: one number differs by more than 0.02, and why.** Test r, first look →
this run:

| variant | S = 250 | S = 500 | S = 1,000 |
|---|---|---|---|
| classic | −0.930 → −0.935 | −0.889 → −0.888 | −0.905 → −0.905 |
| spatial | −0.818 → −0.825 | **−0.888 → −0.815** | −0.883 → −0.883 |

The cells are not the cause. The first look's spatial S = 500 θ, scored on this run's cell, gives
validation r −0.857 and test r −0.888, the first look's own values to within 0.001 (same check for
all six runs: within 0.003). The search is the cause. Optuna's sampler is seeded, so the first,
random trials are the same in both runs. After that, each new trial is chosen from the scores of
the earlier ones, and those scores differ in the last digits: the random-forest and extra-trees
models differ by up to 0.005 in accuracy between the two machines (C1). The two searches had
parted by trial 71 (different hyperparameters). This run's best trial (89) reaches a *better*
validation r than the first look's best (−0.869 vs −0.858), and its test r is worse (−0.815 vs
−0.888). The selected points overlap little (Jaccard 0.23). Both are valid outcomes of the same
protocol. The difference says how far one spatial S = 500 result can move with the search path:
about 0.07 in test r, against 0.00–0.01 at the other five settings. The O4 "cost of coherence" at
S = 500 is therefore 0.072 in this run, not the first look's 0.000.

All three seeds: seed 1 took 28 min, seed 2 45 min (it shared the machine with the review
agents). Audits `logs/2026-10-16/C3_audit_selection_s{0,1,2}.log`: 0 fail in every seed; the only
warnings are 1–3-point ties at the smoothed-classic cut. Seed 1 has two such ties that straddle
the cut, so one point of that ablation's mask depends on point order. Every run used 100 trials,
none failed. Seed 2 spatial at S = 500 also differs from the first look by more than 0.02
(−0.748 → −0.801; classic −0.914 → −0.931 is 0.017, within tolerance), for the same reason as seed
0: the first look's θ on this run's cell gives −0.747 and −0.911, its own values, and the two
searches picked different trials.
Seed 1 at S = 500 agrees (classic −0.936 → −0.948, spatial −0.838 → −0.842).

The per-seed table of test r, baselines and coherence is in O3.

## C4. Controls (S = 250, 500, 1,000; seeds 0–2)

Run with `logs/2026-10-16/C4_chain.sh` (log `logs/2026-10-16/C4_controls.log`): 54 null-select runs, 36 family-split runs and 6 μ-sweep runs, every one with 100 trials and none failed. Audits: `logs/2026-10-16/C4_audit_{null,splits}_s{0,1,2}.log` and `C4_audit_mu_sweep_s0.log`, with reports under `results/2026-10-16/selection_audit/`. All 7 give **0 fail**. Every warning is for the budgets the controls deliberately skip.

The tables are in O3 (null-select, family split) and O4 (μ sweep).

## C5. Regions and recurrence

Spatial S = 250 and 500 and classic S = 500 for each seed. Files: `results/2026-10-16/regions/`, logs
`logs/2026-10-16/C5_regions_*.log`. Every budget matched the component and tile counts its
selection recorded.

Seed 0 spatial S = 500: 20 components in 5 tiles. The 6 components of ≥ 20 points hold 418 of the
500 points, all of them Zone A floodplain (no band negatives):

| cc | points | counties | land cover |
|---:|---:|---|---|
| 0 | 182 | Dixie, Lafayette (Big Bend) | evergreen forest 42% |
| 1 | 90 | Hamilton | evergreen forest 46% |
| 2 | 55 | Bradford | evergreen forest 42% |
| 3 | 39 | Lafayette, Taylor | evergreen forest 46% |
| 4 | 31 | Gulf | grassland/herbaceous 29% |
| 5 | 21 | Bradford | evergreen forest 38% |

They differ from the first look's seed-0 regions because the selection differs (C3). Of the
first look's eight, only Big Bend, Hamilton and Bradford survive in some form; Palm Beach,
Miami-Dade, Gilchrist–Levy and Marion are gone.

Recurrence (`aotil/region_recurrence.py`, `results/2026-10-16/regions/recurrence_spatial_S500.csv`,
log `logs/2026-10-16/C5_recurrence_spatial_S500.log`). "Near" means the share of a region's points
within 2 km of a point of the other seed's regions (≥ 20 points). "Testable" means the share of its
points in tiles that are test tiles in that seed; only there can a region recur.
- **Seed 0: none of its 6 regions recur.** 3 of the 6 (cc 0, 1, 3) are in tiles that are test
  tiles in neither seed 1 nor seed 2, so they could not. Of the three that could:
  - the two Bradford regions (cc 2, 5) were testable in seed 1 and are not near;
  - Gulf (cc 4) was testable in seed 2 and is not near.
- **Seeds 1 and 2 share two recurring regions:**
  - Collier/Hendry: seed 1 cc 0 has 71% near seed 2, and seed 2 cc 0 has 62% near seed 1;
  - Sarasota: seed 1 cc 1 has 100%, seed 2 cc 1 has 84%.
- Seed 2's Calhoun County region (cc 4, FIPS 12013, in the Panhandle) is near seed 1 only through
  seed-1 fragments of < 20 points (`near_s1_any` 0.62).

## C6. Lobes, DEM, soils, distances (seed 0 spatial S = 250 and 500)

GIS outputs are under `results/2026-10-16/gis/` (approved deviation 1). `data/interim/gis/` was only
read: the county and panel line layers, and the 45-feature soil cache, which was copied, not moved.
- **Lobes** (`logs/2026-10-16/C6_lobes_S{250,500}.log`): the edge-direction assertion passes in
  every component. The bounding boxes cover 15,905 km² (S = 250) and 32,322 km² (S = 500) as a union.
- **DEM** (`logs/2026-10-16/C6_dem_{dry,fetch}.log`). The dry run gave whole tiles 3.517 GB, and
  `--remote` (read only the needed blocks of each tile, as RUN-FL-V2 does) at most 2.483 GB. I
  fetched with `--remote --max-gb 3.5`, so the download was under the 3.5 GB stop. Result: 7
  rasters built and 3 identical ones hardlinked, 1.1 GB on disk, 10 min.
  - Two Big Bend boxes have 75% valid cells, presumably because they extend over the Gulf.
  - **Warning from the dry run:** tile n30w082 was republished on 2025-12-08, after the features
    were pulled from Earth Engine's copy. On S = 500 components 2 and 5 (Bradford) the edge test
    may read a newer ground surface than the features saw.
- **Elevation at the vertices** (`logs/2026-10-16/C6_dem_sample_*.log`): every vertex has `z`,
  `z_in` and `z_out` (no `--allow-missing`).
- **Soils** (`logs/2026-10-16/C6_ssurgo{_plan,}.log`): the plan gave ~0.244 GB to fetch and ~1.74 GB
  on disk, under the stop. The copied cache already held 75,155 of the 146,160 polygons. Fetched
  244.5 MB from Soil Data Access in 9 min.
- **Distances** (`logs/2026-10-16/C6_edge_layers_*.log`, coverage JSONs beside the tables). S = 250:
  every layer ok. S = 500 first failed one coverage check, on canals for component 4 (Gulf
  County). The check tests the analysis box, which extends 3 km past the lobes and is wider still
  in longitude/latitude. Its southern 3 km strip (latitude 29.598–29.628) lies in the Gulf, south
  of the last NHDPlus flowline. The vertices themselves are 7.5 km north of that strip.
  - **Code change, outside the brief:** `aotil/edge_layers.py` `load_nhdplus_canals` now fails
    only if the vertices plus a 1 km margin (`CANAL_MARGIN_M`) leave the NHDPlus extents. Otherwise
    the layer is an explained `warn`. Canals are reported on their own and are not part of U.
  - Re-run: written, with 5 explained warnings, all for component 4:
    - the canal note;
    - the four soil layers (soil, soil_water, soil_join_county, soil_join_other). They share the
      soil coverage note: 424 AE vertices fall in no soil map unit (water or a survey gap) and
      are flagged `soil_ok` false. soil_join_other also has no federal or tribal survey join in
      the box. `test_boundary_stat.py`, which imports `edge_layers.py`, still passes.
- **The recurring regions of seeds 1–2** (done after Stop 2, at Victor's request, under the same
  3.5 GB stop). These are Collier/Hendry (component 0) and Sarasota (component 1) in both seeds.
  - **Code change:** `aotil/lobes.py --only-components` (default: all components of ≥ 20 points,
    as before) builds lobes for the listed components only. It still counts every selected point
    as selected, so another component's polygons can never become "unselected" matched Zone A.
  - Logs `logs/2026-10-16/C6_lobes_s{1,2}_S500_recurring.log`: the edge-direction assertion
    passes; the bounding boxes cover 22,100 km² (seed 1) and 18,245 km² (seed 2).
  - DEM dry run (`C6_dem_dry_recurring.log`): `--remote` at most 1.514 GB, rasters at most
    1.237 GB, so under the stop. Fetched 3 rasters plus 1 hardlink in 6 min.
    **Warning:** 4 of the 7 tiles were republished after 2024 (n26w081, n26w082, n27w081,
    n28w083), covering both regions, so the level-set test there may read a newer ground surface
    than the features saw.
  - Soils plan (`C6_ssurgo_plan_recurring.log`): ~0.244 GB to fetch and ~0.71 GB on disk, under the
    stop. Fetched 244 MB.
  - Every vertex has `z`, `z_in` and `z_out`. Distances (`C6_edge_layers_FL_band_AE_s{1,2}_*`): every
    layer ok, no warnings.
  - The component bootstrap has only these 2 components per seed, so its interval is crude.

## Stop 2 (C7): the level-set calibration fails on this run's tables

`aotil/boundary_stat.py --calibrate-only` on the two 7-feature tables. Logs:
`logs/2026-10-16/C7_boundary_stat_calibrate_*.log`; files `results/2026-10-16/boundary_stat/*_calibration.csv`
and `*_windows.parquet`; the testable-only shares in `logs/2026-10-16/C7_calibration_testable.log`.
The null has not run.

The test, step 3 of B3: a 300 m window is "contour-like" if its elevation range is ≤ τ and the
ground drops into the floodplain. That second condition is a median `z_out − z_in` > 0 over ≥ 5
vertices whose inside and outside probes landed on the correct side. The brief's calibration says
most Zone AE windows should pass, since a detailed study's edge is where a computed water surface
meets the ground. **Stop if fewer than half do.**

Share of edge length that passes (region / matched_A / AE):

| test | S = 250 | S = 500 |
|---|---|---|
| **τ = 0.76 m** (the brief's) | 0.481 / 0.509 / **0.438** | 0.486 / 0.474 / **0.418** |
| **τ = 1.5 m** | 0.624 / 0.606 / **0.455** | 0.611 / 0.604 / **0.474** |
| **detrended range ≤ 0.76 m** (range of the residuals after a straight-line fit of z along the edge) | 0.553 / 0.565 / **0.444** | 0.561 / 0.569 / **0.444** |
| range ≤ 0.76 m alone, without the drop condition | 0.752 / 0.788 / 0.944 | 0.771 / 0.678 / 0.830 |
| testable: ≥ 5 vertices with both probes landing right | 0.732 / 0.752 / **0.499** | 0.713 / 0.765 / **0.544** |
| τ = 0.76 m + drop, **among testable windows only** | 0.658 / 0.676 / **0.879** | 0.682 / 0.620 / **0.769** |
| τ = 1.5 m + drop, among testable windows only | 0.853 / 0.805 / 0.912 | 0.856 / 0.789 / 0.872 |
| detrended ≤ 0.76 m + drop, among testable windows only | 0.756 / 0.751 / 0.891 | 0.786 / 0.744 / 0.816 |

What this says, in order:
1. The three tests the brief asked for all leave AE under one half (0.42–0.47), so the stop holds.
2. The cause is not the elevation range. On range alone, AE edge passes most often of the three
   roles (0.94 and 0.83). Detrending, the brief's guess (sloping river surfaces), barely moves it.
3. The cause is testability. Only half of AE edge has ≥ 5 vertices where both 100 m probes landed
   on the correct side. A window the probes cannot test counts as "not contour-like", and
   therefore can count toward U. Most likely the inside probe overshoots: many AE lobes are
   narrower than 100 m, so a point 100 m inside lands outside the lobe.
4. **Among the testable windows the test calibrates.** AE passes 88% (S = 250) and 77% (S = 500)
   at τ = 0.76 m, more than both Zone A roles. A detailed study's edge should behave exactly
   that way.

Options:
- **(a) Primary on testable windows only (recommended).** Keep the brief's test (τ = 0.76 m +
  drop) and compute U and the null over windows with ≥ 5 good probes. It fixes what actually
  fails, keeps the test as written, and passes calibration. Cost: it drops 27–29% of region edge,
  25–24% of matched_A and 50–46% of AE. The full-edge U stays in the report as a sensitivity.
- (b) Keep the brief's test on all windows and accept the failed calibration. The result would
  then mix "not level" with "could not be tested", and that mix differs by role (AE is half
  untestable).
- (c) Drop the drop condition and use range ≤ 0.76 m alone. It passes calibration (AE 0.94 /
  0.83) but loses the check that the ground rises outward.
- (d) Shorter probes: re-run `pull_dem.py sample --offset-m 30` (the probe distance; default 100 m)
  and `edge_layers.py`, then recalibrate. It attacks the cause directly and needs no download,
  but it changes the input tables and needs another calibration round.

The null runs for whichever option you choose, with 200 draws and seed 0.

**Note before you decide.** The calibrate-only run also prints and writes the observed U values
(the null and the bootstrap are not run). I have kept them out of this section so that the
choice can be made on the calibration alone. They are in `logs/2026-10-16/C7_boundary_stat_calibrate_*.log`.
The calibrate-only summary files were later overwritten by the testable-window runs.

**Decision (Victor, 2026-10-17): option (a).** The primary statistic uses testable windows only,
and the full-edge U is kept as a sensitivity. Victor also asked for C6 on the recurring seed-1/2
regions, under the same 3.5 GB stop. Implementation and runs: C7 below. Results: O5.

## C7. The boundary statistic (run notes)

After Stop 2 (decision above), I added `--windows {all,testable}` (default `all`, the brief's rule)
to `aotil/boundary_stat.py` (decision 11 in its docstring). With `testable`:
- The untestable windows are dropped right after the calibration, and each vertex's window index
  is remapped. The statistic, the 2 × 2 tables, the per-component rows, the bootstrap and the null
  therefore all use the same windows.
- The calibration table and `windows.parquet` still cover all windows.

Check: on the S = 250 table, a `--windows testable` run reproduces an `--windows all` run's
"testable-only" sensitivity rows exactly (values, bootstrap intervals, null percentiles; difference
0.0). Its null draws are another realisation of the same null: lobes without a testable window are
not displaced, so the seed hands out the per-lobe shifts in a different order.

Runs (scripts `logs/2026-10-16/C7_run.sh`, `C7_run_recurring.sh`, logs `C7_run*.log`):
- seed 0 S = 250 and S = 500, and the recurring seed-1 and seed-2 regions at S = 500;
- each with `--windows testable` into `results/2026-10-16/boundary_stat/` and `--windows all` into
  `results/2026-10-16/boundary_stat_allwindows/`;
- 200 null draws, 1,000 bootstrap draws, seed 0.

Every run passed the zero-shift self-check. The seed-0 runs took 5–9 min each, the recurring ones
1–2 min. The brief's C7 command differs from what ran: the `--gis-dir` is the run's GIS folder,
the county/panel `--layer-file`s point at `data/interim/gis/`, and `--windows` was added. The flag
names are as defined in B3.

## C8. The Florida map

Script `results/2026-10-16/figures/florida_map.py`, written by a sub-agent and checked here.
Figure `results/2026-10-16/figures/florida_map.{pdf,png}`, log `logs/2026-10-16/C8_florida_map.log`
(it prints the caption).

Panel (a), Florida:
- Zone A and Zone AE flood zones, drawn light;
- counties;
- each seed's 50 km test tiles, one line style per seed;
- the spatial S = 500 regions of all three seeds (6 / 7 / 5 regions of ≥ 20 points), with the
  four recurring ones (Collier–Hendry and Sarasota, seeds 1 and 2) emphasised and labelled.

Panels (b)–(e) zoom on the seed-0 regions (Big Bend, Hamilton, Bradford, Gulf). Each region's Zone
A edge windows are coloured by class: level set, on a soil or map line and not level, neither, or
not testable (grey).

Checks built into the script, which exits 1 if any fails:
- Every dataset point (80,000) and every test point of every seed lies in the 50 km square of
  its tile id.
- The windows, rebuilt with `boundary_stat.py`'s own functions, match the analysis's windows table
  row for row (115,245 windows).

The script is deterministic: a rerun after I moved its cache to the system temp folder gave
byte-identical PDF and PNG. Runtime is 2 min with a cold cache, 25 s warm, at about 2 GB peak
memory. There is no basemap (contextily and cartopy are not installed).

A data quirk the map turned up: one physical polygon is a region lobe of both seed-0 S = 500
components 0 and 3, and another of both 2 and 5. Each copy was sampled on its own component's DEM
raster, and those rasters have different extents, so their elevations differ by up to 0.11 m in
`z_range`. 6 of that polygon's 4,032 windows change level-set class between the copies. Pooled U
counts such a lobe once per component (B3 decision 2). The 6 flipped windows barely matter, but
counting shared lobes once does move the primary (O5, note 4). The map draws each shared window
once.

---

# Code changes

Nothing is committed. Changed files: `aotil/oodselect_paper.py`, `aotil/audit_selection.py`,
`aotil/report_paper.py`, `aotil/test_oodselect_paper.py`, `run_oodselect_local.sh`, `aotil/zoo.py`,
`aotil/edge_layers.py`, `aotil/lobes.py`. New files: `aotil/o1_shift.py`, `aotil/test_o1_shift.py`,
`aotil/boundary_stat.py`, `aotil/test_boundary_stat.py`, `aotil/region_recurrence.py`, and
`results/2026-10-16/figures/florida_map.py`. All defaults are unchanged except the two noted
below as fixes.

## Part B (B1–B3)

Every default is unchanged. The proof for B1: the seed-0 classic and spatial runs at S = 250
(3 trials, 500 epochs), made with the new code and with the committed code (`git show HEAD:...`),
give a byte-identical θ (the per-point selection weights) and identical values in every
`result.csv` column except the run time. The new code only adds two columns, `split` and
`null_seed`, plus the keys `split_rule` and `null_seed` in `meta.json`.

### B1. Null-select and the family-disjoint split

`aotil/oodselect_paper.py`:
- `--null-seed N` (default off). `null_accuracies()`: the ID accuracies of the selection and
  validation models are permuted together, as one `np.random.default_rng(N).permutation` over
  the union of the two blocks; the test models keep theirs. The search is otherwise the same.
  It uses the same internal seed as the real run at that S. That gives each trial the same
  starting θ and makes the first 10 Optuna trials (the random start-up draws) identical. Later
  trials are chosen from the scores so far, so they generally differ. The kept θ is scored on
  the test models against their true accuracies. In `result.csv`, `r_train` and `r_val` are against
  the shuffled accuracies the search saw; `r_test`, `r_full_test` and the scipy columns always use
  the true ones. The brief did not say whether to permute the two blocks together or each block on
  its own; I permuted them together.
- `--split {random,trees->others,others->trees}` (default `random`). `family_split()` uses
  `cells.TREE_FAMILIES` (hgb, rf, et, dt, xgb: 188 of the 300 models; lr, mlp, nb: 112). A random
  25% of the selection family, `np.random.default_rng(--seed)`, is the validation block. trees→others:
  141 selection / 47 validation / 112 test models. others→trees: 84 / 28 / 188.
- Variant names (`variant_name()`): `_trees2others` / `_others2trees`, then `_null<N>`. A control
  run refuses an `--out` folder of any other name, so a null run can never land in `classic/`
  and be read as the real result.

`aotil/audit_selection.py` and `aotil/report_paper.py` (approved deviation 2): both read each run's
split rule, null seed and seed back from its `result.csv` (`run_controls()`) and rebuild the same
split with `make_split()`. The audit then checks `meta.json` against that split, recomputes test r
on that split's test models, and recomputes train/validation r against the accuracies the search
saw. The report computes the random-subset and hardest-points baselines on that split's test models.
Controls get no "Δr vs classic" and stay out of `<cell>_comparison.csv`, which is the cost of
coherence on one shared split. A folder whose runs disagree on (split, null seed) fails, and
so does a folder whose `result.csv` names another variant: a null run copied into `classic/`
fails both the audit and the report. A family-split audit also prints its own full-set r
(trees→others +0.681 on 112 test models, others→trees +0.804 on 188; seed 0).

`run_oodselect_local.sh`: knobs `NULL_SEEDS` (default empty = off), `SPLITS` (default `random`)
and `BUDGETS` (default empty = the paper grid; approved deviation 3). Folder names follow
`variant_name()`.

Tests:
- `aotil/test_oodselect_paper.py` gained section 5 (the split blocks, seeding, the null
  permutation, variant names, reading old and new `result.csv` rows). All pass on the synthetic
  cell and on the 7-feature seed-0 cell (`logs/2026-10-16/B1_test_oodselect_paper_{synthetic,w7s0}.log`).
  Section 1, equivalence with the authors' code under PyTorch Lightning, is skipped because
  `pytorch_lightning` is not installed; this is unrelated to the change.
- Smoke run through `run_oodselect_local.sh` with `TRIALS=3 MAX_EPOCHS=500 BUDGETS="250 500"`:
  defaults plus both family splits, then `NULL_SEEDS="1 2"`. All 20 tasks finished, and
  `audit_selection.py` on all ten variants gives 0 FAIL. The only warnings are the budgets the
  smoke skipped and one tie at the smoothed-classic cut that does not affect the mask. A copy with
  a wrong split label and a wrong null seed fails ten checks, as it should. Files:
  `results/2026-10-16/smoke_B1/`, `logs/2026-10-16/smoke_B1/`. The smoke numbers come from 3
  trials and mean nothing scientifically.

### B2. `aotil/o1_shift.py`

Adopted from `results/2026-10-16/w7_first_look/o1o2.py`, with the same argument order and the
O1 arithmetic unchanged, including `np.round(..., 9)` when p(flood) is rebuilt from the
true-class scores. Changes:
- `r_acc_test60` is dropped. It used its own random 60 models.
- O2 is now `r_full_test`: the selection code's own function (`oodselect_paper.hard_corr` on the
  60 test models of `model_split`), with its Fisher 95% interval. Also `r_all_models`, defined as
  in `report_paper.py`, and `n_models_needed`, the paper's model-count rule
  (`report_paper.model_convergence`).
- Flags `--boot` (200), `--model-stride` and `--seed` (0).
- **A fix found in review, not in the brief: the interval now covers all 300 models.** The first
  look's tile-bootstrap interval was computed on every 10th model, to save time. The zoo cycles
  through the 8 learner families in a fixed order (model i belongs to family i mod 8), and 10
  shares the factor 2 with 8. Every 10th model is therefore always one of dt, hgb, mlp and rf;
  et, lr, nb and xgb never enter. The interval belonged to a 30-model, 4-family statistic, yet it
  was printed beside the 300-model gap. The bootstrap now uses all models (`--model-stride 1`,
  the new default; `--model-stride 10` reproduces the first look exactly). To keep this fast, the
  per-model AUC under bootstrap weights is computed in one vectorised pass. That pass handles
  ties the way sklearn does: a tied positive–negative pair counts as half. It matches sklearn's
  `roc_auc_score` to 1e-15 (`aotil/test_o1_shift.py`, new).
- **The pooled rows** (`cell = pooled`, one per scope; the brief asks for per-seed and pooled
  O1). The point values are means over the three seeds. The interval is a bootstrap over the
  union of the seeds' test tiles (47 tiles, 23 of them test tiles in more than one seed). One
  draw resamples that union, and all three seeds weight their points by the same draw, because
  the seeds share ground and are not independent. `n_A`/`n_AE` on pooled rows are sums over
  seeds; `n_A_distinct`/`n_AE_distinct` count each point once.

Smoke run (seed 0, 5 bootstrap draws, the first look's stride 10): every O1 column within
0.0003 of the first look; `r_full_test` +0.729, the first look's value.

### B3. `aotil/boundary_stat.py` (the O5 statistic) and `aotil/test_boundary_stat.py`

Written by a sub-agent to the brief's spec, then reviewed and fixed (see "Review of Part B" below).

What it computes, in CS terms:
- **Input.** Each flood-zone polygon near a region is cut into "lobes", and each lobe's boundary
  ring is sampled every 30 m. One row is one sample point ("vertex"), with:
  - its ground elevation `z`;
  - the elevations `z_in` and `z_out` of a probe point a short step inside and outside the
    polygon;
  - its distance to the nearest line of each map layer.
- **Step 1, filter.** The script keeps only the floodplain's outer edge. It drops vertices on an
  edge shared with another flood polygon, vertices with no elevation, and vertices outside
  Florida or outside their own map's panels.
- **Step 2, windows.** Each ring is cut into windows of 10 consecutive vertices (300 m). The cut
  follows the ring's original vertex order, so a window never spans a gap left by the filter. A
  window with fewer than 8 usable vertices is dropped.
- **Step 3, level set.** The question: does the edge follow a contour line, as a flat water
  surface meeting the ground would? A window passes if two things hold:
  - the elevation range along it is at most τ = 0.76 m;
  - the ground really drops into the floodplain. That means the median of `z_out − z_in` is
    above 0, over at least 5 vertices whose two probes landed on the correct side.
- **Step 4, on a line.** A window counts as lying on a layer if its median distance to that
  layer is at most d (100 m, and 30 m as a second setting). S is the soil-survey polygon edges.
  M is the nearest of county lines and FIRM panel edges (a FIRM panel is one sheet of the flood
  map, so its edges are arbitrary rectangle sides).
- **Step 5, U.** U is the share of edge length that is not a level set **and** lies on an S or
  M line: edge that looks drawn along a map line rather than along the terrain. It is computed
  per role, where `region` is Zone A edge inside a selected region, `matched_A` is unselected
  Zone A edge in the same tiles, and `AE` is detailed-study edge. The primary statistic is
  U(region) − U(matched_A).
- **Step 6, displaced-edge null.** Each region lobe is shifted rigidly by a random 300–1,000 m
  in a random direction, 200 times. Each time, the distances to the S and M lines are recomputed
  with `edge_layers.py`'s own code, keeping every window's level-set result. This shows how often
  an arbitrary curve of the same shape lands on a line, which matters because soil lines are
  dense in Florida. A self-check confirms that a zero shift reproduces the stored distances
  exactly (difference 0.0 m).
- **Step 7.** A bootstrap over components (1,000 draws) gives the interval.

Choices the spec left open, each written up in the module docstring:
- Windows are weighted by edge length.
- A missing distance counts as "no line".
- A lobe that serves two components counts in both. A "count each lobe once" row moves U(AE) by
  at most 0.009 on the 45-feature smoke. On this run's tables it moves more: the seed-0 S = 500
  primary goes from −0.059 to −0.112 (O5, note 4).
- Null percentiles are mid-rank, plus a one-sided p = (1 + #null ≥ observed)/(1 + draws).
- `--level-test {range,iqr,detrended}` and `--tau` let the decision at the calibration stop be
  applied. `--calibrate-only` stops before the null.

Tests (`.venv/bin/python aotil/test_boundary_stat.py`): 64 checks, ALL PASSED. They include the
brief's three:
- a synthetic square lobe on a synthetic soil line has on-line share 1;
- the same lobe moved 500 m through the null's code has share 0;
- a synthetic DEM flat along the edge and dropping across it has level-set share 1.

The others cover windows, ties, the zero-shift self-check and the sensitivities. Nine
deliberately broken copies of the code each fail them.

**Smoke run on the 45-feature tables (45-feature numbers, not this run's).** Files:
`results/2026-10-16/boundary_stat_45feat/`, logs `logs/2026-10-16/B3_boundary_stat_45feat_S{250,500}.log`.
Run as the preview did (cut every 10 kept vertices, count windows), the code reproduces
`results/2026-10-16/w7_first_look/u_preview_45feat_S500*.csv` exactly (36 numbers, to 1e-16).

| 45-feature, seed 0 spatial | S = 250, d = 100 | S = 250, d = 30 | S = 500, d = 100 | S = 500, d = 30 |
|---|---|---|---|---|
| U region / matched_A / AE | 0.290 / 0.333 / 0.491 | 0.148 / 0.182 / 0.192 | 0.352 / 0.463 / 0.509 | 0.193 / 0.282 / 0.229 |
| **primary** U(region) − U(matched_A) [bootstrap 95%] | −0.042 [−0.189, +0.217] | −0.034 [−0.119, +0.115] | −0.111 [−0.242, +0.123] | −0.089 [−0.171, +0.049] |
| U(region) − U(AE) | −0.200 [−0.287, +0.082] | −0.044 [−0.121, +0.137] | −0.158 [−0.247, +0.004] | −0.036 [−0.100, +0.060] |
| U(region): observed vs null mean [null 2.5–97.5%] | 0.290 vs 0.253 [0.247, 0.260] | 0.148 vs 0.085 [0.080, 0.089] | 0.352 vs 0.324 [0.319, 0.328] | 0.193 vs 0.127 [0.122, 0.131] |
| null percentile of observed U(region) (p) | 100 (0.005) | 100 (0.005) | 100 (0.005) | 100 (0.005) |

Read against O5's rule (primary > 0, interval excludes 0, U(region) above the null's 95th
percentile), the 45-feature regions meet only the third part:
- Their edges lie on soil or map lines more often than shifted copies of themselves.
- They do so less often than nearby unselected Zone A edges.
- The null for the difference is the null for U(region) minus a constant (matched_A is not
  shifted), so its percentile always equals U(region)'s.

The 45-feature calibration (share of edge passing the level-set test, region / matched_A / AE)
already fails the brief's calibration rule:

| test | S = 250 | S = 500 |
|---|---|---|
| range ≤ 0.76 m + drop | 0.652 / 0.609 / **0.234** | 0.590 / 0.499 / **0.300** |
| range ≤ 1.5 m + drop | 0.738 / 0.698 / 0.267 | 0.676 / 0.633 / 0.349 |
| detrended range ≤ 0.76 m + drop | 0.702 / 0.680 / 0.248 | 0.641 / 0.578 / 0.320 |
| range ≤ 0.76 m alone (no drop condition) | AE 0.861 | AE 0.826 |
| ≥ 5 vertices with both probes landing right ("testable") | 0.888 / 0.823 / 0.382 | 0.816 / 0.745 / 0.429 |
| median drop > 0, among windows with ≥ 1 good probe | 0.866 / 0.863 / 0.774 | 0.857 / 0.893 / 0.854 |

**The brief's explanation of the failure does not fit.** The brief guessed that river water
surfaces slope along the reach. But detrending moves AE only from 0.300 to 0.320, and elevation
range alone passes 83–86% of AE edge, more than Zone A. What fails is the probe count. Only
38–43% of AE edge has ≥ 5 vertices where both probes landed on the right side. The inside probe
(100 m in) misses on about half of AE vertices, most likely because many AE lobes are narrower
than 100 m. Among windows where the probes could be read, the drop is positive on 77–85% of AE
edge. No τ or range variant can lift AE over one half. The question for the calibration stop is
therefore how to treat windows the probes cannot test, not which τ to use. Two sensitivity rows
were added for that decision; the primary statistic is unchanged:
- "testable windows only": U over windows with ≥ 5 good probes;
- "within-component": each component's own region − matched_A difference, weighted by its
  region edge.

On the 45-feature S = 500 table, testable-only gives primary −0.067 [−0.165, +0.135] and
secondary +0.017 [−0.033, +0.139]; within-component gives −0.051 [−0.134, +0.109].

### C5 helper: `aotil/region_recurrence.py`

No script existed, so this one is new. For each region (component of ≥ 20 points) of each seed
and each other seed k, it reports:
- `tested_k`: the share of the region's points in seed k's test tiles. This is the denominator,
  because a region can only reappear where its ground is in seed k's test half.
- `near_k`: the share within 2 km of a point of a seed-k region of ≥ 20 points.
- `near_k_tested`: the same share among the points in tested tiles.
- `near_k_any`: the same share against every point seed k selected, components of any size.

"Recurring" means `near_k` ≥ 0.5 for at least one other seed. Distances are straight-line metres
after projecting longitude/latitude onto a flat map (EPSG:5070, the equal-area projection used
elsewhere in the pipeline). Smoke run on the first-look regions: `near_k_any` reproduces the first
look's `recurrence_S500.csv` exactly. The first look counted fragments of any size;
`near_k` counts only regions of ≥ 20 points, as the brief words it.

### Other code changes (not in the brief's Part B)

| file | change | why |
|---|---|---|
| `aotil/zoo.py` `score_rows` | AUC clipped to [0, 1] | Stop 1, Victor's decision (option 3). It changes only values one float step above 1 |
| `aotil/edge_layers.py` `load_nhdplus_canals` | the canal coverage check tests the vertices plus 1 km (`CANAL_MARGIN_M`) instead of the 3 km-buffered box; inside → explained `warn`, outside → FAIL as before | the Gulf County box reached 3 km into the sea, past the last NHDPlus flowline, and the strict check blocked the whole S = 500 table; canals are not part of U |
| `aotil/lobes.py` | `--only-components` (default: all) | C6 for the recurring seed-1/2 regions only, while every selected point still counts as selected |
| `aotil/boundary_stat.py` | `--windows {all,testable}` (default `all`) | Stop 2, Victor's decision (a) |
| `aotil/o1_shift.py` | `--model-stride` default 1 (the first look used 10) | the review found that stride 10 sees only 4 of the 8 learner families (B2) |

### Review of Part B

After B1–B3 were written, a review workflow ran four reviewers, one each for:
- the B3 statistic;
- the B3 null;
- the B1 controls;
- B2 and the helpers.

Each reviewer's findings went to a separate verifier who tried to refute them. Of 20 findings, 9
were confirmed. The fixes went to two fixing agents, and each fix was checked by an independent
verifier. That verifier's checks included deliberately broken copies of the code, to confirm the
tests catch errors.

What was fixed:
- **The O1 interval** (the one high-severity finding): stride 10 meant the interval belonged to a
  4-family statistic. The pooled lower bound drops from 0.015 to 0.004 with all models.
- **Null percentiles** treat values within 1e-12 as ties. Before, floating-point noise split exact
  ties in the per-component nulls.
- **The covered-windows sensitivity** now compares the observed and null U over the same windows.
- **A stale `null.csv`** is now deleted on a calibrate-only rerun.
- **Two sensitivity rows** were added for the calibration decision: testable windows only, and the
  within-component difference.
- **Audit and report checks:**
  - a run whose `result.csv` names another variant than its folder now fails the audit and the
    report;
  - the audit header states each family split's own full-set r.
- **Wording:** the docstring on what a null run shares with the real run (only the first 10
  Optuna trials).
- **Robustness:** `region_recurrence.py` no longer crashes when no region exists.

Not fixed, by decision: the calibration stop is not enforced in code. It is a step for whoever runs
the pipeline, as the other stops are.

Tests, all passing:
- `aotil/test_oodselect_paper.py`, on the synthetic and the real seed-0 cell;
- `aotil/test_boundary_stat.py`, 64 checks;
- `aotil/test_o1_shift.py`.

---

# Stop decisions

1. **Stop 1, C1:** the seed-2 cell audit failed: one training AUC was 1 + 2.2e-16.
   - Victor chose to clip the AUC in `zoo.py` and rebuild the seed-2 cell.
   - The score matrices came out byte-identical, and the re-audit passed.
   - He also approved three deviations at the same time (below, 1–3).
2. **DEM and soil downloads, C6:** the method actually used stayed under 3.5 GB every time, so
   neither stop fired. For the DEM that is `--remote`, which reads only the needed blocks of each
   tile, as RUN-FL-V2 does.
   - Seed-0 boxes: DEM ≤ 2.48 GB, soils ≈ 0.24 GB.
   - Recurring regions: DEM ≤ 1.51 GB, soils ≈ 0.24 GB.
   - The dry runs' whole-tile figure was over 3.5 GB for the seed-0 boxes: 3.517 GB network,
     5.83 GB on disk. That method was not used. If you count the whole-tile figure, this should
     have been a stop.
3. **Stop 2, C7:** only 44% / 42% of Zone AE edge passed the level-set test, against the 50% the
   calibration requires. τ = 1.5 m gave 46% / 47%, and the detrended range 44% / 44%.
   - The cause is the probe count, not τ.
   - Victor chose (a): the primary statistic over testable windows, where AE passes 77–88%.
   - He also asked for C6 on the recurring seed-1/2 regions under the same 3.5 GB stop.

# Disagreements with the brief and the plan

1. **C6 would have overwritten the 45-feature GIS pulls** in the undated `data/interim/gis/`. The
   GIS outputs went to `results/2026-10-16/gis/` instead (approved).
2. **`audit_selection.py` did not accept the control variants.** The brief says it "already
   accepts suffixed variants". It reads any folder, but it checked every run against the 60/20/20
   split and would have failed every family-split and null run. The audit and `report_paper.py`
   now rebuild each run's split (approved).
3. **C4 needed a `BUDGETS` knob** to run only S = 250/500/1,000 (approved).
4. **The O5 primary statistic.** `REGION-ANALYSIS-FLORIDA.md` §3.3 pre-registers region minus
   Zone AE; the brief and the plan use region minus matched Zone A. I followed the brief and
   report both.
5. **The brief's explanation of the calibration failure** (river surfaces sloping along the reach)
   does not fit: detrending barely helps. The probes are the cause (Stop 2).
6. **Expected values from the first look.** Two spatial S = 500 results differ by more than 0.02:
   seed 0 −0.888 → −0.815, seed 2 −0.748 → −0.801. The cause is not the cells: the first look's
   selections score their own values on this run's cells. The Optuna search took a different path
   because tree-ensemble scores differ by up to 0.005 between the two machines. So the brief's
   expected values carry search-path variance of about 0.05–0.07 at spatial S = 500.
7. **The first look's O1 intervals** came from 30 models covering only 4 of the 8 families (B2).
   The brief adopted that script without noting it.
8. **Order:** the C1 cells were built before Part B, because B1's smoke test needs the seed-0 cell.
   No Part B change touched `zoo.py`'s output; the Stop-1 clip came later and is byte-neutral for
   the scores.
9. **The brief's C7 command** has no `--layer-file` arguments and points `--gis-dir` at
   `data/interim/gis`. In this run the region soils and DEM are in `results/2026-10-16/gis/` (item
   1), so the command was adapted, plus `--windows testable` (Stop 2).
10. **`PROJECT-PLAN.md` §3, O3:** "null-select must not invert" proved too strong a requirement.
    The null searches barely fit their shuffled labels, yet still land on strongly anti-correlated
    subsets: 9 of 27 classic and 4 of 27 spatial null runs fall below −0.3. The real inversion is stronger in every case, so the
    comparison that carries information is real against null at the same setting, not null
    against 0. The plan's wording should change before the paper uses it.
