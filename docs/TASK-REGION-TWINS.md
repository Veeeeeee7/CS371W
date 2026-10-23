# Task: reproduce and port the region-twin analysis

Brief for a Claude Code session in this repo, written 2026-10-19. It replaces
`docs/TASK-MATCHING-MODEL-TERRAIN.md` (moved to `Claude outputs/removed-2026-10-21/docs/` on
2026-10-21). **Do not run that brief.**

The paper's region analysis changed. It no longer uses terrain-matched Zone A controls and label
conflict. Instead it compares each region with its **twins**: held-out Zone AE points with the
same inputs. Claude computed every number in the new analysis in its sandbox with three
reference scripts, from the repo's own data:
- the outputs of 2026-10-16 (`cells_w7`, `oodselect_paper`, `regions`);
- `data/processed/dataset_FL.parquet`.

This task does three things:
1. checks that those scripts give the same numbers on Victor's machine;
2. ports them into `aotil/` with tests;
3. writes a short report.

Victor comes from CS, not GIS. Explain GIS terms in CS terms in anything you write for him.

---

## 0. What the analysis does

Everything is per split (seed 0, 1, 2), on land points only (NLCD `landcover` ≠ 11).

- **Regions:** the connected pieces of at least 20 points of the spatial S = 500 selection, from
  `results/2026-10-16/regions/FL_band_AE_s{seed}_spatial_mu3_knn10_S500_points.parquet`. That gives
  16 regions with land points.
- **Twins.** Each region point is matched one-to-one, by optimal assignment, to a held-out Zone AE
  land point, i.e. an ID test point of the cell, `point_id_id`. A twin must:
  - have the same NLCD land-cover class (exact match);
  - be at least 5 km (EPSG:5070) from every point of the set being matched.
- **Matching cost.** Among the candidates, the assignment minimizes the cost per pair: the mean
  of |difference| / SD over the six numeric features
  `slope, tri_300, hand, wetindexcat, log10_dist_river, log10_dist_coast`. Each SD is taken over
  the split's Zone A land test points.
- **d** = the mean cost over pairs: the average difference per feature, in standard deviations.
- **g** = accuracy on the twins minus accuracy on the set. Accuracy is the mean correctness of the
  15 "strongest" test models: those with the highest ID AUROC among the 60 test models of the
  sklearn 60/20/20 model split. `g_all` uses all 60 test models. (Updated 2026-10-21: the ID
  AUROC is the stored `auc_id` of `<tag>_models.csv`, which `zoo.py` ranks on p itself, so tied
  scores stay tied. The sandbox recomputed it from p rebuilt from `S_id`, which changed the
  strongest 15 in split 2.)
- **Reference patches.** Each region gets 200 random connected patches of the same size, of each
  of two kinds. Both kinds are grown by breadth-first search on a 10-nearest-neighbor graph
  (EPSG:5070), and each patch gets its own twins by the same rule.
  - `ref_AE`: held-out Zone AE land points, starting anywhere.
  - `ref_A`: unselected Zone A land points (not in the classic or spatial S = 500 top 500),
    starting in the region's tiles.
- **Recurrence.** For each region, compute r_te (the test models, the eps-probit Pearson of
  `oodselect_paper`) on its points in every split where at least 80% of them are OOD test points.
  Also report the share of those points that split selected.

## 1. Rules

- Do not change:
  - `data/processed/`, `data/processed_w7/`, `data/interim/aotil/cells/`,
    `data/interim/aotil/cells_w7/`;
  - anything under `results/2026-10-14/`, `results/2026-10-16/` or `results/2026-10-17/`;
  - `results/2026-10-19/paper_figures/region_twins_out_sandbox/`, which holds Claude's sandbox
    outputs and is the reference to compare against;
  - `CS371_Cho_Li/`, the paper.
- `export D=<today>`. New outputs go under `results/$D/o5_twins/`, logs under `logs/$D/o5_twins/`.
- New scripts go in `aotil/`, each with a docstring and a test. Add flags without changing defaults.
- Use `.venv/bin/python`. Respect each **Stop**. Do not commit or push.

## 2. Part A: reproduce (minutes)

From `results/2026-10-19/paper_figures/`:

```
../../../.venv/bin/python region_twins_ref.py 0 200
../../../.venv/bin/python region_twins_ref.py 1 200
../../../.venv/bin/python region_twins_ref.py 2 200
../../../.venv/bin/python region_twins_summary.py
../../../.venv/bin/python region_recurrence_ref.py
../../../.venv/bin/python fig_twins.py
../../../.venv/bin/python fig_maps_combined.py
```

The outputs go to `region_twins_out/`. Each split took about 70 s in the sandbox. Compare them
with `region_twins_out_sandbox/`:
- **Region rows** (`set == 'region'`) are deterministic. `d`, `g`, the accuracies, `p_*` and
  `flood_*` must match to 1e-6. **Stop** if they do not, and report which columns differ and by
  how much.
- **Reference-patch rows** use a seeded RNG (`1000 + seed`), so they should also match. If they
  do not (for example because of a numpy version difference), report the per-region percentiles
  and pooled medians next to the sandbox ones. That is not a Stop.
- `recurrence.csv` must match.

(Updated 2026-10-21.) Three scripts changed after the sandbox run, so expect these differences
and do not Stop for them:
- `region_twins_ref.py` ranks the strongest models by the stored `auc_id` (see §0). Split 2
  (seed 1) changes: its region rows differ from the sandbox in `g`, the accuracies and `p_*`
  (`d` does not change, because the twins do not depend on the models), and so do its
  reference-patch rows. Splits 1 and 3 must still match. Wherever this brief compares with
  `region_twins_out_sandbox/`, use the sandbox for seeds 0 and 2 and your Part A output for
  seed 1.
- `region_twins_summary.py` takes the number of draws from the data instead of assuming 200; on
  200 draws its output is the same.
- `fig_twins.py` sets its axis limits from the data. The regenerated figure shows all 6,400
  patches, 318 of which fell outside the fixed axes of the current `fig_twins.pdf`, so the two
  will not look the same.

## 3. Part B: port to `aotil/` (an hour or two)

1. Write `aotil/region_twins.py`, one script with two outputs: the twins and the recurrence.
   - **Flags:** `--seeds`, `--draws` (200), `--excl-km` (5), `--min-size` (20), `--n-strong` (15),
     `--rng-seed` (base 1000), `--out-dir` (required).
   - **Outputs:** the same files and columns as `region_twins_out/`.
   - Rank the strongest models by the stored `auc_id`, as `region_twins_ref.py` does (updated
     2026-10-21).
   - It must reproduce Part A exactly. Compare with the region rows, the pooled summary and
     `recurrence.csv`.
2. Write `aotil/test_region_twins.py` with at least these checks:
   - every twin has the same land cover as its point;
   - every twin is at least 5 km from every point of its set;
   - twins are distinct within a set;
   - `d` equals the mean of the per-pair costs;
   - on a small random instance, the assignment cost equals the brute-force minimum;
   - the region rows equal those in `region_twins_out_sandbox/`.
3. Run it into `results/$D/o5_twins/`.

## 4. Report: `docs/REGION-TWINS-REPORT-$D.md`

Start with one table, "paper numbers". Each row gives the number as the paper states it, the
sandbox value, your value and the file it comes from. The numbers (as the paper stated them on
2026-10-19; the bracketed "now" values, updated 2026-10-21, are what the fixed scripts give on
the same data):

1. **Pooled, 16 regions, 1,114 land points.**
   - d = 0.20 [0.17, 0.23]; the median over pooled Zone AE patches is 0.20.
   - The strongest models' p(flood) is 0.56 on the regions and 0.56 on the twins [now 0.57 and
     0.57: 0.566 / 0.566].
   - Their accuracy is 0.71 on the twins and 0.35 on the regions, so g = +0.36 [0.24, 0.47].
   - The medians over pooled patches are 0.02 (Zone AE) and 0.03 (Zone A). 0 of the 200 pooled
     draws of each kind reach the regions' g.
   - g per split: +0.32, +0.37, +0.38.
   - g with all 60 test models: 0.20 [now 0.19: the pooled `g_all` is 0.1947, 0.643 − 0.448;
     the sandbox has the same value, so 0.20 was mis-rounded].
   - Distance to river, region minus twin, is +0.20 SD, the largest pooled imbalance. Give all
     six features.
2. **By kind** (flood-labeled: 11 regions, 681 points; dry-labeled: 5 regions, 433 points), as
   region / twins:
   - d: .20 (.20) and .20 (.20);
   - p(flood): .46 / .46 and .73 / .72 [now .46 / .47 and .73 / .72];
   - flood-labeled share: .90 / .42 and .19 / .61;
   - accuracy: .41 / .73 and .26 / .67;
   - g: +.32 (median of Zone A patches −.01) and +.41 (.09).
3. **Per region.**
   - In 13 of 16 regions, g is at or above the 95th percentile of the region's Zone A patches.
     Alachua (91st) and Flagler (90th) fall just below, and Bradford has g = +0.03. [Now Alachua
     84.5th and Flagler 90.5th, so "just below" no longer fits Alachua; 13 of 16 is unchanged.]
   - Hamilton: d = 0.32, at the 96.5th percentile of its Zone AE patches.
   - Collier in split 3: d = 0.26, g = +0.06, accuracy on its twins 0.47.
   - Big Bend (split 1, component 0): 49% of its twins are flood-labeled, and accuracy is 40% on
     the region against 77% on its twins.
   - Sarasota (split 2 c1, split 3 c1): 76% and 78% of its twins are flood-labeled.
4. **Recurrence** (r, the split, and the share selected):

   | region | r | split | share selected |
   |---|---|---|---|
   | Bradford | −0.44 and −0.48 (two pieces) | 2 | 0% |
   | Gulf | −0.57 | 3 | 0% |
   | Marion | −0.70 | 1 | 0% |
   | Calhoun | −0.46 | 2 | 25% |

Then:
- every difference from the sandbox values, with its cause;
- the code you added;
- the two figure PDFs you regenerated, and whether they look the same as `fig_twins.pdf` and
  `fig_maps_combined.pdf` in `paper_figures/`.

Do not edit the paper. Tell Victor the date `D` and the report path when you finish or reach a
Stop.
