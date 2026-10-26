# Per-tile shift: run report, D = 2026-10-24

TODO item 3 of `docs/TODO-CLAUDE-CODE.md` ("Report the shift per tile"), decided by Victor on 2026-10-24.
Done by Claude Code on 2026-10-24. Nothing is committed. Outputs are in `results/2026-10-24/`, logs in
`logs/2026-10-24/T*`.

## 0. Summary

- **Code.** `aotil/o1_shift.py` now writes 101 `tile_` columns on every per-cell scope row, and two side
  tables (`o1_shift_tiles.csv`, `o1_shift_families.csv`). Every pre-existing column and value is unchanged,
  checked against `results/2026-10-23/o1_shift.csv` (60 of 60 columns identical as written). Two small new
  scripts: `aotil/o1_tile_null.py` (the null) and `aotil/o1_table.py` (Table 1 as LaTeX).
- **Run.** The per-tile numbers match Claude's sandbox to 4 decimals: accuracy +0.0522 [+0.0332, +0.0720],
  24 of 28 tiles positive, 97.3% of models worse. Even the intervals match: `rng.choice` on the tile names
  and the sandbox's `rng.integers` draw the same tiles from `default_rng(0)`. The only difference is the
  AUROC upper bound (+0.0916 vs +0.0917), from ranking on the rebuilt p as the existing code does. **No Stop
  condition was hit.**
- **Null.** With the zones shuffled within each (tile, label) group, the weighted per-tile accuracy and F1
  gaps stay within 0.006 of 0 (Stop threshold 0.01). Raw F1 reads +0.036 to +0.043, which is the bias the
  weights remove. The pooled accuracy gap reads −0.049 to −0.053 under the null: that is the tile mix alone.
- **Paper.** All nine edits are made, with every draft number checked. **Three sentences of the drafts
  were not true in the data and were reworded** (§6.1): Zone AE does not lie "mostly" in south Florida
  (35%), Zone A does not lie "mostly" in the Panhandle (36%), and "most Zone AE test points lie where every
  model is weak" overstates it. The paper compiles with no undefined references and no new overfull box.
  It has 14 pages and the body ends on page 12, as before.
- **For Victor:** the three rewordings (§6.1; kept by Victor on 2026-10-24), one notation clash ($T$ is both the number of trials and,
  now, the number of test tiles; §8), the 3.1 sentence (§8), and two "aggregate" sentences you may want
  to touch (§7).

## 1. What changed in the code

| File | Change |
|---|---|
| `aotil/o1_shift.py` | New section "per-tile gap". `tile_gaps` computes per test tile and scope the [tile × model] matrices of each half's metric and the gap for five metrics: accuracy and F1 with label weights, AUROC, and raw accuracy and F1. They come from the same TP/FP/FN/TN counts per model and tile that the pooled accuracy uses, so the prediction rule is unchanged (flood iff p > 0.5, a tie counts as dry). AUROC per tile ranks on the rebuilt p (`tile_aucs`, i.e. `weighted_aucs` with unit weights, equal to sklearn to 1e-12). `tile_summary` / `tile_columns` give, per metric: ID, A, gap, ci_lo, ci_hi, worse, tiles_pos, median, wilcoxon_p, n_tiles, n_dropped. The same columns (without the counts) are given for the strongest 15 on the "all" scope. The bootstrap reuses the pooled bootstrap's draws (`tile_draws`, factored out of `o1` with the same RNG calls) and resamples rows of the gap matrix (`draw_matrix`). `main` also writes `<stem>_tiles.csv` and `<stem>_families.csv`. `NEW_PREFIXES` gains `tile_`, so the new columns come last. Pooled rows (several tags) leave `tile_` empty; see §10. The CLI is unchanged. |
| `aotil/o1_tile_null.py` (new) | `shuffle_zones` pools each (tile, label) group's points of both halves and reassigns them at random, keeping the group's count in each half. `null_rows` then computes the `tile_` columns with the same models and draws, and also the pooled accuracy and F1 gaps. |
| `aotil/o1_table.py` (new) | Table 1 as LaTeX, with a pooled block and a per-tile block, from the rows an `o1_shift.py` run prints. Each value is rounded once, from full precision to 3 decimals, as on 2026-10-23. |
| `aotil/test_o1_shift.py` | Checks 7 to 14: the brief's six synthetic cases, the shuffle, and the real cell (§2). |
| `tests/characterization/test_o1.py` (private) | Additions only: the 101 new column names, pins for the two side tables of every `main` case, the new function signatures, and `TILE_METRICS`. |
| `aotil/README.md`, `docs/DECISION-PROVENANCE.md` | The o1_shift.py row and two new rows. Five new provenance rows: the per-tile gap, label weights, interval, pooled rows, and the null. The prevalence reweighting is marked as replaced in the paper (its `adjacc_` columns are kept). |

Definitions as implemented (model i, test tile t; ρ is a half's share of positives in t):
- Weighted accuracy and F1: each ID positive weighs ρ_A(t)/ρ_ID(t), each ID negative (1 − ρ_A(t))/(1 − ρ_ID(t)),
  and Zone A points weigh 1. Accuracy is the weighted share of correct predictions; F1 = 2·TP_w / (2·TP_w + FP_w + FN_w).
- Dropped (NaN, counted in `n_dropped`): every metric when a half has no points in the tile; the weighted
  metrics when Zone A holds a class the ID half lacks; F1 (weighted and raw) when either half has no
  positives (F1 of the flood class is undefined without flood points; this rule is mine, the brief did not
  cover it, and no v3 tile triggers it); AUROC unless both halves hold both classes.
- Point estimate: the mean over models of each model's mean gap over the used tiles. Because a tile is
  dropped for every model or for none, this equals the mean over tiles of the mean-over-models gap.
- Interval: for each of the 2,000 draws, each model's gaps on the drawn tiles are averaged, a tile drawn
  twice counting twice. Drawn tiles where the metric is undefined are skipped. The draw's value is the
  mean over models. The interval is the 2.5th to 97.5th percentile.
- Wilcoxon: `scipy.stats.wilcoxon` (two-sided, scipy defaults) on the 28 (27) mean-over-models tile gaps.

**Characterization goldens.** The old goldens were copied to `Claude outputs/goldens-before-2026-10-24-tile/o1/`
and regenerated. `compare_goldens.py` in that folder is adapted from the 2026-10-23 script: it adds the `tile_`
prefix, allows the two new side-table keys per `main` case, and allows their files to appear in a case's file
list. Its output (`logs/2026-10-24/T0_compare_goldens_o1.log`): **0 problems**. Additions: 1,818 row keys
(18 `o1` rows × 101), 303 CSV columns (3 CSVs × 101), 6 written files, 6 side tables, 852 printed numbers,
and 12 ok lines of the check script. Every old value, array, world file and o5 golden is unchanged. The pins
pass (42 OK, `logs/2026-10-24/T0_char_o1_after_regen.log`).

The bug tests that import `o1_shift` were also run (`tests.bugs.test_followup_test_o1_shift_land`,
`test_followup_o5_pooling_pflood`, `test_stats_01`, `test_stats_02`): 8 of 9 pass. The failure,
`test_stats_01.test_real_offshore_point_not_in_seed0_land`, is the known one from the 2026-10-23 suite: a
v2-era test run on the v3 dataset. It does not involve the new code. The full private suite was not rerun.

## 2. Tests (`aotil/test_o1_shift.py`, all pass)

`logs/2026-10-24/T0_test_o1_shift_v3cell.log` (run with `--cells data/interim/aotil/cells_v3 --draws 3`):

| # | Case | Result |
|---|---|---|
| 7 | One tile: per-tile gap = pooled gap with the same label weights, computed point by point | max diff 8.7e-16 over 5 metrics × 6 models |
| 8 | Two tiles, the ID and Zone A halves identical inside each tile (3:1 and 1:3 copies) | every per-tile gap exactly 0; pooled accuracy gap −0.230 |
| 9 | ID flood share 0.8, Zone A 0.5, same TPR 0.9 and TNR 0.6 | weighted accuracy and F1 gaps 0.0; raw accuracy gap 0.090 (0.84 − 0.75) |
| 10 | A tile whose Zone A half has no negatives, and one with no negatives in either half (like 30_8) | AUROC on 2 of 4 tiles, 2 dropped and counted; accuracy keeps 4 |
| 11 | Bootstrap: every tile drawn once; a draw (T0, T0, T1) | draw = point estimate (diff 0); (2 g_T0 + g_T1)/3 exactly |
| 12 | A class in Zone A that the ID half lacks; and the reverse | weighted accuracy and F1 NaN, dropped and counted, raw kept; the reverse is defined, with weighted ID accuracy = ID TNR |
| 13 | `shuffle_zones` | (tile, label) counts kept per half; each point keeps its label, tile, p and correctness; seeded |
| 14 | Real cell (v3, all points) | 28 tiles; raw accuracy = correctness per tile (diff 0); weighted ID accuracy = ρ_A·TPR + (1 − ρ_A)·TNR (3e-16); AUROC on exactly the 27 tiles with both classes in both halves, = `roc_auc_score` on the rebuilt p (2e-16) |

## 3. Run and expected vs got

```bash
.venv/bin/python aotil/o1_shift.py data/interim/aotil/cells_v3 data/processed_w7/dataset_FL.parquet FL_band_AE_s0 results/2026-10-24/o1_shift.csv --seed 0
```

The run took 6 min 03 s on the laptop (2,000 draws; most of it is the pooled AUROC bootstrap, which is
unchanged; the per-tile part takes about 0.5 s per scope). Log: `logs/2026-10-24/T1_o1_shift.log`. Checks:
`logs/2026-10-24/T2_check_o1.py` → `T2_check_o1.log`.

**Pre-existing columns.** All 60 columns of `results/2026-10-23/o1_shift.csv` are in the new CSV, in the
same order, with identical values as written. The 101 added columns all start with `tile_`.

Scope "all" (land is identical: the land scope drops no point):

| Column | Expected point | Got point | Expected 95% CI | Got 95% CI | Tiles > 0 (exp / got) | Models worse (exp / got) | Median tile gap | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| accuracy, weighted | +0.0522 | **+0.0522** | [+0.0332, +0.0720] | [+0.0332, +0.0720] | 24/28 / **24/28** | 97% / 97.3% | +0.0372 | < 0.0001 |
| F1, weighted | +0.0702 | **+0.0702** | [+0.0449, +0.0976] | [+0.0449, +0.0976] | 23/28 / **23/28** | 95% / 95.3% | +0.0656 | < 0.0001 |
| AUROC (27 tiles) | +0.0599 | **+0.0599** | [+0.0288, +0.0917] | [+0.0288, +0.0916] | 22/27 / **22/27** | 97% / 97.0% | +0.0537 | 0.0008 |
| accuracy, raw | +0.0502 | **+0.0502** | [+0.0262, +0.0788] | [+0.0262, +0.0788] | 22/28 / **22/28** | 92% / 92.0% | +0.0341 | 0.0002 |
| F1, raw | +0.1043 | **+0.1043** | [+0.0663, +0.1490] | [+0.0663, +0.1490] | 24/28 / **24/28** | 99% / 99.3% | +0.0735 | < 0.0001 |

Stop rule: no point estimate differs by more than 0.003 (AUROC 0.005), and no tiles > 0 count differs,
so there is no Stop. Per tile, the accuracy, F1 and raw gaps agree with the sandbox's `a8_per_tile.csv`
to within 2e-15. AUROC per tile agrees to within 0.0026, because the existing code ranks on the rebuilt p
(both classes go through the same float round trip, so tree ties stay ties), while the sandbox ranked on S.

Means of each half over tiles, for Table 1: accuracy ID .758 / Zone A .706; AUROC .809 / .749; F1 .691 / .621.

**The 15 strongest test models**, per tile (scope all): accuracy +0.074 [+0.049, +0.100], all 15 worse, 26/28
tiles positive. F1 +0.088 [+0.058, +0.120]. AUROC +0.073 [+0.036, +0.114]. For comparison, their pooled
accuracy gap is +0.001 [−0.062, +0.063].

**Paired land** (land points in 10 km blocks that hold positives of both zones), per tile: accuracy +0.042
[+0.019, +0.065], 24/28 tiles, 88.7% of models worse. F1 +0.063, AUROC +0.059. Raw accuracy +0.032.

**Named tiles** (scope all; "accuracy" is the 300 models' mean accuracy on both halves of the tile):
30_10 (Palm Beach): 1,264 ID and 192 Zone A points, accuracy 0.5154. Panhandle tiles 16_18 Escambia
0.8283, 18_17 Okaloosa 0.8546, 18_18 Walton 0.8620, 20_18 Jackson 0.8378. All as expected.

**LaTeX table.** `results/2026-10-24/o1_table.tex`, made by
`.venv/bin/python aotil/o1_table.py logs/2026-10-24/T1_o1_shift.log --cell FL_band_AE_s0`. The pooled rows are
identical to `results/2026-10-23/o1_table.tex`.

## 4. The null

```bash
.venv/bin/python aotil/o1_tile_null.py data/interim/aotil/cells_v3 data/processed_w7/dataset_FL.parquet FL_band_AE_s0 results/2026-10-24/o1_tile_null.csv --null-seeds 1 2 3 --seed 0
```

For each seed, the test points in each (tile, label) group are pooled across the two halves and randomly
split again into "Zone A" and "Zone AE", with as many in each as before. Each tile keeps its counts and both
flood shares, and the same 300 models' stored predictions are rescored. Nothing is refit, and it runs in 6 s.
The two pseudo-zones now differ only by chance. Log: `logs/2026-10-24/T3_o1_tile_null.log`.

| Seed | Pooled accuracy gap | Per-tile accuracy (weighted) [95% CI] | Per-tile F1 (weighted) | Per-tile AUROC | Per-tile accuracy, raw | Per-tile F1, raw | Tiles > 0 (accuracy) | Models worse | Strongest 15, accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −0.0522 | +0.0003 [−0.0083, +0.0089] | +0.0060 | −0.0005 | −0.0033 | +0.0426 | 14/28 | 50.7% | −0.0018 |
| 2 | −0.0525 | +0.0002 [−0.0054, +0.0055] | +0.0040 | +0.0181 | −0.0041 | +0.0391 | 12/28 | 42.0% | +0.0023 |
| 3 | −0.0487 | +0.0040 [−0.0028, +0.0118] | +0.0006 | −0.0052 | −0.0008 | +0.0360 | 17/28 | 72.3% | +0.0069 |

(Scope all; land is identical. On paired land, the largest absolute weighted gaps are 0.0047 for accuracy, 0.0087 for F1 and 0.026 for AUROC.)

- **Passes.** The largest absolute weighted gap is 0.0040 for accuracy and 0.0060 for F1 (full precision
  0.00396 and 0.00599), below the Stop threshold of 0.01. AUROC reaches 0.0181. The sandbox's refit null
  gave 0.007 and 0.022.
- Raw per-tile F1 reads **+0.036 to +0.043** with no shift at all, because F1 rises with the share of
  positives, and the halves' shares still differ within a tile under the shuffle. This is the bias the
  weights remove; the sandbox gave +0.035 to +0.038. Raw accuracy shows no such bias (|gap| ≤ 0.004).
- The pooled accuracy gap reads **about −0.05** under every shuffle. That is the tile mix alone, with no
  shift. The real pooled gap (−0.007) is that offset plus a real shift of about +0.05, which is the
  Simpson's paradox in one line.
- Paper wording: "within 0.006 of zero for accuracy and F1, and within 0.019 for AUROC" (bounds rounded
  up from 0.00599 and 0.0181).

## 5. Tables

### 5.1 Per tile (scope all, sorted by the models' accuracy on the tile)

`results/2026-10-24/o1_shift_tiles.csv`; gaps are ID − Zone A, mean over the 300 models; "Acc. gap" and
"F1 gap" are weighted. 30_8 (Monroe) holds only positives, so it has no AUROC.

| Tile | County | n ID | n Zone A | Flood share ID | Flood share Zone A | Accuracy, both halves | Acc. gap | Acc. gap, raw | F1 gap | F1 gap, raw | AUROC gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30_10 | Palm Beach | 1,264 | 192 | 0.5641 | 0.6094 | 0.5154 | +0.0068 | +0.0097 | +0.0122 | −0.0044 | +0.0163 |
| 31_11 | Martin | 643 | 127 | 0.1291 | 0.0551 | 0.5493 | +0.0407 | +0.0460 | −0.0247 | +0.1075 | −0.0876 |
| 26_11 | Manatee | 496 | 93 | 0.4173 | 0.2473 | 0.5787 | +0.0197 | +0.0310 | +0.0624 | +0.1916 | +0.0657 |
| 25_18 | Columbia | 46 | 613 | 0.9565 | 0.9201 | 0.6551 | +0.0276 | +0.0373 | +0.0147 | +0.0252 | −0.0102 |
| 29_10 | Glades | 527 | 817 | 0.6490 | 0.5202 | 0.6611 | +0.0203 | +0.0307 | +0.0024 | +0.0479 | +0.0502 |
| 27_14 | Lake | 517 | 500 | 0.6963 | 0.6580 | 0.6645 | +0.0336 | +0.0188 | +0.0387 | +0.0407 | +0.0537 |
| 29_15 | Volusia | 280 | 381 | 0.4571 | 0.5276 | 0.6681 | +0.0648 | +0.0695 | +0.0915 | +0.0665 | +0.0898 |
| 30_13 | Brevard | 445 | 176 | 0.3730 | 0.4545 | 0.6862 | −0.0021 | −0.0012 | −0.0263 | −0.0660 | −0.0036 |
| 25_15 | Levy | 557 | 748 | 0.9390 | 0.6791 | 0.6890 | +0.1478 | +0.1152 | +0.1672 | +0.1903 | +0.1065 |
| 21_17 | Liberty | 354 | 982 | 0.5311 | 0.5733 | 0.6999 | +0.0860 | +0.0914 | +0.0708 | +0.0617 | +0.0793 |
| 27_17 | St. Johns | 921 | 664 | 0.2237 | 0.2952 | 0.7125 | +0.0693 | +0.0688 | +0.1216 | +0.0598 | +0.1289 |
| 19_16 | Bay | 51 | 41 | 0.9216 | 0.9756 | 0.7320 | +0.1606 | +0.1410 | +0.1307 | +0.1153 | −0.0973 |
| 28_11 | Highlands | 177 | 790 | 0.6271 | 0.6759 | 0.7322 | +0.1020 | +0.1029 | +0.0649 | +0.0568 | +0.1245 |
| 25_17 | Columbia | 315 | 712 | 0.3397 | 0.4677 | 0.7339 | −0.0251 | +0.0378 | −0.0502 | −0.0630 | −0.0751 |
| 24_15 | Dixie | 245 | 41 | 0.8857 | 0.5610 | 0.7514 | +0.1683 | +0.3015 | +0.1822 | +0.3216 | +0.3223 |
| 28_13 | Osceola | 828 | 593 | 0.5809 | 0.4536 | 0.7613 | +0.0148 | +0.0005 | +0.0456 | +0.0771 | +0.0114 |
| 26_14 | Citrus | 931 | 108 | 0.6509 | 0.4074 | 0.7642 | +0.0659 | −0.0053 | +0.1372 | +0.1545 | +0.0779 |
| 20_15 | Gulf | 67 | 13 | 0.9403 | 0.7692 | 0.7689 | −0.0201 | +0.1000 | −0.0156 | +0.0704 | +0.1174 |
| 22_18 | Leon | 37 | 221 | 0.7297 | 0.5068 | 0.7753 | +0.0542 | −0.0072 | +0.0663 | +0.0736 | +0.0329 |
| 29_9 | Collier | 68 | 792 | 0.9265 | 0.9583 | 0.7837 | −0.0154 | −0.0200 | −0.0131 | −0.0194 | +0.0748 |
| 27_19 | Nassau | 107 | 67 | 0.3551 | 0.0746 | 0.7874 | +0.0028 | +0.0267 | +0.0322 | +0.3903 | +0.0445 |
| 19_17 | Washington | 430 | 708 | 0.7814 | 0.4873 | 0.8018 | +0.0711 | +0.0185 | +0.1102 | +0.1270 | +0.0502 |
| 21_18 | Gadsden | 71 | 133 | 0.6620 | 0.2932 | 0.8102 | +0.0208 | −0.0930 | +0.0835 | +0.1166 | +0.0514 |
| 16_18 | Escambia | 169 | 471 | 0.3136 | 0.1062 | 0.8283 | +0.1160 | +0.0737 | +0.2557 | +0.3594 | +0.1824 |
| 20_18 | Jackson | 179 | 724 | 0.3687 | 0.2624 | 0.8378 | +0.0210 | −0.0127 | +0.0472 | +0.0679 | +0.0279 |
| 18_17 | Okaloosa | 322 | 845 | 0.3168 | 0.2178 | 0.8546 | +0.0249 | +0.0136 | +0.1405 | +0.1849 | +0.1174 |
| 18_18 | Walton | 208 | 631 | 0.1298 | 0.2298 | 0.8620 | +0.0511 | +0.0754 | +0.1240 | +0.0734 | +0.0672 |
| 30_8 | Monroe | 580 | 348 | 1.0000 | 1.0000 | 0.9060 | +0.1344 | +0.1344 | +0.0927 | +0.0927 | n/a |

Where the points lie (tile mean latitude; my own split of the 28 tiles, see §6.1): the median latitude
of the Zone AE test points is 28.6°N, of the Zone A points 30.0°N. Tiles below 27.6°N hold 35% of the Zone AE
points and 25% of the Zone A points. Four Panhandle tiles (16_18, 18_17, 18_18, 20_18) hold 21.3% of Zone A
and 8.1% of Zone AE. Tiles where the models average below 0.70 hold 47% of Zone AE and 37% of Zone A. A
tile's latitude goes with its accuracy (Spearman 0.52) and with its Zone A share (0.44).

### 5.2 Per family (scope all)

`results/2026-10-24/o1_shift_families.csv`. Pooled = the paper's ID and OOD columns. Every family does worse
on Zone A per tile.

| Family | Models | Pooled ID | Pooled Zone A | Pooled gap | Per-tile ID | Per-tile Zone A | Per-tile gap [95% CI] | Models worse | Tiles > 0 | Per-tile gap, raw |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Decision tree | 38 | 0.7199 | 0.7303 | −0.0104 | 0.7545 | 0.7083 | +0.0462 [+0.0268, +0.0677] | 94.7% | 24/28 | +0.0403 |
| Extra trees | 38 | 0.7358 | 0.7508 | −0.0150 | 0.7657 | 0.7297 | +0.0360 [+0.0140, +0.0560] | 97.4% | 24/28 | +0.0279 |
| Hist. grad. boosting | 38 | 0.7287 | 0.7413 | −0.0126 | 0.7696 | 0.7154 | +0.0542 [+0.0344, +0.0755] | 100.0% | 24/28 | +0.0479 |
| Logistic regression | 38 | 0.6864 | 0.6705 | +0.0159 | 0.7130 | 0.6587 | +0.0542 [+0.0275, +0.0824] | 94.7% | 22/28 | +0.0510 |
| Multilayer perceptron | 37 | 0.7328 | 0.7381 | −0.0053 | 0.7827 | 0.7141 | +0.0686 [+0.0414, +0.0986] | 94.6% | 22/28 | +0.0674 |
| Naive Bayes | 37 | 0.6917 | 0.7237 | −0.0321 | 0.7197 | 0.6931 | +0.0266 [+0.0004, +0.0510] | 97.3% | 19/28 | +0.0382 |
| Random forest | 37 | 0.7343 | 0.7399 | −0.0056 | 0.7707 | 0.7157 | +0.0550 [+0.0328, +0.0788] | 100.0% | 24/28 | +0.0516 |
| XGBoost | 37 | 0.7455 | 0.7405 | +0.0050 | 0.7923 | 0.7150 | +0.0773 [+0.0529, +0.1026] | 100.0% | 27/28 | +0.0780 |

## 6. Paper edits (`CS371_Cho_Li/acl_latex.tex`)

The file before these edits is kept at `Claude outputs/paper-before-2026-10-24-tile/acl_latex.tex`.
Approach 3.1–3.4 and the subset-search numbers are untouched. No section labels and no new citations were
added. Below, each edit is given as before → after, ready to paste into Overleaf. **New-run numbers** are
listed after each edit. Edits are numbered as in the brief (7a/7b and 8a/8b split items 7 and 8).

### 6.1 Where I departed from the drafts

**Victor kept the three rewordings on 2026-10-24.**

The drafts carried sandbox claims about where the points lie. Three of them are not true in the data:

| Draft | Data | What I wrote |
|---|---|---|
| Intro: "because most Zone AE test points lie where every model is weak" | Only tile 30_10 has every model below 0.70 accuracy, and it holds 12% of Zone AE. Tiles where the models *average* below 0.70 hold 47% of Zone AE (37% of Zone A), so "most" is not true either. | "because more of the Zone AE test points lie where models are weak" |
| Q1: "The Zone AE points lie mostly in south Florida, where all models are weak." | 35% of Zone AE lies below 27.6°N (47% below 28.5°N). Zone AE does lie further south (median 28.6° vs 30.0°), and tile accuracy rises with latitude (Spearman 0.52). | "The Zone AE points lie further south, where the models are weaker." |
| Q1: "The Zone A points lie mostly in the Panhandle, where the models average 0.83 to 0.86." | About 36% of Zone A lies in the Panhandle (by my rough split: west of 84.5°W and north of 29.9°N). Only four Panhandle tiles average 0.83 to 0.86; others are lower (Liberty 0.70, Bay 0.73). | "Four Panhandle tiles hold 21\% of the Zone A points but 8\% of the Zone AE points, and the models average 0.83 to 0.86 there." |

Two more small changes: the null bound is split by metric ("within 0.006 of zero for accuracy and F1, and
within 0.019 for AUROC"), because one bound for all three would have to be the AUROC one; and the 15
strongest models' value carries its interval.

### 6.2 Draft number → my number

| Where | Draft (sandbox) | Mine (this run) | Source |
|---|---|---|---|
| Abstract, Intro, Q1 | models lose 0.052 accuracy | **0.052** (0.0522) | `o1_shift.csv` `tile_acc_gap` |
| Q1 | 0.052 [0.033, 0.072] | **0.052 [0.033, 0.072]** | `tile_acc_ci_lo/hi` |
| Q1 | 24 of the 28 tiles | **24 of 28** | `tile_acc_tiles_pos` |
| Q1 | 97\% of models | **97%** (97.3%) | `tile_acc_worse` |
| Q1 | AUROC +0.060 | **+0.060** (0.0599) | `tile_auc_gap` |
| Q1 | F1 +0.070 | **+0.070** (0.0702) | `tile_f1_gap` |
| Q1 | the 15 strongest lose [your value] | **0.074 [0.049, 0.100]** | `tile_strong15_acc_*` |
| Q1 | within [your bound] of zero | **0.006** (accuracy, F1), **0.019** (AUROC) | `o1_tile_null.csv` |
| Q1 | 1,264 Zone AE / 192 Zone A | **1,264 / 192** | `o1_shift_tiles.csv` 30_10 |
| Q1 | 0.52 at Palm Beach | **0.52** (0.5154) | `acc_both` 30_10 |
| Q1 | 0.83 to 0.86 | **0.83 to 0.86** (0.828 to 0.862) | `acc_both`, four tiles |
| Q1 (new) | — | **21% / 8%** of Zone A / Zone AE in those four tiles | `o1_shift_tiles.csv` |
| Table 1 caption | 28 tiles (27 for AUROC) | **28 (27)** | `tile_*_n_tiles` |
| Table 1 per-tile rows | — | **.758 .706 .052 [.033, .072] 97%; .809 .749 .060 [.029, .092] 97%; .691 .621 .070 [.045, .098] 95%** | `o1_table.tex` |
| Families | within 0.032 (pooled) | **0.032** (naive Bayes −0.0321) | `o1_shift_families.csv` |
| Families | [every family / N of 8] | **every family** | same |
| Families | [smallest, largest] | **0.027 naive Bayes, 0.077 XGBoost** | same |
| Families table, new column | — | **.046 .055 .036 .054 .077 .054 .069 .027** (dt, rf, et, hgb, xgb, lr, mlp, nb; rounded once from full precision) | same |
| Pooled numbers kept | −0.007 [−0.060, 0.045], −0.018, +0.028, 0.85 | unchanged (identical in the new CSV) | `o1_shift.csv` |

The old Q1 paragraph's pooled strongest-15 gap (+0.001 [−0.062, 0.063]) and its prevalence reweighting
(−0.002 [−0.056, 0.063]) are no longer in the paper, since the drafts replace the whole paragraph.

### Edit 1. Abstract

New-run number: 0.052.

Before:

```latex
On 80,000 points across Florida, there is no aggregate shift (an accuracy gap of $-0.007$), but our search finds regions where
```

After:

```latex
On 80,000 points across Florida, the pooled accuracy gap is only $-0.007$, because the test points of the two zones lie in different places. Within the same tiles, models lose 0.052 accuracy on Zone A. Our search finds regions where
```

### Edit 2. Introduction, results paragraph

New-run number: 0.052. Reworded clause: §6.1.

Before:

```latex
First, there is no shift in aggregate. Models are as accurate on Zone A as on held-out Zone AE (a gap of $-0.007$), and accuracy-on-the-line holds with a correlation of 0.85.
```

After:

```latex
First, pooling hides the shift. Pooled, models are as accurate on Zone A as on held-out Zone AE (a gap of $-0.007$), because more of the Zone AE test points lie where models are weak. Within the same tiles, models lose 0.052 accuracy on Zone A, yet accuracy-on-the-line holds with a correlation of 0.85.
```

### Edit 3. Evaluation Metrics, Shift (Q1)

No new numbers. The inline equation would be far wider than a column, so it is a display (`align*`, unnumbered). `\tfrac` keeps its first line inside the column (with `\frac` it overflowed by 3 pt). `\rho` is used for the shares, not `\pi`.

Before:

```latex
Accuracy depends on the share of positives, which differs between the two test sets (Table~\ref{tab:data}). We therefore also reweight Zone A accuracy to the Zone AE share $\pi_\text{AE}$ as $\pi_\text{AE}\,\text{TPR}+(1-\pi_\text{AE})\,\text{TNR}$, where TPR and TNR are the shares of Zone A positives and negatives that a model gets right. Intervals come from 2,000 bootstrap draws of whole test tiles, because points in one tile are not independent. We also report the share of models with a positive gap.
```

After:

```latex
Both test sets cover the same tiles, but in different proportions, so a pooled gap mixes the shift with where the points lie. We therefore also compute the gap within each test tile and average over the $T$ test tiles,
\begin{align*}
\Delta_\text{tile}=\tfrac{1}{N}\sum_i\tfrac{1}{T}\sum_t\big[&\text{metric}(f_i,\mathcal{D}_\text{ID}\cap\mathcal{D}_t)\\
&-\text{metric}(f_i,\mathcal{D}_\text{OOD}\cap\mathcal{D}_t)\big].
\end{align*}
Within a tile, the two sets can still differ in their share of positives, and accuracy and F1 depend on it. F1 in particular rises with the share of positives. We therefore weight the ID positives of tile $t$ by $\rho_t^\text{OOD}/\rho_t^\text{ID}$ and its ID negatives by $(1-\rho_t^\text{OOD})/(1-\rho_t^\text{ID})$, where $\rho_t$ is a set's share of positives in tile $t$. AUROC does not depend on this share, so we compute it without weights on the tiles where both sets hold both classes. Intervals come from 2,000 bootstrap draws of whole test tiles, because points in one tile are not independent. For $\Delta_\text{tile}$, each draw averages the gaps of the drawn tiles. We also report the share of models with a positive gap, and for $\Delta_\text{tile}$ the number of tiles with a positive gap.
```

### Edit 4. Table 1 (tab:shift)

New-run numbers: the three per-tile rows. "Pooled" and "Per tile" block labels added. The caption sentence is appended at the end.

Before:

```latex
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{ID} & \textbf{OOD} & \textbf{Gap [95\% CI]} & \textbf{Worse} \\
\midrule
Accuracy & .722 & .729 & $-$.007 [$-$.060, .045] & 42\% \\
AUROC & .795 & .814 & $-$.018 [$-$.074, .040] & 26\% \\
F1 & .732 & .704 & .028 [$-$.051, .099] & 66\% \\
\bottomrule
\end{tabular}
\caption{The aggregate shift (Q1): the mean of each metric (Equation~\ref{eq:metrics}, on a 0--1 scale) over the 300 models on the ID test set (Zone AE) and the OOD set (Zone A). The gap is ID minus OOD, so a positive gap means worse on Zone A, and its interval resamples whole test tiles. ``Worse'' is the share of models with a positive gap.}
\label{tab:shift}
```

After:

```latex
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{ID} & \textbf{OOD} & \textbf{Gap [95\% CI]} & \textbf{Worse} \\
\midrule
\multicolumn{5}{@{}l}{\textit{Pooled}} \\
Accuracy & .722 & .729 & $-$.007 [$-$.060, .045] & 42\% \\
AUROC & .795 & .814 & $-$.018 [$-$.074, .040] & 26\% \\
F1 & .732 & .704 & .028 [$-$.051, .099] & 66\% \\
\midrule
\multicolumn{5}{@{}l}{\textit{Per tile}} \\
Accuracy & .758 & .706 & .052 [.033, .072] & 97\% \\
AUROC & .809 & .749 & .060 [.029, .092] & 97\% \\
F1 & .691 & .621 & .070 [.045, .098] & 95\% \\
\bottomrule
\end{tabular}
\caption{The aggregate shift (Q1): the mean of each metric (Equation~\ref{eq:metrics}, on a 0--1 scale) over the 300 models on the ID test set (Zone AE) and the OOD set (Zone A). The gap is ID minus OOD, so a positive gap means worse on Zone A, and its interval resamples whole test tiles. ``Worse'' is the share of models with a positive gap. The per-tile rows compute each gap within a test tile and average over the 28 tiles (27 for AUROC), with the ID points of each tile weighted to its OOD share of positives.}
\label{tab:shift}
```

### Edit 5. Results, Q1 paragraph

New-run numbers: 1,264, 192, 0.52, 21%, 8%, 0.83 to 0.86, 24 of 28, 0.052 [0.033, 0.072], 97%, +0.060, +0.070, 0.074 [0.049, 0.100], 0.006, 0.019. Reworded sentences: §6.1.

Before:

```latex
\paragraph{There is no aggregate shift (Q1).} Models trained on Zone AE do as well on Zone A as on held-out Zone AE (Table~\ref{tab:shift}). The accuracy gap is $-0.007$ [$-0.060$, $0.045$], and the AUROC and F1 gaps of $-0.018$ and $+0.028$ have opposite signs and intervals that include zero. The 15 strongest test models show the same, with an accuracy gap of $+0.001$ [$-0.062$, $0.063$]. Reweighting Zone A to the Zone AE flood share moves the accuracy gap only to $-0.002$ [$-0.056$, $0.063$], so the different shares of positives do not hide a gap. Accuracy-on-the-line also holds, with $\text{corr}_\text{te}=+0.85$ on all Zone A points (Figure~\ref{fig:aotl}a). By every aggregate measure, pooling the two zones looks harmless. This is the setting in which aggregation can hide failures \citep{salaudeen2025oodselect}, which is why we search for subsets.
```

After:

```latex
\paragraph{Pooling hides the shift (Q1).} Pooled over the test tiles, models trained on Zone AE do as well on Zone A as on held-out Zone AE (Table~\ref{tab:shift}). The accuracy gap is $-0.007$ [$-0.060$, $0.045$], and the AUROC and F1 gaps of $-0.018$ and $+0.028$ have opposite signs. However, the two test sets do not weight the tiles equally. The Zone AE points lie further south, where the models are weaker. For example, the tile around Palm Beach holds 1,264 Zone AE points and 192 Zone A points, and the models average 0.52 accuracy there. Four Panhandle tiles hold 21\% of the Zone A points but 8\% of the Zone AE points, and the models average 0.83 to 0.86 there. Within a tile, models are less accurate on Zone A in 24 of the 28 tiles. Averaged over tiles, the accuracy gap is 0.052 [0.033, 0.072], and 97\% of models do worse on Zone A. AUROC and F1 agree, with gaps of $+0.060$ and $+0.070$, and the 15 strongest test models lose 0.074 [0.049, 0.100]. When we shuffle which test points belong to each zone within each tile and label, the per-tile gaps stay within 0.006 of zero for accuracy and F1, and within 0.019 for AUROC. Accuracy-on-the-line still holds, with $\text{corr}_\text{te}=+0.85$ on all Zone A points (Figure~\ref{fig:aotl}a). The shift is real, but a pooled measure hides it. This is the setting in which aggregation can hide failures \citep{salaudeen2025oodselect}, which is why we search for subsets.
```

### Edit 6. Results, end of Q3

No numbers.

Before:

```latex
The two procedures look interchangeable in aggregate, yet Zone A holds coherent places where better Zone AE models do worse.
```

After:

```latex
Pooled, the two procedures look interchangeable. Yet Zone A is harder within the same tiles, and it holds coherent places where better Zone AE models do worse.
```

### Edit 7a. Table tab:families

New-run numbers: the Δ_tile column. With a seventh column, the table no longer fits a column; even at `\footnotesize` with 1.5 pt column spacing it was 4 pt too wide (before this change it was already 9 pt too wide at `\small`). So the tabular is wrapped in `\resizebox{\columnwidth}{!}{...}` at `\footnotesize` with 2 pt spacing (scaled about 0.96). This also removes the old overflow. The header is `$\Delta_\text{tile}$`, and the caption gains one sentence that defines it. If you would rather abbreviate family names than scale, drop the `\resizebox` line and its closing `}`.

Before:

```latex
\subsection{Performance Analysis}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Family} & \textbf{Models} & \textbf{ID} & \textbf{OOD} & $\boldsymbol{r}$ & \textbf{Interp.} \\
\midrule
Decision tree & 38 & .720 & .730 & $-$.96 & 0 \\
Random forest & 37 & .734 & .740 & $-$.93 & 12 \\
Extra trees & 38 & .736 & .751 & $-$.94 & 12 \\
Hist.\ grad.\ boosting & 38 & .729 & .741 & $-$.94 & 1 \\
XGBoost & 37 & .745 & .740 & $-$.90 & 6 \\
Logistic regression & 38 & .686 & .670 & $-$.85 & 0 \\
Multilayer perceptron & 37 & .733 & .738 & $-$.96 & 0 \\
Naive Bayes & 37 & .692 & .724 & $-$.66 & 0 \\
\bottomrule
\end{tabular}
\caption{Performance by model family. ID and OOD are the family's mean accuracy on Zone AE and Zone A test points. $r$ is Equation~\ref{eq:corr} on our 500 selected points over all of the family's models, including those that guided the search. Interp.\ counts models with at least 99\% accuracy on their own training rows.}
\label{tab:families}
```

After:

```latex
\subsection{Performance Analysis}

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Family} & \textbf{Models} & \textbf{ID} & \textbf{OOD} & $\boldsymbol{\Delta_\text{tile}}$ & $\boldsymbol{r}$ & \textbf{Interp.} \\
\midrule
Decision tree & 38 & .720 & .730 & .046 & $-$.96 & 0 \\
Random forest & 37 & .734 & .740 & .055 & $-$.93 & 12 \\
Extra trees & 38 & .736 & .751 & .036 & $-$.94 & 12 \\
Hist.\ grad.\ boosting & 38 & .729 & .741 & .054 & $-$.94 & 1 \\
XGBoost & 37 & .745 & .740 & .077 & $-$.90 & 6 \\
Logistic regression & 38 & .686 & .670 & .054 & $-$.85 & 0 \\
Multilayer perceptron & 37 & .733 & .738 & .069 & $-$.96 & 0 \\
Naive Bayes & 37 & .692 & .724 & .027 & $-$.66 & 0 \\
\bottomrule
\end{tabular}}
\caption{Performance by model family. ID and OOD are the family's mean accuracy on Zone AE and Zone A test points. $\Delta_\text{tile}$ is the family's per-tile accuracy gap, ID minus OOD. $r$ is Equation~\ref{eq:corr} on our 500 selected points over all of the family's models, including those that guided the search. Interp.\ counts models with at least 99\% accuracy on their own training rows.}
\label{tab:families}
```

### Edit 7b. Breakdown by model family

New-run numbers: 0.027, 0.077 (and 0.032, unchanged). The first sentence is kept as instructed; see §7.

Before:

```latex
\paragraph{Breakdown by model family.} Every family shows the aggregate picture (Table~\ref{tab:families}). Its Zone A accuracy lies within 0.032 of its ID accuracy, and only logistic regression and XGBoost do slightly worse on Zone A.
```

After:

```latex
\paragraph{Breakdown by model family.} Every family shows the aggregate picture (Table~\ref{tab:families}). Pooled, its Zone A accuracy lies within 0.032 of its ID accuracy. Per tile, every family does worse on Zone A, from 0.027 for naive Bayes to 0.077 for XGBoost.
```

### Edit 8a. Development (first sentence)

No numbers.

Before:

```latex
Seven decisions shaped these experiments.
```

After:

```latex
Eight decisions shaped these experiments.
```

### Edit 8b. Development (end of paragraph)

No numbers.

Before:

```latex
Seventh, we first measured the shift in AUROC. Since OODSelect works with accuracy and the three metrics can disagree in sign, we made accuracy the primary metric before computing any result on the rebuilt data.
```

After:

```latex
Seventh, we first measured the shift in AUROC. Since OODSelect works with accuracy and the three metrics can disagree in sign, we made accuracy the primary metric before computing any result on the rebuilt data. Eighth, the pooled gap was near zero on the rebuilt data. Since the two test sets hold the tiles in different proportions, we added the per-tile gap after seeing this result.
```

## 7. Grep list

Searched for "aggregate", "no shift", "harmless", "interchangeable" and "0.007". Before the edits there
were 11 / 1 / 1 / 1 / 3 hits; after, 6 / 0 / 0 / 1 / 3.

| Line (after) | Hit | What I did |
|---|---|---|
| — (was 38) | Abstract "there is no aggregate shift (... $-0.007$)" | replaced (edit 1) |
| 38 | Abstract "pooled accuracy gap is only $-0.007$" | new text, kept |
| 48 | Intro "This trend can hold in aggregate and still invert on subsets" | kept: it is about accuracy-on-the-line, not the shift |
| — (was 50) | Intro "there is no shift in aggregate" (hits "aggregate" and "no shift") | replaced (edit 2) |
| 50 | Intro "(a gap of $-0.007$)" | kept inside the new text |
| 160 | Q1 question "lose performance on Zone A in aggregate?" | left unchanged, as instructed |
| 258 | Table 1 caption "The aggregate shift (Q1): ..." | kept; the caption's first sentence still describes the pooled rows. You may want "The shift (Q1): ...", since the table now holds both. |
| — (was 291) | old Q1 paragraph (title, "By every aggregate measure", "harmless") | replaced (edit 5) |
| 302 | Q1 "The accuracy gap is $-0.007$ [...]" | kept inside the new text |
| 304 | Q2 "Random subsets follow the aggregate trend ($+0.84$)" | kept: accuracy-on-the-line on all of Zone A |
| 306 | Q3 "Pooled, the two procedures look interchangeable." | new text (edit 6) |
| 358 | Families "Every family shows the aggregate picture" | kept, as instructed ("keep the rest"). It now reads ambiguously; "Every family shows the same picture" would fit both sentences after it. |
| 414 | Limitations "the same pattern of a small aggregate shift" (earlier data) | left unchanged, as instructed. On the earlier data that was the pooled gap. |

## 8. Approach 3.1, and a notation clash

**3.1.** It ends "The shift is the mean of $\text{metric}(f_i,\mathcal{D}_\text{ID})-\text{metric}(f_i,\mathcal{D}_\text{OOD})$
over the $N$ models." Results now says "The shift is real, but a pooled measure hides it", so the paper
treats the per-tile gap as the better measure of the same shift. 3.1 still defines only the pooled one. Not a
contradiction (Evaluation Metrics says "also"), but a reader may stumble on it. Proposed replacement, one sentence, not applied:

```latex
The shift is the mean of $\text{metric}(f_i,\mathcal{D}_\text{ID})-\text{metric}(f_i,\mathcal{D}_\text{OOD})$ over the $N$ models, computed over all test points or within each test tile and then averaged over the tiles.
```

**$T$.** The draft's "average over the $T$ test tiles" reuses $T$, which 3.3 and Experimental Settings use
for the number of search trials ("each search runs $T=100$ trials"). I kept the draft. If you want to avoid
the clash, 3.1 already names the tiles $\mathcal{D}_1,\dots,\mathcal{D}_n$ with $n_\text{train}$ of them for
training. So for example "average over the $n_\text{te}$ test tiles" with $\tfrac{1}{n_\text{te}}$ in the equation
would work, or write the average over $t$ without a symbol for the count.

## 9. Compile and page count

Compiled with `latexmk -pdf` in a scratch copy, before and after the edits.
- No undefined references or citations. The only BibTeX warning is the old empty year in `fema_sg_unit3`.
- Overfull boxes: before, 2 (`tab:search` 11.6 pt and `tab:families` 9.0 pt); after, 1 (`tab:search`
  11.6 pt, unchanged). **No new overfull box**, and the families one is gone.
- **14 pages**, as before. The body (through Team Member Contributions) ends on **page 12** before and after,
  so the paper is about 4 pages over the 8-page limit both before and after this change. The added text
  (about 390 words of LaTeX source, net) did not move the end of the body.
- Pages 7 to 10 were rendered and checked by eye: the display equation, Table 2 with both blocks, the Q1
  paragraph and the families table look right.

## 10. Other notes

- **Pooled rows.** With several tags, the `cell = pooled` rows leave the `tile_` columns empty. The paper
  uses one split. How to pool per-tile statistics over seeds is open: tiles differ by seed, and tiles > 0
  and the Wilcoxon test have no obvious pooled form. If you add splits, this needs a decision.
- **Runtime.** The brief says "everything runs in seconds". The per-tile part does (about 0.5 s per scope,
  6 s for the null), but a full `o1_shift.py` run takes about 6 minutes, because of the pooled AUROC
  bootstrap that already existed.
- **County** in the per-tile table is the most common county among the tile's test points. A point's county
  is the one it lies in (`meta_point_county`), else its flood map's (`county_fips`), as `regions.py` does.
  Names come from `data/interim/gis/counties_FL.parquet`.
- The provenance "Where" line numbers for `o1_shift.py` in rows written before 2026-10-23 were already stale.
  I did not renumber them; the new rows point at the current lines.

## 11. GIS terms in CS terms

- **Tile (50 km tile, `tile_50km`).** A cell of a fixed 50 km × 50 km grid laid over Florida. Each point
  belongs to one tile. Tiles are the unit of the train/test split and of the bootstrap: clusters of
  correlated samples, because nearby points share rivers, maps and survey studies.
- **Zone AE / Zone A.** Two ways FEMA draws the "1% annual flood chance" area. AE is a detailed engineering
  study; A is an approximate one. Here they are two labeling procedures, i.e. two domains: models train on
  AE labels (ID) and are tested on A labels (OOD) in the same test tiles.
- **Flood share (ρ).** The share of positives (label 1 = inside the mapped flood zone) in a set: the class prior.
- **County, FIPS code.** A county is an administrative region (Florida has 67); its FIPS code is a numeric ID
  such as 12099 for Palm Beach. Here it is only a readable name for a tile.
- **Panhandle.** The narrow northwest strip of Florida (Pensacola to around Tallahassee), at about 30 to 31°N.
  "South Florida" here means roughly south of 27.6°N (Lake Okeechobee and below). These splits are informal;
  the paper states only counts and shares that do not depend on them.
- **Land, open water.** NLCD is a 30 m land-cover raster (a class code per pixel); class 11 is open water. A
  "land" point has another class. **Paired land**: land points in 10 km blocks (a finer grid) that hold
  positives of both zones, a scope where both procedures mapped floodplain nearby.

## 12. Files

- Code: `aotil/o1_shift.py`, `aotil/o1_tile_null.py` (new), `aotil/o1_table.py` (new), `aotil/test_o1_shift.py`,
  `tests/characterization/test_o1.py` and `tests/characterization/golden/o1/` (regenerated; additions only).
- Results: `results/2026-10-24/o1_shift.csv`, `o1_shift_tiles.csv`, `o1_shift_families.csv`, `o1_table.tex`,
  `o1_tile_null.csv`.
- Logs: `logs/2026-10-24/T0_test_o1_shift_v3cell.log`, `T0_compare_goldens_o1.log`, `T0_char_o1_after_regen.log`,
  `T1_o1_shift.log`, `T2_check_o1.py`, `T2_check_o1.log`, `T2_o1_table.log` (empty: no warnings), `T3_o1_tile_null.log`.
- Backups: `Claude outputs/goldens-before-2026-10-24-tile/` (old goldens and the comparison script),
  `Claude outputs/paper-before-2026-10-24-tile/acl_latex.tex`.
- Docs: `aotil/README.md`, `docs/DECISION-PROVENANCE.md`, `docs/TODO-CLAUDE-CODE.md` (item 3 moved to Done).
