# Rebuild v3 (land mask) and the paper's experiment on split 1: report, D = 2026-10-23

Brief: `docs/TASK-REBUILD-V3.md`, with Victor's amendment of 2026-10-23 that reports the shift
(C1) in accuracy, AUROC and F1. Outputs: `results/2026-10-23/`. Logs: `logs/2026-10-23/`. Earlier
results (`results/2026-10-14/` to `results/2026-10-19/`), the cell folders `cells/` and
`cells_w7/` and `data/interim/archive/sampling-v1/` are untouched. Nothing is committed.

**Status: done.** Parts A–D ran in order, plus the amendment. One Stop fired: Earth Engine asked for
authentication at A1, and Victor re-authenticated (project `cs371w-gis`). No other Stop condition
was met; each is listed where it applies.

**In one paragraph.** All six claims fixed in §2 hold on split 1 (C1 is reported, not tested). No
point of the rebuilt data lies on open water, and Earth Engine reads the same land cover as the mask
at all 80,000 points. On the new data the aggregate shift is gone: Zone A accuracy is 0.007 *higher*
than held-out Zone AE (−0.007 [−0.060, +0.045]); the AUROC gap (−0.018) and the F1 gap (+0.028)
point in opposite directions, and all three intervals include 0.
Accuracy-on-the-line holds (+0.85). Classic OODSelect inverts more strongly than before (−0.96) and
stays scattered; our penalty keeps a strong inversion (−0.89) in 20 pieces with purity 0.86. The
inversion survives the shuffled-accuracy control, held-out non-interpolating models (−0.89) and
both family-disjoint directions. The 7 regions are new places (none within 2 km of an old split-1
region, partly because split 1 now holds different test tiles), and they show the same concept-shift
pattern as before: inputs as close to their twins as ordinary Zone AE land (d at the 4.5th
percentile), accuracy gap far above every reference patch (g = +0.34). What changes in the paper's
story: the shift sentence (no aggregate gap at all on this split), the hardest-points baseline (now
strongly inverted, −0.68), the smoothing ablation (it loses the inversion outright), the region mix
(4 of 7 regions are dry-labeled and both kinds invert), and every number that needed three splits.

---

## Terms used here, in CS terms

- **Raster:** a 2-D array of values plus an affine transform that maps (row, column) to map
  coordinates. A **pixel** is one cell; NLCD's are 30 m × 30 m.
- **CRS (coordinate reference system), EPSG code:** the coordinate system coordinates are in.
  EPSG:5070 is an equal-area projection of the lower 48 states, in metres, on the NAD83 datum.
  NLCD's own CRS is the same projection on the WGS 84 datum, a nearly identical model of the
  Earth; pyproj maps one to the other with a zero shift here.
- **Native grid:** a raster's own pixel lattice. Reading "on the native grid" means no resampling,
  so the value read is exactly one stored pixel.
- **NLCD:** the National Land Cover Database, one land-cover class per pixel. Class 11 is open
  water. In the mask, 0 means no data (open ocean, outside NLCD).
- **NFHL, Zone A, Zone AE, SFHA:** FEMA's flood map as polygons. SFHA is the 1%-annual-chance
  floodplain; Zone AE is mapped by a detailed study, Zone A by an approximate one. NFHL also has
  polygons of zone **OPEN WATER** (mapped lakes, bays and rivers).
- **Positive:** a point inside a Zone A or AE polygon (flood-labeled). **Band negative:** a point
  0.5–5 km from every floodplain polygon, inside a FIRM panel (a printed map sheet) and inside
  Florida (dry-labeled).
- **Land mask:** the lookup "is this point's NLCD pixel a class other than 0 and 11?".
- **Tile:** a 50 km × 50 km grid square. Train and test sets are whole tiles, so nearby points
  never straddle the split.
- **HAND (`hand`):** height above the nearest drainage channel. **`wetindexcat`:** the mean
  topographic wetness index of the point's NHDPlus catchment, the patch of land that drains to one
  stream segment.

---

## 1. Claims fixed in §2 of the brief

Split 1 (seed 0), 300 models, 60 test models, S = 500 unless stated. "r" is the paper's probit
correlation between ID accuracy and accuracy on the points (`oodselect_paper.hard_corr`).

| # | Claim | Statistic | Value | Holds? |
|---|---|---|---|---|
| C1 | The shift is small in aggregate | Mean over 300 models of ID minus Zone A, all points (here all land); 95% intervals from the same 2,000 draws of whole test tiles. Accuracy is primary (amended, see §6.1) | **accuracy −0.007 [−0.060, +0.045]**, worse in 42% of models; AUROC −0.018 [−0.074, +0.040], 26%; F1 +0.028 [−0.051, +0.099], 66%. Strongest 15: accuracy +0.001 [−0.062, +0.063], AUROC −0.001 [−0.055, +0.049], F1 +0.024 [−0.062, +0.100]. Prevalence check: flood shares 0.514 (Zone A) and 0.547 (Zone AE); Zone A accuracy reweighted to the Zone AE share gives a gap of −0.002 [−0.056, +0.063] | Reported, as fixed: no aggregate shift on this split in any metric; every interval spans 0. Secondary, paired land: accuracy +0.017 [−0.019, +0.053], AUROC +0.014 [−0.026, +0.053], F1 +0.084 [+0.036, +0.126] |
| C2 | Accuracy-on-the-line holds in aggregate | r on all Zone A test points, 60 test models | **+0.849** [+0.758, +0.907] | **Holds** (> +0.3) |
| C3 | OODSelect inverts, but scattered | Classic S = 500 r_te; pieces, tiles, purity; the 3 shuffled-classic runs at S = 500 | **−0.962** [−0.977, −0.936]; 298 pieces over 27 of 28 test tiles; purity 0.114. Shuffled: −0.890, −0.485, +0.366 | **Holds** (below −0.3 and below all three; the closest shuffled run is 0.072 above) |
| C4 | The penalty makes regions at a small cost | Spatial (μ = 3, k = 10) S = 500 r_te, purity, pieces; smoothed-classic purity at S = 250, 500, 1,000 | **r_te −0.886** [−0.931, −0.816]; purity **0.864**; **20 pieces** (classic/5 = 59.6). Purity spatial vs smoothed-classic: 0.695 vs 0.672, 0.864 vs 0.717, 0.865 vs 0.709 | **Holds** on all four conditions (the narrowest is purity at S = 250) |
| C5 | Not memorized label noise | r on the held-out test models with `acc_train_sub` < 0.99, spatial S = 500 selection | **−0.886** [−0.934, −0.808], 51 models. Beside it, the family-disjoint searches at S = 500: spatial trees→others −0.652, others→trees −0.795 (classic −0.868 and −0.869) | **Holds** (< −0.3) |
| C6 | Same inputs, different labels | Over the 7 regions with land points: pooled d and g (strongest 15 test models) against the pooled reference patches | **g = +0.336** [0.139, 0.502], above all 200 draws of both patch kinds (100th percentile); **d = 0.119** [0.084, 0.175], 4.5th percentile of the Zone AE patches | **Holds**: g above the 95th percentile of both kinds; d not above the 95th percentile of the Zone AE patches |

Files: `results/2026-10-23/o1_shift.csv` (C1, C2), `oodselect_paper/FL_band_AE_s0_selection_audit.md`
and `logs/2026-10-23/oodselect_paper/report_local_FL_band_AE_s0.out` (C3, C4),
`selection_audit/{null,splits,mu_sweep}/` (controls), `o5/A_interpolation.csv` (C5),
`paper_figures/region_twins_out/pooled_summary.json` (C6).

The §2 Stop (fewer than 3 regions of at least 20 land points) did not fire: 7 regions.

---

## 2. The paper's numbers, old and new

`CS371_Cho_Li/acl_latex.tex` from Experiments to Discussion, one row per number. "Split 1" is seed 0
of the 2026-10-16 run (and of the 2026-10-17 and 2026-10-19 analyses built on it), given where the
paper gives per-split values or ranges. **Gone with one split** marks numbers that exist only because
of several splits (pooled rows, ranges over splits, recurrence). Design settings that did not change
(300 models, eight families, the 60/20/20 model split, k = 10, μ = 3, 100 Optuna trials, 50,000 steps,
patience 500, m = 20, 5 km twin exclusion, 200 + 200 patches, 1,000 resamples) are not repeated.

Two cautions for every row. Split 1 now holds different test tiles (§3.6). And the paper's Hardest row
and purities used the definitions from before the 2026-10-21 review; on the old data the review's
definitions give Hardest −.21 and purities .09 (classic) and .76 (ours) for split 1. The new values
use the review's definitions.

**Data and Splits**

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| tiles that hold points | 102 | 102 | **98** | `logs/2026-10-23/B2_merge_gee_FL.log` |
| test tiles kept (hold both zones) | 23 to 25 | 24 | **28** | `logs/2026-10-23/C1_audit_cell.log` |
| training tiles | 71 | 71 | **69** | same |
| training points | 25,670 to 29,395 | 28,109 | **29,161** | same |
| ID test points | 10,235 to 11,426 | 10,235 | **10,835** | same |
| OOD points | 12,771 to 15,036 | 12,771 | **12,531** | same |
| flood-labeled share of the test halves | 40 to 56% | 40% (ID), 48% (OOD) | **55% (ID), 51% (OOD)** | same |
| Table 1, split 1: ID / OOD / gap [CI] / worse | .829 / .826 / .004 [−.050, .060] / 56% | same | AUROC **.795 / .814 / −.018 [−.074, .040] / 26%**; accuracy and F1 in §1, LaTeX in `results/2026-10-23/o1_table.tex` | `results/2026-10-23/o1_shift.csv` |
| Table 1, splits 2 and 3, and pooled .836 / .790 / .046 [.004, .094] / 82% | | | **gone with one split** | |

**Models and Search**

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| S = 500 is this share of the OOD points | 3 to 4% | 3.9% | **4.0%** | 500 / 12,531 |
| a split's 300 models train in | about 16 s | 16 s | **17 s** | `logs/2026-10-23/oodselect_paper/zoo_FL_band_AE_s0_local.out` |
| its searches take | 23 to 45 min | 23 min | **31 min** (8 jobs) | `logs/2026-10-23/C3_selections.log` |

**Evaluation Metrics**

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| purity of random points | about 0.035 | 0.039 (500/12,771) | **0.040** (500/12,531) | expected share S/N |

**Development.** "so we report three splits": **gone with one split**. The 0.07 machine-to-machine
change and the "below 0.1" reading rule are not recomputed.

**Results**

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| Figure 1 caption: "in split 2" | split 2 | — | **gone with one split**; redrawn on split 1: r = +0.85 (a), −0.89 (b) | `results/2026-10-23/paper_figures/fig_aotl.pdf` |
| Table 2, All Zone A | +.73 | +.73 | **+.85** | `report_local_FL_band_AE_s0.out` |
| Table 2, Random | +.71 | +.71 | **+.84** | same |
| Table 2, Hardest | −.13 | −.13 (review rule: −.21) | **−.68** | same |
| Table 2, OODSelect (μ = 0) | −.89 (.08) | −.89 (.08) | **−.96 (.11)** | same |
| Table 2, Ours (μ = 3) | −.82 (.75) | −.82 (.75) | **−.89 (.86)** | same |
| Table 2, Smoothing after search | −.86 (.53) | −.86 (.53) | **−.22 (.72)** | `oodselect_paper/FL_band_AE_s0_selection_audit.md` |
| Table 2, Shuffled, OODSelect (median of 3) | +.20 | +.20 | **−.49** (runs −.89, −.49, +.37) | `selection_audit/null/` |
| Table 2, Shuffled, ours (median of 3) | +.59 | +.59 | **+.59** (runs −.69, +.90, +.59) | same |
| Table 2, splits 2 and 3 | | | **gone with one split** | |
| Figure 3 (c): "The 18 regions ... from all three splits" | 18 | 6 | **7**, split 1 only | `results/2026-10-23/o5/regions_land.csv` |
| Figure 3: "Two regions on the open water of Lake Okeechobee are left out" | 2 | 0 | **0**: no point is on water now | |
| "Pooled over splits, ... lose 0.046 AUROC on Zone A" | 0.046 | | **gone with one split**; split 1 now −0.018 AUROC, −0.007 accuracy | `o1_shift.csv` |
| "near zero in split 1 and about 0.07 in splits 2 and 3" | ≈0 / 0.07 | 0.004 | **−0.018** (AUROC), −0.007 (accuracy); splits 2–3 gone | same |
| "r between 0.73 and 0.87" | 0.73 to 0.87 | 0.73 | **0.85**; the range is gone | same |
| OODSelect "r_te between −0.89 and −0.95" | −0.89 to −0.95 | −0.89 | **−0.96** | Table 2 |
| "301 to 309 pieces across 21 to 24 test tiles" | 301–309, 21–24 | 309, 24 of 24 | **298 pieces, 27 of 28 tiles** | `selection_audit.md` |
| "a purity of 0.08 to 0.10" | 0.08–0.10 | 0.08 | **0.11** | same |
| "about 9 unselected points among its 10 nearest neighbors" | about 9 | about 9 | **about 9** (8.9) | 1 − purity |
| Ours "r_te between −0.80 and −0.84" | −0.80 to −0.84 | −0.82 | **−0.89** | Table 2 |
| "cuts the pieces to 20 to 29" | 20–29 | 20 | **20** | `selection_audit.md` |
| "raises purity to 0.75 to 0.81" | 0.75–0.81 | 0.75 | **0.86** | same |
| "touches 5 to 15 test tiles" | 5–15 | 5 | **11** | same |
| "The cost of 0.07 to 0.13 in r_te" | 0.07–0.13 | 0.07 | **0.08** (0.076) | −0.886 vs −0.962 |
| "far less than the gap between OODSelect and the hardest points" | | 0.76 | **0.28**: still larger than the cost, but no longer "far" | −0.962 vs −0.682 |

**Performance Analysis**

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| μ sweep, purity at μ = 0 / 1 / 10 (S = 500) | 0.08 / 0.49 / 0.91 | 0.08 / 0.49 / 0.91 | **0.11 / 0.78 / 0.89** (μ = 3: 0.86) | `selection_audit/mu_sweep/` |
| "while r_te stays between −0.82 and −0.89" | −0.82 to −0.89 | −0.82 to −0.89 | **−0.85 to −0.96** | same |
| Smoothing: lower purity than ours "in all nine settings of split and budget" | 9 of 9 | 3 of 3 | **3 of 3**; nine settings **gone with one split** | `selection_audit.md` |
| Smoothing: "a weaker inversion in eight" | 8 of 9 | 2 of 3 | **3 of 3** (−.31, −.22, +.13 against −.89, −.89, −.88) | same |
| "In split 3 it loses most of the inversion (−0.26 against −0.80)" | | | **gone with one split**; on split 1 it now loses it too (−0.22 against −0.89) | same |
| "The real search beats all 54 shuffled runs" | 54 | 18 | **all 18** (one split) | `selection_audit/null/` |
| shuffled medians, OODSelect / ours | +0.10 / +0.27 | +0.10 / +0.59 | **+0.16 / +0.59** | same |
| "9 of 27 OODSelect runs fall below −0.3" | 9 of 27 | 2 of 9 | **4 of 9** (and 3 of 9 for ours) | same |
| models reaching 99% training accuracy | 26 to 29 per split | 29 | **31**, all tree ensembles (et 12, rf 12, xgb 6, hgb 1); 9 of the 60 test models | `logs/2026-10-23/C6_A_interpolation.log` |
| "Without them, r on our selected points is −0.82 [−0.84, −0.79]" (pooled, circular) | −0.82 | −0.82 circular; −0.82 [−0.89, −0.70] held out | pooled: **gone with one split**. Split 1: **−0.92** [−0.93, −0.89] circular (269 models); **−0.89** [−0.93, −0.81] held out (51) | `o5/A_interpolation.csv` |
| families: "from −0.52 for naive Bayes to −0.86 for random forests" (all of a family's models, circular) | −0.52 to −0.86 | −0.57 (nb) to −0.83 (lr) | **−0.66 (nb) to −0.96 (dt)**, circular | same |
| "the four families that never interpolate all invert" | yes | yes | **yes**: dt −0.96, lr −0.85, mlp −0.96, nb −0.66 | same |

**Error Analysis.** Table 3 on the new data is in §5.

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| regions with land points | 16 | 6 | **7** | `regions_land.csv` |
| flood-labeled / dry-labeled regions | 11 / 5 | 6 / 0 | **3 / 4** | same |
| Table 3, regions (land points) | 11 (681) / 5 (433) | 6 (418) / 0 | **3 (219) / 4 (207)** | same |
| "The inversion lives in the flood-labeled regions (r = −0.70)" | −0.70 | −0.80 | **−0.45** flood / **−0.47** dry (held out, non-interpolating; 60 test models: −0.49 / −0.55): both kinds invert | `A_interpolation.csv` |
| "The dry-labeled regions invert within seven of the eight model families, but not across them" | 7 of 8 | no dry region | across the test models **−0.55**; within all 8 families (nb −0.20, interval spans 0) | same |
| Figure 4 caption: "the 16 regions with land points" | 16 | 6 | **7** | `fig_twins.pdf` |
| d pooled | 0.20 [0.17, 0.23] | 0.20 | **0.12** [0.08, 0.18] | `pooled_summary.json` |
| "the median for random Zone AE patches" | median | | **4.5th percentile** (median patch 0.17) | same |
| p(flood) regions and twins | 0.56 | .46 / .45 | **.52 / .53** | same |
| "accuracy is 0.71 on the twins and 0.35 on the regions" | .71 / .35 | .75 / .42 | **.69 / .35** | same |
| g pooled | +0.36 [0.24, 0.47] | +0.32 | **+0.34** [0.14, 0.50] | same |
| "exceeds all 200 pooled patches of either kind (medians 0.02 and 0.03)" | all 200 | | **all 200** (medians −0.005 and −0.04) | same |
| "The gap holds in each split (+0.32, +0.37, +0.38)" | | +0.32 | **+0.34**; the other splits **gone with one split** | same |
| "and for both kinds of region" | yes | flood only | **yes**: +0.25 flood, +0.43 dry | same |
| flood-labeled regions: Zone A floodplain / twins' Zone AE floodplain | 90% / 42% | 100% / 44% | **67% / 70%** (Levy and Dixie: 83% / 42%) | same |
| dry-labeled regions: the same shares | 19% / 61% | | **10% / 74%** | same |
| "In 13 of the 16 regions, g is at or above the 95th percentile of the region's own Zone A patches" | 13 of 16 | 5 of 6 | **7 of 7** | `regions_summary.csv` |
| "Alachua and Flagler fall just below" | | | **gone** (split-2 regions) | |
| "in Bradford ... (g = +0.03)" | +0.03 | +0.03 | no Bradford region; the nearest analogue is **Hendry/Palm Beach, g = +0.04** (68th percentile of its Zone AE patches) | same |
| Hamilton "farther from its twins than 96% of Zone AE patches (d = 0.32)" | 96%, 0.32 | 96.5%, 0.32 | **95.0%, 0.29** (a new Hamilton region, 4.6 km from the old one) | same |
| "Collier in split 3 ... (d = 0.26) ... (+0.06)" | | | **gone with one split** | |
| Case study: largest flood-labeled region, Big Bend, 182 points, evergreen forest 42%, HAND 1.5 m, twins 49% floodplain, accuracy 40% vs 77% | | same region | no Big Bend region. Largest flood-labeled: **Hendry/Palm Beach**, 114 points, half dry-labeled, cultivated crops 94%, HAND 0.4 m, twins 96% floodplain, accuracy 50% vs 54%. Largest all-flood region: **Levy**, 63 points, evergreen forest 54%, HAND 0.3 m, twins 29% floodplain, accuracy 25% vs 82% | `C5_regions_describe.log`, `regions_summary.csv` |
| Case study: Sarasota, "found in two splits", 76 to 78% of its twins floodplain | | | **gone** (a split-2/3 region; recurrence needs several splits). Clearest dry-labeled case now: **Liberty**, 107 points, 2% flood-labeled, woody wetlands 59%, 0.7 km from a mapped river, HAND 1.7 m, 79% of its twins Zone AE floodplain, accuracy 31% vs 85% | same |

**Discussion (Limitations)**

| Paper text | Paper | Split 1, old | New | Source |
|---|---|---|---|---|
| "our claims rest on three splits and on controls" | | | **gone with one split** | |
| "With all 60 test models ... the gap shrinks to 0.20" (review: 0.19) | 0.20 | 0.08 | **0.22** | `pooled_summary.json` |
| "region points 0.20 standard deviations farther from rivers than their twins" | 0.20 | 0.04 | **0.04**; the largest imbalance is now coast distance, 0.08 | same |
| "only two places recur across splits" | 2 | | **gone with one split** | |
| "on the land of four other places, the inversion persists ... (r of −0.44 to −0.70)" | | | **gone with one split** | |

---

## 3. The sample

### 3.1 What the mask and the OPEN WATER test dropped

From `logs/2026-10-23/B1_sample_points_FL.log`. Candidates are proposals before the 100 m thinning;
each count is of candidates that passed every earlier test of the chain.

| Zone | Label | Candidates | In NFHL OPEN WATER | Not on land per the mask | Share dropped by the two tests |
|---|---|---:|---:|---:|---:|
| A | flood (positive) | 60,000 placed | — | 1,789 | 3.0% |
| A | dry (band) | 370,734 drawn; 32,412 left after the distance, panel and Zone D tests | 565 | 1,309 | 5.8% of those 32,412 |
| AE | flood (positive) | 60,000 placed | — | 8,828 | 14.7% |
| AE | dry (band) | 370,785 drawn; 52,008 left after the distance, panel and Zone D tests | 8,503 | 11,287 | 38.1% of those 52,008 |

The positive test runs on the placed candidates, so it is a straight share of an area-uniform draw:
3.0% of Zone A floodplain area and 14.7% of Zone AE floodplain area is open water or no data. The
shares are consistent with the brief's v2 measurement (on NLCD open water: 35% of Zone AE band
negatives and 16% of Zone AE positives; 5% and 3% for Zone A). Most band candidates fail the 500 m
test (334,705 for A, 311,259 for AE), because the rings are drawn around the floodplain itself.
The 1,844 (A) and 652 (AE) candidates outside the Florida county polygons are counted after the
two new tests.

### 3.2 Supply margins

| Stratum | Kept after thinning | Needed (20,000 + 3%) | Margin |
|---|---:|---:|---:|
| A positives | about 56,500 (97% of 58,211) | 20,600 | 2.7× |
| AE positives | about 49,600 (97% of 51,172) | 20,600 | 2.4× |
| A band | 27,373 | 20,600 | 1.33× |
| AE band | 29,869 | 20,600 | 1.45× |

`--ring-oversample 6` was enough, as the brief expected; the review's scratch build without the mask
had 28,375 (A) and 47,099 (AE) band candidates after thinning. The whole-state 100 m pass dropped
264 A band, 1 A positive, 250 AE band and 0 AE positives, inside the 3% surplus. The build wrote
80,000 points in 1,499 s.

### 3.3 Small polygons

Positives from polygons under 1 km² (`meta_poly_area_km2`):

| | v2 (2026-10-14) | v3 |
|---|---:|---:|
| Zone A | 0.6% (123 of 20,000) | **16.1%** (3,212) |
| Zone AE | 0.6% (120) | **14.2%** (2,846) |

The brief gives 0.7% for v2; I count 123 of 20,000, 0.62%. The jump is the review's fix of positives
weighted by area twice (bug `sampling_01`): polygons under 1 km² hold about 14% of a zone's area, and
now supply that share. Positives come from 4,230 (A) and 4,439 (AE) polygons, against 592 and 950 in
v2; the largest polygon supplies 2.15% (A) and 2.30% (AE) of positives.

### 3.4 Mask class against Earth Engine land cover

| Check | Result | Rule |
|---|---|---|
| raster vs Earth Engine at 2,000 random v2 points (A1) | 2,000 of 2,000 agree | Stop below 99.9% |
| `meta_mask_lc` = `landcover` (B2) | **80,000 of 80,000** | at least 99.9% |
| `landcover` 11 or missing (B2) | **0** points | at most 80 |

No point disagrees, so none is listed. Earth Engine reads the same pixel as the mask because both
project the points' longitude and latitude with the same formula; the NAD83 vs WGS 84 difference
the brief warns of (about 1 m) is applied by neither side.

### 3.5 Covariates and audits

- **Earth Engine (B2):** 40 of 40 chunks in 2.3 min, 0 failures; 0 points without elevation and the
  worst band coverage 100% (v2: 91.9%, the gap being MERIT Hydro's missing open water). The selftest
  returned every band at the three Vermont points (the routine's check). Its printed sanity hint
  ("valley bottom lowest tpi_300, highest upstream area") does not hold on the native grid: the
  valley-bottom point has the lowest HAND but not the lowest tpi_300, and the upland point has the
  largest upstream area. No earlier selftest log exists to compare.
- **Atlas 14 (B3):** 36 new blocks, 35 fetched, 1 failed ("no frequency matrix in the response",
  usually a point outside Atlas 14's area); 1 point has no rainfall values. Not a model input.
- **NHDPlus (B3):** rebuilt with `--overwrite`: 53,094 flowlines and 5,377 coastline features, the
  review's numbers. 99.1% of points lie in a catchment (v2: 90.7%). StreamCat coverage check: 0
  points with a catchment but no StreamCat row (Stop above about 400).
- **Missing features (B4):** `hand` is missing on 16 points and `wetindexcat` on 691 (v2 on land: 44
  and 420; on all points 6,451 and 7,461). 674 of the 691 are Zone AE positives outside every
  NHDPlus catchment (3.4% of AE positives, against 0.1% of AE negatives). That is a label cue inside
  Zone AE, but below the audit's 5-point threshold; the brief expected "a few hundred", and this is
  nearer 700. Zone A has none.
- **Dataset audit (B4, `results/2026-10-23/data_audit/audit.md`):** 14 OK, 1 WRONG, 4 DECIDE (v2: 12 OK,
  1 WRONG, 6 DECIDE).
  - B4, negatives on open water: **0.0 points** of excess in both zones (v2: A +2.2, AE +19.1), and
    0% of band negatives without HAND are on water.
  - B2 (positive shares of the halves, 0.547 vs 0.514, gap 0.033) and B3 (no label-correlated
    missingness) moved from DECIDE to OK.
  - B7 is WRONG as in v2: 2 positives lie up to 218 m outside the TIGER state line (v2: 2, up to
    214 m). Positives are not masked by the county polygons; this is where NFHL and TIGER draw the
    state line differently.
- **Step 6:** `FEATURES.txt` unchanged, its checksum OK, and the other states' Atlas 14 rows
  unchanged.

### 3.6 The split

The cell (C1): train 29,161 | ID test 10,835 | OOD test 12,531 points, tiles train 69 | test 28;
flood-labeled shares ID 0.547, OOD 0.514. `audit_cell.py`: **verdict PASS** (0 FAIL, 0 WARN).

Seed 0 is a new draw of test tiles. `zoo.split_tiles` shuffles the sorted list of tiles that hold
points, and that list changed: 98 tiles instead of 102 (the mask emptied 6 water-only tiles,
`18_16`, `28_18`, `29_16`, `32_7`, `32_8`, `32_9`, and 2 panhandle tiles gained points). Only 10 of the
28 new test tiles were seed-0 test tiles in v2. So "split 1" before and after compares different
test areas as well as different points.

---

## 4. Regions

Spatial selection, S = 500: 20 pieces over 11 test tiles; 7 pieces of at least 20 points hold 426
points (85%). Every point is on land, so every region has its full count of land points, and the
§2 Stop (fewer than 3 regions of at least 20 land points) does not fire. Files:
`results/2026-10-23/regions/`, `results/2026-10-23/o5/regions_land.csv`.

| # | Region (counties) | Points | Type: flood-labeled share | Zone A polygons of its positives (largest, km²) | Ring polygons of its negatives | Land cover | Median HAND, river distance |
|---|---|---:|---|---|---:|---|---|
| 0 | Hendry, Palm Beach | 114 | flood-labeled, 52% | 2 (156.0) | 3 | cultivated crops 94% | 0.4 m, 22.1 km |
| 1 | Liberty | 107 | dry-labeled, 2% | 2 (0.5) | 22 | woody wetlands 59% | 1.7 m, 0.7 km |
| 2 | Levy | 63 | flood-labeled, 100% | 17 (117.6) | 0 | evergreen forest 54% | 0.3 m, 1.5 km |
| 3 | Dixie | 42 | flood-labeled, 57% | 3 (128.4) | 5 | woody wetlands 45% | 2.1 m, 2.7 km |
| 4 | Hamilton | 41 | dry-labeled, 39% | 7 (156.5) | 16 | evergreen forest 46% | 0.7 m, 1.9 km |
| 5 | Hendry | 37 | dry-labeled, 5% | 1 (3.7) | 10 | pasture/hay 38% | 0.6 m, 16.3 km |
| 6 | Palm Beach | 22 | dry-labeled, 0% | 0 | 1 | woody wetlands 50% | 0.4 m, 6.8 km |

A ring polygon is the Zone A polygon whose 5 km band produced a negative; all are Zone A. Region 0
lies in the farmland south of Lake Okeechobee and is an almost even mix of the two labels.

**No old seed-0 region is found again.** No point of a new region lies within 2 km of a point of an
old seed-0 region of 2026-10-16, so no new region meets the footprint rule (at least half of its
points within 2 km). The nearest are the new Hamilton region, 4.6 km from the old one, and the new
Dixie region, 19.7 km from the old Dixie/Lafayette region. By construction most old regions could
not recur: their tiles (`24_16` for Dixie/Lafayette and Lafayette/Taylor, `26_17` for both Bradford
regions, `20_16` for Gulf) are training tiles in v3; only Hamilton's tile `25_18` is a test tile in
both. Log: `logs/2026-10-23/C5_regions_describe.log`.

---

## 5. Twins

`results/2026-10-23/paper_figures/region_twins_out/` (`twins_s0.csv`, `regions_summary.csv`,
`pooled_summary.json`); logs `C7_region_twins_*.log`. 200 Zone AE and 200 Zone A patches per
region, 1,000 bootstrap draws over regions. d is the mean difference per feature in standard
deviations; g is the strongest 15 test models' accuracy on the twins minus on the region.
Percentiles are the share of the region's own patches below its value.

| Region | Kind | n | d | d pct, AE / A patches | Accuracy region / twins | g | g pct, AE / A patches | p(flood) region / twins | Flood-labeled share region / twins |
|---|---|---:|---:|---|---|---:|---|---|---|
| Hendry, Palm Beach | flood | 114 | 0.063 | 8 / 0 | .50 / .54 | +0.036 | 68 / 98 | .53 / .53 | .52 / .96 |
| Liberty | dry | 107 | 0.122 | 21 / 30 | .31 / .85 | +0.536 | 100 / 100 | .62 / .61 | .02 / .79 |
| Levy | flood | 63 | 0.116 | 17 / 0 | .25 / .82 | +0.579 | 100 / 100 | .37 / .38 | 1.00 / .29 |
| Dixie | flood | 42 | 0.135 | 38 / 25 | .33 / .66 | +0.333 | 99 / 97 | .46 / .47 | .57 / .62 |
| Hamilton | dry | 41 | 0.291 | 95 / 100 | .21 / .65 | +0.444 | 100 / 100 | .52 / .52 | .39 / .63 |
| Hendry | dry | 37 | 0.089 | 17 / 2 | .42 / .63 | +0.204 | 91 / 100 | .54 / .57 | .05 / .87 |
| Palm Beach | dry | 22 | 0.099 | 22 / 0 | .32 / .57 | +0.255 | 92 / 100 | .54 / .56 | .00 / .50 |

**Pooled (split 1 alone; size-weighted over the 7 regions, 426 points):**
- d = **0.119** [0.084, 0.175]; median of the pooled Zone AE patches 0.167; the regions sit at the
  **4.5th percentile** of the Zone AE patches (and below every Zone A patch). They are closer to their
  twins than ordinary Zone AE land is to its own.
- g = **+0.336** [0.139, 0.502]; above all 200 pooled patches of both kinds (medians −0.005 Zone AE,
  −0.043 Zone A).
- With all 60 test models instead of the strongest 15: g = 0.216.
- Balance, region minus twin in standard deviations: largest for the coast distance (+0.082) and the
  river distance (+0.042); all six features within 0.1 SD.
- Every region's g is at or above the 95th percentile of its own Zone A patches (7 of 7). Against its
  Zone AE patches, Hendry/Palm Beach is at the 68th percentile (g = +0.036), Hendry at the 91st and
  Palm Beach at the 92nd. Hamilton again has the rarest inputs, d at the 95th percentile of its Zone
  AE patches (v2: 0.32, 96.5th).

**The paper's Table 3 on the new data** (split 1 alone; pairs are region / twins; in parentheses,
the median over pooled patches: Zone AE patches for d, Zone A patches for g; r from
`results/2026-10-23/o5/A_interpolation.csv`, land points):

| | Flood-labeled | Dry-labeled |
|---|---|---|
| Regions (land points) | 3 (219) | 4 (207) |
| r, the 60 test models | −.49 | −.55 |
| r, held-out test models that do not interpolate (51) | −.45 | −.47 |
| Input distance d | .09 (.17) | .15 (.16) |
| Strongest models' p(flood) | .47 / .48 | .58 / .58 |
| Flood-labeled share | .67 / .70 | .10 / .74 |
| Strongest models' accuracy | .39 / .64 | .31 / .74 |
| Accuracy gap g | **+.25** (−.10) | **+.43** (.02) |

The paper's "r, test models" row used the pooled held-out non-interpolating set (review §5
item 14(c)); both versions are given. Both kinds now invert across the test models; in v2 the
dry-labeled kind did not (+.01). The flood-labeled column is dominated by the 114-point
Hendry/Palm Beach region, which is half dry-labeled; without it (Levy and Dixie, 105 points) the
flood-labeled share is .83 against .42 for the twins.

---

## 6. Deviations, code and tests

### 6.1 The amendment: C1 in accuracy, AUROC and F1

Victor's decision of 2026-10-23 arrived while B1 was sampling, before any v3 result existed: no
shift, accuracy or AUROC number of the rebuilt data had been computed or seen. It makes accuracy the
primary metric of C1 and AUROC and F1 secondary, adds the strongest-15 gaps and a prevalence check,
and turns the B2 share rule of `audit_dataset.py` into a reported check. It was implemented before C2,
and C2 ran with the updated script. The C1 row of §2 of the brief is rewritten and marked as amended.

**The W7 check** (`results/2026-10-23/o1_w7_check.csv`, the W7 cells of 2026-10-16 with the archived
v2 dataset, 2,000 draws; logs `amend_o1_w7_check.log` and `amend_w7_check_vs_expected.log`). Scope
"all"; expected values from the amendment in brackets. **Every mean is within 0.001 and every
interval end within 0.01, so the Stop did not fire.**

| Split | AUROC gap | Accuracy ID / Zone A | Accuracy gap [95% CI] | F1 gap | Strongest-15 accuracy gap |
|---|---|---|---|---|---|
| s0 | .0036 (.004) | .7360 / .7466 (.736 / .747) | −.0106 [−.0600, +.0417] (−.011 [−.060, +.042]) | −.0261 (−.026) | +.0103 (+.010) |
| s1 | .0722 (.072) | .7504 / .6845 (.750 / .685) | +.0659 [+.0168, +.1213] (+.066 [+.017, +.120]) | +.0836 (+.084) | +.0830 (+.083) |
| s2 | .0631 (.063) | .7793 / .7226 (.779 / .723) | +.0567 [−.0078, +.1272] (+.057 [−.008, +.122]) | +.0960 (+.096) | +.0839 (+.084) |
| pooled | .0463 (.046) | .7552 / .7179 (.755 / .718) | +.0373 [−.0054, +.0812] (+.037 [−.005, +.083]) | +.0511 (+.051) | +.0591 (+.059) |

The pooled reweighted-accuracy gap is +.0350 [−.0039, +.0732] (expected about +.035): reweighting
Zone A to the Zone AE flood share moves the pooled accuracy gap by −.0023, and each split's by less
than .007. The flood shares (Zone A / Zone AE) are .476 / .403, .421 / .466 and .447 / .539.

**"The AUROC columns must equal results/2026-10-16/o1_shift.csv exactly"** cannot hold as written,
because the 2026-10-21 review changed the O1 code after that file was made. Two comparisons show
that the amendment itself changed nothing:
- The pre-amendment `o1_shift.py` (git HEAD) and the amended one, both at 200 draws on the W7 cells,
  write **identical values in all 26 old columns**; the 39 new columns follow them
  (`logs/2026-10-23/amend_head_vs_new_boot200.log`).
- Against the 2026-10-16 file at the same 200 draws, 15 of 24 columns are identical. The other nine
  differ by at most 0.0003, and the land scope's counts by 1–2 points
  (`logs/2026-10-23/amend_new200_vs_20261016.log`). These are the review's fixes: a point without a
  land-cover class no longer counts as land, p is not rounded, and the "all" scope reads the stored
  AUCs. At the default 2,000 draws the intervals move further, as the review documented.

### 6.2 Other deviations and notes

1. **C3 ran before C2.** The 31-minute search started right after the cell audit and overlapped the
   amendment work; it does not use C2's output, so no number depends on the order.
2. **The B2 change came after Part B's audit.** `results/2026-10-23/data_audit/` was written with the
   old B2 rule, as the brief's Part B intends ("Parts A and B do not change"). Its B2 row is OK
   anyway, because the share gap is 0.033.
3. **shadedX negatives are masked too** when `--land-mask` is on, the same way as positives. The
   brief names positives and band negatives only, but the flag's audit requires every point on land.
   This run builds no shadedX negatives.
4. **How the flag enters the sampler.** The sampling pins fix the parser's option list and the
   signature of `sample_band_negatives`, so `--land-mask` is parsed before the main options and the
   mask and the OPEN WATER polygons reach the samplers in a copy of the build settings. A build
   without the flag gets exactly the settings it got before.
5. **`aotil/o5_interpolation.py --seeds 0` exits 1 after writing its CSV.** Its summary printout
   expects a pooled row, which one seed cannot give (its own code says so). The CSV is complete; the
   numbers in this report come from it.
6. **Figures.** `fig_maps_combined.py` labels the new regions instead of the old place names; the
   other figure scripts changed only in their paths and split. No figure was copied into the paper.
7. **Earth Engine selftest:** see §3.5 for the sanity hint that does not hold.
8. **Data that now point at v3.** `data/processed_w7_minus_*/dataset_FL.parquet` link to
   `data/processed/dataset_FL.parquet`, so they now read v3 points while their `cells_w7_minus_*`
   cells were built on v2. `data/interim/analysis/point_extras_FL.parquet` is still the v2 table,
   keyed by point_id. Neither is used here; rebuild or archive them before using them.
9. **Atlas 14:** one block failed (outside the Atlas 14 project area), leaving 1 point without
   rainfall values. Atlas 14 is not a model input.
10. **`data/processed/dataset_FL.parquet` is tracked by git**, so the v3 merge shows as a modified
    file; the v2 table is in `data/interim/archive/sampling-v2/` with its checksum.

### 6.3 Code added or changed

| File | What |
|---|---|
| `pull_land_mask.py` (new) | Downloads NLCD 2021 `landcover` for Florida from Earth Engine on its native 30 m grid (tiles of 4,096 × 4,096 pixels, resumable), writes `data/interim/gis/nlcd2021_FL.tif` and `.json`, prints pixel counts; `--check` compares it with Earth Engine at 2,000 points. |
| `sample_points.py` | `--land-mask PATH` (off by default): `LandMask`, the land test after placement for positives and shadedX, the OPEN WATER and land tests in the band filter chain, `meta_mask_lc`, the mask path and SHA-256 in the stored settings, and two `--audit` checks. |
| `aotil/regions_land.py` (new) | Writes the regions table (`regions_land.csv`); reproduces `results/2026-10-17/o5/regions_land.csv` byte for byte from `results/2026-10-16/regions/`. |
| `aotil/o1_shift.py` | The amendment: accuracy, F1, the prevalence check and the strongest-15 columns from TP/FP/FN/TN counts per model and tile, in `o1` and `o1_pooled`; accuracy printed first; every old column kept, the new ones written after them. |
| `audit_dataset.py` | B2 reports the share gap (flagged above 0.05) instead of asking for a decision. |
| `results/2026-10-23/paper_figures/*.py` | Copies of the 2026-10-19 scripts reading this run (constants in `common.py`, split 0); `region_recurrence_ref.py` refuses to run on one split. |
| `docs/RUN-FL-V2.md`, `docs/DECISION-PROVENANCE.md`, `docs/TASK-REBUILD-V3.md` | The mask in step 3 (b) with its command; provenance rows for the mask, the B2 rule and the three metrics, the old rows marked replaced; the amended C1 row. |

### 6.4 Tests

All are private (gitignored) except `aotil/test_regions_land.py`, which follows the folder's
convention.

| Test | Checks | Result |
|---|---|---|
| sampler pins (`tests.characterization.test_sampling`) | the sampler with the flag off, unchanged | 70 OK before and after |
| `tests/test_sampling_land_mask.py` (new) | no point on water or off the raster; P1 (1 km² of land) gets twice the points of P2 (1 km², half water); OPEN WATER candidates dropped; the lookup with and without a CRS transform; the flag through `main()` and `--audit` | 7 OK |
| `tests/test_pull_land_mask.py` (new) | grid snapping, tiling, resume, mosaic, sidecar, the Earth Engine check with a fake sampler | 10 OK |
| `aotil/test_regions_land.py` (new) | types, land rule, counties, the byte-for-byte reproduction | all pass |
| `aotil/test_o1_shift.py` (5 checks added) | exact accuracy and F1 for known TP/FP/FN/TN; ties at p = 0.5 predicted dry; integer tile weights = duplicated points; real-cell accuracy = `correctness(S, y)[0].mean(1)`; equal flood shares leave accuracy unchanged | all pass |
| O1 pins (`tests.characterization.test_o1`) | extended only by the new keys and columns (see below) | 42 OK |
| dataset-audit pins (`tests.characterization.test_audit_dataset`) | B2's new verdict | 51 OK |

**Characterization goldens changed, by additions only.** A script compared the regenerated O1 goldens
with the old ones (the old ones and the script are in `Claude outputs/goldens-before-2026-10-23/`):
every old value, array and file is unchanged; the
additions are the `Zo`/`Zi` keys of `cell_arrays`, 936 new row entries of `o1`/`o1_pooled`, 117 new
CSV columns (39 per CSV), 433 printed numbers and 7 more ok lines of the owner's check script. The
pinned column list and row keys in `test_o1.py` were extended by the 39 new names. In the
dataset-audit goldens the only change is the share-gap case of B2, DECIDE → OK, with its inline
expectation in `test_audit_dataset.py`; the o5 goldens are byte-identical.

**The full private suite** (`logs/2026-10-23/D_tests_all_after_leak_fix.log`): 1,208 tests, 3 failures
and 1 error, none from the code:
- the brief's own sentence "Then `aotil/audit_selection.py --smoothed-classic`" is not a complete call,
  and a pin parses every such call in `docs/TASK-*.md`;
- three tests pair v2-era cells or outputs with `data/processed/dataset_FL.parquet`, which now holds v3:
  `test_stats_01` (real case) and `test_zoo_01` pass on the archived v2 dataset (checked), and the
  controls pin reruns `aotil/test_match_controls.py` on the 2026-10-17 O5 outputs.

The first full run had 40 more failures. They came from `tests/bugs/test_sampling_02.py`, which set
`sample_points.PREP_DIR` to the real data folder and never restored it; it is skipped until the points
come from the fixed sampler, so it first ran today, passed, and broke every later test that builds a
synthetic prep folder. It now patches the path for the duration of the test.
