# Project plan: label provenance as distribution shift in flood susceptibility

**v5, 2026-10-16** (model features switched to the Woznicki et al. 2019 predictor set, §0a) ·
v4 2026-10-13 (sampling protocol v2 added the same evening, §2a) · supersedes v3 (2026-09-19, in git history) · aligned with the paper draft
`CS371_Cho_Li/acl_latex.tex` as uploaded 2026-10-13 (introduction, five anticipated outcomes,
three contributions, related work; Methods and Results not written yet)
**Scope:** AI Research Practicum (CS371W), Kevin Cho and Victor Li; assumed to end mid-November 2026
**Companions:** `docs/REGION-ANALYSIS-FLORIDA.md` (v2, the boundary analysis in full),
`docs/SPATIAL-OODSELECT.md` (the Laplacian term and the coherence metrics), `docs/RUN-FL-V2.md`
(the data routine, run order step 1), `aotil/README.md` (what each analysis module does, and the
run order), `docs/TASK-DATA-AUDIT.md` (the data-audit brief, done 2026-10-15),
**`docs/TASK-FULL-RUN-W7.md` (the full-run brief for Claude Code with the v5 features)**

When this plan and the paper disagree, the paper wins and this file gets fixed.

---

## 0a. What changed in v5 (2026-10-16): the model features

**Decision (Victor, 2026-10-16).** Every model trains on the predictor set of Woznicki et al.
(2019, Table 2), mapped onto our columns, minus its two SSURGO soil variables. The list is
`data/processed_w7/FEATURES.txt` (seven lines). `data/processed_w7/dataset_FL.parquet` is a link to
`data/processed/dataset_FL.parquet`, so the points are unchanged. `data/processed/FEATURES.txt`
(45) is unchanged and is no longer the model list.

| Woznicki 2019 predictor | our column | source | difference from theirs |
|---|---|---|---|
| NLCD land cover | `landcover` | NLCD 2021, 30 m | theirs NLCD 2011; a class code every family reads as a number |
| slope | `slope` | 3DEP, 10 m | theirs NED, 30 m |
| DEM5x5 (std. dev. of a 5×5 DEM window) | `tri_300` | 3DEP, 10 m | std. dev. of 10 m elevation in a 300 m cell (theirs: a 150 m window) |
| CTI (compound topographic index) | `wetindexcat` | EPA StreamCat (EnviroAtlas index) | a catchment mean, not a per-pixel value |
| OFD / HOFD (distance to channel; coastlines count as channels) | `log10_dist_river`, `log10_dist_coast` | NHDPlus V2.1 | straight-line distance to mapped flowlines and to the coastline, not along the flow path to a DEM channel |
| VOFD / VDC (vertical distance to channel) | `hand` | MERIT Hydro, 90 m | MERIT's own drainage network |
| fldfreq, fluvclass (SSURGO / STATSGO) | none | none | left out: approximate studies draw Zone A partly from soil maps, so a soil feature could carry the approximate procedure into the model (`CS371_Cho_Li/CHANGES-2026-10-19.md`, item 3.5); SSURGO is pulled only inside the region boxes, for class S of O5 |

**Why.** It is a published predictor list, not one we chose. It is also the model the paper's
Related Work describes: Woznicki et al. trained random forests on pooled Zone A and Zone AE
labels, which is the practice the paper tests.

**What the feature checks said.** `results/2026-10-15/feature_sets/` and
`results/2026-10-16/cited_feature_sets/` compared eight cited feature sets. Woznicki 7 was not the
set with the largest gap (one strong model over 15 tile folds: Woznicki 0.101, all 45 0.084,
Degiorgis/Manfreda 5 0.131).

**First look with the real population** (`results/2026-10-16/w7_first_look/`, Claude's sandbox,
seeds 0–2, reduced budgets; details in `docs/TASK-FULL-RUN-W7.md` §0):

| outcome | seeds 0 / 1 / 2 | 45 features, seed 0 |
|---|---|---|
| O1 AUC gap, all points (share of models worse) | 0.004 (56%) / 0.072 (94%) / 0.063 (95%); intervals exclude 0 on seeds 1–2 | 0.020 (71%) |
| O2 r, 60 test models | +0.729 / +0.863 / +0.872 | +0.880 |
| O3 classic r, S = 500 (pieces, tiles) | −0.889 (305, 24) / −0.936 (325, 24) / −0.914 (303, 21) | −0.889 (221, 23) |
| O4 spatial r, S = 500 (pieces, tiles) | −0.888 (13, 10) / −0.838 (26, 15) / −0.748 (10, 8) | −0.763 (20, 12) |
| O5 | not run | not run |

**What changes.**
- Every model-dependent output is rebuilt with the seven features: cells in
  `data/interim/aotil/cells_w7/`, selections, regions, lobes, and the DEM and SSURGO pulls for the
  new region boxes. All numbers in §3 and in `REGION-ANALYSIS-FLORIDA.md` §2 are 45-feature results.
- The other 38 columns stay in the point table as descriptive variables for the regions (soil,
  climate, rainfall, catchment context). They are not model inputs.
- The Atlas 14 refill no longer blocks the run: Atlas 14 is not a model input.
- Two v4 issues disappear: `channel_slope`'s −9998 code and the `tmean8110*` latitude proxy are
  not in the set.
- One stays: `hand` and `wetindexcat` are missing on open water (8.1% and 9.3% of points, almost
  all band negatives, mostly Zone AE), so "missing" still leans toward "negative" for the hgb and
  xgb families. Report O1 on all points and on land points. In the first look, two seed-1 regions
  are band negatives on Lake Okeechobee's open water.
- Paper: "45 covariates from seven public sources" becomes the seven Woznicki predictors from five
  sources (3DEP, MERIT Hydro, NLCD, NHDPlus, StreamCat). Before/after in
  `CS371_Cho_Li/CHANGES-2026-10-16.md`. Methods must give the mapping table above.
- The full run, seeds 0–2 through the boundary statistic: `docs/TASK-FULL-RUN-W7.md`.

---

## 0. What changed in v4

The paper now fixes the experiment. This version cuts the plan down to what the paper
claims, says what each claim needs, and lists the data that is still missing (§4).

| | v3 (2026-09-19) | v4 (from the paper) |
|---|---|---|
| scope | Florida primary; Iowa/Missouri if time; ten-state dataset | **Florida only** |
| direction | both (AE→A and A→AE) | **AE→A**: train on Zone AE, evaluate on Zone A. The A→AE results stay on disk but are not in the paper |
| selection | aotil sweep (continuous scores, μ sweep) | **OODSelect as the authors released it** (`aotil/oodselect_paper.py`), classic vs. with the Laplacian penalty |
| model population | ≥ 300, seven families | **300 models, eight families** (dt, et, hgb, lr, mlp, nb, rf, xgb), described as tabular learners chosen in OODSelect's style; the count is not cited to Salaudeen et al. |
| explanation | membership model of "in region" on provenance / terrain / climate groups (RQ3, E3) | **boundary-agreement analysis**: do the Zone A boundaries inside each region follow terrain, or pre-existing lines the model never saw (soil-survey lines, county lines, map-sheet edges)? This is G1 of `REGION-ANALYSIS-FLORIDA.md` |
| map | E4 per-region maps | **Contribution 3**: one map of Florida showing where approximate studies differ from detailed ones |
| dropped | RQ4 | RQ4; IA/MO (E6); provenance graph (E5); full ponding analysis (G2 §4.2); DISDE; R6 excision; shadedX. All optional at most (§5) |

---

## 1. The claim, as the paper states it

Zone A and Zone AE share one definition (the 1% annual chance floodplain) but not one
procedure. Zone AE boundaries come from base flood elevations computed in detailed hydrologic
studies. Zone A boundaries come from approximate studies built on soil maps, high-water
profiles, aerial photos of past floods and topographic maps. Prior work pools both into one
label and assumes the model absorbs the difference. **Hypothesis: it cannot, and the
difference appears as a distribution shift wherever the approximate method diverges from the
detailed study.**

The test. If models absorb the shift, accuracy-on-the-line holds: across models, out-of-domain
(Zone A) accuracy tracks in-domain (Zone AE) accuracy. OODSelect searches for Zone A subsets
where that correlation inverts. The graph-Laplacian penalty makes those subsets spatially
coherent regions. A boundary analysis of each region then asks whether the label there follows
something the covariates cannot see.

---

## 2. Setup as the paper states it, checked against the repo

Checked 2026-10-13 from `dataset_FL.parquet` and the `FL_band_AE_s0` cell. These are the **v1
sample** now on disk. §2a lists what the rebuild changes.

| paper says | where it comes from | status |
|---|---|---|
| "98,578 points across Florida" | `data/processed/dataset_FL.parquet` | 98,578 rows ✓. **But 36,000 of them are shadedX negatives, which the band experiment does not use.** The band subset is 62,578 (19,963 positives and 11,326 band negatives per zone). After the seed-0 50 km tile split: train 22,658 · ID test 7,634 · OOD test 9,412. Decide which count Methods reports |
| "45 covariates" | `data/processed/FEATURES.txt` | 45 ✓ |
| "seven public sources" | pull scripts | expected: USGS 3DEP, MERIT Hydro, NLCD, JRC Global Surface Water, NHDPlus V2, EPA StreamCat, NOAA Atlas 14. `log10_dist_coast` comes from NHDPlus V2 Coastline features (`pull_hydro.py`), so seven looks right. **Not verified per feature**; audit item |
| labels from FEMA zones; negatives sampled outside the mapped area | `sample_points.py`; `band` protocol | ✓, but the code samples outside the mapped *floodplain* and inside the mapped *area*. Band rule: ring 500–5,000 m around the polygons that supplied that zone's positives; keep a point only if its nearest SFHA of any zone is 500–5,000 m away and it lies inside a FIRM panel; 200 m thinning. A negative's zone is the zone of the polygons whose ring produced it, so a "Zone A" negative's nearest floodplain can be Zone AE. Florida: 11,326 per zone, 0.57 per positive (supply-limited below the 2:1 cap). Source: NFHL `S_Fld_Haz_Ar` and `S_FIRM_Pan` only |
| spatially contiguous blocks held out | `tile_50km`, one 70/30 split per seed, test tiles must hold both zones | ✓ (71 train / 23 test tiles at seed 0) |
| 300 models, eight families | `aotil/zoo.py`; cells in `data/interim/aotil/cells/` | ✓ dt 38, et 38, hgb 38, lr 38, mlp 37, nb 37, rf 37, xgb 37 |
| train on Zone AE, evaluate on Zone A | cell `FL_band_AE_s0` | ✓, seed 0 only |
| kNN graph over Zone A points, Laplacian penalty, connected regions extracted from the relaxed solution | `aotil/oodselect_paper.py` with μ = 3, k = 10; regions = connected components of the top-S mask on the 10-NN graph | seed 0, μ = 3 only. **No region-extraction script in the repo**; the component tables of 2026-09-27 were computed ad hoc |

## 2a. Sampling protocol v2 (decided 2026-10-13)

Victor's decisions after each sampling setting was checked against the literature. The files on
disk, and every number in §2 and §3, are still the v1 sample until `docs/TASK-DATA-AUDIT.md`
Part A rebuilds Florida. The audit fills in the "achieved" column.

| setting | v1 (built 2026-09-18) | v2 | reason / citation | achieved |
|---|---|---|---|---|
| minimum spacing between points of the same class | 200 m | **100 m** | Mundayatt et al. 2026 (`mundayatt2026`, arXiv:2608.08321, p. 12): "A minimum spacing of 100 m is maintained among the generated locations to reduce excessive cluttering". PDF in `refs/cited/` | 100 m: 0 pairs under 100 m, minimum nearest-neighbour distance 100.114 m (EPSG:5070), median 431 m (real build 2026-10-14, `docs/DATA-AUDIT-2026-10-14.md` §2) |
| minimum spacing across classes | 100 m (half the floor) | **100 m** | same paper, same page: non-occurrences are kept "if they are at least 100 m from known occurrence locations" | 100 m: one whole-state pass at the full floor; 0 pairs under 100 m across any class or zone |
| positives per zone | 20,000 target (19,963 achieved) | **exactly 20,000** | a size budget; not cited | 20,000 A / 20,000 AE (3% surplus cut after the whole-state pass) |
| band negatives per positive | up to 2 (0.57 achieved) | **exactly 1:1** (20,000 per zone) | 1:1 is the common choice: Gu et al. 2024 (`gu2024`), Sinčić et al. 2025 (`sincic2025`), Collins et al. 2022 (`collins2022`, PDF to fetch). Acknowledged as a choice, not an optimum: Woznicki et al. 2019 chose about 1:4 after testing several ratios (`woznicki2019`, p. 5), and Jia et al. 2026 report 1:2 beats 1:1 (`jia2026`, PDF to fetch) | 1.00: 20,000 per zone (`--ring-oversample 6`: 32,998 A and 48,212 AE candidates after thinning, for 20,600 needed) |
| shadedX negatives | 18,000 per zone | **not built for Florida** | not in the paper (§5) | not built (`--protocols band`) |
| Florida total | 98,578 | **80,000** (4 × 20,000) | | 80,000 |
| unchanged | area-proportional draw over the polygons; ≤ 2% of a zone's points from one polygon; band 500–5,000 m from any SFHA; inside a FIRM panel; 10 km blocks; 50 km tiles; seed | | | unchanged, plus one rule added by decision: band negatives must lie inside Florida (`--state-polygons`, TIGER 2026 counties; the real build's mask removed 2,609 A and 1,092 AE candidates; a scratch build without it had 4.4% of band negatives in AL/GA). Largest polygon 2.23% A / 2.09% AE of a zone's positives (the 2% cap applies to the candidate pool, so final shares can exceed it, as in v1); band 500–4,965 m, 40,000/40,000 inside a panel; 1,664 blocks, 102 tiles |

Consequences:
- **Everything downstream is rebuilt**: the covariates for the new points, the cell, the
  selections, the regions. §3's seed-0 numbers and the regions in `REGION-ANALYSIS-FLORIDA.md`
  §2 are v1 results and will be replaced. The paper's "98,578 points" becomes 80,000 if the
  rebuild reaches its targets.
- **1:1 should remove most of O1's base-rate problem** (v1: ID half 0.53 positive, OOD half
  0.64). The audit checks the new halves.
- **At 100 m, two points can occasionally share one ~90 m MERIT Hydro cell** (HAND, upstream
  area), so "one point per raster cell" is no longer the stated reason for the spacing; the cited
  reason is de-cluttering. The audit counts how often it happens.
- **NHDPlus V2 raw data is not on disk** (`data/raw/` holds only the NFHL), and
  `pull_hydro.py` needs catchments and flowlines to place new points. The rebuild needs at least
  the Florida units (03N, 03S, 03W).
- **Citations to fetch into `refs/cited/`** (found by web search; confirm the statements in the
  PDFs before writing prose from them): `collins2022` (Collins et al. 2022, *Environ. Res. Lett.*
  17:034006, doi:10.1088/1748-9326/ac4f0f; 1:1 pseudo-absences chosen after a sensitivity test)
  and `jia2026` (Jia et al. 2026, *JRMGE* 18(4):2859–2880, doi:10.1016/j.jrmge.2025.12.011; 1:2
  better than 1:1, read from the abstract only).

---

## 3. Experiments, one per anticipated outcome

> All numbers in this section are from the v1 sample (200 m spacing, 0.57:1 band negatives).
> They are replaced once the v2 rebuild (§2a) and the seed-0 re-run are done.

"Have" means in the repo now. "Need" means still to run or build.

### O1. The distribution shift is real

> "A model trained on the detailed Zone AE labels scores lower on the approximate Zone A labels
> than on held-out Zone AE labels."

- **Measure.** Per model: score on held-out Zone AE (ID half) vs. on Zone A (OOD half), both
  on the same test tiles. **Report AUC (or balanced accuracy), not plain accuracy.** The two
  halves have different positive shares (ID 0.53, OOD 0.64 at seed 0), so an accuracy gap
  mixes the shift with the base rate.
- **Have.** `FL_band_AE_s0_models.csv`. Mean AUC drop 0.033 (median 0.036; a quarter of models
  lose less than 0.007). Mean accuracy drop 0.021, and about a quarter of models are *more*
  accurate on Zone A. The T4 check (`results/2026-09-26/aef_checks`, 90 models, one seed) puts
  Zone A AUC at 0.84 against ID AUC of 0.90–0.94 for the stronger models.
- **Need.** Seeds 1–2; a tile-bootstrap interval on the gap. The aggregate drop is modest. That
  suits the paper's argument (aggregation hides the failure), but O1 should not promise a large
  drop.

### O2. Accuracy-on-the-line holds in aggregate

- **Measure.** Pearson r of probit ID accuracy vs. probit OOD accuracy across models, on the
  full Zone A set; plus the paper's model-count rule (adding models changes r by < 1%).
- **Have.** r = +0.78 on the 60 test models (`report_paper.py`, seed 0). The count rule is met
  by 20 models.
- **Need.** Seeds 1–2.

### O3. OODSelect finds inverted subsets, but they are scattered

- **Measure.** Classic (μ = 0) paper-protocol OODSelect over the budget grid 10–7,500.
  Test-model r with Fisher 95% CI; random and most-misclassified baselines; coherence metrics
  (kNN purity, connected components, tiles touched).
- **Have (seed 0).** r ≤ −0.3 up to S = 5,000, i.e. 53% of the Zone A set. At S = 500:
  r = −0.72, purity 0.26, 186 components across 22 of 23 tiles.
- **Need.** Seeds 1–2. The two controls the first run skipped: **null-select** (run the
  selection against shuffled ID accuracies; it must not invert) and the **family-disjoint split**
  (select on tree models, report on the others, and the reverse). These separate a real
  spurious-feature effect from a shared model-quality axis.

### O4. The Laplacian penalty gives coherent regions at a small cost in r

- **Measure.** Spatial (μ = 3, k = 10) vs. classic at each budget: change in r and in coherence.
  The paper calls the two objectives competing, so show the tradeoff: a small paper-protocol μ
  sweep (μ ∈ {0, 1, 3, 10}) at S = 250, 500, 1,000. Never pick μ by test r.
- **Have (seed 0, μ = 3).** Cost 0.08–0.14 in r at S = 500–2,500; a gain at S = 250 (−0.69 vs.
  −0.48). At S = 500: purity 0.91, 8 components, 5 tiles. A μ sweep exists only for the older
  aotil optimizer (job 562664).
- **Need.** Seeds 1–2; the paper-protocol μ sweep; the **smoothed-classic** ablation (smooth the
  classic weights with the kNN operator, then take the top S), so better regions can be credited
  to the penalty and not to smoothing in general.

### Region extraction (feeds O5 and the map)

- From `spatial_mu3_knn10/FL_band_AE_s*_S{250,500}_theta.npy`: top-S mask → connected
  components on the same 10-NN graph the selection used → keep components with ≥ 20 points →
  a region table keyed by `point_id`.
- Seed 0, S = 500 gives four Zone A regions (`REGION-ANALYSIS-FLORIDA.md` §2.1): Santa Fe /
  New River swamp (Bradford–Union), Fort Drum / Blue Cypress marsh (Okeechobee–Indian River),
  Osceola swamps (Columbia), and the Everglades edge (Miami-Dade, coastal). All are
  `NOT_POPULATED` approximate Zone A in flat wetland terrain, not river corridors.
- **Need.** `aotil/regions.py` to do this reproducibly. Check which components recur on seeds
  1–2 before treating any of them as an object.

### O5 and Contribution 3. The regions differ from the surrounding Zone A beyond terrain

> "For each region, we conduct GIS analysis that examines whether they align with terrain
> changes or previously existing boundaries that are unknown to the model, such as soil-survey
> lines, county lines, or the edges of map sheets using a boundary-agreement measure."
> Outcome 5: "the selected regions differ from the surrounding unselected Zone A regions beyond
> terrain." Contribution 3: a map of Florida, with boundary analysis against held-out maps.

This is G1 of `REGION-ANALYSIS-FLORIDA.md` §3, with three choices fixed by the paper.

**Terms, in CS language.** A *lobe* is one connected piece of a Zone A polygon (a multipart
polygon exploded into parts). An *elevation level set* is a curve along which the ground
elevation is constant, like a contour line. A detailed study's boundary is where the terrain
meets a computed water surface, so it is close to a level set. A *FIRM panel* is one printed
map sheet of the flood map; its edge is an arbitrary rectangle. A *SSURGO map unit* is one
polygon of the national soil survey; the lines between map units were drawn by soil
scientists, and no covariate in our model carries them (the StreamCat soil columns are
catchment averages). The *displaced-edge null* shifts a real boundary by a random 300–1,000 m
and re-measures, to estimate how often a curve lands near a soil or county line by chance.

1. **The object is the Zone A boundary inside each region**, sampled every 30 m. It is not the
   region's own outline, which the kNN graph draws through points 600–800 m apart and which has
   no precise line. (Paper wording: "whether they align" reads as the region's outline; "whether
   the Zone A boundaries inside them align" says what is measured.)
2. **Reference layers (the "held-out maps").**
   - *Terrain (class E):* elevation level sets from the 3DEP 10 m DEM. A 300 m window of boundary
     whose elevation range is within ±½ contour interval (0.76 m) and that has a real drop across
     it is a water surface: the rule the detailed labels teach, and learnable from HAND and TPI.
   - *Lines the model never saw:* SSURGO soil map-unit lines (class S); county lines and FIRM
     panel edges (class M).
   - *Optional if time:* NWI wetland lines (class W, a different rule on partly-seen covariates),
     PLSS section lines, roads and canals, quad-contour phase (class C).
3. **Contrasts.**
   - *Main comparison for O5:* size-matched unselected Zone A lobes in the same tiles.
   - *Rule contrast:* Zone AE lobes in the same tiles. They should be level sets.
   - *Base rate:* the displaced-edge null, 200 draws. Soil lines are dense in Florida, so every
     number is reported as excess over this null.

**Primary statistic (fix it before looking).** Share of region Zone A boundary length that is
not a level set and lies within 100 m of an S or M line, minus the same share for matched
unselected Zone A lobes, with its displaced-edge null percentile. "Beyond terrain" means the
comparison is made on the part of the boundary that is not explained by a level set.

**Reading.**
- Positive: the regions are where the approximate label follows lines the covariates cannot
  see. That is the paper's hypothesis.
- Zero, with region lobes and unselected lobes both on S/M lines: Zone A is that kind of label
  everywhere, and the regions are where it meets terrain the Zone AE models never saw
  (`REGION-ANALYSIS-FLORIDA.md` §8, last row). Still reportable; outcome 5 is then not met as
  written.
- Region boundaries that are level sets like Zone AE's: the approximate map may be right and
  the models wrong (Story 2 in `REGION-ANALYSIS-FLORIDA.md` §4).

**The label-noise check from Related Work.** "...we test whether memorized noise explains our
inversions" (the paper's current wording, citing Sanyal et al. 2025). Their inversion needs
models that interpolate noisy training labels. Our models train on Zone AE only, so the check asks
whether models memorizing noise in their Zone AE training labels explain the inversion; it cannot
tell whether the Zone A labels they are scored on are noisy. So record each model's training
accuracy and check whether the inversion survives among models that do not interpolate (shallow
trees, LR, NB). **Needs per-model training metrics, which the cells do not store today.**

**The map (Contribution 3).** Florida with Zone A / AE, the regions from all seeds, and each
region's boundary coloured by class. The Flood Insurance Study (FIS) reports for the region
counties (Bradford, Union, Okeechobee, Indian River, Columbia, Miami-Dade) are the documentary
check on how each county's Zone A was drawn.

### Controls the paper needs to defend O3–O5

| control | why | status |
|---|---|---|
| seeds 1–2 (tile split) | one seed and one search per budget; the classic curve is non-monotone | not run |
| null-select | the optimizer can manufacture an inversion from noise | not in the paper protocol yet |
| family-disjoint split | a random model split shares one quality axis | not in the paper protocol yet |
| random and most-misclassified baselines | inversion is not just difficulty | have (seed 0) |
| interpolation check | label-noise inversion requires interpolation (Sanyal et al.) | needs training accuracy per model |
| base rate in O1 | ID 0.53 vs OOD 0.64 positives | report AUC |

---

## 4. Data each experiment needs

This table is the checklist for `docs/TASK-DATA-AUDIT.md`. The status column is the only part
the audit should edit.

| data | used by | where | status (2026-10-15) |
|---|---|---|---|
| Florida sample rebuilt with protocol v2 (§2a), covariates re-pulled | all | `data/processed/dataset_FL.parquet` | done 2026-10-14: 80,000 points; Earth Engine and NHDPlus covariates re-pulled for the new points, Atlas 14 (1,629 cached blocks at v1 block-mean positions) and StreamCat (per-county cache of 2026-09-19) from caches. Atlas 14 missing for 191 points (0.24%, 35 new blocks; every request failed, PFDS answered HTTP 503 to manual checks 23:12–00:14): refill when it answers (open-water blocks may still fail), then rerun from step 7 with `--overwrite` (DATA-AUDIT §12). 2026-10-21: the code review fixed the sampler and the Earth Engine and NHDPlus covariates, so this table is stale until rebuilt (`docs/DATA-SUMMARY.md` §7; `docs/RUN-FL-V2.md`, "Still to run") |
| NHDPlus V2 catchments and flowlines for Florida (03N, 03S, 03W) | covariates for the new points | `data/raw/nhdplus/` | on disk: 03N/03S/03W (547 MB download, 2.0 GB with the extracted units and `NHDPlusV21_FL.gdb`); the gdb reproduces v1's COMIDs exactly and its distances within 7.4e-5 m (smoke test) |
| point table: 45 features, label, domain, neg_protocol, point_id, lon/lat, tile_50km, county_fips, meta_* | all | `data/processed/dataset_FL.parquet` | done: 80,000 rows, 85 columns incl. `meta_src_row_id`, `meta_neg_rank` (86 after the post-review rebuild, which adds `meta_point_county`); audited (`results/2026-10-14/data_audit/`: 12 OK, 1 WRONG (B7: 2 AE positives up to 214 m outside the TIGER state line, accepted), 6 DECIDE (B2, B3, B4, B14, A4.2, A4.3)) |
| feature → source → script → resolution table | Methods ("seven public sources") | — | done: `docs/FEATURE-SOURCES.md` (seven sources, 45 features) |
| how band negatives get a zone | Methods | `sample_points.py` | done: Methods text in `docs/DATA-AUDIT-2026-10-14.md` §5 |
| cell: per-model per-point scores on ID and OOD halves, labels, lon/lat, tile, county | O1–O4 | `data/interim/aotil/cells/FL_band_AE_s0.*` | done v2 seed 0 (audit PASS); seeds 1–2 still to run |
| `point_id` for both halves of the cell; lon/lat for the ID half | joining regions to polygons and features without rounding lon/lat | — | done (`point_id`, `point_id_id`, `lon_id`, `lat_id` in the v2 cell) |
| per-model hyperparameters, seed, features used, training accuracy/AUC | interpolation check; Methods appendix (model grid) | — | done in the v2 cell (`hp_*`, `seed`, `features`, `n_train`, training acc/AUC); AUC tie bug fixed |
| selection outputs per budget (theta, result, trials, meta), classic and spatial | O3, O4, regions | `results/2026-09-24/oodselect_paper/` | done v2 seed 0, classic + spatial μ=3 (`results/2026-10-14/oodselect_paper/`, audit 33/33) |
| region table (point_id → component, cell, variant, S, seed) | O5, map | — | done v2 seed 0 (`results/2026-10-14/regions/`; each row carries `cell`, `variant`, `S`, `seed`); v1 reproduction checked |
| Zone A / AE polygons with geometry, FLD_AR_ID, FLD_ZONE | lobes and edges | `data/interim/prep/haz_FL_5070.parquet` (5.4 GB) | checked (`check_gis_inputs.py`) |
| region lobes, matched unselected Zone A lobes, Zone AE lobes; 30 m edge-vertex table | O5 | — | done for spatial S = 250, 500 (`results/2026-10-14/boundary/`); the matched-lobe target (3 per region lobe) is not met in 7 of 12 components, and some regions are mostly band negatives (S = 250 Palm Beach: 0 positives) |
| per-point extras: `meta_firm_pan`, `meta_near_fld_ar_id`, `edge_dist_m` | O5, map | — | done: `data/interim/analysis/point_extras_FL.parquet` (`meta_edge_dist_m`) |
| 10 m DEM over the analysis areas (elevation at vertices and at displaced vertices) | O5 class E | 3DEP 1/3″ (TNM tiles or GEE `USGS/3DEP/10m`) | done: 3DEP 1/3″ rasters in `data/interim/gis/dem/`; z, z_in, z_out at every vertex; 6 tiles newer than 2024, their version against GEE `USGS/3DEP/10m_collection` unchecked |
| SSURGO map-unit polygons + `muaggatt` (flodfreqdcd, hydclprs, drclassdcd, hydgrpdcd) | O5 class S | Soil Data Access | done: `data/interim/gis/ssurgo/` (197,241 unique polygons; lines with attribute differences). W7 pulls (`results/2026-10-16/gis/ssurgo/`): topology warnings for seed 0 in S = 250 components 0–3 and S = 500 components 0, 1 and 3, and in component 1 of the seed-1 and seed-2 recurring regions (`logs/2026-10-16/C6_ssurgo*.log`). Cause: split pieces, and survey joins written twice where two polygons overlap by a sliver; effect on `d_soil_*` under 0.2 m. Since 2026-10-21 `pull_ssurgo.py` also counts the doubled joins and warns on them (seed-0 S = 500 component 4: about 1.8 km, unwarned before); rerun step 14 of `docs/RUN-FL-V2.md` with `--overwrite` for those log lines |
| county boundaries | O5 class M | TIGER/Line | done: `data/interim/gis/county_lines_FL.parquet` (TIGER 2026) |
| FIRM panel polygons (map-sheet edges) | O5 class M | `data/interim/nfhl_states/pan_FL.parquet` | done: `data/interim/gis/panel_edges_FL.parquet` (per DFIRM) |
| NWI wetlands, PLSS sections, roads, canals | O5 optional classes | USFWS, BLM, Census, USGS | canals from NHDPlus in the vertex-layer table; NWI, PLSS, roads not built |
| FIS reports, six region counties | O5 documentary check | FEMA Map Service Center | manual (Victor); the v2 regions are in nine counties: Hamilton, Miami-Dade, Palm Beach, Dixie, Okeechobee, Indian River, Marion, Nassau, Duval |

---

## 5. Out of scope, or not in the paper

- Iowa / Missouri (E6), the provenance graph (E5), the ten-state transfer matrix.
- The A→AE direction. Its seed-0 results are kept; nothing new is built for it unless the
  paper adds it.
- The full ponding analysis (`REGION-ANALYSIS-FLORIDA.md` §4.2), DISDE, R6 lobe excision. R7
  (membership model) is cheap and can be the comparison if space allows.
- The shadedX negative protocol (the sensitivity arm; the paper never mentions it).
- RQ4, observed flooding.

---

## 6. Order of work

Assumes the project ends mid-November; shift the weeks if the deadline differs.

| week of | do |
|---|---|
| Oct 13 | rebuild Florida with sampling protocol v2 and re-pull its covariates; data audit; seed-0 cell and classic + spatial selections on v2; `aotil/regions.py`; the missing GIS collectors (all in `docs/TASK-DATA-AUDIT.md`) |
| Oct 20 | **v5:** the full run with the seven Woznicki features, `docs/TASK-FULL-RUN-W7.md` Parts A–C5: seeds 0–2 cells; classic and spatial selections; null-select and family-disjoint split (code in B1); paper-protocol μ sweep at three budgets; smoothed-classic ablation; regions and their recurrence |
| Oct 27 | `docs/TASK-FULL-RUN-W7.md` C6–C7: lobes, DEM and SSURGO for the new region boxes; the boundary statistic (code in B3) with the displaced-edge null, matched unselected lobes and the Zone AE contrast; the report |
| Nov 3 | the Florida map (C3); interpolation check; FIS reading; Methods and Results draft |
| Nov 10 | Discussion, figures, trim to length; presentation |

**Minimum viable paper:** O1–O4 on three seeds with null-select; regions on seed 0 checked for
recurrence; the boundary statistic with the S and M layers, terrain level sets, the matched
unselected lobes and the displaced-edge null; one map.

---

## 7. Limitations to carry into the paper

- **The label is a model output, not observed flooding.** The project studies agreement with
  FEMA maps.
- **Negatives tie the label to distance from the label.** Band negatives sit 500–5,000 m from
  any SFHA.
- **One state, one mapping regime.** Florida Zone A is about 80% `NOT_POPULATED` classic
  approximate study.
- **Pseudo-replication.** The largest 1% of polygons supply 8–26% of a zone's points.
  Bootstrap at the tile level.
- **Flat terrain blurs boundary position.** At the regions' 0.2–0.3° slopes, ±0.76 m of
  vertical tolerance is ±150–200 m horizontally. Use the 100 m coincidence band and classify
  300 m windows, never single vertices.
- **The spatial penalty is our modification.** μ = 0 must reproduce the classic run, and μ is
  never chosen by test r.
- **Susceptibility is not hazard or risk.** No return period beyond the 1% label, no exposure.

---

## 8. Where things are

- Paper: `CS371_Cho_Li/acl_latex.tex`, `custom.bib` (gitignored; Overleaf is its version
  control). Cited PDFs in `refs/cited/`.
- Model features (v5): `data/processed_w7/FEATURES.txt` (7), with `data/processed_w7/dataset_FL.parquet`
  linking to the table below; cells `data/interim/aotil/cells_w7/`.
- Data: `data/processed/dataset_FL.parquet` + `FEATURES.txt` (the 45-feature list, v4); polygons
  `data/interim/prep/haz_FL_5070.parquet`; panels, BFE lines, LOMRs
  `data/interim/nfhl_states/{pan,s_bfe,s_lomr}_FL.parquet`; flowlines and StreamCat
  `data/interim/hydro/`; the raw NFHL gdb `data/raw/nfhl/NFHL_Key_Layers.gdb.zip` (19 GB).
- Models and selections: cells `data/interim/aotil/cells/FL_band_AE_s0.*` (v2, 45 features; the
  v1 cells, `FL_band_A_s0` included, are in `data/interim/archive/sampling-v1/`); selections
  `results/2026-09-24/oodselect_paper/{classic,spatial_mu3_knn10}/` (v1),
  `results/2026-10-14/oodselect_paper/` (v2) and `results/2026-10-16/oodselect_paper/` (v5); harness
  `run_oodselect_local.sh` (`aotil/README.md`).
- Specs: `docs/REGION-ANALYSIS-FLORIDA.md` (boundary analysis), `docs/SPATIAL-OODSELECT.md`
  (Laplacian term, coherence metrics, validation rules), `docs/COVARIATES.md`,
  `docs/LABELING-PROBLEMS.md`. The data as built: `docs/DATA-SUMMARY.md`,
  `docs/DATA-AUDIT-2026-10-14.md`. `docs/REGION-ANALYSIS.md` (v1) is history; the v1 ten-state
  notes (`COVARIATE-AUDIT.md`, `SAMPLING-RESULTS.md`, `ASSEMBLY-SHORTCUTS.md`, `DOWNLOAD-GUIDE.md`)
  are in `Claude outputs/removed-2026-10-21/docs/`.
- claude.ai Project docs: `claude/project-index.md` (pointer doc),
  `claude/experiment-plan.md` (mirror of this file).
