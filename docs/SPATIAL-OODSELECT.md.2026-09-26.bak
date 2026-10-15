# Spatially regularized OODSelect — method and the full Florida run

**v1, 2026-09-19** · companion to `docs/PROJECT-PLAN.md` v3 §5 Stage 3 · pilot in
`aotil/README.md` · abstract in the Project doc `claude/project-abstract.md`

This is the implementation plan for the run that follows the Florida pilot. It
says what to change, why, and how hard each piece is, in the order to do it.
Nothing here has been implemented yet; `aotil/oodselect.py` is still the
unconstrained pilot version.

---

## 0. Why the method changes

The pilot ran OODSelect (Salaudeen et al., arXiv:2510.24884) as reimplemented in
`aotil/oodselect.py` on the Zone A / Zone AE benchmark for Florida. Under the
`band` negative protocol the inverse line is real: r = −0.80 to −0.90 on held-out
models, the most-misclassified baseline stays positive, the permutation null
sits near zero, and it holds across family-disjoint model splits. Under
`shadedX` the inversion is not separable from difficulty (`results/aotil_controls_FL.csv`).

The subsets are describable from quarantined provenance metadata — AE→A selects
Zone A positives from the NOT_POPULATED regime within ~30 m of the SFHA
boundary; A→AE selects coastal-floodplain polygons at 92% against a 23% base
rate — but they are **geographically scattered**: at the 5–10% budget they touch
15–22 of the 23 evaluation tiles, and at 25% all 23. A set like that cannot be
handed to a GIS analysis, which reads hydrological and land-use context off a
*place*. So the selection gets a spatial smoothness term and the sets become
regions, at a measured cost in inversion strength.

**Decisions recorded 2026-09-19** (do not reopen without a reason):

| decision | choice |
|---|---|
| headline claim | label provenance is the distribution shift; the spatial method is the instrument, not the headline |
| why spatial coherence | the GIS explanation requires physical regions, not scattered points |
| negative protocol | `band` primary; `shadedX` is the sensitivity arm only |
| scope | Florida primary; Iowa/Missouri (BLE risk-class, redelineation regimes) as the regime contrast if time allows |
| RQ4 (observed flooding) | out of the deliverable |
| objective | the paper's correlation objective; the OOD>ID variant from v2 is dropped |
| wording | "data construction / label production", not "preprocessing"; a graph-**Laplacian** penalty, not Lagrangian; the outcome is a coherence–inversion tradeoff, never "gives coherent sets" |

---

## 1. What "the full Florida run" means

The pilot was 126 models × 4 cells × one tile split, fractions 5/10/25%, in the
pilot chat's cloud container. The full run is:

1. Cells regenerated **with spatial columns saved** (§2), from the current
   `data/processed/dataset_FL.parquet` (regenerated 2026-09-19 13:11), persisted
   in the repo rather than a container.
2. Model population ≥ 300 per cell; stop adding when the aggregate ID→OOD
   correlation on the full OOD set moves < 1% (the paper's rule; check with
   `gate.py` at 150 / 225 / 300).
3. Three tile-split seeds (0, 1, 2), so every headline number carries a spread.
4. E0–E4 below on the two `band` cells; `shadedX` cells run E1 only, as the
   sensitivity arm.
5. Results under `results/aotil/` with the method in the file name; figures
   under `results/aotil/fig/`.
6. E5 (provenance graph) and E6 (IA/MO) only if E0–E4 are done by week 5.

---

## 2. Prerequisites — the cells

`zoo.py` writes `S_ood`, `S_id`, `y_ood`, `tiles_ood` and nothing about location.
Add to the `.npz`, straight from `ood_te`:

    lon, lat, block_id, tile_50km, county_fips, meta_comid, meta_source_cit

and keep `characterise.py`'s frame reconstruction as the cross-check (`assert
len(ood) == S.shape[1]` already exists). Column names confirmed in
`sample_points.py` / `pull_hydro.py`: `lon`, `lat`, `block_id`, `tile_50km`,
`county_fips`, `meta_comid`, `meta_source_cit`, `meta_eff_date`,
`meta_fld_ar_id`, `meta_study_regime`, `meta_study_typ`, `meta_zone_subty`,
`meta_panel_scale`, `meta_panel_typ`, `meta_bfe_km_per_km2`.

Where the cells live: the pilot wrote them to `/home/claude/pilot/cells` inside
the chat's container, which is ephemeral — that is why none are in the repo.
Write them to `data/interim/aotil/cells/` (a 300-model cell is ~20 MB) and
commit them back to the folder after every zoo run. Re-stage the parquet before
running: the dataset changed after the pilot.

Effort: one line in `zoo.py`, plus the re-run.

---

## 3. The method

### 3.1 Objective

The pilot solves, with w = σ(θ) ∈ (0,1)^d and a(w) = S·w / Σw,

    min_θ  corr(x, a(w)) + λ (S_target − Σw)²

Add one term:

    min_θ  corr(x, a(w)) + λ (S_target − Σw)² + μ · P(w)

    P(w) = wᵀ L w / (S_target · k) = Σ_{(i,j)∈E} (w_i − w_j)² / (S_target · k)

where L = D − A is the Laplacian of a symmetric spatial adjacency A over the OOD
evaluation points, and k is the neighbours per point. The normalisation makes
P an **incoherence score in [0, ~1]**: a selection of S_target points none of
which are adjacent cuts ≈ S_target·k edges, each contributing 1, so P ≈ 1; a
compact region cuts only its perimeter, so P ≪ 1. μ then trades correlation
units against incoherence units, and the same μ means the same thing in every
cell. μ = 0 is the pilot exactly.

### 3.2 Gradient

∂P/∂w = 2 L w / (S_target · k). In `select()`, after the cardinality term:

    gw += mu * 2.0 * (L @ w) / (s_target * knn)     # spatial smoothness
    gt  = gw * w * (1 - w)                          # existing sigmoid chain

L is `scipy.sparse.csr_matrix`; the extra cost is one sparse mat-vec per
iteration — nothing against the N×d dense mat-vec already there. Keep the cosine
schedule on lr and the 10× ramp on λ unchanged; μ is constant through training.
Discretise as now (top-S_target of w) and report P on the hard mask as well as
on w.

### 3.3 The graph (`--graph`, `--knn`)

Build once per cell over the OOD evaluation points. Project lon/lat to metres
first (pyproj to EPSG:3086 if available; otherwise x = R·lon·cos(lat₀),
y = R·lat is fine at kNN scale). `scipy.spatial.cKDTree(xy).query(xy, k+1)`,
drop self, symmetrise (A = (A + Aᵀ) > 0), L = diag(A·1) − A.

| graph | edges | use |
|---|---|---|
| `knn` (default, k = 10) | k nearest OOD points, geometric | E0–E4. Points exist only inside flood corridors, so kNN balls follow the corridor — this handles ribbons, which a Euclidean-radius window does not |
| `block` | kNN restricted to the same `block_id` (10 km) | coarse control; approximates unit aggregation |
| `catchment` | kNN restricted to the same `meta_comid` | hydrological units — how a hydrologist would define "a place" |
| `provenance` | kNN restricted to the same `meta_source_cit` (fallback `county_fips`) | E5 — smoothness only within one study. The comparison that tests the thesis |

Restrict-within-group always means kNN *among same-group points*, never a
clique: a 2,000-point county as a clique is 2M edges for nothing.

### 3.4 Choosing μ

Never by test r — that is selection on the outcome. Report the whole curve
(§5 E1) and mark one **coherent operating point** chosen on *selection-model*
coherence: the smallest μ at which kNN purity (§4) ≥ 0.5 on the selection
models. Everything reported at that point is computed on the test models.

---

## 4. Coherence metrics (report for every subset, every method, every control)

For a hard mask M of size S with adjacency A:

| metric | definition | random baseline |
|---|---|---|
| `tiles`, `blocks` | distinct `tile_50km` / `block_id` touched (tiles already reported) | ≈ all |
| `purity` | mean over i∈M of \|N_k(i) ∩ M\| / k | ≈ S/d = f |
| `P_hard` | cut edges / (S·k) | ≈ 1 − f |
| `n_cc`, `largest_cc_frac` | connected components of the induced subgraph A[M][:,M] (`scipy.sparse.csgraph.connected_components`); largest as a fraction of S | many; small |
| `moran_I` (optional) | Moran's I of the indicator 1_M with weights A | ≈ 0 |

A random subset of the same size is the baseline for every one of these, so the
random control already in `analyse.py` doubles as the coherence null.

---

## 5. Experiments

### E0 — Does any coherent inverse region exist? (half a day, before anything else)

Covariance is additive over points:
Cov_m(x_m, mean_{i∈S} S_mi) = (1/|S|) Σ_{i∈S} Cov_m(x_m, S_mi). So compute, on
the **selection** models only,

    c_i = Cov_m(x_m, S_mi)        for every OOD point i

and the slope of any region is mean(c_i in region) / Var(x). Save
`results/aotil/FL_{cell}_pointcov.parquet` (c_i, lon, lat, block_id, tile), then:

- Moran's I of c_i under the kNN weights, against 200 permutations of c_i over
  points. If I is inside its null, c_i is spatial white noise and **no μ > 0 will
  produce a region** — record that and reframe (§7, outcome C) before spending a
  week on the sweep.
- Block means of c_i: count blocks with negative mean, and whether those blocks
  are themselves clustered (Moran's I at block level).
- Map c_i. This map is also the first explanation object: it can be regressed
  on GIS covariates directly.

Pearson r also divides by the spread of the subset mean across models, which is
not additive, so magnitudes differ from what the optimiser reaches; the sign is
set by the covariance, and the sign is what E0 answers.

### E1 — The tradeoff curve (the core result)

For each cell, fraction f ∈ {0.05, 0.10, 0.25}, seed ∈ {0,1,2},
μ ∈ {0, 0.1, 0.3, 1, 3, 10}:

- select on the selection models, pick the restart on the validation models,
  score r on the **test** models (the pilot's 60/20/20), plus the two
  family-disjoint splits (`trees→others`, `others→trees`);
- controls at every μ: `random` (same S), `hard` (most-misclassified on the
  selection models), **`null-select` run with the same μ** — the constraint
  changes what the optimiser can manufacture from noise, so the null must carry
  it too;
- record r_test and every §4 metric.

Figure 1: r_test (y) against purity or 1 − P_hard (x), one line per cell, μ
annotated, the controls as bands. μ = 0 reproduces the pilot; the curve says how
much of the inversion is place-organised and how much is a point type. Vanilla
OODSelect at f = 0.25 already fails for `band_A` (r = +0.57), so expect the
coherent operating point to sit at f ≤ 0.10.

### E2 — Held-out points inside the region (what vanilla OODSelect cannot do)

Within each test tile, split the OOD points into halves H1/H2 at random,
stratified by label. Build the graph and select on H1. Transfer to H2 by the
field: w̃_j = mean of w over the k nearest H1 points of j; region in H2 = top
f·|H2| by w̃. Score r on the test models using H2's region only.

Run at μ = 0 and at the coherent operating point. A place-organised inversion
survives the transfer; a point-type inversion (μ = 0) should collapse toward the
random control, because its w has no spatial structure to transfer. That
collapse is a result in itself and answers the standing critique of OODSelect —
that its subsets are fit to the very points they are evaluated on.

### E3 — Characterise the regions

`characterise.py` on the constrained masks, per connected component as well as
pooled, with the metadata columns extended by `meta_source_cit`,
`meta_eff_date`, `county_fips`, `meta_comid`, `meta_fld_ar_id`. Then the
two-stage explainer from the plan: a membership model (HGB and logistic) of
"in region" on each covariate group separately —

| group | columns |
|---|---|
| provenance | `meta_study_regime`, `meta_study_typ`, `meta_zone_subty`, `meta_panel_scale`, `meta_panel_typ`, `meta_bfe_km_per_km2`, `meta_eff_date`, `meta_source_cit` |
| terrain / hydrology | `hand`, `slope`, `tpi_*`, `log10_upa`, `log10_dist_river`, `log10_dist_coast`, `drainage_density`, `nearest_stream_order`, StreamCat `*cat`/`*ws` |
| climate | `precip8110*`, `tmean8110*`, `runoff*`, `p100_*`, `p2_24hr` |
| urbanicity / surface | `impervious`, `landcover`, `water_frac_900m`, `gsw_occurrence` |

— GroupKFold on `tile_50km`, AUC per group, against a **block permutation
null**: shuffle membership across `block_id` keeping block sizes, 200×, so the
null keeps the spatial autocorrelation a point permutation would destroy.
Provenance beating terrain at equal coherence is RQ3 answered in the
project's favour; the reverse is the honest alternative.

### E4 — GIS context of each region

Per connected component: a map with NHDPlus flowlines, coastline, county
boundaries and FIRM panel boundaries (`S_FIRM_PAN`), flood polygons coloured by
zone subtype; and a one-row summary — n points, positive share, dominant
`meta_zone_subty` / `meta_study_regime` / `meta_panel_scale` /
`meta_source_cit` / `meta_eff_date`, BFE density, mean HAND, imperviousness,
land-cover mix, distance to coast and river. The writeup pairs the description
a hydrologist would give (setting) with the one the metadata gives (how it was
mapped). This is the analysis the spatial constraint exists to make possible.

### E5 — Provenance graph vs geometric graph (time permitting; the thesis test)

Repeat E1 with `--graph provenance`. If the inversion at equal purity is
stronger when smoothness is confined to one study than when it is geometric,
the failure is organised by how the maps were made rather than by where the
land is. Report both curves on one axis.

### E6 — Iowa / Missouri (time permitting)

`zoo.py --state IA|MO --protocol band` both directions, then E1 and E3. Iowa:
84% of Zone A is BLE risk-class and 50% of AE is redelineation; Missouri is BLE
as well. These are the places where provenance differs *between places*, which
Florida (classic regime, ~78% NP) cannot show.

---

## 6. Validation — what must hold before any number is quoted

1. Every r is on models the selection never saw; restarts chosen on validation
   models only.
2. `null-select` at the same μ is near zero. If it inverts, stop.
3. The random control stays positive at every μ.
4. Family-disjoint splits agree in sign with the random split.
5. The coherent operating point is chosen on selection-model coherence, not on
   test r.
6. Spreads across the three tile-split seeds are reported, not the best seed.
7. Anything at the point level is bootstrapped at the tile level
   (pseudo-replication: 200 m spacing inside corridors, 1% of polygons supply
   8–26% of a domain).

---

## 7. Reading the outcomes

| outcome | reading |
|---|---|
| **A.** Inversion survives at the coherent operating point (r_test ≤ −0.5, purity ≥ 0.5), null flat, E2 transfer holds | the provenance failure is place-organised; the regions are the paper's objects and E3/E4 explain them |
| **B.** Inversion survives but weakened (−0.5 < r_test < −0.2) | partly place-organised; report the curve as the decomposition of the pilot's −0.8 into place and point-type components |
| **C.** Inversion dies for every μ > 0, or E0 shows c_i is spatial white noise | AoTIL here is carried by a point type (boundary points of one regime) present everywhere, not by places. Still a clean statement, consistent with the thesis, and the tradeoff curve is the figure; E3 then runs on the μ = 0 subsets as in the pilot |
| **D.** Provenance graph beats geometric graph at equal purity (E5) | the shift is in how the maps were made, measured directly |
| **E.** Terrain group beats provenance group in E3 | regions are geographic after all; the honest alternative, still reportable |

Outcomes A–D are all reportable; C is the negative result and it is written up
as such, not buried.

---

## 8. Effort, easiest to hardest

| step | change | difficulty |
|---|---|---|
| save spatial columns in cells | 1 line in `zoo.py`, re-run zoo | trivial (compute time only) |
| E0 covariance map + Moran's I | ~40 lines, one plot | easy, half a day |
| kNN graph + Laplacian term + `--mu --graph --knn` | ~25 lines in `oodselect.py` | easy |
| coherence metrics | ~30 lines (`csgraph.connected_components`) | easy |
| constrained `null-select` and controls at every μ in `analyse.py` | loop change | easy |
| tradeoff figure | plotting only | easy |
| E2 held-out-point transfer | ~60 lines | moderate |
| E3 membership model + block permutation null | ~80 lines | moderate |
| E4 maps with NFHL / NHDPlus overlays | geopandas plumbing | moderate |
| `provenance` / `catchment` graphs | grouping + within-group kNN | moderate |
| model population to 300 × 4 cells × 3 seeds | compute | time, not code |
| IA / MO | new zoo runs + E1/E3 | a week |

Model zoo cost scales linearly in models and seeds; run the `band` cells first.

---

## 9. Pitfalls specific to this data

- **Coordinates.** kNN in raw degrees is anisotropic; project first.
- **Boundary points.** The pilot's AE→A subset sits within ~30 m of the SFHA
  boundary. A region necessarily includes interior points too, which dilutes
  exactly that signal. This is the expected mechanism behind outcomes B/C, so
  measure it: `meta_dist_to_sfha_m` in the region vs the μ = 0 subset.
- **Top-S discretisation** at small μ can split w into many components; report
  `n_cc` and do not read a 40-component "region" as coherent.
- **Disconnected tiles.** The kNN graph is disconnected across test tiles by
  construction; that is fine and means multiple regions are the normal case.
- **`meta_source_cit` coverage.** Check missingness before using it as the
  provenance group; fall back to `county_fips`.
- **λ ramp vs μ.** λ rises 10× over training; μ is constant. If the cardinality
  drifts under large μ, raise `lam0` rather than schedule μ.
- **Do not select the state on gap size.** Florida was chosen for coverage
  (`aotil/README.md`); IA/MO are chosen for regime contrast, not for effect.

---

## 10. Deliverables

Figures: (1) tradeoff curve per cell with controls; (2) maps, μ = 0 vs coherent
operating point, same cell; (3) c_i map; (4) E2 transfer bars; (5) E3 AUC by
covariate group with null bands; (6) E5 two curves if run.
Tables: pilot vs full-run controls; per-region summary (E4); E3 enrichment.
Code: `aotil/oodselect.py` (+μ), `aotil/graph.py` (adjacency, metrics),
`aotil/pointcov.py` (E0), `aotil/transfer.py` (E2), `aotil/explain.py` (E3),
`aotil/maps.py` (E4). Every result file carries `{state}_{protocol}_{traindom}_s{seed}_mu{mu}_{graph}` in its name.
