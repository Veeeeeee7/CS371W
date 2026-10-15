# Region analysis — what to do with the coherent inverse-line regions (E3/E4 in detail)

> **Status 2026-10-13.** Superseded for Florida by `docs/REGION-ANALYSIS-FLORIDA.md` (v2) and
> `docs/PROJECT-PLAN.md` v4. Kept for the record; do not build from it.

**v1, 2026-09-26** · companion to `docs/SPATIAL-OODSELECT.md` §5 (supersedes its E3/E4
sketches) and `docs/PROJECT-PLAN.md` v3 · reads `results/2026-09-24/oodselect_paper/`
(paper protocol, classic vs spatial μ = 3) and `results/2026-09-19/aotil/` (job 562664)

This document says what analyses to run on the regions that spatially regularized
OODSelect returns, why each one needs a *region* rather than a point set, how to read
each outcome, and in what order to do them. It is written so that the spatial
regularizer is justified by the analyses — they cannot be done without regions — and
not the other way round.

---

## 0. The framing problem and its resolution

### 0.1 Why the two obvious analyses do not need regions

**The per-point membership model** (E3 as first sketched: fit "in subset" on named
covariates, GroupKFold on `tile_50km`, block-permutation null) works as well on 500
scattered points as on eight corridor segments — arguably better, since scattered points
are closer to independent draws. It tests whether the inverse points are *different
places*. It cannot say that a label is wrong, and it would never fail for lack of
coherence, so it cannot justify coherence.

**"Train a mapper on the problematic points and compare it to one trained on the
non-problematic points"** has the same flaw and adds circularity: the points were
selected *because* model populations behave differently on them, so a model trained on
them will behave differently. Nothing new is learned, and no geometry is used.

The test for any candidate analysis in this document: *would it be impossible, or
meaningless, on a scattered point set of the same size?* If not, it does not belong
in the justification of the spatial term (it may still be worth running — R7 is).

### 0.2 Where the hypothesis lives

The abstract's hypothesis is that "the regions where accuracy inverts follow how the
maps were made rather than the landscape". Maps are made in areal units, each with
boundaries and dates:

| unit | where it is in the data |
|---|---|
| DFIRM (countywide study; 99.86% of rows) | `dfirm_id` on every point |
| FIRM panel | `data/interim/nfhl_states/pan_FL.parquet`: `FIRM_PAN`, `EFF_DATE`, `PRE_DATE`, `SCALE`, `PANEL_TYP` (points carry `meta_eff_date`, `meta_panel_scale`, `meta_panel_typ` from the join at `sample_points.py:628`) |
| study / source | `meta_source_cit` (mostly `<NA>` in FL), `meta_study_typ`, `meta_study_regime` |
| flood-hazard polygon | `meta_fld_ar_id` (positives only), geometry in `data/interim/prep/haz_FL_5070.parquet` |
| hydraulic reach | `meta_comid`; flowlines in `data/interim/hydro/river_FL.parquet` (`COMID, LENGTHKM, StreamOrde, TotDASqKM, SLOPE, FTYPE`) |
| detailed-study trace | BFE lines, `data/interim/nfhl_states/s_bfe_FL.parquet`; `meta_bfe_km_per_km2` per DFIRM |

A point set has no boundary and covers no unit: it cannot "coincide with a panel edge",
"take a whole reach", or "be drawn like a buffer". So the hypothesis is testable only on
regions. **That is the direction of justification:** the question requires regions →
the regularizer is the instrument that produces them → the coherence–inversion
tradeoff (E1) shows the instrument is nearly free.

Two measured facts already support this framing and should open the paper's analysis
section:

1. **No single point set is "the" explanandum.** Classic and spatial masks overlap at
   Jaccard 0.00–0.05 for S ≤ 250 (0.13–0.22 at 500–750) yet invert about equally
   (`claude/first-experiment-oodselect-paper.md`). Many point sets carry the inversion;
   the *place* is the stable object, and the spatial run picks the instance that forms
   one.
2. **The inverse property is place-organised whether or not μ is on.** E2: selection on
   half the OOD points transfers to the other half at every μ including 0 (f = 0.10:
   r −0.69 vs random +0.85), because c_i is autocorrelated at the 200 m–2 km scale. μ
   buys coherence, not generalisation — and coherence is exactly what the analyses below
   consume.

### 0.3 Three things "Zone A is different" can mean

| | hypothesis | what it would mean for the mapping task | separated by |
|---|---|---|---|
| **H1** | covariate shift — the inverse regions are different *places* (terrain, hydrology, water proximity) | the model is being asked to extrapolate; not a label problem | R7 (point level), R1 landscape breaks |
| **H2** | label bias — Zone A boundaries in these regions are *systematically* mis-drawn (the approximate method's signature: buffers, fixed widths, ignored terrain) | the label is wrong in a structured way a model can learn and reproduce; pooling A into training teaches it | R1 map breaks, R2, R3, R4 (run structure), R6 |
| **H3** | label noise — iid errors from estimation | pooling adds noise, not bias; more data still helps | R4 (white residual), R2 (fragmentary coverage) |

H2 vs H3 is the distinction the mapping task cares about, and it is the one only regions
can make. H1 is the geographic confound (`water_frac_900m` doing provenance/geography
double duty) and is tested at point level as before. A fourth reading — the inversion is a
property of the model population (a shared quality axis), not of the labels — is handled
by the existing controls (null-select, most-misclassified gap, family-disjoint splits) and
is not repeated here.

### 0.4 Rules that apply to every question

- **Region objects.** Connected components of the top-S mask on the kNN graph
  (`aotil/graph.py`), analysed per component and pooled. A component under 20 points is
  listed but not tested on its own.
- **Three region sets, always side by side:** *classic* (μ = 0), *spatial* (μ = 3), and
  *smoothed-classic* — the μ = 0 weights passed through the E2 kNN-carry operator and
  thresholded to the same S. Smoothed-classic is the post-hoc ablation the paper's
  Discussion promises, and it is the real competitor to the regularizer. The regularizer
  earns its place where spatial beats smoothed-classic on region *quality* (R1
  alignment, R2 unit purity) at equal r — not on r itself.
- **Nulls are geometry-matched, never point permutations.** Primary: the
  *matched-growth null* — from a random seed in the same tile, grow a connected subgraph
  on the same kNN graph by BFS to the same size; 200 draws; report the observed
  statistic's percentile. It uses the same points and the same graph, follows corridors
  automatically, and matches size and tile. Secondary: *shift null* — slide the region's
  own point set along its corridor by a random offset (preserves shape). For R7 the
  block-permutation null from `docs/SPATIAL-OODSELECT.md` §5 E3 stays.
- **Circularity.** Regions are selected on the *selection* models' correctness at these
  points. Anything that reports a model score inside a region (R4) uses *test* models
  only; anything that uses region membership to change *training data* (R6) uses the E2
  split — select on one half of the tiles, carry, act on the other half — and evaluates
  on tiles never used in selection.
- **Both directions.** AE→A is primary (train on the detailed labels, test the
  approximate ones). Run every question on A→AE as well: answers that flip sign between
  directions are about the labels; answers that do not flip are about geography.
- **Seeds.** Regions are only objects if they recur. Run seeds 1–2 under the paper
  protocol first (already on the next-steps list); report components that recur across
  seeds (chance-normalised Jaccard, `graph.py`) as primary and the rest as
  seed-specific.
- **Naming.** Every output carries `{cell}_S{S}_{variant}_R{k}` — cell, budget, region
  set (`classic`, `spatial_mu3_knn10`, `smoothed`), component index.

---

## 1. R0 — Regions and units: the shared machinery

Build once, in a new `aotil/regions.py`; every question below reads from it.

1. **Components.** Load a mask (`_theta.npy` → top-S) and its graph; label components;
   per component write point ids, n, positive share, tiles touched, bounding box, and
   the per-point test-model mean score (for R4).
2. **Unit tables.** Point → reach (`meta_comid`), polygon (`meta_fld_ar_id`), panel,
   DFIRM (`dfirm_id`), study (`meta_source_cit` → fallback `dfirm_id` × `meta_study_typ`
   where the citation is `<NA>`). Two columns have to be added to the dataset:
   - `meta_firm_pan` — the panel id; the join at `sample_points.py:628` already touches
     the panel table and keeps only `EFF_DATE`, `SCALE`, `PANEL_TYP`.
   - `meta_near_fld_ar_id` — nearest SFHA polygon for *negatives* (band negatives have
     `meta_fld_ar_id = <NA>`); the STRtree nearest query that produces
     `meta_dist_to_sfha_m` already finds it.
   - `edge_dist_m` — distance from a positive to its own polygon's boundary.
     `meta_dist_to_sfha_m` is 0 for every positive by construction (the sampling check
     at `sample_points.py:853` requires it), so the "within ~30 m of the boundary"
     mechanism from §9 of the method doc can only be measured with this column.
3. **Along-network coordinate.** Project each point onto its reach's flowline →
   (`comid`, distance along). `river_FL.parquet` has no from/to node ids, so chain
   reaches geometrically (endpoints within 1 m) or pull `PlusFlow`; per-reach is enough
   for R1/R2 because panel and study breaks are at reach scale.
4. **Break catalogue** (for R1), two classes:
   - **M (map-production):** panel edge (panel geometry), DFIRM edge, change of
     `meta_eff_date`, change of `meta_source_cit` / `meta_study_typ`, change of
     `meta_panel_scale`, A→AE transition along the corridor, onset of BFE lines
     (`s_bfe_FL.parquet` — the first BFE line is where the detailed study begins).
   - **L (landscape):** confluence (flowline node with more than one upstream reach),
     change of `StreamOrde`, change of `channel_slope` beyond a threshold, coast.
     HUC-12 divides need the WBD, which is not on disk — optional.
   - A→AE transitions are *both* a map break and, often, a confluence of a studied and
     an unstudied stream; report them as their own class rather than folding them into
     M or L.
5. **Null samplers.** `matched_growth(size, tile, n_draws)` and `shift(region,
   n_draws)` as in §0.4.

Effort: moderate — one to two days, and everything else is a groupby on top of it.

---

## 2. The research questions

Each question below states: what it asks, the hypotheses and how each outcome reads,
why it needs regions, the data it uses and what must be added, the procedure, the null,
the metrics, pitfalls, and effort.

### R1 — Do the regions start and stop at map-production breaks or at landscape breaks?

**The direct test of the abstract's hypothesis.**

*Question.* Where a region ends along the stream network, is that endpoint a place
where the *map* changes (panel, study, effective date, scale, A→AE, BFE onset) or where
the *landscape* changes (confluence, stream order, slope, coast)?

*Hypotheses.* H2 → endpoints align with M breaks far above the null; H1 → with L
breaks; neither above the null → the region is bounded by something else (check the
boundary-point mechanism with `edge_dist_m`, and whether the component is simply
budget-limited — see pitfalls).

*Why regions.* A scattered set has no endpoints. Alignment of a boundary with another
boundary is a property only a region has.

*Data.* R0 break catalogue; region endpoints = region points that have at least one
non-region kNN neighbour along the same reach, reduced to the two extreme along-network
positions per reach segment.

*Procedure.* For every endpoint, the along-network distance to the nearest break of
each type; indicator "within d" for d = 200 m (one spacing) and 500 m. Aggregate per
component and pooled. Repeat on the matched-growth null (same size, same tile, same
graph), 200 draws.

*Metrics.* Per break type: observed alignment fraction, null mean, excess = observed −
null mean, and the observed value's percentile in the null. Summary statistic
pre-registered as primary: **excess(M) − excess(L)**, pooled over components, with its
null percentile. Also report per-type odds ratios so a single dominant break type
(e.g. panel edges) is visible.

*Reading.* excess(M) ≫ excess(L) → provenance-bounded; the reverse → landscape-bounded
(H1, still reportable and the honest alternative); both ≈ 0 → regions are not bounded
by either catalogue (say so; R2 and R4 then carry the argument).

*Pitfalls.* (1) Break types co-occur: panels are gridded, so panel edges cross every
corridor at regular intervals; the null carries the base rate, but report base rates
explicitly. (2) Short components have endpoints close to everything — that is why the
null is size-matched; do not pool components of very different lengths without
weighting by length. (3) A component can end because the budget S ran out, not because
anything changed; compare endpoints of the S = 250 and S = 500 regions — an endpoint
that moves with S is a budget artefact, one that stays is a boundary. (4) Coastal
components (`meta_zone_subty`, `log10_dist_coast`) end at the coast by construction;
analyse riverine and coastal components separately.

*Effort.* Moderate: the break catalogue is most of it (a day); the test itself is
short.

### R2 — Do the regions take whole units or fragments of them?

*Question.* For every reach / polygon / panel / DFIRM a region touches, what fraction
of that unit's OOD points lie inside the region?

*Hypotheses.* H2 → coverage is high and bimodal: units are in or out, because the
failure belongs to the study that drew them. H3 → coverage is low and uniform: errors
are scattered inside units. H1 → intermediate, and coverage follows terrain inside the
unit (the covered part of a reach is its low-HAND part, say).

*Why regions.* "Coverage of a unit" is defined for any point set, but a scattered set
covers every unit it touches at a few percent; only a coherent set can cover a unit
wholly, so the statistic is informative only with regions. This is also the metric on
which the regularizer should beat post-hoc smoothing: a jointly optimised selection can
trade a few points to complete a reach; a smoother cannot.

*Data.* R0 unit tables. Units: reach (`meta_comid`), polygon (`meta_fld_ar_id` for
positives, `meta_near_fld_ar_id` for negatives), panel (`meta_firm_pan`), DFIRM.

*Procedure.* Per unit type: coverage per touched unit; the coverage histogram;
**unit purity** = share of region points that sit in units covered ≥ 80%;
**bimodality** = share of touched units with coverage > 0.8 or < 0.2. Coverage is
bounded by S — a 500-point region cannot cover a 5,000-point polygon — so also report
coverage *relative to the maximum achievable at that S* (min(1, S / unit size)).

*Null.* Matched-growth null for spatial and smoothed-classic; for classic, matched
random point sets of the same size in the same tiles (its own natural null).

*Metrics and reading.* Unit purity and bimodality with null percentiles, for the three
region sets on one axis. Spatial ≫ smoothed-classic ≫ classic on unit purity at equal
r → the regularizer does more than smoothing. Purity high for reaches but low for
panels → the failure is organised by hydraulic reach (how the study was done) rather
than by map sheet — itself a finding.

*Pitfalls.* Units differ in size by orders of magnitude (a reach is 1–3 km; a polygon
can supply up to 2% of a domain by `poly_cap_frac`); report by unit type, never pooled
across types. Band negatives sit in a different polygon from the positives they were
generated from, so reach-level coverage is the cleanest — it is defined for both.

*Effort.* Easy once R0 exists: a groupby and a histogram.

### R3 — Inside the regions, are the Zone A polygons drawn like hydraulic floodplains or like buffers?

**The label-quality question, and the closest thing to RQ4 that needs no observed
flooding.**

*Question.* A detailed hydraulic study puts the floodplain edge at the base flood
elevation, so the edge sits at roughly constant height above the channel (HAND) along a
reach and its width responds to terrain. Approximate methods often do neither — some
Zone A polygons are fixed-width buffers around the centreline. Do the Zone A polygons
inside the inverse regions carry the approximate signature more than Zone A polygons
outside them?

*Hypotheses.* H2 → region polygons show a buffer signature and inconsistent edge HAND;
no difference → the failure is not in polygon geometry (R4 then decides between H3 and
H1).

*Why regions.* Every diagnostic here is a *within-reach* variance or trend along a
polygon edge. It needs a contiguous stretch of edge — points from 20 tiles give no
stretch to measure.

*Data.* Polygon geometry from `haz_FL_5070.parquet` for the polygons the region
touches (R2's polygon set) and for matched non-region Zone A polygons and AE polygons in
the same tiles; the flowline; a HAND raster sampled at polygon-edge vertices (same
source as the `hand` feature, `docs/DATA-SOURCES.md`; new pull — the edge vertices are
new points).

*Procedure*, per polygon × reach:

1. **Edge-HAND consistency.** Sample the polygon boundary every 30 m; read HAND;
   within-reach SD and along-reach trend. Hydraulic edge → low SD, smooth trend;
   buffer edge → HAND wanders with terrain because the edge ignores it.
2. **Buffer signature.** Distribution of edge-to-centreline distance along the reach:
   coefficient of variation < 0.15, or a value that steps only at reach breaks, marks a
   buffer.
3. **Width vs terrain-implied width.** At cross-sections every 100 m, polygon width vs
   the width of the HAND ≤ h band, with h fitted on AE reaches of the same
   `StreamOrde` (that gives "what a detailed study would have drawn here"). Ratio ≫ 1
   over-extended, ≪ 1 under-extended; report the log-ratio's median and spread.
4. **Shape descriptors** (descriptive only): perimeter²/area, vertex density.

*Null / comparison.* Three groups on one axis: region Zone A polygons, matched
non-region Zone A polygons (same tiles, same stream order), AE polygons. Percentile of
the region group against the matched-growth null's polygon sets.

*Reading.* Region polygons buffer-like and edge-inconsistent → H2 with a mechanism
(the paper can say *what* the approximate method did). Region ≈ non-region Zone A →
the polygons are not the problem in a way geometry sees. Region Zone A ≈ AE → the
region is not an approximate-method artefact at all.

*Pitfalls.* HAND is a 30 m product against a 110–130 m ribbon — use the along-reach
statistics, not per-vertex values. Coastal and lacustrine polygons have flat HAND by
nature; restrict to riverine (`meta_zone_subty`, `log10_dist_coast`). Polygons are
multi-reach; the unit is polygon × reach. The terrain-implied width fit (step 3) must
be done on AE tiles disjoint from the region's tiles.

*Effort.* Moderate to hard: the raster sampling is about a day with the existing pull
path; the rest is geopandas. Do after R1/R2/R4.

### R4 — Is the disagreement between AE-trained models and the Zone A labels structured or white inside the regions — and in which direction?

*Question.* Take the test models' per-point outputs at the OOD points, p̄_i = mean test-
model probability of the positive class, and the residual e_i = p̄_i − y_i. Inside a
region, ordered along the network: do the residual signs come in long runs
(structured disagreement — bias) or alternate (white — noise)? And which sign
dominates?

*Hypotheses.* H2 → long runs of one sign; H3 → runs at the null rate; H1 → runs
present but aligned with terrain (check e_i against `hand`, `log10_dist_river` inside
the region).

*Why regions.* Run structure and direction exist only along a contiguous stretch; a
scattered set yields a marginal disagreement rate and nothing else.

*The direction matters for the task.* Two kinds of inverse point:
- **commission** — a Zone A *positive* the good models call dry: the approximate map
  says floodplain where the detailed-label model says none — an over-drawn Zone A,
  a restudy candidate;
- **omission** — a band *negative* (500–5,000 m from any SFHA, inside the mapped
  area) the good models call wet: floodplain the map does not show — literally the
  "unmapped floodplain" the coverage task exists to find.

A region with one dominant sign is a *directional* label problem and can be handed to
a map reader as such; a mixed region is something else. This is the analysis that
connects the inverse-line result to "improve FEMA coverage".

*Data.* Test-model outputs saved per cell by `zoo.py` (never the selection models —
they were optimised on these points); R0 along-network coordinate.

*Procedure.* Per component: Moran's I of sign(e_i) on the region's own subgraph, and
a runs test along each reach; sign composition (commission share, omission share);
mean |e_i|. Same statistics on matched-growth null regions.

*Metrics and reading.* Run statistic percentile against the null (structured vs
white); commission/omission share with a binomial interval; the two directions
(AE→A, A→AE) side by side — a commission region in AE→A that becomes an omission
region in A→AE is the label disagreeing with itself across provenance, the cleanest
possible statement of H2.

*Pitfalls.* e_i is bounded and heteroskedastic — use its sign for structure and
report |e_i| separately. "Models say wet" in the band is a strong claim: check the
covariate profile of omission points (low `hand`, small `log10_dist_river`, high
`gsw_occurrence`) before calling them unmapped floodplain. Under the paper protocol
per-point outputs are binary correctness; use the probability outputs from the zoo for
this question.

*Effort.* Easy to moderate; everything is already saved.

### R5 — Do the regions land where FEMA itself says the map is weak?

*Question.* External validation against FEMA's own map-quality metadata, at the
reach / panel level.

*Data, in order of strength.*
1. **CNMS** (Coordinated Needs Management Strategy) — FEMA's reach-level inventory of
   flood-study validity (Valid / Unverified / Unknown); the NVUE metric is computed
   from it. **Availability for Florida must be verified** before anything is promised.
2. Proxies possibly inside `NFHL_Key_Layers.gdb.zip` (list layers with
   `pyogrio.list_layers`, as `subset_nfhl.py` does; only `S_Fld_Haz_Ar`, `S_FIRM_Pan`
   and `S_BFE` were subset): `S_XS` cross-sections (the physical trace of a detailed
   study — density per reach), `S_LOMR` (map revisions per panel — where the map was
   contested), `S_Submittal_Info` (study method per submittal).
3. Proxies already on disk: panel effective-date age (`EFF_DATE`), `PANEL_TYP`,
   `SCALE`, BFE-line density (`s_bfe_FL.parquet`, `meta_bfe_km_per_km2`),
   `meta_study_typ = NP`.

*Why regions.* All of these are reach- or panel-level attributes; concordance is a
statement about units, not points. The deliverable — a list of reaches to restudy —
is only producible from regions.

*Procedure.* Enrichment of each indicator in region reaches/panels vs the
matched-growth null; and the reverse list: region reaches that CNMS marks *Valid* —
the candidate restudy list, with R3/R4 evidence attached per reach.

*Reading.* Concordance → the method recovers known weakness (validation of the
instrument). Discordance on Valid reaches → a finding, only as strong as R3/R4 make
it. No CNMS → report the proxies and say what could not be checked.

*Pitfalls.* CNMS reaches are not necessarily NHDPlus reaches — match spatially with a
tolerance. Florida effective dates are mostly countywide and recent, so
within-county spread is small — panel-level `SCALE`/`PANEL_TYP` carry more variation.
Do not over-read cross-section density: it is *definitionally* higher in AE.

*Effort.* Proxies: a day. CNMS: depends on access; time-box it.

### R6 — If a practitioner pooled Zone A into training, do the regions hurt, and can they be removed at the unit level?

**The actionable experiment, and the corrected version of the "train on problematic
points" idea.**

*Question.* Prior work pools A and AE. Does pooling the inverse regions into training
cost accuracy on the trusted labels, and can a practitioner act on that by removing
*units* rather than points?

*Conditions.* Train on:

| | training set |
|---|---|
| (i) | AE only |
| (ii) | AE + all A — the pooling recipe of prior work |
| (iii) | AE + A minus region *points* |
| (iv) | AE + A minus the *whole reaches* the regions touch |
| (v) | AE + A minus the same number of random A points / random reaches |
| (vi) | (iii)/(iv) with the classic set and with smoothed-classic |

*Evaluation.* Held-out AE tiles (the trusted labels): AUC and the positive-rate
calibration. Secondary: held-out A outside the regions. Same 300-model population
recipe as `zoo.py` so the effect is over the population, not one learner.

*Circularity.* E2 split: select regions on half the A tiles, carry membership to the
other half (kNN carry for points, reach id for units), do the excision on that half,
evaluate on AE tiles never used in selection. Never excise the points the regions were
selected on.

*Why regions.* Unit-level excision is only well-defined for coherent regions. The
classic set at S = 500 (AE→A) touches 186 components in 22 tiles; excising every reach
it touches removes a large share of the state. The spatial set touches 8 components in
5 tiles. The metric is therefore **ΔAUC(AE) per reach excised** as well as per 1,000
points excised.

*Reading.* (iv) ≥ (iii) > (v) → the regions carry a removable, unit-organised label
problem — the practical payoff, and the sentence the paper wants. (iii) ≈ (v) → the
inverse property does not translate into training harm; still reportable: the
inversion is about model *ranking* under shift, not about what those points teach.
(i) > (ii) → pooling itself hurts, and the region result is a refinement of that.

*Pitfalls.* Effect sizes are small (500 of ~20,000 training points), so use many tile
folds and seeds and report intervals; the difference between (iii) and (v) is the
number, not (ii) vs (iii). Removing whole reaches also removes their band negatives;
keep the class balance comparable across conditions.

*Effort.* Moderate; reuses `zoo.py` and `cells.py`; compute is 35 s per population.

### R7 — The per-point membership model, kept and repositioned

As specified in `docs/SPATIAL-OODSELECT.md` §5 E3: HGB and logistic membership models
per covariate group (provenance / terrain-hydrology / climate / urbanicity), GroupKFold
on `tile_50km`, block-permutation null, AUC per group.

*What it is for now.* It tests H1 (different places) and provenance-*type* enrichment
at point level, and it is the comparison that shows what a point-level analysis can and
cannot say. Run it on all three region sets. Expect it to work nearly as well on the
classic set — that is the point, and the paper should say it: **enrichment is available
without regions; alignment (R1), unit coverage (R2), label geometry (R3), direction
(R4) and unit-level excision (R6) are not.**

Add `edge_dist_m` (R0) to the provenance group to test the boundary-point mechanism
from §9 of the method doc directly.

### R8 — Maps: one figure per component (the E4 deliverable)

As specified in §5 E4 — NHDPlus flowlines, coast, county and panel boundaries, flood
polygons by subtype — plus two overlays from this document: the R1 break catalogue
(M and L breaks as distinct glyphs along the corridor) and the R4 residual sign per
point. One summary row per component: n, positive share, tiles, dominant
`meta_study_regime` / `meta_panel_scale` / `meta_source_cit` / `meta_eff_date`, BFE
density, R1 excess(M)/excess(L), R2 reach purity, R4 run percentile and
commission/omission share, R5 indicators.

---

## 3. Reading the outcomes together

| pattern | verdict |
|---|---|
| R1 M ≫ L, R2 reach purity high and bimodal, R4 runs structured with one dominant sign, R3 buffer signature | **H2, with mechanism** — the failure follows how the maps were made; the regions are restudy candidates (commission) or unmapped-floodplain candidates (omission) |
| R1 M ≫ L, R2 high, R4 structured, R3 no geometric difference | H2 without a geometric mechanism — provenance-organised but not visible in polygon shape; look at the study metadata (R5) for the cause |
| R1 L ≫ M, R7 terrain beats provenance, R4 runs align with terrain | **H1** — the regions are places the AE-trained models extrapolate badly to; a geography result, still reportable, and the honest alternative |
| R2 low and uniform, R4 white, R1 neither | **H3** — scattered label error; coherence was imposed by the graph, not found in the data — say so; the E1 tradeoff curve is the figure |
| R6 (iv) ≥ (iii) > (v) | the label problem is unit-organised and removable — the practical claim |
| spatial > smoothed-classic on R1/R2 at equal r | the regularizer does more than smoothing — the method claim |
| spatial ≈ smoothed-classic on R1/R2 | post-hoc smoothing suffices; the paper's contribution becomes the *analysis* rather than the objective — still fine, and cheaper to defend |

Every row is reportable. The last two rows are the method's own honesty check and
must be run before the paper leans on the objective.

---

## 4. Order of work

Regions to start from (`results/2026-09-24/oodselect_paper/`): AE→A spatial S = 250
and 500 (5 tiles, 7–8 components), A→AE spatial S = 50 (1 tile) and 500 (7 tiles,
23 components), with their classic and smoothed-classic counterparts. Seeds 1–2 in
parallel; re-run the analyses on recurring components once they exist.

| week | do | why first |
|---|---|---|
| 1 | R0; smoothed-classic masks; R2; R4; R8 maps | everything downstream needs R0; R2 and R4 use only saved columns and outputs; the maps show whether the components are what the numbers say |
| 2 | R1 (break catalogue + test); R6 | R1 is the primary test of the hypothesis; R6 is the practical claim and reuses the zoo |
| 3 | R7 on all three region sets; R5 proxies; write up | R7 is cheap and frames the comparison; R5 proxies are a day |
| if time | R3 (raster pull); R5 CNMS; E6 IA/MO | R3 is the mechanism, not the verdict; IA/MO only after Florida is closed |

Minimum viable E3/E4 for the deliverable: R0, R2, R4, R8 and either R1 or R6.

---

## 5. Deliverables

Figures: (7) R1 alignment — excess(M) and excess(L) per component with null bands,
three region sets; (8) R2 unit-coverage histograms per unit type, three region sets
on one axis; (9) R4 along-network residual strip per component with break glyphs;
(10) R6 ΔAUC(AE) per reach excised, conditions (i)–(vi); (11) R8 map per component.
(Figures 1–6 remain as listed in `docs/SPATIAL-OODSELECT.md` §10.)

Tables: per-component summary (R8 row); R1 per-break-type odds ratios; R5 indicator
enrichment and the candidate restudy list; R7 AUC by group × region set.

Code: `aotil/regions.py` (R0), `aotil/breaks.py` (R1), `aotil/coverage.py` (R2),
`aotil/labelgeom.py` (R3), `aotil/residuals.py` (R4), `aotil/external.py` (R5),
`aotil/excise.py` (R6), `aotil/explain.py` (R7), `aotil/maps.py` (R8). Dataset
additions: `meta_firm_pan`, `meta_near_fld_ar_id`, `edge_dist_m`.

---

## 6. Pitfalls specific to region analysis

- **Pseudo-replication is worse inside a compact region, not better.** Tile-level
  bootstrap (`transfer.tile_bootstrap_r`) is weak with 5 tiles; treat *components* as
  the replication unit, report per component, and pool only with the null.
- **Multiple testing.** Six questions × several break types × three region sets. Two
  statistics are pre-registered as primary — R1 excess(M) − excess(L) and R2 reach
  purity — and everything else is reported as percentiles, not as significance.
- **Budget artefacts.** A component's extent depends on S; compare S = 250 and 500 for
  every region-level statistic and treat what moves with S as a budget artefact.
- **Coastal vs riverine.** Coastal components end at the coast, have flat HAND, and are
  dominated by `log10_dist_coast`; analyse separately and say which kind each component
  is.
- **Direction as a sign check.** A statistic that looks the same in AE→A and A→AE is
  about geography; one that flips is about labels. Run both before interpreting either.
- **The selection models are not evidence.** Any per-point model quantity inside a
  region comes from the test models (R4) or from populations trained after the E2
  split (R6).
- **`meta_source_cit` in Florida is mostly `<NA>`**; the study unit falls back to
  `dfirm_id` × `meta_study_typ`, which in a countywide-DFIRM state means the county.
  Say so in the paper; IA/MO are where source citations vary.

---

## 7. The paragraph for the paper

Spatial regularisation is justified by the level at which the explanation and the
action live, not by the appearance of the maps. The hypothesis under test — that
transfer failure follows how the flood maps were produced — is a statement about
studies, panels and reaches, which have boundaries and dates; a point set has neither,
so the hypothesis cannot be tested on one. Two measured facts make the point set the
wrong object anyway: many disjoint point sets carry the inversion equally (classic and
spatial selections overlap at Jaccard ≤ 0.05), and the inverse property transfers to
held-out neighbours whether or not the regulariser is on, so the place, not the points,
is what is stable. The regulariser is the instrument that returns the place; the
coherence–inversion tradeoff shows the instrument costs little; and the analyses that
follow — boundary alignment, unit coverage, label geometry, residual structure and
unit-level excision — are each undefined on scattered points. Point-level enrichment,
which is defined on scattered points, is reported alongside them to show what a
point-level analysis can and cannot say.
