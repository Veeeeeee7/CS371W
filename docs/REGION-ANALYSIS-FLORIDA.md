# Region analysis, Florida revision — which rule drew the label, and does the terrain agree?

> **Status 2026-10-13 (`docs/PROJECT-PLAN.md` v4, from the paper).** The paper commits to G1
> only, in this form: the Zone A boundaries inside the AE→A regions, tested against terrain
> (elevation level sets, class E) and against lines the model never saw: soil-survey lines (S),
> county lines and map-sheet edges (M). W, C and the other M layers are optional. Outcome 5's
> comparison is region lobes vs. matched unselected Zone A lobes in the same tiles (§3.3,
> "region specificity"), with Zone AE lobes as the rule contrast and the displaced-edge null as
> the base rate. G2 shrinks to the cheap form that falls out of G1 (§4.1); §4.2 (ponding),
> DISDE, R6 and the A→AE Everglades Agricultural Area region are not in the paper. §9 is
> history: the paper's current GIS paragraph is in `CS371_Cho_Li/acl_latex.tex`, and the
> `CHANGES-2026-09-27.md` it mentions is no longer in that folder. Data still to collect:
> v4 §4 and `docs/TASK-DATA-AUDIT.md`.
> **Also 2026-10-13:** the regions in §2 come from the v1 sample (200 m spacing, 0.57:1 band
> negatives). Florida is being rebuilt with sampling protocol v2 (`docs/PROJECT-PLAN.md` §2a), so
> the regions will be re-derived; treat §2 as an example of what the method returns.

**v2, 2026-09-27** · supersedes the *choice of analyses* in `docs/REGION-ANALYSIS.md` (v1, R0–R8)
for the Florida cells; keeps its rules (§0.4), nulls, code layout and R6–R8 · reads
`results/2026-09-24/oodselect_paper/` (paper protocol, classic vs spatial μ = 3) and
`results/2026-09-26/aef_checks/` (T4 "Zone A ceiling")

This document answers one question: **which GIS analysis, defined only on spatially coherent
regions, shows that Zone A and Zone AE are a difference the models cannot resolve?** It
replaces the along-network questions of v1 (R1 break alignment, R3 buffer geometry, the
along-network coordinate) because the regions the spatial method actually returned in Florida
are not river corridors (§2). It is written so that the spatial regularizer is justified by the
analysis — the analysis has no object without a region — and not the other way round.

---

## 0. Summary

- **The claim to support** is a *concept shift* — P(flood | covariates) differs between Zone A
  and Zone AE — not a covariate shift. "Models cannot resolve the difference" means two things,
  both of which count: the Zone A label depends on information no covariate carries (a
  soil-survey line, a 1970s flood-prone-area outline, a county's design elevation), or it
  follows a *different rule* on the same covariates than the Zone AE label does (a wetland edge
  rather than an elevation level set). Either way no single predictor fits both zones, and a
  model trained on pooled labels is asked to reproduce a contradiction. (§1)
- **The Florida regions are interior wetland basins, not corridors.** The AE→A spatial regions
  at S = 500 are four patches 15–40 km across in the Santa Fe/New River swamp
  (Bradford–Union), the Fort Drum / Blue Cypress marsh (Okeechobee–Indian River), the Osceola
  swamps (Columbia) and the Everglades edge (Miami-Dade); all 491 positives are `NOT_POPULATED`
  approximate Zone A, median HAND 0.4 m, no visible water, 0.9–3 km from the nearest NHDPlus
  flowline (20 km at the Everglades edge), inside polygons of 140–1,600 km². The A→AE region is the Everglades Agricultural
  Area (Palm Beach–Hendry sugarcane, 17 km from any flowline). Nearest-neighbour spacing inside a
  region is 600–800 m. Confluences, stream order, reach breaks and floodplain width are not
  defined for these objects. (§2)
- **G1 — which rule drew the boundary (headline).** Sample the Zone A polygon edge inside each
  region every 30 m; at each vertex record the DEM elevation and the distance to every
  candidate *generating layer* the approximate method is documented to have used (soil-survey
  map-unit lines, wetland-inventory lines, land-cover edges, topographic contours) and to the
  administrative lines it was drawn on (map-sheet, county, section lines). Classify each stretch
  of edge as an *elevation level set* (the detailed-study rule, a water surface — learnable from
  terrain covariates), a *land-cover/wetland edge* (a different rule on the same covariates), or
  a *soil-survey / administrative line where the terrain does not change* (information no
  covariate carries). Compare with Zone AE edges in the same tiles and with size-matched Zone A
  edges outside the regions, under a displaced-edge null. Boundary overlap statistics
  (Jacquez 1995) — the machinery v1 planned for R1 — with the label's sources as the reference
  boundaries instead of panel edges. (§3)
- **G2 — does the terrain agree (the flood analysis).** Flat-terrain flooding is ponding and
  sheet flow, not riverine conveyance; FEMA's own method for it is a stage–storage
  (reservoir-routing) analysis on the terrain (FEMA Guidance Document 84). Cheap form: is the
  region's Zone A edge a single water surface (edge-elevation range within FEMA's own
  ½-contour-interval boundary standard) — this falls out of the G1 vertex table. Full form: do
  the closed depressions the region sits in fill to the mapped edge under the 1%-annual-chance
  24-hour rainfall (`p100_24hr` is already a covariate)? G2 says *which side is wrong*; G1 says
  *why the models cannot resolve it*. (§4)
- **Aggregate companions** that need no regions but set the claim up: the T4 "Zone A ceiling"
  (OOD AUC on Zone A 0.839–0.846 under six feature sets while ID AUC moves 0.904–0.937), and a
  DISDE decomposition of the AE→A gap into X-shift and Y|X-shift. (§5)
- **What happens to R0–R8**: R0 becomes lobes and edges instead of reaches; R1 folds into G1's
  administrative class; R3 is replaced by G1+G2; R2, R4, R6, R7, R8 stay with their unit changed
  from reach to polygon lobe / soil map unit. (§6) Order of work and minimum viable set in §7;
  the paper text in §9.

---

## 1. The claim, stated so that an analysis can support it

Write x for the covariates, y_AE(x) for the label a detailed study assigns and y_A(x) for the
label an approximate study assigns. A detailed study computes a water-surface elevation from
discharge and surveyed cross-sections and thresholds the terrain against it, so on riverine
reaches y_AE is close to a function of terrain and discharge — of things in x (HAND, upstream
area, slope, rainfall). The approximate methods "overla[id] the approximate outline of the base
floodplain" from "soils mapping, actual high water profiles, aerial photographs of previous
floods, and topographic maps" onto USGS quadrangles (FEMA, *NFIP Floodplain Management Study
Guide*, Unit 3, pp. 3-5 and 3-25 ✅), and in Florida from USGS flood-prone-area quadrangle maps
of the 1970s (Putnam County FIS, p. 14 ✅: approximate elevations "developed from normal depth
calculation and flood prone area maps"). So y_A can depend on

1. things *not in x* — a soil map-unit line, the outline traced on a 1974 quadrangle, a
   county-wide design elevation; or
2. things *in x* but combined by a *different rule* — a wetland edge (land cover) rather than an
   elevation level set.

In the dataset-shift vocabulary (Moreno-Torres et al. 2012) both are **concept shift**,
P(Y | X) differs between domains, as opposed to covariate shift, P(X) differs. "Models cannot
resolve the difference" is the statement that no single f(x) reproduces both y_A and y_AE where
they overlap. Its consequences are exactly the ones the paper already predicts:

- trained on AE, a model gets *worse* on the Zone A regions the *better* it fits AE — the inverse
  line — because fitting the detailed rule more faithfully moves it further from a label made by
  a different rule;
- trained on pooled labels, a model averages two rules and reproduces part of the approximate
  method's outline wherever the approximate rule dominates the training data (the hypothesis in
  paragraph 4 of the introduction);
- no feature set removes the gap (§5, the T4 ceiling): adding covariates helps only against
  covariate shift.

Three kinds of evidence establish it, at three levels:

| level | evidence | needs regions? |
|---|---|---|
| aggregate | OOD AUC on Zone A does not move when the features change (T4); DISDE assigns the gap to Y\|X rather than X | no |
| regional, label | **G1**: inside the failure regions the Zone A boundary is a soil-survey / wetland / administrative line, not an elevation level set; Zone AE boundaries in the same tiles are level sets | yes — needs kilometres of the *same* boundary and a whole lobe to classify |
| regional, physical | **G2**: the terrain says which label the 1% flood would actually reach | yes — a basin fills or it does not; a point set has no basin |

The aggregate level says the difference exists and is not a shortage of covariates. The regional
levels say *what* it is and *who is wrong*. Only the regional levels can be tied to how the maps
were made, which is the paper's headline.

**Why the v1 questions did not fit this claim.** R1 asked where a region *ends* relative to
map-production breaks. That is a statement about *where* failure is organised, not about
*what the label is*; a positive R1 would not show that the difference is unlearnable, and a
negative one would be uninterpretable. In Florida it is also underpowered: DFIRMs are countywide
(one effective date per county), `meta_source_cit` is mostly `<NA>`, and the decision of
2026-09-26 removed A↔AE transitions and BFE onset from the break catalogue, leaving panel edges
(a regular grid with a high base rate) and county lines. And its break catalogue — confluences,
stream order, channel slope, along-network distance — presumes corridors along NHDPlus reaches,
which §2 shows the Florida regions are not. R3 ("buffers") presumed a riverine ribbon and a
hypothesis the literature check found no documentary basis for. Both are replaced, not
discarded: R1's statistic and citations move into G1 (§3.4), and R3's edge-elevation idea
becomes G2's cheap form (§4.1).

---

## 2. What the Florida regions actually are

Decoded from `spatial_mu3_knn10/*_S500_theta.npy` (top-S mask) joined to `dataset_FL.parquet`;
components on the same 10-NN graph the selection used. Only components with ≥ 20 points.

### 2.1 AE→A (`FL_band_AE_s0`, OOD = Zone A; 491 positives, 9 band negatives; 8 components, 5 tiles)

| cc | n | where | counties (DFIRMs) | extent | polygons | median HAND | dist. to flowline | dist. to coast | elev. | land cover | visible water |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 198 | Santa Fe / New River swamp, N Florida | Bradford 12007, Union 12125 | 41 km | 3 (141 km² median) | 0.3 m | 0.9 km | 89 km | 42 m | woody wetland | none (water_frac 0.002) |
| 1 | 142 | Fort Drum / Blue Cypress marsh | Okeechobee 12093, Indian River 12061 | 33 km | 3 (302 km²) | 1.0 m | 1.7 km | 37 km | 9 m | woody wetland | none (0.021) |
| 4 | 102 | Osceola National Forest swamps | Columbia 12023 | 22 km | 2 (458 km²) | 0.3 m | 3.1 km | 107 km | 39 m | woody wetland / pine | none (0.000) |
| 2 | 27 | Everglades edge | Miami-Dade 12086 | 14 km | 1 (1,618 km²) | 0.2 m | 19.6 km | 25 km | 1 m | emergent wetland | some (0.61) |

All 491 positives are `meta_study_regime = NOT_POPULATED` (classic approximate Zone A). Land
cover over the set: 51% woody wetlands, 20% evergreen forest (pine flatwoods), 7% emergent
wetlands. 86% of region positives have no visible water within 900 m, against 76% of other Zone A
positives and 36% of Zone AE positives. Two of the four components (and the EAA component in
the other direction) straddle a county line — they *cross* DFIRM boundaries rather than end at
them; cc 0 sits on the New River, which is itself the Bradford–Union line.

### 2.2 A→AE (`FL_band_A_s0`, OOD = Zone AE; 306 positives, 194 band negatives; 23 components, 7 tiles)

| cc | n | where | counties | extent | notes |
|---|---|---|---|---|---|
| 3 | 286 | Everglades Agricultural Area (sugarcane south of Lake Okeechobee) | Palm Beach 12099, Hendry 12051 | 45 km | 79% positives; 5 AE polygons (330 km² median); land cover *cultivated crops*; 17 km from any flowline; HAND 0.3 m; effective dates 2017–2020; `NOT_POPULATED` |
| 16, 18 | 22, 31 | same area, Palm Beach | 12099 | 9–13 km | as above |
| 1 | 56 | Everglades / Florida Bay coastal marsh | Miami-Dade 12086, Monroe 12087 | 13 km | **100% band negatives**: points 500–5,000 m from any SFHA that Zone-A-trained models call wet (water_frac 0.92) — omission candidates |

### 2.3 Shape

| cc | median NN spacing | convex hull | elongation (σ₁/σ₂) |
|---|---|---|---|
| AE→A 0 | 590 m | 557 km² | 2.1 |
| AE→A 1 | 670 m | 276 km² | 2.1 |
| AE→A 4 | 785 m | 191 km² | 1.2 |
| A→AE 3 | 640 m | 694 km² | 1.2 |

The OOD set is 9,412 points over the whole state, so inside a large polygon the sampled points
sit 600–800 m apart and the kNN graph joins everything within a 15–40 km neighbourhood. A
"region" here is a **basin-scale landscape unit** — a named swamp or marsh complex sampled at
100–300 points — not a 200 m-spaced corridor. That is a better object for GIS than a corridor
would have been: the flood analysis for flat terrain is a basin analysis, and "does the Santa Fe
swamp fill" is a well-posed question in a way that "does this scattered set of 500 points fill"
is not.

### 2.4 Consequences for the analysis design

Dropped as ill-posed for these regions: the along-network coordinate (R0 step 3), confluence /
stream-order / channel-slope breaks (R0 step 4 class L), floodplain-width analyses (R3 steps
2–3), HAND-stage riverine reconstruction, along-reach runs (R4 runs test), reach-level excision
(R6 condition iv as reaches). NHDPlus V2 medium-resolution flowlines miss the small streams,
sloughs and canals that drain these basins; NHDPlus HR would place channels closer, but the
regime is still ponding / sheet flow, not channel conveyance.

Kept and re-targeted: boundary overlap statistics (R1 → G1), edge sampling against terrain (R3
step 1 → G2 cheap form), unit coverage (R2, unit = polygon lobe or soil map unit), residual
structure (R4, Moran's I on the region subgraph; direction), unit excision (R6, unit = polygon
lobe), membership model (R7), maps (R8 with new overlays), FEMA metadata (R5 — now including the
FIS reports and the 1974 flood-prone-area quadrangles that cover cc 4).

Added first step, **region typing** (R0'): classify every component as interior-wetland,
coastal, or riverine from `meta_dist_to_river_m`, `log10_dist_coast`, `hand`, `landcover`,
`meta_zone_subty`; apply G2's ponding form to the first, note the second (coast-bounded by
construction), and use the riverine HAND-stage variant only if a riverine component appears
(seeds 1–2, larger S, or IA/MO).

---

## 3. G1 — Which rule drew the boundary? (headline; needs regions)

### 3.1 Question and hypotheses

Inside each region, what does the Zone A polygon boundary coincide with? Candidate
*generating layers*, grouped by what they mean for a model:

| class | layer | source of the layer | learnable from x? | what coincidence means |
|---|---|---|---|---|
| **E** elevation level set | DEM (3DEP 1/3″) | the detailed-study rule: boundary = terrain ∩ water surface | yes (HAND, TPI; elevation itself was dropped) | the boundary is a water surface — the rule the AE labels teach |
| **W** wetland / land-cover edge | NWI wetland polygons; NLCD class boundaries | aerial-photo interpretation | partly (`landcover` is a covariate) | a different rule on the same covariates: "floodplain = wetland" |
| **S** soil-survey line | SSURGO map-unit boundaries (flooding-frequency class, hydric %, drainage class changes) | NRCS soil survey, 1:20–24k | **no** (StreamCat soils are catchment means) | information no covariate carries |
| **C** topographic contour | 5-ft (1.524 m) contour values of the USGS quadrangle | contour interpolation on the quad | not from HAND; only from absolute elevation, which is a location fingerprint | the "topographic maps" source |
| **M** administrative / cartographic | FIRM panel edges, county (DFIRM) lines, PLSS section lines, roads, canals | the sheet the outline was drawn on | no | R1's map-production class; Wu et al. 2026's county-line cut-offs |

Hypotheses. **H2 (label bias, the paper's hypothesis):** region Zone A edges are W/S/C/M,
Zone AE edges in the same tiles are E, and non-region Zone A edges lie between. **H1 (covariate
shift):** region Zone A edges are E as much as AE edges are — the label follows terrain and the
models fail for another reason (§4 decides which). **H3 (noise):** no class dominates and the
edge is not a level set of anything — scattered error.

### 3.2 Why regions

The statistic is a share of *boundary length*, and its verdict is about a *lobe* — a part of a
polygon drawn in one pass from one source. Both need kilometres of the same boundary: a
displaced-edge null (§3.3) has no power on 200 m of edge, and "this lobe was traced from the soil
survey" is a statement about a unit. A scattered set of 500 points in 22 tiles gives ~1–3 points
per polygon and no lobe to classify. The region — a 15–40 km patch of one or two polygons —
gives tens to hundreds of kilometres of edge from a handful of lobes, and it is the region's
extent that says *which* lobes.

### 3.3 Procedure

Build once (extends R0 `aotil/regions.py`):

1. **Lobes.** From `haz_FL_5070.parquet`, take the polygons the region's positives fall in
   (`meta_fld_ar_id`) and the polygons within 2 km of any region point; explode multipart
   geometries; keep the parts that intersect the region's 10-NN-ball union (or convex hull
   buffered 2 km). These parts are the *region lobes*. Do the same for Zone AE polygons in the
   region's tiles (control E-rule edges) and for Zone A polygons in the same tiles outside the
   region (control, matched on lobe perimeter).
2. **Edge vertices.** Densify each lobe boundary to 30 m and write one row per vertex:
   `lobe_id, zone, in_region, s (along-edge position), lon, lat`.
3. **Terrain at the vertex.** DEM elevation (3DEP 1/3″ via `USGS/3DEP/10m` on GEE, the existing
   `pull_gee.py` path) at the vertex and at ±100 m along the inward and outward normals;
   local across-edge drop Δz = z_out − z_in. Also `hand` (MERIT 90 m, already pulled) for
   continuity with the features.
4. **Distances to layers.** For each vertex, distance to the nearest line of each layer:
   SSURGO map-unit boundary (Soil Data Access spatial query, `mupolygon` for the region's bbox;
   attributes from `muaggatt`: `flodfreqdcd`, `pondfreqprs`, `hydclprs`, `drclassdcd`), NWI
   wetland boundary (USFWS state download or REST), NLCD class boundary (from the `landcover`
   raster), panel edge (`pan_FL.parquet`), county line, PLSS section line (BLM CadNSDI /
   FGDL), road and canal centrelines (TIGER; NHD `CanalDitch`). Record also *which* soil
   attribute changes across the nearest soil line (flooding class, hydric %, drainage).
5. **Classify each 300 m window of edge** (10 vertices) by two facts:
   - *level set?* the window's elevation range ≤ τ, with τ = ½ contour interval = 0.76 m for a
     5-ft quad (FEMA's Floodplain Boundary Standard for approximate studies is ±½ contour
     interval; NRC 2007 ✅) and, as a sensitivity, τ = 0.3 m (lidar-era) — and a real
     across-edge drop (Δz > 0 on the outward side);
   - *on a line?* the window's median distance to layer L ≤ d, d = 30 m (one vertex) and 100 m.
   The 2 × 2 of {level set} × {on layer L} per layer is the result; the cell **{not a level
   set} × {on S or M line}** is the "unresolvable" edge length, **{level set}** is the
   AE-rule edge length, and **{not a level set} × {on W line}** is the different-rule edge
   length.
6. **Contour signature (class C).** Histogram of window-median elevations modulo 1.524 m; a
   boundary traced along quad contours peaks at one phase. Report the peak's height against a
   uniform null. Datum: quads are NGVD29, 3DEP is NAVD88; the offset only shifts the phase.
7. **Tie-in check (optional).** Where a region lobe meets a Zone AE polygon, the edge elevation
   on the A side within 500 m of the contact vs the AE side: a continuous water surface (tie-in
   honoured) or a step. This is not the region-end question that the 2026-09-26 decision
   excluded from R1 — it uses the A/AE contact as a place where both rules drew the same
   terrain, not as a break.

**Nulls.**
- *Displaced-edge null (primary, base rate):* translate the lobe boundary by a random vector of
  300–1,000 m (and/or rotate about the lobe centroid), recompute step 4, 200 draws. Soil and
  wetland lines are dense in Florida — any curve is within 100 m of some soil line — so the base
  rate is high and everything is reported as excess over this null.
- *Rule contrast:* Zone AE lobes in the same tiles (they should be level sets and cross soil
  lines freely).
- *Region specificity:* Zone A lobes in the same tiles outside the region, matched on
  perimeter. If they look like the region lobes, Zone A is a W/S label *everywhere* and the
  regions are where that label meets terrain the AE models never saw (→ §4 and R7 carry the
  region-specific explanation) — still a clean result for the claim.
- *Matched-growth null on the point graph* (v1 §0.4) is not needed here: the object is the
  lobe, not the point set.

**Metrics** (per lobe, per component, pooled by edge length): edge-length share in each 2 × 2
cell per layer; excess over the displaced-edge null with its percentile; the same for AE lobes
and matched Zone A lobes on one axis. Pre-registered primary: **share of region Zone A edge
that is not a level set and lies on an S or M line, minus the same share for Zone AE edges in
the same tiles**, with its null percentile. Secondary: the W share; the C phase peak.

### 3.4 Relation to R1 and to the literature

This is boundary overlap analysis (Jacquez 1995 O_S / O_G; Fortin, Drapeau & Jacquez 1996) —
the method v1 chose for R1 — applied to the *label's* boundary against its *generating* layers
instead of to the *region's* boundary against panel edges. The citations and the seam literature
(Ramcharan 2018; Hengl 2017; Wu 2026) carry over; panel and county lines are class M. The
novelty claim from `claude/region-analysis-lit-review.md` is unchanged in kind and sharper in
content: HAND-vs-FEMA comparisons never split A from AE, and nobody has asked which source a
Zone A boundary was traced from, at scale, or tied that to where a model fails.

### 3.5 Pitfalls

- **Flat terrain makes horizontal position uncertain even when the boundary is right.** At
  slopes of 0.2–0.3° (the region median) a ±0.76 m vertical tolerance is ±150–200 m
  horizontally. Use d = 100 m as the main coincidence band and the level-set test on *windows*,
  never per vertex.
- **Lidar under cypress canopy.** Ground returns under dense swamp canopy carry decimetre errors;
  compare A and AE lobes under the same land cover and use IQR, not range, for the level-set
  test when `landcover ∈ {90, 95}`.
- **Layers are correlated.** Soil lines partly follow vegetation, which partly follows
  micro-topography. The level-set condition is what separates them: a boundary on a soil line
  *with* a terrain drop is E as much as S. Report the joint cells, not marginals.
- **Coastal lobes** (cc 2, A→AE cc 1) are bounded by the shore and by tidal AE/VE zones; type
  them coastal (R0') and report separately.
- **Polygon generalisation.** NFHL polygons were digitised at 1:12,000–1:24,000; vertex density
  and smoothing differ by county. Densify to 30 m before anything and report per county.

**Effort.** Lobes + vertices: a day. GEE elevation pull at vertices: half a day with the existing
pipeline. SDA and NWI pulls for four bounding boxes: a day. Classification, nulls, figures: two
days. **About a week.**

---

## 4. G2 — Does the terrain agree? (the flood analysis; needs regions)

G1 says which rule drew the label. G2 asks which label the water would reach, which is what
decides between the two readings of the inverse line that the coverage task cares about:

- **Story 1 (H2):** the approximate outline is over-drawn — the region is dry under the 1%
  event; the AE-trained models are right to call it dry; pooling Zone A into training teaches the
  error (paragraph 4's hypothesis; R6 should show harm).
- **Story 2 (H1 by site selection):** the region floods; the AE-trained models are wrong because
  detailed studies were only ever done near mapped rivers and the coast, so "flood ⇔ visible
  water nearby" is a shortcut those models learned from *where* AE labels exist — a provenance
  artefact of a different kind; pooling would *help* coverage, and the regions are the unmapped
  floodplain the coverage task wants.

Both are provenance stories and both are reportable; they differ in what a practitioner should
do. Riverine hydraulics cannot referee them here (§2.4). Flat-terrain flooding is **ponding and
sheet flow**, and FEMA's own method for it is a stage–storage analysis on the terrain: "Ponding
shallow flood hazards can be identified by determining the inflow to and outflow from the
ponding area and calculating the storage volume and elevations using reservoir routing
analysis", with "one BFE for each ponding area" (FEMA, *Guidance for Flood Risk Analysis and
Mapping: Shallow Flooding Analyses and Mapping*, Guidance Document 84, Dec 2020 ✅).

### 4.1 Cheap form — is the mapped edge a water surface? (falls out of G1)

A 1% flood in a flat basin has one water-surface elevation per ponding area. So a boundary that
*is* a flood boundary has, per lobe, an edge-elevation range within the boundary standard
(±½ contour interval → 1.5 m total for a 5-ft quad; 0.6 m for lidar-era work). Per lobe report
the edge-elevation IQR and range, for region Zone A, non-region Zone A and Zone AE lobes in the
same tiles. Zone AE lobes are the calibration: their edge sits at the published BFE and their
spread is the measurement floor. A Zone A lobe whose edge wanders over several metres of
elevation is not a single water surface; whatever drew it, no 1% event fills it to that line.
This is R3 step 1 re-targeted at flat terrain; the data are the G1 vertex table, so it costs
nothing extra.

Reading: region Zone A spread ≈ AE spread → the approximate edge is a plausible water surface
(Story 2 stays live; go to 4.2 to check the level); region spread ≫ AE spread and ≫ non-region
Zone A spread → the approximate edge is not a water surface (Story 1, with G1 saying what it is
instead).

### 4.2 Full form — does the basin fill to the mapped edge? (if time)

For each interior-wetland component:

1. **DEM.** 3DEP 1/3″ (10 m) for the component's hull buffered 5 km — 300–1,000 km², 3–10 M
   cells; export from GEE per component or pull the 1° tiles. Florida's 3DEP is lidar-derived.
2. **Depressions and catchments.** WhiteboxTools (`pip install whitebox`): `FillDepressions`,
   `DepthInSink`, `Sink`, `Watershed`/`Basins`; for nested basins the depression hierarchy of
   Wu et al. 2019 (level-set) or Barnes, Callaghan & Wickert 2021 (Fill–Spill–Merge). Assign
   each region point to its depression (or "not in a closed depression" — a slough with an
   outlet, which is the sheet-flow case).
3. **Storage.** Per depression, the stage–storage curve from the DEM (cumulative volume below
   each stage).
4. **Runoff volume.** 1%-annual-chance 24-hour rainfall (`p100_24hr`, Atlas 14, already pulled)
   × catchment area × a runoff coefficient. Use SCS curve numbers from `landcover` and the
   SSURGO hydrologic soil group (`hydgrpdcd`, pulled in G1), and bracket with C = 0.2–0.6 as a
   sensitivity. Antecedent condition matters in Florida (high water table); report the bracket,
   not a point estimate.
5. **Stage and extent.** Invert the stage–storage curve at the runoff volume → ponding elevation
   → inundated extent (DEM ≤ stage inside the depression). For depressions with an outlet,
   report the *spill* elevation as an upper bound on ponding and treat the rest as sheet flow.
6. **Compare** the ponded extent with (a) the Zone A polygon, (b) the AE-trained test models'
   mean prediction inside the region (R4's p̄_i), (c) the AE lobes in the same tiles (they
   should be reproduced at their BFE — the calibration).

Reading: the ponded extent reaches the Zone A edge → Story 2 (the map is right, the models'
shortcut is the failure); it stops well short → Story 1 (over-drawn); it exceeds the Zone A edge →
the approximate map is *under*-drawn there (omission, also a coverage finding). The direction
agrees with R4's residual sign if the models are tracking the terrain.

This is a simplified FEMA ponding analysis, not a hydraulic model; say so. Its value is that it
is independent of both maps and of every model in the population.

**Effort.** 1–2 weeks. The compute is trivial at these sizes; the time goes into the hydrology
choices (outlets, runoff, antecedent conditions). Do 4.1 first; do 4.2 for the two largest
components if week 3 allows.

### 4.3 Riverine variant (only for riverine-typed components)

If a component types as riverine (near an NHDPlus flowline, HAND-controlled), use the HAND-stage
reconstruction instead: fit the stage h per stream-order class on AE reaches in disjoint tiles
(the width of the HAND ≤ h band that best reproduces AE polygons), flood the region's reaches at
that stage, compare with the Zone A polygon; the CFIM 10 m HAND and its hydraulic property
tables are downloadable per HUC6 (`docs/DATA-SOURCES.md` §3). None of the current Florida
components qualify; IA/MO would.

---

## 5. Aggregate companions (no regions needed; they set the claim up)

- **The Zone A ceiling (T4, `results/2026-09-26/aef_checks`).** AE→A OOD AUC is 0.839–0.846 for
  every feature set tried (45 covariates, AlphaEarth 2017/2024, AlphaEarth + terrain, + hydro,
  + 100 m context) while the ID AUC of the same models spans 0.904–0.937. A gap that does not move
  when the covariates change is not a covariate-shift gap. One seed, 90 models, no XGBoost —
  replicate on seeds 1–2 before quoting; then it is one sentence in Results and the natural
  opening of the analysis section.
- **DISDE (Cai, Namkoong & Yadlowsky 2023; arXiv:2303.02011)** decomposes the AE→A performance
  drop into the part explained by P(X) changing and the part explained by P(Y | X) changing,
  under shared support (measured at 68% of points in [0.05, 0.95] propensity). The Y|X share is
  the aggregate number for "cannot resolve". Already on the plan's time-permitting list;
  moderate effort (a domain classifier and reweighting on the existing zoo).
- **Do not use a naive matched-covariate label-rate comparison.** Selecting "look-alike" points
  (no visible water, HAND < 1 m, wetland/pine) gives a positive share of 0.847 in AE and 0.840 in
  A — identical, because band negatives are by construction 500–5,000 m from any SFHA, so any
  covariate profile that resembles a floodplain is positives-heavy in both domains. The negative
  protocol ties the label to distance from the label; only a propensity-based method (DISDE)
  handles that.

---

## 6. What happens to R0–R8

| v1 | status | v2 |
|---|---|---|
| R0 units and along-network coordinate | **re-targeted** | R0': region typing; lobes (exploded polygon parts intersecting the region) as the unit; edge-vertex table (§3.3 steps 1–4) as the shared machinery; `edge_dist_m` (distance from a positive to its own polygon edge) stays and should be the first number computed — it says how far the region points sit from the boundary that labelled them; `meta_firm_pan`, `meta_near_fld_ar_id` stay |
| R1 break alignment | **folded into G1** as class M | panel and county lines are reference layers; confluence / stream-order / slope breaks dropped; the Jacquez machinery and citations move to G1 |
| R2 unit coverage | kept | unit = polygon lobe, SSURGO map unit, NWI polygon; coverage = share of the unit's OOD points in the region; bimodality reading unchanged |
| R3 polygon geometry | **replaced** by G1 + G2 cheap form | edge-HAND consistency → edge-elevation level-set test; buffer signature and width vs terrain-implied width dropped (no ribbons here) |
| R4 residual structure | kept, adapted | Moran's I of sign(e_i) on the region subgraph (600–800 m spacing) instead of runs along a reach; commission/omission shares unchanged and now compared with G2's direction |
| R5 FEMA metadata | kept, **extended** | add: the FIS report for each region county (Bradford, Union, Okeechobee, Indian River, Columbia, Miami-Dade, Palm Beach, Hendry) — the "approximate analyses" section states the method used; the 1974 USGS flood-prone-area quadrangles for the Osceola NF area (OFR 74-1111: Big Gum Swamp, Sanderson N/S, Taylor, Sanderson NW — confirm which quads cc 4 overlaps); CNMS `REASON`/`STUDY_TYPE` if the REST layer works from Victor's machine |
| R6 unit excision | kept | excise whole lobes (or whole `meta_fld_ar_id` polygons) rather than reaches; the E2 split and the random-lobe control (v) unchanged |
| R7 membership model | kept | unchanged; expect `log10_dist_coast`, `water_frac_900m`, `gsw_occurrence`, `landcover` to dominate — that is Story 2's fingerprint at point level and the comparison G1/G2 are measured against |
| R8 maps | kept, new overlays | SSURGO flooding class and hydric %, NWI, NLCD, 3DEP hillshade with depressions, panel and county lines, the G1 edge classification as coloured boundary segments, R4 residual sign |

Rules from v1 §0.4 that still apply: three region sets side by side (classic, spatial μ = 3,
smoothed-classic) for R2/R4/R6/R7 — for G1/G2 the classic set has no lobes to speak of (68
polygons touched by 2–3 points each), which is itself the demonstration; test models only for
anything model-derived; both directions; seeds 1–2 before calling a component an object.

---

## 7. Order of work

Regions to start from: AE→A spatial S = 250 and 500 (components 0, 1, 4; 2 as the coastal
case), A→AE spatial S = 500 (component 3; component 1 as the omission case). Seeds 1–2 in
parallel under the paper protocol; redo on recurring components.

| week | do | why |
|---|---|---|
| 1 | R0' typing; lobes and `edge_dist_m`; the edge-vertex table with DEM (GEE) and panel/county distances; G2 cheap form (edge-elevation spread A vs AE); R2; R4 (Moran's I, direction); R8 first maps | everything else reads the vertex table; the spread result alone already says whether the Zone A edge is a water surface |
| 2 | SDA (SSURGO) and NWI pulls for the four bounding boxes; G1 classification with the displaced-edge null; contour-phase test; FIS reports for the eight counties (an afternoon) | G1 is the primary test; the FIS text anchors it |
| 3 | R6 with lobe excision; R7 on all three region sets; T4 ceiling on seeds 1–2; write up | the practical claim, the comparison, the aggregate sentence |
| if time | G2 full form on components 0 and 4; DISDE; tie-in check; CNMS | the physical referee; the aggregate Y\|X number |

**Minimum viable for the deliverable:** R0', G1 with the displaced-edge null and the AE-lobe
contrast, G2 cheap form, R4 direction, R8 maps, and the FIS-report check. That is one week of
data work plus one of analysis.

---

## 8. Reading the outcomes together

| pattern | verdict |
|---|---|
| G1: region Zone A edge mostly not a level set and on S/M lines; AE edges level sets; G2 cheap: region spread ≫ AE spread; R4: commission dominates; FIS: "flood prone area maps / soils" | **H2 with mechanism** — the approximate outline was traced from soils / quadrangle sources and is not a water surface; the models cannot reproduce it from covariates; pooling teaches it (R6 should show harm) |
| G1: region Zone A edge on W lines (wetland/land-cover), not level set; AE edges level sets | **H2, different-rule form** — Zone A = wetland outline, Zone AE = water surface; two rules on the same covariates; a pooled model averages them |
| G1: region Zone A edges are level sets like AE's; G2 full: the basin fills to the edge; R7: distance to coast / visible water dominate; R4: commission | **Story 2 (H1 by site selection)** — the map is right, the AE-trained models learned where detailed studies are done; the regions are floodplain the coverage task should include; pooling helps (R6 should show benefit) |
| G1: level sets; G2 full: the basin does *not* fill to the edge | H2 without a source signature — a water-surface-shaped boundary at the wrong level; look at the FIS for the elevation source |
| G1: no class dominates; G2 cheap: spread neither like AE nor structured | **H3** — scattered error; coherence was imposed by the graph; say so, the E1 tradeoff curve is the figure |
| non-region Zone A lobes look like region lobes on G1 | Zone A is a W/S label everywhere; the *regions* are explained by where that label meets terrain the AE models never saw (R7, §5) — the claim about the label stands, the claim about the regions shifts to Story 2 |

Every row is reportable. The last row is the one to be ready for: it is the most likely
outcome in Florida, and it is still a paper.

---

## 9. Text for the paper (before / after; more than a prose change — Victor to review)

**Applied 2026-09-27** to `CS371_Cho_Li/acl_latex.tex` in a slightly revised form that names the
construction's lineage (Goodchild & Hunter 1997; Fortin, Drapeau & Jacquez 1996; Cohen et al.
2018): see `CS371_Cho_Li/CHANGES-2026-09-27.md` for the exact text as pasted, the prose fixes,
the novelty verdict (the construction is classic, the application is new — no fourth
contribution) and the five new `custom.bib` entries. The paragraphs below are the earlier drafts
and are kept for the record; §9.2 (outcome 5) was *not* applied and remains Victor's call.

**9.1 The GIS paragraph** (replaces "The GIS analysis then asks one question of each region:
where along the stream network does it begin and end? …" through "…a scattered set of points
has no boundary to compare.")

> The GIS analysis then asks one question of each region: which rule drew the label there? A
> detailed study places the Zone~AE boundary where the terrain meets the base flood elevation,
> so its boundary is a level set of elevation. The approximate methods overlaid the outline of
> the floodplain from soils maps, aerial photographs of past floods and topographic quadrangles
> \citep{fema_sg_unit3}, so a Zone~A boundary may instead follow a soil-survey line, a wetland
> edge, a contour, or the edge of the sheet it was drawn on. We sample each region's Zone~A
> boundary every 30~m and measure, with boundary overlap statistics \citep{jacquez1995}, how
> much of its length coincides with each candidate source---soil map-unit lines,
> wetland-inventory lines, land-cover edges, elevation level sets, and map-sheet or county
> edges---against Zone~AE boundaries in the same tiles and size-matched Zone~A boundaries
> outside the regions, under a null that displaces the boundary to measure how often such
> coincidences arise by chance. The same measurement classifies each stretch of boundary by
> whether a model could have reproduced it from its covariates: an elevation level set is the
> rule the detailed labels teach; a land-cover edge is a different rule on the same covariates;
> a soil-survey or administrative line is information no covariate carries. The regions are
> wetland basins in flat terrain rather than river corridors, so we then ask whether the terrain
> agrees with either label using the flood analysis appropriate to such terrain, a ponding
> analysis \citep{fema_gd84}: whether the mapped boundary is a single water surface, and whether
> the depressions the region occupies fill to it under the 1\%-annual-chance rainfall. Both
> analyses require the spatial penalty. A stretch of boundary long enough to classify and a
> basin that fills or does not exist only for a region; a scattered set of points has neither.

(`fema_gd84` is a new bib entry: FEMA, *Guidance for Flood Risk Analysis and Mapping: Shallow
Flooding Analyses and Mapping*, Guidance Document 84, December 2020. If §4.2 is not going to be
run, cut the ponding sentence to "…whether the mapped boundary is a single water surface, as a
1\%-annual-chance flood in a flat basin must be." and drop the citation.)

**9.2 Anticipated outcome 5** (replaces "Lastly, we expect the boundaries of those regions to
coincide with map-production breaks…" to the end of the paragraph)

> Lastly, we expect the Zone~A boundaries inside those regions to follow soil-survey and wetland
> lines where the terrain does not change, and the Zone~AE boundaries in the same tiles to follow
> elevation level sets, beyond what displaced boundaries and size-matched Zone~A boundaries
> outside the regions show. That pattern would indicate that the two zones encode two label
> rules---one a function of the covariates a model sees, the other not---so that a model fitted
> to one cannot reproduce the other, and a model fitted to both is taught the approximate
> method's outline. The reverse pattern, Zone~A boundaries that follow the terrain as closely as
> Zone~AE's and basins that fill to them, would instead mark floodplain that detailed studies
> never reached and that models trained on them extrapolate poorly to, a covariate shift that is
> equally informative for extending FEMA's coverage.

**9.3 Contribution 3** (replaces "Third, a boundary-alignment test that asks whether the
regions … begin and end at map-production boundaries or at landscape boundaries, a classic GIS
analysis that becomes possible only once the failures form regions.")

> Third, a label-source alignment test that asks which rule drew the Zone~A boundary inside the
> regions where higher in-domain accuracy predicts lower out-of-domain accuracy---an elevation
> level set, a wetland edge, a soil-survey line or a map-sheet edge---and whether the terrain
> agrees with it: a classic GIS boundary analysis that becomes possible only once the failures
> form regions, and that says whether the label is one a covariate model can reproduce at all.

**9.4 Two smaller items.** (a) Paragraph 5, last sentence, is missing a verb: "…we can then
select spatially coherent points will point to the exact regions…" → "…we can then select
spatially coherent points that point to the exact regions where models fail to absorb the
distribution shift and the assumption breaks down." (prose only). (b) The sentence in
paragraph 4 "We hypothesize that a model trained on pooled labels would fall short and reproduce
the errors the approximate method makes" now has its test: G1 says whether there is an
approximate-method outline to reproduce, R6 says whether pooling reproduces it. Consider
adding, after the T4 replication, one sentence to the Results opening: "Across six feature sets
the in-domain AUC of models trained on Zone~AE spans 0.90--0.94 while their AUC on Zone~A stays
at 0.84: the shortfall is not a shortage of covariates."

---

## 10. Data and references

**Data to add** (all free): 3DEP 1/3″ DEM (`USGS/3DEP/10m` on GEE); SSURGO map-unit polygons
and `muaggatt` for four bounding boxes (Soil Data Access, `SDA_Get_Mukey_from_intersection_with_WktWgs84`
for keys, `mupolygon` for geometry — the soilDB `SDA_spatialQuery` documentation shows the
`mupolygon` return ✅); NWI wetlands (USFWS state download for Florida, or the Wetlands
MapServer ✅); NLCD is already the `landcover` covariate; TIGER county lines and roads; PLSS
(BLM CadNSDI); FIS reports from the MSC for eight counties; USGS OFR 74-1111 (Florida
flood-prone-area quadrangles: Big Gum Swamp, Sanderson North/South, Taylor, Sanderson NW; 1974 ✅).
On disk already: flood-hazard polygons with geometry (`haz_FL_5070.parquet`, 5.4 GB — the
analysis runs on the Mac), panels, BFE lines, LOMRs, flowlines, `p100_24hr`.

**Checked this session** (cloud workspace, from the staged dataset and result files): the
component tables in §2; the T4 ceiling numbers; the look-alike positive shares in §5. Not
checked (needs the polygon geometry on the Mac): `edge_dist_m` for region points, lobe
perimeters, and whether region lobes touch Zone AE polygons — compute these first.

**References** (✅ page read this session or in the lit-review check; ⚠ to verify before citing)

- FEMA. *NFIP Floodplain Management Study Guide*, Unit 3, pp. 3-5, 3-6, 3-25 (approximate
  sources; FHBMs; $5,000–10,000 per mile) ✅ — already `fema_sg_unit3`.
- FEMA. *Guidance for Flood Risk Analysis and Mapping: Shallow Flooding Analyses and Mapping*,
  Guidance Document 84, Dec 2020 ✅ (reservoir-routing ponding; one BFE per ponding area).
- FEMA / Putnam County, FL. *Flood Insurance Study*, Feb 2012, p. 14 ✅ ("normal depth
  calculation and flood prone area maps").
- USGS. *Flood-prone area maps of Florida, 7½′ quadrangles: Big Gum Swamp, Sanderson North,
  Sanderson South, Taylor, Sanderson NW*. OFR 74-1111, 1974 ✅ (existence; the quads are in the
  Osceola NF / Baker–Columbia area — confirm which ones cc 4 overlaps); the delineation method
  printed on the sheets (past floods, soils, topography) ⚠ read a sheet.
- NRC 2007. *Elevation Data for Floodplain Mapping*, ch. 3 ✅ (Floodplain Boundary Standard,
  ±½ contour interval for approximate studies).
- Jacquez 1995; Fortin, Drapeau & Jacquez 1996; Jacquez, Maruca & Fortin 2000 ✅ (boundary
  overlap statistics; already in `custom.bib` as `jacquez1995`).
- Moreno-Torres, Raeder, Alaiz-Rodríguez, Chawla & Herrera 2012. A unifying view on dataset
  shift in classification. *Pattern Recognition* 45(1):521–530 (concept shift vs covariate
  shift) — standard reference, add to bib.
- Cai, Namkoong & Yadlowsky 2023. DISDE. arXiv:2303.02011 ✅ (in the project index).
- Wu, Q. et al. 2019. Efficient delineation of nested depression hierarchy in DEMs using the
  level-set method. *JAWRA* 55(2) ✅ (title/venue).
- Barnes, Callaghan & Wickert 2021. Computing water flow through complex landscapes, Part 3:
  Fill–Spill–Merge. *Earth Surf. Dynam.* 9:105–121 ✅.
- WhiteboxTools user manual, Hydrological analysis (`FillDepressions`, `DepthInSink`, `Sink`,
  `StochasticDepressionAnalysis`) ✅.
- Verdi & Dixon 2011. Magnitude and frequency of floods for rural streams in Florida, 2006.
  USGS SIR 2011-5034 ✅ (four hydrologic regions; drainage area and a storage factor) — only
  needed for the riverine variant (§4.3).
- SSURGO `muaggatt` columns `flodfreqdcd`, `flodfreqmax`, `pondfreqprs`, `hydclprs`,
  `drclassdcd`, `drclasswettest`, `hydgrpdcd` ✅ (gSSURGO MUAGGATT metadata; `flodfreqdcd` is
  "the annual probability of a flood event expressed as a class", dominant condition).
- EAA Zone AE base flood elevations in Palm Beach County (a basin-wide design elevation?) ⚠ read
  the Palm Beach FIS before describing component 3 that way.
