# Project plan — Label provenance as distribution shift in flood susceptibility

**v3, 2026-09-19** · supersedes v2 (2026-09-16, in git)
**Scope:** AI Research Practicum, 8 weeks from mid-September
**Anchor:** Salaudeen et al., *Aggregation Hides OOD Generalization Failures from
Spurious Correlations*, NeurIPS 2025 · arXiv:2510.24884 ·
github.com/olawalesalaudeen/OODSELECT
**Companions:** `docs/SPATIAL-OODSELECT.md` (the method and the full Florida run),
`aotil/README.md` (the pilot), `docs/COVARIATES.md`, `docs/SAMPLING-RESULTS.md`,
`docs/COVARIATE-AUDIT.md`, `docs/LABELING-PROBLEMS.md`

---

## 0. What changed in v3

The data and the pilot are done, and the pilot changed the method.

OODSelect finds accuracy-on-the-inverse-line in the Zone A / Zone AE benchmark:
under the `band` protocol, r = −0.80 to −0.90 on held-out models, the
most-misclassified baseline stays positive, the permutation null sits near zero,
and it holds across family-disjoint model splits (`results/aotil_controls_FL.csv`).
The subsets are describable from quarantined provenance metadata — but they are
**geographically scattered**, touching 15–22 of 23 evaluation tiles at the 5–10%
budget. The GIS analysis this project promised reads hydrological and land-use
context off a *place*, so Stage 3 becomes a **spatially regularized OODSelect**
(a graph-Laplacian smoothness penalty on the selection weights), evaluated as a
coherence–inversion tradeoff against the unconstrained method and every control.

Also in v3: the OOD>ID objective from v2 is dropped; RQ4 (observed flooding) is
deferred out of the deliverable; Florida is the primary case and Iowa/Missouri
the regime contrast if time allows; `band` is the primary negative protocol and
`shadedX` the sensitivity arm; §3–§4 now describe the data as built rather than
as planned.

---

## 1. The idea

FEMA's National Flood Hazard Layer draws the 1%-annual-chance floodplain two
different ways. **Zone AE** boundaries come from detailed hydraulic studies with
published Base Flood Elevations. **Zone A** boundaries come from approximate
methods — no hydraulic model, no BFE. Same risk designation, same regulatory
consequence, completely different evidentiary basis.

Anyone training an ML model on NFHL polygons is therefore fitting a label
produced by two different processes, and which process was used is spatially
patterned. A model trained on AE areas learns to approximate *hydraulic
modelling output*. A model trained on A areas learns to approximate *a rough
terrain heuristic*. Those are different target functions sharing a column name.

> **Thesis.** Label provenance is itself a distribution shift. It is invisible in
> aggregate, it is recorded in the data, and it can be mistaken for geographic
> shift. We use a spatially regularized OODSelect to find *where* it bites, as
> regions, and provenance metadata against terrain, hydrology, climate and
> urbanicity — with GIS context for each region — to explain them.

**Why this version.** Geography confounds everything at once — terrain, climate,
land use and mapping practice all move together. Provenance changes one thing
while holding the rest closer to fixed, which makes it a near-natural experiment
rather than a correlational study. It is also the kind of selection-induced split
that Salaudeen et al. (arXiv:2504.00186) argue is the only kind that tests
robustness at all: their audit found selection-induced splits like CivilComments
well-specified at R = −0.47, while PACS (0.84), Camelyon (0.78) and FMoW (0.87)
all failed.

**Why regions.** A subset that is a place can be validated on points the
selection never saw, mapped against flowlines, coastlines, counties and FIRM
panels, and described in the terms a hydrologist and a floodplain manager both
use. A subset scattered across a state can only be described by the metadata of
its points. The constraint costs inversion strength; how much is itself a result.

---

## 2. Research questions

**RQ1 — Does provenance shift exist?** Does a model trained on Zone AE degrade
on Zone A (and vice versa) beyond what urbanicity explains?
*Status:* the Florida pilot says yes under `band`; the pooled attribution found
the `band` gap carried by provenance and the `shadedX` gap by terrain
(`results/`, `aotil/README.md`). Re-confirm on the 2026-09-19 dataset.

**RQ2 — Is the degradation concentrated, and where?** Unconstrained OODSelect
answers the first half: yes, in a point type. The second half is new — does a
spatially regularized selection find geographically coherent regions where the
line inverts, at what cost in inversion strength, and does the inversion survive
transfer to held-out points inside the region?

**RQ3 — Can the regions be named?** Do they align with provenance (study regime,
panel scale, BFE density, zone subtype, source study, effective date) better than
with terrain, hydrology, climate, urbanicity or processing artifacts — measured
against a block permutation null, not eyeballed — and what does the GIS context
of each region say?

**RQ4 — Which side is right?** *Deferred, out of the practicum deliverable.*
Where observed flooding is available, does an AE-trained model predict it better
in Zone A areas than the Zone A map does? Data pointers kept in §3.5.

---

## 3. Data — as built

### 3.1 Label

Binary `label` from NFHL Special Flood Hazard Area polygons, ten states,
**878,336 points in 16,397 blocks**, every stratum balanced between domains.

- **Positives:** points inside the SFHA, 200 m minimum separation, inside the
  ~110–130 m-wide zone ribbons.
- **Negatives:** two protocols, both provenance-blind, carried in `neg_protocol`.
  `band` (a fixed distance band outside the boundary) is **primary**; `shadedX`
  (0.2%-chance polygons) is the **sensitivity arm**. The 0.15 AUC gap between
  them is itself a result: the negative-sampling sensitivity the labelling
  literature puts at 0.13–0.24 AUC, reproduced on one footprint with everything
  else fixed. Detail in `docs/SAMPLING-RESULTS.md` and `docs/LABELING-PROBLEMS.md`.

### 3.2 Features — the model sees only these

`data/processed/FEATURES.txt` is the authoritative list (45 columns as of
2026-09-19): GEE terrain minus absolute elevation (`slope`, aspect sin/cos,
curvature, `tpi_300`, `tri_300`, `tpi_broad`), `hand`, `log10_upa`, surface
(`landcover`, `impervious`, `gsw_occurrence`, `water_frac_900m`), NHDPlus
catchment context (`catch_area_km2`, `nearest_stream_order`, `drainage_density`,
`channel_slope`, `log10_dist_river`, `log10_dist_coast`, `log10_upstream_area`),
StreamCat catchment/watershed means (BFI, hydraulic conductivity, wetness index,
K-factor, precipitation, temperature, runoff, clay/sand, permeability), and
NOAA Atlas 14 depths (`p100_*`, `p2_24hr`, `growth_100_2`).

Three commitments from `docs/COVARIATE-AUDIT.md`: **absolute elevation is out**
(it identifies the state at 58% vs a 10% baseline and supplied 0.26 of the
A-vs-AE AUC; relative elevation is carried by `hand` and the TPIs); **TWI and
SPI are unusable** from this stack (`docs/COVARIATES.md` §4); **`lon`/`lat` and
every `meta_*` column are never features.**

### 3.3 Metadata — carried, never fed to the model

| Field | Use |
|---|---|
| `domain` from `FLD_ZONE` (A vs AE) | **the domain variable.** `STATIC_BFE` is *not* the discriminator — AE elevations live as line features in `S_BFE` |
| `meta_zone_subty`, `meta_study_typ`, `meta_study_regime` | label refinement; regime = classic / BLE risk-class / redelineation |
| `meta_panel_scale`, `meta_panel_typ`, `meta_eff_date`, `meta_source_cit`, `meta_fld_ar_id` | provenance: how, when and by which study the polygon was drawn |
| `meta_bfe_km_per_km2` | the continuous provenance gradient (40× across states) |
| `meta_dist_to_sfha_m`, `meta_poly_area_km2`, `meta_elev` | leakage inventory — quarantined |
| `impervious` decile | urbanicity stratum |
| `state`, `county_fips`, `meta_comid`, `block_id`, `tile_30km/50km/100km`, `lon`, `lat` | spatial CV, graphs and mapping |

### 3.4 Domains and scope

Primary domain variable: **Zone A vs Zone AE, within state**, with the paired
within-block contrast as the primary comparison (in eight of ten states fewer
than a third of blocks hold both domains, so the full-state contrast alone
confounds *where* studies are done with *how*).

States in the dataset: FL, LA, IA, CO, AZ, NC, VT, TX, MN, MO. NC is excluded
from primary analysis (0.7% Zone A). **Analysis scope for the practicum:
Florida primary** (chosen for coverage, not gap size — `aotil/README.md`);
**Iowa and Missouri** as the regime contrast if time allows (IA: 84% of Zone A is
BLE risk-class, 50% of AE is redelineation; MO: BLE). Florida is one regime
(classic, ~78% NP), so between-place provenance contrast needs IA/MO.

### 3.5 Validation data (RQ4, deferred)

USGS High-Water Marks, the USGS Flood Event Viewer, and the Flood Inundation
Mapper. Sparse and biased toward events where crews deployed. Kept as a pointer.

---

## 4. The task

**One row = one sampled point.** No time axis: this predicts flood-prone
*places*, not flood *events*.

- **Input:** the columns in `FEATURES.txt`.
- **Output:** per-point probability of being flood-prone; per-model score is the
  predicted probability of the observed class (AUC has no per-example
  decomposition and a 0.5 threshold means different things across cells).
- **Splits:** on `tile_50km`, never on `block_id` or points (GroupKFold on 10 km
  blocks is not spatial CV — A-vs-AE separability falls 0.86 → 0.69 from 10 km
  to 100 km). Test tiles must hold both domains so ID and OOD are scored on the
  same geography.
- **Derived:** the A↔AE transfer matrix; the discovered regions; the coherence–
  inversion curve; the alignment table; per-region GIS summaries.

**Model population.** The `aotil/zoo.py` population: seven learner families
(HGB, RF, ET, LR, MLP, DT, NB) × capacity × feature subset (0.3/0.6/1.0) × row
subsample (0.15/0.5/1.0), round-robin across families so a family-disjoint
control is possible. Pilot: 126 per cell, ID spread 0.27–0.33, aggregate ID→OOD
r = +0.85 to +0.96. Full run: ≥ 300 per cell, stopping when the aggregate
correlation moves < 1%.

---

## 5. Pipeline

**Stage 0 — Assemble.** Done. NFHL subset to ten states; points sampled and
audited; Earth Engine, NHDPlus/StreamCat, Atlas 14 covariates merged
(`dataset_*.parquet` regenerated 2026-09-19 13:11).

**Stage 1 — Characterise the split.** Done. Flood susceptibility 0.90 AUC vs
A-vs-AE separability 0.71–0.74 after the elevation fix; propensity overlap 68%;
A/AE × urbanicity has support. `docs/COVARIATE-AUDIT.md`, `docs/SAMPLING-RESULTS.md`.

**Stage 2 — Transfer matrix.** `baselines.py`, both directions, paired and
unpaired, with permutation nulls (`results/baselines_*`, `tile_gaps_*`,
`explain_*`). Re-run on the current dataset before quoting numbers.

**Stage 3 — Spatially regularized OODSelect.** The paper's objective,
`min_s corr(acc_ID, acc_s^OOD) + λ(S − ‖s‖₁)²`, reimplemented in
`aotil/oodselect.py` and validated in the pilot with random, most-misclassified,
permutation-null and family-disjoint controls. New: a graph-Laplacian term
`μ · wᵀLw / (S·k)` over a kNN (or catchment, or provenance) graph on the OOD
points, swept over μ, with every control re-run at the same μ and a
held-out-point transfer test inside the region. **Full specification, experiments
E0–E6, effort and pitfalls: `docs/SPATIAL-OODSELECT.md`.**

**Stage 4 — Alignment battery, per region.** For each region: prevalence shift
and membership-predictability AUC for every candidate group below, GroupKFold on
tiles, against a block permutation null; then the GIS context — map with
flowlines, coast, counties and FIRM panels, plus a one-row physical and
provenance summary.

| Candidate | Variables |
|---|---|
| Label provenance | `meta_study_regime`, `meta_study_typ`, `meta_zone_subty`, `meta_panel_scale`, `meta_panel_typ`, `meta_bfe_km_per_km2`, `meta_eff_date`, `meta_source_cit` |
| Urbanicity / surface | `impervious`, `landcover`, `water_frac_900m`, `gsw_occurrence` |
| Terrain / hydrology | `hand`, `slope`, TPIs, `log10_upa`, distances to river and coast, `drainage_density`, `nearest_stream_order`, StreamCat |
| Climate | `precip8110*`, `tmean8110*`, `runoff*`, Atlas 14 depths |
| Processing artifacts | DEM source / resolution where recorded |

**Stage 5 — Decompose.** DISDE (arXiv:2303.02011) on the A↔AE gap, if time.
Shared support measured at 68%, so it is viable.

**Stage 6 — Intervene (stretch).** Regions as groups for Group DRO, or a
re-study priority map from the regions. Improvements are not required by the
project criteria; an explained difference is enough.

---

## 6. Result analysis — what each outcome means

| Outcome | Reading |
|---|---|
| A↔AE gap survives urbanicity stratification | Provenance shift is real. The headline claim. |
| Gap vanishes under stratification | It was urbanicity. Still a clean rural/urban flood transfer result. |
| Gap is strongly asymmetric | Report the asymmetry — direction is informative about which label set is more learnable. |
| Inversion survives at the coherent operating point, null flat, held-out-point transfer holds | The provenance failure is place-organised; the regions are the paper's objects. |
| Inversion survives weakened | Partly place-organised; the tradeoff curve decomposes the pilot's −0.8 into place and point-type components. |
| Inversion dies for every μ > 0 | AoTIL here is carried by a point type present everywhere, not by places. Consistent with the thesis, reported as the negative result, curve as the figure. |
| Provenance graph beats geometric graph at equal coherence | The shift is in how the maps were made, measured directly. |
| Regions align with provenance | Regulatory data heterogeneity masquerading as geographic shift. Novel. |
| Regions align with urbanicity | Rural/urban transfer paper. |
| Regions align with terrain or climate | Conventional geographic-shift finding, solid but less novel. |
| Regions align with DEM resolution | A processing artifact wearing a geography costume. Worth a warning to the field. |
| Regions align with nothing | Reproduces the anchor paper's own limitation in the setting where we predicted it would not apply. Honest negative result. |
| Intervention improves worst-region performance | The diagnosis-to-repair loop closes (stretch). |

Most rows are reportable. That is the reason this project is a reasonable bet
for the weeks that remain.

---

## 7. Limitations

**The label is a model output, not observed flooding.** Everything downstream
inherits this. Strictly, the project studies agreement with FEMA maps, not with
floods. RQ4 would partially escape it and is deferred; state this plainly.

**Negative sampling is a free parameter with large effects.** "Did not flood" is
never observed. The two protocols differ by 0.15 AUC on one footprint, and the
inverse line survives the controls under `band` only — under `shadedX` it is
inseparable from difficulty. Sanyal et al. (arXiv:2406.19049) show label noise
plus interpolation is exactly the condition under which ID/OOD relationships
invert. Mitigation: provenance-blind protocols, both reported, non-interpolating
controls (shallow trees, strong regularisation) in the population.

**The spatial penalty costs inversion strength by construction.** The pilot's
−0.8 comes from one point type — Zone A positives within ~30 m of the SFHA
boundary — picked off every ribbon in the state. A region must include interior
points, which dilutes that signal. Report the tradeoff curve, never a single
operating point, and never choose μ by test r.

**Pseudo-replication inside a region is worse than across the state.** Points
200 m apart in a corridor are near-duplicates and the largest 1% of polygons
supply 8–26% of a domain. Bootstrap at tile level; validate regions on held-out
points, not only held-out models.

**Mixture arithmetic is the default null for any OOD > ID finding.** "Shift is
Good" (arXiv:2510.25108) proves that for K≥3 subpopulations with non-constant
learning curves, a mismatched training mixture beats a matched one for almost
every test mixture. Rule it out before attributing anything to covariates.

**Spatial autocorrelation invalidates the obvious statistics.** Random CV
produces meaningless AUC, and any spatially clustered subset shows apparent
alignment with any spatially clustered variable. Everything needs tile blocking
and block permutation nulls — including the membership models in Stage 4.

**Multicollinearity undermines the attribution step.** Slope, TPIs, HAND and the
StreamCat metrics are correlated. Attribution splits credit unstably; report
covariate *groups*, not single features.

**Model-population diversity is thin on tabular data.** No pretrained backbones
to vary. The population spreads 0.27–0.33 in ID performance, which was enough
for the pilot; if a null result appears, check the spread first.

**DISDE requires shared support.** Measured at 68% of points in [0.05, 0.95].

**No prior baseline exists.** Nobody has published an A/AE transfer matrix.

**The spatial penalty is our modification.** Nobody has tested it. μ = 0 must
reproduce the pilot exactly, and the permutation null must be re-run at every μ.

**Susceptibility is not hazard and not risk.** No return period, no exposure, no
vulnerability. Do not let the writeup drift into risk language.

**Findings are US- and FEMA-specific.** The mechanism generalises; the specific
result does not.

---

## 8. Timeline — weeks remaining

Week 1 (data, viability, pilot) is done. Dates assume the eight weeks run
mid-September to mid-November.

| Week | Work |
|---|---|
| 2 | Cells regenerated with spatial columns, ≥300 models, 3 seeds, persisted in the repo. E0 covariance map and Moran's I — the go/no-go on regions. Laplacian term and coherence metrics implemented; μ = 0 reproduces the pilot. |
| 3 | E1 μ sweep with all controls at every μ; tradeoff figure; coherent operating point. |
| 4 | E2 held-out-point transfer; E3 characterisation and membership models with block permutation null. |
| 5 | E4 maps and per-region GIS summaries; Stage 2 re-run on the current dataset; decide on E5/E6. |
| 6 | E5 provenance-vs-geometric graph and/or E6 Iowa–Missouri; confound checks (noise controls, mixture null). |
| 7 | Robustness across seeds and protocols; DISDE if time; figures final. |
| 8 | Writeup and presentation. |

---

## 9. Immediate next steps

1. `zoo.py`: save `lon, lat, block_id, tile_50km, county_fips, meta_comid,
   meta_source_cit` in the cell `.npz`; write cells to `data/interim/aotil/cells/`
   and commit them back to the folder. Re-stage `dataset_FL.parquet` first.
2. Regenerate the four Florida cells at 300 models, seeds 0–2; `gate.py` at 150 /
   225 / 300 to confirm the correlation has stabilised.
3. E0: per-point covariance map, Moran's I against 200 permutations. This decides
   whether any μ > 0 can succeed.
4. `oodselect.py`: `--mu`, `--graph knn|block|catchment|provenance`, `--knn 10`;
   `aotil/graph.py` for adjacency and the §4 coherence metrics; confirm μ = 0
   matches `results/aotil_controls_FL.csv`.
5. `analyse.py`: sweep μ, `null-select` at the same μ, coherence metrics in every
   row; write `results/aotil/`.
6. Then E2–E4, then decide E5/E6 by week 5.
