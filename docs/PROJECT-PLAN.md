# Project plan — Label provenance as distribution shift in flood susceptibility

**v2, 2026-09-16** · supersedes the 13-week ACS/Folktables plan
**Scope:** AI Research Practicum, 8 weeks
**Anchor:** Salaudeen et al., *Aggregation Hides OOD Generalization Failures from
Spurious Correlations*, NeurIPS 2025 · arXiv:2510.24884 ·
github.com/olawalesalaudeen/OODSELECT

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
> shift. We use OODSelect to find where it bites, named covariates to explain
> where, and observed flooding to test which side is right.

**Why this version.** Geography confounds everything at once — terrain, climate,
land use and mapping practice all move together. Provenance changes one thing
while holding the rest closer to fixed, which makes it a near-natural experiment
rather than a correlational study. It is also the kind of selection-induced split
that Salaudeen et al. (arXiv:2504.00186) argue is the only kind that tests
robustness at all: their audit found selection-induced splits like CivilComments
well-specified at R = −0.47, while PACS (0.84), Camelyon (0.78) and FMoW (0.87)
all failed.

---

## 2. Research questions

**RQ1 — Does provenance shift exist?** Does a model trained on Zone AE areas
degrade on Zone A areas (and vice versa) beyond what urbanicity explains?

**RQ2 — Is the degradation concentrated?** Does OODSelect find coherent subsets
of the target where transfer is anomalous, rather than the loss being uniform?

**RQ3 — Can the subsets be named?** Do they align with provenance, urbanicity,
terrain, climate, or processing artifacts — measured against a spatial
permutation null, not eyeballed?

**RQ4 — Which side is right?** Where observed flooding is available, does an
AE-trained model predict it better in Zone A areas than the Zone A map does?

---

## 3. Data

### 3.1 Label

Binary `flood_prone` from NFHL Special Flood Hazard Area polygons.

- **Positives:** cells inside the SFHA.
- **Negatives:** sampled from a fixed distance band outside the SFHA boundary.
  The protocol must be **identical and provenance-blind** on both sides, with
  band width reported and swept for sensitivity. This is the single easiest
  place in the whole project to manufacture a result.

### 3.2 Features — the model sees only these

| Group | Variables | Source |
|---|---|---|
| Terrain | elevation, slope, plan/profile curvature, TPI, TRI | USGS 3DEP |
| Hydrologic indices | TWI, SPI | derived from DEM |
| Channel proximity | HAND, distance to river, drainage density | CFIM/ORNL (HAND precomputed for CONUS), NHDPlus HR |
| Climate | design-storm rainfall depth | NOAA Atlas 14 / PRISM |
| Surface | land cover class, imperviousness % | NLCD |
| Soil | hydrologic soil group | gNATSGO |

≈13 features, every one physically meaningful on its own. That is the point:
SHAP attribution lands on statements a hydrologist recognises, with no
area-level covariate layer bolted on afterwards.

### 3.3 Metadata — carried, never fed to the model

| Field | Use |
|---|---|
| `FLD_ZONE`, `STATIC_BFE` (null vs populated) | the domain variable |
| `ZONE_SUBTY`, `SFHA_TF` | label refinement |
| study effective date (join to `S_FIRM_PAN`) | vintage stratum |
| NLCD imperviousness decile | urbanicity stratum |
| DEM source / resolution / lidar vintage | artifact check |
| state, county, HUC, lat/lon | spatial CV and mapping |

`STATIC_BFE` populated-vs-null is the cleanest A/AE discriminator — a numeric
null check rather than a string parse.

### 3.4 Domains

Primary domain variable: **Zone A vs Zone AE**, stratified by urbanicity decile.
Secondary: state or HUC, for the conventional geographic comparison.

Scope: ~10 states spanning real physiographic contrast — FL, LA, IA, CO, WA, AZ,
NC, VT, TX, MN — expanding only if the pilot holds.

### 3.5 Validation data (RQ4)

USGS High-Water Marks, the USGS Flood Event Viewer, and the Flood Inundation
Mapper. Sparse and biased toward events where crews deployed, so this is a
spot-check on selected events, not a systematic evaluation.

---

## 4. The task

**One row = one raster cell**, 10–30 m. No time axis: this predicts flood-prone
*places*, not flood *events*.

- **Input:** the ~13 covariates above.
- **Output:** per-cell probability of being flood-prone.
- **Derived:** a susceptibility surface binned into five classes (the artifact a
  planner uses); AUC-ROC and balanced accuracy on held-out regions; the
  discovered subsets; the alignment table; the A↔AE transfer matrix.

**Model population.** RF, XGBoost, LightGBM, MLPs of varying depth, logistic
regression, kNN — varied by algorithm family, hyperparameters, feature subsets,
subsample fraction, seed, **and position along the training trajectory**. That
last axis is not optional: Teney et al. (arXiv:2209.00613) showed that sampling
fixed-epoch models across seeds reproduces exactly the vertical scatter that
hides inversion, while varying epochs within a seed reveals it. Target
N ≈ 1,500–2,000, stopping when the ID/OOD correlation estimate moves <1%.

---

## 5. Pipeline

**Stage 0 — Assemble.** Sample points across the 10 states, label from NFHL,
attach covariates, attach metadata. Earth Engine hosts most layers; HAND is
already computed for CONUS.

**Stage 1 — Characterise the split.** Before any modelling: compute the A/AE
split by imperviousness decile. If the two are near-perfectly separated,
stratification has no support and the design must change. Also compute the
national A vs AE share and the effective-date distribution — if Zone A is a
rounding error, the premise evaporates.

**Stage 2 — Transfer matrix.** Train on A → test on AE, and the reverse, within
each urbanicity stratum. Both directions; the asymmetry is data.

**Stage 3 — OODSelect.** Two objectives:
- the original, `min_s corr(acc_ID, acc_s^OOD) + λ(S − ‖s‖₁)²`
- an OOD>ID variant, `min_s −E_models[acc_s^OOD − acc_ID] + λ(S − ‖s‖₁)²`

Validate the reimplementation against their released subsets on one of their own
benchmarks before trusting it here. Baselines: random selection,
most-misclassified, and whole-stratum selection.

**Stage 4 — Alignment battery.** For each discovered subset, score alignment
against every candidate below by (i) prevalence shift and (ii) membership
predictability AUC, both against a spatial permutation null.

| Candidate | Variables |
|---|---|
| Label provenance | Zone A/AE, `STATIC_BFE` null, study vintage |
| Urbanicity | imperviousness, NLCD developed class |
| Terrain regime | relief, HAND distribution, slope |
| Hydrologic setting | coastal vs inland, drainage density, distance to river |
| Climate | rainfall regime, aridity |
| Processing artifacts | DEM source, resolution, lidar vintage |

**Stage 5 — Decompose.** DISDE (arXiv:2303.02011) on the A↔AE gap. Does it land
in the X-shift terms (composition) or the Y|X term (the feature–label
relationship genuinely changed)? The Y|X case is the clean result.

**Stage 6 — Intervene.** Restrict training to AE only; reweight by study quality;
or use the discovered subsets as groups for Group DRO. Measure on held-out
regions. Then RQ4: compare against observed flooding.

---

## 6. Result analysis — what each outcome means

| Outcome | Reading |
|---|---|
| A↔AE gap survives urbanicity stratification | Provenance shift is real. The headline claim. |
| Gap vanishes under stratification | It was urbanicity. Still a clean rural/urban flood transfer result, which is also unpublished. |
| Gap is strongly asymmetric | Report the asymmetry — direction is informative about which label set is more learnable. |
| A→AE shows OOD > ID | Coarse supervision generalises better; AE labels are more internally consistent. Must survive the confound checks in §7. |
| Subsets align with provenance | Regulatory data heterogeneity masquerading as geographic shift. Novel. |
| Subsets align with urbanicity | Rural/urban transfer paper. |
| Subsets align with terrain or climate | Conventional geographic-shift finding, solid but less novel. |
| Subsets align with DEM resolution | A processing artifact wearing a geography costume. Worth a warning to the field. |
| Subsets align with nothing | Reproduces the anchor paper's own limitation in the setting where we predicted it would not apply. Honest negative result. |
| AE-trained model beats the Zone A map against observed floods | Transfer learning from detailed-study areas outperforms approximate mapping where FEMA never studied properly. The strongest possible outcome. |
| Intervention improves worst-stratum performance | The diagnosis-to-repair loop closes. |

Five of six alignment outcomes are reportable. That is the reason this project is
a reasonable bet for eight weeks.

---

## 7. Limitations

**The label is a model output, not observed flooding.** Everything downstream
inherits this. Strictly, the project studies agreement with FEMA maps, not with
floods. RQ4 partially escapes it; the rest does not. State this plainly rather
than letting a reader assume otherwise.

**Negative sampling is a free parameter with large effects.** "Did not flood" is
never observed. Sample uniformly and the task is trivially easy; sample tight to
the boundary and it is brutally hard. Worse, Zone A boundaries are *noisier*, so
a buffer-based sampler injects more label noise on the A side — which alone
could produce the effect. Sanyal et al. (arXiv:2406.19049) show label noise plus
interpolation is exactly the condition under which ID/OOD relationships invert.
Mitigation: identical provenance-blind protocol, width sweep, and their
non-interpolating controls (early stopping, stronger regularisation). If the
effect dies under those, it was noise.

**Mixture arithmetic is the default null for any OOD > ID finding.** "Shift is
Good" (arXiv:2510.25108) proves that for K≥3 subpopulations with non-constant
learning curves, a mismatched training mixture beats a matched one for almost
every test mixture — with zero transfer and no change in any conditional. Must
be ruled out before attributing anything to covariates.

**Spatial autocorrelation invalidates the obvious statistics.** Adjacent cells
are near-duplicates, so random CV produces meaningless AUC, and any subset of
spatially clustered data shows apparent alignment with any other spatially
clustered variable. Everything needs spatial blocking and permutation nulls.

**Multicollinearity undermines the attribution step.** Elevation, slope, TWI and
HAND are all DEM-derived and correlated. SHAP splits credit among correlated
features unstably, sometimes arbitrarily favouring one. Attribution is Stage 4,
so this is load-bearing, not cosmetic.

**Provenance may be inseparable from urbanicity.** Stratification only works if
each cell of the 2×2 is populated. If A and AE are near-perfectly separated by
development, there is no within-stratum comparison to make and the design must
change. This is why Stage 1 comes before any modelling.

**Model-population diversity is thin on tabular data.** No pretrained backbones
to vary. If diversity is inadequate, a null result may be an artifact of the
model population rather than a fact about the data — and there is no way to tell
after the fact.

**DISDE requires shared support.** If A and AE regions do not overlap in
covariate space, the decomposition is simply invalid. Check before running it.

**No prior baseline exists.** Nobody has published an A/AE transfer matrix, so
there is nothing to anchor against and no external check on whether the numbers
are sane.

**Observed-flood validation is sparse and biased.** High-water marks exist where
crews deployed, which is where damage was notable — not a random sample of
flooding.

**Susceptibility is not hazard and not risk.** This models propensity given
static conditions. No return period, no exposure, no vulnerability. Do not let
the writeup drift into risk language.

**The OOD>ID objective is unvalidated.** It is our modification; nobody has
tested it. An implementation bug would be invisible without a reference result.
Mitigation: validate the *original* objective against their released subsets
first, then change one thing.

**Findings are US- and FEMA-specific.** Zone A/AE is an artifact of one
country's regulatory structure. The mechanism generalises; the specific result
does not.

**Two of eight weeks go to plumbing.** Data assembly is not science and it can
slip. Starting with 10 states rather than 50 is the hedge.

---

## 8. Timeline

| Week | Work |
|---|---|
| 1 | NFHL schema pulled from FEMA directly. A/AE share, effective-date distribution, A/AE × imperviousness cross-tab. **Go/no-go on the premise.** |
| 2 | Assemble the 10-state table: sample points, covariates, metadata. Negative-sampling protocol fixed and documented. |
| 3 | Model population to ~1,500. Transfer matrix, both directions, stratified. |
| 4 | OODSelect implemented and validated against released subsets; both objectives run; three baselines. |
| 5 | Alignment battery with permutation nulls. Maps. |
| 6 | DISDE decomposition; confound checks (noise controls, mixture null). |
| 7 | Intervention + RQ4 observed-flood spot-check. |
| 8 | Writeup and presentation. |

---

## 9. Immediate next steps

1. Download NFHL directly from FEMA for two contrasting states; confirm
   `FLD_ZONE`, `STATIC_BFE`, `ZONE_SUBTY`, and the `S_FIRM_PAN` date join.
2. Compute the A/AE share and the A/AE × imperviousness cross-tab. This one
   table decides whether the project is viable.
3. Clone github.com/olawalesalaudeen/OODSELECT; confirm the released subsets are
   present and the objective matches the paper.
4. Fix and write down the negative-sampling protocol before generating any data.
