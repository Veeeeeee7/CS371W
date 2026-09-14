# Project plan — Explaining OOD failure sets with GIS

**Status:** draft v1, 2026-09-14
**Scope:** AI Research Practicum project, Fall 2026 (~13 weeks)
**Anchor paper:** Salaudeen, Zhang, Alhamoud, Beery, Ghassemi. *Aggregation Hides
Out-of-Distribution Generalization Failures from Spurious Correlations.* NeurIPS 2025.
arXiv:2510.24884 · code: https://github.com/olawalesalaudeen/OODSELECT

---

## 1. The idea

OODSelect shows that "accuracy-on-the-line" — the reassuring finding that models
with higher in-distribution accuracy also do better out-of-distribution — is an
artifact of **aggregation**. Disaggregate the OOD set and you find large,
coherent subsets on which the relationship inverts: better ID accuracy predicts
*worse* OOD accuracy (accuracy-on-the-inverse-line, AoTIL). On chest X-rays,
subsets covering >70% of the standard OOD data flip the correlation to strongly
negative.

The paper's own stated limitation is that it cannot say **what those subsets
are**. Their VLM- and LLM-generated descriptions were inconsistent, and they
attribute this to spurious features being subtle, high-dimensional, or
imperceptible. Metadata rescued the chest X-ray case only because demographic
and clinical attributes happened to be recorded.

Geospatial data is the setting where that limitation dissolves. Every example
carries coordinates, and every location has an arbitrarily rich stack of named,
public covariates attached to it — land cover, urbanicity, density, climate,
industry mix, infrastructure. If the selected subset concentrates somewhere, you
can draw it, and then you can ask what that somewhere is made of.

> **Thesis.** In geospatial ML the aggregation that hides OOD failure is
> *geographic* aggregation. So the hidden failure sets have a map — and a map,
> joined against GIS layers, is an explanation.

---

## 2. Why this is a contribution and not a port

- **OODSelect has only been run on images and text.** Never tabular, never
  geospatial. Running it somewhere new is table stakes; the point is that the new
  setting answers the question the original could not.
- **The geo-robustness literature measures aggregate degradation.** EarthShift
  (2605.29330), GeoSpOT (2604.16220), GeoBS (2509.23482) all quantify how much
  performance drops when you move regions. None of them disaggregate *within* the
  target region to ask whether the drop is uniform or concentrated in an
  inverse-correlated subset. That is a real gap.
- **The nearest relative is not actually close.** "Discovery and Spatial
  Characterisation of Multiple Shortcut Groups" (2608.14051) sounds like this
  project, but its "spatial" means *within-image pixel regions*, not geographic
  space. Worth citing prominently and distinguishing in one sentence.
- **There is a natural point of contact.** Haoran Zhang is a co-author on both
  the OODSelect paper and GeoSpOT. That is the single most useful email you could
  send about this project.

---

## 3. Research questions

**RQ1 — Existence.** Does AoTIL appear in cross-state GIS prediction? Run
OODSelect on pooled cross-state OOD data and ask whether subsets with negative
ID/OOD correlation exist, how large they get, and how negative they go.

**RQ2 — Geography.** Are the selected subsets spatially structured, or spatially
random? Aggregate selections to a geographic unit, map them, and test for spatial
clustering against nulls that already account for state identity.

**RQ3 — Explanation.** Can named GIS covariates predict subset membership? This
is the payoff: if a sparse, interpretable model on public geo layers recovers the
subset, you have described a failure mode the original method could only point at.

**RQ4 — Method (stretch).** Does adding a spatial-coherence penalty to the
OODSelect objective buy interpretability cheaply? Call it **GeoOODSelect**. See §6.

---

## 4. Data

### Primary: Folktables / ACS (arXiv:2108.04884)

Person-level American Community Survey microdata, packaged as five prediction
tasks with a clean per-state domain split.

- **Tasks:** `ACSIncome` (binary, income >$50k) as the workhorse;
  `ACSPublicCoverage` and `ACSMobility` as replication.
- **Domains:** 50 states + PR. ~1.6M rows total for ACSIncome; California alone
  is ~195k.
- **Features:** ~10 named columns (age, education, occupation, hours worked,
  marital status, place of birth, …). Every one is human-readable, which is what
  makes stage 2 tractable.
- **The critical property:** the underlying ACS PUMS rows carry `ST` (state) *and*
  `PUMA` — Public Use Microdata Area, a real census geography of ~100k people. So
  a per-example binary selection vector aggregates into a **PUMA-level selection
  rate**, and PUMAs have shapefiles. This is what makes the whole project
  mappable, and it is the reason to prefer Folktables over any other tabular
  benchmark. PUMA is not in the task's feature list, so it has to be joined back
  from the raw frame by index — see §9.2.

### The GIS covariate stack (the "GIS system" half)

Joined to PUMAs, used only in stage 2 — never given to the models being audited.

| Layer | Source | Variables |
|---|---|---|
| Geometry | Census TIGER/Line PUMA shapefiles | area, centroid, adjacency graph |
| Urbanicity | USDA ERS Rural–Urban Continuum, NCHS Urban–Rural | ordinal class |
| Land cover | NLCD | % developed / cropland / forest / water per PUMA |
| Socioeconomic | ACS PUMA aggregates | median household income, % BA+, % foreign born, industry mix |
| Density | TIGER + ACS | population per sq km |
| Region | Census | region / division |

### Control: TableShift (arXiv:2312.07577)

Non-geographic tabular distribution shifts. Needed to answer the obvious
reviewer question: is anything you find specific to *geography*, or is it just
what happens to tabular data under shift?

---

## 5. Pipeline

### Stage 0 — Build the model population

The part most likely to be underestimated. OODSelect needs a large, *diverse*
population of models (the paper used up to 4200) because the whole method is a
correlation across models. Tabular has no pretrained-backbone axis, so diversity
has to come from elsewhere:

- **Algorithm family** — logistic regression, random forest, XGBoost, LightGBM,
  MLPs of varying depth/width, kNN, and a tabular-foundation-model entry
  (cf. 2607.26000). This is the main diversity axis and the substitute for
  architecture diversity in vision.
- **Hyperparameters** — depth, learning rate, regularization, width.
- **Feature subsets** — random subsets of the ~10 columns.
- **Training subsample fraction** and **seed**.

Target N ≈ 2000–4000. Use the paper's own stopping rule: keep adding models until
the ID/OOD correlation estimate moves by <1%. Split the *model* population into
disjoint train/val/test sets, as the paper does, so reported correlations are on
held-out models.

**Cost:** XGBoost on 200k×10 tabular is seconds on CPU. 4000 models is roughly
3–11 CPU-hours and embarrassingly parallel. This runs overnight on a laptop. That
cheapness is the whole reason to start tabular.

### Stage 1 — Run OODSelect

Two setups:

- **Pooled (primary).** ID = one anchor state; OOD = all other states pooled.
  This is the setup where the aggregation being criticized *is* geographic
  aggregation, so it's the one that matches the thesis.
- **Pairwise (secondary).** ID = state A, OOD = state B over a grid of pairs.
  Gives per-pair AoTIL strength, which is itself a map worth drawing.

Reimplement the relaxed objective from Algorithm 1:

```
min_s  corr(acc_ID, acc_s^OOD) + λ(S − ||s||₁)²     s ∈ [0,1]^d
```

Adam, cosine annealing on both lr and λ, multiple random restarts, probit-
transformed accuracies. **Validate the reimplementation against their released
subsets on one of their own benchmarks before trusting it on ACS** — they
released both code and subsets, so this is a cheap, high-value sanity check.

Sweep subset size S as a fraction of |OOD|.

**Baselines:**
1. Random selection
2. Most-misclassified (distinguishes AoTIL from plain difficulty)
3. **Whole-state selection** — greedily choose entire states to minimize the
   correlation. This one is specific to your setting and it is the most important
   control in the project. If whole-state selection matches OODSelect, the
   spurious feature is simply state identity. That is a clean, reportable finding,
   not a failure — see §7.

### Stage 2a — Geographic explanation ("where")

Aggregate the selection vector to PUMA-level rates
`r_p = (# selected in p) / (# OOD examples in p)`, then:

- **Map** `r_p` across the country.
- **Global structure:** Moran's I on `r_p` under queen-contiguity or k-NN spatial
  weights.
- **Nulls that matter:** compare against (i) the random and most-misclassified
  baselines, and (ii) a permutation null that *preserves per-state selection
  counts*. Without (ii) you will detect "states differ from each other," which
  you already knew.
- **Local structure:** LISA / Getis-Ord G\* to name specific hot spots.

Headline result for RQ2: are AoTIL subsets spatially clustered **beyond what
state identity explains**?

### Stage 2b — Covariate explanation ("why")

Two models, deliberately both interpretable:

- **PUMA level:** regress `r_p` on the GIS covariate stack. Use a spatial lag or
  spatial error model, or GWR/MGWR if you want the explanation itself to vary
  regionally.
- **Example level:** sparse logistic regression, a shallow decision tree, or an
  EBM predicting membership from the ACS features.

Validate with **spatial** cross-validation, not random CV (2605.13689,
2603.29981) — random CV on spatially autocorrelated data produces confident
nonsense. Read the spatial confounding review (2602.17792) before writing this
section; it is the main threat to any causal-sounding claim here. GeoShapley
(2606.23453) is an optional tool for separating location contribution from
feature contribution.

Report: top covariates, effect signs, and how few of them get you most of the
way. Sparsity is the point — "this subset is rural, low-density, high-agriculture
PUMAs in the Plains" is a result; a dense 40-covariate model is not.

---

## 6. Stretch: GeoOODSelect

Add a spatial-coherence penalty to the objective:

```
min_s  corr(acc_ID, acc_s^OOD) + λ(S − ||s||₁)² + μ · sᵀ L s
```

where `L` is the graph Laplacian of the PUMA adjacency graph (or a total-variation
penalty on PUMA-level rates). Sweeping μ trades correlation strength for
geographic contiguity, producing a **coherence–correlation frontier**: how much
AoTIL do you give up to get sets you can actually name?

This is the methodological novelty and the thing that would turn the project into
a paper. It is explicitly a stretch goal — stages 0–2 are the deliverable.

---

## 7. Risks, and what each one turns into

| Risk | Response |
|---|---|
| AoTIL doesn't appear in tabular GIS at all | Week 1–2 feasibility check. If aggregate correlation is already weak, reframe to *when does AoTIL appear*, with TableShift as contrast. Still a result. |
| OODSelect just recovers state identity | The whole-state baseline turns this into the headline: "geographic aggregation hides failure because geography *is* the spurious feature." Reportable either way. |
| Model population too narrow to estimate correlations | Check the <1% stopping rule in week 3. Widen the algorithm-family axis first. |
| Spatial autocorrelation invalidates naive tests | Permutation nulls preserving state counts; spatial CV throughout; cite the confounding review. |
| ACS carries race, sex, ancestry — findings may implicate them | Frame as model auditing, which is the anchor paper's own framing for its CXR result. Report carefully, do not build predictors of protected attributes, and be explicit about what a correlation does and does not establish. |
| Scope creep into imagery | Stages 0–2 on tabular are the deliverable. Imagery is a future-work sentence, not a week. |

---

## 8. Timeline (13 weeks, 2026-09-14 → mid-December)

| Weeks | Work |
|---|---|
| 1–2 | Read the six core papers. Load Folktables. 200-model pilot on one ID/OOD state pair — does aggregate AoTL even hold here? Go/no-go. |
| 3–4 | Full model population (N≈2000). Reimplement OODSelect; **validate against their released subsets** on one of their benchmarks. |
| 5–6 | Run OODSelect across pooled and pairwise setups. All three baselines including whole-state. |
| 7–8 | Build the GIS covariate stack. PUMA shapefile join. First maps. |
| 9–10 | Stage 2a + 2b. Spatial statistics, interpretable explainers, spatial CV. |
| 11 | GeoOODSelect if on schedule; otherwise consolidate and strengthen stage 2. |
| 12–13 | Writeup and presentation. |

Adjust against the practicum's own milestone dates.

---

## 9. Immediate next steps

1. Clone https://github.com/olawalesalaudeen/OODSELECT — confirm the released
   subsets are there and the objective is implemented as described.
2. `pip install folktables`; pull ACSIncome for ~5 states. **Checked already:**
   `ACSIncome`'s feature list is exactly
   `[AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P]` — PUMA is *not*
   a feature. But `ACSDataSource.get_data()` returns the raw ACS PUMS frame,
   which does carry `PUMA` and `ST`, and `adult_filter` / `df_to_pandas` are
   index-preserving boolean operations. So recover it with
   `raw.loc[features.index, ['ST','PUMA']]`. Confirm the index alignment on one
   state before building anything on top of it — the entire mapping story rests
   on this join. (Note: `RELP` was replaced by `RELSHIPP` in ACS 2019+; pin your
   survey year.)
3. Download TIGER/Line PUMA shapefiles; verify the PUMA↔geometry join for one state.
4. Run the 200-model pilot and plot probit ID accuracy vs. OOD accuracy. Is the
   line there?
5. Email Haoran Zhang (co-author on both OODSelect and GeoSpOT) once the pilot
   plot exists — a concrete figure makes that a much better email than a pitch.

---

## 10. Title options

- *Accuracy on the Inverse Line Has a Map: Explaining OOD Failure Subsets with Geographic Covariates*
- *Where Aggregation Hides Failure: Mapping OOD Failure Sets in Cross-State Prediction*
- *The Hidden Geography of Out-of-Distribution Failure*
- *GeoOODSelect: Spatially Coherent Discovery of Out-of-Distribution Failure Sets*
